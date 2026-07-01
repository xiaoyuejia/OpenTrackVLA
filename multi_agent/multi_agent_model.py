#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""早期独立双 Agent OpenTrackVLA MLP 模型定义。

本文件只负责模型结构，不包含数据读取和训练循环。整体思路是：
1. 两个 Agent 分别提供历史粗粒度视觉 token、当前细粒度视觉 token 和 bbox。
2. 视觉 token 经 CrossModalityProjector 投影到 LLM hidden 维度。
3. MultiAgentTVIEmbedder 为 token 加入时间、视角、类别和 Agent 身份嵌入。
4. LLM 通过 self-attention 融合双 Agent 视觉、bbox 和文本指令。
5. 两个 ACT 查询 token 分别回归两个 Agent 的未来路点，GND 查询 token 输出 grounding 结果。

核心类：
- ``MultiAgentOpenTrackVLA``：双 Agent前向主干，返回路点、bbox、可见性和 token logits。
- ``MultiAgentTVIEmbedder``：注入时间、视角、类别和 Agent 身份嵌入。
- ``PlannerHead3L`` / ``GroundingHead``：路点回归与目标定位辅助头。

版本边界：
该文件属于 ``multi_agent/`` 早期独立实验实现。当前主流程优先使用根目录
``model.py::MultiAgentOpenTrackVLA``；扩散版使用
``model_unrealzoo_anchor_diffusion.py``。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# token 类别编号。kind_emb 用这些编号区分“历史视觉/当前视觉/bbox/动作查询/grounding 查询”。
KIND_HISTORY = 0  # 历史帧粗粒度视觉 token
KIND_CURRENT = 1  # 当前帧细粒度视觉 token
KIND_BBOX = 2     # 目标 bbox 条件 token
KIND_ACT = 3      # 动作规划查询 token
KIND_GND = 4      # grounding 查询 token


# ----------------------- 双 Agent模型配置 -----------------------

@dataclass
class MultiAgentModelConfig:
    """双 Agent 模型配置。

    其中 num_agents 固定为 2，对应无人机和机器狗；num_kinds=5 对应上面的 KIND_*。
    alpha_xy 只缩放 x/y 两个空间维度，theta/yaw 默认不缩放。
    """

    llm_name: str = "Qwen/Qwen3-0.6B"
    freeze_llm: bool = True
    n_waypoints: int = 8
    action_dims: int = 3
    max_time: int = 4096
    max_views: int = 2
    num_agents: int = 2
    num_kinds: int = 5
    use_angle_tvi: bool = False
    insert_time_tokens: bool = True
    use_tanh_actions: bool = True
    alpha_xy: Optional[float] = 2.0
    bbox_delta_scale: float = 0.25
    return_token_logits: bool = True
    text_max_length: int = 128


# ----------------------- 视觉投影与路点规划组件 -----------------------

class CrossModalityProjector(nn.Module):
    """把视觉缓存 token 从 vision_feat_dim 投影到 LLM hidden size。

    输入形状:  (B, N, C_vision)
    输出形状:  (B, N, D_llm)
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PlannerHead3L(nn.Module):
    """三层 MLP 规划头。

    输入是某个 ACT 查询 token 在 LLM 输出中的 hidden state，输出固定数量路点。
    """

    def __init__(self, d_model: int, n_waypoints: int, action_dims: int, use_tanh: bool = True):
        super().__init__()
        hid = d_model * 2
        self.n_waypoints = n_waypoints
        self.action_dims = action_dims
        self.use_tanh = use_tanh
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hid),
            nn.GELU(),
            nn.Linear(hid, hid),
            nn.GELU(),
            nn.Linear(hid, n_waypoints * action_dims),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        y = self.mlp(h)
        if self.use_tanh:
            y = torch.tanh(y)
        return y.view(-1, self.n_waypoints, self.action_dims)


# ----------------------- 双 Agent时空视角嵌入 -----------------------

class MultiAgentTVIEmbedder(nn.Module):
    """多 Agent TVI 嵌入器。

    TVI = Temporal / View / Identity 信息注入。这里的 token 表示由四类信息叠加：
    visual_or_bbox_or_query_token + time_emb + view_emb + kind_emb + agent_emb。
    """

    def __init__(
        self,
        d_model: int,
        max_time: int = 4096,
        max_views: int = 2,
        num_agents: int = 2,
        num_kinds: int = 5,
    ):
        super().__init__()
        self.time_emb = nn.Embedding(max_time, d_model)
        self.view_emb = nn.Embedding(max_views, d_model)
        self.kind_emb = nn.Embedding(num_kinds, d_model)
        self.agent_emb = nn.Embedding(num_agents, d_model)
        self.angle_proj = nn.Linear(2, d_model)
        self.bbox_proj = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    #作用：根据输入的 kind_id、agent_id、view_id 生成对应的嵌入向量。kind_id 表示 token 的类别（例如历史视觉、当前视觉、bbox、动作查询、grounding 查询），agent_id 表示是哪个 Agent（例如无人机或机器狗），view_id 表示视角编号（通常与 agent_id 相同）。返回三个嵌入向量，分别是 kind_emb、agent_emb 和 view_emb。
    def _ids(
        self,
        kind_id: int,
        agent_id: int,
        view_id: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        kind = torch.tensor(kind_id, dtype=torch.long, device=device)
        agent = torch.tensor(agent_id, dtype=torch.long, device=device)
        view = torch.tensor(view_id, dtype=torch.long, device=device)
        return kind, agent, view

    #作用：生成非内容类嵌入，包含 token 类型（kind_id）、Agent 身份（agent_id）和视角编号（view_id）。首先调用 _ids 方法获取对应的 kind、agent 和 view 张量，然后通过 kind_emb、agent_emb 和 view_emb 分别获取对应的嵌入向量，最后将它们相加得到最终的类型嵌入。
    def type_embedding(self, kind_id: int, agent_id: int, view_id: int, device: torch.device) -> torch.Tensor:
        """生成非内容类嵌入：token 类型 + Agent 身份 + 视角编号。"""
        kind, agent, view = self._ids(kind_id, agent_id, view_id, device)
        return self.kind_emb(kind) + self.agent_emb(agent) + self.view_emb(view)

    #作用：给视觉 token 加上时间/类型/Agent/View 嵌入。输入 tokens 是视觉 token 的特征表示，t_idx 是对应的时间索引，kind_id、agent_id 和 view_id 分别表示 token 的类别、Agent 身份和视角编号。首先将 t_idx 转换为 long 类型并限制在 time_emb 的范围内，然后调用 type_embedding 方法获取类型嵌入，最后将原始 token、时间嵌入和类型嵌入相加得到最终的嵌入表示。
    def add_visual_tvi(
        self,
        tokens: torch.Tensor,
        t_idx: torch.Tensor,
        kind_id: int,
        agent_id: int,
        view_id: int,
    ) -> torch.Tensor:
        """直接给视觉 token 加上时间/类型/Agent/View 嵌入。

        注意这里是“加到每个视觉 token 上”；后续 _interleave_markers 还可以额外插入显式时间 marker。
        """
        device = tokens.device
        # 不使用 clamp_ 原地操作：agent1/agent2 的 t_idx 通常是同一个 (B,2,N)
        # 张量切出来的 view，原地修改其中一个 view 会破坏另一个 view 的 autograd 版本号。
        t_idx = t_idx.to(device=device, dtype=torch.long).clamp(0, self.time_emb.num_embeddings - 1)
        type_emb = self.type_embedding(kind_id, agent_id, view_id, device)
        return tokens + self.time_emb(t_idx) + type_emb.view(1, 1, -1)

    #作用：生成一个显式插入到序列里的时间 marker token。
    def make_marker(
        self,
        t_scalar: int,
        kind_id: int,
        agent_id: int,
        view_id: int,
        device: torch.device,
    ) -> torch.Tensor:
        """生成一个显式插入到序列里的时间 marker token。"""
        t = int(max(0, min(t_scalar, self.time_emb.num_embeddings - 1)))
        return self.time_emb.weight[t] + self.type_embedding(kind_id, agent_id, view_id, device)

    def make_angle_marker(
        self,
        theta: float,
        kind_id: int,
        agent_id: int,
        view_id: int,
        device: torch.device,
    ) -> torch.Tensor:
        """可选航向角 marker，用 [sin(theta), cos(theta)] 避免角度周期不连续。"""
        theta = (theta + math.pi) % (2 * math.pi) - math.pi
        sincos = torch.tensor([math.sin(theta), math.cos(theta)], dtype=self.angle_proj.weight.dtype, device=device)
        angle = F.linear(sincos, self.angle_proj.weight, self.angle_proj.bias)
        return angle + self.type_embedding(kind_id, agent_id, view_id, device)

    def make_bbox_token(self, bbox: torch.Tensor, agent_id: int, view_id: int) -> torch.Tensor:
        """把归一化 bbox(cx, cy, w, h) 编码成一个条件 token。"""
        if bbox.dim() == 1:
            bbox = bbox.unsqueeze(0)
        bbox = bbox.to(dtype=self.bbox_proj[1].weight.dtype)
        emb = self.bbox_proj(bbox)
        if emb.dim() == 2:
            emb = emb.unsqueeze(1)
        type_emb = self.type_embedding(KIND_BBOX, agent_id, view_id, emb.device)
        return emb + type_emb.view(1, 1, -1)

    def make_query_token(self, base: torch.Tensor, kind_id: int, agent_id: int, view_id: int) -> torch.Tensor:
        """给 ACT/GND 可学习查询 token 加上类型、Agent 和视角身份。"""
        type_emb = self.type_embedding(kind_id, agent_id, view_id, base.device)
        return base + type_emb.view(1, 1, -1)


# ----------------------- 目标定位与可见性预测 -----------------------

class GroundingHead(nn.Module):
    """Grounding 输出头。

    从 GND token 的 hidden state 预测：
    - refined_bbox: 对输入 bbox 做小范围修正
    - visible_score: 每个 Agent 视角下目标是否可见
    - token_logits: 可选的视觉 token 级 grounding 分数
    """

    def __init__(self, d_model: int, num_agents: int = 2, bbox_delta_scale: float = 0.25):
        super().__init__()
        self.num_agents = num_agents
        self.bbox_delta_scale = bbox_delta_scale
        hid = d_model * 2
        self.bbox_delta = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hid),
            nn.GELU(),
            nn.Linear(hid, num_agents * 4),
        )
        self.visible = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hid),
            nn.GELU(),
            nn.Linear(hid, num_agents),
        )
        self.token_query = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_agents * d_model),
        )

    def forward(
        self,
        h_gnd: torch.Tensor,
        bbox_feat: Optional[torch.Tensor] = None,
        visual_tokens: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        B = h_gnd.size(0)
        # bbox_delta 是相对输入 bbox 的修正量，经过 tanh 和 bbox_delta_scale 限制幅度。
        delta = torch.tanh(self.bbox_delta(h_gnd)).view(B, self.num_agents, 4) * self.bbox_delta_scale
        if bbox_feat is None:
            refined_bbox = torch.sigmoid(delta)
        else:
            refined_bbox = (bbox_feat.to(device=h_gnd.device, dtype=delta.dtype) + delta).clamp(0.0, 1.0)

        # 训练时用 logits 配合 binary_cross_entropy_with_logits，避免 AMP autocast 下 BCE 报错。
        # 推理/日志仍保留 sigmoid 后的 visible_score，便于直接解释为概率。
        visible_logits = self.visible(h_gnd)
        visible_score = torch.sigmoid(visible_logits)
        token_logits = None
        if visual_tokens is not None:
            # 用 GND token 生成每个 Agent 的 query，再和视觉 token 做点积得到 token-level 分数。
            queries = self.token_query(h_gnd).view(B, self.num_agents, -1)
            token_logits = torch.einsum("bad,band->ban", queries.float(), visual_tokens.float())
            token_logits = token_logits / math.sqrt(max(1, queries.size(-1)))
        return {
            "refined_bbox": refined_bbox,
            "visible_logits": visible_logits,
            "visible_score": visible_score,
            "token_logits": token_logits,
        }


# ----------------------- 双 Agent模型主干 -----------------------

class MultiAgentOpenTrackVLA(nn.Module):
    """Two-agent OpenTrackVLA variant with bbox conditioning and grounding output.

    Forward accepts either split arguments:
      agent1_coarse_tokens, agent1_coarse_tidx, ...
      agent2_coarse_tokens, agent2_coarse_tidx, ...

    or stacked tensors:
      coarse_tokens=(B, 2, Nc, C), coarse_tidx=(B, 2, Nc),
      fine_tokens=(B, 2, Nf, C), fine_tidx=(B, 2, Nf),
      bbox_feat=(B, 2, 4)
    """

    def __init__(self, cfg: MultiAgentModelConfig, vision_feat_dim: int):
        super().__init__()
        if cfg.num_agents != 2:
            raise ValueError("This implementation currently expects cfg.num_agents == 2.")
        self.cfg = cfg

        rank = int(os.environ.get("RANK", "0"))
        t0 = time.time()
        print(f"[MODEL][rank {rank}] loading LLM weights: {cfg.llm_name}", flush=True)
        try:
            from modelscope import AutoModel as MSAutoModel
            from modelscope import AutoTokenizer as MSAutoTokenizer

            self.llm = MSAutoModel.from_pretrained(
                cfg.llm_name,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
            )
            self.tokenizer = MSAutoTokenizer.from_pretrained(cfg.llm_name)
        except Exception:
            load_kwargs = {}
            if torch.cuda.is_available():
                # transformers 新版中 torch_dtype 已弃用，改用 dtype。
                load_kwargs["dtype"] = torch.bfloat16
            self.llm = AutoModel.from_pretrained(
                cfg.llm_name,
                **load_kwargs,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.llm_name)
        print(f"[MODEL][rank {rank}] LLM/tokenizer loaded in {time.time() - t0:.1f}s", flush=True)

        self.llm.requires_grad_(not cfg.freeze_llm)
        self.llm_dtype = next(self.llm.parameters()).dtype
        self.D = int(self.llm.config.hidden_size)

        self.proj = CrossModalityProjector(vision_feat_dim, self.D)
        self.tvi = MultiAgentTVIEmbedder(
            self.D,
            max_time=cfg.max_time,
            max_views=cfg.max_views,
            num_agents=cfg.num_agents,
            num_kinds=cfg.num_kinds,
        )

        self.act_token_1 = nn.Parameter(torch.zeros(1, 1, self.D))
        self.act_token_2 = nn.Parameter(torch.zeros(1, 1, self.D))
        self.gnd_token = nn.Parameter(torch.zeros(1, 1, self.D))
        nn.init.normal_(self.act_token_1, std=0.02)
        nn.init.normal_(self.act_token_2, std=0.02)
        nn.init.normal_(self.gnd_token, std=0.02)

        self.planner_agent1 = PlannerHead3L(self.D, cfg.n_waypoints, cfg.action_dims, cfg.use_tanh_actions)
        self.planner_agent2 = PlannerHead3L(self.D, cfg.n_waypoints, cfg.action_dims, cfg.use_tanh_actions)
        self.grounding_head = GroundingHead(self.D, cfg.num_agents, cfg.bbox_delta_scale)

        if not cfg.use_angle_tvi:
            for p in self.tvi.angle_proj.parameters():
                p.requires_grad = False

        if cfg.alpha_xy is not None:
            alpha = torch.ones(1, 1, cfg.action_dims, dtype=torch.float32)
            if cfg.action_dims >= 2:
                alpha[..., 0:2] = float(cfg.alpha_xy)
        else:
            alpha = torch.ones(1, 1, cfg.action_dims, dtype=torch.float32)
        self.register_buffer("alpha_task", alpha)

    def _embed_text(self, instructions: List[str], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        tok = self.tokenizer(
            instructions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.text_max_length,
        )
        tok = {k: v.to(device) for k, v in tok.items()}
        emb = self.llm.get_input_embeddings()(tok["input_ids"])
        return emb, tok["attention_mask"]

    def _normalize_bbox(self, bbox: Optional[torch.Tensor], B: int, device: torch.device) -> torch.Tensor:
        """统一 bbox 输入形状为 (B, 2, 4)，并保证数值在 [0, 1]。

        训练数据预处理阶段已经把 UnrealZoo 的 pixel-space xywh 转成了 cxcywh_norm。
        如果这里发现 bbox 大于 1.5，通常说明还没归一化，直接报错避免静默训练坏数据。
        """
        if bbox is None:
            return torch.zeros(B, self.cfg.num_agents, 4, dtype=torch.float32, device=device)
        bbox = bbox.to(device=device, dtype=torch.float32)
        if bbox.dim() == 2:
            bbox = bbox.unsqueeze(1).expand(-1, self.cfg.num_agents, -1)
        if bbox.size(1) != self.cfg.num_agents or bbox.size(-1) != 4:
            raise ValueError(f"bbox_feat must have shape (B, 2, 4) or (B, 4), got {tuple(bbox.shape)}")
        bbox = bbox.clone()
        if bbox.detach().amax() > 1.5:
            raise ValueError("bbox_feat should be normalized to [0, 1] before passing to the model.")
        return bbox.clamp(0.0, 1.0)

    def _interleave_markers(
        self,
        tokens: torch.Tensor,
        t_idx: torch.Tensor,
        kind_id: int,
        agent_id: int,
        view_id: int,
        yaw_per_frame: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """按帧分组，在每个帧块前插入显式时间 marker。

        输入视觉 token 已经加过 time_emb，这里的 marker 是额外的“边界/时间提示 token”，
        作用类似原始 model.py 里的 _interleave_tvi。
        """
        if not self.cfg.insert_time_tokens:
            return tokens

        B, N, _ = tokens.shape
        out_list = []
        for b in range(B):
            tb = t_idx[b]
            xb = tokens[b]
            items = []
            i = 0
            fcount = 0
            while i < N:
                tcur = int(tb[i].item())
                j = i + 1
                while j < N and int(tb[j].item()) == tcur:
                    j += 1
                marker = self.tvi.make_marker(tcur, kind_id, agent_id, view_id, xb.device).unsqueeze(0)
                items.append(marker)
                if self.cfg.use_angle_tvi:
                    theta = 0.0
                    if yaw_per_frame is not None and fcount < yaw_per_frame.size(1):
                        theta = float(yaw_per_frame[b, fcount].item())
                    angle = self.tvi.make_angle_marker(theta, kind_id, agent_id, view_id, xb.device).unsqueeze(0)
                    items.append(angle)
                items.append(xb[i:j])
                i = j
                fcount += 1
            out_list.append(torch.cat(items, dim=0))
        return torch.stack(out_list, dim=0)

    def _encode_agent_stream(
        self,
        coarse_tokens: torch.Tensor,
        coarse_tidx: torch.Tensor,
        fine_tokens: torch.Tensor,
        fine_tidx: torch.Tensor,
        bbox_feat: torch.Tensor,
        agent_id: int,
        yaw_hist: Optional[torch.Tensor] = None,
        yaw_curr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """编码单个 Agent 的视觉流。

        返回:
        - seq: 用于送入 LLM 的序列片段，包含历史视觉、当前视觉和 bbox token。
        - visual_for_grounding: 不含显式 marker 的视觉 token，用于可选 token-level grounding。
        """
        view_id = agent_id
        vis_c = self.proj(coarse_tokens)
        vis_f = self.proj(fine_tokens)

        vis_c = self.tvi.add_visual_tvi(vis_c, coarse_tidx, KIND_HISTORY, agent_id, view_id)
        vis_f = self.tvi.add_visual_tvi(vis_f, fine_tidx, KIND_CURRENT, agent_id, view_id)
        visual_for_grounding = torch.cat([vis_c, vis_f], dim=1)

        seq_c = self._interleave_markers(vis_c, coarse_tidx, KIND_HISTORY, agent_id, view_id, yaw_hist)
        seq_f = self._interleave_markers(vis_f, fine_tidx, KIND_CURRENT, agent_id, view_id, yaw_curr)
        bbox_tok = self.tvi.make_bbox_token(bbox_feat, agent_id, view_id)
        return torch.cat([seq_c, seq_f, bbox_tok], dim=1), visual_for_grounding

    def _split_stacked_inputs(
        self,
        coarse_tokens: Optional[torch.Tensor],
        coarse_tidx: Optional[torch.Tensor],
        fine_tokens: Optional[torch.Tensor],
        fine_tidx: Optional[torch.Tensor],
        agent1_coarse_tokens: Optional[torch.Tensor],
        agent1_coarse_tidx: Optional[torch.Tensor],
        agent1_fine_tokens: Optional[torch.Tensor],
        agent1_fine_tidx: Optional[torch.Tensor],
        agent2_coarse_tokens: Optional[torch.Tensor],
        agent2_coarse_tidx: Optional[torch.Tensor],
        agent2_fine_tokens: Optional[torch.Tensor],
        agent2_fine_tidx: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """支持两种 forward 输入形式。

        1. 堆叠形式: coarse_tokens=(B, 2, N, C)
        2. 分开形式: agent1_coarse_tokens / agent2_coarse_tokens
        训练脚本使用的是第一种堆叠形式。
        """
        if coarse_tokens is not None:
            if coarse_tokens.dim() != 4 or fine_tokens is None or coarse_tidx is None or fine_tidx is None:
                raise ValueError("Stacked inputs require coarse_tokens/fine_tokens=(B, 2, N, C) and *_tidx=(B, 2, N).")
            return (
                coarse_tokens[:, 0],
                coarse_tidx[:, 0],
                fine_tokens[:, 0],
                fine_tidx[:, 0],
                coarse_tokens[:, 1],
                coarse_tidx[:, 1],
                fine_tokens[:, 1],
                fine_tidx[:, 1],
            )

        required = [
            agent1_coarse_tokens,
            agent1_coarse_tidx,
            agent1_fine_tokens,
            agent1_fine_tidx,
            agent2_coarse_tokens,
            agent2_coarse_tidx,
            agent2_fine_tokens,
            agent2_fine_tidx,
        ]
        if any(x is None for x in required):
            raise ValueError("Pass either stacked inputs or all agent1_/agent2_ split inputs.")
        return (
            agent1_coarse_tokens,
            agent1_coarse_tidx,
            agent1_fine_tokens,
            agent1_fine_tidx,
            agent2_coarse_tokens,
            agent2_coarse_tidx,
            agent2_fine_tokens,
            agent2_fine_tidx,
        )

    def forward(
        self,
        coarse_tokens: Optional[torch.Tensor] = None,
        coarse_tidx: Optional[torch.Tensor] = None,
        fine_tokens: Optional[torch.Tensor] = None,
        fine_tidx: Optional[torch.Tensor] = None,
        instructions: Optional[List[str]] = None,
        bbox_feat: Optional[torch.Tensor] = None,
        yaw_hist: Optional[torch.Tensor] = None,
        yaw_curr: Optional[torch.Tensor] = None,
        agent1_coarse_tokens: Optional[torch.Tensor] = None,
        agent1_coarse_tidx: Optional[torch.Tensor] = None,
        agent1_fine_tokens: Optional[torch.Tensor] = None,
        agent1_fine_tidx: Optional[torch.Tensor] = None,
        agent1_bbox_feat: Optional[torch.Tensor] = None,
        agent1_yaw_hist: Optional[torch.Tensor] = None,
        agent1_yaw_curr: Optional[torch.Tensor] = None,
        agent2_coarse_tokens: Optional[torch.Tensor] = None,
        agent2_coarse_tidx: Optional[torch.Tensor] = None,
        agent2_fine_tokens: Optional[torch.Tensor] = None,
        agent2_fine_tidx: Optional[torch.Tensor] = None,
        agent2_bbox_feat: Optional[torch.Tensor] = None,
        agent2_yaw_hist: Optional[torch.Tensor] = None,
        agent2_yaw_curr: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, Dict[str, Optional[torch.Tensor]]]]:
        """前向传播。

        典型输入 shape:
        - coarse_tokens: (B, 2, 124, C)
        - fine_tokens:   (B, 2, 64, C)
        - bbox_feat:     (B, 2, 4)

        默认返回 dict，其中 waypoints shape 为 (B, 2, n_waypoints, 3)。
        """
        device = next(self.parameters()).device
        (
            a1_c,
            a1_ct,
            a1_f,
            a1_ft,
            a2_c,
            a2_ct,
            a2_f,
            a2_ft,
        ) = self._split_stacked_inputs(
            coarse_tokens,
            coarse_tidx,
            fine_tokens,
            fine_tidx,
            agent1_coarse_tokens,
            agent1_coarse_tidx,
            agent1_fine_tokens,
            agent1_fine_tidx,
            agent2_coarse_tokens,
            agent2_coarse_tidx,
            agent2_fine_tokens,
            agent2_fine_tidx,
        )

        a1_c = a1_c.to(device)
        a1_ct = a1_ct.to(device)
        a1_f = a1_f.to(device)
        a1_ft = a1_ft.to(device)
        a2_c = a2_c.to(device)
        a2_ct = a2_ct.to(device)
        a2_f = a2_f.to(device)
        a2_ft = a2_ft.to(device)
        B = a1_c.size(0)

        if instructions is None:
            instructions = ["follow the person"] * B

        if bbox_feat is not None:
            bbox = self._normalize_bbox(bbox_feat, B, device)
        else:
            b1 = agent1_bbox_feat
            b2 = agent2_bbox_feat
            if b1 is None:
                b1 = torch.zeros(B, 4, dtype=torch.float32, device=device)
            if b2 is None:
                b2 = torch.zeros(B, 4, dtype=torch.float32, device=device)
            bbox = torch.stack([b1.to(device), b2.to(device)], dim=1).float().clamp(0.0, 1.0)

        if yaw_hist is not None and yaw_hist.dim() == 3:
            agent1_yaw_hist = yaw_hist[:, 0]
            agent2_yaw_hist = yaw_hist[:, 1]
        if yaw_curr is not None and yaw_curr.dim() == 3:
            agent1_yaw_curr = yaw_curr[:, 0]
            agent2_yaw_curr = yaw_curr[:, 1]

        a1_seq, a1_visual = self._encode_agent_stream(
            a1_c,
            a1_ct,
            a1_f,
            a1_ft,
            bbox[:, 0],
            agent_id=0,
            yaw_hist=agent1_yaw_hist,
            yaw_curr=agent1_yaw_curr,
        )
        a2_seq, a2_visual = self._encode_agent_stream(
            a2_c,
            a2_ct,
            a2_f,
            a2_ft,
            bbox[:, 1],
            agent_id=1,
            yaw_hist=agent2_yaw_hist,
            yaw_curr=agent2_yaw_curr,
        )

        txt_emb, txt_mask = self._embed_text(instructions, device)
        # 三个查询 token：
        # ACT_1 用于 Agent-1 规划，ACT_2 用于 Agent-2 规划，GND 用于 bbox/visibility grounding。
        act1 = self.tvi.make_query_token(self.act_token_1.expand(B, 1, -1), KIND_ACT, agent_id=0, view_id=0)
        act2 = self.tvi.make_query_token(self.act_token_2.expand(B, 1, -1), KIND_ACT, agent_id=1, view_id=1)
        gnd = self.tvi.make_query_token(self.gnd_token.expand(B, 1, -1), KIND_GND, agent_id=0, view_id=0)

        # LLM 输入序列：
        # [文本] + [A1视觉+BBOX] + [ACT1] + [A2视觉+BBOX] + [ACT2] + [GND]
        pieces = [txt_emb, a1_seq, act1, a2_seq, act2, gnd]
        lengths = [p.size(1) for p in pieces]
        # 由于 ACT/GND 不是都在最后，必须显式记录查询 token 的位置。
        act1_pos = lengths[0] + lengths[1]
        act2_pos = lengths[0] + lengths[1] + lengths[2] + lengths[3]
        gnd_pos = sum(lengths) - 1

        seq = torch.cat(pieces, dim=1).to(dtype=self.llm_dtype)
        attn = torch.cat(
            [
                txt_mask,
                torch.ones(B, sum(lengths[1:]), dtype=torch.long, device=device),
            ],
            dim=1,
        )

        out = self.llm(inputs_embeds=seq, attention_mask=attn, output_hidden_states=True, use_cache=False)
        hidden = out.last_hidden_state.float()
        # 取三个查询 token 对应的 LLM 输出 hidden state，送到不同输出头。
        h_act1 = hidden[:, act1_pos, :]
        h_act2 = hidden[:, act2_pos, :]
        h_gnd = hidden[:, gnd_pos, :]

        # 两个 Agent 使用独立规划头，允许学习不同运动学/控制特性。
        agent1_waypoints = self.planner_agent1(h_act1) * self.alpha_task
        agent2_waypoints = self.planner_agent2(h_act2) * self.alpha_task

        visual_tokens = None
        if self.cfg.return_token_logits and a1_visual.size(1) == a2_visual.size(1):
            visual_tokens = torch.stack([a1_visual, a2_visual], dim=1)
        grounding = self.grounding_head(h_gnd, bbox_feat=bbox, visual_tokens=visual_tokens)

        if not return_dict:
            return agent1_waypoints, agent2_waypoints, grounding
        return {
            "agent1_waypoints": agent1_waypoints,
            "agent2_waypoints": agent2_waypoints,
            "waypoints": torch.stack([agent1_waypoints, agent2_waypoints], dim=1),
            "refined_bbox": grounding["refined_bbox"],
            "visible_logits": grounding["visible_logits"],
            "visible_score": grounding["visible_score"],
            "token_logits": grounding["token_logits"],
        }


OpenTrackVLAMultiAgent = MultiAgentOpenTrackVLA
