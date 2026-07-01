#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OpenTrackVLA 原始单 Agent与 UnrealZoo 双 Agent MLP 模型定义。

整体功能：
- 将历史粗粒度视觉 token、当前细粒度视觉 token、目标 bbox 和文本指令送入 LLM。
- 从 ACT 查询 token 回归局部未来路点，并从 GND 查询 token 预测目标框与可见性。
- 同时保留 Habitat 单 Agent模型和 UnrealZoo 双 Agent MLP 规划模型。

核心类：
- ``OpenTrackVLA``：Habitat 原始单 Agent模型，输出 ``waypoints/refined_bbox/visible_logits``。
- ``MultiAgentOpenTrackVLA``：双 Agent共享 LLM、独立 MLP 规划头，输出两个 Agent 的路点。
- ``MultiAgentSeparateOpenTrackVLA``：双 Agent独立 LLM 上下文对照，不拼接两个 Agent 的视觉 pieces。
- ``TVIEmbedder`` / ``MultiAgentTVIEmbedder``：注入时间、视角、token 类别和 Agent 身份。
- ``PlannerHead3L`` / ``GroundingHead``：分别执行路点回归和目标检测/可见性预测。
- ``JsonTrackingDataset``：单 Agent模型独立推理时使用的数据读取器。

关键函数：
- ``integrate_actions_to_waypoints``：将速度动作积分为局部轨迹标签。
- ``collate_batch``：组装单 Agent batch。
- ``_run_inference`` / ``parse_args``：本文件独立推理入口。

版本边界：
- 双 Agent普通 MLP 训练使用 ``train.py --multi_agent``。
- Anchor Diffusion 版本位于 ``model_unrealzoo_anchor_diffusion.py``。
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, Union
import os, json, math, argparse, time
from pathlib import Path
from contextlib import nullcontext
from PIL import Image, ImageDraw

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel
from tools.cache_gridpool import VisionFeatureCacher, VisionCacheConfig, grid_pool_tokens, adapt_siglip_grid


# ----------------------- 通用工具与数据加载 -----------------------

# Silence tokenizers fork warnings in dataloader workers
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


# 双 Agent 模型里的 token 类型编号。
# 数据流含义：
# - HISTORY: 历史帧粗粒度视觉 token，用于理解过去运动趋势。
# - CURRENT: 当前帧细粒度视觉 token，用于定位当前目标。
# - BBOX: 当前目标框 token，把检测/跟踪先验注入 LLM。
# - ACT: 每个 Agent 的动作查询 token，LLM 在这个位置输出规划隐藏状态。
# - GND: grounding 查询 token，LLM 在这个位置输出 bbox/visibility 辅助隐藏状态。
KIND_HISTORY = 0
KIND_CURRENT = 1
KIND_BBOX = 2
KIND_ACT = 3
KIND_GND = 4


def load_tokens_file(path: str) -> torch.Tensor:
    try:
        obj = torch.load(path, map_location='cpu')
    except Exception:
        try:
            obj = torch.load(path, map_location='cpu', weights_only=False)
        except Exception as e:
            raise e
    if isinstance(obj, torch.Tensor):
        return obj.float()
    if isinstance(obj, dict):
        for k in ("V", "Vfine", "Vcoarse", "tokens", "feat", "features"):
            if k in obj and isinstance(obj[k], torch.Tensor):
                t = obj[k]
                if t.dim() == 3 and t.size(0) == 1:
                    t = t[0]
                return t.float()
    raise ValueError(f"Unrecognized token file: {path}")


def integrate_actions_to_waypoints(actions: np.ndarray, n_waypoints: int, dt: float = 0.2) -> np.ndarray:
    a = np.asarray(actions, dtype=np.float32)
    if a.ndim == 1: a = a[None, :]
    T, D = a.shape
    vx = a[:, 0].astype(np.float32)
    vy = a[:, 1].astype(np.float32) if D > 1 else np.zeros_like(vx)
    wz = a[:, 2].astype(np.float32) if D > 2 else np.zeros_like(vx)

    x = np.zeros(T, dtype=np.float32)
    y = np.zeros(T, dtype=np.float32)
    th = np.zeros(T, dtype=np.float32)

    for t in range(1, T):
        th[t] = th[t-1] + wz[t-1] * dt
        c, s = np.cos(th[t-1]), np.sin(th[t-1])
        x[t] = x[t-1] + (c * vx[t-1] - s * vy[t-1]) * dt
        y[t] = y[t-1] + (s * vx[t-1] + c * vy[t-1]) * dt

    traj = np.stack([x, y, th], axis=-1)
    if n_waypoints <= 1: return traj[-1:]
    idx = np.linspace(0, T-1, n_waypoints).round().astype(int)
    return traj[idx]


# Flexible loader for JSON/JSONL datasets and directories

def _read_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    with open(file_path, 'r') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            examples.append(json.loads(s))
    return examples


def load_examples_from_path(train_path: str) -> List[Dict[str, Any]]:
    p = Path(train_path)
    if p.is_dir():
        jsonl_files = sorted(p.rglob('*.jsonl'))
        if len(jsonl_files) == 0:
            raise FileNotFoundError(f"No .jsonl files found under directory: {train_path}")
        all_items: List[Dict[str, Any]] = []
        for fp in jsonl_files:
            all_items.extend(_read_jsonl_file(str(fp)))
        return all_items
    if p.is_file():
        if p.suffix.lower() == '.jsonl':
            return _read_jsonl_file(str(p))
        if p.suffix.lower() == '.json':
            with open(p, 'r') as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"JSON file must contain a list at top-level: {train_path}")
            return data
        raise ValueError(f"Unsupported file type: {train_path}")
    raise FileNotFoundError(f"Path does not exist: {train_path}")


# ----------------------- 单 Agent基础模型组件 -----------------------

class TVIEmbedder(nn.Module):
    def __init__(self, d_model: int, max_time: int = 4096, max_views: int = 1):
        super().__init__()
        self.time_emb   = nn.Embedding(max_time, d_model)
        self.view_emb   = nn.Embedding(max_views, d_model)
        self.kind_emb   = nn.Embedding(2, d_model)
        self.angle_proj = nn.Linear(2, d_model)
        self.bbox_proj  = nn.Linear(4, d_model)

    def make_time_token(self, t_scalar: int, kind_id: int, view_id: int = 0,
                        device: Optional[torch.device] = None) -> torch.Tensor:
        tok = self.time_emb.weight[t_scalar] + self.view_emb.weight[view_id] + self.kind_emb.weight[kind_id]
        return tok.to(device) if device is not None else tok

    def make_angle_token(self, theta: float, kind_id: int, view_id: int = 0,
                         device: Optional[torch.device] = None) -> torch.Tensor:
        theta = (theta + math.pi) % (2*math.pi) - math.pi
        sincos = torch.tensor([math.sin(theta), math.cos(theta)],
                              dtype=self.angle_proj.weight.dtype,
                              device=device)
        ang = F.linear(sincos, self.angle_proj.weight, self.angle_proj.bias)
        tok = ang + self.view_emb.weight[view_id] + self.kind_emb.weight[kind_id]
        return tok

    def make_bbox_token(self, bbox: torch.Tensor, kind_id: int, view_id: int = 1,
                         device: Optional[torch.device] = None) -> torch.Tensor:
        if bbox.dim() == 1:
            bbox = bbox.unsqueeze(0)
        bb = bbox.to(device=device, dtype=self.bbox_proj.weight.dtype)
        emb = F.linear(bb, self.bbox_proj.weight, self.bbox_proj.bias)
        emb = emb + self.view_emb.weight[view_id] + self.kind_emb.weight[kind_id]
        return emb.squeeze(0)

# 功能：跨模态投影器，将视觉token投影到与LLM相同的维度空间
class CrossModalityProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim), nn.GELU(),
            nn.Linear(out_dim, out_dim)
        )
    def forward(self, x): return self.net(x)

# 功能：规划头部，将LLM输出的动作token转换为具体的路径点
class PlannerHead3L(nn.Module):
    def __init__(self, d_model: int, n_waypoints: int, action_dims: int, use_tanh: bool = True):
        super().__init__()
        hid = d_model * 2
        out_dim = n_waypoints * action_dims
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hid), nn.GELU(),
            nn.Linear(hid, hid), nn.GELU(),
            nn.Linear(hid, out_dim)
        )
        self.nw = n_waypoints
        self.ad = action_dims
        self.use_tanh = use_tanh
    def forward(self, act_h: torch.Tensor) -> torch.Tensor:
        y = self.mlp(act_h)
        if self.use_tanh:
            y = torch.tanh(y)
        return y.view(-1, self.nw, self.ad)


# ----------------------- 单 Agent模型配置与主干 -----------------------

@dataclass
class ModelConfig:
    """原始单 Agent OpenTrackVLA 配置。

    数据流相关字段：
    - n_waypoints 控制规划头输出多少个未来路点。
    -  max_time 控制 TVI 时间编码的词表大小，需覆盖 episode 内最大帧数。
    - use_angle_tvi 控制是否把 yaw 信息作为额外 TVI token 注入视觉序列。
    - alpha_xy 用于把模型 normalized XY 输出还原到实际距离尺度，yaw 不缩放。
    """
    llm_name: str = "Qwen/Qwen3-0.6B"
    freeze_llm: bool = True
    n_waypoints: int = 8
    max_time: int = 4096
    beta_nav: float = 10.0
    use_angle_tvi: bool = False
    # Action/target configuration
    use_tanh_actions: bool = True
    alpha_xy: Optional[float] = None


class OpenTrackVLA(nn.Module):
    """原始单 Agent OpenTrackVLA。

    单 Agent 数据流：
    1. Dataset 读取 history 个历史帧 coarse token 和当前帧 fine token。
    2. projector 把视觉 token 从视觉特征维度 C 映射到 LLM hidden size D。
    3. TVI 在视觉 token 中注入时间信息，可选注入 yaw/角度信息。
    4. LLM 输入序列为 [文本指令, 历史视觉, 当前视觉, ACT查询token]。
    5. 取最后 ACT token 的隐藏状态，送入 planner head 输出未来路点。

    这个类保持原单 Agent 能力；双 Agent 逻辑在 MultiAgentOpenTrackVLA 中单独实现。
    """

    def __init__(self, cfg: ModelConfig, vision_feat_dim: int):
        """初始化单 Agent 模型组件。"""
        super().__init__()
        self.cfg = cfg
        # Load LLM - try ModelScope first if available, fallback to HuggingFace
        try:
            from modelscope import AutoModel as MSAutoModel
            from modelscope import AutoTokenizer as MSAutoTokenizer
            print(f"[LLM] Loading from ModelScope: {cfg.llm_name}")
            self.llm = MSAutoModel.from_pretrained(
                cfg.llm_name,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None
            )
            self.tokenizer = MSAutoTokenizer.from_pretrained(cfg.llm_name)
        except Exception as e:
            print(f"[LLM] ModelScope failed ({e}), using HuggingFace")
            self.llm = AutoModel.from_pretrained(
                cfg.llm_name,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None
            )
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.llm_name)
        self.llm.requires_grad_(not cfg.freeze_llm)
        self.D = self.llm.config.hidden_size
        self.proj = CrossModalityProjector(vision_feat_dim, self.D)
        self.proj.requires_grad_(True)
        self.tvi = TVIEmbedder(self.D, max_time=cfg.max_time)
        self.act_token = nn.Parameter(torch.zeros(1, 1, self.D))
        nn.init.normal_(self.act_token, std=0.02)
        action_dims = 3
        self.action_dims = action_dims
        self.planner = PlannerHead3L(self.D, cfg.n_waypoints, action_dims, use_tanh=cfg.use_tanh_actions)
        self.planner.requires_grad_(True)
        if not cfg.use_angle_tvi:
            for p in self.tvi.angle_proj.parameters():
                p.requires_grad = False

        if cfg.alpha_xy is not None:
            vec = [1.0] * action_dims
            if action_dims >= 2:
                vec[0] = cfg.alpha_xy
                vec[1] = cfg.alpha_xy
            alpha = torch.tensor(vec, dtype=torch.float32).view(1, 1, -1)
        else:
            alpha = torch.tensor((1.0, 1.0, 1.0), dtype=torch.float32).view(1, 1, -1)
        self.register_buffer("alpha_task", alpha)

    def _embed_text(self, instructions: List[str], device):
        """把批量文本指令转成 LLM 输入 embedding。

        输出：
        - emb: 文本 token embedding，shape (B, Ltxt, D)
        - attention_mask: 文本 attention mask，shape (B, Ltxt)
        """
        tok = self.tokenizer(instructions, return_tensors='pt', padding=True, truncation=True, max_length=128)
        tok = {k: v.to(device) for k, v in tok.items()}
        emb = self.llm.get_input_embeddings()(tok['input_ids'])
        return emb, tok['attention_mask']

    def _interleave_tvi(self, tokens: torch.Tensor, t_idx: torch.Tensor, kind_id: int,
                        yaw_per_frame: Optional[torch.Tensor] = None, use_angle: bool = False) -> torch.Tensor:
        """在单 Agent 视觉 token 中插入时间/角度 marker。

        输入：
        - tokens: (B, N, D)，已经投影到 LLM 维度的视觉 token。
        - t_idx:  (B, N)，每个 token 属于哪一帧。

        输出：
        - (B, N + marker数, D)，每帧视觉块前会加入时间 marker；
          use_angle=True 时还会加入角度 marker。
        """
        B, N, D = tokens.shape
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
                time_tok = self.tvi.make_time_token(tcur, kind_id, device=xb.device).unsqueeze(0)
                items.append(time_tok)
                if use_angle:
                    theta = 0.0
                    if yaw_per_frame is not None and fcount < yaw_per_frame.size(1):
                        theta = float(yaw_per_frame[b, fcount].item())
                    angle_tok = self.tvi.make_angle_token(theta, kind_id, device=xb.device).unsqueeze(0)
                    items.append(angle_tok)
                items.append(xb[i:j])
                i = j
                fcount += 1
            out_list.append(torch.cat(items, dim=0))
        return torch.stack(out_list, dim=0)

    def forward(self,
                coarse_tokens, coarse_tidx,
                fine_tokens, fine_tidx,
                instructions,
                yaw_hist: Optional[torch.Tensor] = None,
                yaw_curr: Optional[torch.Tensor] = None,
                bbox_feat: Optional[torch.Tensor] = None):
        """单 Agent 前向传播。

        输入：
        - coarse_tokens/coarse_tidx: 历史帧粗 token 与帧编号。
        - fine_tokens/fine_tidx: 当前帧细 token 与帧编号。
        - instructions: 文本任务指令。

        输出：
        - tau_pred: (B, n_waypoints, 3)，未来局部路点 [x, y, theta]。

        bbox_feat 参数保留用于兼容调用接口；单 Agent 当前未使用它。
        """
        device = next(self.parameters()).device
        B = coarse_tokens.size(0)
        vis_c = self.proj(coarse_tokens.to(device))
        vis_f = self.proj(fine_tokens.to(device))
        vis_c = self._interleave_tvi(
            vis_c, coarse_tidx.to(device), kind_id=0,
            yaw_per_frame=yaw_hist, use_angle=self.cfg.use_angle_tvi
        )
        vis_f = self._interleave_tvi(
            vis_f, fine_tidx.to(device), kind_id=1,
            yaw_per_frame=yaw_curr, use_angle=self.cfg.use_angle_tvi
        )
        txt_emb, txt_mask = self._embed_text(instructions, device)
        extra = []        
        act = self.act_token.expand(B, 1, -1)
        pieces = [txt_emb] + ([extra[0]] if extra else []) + [vis_c, vis_f, act]
        seq = torch.cat(pieces, dim=1).to(self.llm.dtype)
        extra_len = (extra[0].size(1) if extra else 0)
        attn = torch.cat([
            txt_mask,
            torch.ones(B, extra_len + vis_c.size(1) + vis_f.size(1) + 1, dtype=torch.long, device=device)
        ], dim=1)
        out = self.llm(inputs_embeds=seq, attention_mask=attn, output_hidden_states=True, use_cache=False)
        h_act = out.last_hidden_state[:, -1, :]
        h_act = h_act.float()
        a_hat = self.planner(h_act)
        tau_pred = a_hat * self.alpha_task
        return tau_pred


@dataclass
# ----------------------- 双 Agent模型配置与组件 -----------------------

class MultiAgentModelConfig:
    """双 Agent 模型配置。

    这个配置同时服务两个变体：
    - model.py-base: 只做双 Agent 视觉/文本融合和 waypoint 预测。
    - model.py-full: 在 base 路径上额外打开 bbox token、GND token 和 grounding head。

    数据流相关字段：
    - num_agents 固定为 2：默认 agent1=无人机，agent2=机器狗。
    - max_views/num_kinds 控制 TVI/agent/type embedding 的词表大小。
    - insert_time_tokens=True 时，会在每一帧视觉 token 前插入时间 marker。
    - use_agent_text_markers 只控制 shared base 中视觉流前的语言 marker；
      关闭后仍保留逐视觉 token 的 agent/view/kind/time embedding。
    - bbox_delta_scale 控制 grounding head 对输入 bbox 的修正幅度。
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
    use_grounding: bool = True
    use_bbox_tokens: bool = True
    return_token_logits: bool = False
    text_max_length: int = 128
    use_agent_text_markers: bool = True

    @property
    def is_base_variant(self) -> bool:
        """True 表示当前配置是无 grounding 的 base 对照模型。"""
        return not self.use_grounding and not self.use_bbox_tokens


class MultiAgentTVIEmbedder(nn.Module):
    """双 Agent 版本的 TVI/类型编码器。

    处理逻辑：
    1. time_emb 表示帧序号，让 LLM 区分历史帧和当前帧的时间位置。
    2. view_emb 表示视角/传感器来源，这里通常 agent0=无人机视角，agent1=机器狗视角。
    3. kind_emb 表示 token 类型，例如历史视觉、当前视觉、bbox、动作查询、grounding 查询。
    4. agent_emb 表示实体身份，避免两个 Agent 的视觉 token 混在同一序列后失去归属。
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

    def type_embedding(self, kind_id: int, agent_id: int, view_id: int, device: torch.device) -> torch.Tensor:
        """生成“类型 + Agent 身份 + 视角”的公共 embedding。

        输出 shape: (D,)
        后续会广播加到一整段视觉 token 或查询 token 上。
        """
        kind = torch.tensor(kind_id, dtype=torch.long, device=device)
        agent = torch.tensor(agent_id, dtype=torch.long, device=device)
        view = torch.tensor(view_id, dtype=torch.long, device=device)
        return self.kind_emb(kind) + self.agent_emb(agent) + self.view_emb(view)

    def add_visual_tvi(
        self,
        tokens: torch.Tensor,
        t_idx: torch.Tensor,
        kind_id: int,
        agent_id: int,
        view_id: int,
    ) -> torch.Tensor:
        """给视觉 token 逐 token 加时间/类型/Agent 编码。

        输入：
        - tokens: (B, N, D)，已经投影到 LLM hidden size 的视觉 token。
        - t_idx:  (B, N)，每个视觉 token 属于第几帧。

        输出：
        - (B, N, D)，shape 不变，但每个 token 已带有时间、来源 Agent、视角和 token 类型信息。
        """
        device = tokens.device
        # Do not use clamp_ here. agent1/agent2 t_idx are often views from the same
        # stacked tensor, and in-place mutation breaks autograd version checks.
        t_idx = t_idx.to(device=device, dtype=torch.long).clamp(0, self.time_emb.num_embeddings - 1)
        type_emb = self.type_embedding(kind_id, agent_id, view_id, device)
        return tokens + self.time_emb(t_idx) + type_emb.view(1, 1, -1)

    def make_marker(
        self,
        t_scalar: int,
        kind_id: int,
        agent_id: int,
        view_id: int,
        device: torch.device,
    ) -> torch.Tensor:
        """构造插入序列中的时间 marker token。

        这个 token 会放在每帧视觉 token 块之前，显式告诉 LLM：
        “接下来这一段视觉特征来自哪个 Agent 的第 t 帧”。
        """
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
        """构造可选 yaw/朝向 marker。

        数据流：theta -> [sin(theta), cos(theta)] -> Linear -> 加 Agent/type/view embedding。
        只有 cfg.use_angle_tvi=True 时，模型才会把该 token 插入视觉序列。
        """
        theta = (theta + math.pi) % (2 * math.pi) - math.pi
        sincos = torch.tensor([math.sin(theta), math.cos(theta)], dtype=self.angle_proj.weight.dtype, device=device)
        angle = F.linear(sincos, self.angle_proj.weight, self.angle_proj.bias)
        return angle + self.type_embedding(kind_id, agent_id, view_id, device)

    def make_bbox_token(self, bbox: torch.Tensor, agent_id: int, view_id: int) -> torch.Tensor:
        """把归一化 bbox 编成一个 LLM token。

        输入 bbox:
        - (B, 4) 或 (4,)，格式为 cx, cy, w, h，取值范围 [0, 1]。

        输出：
        - (B, 1, D)，作为当前 Agent 视觉流末尾的 bbox 先验 token。
        """
        if bbox.dim() == 1:
            bbox = bbox.unsqueeze(0)
        bbox = bbox.to(dtype=self.bbox_proj[1].weight.dtype)
        emb = self.bbox_proj(bbox)
        if emb.dim() == 2:
            emb = emb.unsqueeze(1)
        type_emb = self.type_embedding(KIND_BBOX, agent_id, view_id, emb.device)
        return emb + type_emb.view(1, 1, -1)

    def make_query_token(self, base: torch.Tensor, kind_id: int, agent_id: int, view_id: int) -> torch.Tensor:
        """给可学习查询 token 加类型/Agent/视角标识。

        ACT 查询 token 用于规划头；GND 查询 token 用于 bbox/visibility 辅助头。
        """
        type_emb = self.type_embedding(kind_id, agent_id, view_id, base.device)
        return base + type_emb.view(1, 1, -1)


class GroundingHead(nn.Module):
    """逐 Agent 目标 grounding 辅助头。

    输入：
    - h_gnd: 两个 GND 查询 token 的隐藏状态，shape (B, 2, D)。
    - bbox_feat: 原始 bbox 先验，shape (B, 2, 4)，可选。
    - bbox_valid_mask: 每个 Agent 是否有可用 bbox 先验，shape (B, 2)。
    - visual_tokens: 视觉 token 池，shape (B, 2, Nv, D)，用于 spatial cross-attention。

    输出：
    - refined_bbox: 两个 Agent 各自坐标系下的 bbox，shape (B, 2, 4)。
    - relative_pose: 每个 Agent 坐标系下的目标空间量，shape (B, 2, 5)，
      格式 [dx_m, dy_m, dz_m, sin(d_yaw), cos(d_yaw)]。
    - spatial_emb: GND token 融合视觉 token 后的空间 embedding，shape (B, 2, D)。
    - visible_logits/visible_score: 两个 Agent 视角下目标是否可见。
    - token_logits: 可选 token grounding 分数，便于后续做更细监督。
    """

    def __init__(
        self,
        d_model: int,
        num_agents: int = 2,
        bbox_delta_scale: float = 0.25,
        num_heads: int = 8,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.bbox_delta_scale = bbox_delta_scale
        while num_heads > 1 and d_model % num_heads != 0:
            num_heads -= 1
        hid = d_model * 2
        self.gnd_cross_norm = nn.LayerNorm(d_model)
        self.visual_cross_norm = nn.LayerNorm(d_model)
        self.spatial_cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True,
        )
        self.spatial_ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hid),
            nn.GELU(),
            nn.Linear(hid, d_model),
        )
        self.bbox_absolute = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hid),
            nn.GELU(),
            nn.Linear(hid, 4),
        )
        self.bbox_residual = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hid),
            nn.GELU(),
            nn.Linear(hid, 4),
        )
        self.visibility = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hid),
            nn.GELU(),
            nn.Linear(hid, 1),
        )
        self.relative_pose = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hid),
            nn.GELU(),
            nn.Linear(hid, 5),
        )
        self.per_agent_token_query = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )

    def forward(
        self,
        h_gnd: torch.Tensor,
        bbox_feat: Optional[torch.Tensor] = None,
        bbox_valid_mask: Optional[torch.Tensor] = None,
        visual_tokens: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """根据两个 GND 隐藏状态分别预测 bbox、可见性和相对空间量。

        处理逻辑：
        1. GND token 作为 query，从对应 Agent 的视觉 token 中 cross-attend 目标线索。
        2. 有有效 bbox prior 的 Agent 使用 residual head 做有界修正。
        3. prior 被丢弃或缺失的 Agent 使用 absolute head 从视觉直接检测。
        4. visibility、relative_pose 和 token grounding 都由 spatial_emb 预测。
        """
        if h_gnd.dim() != 3 or h_gnd.size(1) != self.num_agents:
            raise ValueError(f"h_gnd must have shape (B, {self.num_agents}, D), got {tuple(h_gnd.shape)}")
        B = h_gnd.size(0)
        spatial_emb = h_gnd
        if visual_tokens is not None:
            if visual_tokens.dim() != 4 or visual_tokens.size(0) != B or visual_tokens.size(1) != self.num_agents:
                raise ValueError(
                    "visual_tokens must have shape "
                    f"(B, {self.num_agents}, Nv, D), got {tuple(visual_tokens.shape)}"
                )
            q = self.gnd_cross_norm(h_gnd).reshape(B * self.num_agents, 1, -1)
            kv = visual_tokens.to(device=h_gnd.device, dtype=h_gnd.dtype)
            kv = self.visual_cross_norm(kv).reshape(B * self.num_agents, kv.size(2), kv.size(3))
            attn_out, _ = self.spatial_cross_attn(q, kv, kv, need_weights=False)
            spatial_emb = h_gnd + attn_out.reshape(B, self.num_agents, -1)
        spatial_emb = spatial_emb + self.spatial_ffn(spatial_emb)

        absolute_bbox = torch.sigmoid(self.bbox_absolute(spatial_emb))
        delta = torch.tanh(self.bbox_residual(spatial_emb)) * self.bbox_delta_scale

        if bbox_feat is None:
            bbox_prior = torch.zeros_like(absolute_bbox)
            valid_prior = torch.zeros(B, self.num_agents, dtype=torch.bool, device=h_gnd.device)
        else:
            bbox_prior = bbox_feat.to(device=h_gnd.device, dtype=delta.dtype)
            if bbox_valid_mask is None:
                valid_prior = torch.ones(B, self.num_agents, dtype=torch.bool, device=h_gnd.device)
            else:
                valid_prior = bbox_valid_mask.to(device=h_gnd.device, dtype=torch.bool)
                if valid_prior.shape != (B, self.num_agents):
                    raise ValueError(
                        f"bbox_valid_mask must have shape (B, {self.num_agents}), got {tuple(valid_prior.shape)}"
                    )
        residual_bbox = (bbox_prior + delta).clamp(0.0, 1.0)
        refined_bbox = torch.where(valid_prior.unsqueeze(-1), residual_bbox, absolute_bbox)

        visible_logits = self.visibility(spatial_emb).squeeze(-1)
        visible_score = torch.sigmoid(visible_logits)
        relative_pose = self.relative_pose(spatial_emb)
        token_logits = None
        if visual_tokens is not None:
            queries = self.per_agent_token_query(spatial_emb)
            token_logits = torch.einsum("bad,band->ban", queries.float(), visual_tokens.float())
            token_logits = token_logits / math.sqrt(max(1, queries.size(-1)))

        return {
            "refined_bbox": refined_bbox,
            "absolute_bbox": absolute_bbox,
            "bbox_prior_mask": valid_prior,
            "relative_pose": relative_pose,
            "spatial_emb": spatial_emb,
            "visible_logits": visible_logits,
            "visible_score": visible_score,
            "token_logits": token_logits,
        }


# ----------------------- 双 Agent模型主干 -----------------------

class MultiAgentOpenTrackVLA(nn.Module):
    """双 Agent OpenTrackVLA 主模型。

    本类包含两个清晰变体，训练脚本通过 cfg 开关选择：

    model.py-base:
      默认：[任务文本, agent1文本marker, agent1视觉, agent2文本marker, agent2视觉, ACT1, ACT2]
      无 marker：[任务文本, agent1视觉, agent2视觉, ACT1, ACT2]
      -> planner_agent1/2 -> waypoints
      这个路径不使用 bbox token、不创建 GND token、不训练 grounding head。

    model.py-full:
      在 base 序列和 planner 之外，额外启用 bbox token、GND1/GND2、grounding head，
      并把 grounding 的 spatial embedding 注入 ACT hidden。

    典型输入 shape:
      coarse_tokens: (B, 2, Nc, C)
      coarse_tidx:   (B, 2, Nc)
      fine_tokens:   (B, 2, Nf, C)
      fine_tidx:     (B, 2, Nf)
      bbox_feat:     (B, 2, 4)

    核心数据流：
    1. 每个 Agent 的历史粗 token / 当前细 token 先通过 projector 映射到 LLM hidden size。
    2. TVI 给视觉 token 加上时间、Agent、视角、类型编码，并可插入帧级 marker。
    3. base 序列可选是否在两个视觉流前加入 Agent 文本 marker。
    4. ACT1/ACT2 位置的隐藏状态分别进入两个 planner head，输出两套路点。
    5. full 模式才会把 bbox token 拼入视觉流，并追加 GND1/GND2 做 bbox/visibility/relative_pose。
    """

    def __init__(self, cfg: MultiAgentModelConfig, vision_feat_dim: int):
        """初始化模型模块。

        模块组成：
        - llm/tokenizer: 文本和多模态序列的主干。
        - proj: 把 DINO/SigLIP 拼接后的视觉维度 C 投影到 LLM 维度 D。
        - tvi: 注入时间、Agent、视角、token 类型和 bbox 信息。
        - planner_agent1/2: 两个独立规划头，避免无人机/机器狗动作分布被强行共享。
        - grounding_head: 仅 full 模式存在，用于 bbox refinement 与 visibility 辅助监督。
        """
        super().__init__()
        if cfg.num_agents != 2:
            raise ValueError("MultiAgentOpenTrackVLA currently expects exactly two agents.")
        self.cfg = cfg
        rank = int(os.environ.get("RANK", "0"))
        variant_name = "model.py-base" if cfg.is_base_variant else "model.py-full"
        print(f"[MODEL][rank {rank}] variant={variant_name}", flush=True)
        t0 = time.time()
        print(f"[MODEL][rank {rank}] loading LLM weights: {cfg.llm_name}", flush=True)
        load_kwargs = {"dtype": torch.bfloat16} if torch.cuda.is_available() else {}
        use_modelscope = os.environ.get("TRACKVLA_USE_MODELSCOPE", "0").strip().lower() in {"1", "true", "yes", "on"}
        if use_modelscope:
            try:
                from modelscope import AutoModel as MSAutoModel
                from modelscope import AutoTokenizer as MSAutoTokenizer

                self.llm = MSAutoModel.from_pretrained(cfg.llm_name, **load_kwargs)
                self.tokenizer = MSAutoTokenizer.from_pretrained(cfg.llm_name)
            except Exception:
                self.llm = AutoModel.from_pretrained(cfg.llm_name, **load_kwargs)
                self.tokenizer = AutoTokenizer.from_pretrained(cfg.llm_name)
        else:
            self.llm = AutoModel.from_pretrained(cfg.llm_name, **load_kwargs)
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.llm_name)
        print(f"[MODEL][rank {rank}] LLM/tokenizer loaded in {time.time() - t0:.1f}s", flush=True)

        self.llm.requires_grad_(not cfg.freeze_llm)
        self.llm_dtype = next(self.llm.parameters()).dtype
        self.D = int(self.llm.config.hidden_size)

        # =========================
        # Shared model.py-base path
        # =========================
        # base 和 full 都会经过这一路：视觉投影 -> TVI 编码 -> LLM -> 两个 planner head。
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
        nn.init.normal_(self.act_token_1, std=0.02)
        nn.init.normal_(self.act_token_2, std=0.02)

        self.planner_agent1 = PlannerHead3L(self.D, cfg.n_waypoints, cfg.action_dims, cfg.use_tanh_actions)
        self.planner_agent2 = PlannerHead3L(self.D, cfg.n_waypoints, cfg.action_dims, cfg.use_tanh_actions)

        # ====================================
        # Optional model.py-full grounding path
        # ====================================
        # 只有 full 模式会创建 GND token/head。base 模式不会产生这些可训练参数。
        if cfg.use_grounding:
            self.gnd_token_1 = nn.Parameter(torch.zeros(1, 1, self.D))
            self.gnd_token_2 = nn.Parameter(torch.zeros(1, 1, self.D))
            nn.init.normal_(self.gnd_token_1, std=0.02)
            nn.init.normal_(self.gnd_token_2, std=0.02)
            self.grounding_head = GroundingHead(self.D, cfg.num_agents, cfg.bbox_delta_scale)
            self.grounding_to_act = nn.Sequential(
                nn.LayerNorm(self.D),
                nn.Linear(self.D, self.D),
                nn.GELU(),
                nn.Linear(self.D, self.D),
            )
            self.grounding_act_gate = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))
            if not cfg.return_token_logits:
                self.grounding_head.per_agent_token_query.requires_grad_(False)
        else:
            self.grounding_head = None
            self.grounding_to_act = None
            self.register_parameter("gnd_token_1", None)
            self.register_parameter("gnd_token_2", None)
            self.register_parameter("grounding_act_gate", None)

        if not cfg.use_angle_tvi:
            for p in self.tvi.angle_proj.parameters():
                p.requires_grad = False
        if not cfg.use_bbox_tokens:
            self.tvi.bbox_proj.requires_grad_(False)

        alpha = torch.ones(1, 1, cfg.action_dims, dtype=torch.float32)
        if cfg.alpha_xy is not None and cfg.action_dims >= 2:
            alpha[..., 0:2] = float(cfg.alpha_xy)
        self.register_buffer("alpha_task", alpha)

    def train(self, mode: bool = True):
        """Keep the frozen LLM in eval mode while training the lightweight heads."""
        super().train(mode)
        if getattr(self.cfg, "freeze_llm", False):
            self.llm.eval()
        return self

    def _embed_text(self, instructions: List[str], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """把自然语言指令转成 LLM 输入 embedding。

        输入: List[str]，batch 内每条样本一条 instruction。
        输出:
        - txt_emb:  (B, Ltxt, D)
        - txt_mask: (B, Ltxt)，用于 attention_mask。
        """
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
        """检查并归一化 bbox 输入。

        支持：
        - None：返回全 0 bbox。
        - (B, 4)：单 bbox 会扩展给两个 Agent。
        - (B, 2, 4)：每个 Agent 一套 bbox。

        输出固定为 (B, 2, 4)，并 clamp 到 [0, 1]。
        """
        if bbox is None:
            return torch.zeros(B, self.cfg.num_agents, 4, dtype=torch.float32, device=device)
        bbox = bbox.to(device=device, dtype=torch.float32)
        if bbox.dim() == 2:
            bbox = bbox.unsqueeze(1).expand(-1, self.cfg.num_agents, -1)
        if bbox.size(1) != self.cfg.num_agents or bbox.size(-1) != 4:
            raise ValueError(f"bbox_feat must have shape (B, 2, 4) or (B, 4), got {tuple(bbox.shape)}")
        if bbox.detach().amax() > 1.5:
            raise ValueError("bbox_feat should be normalized to [0, 1].")
        return bbox.clone().clamp(0.0, 1.0)

    def _interleave_markers(
        self,
        tokens: torch.Tensor,
        t_idx: torch.Tensor,
        kind_id: int,
        agent_id: int,
        view_id: int,
        yaw_per_frame: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """在视觉 token 序列中插入帧级 marker。

        输入 tokens 已经带有逐 token TVI，shape (B, N, D)。
        t_idx 中相同帧号的一段 token 会被视为同一帧块。

        输出示意：
        [time_marker(t0), frame0_tokens, time_marker(t1), frame1_tokens, ...]
        如果 use_angle_tvi=True，则每帧还会额外插入 angle_marker。
        """
        if not self.cfg.insert_time_tokens:
            return tokens
        B, N, _ = tokens.shape
        device = tokens.device

        # Grouping is metadata work. Keep it on CPU to avoid thousands of
        # CUDA .item() synchronizations in the training forward pass.
        tidx_cpu = t_idx.detach().to("cpu").clamp(0, self.tvi.time_emb.num_embeddings - 1).long()
        first_tidx = tidx_cpu[0].tolist()
        spans: List[Tuple[int, int]] = []
        i = 0
        while i < N:
            j = i + 1
            while j < N and first_tidx[j] == first_tidx[i]:
                j += 1
            spans.append((i, j))
            i = j

        pieces: List[torch.Tensor] = []
        kind = torch.full((B,), int(kind_id), dtype=torch.long, device=device)
        agent = torch.full((B,), int(agent_id), dtype=torch.long, device=device)
        view = torch.full((B,), int(view_id), dtype=torch.long, device=device)
        type_emb = self.tvi.kind_emb(kind) + self.tvi.agent_emb(agent) + self.tvi.view_emb(view)
        for frame_idx, (start, end) in enumerate(spans):
            tvals = tidx_cpu[:, start].to(device=device, non_blocking=True)
            marker = self.tvi.time_emb(tvals) + type_emb
            pieces.append(marker.unsqueeze(1))
            if self.cfg.use_angle_tvi:
                if yaw_per_frame is not None and frame_idx < yaw_per_frame.size(1):
                    theta = yaw_per_frame[:, frame_idx].to(device=device, dtype=torch.float32)
                else:
                    theta = torch.zeros(B, device=device, dtype=torch.float32)
                theta = (theta + math.pi) % (2 * math.pi) - math.pi
                sincos = torch.stack([torch.sin(theta), torch.cos(theta)], dim=-1).to(self.tvi.angle_proj.weight.dtype)
                angle = self.tvi.angle_proj(sincos) + type_emb
                pieces.append(angle.unsqueeze(1))
            pieces.append(tokens[:, start:end])
        return torch.cat(pieces, dim=1)

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

        数据流：
        coarse/fine cache token (C维)
        -> CrossModalityProjector 投影到 D维
        -> add_visual_tvi 加逐 token 时间/Agent/type 信息
        -> _interleave_markers 插入帧级 marker
        -> 可选 make_bbox_token 添加当前 bbox token（full 模式）

        返回：
        - agent_seq: 喂给 LLM 的该 Agent 完整视觉序列。
        - visual_for_grounding: 不含 marker/bbox 的视觉 token，用于可选 token grounding。
        """
        view_id = agent_id
        vis_c = self.proj(coarse_tokens)
        vis_f = self.proj(fine_tokens)
        vis_c = self.tvi.add_visual_tvi(vis_c, coarse_tidx, KIND_HISTORY, agent_id, view_id)
        vis_f = self.tvi.add_visual_tvi(vis_f, fine_tidx, KIND_CURRENT, agent_id, view_id)
        visual_for_grounding = torch.cat([vis_c, vis_f], dim=1)
        seq_c = self._interleave_markers(vis_c, coarse_tidx, KIND_HISTORY, agent_id, view_id, yaw_hist)
        seq_f = self._interleave_markers(vis_f, fine_tidx, KIND_CURRENT, agent_id, view_id, yaw_curr)
        pieces = [seq_c, seq_f]
        if self.cfg.use_bbox_tokens:
            pieces.append(self.tvi.make_bbox_token(bbox_feat, agent_id, view_id))
        return torch.cat(pieces, dim=1), visual_for_grounding

    def _base_grounding_stub(
        self,
        batch_size: int,
        bbox: torch.Tensor,
        has_bbox_prior: bool,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """model.py-base 的 grounding 占位输出。

        base 模式不训练 bbox/visibility/relative_pose。这里仍返回同名字段，
        是为了让 train/eval 代码可以复用 full 模式的输出解析逻辑。
        """
        refined_bbox = bbox if has_bbox_prior else torch.zeros(batch_size, self.cfg.num_agents, 4, device=device)
        return {
            "refined_bbox": refined_bbox,
            "absolute_bbox": refined_bbox,
            "bbox_prior_mask": torch.full(
                (batch_size, self.cfg.num_agents),
                bool(has_bbox_prior),
                dtype=torch.bool,
                device=device,
            ),
            "relative_pose": None,
            "spatial_emb": None,
            "visible_logits": torch.zeros(batch_size, self.cfg.num_agents, dtype=dtype, device=device),
            "visible_score": torch.full(
                (batch_size, self.cfg.num_agents),
                0.5,
                dtype=dtype,
                device=device,
            ),
            "token_logits": None,
        }

    def _embed_agent_role_marker(
        self,
        batch_size: int,
        agent_id: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """把 Agent 身份说明转成文本 marker。

        这是语言 token，不属于视觉流本身，所以不放进 _encode_agent_stream。
        它作为 a*_seq 的前缀，显式告诉 LLM 后续视觉 token 属于哪个实体。
        """
        if agent_id == 0:
            text = "Agent 0 is the aerial drone. The following visual tokens belong to the drone."
        elif agent_id == 1:
            text = "Agent 1 is the ground robot dog. The following visual tokens belong to the robot dog."
        else:
            raise ValueError(f"Unsupported agent_id={agent_id}")
        return self._embed_text([text] * batch_size, device)

    def _default_agent_role_text(self, agent_id: int) -> str:
        if agent_id == 0:
            return "Agent 0 is the aerial drone. The following visual tokens belong to the drone."
        if agent_id == 1:
            return "Agent 1 is the ground robot dog. The following visual tokens belong to the robot dog."
        raise ValueError(f"Unsupported agent_id={agent_id}")

    @staticmethod
    def _ensure_text_batch(texts: Optional[List[str]], fallback: List[str], batch_size: int) -> List[str]:
        if texts is None:
            texts = fallback
        if len(texts) != batch_size:
            if len(texts) == 1:
                texts = texts * batch_size
            else:
                raise ValueError(f"Expected {batch_size} instructions, got {len(texts)}")
        return texts

    def _build_shared_context_inputs(
        self,
        txt_emb: torch.Tensor,
        txt_mask: torch.Tensor,
        a1_seq: torch.Tensor,
        a2_seq: torch.Tensor,
        act1: torch.Tensor,
        act2: torch.Tensor,
        device: torch.device,
        agent1_txt_emb: Optional[torch.Tensor] = None,
        agent1_txt_mask: Optional[torch.Tensor] = None,
        agent2_txt_emb: Optional[torch.Tensor] = None,
        agent2_txt_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], int, int]:
        """构造 shared-context LLM 输入 pieces。

        base 默认顺序：
        [joint_task_text, agent0_instruction, agent0_visual_seq,
         agent1_instruction, agent1_visual_seq, ACT1, ACT2]

        use_agent_text_markers=False 时严格使用：
        [joint_task_text, agent0_visual_seq, agent1_visual_seq, ACT1, ACT2]

        full 顺序保持原始结构：
        [task_text, agent0_visual_seq, agent1_visual_seq, ACT1, ACT2]

        agent instruction 不放进 a1_seq/a2_seq，原因是 a*_seq 表示“视觉 token 流”；
        agent instruction 是语言提示，作为视觉流前缀更清楚，也不污染视觉编码函数。
        """
        B = txt_emb.size(0)
        pieces: List[torch.Tensor] = [txt_emb]
        masks: List[torch.Tensor] = [txt_mask]

        def append_dense(piece: torch.Tensor) -> None:
            pieces.append(piece)
            masks.append(torch.ones(B, piece.size(1), dtype=torch.long, device=device))

        def append_with_mask(piece: torch.Tensor, mask: torch.Tensor) -> None:
            pieces.append(piece)
            masks.append(mask)

        if self.cfg.is_base_variant and self.cfg.use_agent_text_markers:
            if agent1_txt_emb is None or agent1_txt_mask is None:
                marker0, marker0_mask = self._embed_agent_role_marker(B, 0, device)
                agent1_txt_emb, agent1_txt_mask = marker0, marker0_mask
            if agent2_txt_emb is None or agent2_txt_mask is None:
                marker1, marker1_mask = self._embed_agent_role_marker(B, 1, device)
                agent2_txt_emb, agent2_txt_mask = marker1, marker1_mask
            append_with_mask(agent1_txt_emb, agent1_txt_mask)
            append_dense(a1_seq)
            append_with_mask(agent2_txt_emb, agent2_txt_mask)
            append_dense(a2_seq)
        else:
            append_dense(a1_seq)
            append_dense(a2_seq)

        act1_index = len(pieces)
        append_dense(act1)
        act2_index = len(pieces)
        append_dense(act2)
        return pieces, masks, act1_index, act2_index

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
        """兼容两种调用方式，把输入拆成 agent1/agent2。

        方式一：训练时推荐的 stacked 输入，形如 (B, 2, N, C)。
        方式二：显式传 agent1_* / agent2_*，便于调试或单独调用。

        输出顺序：
        agent1 coarse/tidx/fine/tidx, agent2 coarse/tidx/fine/tidx。
        """
        if coarse_tokens is not None:
            if coarse_tokens.dim() != 4 or fine_tokens is None or coarse_tidx is None or fine_tidx is None:
                raise ValueError("Stacked inputs require coarse/fine tokens with shape (B, 2, N, C).")
            return (
                # 切片
                coarse_tokens[:, 0], coarse_tidx[:, 0], fine_tokens[:, 0], fine_tidx[:, 0],
                coarse_tokens[:, 1], coarse_tidx[:, 1], fine_tokens[:, 1], fine_tidx[:, 1],
            )
        required = [
            agent1_coarse_tokens, agent1_coarse_tidx, agent1_fine_tokens, agent1_fine_tidx,
            agent2_coarse_tokens, agent2_coarse_tidx, agent2_fine_tokens, agent2_fine_tidx,
        ]
        if any(x is None for x in required):
            raise ValueError("Pass either stacked inputs or all agent1_/agent2_ split inputs.")
        return (
            agent1_coarse_tokens, agent1_coarse_tidx, agent1_fine_tokens, agent1_fine_tidx,
            agent2_coarse_tokens, agent2_coarse_tidx, agent2_fine_tokens, agent2_fine_tidx,
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
        joint_instructions: Optional[List[str]] = None,
        agent1_instructions: Optional[List[str]] = None,
        agent2_instructions: Optional[List[str]] = None,
        return_dict: bool = True,
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, Dict[str, Optional[torch.Tensor]]]]:
        """双 Agent 前向传播。

        主要输入：
        - coarse_tokens/fine_tokens: 视觉 cache，stacked 形式为 (B, 2, N, C)。
        - bbox_feat: 两个 Agent 的目标框，(B, 2, 4)，格式 cxcywh_norm。
        - instructions: 文本任务指令。

        主要输出：
        - waypoints: (B, 2, n_waypoints, 3)，两个 Agent 的未来局部路点。
        - refined_bbox/visible_score: base 模式是兼容占位；full 模式是 grounding head 输出。
        """
        device = next(self.parameters()).device
        a1_c, a1_ct, a1_f, a1_ft, a2_c, a2_ct, a2_f, a2_ft = self._split_stacked_inputs(
            coarse_tokens, coarse_tidx, fine_tokens, fine_tidx,
            agent1_coarse_tokens, agent1_coarse_tidx, agent1_fine_tokens, agent1_fine_tidx,
            agent2_coarse_tokens, agent2_coarse_tidx, agent2_fine_tokens, agent2_fine_tidx,
        )
        a1_c, a1_ct, a1_f, a1_ft = a1_c.to(device), a1_ct.to(device), a1_f.to(device), a1_ft.to(device)
        a2_c, a2_ct, a2_f, a2_ft = a2_c.to(device), a2_ct.to(device), a2_f.to(device), a2_ft.to(device)
        B = a1_c.size(0)
        if instructions is None:
            instructions = ["follow the person"] * B
        joint_instructions = self._ensure_text_batch(joint_instructions, instructions, B)
        use_agent_text = self.cfg.is_base_variant and self.cfg.use_agent_text_markers
        if (
            self.cfg.is_base_variant
            and not self.cfg.use_agent_text_markers
            and (agent1_instructions is not None or agent2_instructions is not None)
        ):
            raise ValueError(
                "Per-agent instructions were provided while use_agent_text_markers=False. "
                "Pass only the joint instruction for the five-piece shared base sequence."
            )
        if use_agent_text:
            agent1_instructions = self._ensure_text_batch(
                agent1_instructions,
                [self._default_agent_role_text(0)] * B,
                B,
            )
            agent2_instructions = self._ensure_text_batch(
                agent2_instructions,
                [self._default_agent_role_text(1)] * B,
                B,
            )

        has_bbox_prior = bbox_feat is not None or agent1_bbox_feat is not None or agent2_bbox_feat is not None
        if bbox_feat is not None:
            bbox = self._normalize_bbox(bbox_feat, B, device)
        else:
            b1 = torch.zeros(B, 4, dtype=torch.float32, device=device) if agent1_bbox_feat is None else agent1_bbox_feat.to(device)
            b2 = torch.zeros(B, 4, dtype=torch.float32, device=device) if agent2_bbox_feat is None else agent2_bbox_feat.to(device)
            bbox = torch.stack([b1, b2], dim=1).float().clamp(0.0, 1.0)

        if yaw_hist is not None and yaw_hist.dim() == 3:
            agent1_yaw_hist, agent2_yaw_hist = yaw_hist[:, 0], yaw_hist[:, 1]
        if yaw_curr is not None and yaw_curr.dim() == 3:
            agent1_yaw_curr, agent2_yaw_curr = yaw_curr[:, 0], yaw_curr[:, 1]

        a1_seq, a1_visual = self._encode_agent_stream(
            a1_c, a1_ct, a1_f, a1_ft, bbox[:, 0], 0,
            yaw_hist=agent1_yaw_hist, yaw_curr=agent1_yaw_curr,
        )
        a2_seq, a2_visual = self._encode_agent_stream(
            a2_c, a2_ct, a2_f, a2_ft, bbox[:, 1], 1,
            yaw_hist=agent2_yaw_hist, yaw_curr=agent2_yaw_curr,
        )

        # =========================
        # Shared model.py-base path
        # =========================
        # 1. 全局联合指令始终进入 LLM；Agent 文本只在显式开启 marker 时计算。
        txt_emb, txt_mask = self._embed_text(joint_instructions, device)
        agent1_txt_emb = agent1_txt_mask = None
        agent2_txt_emb = agent2_txt_mask = None
        if use_agent_text:
            agent1_txt_emb, agent1_txt_mask = self._embed_text(agent1_instructions, device)
            agent2_txt_emb, agent2_txt_mask = self._embed_text(agent2_instructions, device)
        # 2. base 只需要两个 ACT 查询 token，分别读取两个 Agent 的规划状态。
        act1 = self.tvi.make_query_token(self.act_token_1.expand(B, 1, -1), KIND_ACT, 0, 0)
        act2 = self.tvi.make_query_token(self.act_token_2.expand(B, 1, -1), KIND_ACT, 1, 1)
        # 3. 构造 shared context；无 marker 模式严格为 [文本, a1视觉, a2视觉, ACT1, ACT2]。
        pieces, mask_pieces, act1_index, act2_index = self._build_shared_context_inputs(
            txt_emb,
            txt_mask,
            a1_seq,
            a2_seq,
            act1,
            act2,
            device,
            agent1_txt_emb=agent1_txt_emb,
            agent1_txt_mask=agent1_txt_mask,
            agent2_txt_emb=agent2_txt_emb,
            agent2_txt_mask=agent2_txt_mask,
        )

        def append_dense(piece: torch.Tensor) -> None:
            pieces.append(piece)
            mask_pieces.append(torch.ones(B, piece.size(1), dtype=torch.long, device=device))

        if self.cfg.use_grounding:
            # Optional model.py-full addition:
            # GND token 只在 full 模式追加，位于完整双视角上下文之后。
            gnd1 = self.tvi.make_query_token(self.gnd_token_1.expand(B, 1, -1), KIND_GND, 0, 0)
            gnd2 = self.tvi.make_query_token(self.gnd_token_2.expand(B, 1, -1), KIND_GND, 1, 1)
            append_dense(gnd1)
            append_dense(gnd2)
        lengths = [p.size(1) for p in pieces]
        act1_pos = sum(lengths[:act1_index])
        act2_pos = sum(lengths[:act2_index])

        seq = torch.cat(pieces, dim=1).to(dtype=self.llm_dtype)
        attn = torch.cat(mask_pieces, dim=1)
        # 4. LLM 在同一个上下文里融合文本、无人机视觉流和机器狗视觉流。
        # full 模式下，视觉流尾部还包含 bbox token。
        out = self.llm(inputs_embeds=seq, attention_mask=attn, output_hidden_states=False, use_cache=False)
        hidden = out.last_hidden_state.float()
        h_act1, h_act2 = hidden[:, act1_pos, :], hidden[:, act2_pos, :]
        if self.cfg.use_grounding:
            gnd1_pos = sum(lengths[:-2])
            gnd2_pos = sum(lengths) - 1
            h_gnd = torch.stack([hidden[:, gnd1_pos, :], hidden[:, gnd2_pos, :]], dim=1)
            visual_tokens = torch.stack([a1_visual, a2_visual], dim=1) if a1_visual.size(1) == a2_visual.size(1) else None
            # Without a bbox prior, predict an absolute normalized box. With a prior,
            # preserve the original behavior and predict a bounded refinement.
            grounding = self.grounding_head(
                h_gnd,
                bbox_feat=bbox if has_bbox_prior else None,
                visual_tokens=visual_tokens,
            )
            # 5. 将 grounding 的空间 embedding 注入 ACT 状态，让 bbox/relative pose 辅助监督
            # 不只停留在旁路 head，而能直接影响闭环跟踪动作。
            spatial_emb = grounding["spatial_emb"]
            gate = torch.sigmoid(self.grounding_act_gate).to(dtype=h_act1.dtype)
            h_act1 = h_act1 + gate * self.grounding_to_act(spatial_emb[:, 0])
            h_act2 = h_act2 + gate * self.grounding_to_act(spatial_emb[:, 1])
        else:
            grounding = self._base_grounding_stub(B, bbox, has_bbox_prior, h_act1.dtype, device)

        # 5. Shared waypoint heads: base 和 full 都由两个独立 planner 输出动作。
        agent1_waypoints = self.planner_agent1(h_act1) * self.alpha_task
        agent2_waypoints = self.planner_agent2(h_act2) * self.alpha_task

        if not return_dict:
            return agent1_waypoints, agent2_waypoints, grounding
        return {
            "agent1_waypoints": agent1_waypoints,
            "agent2_waypoints": agent2_waypoints,
            "waypoints": torch.stack([agent1_waypoints, agent2_waypoints], dim=1),
            "refined_bbox": grounding["refined_bbox"],
            "absolute_bbox": grounding["absolute_bbox"],
            "bbox_prior_mask": grounding["bbox_prior_mask"],
            "relative_pose": grounding["relative_pose"],
            "spatial_emb": grounding["spatial_emb"],
            "visible_logits": grounding["visible_logits"],
            "visible_score": grounding["visible_score"],
            "token_logits": grounding["token_logits"],
        }


class MultiAgentSeparateOpenTrackVLA(MultiAgentOpenTrackVLA):
    """双 Agent 独立上下文 waypoint-only 对照模型。

    和 ``MultiAgentOpenTrackVLA`` 的 base 版本相比，本类刻意不把两个 Agent 的
    visual pieces 拼进同一个 LLM 上下文：

    - agent1: [文本, agent1视觉, ACT1] -> LLM -> planner_agent1
    - agent2: [文本, agent2视觉, ACT2] -> LLM -> planner_agent2

    这样可以检验“两个 Agent 共上下文互相可见”是否真的带来收益。为了让对照干净，
    该类只支持 waypoint-only base 设置，不启用 bbox token 和 grounding head。
    """

    def __init__(self, cfg: MultiAgentModelConfig, vision_feat_dim: int):
        if cfg.use_grounding or cfg.use_bbox_tokens:
            raise ValueError(
                "MultiAgentSeparateOpenTrackVLA is a waypoint-only base ablation; "
                "set use_grounding=False and use_bbox_tokens=False."
            )
        super().__init__(cfg, vision_feat_dim)
        rank = int(os.environ.get("RANK", "0"))
        print(f"[MODEL][rank {rank}] separate_agent_context=True", flush=True)

    def _forward_one_agent_context(
        self,
        txt_emb: torch.Tensor,
        txt_mask: torch.Tensor,
        agent_seq: torch.Tensor,
        act_token: torch.Tensor,
    ) -> torch.Tensor:
        """Run one independent [text, one-agent visual, ACT] LLM context."""
        B = agent_seq.size(0)
        device = agent_seq.device
        seq = torch.cat([txt_emb, agent_seq, act_token], dim=1).to(dtype=self.llm_dtype)
        attn = torch.cat(
            [
                txt_mask,
                torch.ones(B, agent_seq.size(1) + 1, dtype=torch.long, device=device),
            ],
            dim=1,
        )
        out = self.llm(inputs_embeds=seq, attention_mask=attn, output_hidden_states=False, use_cache=False)
        return out.last_hidden_state[:, -1, :].float()

    def _role_instructions(self, instructions: List[str], role: str, partner: str) -> List[str]:
        """给独立上下文补充 Agent 身份，避免两次 LLM forward 读到完全相同的角色描述。"""
        return [
            f"You are the {role}. Track the same target person and coordinate with the {partner}. Task: {inst}"
            for inst in instructions
        ]

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
        joint_instructions: Optional[List[str]] = None,
        agent1_instructions: Optional[List[str]] = None,
        agent2_instructions: Optional[List[str]] = None,
        return_dict: bool = True,
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, Dict[str, Optional[torch.Tensor]]]]:
        """双 Agent 独立上下文前向传播。

        输出接口保持和 ``MultiAgentOpenTrackVLA`` 一致，方便复用 train/eval 代码；
        grounding 字段均为 base 兼容占位。
        """
        device = next(self.parameters()).device
        a1_c, a1_ct, a1_f, a1_ft, a2_c, a2_ct, a2_f, a2_ft = self._split_stacked_inputs(
            coarse_tokens, coarse_tidx, fine_tokens, fine_tidx,
            agent1_coarse_tokens, agent1_coarse_tidx, agent1_fine_tokens, agent1_fine_tidx,
            agent2_coarse_tokens, agent2_coarse_tidx, agent2_fine_tokens, agent2_fine_tidx,
        )
        a1_c, a1_ct, a1_f, a1_ft = a1_c.to(device), a1_ct.to(device), a1_f.to(device), a1_ft.to(device)
        a2_c, a2_ct, a2_f, a2_ft = a2_c.to(device), a2_ct.to(device), a2_f.to(device), a2_ft.to(device)
        B = a1_c.size(0)
        if instructions is None:
            instructions = ["follow the person"] * B

        has_bbox_prior = bbox_feat is not None or agent1_bbox_feat is not None or agent2_bbox_feat is not None
        if bbox_feat is not None:
            bbox = self._normalize_bbox(bbox_feat, B, device)
        else:
            b1 = torch.zeros(B, 4, dtype=torch.float32, device=device) if agent1_bbox_feat is None else agent1_bbox_feat.to(device)
            b2 = torch.zeros(B, 4, dtype=torch.float32, device=device) if agent2_bbox_feat is None else agent2_bbox_feat.to(device)
            bbox = torch.stack([b1, b2], dim=1).float().clamp(0.0, 1.0)

        if yaw_hist is not None and yaw_hist.dim() == 3:
            agent1_yaw_hist, agent2_yaw_hist = yaw_hist[:, 0], yaw_hist[:, 1]
        if yaw_curr is not None and yaw_curr.dim() == 3:
            agent1_yaw_curr, agent2_yaw_curr = yaw_curr[:, 0], yaw_curr[:, 1]

        a1_seq, _ = self._encode_agent_stream(
            a1_c, a1_ct, a1_f, a1_ft, bbox[:, 0], 0,
            yaw_hist=agent1_yaw_hist, yaw_curr=agent1_yaw_curr,
        )
        a2_seq, _ = self._encode_agent_stream(
            a2_c, a2_ct, a2_f, a2_ft, bbox[:, 1], 1,
            yaw_hist=agent2_yaw_hist, yaw_curr=agent2_yaw_curr,
        )

        if joint_instructions is None and agent1_instructions is None and agent2_instructions is None:
            drone_instructions = self._role_instructions(instructions, "aerial drone", "ground robot dog")
            dog_instructions = self._role_instructions(instructions, "ground robot dog", "aerial drone")
            txt_emb_drone, txt_mask_drone = self._embed_text(drone_instructions, device)
            txt_emb_dog, txt_mask_dog = self._embed_text(dog_instructions, device)
        else:
            joint_instructions = self._ensure_text_batch(joint_instructions, instructions, B)
            agent1_instructions = self._ensure_text_batch(
                agent1_instructions,
                [self._default_agent_role_text(0)] * B,
                B,
            )
            agent2_instructions = self._ensure_text_batch(
                agent2_instructions,
                [self._default_agent_role_text(1)] * B,
                B,
            )
            joint_emb, joint_mask = self._embed_text(joint_instructions, device)
            agent1_txt_emb, agent1_txt_mask = self._embed_text(agent1_instructions, device)
            agent2_txt_emb, agent2_txt_mask = self._embed_text(agent2_instructions, device)
            txt_emb_drone = torch.cat([joint_emb, agent1_txt_emb], dim=1)
            txt_mask_drone = torch.cat([joint_mask, agent1_txt_mask], dim=1)
            txt_emb_dog = torch.cat([joint_emb, agent2_txt_emb], dim=1)
            txt_mask_dog = torch.cat([joint_mask, agent2_txt_mask], dim=1)
        act1 = self.tvi.make_query_token(self.act_token_1.expand(B, 1, -1), KIND_ACT, 0, 0)
        act2 = self.tvi.make_query_token(self.act_token_2.expand(B, 1, -1), KIND_ACT, 1, 1)

        h_act1 = self._forward_one_agent_context(txt_emb_drone, txt_mask_drone, a1_seq, act1)
        h_act2 = self._forward_one_agent_context(txt_emb_dog, txt_mask_dog, a2_seq, act2)

        grounding = self._base_grounding_stub(B, bbox, has_bbox_prior, h_act1.dtype, device)
        agent1_waypoints = self.planner_agent1(h_act1) * self.alpha_task
        agent2_waypoints = self.planner_agent2(h_act2) * self.alpha_task

        if not return_dict:
            return agent1_waypoints, agent2_waypoints, grounding
        return {
            "agent1_waypoints": agent1_waypoints,
            "agent2_waypoints": agent2_waypoints,
            "waypoints": torch.stack([agent1_waypoints, agent2_waypoints], dim=1),
            "refined_bbox": grounding["refined_bbox"],
            "absolute_bbox": grounding["absolute_bbox"],
            "bbox_prior_mask": grounding["bbox_prior_mask"],
            "relative_pose": grounding["relative_pose"],
            "spatial_emb": grounding["spatial_emb"],
            "visible_logits": grounding["visible_logits"],
            "visible_score": grounding["visible_score"],
            "token_logits": grounding["token_logits"],
        }


OpenTrackVLAMultiAgent = MultiAgentOpenTrackVLA
OpenTrackVLAMultiAgentSeparate = MultiAgentSeparateOpenTrackVLA


# ----------------------- 单 Agent离线推理数据集 -----------------------

@dataclass
class DataConfig:
    train_json: str
    n_waypoints: int = 8
    history: int = 31
    default_dt: float = 0.1
    cache_root: Optional[str] = None


class JsonTrackingDataset(Dataset):
    def __init__(self, cfg: DataConfig):
        super().__init__()
        self.cfg = cfg
        p = Path(cfg.train_json)
        candidate = p if p.is_dir() else p.parent
        max_up = 4
        while max_up >= 0 and not (candidate / 'frames').exists():
            if candidate.parent == candidate:
                break
            candidate = candidate.parent
            max_up -= 1
        self.base_root = candidate
        self.cache_root = Path(cfg.cache_root) if cfg.cache_root is not None else (self.base_root / "vision_cache")
        self._online_encoder: Optional[VisionFeatureCacher] = None
        self._bbox_processor = None
        self._bbox_model = None
        self._lazy = False
        self._index: Optional[List[Tuple[str, int]]] = None
        self.examples: Optional[List[Dict[str, Any]]] = None
        if p.is_file() and p.suffix.lower() == '.json':
            data = load_examples_from_path(cfg.train_json)
            assert isinstance(data, list) and len(data) > 0
            self.examples = data
        else:
            files: List[Path] = []
            if p.is_file() and p.suffix.lower() == '.jsonl':
                files = [p]
            elif p.is_dir():
                files = sorted(p.rglob('*.jsonl'))
            if len(files) == 0:
                raise FileNotFoundError(f"No .jsonl files found under: {cfg.train_json}")
            self._lazy = True
            self._index = []
            for fp in files:
                with open(fp, 'rb') as f:
                    pos = 0
                    while True:
                        line = f.readline()
                        if not line:
                            break
                        if line.strip():
                            self._index.append((str(fp), pos))
                        pos += len(line)
            if len(self._index) == 0:
                raise RuntimeError(f"No examples indexed from .jsonl sources under: {cfg.train_json}")
        H_target = int(self.cfg.history)
        self.coarse_frames = H_target

    def __len__(self):
        if self.examples is not None:
            return len(self.examples)
        if self._index is not None:
            return len(self._index)
        return 0

    def _load_tokens(self, path: str) -> torch.Tensor:
        return load_tokens_file(path)

    def _get_online_encoder(self) -> VisionFeatureCacher:
        if self._online_encoder is None:
            from torch.utils.data import get_worker_info
            worker_info = get_worker_info()
            use_cuda = torch.cuda.is_available() and (worker_info is None)
            cfg = VisionCacheConfig(image_size=384, batch_size=8, device=('cuda' if use_cuda else 'cpu'))
            self._online_encoder = VisionFeatureCacher(cfg)
            self._online_encoder.eval()
        return self._online_encoder

    def _get_bbox_detector(self):
        if self._bbox_model is None or self._bbox_processor is None:
            from torch.utils.data import get_worker_info
            worker_info = get_worker_info()
            use_cuda = torch.cuda.is_available() and (worker_info is None)
            device = torch.device('cuda' if use_cuda else 'cpu')
            try:
                from transformers import AutoProcessor, OmDetTurboForObjectDetection
                self._bbox_processor = AutoProcessor.from_pretrained("omlab/omdet-turbo-swin-tiny-hf")
                self._bbox_model = OmDetTurboForObjectDetection.from_pretrained("omlab/omdet-turbo-swin-tiny-hf").to(device).eval()
            except Exception as _e:
                print(f"[bbox] failed to init omdet-turbo: {_e}")
                self._bbox_processor = None
                self._bbox_model = None
        return self._bbox_processor, self._bbox_model

    @torch.inference_mode()
    def _encode_image_tokens(self, img_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        enc = self._get_online_encoder()
        pil = Image.open(str(img_path)).convert('RGB')
        tok_dino, Hp, Wp = enc._encode_dino([pil])
        tok_sigl = enc._encode_siglip([pil], out_hw=(Hp, Wp))
        Vt_cat = torch.cat([tok_dino, tok_sigl], dim=-1)
        Vfine = grid_pool_tokens(Vt_cat, Hp, Wp, out_tokens=64)
        Vcoarse = grid_pool_tokens(Vt_cat, Hp, Wp, out_tokens=4)# 设置token数量为4，保持全局信息
        return Vcoarse[0].cpu().float(), Vfine[0].cpu().float()

    def _read_indexed_example(self, idx: int) -> Dict[str, Any]:
        assert self._lazy and self._index is not None
        fp, off = self._index[idx]
        with open(fp, 'rb') as f:
            f.seek(off)
            line = f.readline()
        return json.loads(line.decode('utf-8'))

    def get_example(self, idx: int) -> Dict[str, Any]:
        if self._lazy:
            return self._read_indexed_example(idx)
        assert self.examples is not None
        return self.examples[idx]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.get_example(idx)
        H = self.coarse_frames

        curr_path = Path(ex['current'])
        abs_curr_img = curr_path if curr_path.is_absolute() else (self.base_root / curr_path)
        try:
            rel_curr = abs_curr_img.relative_to(self.base_root)
        except ValueError:
            rel_curr = abs_curr_img
        curr_token_dir = self.cache_root / rel_curr.parent
        curr_token_name = rel_curr.stem + "_vfine.pt"
        curr_tok_path = curr_token_dir / curr_token_name
        try:
            fine_tokens = self._load_tokens(str(curr_tok_path))
        except Exception:
            curr_token_dir.mkdir(parents=True, exist_ok=True)
            vc, vf = self._encode_image_tokens(abs_curr_img)
            try:
                torch.save(vf.half(), str(curr_tok_path))
            except Exception:
                pass
            fine_tokens = vf
        fine_tidx = torch.full((fine_tokens.size(0),), fill_value=H, dtype=torch.long)

        imgs_src = ex.get('images', [])
        imgs_trim = imgs_src[-H:]
        missing = H - len(imgs_trim)
        coarse_list, coarse_tidx = [], []
        first_tok: Optional[torch.Tensor] = None
        current_vc: Optional[torch.Tensor] = None
        for t in range(H):
            if t < missing:
                tok = None
            else:
                img_p = imgs_trim[t - missing]
                rp = Path(img_p)
                abs_img = rp if rp.is_absolute() else (self.base_root / rp)
                try:
                    rel_img = abs_img.relative_to(self.base_root)
                except ValueError:
                    rel_img = abs_img
                token_dir = self.cache_root / rel_img.parent
                token_name = rel_img.stem + "_vcoarse.pt"
                tok_path = token_dir / token_name
                try:
                    tok = self._load_tokens(str(tok_path))
                except Exception:
                    token_dir.mkdir(parents=True, exist_ok=True)
                    vc, vf = self._encode_image_tokens(abs_img)
                    try:
                        torch.save(vc.half(), str(tok_path))
                    except Exception:
                        pass
                    tok = vc
                if first_tok is None:
                    first_tok = tok
            if tok is None:
                if first_tok is not None:
                    tok = first_tok
                else:
                    try:
                        if current_vc is None:
                            cur_coarse_name = rel_curr.stem + "_vcoarse.pt"
                            cur_coarse_path = curr_token_dir / cur_coarse_name
                            try:
                                current_vc = self._load_tokens(str(cur_coarse_path))
                            except Exception:
                                vc_tmp, _ = self._encode_image_tokens(abs_curr_img)
                                current_vc = vc_tmp
                        tok = current_vc
                    except Exception:
                        tok = torch.zeros(4, fine_tokens.size(1), dtype=torch.float32)
            coarse_list.append(tok)
            coarse_tidx.append(torch.full((tok.size(0),), fill_value=t, dtype=torch.long))
        coarse_tokens = torch.cat(coarse_list, dim=0)
        coarse_tidx   = torch.cat(coarse_tidx, dim=0)

        yaw_hist = torch.tensor(ex.get('yaw_hist', [0.0]*H), dtype=torch.float32)
        yaw_curr = torch.tensor(ex.get('yaw_curr', 0.0), dtype=torch.float32).view(1)

        if 'waypoints' in ex:
            wp = torch.tensor(ex['waypoints'], dtype=torch.float32)
        else:
            assert 'actions' in ex, "JSON needs either 'waypoints' or 'actions'"
            dt = float(ex.get('dt', self.cfg.default_dt))
            traj = integrate_actions_to_waypoints(np.asarray(ex['actions'], dtype=np.float32), self.cfg.n_waypoints, dt)
            wp = torch.from_numpy(traj)

        if 'valid_mask' in ex:
            vm = torch.tensor(ex['valid_mask'], dtype=torch.bool)
        elif 'valid_idx' in ex:
            vm = torch.zeros(self.cfg.n_waypoints, dtype=torch.bool)
            vm[torch.tensor(ex['valid_idx'], dtype=torch.long)] = True
        else:
            vm = torch.ones(self.cfg.n_waypoints, dtype=torch.bool)

        item: Dict[str, Any] = {
            'coarse_tokens': coarse_tokens,
            'coarse_tidx':   coarse_tidx,
            'fine_tokens':   fine_tokens,
            'fine_tidx':     fine_tidx,
            'yaw_hist':      yaw_hist,
            'yaw_curr':      yaw_curr,
            'waypoints':     wp,
            'valid_mask':    vm,
            'instruction':   ex.get('instruction', 'follow the person'),
            'current_path':  str(abs_curr_img),
        }
        if 'bbox' in ex:
            bb = np.asarray(ex['bbox'], dtype=np.float32)
            if bb.size == 4:
                x0,y0,x1,y1 = bb.tolist()
                w = max(1e-6, x1 - x0); h = max(1e-6, y1 - y0)
                cx = x0 + 0.5*w; cy = y0 + 0.5*h
                try:
                    with Image.open(str(abs_curr_img)) as _im:
                        W,H = _im.size
                    if max(cx,cy,w,h) > 1.5:
                        cx /= W; cy /= H; w /= W; h /= H
                except Exception:
                    pass
                item['bbox_feat'] = torch.tensor([cx, cy, w, h], dtype=torch.float32)
        elif self.cfg.use_bbox_token:
            try:
                proc, det = self._get_bbox_detector()
                if proc is not None and det is not None:
                    with Image.open(str(abs_curr_img)).convert('RGB') as _im:
                        H, W = _im.height, _im.width
                        labels = ["person"]
                        inputs = proc(_im, text=labels, return_tensors='pt')
                        dev = next(det.parameters()).device
                        inputs = {k: v.to(dev) for k, v in inputs.items()}
                        with torch.inference_mode():
                            outputs = det(**inputs)
                        results = proc.post_process_grounded_object_detection(outputs,
                                                                             target_sizes=[(H, W)],
                                                                             text_labels=labels,
                                                                             threshold=0.25,
                                                                             nms_threshold=0.3)
                        res = results[0]
                        boxes = res.get('boxes', None)
                        scores = res.get('scores', None)
                        tlabels = res.get('text_labels', [])
                        if boxes is not None and scores is not None and len(boxes) > 0:
                            best = -1; best_s = -1.0
                            for i in range(len(boxes)):
                                lbl = tlabels[i] if i < len(tlabels) else ''
                                if lbl in ('person','human') and scores[i].item() > best_s:
                                    best_s = scores[i].item(); best = i
                            if best < 0:
                                best = int(torch.argmax(scores).item())
                            x0,y0,x1,y1 = boxes[best].tolist()
                            w = max(1e-6, x1 - x0); h = max(1e-6, y1 - y0)
                            cx = x0 + 0.5*w; cy = y0 + 0.5*h
                            cx /= W; cy /= H; w /= W; h /= H
                            item['bbox_feat'] = torch.tensor([cx, cy, w, h], dtype=torch.float32)
            except Exception:
                pass
        return item


# ----------------------- 单 Agent batch 组装 -----------------------

def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    instr = [b['instruction'] for b in batch]
    return {
        'coarse_tokens': torch.stack([b['coarse_tokens'] for b in batch], dim=0),
        'coarse_tidx':   torch.stack([b['coarse_tidx']   for b in batch], dim=0),
        'fine_tokens':   torch.stack([b['fine_tokens']   for b in batch], dim=0),
        'fine_tidx':     torch.stack([b['fine_tidx']     for b in batch], dim=0),
        'yaw_hist':      torch.stack([b['yaw_hist']      for b in batch], dim=0),
        'yaw_curr':      torch.stack([b['yaw_curr']      for b in batch], dim=0),
        'waypoints':     torch.stack([b['waypoints']     for b in batch], dim=0),
        'valid_mask':    torch.stack([b['valid_mask']    for b in batch], dim=0),
        'instruction':   instr,
        'current_path':  [b['current_path'] for b in batch],
        'bbox_feat':     torch.stack([b['bbox_feat'] if 'bbox_feat' in b else torch.zeros(4, dtype=torch.float32) for b in batch], dim=0),
    }


# ----------------------- 单 Agent离线推理流程 -----------------------

@torch.inference_mode()
def _run_inference(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = cfg.infer_ckpt
    try:
        if ckpt_path is None:
            from glob import glob
            pts = sorted(glob(os.path.join(cfg.out_dir, 'model_epoch*.pt')), key=lambda p: os.path.getmtime(p))
            ckpt_path = pts[-1] if pts else None
    except Exception:
        ckpt_path = None
    if ckpt_path is None or (not os.path.exists(ckpt_path)):
        raise FileNotFoundError(f"No checkpoint found for inference (looked at --infer_ckpt and {cfg.out_dir})")

    obj = torch.load(ckpt_path, map_location=device)
    ck = obj if isinstance(obj, dict) else {}
    ck_cfg = ck.get('config', {})

    n_waypoints = int(ck_cfg.get('n_waypoints', getattr(cfg, 'n_waypoints', 8)))
    use_angle_tvi = bool(ck_cfg.get('use_angle_tvi', False))
    no_tanh_actions = bool(ck_cfg.get('no_tanh_actions', True))
    vision_feat_dim = int(ck_cfg.get('vision_feat_dim', getattr(cfg, 'vision_feat_dim', 1536)))
    alpha_xy        = ck_cfg.get('alpha_xy', getattr(cfg, 'alpha_xy', None))

    model = OpenTrackVLA(
        ModelConfig(
            n_waypoints=n_waypoints,
            beta_nav=float(ck_cfg.get('beta_nav', 10.0)),
            use_angle_tvi=use_angle_tvi,
            use_tanh_actions=(not no_tanh_actions),
            alpha_xy=alpha_xy
        ),
        vision_feat_dim=vision_feat_dim,
    ).to(device).eval()
    msd = ck.get('model_state', None)
    if msd:
        model.load_state_dict(msd, strict=False)

    if cfg.infer_json is None:
        raise ValueError('--infer_json is required for inference')
    vds = JsonTrackingDataset(DataConfig(train_json=cfg.infer_json, n_waypoints=n_waypoints, history=getattr(cfg, 'history', 31), cache_root=getattr(cfg, 'cache_root', None)))
    vdl = DataLoader(vds, batch_size=getattr(cfg, 'batch_size', 2), shuffle=False, num_workers=min(2, getattr(cfg, 'num_workers', 4)), pin_memory=True, collate_fn=collate_batch)

    os.makedirs(cfg.infer_out, exist_ok=True)
    vis_dir = os.path.join(cfg.infer_out, 'vis')
    npz_dir = os.path.join(cfg.infer_out, 'npz')
    if cfg.infer_vis:
        os.makedirs(vis_dir, exist_ok=True)
    if cfg.infer_save_npz:
        os.makedirs(npz_dir, exist_ok=True)

    batches_limit = max(0, int(getattr(cfg, 'infer_batches', 0)))
    bdone = 0
    for bidx, batch in enumerate(vdl):
        coarse_tokens = batch['coarse_tokens'].to(device)
        coarse_tidx   = batch['coarse_tidx'].to(device)
        fine_tokens   = batch['fine_tokens'].to(device)
        fine_tidx     = batch['fine_tidx'].to(device)
        yaw_hist      = batch['yaw_hist'].to(device)
        yaw_curr      = batch['yaw_curr'].to(device)
        instr         = batch['instruction']
        bbox_feat     = batch.get('bbox_feat', None)

        pred = model(
            coarse_tokens, coarse_tidx,
            fine_tokens, fine_tidx,
            instr,
            yaw_hist=yaw_hist if use_angle_tvi else None,
            yaw_curr=yaw_curr if use_angle_tvi else None
        )

        if cfg.infer_save_npz:
            try:
                with torch.no_grad():
                    pred_np = pred.detach().float().cpu().numpy()
                Bcur = pred_np.shape[0]
                for bi in range(Bcur):
                    fpath = os.path.join(npz_dir, f"b{bidx:06d}_i{bi:03d}.npz")
                    np.savez_compressed(
                        fpath,
                        pred=pred_np[bi],
                        instruction=instr[bi],
                        current_path=(batch.get('current_path', [''])[bi] if isinstance(batch.get('current_path', []), list) else '')
                    )
            except Exception:
                pass

        if cfg.infer_vis:
            try:
                with torch.no_grad():
                    pred_draw = pred.detach().float()
                    try:
                        model_inspect = model
                        alpha_vec = getattr(model_inspect, 'alpha_task', None)
                        if alpha_vec is not None and pred_draw.size(-1) >= 2 and alpha_vec.size(-1) >= 2:
                            max_xy = pred_draw[..., 0:2].abs().max().item()
                            if max_xy <= 1.5:
                                ax = alpha_vec[..., 0:2].clamp_min(1e-6).to(pred_draw.device, pred_draw.dtype)
                                pred_draw = pred_draw.clone()
                                pred_draw[..., 0:2] = pred_draw[..., 0:2] * ax
                    except Exception:
                        pass
                    pred_np = pred_draw.cpu().numpy()
                cur_paths = batch.get('current_path', [])
                Bcur = pred_np.shape[0]
                for bi in range(min(Bcur, 4)):
                    cur_path = cur_paths[bi] if isinstance(cur_paths, list) and bi < len(cur_paths) else None
                    if cur_path is None or (not os.path.exists(cur_path)):
                        continue
                    pil_img = Image.open(cur_path).convert('RGB')
                    draw = ImageDraw.Draw(pil_img)
                    w, h = pil_img.size
                    base_x = w // 2
                    base_y = int(h * 0.86)
                    def to_pxxy(traj):
                        pts = []
                        for i in range(min(traj.shape[0], 64)):
                            x, y = float(traj[i, 0]), float(traj[i, 1])
                            px = base_x - int(y * 120)
                            py = base_y - int(x * 120)
                            pts.append((px, py))
                        return pts
                    pts_pred = to_pxxy(pred_np[bi])
                    for i2 in range(1, len(pts_pred)):
                        draw.line([pts_pred[i2-1], pts_pred[i2]], fill=(0, 255, 200), width=6)
                    if pts_pred:
                        r0 = 6
                        sx, sy = pts_pred[0]
                        draw.ellipse([sx-r0, sy-r0, sx+r0, sy+r0], fill=(0,255,0))
                    out_path = os.path.join(vis_dir, f"b{bidx:06d}_i{bi:03d}.jpg")
                    pil_img.save(out_path)
            except Exception:
                pass

        bdone += 1
        if batches_limit and bdone >= batches_limit:
            break

    print(f"[INFER] Done. Outputs under {cfg.infer_out}")


# ----------------------- 命令行参数与程序入口 -----------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--infer_json', type=str, required=True, help='Run inference on this dataset (json/jsonl/dir)')
    ap.add_argument('--infer_ckpt', type=str, default=None, help='Checkpoint to load for inference (defaults to latest in out_dir)')
    ap.add_argument('--out_dir', type=str, default='/data/hdt/ntv_data/ckpt/ckpts_qwen4', help='Directory where checkpoints are stored (for default lookup)')
    ap.add_argument('--infer_out', type=str, default='./infer_out', help='Output directory for inference results')
    ap.add_argument('--infer_batches', type=int, default=0, help='Limit number of batches to run at inference (0 = all)')
    ap.add_argument('--infer_vis', action='store_true', help='Save visualization images during inference')
    ap.add_argument('--infer_save_npz', action='store_true', help='Save npz predictions during inference')
    ap.add_argument('--n_waypoints', type=int, default=8)
    ap.add_argument('--history', type=int, default=31)
    ap.add_argument('--batch_size', type=int, default=2)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--vision_feat_dim', type=int, default=1536)
    ap.add_argument('--cache_root', type=str, default=None)
    ap.add_argument('--alpha_xy', type=float, default=None)
    args = ap.parse_args()
    return args


if __name__ == '__main__':
    cfg = parse_args()
    _run_inference(cfg)
