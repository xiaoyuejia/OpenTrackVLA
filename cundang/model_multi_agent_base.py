#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Clean base multi-agent OpenTrackVLA model.

Compared with the single-agent model, this version only concatenates two
agents' visual-token streams into one LLM context and decodes two ACT tokens.
It deliberately has no grounding head, no bbox token, no visibility head, and
no diffusion planner.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class CrossModalityProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PlannerHead3L(nn.Module):
    """Three-layer MLP from ACT hidden state to future waypoints."""

    def __init__(self, d_model: int, n_waypoints: int, action_dims: int = 3, use_tanh: bool = False):
        super().__init__()
        self.n_waypoints = int(n_waypoints)
        self.action_dims = int(action_dims)
        self.use_tanh = bool(use_tanh)
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, self.n_waypoints * self.action_dims),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        out = self.net(h).view(h.size(0), self.n_waypoints, self.action_dims)
        return torch.tanh(out) if self.use_tanh else out


@dataclass
class BaseMultiAgentModelConfig:
    llm_name: str = "Qwen/Qwen3-0.6B"
    freeze_llm: bool = True
    n_waypoints: int = 10
    action_dims: int = 3
    max_time: int = 512 
    max_views: int = 2
    use_tanh_actions: bool = False # 是否使用tanh激活函数来限制动作输出的范围
    alpha_xy: Optional[float] = 1.0
    text_max_length: int = 128
    insert_time_tokens: bool = True


class BaseMultiAgentOpenTrackVLA(nn.Module):
    """Base two-agent planner with concatenated visual context.

    Input shapes:
    - coarse_tokens: (B, 2, history*4, C)
    - coarse_tidx:   (B, 2, history*4)
    - fine_tokens:   (B, 2, 64, C)
    - fine_tidx:     (B, 2, 64)

    Output:
    - waypoints: (B, 2, n_waypoints, action_dims)
    """

    def __init__(self, cfg: BaseMultiAgentModelConfig, vision_feat_dim: int = 1536):
        super().__init__()
        self.cfg = cfg
        use_modelscope = os.environ.get("TRACKVLA_USE_MODELSCOPE", "0").strip().lower() in {"1", "true", "yes", "on"}
        load_dtype = torch.bfloat16 if torch.cuda.is_available() else None
        load_kwargs = {"dtype": load_dtype} if load_dtype is not None else {}
        rank = os.environ.get("RANK", "0")
        t0 = time.time()
        if use_modelscope:
            try:
                from modelscope import AutoModel as MSAutoModel
                from modelscope import AutoTokenizer as MSAutoTokenizer

                print(f"[LLM][rank {rank}] Loading from ModelScope: {cfg.llm_name}", flush=True)
                self.llm = MSAutoModel.from_pretrained(cfg.llm_name, **load_kwargs)
                print(f"[LLM][rank {rank}] ModelScope model loaded in {time.time() - t0:.1f}s", flush=True)
                self.tokenizer = MSAutoTokenizer.from_pretrained(cfg.llm_name)
                print(f"[LLM][rank {rank}] ModelScope tokenizer loaded in {time.time() - t0:.1f}s", flush=True)
            except Exception as exc:
                print(f"[LLM][rank {rank}] ModelScope failed ({exc}), using HuggingFace/Transformers", flush=True)
                t0 = time.time()
                self.llm = AutoModel.from_pretrained(cfg.llm_name, **load_kwargs)
                print(f"[LLM][rank {rank}] Transformers model loaded in {time.time() - t0:.1f}s", flush=True)
                self.tokenizer = AutoTokenizer.from_pretrained(cfg.llm_name)
                print(f"[LLM][rank {rank}] Transformers tokenizer loaded in {time.time() - t0:.1f}s", flush=True)
        else:
            print(f"[LLM][rank {rank}] Loading from HuggingFace/Transformers: {cfg.llm_name}", flush=True)
            self.llm = AutoModel.from_pretrained(cfg.llm_name, **load_kwargs)
            print(f"[LLM][rank {rank}] Transformers model loaded in {time.time() - t0:.1f}s", flush=True)
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.llm_name)
            print(f"[LLM][rank {rank}] Transformers tokenizer loaded in {time.time() - t0:.1f}s", flush=True)

        t1 = time.time()
        self.llm.requires_grad_(not cfg.freeze_llm)
        print(f"[LLM][rank {rank}] requires_grad set freeze_llm={cfg.freeze_llm} in {time.time() - t1:.1f}s", flush=True)
        self.D = int(self.llm.config.hidden_size)
        self.proj = CrossModalityProjector(vision_feat_dim, self.D)

        self.time_emb = nn.Embedding(cfg.max_time + 1, self.D)
        self.agent_emb = nn.Embedding(2, self.D)
        self.view_emb = nn.Embedding(cfg.max_views, self.D)
        self.kind_emb = nn.Embedding(3, self.D)  # 0=history, 1=current, 2=ACT
        self.act_tokens = nn.Parameter(torch.zeros(2, self.D))
        nn.init.normal_(self.act_tokens, std=0.02)

        self.planner_agent1 = PlannerHead3L(self.D, cfg.n_waypoints, cfg.action_dims, cfg.use_tanh_actions)
        self.planner_agent2 = PlannerHead3L(self.D, cfg.n_waypoints, cfg.action_dims, cfg.use_tanh_actions)

        if cfg.alpha_xy is not None:
            alpha = torch.ones(1, 1, 1, cfg.action_dims, dtype=torch.float32)
            if cfg.action_dims >= 2:
                alpha[..., 0] = float(cfg.alpha_xy)
                alpha[..., 1] = float(cfg.alpha_xy)
        else:
            alpha = torch.ones(1, 1, 1, cfg.action_dims, dtype=torch.float32)
        self.register_buffer("alpha_task", alpha)

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self.cfg, "freeze_llm", False):
            self.llm.eval()
        return self

    @property
    def llm_dtype(self) -> torch.dtype:
        return next(self.llm.parameters()).dtype

    def _embed_text(self, instructions: List[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        tok = self.tokenizer(
            instructions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(self.cfg.text_max_length),
        )
        tok = {key: value.to(device) for key, value in tok.items()}
        emb = self.llm.get_input_embeddings()(tok["input_ids"])
        return emb, tok["attention_mask"]

    def _add_visual_tags(self, x: torch.Tensor, tidx: torch.Tensor, kind_id: int) -> torch.Tensor:
        B, A, N, _D = x.shape
        device = x.device
        tidx = tidx.to(device=device).clamp(min=0, max=int(self.cfg.max_time)).long()
        agent_ids = torch.arange(A, device=device).view(1, A, 1).expand(B, A, N)
        view_ids = agent_ids.clamp(max=self.view_emb.num_embeddings - 1)
        kind_ids = torch.full((B, A, N), int(kind_id), dtype=torch.long, device=device)
        return x + self.time_emb(tidx) + self.agent_emb(agent_ids) + self.view_emb(view_ids) + self.kind_emb(kind_ids)

    def _make_time_marker(self, t_scalar: int, agent_id: int, kind_id: int, device: torch.device) -> torch.Tensor:
        t = int(max(0, min(int(t_scalar), int(self.cfg.max_time))))
        agent = torch.tensor(int(agent_id), dtype=torch.long, device=device)
        view = torch.tensor(min(int(agent_id), self.view_emb.num_embeddings - 1), dtype=torch.long, device=device)
        kind = torch.tensor(int(kind_id), dtype=torch.long, device=device)
        return self.time_emb.weight[t].to(device) + self.agent_emb(agent) + self.view_emb(view) + self.kind_emb(kind)

    def _interleave_time_markers(
        self,
        tokens: torch.Tensor,
        tidx: torch.Tensor,
        agent_id: int,
        kind_id: int,
    ) -> torch.Tensor:
        """Insert one time marker before each contiguous frame-token block.

        Input tokens have already received per-token time/agent/kind tags.
        For history=31, coarse length changes 124 -> 155; current fine changes
        64 -> 65. Disable with cfg.insert_time_tokens=False for ablations.
        """
        if not self.cfg.insert_time_tokens:
            return tokens
        B, N, _D = tokens.shape
        device = tokens.device
        # Keep the grouping logic on CPU. The old implementation called
        # .item() on CUDA tensors inside nested token loops, causing thousands
        # of GPU/CPU synchronizations per forward and making training look
        # frozen. tidx is tiny, so using it as CPU metadata is much cheaper.
        tidx_cpu = tidx.detach().to("cpu").clamp(min=0, max=int(self.cfg.max_time)).long()
        t0 = tidx_cpu[0].tolist()
        spans: List[tuple[int, int]] = []
        i = 0
        while i < N:
            j = i + 1
            while j < N and t0[j] == t0[i]:
                j += 1
            spans.append((i, j))
            i = j

        agent_ids = torch.full((B,), int(agent_id), dtype=torch.long, device=device)
        view_ids = agent_ids.clamp(max=self.view_emb.num_embeddings - 1)
        kind_ids = torch.full((B,), int(kind_id), dtype=torch.long, device=device)
        pieces: List[torch.Tensor] = []
        for start, end in spans:
            tvals = tidx_cpu[:, start].to(device=device, non_blocking=True)
            marker = self.time_emb(tvals) + self.agent_emb(agent_ids) + self.view_emb(view_ids) + self.kind_emb(kind_ids)
            pieces.append(marker.unsqueeze(1))
            pieces.append(tokens[:, start:end])
        return torch.cat(pieces, dim=1)

    def _act_sequence(self, batch_size: int, device: torch.device) -> torch.Tensor:
        agent_ids = torch.arange(2, device=device)
        view_ids = agent_ids.clamp(max=self.view_emb.num_embeddings - 1)
        kind_ids = torch.full((2,), 2, dtype=torch.long, device=device)
        act = self.act_tokens.to(device) + self.agent_emb(agent_ids) + self.view_emb(view_ids) + self.kind_emb(kind_ids)
        return act.unsqueeze(0).expand(batch_size, -1, -1)

    def _debug_forward_stage(self, enabled: bool, rank: str, name: str, t0: float, device: torch.device) -> float:
        if not enabled:
            return t0
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        now = time.time()
        print(f"[FWD][rank {rank}] {name}: {now - t0:.3f}s", flush=True)
        return now

    def forward(
        self,
        coarse_tokens: torch.Tensor,
        coarse_tidx: torch.Tensor,
        fine_tokens: torch.Tensor,
        fine_tidx: torch.Tensor,
        instructions: List[str],
        return_dict: bool = True,
        **_unused: Any,
    ) -> Dict[str, torch.Tensor] | torch.Tensor:
        device = next(self.parameters()).device
        B = coarse_tokens.size(0)
        debug_forward = os.environ.get("TRACKVLA_DEBUG_FORWARD", "0").strip().lower() in {"1", "true", "yes", "on"}
        rank = os.environ.get("RANK", "0")
        t_stage = time.time()
        if debug_forward:
            print(f"[FWD][rank {rank}] start B={B}", flush=True)

        vc = self.proj(coarse_tokens.to(device))
        vf = self.proj(fine_tokens.to(device))
        t_stage = self._debug_forward_stage(debug_forward, rank, "project", t_stage, device)
        vc = self._add_visual_tags(vc, coarse_tidx.to(device), kind_id=0)
        vf = self._add_visual_tags(vf, fine_tidx.to(device), kind_id=1)
        t_stage = self._debug_forward_stage(debug_forward, rank, "visual_tags", t_stage, device)

        # Keep the old single/multi-agent temporal structure: per-token tags first,
        # then a frame-level time marker before each contiguous token block.
        a1_c = self._interleave_time_markers(vc[:, 0], coarse_tidx[:, 0], agent_id=0, kind_id=0)
        a1_f = self._interleave_time_markers(vf[:, 0], fine_tidx[:, 0], agent_id=0, kind_id=1)
        a2_c = self._interleave_time_markers(vc[:, 1], coarse_tidx[:, 1], agent_id=1, kind_id=0)
        a2_f = self._interleave_time_markers(vf[:, 1], fine_tidx[:, 1], agent_id=1, kind_id=1)
        t_stage = self._debug_forward_stage(debug_forward, rank, "interleave_markers", t_stage, device)

        # Keep each agent stream contiguous, then place both ACT tokens at the end.
        agent1_seq = torch.cat([a1_c, a1_f], dim=1)
        agent2_seq = torch.cat([a2_c, a2_f], dim=1)
        act_seq = self._act_sequence(B, device)
        act1 = act_seq[:, 0:1]
        act2 = act_seq[:, 1:2]
        txt_emb, txt_mask = self._embed_text(instructions, device)
        t_stage = self._debug_forward_stage(debug_forward, rank, "text_embed", t_stage, device)

        pieces = [txt_emb, agent1_seq, agent2_seq, act1, act2]
        lengths = [p.size(1) for p in pieces]
        act1_pos = sum(lengths[:3])
        act2_pos = act1_pos + lengths[3]

        visual_len = sum(lengths[1:])
        seq = torch.cat(pieces, dim=1).to(self.llm_dtype)
        attn = torch.cat(
            [
                txt_mask,
                torch.ones(B, visual_len, dtype=torch.long, device=device),
            ],
            dim=1,
        )
        t_stage = self._debug_forward_stage(debug_forward, rank, f"pack seq_len={seq.size(1)}", t_stage, device)

        out = self.llm(inputs_embeds=seq, attention_mask=attn, output_hidden_states=False, use_cache=False)
        t_stage = self._debug_forward_stage(debug_forward, rank, "llm_forward", t_stage, device)
        hidden = out.last_hidden_state.float()
        h_act1 = hidden[:, act1_pos, :]
        h_act2 = hidden[:, act2_pos, :]
        wp1 = self.planner_agent1(h_act1)
        wp2 = self.planner_agent2(h_act2)
        alpha = self.alpha_task.to(device=device, dtype=wp1.dtype)
        wp1 = wp1 * alpha[:, 0]
        wp2 = wp2 * alpha[:, 0]
        waypoints = torch.stack([wp1, wp2], dim=1)
        t_stage = self._debug_forward_stage(debug_forward, rank, "planner", t_stage, device)

        if not return_dict:
            return waypoints
        zeros_bbox = torch.zeros(B, 2, 4, dtype=waypoints.dtype, device=device)
        return {
            "agent1_waypoints": wp1,
            "agent2_waypoints": wp2,
            "waypoints": waypoints,
            "refined_bbox": zeros_bbox,
            "absolute_bbox": zeros_bbox,
            "visible_logits": torch.zeros(B, 2, dtype=waypoints.dtype, device=device),
            "visible_score": torch.full((B, 2), 0.5, dtype=waypoints.dtype, device=device),
        }


def config_to_dict(cfg: BaseMultiAgentModelConfig) -> Dict[str, Any]:
    return dict(cfg.__dict__)
