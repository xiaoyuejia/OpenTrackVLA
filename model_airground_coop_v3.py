#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Three-stream aerial-ground cooperative embodied tracking model.

This module is intentionally independent from :mod:`model_airground_coop`.
It implements three actual information flows with a shared-weight LLM:

1. drone-only observation -> LLM -> drone self action head;
2. robotdog-only observation -> LLM -> dog self action head;
3. both observations + absolute/directed-relative agent poses -> LLM -> two
   cooperative decoders.

Each agent owns one self row containing two task queries (VERIFY then ACT), and
the two agents are packed along the *batch* axis.  The views never share a self
token sequence or attention matrix, so changing one view cannot change the
other view's self trajectory.  VERIFY precedes ACT in the causal sequence: the
action context may use verification evidence, while target-match loss cannot
back-propagate through the later ACT query.  The cooperative pass is separate
and contains both views.

Agent convention: index 0 is the drone and index 1 is the robot dog.  Every
trajectory uses local ``[x, y, yaw]`` waypoints.
"""

from __future__ import annotations

import copy
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from candidate_matching import CandidateTextMatcher


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DRONE = 0
ROBOTDOG = 1
NUM_AGENTS = 2

KIND_HISTORY = 0
KIND_CURRENT = 1
KIND_DETECTION = 2
KIND_ACT = 3
KIND_COOP_ACT = 4
KIND_POSE = 5
KIND_TARGET = 6
KIND_OBSTACLE = 7
KIND_MISSING = 8
KIND_VERIFY = 9

ROUTE_SELF = 0
ROUTE_COOPERATIVE = 1
ROUTE_BELIEF = 2
ROUTE_SEARCH = 3


def candidate_iou_with_prior(
    candidates: torch.Tensor, prior: torch.Tensor
) -> torch.Tensor:
    """Aligned cxcywh IoU for ``(B,A,K,4)`` candidates and ``(B,A,4)`` prior."""
    first=candidates.float(); second=prior.float().unsqueeze(2)
    first_half=first[...,2:].clamp_min(0)*0.5; second_half=second[...,2:].clamp_min(0)*0.5
    low=torch.maximum(first[...,:2]-first_half,second[...,:2]-second_half)
    high=torch.minimum(first[...,:2]+first_half,second[...,:2]+second_half)
    intersection=(high-low).clamp_min(0).prod(dim=-1)
    union=first[...,2:].prod(dim=-1)+second[...,2:].prod(dim=-1)-intersection
    return torch.where(union>0,intersection/union.clamp_min(1.0e-8),torch.zeros_like(union))


class AirGroundVisibilityRouter:
    """Stateful YOLO + LLM target-verification hysteresis for one session.

    The neural model remains stateless.  Evaluation code should create one
    router per scene/session, call :meth:`update` with YOLO detections and the
    model's target-match probabilities, then pass the returned ``visible`` tensor
    to ``route_visibility``.  ``mode`` distinguishes normal self/cooperative
    routing from the both-invisible belief state into a controller-side bounded
    yaw search; search keeps translation at zero and sweeps only +/-30 degrees
    around the confirmed loss heading.
    """

    def __init__(
        self,
        *,
        enter_confidence: float = 0.35,
        exit_confidence: float = 0.20,
        target_match_enter_confidence: float = 0.50,
        target_match_exit_confidence: float = 0.35,
        visible_confirm_frames: int = 2,
        invisible_confirm_frames: int = 2,
        belief_hold_frames: int = 3,
    ):
        if not 0.0 <= exit_confidence <= enter_confidence <= 1.0:
            raise ValueError("Require 0 <= exit_confidence <= enter_confidence <= 1")
        if not (
            0.0
            <= target_match_exit_confidence
            <= target_match_enter_confidence
            <= 1.0
        ):
            raise ValueError(
                "Require 0 <= target_match_exit_confidence <= "
                "target_match_enter_confidence <= 1"
            )
        if min(visible_confirm_frames, invisible_confirm_frames, belief_hold_frames) < 0:
            raise ValueError("Router frame counts must be non-negative")
        self.enter_confidence = float(enter_confidence)
        self.exit_confidence = float(exit_confidence)
        self.target_match_enter_confidence = float(target_match_enter_confidence)
        self.target_match_exit_confidence = float(target_match_exit_confidence)
        self.visible_confirm_frames = max(1, int(visible_confirm_frames))
        self.invisible_confirm_frames = max(1, int(invisible_confirm_frames))
        self.belief_hold_frames = int(belief_hold_frames)
        self.reset()

    def reset(self) -> None:
        self.visible = torch.zeros(NUM_AGENTS, dtype=torch.bool)
        self._visible_count = torch.zeros(NUM_AGENTS, dtype=torch.long)
        self._invisible_count = torch.zeros(NUM_AGENTS, dtype=torch.long)
        self._both_invisible_count = 0

    def update(
        self,
        detection_feat: torch.Tensor,
        target_match_probability: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        detection = torch.as_tensor(detection_feat).detach().float().cpu()
        if detection.shape != (NUM_AGENTS, 6):
            raise ValueError("Visibility router expects detection_feat shape (2,6)")
        target_match = (
            torch.as_tensor(target_match_probability).detach().float().cpu()
        )
        if target_match.shape != (NUM_AGENTS,):
            raise ValueError(
                "Visibility router expects target_match_probability shape (2,)"
            )
        valid = detection[:, 5] > 0.5
        confident_visible = (
            valid
            & (detection[:, 4] >= self.enter_confidence)
            & (target_match >= self.target_match_enter_confidence)
        )
        confident_invisible = (
            ~valid
            | (detection[:, 4] < self.exit_confidence)
            | (target_match < self.target_match_exit_confidence)
        )
        self._visible_count = torch.where(
            confident_visible, self._visible_count + 1, torch.zeros_like(self._visible_count)
        )
        self._invisible_count = torch.where(
            confident_invisible,
            self._invisible_count + 1,
            torch.zeros_like(self._invisible_count),
        )
        self.visible |= self._visible_count >= self.visible_confirm_frames
        self.visible &= ~(self._invisible_count >= self.invisible_confirm_frames)

        mode = torch.full((NUM_AGENTS,), ROUTE_SELF, dtype=torch.long)
        only_one_invisible = ~self.visible & self.visible.flip(0)
        mode[only_one_invisible] = ROUTE_COOPERATIVE
        both_invisible = not bool(self.visible.any())
        if both_invisible:
            self._both_invisible_count += 1
            state = (
                ROUTE_BELIEF
                if self._both_invisible_count <= self.belief_hold_frames
                else ROUTE_SEARCH
            )
            mode.fill_(state)
        else:
            self._both_invisible_count = 0
        return {
            "visible": self.visible.clone(),
            "mode": mode,
            "route_to_cooperative": mode.eq(ROUTE_COOPERATIVE),
            "both_invisible": torch.tensor(both_invisible, dtype=torch.bool),
            "target_match_probability": target_match.clone(),
        }


class CrossModalityProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class PlannerHead3L(nn.Module):
    """Agent-specific trajectory head used only by a self stream."""

    def __init__(self, d_model: int, n_waypoints: int, action_dims: int, use_tanh: bool):
        super().__init__()
        hidden = d_model * 2
        self.n_waypoints = int(n_waypoints)
        self.action_dims = int(action_dims)
        self.use_tanh = bool(use_tanh)
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.n_waypoints * self.action_dims),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        output = self.net(hidden)
        if self.use_tanh:
            output = torch.tanh(output)
        trajectory = output.view(
            hidden.size(0), self.n_waypoints, self.action_dims
        )
        # Waypoint 0 is the local-pose origin, not a future prediction.  The
        # dataset consequently marks it invalid for the waypoint regression
        # loss.  Anchor it explicitly so inference and the kinematic regularizer
        # always start from the actual [x=0, y=0, yaw=0] receiver pose.
        return torch.cat(
            (torch.zeros_like(trajectory[:, :1]), trajectory[:, 1:]), dim=1
        )


class TVIEmbedder(nn.Module):
    """Time/view/agent/kind embeddings shared by all three streams."""

    def __init__(self, d_model: int, max_time: int, num_kinds: int):
        super().__init__()
        self.time_emb = nn.Embedding(max_time, d_model)
        self.view_emb = nn.Embedding(NUM_AGENTS, d_model)
        self.agent_emb = nn.Embedding(NUM_AGENTS, d_model)
        self.kind_emb = nn.Embedding(num_kinds, d_model)
        self.angle_proj = nn.Linear(2, d_model)

    def type_embedding(
        self, kind_id: int, agent_id: int, view_id: int, device: torch.device
    ) -> torch.Tensor:
        kind = torch.tensor(kind_id, dtype=torch.long, device=device)
        agent = torch.tensor(agent_id, dtype=torch.long, device=device)
        view = torch.tensor(view_id, dtype=torch.long, device=device)
        return self.kind_emb(kind) + self.agent_emb(agent) + self.view_emb(view)

    def add_visual(
        self,
        tokens: torch.Tensor,
        time_index: torch.Tensor,
        kind_id: int,
        agent_id: int,
    ) -> torch.Tensor:
        time_index = time_index.to(tokens.device, torch.long).clamp(
            0, self.time_emb.num_embeddings - 1
        )
        type_embedding = self.type_embedding(kind_id, agent_id, agent_id, tokens.device)
        return tokens + self.time_emb(time_index) + type_embedding.view(1, 1, -1)

    def make_query(
        self, base: torch.Tensor, kind_id: int, agent_id: int
    ) -> torch.Tensor:
        embedding = self.type_embedding(kind_id, agent_id, agent_id, base.device)
        return base + embedding.view(1, 1, -1)


class ConditionalJEPAPredictor(nn.Module):
    """Predict missing-view clean tokens conditioned on the cooperative stream."""

    def __init__(
        self,
        llm_dim: int,
        hidden_dim: int,
        max_query_tokens: int,
        decoder_layers: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()
        if hidden_dim <= 0 or max_query_tokens <= 0 or decoder_layers <= 0:
            raise ValueError("JEPA dimensions and decoder_layers must be positive")
        heads = min(int(num_heads), int(hidden_dim))
        while heads > 1 and hidden_dim % heads:
            heads -= 1
        self.max_query_tokens = int(max_query_tokens)
        self.memory_in = nn.Sequential(nn.LayerNorm(llm_dim), nn.Linear(llm_dim, hidden_dim))
        self.query_in = nn.Sequential(nn.LayerNorm(llm_dim), nn.Linear(llm_dim, hidden_dim))
        self.spatial_queries = nn.Parameter(torch.zeros(1, max_query_tokens, hidden_dim))
        nn.init.normal_(self.spatial_queries, std=0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            layer, num_layers=int(decoder_layers), norm=nn.LayerNorm(hidden_dim)
        )
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, llm_dim))

    def forward(self, memory: torch.Tensor, masked_tokens: torch.Tensor) -> torch.Tensor:# memory (B, N, llm_dim)\ masked_tokens (B, M, llm_dim)
        token_count = masked_tokens.size(1)
        if token_count > self.max_query_tokens:
            raise ValueError(
                f"JEPA got {token_count} target queries, max={self.max_query_tokens}"
            )
        query = self.query_in(masked_tokens) + self.spatial_queries[:, :token_count]
        hidden = self.decoder(query, self.memory_in(memory))
        return self.output(hidden) # [B, M, LLM_DIM]


class MultimodalTrajectoryDecoder(nn.Module):
    """Transformer decoder producing K candidate trajectories and mode logits."""

    def __init__(
        self,
        llm_dim: int,
        hidden_dim: int,
        n_waypoints: int,
        action_dims: int,
        num_modes: int,
        encoder_layers: int,
        decoder_layers: int,
        num_heads: int,
        dropout: float,
        use_tanh: bool,
    ):
        super().__init__()
        if min(hidden_dim, n_waypoints, action_dims, num_modes, decoder_layers) <= 0:
            raise ValueError("Cooperative decoder dimensions must be positive")
        heads = min(int(num_heads), int(hidden_dim))
        while heads > 1 and hidden_dim % heads:
            heads -= 1
        self.n_waypoints = int(n_waypoints)
        self.action_dims = int(action_dims)
        self.num_modes = int(num_modes)
        self.use_tanh = bool(use_tanh)
        self.memory_in = nn.Sequential(nn.LayerNorm(llm_dim), nn.Linear(llm_dim, hidden_dim))
        self.context_in = nn.Sequential(nn.LayerNorm(llm_dim), nn.Linear(llm_dim, hidden_dim))
        self.mode_queries = nn.Parameter(torch.zeros(1, num_modes, 1, hidden_dim))
        self.waypoint_queries = nn.Parameter(torch.zeros(1, 1, n_waypoints, hidden_dim))
        nn.init.normal_(self.mode_queries, std=0.02)
        nn.init.normal_(self.waypoint_queries, std=0.02)

        if int(encoder_layers) > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.memory_encoder: nn.Module = nn.TransformerEncoder(
                encoder_layer,
                num_layers=int(encoder_layers),
                norm=nn.LayerNorm(hidden_dim),
                enable_nested_tensor=False,
            )
        else:
            self.memory_encoder = nn.Identity()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=int(decoder_layers),
            norm=nn.LayerNorm(hidden_dim),
        )
        self.action_out = nn.Linear(hidden_dim, action_dims)
        self.mode_out = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )

    def forward(
        self,
        memory: torch.Tensor,
        context: torch.Tensor,
        memory_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = memory.size(0)
        memory_hidden = self.memory_encoder(self.memory_in(memory))
        context_hidden = self.context_in(context).view(batch_size, 1, 1, -1)
        queries = self.mode_queries + self.waypoint_queries + context_hidden
        queries = queries.expand(batch_size, -1, -1, -1)
        decoded = self.decoder(
            queries.reshape(batch_size, self.num_modes * self.n_waypoints, -1),
            memory_hidden,
            memory_key_padding_mask=memory_padding_mask,
        ).view(batch_size, self.num_modes, self.n_waypoints, -1)
        trajectories = self.action_out(decoded)
        if self.use_tanh:
            trajectories = torch.tanh(trajectories)
        # As in the two self planners, token 0 is the fixed local origin.  It is
        # excluded from the clean waypoint loss, so leaving it unconstrained
        # would make the first executable segment depend on an arbitrary value.
        trajectories = torch.cat(
            (
                torch.zeros_like(trajectories[..., :1, :]),
                trajectories[..., 1:, :],
            ),
            dim=-2,
        )
        mode_logits = self.mode_out(decoded.mean(dim=2)).squeeze(-1)
        return trajectories, mode_logits


@dataclass
class AgentStreamEncoding:
    self_stream: torch.Tensor
    self_mask: torch.Tensor
    cooperative_stream: torch.Tensor
    cooperative_mask: torch.Tensor
    cooperative_fine: torch.Tensor
    teacher_fine: torch.Tensor
    target_mask: torch.Tensor
    obstacle_tokens: torch.Tensor
    effective_detection: torch.Tensor
    candidate_valid: torch.Tensor


@dataclass
class AirGroundCoopV3ModelConfig:
    llm_name: str = "Qwen/Qwen3-0.6B"
    freeze_llm: bool = True
    n_waypoints: int = 10
    action_dims: int = 3
    num_modes: int = 4
    max_time: int = 4096
    num_kinds: int = 10
    text_max_length: int = 128
    insert_time_tokens: bool = True
    use_angle_tvi: bool = False
    use_agent_text_markers: bool = True
    use_tanh_actions: bool = False
    alpha_xy: Optional[float] = 1.0
    perception_grid_size: int = 8
    drone_mask_expand_ratio: float = 3.0
    dog_mask_expand_ratio: float = 3.0
    coop_hidden_dim: int = 512
    coop_encoder_layers: int = 1
    coop_decoder_layers: int = 3
    coop_num_heads: int = 8
    coop_dropout: float = 0.0
    jepa_hidden_dim: int = 512
    jepa_decoder_layers: int = 3
    jepa_num_heads: int = 8
    jepa_dropout: float = 0.0
    jepa_momentum: float = 0.996
    detection_confidence_threshold: float = 0.25
    target_match_confidence_threshold: float = 0.50
    candidate_temporal_iou_weight: float = 2.0
    hard_visibility_routing: bool = True
    pose_position_scale_m: float = 20.0
    drone_target_verification_prompt: str = (
        "Compare all aerial-view person candidates with the target rule and visual history. "
        "For an appearance rule, select the matching person. For an initial-target rule, "
        "remember the first observed target's visual identity and keep matching it. "
        "Score every candidate independently; accept only a sufficiently confident best match. "
        "ACT must plan for the accepted target."
    )
    dog_target_verification_prompt: str = (
        "Compare all ground-view person candidates with the target rule and visual history. "
        "For an appearance rule, select the matching person. For an initial-target rule, "
        "remember the first observed target's visual identity and keep matching it. "
        "Score every candidate independently; accept only a sufficiently confident best match. "
        "ACT must plan for the accepted target."
    )


class AirGroundCooperativeVLAV3(nn.Module):
    """Shared-weight LLM with two isolated self flows and one joint flow."""

    def __init__(self, cfg: AirGroundCoopV3ModelConfig, vision_feat_dim: int):
        super().__init__()
        if cfg.action_dims != 3:
            raise ValueError("AirGroundCooperativeVLAV3 expects [x, y, yaw] actions")
        if cfg.num_kinds <= KIND_VERIFY:
            raise ValueError(f"num_kinds must be greater than {KIND_VERIFY}")
        if not 0.0 <= cfg.jepa_momentum < 1.0:
            raise ValueError("jepa_momentum must be in [0,1)")
        if not 0.0 <= cfg.detection_confidence_threshold <= 1.0:
            raise ValueError("detection_confidence_threshold must be in [0,1]")
        if not 0.0 <= cfg.target_match_confidence_threshold <= 1.0:
            raise ValueError("target_match_confidence_threshold must be in [0,1]")
        if cfg.pose_position_scale_m <= 0.0:
            raise ValueError("pose_position_scale_m must be positive")
        if not cfg.drone_target_verification_prompt.strip():
            raise ValueError("drone_target_verification_prompt must not be empty")
        if not cfg.dog_target_verification_prompt.strip():
            raise ValueError("dog_target_verification_prompt must not be empty")
        self.cfg = cfg
        rank = int(os.environ.get("RANK", "0"))
        print(f"[MODEL][rank {rank}] variant=airground_three_stream_v3", flush=True)
        started = time.time()
        load_kwargs = {"dtype": torch.bfloat16} if torch.cuda.is_available() else {}
        self.llm = AutoModel.from_pretrained(cfg.llm_name, **load_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.llm_name)
        print(
            f"[MODEL][rank {rank}] LLM/tokenizer loaded in {time.time() - started:.1f}s",
            flush=True,
        )
        self.llm.requires_grad_(not cfg.freeze_llm)
        self.llm_dtype = next(self.llm.parameters()).dtype
        self.D = int(self.llm.config.hidden_size)

        self.proj = CrossModalityProjector(vision_feat_dim, self.D)
        self.teacher_proj = copy.deepcopy(self.proj)
        self.teacher_proj.requires_grad_(False)
        self.teacher_proj.eval()
        self.tvi = TVIEmbedder(self.D, cfg.max_time, cfg.num_kinds)
        self.perception_grid_proj = nn.Sequential(
            nn.LayerNorm(4), nn.Linear(4, self.D), nn.GELU(), nn.Linear(self.D, self.D)
        )
        self.obstacle_grid_proj = nn.Sequential(
            nn.LayerNorm(4), nn.Linear(4, self.D), nn.GELU(), nn.Linear(self.D, self.D)
        )
        self.detection_proj = nn.Sequential(
            nn.LayerNorm(6), nn.Linear(6, self.D), nn.GELU(), nn.Linear(self.D, self.D)
        )
        self.candidate_roi_proj = nn.Sequential(
            nn.LayerNorm(self.D),
            nn.Linear(self.D, self.D),
            nn.GELU(),
            nn.Linear(self.D, self.D),
        )
        # Positions have a fixed metric scale before projection.  Avoid a
        # per-token LayerNorm here: normalizing each agent pose independently
        # would discard part of the shared-frame translation relationship.
        self.agent_pose_proj = nn.Sequential(
            nn.Linear(4, self.D), nn.GELU(), nn.Linear(self.D, self.D)
        )
        # For each receiver, explicitly encode the other agent in that
        # receiver's local frame.  The frozen LLM no longer has to discover
        # subtraction and SE(2) rotation from two absolute pose tokens alone.
        # No per-token LayerNorm is used because metric magnitude is evidence.
        self.relative_pose_proj = nn.Sequential(
            nn.Linear(5, self.D), nn.GELU(), nn.Linear(self.D, self.D)
        )
        grid_tokens = cfg.perception_grid_size ** 2
        self.obstacle_position = nn.Parameter(torch.zeros(1, grid_tokens, self.D))
        nn.init.normal_(self.obstacle_position, std=0.02)
        self.masked_visual_tokens = nn.Parameter(torch.zeros(NUM_AGENTS, 1, self.D))
        self.self_act_tokens = nn.Parameter(torch.zeros(NUM_AGENTS, 1, self.D))
        self.self_verify_tokens = nn.Parameter(torch.zeros(NUM_AGENTS, 1, self.D))
        self.coop_act_tokens = nn.Parameter(torch.zeros(NUM_AGENTS, 1, self.D))
        nn.init.normal_(self.masked_visual_tokens, std=0.02)
        nn.init.normal_(self.self_act_tokens, std=0.02)
        nn.init.normal_(self.self_verify_tokens, std=0.02)
        nn.init.normal_(self.coop_act_tokens, std=0.02)

        self.self_planners = nn.ModuleList(
            [
                PlannerHead3L(self.D, cfg.n_waypoints, cfg.action_dims, cfg.use_tanh_actions)
                for _ in range(NUM_AGENTS)
            ]
        )
        self.coop_decoders = nn.ModuleList(
            [
                MultimodalTrajectoryDecoder(
                    llm_dim=self.D,
                    hidden_dim=cfg.coop_hidden_dim,
                    n_waypoints=cfg.n_waypoints,
                    action_dims=cfg.action_dims,
                    num_modes=cfg.num_modes,
                    encoder_layers=cfg.coop_encoder_layers,
                    decoder_layers=cfg.coop_decoder_layers,
                    num_heads=cfg.coop_num_heads,
                    dropout=cfg.coop_dropout,
                    use_tanh=cfg.use_tanh_actions,
                )
                for _ in range(NUM_AGENTS)
            ]
        )
        self.jepa_predictors = nn.ModuleList(
            [
                ConditionalJEPAPredictor(
                    llm_dim=self.D,
                    hidden_dim=cfg.jepa_hidden_dim,
                    max_query_tokens=grid_tokens,
                    decoder_layers=cfg.jepa_decoder_layers,
                    num_heads=cfg.jepa_num_heads,
                    dropout=cfg.jepa_dropout,
                )
                for _ in range(NUM_AGENTS)
            ]
        )
        self.jepa_pool_scores = nn.ModuleList(
            [nn.Sequential(nn.LayerNorm(self.D), nn.Linear(self.D, 1)) for _ in range(NUM_AGENTS)]
        )
        self.target_belief_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(self.D), nn.Linear(self.D, self.D), nn.GELU(), nn.Linear(self.D, 5)
                )
                for _ in range(NUM_AGENTS)
            ]
        )
        self.uncertainty_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(self.D),
                    nn.Linear(self.D, max(1, self.D // 2)),
                    nn.GELU(),
                    nn.Linear(max(1, self.D // 2), 1),
                )
                for _ in range(NUM_AGENTS)
            ]
        )
        # This is not a second visibility detector.  It verifies whether the
        # person box proposed by YOLO is the tracked target, conditioned on the
        # detection token, current RGB tokens and the agent's visual history.
        self.target_match_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(self.D), nn.Linear(self.D, self.D), nn.GELU(), nn.Linear(self.D, 1)
                )
                for _ in range(NUM_AGENTS)
            ]
        )
        # VERIFY is after every candidate and therefore provides one global,
        # Qwen-contextualized grounding query.  Match it explicitly against all
        # candidate states instead of scoring each candidate independently.
        self.candidate_matcher = CandidateTextMatcher(
            self.D, hidden_dim=max(128, min(512, self.D))
        )
        # Keep the legacy VERIFY head as a shared calibration bias for the
        # eight independent candidate logits. Rejection is threshold-based;
        # null_target_context is a fallback context, not a ninth class.
        self.null_target_context = nn.Parameter(torch.zeros(NUM_AGENTS, 1, self.D))
        nn.init.normal_(self.null_target_context, std=0.02)
        # Zero-initialized residual preserves the pretrained planner at step 0;
        ## while making the selected target an explicit waypoint condition.
        self.target_context_projs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(self.D),
                    nn.Linear(self.D, self.D),
                    nn.GELU(),
                    nn.Linear(self.D, self.D),
                )
                for _ in range(NUM_AGENTS)
            ]
        )
        for projector in self.target_context_projs:
            nn.init.zeros_(projector[-1].weight)
            nn.init.zeros_(projector[-1].bias)

        alpha = torch.ones(1, 1, cfg.action_dims, dtype=torch.float32)
        if cfg.alpha_xy is not None:
            alpha[..., :2] = float(cfg.alpha_xy)
        self.register_buffer("alpha_task", alpha)
        if not cfg.use_angle_tvi:
            self.tvi.angle_proj.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.cfg.freeze_llm:
            self.llm.eval()
        self.teacher_proj.eval()
        return self

    @torch.no_grad()
    def update_jepa_teacher(self) -> None:
        momentum = float(self.cfg.jepa_momentum)
        for teacher, online in zip(self.teacher_proj.parameters(), self.proj.parameters()):
            teacher.data.mul_(momentum).add_(online.data, alpha=1.0 - momentum)
        for teacher_buffer, online_buffer in zip(self.teacher_proj.buffers(), self.proj.buffers()):
            teacher_buffer.copy_(online_buffer)

    def _embed_text(
        self, texts: List[str], device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tokenized = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.text_max_length,
        )
        tokenized = {key: value.to(device) for key, value in tokenized.items()}
        embeddings = self.llm.get_input_embeddings()(tokenized["input_ids"])
        return embeddings, tokenized["attention_mask"]

    @staticmethod
    def _ensure_text_batch(
        texts: Optional[List[str]], fallback: List[str], batch_size: int
    ) -> List[str]:
        texts = fallback if texts is None else texts
        if len(texts) == 1 and batch_size > 1:
            texts = texts * batch_size
        if len(texts) != batch_size:
            raise ValueError(f"Expected {batch_size} texts, got {len(texts)}")
        return texts

    @staticmethod
    def _role_text(agent_id: int) -> str:
        if agent_id == DRONE:
            return "Aerial drone: independently follow the target person."
        if agent_id == ROBOTDOG:
            return "Ground robot dog: independently follow the target person."
        raise ValueError(agent_id)

    def _interleave_time_markers(
        self,
        tokens: torch.Tensor,
        time_index: torch.Tensor,
        kind_id: int,
        agent_id: int,
        yaw_per_frame: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if not self.cfg.insert_time_tokens:
            return tokens
        batch_size, token_count, _ = tokens.shape
        device = tokens.device
        time_cpu = time_index.detach().to("cpu", torch.long)
        first = time_cpu[0].tolist()
        spans: List[Tuple[int, int]] = []
        start = 0
        while start < token_count:
            end = start + 1
            while end < token_count and first[end] == first[start]:
                end += 1
            spans.append((start, end))
            start = end
        type_embedding = self.tvi.type_embedding(kind_id, agent_id, agent_id, device)
        pieces: List[torch.Tensor] = []
        for frame_index, (start, end) in enumerate(spans):
            time_values = time_cpu[:, start].to(device).clamp(
                0, self.tvi.time_emb.num_embeddings - 1
            )
            pieces.append((self.tvi.time_emb(time_values) + type_embedding).unsqueeze(1))
            if self.cfg.use_angle_tvi:
                if yaw_per_frame is None or frame_index >= yaw_per_frame.size(1):
                    theta = torch.zeros(batch_size, device=device)
                else:
                    theta = yaw_per_frame[:, frame_index].to(device, torch.float32)
                theta = (theta + math.pi) % (2.0 * math.pi) - math.pi
                sincos = torch.stack((torch.sin(theta), torch.cos(theta)), dim=-1)
                angle = self.tvi.angle_proj(
                    sincos.to(self.tvi.angle_proj.weight.dtype)
                ) + type_embedding
                pieces.append(angle.unsqueeze(1))
            pieces.append(tokens[:, start:end])
        return torch.cat(pieces, dim=1)

    def _grid_features(
        self, perception_grid: torch.Tensor, fine_token_count: int
    ) -> torch.Tensor:
        if perception_grid.ndim != 4 or perception_grid.size(-1) != 4:
            raise ValueError(
                "perception_grid per agent must have shape (B,H,W,4), got "
                f"{tuple(perception_grid.shape)}"
            )
        side = int(round(math.sqrt(fine_token_count)))
        if side * side != fine_token_count:
            raise ValueError("fine token count must form a square grid")
        grid = perception_grid.permute(0, 3, 1, 2).float()
        if grid.shape[-2:] != (side, side):
            grid = F.interpolate(grid, size=(side, side), mode="bilinear", align_corners=False)
        return grid.permute(0, 2, 3, 1).reshape(grid.size(0), fine_token_count, 4)

    @staticmethod
    def _masked_grid(perception_grid: torch.Tensor, occluded: torch.Tensor) -> torch.Tensor:
        effective = perception_grid.clone()
        if occluded.any():
            selected = torch.zeros_like(effective[occluded])
            selected[..., 0] = 1.0
            effective[occluded] = selected
        return effective

    @staticmethod
    def _grid_without_detector_target(perception_grid: torch.Tensor) -> torch.Tensor:
        """Remove YOLO's target channel before independently verifying YOLO.

        The target mask and candidate box come from the same detector, so the
        target channel is not independent evidence that the proposal is the
        tracked person.  RGB/history plus the candidate box perform that check;
        the grid remains useful for unknown/free/obstacle scene context.
        """

        output = perception_grid.clone()
        target_mass = output[..., 3].clone()
        output[..., 0] = (output[..., 0] + target_mass).clamp_max(1.0)
        output[..., 3] = 0.0
        return output

    def _bbox_grid_mask(
        self,
        bbox: torch.Tensor,
        valid: torch.Tensor,
        token_count: int,
        expand_ratio: float,
    ) -> torch.Tensor:
        side = int(round(math.sqrt(token_count)))
        if side * side != token_count:
            raise ValueError("fine token count must form a square grid")
        centers = (torch.arange(side, device=bbox.device, dtype=bbox.dtype) + 0.5) / side
        yy, xx = torch.meshgrid(centers, centers, indexing="ij")
        cx, cy, width, height = bbox.unbind(dim=-1)
        square_side = torch.maximum(width, height) * float(expand_ratio)
        x1, x2 = cx - square_side * 0.5, cx + square_side * 0.5
        y1, y2 = cy - square_side * 0.5, cy + square_side * 0.5
        mask = (
            (xx.view(1, side, side) >= x1[:, None, None])
            & (xx.view(1, side, side) <= x2[:, None, None])
            & (yy.view(1, side, side) >= y1[:, None, None])
            & (yy.view(1, side, side) <= y2[:, None, None])
        )
        mask = mask.view(bbox.size(0), token_count) & valid.bool().view(-1, 1)
        # Very small valid detections can fall between all grid-cell centres.
        # Keep at least the nearest target token so every synthetic sample has
        # an actual JEPA target and masked visual region.
        empty_valid = valid.bool() & ~mask.any(dim=1)
        if empty_valid.any():
            nearest_x = (cx * side).long().clamp(0, side - 1)
            nearest_y = (cy * side).long().clamp(0, side - 1)
            rows = torch.nonzero(empty_valid, as_tuple=False).flatten()
            mask[rows, nearest_y[rows] * side + nearest_x[rows]] = True
        return mask

    def _candidate_tokens(
        self,
        visual_fine: torch.Tensor,
        candidate_feat: torch.Tensor,
        candidate_valid: torch.Tensor,
        agent_id: int,
    ) -> torch.Tensor:
        """Build K compact ROI+geometry tokens without another vision encode."""
        batch, token_count, _ = visual_fine.shape
        if candidate_feat.ndim != 3 or candidate_feat.shape[0] != batch or candidate_feat.size(-1) != 6:
            raise ValueError("candidate_feat per agent must have shape (B,K,6)")
        top_k = candidate_feat.size(1)
        if candidate_valid.shape != (batch, top_k):
            raise ValueError("candidate_valid per agent must have shape (B,K)")
        boxes = candidate_feat[..., :4].reshape(batch * top_k, 4)
        valid = candidate_valid.reshape(batch * top_k)
        masks = self._bbox_grid_mask(boxes, valid, token_count, expand_ratio=1.0)
        values = (
            visual_fine.float().unsqueeze(1).expand(-1, top_k, -1, -1)
            .reshape(batch * top_k, token_count, -1)
        )
        denom = masks.sum(dim=1, keepdim=True).clamp_min(1).to(values.dtype)
        pooled = (values * masks.unsqueeze(-1).to(values.dtype)).sum(dim=1) / denom
        pooled = pooled.reshape(batch, top_k, -1)
        token = self.candidate_roi_proj(pooled) + self.detection_proj(
            candidate_feat.to(pooled.dtype)
        )
        token = self.tvi.make_query(token, KIND_DETECTION, agent_id)
        return token * candidate_valid.unsqueeze(-1).to(token.dtype)

    def _encode_agent_streams(
        self,
        coarse_tokens: torch.Tensor,
        coarse_tidx: torch.Tensor,
        fine_tokens: torch.Tensor,
        fine_tidx: torch.Tensor,
        clean_detection: torch.Tensor,
        candidate_feat: torch.Tensor,
        candidate_valid: torch.Tensor,
        target_reference_tokens: torch.Tensor,
        target_reference_valid: torch.Tensor,
        clean_grid: torch.Tensor,
        synthetic_receiver: torch.Tensor,
        coarse_missing_mask: torch.Tensor,
        fine_missing_mask: torch.Tensor,
        agent_id: int,
        yaw_hist: Optional[torch.Tensor],
        yaw_curr: Optional[torch.Tensor],
    ) -> AgentStreamEncoding:
        visual_coarse = self.proj(coarse_tokens) # 1×124×1024 输入LLM的projector
        visual_fine = self.proj(fine_tokens) # 1×64×1024
        target_reference = self.proj(target_reference_tokens.unsqueeze(1))
        target_reference = self.tvi.make_query(
            target_reference, KIND_TARGET, agent_id
        )
        target_reference = target_reference * target_reference_valid[:, None, None].to(
            target_reference.dtype
        )
        candidate_tokens = self._candidate_tokens(
            visual_fine, candidate_feat, candidate_valid, agent_id
        )
        clean_grid = self._grid_without_detector_target(clean_grid) # 1×8×8×4 将target区域变成unkonw区域，避免模型作弊
        clean_grid_features = self._grid_features(clean_grid, fine_tokens.size(1)) # 1×64×4 grid和fine_token维度对齐
        if coarse_missing_mask.shape != coarse_tokens.shape[:2]:
            raise ValueError(
                "coarse_missing_mask must have shape (B,Ncoarse), got "
                f"{tuple(coarse_missing_mask.shape)}"
            )
        if fine_missing_mask.shape != fine_tokens.shape[:2]:
            raise ValueError(
                "fine_missing_mask must have shape (B,Nfine), got "
                f"{tuple(fine_missing_mask.shape)}"
            )
        # A fully missing current observation has no trustworthy perception
        # grid.  ROI-only corruption retains background/free/obstacle context.
        full_current_missing = fine_missing_mask.all(dim=1) # 检查是不是所有token都是是true也就是miss遮挡
        effective_grid = self._masked_grid(clean_grid, full_current_missing) # 1×8×8×4 应用遮挡规则后，当前真正有效且允许模型使用的grid
        effective_grid_features = self._grid_features(effective_grid, fine_tokens.size(1)) # 1×64×4 grid和fine_token维度对齐
        clean_fine = visual_fine + self.perception_grid_proj(
            clean_grid_features.to(visual_fine.dtype)
        )
        cooperative_fine = visual_fine + self.perception_grid_proj(
            effective_grid_features.to(visual_fine.dtype)
        ) # 将图像细粒度特征visual_fine和区域的环境属性effective_grid_features相加
        target_mask = fine_missing_mask.bool() # fine token mask
        missing = self.masked_visual_tokens[agent_id].view(1, 1, -1) # 取出当前智能体专用的“视觉信息缺失”可学习 token，并调整形状，供后面替换缺失的视觉特征。
        cooperative_fine = torch.where(
            fine_missing_mask.unsqueeze(-1),
            missing.to(cooperative_fine.dtype),
            cooperative_fine,
        ) # 将mask的部分替换成可学习的missing 
        cooperative_coarse = torch.where(
            coarse_missing_mask.unsqueeze(-1),
            missing.to(visual_coarse.dtype),
            visual_coarse,
        )
        effective_detection = clean_detection.clone()
        effective_detection[synthetic_receiver] = 0.0

        coarse = self.tvi.add_visual(visual_coarse, coarse_tidx, KIND_HISTORY, agent_id) # 增加视频帧id，类型id，agent id
        coop_coarse = self.tvi.add_visual(
            cooperative_coarse, coarse_tidx, KIND_HISTORY, agent_id
        )
        self_fine = self.tvi.add_visual(clean_fine, fine_tidx, KIND_CURRENT, agent_id)
        coop_fine = self.tvi.add_visual(cooperative_fine, fine_tidx, KIND_CURRENT, agent_id)
        coarse_seq = self._interleave_time_markers(
            coarse, coarse_tidx, KIND_HISTORY, agent_id, yaw_hist
        ) # 1×155×1024 在每一帧4个token前面加入时间序列token
        coop_coarse_seq = self._interleave_time_markers(
            coop_coarse, coarse_tidx, KIND_HISTORY, agent_id, yaw_hist
        )
        self_fine_seq = self._interleave_time_markers(
            self_fine, fine_tidx, KIND_CURRENT, agent_id, yaw_curr
        )
        coop_fine_seq = self._interleave_time_markers(
            coop_fine, fine_tidx, KIND_CURRENT, agent_id, yaw_curr
        )
        # All K candidates precede VERIFY and ACT in the same causal LLM row.
        # The LLM therefore performs text-conditioned matching and action
        # reasoning together, without a separate matching pass.
        self_stream = torch.cat(
            (target_reference, coarse_seq, self_fine_seq, candidate_tokens), dim=1
        )
        # The persistent AT reference is safe in both flows: it is established
        # only from the episode-start target and contains no current GT pose.
        cooperative_stream = torch.cat(
            (target_reference, coop_coarse_seq, coop_fine_seq), dim=1
        )
        visual_mask = torch.ones(
            self_stream.size(0),
            coarse_seq.size(1) + self_fine_seq.size(1),
            dtype=torch.long,
            device=self_stream.device,
        )
        reference_mask = target_reference_valid[:, None].to(
            device=self_stream.device, dtype=torch.long
        )
        self_mask = torch.cat(
            (
                reference_mask,
                visual_mask,
                candidate_valid.to(device=self_stream.device, dtype=torch.long),
            ),
            dim=1,
        )
        cooperative_mask = torch.cat(
            (reference_mask, visual_mask), dim=1
        )
        obstacle_tokens = self.obstacle_grid_proj(
            effective_grid_features.to(clean_fine.dtype)
        ) + self.obstacle_position[:, : fine_tokens.size(1)] # grid属性编码+位置编码
        obstacle_tokens = self.tvi.make_query(obstacle_tokens, KIND_OBSTACLE, agent_id) # 加类型编码+agent编码
        with torch.no_grad():
            teacher_fine = self.teacher_proj(fine_tokens) # teacher net
        return AgentStreamEncoding(
            self_stream=self_stream,
            self_mask=self_mask,
            cooperative_stream=cooperative_stream,
            cooperative_mask=cooperative_mask,
            cooperative_fine=cooperative_fine,
            teacher_fine=teacher_fine,
            target_mask=target_mask,
            obstacle_tokens=obstacle_tokens,
            effective_detection=effective_detection,
            candidate_valid=candidate_valid,
        )

    def _run_self_flows(
        self,
        encodings: List[AgentStreamEncoding],
        instructions: List[str],
        agent_instructions: List[Optional[List[str]]],
        device: torch.device,
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
    ]:
        """Run two agent-isolated rows, each with VERIFY then ACT queries.

        Putting both tasks in one row avoids duplicating every visual token.
        Causal order is deliberate: target-match supervision at VERIFY cannot
        reach the later ACT query, while ACT may condition on verification
        evidence already represented in the shared row.
        """

        batch_size = encodings[DRONE].self_stream.size(0)
        texts: List[str] = []
        verification_prompts = (
            self.cfg.drone_target_verification_prompt,
            self.cfg.dog_target_verification_prompt,
        )
        for agent_id in range(NUM_AGENTS):
            fallback = [self._role_text(agent_id)] * batch_size
            action_batch = self._ensure_text_batch(
                agent_instructions[agent_id], fallback, batch_size
            )
            role_prefix = (
                f"{self._role_text(agent_id)} "
                if self.cfg.use_agent_text_markers
                else ""
            )
            verification_prompt = role_prefix + verification_prompts[agent_id]
            texts.extend(
                [
                    f"Tracking task: {action_text} Target rule: {identity_text} "
                    f"Grounding and action rule: {verification_prompt}"
                    for action_text, identity_text in zip(action_batch, instructions)
                ]
            )
        text, text_mask = self._embed_text(texts, device) # text tokenizer
        streams = torch.cat(
            [encoding.self_stream for encoding in encodings], dim=0
        ) # 2×221×1024
        stream_masks = torch.cat(
            [encoding.self_mask for encoding in encodings], dim=0
        ) # 2×221
        act_tokens = torch.cat(
            [
                self.tvi.make_query(
                    self.self_act_tokens[agent_id].view(1, 1, -1).expand(batch_size, -1, -1),
                    KIND_ACT,
                    agent_id,
                )
                for agent_id in range(NUM_AGENTS)
            ],
            dim=0,
        )
        verify_tokens = torch.cat(
            [
                self.tvi.make_query(
                    self.self_verify_tokens[agent_id]
                    .view(1, 1, -1)
                    .expand(batch_size, -1, -1),
                    KIND_VERIFY,
                    agent_id,
                )
                for agent_id in range(NUM_AGENTS)
            ],
            dim=0,
        )
        # VERIFY is earlier than ACT so its loss has no causal path to ACT.
        query_tokens = torch.cat((verify_tokens, act_tokens), dim=1)
        attention = torch.cat(
            (
                text_mask,
                stream_masks,
                torch.ones(
                    batch_size * NUM_AGENTS,
                    2,
                    dtype=torch.long,
                    device=device,
                ),
            ),
            dim=1,
        )
        hidden = self.llm(
            inputs_embeds=torch.cat((text, streams, query_tokens), dim=1).to(
                self.llm_dtype
            ),
            attention_mask=attention,
            output_hidden_states=False,
            use_cache=False,
        ).last_hidden_state.float()
        verification_hidden = hidden[:, -2]
        action_hidden = hidden[:, -1]
        action_contexts = [
            action_hidden[agent_id * batch_size : (agent_id + 1) * batch_size]
            for agent_id in range(NUM_AGENTS)
        ]
        verification_contexts = [
            verification_hidden[agent_id * batch_size : (agent_id + 1) * batch_size]
            for agent_id in range(NUM_AGENTS)
        ]
        candidate_contexts: List[torch.Tensor] = []
        text_length = text.size(1)
        for agent_id, encoding in enumerate(encodings):
            rows = slice(agent_id * batch_size, (agent_id + 1) * batch_size)
            top_k = encoding.candidate_valid.size(1)
            start = text_length + encoding.self_stream.size(1) - top_k
            candidate_contexts.append(hidden[rows, start : start + top_k])
        return action_contexts, verification_contexts, candidate_contexts

    def _run_cooperative_flow(
        self,
        encodings: List[AgentStreamEncoding],
        agent_poses: torch.Tensor,
        joint_instructions: List[str],
        selected_candidate_contexts: List[torch.Tensor],
        selected_candidate_valid: torch.Tensor,
        device: torch.device,
    ) -> Tuple[
        torch.Tensor,
        List[Tuple[int, int]],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        List[torch.Tensor],
        torch.Tensor,
    ]:
        batch_size = agent_poses.size(0)
        text, text_mask = self._embed_text(joint_instructions, device)
        pieces: List[torch.Tensor] = [text]
        masks: List[torch.Tensor] = [text_mask]

        def append(piece: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[int, int]:
            start = sum(part.size(1) for part in pieces)
            pieces.append(piece)
            masks.append(
                torch.ones(batch_size, piece.size(1), dtype=torch.long, device=device)
                if mask is None
                else mask.to(device, torch.long)
            )
            return start, start + piece.size(1)

        stream_spans: List[Tuple[int, int]] = []
        for encoding in encodings:
            stream_spans.append(append(encoding.cooperative_stream, encoding.cooperative_mask))
        selected_candidate_spans: List[Tuple[int, int]] = []
        for agent_id, context in enumerate(selected_candidate_contexts):
            selected_candidate_spans.append(
                append(
                    context.unsqueeze(1),
                    selected_candidate_valid[:, agent_id : agent_id + 1],
                )
            )
        pose_spans: List[Tuple[int, int]] = []
        for agent_id in range(NUM_AGENTS):
            pose_value = agent_poses[:, agent_id].clone()
            pose_value[:, :2] /= float(self.cfg.pose_position_scale_m)
            pose_embedding = self.agent_pose_proj(
                pose_value.to(self.agent_pose_proj[0].weight.dtype)
            )
            pose_token = self.tvi.make_query(
                pose_embedding.unsqueeze(1), KIND_POSE, agent_id
            )
            pose_spans.append(append(pose_token))
        relative_pose = self._directed_relative_pose_features(agent_poses)
        relative_pose_spans: List[Tuple[int, int]] = []
        for receiver_id in range(NUM_AGENTS):
            relative_embedding = self.relative_pose_proj(
                relative_pose[:, receiver_id].to(
                    self.relative_pose_proj[0].weight.dtype
                )
            )
            relative_token = self.tvi.make_query(
                relative_embedding.unsqueeze(1), KIND_POSE, receiver_id
            )
            relative_pose_spans.append(append(relative_token))
        act_spans: List[Tuple[int, int]] = []
        for agent_id in range(NUM_AGENTS):
            token = self.tvi.make_query(
                self.coop_act_tokens[agent_id].view(1, 1, -1).expand(batch_size, -1, -1),
                KIND_COOP_ACT,
                agent_id,
            )
            act_spans.append(append(token))
        hidden = self.llm(
            inputs_embeds=torch.cat(pieces, dim=1).to(self.llm_dtype),
            attention_mask=torch.cat(masks, dim=1),
            output_hidden_states=False,
            use_cache=False,
        ).last_hidden_state.float()
        pose_hidden = torch.stack([hidden[:, span[0]] for span in pose_spans], dim=1)
        relative_pose_hidden = torch.stack(
            [hidden[:, span[0]] for span in relative_pose_spans], dim=1
        )
        contexts = [hidden[:, span[0]] for span in act_spans]
        selected_candidate_hidden = torch.stack(
            [hidden[:, span[0]] for span in selected_candidate_spans], dim=1
        )
        return (
            hidden,
            stream_spans,
            pose_hidden,
            relative_pose_hidden,
            relative_pose,
            contexts,
            selected_candidate_hidden,
        )

    def _directed_relative_pose_features(
        self, agent_poses: torch.Tensor
    ) -> torch.Tensor:
        """Return the other agent pose in each receiver's local frame.

        Output row ``[:, receiver]`` is
        ``[other_forward, other_right, sin(delta_yaw), cos(delta_yaw), distance]``.
        Metric entries are divided by ``pose_position_scale_m`` before projection.
        """

        if agent_poses.ndim != 3 or agent_poses.shape[1:] != (NUM_AGENTS, 4):
            raise ValueError(
                "agent_poses must have shape (B,2,4), got "
                f"{tuple(agent_poses.shape)}"
            )
        positions = agent_poses[..., :2]
        yaw = torch.atan2(agent_poses[..., 2], agent_poses[..., 3])
        features: List[torch.Tensor] = []
        scale = float(self.cfg.pose_position_scale_m)
        for receiver_id in range(NUM_AGENTS):
            source_id = 1 - receiver_id
            delta_world = positions[:, source_id] - positions[:, receiver_id]
            receiver_yaw = yaw[:, receiver_id]
            cosine = torch.cos(receiver_yaw)
            sine = torch.sin(receiver_yaw)
            forward = cosine * delta_world[:, 0] + sine * delta_world[:, 1]
            right = -sine * delta_world[:, 0] + cosine * delta_world[:, 1]
            delta_yaw = yaw[:, source_id] - receiver_yaw
            distance = torch.linalg.vector_norm(delta_world, dim=-1)
            features.append(
                torch.stack(
                    (
                        forward / scale,
                        right / scale,
                        torch.sin(delta_yaw),
                        torch.cos(delta_yaw),
                        distance / scale,
                    ),
                    dim=-1,
                )
            )
        return torch.stack(features, dim=1)

    def forward(
        self,
        coarse_tokens: torch.Tensor,
        coarse_tidx: torch.Tensor,
        fine_tokens: torch.Tensor,
        fine_tidx: torch.Tensor,
        detection_feat: torch.Tensor,
        perception_grid: torch.Tensor,
        agent_poses: torch.Tensor,
        candidate_feat: Optional[torch.Tensor] = None,
        candidate_valid: Optional[torch.Tensor] = None,
        target_reference_tokens: Optional[torch.Tensor] = None,
        target_reference_valid: Optional[torch.Tensor] = None,
        candidate_prior_bbox: Optional[torch.Tensor] = None,
        candidate_prior_valid: Optional[torch.Tensor] = None,
        synthetic_occlusion: Optional[torch.Tensor] = None,
        coarse_missing_mask: Optional[torch.Tensor] = None,
        fine_missing_mask: Optional[torch.Tensor] = None,
        instructions: Optional[List[str]] = None,
        joint_instructions: Optional[List[str]] = None,
        drone_instructions: Optional[List[str]] = None,
        dog_instructions: Optional[List[str]] = None,
        yaw_hist: Optional[torch.Tensor] = None,
        yaw_curr: Optional[torch.Tensor] = None,
        route_visibility: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        if coarse_tokens.ndim != 4 or coarse_tokens.size(1) != NUM_AGENTS:
            raise ValueError("coarse_tokens must have shape (B,2,N,C)")
        if fine_tokens.ndim != 4 or fine_tokens.size(1) != NUM_AGENTS:
            raise ValueError("fine_tokens must have shape (B,2,N,C)")
        batch_size = coarse_tokens.size(0)
        if detection_feat.shape != (batch_size, NUM_AGENTS, 6):
            raise ValueError("detection_feat must have shape (B,2,6)")
        if perception_grid.ndim != 5 or perception_grid.shape[:2] != (batch_size, NUM_AGENTS):
            raise ValueError("perception_grid must have shape (B,2,H,W,4)")
        if agent_poses.shape != (batch_size, NUM_AGENTS, 4):
            raise ValueError("agent_poses must have shape (B,2,4)")

        device = next(self.parameters()).device
        if self.training:
            self.update_jepa_teacher()
        coarse_tokens = coarse_tokens.to(device) # 1×2×124×1536 历史+当前总共31帧observation,每帧4个token粗粒度representation 
        coarse_tidx = coarse_tidx.to(device) # 1×2×124 每个token的历史帧id
        fine_tokens = fine_tokens.to(device) # 1×2×64×1536 当前帧的observation,64个token的细粒度representation 
        fine_tidx = fine_tidx.to(device) # 1×2×64 每个token的帧id
        detection_feat = detection_feat.to(device, torch.float32).clone() # 1×2×6 检测结果 6维分别是目标的[cx,cy,w,h,s,v] s：置信度，v:{0,1}是否有效
        if candidate_feat is None:
            candidate_feat = detection_feat.unsqueeze(2)
        else:
            candidate_feat = candidate_feat.to(device, torch.float32)
        if candidate_feat.ndim != 4 or candidate_feat.shape[:2] != (batch_size, NUM_AGENTS) or candidate_feat.size(-1) != 6:
            raise ValueError("candidate_feat must have shape (B,2,K,6)")
        if candidate_valid is None:
            candidate_valid = candidate_feat[..., 5] > 0.5
        else:
            candidate_valid = candidate_valid.to(device, torch.bool)
        if candidate_valid.shape != candidate_feat.shape[:3]:
            raise ValueError("candidate_valid must have shape (B,2,K)")
        if target_reference_tokens is None:
            target_reference_tokens = torch.zeros(
                batch_size,
                NUM_AGENTS,
                fine_tokens.size(-1),
                device=device,
                dtype=fine_tokens.dtype,
            )
        else:
            target_reference_tokens = target_reference_tokens.to(device)
        if target_reference_tokens.shape != (
            batch_size, NUM_AGENTS, fine_tokens.size(-1)
        ):
            raise ValueError("target_reference_tokens must have shape (B,2,Cvision)")
        if target_reference_valid is None:
            target_reference_valid = torch.zeros(
                batch_size, NUM_AGENTS, dtype=torch.bool, device=device
            )
        else:
            target_reference_valid = target_reference_valid.to(device, torch.bool)
        if target_reference_valid.shape != (batch_size, NUM_AGENTS):
            raise ValueError("target_reference_valid must have shape (B,2)")
        perception_grid = perception_grid.to(device, torch.float32).clone() # 1×2×8×8×4 当前帧8×8的grid,4维表示每个grid的属性[unknown,free,obstacle,target] unknown：未知区域 free：可通行/空闲区域 obstacle：障碍物区域 target：检测到目标人物的区域
        agent_poses = agent_poses.to(device, torch.float32) # 1×2×4 每个agent的状态 [x,y,sin(yaw),cos(yaw)] xy:坐标 yaw:朝向角
        if synthetic_occlusion is None: # synthetic_occlusion：表示是否认为mask某个agent的observation,模拟agnet丢失目标
            synthetic_occlusion = torch.zeros(
                batch_size, NUM_AGENTS, dtype=torch.bool, device=device
            )
        else:
            synthetic_occlusion = synthetic_occlusion.to(device, torch.bool)
            if synthetic_occlusion.shape != (batch_size, NUM_AGENTS):
                raise ValueError("synthetic_occlusion must have shape (B,2)")
        if (synthetic_occlusion.sum(dim=1) > 1).any():
            raise ValueError("At most one view may be synthetically hidden per sample")
        if coarse_missing_mask is None: # coarse_missing_mask:表示agnet的哪些历史token需要mask
            coarse_missing_mask = synthetic_occlusion.unsqueeze(-1).expand(
                -1, -1, coarse_tokens.size(2)
            )
        else:
            coarse_missing_mask = coarse_missing_mask.to(device, torch.bool)
            if coarse_missing_mask.shape != coarse_tokens.shape[:3]:
                raise ValueError(
                    "coarse_missing_mask must have shape (B,2,Ncoarse)"
                )
        if fine_missing_mask is None: # fine_missing_mask:表示agnet的当前帧哪些token需要mask
            fine_missing_mask = synthetic_occlusion.unsqueeze(-1).expand(
                -1, -1, fine_tokens.size(2)
            )
        else:
            fine_missing_mask = fine_missing_mask.to(device, torch.bool)
            if fine_missing_mask.shape != fine_tokens.shape[:3]:
                raise ValueError("fine_missing_mask must have shape (B,2,Nfine)")
        if (
            coarse_missing_mask & ~synthetic_occlusion.unsqueeze(-1)
        ).any() or (fine_missing_mask & ~synthetic_occlusion.unsqueeze(-1)).any():
            raise ValueError("Only the synthetic receiver may contain missing tokens")

        instructions = self._ensure_text_batch(
            instructions,
            ["Follow the target person without collision."] * batch_size,
            batch_size,
        ) # self 指令
        joint_instructions = self._ensure_text_batch(
            joint_instructions,
            [
                "Use both aerial and ground observations to follow the target person "
                "without collision."
            ]
            * batch_size,
            batch_size,
        ) # joint 指令
        joint_instructions = [
            f"{joint_text} Target identity description: {identity_text}"
            for joint_text, identity_text in zip(joint_instructions, instructions)
        ]
        encodings: List[AgentStreamEncoding] = []
        for agent_id in range(NUM_AGENTS):
            history_yaw = yaw_hist[:, agent_id].to(device) if yaw_hist is not None else None # 历史pose
            current_yaw = yaw_curr[:, agent_id].to(device) if yaw_curr is not None else None # 当前pose
            encodings.append(
                self._encode_agent_streams(
                    coarse_tokens[:, agent_id],
                    coarse_tidx[:, agent_id],
                    fine_tokens[:, agent_id],
                    fine_tidx[:, agent_id],
                    detection_feat[:, agent_id],
                    candidate_feat[:, agent_id],
                    candidate_valid[:, agent_id],
                    target_reference_tokens[:, agent_id],
                    target_reference_valid[:, agent_id],
                    perception_grid[:, agent_id],
                    synthetic_occlusion[:, agent_id],
                    coarse_missing_mask[:, agent_id],
                    fine_missing_mask[:, agent_id],
                    agent_id,
                    history_yaw,
                    current_yaw,
                )
            )

        self_contexts, target_verify_contexts, candidate_contexts = self._run_self_flows(
            encodings,
            instructions,
            [drone_instructions, dog_instructions],
            device,
        )
        verify_context = torch.stack(target_verify_contexts, dim=1)
        candidate_context = torch.stack(candidate_contexts, dim=1)
        relative_candidate_logits = self.candidate_matcher(
            candidate_context, verify_context, candidate_valid
        )
        # 一个共享 VERIFY token 提供整体校准偏置，每个候选分别获得独立的二分类匹配 logit。
        presence_logits = torch.stack(
            [
                self.target_match_heads[agent_id](
                    target_verify_contexts[agent_id]
                ).squeeze(-1)
                for agent_id in range(NUM_AGENTS)
            ],
            dim=1,
        )
        candidate_match_logits = relative_candidate_logits + 0.5 * presence_logits.unsqueeze(-1)
        if not self.training and candidate_prior_bbox is not None:
            prior_bbox = candidate_prior_bbox.to(device, torch.float32)
            prior_valid = (
                candidate_prior_valid.to(device, torch.bool)
                if candidate_prior_valid is not None
                else prior_bbox[..., 2:].gt(0).all(dim=-1)
            )
            if prior_bbox.shape != (batch_size, NUM_AGENTS, 4) or prior_valid.shape != (batch_size, NUM_AGENTS):
                raise ValueError("candidate prior must have shapes (B,2,4) and (B,2)")
            candidate_match_logits = candidate_match_logits + (
                float(self.cfg.candidate_temporal_iou_weight)
                * candidate_iou_with_prior(candidate_feat[..., :4], prior_bbox)
                * prior_valid.unsqueeze(-1).to(candidate_match_logits.dtype)
            )
        candidate_probability = torch.sigmoid(candidate_match_logits)
        selected_probability, selected_index = candidate_probability.max(dim=-1)
        if candidate_probability.size(-1) > 1:
            top_probability = candidate_probability.topk(2, dim=-1).values
            selected_margin = top_probability[..., 0] - top_probability[..., 1]
        else:
            selected_margin = selected_probability
        selected_valid = candidate_valid.gather(
            -1, selected_index.unsqueeze(-1)
        ).squeeze(-1)

        if self.training:
            # 前向使用 argmax 硬选择，反向使用 softmax 梯度更新目标上下文选择器。
            # 阈值只在推理时启用，避免训练初期把所有候选都拒绝。
            hard_weight = F.one_hot(
                selected_index, num_classes=candidate_feat.size(2)
            ).to(candidate_probability.dtype)
            soft_weight = torch.softmax(candidate_match_logits, dim=-1)
            selection_weight = hard_weight + soft_weight - soft_weight.detach()
            selected_accepted = selected_valid
        else:
            selected_accepted = selected_valid & (
                selected_probability >= float(self.cfg.target_match_confidence_threshold)
            )
            selection_weight = F.one_hot(
                selected_index, num_classes=candidate_feat.size(2)
            ).to(candidate_probability.dtype)
            selection_weight = selection_weight * selected_accepted.unsqueeze(-1).to(
                selection_weight.dtype
            )
        selected_candidate_valid = selected_accepted
        # 检测器可见性与匹配器接受状态相互独立：二分类得分最高的候选仍是检测观测，
        # 是否信任其上下文则由 accepted 标志控制。
        selected_detection = candidate_feat.gather(
            2, selected_index[..., None, None].expand(-1, -1, 1, 6)
        ).squeeze(2)
        selected_person_context = (
            candidate_context * selection_weight.unsqueeze(-1)
        ).sum(dim=2)
        selected_candidate_context_tensor = torch.where(
            selected_accepted.unsqueeze(-1),
            selected_person_context,
            self.null_target_context[:, 0].unsqueeze(0).expand(batch_size, -1, -1),
        )
        # SELF always remains clean and may use its selected target.  COOP must
        # not leak a synthetic receiver's clean ROI/context into the masked row.
        self_target_contexts = [
            selected_candidate_context_tensor[:, agent_id]
            for agent_id in range(NUM_AGENTS)
        ]
        cooperative_selected_valid = selected_candidate_valid & ~synthetic_occlusion
        selected_candidate_contexts = [
            self_target_contexts[agent_id]
            * cooperative_selected_valid[:, agent_id, None].to(
                self_target_contexts[agent_id].dtype
            )
            for agent_id in range(NUM_AGENTS)
        ]
        grounded_self_contexts = [
            self_contexts[agent_id]
            + self.target_context_projs[agent_id](self_target_contexts[agent_id])
            for agent_id in range(NUM_AGENTS)
        ]
        self_waypoints = torch.stack(
            [
                self.self_planners[agent_id](grounded_self_contexts[agent_id])
                * self.alpha_task
                for agent_id in range(NUM_AGENTS)
            ],
            dim=1,
        )
        legacy_target_match_logits = presence_logits
        legacy_target_match_probability = torch.sigmoid(presence_logits)
        target_match_probability = selected_probability
        target_match_logits = candidate_match_logits.gather(
            -1, selected_index.unsqueeze(-1)
        ).squeeze(-1)

        (
            coop_hidden,
            stream_spans,
            pose_hidden,
            relative_pose_hidden,
            relative_pose_features,
            coop_contexts,
            selected_candidate_hidden,
        ) = self._run_cooperative_flow(
            encodings,
            agent_poses,
            joint_instructions,
            selected_candidate_contexts,
            cooperative_selected_valid,
            device,
        )
        cooperative_base_memory = torch.cat(
            [
                coop_hidden[:, start:end]
                for start, end in stream_spans
            ]
            + [selected_candidate_hidden, pose_hidden, relative_pose_hidden],
            dim=1,
        )

        jepa_prediction_list: List[torch.Tensor] = []
        jepa_teacher_list: List[torch.Tensor] = []
        jepa_mask_list: List[torch.Tensor] = []
        target_belief_list: List[torch.Tensor] = []
        uncertainty_list: List[torch.Tensor] = []
        candidate_list: List[torch.Tensor] = []
        mode_logit_list: List[torch.Tensor] = []
        selected_list: List[torch.Tensor] = []
        for agent_id in range(NUM_AGENTS):
            jepa_memory = torch.cat(
                (cooperative_base_memory, coop_contexts[agent_id].unsqueeze(1)), dim=1
            )
            prediction_tokens = self.jepa_predictors[agent_id](
                jepa_memory, encodings[agent_id].cooperative_fine.float()
            )
            pool_logits = self.jepa_pool_scores[agent_id](prediction_tokens).squeeze(-1)
            target_mask = encodings[agent_id].target_mask
            pool_mask = torch.where(
                target_mask.any(dim=1, keepdim=True),
                target_mask,
                torch.ones_like(target_mask),
            )
            pool_logits = pool_logits.masked_fill(
                ~pool_mask, torch.finfo(pool_logits.dtype).min
            )
            pool_weights = torch.softmax(pool_logits, dim=1)
            pooled_prediction = (prediction_tokens * pool_weights.unsqueeze(-1)).sum(dim=1)
            target_belief = self.target_belief_heads[agent_id](pooled_prediction)
            uncertainty = self.uncertainty_heads[agent_id](pooled_prediction).squeeze(-1)
            decoder_memory = torch.cat(
                (
                    cooperative_base_memory,
                    prediction_tokens,
                    encodings[agent_id].obstacle_tokens.float(),
                ),
                dim=1,
            )
            candidates, mode_logits = self.coop_decoders[agent_id](
                decoder_memory, coop_contexts[agent_id]
            )
            candidates = candidates * self.alpha_task.unsqueeze(1)
            selected_mode_index = mode_logits.argmax(dim=1)
            selected = candidates[
                torch.arange(batch_size, device=device), selected_mode_index
            ]
            jepa_prediction_list.append(prediction_tokens)
            jepa_teacher_list.append(encodings[agent_id].teacher_fine.detach().float())
            jepa_mask_list.append(target_mask)
            target_belief_list.append(target_belief)
            uncertainty_list.append(uncertainty)
            candidate_list.append(candidates)
            mode_logit_list.append(mode_logits)
            selected_list.append(selected)

        cooperative_candidates = torch.stack(candidate_list, dim=1)
        cooperative_mode_logits = torch.stack(mode_logit_list, dim=1)
        cooperative_waypoints = torch.stack(selected_list, dim=1)
        effective_detection = selected_detection.clone()
        effective_detection[synthetic_occlusion] = 0.0
        if route_visibility is None:
            yolo_visible = (
                (effective_detection[..., 5] > 0.5)
                & (effective_detection[..., 4] >= float(self.cfg.detection_confidence_threshold))
            )
            observed_visible = yolo_visible & (
                target_match_probability
                >= float(self.cfg.target_match_confidence_threshold)
            )
        else:
            yolo_visible = (
                (effective_detection[..., 5] > 0.5)
                & (effective_detection[..., 4] >= float(self.cfg.detection_confidence_threshold))
            )
            observed_visible = route_visibility.to(device, torch.bool)
            if observed_visible.shape != (batch_size, NUM_AGENTS):
                raise ValueError("route_visibility must have shape (B,2)")
        needs_assistance = ~observed_visible
        both_invisible = ~observed_visible.any(dim=1)
        route_to_cooperative = needs_assistance & observed_visible.flip(1)
        route_to_belief = both_invisible[:, None].expand(-1, NUM_AGENTS)
        routing_mode = torch.where(
            route_to_cooperative,
            torch.full_like(route_to_cooperative, ROUTE_COOPERATIVE, dtype=torch.long),
            torch.full_like(route_to_cooperative, ROUTE_SELF, dtype=torch.long),
        )
        routing_mode = torch.where(
            both_invisible[:, None],
            torch.full_like(routing_mode, ROUTE_BELIEF),
            routing_mode,
        )
        if self.training or self.cfg.hard_visibility_routing:
            routed_waypoints = torch.where(
                needs_assistance[:, :, None, None],
                cooperative_waypoints,
                self_waypoints,
            )
        else:
            verified_visibility_probability = (
                yolo_visible.float() * target_match_probability
            )
            routed_waypoints = (
                verified_visibility_probability[:, :, None, None] * self_waypoints
                + (1.0 - verified_visibility_probability[:, :, None, None])
                * cooperative_waypoints
            )
            routed_waypoints = torch.where(
                route_to_belief[:, :, None, None],
                cooperative_waypoints,
                routed_waypoints,
            )

        outputs = {
            "self_waypoints": self_waypoints,
            "drone_self_waypoints": self_waypoints[:, DRONE],
            "dog_self_waypoints": self_waypoints[:, ROBOTDOG],
            "cooperative_candidates": cooperative_candidates,
            "cooperative_mode_logits": cooperative_mode_logits,
            "cooperative_waypoints": cooperative_waypoints,
            "drone_cooperative_waypoints": cooperative_waypoints[:, DRONE],
            "dog_cooperative_waypoints": cooperative_waypoints[:, ROBOTDOG],
            "waypoints": routed_waypoints,
            "agent1_waypoints": routed_waypoints[:, DRONE],
            "agent2_waypoints": routed_waypoints[:, ROBOTDOG],
            "route_to_cooperative": route_to_cooperative,
            "route_to_belief": route_to_belief,
            "routing_mode": routing_mode,
            "yolo_visible": yolo_visible,
            "observed_visible": observed_visible,
            "both_invisible": both_invisible,
            "target_match_logits": target_match_logits,
            "target_match_probability": target_match_probability,
            "legacy_target_match_probability": legacy_target_match_probability,
            "candidate_match_logits": candidate_match_logits,
            "candidate_match_probability": candidate_probability,
            # 保留兼容字段名；这里实际是 8 个独立二分类 logits，不是 K+1 softmax。
            "candidate_class_logits": candidate_match_logits,
            "candidate_selected_index": selected_index,
            "candidate_selected_probability": selected_probability,
            "candidate_selected_margin": selected_margin,
            "candidate_selected_accepted": selected_accepted,
            "selected_candidate_context": selected_candidate_context_tensor,
            "selected_detection_feat": selected_detection,
            "self_action_context": torch.stack(self_contexts, dim=1),
            "grounded_self_action_context": torch.stack(grounded_self_contexts, dim=1),
            "target_verify_context": torch.stack(target_verify_contexts, dim=1),
            "jepa_prediction_tokens": torch.stack(jepa_prediction_list, dim=1),
            "jepa_teacher_tokens": torch.stack(jepa_teacher_list, dim=1),
            "jepa_token_mask": torch.stack(jepa_mask_list, dim=1),
            "target_belief": torch.stack(target_belief_list, dim=1),
            "jepa_uncertainty_logit": torch.stack(uncertainty_list, dim=1),
            "effective_detection_feat": effective_detection,
            "synthetic_occlusion": synthetic_occlusion,
            "coarse_missing_mask": coarse_missing_mask,
            "fine_missing_mask": fine_missing_mask,
            "agent_poses": agent_poses,
            "directed_relative_pose": relative_pose_features,
            "target_reference_valid": target_reference_valid,
        }
        if not return_dict:
            raise ValueError("AirGroundCooperativeVLAV3 supports return_dict=True only")
        return outputs


OpenTrackVLAAirGroundV3 = AirGroundCooperativeVLAV3
