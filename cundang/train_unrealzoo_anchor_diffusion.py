#!/usr/bin/env python3
"""UnrealZoo 双 Agent Anchor Diffusion 专用训练入口。

整体功能：
- 复用 ``train.py`` 的双 Agent数据集、DataLoader、训练循环、日志与 checkpoint 管理。
- 将普通 MLP 模型构建替换为 ``model_unrealzoo_anchor_diffusion.py``。
- 将普通 waypoint MSE 替换为最近锚回归、候选评分 BCE、bbox 和可见性联合损失。

关键函数：
- ``build_anchor_diffusion_multi_agent_model``：构建带两套轨迹锚点的扩散模型。
- ``build_anchor_diffusion_multi_agent_dataset``：构建数据集并支持小样本调试。
- ``forward_anchor_diffusion_multi_agent_loss``：计算论文式扩散跟踪损失与辅助损失。
- ``parse_args``：合并扩散专属参数和 ``train.py`` 通用双 Agent参数。
- ``main``：替换通用训练钩子并启动 dry-run 或正式训练。

输入要求：
- 训练数据和视觉缓存格式与 ``train.py --multi_agent`` 相同。
- 两个 Agent 的轨迹锚点通常由 ``python -m tools.build_unrealzoo_trajectory_anchors`` 生成。

输出写入 ``--out_dir``，默认由 ``sh/train_anchor_diffusion.sh`` 指向
``/data/hdt/ntv_data/ckpt/ckpts_multi_agent_anchor_diffusion``。
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

import cundang.model_unrealzoo_anchor_diffusion as anchor_model
import train as base_train


_BASE_BUILD_MULTI_AGENT_DATASET = base_train.build_multi_agent_dataset


# ----------------------- 扩散模型与数据集构建 -----------------------

def build_anchor_diffusion_multi_agent_model(cfg: base_train.MultiAgentTrainConfig) -> nn.Module:
    """用 ``model_unrealzoo_anchor_diffusion.py`` 构建启用双独立锚点库的扩散模型。"""
    model_cfg = anchor_model.MultiAgentModelConfig(
        llm_name=cfg.llm_name,
        freeze_llm=cfg.freeze_llm,
        n_waypoints=cfg.n_waypoints,
        action_dims=cfg.action_dims,
        use_angle_tvi=cfg.use_angle_tvi,
        insert_time_tokens=cfg.insert_time_tokens,
        use_tanh_actions=(not cfg.no_tanh_actions),
        alpha_xy=cfg.alpha_xy,
        return_token_logits=cfg.return_token_logits,
        use_anchor_diffusion=True,
        diffusion_agent1_anchor_path=cfg.diffusion_agent1_anchor_path,
        diffusion_agent2_anchor_path=cfg.diffusion_agent2_anchor_path,
        diffusion_num_anchors=cfg.diffusion_num_anchors,
        diffusion_hidden_dim=cfg.diffusion_hidden_dim,
        diffusion_depth=cfg.diffusion_depth,
        diffusion_num_heads=cfg.diffusion_num_heads,
        diffusion_mlp_ratio=cfg.diffusion_mlp_ratio,
        diffusion_dropout=cfg.diffusion_dropout,
        diffusion_num_train_timesteps=cfg.diffusion_num_train_timesteps,
        diffusion_train_truncation_steps=cfg.diffusion_train_truncation_steps,
        diffusion_inference_start_timestep=cfg.diffusion_inference_start_timestep,
        diffusion_inference_steps=cfg.diffusion_inference_steps,
        diffusion_score_loss_weight=cfg.diffusion_score_loss_weight,
        diffusion_score_loss_reduction=cfg.diffusion_score_loss_reduction,
        diffusion_deterministic_inference=cfg.diffusion_deterministic_inference,
    )
    return anchor_model.MultiAgentOpenTrackVLA(model_cfg, vision_feat_dim=cfg.vision_feat_dim)


def build_anchor_diffusion_multi_agent_dataset(
    path: str,
    cfg: base_train.MultiAgentTrainConfig,
    cache_root: Optional[str] = None,
) -> base_train.MultiAgentJsonDataset:
    """构建数据集，并可在调试时仅保留前几个样本。"""
    ds = _BASE_BUILD_MULTI_AGENT_DATASET(path, cfg, cache_root=cache_root)
    limit = int(getattr(cfg, "debug_max_samples", 0))
    if limit <= 0 or limit >= len(ds):
        return ds
    if ds.examples is not None:
        ds.examples = ds.examples[:limit]
    elif ds._index is not None:
        ds._index = ds._index[:limit]
    print(f"[DEBUG][ANCHOR_DIFFUSION] limiting dataset to {len(ds)} samples", flush=True)
    return ds


# ----------------------- 扩散联合损失 -----------------------

def build_training_bbox_prior(
    bbox_target: torch.Tensor,
    visible_target: torch.Tensor,
    dropout_prob: float,
    center_jitter_std: float,
    size_jitter_std: float,
    training: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """构造更接近在线跟踪误差的 bbox prior。

    可见且尺寸有效的目标才允许提供 prior。训练时每个 Agent 独立执行：
    - 中心按目标框宽高比例扰动；
    - 宽高在 log-space 扰动；
    - 按 dropout_prob 丢弃 prior，切换到 absolute detection 分支。
    """
    bbox_prior = bbox_target.detach().clone().float()
    bbox_valid = (visible_target > 0.5) & (bbox_target[..., 2] > 1e-4) & (bbox_target[..., 3] > 1e-4)
    if training:
        center_scale = bbox_prior[..., 2:4].clamp_min(1e-3)
        center_noise = torch.randn_like(bbox_prior[..., 0:2]) * float(center_jitter_std) * center_scale
        size_noise = torch.randn_like(bbox_prior[..., 2:4]) * float(size_jitter_std)
        bbox_prior[..., 0:2] = bbox_prior[..., 0:2] + center_noise
        bbox_prior[..., 2:4] = bbox_prior[..., 2:4] * torch.exp(size_noise)
        if dropout_prob > 0.0:
            keep_prior = torch.rand_like(visible_target.float()) >= float(dropout_prob)
            bbox_valid = bbox_valid & keep_prior

    bbox_prior = bbox_prior.clamp(0.0, 1.0)
    bbox_prior = torch.where(bbox_valid.unsqueeze(-1), bbox_prior, torch.zeros_like(bbox_prior))
    return bbox_prior, bbox_valid


def visible_bbox_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    visible_target: torch.Tensor,
) -> torch.Tensor:
    """仅对可见且有有效尺寸的目标计算鲁棒 bbox 回归损失。"""
    valid = (visible_target > 0.5) & (target[..., 2] > 1e-4) & (target[..., 3] > 1e-4)
    per_coord = F.smooth_l1_loss(prediction.float(), target.float(), reduction="none", beta=0.05)
    mask = valid.unsqueeze(-1).to(dtype=per_coord.dtype)
    return (per_coord * mask).sum() / (mask.sum() * prediction.size(-1)).clamp_min(1.0)


def forward_anchor_diffusion_multi_agent_loss(
    model: nn.Module,
    batch: Dict[str, Any],
    cfg: base_train.MultiAgentTrainConfig,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """计算论文式扩散轨迹损失，并保留 bbox/visibility 辅助监督。

    ``action_loss`` 已经是两个 Agent 各自的：
    最近锚轨迹回归损失 + diffusion_score_loss_weight * 候选评分 BCE。
    因此这里不再调用旧版 waypoint masked MSE。
    """
    bbox_target = batch["bbox_feat"].to(device, non_blocking=True)
    visible_target = batch["visible"].to(device, non_blocking=True)
    bbox_input, bbox_valid_mask = build_training_bbox_prior(
        bbox_target=bbox_target,
        visible_target=visible_target,
        dropout_prob=cfg.bbox_dropout_prob,
        center_jitter_std=cfg.bbox_center_jitter_std,
        size_jitter_std=cfg.bbox_size_jitter_std,
        training=model.training,
    )
    if not model.training and cfg.val_bbox_source == "none":
        # 离线验证默认模拟首帧无真值框，bbox 监督目标仍保留用于计算 grounding loss。
        bbox_input = torch.zeros_like(bbox_input)
        bbox_valid_mask = torch.zeros_like(bbox_valid_mask, dtype=torch.bool)

    out = model(
        coarse_tokens=batch["coarse_tokens"].to(device, non_blocking=True),
        coarse_tidx=batch["coarse_tidx"].to(device, non_blocking=True),
        fine_tokens=batch["fine_tokens"].to(device, non_blocking=True),
        fine_tidx=batch["fine_tidx"].to(device, non_blocking=True),
        instructions=batch["instruction"],
        bbox_feat=bbox_input,
        bbox_valid_mask=bbox_valid_mask,
        yaw_hist=batch["yaw_hist"].to(device, non_blocking=True) if cfg.use_angle_tvi else None,
        yaw_curr=batch["yaw_curr"].to(device, non_blocking=True) if cfg.use_angle_tvi else None,
        # 扩散头需要 GT 路点完成最近锚分配、轨迹回归和候选评分监督。
        target_waypoints=batch["waypoints"].to(device, non_blocking=True),
        valid_mask=batch["valid_mask"].to(device, non_blocking=True),
    )
    if "action_loss" not in out:
        raise RuntimeError("Anchor diffusion model did not return action_loss. Check use_anchor_diffusion=True.")
    loss_nav = out["action_loss"]
    regression_loss = out["regression_loss"]
    score_loss = out["score_loss"]

    refined_bbox = out.get("refined_bbox")
    if cfg.beta_bbox != 0.0 and refined_bbox is not None:
        loss_bbox = visible_bbox_smooth_l1(refined_bbox, bbox_target, visible_target)
    else:
        loss_bbox = loss_nav.new_tensor(0.0)

    visible_logits = out.get("visible_logits")
    if cfg.beta_visible != 0.0 and visible_logits is not None:
        loss_visible = F.binary_cross_entropy_with_logits(visible_logits.float(), visible_target.float())
    else:
        loss_visible = loss_nav.new_tensor(0.0)

    loss = cfg.beta_nav * loss_nav + cfg.beta_bbox * loss_bbox + cfg.beta_visible * loss_visible
    return loss, {
        "loss_nav": loss_nav.detach(),
        "regression_loss": regression_loss.detach(),
        "score_loss": score_loss.detach(),
        "loss_bbox": loss_bbox.detach(),
        "loss_visible": loss_visible.detach(),
        "pred": out["waypoints"].detach(),
    }


# ----------------------- 命令行参数与训练入口 -----------------------

def parse_args() -> base_train.MultiAgentTrainConfig:
    """先解析扩散专属参数，再复用 train.py 的双 Agent 通用参数解析器。"""
    diffusion_parser = argparse.ArgumentParser(add_help=False)
    diffusion_parser.add_argument("--diffusion_agent1_anchor_path", type=str, default=None)
    diffusion_parser.add_argument("--diffusion_agent2_anchor_path", type=str, default=None)
    diffusion_parser.add_argument("--diffusion_num_anchors", type=int, default=40)
    diffusion_parser.add_argument("--diffusion_hidden_dim", type=int, default=768)
    diffusion_parser.add_argument("--diffusion_depth", type=int, default=12)
    diffusion_parser.add_argument("--diffusion_num_heads", type=int, default=12)
    diffusion_parser.add_argument("--diffusion_mlp_ratio", type=float, default=4.0)
    diffusion_parser.add_argument("--diffusion_dropout", type=float, default=0.0)
    diffusion_parser.add_argument("--diffusion_num_train_timesteps", type=int, default=1000)
    diffusion_parser.add_argument("--diffusion_train_truncation_steps", type=int, default=50)
    diffusion_parser.add_argument("--diffusion_inference_start_timestep", type=int, default=10)
    diffusion_parser.add_argument("--diffusion_inference_steps", type=int, default=2)
    diffusion_parser.add_argument("--diffusion_score_loss_weight", type=float, default=100.0)
    diffusion_parser.add_argument(
        "--diffusion_score_loss_reduction",
        choices=("mean", "sum"),
        default="mean",
        help="Reduce BCE over anchor modes by mean (stable default) or sum (literal paper formula).",
    )
    diffusion_parser.add_argument(
        "--diffusion_deterministic_inference",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    diffusion_parser.add_argument(
        "--debug_max_samples",
        type=int,
        default=0,
        help="Only use the first N samples for an end-to-end debugger run; 0 uses the full dataset.",
    )
    diffusion_parser.add_argument(
        "--bbox_center_jitter_std",
        type=float,
        default=0.25,
        help="Std of training bbox center noise, relative to the target bbox width/height.",
    )
    diffusion_parser.add_argument(
        "--bbox_size_jitter_std",
        type=float,
        default=0.20,
        help="Std of log-space training bbox width/height noise.",
    )
    if "-h" in sys.argv or "--help" in sys.argv:
        print("\nAnchor diffusion additional options:")
        diffusion_parser.print_help()
    diffusion_args, remaining = diffusion_parser.parse_known_args()

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *remaining]
        cfg = base_train.parse_multi_agent_args()
    finally:
        sys.argv = original_argv

    for key, value in vars(diffusion_args).items():
        setattr(cfg, key, value)
    # 写入 checkpoint config，便于区分旧共享 GND 与当前双 GND 架构。
    setattr(cfg, "use_anchor_diffusion", True)
    setattr(cfg, "grounding_architecture", "dual_agent_gnd_v2")
    setattr(cfg, "multimodal_sequence_layout", "dual_visual_before_queries_v2")
    setattr(cfg, "action_scale_version", "anchor_maxabs_v2")
    if not cfg.dry_run:
        if not cfg.diffusion_agent1_anchor_path or not cfg.diffusion_agent2_anchor_path:
            raise ValueError(
                "Training requires --diffusion_agent1_anchor_path and --diffusion_agent2_anchor_path. "
                "Run `python -m tools.build_unrealzoo_trajectory_anchors` first."
            )
        if cfg.diffusion_hidden_dim % cfg.diffusion_num_heads != 0:
            raise ValueError("diffusion_hidden_dim must be divisible by diffusion_num_heads.")
    return cfg


def main() -> None:
    cfg = parse_args()
    # 复用成熟训练循环，同时仅在当前进程替换模型和 loss，不修改 train.py/model.py。
    base_train.build_multi_agent_dataset = build_anchor_diffusion_multi_agent_dataset
    base_train.build_multi_agent_model = build_anchor_diffusion_multi_agent_model
    base_train.forward_multi_agent_loss = forward_anchor_diffusion_multi_agent_loss
    base_train.train_multi_agent(cfg)


if __name__ == "__main__":
    main()
