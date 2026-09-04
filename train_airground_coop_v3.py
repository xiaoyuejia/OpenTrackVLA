#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train the independent three-stream air-ground cooperative V3 model.

This entry point is the only AirGround model trainer in this repository. It
reuses the V3 common dataset/DDP/optimizer loop from
``train_airground_v3_common.py`` and installs V3-specific dataset, model and loss callbacks in the
current Python process.

Information-flow contract:

* drone self: independent action/verification prompt rows -> shared-weight LLM
  -> drone self head and target-match head;
* dog self: independent action/verification prompt rows -> shared-weight LLM
  -> dog self head and target-match head;
* cooperative: visible source + missing receiver + two agent poses -> shared-weight LLM ->
  two symmetric multimodal cooperative decoders.

The model receives offline YOLO detections and category-free perception grids.
Ground-truth visibility/target pose are supervision labels only.  Receiver
corruption follows an epoch curriculum ranging from target-ROI masking to
current/recent/full visual loss.  A bounded physical receiver relocation is
used only when the current frame is fully missing.  Instead of asking the
receiver to jump onto a rigidly re-expressed clean path, the cooperative loss
uses a speed- and yaw-rate-limited recovery trajectory from its current origin.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
import math
import multiprocessing as mp
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml

import train_airground_v3_common as base_train
from model_airground_coop_v3 import (
    DRONE,
    NUM_AGENTS,
    ROBOTDOG,
    AirGroundCoopV3ModelConfig,
    AirGroundCooperativeVLAV3,
)
from offline_detection_segmentation.core import (
    LEGACY_SCHEMA_VERSIONS,
    SCHEMA_VERSION,
    cache_path_for_image,
)


ARCHITECTURE = "airground_three_stream_cooperative_v3"
MODEL_SOURCE = "model_airground_coop_v3.py"
COOPERATIVE_TARGET_FRAME_VERSION = "receiver_feasible_recovery_v1"
RECEIVER_CORRUPTION_VERSION = "roi_temporal_curriculum_v1"
RELATIVE_POSE_VERSION = "directed_receiver_local_v1"

CORRUPTION_NATURAL = 0
CORRUPTION_ROI_ONLY = 1
CORRUPTION_CURRENT_FULL = 2
CORRUPTION_RECENT_FULL = 3
CORRUPTION_ALL_FULL = 4


def _default_receiver_curriculum() -> List[Dict[str, Any]]:
    return [
        {
            "end_epoch": 3,
            "assistance_probability": 0.70,
            "roi_only_probability": 0.70,
            "current_full_probability": 0.30,
            "recent_full_probability": 0.00,
            "all_full_probability": 0.00,
            "recent_history_lengths": [2],
            "pose_perturb_probability": 0.00,
            "pose_translation_max_m": 0.25,
            "pose_yaw_max_deg": 10.0,
        },
        {
            "end_epoch": 7,
            "assistance_probability": 0.85,
            "roi_only_probability": 0.40,
            "current_full_probability": 0.30,
            "recent_full_probability": 0.25,
            "all_full_probability": 0.05,
            "recent_history_lengths": [2, 4],
            "pose_perturb_probability": 0.50,
            "pose_translation_max_m": 0.50,
            "pose_yaw_max_deg": 30.0,
        },
        {
            "end_epoch": 1000000,
            "assistance_probability": 0.90,
            "roi_only_probability": 0.30,
            "current_full_probability": 0.25,
            "recent_full_probability": 0.35,
            "all_full_probability": 0.10,
            "recent_history_lengths": [2, 4, 8, 16],
            "pose_perturb_probability": 0.50,
            "pose_translation_max_m": 0.50,
            "pose_yaw_max_deg": 30.0,
        },
    ]
DEFAULT_CONFIG = (
    Path(__file__).resolve().parent / "config/airground_cooperative_tracking_v3.yaml"
)
_BASE_COLLATE = base_train.collate_multi_agent_batch


@dataclass
class AirGroundV3DataConfig(base_train.MultiAgentDataConfig):
    perception_cache_root: str = "offline_detection_segmentation/outputs/full_cache"
    candidate_cache_root: str = ""
    require_topk_candidate_cache: bool = False
    allow_missing_perception_cache: bool = False
    perception_grid_size: int = 8
    target_match_iou_threshold: float = 0.30
    use_target_reference: bool = True
    target_reference_dropout_prob: float = 0.0
    synthetic_false_positive_prob: float = 0.25
    temporal_stride: int = 5
    candidate_top_k: int = 8
    shuffle_candidates: bool = True
    synthetic_drone_occlusion_prob: float = 0.50
    synthetic_dog_occlusion_prob: float = 0.50
    deterministic_occlusion: bool = False
    occlusion_seed: int = 731
    target_roi_expand_ratio: float = 1.50
    receiver_corruption_curriculum: List[Dict[str, Any]] = field(
        default_factory=_default_receiver_curriculum
    )
    pose_global_rotation_max_deg: float = 180.0
    pose_global_translation_max_m: float = 20.0
    drone_max_speed_mps: float = 2.5
    dog_max_speed_mps: float = 2.5
    drone_max_yaw_rate_rps: float = 1.5
    dog_max_yaw_rate_rps: float = 1.5


@dataclass
class AirGroundV3TrainConfig(base_train.MultiAgentTrainConfig):
    # Fixed data split and offline perception.
    train_episodes_manifest: str = ""
    val_episodes_manifest: str = ""
    # Select which agent contributes supervision.  The other observation can
    # still be consumed as cooperative context, but none of its output losses
    # contribute gradients in single-agent mode.
    train_agent: str = "both"
    expected_train_episodes: int = 2142
    expected_val_episodes: int = 913
    perception_cache_root: str = "offline_detection_segmentation/outputs/full_cache"
    candidate_cache_root: str = ""
    require_topk_candidate_cache: bool = False
    candidate_top_k: int = 8
    allow_missing_perception_cache: bool = False
    perception_grid_size: int = 8
    target_match_iou_threshold: float = 0.30
    use_target_reference: bool = True
    train_target_reference_dropout_prob: float = 0.20
    val_target_reference_dropout_prob: float = 0.0
    train_synthetic_false_positive_prob: float = 0.25
    val_synthetic_false_positive_prob: float = 0.0
    train_temporal_stride: int = 5
    train_synthetic_drone_occlusion_prob: float = 0.50
    train_synthetic_dog_occlusion_prob: float = 0.50
    val_synthetic_drone_occlusion_prob: float = 0.50
    val_synthetic_dog_occlusion_prob: float = 0.50
    occlusion_seed: int = 731
    target_roi_expand_ratio: float = 1.50
    receiver_corruption_curriculum: List[Dict[str, Any]] = field(
        default_factory=_default_receiver_curriculum
    )
    pose_global_rotation_max_deg: float = 180.0
    pose_global_translation_max_m: float = 20.0
    pose_position_scale_m: float = 20.0

    # Three-stream model.
    num_modes: int = 4
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
    drone_target_verification_prompt: str = (
        "YOLO proposes up to eight people in the aerial view. Compare every "
        "candidate ROI with the target identity description and visual history; "
        "rank the tracked person and reject the set when no candidate matches."
    )
    dog_target_verification_prompt: str = (
        "YOLO proposes up to eight people in the ground view. Compare every "
        "candidate ROI with the target identity description and visual history; "
        "rank the tracked person and reject the set when no candidate matches."
    )

    # V3 losses. beta_nav inherited from the base config weights self streams.
    beta_cooperative_waypoint: float = 100.0
    beta_mode_classification: float = 1.0
    beta_jepa: float = 1.0
    beta_target_belief: float = 1.0
    beta_target_match: float = 1.0
    beta_uncertainty: float = 0.1
    beta_smoothness: float = 0.1
    beta_kinematics: float = 0.1
    beta_diversity: float = 0.1
    beta_obstacle: float = 0.0
    diversity_margin_m: float = 0.25
    drone_max_speed_mps: float = 2.5
    dog_max_speed_mps: float = 2.5
    drone_max_yaw_rate_rps: float = 1.5
    dog_max_yaw_rate_rps: float = 1.5
    dog_lateral_tolerance_m: float = 0.05
    # V3 predicts a complete future local pose.  RobotDog y is a position to
    # recover, not a lateral velocity command, so it needs the same waypoint
    # supervision as x.  A relocated synthetic receiver is supervised by a
    # dynamically feasible recovery trajectory before this loss is evaluated.
    dog_lateral_loss_weight: float = 1.0
    enable_projected_obstacle_loss: bool = False

    compact_terminal_log: bool = True
    model_architecture: str = ARCHITECTURE
    model_source: str = MODEL_SOURCE
    cooperative_target_frame_version: str = COOPERATIVE_TARGET_FRAME_VERSION
    receiver_corruption_version: str = RECEIVER_CORRUPTION_VERSION
    relative_pose_version: str = RELATIVE_POSE_VERSION


def _as_mapping(value: Any, name: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"YAML section {name!r} must be a mapping")
    return dict(value)


def load_config(path: Path) -> AirGroundV3TrainConfig:
    """Load and validate the canonical V3 YAML before model creation."""

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw = _as_mapping(raw, "root")
    actual_architecture = str(raw.get("architecture", ""))
    if actual_architecture != ARCHITECTURE:
        raise ValueError(
            f"This launcher requires architecture={ARCHITECTURE!r}, "
            f"got {actual_architecture!r}"
        )
    merged: Dict[str, Any] = {}
    for section in ("data", "model", "optimization", "runtime"):
        value = raw.get(section, {})
        if value:
            merged.update(_as_mapping(value, section))
    for launcher_only in ("cuda_visible_devices", "nproc_per_node"):
        merged.pop(launcher_only, None)
    unknown = sorted(set(merged) - set(AirGroundV3TrainConfig.__dataclass_fields__))
    if unknown:
        raise ValueError(f"Unsupported V3 YAML keys: {unknown}")
    merged["model_architecture"] = ARCHITECTURE
    merged["model_source"] = MODEL_SOURCE
    return AirGroundV3TrainConfig(**merged)


def _manifest_files(path: Path) -> Tuple[set[str], set[Path]]:
    entries = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise ValueError(f"Episode manifest must be a JSON list: {path}")
    episode_ids: set[str] = set()
    files: set[Path] = set()
    for item in entries:
        if not isinstance(item, dict) or not item.get("episode_id") or not item.get("jsonl"):
            raise ValueError(f"Malformed manifest entry in {path}: {item!r}")
        episode_ids.add(str(item["episode_id"]))
        files.add(Path(str(item["jsonl"])).resolve())
    if len(episode_ids) != len(entries) or len(files) != len(entries):
        raise ValueError(f"Duplicate entries in manifest: {path}")
    return episode_ids, files


def validate_fixed_split(cfg: AirGroundV3TrainConfig) -> None:
    train_manifest = Path(cfg.train_episodes_manifest).resolve()
    val_manifest = Path(cfg.val_episodes_manifest).resolve()
    train_root = Path(cfg.train_json).resolve()
    val_root = Path(cfg.val_json).resolve() if cfg.val_json else None
    for path in (train_manifest, val_manifest, train_root, val_root):
        if path is not None and not path.exists():
            raise FileNotFoundError(path)
    train_ids, train_files = _manifest_files(train_manifest)
    val_ids, val_files = _manifest_files(val_manifest)
    if len(train_ids) != int(cfg.expected_train_episodes):
        raise ValueError(
            f"Expected {cfg.expected_train_episodes} train episodes, got {len(train_ids)}"
        )
    if len(val_ids) != int(cfg.expected_val_episodes):
        raise ValueError(
            f"Expected {cfg.expected_val_episodes} val episodes, got {len(val_ids)}"
        )
    if train_ids & val_ids or train_files & val_files:
        raise ValueError("Train/validation manifests overlap")
    scanned_train = {path.resolve() for path in train_root.rglob("*.jsonl")}
    if scanned_train != train_files:
        raise ValueError(
            "train_json differs from manifest: "
            f"extra={len(scanned_train - train_files)} "
            f"missing={len(train_files - scanned_train)}"
        )
    if val_root is not None:
        scanned_val = {path.resolve() for path in val_root.rglob("*.jsonl")}
        if scanned_val != val_files:
            raise ValueError(
                "val_json differs from manifest: "
                f"extra={len(scanned_val - val_files)} "
                f"missing={len(val_files - scanned_val)}"
            )
    print(
        f"[SPLIT][AIRGROUND_V3] fixed 70:30 verified: "
        f"train={len(train_ids)} val={len(val_ids)}",
        flush=True,
    )


def agent_poses_from_unreal(
    drone_pose: Any, dog_pose: Any
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return two world-frame ``[x_m,y_m,sin(yaw),cos(yaw)]`` poses."""

    try:
        output = []
        for raw_pose in (drone_pose, dog_pose):
            pose = np.asarray(raw_pose, dtype=np.float32)
            if pose.ndim != 1 or pose.shape[0] < 6:
                raise ValueError("agent pose needs at least six values")
            yaw = math.radians(float(pose[4]))
            output.append(
                torch.tensor(
                    [
                        float(pose[0] / base_train.UNREAL_UNITS_PER_METER),
                        float(pose[1] / base_train.UNREAL_UNITS_PER_METER),
                        math.sin(yaw),
                        math.cos(yaw),
                    ],
                    dtype=torch.float32,
                )
            )
        return torch.stack(output), torch.tensor(True, dtype=torch.bool)
    except Exception:
        return torch.zeros(NUM_AGENTS, 4, dtype=torch.float32), torch.tensor(
            False, dtype=torch.bool
        )


def transform_agent_poses_shared_se2(
    agent_poses: torch.Tensor,
    rotation_rad: float,
    translation_xy_m: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Express both poses in one randomly transformed, pair-centred frame."""

    if agent_poses.shape != (NUM_AGENTS, 4):
        raise ValueError("agent_poses must have shape (2,4)")
    translation = torch.as_tensor(translation_xy_m, dtype=torch.float32).view(2)
    centre = agent_poses[:, :2].mean(dim=0)
    angle = torch.tensor(float(rotation_rad), dtype=torch.float32)
    cosine, sine = torch.cos(angle), torch.sin(angle)
    rotation = torch.stack(
        (torch.stack((cosine, -sine)), torch.stack((sine, cosine)))
    )
    output = agent_poses.clone().float()
    output[:, :2] = (agent_poses[:, :2] - centre) @ rotation.T + translation
    yaw = torch.atan2(agent_poses[:, 2], agent_poses[:, 3]) + angle
    output[:, 2] = torch.sin(yaw)
    output[:, 3] = torch.cos(yaw)
    return output, centre


def perturb_receiver_pose(
    receiver_pose: torch.Tensor,
    forward_offset_m: float,
    right_offset_m: float,
    yaw_offset_rad: float,
) -> torch.Tensor:
    """Apply one receiver-only perturbation in the receiver's local frame."""

    yaw = torch.atan2(receiver_pose[2], receiver_pose[3])
    forward = torch.stack((torch.cos(yaw), torch.sin(yaw)))
    right = torch.stack((-torch.sin(yaw), torch.cos(yaw)))
    output = receiver_pose.clone().float()
    output[:2] += float(forward_offset_m) * forward + float(right_offset_m) * right
    perturbed_yaw = yaw + float(yaw_offset_rad)
    output[2] = torch.sin(perturbed_yaw)
    output[3] = torch.cos(perturbed_yaw)
    return output


def _wrap_tensor_angle(value: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(value), torch.cos(value))


def build_feasible_receiver_recovery_target(
    waypoints: torch.Tensor,
    perturbation: torch.Tensor,
    *,
    dt: float,
    max_speed_mps: float,
    max_yaw_rate_rps: float,
    nonholonomic: bool,
) -> torch.Tensor:
    """Build a dynamically reachable target from a counterfactual receiver pose.

    The transformed clean trajectory is a *reference*, not a label.  Starting
    from the structural origin, each output step moves toward that reference
    while respecting translation and yaw-rate limits.  This removes the old
    discontinuity where waypoint zero stayed at the origin but waypoint one
    instantly absorbed the full receiver translation/yaw perturbation.
    """

    clean = torch.as_tensor(waypoints)
    delta = torch.as_tensor(
        perturbation, dtype=clean.dtype, device=clean.device
    ).reshape(-1)
    if clean.ndim != 2 or clean.size(0) < 1 or clean.size(-1) < 3:
        raise ValueError(
            "waypoints must have shape (N>=1,D>=3), got "
            f"{tuple(clean.shape)}"
        )
    if delta.numel() != 3:
        raise ValueError("perturbation must contain [forward_m,right_m,yaw_rad]")
    if not torch.isfinite(clean[..., :3]).all() or not torch.isfinite(delta).all():
        raise ValueError("waypoints and perturbation must be finite")
    if dt <= 0.0 or max_speed_mps <= 0.0 or max_yaw_rate_rps <= 0.0:
        raise ValueError("dt and receiver motion limits must be positive")
    if bool(delta.abs().max() <= 1.0e-8):
        return clean.clone()

    cosine = torch.cos(delta[2])
    sine = torch.sin(delta[2])
    relative_xy = clean[:, :2] - delta[:2]
    reference = clean.clone()
    reference[:, 0] = cosine * relative_xy[:, 0] + sine * relative_xy[:, 1]
    reference[:, 1] = -sine * relative_xy[:, 0] + cosine * relative_xy[:, 1]
    reference[:, 2] = _wrap_tensor_angle(clean[:, 2] - delta[2])

    output = torch.zeros_like(clean)
    max_translation = clean.new_tensor(float(max_speed_mps) * float(dt))
    max_yaw_step = clean.new_tensor(float(max_yaw_rate_rps) * float(dt))
    for index in range(1, clean.size(0)):
        current_xy = output[index - 1, :2]
        current_yaw = output[index - 1, 2]
        position_error = reference[index, :2] - current_xy
        distance = torch.linalg.vector_norm(position_error)

        if nonholonomic:
            bearing = torch.atan2(position_error[1], position_error[0])
            desired_yaw = torch.where(
                distance > 1.0e-5, bearing, reference[index, 2]
            )
            yaw_error = _wrap_tensor_angle(desired_yaw - current_yaw)
            yaw_step = yaw_error.clamp(-max_yaw_step, max_yaw_step)
            next_yaw = _wrap_tensor_angle(current_yaw + yaw_step)
            # Rotate first when the reference is behind the RobotDog.  The
            # midpoint heading gives a stable constant-curvature approximation.
            alignment = torch.cos(yaw_error).clamp(0.0, 1.0)
            forward = torch.minimum(distance, max_translation) * alignment
            midpoint_yaw = current_yaw + 0.5 * yaw_step
            step_xy = forward * torch.stack(
                (torch.cos(midpoint_yaw), torch.sin(midpoint_yaw))
            )
        else:
            scale = torch.minimum(
                clean.new_tensor(1.0), max_translation / distance.clamp_min(1.0e-8)
            )
            step_xy = position_error * scale
            yaw_error = _wrap_tensor_angle(reference[index, 2] - current_yaw)
            yaw_step = yaw_error.clamp(-max_yaw_step, max_yaw_step)
            next_yaw = _wrap_tensor_angle(current_yaw + yaw_step)

        output[index, :2] = current_xy + step_xy
        output[index, 2] = next_yaw
        if clean.size(-1) > 3:
            output[index, 3:] = clean[index, 3:]
    return output


def transform_relative_target_pose_to_receiver(
    relative_target_pose: torch.Tensor, perturbation: torch.Tensor
) -> torch.Tensor:
    """Re-express the current target belief from a relocated receiver pose."""

    value = torch.as_tensor(relative_target_pose)
    delta = torch.as_tensor(
        perturbation, dtype=value.dtype, device=value.device
    ).reshape(-1)
    if value.shape != (5,) or delta.numel() != 3:
        raise ValueError("target pose must be (5,) and perturbation must be (3,)")
    output = value.clone()
    relative_xy = value[:2] - delta[:2]
    cosine = torch.cos(delta[2])
    sine = torch.sin(delta[2])
    output[0] = cosine * relative_xy[0] + sine * relative_xy[1]
    output[1] = -sine * relative_xy[0] + cosine * relative_xy[1]
    clean_yaw = torch.atan2(value[3], value[4])
    new_yaw = _wrap_tensor_angle(clean_yaw - delta[2])
    output[3] = torch.sin(new_yaw)
    output[4] = torch.cos(new_yaw)
    return output


def build_cooperative_waypoint_targets(
    waypoints: torch.Tensor,
    synthetic_occlusion: torch.Tensor,
    receiver_pose_perturbation: torch.Tensor,
    *,
    dt: float,
    drone_max_speed_mps: float,
    dog_max_speed_mps: float,
    drone_max_yaw_rate_rps: float,
    dog_max_yaw_rate_rps: float,
) -> torch.Tensor:
    """Build clean source targets plus a feasible synthetic receiver recovery."""

    clean = torch.as_tensor(waypoints)
    occluded = torch.as_tensor(synthetic_occlusion, dtype=torch.bool)
    perturbations = torch.as_tensor(
        receiver_pose_perturbation,
        dtype=clean.dtype,
        device=clean.device,
    )
    if clean.ndim != 3 or clean.size(0) != NUM_AGENTS or clean.size(-1) < 3:
        raise ValueError(
            "waypoints must have shape (2,N>=1,D>=3), got "
            f"{tuple(clean.shape)}"
        )
    if occluded.shape != (NUM_AGENTS,):
        raise ValueError(
            "synthetic_occlusion must have shape (2,), got "
            f"{tuple(occluded.shape)}"
        )
    if perturbations.shape != (NUM_AGENTS, 3):
        raise ValueError(
            "receiver_pose_perturbation must have shape (2,3), got "
            f"{tuple(perturbations.shape)}"
        )
    if int(occluded.sum().item()) > 1:
        raise ValueError("V3 synthetic receiver augmentation must mask at most one agent")

    output = clean.clone()
    for agent_id in range(NUM_AGENTS):
        if bool(occluded[agent_id]) and bool(
            perturbations[agent_id].abs().max() > 1.0e-8
        ):
            output[agent_id] = build_feasible_receiver_recovery_target(
                clean[agent_id],
                perturbations[agent_id],
                dt=float(dt),
                max_speed_mps=(
                    float(drone_max_speed_mps)
                    if agent_id == DRONE
                    else float(dog_max_speed_mps)
                ),
                max_yaw_rate_rps=(
                    float(drone_max_yaw_rate_rps)
                    if agent_id == DRONE
                    else float(dog_max_yaw_rate_rps)
                ),
                nonholonomic=agent_id == ROBOTDOG,
            )
    return output


def cxcywh_iou(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Pairwise-aligned IoU for normalized ``cx,cy,w,h`` boxes."""

    if first.shape != second.shape or first.shape[-1] != 4:
        raise ValueError(
            f"cxcywh_iou requires matching (...,4) tensors, got "
            f"{tuple(first.shape)} and {tuple(second.shape)}"
        )
    first = first.float()
    second = second.float()
    first_half = first[..., 2:].clamp_min(0.0) * 0.5
    second_half = second[..., 2:].clamp_min(0.0) * 0.5
    first_min, first_max = first[..., :2] - first_half, first[..., :2] + first_half
    second_min, second_max = second[..., :2] - second_half, second[..., :2] + second_half
    intersection_size = (
        torch.minimum(first_max, second_max) - torch.maximum(first_min, second_min)
    ).clamp_min(0.0)
    intersection = intersection_size[..., 0] * intersection_size[..., 1]
    first_area = (first[..., 2] * first[..., 3]).clamp_min(0.0)
    second_area = (second[..., 2] * second[..., 3]).clamp_min(0.0)
    union = first_area + second_area - intersection
    return torch.where(union > 0.0, intersection / union.clamp_min(1.0e-8), torch.zeros_like(union))


class AirGroundV3JsonDataset(base_train.MultiAgentJsonDataset):
    """Add perception plus curriculum receiver-recovery supervision."""

    cfg: AirGroundV3DataConfig

    def __init__(self, cfg: AirGroundV3DataConfig):
        super().__init__(cfg)
        self.perception_cache_root = Path(cfg.perception_cache_root).resolve()
        self.candidate_cache_root = (
            Path(cfg.candidate_cache_root).resolve()
            if cfg.candidate_cache_root
            else None
        )
        self._candidate_bundle_cache: OrderedDict[
            str, Tuple[Dict[str, int], torch.Tensor, torch.Tensor, torch.Tensor]
        ] = OrderedDict()
        # Shared STT/DT/AT temporal identity evidence: keep one pooled initial
        # target ROI per episode/agent so shuffled frames see persistent memory.
        self._initial_reference_cache: OrderedDict[
            str, Tuple[torch.Tensor, torch.Tensor]
        ] = OrderedDict()
        if not self.perception_cache_root.is_dir():
            raise FileNotFoundError(self.perception_cache_root)
        # The sampler updates this value before every epoch.  multiprocessing.Value
        # keeps persistent DataLoader workers on the same curriculum stage.
        initial_epoch = 1000000 if cfg.deterministic_occlusion else 0
        self._curriculum_epoch = mp.Value("i", initial_epoch, lock=False)

    def set_epoch(self, epoch: int) -> None:
        self._curriculum_epoch.value = int(epoch)

    def _current_curriculum_stage(self) -> Dict[str, Any]:
        stages = list(self.cfg.receiver_corruption_curriculum)
        if not stages:
            raise ValueError("receiver_corruption_curriculum must not be empty")
        epoch_value = int(getattr(getattr(self, "_curriculum_epoch", None), "value", 0))
        epoch_number = epoch_value + 1
        for raw_stage in stages:
            stage = dict(raw_stage)
            if epoch_number <= int(stage.get("end_epoch", 0)):
                return stage
        return dict(stages[-1])

    @staticmethod
    def _agent_record(example: Dict[str, Any], name: str, legacy_index: int) -> Dict[str, Any]:
        agents = example.get("agents")
        if isinstance(agents, dict) and isinstance(agents.get(name), dict):
            return agents[name]
        prefix = f"agent{legacy_index}_"
        return {
            "name": example.get(f"agent{legacy_index}_name", name),
            "current": example.get(prefix + "current"),
            "pose": example.get(prefix + "pose"),
            "target_visible": example.get(prefix + "target_visible", True),
        }

    def _load_candidate_features(self, relative: Path) -> Optional[torch.Tensor]:
        if self.candidate_cache_root is None:
            return None
        alternatives=[relative]
        # New compact data layout stores caches directly under
        # ``perception_cache/{dt,at,stt}``, while JSONL frame paths retain the
        # logical ``frames/{kind}/...`` prefix.
        if "frames" in relative.parts:
            parts=list(relative.parts)
            alternatives.append(Path(*parts[parts.index("frames") + 1:]))
        parts=list(relative.parts)
        if "frames" in parts and "at" in parts:
            parts[parts.index("at")]="dt"; alternatives.append(Path(*parts))
        for candidate_relative in alternatives:
            view=candidate_relative.parent
            bundle=self.candidate_cache_root/view.parent/f'{view.name}.candidates.npz'
            if not bundle.is_file() and view.name in {"drone", "robotdog"}:
                # Compact candidate cache stores one bundle per agent under
                # the episode view directory: .../<episode>/drone.candidates.npz.
                bundle = self.candidate_cache_root/view.parent/f'{view.name}.candidates.npz'
            if bundle.is_file():
                key=str(bundle); cached=self._candidate_bundle_cache.get(key)
                if cached is None:
                    with np.load(bundle,allow_pickle=False) as data:
                        schema=str(np.asarray(data['schema_version']).item())
                        if schema!='person_candidates.bundle.v1': raise ValueError(f'Candidate schema mismatch at {bundle}: {schema!r}')
                        stems=[str(value) for value in np.asarray(data['frame_stems']).tolist()]
                        boxes=torch.from_numpy(np.asarray(data['boxes_cxcywh_norm'],dtype=np.float32)); scores=torch.from_numpy(np.asarray(data['scores'],dtype=np.float32)); valid=torch.from_numpy(np.asarray(data['valid'],dtype=np.bool_))
                    cached=({stem:index for index,stem in enumerate(stems)},boxes,scores,valid); self._candidate_bundle_cache[key]=cached; self._candidate_bundle_cache.move_to_end(key)
                    while len(self._candidate_bundle_cache)>32: self._candidate_bundle_cache.popitem(last=False)
                index_by_stem,boxes,scores,valid=cached; row=index_by_stem.get(candidate_relative.stem)
                if row is None: raise KeyError(f'{candidate_relative.stem} missing from {bundle}')
                mask=valid[row]
                return torch.cat((boxes[row][mask],scores[row][mask,None],torch.ones(int(mask.sum()),1)),dim=-1)
            per_frame=self.candidate_cache_root/candidate_relative.parent/f'{candidate_relative.stem}.candidates.npz'
            if per_frame.is_file():
                with np.load(per_frame,allow_pickle=False) as data:
                    schema=str(np.asarray(data['schema_version']).item())
                    if schema!='person_candidates.topk.v1': raise ValueError(f'Candidate schema mismatch at {per_frame}: {schema!r}')
                    boxes=torch.from_numpy(np.asarray(data['boxes_cxcywh_norm'],dtype=np.float32).reshape(-1,4)); scores=torch.from_numpy(np.asarray(data['scores'],dtype=np.float32).reshape(-1,1))
                return torch.cat((boxes,scores,torch.ones(scores.size(0),1)),dim=-1)
        return None

    def _load_perception(
        self, image_relative_path: str
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        relative = Path(str(image_relative_path))
        if relative.is_absolute():
            try:
                relative = relative.relative_to(self.base_root)
            except ValueError as exc:
                raise ValueError(f"Image is outside dataset root: {relative}") from exc
        path = cache_path_for_image(self.perception_cache_root, relative)
        if not path.is_file() and "frames" in relative.parts:
            compact = Path(*relative.parts[relative.parts.index("frames") + 1:])
            path = cache_path_for_image(self.perception_cache_root, compact)
        if not path.is_file() and "frames" in relative.parts and "at" in relative.parts:
            fallback_parts = list(relative.parts)
            fallback_parts[fallback_parts.index("at")] = "dt"
            dt_path = cache_path_for_image(
                self.perception_cache_root, Path(*fallback_parts)
            )
            if dt_path.is_file():
                path = dt_path
        if not path.is_file():
            # Compact layout may intentionally keep only the Top-K candidate
            # cache for DT/AT.  Promote candidates to a valid lightweight
            # perception input instead of failing on a missing per-frame grid.
            compact_candidates = self._load_candidate_features(relative)
            if compact_candidates is not None:
                side = int(self.cfg.perception_grid_size)
                grid = torch.zeros(side, side, 4, dtype=torch.float32)
                grid[..., 0] = 1.0
                has_person = bool(compact_candidates.numel())
                feat = (
                    compact_candidates[0].clone()
                    if has_person
                    else torch.zeros(6, dtype=torch.float32)
                )
                return (
                    feat,
                    grid,
                    # Cache validity and person presence are different facts.
                    # An empty but successfully loaded Top-K cache remains a
                    # valid observation, but has no candidate BCE supervision.
                    torch.tensor(True, dtype=torch.bool),
                    torch.empty(0, 4),
                    torch.empty(0),
                    compact_candidates,
                )
            if not self.cfg.allow_missing_perception_cache:
                raise FileNotFoundError(
                    f"Missing offline perception cache: {path}. Finish the offline YOLO "
                    "cache or explicitly enable the missing-cache ablation."
                )
            side = int(self.cfg.perception_grid_size)
            grid = torch.zeros(side, side, 4, dtype=torch.float32)
            grid[..., 0] = 1.0
            return (
                torch.zeros(6),
                grid,
                torch.tensor(False, dtype=torch.bool),
                torch.empty(0, 4),
                torch.empty(0),
                torch.empty(0, 6),
            )
        with np.load(path, allow_pickle=False) as cache:
            schema = str(np.asarray(cache["schema_version"]).item())
            if schema not in LEGACY_SCHEMA_VERSIONS:
                raise ValueError(f"Perception schema mismatch at {path}: {schema!r}")
            box = np.asarray(cache["person_box_cxcywh_norm"], dtype=np.float32)
            grid_np = np.asarray(cache["mask_grid"], dtype=np.float32)
            valid = bool(np.asarray(cache["person_valid"]).item())
            score = float(np.asarray(cache["person_score"]).item())
            obstacle_xyxy = np.asarray(cache["obstacle_boxes_xyxy"], dtype=np.float32)
            obstacle_scores = np.asarray(cache["obstacle_scores"], dtype=np.float32)
            has_candidates = "person_candidates_xyxy" in cache.files
            candidate_xyxy = (
                np.asarray(cache["person_candidates_xyxy"], dtype=np.float32).reshape(-1, 4)
                if has_candidates
                else np.empty((0, 4), dtype=np.float32)
            )
            candidate_scores = (
                np.asarray(cache["person_candidate_scores"], dtype=np.float32).reshape(-1)
                if has_candidates
                else np.empty((0,), dtype=np.float32)
            )
            metadata = json.loads(str(np.asarray(cache["metadata_json"]).item()))
        expected_grid = (
            int(self.cfg.perception_grid_size),
            int(self.cfg.perception_grid_size),
            4,
        )
        if box.shape != (4,) or grid_np.shape != expected_grid:
            raise ValueError(
                f"Malformed perception cache {path}: box={box.shape} grid={grid_np.shape}, "
                f"expected grid={expected_grid}"
            )
        feat = torch.tensor(np.r_[box, score, float(valid)], dtype=torch.float32)
        if not valid:
            feat[:5] = 0.0
        if obstacle_xyxy.ndim != 2 or obstacle_xyxy.shape[1:] != (4,):
            raise ValueError(f"Malformed obstacle boxes in {path}: {obstacle_xyxy.shape}")
        if obstacle_scores.shape != (obstacle_xyxy.shape[0],):
            raise ValueError(f"Malformed obstacle scores in {path}: {obstacle_scores.shape}")
        width = float(metadata.get("image_width", 0))
        height = float(metadata.get("image_height", 0))
        if width <= 0.0 or height <= 0.0:
            raise ValueError(f"Missing image dimensions in perception metadata: {path}")
        obstacle_boxes = torch.from_numpy(obstacle_xyxy.copy())
        if obstacle_boxes.numel():
            x1, y1, x2, y2 = obstacle_boxes.unbind(dim=-1)
            obstacle_boxes = torch.stack(
                (
                    (x1 + x2) / (2.0 * width),
                    (y1 + y2) / (2.0 * height),
                    (x2 - x1) / width,
                    (y2 - y1) / height,
                ),
                dim=-1,
            ).clamp(0.0, 1.0)
        candidate_feat = self._load_candidate_features(relative)
        if candidate_feat is None and self.cfg.require_topk_candidate_cache and (
            "dt" in relative.parts or "at" in relative.parts
        ):
            raise FileNotFoundError(
                f"Missing top-K person candidate cache for: {relative}"
            )
        elif candidate_feat is None and has_candidates:
            if candidate_xyxy.shape[0] != candidate_scores.shape[0]:
                raise ValueError(f"Malformed person candidates in {path}")
            candidate_boxes = torch.from_numpy(candidate_xyxy.copy())
            x1, y1, x2, y2 = candidate_boxes.unbind(dim=-1) if candidate_boxes.numel() else ([], [], [], [])
            if candidate_boxes.numel():
                candidate_boxes = torch.stack(((x1 + x2) / (2.0 * width), (y1 + y2) / (2.0 * height), (x2 - x1) / width, (y2 - y1) / height), dim=-1).clamp(0.0, 1.0)
            candidate_feat = torch.cat((candidate_boxes, torch.from_numpy(candidate_scores[:, None]), torch.ones(candidate_scores.shape[0], 1)), dim=-1) if candidate_scores.size else torch.empty(0, 6)
        elif candidate_feat is None:
            candidate_feat = feat.view(1, 6) if valid else torch.empty(0, 6)
        # Legacy v2 grids treated every non-top1 person as an obstacle. Once
        # top-K boxes are available, remove person-covered cells from the
        # target/obstacle channels; candidate tokens now carry their geometry.
        if candidate_feat.numel():
            grid_h,grid_w=grid_np.shape[:2]
            xs=(np.arange(grid_w,dtype=np.float32)+0.5)/grid_w
            ys=(np.arange(grid_h,dtype=np.float32)+0.5)/grid_h
            yy,xx=np.meshgrid(ys,xs,indexing='ij')
            person_cells=np.zeros((grid_h,grid_w),dtype=bool)
            for cx,cy,bw,bh in candidate_feat[:,:4].numpy():
                person_cells |= (np.abs(xx-cx)<=bw*0.5) & (np.abs(yy-cy)<=bh*0.5)
            moved=grid_np[...,2]+grid_np[...,3]
            grid_np[...,0][person_cells]=np.clip(grid_np[...,0][person_cells]+moved[person_cells],0.0,1.0)
            grid_np[...,2][person_cells]=0.0
            grid_np[...,3][person_cells]=0.0
        return (
            feat,
            torch.from_numpy(grid_np),
            torch.tensor(True, dtype=torch.bool),
            obstacle_boxes,
            torch.from_numpy(obstacle_scores.copy()),
            candidate_feat,
        )

    def _sample_false_positive_detection(
        self,
        detection: torch.Tensor,
        obstacle_boxes: torch.Tensor,
        obstacle_scores: torch.Tensor,
        gt_bbox: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Create a low-IoU YOLO-like proposal for verifier hard-negative mining."""

        candidates: List[torch.Tensor] = []
        candidate_scores: List[float] = []
        for box, score in zip(obstacle_boxes, obstacle_scores):
            if bool((box[2:] > 0.0).all()):
                candidates.append(box.float())
                candidate_scores.append(float(score))

        # Most frames contain no secondary detection.  Corner-shifted boxes keep
        # the original proposal scale but force the verifier to bind the numeric
        # proposal to RGB content instead of learning the dataset's centre prior.
        width = float(detection[2].clamp(0.03, 0.60))
        height = float(detection[3].clamp(0.03, 0.80))
        for cx, cy in (
            (width * 0.5, height * 0.5),
            (1.0 - width * 0.5, height * 0.5),
            (width * 0.5, 1.0 - height * 0.5),
            (1.0 - width * 0.5, 1.0 - height * 0.5),
        ):
            candidates.append(detection.new_tensor([cx, cy, width, height]))
            candidate_scores.append(float(detection[4]))
        if not candidates:
            return None

        boxes = torch.stack(candidates)
        gt = gt_bbox.view(1, 4).expand_as(boxes)
        valid = cxcywh_iou(boxes, gt) < float(self.cfg.target_match_iou_threshold)
        valid &= (boxes[:, 2] > 0.0) & (boxes[:, 3] > 0.0)
        rows = torch.nonzero(valid, as_tuple=False).flatten()
        if not rows.numel():
            return None
        selected = int(rows[torch.randint(rows.numel(), ()).item()])
        score = max(0.5, candidate_scores[selected])
        return torch.cat(
            (boxes[selected].clamp(0.0, 1.0), detection.new_tensor([score, 1.0]))
        )

    def _occlusion_unit(self, example: Dict[str, Any], index: int) -> float:
        if not self.cfg.deterministic_occlusion:
            return float(torch.rand(()).item())
        key = (
            f"{self.cfg.occlusion_seed}:{example.get('episode_id', '')}:"
            f"{example.get('step_index', index)}"
        ).encode("utf-8")
        return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / float(2**64)

    def _augmentation_generator(
        self, example: Dict[str, Any], index: int, stream: str
    ) -> torch.Generator:
        generator = torch.Generator()
        if self.cfg.deterministic_occlusion:
            key = (
                f"{self.cfg.occlusion_seed}:{stream}:{example.get('episode_id', '')}:"
                f"{example.get('step_index', index)}"
            ).encode("utf-8")
            seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2**63 - 1)
        else:
            seed = int(torch.randint(0, 2**31 - 1, ()).item())
        generator.manual_seed(seed)
        return generator

    @staticmethod
    def _uniform(
        generator: torch.Generator, low: float, high: float
    ) -> float:
        return float(low + (high - low) * torch.rand((), generator=generator).item())

    def _sample_shared_se2(
        self, example: Dict[str, Any], index: int
    ) -> Tuple[float, torch.Tensor]:
        generator = self._augmentation_generator(example, index, "shared_se2")
        max_angle = math.radians(float(self.cfg.pose_global_rotation_max_deg))
        max_translation = float(self.cfg.pose_global_translation_max_m)
        angle = self._uniform(generator, -max_angle, max_angle)
        translation = torch.tensor(
            [
                self._uniform(generator, -max_translation, max_translation),
                self._uniform(generator, -max_translation, max_translation),
            ],
            dtype=torch.float32,
        )
        return angle, translation

    def _sample_receiver_pose_perturbation(
        self, example: Dict[str, Any], index: int, stage: Dict[str, Any]
    ) -> torch.Tensor:
        """Return bounded physical receiver relocation for full-current masks."""

        generator = self._augmentation_generator(example, index, "receiver_pose")
        if torch.rand((), generator=generator).item() >= float(
            stage.get("pose_perturb_probability", 0.0)
        ):
            return torch.zeros(3, dtype=torch.float32)
        radius = self._uniform(
            generator, 0.0, float(stage.get("pose_translation_max_m", 0.0))
        )
        yaw_limit = math.radians(float(stage.get("pose_yaw_max_deg", 0.0)))
        yaw = self._uniform(generator, -yaw_limit, yaw_limit)
        direction = self._uniform(generator, -math.pi, math.pi)
        return torch.tensor(
            [radius * math.cos(direction), radius * math.sin(direction), yaw],
            dtype=torch.float32,
        )

    def _sample_synthetic_occlusion(
        self, example: Dict[str, Any], index: int, candidate: bool
    ) -> torch.Tensor:
        output = torch.zeros(NUM_AGENTS, dtype=torch.bool)
        if not candidate:
            return output
        stage = self._current_curriculum_stage()
        assistance_probability = float(stage.get("assistance_probability", 1.0))
        unit = self._occlusion_unit(example, index)
        if unit >= assistance_probability:
            return output
        drone_weight = float(self.cfg.synthetic_drone_occlusion_prob)
        dog_weight = float(self.cfg.synthetic_dog_occlusion_prob)
        total_weight = drone_weight + dog_weight
        if total_weight <= 0.0:
            return output
        conditional = unit / max(assistance_probability, 1.0e-12)
        if conditional < drone_weight / total_weight:
            output[DRONE] = True
        else:
            output[ROBOTDOG] = True
        return output

    def _sample_corruption_mode(
        self, example: Dict[str, Any], index: int, stage: Dict[str, Any]
    ) -> int:
        probabilities = (
            float(stage.get("roi_only_probability", 0.0)),
            float(stage.get("current_full_probability", 0.0)),
            float(stage.get("recent_full_probability", 0.0)),
            float(stage.get("all_full_probability", 0.0)),
        )
        total = sum(probabilities)
        if total <= 0.0:
            raise ValueError("A curriculum stage must enable one corruption mode")
        generator = self._augmentation_generator(example, index, "corruption_mode")
        unit = torch.rand((), generator=generator).item() * total
        cumulative = 0.0
        modes = (
            CORRUPTION_ROI_ONLY,
            CORRUPTION_CURRENT_FULL,
            CORRUPTION_RECENT_FULL,
            CORRUPTION_ALL_FULL,
        )
        for mode, probability in zip(modes, probabilities):
            cumulative += probability
            if unit < cumulative:
                return mode
        return modes[-1]

    def _sample_recent_history_length(
        self, example: Dict[str, Any], index: int, stage: Dict[str, Any]
    ) -> int:
        lengths = [int(value) for value in stage.get("recent_history_lengths", [2])]
        lengths = [value for value in lengths if value > 0]
        if not lengths:
            raise ValueError("recent_history_lengths must contain a positive value")
        generator = self._augmentation_generator(example, index, "history_length")
        selected = int(torch.randint(len(lengths), (), generator=generator).item())
        return lengths[selected]

    @staticmethod
    def _bbox_token_mask(
        bbox: torch.Tensor, token_count: int, expand_ratio: float
    ) -> torch.Tensor:
        side = int(round(math.sqrt(token_count)))
        if side * side != token_count:
            raise ValueError("fine token count must form a square grid")
        cx, cy, width, height = bbox.float()
        half_width = 0.5 * width.clamp_min(0.0) * float(expand_ratio)
        half_height = 0.5 * height.clamp_min(0.0) * float(expand_ratio)
        centers = (torch.arange(side, dtype=torch.float32) + 0.5) / side
        yy, xx = torch.meshgrid(centers, centers, indexing="ij")
        mask = (
            (xx >= cx - half_width)
            & (xx <= cx + half_width)
            & (yy >= cy - half_height)
            & (yy <= cy + half_height)
        ).reshape(-1)
        if not bool(mask.any()):
            nearest_x = int((cx * side).long().clamp(0, side - 1))
            nearest_y = int((cy * side).long().clamp(0, side - 1))
            mask[nearest_y * side + nearest_x] = True
        return mask

    def _task_type(self, example: Dict[str, Any]) -> str:
        if str(example.get("task_variant", "")).startswith("at_"):
            return "at"
        current = str(
            example.get("agent1_current")
            or self._agent_record(example, "drone", 1).get("current")
            or ""
        )
        parts = Path(current).parts
        if "frames" in parts and parts.index("frames") + 1 < len(parts):
            return str(parts[parts.index("frames") + 1]).lower()
        return "stt"

    def _initial_target_reference(
        self, index: int, example: Dict[str, Any], sample: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feature_dim = int(sample["fine_tokens"].size(-1))
        empty = torch.zeros(NUM_AGENTS, feature_dim, dtype=torch.float32)
        invalid = torch.zeros(NUM_AGENTS, dtype=torch.bool)
        # One shared temporal-identity mechanism is trained jointly on STT,
        # DT, and AT.  Their instructions define the target differently, but
        # all three benefit from preserving the episode-start visual identity.
        # Step zero intentionally has no reference, so grounding cannot receive
        # a circular GT identity answer on the frame where memory is acquired.
        if int(example.get("step_index", 0)) <= 0:
            return empty, invalid

        source_path: Optional[str] = None
        if self._lazy and self._index is not None and self._files is not None:
            source_id = int(self._index[index][0])
            source_path = self._files[source_id]
        cache_key = source_path or str(example.get("episode_id", index))
        cached = self._initial_reference_cache.get(cache_key)
        if cached is not None:
            self._initial_reference_cache.move_to_end(cache_key)
            return cached[0].clone(), cached[1].clone()

        if source_path is not None:
            with open(source_path, "rb") as handle:
                first_line = next(line for line in handle if line.strip())
            first = json.loads(first_line.decode("utf-8"))
        else:
            # In-memory test datasets can only provide a reference when the
            # current example itself is the episode start.
            return empty, invalid

        references = empty.clone()
        reference_valid = invalid.clone()
        for agent_id, (name, legacy_index) in enumerate(
            (("drone", 1), ("robotdog", 2))
        ):
            record = self._agent_record(first, name, legacy_index)
            current = record.get("current") or first.get(f"agent{legacy_index}_current")
            bbox = record.get("bbox") or first.get(f"agent{legacy_index}_bbox")
            bbox_valid = bool(
                record.get(
                    "bbox_valid_mask",
                    first.get(f"agent{legacy_index}_bbox_valid_mask", bbox is not None),
                )
            )
            if not current or bbox is None or not bbox_valid:
                continue
            frame_path = base_train.resolve_multi_agent_path(self.base_root, str(current))
            fine = self._load_or_encode_tokens(frame_path, fine=True).float()
            mask = self._bbox_token_mask(
                torch.as_tensor(bbox, dtype=torch.float32), fine.size(0), 1.0
            )
            references[agent_id] = fine[mask].mean(dim=0)
            reference_valid[agent_id] = True

        self._initial_reference_cache[cache_key] = (references, reference_valid)
        self._initial_reference_cache.move_to_end(cache_key)
        # Keep episode references resident within a worker.  A tiny LRU would
        # thrash under shuffled joint training and repeatedly reload frame zero.
        while len(self._initial_reference_cache) > 16384:
            self._initial_reference_cache.popitem(last=False)
        return references.clone(), reference_valid.clone()

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = super().__getitem__(index)
        example = self.get_example(index)
        order = list(example.get("agent_order", ["drone", "robotdog"]))
        if order != ["drone", "robotdog"]:
            raise ValueError(f"Unexpected agent order {order}; expected drone, robotdog")
        drone = self._agent_record(example, "drone", 1)
        dog = self._agent_record(example, "robotdog", 2)
        drone_perception = self._load_perception(
            str(drone.get("current") or sample["current_path"][DRONE])
        )
        dog_perception = self._load_perception(
            str(dog.get("current") or sample["current_path"][ROBOTDOG])
        )
        detection_feat = torch.stack((drone_perception[0], dog_perception[0]))
        candidate_top_k = max(1, int(getattr(self.cfg, "candidate_top_k", 8)))
        candidate_feat = torch.zeros(NUM_AGENTS, candidate_top_k, 6, dtype=torch.float32)
        candidate_valid = torch.zeros(NUM_AGENTS, candidate_top_k, dtype=torch.bool)
        for agent_id, perception in enumerate((drone_perception, dog_perception)):
            rows = perception[5][:candidate_top_k]
            if rows.numel():
                candidate_feat[agent_id, : rows.size(0)] = rows
                candidate_valid[agent_id, : rows.size(0)] = rows[:, 5] > 0.5
                detection_feat[agent_id] = rows[0]
        perception_grid = torch.stack((drone_perception[1], dog_perception[1]))
        perception_cache_valid = torch.stack((drone_perception[2], dog_perception[2]))
        raw_agent_poses, agent_pose_valid = agent_poses_from_unreal(
            drone.get("pose"), dog.get("pose")
        )

        gt_visible = sample["visible"].bool()
        gt_bbox = sample["bbox_feat"].float()
        bbox_valid_mask = sample["bbox_valid_mask"].bool()
        candidate_iou = torch.zeros(NUM_AGENTS, candidate_top_k, dtype=torch.float32)
        candidate_match_label = torch.zeros(NUM_AGENTS, candidate_top_k, dtype=torch.bool)
        candidate_match_valid = candidate_valid.clone()
        for agent_id in range(NUM_AGENTS):
            if bool(gt_visible[agent_id] and bbox_valid_mask[agent_id]):
                candidate_iou[agent_id] = cxcywh_iou(
                    candidate_feat[agent_id, :, :4], gt_bbox[agent_id].view(1, 4).expand(candidate_top_k, -1)
                )
                candidate_match_label[agent_id] = candidate_iou[agent_id] >= float(self.cfg.target_match_iou_threshold)
        candidate_match_valid &= ((~gt_visible) | bbox_valid_mask).unsqueeze(-1)
        if self.cfg.shuffle_candidates and candidate_top_k > 1:
            generator=self._augmentation_generator(example,index,'candidate_order')
            order=torch.randperm(candidate_top_k,generator=generator)
            candidate_feat=candidate_feat[:,order]
            candidate_valid=candidate_valid[:,order]
            candidate_iou=candidate_iou[:,order]
            candidate_match_label=candidate_match_label[:,order]
            candidate_match_valid=candidate_match_valid[:,order]
        synthetic_false_positive = torch.zeros(NUM_AGENTS, dtype=torch.bool)
        for agent_id, perception in enumerate((drone_perception, dog_perception)):
            can_replace = bool(
                gt_visible[agent_id]
                and detection_feat[agent_id, 5] > 0.5
                and bbox_valid_mask[agent_id]
            )
            if (
                candidate_top_k == 1
                and can_replace
                and torch.rand(()).item() < float(self.cfg.synthetic_false_positive_prob)
            ):
                replacement = self._sample_false_positive_detection(
                    detection_feat[agent_id],
                    perception[3],
                    perception[4],
                    gt_bbox[agent_id],
                )
                if replacement is not None:
                    detection_feat[agent_id] = replacement
                    synthetic_false_positive[agent_id] = True

        detector_visible = detection_feat[:, 5] > 0.5
        yolo_target_iou = cxcywh_iou(detection_feat[:, :4], gt_bbox)
        # The verifier answers a conditional question: given a valid YOLO person
        # proposal, is that proposal the tracked target?  A valid proposal that
        # does not overlap the visible GT target is a hard negative.  GT boxes
        # are labels only and are never forwarded to the model.
        # An invisible target is a valid negative without a box.  A target marked
        # visible but lacking a valid GT box is ambiguous and must not become a
        # hard negative merely because YOLO produced a person proposal.
        target_label_valid = (~gt_visible) | bbox_valid_mask
        target_match_valid = detector_visible & perception_cache_valid & target_label_valid
        target_match_label = (
            gt_visible
            & detector_visible
            & bbox_valid_mask
            & (yolo_target_iou >= float(self.cfg.target_match_iou_threshold))
        )
        # Top-K, not the detector's confidence-ranked first box, defines
        # whether the tracked target is available.  This is the central Top-K
        # contract: a target at rank 2..K must still supervise SELF navigation.
        candidate_target_available = (
            gt_visible
            & bbox_valid_mask
            & candidate_match_label.any(dim=-1)
        )
        verified_detector_visible = candidate_target_available
        synthetic_candidate = bool(
            candidate_target_available.all()
            and perception_cache_valid.all()
            and agent_pose_valid
        )
        synthetic_occlusion = self._sample_synthetic_occlusion(
            example, index, synthetic_candidate
        )
        stage = self._current_curriculum_stage()
        corruption_mode = torch.full(
            (NUM_AGENTS,), CORRUPTION_NATURAL, dtype=torch.long
        )
        history_mask_frames = torch.zeros(NUM_AGENTS, dtype=torch.long)
        coarse_missing_mask = torch.zeros_like(sample["coarse_tidx"], dtype=torch.bool)
        fine_missing_mask = torch.zeros(
            sample["fine_tokens"].shape[:2], dtype=torch.bool
        )
        receiver_rows = torch.nonzero(synthetic_occlusion, as_tuple=False).flatten()
        if receiver_rows.numel():
            receiver = int(receiver_rows[0])
            mode = self._sample_corruption_mode(example, index, stage)
            corruption_mode[receiver] = mode
            if mode == CORRUPTION_ROI_ONLY:
                fine_missing_mask[receiver] = self._bbox_token_mask(
                    detection_feat[receiver, :4],
                    sample["fine_tokens"].size(1),
                    float(self.cfg.target_roi_expand_ratio),
                )
            elif mode == CORRUPTION_CURRENT_FULL:
                fine_missing_mask[receiver] = True
            elif mode == CORRUPTION_RECENT_FULL:
                fine_missing_mask[receiver] = True
                history_length = min(
                    int(self.cfg.history),
                    self._sample_recent_history_length(example, index, stage),
                )
                history_mask_frames[receiver] = history_length
                coarse_missing_mask[receiver] = (
                    sample["coarse_tidx"][receiver]
                    >= int(self.cfg.history) - history_length
                )
            elif mode == CORRUPTION_ALL_FULL:
                fine_missing_mask[receiver] = True
                coarse_missing_mask[receiver] = True
                history_mask_frames[receiver] = int(self.cfg.history)
            else:
                raise RuntimeError(f"Unknown receiver corruption mode {mode}")

        receiver_pose_perturbation = torch.zeros(NUM_AGENTS, 3, dtype=torch.float32)
        shared_rotation_rad = 0.0
        shared_translation_m = torch.zeros(2, dtype=torch.float32)
        agent_poses = raw_agent_poses.clone()
        if agent_pose_valid:
            shared_rotation_rad, shared_translation_m = self._sample_shared_se2(
                example, index
            )
            agent_poses, _ = transform_agent_poses_shared_se2(
                raw_agent_poses, shared_rotation_rad, shared_translation_m
            )
            if receiver_rows.numel():
                receiver = int(receiver_rows[0])
                # ROI_ONLY retains a current background captured at the clean
                # receiver pose, so a counterfactual physical relocation would
                # make the visual and pose inputs inconsistent.  Pose relocation
                # is allowed only when the complete current observation is gone.
                if int(corruption_mode[receiver]) in {
                    CORRUPTION_CURRENT_FULL,
                    CORRUPTION_RECENT_FULL,
                    CORRUPTION_ALL_FULL,
                }:
                    delta = self._sample_receiver_pose_perturbation(
                        example, index, stage
                    )
                    receiver_pose_perturbation[receiver] = delta
                    agent_poses[receiver] = perturb_receiver_pose(
                        agent_poses[receiver],
                        float(delta[0]),
                        float(delta[1]),
                        float(delta[2]),
                    )
        # During training the GT-derived match label supplies the routing target.
        # At inference the LLM target-match head replaces this label.
        effective_visible = (
            verified_detector_visible
            & perception_cache_valid
            & ~synthetic_occlusion
        )
        cooperative_target = (
            ~effective_visible
            & effective_visible.flip(0)
            & perception_cache_valid.all()
            & target_label_valid.all()
            & agent_pose_valid
        )
        # Synthetic corruption is confined to the cooperative stream.  Both
        # self rows still receive clean visual/detection inputs and therefore
        # retain clean self-action and VERIFY supervision.
        self_target = verified_detector_visible & perception_cache_valid
        # A physically relocated receiver starts at its own origin and moves
        # toward the transformed clean reference under per-step motion limits.
        # It is never asked to absorb the full relocation at waypoint one.
        cooperative_waypoints = build_cooperative_waypoint_targets(
            sample["waypoints"],
            synthetic_occlusion,
            receiver_pose_perturbation,
            dt=float(sample["dt"]),
            drone_max_speed_mps=float(self.cfg.drone_max_speed_mps),
            dog_max_speed_mps=float(self.cfg.dog_max_speed_mps),
            drone_max_yaw_rate_rps=float(self.cfg.drone_max_yaw_rate_rps),
            dog_max_yaw_rate_rps=float(self.cfg.dog_max_yaw_rate_rps),
        )
        target_pose = sample["relative_pose"].clone()
        for agent_id in range(NUM_AGENTS):
            if bool(receiver_pose_perturbation[agent_id].abs().max() > 1.0e-8):
                target_pose[agent_id] = transform_relative_target_pose_to_receiver(
                    target_pose[agent_id], receiver_pose_perturbation[agent_id]
                )
        target_pose_valid = sample["relative_pose_valid"].clone()
        if self.cfg.use_target_reference:
            target_reference_tokens, target_reference_valid = self._initial_target_reference(
                index, example, sample
            )
            # Shared dropout prevents the jointly trained matcher from treating a
            # perfect reference as mandatory and ignoring STT/DT/AT instructions.
            reference_dropout = float(self.cfg.target_reference_dropout_prob)
            if reference_dropout > 0.0 and bool(target_reference_valid.any()):
                generator = self._augmentation_generator(example, index, "reference_dropout")
                if float(torch.rand((), generator=generator).item()) < reference_dropout:
                    target_reference_tokens.zero_()
                    target_reference_valid.zero_()
        else:
            # 保留 reference 实现以供后续消融，但本实验既不加载也不传入身份 reference。
            target_reference_tokens = torch.zeros(
                NUM_AGENTS, sample["fine_tokens"].size(-1), dtype=sample["fine_tokens"].dtype
            )
            target_reference_valid = torch.zeros(NUM_AGENTS, dtype=torch.bool)

        sample.update(
            {
                "detection_feat": detection_feat,
                "candidate_feat": candidate_feat,
                "candidate_valid": candidate_valid,
                "candidate_iou": candidate_iou,
                "candidate_match_label": candidate_match_label,
                "candidate_match_valid": candidate_match_valid,
                "candidate_target_available": candidate_target_available,
                "target_reference_tokens": target_reference_tokens,
                "target_reference_valid": target_reference_valid,
                "task_type": self._task_type(example),
                "perception_grid": perception_grid,
                "perception_cache_valid": perception_cache_valid,
                "yolo_target_iou": yolo_target_iou,
                "bbox_valid_mask": bbox_valid_mask,
                "target_match_label": target_match_label,
                "target_match_valid": target_match_valid,
                "synthetic_false_positive": synthetic_false_positive,
                "agent_poses": agent_poses,
                "agent_pose_valid": agent_pose_valid,
                "shared_se2": torch.tensor(
                    [
                        shared_rotation_rad,
                        float(shared_translation_m[0]),
                        float(shared_translation_m[1]),
                    ],
                    dtype=torch.float32,
                ),
                "receiver_pose_perturbation": receiver_pose_perturbation,
                "synthetic_occlusion": synthetic_occlusion,
                "receiver_corruption_mode": corruption_mode,
                "receiver_history_mask_frames": history_mask_frames,
                "coarse_missing_mask": coarse_missing_mask,
                "fine_missing_mask": fine_missing_mask,
                "cooperative_waypoints": cooperative_waypoints,
                "effective_visible": effective_visible,
                "self_target": self_target,
                "cooperative_target": cooperative_target,
                "jepa_valid": fine_missing_mask.any(dim=-1),
                "target_pose": target_pose,
                "target_pose_valid": target_pose_valid,
            }
        )
        return sample


class RotatingTemporalStrideDistributedSampler(torch.utils.data.Sampler[int]):
    """Subsample adjacent frames while rotating the episode-local offset.

    Each episode receives its own deterministic random permutation of temporal
    offsets. Thus ``stride`` epochs cover every frame exactly once, without all
    STT/DT/AT episodes sharing the same phase in an epoch.
    The sampled order remains block-local for vision-cache reuse and is padded
    to an equal, fixed length on all DDP ranks.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        block_size: int,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if num_replicas <= 0 or not 0 <= rank < num_replicas:
            raise ValueError(f"Invalid sampler rank={rank}/{num_replicas}")
        self.dataset = dataset
        self.block_size = int(block_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        cfg = getattr(dataset, "cfg", None)
        self.temporal_stride = int(getattr(cfg, "temporal_stride", 1))
        if self.temporal_stride <= 0:
            raise ValueError("temporal_stride must be positive")
        self._episode_groups = self._build_episode_groups()
        samples_per_epoch = math.ceil(len(dataset) / self.temporal_stride)
        self.num_samples = math.ceil(samples_per_epoch / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        if self.rank == 0:
            print(
                "[SAMPLER][AIRGROUND_V3] "
                f"temporal_stride={self.temporal_stride} "
                f"samples_per_epoch={self.total_size} randomized_episode_offset=True",
                flush=True,
            )

    def _build_episode_groups(self) -> List[List[int]]:
        lazy_index = getattr(self.dataset, "_index", None)
        if lazy_index is None or len(lazy_index) == 0:
            return [list(range(len(self.dataset)))]
        groups: List[List[int]] = []
        start = 0
        while start < len(lazy_index):
            source = lazy_index[start][0]
            end = start + 1
            while end < len(lazy_index) and lazy_index[end][0] == source:
                end += 1
            groups.append(list(range(start, end)))
            start = end
        return groups

    def __iter__(self) -> Iterator[int]:
        cycle = self.epoch // self.temporal_stride
        phase = self.epoch % self.temporal_stride
        offsets: List[int] = []
        for group_id in range(len(self._episode_groups)):
            generator = torch.Generator()
            generator.manual_seed(
                self.seed + cycle * 1_000_003 + group_id * 97
            )
            offsets.append(
                int(torch.randperm(self.temporal_stride, generator=generator)[phase])
            )
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        selected_indices = [
            index
            for group_id, group in enumerate(self._episode_groups)
            for index in group[offsets[group_id] :: self.temporal_stride]
        ]
        if self.block_size == 1:
            # MiniCPM-Robot style: concatenate all task datasets first, then
            # globally shuffle individual samples.  This lets every batch mix
            # STT/DT/AT naturally according to their sample counts.
            order = (
                torch.randperm(len(selected_indices), generator=generator).tolist()
                if self.shuffle
                else list(range(len(selected_indices)))
            )
            indices = [selected_indices[index] for index in order]
        else:
            blocks = [
                selected_indices[start : start + self.block_size]
                for start in range(0, len(selected_indices), self.block_size)
            ]
            order = (
                torch.randperm(len(blocks), generator=generator).tolist()
                if self.shuffle
                else list(range(len(blocks)))
            )
            indices = [index for block_id in order for index in blocks[block_id]]
        if not indices:
            return iter(())
        if len(indices) < self.total_size:
            padding = self.total_size - len(indices)
            indices += (indices * math.ceil(padding / len(indices)))[:padding]
        else:
            indices = indices[: self.total_size]
        start = self.rank * self.num_samples
        return iter(indices[start : start + self.num_samples])

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        setter = getattr(self.dataset, "set_epoch", None)
        if callable(setter):
            setter(epoch)


def collate_airground_v3_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    output = _BASE_COLLATE(batch)
    output["task_type"] = [str(item["task_type"]) for item in batch]
    for key in (
        "detection_feat",
        "candidate_feat",
        "candidate_valid",
        "candidate_iou",
        "candidate_match_label",
        "candidate_match_valid",
        "candidate_target_available",
        "target_reference_tokens",
        "target_reference_valid",
        "perception_grid",
        "perception_cache_valid",
        "yolo_target_iou",
        "bbox_valid_mask",
        "target_match_label",
        "target_match_valid",
        "synthetic_false_positive",
        "agent_poses",
        "agent_pose_valid",
        "shared_se2",
        "receiver_pose_perturbation",
        "synthetic_occlusion",
        "receiver_corruption_mode",
        "receiver_history_mask_frames",
        "coarse_missing_mask",
        "fine_missing_mask",
        "cooperative_waypoints",
        "effective_visible",
        "self_target",
        "cooperative_target",
        "jepa_valid",
        "target_pose",
        "target_pose_valid",
    ):
        output[key] = torch.stack([item[key] for item in batch])
    return output


def apply_airground_v3_defaults(
    cfg: AirGroundV3TrainConfig,
) -> AirGroundV3TrainConfig:
    if cfg.action_dims != 3 or cfg.n_waypoints < 3:
        raise ValueError("V3 requires action_dims=3 and n_waypoints>=3")
    probabilities = (
        cfg.train_synthetic_drone_occlusion_prob,
        cfg.train_synthetic_dog_occlusion_prob,
        cfg.val_synthetic_drone_occlusion_prob,
        cfg.val_synthetic_dog_occlusion_prob,
        cfg.train_synthetic_false_positive_prob,
        cfg.val_synthetic_false_positive_prob,
    )
    if any(not 0.0 <= float(value) <= 1.0 for value in probabilities):
        raise ValueError("Every synthetic augmentation probability must be in [0,1]")
    if (
        cfg.train_synthetic_drone_occlusion_prob
        + cfg.train_synthetic_dog_occlusion_prob
        > 1.0
        or cfg.val_synthetic_drone_occlusion_prob
        + cfg.val_synthetic_dog_occlusion_prob
        > 1.0
    ):
        raise ValueError("Drone+dog occlusion probabilities must not exceed 1")
    if cfg.num_modes <= 0:
        raise ValueError("num_modes must be positive")
    pose_nonnegative = (
        cfg.pose_global_rotation_max_deg,
        cfg.pose_global_translation_max_m,
    )
    if any(float(value) < 0.0 for value in pose_nonnegative):
        raise ValueError("V3 pose augmentation magnitudes must be non-negative")
    if cfg.target_roi_expand_ratio <= 0.0:
        raise ValueError("target_roi_expand_ratio must be positive")
    stages = list(cfg.receiver_corruption_curriculum)
    if not stages:
        raise ValueError("receiver_corruption_curriculum must not be empty")
    previous_end = 0
    for index, raw_stage in enumerate(stages):
        stage = dict(raw_stage)
        end_epoch = int(stage.get("end_epoch", 0))
        if end_epoch <= previous_end:
            raise ValueError("curriculum end_epoch values must strictly increase")
        previous_end = end_epoch
        probability_names = (
            "assistance_probability",
            "roi_only_probability",
            "current_full_probability",
            "recent_full_probability",
            "all_full_probability",
            "pose_perturb_probability",
        )
        for name in probability_names:
            value = float(stage.get(name, 0.0))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"curriculum stage {index} {name} must be in [0,1]")
        mode_total = sum(
            float(stage.get(name, 0.0))
            for name in (
                "roi_only_probability",
                "current_full_probability",
                "recent_full_probability",
                "all_full_probability",
            )
        )
        if mode_total <= 0.0:
            raise ValueError(f"curriculum stage {index} has no corruption mode")
        if float(stage.get("pose_translation_max_m", 0.0)) < 0.0:
            raise ValueError("pose_translation_max_m must be non-negative")
        if float(stage.get("pose_yaw_max_deg", 0.0)) < 0.0:
            raise ValueError("pose_yaw_max_deg must be non-negative")
        lengths = [int(value) for value in stage.get("recent_history_lengths", [])]
        if float(stage.get("recent_full_probability", 0.0)) > 0.0 and not any(
            value > 0 for value in lengths
        ):
            raise ValueError("recent-full curriculum stages need a positive history length")
    if cfg.cooperative_target_frame_version != COOPERATIVE_TARGET_FRAME_VERSION:
        raise ValueError("Wrong cooperative_target_frame_version for receiver recovery V3")
    if cfg.receiver_corruption_version != RECEIVER_CORRUPTION_VERSION:
        raise ValueError("Wrong receiver_corruption_version for receiver recovery V3")
    if cfg.relative_pose_version != RELATIVE_POSE_VERSION:
        raise ValueError("Wrong relative_pose_version for receiver recovery V3")
    if cfg.pose_position_scale_m <= 0.0:
        raise ValueError("pose_position_scale_m must be positive")
    if cfg.train_temporal_stride <= 0:
        raise ValueError("train_temporal_stride must be positive")
    if cfg.perception_grid_size <= 0:
        raise ValueError("perception_grid_size must be positive")
    if cfg.coop_hidden_dim <= 0 or cfg.coop_num_heads <= 0:
        raise ValueError("Cooperative decoder dimensions must be positive")
    if cfg.coop_hidden_dim % cfg.coop_num_heads:
        raise ValueError("coop_hidden_dim must be divisible by coop_num_heads")
    if cfg.coop_decoder_layers <= 0 or cfg.coop_encoder_layers < 0:
        raise ValueError("Cooperative decoder needs >=1 layer and encoder >=0 layers")
    if cfg.jepa_hidden_dim <= 0 or cfg.jepa_num_heads <= 0:
        raise ValueError("JEPA dimensions must be positive")
    if cfg.jepa_hidden_dim % cfg.jepa_num_heads or cfg.jepa_decoder_layers <= 0:
        raise ValueError("Invalid JEPA heads/layers")
    if not 0.0 <= cfg.detection_confidence_threshold <= 1.0:
        raise ValueError("detection_confidence_threshold must be in [0,1]")
    if not 0.0 <= cfg.target_match_iou_threshold <= 1.0:
        raise ValueError("target_match_iou_threshold must be in [0,1]")
    if not 0.0 <= cfg.target_match_confidence_threshold <= 1.0:
        raise ValueError("target_match_confidence_threshold must be in [0,1]")
    if not cfg.drone_target_verification_prompt.strip():
        raise ValueError("drone_target_verification_prompt must not be empty")
    if not cfg.dog_target_verification_prompt.strip():
        raise ValueError("dog_target_verification_prompt must not be empty")
    if cfg.dog_lateral_loss_weight <= 0.0:
        raise ValueError(
            "V3 requires dog_lateral_loss_weight>0 because waypoint y is a "
            "supervised future position, not an executable lateral velocity"
        )
    if cfg.drone_max_speed_mps <= 0.0 or cfg.dog_max_speed_mps <= 0.0:
        raise ValueError("V3 kinematic maximum speeds must be positive")
    loss_fields = (
        "beta_nav",
        "beta_cooperative_waypoint",
        "beta_mode_classification",
        "beta_jepa",
        "beta_target_belief",
        "beta_target_match",
        "beta_uncertainty",
        "beta_smoothness",
        "beta_kinematics",
        "beta_diversity",
        "beta_obstacle",
    )
    for name in loss_fields:
        if float(getattr(cfg, name)) < 0.0:
            raise ValueError(f"{name} must be non-negative")
    if cfg.beta_obstacle > 0.0 and not cfg.enable_projected_obstacle_loss:
        raise ValueError(
            "beta_obstacle>0 requires a validated image-to-local-ground projection. "
            "The current YOLO mask grid is in image coordinates; keep beta_obstacle=0 "
            "until projected obstacle supervision is added."
        )
    if cfg.enable_projected_obstacle_loss:
        raise NotImplementedError(
            "Projected obstacle geometry is not present in the current dataset. "
            "Do not compare local waypoints directly with image-space masks."
        )
    if cfg.base_model or cfg.separate_agent_context:
        raise ValueError("V3 controls its three streams explicitly")
    if cfg.train_agent not in {"both", "drone", "robotdog"}:
        raise ValueError(
            "train_agent must be one of 'both', 'drone', or 'robotdog', "
            f"got {cfg.train_agent!r}"
        )
    cfg.base_model = False
    cfg.base_model_architecture = ARCHITECTURE
    cfg.use_roi_tokens = False
    cfg.use_bbox_tokens = False
    cfg.use_bbox_text_prompt = False
    cfg.use_grounding = False
    cfg.use_visual_section_markers = False
    cfg.beta_bbox = 0.0
    cfg.beta_visible = 0.0
    cfg.beta_relative_pose = 0.0
    cfg.beta_control_drone = 0.0
    cfg.beta_control_dog = 0.0
    cfg.bbox_dropout_prob = 0.0
    cfg.bbox_text_dropout_prob = 0.0
    cfg.return_token_logits = False
    cfg.val_bbox_source = "none"
    cfg.ddp_find_unused_parameters = False
    cfg.model_architecture = ARCHITECTURE
    cfg.model_source = MODEL_SOURCE
    if cfg.joint_instruction_override is None:
        cfg.joint_instruction_override = (
            "Use both aerial and ground observations to follow the person without collision."
        )
    if cfg.lr_scheduler not in {"constant", "cosine"}:
        raise ValueError(f"Unsupported lr_scheduler={cfg.lr_scheduler!r}")
    if cfg.lr <= 0.0 or cfg.min_lr < 0.0 or cfg.min_lr > cfg.lr:
        raise ValueError("Require lr>0 and 0<=min_lr<=lr")
    if cfg.log_every <= 0 or cfg.max_steps < 0:
        raise ValueError("Require log_every>0 and max_steps>=0")
    return cfg


def build_airground_v3_model(cfg: AirGroundV3TrainConfig) -> torch.nn.Module:
    cfg = apply_airground_v3_defaults(cfg)
    model = AirGroundCooperativeVLAV3(
        AirGroundCoopV3ModelConfig(
            llm_name=cfg.llm_name,
            freeze_llm=cfg.freeze_llm,
            n_waypoints=cfg.n_waypoints,
            action_dims=cfg.action_dims,
            num_modes=cfg.num_modes,
            text_max_length=cfg.text_max_length,
            insert_time_tokens=cfg.insert_time_tokens,
            use_angle_tvi=cfg.use_angle_tvi,
            use_agent_text_markers=cfg.use_agent_text_markers,
            use_tanh_actions=not cfg.no_tanh_actions,
            alpha_xy=cfg.alpha_xy,
            perception_grid_size=cfg.perception_grid_size,
            drone_mask_expand_ratio=cfg.drone_mask_expand_ratio,
            dog_mask_expand_ratio=cfg.dog_mask_expand_ratio,
            coop_hidden_dim=cfg.coop_hidden_dim,
            coop_encoder_layers=cfg.coop_encoder_layers,
            coop_decoder_layers=cfg.coop_decoder_layers,
            coop_num_heads=cfg.coop_num_heads,
            coop_dropout=cfg.coop_dropout,
            jepa_hidden_dim=cfg.jepa_hidden_dim,
            jepa_decoder_layers=cfg.jepa_decoder_layers,
            jepa_num_heads=cfg.jepa_num_heads,
            jepa_dropout=cfg.jepa_dropout,
            jepa_momentum=cfg.jepa_momentum,
            detection_confidence_threshold=cfg.detection_confidence_threshold,
            target_match_confidence_threshold=cfg.target_match_confidence_threshold,
            candidate_temporal_iou_weight=cfg.candidate_temporal_iou_weight,
            hard_visibility_routing=cfg.hard_visibility_routing,
            pose_position_scale_m=cfg.pose_position_scale_m,
            drone_target_verification_prompt=cfg.drone_target_verification_prompt,
            dog_target_verification_prompt=cfg.dog_target_verification_prompt,
        ),
        vision_feat_dim=cfg.vision_feat_dim,
    )
    original_load = model.load_state_dict

    def strict_v3_load(state_dict: Dict[str, Any], *args: Any, **kwargs: Any):
        normalized_keys = [str(key).removeprefix("module.") for key in state_dict]
        required = (
            "self_act_tokens",
            "self_verify_tokens",
            "coop_act_tokens",
            "self_planners.0.",
            "self_planners.1.",
            "coop_decoders.0.",
            "coop_decoders.1.",
            "jepa_predictors.0.",
            "jepa_predictors.1.",
            "teacher_proj.",
            "target_belief_heads.0.",
            "target_belief_heads.1.",
            "target_match_heads.0.",
            "target_match_heads.1.",
            "obstacle_grid_proj.",
            "detection_proj.",
            "candidate_roi_proj.",
            "agent_pose_proj.",
            "relative_pose_proj.",
        )
        missing = [
            prefix
            for prefix in required
            if not any(key.startswith(prefix) for key in normalized_keys)
        ]
        if missing:
            raise RuntimeError(
                "Refusing a non-V3/incompatible checkpoint in the V3 model; "
                f"missing V3 keys {missing}"
            )
        return original_load(state_dict, *args, **kwargs)

    model.load_state_dict = strict_v3_load  # type: ignore[method-assign]
    return model


def build_airground_v3_dataset(
    path: str, cfg: AirGroundV3TrainConfig, cache_root: Optional[str] = None
) -> AirGroundV3JsonDataset:
    is_validation = bool(cfg.val_json) and Path(path).resolve() == Path(cfg.val_json).resolve()
    return AirGroundV3JsonDataset(
        AirGroundV3DataConfig(
            train_json=path,
            n_waypoints=cfg.n_waypoints,
            history=cfg.history,
            cache_root=cfg.cache_root if cache_root is None else cache_root,
            action_dims=cfg.action_dims,
            online_encode_missing=cfg.online_encode_missing,
            image_size=cfg.image_size,
            vision_resize_mode=cfg.vision_resize_mode,
            use_roi_tokens=False,
            use_bbox_text_prompt=False,
            coarse_cache_size=cfg.coarse_cache_size,
            global_image_only=False,
            require_recorded_waypoints=True,
            perception_cache_root=cfg.perception_cache_root,
            candidate_cache_root=cfg.candidate_cache_root,
            require_topk_candidate_cache=cfg.require_topk_candidate_cache,
            candidate_top_k=cfg.candidate_top_k,
            shuffle_candidates=not is_validation,
            allow_missing_perception_cache=cfg.allow_missing_perception_cache,
            perception_grid_size=cfg.perception_grid_size,
            target_match_iou_threshold=cfg.target_match_iou_threshold,
            use_target_reference=cfg.use_target_reference,
            target_reference_dropout_prob=(
                cfg.val_target_reference_dropout_prob
                if is_validation
                else cfg.train_target_reference_dropout_prob
            ),
            synthetic_false_positive_prob=(
                cfg.val_synthetic_false_positive_prob
                if is_validation
                else cfg.train_synthetic_false_positive_prob
            ),
            temporal_stride=(1 if is_validation else cfg.train_temporal_stride),
            synthetic_drone_occlusion_prob=(
                cfg.val_synthetic_drone_occlusion_prob
                if is_validation
                else cfg.train_synthetic_drone_occlusion_prob
            ),
            synthetic_dog_occlusion_prob=(
                cfg.val_synthetic_dog_occlusion_prob
                if is_validation
                else cfg.train_synthetic_dog_occlusion_prob
            ),
            deterministic_occlusion=is_validation,
            occlusion_seed=cfg.occlusion_seed,
            target_roi_expand_ratio=cfg.target_roi_expand_ratio,
            receiver_corruption_curriculum=cfg.receiver_corruption_curriculum,
            pose_global_rotation_max_deg=cfg.pose_global_rotation_max_deg,
            pose_global_translation_max_m=cfg.pose_global_translation_max_m,
            drone_max_speed_mps=cfg.drone_max_speed_mps,
            dog_max_speed_mps=cfg.dog_max_speed_mps,
            drone_max_yaw_rate_rps=cfg.drone_max_yaw_rate_rps,
            dog_max_yaw_rate_rps=cfg.dog_max_yaw_rate_rps,
        )
    )


def _masked_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    valid = valid.to(values.device, torch.bool)
    if valid.shape != values.shape:
        valid = torch.broadcast_to(valid, values.shape)
    if valid.any():
        return values[valid].mean()
    # Keep the complete graph connected for DDP even when this batch contains
    # no sample of a particular cooperative direction.
    return values.sum() * 0.0


def _waypoint_losses(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
    dt: torch.Tensor,
    cfg: AirGroundV3TrainConfig,
    alpha_task: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    pred_normalized, gt_normalized = base_train.normalize_multi_agent_xy_by_alpha(
        pred, gt, alpha_task
    )
    return base_train.weighted_multi_agent_waypoint_loss(
        pred_normalized,
        gt_normalized,
        mask,
        dt,
        loss_type=cfg.nav_loss_type,
        smooth_l1_beta=cfg.smooth_l1_beta,
        yaw_weight=cfg.yaw_loss_weight,
        final_weight=cfg.final_waypoint_loss_weight,
        turn_sample_weight=cfg.turn_sample_weight,
        turn_rate_threshold=cfg.turn_rate_threshold,
        turn_angle_threshold=cfg.turn_angle_threshold,
        stop_sample_weight=cfg.stop_sample_weight,
        stop_speed_threshold=cfg.stop_speed_threshold,
        stop_window=cfg.stop_window,
        dog_lateral_loss_weight=cfg.dog_lateral_loss_weight,
    )


def _candidate_waypoint_losses(
    candidates: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
    dt: torch.Tensor,
    cfg: AirGroundV3TrainConfig,
    alpha_task: torch.Tensor,
) -> torch.Tensor:
    """Return behavior-weighted candidate losses with shape ``(B,2,K)``."""

    batch_size, agents, modes, waypoints, action_dims = candidates.shape
    if agents != NUM_AGENTS or action_dims != 3:
        raise ValueError(f"Unexpected cooperative candidate shape {tuple(candidates.shape)}")
    flat_pred = candidates.permute(0, 2, 1, 3, 4).reshape(
        batch_size * modes, agents, waypoints, action_dims
    )
    flat_gt = gt[:, None].expand(-1, modes, -1, -1, -1).reshape_as(flat_pred)
    flat_mask = mask[:, None].expand(-1, modes, -1, -1).reshape(
        batch_size * modes, agents, waypoints
    )
    flat_dt = dt[:, None].expand(-1, modes).reshape(batch_size * modes)
    flat_loss, _ = _waypoint_losses(
        flat_pred, flat_gt, flat_mask, flat_dt, cfg, alpha_task
    )
    return flat_loss.view(batch_size, modes, agents).permute(0, 2, 1)


def _smoothness_per_sample(
    candidates: torch.Tensor, valid_mask: torch.Tensor
) -> torch.Tensor:
    """Second-difference regularizer, averaged over modes, shape ``(B,2)``."""

    first = candidates[..., 1:, :] - candidates[..., :-1, :]
    second_xy = first[..., 1:, :2] - first[..., :-1, :2]
    first_yaw = torch.atan2(torch.sin(first[..., 2]), torch.cos(first[..., 2]))
    second_yaw = first_yaw[..., 1:] - first_yaw[..., :-1]
    second_yaw = torch.atan2(torch.sin(second_yaw), torch.cos(second_yaw))
    value = second_xy.square().sum(dim=-1) + second_yaw.square()
    triple_valid = (
        valid_mask[..., 2:] & valid_mask[..., 1:-1] & valid_mask[..., :-2]
    )
    valid = triple_valid[:, :, None, :].expand_as(value)
    numerator = (value * valid).sum(dim=(-1, -2))
    denominator = valid.sum(dim=(-1, -2)).clamp_min(1)
    return numerator / denominator


def _kinematics_per_sample(
    candidates: torch.Tensor,
    valid_mask: torch.Tensor,
    dt: torch.Tensor,
    cfg: AirGroundV3TrainConfig,
) -> torch.Tensor:
    """Agent-aware speed/yaw and non-holonomic dog constraints, shape ``(B,2)``."""

    delta = candidates[..., 1:, :] - candidates[..., :-1, :]
    # Dataset waypoint 0 is the structural [0,0,0] origin and is deliberately
    # false in valid_mask because it is not a regression target.  It is still a
    # valid kinematic endpoint, otherwise the first executable segment would
    # escape speed/yaw/non-holonomic regularization entirely.
    kinematic_valid = valid_mask.clone()
    kinematic_valid[..., 0] = True
    step_valid = kinematic_valid[..., 1:] & kinematic_valid[..., :-1]
    step_dt = dt.clamp_min(1.0e-4)[:, None, None, None]
    speed = torch.linalg.norm(delta[..., :2], dim=-1) / step_dt
    yaw_delta = torch.atan2(torch.sin(delta[..., 2]), torch.cos(delta[..., 2]))
    yaw_rate = yaw_delta.abs() / step_dt
    max_speed = candidates.new_tensor(
        [cfg.drone_max_speed_mps, cfg.dog_max_speed_mps]
    ).view(1, NUM_AGENTS, 1, 1)
    max_yaw_rate = candidates.new_tensor(
        [cfg.drone_max_yaw_rate_rps, cfg.dog_max_yaw_rate_rps]
    ).view(1, NUM_AGENTS, 1, 1)
    value = F.relu(speed - max_speed).square() + F.relu(
        yaw_rate - max_yaw_rate
    ).square()

    # Transform each displacement into the segment's midpoint body heading.
    # For a feasible constant-curvature unicycle arc this makes the residual
    # lateral component zero (up to discretization), while still penalizing a
    # physically impossible RobotDog sidestep.
    previous_yaw = candidates[:, ROBOTDOG, :, :-1, 2]
    dog_yaw_delta = torch.atan2(
        torch.sin(delta[:, ROBOTDOG, :, :, 2]),
        torch.cos(delta[:, ROBOTDOG, :, :, 2]),
    )
    midpoint_yaw = previous_yaw + 0.5 * dog_yaw_delta
    dog_dx = delta[:, ROBOTDOG, :, :, 0]
    dog_dy = delta[:, ROBOTDOG, :, :, 1]
    dog_lateral = (
        -torch.sin(midpoint_yaw) * dog_dx
        + torch.cos(midpoint_yaw) * dog_dy
    )
    dog_lateral_penalty = F.relu(
        dog_lateral.abs() - float(cfg.dog_lateral_tolerance_m)
    ).square()
    value[:, ROBOTDOG] = value[:, ROBOTDOG] + dog_lateral_penalty
    valid = step_valid[:, :, None, :].expand_as(value)
    numerator = (value * valid).sum(dim=(-1, -2))
    denominator = valid.sum(dim=(-1, -2)).clamp_min(1)
    return numerator / denominator


def _diversity_per_sample(
    candidates: torch.Tensor, margin: float
) -> torch.Tensor:
    """Penalize collapsed final XY endpoints; return shape ``(B,2)``."""

    modes = candidates.size(2)
    if modes <= 1:
        return candidates.sum(dim=(2, 3, 4)) * 0.0
    endpoints = candidates[..., -1, :2]
    distances = torch.linalg.norm(
        endpoints[:, :, :, None] - endpoints[:, :, None, :], dim=-1
    )
    pair_mask = torch.triu(
        torch.ones(modes, modes, dtype=torch.bool, device=candidates.device), diagonal=1
    )
    penalties = F.relu(float(margin) - distances).square()
    return penalties[..., pair_mask].mean(dim=-1)


def forward_airground_v3_loss(
    model: torch.nn.Module,
    batch: Dict[str, Any],
    cfg: AirGroundV3TrainConfig,
    device: torch.device,
):
    instructions = batch["instruction"]
    batch_size = len(instructions)
    joint = [cfg.joint_instruction_override or cfg.instruction_override] * batch_size
    if not (cfg.joint_instruction_override or cfg.instruction_override):
        joint = instructions
    candidate_feat = batch.get("candidate_feat")
    if candidate_feat is None:
        candidate_feat = batch["detection_feat"].unsqueeze(2)
    candidate_valid = batch.get("candidate_valid")
    if candidate_valid is None:
        candidate_valid = candidate_feat[..., 5] > 0.5
    target_reference_tokens = batch.get("target_reference_tokens")
    if target_reference_tokens is None:
        target_reference_tokens = torch.zeros(
            batch["fine_tokens"].size(0),
            NUM_AGENTS,
            batch["fine_tokens"].size(-1),
            dtype=batch["fine_tokens"].dtype,
        )
    target_reference_valid = batch.get("target_reference_valid")
    if target_reference_valid is None:
        target_reference_valid = torch.zeros(
            batch["fine_tokens"].size(0), NUM_AGENTS, dtype=torch.bool
        )
    output = model(
        coarse_tokens=batch["coarse_tokens"].to(device),
        coarse_tidx=batch["coarse_tidx"].to(device),
        fine_tokens=batch["fine_tokens"].to(device),
        fine_tidx=batch["fine_tidx"].to(device),
        detection_feat=batch["detection_feat"].to(device),
        candidate_feat=candidate_feat.to(device),
        candidate_valid=candidate_valid.to(device),
        target_reference_tokens=target_reference_tokens.to(device),
        target_reference_valid=target_reference_valid.to(device),
        perception_grid=batch["perception_grid"].to(device),
        agent_poses=batch["agent_poses"].to(device),
        synthetic_occlusion=batch["synthetic_occlusion"].to(device),
        coarse_missing_mask=batch["coarse_missing_mask"].to(device),
        fine_missing_mask=batch["fine_missing_mask"].to(device),
        instructions=instructions,
        joint_instructions=joint,
        drone_instructions=(
            [cfg.agent1_instruction_override] * batch_size
            if cfg.agent1_instruction_override
            else None
        ),
        dog_instructions=(
            [cfg.agent2_instruction_override] * batch_size
            if cfg.agent2_instruction_override
            else None
        ),
        yaw_hist=batch["yaw_hist"].to(device) if cfg.use_angle_tvi else None,
        yaw_curr=batch["yaw_curr"].to(device) if cfg.use_angle_tvi else None,
        route_visibility=batch["effective_visible"].to(device),
    )

    # Self/VERIFY supervision always uses the original clean local labels.
    # A physically relocated cooperative receiver instead follows the bounded
    # recovery target constructed from the transformed clean-path reference.
    gt = batch["waypoints"].to(device)
    cooperative_gt = batch["cooperative_waypoints"].to(device)
    valid_mask = batch["valid_mask"].to(device).bool()
    dt = batch["dt"].to(device)
    model_plain = (
        model.module
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model
    )
    alpha_task = model_plain.alpha_task

    self_per_agent, _ = _waypoint_losses(
        output["self_waypoints"], gt, valid_mask, dt, cfg, alpha_task
    )
    self_target = batch["self_target"].to(device).bool()
    cooperative_target = batch["cooperative_target"].to(device).bool()
    routed_gt = torch.where(
        cooperative_target[:, :, None, None],
        cooperative_gt,
        gt,
    )
    routed_per_agent, routed_metrics = _waypoint_losses(
        output["waypoints"],
        routed_gt,
        valid_mask,
        dt,
        cfg,
        alpha_task,
    )
    loss_self_drone = _masked_mean(self_per_agent[:, DRONE], self_target[:, DRONE])
    loss_self_dog = _masked_mean(self_per_agent[:, ROBOTDOG], self_target[:, ROBOTDOG])
    agent_weights = gt.new_tensor([cfg.drone_loss_weight, cfg.dog_loss_weight])
    if cfg.train_agent == "drone":
        agent_weights[ROBOTDOG] = 0.0
    elif cfg.train_agent == "robotdog":
        agent_weights[DRONE] = 0.0
    elif cfg.train_agent != "both":
        raise ValueError(f"Unsupported train_agent={cfg.train_agent!r}")
    supervised_agent = agent_weights.gt(0.0)[None, :]
    divisor = (
        float(agent_weights.sum().item()) if cfg.normalize_agent_loss_weights else 1.0
    )
    loss_self = (
        agent_weights[DRONE] * loss_self_drone
        + agent_weights[ROBOTDOG] * loss_self_dog
    ) / divisor

    candidate_losses = _candidate_waypoint_losses(
        output["cooperative_candidates"],
        cooperative_gt,
        valid_mask,
        dt,
        cfg,
        alpha_task,
    )
    best_candidate_loss, best_candidate_index = candidate_losses.min(dim=-1)
    loss_coop_drone = _masked_mean(
        best_candidate_loss[:, DRONE], cooperative_target[:, DRONE]
    )
    loss_coop_dog = _masked_mean(
        best_candidate_loss[:, ROBOTDOG], cooperative_target[:, ROBOTDOG]
    )
    loss_cooperative = (
        agent_weights[DRONE] * loss_coop_drone
        + agent_weights[ROBOTDOG] * loss_coop_dog
    ) / divisor
    mode_cross_entropy = F.cross_entropy(
        output["cooperative_mode_logits"].reshape(-1, cfg.num_modes),
        best_candidate_index.detach().reshape(-1),
        reduction="none",
    ).view(batch_size, NUM_AGENTS)
    loss_mode = _masked_mean(
        mode_cross_entropy, cooperative_target & supervised_agent
    )

    token_cosine = 1.0 - F.cosine_similarity(
        output["jepa_prediction_tokens"].float(),
        output["jepa_teacher_tokens"].float(),
        dim=-1,
    )
    jepa_valid = batch["jepa_valid"].to(device).bool()
    jepa_token_valid = (
        output["jepa_token_mask"].bool()
        & jepa_valid[..., None]
        & supervised_agent[..., None]
    )
    loss_jepa = _masked_mean(token_cosine, jepa_token_valid)

    target_pose = batch["target_pose"].to(device).float()
    target_pose_valid = (
        batch["target_pose_valid"].to(device).bool()
        & cooperative_target
        & supervised_agent
    )
    belief_per_agent = F.smooth_l1_loss(
        output["target_belief"].float(), target_pose, reduction="none"
    ).mean(dim=-1)
    loss_belief = _masked_mean(belief_per_agent, target_pose_valid)
    log_variance = output["jepa_uncertainty_logit"].float().clamp(-5.0, 5.0)
    uncertainty_nll = (
        torch.exp(-log_variance) * belief_per_agent.detach() + log_variance
    )
    loss_uncertainty = _masked_mean(uncertainty_nll, target_pose_valid)

    # 8 个候选分别执行独立二分类验证。最大 IoU 的有效候选是唯一正样本，
    # 其余有效候选为负样本；正负两组等权，避免常见的 1:7 BCE 不平衡。
    if "candidate_class_logits" in output and "candidate_iou" in batch:
        class_logits = output["candidate_class_logits"].float()
        candidate_iou = batch["candidate_iou"].to(device).float()
        candidate_valid = batch["candidate_valid"].to(device).bool()
        masked_iou = candidate_iou.masked_fill(~candidate_valid, -1.0)
        best_iou, best_index = masked_iou.max(dim=-1)
        visible = batch["visible"].to(device).bool()
        bbox_valid = batch["bbox_valid_mask"].to(device).bool()
        has_target = (
            visible
            & bbox_valid
            & (best_iou >= float(cfg.target_match_iou_threshold))
        )
        class_valid = (
            ((~visible) | bbox_valid)
            & batch["perception_cache_valid"].to(device).bool()
            & supervised_agent
        )
        positive_agent = has_target & class_valid
        positive_logit = class_logits.gather(
            -1, best_index.unsqueeze(-1)
        ).squeeze(-1)
        # 候选全部无效时会 gather 到掩码后的有限最小 logit；
        # softplus(-finite_min) 会溢出为 +inf，而 inf*0 会污染 masked mean。
        # 因此必须在计算 softplus 前把无监督行替换为零。
        safe_positive_logit = torch.where(
            positive_agent, positive_logit, torch.zeros_like(positive_logit)
        )
        loss_positive = _masked_mean(
            F.softplus(-safe_positive_logit), positive_agent
        )
        positive_mask = F.one_hot(
            best_index, num_classes=class_logits.size(-1)
        ).bool()
        negative_mask = (
            candidate_valid
            & ~positive_mask
            & positive_agent.unsqueeze(-1)
        )
        loss_negative = _masked_mean(F.softplus(class_logits), negative_mask)
        loss_target_match = 0.5 * (loss_positive + loss_negative)
        loss_candidate_bce = loss_target_match
        loss_candidate_rank = loss_target_match.detach() * 0.0
        with torch.no_grad():
            selected_index = output["candidate_selected_index"]
            selected_is_target = selected_index.eq(best_index)
            no_target_agent = (~has_target) & class_valid
            target_match_accuracy = _masked_mean(
                selected_is_target.float(), positive_agent
            )
            threshold_accept = (
                output["candidate_selected_probability"]
                >= float(cfg.target_match_confidence_threshold)
            ) & candidate_valid.gather(-1, selected_index.unsqueeze(-1)).squeeze(-1)
            no_target_false_accept_rate = _masked_mean(
                threshold_accept.float(), no_target_agent
            )
            target_match_positive_fraction = _masked_mean(
                has_target.float(), class_valid
            )
            eligible_agent = visible & bbox_valid & supervised_agent
            candidate_top8_recall = _masked_mean(
                has_target.float(), eligible_agent
            )
            binary_label = positive_mask & positive_agent.unsqueeze(-1)
            binary_valid = candidate_valid & positive_agent.unsqueeze(-1)
            binary_prediction = torch.sigmoid(class_logits) >= float(
                cfg.target_match_confidence_threshold
            )
            candidate_class_accuracy = _masked_mean(
                binary_prediction.eq(binary_label).float(), binary_valid
            )
            candidate_probability = torch.sigmoid(class_logits)
            candidate_positive_probability = _masked_mean(
                candidate_probability.gather(-1, best_index.unsqueeze(-1)).squeeze(-1),
                positive_agent,
            )
            max_negative_probability = candidate_probability.masked_fill(
                ~negative_mask, -1.0
            ).max(dim=-1).values
            candidate_max_negative_probability = _masked_mean(
                max_negative_probability, negative_mask.any(dim=-1)
            )
            candidate_threshold_target_recall = _masked_mean(
                (threshold_accept & selected_is_target).float(), positive_agent
            )
    else:
        target_match_label = batch["target_match_label"].to(device).float()
        target_match_valid = (
            batch["target_match_valid"].to(device).bool() & supervised_agent
        )
        target_match_bce = F.binary_cross_entropy_with_logits(
            output["target_match_logits"].float(), target_match_label, reduction="none"
        )
        loss_target_match = _masked_mean(target_match_bce, target_match_valid)
        loss_candidate_bce = loss_target_match
        loss_candidate_rank = loss_target_match * 0.0
        with torch.no_grad():
            prediction = output["target_match_probability"] >= float(
                cfg.target_match_confidence_threshold
            )
            target_match_accuracy = _masked_mean(
                prediction.eq(target_match_label.bool()).float(), target_match_valid
            )
            target_match_positive_fraction = _masked_mean(
                target_match_label, target_match_valid
            )
            candidate_top8_recall = target_match_positive_fraction
            no_target_false_accept_rate = loss_target_match.detach() * 0.0
            candidate_class_accuracy = target_match_accuracy
            positive_mask = target_match_label.bool() & target_match_valid
            negative_mask = (~target_match_label.bool()) & target_match_valid
            candidate_positive_probability = _masked_mean(
                output["target_match_probability"], positive_mask
            )
            candidate_max_negative_probability = _masked_mean(
                output["target_match_probability"], negative_mask
            )
            candidate_threshold_target_recall = _masked_mean(
                (prediction & target_match_label.bool()).float(), positive_mask
            )

    smoothness = _smoothness_per_sample(
        output["cooperative_candidates"].float(), valid_mask
    )
    kinematics = _kinematics_per_sample(
        output["cooperative_candidates"].float(), valid_mask, dt, cfg
    )
    diversity = _diversity_per_sample(
        output["cooperative_candidates"].float(), cfg.diversity_margin_m
    )
    regularizer_target = cooperative_target & supervised_agent
    loss_smoothness = _masked_mean(smoothness, regularizer_target)
    loss_kinematics = _masked_mean(kinematics, regularizer_target)
    loss_diversity = _masked_mean(diversity, regularizer_target)
    # Image-space YOLO masks are consumed by the decoder as tokens.  A numerical
    # obstacle collision loss remains exactly zero until a calibrated local-ground
    # projection is available; apply_airground_v3_defaults rejects unsafe enabling.
    loss_obstacle = output["cooperative_candidates"].sum() * 0.0

    loss = (
        cfg.beta_nav * loss_self
        + cfg.beta_cooperative_waypoint * loss_cooperative
        + cfg.beta_mode_classification * loss_mode
        + cfg.beta_jepa * loss_jepa
        + cfg.beta_target_belief * loss_belief
        + cfg.beta_target_match * loss_target_match
        + cfg.beta_uncertainty * loss_uncertainty
        + cfg.beta_smoothness * loss_smoothness
        + cfg.beta_kinematics * loss_kinematics
        + cfg.beta_diversity * loss_diversity
        + cfg.beta_obstacle * loss_obstacle
    )

    xy = routed_metrics["xy_per_agent"]
    yaw = routed_metrics["yaw_per_agent"]
    final = routed_metrics["final_per_agent"]
    weighted_xy = (
        agent_weights[DRONE] * xy[:, DRONE].mean()
        + agent_weights[ROBOTDOG] * xy[:, ROBOTDOG].mean()
    ) / divisor
    weighted_yaw = (
        agent_weights[DRONE] * yaw[:, DRONE].mean()
        + agent_weights[ROBOTDOG] * yaw[:, ROBOTDOG].mean()
    ) / divisor
    weighted_final = (
        agent_weights[DRONE] * final[:, DRONE].mean()
        + agent_weights[ROBOTDOG] * final[:, ROBOTDOG].mean()
    ) / divisor
    loss_nav = loss_self + loss_cooperative
    zero = loss.new_zeros(())
    return loss, {
        # Required by the generic multi-agent loop.
        "loss_nav": loss_nav.detach(),
        "loss_nav_drone": (loss_self_drone + loss_coop_drone).detach(),
        "loss_nav_dog": (loss_self_dog + loss_coop_dog).detach(),
        "loss_nav_xy": weighted_xy.detach(),
        "loss_nav_yaw": weighted_yaw.detach(),
        "loss_nav_final": weighted_final.detach(),
        "loss_control_drone": zero,
        "loss_control_dog": zero,
        "val_xy_mse_drone": xy[:, DRONE].mean().detach(),
        "val_xy_mse_robotdog": xy[:, ROBOTDOG].mean().detach(),
        "val_yaw_mse_drone": yaw[:, DRONE].mean().detach(),
        "val_yaw_mse_robotdog": yaw[:, ROBOTDOG].mean().detach(),
        "val_final_waypoint_error_drone": final[:, DRONE].mean().detach(),
        "val_final_waypoint_error_robotdog": final[:, ROBOTDOG].mean().detach(),
        "turn_fraction": routed_metrics["turn_mask"].float().mean().detach(),
        "stop_fraction": routed_metrics["stop_mask"].float().mean().detach(),
        "turn_fraction_drone": routed_metrics["turn_mask"][:, DRONE].float().mean().detach(),
        "turn_fraction_dog": routed_metrics["turn_mask"][:, ROBOTDOG].float().mean().detach(),
        "stop_fraction_drone": routed_metrics["stop_mask"][:, DRONE].float().mean().detach(),
        "stop_fraction_dog": routed_metrics["stop_mask"][:, ROBOTDOG].float().mean().detach(),
        "behavior_weight_mean": routed_metrics["behavior_weight"].mean().detach(),
        # Compatibility aliases used by the existing terminal/CSV logger.
        "loss_bbox": loss_jepa.detach(),
        "loss_visible": loss_target_match.detach(),
        "loss_relative_pose": loss_belief.detach(),
        "regression_loss": loss_cooperative.detach(),
        "score_loss": loss_mode.detach(),
        "loss_dog_normal": loss_self_dog.detach(),
        "loss_dog_guided": loss_coop_dog.detach(),
        "pred": output["waypoints"].detach(),
        # V3-specific metrics remain available to tests/debug callers.
        "loss_self": loss_self.detach(),
        "loss_self_drone": loss_self_drone.detach(),
        "loss_self_dog": loss_self_dog.detach(),
        "loss_cooperative": loss_cooperative.detach(),
        "loss_coop_drone": loss_coop_drone.detach(),
        "loss_coop_dog": loss_coop_dog.detach(),
        "loss_mode": loss_mode.detach(),
        "loss_jepa": loss_jepa.detach(),
        "loss_belief": loss_belief.detach(),
        "loss_target_match": loss_target_match.detach(),
        "loss_candidate_bce": loss_candidate_bce.detach(),
        "loss_candidate_rank": loss_candidate_rank.detach(),
        "candidate_top8_recall": candidate_top8_recall.detach(),
        "target_match_accuracy": target_match_accuracy.detach(),
        "candidate_class_accuracy": candidate_class_accuracy.detach(),
        "candidate_positive_probability": candidate_positive_probability.detach(),
        "candidate_max_negative_probability": candidate_max_negative_probability.detach(),
        "candidate_threshold_target_recall": candidate_threshold_target_recall.detach(),
        "no_target_false_accept_rate": no_target_false_accept_rate.detach(),
        "target_match_positive_fraction": target_match_positive_fraction.detach(),
        "synthetic_false_positive_fraction": batch["synthetic_false_positive"]
        .float()
        .mean(),
        "loss_uncertainty": loss_uncertainty.detach(),
        "loss_smoothness": loss_smoothness.detach(),
        "loss_kinematics": loss_kinematics.detach(),
        "loss_diversity": loss_diversity.detach(),
        "loss_obstacle": loss_obstacle.detach(),
        "synthetic_drone_fraction": batch["synthetic_occlusion"][:, DRONE].float().mean(),
        "synthetic_dog_fraction": batch["synthetic_occlusion"][:, ROBOTDOG].float().mean(),
        "pose_perturbation_fraction": (
            batch["receiver_pose_perturbation"][..., :2]
            .norm(dim=-1)
            .gt(1.0e-8)
            .float()
            .mean()
        ),
        "roi_only_fraction": batch["receiver_corruption_mode"]
        .eq(CORRUPTION_ROI_ONLY)
        .float()
        .mean(),
        "current_full_fraction": batch["receiver_corruption_mode"]
        .eq(CORRUPTION_CURRENT_FULL)
        .float()
        .mean(),
        "recent_full_fraction": batch["receiver_corruption_mode"]
        .eq(CORRUPTION_RECENT_FULL)
        .float()
        .mean(),
        "all_full_fraction": batch["receiver_corruption_mode"]
        .eq(CORRUPTION_ALL_FULL)
        .float()
        .mean(),
        "coop_drone_fraction": cooperative_target[:, DRONE].float().mean().detach(),
        "coop_dog_fraction": cooperative_target[:, ROBOTDOG].float().mean().detach(),
    }


def install_airground_v3_overrides() -> None:
    """Install callbacks only inside this V3 training process."""

    base_train.apply_training_defaults = apply_airground_v3_defaults
    base_train.build_multi_agent_model = build_airground_v3_model
    base_train.build_multi_agent_dataset = build_airground_v3_dataset
    base_train.collate_multi_agent_batch = collate_airground_v3_batch
    base_train.forward_multi_agent_loss = forward_airground_v3_loss
    base_train.multi_agent_model_type_name = lambda _cfg: ARCHITECTURE
    base_train.LocalityAwareDistributedSampler = (
        RotatingTemporalStrideDistributedSampler
    )


def run_dry_run(cfg: AirGroundV3TrainConfig) -> None:
    """Validate one paired cache and one paired vision-token sample without an LLM."""

    perception_root = Path(cfg.perception_cache_root).resolve()
    if not perception_root.is_dir():
        raise FileNotFoundError(perception_root)
    pair: Optional[Tuple[Path, Path]] = None
    for drone_cache in perception_root.rglob("drone/*.perception.npz"):
        dog_cache = Path(str(drone_cache).replace("/drone/", "/robotdog/"))
        if dog_cache.is_file():
            pair = drone_cache, dog_cache
            break
    if pair is None:
        raise FileNotFoundError(f"No completed drone/robotdog cache pair under {perception_root}")
    perception_summary = []
    for path in pair:
        with np.load(path, allow_pickle=False) as cache:
            if str(cache["schema_version"].item()) not in LEGACY_SCHEMA_VERSIONS:
                raise ValueError(f"Wrong schema: {path}")
            perception_summary.append(
                (
                    path.parent.name,
                    tuple(cache["mask_grid"].shape),
                    bool(cache["person_valid"].item()),
                )
            )
    first_jsonl = next(Path(cfg.train_json).rglob("*.jsonl"))
    first = json.loads(first_jsonl.open("r", encoding="utf-8").readline())
    base_root = base_train.find_multi_agent_base_root(Path(cfg.train_json))
    token_summary = []
    for name in ("drone", "robotdog"):
        frame = base_train.resolve_multi_agent_path(base_root, first["agents"][name]["current"])
        coarse, fine = base_train.multi_agent_token_paths_for_frame(
            base_root, Path(cfg.cache_root), frame
        )
        if not coarse.is_file() or not fine.is_file():
            raise FileNotFoundError(f"Missing vision cache: {coarse} / {fine}")
        token_summary.append(
            (
                name,
                tuple(torch.load(coarse, map_location="cpu").shape),
                tuple(torch.load(fine, map_location="cpu").shape),
            )
        )
    print(
        f"[DRY_RUN][AIRGROUND_V3] architecture={ARCHITECTURE} "
        f"vision={token_summary} perception={perception_summary}",
        flush=True,
    )
    print(
        "[DRY_RUN][AIRGROUND_V3] three flows are separate; receiver corruption "
        f"uses {len(cfg.receiver_corruption_curriculum)} curriculum stages with "
        "ROI/current/recent/all masking and bounded feasible-recovery targets; "
        f"receiver choice weights=(drone={cfg.train_synthetic_drone_occlusion_prob}, "
        f"dog={cfg.train_synthetic_dog_occlusion_prob}); target-verifier synthetic "
        f"false-positive p={cfg.train_synthetic_false_positive_prob}; "
        f"temporal_stride={cfg.train_temporal_stride}; obstacle loss is disabled "
        "until local-ground projection is calibrated.",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--debug-max-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--train-agent",
        choices=("both", "drone", "robotdog"),
        default=None,
        help="Only the selected agent contributes training losses.",
    )
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--perception-cache-root", type=str, default=None)
    parser.add_argument("--drone-occlusion-prob", type=float, default=None)
    parser.add_argument("--dog-occlusion-prob", type=float, default=None)
    parser.add_argument("--temporal-stride", type=int, default=None)
    parser.add_argument(
        "--distributed",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser.parse_args()


def apply_cli_overrides(
    cfg: AirGroundV3TrainConfig, args: argparse.Namespace
) -> AirGroundV3TrainConfig:
    for argument, field in (
        ("batch_size", "batch_size"),
        ("epochs", "epochs"),
        ("train_agent", "train_agent"),
        ("grad_accum_steps", "grad_accum_steps"),
        ("lr", "lr"),
        ("num_workers", "num_workers"),
        ("out_dir", "out_dir"),
        ("max_steps", "max_steps"),
        ("log_every", "log_every"),
        ("perception_cache_root", "perception_cache_root"),
    ):
        value = getattr(args, argument)
        if value is not None:
            setattr(cfg, field, value)
    if args.drone_occlusion_prob is not None:
        cfg.train_synthetic_drone_occlusion_prob = float(args.drone_occlusion_prob)
        cfg.val_synthetic_drone_occlusion_prob = float(args.drone_occlusion_prob)
    if args.dog_occlusion_prob is not None:
        cfg.train_synthetic_dog_occlusion_prob = float(args.dog_occlusion_prob)
        cfg.val_synthetic_dog_occlusion_prob = float(args.dog_occlusion_prob)
    if args.temporal_stride is not None:
        cfg.train_temporal_stride = int(args.temporal_stride)
    if args.distributed is not None:
        cfg.distributed = bool(args.distributed)
    return cfg


def main() -> int:
    args = parse_args()
    cfg = apply_cli_overrides(load_config(args.config.resolve()), args)
    cfg = apply_airground_v3_defaults(cfg)
    cfg.resume = bool(cfg.resume or args.resume)
    validate_fixed_split(cfg)
    install_airground_v3_overrides()
    if args.dry_run:
        run_dry_run(cfg)
        return 0
    if args.debug_max_steps:
        if args.debug_max_steps < 0:
            raise ValueError("--debug-max-steps must be non-negative")
        cfg.distributed = False
        cfg.batch_size = 1
        cfg.num_workers = 0
        cfg.prefetch_factor = 1
        cfg.progress = False
        cfg.log_every = 1
        cfg.save_every = 0
        cfg.save_every_epochs = 0
        cfg.eval_every = 0
        cfg.val_json = None
        cfg.val_cache_root = None
        cfg.max_steps = int(args.debug_max_steps)
        cfg.out_dir = f"{cfg.out_dir}_debug"
        print(
            f"[DEBUG][AIRGROUND_V3] max_steps={cfg.max_steps} out={cfg.out_dir}",
            flush=True,
        )
    base_train.train_airground_v3(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
