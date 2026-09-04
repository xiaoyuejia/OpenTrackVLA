#!/usr/bin/env python3
"""Closed-loop UnrealZoo inference for airground_three_stream_cooperative_v3.

Inference contract (no target oracle input):

    drone RGB + RobotDog RGB
      -> online DINO/SigLIP features
      -> online YOLO target/obstacle perception
      -> isolated drone/dog self flows + symmetric cooperative flow
      -> YOLO + LLM VERIFY visibility router
      -> calibrated inverse fixed-dt controller
      -> drone and RobotDog actions

The two follower poses are supplied as separate shared-frame pose tokens.
Simulator target pose, target box and target action are never
passed to the model. Target masks/boxes read by the shared runtime are used only
for evaluation metrics and videos.

Example:
    python eval_airground_coop_v3.py \
      --ckpt output/airground_three_stream_cooperative_v3_receiver_target_qwen06b/best_val.pt \
      --save-path output/eval_airground_coop_v3_receiver_target_val100_action_replay \
      --env-id UnrealTrack-Greek_Island-ContinuousColor-v0 --episodes 2
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import sys
import time
import uuid
from collections import deque
from dataclasses import dataclass
from multiprocessing.connection import Client
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from eval_airground_v3_runtime import (
    AirGroundV3RuntimePlanner,
    _append_default,
    _clean_state_dict,
    _option_value,
    build_runtime_argv,
)
from model_airground_coop_v3 import (
    ROUTE_BELIEF,
    ROUTE_COOPERATIVE,
    ROUTE_SEARCH,
    ROUTE_SELF,
    AirGroundCoopV3ModelConfig,
    AirGroundCooperativeVLAV3,
    AirGroundVisibilityRouter,
)
from offline_detection_segmentation.core import mask_to_grid
from offline_detection_segmentation.models import OfflinePerceptionPipeline
from tools.cache_gridpool import VisionCacheConfig, VisionFeatureCacher, grid_pool_tokens


ARCHITECTURE = "airground_three_stream_cooperative_v3"
COOPERATIVE_TARGET_FRAME_VERSION = "receiver_feasible_recovery_v1"
RECEIVER_CORRUPTION_VERSION = "roi_temporal_curriculum_v1"
RELATIVE_POSE_VERSION = "directed_receiver_local_v1"
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PERCEPTION_CONFIG = PROJECT_ROOT / "offline_detection_segmentation/config.yaml"
UNREAL_UNITS_PER_METER = 100.0
SEARCH_HALF_ANGLE_RAD = math.radians(30.0)
SEARCH_ENDPOINT_TOLERANCE_RAD = math.radians(1.0)
ROBOTDOG_WAYPOINT_Y_MODE = "v3_nonholonomic_projection"


def _wrap_angle(value: np.ndarray | float) -> np.ndarray | float:
    """Wrap radians to [-pi, pi] without changing scalar/array shape."""

    return np.arctan2(np.sin(value), np.cos(value))


def _project_robotdog_waypoints_to_nonholonomic(
    waypoints: np.ndarray,
    *,
    control_dt: float,
    source_dt: float,
    horizon_steps: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Convert local ``[x,y,yaw]`` poses into V3 unicycle control waypoints.

    The input remains the model's complete future local pose trajectory.  For
    every segment, x/y and the predicted change in future yaw are jointly fit to
    a constant-curvature unicycle arc.  The returned x channel is cumulative
    signed distance along those arcs and y is zero, allowing the inherited
    calibrated controller to consume it as forward-only RobotDog motion.  The
    returned yaw channel is rescaled so that the inherited ``yaw / control_dt``
    operation means ``predicted future yaw / waypoint horizon time``.

    This is a control-space projection only.  Training and planner/video debug
    retain the original clean local ``[x,y,yaw]`` waypoint representation.
    """

    poses = np.asarray(waypoints, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[0] < 2 or poses.shape[1] < 3:
        raise ValueError(
            "RobotDog waypoints must have shape (N>=2,D>=3), got "
            f"{poses.shape}"
        )
    if not np.isfinite(poses[:, :3]).all():
        raise ValueError("RobotDog waypoints contain non-finite x/y/yaw values")
    if control_dt <= 0.0 or source_dt <= 0.0 or horizon_steps <= 0:
        raise ValueError("control_dt, source_dt and horizon_steps must be positive")

    count = poses.shape[0]
    projected = poses.copy()
    projected[:, :3] = 0.0
    forward_displacements = np.zeros(count - 1, dtype=np.float64)
    lateral_residuals = np.zeros(count - 1, dtype=np.float64)
    yaw_deltas = np.zeros(count - 1, dtype=np.float64)

    cumulative_forward = 0.0
    for index in range(1, count):
        previous_yaw = float(poses[index - 1, 2])
        yaw_delta = float(
            _wrap_angle(float(poses[index, 2]) - previous_yaw)
        )
        cos_previous = math.cos(previous_yaw)
        sin_previous = math.sin(previous_yaw)
        dx = float(poses[index, 0] - poses[index - 1, 0])
        dy = float(poses[index, 1] - poses[index - 1, 1])
        body_x = cos_previous * dx + sin_previous * dy
        body_y = -sin_previous * dx + cos_previous * dy

        # A unit forward arc s with heading change dtheta produces body-frame
        # displacement [s*sin(dtheta)/dtheta,
        #               s*(1-cos(dtheta))/dtheta].  Project the predicted x/y
        # onto this feasible direction.  The stable zero-angle limit is [1,0].
        if abs(yaw_delta) < 1.0e-6:
            basis_x = 1.0 - yaw_delta * yaw_delta / 6.0
            basis_y = 0.5 * yaw_delta
        else:
            basis_x = math.sin(yaw_delta) / yaw_delta
            basis_y = (1.0 - math.cos(yaw_delta)) / yaw_delta
        denominator = max(basis_x * basis_x + basis_y * basis_y, 1.0e-12)
        forward = (basis_x * body_x + basis_y * body_y) / denominator
        residual_x = body_x - forward * basis_x
        residual_y = body_y - forward * basis_y
        # Preserve the signed cross-track direction for diagnosis.
        lateral_sign = (
            1.0
            if (-basis_y * residual_x + basis_x * residual_y) >= 0.0
            else -1.0
        )
        lateral_residual = lateral_sign * math.hypot(residual_x, residual_y)

        cumulative_forward += forward
        projected[index, 0] = cumulative_forward
        forward_displacements[index - 1] = forward
        lateral_residuals[index - 1] = lateral_residual
        yaw_deltas[index - 1] = yaw_delta

        source_step = max(
            1,
            round(index * max(1, int(horizon_steps)) / (count - 1)),
        )
        horizon_time = max(source_step * float(source_dt), 1.0e-6)
        future_yaw = float(_wrap_angle(float(poses[index, 2])))
        projected[index, 2] = future_yaw * float(control_dt) / horizon_time

    return projected, {
        "forward_displacements_m": forward_displacements,
        "lateral_residuals_m": lateral_residuals,
        "yaw_deltas_rad": yaw_deltas,
    }


def _resolve_checkpoint(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        return path
    best = path / "best_val.pt"
    if best.is_file():
        return best
    checkpoints = sorted(path.glob("model_epoch*.pt"), key=lambda item: item.stat().st_mtime)
    if not checkpoints:
        raise FileNotFoundError(f"No best_val.pt or model_epoch*.pt under {path}")
    return checkpoints[-1]


def _load_payload(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(str(path), map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError("Air-ground checkpoint must be a dictionary payload")
    config = dict(payload.get("config") or {})
    architecture = config.get("model_architecture") or config.get("model_type")
    if architecture != ARCHITECTURE:
        raise ValueError(f"Expected {ARCHITECTURE!r}, got {architecture!r}")
    expected_versions = {
        "cooperative_target_frame_version": COOPERATIVE_TARGET_FRAME_VERSION,
        "receiver_corruption_version": RECEIVER_CORRUPTION_VERSION,
        "relative_pose_version": RELATIVE_POSE_VERSION,
    }
    incompatible = {
        name: config.get(name)
        for name, expected in expected_versions.items()
        if config.get(name) != expected
    }
    if incompatible:
        raise ValueError(
            "Checkpoint predates the receiver-recovery V3 contract; "
            f"expected {expected_versions}, found {incompatible}"
        )
    return payload


def _model_config(config: dict[str, Any]) -> AirGroundCoopV3ModelConfig:
    return AirGroundCoopV3ModelConfig(
        llm_name=str(config.get("llm_name", "Qwen/Qwen3-0.6B")),
        freeze_llm=True,
        n_waypoints=int(config.get("n_waypoints", 10)),
        action_dims=int(config.get("action_dims", 3)),
        num_modes=int(config.get("num_modes", 4)),
        text_max_length=int(config.get("text_max_length", 128)),
        insert_time_tokens=bool(config.get("insert_time_tokens", True)),
        use_angle_tvi=bool(config.get("use_angle_tvi", False)),
        use_agent_text_markers=bool(config.get("use_agent_text_markers", True)),
        use_tanh_actions=not bool(config.get("no_tanh_actions", True)),
        alpha_xy=config.get("alpha_xy", 1.0),
        perception_grid_size=int(config.get("perception_grid_size", 8)),
        drone_mask_expand_ratio=float(config.get("drone_mask_expand_ratio", 3.0)),
        dog_mask_expand_ratio=float(config.get("dog_mask_expand_ratio", 3.0)),
        coop_hidden_dim=int(config.get("coop_hidden_dim", 512)),
        coop_encoder_layers=int(config.get("coop_encoder_layers", 1)),
        coop_decoder_layers=int(config.get("coop_decoder_layers", 3)),
        coop_num_heads=int(config.get("coop_num_heads", 8)),
        coop_dropout=float(config.get("coop_dropout", 0.0)),
        jepa_hidden_dim=int(config.get("jepa_hidden_dim", 512)),
        jepa_decoder_layers=int(config.get("jepa_decoder_layers", 3)),
        jepa_num_heads=int(config.get("jepa_num_heads", 8)),
        jepa_dropout=float(config.get("jepa_dropout", 0.0)),
        jepa_momentum=float(config.get("jepa_momentum", 0.996)),
        detection_confidence_threshold=float(
            config.get("detection_confidence_threshold", 0.25)
        ),
        target_match_confidence_threshold=float(
            os.environ.get(
                "AIRGROUND_V3_TARGET_MATCH_THRESHOLD",
                config.get("target_match_confidence_threshold", 0.50),
            )
        ),
        candidate_temporal_iou_weight=float(
            config.get("candidate_temporal_iou_weight", 2.0)
        ),
        hard_visibility_routing=bool(config.get("hard_visibility_routing", True)),
        pose_position_scale_m=float(config.get("pose_position_scale_m", 20.0)),
        drone_target_verification_prompt=str(
            config.get(
                "drone_target_verification_prompt",
                AirGroundCoopV3ModelConfig.drone_target_verification_prompt,
            )
        ),
        dog_target_verification_prompt=str(
            config.get(
                "dog_target_verification_prompt",
                AirGroundCoopV3ModelConfig.dog_target_verification_prompt,
            )
        ),
    )


def _perception_config(args: argparse.Namespace, grid_size: int) -> dict[str, Any]:
    path = Path(args.perception_config or DEFAULT_PERCEPTION_CONFIG).expanduser().resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Perception config must be a mapping: {path}")
    config = copy.deepcopy(raw)
    for section in ("models", "runtime", "thresholds", "target", "output"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing mapping {section!r} in {path}")

    device_index = args.device
    if device_index is None:
        device_index = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif str(device_index) == "cuda":
        device_index = "cuda:0"
    config["runtime"]["device"] = str(device_index)
    config["runtime"]["batch_size"] = 2
    # FP16 conversion in the installed Ultralytics/CUDA combination can hit a
    # kernel watchdog timeout. Online batch=2 is small, so default to the
    # verified-stable FP32 path; --yolo-half remains an explicit opt-in.
    config["runtime"]["half"] = bool(args.yolo_half) if args.yolo_half is not None else False
    if args.yolo_weights is not None:
        config["models"]["yolo_weights"] = str(args.yolo_weights)
    if args.yolo_image_size is not None:
        config["models"]["yolo_image_size"] = int(args.yolo_image_size)
    if args.person_confidence is not None:
        config["thresholds"]["person_confidence"] = float(args.person_confidence)
    if args.object_confidence is not None:
        config["thresholds"]["object_confidence"] = float(args.object_confidence)

    weights = Path(str(config["models"]["yolo_weights"])).expanduser()
    if not weights.is_absolute():
        weights = (PROJECT_ROOT / weights).resolve()
    if not weights.is_file():
        raise FileNotFoundError(weights)
    config["models"]["yolo_weights"] = str(weights)
    config["output"]["grid_size"] = [int(grid_size), int(grid_size)]
    return config


def _wrap_degrees(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def _agent_poses(drone_pose: Any, robotdog_pose: Any) -> torch.Tensor:
    """Return two true poses in one pair-centred metric coordinate frame."""

    values = []
    for raw_pose in (drone_pose, robotdog_pose):
        pose = np.asarray(raw_pose, dtype=np.float32)
        if pose.ndim != 1 or pose.size < 6:
            raise ValueError(
                "drone_pose and robotdog_pose must each contain at least 6 values"
            )
        yaw = math.radians(float(pose[4]))
        values.append(
            [
                float(pose[0] / UNREAL_UNITS_PER_METER),
                float(pose[1] / UNREAL_UNITS_PER_METER),
                math.sin(yaw),
                math.cos(yaw),
            ]
        )
    output = torch.tensor(values, dtype=torch.float32)
    output[:, :2] -= output[:, :2].mean(dim=0, keepdim=True)
    return output


def _llm_verified_boxes(
    boxes: Sequence[Sequence[float]],
    stable_visible: Sequence[bool],
) -> list[list[float]]:
    """Expose a YOLO target box only after stable LLM verification."""

    if len(boxes) != 2 or len(stable_visible) != 2:
        raise ValueError("V3 verified boxes require exactly two agents")
    verified: list[list[float]] = []
    for box, accepted in zip(boxes, stable_visible):
        values = np.asarray(box, dtype=np.float32).reshape(-1)
        if values.size != 4:
            raise ValueError("Each V3 target box must contain cx,cy,w,h")
        verified.append(
            [float(value) for value in values]
            if bool(accepted)
            else [0.0, 0.0, 0.0, 0.0]
        )
    return verified


@dataclass(frozen=True)
class BBoxMotionProfile:
    """Per-agent image-space feedback limits for the V3 evaluator."""

    name: str
    # Kept for CLI/config compatibility; initial-reference ratios determine
    # the active height response below.
    height_normal: float
    height_far: float
    min_speed_gain: float
    center_reference: float
    center_left_safe: float
    center_right_safe: float
    center_left_full: float
    center_right_full: float
    max_speed_gain: float
    max_forward_residual_mps: float
    max_yaw_gain: float
    max_yaw_residual_rps: float
    edge_speed_suppression: float
    max_translation_speed_mps: float
    max_yaw_rate_rps: float
    holonomic_translation: bool
    turn_in_place_on_edge: bool


@dataclass
class BBoxMotionState:
    ema_height: float | None = None
    ema_center_x: float | None = None
    reference_height: float | None = None
    relative_height_shrink: float = 0.0
    relative_height_error: float = 0.0
    valid_streak: int = 0
    shrink_streak: int = 0
    reliable: bool = False
    detector_valid: bool = False
    detector_score: float = 0.0


class BBoxMotionController:
    """Causal initial-reference bbox speed and direction controller for V3.

    Each agent records the first trusted target-box height after reset. A
    configurable dead band suppresses speed correction around that reference;
    smaller boxes add bounded catch-up speed and larger boxes apply bounded
    slowdown while moving forward. During reverse motion those distance
    responses invert: larger boxes increase reverse magnitude and smaller
    boxes slow reverse motion. Horizontal center independently produces
    bounded yaw correction.
    """

    def __init__(self, args: argparse.Namespace):
        self.enabled = bool(getattr(args, "bbox_motion_control", True))
        self.min_confidence = float(getattr(args, "bbox_motion_min_confidence", 0.25))
        self.ema_alpha = float(getattr(args, "bbox_motion_ema_alpha", 0.20))
        self.min_valid_frames = int(getattr(args, "bbox_motion_min_valid_frames", 2))
        self.min_shrink_frames = int(getattr(args, "bbox_motion_min_shrink_frames", 3))
        self.height_tolerance_ratio = float(
            getattr(args, "bbox_motion_height_tolerance_ratio", 0.20)
        )
        self.height_response_ratio = float(
            getattr(args, "bbox_motion_height_response_ratio", 0.50)
        )
        if not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError("bbox_motion_ema_alpha must lie in (0,1]")
        if self.min_valid_frames < 1 or self.min_shrink_frames < 1:
            raise ValueError("bbox motion frame counts must be positive")
        if not 0.0 <= self.height_tolerance_ratio < self.height_response_ratio:
            raise ValueError(
                "bbox_motion_height_tolerance_ratio must be non-negative and "
                "smaller than bbox_motion_height_response_ratio"
            )
        if self.height_response_ratio > 1.0:
            raise ValueError("bbox_motion_height_response_ratio must be <= 1")

        drone_max_speed = float(getattr(args, "drone_max_speed", None) or 2.50)
        dog_max_speed = float(getattr(args, "robotdog_max_speed", None) or 2.50)
        self.profiles = (
            BBoxMotionProfile(
                name="drone",
                height_normal=float(getattr(args, "drone_bbox_height_normal", 0.150)),
                height_far=float(getattr(args, "drone_bbox_height_far", 0.120)),
                min_speed_gain=float(
                    getattr(args, "drone_bbox_min_speed_gain", 0.50)
                ),
                center_reference=0.49766,
                center_left_safe=0.40625,
                center_right_safe=0.59219,
                center_left_full=0.38281,
                center_right_full=0.61484,
                max_speed_gain=float(getattr(args, "drone_bbox_max_speed_gain", 1.50)),
                max_forward_residual_mps=0.15,
                max_yaw_gain=1.50,
                max_yaw_residual_rps=float(
                    getattr(args, "drone_bbox_max_yaw_residual", 0.12)
                ),
                edge_speed_suppression=0.70,
                max_translation_speed_mps=drone_max_speed,
                max_yaw_rate_rps=float(getattr(args, "drone_max_yaw_rate", 0.40)),
                holonomic_translation=True,
                turn_in_place_on_edge=False,
            ),
            BBoxMotionProfile(
                name="robotdog",
                height_normal=float(getattr(args, "robotdog_bbox_height_normal", 0.220)),
                height_far=float(getattr(args, "robotdog_bbox_height_far", 0.160)),
                min_speed_gain=float(
                    getattr(args, "robotdog_bbox_min_speed_gain", 0.50)
                ),
                center_reference=0.49687,
                # The dog training set is nearly perfectly centered.  Use a
                # wider runtime dead band than its misleadingly narrow P10/P90.
                center_left_safe=0.45,
                center_right_safe=0.55,
                center_left_full=0.35,
                center_right_full=0.65,
                max_speed_gain=float(getattr(args, "robotdog_bbox_max_speed_gain", 2.00)),
                max_forward_residual_mps=0.15,
                max_yaw_gain=1.30,
                max_yaw_residual_rps=float(
                    getattr(args, "robotdog_bbox_max_yaw_residual", 0.25)
                ),
                edge_speed_suppression=0.80,
                max_translation_speed_mps=dog_max_speed,
                max_yaw_rate_rps=float(getattr(args, "robotdog_max_yaw_rate", 1.0)),
                holonomic_translation=False,
                turn_in_place_on_edge=True,
            ),
        )
        for profile in self.profiles:
            if not 0.0 < profile.height_far < profile.height_normal:
                raise ValueError(
                    f"{profile.name} bbox heights must satisfy 0 < far < normal"
                )
            if (
                not 0.0 < profile.min_speed_gain <= 1.0
                or profile.max_speed_gain < 1.0
                or profile.max_yaw_gain < 1.0
            ):
                raise ValueError(
                    f"{profile.name} bbox gains must satisfy "
                    "0 < min_speed_gain <= 1, max gains >= 1"
                )
        self.states = [BBoxMotionState(), BBoxMotionState()]

    def reset(self) -> None:
        self.states = [BBoxMotionState(), BBoxMotionState()]

    @staticmethod
    def _valid_box(box: Any) -> tuple[float, float] | None:
        try:
            values = np.asarray(box, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            return None
        if values.size < 4 or not np.isfinite(values[:4]).all():
            return None
        center_x = float(values[0])
        height = float(values[3])
        if height <= 0.0 or center_x < 0.0 or center_x > 1.0:
            return None
        return center_x, height

    def observe(self, agent: int, box: Any, score: float, valid: bool) -> dict[str, Any]:
        state = self.states[agent]
        parsed = self._valid_box(box)
        try:
            detector_score = float(score)
        except (TypeError, ValueError):
            detector_score = 0.0
        if not math.isfinite(detector_score):
            detector_score = 0.0
        trusted_detection = bool(
            self.enabled
            and valid
            and parsed is not None
            and detector_score >= self.min_confidence
        )
        state.detector_valid = bool(valid and parsed is not None)
        state.detector_score = detector_score
        if not trusted_detection:
            state.valid_streak = 0
            state.shrink_streak = 0
            state.relative_height_shrink = 0.0
            state.reliable = False
            return self._state_debug(agent)

        assert parsed is not None
        center_x, height = parsed
        if state.reference_height is None:
            state.reference_height = height
        continuing = state.valid_streak > 0 and state.ema_height is not None
        if continuing:
            previous_height = float(state.ema_height)
            state.ema_height = (
                (1.0 - self.ema_alpha) * previous_height + self.ema_alpha * height
            )
            state.ema_center_x = (
                (1.0 - self.ema_alpha) * float(state.ema_center_x)
                + self.ema_alpha * center_x
            )
            state.relative_height_shrink = max(
                0.0,
                (previous_height - float(state.ema_height)) / max(previous_height, 1e-6),
            )
        else:
            state.ema_height = height
            state.ema_center_x = center_x
            state.relative_height_shrink = 0.0

        state.relative_height_error = (
            (float(state.reference_height) - float(state.ema_height))
            / max(float(state.reference_height), 1.0e-6)
        )

        state.valid_streak += 1
        # Keep the old shrink-trend statistic for diagnostics, but the active
        # speed correction is now driven only by the initial-height reference.
        if state.relative_height_shrink > 0.002:
            state.shrink_streak += 1
        else:
            state.shrink_streak = 0
        state.reliable = state.valid_streak >= self.min_valid_frames
        return self._state_debug(agent)

    def _edge(self, agent: int) -> tuple[float, float]:
        state, profile = self.states[agent], self.profiles[agent]
        center_x = state.ema_center_x
        if center_x is None:
            return 0.0, 0.0
        if center_x < profile.center_left_safe:
            amount = (profile.center_left_safe - center_x) / max(
                profile.center_left_safe - profile.center_left_full, 1e-6
            )
            return float(np.clip(amount, 0.0, 1.0)), -1.0
        if center_x > profile.center_right_safe:
            amount = (center_x - profile.center_right_safe) / max(
                profile.center_right_full - profile.center_right_safe, 1e-6
            )
            return float(np.clip(amount, 0.0, 1.0)), 1.0
        return 0.0, 0.0

    def _state_debug(self, agent: int) -> dict[str, Any]:
        state, profile = self.states[agent], self.profiles[agent]
        q_short = 0.0
        q_close = 0.0
        if state.reference_height is not None and state.ema_height is not None:
            error = float(state.relative_height_error)
            response_span = max(
                self.height_response_ratio - self.height_tolerance_ratio,
                1.0e-6,
            )
            q_short = float(
                np.clip(
                    (error - self.height_tolerance_ratio) / response_span,
                    0.0,
                    1.0,
                )
            )
            q_close = float(
                np.clip(
                    (-error - self.height_tolerance_ratio) / response_span,
                    0.0,
                    1.0,
                )
            )
        q_shrink = float(
            np.clip((state.relative_height_shrink - 0.002) / 0.010, 0.0, 1.0)
        )
        if state.shrink_streak < self.min_shrink_frames:
            q_shrink = 0.0
        q_edge, edge_sign = self._edge(agent)
        return {
            "agent": profile.name,
            "enabled": self.enabled,
            "reliable": state.reliable,
            "detector_valid": state.detector_valid,
            "detector_score": state.detector_score,
            "valid_streak": state.valid_streak,
            "shrink_streak": state.shrink_streak,
            "reference_height": state.reference_height,
            "height": state.ema_height,
            "height_ratio": (
                float(state.ema_height) / float(state.reference_height)
                if state.reference_height is not None and state.ema_height is not None
                else None
            ),
            "relative_height_error": state.relative_height_error,
            "height_tolerance_ratio": self.height_tolerance_ratio,
            "height_response_ratio": self.height_response_ratio,
            "center_x": state.ema_center_x,
            "relative_height_shrink": state.relative_height_shrink,
            # Keep q_far as a compatibility alias for the new short-target
            # response; q_shrink remains diagnostic only and no longer drives
            # the speed correction.
            "q_far": q_short,
            "q_short": q_short,
            "q_close": q_close,
            "q_shrink": q_shrink,
            "q_edge": q_edge,
            "edge_sign": edge_sign,
        }

    def adjust(
        self,
        raw_drone: np.ndarray,
        raw_robotdog: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        adjusted = [
            np.asarray(raw_drone, dtype=np.float64).copy(),
            np.asarray(raw_robotdog, dtype=np.float64).copy(),
        ]
        agent_debug: list[dict[str, Any]] = []
        for agent, velocity in enumerate(adjusted):
            profile = self.profiles[agent]
            debug = self._state_debug(agent)
            before = velocity.copy()
            speed_gain = 1.0
            forward_residual = 0.0
            speed_correction_applied = False
            speed_correction_direction = "none"
            yaw_gain = 1.0
            yaw_residual = 0.0
            q_short = float(debug["q_short"])
            q_close = float(debug["q_close"])
            q_catch = q_short
            turn_in_place_applied = False
            if self.enabled and self.states[agent].reliable:
                q_edge = float(debug["q_edge"])
                turn_in_place = profile.turn_in_place_on_edge and q_edge > 0.0
                edge_factor = max(
                    0.0,
                    1.0 - profile.edge_speed_suppression * q_edge,
                )
                if not turn_in_place:
                    if velocity[0] >= 0.0:
                        if q_short > 0.0:
                            speed_gain = 1.0 + (
                                profile.max_speed_gain - 1.0
                            ) * q_short * edge_factor
                            forward_residual = (
                                profile.max_forward_residual_mps
                                * q_short
                                * edge_factor
                            )
                            if profile.holonomic_translation:
                                velocity[:2] *= speed_gain
                            else:
                                velocity[0] *= speed_gain
                            velocity[0] += forward_residual
                            speed_correction_direction = "forward"
                            speed_correction_applied = True
                        elif q_close > 0.0:
                            # A large bbox means the target is close. Slow
                            # policy-controlled forward motion so the agent
                            # does not keep closing at the catch-up rate.
                            speed_gain = max(
                                profile.min_speed_gain,
                                1.0 - (1.0 - profile.min_speed_gain) * q_close,
                            )
                            if profile.holonomic_translation:
                                velocity[:2] *= speed_gain
                            else:
                                velocity[0] *= speed_gain
                            speed_correction_direction = "forward"
                            speed_correction_applied = True
                    elif q_close > 0.0:
                        # Reverse motion has the opposite distance response:
                        # a large bbox indicates a close target, so increase
                        # the reverse magnitude to create separation.
                        speed_gain = 1.0 + (
                            profile.max_speed_gain - 1.0
                        ) * q_close * edge_factor
                        forward_residual = -(
                            profile.max_forward_residual_mps
                            * q_close
                            * edge_factor
                        )
                        if profile.holonomic_translation:
                            velocity[:2] *= speed_gain
                        else:
                            velocity[0] *= speed_gain
                        velocity[0] += forward_residual
                        speed_correction_direction = "reverse"
                        speed_correction_applied = True
                    elif q_short > 0.0:
                        # A small bbox indicates a distant target, so reduce
                        # reverse motion instead of accelerating away from it.
                        speed_gain = max(
                            profile.min_speed_gain,
                            1.0 - (1.0 - profile.min_speed_gain) * q_short,
                        )
                        if profile.holonomic_translation:
                            velocity[:2] *= speed_gain
                        else:
                            velocity[0] *= speed_gain
                        speed_correction_direction = "reverse"
                        speed_correction_applied = True

                edge_sign = float(debug["edge_sign"])
                if edge_sign and velocity[2] * edge_sign > 0.0:
                    yaw_gain = 1.0 + (profile.max_yaw_gain - 1.0) * q_edge
                    velocity[2] *= yaw_gain
                yaw_residual = (
                    profile.max_yaw_residual_rps
                    * edge_sign
                    * q_edge
                )
                velocity[2] += yaw_residual
                if turn_in_place:
                    velocity[0] = 0.0
                    turn_in_place_applied = True

            # These are physical V3 limits, not merely bbox-residual limits.
            # Enforce them even before VERIFY becomes temporally reliable or
            # when bbox motion correction is disabled.
            if profile.holonomic_translation:
                norm = float(np.linalg.norm(velocity[:2]))
                if norm > profile.max_translation_speed_mps:
                    velocity[:2] *= profile.max_translation_speed_mps / max(
                        norm, 1e-6
                    )
            else:
                velocity[0] = float(
                    np.clip(
                        velocity[0],
                        -profile.max_translation_speed_mps,
                        profile.max_translation_speed_mps,
                    )
                )
            velocity[2] = float(
                np.clip(
                    velocity[2],
                    -profile.max_yaw_rate_rps,
                    profile.max_yaw_rate_rps,
                )
            )

            debug.update(
                {
                    "q_catch": q_catch,
                    "q_close": q_close,
                    "speed_correction_applied": speed_correction_applied,
                    "speed_correction_direction": speed_correction_direction,
                    "turn_in_place_applied": turn_in_place_applied,
                    "speed_gain": speed_gain,
                    "forward_residual_mps": forward_residual,
                    "yaw_gain": yaw_gain,
                    "yaw_residual_rps": yaw_residual,
                    "raw_velocity_before": before.tolist(),
                    "raw_velocity_after": velocity.tolist(),
                }
            )
            agent_debug.append(debug)
        return adjusted[0], adjusted[1], {
            "controller": "yolo_llm_verified_bbox_height_cx_v3",
            "enabled": self.enabled,
            "speed_correction_applied": any(
                bool(item["speed_correction_applied"]) for item in agent_debug
            ),
            "agents": agent_debug,
        }


class AirGroundCoopV3Planner(AirGroundV3RuntimePlanner):
    """在线三流 V3 planner，并为每个 session 独立维护验证路由状态。"""

    # 通知共享 UnrealZoo runtime 传入两个跟随者的原始位姿，并在 policy 间隔内
    # 按 0.1 s 补充历史视觉；补充历史不会额外执行 Qwen forward。
    requires_inter_agent_pose = True
    requires_agent_poses = True
    supports_intermediate_observation = True

    def __init__(self, args: argparse.Namespace):
        # The V3 planner owns model construction; the runtime base only provides
        # temporal history and inverse fixed-dt control.
        self.args = args
        self.device = torch.device(
            args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.vision_amp = bool(getattr(args, "vision_amp", True))
        self.inference_amp = bool(getattr(args, "inference_amp", True))
        self.ckpt_path = _resolve_checkpoint(Path(args.ckpt))
        print(f"[airground-v3] loading checkpoint {self.ckpt_path}", flush=True)
        payload = _load_payload(self.ckpt_path)
        self.config = dict(payload.get("config") or {})
        # reference 实现继续保留；若 checkpoint 的 use_target_reference=false，
        # 评估时不建立也不输入动态身份 reference。
        self.use_target_reference = bool(self.config.get("use_target_reference", True))
        state = _clean_state_dict(payload.get("model_state") or {})
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
        absent = [
            prefix for prefix in required
            if not any(key.startswith(prefix) for key in state)
        ]
        if absent:
            raise ValueError(
                "Checkpoint is not a complete air-ground V3 checkpoint; "
                f"missing prefixes: {absent}"
            )

        cfg = _model_config(self.config)
        self.history = int(self.config.get("history", args.history))
        self.history_frame_dt = float(args.history_frame_dt or args.dt)
        self.n_waypoints = int(cfg.n_waypoints)
        self.use_roi_tokens = False
        self.use_bbox_text_prompt = False
        self.use_visual_section_markers = False
        self.roi_bbox_source = "online_yolo_llm_verified"
        self.roi_expand_ratio = 0.0
        self.roi_token_count = 0
        self.roi_make_square = False
        self.evaluation_protocol = (
            "airground_v3_receiver_feasible_recovery_yolo_llm_verify_hysteresis"
        )
        self.ckpt_bbox_dropout_prob = 0.0

        print(f"[airground-v3] building {ARCHITECTURE} on {self.device}", flush=True)
        model = AirGroundCooperativeVLAV3(
            cfg,
            vision_feat_dim=int(
                self.config.get("vision_feat_dim", args.vision_feat_dim)
            ),
        )
        missing, unexpected = model.load_state_dict(state, strict=False)
        allowed_missing_prefixes = (
            "candidate_matcher.",
            "null_target_context",
            "target_context_projs.",
        )
        allowed_unexpected_prefixes = ("candidate_match_head.",)
        disallowed_missing = sorted(
            key for key in missing
            if not key.startswith(allowed_missing_prefixes)
        )
        disallowed_unexpected = sorted(
            key for key in unexpected
            if not key.startswith(allowed_unexpected_prefixes)
        )
        if disallowed_missing or disallowed_unexpected:
            raise RuntimeError(
                "Could not compatibly load the receiver-target V3 checkpoint; "
                f"missing={disallowed_missing} unexpected={disallowed_unexpected}"
            )
        if missing:
            print(
                "[airground-v3] legacy single-candidate checkpoint: "
                "candidate matcher initialized but used only when top-K input is present",
                flush=True,
            )
        self.model = model.to(self.device).eval()
        del state, payload
        print(
            f"[airground-v3] model loaded strictly history={self.history} "
            f"waypoints={self.n_waypoints} modes={cfg.num_modes}",
            flush=True,
        )

        print("[airground-v3] loading online DINO + SigLIP encoders", flush=True)
        started = time.time()
        self.encoder = VisionFeatureCacher(
            VisionCacheConfig(
                image_size=int(args.image_size),
                batch_size=2,
                device=str(self.device),
                resize_mode=str(args.vision_resize_mode),
            )
        ).eval()
        print(
            f"[airground-v3] visual encoders loaded in "
            f"{time.time() - started:.1f}s",
            flush=True,
        )

        print("[airground-v3] loading online YOLO instance segmentation", flush=True)
        started = time.time()
        self.perception = OfflinePerceptionPipeline(
            _perception_config(args, cfg.perception_grid_size)
        )
        print(
            f"[airground-v3] YOLO loaded in {time.time() - started:.1f}s",
            flush=True,
        )

        capacity = max(self.history * 4, self.history + 1)
        self.histories: list[deque[tuple[float, torch.Tensor]]] = [
            deque(maxlen=capacity),
            deque(maxlen=capacity),
        ]
        self.inverse_velocity_state: np.ndarray | None = None
        self.inverse_velocity_reference: np.ndarray | None = None
        self.last_inverse_command: np.ndarray | None = None
        self.last_waypoints: np.ndarray | None = None
        self.last_policy_debug: dict[str, Any] = {}
        self.visibility_router = self.new_visibility_router()
        self.bbox_motion_controller = self.new_bbox_motion_controller()
        self.last_navigation_waypoints: np.ndarray | None = None
        self.search_center_yaw_rad: np.ndarray | None = None
        self.search_target_direction = np.ones(2, dtype=np.float32)
        self.last_candidate_bbox = np.zeros((2, 4), dtype=np.float32)
        self.last_candidate_valid = np.zeros(2, dtype=bool)
        self.target_reference_tokens: torch.Tensor | None = None
        self.target_reference_valid = np.zeros(2, dtype=bool)
        self.pending_target_reference_tokens: torch.Tensor | None = None
        self.pending_target_reference_bbox = np.zeros((2, 4), dtype=np.float32)
        self.pending_target_reference_valid = np.zeros(2, dtype=bool)
        self.target_reference_confirm_count = np.zeros(2, dtype=np.int64)
        self.target_reference_miss_count = np.zeros(2, dtype=np.int64)
        print(
            "[airground-v3] routing=YOLO+LLM_VERIFY+hysteresis "
            "flows=self/self/cooperative; "
            f"bbox_motion={self.bbox_motion_controller.enabled}",
            flush=True,
        )

    def new_visibility_router(self) -> AirGroundVisibilityRouter:
        """Return a fresh router for an isolated evaluation session."""

        target_enter = float(self.model.cfg.target_match_confidence_threshold)
        detector_enter = max(
            0.35, float(self.model.cfg.detection_confidence_threshold)
        )
        return AirGroundVisibilityRouter(
            enter_confidence=detector_enter,
            exit_confidence=min(0.20, detector_enter),
            target_match_enter_confidence=target_enter,
            target_match_exit_confidence=min(0.35, target_enter),
            visible_confirm_frames=2,
            invisible_confirm_frames=2,
            belief_hold_frames=3,
        )

    def new_bbox_motion_controller(self) -> BBoxMotionController:
        """Return a fresh bbox controller for an isolated evaluation session."""

        return BBoxMotionController(self.args)

    def reset(self) -> None:
        super().reset()
        self.last_policy_debug = {}
        self.visibility_router.reset()
        self.bbox_motion_controller.reset()
        self.last_navigation_waypoints = None
        self.search_center_yaw_rad = None
        self.search_target_direction = np.ones(2, dtype=np.float32)
        self.last_candidate_bbox.fill(0.0)
        self.last_candidate_valid.fill(False)
        self.target_reference_tokens = None
        self.target_reference_valid.fill(False)
        self.pending_target_reference_tokens = None
        self.pending_target_reference_bbox.fill(0.0)
        self.pending_target_reference_valid.fill(False)
        self.target_reference_confirm_count.fill(0)
        self.target_reference_miss_count.fill(0)

    @staticmethod
    def _pool_candidate_reference(
        fine_tokens: torch.Tensor, bbox: torch.Tensor
    ) -> torch.Tensor:
        token_count = int(fine_tokens.size(0))
        side = int(round(math.sqrt(token_count)))
        if side * side != token_count:
            raise ValueError("fine token count must form a square grid")
        cx, cy, width, height = bbox.float()
        half_width = width.clamp_min(0.0) * 0.5
        half_height = height.clamp_min(0.0) * 0.5
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
        return fine_tokens[mask].float().mean(dim=0)

    @staticmethod
    def _bbox_iou_cxcywh(first: np.ndarray, second: np.ndarray) -> float:
        first = np.asarray(first, dtype=np.float32)
        second = np.asarray(second, dtype=np.float32)
        first_xyxy = np.asarray(
            [first[0] - first[2] / 2, first[1] - first[3] / 2,
             first[0] + first[2] / 2, first[1] + first[3] / 2]
        )
        second_xyxy = np.asarray(
            [second[0] - second[2] / 2, second[1] - second[3] / 2,
             second[0] + second[2] / 2, second[1] + second[3] / 2]
        )
        intersection = max(
            0.0, min(first_xyxy[2], second_xyxy[2]) - max(first_xyxy[0], second_xyxy[0])
        ) * max(
            0.0, min(first_xyxy[3], second_xyxy[3]) - max(first_xyxy[1], second_xyxy[1])
        )
        union = max(0.0, first[2] * first[3]) + max(0.0, second[2] * second[3]) - intersection
        return float(intersection / union) if union > 1e-8 else 0.0

    def _update_target_reference_memory(
        self,
        fine_tokens: torch.Tensor,
        selected_detection: torch.Tensor,
        accepted: torch.Tensor,
        confidence: torch.Tensor,
    ) -> None:
        """Confirm, freeze, and if necessary release shared STT/DT/AT identity memory."""
        feature_dim = int(fine_tokens.size(-1))
        if not hasattr(self, "target_reference_valid"):
            self.target_reference_valid = np.zeros(2, dtype=bool)
        if not hasattr(self, "target_reference_tokens"):
            self.target_reference_tokens = None
        if not hasattr(self, "pending_target_reference_tokens"):
            self.pending_target_reference_tokens = None
            self.pending_target_reference_bbox = np.zeros((2, 4), dtype=np.float32)
            self.pending_target_reference_valid = np.zeros(2, dtype=bool)
            self.target_reference_confirm_count = np.zeros(2, dtype=np.int64)
            self.target_reference_miss_count = np.zeros(2, dtype=np.int64)
        if self.target_reference_tokens is None:
            self.target_reference_tokens = torch.zeros(2, feature_dim, dtype=fine_tokens.dtype)
        if self.pending_target_reference_tokens is None:
            self.pending_target_reference_tokens = torch.zeros(
                2, feature_dim, dtype=fine_tokens.dtype
            )

        minimum_confidence = float(
            self.config.get("target_reference_min_confidence", 0.65)
        )
        confirm_frames = max(
            1, int(self.config.get("target_reference_confirm_frames", 3))
        )
        release_frames = max(
            1, int(self.config.get("target_reference_release_frames", 12))
        )
        minimum_similarity = float(
            self.config.get("target_reference_min_cosine_similarity", 0.70)
        )
        minimum_iou = float(
            self.config.get("target_reference_min_bbox_iou", 0.10)
        )

        selected_np = selected_detection.detach().float().cpu().numpy()
        accepted_np = accepted.detach().bool().cpu().numpy()
        confidence_np = confidence.detach().float().cpu().numpy()
        for agent_id in range(2):
            observation_valid = bool(
                accepted_np[agent_id]
                and selected_np[agent_id, 5] > 0.5
                and confidence_np[agent_id] >= minimum_confidence
            )
            if self.target_reference_valid[agent_id]:
                # Freeze a confirmed anchor instead of EMA-updating it into a
                # distractor. Sustained NULL/rejection releases the anchor and
                # lets instruction grounding acquire a new multi-frame hypothesis.
                if observation_valid:
                    self.target_reference_miss_count[agent_id] = 0
                else:
                    self.target_reference_miss_count[agent_id] += 1
                    if self.target_reference_miss_count[agent_id] >= release_frames:
                        self.target_reference_valid[agent_id] = False
                        self.pending_target_reference_valid[agent_id] = False
                        self.target_reference_confirm_count[agent_id] = 0
                        self.target_reference_miss_count[agent_id] = 0
                continue

            if not observation_valid:
                self.pending_target_reference_valid[agent_id] = False
                self.target_reference_confirm_count[agent_id] = 0
                continue

            bbox = selected_np[agent_id, :4]
            roi = self._pool_candidate_reference(
                fine_tokens[agent_id], torch.from_numpy(bbox)
            )
            consistent = False
            if self.pending_target_reference_valid[agent_id]:
                previous = self.pending_target_reference_tokens[agent_id].float()
                cosine = float(
                    F.cosine_similarity(roi[None], previous[None], dim=-1).item()
                )
                bbox_iou = self._bbox_iou_cxcywh(
                    bbox, self.pending_target_reference_bbox[agent_id]
                )
                consistent = cosine >= minimum_similarity or bbox_iou >= minimum_iou
            if consistent:
                count = int(self.target_reference_confirm_count[agent_id])
                self.pending_target_reference_tokens[agent_id] = (
                    self.pending_target_reference_tokens[agent_id] * count + roi
                ) / float(count + 1)
                self.target_reference_confirm_count[agent_id] = count + 1
            else:
                self.pending_target_reference_tokens[agent_id] = roi
                self.pending_target_reference_valid[agent_id] = True
                self.target_reference_confirm_count[agent_id] = 1
            self.pending_target_reference_bbox[agent_id] = bbox

            if self.target_reference_confirm_count[agent_id] >= confirm_frames:
                self.target_reference_tokens[agent_id] = (
                    self.pending_target_reference_tokens[agent_id].clone()
                )
                self.target_reference_valid[agent_id] = True
                self.target_reference_miss_count[agent_id] = 0
                self.pending_target_reference_valid[agent_id] = False
                self.target_reference_confirm_count[agent_id] = 0

    @staticmethod
    def _pil_pair(
        drone_frame_bgr: np.ndarray,
        dog_frame_bgr: np.ndarray,
    ) -> list[Image.Image]:
        images = []
        for frame in (drone_frame_bgr, dog_frame_bgr):
            value = np.asarray(frame)
            if value.ndim == 2:
                value = cv2.cvtColor(
                    value.astype(np.uint8), cv2.COLOR_GRAY2BGR
                )
            if value.ndim != 3 or value.shape[2] < 3:
                raise ValueError(f"Expected BGR image, got {value.shape}")
            value = np.clip(
                value[:, :, :3], 0, 255
            ).astype(np.uint8, copy=False)
            images.append(
                Image.fromarray(cv2.cvtColor(value, cv2.COLOR_BGR2RGB))
            )
        return images

    @staticmethod
    def _perception_tensors(
        predictions: Sequence[Any],
        grid_size: int,
        top_k: int = 8,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = []
        grids = []
        candidates = []
        candidate_masks = []
        for prediction in predictions:
            valid = bool(prediction.person_valid)
            feature = np.r_[
                prediction.person_box_cxcywh_norm,
                float(prediction.person_score),
                float(valid),
            ].astype(np.float32)
            if not valid:
                feature[:5] = 0.0
            features.append(feature)
            grids.append(
                mask_to_grid(
                    prediction.scene_mask,
                    (grid_size, grid_size),
                )
            )
            height, width = prediction.scene_mask.shape
            boxes_xyxy = np.asarray(
                getattr(prediction, "person_candidates_xyxy", np.empty((0, 4))),
                dtype=np.float32,
            )[:top_k]
            scores = np.asarray(
                getattr(prediction, "person_candidate_scores", np.empty((0,))),
                dtype=np.float32,
            )[:top_k]
            padded = np.zeros((top_k, 6), dtype=np.float32)
            count = min(len(boxes_xyxy), len(scores), top_k)
            if count:
                x1, y1, x2, y2 = boxes_xyxy[:count].T
                padded[:count, :4] = np.stack(
                    (
                        (x1 + x2) / (2.0 * width),
                        (y1 + y2) / (2.0 * height),
                        (x2 - x1) / width,
                        (y2 - y1) / height,
                    ),
                    axis=-1,
                )
                padded[:count, 4] = scores[:count]
                padded[:count, 5] = 1.0
            candidates.append(padded)
            candidate_masks.append(np.arange(top_k) < count)
        return (
            torch.from_numpy(np.stack(features)),
            torch.from_numpy(np.stack(grids)),
            torch.from_numpy(np.stack(candidates)),
            torch.from_numpy(np.stack(candidate_masks)),
        )

    def _search_waypoints(
        self,
        template: np.ndarray,
        agent_poses: torch.Tensor | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build rotation-only trajectories toward bounded search endpoints."""

        poses = np.asarray(torch.as_tensor(agent_poses).detach().float().cpu())
        if poses.shape != (2, 4):
            raise ValueError(
                "V3 search expects agent_poses with shape (2,4), got "
                f"{poses.shape}"
            )
        current_yaw = np.arctan2(poses[:, 2], poses[:, 3])
        if self.search_center_yaw_rad is None:
            self.search_center_yaw_rad = current_yaw.astype(np.float32, copy=True)
            self.search_target_direction = np.ones(2, dtype=np.float32)

        center = np.asarray(self.search_center_yaw_rad, dtype=np.float32)
        offset_from_center = np.asarray(
            _wrap_angle(current_yaw - center), dtype=np.float32
        )
        reached_positive = (
            (self.search_target_direction > 0.0)
            & (offset_from_center >= SEARCH_HALF_ANGLE_RAD - SEARCH_ENDPOINT_TOLERANCE_RAD)
        )
        reached_negative = (
            (self.search_target_direction < 0.0)
            & (offset_from_center <= -SEARCH_HALF_ANGLE_RAD + SEARCH_ENDPOINT_TOLERANCE_RAD)
        )
        self.search_target_direction[reached_positive | reached_negative] *= -1.0

        target_yaw = np.asarray(
            _wrap_angle(
                center + self.search_target_direction * SEARCH_HALF_ANGLE_RAD
            ),
            dtype=np.float32,
        )
        yaw_error = np.asarray(
            _wrap_angle(target_yaw - current_yaw), dtype=np.float32
        )
        search = np.zeros_like(template, dtype=np.float32)
        for agent in range(2):
            search[agent, :, 2] = np.linspace(
                0.0,
                float(yaw_error[agent]),
                search.shape[1],
                dtype=np.float32,
            )
        return search, target_yaw, yaw_error

    def _route_waypoints(
        self,
        output: dict[str, torch.Tensor],
        detection: torch.Tensor,
        agent_poses: torch.Tensor | np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
        target_match = (
            output["target_match_probability"][0].detach().float().cpu()
        )
        route = self.visibility_router.update(detection, target_match)
        mode = route["mode"].numpy()
        stable_visible = route["visible"].numpy()
        self_waypoints = (
            output["self_waypoints"][0].detach().float().cpu().numpy()
        )
        cooperative = (
            output["cooperative_waypoints"][0].detach().float().cpu().numpy()
        )
        routed = np.empty_like(self_waypoints)
        both_invisible = bool(route["both_invisible"])
        pose_values = (
            torch.as_tensor(agent_poses).detach().float().cpu().numpy()
        )
        if pose_values.shape != (2, 4):
            raise ValueError(
                "V3 routing expects agent_poses with shape (2,4), got "
                f"{pose_values.shape}"
            )
        current_yaw = np.arctan2(pose_values[:, 2], pose_values[:, 3])
        if both_invisible and self.search_center_yaw_rad is None:
            # Lock the search origin when loss is confirmed, including the
            # short BELIEF hold, rather than several frames later at SEARCH.
            self.search_center_yaw_rad = current_yaw.astype(np.float32, copy=True)
            self.search_target_direction = np.ones(2, dtype=np.float32)
        elif not both_invisible:
            self.search_center_yaw_rad = None
            self.search_target_direction = np.ones(2, dtype=np.float32)

        search_target_yaw: np.ndarray | None = None
        search_yaw_error: np.ndarray | None = None
        search: np.ndarray | None = None
        if bool(np.any(mode == ROUTE_SEARCH)):
            search, search_target_yaw, search_yaw_error = self._search_waypoints(
                self_waypoints, agent_poses
            )
        for agent in range(2):
            if mode[agent] == ROUTE_SELF:
                routed[agent] = self_waypoints[agent]
            elif mode[agent] == ROUTE_COOPERATIVE:
                routed[agent] = cooperative[agent]
            elif mode[agent] == ROUTE_BELIEF:
                routed[agent] = (
                    cooperative[agent]
                    if self.last_navigation_waypoints is None
                    else self.last_navigation_waypoints[agent]
                )
            elif mode[agent] == ROUTE_SEARCH:
                assert search is not None
                routed[agent] = search[agent]
            else:
                raise RuntimeError(f"Unknown V3 routing mode {mode[agent]}")

        # Preserve the latest trajectory supported by at least one verified
        # view. It is reused only for the three-frame both-invisible belief hold.
        if bool(stable_visible.any()):
            self.last_navigation_waypoints = routed.copy()

        mode_names = {
            ROUTE_SELF: "self",
            ROUTE_COOPERATIVE: "cooperative",
            ROUTE_BELIEF: "belief",
            ROUTE_SEARCH: "search",
        }
        debug = {
            "routing_mode": [int(value) for value in mode.tolist()],
            "routing_mode_name": [mode_names[int(value)] for value in mode],
            "verified_visible": [bool(value) for value in stable_visible],
            "both_invisible": both_invisible,
            "route_to_cooperative": [
                bool(value) for value in route["route_to_cooperative"].tolist()
            ],
            "target_match_probability": [
                float(value) for value in target_match.tolist()
            ],
            "search_center_yaw_degrees": (
                None
                if self.search_center_yaw_rad is None
                else np.degrees(self.search_center_yaw_rad).tolist()
            ),
            "search_target_yaw_degrees": (
                None
                if search_target_yaw is None
                else np.degrees(search_target_yaw).tolist()
            ),
            "search_yaw_error_degrees": (
                None
                if search_yaw_error is None
                else np.degrees(search_yaw_error).tolist()
            ),
        }
        return routed, debug, stable_visible

    @torch.inference_mode()
    def observe(
        self,
        drone_frame_bgr: np.ndarray,
        robotdog_frame_bgr: np.ndarray,
        *,
        observation_time: float | None = None,
    ) -> dict[str, float]:
        """只编码中间物理帧并写入 0.1 s 历史，不执行感知或 Qwen。"""
        images = self._pil_pair(drone_frame_bgr, robotdog_frame_bgr)
        try:
            encoding_started = time.perf_counter()
            encoder_amp = self.device.type == "cuda" and self.vision_amp
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=encoder_amp,
            ):
                dino, grid_h, grid_w = self.encoder._encode_dino(images)
                siglip = self.encoder._encode_siglip(
                    images, out_hw=(grid_h, grid_w)
                )
            visual = torch.cat((dino, siglip), dim=-1)
            coarse = grid_pool_tokens(
                visual, grid_h, grid_w, out_tokens=4
            ).float().cpu()
            encoding_seconds = time.perf_counter() - encoding_started
        finally:
            for image in images:
                image.close()
        now = time.monotonic() if observation_time is None else float(observation_time)
        for agent_id in range(2):
            self.histories[agent_id].append((now, coarse[agent_id]))
        return {"encoding_time": float(encoding_seconds)}

    @torch.inference_mode()
    def predict(
        self,
        drone_frame_bgr: np.ndarray,
        robotdog_frame_bgr: np.ndarray,
        _drone_bbox: Any,
        _robotdog_bbox: Any,
        instruction: str,
        *,
        joint_instruction: str | None = None,
        agent1_instruction: str | None = None,
        agent2_instruction: str | None = None,
        observation_time: float | None = None,
        drone_pose: Any = None,
        robotdog_pose: Any = None,
        **_unused: Any,
    ) -> dict[str, Any]:
        if drone_pose is None or robotdog_pose is None:
            raise ValueError("Air-ground V3 inference requires both follower poses")
        images = self._pil_pair(drone_frame_bgr, robotdog_frame_bgr)
        try:
            perception_started = time.perf_counter()
            predictions = self.perception.predict(images)
            perception_seconds = time.perf_counter() - perception_started
            perception_output = self._perception_tensors(
                predictions,
                int(self.model.cfg.perception_grid_size),
            )
            if len(perception_output) == 2:
                detection, perception_grid = perception_output
                candidate_feat = detection.unsqueeze(1)
                candidate_valid = candidate_feat[..., 5] > 0.5
            else:
                detection, perception_grid, candidate_feat, candidate_valid = perception_output

            encoding_started = time.perf_counter()
            encoder_amp = self.device.type == "cuda" and self.vision_amp
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=encoder_amp,
            ):
                dino, grid_h, grid_w = self.encoder._encode_dino(images)
                siglip = self.encoder._encode_siglip(
                    images, out_hw=(grid_h, grid_w)
                )
            visual = torch.cat((dino, siglip), dim=-1)
            coarse = grid_pool_tokens(
                visual, grid_h, grid_w, out_tokens=4
            ).float().cpu()
            fine = grid_pool_tokens(
                visual, grid_h, grid_w, out_tokens=64
            ).float().cpu()
            encoding_seconds = time.perf_counter() - encoding_started
        finally:
            for image in images:
                image.close()

        now = (
            time.monotonic()
            if observation_time is None
            else float(observation_time)
        )
        histories, history_ids = zip(
            *(
                self._history(index, coarse[index], now)
                for index in range(2)
            )
        )
        agent_poses = _agent_poses(drone_pose, robotdog_pose)
        base_text = str(
            self.config.get("instruction_override") or instruction
        )
        joint_text = str(
            joint_instruction
            or self.config.get("joint_instruction_override")
            or base_text
        )
        drone_text = str(
            agent1_instruction
            or self.config.get("agent1_instruction_override")
            or base_text
        )
        dog_text = str(
            agent2_instruction
            or self.config.get("agent2_instruction_override")
            or base_text
        )
        prior_bbox = getattr(
            self, "last_candidate_bbox", np.zeros((2, 4), dtype=np.float32)
        )
        prior_valid = getattr(
            self, "last_candidate_valid", np.zeros(2, dtype=bool)
        )
        # The same temporal identity memory is available to joint STT/DT/AT.
        # Task text still defines the target; memory only carries a confirmed
        # visual identity across time and never selects a task-specific model.
        stored_reference = getattr(self, "target_reference_tokens", None)
        if not getattr(self, "use_target_reference", True) or stored_reference is None:
            reference_tokens = torch.zeros(2, fine.size(-1), dtype=fine.dtype)
        else:
            reference_tokens = stored_reference
        reference_valid = (
            getattr(self, "target_reference_valid", np.zeros(2, dtype=bool))
            if getattr(self, "use_target_reference", True)
            else np.zeros(2, dtype=bool)
        )
        model_inputs = {
            "coarse_tokens": torch.stack(histories).unsqueeze(0).to(self.device),
            "coarse_tidx": torch.stack(history_ids).unsqueeze(0).to(self.device),
            "fine_tokens": fine.unsqueeze(0).to(self.device),
            "fine_tidx": torch.full(
                (1, 2, fine.size(1)),
                self.history,
                dtype=torch.long,
                device=self.device,
            ),
            "detection_feat": detection.unsqueeze(0).to(self.device),
            "candidate_feat": candidate_feat.unsqueeze(0).to(self.device),
            "candidate_valid": candidate_valid.unsqueeze(0).to(self.device),
            "target_reference_tokens": reference_tokens.unsqueeze(0).to(self.device),
            "target_reference_valid": torch.from_numpy(reference_valid).unsqueeze(0).to(self.device),
            "candidate_prior_bbox": torch.from_numpy(prior_bbox).unsqueeze(0).to(self.device),
            "candidate_prior_valid": torch.from_numpy(prior_valid).unsqueeze(0).to(self.device),
            "perception_grid": perception_grid.unsqueeze(0).to(self.device),
            "agent_poses": agent_poses.unsqueeze(0).to(self.device),
            # Receiver corruption/relocation curriculum is training-only.
            "synthetic_occlusion": torch.zeros(
                1, 2, dtype=torch.bool, device=self.device
            ),
            "coarse_missing_mask": torch.zeros(
                1,
                2,
                histories[0].size(0),
                dtype=torch.bool,
                device=self.device,
            ),
            "fine_missing_mask": torch.zeros(
                1,
                2,
                fine.size(1),
                dtype=torch.bool,
                device=self.device,
            ),
            "instructions": [base_text],
            "joint_instructions": [joint_text],
            "drone_instructions": [drone_text],
            "dog_instructions": [dog_text],
            "return_dict": True,
        }
        # CUDA launches are asynchronous. Synchronize around the forward pass
        # so model_time is a real input-to-output latency rather than CPU
        # dispatch time, and so the benchmark cannot hide work in later routing.
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        model_started = time.perf_counter()
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda" and self.inference_amp,
        ):
            output = self.model(**model_inputs)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        model_seconds = time.perf_counter() - model_started

        if "selected_detection_feat" in output:
            if not hasattr(self, "last_candidate_bbox"):
                self.last_candidate_bbox=np.zeros((2,4),dtype=np.float32)
                self.last_candidate_valid=np.zeros(2,dtype=bool)
            selected_now=output["selected_detection_feat"][0].detach().float().cpu().numpy()
            accepted_now=output["candidate_selected_accepted"][0].detach().bool().cpu().numpy()
            for agent_id in range(2):
                if bool(accepted_now[agent_id] and selected_now[agent_id,5]>0.5):
                    self.last_candidate_bbox[agent_id]=selected_now[agent_id,:4]
                    self.last_candidate_valid[agent_id]=True
            if "candidate_selected_probability" in output:
                # 8 个候选分数是独立 sigmoid 概率，不存在第 9 个 NULL softmax 类。
                selected_confidence = output["candidate_selected_probability"][0].detach().float().cpu()
            else:
                selected_confidence = output["target_match_probability"][0].detach().float().cpu()
            if getattr(self, "use_target_reference", True):
                self._update_target_reference_memory(
                    fine,
                    output["selected_detection_feat"][0].detach().float().cpu(),
                    output["candidate_selected_accepted"][0].detach().bool().cpu(),
                    selected_confidence,
                )
        waypoints, routing_debug, stable_visible = self._route_waypoints(
            output, detection, agent_poses
        )
        if "selected_detection_feat" in output:
            selected_detection = output["selected_detection_feat"][0].detach().float().cpu()
            boxes = selected_detection[:, :4].tolist()
            scores = selected_detection[:, 4].tolist()
            valid = selected_detection[:, 5].gt(0.5).tolist()
        else:
            boxes = [prediction.person_box_cxcywh_norm.tolist() for prediction in predictions]
            valid = [bool(prediction.person_valid) for prediction in predictions]
            scores = [float(prediction.person_score) for prediction in predictions]
        target_match = routing_debug["target_match_probability"]
        bbox_motion_observations = []
        for agent, prediction in enumerate(predictions):
            bbox_motion_observations.append(
                self.bbox_motion_controller.observe(
                    agent,
                    prediction.person_box_cxcywh_norm,
                    float(prediction.person_score),
                    bool(stable_visible[agent] and prediction.person_valid),
                )
            )

        mode_logits = (
            output["cooperative_mode_logits"][0].detach().float().cpu()
        )
        mode_probability = torch.softmax(mode_logits, dim=-1)
        selected_modes = mode_logits.argmax(dim=-1)
        target_belief = (
            output["target_belief"][0].detach().float().cpu().numpy()
        )
        uncertainty_log_variance = (
            output["jepa_uncertainty_logit"][0]
            .detach()
            .float()
            .clamp(-5.0, 5.0)
            .cpu()
            .numpy()
        )
        verified_scores = [
            float(scores[index] * target_match[index])
            if valid[index]
            else 0.0
            for index in range(2)
        ]
        # The video-facing box is deliberately stricter than the raw YOLO
        # proposal. A candidate appears as the tracked target only after the
        # LLM verifier and temporal hysteresis have accepted it.
        verified_boxes = _llm_verified_boxes(boxes, stable_visible)

        self.last_waypoints = waypoints
        self.last_policy_debug = {
            **routing_debug,
            "online_person_valid": valid,
            "online_person_scores": scores,
            "verified_target_scores": verified_scores,
            "cooperative_selected_mode": [
                int(value) for value in selected_modes.tolist()
            ],
            "cooperative_mode_probability": mode_probability.tolist(),
            "target_belief": target_belief.tolist(),
            "jepa_uncertainty_log_variance": (
                uncertainty_log_variance.tolist()
            ),
            "bbox_motion_observations": bbox_motion_observations,
            "agent_poses": agent_poses.tolist(),
            "target_reference_valid": getattr(
                self, "target_reference_valid", np.zeros(2, dtype=bool)
            ).tolist(),
            "target_reference_confirm_count": getattr(
                self, "target_reference_confirm_count", np.zeros(2, dtype=np.int64)
            ).tolist(),
            "target_reference_miss_count": getattr(
                self, "target_reference_miss_count", np.zeros(2, dtype=np.int64)
            ).tolist(),
            "perception_time_seconds": perception_seconds,
            "model_time_seconds": model_seconds,
        }
        self_waypoints = (
            output["self_waypoints"][0].detach().float().cpu().numpy()
        )
        cooperative_waypoints = (
            output["cooperative_waypoints"][0]
            .detach()
            .float()
            .cpu()
            .numpy()
        )
        return {
            "waypoints": waypoints,
            "drone_self_waypoints": self_waypoints[0],
            "dog_self_waypoints": self_waypoints[1],
            "drone_cooperative_waypoints": cooperative_waypoints[0],
            "dog_cooperative_waypoints": cooperative_waypoints[1],
            "cooperative_candidates": (
                output["cooperative_candidates"][0]
                .detach()
                .float()
                .cpu()
                .numpy()
            ),
            "bbox_input": boxes,
            "bbox_source": "online_yolo_llm_verified",
            "bbox_source_display": "online_yolo+LLM_VERIFY",
            "refined_bbox": verified_boxes,
            "raw_refined_bbox": boxes,
            "absolute_bbox": boxes,
            "bbox_fallback_to_absolute": [False, False],
            "bbox_display_label": [
                f"LLM verified target p={target_match[0]:.2f}",
                f"LLM verified target p={target_match[1]:.2f}",
            ],
            "visible_score": verified_scores,
            "best_candidate": (
                output.get("candidate_selected_index", torch.full((1, 2), -1, device=self.device))[0]
                .detach().long().cpu().tolist()
            ),
            "best_candidate_score": output.get(
                "candidate_selected_probability",
                torch.as_tensor(target_match, device=self.device).unsqueeze(0),
            )[0].detach().float().cpu().tolist(),
            "best_candidate_margin": output.get(
                "candidate_selected_margin",
                torch.zeros(1, 2, device=self.device),
            )[0].detach().float().cpu().tolist(),
            "candidate_boxes": candidate_feat[..., :4].tolist(),
            "candidate_detector_scores": candidate_feat[..., 4].tolist(),
            "candidate_valid": candidate_valid.tolist(),
            "global_encoding_time": encoding_seconds,
            "perception_time": perception_seconds,
            "model_time": model_seconds,
            "roi_encoding_time": 0.0,
            "roi_bbox_source": "online_yolo_llm_verified",
            "evaluation_protocol": self.evaluation_protocol,
            "roi_valid": [bool(value) for value in stable_visible],
            "roi_crop_xyxy": None,
            "roi_expand_ratio": None,
            "roi_token_count": None,
            **self.last_policy_debug,
        }

    def _adjust_raw_desired_velocities(
        self,
        raw_drone: np.ndarray,
        raw_robotdog: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        return self.bbox_motion_controller.adjust(raw_drone, raw_robotdog)

    def waypoints_to_actions(
        self,
        waypoints: np.ndarray,
        *,
        realtime_control_period_seconds: float | None = None,
    ):
        original = np.asarray(waypoints)
        if original.ndim != 3 or original.shape[0] != 2 or original.shape[2] < 3:
            raise ValueError(
                "V3 waypoints must have shape (2,N,D>=3), got "
                f"{original.shape}"
            )
        robotdog_y_mode = str(
            getattr(
                self.args,
                "robotdog_waypoint_y_mode",
                ROBOTDOG_WAYPOINT_Y_MODE,
            )
        )
        if robotdog_y_mode != ROBOTDOG_WAYPOINT_Y_MODE:
            raise ValueError(
                "Unsupported --robotdog-waypoint-y-mode "
                f"{robotdog_y_mode!r}; expected {ROBOTDOG_WAYPOINT_Y_MODE!r}"
            )
        control_waypoints = original.copy()
        source_dt = float(self.args.waypoint_source_dt or self.args.dt)
        projected_dog, projection = (
            _project_robotdog_waypoints_to_nonholonomic(
                original[1],
                control_dt=float(self.args.dt),
                source_dt=source_dt,
                horizon_steps=int(self.args.waypoint_horizon_steps),
            )
        )
        control_waypoints[1] = projected_dog
        drone_action, dog_action, debug = super().waypoints_to_actions(
            control_waypoints,
            realtime_control_period_seconds=realtime_control_period_seconds,
        )
        dog_index = int(debug["robotdog_waypoint_index"])
        translation_index = min(
            dog_index + int(self.args.ground_translation_delay_steps),
            original.shape[1] - 1,
        )
        if translation_index > dog_index:
            selected_forward = float(
                projected_dog[translation_index, 0] - projected_dog[dog_index, 0]
            )
            selected_residuals = projection["lateral_residuals_m"][
                dog_index:translation_index
            ]
            start_step = max(
                1,
                round(
                    dog_index
                    * max(1, int(self.args.waypoint_horizon_steps))
                    / (original.shape[1] - 1)
                ),
            )
            end_step = max(
                1,
                round(
                    translation_index
                    * max(1, int(self.args.waypoint_horizon_steps))
                    / (original.shape[1] - 1)
                ),
            )
            selected_time = max(
                (end_step - start_step) * source_dt,
                1.0e-6,
            )
        else:
            selected_forward = float(projected_dog[dog_index, 0])
            selected_residuals = projection["lateral_residuals_m"][:dog_index]
            selected_time = max(float(debug["robotdog_horizon_dt"]), 1.0e-6)
        lateral_residual = float(
            np.linalg.norm(selected_residuals) if selected_residuals.size else 0.0
        )

        debug.update(self.last_policy_debug)
        debug["action_source"] = (
            "airground_v3_routed_waypoints_nonholonomic_pose_projection"
        )
        debug["drone_action_source"] = (
            self.last_policy_debug.get("routing_mode_name", ["unknown"])[0]
        )
        debug["robotdog_action_source"] = (
            self.last_policy_debug.get(
                "routing_mode_name", ["unknown", "unknown"]
            )[1]
        )
        # Parent debug saw the projected control waypoint; expose both that and
        # the actual model pose waypoint so trajectory analysis remains honest.
        debug["robotdog_waypoint"] = original[1, dog_index, :3].tolist()
        debug["robotdog_control_waypoint"] = control_waypoints[
            1, dog_index, :3
        ].tolist()
        debug["robotdog_nonholonomic_projected_waypoint"] = projected_dog[
            dog_index, :3
        ].tolist()
        debug["robotdog_translation_waypoint_index"] = int(translation_index)
        debug["robotdog_forward_segment_displacement_m"] = selected_forward
        debug["robotdog_lateral_residual_m"] = lateral_residual
        debug["robotdog_lateral_residual_mps"] = lateral_residual / selected_time
        debug["robotdog_yaw_target_rad"] = float(original[1, dog_index, 2])
        debug["robotdog_yaw_horizon_seconds"] = float(
            debug["robotdog_horizon_dt"]
        )
        debug["robotdog_waypoint_y_control_mode"] = robotdog_y_mode
        debug["robotdog_lateral_ignored"] = 0.0
        return drone_action, dog_action, debug

class RemoteAirGroundCoopV3Planner:
    """每个 worker 使用的远程代理，共享单个常驻 GPU 的 V3 planner。"""

    # 保留共享评估器的旧能力标志，同时暴露 V3 所需能力。
    requires_inter_agent_pose = True
    requires_agent_poses = True
    supports_intermediate_observation = True

    def __init__(self, args: argparse.Namespace):
        socket_path = os.environ.get("EVAL_AIRGROUND_V3_SERVER_SOCKET")
        if not socket_path:
            raise RuntimeError(
                "EVAL_AIRGROUND_V3_SERVER_SOCKET is required for remote inference"
            )
        authkey = os.environ.get(
            "EVAL_AIRGROUND_V3_SERVER_AUTHKEY", "eval_airground_v3"
        ).encode("utf-8")
        self.connection = Client(socket_path, family="AF_UNIX", authkey=authkey)
        self.session = os.environ.get(
            "EVAL_AIRGROUND_V3_SESSION",
            f"pid-{os.getpid()}-{uuid.uuid4().hex}",
        )
        metadata = self._request("init", args=vars(args))
        self.ckpt_path = Path(metadata["ckpt_path"])
        for name in (
            "history", "history_frame_dt", "n_waypoints", "use_roi_tokens",
            "use_bbox_text_prompt", "use_visual_section_markers", "roi_bbox_source",
            "roi_expand_ratio", "roi_token_count", "roi_make_square",
            "evaluation_protocol", "ckpt_bbox_dropout_prob",
            "supports_intermediate_observation",
        ):
            setattr(self, name, metadata[name])
        print(
            f"[airground-v3-client] connected session={self.session} "
            f"server={socket_path}",
            flush=True,
        )

    def _request(self, operation: str, **payload: Any) -> Any:
        self.connection.send(
            {"op": operation, "session": self.session, **payload}
        )
        response = self.connection.recv()
        if not response.get("ok", False):
            raise RuntimeError(
                f"shared air-ground V3 server {operation} failed: "
                f"{response.get('error')}"
            )
        return response["value"]

    def reset(self) -> None:
        self._request("reset")

    def update_realized_velocities(
        self, _drone: list[float], _robotdog: list[float]
    ) -> None:
        # inverse_fixed_dt keeps command-side controller state on the server.
        return None

    def observe(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        return self._request("observe", args=args, kwargs=kwargs)

    def predict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self._request("predict", args=args, kwargs=kwargs)

    def waypoints_to_actions(
        self, *args: Any, **kwargs: Any
    ) -> tuple[list[float], list[float], dict[str, Any]]:
        return self._request("actions", args=args, kwargs=kwargs)


def _parse_entry_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--ckpt")
    parser.add_argument("--save-path")
    parser.add_argument("--help", action="store_true")
    known, _ = parser.parse_known_args(argv)
    if known.help:
        print(__doc__)
        raise SystemExit(0)
    if not known.ckpt:
        raise ValueError("--ckpt is required")
    if not known.save_path:
        raise ValueError("--save-path is required")
    return known


def _runtime_argv(user_argv: Sequence[str]) -> list[str]:
    argv = build_runtime_argv(user_argv)
    # V3 evaluations should run the complete recorded sequence even when a
    # follower falls behind.  The fixed recordings are shorter than this
    # consecutive-loss tolerance, while an explicit caller value still wins.
    _append_default(argv, "--max-lost-steps", "400")
    # 当前 checkpoint 有 8 个连续 token：index 0 是无效原点，index 1..7
    # 对应 0.1..0.7 s，因此最后一个 waypoint 的 source step 必须是 7。
    _append_default(argv, "--waypoint-horizon-steps", "7")
    _append_default(argv, "--waypoint-source-dt", "0.1")
    _append_default(argv, "--drone-max-speed", "2.5")
    _append_default(argv, "--robotdog-max-speed", "2.5")
    _append_default(
        argv,
        "--robotdog-waypoint-y-mode",
        "v3_nonholonomic_projection",
    )
    _append_default(argv, "--bbox-motion-height-tolerance-ratio", "0.20")
    _append_default(argv, "--bbox-motion-height-response-ratio", "0.50")
    _append_default(argv, "--robotdog-bbox-max-speed-gain", "2.00")
    return argv


def _print_protocol(argv: Sequence[str]) -> None:
    print("=" * 79, flush=True)
    print("Air-ground V3 three-stream closed-loop inference", flush=True)
    print("  model input: RGB history + online YOLO + two follower pose tokens", flush=True)
    print("  target oracle: disabled (GT masks/poses are metrics-only)", flush=True)
    print("  flows: drone-self / dog-self / symmetric cooperative", flush=True)
    print(
        "  target box: online YOLO proposal accepted only after LLM VERIFY "
        "+ temporal hysteresis",
        flush=True,
    )
    print(
        "  routing: self / cooperative / short belief / bounded +/-30 deg search",
        flush=True,
    )
    print(
        "  bbox motion:       "
        + (
            "disabled"
            if "--no-bbox-motion-control" in argv
            else "LLM-verified bbox cx -> yaw; initial-height speed correction enabled"
        ),
        flush=True,
    )
    print(
        "  waypoint control: inverse_fixed_dt, "
        f"horizon={_option_value(argv, '--waypoint-horizon-steps')}",
        flush=True,
    )
    print(
        "  robotdog waypoint y: "
        f"{_option_value(argv, '--robotdog-waypoint-y-mode')}",
        flush=True,
    )
    print(
        "  bbox height dead band: "
        f"+/-{100.0 * float(_option_value(argv, '--bbox-motion-height-tolerance-ratio')):.0f}%",
        flush=True,
    )
    print(
        "  robotdog bbox control: max speed gain="
        f"{_option_value(argv, '--robotdog-bbox-max-speed-gain')}, "
        "turn in place outside cx dead band",
        flush=True,
    )
    print("=" * 79, flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    user_argv = list(sys.argv[1:] if argv is None else argv)
    try:
        _parse_entry_args(user_argv)
        runtime_argv = _runtime_argv(user_argv)
    except ValueError as exc:
        print(f"[eval_airground_coop_v3] configuration error: {exc}", file=sys.stderr)
        return 2
    _print_protocol(runtime_argv)

    import eval_unrealzoo_multi_agent as runtime

    runtime.UnrealZooMultiAgentPlanner = (
        RemoteAirGroundCoopV3Planner
        if os.environ.get("EVAL_AIRGROUND_V3_SERVER_SOCKET")
        else AirGroundCoopV3Planner
    )
    old_argv = sys.argv
    try:
        sys.argv = ["eval_airground_coop_v3.py", *runtime_argv]
        return int(runtime.main())
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    raise SystemExit(main())
