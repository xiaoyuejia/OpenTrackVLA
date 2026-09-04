#!/usr/bin/env python3
"""Shared fixed-step waypoint inverse dynamics for preprocessing and training.

The controller consumes geometric waypoints expressed in the current agent
body frame.  It never consumes simulator pose or realized-velocity feedback.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch


INVERSE_CONTROL_VERSION = "recorded_pose_fixed_dt_inverse_v1"


@dataclass(frozen=True)
class InverseDynamicsConfig:
    dt: float = 0.1
    waypoint_index: int = 1
    ground_translation_delay_steps: int = 1
    ground_yaw_gain: float = 0.4
    drone_a_forward: float = 0.969
    drone_b_forward: float = 0.0301
    drone_a_lateral: float = 0.969
    drone_b_lateral: float = 0.0301
    drone_yaw_a: float = 0.464
    drone_yaw_b: float = 0.359
    drone_xy_smoothing_alpha: float = 0.20
    drone_yaw_smoothing_alpha: float = 0.25
    robotdog_speed_smoothing_alpha: float = 0.30
    robotdog_yaw_smoothing_alpha: float = 0.30

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def initial_inverse_state() -> tuple[np.ndarray, np.ndarray]:
    """Return command-rollout actuator state and filtered reference state."""
    return np.zeros((2, 3), dtype=np.float64), np.zeros((2, 3), dtype=np.float64)


def _required_waypoint_mask(valid_mask: np.ndarray, cfg: InverseDynamicsConfig) -> np.ndarray:
    result = np.zeros((2, 3), dtype=bool)
    index = int(cfg.waypoint_index)
    dog_translation_index = index + int(cfg.ground_translation_delay_steps)
    if valid_mask.ndim == 2 and valid_mask.shape[0] == 2 and index < valid_mask.shape[1]:
        result[0, :] = bool(valid_mask[0, index])
        result[1, 2] = bool(valid_mask[1, index])
        if dog_translation_index < valid_mask.shape[1]:
            dog_translation_valid = bool(
                valid_mask[1, index] and valid_mask[1, dog_translation_index]
            )
            result[1, 0] = dog_translation_valid
            # The UnrealZoo ground actor has no lateral actuator.
            result[1, 1] = False
    return result


def inverse_step_numpy(
    waypoints: np.ndarray,
    valid_mask: np.ndarray,
    state_before: np.ndarray,
    reference_before: np.ndarray,
    cfg: InverseDynamicsConfig,
) -> dict[str, np.ndarray]:
    """Convert one two-agent waypoint prediction to normalized BP controls.

    Agent order is ``[drone, robotdog]``. Controls use physical normalized
    units ``[forward_mps, right_mps, yaw_command_radps]``. For robotdog, the
    third value is the BP-equivalent yaw command rate, i.e. desired yaw rate
    divided by ``ground_yaw_gain``.
    """
    points = np.asarray(waypoints, dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=bool)
    state = np.asarray(state_before, dtype=np.float64).reshape(2, 3)
    reference = np.asarray(reference_before, dtype=np.float64).reshape(2, 3)
    if points.ndim != 3 or points.shape[0] != 2 or points.shape[2] < 3:
        raise ValueError(f"Expected waypoint shape (2,M,3), got {points.shape}")
    if mask.shape != points.shape[:2]:
        raise ValueError(f"Expected valid mask shape {points.shape[:2]}, got {mask.shape}")

    index = int(cfg.waypoint_index)
    if index <= 0 or index >= points.shape[1]:
        raise ValueError(f"waypoint_index={index} is unavailable for {points.shape[1]} waypoints")
    dt = float(cfg.dt)
    dog_translation_index = min(
        index + int(cfg.ground_translation_delay_steps), points.shape[1] - 1
    )

    raw_drone = points[0, index, :3] / dt
    if dog_translation_index > index:
        dog_xy = (
            points[1, dog_translation_index, :2] - points[1, index, :2]
        ) / ((dog_translation_index - index) * dt)
    else:
        dog_xy = points[1, index, :2] / (index * dt)
    raw_dog = np.asarray(
        [dog_xy[0], dog_xy[1], points[1, index, 2] / (index * dt)],
        dtype=np.float64,
    )

    desired_drone = reference[0].copy()
    desired_drone[:2] += float(cfg.drone_xy_smoothing_alpha) * (
        raw_drone[:2] - desired_drone[:2]
    )
    desired_drone[2] += float(cfg.drone_yaw_smoothing_alpha) * (
        raw_drone[2] - desired_drone[2]
    )
    desired_dog = reference[1].copy()
    desired_dog[:2] += float(cfg.robotdog_speed_smoothing_alpha) * (
        raw_dog[:2] - desired_dog[:2]
    )
    desired_dog[2] += float(cfg.robotdog_yaw_smoothing_alpha) * (
        raw_dog[2] - desired_dog[2]
    )
    reference_after = np.stack([desired_drone, desired_dog], axis=0)

    a_xy = np.asarray([cfg.drone_a_forward, cfg.drone_a_lateral], dtype=np.float64)
    b_xy = np.asarray([cfg.drone_b_forward, cfg.drone_b_lateral], dtype=np.float64)
    drone_xy_command = (desired_drone[:2] - a_xy * state[0, :2]) / b_xy
    drone_yaw_command = (
        desired_drone[2] - float(cfg.drone_yaw_a) * state[0, 2]
    ) / float(cfg.drone_yaw_b)
    control = np.asarray(
        [
            [drone_xy_command[0], drone_xy_command[1], drone_yaw_command],
            [desired_dog[0], 0.0, desired_dog[2] / float(cfg.ground_yaw_gain)],
        ],
        dtype=np.float64,
    )
    drone_state_after = np.asarray(
        [
            a_xy[0] * state[0, 0] + b_xy[0] * drone_xy_command[0],
            a_xy[1] * state[0, 1] + b_xy[1] * drone_xy_command[1],
            float(cfg.drone_yaw_a) * state[0, 2]
            + float(cfg.drone_yaw_b) * drone_yaw_command,
        ],
        dtype=np.float64,
    )
    state_after = np.stack([drone_state_after, desired_dog], axis=0)
    env_action = np.asarray(
        [
            [control[0, 0] * dt, control[0, 1] * dt, 0.0, control[0, 2]],
            [np.degrees(control[1, 2] * dt), control[1, 0] * 100.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    return {
        "raw_desired_velocity": np.stack([raw_drone, raw_dog], axis=0),
        "desired_velocity": reference_after,
        "control": control,
        "env_action": env_action,
        "valid_mask": _required_waypoint_mask(mask, cfg),
        "state_after": state_after,
        "reference_after": reference_after,
    }


def inverse_step_torch(
    waypoints: torch.Tensor,
    state_before: torch.Tensor,
    reference_before: torch.Tensor,
    cfg: InverseDynamicsConfig,
) -> torch.Tensor:
    """Differentiable batched counterpart of :func:`inverse_step_numpy`."""
    if waypoints.ndim != 4 or waypoints.shape[1] != 2 or waypoints.shape[-1] < 3:
        raise ValueError(f"Expected waypoint shape (B,2,M,3), got {tuple(waypoints.shape)}")
    index = int(cfg.waypoint_index)
    if index <= 0 or index >= waypoints.shape[2]:
        raise ValueError(f"waypoint_index={index} is unavailable for {waypoints.shape[2]} waypoints")
    dt = float(cfg.dt)
    dog_translation_index = min(
        index + int(cfg.ground_translation_delay_steps), waypoints.shape[2] - 1
    )
    raw_drone = waypoints[:, 0, index, :3] / dt
    if dog_translation_index > index:
        dog_xy = (
            waypoints[:, 1, dog_translation_index, :2]
            - waypoints[:, 1, index, :2]
        ) / ((dog_translation_index - index) * dt)
    else:
        dog_xy = waypoints[:, 1, index, :2] / (index * dt)
    raw_dog = torch.cat(
        [dog_xy, waypoints[:, 1, index, 2:3] / (index * dt)], dim=-1
    )

    drone_alpha = raw_drone.new_tensor(
        [cfg.drone_xy_smoothing_alpha, cfg.drone_xy_smoothing_alpha, cfg.drone_yaw_smoothing_alpha]
    )
    dog_alpha = raw_dog.new_tensor(
        [cfg.robotdog_speed_smoothing_alpha, cfg.robotdog_speed_smoothing_alpha, cfg.robotdog_yaw_smoothing_alpha]
    )
    desired_drone = reference_before[:, 0] + drone_alpha * (
        raw_drone - reference_before[:, 0]
    )
    desired_dog = reference_before[:, 1] + dog_alpha * (
        raw_dog - reference_before[:, 1]
    )
    a_xy = waypoints.new_tensor([cfg.drone_a_forward, cfg.drone_a_lateral])
    b_xy = waypoints.new_tensor([cfg.drone_b_forward, cfg.drone_b_lateral])
    drone_xy = (desired_drone[:, :2] - a_xy * state_before[:, 0, :2]) / b_xy
    drone_yaw = (
        desired_drone[:, 2:3] - float(cfg.drone_yaw_a) * state_before[:, 0, 2:3]
    ) / float(cfg.drone_yaw_b)
    drone_control = torch.cat([drone_xy, drone_yaw], dim=-1)
    dog_control = torch.stack(
        [
            desired_dog[:, 0],
            torch.zeros_like(desired_dog[:, 0]),
            desired_dog[:, 2] / float(cfg.ground_yaw_gain),
        ],
        dim=-1,
    )
    return torch.stack([drone_control, dog_control], dim=1)
