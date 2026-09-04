#!/usr/bin/env python3
"""V3-only temporal-history, inverse-control, and CLI policy helpers.

Model construction and online perception live in :mod:`eval_airground_coop_v3`.
This module intentionally contains no loader or implementation for older model
families.
"""

from __future__ import annotations

import argparse
import math
from collections import deque
from dataclasses import dataclass
from typing import Any
from typing import Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class EvaluationPolicy:
    """The fixed protocol used for reported AirGround-Coop V3 metrics."""

    bbox_source: str = "none"
    drone_min_follow_distance_m: float = 1.0
    drone_max_follow_distance_m: float = 6.0
    robotdog_min_follow_distance_m: float = 1.0
    robotdog_max_follow_distance_m: float = 6.0
    human_collision_distance_m: float = 0.5
    waypoint_control_mode: str = "inverse_fixed_dt"
    policy_inference_stride: int = 5


POLICY = EvaluationPolicy()


def _clean_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remove the DDP prefix without changing any model weights."""

    if not isinstance(state, dict):
        raise TypeError("Checkpoint model_state must be a dictionary")
    return {
        (name.removeprefix("module.")): value
        for name, value in state.items()
    }


def _future_step(index: int, waypoint_count: int, horizon_steps: int) -> int:
    """Map an origin-inclusive waypoint token to its source action step."""

    if index <= 0 or waypoint_count <= 1:
        raise ValueError("A control waypoint must be a future point (index >= 1)")
    return max(1, round(index * max(1, horizon_steps) / (waypoint_count - 1)))


class AirGroundV3RuntimePlanner:
    """Reusable state/history and inverse controller for the V3 planner.

    The concrete subclass must construct the V3 model and visual frontends.
    """

    UNREAL_UNITS_PER_METER = 100.0

    # Model construction is implemented by the concrete V3 planner.

    def reset(self) -> None:
        for history in self.histories:
            history.clear()
        self.inverse_velocity_state = None
        self.inverse_velocity_reference = None
        self.last_inverse_command = None
        self.last_waypoints = None

    def update_realized_velocities(self, _drone: list[float], _robotdog: list[float]) -> None:
        """No pose feedback: inverse dynamics state is command-side only."""

    def _adjust_raw_desired_velocities(
        self,
        raw_drone: np.ndarray,
        raw_robotdog: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Optional policy-specific adjustment before smoothing/inversion.

        The default V3 runtime leaves waypoint-derived velocities unchanged.
        Online-perception policies may override this
        hook to add causal image-space feedback without modifying model
        waypoints or actuator-space commands.
        """

        return raw_drone, raw_robotdog, {}

    def _history(self, agent: int, coarse: torch.Tensor, now: float) -> tuple[torch.Tensor, torch.Tensor]:
        entries = list(self.histories[agent])
        if not entries:
            frames = [coarse.cpu()]
        else:
            frames = []
            cursor = 0
            for index in range(self.history):
                target_time = now - (self.history - index) * self.history_frame_dt
                while cursor + 1 < len(entries) and entries[cursor + 1][0] <= target_time + 1e-6:
                    cursor += 1
                frames.append(entries[cursor][1])
        frames = [frames[0]] * max(0, self.history - len(frames)) + frames
        self.histories[agent].append((now, coarse.cpu()))
        tokens = torch.cat(frames, dim=0)
        time_ids = torch.cat([
            torch.full((frame.size(0),), index, dtype=torch.long)
            for index, frame in enumerate(frames)
        ])
        return tokens, time_ids

    # Online perception and model forward are implemented by the concrete V3 planner.

    def waypoints_to_actions(
        self,
        waypoints: np.ndarray,
        *,
        realtime_control_period_seconds: float | None = None,
    ) -> tuple[list[float], list[float], dict[str, Any]]:
        """Convert model waypoints through the calibrated 0.1-s inverse model."""

        del realtime_control_period_seconds
        if not self.args.deterministic_step or abs(float(self.args.dt) - 0.1) > 1e-8:
            raise ValueError("AirGround-Coop V3 inverse_fixed_dt evaluation requires deterministic --dt 0.1")
        count = int(waypoints.shape[1])
        horizon = max(1, int(self.args.waypoint_horizon_steps))
        drone_index = int(np.clip(self.args.drone_waypoint_index, 1, count - 1))
        dog_index = int(np.clip(self.args.robotdog_waypoint_index, 1, count - 1))
        source_dt = float(self.args.waypoint_source_dt or self.args.dt)

        def waypoint_time(index: int) -> tuple[int, float]:
            step = _future_step(index, count, horizon)
            return step, max(step * source_dt, 1e-6)

        drone_step, drone_time = waypoint_time(drone_index)
        dog_step, dog_time = waypoint_time(dog_index)
        dt = float(self.args.dt)
        raw_drone = np.asarray(waypoints[0, drone_index, :3], dtype=np.float64) / drone_time
        state = np.zeros((2, 3), dtype=np.float64) if self.inverse_velocity_state is None else self.inverse_velocity_state
        previous_drone = state[0]
        next_dog_index = min(dog_index + int(self.args.ground_translation_delay_steps), count - 1)
        if next_dog_index > dog_index:
            _, next_time = waypoint_time(next_dog_index)
            dog_xy = (waypoints[1, next_dog_index, :2] - waypoints[1, dog_index, :2]) / max(next_time - dog_time, 1e-6)
        else:
            dog_xy = waypoints[1, dog_index, :2] / dog_time
        raw_dog = np.asarray((dog_xy[0], dog_xy[1], waypoints[1, dog_index, 2] / dt), dtype=np.float64)

        raw_drone_before_adjustment = raw_drone.copy()
        raw_dog_before_adjustment = raw_dog.copy()
        raw_drone, raw_dog, velocity_adjustment = self._adjust_raw_desired_velocities(
            raw_drone,
            raw_dog,
        )
        raw_drone = np.asarray(raw_drone, dtype=np.float64)
        raw_dog = np.asarray(raw_dog, dtype=np.float64)
        if raw_drone.shape != (3,) or raw_dog.shape != (3,):
            raise ValueError("Raw desired velocity adjustment must preserve [x,y,yaw] shape")
        if not np.isfinite(raw_drone).all() or not np.isfinite(raw_dog).all():
            raise ValueError("Raw desired velocity adjustment produced non-finite values")

        reference = np.zeros((2, 3), dtype=np.float64) if self.inverse_velocity_reference is None else self.inverse_velocity_reference
        desired_drone = reference[0] + np.asarray([
            self.args.drone_inverse_xy_smoothing_alpha * (raw_drone[0] - reference[0, 0]),
            self.args.drone_inverse_xy_smoothing_alpha * (raw_drone[1] - reference[0, 1]),
            self.args.drone_inverse_yaw_smoothing_alpha * (raw_drone[2] - reference[0, 2]),
        ])
        desired_dog = reference[1] + np.asarray([
            self.args.robotdog_inverse_speed_smoothing_alpha * (raw_dog[0] - reference[1, 0]),
            self.args.robotdog_inverse_speed_smoothing_alpha * (raw_dog[1] - reference[1, 1]),
            self.args.robotdog_inverse_yaw_smoothing_alpha * (raw_dog[2] - reference[1, 2]),
        ])
        self.inverse_velocity_reference = np.asarray((desired_drone, desired_dog))
        a_xy = np.asarray((self.args.drone_inverse_a_forward, self.args.drone_inverse_a_lateral), dtype=np.float64)
        b_xy = np.asarray((self.args.drone_inverse_b_forward, self.args.drone_inverse_b_lateral), dtype=np.float64)
        command_xy = (desired_drone[:2] - a_xy * previous_drone[:2]) / np.maximum(b_xy, 1e-6)
        yaw_command = (desired_drone[2] - self.args.drone_inverse_yaw_a * previous_drone[2]) / max(self.args.drone_inverse_yaw_b, 1e-6)
        drone_action = [float(command_xy[0] * dt), float(command_xy[1] * dt), 0.0, float(yaw_command)]
        dog_turn = math.degrees(float(desired_dog[2]) * dt) / max(float(self.args.ground_yaw_gain), 1e-6)
        dog_action = [float(dog_turn), float(desired_dog[0] * self.UNREAL_UNITS_PER_METER)]
        predicted_drone = np.asarray((
            a_xy[0] * previous_drone[0] + b_xy[0] * command_xy[0],
            a_xy[1] * previous_drone[1] + b_xy[1] * command_xy[1],
            self.args.drone_inverse_yaw_a * previous_drone[2] + self.args.drone_inverse_yaw_b * yaw_command,
        ))
        self.inverse_velocity_state = np.asarray((predicted_drone, desired_dog))
        command = np.asarray(((command_xy[0], command_xy[1], yaw_command), (desired_dog[0], 0.0, desired_dog[2])))
        delta = np.zeros((2, 3)) if self.last_inverse_command is None else command - self.last_inverse_command
        self.last_inverse_command = command
        return drone_action, dog_action, {
            "action_source": "model_waypoint_inverse_fixed_dt",
            "waypoint_control_mode": "inverse_fixed_dt",
            "waypoint_index": int(self.args.waypoint_index),
            "drone_waypoint_index": drone_index,
            "robotdog_waypoint_index": dog_index,
            "waypoint_horizon_steps": horizon,
            "waypoint_source_dt_seconds": source_dt,
            "drone_waypoint_source_step": drone_step,
            "robotdog_waypoint_source_step": dog_step,
            "drone_horizon_dt": drone_time,
            "robotdog_horizon_dt": dog_time,
            "drone_waypoint": waypoints[0, drone_index, :3].tolist(),
            "robotdog_waypoint": waypoints[1, dog_index, :3].tolist(),
            "drone_velocity_pred": desired_drone.tolist(),
            "robotdog_velocity_pred": desired_dog.tolist(),
            "raw_drone_velocity_before_adjustment": raw_drone_before_adjustment.tolist(),
            "raw_robotdog_velocity_before_adjustment": raw_dog_before_adjustment.tolist(),
            "raw_drone_velocity_after_adjustment": raw_drone.tolist(),
            "raw_robotdog_velocity_after_adjustment": raw_dog.tolist(),
            "velocity_adjustment": velocity_adjustment,
            "drone_action_dt_seconds": dt,
            "drone_physical_velocity_command": [float(command_xy[0]), float(command_xy[1]), 0.0],
            "robotdog_physical_velocity_command": [float(desired_dog[0]), 0.0, float(desired_dog[2])],
            "drone_yaw_command": float(yaw_command),
            "robotdog_lateral_ignored": float(desired_dog[1]),
            "drone_inverse": {"state_source": "internal_command_rollout_no_pose_feedback", "command_delta": delta[0].tolist()},
            "robotdog_inverse": {"state_source": "command_only", "command_delta": delta[1].tolist()},
        }


def _option_value(argv: Sequence[str], option: str) -> str | None:
    """Return an option value for ``--name value`` or ``--name=value``."""

    equals_prefix = f"{option}="
    for index, token in enumerate(argv):
        if token.startswith(equals_prefix):
            return token[len(equals_prefix) :]
        if token == option:
            if index + 1 >= len(argv):
                raise ValueError(f"{option} requires a value")
            return argv[index + 1]
    return None


def _has_flag(argv: Sequence[str], option: str) -> bool:
    return option in argv or any(token.startswith(f"{option}=") for token in argv)


def _require_value(argv: Sequence[str], option: str, expected: str) -> None:
    actual = _option_value(argv, option)
    if actual is not None and actual != expected:
        raise ValueError(f"{option} must be {expected!r} for AirGround-Coop V3, got {actual!r}")


def _reject_oracles(argv: Sequence[str]) -> None:
    """Fail closed when a command tries to replace model control or inputs."""

    _require_value(argv, "--bbox-source", POLICY.bbox_source)
    _require_value(argv, "--drone-min-follow-dist", str(POLICY.drone_min_follow_distance_m))
    _require_value(argv, "--drone-max-follow-dist", str(POLICY.drone_max_follow_distance_m))
    _require_value(argv, "--robotdog-min-follow-dist", str(POLICY.robotdog_min_follow_distance_m))
    _require_value(argv, "--robotdog-max-follow-dist", str(POLICY.robotdog_max_follow_distance_m))
    _require_value(argv, "--oracle-drone-action-source", "none")
    _require_value(argv, "--oracle-robotdog-action-source", "none")
    _require_value(argv, "--waypoint-control-mode", POLICY.waypoint_control_mode)

    forbidden_flags = (
        "--oracle-heading-assist",
        "--use-roi-tokens",  # current ROI mode crops with a GT target box
        "--face-target-before-step",
    )
    for option in forbidden_flags:
        if _has_flag(argv, option):
            raise ValueError(f"{option} is not allowed: V3 is model-only evaluation")

    # Boolean argparse options need separate checks because ``--no-*`` is a
    # legal and harmless explicit spelling.
    if _has_flag(argv, "--bbox-source") and _option_value(argv, "--bbox-source") != "none":
        raise ValueError("Ground-truth or recurrent bbox inputs are not allowed")


def _append_default(argv: list[str], option: str, value: str) -> None:
    if _option_value(argv, option) is None:
        argv.extend((option, value))


def build_runtime_argv(user_argv: Sequence[str]) -> list[str]:
    """Validate the strict protocol and fill in its auditable defaults."""

    argv = list(user_argv)
    _reject_oracles(argv)

    _append_default(argv, "--bbox-source", POLICY.bbox_source)
    _append_default(argv, "--drone-min-follow-dist", str(POLICY.drone_min_follow_distance_m))
    _append_default(argv, "--drone-max-follow-dist", str(POLICY.drone_max_follow_distance_m))
    _append_default(argv, "--robotdog-min-follow-dist", str(POLICY.robotdog_min_follow_distance_m))
    _append_default(argv, "--robotdog-max-follow-dist", str(POLICY.robotdog_max_follow_distance_m))
    _append_default(argv, "--human-collision-distance", str(POLICY.human_collision_distance_m))
    _append_default(argv, "--waypoint-control-mode", POLICY.waypoint_control_mode)
    _append_default(argv, "--policy-inference-stride", str(POLICY.policy_inference_stride))
    _append_default(argv, "--oracle-drone-action-source", "none")
    _append_default(argv, "--oracle-robotdog-action-source", "none")

    # Make the safe Boolean settings explicit in the process command line and
    # saved setup JSON.  ``face-target-before-step`` is a one-way legacy
    # switch, so its absence is the explicit safe/default state.
    if not _has_flag(argv, "--no-oracle-heading-assist"):
        argv.append("--no-oracle-heading-assist")
    return argv

