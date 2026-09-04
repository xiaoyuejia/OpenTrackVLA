#!/usr/bin/env python3
"""Replay one hand_step episode using recorded or pose-derived velocities.

The source recording has no explicit human base_velocity field.  Its human
velocity is therefore derived from target_pose -> target_pose_after_action and
converted to the same [turn_deg, speed_cm_s] BP command used by the character.
Drone and robotdog use either recorded body-frame ``base_velocity`` or
``commanded_base_velocity``; ``base_velocity`` remains the default.  The
``pose_delta_velocity`` source derives an executable body velocity from two
successive recorded poses under a caller-specified fixed timestep.  It is used
to test whether the BP executor has a stable velocity-to-motion mapping.

This replay uses the fixed-step protocol from collection: the world is paused
while commands are prepared, then one ``resume``/``pause`` pulse advances one
UE tick.  Host-side JSON/image processing cannot add simulation time.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _arg_value(name: str, default: str) -> str:
    prefix = f"{name}="
    for index, value in enumerate(sys.argv):
        if value == name and index + 1 < len(sys.argv):
            return sys.argv[index + 1]
        if value.startswith(prefix):
            return value[len(prefix):]
    return default


_PREPARSED_TIMING_MODE = _arg_value("--timing-mode", "fixed")
if _PREPARSED_TIMING_MODE == "fixed":
    os.environ.setdefault("UNREALZOO_FIXED_TIMESTEP", _arg_value("--dt", "0.1"))
else:
    os.environ.pop("UNREALZOO_FIXED_TIMESTEP", None)

from tools.history.replay_unrealzoo_recording import (  # noqa: E402
    UNREAL_UNITS_PER_METER,
    build_env_args,
    classify_coop_agents,
    dog_args,
    drone_args,
    make_env,
    maybe_set_drone_yaw,
    read_json,
    reset_env,
    restore_pose,
    set_drone_camera,
    set_ground_yaw,
    set_robotdog_camera,
    write_json,
)
from generate_aerial_ground_human_tracking_small import (  # noqa: E402
    capture_color_mask_snapshot,
    get_global_frame,
)
from generate_drone_human_tracking_small import ensure_bgr_uint8  # noqa: E402


def wrap_deg(value: float) -> float:
    return float((float(value) + 180.0) % 360.0 - 180.0)


def row_dt(row: dict[str, Any], fallback: float) -> float:
    for key in ("effective_dt_s", "base_velocity_dt_s", "dt", "training_dt_s"):
        value = row.get(key)
        if isinstance(value, (int, float)) and float(value) > 0.0:
            return float(value)
    return float(fallback)


def valid_pose(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 6:
        return None
    try:
        result = [float(item) for item in value[:6]]
    except Exception:
        return None
    return result if np.isfinite(result).all() else None


def pose_error_m(actual: list[float], expected: list[float]) -> float:
    return float(
        np.linalg.norm(
            np.asarray(actual[:3], dtype=np.float64)
            - np.asarray(expected[:3], dtype=np.float64)
        )
        / UNREAL_UNITS_PER_METER
    )


def yaw_error_deg(actual: list[float], expected: list[float]) -> float:
    return abs(wrap_deg(float(actual[4]) - float(expected[4])))


def next_or_after(
    rows: list[dict[str, Any]], index: int, pose_key: str
) -> list[float]:
    row = rows[index]
    after = valid_pose(row.get(f"{pose_key}_after_action"))
    if after is not None:
        return after
    if index + 1 < len(rows):
        following = valid_pose(rows[index + 1].get(pose_key))
        if following is not None:
            return following
    current = valid_pose(row.get(pose_key))
    if current is None:
        raise ValueError(f"step {index + 1} missing {pose_key}")
    return current


def body_velocity_from_pose_delta(
    before: list[float], after: list[float], dt: float
) -> tuple[float, float, float]:
    """Return local vx, vy in m/s and yaw rate in rad/s."""
    dt = max(float(dt), 1e-6)
    world_delta_m = (
        np.asarray(after[:2], dtype=np.float64)
        - np.asarray(before[:2], dtype=np.float64)
    ) / UNREAL_UNITS_PER_METER
    yaw = math.radians(float(before[4]))
    forward = math.cos(yaw) * world_delta_m[0] + math.sin(yaw) * world_delta_m[1]
    lateral = -math.sin(yaw) * world_delta_m[0] + math.cos(yaw) * world_delta_m[1]
    yaw_rate = math.radians(wrap_deg(float(after[4]) - float(before[4]))) / dt
    return float(forward / dt), float(lateral / dt), float(yaw_rate)


def drone_velocity_action(
    row: dict[str, Any], source: str, dt: float
) -> tuple[list[float], list[float]]:
    raw = row.get(source)
    if not isinstance(raw, (list, tuple)) or len(raw) < 3:
        raise ValueError(f"drone step={row.get('step')} missing {source}")
    velocity = [float(value) for value in raw[:3]]
    # BP_drone expects step-like translation values and a yaw-rate command.
    action = [velocity[0] * dt, velocity[1] * dt, 0.0, velocity[2]]
    return action, velocity


def ground_velocity_action(
    row: dict[str, Any], source: str, dt: float
) -> tuple[list[float], list[float]]:
    raw = row.get(source)
    if not isinstance(raw, (list, tuple)) or len(raw) < 3:
        raise ValueError(f"robotdog step={row.get('step')} missing {source}")
    velocity = [float(value) for value in raw[:3]]
    # BP character expects [turn_delta_deg, forward_speed_cm_s].  Lateral
    # velocity is retained in the report because this actuator cannot execute it.
    action = [math.degrees(velocity[2] * dt), velocity[0] * UNREAL_UNITS_PER_METER]
    return action, velocity


def human_velocity_action(velocity: list[float], dt: float) -> list[float]:
    """Convert a body velocity to the character BP command space."""
    return [math.degrees(float(velocity[2]) * float(dt)), float(velocity[0]) * UNREAL_UNITS_PER_METER]


def pose_delta_velocity(
    row: dict[str, Any],
    index: int,
    rows: list[dict[str, Any]],
    pose_key: str,
    dt: float,
) -> tuple[list[float], list[float], list[float]]:
    """Derive body velocity from one recorded pose transition.

    ``dt`` deliberately comes from the replay protocol rather than a row
    metadata field.  This makes the experiment explicit: all source pose
    intervals are treated as one fixed simulation action of duration ``dt``.
    """
    before = valid_pose(row.get(pose_key))
    after = next_or_after(rows, index, pose_key)
    if before is None:
        raise ValueError(f"step={row.get('step')} missing {pose_key}")
    velocity = list(body_velocity_from_pose_delta(before, after, dt))
    world_delta_m = (
        np.asarray(after[:2], dtype=np.float64)
        - np.asarray(before[:2], dtype=np.float64)
    ) / UNREAL_UNITS_PER_METER
    return velocity, [float(value) for value in before], [float(value) for value in world_delta_m]


def drone_pose_delta_action(
    row: dict[str, Any],
    index: int,
    rows: list[dict[str, Any]],
    dt: float,
) -> tuple[list[float], list[float]]:
    velocity, _before, _world_delta = pose_delta_velocity(
        row, index, rows, "drone_pose", dt
    )
    # BP_drone expects a per-step translation and a yaw-rate command.
    return [velocity[0] * dt, velocity[1] * dt, 0.0, velocity[2]], velocity


def ground_pose_delta_action(
    row: dict[str, Any],
    index: int,
    rows: list[dict[str, Any]],
    dt: float,
) -> tuple[list[float], list[float]]:
    velocity, _before, _world_delta = pose_delta_velocity(
        row, index, rows, "robotdog_pose", dt
    )
    # The character BP has no lateral command.  Keep vy in replay_velocity so
    # the report exposes any unexecutable source component.
    return [math.degrees(velocity[2] * dt), velocity[0] * UNREAL_UNITS_PER_METER], velocity


def human_derived_base_velocity_action(
    row: dict[str, Any], index: int, rows: list[dict[str, Any]], dt: float
) -> tuple[list[float], list[float], list[float]]:
    before = valid_pose(row.get("target_pose"))
    after = next_or_after(rows, index, "target_pose")
    if before is None:
        raise ValueError(f"human step={row.get('step')} missing target_pose")
    velocity = list(body_velocity_from_pose_delta(before, after, dt))
    action = human_velocity_action(velocity, dt)
    world_delta_m = (
        np.asarray(after[:2], dtype=np.float64)
        - np.asarray(before[:2], dtype=np.float64)
    ) / UNREAL_UNITS_PER_METER
    return action, velocity, [float(value) for value in world_delta_m]


def recorded_target_action(row: dict[str, Any]) -> list[float]:
    raw = row.get("target_action")
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        raise ValueError(f"human step={row.get('step')} missing target_action")
    return [float(raw[0]), float(raw[1])]


def recorded_replay_target_velocity(row: dict[str, Any]) -> list[float]:
    raw = row.get("target_replay_velocity")
    if not isinstance(raw, (list, tuple)) or len(raw) < 3:
        raise ValueError(
            f"human step={row.get('step')} missing target_replay_velocity from prior pass"
        )
    velocity = [float(value) for value in raw[:3]]
    if not np.isfinite(np.asarray(velocity, dtype=np.float64)).all():
        raise ValueError(f"human step={row.get('step')} has non-finite target_replay_velocity")
    return velocity


def set_fixed_clock(env, dt: float) -> None:
    unrealcv = env.unwrapped.unrealcv
    unrealcv.set_global_time_dilation(1.0)
    unrealcv.set_max_FPS(1.0 / float(dt))
    unrealcv.set_pause()
    if not unrealcv.get_is_paused():
        raise RuntimeError("fixed-step replay could not pause UE")


def refresh_poses(env) -> list[list[float]]:
    unwrapped = env.unwrapped
    poses, _camera_poses, _images, _masks, _depths = unwrapped.unrealcv.get_pose_img_batch(
        unwrapped.player_list,
        [],
        [False, False, False, False],
    )
    unwrapped.obj_poses = poses
    return poses


def fixed_step(env, actions: list[list[float] | None]) -> tuple[list[list[float]], float]:
    unwrapped = env.unwrapped
    unrealcv = unwrapped.unrealcv
    if not unrealcv.get_is_paused():
        raise RuntimeError("fixed-step replay expected UE to be paused")
    commands = [
        unrealcv.set_move_bp(name, action, return_cmd=True)
        for name, action in zip(unwrapped.player_list, actions)
        if action is not None
    ]
    unrealcv.batch_cmd(commands, None)
    pulse_start = time.perf_counter()
    unrealcv.set_resume()
    unrealcv.set_pause()
    pulse_wall_time = time.perf_counter() - pulse_start
    if not unrealcv.get_is_paused():
        raise RuntimeError("fixed-step replay did not return UE to paused state")
    return refresh_poses(env), pulse_wall_time


def set_recorded_realtime_clock(env) -> None:
    """Pause UE between actions; each action pulse uses a recorded wall duration."""
    unrealcv = env.unwrapped.unrealcv
    unrealcv.set_global_time_dilation(1.0)
    unrealcv.set_max_FPS(120.0)
    unrealcv.set_pause()
    if not unrealcv.get_is_paused():
        raise RuntimeError("recorded-realtime replay could not pause UE")


def _send_actions(env, actions: list[list[float] | None]) -> None:
    unrealcv = env.unwrapped.unrealcv
    commands = [
        unrealcv.set_move_bp(name, action, return_cmd=True)
        for name, action in zip(env.unwrapped.player_list, actions)
        if action is not None
    ]
    unrealcv.batch_cmd(commands, None)


def recorded_realtime_step(
    env, actions: list[list[float] | None], duration_s: float
) -> tuple[list[list[float]], float]:
    """Advance a paused UE world for the recorded real-time action duration."""
    unrealcv = env.unwrapped.unrealcv
    if not unrealcv.get_is_paused():
        raise RuntimeError("recorded-realtime replay expected UE to be paused")
    _send_actions(env, actions)
    pulse_start = time.perf_counter()
    unrealcv.set_resume()
    time.sleep(max(0.0, float(duration_s)))
    unrealcv.set_pause()
    pulse_wall_time = time.perf_counter() - pulse_start
    if not unrealcv.get_is_paused():
        raise RuntimeError("recorded-realtime replay did not return UE to paused state")
    return refresh_poses(env), pulse_wall_time


def apply_recorded_bp_interval(env, names: list[str], interval_ms: int) -> None:
    """Apply the recorded BP interval without changing UE's global clock."""
    interval_ms = max(1, int(interval_ms))
    for name in names:
        try:
            env.unwrapped.unrealcv.set_interval(name, interval_ms)
        except TypeError:
            env.unwrapped.unrealcv.set_interval(interval_ms, name)


def write_video_frame(
    writers: dict[str, cv2.VideoWriter],
    key: str,
    path: Path,
    frame: np.ndarray,
    fps: float,
) -> None:
    frame = ensure_bgr_uint8(frame)
    writer = writers.get(key)
    if writer is None:
        height, width = frame.shape[:2]
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            (int(width), int(height)),
        )
        if not writer.isOpened():
            raise RuntimeError(f"could not open video writer: {path}")
        writers[key] = writer
    writer.write(frame)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recorded-dir",
        type=Path,
        default=Path("/data/hdt/ntv_data/sim_data/experment/hand_step"),
    )
    parser.add_argument("--episode", default="0")
    parser.add_argument(
        "--env-id",
        default="UnrealTrack-Demonstration_Castle-ContinuousColor-v0",
    )
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--ue-interval-ms", type=int, default=100)
    parser.add_argument(
        "--timing-mode",
        choices=("fixed", "recorded_realtime"),
        default="fixed",
        help=(
            "fixed uses one UE fixed tick per row. recorded_realtime disables "
            "the fixed timestep and holds every recorded command for that row's "
            "effective_dt_s wall duration."
        ),
    )
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--render-gpu", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--offscreen", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--launch-retries", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument(
        "--action-source",
        choices=(
            "base_velocity",
            "commanded_base_velocity",
            "pose_delta_velocity",
            "replay_velocity",
        ),
        default="base_velocity",
        help=(
            "base_velocity and commanded_base_velocity use recorded fields; "
            "pose_delta_velocity derives each agent command from adjacent "
            "recorded poses using --dt; replay_velocity executes the exact "
            "replay_velocity field written by a prior pass."
        ),
    )
    parser.add_argument(
        "--target-action-source",
        choices=("recorded_target_action", "derived_pose_velocity", "replay_velocity"),
        default="recorded_target_action",
        help=(
            "Human control source. recorded_target_action replays the original "
            "keyboard/BP command; derived_pose_velocity is only a fallback when "
            "the source lacks target_action; replay_velocity reuses a prior "
            "pass's target_replay_velocity exactly."
        ),
    )
    parser.add_argument("--human-appearance-id", type=int, default=5)
    parser.add_argument("--robotdog-appearance-id", type=int, default=27)
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write-global-video", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.dt <= 0.0:
        raise ValueError("--dt must be positive")
    if args.timing_mode == "fixed" and args.ue_interval_ms != int(round(args.dt * 1000.0)):
        raise ValueError("--ue-interval-ms must equal dt * 1000 for fixed-step replay")
    if args.timing_mode == "fixed":
        os.environ["UNREALZOO_FIXED_TIMESTEP"] = f"{args.dt:.9g}"
    else:
        os.environ.pop("UNREALZOO_FIXED_TIMESTEP", None)
    random.seed(args.seed)
    np.random.seed(args.seed)

    source_dir = args.recorded_dir.expanduser().resolve()
    drone_path = source_dir / f"{args.episode}_drone_info.json"
    dog_path = source_dir / f"{args.episode}_robotdog_info.json"
    stat_path = source_dir / f"{args.episode}.json"
    drone_rows = read_json(drone_path)
    dog_rows = read_json(dog_path)
    stat = read_json(stat_path) if stat_path.is_file() else None
    if not isinstance(drone_rows, list) or not isinstance(dog_rows, list):
        raise ValueError("source info files must contain JSON lists")
    count = min(len(drone_rows), len(dog_rows))
    if args.max_steps > 0:
        count = min(count, args.max_steps)
    if count <= 0:
        raise ValueError("no replay steps")

    env_args = build_env_args(
        args,
        drone_rows[0],
        stat if isinstance(stat, dict) else None,
    )
    env_args.top_view_height = None
    args.save_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.save_dir / "replay_config.json",
        {
            "protocol": f"{args.timing_mode}_{args.action_source}",
            "source_dir": str(source_dir),
            "episode": str(args.episode),
            "env_id": args.env_id,
            "dt": args.dt,
            "configured_ue_interval_ms": args.ue_interval_ms,
            "timing_mode": args.timing_mode,
            "bp_interval_policy": (
                "recorded_per_step" if args.timing_mode == "recorded_realtime" else "fixed_dt"
            ),
            "human_velocity_diagnostic_source": "derived_target_pose_delta",
            "pose_delta_assumption": (
                f"every source pose interval is {args.dt:.9g}s"
                if args.action_source == "pose_delta_velocity"
                else None
            ),
            "target_action_source": args.target_action_source,
            "drone_velocity_source": args.action_source,
            "robotdog_velocity_source": args.action_source,
            "appearance": {
                "human": args.human_appearance_id,
                "robotdog": args.robotdog_appearance_id,
            },
            "save_video": bool(args.save_video),
            "write_global_video": bool(args.write_global_video),
        },
    )

    print(f"[replay] source={source_dir} episode={args.episode} steps={count}", flush=True)
    print(
        f"[replay] human={args.target_action_source} drone={args.action_source} "
        f"robotdog={args.action_source} timing={args.timing_mode} "
        f"dt={args.dt} ue_interval_ms={args.ue_interval_ms}",
        flush=True,
    )

    video_paths = {
        "drone": args.save_dir / f"{args.episode}_drone.mp4",
        "robotdog": args.save_dir / f"{args.episode}_robotdog.mp4",
        "global": args.save_dir / f"{args.episode}_global.mp4",
    }
    video_writers: dict[str, cv2.VideoWriter] = {}
    env = make_env(env_args)
    try:
        reset_env(env, env_args)
        target_id, dog_id, drone_id = classify_coop_agents(env)
        players = env.unwrapped.player_list
        target_name, dog_name, drone_name = (
            players[target_id],
            players[dog_id],
            players[drone_id],
        )
        env.unwrapped.unrealcv.set_appearance(target_name, int(args.human_appearance_id))
        env.unwrapped.unrealcv.set_appearance(dog_name, int(args.robotdog_appearance_id))
        appearances = {
            target_name: int(args.human_appearance_id),
            dog_name: int(args.robotdog_appearance_id),
        }

        first_target = valid_pose(drone_rows[0].get("target_pose"))
        first_dog = valid_pose(dog_rows[0].get("robotdog_pose"))
        first_drone = valid_pose(drone_rows[0].get("drone_pose"))
        if not first_target or not first_dog or not first_drone:
            raise ValueError("missing initial target/dog/drone pose")
        restore_pose(env, target_name, first_target)
        restore_pose(env, dog_name, first_dog, lambda yaw: set_ground_yaw(env, dog_name, yaw))
        restore_pose(env, drone_name, first_drone, lambda yaw: maybe_set_drone_yaw(env, drone_name, yaw))

        first_dog_row = dog_rows[0]
        first_drone_row = drone_rows[0]
        set_robotdog_camera(
            env,
            dog_name,
            dog_id,
            [float(v) for v in first_dog_row.get("robotdog_camera_mount", [140, 0, 110])[:3]],
            float(first_dog_row.get("robotdog_camera_pitch", -15.0)),
            float(first_dog_row.get("robotdog_camera_yaw_offset", 0.0)),
            dog_args(env_args),
        )
        set_drone_camera(
            env,
            drone_name,
            drone_id,
            first_drone,
            float(first_drone_row.get("drone_camera_pitch", -40.0)),
            float(first_drone_row.get("drone_camera_yaw_offset", 0.0)),
            drone_args(env_args),
        )
        if args.timing_mode == "fixed":
            set_fixed_clock(env, args.dt)
        else:
            set_recorded_realtime_clock(env)
        # Reapply after the pause barrier so reset-time BP state cannot move
        # the first measured pose before the first command.
        restore_pose(env, target_name, first_target)
        restore_pose(env, dog_name, first_dog, lambda yaw: set_ground_yaw(env, dog_name, yaw))
        restore_pose(env, drone_name, first_drone, lambda yaw: maybe_set_drone_yaw(env, drone_name, yaw))
        refresh_poses(env)

        replay_drone_rows: list[dict[str, Any]] = []
        replay_dog_rows: list[dict[str, Any]] = []
        target_errors: list[float] = []
        dog_errors: list[float] = []
        drone_errors: list[float] = []
        dog_yaw_errors: list[float] = []
        drone_yaw_errors: list[float] = []
        dog_lateral: list[float] = []
        human_lateral: list[float] = []
        recorded_dts: list[float] = []
        recorded_intervals_ms: list[int] = []
        pulse_wall_times: list[float] = []

        for index in range(count):
            drone_row = drone_rows[index]
            dog_row = dog_rows[index]
            source_dt = row_dt(drone_row, args.dt)
            dt = (
                args.dt
                if args.action_source in {"pose_delta_velocity", "replay_velocity"}
                else source_dt
            )
            if (
                args.timing_mode == "fixed"
                and args.action_source not in {"pose_delta_velocity", "replay_velocity"}
                and abs(dt - args.dt) > 1e-6
            ):
                raise ValueError(
                    f"step {index + 1} has dt={dt}, expected fixed dt={args.dt}; "
                    "use --timing-mode recorded_realtime for variable recorded dt"
                )
            recorded_interval_ms = int(
                round(float(drone_row.get("ue_interval_ms", args.ue_interval_ms)))
            )
            if args.timing_mode == "recorded_realtime":
                apply_recorded_bp_interval(
                    env,
                    [target_name, dog_name, drone_name],
                    recorded_interval_ms,
                )
            recorded_dts.append(float(dt))
            recorded_intervals_ms.append(int(recorded_interval_ms))
            derived_target_action, human_velocity, _human_world_delta = human_derived_base_velocity_action(
                drone_row, index, drone_rows, dt
            )
            if args.target_action_source == "recorded_target_action":
                target_action = recorded_target_action(drone_row)
                # This diagnostic is not the exact keyboard command velocity,
                # but preserves the measured pose-delta estimate for reports.
                target_velocity_used = human_velocity
            elif args.target_action_source == "derived_pose_velocity":
                target_action = derived_target_action
                target_velocity_used = human_velocity
            else:
                target_velocity_used = recorded_replay_target_velocity(drone_row)
                target_action = human_velocity_action(target_velocity_used, dt)
            if args.action_source == "pose_delta_velocity":
                drone_action, drone_velocity = drone_pose_delta_action(
                    drone_row, index, drone_rows, dt
                )
                dog_action, dog_velocity = ground_pose_delta_action(
                    dog_row, index, dog_rows, dt
                )
            elif args.action_source == "replay_velocity":
                drone_action, drone_velocity = drone_velocity_action(
                    drone_row, "replay_velocity", dt
                )
                dog_action, dog_velocity = ground_velocity_action(
                    dog_row, "replay_velocity", dt
                )
            else:
                drone_action, drone_velocity = drone_velocity_action(
                    drone_row, args.action_source, dt
                )
                dog_action, dog_velocity = ground_velocity_action(
                    dog_row, args.action_source, dt
                )
            before_poses = refresh_poses(env)
            before_target = list(before_poses[target_id])
            before_dog = list(before_poses[dog_id])
            before_drone = list(before_poses[drone_id])
            if args.save_video:
                set_robotdog_camera(
                    env,
                    dog_name,
                    dog_id,
                    [float(v) for v in first_dog_row.get("robotdog_camera_mount", [140, 0, 110])[:3]],
                    float(first_dog_row.get("robotdog_camera_pitch", -15.0)),
                    float(first_dog_row.get("robotdog_camera_yaw_offset", 0.0)),
                    dog_args(env_args),
                )
                set_drone_camera(
                    env,
                    drone_name,
                    drone_id,
                    before_drone,
                    float(first_drone_row.get("drone_camera_pitch", -40.0)),
                    float(first_drone_row.get("drone_camera_yaw_offset", 0.0)),
                    drone_args(env_args),
                )
                observation, _masks = capture_color_mask_snapshot(env, include_masks=False)
                write_video_frame(
                    video_writers,
                    "drone",
                    video_paths["drone"],
                    observation[drone_id],
                    1.0 / args.dt,
                )
                write_video_frame(
                    video_writers,
                    "robotdog",
                    video_paths["robotdog"],
                    observation[dog_id],
                    1.0 / args.dt,
                )
                if args.write_global_video:
                    global_frame = get_global_frame(
                        env,
                        env_args,
                        before_target,
                        before_dog,
                        before_drone,
                    )
                    if global_frame is not None:
                        write_video_frame(
                            video_writers,
                            "global",
                            video_paths["global"],
                            global_frame,
                            1.0 / args.dt,
                        )
            actions: list[list[float] | None] = [None for _ in players]
            actions[target_id] = target_action
            actions[dog_id] = dog_action
            actions[drone_id] = drone_action
            if args.timing_mode == "fixed":
                after_poses, pulse_wall_time = fixed_step(env, actions)
            else:
                after_poses, pulse_wall_time = recorded_realtime_step(env, actions, dt)
            pulse_wall_times.append(float(pulse_wall_time))
            after_target = list(after_poses[target_id])
            after_dog = list(after_poses[dog_id])
            after_drone = list(after_poses[drone_id])
            source_target_after = next_or_after(drone_rows, index, "target_pose")
            source_dog_after = next_or_after(dog_rows, index, "robotdog_pose")
            source_drone_after = next_or_after(drone_rows, index, "drone_pose")
            target_err = pose_error_m(after_target, source_target_after)
            dog_err = pose_error_m(after_dog, source_dog_after)
            drone_err = pose_error_m(after_drone, source_drone_after)
            target_errors.append(target_err)
            dog_errors.append(dog_err)
            drone_errors.append(drone_err)
            dog_yaw_errors.append(yaw_error_deg(after_dog, source_dog_after))
            drone_yaw_errors.append(yaw_error_deg(after_drone, source_drone_after))
            dog_lateral.append(abs(float(dog_velocity[1])))
            human_lateral.append(abs(float(human_velocity[1])))

            common = {
                "step": index + 1,
                "dt": dt,
                "source_row_dt_s": source_dt,
                "effective_dt_s": dt,
                "timing_mode": args.timing_mode,
                "fixed_timestep_seconds": args.dt if args.timing_mode == "fixed" else None,
                "recorded_action_duration_s": dt,
                "ue_interval_ms": recorded_interval_ms,
                "target_action": [float(v) for v in target_action],
                "source_target_action": drone_row.get("target_action"),
                "target_action_source": args.target_action_source,
                "derived_target_action": [float(v) for v in derived_target_action],
                "target_replay_velocity": [float(v) for v in target_velocity_used],
                "target_pose": before_target,
                "target_pose_after_action": after_target,
                "source_target_pose": valid_pose(drone_row.get("target_pose")),
                "source_target_pose_after_action": source_target_after,
                "step_wall_time_s": pulse_wall_time,
                "target_after_error_m": target_err,
            }
            replay_drone_rows.append(
                {
                    **common,
                    "base_velocity": drone_row.get("base_velocity"),
                    "commanded_base_velocity": drone_row.get("commanded_base_velocity"),
                    "replay_velocity_source": args.action_source,
                    "replay_velocity": [float(v) for v in drone_velocity],
                    "replay_velocity_derivation": (
                        "pose_delta_fixed_dt" if args.action_source == "pose_delta_velocity" else None
                    ),
                    "base_velocity_units": "mps_mps_radps",
                    "env_action": [float(v) for v in drone_action],
                    "env_action_space": "drone set_move_bp [dx_m, dy_m, dz_m, yaw_rate_radps]",
                    "drone_pose": before_drone,
                    "drone_pose_after_action": after_drone,
                    "source_drone_pose": valid_pose(drone_row.get("drone_pose")),
                    "source_drone_pose_after_action": source_drone_after,
                    "drone_after_error_m": drone_err,
                    "drone_yaw_after_error_deg": drone_yaw_errors[-1],
                    "drone_camera_pitch": first_drone_row.get("drone_camera_pitch", -40.0),
                    "drone_camera_yaw_offset": first_drone_row.get("drone_camera_yaw_offset", 0.0),
                }
            )
            replay_dog_rows.append(
                {
                    **common,
                    "base_velocity": dog_row.get("base_velocity"),
                    "commanded_base_velocity": dog_row.get("commanded_base_velocity"),
                    "replay_velocity_source": args.action_source,
                    "replay_velocity": [float(v) for v in dog_velocity],
                    "replay_velocity_derivation": (
                        "pose_delta_fixed_dt" if args.action_source == "pose_delta_velocity" else None
                    ),
                    "base_velocity_units": "mps_mps_radps",
                    "env_action": [float(v) for v in dog_action],
                    "env_action_space": "robotdog set_move_bp [turn_deg, speed_cm_s]",
                    "robotdog_pose": before_dog,
                    "robotdog_pose_after_action": after_dog,
                    "source_robotdog_pose": valid_pose(dog_row.get("robotdog_pose")),
                    "source_robotdog_pose_after_action": source_dog_after,
                    "robotdog_after_error_m": dog_err,
                    "robotdog_yaw_after_error_deg": dog_yaw_errors[-1],
                    "robotdog_camera_mount": first_dog_row.get("robotdog_camera_mount"),
                    "robotdog_camera_pitch": first_dog_row.get("robotdog_camera_pitch"),
                    "robotdog_camera_yaw_offset": first_dog_row.get("robotdog_camera_yaw_offset"),
                }
            )
            if index < 5 or (index + 1) % 50 == 0:
                print(
                    f"[step {index + 1}] target_err={target_err:.3f}m "
                    f"dog_err={dog_err:.3f}m drone_err={drone_err:.3f}m "
                    f"dog_yaw={dog_yaw_errors[-1]:.2f}deg drone_yaw={drone_yaw_errors[-1]:.2f}deg",
                    flush=True,
                )

        summary = {
            "protocol": f"{args.timing_mode}_{args.action_source}",
            "episode": str(args.episode),
            "steps": count,
            "dt_s": args.dt,
            "configured_ue_interval_ms": args.ue_interval_ms,
            "timing_mode": args.timing_mode,
            "recorded_action_dt_s": {
                "mean": float(np.mean(recorded_dts)),
                "min": float(np.min(recorded_dts)),
                "max": float(np.max(recorded_dts)),
            },
            "recorded_bp_interval_ms": {
                "min": int(np.min(recorded_intervals_ms)),
                "max": int(np.max(recorded_intervals_ms)),
                "unique": sorted(set(recorded_intervals_ms)),
            },
            "actual_action_pulse_wall_time_s": {
                "mean": float(np.mean(pulse_wall_times)),
                "min": float(np.min(pulse_wall_times)),
                "max": float(np.max(pulse_wall_times)),
            },
            "human_velocity_diagnostic_source": "derived_target_pose_delta",
            "target_action_source": args.target_action_source,
            "drone_velocity_source": args.action_source,
            "robotdog_velocity_source": args.action_source,
            "appearances": appearances,
            "mean_target_after_error_m": float(np.mean(target_errors)),
            "max_target_after_error_m": float(np.max(target_errors)),
            "mean_robotdog_after_error_m": float(np.mean(dog_errors)),
            "max_robotdog_after_error_m": float(np.max(dog_errors)),
            "mean_drone_after_error_m": float(np.mean(drone_errors)),
            "max_drone_after_error_m": float(np.max(drone_errors)),
            "mean_robotdog_yaw_error_deg": float(np.mean(dog_yaw_errors)),
            "max_robotdog_yaw_error_deg": float(np.max(dog_yaw_errors)),
            "mean_drone_yaw_error_deg": float(np.mean(drone_yaw_errors)),
            "max_drone_yaw_error_deg": float(np.max(drone_yaw_errors)),
            "mean_robotdog_abs_lateral_velocity_mps": float(np.mean(dog_lateral)),
            "max_robotdog_abs_lateral_velocity_mps": float(np.max(dog_lateral)),
            "mean_human_abs_lateral_velocity_mps": float(np.mean(human_lateral)),
            "max_human_abs_lateral_velocity_mps": float(np.max(human_lateral)),
            "source": {
                "drone_info": str(drone_path),
                "robotdog_info": str(dog_path),
                "stat": str(stat_path) if stat_path.is_file() else None,
            },
            "outputs": {
                "drone_info": str(args.save_dir / f"{args.episode}_drone_info.json"),
                "robotdog_info": str(args.save_dir / f"{args.episode}_robotdog_info.json"),
                "drone_video": str(video_paths["drone"]) if args.save_video else None,
                "robotdog_video": str(video_paths["robotdog"]) if args.save_video else None,
                "global_video": (
                    str(video_paths["global"])
                    if args.save_video and args.write_global_video
                    else None
                ),
            },
        }
        write_json(args.save_dir / f"{args.episode}_drone_info.json", replay_drone_rows)
        write_json(args.save_dir / f"{args.episode}_robotdog_info.json", replay_dog_rows)
        write_json(args.save_dir / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    finally:
        for writer in video_writers.values():
            writer.release()
        try:
            env.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
