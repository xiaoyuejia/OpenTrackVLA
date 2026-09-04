#!/usr/bin/env python3
"""Lightweight UnrealZoo recording replay.

This script does not load a model and does not use the eval deterministic
pause/resume step.  It reuses the collection-time ``data_collection_step_pose_only``
path and replays recorded per-agent actions from *_info.json files.
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
from types import SimpleNamespace
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
UNREALZOO_ROOT = REPO_ROOT / "unrealzoo-gym"
DATA_RECORDING_DIR = UNREALZOO_ROOT / "example" / "DataRecording"
for _path in (UNREALZOO_ROOT, DATA_RECORDING_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def _preparse_env_id(default: str = "UnrealTrack-Demonstration_Castle-ContinuousColor-v0") -> str:
    for idx, arg in enumerate(sys.argv):
        if arg == "--env-id" and idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
        if arg.startswith("--env-id="):
            return arg.split("=", 1)[1]
    return default


os.environ.setdefault("UNREALZOO_FAST_ENV_ID", _preparse_env_id())

from generate_aerial_ground_human_tracking_small import (  # noqa: E402
    action_for_target_space,
    classify_coop_agents,
    data_collection_step_pose_only,
    dog_args,
    drone_args,
    make_env,
    reset_env,
)
from generate_drone_human_tracking_small import (  # noqa: E402
    UNREAL_UNITS_PER_METER,
    distance_xy_m,
    maybe_set_drone_yaw,
    pose_xyz,
    safe_slug,
    set_drone_camera,
    yaw_deg,
)
from generate_robotdog_human_tracking_small import (  # noqa: E402
    set_ground_yaw,
    set_robotdog_camera,
)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def unwrap_episode_source(source_dir: Path, episode: str) -> tuple[Path, Path, Path | None]:
    drone_path = source_dir / f"{episode}_drone_info.json"
    dog_path = source_dir / f"{episode}_robotdog_info.json"
    stat_path = source_dir / f"{episode}.json"
    if not drone_path.is_file():
        raise FileNotFoundError(drone_path)
    if not dog_path.is_file():
        raise FileNotFoundError(dog_path)
    return drone_path, dog_path, stat_path if stat_path.is_file() else None


def wrap_deg(value: float) -> float:
    return float((value + 180.0) % 360.0 - 180.0)


def pose_error_m(a: list[float], b: list[float]) -> float:
    return float(np.linalg.norm(np.asarray(a[:3], dtype=np.float64) - np.asarray(b[:3], dtype=np.float64)) / UNREAL_UNITS_PER_METER)


def yaw_error_deg(a: list[float], b: list[float]) -> float:
    return abs(wrap_deg(float(a[4]) - float(b[4])))


def restore_pose(env, obj_name: str, pose: list[float], yaw_setter=None) -> None:
    env.unwrapped.unrealcv.set_obj_location(obj_name, pose[:3])
    try:
        env.unwrapped.unrealcv.set_obj_rotation(obj_name, pose[3:6])
    except Exception:
        pass
    if yaw_setter is not None:
        try:
            yaw_setter(float(pose[4]))
        except Exception:
            pass


def recorded_drone_action(record: dict[str, Any], source: str, dt: float) -> list[float]:
    raw = record.get(source)
    if not isinstance(raw, list):
        raise ValueError(f"drone record step={record.get('step')} missing {source}")
    values = [float(v) for v in raw]
    if len(values) >= 4:
        return [values[0], values[1], values[2], values[3]]
    if len(values) >= 3:
        return [values[0] * dt, values[1] * dt, 0.0, values[2]]
    raise ValueError(f"invalid drone {source}: {raw}")


def recorded_dog_action(record: dict[str, Any], source: str) -> Any:
    for key in [source, "env_action", "ground_action", "controller_ground_action"]:
        raw = record.get(key)
        if isinstance(raw, list) and len(raw) >= 2:
            return [float(raw[0]), float(raw[1])]
    return None


def build_env_args(args: argparse.Namespace, first_drone: dict[str, Any], stat: dict[str, Any] | None) -> SimpleNamespace:
    dt = float(args.dt if args.dt is not None else first_drone.get("dt", 0.1))
    ue_interval_ms = int(args.ue_interval_ms if args.ue_interval_ms is not None else first_drone.get("ue_interval_ms", 1000))
    return SimpleNamespace(
        env_id=args.env_id,
        width=args.width,
        height=args.height,
        fps=10,
        dt=dt,
        ue_interval_ms=ue_interval_ms,
        offscreen=args.offscreen,
        render_gpu=args.render_gpu,
        disable_ue_input=True,
        time_dilation=-1,
        seed=args.seed,
        launch_retries=args.launch_retries,
        require_visual_target=False,
        require_centered_target=False,
        use_mask_visibility=True,
        min_visible_ratio=0.001,
        target_center_tolerance=0.35,
        open_spawn=True,
        min_open_clearance=300.0,
        ground_navmesh_tolerance=300.0,
        drone_navmesh_tolerance=800.0,
        robotdog_ideal_follow_dist=6.25,
        robotdog_min_follow_dist=3.5,
        robotdog_max_follow_dist=8.0,
        robotdog_max_speed=1.2,
        robotdog_max_lateral_speed=0.45,
        robotdog_max_yaw_rate=1.0,
        robotdog_camera_forward=140.0,
        robotdog_camera_lateral=0.0,
        robotdog_camera_height=110.0,
        robotdog_camera_mounts="140:0:110,170:0:120,110:0:95,0:120:110,0:90:100,40:90:110,40:-90:110",
        robotdog_camera_fixed_pitch=None,
        robotdog_camera_pitches="-15,-8,0,8,15,22,-22",
        robotdog_camera_yaw_offsets="0,-8,8,-15,15",
        robotdog_camera_mode="fixed",
        robotdog_fov=95.0,
        max_self_visible_ratio=0.015,
        drone_ideal_follow_dist=4.25,
        drone_min_follow_dist=3.0,
        drone_max_follow_dist=6.5,
        drone_height=400.0,
        drone_max_speed=1.2,
        follow_behind=True,
        drone_camera_fixed_pitch=-60.0,
        drone_camera_pitches="-60",
        drone_camera_fixed_yaw=0.0,
        drone_camera_yaw_offsets="0",
        drone_camera_mode="fixed",
        lock_drone_camera_world_xy=True,
        drone_camera_forward_offset=35.0,
        drone_camera_z_offset=-60.0,
        drone_fov=100.0,
        max_camera_search_candidates=12,
        snap_heading=bool(first_drone.get("snap_heading", False)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay one recorded UnrealZoo multi-agent episode without model eval overhead.")
    parser.add_argument("--recorded-dir", type=Path, required=True)
    parser.add_argument("--episode", default="0")
    parser.add_argument("--env-id", default="UnrealTrack-Demonstration_Castle-ContinuousColor-v0")
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=0, help="0 means all recorded steps.")
    parser.add_argument("--drone-action-source", default="env_action", choices=["env_action", "drone_action", "commanded_base_velocity", "base_velocity"])
    parser.add_argument("--dog-action-source", default="env_action")
    parser.add_argument("--target-replay", default="action", choices=["action", "pose_action", "pose_only"])
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--ue-interval-ms", type=int, default=None)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--render-gpu", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--offscreen", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--launch-retries", type=int, default=5)
    parser.add_argument("--restore-cameras", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stop-after-step", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--reset-agents-each-step",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Before every action, restore target/drone/robotdog to the recorded "
            "pre-action poses for that step. This isolates single-step action "
            "reproducibility from accumulated pose/yaw drift."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    drone_path, dog_path, stat_path = unwrap_episode_source(args.recorded_dir, args.episode)
    drone_records = read_json(drone_path)
    dog_records = read_json(dog_path)
    source_stat = read_json(stat_path) if stat_path is not None else None
    if not isinstance(drone_records, list) or not drone_records:
        raise ValueError(f"empty drone records: {drone_path}")
    if not isinstance(dog_records, list) or not dog_records:
        raise ValueError(f"empty robotdog records: {dog_path}")

    env_args = build_env_args(args, drone_records[0], source_stat)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.save_dir / "replay_config.json", jsonable(vars(args) | {"env_args": vars(env_args)}))

    print(f"[replay] env={args.env_id} episode={args.episode} source={args.recorded_dir}", flush=True)
    print(
        f"[replay] drone_action_source={args.drone_action_source} dog_action_source={args.dog_action_source} "
        f"target_replay={args.target_replay} dt={env_args.dt} ue_interval_ms={env_args.ue_interval_ms}",
        flush=True,
    )

    env = make_env(env_args)
    try:
        reset_env(env, env_args)
        target_id, robotdog_id, drone_id = classify_coop_agents(env)
        players = env.unwrapped.player_list
        target_name = players[target_id]
        robotdog_name = players[robotdog_id]
        drone_name = players[drone_id]

        restore_pose(env, target_name, drone_records[0]["target_pose"])
        restore_pose(
            env,
            robotdog_name,
            dog_records[0]["robotdog_pose"],
            yaw_setter=lambda yaw: set_ground_yaw(env, robotdog_name, yaw),
        )
        restore_pose(
            env,
            drone_name,
            drone_records[0]["drone_pose"],
            yaw_setter=lambda yaw: maybe_set_drone_yaw(env, drone_name, yaw),
        )

        if args.restore_cameras:
            first_dog = dog_records[0]
            mount = first_dog.get("robotdog_camera_mount")
            if isinstance(mount, list) and len(mount) >= 3:
                set_robotdog_camera(
                    env,
                    robotdog_name,
                    robotdog_id,
                    [float(v) for v in mount[:3]],
                    float(first_dog.get("robotdog_camera_pitch", -8.0)),
                    float(first_dog.get("robotdog_camera_yaw_offset", 0.0)),
                    dog_args(env_args),
                )
            set_drone_camera(
                env,
                drone_name,
                drone_id,
                list(env.unwrapped.obj_poses[drone_id]),
                float(drone_records[0].get("drone_camera_pitch", -60.0)),
                float(drone_records[0].get("drone_camera_yaw_offset", 0.0)),
                drone_args(env_args),
            )

        try:
            env.unwrapped.unrealcv.set_resume()
        except Exception:
            pass

        max_steps = min(len(drone_records), len(dog_records))
        if args.max_steps > 0:
            max_steps = min(max_steps, args.max_steps)

        replay_infos: list[dict[str, Any]] = []
        for idx in range(max_steps):
            drone_rec = drone_records[idx]
            dog_rec = dog_records[idx]
            if args.reset_agents_each_step:
                restore_pose(env, target_name, drone_rec["target_pose"])
                restore_pose(
                    env,
                    robotdog_name,
                    dog_rec["robotdog_pose"],
                    yaw_setter=lambda yaw: set_ground_yaw(env, robotdog_name, yaw),
                )
                restore_pose(
                    env,
                    drone_name,
                    drone_rec["drone_pose"],
                    yaw_setter=lambda yaw: maybe_set_drone_yaw(env, drone_name, yaw),
                )
            elif args.target_replay in {"pose_action", "pose_only"}:
                restore_pose(env, target_name, drone_rec["target_pose"])

            before_drone_pose = list(env.unwrapped.obj_poses[drone_id])
            before_dog_pose = list(env.unwrapped.obj_poses[robotdog_id])
            before_target_pose = list(env.unwrapped.obj_poses[target_id])

            dt = float(drone_rec.get("base_velocity_dt_s") or drone_rec.get("effective_dt_s") or env_args.dt)
            drone_action = recorded_drone_action(drone_rec, args.drone_action_source, dt)
            dog_action = recorded_dog_action(dog_rec, args.dog_action_source)
            target_action = None
            if args.target_replay in {"action", "pose_action"}:
                raw_target_action = drone_rec.get("target_action")
                if isinstance(raw_target_action, list):
                    target_action = action_for_target_space(env, target_id, [float(v) for v in raw_target_action])

            actions = [None for _ in players]
            actions[target_id] = target_action
            actions[robotdog_id] = dog_action
            actions[drone_id] = drone_action

            step_t0 = time.perf_counter()
            _obs, _rewards, done, info = data_collection_step_pose_only(env, actions)
            step_wall = time.perf_counter() - step_t0
            after_drone_pose = list(env.unwrapped.obj_poses[drone_id])
            after_dog_pose = list(env.unwrapped.obj_poses[robotdog_id])
            after_target_pose = list(env.unwrapped.obj_poses[target_id])

            source_drone_after = drone_rec.get("drone_pose_after_action") or (
                drone_records[idx + 1].get("drone_pose") if idx + 1 < len(drone_records) else drone_rec.get("drone_pose")
            )
            source_dog_after = dog_rec.get("robotdog_pose_after_action") or (
                dog_records[idx + 1].get("robotdog_pose") if idx + 1 < len(dog_records) else dog_rec.get("robotdog_pose")
            )
            source_target_after = drone_rec.get("target_pose_after_action") or (
                drone_records[idx + 1].get("target_pose") if idx + 1 < len(drone_records) else drone_rec.get("target_pose")
            )
            record = {
                "step": idx + 1,
                "reset_agents_each_step": bool(args.reset_agents_each_step),
                "actions": {
                    "target": target_action,
                    "robotdog": dog_action,
                    "drone": drone_action,
                },
                "before": {
                    "target_pose": before_target_pose,
                    "robotdog_pose": before_dog_pose,
                    "drone_pose": before_drone_pose,
                },
                "after": {
                    "target_pose": after_target_pose,
                    "robotdog_pose": after_dog_pose,
                    "drone_pose": after_drone_pose,
                },
                "source_after": {
                    "target_pose": source_target_after,
                    "robotdog_pose": source_dog_after,
                    "drone_pose": source_drone_after,
                },
                "errors": {
                    "target_after_m": pose_error_m(after_target_pose, source_target_after),
                    "robotdog_after_m": pose_error_m(after_dog_pose, source_dog_after),
                    "drone_after_m": pose_error_m(after_drone_pose, source_drone_after),
                    "drone_yaw_after_deg": yaw_error_deg(after_drone_pose, source_drone_after),
                    "drone_distance_to_target_m": distance_xy_m(after_drone_pose, after_target_pose),
                    "source_drone_distance_to_target_m": distance_xy_m(source_drone_after, source_target_after),
                },
                "step_wall_time_s": float(step_wall),
                "done": bool(done),
            }
            replay_infos.append(record)
            if idx < 5 or (idx + 1) % 50 == 0:
                err = record["errors"]
                print(
                    f"[step {idx + 1}] drone_err={err['drone_after_m']:.3f}m "
                    f"yaw_err={err['drone_yaw_after_deg']:.2f}deg "
                    f"target_err={err['target_after_m']:.4f}m "
                    f"d={err['drone_distance_to_target_m']:.2f}m",
                    flush=True,
                )
            if done:
                break

        drone_errors = [item["errors"]["drone_after_m"] for item in replay_infos]
        yaw_errors = [item["errors"]["drone_yaw_after_deg"] for item in replay_infos]
        target_errors = [item["errors"]["target_after_m"] for item in replay_infos]
        summary = {
            "episode": args.episode,
            "steps": len(replay_infos),
            "source": {
                "drone_info": str(drone_path.resolve()),
                "robotdog_info": str(dog_path.resolve()),
                "stat": str(stat_path.resolve()) if stat_path else None,
            },
            "env_id": args.env_id,
            "drone_action_source": args.drone_action_source,
            "dog_action_source": args.dog_action_source,
            "target_replay": args.target_replay,
            "reset_agents_each_step": bool(args.reset_agents_each_step),
            "dt": float(env_args.dt),
            "ue_interval_ms": int(env_args.ue_interval_ms),
            "mean_drone_after_error_m": float(np.mean(drone_errors)) if drone_errors else None,
            "max_drone_after_error_m": float(np.max(drone_errors)) if drone_errors else None,
            "mean_drone_yaw_after_error_deg": float(np.mean(yaw_errors)) if yaw_errors else None,
            "max_drone_yaw_after_error_deg": float(np.max(yaw_errors)) if yaw_errors else None,
            "mean_target_after_error_m": float(np.mean(target_errors)) if target_errors else None,
            "max_target_after_error_m": float(np.max(target_errors)) if target_errors else None,
            "mean_step_wall_time_s": float(np.mean([item["step_wall_time_s"] for item in replay_infos])) if replay_infos else None,
        }
        write_json(args.save_dir / "replay_info.json", replay_infos)
        write_json(args.save_dir / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    finally:
        try:
            env.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
