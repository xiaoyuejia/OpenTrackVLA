#!/usr/bin/env python3
"""Recollect exact Unreal object-mask boxes for one saved replay episode.

The saved pre-action poses are restored for every frame.  No inverse dynamics
is executed and videos are not rewritten.  JSON replacement is atomic and the
original files are copied to a caller-provided backup directory first.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2

BASE_REPO = Path("/data/hdt/newtrackvla修改/newtrackvla_base")
for path in (
    BASE_REPO,
    BASE_REPO / "unrealzoo-gym",
    BASE_REPO / "unrealzoo-gym/example/DataRecording",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _argv_value(name: str, default: str) -> str:
    prefix = name + "="
    for index, value in enumerate(sys.argv):
        if value == name and index + 1 < len(sys.argv):
            return sys.argv[index + 1]
        if value.startswith(prefix):
            return value[len(prefix) :]
    return default


os.environ.setdefault("UNREALZOO_FIXED_TIMESTEP", _argv_value("--dt", "0.1"))

from tools.history.replay_unrealzoo_recording import (  # noqa: E402
    build_env_args,
    classify_coop_agents,
    dog_args,
    drone_args,
    make_env,
    maybe_set_drone_yaw,
    reset_env,
    restore_pose,
    set_drone_camera,
    set_ground_yaw,
    set_robotdog_camera,
)
from generate_robotdog_human_tracking_small import object_mask_ratio_and_bbox  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-dir", type=Path, required=True)
    parser.add_argument("--env-id", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--backup-root", type=Path, required=True)
    parser.add_argument("--render-gpu", type=int, required=True)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--launch-retries", type=int, default=5)
    parser.add_argument("--post-reset-settle-s", type=float, default=3.0)
    parser.add_argument("--snapshot-retries", type=int, default=8)
    parser.add_argument(
        "--render-settle-s",
        type=float,
        default=0.12,
        help="Render-thread barrier after pose/camera mutation and before mask capture.",
    )
    parser.add_argument(
        "--first-frame-settle-s",
        type=float,
        default=3.0,
        help="Additional first-frame wait for ProxyAnnotator after appearance changes.",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".bbox_tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def center_fields(bbox: list[int], width: int, height: int) -> tuple[float, bool]:
    if len(bbox) < 4 or bbox[2] <= 0 or bbox[3] <= 0:
        return 1.0, False
    cx = bbox[0] + 0.5 * bbox[2]
    cy = bbox[1] + 0.5 * bbox[3]
    error = ((cx - width / 2) ** 2 + (cy - height / 2) ** 2) ** 0.5
    error /= max((width**2 + height**2) ** 0.5, 1.0)
    return float(error), bool(error <= 0.15)


def read_mask(unrealcv, camera_id: int, attempts: int):
    error = None
    for retry in range(max(1, attempts)):
        try:
            image = unrealcv.read_image(camera_id, "object_mask", "direct")
            if image is not None and getattr(image, "size", 0) > 0:
                return image
        except Exception as exc:  # UnrealCV can transiently drop a response.
            error = exc
        time.sleep(0.15 * (retry + 1))
    raise RuntimeError(f"object-mask capture failed for camera {camera_id}: {error}")


def episode_number(directory: Path) -> str:
    if not directory.name.startswith("episode_"):
        raise ValueError(f"invalid episode directory: {directory}")
    return directory.name.removeprefix("episode_")


def main() -> int:
    args = parse_args()
    directory = args.episode_dir.resolve()
    root = args.data_root.resolve()
    number = episode_number(directory)
    drone_path = directory / f"{number}_drone_info.json"
    dog_path = directory / f"{number}_robotdog_info.json"
    marker = directory / "bbox_recollection.json"
    if args.resume and marker.is_file():
        state = load_json(marker)
        if state.get("status") == "complete":
            print(f"[skip] {directory}", flush=True)
            return 0
    drone_rows, dog_rows = load_json(drone_path), load_json(dog_path)
    count = min(len(drone_rows), len(dog_rows))
    if count < 1:
        raise ValueError(f"empty info records: {directory}")
    env_id = args.env_id
    stat_path = directory / f"{number}.json"
    stat = load_json(stat_path) if stat_path.is_file() else None
    base = SimpleNamespace(
        env_id=env_id,
        width=args.width,
        height=args.height,
        dt=args.dt,
        ue_interval_ms=int(round(args.dt * 1000)),
        offscreen=True,
        render_gpu=args.render_gpu,
        seed=args.seed,
        launch_retries=args.launch_retries,
    )
    env_args = build_env_args(base, drone_rows[0], stat if isinstance(stat, dict) else None)
    env_args.top_view_height = None
    env = make_env(env_args)
    try:
        reset_env(env, env_args)
        if args.post_reset_settle_s:
            time.sleep(args.post_reset_settle_s)
        target_id, dog_id, drone_id = classify_coop_agents(env)
        players = env.unwrapped.player_list
        target_name, dog_name, drone_name = players[target_id], players[dog_id], players[drone_id]
        env.unwrapped.unrealcv.set_appearance(target_name, 5)
        env.unwrapped.unrealcv.set_appearance(dog_name, 27)
        unrealcv = env.unwrapped.unrealcv
        unrealcv.set_pause()

        for index in range(count):
            drone = drone_rows[index]
            dog = dog_rows[index]
            target_pose = drone.get("target_pose") or dog.get("target_pose")
            restore_pose(env, target_name, target_pose)
            restore_pose(env, dog_name, dog["robotdog_pose"], lambda yaw: set_ground_yaw(env, dog_name, yaw))
            restore_pose(env, drone_name, drone["drone_pose"], lambda yaw: maybe_set_drone_yaw(env, drone_name, yaw))
            mount = [float(v) for v in dog.get("robotdog_camera_mount", [170, 0, 120])[:3]]
            dog_pitch = float(dog.get("robotdog_camera_pitch", -8.0))
            dog_yaw = float(dog.get("robotdog_camera_yaw_offset", 0.0))
            drone_pitch = float(drone.get("drone_camera_pitch", -40.0))
            drone_yaw = float(drone.get("drone_camera_yaw_offset", 0.0))
            set_robotdog_camera(env, dog_name, dog_id, mount, dog_pitch, dog_yaw, dog_args(env_args))
            set_drone_camera(env, drone_name, drone_id, drone["drone_pose"], drone_pitch, drone_yaw, drone_args(env_args))
            # UE5.6 acknowledges pose/camera commands before the render thread
            # has retired the previous annotation capture.  Back-to-back pose
            # mutation and mask capture races ProxyAnnotator and segfaults in
            # MovieQualityRenderComponent.  A short wall-time barrier lets the
            # paused renderer finish without advancing simulation state.
            settle = args.first_frame_settle_s if index == 0 else args.render_settle_s
            if settle > 0:
                time.sleep(settle)
            drone_mask = read_mask(unrealcv, env.unwrapped.cam_list[drone_id], args.snapshot_retries)
            dog_mask = read_mask(unrealcv, env.unwrapped.cam_list[dog_id], args.snapshot_retries)
            for row, mask in ((drone, drone_mask), (dog, dog_mask)):
                visibility, visible, bbox = object_mask_ratio_and_bbox(env, env.unwrapped.cam_list[drone_id if row is drone else dog_id], target_name, mask)
                bbox = [int(value) for value in bbox]
                center_error, centered = center_fields(bbox, args.width, args.height)
                row["target_bbox"] = bbox
                row["target_visible"] = bool(visible)
                row["target_visibility"] = float(visibility)
                row["target_center_error"] = center_error
                row["target_centered"] = bool(visible and centered)
                row["target_bbox_source"] = "replay_object_mask_pose_restore"
            if (index + 1) % 50 == 0 or index + 1 == count:
                print(f"[frame {index + 1}/{count}] {directory}", flush=True)

        relative = directory.relative_to(root)
        backup_dir = args.backup_root.resolve() / relative
        backup_dir.mkdir(parents=True, exist_ok=True)
        for source in (drone_path, dog_path):
            destination = backup_dir / source.name
            if not destination.exists():
                shutil.copy2(source, destination)
        atomic_json(drone_path, drone_rows)
        atomic_json(dog_path, dog_rows)
        atomic_json(marker, {
            "status": "complete",
            "frames": count,
            "bbox_source": "replay_object_mask_pose_restore",
            "backup_dir": str(backup_dir),
        })
        print(f"[complete] {directory} frames={count}", flush=True)
        return 0
    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
