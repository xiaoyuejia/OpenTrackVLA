"""Keyboard-controlled drone-human tracking data collection.

This script creates one human target and one drone in a street-like UnrealZoo
tracking scene. Both agents are controlled by keyboard input:

    Human: I/K forward/backward, J/L turn left/right
    Drone: W/S forward/backward, A/D left/right, E/Q up/down, Z/C yaw
    Recording: P toggles recording, Esc quits and saves

The saved drone-view video uses a world-locked UnrealCV camera whose x/y match
the drone x/y on every timestep. The camera is rotated to look at the human so
manual control can focus on moving the human and drone.
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

for font_dir in (
    "/usr/share/fonts/truetype/dejavu",
    "/usr/share/fonts/truetype",
    "/home/hdt/miniconda3/envs/follow/fonts",
):
    if "QT_QPA_FONTDIR" not in os.environ and Path(font_dir).exists():
        os.environ["QT_QPA_FONTDIR"] = font_dir
        break

import cv2
import gym
import numpy as np

try:
    from pynput import keyboard
except Exception as exc:  # pragma: no cover - depends on local GUI setup.
    keyboard = None
    KEYBOARD_IMPORT_ERROR = exc
else:
    KEYBOARD_IMPORT_ERROR = None


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import gym_unrealcv  # noqa: F401  # Registers UnrealZoo env ids.
from gym_unrealcv.envs.wrappers import augmentation, configUE, time_dilation

from generate_drone_human_tracking_small import (
    DEFAULT_INSTRUCTION,
    UNREAL_UNITS_PER_METER,
    bbox_center_error,
    bbox_centered,
    classify_agents,
    collision_from_info,
    data_collection_reset,
    data_collection_step,
    distance_m,
    distance_xy_m,
    draw_bbox,
    ensure_bgr_uint8,
    get_top_view_frame,
    heading_deg,
    jsonable,
    maybe_set_drone_yaw,
    pick_open_start,
    pose_xyz,
    resize_for_monitor,
    safe_slug,
    save_mp4,
    target_visibility,
    update_observation,
    wrap_deg,
    write_json,
    yaw_deg,
    yaw_forward_xy,
    yaw_to_quat_y,
)


DEFAULT_ENV_ID = "UnrealTrack-DowntownWest-ContinuousColor-v0"
DEFAULT_OUT_DIR = "/data/hdt/ntv_data/sim_data/unrealzoo_keyboard_drone_human"


KEY_STATE = {
    "i": False,
    "j": False,
    "k": False,
    "l": False,
    "w": False,
    "a": False,
    "s": False,
    "d": False,
    "q": False,
    "e": False,
    "z": False,
    "c": False,
    "p": False,
    "esc": False,
}
PENDING_EVENTS: set[str] = set()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Keyboard-control human and drone tracking data collection.")
    parser.add_argument("--env-id", default=DEFAULT_ENV_ID)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--episode-id", default=None)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    parser.add_argument("--record", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--offscreen", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--time-dilation", type=int, default=-1)
    parser.add_argument("--launch-retries", type=int, default=5)
    parser.add_argument("--distractors", type=int, default=0)

    parser.add_argument("--human-speed", type=float, default=90.0, help="Human forward speed, Unreal units/s.")
    parser.add_argument("--human-turn", type=float, default=30.0, help="Human turn command angle.")
    parser.add_argument("--drone-speed", type=float, default=0.9, help="Drone planar speed command.")
    parser.add_argument("--drone-vertical-speed", type=float, default=0.6)
    parser.add_argument("--drone-yaw-speed", type=float, default=0.7)

    parser.add_argument("--initial-drone-dist", type=float, default=280.0, help="Initial horizontal distance, Unreal units.")
    parser.add_argument("--initial-drone-height", type=float, default=220.0, help="Initial height above human, Unreal units.")
    parser.add_argument("--open-spawn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--open-spawn-radius", type=float, default=900.0)
    parser.add_argument("--min-open-clearance", type=float, default=300.0)
    parser.add_argument("--open-spawn-candidates", type=int, default=96)
    parser.add_argument("--drone-navmesh-tolerance", type=float, default=450.0)

    parser.add_argument("--camera-z-offset", type=float, default=0.0)
    parser.add_argument("--camera-fov", type=float, default=100.0)
    parser.add_argument("--camera-fixed-pitch", type=float, default=-60.0)
    parser.add_argument("--monitor", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--monitor-top-view", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--monitor-scale", type=float, default=0.75)
    parser.add_argument("--min-visible-ratio", type=float, default=0.001)
    parser.add_argument("--target-center-tolerance", type=float, default=0.30)
    return parser.parse_args()


def on_press(key) -> None:
    try:
        char = key.char.lower()
    except AttributeError:
        if key == keyboard.Key.esc:
            KEY_STATE["esc"] = True
            PENDING_EVENTS.add("esc")
        return
    if char in KEY_STATE:
        if not KEY_STATE[char]:
            PENDING_EVENTS.add(char)
        KEY_STATE[char] = True


def on_release(key) -> None:
    try:
        char = key.char.lower()
    except AttributeError:
        return
    if char in KEY_STATE:
        KEY_STATE[char] = False


def consume_event(name: str) -> bool:
    if name in PENDING_EVENTS:
        PENDING_EVENTS.remove(name)
        return True
    return False


def human_action(args: argparse.Namespace) -> list[float]:
    turn = 0.0
    speed = 0.0
    if KEY_STATE["i"]:
        speed += args.human_speed
    if KEY_STATE["k"]:
        speed -= args.human_speed
    if KEY_STATE["j"]:
        turn -= args.human_turn
    if KEY_STATE["l"]:
        turn += args.human_turn
    return [float(turn), float(speed)]


def drone_action(args: argparse.Namespace) -> list[float]:
    vx = 0.0
    vy = 0.0
    vz = 0.0
    vyaw = 0.0
    if KEY_STATE["w"]:
        vx += args.drone_speed
    if KEY_STATE["s"]:
        vx -= args.drone_speed
    if KEY_STATE["d"]:
        vy += args.drone_speed
    if KEY_STATE["a"]:
        vy -= args.drone_speed
    if KEY_STATE["e"]:
        vz += args.drone_vertical_speed
    if KEY_STATE["q"]:
        vz -= args.drone_vertical_speed
    if KEY_STATE["c"]:
        vyaw += args.drone_yaw_speed
    if KEY_STATE["z"]:
        vyaw -= args.drone_yaw_speed
    return [float(vx), float(vy), float(vz), float(vyaw)]


def make_env(args: argparse.Namespace):
    env = gym.make(args.env_id)
    env = configUE.ConfigUEWrapper(env, offscreen=args.offscreen, resolution=(args.width, args.height))
    env.unwrapped.agents_category = ["player", "drone"]
    env.unwrapped.require_visual_target = True
    if args.time_dilation > 0:
        env = time_dilation.TimeDilationWrapper(env, args.time_dilation)
    env = augmentation.RandomPopulationWrapper(env, 2, 2, random_target=False)
    env.seed(args.seed)
    return env


def set_human_pose(env, target_name: str, location: list[float], yaw: float) -> None:
    env.unwrapped.unrealcv.set_obj_location(target_name, location)
    try:
        env.unwrapped.unrealcv.set_obj_rotation(target_name, [0.0, float(yaw), 0.0])
    except Exception:
        pass


def set_manual_drone_camera(
    env,
    drone_id: int,
    target_id: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    cam_id = env.unwrapped.cam_list[drone_id]
    drone_pose = list(env.unwrapped.obj_poses[drone_id])
    cam_loc = [
        float(drone_pose[0]),
        float(drone_pose[1]),
        float(drone_pose[2]) + float(args.camera_z_offset),
    ]
    yaw = yaw_deg(drone_pose)
    pitch = float(args.camera_fixed_pitch)
    try:
        env.unwrapped.unrealcv.set_cam_location(cam_id, cam_loc)
        env.unwrapped.unrealcv.set_cam_rotation(cam_id, [float(pitch), float(yaw), 0.0])
        if args.camera_fov > 0:
            env.unwrapped.unrealcv.set_cam_fov(cam_id, float(args.camera_fov))
    except Exception:
        pass
    return {"camera_location": cam_loc, "camera_pitch": float(pitch), "camera_yaw": float(yaw)}


def place_initial_scene(env, args: argparse.Namespace, rng: random.Random) -> tuple[np.ndarray, dict[str, Any]]:
    obs = data_collection_reset(env, args)
    target_id, drone_id, _distractor_ids = classify_agents(env)
    env.unwrapped.target_id = target_id
    env.unwrapped.tracker_id = drone_id
    env.unwrapped.protagonist_id = drone_id
    players = env.unwrapped.player_list
    target_name = players[target_id]
    drone_name = players[drone_id]

    try:
        env.unwrapped.unrealcv.set_appearance(target_name, int(rng.randint(1, 18)))
    except Exception:
        pass

    if args.open_spawn:
        human_loc = pick_open_start(env, rng, args)
    else:
        human_loc = list(env.unwrapped.obj_poses[target_id][:3])
    human_yaw = rng.uniform(-180.0, 180.0)
    set_human_pose(env, target_name, human_loc, human_yaw)
    update_observation(env, refresh_cameras=True)

    # Try several around-target starts and keep the first one that sees the human.
    offsets = [180.0, 150.0, 210.0, 120.0, 240.0, 90.0, 270.0, 0.0]
    best_obs = obs
    best_visibility = -1.0
    best_pose: list[float] | None = None
    for offset in offsets:
        angle = math.radians(human_yaw + offset)
        drone_loc = [
            float(human_loc[0] + args.initial_drone_dist * math.cos(angle)),
            float(human_loc[1] + args.initial_drone_dist * math.sin(angle)),
            float(human_loc[2] + args.initial_drone_height),
        ]
        env.unwrapped.unrealcv.set_move_bp(drone_name, [0.0, 0.0, 0.0, 0.0])
        env.unwrapped.unrealcv.set_obj_location(drone_name, drone_loc)
        maybe_set_drone_yaw(env, drone_name, heading_deg(np.asarray(drone_loc), pose_xyz(list(env.unwrapped.obj_poses[target_id]))))
        obs = update_observation(env, refresh_cameras=True)
        set_manual_drone_camera(env, drone_id, target_id, args)
        obs = update_observation(env, refresh_cameras=False)
        drone_pose = list(env.unwrapped.obj_poses[drone_id])
        target_pose = list(env.unwrapped.obj_poses[target_id])
        visibility, visible, _bbox = target_visibility(env, drone_id, target_name, drone_pose, target_pose, use_mask=True)
        if visible and visibility > best_visibility:
            best_visibility = visibility
            best_pose = drone_pose
            best_obs = obs
        if visible and visibility >= args.min_visible_ratio:
            break

    if best_pose is None:
        best_pose = list(env.unwrapped.obj_poses[drone_id])
    obs = best_obs
    try:
        env.unwrapped.unrealcv.set_max_speed(target_name, float(args.human_speed))
    except Exception:
        pass

    return obs, {
        "target_id": target_id,
        "drone_id": drone_id,
        "target_name": target_name,
        "drone_name": drone_name,
        "start_target_pose": list(env.unwrapped.obj_poses[target_id]),
        "start_drone_pose": best_pose,
    }


def show_controls() -> None:
    print("\n=== Keyboard drone-human tracking ===")
    print("Human: I/K forward-backward, J/L turn left-right")
    print("Drone: W/S forward-backward, A/D left-right, E/Q up-down, Z/C yaw")
    print("P toggles recording, Esc quits and saves")
    print("The saved camera follows drone x/y and looks at the human.\n")


def show_monitor(
    env,
    args: argparse.Namespace,
    frame: np.ndarray,
    bbox: list[int],
    drone_pose: list[float],
    target_pose: list[float],
    dist: float,
    visible: bool,
    recording: bool,
) -> None:
    if not args.monitor:
        return
    view = draw_bbox(frame, bbox)
    cv2.putText(
        view,
        f"dist={dist:.2f}m visible={int(visible)} rec={int(recording)}",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    cv2.imshow("keyboard_drone_view", resize_for_monitor(view, args.monitor_scale))
    if args.monitor_top_view:
        top = get_top_view_frame(env, drone_pose, target_pose)
        if top is not None:
            cv2.putText(top, "global top view", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("keyboard_global_top_view", resize_for_monitor(top, args.monitor_scale))
    cv2.waitKey(1)


def build_train_episode(
    args: argparse.Namespace,
    episode_id: str,
    setup: dict[str, Any],
    record_infos: list[dict[str, Any]],
) -> dict[str, Any]:
    target_start = record_infos[0]["target_pose"] if record_infos else setup["start_target_pose"]
    drone_start = record_infos[0]["drone_pose"] if record_infos else setup["start_drone_pose"]
    target_final = record_infos[-1]["target_pose"] if record_infos else target_start
    return {
        "episode_id": str(episode_id),
        "scene_id": args.env_id,
        "scene_dataset_config": "unrealzoo_keyboard",
        "additional_obj_config_paths": [],
        "start_position": [float(v) for v in target_start[:3]],
        "start_rotation": yaw_to_quat_y(yaw_deg(target_start)),
        "info": {
            "dist_ratio": 1.0,
            "robot_position": [float(v) for v in drone_start[:3]],
            "robot_rotation": math.radians(yaw_deg(drone_start)),
            "human_num": 1,
            "num_goals_for_main_human": 1,
            "num_goals_for_other_human": 0,
            "main_humanoid_name": setup["target_name"],
            "main_human_semantic_id": 0,
            "extra_humanoid_names": [],
            "instruction": DEFAULT_INSTRUCTION,
            "episode_mode": "keyboard_stt",
            "seed": int(args.seed),
            "drone_name": setup["drone_name"],
            "base_velocity_convention": "drone body frame command [vx, vy, yaw_rate]",
            "human_1_start_position": [float(v) for v in target_start[:3]],
            "human_1_start_rotation": math.radians(yaw_deg(target_start)),
            "human_1_waypoint_1_position": [float(v) for v in target_final[:3]],
        },
        "goals": [{"position": [[float(v) for v in target_final[:3]]], "radius": None}],
        "start_room": None,
        "shortest_paths": None,
    }


def save_outputs(
    args: argparse.Namespace,
    episode_id: str,
    scene_dir: Path,
    frames: list[np.ndarray],
    record_infos: list[dict[str, Any]],
    setup: dict[str, Any],
) -> None:
    if not frames:
        print("[save] no recorded frames; nothing to write", flush=True)
        return
    video_path = scene_dir / f"{episode_id}.mp4"
    info_path = scene_dir / f"{episode_id}_info.json"
    stat_path = scene_dir / f"{episode_id}.json"
    train_path = args.out_dir / "train.json"
    save_mp4(frames, video_path, args.fps)
    write_json(info_path, record_infos)

    following_step = sum(1 for item in record_infos if item["target_visible"])
    total_step = len(record_infos)
    stat = {
        "finish": True,
        "status": "Keyboard",
        "success": 1.0,
        "following_rate": following_step / max(total_step, 1),
        "following_step": following_step,
        "total_step": total_step,
        "collision": float(any(item.get("collision", False) for item in record_infos)),
        "instruction": DEFAULT_INSTRUCTION,
    }
    write_json(stat_path, stat)

    episode_config = build_train_episode(args, episode_id, setup, record_infos)
    write_json(train_path, {"episodes": [episode_config]})
    print(f"[save] video={video_path}", flush=True)
    print(f"[save] info={info_path}", flush=True)
    print(f"[save] stat={stat_path}", flush=True)
    print(f"[save] train={train_path}", flush=True)


def main() -> int:
    args = parse_args()
    if keyboard is None:
        raise RuntimeError(f"pynput keyboard is unavailable: {KEYBOARD_IMPORT_ERROR}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    episode_id = args.episode_id or time.strftime("%Y%m%d_%H%M%S")
    scene_key = safe_slug(args.env_id)
    scene_dir = args.out_dir / f"seed_{args.seed}" / scene_key
    scene_dir.mkdir(parents=True, exist_ok=True)

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    show_controls()

    env = make_env(args)
    rng = random.Random(args.seed)
    frames: list[np.ndarray] = []
    record_infos: list[dict[str, Any]] = []
    recording = bool(args.record)
    last_info: dict[str, Any] | None = None
    t0 = time.time()
    setup: dict[str, Any] | None = None

    try:
        obs, setup = place_initial_scene(env, args, rng)
        target_id = setup["target_id"]
        drone_id = setup["drone_id"]
        target_name = setup["target_name"]
        players = env.unwrapped.player_list

        for step_idx in range(args.max_steps):
            if consume_event("esc") or KEY_STATE["esc"]:
                break
            if consume_event("p"):
                recording = not recording
                print(f"[record] {'on' if recording else 'off'} at step {step_idx}", flush=True)

            prev_drone_pose = list(env.unwrapped.obj_poses[drone_id])
            h_action = human_action(args)
            d_action = drone_action(args)
            actions = [None for _ in players]
            actions[target_id] = h_action
            actions[drone_id] = d_action
            obs, _rewards, done, last_info = data_collection_step(env, actions)

            camera_info = set_manual_drone_camera(env, drone_id, target_id, args)
            obs = update_observation(env, refresh_cameras=False)
            drone_pose = list(env.unwrapped.obj_poses[drone_id])
            target_pose = list(env.unwrapped.obj_poses[target_id])
            visibility, visible_hint, bbox = target_visibility(
                env,
                drone_id,
                target_name,
                drone_pose,
                target_pose,
                use_mask=True,
            )
            dist = distance_xy_m(drone_pose, target_pose)
            dist_3d = distance_m(drone_pose, target_pose)
            target_visible = bool(visible_hint and visibility >= args.min_visible_ratio)
            frame = ensure_bgr_uint8(obs[drone_id])
            measured_delta = float(
                np.linalg.norm(np.asarray(drone_pose[:2]) - np.asarray(prev_drone_pose[:2])) / UNREAL_UNITS_PER_METER
            )
            collision = collision_from_info(last_info, drone_id, target_id, dist, drone_pose, target_pose)
            item = {
                "step": step_idx + 1,
                "dis_to_human": float(dist),
                "facing": 1.0 if target_visible else 0.0,
                "base_velocity": [float(d_action[0]), float(d_action[1]), float(d_action[3])],
                "drone_action": [float(v) for v in d_action],
                "human_action": [float(v) for v in h_action],
                "target_visible": bool(target_visible),
                "target_visibility": float(visibility),
                "target_bbox": bbox,
                "target_center_error": bbox_center_error(bbox, args),
                "target_centered": bool(target_visible and bbox_centered(bbox, args)),
                "dis_to_human_3d": float(dist_3d),
                "drone_motion_delta_m": measured_delta,
                "camera_location": camera_info["camera_location"],
                "camera_pitch": camera_info["camera_pitch"],
                "camera_yaw": camera_info["camera_yaw"],
                "collision": bool(collision),
                "drone_pose": drone_pose,
                "target_pose": target_pose,
            }
            if recording:
                frames.append(frame.copy())
                record_infos.append(item)

            show_monitor(env, args, frame, bbox, drone_pose, target_pose, dist, target_visible, recording)
            if done:
                print("[env] done=True; exiting", flush=True)
                break

        fps = (len(record_infos) if recording else args.max_steps) / max(time.time() - t0, 1e-6)
        print(f"[done] recorded_steps={len(record_infos)} approx_fps={fps:.2f}", flush=True)
    finally:
        try:
            env.close()
        except Exception:
            pass
        try:
            listener.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()

    if args.record and setup is not None:
        save_outputs(args, str(episode_id), scene_dir, frames, record_infos, setup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
