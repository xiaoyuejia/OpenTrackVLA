"""Collect aerial-ground cooperative human tracking data.

One target human walks with NavMesh, while a ground robot dog and an aerial
drone actively track the same human. The script saves three synchronized videos:

    <episode_id>_drone.mp4      drone first-person RGB
    <episode_id>_robotdog.mp4   robot dog first-person RGB
    <episode_id>_global.mp4     global top-view RGB

and per-agent step logs:

    <episode_id>_drone_info.json
    <episode_id>_robotdog_info.json
    <episode_id>.json
    train.json

UnrealZoo represents robot dogs as BP_Character/player appearances with
set_app IDs 20..33, so the population is ["player", "player", "drone"]:
target human, robot dog, drone.
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

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

try:
    from pynput import keyboard
except Exception as exc:  # pragma: no cover - depends on local GUI/input setup.
    keyboard = None
    KEYBOARD_IMPORT_ERROR = exc
else:
    KEYBOARD_IMPORT_ERROR = None

KEY_STATE = {
    "w": False,
    "a": False,
    "s": False,
    "d": False,
    "esc": False,
}


def on_press(key) -> None:
    try:
        char = key.char.lower()
    except AttributeError:
        if keyboard is not None and key == keyboard.Key.esc:
            KEY_STATE["esc"] = True
        return
    if char in KEY_STATE:
        KEY_STATE[char] = True


def on_release(key) -> None:
    try:
        char = key.char.lower()
    except AttributeError:
        if keyboard is not None and key == keyboard.Key.esc:
            KEY_STATE["esc"] = False
        return
    if char in KEY_STATE:
        KEY_STATE[char] = False


def start_keyboard_listener(args: argparse.Namespace):
    if not args.keyboard_human:
        return None
    if keyboard is None:
        raise RuntimeError(f"--keyboard-human requires pynput keyboard support: {KEYBOARD_IMPORT_ERROR}")
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    print("[keyboard] target control enabled: W/S forward/back, A/D turn, Esc ends current episode", flush=True)
    return listener


def keyboard_human_action(args: argparse.Namespace) -> list[float]:
    turn = 0.0
    speed = 0.0
    if KEY_STATE["w"]:
        speed += float(args.human_speed)
    if KEY_STATE["s"]:
        speed -= float(args.human_speed) * float(args.human_reverse_scale)
    if KEY_STATE["a"]:
        turn -= float(args.human_turn)
    if KEY_STATE["d"]:
        turn += float(args.human_turn)
    return [turn, speed]


def set_unrealcv_window_input(env, enable_input: bool) -> None:
    unwrapped = env.unwrapped
    config_path = Path(getattr(unwrapped.ue_binary, "path2unrealcv", ""))
    if not config_path.exists():
        print(f"[ue-input] unrealcv.ini not found: {config_path}", flush=True)
        return

    desired = f"EnableInput={'True' if enable_input else 'False'}"
    lines = config_path.read_text(encoding="utf-8").splitlines()
    updated_lines: list[str] = []
    changed = False
    found = False
    for line in lines:
        if line.strip().lower().startswith("enableinput="):
            updated_lines.append(desired)
            found = True
            changed = line.strip() != desired
        else:
            updated_lines.append(line)
    if not found:
        updated_lines.append(desired)
        changed = True
    if changed:
        config_path.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")
    print(f"[ue-input] {desired} in {config_path}", flush=True)


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

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import gym_unrealcv  # noqa: F401  # Registers UnrealZoo env ids.
from gym_unrealcv.envs.wrappers import augmentation, configUE, time_dilation

from generate_drone_human_tracking_small import (
    DEFAULT_INSTRUCTION,
    EpisodeSkipped,
    OracleDroneHumanTracker,
    UNREAL_UNITS_PER_METER,
    bbox_center_error,
    bbox_centered,
    choose_drone_camera_for_current_pose,
    collision_from_info,
    data_collection_reset,
    data_collection_step,
    distance_m,
    distance_xy_m,
    draw_bbox,
    ensure_bgr_uint8,
    heading_deg,
    height_gap_m,
    maybe_resample_target_goal,
    maybe_set_drone_yaw,
    pick_open_start,
    pick_reachable_goal,
    pose_xyz,
    resize_for_monitor,
    safe_slug,
    save_mp4,
    update_observation,
    velocity_from_pose_delta as drone_velocity_from_pose_delta,
    write_json,
    yaw_deg,
    yaw_to_quat_y,
    yaw_forward_xy,
)
from generate_robotdog_human_tracking_small import (
    OracleRobotDogHumanTracker,
    choose_robotdog_camera_for_current_pose,
    set_ground_yaw,
    set_episode_appearances,
    velocity_from_pose_delta as robotdog_velocity_from_pose_delta,
)


DEFAULT_ENV_ID = "UnrealTrack-DowntownWest-ContinuousColor-v0"
DEFAULT_OUT_DIR = "/data/hdt/ntv_data/sim_data/unrealzoo_aerial_ground_human"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect cooperative robotdog+drone human tracking data.")
    parser.add_argument("--env-id", default=DEFAULT_ENV_ID)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max-attempts", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--render-gpu", type=int, default=None)
    parser.add_argument("--brightness-scale", type=float, default=1.0, help="Global image brightness multiplier applied to saved frames.")
    parser.add_argument("--brightness-offset", type=float, default=0.0, help="Global image brightness offset applied to saved frames.")
    parser.add_argument("--brightness-config", type=Path, default=None, help="Optional per-map brightness JSON file.")
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument(
        "--ue-interval-ms",
        type=int,
        default=None,
        help="UE action interval in milliseconds. Defaults to the scene JSON interval; pass this to override.",
    )
    parser.add_argument("--offscreen", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--time-dilation", type=int, default=-1)
    parser.add_argument("--launch-retries", type=int, default=5)
    parser.add_argument("--monitor", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--monitor-interval", type=int, default=2)
    parser.add_argument("--monitor-scale", type=float, default=0.7)
    parser.add_argument("--write-global-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--top-view-height", type=float, default=None)
    parser.add_argument("--debug-motion", action="store_true")

    #parser.add_argument("--human-speed", type=float, default=90.0)
    parser.add_argument("--human-goal-min-distance", type=float, default=700.0)
    parser.add_argument("--human-goal-max-distance", type=float, default=2200.0)
    parser.add_argument("--human-path-file", type=Path, default=None, help="Closed-loop waypoint file for the target human.")
    parser.add_argument("--human-path-loop", action=argparse.BooleanOptionalAction, default=True, help="Loop back to the first waypoint after the last waypoint.")
    parser.add_argument("--human-waypoint-reach-distance", type=float, default=150.0, help="Unreal-unit distance for considering a waypoint reached.")
    parser.add_argument("--human-waypoint-stall-window", type=int, default=20, help="Recent step window for target stuck detection while following waypoints.")
    parser.add_argument("--human-waypoint-stall-distance", type=float, default=20.0, help="Minimum target movement in Unreal units over the stall window.")
    parser.add_argument("--fixed-spawn-file", type=Path, default=None, help="Accepted for train.py compatibility.")

    parser.add_argument("--robotdog-ideal-follow-dist", type=float, default=6.0)
    parser.add_argument("--robotdog-min-follow-dist", type=float, default=4.5)
    parser.add_argument("--robotdog-max-follow-dist", type=float, default=8.0)
    parser.add_argument("--robotdog-max-speed", type=float, default=1.2)
    parser.add_argument("--robotdog-max-lateral-speed", type=float, default=0.45)
    parser.add_argument("--robotdog-max-yaw-rate", type=float, default=1.0)

    parser.add_argument("--drone-ideal-follow-dist", type=float, default=4.0)
    parser.add_argument("--drone-min-follow-dist", type=float, default=3.0)
    parser.add_argument("--drone-max-follow-dist", type=float, default=5.5)
    parser.add_argument("--drone-height", type=float, default=400.0)
    parser.add_argument("--drone-max-speed", type=float, default=0.12)

    parser.add_argument("--open-spawn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--open-spawn-radius", type=float, default=900.0)
    parser.add_argument("--min-open-clearance", type=float, default=300.0)
    parser.add_argument("--open-spawn-candidates", type=int, default=128)
    parser.add_argument("--ground-navmesh-tolerance", type=float, default=300.0)
    parser.add_argument("--drone-navmesh-tolerance", type=float, default=800.0)

    parser.add_argument("--human-appearance-min", type=int, default=1)
    parser.add_argument("--human-appearance-max", type=int, default=18)
    parser.add_argument("--robotdog-appearance-min", type=int, default=20)
    parser.add_argument("--robotdog-appearance-max", type=int, default=33)

    parser.add_argument("--min-visible-ratio", type=float, default=0.001)
    parser.add_argument("--target-center-tolerance", type=float, default=0.35)
    parser.add_argument("--require-centered-target", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-mask-visibility", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-visual-target", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--robotdog-camera-forward", type=float, default=140.0)
    parser.add_argument("--robotdog-camera-lateral", type=float, default=0.0)
    parser.add_argument("--robotdog-camera-height", type=float, default=110.0)
    parser.add_argument(
        "--robotdog-camera-mounts",
        default="140:0:110,170:0:120,110:0:95,0:120:110,0:90:100,40:90:110,40:-90:110",
    )
    parser.add_argument("--robotdog-camera-fixed-pitch", type=float, default=None)
    parser.add_argument("--robotdog-camera-pitches", default="-15,-8,0,8,15,22,-22")
    parser.add_argument("--robotdog-camera-yaw-offsets", default="0,-8,8,-15,15")
    parser.add_argument("--robotdog-camera-mode", choices=["fixed", "oracle"], default="fixed")
    parser.add_argument("--robotdog-fov", type=float, default=95.0)
    parser.add_argument("--max-self-visible-ratio", type=float, default=0.015)

    parser.add_argument("--drone-camera-fixed-pitch", type=float, default=-60.0)
    parser.add_argument("--drone-camera-pitches", default="-60")
    parser.add_argument("--drone-camera-fixed-yaw", type=float, default=0.0)
    parser.add_argument("--drone-camera-yaw-offsets", default="0")
    parser.add_argument("--drone-camera-mode", choices=["fixed", "oracle"], default="fixed")
    parser.add_argument("--lock-drone-camera-world-xy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--drone-camera-z-offset", type=float, default=0.0)
    parser.add_argument("--drone-fov", type=float, default=100.0)
    parser.add_argument("--max-camera-search-candidates", type=int, default=12)
    parser.add_argument(
        "--snap-heading",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use simulator-side oracle heading. Disabled by default so drone and "
            "robotdog yaw are executed as actions and become learnable labels."
        ),
    )
    parser.add_argument("--follow-behind", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--human-speed", type=float, default=90.0, help="Target forward speed, Unreal units/s.")
    parser.add_argument("--human-turn", type=float, default=5.0, help="Target turning command, degrees.")
    parser.add_argument("--human-reverse-scale", type=float, default=0.5, help="Backward speed multiplier for keyboard control.")
    parser.add_argument(
        "--disable-ue-input",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write EnableInput=False to unrealcv.ini before launch so the UE window ignores keyboard/mouse input.",
    )
    parser.add_argument(
        "--keyboard-human",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use WASD keyboard control for the target human instead of NavMesh walking. Use --no-keyboard-human for NavMesh.",
    )
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def resolve_ue_interval_ms(args: argparse.Namespace, default_interval: int | float) -> int:
    interval = getattr(args, "ue_interval_ms", None)
    if interval is None:
        interval = int(default_interval)
    interval = int(interval)
    if interval <= 0:
        raise ValueError(f"UE interval must be positive, got {interval}")
    args.ue_interval_ms = interval
    return interval


def make_env(args: argparse.Namespace):
    env = gym.make(args.env_id)
    env = configUE.ConfigUEWrapper(
        env,
        offscreen=args.offscreen,
        resolution=(args.width, args.height),
        gpu_id=getattr(args, "render_gpu", None),
    )
    set_unrealcv_window_input(env, enable_input=not args.disable_ue_input)
    env.unwrapped.agents_category = ["player", "player", "drone"]
    env.unwrapped.require_visual_target = bool(args.require_visual_target)
    env.unwrapped.interval = resolve_ue_interval_ms(args, getattr(env.unwrapped, "interval", int(round(float(args.dt) * 1000.0))))
    if args.time_dilation > 0:
        env = time_dilation.TimeDilationWrapper(env, args.time_dilation)
    env = augmentation.RandomPopulationWrapper(env, 3, 3, random_target=False)
    env.seed(args.seed)
    return env


def action_for_target_space(env, target_id: int, move_action: list[float]):
    action_space = env.unwrapped.action_space[target_id]
    if hasattr(action_space, "spaces"):
        return (move_action, 0, 0)
    return move_action


def classify_coop_agents(env) -> tuple[int, int, int]:
    players = env.unwrapped.player_list
    player_ids = [
        idx for idx, obj in enumerate(players) if env.unwrapped.agents[obj].get("agent_type") == "player"
    ]
    drone_ids = [
        idx for idx, obj in enumerate(players) if env.unwrapped.agents[obj].get("agent_type") == "drone"
    ]
    if len(player_ids) < 2 or not drone_ids:
        raise RuntimeError(
            f"Need target player, robotdog player, and drone. Got "
            f"{[(obj, env.unwrapped.agents[obj].get('agent_type')) for obj in players]}"
        )
    return player_ids[0], player_ids[1], drone_ids[0]


def dog_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        ideal_follow_dist=args.robotdog_ideal_follow_dist,
        min_follow_dist=args.robotdog_min_follow_dist,
        max_follow_dist=args.robotdog_max_follow_dist,
        robotdog_max_speed=args.robotdog_max_speed,
        robotdog_max_lateral_speed=args.robotdog_max_lateral_speed,
        robotdog_max_yaw_rate=args.robotdog_max_yaw_rate,
        dt=args.dt,
        open_spawn=args.open_spawn,
        ground_navmesh_tolerance=args.ground_navmesh_tolerance,
        min_open_clearance=args.min_open_clearance,
        robotdog_camera_forward=args.robotdog_camera_forward,
        robotdog_camera_lateral=args.robotdog_camera_lateral,
        robotdog_camera_height=args.robotdog_camera_height,
        robotdog_camera_mounts=args.robotdog_camera_mounts,
        robotdog_camera_fixed_pitch=args.robotdog_camera_fixed_pitch,
        robotdog_camera_pitches=args.robotdog_camera_pitches,
        robotdog_camera_yaw_offsets=args.robotdog_camera_yaw_offsets,
        robotdog_camera_mode=args.robotdog_camera_mode,
        robotdog_fov=args.robotdog_fov,
        max_camera_search_candidates=args.max_camera_search_candidates,
        max_self_visible_ratio=args.max_self_visible_ratio,
        use_mask_visibility=args.use_mask_visibility,
        require_centered_target=args.require_centered_target,
        min_visible_ratio=args.min_visible_ratio,
        target_center_tolerance=args.target_center_tolerance,
        width=args.width,
        height=args.height,
        reset_area=None,
    )


def drone_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        ideal_follow_dist=args.drone_ideal_follow_dist,
        min_follow_dist=args.drone_min_follow_dist,
        max_follow_dist=args.drone_max_follow_dist,
        drone_height=args.drone_height,
        drone_max_speed=args.drone_max_speed,
        dt=args.dt,
        follow_behind=args.follow_behind,
        open_spawn=args.open_spawn,
        drone_navmesh_tolerance=args.drone_navmesh_tolerance,
        min_open_clearance=args.min_open_clearance,
        use_mask_visibility=args.use_mask_visibility,
        require_visual_target=args.require_visual_target,
        require_centered_target=args.require_centered_target,
        min_visible_ratio=args.min_visible_ratio,
        target_center_tolerance=args.target_center_tolerance,
        drone_camera_fixed_pitch=args.drone_camera_fixed_pitch,
        drone_camera_pitches=args.drone_camera_pitches,
        drone_camera_fixed_yaw=args.drone_camera_fixed_yaw,
        drone_camera_yaw_offsets=args.drone_camera_yaw_offsets,
        drone_camera_mode=args.drone_camera_mode,
        lock_drone_camera_world_xy=args.lock_drone_camera_world_xy,
        drone_camera_z_offset=args.drone_camera_z_offset,
        drone_fov=args.drone_fov,
        max_camera_search_candidates=args.max_camera_search_candidates,
        width=args.width,
        height=args.height,
        snap_heading=args.snap_heading,
    )


def reset_env(env, args: argparse.Namespace):
    reset_args = SimpleNamespace(distractors=1, launch_retries=args.launch_retries)
    return data_collection_reset(env, reset_args)


def map_name_from_env_id(env_id: str) -> str:
    prefix = "UnrealTrack-"
    suffix = "-ContinuousColor-v0"
    if env_id.startswith(prefix) and env_id.endswith(suffix):
        return env_id[len(prefix) : -len(suffix)]
    return env_id


def load_brightness_settings(args: argparse.Namespace) -> tuple[float, float]:
    scale = float(args.brightness_scale)
    offset = float(args.brightness_offset)
    if args.brightness_config is None:
        return scale, offset
    config_path = Path(args.brightness_config)
    if not config_path.exists():
        raise FileNotFoundError(f"brightness config not found: {config_path}")
    data = json.loads(config_path.read_text(encoding="utf-8"))
    maps = data.get("maps", data) if isinstance(data, dict) else {}
    map_key = map_name_from_env_id(args.env_id)
    entry = maps.get(map_key) or maps.get(args.env_id)
    if entry is None:
        return scale, offset
    if isinstance(entry, (int, float)):
        scale = float(entry)
    elif isinstance(entry, dict):
        scale = float(entry.get("scale", entry.get("brightness_scale", scale)))
        offset = float(entry.get("offset", entry.get("brightness_offset", offset)))
    else:
        raise ValueError(f"invalid brightness entry for {map_key}: {entry!r}")
    return scale, offset


def apply_brightness(frame: np.ndarray, scale: float, offset: float) -> np.ndarray:
    if abs(scale - 1.0) < 1e-9 and abs(offset) < 1e-9:
        return frame
    return cv2.convertScaleAbs(frame, alpha=float(scale), beta=float(offset))


def waypoint_xyz(value: Any, label: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        raise ValueError(f"{label} must be [x, y, z], got {value!r}")
    return [float(value[0]), float(value[1]), float(value[2])]


def load_human_loop_waypoints(args: argparse.Namespace) -> list[list[float]]:
    if args.human_path_file is None:
        return []
    path_file = Path(args.human_path_file)
    if not path_file.exists():
        raise FileNotFoundError(f"human path file not found: {path_file}")
    data = json.loads(path_file.read_text(encoding="utf-8"))
    maps = data.get("maps", data) if isinstance(data, dict) else {}
    map_key = map_name_from_env_id(args.env_id)
    entry = maps.get(map_key) or maps.get(args.env_id)
    if entry is None:
        raise EpisodeSkipped(f"human path file has no entry for {map_key}")
    raw_waypoints = entry.get("waypoints", entry) if isinstance(entry, dict) else entry
    waypoints = [waypoint_xyz(point, f"{map_key}.waypoints[{idx}]") for idx, point in enumerate(raw_waypoints)]
    if len(waypoints) < 2:
        raise EpisodeSkipped(f"human path for {map_key} needs at least two waypoints")
    return waypoints


def distance_unreal_units(pose_a: list[float] | np.ndarray, pose_b: list[float] | np.ndarray) -> float:
    return float(np.linalg.norm(pose_xyz(pose_a) - pose_xyz(pose_b)))


def maybe_advance_human_loop_goal(env, setup_info: dict[str, Any], target_pose: list[float], recent_positions: list[np.ndarray], args: argparse.Namespace) -> None:
    waypoints = setup_info.get("human_waypoints") or []
    if len(waypoints) < 2:
        return
    reached = distance_unreal_units(target_pose, setup_info["target_goal"]) < float(args.human_waypoint_reach_distance)
    stalled = False
    stall_window = max(2, int(args.human_waypoint_stall_window))
    if len(recent_positions) >= stall_window:
        stalled = float(np.linalg.norm(recent_positions[-1] - recent_positions[-stall_window])) < float(args.human_waypoint_stall_distance)
    if not reached and not stalled:
        return

    current_index = int(setup_info.get("human_waypoint_index", 0))
    next_index = current_index + 1
    if next_index >= len(waypoints):
        if not args.human_path_loop:
            setup_info["human_path_finished"] = True
            print("[human-path] finished all waypoints; ending collection", flush=True)
            return
        next_index = 0
    new_goal = waypoints[next_index]
    setup_info["human_waypoint_index"] = next_index
    setup_info["target_goal"] = new_goal
    setup_info["target_waypoints"].append(new_goal)
    env.unwrapped.unrealcv.nav_to_goal(setup_info["target_name"], new_goal)
    print(f"[human-path] next waypoint {next_index + 1}/{len(waypoints)} goal={new_goal}", flush=True)


def choose_human_goal(env, target_name: str, target_pose: list[float], rng: random.Random, args: argparse.Namespace):
    return pick_reachable_goal(
        env,
        target_name,
        rng,
        avoid_pos=target_pose,
        min_distance=args.human_goal_min_distance,
        max_distance=args.human_goal_max_distance,
        max_trials=32,
    )


def place_initial_followers(
    env,
    target_id: int,
    robotdog_id: int,
    drone_id: int,
    target_goal: list[float],
    args: argparse.Namespace,
) -> None:
    players = env.unwrapped.player_list
    target_pose = list(env.unwrapped.obj_poses[target_id])
    target_xyz = pose_xyz(target_pose)
    goal_direction = np.asarray(target_goal[:2], dtype=np.float64) - target_xyz[:2]
    if np.linalg.norm(goal_direction) < 1e-6:
        goal_direction = yaw_forward_xy(yaw_deg(target_pose))
    forward = goal_direction / max(float(np.linalg.norm(goal_direction)), 1e-6)
    behind = -forward
    dog_distance = float(args.robotdog_ideal_follow_dist)

    dog_loc = [
        float(target_xyz[0] + behind[0] * dog_distance * UNREAL_UNITS_PER_METER),
        float(target_xyz[1] + behind[1] * dog_distance * UNREAL_UNITS_PER_METER),
        float(target_xyz[2]),
    ]
    drone_loc = [
        float(target_xyz[0] + behind[0] * args.drone_ideal_follow_dist * UNREAL_UNITS_PER_METER),
        float(target_xyz[1] + behind[1] * args.drone_ideal_follow_dist * UNREAL_UNITS_PER_METER),
        float(target_xyz[2] + args.drone_height),
    ]

    env.unwrapped.unrealcv.set_obj_location(players[robotdog_id], dog_loc)
    set_ground_yaw(env, players[robotdog_id], heading_deg(np.asarray(dog_loc), target_xyz))
    env.unwrapped.unrealcv.set_move_bp(players[drone_id], [0.0, 0.0, 0.0, 0.0])
    env.unwrapped.unrealcv.set_obj_location(players[drone_id], drone_loc)
    maybe_set_drone_yaw(env, players[drone_id], heading_deg(np.asarray(drone_loc), target_xyz))
    update_observation(env, refresh_cameras=True)


def get_global_frame(env, args: argparse.Namespace, target_pose: list[float], dog_pose: list[float], drone_pose: list[float]):
    unwrapped = env.unwrapped
    if not getattr(unwrapped, "cam_id", None):
        return None
    cam_id = unwrapped.cam_id[0]
    center = [
        float((target_pose[0] + dog_pose[0] + drone_pose[0]) / 3.0),
        float((target_pose[1] + dog_pose[1] + drone_pose[1]) / 3.0),
        float((target_pose[2] + dog_pose[2] + drone_pose[2]) / 3.0),
    ]
    try:
        if args.top_view_height is not None:
            old_height = getattr(unwrapped, "height_top_view", None)
            unwrapped.height_top_view = float(args.top_view_height)
            unwrapped.set_topview(center, cam_id)
            if old_height is not None:
                unwrapped.height_top_view = old_height
        else:
            unwrapped.set_topview(center, cam_id)
        return ensure_bgr_uint8(unwrapped.unrealcv.get_image(cam_id, "lit", "bmp"))
    except Exception:
        return None


def xy_delta_m(prev_pose: list[float], current_pose: list[float]) -> float:
    prev_xy = np.asarray(prev_pose[:2], dtype=np.float64)
    current_xy = np.asarray(current_pose[:2], dtype=np.float64)
    return float(np.linalg.norm(current_xy - prev_xy) / UNREAL_UNITS_PER_METER)


def show_monitor(args, drone_frame, dog_frame, global_frame, drone_bbox, dog_bbox, step_idx, dist_drone, dist_dog):
    if not args.monitor:
        return
    if step_idx % max(int(args.monitor_interval), 1) != 0:
        return
    drone_view = draw_bbox(drone_frame, drone_bbox, "target")
    dog_view = draw_bbox(dog_frame, dog_bbox, "target")
    cv2.putText(drone_view, f"drone step={step_idx} dist={dist_drone:.2f}m", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(dog_view, f"robotdog step={step_idx} dist={dist_dog:.2f}m", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow("coop_drone_view", resize_for_monitor(drone_view, args.monitor_scale))
    cv2.imshow("coop_robotdog_view", resize_for_monitor(dog_view, args.monitor_scale))
    if global_frame is not None:
        top = global_frame.copy()
        cv2.putText(top, "global top view", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("coop_global_view", resize_for_monitor(top, args.monitor_scale))
    cv2.waitKey(1)


def build_train_episode(args, episode_id, setup, drone_info, dog_info):
    target_start = setup["target_start_pose"]
    robotdog_start = setup["robotdog_start_pose"]
    drone_start = setup["drone_start_pose"]
    info = {
        "dist_ratio": 1.0,
        "robot_position": [float(v) for v in robotdog_start[:3]],
        "robot_rotation": math.radians(yaw_deg(robotdog_start)),
        "drone_position": [float(v) for v in drone_start[:3]],
        "drone_rotation": math.radians(yaw_deg(drone_start)),
        "human_num": 1,
        "num_goals_for_main_human": len(setup["target_waypoints"]),
        "num_goals_for_other_human": 0,
        "main_humanoid_name": setup["target_name"],
        "main_human_semantic_id": 0,
        "extra_humanoid_names": [],
        "instruction": DEFAULT_INSTRUCTION,
        "episode_mode": "aerial_ground_stt",
        "seed": args.seed,
        "robot_type": "robotdog+drone",
        "robotdog_name": setup["robotdog_name"],
        "drone_name": setup["drone_name"],
        "target_control": setup.get("control_mode", "navmesh"),
        "base_velocity_convention": {
            "robotdog": "robotdog body frame [vx_mps, vy_mps, yaw_rate_radps]",
            "drone": "drone body frame command/measurement [vx_mps, vy_mps, yaw_rate_radps]",
        },
        "human_1_start_position": [float(v) for v in target_start[:3]],
        "human_1_start_rotation": math.radians(yaw_deg(target_start)),
    }
    for idx, waypoint in enumerate(setup["target_waypoints"][:3], start=1):
        info[f"human_1_waypoint_{idx}_position"] = [float(v) for v in waypoint]

    return {
        "episode_id": str(episode_id),
        "scene_id": args.env_id,
        "scene_dataset_config": "unrealzoo_cooperative",
        "additional_obj_config_paths": [],
        "start_position": [float(v) for v in target_start[:3]],
        "start_rotation": yaw_to_quat_y(yaw_deg(target_start)),
        "info": info,
        "goals": [{"position": [[float(v) for v in wp]], "radius": None} for wp in setup["target_waypoints"]],
        "start_room": None,
        "shortest_paths": setup["target_path"] or None,
    }


def run_episode(env, args: argparse.Namespace, episode_id: int, rng: random.Random, drone_tracker, dog_tracker):
    obs = reset_env(env, args)
    target_id, robotdog_id, drone_id = classify_coop_agents(env)
    env.unwrapped.target_id = target_id
    env.unwrapped.tracker_id = drone_id
    env.unwrapped.protagonist_id = drone_id
    players = env.unwrapped.player_list
    target_name = players[target_id]
    robotdog_name = players[robotdog_id]
    drone_name = players[drone_id]
    if hasattr(dog_tracker, "last_speed_cm_s"):
        dog_tracker.last_speed_cm_s = 0.0
    for attr in ("last_vx", "last_vy", "last_w"):
        if hasattr(drone_tracker, attr):
            setattr(drone_tracker, attr, 0.0)
    if hasattr(drone_tracker, "prev_target_xy"):
        drone_tracker.prev_target_xy = None
    human_waypoints = load_human_loop_waypoints(args)
    brightness_scale, brightness_offset = load_brightness_settings(args)
    if abs(brightness_scale - 1.0) >= 1e-9 or abs(brightness_offset) >= 1e-9:
        print(
            f"[brightness] map={map_name_from_env_id(args.env_id)} scale={brightness_scale} offset={brightness_offset}",
            flush=True,
        )

    appearances = set_episode_appearances(env, target_id, robotdog_id, [], rng, args)
    if human_waypoints:
        target_start = human_waypoints[0]
        env.unwrapped.unrealcv.set_obj_location(target_name, target_start)
        print(f"[human-path] start={target_start} waypoints={len(human_waypoints)} loop={args.human_path_loop}", flush=True)
    elif args.open_spawn:
        target_start = pick_open_start(env, rng, args)
        env.unwrapped.unrealcv.set_obj_location(target_name, target_start)
    update_observation(env, refresh_cameras=True)
    target_pose = list(env.unwrapped.obj_poses[target_id])
    if human_waypoints:
        human_waypoint_index = 1
        target_goal = human_waypoints[human_waypoint_index]
        target_path = []
    else:
        human_waypoint_index = None
        target_goal, target_path = choose_human_goal(env, target_name, target_pose, rng, args)
    goal_direction = np.asarray(target_goal[:2]) - np.asarray(target_pose[:2])
    if np.linalg.norm(goal_direction) > 1e-6:
        target_yaw = math.degrees(math.atan2(float(goal_direction[1]), float(goal_direction[0])))
        try:
            env.unwrapped.unrealcv.set_obj_rotation(target_name, [0.0, target_yaw, 0.0])
        except Exception:
            pass
    try:
        env.unwrapped.unrealcv.set_max_speed(target_name, float(args.human_speed))
        env.unwrapped.unrealcv.set_max_speed(robotdog_name, float(args.robotdog_max_speed) * UNREAL_UNITS_PER_METER)
    except Exception:
        pass

    update_observation(env, refresh_cameras=True)
    place_initial_followers(env, target_id, robotdog_id, drone_id, target_goal, args)
    if args.keyboard_human:
        try:
            env.unwrapped.unrealcv.set_move_bp(target_name, [0.0, 0.0])
        except Exception:
            pass
    else:
        env.unwrapped.unrealcv.nav_to_goal(target_name, target_goal)

    dog_cfg = dog_args(args)
    drone_cfg = drone_args(args)
    (
        obs,
        dog_pose,
        target_pose,
        dog_visibility,
        dog_visible_hint,
        dog_bbox,
        dog_pitch,
        dog_yaw_offset,
        dog_mount,
        dog_self_visibility,
        dog_self_bbox,
    ) = choose_robotdog_camera_for_current_pose(env, target_id, robotdog_id, dog_cfg)
    (
        obs,
        drone_pose,
        target_pose,
        drone_visibility,
        drone_visible_hint,
        drone_bbox,
        drone_pitch,
        drone_yaw_offset,
    ) = choose_drone_camera_for_current_pose(env, target_id, drone_id, drone_cfg)
    if args.require_visual_target:
        dog_visible = bool(dog_visible_hint and dog_visibility >= args.min_visible_ratio)
        drone_visible = bool(drone_visible_hint and drone_visibility >= args.min_visible_ratio)
        if not dog_visible or not drone_visible:
            raise EpisodeSkipped(
                f"initial cooperative views not visible: robotdog={dog_visible} drone={drone_visible}"
            )

    setup = {
        "target_id": target_id,
        "robotdog_id": robotdog_id,
        "drone_id": drone_id,
        "target_name": target_name,
        "robotdog_name": robotdog_name,
        "drone_name": drone_name,
        "appearances": appearances,
        "target_goal": target_goal,
        "target_waypoints": [target_goal],
        "target_path": target_path,
        "human_waypoints": human_waypoints,
        "human_waypoint_index": human_waypoint_index,
        "human_path_finished": False,
        "target_start_pose": list(env.unwrapped.obj_poses[target_id]),
        "robotdog_start_pose": list(env.unwrapped.obj_poses[robotdog_id]),
        "drone_start_pose": list(env.unwrapped.obj_poses[drone_id]),
        "control_mode": "keyboard_wasd" if args.keyboard_human else ("path_loop" if human_waypoints else "navmesh"),
    }
    recent_target_positions: list[np.ndarray] = []
    frames_drone: list[np.ndarray] = []
    frames_dog: list[np.ndarray] = []
    frames_global: list[np.ndarray] = []
    drone_infos: list[dict[str, Any]] = []
    dog_infos: list[dict[str, Any]] = []
    last_info = None
    collision = False
    manual_stop = False
    t0 = time.time()

    for step_idx in range(args.max_steps):
        if args.keyboard_human and KEY_STATE["esc"]:
            manual_stop = True
            break

        current_target_pose = list(env.unwrapped.obj_poses[target_id])

        current_dog_pose = list(env.unwrapped.obj_poses[robotdog_id])
        if args.snap_heading:
            set_ground_yaw(
                env,
                robotdog_name,
                heading_deg(pose_xyz(current_dog_pose), pose_xyz(current_target_pose)),
            )
        prev_dog_pose = list(env.unwrapped.obj_poses[robotdog_id])
        dog_tracker.snap_heading = bool(args.snap_heading)
        dog_action, dog_commanded_velocity = dog_tracker.act(prev_dog_pose, current_target_pose)

        current_drone_pose = list(env.unwrapped.obj_poses[drone_id])
        if args.snap_heading:
            maybe_set_drone_yaw(
                env,
                drone_name,
                heading_deg(pose_xyz(current_drone_pose), pose_xyz(current_target_pose)),
            )
        prev_drone_pose = list(env.unwrapped.obj_poses[drone_id])
        drone_tracker.snap_heading = bool(args.snap_heading)
        drone_action = drone_tracker.act(prev_drone_pose, current_target_pose)
        drone_commanded_velocity = [float(drone_action[0]), float(drone_action[1]), float(drone_action[3])]
        human_action = keyboard_human_action(args) if args.keyboard_human else None
        target_action = action_for_target_space(env, target_id, human_action) if human_action is not None else None

        actions = [None for _ in players]
        actions[target_id] = target_action
        actions[robotdog_id] = dog_action
        actions[drone_id] = drone_action
        step_t0 = time.perf_counter()
        obs, _rewards, done, last_info = data_collection_step(env, actions)
        step_wall_time_s = time.perf_counter() - step_t0
        dog_pose_after_action = list(env.unwrapped.obj_poses[robotdog_id])
        drone_pose_after_action = list(env.unwrapped.obj_poses[drone_id])

        dog_pose = list(env.unwrapped.obj_poses[robotdog_id])
        target_pose = list(env.unwrapped.obj_poses[target_id])
        (
            obs,
            dog_pose,
            target_pose,
            dog_visibility,
            dog_visible_hint,
            dog_bbox,
            dog_pitch,
            dog_yaw_offset,
            dog_mount,
            dog_self_visibility,
            dog_self_bbox,
        ) = choose_robotdog_camera_for_current_pose(env, target_id, robotdog_id, dog_cfg)

        (
            obs,
            drone_pose,
            target_pose,
            drone_visibility,
            drone_visible_hint,
            drone_bbox,
            drone_pitch,
            drone_yaw_offset,
        ) = choose_drone_camera_for_current_pose(env, target_id, drone_id, drone_cfg)

        drone_frame = ensure_bgr_uint8(obs[drone_id])
        dog_frame = ensure_bgr_uint8(obs[robotdog_id])
        global_frame = get_global_frame(env, args, target_pose, dog_pose, drone_pose)
        drone_frame = apply_brightness(drone_frame, brightness_scale, brightness_offset)
        dog_frame = apply_brightness(dog_frame, brightness_scale, brightness_offset)
        if global_frame is not None:
            global_frame = apply_brightness(global_frame, brightness_scale, brightness_offset)
        frames_drone.append(drone_frame.copy())
        frames_dog.append(dog_frame.copy())
        if args.write_global_video and global_frame is not None:
            frames_global.append(global_frame.copy())

        drone_base_velocity = drone_velocity_from_pose_delta(prev_drone_pose, drone_pose_after_action, args.dt)
        dog_base_velocity = robotdog_velocity_from_pose_delta(prev_dog_pose, dog_pose_after_action, dog_cfg)
        dist_drone = distance_xy_m(drone_pose, target_pose)
        dist_dog = distance_xy_m(dog_pose, target_pose)
        drone_visible = bool(drone_visible_hint and drone_visibility >= args.min_visible_ratio)
        dog_visible = bool(dog_visible_hint and dog_visibility >= args.min_visible_ratio)
        drone_collision = collision_from_info(last_info, drone_id, target_id, dist_drone, drone_pose, target_pose)
        dog_collision = bool(dist_dog < 0.8 and height_gap_m(dog_pose, target_pose) < 1.0)
        collision = collision or drone_collision or dog_collision

        drone_infos.append(
            {
                "step": step_idx + 1,
                "dis_to_human": float(dist_drone),
                "facing": 1.0 if drone_visible else 0.0,
                "base_velocity": drone_base_velocity,
                "commanded_base_velocity": drone_commanded_velocity,
                "drone_action": [float(v) for v in drone_action],
                "dt": float(args.dt),
                "ue_interval_ms": int(args.ue_interval_ms),
                "step_wall_time_s": float(step_wall_time_s),
                "target_visible": bool(drone_visible),
                "target_visibility": float(drone_visibility),
                "target_bbox": drone_bbox,
                "target_center_error": bbox_center_error(drone_bbox, drone_cfg),
                "target_centered": bool(drone_visible and bbox_centered(drone_bbox, drone_cfg)),
                "drone_camera_pitch": float(drone_pitch),
                "drone_camera_yaw_offset": float(drone_yaw_offset),
                "dis_to_human_3d": float(distance_m(drone_pose, target_pose)),
                "collision": bool(drone_collision),
                "drone_pose_after_action": drone_pose_after_action,
                "drone_pose": drone_pose,
                "target_pose": target_pose,
                "target_action": [float(v) for v in human_action] if human_action is not None else None,
                "target_control": setup["control_mode"],
            }
        )
        dog_infos.append(
            {
                "step": step_idx + 1,
                "dis_to_human": float(dist_dog),
                "facing": 1.0 if dog_visible else 0.0,
                "base_velocity": dog_base_velocity,
                "commanded_base_velocity": [float(v) for v in dog_commanded_velocity],
                "ground_action": [float(v) for v in dog_action],
                "dt": float(args.dt),
                "ue_interval_ms": int(args.ue_interval_ms),
                "step_wall_time_s": float(step_wall_time_s),
                "target_visible": bool(dog_visible),
                "target_visibility": float(dog_visibility),
                "target_bbox": dog_bbox,
                "target_center_error": bbox_center_error(dog_bbox, dog_cfg),
                "target_centered": bool(dog_visible and bbox_centered(dog_bbox, dog_cfg)),
                "robotdog_camera_pitch": float(dog_pitch),
                "robotdog_camera_yaw_offset": float(dog_yaw_offset),
                "robotdog_camera_mount": dog_mount,
                "robotdog_self_visibility": float(dog_self_visibility),
                "robotdog_self_bbox": dog_self_bbox,
                "dis_to_human_3d": float(distance_m(dog_pose, target_pose)),
                "collision": bool(dog_collision),
                "robotdog_pose_after_action": dog_pose_after_action,
                "robotdog_pose": dog_pose,
                "target_pose": target_pose,
                "target_action": [float(v) for v in human_action] if human_action is not None else None,
                "target_control": setup["control_mode"],
            }
        )

        if args.debug_motion and (step_idx < 5 or (step_idx + 1) % 10 == 0):
            print(
                f"[motion] ep={episode_id} step={step_idx + 1} "
                f"drone_dist={dist_drone:.2f} dog_dist={dist_dog:.2f} "
                f"drone_vis={int(drone_visible)} dog_vis={int(dog_visible)}",
                flush=True,
            )
            print(
                f"[pose_delta] ep={episode_id} step={step_idx + 1} "
                f"dt={args.dt:.3f}s ue_interval={args.ue_interval_ms}ms wall={step_wall_time_s:.3f}s",
                flush=True,
            )
            print(
                f"  drone: prev->after_step={xy_delta_m(prev_drone_pose, drone_pose_after_action):.4f}m "
                f"after_step->after_camera={xy_delta_m(drone_pose_after_action, drone_pose):.4f}m "
                f"prev->after_camera={xy_delta_m(prev_drone_pose, drone_pose):.4f}m "
                f"cmd={drone_commanded_velocity} base_after_step={drone_base_velocity}",
                flush=True,
            )
            print(
                f"  dog: prev->after_step={xy_delta_m(prev_dog_pose, dog_pose_after_action):.4f}m "
                f"after_step->after_camera={xy_delta_m(dog_pose_after_action, dog_pose):.4f}m "
                f"prev->after_camera={xy_delta_m(prev_dog_pose, dog_pose):.4f}m "
                f"cmd={[float(v) for v in dog_commanded_velocity]} base_after_step={dog_base_velocity}",
                flush=True,
            )

        show_monitor(args, drone_frame, dog_frame, global_frame, drone_bbox, dog_bbox, step_idx + 1, dist_drone, dist_dog)
        recent_target_positions.append(pose_xyz(target_pose))
        if not args.keyboard_human:
            if setup.get("human_waypoints"):
                maybe_advance_human_loop_goal(env, setup, target_pose, recent_target_positions, args)
                if setup.get("human_path_finished"):
                    break
            else:
                maybe_resample_target_goal(env, setup, rng, target_pose, recent_target_positions, args)
        if done or collision:
            break

    if not drone_infos:
        raise EpisodeSkipped("episode stopped before any frames were collected")

    elapsed = max(time.time() - t0, 1e-6)
    total_step = len(drone_infos)
    drone_following = sum(1 for item in drone_infos if item["target_visible"])
    dog_following = sum(1 for item in dog_infos if item["target_visible"])
    status = "ManualStop" if manual_stop else ("Collision" if collision else "Success")
    stat = {
        "finish": True,
        "status": status,
        "success": 0.0 if collision else 1.0,
        "total_step": total_step,
        "collision": 1.0 if collision else 0.0,
        "drone_following_rate": drone_following / max(total_step, 1),
        "robotdog_following_rate": dog_following / max(total_step, 1),
        "instruction": DEFAULT_INSTRUCTION,
        "dt": float(args.dt),
        "ue_interval_ms": int(args.ue_interval_ms),
        "fps": total_step / elapsed,
        "target_control": setup["control_mode"],
        "brightness_scale": float(brightness_scale),
        "brightness_offset": float(brightness_offset),
    }
    return {
        "episode_id": str(episode_id),
        "setup": setup,
        "stat": stat,
        "drone_infos": drone_infos,
        "dog_infos": dog_infos,
        "frames_drone": frames_drone,
        "frames_dog": frames_dog,
        "frames_global": frames_global,
    }


def write_episode_outputs(args: argparse.Namespace, result: dict[str, Any]) -> dict[str, Any]:
    episode_id = result["episode_id"]
    scene_dir = args.out_dir / f"seed_{args.seed}" / safe_slug(args.env_id)
    scene_dir.mkdir(parents=True, exist_ok=True)

    drone_video = scene_dir / f"{episode_id}_drone.mp4"
    dog_video = scene_dir / f"{episode_id}_robotdog.mp4"
    global_video = scene_dir / f"{episode_id}_global.mp4"
    drone_info_path = scene_dir / f"{episode_id}_drone_info.json"
    dog_info_path = scene_dir / f"{episode_id}_robotdog_info.json"
    stat_path = scene_dir / f"{episode_id}.json"

    save_mp4(result["frames_drone"], drone_video, args.fps)
    save_mp4(result["frames_dog"], dog_video, args.fps)
    if result["frames_global"]:
        save_mp4(result["frames_global"], global_video, args.fps)
    write_json(drone_info_path, result["drone_infos"])
    write_json(dog_info_path, result["dog_infos"])
    write_json(stat_path, result["stat"])

    print(
        f"[episode {episode_id}] drone={drone_video} robotdog={dog_video} "
        f"global={global_video if result['frames_global'] else 'none'}",
        flush=True,
    )
    return {
        "episode_id": episode_id,
        "drone_video": drone_video,
        "robotdog_video": dog_video,
        "global_video": global_video if result["frames_global"] else None,
        "drone_info": drone_info_path,
        "robotdog_info": dog_info_path,
        "stat": stat_path,
        "episode_config": build_train_episode(args, episode_id, result["setup"], result["drone_infos"], result["dog_infos"]),
    }


def main() -> int:
    args = parse_args()
    set_global_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    listener = start_keyboard_listener(args)
    env = make_env(args)
    drone_tracker = OracleDroneHumanTracker(
        ideal_follow_dist=args.drone_ideal_follow_dist,
        min_follow_dist=args.drone_min_follow_dist,
        max_follow_dist=args.drone_max_follow_dist,
        human_speed_mps=args.human_speed / UNREAL_UNITS_PER_METER,
        max_vx=args.drone_max_speed,
        snap_heading=args.snap_heading,
    )
    dog_tracker = OracleRobotDogHumanTracker(
        ideal_follow_dist=args.robotdog_ideal_follow_dist,
        min_follow_dist=args.robotdog_min_follow_dist,
        max_follow_dist=args.robotdog_max_follow_dist,
        human_speed_cm_s=args.human_speed,
        max_speed_cm_s=args.robotdog_max_speed * UNREAL_UNITS_PER_METER,
        dt=args.dt,
        snap_heading=args.snap_heading,
    )
    rng = random.Random(args.seed)
    saved: list[dict[str, Any]] = []
    attempts = 0
    max_attempts = int(args.max_attempts) if args.max_attempts is not None else max(args.episodes * 3, args.episodes)

    try:
        while len(saved) < args.episodes and attempts < max_attempts:
            episode_id = len(saved)
            attempts += 1
            print(f"[episode {episode_id}] start attempt={attempts}", flush=True)
            try:
                result = run_episode(env, args, episode_id, rng, drone_tracker, dog_tracker)
            except EpisodeSkipped as exc:
                print(f"[episode {episode_id}] skipped: {exc}", flush=True)
                continue
            saved.append(write_episode_outputs(args, result))
            stat = result["stat"]
            print(
                f"[episode {episode_id}] status={stat['status']} steps={stat['total_step']} "
                f"drone_rate={stat['drone_following_rate']:.2f} dog_rate={stat['robotdog_following_rate']:.2f}",
                flush=True,
            )
    finally:
        if listener is not None:
            listener.stop()
        env.close()
        if args.monitor:
            cv2.destroyAllWindows()

    train_path = args.out_dir / "train.json"
    write_json(train_path, {"episodes": [item["episode_config"] for item in saved]})
    print(f"[done] train={train_path}", flush=True)
    print(f"[done] saved_episodes={len(saved)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
