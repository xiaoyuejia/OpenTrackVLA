"""Collect small UnrealZoo robot-dog-human tracking episodes.

Output layout follows the newtrack/OpenTrackVLA simulation sample style:

    <out_dir>/
        train.json
        seed_<seed>/
            <scene_key>/
                <episode_id>.mp4
                <episode_id>_info.json
                <episode_id>.json

UnrealZoo v3.0 represents the robot dog as a BP_Character/player appearance
instead of a separate robotdog agent type. This script therefore spawns two
``player`` agents: player 0 is the target human (set_app 1..18) and player 1 is
the tracker robot dog (set_app 20..33).

Per-step ``base_velocity`` is saved as [vx, vy, w] in the robot-dog body frame:
vx/vy are meters per second, and w is yaw rate in radians per second.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from pynput import keyboard
except Exception as exc:  # pragma: no cover - depends on local GUI/input setup.
    keyboard = None
    KEYBOARD_IMPORT_ERROR = exc
else:
    KEYBOARD_IMPORT_ERROR = None

KEY_STATE = {
    "w": False,
    "s": False,
    "a": False,
    "d": False,
    "esc": False,
}

def on_press(key):
    try:
        KEY_STATE[key.char.lower()] = True
    except AttributeError:
        if keyboard is not None and key == keyboard.Key.esc:
            KEY_STATE["esc"] = True

def on_release(key):
    try:
        KEY_STATE[key.char.lower()] = False
    except AttributeError:
        pass
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

import gym_unrealcv  # noqa: F401  # Registers UnrealZoo gym envs.
from gym_unrealcv.envs.utils import misc
from gym_unrealcv.envs.wrappers import augmentation, configUE, time_dilation

from generate_drone_human_tracking_small import (
    DEFAULT_INSTRUCTION,
    EpisodeSkipped,
    UNREAL_UNITS_PER_METER,
    bbox_center_error,
    bbox_centered,
    boundary_clearance_xy,
    data_collection_reset,
    data_collection_step,
    distance_m,
    distance_unreal,
    distance_xy_m,
    draw_bbox,
    ensure_bgr_uint8,
    get_top_view_frame,
    heading_deg,
    height_gap_m,
    jsonable,
    maybe_resample_target_goal,
    normalize_xy,
    numeric_vector,
    open_area_score,
    pick_open_start,
    pick_reachable_goal,
    pose_xyz,
    relpath,
    resize_for_monitor,
    safe_slug,
    safe_start_points,
    save_mp4,
    target_visibility,
    update_observation,
    wrap_deg,
    write_json,
    yaw_deg,
    yaw_to_quat_y,
    yaw_forward_xy,
)


DEFAULT_ENV_ID = "UnrealTrack-DowntownWest-ContinuousColor-v0"
DEFAULT_OUT_DIR = "/data/hdt/ntv_data/sim_data/unrealzoo_robotdog_human"


def parse_float_csv(text: str) -> list[float]:
    values: list[float] = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect OpenTrackVLA-style UnrealZoo robot-dog-human tracking data."
    )
    parser.add_argument("--env-id", default=DEFAULT_ENV_ID, help="Street-like UnrealZoo gym env id.")
    parser.add_argument("--episodes", type=int, default=2, help="Number of successful episodes to save.")
    parser.add_argument("--max-attempts", type=int, default=None, help="Maximum attempts before stopping.")
    parser.add_argument("--max-steps", type=int, default=80, help="Maximum steps per episode.")
    parser.add_argument("--seed", type=int, default=100, help="Random seed.")
    parser.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR), help="Output root.")
    parser.add_argument("--fps", type=int, default=10, help="Saved MP4 FPS.")
    parser.add_argument("--width", type=int, default=640, help="RGB frame width.")
    parser.add_argument("--height", type=int, default=480, help="RGB frame height.")
    parser.add_argument("--dt", type=float, default=0.1, help="Control/logging interval in seconds.")

    parser.add_argument("--ideal-follow-dist", type=float, default=6.0, help="Ideal dog-human distance, meters.")
    parser.add_argument("--min-follow-dist", type=float, default=4.5, help="Minimum dog-human distance, meters.")
    parser.add_argument("--max-follow-dist", type=float, default=8.0, help="Maximum normal dog-human distance, meters.")
    parser.add_argument("--max-lost-steps", type=int, default=20, help="Stop after this many lost steps.")
    parser.add_argument("--human-speed", type=float, default=90.0, help="Target NavMesh speed, Unreal units/s.")
    parser.add_argument("--robotdog-max-speed", type=float, default=1.05, help="Robot dog speed limit, meters/s.")
    parser.add_argument("--robotdog-max-lateral-speed", type=float, default=0.45, help="Logged lateral speed clip, m/s.")
    parser.add_argument("--robotdog-max-yaw-rate", type=float, default=1.0, help="Logged yaw-rate clip, rad/s.")
    parser.add_argument(
        "--snap-heading",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use simulator-side oracle heading. Disabled by default so turn "
            "actions and executed yaw-rate are preserved as learnable labels."
        ),
    )
    parser.add_argument(
        "--kinematic-follow",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Debug fallback: place the dog in a stable follow pose each step and derive [vx, vy, w] labels.",
    )

    parser.add_argument("--open-spawn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--open-spawn-radius", type=float, default=900.0, help="Open-area scoring radius, Unreal units.")
    parser.add_argument("--min-open-clearance", type=float, default=350.0, help="Boundary clearance, Unreal units.")
    parser.add_argument("--open-spawn-candidates", type=int, default=128, help="Safe-start candidates to score.")
    parser.add_argument("--ground-navmesh-tolerance", type=float, default=300.0, help="Nearest safe_start tolerance.")
    parser.add_argument("--human-goal-min-distance", type=float, default=700.0, help="Min human walk distance, Unreal units.")
    parser.add_argument("--human-goal-max-distance", type=float, default=2200.0, help="Max human walk distance, Unreal units.")
    parser.add_argument("--distractors", type=int, default=0, help="Optional extra humans, clipped to [0, 3].")

    parser.add_argument("--human-appearance-min", type=int, default=1)
    parser.add_argument("--human-appearance-max", type=int, default=18)
    parser.add_argument("--robotdog-appearance-min", type=int, default=20)
    parser.add_argument("--robotdog-appearance-max", type=int, default=33)

    parser.add_argument("--robotdog-camera-forward", type=float, default=140.0, help="Dog camera x offset.")
    parser.add_argument("--robotdog-camera-lateral", type=float, default=0.0, help="Dog camera y offset.")
    parser.add_argument("--robotdog-camera-height", type=float, default=110.0, help="Dog camera z offset.")
    parser.add_argument(
        "--robotdog-camera-mounts",
        default="140:0:110,170:0:120,110:0:95,0:120:110,0:90:100,40:90:110,40:-90:110",
        help="Comma-separated x:y:z relative camera mount candidates. Used to avoid the dog head/body.",
    )
    parser.add_argument(
        "--robotdog-camera-fixed-pitch",
        type=float,
        default=None,
        help="Fixed relative camera pitch. If omitted, search candidate pitches.",
    )
    parser.add_argument(
        "--robotdog-camera-pitches",
        default="-15,-8,0,8,15,22,-22",
        help="Comma-separated pitch candidates for keeping the human in frame.",
    )
    parser.add_argument(
        "--robotdog-camera-yaw-offsets",
        default="0,-8,8,-15,15",
        help="Comma-separated yaw-offset candidates for centering the target.",
    )
    parser.add_argument(
        "--robotdog-camera-mode",
        choices=["fixed", "oracle"],
        default="fixed",
        help="fixed keeps one body-bound camera; oracle searches mount/pitch/yaw to keep the target centered.",
    )
    parser.add_argument("--robotdog-fov", type=float, default=95.0, help="Robot dog camera FOV.")
    parser.add_argument("--max-camera-search-candidates", type=int, default=28)
    parser.add_argument(
        "--max-self-visible-ratio",
        type=float,
        default=0.015,
        help="Reject camera candidates where the robot dog itself occupies too much object-mask area.",
    )
    parser.add_argument("--use-mask-visibility", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-visual-target", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-visible-ratio", type=float, default=0.001)
    parser.add_argument("--min-episode-visible-rate", type=float, default=0.75)
    parser.add_argument("--target-center-tolerance", type=float, default=0.30)
    parser.add_argument("--min-episode-centered-rate", type=float, default=0.45)
    parser.add_argument("--require-centered-target", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--render", action="store_true", help="Show robot dog RGB view.")
    parser.add_argument("--monitor", action="store_true", help="Show robot dog view and optional top view.")
    parser.add_argument("--monitor-interval", type=int, default=2)
    parser.add_argument("--monitor-scale", type=float, default=0.75)
    parser.add_argument("--monitor-top-view", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--monitor-dog-view", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug-motion", action="store_true", help="Print robot dog command and measured motion.")
    parser.add_argument("--write-frames", action="store_true", help="Also save debug jpg frames.")
    parser.add_argument("--offscreen", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--time-dilation", type=int, default=-1)
    parser.add_argument("--launch-retries", type=int, default=2)
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def set_ground_yaw(env, obj_name: str, yaw: float) -> bool:
    unrealcv = env.unwrapped.unrealcv
    try:
        unrealcv.set_obj_rotation(obj_name, [0.0, float(yaw), 0.0])
        return True
    except Exception:
        pass
    try:
        unrealcv.set_rotation(obj_name, float(yaw))
        return True
    except Exception:
        return False


def classify_agents(env) -> tuple[int, int, list[int]]:
    players = env.unwrapped.player_list
    player_ids = [
        idx for idx, obj in enumerate(players) if env.unwrapped.agents[obj].get("agent_type") == "player"
    ]
    if len(player_ids) < 2:
        raise RuntimeError(
            "Need at least two player agents: one human target and one robot dog tracker. "
            f"Got {[(obj, env.unwrapped.agents[obj].get('agent_type')) for obj in players]}"
        )
    return player_ids[0], player_ids[1], player_ids[2:]


def set_episode_appearances(
    env,
    target_id: int,
    robotdog_id: int,
    distractor_ids: list[int],
    rng: random.Random,
    args: argparse.Namespace,
) -> dict[str, int]:
    players = env.unwrapped.player_list
    human_min = int(args.human_appearance_min)
    human_max = int(args.human_appearance_max)
    dog_min = int(args.robotdog_appearance_min)
    dog_max = int(args.robotdog_appearance_max)
    appearances: dict[str, int] = {}

    target_app = rng.randint(human_min, human_max)
    env.unwrapped.unrealcv.set_appearance(players[target_id], target_app)
    appearances[players[target_id]] = target_app

    dog_app = rng.randint(dog_min, dog_max)
    env.unwrapped.unrealcv.set_appearance(players[robotdog_id], dog_app)
    appearances[players[robotdog_id]] = dog_app

    for idx in distractor_ids:
        app_id = rng.randint(human_min, human_max)
        env.unwrapped.unrealcv.set_appearance(players[idx], app_id)
        appearances[players[idx]] = app_id
    return appearances


def is_open_ground_location(env, location: list[float], args: argparse.Namespace) -> bool:
    if not args.open_spawn:
        return True
    loc = numeric_vector(location, "ground_location", min_size=3)
    if boundary_clearance_xy(loc, env.unwrapped.reset_area) < float(args.min_open_clearance) * 0.5:
        return False
    points = safe_start_points(env)
    if points.size == 0:
        return True
    nearest = float(np.min(np.linalg.norm(points[:, :2] - loc[:2], axis=1)))
    return nearest <= float(args.ground_navmesh_tolerance)


def place_target_at_open_start(env, target_id: int, rng: random.Random, args: argparse.Namespace) -> None:
    if not args.open_spawn:
        return
    start = pick_open_start(env, rng, args)
    target_name = env.unwrapped.player_list[target_id]
    yaw = rng.uniform(-180.0, 180.0)
    try:
        env.unwrapped.unrealcv.nav_to_goal(target_name, start)
    except Exception:
        pass
    env.unwrapped.unrealcv.set_obj_location(target_name, start)
    set_ground_yaw(env, target_name, yaw)
    update_observation(env, refresh_cameras=True)


def choose_open_goal(
    env,
    target_name: str,
    target_pose: list[float],
    rng: random.Random,
    args: argparse.Namespace,
) -> tuple[list[float], list[list[float]]]:
    if args.open_spawn:
        candidates = [list(p) for p in safe_start_points(env)]
        rng.shuffle(candidates)
        scored: list[tuple[float, list[float]]] = []
        start_xy = pose_xyz(target_pose)[:2]
        for point in candidates[: max(1, int(args.open_spawn_candidates))]:
            dist = float(np.linalg.norm(pose_xyz(point)[:2] - start_xy))
            if dist < args.human_goal_min_distance or dist > args.human_goal_max_distance:
                continue
            scored.append((open_area_score(env, point, args), point))
        scored.sort(key=lambda item: item[0], reverse=True)
        for _score, goal in scored[:12]:
            try:
                path = env.unwrapped.unrealcv.find_path(target_name, goal)
            except Exception:
                path = []
            if path and len(path) > 1:
                return goal, path

    return pick_reachable_goal(
        env,
        target_name,
        rng,
        avoid_pos=target_pose,
        min_distance=args.human_goal_min_distance,
        max_distance=args.human_goal_max_distance,
    )


def camera_pitch_candidates(args: argparse.Namespace) -> list[float]:
    if args.robotdog_camera_fixed_pitch is not None:
        return [float(args.robotdog_camera_fixed_pitch)]
    values = parse_float_csv(args.robotdog_camera_pitches)
    return values or [0.0]


def camera_yaw_candidates(args: argparse.Namespace) -> list[float]:
    values = parse_float_csv(args.robotdog_camera_yaw_offsets)
    return values or [0.0]


def parse_mount_csv(text: str) -> list[list[float]]:
    mounts: list[list[float]] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        parts = [part.strip() for part in item.replace("/", ":").split(":")]
        if len(parts) != 3:
            continue
        try:
            mount = [float(parts[0]), float(parts[1]), float(parts[2])]
        except ValueError:
            continue
        if not any(np.allclose(mount, existing, atol=1e-3) for existing in mounts):
            mounts.append(mount)
    return mounts


def camera_mount_candidates(args: argparse.Namespace) -> list[list[float]]:
    defaults = [[float(args.robotdog_camera_forward), float(args.robotdog_camera_lateral), float(args.robotdog_camera_height)]]
    if getattr(args, "robotdog_camera_mode", "fixed") == "fixed":
        return defaults
    configured = parse_mount_csv(args.robotdog_camera_mounts)
    mounts: list[list[float]] = []
    for mount in defaults + configured:
        if not any(np.allclose(mount, existing, atol=1e-3) for existing in mounts):
            mounts.append(mount)
    return mounts


def camera_search_configs(args: argparse.Namespace) -> list[tuple[list[float], float, float]]:
    if getattr(args, "robotdog_camera_mode", "fixed") == "fixed":
        mount = camera_mount_candidates(args)[0]
        pitch_candidates = camera_pitch_candidates(args)
        yaw_candidates = camera_yaw_candidates(args)
        pitch = float(pitch_candidates[0]) if pitch_candidates else 0.0
        yaw = float(yaw_candidates[0]) if yaw_candidates else 0.0
        return [(mount, pitch, yaw)]
    configs = [
        (mount, pitch, yaw)
        for mount in camera_mount_candidates(args)
        for pitch in camera_pitch_candidates(args)
        for yaw in camera_yaw_candidates(args)
    ]
    return configs[: max(1, int(args.max_camera_search_candidates))]


def set_robotdog_camera(
    env,
    robotdog_name: str,
    robotdog_id: int,
    mount: list[float],
    pitch: float,
    yaw_offset: float,
    args: argparse.Namespace,
) -> None:
    try:
        env.unwrapped.unrealcv.set_cam(
            robotdog_name,
            [float(mount[0]), float(mount[1]), float(mount[2])],
            [0.0, float(pitch), float(yaw_offset)],
        )
    except Exception:
        pass
    cam_id = env.unwrapped.cam_list[robotdog_id]
    if cam_id is not None and cam_id >= 0 and args.robotdog_fov > 0:
        try:
            env.unwrapped.unrealcv.set_cam_fov(cam_id, float(args.robotdog_fov))
        except Exception:
            pass


def object_mask_ratio_and_bbox(env, cam_id: int, object_name: str) -> tuple[float, bool, list[int]]:
    if cam_id is None or cam_id < 0:
        return 0.0, False, [0, 0, 0, 0]
    try:
        mask_img = env.unwrapped.unrealcv.read_image(cam_id, "object_mask", "direct")
        mask, bbox = env.unwrapped.unrealcv.get_bbox(mask_img, object_name, normalize=False)
        visible_pixels = int(np.count_nonzero(mask))
        total_pixels = max(int(mask_img.shape[0] * mask_img.shape[1]), 1)
        return float(visible_pixels / total_pixels), visible_pixels > 0, [int(v) for v in bbox]
    except Exception:
        return 0.0, False, [0, 0, 0, 0]


def visual_metrics(
    env,
    robotdog_id: int,
    target_name: str,
    robotdog_name: str,
    robotdog_pose: list[float],
    target_pose: list[float],
    args: argparse.Namespace,
) -> tuple[float, bool, list[int], float, list[int]]:
    if not args.use_mask_visibility:
        visibility, visible, bbox = target_visibility(
            env,
            robotdog_id,
            target_name,
            robotdog_pose,
            target_pose,
            use_mask=False,
        )
        return visibility, visible, bbox, 0.0, [0, 0, 0, 0]

    cam_id = env.unwrapped.cam_list[robotdog_id]
    target_ratio, target_visible, target_bbox = object_mask_ratio_and_bbox(env, cam_id, target_name)
    self_ratio, _self_visible, self_bbox = object_mask_ratio_and_bbox(env, cam_id, robotdog_name)
    require_visual = bool(getattr(env.unwrapped, "require_visual_target", True))
    if target_visible or require_visual:
        return target_ratio, target_visible, target_bbox, self_ratio, self_bbox

    geom_visible = abs(misc.get_direction(robotdog_pose, target_pose)) < 65.0 and distance_xy_m(
        robotdog_pose, target_pose
    ) < 6.0
    return (1.0 if geom_visible else target_ratio), bool(geom_visible or target_visible), target_bbox, self_ratio, self_bbox


def desired_follow_location( 
    target_pose: list[float], 
    current_robotdog_pose: list[float] | None, 
    args: argparse.Namespace, 
    angle_offset: float = 0.0, 
    follow_direction_xy: np.ndarray | None = None, 
) -> list[float]: 
    target_xyz = pose_xyz(target_pose) 
    forward = normalize_xy( 
        numeric_vector(follow_direction_xy, "follow_direction_xy", min_size=2)[:2]
        if follow_direction_xy is not None 
        else np.zeros(2), 
        fallback=yaw_forward_xy(yaw_deg(target_pose)), 
    ) 
    if current_robotdog_pose is not None and np.linalg.norm(forward) < 1e-6: 
        forward = normalize_xy(target_xyz[:2] - pose_xyz(current_robotdog_pose)[:2], fallback=np.asarray([1.0, 0.0])) 
    
    base_angle = math.atan2(float(-forward[1]), float(-forward[0])) + angle_offset 
    
    radius_cm = float(args.ideal_follow_dist) * UNREAL_UNITS_PER_METER

    return [ 
        float(target_xyz[0] + radius_cm * math.cos(base_angle)), 
        float(target_xyz[1] + radius_cm * math.sin(base_angle)), 
        float(target_xyz[2]), 
    ] 


def speed_limited_location(
    current_pose: list[float] | None,
    desired_loc: list[float],
    args: argparse.Namespace,
) -> list[float]:
    if current_pose is None:
        return desired_loc
    current = pose_xyz(current_pose)
    desired = pose_xyz(desired_loc)
    delta_xy = desired[:2] - current[:2]
    norm = float(np.linalg.norm(delta_xy))
    max_step = max(float(args.robotdog_max_speed), 0.05) * UNREAL_UNITS_PER_METER * max(float(args.dt), 1e-6)
    new_loc = current.copy()
    if norm > max_step:
        new_loc[:2] += delta_xy / norm * max_step
    else:
        new_loc[:2] = desired[:2]
    new_loc[2] = desired[2]
    return [float(v) for v in new_loc]


def velocity_from_pose_delta(
    prev_pose: list[float],
    current_pose: list[float],
    args: argparse.Namespace,
) -> list[float]:
    dt = max(float(args.dt), 1e-6)
    delta_m = (pose_xyz(current_pose)[:2] - pose_xyz(prev_pose)[:2]) / (UNREAL_UNITS_PER_METER * dt)
    yaw = math.radians(yaw_deg(prev_pose))
    forward = np.array([math.cos(yaw), math.sin(yaw)])
    right = np.array([-math.sin(yaw), math.cos(yaw)])
    vx = float(np.clip(np.dot(delta_m, forward), -args.robotdog_max_speed, args.robotdog_max_speed))
    vy = float(np.clip(np.dot(delta_m, right), -args.robotdog_max_lateral_speed, args.robotdog_max_lateral_speed))
    w = float(
        np.clip(
            math.radians(wrap_deg(yaw_deg(current_pose) - yaw_deg(prev_pose))) / dt,
            -args.robotdog_max_yaw_rate,
            args.robotdog_max_yaw_rate,
        )
    )
    return [vx, vy, w]


@dataclass
class OracleRobotDogHumanTracker:
    """Ground controller using UnrealZoo's [turn_angle_deg, speed_cm_s] action."""

    ideal_follow_dist: float = 2.2
    min_follow_dist: float = 1.2
    max_follow_dist: float = 3.5
    human_speed_cm_s: float = 90.0
    max_speed_cm_s: float = 105.0
    max_turn_deg: float = 30.0
    dt: float = 0.1
    snap_heading: bool = True
    last_speed_cm_s: float = 0.0

    def act(self, robotdog_pose: list[float], target_pose: list[float]) -> tuple[list[float], list[float]]:
        bearing = float(misc.get_direction(robotdog_pose, target_pose))
        dist = distance_xy_m(robotdog_pose, target_pose)
        dist_error = dist - self.ideal_follow_dist
        dead_zone = max(0.25, 0.08 * self.ideal_follow_dist)
        kp_cm_s_per_m = 35.0

        if dist < self.min_follow_dist:
            target_speed_cm_s = -0.30 * self.max_speed_cm_s
        elif dist > self.max_follow_dist:
            target_speed_cm_s = self.max_speed_cm_s
        elif abs(dist_error) <= dead_zone:
            target_speed_cm_s = self.human_speed_cm_s
        else:
            target_speed_cm_s = self.human_speed_cm_s + kp_cm_s_per_m * dist_error
            if dist < self.ideal_follow_dist:
                target_speed_cm_s = min(target_speed_cm_s, 0.85 * self.human_speed_cm_s)

        target_speed_cm_s = float(np.clip(target_speed_cm_s, -0.30 * self.max_speed_cm_s, self.max_speed_cm_s))
        accel_limit = max(18.0, 0.25 * self.max_speed_cm_s)
        speed_cm_s = float(
            self.last_speed_cm_s
            + np.clip(target_speed_cm_s - self.last_speed_cm_s, -accel_limit, accel_limit)
        )
        self.last_speed_cm_s = speed_cm_s

        turn_deg = 0.0 if self.snap_heading else float(np.clip(bearing, -self.max_turn_deg, self.max_turn_deg))
        base_velocity = [
            float(speed_cm_s / UNREAL_UNITS_PER_METER),
            0.0,
            float(np.clip(math.radians(bearing) / max(self.dt, 1e-6), -1.0, 1.0)),
        ]
        return [turn_deg, speed_cm_s], base_velocity


def place_robotdog_following_target(
    env,
    target_id: int,
    robotdog_id: int,
    args: argparse.Namespace,
    current_robotdog_pose: list[float] | None = None,
    refresh_cameras: bool = False,
    follow_direction_xy: np.ndarray | None = None,
) -> tuple[np.ndarray, list[float], list[float], float, bool, list[int], float, float, list[float], float, list[int]]:
    unwrapped = env.unwrapped
    players = unwrapped.player_list
    target_name = players[target_id]
    robotdog_name = players[robotdog_id]
    target_pose = list(unwrapped.obj_poses[target_id])
    angle_offsets = [0.0, math.radians(15.0), math.radians(-15.0), math.radians(30.0), math.radians(-30.0),math.radians(45.0),math.radians(-45.0)]
    best: tuple[np.ndarray, list[float], list[float], float, bool, list[int], float, float, list[float], float, list[int]] | None = None
    best_score = -float("inf")
    for offset in angle_offsets:
        desired_loc = desired_follow_location(target_pose, current_robotdog_pose, args, offset, follow_direction_xy)
        dog_loc = speed_limited_location(current_robotdog_pose, desired_loc, args)
        if not is_open_ground_location(env, dog_loc, args):
            continue
        dog_yaw = heading_deg(np.asarray(dog_loc, dtype=np.float64), pose_xyz(target_pose))
        try:
            unwrapped.unrealcv.set_move_bp(robotdog_name, [0.0, 0.0])
        except Exception:
            pass
        unwrapped.unrealcv.set_obj_location(robotdog_name, dog_loc)
        set_ground_yaw(env, robotdog_name, dog_yaw)
        obs = update_observation(env, refresh_cameras=refresh_cameras)
        dog_pose = list(unwrapped.obj_poses[robotdog_id])
        refreshed_target_pose = list(unwrapped.obj_poses[target_id])
        for mount, pitch, yaw_offset in camera_search_configs(args):
            set_robotdog_camera(env, robotdog_name, robotdog_id, mount, pitch, yaw_offset, args)
            obs = update_observation(env, refresh_cameras=False)
            dog_pose = list(unwrapped.obj_poses[robotdog_id])
            refreshed_target_pose = list(unwrapped.obj_poses[target_id])
            visibility, visible, bbox, self_visibility, self_bbox = visual_metrics(
                env,
                robotdog_id,
                target_name,
                robotdog_name,
                dog_pose,
                refreshed_target_pose,
                args,
            )
            center_error = bbox_center_error(bbox, args)
            dist_penalty = abs(distance_xy_m(dog_pose, refreshed_target_pose) - float(args.ideal_follow_dist)) * 0.05
            self_penalty = max(0.0, self_visibility - float(args.max_self_visible_ratio)) * 4.0
            score = float(visibility) - 0.30 * (center_error if math.isfinite(center_error) else 2.0) - dist_penalty - self_penalty
            if score > best_score:
                best_score = score
                best = (
                    obs,
                    dog_pose,
                    refreshed_target_pose,
                    visibility,
                    visible,
                    bbox,
                    pitch,
                    yaw_offset,
                    [float(v) for v in mount],
                    float(self_visibility),
                    self_bbox,
                )
            centered_ok = (not args.require_centered_target) or bbox_centered(bbox, args)
            self_ok = self_visibility <= float(args.max_self_visible_ratio)
            if visible and visibility >= args.min_visible_ratio and centered_ok and self_ok:
                return (
                    obs,
                    dog_pose,
                    refreshed_target_pose,
                    visibility,
                    visible,
                    bbox,
                    pitch,
                    yaw_offset,
                    [float(v) for v in mount],
                    float(self_visibility),
                    self_bbox,
                )

    if best is None:
        raise EpisodeSkipped("could not place robot dog near target")
    return best


def choose_robotdog_camera_for_current_pose(
    env,
    target_id: int,
    robotdog_id: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[float], list[float], float, bool, list[int], float, float, list[float], float, list[int]]:
    unwrapped = env.unwrapped
    players = unwrapped.player_list
    target_name = players[target_id]
    robotdog_name = players[robotdog_id]
    dog_pose = list(unwrapped.obj_poses[robotdog_id])
    target_pose = list(unwrapped.obj_poses[target_id])

    best: tuple[np.ndarray, list[float], list[float], float, bool, list[int], float, float, list[float], float, list[int]] | None = None
    best_score = -float("inf")
    for mount, pitch, yaw_offset in camera_search_configs(args):
        set_robotdog_camera(env, robotdog_name, robotdog_id, mount, pitch, yaw_offset, args)
        obs = update_observation(env, refresh_cameras=False)
        dog_pose = list(unwrapped.obj_poses[robotdog_id])
        target_pose = list(unwrapped.obj_poses[target_id])
        visibility, visible, bbox, self_visibility, self_bbox = visual_metrics(
            env,
            robotdog_id,
            target_name,
            robotdog_name,
            dog_pose,
            target_pose,
            args,
        )
        center_error = bbox_center_error(bbox, args)
        self_penalty = max(0.0, self_visibility - float(args.max_self_visible_ratio)) * 4.0
        score = float(visibility) - 0.30 * (center_error if math.isfinite(center_error) else 2.0) - self_penalty
        if score > best_score:
            best_score = score
            best = (
                obs,
                dog_pose,
                target_pose,
                visibility,
                visible,
                bbox,
                pitch,
                yaw_offset,
                [float(v) for v in mount],
                float(self_visibility),
                self_bbox,
            )
        centered_ok = (not args.require_centered_target) or bbox_centered(bbox, args)
        self_ok = self_visibility <= float(args.max_self_visible_ratio)
        if visible and visibility >= args.min_visible_ratio and centered_ok and self_ok:
            return (
                obs,
                dog_pose,
                target_pose,
                visibility,
                visible,
                bbox,
                pitch,
                yaw_offset,
                [float(v) for v in mount],
                float(self_visibility),
                self_bbox,
            )
    if best is None:
        raise EpisodeSkipped("could not select a robot dog camera pose")
    return best


def sample_visible_robotdog_start(
    env,
    target_id: int,
    robotdog_id: int,
    args: argparse.Namespace,
    follow_direction_xy: np.ndarray | None = None,
) -> tuple[np.ndarray, list[float], list[float], float, bool, list[int], float, float, list[float], float, list[int]]:
    (
        obs,
        dog_pose,
        target_pose,
        visibility,
        visible,
        bbox,
        pitch,
        yaw_offset,
        mount,
        self_visibility,
        self_bbox,
    ) = place_robotdog_following_target(
        env,
        target_id,
        robotdog_id,
        args,
        current_robotdog_pose=None,
        refresh_cameras=True,
        follow_direction_xy=follow_direction_xy,
    )
    if not visible or visibility < args.min_visible_ratio:
        raise EpisodeSkipped("could not initialize robot dog with visible target")
    if self_visibility > args.max_self_visible_ratio:
        raise EpisodeSkipped(
            f"robot dog camera is self-occluded at start: self_visibility={self_visibility:.4f}"
        )
    return obs, dog_pose, target_pose, visibility, visible, bbox, pitch, yaw_offset, mount, self_visibility, self_bbox


def build_episode_config(
    episode_id: int,
    env_id: str,
    seed: int,
    target_name: str,
    robotdog_name: str,
    target_pose: list[float],
    robotdog_pose: list[float],
    target_waypoints: list[list[float]],
    target_path: list[list[float]],
    distractor_ids: list[int],
    distractor_goals: dict[str, list[float]],
    appearances: dict[str, int],
    players: list[str],
    unwrapped,
) -> dict[str, Any]:
    target_position = [float(v) for v in target_pose[:3]]
    robot_position = [float(v) for v in robotdog_pose[:3]]
    info: dict[str, Any] = {
        "dist_ratio": 1.0,
        "robot_position": robot_position,
        "robot_rotation": math.radians(yaw_deg(robotdog_pose)),
        "human_num": 1 + len(distractor_ids),
        "num_goals_for_main_human": max(1, len(target_waypoints)),
        "num_goals_for_other_human": 1 if distractor_ids else 0,
        "main_humanoid_name": target_name,
        "main_human_semantic_id": 0,
        "extra_humanoid_names": [players[idx] for idx in distractor_ids],
        "instruction": DEFAULT_INSTRUCTION,
        "episode_mode": "stt",
        "seed": seed,
        "robot_type": "robotdog",
        "robotdog_name": robotdog_name,
        "robotdog_appearance_id": appearances.get(robotdog_name),
        "human_appearance_id": appearances.get(target_name),
        "base_velocity_convention": "robotdog body frame [vx_mps, vy_mps, yaw_rate_radps]",
        "human_1_start_position": target_position,
        "human_1_start_rotation": math.radians(yaw_deg(target_pose)),
    }
    for waypoint_idx, waypoint in enumerate(target_waypoints[:3], start=1):
        info[f"human_1_waypoint_{waypoint_idx}_position"] = [float(v) for v in waypoint]
    for offset, idx in enumerate(distractor_ids, start=2):
        name = players[idx]
        pose = list(unwrapped.obj_poses[idx])
        info[f"human_{offset}_start_position"] = [float(v) for v in pose[:3]]
        info[f"human_{offset}_start_rotation"] = math.radians(yaw_deg(pose))
        info[f"human_{offset}_waypoint_1_position"] = [float(v) for v in distractor_goals[name]]

    goal_positions = [[[float(v) for v in wp]] for wp in target_waypoints] or [[[float(v) for v in target_position]]]
    return {
        "episode_id": str(episode_id),
        "scene_id": env_id,
        "scene_dataset_config": "unrealzoo",
        "additional_obj_config_paths": [],
        "start_position": target_position,
        "start_rotation": yaw_to_quat_y(yaw_deg(target_pose)),
        "info": info,
        "goals": [{"position": pos, "radius": None} for pos in goal_positions],
        "start_room": None,
        "shortest_paths": target_path or None,
    }


def setup_episode(env, episode_id: int, seed: int, rng: random.Random, args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    obs = data_collection_reset(env, args)
    target_id, robotdog_id, distractor_ids = classify_agents(env)
    unwrapped = env.unwrapped
    unwrapped.target_id = target_id
    unwrapped.tracker_id = robotdog_id
    unwrapped.protagonist_id = robotdog_id
    players = unwrapped.player_list
    target_name = players[target_id]
    robotdog_name = players[robotdog_id]

    appearances = set_episode_appearances(env, target_id, robotdog_id, distractor_ids, rng, args)
    place_target_at_open_start(env, target_id, rng, args)
    target_pose = list(unwrapped.obj_poses[target_id])
    target_goal, target_path = choose_open_goal(env, target_name, target_pose, rng, args)
    goal_direction = pose_xyz(target_goal)[:2] - pose_xyz(target_pose)[:2]
    if np.linalg.norm(goal_direction) > 1e-6:
        set_ground_yaw(env, target_name, math.degrees(math.atan2(float(goal_direction[1]), float(goal_direction[0]))))

    try:
        unwrapped.unrealcv.set_max_speed(target_name, float(args.human_speed))
        unwrapped.unrealcv.set_max_speed(robotdog_name, float(args.robotdog_max_speed) * UNREAL_UNITS_PER_METER)
    except Exception:
        pass
    update_observation(env, refresh_cameras=True)

    (
        obs,
        dog_pose,
        target_pose,
        visibility,
        visible,
        bbox,
        pitch,
        yaw_offset,
        mount,
        self_visibility,
        self_bbox,
    ) = sample_visible_robotdog_start(
        env,
        target_id,
        robotdog_id,
        args,
        follow_direction_xy=goal_direction,
    )
    if not visible:
        raise EpisodeSkipped("initial robot dog frame does not contain target")
    unwrapped.unrealcv.nav_to_goal(target_name, target_goal)

    distractor_goals: dict[str, list[float]] = {}
    for distractor_id in distractor_ids:
        name = players[distractor_id]
        if args.open_spawn:
            start = pick_open_start(env, rng, args, avoid_pos=target_pose)
            unwrapped.unrealcv.set_obj_location(name, start)
        goal, _path = pick_reachable_goal(env, name, rng, max_trials=8)
        distractor_goals[name] = goal
        try:
            unwrapped.unrealcv.set_max_speed(name, float(args.human_speed))
            unwrapped.unrealcv.nav_to_goal(name, goal)
        except Exception:
            pass

    obs = update_observation(env)
    episode_config = build_episode_config(
        episode_id=episode_id,
        env_id=args.env_id,
        seed=seed,
        target_name=target_name,
        robotdog_name=robotdog_name,
        target_pose=list(unwrapped.obj_poses[target_id]),
        robotdog_pose=list(unwrapped.obj_poses[robotdog_id]),
        target_waypoints=[target_goal],
        target_path=target_path,
        distractor_ids=distractor_ids,
        distractor_goals=distractor_goals,
        appearances=appearances,
        players=players,
        unwrapped=unwrapped,
    )
    return obs, {
        "target_id": target_id,
        "robotdog_id": robotdog_id,
        "distractor_ids": distractor_ids,
        "target_name": target_name,
        "robotdog_name": robotdog_name,
        "target_goal": target_goal,
        "target_waypoints": [target_goal],
        "target_path": target_path,
        "distractor_goals": distractor_goals,
        "appearances": appearances,
        "episode_config": episode_config,
        "initial_visibility": visibility,
        "initial_bbox": bbox,
        "initial_camera_pitch": pitch,
        "initial_camera_yaw_offset": yaw_offset,
        "initial_camera_mount": mount,
        "initial_self_visibility": self_visibility,
        "initial_self_bbox": self_bbox,
    }


def collision_from_info(
    info: dict[str, Any] | None,
    robotdog_id: int,
    target_id: int,
    dist_xy: float,
    robotdog_pose: list[float],
    target_pose: list[float],
) -> bool:
    if info:
        metrics = info.get("metrics", {})
        mat = metrics.get("collision") if isinstance(metrics, dict) else None
        if mat is not None:
            try:
                return bool(np.asarray(mat)[robotdog_id, target_id])
            except Exception:
                pass
        for key in ("collision", "Collision"):
            if key in info and isinstance(info[key], (bool, int, float)):
                return bool(info[key])
    return bool(dist_xy < 0.7 and height_gap_m(robotdog_pose, target_pose) < 1.0)


def maybe_show_monitor(
    env,
    args: argparse.Namespace,
    step_idx: int,
    frame: np.ndarray,
    target_bbox: list[int],
    robotdog_pose: list[float],
    target_pose: list[float],
    dist: float,
    visible: bool,
    centered: bool,
) -> None:
    if not args.monitor and not args.render:
        return
    if args.monitor_interval > 1 and step_idx % args.monitor_interval != 0:
        return
    try:
        if args.monitor_dog_view or args.render:
            view = draw_bbox(frame, target_bbox)
            cv2.putText(
                view,
                f"dist={dist:.2f}m visible={int(visible)} centered={int(centered)}",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.imshow("robotdog_view", resize_for_monitor(view, args.monitor_scale))
        if args.monitor and args.monitor_top_view:
            top = get_top_view_frame(env, robotdog_pose, target_pose)
            if top is not None:
                cv2.putText(top, "global top view", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("global_top_view", resize_for_monitor(top, args.monitor_scale))
        cv2.waitKey(1)
    except Exception as exc:
        print(f"[monitor] disabled after cv2/unrealcv display error: {exc}", flush=True)
        args.monitor = False
        args.render = False


def run_episode(
    env,
    args: argparse.Namespace,
    episode_id: int,
    rng: random.Random,
    tracker: OracleRobotDogHumanTracker,
    target_id,
    robotdog_id
) -> dict[str, Any]:
    obs, setup_info = setup_episode(env, episode_id, args.seed, rng, args)
    unwrapped = env.unwrapped
    players = unwrapped.player_list
    target_id = setup_info["target_id"]
    robotdog_id = setup_info["robotdog_id"]
    target_name = setup_info["target_name"]
    robotdog_name = setup_info["robotdog_name"]

    scene_key = safe_slug(args.env_id)
    scene_dir = args.out_dir / f"seed_{args.seed}" / scene_key
    frame_dir = scene_dir / str(episode_id) / "frames"
    frames: list[np.ndarray] = []
    record_infos: list[dict[str, Any]] = []
    last_info: dict[str, Any] | None = None
    lost_count = 0
    collision = False
    status = "Normal"
    recent_target_positions: list[np.ndarray] = []
    t0 = time.time()

    for step_idx in range(args.max_steps):
        t0 = time.time()

        # 1. 获取当前状态
        # 确保 target_id 和 robotdog_id 是有效的索引
        current_target_pose = list(env.unwrapped.obj_poses[target_id])
        current_dog_pose = list(env.unwrapped.obj_poses[robotdog_id])

        # 2. 计算机器狗的自动跟踪动作 (保持原有逻辑)
        # tracker.act 会返回 [line_vel, turn_vel] 或者类似的格式
        dog_action, _ = tracker.act(current_dog_pose, current_target_pose)

        # 3. 【核心修改】生成人的键盘控制动作
        human_action = [0.0, 0.0]  # [turn_speed, forward_speed]

        # W/S 控制前后移动
        if KEY_STATE['w']:
            human_action[1] = args.human_speed      # 前进
        elif KEY_STATE['s']:
            human_action[1] = -args.human_speed * 0.5 # 后退（通常慢一点）

        # A/D 控制左右旋转
        if KEY_STATE['a']:
            human_action[0] = -args.human_turn       # 左转
        elif KEY_STATE['d']:
            human_action[0] = args.human_turn        # 右转

        # 4. 组装所有角色的动作列表
        # 这里的 players 列表必须包含 target 和 robotdog
        actions = [None for _ in players]

        # 将动作分配给对应的 ID
        actions[target_id] = human_action   # 人由键盘控制
        actions[robotdog_id] = dog_action   # 狗由 Tracker 自动控制

        # 5. 执行环境步进 (Step)
        # data_collection_step 是你项目中封装好的函数
        obs, _rewards, done, last_info = data_collection_step(env, actions)

        # 6. 检查是否退出 (按 ESC 键)
        if KEY_STATE['esc']:
            print("检测到 ESC 键，提前结束 Episode...")
            break

        # 7. 保存数据 (保持原有逻辑)
        # 这里通常是把 obs, actions 写入 npz 或 json 文件
        # ... (保留你原有的 save_data 相关代码) ...

        # 8. 渲染或显示 (如果有)
        # ... (保留你原有的 render 代码) ...

        # 9. 简单的帧率控制 (可选)
        # time.sleep(0.02)

    following_step = sum(
        1
        for item in record_infos
        if item["target_visible"]
        and args.min_follow_dist <= item["dis_to_human"] <= args.max_follow_dist
        and item.get("robotdog_self_visibility", 0.0) <= args.max_self_visible_ratio
    )
    total_step = len(record_infos)
    following_rate = following_step / max(total_step, 1)
    visible_step = sum(1 for item in record_infos if item["target_visible"])
    visible_rate = visible_step / max(total_step, 1)
    centered_step = sum(1 for item in record_infos if item.get("target_centered"))
    centered_rate = centered_step / max(total_step, 1)
    clear_camera_step = sum(
        1 for item in record_infos if item.get("robotdog_self_visibility", 0.0) <= args.max_self_visible_ratio
    )
    clear_camera_rate = clear_camera_step / max(total_step, 1)
    if status == "Normal" and following_rate >= 0.6:
        status = "Success"
    finish = status in ("Success", "Normal")
    success = 1.0 if status in ("Success", "Normal") else 0.0

    setup_info["episode_config"] = build_episode_config(
        episode_id=episode_id,
        env_id=args.env_id,
        seed=args.seed,
        target_name=setup_info["target_name"],
        robotdog_name=setup_info["robotdog_name"],
        target_pose=record_infos[0]["target_pose"] if record_infos else list(unwrapped.obj_poses[target_id]),
        robotdog_pose=record_infos[0]["robotdog_pose"] if record_infos else list(unwrapped.obj_poses[robotdog_id]),
        target_waypoints=setup_info["target_waypoints"],
        target_path=setup_info["target_path"],
        distractor_ids=setup_info["distractor_ids"],
        distractor_goals=setup_info["distractor_goals"],
        appearances=setup_info["appearances"],
        players=players,
        unwrapped=unwrapped,
    )

    video_path = scene_dir / f"{episode_id}.mp4"
    info_path = scene_dir / f"{episode_id}_info.json"
    stat_path = scene_dir / f"{episode_id}.json"
    save_mp4(frames, video_path, args.fps)
    write_json(info_path, record_infos)
    episode_stat = {
        "finish": bool(finish),
        "status": status,
        "success": float(success),
        "following_rate": float(following_rate),
        "following_step": int(following_step),
        "visible_rate": float(visible_rate),
        "visible_step": int(visible_step),
        "centered_rate": float(centered_rate),
        "centered_step": int(centered_step),
        "clear_camera_rate": float(clear_camera_rate),
        "clear_camera_step": int(clear_camera_step),
        "total_step": int(total_step),
        "collision": 1.0 if collision else 0.0,
        "instruction": DEFAULT_INSTRUCTION,
    }
    write_json(stat_path, episode_stat)

    elapsed = max(time.time() - t0, 1e-6)
    return {
        "episode_id": str(episode_id),
        "scene_key": scene_key,
        "video": video_path,
        "info": info_path,
        "stat": stat_path,
        "record_infos": record_infos,
        "episode_config": setup_info["episode_config"],
        "episode_stat": episode_stat,
        "fps": total_step / elapsed,
    }


def write_train_json(out_dir: Path, episode_configs: list[dict[str, Any]]) -> Path:
    path = out_dir / "train.json"
    write_json(path, {"episodes": episode_configs})
    return path


def make_env(args: argparse.Namespace):
    env = gym.make(args.env_id)
    env = configUE.ConfigUEWrapper(env, offscreen=args.offscreen, resolution=(args.width, args.height))
    distractors = int(np.clip(args.distractors, 0, 3))
    population = 2 + distractors
    env.unwrapped.agents_category = ["player"] * population
    env.unwrapped.require_visual_target = bool(args.require_visual_target)
    if args.time_dilation > 0:
        env = time_dilation.TimeDilationWrapper(env, args.time_dilation)
    env = augmentation.RandomPopulationWrapper(env, population, population, random_target=False)
    env.seed(args.seed)
    return env


def validate_outputs(args: argparse.Namespace, results: list[dict[str, Any]], train_path: Path) -> None:
    train = json.loads(train_path.read_text(encoding="utf-8"))
    print(f"[check] train episodes={len(train['episodes'])}", flush=True)
    for result in results:
        infos = json.loads(result["info"].read_text(encoding="utf-8"))
        stat = json.loads(result["stat"].read_text(encoding="utf-8"))
        velocities_ok = all(len(item.get("base_velocity", [])) == 3 for item in infos)
        distances = [float(item["dis_to_human"]) for item in infos]
        visible_rate = sum(1 for item in infos if item.get("target_visible")) / max(len(infos), 1)
        centered_rate = sum(1 for item in infos if item.get("target_centered")) / max(len(infos), 1)
        clear_camera_rate = sum(
            1 for item in infos if item.get("robotdog_self_visibility", 0.0) <= args.max_self_visible_ratio
        ) / max(len(infos), 1)
        cap = cv2.VideoCapture(str(result["video"]))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        print(
            f"[check] ep={result['episode_id']} mp4_frames={frame_count} "
            f"steps={len(infos)} velocities_ok={velocities_ok} "
            f"avg_dist={np.mean(distances) if distances else 0:.2f} "
            f"following_rate={stat['following_rate']:.2f} visible_rate={visible_rate:.2f} "
            f"centered_rate={centered_rate:.2f} clear_camera_rate={clear_camera_rate:.2f}",
            flush=True,
        )
        if stat["following_rate"] < 0.6:
            print(f"[warn] episode {result['episode_id']} following_rate < 0.6", flush=True)
        if visible_rate < args.min_episode_visible_rate:
            print(f"[warn] episode {result['episode_id']} visible_rate < {args.min_episode_visible_rate:.2f}", flush=True)
        if centered_rate < args.min_episode_centered_rate:
            print(
                f"[warn] episode {result['episode_id']} centered_rate < {args.min_episode_centered_rate:.2f}",
                flush=True,
            )
        if clear_camera_rate < 0.8:
            print(f"[warn] episode {result['episode_id']} camera sees too much robot dog body/head", flush=True)
        if distances and not (args.min_follow_dist <= float(np.mean(distances)) <= args.max_follow_dist):
            print(f"[warn] episode {result['episode_id']} average distance outside expected range", flush=True)
        if frame_count <= 0:
            print(f"[warn] episode {result['episode_id']} mp4 has no readable frames", flush=True)


def main() -> int:
    args = parse_args()
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    set_global_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(args)
    tracker = OracleRobotDogHumanTracker(
        ideal_follow_dist=args.ideal_follow_dist,
        min_follow_dist=args.min_follow_dist,
        max_follow_dist=args.max_follow_dist,
        human_speed_cm_s=args.human_speed,
        max_speed_cm_s=args.robotdog_max_speed * UNREAL_UNITS_PER_METER,
        dt=args.dt,
        snap_heading=args.snap_heading,
    )
    rng = random.Random(args.seed)
    results: list[dict[str, Any]] = []
    episode_configs: list[dict[str, Any]] = []
    attempts = 0
    max_attempts = int(args.max_attempts) if args.max_attempts is not None else max(args.episodes * 4, args.episodes)

    try:
        while len(results) < args.episodes and attempts < max_attempts:
            episode_id = len(results)
            attempts += 1
            print(f"[episode {episode_id}] start attempt={attempts}", flush=True)
            try:
                result = run_episode(env, args, episode_id, rng, tracker)
            except EpisodeSkipped as exc:
                print(f"[episode {episode_id}] skipped: {exc}", flush=True)
                continue
            stat = result["episode_stat"]
            if (
                stat["status"] in ("Lost", "Collision")
                or stat["following_rate"] < 0.6
                or stat["visible_rate"] < args.min_episode_visible_rate
                or stat["centered_rate"] < args.min_episode_centered_rate
                or stat["clear_camera_rate"] < 0.8
            ):
                print(
                    f"[episode {episode_id}] rejected status={stat['status']} "
                    f"following_rate={stat['following_rate']:.2f} visible_rate={stat['visible_rate']:.2f} "
                    f"centered_rate={stat['centered_rate']:.2f} clear_camera_rate={stat['clear_camera_rate']:.2f}; retrying",
                    flush=True,
                )
                continue
            results.append(result)
            episode_configs.append(result["episode_config"])
            train_path = write_train_json(args.out_dir, episode_configs)
            print(
                f"[episode {episode_id}] status={stat['status']} steps={stat['total_step']} "
                f"following_rate={stat['following_rate']:.2f} visible_rate={stat['visible_rate']:.2f} "
                f"centered_rate={stat['centered_rate']:.2f} clear_camera_rate={stat['clear_camera_rate']:.2f} "
                f"fps={result['fps']:.2f} "
                f"video={result['video']} train={train_path}",
                flush=True,
            )
    finally:
        env.close()
        if args.render or args.monitor:
            cv2.destroyAllWindows()

    train_path = write_train_json(args.out_dir, episode_configs)
    validate_outputs(args, results, train_path)
    print(f"[done] train={train_path}", flush=True)
    print(f"[done] saved_episodes={len(results)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
