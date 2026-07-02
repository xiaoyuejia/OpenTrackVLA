"""Collect small UnrealZoo drone-human tracking episodes.

The default output mirrors the newtrack/OpenTrackVLA simulation layout:

    <out_dir>/
        train.json
        seed_<seed>/
            <scene_key>/
                <episode_id>.mp4
                <episode_id>_info.json
                <episode_id>.json

Drone commands are sent to UnrealZoo as [vx, vy, vz, vyaw]. Per-step training
labels are saved as [vx, vy, w], where w is the same yaw command.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
import types
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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gym_unrealcv  # noqa: F401  # Registers UnrealZoo gym envs.
from gym_unrealcv.envs.base_env import UnrealCv_base
from gym_unrealcv.envs.utils import misc
from gym_unrealcv.envs.wrappers import augmentation, configUE, time_dilation
from unrealcv import api as unrealcv_api


DEFAULT_ENV_ID = "UnrealTrack-DowntownWest-ContinuousColor-v0"
DEFAULT_OUT_DIR = "/data/hdt/ntv_data/sim_data/unrealzoo_drone_human"
DEFAULT_INSTRUCTION = (
    "The aerial drone and the ground robot dog must cooperatively track the same target person. "
    "The drone should follow the person from the air, and the robot dog should follow the same person on the ground."
)
UNREAL_UNITS_PER_METER = 100.0


class EpisodeSkipped(RuntimeError):
    """Raised when a reset cannot produce a visible target start."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect OpenTrackVLA-style UnrealZoo drone-human tracking data."
    )
    parser.add_argument("--env-id", default=DEFAULT_ENV_ID, help="UnrealZoo gym env id.")
    parser.add_argument("--episodes", type=int, default=2, help="Number of successful episodes to save.")
    parser.add_argument("--max-attempts", type=int, default=None, help="Maximum episode attempts before stopping.")
    parser.add_argument("--max-steps", type=int, default=80, help="Maximum steps per episode.")
    parser.add_argument("--seed", type=int, default=100, help="Random seed.")
    parser.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR), help="Output root.")
    parser.add_argument("--fps", type=int, default=10, help="Saved MP4 FPS.")
    parser.add_argument("--ideal-follow-dist", type=float, default=2.8, help="Ideal distance to human, meters.")
    parser.add_argument("--min-follow-dist", type=float, default=1.5, help="Minimum distance before retreat, meters.")
    parser.add_argument("--max-follow-dist", type=float, default=4.0, help="Maximum normal follow distance, meters.")
    parser.add_argument("--drone-height", type=float, default=1000.0, help="Drone height above target, Unreal units.")
    parser.add_argument("--human-speed", type=float, default=90.0, help="Target human NavMesh speed, Unreal units per second.")
    parser.add_argument("--drone-max-speed", type=float, default=0.1, help="Drone follow speed limit, meters per second.")
    parser.add_argument(
        "--snap-heading",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use simulator-side oracle heading. Disabled by default so yaw is "
            "executed by the expert action and can be learned by the model."
        ),
    )
    parser.add_argument("--follow-behind", action=argparse.BooleanOptionalAction, default=True, help="Keep drone behind the target human's walking direction.")
    parser.add_argument("--max-lost-steps", type=int, default=20, help="Stop after this many lost steps.")
    parser.add_argument(
        "--min-visible-ratio",
        type=float,
        default=0.001,
        help="Object-mask visibility threshold for target_visible.",
    )
    parser.add_argument(
        "--min-episode-visible-rate",
        type=float,
        default=0.8,
        help="Reject saved episodes below this target_visible ratio.",
    )
    parser.add_argument(
        "--target-center-tolerance",
        type=float,
        default=0.25,
        help="Max normalized bbox-center error from image center for target_centered.",
    )
    parser.add_argument(
        "--min-episode-centered-rate",
        type=float,
        default=0.5,
        help="Reject saved episodes below this target_centered ratio.",
    )
    parser.add_argument(
        "--require-centered-target",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer camera poses that put the target bbox center near the image center.",
    )
    parser.add_argument(
        "--drone-camera-mode",
        choices=["fixed", "oracle"],
        default="fixed",
        help="fixed keeps one body-bound camera; oracle searches pitch/yaw to keep the target centered.",
    )
    parser.add_argument("--render", action="store_true", help="Show drone RGB frames with cv2.")
    parser.add_argument("--monitor", action="store_true", help="Show live drone and top-view monitor windows.")
    parser.add_argument("--monitor-interval", type=int, default=2, help="Refresh monitor every N steps.")
    parser.add_argument("--monitor-scale", type=float, default=0.75, help="Scale factor for monitor windows.")
    parser.add_argument("--monitor-top-view", action=argparse.BooleanOptionalAction, default=True, help="Show global top-view monitor.")
    parser.add_argument("--monitor-drone-view", action=argparse.BooleanOptionalAction, default=True, help="Show drone first-person monitor.")
    parser.add_argument("--debug-motion", action="store_true", help="Print drone command and measured motion.")
    parser.add_argument(
        "--offscreen",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Launch UE with offscreen rendering. Use --no-offscreen to disable.",
    )
    parser.add_argument(
        "--time-dilation",
        type=int,
        default=-1,
        help="Simulator time dilation; <=0 leaves the env default unchanged.",
    )
    parser.add_argument("--write-frames", action="store_true", help="Also save per-frame jpgs for debugging.")
    parser.add_argument(
        "--write-vla-windows",
        action="store_true",
        help="Also write old-style windowed dataset.json. Implies --write-frames.",
    )
    parser.add_argument("--history-len", type=int, default=8, help="Past frames per optional VLA item.")
    parser.add_argument("--future-horizon", type=int, default=9, help="Future actions per optional VLA item.")
    parser.add_argument("--dt", type=float, default=0.1, help="Integration interval for optional VLA trajectories.")
    parser.add_argument("--human-goal-min-distance", type=float, default=600.0, help="Min target-human goal distance, Unreal units.")
    parser.add_argument("--human-goal-max-distance", type=float, default=2500.0, help="Max target-human goal distance, Unreal units.")
    parser.add_argument("--distractors", type=int, default=0, help="Optional extra humans, clipped to [0, 3].")
    parser.add_argument("--width", type=int, default=640, help="RGB frame width.")
    parser.add_argument("--height", type=int, default=480, help="RGB frame height.")
    parser.add_argument("--noise-std", type=float, default=0.0, help="Small command noise; disabled by default.")
    parser.add_argument("--launch-retries", type=int, default=2, help="Retry UE/UnrealCV startup on socket init failure.")
    parser.add_argument(
        "--open-spawn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Place the target and drone in a more open safe_start area before collecting.",
    )
    parser.add_argument(
        "--open-spawn-radius",
        type=float,
        default=900.0,
        help="Radius in Unreal units used to score open spawn areas.",
    )
    parser.add_argument(
        "--min-open-clearance",
        type=float,
        default=300.0,
        help="Minimum XY clearance from reset-area boundaries for open spawns, Unreal units.",
    )
    parser.add_argument(
        "--open-spawn-candidates",
        type=int,
        default=96,
        help="Number of safe_start candidates to score for open spawning.",
    )
    parser.add_argument(
        "--drone-navmesh-tolerance",
        type=float,
        default=450.0,
        help="Max XY distance from drone follow point to a safe_start point when checking open placement.",
    )
    parser.add_argument(
        "--kinematic-follow",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Debug fallback: keep the drone in a visible follow pose and derive [vx, vy, w] labels from pose deltas.",
    )
    parser.add_argument(
        "--use-mask-visibility",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use UnrealCV object-mask visibility checks. Enabled by default to ensure the person is in frame.",
    )
    parser.add_argument(
        "--require-visual-target",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require actual camera/mask visibility instead of accepting geometric visibility fallback.",
    )
    parser.add_argument(
        "--drone-camera-pitches",
        default="-60",
        help="Comma-separated drone camera pitch candidates searched to keep the target in frame.",
    )
    parser.add_argument(
        "--drone-camera-fixed-pitch",
        type=float,
        default=-60.0,
        help="Use one fixed drone camera pitch, e.g. 45 or -45, instead of searching pitch candidates.",
    )
    parser.add_argument(
        "--drone-camera-fixed-yaw",
        type=float,
        default=0.0,
        help="Use this fixed relative camera yaw offset when --drone-camera-fixed-pitch is set.",
    )
    parser.add_argument(
        "--drone-camera-yaw-offsets",
        default="0",
        help="Comma-separated drone camera yaw offsets searched to keep the target centered.",
    )
    parser.add_argument(
        "--lock-drone-camera-world-xy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force the UnrealCV camera world x/y to match the drone x/y before each capture.",
    )
    parser.add_argument(
        "--drone-camera-z-offset",
        type=float,
        default=0.0,
        help="World z offset added when --lock-drone-camera-world-xy is enabled.",
    )
    parser.add_argument(
        "--max-camera-search-candidates",
        type=int,
        default=12,
        help="Limit camera pitch/yaw candidates per drone pose to keep collection responsive.",
    )
    parser.add_argument("--drone-fov", type=float, default=100.0, help="Drone camera FOV used during collection.")
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def safe_slug(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "scene"


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    return value


def ensure_bgr_uint8(frame: np.ndarray) -> np.ndarray:
    img = np.asarray(frame)
    if img.ndim == 3 and img.shape[2] > 3:
        img = img[:, :, :3]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def numeric_vector(values: Any, label: str, min_size: int = 1, skip_episode: bool = True) -> np.ndarray:
    try:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        if skip_episode:
            raise EpisodeSkipped(f"invalid numeric {label}: {values!r}") from exc
        return np.empty((0,), dtype=np.float64)
    if arr.size < min_size or not np.all(np.isfinite(arr[:min_size])):
        if skip_episode:
            raise EpisodeSkipped(f"invalid numeric {label}: {values!r}") from None
        return np.empty((0,), dtype=np.float64)
    return arr


def pose_xyz(pose: list[float] | np.ndarray) -> np.ndarray:
    return numeric_vector(pose, "pose", min_size=3)[:3]


def yaw_deg(pose: list[float] | np.ndarray) -> float:
    arr = numeric_vector(pose, "pose", min_size=3)
    return float(arr[4]) if arr.size > 4 else 0.0


def yaw_to_quat_y(yaw_degrees: float) -> list[float]:
    half = math.radians(yaw_degrees) / 2.0
    return [0.0, math.sin(half), 0.0, math.cos(half)]


def distance_unreal(pose_a: list[float] | np.ndarray, pose_b: list[float] | np.ndarray) -> float:
    return float(np.linalg.norm(pose_xyz(pose_a) - pose_xyz(pose_b)))


def distance_m(pose_a: list[float] | np.ndarray, pose_b: list[float] | np.ndarray) -> float:
    return distance_unreal(pose_a, pose_b) / UNREAL_UNITS_PER_METER


def distance_xy_m(pose_a: list[float] | np.ndarray, pose_b: list[float] | np.ndarray) -> float:
    xy_a = numeric_vector(pose_a, "pose_a", min_size=2)[:2]
    xy_b = numeric_vector(pose_b, "pose_b", min_size=2)[:2]
    return float(np.linalg.norm(xy_a - xy_b) / UNREAL_UNITS_PER_METER)


def height_gap_m(pose_a: list[float] | np.ndarray, pose_b: list[float] | np.ndarray) -> float:
    return float(abs(float(pose_xyz(pose_a)[2]) - float(pose_xyz(pose_b)[2])) / UNREAL_UNITS_PER_METER)


def yaw_forward_xy(yaw_degrees: float) -> np.ndarray:
    yaw = math.radians(yaw_degrees)
    return np.asarray([math.cos(yaw), math.sin(yaw)], dtype=np.float64)


def normalize_xy(vector: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm > 1e-6:
        return vector / norm
    if fallback is not None:
        fallback_norm = float(np.linalg.norm(fallback))
        if fallback_norm > 1e-6:
            return fallback / fallback_norm
    return np.asarray([1.0, 0.0], dtype=np.float64)


def heading_deg(from_xyz: np.ndarray, to_xyz: np.ndarray) -> float:
    delta = numeric_vector(to_xyz, "to_xyz", min_size=2)[:2] - numeric_vector(from_xyz, "from_xyz", min_size=2)[:2]
    return float(math.degrees(math.atan2(delta[1], delta[0])))


def wrap_deg(angle: float) -> float:
    return float((angle + 180.0) % 360.0 - 180.0)


def parse_float_csv(text: str) -> list[float]:
    values: list[float] = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    return values


def relpath(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


@dataclass
class OracleDroneHumanTracker:
    """Smooth body-frame tracker for drone-human following.

    The command is [vx, vy, vz, vyaw].  A small feed-forward term follows the
    target's recent motion, while a low-gain distance correction keeps the
    drone near the desired radius without the old chase/brake oscillation.
    """

    ideal_follow_dist: float = 4.0
    min_follow_dist: float = 3.0
    max_follow_dist: float = 5.5
    human_speed_mps: float = 0.9
    max_vx: float = 0.12
    max_vy: float = 0.05
    max_w: float = 0.4
    turn_thresh: float = 0.15
    noise_std: float = 0.0
    snap_heading: bool = True
    last_vx: float = 0.0
    last_vy: float = 0.0
    last_w: float = 0.0
    prev_target_xy: Any = None

    def _turn_speed(self, bearing_rad: float, scale: float = 1.0) -> float:
        return float(np.clip(-1.2 * bearing_rad * scale, -self.max_w, self.max_w))

    def act(self, drone_pose: list[float], target_pose: list[float]) -> list[float]:
        bearing_deg = float(misc.get_direction(drone_pose, target_pose))
        bearing_rad = math.radians(bearing_deg)
        dist = distance_xy_m(drone_pose, target_pose)
        yaw = math.radians(yaw_deg(drone_pose))
        forward = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
        right = np.array([-math.sin(yaw), math.cos(yaw)], dtype=np.float64)

        target_xy = pose_xyz(target_pose)[:2].astype(np.float64)
        if self.prev_target_xy is None:
            target_delta_m = np.zeros(2, dtype=np.float64)
        else:
            target_delta_m = (target_xy - np.asarray(self.prev_target_xy, dtype=np.float64)) / UNREAL_UNITS_PER_METER
        self.prev_target_xy = target_xy.copy()

        target_forward_step = float(np.dot(target_delta_m, forward))
        target_right_step = float(np.dot(target_delta_m, right))
        feedforward_vx = 0.35 * target_forward_step
        feedforward_vy = 0.20 * target_right_step

        error = dist - self.ideal_follow_dist
        dead_zone = 0.45
        if abs(error) <= dead_zone:
            distance_vx = 0.0
        else:
            distance_vx = 0.045 * (error - math.copysign(dead_zone, error))

        target_vx = feedforward_vx + distance_vx
        if dist > self.max_follow_dist:
            target_vx = max(target_vx, 0.08)
        elif dist < self.min_follow_dist:
            target_vx = min(target_vx, -0.04)

        bearing_vy = -0.018 * math.sin(bearing_rad)
        target_vy = feedforward_vy + bearing_vy
        target_w = 0.0 if self.snap_heading else self._turn_speed(bearing_rad, scale=0.7)

        if self.noise_std > 0:
            target_vx += float(np.random.normal(0.0, self.noise_std))
            target_vy += float(np.random.normal(0.0, self.noise_std))
            target_w += float(np.random.normal(0.0, self.noise_std * 0.5))

        target_vx = float(np.clip(target_vx, -0.06, self.max_vx))
        target_vy = float(np.clip(target_vy, -self.max_vy, self.max_vy))
        target_w = float(np.clip(target_w, -self.max_w, self.max_w))

        vx = float(self.last_vx + np.clip(target_vx - self.last_vx, -0.02, 0.02))
        vy = float(self.last_vy + np.clip(target_vy - self.last_vy, -0.015, 0.015))
        w = float(self.last_w + np.clip(target_w - self.last_w, -0.10, 0.10))
        self.last_vx, self.last_vy, self.last_w = vx, vy, w
        return [vx, vy, 0.0, w]

def classify_agents(env) -> tuple[int, int, list[int]]:
    players = env.unwrapped.player_list
    human_ids = [
        idx for idx, obj in enumerate(players) if env.unwrapped.agents[obj].get("agent_type") == "player"
    ]
    drone_ids = [
        idx for idx, obj in enumerate(players) if env.unwrapped.agents[obj].get("agent_type") == "drone"
    ]
    if not human_ids or not drone_ids:
        raise RuntimeError(
            f"Need at least one player and one drone, got agents: "
            f"{[(obj, env.unwrapped.agents[obj].get('agent_type')) for obj in players]}"
        )
    target_id = human_ids[0]
    drone_id = drone_ids[0]
    return target_id, drone_id, [idx for idx in human_ids if idx != target_id]


def safe_start_points(env) -> np.ndarray:
    points: list[np.ndarray] = []
    for idx, point in enumerate(getattr(env.unwrapped, "safe_start", []) or []):
        arr = numeric_vector(point, f"safe_start[{idx}]", min_size=3, skip_episode=False)
        if arr.size >= 3:
            points.append(arr[:3])
    if not points:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(points, dtype=np.float64)


def boundary_clearance_xy(point: np.ndarray, reset_area: list[float]) -> float:
    if reset_area is None or len(reset_area) < 4:
        return float("inf")
    x_min, x_max, y_min, y_max = [float(v) for v in reset_area[:4]]
    arr = numeric_vector(point, "point", min_size=2)
    x, y = float(arr[0]), float(arr[1])
    return min(x - x_min, x_max - x, y - y_min, y_max - y)


def open_area_score(env, point: list[float] | np.ndarray, args: argparse.Namespace) -> float:
    """Score safe_start points; high values prefer broad local NavMesh coverage."""
    points = safe_start_points(env)
    if points.size == 0:
        return 0.0
    p = numeric_vector(point, "open_area_point", min_size=3)
    deltas = points[:, :2] - p[:2]
    distances = np.linalg.norm(deltas, axis=1)
    radius = max(float(args.open_spawn_radius), 1.0)
    local = (distances > 120.0) & (distances <= radius)
    local_count = int(np.count_nonzero(local))

    sector_count = 0
    if local_count > 0:
        angles = np.arctan2(deltas[local, 1], deltas[local, 0])
        sectors = np.floor(((angles + math.pi) / (2.0 * math.pi)) * 8.0).astype(int)
        sector_count = len(set(np.clip(sectors, 0, 7).tolist()))

    boundary = boundary_clearance_xy(p, env.unwrapped.reset_area)
    if boundary < float(args.min_open_clearance):
        return -1e9 + boundary
    return sector_count * 1000.0 + min(local_count, 80) * 20.0 + min(boundary, 2000.0)


def pick_open_start(env, rng: random.Random, args: argparse.Namespace, avoid_pos: list[float] | None = None) -> list[float]:
    """
    在环境中的安全起始点中，选择一个周围空间开阔的作为智能体的起始位置。

    该函数通过以下步骤工作：
    1. 获取所有预设的安全起始点。
    2. 随机选择一部分候选点进行评估，以提高效率。
    3. 如果提供了 `avoid_pos`，则排除掉距离该位置过近的候选点。
    4. 使用 `open_area_score` 函数为每个候选点计算“开阔度”得分。
    5. 从得分最高的候选项中随机选择一个点，以在保证质量的同时增加多样性。

    Args:
        env: 模拟环境对象。
        rng: 随机数生成器实例。
        args: 包含配置参数（如 `open_spawn_candidates`）的命名空间。
        avoid_pos: 一个可选的坐标 [x, y, z]，函数将确保所选的起始点与此位置保持足够的距离。

    Raises:
        EpisodeSkipped: 如果环境中没有定义任何安全的起始点。

    Returns:
        一个表示所选起始位置的坐标 [x, y, z]。
    """
    candidates = [list(p) for p in safe_start_points(env)]
    if not candidates:
        raise EpisodeSkipped("safe_start is empty")
    rng.shuffle(candidates)
    scored = []
    for point in candidates[: max(1, int(args.open_spawn_candidates))]:
        if avoid_pos is not None:
            if np.linalg.norm(np.asarray(point[:2]) - np.asarray(avoid_pos[:2])) < 300.0:
                continue
        scored.append((open_area_score(env, point, args), point))
    if not scored:
        return rng.choice(candidates)
    scored.sort(key=lambda item: item[0], reverse=True)
    top = [point for score, point in scored[: min(8, len(scored))] if score > -1e8]
    return rng.choice(top) if top else scored[0][1]


def is_open_drone_location(env, location: list[float], args: argparse.Namespace) -> bool:
    if not args.open_spawn:
        return True
    loc = numeric_vector(location, "drone_location", min_size=3)
    if boundary_clearance_xy(loc, env.unwrapped.reset_area) < float(args.min_open_clearance) * 0.5:
        return False
    points = safe_start_points(env)
    if points.size == 0:
        return True
    nearest = float(np.min(np.linalg.norm(points[:, :2] - loc[:2], axis=1)))
    return nearest <= float(args.drone_navmesh_tolerance)


def place_target_at_open_start(
    env,
    target_id: int,
    rng: random.Random,
    args: argparse.Namespace,
) -> None:
    if not args.open_spawn:
        return
    unwrapped = env.unwrapped
    target_name = unwrapped.player_list[target_id]
    start = pick_open_start(env, rng, args)
    yaw = rng.uniform(-180.0, 180.0)
    try:
        unwrapped.unrealcv.nav_to_goal(target_name, start)
    except Exception:
        pass
    unwrapped.unrealcv.set_obj_location(target_name, start)
    try:
        unwrapped.unrealcv.set_obj_rotation(target_name, [0.0, yaw, 0.0])
    except Exception:
        pass
    update_observation(env, refresh_cameras=True)


def pick_reachable_goal(
    env,
    obj_name: str,
    rng: random.Random,
    avoid_pos: list[float] | None = None,
    min_distance: float = 300.0,
    max_distance: float | None = None,
    max_trials: int = 24,
) -> tuple[list[float], list[list[float]]]:
    candidates = [list(p) for p in safe_start_points(env)]
    if not candidates:
        raise EpisodeSkipped("safe_start is empty")
    rng.shuffle(candidates)
    for goal in candidates[:max_trials]:
        if avoid_pos is not None:
            dist = float(np.linalg.norm(np.asarray(goal[:2]) - np.asarray(avoid_pos[:2])))
            if dist < min_distance:
                continue
            if max_distance is not None and dist > max_distance:
                continue
        try:
            path = env.unwrapped.unrealcv.find_path(obj_name, goal)
        except Exception:
            path = []
        if path and len(path) > 1:
            return goal, path
    for goal in candidates:
        if avoid_pos is None:
            return goal, []
        dist = float(np.linalg.norm(np.asarray(goal[:2]) - np.asarray(avoid_pos[:2])))
        if dist >= min_distance and (max_distance is None or dist <= max_distance):
            return goal, []
    return rng.choice(candidates), []


def maybe_set_drone_yaw(env, drone_name: str, yaw: float) -> bool:
    unrealcv = env.unwrapped.unrealcv
    # BP_drone01's set_rotation command reports object yaw offset by ~180 deg
    # compared with Unreal's world heading. Add 180 here so pose yaw faces target.
    drone_cmd_yaw = wrap_deg(yaw + 180.0)
    try:
        unrealcv.set_rotation(drone_name, drone_cmd_yaw)
        return True
    except Exception:
        try:
            unrealcv.set_obj_rotation(drone_name, [0.0, yaw, 0.0])
            return True
        except Exception:
            return False


def update_observation(env, refresh_cameras: bool = True):
    unwrapped = env.unwrapped
    if refresh_cameras and hasattr(unwrapped, "update_camera_assignments"):
        unwrapped.update_camera_assignments()
    obs, obj_poses, img_show = unwrapped.update_observation(
        unwrapped.player_list,
        unwrapped.cam_list,
        unwrapped.cam_flag,
        unwrapped.observation_type,
    )
    unwrapped.obj_poses = obj_poses
    unwrapped.img_show = img_show
    return obs


def target_mask_visibility(env, cam_id: int, target_name: str) -> tuple[float, bool, list[int]]:
    if cam_id is None or cam_id < 0:
        return 0.0, False, [0, 0, 0, 0]
    try:
        mask_img = env.unwrapped.unrealcv.read_image(cam_id, "object_mask", "direct")
        mask, bbox = env.unwrapped.unrealcv.get_bbox(mask_img, target_name, normalize=False)
        visible_pixels = int(np.count_nonzero(mask))
        total_pixels = max(int(mask_img.shape[0] * mask_img.shape[1]), 1)
        ratio = float(visible_pixels / total_pixels)
        return ratio, visible_pixels > 0, [int(v) for v in bbox]
    except Exception:
        return 0.0, False, [0, 0, 0, 0]


def bbox_center_error(bbox: list[int], args: argparse.Namespace) -> float:
    if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
        return float("inf")
    cx = (float(bbox[0]) + float(bbox[2]) * 0.5) / max(float(args.width), 1.0)
    cy = (float(bbox[1]) + float(bbox[3]) * 0.5) / max(float(args.height), 1.0)
    return float(math.hypot(cx - 0.5, cy - 0.5))


def bbox_centered(bbox: list[int], args: argparse.Namespace) -> bool:
    return bbox_center_error(bbox, args) <= float(args.target_center_tolerance)


def target_visibility(
    env,
    drone_id: int,
    target_name: str,
    drone_pose: list[float],
    target_pose: list[float],
    use_mask: bool = False,
) -> tuple[float, bool, list[int]]:
    cam_id = env.unwrapped.cam_list[drone_id]
    geom_visible = abs(misc.get_direction(drone_pose, target_pose)) < 65.0 and distance_xy_m(drone_pose, target_pose) < 6.0
    if not use_mask:
        return (1.0 if geom_visible else 0.0), geom_visible, [0, 0, 0, 0]
    visibility, visible, bbox = target_mask_visibility(env, cam_id, target_name)
    require_visual = bool(getattr(env.unwrapped, "require_visual_target", True))
    if visible or require_visual:
        return visibility, visible, bbox
    return (1.0 if geom_visible else visibility), bool(geom_visible or visible), bbox


def set_drone_camera(
    env,
    drone_name: str,
    drone_id: int,
    drone_pose: list[float] | None,
    pitch: float,
    yaw_offset: float,
    args: argparse.Namespace,
) -> None:
    cam_id = env.unwrapped.cam_list[drone_id]
    try:
        env.unwrapped.unrealcv.set_cam(drone_name, [0.0, 0.0, 0.0], [0.0, pitch, yaw_offset])
    except Exception:
        pass
    if (
        getattr(args, "lock_drone_camera_world_xy", True)
        and drone_pose is not None
        and cam_id is not None
        and cam_id >= 0
    ):
        # BP_drone has its own camera attachment behavior; for data capture we
        # explicitly lock the actual UnrealCV camera to the drone world x/y so
        # the collected frame always follows the moving drone.
        cam_loc = [
            float(drone_pose[0]),
            float(drone_pose[1]),
            float(drone_pose[2]) + float(getattr(args, "drone_camera_z_offset", 0.0)),
        ]
        cam_yaw = wrap_deg(yaw_deg(drone_pose) + float(yaw_offset))
        try:
            env.unwrapped.unrealcv.set_cam_location(cam_id, cam_loc)
            env.unwrapped.unrealcv.set_cam_rotation(cam_id, [float(pitch), cam_yaw, 0.0])
        except Exception:
            pass
    if cam_id is not None and cam_id >= 0 and args.drone_fov > 0:
        try:
            env.unwrapped.unrealcv.set_cam_fov(cam_id, float(args.drone_fov))
        except Exception:
            pass


def camera_pitch_candidates(drone_pose: list[float], target_pose: list[float], args: argparse.Namespace) -> list[float]:
    if args.drone_camera_fixed_pitch is not None:
        return [float(args.drone_camera_fixed_pitch)]
    configured = parse_float_csv(args.drone_camera_pitches)
    horizontal_cm = max(distance_xy_m(drone_pose, target_pose) * UNREAL_UNITS_PER_METER, 1.0)
    dz_cm = float(pose_xyz(target_pose)[2] - pose_xyz(drone_pose)[2])
    look_down = math.degrees(math.atan2(abs(dz_cm), horizontal_cm))
    analytic = [-look_down, look_down, -(look_down + 10.0), look_down + 10.0, -(look_down - 10.0), look_down - 10.0]
    values: list[float] = []
    for value in analytic + configured:
        value = float(np.clip(value, -80.0, 80.0))
        if all(abs(value - existing) > 1e-3 for existing in values):
            values.append(value)
    return values


def camera_yaw_candidates(args: argparse.Namespace) -> list[float]:
    if args.drone_camera_fixed_pitch is not None:
        return [float(args.drone_camera_fixed_yaw)]
    values = parse_float_csv(args.drone_camera_yaw_offsets)
    return values or [0.0]


def camera_search_pairs(drone_pose: list[float], target_pose: list[float], args: argparse.Namespace) -> list[tuple[float, float]]:
    if getattr(args, "drone_camera_mode", "fixed") == "fixed":
        pitch = (
            float(args.drone_camera_fixed_pitch)
            if args.drone_camera_fixed_pitch is not None
            else float(parse_float_csv(args.drone_camera_pitches)[0])
        )
        return [(pitch, float(args.drone_camera_fixed_yaw))]
    pairs = [(pitch, yaw) for pitch in camera_pitch_candidates(drone_pose, target_pose, args) for yaw in camera_yaw_candidates(args)]
    if args.drone_camera_fixed_pitch is not None:
        return pairs[:1]
    max_pairs = max(1, int(args.max_camera_search_candidates))
    return pairs[:max_pairs]


def desired_follow_location(
    target_pose: list[float],
    current_drone_pose: list[float] | None,
    args: argparse.Namespace,
    angle_offset: float = 0.0,
    follow_direction_xy: np.ndarray | None = None,
) -> list[float]:
    target_xyz = pose_xyz(target_pose)
    radius_cm = args.ideal_follow_dist * UNREAL_UNITS_PER_METER
    if args.follow_behind:
        forward = normalize_xy(
            numeric_vector(follow_direction_xy, "follow_direction_xy", min_size=2)[:2] if follow_direction_xy is not None else np.zeros(2),
            fallback=yaw_forward_xy(yaw_deg(target_pose)),
        )
        direction = -forward
    elif current_drone_pose is not None:
        direction = pose_xyz(current_drone_pose)[:2] - target_xyz[:2]
        if np.linalg.norm(direction) < 1e-6:
            direction = np.array([math.cos(math.radians(yaw_deg(target_pose) + 180.0)),
                                  math.sin(math.radians(yaw_deg(target_pose) + 180.0))])
    else:
        direction = np.array([math.cos(math.radians(yaw_deg(target_pose) + 180.0)),
                              math.sin(math.radians(yaw_deg(target_pose) + 180.0))])
    direction = normalize_xy(numeric_vector(direction, "follow_direction", min_size=2)[:2])
    base_angle = math.atan2(float(direction[1]), float(direction[0])) + angle_offset
    return [
        float(target_xyz[0] + radius_cm * math.cos(base_angle)),
        float(target_xyz[1] + radius_cm * math.sin(base_angle)),
        float(target_xyz[2] + args.drone_height),
    ]


def speed_limited_location(
    current_drone_pose: list[float] | None,
    desired_loc: list[float],
    args: argparse.Namespace,
) -> list[float]:
    if current_drone_pose is None:
        return desired_loc
    current = pose_xyz(current_drone_pose)
    desired = pose_xyz(desired_loc)
    delta = desired - current
    max_step = max(float(args.drone_max_speed), 0.05) * UNREAL_UNITS_PER_METER * max(float(args.dt), 1e-6)
    horizontal_delta = delta[:2]
    horizontal_norm = float(np.linalg.norm(horizontal_delta))
    new_loc = current.copy()
    if horizontal_norm > max_step:
        new_loc[:2] += horizontal_delta / horizontal_norm * max_step
    else:
        new_loc[:2] = desired[:2]
    # Keep altitude tightly controlled so the camera angle remains stable.
    new_loc[2] = desired[2]
    return [float(v) for v in new_loc]


def place_drone_following_target(
    env,
    target_id: int,
    drone_id: int,
    args: argparse.Namespace,
    current_drone_pose: list[float] | None = None,
    refresh_cameras: bool = False,
    follow_direction_xy: np.ndarray | None = None,
) -> tuple[np.ndarray, list[float], list[float], float, bool, list[int], float, float]:
    unwrapped = env.unwrapped
    target_name = unwrapped.player_list[target_id]
    drone_name = unwrapped.player_list[drone_id]
    target_pose = list(unwrapped.obj_poses[target_id])
    if args.follow_behind:
        angle_offsets = [0.0, math.radians(15.0), math.radians(-15.0), math.radians(30.0), math.radians(-30.0)]
    else:
        angle_offsets = [
            0.0,
            math.radians(35.0),
            math.radians(-35.0),
            math.radians(70.0),
            math.radians(-70.0),
            math.radians(110.0),
            math.radians(-110.0),
            math.radians(180.0),
        ]
    best: tuple[np.ndarray, list[float], list[float], float, bool, list[int], float, float] | None = None
    best_score = -float("inf")

    for offset in angle_offsets:
        desired_loc = desired_follow_location(target_pose, current_drone_pose, args, offset, follow_direction_xy)
        drone_loc = speed_limited_location(current_drone_pose, desired_loc, args)
        if not is_open_drone_location(env, drone_loc, args):
            continue
        drone_yaw = heading_deg(np.asarray(drone_loc, dtype=np.float64), pose_xyz(target_pose))
        unwrapped.unrealcv.set_move_bp(drone_name, [0.0, 0.0, 0.0, 0.0])
        unwrapped.unrealcv.set_obj_location(drone_name, drone_loc)
        maybe_set_drone_yaw(env, drone_name, drone_yaw)
        obs = update_observation(env, refresh_cameras=refresh_cameras)
        drone_pose = list(unwrapped.obj_poses[drone_id])
        refreshed_target_pose = list(unwrapped.obj_poses[target_id])
        for pitch, yaw_offset in camera_search_pairs(drone_pose, refreshed_target_pose, args):
            set_drone_camera(env, drone_name, drone_id, drone_pose, pitch, yaw_offset, args)
            obs = update_observation(env, refresh_cameras=False)
            drone_pose = list(unwrapped.obj_poses[drone_id])
            refreshed_target_pose = list(unwrapped.obj_poses[target_id])
            visibility, visible, bbox = target_visibility(
                env,
                drone_id,
                target_name,
                drone_pose,
                refreshed_target_pose,
                use_mask=args.use_mask_visibility,
            )
            center_error = bbox_center_error(bbox, args)
            score = float(visibility) - 0.25 * (center_error if math.isfinite(center_error) else 2.0)
            if score > best_score:
                best_score = score
                best = (obs, drone_pose, refreshed_target_pose, visibility, visible, bbox, pitch, yaw_offset)
            centered_ok = (not args.require_centered_target) or bbox_centered(bbox, args)
            if visible and visibility >= args.min_visible_ratio and centered_ok:
                return obs, drone_pose, refreshed_target_pose, visibility, visible, bbox, pitch, yaw_offset

    if best is None:
        raise EpisodeSkipped("could not place drone near target")
    return best


def choose_drone_camera_for_current_pose(
    env,
    target_id: int,
    drone_id: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[float], list[float], float, bool, list[int], float, float]:
    unwrapped = env.unwrapped
    target_name = unwrapped.player_list[target_id]
    drone_name = unwrapped.player_list[drone_id]
    drone_pose = list(unwrapped.obj_poses[drone_id])
    target_pose = list(unwrapped.obj_poses[target_id])
    if args.snap_heading:
        maybe_set_drone_yaw(env, drone_name, heading_deg(pose_xyz(drone_pose), pose_xyz(target_pose)))
        obs = update_observation(env, refresh_cameras=False)
        drone_pose = list(unwrapped.obj_poses[drone_id])
        target_pose = list(unwrapped.obj_poses[target_id])

    best: tuple[np.ndarray, list[float], list[float], float, bool, list[int], float, float] | None = None
    best_score = -float("inf")
    for pitch, yaw_offset in camera_search_pairs(drone_pose, target_pose, args):
        set_drone_camera(env, drone_name, drone_id, drone_pose, pitch, yaw_offset, args)
        obs = update_observation(env, refresh_cameras=False)
        drone_pose = list(unwrapped.obj_poses[drone_id])
        target_pose = list(unwrapped.obj_poses[target_id])
        visibility, visible, bbox = target_visibility(
            env,
            drone_id,
            target_name,
            drone_pose,
            target_pose,
            use_mask=args.use_mask_visibility,
        )
        center_error = bbox_center_error(bbox, args)
        score = float(visibility) - 0.35 * (center_error if math.isfinite(center_error) else 2.0)
        if score > best_score:
            best_score = score
            best = (obs, drone_pose, target_pose, visibility, visible, bbox, pitch, yaw_offset)
        centered_ok = (not args.require_centered_target) or bbox_centered(bbox, args)
        if visible and visibility >= args.min_visible_ratio and centered_ok:
            return obs, drone_pose, target_pose, visibility, visible, bbox, pitch, yaw_offset

    if best is None:
        raise EpisodeSkipped("could not select a drone camera pose")
    return best


def velocity_from_pose_delta(prev_pose: list[float], current_pose: list[float], dt: float) -> list[float]:
    dt = max(dt, 1e-6)
    delta_m = (pose_xyz(current_pose)[:2] - pose_xyz(prev_pose)[:2]) / (UNREAL_UNITS_PER_METER * dt)
    yaw = math.radians(yaw_deg(prev_pose))
    forward = np.array([math.cos(yaw), math.sin(yaw)])
    right = np.array([-math.sin(yaw), math.cos(yaw)])
    vx = float(np.clip(np.dot(delta_m, forward), -1.0, 1.0))
    vy = float(np.clip(np.dot(delta_m, right), -0.5, 0.5))
    w = float(np.clip(math.radians(wrap_deg(yaw_deg(current_pose) - yaw_deg(prev_pose))) / dt, -0.8, 0.8))
    return [vx, vy, w]


def sample_visible_drone_start(
    env,
    target_id: int,
    drone_id: int,
    rng: random.Random,
    args: argparse.Namespace,
    max_attempts: int = 20,
    follow_direction_xy: np.ndarray | None = None,
) -> tuple[np.ndarray, list[float], float]:
    del rng, max_attempts
    obs, drone_pose, target_pose, _visibility, visible, _bbox, _pitch, _yaw_offset = place_drone_following_target(
        env,
        target_id,
        drone_id,
        args,
        refresh_cameras=True,
        follow_direction_xy=follow_direction_xy,
    )
    if not visible:
        raise EpisodeSkipped("could not place drone with target visible after 20 attempts")
    return obs, drone_pose[:3], heading_deg(pose_xyz(drone_pose), pose_xyz(target_pose))


def data_collection_step(env, actions: list[Any]):
    """Step only the base simulator API, skipping Track reward/visibility logic."""
    return UnrealCv_base.step(env.unwrapped, actions)


def patch_unrealcv_interval_call(unrealcv) -> None:
    """Make this script tolerant of base_env's historical set_interval arg order."""
    if getattr(unrealcv, "_drone_tracking_interval_patch", False):
        return

    original_set_interval = unrealcv.set_interval

    def patched_set_interval(self, player, interval):
        if isinstance(interval, str) and not isinstance(player, str):
            return original_set_interval(interval, player)
        return original_set_interval(player, interval)

    unrealcv.set_interval = types.MethodType(patched_set_interval, unrealcv)
    unrealcv._drone_tracking_interval_patch = True


def patch_unrealcv_camera_query() -> None:
    """Retry early camera-list queries that can race UE startup."""
    cls = unrealcv_api.UnrealCv_API
    if getattr(cls, "_drone_tracking_camera_patch", False):
        return

    def robust_get_camera_num(self):
        last_exc: Exception | None = None
        for _attempt in range(30):
            try:
                if not self.client.isconnected():
                    self.client.connect()
                res = self.client.request("vget /cameras")
                if res:
                    return len(res.split())
            except Exception as exc:
                last_exc = exc
            time.sleep(1.0)
        if last_exc is not None:
            raise RuntimeError("UnrealCV camera query failed after retries") from last_exc
        raise RuntimeError("UnrealCV camera query returned no cameras after retries")

    cls.get_camera_num = robust_get_camera_num
    cls._drone_tracking_camera_patch = True


def close_unreal_process(unwrapped) -> None:
    try:
        if getattr(unwrapped, "unrealcv", None) is not None:
            unwrapped.unrealcv.client.disconnect()
    except Exception:
        pass
    try:
        unwrapped.ue_binary.close()
    except Exception:
        pass
    unwrapped.launched = False


def launch_unreal_with_retry(unwrapped, retries: int) -> bool:
    attempts = max(1, int(retries))
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        print(f"[reset] launch UE attempt={attempt}/{attempts}", flush=True)
        try:
            patch_unrealcv_camera_query()
            launched = unwrapped.launch_ue_env()
            patch_unrealcv_interval_call(unwrapped.unrealcv)
            return bool(launched)
        except Exception as exc:
            last_exc = exc
            print(f"[reset] UE launch failed: {exc}", flush=True)
            close_unreal_process(unwrapped)
            if attempt < attempts:
                time.sleep(5.0)
    if last_exc is not None:
        raise last_exc
    return False


def data_collection_reset(env, args: argparse.Namespace):
    """Reset population through the base env, bypassing Track.reset tracker logic."""
    unwrapped = env.unwrapped
    distractors = int(np.clip(args.distractors, 0, 3))
    population = 2 + distractors
    unwrapped.num_agents = population
    if not unwrapped.launched:
        unwrapped.launched = launch_unreal_with_retry(unwrapped, args.launch_retries)
        print("[reset] init agents/objects", flush=True)
        unwrapped.init_agents()
        unwrapped.init_objects()
    else:
        patch_unrealcv_interval_call(unwrapped.unrealcv)
    print(f"[reset] set_population={population}", flush=True)
    unwrapped.set_population(population)
    print("[reset] base reset", flush=True)
    return UnrealCv_base.reset(unwrapped)


def build_episode_config(
    episode_id: int,
    env_id: str,
    seed: int,
    target_name: str,
    drone_name: str,
    target_pose: list[float],
    drone_pose: list[float],
    target_waypoints: list[list[float]],
    target_path: list[list[float]],
    distractor_ids: list[int],
    distractor_goals: dict[str, list[float]],
    players: list[str],
    unwrapped,
) -> dict[str, Any]:
    target_position = [float(v) for v in target_pose[:3]]
    drone_position = [float(v) for v in drone_pose[:3]]
    info: dict[str, Any] = {
        "dist_ratio": 1.0,
        "robot_position": drone_position,
        "robot_rotation": math.radians(yaw_deg(drone_pose)),
        "human_num": 1 + len(distractor_ids),
        "num_goals_for_main_human": max(1, len(target_waypoints)),
        "num_goals_for_other_human": 1 if distractor_ids else 0,
        "main_humanoid_name": target_name,
        "main_human_semantic_id": 0,
        "extra_humanoid_names": [players[idx] for idx in distractor_ids],
        "instruction": DEFAULT_INSTRUCTION,
        "episode_mode": "stt",
        "seed": seed,
        "drone_name": drone_name,
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
    target_id, drone_id, distractor_ids = classify_agents(env)

    unwrapped = env.unwrapped
    unwrapped.target_id = target_id
    unwrapped.tracker_id = drone_id
    unwrapped.protagonist_id = drone_id

    players = unwrapped.player_list
    target_name = players[target_id]
    drone_name = players[drone_id]

    place_target_at_open_start(env, target_id, rng, args)
    target_pose = list(unwrapped.obj_poses[target_id])
    if args.open_spawn:
        target_goal = pick_open_start(env, rng, args, avoid_pos=target_pose)
        goal_dist = float(np.linalg.norm(np.asarray(target_goal[:2]) - np.asarray(target_pose[:2])))
        if goal_dist < args.human_goal_min_distance or goal_dist > args.human_goal_max_distance:
            target_goal, target_path = pick_reachable_goal(
                env,
                target_name,
                rng,
                avoid_pos=target_pose,
                min_distance=args.human_goal_min_distance,
                max_distance=args.human_goal_max_distance,
            )
        else:
            target_path = []
        try:
            if not target_path:
                target_path = unwrapped.unrealcv.find_path(target_name, target_goal)
        except Exception:
            target_path = []
        if not target_path or len(target_path) <= 1:
            target_goal, target_path = pick_reachable_goal(
                env,
                target_name,
                rng,
                avoid_pos=target_pose,
                min_distance=args.human_goal_min_distance,
                max_distance=args.human_goal_max_distance,
            )
    else:
        target_goal, target_path = pick_reachable_goal(
            env,
            target_name,
            rng,
            avoid_pos=target_pose,
            min_distance=args.human_goal_min_distance,
            max_distance=args.human_goal_max_distance,
        )
    goal_direction = pose_xyz(target_goal)[:2] - pose_xyz(target_pose)[:2]
    if np.linalg.norm(goal_direction) > 1e-6:
        target_yaw = math.degrees(math.atan2(float(goal_direction[1]), float(goal_direction[0])))
        try:
            unwrapped.unrealcv.set_obj_rotation(target_name, [0.0, target_yaw, 0.0])
        except Exception:
            pass
    try:
        unwrapped.unrealcv.set_max_speed(target_name, float(args.human_speed))
    except Exception:
        pass
    update_observation(env, refresh_cameras=True)
    obs, _drone_start, _drone_yaw = sample_visible_drone_start(
        env,
        target_id,
        drone_id,
        rng,
        args,
        follow_direction_xy=goal_direction,
    )
    unwrapped.unrealcv.nav_to_goal(target_name, target_goal)

    distractor_goals: dict[str, list[float]] = {}
    for distractor_id in distractor_ids:
        name = players[distractor_id]
        if args.open_spawn:
            start = pick_open_start(env, rng, args, avoid_pos=target_pose)
            unwrapped.unrealcv.set_obj_location(name, start)
        goal, _ = pick_reachable_goal(env, name, rng, max_trials=8)
        distractor_goals[name] = goal
        unwrapped.unrealcv.nav_to_goal(name, goal)

    obs = update_observation(env)
    episode_config = build_episode_config(
        episode_id=episode_id,
        env_id=args.env_id,
        seed=seed,
        target_name=target_name,
        drone_name=drone_name,
        target_pose=list(unwrapped.obj_poses[target_id]),
        drone_pose=list(unwrapped.obj_poses[drone_id]),
        target_waypoints=[target_goal],
        target_path=target_path,
        distractor_ids=distractor_ids,
        distractor_goals=distractor_goals,
        players=players,
        unwrapped=unwrapped,
    )
    return obs, {
        "target_id": target_id,
        "drone_id": drone_id,
        "distractor_ids": distractor_ids,
        "target_name": target_name,
        "drone_name": drone_name,
        "target_goal": target_goal,
        "target_waypoints": [target_goal],
        "target_path": target_path,
        "distractor_goals": distractor_goals,
        "episode_config": episode_config,
    }


def collision_from_info(
    info: dict[str, Any] | None,
    drone_id: int,
    target_id: int,
    dist_xy: float,
    drone_pose: list[float],
    target_pose: list[float],
) -> bool:
    if info:
        metrics = info.get("metrics", {})
        mat = metrics.get("collision") if isinstance(metrics, dict) else None
        if mat is not None:
            try:
                return bool(np.asarray(mat)[drone_id, target_id])
            except Exception:
                pass
        for key in ("collision", "Collision"):
            if key in info and isinstance(info[key], (bool, int, float)):
                return bool(info[key])
    return bool(dist_xy < 0.8 and height_gap_m(drone_pose, target_pose) < 1.0)


def save_mp4(frames: list[np.ndarray], path: Path, fps: int) -> None:
    if not frames:
        raise ValueError(f"No frames to save for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = ensure_bgr_uint8(frames[0]).shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise IOError(f"Failed to open video writer: {path}")
    try:
        for frame in frames:
            img = ensure_bgr_uint8(frame)
            if img.shape[1] != w or img.shape[0] != h:
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            writer.write(img)
    finally:
        writer.release()


def draw_bbox(frame: np.ndarray, bbox: list[int], label: str = "target") -> np.ndarray:
    img = frame.copy()
    if len(bbox) == 4 and bbox[2] > 0 and bbox[3] > 0:
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, label, (x, max(20, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cx, cy = x + w // 2, y + h // 2
        cv2.drawMarker(img, (cx, cy), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2)
    h_img, w_img = img.shape[:2]
    cv2.drawMarker(img, (w_img // 2, h_img // 2), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=16, thickness=2)
    return img


def resize_for_monitor(frame: np.ndarray, scale: float) -> np.ndarray:
    img = ensure_bgr_uint8(frame)
    if scale <= 0 or abs(scale - 1.0) < 1e-6:
        return img
    h, w = img.shape[:2]
    return cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)


def get_top_view_frame(env, drone_pose: list[float], target_pose: list[float]) -> np.ndarray | None:
    unwrapped = env.unwrapped
    if not getattr(unwrapped, "cam_id", None):
        return None
    cam_id = unwrapped.cam_id[0]
    center = [
        float((drone_pose[0] + target_pose[0]) * 0.5),
        float((drone_pose[1] + target_pose[1]) * 0.5),
        float((drone_pose[2] + target_pose[2]) * 0.5),
    ]
    try:
        unwrapped.set_topview(center, cam_id)
        return ensure_bgr_uint8(unwrapped.unrealcv.get_image(cam_id, "lit", "bmp"))
    except Exception:
        return None


def maybe_show_monitor(
    env,
    args: argparse.Namespace,
    step_idx: int,
    drone_frame: np.ndarray,
    target_bbox: list[int],
    drone_pose: list[float],
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
        if args.monitor_drone_view or args.render:
            view = draw_bbox(drone_frame, target_bbox)
            cv2.putText(
                view,
                f"dist={dist:.2f}m visible={int(visible)} centered={int(centered)}",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.imshow("drone_view", resize_for_monitor(view, args.monitor_scale))
        if args.monitor and args.monitor_top_view:
            top = get_top_view_frame(env, drone_pose, target_pose)
            if top is not None:
                cv2.putText(top, "global top view", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("global_top_view", resize_for_monitor(top, args.monitor_scale))
        cv2.waitKey(1)
    except Exception as exc:
        print(f"[monitor] disabled after cv2/unrealcv display error: {exc}", flush=True)
        args.monitor = False
        args.render = False


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(obj), indent=2), encoding="utf-8")


def maybe_resample_target_goal(
    env,
    setup_info: dict[str, Any],
    rng: random.Random,
    target_pose: list[float],
    recent_positions: list[np.ndarray],
    args: argparse.Namespace,
) -> None:
    target_name = setup_info["target_name"]
    goal = setup_info["target_goal"]
    reached = distance_unreal(target_pose, goal) < 150.0
    stalled = False
    if len(recent_positions) >= 20:
        stalled = float(np.linalg.norm(recent_positions[-1] - recent_positions[-20])) < 20.0
    if not reached and not stalled:
        return

    new_goal, path = pick_reachable_goal(
        env,
        target_name,
        rng,
        avoid_pos=target_pose,
        min_distance=args.human_goal_min_distance,
        max_distance=args.human_goal_max_distance,
    )
    setup_info["target_goal"] = new_goal
    setup_info["target_waypoints"].append(new_goal)
    if path:
        setup_info["target_path"] = path
    env.unwrapped.unrealcv.nav_to_goal(target_name, new_goal)


def run_episode(
    env,
    args: argparse.Namespace,
    episode_id: int,
    rng: random.Random,
    tracker: OracleDroneHumanTracker,
) -> dict[str, Any]:
    obs, setup_info = setup_episode(env, episode_id, args.seed, rng, args)
    unwrapped = env.unwrapped
    players = unwrapped.player_list
    target_id = setup_info["target_id"]
    drone_id = setup_info["drone_id"]
    target_name = setup_info["target_name"]

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
        # --- A. 获取当前状态 ---
        # 假设 env.unwrapped.obj_poses 存储了所有物体的位姿
        current_target_pose = list(env.unwrapped.obj_poses[target_id])
        current_drone_pose = list(env.unwrapped.obj_poses[drone_id])

        # --- B. 计算无人机动作 (保持自动跟踪) ---
        # tracker.act 会根据目标和无人机的位置差计算动作
        drone_action = tracker.act(current_drone_pose, current_target_pose)

        # --- C. 计算人的动作 (改为键盘控制) ---
        human_action = [0.0, 0.0] # [turn_speed, forward_speed]

        # 键盘控制逻辑：W/S 控制前后，A/D 控制转向
        if KEY_STATE['w']:
            human_action[1] = args.human_speed      # 前进
        elif KEY_STATE['s']:
            human_action[1] = -args.human_speed * 0.5  # 后退，速度减半

        if KEY_STATE['a']:
            human_action[0] = -30.0   # 左转角速度 (根据实际环境调整数值)
        elif KEY_STATE['d']:
            human_action[0] = 30.0    # 右转角速度

        # 检查是否按下 ESC 退出
        if KEY_STATE['esc']:
            print("检测到 ESC 键，正在停止录制...")
            break

        # --- D. 组装 Actions ---
        actions = [None for _ in players]
        actions[target_id] = human_action  # <--- 将键盘动作赋给 target (人)
        actions[drone_id] = drone_action   # <--- 将算法动作赋给 drone

        # --- E. 执行步进 ---
        obs, _rewards, done, last_info = data_collection_step(env, actions)

        # --- F. 数据保存逻辑 (保持原有代码不变) ---
        # ... (这里保留原本保存图片和数据的代码) ...

    following_step = sum(
        1
        for item in record_infos
        if item["target_visible"] and args.min_follow_dist <= item["dis_to_human"] <= args.max_follow_dist
    )
    total_step = len(record_infos)
    following_rate = following_step / max(total_step, 1)
    visible_step = sum(1 for item in record_infos if item["target_visible"])
    visible_rate = visible_step / max(total_step, 1)
    centered_step = sum(1 for item in record_infos if item.get("target_centered"))
    centered_rate = centered_step / max(total_step, 1)
    if status == "Normal" and following_rate >= 0.6:
        status = "Success"
    finish = status in ("Success", "Normal")
    success = 1.0 if status in ("Success", "Normal") else 0.0

    setup_info["episode_config"] = build_episode_config(
        episode_id=episode_id,
        env_id=args.env_id,
        seed=args.seed,
        target_name=setup_info["target_name"],
        drone_name=setup_info["drone_name"],
        target_pose=record_infos[0]["target_pose"] if record_infos else list(unwrapped.obj_poses[target_id]),
        drone_pose=record_infos[0]["drone_pose"] if record_infos else list(unwrapped.obj_poses[drone_id]),
        target_waypoints=setup_info["target_waypoints"],
        target_path=setup_info["target_path"],
        distractor_ids=setup_info["distractor_ids"],
        distractor_goals=setup_info["distractor_goals"],
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


def build_vla_dataset(episode_results: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    dataset: list[dict[str, Any]] = []
    for result in episode_results:
        infos = result["record_infos"]
        scene_key = result["scene_key"]
        episode_id = result["episode_id"]
        for idx in range(0, max(0, len(infos) - args.future_horizon + 1)):
            future = infos[idx : idx + args.future_horizon]
            actions = [list(item["base_velocity"]) for item in future]
            cumulative = np.zeros(3, dtype=np.float64)
            trajectory = []
            for action in actions:
                cumulative += np.asarray(action, dtype=np.float64) * args.dt
                trajectory.append([float(v) for v in cumulative])
            history_start = max(0, idx - args.history_len)
            frame_root = Path(f"seed_{args.seed}") / scene_key / episode_id / "frames"
            dataset.append(
                {
                    "images": [(frame_root / f"frame_{j + 1:05d}.jpg").as_posix() for j in range(history_start, idx)],
                    "current": (frame_root / f"frame_{idx + 1:05d}.jpg").as_posix(),
                    "instruction": DEFAULT_INSTRUCTION,
                    "trajectory": trajectory,
                    "actions": actions,
                    "collision": bool(any(item["dis_to_human"] < 1.0 for item in future)),
                    "target_distance": float(infos[idx]["dis_to_human"]),
                }
            )
    return dataset


def write_train_json(out_dir: Path, episode_configs: list[dict[str, Any]]) -> Path:
    path = out_dir / "train.json"
    write_json(path, {"episodes": episode_configs})
    return path


def make_env(args: argparse.Namespace):
    env = gym.make(args.env_id)
    env = configUE.ConfigUEWrapper(env, offscreen=args.offscreen, resolution=(args.width, args.height))
    distractors = int(np.clip(args.distractors, 0, 3))
    env.unwrapped.agents_category = ["player"] * (1 + distractors) + ["drone"]
    env.unwrapped.require_visual_target = bool(args.require_visual_target)
    if args.time_dilation > 0:
        env = time_dilation.TimeDilationWrapper(env, args.time_dilation)
    population = 2 + distractors
    env = augmentation.RandomPopulationWrapper(env, population, population, random_target=False)
    env.seed(args.seed)
    return env


def validate_outputs(args: argparse.Namespace, results: list[dict[str, Any]], train_path: Path) -> None:
    train = json.loads(train_path.read_text(encoding="utf-8"))
    print(f"[check] train episodes={len(train['episodes'])}")
    for result in results:
        infos = json.loads(result["info"].read_text(encoding="utf-8"))
        stat = json.loads(result["stat"].read_text(encoding="utf-8"))
        velocities_ok = all(len(item.get("base_velocity", [])) == 3 for item in infos)
        distances = [float(item["dis_to_human"]) for item in infos]
        visible_rate = sum(1 for item in infos if item.get("target_visible")) / max(len(infos), 1)
        centered_rate = sum(1 for item in infos if item.get("target_centered")) / max(len(infos), 1)
        cap = cv2.VideoCapture(str(result["video"]))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        print(
            f"[check] ep={result['episode_id']} mp4_frames={frame_count} "
            f"steps={len(infos)} velocities_ok={velocities_ok} "
            f"avg_dist={np.mean(distances) if distances else 0:.2f} "
            f"following_rate={stat['following_rate']:.2f} visible_rate={visible_rate:.2f} centered_rate={centered_rate:.2f}"
        )
        if stat["following_rate"] < 0.6:
            print(f"[warn] episode {result['episode_id']} following_rate < 0.6")
        if visible_rate < args.min_episode_visible_rate:
            print(f"[warn] episode {result['episode_id']} visible_rate < {args.min_episode_visible_rate:.2f}")
        if centered_rate < args.min_episode_centered_rate:
            print(f"[warn] episode {result['episode_id']} centered_rate < {args.min_episode_centered_rate:.2f}")
        if distances and not (args.min_follow_dist <= float(np.mean(distances)) <= args.max_follow_dist):
            print(f"[warn] episode {result['episode_id']} average distance outside expected range")
        if frame_count <= 0:
            print(f"[warn] episode {result['episode_id']} mp4 has no readable frames")


def main() -> int:
    args = parse_args()
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    print(">>> 键盘监听已启动！使用 W/A/S/D 控制目标移动，ESC 退出 <<<")
    if args.write_vla_windows:
        args.write_frames = True
    set_global_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(args)
    tracker = OracleDroneHumanTracker(
        ideal_follow_dist=args.ideal_follow_dist,
        min_follow_dist=args.min_follow_dist,
        max_follow_dist=args.max_follow_dist,
        human_speed_mps=args.human_speed / UNREAL_UNITS_PER_METER,
        max_vx=args.drone_max_speed,
        noise_std=args.noise_std,
        snap_heading=args.snap_heading,
    )
    rng = random.Random(args.seed)
    results: list[dict[str, Any]] = []
    episode_configs: list[dict[str, Any]] = []
    attempts = 0
    max_attempts = int(args.max_attempts) if args.max_attempts is not None else max(args.episodes * 3, args.episodes)

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
            ):
                print(
                    f"[episode {episode_id}] rejected status={stat['status']} "
                    f"following_rate={stat['following_rate']:.2f} "
                    f"visible_rate={stat['visible_rate']:.2f} "
                    f"centered_rate={stat['centered_rate']:.2f}; retrying",
                    flush=True,
                )
                continue
            results.append(result)
            episode_configs.append(result["episode_config"])
            train_path = write_train_json(args.out_dir, episode_configs)
            print(
                f"[episode {episode_id}] status={stat['status']} steps={stat['total_step']} "
                f"following_rate={stat['following_rate']:.2f} visible_rate={stat['visible_rate']:.2f} "
                f"centered_rate={stat['centered_rate']:.2f} fps={result['fps']:.2f} "
                f"video={result['video']} train={train_path}",
                flush=True,
            )
    finally:
        env.close()
        if args.render:
            cv2.destroyAllWindows()

    train_path = write_train_json(args.out_dir, episode_configs)
    if args.write_vla_windows:
        dataset = build_vla_dataset(results, args)
        dataset_path = args.out_dir / "dataset.json"
        write_json(dataset_path, dataset)
        print(f"[done] dataset={dataset_path} items={len(dataset)}", flush=True)

    validate_outputs(args, results, train_path)
    print(f"[done] train={train_path}", flush=True)
    print(f"[done] saved_episodes={len(results)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
