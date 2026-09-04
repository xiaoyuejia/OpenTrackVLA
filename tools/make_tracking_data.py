#!/usr/bin/env python3

import argparse
import bisect
import hashlib
import json
import os
import re
import shutil
import subprocess
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    from tools.bbox_spatial import (
        bbox_prompt_from_spatial,
        bbox_spatial_fields,
        normalize_bbox_xywh_to_cxcywh,
    )
except ModuleNotFoundError:  # Support `python tools/make_tracking_data.py`.
    from bbox_spatial import (  # type: ignore
        bbox_prompt_from_spatial,
        bbox_spatial_fields,
        normalize_bbox_xywh_to_cxcywh,
    )

try:
    from tools.waypoint_inverse_dynamics import (
        INVERSE_CONTROL_VERSION,
        InverseDynamicsConfig,
        initial_inverse_state,
        inverse_step_numpy,
    )
except ModuleNotFoundError:  # Support `python tools/make_tracking_data.py`.
    from waypoint_inverse_dynamics import (  # type: ignore
        INVERSE_CONTROL_VERSION,
        InverseDynamicsConfig,
        initial_inverse_state,
        inverse_step_numpy,
    )


@dataclass
class EpisodePaths:
    seed_dir: Path
    run_dir: Path
    stem: str  # without suffixes, e.g., "2" for 2.mp4 / 2_info.json
    mp4: Optional[Path]
    info_json: Path


DEFAULT_MULTI_AGENT_INSTRUCTION = "Follow the person."
MULTI_AGENT_SCHEMA_VERSION = "multi_agent_tracking_v2_origin_waypoint"
RECORDED_POSE_SCHEMA_VERSION = "multi_agent_tracking_v5_global_waypoint"
ACTION_FIELD_AUTO = "auto"


@dataclass
class PairedEpisode:
    """双 Agent 原始 episode 的文件集合。

    数据流：
    - input_root/run_dir 用于定位 UnrealZoo 原始采集目录。
    - drone/robotdog 的 mp4 用于抽帧。
    - drone/robotdog 的 *_info.json 用于读取动作、bbox、可见性等逐步标签。
    - status_json 是 episode 级质量过滤信息，可选。
    """

    input_root: Path
    run_dir: Path
    rel_run_dir: Path
    stem: str
    drone_mp4: Path
    robotdog_mp4: Path
    drone_info_json: Path
    robotdog_info_json: Path
    status_json: Optional[Path]


def find_ffmpeg_executable() -> Optional[str]:
    """Return path to ffmpeg if available, else None."""
    return shutil.which("ffmpeg")


def natural_sort_key(s: str):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


def list_sorted_images(directory: Path) -> List[Path]:
    image_paths = [p for p in directory.glob("*.jpg")]
    image_paths.sort(key=lambda p: natural_sort_key(p.name))
    return image_paths


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def extract_frames_ffmpeg(ffmpeg_path: str, mp4_path: Path, out_dir: Path, quality: int = 2) -> List[Path]:
    """Extract all frames from mp4 using ffmpeg. Returns list of frame paths."""
    ensure_dir(out_dir)
    for stale_frame in list_sorted_images(out_dir):
        stale_frame.unlink()
    pattern = str(out_dir / "frame_%05d.jpg")
    cmd = [
        ffmpeg_path,
        "-y",
        "-i",
        str(mp4_path),
        "-q:v",
        str(quality),
        str(pattern),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return list_sorted_images(out_dir)


def sample_indices(num_items: int, target_count: int) -> List[int]:
    """Evenly sample indices from [0, num_items-1] to length target_count (<= num_items)."""
    if target_count <= 0 or num_items <= 0:
        return []
    if target_count >= num_items:
        return list(range(num_items))
    # Even spacing
    import numpy as np
    positions = np.linspace(0, num_items - 1, target_count)
    return [int(round(p)) for p in positions]


def pad_to_length(items: List, length: int) -> List:
    if length <= 0:
        return []
    if not items:
        # replicate a placeholder value (should not happen normally)
        return [items for _ in range(length)]  # type: ignore
    if len(items) >= length:
        return items[:length]
    padded = list(items)
    last = items[-1]
    while len(padded) < length:
        padded.append(last)
    return padded


def load_episode_info(info_json_path: Path) -> List[dict]:
    with open(info_json_path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {info_json_path}, found: {type(data)}")
    return data


def load_episode_status(status_json_path: Path) -> Optional[dict]:
    try:
        with open(status_json_path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def action_field_order(preferred_field: Optional[str]) -> List[str]:
    """Return action label preference.

    This is the default order used for Habitat, legacy rows, and current
    UnrealZoo rows. Pass ``--action_field commanded_base_velocity`` explicitly
    when that command label should be preferred.
    """
    preferred = (preferred_field or ACTION_FIELD_AUTO).strip()
    if not preferred or preferred == ACTION_FIELD_AUTO:
        return ["base_velocity", "commanded_base_velocity"]
    order = [preferred]
    for key in ("commanded_base_velocity", "base_velocity"):
        if key not in order:
            order.append(key)
    return order


def extract_action3_from_step(step: Dict[str, Any], preferred_field: Optional[str]) -> List[float]:
    """Extract [forward, lateral, yaw_rate] from one per-step record."""
    vals: Any = None
    if (preferred_field or ACTION_FIELD_AUTO).strip() == ACTION_FIELD_AUTO and (
        step.get("command_label_source") or step.get("env_action") is not None
    ):
        field_order = ["base_velocity", "commanded_base_velocity"]
    else:
        field_order = action_field_order(preferred_field)
    for key in field_order:
        candidate = step.get(key)
        if isinstance(candidate, list) and len(candidate) >= 3:
            vals = candidate
            break
    if not isinstance(vals, list) or len(vals) < 3:
        vals = [0.0, 0.0, 0.0]
    return [float(vals[0]), float(vals[1]), float(vals[2])]


def build_actions_from_info(steps: List[dict], preferred_field: Optional[str] = ACTION_FIELD_AUTO) -> List[List[float]]:
    """Extract action labels as [forward, lateral, yaw_rate] per step."""
    actions: List[List[float]] = []
    for step in steps:
        actions.append(extract_action3_from_step(step, preferred_field))
    return actions


def integrate_future_trajectory(
    actions: List[List[float]], start_index: int, horizon: int, dt: float = 1.0
) -> List[List[float]]:
    """Integrate future base velocities into a local trajectory starting at [0,0,0].

    Returns a list of [x, y, theta], including the origin as the first point.
    Uses actions[start_index : start_index + horizon + 1] (clamped to available actions).
    """
    x, y, theta = 0.0, 0.0, 0.0
    trajectory: List[List[float]] = []
    if horizon <= 0 or start_index >= len(actions):
        return trajectory
    end_index = min(len(actions) - 1, start_index + horizon)
    for k in range(start_index, end_index + 1):
        vx, vy, wz = actions[k]
        # Rotate body-frame linear velocities into the initial local frame by current heading
        dx_global = vx * math.cos(theta) - vy * math.sin(theta)
        dy_global = vx * math.sin(theta) + vy * math.cos(theta)
        # accumulate displacement in local frame
        x += dx_global * dt
        y += dy_global * dt
        theta += wz * dt
        trajectory.append([x, y, theta])
    return trajectory


def build_indicator_curve(
    actions: List[List[float]], start_index: int, horizon: int, dt: float
) -> List[List[float]]:
    """Return a short local curve anchored at origin as [[x,y], ...].

    Integrates actions from start_index for `horizon` steps (clamped),
    accumulating displacement in the local robot frame.
    """
    x, y, theta = 0.0, 0.0, 0.0
    curve_xy: List[List[float]] = []
    if horizon <= 0 or start_index >= len(actions):
        return curve_xy
    end_index = min(len(actions) - 1, start_index + horizon)
    # Always include origin as the first point for stability
    curve_xy.append([0.0, 0.0])
    for k in range(start_index, end_index + 1):
        vx, vy, wz = actions[k]
        dx_global = vx * math.cos(theta) - vy * math.sin(theta)
        dy_global = vx * math.sin(theta) + vy * math.cos(theta)
        x += dx_global * dt
        y += dy_global * dt
        theta += wz * dt
        curve_xy.append([x, y])
    return curve_xy


def make_episode_json(
    rel_frame_paths: List[str],
    actions_7d: List[List[float]],
    episode_id: str,
    instruction: str,
) -> dict:
    return {
        "episode_id": episode_id,
        "frames": rel_frame_paths,
        "actions": actions_7d,
        "instruction": instruction,
    }


def collect_episode_pairs(input_root: Path) -> List[EpisodePaths]:
    """Find (<k>.mp4, <k>_info.json) pairs under input_root."""
    episodes: List[EpisodePaths] = []

    for seed_dir in sorted(input_root.glob("seed_*")):
        if not seed_dir.is_dir():
            continue
        for run_dir in sorted(seed_dir.iterdir()):
            if not run_dir.is_dir():
                continue

            # Find all *_info.json files inside run_dir
            for info_json in sorted(run_dir.glob("*_info.json")):
                stem = info_json.name[:-10]  # remove _info.json
                mp4 = (run_dir / f"{stem}.mp4")
                episodes.append(
                    EpisodePaths(
                        seed_dir=seed_dir,
                        run_dir=run_dir,
                        stem=stem,
                        mp4=mp4 if mp4.exists() else None,
                        info_json=info_json,
                    )
                )
    return episodes


def should_keep_episode(run_dir: Path, stem: str, only_success: bool,
                        min_following_rate: float = 0.0,
                        min_total_steps: int = 0,
                        exclude_collision: bool = False,
                        agent: str = "robotdog") -> bool:
    """
    判断是否保留该 episode

    Args:
        run_dir: episode 所在目录
        stem: episode 文件名前缀
        only_success: 是否只保留成功的 episode
        min_following_rate: 最小跟踪率阈值 (0-1)
        min_total_steps: 最小 episode 步数
        exclude_collision: 是否排除碰撞的 episode

    Returns:
        是否保留该 episode
    """
    status_path = run_dir / f"{stem}.json"
    status = load_episode_status(status_path)

    # 如果没有状态文件，根据 only_success 决定
    if not status:
        return not only_success

    # 检查碰撞
    if exclude_collision:
        collision = status.get("collision", 0)
        if isinstance(collision, (int, float)) and collision > 0:
            return False
        status_str = str(status.get("status", "")).lower()
        if "collision" in status_str:
            return False

    # 检查跟踪率
    if agent == "drone":
        following_rate = status.get("drone_following_rate", status.get("following_rate", 0))
    elif agent == "robotdog":
        following_rate = status.get("robotdog_following_rate", status.get("following_rate", 0))
    else:
        following_rate = status.get(
            "following_rate",
            status.get("robotdog_following_rate", status.get("drone_following_rate", 0)),
        )
    if isinstance(following_rate, (int, float)) and following_rate < min_following_rate:
        return False

    # 检查 episode 长度
    total_step = status.get("total_step", 0)
    if min_total_steps > 0:
        if not isinstance(total_step, (int, float)) or int(total_step) < min_total_steps:
            return False

    # 如果不需要只保留成功的，到这里就通过了
    if not only_success:
        return True

    # 检查成功条件（更严格）
    success_val = status.get("success")
    finish = status.get("finish")
    status_str = str(status.get("status", "")).lower()

    # 成功条件：
    # 1. success > 0
    # 2. 或者 status 包含 "success"
    # 3. 或者 finish=True 且 following_rate >= 0.5
    is_success = (isinstance(success_val, (int, float)) and success_val > 0) or ("success" in status_str)

    if is_success:
        return True

    # 对于 finish=True 但不是明确 success 的，需要检查 following_rate
    if finish:
        return isinstance(following_rate, (int, float)) and following_rate >= 0.5

    return False


def infer_multi_agent_rel_run_dir(input_root: Path, run_dir: Path) -> Path:
    """把原始 run_dir 映射成输出数据里的相对目录。

    数据流示例：
    /data/.../sim_data/unrealzoo_x/seed_100/SceneA
    -> seed_100/SceneA

    这样 frames/ 和 jsonl/ 都会镜像原始采集目录结构，方便回查。
    """
    try:
        rel = run_dir.resolve().relative_to(input_root.resolve())
        if str(rel) != ".":
            return rel
    except ValueError:
        pass
    if run_dir.parent.name.startswith("seed_"):
        return Path(run_dir.parent.name) / run_dir.name
    return Path(run_dir.name)


def collect_multi_agent_paired_episodes(input_root: Path) -> List[PairedEpisode]:
    """递归查找无人机/机器狗成对 episode。

    输入目录中期望存在：
    - <id>_drone.mp4
    - <id>_drone_info.json
    - <id>_robotdog.mp4
    - <id>_robotdog_info.json

    输出只保留四个文件都存在的 episode，避免后续抽帧或训练读取时才报错。
    """
    episodes: List[PairedEpisode] = []
    for drone_info in sorted(input_root.rglob("*_drone_info.json")):
        run_dir = drone_info.parent
        stem = drone_info.name[: -len("_drone_info.json")]
        robotdog_info = run_dir / f"{stem}_robotdog_info.json"
        drone_mp4 = run_dir / f"{stem}_drone.mp4"
        robotdog_mp4 = run_dir / f"{stem}_robotdog.mp4"
        if not robotdog_info.exists() or not drone_mp4.exists() or not robotdog_mp4.exists():
            continue
        status_json = run_dir / f"{stem}.json"
        episodes.append(
            PairedEpisode(
                input_root=input_root,
                run_dir=run_dir,
                rel_run_dir=infer_multi_agent_rel_run_dir(input_root, run_dir),
                stem=stem,
                drone_mp4=drone_mp4,
                robotdog_mp4=robotdog_mp4,
                drone_info_json=drone_info,
                robotdog_info_json=robotdog_info,
                status_json=status_json if status_json.exists() else None,
            )
        )
    return episodes


def should_keep_multi_agent_episode(
    ep: PairedEpisode,
    only_success: bool,
    min_agent_following_rate: float,
    min_total_steps: int,
    exclude_collision: bool,
) -> bool:
    """根据 episode 级状态文件过滤双 Agent 数据。

    处理逻辑：
    1. 无状态文件时，只有 only_success=False 才保留。
    2. exclude_collision=True 时排除 episode 级碰撞。
    3. min_agent_following_rate 会分别检查 drone/robotdog 的跟踪率。
    4. only_success=True 时要求 success/status/finish 至少一个成功信号成立。
    """
    if ep.status_json is None:
        return not only_success
    status = load_episode_status(ep.status_json)
    if not isinstance(status, dict):
        return not only_success

    if exclude_collision:
        collision = status.get("collision", 0)
        if isinstance(collision, (int, float)) and collision > 0:
            return False
        if "collision" in str(status.get("status", "")).lower():
            return False

    if min_total_steps > 0:
        total_step = status.get("total_step", 0)
        if not isinstance(total_step, (int, float)) or int(total_step) < min_total_steps:
            return False

    if min_agent_following_rate > 0:
        drone_rate = status.get("drone_following_rate", status.get("following_rate", 0.0))
        dog_rate = status.get("robotdog_following_rate", status.get("following_rate", 0.0))
        if float(drone_rate or 0.0) < min_agent_following_rate:
            return False
        if float(dog_rate or 0.0) < min_agent_following_rate:
            return False
        # centered/distance rates are retained as diagnostics and are not
        # downstream hard filters.

    if not only_success:
        return True

    success_val = status.get("success", 0)
    finish = bool(status.get("finish", False))
    status_str = str(status.get("status", "")).lower()
    return (isinstance(success_val, (int, float)) and success_val > 0) or ("success" in status_str) or finish


def extract_multi_agent_frames_ffmpeg(
    ffmpeg_path: str,
    mp4_path: Path,
    out_dir: Path,
    quality: int,
    reuse_existing: bool,
    expected_count: Optional[int] = None,
) -> List[Path]:
    """为单个 Agent 视频抽帧。

    数据流：
    mp4 -> frames/<seed>/<scene>/<episode>/<agent>/frame_00001.jpg

    reuse_existing=True 时，如果输出目录已有图片，会直接复用，避免反复跑 ffmpeg。
    """
    ensure_dir(out_dir)
    existing = list_sorted_images(out_dir)
    if reuse_existing and existing:
        count_matches = expected_count is None or len(existing) == int(expected_count)
        source_mtime = mp4_path.stat().st_mtime
        frames_are_current = min(path.stat().st_mtime for path in existing) >= source_mtime
        if count_matches and frames_are_current:
            return existing
    return extract_frames_ffmpeg(ffmpeg_path, mp4_path, out_dir, quality=quality)


def to_multi_agent_action3(step: Dict[str, Any], agent_name: str, preferred_field: str) -> List[float]:
    """把 UnrealZoo info 中的动作字段统一成 [vx, vy, yaw_rate]。

    处理逻辑：
    - 默认 auto 优先使用 base_velocity。
    - 如果需要当前 UnrealZoo 的控制器命令标签，显式传 --action_field commanded_base_velocity。
    - drone_action 常见为 4 维，此时第 4 维作为 yaw_rate。
    - 缺失或格式异常时补 0，保证后续积分稳定。
    """
    if (preferred_field or ACTION_FIELD_AUTO).strip() == ACTION_FIELD_AUTO and (
        step.get("command_label_source") or step.get("env_action") is not None
    ):
        field_order = ["base_velocity", "commanded_base_velocity"]
    else:
        field_order = action_field_order(preferred_field)
    if agent_name == "drone":
        field_order.extend(["drone_action"])
    elif agent_name == "robotdog":
        field_order.extend(["ground_action"])

    vals: Any = None
    chosen_key = ""
    for key in field_order:
        vals = step.get(key)
        if isinstance(vals, list) and vals:
            chosen_key = key
            break

    if not isinstance(vals, list):
        vals = [0.0, 0.0, 0.0]
    if agent_name == "drone" and chosen_key == "drone_action" and len(vals) >= 4:
        vals = [vals[0], vals[1], vals[3]]

    out = []
    for i in range(3):
        try:
            out.append(float(vals[i]))
        except Exception:
            out.append(0.0)
    return out


def build_multi_agent_actions(steps: List[Dict[str, Any]], agent_name: str, preferred_field: str) -> List[List[float]]:
    """从 info.json 的逐步记录中提取动作序列。"""
    return [to_multi_agent_action3(step, agent_name, preferred_field) for step in steps]


def multi_agent_step_dt(step: Dict[str, Any], fallback_dt: float) -> float:
    """Return the simulation duration represented by one recorded row."""
    for key in ("effective_dt_s", "base_velocity_dt_s", "fixed_timestep_seconds", "training_dt_s", "dt"):
        try:
            value = float(step.get(key))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0.0:
            return value
    return float(fallback_dt)


def integrate_multi_agent_actions(
    actions: List[List[float]],
    start_index: int,
    horizon_steps: int,
    dt: float,
    step_dts: Optional[List[float]] = None,
) -> List[List[float]]:
    """把未来速度积分成局部路点标签。

    输入：
    - actions[k] = [vx, vy, yaw_rate]，位于 Agent 自身局部坐标系。

    输出：
    - points[0] = [0, 0, 0]，表示当前帧的局部原点。
    - points[k] 表示执行未来 k 个动作后的局部位姿，因此 horizon_steps=9
      时正常返回 10 个点，对应 0.0s 到 0.9s。
    """
    x, y, theta = 0.0, 0.0, 0.0
    points: List[List[float]] = [[0.0, 0.0, 0.0]]
    end = min(len(actions), start_index + max(0, horizon_steps))
    for k in range(start_index, end):
        vx, vy, wz = actions[k]
        step_dt = float(step_dts[k]) if step_dts is not None and k < len(step_dts) else float(dt)
        dx = vx * math.cos(theta) - vy * math.sin(theta)
        dy = vx * math.sin(theta) + vy * math.cos(theta)
        x += dx * step_dt
        y += dy * step_dt
        theta += wz * step_dt
        points.append([x, y, theta])
    return points


def advance_local_pose(
    pose: List[float],
    action: List[float],
    duration_s: float,
) -> List[float]:
    """Advance a local SE(2) pose with a body-frame velocity command."""
    x, y, theta = [float(value) for value in pose[:3]]
    vx, vy, wz = [float(value) for value in action[:3]]
    duration_s = float(duration_s)
    dx = vx * math.cos(theta) - vy * math.sin(theta)
    dy = vx * math.sin(theta) + vy * math.cos(theta)
    return [
        x + dx * duration_s,
        y + dy * duration_s,
        theta + wz * duration_s,
    ]


def integrate_actions_at_fixed_times(
    actions: List[List[float]],
    step_dts: List[float],
    start_index: int,
    output_dt: float,
    n_waypoints: int,
) -> Tuple[List[List[float]], List[int]]:
    """Integrate variable-duration actions and sample at fixed future times.

    Output waypoint ``i`` always represents ``i * output_dt`` seconds from the
    current observation.  The returned source indices identify every recorded
    action interval consumed to cover the fixed-time horizon.
    """
    if n_waypoints <= 0:
        return [], []
    if output_dt <= 0.0:
        raise ValueError(f"output_dt must be positive, got {output_dt}")

    points: List[List[float]] = [[0.0, 0.0, 0.0]]
    source_indices: List[int] = []
    state = [0.0, 0.0, 0.0]
    elapsed = 0.0
    next_output_index = 1
    index = int(start_index)
    epsilon = 1e-9

    while next_output_index < n_waypoints and index < len(actions) and index < len(step_dts):
        duration = float(step_dts[index])
        if not math.isfinite(duration) or duration <= 0.0:
            break
        action = actions[index]
        segment_start_state = state
        segment_end = elapsed + duration
        source_indices.append(index)

        while next_output_index < n_waypoints:
            target_time = next_output_index * float(output_dt)
            if target_time > segment_end + epsilon:
                break
            points.append(
                advance_local_pose(
                    segment_start_state,
                    action,
                    max(0.0, target_time - elapsed),
                )
            )
            next_output_index += 1

        state = advance_local_pose(segment_start_state, action, duration)
        elapsed = segment_end
        index += 1

    return points, source_indices


def build_observation_times(step_dts: List[float]) -> List[float]:
    """Return observation times where row ``i`` starts at ``times[i]``."""
    times = [0.0]
    for duration in step_dts[:-1]:
        value = float(duration)
        if not math.isfinite(value) or value <= 0.0:
            value = 0.0
        times.append(times[-1] + value)
    return times


def fixed_time_history_indices(
    observation_times: List[float],
    current_index: int,
    history: int,
    frame_dt: float,
) -> List[int]:
    """Select previous observations nearest to fixed past-time offsets."""
    if history <= 0 or current_index <= 0:
        return []
    if frame_dt <= 0.0:
        raise ValueError(f"history frame dt must be positive, got {frame_dt}")
    current_time = float(observation_times[current_index])
    available = observation_times[:current_index]
    indices: List[int] = []
    for slot in range(history, 0, -1):
        target_time = current_time - slot * float(frame_dt)
        right = bisect.bisect_left(available, target_time)
        if right <= 0:
            chosen = 0
        elif right >= len(available):
            chosen = len(available) - 1
        else:
            left = right - 1
            chosen = left if target_time - available[left] <= available[right] - target_time else right
        indices.append(chosen)
    return indices


def pose_xy_distance_m(first: Any, second: Any) -> Optional[float]:
    if not isinstance(first, list) or not isinstance(second, list) or len(first) < 2 or len(second) < 2:
        return None
    try:
        dx = float(first[0]) - float(second[0])
        dy = float(first[1]) - float(second[1])
    except (TypeError, ValueError):
        return None
    return math.hypot(dx, dy) / 100.0


def wrap_radians(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def recorded_pose_waypoints(
    steps: List[Dict[str, Any]],
    agent_name: str,
    start_index: int,
    n_waypoints: int,
) -> Tuple[List[List[float]], List[bool], List[int]]:
    """Build local waypoints from consecutive recorded poses without interpolation.

    Row ``start_index + k`` is declared to represent ``k * 0.1s`` by the
    dataset protocol. The source timestamps and recorded velocity fields are
    intentionally not used.
    """
    pose_key = f"{agent_name}_pose"
    if start_index >= len(steps) or n_waypoints <= 0:
        return [], [], []
    origin = steps[start_index].get(pose_key)
    if not isinstance(origin, list) or len(origin) < 5:
        return [], [], []
    try:
        origin_x = float(origin[0])
        origin_y = float(origin[1])
        origin_yaw = math.radians(float(origin[4]))
    except (TypeError, ValueError):
        return [], [], []

    waypoints: List[List[float]] = []
    valid_mask: List[bool] = []
    source_indices: List[int] = []
    last = [0.0, 0.0, 0.0]
    for offset in range(n_waypoints):
        index = start_index + offset
        if index >= len(steps):
            waypoints.append(last.copy())
            valid_mask.append(False)
            continue
        pose = steps[index].get(pose_key)
        if not isinstance(pose, list) or len(pose) < 5:
            waypoints.append(last.copy())
            valid_mask.append(False)
            continue
        try:
            dx = (float(pose[0]) - origin_x) / 100.0
            dy = (float(pose[1]) - origin_y) / 100.0
            yaw = math.radians(float(pose[4]))
        except (TypeError, ValueError):
            waypoints.append(last.copy())
            valid_mask.append(False)
            continue
        last = [
            float(math.cos(origin_yaw) * dx + math.sin(origin_yaw) * dy),
            float(-math.sin(origin_yaw) * dx + math.cos(origin_yaw) * dy),
            float(wrap_radians(yaw - origin_yaw)),
        ]
        waypoints.append(last.copy())
        # waypoint[0] is a structural origin and is excluded from prediction loss.
        valid_mask.append(offset > 0)
        source_indices.append(index)
    return waypoints, valid_mask, source_indices


def build_inverse_control_annotations(
    drone_steps: List[Dict[str, Any]],
    robotdog_steps: List[Dict[str, Any]],
    n_waypoints: int,
    cfg: InverseDynamicsConfig,
    *,
    drone_max_translation_command_mps: float,
    drone_max_yaw_command_radps: float,
    robotdog_max_speed_command_mps: float,
    robotdog_max_yaw_command_radps: float,
) -> List[Dict[str, Any]]:
    """Roll GT command state once per episode and annotate every source row."""
    count = min(len(drone_steps), len(robotdog_steps))
    state, reference = initial_inverse_state()
    annotations: List[Dict[str, Any]] = []
    limits = np.asarray(
        [
            [drone_max_translation_command_mps, drone_max_translation_command_mps, drone_max_yaw_command_radps],
            [robotdog_max_speed_command_mps, 0.0, robotdog_max_yaw_command_radps],
        ],
        dtype=np.float64,
    )
    for index in range(count):
        drone_wp, drone_valid, _ = recorded_pose_waypoints(
            drone_steps, "drone", index, n_waypoints
        )
        dog_wp, dog_valid, _ = recorded_pose_waypoints(
            robotdog_steps, "robotdog", index, n_waypoints
        )
        state_before = state.copy()
        reference_before = reference.copy()
        result = inverse_step_numpy(
            np.asarray([drone_wp, dog_wp], dtype=np.float64),
            np.asarray([drone_valid, dog_valid], dtype=bool),
            state_before,
            reference_before,
            cfg,
        )
        control = result["control"]
        required_mask = result["valid_mask"].copy()
        finite_mask = np.isfinite(control)
        within_range = np.ones_like(required_mask)
        within_range[0] = np.abs(control[0]) <= limits[0]
        within_range[1, 0] = abs(float(control[1, 0])) <= limits[1, 0]
        within_range[1, 1] = False
        within_range[1, 2] = abs(float(control[1, 2])) <= limits[1, 2]
        control_mask = required_mask & finite_mask & within_range
        out_of_range = required_mask & ~within_range
        annotations.append(
            {
                "inverse_control_version": INVERSE_CONTROL_VERSION,
                "inverse_control_config": cfg.to_dict(),
                "inverse_control_state_before": state_before.tolist(),
                "inverse_control_reference_before": reference_before.tolist(),
                "inverse_control_target": control.tolist(),
                "inverse_env_action_target": result["env_action"].tolist(),
                "inverse_raw_desired_velocity": result["raw_desired_velocity"].tolist(),
                "inverse_desired_velocity": result["desired_velocity"].tolist(),
                "inverse_control_valid_mask": control_mask.tolist(),
                "inverse_control_out_of_range": out_of_range.tolist(),
            }
        )
        # Roll command-side state even when an extreme control target is masked.
        # This preserves temporal causality without reading realized UE state.
        state = result["state_after"]
        reference = result["reference_after"]
    return annotations


def multi_agent_quality_mask(
    steps: List[Dict[str, Any]],
    agent_name: str,
    fallback_dt: float,
    *,
    min_dt: float,
    max_dt: float,
    max_after_action_gap_m: float,
    exclude_snap_heading: bool,
) -> List[bool]:
    """Build a per-action validity mask for time-aligned supervision."""
    pose_key = f"{agent_name}_pose"
    after_key = f"{agent_name}_pose_after_action"
    result: List[bool] = []
    for index, step in enumerate(steps):
        dt = multi_agent_step_dt(step, fallback_dt)
        valid = math.isfinite(dt) and float(min_dt) <= dt <= float(max_dt)
        valid = valid and not bool_multi_agent_field(step, "collision", False)
        if exclude_snap_heading:
            valid = valid and step.get("snap_heading") is not True
        if max_after_action_gap_m >= 0.0 and index + 1 < len(steps):
            gap = pose_xy_distance_m(step.get(after_key), steps[index + 1].get(pose_key))
            valid = valid and gap is not None and gap <= float(max_after_action_gap_m)
        result.append(bool(valid))
    return result


def load_manifest_episode_keys(path: Optional[str], split: str) -> set[str]:
    if not path:
        return set()
    manifest_path = Path(path).expanduser().resolve()
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    items = payload.get(split, []) if isinstance(payload, dict) else []
    if not isinstance(items, list):
        raise ValueError(f"Manifest split {split!r} must be a list: {manifest_path}")
    keys = set()
    for item in items:
        if isinstance(item, dict) and isinstance(item.get("key"), str):
            keys.add(item["key"].strip("/"))
    return keys


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest_info_hashes(path: Optional[str], split: str) -> set[str]:
    """Hash manifest info files so flattened and raw episode ids can match."""
    if not path:
        return set()
    manifest_path = Path(path).expanduser().resolve()
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    items = payload.get(split, []) if isinstance(payload, dict) else []
    input_root_value = payload.get("input_root") if isinstance(payload, dict) else None
    if not isinstance(items, list) or not isinstance(input_root_value, str):
        raise ValueError(f"Manifest needs input_root and list split {split!r}: {manifest_path}")
    input_root = Path(input_root_value).expanduser().resolve()
    hashes: set[str] = set()
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("info"), str):
            continue
        info_path = input_root / item["info"]
        if not info_path.is_file():
            raise FileNotFoundError(f"Manifest info file does not exist: {info_path}")
        hashes.add(sha256_file(info_path))
    return hashes


def resample_multi_agent_waypoints(points: List[List[float]], n_waypoints: int) -> Tuple[List[List[float]], List[bool]]:
    """把变长未来轨迹重采样为固定 n_waypoints。

    输出：
    - waypoints: 固定长度路点；不足时复制最后一个有效点补齐。
    - valid_mask: 哪些位置来自真实未来轨迹，训练 loss 只在 True 上计算。
    """
    if n_waypoints <= 0:
        return [], []
    if not points:
        return [[0.0, 0.0, 0.0] for _ in range(n_waypoints)], [False for _ in range(n_waypoints)]
    if len(points) == n_waypoints:
        return points, [True for _ in range(n_waypoints)]
    if len(points) == 1:
        return [points[0] for _ in range(n_waypoints)], [i == 0 for i in range(n_waypoints)]

    idxs = []
    for i in range(n_waypoints):
        pos = round(i * (len(points) - 1) / max(1, n_waypoints - 1))
        idxs.append(int(pos))
    # Repeated indices are interpolation artifacts, not independent future
    # states. Mark only the first occurrence valid so they cannot receive
    # duplicate loss weight when a caller requests more output points than the
    # available trajectory contains.
    seen = set()
    valid = []
    for idx in idxs:
        valid.append(idx not in seen)
        seen.add(idx)
    return [points[i] for i in idxs], valid


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def normalize_multi_agent_bbox_xywh(raw_bbox: Any, width: int, height: int) -> List[float]:
    """把 UnrealZoo bbox 转为模型使用的 cxcywh_norm。

    输入通常是像素 xywh: [x, y, w, h]。
    输出是归一化中心框: [cx/W, cy/H, w/W, h/H]。

    如果输入已经是 0-1 范围，会按 xywh_norm 解释后转成中心点形式。
    """
    return normalize_bbox_xywh_to_cxcywh(raw_bbox, width, height)


def read_multi_agent_image_size(path: Path, fallback_width: int, fallback_height: int) -> Tuple[int, int]:
    """读取图片宽高，用于 bbox 像素坐标归一化。"""
    try:
        with Image.open(str(path)) as im:
            return im.size
    except Exception:
        return fallback_width, fallback_height


def bool_multi_agent_field(step: Dict[str, Any], key: str, default: bool = False) -> bool:
    val = step.get(key, default)
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    return default


def float_multi_agent_field(step: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(step.get(key, default))
    except Exception:
        return default


def raw_bbox_valid_xywh(value: Any) -> bool:
    """Return whether a raw per-frame GT box can supervise bbox-dependent losses."""
    try:
        if not isinstance(value, (list, tuple)) or len(value) < 4:
            return False
        x, y, width, height = (float(item) for item in value[:4])
        return all(math.isfinite(item) for item in (x, y, width, height)) and width > 0.0 and height > 0.0
    except (TypeError, ValueError, OverflowError):
        return False


def make_multi_agent_rel_frame_paths(output_root: Path, frame_paths: Iterable[Path]) -> List[str]:
    """把绝对帧路径转为 JSON 中保存的相对路径。

    训练 Dataset 后续会用 data_root + rel_path 找图片，并用同样 rel_path 映射 vision_cache。
    """
    rels: List[str] = []
    for path in frame_paths:
        try:
            # Preserve the logical path through a namespaced frame symlink.
            # Resolving first would collapse data7_7 and data7_8_new episodes
            # with equal scene/id names onto the same cache key.
            rel = path.absolute().relative_to(output_root.absolute())
        except ValueError:
            try:
                rel = path.resolve().relative_to(output_root.resolve())
            except ValueError:
                rel = path
        rels.append(rel.as_posix())
    return rels


def load_multi_agent_instruction(ep: PairedEpisode, override: Optional[str]) -> str:
    """读取训练指令文本。

    优先级：
    CLI --instruction > episode status.json 中的 instruction > 默认跟踪指令。
    """
    if override and override.strip():
        return override.strip()
    if ep.status_json is not None:
        status = load_episode_status(ep.status_json)
        instr = status.get("instruction") if isinstance(status, dict) else None
        if isinstance(instr, str) and instr.strip():
            return instr.strip()
    return DEFAULT_MULTI_AGENT_INSTRUCTION


def build_multi_agent_payload(
    agent_name: str,
    images_window: List[str],
    current_frame: str,
    step: Dict[str, Any],
    waypoints: List[List[float]],
    valid_mask: List[bool],
    image_width: int,
    image_height: int,
) -> Dict[str, Any]:
    """构造 JSON 中单个 Agent 的结构化字段。

    这些字段同时服务两件事：
    - 训练：只读取 images/current/waypoints/valid_mask。
    - 调试：保留 visibility、distance、pose 等原始采集信息，方便回查样本质量。
    """
    pose_key = f"{agent_name}_pose"
    raw_bbox = step.get("target_bbox", [0, 0, 0, 0])
    bbox = normalize_bbox_xywh_to_cxcywh(raw_bbox, image_width, image_height)
    return {
        "name": agent_name,
        "images": images_window,
        "current": current_frame,
        "target_visible": bool_multi_agent_field(step, "target_visible", False),
        "bbox": bbox,
        "bbox_format": "cxcywh_norm",
        "bbox_raw_xywh": raw_bbox,
        "bbox_valid_mask": raw_bbox_valid_xywh(raw_bbox),
        "target_visibility": float_multi_agent_field(step, "target_visibility", 0.0),
        "target_center_error": float_multi_agent_field(step, "target_center_error", 0.0),
        "target_centered": bool_multi_agent_field(step, "target_centered", False),
        "target_distance": float_multi_agent_field(step, "dis_to_human", 0.0),
        "collision": bool_multi_agent_field(step, "collision", False),
        "waypoints": waypoints,
        "trajectory": waypoints,
        "valid_mask": valid_mask,
        "pose": step.get(pose_key),
        "target_pose": step.get("target_pose"),
    }


def build_multi_agent_samples_for_episode(
    ep: PairedEpisode,
    args: argparse.Namespace,
    output_root: Path,
    drone_frames: List[Path],
    robotdog_frames: List[Path],
    drone_steps: List[Dict[str, Any]],
    robotdog_steps: List[Dict[str, Any]],
    output_rel_run_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """把一个双 Agent episode 转成多条滑动窗口训练样本。

    第 j 条样本的数据流：
    - 历史帧: [j-history, j)，分别来自无人机和机器狗两个视频。
    - 当前帧: j，用于 fine token 和 bbox。
    - 标签: 直接读取原始 JSON 中未来记录 pose 并转成局部绝对 waypoint；不积分速度。
    - 输出: 同时写 agents 结构化字段和 agent1_/agent2_ 扁平字段。
    """
    agent_order = [args.agent1, args.agent2]
    if sorted(agent_order) != ["drone", "robotdog"]:
        raise ValueError("--agent1/--agent2 must be drone and robotdog in either order")

    frame_map = {"drone": drone_frames, "robotdog": robotdog_frames}
    step_map = {"drone": drone_steps, "robotdog": robotdog_steps}
    waypoint_label_source = str(getattr(args, "waypoint_label_source", "recorded_pose_fixed_dt"))
    if waypoint_label_source != "recorded_pose_fixed_dt":
        raise ValueError(
            "The global-image base dataset only supports waypoint_label_source="
            "recorded_pose_fixed_dt; velocity integration is intentionally disabled."
        )
    history_frame_dt = float(args.dt)
    filter_quality = bool(getattr(args, "filter_time_aligned_quality", False))
    quality_masks = {
        agent_name: multi_agent_quality_mask(
            step_map[agent_name],
            agent_name,
            args.dt,
            min_dt=float(getattr(args, "min_effective_dt", 0.0)),
            max_dt=float(getattr(args, "max_effective_dt", float("inf"))),
            max_after_action_gap_m=float(getattr(args, "max_after_action_gap_m", -1.0)),
            exclude_snap_heading=bool(getattr(args, "exclude_snap_heading", False)),
        )
        for agent_name in ("drone", "robotdog")
    }

    max_len = min(len(drone_frames), len(robotdog_frames), len(drone_steps), len(robotdog_steps))
    if max_len <= 0:
        return []

    rel_frames = {
        "drone": make_multi_agent_rel_frame_paths(output_root, drone_frames[:max_len]),
        "robotdog": make_multi_agent_rel_frame_paths(output_root, robotdog_frames[:max_len]),
    }
    image_sizes = {
        agent_name: read_multi_agent_image_size(frame_map[agent_name][0], 640, 480)
        for agent_name in ("drone", "robotdog")
    }

    instruction = load_multi_agent_instruction(ep, args.instruction)
    history = max(0, int(args.history))
    horizon = max(1, int(args.horizon))
    n_waypoints = max(1, int(args.n_waypoints))
    dt = float(args.dt)
    if abs(dt - 0.1) > 1e-8:
        raise ValueError("recorded_pose_fixed_dt requires --dt 0.1")
    samples: List[Dict[str, Any]] = []

    for j in range(max_len):
        history_indices = list(range(max(0, j - history), j))

        agents: Dict[str, Dict[str, Any]] = {}
        skip = False
        for agent_name in ("drone", "robotdog"):
            steps = step_map[agent_name]
            frame_rels = rel_frames[agent_name]
            step = steps[j]

            if args.skip_collision_steps and bool_multi_agent_field(step, "collision", False):
                skip = True
                break
            if args.require_visible and not bool_multi_agent_field(step, "target_visible", False):
                skip = True
                break
            if float_multi_agent_field(step, "target_visibility", 0.0) < args.min_target_visibility:
                skip = True
                break

            images_window = [frame_rels[index] for index in history_indices]
            current_frame = frame_rels[j]
            waypoints, valid_mask, source_action_indices = recorded_pose_waypoints(
                steps,
                agent_name,
                j,
                n_waypoints,
            )
            future_dts = [dt for _ in source_action_indices]
            prediction_mask = valid_mask[1:]
            if (not args.allow_partial_horizon) and not all(prediction_mask):
                skip = True
                break
            if (not args.allow_partial_horizon) and len(waypoints) != n_waypoints:
                skip = True
                break

            if filter_quality:
                quality_indices = list(source_action_indices)
                if not quality_indices:
                    skip = True
                    break
                if bool(getattr(args, "quality_filter_history", False)) and history_indices:
                    quality_indices = list(range(history_indices[0], source_action_indices[-1] + 1))
                mask = quality_masks[agent_name]
                if any(index >= len(mask) or not mask[index] for index in quality_indices):
                    skip = True
                    break

            agents[agent_name] = build_multi_agent_payload(
                agent_name,
                images_window,
                current_frame,
                step,
                waypoints,
                valid_mask,
                image_sizes[agent_name][0],
                image_sizes[agent_name][1],
            )
            agents[agent_name]["action_dts"] = future_dts
            agents[agent_name]["source_action_indices"] = source_action_indices
            agents[agent_name]["history_source_indices"] = history_indices
            agents[agent_name]["waypoint_times_s"] = [index * dt for index in range(len(waypoints))]
            agents[agent_name]["waypoint_label_source"] = waypoint_label_source

        if skip:
            continue

        a1, a2 = agent_order
        sample_rel_run_dir = output_rel_run_dir or ep.rel_run_dir
        sample = {
            "schema_version": RECORDED_POSE_SCHEMA_VERSION,
            "episode_id": f"{sample_rel_run_dir.as_posix()}/{ep.stem}",
            "episode_stem": ep.stem,
            "rel_run_dir": sample_rel_run_dir.as_posix(),
            "step_index": j,
            "instruction": instruction,
            "dt": dt,
            "history": history,
            "horizon": horizon,
            "n_waypoints": n_waypoints,
            "time_alignment": "recorded_row_index_assumed_fixed_dt",
            "waypoint_label_source": waypoint_label_source,
            "history_frame_dt_s": history_frame_dt,
            "waypoint_dt_s": dt,
            "agent_order": agent_order,
            "agents": agents,
            "agent1_name": a1,
            "agent2_name": a2,
            "agent1_images": agents[a1]["images"],
            "agent1_current": agents[a1]["current"],
            "agent1_waypoints": agents[a1]["waypoints"],
            "agent1_valid_mask": agents[a1]["valid_mask"],
            "agent2_images": agents[a2]["images"],
            "agent2_current": agents[a2]["current"],
            "agent2_waypoints": agents[a2]["waypoints"],
            "agent2_valid_mask": agents[a2]["valid_mask"],
            "waypoints": [agents[a1]["waypoints"], agents[a2]["waypoints"]],
            "valid_mask": [agents[a1]["valid_mask"], agents[a2]["valid_mask"]],
            "bbox_feat": [agents[a1]["bbox"], agents[a2]["bbox"]],
            "bbox_valid_mask": [agents[a1]["bbox_valid_mask"], agents[a2]["bbox_valid_mask"]],
        }
        samples.append(sample)

    return samples


def write_multi_agent_jsonl(path: Path, samples: List[Dict[str, Any]]) -> None:
    """写出 per-episode JSONL，每行一条训练样本。"""
    ensure_dir(path.parent)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def parse_multi_agent_args() -> argparse.Namespace:
    """双 Agent 数据处理命令行参数。

    注意：原来的单 Agent 参数和 main() 保持不变；只有入口带 --multi_agent 时才使用这组参数。
    """
    parser = argparse.ArgumentParser(description="Build two-agent TrackVLA JSONL data from UnrealZoo aerial-ground episodes.")
    parser.add_argument("--input_root", type=str, required=True, help="Root containing *_drone_info.json and *_robotdog_info.json pairs.")
    parser.add_argument("--output_root", type=str, required=True, help="Output training data root.")
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="",
        help="Relative namespace below frames/ and jsonl/ for multi-source datasets.",
    )
    parser.add_argument("--history", type=int, default=31)
    parser.add_argument("--horizon", type=int, default=8, help="Metadata only; waypoint labels use the recorded-pose horizon.")
    parser.add_argument("--n_waypoints", type=int, default=8)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument(
        "--waypoint_label_source",
        choices=("recorded_pose_fixed_dt",),
        default="recorded_pose_fixed_dt",
        help=(
            "Use consecutive recorded agent poses directly as local absolute waypoints; "
            "velocity integration is disabled for the global-image base dataset."
        ),
    )
    parser.add_argument("--agent1", type=str, default="drone", choices=["drone", "robotdog"])
    parser.add_argument("--agent2", type=str, default="robotdog", choices=["drone", "robotdog"])
    parser.add_argument("--instruction", type=str, default=None)
    parser.add_argument("--only_success", action="store_true")
    parser.add_argument("--exclude_collision", action="store_true")
    parser.add_argument("--skip_collision_steps", action="store_true")
    parser.add_argument("--require_visible", action="store_true")
    parser.add_argument("--min_target_visibility", type=float, default=0.0)
    parser.add_argument("--min_agent_following_rate", type=float, default=0.8)
    parser.add_argument("--min_total_steps", type=int, default=0)
    parser.add_argument("--allow_partial_horizon", action="store_true")
    parser.add_argument("--ffmpeg_quality", type=int, default=2)
    parser.add_argument("--reuse_existing_frames", action="store_true", default=True)
    parser.add_argument("--no_reuse_existing_frames", dest="reuse_existing_frames", action="store_false")
    parser.add_argument("--fallback_width", type=int, default=640)
    parser.add_argument("--fallback_height", type=int, default=480)
    parser.add_argument("--out_file", type=str, default=None, help="Aggregated dataset JSON. Defaults to <output_root>/dataset.json.")
    parser.add_argument("--no_aggregate", action="store_true", help="Do not write aggregated dataset JSON.")
    parser.add_argument("--max_episodes", type=int, default=0)
    parser.add_argument("--exclude_manifest", type=str, default=None, help="Manifest whose selected split is excluded by episode key.")
    parser.add_argument("--exclude_manifest_split", type=str, default="test")
    parser.add_argument(
        "--exclude_content_manifest",
        type=str,
        default=None,
        help="Exclude episodes whose drone info SHA256 matches the selected manifest split.",
    )
    parser.add_argument("--fixed_time_resampling", action="store_true")
    parser.add_argument("--history_frame_dt", type=float, default=0.0)
    parser.add_argument("--filter_time_aligned_quality", action="store_true")
    parser.add_argument("--quality_filter_history", action="store_true")
    parser.add_argument("--min_effective_dt", type=float, default=0.09)
    parser.add_argument("--max_effective_dt", type=float, default=0.14)
    parser.add_argument("--max_after_action_gap_m", type=float, default=0.05)
    parser.add_argument("--exclude_snap_heading", action="store_true")
    parser.add_argument("--require_fixed_dt", action="store_true", help="Skip episodes containing rows outside dt tolerance.")
    parser.add_argument("--fixed_dt_tolerance", type=float, default=1e-4)
    parser.add_argument("--prune_stale_jsonl", action="store_true", help="Delete JSONL files not written by this run.")
    parser.add_argument("--dry_run", action="store_true", help="Only scan paired episodes; do not extract frames or write samples.")
    return parser.parse_args()


def main_multi_agent() -> None:
    """双 Agent 数据处理入口。

    总数据流：
    UnrealZoo 原始双视频/双 info
    -> ffmpeg 抽帧到 output_root/frames
    -> 读取动作、bbox、visibility
    -> 滑动窗口构造 agent1/agent2 样本
    -> 写 output_root/jsonl 和可选 dataset.json
    """
    args = parse_multi_agent_args()
    if args.agent1 == args.agent2:
        raise ValueError("--agent1 and --agent2 must be different")

    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_prefix = Path(args.output_prefix.strip()) if args.output_prefix.strip() else Path()
    if output_prefix.is_absolute() or ".." in output_prefix.parts:
        raise ValueError(f"--output_prefix must be a safe relative path: {args.output_prefix!r}")
    frames_root = output_root / "frames"
    jsonl_root = output_root / "jsonl"
    out_file = Path(args.out_file).resolve() if args.out_file else (output_root / "dataset.json")

    episodes = collect_multi_agent_paired_episodes(input_root)
    excluded_keys = load_manifest_episode_keys(args.exclude_manifest, args.exclude_manifest_split)
    if excluded_keys:
        episodes = [
            ep for ep in episodes
            if f"{ep.rel_run_dir.as_posix()}/{ep.stem}" not in excluded_keys
        ]
        print(f"Excluded manifest episodes: {len(excluded_keys)}")
    excluded_hashes = load_manifest_info_hashes(args.exclude_content_manifest, args.exclude_manifest_split)
    if excluded_hashes:
        before = len(episodes)
        episodes = [ep for ep in episodes if sha256_file(ep.drone_info_json) not in excluded_hashes]
        print(f"Excluded content-matched episodes: {before - len(episodes)} / {len(excluded_hashes)} hashes")
    if args.max_episodes > 0:
        episodes = episodes[: args.max_episodes]

    print(f"Found paired episodes: {len(episodes)}")
    if args.dry_run:
        for ep in episodes[:20]:
            print(f"  {ep.rel_run_dir.as_posix()}/{ep.stem}")
        return

    ffmpeg_path = find_ffmpeg_executable()
    if ffmpeg_path is None:
        raise RuntimeError("ffmpeg not found in PATH. Please install ffmpeg or add it to PATH.")

    ensure_dir(frames_root)
    ensure_dir(jsonl_root)
    if not args.no_aggregate:
        ensure_dir(out_file.parent)

    kept = 0
    written_jsonl = 0
    written_samples = 0
    skipped_status = 0
    skipped_load = 0
    skipped_empty = 0
    all_samples: List[Dict[str, Any]] = []
    written_paths: set[Path] = set()

    for ep in episodes:
        if not should_keep_multi_agent_episode(
            ep,
            only_success=args.only_success,
            min_agent_following_rate=args.min_agent_following_rate,
            min_total_steps=args.min_total_steps,
            exclude_collision=args.exclude_collision,
        ):
            skipped_status += 1
            continue
        kept += 1

        try:
            drone_steps = load_episode_info(ep.drone_info_json)
            robotdog_steps = load_episode_info(ep.robotdog_info_json)
        except Exception as exc:
            skipped_load += 1
            print(f"[WARN] failed to load info for {ep.rel_run_dir.as_posix()}/{ep.stem}: {exc}")
            continue

        if args.require_fixed_dt:
            expected_dt = float(args.dt)
            tolerance = max(0.0, float(args.fixed_dt_tolerance))
            all_step_dts = [
                multi_agent_step_dt(step, expected_dt)
                for step in (drone_steps + robotdog_steps)
            ]
            if any(abs(value - expected_dt) > tolerance for value in all_step_dts):
                skipped_empty += 1
                print(
                    f"[SKIP] {ep.rel_run_dir.as_posix()}/{ep.stem}: "
                    f"non-fixed dt outside {expected_dt:.6f}+/-{tolerance:.6f}s"
                )
                continue

        output_rel_run_dir = output_prefix / ep.rel_run_dir
        rel_episode_dir = output_rel_run_dir / ep.stem
        drone_frame_dir = frames_root / rel_episode_dir / "drone"
        robotdog_frame_dir = frames_root / rel_episode_dir / "robotdog"

        try:
            drone_frames = extract_multi_agent_frames_ffmpeg(
                ffmpeg_path,
                ep.drone_mp4,
                drone_frame_dir,
                args.ffmpeg_quality,
                args.reuse_existing_frames,
                expected_count=len(drone_steps),
            )
            robotdog_frames = extract_multi_agent_frames_ffmpeg(
                ffmpeg_path,
                ep.robotdog_mp4,
                robotdog_frame_dir,
                args.ffmpeg_quality,
                args.reuse_existing_frames,
                expected_count=len(robotdog_steps),
            )
        except subprocess.CalledProcessError as exc:
            skipped_load += 1
            print(f"[WARN] ffmpeg failed for {ep.rel_run_dir.as_posix()}/{ep.stem}: {exc}")
            continue

        samples = build_multi_agent_samples_for_episode(
            ep,
            args,
            output_root,
            drone_frames,
            robotdog_frames,
            drone_steps,
            robotdog_steps,
            output_rel_run_dir=output_rel_run_dir,
        )
        if not samples:
            skipped_empty += 1
            continue

        jsonl_path = jsonl_root / output_rel_run_dir / f"{ep.stem}.jsonl"
        write_multi_agent_jsonl(jsonl_path, samples)
        written_paths.add(jsonl_path.resolve())
        written_jsonl += 1
        written_samples += len(samples)
        if not args.no_aggregate:
            all_samples.extend(samples)
        print(f"[OK] {output_rel_run_dir.as_posix()}/{ep.stem}: samples={len(samples)}")

    if (not args.no_aggregate) and all_samples:
        with out_file.open("w", encoding="utf-8") as f:
            json.dump(all_samples, f, ensure_ascii=False)

    if args.prune_stale_jsonl and written_jsonl == 0:
        raise RuntimeError(
            "Refusing to prune JSONL because this preprocessing run wrote zero episodes. "
            "Check the fixed-dt filter and split manifest."
        )
    if args.prune_stale_jsonl:
        prune_root = jsonl_root / output_prefix
        stale_paths = [path for path in prune_root.rglob("*.jsonl") if path.resolve() not in written_paths]
        for path in stale_paths:
            path.unlink()
        print(f"Pruned stale JSONL files: {len(stale_paths)}")

    print(f"Kept episodes: {kept}")
    print(f"Written JSONL files: {written_jsonl}")
    print(f"Written samples: {written_samples}")
    print(f"Skipped by status: {skipped_status}")
    print(f"Skipped by load/extract: {skipped_load}")
    print(f"Skipped empty: {skipped_empty}")
    print(f"Output root: {output_root}")
    if (not args.no_aggregate) and all_samples:
        print(f"Aggregated dataset JSON: {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Make TrackVLA training data from mass_train outputs")
    parser.add_argument("--input_root", type=str, required=True, help="Path to mass_train root (e.g., exp_results/mass_train)")
    parser.add_argument("--output_root", type=str, required=True, help="Output root for training data (e.g., /data/hdt/ntv_data/data/track)")
    parser.add_argument("--max_frames", type=int, default=32, help="[Deprecated] Ignored. All frames will be used.")
    parser.add_argument("--only_success", action="store_true", help="Keep only successful episodes if status json exists")
    parser.add_argument("--agent", choices=["robotdog", "drone"], default="robotdog")
    parser.add_argument(
        "--min_following_rate",
        type=float,
        default=0.0,
        help="Minimum following rate threshold (0-1). Episodes below this are excluded. Recommended: 0.3-0.5",
    )
    parser.add_argument(
        "--exclude_collision",
        action="store_true",
        help="Exclude episodes with collision. Recommended for high-quality training data.",
    )
    parser.add_argument(
        "--min_total_steps",
        type=int,
        default=0,
        help="Minimum episode length. Episodes with status total_step below this are excluded.",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
        help=(
            "Instruction to use for all samples; if omitted, use per-episode status JSON "
            "when available, otherwise a sensible default."
        ),
    )
    parser.add_argument("--history", type=int, default=31, help="Number of previous frames for each sample window")
    parser.add_argument(
        "--out_file",
        type=str,
        default=None,
        help="Path to aggregated dataset JSON (default: <output_root>/dataset.json)",
    )
    parser.add_argument(
        "--no_aggregate",
        action="store_true",
        help="Write only per-episode JSONL shards and skip the large aggregated dataset.json.",
    )
    parser.add_argument("--horizon", type=int, default=8, help="Future action horizon to integrate for trajectory")
    parser.add_argument("--dt", type=float, default=0.1, help="Time step per action for integration")
    parser.add_argument(
        "--action_field",
        type=str,
        default=ACTION_FIELD_AUTO,
        help=(
            "Preferred action field. Auto uses base_velocity first; pass "
            "commanded_base_velocity explicitly to use current UnrealZoo controller "
            "command labels."
        ),
    )
    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    frames_root = output_root / "frames"
    ensure_dir(frames_root)
    jsonl_root = output_root / "jsonl"
    ensure_dir(jsonl_root)
    # Aggregated dataset will be written at the end
    out_file = Path(args.out_file).resolve() if args.out_file else (output_root / "dataset.json")
    ensure_dir(out_file.parent)

    ffmpeg_path = find_ffmpeg_executable()
    if ffmpeg_path is None:
        raise RuntimeError("ffmpeg not found in PATH. Please install ffmpeg or add it to PATH.")

    episodes = collect_episode_pairs(input_root)
    total = len(episodes)
    kept = 0
    num_samples = 0
    jsonl_files_written = 0
    episodes_no_samples = 0
    skipped_no_video = 0
    skipped_no_frames = 0
    all_samples: List[dict] = []

    for ep in episodes:
        kept += 1
        if not should_keep_episode(ep.run_dir, ep.stem, args.only_success,
                                   min_following_rate=args.min_following_rate,
                                   min_total_steps=args.min_total_steps,
                                   exclude_collision=args.exclude_collision,
                                   agent=args.agent):
            continue

        if ep.mp4 is None or not ep.mp4.exists():
            skipped_no_video += 1
            continue

        try:
            steps = load_episode_info(ep.info_json)
        except Exception as e:
            print(f"[WARN] Failed to load info for {ep.info_json}: {e}")
            continue

        # Determine instruction: if CLI provides one, always use it; otherwise try per-episode JSON, else fallback
        instruction_text = args.instruction.strip() if isinstance(args.instruction, str) and args.instruction.strip() else None
        status_path = ep.run_dir / f"{ep.stem}.json"
        if instruction_text is None:
            status = load_episode_status(status_path)
            if status:
                instr_candidate = status.get("instruction")
                if isinstance(instr_candidate, str) and instr_candidate.strip():
                    instruction_text = instr_candidate.strip()
        if instruction_text is None:
            instruction_text = "Follow the target person without collision."

        # Paths for frames extraction
        rel_frames_dir = Path(ep.seed_dir.name) / ep.run_dir.name / ep.stem
        abs_frames_dir = frames_root / rel_frames_dir
        try:
            frame_paths = extract_frames_ffmpeg(ffmpeg_path, ep.mp4, abs_frames_dir)
        except subprocess.CalledProcessError as e:
            print(f"[WARN] ffmpeg failed for {ep.mp4}: {e}")
            continue

        if not frame_paths:
            skipped_no_frames += 1
            continue

        # Use ALL frames; align actions length to number of frames
        desired_len = len(frame_paths)
        if desired_len == 0:
            skipped_no_frames += 1
            continue

        actions_full = build_actions_from_info(steps, args.action_field)
        if len(actions_full) >= desired_len:
            actions = actions_full[:desired_len]
        else:
            actions = pad_to_length(actions_full, desired_len)

        # Build relative paths for JSON for all frames (no skipping)
        rel_frame_paths = [str((Path("frames") / rel_frames_dir / p.name).as_posix()) for p in frame_paths]
        image_width, image_height = read_multi_agent_image_size(frame_paths[0], 640, 480)

        # Build sliding-window samples
        history = max(0, int(args.history))
        episode_samples: List[dict] = []
        if len(rel_frame_paths) > 0:
            for j in range(0, len(rel_frame_paths)):
                if history > 0:
                    start_idx = max(0, j - history)
                    images_window = rel_frame_paths[start_idx:j]
                else:
                    images_window = []
                current_frame = rel_frame_paths[j]
                # Compute future trajectory from j-th to j+horizon-th action (inclusive)
                horizon = int(args.horizon)
                dt = float(args.dt)
                # Require that we have full horizon in the ORIGINAL steps (no padding)
                if j + horizon > len(actions_full) - 1:
                    continue
                # Integrate and slice from original actions
                traj = integrate_future_trajectory(actions_full, start_index=j, horizon=horizon, dt=dt)
                # Build indicator curve (x,y only), same horizon and dt
                #indicator_xy = build_indicator_curve(actions_full, start_index=j, horizon=horizon, dt=dt)
                # Include the corresponding future actions in the sample (exact horizon+1 length)
                end_index = j + horizon
                future_actions = actions_full[j : end_index + 1]
                step_info = steps[j] if j < len(steps) else {}
                collision_flag = bool(step_info.get("collision", False))
                target_distance = step_info.get("target_distance", step_info.get("dis_to_human", 0.0))
                bbox = normalize_bbox_xywh_to_cxcywh(step_info.get("target_bbox"), image_width, image_height)
                prev_bbox = None
                if j > 0 and j - 1 < len(steps):
                    prev_step = steps[j - 1]
                    prev_bbox = normalize_bbox_xywh_to_cxcywh(prev_step.get("target_bbox"), image_width, image_height)
                spatial = bbox_spatial_fields(bbox, prev_bbox)
                sample = {
                    "images": images_window,
                    "current": current_frame,
                    "instruction": instruction_text,
                    "trajectory": traj,
                    "actions": future_actions,
                    "bbox": bbox,
                    "bbox_format": "cxcywh_norm",
                    "bbox_raw_xywh": step_info.get("target_bbox", [0, 0, 0, 0]),
                    "bbox_spatial": spatial,
                    "bbox_prompt_text": bbox_prompt_from_spatial([spatial], [args.agent]),
                    "collision": collision_flag,
                    "target_distance": float(target_distance) if target_distance is not None else 0.0,
                }
                episode_samples.append(sample)

        # Write per-episode JSONL and update aggregates
        if episode_samples:
            # Mirror input structure under jsonl root: <seed>/<run>/<stem>.jsonl
            rel_jsonl_dir = Path(ep.seed_dir.name) / ep.run_dir.name
            abs_jsonl_dir = jsonl_root / rel_jsonl_dir
            ensure_dir(abs_jsonl_dir)
            jsonl_path = abs_jsonl_dir / f"{ep.stem}.jsonl"
            with open(jsonl_path, "w") as f:
                for s in episode_samples:
                    f.write(json.dumps(s) + "\n")
            jsonl_files_written += 1
            if not args.no_aggregate:
                all_samples.extend(episode_samples)
            num_samples += len(episode_samples)
        else:
            episodes_no_samples += 1

    # Write aggregated dataset JSON (if any samples)
    if (not args.no_aggregate) and all_samples:
        with open(out_file, "w") as f:
            json.dump(all_samples, f)

    print(f"Found episodes: {total}")
    print(f"Written samples: {num_samples}")
    print(f"Per-episode JSONL files written: {jsonl_files_written}")
    if episodes_no_samples:
        print(f"Episodes with no samples: {episodes_no_samples}")
    print(f"Skipped (no video): {skipped_no_video}")
    print(f"Skipped (no frames): {skipped_no_frames}")
    if (not args.no_aggregate) and all_samples:
        print(f"Aggregated dataset file: {out_file}")
    print(f"Output dataset root: {output_root}")


if __name__ == "__main__":
    if "--multi_agent" in sys.argv:
        sys.argv.remove("--multi_agent")
        main_multi_agent()
    else:
        main()
