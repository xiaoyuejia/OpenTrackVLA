#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""UnrealZoo 双 Agent MLP / Anchor Diffusion 闭环评估入口。

整体功能：
- 创建无人机、机器狗与行人的 UnrealZoo 场景，逐帧运行双 Agent闭环控制。
- 可只重放离线采集的人体世界坐标轨迹，两个 Agent 与后续 RGB 仍由在线仿真闭环产生。
- 根据 checkpoint 内容自动选择 ``model.py`` MLP 或 Anchor Diffusion 模型。
- 支持无真值 bbox 的完整检测/跟踪评估，并统计距离、可见性、碰撞和 bbox IoU。
- 将预测局部轨迹绘制为平滑彩色曲线，保存逐 episode 视频、JSON 与汇总指标。

核心类：
- ``UnrealZooMultiAgentPlanner``：加载 checkpoint、编码双视角历史、预测 bbox 与轨迹并转为动作。

关键函数：
- ``load_recorded_target_trajectories``：从采集 JSON 读取仅用于驱动人的世界坐标轨迹。
- ``setup_episode``：随机初始化目标，或使用录制轨迹首帧初始化目标与两个 Agent。
- ``run_episode``：执行单条闭环评估 episode。
- ``_render_bgr_frame_with_traj``：在 RGB/BGR 帧上绘制平滑预测轨迹。
- ``write_episode_outputs``：保存视频和结构化结果。
- ``parse_args`` / ``main``：配置并启动多 episode 评估。

主要输入输出：
- 输入为 checkpoint 文件或目录、UnrealZoo 环境、episode 数、bbox 来源和控制参数。
- 输出位于 ``--save-path``，可由 ``python -m tools.calculate_unrealzoo_metrics`` 汇总。

Habitat / EVT-Bench 原始评估链路保持为：
``sh/eval.sh -> eval.py -> tools/trained_agent.py``。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.interpolate import CubicSpline

from tools.cache_gridpool import VisionCacheConfig, VisionFeatureCacher, grid_pool_tokens


REPO_ROOT = Path(__file__).resolve().parent
UNREALZOO_ROOT = REPO_ROOT / "unrealzoo-gym"
DATA_RECORDING_DIR = UNREALZOO_ROOT / "example" / "DataRecording"
for _path in (UNREALZOO_ROOT, DATA_RECORDING_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


# ----------------------- UnrealZoo 启动与环境注册 -----------------------

def _preparse_env_id(default: str = "UnrealTrack-DowntownWest-ContinuousColor-v0") -> str:
    """Read --env-id before importing gym_unrealcv so it can fast-register it."""
    for idx, arg in enumerate(sys.argv):
        if arg == "--env-id" and idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
        if arg.startswith("--env-id="):
            return arg.split("=", 1)[1]
    return default


_startup_env_id = _preparse_env_id()
os.environ.setdefault("UNREALZOO_FAST_ENV_ID", _startup_env_id)
print(f"[startup] importing UnrealZoo with fast env registration: {_startup_env_id}", flush=True)

# Importing gym_unrealcv registers UnrealZoo gym environment ids.
import gym  # noqa: E402
import gym_unrealcv  # noqa: F401,E402
print("[startup] UnrealZoo imports finished", flush=True)

from generate_aerial_ground_human_tracking_small import (  # noqa: E402
    DEFAULT_ENV_ID,
    DEFAULT_INSTRUCTION,
    classify_coop_agents,
    dog_args,
    drone_args,
    get_global_frame,
    make_env,
    place_initial_followers,
    reset_env,
)
from generate_drone_human_tracking_small import (  # noqa: E402
    EpisodeSkipped,
    UNREAL_UNITS_PER_METER,
    bbox_center_error,
    bbox_centered,
    choose_drone_camera_for_current_pose,
    collision_from_info as drone_collision_from_info,
    data_collection_step,
    distance_m,
    distance_xy_m,
    ensure_bgr_uint8,
    heading_deg,
    maybe_resample_target_goal,
    maybe_set_drone_yaw,
    pick_open_start,
    pick_reachable_goal,
    pose_xyz,
    safe_slug,
    save_mp4,
    set_drone_camera,
    target_mask_visibility,
    update_observation,
    write_json,
    yaw_deg,
)
from generate_robotdog_human_tracking_small import (  # noqa: E402
    choose_robotdog_camera_for_current_pose,
    collision_from_info as robotdog_collision_from_info,
    set_episode_appearances,
    set_ground_yaw,
    set_robotdog_camera,
)


# ----------------------- 录制人体轨迹读取与重放 -----------------------

def _parse_episode_filter(raw: Optional[str]) -> Optional[set[str]]:
    """解析逗号分隔的录制 episode 编号；为空时使用目录中的全部 episode。"""
    if raw is None or not raw.strip():
        return None
    return {item.strip() for item in raw.split(",") if item.strip()}


def load_recorded_target_trajectories(
    source: Path,
    episode_filter: Optional[set[str]] = None,
    env_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    """从原始 episode 或 2:1 split manifest 读取人的世界坐标轨迹。

    每帧读取 ``target_pose=[x, y, z, pitch, yaw, roll]``。如果源文件里包含
    drone/robotdog 的首帧位姿和相机字段，也会保存在 trajectory 元数据中，
    用于可选的 recorded-pose 初始化。

    当 ``source`` 是 ``split_manifest.json`` 时，只读取其中 ``test`` 清单，并按
    ``env_id`` 过滤场景，确保闭环仿真场景与录制人体世界坐标一致。
    """
    if source.is_file():
        with source.open("r", encoding="utf-8") as handle:
            source_obj = json.load(handle)
        if isinstance(source_obj, dict) and isinstance(source_obj.get("test"), list):
            input_root = Path(source_obj.get("input_root", source.parent)).expanduser()
            output_root = Path(source_obj.get("output_root", source.parent)).expanduser()
            candidates = []
            for item in source_obj["test"]:
                if env_id is not None and item.get("scene") != env_id:
                    continue
                if "key" in item:
                    key = str(item["key"])
                    path_candidates = [
                        source.parent / "test_raw" / f"{key}_drone_info.json",
                        output_root / "test_raw" / f"{key}_drone_info.json",
                        input_root / f"{key}_drone_info.json",
                    ]
                    path = next(
                        (candidate for candidate in path_candidates if candidate.is_file()),
                        path_candidates[0],
                    )
                    candidates.append((path, key))
                elif "info" in item:
                    info_rel = str(item["info"])
                    episode_name = str(item.get("relative_dir", Path(info_rel).parent)) + "/" + str(
                        item.get("stem", Path(info_rel).name.removesuffix("_info.json"))
                    )
                    path_candidates = [
                        source.parent / "test_raw" / info_rel,
                        output_root / "test_raw" / info_rel,
                        input_root / info_rel,
                    ]
                    path = next(
                        (candidate for candidate in path_candidates if candidate.is_file()),
                        path_candidates[0],
                    )
                    candidates.append((path, episode_name))
                else:
                    raise KeyError(f"split manifest item has neither key nor info: {item}")
        else:
            candidates = [(source, source.name.removesuffix("_drone_info.json"))]
    elif source.is_dir():
        candidates = [
            (path, str(path.relative_to(source)).removesuffix("_drone_info.json"))
            for path in sorted(source.rglob("*_drone_info.json"), key=lambda path: str(path))
            if env_id is None or path.parent.name == env_id
        ]
    else:
        raise FileNotFoundError(f"recorded target source does not exist: {source}")

    trajectories: list[dict[str, Any]] = []
    for path, episode_name in candidates:
        short_name = path.name.removesuffix("_drone_info.json")
        if episode_filter is not None and episode_name not in episode_filter and short_name not in episode_filter:
            continue
        if not path.is_file():
            raise FileNotFoundError(f"recorded target episode listed by {source} does not exist: {path}")
        with path.open("r", encoding="utf-8") as handle:
            records = json.load(handle)
        companion_path: Optional[Path] = None
        if path.name.endswith("_drone_info.json"):
            companion_path = path.with_name(
                path.name.removesuffix("_drone_info.json") + "_robotdog_info.json"
            )
        elif path.name.endswith("_robotdog_info.json"):
            companion_path = path.with_name(
                path.name.removesuffix("_robotdog_info.json") + "_drone_info.json"
            )
        companion_records: list[Any] = []
        if companion_path is not None and companion_path.is_file():
            with companion_path.open("r", encoding="utf-8") as handle:
                companion_records = json.load(handle)
        poses: list[list[float]] = []
        for frame_idx, record in enumerate(records):
            raw_pose = record.get("target_pose") if isinstance(record, dict) else None
            if not isinstance(raw_pose, (list, tuple)) or len(raw_pose) < 6:
                raise ValueError(f"{path}: frame {frame_idx} has invalid target_pose")
            poses.append([float(value) for value in raw_pose[:6]])
        if not poses:
            raise ValueError(f"{path}: no target_pose records found")
        first = dict(records[0]) if records and isinstance(records[0], dict) else {}
        companion_first = (
            companion_records[0]
            if companion_records and isinstance(companion_records[0], dict)
            else {}
        )
        # The collector writes role-specific state to separate files. Merge
        # non-null fields so one trajectory carries both follower start poses.
        for key, value in companion_first.items():
            if value is not None and first.get(key) is None:
                first[key] = value
        robotdog_pose = first.get("robotdog_pose")
        drone_pose = first.get("drone_pose")
        robotdog_camera = {
            "mount": first.get("robotdog_camera_mount"),
            "pitch": first.get("robotdog_camera_pitch"),
            "yaw_offset": first.get("robotdog_camera_yaw_offset"),
        }
        drone_camera = {
            "pitch": first.get("drone_camera_pitch"),
            "yaw_offset": first.get("drone_camera_yaw_offset"),
        }
        trajectories.append(
            {
                "episode_name": episode_name,
                "source": str(path.resolve()),
                "companion_source": (
                    str(companion_path.resolve())
                    if companion_path is not None and companion_path.is_file()
                    else None
                ),
                "poses": poses,
                "robotdog_pose": [float(value) for value in robotdog_pose[:6]]
                if isinstance(robotdog_pose, list) and len(robotdog_pose) >= 6
                else None,
                "robotdog_camera": robotdog_camera,
                "drone_pose": [float(value) for value in drone_pose[:6]]
                if isinstance(drone_pose, list) and len(drone_pose) >= 6
                else None,
                "drone_camera": drone_camera,
            }
        )
    if not trajectories:
        requested = "all" if episode_filter is None else ",".join(sorted(episode_filter))
        raise ValueError(f"no recorded target trajectories found in {source} (episodes={requested})")
    return trajectories


def _set_recorded_target_pose(env, setup: dict[str, Any], pose: list[float]) -> None:
    """将人精确放到录制位姿，并让导航目标停在当前位置以避免 AI 自主移动。"""
    env.unwrapped.unrealcv.set_obj_location(setup["target_name"], pose[:3])
    try:
        env.unwrapped.unrealcv.set_obj_rotation(setup["target_name"], pose[3:6])
    except Exception:
        pass
    try:
        env.unwrapped.unrealcv.nav_to_goal(setup["target_name"], pose[:3])
    except Exception:
        pass


def _restore_pose(env, obj_name: str, pose: Optional[list[float]], yaw_setter=None) -> bool:
    if not isinstance(pose, list) or len(pose) < 6:
        return False
    env.unwrapped.unrealcv.set_obj_location(obj_name, pose[:3])
    try:
        env.unwrapped.unrealcv.set_obj_rotation(obj_name, pose[3:6])
    except Exception:
        if yaw_setter is not None:
            try:
                yaw_setter(float(pose[4]))
            except Exception:
                pass
    return True


def restore_recorded_agent_initial_poses(
    env,
    args: argparse.Namespace,
    robotdog_name: str,
    drone_name: str,
    target_trajectory: Optional[dict[str, Any]],
) -> dict[str, bool]:
    """Restore both follower poses from recorded test data when available."""
    restored = {"robotdog": False, "drone": False}
    if target_trajectory is None or not bool(getattr(args, "init_from_recorded_agent_poses", False)):
        return restored

    restored["robotdog"] = _restore_pose(
        env,
        robotdog_name,
        target_trajectory.get("robotdog_pose"),
        yaw_setter=lambda yaw: set_ground_yaw(env, robotdog_name, yaw),
    )
    restored["drone"] = _restore_pose(
        env,
        drone_name,
        target_trajectory.get("drone_pose"),
        yaw_setter=lambda yaw: maybe_set_drone_yaw(env, drone_name, yaw),
    )
    return restored


def apply_recorded_agent_cameras(
    target_trajectory: Optional[dict[str, Any]],
    dog_mount: Any,
    dog_pitch: float,
    dog_yaw_offset: float,
    drone_pitch: float,
    drone_yaw_offset: float,
) -> tuple[Any, float, float, float, float, dict[str, bool]]:
    """Optionally replace searched cameras with recorded first-frame cameras."""
    restored = {"robotdog": False, "drone": False}
    if target_trajectory is None:
        return dog_mount, dog_pitch, dog_yaw_offset, drone_pitch, drone_yaw_offset, restored

    dog_camera = target_trajectory.get("robotdog_camera") or {}
    mount = dog_camera.get("mount")
    pitch = dog_camera.get("pitch")
    yaw_offset = dog_camera.get("yaw_offset")
    if isinstance(mount, list) and len(mount) >= 3:
        dog_mount = [float(value) for value in mount[:3]]
        restored["robotdog"] = True
    if isinstance(pitch, (int, float)):
        dog_pitch = float(pitch)
        restored["robotdog"] = True
    if isinstance(yaw_offset, (int, float)):
        dog_yaw_offset = float(yaw_offset)
        restored["robotdog"] = True

    drone_camera = target_trajectory.get("drone_camera") or {}
    pitch = drone_camera.get("pitch")
    yaw_offset = drone_camera.get("yaw_offset")
    if isinstance(pitch, (int, float)):
        drone_pitch = float(pitch)
        restored["drone"] = True
    if isinstance(yaw_offset, (int, float)):
        drone_yaw_offset = float(yaw_offset)
        restored["drone"] = True

    return dog_mount, dog_pitch, dog_yaw_offset, drone_pitch, drone_yaw_offset, restored


# ----------------------- checkpoint 与 bbox 工具 -----------------------

def _cleanup_state_dict_keys(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remove DDP ``module.`` prefixes if the checkpoint was trained with DDP."""
    if not isinstance(state, dict):
        return {}
    if any(k.startswith("module.") for k in state.keys()):
        return {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


def _latest_checkpoint(path: Path) -> Optional[Path]:
    if path.is_file():
        return path
    if not path.exists():
        return None
    candidates = sorted(path.glob("model_epoch*.pt"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def _normalize_bbox_xywh(raw_bbox: Any, width: int, height: int) -> list[float]:
    """Convert UnrealZoo pixel xywh bbox to normalized cxcywh."""
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) < 4:
        return [0.0, 0.0, 0.0, 0.0]
    try:
        x, y, w, h = [float(v) for v in raw_bbox[:4]]
    except Exception:
        return [0.0, 0.0, 0.0, 0.0]

    def c01(v: float) -> float:
        return max(0.0, min(1.0, float(v)))

    if max(abs(x), abs(y), abs(w), abs(h)) <= 1.5:
        return [c01(x + 0.5 * w), c01(y + 0.5 * h), c01(w), c01(h)]
    width = max(1, int(width))
    height = max(1, int(height))
    return [c01((x + 0.5 * w) / width), c01((y + 0.5 * h) / height), c01(w / width), c01(h / height)]


def _bbox_cxcywh_to_xywh(bbox: Any, width: int, height: int) -> list[int]:
    if not isinstance(bbox, (list, tuple, np.ndarray)) or len(bbox) < 4:
        return [0, 0, 0, 0]
    cx, cy, bw, bh = [float(v) for v in bbox[:4]]
    x = int(round((cx - 0.5 * bw) * width))
    y = int(round((cy - 0.5 * bh) * height))
    w = int(round(bw * width))
    h = int(round(bh * height))
    return [x, y, max(0, w), max(0, h)]


def _bbox_iou_cxcywh(pred: Any, target: Any) -> float:
    def corners(box: Any) -> tuple[float, float, float, float]:
        cx, cy, w, h = [float(v) for v in box[:4]]
        return cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h

    try:
        px1, py1, px2, py2 = corners(pred)
        tx1, ty1, tx2, ty2 = corners(target)
    except Exception:
        return 0.0
    inter = max(0.0, min(px2, tx2) - max(px1, tx1)) * max(0.0, min(py2, ty2) - max(py1, ty1))
    pred_area = max(0.0, px2 - px1) * max(0.0, py2 - py1)
    target_area = max(0.0, tx2 - tx1) * max(0.0, ty2 - ty1)
    return float(inter / max(pred_area + target_area - inter, 1e-8))


# ----------------------- bbox 与预测轨迹可视化 -----------------------

def _draw_predicted_bbox(frame_bgr: np.ndarray, bbox_norm: Any, label: str = "model bbox") -> np.ndarray:
    out = ensure_bgr_uint8(frame_bgr).copy()
    x, y, w, h = _bbox_cxcywh_to_xywh(bbox_norm, out.shape[1], out.shape[0])
    if w > 0 and h > 0:
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(out.shape[1] - 1, x + w), min(out.shape[0] - 1, y + h)
        color = (255, 180, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        cv2.drawMarker(
            out,
            ((x1 + x2) // 2, (y1 + y2) // 2),
            color,
            markerType=cv2.MARKER_CROSS,
            markerSize=14,
            thickness=2,
        )
    return out


def _smooth_trajectory_pixels(traj_xyz: Any, width: int, height: int, scale: float) -> np.ndarray:
    """将稀疏局部轨迹转换成平滑像素曲线，不绘制 waypoint 点标记。"""
    traj = np.asarray(traj_xyz) if traj_xyz is not None else np.empty((0, 0))
    if traj.ndim != 2 or traj.shape[0] == 0 or traj.shape[1] == 0:
        return np.empty((0, 2), dtype=np.float32)
    base_x, base_y = width * 0.5, height * 0.86
    points = [(base_x, base_y)]
    for waypoint in traj[:64]:
        x = float(waypoint[0])
        y = float(waypoint[1]) if waypoint.size >= 2 else 0.0
        if math.isfinite(x) and math.isfinite(y):
            # Body-frame +Y is right, so positive lateral displacement must
            # also appear on the right side of the camera overlay.
            points.append((base_x + y * scale, base_y - x * scale))
    points_np = np.asarray(points, dtype=np.float32)
    if points_np.shape[0] < 2:
        return np.empty((0, 2), dtype=np.float32)

    # 用累计弧长参数化并删除重复点，避免三次样条在原地 waypoint 处出现尖角。
    segment_length = np.linalg.norm(np.diff(points_np, axis=0), axis=1)
    keep = np.concatenate([[True], segment_length > 1e-3])
    points_np = points_np[keep]
    if points_np.shape[0] < 2:
        return np.empty((0, 2), dtype=np.float32)
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points_np, axis=0), axis=1))])
    sample_count = int(np.clip(math.ceil(float(arc[-1]) * 1.5), 32, 256))
    dense_arc = np.linspace(0.0, float(arc[-1]), sample_count)
    if points_np.shape[0] >= 3:
        dense_x = CubicSpline(arc, points_np[:, 0], bc_type="natural")(dense_arc)
        dense_y = CubicSpline(arc, points_np[:, 1], bc_type="natural")(dense_arc)
        dense = np.stack([dense_x, dense_y], axis=-1)
    else:
        dense = np.stack(
            [np.interp(dense_arc, arc, points_np[:, 0]), np.interp(dense_arc, arc, points_np[:, 1])],
            axis=-1,
        )
    # 限制极端外推坐标，避免 OpenCV 接收超大整数。
    dense[:, 0] = np.clip(dense[:, 0], -width, width * 2)
    dense[:, 1] = np.clip(dense[:, 1], -height, height * 2)
    return dense.astype(np.float32)


def _render_bgr_frame_with_traj(frame_bgr: np.ndarray, traj_xyz: Any, scale: float) -> np.ndarray:
    """绘制无 waypoint 点标记的平滑渐变轨迹曲线。"""
    out = ensure_bgr_uint8(frame_bgr).copy()
    h, w = out.shape[:2]
    curve = _smooth_trajectory_pixels(traj_xyz, w, h, scale)
    if curve.shape[0] < 2:
        return out

    # 2 倍分辨率绘制后缩小，使渐变曲线在视频中更平滑。
    supersample = 2
    canvas = cv2.resize(out, (w * supersample, h * supersample), interpolation=cv2.INTER_LINEAR)
    curve_hi = np.rint(curve * supersample).astype(np.int32)
    shadow_width = 8 * supersample
    color_width = 4 * supersample
    cv2.polylines(canvas, [curve_hi], False, (12, 12, 12), shadow_width, cv2.LINE_AA)

    # TURBO 色图：近处蓝绿，远处黄红；短线段足够密集，视觉上是一条连续彩色曲线。
    gradient = cv2.applyColorMap(
        np.linspace(35, 235, curve_hi.shape[0] - 1, dtype=np.uint8).reshape(-1, 1),
        cv2.COLORMAP_TURBO,
    ).reshape(-1, 3)
    for idx, (start, end) in enumerate(zip(curve_hi[:-1], curve_hi[1:])):
        cv2.line(canvas, tuple(start), tuple(end), tuple(int(v) for v in gradient[idx]), color_width, cv2.LINE_AA)
    return cv2.resize(canvas, (w, h), interpolation=cv2.INTER_AREA)


def _render_rgb_frame_with_traj(rgb_frame: np.ndarray, traj_xyz: Any, scale: float = 120.0) -> np.ndarray:
    """RGB 兼容入口，内部复用 BGR 平滑曲线实现。"""
    bgr = cv2.cvtColor(np.asarray(rgb_frame, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    rendered = _render_bgr_frame_with_traj(bgr, traj_xyz, scale)
    return cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)


def _overlay_text(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    out = frame.copy()
    y = 24
    for line in lines:
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1)
        y += 22
    return out


def _candidate_label(candidate: Any, score: Any) -> str:
    """Format Anchor Diffusion mode selection without cluttering MLP videos."""
    if candidate is None or score is None:
        return "planner=MLP"
    return f"mode={int(candidate)} score={float(score):.2f}"


# ----------------------- 双 Agent模型加载与在线规划 -----------------------

class UnrealZooMultiAgentPlanner:
    """Online wrapper: RGB frames -> visual tokens -> model waypoints -> UE actions."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt_path = _latest_checkpoint(Path(args.ckpt))
        if ckpt_path is None:
            raise FileNotFoundError(f"No checkpoint found from --ckpt: {args.ckpt}")
        self.ckpt_path = ckpt_path

        print(f"[startup] loading checkpoint metadata/state from CPU: {ckpt_path}", flush=True)
        load_t0 = time.time()
        try:
            obj = torch.load(str(ckpt_path), map_location="cpu")
        except Exception:
            obj = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        print(f"[startup] checkpoint loaded in {time.time() - load_t0:.1f}s", flush=True)
        ckpt_cfg = obj.get("config", {}) if isinstance(obj, dict) else {}
        state = _cleanup_state_dict_keys(obj.get("model_state", {}) if isinstance(obj, dict) else {})
        if ckpt_cfg.get("model_type") == "base_multi_agent_concat":
            raise RuntimeError(
                "This checkpoint is a train_multi_agent_base.py / model_multi_agent_base.py base-concat checkpoint. "
                "Evaluate it with eval_unrealzoo_multi_agent_base.py, for example by setting "
                "EVAL_SCRIPT=eval_unrealzoo_multi_agent_base.py in sh/run_multi_agent_eval.sh."
            )
        self.use_anchor_diffusion = bool(
            ckpt_cfg.get("use_anchor_diffusion", False)
            or "planner_agent1.anchors" in state
            or "planner_agent2.anchors" in state
        )
        if self.use_anchor_diffusion:
            expected_metadata = {
                "grounding_architecture": "dual_agent_gnd_v2",
                "multimodal_sequence_layout": "dual_visual_before_queries_v2",
                "action_scale_version": "anchor_maxabs_v2",
            }
            incompatible = {
                key: ckpt_cfg.get(key)
                for key, expected in expected_metadata.items()
                if ckpt_cfg.get(key) != expected
            }
            if incompatible:
                raise RuntimeError(
                    "Checkpoint is incompatible with the corrected Anchor Diffusion architecture. "
                    f"Expected {expected_metadata}, found {incompatible}. Retrain from scratch."
                )
            from cundang.model_unrealzoo_anchor_diffusion import MultiAgentModelConfig, MultiAgentOpenTrackVLA
            MultiAgentModelClass = MultiAgentOpenTrackVLA
            model_source = "model_unrealzoo_anchor_diffusion.py Anchor Diffusion"
            # 早期自包含 scheduler 将 alphas_cumprod 写入 checkpoint；diffusers
            # 调度器会自行重建同类状态，因此加载新版模型时丢弃旧 buffer。
            state = {k: v for k, v in state.items() if not k.endswith("scheduler.alphas_cumprod")}
        else:
            from model import MultiAgentModelConfig, MultiAgentOpenTrackVLA, MultiAgentSeparateOpenTrackVLA
            use_separate_context = bool(
                ckpt_cfg.get("separate_agent_context", False)
                or ckpt_cfg.get("model_type") == "model_py_multi_agent_separate_base"
            )
            MultiAgentModelClass = MultiAgentSeparateOpenTrackVLA if use_separate_context else MultiAgentOpenTrackVLA
            model_source = "model.py separate-context MLP planner" if use_separate_context else "model.py MLP planner"
        has_planner_agent1 = any(k.startswith("planner_agent1.") for k in state)
        has_planner_agent2 = any(k.startswith("planner_agent2.") for k in state)
        print(
            f"[planner] checkpoint model_type={ckpt_cfg.get('model_type')} "
            f"separate_agent_context={ckpt_cfg.get('separate_agent_context')} "
            f"model_source={model_source}",
            flush=True,
        )
        print(
            "[planner] agent_order=drone,robotdog "
            "binding=drone:planner_agent1/waypoints[0],robotdog:planner_agent2/waypoints[1] "
            f"planner_agent1={has_planner_agent1} planner_agent2={has_planner_agent2}",
            flush=True,
        )
        if not has_planner_agent1 or not has_planner_agent2:
            raise RuntimeError(
                "Checkpoint does not contain both planner_agent1.* and planner_agent2.* weights. "
                "Cannot bind drone to planner_agent1 and robotdog to planner_agent2 safely."
            )
        self.ckpt_bbox_dropout_prob = float(ckpt_cfg.get("bbox_dropout_prob", 0.0))
        if args.bbox_source in {"model", "none"} and self.ckpt_bbox_dropout_prob <= 0.0:
            print(
                "[planner][warn] checkpoint was not trained with bbox_dropout_prob > 0; "
                "prior-free detection may be unreliable and should be treated as an ablation.",
                flush=True,
            )
        self.history = int(ckpt_cfg.get("history", args.history))
        self.history_frame_dt = float(
            args.history_frame_dt if args.history_frame_dt > 0.0 else args.dt
        )
        self.n_waypoints = int(ckpt_cfg.get("n_waypoints", args.n_waypoints))
        self.action_dims = int(ckpt_cfg.get("action_dims", 3))
        vision_feat_dim = int(ckpt_cfg.get("vision_feat_dim", args.vision_feat_dim))

        model_cfg_kwargs = dict(
            llm_name=str(ckpt_cfg.get("llm_name", args.llm_name)),
            freeze_llm=True,
            n_waypoints=self.n_waypoints,
            action_dims=self.action_dims,
            use_angle_tvi=bool(ckpt_cfg.get("use_angle_tvi", args.use_angle_tvi)),
            insert_time_tokens=bool(ckpt_cfg.get("insert_time_tokens", True)),
            use_tanh_actions=not bool(ckpt_cfg.get("no_tanh_actions", args.no_tanh_actions)),
            alpha_xy=ckpt_cfg.get("alpha_xy", args.alpha_xy),
            return_token_logits=False,
        )
        if not self.use_anchor_diffusion:
            model_cfg_kwargs.update(
                use_grounding=bool(ckpt_cfg.get("use_grounding", True)),
                use_bbox_tokens=bool(ckpt_cfg.get("use_bbox_tokens", True)),
                use_agent_text_markers=bool(ckpt_cfg.get("use_agent_text_markers", True)),
            )
        if self.use_anchor_diffusion:
            model_cfg_kwargs.update(
                use_anchor_diffusion=True,
                diffusion_anchor_path=ckpt_cfg.get("diffusion_anchor_path"),
                diffusion_agent1_anchor_path=ckpt_cfg.get("diffusion_agent1_anchor_path"),
                diffusion_agent2_anchor_path=ckpt_cfg.get("diffusion_agent2_anchor_path"),
                diffusion_num_anchors=int(ckpt_cfg.get("diffusion_num_anchors", args.diffusion_num_anchors)),
                diffusion_hidden_dim=int(ckpt_cfg.get("diffusion_hidden_dim", args.diffusion_hidden_dim)),
                diffusion_depth=int(ckpt_cfg.get("diffusion_depth", args.diffusion_depth)),
                diffusion_num_heads=int(ckpt_cfg.get("diffusion_num_heads", args.diffusion_num_heads)),
                diffusion_mlp_ratio=float(ckpt_cfg.get("diffusion_mlp_ratio", 4.0)),
                diffusion_dropout=float(ckpt_cfg.get("diffusion_dropout", 0.0)),
                diffusion_num_train_timesteps=int(ckpt_cfg.get("diffusion_num_train_timesteps", 1000)),
                diffusion_train_truncation_steps=int(ckpt_cfg.get("diffusion_train_truncation_steps", 50)),
                diffusion_inference_start_timestep=int(ckpt_cfg.get("diffusion_inference_start_timestep", 10)),
                diffusion_inference_steps=int(ckpt_cfg.get("diffusion_inference_steps", 2)),
                diffusion_score_loss_weight=float(ckpt_cfg.get("diffusion_score_loss_weight", 100.0)),
                diffusion_score_loss_reduction=str(ckpt_cfg.get("diffusion_score_loss_reduction", "mean")),
                diffusion_deterministic_inference=bool(args.diffusion_deterministic_inference),
            )
        model_cfg = MultiAgentModelConfig(**model_cfg_kwargs)
        if not self.use_anchor_diffusion and model_cfg.is_base_variant:
            layout = (
                "[joint_text, agent1_text, agent1_visual, agent2_text, agent2_visual, ACT1, ACT2]"
                if model_cfg.use_agent_text_markers
                else "[joint_text, agent1_visual, agent2_visual, ACT1, ACT2]"
            )
            print(f"[planner] shared_context_layout={layout}", flush=True)
        print(f"[startup] building {model_source} on {self.device}", flush=True)
        self.model = MultiAgentModelClass(model_cfg, vision_feat_dim=vision_feat_dim).to(self.device).eval()
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if not any(k.startswith("planner_agent1") for k in state.keys()):
            raise ValueError(
                "This checkpoint does not look like a MultiAgentOpenTrackVLA checkpoint. "
                "Habitat single-agent checkpoints should still be evaluated with sh/eval.sh."
            )
        if self.use_anchor_diffusion and "gnd_token_1" not in state:
            print(
                "[planner][warn] checkpoint uses the old shared-GND grounding architecture; "
                "planning weights were loaded, but dual-GND bbox/visibility outputs are newly initialized. "
                "Retrain before evaluating grounding metrics.",
                flush=True,
            )
        critical_prefixes = ["planner_agent1.", "planner_agent2.", "proj.", "tvi."]
        if self.use_anchor_diffusion or bool(ckpt_cfg.get("use_grounding", True)):
            critical_prefixes.extend(["grounding_head.", "grounding_to_act."])
        critical_missing = [key for key in missing if key.startswith(tuple(critical_prefixes))]
        if self.use_anchor_diffusion:
            critical_missing.extend(
                key for key in missing
                if key in {"act_token_1", "act_token_2", "gnd_token_1", "gnd_token_2"}
            )
        elif bool(ckpt_cfg.get("use_grounding", True)):
            critical_missing.extend(
                key for key in missing
                if key in {"act_token_1", "act_token_2", "gnd_token_1", "gnd_token_2", "grounding_act_gate"}
            )
        if critical_missing:
            raise RuntimeError(f"Checkpoint is missing critical model weights: {critical_missing[:20]}")
        print(
            f"[planner] loaded {ckpt_path} missing={len(missing)} unexpected={len(unexpected)} "
            f"history={self.history} history_frame_dt={self.history_frame_dt:.3f}s "
            f"n_waypoints={self.n_waypoints} anchor_diffusion={self.use_anchor_diffusion}",
            flush=True,
        )

        print("[startup] loading online DINO + SigLIP visual encoders", flush=True)
        vision_t0 = time.time()
        self.encoder = VisionFeatureCacher(
            VisionCacheConfig(image_size=args.image_size, batch_size=2, device=str(self.device))
        ).eval()
        print(f"[startup] visual encoders loaded in {time.time() - vision_t0:.1f}s", flush=True)
        history_capacity = max(self.history * 4, self.history + 1)
        self.histories: list[deque[tuple[float, torch.Tensor]]] = [
            deque(maxlen=history_capacity),
            deque(maxlen=history_capacity),
        ]
        self.last_waypoints: Optional[np.ndarray] = None
        self.last_predicted_bbox: Optional[np.ndarray] = None

    def reset(self) -> None:
        for hist in self.histories:
            hist.clear()
        self.last_waypoints = None
        self.last_predicted_bbox = None

    @torch.inference_mode()
    def _encode_pair(self, drone_frame_bgr: np.ndarray, dog_frame_bgr: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        pils = []
        for frame in (drone_frame_bgr, dog_frame_bgr):
            rgb = cv2.cvtColor(ensure_bgr_uint8(frame), cv2.COLOR_BGR2RGB)
            pils.append(Image.fromarray(rgb))
        tok_dino, hp, wp = self.encoder._encode_dino(pils)
        tok_sigl = self.encoder._encode_siglip(pils, out_hw=(hp, wp))
        tokens = torch.cat([tok_dino, tok_sigl], dim=-1)
        vfine = grid_pool_tokens(tokens, hp, wp, out_tokens=64).float().cpu()
        vcoarse = grid_pool_tokens(tokens, hp, wp, out_tokens=4).float().cpu()
        return vcoarse, vfine

    def _history_tensor(
        self,
        agent_idx: int,
        current_coarse: torch.Tensor,
        observation_time: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hist = self.histories[agent_idx]
        current_coarse = current_coarse.cpu()

        # Training uses consecutive 0.1 s frames. Realtime inference is much
        # slower, so sample sparse observations onto the same temporal grid
        # instead of treating each inference call as one 0.1 s frame.
        entries = list(hist)
        if not entries:
            frames = [current_coarse]
        else:
            target_times = [
                observation_time - (self.history - index) * self.history_frame_dt
                for index in range(self.history)
            ]
            frames = []
            entry_index = 0
            for target_time in target_times:
                while (
                    entry_index + 1 < len(entries)
                    and entries[entry_index + 1][0] <= target_time
                ):
                    entry_index += 1
                frames.append(entries[entry_index][1])
        if len(frames) < self.history:
            frames = [frames[0]] * (self.history - len(frames)) + frames
        coarse = torch.cat(frames, dim=0)
        tidx = torch.cat([torch.full((tok.size(0),), i, dtype=torch.long) for i, tok in enumerate(frames)], dim=0)
        hist.append((float(observation_time), current_coarse))
        return coarse, tidx

    @torch.inference_mode()
    def predict(
        self,
        drone_frame_bgr: np.ndarray,
        dog_frame_bgr: np.ndarray,
        drone_bbox: Optional[list[float]],
        dog_bbox: Optional[list[float]],
        instruction: str,
        joint_instruction: Optional[str] = None,
        agent1_instruction: Optional[str] = None,
        agent2_instruction: Optional[str] = None,
        observation_time: Optional[float] = None,
    ) -> dict[str, Any]:
        vcoarse, vfine = self._encode_pair(drone_frame_bgr, dog_frame_bgr)
        if observation_time is None:
            observation_time = time.monotonic()
        coarse_items = []
        coarse_tidx_items = []
        for agent_idx in range(2):
            c, ct = self._history_tensor(
                agent_idx,
                vcoarse[agent_idx],
                float(observation_time),
            )
            coarse_items.append(c)
            coarse_tidx_items.append(ct)

        coarse_tokens = torch.stack(coarse_items, dim=0).unsqueeze(0).to(self.device)
        coarse_tidx = torch.stack(coarse_tidx_items, dim=0).unsqueeze(0).to(self.device)
        fine_tokens = vfine.unsqueeze(0).to(self.device)
        fine_tidx = torch.full((1, 2, vfine.size(1)), self.history, dtype=torch.long, device=self.device)
        bbox_input = None
        if self.args.bbox_source == "ground_truth":
            bbox_input = [drone_bbox or [0.0] * 4, dog_bbox or [0.0] * 4]
        elif self.args.bbox_source == "model" and self.last_predicted_bbox is not None:
            bbox_input = self.last_predicted_bbox.tolist()
        bbox_feat = (
            torch.tensor([bbox_input], dtype=torch.float32, device=self.device)
            if bbox_input is not None
            else None
        )

        model_kwargs = dict(
            coarse_tokens=coarse_tokens,
            coarse_tidx=coarse_tidx,
            fine_tokens=fine_tokens,
            fine_tidx=fine_tidx,
            instructions=[instruction],
            bbox_feat=bbox_feat,
            return_dict=True,
        )
        if not self.use_anchor_diffusion:
            model_kwargs.update(
                joint_instructions=[joint_instruction or instruction],
                agent1_instructions=[agent1_instruction] if agent1_instruction else None,
                agent2_instructions=[agent2_instruction] if agent2_instruction else None,
            )
        out = self.model(**model_kwargs)
        waypoints = out["waypoints"].detach().float().cpu().numpy()[0]
        raw_refined_bbox = out["refined_bbox"].detach().float().cpu().numpy()[0]
        absolute_bbox = out.get("absolute_bbox")
        absolute_bbox = (
            absolute_bbox.detach().float().cpu().numpy()[0]
            if absolute_bbox is not None
            else raw_refined_bbox.copy()
        )
        predicted_bbox = raw_refined_bbox.copy()
        bbox_fallback_to_absolute = [False, False]
        # 循环使用上一帧 bbox 时，residual head 的宽高可能逐帧收缩到 0。
        # 退化后回退到同一模型的 absolute detection head，避免坏框永久传播。
        if self.args.bbox_source == "model":
            min_size = float(getattr(self.args, "bbox_min_size", 0.01))
            for agent_idx in range(predicted_bbox.shape[0]):
                box = predicted_bbox[agent_idx]
                invalid = (not np.isfinite(box).all()) or float(box[2]) < min_size or float(box[3]) < min_size
                if invalid:
                    predicted_bbox[agent_idx] = absolute_bbox[agent_idx]
                    bbox_fallback_to_absolute[agent_idx] = True
        self.last_waypoints = waypoints
        if self.args.bbox_source == "model":
            self.last_predicted_bbox = predicted_bbox
        candidate_logits = out.get("candidate_logits")
        candidate_scores = out.get("candidate_scores")
        best_candidate = None
        best_candidate_score = None
        if candidate_logits is not None:
            candidate_logits_np = candidate_logits.detach().float().cpu().numpy()[0]
            best_candidate = candidate_logits_np.argmax(axis=-1).tolist()
        if candidate_scores is not None and best_candidate is not None:
            candidate_scores_np = candidate_scores.detach().float().cpu().numpy()[0]
            best_candidate_score = [
                float(candidate_scores_np[agent_idx, anchor_idx])
                for agent_idx, anchor_idx in enumerate(best_candidate)
            ]
        return {
            "waypoints": waypoints,
            "bbox_input": bbox_input,
            "bbox_source": self.args.bbox_source,
            "visible_score": out.get("visible_score").detach().float().cpu().numpy()[0].tolist()
            if out.get("visible_score") is not None
            else None,
            "refined_bbox": predicted_bbox.tolist(),
            "raw_refined_bbox": raw_refined_bbox.tolist(),
            "absolute_bbox": absolute_bbox.tolist(),
            "bbox_fallback_to_absolute": bbox_fallback_to_absolute,
            "best_candidate": best_candidate,
            "best_candidate_score": best_candidate_score,
        }

    def waypoints_to_actions(
        self,
        waypoints: np.ndarray,
        realtime_control_period_seconds: Optional[float] = None,
    ) -> tuple[list[float], list[float], dict[str, Any]]:
        """Convert model outputs to UnrealZoo actions.

        Agent order follows training: agent1=drone, agent2=robotdog.
        For robotdog, UnrealZoo only accepts [turn_deg, speed_cm_s], so the
        predicted lateral velocity is logged but not executed.
        """
        default_idx = int(self.args.waypoint_index)
        realtime_timing = bool(getattr(self.args, "realtime_waypoint_timing", False))
        realtime_elapsed_raw = None
        realtime_elapsed_clipped = None
        if realtime_timing:
            if bool(getattr(self.args, "deterministic_step", True)):
                raise ValueError("--realtime-waypoint-timing requires --no-deterministic-step")
            if realtime_control_period_seconds is None:
                raise ValueError("Realtime waypoint timing requires an observed control period")
            realtime_elapsed_raw = max(0.0, float(realtime_control_period_seconds))
            realtime_elapsed_clipped = float(
                np.clip(
                    realtime_elapsed_raw,
                    float(self.args.realtime_waypoint_min_seconds),
                    float(self.args.realtime_waypoint_max_seconds),
                )
            )

            horizon_steps = max(1, int(self.args.waypoint_horizon_steps))
            waypoint_count = int(waypoints.shape[1])

            def source_time(index: int) -> float:
                source_index = (
                    0
                    if waypoint_count <= 1
                    else int(round(index * (horizon_steps - 1) / (waypoint_count - 1)))
                )
                return max((source_index + 1) * float(self.args.dt), 1e-6)

            default_idx = min(
                range(waypoint_count),
                key=lambda index: abs(source_time(index) - realtime_elapsed_clipped),
            )
        drone_idx = int(
            np.clip(
                default_idx
                if realtime_timing
                else self.args.drone_waypoint_index
                if self.args.drone_waypoint_index is not None
                else default_idx,
                0,
                waypoints.shape[1] - 1,
            )
        )
        dog_idx = int(
            np.clip(
                default_idx
                if realtime_timing
                else self.args.robotdog_waypoint_index
                if self.args.robotdog_waypoint_index is not None
                else default_idx,
                0,
                waypoints.shape[1] - 1,
            )
        )
        # Training integrates `waypoint_horizon_steps` actions and then
        # resamples those points to n_waypoints outputs. Recover the source
        # action step instead of assuming adjacent output tokens are one dt
        # apart (the current data uses 9 source steps and 10 output points).
        horizon_steps = max(1, int(self.args.waypoint_horizon_steps))
        waypoint_count = int(waypoints.shape[1])

        def waypoint_time(index: int) -> tuple[int, float]:
            source_index = (
                0
                if waypoint_count <= 1
                else int(round(index * (horizon_steps - 1) / (waypoint_count - 1)))
            )
            source_step = source_index + 1
            return source_step, max(source_step * float(self.args.dt), 1e-6)

        drone_source_step, drone_horizon_dt = waypoint_time(drone_idx)
        dog_source_step, dog_horizon_dt = waypoint_time(dog_idx)
        drone_vel = waypoints[0, drone_idx, :3] / drone_horizon_dt
        dog_vel = waypoints[1, dog_idx, :3] / dog_horizon_dt

        drone_vx = float(np.clip(drone_vel[0] * self.args.drone_vx_scale, -self.args.drone_max_vx, self.args.drone_max_vx))
        drone_vy = float(np.clip(drone_vel[1] * self.args.drone_vy_scale, -self.args.drone_max_vy, self.args.drone_max_vy))
        drone_yaw_unclipped = float(
            drone_vel[2] * self.args.drone_yaw_sign * self.args.drone_yaw_scale
        )
        drone_w = float(
            np.clip(
                drone_yaw_unclipped,
                -self.args.drone_max_yaw_rate,
                self.args.drone_max_yaw_rate,
            )
        )
        drone_action = [drone_vx, drone_vy, 0.0, drone_w]

        dog_speed = float(
            np.clip(
                dog_vel[0] * UNREAL_UNITS_PER_METER * float(self.args.robotdog_speed_gain),
                -self.args.robotdog_max_speed * UNREAL_UNITS_PER_METER,
                self.args.robotdog_max_speed * UNREAL_UNITS_PER_METER,
            )
        )
        dog_turn = float(
            np.clip(
                math.degrees(
                    dog_vel[2]
                    * self.args.robotdog_yaw_sign
                    * self.args.robotdog_yaw_scale
                ),
                -self.args.robotdog_max_turn_deg,
                self.args.robotdog_max_turn_deg,
            )
        )
        dog_action = [dog_turn, dog_speed]

        debug = {
            "waypoint_index": default_idx,
            "drone_waypoint_index": drone_idx,
            "robotdog_waypoint_index": dog_idx,
            "waypoint_horizon_steps": horizon_steps,
            "drone_waypoint_source_step": drone_source_step,
            "robotdog_waypoint_source_step": dog_source_step,
            "action_source": "derived_from_model_waypoints",
            "drone_horizon_dt": float(drone_horizon_dt),
            "robotdog_horizon_dt": float(dog_horizon_dt),
            "drone_waypoint": [float(v) for v in waypoints[0, drone_idx, :3].tolist()],
            "robotdog_waypoint": [float(v) for v in waypoints[1, dog_idx, :3].tolist()],
            "drone_velocity_pred": [float(v) for v in drone_vel.tolist()],
            "robotdog_velocity_pred": [float(v) for v in dog_vel.tolist()],
            "drone_yaw_scale": float(self.args.drone_yaw_scale),
            "drone_yaw_unclipped": float(drone_yaw_unclipped),
            "drone_yaw_command": float(drone_w),
            "robotdog_speed_gain": float(self.args.robotdog_speed_gain),
            "robotdog_yaw_scale": float(self.args.robotdog_yaw_scale),
            "robotdog_lateral_ignored": float(dog_vel[1]),
            "realtime_waypoint_timing": realtime_timing,
            "realtime_timing_source": "previous_observation_interval_wall_clock" if realtime_timing else None,
            "realtime_control_period_seconds_raw": realtime_elapsed_raw,
            "realtime_control_period_seconds_clipped": realtime_elapsed_clipped,
            # Compatibility aliases for results produced by the first realtime implementation.
            "realtime_elapsed_seconds_raw": realtime_elapsed_raw,
            "realtime_elapsed_seconds_clipped": realtime_elapsed_clipped,
        }
        return drone_action, dog_action, debug


# ----------------------- Episode 初始化与观测读取 -----------------------

def align_ideal_follow_distances(args: argparse.Namespace) -> dict[str, float]:
    """Use each configured follow range midpoint as the spawn distance."""
    values: dict[str, float] = {}
    for agent in ("robotdog", "drone"):
        min_dist = float(getattr(args, f"{agent}_min_follow_dist"))
        max_dist = float(getattr(args, f"{agent}_max_follow_dist"))
        if min_dist <= 0.0 or max_dist < min_dist:
            raise ValueError(
                f"Invalid {agent} follow range: min={min_dist}, max={max_dist}"
            )
        ideal_dist = 0.5 * (min_dist + max_dist)
        setattr(args, f"{agent}_ideal_follow_dist", ideal_dist)
        values[agent] = ideal_dist
    return values


def setup_episode(
    env,
    args: argparse.Namespace,
    episode_id: int,
    rng: random.Random,
    target_trajectory: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    if bool(getattr(args, "deterministic_step", True)):
        unrealcv = getattr(env.unwrapped, "unrealcv", None)
        if unrealcv is not None:
            try:
                unrealcv.set_resume()
            except Exception:
                pass
    reset_env(env, args)
    target_id, robotdog_id, drone_id = classify_coop_agents(env)
    env.unwrapped.target_id = target_id
    env.unwrapped.tracker_id = drone_id
    env.unwrapped.protagonist_id = drone_id
    players = env.unwrapped.player_list
    target_name = players[target_id]
    robotdog_name = players[robotdog_id]
    drone_name = players[drone_id]

    appearances = set_episode_appearances(env, target_id, robotdog_id, [], rng, args)
    if target_trajectory is not None:
        first_pose = target_trajectory["poses"][0]
        env.unwrapped.unrealcv.set_obj_location(target_name, first_pose[:3])
        try:
            env.unwrapped.unrealcv.set_obj_rotation(target_name, first_pose[3:6])
        except Exception:
            pass
    elif args.open_spawn:
        target_start = pick_open_start(env, rng, args)
        env.unwrapped.unrealcv.set_obj_location(target_name, target_start)
    update_observation(env, refresh_cameras=True)
    target_pose = list(env.unwrapped.obj_poses[target_id])
    if target_trajectory is not None:
        replay_mode = _target_replay_mode(args)
        if replay_mode == "path_goal":
            target_path = _target_path_waypoints(target_trajectory, args)
            target_goal = list((target_path[-1] if target_path else target_trajectory["poses"][-1][:3]))
        else:
            target_goal = list(target_trajectory["poses"][-1][:3])
            target_path = [list(pose[:3]) for pose in target_trajectory["poses"]]
    else:
        replay_mode = "simulator_navigation"
        target_goal, target_path = pick_reachable_goal(
            env,
            target_name,
            rng,
            avoid_pos=target_pose,
            min_distance=args.human_goal_min_distance,
            max_distance=args.human_goal_max_distance,
            max_trials=32,
        )
    # Use the first path segment only for fallback follower placement. A
    # recorded episode keeps its authored target rotation, as Habitat does.
    initial_motion_goal = target_goal
    if target_trajectory is not None and replay_mode == "path_goal" and len(target_path) >= 2:
        initial_motion_goal = list(target_path[1])
    goal_direction = np.asarray(initial_motion_goal[:2]) - np.asarray(target_pose[:2])
    if target_trajectory is None and np.linalg.norm(goal_direction) > 1e-6:
        target_yaw = math.degrees(math.atan2(float(goal_direction[1]), float(goal_direction[0])))
        try:
            env.unwrapped.unrealcv.set_obj_rotation(target_name, [0.0, target_yaw, 0.0])
        except Exception:
            pass
    try:
        target_speed = 0.0 if target_trajectory is not None and replay_mode == "pose" else float(args.human_speed)
        env.unwrapped.unrealcv.set_max_speed(target_name, target_speed)
        env.unwrapped.unrealcv.set_max_speed(robotdog_name, float(args.robotdog_max_speed) * UNREAL_UNITS_PER_METER)
    except Exception:
        pass

    update_observation(env, refresh_cameras=True)
    initial_follow_distances = align_ideal_follow_distances(args)
    use_recorded_agent_poses = bool(
        target_trajectory is not None
        and getattr(args, "init_from_recorded_agent_poses", False)
    )
    if not use_recorded_agent_poses:
        place_initial_followers(env, target_id, robotdog_id, drone_id, initial_motion_goal, args)
    restored_agent_poses = restore_recorded_agent_initial_poses(
        env,
        args,
        robotdog_name,
        drone_name,
        target_trajectory,
    )
    if use_recorded_agent_poses and not all(restored_agent_poses.values()):
        raise EpisodeSkipped(
            "recorded Habitat-style initialization requires both drone and "
            f"robotdog poses, got {restored_agent_poses}"
        )
    if target_trajectory is not None:
        if replay_mode in {"pose", "path_goal"}:
            # Keep the recorded start pose fixed until frame 0 has been consumed.
            env.unwrapped.unrealcv.nav_to_goal(target_name, target_pose[:3])
        elif replay_mode == "nav_goal":
            env.unwrapped.unrealcv.nav_to_goal(target_name, target_goal)
    else:
        env.unwrapped.unrealcv.nav_to_goal(target_name, target_goal)

    dog_cfg = dog_args(args)
    drone_cfg = drone_args(args)
    (
        _obs,
        dog_pose,
        target_pose,
        dog_visibility,
        dog_visible,
        dog_bbox,
        dog_pitch,
        dog_yaw_offset,
        dog_mount,
        dog_self_visibility,
        dog_self_bbox,
    ) = choose_robotdog_camera_for_current_pose(env, target_id, robotdog_id, dog_cfg)
    (
        _obs,
        drone_pose,
        target_pose,
        drone_visibility,
        drone_visible,
        drone_bbox,
        drone_pitch,
        drone_yaw_offset,
    ) = choose_drone_camera_for_current_pose(env, target_id, drone_id, drone_cfg)
    if bool(getattr(args, "init_from_recorded_agent_poses", False)):
        (
            dog_mount,
            dog_pitch,
            dog_yaw_offset,
            drone_pitch,
            drone_yaw_offset,
            restored_agent_cameras,
        ) = apply_recorded_agent_cameras(
            target_trajectory,
            dog_mount,
            dog_pitch,
            dog_yaw_offset,
            drone_pitch,
            drone_yaw_offset,
        )
        set_robotdog_camera(env, robotdog_name, robotdog_id, dog_mount, dog_pitch, dog_yaw_offset, dog_cfg)
        set_drone_camera(
            env,
            drone_name,
            drone_id,
            list(env.unwrapped.obj_poses[drone_id]),
            drone_pitch,
            drone_yaw_offset,
            drone_cfg,
        )
        obs = update_observation(env, refresh_cameras=False)
        dog_pose = list(env.unwrapped.obj_poses[robotdog_id])
        drone_pose = list(env.unwrapped.obj_poses[drone_id])
        target_pose = list(env.unwrapped.obj_poses[target_id])
        dog_visibility, dog_visible, dog_bbox = target_mask_visibility(
            env, env.unwrapped.cam_list[robotdog_id], target_name
        )
        drone_visibility, drone_visible, drone_bbox = target_mask_visibility(
            env, env.unwrapped.cam_list[drone_id], target_name
        )
    else:
        restored_agent_cameras = {"robotdog": False, "drone": False}
    if args.require_visual_target:
        dog_ok = bool(dog_visible and dog_visibility >= args.min_visible_ratio)
        drone_ok = bool(drone_visible and drone_visibility >= args.min_visible_ratio)
        agent = getattr(args, "agent", None)
        if agent == "drone":
            if not drone_ok:
                raise EpisodeSkipped("initial target is not visible for drone")
        elif agent == "robotdog":
            if not dog_ok:
                raise EpisodeSkipped("initial target is not visible for robotdog")
        elif not (dog_ok and drone_ok):
            raise EpisodeSkipped(
                f"initial target is not visible for both agents: robotdog={dog_ok} drone={drone_ok}"
            )

    stop_wait_min = max(0, int(getattr(args, "target_stop_wait_min_steps", 5)))
    stop_wait_max = max(stop_wait_min, int(getattr(args, "target_stop_wait_max_steps", 15)))
    return {
        "episode_id": str(episode_id),
        "target_motion_mode": f"recorded_{replay_mode}" if target_trajectory is not None else "simulator_navigation",
        "recorded_target_source": target_trajectory["source"] if target_trajectory is not None else None,
        "recorded_target_episode": target_trajectory["episode_name"] if target_trajectory is not None else None,
        "recorded_target_pose_count": len(target_trajectory["poses"]) if target_trajectory is not None else None,
        "target_id": target_id,
        "robotdog_id": robotdog_id,
        "drone_id": drone_id,
        "target_name": target_name,
        "robotdog_name": robotdog_name,
        "drone_name": drone_name,
        "appearances": appearances,
        "target_goal": target_goal,
        "initial_motion_goal": initial_motion_goal,
        "initial_motion_yaw": float(
            math.degrees(math.atan2(float(goal_direction[1]), float(goal_direction[0])))
        )
        if np.linalg.norm(goal_direction) > 1e-6
        else float(yaw_deg(target_pose)),
        "target_waypoints": [target_goal],
        "target_path": target_path,
        "target_replay_mode": replay_mode if target_trajectory is not None else None,
        "target_stopped": False,
        "target_stop_step": None,
        "target_stop_wait_count": 0,
        "target_stop_wait_steps": rng.randint(stop_wait_min, stop_wait_max),
        "init_from_recorded_agent_poses": bool(getattr(args, "init_from_recorded_agent_poses", False)),
        "restored_recorded_agent_poses": restored_agent_poses,
        "restored_recorded_agent_cameras": restored_agent_cameras,
        "initial_distance_xy_m": {
            "robotdog": float(distance_xy_m(dog_pose, target_pose)),
            "drone": float(distance_xy_m(drone_pose, target_pose)),
        },
        "follow_distance_config_m": {
            "robotdog": {
                "min": float(args.robotdog_min_follow_dist),
                "ideal": float(initial_follow_distances["robotdog"]),
                "max": float(args.robotdog_max_follow_dist),
            },
            "drone": {
                "min": float(args.drone_min_follow_dist),
                "ideal": float(initial_follow_distances["drone"]),
                "max": float(args.drone_max_follow_dist),
            },
        },
        "dog_camera": {
            "mount": [float(v) for v in dog_mount],
            "pitch": float(dog_pitch),
            "yaw_offset": float(dog_yaw_offset),
            "self_visibility": float(dog_self_visibility),
            "self_bbox": dog_self_bbox,
        },
        "drone_camera": {
            "pitch": float(drone_pitch),
            "yaw_offset": float(drone_yaw_offset),
        },
        "initial_bbox": {
            "drone": drone_bbox,
            "robotdog": dog_bbox,
        },
        "initial_pose": {
            "target": list(env.unwrapped.obj_poses[target_id]),
            "drone": drone_pose,
            "robotdog": dog_pose,
        },
    }


def _target_replay_mode(args: argparse.Namespace) -> str:
    return str(getattr(args, "target_replay_mode", "pose"))


def _target_path_waypoints(trajectory: dict[str, Any], args: argparse.Namespace) -> list[list[float]]:
    poses = trajectory.get("poses") or []
    if not poses:
        return []
    min_spacing = max(float(getattr(args, "target_path_min_spacing", 100.0)), 0.0)
    waypoints: list[list[float]] = []
    last_xy: Optional[np.ndarray] = None
    for pose in poses:
        if not isinstance(pose, list) or len(pose) < 3:
            continue
        xyz = [float(value) for value in pose[:3]]
        xy = np.asarray(xyz[:2], dtype=np.float64)
        if last_xy is None or float(np.linalg.norm(xy - last_xy)) >= min_spacing:
            waypoints.append(xyz)
            last_xy = xy
    final_pose = poses[-1]
    if isinstance(final_pose, list) and len(final_pose) >= 3:
        final_xyz = [float(value) for value in final_pose[:3]]
        if not waypoints or float(np.linalg.norm(np.asarray(final_xyz[:2]) - np.asarray(waypoints[-1][:2]))) > 1e-6:
            waypoints.append(final_xyz)
    return waypoints


def start_recorded_path_navigation(env, args: argparse.Namespace, setup: dict[str, Any], trajectory: dict[str, Any]) -> None:
    waypoints = _target_path_waypoints(trajectory, args)
    setup["target_path_waypoints"] = waypoints
    setup["target_path_index"] = 0
    if len(waypoints) < 2:
        return
    try:
        env.unwrapped.unrealcv.set_max_speed(setup["target_name"], float(args.human_speed))
    except Exception:
        pass
    setup["target_path_index"] = 1


def update_recorded_path_navigation(env, args: argparse.Namespace, setup: dict[str, Any]) -> None:
    waypoints = setup.get("target_path_waypoints") or []
    if len(waypoints) < 2:
        return
    target_pose = list(env.unwrapped.obj_poses[setup["target_id"]])
    target_xy = np.asarray(target_pose[:2], dtype=np.float64)
    reach = max(float(getattr(args, "target_path_reach_distance", 120.0)), 1.0)
    current_index = int(setup.get("target_path_index", 1))

    while current_index < len(waypoints) - 1:
        goal_xy = np.asarray(waypoints[current_index][:2], dtype=np.float64)
        if float(np.linalg.norm(goal_xy - target_xy)) > reach:
            break
        current_index += 1

    if current_index != int(setup.get("target_path_index", 1)):
        setup["target_path_index"] = current_index
        env.unwrapped.unrealcv.nav_to_goal(setup["target_name"], waypoints[current_index])


def update_habitat_target_stop_state(
    env,
    args: argparse.Namespace,
    setup: dict[str, Any],
    *,
    step_idx: int,
    recorded_pose_exhausted: bool = False,
) -> bool:
    """Stop the target at its final goal while followers stay model-controlled."""
    if bool(setup.get("target_stopped", False)):
        _set_recorded_target_pose(
            env,
            setup,
            list(env.unwrapped.obj_poses[setup["target_id"]]),
        )
        return True

    replay_mode = setup.get("target_replay_mode")
    reached = bool(replay_mode == "pose" and recorded_pose_exhausted)
    final_goal: Optional[list[float]] = None
    if replay_mode == "path_goal":
        path = setup.get("target_path_waypoints") or []
        if path:
            final_goal = path[-1]
            reached = int(setup.get("target_path_index", 0)) >= len(path) - 1
    elif replay_mode != "pose":
        final_goal = setup.get("target_goal")

    if reached and final_goal is not None:
        target_pose = list(env.unwrapped.obj_poses[setup["target_id"]])
        goal_distance = float(
            np.linalg.norm(
                np.asarray(target_pose[:2], dtype=np.float64)
                - np.asarray(final_goal[:2], dtype=np.float64)
            )
        )
        reached = goal_distance <= max(float(args.target_goal_reach_distance), 1.0)

    if not reached:
        return False

    target_pose = list(env.unwrapped.obj_poses[setup["target_id"]])
    _set_recorded_target_pose(env, setup, target_pose)
    setup["target_stopped"] = True
    setup["target_stop_step"] = int(step_idx + 1)
    setup["target_stop_pose"] = target_pose
    print(
        f"[target-stop] ep={setup['episode_id']} step={step_idx + 1} "
        f"wait_steps={setup['target_stop_wait_steps']}",
        flush=True,
    )
    return True


def resume_recorded_path_navigation(env, setup: dict[str, Any]) -> None:
    """Resume the target only for the short simulator action pulse."""
    waypoints = setup.get("target_path_waypoints") or []
    if len(waypoints) < 2 or bool(setup.get("target_stopped", False)):
        return
    current_index = int(np.clip(setup.get("target_path_index", 1), 1, len(waypoints) - 1))
    env.unwrapped.unrealcv.nav_to_goal(setup["target_name"], waypoints[current_index])


def configure_deterministic_clock(env, args: argparse.Namespace) -> None:
    """Configure either fixed-step barriers or continuously running realtime UE."""
    unrealcv = env.unwrapped.unrealcv
    if not bool(getattr(args, "deterministic_step", True)):
        unrealcv.set_global_time_dilation(1.0)
        unrealcv.set_max_FPS(float(args.fps))
        unrealcv.set_resume()
        return
    dt = float(args.dt)
    if dt <= 0.0:
        raise ValueError(f"--dt must be positive in deterministic mode, got {dt}")
    unrealcv.set_global_time_dilation(1.0)
    unrealcv.set_max_FPS(1.0 / dt)
    unrealcv.set_pause()
    if not unrealcv.get_is_paused():
        raise RuntimeError("Unreal deterministic step initialization failed: world is not paused")


def deterministic_data_collection_step(
    env,
    args: argparse.Namespace,
    setup: dict[str, Any],
    drone_action: list[float],
    dog_action: list[float],
):
    """Advance exactly one fixed-delta UE frame and return paused observations."""
    unrealcv = env.unwrapped.unrealcv
    if not unrealcv.get_is_paused():
        raise RuntimeError("Unreal world advanced outside deterministic step: expected paused state")

    commands = [
        unrealcv.set_move_bp(setup["drone_name"], drone_action, return_cmd=True),
        unrealcv.set_move_bp(setup["robotdog_name"], dog_action, return_cmd=True),
    ]
    unrealcv.batch_cmd(commands, None)
    pulse_t0 = time.monotonic()
    unrealcv.set_resume()
    # UnrealCV game commands are dispatched on successive game frames. With a
    # fixed timestep, resume followed immediately by pause advances exactly the
    # one frame between those two command dispatches.
    unrealcv.set_pause()
    pulse_wall_seconds = time.monotonic() - pulse_t0
    if not unrealcv.get_is_paused():
        raise RuntimeError("Unreal deterministic step failed to return to paused state")

    idle_actions = [None for _ in env.unwrapped.player_list]
    obs, rewards, done, info = data_collection_step(env, idle_actions)
    info["Action"] = [
        drone_action if idx == setup["drone_id"] else dog_action if idx == setup["robotdog_id"] else None
        for idx in range(len(idle_actions))
    ]
    return obs, rewards, done, info, pulse_wall_seconds


def _lock_eval_cameras(env, setup: dict[str, Any], args: argparse.Namespace) -> None:
    """Refresh fixed cameras without using target pose to choose a new view."""
    drone_id = setup["drone_id"]
    robotdog_id = setup["robotdog_id"]
    drone_name = setup["drone_name"]
    dog_name = setup["robotdog_name"]
    drone_pose = list(env.unwrapped.obj_poses[drone_id])
    try:
        set_drone_camera(
            env,
            drone_name,
            drone_id,
            drone_pose,
            setup["drone_camera"]["pitch"],
            setup["drone_camera"]["yaw_offset"],
            drone_args(args),
        )
    except Exception:
        pass
    try:
        set_robotdog_camera(
            env,
            dog_name,
            robotdog_id,
            setup["dog_camera"]["mount"],
            setup["dog_camera"]["pitch"],
            setup["dog_camera"]["yaw_offset"],
            dog_args(args),
        )
    except Exception:
        pass


def _read_agent_pair(
    env,
    setup: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[
    tuple[np.ndarray, float, bool, list[int], list[float]],
    tuple[np.ndarray, float, bool, list[int], list[float]],
]:
    """Refresh RGB observations once, then read drone and robotdog views."""
    obs = update_observation(env, refresh_cameras=False)

    def read_one(agent_id: int) -> tuple[np.ndarray, float, bool, list[int], list[float]]:
        frame = ensure_bgr_uint8(obs[agent_id])
        visibility, visible, bbox = target_mask_visibility(
            env, env.unwrapped.cam_list[agent_id], setup["target_name"]
        )
        bbox_norm = _normalize_bbox_xywh(bbox, args.width, args.height)
        return frame, float(visibility), bool(visible), bbox, bbox_norm

    return read_one(setup["drone_id"]), read_one(setup["robotdog_id"])


# ----------------------- 单 Episode 闭环评估 -----------------------

def run_episode(
    env,
    args: argparse.Namespace,
    planner: UnrealZooMultiAgentPlanner,
    episode_id: int,
    rng: random.Random,
    target_trajectory: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    setup = setup_episode(env, args, episode_id, rng, target_trajectory=target_trajectory)
    planner.reset()
    target_replay_mode = _target_replay_mode(args)
    if target_trajectory is not None and target_replay_mode == "path_goal":
        start_recorded_path_navigation(env, args, setup, target_trajectory)
    configure_deterministic_clock(env, args)
    frames_drone: list[np.ndarray] = []
    frames_dog: list[np.ndarray] = []
    frames_global: list[np.ndarray] = []
    drone_infos: list[dict[str, Any]] = []
    dog_infos: list[dict[str, Any]] = []
    combined_infos: list[dict[str, Any]] = []
    recent_target_positions: list[np.ndarray] = []
    lost_count = 0
    failure_count = 0
    collision = False
    status = "Normal"
    t0 = time.monotonic()
    last_info = None
    previous_observation_wall_time: Optional[float] = None
    planner_debug_steps = max(0, int(getattr(args, "planner_debug_steps", 0) or 0))

    # Recorded path length no longer terminates the episode immediately. As in
    # Habitat, the target can stop at its final goal while the followers keep
    # receiving and executing model actions for a short observation window.
    episode_max_steps = args.max_steps

    for step_idx in range(episode_max_steps):
        if bool(setup.get("target_stopped", False)):
            update_habitat_target_stop_state(env, args, setup, step_idx=step_idx)
        elif target_trajectory is not None and target_replay_mode == "pose":
            current_pose = target_trajectory["poses"][min(step_idx, len(target_trajectory["poses"]) - 1)]
            _set_recorded_target_pose(env, setup, current_pose)
            update_observation(env, refresh_cameras=False)
        elif target_trajectory is not None and target_replay_mode == "path_goal":
            update_recorded_path_navigation(env, args, setup)
        target_stopped = update_habitat_target_stop_state(
            env,
            args,
            setup,
            step_idx=step_idx,
            recorded_pose_exhausted=bool(
                target_trajectory is not None
                and target_replay_mode == "pose"
                and step_idx >= len(target_trajectory["poses"]) - 1
            ),
        )
        target_pose_before = list(env.unwrapped.obj_poses[setup["target_id"]])
        if args.face_target_before_step:
            # Optional oracle camera/body heading. Disabled by default because it
            # uses the simulator target pose and can hide yaw-control problems.
            set_ground_yaw(
                env,
                setup["robotdog_name"],
                heading_deg(pose_xyz(list(env.unwrapped.obj_poses[setup["robotdog_id"]])), pose_xyz(target_pose_before)),
            )
            maybe_set_drone_yaw(
                env,
                setup["drone_name"],
                heading_deg(pose_xyz(list(env.unwrapped.obj_poses[setup["drone_id"]])), pose_xyz(target_pose_before)),
            )

        _lock_eval_cameras(env, setup, args)
        drone_input, dog_input = _read_agent_pair(env, setup, args)
        observation_wall_time = time.monotonic()
        if previous_observation_wall_time is None:
            # No measured period exists on the first iteration. Start at the
            # longest trained horizon because this synchronous evaluator holds
            # the command until the next roughly one-second model update.
            realtime_control_period_seconds = float(args.realtime_waypoint_max_seconds)
        else:
            realtime_control_period_seconds = (
                observation_wall_time - previous_observation_wall_time
            )
        previous_observation_wall_time = observation_wall_time
        drone_input_frame, drone_input_vis, drone_input_visible, drone_input_bbox, drone_bbox_norm = drone_input
        dog_input_frame, dog_input_vis, dog_input_visible, dog_input_bbox, dog_bbox_norm = dog_input

        # 每一步调用模型推理
        gt_bbox_prior = args.bbox_source == "ground_truth"
        pred = planner.predict(
            drone_input_frame,
            dog_input_frame,
            drone_bbox_norm if gt_bbox_prior else None,
            dog_bbox_norm if gt_bbox_prior else None,
            args.instruction,
            joint_instruction=getattr(args, "joint_instruction", None),
            agent1_instruction=getattr(args, "agent1_instruction", None),
            agent2_instruction=getattr(args, "agent2_instruction", None),
            observation_time=observation_wall_time,
        )
        observation_to_action_seconds = time.monotonic() - observation_wall_time
        drone_action, dog_action, action_debug = planner.waypoints_to_actions(
            pred["waypoints"],
            realtime_control_period_seconds=realtime_control_period_seconds,
        )
        predicted_bbox = pred["refined_bbox"]
        bbox_fallback = pred.get("bbox_fallback_to_absolute") or [False, False]
        visible_score = pred.get("visible_score") or [0.0, 0.0]
        best_candidate = pred.get("best_candidate") or [None, None]
        best_candidate_score = pred.get("best_candidate_score") or [None, None]
        drone_bbox_iou = _bbox_iou_cxcywh(predicted_bbox[0], drone_bbox_norm)
        dog_bbox_iou = _bbox_iou_cxcywh(predicted_bbox[1], dog_bbox_norm)

        actions = [None for _ in env.unwrapped.player_list]
        actions[setup["drone_id"]] = drone_action
        actions[setup["robotdog_id"]] = dog_action
        drone_pose_before_action = list(env.unwrapped.obj_poses[setup["drone_id"]])
        dog_pose_before_action = list(env.unwrapped.obj_poses[setup["robotdog_id"]])
        target_pose_before_action = list(env.unwrapped.obj_poses[setup["target_id"]])
        pulse_recorded_target = bool(
            target_trajectory is not None and target_replay_mode == "path_goal"
        )
        if pulse_recorded_target and not target_stopped:
            # Habitat applies the human oracle action in the same simulator step
            # as the follower action, after both policies consumed frame 0.
            resume_recorded_path_navigation(env, setup)
        if bool(getattr(args, "deterministic_step", True)):
            _obs, _rewards, done, last_info, pulse_wall_seconds = deterministic_data_collection_step(
                env,
                args,
                setup,
                drone_action,
                dog_action,
            )
        else:
            pulse_t0 = time.monotonic()
            _obs, _rewards, done, last_info = data_collection_step(env, actions)
            pulse_wall_seconds = time.monotonic() - pulse_t0
        drone_pose_after_action = list(env.unwrapped.obj_poses[setup["drone_id"]])
        dog_pose_after_action = list(env.unwrapped.obj_poses[setup["robotdog_id"]])
        target_pose_after_action = list(env.unwrapped.obj_poses[setup["target_id"]])

        if (
            target_trajectory is not None
            and target_replay_mode == "pose"
            and not target_stopped
        ):
            next_pose = target_trajectory["poses"][min(step_idx + 1, len(target_trajectory["poses"]) - 1)]
            _set_recorded_target_pose(env, setup, next_pose)
            update_observation(env, refresh_cameras=False)

        _lock_eval_cameras(env, setup, args)
        drone_after, dog_after = _read_agent_pair(env, setup, args)
        drone_frame, drone_vis, drone_visible, drone_bbox, drone_bbox_norm_after = drone_after
        dog_frame, dog_vis, dog_visible, dog_bbox, dog_bbox_norm_after = dog_after
        drone_pose = list(env.unwrapped.obj_poses[setup["drone_id"]])
        dog_pose = list(env.unwrapped.obj_poses[setup["robotdog_id"]])
        target_pose = list(env.unwrapped.obj_poses[setup["target_id"]])

        drone_dist = distance_xy_m(drone_pose, target_pose)
        dog_dist = distance_xy_m(dog_pose, target_pose)
        drone_collision = drone_collision_from_info(last_info, setup["drone_id"], setup["target_id"], drone_dist, drone_pose, target_pose)
        dog_collision = robotdog_collision_from_info(last_info, setup["robotdog_id"], setup["target_id"], dog_dist, dog_pose, target_pose)
        collision = collision or drone_collision or dog_collision

        drone_following = bool(drone_visible and drone_dist <= args.drone_success_distance)
        dog_following = bool(dog_visible and dog_dist <= args.robotdog_success_distance)
        joint_following = bool(drone_following and dog_following)
        if joint_following:
            lost_count = 0
            failure_count = 0
        else:
            too_far = drone_dist > args.drone_lost_distance or dog_dist > args.robotdog_lost_distance
            # Legacy OpenTrackVLA declares Lost only after consecutive
            # out-of-range frames. Temporary invisibility lowers TR but does
            # not itself advance the Lost counter.
            lost_count = lost_count + 1 if too_far else 0
            if step_idx + 1 > args.failure_warmup_steps:
                failure_count += 1

        global_frame = get_global_frame(env, args, target_pose, dog_pose, drone_pose)
        drone_vis_frame = drone_input_frame
        dog_vis_frame = dog_input_frame
        if args.trajectory_overlay:
            drone_vis_frame = _render_bgr_frame_with_traj(
                drone_vis_frame, pred["waypoints"][0], scale=args.trajectory_scale
            )
            dog_vis_frame = _render_bgr_frame_with_traj(
                dog_vis_frame, pred["waypoints"][1], scale=args.trajectory_scale
            )
        drone_vis_frame = _overlay_text(
            _draw_predicted_bbox(drone_vis_frame, predicted_bbox[0], "model bbox"),
            [
                f"ep={episode_id} step={step_idx + 1}",
                f"drone d={drone_dist:.2f} bbox_iou={drone_bbox_iou:.2f}",
                f"bbox_source={args.bbox_source} vis={float(visible_score[0]):.2f} abs_fallback={int(bbox_fallback[0])}",
                _candidate_label(best_candidate[0], best_candidate_score[0]),
                f"a=[{drone_action[0]:.2f},{drone_action[1]:.2f},{drone_action[3]:.2f}]",
            ],
        )
        dog_vis_frame = _overlay_text(
            _draw_predicted_bbox(dog_vis_frame, predicted_bbox[1], "model bbox"),
            [
                f"ep={episode_id} step={step_idx + 1}",
                f"dog d={dog_dist:.2f} bbox_iou={dog_bbox_iou:.2f}",
                f"bbox_source={args.bbox_source} vis={float(visible_score[1]):.2f} abs_fallback={int(bbox_fallback[1])}",
                _candidate_label(best_candidate[1], best_candidate_score[1]),
                f"a=[turn {dog_action[0]:.1f}, speed {dog_action[1]:.1f}]",
            ],
        )
        frames_drone.append(drone_vis_frame)
        frames_dog.append(dog_vis_frame)
        if args.write_global_video and global_frame is not None:
            frames_global.append(global_frame)

        drone_infos.append(
            {
                "step": step_idx + 1,
                "dis_to_human": float(drone_dist),
                "dis_to_human_3d": float(distance_m(drone_pose, target_pose)),
                "facing": 1.0 if drone_visible else 0.0,
                "target_visible": bool(drone_visible),
                "target_visibility": float(drone_vis),
                "target_bbox": drone_bbox,
                "bbox_feat": drone_bbox_norm_after,
                "model_bbox_input": pred.get("bbox_input", [None, None])[0] if pred.get("bbox_input") else None,
                "predicted_bbox": predicted_bbox[0],
                "raw_refined_bbox": pred.get("raw_refined_bbox", [None, None])[0],
                "absolute_bbox": pred.get("absolute_bbox", [None, None])[0],
                "bbox_fallback_to_absolute": bool(bbox_fallback[0]),
                "bbox_iou": float(drone_bbox_iou),
                "predicted_visible_score": float(visible_score[0]),
                "input_target_visible": bool(drone_input_visible),
                "input_target_visibility": float(drone_input_vis),
                "predicted_waypoints": pred["waypoints"][0].tolist(),
                "best_candidate": best_candidate[0],
                "best_candidate_score": best_candidate_score[0],
                "target_center_error": bbox_center_error(drone_bbox, args),
                "target_centered": bool(drone_visible and bbox_centered(drone_bbox, args)),
                "base_velocity": action_debug["drone_velocity_pred"],
                "drone_action": [float(v) for v in drone_action],
                "action_pulse_displacement_m": float(
                    distance_xy_m(drone_pose_before_action, drone_pose_after_action)
                ),
                "action_pulse_wall_seconds": float(pulse_wall_seconds),
                "following": bool(drone_following),
                "collision": bool(drone_collision),
                "drone_pose": drone_pose,
                "target_pose": target_pose,
            }
        )
        dog_infos.append(
            {
                "step": step_idx + 1,
                "dis_to_human": float(dog_dist),
                "dis_to_human_3d": float(distance_m(dog_pose, target_pose)),
                "facing": 1.0 if dog_visible else 0.0,
                "target_visible": bool(dog_visible),
                "target_visibility": float(dog_vis),
                "target_bbox": dog_bbox,
                "bbox_feat": dog_bbox_norm_after,
                "model_bbox_input": pred.get("bbox_input", [None, None])[1] if pred.get("bbox_input") else None,
                "predicted_bbox": predicted_bbox[1],
                "raw_refined_bbox": pred.get("raw_refined_bbox", [None, None])[1],
                "absolute_bbox": pred.get("absolute_bbox", [None, None])[1],
                "bbox_fallback_to_absolute": bool(bbox_fallback[1]),
                "bbox_iou": float(dog_bbox_iou),
                "predicted_visible_score": float(visible_score[1]),
                "input_target_visible": bool(dog_input_visible),
                "input_target_visibility": float(dog_input_vis),
                "predicted_waypoints": pred["waypoints"][1].tolist(),
                "best_candidate": best_candidate[1],
                "best_candidate_score": best_candidate_score[1],
                "target_center_error": bbox_center_error(dog_bbox, args),
                "target_centered": bool(dog_visible and bbox_centered(dog_bbox, args)),
                "base_velocity": action_debug["robotdog_velocity_pred"],
                "ground_action": [float(v) for v in dog_action],
                "action_pulse_displacement_m": float(
                    distance_xy_m(dog_pose_before_action, dog_pose_after_action)
                ),
                "action_pulse_wall_seconds": float(pulse_wall_seconds),
                "following": bool(dog_following),
                "collision": bool(dog_collision),
                "robotdog_lateral_ignored": action_debug["robotdog_lateral_ignored"],
                "robotdog_pose": dog_pose,
                "target_pose": target_pose,
            }
        )
        combined_infos.append(
            {
                "step": step_idx + 1,
                "joint_following": joint_following,
                "drone_following": bool(drone_following),
                "robotdog_following": bool(dog_following),
                "collision": bool(drone_collision or dog_collision),
                "lost_count": int(lost_count),
                "failure_count": int(failure_count),
                "visible_score": pred.get("visible_score"),
                "refined_bbox": pred.get("refined_bbox"),
                "best_candidate": pred.get("best_candidate"),
                "best_candidate_score": pred.get("best_candidate_score"),
                "bbox_source": args.bbox_source,
                "bbox_input": pred.get("bbox_input"),
                "bbox_fallback_to_absolute": bbox_fallback,
                "drone_bbox_iou": float(drone_bbox_iou),
                "robotdog_bbox_iou": float(dog_bbox_iou),
                "drone_visible_correct": bool((float(visible_score[0]) >= 0.5) == bool(drone_input_visible)),
                "robotdog_visible_correct": bool((float(visible_score[1]) >= 0.5) == bool(dog_input_visible)),
                "action_debug": action_debug,
                "target_action_pulse_displacement_m": float(
                    distance_xy_m(target_pose_before_action, target_pose_after_action)
                ),
                "action_pulse_wall_seconds": float(pulse_wall_seconds),
                "observation_to_action_seconds": float(observation_to_action_seconds),
                "realtime_control_period_seconds": float(realtime_control_period_seconds),
                "deterministic_step": bool(getattr(args, "deterministic_step", True)),
                "fixed_timestep_seconds": float(args.dt),
                "target_motion_mode": setup["target_motion_mode"],
                "recorded_target_frame": step_idx + 1 if target_trajectory is not None else None,
                "target_stopped": bool(target_stopped),
                "target_stop_wait_count": int(setup.get("target_stop_wait_count", 0)),
                "target_stop_wait_steps": int(setup["target_stop_wait_steps"]),
            }
        )

        if planner_debug_steps > 0 and step_idx < planner_debug_steps:
            debug_dir = Path(args.save_path) / f"seed_{args.seed}" / safe_slug(args.env_id)
            debug_dir.mkdir(parents=True, exist_ok=True)
            debug_record = {
                "episode_id": int(episode_id),
                "step": int(step_idx + 1),
                "agent_order": ["drone", "robotdog"],
                "joint_instruction": getattr(args, "joint_instruction", None) or args.instruction,
                "agent1_instruction": getattr(args, "agent1_instruction", None),
                "agent2_instruction": getattr(args, "agent2_instruction", None),
                "planner_binding": {
                    "drone": "planner_agent1 -> waypoints[0]",
                    "robotdog": "planner_agent2 -> waypoints[1]",
                },
                "drone_waypoint_index": int(action_debug["drone_waypoint_index"]),
                "robotdog_waypoint_index": int(action_debug["robotdog_waypoint_index"]),
                "drone_waypoint": pred["waypoints"][0, action_debug["drone_waypoint_index"], :3].tolist(),
                "robotdog_waypoint": pred["waypoints"][1, action_debug["robotdog_waypoint_index"], :3].tolist(),
                "drone_action": [float(v) for v in drone_action],
                "robotdog_action": [float(v) for v in dog_action],
                "drone_velocity_pred": action_debug["drone_velocity_pred"],
                "robotdog_velocity_pred": action_debug["robotdog_velocity_pred"],
                "drone_distance_m": float(drone_dist),
                "robotdog_distance_m": float(dog_dist),
                "drone_action_pulse_displacement_m": float(
                    distance_xy_m(drone_pose_before_action, drone_pose_after_action)
                ),
                "robotdog_action_pulse_displacement_m": float(
                    distance_xy_m(dog_pose_before_action, dog_pose_after_action)
                ),
                "target_action_pulse_displacement_m": float(
                    distance_xy_m(target_pose_before_action, target_pose_after_action)
                ),
                "action_pulse_wall_seconds": float(pulse_wall_seconds),
                "observation_to_action_seconds": float(observation_to_action_seconds),
                "realtime_control_period_seconds": float(realtime_control_period_seconds),
                "deterministic_step": bool(getattr(args, "deterministic_step", True)),
                "fixed_timestep_seconds": float(args.dt),
                "drone_visible": bool(drone_visible),
                "robotdog_visible": bool(dog_visible),
                "joint_following": bool(joint_following),
                "action_debug": action_debug,
            }
            with (debug_dir / "planner_debug.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(debug_record, ensure_ascii=False) + "\n")

        if args.debug_motion and (step_idx < 5 or (step_idx + 1) % 10 == 0):
            print(
                f"[eval] ep={episode_id} step={step_idx + 1} "
                f"drone_d={drone_dist:.2f} dog_d={dog_dist:.2f} "
                f"drone_vis={int(drone_visible)} dog_vis={int(dog_visible)} "
                f"lost={lost_count} failure={failure_count}",
                flush=True,
            )

        recent_target_positions.append(pose_xyz(target_pose))
        if target_stopped:
            setup["target_stop_wait_count"] = int(setup.get("target_stop_wait_count", 0)) + 1
        if collision:
            status = "Collision"
            break
        if args.max_lost_steps > 0 and lost_count >= args.max_lost_steps:
            status = "Lost"
            print(
                f"[early-stop] ep={episode_id} step={step_idx + 1} status={status} "
                f"lost_count={lost_count}>={args.max_lost_steps}",
                flush=True,
            )
            break
        if args.max_failure_steps > 0 and failure_count >= args.max_failure_steps:
            status = "PersistentFailure"
            print(
                f"[early-stop] ep={episode_id} step={step_idx + 1} status={status} "
                f"failure_count={failure_count}>={args.max_failure_steps}",
                flush=True,
            )
            break
        if args.max_episode_seconds > 0 and time.monotonic() - t0 >= args.max_episode_seconds:
            status = "Timeout"
            print(
                f"[early-stop] ep={episode_id} step={step_idx + 1} status={status} "
                f"elapsed={time.monotonic() - t0:.1f}s>={args.max_episode_seconds:.1f}s",
                flush=True,
            )
            break
        if (
            target_stopped
            and int(setup["target_stop_wait_count"]) >= int(setup["target_stop_wait_steps"])
        ):
            status = "TargetStopped"
            break
        if done:
            status = "EnvDone"
            break

    elapsed = max(time.monotonic() - t0, 1e-6)
    total_step = len(combined_infos)
    drone_rate = sum(1 for item in drone_infos if item["following"]) / max(total_step, 1)
    dog_rate = sum(1 for item in dog_infos if item["following"]) / max(total_step, 1)
    joint_rate = sum(1 for item in combined_infos if item["joint_following"]) / max(total_step, 1)
    drone_bbox_iou_mean = sum(item["drone_bbox_iou"] for item in combined_infos) / max(total_step, 1)
    dog_bbox_iou_mean = sum(item["robotdog_bbox_iou"] for item in combined_infos) / max(total_step, 1)
    visible_correct = sum(
        int(item["drone_visible_correct"]) + int(item["robotdog_visible_correct"])
        for item in combined_infos
    )
    visible_accuracy = visible_correct / max(total_step * 2, 1)

    # Match the legacy OpenTrackVLA terminal-success semantics. Per-step TR
    # remains upper-bound-only, while an episode that ends early must finish
    # inside both agents' configured follow-distance bands. A full-horizon
    # episode falls back to the final per-step joint-following result.
    completed_full_horizon = bool(total_step >= episode_max_steps)
    final_drone_distance = float(drone_infos[-1]["dis_to_human"]) if drone_infos else float("inf")
    final_dog_distance = float(dog_infos[-1]["dis_to_human"]) if dog_infos else float("inf")
    final_drone_following = bool(drone_infos[-1]["following"]) if drone_infos else False
    final_dog_following = bool(dog_infos[-1]["following"]) if dog_infos else False
    final_joint_following = bool(final_drone_following and final_dog_following)
    final_drone_in_range = bool(
        args.drone_min_follow_dist <= final_drone_distance <= args.drone_max_follow_dist
    )
    final_dog_in_range = bool(
        args.robotdog_min_follow_dist <= final_dog_distance <= args.robotdog_max_follow_dist
    )
    final_joint_in_range = bool(final_drone_in_range and final_dog_in_range)
    terminal_following_success = bool(
        final_joint_following
        and (completed_full_horizon or final_joint_in_range)
    )
    success = bool(
        (not collision)
        and status not in {"Lost", "Collision", "PersistentFailure", "Timeout"}
        and (completed_full_horizon or total_step >= max(1, args.min_success_steps))
        and terminal_following_success
    )
    if success:
        status = "Success"
    elif status == "Normal":
        status = "Failed"

    stat = {
        "finish": bool(
            total_step >= episode_max_steps
            or status in {"Success", "Lost", "Collision", "PersistentFailure", "Timeout", "EnvDone"}
        ),
        "status": status,
        "success": 1.0 if success else 0.0,
        "total_step": total_step,
        "collision": 1.0 if collision else 0.0,
        "joint_following_rate": float(joint_rate),
        "drone_following_rate": float(drone_rate),
        "robotdog_following_rate": float(dog_rate),
        "completed_full_horizon": completed_full_horizon,
        "final_drone_distance": final_drone_distance,
        "final_robotdog_distance": final_dog_distance,
        "final_drone_following": final_drone_following,
        "final_robotdog_following": final_dog_following,
        "final_joint_following": final_joint_following,
        "final_drone_in_range": final_drone_in_range,
        "final_robotdog_in_range": final_dog_in_range,
        "final_joint_in_range": final_joint_in_range,
        "terminal_following_success": terminal_following_success,
        "drone_bbox_iou_mean": float(drone_bbox_iou_mean),
        "robotdog_bbox_iou_mean": float(dog_bbox_iou_mean),
        "visible_accuracy": float(visible_accuracy),
        "final_lost_count": int(lost_count),
        "final_failure_count": int(failure_count),
        "max_lost_steps": int(args.max_lost_steps),
        "max_failure_steps": int(args.max_failure_steps),
        "failure_warmup_steps": int(args.failure_warmup_steps),
        "max_episode_seconds": float(args.max_episode_seconds),
        "bbox_source": args.bbox_source,
        "ckpt_bbox_dropout_prob": float(planner.ckpt_bbox_dropout_prob),
        "instruction": args.instruction,
        "joint_instruction": getattr(args, "joint_instruction", None) or args.instruction,
        "agent1_instruction": getattr(args, "agent1_instruction", None),
        "agent2_instruction": getattr(args, "agent2_instruction", None),
        "fps": total_step / elapsed,
        "ckpt": str(planner.ckpt_path),
        "env_id": args.env_id,
        "model_type": "multi_agent",
        "target_motion_mode": setup["target_motion_mode"],
        "target_stopped": bool(setup.get("target_stopped", False)),
        "target_stop_step": setup.get("target_stop_step"),
        "target_stop_wait_count": int(setup.get("target_stop_wait_count", 0)),
        "target_stop_wait_steps": int(setup["target_stop_wait_steps"]),
        "target_goal_reach_distance": float(args.target_goal_reach_distance),
        "action_pulse_control": bool(getattr(args, "deterministic_step", True)),
        "deterministic_step": bool(getattr(args, "deterministic_step", True)),
        "fixed_timestep_seconds": (
            float(args.dt) if bool(getattr(args, "deterministic_step", True)) else None
        ),
        "realtime_waypoint_timing": bool(
            getattr(args, "realtime_waypoint_timing", False)
        ),
        "history_frame_dt": float(planner.history_frame_dt),
        "realtime_waypoint_timing_source": (
            "previous_observation_interval_wall_clock"
            if bool(getattr(args, "realtime_waypoint_timing", False))
            else None
        ),
        "deterministic_step_backend": (
            "ue_fixed_timestep_resume_pause"
            if bool(getattr(args, "deterministic_step", True))
            else "ue_realtime_continuous"
        ),
        "recorded_target_source": setup["recorded_target_source"],
        "recorded_target_episode": setup["recorded_target_episode"],
        "success_rule": {
            "full_horizon": "final_joint_following",
            "early_end": "final_joint_following_and_both_agents_in_follow_range",
            "drone_final_range": [
                float(args.drone_min_follow_dist),
                float(args.drone_max_follow_dist),
            ],
            "robotdog_final_range": [
                float(args.robotdog_min_follow_dist),
                float(args.robotdog_max_follow_dist),
            ],
            "joint_following_rate_diagnostic_only": True,
            "early_end_min_success_steps": args.min_success_steps,
            "collision_required": False,
        },
    }
    return {
        "episode_id": str(episode_id),
        "setup": setup,
        "stat": stat,
        "drone_infos": drone_infos,
        "robotdog_infos": dog_infos,
        "combined_infos": combined_infos,
        "frames_drone": frames_drone,
        "frames_robotdog": frames_dog,
        "frames_global": frames_global,
    }


# ----------------------- 结果保存与命令行入口 -----------------------

def write_episode_outputs(args: argparse.Namespace, result: dict[str, Any]) -> None:
    scene_dir = Path(args.save_path) / f"seed_{args.seed}" / safe_slug(args.env_id)
    scene_dir.mkdir(parents=True, exist_ok=True)
    episode_id = result["episode_id"]
    write_json(scene_dir / f"{episode_id}.json", result["stat"])
    write_json(scene_dir / f"{episode_id}_drone_info.json", result["drone_infos"])
    write_json(scene_dir / f"{episode_id}_robotdog_info.json", result["robotdog_infos"])
    write_json(scene_dir / f"{episode_id}_combined_info.json", result["combined_infos"])
    write_json(scene_dir / f"{episode_id}_setup.json", result["setup"])
    if args.save_video:
        save_mp4(result["frames_drone"], scene_dir / f"{episode_id}_drone.mp4", args.fps)
        save_mp4(result["frames_robotdog"], scene_dir / f"{episode_id}_robotdog.mp4", args.fps)
        if result["frames_global"]:
            save_mp4(result["frames_global"], scene_dir / f"{episode_id}_global.mp4", args.fps)
    print(f"[write] {scene_dir / f'{episode_id}.json'}", flush=True)


def reset_planner_debug_file(args: argparse.Namespace) -> None:
    if max(0, int(getattr(args, "planner_debug_steps", 0) or 0)) <= 0:
        return
    debug_path = Path(args.save_path) / f"seed_{args.seed}" / safe_slug(args.env_id) / "planner_debug.jsonl"
    if debug_path.exists():
        debug_path.unlink()
    print(f"[init] planner debug will be written to {debug_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MultiAgentOpenTrackVLA in UnrealZoo.")
    parser.add_argument("--ckpt", required=True, help="Multi-agent checkpoint file or directory containing model_epoch*.pt.")
    parser.add_argument("--save-path", default="/data/hdt/ntv_data/sim_data/eval/unrealzoo_multi_agent")
    parser.add_argument("--env-id", default=DEFAULT_ENV_ID)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument(
        "--recorded-target-dir",
        default=None,
        help="Replay only target_pose from *_drone_info.json while agents and observations remain online closed-loop.",
    )
    parser.add_argument(
        "--recorded-target-episodes",
        default=None,
        help="Optional comma-separated recorded episode names, for example: 0,1,2.",
    )
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument(
        "--render-gpu",
        type=int,
        default=None,
        help="Physical GPU index used by Unreal Engine Vulkan rendering; independent of CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument(
        "--joint-instruction",
        default=None,
        help="Optional global joint task instruction; defaults to --instruction.",
    )
    parser.add_argument(
        "--agent1-instruction",
        default=None,
        help="Optional agent1/drone-specific instruction.",
    )
    parser.add_argument(
        "--agent2-instruction",
        default=None,
        help="Optional agent2/robotdog-specific instruction.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--llm-name", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--vision-feat-dim", type=int, default=1536)
    parser.add_argument("--n-waypoints", type=int, default=8)
    parser.add_argument("--history", type=int, default=31)
    parser.add_argument(
        "--history-frame-dt",
        type=float,
        default=0.1,
        help="Training-frame interval used to resample sparse realtime history tokens.",
    )
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--alpha-xy", type=float, default=1.0)
    parser.add_argument("--use-angle-tvi", action="store_true")
    parser.add_argument("--no-tanh-actions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--diffusion-num-anchors", type=int, default=40)
    parser.add_argument("--diffusion-hidden-dim", type=int, default=768)
    parser.add_argument("--diffusion-depth", type=int, default=12)
    parser.add_argument("--diffusion-num-heads", type=int, default=12)
    parser.add_argument(
        "--diffusion-deterministic-inference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use zero noise at diffusion inference for reproducible trajectories.",
    )
    parser.add_argument(
        "--bbox-source",
        choices=["model", "ground_truth", "none"],
        default="model",
        help="model: first-frame detection then recurrent predicted-box tracking; ground_truth: simulator bbox prior; none: no bbox prior on every frame.",
    )
    parser.add_argument(
        "--bbox-min-size",
        type=float,
        default=0.01,
        help="Fallback to the model absolute bbox head when recurrent refined bbox width/height is smaller than this.",
    )

    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument(
        "--deterministic-step",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pause during inference and advance one UE fixed-timestep render frame per action.",
    )
    parser.add_argument(
        "--realtime-waypoint-timing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Keep simulation running during inference and select the waypoint nearest to the "
            "previous wall-clock observation interval, which estimates how long the next command "
            "will be held. Requires --no-deterministic-step."
        ),
    )
    parser.add_argument(
        "--realtime-waypoint-min-seconds",
        type=float,
        default=0.1,
        help="Lower clamp for realtime observation-to-action waypoint mapping.",
    )
    parser.add_argument(
        "--realtime-waypoint-max-seconds",
        type=float,
        default=0.9,
        help="Upper clamp for realtime observation-to-action waypoint mapping.",
    )
    parser.add_argument("--offscreen", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--time-dilation", type=int, default=-1)
    parser.add_argument("--disable-ue-input", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--launch-retries", type=int, default=5)
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write-global-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trajectory-overlay", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trajectory-scale", type=float, default=120.0, help="Trajectory overlay pixels per meter.")
    parser.add_argument("--top-view-height", type=float, default=None)
    parser.add_argument("--debug-motion", action="store_true")
    parser.add_argument(
        "--planner-debug-steps",
        type=int,
        default=5,
        help="Append planner/action binding debug JSONL for the first N steps of each episode; 0 disables.",
    )
    parser.add_argument("--monitor", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--monitor-interval", type=int, default=2)
    parser.add_argument("--monitor-scale", type=float, default=0.7)
    parser.add_argument("--brightness-scale", type=float, default=1.0)
    parser.add_argument("--brightness-offset", type=float, default=0.0)
    parser.add_argument("--brightness-config", type=Path, default=None)

    parser.add_argument("--human-speed", type=float, default=90.0)
    parser.add_argument("--human-turn", type=float, default=5.0)
    parser.add_argument("--human-reverse-scale", type=float, default=0.5)
    parser.add_argument("--human-goal-min-distance", type=float, default=700.0)
    parser.add_argument("--human-goal-max-distance", type=float, default=2200.0)
    parser.add_argument("--human-path-file", type=Path, default=None)
    parser.add_argument("--human-path-loop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--human-waypoint-reach-distance", type=float, default=150.0)
    parser.add_argument("--human-waypoint-stall-window", type=int, default=20)
    parser.add_argument("--human-waypoint-stall-distance", type=float, default=20.0)
    parser.add_argument("--keyboard-human", action="store_true")
    parser.add_argument("--open-spawn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--open-spawn-radius", type=float, default=900.0)
    parser.add_argument("--min-open-clearance", type=float, default=300.0)
    parser.add_argument("--open-spawn-candidates", type=int, default=128)
    parser.add_argument("--ground-navmesh-tolerance", type=float, default=300.0)
    parser.add_argument("--drone-navmesh-tolerance", type=float, default=800.0)
    parser.add_argument("--require-visual-target", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require-centered-target", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-mask-visibility", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-visible-ratio", type=float, default=0.001)
    parser.add_argument("--target-center-tolerance", type=float, default=0.35)

    parser.add_argument("--human-appearance-min", type=int, default=1)
    parser.add_argument("--human-appearance-max", type=int, default=18)
    parser.add_argument("--robotdog-appearance-min", type=int, default=20)
    parser.add_argument("--robotdog-appearance-max", type=int, default=33)

    parser.add_argument("--robotdog-ideal-follow-dist", type=float, default=6.25)
    # 狗的最大最小跟踪距离
    parser.add_argument("--robotdog-min-follow-dist", type=float, default=4.5)
    parser.add_argument("--robotdog-max-follow-dist", type=float, default=8.0)
    parser.add_argument("--robotdog-max-speed", type=float, default=1.05)
    parser.add_argument("--robotdog-max-lateral-speed", type=float, default=0.45)
    parser.add_argument("--robotdog-max-yaw-rate", type=float, default=1.0)
    parser.add_argument("--robotdog-max-turn-deg", type=float, default=30.0)
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

    parser.add_argument("--drone-ideal-follow-dist", type=float, default=4.5)
    # 无人机的最大最小跟踪距离
    parser.add_argument("--drone-min-follow-dist", type=float, default=2.5)
    parser.add_argument("--drone-max-follow-dist", type=float, default=6.5)
    parser.add_argument("--drone-height", type=float, default=600.0)
    parser.add_argument("--drone-max-speed", type=float, default=0.12)
    parser.add_argument("--drone-max-vx", type=float, default=0.12)
    parser.add_argument("--drone-max-vy", type=float, default=0.05)
    parser.add_argument("--drone-max-yaw-rate", type=float, default=1.0)
    parser.add_argument("--drone-camera-fixed-pitch", type=float, default=-60.0)
    parser.add_argument("--drone-camera-pitches", default="-60")
    parser.add_argument("--drone-camera-fixed-yaw", type=float, default=0.0)
    parser.add_argument("--drone-camera-yaw-offsets", default="0")
    parser.add_argument("--drone-camera-mode", choices=["fixed", "oracle"], default="fixed")
    parser.add_argument("--lock-drone-camera-world-xy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--drone-camera-z-offset", type=float, default=0.0)
    parser.add_argument("--drone-fov", type=float, default=100.0)
    parser.add_argument("--max-camera-search-candidates", type=int, default=12)
    parser.add_argument("--snap-heading", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--follow-behind", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--face-target-before-step", action="store_true")
    parser.add_argument(
        "--init-from-recorded-agent-poses",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For recorded target eval, restore drone/robotdog first-frame poses and cameras when present.",
    )

    parser.add_argument("--waypoint-index", type=int, default=9)
    parser.add_argument("--drone-waypoint-index", type=int, default=9)
    parser.add_argument("--robotdog-waypoint-index", type=int, default=9)
    parser.add_argument(
        "--waypoint-horizon-steps",
        type=int,
        default=9,
        help="Future action steps integrated before resampling; must match the training data horizon.",
    )
    parser.add_argument("--drone-vx-scale", type=float, default=0.12)
    parser.add_argument("--drone-vy-scale", type=float, default=0.1)
    parser.add_argument("--drone-yaw-sign", type=float, default=1.0)
    parser.add_argument(
        "--drone-yaw-scale",
        type=float,
        default=3.0,
        help="Scale the predicted drone yaw rate before max-yaw-rate clipping.",
    )
    parser.add_argument("--robotdog-yaw-sign", type=float, default=1.0)
    parser.add_argument(
        "--robotdog-yaw-scale",
        type=float,
        default=1.0,
        help="Scale robotdog predicted yaw before converting to degrees and clipping.",
    )
    parser.add_argument("--robotdog-speed-gain", type=float, default=1.15)
    parser.add_argument("--drone-success-distance", type=float, default=6.5)
    parser.add_argument("--robotdog-success-distance", type=float, default=8.0)
    parser.add_argument("--drone-lost-distance", type=float, default=8.0)
    parser.add_argument("--robotdog-lost-distance", type=float, default=10.0)
    parser.add_argument(
        "--target-replay-mode",
        choices=["nav_goal", "pose", "path_goal"],
        default="path_goal",
        help="How to replay recorded target trajectories during closed-loop multi-agent eval.",
    )
    parser.add_argument("--target-path-min-spacing", type=float, default=100.0)
    parser.add_argument("--target-path-reach-distance", type=float, default=120.0)
    parser.add_argument(
        "--target-goal-reach-distance",
        type=float,
        default=50.0,
        help="Final target-goal threshold in Unreal units; 50 equals Habitat's 0.5 m.",
    )
    parser.add_argument("--target-stop-wait-min-steps", type=int, default=5)
    parser.add_argument("--target-stop-wait-max-steps", type=int, default=15)
    parser.add_argument("--max-lost-steps", type=int, default=20)
    parser.add_argument(
        "--max-failure-steps",
        type=int,
        default=0,
        help="Early stop after this many consecutive post-warmup steps without joint following; 0 disables.",
    )
    parser.add_argument(
        "--failure-warmup-steps",
        type=int,
        default=20,
        help="Do not count persistent joint-following failures during the first N steps.",
    )
    parser.add_argument(
        "--max-episode-seconds",
        type=float,
        default=0.0,
        help="Hard wall-clock timeout per episode in seconds; 0 disables.",
    )
    parser.add_argument("--success-rate-threshold", type=float, default=0.5)
    parser.add_argument("--min-success-steps", type=int, default=20)
    args = parser.parse_args()
    args.out_dir = Path(args.save_path)
    return args


def main() -> int:
    args = parse_args()
    if args.realtime_waypoint_timing and args.deterministic_step:
        raise ValueError("--realtime-waypoint-timing requires --no-deterministic-step")
    if args.realtime_waypoint_min_seconds <= 0.0:
        raise ValueError("--realtime-waypoint-min-seconds must be positive")
    if args.realtime_waypoint_max_seconds < args.realtime_waypoint_min_seconds:
        raise ValueError(
            "--realtime-waypoint-max-seconds must be >= --realtime-waypoint-min-seconds"
        )
    if args.deterministic_step:
        os.environ["UNREALZOO_FIXED_TIMESTEP"] = str(float(args.dt))
    else:
        os.environ.pop("UNREALZOO_FIXED_TIMESTEP", None)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    Path(args.save_path).mkdir(parents=True, exist_ok=True)
    reset_planner_debug_file(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    target_trajectories: Optional[list[dict[str, Any]]] = None
    if args.recorded_target_dir:
        target_trajectories = load_recorded_target_trajectories(
            Path(args.recorded_target_dir),
            _parse_episode_filter(args.recorded_target_episodes),
            env_id=args.env_id,
        )
        print(
            f"[init] loaded {len(target_trajectories)} recorded human trajectories "
            f"from {args.recorded_target_dir}; target_replay_mode={args.target_replay_mode}",
            flush=True,
        )
        if args.episodes > len(target_trajectories):
            print(
                f"[init] requested episodes={args.episodes}, trajectories={len(target_trajectories)}; "
                "recorded trajectories will be reused cyclically",
                flush=True,
            )
    print(f"[init] UnrealZoo env={args.env_id} ckpt={args.ckpt} save={args.save_path}", flush=True)
    planner = UnrealZooMultiAgentPlanner(args)
    print("[startup] creating UnrealZoo Gym environment (UE launches on first episode reset)", flush=True)
    env = make_env(args)
    print("[startup] UnrealZoo Gym environment created", flush=True)
    rng = random.Random(args.seed)
    saved = 0
    attempts = 0
    max_attempts = int(args.max_attempts) if args.max_attempts is not None else max(args.episodes * 3, args.episodes)
    try:
        while saved < args.episodes and attempts < max_attempts:
            attempts += 1
            episode_id = saved
            print(f"[episode {episode_id}] attempt={attempts}", flush=True)
            try:
                target_trajectory = (
                    target_trajectories[saved % len(target_trajectories)]
                    if target_trajectories is not None
                    else None
                )
                if target_trajectory is not None:
                    print(
                        f"[episode {episode_id}] replay human={target_trajectory['episode_name']} "
                        f"poses={len(target_trajectory['poses'])}",
                        flush=True,
                    )
                result = run_episode(
                    env,
                    args,
                    planner,
                    episode_id,
                    rng,
                    target_trajectory=target_trajectory,
                )
            except EpisodeSkipped as exc:
                print(f"[episode {episode_id}] skipped: {exc}", flush=True)
                continue
            write_episode_outputs(args, result)
            stat = result["stat"]
            print(
                f"[episode {episode_id}] status={stat['status']} success={stat['success']} "
                f"steps={stat['total_step']} joint_tr={stat['joint_following_rate']:.3f} "
                f"drone_tr={stat['drone_following_rate']:.3f} dog_tr={stat['robotdog_following_rate']:.3f} "
                f"bbox_iou=({stat['drone_bbox_iou_mean']:.3f},{stat['robotdog_bbox_iou_mean']:.3f})",
                flush=True,
            )
            saved += 1
    finally:
        env.close()
    print(f"[done] saved={saved} attempts={attempts} save_path={args.save_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
