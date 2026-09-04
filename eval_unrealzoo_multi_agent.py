#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AirGround-Coop V3 的 UnrealZoo 双 Agent 闭环运行时。

整体功能：
- 创建无人机、机器狗与行人的 UnrealZoo 场景，逐帧运行双 Agent闭环控制。
- 可只重放离线采集的人体世界坐标轨迹，两个 Agent 与后续 RGB 仍由在线仿真闭环产生。
- 模型与在线感知由 ``eval_airground_coop_v3.py`` 注入。
- 支持无真值 bbox 的完整检测/跟踪评估，并统计距离、可见性、碰撞和 bbox IoU。
- 将预测局部轨迹绘制为平滑彩色曲线，保存逐 episode 视频、JSON 与汇总指标。

关键函数：
- ``load_recorded_target_trajectories``：从采集 JSON 读取仅用于驱动人的世界坐标轨迹。
- ``setup_episode``：随机初始化目标，或使用 test 轨迹首帧初始化目标并将两个 Agent 放到人身后。
- ``run_episode``：执行单条闭环评估 episode。
- ``_render_bgr_frame_with_traj``：在 RGB/BGR 帧上绘制平滑预测轨迹。
- ``write_episode_outputs``：保存视频和结构化结果。
- ``parse_args`` / ``main``：配置并启动多 episode 评估。

主要输入输出：
- 输入为 V3 planner、UnrealZoo 环境、episode 数和控制参数。
- 输出位于 ``--save-path``，可由 ``python -m tools.calculate_unrealzoo_metrics`` 汇总。

"""

from __future__ import annotations

import argparse
import copy
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

from tools.cache_gridpool import (
    VisionCacheConfig,
    VisionFeatureCacher,
    crop_target_roi,
    grid_pool_tokens,
)
from tools.bbox_spatial import bbox_prompt_from_spatial, bbox_spatial_fields

ROI_VISUAL_LAYOUT_PROMPT = (
    "Visual layout: GLOBAL_HISTORY and GLOBAL_CURRENT encode scene geometry; "
    "TARGET_ROI encodes the target person's identity and local motion. Combine all three."
)


def prepend_roi_visual_layout_prompt(text: str) -> str:
    text = str(text or "").strip()
    if ROI_VISUAL_LAYOUT_PROMPT in text:
        return text
    return f"Task: {text}\n{ROI_VISUAL_LAYOUT_PROMPT}"


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

# 导入 gym_unrealcv 会注册 UnrealZoo 环境 ID。
import gym  # noqa: E402

# 旧版 Gym 的 entry-point loader 依赖已被 setuptools>=82 删除的 pkg_resources。
# 使用等价的 importlib loader，避免环境已成功注册却在 gym.make() 时失败。
try:  # noqa: SIM105
    import pkg_resources as _pkg_resources  # noqa: F401,E402
except ModuleNotFoundError:
    import importlib  # noqa: E402

    def _load_gym_entry_point(name: str):
        module_name, separator, attribute_path = str(name).partition(":")
        if not separator:
            raise ValueError(f"Invalid Gym entry point: {name!r}")
        value = importlib.import_module(module_name)
        for attribute in attribute_path.split("."):
            value = getattr(value, attribute)
        return value

    gym.envs.registration.load = _load_gym_entry_point

import gym_unrealcv  # noqa: F401,E402
print("[startup] UnrealZoo imports finished", flush=True)

from generate_aerial_ground_human_tracking_small import (  # noqa: E402
    DEFAULT_ENV_ID,
    DEFAULT_INSTRUCTION,
    HUMAN_SPEED_CHOICES_MPS as RECORDED_HUMAN_SPEED_CHOICES_MPS,
    agent_max_speed_for_human_speed as recorded_agent_max_speed_for_human_speed,
    normalize_speed_args,
    action_for_target_space,
    classify_coop_agents,
    capture_color_mask_snapshot,
    data_collection_step_pose_only,
    dog_args,
    drone_args,
    get_global_frame,
    make_env,
    place_initial_followers,
    reset_env,
)

# Evaluation also supports the slower target-speed stress case.  Keep this
# mapping local so changing the evaluation protocol does not alter collection
# defaults in UnrealZoo's recording scripts.
HUMAN_SPEED_CHOICES_MPS = (0.5,) + tuple(RECORDED_HUMAN_SPEED_CHOICES_MPS)


def agent_max_speed_for_human_speed(human_speed_mps: float) -> float:
    key = round(float(human_speed_mps), 3)
    if key in tuple(round(float(value), 3) for value in RECORDED_HUMAN_SPEED_CHOICES_MPS):
        return recorded_agent_max_speed_for_human_speed(human_speed_mps)
    # Evaluation target speed is allowed to be arbitrary. Unknown speeds use
    # the 0.9 m/s agent profile so target speed does not silently change limits.
    return 1.20
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
    manifest_items: dict[str, dict[str, Any]] = {}
    if source.is_file():
        with source.open("r", encoding="utf-8") as handle:
            source_obj = json.load(handle)
        if isinstance(source_obj, dict) and isinstance(source_obj.get("test"), list):
            input_root = Path(source_obj.get("input_root", source.parent)).expanduser()
            output_root = Path(source_obj.get("output_root", source.parent)).expanduser()
            source_roots = [
                Path(root).expanduser()
                for root in source_obj.get("source_roots", [])
            ]
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
                        *(root / f"{key}_drone_info.json" for root in source_roots),
                    ]
                    path = next(
                        (candidate for candidate in path_candidates if candidate.is_file()),
                        path_candidates[0],
                    )
                    candidates.append((path, key))
                    manifest_items[key] = dict(item)
                elif "info" in item:
                    info_rel = str(item["info"])
                    episode_name = str(item.get("relative_dir", Path(info_rel).parent)) + "/" + str(
                        item.get("stem", Path(info_rel).name.removesuffix("_info.json"))
                    )
                    path_candidates = [
                        Path(info_rel).expanduser(),
                        source.parent / "test_raw" / info_rel,
                        output_root / "test_raw" / info_rel,
                        input_root / info_rel,
                        *(root / info_rel for root in source_roots),
                    ]
                    path = next(
                        (candidate for candidate in path_candidates if candidate.is_file()),
                        path_candidates[0],
                    )
                    candidates.append((path, episode_name))
                    manifest_items[episode_name] = dict(item)
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
        manifest_item = manifest_items.get(episode_name, {})
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
        replay_meta: dict[str, Any] = {}
        replay_meta_path = manifest_item.get("replay_meta")
        if replay_meta_path:
            replay_meta_file = Path(str(replay_meta_path)).expanduser()
            if not replay_meta_file.is_file():
                raise FileNotFoundError(
                    f"replay_meta listed by manifest does not exist: {replay_meta_file}"
                )
            with replay_meta_file.open("r", encoding="utf-8") as handle:
                replay_meta = json.load(handle)
            distractor_poses = replay_meta.get("distractor_poses_per_frame") or []
            distractor_actions = replay_meta.get("distractor_actions_per_frame") or []
            if len(distractor_poses) < len(records) or len(distractor_actions) < len(records):
                raise ValueError(
                    f"{replay_meta_file}: replay distractor frames shorter than target "
                    f"trajectory ({len(distractor_poses)}/{len(distractor_actions)} vs {len(records)})"
                )
            expected_humans = int(replay_meta.get("human_count") or 0)
            appearance_ids = replay_meta.get("human_appearance_ids") or []
            if expected_humans and len(appearance_ids) != expected_humans:
                raise ValueError(
                    f"{replay_meta_file}: human appearance count {len(appearance_ids)} "
                    f"does not match human_count={expected_humans}"
                )
        poses: list[list[float]] = []
        drone_action_records: list[dict[str, Any]] = []
        for frame_idx, record in enumerate(records):
            raw_pose = record.get("target_pose") if isinstance(record, dict) else None
            if not isinstance(raw_pose, (list, tuple)) or len(raw_pose) < 6:
                raise ValueError(f"{path}: frame {frame_idx} has invalid target_pose")
            poses.append([float(value) for value in raw_pose[:6]])
            drone_action_records.append(dict(record) if isinstance(record, dict) else {})
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
                "drone_action_records": drone_action_records,
                "robotdog_action_records": [
                    dict(record) if isinstance(record, dict) else {}
                    for record in companion_records
                ],
                "instruction": manifest_item.get("instruction"),
                "task_type": manifest_item.get("task_type"),
                "replay_meta_source": str(
                    Path(str(replay_meta_path)).expanduser().resolve()
                ) if replay_meta_path else None,
                "replay_meta": replay_meta,
            }
        )
    if not trajectories:
        requested = "all" if episode_filter is None else ",".join(sorted(episode_filter))
        raise ValueError(f"no recorded target trajectories found in {source} (episodes={requested})")
    return trajectories


def _recorded_dt_from_action_record(
    record: dict[str, Any],
    default_dt: float,
    override_dt: float = 0.0,
) -> float:
    if float(override_dt) > 0.0:
        return float(override_dt)
    for key in ("base_velocity_dt_s", "effective_dt_s", "dt", "training_dt_s"):
        value = record.get(key)
        if isinstance(value, (int, float)) and float(value) > 0.0:
            return float(value)
    return float(default_dt)


def _recorded_drone_action_from_record(
    record: dict[str, Any],
    source: str,
    default_dt: float,
    override_dt: float = 0.0,
) -> Optional[list[float]]:
    """Return a BP_drone set_move_bp action from one recorded info frame.

    Four-dimensional fields such as ``env_action`` are already in the env action
    space. Three-dimensional velocity fields are converted with the same
    collection/eval convention: translation is velocity * dt, yaw remains a yaw
    rate command.
    """
    if source == "none":
        return None
    raw = record.get(source)
    if not isinstance(raw, (list, tuple)):
        return None
    try:
        values = [float(value) for value in raw]
    except Exception:
        return None
    if len(values) >= 4:
        return [values[0], values[1], values[2], values[3]]
    if len(values) >= 3:
        dt = _recorded_dt_from_action_record(record, default_dt, override_dt=override_dt)
        return [values[0] * dt, values[1] * dt, 0.0, values[2]]
    return None


def recorded_drone_action_for_step(
    target_trajectory: Optional[dict[str, Any]],
    args: argparse.Namespace,
    step_idx: int,
) -> tuple[Optional[list[float]], dict[str, Any]]:
    source = str(getattr(args, "oracle_drone_action_source", "none") or "none")
    debug = {
        "enabled": source != "none",
        "source": source,
        "record_index": None,
        "fallback": None,
    }
    if source == "none":
        return None, debug
    if target_trajectory is None:
        debug["fallback"] = "no_recorded_target_trajectory"
        return None, debug
    records = target_trajectory.get("drone_action_records") or []
    if not records:
        debug["fallback"] = "no_drone_action_records"
        return None, debug
    if bool(getattr(args, "oracle_drone_action_hold_last", True)):
        record_idx = min(step_idx, len(records) - 1)
    elif step_idx >= len(records):
        debug["fallback"] = "record_exhausted"
        return None, debug
    else:
        record_idx = step_idx
    record = records[record_idx]
    if not isinstance(record, dict):
        debug["fallback"] = "invalid_record"
        return None, debug
    default_dt = float(getattr(args, "dt", 0.1))
    override_dt = float(getattr(args, "oracle_drone_velocity_dt", 0.0) or 0.0)
    action = _recorded_drone_action_from_record(record, source, default_dt, override_dt=override_dt)
    debug.update(
        {
            "record_index": int(record_idx),
            "record_step": record.get("step"),
            "record_dt": _recorded_dt_from_action_record(record, default_dt, override_dt=override_dt),
            "record_dt_overridden": bool(override_dt > 0.0),
            "record_command_label_source": record.get("command_label_source"),
        }
    )
    if action is None:
        debug["fallback"] = f"missing_or_invalid_{source}"
        return None, debug
    return action, debug


def _recorded_robotdog_action_from_record(
    record: dict[str, Any],
    source: str,
    default_dt: float,
    override_dt: float = 0.0,
) -> Optional[list[float]]:
    if source == "none":
        return None
    raw = record.get(source)
    if not isinstance(raw, (list, tuple)):
        return None
    try:
        values = [float(value) for value in raw]
    except Exception:
        return None
    if source in {"env_action", "ground_action", "controller_ground_action"} and len(values) >= 2:
        return [values[0], values[1]]
    if len(values) >= 3:
        dt = _recorded_dt_from_action_record(record, default_dt, override_dt=override_dt)
        turn_deg = math.degrees(values[2] * dt)
        speed_cm_s = values[0] * UNREAL_UNITS_PER_METER
        return [float(turn_deg), float(speed_cm_s)]
    if len(values) >= 2:
        return [values[0], values[1]]
    return None


def recorded_robotdog_action_for_step(
    target_trajectory: Optional[dict[str, Any]],
    args: argparse.Namespace,
    step_idx: int,
) -> tuple[Optional[list[float]], dict[str, Any]]:
    source = str(getattr(args, "oracle_robotdog_action_source", "none") or "none")
    debug = {
        "enabled": source != "none",
        "source": source,
        "record_index": None,
        "fallback": None,
    }
    if source == "none":
        return None, debug
    if target_trajectory is None:
        debug["fallback"] = "no_recorded_target_trajectory"
        return None, debug
    records = target_trajectory.get("robotdog_action_records") or []
    if not records:
        debug["fallback"] = "no_robotdog_action_records"
        return None, debug
    if bool(getattr(args, "oracle_robotdog_action_hold_last", True)):
        record_idx = min(step_idx, len(records) - 1)
    elif step_idx >= len(records):
        debug["fallback"] = "record_exhausted"
        return None, debug
    else:
        record_idx = step_idx
    record = records[record_idx]
    if not isinstance(record, dict):
        debug["fallback"] = "invalid_record"
        return None, debug
    default_dt = float(getattr(args, "dt", 0.1))
    override_dt = float(getattr(args, "oracle_robotdog_velocity_dt", 0.0) or 0.0)
    action = _recorded_robotdog_action_from_record(record, source, default_dt, override_dt=override_dt)
    debug.update(
        {
            "record_index": int(record_idx),
            "record_step": record.get("step"),
            "record_dt": _recorded_dt_from_action_record(record, default_dt, override_dt=override_dt),
            "record_dt_overridden": bool(override_dt > 0.0),
            "record_command_label_source": record.get("command_label_source"),
        }
    )
    if action is None:
        debug["fallback"] = f"missing_or_invalid_{source}"
        return None, debug
    return action, debug


def recorded_target_action_for_step(
    target_trajectory: Optional[dict[str, Any]],
    args: argparse.Namespace,
    step_idx: int,
    current_pose: Optional[list[float]] = None,
) -> tuple[Optional[list[float]], dict[str, Any]]:
    debug: dict[str, Any] = {
        "enabled": True,
        "source": "target_pose_inverse_fixed_dt",
        "record_index": None,
        "fallback": None,
    }
    if target_trajectory is None:
        debug["fallback"] = "no_recorded_target_trajectory"
        return None, debug
    poses = target_trajectory.get("poses") or []
    if step_idx >= len(poses) - 1:
        debug["fallback"] = "record_exhausted"
        return None, debug
    dt = float(getattr(args, "dt", 0.1))
    if abs(dt - 0.1) > 1e-8:
        raise ValueError("recorded target pose inverse replay requires --dt 0.1")
    reference_before = poses[step_idx]
    reference_after = poses[step_idx + 1]
    delay_steps = int(getattr(args, "target_ground_translation_delay_steps", 1))
    translation_start = min(step_idx + delay_steps, len(poses) - 1)
    translation_end = min(translation_start + 1, len(poses) - 1)
    source_velocity = measured_body_velocity(
        poses[translation_start], poses[translation_end], dt
    )
    actual = current_pose if current_pose is not None else reference_before
    correction = measured_body_velocity(actual, reference_before, max(dt, float(
        getattr(args, "target_inverse_position_feedback_time_s", 0.5)
    )))
    max_feedback = float(getattr(args, "target_inverse_max_forward_feedback_mps", 2.0))
    forward_speed = float(source_velocity[0]) + float(
        np.clip(correction[0], -max_feedback, max_feedback)
    )
    desired_yaw_delta_deg = _wrap_degrees(
        float(reference_after[4]) - float(actual[4])
    )
    yaw_gain = max(float(getattr(args, "target_ground_yaw_gain", 0.4)), 1e-6)
    turn_command_deg = desired_yaw_delta_deg / yaw_gain
    debug.update(
        {
            "record_index": int(step_idx),
            "dt_s": dt,
            "translation_delay_steps": delay_steps,
            "translation_pose_indices": [translation_start, translation_end],
            "source_body_velocity_mps": source_velocity,
            "position_feedback_body_velocity_mps": correction,
            "desired_forward_speed_mps": forward_speed,
            "unexecutable_lateral_speed_mps": float(source_velocity[1] + correction[1]),
            "desired_yaw_delta_deg": desired_yaw_delta_deg,
            "ground_yaw_gain": yaw_gain,
            "turn_command_deg": turn_command_deg,
        }
    )
    return [float(turn_command_deg), float(forward_speed * UNREAL_UNITS_PER_METER)], debug


def replay_distractor_actions_for_step(
    target_trajectory: Optional[dict[str, Any]], step_idx: int
) -> list[list[float]]:
    """Read recorded distractor ground actions; no model/oracle follower action."""
    if target_trajectory is None:
        return []
    meta = target_trajectory.get("replay_meta") or {}
    rows = meta.get("distractor_actions_per_frame") or []
    if step_idx >= len(rows) or not isinstance(rows[step_idx], list):
        return []
    actions: list[list[float]] = []
    for item in rows[step_idx]:
        raw = item.get("action") if isinstance(item, dict) else None
        if isinstance(raw, (list, tuple)) and len(raw) >= 2:
            actions.append([float(raw[0]), float(raw[1])])
    return actions


def apply_recorded_step_timing(
    env,
    setup: dict[str, Any],
    target_trajectory: Optional[dict[str, Any]],
    args: argparse.Namespace,
    step_idx: int,
) -> Optional[dict[str, Any]]:
    if not bool(getattr(args, "oracle_recorded_step_timing", False)):
        return None
    if target_trajectory is None:
        return {"enabled": True, "fallback": "no_recorded_target_trajectory"}
    records = target_trajectory.get("drone_action_records") or []
    if not records:
        return {"enabled": True, "fallback": "no_drone_action_records"}
    record_idx = min(step_idx, len(records) - 1)
    record = records[record_idx]
    interval_ms = record.get("ue_interval_ms")
    if not isinstance(interval_ms, (int, float)) or float(interval_ms) <= 0.0:
        dt = _recorded_dt_from_action_record(record, float(getattr(args, "dt", 0.1)))
        interval_ms = int(round(dt * 1000.0))
    interval_ms = max(1, int(round(float(interval_ms))))
    try:
        env.unwrapped.interval = interval_ms
    except Exception:
        pass
    applied: list[str] = []
    for name in (setup.get("target_name"), setup.get("robotdog_name"), setup.get("drone_name")):
        if not name:
            continue
        try:
            env.unwrapped.unrealcv.set_interval(name, interval_ms)
            applied.append(str(name))
        except Exception:
            pass
    return {
        "enabled": True,
        "record_index": int(record_idx),
        "record_step": record.get("step"),
        "ue_interval_ms": int(interval_ms),
        "applied_to": applied,
        "effective_dt_s": record.get("effective_dt_s"),
        "base_velocity_dt_s": record.get("base_velocity_dt_s"),
    }


def _set_recorded_target_pose(env, setup: dict[str, Any], pose: list[float]) -> None:
    """Set an initial/current hold pose; never used for future action replay."""
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
        pass
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
    """Replace searched cameras only when each recorded camera is complete."""
    restored = {"robotdog": False, "drone": False}
    if target_trajectory is None:
        return dog_mount, dog_pitch, dog_yaw_offset, drone_pitch, drone_yaw_offset, restored

    dog_camera = target_trajectory.get("robotdog_camera") or {}
    mount = dog_camera.get("mount")
    pitch = dog_camera.get("pitch")
    yaw_offset = dog_camera.get("yaw_offset")
    dog_camera_complete = (
        isinstance(mount, list)
        and len(mount) >= 3
        and isinstance(pitch, (int, float))
        and isinstance(yaw_offset, (int, float))
    )
    if dog_camera_complete:
        dog_mount = [float(value) for value in mount[:3]]
        dog_pitch = float(pitch)
        dog_yaw_offset = float(yaw_offset)
        restored["robotdog"] = True

    drone_camera = target_trajectory.get("drone_camera") or {}
    pitch = drone_camera.get("pitch")
    yaw_offset = drone_camera.get("yaw_offset")
    drone_camera_complete = (
        isinstance(pitch, (int, float))
        and isinstance(yaw_offset, (int, float))
    )
    if drone_camera_complete:
        drone_pitch = float(pitch)
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


def _draw_roi_crop_xyxy(frame_bgr: np.ndarray, crop_xyxy: Any, label: str = "oracle ROI crop") -> np.ndarray:
    """Draw the ROI crop boundary for debugging only; model still sees the full frame branch."""
    out = ensure_bgr_uint8(frame_bgr).copy()
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in crop_xyxy[:4]]
    except Exception:
        return out
    if x2 <= x1 or y2 <= y1:
        return out
    h, w = out.shape[:2]
    x1, y1 = max(0, min(w - 1, x1)), max(0, min(h - 1, y1))
    x2, y2 = max(0, min(w - 1, x2)), max(0, min(h - 1, y2))
    color = (0, 220, 255)
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    cv2.putText(out, label, (x1, min(h - 8, max(20, y2 + 18))), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
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


def _trajectory_source_label(action_debug: dict[str, Any], agent_index: int) -> str:
    """Return the branch that produced the trajectory rendered for an agent."""
    mode_names = action_debug.get("routing_mode_name")
    mode: Any = None
    if isinstance(mode_names, np.ndarray):
        if mode_names.ndim > 0 and mode_names.size > agent_index:
            mode = mode_names.reshape(-1)[agent_index]
    elif isinstance(mode_names, (list, tuple)) and len(mode_names) > agent_index:
        mode = mode_names[agent_index]
    if mode is None:
        source_key = "drone_action_source" if agent_index == 0 else "robotdog_action_source"
        mode = action_debug.get(source_key)
    if mode is None:
        mode = action_debug.get("action_source")

    normalized = str(mode or "unknown").strip().lower()
    labels = {
        "self": "SELF",
        "cooperative": "COOPERATIVE",
        "belief": "BELIEF",
        "search": "SEARCH",
    }
    if normalized in labels:
        return labels[normalized]
    if not normalized or normalized == "none":
        return "UNKNOWN"
    # Keep legacy planner action-source fallbacks readable in a narrow overlay.
    return normalized.replace("_", " ").upper()[:32]


def _candidate_label(candidate: Any, score: Any) -> str:
    """Format the V3 cooperative mode selection for video overlays."""
    if candidate is None or score is None:
        return "planner=V3"
    return f"mode={int(candidate)} score={float(score):.2f}"


# ----------------------- V3 planner injection -----------------------

def waypoint_index_to_source_step(
    index: int,
    waypoint_count: int,
    horizon_steps: int,
) -> int:
    """Map an origin-inclusive waypoint index to a positive source action step."""
    if waypoint_count <= 1 or index <= 0:
        raise ValueError("Waypoint action selection requires a future point with index >= 1")
    return max(1, int(round(index * max(1, horizon_steps) / (waypoint_count - 1))))


class UnrealZooMultiAgentPlanner:
    """Injection slot replaced by eval_airground_coop_v3 before runtime main."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError(
            "Use eval_airground_coop_v3.py; the shared UnrealZoo runtime "
            "does not load a model directly."
        )


def align_ideal_follow_distances(args: argparse.Namespace) -> dict[str, float]:
    """Resolve the follower spawn distances used by place_initial_followers."""
    values: dict[str, float] = {}
    shared_init_dist = getattr(args, "init_follower_distance", None)
    if shared_init_dist is not None and float(shared_init_dist) <= 0.0:
        shared_init_dist = None
    for agent in ("robotdog", "drone"):
        min_dist = float(getattr(args, f"{agent}_min_follow_dist"))
        max_dist = float(getattr(args, f"{agent}_max_follow_dist"))
        if min_dist <= 0.0 or max_dist < min_dist:
            raise ValueError(
                f"Invalid {agent} follow range: min={min_dist}, max={max_dist}"
            )
        agent_init_dist = getattr(args, f"init_{agent}_distance", None)
        if agent_init_dist is not None:
            if float(agent_init_dist) <= 0.0:
                raise ValueError(f"Invalid {agent} init distance: {agent_init_dist}")
            ideal_dist = float(agent_init_dist)
        elif shared_init_dist is not None:
            ideal_dist = float(shared_init_dist)
        else:
            ideal_dist = 0.5 * (min_dist + max_dist)
        if ideal_dist <= 0.0:
            raise ValueError(f"Invalid {agent} init distance: {ideal_dist}")
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
    reset_env(env, args)  # 重置环境
    replay_meta = (target_trajectory or {}).get("replay_meta") or {}
    replay_distractor_count = int(replay_meta.get("distractor_count") or len(replay_meta.get("distractors") or []))
    if target_trajectory is not None and replay_distractor_count > 0:
        # Expand the UE population to the recorded target + distractors before
        # selecting cameras.  The follower IDs stay 1 (dog) and 2 (drone).
        env.unwrapped.agents_category = ["player", "player", "drone"] + [
            "player"
        ] * replay_distractor_count
        env.unwrapped.set_population(3 + replay_distractor_count)
        target_id, robotdog_id, drone_id = 0, 1, 2
        replay_distractor_ids = list(range(3, 3 + replay_distractor_count))
    else:
        target_id, robotdog_id, drone_id = classify_coop_agents(env)  # 确定环境对象
        replay_distractor_ids = []
    env.unwrapped.target_id = target_id  #
    env.unwrapped.tracker_id = drone_id
    env.unwrapped.protagonist_id = drone_id
    players = env.unwrapped.player_list
    target_name = players[target_id]
    robotdog_name = players[robotdog_id]
    drone_name = players[drone_id]
    env.unwrapped.unrealcv.batch_cmd(
        [
            env.unwrapped.unrealcv.set_move_bp(
                drone_name, [0.0, 0.0, 0.0, 0.0], return_cmd=True
            ),
            env.unwrapped.unrealcv.set_move_bp(
                robotdog_name, [0.0, 0.0], return_cmd=True
            ),
        ],
        None,
    )

    appearances = set_episode_appearances(env, target_id, robotdog_id, [], rng, args)
    replay_appearance_map = replay_meta.get("appearance_map") or {}
    if target_trajectory is not None and replay_appearance_map:
        # Replay metadata names the exact actor slots used during AT rerender.
        # Keep distractor appearances deterministic instead of randomizing them.
        for actor_name, appearance_id in replay_appearance_map.items():
            if actor_name in players:
                env.unwrapped.unrealcv.set_appearance(
                    actor_name, int(appearance_id)
                )
    if target_trajectory is not None and replay_distractor_ids:
        human_appearance_ids = list(replay_meta.get("human_appearance_ids") or [])
        for ordinal, actor_id in enumerate([target_id, *replay_distractor_ids]):
            if ordinal < len(human_appearance_ids):
                env.unwrapped.unrealcv.set_appearance(
                    players[actor_id], int(human_appearance_ids[ordinal])
                )
        first_distractors = replay_meta.get("distractor_poses_per_frame") or []
        if first_distractors and isinstance(first_distractors[0], dict):
            for ordinal, actor_id in enumerate(replay_distractor_ids):
                if ordinal >= len(first_distractors[0]):
                    break
                pose = list(first_distractors[0].values())[ordinal]
                if isinstance(pose, list) and len(pose) >= 3:
                    env.unwrapped.unrealcv.set_obj_location(players[actor_id], pose[:3])
                    if len(pose) >= 6:
                        env.unwrapped.unrealcv.set_obj_rotation(players[actor_id], pose[3:6])
                try:
                    env.unwrapped.unrealcv.set_max_speed(
                        players[actor_id],
                        float(args.environment_ground_max_speed_mps) * UNREAL_UNITS_PER_METER,
                    )
                    env.unwrapped.unrealcv.set_acceleration(
                        players[actor_id], float(args.ground_acceleration)
                    )
                except Exception:
                    pass
    if target_trajectory is not None:
        first_pose = target_trajectory["poses"][0]
        env.unwrapped.unrealcv.set_obj_location(target_name, first_pose[:3])
        try:
            env.unwrapped.unrealcv.set_obj_rotation(target_name, first_pose[3:6])
        except Exception:
            pass
        try:
            env.unwrapped.unrealcv.set_max_speed(target_name, 0.0)
            env.unwrapped.unrealcv.nav_to_goal(target_name, first_pose[:3])
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
    if target_trajectory is not None and len(target_path) >= 2:
        first_xy = np.asarray(target_path[0][:2], dtype=np.float64)
        for candidate in target_path[1:]:
            if float(np.linalg.norm(np.asarray(candidate[:2], dtype=np.float64) - first_xy)) > 1e-3:
                # This pose is used only for initialization direction. Action
                # replay never teleports the target to recorded future poses.
                initial_motion_goal = list(candidate)
                break
    goal_direction = np.asarray(initial_motion_goal[:2]) - np.asarray(target_pose[:2])
    if target_trajectory is None and np.linalg.norm(goal_direction) > 1e-6:
        target_yaw = math.degrees(math.atan2(float(goal_direction[1]), float(goal_direction[0])))
        try:
            env.unwrapped.unrealcv.set_obj_rotation(target_name, [0.0, target_yaw, 0.0])
        except Exception:
            pass
    try:
        # Use the same unrestrictive BP ceiling and acceleration as the
        # verified fixed-step replay.  Movement is still determined by the
        # per-step action; this prevents UE from silently clipping it or
        # adding a random/default acceleration ramp.
        target_speed = (
            0.0
            if target_trajectory is not None and replay_mode == "path_goal"
            else float(args.environment_ground_max_speed_mps) * UNREAL_UNITS_PER_METER
        )
        env.unwrapped.unrealcv.set_max_speed(target_name, target_speed)
        env.unwrapped.unrealcv.set_max_speed(
            robotdog_name,
            float(args.environment_ground_max_speed_mps) * UNREAL_UNITS_PER_METER,
        )
        env.unwrapped.unrealcv.set_acceleration(target_name, float(args.ground_acceleration))
        env.unwrapped.unrealcv.set_acceleration(robotdog_name, float(args.ground_acceleration))
    except Exception:
        pass

    update_observation(env, refresh_cameras=True)
    initial_follow_distances = align_ideal_follow_distances(args)
    init_followers_behind_target = bool(
        getattr(args, "init_followers_behind_target", True)
    )
    use_recorded_agent_poses = bool(
        target_trajectory is not None
        and getattr(args, "init_from_recorded_agent_poses", False)
        and not init_followers_behind_target
    )
    if init_followers_behind_target or not use_recorded_agent_poses:
        place_initial_followers(env, target_id, robotdog_id, drone_id, initial_motion_goal, args)
    restored_agent_poses = (
        restore_recorded_agent_initial_poses(
            env,
            args,
            robotdog_name,
            drone_name,
            target_trajectory,
        )
        if use_recorded_agent_poses
        else {"robotdog": False, "drone": False}
    )
    if use_recorded_agent_poses and not all(restored_agent_poses.values()):
        # Establish a valid fallback for both agents, then overwrite whichever
        # recorded poses are available. Keep the restored flags for auditing.
        place_initial_followers(env, target_id, robotdog_id, drone_id, initial_motion_goal, args)
        restored_agent_poses = restore_recorded_agent_initial_poses(
            env,
            args,
            robotdog_name,
            drone_name,
            target_trajectory,
        )
    if target_trajectory is not None:
        if replay_mode == "path_goal":
            # Keep the recorded start pose fixed until frame 0 has been consumed.
            first_pose = target_trajectory["poses"][0]
            _set_recorded_target_pose(
                env,
                {
                    "target_name": target_name,
                },
                first_pose,
            )
            target_pose = list(first_pose)
        elif replay_mode == "nav_goal":
            env.unwrapped.unrealcv.nav_to_goal(target_name, target_goal)
    else:
        env.unwrapped.unrealcv.nav_to_goal(target_name, target_goal)

    # Freeze only after actor placement has completed. UE needs a live frame
    # while reset-time appearance and pose changes are applied, but the
    # following camera/mask search is slow enough for actors to drift metres.
    if bool(getattr(args, "deterministic_step", True)):
        configure_deterministic_clock(env, args)
    else:
        env.unwrapped.unrealcv.set_pause()
        if not env.unwrapped.unrealcv.get_is_paused():
            raise RuntimeError("Unreal setup failed to pause the world")

    # Placement commands above are issued while UE is live so reset-time actor
    # changes can settle. Re-apply the authored first-frame state after the
    # pause barrier so residual BP velocity cannot shift the actual frame-0
    # observation away from the recording.
    if target_trajectory is not None and replay_mode != "nav_goal":
        _set_recorded_target_pose(
            env,
            {"target_name": target_name},
            target_trajectory["poses"][0],
        )
    if use_recorded_agent_poses:
        restored_agent_poses = restore_recorded_agent_initial_poses(
            env,
            args,
            robotdog_name,
            drone_name,
            target_trajectory,
        )
        if not all(restored_agent_poses.values()):
            place_initial_followers(
                env,
                target_id,
                robotdog_id,
                drone_id,
                initial_motion_goal,
                args,
            )
            restored_agent_poses = restore_recorded_agent_initial_poses(
                env,
                args,
                robotdog_name,
                drone_name,
                target_trajectory,
            )
        update_observation(env, refresh_cameras=True)

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
    if use_recorded_agent_poses:
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
        "target_waypoints": (
            target_path
            if target_trajectory is not None and replay_mode == "path_goal"
            else [target_goal]
        ),
        "target_path": target_path,
        "instruction": target_trajectory.get("instruction") if target_trajectory is not None else None,
        "replay_distractor_ids": replay_distractor_ids,
        "replay_distractor_names": [players[index] for index in replay_distractor_ids],
                "target_replay_mode": replay_mode if target_trajectory is not None else None,
        "replay_distractors": bool(target_trajectory is not None and replay_meta.get("distractor_poses_per_frame")),
        "replay_distractor_count": int(len(replay_meta.get("distractors") or [])),
        "replay_distractor_motion_policy": (
            replay_meta.get("target_motion_source")
            if target_trajectory is not None
            else None
        ),
        "ue_interval_ms": int(getattr(args, "ue_interval_ms", None) or round(float(args.dt) * 1000.0)),
        "bp_interval_s": float(getattr(args, "ue_interval_ms", None) or round(float(args.dt) * 1000.0)) / 1000.0,
        "velocity_feedback": {
            "drone_translation_gain": float(args.drone_velocity_feedback_gain),
            "drone_yaw_gain": float(args.drone_yaw_feedback_gain),
            "drone_max_translation": float(args.drone_feedback_max_translation),
            "drone_max_yaw_rate": float(args.drone_feedback_max_yaw_rate),
            "robotdog_translation_gain": float(args.robotdog_velocity_feedback_gain),
            "robotdog_yaw_gain": float(args.robotdog_yaw_feedback_gain),
            "robotdog_max_translation": float(args.robotdog_feedback_max_translation),
            "robotdog_max_yaw_rate": float(args.robotdog_feedback_max_yaw_rate),
        },
        "action_gain": {
            "drone_speed": float(args.drone_speed_gain),
            "drone_yaw": float(args.drone_yaw_scale),
            "robotdog_speed": float(args.robotdog_speed_gain),
            "robotdog_yaw": float(args.robotdog_yaw_scale),
        },
        "action_selection": {
            "waypoint_control_mode": str(args.waypoint_control_mode),
            "waypoint_index": int(args.waypoint_index),
            "drone_waypoint_index": int(
                args.drone_waypoint_index
                if args.drone_waypoint_index is not None
                else args.waypoint_index
            ),
            "robotdog_waypoint_index": int(
                args.robotdog_waypoint_index
                if args.robotdog_waypoint_index is not None
                else args.waypoint_index
            ),
            "waypoint_horizon_steps": int(args.waypoint_horizon_steps),
            "waypoint_source_dt_s": float(args.waypoint_source_dt or args.dt),
            "ground_translation_delay_steps": int(args.ground_translation_delay_steps),
            "ground_yaw_gain": float(args.ground_yaw_gain),
            "drone_inverse_coefficients": {
                "a_forward": float(args.drone_inverse_a_forward),
                "b_forward": float(args.drone_inverse_b_forward),
                "a_lateral": float(args.drone_inverse_a_lateral),
                "b_lateral": float(args.drone_inverse_b_lateral),
                "yaw_a": float(args.drone_inverse_yaw_a),
                "yaw_b": float(args.drone_inverse_yaw_b),
            },
            "inverse_command_smoothing": {
                "drone_xy_alpha": float(args.drone_inverse_xy_smoothing_alpha),
                "drone_yaw_alpha": float(args.drone_inverse_yaw_smoothing_alpha),
                "robotdog_speed_alpha": float(args.robotdog_inverse_speed_smoothing_alpha),
                "robotdog_yaw_alpha": float(args.robotdog_inverse_yaw_smoothing_alpha),
            },
            "state_feedback": "none_internal_command_rollout" if args.waypoint_control_mode == "inverse_fixed_dt" else "legacy_pose_velocity_feedback",
        },
        "target_path_sampling": {
            "source": "test_json_target_pose" if target_trajectory is not None else "simulator_navigation",
            "min_spacing_unreal_units": float(getattr(args, "target_path_min_spacing", 100.0)),
            "num_waypoints": len(target_path),
            "uses_start_and_final_pose": bool(target_trajectory is not None and len(target_path) >= 2),
        },
        "target_stopped": False,
        "target_stop_step": None,
        "target_stop_wait_count": 0,
        "target_stop_wait_steps": rng.randint(stop_wait_min, stop_wait_max),
        "init_from_recorded_agent_poses": bool(getattr(args, "init_from_recorded_agent_poses", False)),
        "init_followers_behind_target": init_followers_behind_target,
        "init_agent_pose_policy": (
            "behind_target_fixed_distance"
            if init_followers_behind_target
            else "recorded_agent_poses"
            if use_recorded_agent_poses and all(restored_agent_poses.values())
            else "recorded_agent_poses_with_ideal_fallback"
            if use_recorded_agent_poses
            else "behind_target_fixed_distance_fallback"
        ),
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
    # Teleport-based target replay modes are intentionally unsupported.
    mode = str(getattr(args, "target_replay_mode", "action"))
    if mode not in {"nav_goal", "action", "path_goal"}:
        raise ValueError(
            f"unsupported target replay mode {mode!r}; use action, path_goal, or nav_goal"
        )
    return mode


def _wrap_degrees(degrees: float) -> float:
    return float((degrees + 180.0) % 360.0 - 180.0)


def measured_body_velocity(
    pose_before: list[float],
    pose_after: list[float],
    dt_seconds: float,
) -> list[float]:
    """Measure body-frame [vx, vy, yaw_rate] from one action pulse."""
    dt_seconds = max(float(dt_seconds), 1e-6)
    delta_xy_m = (pose_xyz(pose_after)[:2] - pose_xyz(pose_before)[:2]) / UNREAL_UNITS_PER_METER
    yaw = math.radians(yaw_deg(pose_before))
    forward = np.asarray([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
    right = np.asarray([-math.sin(yaw), math.cos(yaw)], dtype=np.float64)
    return [
        float(np.dot(delta_xy_m, forward) / dt_seconds),
        float(np.dot(delta_xy_m, right) / dt_seconds),
        float(math.radians(_wrap_degrees(yaw_deg(pose_after) - yaw_deg(pose_before))) / dt_seconds),
    ]


def apply_oracle_heading_assist(
    args: argparse.Namespace,
    drone_pose: list[float],
    dog_pose: list[float],
    target_pose: list[float],
    drone_action: list[float],
    dog_action: list[float],
    action_debug: dict[str, Any],
) -> None:
    """Debug assist: keep model translation, replace yaw/turn from target bearing."""
    if not bool(getattr(args, "oracle_heading_assist", False)):
        return

    drone_error_deg = _wrap_degrees(
        heading_deg(pose_xyz(drone_pose), pose_xyz(target_pose)) - yaw_deg(drone_pose)
    )
    dog_error_deg = _wrap_degrees(
        heading_deg(pose_xyz(dog_pose), pose_xyz(target_pose)) - yaw_deg(dog_pose)
    )
    drone_yaw = float(
        np.clip(
            math.radians(drone_error_deg) * float(args.drone_heading_assist_gain),
            -float(args.drone_max_yaw_rate),
            float(args.drone_max_yaw_rate),
        )
    )
    dog_turn = float(
        np.clip(
            dog_error_deg * float(args.robotdog_heading_assist_gain),
            -float(args.robotdog_max_turn_deg),
            float(args.robotdog_max_turn_deg),
        )
    )
    model_drone_yaw = float(drone_action[3])
    model_dog_turn = float(dog_action[0])
    drone_action[3] = drone_yaw
    dog_action[0] = dog_turn
    action_debug.update(
        {
            "oracle_heading_assist": True,
            "drone_model_yaw_command_before_assist": model_drone_yaw,
            "robotdog_model_turn_before_assist": model_dog_turn,
            "drone_heading_error_deg": float(drone_error_deg),
            "robotdog_heading_error_deg": float(dog_error_deg),
            "drone_heading_assist_gain": float(args.drone_heading_assist_gain),
            "robotdog_heading_assist_gain": float(args.robotdog_heading_assist_gain),
            "drone_yaw_command_after_assist": float(drone_yaw),
            "robotdog_turn_after_assist": float(dog_turn),
        }
    )


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


def _target_path_follow_points(trajectory: dict[str, Any]) -> list[list[float]]:
    """Dense recorded xyz path used for fixed-step target replay."""
    points: list[list[float]] = []
    for pose in trajectory.get("poses") or []:
        if not isinstance(pose, list) or len(pose) < 3:
            continue
        xyz = [float(value) for value in pose[:3]]
        if points:
            prev = np.asarray(points[-1][:2], dtype=np.float64)
            cur = np.asarray(xyz[:2], dtype=np.float64)
            if float(np.linalg.norm(cur - prev)) <= 1e-6:
                continue
        points.append(xyz)
    return points


def start_recorded_path_navigation(env, args: argparse.Namespace, setup: dict[str, Any], trajectory: dict[str, Any]) -> None:
    waypoints = (
        setup.get("target_path")
        if setup.get("target_replay_mode") == "path_goal"
        else None
    ) or _target_path_waypoints(trajectory, args)
    follow_points = _target_path_follow_points(trajectory)
    if len(follow_points) < 2:
        follow_points = waypoints
    setup["target_path_waypoints"] = waypoints
    setup["target_path_follow_points"] = follow_points
    setup["target_path_replay_policy"] = "action_walk_recorded_path"
    setup["target_path_index"] = 0
    setup["target_path_segment_index"] = 0
    setup["target_path_completed"] = False
    setup["target_path_last_xy"] = list(waypoints[0][:2]) if waypoints else None
    if len(waypoints) < 2 or len(follow_points) < 2:
        setup["target_path_completed"] = True
        return
    try:
        env.unwrapped.unrealcv.set_max_speed(setup["target_name"], float(args.human_speed))
    except Exception:
        pass
    try:
        env.unwrapped.unrealcv.set_move_bp(setup["target_name"], [0.0, 0.0])
    except Exception:
        pass
    setup["target_path_stall_count"] = 0
    target_pose = list(env.unwrapped.obj_poses[setup["target_id"]])
    setup["target_path_last_xy"] = [float(target_pose[0]), float(target_pose[1])]


def _update_sparse_target_path_index(setup: dict[str, Any], target_xy: np.ndarray) -> None:
    waypoints = setup.get("target_path_waypoints") or []
    if len(waypoints) < 2:
        setup["target_path_index"] = 0
        return
    current_index = int(np.clip(setup.get("target_path_index", 0), 0, len(waypoints) - 1))
    while current_index < len(waypoints) - 1:
        next_xy = np.asarray(waypoints[current_index + 1][:2], dtype=np.float64)
        cur_xy = np.asarray(waypoints[current_index][:2], dtype=np.float64)
        if float(np.linalg.norm(target_xy - next_xy)) > float(np.linalg.norm(target_xy - cur_xy)):
            break
        current_index += 1
    setup["target_path_index"] = current_index


def update_recorded_path_navigation(env, args: argparse.Namespace, setup: dict[str, Any]) -> None:
    waypoints = setup.get("target_path_waypoints") or []
    if len(waypoints) < 2:
        return
    target_pose = list(env.unwrapped.obj_poses[setup["target_id"]])
    target_xy = np.asarray(target_pose[:2], dtype=np.float64)
    _update_sparse_target_path_index(setup, target_xy)
    setup["target_path_last_xy"] = [float(target_xy[0]), float(target_xy[1])]


def _target_stop_action(env, setup: dict[str, Any]):
    setup["target_path_action_debug"] = {
        "policy": setup.get("target_path_replay_policy"),
        "segment_index": int(setup.get("target_path_segment_index", 0)),
        "segment_count": max(len(setup.get("target_path_follow_points") or []) - 1, 0),
        "turn_deg": 0.0,
        "speed_uu_s": 0.0,
        "completed": bool(setup.get("target_path_completed", False)),
    }
    return action_for_target_space(env, setup["target_id"], [0.0, 0.0])


def target_path_step_action(env, args: argparse.Namespace, setup: dict[str, Any]):
    """Drive recorded target replay through the same action channel as collection."""
    if bool(setup.get("target_stopped", False)):
        return _target_stop_action(env, setup)

    points = setup.get("target_path_follow_points") or []
    if len(points) < 2:
        setup["target_path_completed"] = True
        return _target_stop_action(env, setup)

    target_pose = list(env.unwrapped.obj_poses[setup["target_id"]])
    target_xy = np.asarray(target_pose[:2], dtype=np.float64)
    segment_index = int(
        np.clip(setup.get("target_path_segment_index", 0), 0, len(points) - 2)
    )
    nominal_step_uu = max(abs(float(args.human_speed)) * max(float(args.dt), 1e-6), 1.0)
    reach = max(nominal_step_uu * 1.5, 5.0)

    while segment_index < len(points) - 1:
        current_xy = np.asarray(points[segment_index][:2], dtype=np.float64)
        next_xy = np.asarray(points[segment_index + 1][:2], dtype=np.float64)
        segment_xy = next_xy - current_xy
        reached = float(np.linalg.norm(next_xy - target_xy)) <= reach
        # A turning actor can cross a waypoint between two fixed ticks without
        # ever entering the small reach radius. Advance once it has crossed
        # the waypoint plane along the recorded segment, otherwise it turns
        # back and oscillates forever around that point.
        passed = bool(
            float(np.dot(target_xy - next_xy, segment_xy)) >= 0.0
            if float(np.linalg.norm(segment_xy)) > 1e-6
            else True
        )
        if not reached and not passed:
            break
        segment_index += 1

    setup["target_path_segment_index"] = segment_index
    _update_sparse_target_path_index(setup, target_xy)
    if segment_index >= len(points) - 1:
        setup["target_path_completed"] = True
        return _target_stop_action(env, setup)

    next_xy = np.asarray(points[segment_index + 1][:2], dtype=np.float64)
    delta_xy = next_xy - target_xy
    if float(np.linalg.norm(delta_xy)) <= 1e-6:
        turn_deg = 0.0
    else:
        desired_yaw = math.degrees(math.atan2(float(delta_xy[1]), float(delta_xy[0])))
        turn_deg = float(
            np.clip(
                _wrap_degrees(desired_yaw - float(yaw_deg(target_pose))),
                -float(args.human_turn),
                float(args.human_turn),
            )
        )
    speed = float(args.human_speed)
    setup["target_path_action_debug"] = {
        "policy": "action_walk_recorded_path",
        "segment_index": int(segment_index),
        "segment_count": int(len(points) - 1),
        "turn_deg": float(turn_deg),
        "speed_uu_s": float(speed),
        "target_yaw_deg": float(yaw_deg(target_pose)),
        "completed": False,
    }
    return action_for_target_space(env, setup["target_id"], [turn_deg, speed])


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
    reached = bool(replay_mode == "action" and recorded_pose_exhausted)
    final_goal: Optional[list[float]] = None
    if replay_mode == "path_goal":
        path = setup.get("target_path_waypoints") or []
        if path:
            final_goal = path[-1]
            if setup.get("target_path_replay_policy") == "action_walk_recorded_path":
                reached = bool(setup.get("target_path_completed", False))
            else:
                reached = bool(setup.get("target_path_completed", False)) or int(
                    setup.get("target_path_index", 0)
                ) >= len(path) - 1
    elif replay_mode != "action":
        final_goal = setup.get("target_goal")

    skip_final_distance_check = bool(
        replay_mode == "path_goal"
        and setup.get("target_path_replay_policy") == "action_walk_recorded_path"
        and setup.get("target_path_completed", False)
    )
    if reached and final_goal is not None and not skip_final_distance_check:
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
    if setup.get("target_path_replay_policy") == "action_walk_recorded_path":
        return
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
    target_action: Any = None,
    distractor_actions: Optional[list[Any]] = None,
):
    """Advance exactly one fixed-delta UE frame and return paused observations."""
    unrealcv = env.unwrapped.unrealcv
    pause_check_stride = int(
        getattr(args, "deterministic_pause_check_stride", 1) or 0
    )
    verify_pause = bool(
        pause_check_stride > 0
        and int(env.unwrapped.count_steps) % pause_check_stride == 0
    )
    if verify_pause and not unrealcv.get_is_paused():
        raise RuntimeError("Unreal world advanced outside deterministic step: expected paused state")

    commands = [
        unrealcv.set_move_bp(setup["drone_name"], drone_action, return_cmd=True),
        unrealcv.set_move_bp(setup["robotdog_name"], dog_action, return_cmd=True),
    ]
    if target_action is not None or distractor_actions:
        actions = [None for _ in env.unwrapped.player_list]
        if target_action is not None:
            actions[setup["target_id"]] = target_action
        for actor_id, action in zip(
            setup.get("replay_distractor_ids", []), distractor_actions or []
        ):
            actions[actor_id] = action_for_target_space(env, actor_id, action)
        actions2move, actions2turn, actions2animate = env.unwrapped.action_mapping(
            actions, env.unwrapped.player_list
        )
        actor_ids = [setup["target_id"]] + list(setup.get("replay_distractor_ids", []))
        for actor_id in actor_ids:
            actor_name = env.unwrapped.player_list[actor_id]
            if actions2move[actor_id] is not None:
                commands.append(unrealcv.set_move_bp(actor_name, actions2move[actor_id], return_cmd=True))
            if actions2turn[actor_id] is not None:
                commands.append(unrealcv.set_cam(actor_name, env.unwrapped.agents[actor_name]["relative_location"], actions2turn[actor_id], return_cmd=True))
            if actions2animate[actor_id] is not None:
                commands.append(unrealcv.set_animation(actor_name, actions2animate[actor_id], return_cmd=True))
    unrealcv.batch_cmd(commands, None)
    pulse_t0 = time.monotonic()
    # UnrealCV dispatches these game commands on successive fixed-delta
    # frames. They must be separate requests: batching resume and pause keeps
    # both commands in the same paused frame and advances no simulation.
    unrealcv.set_resume()
    unrealcv.set_pause()
    pulse_wall_seconds = time.monotonic() - pulse_t0
    if verify_pause and not unrealcv.get_is_paused():
        raise RuntimeError("Unreal deterministic step failed to return to paused state")

    idle_actions = [None for _ in env.unwrapped.player_list]
    if bool(getattr(args, "fast_eval_io", False)):
        # The action pulse above already advanced the paused world.  Only the
        # resulting object poses are needed here for realized velocity,
        # collision and following-distance metrics.  The legacy base step also
        # renders every camera even though those post-action images are thrown
        # away before the next observation snapshot.
        obs, rewards, done, info = data_collection_step_pose_only(
            env, idle_actions
        )
    else:
        obs, rewards, done, info = data_collection_step(env, idle_actions)
    info["Action"] = [
        drone_action
        if idx == setup["drone_id"]
        else dog_action
        if idx == setup["robotdog_id"]
        else target_action
        if idx == setup["target_id"]
        else None
        for idx in range(len(idle_actions))
    ]
    return obs, rewards, done, info, pulse_wall_seconds


def _jsonable_action(action: Any) -> Any:
    if action is None:
        return None
    if isinstance(action, np.ndarray):
        return action.tolist()
    if isinstance(action, (list, tuple)):
        return [_jsonable_action(value) for value in action]
    if isinstance(action, (np.floating, float)):
        return float(action)
    if isinstance(action, (np.integer, int)):
        return int(action)
    return action


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
    *,
    include_rgb: bool = True,
    include_mask: bool = True,
) -> tuple[
    tuple[Optional[np.ndarray], Optional[float], Optional[bool], Optional[list[int]], Optional[list[float]]],
    tuple[Optional[np.ndarray], Optional[float], Optional[bool], Optional[list[int]], Optional[list[float]]],
]:
    """Batch poses, follower masks and optionally RGB into one snapshot."""
    fast_eval_io = bool(getattr(args, "fast_eval_io", False))
    if fast_eval_io:
        # V3 consumes only the drone and RobotDog views.  Keep all player poses
        # in the same batch for metric/control alignment, but do not render the
        # target or global cameras.  Masks remain enabled because visibility
        # and bbox IoU are evaluation metrics.
        unwrapped = env.unwrapped
        follower_ids = (setup["drone_id"], setup["robotdog_id"])
        follower_cams = [unwrapped.cam_list[index] for index in follower_ids]
        reuse_post_action_poses = bool(
            getattr(args, "reuse_post_action_poses", False)
        )
        if reuse_post_action_poses and not include_rgb and not include_mask:
            obj_poses, imgs, masks = [], [], []
        else:
            obj_poses, _cam_poses, imgs, masks, _depths = (
                unwrapped.unrealcv.get_pose_img_batch(
                    unwrapped.player_list,
                    follower_cams,
                    [False, bool(include_rgb), bool(include_mask), False],
                    mask_mode=str(getattr(args, "mask_image_format", "png")),
                    include_obj_pose=not reuse_post_action_poses,
                )
            )
        if obj_poses:
            unwrapped.obj_poses = obj_poses
        obs_by_agent = (
            {
                agent_id: imgs[pair_index]
                for pair_index, agent_id in enumerate(follower_ids)
            }
            if include_rgb
            else {}
        )
        masks_by_agent = (
            {
                agent_id: masks[pair_index]
                for pair_index, agent_id in enumerate(follower_ids)
            }
            if include_mask
            else {}
        )
    else:
        obs, masks = capture_color_mask_snapshot(env, include_masks=True)

    def read_one(
        agent_id: int,
    ) -> tuple[Optional[np.ndarray], Optional[float], Optional[bool], Optional[list[int]], Optional[list[float]]]:
        frame = (
            ensure_bgr_uint8(
                obs_by_agent[agent_id] if fast_eval_io else obs[agent_id]
            )
            if include_rgb
            else None
        )
        if not include_mask:
            return frame, None, None, None, None
        visibility, visible, bbox = target_mask_visibility(
            env,
            env.unwrapped.cam_list[agent_id],
            setup["target_name"],
            mask_img=(
                masks_by_agent[agent_id]
                if fast_eval_io
                else masks[agent_id]
            ),
        )
        bbox_norm = _normalize_bbox_xywh(bbox, args.width, args.height)
        return frame, float(visibility), bool(visible), bbox, bbox_norm

    return read_one(setup["drone_id"]), read_one(setup["robotdog_id"])


def _rollout_future_waypoint_segment(
    waypoints: Any,
    offset: int,
) -> np.ndarray:
    """Rebase a cached ego-frame trajectory at a later physics step.

    The model trajectory is origin-inclusive and expressed in the body frame
    at its policy observation.  On a held policy step, convert the remaining
    future path into the body frame at ``offset`` so the inverse controller
    consumes the next predicted segment instead of repeating the first action.
    """

    values = np.asarray(waypoints, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] != 2 or values.shape[2] < 3:
        raise ValueError(
            "Policy waypoints must have shape (2,N,D>=3), got "
            f"{values.shape}"
        )
    count = int(values.shape[1])
    anchor_index = int(np.clip(offset, 0, max(count - 1, 0)))
    if anchor_index == 0:
        return values.copy()
    source_indices = np.clip(
        np.arange(count, dtype=np.int64) + anchor_index,
        0,
        count - 1,
    )
    rebased = values[:, source_indices].copy()
    for agent_id in range(values.shape[0]):
        anchor = values[agent_id, anchor_index]
        delta_xy = rebased[agent_id, :, :2] - anchor[:2]
        yaw = float(anchor[2])
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        x_local = cosine * delta_xy[:, 0] + sine * delta_xy[:, 1]
        y_local = -sine * delta_xy[:, 0] + cosine * delta_xy[:, 1]
        rebased[agent_id, :, 0] = x_local
        rebased[agent_id, :, 1] = y_local
        yaw_delta = rebased[agent_id, :, 2] - yaw
        rebased[agent_id, :, 2] = np.arctan2(
            np.sin(yaw_delta),
            np.cos(yaw_delta),
        )
        rebased[agent_id, 0, :3] = 0.0
    return rebased


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
    configure_deterministic_clock(env, args)
    if target_trajectory is not None and target_replay_mode == "path_goal":
        start_recorded_path_navigation(env, args, setup, target_trajectory)
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
    # Keep UE contact diagnostics separate, but make the primary episode
    # collision follow the configured evaluation rule below.
    physical_collision = False
    # Habitat-style episode metric: once either follower comes within the
    # configured distance of the human, the metric stays 1 for this episode.
    # This is deliberately separate from the UE physical collision flag.
    human_collision = False
    drone_human_collision = False
    robotdog_human_collision = False
    status = "Normal"
    t0 = time.monotonic()
    # Pure neural-network forward latency: starts immediately before the
    # planner/model call and ends when its output tensor is returned. This is
    # intentionally separate from camera/UE I/O, preprocessing, perception,
    # feature encoding, routing, action conversion, IPC and video writing.
    model_inference_seconds_total = 0.0
    model_inference_steps = 0
    last_info = None
    previous_observation_wall_time: Optional[float] = None
    planner_debug_steps = max(0, int(getattr(args, "planner_debug_steps", 0) or 0))
    policy_inference_stride = int(getattr(args, "policy_inference_stride", 1) or 1)
    if policy_inference_stride < 1:
        raise ValueError("--policy-inference-stride must be >= 1")
    skip_rgb_between_policy_steps = bool(
        getattr(args, "skip_rgb_between_policy_steps", True)
    )
    metric_mask_stride = int(getattr(args, "metric_mask_stride", 1) or 1)
    if metric_mask_stride < 1:
        raise ValueError("--metric-mask-stride must be >= 1")
    cached_policy_prediction: Optional[dict[str, Any]] = None
    cached_drone_action: Optional[list[float]] = None
    cached_dog_action: Optional[list[float]] = None
    cached_action_debug: Optional[dict[str, Any]] = None
    cached_policy_waypoints: Optional[np.ndarray] = None
    cached_drone_frame: Optional[np.ndarray] = None
    cached_dog_frame: Optional[np.ndarray] = None
    cached_drone_mask_metrics: Optional[tuple[float, bool, list[int], list[float]]] = None
    cached_dog_mask_metrics: Optional[tuple[float, bool, list[int], list[float]]] = None

    # Recorded path length no longer terminates the episode immediately. As in
    # Habitat, the target can stop at its final goal while the followers keep
    # receiving and executing model actions for a short observation window.
    episode_max_steps = args.max_steps

    for step_idx in range(episode_max_steps):
        loop_start_monotonic = time.monotonic()
        if bool(setup.get("target_stopped", False)):
            update_habitat_target_stop_state(env, args, setup, step_idx=step_idx)
        elif target_trajectory is not None and target_replay_mode == "path_goal":
            update_recorded_path_navigation(env, args, setup)
        target_stopped = update_habitat_target_stop_state(
            env,
            args,
            setup,
            step_idx=step_idx,
            recorded_pose_exhausted=bool(
                target_trajectory is not None
                and target_replay_mode == "action"
                and step_idx >= len(target_trajectory["poses"]) - 1
            ),
        )
        target_action = None
        if target_trajectory is not None and target_replay_mode == "path_goal":
            target_action = (
                _target_stop_action(env, setup)
                if target_stopped
                else target_path_step_action(env, args, setup)
            )
        elif target_trajectory is not None and target_replay_mode == "action":
            if target_stopped:
                target_action = _target_stop_action(env, setup)
                setup["target_path_action_debug"] = {
                    "policy": f"recorded_{target_replay_mode}",
                    "fallback": "target_stopped",
                }
            else:
                raw_target_action, target_action_debug = recorded_target_action_for_step(
                    target_trajectory,
                    args,
                    step_idx,
                    current_pose=list(env.unwrapped.obj_poses[setup["target_id"]]),
                )
                target_action = (
                    action_for_target_space(env, setup["target_id"], raw_target_action)
                    if raw_target_action is not None
                    else None
                )
                setup["target_path_action_debug"] = {
                    "policy": f"recorded_{target_replay_mode}",
                    **target_action_debug,
                }
        target_pose_before = list(env.unwrapped.obj_poses[setup["target_id"]])
        if args.face_target_before_step:
            # Legacy dog-only oracle heading. Drone yaw must remain model-action
            # controlled during tracking; recorded yaw restore is init-only.
            set_ground_yaw(
                env,
                setup["robotdog_name"],
                heading_deg(pose_xyz(list(env.unwrapped.obj_poses[setup["robotdog_id"]])), pose_xyz(target_pose_before)),
            )

        policy_inference_step = (
            cached_policy_prediction is None
            or step_idx % policy_inference_stride == 0
        )
        intermediate_observation_step = bool(
            not policy_inference_step
            and getattr(planner, "supports_intermediate_observation", False)
        )
        # V3 的训练历史间隔是 0.1 s。即使 Qwen 每 5 步才运行一次，中间物理步
        # 仍需读取 RGB 并只更新视觉历史，不能把 0.5 s 稀疏帧伪装成 0.1 s 历史。
        include_policy_rgb = bool(
            policy_inference_step
            or intermediate_observation_step
            or not skip_rgb_between_policy_steps
        )
        metric_mask_step = bool(
            cached_drone_mask_metrics is None
            or cached_dog_mask_metrics is None
            or step_idx % metric_mask_stride == 0
            # Terminal success must use a current, not held, visibility mask.
            or target_stopped
        )
        if include_policy_rgb or metric_mask_step:
            _lock_eval_cameras(env, setup, args)
        snapshot_start_monotonic = time.monotonic()
        drone_input, dog_input = _read_agent_pair(
            env,
            setup,
            args,
            include_rgb=include_policy_rgb,
            include_mask=metric_mask_step,
        )
        observation_wall_time = time.monotonic()
        snapshot_query_seconds = observation_wall_time - snapshot_start_monotonic
        if previous_observation_wall_time is None:
            # Match collection: use nominal dt for the first action, then use
            # the previous measured observation interval.
            realtime_control_period_seconds = float(args.dt)
        else:
            realtime_control_period_seconds = (
                observation_wall_time - previous_observation_wall_time
            )
        previous_observation_wall_time = observation_wall_time
        drone_input_frame, drone_input_vis, drone_input_visible, drone_input_bbox, drone_bbox_norm = drone_input
        dog_input_frame, dog_input_vis, dog_input_visible, dog_input_bbox, dog_bbox_norm = dog_input
        if drone_input_frame is not None:
            cached_drone_frame = drone_input_frame
        if dog_input_frame is not None:
            cached_dog_frame = dog_input_frame
        if cached_drone_frame is None or cached_dog_frame is None:
            raise RuntimeError("The first policy step must capture both follower RGB frames")
        drone_input_frame = cached_drone_frame
        dog_input_frame = cached_dog_frame
        if metric_mask_step:
            assert drone_input_vis is not None
            assert drone_input_visible is not None
            assert drone_input_bbox is not None
            assert drone_bbox_norm is not None
            assert dog_input_vis is not None
            assert dog_input_visible is not None
            assert dog_input_bbox is not None
            assert dog_bbox_norm is not None
            cached_drone_mask_metrics = (
                float(drone_input_vis),
                bool(drone_input_visible),
                list(drone_input_bbox),
                list(drone_bbox_norm),
            )
            cached_dog_mask_metrics = (
                float(dog_input_vis),
                bool(dog_input_visible),
                list(dog_input_bbox),
                list(dog_bbox_norm),
            )
        assert cached_drone_mask_metrics is not None
        assert cached_dog_mask_metrics is not None
        (
            drone_input_vis,
            drone_input_visible,
            drone_input_bbox,
            drone_bbox_norm,
        ) = cached_drone_mask_metrics
        (
            dog_input_vis,
            dog_input_visible,
            dog_input_bbox,
            dog_bbox_norm,
        ) = cached_dog_mask_metrics
        # A deterministic episode advances exactly one dt per loop while the
        # world is paused during inference. Wall-clock inference latency must
        # not stretch the model's temporal history: training uses consecutive
        # simulation frames, not frames spaced by GPU/UnrealCV runtime.
        planner_observation_time = (
            float(step_idx) * float(args.dt)
            if bool(getattr(args, "deterministic_step", True))
            else observation_wall_time
        )
        episode_instruction = str(
            setup.get("instruction") or args.instruction
        )

        # 每一步调用模型推理
        gt_bbox_prior = args.bbox_source == "ground_truth"
        planner_pose_kwargs: dict[str, Any] = {}
        if bool(getattr(planner, "requires_inter_agent_pose", False)):
            # Agent poses are part of the air-ground observation contract. They
            # contain no target pose/box and are sampled at the same observation
            # instant as the two RGB frames.
            planner_pose_kwargs = {
                "drone_pose": list(env.unwrapped.obj_poses[setup["drone_id"]]),
                "robotdog_pose": list(env.unwrapped.obj_poses[setup["robotdog_id"]]),
            }
        intermediate_observation_debug: dict[str, Any] = {}
        if intermediate_observation_step:
            intermediate_observation_debug = dict(
                planner.observe(
                    drone_input_frame,
                    dog_input_frame,
                    observation_time=planner_observation_time,
                )
                or {}
            )
        if policy_inference_step:
            pred = planner.predict(
                drone_input_frame,
                dog_input_frame,
                drone_bbox_norm if gt_bbox_prior else None,
                dog_bbox_norm if gt_bbox_prior else None,
                episode_instruction,
                joint_instruction=episode_instruction,
                agent1_instruction=episode_instruction,
                agent2_instruction=episode_instruction,
                observation_time=planner_observation_time,
                drone_roi_bbox=drone_input_bbox if planner.use_roi_tokens else None,
                dog_roi_bbox=dog_input_bbox if planner.use_roi_tokens else None,
                drone_bbox_prompt=drone_bbox_norm if planner.use_bbox_text_prompt else None,
                dog_bbox_prompt=dog_bbox_norm if planner.use_bbox_text_prompt else None,
                **planner_pose_kwargs,
            )
        else:
            assert cached_policy_prediction is not None
            pred = dict(cached_policy_prediction)
            pred["global_encoding_time"] = 0.0
            pred["perception_time"] = 0.0
            pred["model_time"] = 0.0
            pred["model_time_seconds"] = 0.0
        model_time_seconds = pred.get("model_time")
        if model_time_seconds is None:
            model_time_seconds = (pred.get("model_time_seconds") if isinstance(pred, dict) else None)
        if policy_inference_step and model_time_seconds is not None:
            model_inference_seconds_total += max(float(model_time_seconds), 0.0)
            model_inference_steps += 1
        if policy_inference_step:
            drone_action, dog_action, action_debug = planner.waypoints_to_actions(
                pred["waypoints"],
                realtime_control_period_seconds=realtime_control_period_seconds,
            )
            cached_policy_prediction = dict(pred)
            cached_policy_waypoints = np.asarray(
                pred["waypoints"], dtype=np.float32
            ).copy()
            cached_drone_action = [float(value) for value in drone_action]
            cached_dog_action = [float(value) for value in dog_action]
            cached_action_debug = copy.deepcopy(action_debug)
        else:
            rollout_mode = str(
                getattr(args, "policy_action_rollout", "future_segment")
            )
            if rollout_mode == "future_segment":
                assert cached_policy_waypoints is not None
                rollout_offset = int(step_idx % policy_inference_stride)
                rollout_waypoints = _rollout_future_waypoint_segment(
                    cached_policy_waypoints,
                    rollout_offset,
                )
                pred["waypoints"] = rollout_waypoints
                drone_action, dog_action, action_debug = planner.waypoints_to_actions(
                    rollout_waypoints,
                    realtime_control_period_seconds=realtime_control_period_seconds,
                )
                action_debug["policy_rollout_offset"] = rollout_offset
            elif rollout_mode == "hold":
                assert cached_drone_action is not None
                assert cached_dog_action is not None
                assert cached_action_debug is not None
                drone_action = list(cached_drone_action)
                dog_action = list(cached_dog_action)
                action_debug = copy.deepcopy(cached_action_debug)
            else:
                raise ValueError(
                    f"Unsupported --policy-action-rollout {rollout_mode!r}"
                )
        action_debug["policy_inference_stride"] = policy_inference_stride
        action_debug["policy_interval_seconds"] = policy_inference_stride * float(args.dt)
        action_debug["environment_step_dt_seconds"] = float(args.dt)
        action_debug["waypoint_step_dt_seconds"] = float(
            args.waypoint_source_dt or args.dt
        )
        action_debug["history_frame_dt_seconds"] = float(
            getattr(planner, "history_frame_dt", args.dt)
        )
        action_debug["policy_inference_step"] = bool(policy_inference_step)
        action_debug["policy_action_held"] = bool(not policy_inference_step)
        action_debug["intermediate_history_observation"] = bool(
            intermediate_observation_step
        )
        action_debug["intermediate_history_encoding_time"] = float(
            intermediate_observation_debug.get("encoding_time", 0.0)
        )
        model_drone_action_before_override = [float(v) for v in drone_action]
        predicted_bbox = pred["refined_bbox"]
        bbox_display_label = pred.get("bbox_display_label") or [
            "model bbox",
            "model bbox",
        ]
        bbox_source_display = pred.get("bbox_source_display", args.bbox_source)
        bbox_fallback = pred.get("bbox_fallback_to_absolute") or [False, False]
        visible_score = pred.get("visible_score") or [0.0, 0.0]
        best_candidate = pred.get("best_candidate") or [None, None]
        best_candidate_score = pred.get("best_candidate_score") or [None, None]
        drone_bbox_iou = _bbox_iou_cxcywh(predicted_bbox[0], drone_bbox_norm)
        dog_bbox_iou = _bbox_iou_cxcywh(predicted_bbox[1], dog_bbox_norm)
        action_compute_wall_time = time.monotonic()
        observation_to_action_seconds = action_compute_wall_time - observation_wall_time

        actions = [None for _ in env.unwrapped.player_list]
        actions[setup["drone_id"]] = drone_action
        actions[setup["robotdog_id"]] = dog_action
        # This batch snapshot is observation_t. Reuse the same poses, RGB and
        # masks for model input, metrics and video so evaluation follows the
        # collection contract observation_t -> action_t.
        drone_pose = list(env.unwrapped.obj_poses[setup["drone_id"]])
        dog_pose = list(env.unwrapped.obj_poses[setup["robotdog_id"]])
        target_pose = list(env.unwrapped.obj_poses[setup["target_id"]])
        drone_pose_before_action = list(drone_pose)
        dog_pose_before_action = list(dog_pose)
        target_pose_before_action = list(target_pose)
        drone_frame = drone_input_frame
        dog_frame = dog_input_frame
        drone_vis = drone_input_vis
        dog_vis = dog_input_vis
        drone_visible = drone_input_visible
        dog_visible = dog_input_visible
        drone_bbox = drone_input_bbox
        dog_bbox = dog_input_bbox
        drone_bbox_norm_at_observation = drone_bbox_norm
        dog_bbox_norm_at_observation = dog_bbox_norm
        apply_oracle_heading_assist(
            args,
            drone_pose_before_action,
            dog_pose_before_action,
            target_pose_before_action,
            drone_action,
            dog_action,
            action_debug,
        )
        oracle_drone_action, oracle_drone_action_debug = recorded_drone_action_for_step(
            target_trajectory,
            args,
            step_idx,
        )
        action_debug["model_drone_action_before_oracle"] = model_drone_action_before_override
        action_debug["oracle_drone_action_debug"] = oracle_drone_action_debug
        if oracle_drone_action is not None:
            action_debug["model_drone_action_after_heading_assist"] = [float(v) for v in drone_action]
            action_debug["oracle_drone_action"] = [float(v) for v in oracle_drone_action]
            action_debug["action_source"] = (
                f"recorded_drone_{oracle_drone_action_debug.get('source', 'unknown')}"
            )
            drone_action = [float(v) for v in oracle_drone_action]
            executed_dt = max(
                float(action_debug.get("drone_action_dt_seconds") or getattr(args, "dt", 0.1)),
                1e-6,
            )
            action_debug["drone_physical_velocity_command"] = [
                float(drone_action[0]) / executed_dt,
                float(drone_action[1]) / executed_dt,
                float(drone_action[2]) / executed_dt,
            ]
            action_debug["drone_yaw_command"] = float(drone_action[3])
        actions[setup["drone_id"]] = drone_action
        oracle_robotdog_action, oracle_robotdog_action_debug = recorded_robotdog_action_for_step(
            target_trajectory,
            args,
            step_idx,
        )
        action_debug["oracle_robotdog_action_debug"] = oracle_robotdog_action_debug
        if oracle_robotdog_action is not None:
            action_debug["model_robotdog_action_before_oracle"] = [float(v) for v in dog_action]
            action_debug["oracle_robotdog_action"] = [float(v) for v in oracle_robotdog_action]
            dog_action = [float(v) for v in oracle_robotdog_action]
            action_debug["robotdog_action_source"] = (
                f"recorded_robotdog_{oracle_robotdog_action_debug.get('source', 'unknown')}"
            )
        actions[setup["robotdog_id"]] = dog_action
        recorded_step_timing_debug = apply_recorded_step_timing(
            env,
            setup,
            target_trajectory,
            args,
            step_idx,
        )
        if recorded_step_timing_debug is not None:
            action_debug["recorded_step_timing"] = recorded_step_timing_debug
        action_send_wall_time = time.monotonic()
        action_step_start_monotonic = time.monotonic()
        replay_distractor_actions = replay_distractor_actions_for_step(
            target_trajectory, step_idx
        )
        action_debug["replay_distractor_actions"] = replay_distractor_actions
        if bool(getattr(args, "deterministic_step", True)):
            _obs, _rewards, done, last_info, pulse_wall_seconds = deterministic_data_collection_step(
                env,
                args,
                setup,
                drone_action,
                dog_action,
                target_action=target_action,
                distractor_actions=replay_distractor_actions,
            )
        else:
            if target_action is not None:
                actions[setup["target_id"]] = target_action
            actions[setup["robotdog_id"]] = action_for_target_space(env, setup["robotdog_id"], dog_action)
            for actor_id, distractor_action in zip(
                setup.get("replay_distractor_ids", []), replay_distractor_actions
            ):
                actions[actor_id] = distractor_action
            pulse_t0 = time.monotonic()
            _obs, _rewards, done, last_info = data_collection_step_pose_only(env, actions)
            pulse_wall_seconds = time.monotonic() - pulse_t0
        action_step_seconds = time.monotonic() - action_step_start_monotonic
        drone_pose_after_action = list(env.unwrapped.obj_poses[setup["drone_id"]])
        dog_pose_after_action = list(env.unwrapped.obj_poses[setup["robotdog_id"]])
        target_pose_after_action = list(env.unwrapped.obj_poses[setup["target_id"]])
        realized_dt_seconds = (
            float(args.dt)
            if bool(getattr(args, "deterministic_step", True))
            else max(float(realtime_control_period_seconds), 1e-6)
        )
        drone_realized_velocity = measured_body_velocity(
            drone_pose_before_action,
            drone_pose_after_action,
            realized_dt_seconds,
        )
        dog_realized_velocity = measured_body_velocity(
            dog_pose_before_action,
            dog_pose_after_action,
            realized_dt_seconds,
        )
        drone_commanded_velocity = [
            float(drone_action[0]) / realized_dt_seconds,
            float(drone_action[1]) / realized_dt_seconds,
            float(drone_action[3]),
        ]
        dog_commanded_velocity = [
            float(dog_action[1]) / UNREAL_UNITS_PER_METER,
            0.0,
            math.radians(float(dog_action[0])) / realized_dt_seconds,
        ]
        if (
            str(getattr(args, "oracle_drone_action_source", "none")) == "none"
            and str(getattr(args, "oracle_robotdog_action_source", "none")) == "none"
        ):
            planner.update_realized_velocities(
                drone_realized_velocity,
                dog_realized_velocity,
            )

        # There is deliberately no post-action RGB+mask request here. The next
        # loop's single batch request becomes observation_{t+1}. Post-action
        # collision checks use the pose-only refresh from the action step.
        drone_dist = distance_xy_m(drone_pose, target_pose)
        dog_dist = distance_xy_m(dog_pose, target_pose)
        drone_dist_after_action = distance_xy_m(drone_pose_after_action, target_pose_after_action)
        dog_dist_after_action = distance_xy_m(dog_pose_after_action, target_pose_after_action)
        drone_human_collision_current = (
            drone_dist_after_action < float(args.human_collision_distance)
        )
        robotdog_human_collision_current = (
            dog_dist_after_action < float(args.human_collision_distance)
        )
        drone_human_collision = drone_human_collision or drone_human_collision_current
        robotdog_human_collision = robotdog_human_collision or robotdog_human_collision_current
        human_collision = human_collision or (
            drone_human_collision_current or robotdog_human_collision_current
        )
        drone_collision = drone_collision_from_info(
            last_info,
            setup["drone_id"],
            setup["target_id"],
            drone_dist_after_action,
            drone_pose_after_action,
            target_pose_after_action,
        )
        dog_collision = robotdog_collision_from_info(
            last_info,
            setup["robotdog_id"],
            setup["target_id"],
            dog_dist_after_action,
            dog_pose_after_action,
            target_pose_after_action,
        )
        physical_collision = physical_collision or drone_collision or dog_collision
        # The evaluation contract treats either a UE physical contact or an
        # after-action human distance strictly below the configured threshold
        # as a collision.  This primary flag controls early stopping, SR and
        # aggregate CR; the individual fields remain available for analysis.
        collision = physical_collision or human_collision

        drone_follow_distance = drone_dist_after_action
        dog_follow_distance = dog_dist_after_action
        drone_following = bool(
            drone_visible
            and args.drone_min_follow_dist <= drone_follow_distance <= args.drone_max_follow_dist
        )
        dog_following = bool(
            dog_visible
            and args.robotdog_min_follow_dist <= dog_follow_distance <= args.robotdog_max_follow_dist
        )
        joint_following = bool(drone_following and dog_following)
        if joint_following:
            lost_count = 0
            failure_count = 0
        else:
            too_far = (
                drone_follow_distance > args.drone_lost_distance
                or dog_follow_distance > args.robotdog_lost_distance
            )
            # Legacy OpenTrackVLA declares Lost only after consecutive
            # out-of-range frames. Temporary invisibility lowers TR but does
            # not itself advance the Lost counter.
            lost_count = lost_count + 1 if too_far else 0
            if step_idx + 1 > args.failure_warmup_steps:
                failure_count += 1

        global_frame = None
        if args.save_video and args.write_global_video:
            global_frame = get_global_frame(
                env, args, target_pose, dog_pose, drone_pose
            )
        loop_end_monotonic = time.monotonic()
        loop_wall_time_seconds = loop_end_monotonic - loop_start_monotonic
        drone_vis_frame = drone_input_frame
        dog_vis_frame = dog_input_frame
        if args.save_video and args.trajectory_overlay:
            drone_vis_frame = _render_bgr_frame_with_traj(
                drone_vis_frame, pred["waypoints"][0], scale=args.trajectory_scale
            )
            dog_vis_frame = _render_bgr_frame_with_traj(
                dog_vis_frame, pred["waypoints"][1], scale=args.trajectory_scale
            )
        if (
            args.save_video
            and planner.use_roi_tokens
            and pred.get("roi_crop_xyxy") is not None
        ):
            drone_vis_frame = _draw_roi_crop_xyxy(drone_vis_frame, pred["roi_crop_xyxy"][0], "oracle ROI crop")
            dog_vis_frame = _draw_roi_crop_xyxy(dog_vis_frame, pred["roi_crop_xyxy"][1], "oracle ROI crop")
        if args.save_video:
            drone_vis_frame = _overlay_text(
                _draw_predicted_bbox(
                    drone_vis_frame,
                    predicted_bbox[0],
                    str(bbox_display_label[0]),
                ),
                [
                    f"ep={episode_id} step={step_idx + 1}",
                    f"trajectory={_trajectory_source_label(action_debug, 0)}",
                    f"drone d={drone_dist:.2f} bbox_iou={drone_bbox_iou:.2f}",
                    f"bbox_source={bbox_source_display} vis={float(visible_score[0]):.2f} abs_fallback={int(bbox_fallback[0])}",
                    (
                        f"roi={pred.get('roi_bbox_source')} valid={int(bool((pred.get('roi_valid') or [False, False])[0]))}"
                        if planner.use_roi_tokens
                        else "roi=off"
                    ),
                    _candidate_label(best_candidate[0], best_candidate_score[0]),
                    "cmd_v="
                    f"[{action_debug['drone_physical_velocity_command'][0]:.2f},"
                    f"{action_debug['drone_physical_velocity_command'][1]:.2f}]m/s "
                    f"yaw={drone_action[3]:.2f}rad/s",
                    "ctrl="
                    f"{action_debug.get('waypoint_control_mode', args.waypoint_control_mode)} "
                    f"env=[{drone_action[0]:.3f},{drone_action[1]:.3f},"
                    f"{drone_action[2]:.3f},{drone_action[3]:.3f}]",
                ],
            )
            dog_vis_frame = _overlay_text(
                _draw_predicted_bbox(
                    dog_vis_frame,
                    predicted_bbox[1],
                    str(bbox_display_label[1]),
                ),
                [
                    f"ep={episode_id} step={step_idx + 1}",
                    f"trajectory={_trajectory_source_label(action_debug, 1)}",
                    f"dog d={dog_dist:.2f} bbox_iou={dog_bbox_iou:.2f}",
                    f"bbox_source={bbox_source_display} vis={float(visible_score[1]):.2f} abs_fallback={int(bbox_fallback[1])}",
                    (
                        f"roi={pred.get('roi_bbox_source')} valid={int(bool((pred.get('roi_valid') or [False, False])[1]))}"
                        if planner.use_roi_tokens
                        else "roi=off"
                    ),
                    _candidate_label(best_candidate[1], best_candidate_score[1]),
                    f"cmd_v={dog_action[1] / UNREAL_UNITS_PER_METER:.2f}m/s "
                    f"turn={dog_action[0]:.1f}deg",
                    "ctrl="
                    f"{action_debug.get('waypoint_control_mode', args.waypoint_control_mode)} "
                    f"env=[turn={dog_action[0]:.1f}deg,speed="
                    f"{dog_action[1] / UNREAL_UNITS_PER_METER:.2f}m/s]",
                ],
            )
            frames_drone.append(drone_vis_frame)
            frames_dog.append(dog_vis_frame)
            if args.write_global_video and global_frame is not None:
                frames_global.append(
                    _overlay_text(
                        global_frame,
                        [
                            f"ep={episode_id} step={step_idx + 1}",
                            f"drone trajectory={_trajectory_source_label(action_debug, 0)}",
                            f"robotdog trajectory={_trajectory_source_label(action_debug, 1)}",
                        ],
                    )
                )

        roi_valid_values = pred.get("roi_valid") or [None, None]
        roi_crop_values = pred.get("roi_crop_xyxy") or [None, None]
        drone_infos.append(
            {
                "step": step_idx + 1,
                "dis_to_human": float(drone_dist),
                "dis_to_human_3d": float(distance_m(drone_pose, target_pose)),
                "facing": 1.0 if drone_visible else 0.0,
                "target_visible": bool(drone_visible),
                "target_visibility": float(drone_vis),
                "target_bbox": drone_bbox,
                "bbox_feat": drone_bbox_norm_at_observation,
                "model_bbox_input": pred.get("bbox_input", [None, None])[0] if pred.get("bbox_input") else None,
                "roi_bbox_source": pred.get("roi_bbox_source"),
                "evaluation_protocol": pred.get("evaluation_protocol"),
                "roi_valid": roi_valid_values[0],
                "roi_crop_xyxy": roi_crop_values[0],
                "roi_expand_ratio": pred.get("roi_expand_ratio"),
                "roi_token_count": pred.get("roi_token_count"),
                "bbox_prompt_text": pred.get("bbox_prompt_text"),
                "global_encoding_time_s": pred.get("global_encoding_time"),
                "model_time_s": None if model_time_seconds is None else float(model_time_seconds),
                "roi_encoding_time_s": pred.get("roi_encoding_time"),
                "predicted_bbox": predicted_bbox[0],
                "raw_refined_bbox": pred.get("raw_refined_bbox", [None, None])[0],
                "absolute_bbox": pred.get("absolute_bbox", [None, None])[0],
                "bbox_fallback_to_absolute": bool(bbox_fallback[0]),
                "bbox_iou": float(drone_bbox_iou),
                "predicted_visible_score": float(visible_score[0]),
                "input_target_visible": bool(drone_input_visible),
                "input_target_visibility": float(drone_input_vis),
                "predicted_waypoints": pred["waypoints"][0].tolist(),
                "waypoint_control_mode": action_debug.get("waypoint_control_mode", args.waypoint_control_mode),
                "inverse_control": action_debug.get("drone_inverse"),
                "best_candidate": best_candidate[0],
                "best_candidate_score": best_candidate_score[0],
                "target_center_error": bbox_center_error(drone_bbox, args),
                "target_centered": bool(drone_visible and bbox_centered(drone_bbox, args)),
                "base_velocity": drone_realized_velocity,
                "desired_base_velocity": action_debug["drone_velocity_pred"],
                "commanded_base_velocity": drone_commanded_velocity,
                "base_velocity_dt_s": float(realized_dt_seconds),
                "model_drone_action": action_debug.get("model_drone_action_before_oracle"),
                "oracle_drone_action": action_debug.get("oracle_drone_action"),
                "oracle_drone_action_debug": action_debug.get("oracle_drone_action_debug"),
                "drone_action": [float(v) for v in drone_action],
                "env_action": [float(v) for v in drone_action],
                "env_action_space": "drone set_move_bp raw [x_step_like, y_step_like, z_step_like, yaw_rate]",
                "command_label_source": action_debug.get("action_source", "model_waypoint_to_env_action"),
                "obs_action_alignment": "obs_t_action_t",
                "snap_heading": bool(args.snap_heading),
                "yaw_control_mode": (
                    "recorded_oracle_action"
                    if action_debug.get("oracle_drone_action") is not None
                    else "oracle_snap"
                    if args.snap_heading
                    else "action"
                ),
                "snapshot_query_time_s": float(snapshot_query_seconds),
                "observation_timestamp_s": float(observation_wall_time),
                "action_compute_timestamp_s": float(action_compute_wall_time),
                "action_send_timestamp_s": float(action_send_wall_time),
                "loop_end_monotonic_s": float(loop_end_monotonic),
                "loop_wall_time_s": float(loop_wall_time_seconds),
                "realtime_control_period_seconds": float(realtime_control_period_seconds),
                "fixed_timestep_seconds": (
                    float(args.dt)
                    if bool(getattr(args, "deterministic_step", True))
                    else None
                ),
                "action_pulse_displacement_m": float(
                    distance_xy_m(drone_pose_before_action, drone_pose_after_action)
                ),
                "action_pulse_wall_seconds": float(pulse_wall_seconds),
                "action_step_time_s": float(action_step_seconds),
                "following": bool(drone_following),
                "following_distance_m": float(drone_follow_distance),
                "following_distance_source": "after_action",
                "collision": bool(drone_collision),
                "human_collision": bool(drone_human_collision),
                "human_collision_current": bool(drone_human_collision_current),
                "human_collision_distance_m": float(args.human_collision_distance),
                "drone_pose": drone_pose,
                "drone_pose_after_action": drone_pose_after_action,
                "target_pose": target_pose,
                "target_pose_after_action": target_pose_after_action,
                "dis_to_human_after_action": float(drone_dist_after_action),
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
                "bbox_feat": dog_bbox_norm_at_observation,
                "model_bbox_input": pred.get("bbox_input", [None, None])[1] if pred.get("bbox_input") else None,
                "roi_bbox_source": pred.get("roi_bbox_source"),
                "evaluation_protocol": pred.get("evaluation_protocol"),
                "roi_valid": roi_valid_values[1],
                "roi_crop_xyxy": roi_crop_values[1],
                "roi_expand_ratio": pred.get("roi_expand_ratio"),
                "roi_token_count": pred.get("roi_token_count"),
                "global_encoding_time_s": pred.get("global_encoding_time"),
                "perception_time_s": pred.get("perception_time"),
                "model_time_s": None if model_time_seconds is None else float(model_time_seconds),
                "roi_encoding_time_s": pred.get("roi_encoding_time"),
                "predicted_bbox": predicted_bbox[1],
                "raw_refined_bbox": pred.get("raw_refined_bbox", [None, None])[1],
                "absolute_bbox": pred.get("absolute_bbox", [None, None])[1],
                "bbox_fallback_to_absolute": bool(bbox_fallback[1]),
                "bbox_iou": float(dog_bbox_iou),
                "predicted_visible_score": float(visible_score[1]),
                "input_target_visible": bool(dog_input_visible),
                "input_target_visibility": float(dog_input_vis),
                "predicted_waypoints": pred["waypoints"][1].tolist(),
                "waypoint_control_mode": action_debug.get("waypoint_control_mode", args.waypoint_control_mode),
                "inverse_control": action_debug.get("robotdog_inverse"),
                "best_candidate": best_candidate[1],
                "best_candidate_score": best_candidate_score[1],
                "target_center_error": bbox_center_error(dog_bbox, args),
                "target_centered": bool(dog_visible and bbox_centered(dog_bbox, args)),
                "base_velocity": dog_realized_velocity,
                "desired_base_velocity": action_debug["robotdog_velocity_pred"],
                "commanded_base_velocity": dog_commanded_velocity,
                "base_velocity_dt_s": float(realized_dt_seconds),
                "model_robotdog_action": action_debug.get("model_robotdog_action_before_oracle"),
                "oracle_robotdog_action": action_debug.get("oracle_robotdog_action"),
                "oracle_robotdog_action_debug": action_debug.get("oracle_robotdog_action_debug"),
                "ground_action": [float(v) for v in dog_action],
                "env_action": [float(v) for v in dog_action],
                "env_action_space": "robotdog set_move_bp [turn_deg, speed_cm_s]",
                "command_label_source": action_debug.get("robotdog_action_source", "model_waypoint_to_env_action"),
                "obs_action_alignment": "obs_t_action_t",
                "snap_heading": bool(args.snap_heading),
                "yaw_control_mode": "oracle_snap" if args.snap_heading else "action",
                "snapshot_query_time_s": float(snapshot_query_seconds),
                "observation_timestamp_s": float(observation_wall_time),
                "action_compute_timestamp_s": float(action_compute_wall_time),
                "action_send_timestamp_s": float(action_send_wall_time),
                "loop_end_monotonic_s": float(loop_end_monotonic),
                "loop_wall_time_s": float(loop_wall_time_seconds),
                "realtime_control_period_seconds": float(realtime_control_period_seconds),
                "fixed_timestep_seconds": (
                    float(args.dt)
                    if bool(getattr(args, "deterministic_step", True))
                    else None
                ),
                "action_pulse_displacement_m": float(
                    distance_xy_m(dog_pose_before_action, dog_pose_after_action)
                ),
                "action_pulse_wall_seconds": float(pulse_wall_seconds),
                "following": bool(dog_following),
                "following_distance_m": float(dog_follow_distance),
                "following_distance_source": "after_action",
                "collision": bool(dog_collision),
                "human_collision": bool(robotdog_human_collision),
                "human_collision_current": bool(robotdog_human_collision_current),
                "human_collision_distance_m": float(args.human_collision_distance),
                "robotdog_lateral_ignored": action_debug["robotdog_lateral_ignored"],
                "robotdog_pose": dog_pose,
                "robotdog_pose_after_action": dog_pose_after_action,
                "target_pose": target_pose,
                "target_pose_after_action": target_pose_after_action,
                "dis_to_human_after_action": float(dog_dist_after_action),
            }
        )
        combined_infos.append(
            {
                "step": step_idx + 1,
                "joint_following": joint_following,
                "drone_following": bool(drone_following),
                "robotdog_following": bool(dog_following),
                "drone_following_distance": float(drone_follow_distance),
                "robotdog_following_distance": float(dog_follow_distance),
                "following_distance_source": "after_action",
                "collision": bool(collision),
                "physical_collision": bool(drone_collision or dog_collision),
                "human_collision": bool(human_collision),
                "human_collision_current": bool(
                    drone_human_collision_current or robotdog_human_collision_current
                ),
                "drone_human_collision": bool(drone_human_collision),
                "robotdog_human_collision": bool(robotdog_human_collision),
                "human_collision_distance_m": float(args.human_collision_distance),
                "lost_count": int(lost_count),
                "failure_count": int(failure_count),
                "visible_score": pred.get("visible_score"),
                "refined_bbox": pred.get("refined_bbox"),
                "best_candidate": pred.get("best_candidate"),
                "best_candidate_score": pred.get("best_candidate_score"),
                "bbox_source": args.bbox_source,
                "bbox_input": pred.get("bbox_input"),
                "roi_bbox_source": pred.get("roi_bbox_source"),
                "evaluation_protocol": pred.get("evaluation_protocol"),
                "roi_valid": pred.get("roi_valid"),
                "roi_crop_xyxy": pred.get("roi_crop_xyxy"),
                "roi_expand_ratio": pred.get("roi_expand_ratio"),
                "roi_token_count": pred.get("roi_token_count"),
                "global_encoding_time_s": pred.get("global_encoding_time"),
                "perception_time_s": pred.get("perception_time"),
                "model_time_s": None if model_time_seconds is None else float(model_time_seconds),
                "roi_encoding_time_s": pred.get("roi_encoding_time"),
                "bbox_fallback_to_absolute": bbox_fallback,
                "target_action": _jsonable_action(target_action),
                "target_action_debug": setup.get("target_path_action_debug"),
                "oracle_drone_action": action_debug.get("oracle_drone_action"),
                "oracle_drone_action_debug": action_debug.get("oracle_drone_action_debug"),
                "oracle_robotdog_action": action_debug.get("oracle_robotdog_action"),
                "oracle_robotdog_action_debug": action_debug.get("oracle_robotdog_action_debug"),
                "recorded_step_timing": action_debug.get("recorded_step_timing"),
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
                "snapshot_query_time_s": float(snapshot_query_seconds),
                "action_step_time_s": float(action_step_seconds),
                "observation_timestamp_s": float(observation_wall_time),
                "action_compute_timestamp_s": float(action_compute_wall_time),
                "action_send_timestamp_s": float(action_send_wall_time),
                "loop_end_monotonic_s": float(loop_end_monotonic),
                "loop_wall_time_s": float(loop_wall_time_seconds),
                "realtime_control_period_seconds": float(realtime_control_period_seconds),
                "deterministic_step": bool(getattr(args, "deterministic_step", True)),
                "fixed_timestep_seconds": (
                    float(args.dt)
                    if bool(getattr(args, "deterministic_step", True))
                    else None
                ),
                "obs_action_alignment": "obs_t_action_t",
                "snap_heading": bool(args.snap_heading),
                "yaw_control_mode": "oracle_snap" if args.snap_heading else "action",
                "observation_snapshot_count": 1,
                "metric_mask_sampled": bool(metric_mask_step),
                "post_action_rgb_mask_capture": False,
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
                "target_action": _jsonable_action(target_action),
                "target_action_debug": setup.get("target_path_action_debug"),
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
                "fixed_timestep_seconds": (
                    float(args.dt)
                    if bool(getattr(args, "deterministic_step", True))
                    else None
                ),
                "obs_action_alignment": "obs_t_action_t",
                "snap_heading": bool(args.snap_heading),
                "yaw_control_mode": "oracle_snap" if args.snap_heading else "action",
                "observation_snapshot_count": 1,
                "post_action_rgb_mask_capture": False,
                "drone_visible": bool(drone_visible),
                "robotdog_visible": bool(dog_visible),
                "joint_following": bool(joint_following),
                "action_debug": action_debug,
                "roi_bbox_source": pred.get("roi_bbox_source"),
                "evaluation_protocol": pred.get("evaluation_protocol"),
                "roi_valid": pred.get("roi_valid"),
                "roi_crop_xyxy": pred.get("roi_crop_xyxy"),
                "roi_expand_ratio": pred.get("roi_expand_ratio"),
                "roi_token_count": pred.get("roi_token_count"),
                "global_encoding_time_s": pred.get("global_encoding_time"),
                "roi_encoding_time_s": pred.get("roi_encoding_time"),
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
    drone_centered_rate = sum(
        1 for item in drone_infos if item.get("target_centered", False)
    ) / max(total_step, 1)
    dog_centered_rate = sum(
        1 for item in dog_infos if item.get("target_centered", False)
    ) / max(total_step, 1)
    drone_bbox_iou_mean = sum(item["drone_bbox_iou"] for item in combined_infos) / max(total_step, 1)
    dog_bbox_iou_mean = sum(item["robotdog_bbox_iou"] for item in combined_infos) / max(total_step, 1)
    visible_correct = sum(
        int(item["drone_visible_correct"]) + int(item["robotdog_visible_correct"])
        for item in combined_infos
    )
    visible_accuracy = visible_correct / max(total_step * 2, 1)
    def _timing_sum(key: str) -> float:
        return sum(float(item.get(key) or 0.0) for item in combined_infos)

    # Match the legacy OpenTrackVLA terminal-success semantics. Per-step TR
    # remains upper-bound-only, while an episode that ends early must finish
    # inside both agents' configured follow-distance bands. A full-horizon
    # episode falls back to the final per-step joint-following result.
    completed_full_horizon = bool(total_step >= episode_max_steps)
    final_drone_distance = (
        float(drone_infos[-1].get("dis_to_human_after_action", drone_infos[-1]["dis_to_human"]))
        if drone_infos
        else float("inf")
    )
    final_dog_distance = (
        float(dog_infos[-1].get("dis_to_human_after_action", dog_infos[-1]["dis_to_human"]))
        if dog_infos
        else float("inf")
    )
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
            or status in {
                "Success",
                "Lost",
                "Collision",
                "PersistentFailure",
                "Timeout",
                "TargetStopped",
                "EnvDone",
            }
        ),
        "status": status,
        "success": 1.0 if success else 0.0,
        "total_step": total_step,
        "collision": 1.0 if collision else 0.0,
        "physical_collision": 1.0 if physical_collision else 0.0,
        "human_collision": 1.0 if human_collision else 0.0,
        "drone_human_collision": 1.0 if drone_human_collision else 0.0,
        "robotdog_human_collision": 1.0 if robotdog_human_collision else 0.0,
        "human_collision_distance_m": float(args.human_collision_distance),
        "joint_following_rate": float(joint_rate),
        "drone_following_rate": float(drone_rate),
        "robotdog_following_rate": float(dog_rate),
        "drone_centered_rate": float(drone_centered_rate),
        "robotdog_centered_rate": float(dog_centered_rate),
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
        "use_roi_tokens": bool(planner.use_roi_tokens),
        "roi_bbox_source": planner.roi_bbox_source if planner.use_roi_tokens else None,
        "evaluation_protocol": planner.evaluation_protocol if planner.use_roi_tokens else None,
        "roi_expand_ratio": float(planner.roi_expand_ratio) if planner.use_roi_tokens else None,
        "roi_token_count": int(planner.roi_token_count) if planner.use_roi_tokens else None,
        "roi_make_square": bool(planner.roi_make_square) if planner.use_roi_tokens else None,
        "use_visual_section_markers": bool(planner.use_visual_section_markers),
        "visual_section_marker_version": (
            "global_history_current_target_roi_markers_v1"
            if planner.use_visual_section_markers
            else None
        ),
        "roi_prompt_version": "roi_visual_layout_prompt_v1" if planner.use_roi_tokens else None,
        "bbox_usage_note": (
            "roi_bbox is ground-truth and used only for image cropping; bbox_feat=None."
            if planner.use_roi_tokens
            else None
        ),
        "ckpt_bbox_dropout_prob": float(planner.ckpt_bbox_dropout_prob),
        "instruction": setup.get("instruction") or args.instruction,
        "joint_instruction": setup.get("instruction") or getattr(args, "joint_instruction", None) or args.instruction,
        "agent1_instruction": setup.get("instruction") or getattr(args, "agent1_instruction", None),
        "agent2_instruction": setup.get("instruction") or getattr(args, "agent2_instruction", None),
        "replay_distractors": bool(setup.get("replay_distractors", False)),
        "replay_distractor_count": int(setup.get("replay_distractor_count", 0)),
        "metric_target_actor_id": int(setup["target_id"]),
        "metric_target_source": "env.obj_poses[target_id]",
        # Legacy end-to-end episode throughput retained for compatibility.
        "fps": total_step / elapsed,
        # Requested model-only speed. This is the reciprocal of the mean pure
        # forward latency and excludes all non-model work.
        "model_inference_seconds_total": float(model_inference_seconds_total),
        "model_inference_steps": int(model_inference_steps),
        "model_latency_ms": (
            float(model_inference_seconds_total / model_inference_steps * 1000.0)
            if model_inference_steps else 0.0
        ),
        "model_fps": (
            float(model_inference_steps / model_inference_seconds_total)
            if model_inference_seconds_total > 0.0 else 0.0
        ),
        "snapshot_seconds_total": _timing_sum("snapshot_query_time_s"),
        "planner_seconds_total": _timing_sum("observation_to_action_seconds"),
        "vision_encoding_seconds_total": _timing_sum("global_encoding_time_s"),
        "perception_seconds_total": _timing_sum("perception_time_s"),
        "action_pulse_seconds_total": _timing_sum("action_pulse_wall_seconds"),
        "action_step_seconds_total": _timing_sum("action_step_time_s"),
        "loop_seconds_total": _timing_sum("loop_wall_time_s"),
        "ckpt": str(planner.ckpt_path),
        "env_id": args.env_id,
        "model_type": "multi_agent",
        "target_motion_mode": setup["target_motion_mode"],
        "target_stopped": bool(setup.get("target_stopped", False)),
        "target_stop_step": setup.get("target_stop_step"),
        "target_stop_wait_count": int(setup.get("target_stop_wait_count", 0)),
        "target_stop_wait_steps": int(setup["target_stop_wait_steps"]),
        "oracle_drone_action_source": str(getattr(args, "oracle_drone_action_source", "none")),
        "oracle_drone_action_hold_last": bool(getattr(args, "oracle_drone_action_hold_last", True)),
        "oracle_robotdog_action_source": str(getattr(args, "oracle_robotdog_action_source", "none")),
        "oracle_robotdog_action_hold_last": bool(getattr(args, "oracle_robotdog_action_hold_last", True)),
        "oracle_recorded_step_timing": bool(getattr(args, "oracle_recorded_step_timing", False)),
        "target_goal_reach_distance": float(args.target_goal_reach_distance),
        "action_pulse_control": bool(getattr(args, "deterministic_step", True)),
        "obs_action_alignment": "obs_t_action_t",
        "snap_heading": bool(args.snap_heading),
        "yaw_control_mode": "oracle_snap" if args.snap_heading else "action",
        "observation_snapshot_policy": (
            "sampled_rgb_and_mask_with_pose_only_intermediate_steps"
            if int(getattr(args, "metric_mask_stride", 1) or 1) > 1
            else "rgb_mask_on_policy_steps_mask_only_between_policy_steps"
            if int(getattr(args, "policy_inference_stride", 1) or 1) > 1
            and bool(getattr(args, "skip_rgb_between_policy_steps", True))
            else "one_pre_action_rgb_mask_batch_per_step"
        ),
        "policy_inference_stride": int(
            getattr(args, "policy_inference_stride", 1) or 1
        ),
        "policy_interval_seconds": int(
            getattr(args, "policy_inference_stride", 1) or 1
        ) * float(args.dt),
        "environment_step_dt_seconds": float(args.dt),
        "waypoint_step_dt_seconds": float(args.waypoint_source_dt or args.dt),
        "policy_action_rollout": str(
            getattr(args, "policy_action_rollout", "future_segment")
        ),
        "skip_rgb_between_policy_steps": bool(
            getattr(args, "skip_rgb_between_policy_steps", True)
        ),
        "mask_image_format": str(getattr(args, "mask_image_format", "png")),
        "metric_mask_stride": int(getattr(args, "metric_mask_stride", 1) or 1),
        "metric_mask_sampling": (
            "exact_every_step"
            if int(getattr(args, "metric_mask_stride", 1) or 1) == 1
            else "periodic_zero_order_hold_with_exact_target_stopped_steps"
        ),
        "reuse_post_action_poses": bool(
            getattr(args, "reuse_post_action_poses", False)
        ),
        "post_action_rgb_mask_capture": False,
        "deterministic_step": bool(getattr(args, "deterministic_step", True)),
        "deterministic_pause_check_stride": int(
            getattr(args, "deterministic_pause_check_stride", 1) or 0
        ),
        "fixed_timestep_seconds": (
            float(args.dt) if bool(getattr(args, "deterministic_step", True)) else None
        ),
        "ue_interval_ms": int(getattr(args, "ue_interval_ms", None) or round(float(args.dt) * 1000.0)),
        "bp_interval_s": float(getattr(args, "ue_interval_ms", None) or round(float(args.dt) * 1000.0)) / 1000.0,
        "velocity_feedback_config": setup.get("velocity_feedback"),
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
            "following_distance_source": "after_action",
            "drone_final_range": [
                float(args.drone_min_follow_dist),
                float(args.drone_max_follow_dist),
            ],
            "robotdog_final_range": [
                float(args.robotdog_min_follow_dist),
                float(args.robotdog_max_follow_dist),
            ],
            "joint_following_rate_diagnostic_only": True,
            "centered_rate_diagnostic_only": True,
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
    parser.add_argument("--save-path", default="/data/yh/newtrackvla修改/newtrackvla_base_yh_clean/output/eval_unrealzoo_multi_agent")
    parser.add_argument("--env-id", default=DEFAULT_ENV_ID)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument(
        "--recorded-target-dir",
        default=None,
        help="Replay the recorded human trajectory from *_drone_info.json with action pulses while agents remain closed-loop.",
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
    parser.add_argument(
        "--use-bbox-text-prompt",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Append per-step bbox spatial text prompt; default follows checkpoint config.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--llm-name", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--text-max-length", type=int, default=192)
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
    parser.add_argument("--vision-resize-mode", choices=("letterbox", "stretch"), default="letterbox")
    parser.add_argument(
        "--vision-amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run DINO/SigLIP image encoders under CUDA bfloat16 autocast when available.",
    )
    parser.add_argument(
        "--inference-amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the V3 model forward under CUDA bfloat16 autocast when available.",
    )
    parser.add_argument(
        "--perception-config",
        type=Path,
        default=None,
        help="YOLO perception YAML used by air-ground cooperative inference.",
    )
    parser.add_argument("--yolo-weights", type=Path, default=None)
    parser.add_argument("--yolo-image-size", type=int, default=None)
    parser.add_argument("--person-confidence", type=float, default=None)
    parser.add_argument("--object-confidence", type=float, default=None)
    parser.add_argument(
        "--yolo-half",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use FP16 for online YOLO inference; defaults to the perception YAML.",
    )
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
    parser.add_argument(
        "--use-roi-tokens",
        action="store_true",
        help=(
            "Enable oracle target ROI visual tokens. In this protocol bbox is used only to crop ROI images; "
            "bbox_feat and bbox tokens are disabled."
        ),
    )
    parser.add_argument(
        "--roi-bbox-source",
        choices=["ground_truth", "external_detector", "external_tracker", "previous_model_prediction"],
        default="ground_truth",
        help="Source for ROI crop boxes. Only ground_truth is implemented in this oracle upper-bound version.",
    )
    parser.add_argument("--roi-expand-ratio", type=float, default=1.5)
    parser.add_argument("--roi-token-count", type=int, default=16)
    parser.add_argument("--roi-make-square", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--roi-override-checkpoint-config",
        action="store_true",
        help="Use command-line ROI crop settings instead of ROI settings saved in the checkpoint config.",
    )
    parser.add_argument(
        "--use-visual-section-markers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Insert learned GLOBAL_HISTORY/GLOBAL_CURRENT/TARGET_ROI section marker tokens. Usually read from ROI checkpoint config.",
    )

    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument(
        "--ue-interval-ms",
        type=int,
        default=None,
        help="BP movement interval. Deterministic evaluation requires round(dt*1000).",
    )
    parser.add_argument(
        "--deterministic-step",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fixed-step evaluation mode. Default aligns eval with step-based training labels.",
    )
    parser.add_argument(
        "--deterministic-pause-check-stride",
        type=int,
        default=1,
        help=(
            "Verify UE paused state every N physics steps in deterministic "
            "mode; 1 is the canonical defensive check and 0 disables polling."
        ),
    )
    parser.add_argument(
        "--realtime-waypoint-timing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Optional realtime mode: keep simulation running during inference and select the waypoint nearest to the "
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
    parser.add_argument(
        "--fast-eval-io",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Skip unused post-action rendering and capture only the two "
            "follower RGB/mask views. This preserves closed-loop actions and "
            "metrics; intended for the isolated V3 fast evaluator."
        ),
    )
    parser.add_argument(
        "--mask-image-format",
        choices=("png", "bmp"),
        default="png",
        help=(
            "Wire format for exact per-step UnrealCV object masks. BMP avoids "
            "PNG compression/decompression CPU cost at the expense of larger "
            "local-socket payloads; png preserves the canonical default."
        ),
    )
    parser.add_argument(
        "--metric-mask-stride",
        type=int,
        default=1,
        help=(
            "Capture exact GT object masks every N physics steps and use the "
            "latest sample between captures. Target-stopped terminal steps "
            "always capture exact masks. 1 preserves exact per-step metrics."
        ),
    )
    parser.add_argument(
        "--reuse-post-action-poses",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "In deterministic paused-world evaluation, reuse the exact poses "
            "already refreshed after the prior action instead of querying the "
            "same object poses again with the next image snapshot."
        ),
    )
    parser.add_argument(
        "--policy-inference-stride",
        type=int,
        default=1,
        help=(
            "Run the visual policy every N deterministic physics steps and hold "
            "the most recent action between policy steps. Physics and GT-mask "
            "metrics remain evaluated every step; 1 preserves the canonical protocol."
        ),
    )
    parser.add_argument(
        "--skip-rgb-between-policy-steps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When policy-inference-stride > 1, request pose+object-mask without "
            "unused RGB on held-action steps."
        ),
    )
    parser.add_argument(
        "--policy-action-rollout",
        choices=("future_segment", "hold"),
        default="future_segment",
        help=(
            "Between policy inferences, either execute successive cached future "
            "trajectory segments or repeat the first action."
        ),
    )
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

    parser.add_argument(
        "--human-speed",
        type=float,
        default=0.5,
        help="Target human speed in m/s; arbitrary positive values are accepted.",
    )
    parser.add_argument(
        "--environment-ground-max-speed-mps",
        type=float,
        default=100.0,
        help="UE BP max-speed ceiling for human and robotdog; 100 matches the verified fixed-step replay without clipping actions.",
    )
    parser.add_argument(
        "--ground-acceleration",
        type=float,
        default=10000.0,
        help="UE BP human/robotdog acceleration; 10000 matches the verified fixed-step replay response.",
    )
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
    parser.add_argument("--require-visual-target", action=argparse.BooleanOptionalAction, default=True)
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
    parser.add_argument("--robotdog-min-follow-dist", type=float, default=1.0)
    parser.add_argument("--robotdog-max-follow-dist", type=float, default=8.0)
    parser.add_argument("--robotdog-max-speed", type=float, default=None, help="Robot dog speed limit, meters/s. Defaults from --human-speed.")
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

    parser.add_argument("--drone-ideal-follow-dist", type=float, default=4.25)
    # 无人机的最大最小跟踪距离
    parser.add_argument("--drone-min-follow-dist", type=float, default=1.0)
    parser.add_argument("--drone-max-follow-dist", type=float, default=6.5)
    parser.add_argument(
        "--human-collision-distance",
        type=float,
        default=0.5,
        help="Habitat-style persistent human proximity collision threshold in meters.",
    )
    parser.add_argument("--drone-height", type=float, default=400.0)
    parser.add_argument("--drone-max-speed", type=float, default=None, help="Drone physical speed limit, meters/s. Defaults from --human-speed.")
    parser.add_argument("--drone-max-vx", type=float, default=None, help="Legacy drone vx clip in m/s. Defaults from --drone-max-speed.")
    parser.add_argument("--drone-max-vy", type=float, default=None, help="Legacy drone vy clip in m/s. Defaults from --drone-max-speed.")
    parser.add_argument(
        "--clip-translational-actions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Legacy compatibility switch retained for old configs; model action "
            "conversion now leaves all drone/dog speed and turn outputs unclipped."
        ),
    )
    parser.add_argument("--drone-max-yaw-rate", type=float, default=0.4)
    parser.add_argument("--drone-camera-fixed-pitch", type=float, default=-60.0)
    parser.add_argument("--drone-camera-pitches", default="-60")
    parser.add_argument("--drone-camera-fixed-yaw", type=float, default=0.0)
    parser.add_argument("--drone-camera-yaw-offsets", default="0")
    parser.add_argument("--drone-camera-mode", choices=["fixed", "oracle"], default="fixed")
    parser.add_argument("--lock-drone-camera-world-xy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--drone-camera-forward-offset", type=float, default=35.0)
    parser.add_argument("--drone-camera-z-offset", type=float, default=-60.0)
    parser.add_argument("--drone-fov", type=float, default=100.0)
    parser.add_argument("--max-camera-search-candidates", type=int, default=12)
    parser.add_argument("--snap-heading", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--follow-behind", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--face-target-before-step", action="store_true")
    parser.add_argument(
        "--oracle-heading-assist",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Debug ablation: keep model translation commands but replace drone yaw "
            "and robotdog turn with target-bearing oracle commands."
        ),
    )
    parser.add_argument(
        "--oracle-drone-action-source",
        choices=[
            "none",
            "env_action",
            "drone_action",
            "base_velocity",
            "commanded_base_velocity",
            "controller_commanded_base_velocity",
            "original_env_action",
            "original_base_velocity",
        ],
        default="none",
        help=(
            "Debug ablation: replace the model-produced drone env action with "
            "the selected per-frame field from the recorded *_drone_info.json. "
            "Use env_action to replay the exact command sent during collection."
        ),
    )
    parser.add_argument(
        "--oracle-drone-action-hold-last",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When the eval episode outlives the recorded action list, keep replaying the last recorded drone action.",
    )
    parser.add_argument(
        "--oracle-drone-velocity-dt",
        type=float,
        default=0.0,
        help=(
            "Override dt for converting 3D recorded velocity fields into BP_drone actions. "
            "0 means use per-frame base_velocity_dt_s/effective_dt_s/dt when present, else --dt."
        ),
    )
    parser.add_argument(
        "--oracle-robotdog-action-source",
        choices=[
            "none",
            "env_action",
            "ground_action",
            "controller_ground_action",
            "commanded_base_velocity",
            "controller_commanded_base_velocity",
            "base_velocity",
        ],
        default="none",
        help="Debug ablation: replace model robotdog action with the selected recorded *_robotdog_info.json field.",
    )
    parser.add_argument(
        "--oracle-robotdog-action-hold-last",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When eval outlives recorded robotdog actions, keep replaying the last recorded robotdog action.",
    )
    parser.add_argument(
        "--oracle-robotdog-velocity-dt",
        type=float,
        default=0.0,
        help=(
            "Override dt for converting 3D recorded robotdog velocity fields into [turn_deg, speed_cm_s]. "
            "0 means use per-frame base_velocity_dt_s/effective_dt_s/dt when present, else --dt."
        ),
    )
    parser.add_argument(
        "--oracle-recorded-step-timing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Before each recorded-action step, apply the per-step recorded ue_interval_ms to target, robotdog and drone BP intervals.",
    )
    parser.add_argument(
        "--drone-heading-assist-gain",
        type=float,
        default=1.2,
        help="Proportional gain from drone heading error radians to yaw-rate command.",
    )
    parser.add_argument(
        "--robotdog-heading-assist-gain",
        type=float,
        default=1.0,
        help="Proportional gain from robotdog heading error degrees to turn command.",
    )
    parser.add_argument(
        "--init-from-recorded-agent-poses",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For recorded target eval, restore each episode's drone/robotdog "
            "first-frame poses and complete recorded cameras, falling back to "
            "ideal poses or default camera search when metadata is missing."
        ),
    )
    parser.add_argument(
        "--init-followers-behind-target",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For recorded target eval, place drone/robotdog behind the target's "
            "first sampled path segment using the configured init distance."
        ),
    )
    parser.add_argument(
        "--init-follower-distance",
        type=float,
        default=4.5,
        help=(
            "Default behind-target spawn distance in meters for both drone and robotdog. "
            "Set to <=0 to fall back to each min/max follow-distance midpoint."
        ),
    )
    parser.add_argument(
        "--init-drone-distance",
        type=float,
        default=None,
        help="Optional drone-only behind-target spawn distance in meters.",
    )
    parser.add_argument(
        "--init-robotdog-distance",
        type=float,
        default=None,
        help="Optional robotdog-only behind-target spawn distance in meters.",
    )

    parser.add_argument("--waypoint-index", type=int, default=9)
    parser.add_argument("--drone-waypoint-index", type=int, default=9)
    parser.add_argument("--robotdog-waypoint-index", type=int, default=9)
    parser.add_argument(
        "--robotdog-waypoint-y-mode",
        choices=("v3_nonholonomic_projection",),
        default="v3_nonholonomic_projection",
        help="Project V3 RobotDog pose waypoints to executable nonholonomic motion.",
    )
    parser.add_argument(
        "--waypoint-control-mode",
        choices=["inverse_fixed_dt", "direct_velocity"],
        default="inverse_fixed_dt",
        help=(
            "inverse_fixed_dt maps model waypoints through the calibrated UE/BP inverse dynamics; "
            "direct_velocity preserves the previous waypoint/time adapter."
        ),
    )
    parser.add_argument(
        "--ground-translation-delay-steps",
        type=int,
        choices=[0, 1],
        default=1,
        help="Observed robotdog BP forward-speed delay used by inverse_fixed_dt.",
    )
    parser.add_argument(
        "--ground-yaw-gain",
        type=float,
        default=0.4,
        help="Observed robotdog BP relation: realized_yaw_delta_deg = gain * turn_command_deg.",
    )
    parser.add_argument("--drone-inverse-a-forward", type=float, default=0.969)
    parser.add_argument("--drone-inverse-b-forward", type=float, default=0.0301)
    parser.add_argument("--drone-inverse-a-lateral", type=float, default=0.969)
    parser.add_argument("--drone-inverse-b-lateral", type=float, default=0.0301)
    parser.add_argument("--drone-inverse-yaw-a", type=float, default=0.464)
    parser.add_argument("--drone-inverse-yaw-b", type=float, default=0.359)
    parser.add_argument(
        "--drone-inverse-xy-smoothing-alpha",
        type=float,
        default=0.2,
        help="Causal low-pass alpha for model waypoint XY velocity before inverse dynamics; 1 disables smoothing.",
    )
    parser.add_argument(
        "--drone-inverse-yaw-smoothing-alpha",
        type=float,
        default=0.25,
        help="Causal low-pass alpha for model waypoint yaw rate before inverse dynamics; 1 disables smoothing.",
    )
    parser.add_argument(
        "--robotdog-inverse-speed-smoothing-alpha",
        type=float,
        default=0.3,
        help="Causal low-pass alpha for robotdog inverse forward/lateral waypoint velocity; 1 disables smoothing.",
    )
    parser.add_argument(
        "--robotdog-inverse-yaw-smoothing-alpha",
        type=float,
        default=0.3,
        help="Causal low-pass alpha for robotdog inverse yaw rate; 1 disables smoothing.",
    )
    parser.add_argument(
        "--waypoint-horizon-steps",
        type=int,
        default=9,
        help="Future action steps integrated before resampling; must match the training data horizon.",
    )
    parser.add_argument(
        "--waypoint-source-dt",
        type=float,
        default=None,
        help=(
            "Time interval represented by one source action in the training waypoint labels. "
            "Defaults to --dt for legacy fixed-rate datasets. Keep --dt as the simulator "
            "control period when processed data was integrated with a different effective_dt_s."
        ),
    )
    parser.add_argument("--drone-vx-scale", type=float, default=1.0)
    parser.add_argument("--drone-vy-scale", type=float, default=1.0)
    parser.add_argument("--drone-speed-gain", type=float, default=1.0)
    parser.add_argument("--drone-velocity-feedback-gain", type=float, default=1.0)
    parser.add_argument("--drone-yaw-feedback-gain", type=float, default=1.0)
    parser.add_argument("--drone-feedback-max-translation", type=float, default=0.6)
    parser.add_argument("--drone-feedback-max-yaw-rate", type=float, default=0.4)
    parser.add_argument("--drone-yaw-sign", type=float, default=1.0)
    parser.add_argument(
        "--drone-yaw-scale",
        type=float,
        default=1.0,
        help="Scale the predicted drone yaw rate before max-yaw-rate clipping.",
    )
    parser.add_argument("--robotdog-yaw-sign", type=float, default=1.0)
    parser.add_argument(
        "--robotdog-yaw-scale",
        type=float,
        default=1.0,
        help="Scale robotdog predicted yaw before converting to degrees and clipping.",
    )
    parser.add_argument("--robotdog-speed-gain", type=float, default=1.0)
    parser.add_argument("--robotdog-velocity-feedback-gain", type=float, default=1.0)
    parser.add_argument("--robotdog-yaw-feedback-gain", type=float, default=1.0)
    parser.add_argument("--robotdog-feedback-max-translation", type=float, default=0.6)
    parser.add_argument("--robotdog-feedback-max-yaw-rate", type=float, default=0.8)
    parser.add_argument(
        "--bbox-motion-control",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable causal V3 bbox feedback. Verified bbox cx affects yaw only; "
            "height-based speed parameters are inactive."
        ),
    )
    parser.add_argument("--bbox-motion-min-confidence", type=float, default=0.25)
    parser.add_argument("--bbox-motion-ema-alpha", type=float, default=0.20)
    parser.add_argument("--bbox-motion-min-valid-frames", type=int, default=2)
    parser.add_argument("--bbox-motion-min-shrink-frames", type=int, default=3)
    parser.add_argument(
        "--bbox-motion-height-tolerance-ratio",
        type=float,
        default=0.20,
        help="Relative dead band around each episode's first trusted bbox height.",
    )
    parser.add_argument(
        "--bbox-motion-height-response-ratio",
        type=float,
        default=0.50,
        help="Relative height deviation at which bbox speed correction saturates.",
    )
    parser.add_argument(
        "--drone-bbox-height-normal",
        type=float,
        default=0.150,
        help="Inactive compatibility parameter; V3 direction-only feedback ignores it.",
    )
    parser.add_argument(
        "--drone-bbox-height-far",
        type=float,
        default=0.120,
        help="Inactive compatibility parameter; V3 direction-only feedback ignores it.",
    )
    parser.add_argument("--drone-bbox-max-speed-gain", type=float, default=1.50, help="Inactive compatibility parameter; ignored by V3.")
    parser.add_argument("--drone-bbox-min-speed-gain", type=float, default=0.50, help="Inactive compatibility parameter; ignored by V3.")
    parser.add_argument("--drone-bbox-max-yaw-residual", type=float, default=0.12)
    parser.add_argument(
        "--robotdog-bbox-height-normal",
        type=float,
        default=0.220,
        help="Inactive compatibility parameter; V3 direction-only feedback ignores it.",
    )
    parser.add_argument(
        "--robotdog-bbox-height-far",
        type=float,
        default=0.160,
        help="Inactive compatibility parameter; V3 direction-only feedback ignores it.",
    )
    parser.add_argument("--robotdog-bbox-max-speed-gain", type=float, default=1.25, help="Inactive compatibility parameter; ignored by V3.")
    parser.add_argument("--robotdog-bbox-min-speed-gain", type=float, default=0.50, help="Inactive compatibility parameter; ignored by V3.")
    parser.add_argument("--robotdog-bbox-max-yaw-residual", type=float, default=0.25)
    parser.add_argument("--drone-success-distance", type=float, default=5.5)
    parser.add_argument("--robotdog-success-distance", type=float, default=8.0)
    parser.add_argument("--drone-lost-distance", type=float, default=9.0)
    parser.add_argument("--robotdog-lost-distance", type=float, default=10.0)
    parser.add_argument(
        "--target-replay-mode",
        choices=["nav_goal", "action", "path_goal"],
        default="action",
        help="How to replay recorded target trajectories during closed-loop multi-agent eval.",
    )
    parser.add_argument("--target-ground-translation-delay-steps", type=int, choices=(0, 1), default=1)
    parser.add_argument("--target-ground-yaw-gain", type=float, default=0.4)
    parser.add_argument("--target-inverse-position-feedback-time-s", type=float, default=0.5)
    parser.add_argument("--target-inverse-max-forward-feedback-mps", type=float, default=2.0)
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
    parser.add_argument("--success-rate-threshold", type=float, default=0.8)
    parser.add_argument("--min-centered-rate", type=float, default=0.8)
    parser.add_argument("--min-success-steps", type=int, default=20)
    args = parser.parse_args()
    args.out_dir = Path(args.save_path)
    return args


def main() -> int:
    args = parse_args()
    normalize_speed_args(args)
    default_agent_speed = agent_max_speed_for_human_speed(float(args.human_speed_mps))
    if args.robotdog_max_speed is None:
        args.robotdog_max_speed = default_agent_speed
    if args.drone_max_speed is None:
        args.drone_max_speed = default_agent_speed
    if args.drone_max_vx is None:
        args.drone_max_vx = float(args.drone_max_speed)
    if args.drone_max_vy is None:
        args.drone_max_vy = float(args.drone_max_speed)
    print(
        f"[speed-profile] human={args.human_speed_mps:.2f}m/s "
        f"robotdog_max={args.robotdog_max_speed:.2f}m/s drone_max={args.drone_max_speed:.2f}m/s",
        flush=True,
    )
    if args.snap_heading or args.face_target_before_step:
        print(
            "[control][warn] legacy oracle heading flag is enabled; drone yaw still remains action-controlled after init",
            flush=True,
        )
    else:
        print(
            "[control] snap_heading=False face_target_before_step=False yaw_control_mode=action",
            flush=True,
        )
    print(
        f"[control] waypoint_control_mode={args.waypoint_control_mode} "
        f"ground_delay={args.ground_translation_delay_steps} "
        f"ground_yaw_gain={args.ground_yaw_gain}",
        flush=True,
    )
    if args.oracle_drone_action_source != "none":
        print(
            f"[control] oracle_drone_action_source={args.oracle_drone_action_source} "
            f"hold_last={bool(args.oracle_drone_action_hold_last)}",
            flush=True,
        )
    if args.oracle_robotdog_action_source != "none":
        print(
            f"[control] oracle_robotdog_action_source={args.oracle_robotdog_action_source} "
            f"hold_last={bool(args.oracle_robotdog_action_hold_last)}",
            flush=True,
        )
    if args.oracle_recorded_step_timing:
        print("[control] oracle_recorded_step_timing=True", flush=True)
    if args.realtime_waypoint_timing and args.deterministic_step:
        raise ValueError("--realtime-waypoint-timing requires --no-deterministic-step")
    if args.deterministic_pause_check_stride < 0:
        raise ValueError("--deterministic-pause-check-stride must be >= 0")
    if args.reuse_post_action_poses and (
        not args.deterministic_step or not args.fast_eval_io
    ):
        raise ValueError(
            "--reuse-post-action-poses requires --deterministic-step --fast-eval-io"
        )
    if args.realtime_waypoint_min_seconds <= 0.0:
        raise ValueError("--realtime-waypoint-min-seconds must be positive")
    if args.realtime_waypoint_max_seconds < args.realtime_waypoint_min_seconds:
        raise ValueError(
            "--realtime-waypoint-max-seconds must be >= --realtime-waypoint-min-seconds"
        )
    if args.waypoint_source_dt is not None and args.waypoint_source_dt <= 0.0:
        raise ValueError("--waypoint-source-dt must be positive")
    if min(args.waypoint_index, args.drone_waypoint_index, args.robotdog_waypoint_index) < 1:
        raise ValueError("waypoint index 0 is the local origin; choose a future waypoint index >= 1")
    if args.waypoint_control_mode == "inverse_fixed_dt":
        if not args.deterministic_step or abs(float(args.dt) - 0.1) > 1e-8:
            raise ValueError("--waypoint-control-mode inverse_fixed_dt requires --deterministic-step --dt 0.1")
        if args.realtime_waypoint_timing:
            raise ValueError("inverse_fixed_dt does not support --realtime-waypoint-timing")
        if args.ground_yaw_gain <= 0.0:
            raise ValueError("--ground-yaw-gain must be positive")
        if min(
            args.drone_inverse_b_forward,
            args.drone_inverse_b_lateral,
            args.drone_inverse_yaw_b,
        ) <= 0.0:
            raise ValueError("drone inverse b coefficients must be positive")
        for name in (
            "drone_inverse_xy_smoothing_alpha",
            "drone_inverse_yaw_smoothing_alpha",
            "robotdog_inverse_speed_smoothing_alpha",
            "robotdog_inverse_yaw_smoothing_alpha",
        ):
            value = float(getattr(args, name))
            if not 0.0 < value <= 1.0:
                raise ValueError(f"--{name.replace('_', '-')} must be in (0, 1]")
    if args.oracle_drone_velocity_dt < 0.0:
        raise ValueError("--oracle-drone-velocity-dt must be >= 0")
    if args.oracle_robotdog_velocity_dt < 0.0:
        raise ValueError("--oracle-robotdog-velocity-dt must be >= 0")
    if args.oracle_drone_action_source != "none" and not args.recorded_target_dir:
        raise ValueError("--oracle-drone-action-source requires --recorded-target-dir")
    if args.oracle_robotdog_action_source != "none" and not args.recorded_target_dir:
        raise ValueError("--oracle-robotdog-action-source requires --recorded-target-dir")
    if args.oracle_recorded_step_timing and not args.recorded_target_dir:
        raise ValueError("--oracle-recorded-step-timing requires --recorded-target-dir")
    if args.human_collision_distance < 0.0:
        raise ValueError("--human-collision-distance must be non-negative")
    for name in (
        "drone_speed_gain",
        "drone_yaw_scale",
        "robotdog_speed_gain",
        "robotdog_yaw_scale",
        "drone_velocity_feedback_gain",
        "drone_yaw_feedback_gain",
        "drone_feedback_max_translation",
        "drone_feedback_max_yaw_rate",
        "robotdog_velocity_feedback_gain",
        "robotdog_yaw_feedback_gain",
        "robotdog_feedback_max_translation",
        "robotdog_feedback_max_yaw_rate",
    ):
        if float(getattr(args, name)) < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    if args.deterministic_step:
        fixed_interval_ms = int(round(float(args.dt) * 1000.0))
        if args.ue_interval_ms is not None and int(args.ue_interval_ms) != fixed_interval_ms:
            raise ValueError(
                "deterministic evaluation requires --ue-interval-ms to equal "
                f"round(--dt * 1000)={fixed_interval_ms}; got {args.ue_interval_ms}"
            )
        args.ue_interval_ms = fixed_interval_ms
        os.environ["UNREALZOO_FIXED_TIMESTEP"] = str(float(args.dt))
    else:
        if args.ue_interval_ms is not None and int(args.ue_interval_ms) <= 0:
            raise ValueError("--ue-interval-ms must be positive")
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
