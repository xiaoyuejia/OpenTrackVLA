#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 UnrealZoo 双 Agent 原始采集结果生成训练 JSONL。

输入是 paired episode:
- <id>_drone.mp4 / <id>_drone_info.json
- <id>_robotdog.mp4 / <id>_robotdog_info.json

输出是面向 MultiAgentOpenTrackVLA 的数据结构:
- agent1_*: 默认无人机
- agent2_*: 默认机器狗
- bbox_feat: 两个 Agent 的归一化 bbox，shape 逻辑为 (2, 4)
- waypoints: 两个 Agent 的未来路点，shape 逻辑为 (2, N, 3)
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image


DEFAULT_INSTRUCTION = "Follow the target person without collision."
SCHEMA_VERSION = "multi_agent_tracking_v1"
ACTION_FIELD_AUTO = "auto"


@dataclass
class PairedEpisode:
    """一个双 Agent episode 的所有原始文件路径。"""

    input_root: Path
    run_dir: Path
    rel_run_dir: Path
    stem: str
    drone_mp4: Path
    robotdog_mp4: Path
    drone_info_json: Path
    robotdog_info_json: Path
    status_json: Optional[Path]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def natural_sort_key(s: str) -> List[Any]:
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


def list_sorted_images(directory: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    imgs = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in exts]
    imgs.sort(key=lambda p: natural_sort_key(p.name))
    return imgs


def find_ffmpeg_executable() -> Optional[str]:
    return shutil.which("ffmpeg")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_steps(path: Path) -> List[Dict[str, Any]]:
    obj = load_json(path)
    if not isinstance(obj, list):
        raise ValueError(f"Expected list in {path}, got {type(obj)}")
    return obj


def infer_rel_run_dir(input_root: Path, run_dir: Path) -> Path:
    try:
        rel = run_dir.resolve().relative_to(input_root.resolve())
        if str(rel) != ".":
            return rel
    except ValueError:
        pass

    if run_dir.parent.name.startswith("seed_"):
        return Path(run_dir.parent.name) / run_dir.name
    return Path(run_dir.name)


def collect_paired_episodes(input_root: Path) -> List[PairedEpisode]:
    """递归查找 drone/robotdog 成对 episode。

    只保留同时存在 drone_info、robotdog_info、drone.mp4、robotdog.mp4 的样本。
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
                rel_run_dir=infer_rel_run_dir(input_root, run_dir),
                stem=stem,
                drone_mp4=drone_mp4,
                robotdog_mp4=robotdog_mp4,
                drone_info_json=drone_info,
                robotdog_info_json=robotdog_info,
                status_json=status_json if status_json.exists() else None,
            )
        )
    return episodes


def episode_status_ok(
    ep: PairedEpisode,
    only_success: bool,
    min_agent_following_rate: float,
    min_total_steps: int,
    exclude_collision: bool,
) -> bool:
    """根据 episode 级状态文件过滤数据。

    支持只保留成功、排除碰撞、最小跟踪率、最小步数等质量筛选条件。
    """
    if ep.status_json is None:
        return not only_success

    try:
        status = load_json(ep.status_json)
    except Exception:
        return not only_success
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

    if not only_success:
        return True

    success = status.get("success", 0)
    finish = bool(status.get("finish", False))
    status_str = str(status.get("status", "")).lower()
    return (isinstance(success, (int, float)) and success > 0) or ("success" in status_str) or finish


def extract_frames_ffmpeg(
    ffmpeg_path: str,
    mp4_path: Path,
    out_dir: Path,
    quality: int,
    reuse_existing: bool = True,
) -> List[Path]:
    """用 ffmpeg 从 mp4 抽帧。

    reuse_existing=True 时，如果目标目录已有图片就直接复用，避免重复抽帧。
    """
    ensure_dir(out_dir)
    existing = list_sorted_images(out_dir)
    if reuse_existing and existing:
        return existing

    pattern = str(out_dir / "frame_%05d.jpg")
    cmd = [
        ffmpeg_path,
        "-y",
        "-i",
        str(mp4_path),
        "-q:v",
        str(quality),
        pattern,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return list_sorted_images(out_dir)


def read_image_size(path: Path, fallback_width: int, fallback_height: int) -> Tuple[int, int]:
    try:
        with Image.open(str(path)) as im:
            return im.size
    except Exception:
        return fallback_width, fallback_height


def action_field_order(preferred_field: Optional[str]) -> List[str]:
    """Choose label fields.

    ``auto`` uses ``base_velocity`` first. This preserves Habitat, where
    ``base_velocity`` is the expert command, and matches current UnrealZoo
    robotdog data where it is the executed body-frame motion.
    """
    preferred = (preferred_field or ACTION_FIELD_AUTO).strip()
    if not preferred or preferred == ACTION_FIELD_AUTO:
        return ["base_velocity", "commanded_base_velocity"]
    order = [preferred]
    for key in ("commanded_base_velocity", "base_velocity"):
        if key not in order:
            order.append(key)
    return order


def to_action3(step: Dict[str, Any], agent_name: str, preferred_field: str) -> List[float]:
    """把 info.json 中不同来源的动作字段统一成 [vx, vy, yaw_rate]。

    默认 auto 优先使用 commanded_base_velocity 作为 oracle 命令监督；
    没有该字段时回退到 base_velocity，从而保留 Habitat/旧数据行为。
    drone_action 是 4 维时，yaw_rate 位于第 4 个元素，因此单独处理。
    """
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


def build_actions(steps: List[Dict[str, Any]], agent_name: str, preferred_field: str) -> List[List[float]]:
    return [to_action3(step, agent_name, preferred_field) for step in steps]


def integrate_actions(actions: List[List[float]], start_index: int, horizon_steps: int, dt: float) -> List[List[float]]:
    """把未来速度积分为局部坐标系下的路点 [x, y, theta]。"""
    x, y, theta = 0.0, 0.0, 0.0
    points: List[List[float]] = []
    end = min(len(actions), start_index + max(0, horizon_steps))
    for k in range(start_index, end):
        vx, vy, wz = actions[k]
        dx = vx * math.cos(theta) - vy * math.sin(theta)
        dy = vx * math.sin(theta) + vy * math.cos(theta)
        x += dx * dt
        y += dy * dt
        theta += wz * dt
        points.append([x, y, theta])
    return points


def resample_waypoints(points: List[List[float]], n_waypoints: int) -> Tuple[List[List[float]], List[bool]]:
    """把积分得到的未来轨迹重采样成固定数量路点。"""
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
    return [points[i] for i in idxs], [True for _ in idxs]


def normalize_bbox_xywh(raw_bbox: Any, width: int, height: int) -> List[float]:
    """把 UnrealZoo 的 bbox 从 pixel xywh 转成 cxcywh_norm。

    如果输入已经像是 0-1 归一化，则直接按 xywh_norm 处理并转中心点形式。
    """
    if not isinstance(raw_bbox, list) or len(raw_bbox) < 4:
        return [0.0, 0.0, 0.0, 0.0]
    try:
        x, y, w, h = [float(v) for v in raw_bbox[:4]]
    except Exception:
        return [0.0, 0.0, 0.0, 0.0]

    if max(abs(x), abs(y), abs(w), abs(h)) <= 1.5:
        cx, cy = x + 0.5 * w, y + 0.5 * h
        return [clamp01(cx), clamp01(cy), clamp01(w), clamp01(h)]

    width = max(1, int(width))
    height = max(1, int(height))
    cx = (x + 0.5 * w) / width
    cy = (y + 0.5 * h) / height
    return [clamp01(cx), clamp01(cy), clamp01(w / width), clamp01(h / height)]


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def bool_field(step: Dict[str, Any], key: str, default: bool = False) -> bool:
    val = step.get(key, default)
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    return default


def float_field(step: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(step.get(key, default))
    except Exception:
        return default


def make_rel_frame_paths(output_root: Path, frame_paths: Iterable[Path]) -> List[str]:
    rels = []
    for path in frame_paths:
        try:
            rel = path.resolve().relative_to(output_root.resolve())
        except ValueError:
            rel = path
        rels.append(rel.as_posix())
    return rels


def load_instruction(ep: PairedEpisode, override: Optional[str]) -> str:
    if override and override.strip():
        return override.strip()
    if ep.status_json is not None:
        try:
            status = load_json(ep.status_json)
            instr = status.get("instruction") if isinstance(status, dict) else None
            if isinstance(instr, str) and instr.strip():
                return instr.strip()
        except Exception:
            pass
    return DEFAULT_INSTRUCTION


def build_agent_payload(
    agent_name: str,
    images_window: List[str],
    current_frame: str,
    step: Dict[str, Any],
    actions_future: List[List[float]],
    waypoints: List[List[float]],
    valid_mask: List[bool],
    bbox_norm: List[float],
) -> Dict[str, Any]:
    """构造单个 Agent 的结构化样本字段。"""
    pose_key = f"{agent_name}_pose"
    return {
        "name": agent_name,
        "images": images_window,
        "current": current_frame,
        "bbox": bbox_norm,
        "bbox_format": "cxcywh_norm",
        "bbox_raw_xywh": step.get("target_bbox", [0, 0, 0, 0]),
        "target_visible": bool_field(step, "target_visible", False),
        "target_visibility": float_field(step, "target_visibility", 0.0),
        "target_center_error": float_field(step, "target_center_error", 0.0),
        "target_centered": bool_field(step, "target_centered", False),
        "target_distance": float_field(step, "dis_to_human", 0.0),
        "collision": bool_field(step, "collision", False),
        "actions": actions_future,
        "waypoints": waypoints,
        "trajectory": waypoints,
        "valid_mask": valid_mask,
        "pose": step.get(pose_key),
        "target_pose": step.get("target_pose"),
    }


def build_samples_for_episode(
    ep: PairedEpisode,
    args: argparse.Namespace,
    output_root: Path,
    drone_frames: List[Path],
    robotdog_frames: List[Path],
    drone_steps: List[Dict[str, Any]],
    robotdog_steps: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """把一个 paired episode 转换为多个滑动窗口训练样本。

    第 j 个样本使用 [j-history, j) 作为历史帧，j 作为当前帧，
    并从 j 开始向未来积分 horizon 步动作得到路点标签。
    """
    agent_order = [args.agent1, args.agent2]
    if sorted(agent_order) != ["drone", "robotdog"]:
        raise ValueError("--agent1/--agent2 must be drone and robotdog in either order")

    frame_map = {"drone": drone_frames, "robotdog": robotdog_frames}
    step_map = {"drone": drone_steps, "robotdog": robotdog_steps}

    actions_map = {
        "drone": build_actions(drone_steps, "drone", args.action_field),
        "robotdog": build_actions(robotdog_steps, "robotdog", args.action_field),
    }

    max_len = min(len(drone_frames), len(robotdog_frames), len(drone_steps), len(robotdog_steps))
    if max_len <= 0:
        return []

    drone_w, drone_h = read_image_size(drone_frames[0], args.fallback_width, args.fallback_height)
    dog_w, dog_h = read_image_size(robotdog_frames[0], args.fallback_width, args.fallback_height)
    size_map = {"drone": (drone_w, drone_h), "robotdog": (dog_w, dog_h)}

    rel_frames = {
        "drone": make_rel_frame_paths(output_root, drone_frames[:max_len]),
        "robotdog": make_rel_frame_paths(output_root, robotdog_frames[:max_len]),
    }

    instruction = load_instruction(ep, args.instruction)
    samples: List[Dict[str, Any]] = []
    history = max(0, int(args.history))
    horizon = max(1, int(args.horizon))
    n_waypoints = max(1, int(args.n_waypoints))
    dt = float(args.dt)

    for j in range(max_len):
        if (not args.allow_partial_horizon) and (j + horizon > max_len):
            continue

        agents: Dict[str, Dict[str, Any]] = {}
        skip = False
        for agent_name in ("drone", "robotdog"):
            steps = step_map[agent_name]
            actions = actions_map[agent_name]
            frame_rels = rel_frames[agent_name]
            step = steps[j]
            if args.skip_collision_steps and bool_field(step, "collision", False):
                skip = True
                break
            if args.require_visible and not bool_field(step, "target_visible", False):
                skip = True
                break
            if float_field(step, "target_visibility", 0.0) < args.min_target_visibility:
                skip = True
                break

            # 历史帧不包含当前帧；当前帧单独放到 current 字段，和原始单 Agent 数据格式保持一致。
            start_idx = max(0, j - history)
            images_window = frame_rels[start_idx:j]
            current_frame = frame_rels[j]
            future_actions = actions[j : min(len(actions), j + horizon)]
            points = integrate_actions(actions, j, horizon, dt)
            waypoints, valid_mask = resample_waypoints(points, n_waypoints)
            if (not args.allow_partial_horizon) and not all(valid_mask):
                skip = True
                break

            width, height = size_map[agent_name]
            bbox = normalize_bbox_xywh(step.get("target_bbox"), width, height)
            agents[agent_name] = build_agent_payload(
                agent_name,
                images_window,
                current_frame,
                step,
                future_actions,
                waypoints,
                valid_mask,
                bbox,
            )

        if skip:
            continue

        a1, a2 = agent_order
        # 同时写结构化 agents 字段和扁平 agent1/agent2 字段。
        # 扁平字段让训练 Dataset 更简单；结构化字段方便调试和后续扩展。
        sample = {
            "schema_version": SCHEMA_VERSION,
            "episode_id": f"{ep.rel_run_dir.as_posix()}/{ep.stem}",
            "episode_stem": ep.stem,
            "rel_run_dir": ep.rel_run_dir.as_posix(),
            "step_index": j,
            "instruction": instruction,
            "dt": dt,
            "history": history,
            "horizon": horizon,
            "n_waypoints": n_waypoints,
            "agent_order": agent_order,
            "agents": agents,
            "agent1_name": a1,
            "agent2_name": a2,
            "agent1_images": agents[a1]["images"],
            "agent1_current": agents[a1]["current"],
            "agent1_bbox": agents[a1]["bbox"],
            "agent1_actions": agents[a1]["actions"],
            "agent1_waypoints": agents[a1]["waypoints"],
            "agent1_valid_mask": agents[a1]["valid_mask"],
            "agent2_images": agents[a2]["images"],
            "agent2_current": agents[a2]["current"],
            "agent2_bbox": agents[a2]["bbox"],
            "agent2_actions": agents[a2]["actions"],
            "agent2_waypoints": agents[a2]["waypoints"],
            "agent2_valid_mask": agents[a2]["valid_mask"],
            "bbox_feat": [agents[a1]["bbox"], agents[a2]["bbox"]],
            "waypoints": [agents[a1]["waypoints"], agents[a2]["waypoints"]],
            "valid_mask": [agents[a1]["valid_mask"], agents[a2]["valid_mask"]],
        }
        samples.append(sample)

    return samples


def write_jsonl(path: Path, samples: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    """命令行参数。"""
    parser = argparse.ArgumentParser(description="Build two-agent TrackVLA JSONL data from UnrealZoo aerial-ground episodes.")
    parser.add_argument("--input_root", type=str, required=True, help="Root containing *_drone_info.json and *_robotdog_info.json pairs.")
    parser.add_argument("--output_root", type=str, required=True, help="Output training data root.")
    parser.add_argument("--history", type=int, default=31)
    parser.add_argument("--horizon", type=int, default=8, help="Number of future velocity steps used to build waypoints.")
    parser.add_argument("--n_waypoints", type=int, default=8)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--agent1", type=str, default="drone", choices=["drone", "robotdog"])
    parser.add_argument("--agent2", type=str, default="robotdog", choices=["drone", "robotdog"])
    parser.add_argument(
        "--action_field",
        type=str,
        default=ACTION_FIELD_AUTO,
        help=(
            "Preferred action field in each *_info.json step. Default auto uses "
            "base_velocity first and falls back to commanded_base_velocity."
        ),
    )
    parser.add_argument("--instruction", type=str, default=None)
    parser.add_argument("--only_success", action="store_true")
    parser.add_argument("--exclude_collision", action="store_true")
    parser.add_argument("--skip_collision_steps", action="store_true")
    parser.add_argument("--require_visible", action="store_true")
    parser.add_argument("--min_target_visibility", type=float, default=0.0)
    parser.add_argument("--min_agent_following_rate", type=float, default=0.0)
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
    parser.add_argument("--dry_run", action="store_true", help="Only scan paired episodes; do not extract frames or write samples.")
    return parser.parse_args()


def main() -> None:
    """脚本入口。"""
    args = parse_args()
    if args.agent1 == args.agent2:
        raise ValueError("--agent1 and --agent2 must be different")

    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    frames_root = output_root / "frames"
    jsonl_root = output_root / "jsonl"
    out_file = Path(args.out_file).resolve() if args.out_file else (output_root / "dataset.json")

    episodes = collect_paired_episodes(input_root)
    if args.max_episodes > 0:
        episodes = episodes[: args.max_episodes]

    print(f"Found paired episodes: {len(episodes)}")
    if args.dry_run:
        for ep in episodes[:20]:
            print(f"  {ep.rel_run_dir.as_posix()}/{ep.stem}")
        return

    ffmpeg_path = find_ffmpeg_executable()
    if ffmpeg_path is None:
        raise RuntimeError("ffmpeg not found in PATH.")

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

    for ep in episodes:
        if not episode_status_ok(
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
            drone_steps = load_steps(ep.drone_info_json)
            robotdog_steps = load_steps(ep.robotdog_info_json)
        except Exception as exc:
            skipped_load += 1
            print(f"[WARN] failed to load info for {ep.rel_run_dir.as_posix()}/{ep.stem}: {exc}")
            continue

        rel_episode_dir = ep.rel_run_dir / ep.stem
        drone_frame_dir = frames_root / rel_episode_dir / "drone"
        robotdog_frame_dir = frames_root / rel_episode_dir / "robotdog"

        try:
            drone_frames = extract_frames_ffmpeg(ffmpeg_path, ep.drone_mp4, drone_frame_dir, args.ffmpeg_quality, args.reuse_existing_frames)
            robotdog_frames = extract_frames_ffmpeg(ffmpeg_path, ep.robotdog_mp4, robotdog_frame_dir, args.ffmpeg_quality, args.reuse_existing_frames)
        except subprocess.CalledProcessError as exc:
            skipped_load += 1
            print(f"[WARN] ffmpeg failed for {ep.rel_run_dir.as_posix()}/{ep.stem}: {exc}")
            continue

        samples = build_samples_for_episode(ep, args, output_root, drone_frames, robotdog_frames, drone_steps, robotdog_steps)
        if not samples:
            skipped_empty += 1
            continue

        jsonl_path = jsonl_root / ep.rel_run_dir / f"{ep.stem}.jsonl"
        write_jsonl(jsonl_path, samples)
        written_jsonl += 1
        written_samples += len(samples)
        if not args.no_aggregate:
            all_samples.extend(samples)
        print(f"[OK] {ep.rel_run_dir.as_posix()}/{ep.stem}: samples={len(samples)}")

    if (not args.no_aggregate) and all_samples:
        with out_file.open("w", encoding="utf-8") as f:
            json.dump(all_samples, f, ensure_ascii=False)

    print(f"Kept episodes: {kept}")
    print(f"Written JSONL files: {written_jsonl}")
    print(f"Written samples: {written_samples}")
    print(f"Skipped by status: {skipped_status}")
    print(f"Skipped by load/extract: {skipped_load}")
    print(f"Skipped empty: {skipped_empty}")
    print(f"Output root: {output_root}")
    if (not args.no_aggregate) and all_samples:
        print(f"Aggregated dataset JSON: {out_file}")


if __name__ == "__main__":
    main()
