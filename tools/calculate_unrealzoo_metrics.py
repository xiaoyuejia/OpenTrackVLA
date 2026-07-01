#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize UnrealZoo multi-agent evaluation results."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARN] failed to parse {path}: {exc}")
        return None
    return obj if isinstance(obj, dict) else None


def iter_episode_jsons(eval_dir: Path) -> list[Path]:
    files = sorted(eval_dir.rglob("*.json"))
    out = []
    for path in files:
        name = path.name
        if (
            name.endswith("_info.json")
            or name.endswith("_combined_info.json")
            or name.endswith("_drone_info.json")
            or name.endswith("_robotdog_info.json")
            or name.endswith("_setup.json")
        ):
            continue
        out.append(path)
    return out


def calculate_metrics(eval_dir: Path) -> dict[str, Any] | None:
    json_files = iter_episode_jsons(eval_dir)
    if not json_files:
        return None

    total = 0
    success = 0.0
    collision = 0.0
    joint_rates = []
    drone_rates = []
    dog_rates = []
    total_steps = []
    fps_values = []
    drone_bbox_ious = []
    dog_bbox_ious = []
    visible_accuracies = []
    bbox_sources = defaultdict(int)
    status_counts = defaultdict(int)

    for path in json_files:
        data = _load_json(path)
        if data is None:
            continue
        if not any(key in data for key in ("success", "total_step", "status", "joint_following_rate")):
            continue
        total += 1
        success += float(data.get("success", 0.0))
        collision += float(data.get("collision", 0.0))
        drone_rate = float(data.get("drone_following_rate", 0.0))
        dog_rate = float(data.get("robotdog_following_rate", 0.0))
        joint_rate = float(data.get("joint_following_rate", min(drone_rate, dog_rate)))
        joint_rates.append(joint_rate)
        drone_rates.append(drone_rate)
        dog_rates.append(dog_rate)
        total_steps.append(float(data.get("total_step", 0.0)))
        if "fps" in data:
            fps_values.append(float(data.get("fps", 0.0)))
        if "drone_bbox_iou_mean" in data:
            drone_bbox_ious.append(float(data["drone_bbox_iou_mean"]))
        if "robotdog_bbox_iou_mean" in data:
            dog_bbox_ious.append(float(data["robotdog_bbox_iou_mean"]))
        if "visible_accuracy" in data:
            visible_accuracies.append(float(data["visible_accuracy"]))
        bbox_sources[str(data.get("bbox_source", "unknown"))] += 1
        status_counts[str(data.get("status", "Unknown"))] += 1

    if total == 0:
        return None

    def mean(values: list[float]) -> float:
        return sum(values) / max(len(values), 1)

    return {
        "total_episodes": total,
        "SR": success / total * 100.0,
        "CR": collision / total * 100.0,
        "JointTR": mean(joint_rates) * 100.0,
        "DroneTR": mean(drone_rates) * 100.0,
        "RobotDogTR": mean(dog_rates) * 100.0,
        "avg_steps": mean(total_steps),
        "avg_fps": mean(fps_values) if fps_values else 0.0,
        "DroneBBoxIoU": mean(drone_bbox_ious) * 100.0 if drone_bbox_ious else 0.0,
        "RobotDogBBoxIoU": mean(dog_bbox_ious) * 100.0 if dog_bbox_ious else 0.0,
        "VisibleAcc": mean(visible_accuracies) * 100.0 if visible_accuracies else 0.0,
        "bbox_sources": dict(bbox_sources),
        "success_count": int(success),
        "collision_count": int(collision),
        "status_counts": dict(status_counts),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Calculate UnrealZoo multi-agent evaluation metrics.")
    parser.add_argument("--eval-dir", type=Path, default=Path("/data/hdt/ntv_data/sim_data/eval/unrealzoo_multi_agent"))
    args = parser.parse_args()

    metrics = calculate_metrics(args.eval_dir)
    if metrics is None:
        print(f"Error: no episode result json found under {args.eval_dir}")
        return 1

    print("=" * 72)
    print(f"UnrealZoo multi-agent eval: {args.eval_dir}")
    print("=" * 72)
    print(f"Episodes:          {metrics['total_episodes']}")
    print(f"SR ↑:              {metrics['SR']:.2f}%")
    print(f"Joint TR ↑:        {metrics['JointTR']:.2f}%")
    print(f"Drone TR ↑:        {metrics['DroneTR']:.2f}%")
    print(f"RobotDog TR ↑:     {metrics['RobotDogTR']:.2f}%")
    print(f"CR ↓:              {metrics['CR']:.2f}%")
    print(f"Drone bbox IoU ↑:  {metrics['DroneBBoxIoU']:.2f}%")
    print(f"Dog bbox IoU ↑:    {metrics['RobotDogBBoxIoU']:.2f}%")
    print(f"Visibility Acc ↑:  {metrics['VisibleAcc']:.2f}%")
    print(f"BBox sources:      {metrics['bbox_sources']}")
    print(f"Avg steps:         {metrics['avg_steps']:.1f}")
    print(f"Avg FPS:           {metrics['avg_fps']:.2f}")
    print(f"Success count:     {metrics['success_count']}")
    print(f"Collision count:   {metrics['collision_count']}")
    print(f"Status counts:     {metrics['status_counts']}")
    print()
    print(f"表格格式: {metrics['SR']:.1f} / {metrics['JointTR']:.1f} / {metrics['DroneTR']:.1f} / {metrics['RobotDogTR']:.1f} / {metrics['CR']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
