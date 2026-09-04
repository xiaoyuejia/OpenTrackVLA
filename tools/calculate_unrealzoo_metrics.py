#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize UnrealZoo multi-agent evaluation results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


EPISODE_CSV_FIELDS = (
    "row_type",
    "result_path",
    "scene",
    "output_episode_id",
    "recorded_target_episode",
    "status",
    "episode_count",
    "success",
    "success_count",
    "success_rate_pct",
    "collision",
    "collision_count",
    "collision_rate_pct",
    "human_collision",
    "human_collision_count",
    "human_collision_rate_pct",
    "total_step",
    "fps",
    "model_latency_ms",
    "model_fps",
    "joint_tr_pct",
    "drone_tr_pct",
    "robotdog_tr_pct",
    "drone_centered_pct",
    "robotdog_centered_pct",
    "visible_accuracy_pct",
    "drone_bbox_iou_pct",
    "robotdog_bbox_iou_pct",
    "completed_full_horizon",
    "final_drone_distance_m",
    "final_robotdog_distance_m",
    "final_drone_following",
    "final_robotdog_following",
    "final_joint_following",
    "final_drone_in_range",
    "final_robotdog_in_range",
    "final_joint_in_range",
    "terminal_following_success",
    "final_lost_count",
    "final_failure_count",
    "target_motion_mode",
    "target_stopped",
    "target_stop_step",
    "checkpoint",
    "bbox_source",
    "roi_bbox_source",
    "evaluation_protocol",
    "deterministic_step",
    "fixed_timestep_seconds",
    "init_agent_pose_policy",
    "init_from_recorded_agent_poses",
    "init_followers_behind_target",
    "restored_drone_pose",
    "restored_robotdog_pose",
    "restored_drone_camera",
    "restored_robotdog_camera",
    "drone_camera_pitch_deg",
    "drone_camera_yaw_offset_deg",
    "robotdog_camera_mount",
    "robotdog_camera_pitch_deg",
    "robotdog_camera_yaw_offset_deg",
    "initial_drone_distance_m",
    "initial_robotdog_distance_m",
    "waypoint_index",
    "drone_waypoint_index",
    "robotdog_waypoint_index",
    "waypoint_horizon_steps",
    "waypoint_source_dt_s",
    "drone_speed_gain",
    "robotdog_speed_gain",
    "drone_yaw_gain",
    "robotdog_yaw_gain",
    "drone_velocity_feedback_gain",
    "drone_yaw_feedback_gain",
    "robotdog_velocity_feedback_gain",
    "robotdog_yaw_feedback_gain",
    "status_counts",
)


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


def _pct(data: dict[str, Any], key: str) -> float:
    return float(data.get(key, 0.0)) * 100.0


def collect_episode_rows(eval_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in iter_episode_jsons(eval_dir):
        data = _load_json(path)
        if data is None or not any(
            key in data for key in ("success", "total_step", "status", "joint_following_rate")
        ):
            continue
        setup = _load_json(path.with_name(f"{path.stem}_setup.json")) or {}
        initial_distance = setup.get("initial_distance_xy_m") or {}
        action_selection = setup.get("action_selection") or {}
        action_gain = setup.get("action_gain") or {}
        feedback = setup.get("velocity_feedback") or {}
        restored_poses = setup.get("restored_recorded_agent_poses") or {}
        restored_cameras = setup.get("restored_recorded_agent_cameras") or {}
        drone_camera = setup.get("drone_camera") or {}
        dog_camera = setup.get("dog_camera") or {}
        rows.append(
            {
                "row_type": "episode",
                "result_path": str(path.relative_to(eval_dir)),
                "scene": str(data.get("env_id") or path.parent.name),
                "output_episode_id": path.stem,
                "recorded_target_episode": str(data.get("recorded_target_episode") or ""),
                "status": str(data.get("status", "Unknown")),
                "episode_count": 1,
                "success": float(data.get("success", 0.0)),
                "success_count": int(float(data.get("success", 0.0)) > 0.0),
                "success_rate_pct": _pct(data, "success"),
                "collision": float(data.get("collision", 0.0)),
                "collision_count": int(float(data.get("collision", 0.0)) > 0.0),
                "collision_rate_pct": _pct(data, "collision"),
                "human_collision": float(data.get("human_collision", 0.0)),
                "human_collision_count": int(float(data.get("human_collision", 0.0)) > 0.0),
                "human_collision_rate_pct": _pct(data, "human_collision"),
                "total_step": int(data.get("total_step", 0)),
                "fps": float(data.get("fps", 0.0)),
                "model_latency_ms": float(data.get("model_latency_ms", 0.0)),
                "model_fps": float(data.get("model_fps", 0.0)),
                "joint_tr_pct": _pct(data, "joint_following_rate"),
                "drone_tr_pct": _pct(data, "drone_following_rate"),
                "robotdog_tr_pct": _pct(data, "robotdog_following_rate"),
                "drone_centered_pct": _pct(data, "drone_centered_rate"),
                "robotdog_centered_pct": _pct(data, "robotdog_centered_rate"),
                "visible_accuracy_pct": _pct(data, "visible_accuracy"),
                "drone_bbox_iou_pct": _pct(data, "drone_bbox_iou_mean"),
                "robotdog_bbox_iou_pct": _pct(data, "robotdog_bbox_iou_mean"),
                "completed_full_horizon": bool(data.get("completed_full_horizon", False)),
                "final_drone_distance_m": float(data.get("final_drone_distance", 0.0)),
                "final_robotdog_distance_m": float(data.get("final_robotdog_distance", 0.0)),
                "final_drone_following": bool(data.get("final_drone_following", False)),
                "final_robotdog_following": bool(data.get("final_robotdog_following", False)),
                "final_joint_following": bool(data.get("final_joint_following", False)),
                "final_drone_in_range": bool(data.get("final_drone_in_range", False)),
                "final_robotdog_in_range": bool(data.get("final_robotdog_in_range", False)),
                "final_joint_in_range": bool(data.get("final_joint_in_range", False)),
                "terminal_following_success": bool(data.get("terminal_following_success", False)),
                "final_lost_count": int(data.get("final_lost_count", 0)),
                "final_failure_count": int(data.get("final_failure_count", 0)),
                "target_motion_mode": str(data.get("target_motion_mode", "")),
                "target_stopped": bool(data.get("target_stopped", False)),
                "target_stop_step": data.get("target_stop_step"),
                "checkpoint": str(data.get("ckpt", "")),
                "bbox_source": str(data.get("bbox_source", "")),
                "roi_bbox_source": str(data.get("roi_bbox_source", "")),
                "evaluation_protocol": str(data.get("evaluation_protocol", "")),
                "deterministic_step": bool(data.get("deterministic_step", False)),
                "fixed_timestep_seconds": float(data.get("fixed_timestep_seconds", 0.0)),
                "init_agent_pose_policy": str(setup.get("init_agent_pose_policy", "")),
                "init_from_recorded_agent_poses": bool(setup.get("init_from_recorded_agent_poses", False)),
                "init_followers_behind_target": bool(setup.get("init_followers_behind_target", False)),
                "restored_drone_pose": bool(restored_poses.get("drone", False)),
                "restored_robotdog_pose": bool(restored_poses.get("robotdog", False)),
                "restored_drone_camera": bool(restored_cameras.get("drone", False)),
                "restored_robotdog_camera": bool(restored_cameras.get("robotdog", False)),
                "drone_camera_pitch_deg": drone_camera.get("pitch"),
                "drone_camera_yaw_offset_deg": drone_camera.get("yaw_offset"),
                "robotdog_camera_mount": json.dumps(dog_camera.get("mount"), separators=(",", ":")),
                "robotdog_camera_pitch_deg": dog_camera.get("pitch"),
                "robotdog_camera_yaw_offset_deg": dog_camera.get("yaw_offset"),
                "initial_drone_distance_m": initial_distance.get("drone"),
                "initial_robotdog_distance_m": initial_distance.get("robotdog"),
                "waypoint_index": action_selection.get("waypoint_index"),
                "drone_waypoint_index": action_selection.get("drone_waypoint_index"),
                "robotdog_waypoint_index": action_selection.get("robotdog_waypoint_index"),
                "waypoint_horizon_steps": action_selection.get("waypoint_horizon_steps"),
                "waypoint_source_dt_s": action_selection.get("waypoint_source_dt_s"),
                "drone_speed_gain": action_gain.get("drone_speed"),
                "robotdog_speed_gain": action_gain.get("robotdog_speed"),
                "drone_yaw_gain": action_gain.get("drone_yaw"),
                "robotdog_yaw_gain": action_gain.get("robotdog_yaw"),
                "drone_velocity_feedback_gain": feedback.get("drone_translation_gain"),
                "drone_yaw_feedback_gain": feedback.get("drone_yaw_gain"),
                "robotdog_velocity_feedback_gain": feedback.get("robotdog_translation_gain"),
                "robotdog_yaw_feedback_gain": feedback.get("robotdog_yaw_gain"),
                "status_counts": "",
            }
        )
    rows.sort(key=lambda row: (row["scene"], row["recorded_target_episode"], row["output_episode_id"]))
    return rows


def write_episode_metrics_csv(
    eval_dir: Path,
    output_csv: Path,
    metrics: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    aggregate = {field: "" for field in EPISODE_CSV_FIELDS}
    aggregate.update(
        {
            "row_type": "aggregate",
            "result_path": str(eval_dir),
            "scene": "__all__",
            "output_episode_id": "__aggregate__",
            "status": "Aggregate",
            "episode_count": metrics["total_episodes"],
            "success_count": metrics["success_count"],
            "success_rate_pct": metrics["SR"],
            "collision_count": metrics["collision_count"],
            "collision_rate_pct": metrics["CR"],
            "human_collision": metrics["HumanCollisionRate"] / 100.0,
            "human_collision_count": metrics["human_collision_count"],
            "human_collision_rate_pct": metrics["HumanCollisionRate"],
            "total_step": metrics["avg_steps"],
            "fps": metrics["avg_fps"],
            "model_latency_ms": metrics["avg_model_latency_ms"],
            "model_fps": metrics["avg_model_fps"],
            "joint_tr_pct": metrics["JointTR"],
            "drone_tr_pct": metrics["DroneTR"],
            "robotdog_tr_pct": metrics["RobotDogTR"],
            "drone_centered_pct": metrics["DroneCentered"],
            "robotdog_centered_pct": metrics["RobotDogCentered"],
            "visible_accuracy_pct": metrics["VisibleAcc"],
            "drone_bbox_iou_pct": metrics["DroneBBoxIoU"],
            "robotdog_bbox_iou_pct": metrics["RobotDogBBoxIoU"],
            "status_counts": json.dumps(metrics["status_counts"], ensure_ascii=False, sort_keys=True),
        }
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EPISODE_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(aggregate)


def calculate_metrics(eval_dir: Path) -> dict[str, Any] | None:
    json_files = iter_episode_jsons(eval_dir)
    if not json_files:
        return None

    total = 0
    success = 0.0
    collision = 0.0
    human_collision = 0.0
    joint_rates = []
    drone_rates = []
    dog_rates = []
    drone_centered_rates = []
    dog_centered_rates = []
    total_steps = []
    fps_values = []
    model_latency_values = []
    model_fps_values = []
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
        human_collision += float(data.get("human_collision", 0.0))
        drone_rate = float(data.get("drone_following_rate", 0.0))
        dog_rate = float(data.get("robotdog_following_rate", 0.0))
        joint_rate = float(data.get("joint_following_rate", min(drone_rate, dog_rate)))
        joint_rates.append(joint_rate)
        drone_rates.append(drone_rate)
        dog_rates.append(dog_rate)
        if "drone_centered_rate" in data:
            drone_centered_rates.append(float(data["drone_centered_rate"]))
        if "robotdog_centered_rate" in data:
            dog_centered_rates.append(float(data["robotdog_centered_rate"]))
        total_steps.append(float(data.get("total_step", 0.0)))
        if "fps" in data:
            fps_values.append(float(data.get("fps", 0.0)))
        if "model_latency_ms" in data:
            model_latency_values.append(float(data.get("model_latency_ms", 0.0)))
        if "model_fps" in data:
            model_fps_values.append(float(data.get("model_fps", 0.0)))
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
        "HumanCollisionRate": human_collision / total * 100.0,
        "JointTR": mean(joint_rates) * 100.0,
        "DroneTR": mean(drone_rates) * 100.0,
        "RobotDogTR": mean(dog_rates) * 100.0,
        "DroneCentered": mean(drone_centered_rates) * 100.0 if drone_centered_rates else 0.0,
        "RobotDogCentered": mean(dog_centered_rates) * 100.0 if dog_centered_rates else 0.0,
        "avg_steps": mean(total_steps),
        "avg_fps": mean(fps_values) if fps_values else 0.0,
        "avg_model_latency_ms": mean(model_latency_values) if model_latency_values else 0.0,
        "avg_model_fps": mean(model_fps_values) if model_fps_values else 0.0,
        "DroneBBoxIoU": mean(drone_bbox_ious) * 100.0 if drone_bbox_ious else 0.0,
        "RobotDogBBoxIoU": mean(dog_bbox_ious) * 100.0 if dog_bbox_ious else 0.0,
        "VisibleAcc": mean(visible_accuracies) * 100.0 if visible_accuracies else 0.0,
        "bbox_sources": dict(bbox_sources),
        "success_count": int(success),
        "collision_count": int(collision),
        "human_collision_count": int(human_collision),
        "status_counts": dict(status_counts),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Calculate UnrealZoo multi-agent evaluation metrics.")
    parser.add_argument("--eval-dir", type=Path, default=Path("/data/hdt/ntv_data/sim_data/eval/unrealzoo_multi_agent"))
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--expected-episodes", type=int, default=None)
    parser.add_argument("--require-exact-episodes", action="store_true")
    args = parser.parse_args()

    metrics = calculate_metrics(args.eval_dir)
    if metrics is None:
        print(f"Error: no episode result json found under {args.eval_dir}")
        return 1

    rows = collect_episode_rows(args.eval_dir)
    if args.expected_episodes is not None and len(rows) != args.expected_episodes:
        message = (
            f"episode count mismatch under {args.eval_dir}: "
            f"expected={args.expected_episodes} actual={len(rows)}"
        )
        if args.require_exact_episodes:
            print(f"Error: {message}")
            return 2
        print(f"[WARN] {message}")
    if args.output_csv is not None:
        write_episode_metrics_csv(args.eval_dir, args.output_csv, metrics, rows)
        print(f"[CSV] wrote {len(rows)} episode rows + aggregate: {args.output_csv}")

    print("=" * 72)
    print(f"UnrealZoo multi-agent eval: {args.eval_dir}")
    print("=" * 72)
    print(f"Episodes:          {metrics['total_episodes']}")
    print(f"SR ↑:              {metrics['SR']:.2f}%")
    print(f"Joint TR ↑:        {metrics['JointTR']:.2f}%")
    print(f"Drone TR ↑:        {metrics['DroneTR']:.2f}%")
    print(f"RobotDog TR ↑:     {metrics['RobotDogTR']:.2f}%")
    print(f"Drone centered ↑:  {metrics['DroneCentered']:.2f}%")
    print(f"Dog centered ↑:    {metrics['RobotDogCentered']:.2f}%")
    print(f"CR ↓:              {metrics['CR']:.2f}%")
    print(f"Human collision ↓: {metrics['HumanCollisionRate']:.2f}%")
    print(f"Drone bbox IoU ↑:  {metrics['DroneBBoxIoU']:.2f}%")
    print(f"Dog bbox IoU ↑:    {metrics['RobotDogBBoxIoU']:.2f}%")
    print(f"Visibility Acc ↑:  {metrics['VisibleAcc']:.2f}%")
    print(f"BBox sources:      {metrics['bbox_sources']}")
    print(f"Avg steps:         {metrics['avg_steps']:.1f}")
    print(f"Avg FPS:           {metrics['avg_fps']:.2f}")
    if metrics["avg_model_latency_ms"] > 0.0:
        print(f"Model latency:     {metrics['avg_model_latency_ms']:.2f} ms")
        print(f"Model FPS:         {metrics['avg_model_fps']:.2f}")
    print(f"Success count:     {metrics['success_count']}")
    print(f"Collision count:   {metrics['collision_count']}")
    print(f"Status counts:     {metrics['status_counts']}")
    print()
    print(f"表格格式: {metrics['SR']:.1f} / {metrics['JointTR']:.1f} / {metrics['DroneTR']:.1f} / {metrics['RobotDogTR']:.1f} / {metrics['CR']:.2f} / {metrics['HumanCollisionRate']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
