#!/usr/bin/env python3
"""Recalculate tracking metrics from saved per-step distance traces.

This intentionally does not mutate the original episode JSON files.  It keeps
the original metric values as a reference and writes an alternate aggregate
where both followers use a 1 m minimum follow distance and any after-action
human distance strictly below 0.5 m is counted as a collision.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def stat_files(root: Path):
    for path in sorted(root.rglob("*.json")):
        if path.name.endswith(("_info.json", "_combined_info.json", "_drone_info.json", "_robotdog_info.json", "_setup.json")):
            continue
        try:
            data = load(path)
        except Exception:
            continue
        if isinstance(data, dict) and "total_step" in data and "status" in data:
            yield path, data


def after_action_distance(item: dict) -> float:
    value = item.get("dis_to_human_after_action", item.get("dis_to_human"))
    try:
        value = float(value)
    except (TypeError, ValueError):
        return math.inf
    return value if math.isfinite(value) else math.inf


def summarize(rows: list[dict], prefix: str) -> dict:
    n = len(rows)
    avg = lambda key: mean(float(row[key]) for row in rows) if rows else 0.0
    return {
        "episodes": n,
        "success_count": sum(int(row[f"{prefix}success"]) for row in rows),
        "success_rate_pct": 100.0 * mean(row[f"{prefix}success"] for row in rows) if rows else 0.0,
        "collision_count": sum(int(row[f"{prefix}collision"]) for row in rows),
        "collision_rate_pct": 100.0 * mean(row[f"{prefix}collision"] for row in rows) if rows else 0.0,
        "human_collision_count": sum(int(row[f"{prefix}human_collision"]) for row in rows),
        "human_collision_rate_pct": 100.0 * mean(row[f"{prefix}human_collision"] for row in rows) if rows else 0.0,
        "joint_tr_pct": 100.0 * avg(f"{prefix}joint_tr"),
        "drone_tr_pct": 100.0 * avg(f"{prefix}drone_tr"),
        "robotdog_tr_pct": 100.0 * avg(f"{prefix}robotdog_tr"),
        "drone_centered_pct": 100.0 * avg("original_drone_centered"),
        "robotdog_centered_pct": 100.0 * avg("original_robotdog_centered"),
        "visible_accuracy_pct": 100.0 * avg("original_visible_accuracy"),
        "drone_bbox_iou_pct": 100.0 * avg("original_drone_bbox_iou"),
        "robotdog_bbox_iou_pct": 100.0 * avg("original_robotdog_bbox_iou"),
        "avg_steps": avg("total_step"),
        "avg_fps": avg("original_fps"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--min-follow-distance", type=float, default=1.0)
    parser.add_argument("--human-collision-distance", type=float, default=0.5)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for stat_path, stat in stat_files(args.eval_dir):
        stem = stat_path.name.removesuffix(".json")
        drone_path = stat_path.with_name(stem + "_drone_info.json")
        dog_path = stat_path.with_name(stem + "_robotdog_info.json")
        if not drone_path.is_file() or not dog_path.is_file():
            continue
        try:
            drone = load(drone_path)
            dog = load(dog_path)
        except Exception:
            continue
        steps = min(len(drone), len(dog))
        if not steps:
            continue

        drone_dist = [after_action_distance(x) for x in drone[:steps]]
        dog_dist = [after_action_distance(x) for x in dog[:steps]]
        drone_visible = [bool(x.get("target_visible", True)) for x in drone[:steps]]
        dog_visible = [bool(x.get("target_visible", True)) for x in dog[:steps]]
        drone_follow = [v and args.min_follow_distance <= d <= 6.5 for v, d in zip(drone_visible, drone_dist)]
        dog_follow = [v and args.min_follow_distance <= d <= 8.0 for v, d in zip(dog_visible, dog_dist)]
        joint_follow = [a and b for a, b in zip(drone_follow, dog_follow)]
        proximity_collision = any(
            d < args.human_collision_distance for d in (*drone_dist, *dog_dist)
        )
        physical_collision = bool(stat.get("collision", 0.0))
        alt_collision = physical_collision or proximity_collision
        status = str(stat.get("status", "Unknown"))
        total_step = int(stat.get("total_step", steps))
        full_horizon = bool(stat.get("completed_full_horizon", False))
        final_drone_in_range = args.min_follow_distance <= drone_dist[-1] <= 6.5
        final_dog_in_range = args.min_follow_distance <= dog_dist[-1] <= 8.0
        terminal = bool(joint_follow[-1] and (full_horizon or (final_drone_in_range and final_dog_in_range)))
        alt_success = bool(
            not alt_collision
            and status not in {"Lost", "Collision", "PersistentFailure", "Timeout"}
            and (full_horizon or total_step >= 20)
            and terminal
        )
        rows.append({
            "result_path": str(stat_path.relative_to(args.eval_dir)),
            "recorded_target_episode": str(stat.get("recorded_target_episode", "")),
            "status_original": status,
            "total_step": total_step,
            "original_success": float(stat.get("success", 0.0)),
            "original_collision": float(stat.get("collision", 0.0)),
            "original_human_collision": float(stat.get("human_collision", 0.0)),
            "original_joint_tr": float(stat.get("joint_following_rate", 0.0)),
            "original_drone_tr": float(stat.get("drone_following_rate", 0.0)),
            "original_robotdog_tr": float(stat.get("robotdog_following_rate", 0.0)),
            "original_drone_centered": float(stat.get("drone_centered_rate", 0.0)),
            "original_robotdog_centered": float(stat.get("robotdog_centered_rate", 0.0)),
            "original_visible_accuracy": float(stat.get("visible_accuracy", 0.0)),
            "original_drone_bbox_iou": float(stat.get("drone_bbox_iou_mean", 0.0)),
            "original_robotdog_bbox_iou": float(stat.get("robotdog_bbox_iou_mean", 0.0)),
            "original_fps": float(stat.get("fps", 0.0)),
            "alt_success": float(alt_success),
            "alt_collision": float(alt_collision),
            "alt_human_collision": float(proximity_collision),
            "alt_joint_tr": sum(joint_follow) / len(joint_follow),
            "alt_drone_tr": sum(drone_follow) / len(drone_follow),
            "alt_robotdog_tr": sum(dog_follow) / len(dog_follow),
            "alt_final_drone_distance_m": drone_dist[-1],
            "alt_final_robotdog_distance_m": dog_dist[-1],
            "alt_terminal_following_success": float(terminal),
        })

    original = summarize(rows, "original_")
    alternate = summarize(rows, "alt_")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]) if rows else ["result_path"])
        writer.writeheader()
        writer.writerows(rows)

    def pct(key):
        return f"{alternate[key]:.2f}%"

    def delta(key):
        return f"{alternate[key] - original[key]:+.2f}"

    md = f"# Alternative distance-threshold metrics\n\n"
    md += f"Evaluation directory: `{args.eval_dir}`\n\n"
    md += "This is a recalculation from saved after-action distance traces; original episode JSON files are unchanged.\n\n"
    md += f"- Episodes included: **{len(rows)}**\n"
    md += f"- Alternative minimum follow distance: **{args.min_follow_distance:.1f} m** for both drone and RobotDog\n"
    md += f"- Alternative human collision rule: after-action XY distance **< {args.human_collision_distance:.1f} m**; physical UE collision is also retained in the combined collision column\n"
    md += "- Maximum follow distances retained: drone 6.5 m, RobotDog 8.0 m\n\n"
    md += "## Comparison\n\n"
    md += "| Metric | Original | Alternative | Delta |\n|---|---:|---:|---:|\n"
    for label, key in [
        ("Success rate", "success_rate_pct"), ("Collision rate", "collision_rate_pct"),
        ("Human/proximity collision rate", "human_collision_rate_pct"),
        ("Joint tracking rate", "joint_tr_pct"), ("Drone tracking rate", "drone_tr_pct"),
        ("RobotDog tracking rate", "robotdog_tr_pct"), ("Drone centered rate", "drone_centered_pct"),
        ("RobotDog centered rate", "robotdog_centered_pct"), ("Visibility accuracy", "visible_accuracy_pct"),
    ]:
        md += f"| {label} | {original[key]:.2f}% | {alternate[key]:.2f}% | {delta(key)} pp |\n"
    md += f"\nSuccess count: original {original['success_count']}/{len(rows)}, alternative {alternate['success_count']}/{len(rows)}.\n"
    md += f"Collision count: original {original['collision_count']}/{len(rows)}, alternative {alternate['collision_count']}/{len(rows)}.\n"
    md += f"Proximity-collision count under <{args.human_collision_distance:.1f} m: {alternate['human_collision_count']}/{len(rows)}.\n"
    md += "\nThe alternative score is diagnostic only; it does not overwrite the original episode results or `metrics_current.csv`.\n"
    args.output_md.write_text(md, encoding="utf-8")
    print(json.dumps({"episodes": len(rows), "original": original, "alternate": alternate}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
