#!/usr/bin/env python3
"""Audit a mirrored YOLO bbox-fill output and list frames still missing boxes."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


def bbox_valid(value: Any) -> bool:
    try:
        values = [float(item) for item in value[:4]]
        return len(values) == 4 and all(math.isfinite(item) for item in values) and values[2] > 0 and values[3] > 0
    except (TypeError, ValueError, OverflowError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    root = args.output_root.resolve()
    totals: Counter[str] = Counter()
    groups: dict[str, Counter[str]] = defaultdict(Counter)
    missing_rows = []
    drone_infos = sorted(root.rglob("*_drone_info.json"), key=lambda path: str(path))
    for drone_info in drone_infos:
        episode = drone_info.name[: -len("_drone_info.json")]
        relative_dir = drone_info.parent.relative_to(root)
        group = relative_dir.parts[0]
        totals["episodes"] += 1
        groups[group]["episodes"] += 1
        for agent in ("drone", "robotdog"):
            info_path = drone_info.parent / f"{episode}_{agent}_info.json"
            video_path = drone_info.parent / f"{episode}_{agent}.mp4"
            if not info_path.is_file():
                totals["missing_info_files"] += 1
                continue
            if not video_path.exists():
                totals["missing_video_links"] += 1
            rows = json.loads(info_path.read_text(encoding="utf-8"))
            for index, row in enumerate(rows):
                source = str(row.get("bbox_source", "unknown"))
                valid = bbox_valid(row.get("target_bbox")) and bool(row.get("bbox_valid_mask", False))
                totals["agent_frames"] += 1
                totals[f"source_{source}"] += 1
                totals["valid"] += int(valid)
                totals["invalid"] += int(not valid)
                groups[group]["agent_frames"] += 1
                groups[group][f"source_{source}"] += 1
                groups[group]["valid"] += int(valid)
                groups[group]["invalid"] += int(not valid)
                if not valid:
                    missing_rows.append({
                        "relative_dir": str(relative_dir),
                        "episode": episode,
                        "agent": agent,
                        "frame_index": index,
                        "target_visible": bool(row.get("target_visible", False)),
                        "bbox_source": source,
                    })

    missing_csv = root / "remaining_missing_frames.csv"
    with missing_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_dir", "episode", "agent", "frame_index", "target_visible", "bbox_source"])
        writer.writeheader()
        writer.writerows(missing_rows)
    summary = {
        "created_at": datetime.now().isoformat(),
        "output_root": str(root),
        **dict(totals),
        "fill_rate_among_originally_missing": (
            totals["source_yolo_finetuned"]
            / max(totals["source_yolo_finetuned"] + totals["source_missing"], 1)
        ),
        "groups": {name: dict(counts) for name, counts in sorted(groups.items())},
        "remaining_missing_csv": str(missing_csv),
    }
    summary_path = root / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
