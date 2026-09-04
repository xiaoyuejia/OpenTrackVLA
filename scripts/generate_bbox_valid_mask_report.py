#!/usr/bin/env python3
"""Generate an agent/frame-level bbox validity mask from raw episode metadata."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path


def bbox_valid(value: object) -> bool:
    try:
        if not isinstance(value, (list, tuple)) or len(value) < 4:
            return False
        coords = [float(item) for item in value[:4]]
        return all(math.isfinite(item) for item in coords) and coords[2] > 0.0 and coords[3] > 0.0
    except (TypeError, ValueError, OverflowError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    episodes = sorted({(row["relative_dir"], str(row["episode"])) for row in manifest["files"]})
    source_root = Path(manifest["source_root"])
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()

    fields = [
        "relative_dir", "episode", "agent", "frame_index", "target_visible",
        "bbox_valid_mask", "semantic_conflict", "target_bbox",
    ]
    with args.output_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for relative_dir, episode in episodes:
            for agent in ("drone", "robotdog"):
                info_path = source_root / relative_dir / f"{episode}_{agent}_info.json"
                rows = json.loads(info_path.read_text(encoding="utf-8"))
                for frame_index, row in enumerate(rows):
                    raw_bbox = row.get("target_bbox", [0, 0, 0, 0])
                    valid = bbox_valid(raw_bbox)
                    visible = bool(row.get("target_visible", False))
                    conflict = visible and not valid
                    writer.writerow({
                        "relative_dir": relative_dir,
                        "episode": episode,
                        "agent": agent,
                        "frame_index": frame_index,
                        "target_visible": visible,
                        "bbox_valid_mask": valid,
                        "semantic_conflict": conflict,
                        "target_bbox": json.dumps(raw_bbox, ensure_ascii=False),
                    })
                    counts["agent_frames"] += 1
                    counts[f"{agent}_frames"] += 1
                    counts["bbox_valid"] += int(valid)
                    counts["bbox_invalid"] += int(not valid)
                    counts["visible_bbox_conflict"] += int(conflict)
                    counts[f"{agent}_visible_bbox_conflict"] += int(conflict)

    summary_path = args.summary or args.output_csv.with_suffix(".summary.json")
    summary = {
        "created_at": datetime.now().isoformat(),
        "source_root": str(source_root),
        "manifest": str(args.manifest.resolve()),
        "episodes": len(episodes),
        **dict(counts),
        "output_csv": str(args.output_csv.resolve()),
        "definition": "finite raw xywh with width > 0 and height > 0",
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
