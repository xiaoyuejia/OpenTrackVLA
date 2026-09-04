#!/usr/bin/env python3
"""Build a split_manifest.json where bbox-empty episodes are held out for eval."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def bbox_is_zero(record: dict[str, Any], field: str) -> bool:
    bbox = record.get(field)
    if not isinstance(bbox, list) or len(bbox) < 4:
        return True
    try:
        return float(bbox[2]) <= 0.0 or float(bbox[3]) <= 0.0
    except Exception:
        return True


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"expected list records: {path}")
    return [item if isinstance(item, dict) else {} for item in data]


def episode_item(input_root: Path, drone_path: Path, zero: int, total: int) -> dict[str, Any]:
    rel = drone_path.relative_to(input_root)
    scene = drone_path.parent.name
    stem = drone_path.name.removesuffix("_drone_info.json")
    return {
        "key": str(rel).removesuffix("_drone_info.json"),
        "info": str(rel),
        "relative_dir": str(rel.parent),
        "stem": stem,
        "scene": scene,
        "source_group": rel.parts[0] if rel.parts else "",
        "zero_bbox_frames": int(zero),
        "bbox_frames": int(total),
        "zero_bbox_ratio": float(zero / total) if total > 0 else 1.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bbox-field", default="target_bbox")
    parser.add_argument(
        "--min-zero-ratio",
        type=float,
        default=1.0,
        help="Episode goes to test when zero_bbox_frames / bbox_frames >= this value.",
    )
    args = parser.parse_args()

    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"input root does not exist: {input_root}")

    train: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    scene_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "train": 0, "test": 0})

    for drone_path in sorted(input_root.rglob("*_drone_info.json"), key=lambda path: str(path)):
        dog_path = drone_path.with_name(drone_path.name.removesuffix("_drone_info.json") + "_robotdog_info.json")
        drone_records = load_records(drone_path)
        dog_records = load_records(dog_path) if dog_path.is_file() else []
        records = drone_records + dog_records
        total = len(records)
        zero = sum(1 for record in records if bbox_is_zero(record, args.bbox_field))
        item = episode_item(input_root, drone_path, zero, total)
        split = test if total == 0 or zero / max(total, 1) >= args.min_zero_ratio else train
        split.append(item)
        scene = item["scene"]
        scene_counts[scene]["total"] += 1
        scene_counts[scene]["test" if split is test else "train"] += 1

    manifest = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "split_mode": "bbox_zero_holdout",
        "bbox_field": args.bbox_field,
        "min_zero_ratio": float(args.min_zero_ratio),
        "episode_count": len(train) + len(test),
        "train_count": len(train),
        "test_count": len(test),
        "scene_counts": dict(sorted(scene_counts.items())),
        "train": sorted(train, key=lambda item: item["key"]),
        "test": sorted(test, key=lambda item: item["key"]),
    }

    output_root.mkdir(parents=True, exist_ok=True)
    out_path = output_root / "split_manifest.json"
    out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k not in {"train", "test"}}, ensure_ascii=False, indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
