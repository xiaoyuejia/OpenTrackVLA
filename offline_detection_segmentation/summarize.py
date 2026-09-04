"""Summarize a cache run and build a multi-scene visualization contact sheet."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .core import LABEL_OBSTACLE
from .rendering import visualization_path_for_image


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize one offline perception output")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--columns", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output_root = args.output_root
    cache_files = sorted((output_root / "frames").rglob("*.perception.npz"))
    if not cache_files:
        raise FileNotFoundError(f"no perception caches under {output_root / 'frames'}")

    records: List[Dict[str, Any]] = []
    for cache_path in cache_files:
        with np.load(cache_path, allow_pickle=False) as cache:
            metadata = json.loads(str(cache["metadata_json"].item()))
            scene_mask = cache["scene_mask"]
            image_relative_path = Path(metadata["image_relative_path"])
            parts = image_relative_path.parts
            scene = parts[1] if len(parts) > 1 and parts[0] == "frames" else parts[0]
            agent = parts[-2] if len(parts) >= 2 else "unknown"
            records.append(
                {
                    "cache": str(cache_path),
                    "image_relative_path": image_relative_path.as_posix(),
                    "visualization": str(
                        visualization_path_for_image(output_root, image_relative_path)
                    ),
                    "scene": scene,
                    "agent": agent,
                    "person_valid": bool(cache["person_valid"].item()),
                    "person_score": float(cache["person_score"].item()),
                    "obstacle_instances": int(len(cache["obstacle_scores"])),
                    "obstacle_ratio": float(np.mean(scene_mask == LABEL_OBSTACLE)),
                }
            )

    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["scene"], record["agent"])].append(record)

    groups: List[Dict[str, Any]] = []
    for (scene, agent), values in sorted(grouped.items()):
        person_scores = [value["person_score"] for value in values if value["person_valid"]]
        groups.append(
            {
                "scene": scene,
                "agent": agent,
                "frames": len(values),
                "person_found": sum(int(value["person_valid"]) for value in values),
                "person_score_mean": float(np.mean(person_scores)) if person_scores else 0.0,
                "obstacle_instances_mean": float(
                    np.mean([value["obstacle_instances"] for value in values])
                ),
                "obstacle_ratio_mean": float(
                    np.mean([value["obstacle_ratio"] for value in values])
                ),
                "obstacle_ratio_max": float(
                    np.max([value["obstacle_ratio"] for value in values])
                ),
            }
        )

    summary = {
        "output_root": str(output_root),
        "frames": len(records),
        "scenes": len({record["scene"] for record in records}),
        "person_found": sum(int(record["person_valid"]) for record in records),
        "frames_with_obstacles": sum(
            int(record["obstacle_instances"] > 0) for record in records
        ),
        "obstacle_ratio_mean": float(
            np.mean([record["obstacle_ratio"] for record in records])
        ),
        "obstacle_ratio_max": float(
            np.max([record["obstacle_ratio"] for record in records])
        ),
        "groups": groups,
        "records": records,
    }
    summary_path = output_root / "evaluation_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    representatives = [values[0] for _, values in sorted(grouped.items())]
    columns = max(1, int(args.columns))
    tile_width, image_height, label_height = 320, 240, 34
    tile_height = image_height + label_height
    rows = int(math.ceil(len(representatives) / columns))
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), (25, 25, 25))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, record in enumerate(representatives):
        preview = Image.open(record["visualization"]).convert("RGB")
        preview = ImageOps.fit(preview, (tile_width, image_height), method=Image.Resampling.LANCZOS)
        x = (index % columns) * tile_width
        y = (index // columns) * tile_height
        sheet.paste(preview, (x, y))
        scene_short = record["scene"].replace("UnrealTrack-", "")
        if len(scene_short) > 31:
            scene_short = scene_short[:28] + "..."
        line1 = f"{scene_short} | {record['agent']}"
        line2 = (
            f"person={int(record['person_valid'])} score={record['person_score']:.2f} "
            f"obs={record['obstacle_ratio']:.1%} n={record['obstacle_instances']}"
        )
        draw.rectangle((x, y + image_height, x + tile_width, y + tile_height), fill=(0, 0, 0))
        draw.text((x + 4, y + image_height + 2), line1, fill=(255, 255, 255), font=font)
        draw.text((x + 4, y + image_height + 17), line2, fill=(220, 220, 220), font=font)

    contact_sheet_path = output_root / "contact_sheet.jpg"
    sheet.save(contact_sheet_path, quality=92, optimize=True)

    print(
        f"frames={summary['frames']} scenes={summary['scenes']} "
        f"person={summary['person_found']} obstacle_frames={summary['frames_with_obstacles']} "
        f"obstacle_mean={summary['obstacle_ratio_mean']:.2%} "
        f"obstacle_max={summary['obstacle_ratio_max']:.2%}"
    )
    for group in groups:
        print(
            f"{group['scene']} | {group['agent']} | "
            f"person={group['person_found']}/{group['frames']} "
            f"score={group['person_score_mean']:.2f} "
            f"obs_mean={group['obstacle_ratio_mean']:.2%} "
            f"obs_max={group['obstacle_ratio_max']:.2%}"
        )
    print(f"summary={summary_path}")
    print(f"contact_sheet={contact_sheet_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
