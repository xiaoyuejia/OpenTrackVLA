#!/usr/bin/env python3
"""Build a deterministic recorded-trajectory eval manifest from data_lost."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def numeric_stem(path: Path) -> tuple[int, str]:
    stem = path.name.removesuffix("_drone_info.json")
    return (int(stem), stem) if stem.isdigit() else (2**63 - 1, stem)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--per-scene", type=int, default=25)
    parser.add_argument(
        "--total",
        type=int,
        default=None,
        help="Select this many episodes total while covering every scene.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = args.input_root.expanduser().resolve()
    if args.per_scene <= 0:
        raise SystemExit("--per-scene must be positive")

    groups: dict[str, list[Path]] = defaultdict(list)
    for path in root.rglob("*_drone_info.json"):
        groups[path.parent.name].append(path)

    if args.total is not None:
        if args.total < len(groups):
            raise SystemExit(
                f"--total={args.total} cannot cover all {len(groups)} scenes"
            )
        if args.total > sum(len(paths) for paths in groups.values()):
            raise SystemExit("--total exceeds the number of available episodes")

        rng = random.Random(args.seed)
        selected_by_scene: dict[str, list[Path]] = {}
        remaining_by_scene: dict[str, list[Path]] = {}
        for scene in sorted(groups):
            candidates = sorted(groups[scene], key=numeric_stem)
            rng.shuffle(candidates)
            selected_by_scene[scene] = [candidates[0]]
            remaining_by_scene[scene] = candidates[1:]

        remaining = args.total - len(groups)
        eligible = [scene for scene in sorted(groups) if remaining_by_scene[scene]]
        rng.shuffle(eligible)
        while remaining:
            progressed = False
            for scene in eligible:
                if remaining == 0:
                    break
                if remaining_by_scene[scene]:
                    selected_by_scene[scene].append(remaining_by_scene[scene].pop())
                    remaining -= 1
                    progressed = True
            if not progressed:
                raise SystemExit("not enough episodes to satisfy --total")
    else:
        selected_by_scene = {
            scene: sorted(paths, key=numeric_stem)[: args.per_scene]
            for scene, paths in groups.items()
        }

    items = []
    scene_counts = {}
    for scene in sorted(groups):
        selected = sorted(selected_by_scene[scene], key=numeric_stem)
        scene_counts[scene] = {
            "available": len(groups[scene]),
            "total": len(selected),
            "train": 0,
            "test": len(selected),
        }
        for info in selected:
            stem = info.name.removesuffix("_drone_info.json")
            relative_dir = str(info.parent.relative_to(root))
            items.append(
                {
                    "scene": scene,
                    "stem": stem,
                    "relative_dir": relative_dir,
                    "info": str(info.relative_to(root)),
                }
            )

    if not items:
        raise SystemExit(f"no *_drone_info.json files found under {root}")

    manifest = {
        "description": (
            f"{args.total} recorded evaluation trajectories covering every scene from {root}"
            if args.total is not None
            else f"Up to {args.per_scene} recorded evaluation trajectories per scene from {root}"
        ),
        "input_root": str(root),
        "output_root": str(args.output.expanduser().resolve().parent),
        "episode_count": len(items),
        "test_count": len(items),
        "requested_per_scene": args.per_scene,
        "requested_total": args.total,
        "selection_seed": args.seed if args.total is not None else None,
        "selection": (
            "seeded scene coverage without replacement"
            if args.total is not None
            else "numeric stem order without replacement"
        ),
        "scene_counts": scene_counts,
        "test": items,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}: {len(items)} episodes across {len(groups)} scenes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
