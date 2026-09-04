#!/usr/bin/env python3
"""Build a scene-covered 100-episode evaluation subset from the fixed 70:30 split.

The dataset itself is never duplicated.  The output is a self-contained
evaluation reference directory containing symlinks to each selected episode's
two videos and three JSON files, plus a manifest accepted by
``eval_unrealzoo_multi_agent.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


DEFAULT_SPLIT = Path(
    "/data/hdt/ntv_data/data/"
    "data7_8_camera_m40_pose_fixed_dt_exact_bbox_global_base_split_70_30/"
    "val_episodes.json"
)
DEFAULT_CAMERA_ROOT = Path(
    "/data/hdt/ntv_data/sim_data/data7_8_by_camera"
)
DEFAULT_CAMERA_MANIFEST = (
    DEFAULT_CAMERA_ROOT
    / "new_effective_dt/camera_drone_m40_robotdog_m8_mount_170_0_120/manifest.json"
)


def stable_order(item: dict) -> str:
    raw = f"{item['scene']}/{item['stem']}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def allocate(groups: dict[str, list[dict]], target: int) -> dict[str, int]:
    if target < len(groups):
        raise ValueError(f"target={target} is smaller than scene count={len(groups)}")
    if target > sum(map(len, groups.values())):
        raise ValueError("target is greater than available validation episodes")
    counts = {scene: 1 for scene in groups}
    remaining = target - len(groups)
    capacity = {scene: len(items) - 1 for scene, items in groups.items()}
    capacity_total = sum(capacity.values())
    quotas = {
        scene: remaining * capacity[scene] / capacity_total
        for scene in groups
    }
    for scene in groups:
        counts[scene] += int(quotas[scene])
    unassigned = target - sum(counts.values())
    ranked = sorted(
        groups,
        key=lambda scene: (-(quotas[scene] - int(quotas[scene])), scene),
    )
    for scene in ranked:
        if not unassigned:
            break
        if counts[scene] < len(groups[scene]):
            counts[scene] += 1
            unassigned -= 1
    if unassigned:
        raise RuntimeError("could not allocate requested subset across scenes")
    return counts


def symlink(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--val-episodes", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--camera-root", type=Path, default=DEFAULT_CAMERA_ROOT)
    parser.add_argument("--camera-manifest", type=Path, default=DEFAULT_CAMERA_MANIFEST)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    val_episodes = json.loads(args.val_episodes.read_text(encoding="utf-8"))
    camera_manifest = json.loads(args.camera_manifest.read_text(encoding="utf-8"))
    source_by_key = {
        (item["scene_id"], str(item["episode_id"])): item
        for item in camera_manifest["episodes"]
    }
    groups: dict[str, list[dict]] = defaultdict(list)
    for entry in val_episodes:
        key = (entry["scene"], str(entry["stem"]))
        if key not in source_by_key:
            raise KeyError(f"split episode absent from by_camera manifest: {key}")
        groups[entry["scene"]].append(entry)
    groups = dict(sorted(groups.items()))
    counts = allocate(groups, args.episodes)
    selected = []
    for scene, entries in groups.items():
        selected.extend(sorted(entries, key=stable_order)[: counts[scene]])
    selected.sort(key=lambda item: (item["scene"], int(item["stem"])))

    args.output.mkdir(parents=True)
    raw_root = args.output / "test_raw"
    manifest_items = []
    required = (".json", "_drone.mp4", "_drone_info.json", "_robotdog.mp4", "_robotdog_info.json")
    for entry in selected:
        source = source_by_key[(entry["scene"], str(entry["stem"]))]
        files = source["files"]
        for suffix in required:
            rel = Path(files[suffix])
            symlink(args.camera_root / rel, raw_root / rel)
        info_rel = files["_drone_info.json"]
        manifest_items.append(
            {
                "scene": entry["scene"],
                "stem": str(entry["stem"]),
                "relative_dir": str(Path(info_rel).parent),
                "info": info_rel,
            }
        )

    scene_counts = {scene: {"test": counts[scene]} for scene in groups}
    manifest = {
        "description": "100-episode scene-covered subset of the fixed data7_8 by_camera 70:30 validation split",
        "input_root": str(args.camera_root),
        "output_root": str(args.output),
        "episode_count": len(selected),
        "test_count": len(selected),
        "scene_counts": scene_counts,
        "test": manifest_items,
    }
    (args.output / "eval_manifest_100.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "source_val_episodes_913.json").write_text(
        json.dumps(val_episodes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    index = ["scene\tstem\tdata_dir"]
    index.extend(
        f"{item['scene']}\t{item['stem']}\t{item['relative_dir']}"
        for item in manifest_items
    )
    (args.output / "index.tsv").write_text("\n".join(index) + "\n", encoding="utf-8")
    (args.output / "README.md").write_text(
        "# data7_8 by-camera validation reference (100)\n\n"
        "This directory is a link-only reference: media and episode JSON files remain in "
        "`data7_8_by_camera`. `eval_manifest_100.json` is the exact evaluation input. "
        "The 100 episodes are selected deterministically from the fixed 913-episode "
        "70:30 validation split, with every scene represented.\n",
        encoding="utf-8",
    )
    print(f"created={args.output}")
    print(f"episodes={len(selected)} scenes={len(groups)}")
    print("scene_counts=" + json.dumps({scene: counts[scene] for scene in groups}, sort_keys=True))


if __name__ == "__main__":
    main()
