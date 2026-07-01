#!/usr/bin/env python3
"""Collect Drone episodes into the single-agent sim_data layout."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path


DRONE_INFO_SUFFIX = "_drone_info.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Group Drone recordings by scene and assign consecutive episode IDs."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed-name", default="seed_100")
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of creating space-saving hard links.",
    )
    return parser.parse_args()


def transfer(source: Path, destination: Path, copy_files: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if copy_files:
        shutil.copy2(source, destination)
    else:
        os.link(source, destination)


def main() -> int:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()

    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output root is not empty: {output_root}")

    episodes_by_scene: dict[str, list[tuple[Path, Path, Path]]] = defaultdict(list)
    for info_path in sorted(input_root.rglob(f"*{DRONE_INFO_SUFFIX}")):
        stem = info_path.name[: -len(DRONE_INFO_SUFFIX)]
        video_path = info_path.with_name(f"{stem}_drone.mp4")
        status_path = info_path.with_name(f"{stem}.json")
        missing = [path for path in (video_path, status_path) if not path.is_file()]
        if missing:
            names = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(f"Missing files paired with {info_path}: {names}")
        episodes_by_scene[info_path.parent.name].append((video_path, info_path, status_path))

    if not episodes_by_scene:
        raise RuntimeError(f"No Drone episodes found under {input_root}")

    manifest: list[dict[str, str | int]] = []
    seed_root = output_root / args.seed_name
    for scene in sorted(episodes_by_scene):
        for episode_id, (video_path, info_path, status_path) in enumerate(episodes_by_scene[scene]):
            scene_root = seed_root / scene
            transfer(video_path, scene_root / f"{episode_id}.mp4", args.copy)
            transfer(info_path, scene_root / f"{episode_id}_info.json", args.copy)
            transfer(status_path, scene_root / f"{episode_id}.json", args.copy)
            manifest.append(
                {
                    "scene": scene,
                    "episode_id": episode_id,
                    "source_video": str(video_path.relative_to(input_root)),
                    "source_info": str(info_path.relative_to(input_root)),
                    "source_status": str(status_path.relative_to(input_root)),
                }
            )

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "organization_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Scenes: {len(episodes_by_scene)}")
    print(f"Episodes: {len(manifest)}")
    print(f"Files transferred: {len(manifest) * 3}")
    print(f"Method: {'copy' if args.copy else 'hardlink'}")
    print(f"Output: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
