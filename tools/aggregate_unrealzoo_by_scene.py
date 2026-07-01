#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把递归分布的 UnrealZoo 双 Agent 原始数据按场景聚合并重新编号。

输入可以包含任意层级的采集批次目录。程序通过 ``*_drone_info.json`` 发现 episode，
将同一场景的数据放入同一目录，并重新编号为 ``0, 1, 2, ...``。默认使用硬链接，
不会复制大型视频。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REQUIRED_SUFFIXES = (
    "_drone.mp4",
    "_drone_info.json",
    "_robotdog.mp4",
    "_robotdog_info.json",
)
OPTIONAL_SUFFIXES = (
    ".json",
    "_global.mp4",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate UnrealZoo episodes by scene.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--copy", action="store_true", help="Copy files instead of using hard links.")
    return parser.parse_args()


def discover_episodes(input_root: Path) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    incomplete: list[dict[str, Any]] = []
    for drone_info in sorted(input_root.rglob("*_drone_info.json")):
        stem = drone_info.name[: -len("_drone_info.json")]
        source_dir = drone_info.parent
        required = [source_dir / f"{stem}{suffix}" for suffix in REQUIRED_SUFFIXES]
        missing = [path.name for path in required if not path.is_file()]
        if missing:
            incomplete.append(
                {
                    "source": str(drone_info.relative_to(input_root)),
                    "missing": missing,
                }
            )
            continue
        episodes.append(
            {
                "scene": source_dir.name,
                "source_dir": source_dir,
                "source_stem": stem,
                "source_key": str((source_dir / stem).relative_to(input_root)),
            }
        )
    if incomplete:
        raise RuntimeError("Found incomplete episodes:\n" + json.dumps(incomplete[:20], ensure_ascii=False, indent=2))
    if not episodes:
        raise RuntimeError(f"No complete episodes found under {input_root}")
    return episodes


def link_or_copy(source: Path, destination: Path, copy_only: bool) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    if not copy_only:
        try:
            os.link(source, destination)
            return "hardlink"
        except OSError:
            pass
    shutil.copy2(source, destination)
    return "copy"


def main() -> int:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if output_root == input_root or input_root in output_root.parents:
        raise ValueError("--output-root must not be inside --input-root")
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"--output-root must be empty: {output_root}")

    episodes = discover_episodes(input_root)
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        by_scene[episode["scene"]].append(episode)

    transfer_counts: Counter[str] = Counter()
    manifest_episodes: list[dict[str, Any]] = []
    for scene in sorted(by_scene):
        scene_episodes = sorted(by_scene[scene], key=lambda item: item["source_key"])
        for episode_id, episode in enumerate(scene_episodes):
            files: dict[str, str] = {}
            for suffix in REQUIRED_SUFFIXES + OPTIONAL_SUFFIXES:
                source = episode["source_dir"] / f"{episode['source_stem']}{suffix}"
                if not source.is_file():
                    continue
                destination = output_root / scene / f"{episode_id}{suffix}"
                method = link_or_copy(source, destination, args.copy)
                transfer_counts[method] += 1
                files[suffix] = str(destination.relative_to(output_root))
            manifest_episodes.append(
                {
                    "scene": scene,
                    "episode_id": str(episode_id),
                    "source": episode["source_key"],
                    "files": files,
                }
            )

    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "episode_count": len(manifest_episodes),
        "scene_count": len(by_scene),
        "scene_episode_counts": {scene: len(by_scene[scene]) for scene in sorted(by_scene)},
        "transfer_counts": dict(transfer_counts),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(
        json.dumps({"summary": summary, "episodes": manifest_episodes}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
