#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按原始数据聚合 manifest 迁移已抽取 frames 和视觉缓存。

旧处理目录通常保留采集批次层级：
``frames/<source_key>/<agent>/frame_x.jpg``。
本工具将其硬链接为：
``frames/seed_hand/<scene>/<episode_id>/<agent>/frame_x.jpg``。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate processed UnrealZoo data using aggregate manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-processed-root", type=Path, required=True)
    parser.add_argument("--output-processed-root", type=Path, required=True)
    parser.add_argument("--include-cache", action="store_true")
    parser.add_argument("--copy", action="store_true")
    return parser.parse_args()


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


def migrate_tree(source: Path, destination: Path, copy_only: bool, counts: Counter[str]) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    for source_file in source.rglob("*"):
        if not source_file.is_file():
            continue
        destination_file = destination / source_file.relative_to(source)
        counts[link_or_copy(source_file, destination_file, copy_only)] += 1


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    source_root = args.source_processed_root.expanduser().resolve()
    output_root = args.output_processed_root.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    counts: Counter[str] = Counter()

    for episode in manifest["episodes"]:
        source_key = Path(episode["source"])
        destination_key = Path("seed_hand") / episode["scene"] / episode["episode_id"]
        migrate_tree(
            source_root / "frames" / source_key,
            output_root / "frames" / destination_key,
            args.copy,
            counts,
        )
        if args.include_cache:
            migrate_tree(
                source_root / "vision_cache" / "frames" / source_key,
                output_root / "vision_cache" / "frames" / destination_key,
                args.copy,
                counts,
            )

    summary = {
        "manifest": str(manifest_path),
        "source_processed_root": str(source_root),
        "output_processed_root": str(output_root),
        "episode_count": len(manifest["episodes"]),
        "include_cache": args.include_cache,
        "transfer_counts": dict(counts),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "migration_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
