#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将 UnrealZoo 双 Agent 原始 episode 按场景固定比例划分为训练集和测试集。

每个 episode 以 ``*_drone_info.json`` 为索引，要求同时存在 Drone/RobotDog 的
视频和 info。输出默认使用硬链接，不复制大型视频；目录结构保持与输入一致。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
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
    parser = argparse.ArgumentParser(description="Split UnrealZoo multi-agent episodes into train/test sets.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-parts", type=int, default=10)
    parser.add_argument("--test-parts", type=int, default=1)
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=None,
        help="Deprecated global ratio. If set, it is converted to train/test parts for compatibility.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy", action="store_true", help="Copy files instead of using hard links.")
    return parser.parse_args()


def discover_episodes(input_root: Path) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    incomplete: list[str] = []
    for drone_info in sorted(input_root.rglob("*_drone_info.json")):
        stem = drone_info.name[: -len("_drone_info.json")]
        source_dir = drone_info.parent
        required = [source_dir / f"{stem}{suffix}" for suffix in REQUIRED_SUFFIXES]
        missing = [path.name for path in required if not path.is_file()]
        if missing:
            incomplete.append(f"{drone_info.relative_to(input_root)}: missing={missing}")
            continue
        episodes.append(
            {
                "key": str((source_dir / stem).relative_to(input_root)),
                "relative_dir": str(source_dir.relative_to(input_root)),
                "stem": stem,
                "scene": source_dir.name,
            }
        )
    if incomplete:
        raise RuntimeError("Found incomplete episodes:\n" + "\n".join(incomplete[:20]))
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


def materialize_split(
    input_root: Path,
    output_root: Path,
    split_name: str,
    episodes: list[dict[str, Any]],
    copy_only: bool,
) -> dict[str, int]:
    counts = {"hardlink": 0, "copy": 0}
    for episode in episodes:
        source_dir = input_root / episode["relative_dir"]
        destination_dir = output_root / split_name / episode["relative_dir"]
        stem = episode["stem"]
        for suffix in REQUIRED_SUFFIXES + OPTIONAL_SUFFIXES:
            source = source_dir / f"{stem}{suffix}"
            if not source.is_file():
                continue
            method = link_or_copy(source, destination_dir / source.name, copy_only)
            counts[method] += 1
    return counts


def main() -> int:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if output_root == input_root or input_root in output_root.parents:
        raise ValueError("--output-root must not be inside --input-root")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output root is not empty: {output_root}")
    if args.train_ratio is not None:
        if not 0.0 < args.train_ratio < 1.0:
            raise ValueError("--train-ratio must be between 0 and 1")
        args.train_parts = max(1, round(args.train_ratio * 1000))
        args.test_parts = max(1, round((1.0 - args.train_ratio) * 1000))
    if args.train_parts < 1 or args.test_parts < 1:
        raise ValueError("--train-parts and --test-parts must both be positive")

    episodes = discover_episodes(input_root)
    by_scene: dict[str, list[dict[str, Any]]] = {}
    for episode in episodes:
        by_scene.setdefault(episode["scene"], []).append(episode)

    train_episodes: list[dict[str, Any]] = []
    test_episodes: list[dict[str, Any]] = []
    scene_counts: dict[str, dict[str, int]] = {}
    denominator = args.train_parts + args.test_parts

    for scene, scene_episodes in sorted(by_scene.items()):
        shuffled = scene_episodes.copy()
        random.Random(f"{args.seed}:{scene}").shuffle(shuffled)
        test_count = max(1, round(len(shuffled) * args.test_parts / denominator))
        if len(shuffled) > 1:
            test_count = min(test_count, len(shuffled) - 1)
        scene_test = sorted(shuffled[:test_count], key=lambda item: item["key"])
        scene_train = sorted(shuffled[test_count:], key=lambda item: item["key"])
        train_episodes.extend(scene_train)
        test_episodes.extend(scene_test)
        scene_counts[scene] = {
            "total": len(scene_episodes),
            "train": len(scene_train),
            "test": len(scene_test),
        }

    train_episodes = sorted(train_episodes, key=lambda item: item["key"])
    test_episodes = sorted(test_episodes, key=lambda item: item["key"])

    transfer_counts = {
        "train": materialize_split(input_root, output_root, "train_raw", train_episodes, args.copy),
        "test": materialize_split(input_root, output_root, "test_raw", test_episodes, args.copy),
    }
    manifest = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "seed": args.seed,
        "train_parts": args.train_parts,
        "test_parts": args.test_parts,
        "episode_count": len(episodes),
        "train_count": len(train_episodes),
        "test_count": len(test_episodes),
        "scene_counts": scene_counts,
        "transfer_counts": transfer_counts,
        "train": train_episodes,
        "test": test_episodes,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in manifest.items() if key not in {"train", "test"}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
