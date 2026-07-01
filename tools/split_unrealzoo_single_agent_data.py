#!/usr/bin/env python3
"""Split single-agent UnrealZoo episodes per scene into train and test sets."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a deterministic per-scene single-agent train/test split.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-parts", type=int, default=10)
    parser.add_argument("--test-parts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy", action="store_true", help="Copy files instead of using hard links.")
    return parser.parse_args()


def transfer(source: Path, destination: Path, copy_files: bool) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    if copy_files:
        shutil.copy2(source, destination)
        return "copy"
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def discover(input_root: Path) -> dict[str, list[dict[str, Any]]]:
    scenes: dict[str, list[dict[str, Any]]] = {}
    for info_path in sorted(input_root.rglob("*_info.json"), key=lambda path: str(path)):
        stem = info_path.name.removesuffix("_info.json")
        video_path = info_path.with_name(f"{stem}.mp4")
        status_path = info_path.with_name(f"{stem}.json")
        missing = [path.name for path in (video_path, status_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{info_path}: missing paired files {missing}")
        scene = info_path.parent.name
        scenes.setdefault(scene, []).append(
            {
                "scene": scene,
                "stem": stem,
                "relative_dir": str(info_path.parent.relative_to(input_root)),
                "video": str(video_path.relative_to(input_root)),
                "info": str(info_path.relative_to(input_root)),
                "status": str(status_path.relative_to(input_root)),
            }
        )
    if not scenes:
        raise RuntimeError(f"No <stem>.mp4 + <stem>_info.json episodes found under {input_root}")
    return scenes


def main() -> int:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")
    if args.train_parts < 1 or args.test_parts < 1:
        raise ValueError("--train-parts and --test-parts must both be positive")
    if output_root == input_root or input_root in output_root.parents:
        raise ValueError("--output-root must not be inside --input-root")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output root is not empty: {output_root}")

    scenes = discover(input_root)
    train: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    scene_counts: dict[str, dict[str, int]] = {}
    denominator = args.train_parts + args.test_parts

    for scene, episodes in sorted(scenes.items()):
        shuffled = episodes.copy()
        random.Random(f"{args.seed}:{scene}").shuffle(shuffled)
        test_count = max(1, round(len(shuffled) * args.test_parts / denominator))
        if len(shuffled) > 1:
            test_count = min(test_count, len(shuffled) - 1)
        scene_test = sorted(shuffled[:test_count], key=lambda item: (item["relative_dir"], item["stem"]))
        scene_train = sorted(shuffled[test_count:], key=lambda item: (item["relative_dir"], item["stem"]))
        train.extend(scene_train)
        test.extend(scene_test)
        scene_counts[scene] = {"total": len(episodes), "train": len(scene_train), "test": len(scene_test)}

    transfer_counts = {"hardlink": 0, "copy": 0}
    for split_name, episodes in (("train_raw", train), ("test_raw", test)):
        for item in episodes:
            relative_dir = Path(item["relative_dir"])
            for key in ("video", "info", "status"):
                source = input_root / item[key]
                destination = output_root / split_name / relative_dir / source.name
                method = transfer(source, destination, args.copy)
                transfer_counts[method] += 1

    manifest = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "seed": args.seed,
        "train_parts": args.train_parts,
        "test_parts": args.test_parts,
        "episode_count": len(train) + len(test),
        "train_count": len(train),
        "test_count": len(test),
        "scene_counts": scene_counts,
        "transfer_counts": transfer_counts,
        "train": train,
        "test": test,
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
