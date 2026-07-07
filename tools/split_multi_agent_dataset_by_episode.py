#!/usr/bin/env python3
"""Split processed multi-agent JSONL data by episode.

The script never moves or deletes source data. It writes episode lists and,
unless ``--list-only`` is used, creates a lightweight split tree with JSONL
links plus frames/vision_cache links so train.py can consume:

    <out_root>/train/jsonl
    <out_root>/val/jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_first_jsonl(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    return {}


def episode_record(jsonl_root: Path, path: Path) -> dict[str, Any]:
    rel = path.relative_to(jsonl_root)
    scene = rel.parent.as_posix()
    stem = path.stem
    first = read_first_jsonl(path)
    episode_id = str(first.get("episode_id") or f"{scene}/{stem}")
    samples = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples += 1
    return {
        "episode_id": episode_id,
        "scene": scene,
        "stem": stem,
        "jsonl": str(path),
        "relative_jsonl": rel.as_posix(),
        "samples": samples,
    }


def link_or_copy(src: Path, dst: Path, mode: str, directory: bool = False) -> None:
    if mode == "skip" or dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        os.symlink(src, dst, target_is_directory=directory)
    elif mode == "hardlink" and not directory:
        os.link(src, dst)
    elif mode == "copy":
        if directory:
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    else:
        raise ValueError(f"unsupported link mode {mode!r} for directory={directory}")


def write_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = path.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode_id", "scene", "stem", "relative_jsonl", "samples"])
        writer.writeheader()
        for row in records:
            writer.writerow({key: row.get(key, "") for key in writer.fieldnames})


def stratified_split(records: list[dict[str, Any]], val_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_scene[str(record["scene"])].append(record)
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    for scene, scene_records in sorted(by_scene.items()):
        items = list(scene_records)
        rng.shuffle(items)
        n_val = int(round(len(items) * val_ratio))
        if len(items) > 1 and val_ratio > 0.0:
            n_val = max(1, min(len(items) - 1, n_val))
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    train_ids = {row["episode_id"] for row in train}
    val_ids = {row["episode_id"] for row in val}
    overlap = train_ids & val_ids
    if overlap:
        raise RuntimeError(f"episode overlap detected: {sorted(overlap)[:5]}")
    return train, val


def create_split_tree(records: list[dict[str, Any]], jsonl_root: Path, split_jsonl_root: Path, mode: str) -> None:
    for record in records:
        src = Path(str(record["jsonl"]))
        dst = split_jsonl_root / str(record["relative_jsonl"])
        link_or_copy(src, dst, mode, directory=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None, help="Root containing frames/ and vision_cache/. Defaults to jsonl-root parent.")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--link-mode", choices=["symlink", "hardlink", "copy", "skip"], default="symlink")
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()

    jsonl_root = args.jsonl_root.resolve()
    data_root = args.data_root.resolve() if args.data_root else jsonl_root.parent.resolve()
    out_root = args.out_root.resolve()
    files = sorted(jsonl_root.rglob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no jsonl files under {jsonl_root}")
    records = [episode_record(jsonl_root, path) for path in files]
    train, val = stratified_split(records, float(args.val_ratio), int(args.seed))

    out_root.mkdir(parents=True, exist_ok=True)
    write_records(out_root / "train_episodes.json", train)
    write_records(out_root / "val_episodes.json", val)

    scene_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"train": 0, "val": 0, "samples_train": 0, "samples_val": 0})
    for split_name, split_records in (("train", train), ("val", val)):
        for record in split_records:
            scene_stats[str(record["scene"])][split_name] += 1
            scene_stats[str(record["scene"])][f"samples_{split_name}"] += int(record["samples"])
    report = {
        "jsonl_root": str(jsonl_root),
        "data_root": str(data_root),
        "out_root": str(out_root),
        "seed": int(args.seed),
        "val_ratio": float(args.val_ratio),
        "episodes_total": len(records),
        "episodes_train": len(train),
        "episodes_val": len(val),
        "samples_train": sum(int(row["samples"]) for row in train),
        "samples_val": sum(int(row["samples"]) for row in val),
        "scene_count": len(scene_stats),
        "scene_stats": scene_stats,
    }
    (out_root / "split_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.list_only:
        create_split_tree(train, jsonl_root, out_root / "train" / "jsonl", args.link_mode)
        create_split_tree(val, jsonl_root, out_root / "val" / "jsonl", args.link_mode)
        for name in ("frames", "vision_cache"):
            src = data_root / name
            if src.exists():
                link_or_copy(src, out_root / "train" / name, "symlink", directory=True)
                link_or_copy(src, out_root / "val" / name, "symlink", directory=True)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
