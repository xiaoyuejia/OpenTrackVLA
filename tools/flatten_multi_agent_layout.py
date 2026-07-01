#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


RAW_SUFFIXES = [
    ".json",
    "_drone.mp4",
    "_drone_info.json",
    "_robotdog.mp4",
    "_robotdog_info.json",
    "_global.mp4",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_symlink(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    if dst.is_symlink():
        current = os.readlink(dst)
        if Path(current) == src:
            return
        dst.unlink()
    elif dst.exists():
        return
    dst.symlink_to(src)


def iter_raw_episodes(raw_root: Path) -> Iterable[Tuple[Path, str, Path]]:
    for drone_info in sorted(raw_root.rglob("*_drone_info.json")):
        stem = drone_info.name[: -len("_drone_info.json")]
        run_dir = drone_info.parent
        if not (run_dir / f"{stem}_robotdog_info.json").exists():
            continue
        if not (run_dir / f"{stem}_drone.mp4").exists():
            continue
        if not (run_dir / f"{stem}_robotdog.mp4").exists():
            continue
        yield run_dir, stem, run_dir.relative_to(raw_root)


def make_flat_stem(rel_run_dir: Path, stem: str) -> str:
    parts = list(rel_run_dir.parts)
    if len(parts) >= 2 and parts[-1].startswith("UnrealTrack-"):
        parts = parts[:-1]
    return "__".join(parts + [stem])


def replace_strings(obj: Any, old_prefix: str, new_prefix: str) -> Any:
    if isinstance(obj, str):
        return obj.replace(old_prefix, new_prefix)
    if isinstance(obj, list):
        return [replace_strings(v, old_prefix, new_prefix) for v in obj]
    if isinstance(obj, dict):
        return {k: replace_strings(v, old_prefix, new_prefix) for k, v in obj.items()}
    return obj


def rewrite_jsonl(src_jsonl: Path, dst_jsonl: Path, old_prefix: str, new_prefix: str, env_name: str, flat_stem: str) -> int:
    ensure_dir(dst_jsonl.parent)
    n = 0
    with src_jsonl.open("r", encoding="utf-8") as fin, dst_jsonl.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            sample: Dict[str, Any] = json.loads(line)
            sample = replace_strings(sample, old_prefix, new_prefix)
            sample["episode_id"] = f"{env_name}/{flat_stem}"
            sample["rel_run_dir"] = env_name
            sample["episode_stem"] = flat_stem
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
            n += 1
    return n


def flatten_split(args: argparse.Namespace, split: str) -> Dict[str, int]:
    raw_root = args.raw_split_root / f"{split}_raw"
    src_data = args.data_root / split
    dst_raw = args.flat_raw_root / f"{split}_raw"
    dst_data = args.flat_data_root / split

    stats = {
        "raw_episodes": 0,
        "raw_links": 0,
        "jsonl_written": 0,
        "samples": 0,
        "frame_links": 0,
        "cache_links": 0,
        "missing_jsonl": 0,
        "missing_frames": 0,
    }

    for run_dir, stem, rel_run_dir in iter_raw_episodes(raw_root):
        env_name = rel_run_dir.parts[-1]
        flat_stem = make_flat_stem(rel_run_dir, stem)
        stats["raw_episodes"] += 1

        for suffix in RAW_SUFFIXES:
            src = run_dir / f"{stem}{suffix}"
            if src.exists():
                safe_symlink(src.resolve(), dst_raw / env_name / f"{flat_stem}{suffix}")
                stats["raw_links"] += 1

        src_episode_frames = src_data / "frames" / rel_run_dir / stem
        dst_episode_frames = dst_data / "frames" / env_name / flat_stem
        if src_episode_frames.exists():
            safe_symlink(src_episode_frames.resolve(), dst_episode_frames)
            stats["frame_links"] += 1
        else:
            stats["missing_frames"] += 1

        src_cache = src_data / "vision_cache" / "frames" / rel_run_dir / stem
        dst_cache = dst_data / "vision_cache" / "frames" / env_name / flat_stem
        if src_cache.exists():
            safe_symlink(src_cache.resolve(), dst_cache)
            stats["cache_links"] += 1

        src_jsonl = src_data / "jsonl" / rel_run_dir / f"{stem}.jsonl"
        dst_jsonl = dst_data / "jsonl" / env_name / f"{flat_stem}.jsonl"
        if src_jsonl.exists():
            old_prefix = f"frames/{rel_run_dir.as_posix()}/{stem}/"
            new_prefix = f"frames/{env_name}/{flat_stem}/"
            stats["samples"] += rewrite_jsonl(src_jsonl, dst_jsonl, old_prefix, new_prefix, env_name, flat_stem)
            stats["jsonl_written"] += 1
        else:
            stats["missing_jsonl"] += 1

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flatten nested two-agent split/data layout to old shallow layout.")
    parser.add_argument("--raw_split_root", type=Path, default=Path("/data/hdt/ntv_data/sim_data/data_multi_agent_split_10to1"))
    parser.add_argument("--data_root", type=Path, default=Path("/data/hdt/ntv_data/data/data_multi_agent_10to1"))
    parser.add_argument("--flat_raw_root", type=Path, default=Path("/data/hdt/ntv_data/sim_data/data_multi_agent_split_10to1_flat"))
    parser.add_argument("--flat_data_root", type=Path, default=Path("/data/hdt/ntv_data/data/data_multi_agent_10to1_flat"))
    parser.add_argument("--splits", nargs="+", default=["train", "test"], choices=["train", "test"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for split in args.splits:
        stats = flatten_split(args, split)
        print(split, stats)
    print(f"Flat raw root: {args.flat_raw_root}")
    print(f"Flat data root: {args.flat_data_root}")


if __name__ == "__main__":
    main()
