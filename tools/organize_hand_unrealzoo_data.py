#!/usr/bin/env python3
"""整理手工采集的 UnrealZoo 双 Agent episode，并汇总路径配置。"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional


EPISODE_SUFFIXES = (
    ".json",
    "_drone.mp4",
    "_drone_info.json",
    "_robotdog.mp4",
    "_robotdog_info.json",
    "_global.mp4",
)
REQUIRED_SUFFIXES = (
    "_drone.mp4",
    "_drone_info.json",
    "_robotdog.mp4",
    "_robotdog_info.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Organize hand-collected UnrealZoo multi-agent episodes.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--copy", action="store_true", help="Copy files instead of using hard links when possible.")
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


def episode_stems(scene_dir: Path) -> list[str]:
    suffix = "_drone_info.json"
    return sorted(path.name[: -len(suffix)] for path in scene_dir.glob(f"*{suffix}"))


def parse_json_with_missing_comma_repair(text: str) -> tuple[Any, Optional[str]]:
    """解析 JSON；仅尝试修复报错行之前列表项末尾缺少逗号的情况。"""
    try:
        return json.loads(text), None
    except json.JSONDecodeError as exc:
        lines = text.splitlines()
        previous = exc.lineno - 2
        if previous < 0 or previous >= len(lines):
            raise
        stripped = lines[previous].rstrip()
        if not stripped.endswith(("]", "}")) or stripped.endswith((",", "],", "},")):
            raise
        lines[previous] = stripped + ","
        repaired = "\n".join(lines) + ("\n" if text.endswith("\n") else "")
        return json.loads(repaired), repaired


def main() -> int:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = (args.output_root or (input_root / "organized")).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    source_dirs = sorted(
        path for path in input_root.iterdir()
        if path.is_dir() and path.resolve() != output_root
    )
    scene_counts: dict[str, int] = defaultdict(int)
    episodes: list[dict[str, Any]] = []
    path_configs: list[dict[str, Any]] = []
    missing_path_configs: list[str] = []
    invalid_path_configs: list[dict[str, str]] = []
    repaired_path_configs: list[str] = []
    incomplete_episodes: list[dict[str, Any]] = []
    transfer_counts: dict[str, int] = defaultdict(int)

    path_config_dir = output_root / "path_configs"
    for source_dir in source_dirs:
        source_id = source_dir.name
        path_config = source_dir / "1.json"
        if path_config.exists():
            config_dest = path_config_dir / f"{source_id}.json"
            record: dict[str, Any] = {
                "source_id": source_id,
                "source_file": str(path_config.relative_to(input_root)),
                "organized_file": str(config_dest.relative_to(output_root)),
            }
            try:
                config_obj, repaired_text = parse_json_with_missing_comma_repair(path_config.read_text(encoding="utf-8"))
                record["config"] = config_obj
                if repaired_text is None:
                    transfer_counts[link_or_copy(path_config, config_dest, args.copy)] += 1
                else:
                    config_dest.parent.mkdir(parents=True, exist_ok=True)
                    if config_dest.exists():
                        config_dest.unlink()
                    config_dest.write_text(repaired_text, encoding="utf-8")
                    record["repaired"] = True
                    repaired_path_configs.append(source_id)
                    transfer_counts["repaired_copy"] += 1
            except json.JSONDecodeError as exc:
                error = f"{exc.msg} at line {exc.lineno}, column {exc.colno}"
                transfer_counts[link_or_copy(path_config, config_dest, args.copy)] += 1
                record["config"] = None
                record["parse_error"] = error
                invalid_path_configs.append({"source_id": source_id, "error": error})
            path_configs.append(record)
        else:
            missing_path_configs.append(source_id)

        for scene_dir in sorted(path for path in source_dir.iterdir() if path.is_dir()):
            scene_id = scene_dir.name
            for source_stem in episode_stems(scene_dir):
                required = [scene_dir / f"{source_stem}{suffix}" for suffix in REQUIRED_SUFFIXES]
                missing = [path.name for path in required if not path.exists()]
                if missing:
                    incomplete_episodes.append(
                        {"source_id": source_id, "scene_id": scene_id, "source_stem": source_stem, "missing": missing}
                    )
                    continue

                episode_id = str(scene_counts[scene_id])
                scene_counts[scene_id] += 1
                destination_dir = output_root / "seed_hand" / scene_id
                files: dict[str, str] = {}
                for suffix in EPISODE_SUFFIXES:
                    source_file = scene_dir / f"{source_stem}{suffix}"
                    if not source_file.exists():
                        continue
                    destination = destination_dir / f"{episode_id}{suffix}"
                    transfer_counts[link_or_copy(source_file, destination, args.copy)] += 1
                    files[suffix] = str(destination.relative_to(output_root))

                episodes.append(
                    {
                        "episode_id": episode_id,
                        "scene_id": scene_id,
                        "source_id": source_id,
                        "source_stem": source_stem,
                        "human_path_config": (
                            f"path_configs/{source_id}.json" if path_config.exists() else None
                        ),
                        "files": files,
                    }
                )

    summary = {
        "source_root": str(input_root),
        "organized_root": str(output_root),
        "episode_count": len(episodes),
        "scene_count": len(scene_counts),
        "path_config_count": len(path_configs),
        "missing_path_configs": missing_path_configs,
        "repaired_path_configs": repaired_path_configs,
        "invalid_path_configs": invalid_path_configs,
        "incomplete_episodes": incomplete_episodes,
        "transfer_counts": dict(transfer_counts),
    }
    train = {
        "metadata": {
            "description": "Aggregated hand-collected UnrealZoo path configurations and organized episode index.",
            **summary,
        },
        "episodes": episodes,
        "path_configs": path_configs,
    }
    (output_root / "train.json").write_text(json.dumps(train, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "manifest.json").write_text(
        json.dumps({"summary": summary, "episodes": episodes}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
