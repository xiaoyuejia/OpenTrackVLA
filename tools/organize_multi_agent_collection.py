#!/usr/bin/env python3
"""Merge scattered UnrealZoo multi-agent collection folders by scene.

The output layout matches ``sim_data/New_paths_training_multi_agent``:

    output_root/
      UnrealTrack-Scene-ContinuousColor-v0/
        0.json
        0_drone_info.json
        0_robotdog_info.json
        0_drone.mp4
        0_robotdog.mp4
        0_global.mp4  # copied when present

By default the script only reports what it would keep/drop. Pass ``--apply`` to
create the output tree.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_SUFFIXES = ("json", "drone_info.json", "robotdog_info.json", "drone.mp4", "robotdog.mp4")
OPTIONAL_SUFFIXES = ("global.mp4",)


@dataclass
class Episode:
    source_root: Path
    scene: str
    stem: str
    summary_path: Path
    files: dict[str, Path]
    stat: dict[str, Any]


@dataclass
class Decision:
    episode: Episode
    keep: bool
    reason: str


def _float_stat(stat: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = stat.get(key, default)
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _int_stat(stat: dict[str, Any], key: str, default: int = 0) -> int:
    try:
        value = stat.get(key, default)
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def discover_episodes(source_roots: list[Path]) -> tuple[list[Episode], Counter[str]]:
    episodes: list[Episode] = []
    problems: Counter[str] = Counter()
    seen_summary_paths: set[Path] = set()

    for source_root in source_roots:
        for drone_info in sorted(source_root.rglob("*_drone_info.json")):
            scene_dir = drone_info.parent
            if not scene_dir.name.startswith("UnrealTrack-"):
                continue
            stem = drone_info.name[: -len("_drone_info.json")]
            summary_path = scene_dir / f"{stem}.json"
            if summary_path in seen_summary_paths:
                continue
            seen_summary_paths.add(summary_path)

            files = {
                "json": summary_path,
                "drone_info.json": drone_info,
                "robotdog_info.json": scene_dir / f"{stem}_robotdog_info.json",
                "drone.mp4": scene_dir / f"{stem}_drone.mp4",
                "robotdog.mp4": scene_dir / f"{stem}_robotdog.mp4",
            }
            for suffix in OPTIONAL_SUFFIXES:
                path = scene_dir / f"{stem}_{suffix}"
                if path.exists():
                    files[suffix] = path

            missing = [suffix for suffix in REQUIRED_SUFFIXES if not files[suffix].is_file()]
            if missing:
                problems[f"missing:{','.join(missing)}"] += 1
                continue

            try:
                stat = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                problems["bad_summary_json"] += 1
                continue
            if not isinstance(stat, dict):
                problems["summary_not_object"] += 1
                continue

            episodes.append(
                Episode(
                    source_root=source_root,
                    scene=scene_dir.name,
                    stem=stem,
                    summary_path=summary_path,
                    files=files,
                    stat=stat,
                )
            )

    return episodes, problems


def load_progress_rejections(progress_json: Path | None) -> set[Path]:
    """Return summary JSON paths rejected by the batch collector quality gate."""
    if progress_json is None:
        return set()
    data = json.loads(progress_json.read_text(encoding="utf-8"))
    rejected: set[Path] = set()
    maps = data.get("maps", {}) if isinstance(data, dict) else {}
    if not isinstance(maps, dict):
        return rejected
    for entry in maps.values():
        if not isinstance(entry, dict):
            continue
        for rejection in entry.get("quality_rejections", []) or []:
            if not isinstance(rejection, dict):
                continue
            for reason in rejection.get("reasons", []) or []:
                if not isinstance(reason, dict):
                    continue
                file_path = reason.get("file")
                if isinstance(file_path, str) and file_path:
                    rejected.add(Path(file_path).resolve())
    return rejected


def decide(
    episode: Episode,
    min_steps: int,
    min_drone_follow: float,
    min_robotdog_follow: float,
    require_success: bool,
    drop_collisions: bool,
    rejected_summary_paths: set[Path] | None = None,
) -> Decision:
    stat = episode.stat
    if rejected_summary_paths and episode.summary_path.resolve() in rejected_summary_paths:
        return Decision(episode, False, "quality_rejected_by_progress")

    total_step = _int_stat(stat, "total_step", 0)
    if total_step < min_steps:
        return Decision(episode, False, f"short_steps<{min_steps}")

    collision = _float_stat(stat, "collision", 0.0)
    if drop_collisions and collision > 0.0:
        return Decision(episode, False, "collision")

    if require_success and _float_stat(stat, "success", 0.0) < 1.0:
        return Decision(episode, False, "not_success")

    drone_follow = _float_stat(stat, "drone_following_rate", 0.0)
    if drone_follow < min_drone_follow:
        return Decision(episode, False, f"low_drone_follow<{min_drone_follow:g}")

    robotdog_follow = _float_stat(stat, "robotdog_following_rate", 0.0)
    if robotdog_follow < min_robotdog_follow:
        return Decision(episode, False, f"low_robotdog_follow<{min_robotdog_follow:g}")

    return Decision(episode, True, "keep")


def link_or_copy(src: Path, dst: Path, method: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if method == "copy":
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_outputs(
    decisions: list[Decision],
    output_root: Path,
    method: str,
    overwrite: bool,
) -> None:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"output root already exists: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    kept_by_scene: dict[str, list[Decision]] = defaultdict(list)
    for decision in decisions:
        if decision.keep:
            kept_by_scene[decision.episode.scene].append(decision)

    manifest: dict[str, Any] = {
        "output_root": str(output_root),
        "copy_method": method,
        "scenes": {},
        "episodes": [],
    }

    for scene in sorted(kept_by_scene):
        scene_dir = output_root / scene
        scene_dir.mkdir(parents=True, exist_ok=True)
        scene_decisions = kept_by_scene[scene]
        manifest["scenes"][scene] = len(scene_decisions)
        for new_idx, decision in enumerate(scene_decisions):
            episode = decision.episode
            new_stem = str(new_idx)
            rename_map = {
                "json": f"{new_stem}.json",
                "drone_info.json": f"{new_stem}_drone_info.json",
                "robotdog_info.json": f"{new_stem}_robotdog_info.json",
                "drone.mp4": f"{new_stem}_drone.mp4",
                "robotdog.mp4": f"{new_stem}_robotdog.mp4",
                "global.mp4": f"{new_stem}_global.mp4",
            }
            for suffix, filename in rename_map.items():
                src = episode.files.get(suffix)
                if src is not None and src.exists():
                    link_or_copy(src, scene_dir / filename, method)
            manifest["episodes"].append(
                {
                    "scene": scene,
                    "new_stem": new_stem,
                    "source_summary": str(episode.summary_path),
                    "source_root": str(episode.source_root),
                    "source_stem": episode.stem,
                    "total_step": _int_stat(episode.stat, "total_step", 0),
                    "drone_following_rate": _float_stat(episode.stat, "drone_following_rate", 0.0),
                    "robotdog_following_rate": _float_stat(episode.stat, "robotdog_following_rate", 0.0),
                    "drone_centered_rate": _float_stat(episode.stat, "drone_centered_rate", 0.0),
                    "robotdog_centered_rate": _float_stat(episode.stat, "robotdog_centered_rate", 0.0),
                    "drone_distance_rate": _float_stat(episode.stat, "drone_distance_rate", 0.0),
                    "robotdog_distance_rate": _float_stat(episode.stat, "robotdog_distance_rate", 0.0),
                    "effective_dt_s_mean": _float_stat(episode.stat, "effective_dt_s_mean", 0.0),
                    "success": _float_stat(episode.stat, "success", 0.0),
                    "collision": _float_stat(episode.stat, "collision", 0.0),
                }
            )

    (output_root / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def print_report(decisions: list[Decision], problems: Counter[str], output_root: Path) -> None:
    total = len(decisions)
    kept = sum(1 for item in decisions if item.keep)
    dropped = total - kept
    print(f"[scan] complete_episodes={total} keep={kept} drop={dropped} output={output_root}")
    if problems:
        print("[scan] discovery_problems")
        for reason, count in sorted(problems.items()):
            print(f"  {reason}: {count}")

    reason_counts = Counter(item.reason for item in decisions if not item.keep)
    if reason_counts:
        print("[filter] drop_reasons")
        for reason, count in sorted(reason_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {reason}: {count}")

    by_scene_total: Counter[str] = Counter(item.episode.scene for item in decisions)
    by_scene_keep: Counter[str] = Counter(item.episode.scene for item in decisions if item.keep)
    print("[scene] kept/total")
    for scene in sorted(by_scene_total):
        print(f"  {scene}: {by_scene_keep[scene]}/{by_scene_total[scene]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        action="append",
        type=Path,
        default=[],
        help="Source folder to scan. Can be passed more than once.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/hdt/ntv_data/sim_data/New_paths_training_multi_agent_cleaned"),
    )
    parser.add_argument("--min-steps", type=int, default=100)
    parser.add_argument("--min-drone-follow", type=float, default=0.5)
    parser.add_argument("--min-robotdog-follow", type=float, default=0.5)
    parser.add_argument("--require-success", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--drop-collisions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--progress-json",
        type=Path,
        default=None,
        help="Optional batch train_progress.json containing collector quality rejections.",
    )
    parser.add_argument(
        "--drop-progress-rejections",
        action="store_true",
        help="Drop episodes listed in --progress-json quality_rejections.",
    )
    parser.add_argument("--method", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument("--apply", action="store_true", help="Write the organized output tree.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_roots = args.source_root or [
        Path("/data/hdt/ntv_data/New Folder"),
        Path("/data/hdt/ntv_data/New Folder1"),
    ]
    episodes, problems = discover_episodes(source_roots)
    rejected_summary_paths = (
        load_progress_rejections(args.progress_json)
        if args.drop_progress_rejections
        else set()
    )
    decisions = [
        decide(
            episode,
            min_steps=args.min_steps,
            min_drone_follow=args.min_drone_follow,
            min_robotdog_follow=args.min_robotdog_follow,
            require_success=bool(args.require_success),
            drop_collisions=bool(args.drop_collisions),
            rejected_summary_paths=rejected_summary_paths,
        )
        for episode in episodes
    ]
    print_report(decisions, problems, args.output_root)
    if args.apply:
        write_outputs(decisions, args.output_root, args.method, bool(args.overwrite))
        print(f"[done] wrote {args.output_root}")
    else:
        print("[dry-run] pass --apply to write files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
