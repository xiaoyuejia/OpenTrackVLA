#!/usr/bin/env python3
"""Flatten multi-people recordings into the data7_8 on-disk layout.

The source layout is ``<group>/seed_100/<map>/<episode files>`` while the
data7_8 layout is ``<map>/<episode files>``.  A map occurs in several source
groups, so episode numbers must be made unique per map.  This tool creates a
non-destructive normalized view using hard links by default.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path


EPISODE_JSON_RE = re.compile(r"^(\d+)\.json$")
ARTIFACT_TEMPLATES = (
    "{id}.json",
    "{id}_drone.mp4",
    "{id}_robotdog.mp4",
    "{id}_drone_info.json",
    "{id}_robotdog_info.json",
)


def numeric_first(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)


def scan_source(source: Path) -> dict[str, list[tuple[str, str, int, Path]]]:
    """Return all complete recordings keyed by map name."""
    grouped: dict[str, list[tuple[str, str, int, Path]]] = defaultdict(list)
    for group_dir in sorted((p for p in source.iterdir() if p.is_dir()), key=lambda p: numeric_first(p.name)):
        for seed_dir in sorted((p for p in group_dir.iterdir() if p.is_dir()), key=lambda p: p.name):
            for map_dir in sorted((p for p in seed_dir.iterdir() if p.is_dir()), key=lambda p: p.name):
                for path in sorted(map_dir.iterdir(), key=lambda p: p.name):
                    match = EPISODE_JSON_RE.match(path.name)
                    if not match:
                        continue
                    source_id = int(match.group(1))
                    missing = [
                        template.format(id=source_id)
                        for template in ARTIFACT_TEMPLATES
                        if not (map_dir / template.format(id=source_id)).is_file()
                    ]
                    if missing:
                        raise RuntimeError(
                            f"incomplete episode: {map_dir} id={source_id}; missing={missing}"
                        )
                    grouped[map_dir.name].append((group_dir.name, seed_dir.name, source_id, map_dir))
    if not grouped:
        raise RuntimeError(f"no recordings found under {source}")
    return grouped


def ensure_empty_destination(destination: Path) -> None:
    if destination.exists():
        entries = list(destination.iterdir())
        if entries:
            raise RuntimeError(
                f"destination is not empty: {destination}. Refusing to mix runs."
            )
    else:
        destination.mkdir(parents=True)


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    if copy:
        import shutil

        shutil.copy2(src, dst)
    else:
        os.link(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--copy", action="store_true", help="copy files instead of creating hard links")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source = args.source.resolve()
    destination = args.destination.resolve()
    if not source.is_dir():
        raise RuntimeError(f"source does not exist: {source}")
    if source == destination or source in destination.parents:
        raise RuntimeError("destination must be outside the source tree")

    grouped = scan_source(source)
    rows: list[dict[str, object]] = []
    for map_name, records in sorted(grouped.items()):
        records.sort(key=lambda row: (numeric_first(row[0]), row[1], row[2]))
        for normalized_id, (group, seed, source_id, map_dir) in enumerate(records):
            rows.append(
                {
                    "map": map_name,
                    "normalized_episode_id": normalized_id,
                    "source_group": group,
                    "source_seed": seed,
                    "source_episode_id": source_id,
                    "source_directory": str(map_dir),
                }
            )

    print(
        f"[plan] maps={len(grouped)} episodes={len(rows)} "
        f"artifacts={len(rows) * len(ARTIFACT_TEMPLATES)} mode={'copy' if args.copy else 'hardlink'}"
    )
    if args.dry_run:
        return 0

    ensure_empty_destination(destination)
    try:
        for row in rows:
            map_dir = destination / str(row["map"])
            map_dir.mkdir(exist_ok=True)
            source_dir = Path(str(row["source_directory"]))
            source_id = int(row["source_episode_id"])
            target_id = int(row["normalized_episode_id"])
            for template in ARTIFACT_TEMPLATES:
                src = source_dir / template.format(id=source_id)
                dst = map_dir / template.format(id=target_id)
                link_or_copy(src, dst, args.copy)

        index = {
            "format": "data7_8_flat_map_layout_v1",
            "source": str(source),
            "link_mode": "copy" if args.copy else "hardlink",
            "episode_count": len(rows),
            "map_count": len(grouped),
            "episodes": rows,
        }
        (destination / "source_episode_index.json").write_text(
            json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except Exception:
        print("[error] output may be partial; remove only the destination and rerun.", file=sys.stderr)
        raise

    print(f"[done] destination={destination} maps={len(grouped)} episodes={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
