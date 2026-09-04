#!/usr/bin/env python3
"""Create a non-duplicating original-data reference tree beside eval outputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable


EPISODE_SUFFIXES = (
    ".json",
    "_drone.mp4",
    "_robotdog.mp4",
    "_drone_info.json",
    "_robotdog_info.json",
)


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"manifest is not a JSON object: {path}")
    return payload


def checked_key(item: dict[str, Any]) -> Path:
    raw = str(item.get("key", ""))
    path = Path(raw)
    if not raw or path.is_absolute() or ".." in path.parts or len(path.parts) < 2:
        raise ValueError(f"unsafe or invalid manifest key: {raw!r}")
    return path


def link_exact(source: Path, target: Path) -> None:
    if target.is_symlink():
        if target.resolve(strict=False) == source.resolve(strict=False):
            return
        raise FileExistsError(f"existing link points elsewhere: {target} -> {target.readlink()}")
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing path: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(source, target)


def episode_items(manifest: dict[str, Any], split: str) -> list[dict[str, Any]]:
    value = manifest.get(split, [])
    if not isinstance(value, list):
        raise ValueError(f"manifest split {split!r} is not a list")
    return [item for item in value if isinstance(item, dict)]


def write_index(path: Path, rows: Iterable[tuple[str, str, Path]]) -> None:
    if path.exists() and not path.is_file():
        raise FileExistsError(f"index path is not a regular file: {path}")
    content = "split\tkey\toriginal_episode_dir\n" + "".join(
        f"{split}\t{key}\t{source}\n" for split, key, source in rows
    )
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Full 918-episode split manifest.")
    parser.add_argument("--eval100-manifest", type=Path, required=True, help="Current 100-episode eval manifest.")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    full = load_manifest(args.manifest)
    eval100 = load_manifest(args.eval100_manifest)
    source_root = Path(str(full.get("input_root", "")))
    if not source_root.is_dir():
        raise FileNotFoundError(f"manifest input_root does not exist: {source_root}")

    train = episode_items(full, "train")
    test = episode_items(full, "test")
    eval100_test = episode_items(eval100, "test")
    all_rows = [("train", item) for item in train] + [("test", item) for item in test]
    keys = [str(checked_key(item)) for _, item in all_rows]
    if len(keys) != len(set(keys)):
        raise ValueError("full manifest contains duplicate episode keys")
    known_keys = set(keys)
    missing_eval100 = [str(checked_key(item)) for item in eval100_test if str(checked_key(item)) not in known_keys]
    if missing_eval100:
        raise ValueError(f"eval100 manifest has keys absent from full manifest: {missing_eval100[:3]}")

    root = args.output_root
    all_root = root / "all_918"
    entries: dict[str, Path] = {}
    missing_files: list[Path] = []
    index_rows: list[tuple[str, str, Path]] = []
    for split, item in all_rows:
        key = checked_key(item)
        stem = str(item.get("stem", key.name))
        source_dir = source_root / str(item.get("relative_dir", key.parent))
        target_dir = all_root / key
        entries[str(key)] = target_dir
        index_rows.append((split, str(key), source_dir))
        for suffix in EPISODE_SUFFIXES:
            source = source_dir / f"{stem}{suffix}"
            if not source.exists():
                missing_files.append(source)
                continue
            link_exact(source, target_dir / source.name)

    if missing_files:
        preview = "\n".join(str(path) for path in missing_files[:20])
        raise FileNotFoundError(f"missing {len(missing_files)} expected source files; first entries:\n{preview}")

    for name, items in (("train_493", train), ("test_425", test), ("eval100", eval100_test)):
        for item in items:
            key = checked_key(item)
            link_exact(entries[str(key)], root / name / key)

    root.mkdir(parents=True, exist_ok=True)
    link_exact(args.manifest, root / "split_manifest_918.json")
    link_exact(args.eval100_manifest, root / "split_manifest_100.json")
    write_index(root / "index.tsv", index_rows)
    readme = root / "README.md"
    readme.write_text(
        "# Original data7_8 reference\n\n"
        f"- `all_918`: all original episodes in the full manifest ({len(all_rows)}).\n"
        f"- `train_493`: training split aliases ({len(train)}).\n"
        f"- `test_425`: held-out test split aliases ({len(test)}).\n"
        f"- `eval100`: exact test subset currently being evaluated ({len(eval100_test)}).\n\n"
        "Each episode directory contains symbolic links to the original `drone.mp4`, "
        "`robotdog.mp4`, per-agent info JSON, and episode JSON. No video data is copied.\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_root": str(root),
                "all_918": len(all_rows),
                "train_493": len(train),
                "test_425": len(test),
                "eval100": len(eval100_test),
                "files_per_episode": len(EPISODE_SUFFIXES),
                "copied_video_bytes": 0,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
