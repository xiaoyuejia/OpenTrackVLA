"""Build the offline per-frame detection + segmentation cache."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Set

import yaml
import numpy as np
from PIL import Image

from .core import SCHEMA_VERSION, cache_path_for_image, write_cache
from .rendering import save_visualization, visualization_path_for_image


@dataclass
class FrameRecord:
    relative_path: Path
    source_path: Path
    splits: Set[str] = field(default_factory=set)
    agents: Set[str] = field(default_factory=set)


def _path_values(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _path_values(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _path_values(item)


def _safe_relative_path(raw_path: str, split_root: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        try:
            path = path.relative_to(split_root.resolve())
        except ValueError:
            # Absolute paths outside the split are uncommon; isolate them by basename.
            path = Path("external") / path.name
    if ".." in path.parts:
        raise ValueError(f"unsafe dataset path: {raw_path}")
    return path


def discover_frames(
    dataset_root: Path,
    splits: Sequence[str],
    *,
    include_history: bool,
    max_records: Optional[int] = None,
    scenes_per_split: Optional[int] = None,
    episodes_per_scene: Optional[int] = None,
    rows_per_episode: Optional[int] = None,
) -> List[FrameRecord]:
    """Read RGB paths from JSONL only. Existing GT boxes are intentionally ignored."""

    if max_records is not None and max_records <= 0:
        return []

    records: MutableMapping[str, FrameRecord] = {}
    requested_suffixes = ["current"]
    if include_history:
        requested_suffixes.append("history")

    for split in splits:
        split_root = dataset_root / split
        jsonl_root = split_root / "jsonl"
        if not jsonl_root.is_dir():
            raise FileNotFoundError(f"missing JSONL directory: {jsonl_root}")

        jsonl_files = sorted(jsonl_root.rglob("*.jsonl"))
        if not jsonl_files:
            raise FileNotFoundError(f"no .jsonl files found under {jsonl_root}")

        if scenes_per_split is not None:
            grouped: Dict[str, List[Path]] = {}
            for jsonl_path in jsonl_files:
                relative_parts = jsonl_path.relative_to(jsonl_root).parts
                scene_name = relative_parts[0] if len(relative_parts) > 1 else jsonl_path.parent.name
                grouped.setdefault(scene_name, []).append(jsonl_path)

            scene_names = _evenly_spaced(sorted(grouped), scenes_per_split)
            selected_files: List[Path] = []
            episode_limit = episodes_per_scene if episodes_per_scene is not None else 1
            for scene_name in scene_names:
                selected_files.extend(
                    _evenly_spaced(sorted(grouped[scene_name]), episode_limit)
                )
            jsonl_files = selected_files

        for jsonl_path in jsonl_files:
            with jsonl_path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if rows_per_episode is not None and line_number > rows_per_episode:
                        break
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(f"invalid JSON at {jsonl_path}:{line_number}") from error

                    for agent in ("agent1", "agent2"):
                        for suffix in requested_suffixes:
                            field_name = f"{agent}_{suffix}"
                            for raw_path in _path_values(row.get(field_name)):
                                relative_path = _safe_relative_path(raw_path, split_root)
                                source_path = Path(raw_path)
                                if not source_path.is_absolute():
                                    source_path = split_root / source_path
                                key = relative_path.as_posix()
                                if key not in records:
                                    records[key] = FrameRecord(
                                        relative_path=relative_path,
                                        source_path=source_path,
                                    )
                                records[key].splits.add(str(split))
                                records[key].agents.add(agent)
                                if max_records is not None and len(records) >= max_records:
                                    return [records[key] for key in sorted(records)]

    return [records[key] for key in sorted(records)]


def discover_frame_list(
    dataset_root: Path,
    record_list: Path,
    *,
    max_records: Optional[int] = None,
    num_shards: int = 1,
    shard_index: int = 0,
) -> List[FrameRecord]:
    """Read a prebuilt newline-delimited JPEG list without reparsing JSONL."""

    if num_shards <= 0 or shard_index < 0 or shard_index >= num_shards:
        raise ValueError("invalid record-list shard parameters")
    records: List[FrameRecord] = []
    ordinal = 0
    with record_list.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            value = raw_line.strip()
            if not value:
                continue
            if ordinal % num_shards != shard_index:
                ordinal += 1
                continue
            ordinal += 1
            relative = Path(value)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe record-list path: {value}")
            records.append(FrameRecord(relative_path=relative, source_path=dataset_root / relative))
            if max_records is not None and len(records) >= max_records:
                break
    return records


def _evenly_spaced(values: Sequence[Any], limit: int) -> List[Any]:
    """Select deterministic values across the full sorted range, not just its head."""

    if limit <= 0:
        raise ValueError("sampling limits must be positive")
    values = list(values)
    if limit >= len(values):
        return values
    if limit == 1:
        return [values[0]]
    indices = np.linspace(0, len(values) - 1, limit)
    return [values[int(round(index))] for index in indices]


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"config must contain a YAML mapping: {path}")
    return config


def _parse_args() -> argparse.Namespace:
    package_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Offline YOLO-only target and foreground-obstacle cache builder."
    )
    parser.add_argument("--config", type=Path, default=package_root / "config.yaml")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--splits", nargs="+")
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="split the deterministic frame list into this many independent workers",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="zero-based worker index in [0, num-shards)",
    )
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--scenes-per-split",
        type=int,
        help="deterministically sample this many distinct scene directories per split",
    )
    parser.add_argument(
        "--episodes-per-scene",
        type=int,
        help="episodes sampled from each selected scene (default: 1)",
    )
    parser.add_argument(
        "--rows-per-episode",
        type=int,
        help="JSONL time steps read from each sampled episode",
    )
    parser.add_argument(
        "--record-list",
        type=Path,
        help="prebuilt newline-delimited dataset-relative JPEG list (skips JSONL parsing)",
    )
    parser.add_argument("--include-history", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-half", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="discover paths and print the plan without loading models or writing files",
    )
    return parser.parse_args()


def _apply_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    config.setdefault("runtime", {})
    config.setdefault("dataset", {})
    if args.dataset_root is not None:
        config["dataset"]["root"] = str(args.dataset_root)
    if args.output_root is not None:
        config["dataset"]["output_root"] = str(args.output_root)
    if args.splits is not None:
        config["dataset"]["splits"] = list(args.splits)
    if args.device is not None:
        config["runtime"]["device"] = args.device
    if args.batch_size is not None:
        config["runtime"]["batch_size"] = args.batch_size
    if args.no_half:
        config["runtime"]["half"] = False
    return config


def _chunks(values: Sequence[FrameRecord], size: int) -> Iterable[Sequence[FrameRecord]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _select_shard(
    records: Sequence[FrameRecord], *, num_shards: int, shard_index: int
) -> List[FrameRecord]:
    """Assign every deterministic record to exactly one independent worker."""

    if num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")
    return list(records[shard_index::num_shards])


def _cache_is_current(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as cache:
            return str(cache["schema_version"].item()) == SCHEMA_VERSION
    except (OSError, KeyError, ValueError):
        return False


def _is_unrecoverable_cuda_error(error: BaseException) -> bool:
    """Return true when the CUDA context must be recreated before more inference."""

    message = str(error).lower()
    fatal_markers = (
        "launch timed out",
        "device-side assert",
        "illegal memory access",
        "unspecified launch failure",
        "context is destroyed",
    )
    return "cuda" in message and any(marker in message for marker in fatal_markers)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(dict(value), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary_path, path)


def _process_batch(
    batch: Sequence[FrameRecord],
    *,
    pipeline: Any,
    output_root: Path,
    config: Mapping[str, Any],
    overwrite: bool,
) -> Dict[str, int]:
    save_previews = bool(config["output"].get("save_visualizations", True))
    pending = [
        record
        for record in batch
        if overwrite
        or not _cache_is_current(cache_path_for_image(output_root, record.relative_path))
        or (
            save_previews
            and not visualization_path_for_image(output_root, record.relative_path).is_file()
        )
    ]
    if not pending:
        return {
            "cache_written": 0,
            "visualizations_written": 0,
            "skipped": len(batch),
            "person_found": 0,
        }

    images: List[Image.Image] = []
    for record in pending:
        with Image.open(record.source_path) as image:
            images.append(image.convert("RGB"))

    predictions = pipeline.predict(images)
    if len(predictions) != len(pending):
        raise RuntimeError("pipeline output length does not match pending frame count")

    grid_size_raw = config["output"]["grid_size"]
    grid_size = (int(grid_size_raw[0]), int(grid_size_raw[1]))
    person_found = 0
    cache_written = 0
    visualizations_written = 0
    for record, image, prediction in zip(pending, images, predictions):
        cache_path = cache_path_for_image(output_root, record.relative_path)
        replace_cache = overwrite or not _cache_is_current(cache_path)
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "image_relative_path": record.relative_path.as_posix(),
            "source_path": str(record.source_path),
            "splits": sorted(record.splits),
            "agents": sorted(record.agents),
            "image_width": image.width,
            "image_height": image.height,
            "models": dict(config["models"]),
            "thresholds": dict(config["thresholds"]),
            "rgb_only": True,
            "uses_ground_truth_box": False,
        }
        if replace_cache:
            write_cache(cache_path, prediction, grid_size=grid_size, metadata=metadata)
            cache_written += 1
        if save_previews:
            visualization_path = visualization_path_for_image(output_root, record.relative_path)
            if overwrite or replace_cache or not visualization_path.is_file():
                save_visualization(
                    visualization_path,
                    image,
                    prediction.scene_mask,
                    person_valid=prediction.person_valid,
                    person_box_xyxy=prediction.person_box_xyxy,
                    person_score=prediction.person_score,
                    alpha=float(config["output"].get("visualization_alpha", 0.45)),
                    show_unknown=bool(config["output"].get("visualize_unknown", False)),
                    jpeg_quality=int(config["output"].get("visualization_jpeg_quality", 90)),
                )
                visualizations_written += 1
        person_found += int(prediction.person_valid)
    return {
        "cache_written": cache_written,
        "visualizations_written": visualizations_written,
        "skipped": len(batch) - len(pending),
        "person_found": person_found,
    }


def main() -> int:
    args = _parse_args()
    config = _apply_overrides(_load_config(args.config), args)
    dataset_root = Path(config["dataset"]["root"]).expanduser()
    output_root = Path(config["dataset"]["output_root"]).expanduser()
    splits = list(config["dataset"].get("splits", ["train", "val"]))

    print(f"[discover] dataset={dataset_root}")
    if args.record_list is not None:
        records = discover_frame_list(
            dataset_root,
            args.record_list,
            max_records=args.max_frames,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
        )
        print(f"[discover] source=record-list {args.record_list}")
    else:
        records = discover_frames(
            dataset_root,
            splits,
            include_history=bool(args.include_history),
            max_records=args.max_frames,
            scenes_per_split=args.scenes_per_split,
            episodes_per_scene=args.episodes_per_scene,
            rows_per_episode=args.rows_per_episode,
        )
    if args.max_frames is not None:
        if args.max_frames < 0:
            raise ValueError("--max-frames must be non-negative")
        records = records[: args.max_frames]
    total_records = len(records)
    if args.record_list is None:
        records = _select_shard(
            records,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
        )
    print(f"[discover] unique_frames={total_records:,} splits={','.join(splits)}")
    if args.num_shards > 1:
        print(
            f"[shard] index={args.shard_index}/{args.num_shards} "
            f"frames={len(records):,}"
        )
    print(f"[output] {output_root}")
    for record in records[:3]:
        print(f"[sample] {record.relative_path} -> {record.source_path}")

    if args.dry_run:
        print("[dry-run] model loading and cache writes were skipped")
        return 0
    if not records:
        print("[done] no frames selected")
        return 0

    missing = [record.source_path for record in records if not record.source_path.is_file()]
    if missing:
        preview = "\n".join(str(path) for path in missing[:10])
        raise FileNotFoundError(f"{len(missing)} source images are missing; first paths:\n{preview}")

    # Delayed import keeps discovery/dry-run usable before model dependencies are installed.
    from .models import OfflinePerceptionPipeline

    print("[models] loading YOLO instance segmentation")
    pipeline = OfflinePerceptionPipeline(config)
    batch_size = int(config["runtime"].get("batch_size", 4))
    if batch_size <= 0:
        raise ValueError("runtime.batch_size must be positive")

    output_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    stats = {
        "cache_written": 0,
        "visualizations_written": 0,
        "skipped": 0,
        "person_found": 0,
        "failed": 0,
    }
    failures: List[Dict[str, str]] = []
    processed = 0
    for batch in _chunks(records, batch_size):
        try:
            result = _process_batch(
                batch,
                pipeline=pipeline,
                output_root=output_root,
                config=config,
                overwrite=bool(args.overwrite),
            )
            for key, value in result.items():
                stats[key] += value
        except Exception as batch_error:
            if _is_unrecoverable_cuda_error(batch_error):
                raise RuntimeError(
                    "unrecoverable CUDA context error; restart this shard with a smaller batch"
                ) from batch_error
            # Retry one-by-one so a corrupt frame cannot discard a full expensive batch.
            if len(batch) == 1:
                record = batch[0]
                stats["failed"] += 1
                failures.append(
                    {"image": record.relative_path.as_posix(), "error": repr(batch_error)}
                )
                print(f"[error] {record.relative_path}: {batch_error}", file=sys.stderr)
            else:
                print(f"[warn] batch failed, retrying individually: {batch_error}", file=sys.stderr)
                for record in batch:
                    try:
                        result = _process_batch(
                            [record],
                            pipeline=pipeline,
                            output_root=output_root,
                            config=config,
                            overwrite=bool(args.overwrite),
                        )
                        for key, value in result.items():
                            stats[key] += value
                    except Exception as frame_error:
                        stats["failed"] += 1
                        failures.append(
                            {"image": record.relative_path.as_posix(), "error": repr(frame_error)}
                        )
                        print(f"[error] {record.relative_path}: {frame_error}", file=sys.stderr)

        processed += len(batch)
        if processed == len(records) or processed % max(batch_size * 10, 10) == 0:
            elapsed = max(time.time() - started, 1e-6)
            print(
                f"[progress] {processed:,}/{len(records):,} "
                f"cache={stats['cache_written']:,} "
                f"visualizations={stats['visualizations_written']:,} "
                f"skipped={stats['skipped']:,} "
                f"failed={stats['failed']:,} rate={processed / elapsed:.2f} frames/s"
            )

    elapsed = time.time() - started
    summary: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "splits": splits,
        "selected_frames": len(records),
        "total_selected_frames": total_records,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "include_history": bool(args.include_history),
        "sampling": {
            "scenes_per_split": args.scenes_per_split,
            "episodes_per_scene": args.episodes_per_scene,
            "rows_per_episode": args.rows_per_episode,
        },
        "elapsed_seconds": elapsed,
        **stats,
    }
    if args.num_shards == 1:
        summary_path = output_root / "summary.json"
        failures_path = output_root / "failures.jsonl"
    else:
        shard_suffix = f"shard-{args.shard_index:05d}-of-{args.num_shards:05d}"
        summary_path = output_root / f"summary.{shard_suffix}.json"
        failures_path = output_root / f"failures.{shard_suffix}.jsonl"
    _write_json(summary_path, summary)
    if failures:
        with failures_path.open("w", encoding="utf-8") as handle:
            for failure in failures:
                handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
    elif failures_path.exists():
        # Do not leave a stale failure report after a clean overwrite/retry run.
        failures_path.unlink()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
