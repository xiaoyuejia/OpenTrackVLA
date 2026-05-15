#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from make_tracking_data import (
    collect_episode_pairs,
    load_episode_info,
    load_episode_status,
    should_keep_episode,
)


@dataclass
class EpisodeSummary:
    stem: str
    run_dir: Path
    status: Dict
    steps_len: int
    frame_count: Optional[int]
    sample_count_est: int


@dataclass
class AggregateSummary:
    episode_count: int
    step_record_count: int
    sample_count_est: int
    metric_means: Dict[str, float]


METRIC_KEYS = [
    "success",
    "following_rate",
    "following_step",
    "total_step",
    "collision",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Summarize raw/filtered STT training data and estimate training time."
    )
    ap.add_argument(
        "--input_root",
        type=str,
        required=True,
        help="Raw collection root, e.g. sim_data/stt_train",
    )
    ap.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Processed dataset root, e.g. data/sample. If present, counts actual jsonl samples when available.",
    )
    ap.add_argument("--only_success", action="store_true", help="Apply only_success filter")
    ap.add_argument(
        "--min_following_rate",
        type=float,
        default=0.0,
        help="Minimum following rate for filtered statistics",
    )
    ap.add_argument(
        "--exclude_collision",
        action="store_true",
        help="Exclude collision episodes from filtered statistics",
    )
    ap.add_argument(
        "--min_total_steps",
        type=int,
        default=0,
        help="Minimum total episode steps for filtered statistics",
    )
    ap.add_argument("--history", type=int, default=31, help="For reporting only")
    ap.add_argument("--horizon", type=int, default=8, help="Future horizon used to estimate sample count")
    ap.add_argument("--epochs", type=int, default=2, help="Planned training epochs")
    ap.add_argument("--batch_size", type=int, default=8, help="Per-device batch size")
    ap.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs/devices used for training")
    ap.add_argument(
        "--seconds_per_step",
        type=float,
        default=None,
        help="Manual estimate of one optimizer step time in seconds",
    )
    ap.add_argument(
        "--samples_per_second",
        type=float,
        default=None,
        help="Manual throughput estimate in samples/sec across all devices",
    )
    ap.add_argument(
        "--train_log_csv",
        type=str,
        default=None,
        help="Optional training CSV log to infer throughput from a previous run",
    )
    return ap.parse_args()


def safe_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def count_lines(path: Path) -> int:
    total = 0
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                total += 1
    return total


def maybe_count_frames(output_root: Optional[Path], seed_name: str, run_name: str, stem: str) -> Optional[int]:
    if output_root is None:
        return None
    frame_dir = output_root / "frames" / seed_name / run_name / stem
    if not frame_dir.exists():
        return None
    return len(list(frame_dir.glob("*.jpg")))


def estimate_sample_count(steps_len: int, horizon: int, frame_count: Optional[int]) -> int:
    usable_frames = frame_count if frame_count is not None else steps_len
    usable = min(steps_len, usable_frames)
    return max(0, usable - horizon)


def build_episode_summaries(
    input_root: Path,
    output_root: Optional[Path],
    horizon: int,
) -> Tuple[List[EpisodeSummary], int]:
    episodes = collect_episode_pairs(input_root)
    summaries: List[EpisodeSummary] = []
    missing_status = 0

    for ep in episodes:
        status_path = ep.run_dir / f"{ep.stem}.json"
        status = load_episode_status(status_path)
        if not status:
            missing_status += 1
            continue
        try:
            steps = load_episode_info(ep.info_json)
        except Exception:
            steps = []
        frame_count = maybe_count_frames(output_root, ep.seed_dir.name, ep.run_dir.name, ep.stem)
        summaries.append(
            EpisodeSummary(
                stem=ep.stem,
                run_dir=ep.run_dir,
                status=status,
                steps_len=len(steps),
                frame_count=frame_count,
                sample_count_est=estimate_sample_count(len(steps), horizon, frame_count),
            )
        )
    return summaries, missing_status


def aggregate_episode_summaries(items: List[EpisodeSummary]) -> AggregateSummary:
    metric_totals = {k: 0.0 for k in METRIC_KEYS}
    total_steps = 0
    total_samples = 0

    for item in items:
        total_steps += item.steps_len
        total_samples += item.sample_count_est
        for key in METRIC_KEYS:
            metric_totals[key] += safe_float(item.status.get(key, 0.0))

    count = len(items)
    means = {k: (metric_totals[k] / count if count > 0 else 0.0) for k in METRIC_KEYS}
    return AggregateSummary(
        episode_count=count,
        step_record_count=total_steps,
        sample_count_est=total_samples,
        metric_means=means,
    )


def filter_episode_summaries(
    items: List[EpisodeSummary],
    only_success: bool,
    min_following_rate: float,
    exclude_collision: bool,
    min_total_steps: int,
) -> List[EpisodeSummary]:
    kept: List[EpisodeSummary] = []
    for item in items:
        if should_keep_episode(
            item.run_dir,
            item.stem,
            only_success=only_success,
            min_following_rate=min_following_rate,
            exclude_collision=exclude_collision,
            min_total_steps=min_total_steps,
        ):
            kept.append(item)
    return kept


def count_actual_jsonl_samples(output_root: Optional[Path]) -> Optional[int]:
    if output_root is None:
        return None
    jsonl_root = output_root / "jsonl"
    if not jsonl_root.exists():
        return None
    total = 0
    found = False
    for fp in jsonl_root.rglob("*.jsonl"):
        found = True
        total += count_lines(fp)
    return total if found else None


def infer_seconds_per_step_from_csv(
    csv_path: Optional[Path],
    effective_batch_size: int,
) -> Tuple[Optional[float], Optional[str]]:
    if csv_path is None or not csv_path.exists():
        return None, None

    with open(csv_path, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None, None

    direct_keys = [
        "step_time_sec",
        "iter_time_sec",
        "seconds_per_step",
        "sec_per_step",
        "step_time",
        "iter_time",
    ]
    throughput_keys = [
        "steps_per_sec",
        "iters_per_sec",
        "samples_per_sec",
    ]

    for key in direct_keys:
        vals = [safe_float(r.get(key, None), default=-1.0) for r in rows]
        vals = [v for v in vals if v > 0]
        if vals:
            return sum(vals[-min(20, len(vals)):]) / min(20, len(vals)), f"csv:{key}"

    for key in throughput_keys:
        vals = [safe_float(r.get(key, None), default=-1.0) for r in rows]
        vals = [v for v in vals if v > 0]
        if not vals:
            continue
        avg = sum(vals[-min(20, len(vals)):]) / min(20, len(vals))
        if key == "samples_per_sec":
            if effective_batch_size <= 0:
                return None, None
            return effective_batch_size / avg, f"csv:{key}"
        return 1.0 / avg, f"csv:{key}"

    return None, None


def format_hours(seconds: float) -> str:
    mins = seconds / 60.0
    hours = mins / 60.0
    return f"{seconds:.1f}s ({mins:.1f}m / {hours:.2f}h)"


def print_summary_block(title: str, summary: AggregateSummary) -> None:
    print(title)
    print(f"  episodes: {summary.episode_count}")
    print(f"  step_records: {summary.step_record_count}")
    print(f"  sample_count_est: {summary.sample_count_est}")
    for key in METRIC_KEYS:
        print(f"  mean_{key}: {summary.metric_means[key]:.6f}")


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve() if args.output_root else None

    summaries, missing_status = build_episode_summaries(
        input_root=input_root,
        output_root=output_root,
        horizon=args.horizon,
    )
    raw_summary = aggregate_episode_summaries(summaries)
    filtered_items = filter_episode_summaries(
        summaries,
        only_success=args.only_success,
        min_following_rate=args.min_following_rate,
        exclude_collision=args.exclude_collision,
        min_total_steps=args.min_total_steps,
    )
    filtered_summary = aggregate_episode_summaries(filtered_items)

    actual_jsonl_samples = count_actual_jsonl_samples(output_root)
    train_samples = (
        actual_jsonl_samples
        if actual_jsonl_samples is not None
        else filtered_summary.sample_count_est
    )

    effective_batch_size = max(1, args.batch_size * args.num_gpus)
    steps_per_epoch = math.ceil(train_samples / effective_batch_size) if train_samples > 0 else 0
    total_steps = steps_per_epoch * max(1, args.epochs)

    seconds_per_step = args.seconds_per_step
    seconds_source = "arg:seconds_per_step" if seconds_per_step else None
    if seconds_per_step is None and args.samples_per_second is not None and args.samples_per_second > 0:
        seconds_per_step = effective_batch_size / args.samples_per_second
        seconds_source = "arg:samples_per_second"
    if seconds_per_step is None and args.train_log_csv:
        seconds_per_step, seconds_source = infer_seconds_per_step_from_csv(
            Path(args.train_log_csv).resolve(),
            effective_batch_size=effective_batch_size,
        )

    print("=== Dataset Summary ===")
    print(f"input_root: {input_root}")
    print(f"output_root: {output_root if output_root else '(not set)'}")
    print(f"missing_status_files: {missing_status}")
    print_summary_block("raw_collection", raw_summary)
    print_summary_block("filtered_collection", filtered_summary)

    if actual_jsonl_samples is not None:
        print(f"actual_filtered_jsonl_samples: {actual_jsonl_samples}")
    else:
        print("actual_filtered_jsonl_samples: unavailable (jsonl not found, using estimate)")

    print("\n=== Filter Settings ===")
    print(f"only_success: {args.only_success}")
    print(f"min_following_rate: {args.min_following_rate}")
    print(f"exclude_collision: {args.exclude_collision}")
    print(f"min_total_steps: {args.min_total_steps}")
    print(f"history: {args.history}")
    print(f"horizon: {args.horizon}")

    print("\n=== Training Estimate ===")
    print(f"train_samples_used: {train_samples}")
    print(f"epochs: {args.epochs}")
    print(f"batch_size_per_gpu: {args.batch_size}")
    print(f"num_gpus: {args.num_gpus}")
    print(f"effective_batch_size: {effective_batch_size}")
    print(f"steps_per_epoch: {steps_per_epoch}")
    print(f"total_steps: {total_steps}")
    if seconds_per_step is not None:
        total_seconds = total_steps * seconds_per_step
        print(f"seconds_per_step: {seconds_per_step:.4f} ({seconds_source})")
        print(f"estimated_train_time: {format_hours(total_seconds)}")
    else:
        print("seconds_per_step: unavailable")
        print("estimated_train_time: unavailable (pass --seconds_per_step, --samples_per_second, or --train_log_csv)")


if __name__ == "__main__":
    main()
