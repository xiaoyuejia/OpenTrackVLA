#!/usr/bin/env python3
"""Extract all raw data_arr videos into processed/joint/frames safely."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


# Replay evaluation is kept first so its derived evaluation/cache pipeline can
# become usable without waiting for the much larger training corpus.
SOURCES = (("at_eval_replay", "at"), ("dt", "dt"), ("stt", "stt"), ("at", "at"))


def frame_count(path: Path) -> int:
    return sum(1 for _ in path.glob("frame_*.jpg"))


def extract_one(task: tuple[Path, Path, Path, str]) -> tuple[str, int, str]:
    video, info, destination, ffmpeg = task
    expected = len(json.loads(info.read_text(encoding="utf-8")))
    marker = destination / ".complete.json"
    if marker.is_file() and frame_count(destination) == expected:
        return "skipped", expected, str(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.extract-", dir=destination.parent))
    try:
        subprocess.run(
            [ffmpeg, "-nostdin", "-loglevel", "error", "-threads", "1", "-i", str(video),
             "-q:v", "2", str(temporary / "frame_%05d.jpg")],
            check=True,
        )
        actual = frame_count(temporary)
        if actual != expected:
            raise RuntimeError(f"frame mismatch video={video} expected={expected} actual={actual}")
        (temporary / ".complete.json").write_text(
            json.dumps({"source": str(video), "frames": actual, "completed_at": time.strftime("%FT%T%z")}) + "\n",
            encoding="utf-8",
        )
        if destination.exists():
            stale = destination.with_name(f".{destination.name}.incomplete-{os.getpid()}-{int(time.time())}")
            destination.rename(stale)
        temporary.rename(destination)
        return "written", actual, str(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", type=Path, default=Path("/data/yh/data/raw"))
    p.add_argument("--output-root", type=Path, default=Path("/data/yh/data/processed/frames"))
    p.add_argument("--jobs", type=int, default=2)
    p.add_argument("--ffmpeg", default="ffmpeg")
    args = p.parse_args()
    if args.jobs < 1:
        raise SystemExit("--jobs must be positive")
    tasks = []
    destinations = set()
    for source_name, output_name in SOURCES:
        source = args.raw_root / source_name
        for video in sorted(source.rglob("*_drone.mp4")) + sorted(source.rglob("*_robotdog.mp4")):
            suffix = "_drone.mp4" if video.name.endswith("_drone.mp4") else "_robotdog.mp4"
            agent = suffix.removeprefix("_").removesuffix(".mp4")
            stem = video.name.removesuffix(suffix)
            info = video.with_name(f"{stem}_{agent}_info.json")
            if not info.is_file():
                raise FileNotFoundError(info)
            rel = video.parent.relative_to(source)
            destination = args.output_root / output_name / rel / stem / agent
            if destination in destinations:
                raise RuntimeError(f"overlapping output episode: {destination}")
            destinations.add(destination)
            tasks.append((video, info, destination, args.ffmpeg))
    print(f"[frames] videos={len(tasks)} jobs={args.jobs} output={args.output_root}", flush=True)
    counts = {"written": 0, "skipped": 0, "failed": 0}
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(extract_one, task): task[0] for task in tasks}
        for index, future in enumerate(as_completed(futures), 1):
            try:
                state, _, _ = future.result()
                counts[state] += 1
            except Exception as exc:
                counts["failed"] += 1
                print(f"[ERROR] {futures[future]}: {exc}", flush=True)
            if index == 1 or index % 20 == 0 or index == len(tasks):
                print(f"[progress] {index}/{len(tasks)} {counts}", flush=True)
    print(f"[done] {counts}", flush=True)
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
