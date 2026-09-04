#!/usr/bin/env python3
"""Save exact target-bbox crops referenced by multi-agent JSONL samples."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--raw-root", type=Path, default=None, help="Raw episode root; avoids scanning large JSONL files.")
    p.add_argument("--json-root", type=Path, default=None)
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--quality", type=int, default=95)
    return p.parse_args()


def add_ref(refs: dict[Path, list[float]], data_root: Path, current: Any, bbox: Any) -> None:
    if not isinstance(current, str) or not isinstance(bbox, list) or len(bbox) < 4:
        return
    try:
        values = [float(v) for v in bbox[:4]]
    except Exception:
        return
    if not all(0.0 <= v <= 1.0 for v in values) or values[2] <= 0.0 or values[3] <= 0.0:
        return
    path = Path(current)
    if not path.is_absolute():
        path = data_root / path
    path = path.resolve()
    previous = refs.get(path)
    if previous is not None and any(abs(a - b) > 1e-6 for a, b in zip(previous, values)):
        raise ValueError(f"conflicting bbox for {path}: {previous} vs {values}")
    refs[path] = values


def collect_refs(data_root: Path, json_root: Path) -> dict[Path, list[float]]:
    refs: dict[Path, list[float]] = {}
    files = sorted(json_root.rglob("*.jsonl")) if json_root.is_dir() else [json_root]
    for path in files:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                sample = json.loads(line)
                if not isinstance(sample, dict):
                    continue
                for agent in ("agent1", "agent2"):
                    payload = sample.get("agents", {}).get(agent) if isinstance(sample.get("agents"), dict) else None
                    if isinstance(payload, dict):
                        add_ref(refs, data_root, payload.get("current"), payload.get("bbox"))
                add_ref(refs, data_root, sample.get("agent1_current"), sample.get("agent1_bbox"))
                add_ref(refs, data_root, sample.get("agent2_current"), sample.get("agent2_bbox"))
    return refs


def crop_one(item: tuple[Path, list[float], Path, int]) -> tuple[str, bool, list[int] | None]:
    source, bbox, destination, quality = item
    try:
        with Image.open(source) as image:
            image = image.convert("RGB")
            width, height = image.size
            cx, cy, bw, bh = bbox
            x1 = max(0, min(width, int(round((cx - 0.5 * bw) * width))))
            y1 = max(0, min(height, int(round((cy - 0.5 * bh) * height))))
            x2 = max(0, min(width, int(round((cx + 0.5 * bw) * width))))
            y2 = max(0, min(height, int(round((cy + 0.5 * bh) * height))))
            if x2 <= x1 or y2 <= y1:
                return str(source), False, None
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                image.crop((x1, y1, x2, y2)).save(destination, format="JPEG", quality=quality, optimize=False)
            return str(source), True, [x1, y1, x2, y2]
    except Exception:
        return str(source), False, None


def crop_pixel_one(item: tuple[Path, list[float], Path, int]) -> tuple[bool, list[int] | None]:
    source, bbox, destination, quality = item
    try:
        with Image.open(source) as image:
            image = image.convert("RGB")
            width, height = image.size
            x, y, bw, bh = bbox
            x1 = max(0, min(width, int(x)))
            y1 = max(0, min(height, int(y)))
            x2 = max(0, min(width, int(x + bw)))
            y2 = max(0, min(height, int(y + bh)))
            if x2 <= x1 or y2 <= y1:
                return False, None
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                image.crop((x1, y1, x2, y2)).save(destination, format="JPEG", quality=quality, optimize=False)
            return True, [x1, y1, x2, y2]
    except Exception:
        return False, None


def raw_jobs(data_root: Path, raw_root: Path, output_root: Path, quality: int):
    frames_root = data_root / "frames"
    for scene_dir in sorted(p for p in frames_root.iterdir() if p.is_dir()):
        for episode_dir in sorted(p for p in scene_dir.iterdir() if p.is_dir()):
            raw_dir = raw_root / scene_dir.name
            drone_info = raw_dir / f"{episode_dir.name}_drone_info.json"
            dog_info = raw_dir / f"{episode_dir.name}_robotdog_info.json"
            if not drone_info.is_file() or not dog_info.is_file():
                continue
            try:
                drone_rows = json.loads(drone_info.read_text(encoding="utf-8"))
                dog_rows = json.loads(dog_info.read_text(encoding="utf-8"))
            except Exception:
                continue
            for agent, rows in (("drone", drone_rows), ("robotdog", dog_rows)):
                frame_dir = episode_dir / agent
                frame_paths = sorted(frame_dir.glob("frame_*.jpg"))
                for index, source in enumerate(frame_paths):
                    if index >= len(rows) or not isinstance(rows[index], dict):
                        continue
                    bbox = rows[index].get("target_bbox")
                    if not isinstance(bbox, list) or len(bbox) < 4:
                        continue
                    try:
                        pixel_bbox = [float(v) for v in bbox[:4]]
                    except Exception:
                        continue
                    if pixel_bbox[2] <= 0 or pixel_bbox[3] <= 0:
                        continue
                    destination = output_root / scene_dir.name / episode_dir.name / agent / source.name
                    yield source, pixel_bbox, destination, quality


def main() -> int:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    json_root = (args.json_root or data_root / "jsonl").expanduser().resolve()
    raw_root = args.raw_root.expanduser().resolve() if args.raw_root else None
    output_root = (args.output_root or data_root / "person_crops").expanduser().resolve()
    completed = 0
    failed = 0
    referenced = 0
    if raw_root is not None:
        # Process one episode at a time to keep memory bounded while avoiding
        # a full scan of the very large training JSONL files.
        current_jobs: list[tuple[Path, list[float], Path, int]] = []
        with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
            for job in raw_jobs(data_root, raw_root, output_root, int(args.quality)):
                current_jobs.append(job)
                if len(current_jobs) >= 2048:
                    referenced += len(current_jobs)
                    for ok, _crop in pool.map(crop_pixel_one, current_jobs):
                        completed += int(ok)
                        failed += int(not ok)
                    current_jobs.clear()
            if current_jobs:
                referenced += len(current_jobs)
                for ok, _crop in pool.map(crop_pixel_one, current_jobs):
                    completed += int(ok)
                    failed += int(not ok)
    else:
        refs = collect_refs(data_root, json_root)
        jobs: list[tuple[Path, list[float], Path, int]] = []
        for source, bbox in sorted(refs.items(), key=lambda item: str(item[0])):
            rel = source.relative_to(data_root)
            jobs.append((source, bbox, output_root / rel, int(args.quality)))
        referenced = len(jobs)
        with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
            for source, ok, crop_xyxy in pool.map(crop_one, jobs):
                completed += int(ok)
                failed += int(not ok)
    summary = {
        "data_root": str(data_root),
        "json_root": str(json_root) if raw_root is None else None,
        "raw_root": str(raw_root) if raw_root is not None else None,
        "output_root": str(output_root),
        "crop_mode": "exact_xyxy_from_cxcywh_norm",
        "expand_ratio": 1.0,
        "make_square": False,
        "referenced_frames": referenced,
        "written_or_existing": completed,
        "failed": failed,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
