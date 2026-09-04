#!/usr/bin/env python3
"""Fill missing raw target boxes in quarantined videos with a fine-tuned YOLO model.

The source tree is never modified. Output metadata mirrors the source layout and
videos are symlinked, so the result is independently consumable without copying
hundreds of GB of encoded video.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2


def bbox_valid(value: Any) -> bool:
    try:
        if not isinstance(value, (list, tuple)) or len(value) < 4:
            return False
        values = [float(item) for item in value[:4]]
        return all(math.isfinite(item) for item in values) and values[2] > 0.0 and values[3] > 0.0
    except (TypeError, ValueError, OverflowError):
        return False


def episode_pairs(root: Path) -> list[tuple[Path, str]]:
    pairs = []
    for drone_info in sorted(root.rglob("*_drone_info.json"), key=lambda path: str(path)):
        episode = drone_info.name[: -len("_drone_info.json")]
        dog_info = drone_info.with_name(f"{episode}_robotdog_info.json")
        if dog_info.is_file():
            pairs.append((drone_info.parent, episode))
    return pairs


def link_or_copy_episode_files(source_dir: Path, output_dir: Path, episode: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for source in sorted(source_dir.glob(f"{episode}_*")):
        if source.name.endswith("_info.json"):
            continue
        destination = output_dir / source.name
        if destination.exists() or destination.is_symlink():
            continue
        destination.symlink_to(source.resolve())
    status = source_dir / f"{episode}.json"
    destination = output_dir / status.name
    if status.is_file() and not destination.exists() and not destination.is_symlink():
        destination.symlink_to(status.resolve())


def fill_agent(
    model: Any,
    info_path: Path,
    video_path: Path,
    output_path: Path,
    *,
    device: str,
    imgsz: int,
    confidence: float,
    batch_size: int,
) -> Counter[str]:
    rows = json.loads(info_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise TypeError(f"expected list in {info_path}")
    counts: Counter[str] = Counter()
    invalid_indices = [index for index, row in enumerate(rows) if not bbox_valid(row.get("target_bbox"))]
    counts["frames"] = len(rows)
    counts["original_gt"] = len(rows) - len(invalid_indices)
    counts["requested"] = len(invalid_indices)

    for row in rows:
        if bbox_valid(row.get("target_bbox")):
            row["bbox_valid_mask"] = True
            row.setdefault("bbox_source", "exact_gt")
            row.setdefault("bbox_confidence", 1.0)
        else:
            row["bbox_valid_mask"] = False
            row["bbox_source"] = "missing"
            row["bbox_confidence"] = 0.0

    if invalid_indices:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"cannot open video: {video_path}")
        wanted = set(invalid_indices)
        images: list[Any] = []
        indices: list[int] = []

        def infer_batch() -> None:
            if not images:
                return
            results = model.predict(
                images,
                device=device,
                imgsz=imgsz,
                conf=confidence,
                classes=[0],
                max_det=20,
                half=True,
                verbose=False,
            )
            for frame_index, result in zip(indices, results):
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    counts["not_detected"] += 1
                    continue
                scores = boxes.conf.detach().cpu()
                selected = int(scores.argmax().item())
                x1, y1, x2, y2 = [float(value) for value in boxes.xyxy[selected].detach().cpu().tolist()]
                height, width = result.orig_shape
                x1 = min(max(x1, 0.0), float(width))
                y1 = min(max(y1, 0.0), float(height))
                x2 = min(max(x2, 0.0), float(width))
                y2 = min(max(y2, 0.0), float(height))
                replacement = [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]
                if not bbox_valid(replacement):
                    counts["not_detected"] += 1
                    continue
                row = rows[frame_index]
                row["target_bbox"] = replacement
                row["bbox_valid_mask"] = True
                row["bbox_source"] = "yolo_finetuned"
                row["bbox_confidence"] = float(scores[selected].item())
                row["bbox_model"] = str(model.ckpt_path)
                row["bbox_selection"] = "highest_confidence_person"
                counts["filled"] += 1
            images.clear()
            indices.clear()

        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index in wanted:
                images.append(frame)
                indices.append(frame_index)
                if len(images) >= batch_size:
                    infer_batch()
            frame_index += 1
        infer_batch()
        capture.release()
        counts["decoded_frames"] = frame_index
        if frame_index < len(rows):
            counts["metadata_beyond_video"] = len(rows) - frame_index
            counts["not_detected"] += sum(index >= frame_index for index in invalid_indices)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    from ultralytics import YOLO

    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    episodes = episode_pairs(input_root)
    episodes = [item for index, item in enumerate(episodes) if index % args.num_shards == args.shard_index]
    if args.max_episodes:
        episodes = episodes[: args.max_episodes]
    model = YOLO(str(args.weights.resolve()))
    totals: Counter[str] = Counter()
    completed = 0
    for source_dir, episode in episodes:
        relative_dir = source_dir.relative_to(input_root)
        output_dir = output_root / relative_dir
        outputs = [output_dir / f"{episode}_{agent}_info.json" for agent in ("drone", "robotdog")]
        if not args.overwrite and all(path.is_file() for path in outputs):
            totals["episodes_skipped_existing"] += 1
            continue
        link_or_copy_episode_files(source_dir, output_dir, episode)
        episode_counts: Counter[str] = Counter()
        for agent, output_path in zip(("drone", "robotdog"), outputs):
            counts = fill_agent(
                model,
                source_dir / f"{episode}_{agent}_info.json",
                source_dir / f"{episode}_{agent}.mp4",
                output_path,
                device=args.device,
                imgsz=args.imgsz,
                confidence=args.conf,
                batch_size=args.batch,
            )
            episode_counts.update(counts)
        totals.update(episode_counts)
        completed += 1
        totals["episodes_completed"] += 1
        print(
            f"[shard {args.shard_index}] {completed}/{len(episodes)} {relative_dir}/{episode} "
            f"requested={episode_counts['requested']} filled={episode_counts['filled']} "
            f"missing={episode_counts['not_detected']}",
            flush=True,
        )

    output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "created_at": datetime.now().isoformat(),
        "input_root": str(input_root),
        "output_root": str(output_root),
        "weights": str(args.weights.resolve()),
        "device": args.device,
        "imgsz": args.imgsz,
        "confidence": args.conf,
        "bbox_selection": "highest_confidence_person",
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        **dict(totals),
    }
    summary_path = output_root / f"summary_shard_{args.shard_index:02d}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
