#!/usr/bin/env python3
"""Precompute compact top-K person proposals for DT frames.

DT and AT share the same rendered frames, so only ``frames/dt`` is processed;
the training loader maps logical AT paths to this cache.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

from offline_detection_segmentation.models import YOLOInstanceSegmenter


SCHEMA = "person_candidates.topk.v1"


def output_path(output_root: Path, relative: Path) -> Path:
    return output_root / relative.parent / f"{relative.stem}.candidates.npz"


def write_atomic(path: Path, boxes: np.ndarray, scores: np.ndarray, top_k: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            schema_version=np.asarray(SCHEMA),
            top_k=np.asarray(top_k, dtype=np.int16),
            boxes_cxcywh_norm=np.asarray(boxes, dtype=np.float16),
            scores=np.asarray(scores, dtype=np.float16),
        )
    os.replace(temporary, path)


def format_seconds(value: float) -> str:
    value = max(0, int(value)); hours, rem = divmod(value, 3600); minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def iter_records(record_list: Path | None, frame_root: Path, shard_index: int, num_shards: int):
    if record_list is not None:
        with record_list.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                value = line.strip()
                if value and index % num_shards == shard_index:
                    yield Path(value)
        return
    index = 0
    for directory, _, names in os.walk(frame_root):
        for name in sorted(names):
            if Path(name).suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            if index % num_shards == shard_index:
                yield Path(directory) / name
            index += 1


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--record-list", type=Path)
    p.add_argument("--frame-root", type=Path, default=Path("/data/yh/data/processed/frames/dt"))
    p.add_argument("--data-root", type=Path, default=Path("/data/yh/data/processed"))
    p.add_argument("--output-root", type=Path, default=Path("/data/yh/data/processed/perception_cache"))
    p.add_argument("--weights", default="/data/yh/newtrackvla修改/newtrackvla_base_yh_clean/repo/offline_detection_segmentation/weights/yolo11m-seg.pt")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--image-size", type=int, default=640)
    p.add_argument("--confidence", type=float, default=0.15)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    if not 0 <= args.shard_index < args.num_shards: raise ValueError("invalid shard")
    if args.limit > 0:
        total = args.limit
    else:
        total = sum(1 for _ in iter_records(args.record_list, args.frame_root, args.shard_index, args.num_shards))
    detector = YOLOInstanceSegmenter(
        args.weights, device=args.device, image_size=args.image_size,
        confidence=args.confidence, half=True,
    )
    started=time.time(); processed=skipped=written=0; paths=[]; images=[]
    def flush():
        nonlocal processed,written,paths,images
        if not paths: return
        predictions=detector.predict_person_candidates(images, top_k=args.top_k)
        for path,(boxes,scores) in zip(paths,predictions):
            relative=path.resolve().relative_to(args.data_root.resolve())
            write_atomic(output_path(args.output_root,relative),boxes,scores,args.top_k); written+=1; processed+=1
        for image in images: image.close()
        paths=[]; images=[]
        elapsed=time.time()-started; eta=elapsed/max(1,processed+skipped)*max(0,total-processed-skipped)
        print(f"[candidates] shard={args.shard_index}/{args.num_shards} {processed+skipped}/{total} written={written} skipped={skipped} elapsed={format_seconds(elapsed)} eta={format_seconds(eta)}",flush=True)
    for source in iter_records(args.record_list,args.frame_root,args.shard_index,args.num_shards):
        if processed+skipped+len(paths)>=total: break
        path=source if source.is_absolute() else args.data_root/source
        relative=path.resolve().relative_to(args.data_root.resolve())
        target=output_path(args.output_root,relative)
        if target.is_file(): skipped+=1; continue
        paths.append(path); images.append(Image.open(path).convert("RGB"))
        if len(paths)>=args.batch_size: flush()
    flush()
    print(f"[done] shard={args.shard_index}/{args.num_shards} total={total} written={written} skipped={skipped}",flush=True)
    return 0


if __name__ == "__main__": raise SystemExit(main())
