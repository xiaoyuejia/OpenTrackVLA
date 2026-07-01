#!/usr/bin/env python3
"""根据 UnrealZoo ``*_info.json`` 在视频中绘制目标 bbox，生成检查视频。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render target_bbox from UnrealZoo info JSON onto videos.")
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-side-by-side", action="store_true", help="Do not generate drone/robotdog combined videos.")
    return parser.parse_args()


def load_steps(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return [step if isinstance(step, dict) else {} for step in data]


def bbox_xywh(step: dict[str, Any], width: int, height: int) -> tuple[int, int, int, int] | None:
    raw = step.get("target_bbox")
    if not isinstance(raw, list) or len(raw) < 4:
        return None
    try:
        x, y, w, h = [float(value) for value in raw[:4]]
    except (TypeError, ValueError):
        return None
    if max(abs(x), abs(y), abs(w), abs(h)) <= 1.5:
        x, y, w, h = x * width, y * height, w * width, h * height
    x1 = int(round(np.clip(x, 0, width - 1)))
    y1 = int(round(np.clip(y, 0, height - 1)))
    x2 = int(round(np.clip(x + w, 0, width - 1)))
    y2 = int(round(np.clip(y + h, 0, height - 1)))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def draw_bbox(frame: np.ndarray, step: dict[str, Any], frame_index: int, agent: str) -> np.ndarray:
    out = frame.copy()
    height, width = out.shape[:2]
    visible = bool(step.get("target_visible", False))
    bbox = bbox_xywh(step, width, height)
    color = (0, 255, 0) if visible else (0, 165, 255)
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
        cv2.putText(
            out,
            f"human bbox | visible={int(visible)}",
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        out,
        f"{agent} | frame={frame_index} | step={step.get('step', frame_index)}",
        (12, height - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def render_view(video_path: Path, info_path: Path, output_path: Path, agent: str) -> tuple[int, float, tuple[int, int]]:
    steps = load_steps(info_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 10.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Unable to create video: {output_path}")

    count = 0
    while count < len(steps):
        ok, frame = capture.read()
        if not ok:
            break
        writer.write(draw_bbox(frame, steps[count], count, agent))
        count += 1
    capture.release()
    writer.release()
    if count != len(steps):
        raise RuntimeError(f"Frame/JSON mismatch for {video_path.name}: rendered={count}, json={len(steps)}")
    return count, fps, (width, height)


def combine_views(left_path: Path, right_path: Path, output_path: Path) -> int:
    left = cv2.VideoCapture(str(left_path))
    right = cv2.VideoCapture(str(right_path))
    fps = left.get(cv2.CAP_PROP_FPS) or 10.0
    width = int(left.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(left.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width * 2, height))
    count = 0
    while True:
        ok_left, frame_left = left.read()
        ok_right, frame_right = right.read()
        if not ok_left or not ok_right:
            break
        writer.write(np.concatenate([frame_left, frame_right], axis=1))
        count += 1
    left.release()
    right.release()
    writer.release()
    return count


def main() -> int:
    args = parse_args()
    scene_dir = args.scene_dir.expanduser().resolve()
    output_dir = (args.output_dir or (scene_dir / "bbox_check")).expanduser().resolve()
    stems = sorted(path.name.removesuffix("_drone.mp4") for path in scene_dir.glob("*_drone.mp4"))
    if not stems:
        raise FileNotFoundError(f"No *_drone.mp4 files found in {scene_dir}")

    for stem in stems:
        rendered: dict[str, Path] = {}
        for agent in ("drone", "robotdog"):
            video = scene_dir / f"{stem}_{agent}.mp4"
            info = scene_dir / f"{stem}_{agent}_info.json"
            output = output_dir / f"{stem}_{agent}_bbox.mp4"
            count, fps, size = render_view(video, info, output, agent)
            rendered[agent] = output
            print(f"[OK] {output.name}: frames={count} fps={fps:g} size={size[0]}x{size[1]}")
        if not args.no_side_by_side:
            combined = output_dir / f"{stem}_drone_robotdog_bbox.mp4"
            count = combine_views(rendered["drone"], rendered["robotdog"], combined)
            print(f"[OK] {combined.name}: frames={count}")
    print(f"[DONE] bbox check videos: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
