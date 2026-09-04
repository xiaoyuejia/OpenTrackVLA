"""Render one cached mask over its source RGB image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image

from .rendering import render_scene_overlay


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize an offline perception .npz cache")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--image", type=Path, help="defaults to metadata source_path")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--show-unknown", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    with np.load(args.cache, allow_pickle=False) as cache:
        scene_mask = cache["scene_mask"]
        person_valid = bool(cache["person_valid"].item())
        person_box = cache["person_box_xyxy"].astype(np.float32)
        person_score = float(cache["person_score"].item())
        metadata = json.loads(str(cache["metadata_json"].item()))

    image_path = args.image or Path(metadata["source_path"])
    image = Image.open(image_path).convert("RGB")
    if scene_mask.shape != (image.height, image.width):
        raise ValueError(
            f"cache/image size mismatch: mask={scene_mask.shape}, image={(image.height, image.width)}"
        )

    rendered = render_scene_overlay(
        image,
        scene_mask,
        person_valid=person_valid,
        person_box_xyxy=person_box,
        person_score=person_score,
        alpha=args.alpha,
        show_unknown=args.show_unknown,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered.save(args.output)
    print(f"saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
