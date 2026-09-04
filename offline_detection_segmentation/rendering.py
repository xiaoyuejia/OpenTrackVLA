"""Shared visualization helpers for cache generation and manual inspection."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .core import LABEL_OBSTACLE, LABEL_TARGET, LABEL_UNKNOWN


OBSTACLE_COLOR: Tuple[int, int, int] = (235, 50, 45)
TARGET_COLOR: Tuple[int, int, int] = (30, 220, 80)
UNKNOWN_COLOR: Tuple[int, int, int] = (255, 190, 0)


def visualization_path_for_image(output_root: Path, image_relative_path: Path) -> Path:
    relative = Path(image_relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe relative image path: {relative}")
    return (
        Path(output_root)
        / "visualizations"
        / relative.parent
        / f"{relative.stem}.perception.jpg"
    )


def render_scene_overlay(
    image: Image.Image,
    scene_mask: np.ndarray,
    *,
    person_valid: bool,
    person_box_xyxy: Sequence[float],
    person_score: float,
    alpha: float = 0.45,
    show_unknown: bool = False,
) -> Image.Image:
    """Overlay the unified obstacle mask and target-person result on RGB."""

    image = image.convert("RGB")
    mask = np.asarray(scene_mask)
    if mask.shape != (image.height, image.width):
        raise ValueError(
            f"mask/image size mismatch: mask={mask.shape}, image={(image.height, image.width)}"
        )

    base = np.asarray(image, dtype=np.float32)
    overlay = base.copy()
    alpha = float(np.clip(alpha, 0.0, 1.0))
    colors = [(LABEL_OBSTACLE, OBSTACLE_COLOR), (LABEL_TARGET, TARGET_COLOR)]
    if show_unknown:
        colors.insert(0, (LABEL_UNKNOWN, UNKNOWN_COLOR))
    for label_id, color in colors:
        pixels = mask == label_id
        overlay[pixels] = base[pixels] * (1.0 - alpha) + np.asarray(color) * alpha
    rendered = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))

    draw = ImageDraw.Draw(rendered)
    font = ImageFont.load_default()
    line_width = max(2, min(image.width, image.height) // 250)
    if person_valid:
        box = tuple(float(value) for value in person_box_xyxy)
        draw.rectangle(box, outline=TARGET_COLOR, width=line_width)
        label = f"target person {float(person_score):.2f}"
        text_x = max(0.0, box[0])
        text_y = max(0.0, box[1] - 14.0)
        text_box = draw.textbbox((text_x, text_y), label, font=font)
        draw.rectangle(text_box, fill=(0, 0, 0))
        draw.text((text_x, text_y), label, fill=TARGET_COLOR, font=font)

    obstacle_ratio = float(np.mean(mask == LABEL_OBSTACLE))
    legend = f"red=unified obstacle mask  green=target person  obstacle={obstacle_ratio:.1%}"
    text_box = draw.textbbox((7, 7), legend, font=font)
    draw.rectangle((4, 4, text_box[2] + 4, text_box[3] + 4), fill=(0, 0, 0))
    draw.text((7, 7), legend, fill=(255, 255, 255), font=font)
    return rendered


def save_visualization(
    output_path: Path,
    image: Image.Image,
    scene_mask: np.ndarray,
    *,
    person_valid: bool,
    person_box_xyxy: Sequence[float],
    person_score: float,
    alpha: float,
    show_unknown: bool,
    jpeg_quality: int,
) -> None:
    rendered = render_scene_overlay(
        image,
        scene_mask,
        person_valid=person_valid,
        person_box_xyxy=person_box_xyxy,
        person_score=person_score,
        alpha=alpha,
        show_unknown=show_unknown,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    rendered.save(
        temporary_path,
        format="JPEG",
        quality=int(np.clip(jpeg_quality, 1, 100)),
        optimize=True,
    )
    os.replace(temporary_path, output_path)
