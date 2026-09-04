"""Model-independent YOLO instance fusion and cache helpers.

The cache deliberately discards non-person categories. The single target person is
kept separate; every other detected foreground instance becomes one obstacle mask.
Pixels not covered by a YOLO instance remain unknown.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


SCHEMA_VERSION = "offline_perception.yolo_only.v3"
LEGACY_SCHEMA_VERSIONS = {"offline_perception.yolo_only.v2", SCHEMA_VERSION}

LABEL_UNKNOWN = 0
LABEL_FREE = 1
LABEL_OBSTACLE = 2
LABEL_TARGET = 3
NUM_LABELS = 4

LABEL_NAMES = {
    LABEL_UNKNOWN: "unknown",
    LABEL_FREE: "free",
    LABEL_OBSTACLE: "obstacle",
    LABEL_TARGET: "target_person",
}


@dataclass
class InstancePrediction:
    """One YOLO instance prediction in original-image coordinates."""

    box_xyxy: np.ndarray
    score: float
    class_id: int
    class_name: str
    mask: np.ndarray


@dataclass
class FusedPrediction:
    """Category-free scene representation consumed by the cache writer."""

    scene_mask: np.ndarray
    person_valid: bool
    person_box_xyxy: np.ndarray
    person_box_cxcywh_norm: np.ndarray
    person_score: float
    obstacle_boxes_xyxy: np.ndarray
    obstacle_scores: np.ndarray
    # All person proposals, sorted by detector confidence.  The legacy
    # person_* fields above remain the highest-confidence proposal so existing
    # consumers can continue to operate while candidate-aware code migrates.
    person_candidates_xyxy: np.ndarray = field(
        default_factory=lambda: np.empty((0, 4), dtype=np.float32)
    )
    person_candidate_scores: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=np.float32)
    )


def _normalize_name(value: str) -> str:
    value = str(value).strip().lower().replace("_", "-")
    return re.sub(r"\s+", " ", value)


def _matches_any(label_name: str, patterns: Iterable[str]) -> bool:
    """Match ADE20K-style labels such as ``sidewalk, pavement`` safely."""

    name = _normalize_name(label_name)
    alternatives = [_normalize_name(part) for part in re.split(r"[,;/]", name)]
    for raw_pattern in patterns:
        pattern = _normalize_name(raw_pattern)
        if not pattern:
            continue
        for alternative in alternatives:
            if alternative == pattern:
                return True
            if re.search(r"(?<![a-z0-9])" + re.escape(pattern) + r"(?![a-z0-9])", alternative):
                return True
    return False


def _ensure_mask(mask: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    value = np.asarray(mask)
    if value.shape != shape:
        raise ValueError(f"instance mask has shape {value.shape}, expected {shape}")
    return value.astype(bool, copy=False)


def _normalized_box(box_xyxy: np.ndarray, width: int, height: int) -> np.ndarray:
    x1, y1, x2, y2 = np.asarray(box_xyxy, dtype=np.float32)
    x1 = np.clip(x1, 0.0, float(width))
    x2 = np.clip(x2, 0.0, float(width))
    y1 = np.clip(y1, 0.0, float(height))
    y2 = np.clip(y2, 0.0, float(height))
    return np.asarray(
        [
            (x1 + x2) / (2.0 * width),
            (y1 + y2) / (2.0 * height),
            max(0.0, x2 - x1) / width,
            max(0.0, y2 - y1) / height,
        ],
        dtype=np.float32,
    )


def fuse_instances(
    image_shape: Tuple[int, int],
    instances: Sequence[InstancePrediction],
    *,
    person_confidence: float,
    object_confidence: float,
    person_label_patterns: Sequence[str],
) -> FusedPrediction:
    """Fuse YOLO instances into a disjoint target/obstacle/unknown mask.

    Only the highest-confidence person is the target. Every remaining instance,
    including an additional person, is treated as an obstacle without preserving
    its category. Target person has precedence over overlapping obstacle masks.
    """

    height, width = int(image_shape[0]), int(image_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"image_shape must be positive HxW, got {image_shape}")

    person_candidates = [
        index
        for index, instance in enumerate(instances)
        if _matches_any(instance.class_name, person_label_patterns)
        and float(instance.score) >= float(person_confidence)
    ]
    person_candidates.sort(key=lambda index: float(instances[index].score), reverse=True)
    target_index: Optional[int] = person_candidates[0] if person_candidates else None

    instance_obstacle = np.zeros((height, width), dtype=bool)
    obstacle_boxes: List[np.ndarray] = []
    obstacle_scores: List[float] = []
    person_candidate_set = set(person_candidates)
    for index, instance in enumerate(instances):
        if index in person_candidate_set or float(instance.score) < float(object_confidence):
            continue
        instance_obstacle |= _ensure_mask(instance.mask, (height, width))
        obstacle_boxes.append(np.asarray(instance.box_xyxy, dtype=np.float32).reshape(4))
        obstacle_scores.append(float(instance.score))

    person_mask = np.zeros((height, width), dtype=bool)
    person_box = np.zeros(4, dtype=np.float32)
    person_box_norm = np.zeros(4, dtype=np.float32)
    person_score = 0.0
    person_valid = target_index is not None
    if target_index is not None:
        target = instances[target_index]
        for person_index in person_candidates:
            person_mask |= _ensure_mask(instances[person_index].mask, (height, width))
        person_box = np.asarray(target.box_xyxy, dtype=np.float32).reshape(4)
        person_box_norm = _normalized_box(person_box, width, height)
        person_score = float(target.score)

    obstacle = instance_obstacle & ~person_mask

    scene_mask = np.full((height, width), LABEL_UNKNOWN, dtype=np.uint8)
    scene_mask[obstacle] = LABEL_OBSTACLE
    scene_mask[person_mask] = LABEL_TARGET

    boxes_array = (
        np.stack(obstacle_boxes).astype(np.float32, copy=False)
        if obstacle_boxes
        else np.empty((0, 4), dtype=np.float32)
    )
    scores_array = np.asarray(obstacle_scores, dtype=np.float32)
    person_boxes_array = (
        np.stack([np.asarray(instances[index].box_xyxy, dtype=np.float32).reshape(4) for index in person_candidates])
        if person_candidates
        else np.empty((0, 4), dtype=np.float32)
    )
    person_scores_array = np.asarray(
        [float(instances[index].score) for index in person_candidates], dtype=np.float32
    )
    return FusedPrediction(
        scene_mask=scene_mask,
        person_valid=person_valid,
        person_box_xyxy=person_box,
        person_box_cxcywh_norm=person_box_norm,
        person_score=person_score,
        obstacle_boxes_xyxy=boxes_array,
        obstacle_scores=scores_array,
        person_candidates_xyxy=person_boxes_array,
        person_candidate_scores=person_scores_array,
    )


def mask_to_grid(scene_mask: np.ndarray, grid_size: Tuple[int, int]) -> np.ndarray:
    """Convert the full mask to per-cell ratios [unknown, free, obstacle, target]."""

    mask = np.asarray(scene_mask)
    if mask.ndim != 2:
        raise ValueError(f"scene_mask must be HxW, got {mask.shape}")
    grid_h, grid_w = (int(grid_size[0]), int(grid_size[1]))
    if grid_h <= 0 or grid_w <= 0:
        raise ValueError(f"grid_size must be positive, got {grid_size}")

    height, width = mask.shape
    y_edges = np.linspace(0, height, grid_h + 1, dtype=np.int64)
    x_edges = np.linspace(0, width, grid_w + 1, dtype=np.int64)
    output = np.zeros((grid_h, grid_w, NUM_LABELS), dtype=np.float32)
    for row in range(grid_h):
        for column in range(grid_w):
            cell = mask[y_edges[row] : y_edges[row + 1], x_edges[column] : x_edges[column + 1]]
            if cell.size == 0:
                output[row, column, LABEL_UNKNOWN] = 1.0
                continue
            counts = np.bincount(cell.reshape(-1), minlength=NUM_LABELS)[:NUM_LABELS]
            output[row, column] = counts.astype(np.float32) / float(cell.size)
    return output


def cache_path_for_image(output_root: Path, image_relative_path: Path) -> Path:
    """Map ``frames/.../0001.jpg`` to ``frames/.../0001.perception.npz``."""

    relative = Path(image_relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe relative image path: {relative}")
    return Path(output_root) / relative.parent / f"{relative.stem}.perception.npz"


def write_cache(
    path: Path,
    prediction: FusedPrediction,
    *,
    grid_size: Tuple[int, int],
    metadata: Mapping[str, Any],
) -> None:
    """Atomically write one compressed frame cache."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    metadata_json = json.dumps(dict(metadata), ensure_ascii=False, sort_keys=True)
    with temporary_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            schema_version=np.asarray(SCHEMA_VERSION),
            scene_mask=prediction.scene_mask.astype(np.uint8, copy=False),
            mask_grid=mask_to_grid(prediction.scene_mask, grid_size).astype(np.float16),
            person_valid=np.asarray(prediction.person_valid, dtype=np.bool_),
            person_box_xyxy=prediction.person_box_xyxy.astype(np.float32, copy=False),
            person_box_cxcywh_norm=prediction.person_box_cxcywh_norm.astype(
                np.float32, copy=False
            ),
            person_score=np.asarray(prediction.person_score, dtype=np.float32),
            obstacle_boxes_xyxy=prediction.obstacle_boxes_xyxy.astype(np.float32, copy=False),
            obstacle_scores=prediction.obstacle_scores.astype(np.float32, copy=False),
            person_candidates_xyxy=prediction.person_candidates_xyxy.astype(np.float32, copy=False),
            person_candidate_scores=prediction.person_candidate_scores.astype(np.float32, copy=False),
            metadata_json=np.asarray(metadata_json),
        )
    os.replace(temporary_path, path)


def read_cache_metadata(cache_path: Path) -> Dict[str, Any]:
    """Read cache metadata without enabling pickle."""

    with np.load(cache_path, allow_pickle=False) as cache:
        return json.loads(str(cache["metadata_json"].item()))
