"""Lazy model wrapper for YOLO/YOLOE instance segmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Mapping, Sequence, Tuple

import numpy as np
from PIL import Image

from .core import (
    FusedPrediction,
    InstancePrediction,
    fuse_instances,
)


def _class_name(names: Any, class_id: int) -> str:
    if isinstance(names, Mapping):
        return str(names.get(class_id, names.get(str(class_id), class_id)))
    if isinstance(names, Sequence) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def _rectangle_mask(box: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    height, width = shape
    x1, y1, x2, y2 = np.rint(box).astype(np.int64)
    x1, x2 = np.clip([x1, x2], 0, width)
    y1, y2 = np.clip([y1, y2], 0, height)
    output = np.zeros((height, width), dtype=bool)
    if x2 > x1 and y2 > y1:
        output[y1:y2, x1:x2] = True
    return output


class YOLOInstanceSegmenter:
    def __init__(
        self,
        weights: str,
        *,
        device: str,
        image_size: int,
        confidence: float,
        half: bool,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                "ultralytics is required. Install offline_detection_segmentation/requirements.txt"
            ) from error

        weights_path = Path(weights)
        if weights_path.suffix == ".pt" and weights_path.parent != Path("."):
            weights_path.parent.mkdir(parents=True, exist_ok=True)
        self.model = YOLO(str(weights_path))
        self.device = device
        self.image_size = int(image_size)
        self.confidence = float(confidence)
        self.half = bool(half and str(device).startswith("cuda"))

    def predict(self, images: Sequence[Image.Image]) -> List[List[InstancePrediction]]:
        # Ultralytics interprets ndarray sources as BGR, while our PIL inputs are RGB.
        sources = [np.asarray(image, dtype=np.uint8)[:, :, ::-1] for image in images]
        results = self.model.predict(
            source=sources,
            imgsz=self.image_size,
            conf=self.confidence,
            device=self.device,
            half=self.half,
            retina_masks=True,
            verbose=False,
            stream=False,
        )

        batches: List[List[InstancePrediction]] = []
        for image, result in zip(images, results):
            width, height = image.size
            instances: List[InstancePrediction] = []
            if result.boxes is None:
                batches.append(instances)
                continue

            boxes = result.boxes.xyxy.detach().float().cpu().numpy()
            scores = result.boxes.conf.detach().float().cpu().numpy()
            classes = result.boxes.cls.detach().long().cpu().numpy()
            raw_masks = None if result.masks is None else result.masks.data.detach().float().cpu().numpy()

            for index, (box, score, class_id) in enumerate(zip(boxes, scores, classes)):
                if raw_masks is not None and index < len(raw_masks):
                    mask_image = Image.fromarray((raw_masks[index] >= 0.5).astype(np.uint8) * 255)
                    if mask_image.size != (width, height):
                        mask_image = mask_image.resize((width, height), resample=Image.Resampling.NEAREST)
                    mask = np.asarray(mask_image, dtype=np.uint8) > 0
                else:
                    mask = _rectangle_mask(box, (height, width))
                class_id_int = int(class_id)
                instances.append(
                    InstancePrediction(
                        box_xyxy=np.asarray(box, dtype=np.float32),
                        score=float(score),
                        class_id=class_id_int,
                        class_name=_class_name(result.names, class_id_int),
                        mask=mask,
                    )
                )
            batches.append(instances)
        return batches

    def predict_person_candidates(
        self,
        images: Sequence[Image.Image],
        *,
        person_label_patterns: Sequence[str] = ("person",),
        top_k: int = 8,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Return confidence-sorted person boxes without materializing masks."""
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        sources = [np.asarray(image, dtype=np.uint8)[:, :, ::-1] for image in images]
        results = self.model.predict(
            source=sources,
            imgsz=self.image_size,
            conf=self.confidence,
            device=self.device,
            half=self.half,
            retina_masks=False,
            verbose=False,
            stream=False,
        )
        output: List[Tuple[np.ndarray, np.ndarray]] = []
        normalized_patterns = {str(value).strip().lower() for value in person_label_patterns}
        for image, result in zip(images, results):
            width, height = image.size
            if result.boxes is None:
                output.append((np.empty((0, 4), np.float32), np.empty((0,), np.float32)))
                continue
            boxes = result.boxes.xyxy.detach().float().cpu().numpy()
            scores = result.boxes.conf.detach().float().cpu().numpy()
            classes = result.boxes.cls.detach().long().cpu().numpy()
            rows = []
            for box, score, class_id in zip(boxes, scores, classes):
                name = _class_name(result.names, int(class_id)).strip().lower()
                if name not in normalized_patterns:
                    continue
                x1, y1, x2, y2 = np.asarray(box, dtype=np.float32)
                rows.append((float(score), np.asarray([
                    (x1 + x2) / (2.0 * width),
                    (y1 + y2) / (2.0 * height),
                    max(0.0, x2 - x1) / width,
                    max(0.0, y2 - y1) / height,
                ], dtype=np.float32)))
            rows.sort(key=lambda item: item[0], reverse=True)
            rows = rows[:top_k]
            output.append((
                np.stack([row[1] for row in rows]) if rows else np.empty((0, 4), np.float32),
                np.asarray([row[0] for row in rows], dtype=np.float32),
            ))
        return output


class OfflinePerceptionPipeline:
    """Run one YOLO model and merge all non-target foreground instances."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        model_config = config["models"]
        threshold_config = config["thresholds"]
        runtime_config = config["runtime"]
        device = str(runtime_config["device"])
        half = bool(runtime_config.get("half", True))

        minimum_yolo_confidence = min(
            float(threshold_config["person_confidence"]),
            float(threshold_config["object_confidence"]),
        )
        self.yolo = YOLOInstanceSegmenter(
            str(model_config["yolo_weights"]),
            device=device,
            image_size=int(model_config.get("yolo_image_size", 960)),
            confidence=minimum_yolo_confidence,
            half=half,
        )
        self.config = config

    def predict(self, images: Sequence[Image.Image]) -> List[FusedPrediction]:
        instances_batch = self.yolo.predict(images)
        if len(instances_batch) != len(images):
            raise RuntimeError("model output batch size does not match input batch size")

        threshold_config = self.config["thresholds"]
        return [
            fuse_instances(
                (image.height, image.width),
                instances,
                person_confidence=float(threshold_config["person_confidence"]),
                object_confidence=float(threshold_config["object_confidence"]),
                person_label_patterns=list(
                    self.config.get("target", {}).get(
                        "person_labels", ["person", "human", "man", "woman"]
                    )
                ),
            )
            for image, instances in zip(images, instances_batch)
        ]
