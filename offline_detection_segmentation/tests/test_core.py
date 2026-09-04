from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from offline_detection_segmentation.core import (
    LABEL_FREE,
    LABEL_OBSTACLE,
    LABEL_TARGET,
    LABEL_UNKNOWN,
    FusedPrediction,
    InstancePrediction,
    fuse_instances,
    mask_to_grid,
    write_cache,
)
from offline_detection_segmentation.rendering import render_scene_overlay
from offline_detection_segmentation.precompute import (
    FrameRecord,
    _cache_is_current,
    _is_unrecoverable_cuda_error,
    _select_shard,
)


class FusionTest(unittest.TestCase):
    def test_deterministic_shards_are_disjoint_and_complete(self) -> None:
        records = [
            FrameRecord(Path(f"frame_{index}.jpg"), Path(f"/tmp/frame_{index}.jpg"))
            for index in range(11)
        ]
        shards = [
            _select_shard(records, num_shards=4, shard_index=index)
            for index in range(4)
        ]
        paths = [record.relative_path for shard in shards for record in shard]
        self.assertEqual(len(paths), len(set(paths)))
        self.assertEqual(set(paths), {record.relative_path for record in records})

    def test_unrecoverable_cuda_error_detection(self) -> None:
        self.assertTrue(
            _is_unrecoverable_cuda_error(
                RuntimeError("CUDA error: the launch timed out and was terminated")
            )
        )
        self.assertFalse(_is_unrecoverable_cuda_error(RuntimeError("corrupt image")))

    def test_current_cache_is_detected_for_resume(self) -> None:
        prediction = FusedPrediction(
            scene_mask=np.asarray([[LABEL_UNKNOWN]], dtype=np.uint8),
            person_valid=False,
            person_box_xyxy=np.zeros(4, dtype=np.float32),
            person_box_cxcywh_norm=np.zeros(4, dtype=np.float32),
            person_score=0.0,
            obstacle_boxes_xyxy=np.empty((0, 4), dtype=np.float32),
            obstacle_scores=np.empty((0,), dtype=np.float32),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.npz"
            write_cache(path, prediction, grid_size=(1, 1), metadata={})
            self.assertTrue(_cache_is_current(path))

    def test_fusion_precedence_and_single_person(self) -> None:
        person_mask = np.zeros((4, 4), dtype=bool)
        person_mask[0, 0] = True
        second_person_mask = np.zeros((4, 4), dtype=bool)
        second_person_mask[2, 2] = True
        crate_mask = np.zeros((4, 4), dtype=bool)
        crate_mask[1, 0] = True
        instances = [
            InstancePrediction(np.array([0, 0, 1, 1]), 0.9, 0, "person", person_mask),
            InstancePrediction(np.array([2, 2, 3, 3]), 0.6, 0, "person", second_person_mask),
            InstancePrediction(np.array([0, 1, 1, 2]), 0.7, 1, "box", crate_mask),
        ]
        fused = fuse_instances(
            (4, 4),
            instances,
            person_confidence=0.25,
            object_confidence=0.25,
            person_label_patterns=["person", "human", "man", "woman"],
        )

        self.assertTrue(fused.person_valid)
        self.assertEqual(fused.scene_mask[0, 0], LABEL_TARGET)
        self.assertEqual(fused.scene_mask[1, 0], LABEL_OBSTACLE)
        self.assertEqual(fused.scene_mask[2, 2], LABEL_OBSTACLE)
        self.assertEqual(fused.scene_mask[0, 1], LABEL_UNKNOWN)
        self.assertEqual(fused.scene_mask[2, 0], LABEL_UNKNOWN)
        self.assertEqual(fused.obstacle_boxes_xyxy.shape, (2, 4))

    def test_grid_ratios_sum_to_one(self) -> None:
        mask = np.asarray([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=np.uint8)
        grid = mask_to_grid(mask, (1, 2))
        np.testing.assert_allclose(grid.sum(axis=-1), 1.0)
        np.testing.assert_allclose(grid[0, 0], [0.5, 0.5, 0.0, 0.0])
        np.testing.assert_allclose(grid[0, 1], [0.0, 0.0, 0.5, 0.5])

    def test_cache_round_trip(self) -> None:
        prediction = FusedPrediction(
            scene_mask=np.asarray([[0, 1], [2, 3]], dtype=np.uint8),
            person_valid=True,
            person_box_xyxy=np.asarray([1, 1, 2, 2], dtype=np.float32),
            person_box_cxcywh_norm=np.asarray([0.75, 0.75, 0.5, 0.5], dtype=np.float32),
            person_score=0.8,
            obstacle_boxes_xyxy=np.empty((0, 4), dtype=np.float32),
            obstacle_scores=np.empty((0,), dtype=np.float32),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.npz"
            write_cache(path, prediction, grid_size=(1, 1), metadata={"source": "frame.jpg"})
            with np.load(path, allow_pickle=False) as cache:
                np.testing.assert_array_equal(cache["scene_mask"], prediction.scene_mask)
                self.assertEqual(json.loads(str(cache["metadata_json"].item()))["source"], "frame.jpg")

    def test_visualization_size(self) -> None:
        image = Image.new("RGB", (4, 4), color=(100, 100, 100))
        mask = np.asarray(
            [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]],
            dtype=np.uint8,
        )
        rendered = render_scene_overlay(
            image,
            mask,
            person_valid=True,
            person_box_xyxy=[2, 0, 4, 4],
            person_score=0.9,
        )
        self.assertEqual(rendered.size, image.size)


if __name__ == "__main__":
    unittest.main()
