import numpy as np
import torch

from candidate_matching import CandidateTextMatcher, select_top_candidate
from offline_detection_segmentation.core import InstancePrediction, SCHEMA_VERSION, fuse_instances


def test_fuse_preserves_all_person_candidates_and_legacy_target():
    mask = np.zeros((20, 20), dtype=bool)
    instances = [
        InstancePrediction(np.array([1, 1, 5, 10]), 0.55, 0, "person", mask),
        InstancePrediction(np.array([10, 2, 16, 14]), 0.92, 0, "person", mask),
        InstancePrediction(np.array([2, 12, 8, 19]), 0.80, 1, "car", mask),
    ]
    fused = fuse_instances(
        (20, 20), instances, person_confidence=0.25, object_confidence=0.25, person_label_patterns=["person"]
    )
    assert SCHEMA_VERSION.endswith("v3")
    assert fused.person_candidates_xyxy.shape == (2, 4)
    assert np.allclose(fused.person_candidate_scores, [0.92, 0.55])
    assert fused.person_box_xyxy.tolist() == [10.0, 2.0, 16.0, 14.0]


def test_top8_selector_uses_threshold_but_margin_is_diagnostic_only():
    logits = torch.tensor([[[3.0, 0.0, -2.0, -9.0]]])
    valid = torch.tensor([[[True, True, True, False]]])
    selected = select_top_candidate(logits, valid, enter_threshold=0.70, margin_threshold=0.15)
    assert selected.index.item() == 0
    assert selected.accepted.item()
    ambiguous = select_top_candidate(torch.tensor([[[0.9, 0.8, -1.0, -9.0]]]), valid, enter_threshold=0.70, margin_threshold=0.15)
    assert ambiguous.accepted.item()
    assert ambiguous.margin.item() < 0.15


def test_candidate_matcher_cosine_residual_starts_disabled():
    matcher = CandidateTextMatcher(8, hidden_dim=16)
    torch.testing.assert_close(matcher.cosine_gain.detach(), torch.tensor(0.0))


def test_candidate_matcher_masks_padding():
    matcher = CandidateTextMatcher(8, hidden_dim=16)
    candidates = torch.randn(2, 2, 8, 8)
    text = torch.randn(2, 2, 8)
    valid = torch.ones(2, 2, 8, dtype=torch.bool)
    valid[..., 6:] = False
    logits = matcher(candidates, text, valid)
    assert logits.shape == (2, 2, 8)
    assert (logits[..., 7] < -1.0e30).all()
