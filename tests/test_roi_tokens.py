import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.cache_gridpool import (
    VisionFeatureCacher,
    build_roi_cache_payload,
    crop_target_roi,
    grid_pool_tokens,
    roi_cache_matches,
)


def _assert_xyxy_inside(crop, width, height):
    x1, y1, x2, y2 = crop
    assert 0 <= x1 < x2 <= width
    assert 0 <= y1 < y2 <= height


def test_crop_target_roi_normal_cxcywh_norm_square():
    image = Image.new("RGB", (640, 480), "white")
    roi, valid, crop = crop_target_roi(
        image,
        [0.5, 0.5, 0.2, 0.1],
        bbox_format="cxcywh_norm",
        expand_ratio=1.5,
        make_square=True,
    )

    assert valid is True
    assert crop == (224, 144, 416, 336)
    assert roi.size == (192, 192)


def test_crop_target_roi_boundaries_and_invalid_fallbacks():
    cases = [
        ([0, 0, 20, 40], "xywh_pixel", True),
        ([-10, -10, 50, 50], "xywh_pixel", True),
        ([0, 0, 0, 0], "xywh_pixel", False),
        ([20, 20, -5, 10], "xywh_pixel", False),
        ([math.nan, 20, 10, 10], "xywh_pixel", False),
        ([10, 10, 2, 2], "xywh_pixel", False),
    ]
    for bbox, bbox_format, expect_valid in cases:
        image = Image.new("RGB", (640, 480), "white")
        roi, valid, crop = crop_target_roi(
            image,
            bbox,
            bbox_format=bbox_format,
            expand_ratio=1.5,
            make_square=True,
            min_crop_size=8,
        )

        assert valid is expect_valid
        _assert_xyxy_inside(crop, 640, 480)
        assert roi.size[0] > 0 and roi.size[1] > 0
        if not expect_valid:
            assert crop == (80, 0, 560, 480)
            assert roi.size == (480, 480)


def test_crop_target_roi_non_square_original_invalid_center_square():
    image = Image.new("RGB", (320, 640), "white")
    roi, valid, crop = crop_target_roi(image, [0, 0, 0, 0], bbox_format="xywh_pixel")

    assert valid is False
    assert crop == (0, 160, 320, 480)
    assert roi.size == (320, 320)


def test_pooled_encoding_shapes_with_mock_encoder():
    image_batch = [Image.new("RGB", (640, 480), "white") for _ in range(3)]
    enc = object.__new__(VisionFeatureCacher)

    def fake_dino(pils):
        return torch.randn(len(pils), 8 * 8, 5), 8, 8

    def fake_siglip(pils, out_hw):
        assert out_hw == (8, 8)
        return torch.randn(len(pils), 8 * 8, 7)

    enc._encode_dino = fake_dino
    enc._encode_siglip = fake_siglip

    fine = enc.encode_pooled_tokens(image_batch, out_tokens=64)
    coarse = grid_pool_tokens(torch.randn(3, 8 * 8, 12), 8, 8, out_tokens=4)
    roi = enc.encode_pooled_tokens(image_batch, out_tokens=16)

    assert coarse.shape == (3, 4, 12)
    assert fine.shape == (3, 64, 12)
    assert roi.shape == (3, 16, 12)


def test_roi_cache_payload_schema_and_mismatch_detection():
    payload = build_roi_cache_payload(
        torch.randn(16, 12),
        roi_token_count=16,
        roi_expand_ratio=1.5,
        roi_make_square=True,
        bbox_format="cxcywh_norm",
        roi_valid=True,
        crop_xyxy=(1, 2, 31, 32),
    )

    assert payload["schema_version"] == "roi_tokens_v1"
    assert payload["tokens"].dtype == torch.float16
    assert roi_cache_matches(payload, 16, 1.5, True)
    assert not roi_cache_matches(payload, 9, 1.5, True)
    assert not roi_cache_matches(payload, 16, 2.0, True)
    assert not roi_cache_matches(payload, 16, 1.5, False)


class _DummyTokenizer:
    def __call__(self, texts, return_tensors, padding, truncation, max_length):
        batch = len(texts)
        return {
            "input_ids": torch.ones(batch, 4, dtype=torch.long),
            "attention_mask": torch.ones(batch, 4, dtype=torch.long),
        }


class _DummyBackbone(torch.nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.emb = torch.nn.Embedding(128, hidden_size)

    def get_input_embeddings(self):
        return self.emb

    def forward(self, inputs_embeds, attention_mask=None, output_hidden_states=False, use_cache=False):
        return SimpleNamespace(last_hidden_state=inputs_embeds)


class _DummyAutoModel:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        return _DummyBackbone()


class _DummyAutoTokenizer:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        return _DummyTokenizer()


def _patch_dummy_llm():
    import model as model_module

    dummy_modelscope = SimpleNamespace(AutoModel=_DummyAutoModel, AutoTokenizer=_DummyAutoTokenizer)
    sys.modules["modelscope"] = dummy_modelscope
    model_module.AutoModel.from_pretrained = _DummyAutoModel.from_pretrained
    model_module.AutoTokenizer.from_pretrained = _DummyAutoTokenizer.from_pretrained
    return model_module


def _make_multi_agent_inputs(batch=2, history=3, vision_dim=12, roi_tokens=16):
    coarse_tokens = torch.randn(batch, 2, history * 4, vision_dim)
    coarse_tidx_one = torch.arange(history).repeat_interleave(4)
    coarse_tidx = coarse_tidx_one.view(1, 1, -1).expand(batch, 2, -1).clone()
    fine_tokens = torch.randn(batch, 2, 64, vision_dim)
    fine_tidx = torch.full((batch, 2, 64), history, dtype=torch.long)
    roi = torch.randn(batch, 2, roi_tokens, vision_dim)
    roi_tidx = torch.full((batch, 2, roi_tokens), history, dtype=torch.long)
    roi_valid = torch.tensor([[True, False], [True, True]], dtype=torch.bool)
    instructions = ["follow the target"] * batch
    return coarse_tokens, coarse_tidx, fine_tokens, fine_tidx, roi, roi_tidx, roi_valid, instructions


def test_multi_agent_forward_with_roi_tokens_and_no_bbox_feat():
    dummy_llm = _patch_dummy_llm()
    cfg = dummy_llm.MultiAgentModelConfig(
        llm_name="dummy",
        n_waypoints=5,
        action_dims=3,
        use_grounding=False,
        use_bbox_tokens=False,
        use_roi_tokens=True,
        roi_token_count=16,
        freeze_llm=True,
    )
    model = dummy_llm.MultiAgentOpenTrackVLA(cfg, vision_feat_dim=12)
    inputs = _make_multi_agent_inputs()

    assert model.cfg.use_bbox_tokens is False
    assert model.cfg.use_grounding is False
    out = model(
        coarse_tokens=inputs[0],
        coarse_tidx=inputs[1],
        fine_tokens=inputs[2],
        fine_tidx=inputs[3],
        roi_tokens=inputs[4],
        roi_tidx=inputs[5],
        roi_valid=inputs[6],
        instructions=inputs[7],
        bbox_feat=None,
        return_dict=True,
    )

    assert out["waypoints"].shape == (2, 2, 5, 3)


def test_multi_agent_forward_without_roi_keeps_old_call_shape():
    dummy_llm = _patch_dummy_llm()
    cfg = dummy_llm.MultiAgentModelConfig(
        llm_name="dummy",
        n_waypoints=5,
        action_dims=3,
        use_grounding=False,
        use_bbox_tokens=False,
        use_roi_tokens=False,
        freeze_llm=True,
    )
    model = dummy_llm.MultiAgentOpenTrackVLA(cfg, vision_feat_dim=12)
    inputs = _make_multi_agent_inputs()
    out = model(
        coarse_tokens=inputs[0],
        coarse_tidx=inputs[1],
        fine_tokens=inputs[2],
        fine_tidx=inputs[3],
        instructions=inputs[7],
        bbox_feat=None,
        return_dict=True,
    )

    assert out["waypoints"].shape == (2, 2, 5, 3)


def test_old_five_kind_checkpoint_expands_when_roi_disabled():
    dummy_llm = _patch_dummy_llm()
    cfg = dummy_llm.MultiAgentModelConfig(
        llm_name="dummy",
        n_waypoints=5,
        action_dims=3,
        use_grounding=False,
        use_bbox_tokens=False,
        use_roi_tokens=False,
        freeze_llm=True,
    )
    model = dummy_llm.MultiAgentOpenTrackVLA(cfg, vision_feat_dim=12)
    state = model.state_dict()
    state["tvi.kind_emb.weight"] = state["tvi.kind_emb.weight"][:5].clone()

    missing, unexpected = model.load_state_dict(state, strict=False)

    assert unexpected == []
    assert "tvi.kind_emb.weight" not in missing


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("test_roi_tokens.py: all checks passed")
