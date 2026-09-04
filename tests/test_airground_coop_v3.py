from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import model_airground_coop_v3 as model_module
from model_airground_coop_v3 import (
    DRONE,
    ROBOTDOG,
    ROUTE_BELIEF,
    ROUTE_COOPERATIVE,
    ROUTE_SEARCH,
    ROUTE_SELF,
    AirGroundCoopV3ModelConfig,
    AirGroundCooperativeVLAV3,
    AirGroundVisibilityRouter,
    MultimodalTrajectoryDecoder,
)
from train_airground_coop_v3 import (
    AirGroundV3DataConfig,
    AirGroundV3JsonDataset,
    AirGroundV3TrainConfig,
    CORRUPTION_ALL_FULL,
    CORRUPTION_CURRENT_FULL,
    CORRUPTION_RECENT_FULL,
    CORRUPTION_ROI_ONLY,
    RotatingTemporalStrideDistributedSampler,
    apply_airground_v3_defaults,
    build_feasible_receiver_recovery_target,
    build_airground_v3_model,
    cxcywh_iou,
    forward_airground_v3_loss,
    _kinematics_per_sample,
    agent_poses_from_unreal,
    build_cooperative_waypoint_targets,
    load_config,
    perturb_receiver_pose,
    transform_agent_poses_shared_se2,
)


class _TinyCausalLLM(nn.Module):
    """Small sequence mixer whose token output depends on earlier tokens."""

    def __init__(self, hidden_size: int = 24):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = nn.Embedding(64, hidden_size)
        self.projection = nn.Linear(hidden_size, hidden_size)
        self.call_batch_sizes = []

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, inputs_embeds, attention_mask, **kwargs):
        del kwargs
        self.call_batch_sizes.append(inputs_embeds.size(0))
        valid = attention_mask.to(inputs_embeds.dtype).unsqueeze(-1)
        cumulative = (inputs_embeds * valid).cumsum(dim=1)
        count = valid.cumsum(dim=1).clamp_min(1.0)
        mixed = inputs_embeds + cumulative / count
        return SimpleNamespace(last_hidden_state=self.projection(mixed))


class _TinyTokenizer:
    def __call__(self, texts, **kwargs):
        del kwargs
        lengths = [max(1, min(6, len(text.split()))) for text in texts]
        width = max(lengths)
        ids = torch.zeros(len(texts), width, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for row, length in enumerate(lengths):
            ids[row, :length] = torch.arange(1, length + 1)
            mask[row, :length] = 1
        return {"input_ids": ids, "attention_mask": mask}


def _build_tiny_model(monkeypatch: pytest.MonkeyPatch) -> AirGroundCooperativeVLAV3:
    monkeypatch.setattr(
        model_module.AutoModel,
        "from_pretrained",
        lambda *args, **kwargs: _TinyCausalLLM(),
    )
    monkeypatch.setattr(
        model_module.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: _TinyTokenizer(),
    )
    torch.manual_seed(7)
    return AirGroundCooperativeVLAV3(
        AirGroundCoopV3ModelConfig(
            llm_name="tiny",
            freeze_llm=True,
            n_waypoints=4,
            num_modes=3,
            perception_grid_size=4,
            insert_time_tokens=False,
            coop_hidden_dim=16,
            coop_encoder_layers=1,
            coop_decoder_layers=1,
            coop_num_heads=4,
            coop_dropout=0.0,
            jepa_hidden_dim=16,
            jepa_decoder_layers=1,
            jepa_num_heads=4,
            jepa_dropout=0.0,
        ),
        vision_feat_dim=12,
    )


def _model_inputs(batch_size: int = 2) -> dict:
    torch.manual_seed(11)
    detection = torch.tensor(
        [
            [[0.45, 0.50, 0.20, 0.30, 0.90, 1.0], [0.55, 0.48, 0.25, 0.20, 0.80, 1.0]],
            [[0.52, 0.44, 0.18, 0.24, 0.85, 1.0], [0.42, 0.57, 0.22, 0.28, 0.75, 1.0]],
        ],
        dtype=torch.float32,
    )[:batch_size]
    grid = torch.zeros(batch_size, 2, 4, 4, 4)
    grid[..., 0] = 0.7
    grid[..., 2] = 0.3
    grid[:, :, 1:3, 1:3, 3] = 0.5
    return {
        "coarse_tokens": torch.randn(batch_size, 2, 8, 12),
        "coarse_tidx": torch.arange(8).view(1, 1, 8).expand(batch_size, 2, -1),
        "fine_tokens": torch.randn(batch_size, 2, 16, 12),
        "fine_tidx": torch.full((batch_size, 2, 16), 31),
        "detection_feat": detection,
        "perception_grid": grid,
        "agent_poses": torch.tensor(
            [[[1.0, 2.0, 0.0, 1.0], [-1.0, -2.0, 1.0, 0.0]]],
            dtype=torch.float32,
        ).expand(batch_size, -1, -1).clone(),
        "instructions": ["Follow the target person."] * batch_size,
    }


def test_agent_poses_are_two_independent_world_pose_tokens() -> None:
    value, valid = agent_poses_from_unreal(
        [0.0, 100.0, 50.0, 0.0, 120.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 90.0, 0.0],
    )
    assert valid.item()
    torch.testing.assert_close(
        value,
        torch.tensor(
            [
                [0.0, 1.0, 3.0**0.5 / 2.0, -0.5],
                [0.0, 0.0, 1.0, 0.0],
            ]
        ),
        atol=1e-5,
        rtol=1e-5,
    )


def test_shared_se2_preserves_pair_distance_and_yaw_difference() -> None:
    poses = torch.tensor(
        [[2.0, 3.0, 0.0, 1.0], [5.0, 7.0, 1.0, 0.0]], dtype=torch.float32
    )
    transformed, centre = transform_agent_poses_shared_se2(
        poses, rotation_rad=0.7, translation_xy_m=torch.tensor([8.0, -3.0])
    )
    torch.testing.assert_close(centre, torch.tensor([3.5, 5.0]))
    torch.testing.assert_close(
        torch.linalg.vector_norm(transformed[0, :2] - transformed[1, :2]),
        torch.tensor(5.0),
    )
    torch.testing.assert_close(transformed[:, :2].mean(dim=0), torch.tensor([8.0, -3.0]))
    original_delta = torch.atan2(poses[0, 2], poses[0, 3]) - torch.atan2(
        poses[1, 2], poses[1, 3]
    )
    transformed_delta = torch.atan2(
        transformed[0, 2], transformed[0, 3]
    ) - torch.atan2(transformed[1, 2], transformed[1, 3])
    torch.testing.assert_close(torch.sin(original_delta), torch.sin(transformed_delta))
    torch.testing.assert_close(torch.cos(original_delta), torch.cos(transformed_delta))


def test_directed_relative_pose_is_invariant_to_shared_se2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _build_tiny_model(monkeypatch).eval()
    poses = torch.tensor(
        [[[2.0, 3.0, 0.0, 1.0], [5.0, 7.0, 1.0, 0.0]]],
        dtype=torch.float32,
    )
    transformed, _ = transform_agent_poses_shared_se2(
        poses[0], rotation_rad=0.7, translation_xy_m=torch.tensor([8.0, -3.0])
    )
    original_features = model._directed_relative_pose_features(poses)
    transformed_features = model._directed_relative_pose_features(
        transformed.unsqueeze(0)
    )
    torch.testing.assert_close(
        original_features, transformed_features, atol=1.0e-5, rtol=1.0e-5
    )
    # Directed rows are receiver-centric and therefore have opposite yaw deltas.
    torch.testing.assert_close(
        original_features[0, DRONE, 2], -original_features[0, ROBOTDOG, 2]
    )


def test_receiver_pose_perturbation_changes_only_the_corrupted_input_pose() -> None:
    clean = torch.tensor([2.0, 3.0, 0.0, 1.0])
    perturbed = perturb_receiver_pose(
        clean, forward_offset_m=1.0, right_offset_m=-0.5, yaw_offset_rad=0.25
    )
    torch.testing.assert_close(perturbed[:2], torch.tensor([3.0, 2.5]))
    torch.testing.assert_close(perturbed[2], torch.sin(torch.tensor(0.25)))
    torch.testing.assert_close(perturbed[3], torch.cos(torch.tensor(0.25)))
    torch.testing.assert_close(clean, torch.tensor([2.0, 3.0, 0.0, 1.0]))


def test_receiver_recovery_target_respects_origin_and_motion_limits() -> None:
    clean = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    )
    recovered = build_feasible_receiver_recovery_target(
        clean,
        torch.tensor([2.0, 0.0, math.pi / 2.0]),
        dt=0.1,
        max_speed_mps=2.5,
        max_yaw_rate_rps=1.5,
        nonholonomic=False,
    )
    torch.testing.assert_close(recovered[0], torch.zeros(3))
    step_distance = torch.linalg.vector_norm(
        recovered[1:, :2] - recovered[:-1, :2], dim=-1
    )
    yaw_step = torch.atan2(
        torch.sin(recovered[1:, 2] - recovered[:-1, 2]),
        torch.cos(recovered[1:, 2] - recovered[:-1, 2]),
    ).abs()
    assert step_distance.max().item() <= 0.25 + 1.0e-6
    assert yaw_step.max().item() <= 0.15 + 1.0e-6
    # The first target cannot absorb the old 2 m / 90 degree discontinuity.
    assert step_distance[0].item() < 2.0
    assert yaw_step[0].item() < math.pi / 2.0


def test_zero_receiver_perturbation_preserves_clean_target() -> None:
    clean = torch.tensor(
        [[0.0, 0.0, 0.0], [0.2, 0.1, 0.05], [0.4, 0.2, 0.1]]
    )
    recovered = build_feasible_receiver_recovery_target(
        clean,
        torch.zeros(3),
        dt=0.1,
        max_speed_mps=2.5,
        max_yaw_rate_rps=1.5,
        nonholonomic=False,
    )
    torch.testing.assert_close(recovered, clean)


def test_only_the_synthetic_receiver_gets_feasible_recovery_targets() -> None:
    clean = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        ]
    )
    targets = build_cooperative_waypoint_targets(
        clean,
        torch.tensor([False, True]),
        torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, math.pi / 2.0]]),
        dt=0.1,
        drone_max_speed_mps=2.5,
        dog_max_speed_mps=2.5,
        drone_max_yaw_rate_rps=1.5,
        dog_max_yaw_rate_rps=1.5,
    )
    torch.testing.assert_close(targets[DRONE], clean[DRONE])
    torch.testing.assert_close(targets[ROBOTDOG, 0], torch.zeros(3))
    assert torch.linalg.vector_norm(targets[ROBOTDOG, 1, :2]).item() <= 0.25 + 1e-6
    assert abs(float(targets[ROBOTDOG, 1, 2])) <= 0.15 + 1e-6


def test_visibility_router_hysteresis_and_both_invisible_search_state() -> None:
    router = AirGroundVisibilityRouter(
        enter_confidence=0.35,
        exit_confidence=0.20,
        visible_confirm_frames=2,
        invisible_confirm_frames=2,
        belief_hold_frames=1,
    )
    detection = torch.tensor(
        [[0.5, 0.5, 0.2, 0.2, 0.9, 1.0], [0.5, 0.5, 0.2, 0.2, 0.8, 1.0]]
    )
    match = torch.ones(2)
    first = router.update(detection, match)
    assert first["mode"].tolist() == [ROUTE_BELIEF, ROUTE_BELIEF]
    second = router.update(detection, match)
    assert second["mode"].tolist() == [ROUTE_SELF, ROUTE_SELF]

    # One low frame does not switch. Two consecutive low frames switch only the
    # dog to cooperative mode because the drone remains visible.
    dog_missing = detection.clone()
    dog_missing[ROBOTDOG, 4:] = 0.0
    assert router.update(dog_missing, match)["mode"].tolist() == [ROUTE_SELF, ROUTE_SELF]
    switched = router.update(dog_missing, match)
    assert switched["mode"].tolist() == [ROUTE_SELF, ROUTE_COOPERATIVE]

    both_missing = dog_missing.clone()
    both_missing[DRONE, 4:] = 0.0
    router.update(both_missing, match)
    belief = router.update(both_missing, match)
    assert belief["mode"].tolist() == [ROUTE_BELIEF, ROUTE_BELIEF]
    search = router.update(both_missing, match)
    assert search["mode"].tolist() == [ROUTE_SEARCH, ROUTE_SEARCH]

    # A valid, high-confidence YOLO box is still rejected after two frames when
    # the LLM decides that the detected person is not the specified target.
    router.reset()
    router.update(detection, torch.ones(2))
    router.update(detection, torch.ones(2))
    mismatch = torch.tensor([1.0, 0.0])
    assert router.update(detection, mismatch)["mode"].tolist() == [ROUTE_SELF, ROUTE_SELF]
    assert router.update(detection, mismatch)["mode"].tolist() == [
        ROUTE_SELF,
        ROUTE_COOPERATIVE,
    ]


def test_cxcywh_iou_for_target_match_labels() -> None:
    first = torch.tensor(
        [[0.5, 0.5, 0.2, 0.4], [0.1, 0.1, 0.1, 0.1], [0.5, 0.5, 0.0, 0.2]]
    )
    second = torch.tensor(
        [[0.5, 0.5, 0.2, 0.4], [0.9, 0.9, 0.1, 0.1], [0.5, 0.5, 0.2, 0.2]]
    )
    torch.testing.assert_close(cxcywh_iou(first, second), torch.tensor([1.0, 0.0, 0.0]))


def test_bbox_valid_mask_is_loaded_and_legacy_data_is_derived() -> None:
    dataset = object.__new__(AirGroundV3JsonDataset)
    bbox = torch.tensor([[0.5, 0.5, 0.2, 0.3], [0.0, 0.0, 0.0, 0.0]])
    explicit = dataset._load_bbox_valid_mask(
        {"bbox_valid_mask": [False, True]}, bbox
    )
    derived = dataset._load_bbox_valid_mask({}, bbox)
    torch.testing.assert_close(explicit, torch.tensor([False, True]))
    torch.testing.assert_close(derived, torch.tensor([True, False]))


def test_synthetic_false_positive_is_low_iou() -> None:
    dataset = object.__new__(AirGroundV3JsonDataset)
    dataset.cfg = AirGroundV3DataConfig(
        train_json="unused", target_match_iou_threshold=0.3
    )
    torch.manual_seed(5)
    detection = torch.tensor([0.5, 0.5, 0.2, 0.3, 0.9, 1.0])
    gt_bbox = torch.tensor([0.5, 0.5, 0.2, 0.3])
    replacement = dataset._sample_false_positive_detection(
        detection,
        torch.empty(0, 4),
        torch.empty(0),
        gt_bbox,
    )
    assert replacement is not None
    assert replacement[5].item() == 1.0
    assert cxcywh_iou(replacement[:4], gt_bbox).item() < 0.3


def test_detector_target_channel_is_not_verifier_evidence() -> None:
    grid = torch.zeros(1, 2, 2, 4)
    grid[..., 0] = 0.6
    grid[..., 3] = 0.4
    cleaned = AirGroundCooperativeVLAV3._grid_without_detector_target(grid)
    assert cleaned[..., 3].eq(0.0).all()
    assert cleaned[..., 0].eq(1.0).all()


def test_multimodal_decoder_shape_and_gradient() -> None:
    decoder = MultimodalTrajectoryDecoder(
        llm_dim=24,
        hidden_dim=16,
        n_waypoints=5,
        action_dims=3,
        num_modes=4,
        encoder_layers=1,
        decoder_layers=2,
        num_heads=4,
        dropout=0.0,
        use_tanh=False,
    )
    memory = torch.randn(2, 11, 24, requires_grad=True)
    context = torch.randn(2, 24, requires_grad=True)
    trajectories, logits = decoder(memory, context)
    assert trajectories.shape == (2, 4, 5, 3)
    assert logits.shape == (2, 4)
    torch.testing.assert_close(
        trajectories[..., 0, :], torch.zeros_like(trajectories[..., 0, :])
    )
    (trajectories.square().mean() + logits.square().mean()).backward()
    assert memory.grad is not None
    assert context.grad is not None
    assert decoder.mode_queries.grad is not None
    assert decoder.waypoint_queries.grad is not None


def test_full_v3_forward_is_symmetric_and_routes_hidden_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _build_tiny_model(monkeypatch)
    inputs = _model_inputs()
    synthetic = torch.tensor([[False, True], [True, False]])
    output = model(
        **inputs,
        synthetic_occlusion=synthetic,
        route_visibility=~synthetic,
    )
    assert output["self_waypoints"].shape == (2, 2, 4, 3)
    assert output["cooperative_candidates"].shape == (2, 2, 3, 4, 3)
    torch.testing.assert_close(
        output["self_waypoints"][..., 0, :],
        torch.zeros_like(output["self_waypoints"][..., 0, :]),
    )
    torch.testing.assert_close(
        output["cooperative_candidates"][..., 0, :],
        torch.zeros_like(output["cooperative_candidates"][..., 0, :]),
    )
    assert output["cooperative_mode_logits"].shape == (2, 2, 3)
    assert output["target_match_logits"].shape == (2, 2)
    assert output["target_match_probability"].shape == (2, 2)
    assert output["directed_relative_pose"].shape == (2, 2, 5)
    # Two agent-isolated self rows, followed by one joint cooperative row.
    assert model.llm.call_batch_sizes == [4, 2]
    assert output["self_action_context"].shape == (2, 2, 24)
    assert output["target_verify_context"].shape == (2, 2, 24)
    assert not torch.allclose(
        output["self_action_context"], output["target_verify_context"]
    )
    assert output["jepa_prediction_tokens"].shape == (2, 2, 16, 24)
    assert output["jepa_token_mask"][0, ROBOTDOG].all()
    assert not output["jepa_token_mask"][0, DRONE].any()
    assert output["jepa_token_mask"][1, DRONE].all()
    assert not output["jepa_token_mask"][1, ROBOTDOG].any()
    assert output["route_to_cooperative"].tolist() == [[False, True], [True, False]]
    torch.testing.assert_close(
        output["waypoints"][0, DRONE], output["self_waypoints"][0, DRONE]
    )
    torch.testing.assert_close(
        output["waypoints"][0, ROBOTDOG], output["cooperative_waypoints"][0, ROBOTDOG]
    )
    output["waypoints"].square().mean().backward()
    assert model.self_planners[DRONE].net[-1].weight.grad is not None
    assert model.self_planners[ROBOTDOG].net[-1].weight.grad is not None
    assert model.coop_decoders[DRONE].action_out.weight.grad is not None
    assert model.coop_decoders[ROBOTDOG].action_out.weight.grad is not None
    assert model.jepa_predictors[DRONE].output[-1].weight.grad is not None
    assert model.jepa_predictors[ROBOTDOG].output[-1].weight.grad is not None


def test_partial_receiver_masks_preserve_early_history_and_clean_self_flow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _build_tiny_model(monkeypatch).eval()
    inputs = _model_inputs(batch_size=1)
    receiver = torch.tensor([[False, True]])
    coarse_mask = torch.zeros(1, 2, 8, dtype=torch.bool)
    coarse_mask[0, ROBOTDOG, -2:] = True
    fine_mask = torch.zeros(1, 2, 16, dtype=torch.bool)
    fine_mask[0, ROBOTDOG, 5:9] = True
    with torch.no_grad():
        output = model(
            **inputs,
            synthetic_occlusion=receiver,
            coarse_missing_mask=coarse_mask,
            fine_missing_mask=fine_mask,
            route_visibility=~receiver,
        )
    assert output["jepa_token_mask"][0, ROBOTDOG].sum().item() == 4
    torch.testing.assert_close(output["coarse_missing_mask"], coarse_mask)
    torch.testing.assert_close(output["fine_missing_mask"], fine_mask)

    changed_inputs = {key: value for key, value in inputs.items()}
    changed_inputs["fine_tokens"] = inputs["fine_tokens"].clone()
    changed_inputs["fine_tokens"][0, ROBOTDOG, 5:9, 0] += 1000.0
    with torch.no_grad():
        changed = model(
            **changed_inputs,
            synthetic_occlusion=receiver,
            coarse_missing_mask=coarse_mask,
            fine_missing_mask=fine_mask,
            route_visibility=~receiver,
        )
    # Cooperative receiver cannot observe the replaced ROI tokens, while the
    # isolated clean self row intentionally still can.
    torch.testing.assert_close(
        output["dog_cooperative_waypoints"],
        changed["dog_cooperative_waypoints"],
        atol=0,
        rtol=0,
    )
    assert not torch.allclose(output["dog_self_waypoints"], changed["dog_self_waypoints"])


def test_verify_prompt_and_query_have_an_independent_gradient_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _build_tiny_model(monkeypatch).eval()
    inputs = _model_inputs(batch_size=1)
    output = model(
        **inputs,
        route_visibility=torch.ones(1, 2, dtype=torch.bool),
    )
    output["target_match_logits"].sum().backward()
    assert model.self_verify_tokens.grad is not None
    assert model.self_verify_tokens.grad.abs().sum().item() > 0.0
    assert model.self_act_tokens.grad is None or torch.count_nonzero(
        model.self_act_tokens.grad
    ).item() == 0
    assert all(parameter.grad is None for parameter in model.self_planners.parameters())


def test_llm_target_verifier_gates_valid_yolo_detection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _build_tiny_model(monkeypatch).eval()
    inputs = _model_inputs(batch_size=1)
    with torch.no_grad():
        # 将 VERIFY 校准偏置与候选相对分数隔离。
        for parameter in model.candidate_matcher.fusion.parameters():
            parameter.zero_()
        model.candidate_matcher.temperature.fill_(math.log(100.0))
        for head in model.target_match_heads:
            for parameter in head.parameters():
                parameter.zero_()
            head[-1].bias.fill_(10.0)
        accepted = model(**inputs)
        assert accepted["yolo_visible"].all()
        assert accepted["observed_visible"].all()
        assert accepted["routing_mode"].eq(ROUTE_SELF).all()

        for head in model.target_match_heads:
            head[-1].bias.fill_(-10.0)
        rejected = model(**inputs)
        assert rejected["yolo_visible"].all()
        assert not rejected["observed_visible"].any()
        assert rejected["routing_mode"].eq(ROUTE_BELIEF).all()


def test_top8_candidates_are_matched_inside_existing_two_llm_flows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _build_tiny_model(monkeypatch).eval()
    inputs = _model_inputs(batch_size=2)
    candidates = torch.zeros(2, 2, 8, 6)
    valid = torch.zeros(2, 2, 8, dtype=torch.bool)
    for index in range(5):
        candidates[:, :, index] = torch.tensor(
            [0.1 + index * 0.15, 0.5, 0.1, 0.3, 0.9 - index * 0.05, 1.0]
        )
        valid[:, :, index] = True
    output = model(**inputs, candidate_feat=candidates, candidate_valid=valid)
    assert output["candidate_match_logits"].shape == (2, 2, 8)
    assert output["candidate_class_logits"].shape == (2, 2, 8)
    assert output["candidate_match_probability"].shape == (2, 2, 8)
    assert "candidate_null_probability" not in output
    assert output["selected_candidate_context"].shape == (2, 2, 24)
    assert output["grounded_self_action_context"].shape == (2, 2, 24)
    assert output["candidate_selected_index"].shape == (2, 2)
    assert output["selected_detection_feat"].shape == (2, 2, 6)
    # One packed self call plus the existing cooperative call; top-K matching
    # must not introduce a third language-model forward.
    assert model.llm.call_batch_sizes == [4, 2]


def test_self_streams_do_not_leak_the_other_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _build_tiny_model(monkeypatch).eval()
    inputs = _model_inputs()
    with torch.no_grad():
        baseline = model(**inputs)

        dog_changed = {key: value for key, value in inputs.items()}
        dog_changed["coarse_tokens"] = inputs["coarse_tokens"].clone()
        dog_changed["fine_tokens"] = inputs["fine_tokens"].clone()
        dog_changed["detection_feat"] = inputs["detection_feat"].clone()
        dog_changed["perception_grid"] = inputs["perception_grid"].clone()
        dog_changed["agent_poses"] = inputs["agent_poses"].clone()
        dog_changed["coarse_tokens"][:, ROBOTDOG] += 20.0
        dog_changed["fine_tokens"][:, ROBOTDOG] -= 15.0
        dog_changed["detection_feat"][:, ROBOTDOG, :4] = 0.05
        dog_changed["perception_grid"][:, ROBOTDOG] = 0.0
        dog_changed["agent_poses"][:, ROBOTDOG, :2] += 9.0
        changed = model(**dog_changed)
        torch.testing.assert_close(
            baseline["drone_self_waypoints"], changed["drone_self_waypoints"], atol=0, rtol=0
        )
        assert not torch.allclose(
            baseline["drone_cooperative_waypoints"],
            changed["drone_cooperative_waypoints"],
        )

        drone_changed = {key: value for key, value in inputs.items()}
        drone_changed["coarse_tokens"] = inputs["coarse_tokens"].clone()
        drone_changed["fine_tokens"] = inputs["fine_tokens"].clone()
        drone_changed["detection_feat"] = inputs["detection_feat"].clone()
        drone_changed["perception_grid"] = inputs["perception_grid"].clone()
        drone_changed["coarse_tokens"][:, DRONE] -= 17.0
        drone_changed["fine_tokens"][:, DRONE] += 13.0
        drone_changed["detection_feat"][:, DRONE, :4] = 0.95
        drone_changed["perception_grid"][:, DRONE] = 1.0
        changed = model(**drone_changed)
        torch.testing.assert_close(
            baseline["dog_self_waypoints"], changed["dog_self_waypoints"], atol=0, rtol=0
        )


def test_jointly_masked_receiver_visual_cannot_change_cooperative_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _build_tiny_model(monkeypatch).eval()
    inputs = _model_inputs(batch_size=1)
    synthetic = torch.tensor([[False, True]])
    with torch.no_grad():
        baseline = model(
            **inputs,
            synthetic_occlusion=synthetic,
            route_visibility=~synthetic,
        )
        changed_inputs = {key: value for key, value in inputs.items()}
        for name in ("coarse_tokens", "fine_tokens", "detection_feat", "perception_grid"):
            changed_inputs[name] = inputs[name].clone()
        changed_inputs["coarse_tokens"][:, ROBOTDOG] += 1000.0
        changed_inputs["fine_tokens"][:, ROBOTDOG] -= 1000.0
        changed_inputs["detection_feat"][:, ROBOTDOG] = 1000.0
        changed_inputs["perception_grid"][:, ROBOTDOG] = 1000.0
        changed = model(
            **changed_inputs,
            synthetic_occlusion=synthetic,
            route_visibility=~synthetic,
        )
    torch.testing.assert_close(
        baseline["dog_cooperative_waypoints"],
        changed["dog_cooperative_waypoints"],
        atol=0,
        rtol=0,
    )
    assert not torch.allclose(
        baseline["dog_self_waypoints"], changed["dog_self_waypoints"]
    )


def test_drone_self_gradient_does_not_reach_dog_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _build_tiny_model(monkeypatch).eval()
    inputs = _model_inputs(batch_size=1)
    inputs["coarse_tokens"] = inputs["coarse_tokens"].requires_grad_()
    inputs["fine_tokens"] = inputs["fine_tokens"].requires_grad_()
    output = model(**inputs)
    output["drone_self_waypoints"].square().sum().backward()
    assert inputs["coarse_tokens"].grad[:, DRONE].abs().sum().item() > 0
    assert inputs["fine_tokens"].grad[:, DRONE].abs().sum().item() > 0
    torch.testing.assert_close(
        inputs["coarse_tokens"].grad[:, ROBOTDOG],
        torch.zeros_like(inputs["coarse_tokens"].grad[:, ROBOTDOG]),
    )
    torch.testing.assert_close(
        inputs["fine_tokens"].grad[:, ROBOTDOG],
        torch.zeros_like(inputs["fine_tokens"].grad[:, ROBOTDOG]),
    )


def test_two_synthetic_occlusions_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _build_tiny_model(monkeypatch)
    inputs = _model_inputs(batch_size=1)
    with pytest.raises(ValueError, match="At most one view"):
        model(**inputs, synthetic_occlusion=torch.tensor([[True, True]]))


def test_tiny_valid_bbox_still_masks_one_jepa_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _build_tiny_model(monkeypatch)
    mask = model._bbox_grid_mask(
        torch.tensor([[0.01, 0.01, 0.001, 0.001]]),
        torch.tensor([True]),
        token_count=16,
        expand_ratio=3.0,
    )
    assert mask.shape == (1, 16)
    assert mask.sum().item() == 1


def test_dataset_occlusion_draw_is_curriculum_gated_and_mutually_exclusive() -> None:
    dataset = object.__new__(AirGroundV3JsonDataset)
    dataset.cfg = AirGroundV3DataConfig(
        train_json="unused",
        synthetic_drone_occlusion_prob=0.5,
        synthetic_dog_occlusion_prob=0.5,
        deterministic_occlusion=True,
    )
    selected = 0
    for index in range(50):
        value = dataset._sample_synthetic_occlusion(
            {"episode_id": "episode", "step_index": index}, index, True
        )
        assert value.dtype == torch.bool
        assert value.sum().item() <= 1
        selected += int(value.sum())
    assert 0 < selected < 50


def test_rotating_temporal_stride_covers_every_episode_frame() -> None:
    class _Dataset:
        cfg = SimpleNamespace(temporal_stride=3)
        _index = [
            *(("episode_a", index) for index in range(6)),
            *(("episode_b", index) for index in range(6)),
        ]

        def __len__(self):
            return len(self._index)

    sampler = RotatingTemporalStrideDistributedSampler(
        _Dataset(), block_size=2, shuffle=False
    )
    selected = []
    for epoch in range(3):
        sampler.set_epoch(epoch)
        indices = list(sampler)
        assert len(indices) == 4
        selected.extend(indices)
    assert sorted(selected) == list(range(12))


class _FakeV3Model(nn.Module):
    def __init__(self, self_waypoints: torch.Tensor, candidates: torch.Tensor):
        super().__init__()
        self.self_parameter = nn.Parameter(self_waypoints)
        self.candidate_parameter = nn.Parameter(candidates)
        self.mode_logits = nn.Parameter(
            torch.zeros(candidates.size(0), 2, candidates.size(2))
        )
        self.target_match_parameter = nn.Parameter(
            torch.zeros(candidates.size(0), 2)
        )
        self.target_belief_parameter = nn.Parameter(
            torch.zeros(candidates.size(0), 2, 5)
        )
        self.register_buffer("alpha_task", torch.ones(1, 1, 3))

    def forward(self, **kwargs):
        del kwargs
        batch_size, _, modes, waypoints, _ = self.candidate_parameter.shape
        selected = self.candidate_parameter[:, :, 0]
        zero_tokens = self.self_parameter.new_zeros(batch_size, 2, waypoints, 8)
        zero_mask = torch.zeros(
            batch_size, 2, waypoints, dtype=torch.bool, device=self.self_parameter.device
        )
        return {
            "self_waypoints": self.self_parameter,
            "cooperative_candidates": self.candidate_parameter,
            "cooperative_mode_logits": self.mode_logits,
            "cooperative_waypoints": selected,
            "waypoints": torch.zeros_like(self.self_parameter),
            "target_match_logits": self.target_match_parameter,
            "target_match_probability": torch.sigmoid(self.target_match_parameter),
            "jepa_prediction_tokens": zero_tokens,
            "jepa_teacher_tokens": zero_tokens,
            "jepa_token_mask": zero_mask,
            "target_belief": self.target_belief_parameter,
            "jepa_uncertainty_logit": self.self_parameter.new_zeros(batch_size, 2),
        }


class _FakeGroundingV3Model(_FakeV3Model):
    def __init__(self):
        super().__init__(
            torch.zeros(2, 2, 4, 3),
            torch.zeros(2, 2, 2, 4, 3),
        )
        self.candidate_class_parameter = nn.Parameter(torch.zeros(2, 2, 2))

    def forward(self, **kwargs):
        output = super().forward(**kwargs)
        output["candidate_class_logits"] = self.candidate_class_parameter
        output["candidate_selected_index"] = self.candidate_class_parameter.argmax(-1)
        output["candidate_selected_probability"] = torch.sigmoid(
            self.candidate_class_parameter
        ).max(-1).values
        output["candidate_selected_accepted"] = torch.ones(2, 2, dtype=torch.bool)
        return output


class _FakeJepaV3Model(_FakeV3Model):
    """Loss-only model with a nontrivial clean-pose JEPA teacher."""

    def __init__(self):
        super().__init__(
            torch.zeros(2, 2, 4, 3),
            torch.zeros(2, 2, 2, 4, 3),
        )
        prediction = torch.zeros(2, 2, 4, 2)
        prediction[..., 1] = 1.0
        self.jepa_prediction_parameter = nn.Parameter(prediction)
        teacher = torch.zeros(2, 2, 4, 2)
        teacher[..., 0] = 1.0
        self.register_buffer("jepa_teacher_target", teacher)

    def forward(self, **kwargs):
        fine_missing_mask = kwargs["fine_missing_mask"].bool()
        output = super().forward(**kwargs)
        output["jepa_prediction_tokens"] = self.jepa_prediction_parameter
        output["jepa_teacher_tokens"] = self.jepa_teacher_target
        output["jepa_token_mask"] = fine_missing_mask
        return output


def _loss_batch(n_waypoints: int = 4) -> dict:
    batch_size = 2
    zeros = torch.zeros(batch_size, 2, n_waypoints, 3)
    return {
        "instruction": ["Follow the person."] * batch_size,
        "coarse_tokens": torch.zeros(batch_size, 2, 4, 3),
        "coarse_tidx": torch.zeros(batch_size, 2, 4, dtype=torch.long),
        "fine_tokens": torch.zeros(batch_size, 2, 4, 3),
        "fine_tidx": torch.zeros(batch_size, 2, 4, dtype=torch.long),
        "detection_feat": torch.ones(batch_size, 2, 6),
        "candidate_feat": torch.ones(batch_size, 2, 2, 6),
        "candidate_valid": torch.ones(batch_size, 2, 2, dtype=torch.bool),
        "candidate_iou": torch.tensor(
            [[[0.1, 0.8], [0.9, 0.1]], [[0.7, 0.2], [0.1, 0.6]]]
        ),
        "candidate_match_label": torch.tensor(
            [[[False, True], [True, False]], [[True, False], [False, True]]]
        ),
        "candidate_match_valid": torch.ones(batch_size, 2, 2, dtype=torch.bool),
        "perception_grid": torch.zeros(batch_size, 2, 2, 2, 4),
        "perception_cache_valid": torch.ones(batch_size, 2, dtype=torch.bool),
        "yolo_target_iou": torch.ones(batch_size, 2),
        "target_match_label": torch.ones(batch_size, 2, dtype=torch.bool),
        "target_match_valid": torch.ones(batch_size, 2, dtype=torch.bool),
        "synthetic_false_positive": torch.zeros(batch_size, 2, dtype=torch.bool),
        "agent_poses": torch.tensor(
            [[[1.0, 0.0, 0.0, 1.0], [-1.0, 0.0, 0.0, 1.0]]]
        ).expand(batch_size, -1, -1).clone(),
        "receiver_pose_perturbation": torch.tensor(
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.5]], [[1.0, 0.0, 0.5], [0.0, 0.0, 0.0]]]
        ),
        "synthetic_occlusion": torch.tensor([[False, True], [True, False]]),
        "receiver_corruption_mode": torch.tensor(
            [[0, CORRUPTION_CURRENT_FULL], [CORRUPTION_CURRENT_FULL, 0]]
        ),
        "receiver_history_mask_frames": torch.zeros(batch_size, 2, dtype=torch.long),
        "coarse_missing_mask": torch.tensor(
            [[[False] * 4, [False] * 4], [[False] * 4, [False] * 4]]
        ),
        "fine_missing_mask": torch.tensor(
            [[[False] * 4, [True] * 4], [[True] * 4, [False] * 4]]
        ),
        "effective_visible": torch.tensor([[True, False], [False, True]]),
        # Cooperative corruption never removes either clean self row.
        "self_target": torch.ones(batch_size, 2, dtype=torch.bool),
        "cooperative_target": torch.tensor([[False, True], [True, False]]),
        "jepa_valid": torch.zeros(batch_size, 2, dtype=torch.bool),
        "target_pose": torch.zeros(batch_size, 2, 5),
        "target_pose_valid": torch.ones(batch_size, 2, dtype=torch.bool),
        "visible": torch.ones(batch_size, 2),
        "bbox_valid_mask": torch.ones(batch_size, 2, dtype=torch.bool),
        "yaw_hist": torch.zeros(batch_size, 2, 1),
        "yaw_curr": torch.zeros(batch_size, 2, 1),
        "waypoints": zeros,
        # Cooperative candidates are supervised in the receiver's perturbed
        # local frame.  The default fixture uses the zero clean trajectory so
        # existing self/VERIFY/JEPA tests remain focused on their own paths.
        "cooperative_waypoints": zeros.clone(),
        "valid_mask": torch.ones(batch_size, 2, n_waypoints, dtype=torch.bool),
        "dt": torch.full((batch_size,), 0.1),
    }


def _loss_cfg() -> AirGroundV3TrainConfig:
    return AirGroundV3TrainConfig(
        train_json="unused",
        n_waypoints=4,
        num_modes=2,
        beta_nav=1.0,
        beta_cooperative_waypoint=1.0,
        beta_mode_classification=0.0,
        beta_jepa=0.0,
        beta_target_belief=0.0,
        beta_target_match=0.0,
        beta_uncertainty=0.0,
        beta_smoothness=0.0,
        beta_kinematics=0.0,
        beta_diversity=0.0,
        beta_obstacle=0.0,
        drone_loss_weight=1.0,
        dog_loss_weight=1.0,
        nav_loss_type="mse",
    )


def test_jepa_loss_supervises_masked_receiver_against_clean_pose_teacher() -> None:
    batch = _loss_batch()
    batch["jepa_valid"] = batch["synthetic_occlusion"].clone()
    cfg = _loss_cfg()
    cfg.beta_nav = 0.0
    cfg.beta_cooperative_waypoint = 0.0
    cfg.beta_jepa = 1.0
    model = _FakeJepaV3Model()

    loss, metrics = forward_airground_v3_loss(
        model, batch, cfg, torch.device("cpu")
    )

    assert metrics["loss_jepa"].item() > 0.9
    torch.testing.assert_close(loss, metrics["loss_jepa"])
    loss.backward()
    grad = model.jepa_prediction_parameter.grad
    assert grad is not None
    active_receiver_grad = grad[0, ROBOTDOG].abs().sum() + grad[1, DRONE].abs().sum()
    inactive_source_grad = grad[0, DRONE].abs().sum() + grad[1, ROBOTDOG].abs().sum()
    assert active_receiver_grad.item() > 0.0
    assert inactive_source_grad.item() == pytest.approx(0.0)


def test_synthetic_receiver_keeps_clean_self_supervision() -> None:
    batch = _loss_batch()
    self_waypoints = torch.zeros(2, 2, 4, 3)
    # Put error only on the synthetic receivers. It must still contribute to
    # clean self loss even though those rows are corrupted in the cooperative flow.
    self_waypoints[0, ROBOTDOG, :, 0] = 1.0
    self_waypoints[1, DRONE, :, 0] = 1.0
    candidates = torch.full((2, 2, 2, 4, 3), 100.0)
    # Only (sample0, dog) and (sample1, drone) cooperative branches are active.
    candidates[0, ROBOTDOG, 0] = 0.0
    candidates[1, DRONE, 1] = 0.0
    model = _FakeV3Model(self_waypoints, candidates)
    loss, metrics = forward_airground_v3_loss(
        model, batch, _loss_cfg(), torch.device("cpu")
    )
    assert loss.item() > 0.0
    assert metrics["loss_self"].item() > 0.0
    torch.testing.assert_close(metrics["loss_cooperative"], torch.tensor(0.0))
    loss.backward()
    assert model.self_parameter.grad is not None
    assert model.candidate_parameter.grad is not None
    assert model.self_parameter.grad[0, ROBOTDOG].abs().sum().item() > 0.0
    assert model.self_parameter.grad[1, DRONE].abs().sum().item() > 0.0


def test_active_cooperative_candidates_are_supervised() -> None:
    batch = _loss_batch()
    self_waypoints = torch.zeros(2, 2, 4, 3)
    candidates = torch.zeros(2, 2, 2, 4, 3)
    candidates[0, ROBOTDOG, :, :, 0] = 1.0
    candidates[1, DRONE, :, :, 0] = 1.0
    model = _FakeV3Model(self_waypoints, candidates)
    loss, metrics = forward_airground_v3_loss(
        model, batch, _loss_cfg(), torch.device("cpu")
    )
    assert loss.item() > 0.0
    assert metrics["loss_cooperative"].item() > 0.0


def test_cooperative_robotdog_y_position_has_loss_and_gradient() -> None:
    batch = _loss_batch()
    # Match the real dataset: waypoint 0 is an unsupervised structural origin.
    batch["valid_mask"][..., 0] = False
    candidates = torch.zeros(2, 2, 2, 4, 3)
    # Isolate a y-position error on the active RobotDog receiver only.
    candidates[0, ROBOTDOG, :, 1:, 1] = 1.0
    model = _FakeV3Model(torch.zeros(2, 2, 4, 3), candidates)
    cfg = _loss_cfg()
    cfg.beta_nav = 0.0

    loss, metrics = forward_airground_v3_loss(
        model, batch, cfg, torch.device("cpu")
    )

    assert cfg.dog_lateral_loss_weight == pytest.approx(1.0)
    assert metrics["loss_cooperative"].item() > 0.0
    loss.backward()
    gradient = model.candidate_parameter.grad
    assert gradient is not None
    assert gradient[0, ROBOTDOG, :, 1:, 1].abs().sum().item() > 0.0
    # No x/yaw error was introduced in the active RobotDog candidate.
    assert gradient[0, ROBOTDOG, :, 1:, 0].abs().sum().item() == pytest.approx(0.0)
    assert gradient[0, ROBOTDOG, :, 1:, 2].abs().sum().item() == pytest.approx(0.0)


def test_kinematics_regularizes_origin_to_first_future_waypoint() -> None:
    candidates = torch.zeros(1, 2, 1, 3, 3)
    candidates[0, ROBOTDOG, 0, 1:, 0] = 1.0
    valid_mask = torch.tensor([[[False, True, True], [False, True, True]]])
    cfg = _loss_cfg()
    cfg.dog_max_speed_mps = 2.5

    value = _kinematics_per_sample(
        candidates,
        valid_mask,
        torch.tensor([0.1]),
        cfg,
    )

    # Only origin->waypoint1 violates 2.5 m/s; this would be zero if the
    # structural origin were incorrectly excluded by valid_mask[0]==False.
    assert value[0, ROBOTDOG].item() > 0.0


def test_cooperative_loss_uses_feasible_receiver_recovery_waypoints() -> None:
    batch = _loss_batch()
    batch["valid_mask"][..., 0] = False
    batch["waypoints"][0, ROBOTDOG, 1:, 0] = 1.0
    batch["waypoints"][1, DRONE, 1:, 0] = 1.0
    # A ninety-degree receiver relocation produces a gradual, rate-limited
    # recovery target. The source stays clean and waypoint zero stays at origin.
    batch["receiver_pose_perturbation"] = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, math.pi / 2.0]],
            [[0.0, 0.0, math.pi / 2.0], [0.0, 0.0, 0.0]],
        ]
    )
    batch["cooperative_waypoints"] = torch.stack(
        [
            build_cooperative_waypoint_targets(
                batch["waypoints"][index], batch["synthetic_occlusion"][index],
                batch["receiver_pose_perturbation"][index],
                dt=0.1,
                drone_max_speed_mps=2.5,
                dog_max_speed_mps=2.5,
                drone_max_yaw_rate_rps=1.5,
                dog_max_yaw_rate_rps=1.5,
            )
            for index in range(batch["waypoints"].size(0))
        ]
    )
    candidates = torch.full((2, 2, 2, 4, 3), 50.0)
    candidates[0, ROBOTDOG, 0] = batch["cooperative_waypoints"][0, ROBOTDOG]
    candidates[1, DRONE, 1] = batch["cooperative_waypoints"][1, DRONE]
    model = _FakeV3Model(torch.zeros(2, 2, 4, 3), candidates)
    cfg = _loss_cfg()
    cfg.beta_nav = 0.0
    loss, metrics = forward_airground_v3_loss(
        model, batch, cfg, torch.device("cpu")
    )
    torch.testing.assert_close(metrics["loss_cooperative"], torch.tensor(0.0))
    torch.testing.assert_close(loss, torch.tensor(0.0))

    # Predicting the unrelocated clean trajectory is wrong and must
    # receive a non-zero cooperative gradient.
    clean_frame_candidates = candidates.detach().clone()
    clean_frame_candidates[0, ROBOTDOG, 0] = batch["waypoints"][0, ROBOTDOG]
    clean_frame_candidates[1, DRONE, 1] = batch["waypoints"][1, DRONE]
    clean_frame_model = _FakeV3Model(
        torch.zeros(2, 2, 4, 3), clean_frame_candidates
    )
    clean_loss, clean_metrics = forward_airground_v3_loss(
        clean_frame_model, batch, cfg, torch.device("cpu")
    )
    assert clean_metrics["loss_cooperative"].item() > 0.0
    assert clean_loss.item() > 0.0


def test_synthetic_receiver_keeps_clean_target_pose_supervision() -> None:
    batch = _loss_batch()
    batch["target_pose"][0, ROBOTDOG, 0] = 1.0
    batch["target_pose"][1, DRONE, 0] = -1.0
    model = _FakeV3Model(
        torch.zeros(2, 2, 4, 3),
        torch.zeros(2, 2, 2, 4, 3),
    )
    cfg = _loss_cfg()
    cfg.beta_nav = 0.0
    cfg.beta_cooperative_waypoint = 0.0
    cfg.beta_target_belief = 1.0

    loss, metrics = forward_airground_v3_loss(
        model, batch, cfg, torch.device("cpu")
    )

    assert metrics["loss_belief"].item() > 0.0
    torch.testing.assert_close(loss, metrics["loss_belief"])
    loss.backward()
    grad = model.target_belief_parameter.grad
    assert grad is not None
    assert grad[0, ROBOTDOG].abs().sum().item() > 0.0
    assert grad[1, DRONE].abs().sum().item() > 0.0
    assert grad[0, DRONE].abs().sum().item() == pytest.approx(0.0)
    assert grad[1, ROBOTDOG].abs().sum().item() == pytest.approx(0.0)


def test_binary_grounding_balances_positive_and_negative_bce() -> None:
    batch = _loss_batch()
    # 没有匹配 Top-K 候选的视角直接排除，不分配第 9 个 NULL 类。
    batch["candidate_iou"][1, ROBOTDOG] = torch.tensor([0.1, 0.2])
    batch["candidate_match_label"][1, ROBOTDOG] = False
    model = _FakeGroundingV3Model()
    cfg = _loss_cfg()
    cfg.beta_nav = 0.0
    cfg.beta_cooperative_waypoint = 0.0
    cfg.beta_target_match = 1.0

    loss, metrics = forward_airground_v3_loss(
        model, batch, cfg, torch.device("cpu")
    )
    torch.testing.assert_close(metrics["loss_target_match"], torch.tensor(2.0).log())
    loss.backward()
    grad = model.candidate_class_parameter.grad
    assert grad is not None
    # sample0/drone 的候选 1 是唯一的最大 IoU 正样本。
    assert grad[0, DRONE, 1] < 0
    assert grad[0, DRONE, 0] > 0
    # 未匹配视角不产生候选 BCE 监督。
    assert torch.count_nonzero(grad[1, ROBOTDOG]).item() == 0


def test_binary_grounding_all_unmatched_rows_stay_finite() -> None:
    batch = _loss_batch()
    batch["candidate_iou"].fill_(0.1)
    batch["candidate_match_label"].zero_()
    model = _FakeGroundingV3Model()
    with torch.no_grad():
        model.candidate_class_parameter.fill_(torch.finfo(torch.float32).min)
    cfg = _loss_cfg()
    cfg.beta_nav = 0.0
    cfg.beta_cooperative_waypoint = 0.0
    cfg.beta_target_match = 1.0

    loss, metrics = forward_airground_v3_loss(
        model, batch, cfg, torch.device("cpu")
    )
    assert torch.isfinite(loss)
    torch.testing.assert_close(metrics["loss_target_match"], torch.tensor(0.0))
    loss.backward()
    assert model.candidate_class_parameter.grad is not None
    assert torch.count_nonzero(model.candidate_class_parameter.grad).item() == 0


def test_target_match_loss_supervises_verifier_logits() -> None:
    batch = _loss_batch()
    batch["target_match_label"] = torch.tensor(
        [[True, False], [False, True]], dtype=torch.bool
    )
    model = _FakeV3Model(
        torch.zeros(2, 2, 4, 3),
        torch.zeros(2, 2, 2, 4, 3),
    )
    cfg = _loss_cfg()
    cfg.beta_nav = 0.0
    cfg.beta_cooperative_waypoint = 0.0
    cfg.beta_target_match = 1.0
    loss, metrics = forward_airground_v3_loss(
        model, batch, cfg, torch.device("cpu")
    )
    torch.testing.assert_close(loss, torch.tensor(2.0).log())
    torch.testing.assert_close(metrics["loss_target_match"], torch.tensor(2.0).log())
    loss.backward()
    assert model.target_match_parameter.grad is not None
    assert model.target_match_parameter.grad.abs().sum().item() > 0.0
    # The two synthetic receivers retain VERIFY supervision from their clean
    # self rows: sample0 dog and sample1 drone.
    assert model.target_match_parameter.grad[0, ROBOTDOG].abs().item() > 0.0
    assert model.target_match_parameter.grad[1, DRONE].abs().item() > 0.0


def test_v3_config_rejects_v1_and_unsafe_obstacle_loss(tmp_path: Path) -> None:
    v1 = tmp_path / "v1.yaml"
    v1.write_text("architecture: airground_cooperative_closed_loop_v2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="requires architecture"):
        load_config(v1)
    v2 = tmp_path / "v2.yaml"
    v2.write_text("architecture: airground_three_stream_cooperative_v2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="requires architecture"):
        load_config(v2)
    cfg = AirGroundV3TrainConfig(
        train_json="unused",
        beta_obstacle=1.0,
        enable_projected_obstacle_loss=False,
    )
    with pytest.raises(ValueError, match="validated image-to-local-ground projection"):
        apply_airground_v3_defaults(cfg)

    data_defaults = AirGroundV3DataConfig(train_json="unused")
    assert (
        data_defaults.synthetic_drone_occlusion_prob
        + data_defaults.synthetic_dog_occlusion_prob
        == pytest.approx(1.0)
    )
    train_defaults = AirGroundV3TrainConfig(train_json="unused")
    assert (
        train_defaults.train_synthetic_drone_occlusion_prob
        + train_defaults.train_synthetic_dog_occlusion_prob
        == pytest.approx(1.0)
    )
    assert (
        train_defaults.val_synthetic_drone_occlusion_prob
        + train_defaults.val_synthetic_dog_occlusion_prob
        == pytest.approx(1.0)
    )

    repository_v3 = Path(__file__).resolve().parents[1] / "config/airground_cooperative_tracking_v3.yaml"
    v3_cfg = load_config(repository_v3)
    assert "all aerial-view person candidates" in v3_cfg.drone_target_verification_prompt
    assert "initial-target rule" in v3_cfg.dog_target_verification_prompt
    assert "Score every candidate independently" in v3_cfg.drone_target_verification_prompt

    exp9_cfg = load_config(
        Path(__file__).resolve().parents[1] / "exp9_4/config/train_s5_e10.yaml"
    )
    assert exp9_cfg.use_target_reference is False
    assert exp9_cfg.train_temporal_stride == 5
    assert exp9_cfg.epochs == 10
    assert v3_cfg.train_temporal_stride == 5
    assert v3_cfg.epochs == 10
    assert v3_cfg.batch_size == 48
    assert v3_cfg.grad_accum_steps == 2
    assert len(v3_cfg.receiver_corruption_curriculum) == 3
    assert v3_cfg.receiver_corruption_curriculum[-1][
        "pose_translation_max_m"
    ] == pytest.approx(0.5)
    assert v3_cfg.receiver_corruption_curriculum[-1][
        "pose_yaw_max_deg"
    ] == pytest.approx(30.0)
    assert v3_cfg.pose_position_scale_m == pytest.approx(20.0)
    assert v3_cfg.dog_lateral_loss_weight == pytest.approx(1.0)
    assert v3_cfg.drone_max_speed_mps == pytest.approx(2.5)
    assert v3_cfg.dog_max_speed_mps == pytest.approx(2.5)
    assert v3_cfg.out_dir.endswith(
        "airground_three_stream_cooperative_v3_receiver_target_qwen06b"
    )
    assert v3_cfg.cooperative_target_frame_version == "receiver_feasible_recovery_v1"
    assert v3_cfg.receiver_corruption_version == "roi_temporal_curriculum_v1"
    assert v3_cfg.relative_pose_version == "directed_receiver_local_v1"
    assert (
        v3_cfg.train_synthetic_drone_occlusion_prob
        + v3_cfg.train_synthetic_dog_occlusion_prob
        == pytest.approx(1.0)
    )
    assert (
        v3_cfg.val_synthetic_drone_occlusion_prob
        + v3_cfg.val_synthetic_dog_occlusion_prob
        == pytest.approx(1.0)
    )
def test_v3_checkpoint_loader_rejects_v1_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        model_module.AutoModel,
        "from_pretrained",
        lambda *args, **kwargs: _TinyCausalLLM(),
    )
    monkeypatch.setattr(
        model_module.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: _TinyTokenizer(),
    )
    cfg = _loss_cfg()
    cfg.vision_feat_dim = 12
    cfg.coop_hidden_dim = 16
    cfg.coop_num_heads = 4
    cfg.coop_encoder_layers = 1
    cfg.coop_decoder_layers = 1
    cfg.jepa_hidden_dim = 16
    cfg.jepa_num_heads = 4
    cfg.jepa_decoder_layers = 1
    model = build_airground_v3_model(cfg)
    with pytest.raises(RuntimeError, match="non-V3/incompatible checkpoint"):
        model.load_state_dict({"act_token_guide_dog": torch.zeros(1)})
