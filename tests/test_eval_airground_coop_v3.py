from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import eval_airground_coop_v3 as eval_v3_module
import eval_airground_coop_v3_server as eval_v3_server_module
from eval_unrealzoo_multi_agent import _rollout_future_waypoint_segment
from eval_airground_coop_v3 import (
    AirGroundCoopV3Planner,
    BBoxMotionController,
    _agent_poses,
    _llm_verified_boxes,
    _load_payload,
    _project_robotdog_waypoints_to_nonholonomic,
    _runtime_argv,
)
from eval_airground_v3_runtime import AirGroundV3RuntimePlanner
from eval_airground_coop_v3_server import SESSION_FIELDS, SharedAirGroundV3Server
from model_airground_coop_v3 import (
    ROUTE_BELIEF,
    ROUTE_COOPERATIVE,
    ROUTE_SEARCH,
    ROUTE_SELF,
    AirGroundVisibilityRouter,
)


def _bbox_args(**overrides):
    values = {
        "bbox_motion_control": True,
        "bbox_motion_min_confidence": 0.25,
        "bbox_motion_ema_alpha": 0.20,
        "bbox_motion_min_valid_frames": 2,
        "bbox_motion_min_shrink_frames": 3,
        "bbox_motion_height_tolerance_ratio": 0.20,
        "bbox_motion_height_response_ratio": 0.50,
        "drone_bbox_height_normal": 0.150,
        "drone_bbox_height_far": 0.120,
        "drone_bbox_max_speed_gain": 1.50,
        "drone_bbox_min_speed_gain": 0.50,
        "drone_bbox_max_yaw_residual": 0.12,
        "robotdog_bbox_height_normal": 0.220,
        "robotdog_bbox_height_far": 0.160,
        "robotdog_bbox_max_speed_gain": 1.50,
        "robotdog_bbox_min_speed_gain": 0.50,
        "robotdog_bbox_max_yaw_residual": 0.25,
        "drone_max_speed": 1.20,
        "robotdog_max_speed": 1.20,
        "drone_max_yaw_rate": 0.40,
        "robotdog_max_yaw_rate": 1.00,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _routing_output(target_match=(1.0, 1.0)):
    self_waypoints = torch.zeros(1, 2, 4, 3)
    cooperative_waypoints = torch.zeros_like(self_waypoints)
    self_waypoints[0, 0, :, 0] = 1.0
    self_waypoints[0, 1, :, 0] = 2.0
    cooperative_waypoints[0, 0, :, 0] = 10.0
    cooperative_waypoints[0, 1, :, 0] = 20.0
    return {
        "self_waypoints": self_waypoints,
        "cooperative_waypoints": cooperative_waypoints,
        "target_match_probability": torch.tensor([target_match]),
    }


def _planner_for_routing() -> AirGroundCoopV3Planner:
    planner = AirGroundCoopV3Planner.__new__(AirGroundCoopV3Planner)
    planner.visibility_router = AirGroundVisibilityRouter(
        enter_confidence=0.35,
        exit_confidence=0.20,
        target_match_enter_confidence=0.50,
        target_match_exit_confidence=0.35,
        visible_confirm_frames=2,
        invisible_confirm_frames=2,
        belief_hold_frames=1,
    )
    planner.last_navigation_waypoints = None
    planner.search_center_yaw_rad = None
    planner.search_target_direction = np.ones(2, dtype=np.float32)
    return planner


def test_rollout_future_waypoint_segment_rebases_straight_path():
    waypoints = np.zeros((2, 4, 3), dtype=np.float32)
    waypoints[:, :, 0] = np.arange(4, dtype=np.float32)

    rebased = _rollout_future_waypoint_segment(waypoints, 1)

    np.testing.assert_allclose(rebased[:, :, 0], [[0, 1, 2, 2], [0, 1, 2, 2]])
    np.testing.assert_allclose(rebased[:, :, 1:], 0.0)


def test_rollout_future_waypoint_segment_rotates_into_anchor_frame():
    waypoints = np.zeros((2, 3, 3), dtype=np.float32)
    waypoints[:, 1] = [1.0, 0.0, np.pi / 2]
    waypoints[:, 2] = [1.0, 1.0, np.pi / 2]

    rebased = _rollout_future_waypoint_segment(waypoints, 1)

    np.testing.assert_allclose(rebased[:, 0, :3], 0.0, atol=1e-6)
    np.testing.assert_allclose(rebased[:, 1, :2], [[1, 0], [1, 0]], atol=1e-6)
    np.testing.assert_allclose(rebased[:, 1, 2], 0.0, atol=1e-6)


def _agent_pose_tokens(yaw_degrees=(0.0, 0.0)) -> torch.Tensor:
    yaw = torch.deg2rad(torch.tensor(yaw_degrees, dtype=torch.float32))
    return torch.stack(
        (
            torch.zeros(2),
            torch.zeros(2),
            torch.sin(yaw),
            torch.cos(yaw),
        ),
        dim=-1,
    )


def test_v3_routes_self_and_cooperative_after_llm_hysteresis():
    planner = _planner_for_routing()
    detection = torch.tensor(
        [
            [0.5, 0.5, 0.2, 0.2, 0.9, 1.0],
            [0.5, 0.5, 0.2, 0.2, 0.9, 1.0],
        ]
    )
    first, debug, _ = planner._route_waypoints(
        _routing_output(), detection, _agent_pose_tokens()
    )
    assert debug["routing_mode"] == [ROUTE_BELIEF, ROUTE_BELIEF]
    np.testing.assert_allclose(first[:, 0, 0], [10.0, 20.0])

    second, debug, visible = planner._route_waypoints(
        _routing_output(), detection, _agent_pose_tokens()
    )
    assert debug["routing_mode"] == [ROUTE_SELF, ROUTE_SELF]
    assert visible.tolist() == [True, True]
    np.testing.assert_allclose(second[:, 0, 0], [1.0, 2.0])

    mismatch = _routing_output((1.0, 0.0))
    planner._route_waypoints(mismatch, detection, _agent_pose_tokens())
    routed, debug, visible = planner._route_waypoints(
        mismatch, detection, _agent_pose_tokens()
    )
    assert debug["routing_mode"] == [ROUTE_SELF, ROUTE_COOPERATIVE]
    assert visible.tolist() == [True, False]
    np.testing.assert_allclose(routed[:, 0, 0], [1.0, 20.0])


def test_v3_both_invisible_transitions_from_belief_to_bounded_yaw_search():
    planner = _planner_for_routing()
    detection = torch.tensor(
        [
            [0.5, 0.5, 0.2, 0.2, 0.9, 1.0],
            [0.5, 0.5, 0.2, 0.2, 0.9, 1.0],
        ]
    )
    planner._route_waypoints(_routing_output(), detection, _agent_pose_tokens())
    planner._route_waypoints(_routing_output(), detection, _agent_pose_tokens())
    mismatch = _routing_output((0.0, 0.0))
    loss_pose = _agent_pose_tokens((20.0, 175.0))
    planner._route_waypoints(mismatch, detection, loss_pose)
    belief, debug, _ = planner._route_waypoints(mismatch, detection, loss_pose)
    assert debug["routing_mode"] == [ROUTE_BELIEF, ROUTE_BELIEF]
    assert np.any(belief[..., :2] != 0.0)

    search, debug, _ = planner._route_waypoints(mismatch, detection, loss_pose)
    assert debug["routing_mode"] == [ROUTE_SEARCH, ROUTE_SEARCH]
    assert debug["routing_mode_name"] == ["search", "search"]
    np.testing.assert_allclose(search[..., :2], 0.0)
    np.testing.assert_allclose(debug["search_center_yaw_degrees"], [20.0, 175.0])
    np.testing.assert_allclose(debug["search_target_yaw_degrees"], [50.0, -155.0])
    np.testing.assert_allclose(
        search[:, -1, 2], np.deg2rad([30.0, 30.0]), atol=1.0e-5
    )

    # Each agent reverses only after reaching its own +30-degree endpoint.
    reverse_pose = _agent_pose_tokens((50.0, -155.0))
    reverse, debug, _ = planner._route_waypoints(mismatch, detection, reverse_pose)
    np.testing.assert_allclose(
        debug["search_target_yaw_degrees"], [-10.0, 145.0], atol=1.0e-4
    )
    np.testing.assert_allclose(
        reverse[:, -1, 2], np.deg2rad([-60.0, -60.0]), atol=1.0e-5
    )

    # The center is fixed at loss time even after the physical heading moves.
    endpoint_pose = _agent_pose_tokens((-10.0, 145.0))
    _, debug, _ = planner._route_waypoints(mismatch, detection, endpoint_pose)
    np.testing.assert_allclose(
        debug["search_center_yaw_degrees"], [20.0, 175.0], atol=1.0e-4
    )
    np.testing.assert_allclose(
        debug["search_target_yaw_degrees"], [50.0, -155.0], atol=1.0e-4
    )


def test_v3_shared_server_session_has_no_removed_search_state(
    monkeypatch: pytest.MonkeyPatch,
):
    class FakePlanner:
        def __init__(self, args):
            self.args = args
            self.ckpt_path = args.ckpt
            self.history = 3
            self.history_frame_dt = 0.1
            self.n_waypoints = 4
            self.use_roi_tokens = False
            self.use_bbox_text_prompt = False
            self.use_visual_section_markers = False
            self.roi_bbox_source = "none"
            self.roi_expand_ratio = 0.0
            self.roi_token_count = 0
            self.roi_make_square = False
            self.evaluation_protocol = "test"
            self.ckpt_bbox_dropout_prob = 0.0
            self.supports_intermediate_observation = True
            self.histories = []
            self.inverse_velocity_state = None
            self.inverse_velocity_reference = None
            self.last_inverse_command = None
            self.last_waypoints = None
            self.last_policy_debug = {}
            self.bbox_motion_controller = object()
            self.visibility_router = object()
            self.last_navigation_waypoints = None
            self.search_center_yaw_rad = None
            self.search_target_direction = np.ones(2, dtype=np.float32)

        @staticmethod
        def new_bbox_motion_controller():
            return object()

        @staticmethod
        def new_visibility_router():
            return object()

        def reset(self):
            return None

    monkeypatch.setattr(eval_v3_server_module, "AirGroundCoopV3Planner", FakePlanner)
    server = SharedAirGroundV3Server()
    metadata = server.request(
        {"op": "init", "session": "worker-0", "args": {"ckpt": "dummy.pt"}}
    )

    assert metadata["ckpt_path"] == "dummy.pt"
    assert set(server.states["worker-0"]) == set(SESSION_FIELDS)
    assert "search_yaw_sign" not in SESSION_FIELDS
    assert "search_center_yaw_rad" in SESSION_FIELDS
    assert "search_target_direction" in SESSION_FIELDS
    assert set(server.states["worker-0"]) == set(SESSION_FIELDS)
    # Exercise restore + save too; search state must be session-local.
    server.request({"op": "reset", "session": "worker-0"})


def test_video_box_is_exposed_only_after_llm_verification():
    boxes = [[0.4, 0.5, 0.2, 0.3], [0.6, 0.5, 0.1, 0.2]]
    assert _llm_verified_boxes(boxes, [True, False]) == [
        pytest.approx(boxes[0]),
        [0.0, 0.0, 0.0, 0.0],
    ]


def test_v3_bbox_direction_logic_is_llm_gated():
    controller = BBoxMotionController(_bbox_args())
    initial_box = [0.75, 0.5, 0.1, 0.2]
    short_box = [0.75, 0.5, 0.1, 0.01]
    # A high-confidence YOLO proposal rejected by VERIFY is passed as invalid.
    controller.observe(0, short_box, 0.95, False)
    controller.observe(0, short_box, 0.95, False)
    unchanged, _, _ = controller.adjust(
        np.asarray([0.4, 0.0, 0.0]),
        np.asarray([0.4, 0.0, 0.0]),
    )
    np.testing.assert_allclose(unchanged, [0.4, 0.0, 0.0])

    controller.observe(0, initial_box, 0.95, True)
    controller.observe(0, initial_box, 0.95, True)
    for _ in range(5):
        controller.observe(0, short_box, 0.95, True)
    adjusted, _, debug = controller.adjust(
        np.asarray([0.4, 0.0, 0.0]),
        np.asarray([0.4, 0.0, 0.0]),
    )
    assert adjusted[0] > 0.4
    assert adjusted[1] == pytest.approx(0.0)
    assert adjusted[2] > 0.0
    assert debug["controller"] == "yolo_llm_verified_bbox_height_cx_v3"
    assert debug["agents"][0]["speed_correction_applied"] is True


def test_v3_initial_bbox_height_dead_band_does_not_adjust_speed():
    controller = BBoxMotionController(_bbox_args())
    controller.observe(0, [0.50, 0.50, 0.1, 0.20], 0.95, True)
    controller.observe(0, [0.50, 0.50, 0.1, 0.20], 0.95, True)
    for _ in range(5):
        controller.observe(0, [0.50, 0.50, 0.1, 0.17], 0.95, True)
    drone, _, debug = controller.adjust(
        np.asarray([0.4, 0.1, 0.0]),
        np.asarray([0.4, 0.0, 0.0]),
    )
    np.testing.assert_allclose(drone, [0.4, 0.1, 0.0])
    assert debug["agents"][0]["reference_height"] == pytest.approx(0.20)
    assert debug["agents"][0]["speed_correction_applied"] is False
    assert debug["agents"][0]["speed_gain"] == pytest.approx(1.0)
    assert debug["agents"][0]["forward_residual_mps"] == pytest.approx(0.0)


def test_v3_bbox_short_accelerates_and_long_slows_relative_to_initial_height():
    short = BBoxMotionController(_bbox_args())
    long = BBoxMotionController(_bbox_args())
    for controller in (short, long):
        controller.observe(0, [0.50, 0.50, 0.1, 0.20], 0.95, True)
        controller.observe(0, [0.50, 0.50, 0.1, 0.20], 0.95, True)
    for _ in range(8):
        short.observe(0, [0.50, 0.50, 0.1, 0.05], 0.95, True)
        long.observe(0, [0.50, 0.50, 0.1, 0.40], 0.95, True)

    short_drone, _, short_debug = short.adjust(
        np.asarray([0.4, 0.0, 0.0]), np.asarray([0.4, 0.0, 0.0])
    )
    long_drone, _, long_debug = long.adjust(
        np.asarray([0.4, 0.0, 0.0]), np.asarray([0.4, 0.0, 0.0])
    )
    assert short_drone[0] > 0.4
    assert long_drone[0] < 0.4
    assert short_debug["agents"][0]["speed_correction_applied"] is True
    assert long_debug["agents"][0]["speed_correction_applied"] is True


def test_v3_reverse_motion_accelerates_when_bbox_is_large():
    controller = BBoxMotionController(_bbox_args())
    for agent in (0, 1):
        controller.observe(agent, [0.50, 0.50, 0.1, 0.20], 0.95, True)
        controller.observe(agent, [0.50, 0.50, 0.1, 0.20], 0.95, True)
    for _ in range(8):
        for agent in (0, 1):
            controller.observe(agent, [0.50, 0.50, 0.1, 0.40], 0.95, True)

    drone, dog, debug = controller.adjust(
        np.asarray([-0.4, 0.0, 0.0]), np.asarray([-0.4, 0.0, 0.0])
    )

    assert drone[0] < -0.4
    assert dog[0] < -0.4
    assert debug["agents"][0]["speed_correction_direction"] == "reverse"
    assert debug["agents"][1]["speed_correction_direction"] == "reverse"
    assert debug["agents"][0]["speed_correction_applied"] is True
    assert debug["agents"][1]["speed_correction_applied"] is True
    assert debug["agents"][0]["forward_residual_mps"] < 0.0
    assert debug["agents"][1]["forward_residual_mps"] < 0.0


def test_v3_reverse_motion_slows_when_bbox_is_small():
    controller = BBoxMotionController(_bbox_args())
    for agent in (0, 1):
        controller.observe(agent, [0.50, 0.50, 0.1, 0.20], 0.95, True)
        controller.observe(agent, [0.50, 0.50, 0.1, 0.20], 0.95, True)
    for _ in range(8):
        for agent in (0, 1):
            controller.observe(agent, [0.50, 0.50, 0.1, 0.05], 0.95, True)

    drone, dog, debug = controller.adjust(
        np.asarray([-0.4, 0.0, 0.0]), np.asarray([-0.4, 0.0, 0.0])
    )

    assert -0.4 < drone[0] < 0.0
    assert -0.4 < dog[0] < 0.0
    assert debug["agents"][0]["speed_correction_direction"] == "reverse"
    assert debug["agents"][1]["speed_correction_direction"] == "reverse"
    assert debug["agents"][0]["speed_correction_applied"] is True
    assert debug["agents"][1]["speed_correction_applied"] is True
    assert debug["agents"][0]["forward_residual_mps"] == pytest.approx(0.0)
    assert debug["agents"][1]["forward_residual_mps"] == pytest.approx(0.0)


def test_v3_robotdog_uses_stronger_forward_correction_when_target_is_centered():
    controller = BBoxMotionController(_bbox_args())
    controller.observe(1, [0.50, 0.50, 0.1, 0.20], 0.95, True)
    controller.observe(1, [0.50, 0.50, 0.1, 0.20], 0.95, True)
    for _ in range(20):
        controller.observe(1, [0.50, 0.50, 0.1, 0.01], 0.95, True)

    _, dog, debug = controller.adjust(
        np.asarray([0.4, 0.0, 0.0]), np.asarray([0.4, 0.0, 0.0])
    )

    assert debug["agents"][1]["q_short"] == pytest.approx(1.0)
    assert debug["agents"][1]["speed_gain"] == pytest.approx(1.50)
    assert debug["agents"][1]["forward_residual_mps"] == pytest.approx(0.15)
    assert dog[0] == pytest.approx(0.75)
    assert debug["agents"][1]["turn_in_place_applied"] is False


def test_v3_robotdog_stops_forward_motion_during_bbox_turn_correction():
    controller = BBoxMotionController(_bbox_args())
    for _ in range(2):
        controller.observe(1, [0.70, 0.50, 0.1, 0.20], 0.95, True)

    drone, dog, debug = controller.adjust(
        np.asarray([0.4, 0.0, 0.0]), np.asarray([0.8, 0.0, 0.0])
    )

    np.testing.assert_allclose(drone, [0.4, 0.0, 0.0])
    assert dog[0] == pytest.approx(0.0)
    assert dog[2] > 0.0
    assert debug["agents"][1]["speed_correction_applied"] is False
    assert debug["agents"][1]["turn_in_place_applied"] is True


def test_v3_speed_limit_is_hard_even_before_bbox_controller_is_reliable():
    controller = BBoxMotionController(
        _bbox_args(
            bbox_motion_control=False,
            drone_max_speed=2.5,
            robotdog_max_speed=2.5,
        )
    )
    drone, dog, _ = controller.adjust(
        np.asarray([3.0, 4.0, 0.0]),
        np.asarray([3.0, 0.0, 0.0]),
    )
    assert np.linalg.norm(drone[:2]) == pytest.approx(2.5)
    assert dog[0] == pytest.approx(2.5)


def test_v3_loader_rejects_v1_architecture(tmp_path):
    checkpoint = tmp_path / "v1.pt"
    torch.save(
        {"config": {"model_architecture": "airground_three_stream_cooperative_v2"}},
        checkpoint,
    )
    with pytest.raises(ValueError, match="airground_three_stream_cooperative_v3"):
        _load_payload(checkpoint)


def test_v3_loader_rejects_old_receiver_frame_contract(tmp_path):
    checkpoint = tmp_path / "old_receiver_frame.pt"
    torch.save(
        {"config": {"model_architecture": "airground_three_stream_cooperative_v3"}},
        checkpoint,
    )
    with pytest.raises(ValueError, match="receiver-recovery V3 contract"):
        _load_payload(checkpoint)


def test_v3_inference_builds_two_pair_centred_pose_tokens():
    poses = _agent_poses(
        [100.0, 300.0, 50.0, 0.0, 90.0, 0.0],
        [500.0, 100.0, 0.0, 0.0, 0.0, 0.0],
    )
    assert poses.shape == (2, 4)
    torch.testing.assert_close(poses[:, :2].mean(dim=0), torch.zeros(2))
    torch.testing.assert_close(poses[0], torch.tensor([-2.0, 1.0, 1.0, 0.0]))
    torch.testing.assert_close(poses[1], torch.tensor([2.0, -1.0, 0.0, 1.0]))


class _ClosableImage:
    def close(self):
        return None


class _FakeReceiverTargetV3Model:
    cfg = SimpleNamespace(perception_grid_size=1)

    def __call__(self, **kwargs):
        assert tuple(kwargs["agent_poses"].shape) == (1, 2, 4)
        assert not kwargs["synthetic_occlusion"].any()
        assert not kwargs["coarse_missing_mask"].any()
        assert not kwargs["fine_missing_mask"].any()
        assert "joint_instructions" in kwargs
        waypoints = torch.zeros(1, 2, 3, 3)
        waypoints[0, 0, 1:, 0] = torch.tensor([1.0, 2.0])
        waypoints[0, 1, 1:, 0] = torch.tensor([0.5, 1.0])
        return {
            "self_waypoints": waypoints,
            "cooperative_waypoints": waypoints + 1.0,
            "cooperative_candidates": (waypoints + 1.0).unsqueeze(2),
            "cooperative_mode_logits": torch.zeros(1, 2, 1),
            "target_match_probability": torch.tensor([[0.9, 0.8]]),
            "target_belief": torch.zeros(1, 2, 5),
            "jepa_uncertainty_logit": torch.zeros(1, 2),
        }


class _FakeBBoxMotionController:
    enabled = True

    def observe(self, agent, box, score, valid):
        return {"agent": int(agent), "valid": bool(valid)}


def test_v3_bgr_input_is_converted_to_rgb_for_visual_models():
    bgr_red = np.asarray([[[0, 0, 255]]], dtype=np.uint8)
    images = AirGroundCoopV3Planner._pil_pair(bgr_red, bgr_red)
    try:
        assert np.asarray(images[0])[0, 0].tolist() == [255, 0, 0]
    finally:
        for image in images:
            image.close()


def test_v3_intermediate_observe_updates_history_without_model_forward():
    planner = AirGroundCoopV3Planner.__new__(AirGroundCoopV3Planner)
    planner.device = torch.device("cpu")
    planner.vision_amp = False
    planner.histories = [deque(), deque()]
    planner._pil_pair = lambda *_args: [_ClosableImage(), _ClosableImage()]
    planner.encoder = SimpleNamespace(
        _encode_dino=lambda _images: (torch.ones(2, 4, 2), 2, 2),
        _encode_siglip=lambda _images, out_hw: torch.ones(2, 4, 3),
    )

    debug = planner.observe(
        np.zeros((2, 2, 3), dtype=np.uint8),
        np.zeros((2, 2, 3), dtype=np.uint8),
        observation_time=0.1,
    )

    assert debug["encoding_time"] >= 0.0
    assert [len(history) for history in planner.histories] == [1, 1]
    assert planner.histories[0][0][0] == pytest.approx(0.1)
    assert planner.histories[0][0][1].shape == (4, 5)


def test_v3_predict_disables_all_training_corruption(monkeypatch):
    planner = AirGroundCoopV3Planner.__new__(AirGroundCoopV3Planner)
    planner.device = torch.device("cpu")
    planner.model = _FakeReceiverTargetV3Model()
    planner.config = {
        "instruction_override": "follow",
        "joint_instruction_override": "joint follow",
        "agent1_instruction_override": "drone follow",
        "agent2_instruction_override": "dog follow",
    }
    planner.history = 2
    planner.history_frame_dt = 0.1
    planner.histories = [deque(), deque()]
    planner.bbox_motion_controller = _FakeBBoxMotionController()
    planner.evaluation_protocol = "receiver-target-test"
    planner._pil_pair = lambda *_args: [_ClosableImage(), _ClosableImage()]
    predictions = [
        SimpleNamespace(
            person_box_cxcywh_norm=np.asarray([0.5, 0.5, 0.2, 0.3]),
            person_score=0.9,
            person_valid=True,
        ),
        SimpleNamespace(
            person_box_cxcywh_norm=np.asarray([0.4, 0.5, 0.2, 0.3]),
            person_score=0.8,
            person_valid=True,
        ),
    ]
    planner.perception = SimpleNamespace(predict=lambda _images: predictions)
    planner._perception_tensors = lambda _predictions, _grid: (
        torch.tensor(
            [
                [0.5, 0.5, 0.2, 0.3, 0.9, 1.0],
                [0.4, 0.5, 0.2, 0.3, 0.8, 1.0],
            ]
        ),
        torch.zeros(2, 1, 1, 4),
    )
    planner.encoder = SimpleNamespace(
        _encode_dino=lambda _images: (torch.zeros(2, 1, 2), 1, 1),
        _encode_siglip=lambda _images, out_hw: torch.zeros(2, 1, 2),
    )
    planner._route_waypoints = lambda output, detection, poses: (
        output["self_waypoints"][0].numpy(),
        {
            "routing_mode": [ROUTE_SELF, ROUTE_SELF],
            "routing_mode_name": ["self", "self"],
            "route_to_cooperative": [False, False],
            "target_match_probability": [0.9, 0.8],
        },
        np.asarray([True, True]),
    )
    monkeypatch.setattr(
        eval_v3_module,
        "grid_pool_tokens",
        lambda visual, _h, _w, out_tokens: torch.zeros(
            visual.size(0), out_tokens, visual.size(-1)
        ),
    )

    result = planner.predict(
        np.zeros((2, 2, 3), dtype=np.uint8),
        np.zeros((2, 2, 3), dtype=np.uint8),
        None,
        None,
        "follow",
        observation_time=0.0,
        drone_pose=[100.0, 200.0, 0.0, 0.0, 0.0, 0.0],
        robotdog_pose=[300.0, 200.0, 0.0, 0.0, 90.0, 0.0],
    )

    assert result["routing_mode_name"] == ["self", "self"]
    assert result["agent_poses"][0][:2] == pytest.approx([-1.0, 0.0])
    assert result["agent_poses"][1][:2] == pytest.approx([1.0, 0.0])


def test_v3_nonholonomic_projection_recovers_constant_curvature_arc():
    radius = 2.0
    theta = np.asarray([0.0, 0.2, 0.4, 0.6], dtype=np.float64)
    poses = np.stack(
        (
            radius * np.sin(theta),
            radius * (1.0 - np.cos(theta)),
            theta,
        ),
        axis=-1,
    )

    projected, diagnostic = _project_robotdog_waypoints_to_nonholonomic(
        poses,
        control_dt=0.1,
        source_dt=0.1,
        horizon_steps=3,
    )

    # x/y jointly recover signed arc length; the physical lateral channel is 0.
    np.testing.assert_allclose(projected[:, 0], radius * theta, atol=1.0e-10)
    np.testing.assert_allclose(projected[:, 1], 0.0, atol=0.0)
    np.testing.assert_allclose(
        diagnostic["forward_displacements_m"], radius * 0.2, atol=1.0e-10
    )
    np.testing.assert_allclose(
        diagnostic["lateral_residuals_m"], 0.0, atol=1.0e-10
    )
    # Every point asks the inherited controller for theta(t)/t = 2 rad/s.
    np.testing.assert_allclose(projected[1:, 2] / 0.1, 2.0, atol=1.0e-10)


def test_v3_nonholonomic_projection_reports_infeasible_sideways_jump():
    poses = np.asarray(
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0]],
        dtype=np.float64,
    )
    projected, diagnostic = _project_robotdog_waypoints_to_nonholonomic(
        poses,
        control_dt=0.1,
        source_dt=0.1,
        horizon_steps=2,
    )

    np.testing.assert_allclose(projected[:, :2], 0.0, atol=0.0)
    np.testing.assert_allclose(
        diagnostic["lateral_residuals_m"], [1.0, 1.0], atol=1.0e-12
    )


def test_v3_controller_projects_pose_before_shared_inverse_controller(
    monkeypatch: pytest.MonkeyPatch,
):
    captured = {}

    def fake_parent_controller(
        _self,
        control_waypoints,
        *,
        realtime_control_period_seconds=None,
    ):
        del realtime_control_period_seconds
        captured["waypoints"] = np.asarray(control_waypoints).copy()
        return [0.0] * 4, [0.0] * 2, {
            "robotdog_waypoint_index": 1,
            "robotdog_horizon_dt": 0.1,
            "robotdog_waypoint": control_waypoints[1, 1, :3].tolist(),
        }

    monkeypatch.setattr(
        AirGroundV3RuntimePlanner, "waypoints_to_actions", fake_parent_controller
    )
    planner = AirGroundCoopV3Planner.__new__(AirGroundCoopV3Planner)
    planner.args = SimpleNamespace(
        dt=0.1,
        waypoint_source_dt=0.1,
        waypoint_horizon_steps=3,
        ground_translation_delay_steps=0,
        robotdog_waypoint_y_mode="v3_nonholonomic_projection",
    )
    planner.last_policy_debug = {
        "routing_mode_name": ["SELF", "COOPERATIVE"]
    }
    radius = 2.0
    theta = np.asarray([0.0, 0.2, 0.4, 0.6])
    dog = np.stack(
        (
            radius * np.sin(theta),
            radius * (1.0 - np.cos(theta)),
            theta,
        ),
        axis=-1,
    )
    waypoints = np.stack((np.zeros_like(dog), dog), axis=0)

    _, _, debug = planner.waypoints_to_actions(waypoints)

    np.testing.assert_allclose(captured["waypoints"][1, :, 0], radius * theta)
    np.testing.assert_allclose(captured["waypoints"][1, :, 1], 0.0)
    np.testing.assert_allclose(debug["robotdog_waypoint"], dog[1])
    np.testing.assert_allclose(
        debug["robotdog_nonholonomic_projected_waypoint"],
        captured["waypoints"][1, 1, :3],
    )
    assert debug["robotdog_lateral_ignored"] == pytest.approx(0.0)
    assert debug["robotdog_action_source"] == "COOPERATIVE"
    assert "nonholonomic_pose_projection" in debug["action_source"]


def test_v3_runtime_defaults_preserve_full_sequence_and_speed_limits():
    argv = _runtime_argv([])
    assert argv[argv.index("--max-lost-steps") + 1] == "400"
    assert argv[argv.index("--drone-max-speed") + 1] == "2.5"
    assert argv[argv.index("--robotdog-max-speed") + 1] == "2.5"
    assert argv[argv.index("--policy-inference-stride") + 1] == "5"
    # policy 每 0.5 s 调用一次，但每个 future-segment waypoint 和逆控制动作
    # 仍然对应一个原始记录的 0.1 s 物理步。
    assert argv[argv.index("--waypoint-source-dt") + 1] == "0.1"
    assert argv[argv.index("--waypoint-horizon-steps") + 1] == "7"
    assert argv[argv.index("--robotdog-waypoint-y-mode") + 1] == (
        "v3_nonholonomic_projection"
    )
    assert argv[argv.index("--bbox-motion-height-tolerance-ratio") + 1] == "0.20"
    assert argv[argv.index("--bbox-motion-height-response-ratio") + 1] == "0.50"
    assert argv[argv.index("--robotdog-bbox-max-speed-gain") + 1] == "2.00"
    explicit_lost = _runtime_argv(["--max-lost-steps", "25"])
    assert explicit_lost[explicit_lost.index("--max-lost-steps") + 1] == "25"
    # Sanity-check that this behavior is V3-local, not a monkey-patched module.
    assert eval_v3_module.ARCHITECTURE == "airground_three_stream_cooperative_v3"


def test_shared_target_reference_confirms_then_releases() -> None:
    planner = AirGroundCoopV3Planner.__new__(AirGroundCoopV3Planner)
    planner.config = {
        "target_reference_confirm_frames": 3,
        "target_reference_release_frames": 2,
    }
    fine = torch.tensor(
        [[[1.0, 0.0]] * 4, [[0.0, 1.0]] * 4]
    )
    selected = torch.tensor(
        [[0.25, 0.25, 0.5, 0.5, 0.9, 1.0]] * 2
    )
    accepted = torch.tensor([True, True])
    confidence = torch.tensor([0.9, 0.9])

    for _ in range(2):
        planner._update_target_reference_memory(fine, selected, accepted, confidence)
    assert not planner.target_reference_valid.any()
    planner._update_target_reference_memory(fine, selected, accepted, confidence)
    assert planner.target_reference_valid.all()
    frozen = planner.target_reference_tokens.clone()

    planner._update_target_reference_memory(
        fine, selected, torch.tensor([False, False]), confidence
    )
    assert planner.target_reference_valid.all()
    torch.testing.assert_close(planner.target_reference_tokens, frozen)
    planner._update_target_reference_memory(
        fine, selected, torch.tensor([False, False]), confidence
    )
    assert not planner.target_reference_valid.any()
