import torch

from model import MultiAgentModelConfig, MultiAgentOpenTrackVLA
from train import (
    MultiAgentTrainConfig,
    apply_multi_agent_base_defaults,
    build_multi_agent_lr_scheduler,
    forward_multi_agent_loss,
    multi_agent_lr_for_step,
)


def test_no_marker_shared_context_has_five_pieces() -> None:
    model = MultiAgentOpenTrackVLA.__new__(MultiAgentOpenTrackVLA)
    torch.nn.Module.__init__(model)
    model.cfg = MultiAgentModelConfig(
        use_grounding=False,
        use_bbox_tokens=False,
        use_agent_text_markers=False,
    )
    batch_size, hidden_size = 2, 4
    pieces, _, act1_index, act2_index = model._build_shared_context_inputs(
        torch.zeros(batch_size, 3, hidden_size),
        torch.ones(batch_size, 3, dtype=torch.long),
        torch.zeros(batch_size, 5, hidden_size),
        torch.zeros(batch_size, 7, hidden_size),
        torch.zeros(batch_size, 1, hidden_size),
        torch.zeros(batch_size, 1, hidden_size),
        torch.device("cpu"),
    )

    assert [piece.size(1) for piece in pieces] == [3, 5, 7, 1, 1]
    assert (act1_index, act2_index) == (3, 4)


def test_cosine_warmup_lr_boundaries() -> None:
    total_steps = 22_340
    assert multi_agent_lr_for_step(0, total_steps, 2e-5, 2e-6, 500) == 2e-6
    assert multi_agent_lr_for_step(500, total_steps, 2e-5, 2e-6, 500) == 2e-5
    assert multi_agent_lr_for_step(total_steps, total_steps, 2e-5, 2e-6, 500) == 2e-6

    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.AdamW([parameter], lr=2e-5)
    cfg = MultiAgentTrainConfig(
        train_json="unused",
        base_model=True,
        lr=2e-5,
        lr_scheduler="cosine",
        warmup_steps=500,
        min_lr=2e-6,
    )
    scheduler = build_multi_agent_lr_scheduler(optimizer, cfg, total_steps)
    assert scheduler is not None
    assert optimizer.param_groups[0]["lr"] == 2e-6


class _FixedWaypointModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("alpha_task", torch.ones(1, 1, 3))

    def forward(self, **_):
        drone = torch.ones(2, 3)
        robotdog = torch.full((2, 3), 3.0)
        return {
            "waypoints": torch.stack([drone, robotdog]).unsqueeze(0),
            "refined_bbox": None,
            "visible_logits": None,
            "relative_pose": None,
        }


def test_equal_normalized_agent_loss_preserves_single_agent_scale() -> None:
    cfg = apply_multi_agent_base_defaults(
        MultiAgentTrainConfig(
            train_json="unused",
            base_model=True,
            drone_loss_weight=1.0,
            dog_loss_weight=1.0,
            normalize_agent_loss_weights=True,
            beta_nav=100.0,
        )
    )
    batch = {
        "bbox_feat": torch.zeros(1, 2, 4),
        "instruction": ["task"],
        "coarse_tokens": torch.zeros(1, 2, 1, 1),
        "coarse_tidx": torch.zeros(1, 2, 1, dtype=torch.long),
        "fine_tokens": torch.zeros(1, 2, 1, 1),
        "fine_tidx": torch.zeros(1, 2, 1, dtype=torch.long),
        "waypoints": torch.zeros(1, 2, 2, 3),
        "valid_mask": torch.ones(1, 2, 2, dtype=torch.bool),
        "dt": torch.tensor([0.1]),
        "yaw_hist": torch.zeros(1, 2, 1),
        "yaw_curr": torch.zeros(1, 2, 1),
    }

    loss, metrics = forward_multi_agent_loss(
        _FixedWaypointModel(),
        batch,
        cfg,
        torch.device("cpu"),
    )

    assert metrics["loss_nav_drone"].item() == 1.0
    assert metrics["loss_nav_dog"].item() == 9.0
    assert metrics["loss_nav"].item() == 5.0
    assert loss.item() == 500.0


def test_old_base_defaults_keep_markers_and_constant_lr() -> None:
    cfg = apply_multi_agent_base_defaults(
        MultiAgentTrainConfig(train_json="unused", base_model=True)
    )
    assert cfg.use_agent_text_markers is True
    assert cfg.lr_scheduler == "constant"
    assert cfg.drone_loss_weight == 2.0
