#!/usr/bin/env python3
"""从双 Agent 训练标签生成 Anchor Diffusion 使用的轨迹锚点。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np

import cundang.model_unrealzoo_anchor_diffusion as anchor_model
import train


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Cluster drone/robotdog waypoint labels into separate anchor banks.")
    ap.add_argument("--train_json", type=str, required=True, help="dataset.json, one .jsonl, or a JSONL directory.")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--n_waypoints", type=int, default=8)
    ap.add_argument("--action_dims", type=int, default=3)
    ap.add_argument("--num_anchors", type=int, default=40)
    ap.add_argument("--num_iters", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_samples", type=int, default=0, help="0 uses every training sample.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    ds = train.MultiAgentJsonDataset(
        train.MultiAgentDataConfig(
            train_json=args.train_json,
            n_waypoints=args.n_waypoints,
            action_dims=args.action_dims,
        )
    )
    limit = len(ds) if args.max_samples <= 0 else min(len(ds), args.max_samples)
    trajectories: List[List[np.ndarray]] = [[], []]
    valid_masks: List[List[np.ndarray]] = [[], []]
    for index in range(limit):
        # 这里只读取标签，不读取图像或 vision_cache，因此生成锚点速度较快。
        waypoints, valid_mask = ds._load_targets(ds.get_example(index))
        for agent_idx in range(2):
            trajectories[agent_idx].append(waypoints[agent_idx].numpy())
            valid_masks[agent_idx].append(valid_mask[agent_idx].numpy())

    if limit < args.num_anchors:
        raise ValueError(f"Need at least {args.num_anchors} samples, but only found {limit}.")

    output_dir = Path(args.out_dir).expanduser().resolve()
    names = ("agent1_drone", "agent2_robotdog")
    for agent_idx, name in enumerate(names):
        traj = np.stack(trajectories[agent_idx]).astype(np.float32)
        mask = np.stack(valid_masks[agent_idx]).astype(np.float32)
        anchors, assignment = anchor_model.fit_trajectory_anchors_kmeans(
            traj,
            num_anchors=args.num_anchors,
            valid_mask=mask,
            num_iters=args.num_iters,
            seed=args.seed + agent_idx,
        )
        output_path = output_dir / f"{name}_anchors.npy"
        anchor_model.save_trajectory_anchors(
            output_path,
            anchors,
            metadata={
                "source": str(Path(args.train_json).expanduser().resolve()),
                "agent_index": agent_idx,
                "agent_name": name,
                "samples": limit,
                "coordinate_system": "local trajectory coordinates from training waypoint labels",
                "units": "same as dataset waypoints",
                "seed": args.seed + agent_idx,
            },
        )
        counts = np.bincount(assignment, minlength=args.num_anchors)
        print(
            f"[ANCHOR] {name}: saved {output_path} shape={anchors.shape} "
            f"cluster_size_min={counts.min()} max={counts.max()}",
            flush=True,
        )


if __name__ == "__main__":
    main()
