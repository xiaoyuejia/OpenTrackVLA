#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate the clean base multi-agent model in UnrealZoo.

The environment loop is reused from eval_unrealzoo_multi_agent.py, but the
planner loaded here is strictly the base concat model:
two visual streams -> one LLM context -> two ACT waypoint heads.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import random
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
from PIL import Image

from cundang.model_multi_agent_base import BaseMultiAgentModelConfig, BaseMultiAgentOpenTrackVLA
from tools.cache_gridpool import VisionCacheConfig, VisionFeatureCacher, grid_pool_tokens

from eval_unrealzoo_multi_agent import (  # Reuse the existing UE loop only.
    DEFAULT_ENV_ID,
    DEFAULT_INSTRUCTION,
    UNREAL_UNITS_PER_METER,
    _parse_episode_filter,
    ensure_bgr_uint8,
    load_recorded_target_trajectories,
    make_env,
    reset_planner_debug_file,
    run_episode,
    write_episode_outputs,
)


def latest_checkpoint(path: str | Path) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    files = sorted(path.glob("model_epoch*_step*.pt"))
    if not files:
        raise FileNotFoundError(f"No model_epoch*_step*.pt under {path}")
    return files[-1]


class BaseUnrealZooMultiAgentPlanner:
    """Online planner wrapper for BaseMultiAgentOpenTrackVLA."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.ckpt_path = latest_checkpoint(args.ckpt)
        ckpt = torch.load(self.ckpt_path, map_location="cpu")
        cfg = ckpt.get("config", {})
        state = ckpt.get("model_state") or ckpt.get("state_dict") or ckpt
        has_planner_agent1 = any(k.startswith("planner_agent1.") for k in state)
        has_planner_agent2 = any(k.startswith("planner_agent2.") for k in state)
        print(
            "[planner] evaluator=eval_unrealzoo_multi_agent_base.py "
            "model_class=BaseMultiAgentOpenTrackVLA agent_order=drone,robotdog "
            "binding=drone:planner_agent1/waypoints[0],robotdog:planner_agent2/waypoints[1]",
            flush=True,
        )
        print(
            f"[planner] checkpoint model_type={cfg.get('model_type')} "
            f"train_json={cfg.get('train_json')} "
            f"use_grounding={cfg.get('use_grounding')} use_bbox_tokens={cfg.get('use_bbox_tokens')} "
            f"planner_agent1={has_planner_agent1} planner_agent2={has_planner_agent2}",
            flush=True,
        )
        if not has_planner_agent1 or not has_planner_agent2:
            raise RuntimeError(
                "Checkpoint does not contain both planner_agent1.* and planner_agent2.* weights. "
                "Cannot bind drone to planner_agent1 and robotdog to planner_agent2 safely."
            )
        if cfg.get("model_type") not in {None, "base_multi_agent_concat"}:
            print(f"[planner][warn] checkpoint model_type={cfg.get('model_type')} is not base_multi_agent_concat", flush=True)

        model_cfg = BaseMultiAgentModelConfig(
            llm_name=str(cfg.get("llm_name", args.llm_name)),
            freeze_llm=True,
            n_waypoints=int(cfg.get("n_waypoints", args.n_waypoints)),
            action_dims=int(cfg.get("action_dims", 3)),
            use_tanh_actions=bool(cfg.get("use_tanh_actions", False)),
            alpha_xy=cfg.get("alpha_xy", args.alpha_xy),
        )
        vision_feat_dim = int(cfg.get("vision_feat_dim", args.vision_feat_dim))
        print(f"[planner] building base concat model on {self.device}", flush=True)
        self.model = BaseMultiAgentOpenTrackVLA(model_cfg, vision_feat_dim=vision_feat_dim).to(self.device).eval()
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        critical = [k for k in missing if k.startswith(("planner_agent", "proj.", "act_tokens"))]
        if critical:
            raise RuntimeError(f"Checkpoint missing critical base weights: {critical[:20]}")
        print(
            f"[planner] loaded={self.ckpt_path} missing={len(missing)} unexpected={len(unexpected)} "
            f"history={model_cfg.__dict__.get('history', args.history)} n_waypoints={model_cfg.n_waypoints}",
            flush=True,
        )

        self.history = int(cfg.get("history", args.history))
        self.n_waypoints = int(cfg.get("n_waypoints", args.n_waypoints))
        self.ckpt_bbox_dropout_prob = 0.0
        self.histories = [deque(maxlen=self.history), deque(maxlen=self.history)]
        self.last_waypoints: Optional[np.ndarray] = None

        print("[startup] loading online DINO + SigLIP visual encoders", flush=True)
        t0 = time.time()
        self.encoder = VisionFeatureCacher(
            VisionCacheConfig(image_size=args.image_size, batch_size=2, device=str(self.device))
        ).eval()
        print(f"[startup] visual encoders loaded in {time.time() - t0:.1f}s", flush=True)

    def reset(self) -> None:
        for hist in self.histories:
            hist.clear()
        self.last_waypoints = None

    @torch.inference_mode()
    def _encode_pair(self, drone_frame_bgr: np.ndarray, dog_frame_bgr: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        pils = []
        for frame in (drone_frame_bgr, dog_frame_bgr):
            rgb = cv2.cvtColor(ensure_bgr_uint8(frame), cv2.COLOR_BGR2RGB)
            pils.append(Image.fromarray(rgb))
        tok_dino, hp, wp = self.encoder._encode_dino(pils)
        tok_sigl = self.encoder._encode_siglip(pils, out_hw=(hp, wp))
        tokens = torch.cat([tok_dino, tok_sigl], dim=-1)
        vfine = grid_pool_tokens(tokens, hp, wp, out_tokens=64).float().cpu()
        vcoarse = grid_pool_tokens(tokens, hp, wp, out_tokens=4).float().cpu()
        return vcoarse, vfine

    def _history_tensor(self, agent_idx: int, current_coarse: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hist = self.histories[agent_idx]
        hist.append(current_coarse.cpu())
        frames = list(hist)
        if len(frames) < self.history:
            frames = [frames[0]] * (self.history - len(frames)) + frames
        frames = frames[-self.history :]
        coarse = torch.cat(frames, dim=0)
        tidx = torch.cat([torch.full((tok.size(0),), i, dtype=torch.long) for i, tok in enumerate(frames)], dim=0)
        return coarse, tidx

    @torch.inference_mode()
    def predict(
        self,
        drone_frame_bgr: np.ndarray,
        dog_frame_bgr: np.ndarray,
        _drone_bbox: Optional[list[float]],
        _dog_bbox: Optional[list[float]],
        instruction: str,
        joint_instruction: Optional[str] = None,
        agent1_instruction: Optional[str] = None,
        agent2_instruction: Optional[str] = None,
    ) -> dict[str, Any]:
        vcoarse, vfine = self._encode_pair(drone_frame_bgr, dog_frame_bgr)
        coarse_items = []
        tidx_items = []
        for agent_idx in range(2):
            coarse, tidx = self._history_tensor(agent_idx, vcoarse[agent_idx])
            coarse_items.append(coarse)
            tidx_items.append(tidx)
        out = self.model(
            coarse_tokens=torch.stack(coarse_items, dim=0).unsqueeze(0).to(self.device),
            coarse_tidx=torch.stack(tidx_items, dim=0).unsqueeze(0).to(self.device),
            fine_tokens=vfine.unsqueeze(0).to(self.device),
            fine_tidx=torch.full((1, 2, vfine.size(1)), self.history, dtype=torch.long, device=self.device),
            instructions=[instruction],
            return_dict=True,
        )
        waypoints = out["waypoints"].detach().float().cpu().numpy()[0]
        self.last_waypoints = waypoints
        zero_bbox = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
        return {
            "waypoints": waypoints,
            "bbox_input": None,
            "bbox_source": "none",
            "visible_score": [0.5, 0.5],
            "refined_bbox": zero_bbox,
            "raw_refined_bbox": zero_bbox,
            "absolute_bbox": zero_bbox,
            "bbox_fallback_to_absolute": [True, True],
            "best_candidate": [None, None],
            "best_candidate_score": [None, None],
        }

    def waypoints_to_actions(self, waypoints: np.ndarray) -> tuple[list[float], list[float], dict[str, Any]]:
        default_idx = int(self.args.waypoint_index)
        drone_idx = int(
            np.clip(
                self.args.drone_waypoint_index if self.args.drone_waypoint_index is not None else default_idx,
                0,
                waypoints.shape[1] - 1,
            )
        )
        dog_idx = int(
            np.clip(
                self.args.robotdog_waypoint_index if self.args.robotdog_waypoint_index is not None else default_idx,
                0,
                waypoints.shape[1] - 1,
            )
        )
        horizon_steps = max(1, int(self.args.waypoint_horizon_steps))
        waypoint_count = int(waypoints.shape[1])

        def waypoint_time(index: int) -> tuple[int, float]:
            source_index = (
                0
                if waypoint_count <= 1
                else int(round(index * (horizon_steps - 1) / (waypoint_count - 1)))
            )
            source_step = source_index + 1
            return source_step, max(source_step * float(self.args.dt), 1e-6)

        drone_source_step, drone_horizon_dt = waypoint_time(drone_idx)
        dog_source_step, dog_horizon_dt = waypoint_time(dog_idx)
        drone_vel = waypoints[0, drone_idx, :3] / drone_horizon_dt
        dog_vel = waypoints[1, dog_idx, :3] / dog_horizon_dt
        drone_waypoint = waypoints[0, drone_idx, :3]
        dog_waypoint = waypoints[1, dog_idx, :3]

        drone_action = [
            float(np.clip(drone_vel[0] * self.args.drone_vx_scale, -self.args.drone_max_vx, self.args.drone_max_vx)),
            float(np.clip(drone_vel[1] * self.args.drone_vy_scale, -self.args.drone_max_vy, self.args.drone_max_vy)),
            0.0,
            float(
                np.clip(
                    drone_vel[2] * self.args.drone_yaw_sign * self.args.drone_yaw_scale,
                    -self.args.drone_max_yaw_rate,
                    self.args.drone_max_yaw_rate,
                )
            ),
        ]
        dog_action = [
            float(np.clip(math.degrees(dog_vel[2] * self.args.robotdog_yaw_sign), -self.args.robotdog_max_turn_deg, self.args.robotdog_max_turn_deg)),
            float(
                np.clip(
                    dog_vel[0] * UNREAL_UNITS_PER_METER * float(self.args.robotdog_speed_gain),
                    -self.args.robotdog_max_speed * UNREAL_UNITS_PER_METER,
                    self.args.robotdog_max_speed * UNREAL_UNITS_PER_METER,
                )
            ),
        ]
        return drone_action, dog_action, {
            "waypoint_index": default_idx,
            "drone_waypoint_index": drone_idx,
            "robotdog_waypoint_index": dog_idx,
            "waypoint_horizon_steps": horizon_steps,
            "drone_waypoint_source_step": drone_source_step,
            "robotdog_waypoint_source_step": dog_source_step,
            "action_source": "base_concat_waypoints",
            "drone_horizon_dt": float(drone_horizon_dt),
            "robotdog_horizon_dt": float(dog_horizon_dt),
            "agent_order": ["drone", "robotdog"],
            "planner_binding": {
                "drone": "planner_agent1 -> waypoints[0]",
                "robotdog": "planner_agent2 -> waypoints[1]",
            },
            "drone_waypoint": [float(v) for v in drone_waypoint.tolist()],
            "robotdog_waypoint": [float(v) for v in dog_waypoint.tolist()],
            "drone_velocity_pred": [float(v) for v in drone_vel.tolist()],
            "robotdog_velocity_pred": [float(v) for v in dog_vel.tolist()],
            "robotdog_speed_gain": float(self.args.robotdog_speed_gain),
            "robotdog_lateral_ignored": float(dog_vel[1]),
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate clean base multi-agent model in UnrealZoo.")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--save-path", default="/data/hdt/ntv_data/sim_data/eval/unrealzoo_multi_agent_base")
    p.add_argument("--env-id", default=DEFAULT_ENV_ID)
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--recorded-target-dir", default=None)
    p.add_argument("--recorded-target-episodes", default=None)
    p.add_argument("--max-attempts", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--render-gpu", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    p.add_argument("--joint-instruction", default=None)
    p.add_argument("--agent1-instruction", default=None)
    p.add_argument("--agent2-instruction", default=None)
    p.add_argument("--llm-name", default="Qwen/Qwen3-0.6B")
    p.add_argument("--vision-feat-dim", type=int, default=1536)
    p.add_argument("--n-waypoints", type=int, default=10)
    p.add_argument("--history", type=int, default=31)
    p.add_argument("--image-size", type=int, default=384)
    p.add_argument("--alpha-xy", type=float, default=1.0)

    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument(
        "--deterministic-step",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--offscreen", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--time-dilation", type=int, default=-1)
    p.add_argument("--disable-ue-input", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--launch-retries", type=int, default=5)
    p.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--write-global-video", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--trajectory-overlay", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--trajectory-scale", type=float, default=120.0)
    p.add_argument("--top-view-height", type=float, default=None)
    p.add_argument("--debug-motion", action="store_true")
    p.add_argument(
        "--planner-debug-steps",
        type=int,
        default=5,
        help="Append planner/action binding debug JSONL for the first N steps of each episode; 0 disables.",
    )
    p.add_argument("--monitor", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--monitor-interval", type=int, default=2)
    p.add_argument("--monitor-scale", type=float, default=0.7)
    p.add_argument("--brightness-scale", type=float, default=1.0)
    p.add_argument("--brightness-offset", type=float, default=0.0)
    p.add_argument("--brightness-config", type=Path, default=None)

    p.add_argument("--human-speed", type=float, default=90.0)
    p.add_argument("--human-turn", type=float, default=5.0)
    p.add_argument("--human-reverse-scale", type=float, default=0.5)
    p.add_argument("--human-goal-min-distance", type=float, default=700.0)
    p.add_argument("--human-goal-max-distance", type=float, default=2200.0)
    p.add_argument("--human-path-file", type=Path, default=None)
    p.add_argument("--human-path-loop", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--human-waypoint-reach-distance", type=float, default=150.0)
    p.add_argument("--human-waypoint-stall-window", type=int, default=20)
    p.add_argument("--human-waypoint-stall-distance", type=float, default=20.0)
    p.add_argument("--keyboard-human", action="store_true")
    p.add_argument("--open-spawn", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--open-spawn-radius", type=float, default=900.0)
    p.add_argument("--min-open-clearance", type=float, default=300.0)
    p.add_argument("--open-spawn-candidates", type=int, default=128)
    p.add_argument("--ground-navmesh-tolerance", type=float, default=300.0)
    p.add_argument("--drone-navmesh-tolerance", type=float, default=800.0)
    p.add_argument("--require-visual-target", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--require-centered-target", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--use-mask-visibility", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--min-visible-ratio", type=float, default=0.001)
    p.add_argument("--target-center-tolerance", type=float, default=0.35)
    p.add_argument("--human-appearance-min", type=int, default=1)
    p.add_argument("--human-appearance-max", type=int, default=18)
    p.add_argument("--robotdog-appearance-min", type=int, default=20)
    p.add_argument("--robotdog-appearance-max", type=int, default=33)

    p.add_argument("--robotdog-ideal-follow-dist", type=float, default=2.2)
    p.add_argument("--robotdog-min-follow-dist", type=float, default=1.0)
    p.add_argument("--robotdog-max-follow-dist", type=float, default=4.0)
    p.add_argument("--robotdog-max-speed", type=float, default=1.05)
    p.add_argument("--robotdog-speed-gain", type=float, default=1.15)
    p.add_argument("--robotdog-max-lateral-speed", type=float, default=0.45)
    p.add_argument("--robotdog-max-yaw-rate", type=float, default=1.0)
    p.add_argument("--robotdog-max-turn-deg", type=float, default=30.0)
    p.add_argument("--robotdog-camera-forward", type=float, default=140.0)
    p.add_argument("--robotdog-camera-lateral", type=float, default=0.0)
    p.add_argument("--robotdog-camera-height", type=float, default=110.0)
    p.add_argument("--robotdog-camera-mounts", default="140:0:110,170:0:120,110:0:95,0:120:110,0:90:100,40:90:110,40:-90:110")
    p.add_argument("--robotdog-camera-fixed-pitch", type=float, default=None)
    p.add_argument("--robotdog-camera-pitches", default="-15,-8,0,8,15,22,-22")
    p.add_argument("--robotdog-camera-yaw-offsets", default="0,-8,8,-15,15")
    p.add_argument("--robotdog-camera-mode", choices=["fixed", "oracle"], default="fixed")
    p.add_argument("--robotdog-fov", type=float, default=95.0)
    p.add_argument("--max-self-visible-ratio", type=float, default=0.015)

    p.add_argument("--drone-ideal-follow-dist", type=float, default=4.0)
    p.add_argument("--drone-min-follow-dist", type=float, default=3.0)
    p.add_argument("--drone-max-follow-dist", type=float, default=5.5)
    p.add_argument("--drone-height", type=float, default=600.0)
    p.add_argument("--drone-max-speed", type=float, default=0.12)
    p.add_argument("--drone-max-vx", type=float, default=0.12)
    p.add_argument("--drone-max-vy", type=float, default=0.05)
    p.add_argument("--drone-max-yaw-rate", type=float, default=0.0)
    p.add_argument("--drone-camera-fixed-pitch", type=float, default=-60.0)
    p.add_argument("--drone-camera-pitches", default="-60")
    p.add_argument("--drone-camera-fixed-yaw", type=float, default=0.0)
    p.add_argument("--drone-camera-yaw-offsets", default="0")
    p.add_argument("--drone-camera-mode", choices=["fixed", "oracle"], default="fixed")
    p.add_argument("--lock-drone-camera-world-xy", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--drone-camera-z-offset", type=float, default=0.0)
    p.add_argument("--drone-fov", type=float, default=100.0)
    p.add_argument("--max-camera-search-candidates", type=int, default=12)
    p.add_argument("--snap-heading", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--follow-behind", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--face-target-before-step", action="store_true")
    p.add_argument(
        "--init-from-recorded-agent-poses",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For recorded target eval, restore drone/robotdog first-frame poses and cameras when present.",
    )

    p.add_argument("--waypoint-index", type=int, default=9)
    p.add_argument("--drone-waypoint-index", type=int, default=9)
    p.add_argument("--robotdog-waypoint-index", type=int, default=9)
    p.add_argument("--waypoint-horizon-steps", type=int, default=9)
    p.add_argument("--drone-vx-scale", type=float, default=0.12)
    p.add_argument("--drone-vy-scale", type=float, default=0.1)
    p.add_argument("--drone-yaw-sign", type=float, default=1.0)
    p.add_argument("--drone-yaw-scale", type=float, default=3.0)
    p.add_argument("--robotdog-yaw-sign", type=float, default=1.0)
    p.add_argument("--drone-success-distance", type=float, default=5.5)
    p.add_argument("--robotdog-success-distance", type=float, default=8.0)
    p.add_argument("--drone-lost-distance", type=float, default=5.5)
    p.add_argument("--robotdog-lost-distance", type=float, default=8.0)
    p.add_argument("--max-lost-steps", type=int, default=20)
    p.add_argument("--max-failure-steps", type=int, default=50)
    p.add_argument("--failure-warmup-steps", type=int, default=20)
    p.add_argument("--max-episode-seconds", type=float, default=0.0)
    p.add_argument("--success-rate-threshold", type=float, default=0.5)
    p.add_argument("--min-success-steps", type=int, default=20)
    p.add_argument(
        "--target-replay-mode",
        choices=["nav_goal", "pose", "path_goal"],
        default="path_goal",
        help="How to replay recorded target trajectories during closed-loop multi-agent eval.",
    )
    p.add_argument("--target-path-min-spacing", type=float, default=100.0)
    p.add_argument("--target-path-reach-distance", type=float, default=120.0)
    p.add_argument(
        "--target-goal-reach-distance",
        type=float,
        default=50.0,
        help="Final target-goal threshold in Unreal units; 50 equals Habitat's 0.5 m.",
    )
    p.add_argument("--target-stop-wait-min-steps", type=int, default=5)
    p.add_argument("--target-stop-wait-max-steps", type=int, default=15)
    p.add_argument("--bbox-source", choices=["none", "ground_truth", "model"], default="none")
    args = p.parse_args()
    args.out_dir = Path(args.save_path)
    return args


def main() -> int:
    args = parse_args()
    if args.deterministic_step:
        os.environ["UNREALZOO_FIXED_TIMESTEP"] = str(float(args.dt))
    else:
        os.environ.pop("UNREALZOO_FIXED_TIMESTEP", None)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    Path(args.save_path).mkdir(parents=True, exist_ok=True)
    reset_planner_debug_file(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    target_trajectories = None
    if args.recorded_target_dir:
        target_trajectories = load_recorded_target_trajectories(
            Path(args.recorded_target_dir),
            _parse_episode_filter(args.recorded_target_episodes),
            env_id=args.env_id,
        )
        print(f"[init] loaded {len(target_trajectories)} recorded target trajectories", flush=True)

    print(f"[init] base eval env={args.env_id} ckpt={args.ckpt} save={args.save_path}", flush=True)
    planner = BaseUnrealZooMultiAgentPlanner(args)
    env = make_env(args)
    rng = random.Random(args.seed)
    saved = 0
    attempts = 0
    max_attempts = int(args.max_attempts) if args.max_attempts is not None else max(args.episodes * 3, args.episodes)
    try:
        while saved < args.episodes and attempts < max_attempts:
            attempts += 1
            episode_id = saved
            target_trajectory = (
                target_trajectories[saved % len(target_trajectories)]
                if target_trajectories is not None
                else None
            )
            print(f"[episode {episode_id}] attempt={attempts}", flush=True)
            try:
                result = run_episode(env, args, planner, episode_id, rng, target_trajectory=target_trajectory)
            except Exception as exc:
                # Preserve EpisodeSkipped behavior without importing the class twice.
                if exc.__class__.__name__ == "EpisodeSkipped":
                    print(f"[episode {episode_id}] skipped: {exc}", flush=True)
                    continue
                raise
            write_episode_outputs(args, result)
            print(f"[episode {episode_id}] status={result['stat']['status']} TR={result['stat']['joint_following_rate']:.3f}", flush=True)
            saved += 1
    finally:
        try:
            env.close()
        except Exception:
            pass
    return 0 if saved >= args.episodes else 1


if __name__ == "__main__":
    raise SystemExit(main())
