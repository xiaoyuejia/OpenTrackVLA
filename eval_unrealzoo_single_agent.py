#!/usr/bin/env python3
"""Evaluate a single-agent OpenTrackVLA checkpoint in UnrealZoo."""

from __future__ import annotations

import argparse
import json
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

import eval_unrealzoo_multi_agent as common
from model import ModelConfig, OpenTrackVLA
from tools.cache_gridpool import VisionCacheConfig, VisionFeatureCacher, grid_pool_tokens


def resolve_manifest_item_path(
    manifest_path: Path,
    manifest: dict[str, Any],
    split_name: str,
    item: dict[str, Any],
    key: str,
) -> Path:
    """Find a split item file, preferring materialized split_raw over original raw."""
    if key in item:
        relative_path = Path(item[key])
    elif key in {"drone_info", "robotdog_info"}:
        relative_dir = Path(str(item["relative_dir"]))
        relative_path = relative_dir / f"{item['stem']}_{key}.json"
    else:
        raise KeyError(
            f"Manifest item has no {key!r} path and it cannot be inferred: {item}"
        )
    output_root = Path(manifest.get("output_root", manifest_path.parent))
    candidates = [
        manifest_path.parent / f"{split_name}_raw" / relative_path,
        output_root / f"{split_name}_raw" / relative_path,
    ]
    input_root = manifest.get("input_root")
    if input_root:
        candidates.append(Path(input_root) / relative_path)

    for path in candidates:
        if path.is_file():
            return path

    checked = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Cannot find {split_name} {key} for {item.get('scene')} {item.get('stem')}.\nChecked:\n  {checked}")


def load_test_trajectories(
    manifest_path: Path,
    env_id: str,
    agent: str,
) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    trajectories: list[dict[str, Any]] = []
    for item in manifest.get("test", []):
        if item.get("scene") != env_id:
            continue
        info_key = "info" if "info" in item else f"{agent}_info"
        info_path = resolve_manifest_item_path(
            manifest_path,
            manifest,
            "test",
            item,
            info_key,
        )
        records = json.loads(info_path.read_text(encoding="utf-8"))
        poses = [
            [float(value) for value in record["target_pose"][:6]]
            for record in records
            if isinstance(record, dict) and isinstance(record.get("target_pose"), list)
        ]
        if not poses:
            raise ValueError(f"No target_pose records in {info_path}")
        first = records[0] if records and isinstance(records[0], dict) else {}
        robotdog_pose = first.get("robotdog_pose")
        drone_pose = first.get("drone_pose")
        robotdog_camera = {
            "mount": first.get("robotdog_camera_mount"),
            "pitch": first.get("robotdog_camera_pitch"),
            "yaw_offset": first.get("robotdog_camera_yaw_offset"),
        }
        drone_camera = {
            "pitch": first.get("drone_camera_pitch"),
            "yaw_offset": first.get("drone_camera_yaw_offset"),
        }
        trajectories.append(
            {
                "episode_name": f"{item['relative_dir']}/{item['stem']}",
                "source": str(info_path),
                "poses": poses,
                "robotdog_pose": [float(value) for value in robotdog_pose[:6]]
                if isinstance(robotdog_pose, list) and len(robotdog_pose) >= 6
                else None,
                "robotdog_camera": robotdog_camera,
                "drone_pose": [float(value) for value in drone_pose[:6]]
                if isinstance(drone_pose, list) and len(drone_pose) >= 6
                else None,
                "drone_camera": drone_camera,
            }
        )
    if not trajectories:
        raise ValueError(f"No test trajectories for env_id={env_id} in {manifest_path}")
    return trajectories


def body_velocity_from_pose_delta(
    prev_pose: list[float],
    current_pose: list[float],
    dt: float,
) -> list[float]:
    """Estimate executed [vx, vy, yaw_rate] in the agent local frame."""
    dt = max(float(dt), 1e-6)
    prev_xy = np.asarray(prev_pose[:2], dtype=np.float64)
    curr_xy = np.asarray(current_pose[:2], dtype=np.float64)
    delta_m = (curr_xy - prev_xy) / (common.UNREAL_UNITS_PER_METER * dt)
    yaw = math.radians(common.yaw_deg(prev_pose))
    forward = np.asarray([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
    right = np.asarray([-math.sin(yaw), math.cos(yaw)], dtype=np.float64)
    prev_yaw = common.yaw_deg(prev_pose)
    curr_yaw = common.yaw_deg(current_pose)
    delta_yaw = (curr_yaw - prev_yaw + 180.0) % 360.0 - 180.0
    return [
        float(np.dot(delta_m, forward)),
        float(np.dot(delta_m, right)),
        float(math.radians(delta_yaw) / dt),
    ]


def read_live_agent_pose(env, agent_name: str) -> list[float]:
    """Read the action-dispatch pose without refreshing the camera frame."""
    unrealcv = env.unwrapped.unrealcv
    location = unrealcv.get_obj_location(agent_name)
    rotation = unrealcv.get_obj_rotation(agent_name)
    return [float(value) for value in [*location[:3], *rotation[:3]]]


class UnrealZooSingleAgentPlanner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt_path = common._latest_checkpoint(Path(args.ckpt))
        if ckpt_path is None:
            raise FileNotFoundError(f"No checkpoint found from --ckpt: {args.ckpt}")
        self.ckpt_path = ckpt_path
        try:
            obj = torch.load(str(ckpt_path), map_location="cpu")
        except Exception:
            obj = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        ckpt_cfg = obj.get("config", {}) if isinstance(obj, dict) else {}
        state = common._cleanup_state_dict_keys(obj.get("model_state", {}) if isinstance(obj, dict) else {})
        self.history = int(ckpt_cfg.get("history", args.history))
        self.history_frame_dt = float(
            args.history_frame_dt if args.history_frame_dt > 0.0 else args.dt
        )
        self.history_sampling_mode = str(args.history_sampling_mode)
        self.n_waypoints = int(ckpt_cfg.get("n_waypoints", args.n_waypoints))
        vision_feat_dim = int(ckpt_cfg.get("vision_feat_dim", args.vision_feat_dim))
        model_cfg = ModelConfig(
            llm_name=str(ckpt_cfg.get("llm_name", args.llm_name)),
            freeze_llm=True,
            n_waypoints=self.n_waypoints,
            use_angle_tvi=bool(ckpt_cfg.get("use_angle_tvi", False)),
            use_tanh_actions=not bool(ckpt_cfg.get("no_tanh_actions", True)),
            alpha_xy=ckpt_cfg.get("alpha_xy", args.alpha_xy),
        )
        self.model = OpenTrackVLA(model_cfg, vision_feat_dim=vision_feat_dim).to(self.device).eval()
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if not any(key.startswith("planner.") for key in state):
            raise ValueError("Checkpoint does not look like a single-agent OpenTrackVLA checkpoint")
        optional_missing = {"tvi.bbox_proj.weight", "tvi.bbox_proj.bias"}
        critical = [
            key
            for key in missing
            if key not in optional_missing
            and (key.startswith(("planner.", "proj.", "tvi.")) or key == "act_token")
        ]
        if critical:
            raise RuntimeError(f"Checkpoint is missing critical weights: {critical[:20]}")
        print(
            f"[planner] loaded={ckpt_path} history={self.history} n_waypoints={self.n_waypoints} "
            f"history_mode={self.history_sampling_mode} history_frame_dt={self.history_frame_dt:.3f}s "
            f"missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )
        self.encoder = VisionFeatureCacher(
            VisionCacheConfig(image_size=args.image_size, batch_size=1, device=str(self.device))
        ).eval()
        self.history_tokens: deque[tuple[float, torch.Tensor]] = deque(
            maxlen=max(self.history * 4, self.history + 1)
        )

    def reset(self) -> None:
        self.history_tokens.clear()

    @torch.inference_mode()
    def _encode(self, frame_bgr: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        rgb = cv2.cvtColor(common.ensure_bgr_uint8(frame_bgr), cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        tok_dino, hp, wp = self.encoder._encode_dino([pil])
        tok_sigl = self.encoder._encode_siglip([pil], out_hw=(hp, wp))
        tokens = torch.cat([tok_dino, tok_sigl], dim=-1)
        fine = grid_pool_tokens(tokens, hp, wp, out_tokens=64)[0].float().cpu()
        coarse = grid_pool_tokens(tokens, hp, wp, out_tokens=4)[0].float().cpu()
        return coarse, fine

    @torch.inference_mode()
    def predict(
        self,
        frame_bgr: np.ndarray,
        instruction: str,
        observation_time: Optional[float] = None,
    ) -> np.ndarray:
        current_coarse, current_fine = self._encode(frame_bgr)
        if observation_time is None:
            observation_time = time.monotonic()
        previous = list(self.history_tokens)
        if previous:
            if self.history_sampling_mode == "inference_steps":
                previous_tokens = [entry[1] for entry in previous]
                frames = (
                    [previous_tokens[0]] * max(0, self.history - len(previous_tokens))
                    + previous_tokens
                )[-self.history :]
            else:
                target_times = [
                    observation_time - (self.history - index) * self.history_frame_dt
                    for index in range(self.history)
                ]
                frames = []
                previous_index = 0
                for target_time in target_times:
                    while (
                        previous_index + 1 < len(previous)
                        and previous[previous_index + 1][0] <= target_time
                    ):
                        previous_index += 1
                    frames.append(previous[previous_index][1])
        else:
            frames = [current_coarse] * self.history
        coarse = torch.cat(frames, dim=0).unsqueeze(0).to(self.device)
        coarse_tidx = torch.cat(
            [torch.full((token.size(0),), idx, dtype=torch.long) for idx, token in enumerate(frames)]
        ).unsqueeze(0).to(self.device)
        fine = current_fine.unsqueeze(0).to(self.device)
        fine_tidx = torch.full((1, current_fine.size(0)), self.history, dtype=torch.long, device=self.device)
        pred = self.model(coarse, coarse_tidx, fine, fine_tidx, [instruction])
        self.history_tokens.append((float(observation_time), current_coarse))
        return pred.detach().float().cpu().numpy()[0]

    def action_from_waypoints(
        self,
        waypoints: np.ndarray,
        realtime_control_period_seconds: Optional[float] = None,
    ) -> tuple[list[float], list[float], dict[str, Any]]:
        waypoint_count = int(waypoints.shape[0])
        horizon_steps = max(1, int(self.args.waypoint_horizon_steps))

        def waypoint_time(index: int) -> tuple[int, float]:
            if self.args.waypoint_command_dt > 0.0:
                return max(index, 0), float(self.args.waypoint_command_dt)
            # train.py ignores the JSON `trajectory` field when `actions` are
            # present and reintegrates ten actions into ten outputs with
            # waypoint[0] fixed at the local origin. Therefore index 9 is the
            # state after nine dt intervals, not ten.
            source_step = max(index, 0)
            return source_step, max(source_step * float(self.args.dt), 1e-6)

        raw_period = None
        clipped_period = None
        if self.args.realtime_waypoint_timing:
            if realtime_control_period_seconds is None:
                raise ValueError("Realtime waypoint timing requires an observed control period")
            raw_period = max(0.0, float(realtime_control_period_seconds))
            clipped_period = float(
                np.clip(
                    raw_period,
                    self.args.realtime_waypoint_min_seconds,
                    self.args.realtime_waypoint_max_seconds,
                )
            )
            idx = min(
                range(waypoint_count),
                key=lambda index: abs(waypoint_time(index)[1] - clipped_period),
            )
        else:
            idx = int(np.clip(self.args.waypoint_index, 0, waypoint_count - 1))

        source_step, horizon_dt = waypoint_time(idx)
        velocity = waypoints[idx, :3] / horizon_dt
        action_debug = {
            "waypoint_index": int(idx),
            "waypoint_source_step": int(source_step),
            "waypoint_horizon_dt": float(horizon_dt),
            "realtime_waypoint_timing": bool(self.args.realtime_waypoint_timing),
            "realtime_control_period_seconds_raw": raw_period,
            "realtime_control_period_seconds_clipped": clipped_period,
        }
        if self.args.agent == "drone":
            vx = float(
                np.clip(
                    velocity[0] * float(self.args.drone_vx_scale),
                    -self.args.drone_max_vx,
                    self.args.drone_max_vx,
                )
            )
            vy = float(
                np.clip(
                    velocity[1] * float(self.args.drone_vy_scale),
                    -self.args.drone_max_vy,
                    self.args.drone_max_vy,
                )
            )
            yaw_rate = float(
                np.clip(
                    velocity[2]
                    * float(self.args.drone_yaw_sign)
                    * float(self.args.drone_yaw_scale),
                    -self.args.drone_max_yaw_rate,
                    self.args.drone_max_yaw_rate,
                )
            )
            return (
                [vx, vy, 0.0, yaw_rate],
                [float(value) for value in velocity.tolist()],
                action_debug,
            )

        speed = float(
            np.clip(
                velocity[0] * common.UNREAL_UNITS_PER_METER * float(self.args.robotdog_speed_gain),
                -self.args.robotdog_max_speed * common.UNREAL_UNITS_PER_METER,
                self.args.robotdog_max_speed * common.UNREAL_UNITS_PER_METER,
            )
        )
        turn = float(
            np.clip(
                math.degrees(
                    velocity[2]
                    * self.args.robotdog_yaw_sign
                    * self.args.robotdog_yaw_scale
                ),
                -self.args.robotdog_max_turn_deg,
                self.args.robotdog_max_turn_deg,
            )
        )
        return [turn, speed], [float(value) for value in velocity.tolist()], action_debug


def restore_recorded_robotdog_initial_state(
    env,
    args: argparse.Namespace,
    setup: dict[str, Any],
    trajectory: dict[str, Any],
) -> bool:
    """Restore RobotDog's first-frame test pose and camera, if present."""
    robotdog_pose = trajectory.get("robotdog_pose")
    if not isinstance(robotdog_pose, list) or len(robotdog_pose) < 6:
        return False

    robotdog_name = setup["robotdog_name"]
    env.unwrapped.unrealcv.set_obj_location(robotdog_name, robotdog_pose[:3])
    try:
        env.unwrapped.unrealcv.set_obj_rotation(robotdog_name, robotdog_pose[3:6])
    except Exception:
        try:
            common.set_ground_yaw(env, robotdog_name, float(robotdog_pose[4]))
        except Exception:
            pass

    camera = trajectory.get("robotdog_camera") or {}
    mount = camera.get("mount")
    pitch = camera.get("pitch")
    yaw_offset = camera.get("yaw_offset")
    if isinstance(mount, list) and len(mount) >= 3:
        setup["dog_camera"]["mount"] = [float(value) for value in mount[:3]]
    if isinstance(pitch, (int, float)):
        setup["dog_camera"]["pitch"] = float(pitch)
    if isinstance(yaw_offset, (int, float)):
        setup["dog_camera"]["yaw_offset"] = float(yaw_offset)

    setup["initial_pose"]["robotdog"] = list(robotdog_pose)
    common._lock_eval_cameras(env, setup, args)
    common.update_observation(env, refresh_cameras=False)
    return True


def restore_recorded_drone_initial_state(
    env,
    args: argparse.Namespace,
    setup: dict[str, Any],
    trajectory: dict[str, Any],
) -> bool:
    """Restore Drone's first-frame test pose and camera, if present."""
    drone_pose = trajectory.get("drone_pose")
    if not isinstance(drone_pose, list) or len(drone_pose) < 6:
        return False

    drone_name = setup["drone_name"]
    env.unwrapped.unrealcv.set_obj_location(drone_name, drone_pose[:3])
    try:
        env.unwrapped.unrealcv.set_obj_rotation(drone_name, drone_pose[3:6])
    except Exception:
        try:
            common.maybe_set_drone_yaw(env, drone_name, float(drone_pose[4]))
        except Exception:
            pass

    camera = trajectory.get("drone_camera") or {}
    pitch = camera.get("pitch")
    yaw_offset = camera.get("yaw_offset")
    if isinstance(pitch, (int, float)):
        setup["drone_camera"]["pitch"] = float(pitch)
    if isinstance(yaw_offset, (int, float)):
        setup["drone_camera"]["yaw_offset"] = float(yaw_offset)

    setup["initial_pose"]["drone"] = list(drone_pose)
    common._lock_eval_cameras(env, setup, args)
    common.update_observation(env, refresh_cameras=False)
    return True


def restore_recorded_agent_initial_state(
    env,
    args: argparse.Namespace,
    setup: dict[str, Any],
    trajectory: dict[str, Any],
) -> bool:
    if args.agent == "drone":
        return restore_recorded_drone_initial_state(env, args, setup, trajectory)
    return restore_recorded_robotdog_initial_state(env, args, setup, trajectory)


def restore_recorded_target_initial_state(env, setup: dict[str, Any], trajectory: dict[str, Any]) -> bool:
    """Restore target to the first recorded pose after setup camera searches."""
    poses = trajectory.get("poses")
    if not isinstance(poses, list) or not poses:
        return False
    first_pose = poses[0]
    if not isinstance(first_pose, list) or len(first_pose) < 6:
        return False

    target_name = setup["target_name"]
    env.unwrapped.unrealcv.set_obj_location(target_name, first_pose[:3])
    try:
        env.unwrapped.unrealcv.set_obj_rotation(target_name, first_pose[3:6])
    except Exception:
        pass
    try:
        # Keep the human fixed for the first input frame. In nav_goal mode,
        # navigation is started only after the model has consumed that frame.
        env.unwrapped.unrealcv.nav_to_goal(target_name, first_pose[:3])
    except Exception:
        pass
    setup["initial_pose"]["target"] = list(first_pose)
    return True


def settle_initial_state(env, args: argparse.Namespace, setup: dict[str, Any]) -> int:
    """Let Unreal update animation/camera buffers before the model sees frame 0."""
    steps = max(int(args.settle_steps), 0)
    for _ in range(steps):
        try:
            common._lock_eval_cameras(env, setup, args)
            actions = [None for _ in env.unwrapped.player_list]
            common.data_collection_step(env, actions)
        except Exception:
            break
    if steps > 0:
        common._lock_eval_cameras(env, setup, args)
        common.update_observation(env, refresh_cameras=False)
    return steps


def flush_initial_observation(env, args: argparse.Namespace, setup: dict[str, Any]) -> bool:
    """Flush UnrealCV RGB/object-mask buffers after teleporting actors/cameras."""
    if not args.flush_initial_observation:
        return False
    try:
        common._lock_eval_cameras(env, setup, args)
        common.update_observation(env, refresh_cameras=False)
        common.target_mask_visibility(
            env,
            env.unwrapped.cam_list[setup[f"{args.agent}_id"]],
            setup["target_name"],
        )
        common.update_observation(env, refresh_cameras=False)
        return True
    except Exception:
            return False


def _target_path_waypoints(trajectory: dict[str, Any], args: argparse.Namespace) -> list[list[float]]:
    poses = trajectory.get("poses") or []
    if not poses:
        return []
    min_spacing = max(float(args.target_path_min_spacing), 0.0)
    waypoints: list[list[float]] = []
    last_xy: np.ndarray | None = None
    for pose in poses:
        if not isinstance(pose, list) or len(pose) < 3:
            continue
        xyz = [float(value) for value in pose[:3]]
        xy = np.asarray(xyz[:2], dtype=np.float64)
        if last_xy is None or float(np.linalg.norm(xy - last_xy)) >= min_spacing:
            waypoints.append(xyz)
            last_xy = xy
    final_pose = poses[-1]
    if isinstance(final_pose, list) and len(final_pose) >= 3:
        final_xyz = [float(value) for value in final_pose[:3]]
        if not waypoints or float(np.linalg.norm(np.asarray(final_xyz[:2]) - np.asarray(waypoints[-1][:2]))) > 1e-6:
            waypoints.append(final_xyz)
    return waypoints


def start_recorded_path_navigation(env, args: argparse.Namespace, setup: dict[str, Any], trajectory: dict[str, Any]) -> None:
    waypoints = _target_path_waypoints(trajectory, args)
    setup["target_path_waypoints"] = waypoints
    setup["target_path_index"] = 0
    if len(waypoints) < 2:
        return
    try:
        env.unwrapped.unrealcv.set_max_speed(setup["target_name"], float(args.human_speed))
    except Exception:
        pass
    setup["target_path_index"] = 1


def update_recorded_path_navigation(env, args: argparse.Namespace, setup: dict[str, Any]) -> None:
    waypoints = setup.get("target_path_waypoints") or []
    if len(waypoints) < 2:
        return
    target_pose = list(env.unwrapped.obj_poses[setup["target_id"]])
    target_xy = np.asarray(target_pose[:2], dtype=np.float64)
    reach = max(float(args.target_path_reach_distance), 1.0)
    current_index = int(setup.get("target_path_index", 1))

    while current_index < len(waypoints) - 1:
        goal_xy = np.asarray(waypoints[current_index][:2], dtype=np.float64)
        if float(np.linalg.norm(goal_xy - target_xy)) > reach:
            break
        current_index += 1

    if current_index != int(setup.get("target_path_index", 1)):
        setup["target_path_index"] = current_index
        try:
            env.unwrapped.unrealcv.nav_to_goal(setup["target_name"], waypoints[current_index])
        except Exception:
            pass


def run_episode(
    env,
    args: argparse.Namespace,
    planner: UnrealZooSingleAgentPlanner,
    episode_id: int,
    rng: random.Random,
    trajectory: dict[str, Any],
) -> dict[str, Any]:
    setup = common.setup_episode(env, args, episode_id, rng, target_trajectory=trajectory)
    restored_target_initial_state = restore_recorded_target_initial_state(env, setup, trajectory)
    restored_initial_state = False
    if args.init_from_recorded_agent_pose:
        restored_initial_state = restore_recorded_agent_initial_state(env, args, setup, trajectory)
        if not restored_initial_state:
            raise common.EpisodeSkipped(
                f"recorded Habitat-style initialization requires a first-frame {args.agent} pose"
            )
    settled_steps = settle_initial_state(env, args, setup)
    flushed_initial_observation = flush_initial_observation(env, args, setup)
    if args.target_replay_mode == "path_goal":
        start_recorded_path_navigation(env, args, setup, trajectory)
    planner.reset()
    # Keep UE live regardless of waypoint-selection policy.  Disabling
    # realtime_waypoint_timing now means "use the configured waypoint index",
    # not "pause the simulator".
    env.unwrapped.unrealcv.set_global_time_dilation(1.0)
    env.unwrapped.unrealcv.set_max_FPS(float(args.fps))
    env.unwrapped.unrealcv.set_resume()
    frames: list[np.ndarray] = []
    infos: list[dict[str, Any]] = []
    collision = False
    lost_count = 0
    status = "Normal"
    last_info = None
    start_time = time.time()
    # Keep evaluating after the target reaches its goal so the model must react
    # to a stationary person, matching the Habitat 5-15 step stop window.
    episode_steps = args.max_steps
    agent_name = args.agent
    agent_id = setup[f"{agent_name}_id"]
    success_distance = float(getattr(args, f"{agent_name}_success_distance"))
    lost_distance = float(getattr(args, f"{agent_name}_lost_distance", 0.0))
    overlay_name = "dog" if agent_name == "robotdog" else "drone"
    previous_observation_wall_time: Optional[float] = None

    for step_idx in range(episode_steps):
        if bool(setup.get("target_stopped", False)):
            common.update_habitat_target_stop_state(
                env, args, setup, step_idx=step_idx
            )
        elif args.target_replay_mode == "pose":
            current_pose = trajectory["poses"][min(step_idx, len(trajectory["poses"]) - 1)]
            common._set_recorded_target_pose(env, setup, current_pose)
        elif args.target_replay_mode == "path_goal":
            update_recorded_path_navigation(env, args, setup)
        target_stopped = common.update_habitat_target_stop_state(
            env,
            args,
            setup,
            step_idx=step_idx,
            recorded_pose_exhausted=bool(
                args.target_replay_mode == "pose"
                and step_idx >= len(trajectory["poses"]) - 1
            ),
        )
        if args.face_target_before_step:
            target_pose_before = list(env.unwrapped.obj_poses[setup["target_id"]])
            if agent_name == "drone":
                common.maybe_set_drone_yaw(
                    env,
                    setup["drone_name"],
                    common.heading_deg(
                        common.pose_xyz(list(env.unwrapped.obj_poses[agent_id])),
                        common.pose_xyz(target_pose_before),
                    ),
                )
            else:
                common.set_ground_yaw(
                    env,
                    setup["robotdog_name"],
                    common.heading_deg(
                        common.pose_xyz(list(env.unwrapped.obj_poses[agent_id])),
                        common.pose_xyz(target_pose_before),
                    ),
                )
        common._lock_eval_cameras(env, setup, args)
        drone_input, dog_input = common._read_agent_pair(env, setup, args)
        agent_input = drone_input if agent_name == "drone" else dog_input
        input_frame, input_visibility, input_visible, input_bbox, _input_bbox_norm = agent_input
        observation_wall_time = time.monotonic()
        if previous_observation_wall_time is None:
            # The first action has no previous observation interval. Use the
            # longest trained horizon because it remains active until the next
            # observation completes.
            realtime_control_period_seconds = float(
                args.realtime_waypoint_max_seconds
            )
        else:
            # Include inference, simulator stepping, camera reads, visibility
            # checks, and rendering from the preceding control iteration.
            realtime_control_period_seconds = (
                observation_wall_time - previous_observation_wall_time
            )
        previous_observation_wall_time = observation_wall_time
        input_agent_pose = list(env.unwrapped.obj_poses[agent_id])
        input_target_pose = list(env.unwrapped.obj_poses[setup["target_id"]])
        input_distance = common.distance_xy_m(input_agent_pose, input_target_pose)
        waypoints = planner.predict(
            input_frame,
            args.instruction,
            observation_time=observation_wall_time,
        ) # 预测路点
        observation_to_action_seconds = time.monotonic() - observation_wall_time
        pre_action_agent_pose = read_live_agent_pose(env, setup[f"{agent_name}_name"])
        agent_action, predicted_velocity, action_debug = planner.action_from_waypoints(
            waypoints,
            realtime_control_period_seconds=realtime_control_period_seconds,
        )
        action_debug["observation_to_action_seconds"] = float(
            observation_to_action_seconds
        )
        action_debug["ego_motion_compensated"] = False
        action_debug["inference_motion_delta_m"] = float(
            np.linalg.norm(
                np.asarray(pre_action_agent_pose[:2], dtype=np.float64)
                - np.asarray(input_agent_pose[:2], dtype=np.float64)
            )
            / common.UNREAL_UNITS_PER_METER
        )

        if args.target_replay_mode == "nav_goal" and step_idx == 0:
            try:
                env.unwrapped.unrealcv.set_max_speed(setup["target_name"], float(args.human_speed))
                env.unwrapped.unrealcv.nav_to_goal(setup["target_name"], setup["target_goal"])
            except Exception:
                pass
        elif args.target_replay_mode == "path_goal" and not target_stopped:
            # Start/resume the human only after the model consumed this frame.
            common.resume_recorded_path_navigation(env, setup)

        actions = [None for _ in env.unwrapped.player_list]
        actions[agent_id] = agent_action
        # 去环境执行动作并收集数据
        _obs, _rewards, done, last_info = common.data_collection_step(env, actions)
        if args.target_replay_mode == "pose" and not target_stopped:
            next_pose = trajectory["poses"][min(step_idx + 1, len(trajectory["poses"]) - 1)]
            common._set_recorded_target_pose(env, setup, next_pose)
        common._lock_eval_cameras(env, setup, args)
        drone_after, dog_after = common._read_agent_pair(env, setup, args)
        agent_after = drone_after if agent_name == "drone" else dog_after
        _agent_frame, visibility, visible, bbox, _bbox_norm = agent_after
        agent_pose = list(env.unwrapped.obj_poses[agent_id])
        target_pose = list(env.unwrapped.obj_poses[setup["target_id"]])
        distance = common.distance_xy_m(agent_pose, target_pose)
        executed_base_velocity = body_velocity_from_pose_delta(
            pre_action_agent_pose, agent_pose, args.dt
        )
        agent_motion_delta_m = float(
            np.linalg.norm(
                np.asarray(agent_pose[:2], dtype=np.float64)
                - np.asarray(input_agent_pose[:2], dtype=np.float64)
            )
            / common.UNREAL_UNITS_PER_METER
        )
        if agent_name == "drone":
            step_collision = common.drone_collision_from_info(
                last_info, agent_id, setup["target_id"], distance, agent_pose, target_pose
            )
        else:
            step_collision = common.robotdog_collision_from_info(
                last_info, agent_id, setup["target_id"], distance, agent_pose, target_pose
            )
        collision = collision or step_collision
        in_success_distance = bool(distance <= success_distance)
        following = bool(visible and (in_success_distance or not args.require_success_distance))
        if following:
            lost_count = 0
        elif lost_distance > 0.0 and distance > lost_distance:
            lost_count += 1
        elif lost_distance <= 0.0:
            # Backward-compatible behavior: every non-following frame counts.
            lost_count += 1

        rendered = input_frame
        if args.trajectory_overlay:
            rendered = common._render_bgr_frame_with_traj(rendered, waypoints, args.trajectory_scale)
        if agent_name == "drone":
            action_text = f"a=[vx {agent_action[0]:.2f}, vy {agent_action[1]:.2f}, yaw {agent_action[3]:.2f}]"
        else:
            action_text = f"a=[turn {agent_action[0]:.1f}, speed {agent_action[1]:.1f}]"
        rendered = common._overlay_text(
            rendered,
            [
                f"ep={episode_id} step={step_idx + 1}",
                f"{overlay_name} d={distance:.2f} visible={int(visible)} near={int(in_success_distance)} following={int(following)}",
                action_text,
            ],
        )
        if args.save_video:
            frames.append(rendered)
        target_motion_vec = (
            np.asarray(target_pose[:2], dtype=np.float64)
            - np.asarray(input_target_pose[:2], dtype=np.float64)
        )
        target_motion_step_m = float(np.linalg.norm(target_motion_vec) / common.UNREAL_UNITS_PER_METER)
        relative_forward_m = None
        behind_target = None
        if target_motion_step_m > 0.01:
            target_forward = target_motion_vec / max(float(np.linalg.norm(target_motion_vec)), 1e-6)
            agent_relative_xy = (
                np.asarray(agent_pose[:2], dtype=np.float64)
                - np.asarray(target_pose[:2], dtype=np.float64)
            )
            relative_forward_m = float(np.dot(agent_relative_xy, target_forward) / common.UNREAL_UNITS_PER_METER)
            behind_target = bool(relative_forward_m <= 0.0)
        info = {
            "step": step_idx + 1,
            "agent": agent_name,
            "dis_to_human": float(distance),
            "dis_to_human_3d": float(common.distance_m(agent_pose, target_pose)),
            "target_visible": bool(visible),
            "target_visibility": float(visibility),
            "target_bbox": bbox,
            "input_dis_to_human": float(input_distance),
            "input_target_visible": bool(input_visible),
            "input_target_visibility": float(input_visibility),
            "input_target_bbox": input_bbox,
            f"input_{agent_name}_pose": input_agent_pose,
            "input_target_pose": input_target_pose,
            "predicted_waypoints": waypoints.tolist(),
            "predicted_base_velocity": predicted_velocity,
            "action_debug": action_debug,
            "realtime_control_period_seconds": float(
                realtime_control_period_seconds
            ),
            "executed_base_velocity": executed_base_velocity,
            f"{agent_name}_motion_delta_m": agent_motion_delta_m,
            "base_velocity": executed_base_velocity,
            "agent_action": [float(value) for value in agent_action],
            "following": following,
            "in_success_distance": in_success_distance,
            "target_motion_step_m": target_motion_step_m,
            "relative_forward_m": relative_forward_m,
            "behind_target": behind_target,
            "front_overshoot": bool(relative_forward_m is not None and relative_forward_m > 0.0),
            "collision": bool(step_collision),
            f"{agent_name}_pose": agent_pose,
            "target_pose": target_pose,
            "target_stopped": bool(target_stopped),
            "target_stop_wait_count": int(setup.get("target_stop_wait_count", 0)),
            "target_stop_wait_steps": int(setup["target_stop_wait_steps"]),
        }
        if agent_name == "robotdog":
            info["ground_action"] = [float(value) for value in agent_action]
        else:
            info["drone_action"] = [float(value) for value in agent_action]
            info["commanded_base_velocity"] = [
                float(agent_action[0]),
                float(agent_action[1]),
                float(agent_action[3]),
            ]
        infos.append(info)
        if target_stopped:
            setup["target_stop_wait_count"] = int(setup.get("target_stop_wait_count", 0)) + 1
        if collision:
            status = "Collision"
            break
        if args.max_lost_steps > 0 and lost_count >= args.max_lost_steps:
            status = "Lost"
            break
        if args.max_episode_seconds > 0 and time.time() - start_time >= args.max_episode_seconds:
            status = "Timeout"
            break
        if (
            target_stopped
            and int(setup["target_stop_wait_count"]) >= int(setup["target_stop_wait_steps"])
        ):
            status = "TargetStopped"
            break
        if done:
            status = "EnvDone"
            break

    total_steps = len(infos)
    following_rate = sum(int(item["following"]) for item in infos) / max(total_steps, 1)
    final_following = bool(infos and infos[-1]["following"])
    final_distance = float(infos[-1]["dis_to_human"]) if infos else float("inf")
    final_follow_range_ok = bool(
        float(getattr(args, f"{agent_name}_min_follow_dist"))
        <= final_distance
        <= float(getattr(args, f"{agent_name}_max_follow_dist"))
    )
    near_rate = sum(int(item.get("in_success_distance", False)) for item in infos) / max(total_steps, 1)
    distances = [item["dis_to_human"] for item in infos]
    target_step_distances = []
    agent_step_distances = []
    relative_forward_values = [
        float(item["relative_forward_m"])
        for item in infos
        if item.get("relative_forward_m") is not None
    ]
    behind_values = [
        bool(item["behind_target"])
        for item in infos
        if item.get("behind_target") is not None
    ]
    front_crossings = 0
    if len(relative_forward_values) > 1:
        signs = [1 if value > 0.0 else -1 for value in relative_forward_values]
        front_crossings = sum(
            int(prev != curr) for prev, curr in zip(signs[:-1], signs[1:])
        )
    pose_key = f"{agent_name}_pose"
    for prev, curr in zip(infos[:-1], infos[1:]):
        prev_target = prev.get("target_pose")
        curr_target = curr.get("target_pose")
        if isinstance(prev_target, list) and isinstance(curr_target, list) and len(prev_target) >= 2 and len(curr_target) >= 2:
            target_step_distances.append(
                float(
                    math.dist(
                        [float(prev_target[0]), float(prev_target[1])],
                        [float(curr_target[0]), float(curr_target[1])],
                    )
                    / common.UNREAL_UNITS_PER_METER
                )
            )
        prev_agent = prev.get(pose_key)
        curr_agent = curr.get(pose_key)
        if isinstance(prev_agent, list) and isinstance(curr_agent, list) and len(prev_agent) >= 2 and len(curr_agent) >= 2:
            agent_step_distances.append(
                float(
                    math.dist(
                        [float(prev_agent[0]), float(prev_agent[1])],
                        [float(curr_agent[0]), float(curr_agent[1])],
                    )
                    / common.UNREAL_UNITS_PER_METER
                )
            )
    success = bool(
        not collision
        and status not in {"Lost", "Collision", "Timeout"}
        and total_steps >= args.min_success_steps
        and following_rate >= args.success_rate_threshold
        and final_following
        and (
            not args.require_final_follow_range_on_target_stop
            or not bool(setup.get("target_stopped", False))
            or final_follow_range_ok
        )
    )
    status = "Success" if success else ("Failed" if status == "Normal" else status)
    elapsed = max(time.time() - start_time, 1e-6)
    stat = {
        "finish": True,
        "status": status,
        "success": 1.0 if success else 0.0,
        "collision": 1.0 if collision else 0.0,
        "total_step": total_steps,
        f"{agent_name}_following_rate": float(following_rate),
        "following_rate": float(following_rate),
        "final_following": final_following,
        "final_follow_range_ok": final_follow_range_ok,
        "near_rate": float(near_rate),
        "avg_distance": float(np.mean(distances)) if distances else 0.0,
        "min_distance": float(np.min(distances)) if distances else 0.0,
        "final_distance": float(distances[-1]) if distances else 0.0,
        "target_step_distance_mean_m": float(np.mean(target_step_distances)) if target_step_distances else 0.0,
        "target_step_distance_p90_m": float(np.percentile(target_step_distances, 90)) if target_step_distances else 0.0,
        f"{agent_name}_step_distance_mean_m": float(np.mean(agent_step_distances)) if agent_step_distances else 0.0,
        f"{agent_name}_step_distance_p90_m": float(np.percentile(agent_step_distances, 90)) if agent_step_distances else 0.0,
        "behind_rate": float(np.mean(behind_values)) if behind_values else 0.0,
        "front_overshoot_rate": float(1.0 - np.mean(behind_values)) if behind_values else 0.0,
        "front_crossings": int(front_crossings),
        "relative_forward_mean_m": float(np.mean(relative_forward_values)) if relative_forward_values else 0.0,
        "relative_forward_min_m": float(np.min(relative_forward_values)) if relative_forward_values else 0.0,
        "relative_forward_max_m": float(np.max(relative_forward_values)) if relative_forward_values else 0.0,
        "fps": total_steps / elapsed,
        "ckpt": str(planner.ckpt_path),
        "env_id": args.env_id,
        "agent": agent_name,
        "model_type": f"single_agent_{agent_name}",
        "target_motion_mode": "recorded_pose_replay"
        if args.target_replay_mode == "pose"
        else ("recorded_path_navigation" if args.target_replay_mode == "path_goal" else "recorded_start_goal_navigation"),
        "target_replay_mode": args.target_replay_mode,
        "target_stopped": bool(setup.get("target_stopped", False)),
        "target_stop_step": setup.get("target_stop_step"),
        "target_stop_wait_count": int(setup.get("target_stop_wait_count", 0)),
        "target_stop_wait_steps": int(setup["target_stop_wait_steps"]),
        "target_goal_reach_distance": float(args.target_goal_reach_distance),
        "target_path_waypoints": len(setup.get("target_path_waypoints") or []),
        "target_path_index_final": int(setup.get("target_path_index", 0)),
        "face_target_before_step": bool(args.face_target_before_step),
        "human_speed": float(args.human_speed),
        "robotdog_speed_gain": float(args.robotdog_speed_gain),
        "robotdog_max_speed": float(args.robotdog_max_speed),
        "waypoint_index": int(args.waypoint_index),
        "waypoint_horizon_steps": int(args.waypoint_horizon_steps),
        "realtime_waypoint_timing": bool(args.realtime_waypoint_timing),
        "dt": float(args.dt),
        "history_frame_dt": float(planner.history_frame_dt),
        "history_sampling_mode": planner.history_sampling_mode,
        "waypoint_command_dt": (
            float(args.waypoint_command_dt)
            if args.waypoint_command_dt > 0.0
            else None
        ),
        "init_from_recorded_agent_pose": bool(args.init_from_recorded_agent_pose),
        f"restored_recorded_{agent_name}_initial_state": bool(restored_initial_state),
        "restored_recorded_target_initial_state": bool(restored_target_initial_state),
        "success_distance": success_distance,
        "lost_distance": lost_distance if lost_distance > 0.0 else None,
        "require_success_distance": bool(args.require_success_distance),
        "settle_steps": int(settled_steps),
        "flushed_initial_observation": bool(flushed_initial_observation),
        "recorded_target_source": trajectory["source"],
        "recorded_target_episode": trajectory["episode_name"],
    }
    return {"episode_id": str(episode_id), "stat": stat, "infos": infos, "frames": frames}


def write_result(args: argparse.Namespace, result: dict[str, Any]) -> None:
    scene_dir = Path(args.save_path) / f"seed_{args.seed}" / common.safe_slug(args.env_id)
    scene_dir.mkdir(parents=True, exist_ok=True)
    episode_id = result["episode_id"]
    common.write_json(scene_dir / f"{episode_id}.json", result["stat"])
    common.write_json(scene_dir / f"{episode_id}_info.json", result["infos"])
    if args.save_video:
        common.save_mp4(result["frames"], scene_dir / f"{episode_id}.mp4", args.fps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate single-agent OpenTrackVLA in UnrealZoo.")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--agent", choices=["robotdog", "drone"], default="robotdog")
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--save-path", default="/data/hdt/ntv_data/sim_data/eval/unrealzoo_single_agent")
    parser.add_argument("--env-id", required=True)
    parser.add_argument("--episodes", type=int, default=0, help="0 evaluates every test trajectory for this scene.")
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--render-gpu", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--instruction", default=common.DEFAULT_INSTRUCTION)
    parser.add_argument("--llm-name", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--vision-feat-dim", type=int, default=1536)
    parser.add_argument("--n-waypoints", type=int, default=10)
    parser.add_argument("--history", type=int, default=31)
    parser.add_argument(
        "--history-frame-dt",
        type=float,
        default=0.1,
        help="Training-frame interval used to resample sparse realtime history tokens.",
    )
    parser.add_argument(
        "--history-sampling-mode",
        choices=("time_grid", "inference_steps"),
        default="time_grid",
        help="time_grid aligns history to training FPS; inference_steps reproduces legacy evaluation.",
    )
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--alpha-xy", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--offscreen", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--time-dilation", type=int, default=-1)
    parser.add_argument("--disable-ue-input", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--launch-retries", type=int, default=5)
    parser.add_argument("--ue-sleep-time", type=float, default=20.0)
    parser.add_argument("--settle-steps", type=int, default=1)
    parser.add_argument("--flush-initial-observation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trajectory-overlay", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trajectory-scale", type=float, default=120.0)
    parser.add_argument("--waypoint-index", type=int, default=1) # 使用第几个航点来计算动作
    parser.add_argument("--waypoint-horizon-steps", type=int, default=9)
    parser.add_argument(
        "--waypoint-command-dt",
        type=float,
        default=0.0,
        help="Legacy fixed divisor for the selected waypoint; <=0 uses index*dt.",
    )
    parser.add_argument(
        "--realtime-waypoint-timing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Keep UE running and select the waypoint nearest to this step's "
            "measured observation-to-action inference latency."
        ),
    )
    parser.add_argument("--realtime-waypoint-min-seconds", type=float, default=0.1)
    parser.add_argument("--realtime-waypoint-max-seconds", type=float, default=0.9)
    parser.add_argument("--drone-vx-scale", type=float, default=0.15)
    parser.add_argument("--drone-vy-scale", type=float, default=0.1)
    parser.add_argument("--drone-yaw-sign", type=float, default=1.0)
    parser.add_argument("--drone-yaw-scale", type=float, default=1.0)
    parser.add_argument("--drone-max-vx", type=float, default=0.15)
    parser.add_argument("--drone-max-vy", type=float, default=0.05)
    parser.add_argument("--drone-max-yaw-rate", type=float, default=0.0)
    parser.add_argument("--robotdog-yaw-sign", type=float, default=1.0)
    parser.add_argument("--robotdog-yaw-scale", type=float, default=1.0)
    parser.add_argument("--robotdog-speed-gain", type=float, default=1.15) # 狗的速度增益
    parser.add_argument("--robotdog-max-speed", type=float, default=1.05)
    parser.add_argument("--robotdog-max-turn-deg", type=float, default=30.0)
    parser.add_argument("--drone-success-distance", type=float, default=4.0)
    parser.add_argument("--robotdog-success-distance", type=float, default=8.0)
    parser.add_argument(
        "--robotdog-lost-distance",
        type=float,
        default=0.0,
        help="Distance in meters beyond which lost_count increases; <=0 keeps legacy non-following counting.",
    )
    parser.add_argument(
        "--require-success-distance",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Require distance <= robotdog_success_distance to count as following. "
            "Default False matches the UnrealZoo robotdog collection metric, "
            "which counted following by target visibility."
        ),
    )
    parser.add_argument("--max-lost-steps", type=int, default=30)
    parser.add_argument("--max-episode-seconds", type=float, default=600.0)
    parser.add_argument("--success-rate-threshold", type=float, default=0.5)
    parser.add_argument("--min-success-steps", type=int, default=20)
    parser.add_argument(
        "--require-final-follow-range-on-target-stop",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When the target reaches its goal early, require final distance to lie within "
            "the configured [min_follow_dist, max_follow_dist] interval."
        ),
    )
    parser.add_argument(
        "--face-target-before-step",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Optionally rotate the selected agent toward the target before each model step, matching snap-heading collection.",
    )
    parser.add_argument(
        "--target-replay-mode",
        choices=["nav_goal", "pose", "path_goal"],
        default="nav_goal",
        help=(
            "nav_goal: use the test trajectory start/goal but let UnrealZoo animate the human; "
            "pose: set recorded target_pose every frame, which preserves XY path but makes the character slide; "
            "path_goal: follow recorded test trajectory as sequential NavMesh goals for real walking animation."
        ),
    )
    parser.add_argument(
        "--target-path-min-spacing",
        type=float,
        default=100.0,
        help="Minimum XY spacing in Unreal units when subsampling recorded target poses into NavMesh goals.",
    )
    parser.add_argument(
        "--target-path-reach-distance",
        type=float,
        default=120.0,
        help="Advance to the next recorded path goal when the human is within this XY distance in Unreal units.",
    )
    parser.add_argument(
        "--init-from-recorded-agent-pose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Initialize the selected agent pose and camera from the first frame of the test episode.",
    )
    parser.add_argument("--human-speed", type=float, default=90.0)# 设置人的速度
    parser.add_argument(
        "--target-goal-reach-distance",
        type=float,
        default=50.0,
        help="Final target-goal threshold in Unreal units; 50 equals Habitat's 0.5 m.",
    )
    parser.add_argument("--target-stop-wait-min-steps", type=int, default=5)
    parser.add_argument("--target-stop-wait-max-steps", type=int, default=15)
    parser.add_argument("--human-goal-min-distance", type=float, default=700.0)
    parser.add_argument("--human-goal-max-distance", type=float, default=2200.0)
    parser.add_argument("--open-spawn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--open-spawn-radius", type=float, default=900.0)
    parser.add_argument("--min-open-clearance", type=float, default=300.0)
    parser.add_argument("--open-spawn-candidates", type=int, default=128)
    parser.add_argument("--ground-navmesh-tolerance", type=float, default=300.0)
    parser.add_argument("--drone-navmesh-tolerance", type=float, default=600.0)
    parser.add_argument("--require-visual-target", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require-centered-target", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-mask-visibility", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-visible-ratio", type=float, default=0.001)
    parser.add_argument("--target-center-tolerance", type=float, default=0.35)
    parser.add_argument("--human-appearance-min", type=int, default=1)
    parser.add_argument("--human-appearance-max", type=int, default=18)
    parser.add_argument("--robotdog-appearance-min", type=int, default=20)
    parser.add_argument("--robotdog-appearance-max", type=int, default=33)
    parser.add_argument("--robotdog-ideal-follow-dist", type=float, default=2.2)
    parser.add_argument("--robotdog-min-follow-dist", type=float, default=1.0)
    parser.add_argument("--robotdog-max-follow-dist", type=float, default=4.0)
    parser.add_argument("--robotdog-max-lateral-speed", type=float, default=0.45)
    parser.add_argument("--robotdog-max-yaw-rate", type=float, default=1.0)
    parser.add_argument("--robotdog-camera-forward", type=float, default=140.0)
    parser.add_argument("--robotdog-camera-lateral", type=float, default=0.0)
    parser.add_argument("--robotdog-camera-height", type=float, default=110.0)
    parser.add_argument("--robotdog-camera-mounts", default="140:0:110,170:0:120,110:0:95,0:120:110")
    parser.add_argument("--robotdog-camera-fixed-pitch", type=float, default=None)
    parser.add_argument("--robotdog-camera-pitches", default="-15,-8,0,8,15,22,-22")
    parser.add_argument("--robotdog-camera-yaw-offsets", default="0,-8,8,-15,15")
    parser.add_argument("--robotdog-camera-mode", choices=["fixed", "oracle"], default="fixed")
    parser.add_argument("--robotdog-fov", type=float, default=95.0)
    parser.add_argument("--max-self-visible-ratio", type=float, default=0.015)
    parser.add_argument("--drone-ideal-follow-dist", type=float, default=2.8)
    parser.add_argument("--drone-min-follow-dist", type=float, default=1.5)
    parser.add_argument("--drone-max-follow-dist", type=float, default=4.0)
    parser.add_argument("--drone-height", type=float, default=1000.0)
    parser.add_argument("--drone-max-speed", type=float, default=0.15)
    parser.add_argument("--drone-camera-fixed-pitch", type=float, default=-60.0)
    parser.add_argument("--drone-camera-pitches", default="-60")
    parser.add_argument("--drone-camera-fixed-yaw", type=float, default=0.0)
    parser.add_argument("--drone-camera-yaw-offsets", default="0")
    parser.add_argument("--drone-camera-mode", choices=["fixed", "oracle"], default="fixed")
    parser.add_argument("--lock-drone-camera-world-xy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--drone-camera-z-offset", type=float, default=0.0)
    parser.add_argument("--drone-fov", type=float, default=100.0)
    parser.add_argument("--max-camera-search-candidates", type=int, default=12)
    parser.add_argument("--snap-heading", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--follow-behind", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--top-view-height", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.realtime_waypoint_min_seconds <= 0.0:
        raise ValueError("--realtime-waypoint-min-seconds must be positive")
    if args.realtime_waypoint_max_seconds < args.realtime_waypoint_min_seconds:
        raise ValueError(
            "--realtime-waypoint-max-seconds must be >= --realtime-waypoint-min-seconds"
        )
    trajectories = load_test_trajectories(args.test_manifest, args.env_id, args.agent)
    if args.episodes > 0:
        trajectories = trajectories[: args.episodes]
    Path(args.save_path).mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    planner = UnrealZooSingleAgentPlanner(args)
    env = common.make_env(args)
    rng = random.Random(args.seed)
    try:
        for episode_id, trajectory in enumerate(trajectories):
            print(f"[episode {episode_id}] replay={trajectory['episode_name']}", flush=True)
            try:
                result = run_episode(env, args, planner, episode_id, rng, trajectory)
            except common.EpisodeSkipped as exc:
                print(f"[episode {episode_id}] skipped: {exc}", flush=True)
                continue
            write_result(args, result)
            rate_key = f"{args.agent}_following_rate"
            print(
                f"[episode {episode_id}] status={result['stat']['status']} "
                f"TR={result['stat'][rate_key]:.3f}",
                flush=True,
            )
    finally:
        env.close()
    print(f"[done] episodes={len(trajectories)} save={args.save_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
