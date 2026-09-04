#!/usr/bin/env python3
"""Rerender one data_arr AT shard with recorded target/follower geometry."""

from __future__ import annotations

import argparse, copy, json, math, os, random, signal, sys, time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("PYNPUT_BACKEND", "dummy")
# gym_unrealcv normally registers the complete catalog of several thousand
# environments at import time.  Replay uses one Track environment per worker;
# enabling its fast-registration path avoids minutes of CPU-bound startup when
# multiple workers are launched concurrently.
os.environ.setdefault(
    "UNREALZOO_FAST_ENV_ID",
    "UnrealTrack-KoreanPalace-ContinuousColor-v0",
)

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT, ROOT/"unrealzoo-gym", ROOT/"unrealzoo-gym/example/DataRecording"):
    sys.path.insert(0, str(value))

import gym_unrealcv as _gym_unrealcv  # noqa: E402

from generate_aerial_ground_human_tracking_small import (  # noqa: E402
    capture_color_mask_snapshot, make_env, set_drone_camera,
)
from generate_drone_human_tracking_small import (  # noqa: E402
    data_collection_reset, ensure_bgr_uint8, pick_open_start,
    pick_reachable_goal, safe_start_points, yaw_deg,
)
from eval_unrealzoo_multi_agent import recorded_target_action_for_step  # noqa: E402
from generate_robotdog_human_tracking_small import (  # noqa: E402
    object_mask_ratio_and_bbox, set_robotdog_camera,
)
from tools.legacy_replay_kinematics import (  # noqa: E402
    LEGACY_REPLAY_CONFIG,
    capture_color_mask_snapshot_stable,
    fixed_step as legacy_fixed_step,
    ground_control as legacy_ground_control,
    drone_control as legacy_drone_control,
    body_xy_velocity as legacy_body_xy_velocity,
)


def atomic_json(path: Path, value) -> None:
    tmp=path.with_suffix(path.suffix+f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); tmp.replace(path)


def restore_pose(env, name: str, pose) -> None:
    if not isinstance(pose,list) or len(pose)<5: raise ValueError(f"invalid pose {name}: {pose}")
    env.unwrapped.unrealcv.set_obj_location(name,[float(v) for v in pose[:3]])
    env.unwrapped.unrealcv.set_obj_rotation(name,[0.0,float(pose[4]),0.0])


def args_for_env(item, args):
    return SimpleNamespace(
        env_id=item["scene"], width=args.width, height=args.height, dt=0.1,
        ue_interval_ms=100, offscreen=True, render_gpu=args.render_gpu,
        disable_ue_input=True, require_visual_target=False, time_dilation=-1,
        seed=int(item["distractor_seed"]), launch_retries=5,
    )


def setup_population(env, item, env_args):
    d=int(item["distractor_count"])
    # The legacy collector clips ``distractors`` to 3. First establish the
    # canonical target/dog/drone slots, then expand through BaseEnv directly.
    env.unwrapped.agents_category=["player","player","drone"]
    reset_args=SimpleNamespace(distractors=1,launch_retries=env_args.launch_retries)
    data_collection_reset(env,reset_args)
    env.unwrapped.agents_category=["player","player","drone"]+["player"]*d
    env.unwrapped.set_population(3+d)
    players=env.unwrapped.player_list
    if len(players)!=3+d: raise RuntimeError(f"population mismatch {len(players)} != {3+d}")
    target,dog,drone=0,1,2; distractors=list(range(3,3+d))
    appearance_ids = list(item.get("human_appearance_ids") or [int(item["appearance_id"])] * (1 + d))
    if len(appearance_ids) != 1 + d:
        raise ValueError(f"appearance_ids length mismatch: {len(appearance_ids)} != {1+d}")
    for actor_index, app in zip([target, *distractors], appearance_ids):
        env.unwrapped.unrealcv.set_appearance(players[actor_index], int(app))
    env.unwrapped.unrealcv.set_appearance(players[dog],27)
    return players,target,dog,drone,distractors,{
        players[idx]: int(app)
        for idx, app in zip([target, *distractors], appearance_ids)
    }


def _nearby_safe_point(env, rng, reference, min_distance=250.0, max_distance=1000.0):
    candidates=[]
    for value in safe_start_points(env):
        point=[float(v) for v in value[:3]]
        distance=float(np.linalg.norm(np.asarray(point[:2])-np.asarray(reference[:2])))
        if min_distance <= distance <= max_distance:
            candidates.append((distance,point))
    if candidates:
        candidates.sort(key=lambda value:value[0])
        return rng.choice(candidates[:min(12,len(candidates))])[1]
    return pick_open_start(env,rng,SimpleNamespace(open_spawn_radius=1400.0,open_spawn_candidates=128,min_open_clearance=100.0),avoid_pos=None)


def _formation_offset_m(formation, frame_index: int) -> tuple[float, float]:
    """Return a smooth deterministic forward/right target-local offset."""
    time_s=float(frame_index)*0.1
    phase=float(formation.get("jitter_phase_rad",0.0))
    period=max(float(formation.get("jitter_period_s",9.0)),1.0)
    wave=2.0*math.pi*time_s/period+phase
    forward=(
        float(formation.get("forward_m",3.0))
        +float(formation.get("jitter_forward_m",0.0))
        +float(formation.get("jitter_forward_amplitude_m",0.0))*math.sin(wave)
        +float(formation.get("jitter_forward_secondary_m",0.0))*math.sin(1.71*wave+1.3)
    )
    right=(
        float(formation.get("right_m",0.0))
        +float(formation.get("jitter_right_m",0.0))
        +float(formation.get("jitter_right_amplitude_m",0.0))*math.sin(0.73*wave+0.9)
        +float(formation.get("jitter_right_secondary_m",0.0))*math.sin(1.37*wave+2.1)
    )
    return forward,right


def setup_distractors(env, item, players, distractors, target_pose, target_goal):
    rng=random.Random(int(item["distractor_seed"])); configs=[]
    formation_list=item.get("distractor_formation") or []
    for ordinal,idx in enumerate(distractors):
        name=players[idx]
        formation=formation_list[ordinal] if ordinal < len(formation_list) else {"forward_m":3.0,"right_m":0.0}
        forward,right=_formation_offset_m(formation,0)
        yaw_rad=math.radians(float(target_pose[4]))
        c,s=math.cos(yaw_rad),math.sin(yaw_rad)
        start=[
            float(target_pose[0])+100.0*(c*forward-s*right),
            float(target_pose[1])+100.0*(s*forward+c*right),
            float(target_pose[2]),
        ]
        # Validate that the formation point is connected to the target's
        # walkable NavMesh region.  Unlike sampling the tiny safe_start list,
        # this keeps every actor close while avoiding duplicate spawn points.
        try:
            spawn_path=env.unwrapped.unrealcv.find_path(players[0],start)
        except Exception:
            spawn_path=[]
        if not spawn_path or len(spawn_path)<2:
            start=_nearby_safe_point(env,rng,target_pose,200.0,700.0)
        env.unwrapped.unrealcv.set_obj_location(name,start)
        yaw=float(target_pose[4]); env.unwrapped.unrealcv.set_obj_rotation(name,[0,yaw,0])
        goal=_nearby_safe_point(env,rng,target_goal,200.0,800.0)
        try:
            path=env.unwrapped.unrealcv.find_path(name,goal)
        except Exception:
            path=[]
        if not path or len(path)<2:
            goal,path=pick_reachable_goal(env,name,rng,avoid_pos=start,min_distance=200,max_distance=900,max_trials=16)
        # The target averages slightly above 0.9 m/s in some DT episodes.
        # Allow a modest 1.35 m/s catch-up ceiling; once in formation the
        # closed-loop command naturally settles near the target's speed.
        env.unwrapped.unrealcv.set_max_speed(name,135.0)
        env.unwrapped.unrealcv.set_acceleration(name,300.0)
        configs.append({
            "actor":name,"start":start,"yaw_deg":yaw,"goal":goal,"path":path,
            "max_speed_mps":1.35,"acceleration_uu_s2":300.0,
            "orbit_radius_m":3.0 + 0.35 * ((idx - 3) % 7),
            "orbit_phase_rad":(2.0 * math.pi * (idx - 3) / max(len(distractors), 1)),
            "motion_controller":"target_relative_body_action_v1",
            "spawn_path_valid": bool(spawn_path and len(spawn_path)>=2),
            "formation": formation,
        })
    return configs


def _wrap_degrees(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def distractor_action_commands(
    env, players, distractors, configs, target_pose, frame_index,
    source_target_before, source_target_after, source_target_next,
):
    """Drive humans in a target-relative formation using legacy inverse control."""
    target_xy=np.asarray(target_pose[:2],dtype=np.float64)
    target_yaw=float(target_pose[4])
    current_source=np.asarray(source_target_before[:2],dtype=np.float64)
    source_after=np.asarray(source_target_after[:2],dtype=np.float64)
    source_next=np.asarray(source_target_next[:2],dtype=np.float64)
    delta_after=source_after-current_source
    delta_next=source_next-source_after
    c,s=math.cos(math.radians(target_yaw)),math.sin(math.radians(target_yaw))
    actions=[]; debug=[]
    for ordinal,(idx, config) in enumerate(zip(distractors, configs)):
        pose=np.asarray(env.unwrapped.obj_poses[idx],dtype=np.float64)
        formation=(config.get("formation") or {})
        forward,right=_formation_offset_m(formation,frame_index)
        offset=np.asarray([c*forward-s*right,s*forward+c*right])*100.0
        desired=target_xy+offset+delta_after
        next_offset=offset
        next_xy=target_xy+offset+delta_after+delta_next
        desired_yaw=math.degrees(math.atan2(float(desired[1]-pose[1]),float(desired[0]-pose[0])))
        next_yaw=math.degrees(math.atan2(float(next_xy[1]-desired[1]),float(next_xy[0]-desired[0])))
        reference_before=pose.tolist()
        reference_after=[float(desired[0]),float(desired[1]),float(pose[2]),0.0,float(desired_yaw),0.0]
        reference_next=[float(next_xy[0]),float(next_xy[1]),float(pose[2]),0.0,float(next_yaw),0.0]
        action,info=legacy_ground_control(
            pose.tolist(),reference_before,reference_after,reference_next,0.1,
            LEGACY_REPLAY_CONFIG["ground_turn_step_gain"],1.35,
            LEGACY_REPLAY_CONFIG["ground_max_turn_deg"],
            LEGACY_REPLAY_CONFIG["ground_translation_delay_steps"],
            LEGACY_REPLAY_CONFIG["ground_position_feedback_time_s"],
            "vector_2d_stable",0.12,20.0,reference_next,
            False,110.0,70.0,None,3.0,0.25,1.5,
            1.5,
            LEGACY_REPLAY_CONFIG["ground_speed_model"],None,0.0,
        )
        # ``vector_2d_stable`` encodes a point behind the actor as near-zero
        # turn plus negative speed.  Clamping only the speed would leave the
        # pedestrian stuck.  Turn toward the raw vector first, then walk only
        # forward once sufficiently aligned.
        if float(action[1]) < 0.0:
            raw_heading=float(info.get("vector_yaw_delta_deg") or 0.0)
            action[0]=float(np.clip(
                raw_heading/max(float(LEGACY_REPLAY_CONFIG["ground_turn_step_gain"]),1e-9),
                -float(LEGACY_REPLAY_CONFIG["ground_max_turn_deg"]),
                float(LEGACY_REPLAY_CONFIG["ground_max_turn_deg"]),
            ))
            action[1]=0.0 if abs(raw_heading)>55.0 else abs(float(action[1]))
            info["reverse_suppressed"] = True
            info["forward_only_raw_heading_deg"] = raw_heading
        else:
            info["reverse_suppressed"] = False
        actions.append(action)
        debug.append({"actor":players[idx],"desired_xy_uu":desired.tolist(),"reference_after":reference_after,"reference_next":reference_next,"formation_offset_m":[forward,right],"action":action,"legacy_ground":info})
    return actions,debug


def _ground_walk_action(current_pose, next_pose, dt=0.1, speed_uu_s=90.0):
    """Convert two world poses to the same [turn_deg, forward_cm_s] contract."""
    current=np.asarray(current_pose,dtype=np.float64); target=np.asarray(next_pose,dtype=np.float64)
    dx,dy=float(target[0]-current[0]),float(target[1]-current[1])
    yaw=math.radians(float(current[4])); c,s=math.cos(yaw),math.sin(yaw)
    forward=(c*dx+s*dy)/max(dt,1e-6)
    desired=math.degrees(math.atan2(dy,dx)) if abs(dx)+abs(dy)>1e-6 else float(current[4])
    yaw_error=_wrap_degrees(desired-float(current[4]))
    turn=float(np.clip(yaw_error/0.4,-45.0,45.0))
    # Preserve recorded forward speed when moving; stop only for a genuinely
    # stationary target frame. Lateral displacement is intentionally ignored
    # by the non-holonomic actor, as in the original DT controller.
    speed=float(np.clip(forward,-speed_uu_s,speed_uu_s)) if abs(dx)+abs(dy)>1e-3 else 0.0
    return [turn,speed],{"turn_deg":turn,"speed_uu_s":speed,"forward_from_pose_mps":forward/100.0}


def _legacy_ground_args():
    return SimpleNamespace(
        ground_vector_min_speed_mps=0.12,
        ground_vector_max_heading_delta_deg=20.0,
        ground_reverse_enter_deg=110.0,
        ground_reverse_exit_deg=70.0,
        ground_heading_goal_rate_deg=3.0,
        ground_turn_slowdown_floor=0.25,
        ground_heading_deadband_deg=1.5,
    )


def writer(path: Path,w:int,h:int,fps:int):
    path.parent.mkdir(parents=True,exist_ok=True)
    obj=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"mp4v"),fps,(w,h))
    if not obj.isOpened(): raise RuntimeError(f"video writer failed {path}")
    return obj


def run_one(item,args):
    source=Path(args.source_root)/item["raw_relative_dir"]; stem=item["stem"]
    out=Path(args.output_root)/item["scene"]/item["source_batch"]; marker=out/f"{stem}.complete.json"
    if args.resume and marker.is_file(): return {"skipped":True,"frames":0,"seconds":0}
    # Replace the outer episode alarm with a short startup deadline. It is
    # promoted to the full episode deadline only after the first physical UE
    # fixed-step succeeds.
    signal.setitimer(signal.ITIMER_REAL,args.startup_timeout_s)
    drone_rows=json.load(open(source/f"{stem}_drone_info.json")); dog_rows=json.load(open(source/f"{stem}_robotdog_info.json"))
    count=min(len(drone_rows),len(dog_rows));
    if args.max_frames > 0: count=min(count,args.max_frames)
    # Fast registration mode intentionally registers only one scene at import
    # time.  Evaluation spans many UnrealZoo maps, so register each scene just
    # before constructing its environment; duplicate registration is harmless.
    try:
        _gym_unrealcv._register_fast_env(item["scene"])
    except Exception:
        pass
    env_args=args_for_env(item,args); env=make_env(env_args)
    start_time=time.monotonic(); temp_d=out/f".{stem}_drone.tmp.mp4"; temp_g=out/f".{stem}_robotdog.tmp.mp4"
    out.mkdir(parents=True,exist_ok=True); wd=wg=None
    try:
        players,target,dog,drone,distractors,appearance_map=setup_population(env,item,env_args)
        target_name,dog_name,drone_name=players[target],players[dog],players[drone]
        restore_pose(env,target_name,drone_rows[0]["target_pose"])
        configs=setup_distractors(env,item,players,distractors,drone_rows[0]["target_pose"],drone_rows[-1].get("target_pose") or drone_rows[-1]["target_pose"])
        u=env.unwrapped.unrealcv; u.set_global_time_dilation(1.0);u.set_max_FPS(10.0);u.set_pause()
        # Match the verified recorded-action evaluator: BP ceilings are
        # deliberately unrestrictive and the per-step action controls actual
        # motion. A low CharacterMovement ceiling otherwise turns 90 cm/s
        # target commands into the observed ~0.3 m/s crawl.
        u.set_max_speed(target_name, 10000.0)
        u.set_acceleration(target_name, 10000.0)
        u.set_max_speed(dog_name, 10000.0)
        u.set_acceleration(dog_name, 10000.0)
        for idx in distractors:
            u.set_max_speed(players[idx], 10000.0)
            u.set_acceleration(players[idx], 10000.0)
        wd=writer(temp_d,args.width,args.height,10);wg=writer(temp_g,args.width,args.height,10)
        new_d=[];new_g=[];distractor_frames=[];distractor_command_frames=[]
        target_frames=[];dog_frames=[];drone_frames=[]
        progress_path=out/f"{stem}.progress.json"
        previous_actual={"dog":None,"drone":None}
        for i in range(count):
            dr=copy.deepcopy(drone_rows[i]); gr=copy.deepcopy(dog_rows[i])
            # Only frame zero is restored. Subsequent target motion is produced
            # by an explicit action and never by teleporting to target_pose[i].
            if i == 0:
                restore_pose(env,target_name,dr.get("target_pose") or gr.get("target_pose"))
                restore_pose(env,dog_name,gr["robotdog_pose"]);restore_pose(env,drone_name,dr["drone_pose"])
            actual_drone_for_camera=(
                list(dr["drone_pose"]) if i==0 else list(env.unwrapped.obj_poses[drone])
            )
            mount=[float(v) for v in gr.get("robotdog_camera_mount",[170,0,120])[:3]]
            set_robotdog_camera(env,dog_name,dog,mount,float(gr.get("robotdog_camera_pitch",-8)),float(gr.get("robotdog_camera_yaw_offset",0)),SimpleNamespace(robotdog_fov=95))
            set_drone_camera(env,drone_name,drone,actual_drone_for_camera,float(dr.get("drone_camera_pitch",-40)),float(dr.get("drone_camera_yaw_offset",0)),SimpleNamespace(lock_drone_camera_world_xy=True,drone_camera_forward_offset=35,drone_camera_z_offset=-60,drone_fov=100))
            obs,masks=capture_color_mask_snapshot_stable(
                env, include_masks=True, attempts=args.snapshot_attempts,
                mode=args.snapshot_mode, render_sync_s=args.snapshot_render_sync,
            )
            wd.write(ensure_bgr_uint8(obs[drone]));wg.write(ensure_bgr_uint8(obs[dog]))
            for row,agent in ((dr,drone),(gr,dog)):
                ratio,visible,bbox=object_mask_ratio_and_bbox(env,env.unwrapped.cam_list[agent],target_name,masks[agent])
                row.update(target_bbox=bbox,target_visible=bool(visible),target_visibility=float(ratio),instruction=item["instruction"],task_variant=item["task_variant"],appearance_id=item["appearance_id"],distractor_count=len(distractors),distractor_seed=item["distractor_seed"])
            new_d.append(dr);new_g.append(gr)
            distractor_frames.append({
                players[idx]: [float(v) for v in env.unwrapped.obj_poses[idx]]
                for idx in distractors
            })
            actual_target_pose=list(env.unwrapped.obj_poses[target])
            actual_dog=list(env.unwrapped.obj_poses[dog])
            actual_drone=list(env.unwrapped.obj_poses[drone])
            target_frames.append(actual_target_pose)
            dog_frames.append(actual_dog)
            drone_frames.append(actual_drone)
            dr["replay_target_pose_actual"] = actual_target_pose
            gr["replay_target_pose_actual"] = actual_target_pose
            dr["replay_drone_pose_actual"] = actual_drone
            gr["replay_robotdog_pose_actual"] = actual_dog
            distractor_commands, distractor_action_debug = distractor_action_commands(
                env, players, distractors, configs,
                actual_target_pose, i,
                drone_rows[i].get("target_pose"),
                drone_rows[min(i + 1, count - 1)].get("target_pose"),
                drone_rows[min(i + 2, count - 1)].get("target_pose"),
            )
            distractor_command_frames.append(distractor_action_debug)
            target_action_debug={"source":"legacy_ground_control"}
            target_action=[0.0,0.0]
            if i + 1 < count:
                ref_before=drone_rows[i].get("target_pose")
                ref_after=drone_rows[i+1].get("target_pose")
                ref_next=drone_rows[min(i+2,count-1)].get("target_pose")
                target_action,target_action_debug=legacy_ground_control(
                    actual_target_pose, ref_before, ref_after, ref_next, 0.1,
                    LEGACY_REPLAY_CONFIG["ground_turn_step_gain"], 100.0,
                    LEGACY_REPLAY_CONFIG["ground_max_turn_deg"],
                    LEGACY_REPLAY_CONFIG["ground_translation_delay_steps"],
                    LEGACY_REPLAY_CONFIG["ground_position_feedback_time_s"],
                    LEGACY_REPLAY_CONFIG["ground_control_mode"],
                    0.12,20.0,ref_next,False,110.0,70.0,None,3.0,0.25,1.5,
                    LEGACY_REPLAY_CONFIG["ground_max_forward_feedback_mps"],
                    LEGACY_REPLAY_CONFIG["ground_speed_model"],None,0.0,
                )
            dog_forward_velocity=(
                float(legacy_body_xy_velocity(previous_actual["dog"],actual_dog,0.1)[0])
                if previous_actual["dog"] is not None else 0.0
            )
            dog_ref_before=dog_rows[i].get("robotdog_pose")
            dog_ref_after=dog_rows[min(i+1,count-1)].get("robotdog_pose")
            dog_ref_next=dog_rows[min(i+2,count-1)].get("robotdog_pose")
            dog_action,dog_action_debug=legacy_ground_control(
                actual_dog,dog_ref_before,dog_ref_after,dog_ref_next,0.1,
                LEGACY_REPLAY_CONFIG["ground_turn_step_gain"],100.0,
                LEGACY_REPLAY_CONFIG["ground_max_turn_deg"],1,1.0,"source_yaw",
                0.12,20.0,dog_ref_next,False,110.0,70.0,None,3.0,0.25,1.5,
                float("inf"),"legacy_preview",None,dog_forward_velocity,
            )
            drone_args=SimpleNamespace(
                drone_a_forward=0.969,drone_b_forward=0.0301,
                drone_a_lateral=0.969,drone_b_lateral=0.0301,
                drone_yaw_a=0.464,drone_yaw_b=0.359,
                drone_max_command_mps=100.0,drone_max_yaw_command_radps=8.0,
            )
            drone_ref=drone_rows[min(i+1,count-1)].get("drone_pose")
            drone_action,drone_action_debug=legacy_drone_control(
                actual_drone,previous_actual["drone"],drone_ref,0.1,drone_args,
            )
            # Use the validated legacy fixed_step command/clock path. It sends
            # all actor actions, resumes exactly one fixed tick, pauses again,
            # and refreshes poses with the old replay semantics.
            actions=[None for _ in env.unwrapped.player_list]
            actions[target]=target_action or [0.0,0.0]
            actions[dog]=dog_action
            actions[drone]=drone_action
            for idx, action in zip(distractors, distractor_commands):
                actions[idx]=action
            legacy_fixed_step(env, actions)
            if i==0:
                signal.setitimer(signal.ITIMER_REAL,args.episode_timeout_s)
            previous_actual={"dog":actual_dog,"drone":actual_drone}
            # Written only after a real fixed-step action pulse and pose
            # refresh. A UE process with no progress file has not replayed a
            # single physical step yet.
            atomic_json(progress_path,{
                "status":"running","episode_id":item["episode_id"],
                "frames_completed":i+1,"expected_frames":count,
                "last_target_pose":actual_target_pose,
                "last_robotdog_pose":actual_dog,"last_drone_pose":actual_drone,
                "updated_unix_s":time.time(),
            })
            if distractor_action_debug:
                # Store the command that produced the next fixed-step pose.
                if i == 0:
                    configs[0]["action_debug_first_frame"] = distractor_action_debug[0]
            if i == 0:
                configs[0]["target_action_debug_first_frame"] = target_action_debug
                configs[0]["dog_action_debug_first_frame"] = dog_action_debug
                configs[0]["drone_action_debug_first_frame"] = drone_action_debug
        wd.release();wg.release();wd=wg=None
        (out/f"{stem}_drone.mp4").unlink(missing_ok=True);temp_d.replace(out/f"{stem}_drone.mp4")
        (out/f"{stem}_robotdog.mp4").unlink(missing_ok=True);temp_g.replace(out/f"{stem}_robotdog.mp4")
        atomic_json(out/f"{stem}_drone_info.json",new_d);atomic_json(out/f"{stem}_robotdog_info.json",new_g)
        motion=[]
        for index, frame in enumerate(distractor_frames):
            target_xy=np.asarray(target_frames[index][:2],dtype=np.float64)
            for actor,pose in frame.items():
                previous=None if index==0 else distractor_frames[index-1].get(actor)
                speed=0.0 if previous is None else float(np.linalg.norm(np.asarray(pose[:2])-np.asarray(previous[:2]))/0.1/100.0)
                distance=float(np.linalg.norm(np.asarray(pose[:2])-target_xy)/100.0)
                motion.append({"frame":index,"actor":actor,"speed_mps":speed,"distance_to_target_m":distance})
        atomic_json(out/f"{stem}_at_episode.json",{
            **item,
            "appearance_mode": "distinct_deterministic",
            "appearance_map": appearance_map,
            "distractors": configs,
            "distractor_poses_per_frame": distractor_frames,
            "distractor_actions_per_frame": distractor_command_frames,
            "distractor_motion_per_frame": motion,
            "target_poses_per_frame": target_frames,
            "robotdog_poses_per_frame": dog_frames,
            "drone_poses_per_frame": drone_frames,
            "target_motion_source": "explicit_recorded_pose_action_no_teleport_after_frame0",
            "pose_frame_alignment": "observation_t_before_fixed_step_t",
        })
        elapsed=time.monotonic()-start_time;atomic_json(marker,{"status":"complete","frames":count,"seconds":elapsed,"steps_per_second":count/elapsed})
        progress_path.unlink(missing_ok=True)
        return {"skipped":False,"frames":count,"seconds":elapsed}
    finally:
        if wd:wd.release()
        if wg:wg.release()
        temp_d.unlink(missing_ok=True);temp_g.unlink(missing_ok=True)
        try:env.close()
        except Exception:pass


def main():
    p=argparse.ArgumentParser();p.add_argument("--plan",type=Path,required=True);p.add_argument("--source-root",type=Path,default=Path("/data/hdt/ntv_data/sim_data/data_arr"));p.add_argument("--output-root",type=Path,default=Path("/data/hdt/ntv_data/data_arr_at_v1"));p.add_argument("--render-gpu",type=int,required=True);p.add_argument("--episode-id");p.add_argument("--plan-indices",help="Comma-separated zero-based indices from the original plan");p.add_argument("--exclude-scenes",default="",help="Comma-separated scene names to leave pending");p.add_argument("--shard-index",type=int,default=0);p.add_argument("--shard-count",type=int,default=1);p.add_argument("--max-episodes",type=int,default=0);p.add_argument("--max-frames",type=int,default=0);p.add_argument("--episode-retries",type=int,default=2);p.add_argument("--startup-timeout-s",type=float,default=600.0);p.add_argument("--episode-timeout-s",type=float,default=900.0);p.add_argument("--snapshot-mode",choices=("batch","sequential"),default="batch");p.add_argument("--snapshot-attempts",type=int,default=3);p.add_argument("--snapshot-render-sync",type=float,default=0.02);p.add_argument("--width",type=int,default=640);p.add_argument("--height",type=int,default=480);p.add_argument("--resume",action=argparse.BooleanOptionalAction,default=True);args=p.parse_args()
    if args.episode_retries < 0: raise ValueError("--episode-retries must be non-negative")
    if args.episode_timeout_s <= 0: raise ValueError("--episode-timeout-s must be positive")
    if args.startup_timeout_s <= 0: raise ValueError("--startup-timeout-s must be positive")
    if args.snapshot_attempts < 1 or args.snapshot_render_sync < 0: raise ValueError("invalid snapshot settings")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count: raise ValueError("invalid shard index/count")
    items=json.load(open(args.plan))
    excluded_scenes={value.strip() for value in args.exclude_scenes.split(",") if value.strip()}
    if excluded_scenes:
        items=[item for item in items if item.get("scene") not in excluded_scenes]
    if args.episode_id and args.plan_indices: raise ValueError("--episode-id and --plan-indices are mutually exclusive")
    if args.plan_indices:
        indices=[int(value.strip()) for value in args.plan_indices.split(",") if value.strip()]
        if not indices or len(indices)!=len(set(indices)) or min(indices)<0 or max(indices)>=len(items): raise ValueError("invalid --plan-indices")
        items=[items[index] for index in indices]
    elif args.episode_id:
        items=[item for item in items if item.get("episode_id")==args.episode_id]
        if not items: raise ValueError(f"episode not found in plan: {args.episode_id}")
    elif args.shard_count > 1:
        items=[item for index,item in enumerate(items) if index % args.shard_count == args.shard_index]
    items=items[:args.max_episodes or None];totf=0;tots=0
    for n,item in enumerate(items,1):
        for attempt in range(args.episode_retries+1):
            try:
                print(f"[episode-start] {item['episode_id']} attempt={attempt+1}/{args.episode_retries+1}",flush=True)
                def _timeout(_signum, _frame):
                    raise TimeoutError("replay deadline expired; inspect progress.json to distinguish startup from active stepping")
                previous_handler=signal.signal(signal.SIGALRM,_timeout)
                signal.setitimer(signal.ITIMER_REAL,args.episode_timeout_s)
                try:
                    result=run_one(item,args)
                finally:
                    signal.setitimer(signal.ITIMER_REAL,0)
                    signal.signal(signal.SIGALRM,previous_handler)
                print(f"[episode-end] {item['episode_id']} {result}",flush=True)
                break
            except Exception as exc:
                if isinstance(exc, TimeoutError):
                    timeout_dir=Path(args.output_root)/item["scene"]/item["source_batch"]
                    atomic_json(timeout_dir/f"{item['stem']}.timeout.json", {
                        "status":"timeout",
                        "episode_id":item["episode_id"],
                        "attempt":attempt+1,
                        "timeout_s":args.episode_timeout_s,
                        "message":str(exc),
                    })
                if attempt >= args.episode_retries: raise
                print(f"[episode-retry] {item['episode_id']} attempt={attempt+1}/{args.episode_retries+1} failed: {exc}",flush=True)
                time.sleep(2.0)
        totf+=result["frames"];tots+=result["seconds"];print(f"[{n}/{len(items)}] {item['episode_id']} {result} total_sps={totf/max(tots,1e-6):.2f}",flush=True)
    return 0
if __name__=="__main__":raise SystemExit(main())
