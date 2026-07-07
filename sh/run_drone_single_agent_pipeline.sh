#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/hdt/miniconda3/envs/omtracknew/bin/python}"
SOURCE_ROOT="${SOURCE_ROOT:-/data/hdt/ntv_data/sim_data/New_paths_training}"
RAW_ROOT="${RAW_ROOT:-/data/hdt/ntv_data/sim_data/New_paths_training_drone}"
SPLIT_ROOT="${SPLIT_ROOT:-/data/hdt/ntv_data/sim_data/New_paths_training_drone_split_10to1}"
TRAIN_DATA_ROOT="${TRAIN_DATA_ROOT:-/data/hdt/ntv_data/data/New_paths_training_drone_single_10to1/train}"
CKPT_DIR="${CKPT_DIR:-/data/hdt/ntv_data/ckpt/New_paths_training_drone_single_frozen_qwen_10ep}"
EVAL_ROOT="${EVAL_ROOT:-/data/hdt/ntv_data/sim_data/eval/New_paths_training_drone_single_10to19}"

RUN_ORGANIZE="${RUN_ORGANIZE:-1}"
RUN_SPLIT="${RUN_SPLIT:-1}"
RUN_PROCESS="${RUN_PROCESS:-1}"
RUN_CACHE="${RUN_CACHE:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

TRAIN_GPUS="${TRAIN_GPUS:-6}"
NUM_GPUS="${NUM_GPUS:-1}"
CACHE_GPU="${CACHE_GPU:-6}"
EVAL_GPU="${EVAL_GPU:-6}"
RENDER_GPU="${RENDER_GPU:-6}"
SAVE_EVAL_VIDEO="${SAVE_EVAL_VIDEO:-1}"
EVAL_REQUIRE_SUCCESS_DISTANCE="${EVAL_REQUIRE_SUCCESS_DISTANCE:-1}"
EVAL_DRONE_IDEAL_FOLLOW_DIST="${EVAL_DRONE_IDEAL_FOLLOW_DIST:-4.0}"
# 对齐 /data/cyj aerial-ground 采集默认值：理想跟踪距离，单位 m
EVAL_DRONE_MIN_FOLLOW_DIST="${EVAL_DRONE_MIN_FOLLOW_DIST:-3.0}"
# 对齐采集默认值：最近跟踪距离，单位 m
EVAL_DRONE_MAX_FOLLOW_DIST="${EVAL_DRONE_MAX_FOLLOW_DIST:-5.5}"
# 对齐采集默认值：最远正常跟踪距离，单位 m
EVAL_DRONE_HEIGHT="${EVAL_DRONE_HEIGHT:-400}"
# 新数据实际 drone_height 分布为 500/550/600/650/700，600 是中位数；评估默认仍会优先恢复 test 首帧记录位姿。

EVAL_SETTLE_STEPS="${EVAL_SETTLE_STEPS:-1}"
EVAL_DISABLE_UE_INPUT="${EVAL_DISABLE_UE_INPUT:-1}"
EVAL_LAUNCH_RETRIES="${EVAL_LAUNCH_RETRIES:-5}"
# UE 启动失败后的重试次数
EVAL_UE_SLEEP_TIME="${EVAL_UE_SLEEP_TIME:-20}"
# 启动 UE 后等待 UnrealCV 端口起来的秒数

EVAL_TARGET_REPLAY_MODE="${EVAL_TARGET_REPLAY_MODE:-path_goal}"
# path_goal: 把 test 集 target_pose 抽成 NavMesh 路径点，让人真实行走；pose 是逐帧强制放置，nav_goal 只用首尾点。
EVAL_TARGET_PATH_MIN_SPACING="${EVAL_TARGET_PATH_MIN_SPACING:-100}"
EVAL_TARGET_PATH_REACH_DISTANCE="${EVAL_TARGET_PATH_REACH_DISTANCE:-120}"
EVAL_TARGET_GOAL_REACH_DISTANCE="${EVAL_TARGET_GOAL_REACH_DISTANCE:-50}"
EVAL_TARGET_STOP_WAIT_MIN_STEPS="${EVAL_TARGET_STOP_WAIT_MIN_STEPS:-5}"
EVAL_TARGET_STOP_WAIT_MAX_STEPS="${EVAL_TARGET_STOP_WAIT_MAX_STEPS:-5}"

EVAL_WAYPOINT_INDEX="${EVAL_WAYPOINT_INDEX:-9}" # 约 0.9 秒后的累计 waypoint，用于覆盖约 1 秒推理周期
EVAL_WAYPOINT_HORIZON_STEPS="${EVAL_WAYPOINT_HORIZON_STEPS:-9}"
EVAL_WAYPOINT_COMMAND_DT="${EVAL_WAYPOINT_COMMAND_DT:-0}"
EVAL_WAYPOINT_COMMAND_DT_BY_SCENE="${EVAL_WAYPOINT_COMMAND_DT_BY_SCENE:-}"
EVAL_DT="${EVAL_DT:-0.1}"
EVAL_HISTORY_FRAME_DT="${EVAL_HISTORY_FRAME_DT:-0.1}"
EVAL_HISTORY_SAMPLING_MODE="${EVAL_HISTORY_SAMPLING_MODE:-time_grid}"
EVAL_FPS="${EVAL_FPS:-10}"
EVAL_DETERMINISTIC_STEP="${EVAL_DETERMINISTIC_STEP:-0}"
EVAL_REALTIME_WAYPOINT_TIMING="${EVAL_REALTIME_WAYPOINT_TIMING:-1}"
EVAL_REALTIME_WAYPOINT_MIN_SECONDS="${EVAL_REALTIME_WAYPOINT_MIN_SECONDS:-0.1}"
EVAL_REALTIME_WAYPOINT_MAX_SECONDS="${EVAL_REALTIME_WAYPOINT_MAX_SECONDS:-0.9}"
EVAL_SCENES="${EVAL_SCENES:-}"
EVAL_EPISODES="${EVAL_EPISODES:-manifest}"
EVAL_DRONE_VX_SCALE="${EVAL_DRONE_VX_SCALE:-0.12}" # base_velocity 约 1.0m/s -> 底层 vx 命令约 0.12
EVAL_DRONE_VY_SCALE="${EVAL_DRONE_VY_SCALE:-0.1}"  # base_velocity 约 0.5m/s -> 底层 vy 命令约 0.05
: "${EVAL_DRONE_YAW_SIGN:=1.0}"
EVAL_DRONE_YAW_SCALE="${EVAL_DRONE_YAW_SCALE:-1.0}"
# 对齐当前新训练集：平移映射到底层 step-like 命令，yaw 为真实可学习角速度。
EVAL_DRONE_MAX_YAW_RATE="${EVAL_DRONE_MAX_YAW_RATE:-0.4}"
EVAL_DRONE_SUCCESS_DISTANCE="${EVAL_DRONE_SUCCESS_DISTANCE:-5.5}" # 调：可见且距离不超过该值才算跟上
# 默认对齐采集 max-follow-dist；如果只想放宽指标，可单独改大这个值
EVAL_MAX_LOST_STEPS="${EVAL_MAX_LOST_STEPS:-20}" #调
# 连续多少步没跟上就判丢失
EVAL_MAX_EPISODE_SECONDS="${EVAL_MAX_EPISODE_SECONDS:-600.0}"
EVAL_MIN_SUCCESS_STEPS="${EVAL_MIN_SUCCESS_STEPS:-20}"
EVAL_SUCCESS_RATE_THRESHOLD="${EVAL_SUCCESS_RATE_THRESHOLD:-0.8}"
EVAL_MIN_CENTERED_RATE="${EVAL_MIN_CENTERED_RATE:-0.8}"
EVAL_INIT_FROM_RECORDED_AGENT_POSE="${EVAL_INIT_FROM_RECORDED_AGENT_POSE:-0}"
EVAL_INIT_FOLLOWERS_BEHIND_TARGET="${EVAL_INIT_FOLLOWERS_BEHIND_TARGET:-1}"
EVAL_FACE_TARGET_BEFORE_STEP="${EVAL_FACE_TARGET_BEFORE_STEP:-0}"
# 是否在每步模型输入前让无人机朝向目标；1 更接近采集，0 更严格评估模型 yaw 控制


EVAL_HUMAN_SPEED="${EVAL_HUMAN_SPEED:-0.9}" # m/s; allowed: 0.9, 1.0, 1.1, 1.2
agent_speed_for_human() {
    case "$1" in
        0.9|.9) echo "1.20" ;;
        1.0|1) echo "1.35" ;;
        1.1) echo "1.50" ;;
        1.2) echo "1.60" ;;
        *) echo "Unsupported EVAL_HUMAN_SPEED=$1; choose 0.9, 1.0, 1.1, or 1.2 m/s" >&2; exit 2 ;;
    esac
}
agent_lateral_speed_for_human() {
    case "$1" in
        0.9|.9) echo "0.60" ;;
        1.0|1) echo "0.675" ;;
        1.1) echo "0.75" ;;
        1.2) echo "0.80" ;;
        *) echo "Unsupported EVAL_HUMAN_SPEED=$1; choose 0.9, 1.0, 1.1, or 1.2 m/s" >&2; exit 2 ;;
    esac
}
DEFAULT_AGENT_MAX_SPEED="$(agent_speed_for_human "${EVAL_HUMAN_SPEED}")"
DEFAULT_AGENT_LATERAL_SPEED="$(agent_lateral_speed_for_human "${EVAL_HUMAN_SPEED}")"
EVAL_DRONE_MAX_SPEED="${EVAL_DRONE_MAX_SPEED:-${DEFAULT_AGENT_MAX_SPEED}}"
EVAL_DRONE_MAX_VX="${EVAL_DRONE_MAX_VX:-${DEFAULT_AGENT_MAX_SPEED}}"
EVAL_DRONE_MAX_VY="${EVAL_DRONE_MAX_VY:-${DEFAULT_AGENT_LATERAL_SPEED}}"
EVAL_HUMAN_GOAL_MIN_DISTANCE="${EVAL_HUMAN_GOAL_MIN_DISTANCE:-700}"
EVAL_HUMAN_GOAL_MAX_DISTANCE="${EVAL_HUMAN_GOAL_MAX_DISTANCE:-2200}"
EVAL_OPEN_SPAWN="${EVAL_OPEN_SPAWN:-1}"
EVAL_OPEN_SPAWN_RADIUS="${EVAL_OPEN_SPAWN_RADIUS:-900}"
EVAL_MIN_OPEN_CLEARANCE="${EVAL_MIN_OPEN_CLEARANCE:-300}"
EVAL_OPEN_SPAWN_CANDIDATES="${EVAL_OPEN_SPAWN_CANDIDATES:-128}"
EVAL_GROUND_NAVMESH_TOLERANCE="${EVAL_GROUND_NAVMESH_TOLERANCE:-300}"
EVAL_DRONE_NAVMESH_TOLERANCE="${EVAL_DRONE_NAVMESH_TOLERANCE:-800}"
EVAL_REQUIRE_VISUAL_TARGET="${EVAL_REQUIRE_VISUAL_TARGET:-1}"
EVAL_REQUIRE_CENTERED_TARGET="${EVAL_REQUIRE_CENTERED_TARGET:-0}"
EVAL_USE_MASK_VISIBILITY="${EVAL_USE_MASK_VISIBILITY:-1}"
EVAL_MIN_VISIBLE_RATIO="${EVAL_MIN_VISIBLE_RATIO:-0.001}"
EVAL_TARGET_CENTER_TOLERANCE="${EVAL_TARGET_CENTER_TOLERANCE:-0.35}"
EVAL_DRONE_CAMERA_FIXED_PITCH="${EVAL_DRONE_CAMERA_FIXED_PITCH:--60}"
EVAL_DRONE_CAMERA_PITCHES="${EVAL_DRONE_CAMERA_PITCHES:--60}"
EVAL_DRONE_CAMERA_FIXED_YAW="${EVAL_DRONE_CAMERA_FIXED_YAW:-0}"
EVAL_DRONE_CAMERA_YAW_OFFSETS="${EVAL_DRONE_CAMERA_YAW_OFFSETS:-0}"
EVAL_ROBOTDOG_CAMERA_MODE="${EVAL_ROBOTDOG_CAMERA_MODE:-fixed}"
EVAL_DRONE_CAMERA_MODE="${EVAL_DRONE_CAMERA_MODE:-fixed}"
EVAL_LOCK_DRONE_CAMERA_WORLD_XY="${EVAL_LOCK_DRONE_CAMERA_WORLD_XY:-1}"
EVAL_DRONE_CAMERA_FORWARD_OFFSET="${EVAL_DRONE_CAMERA_FORWARD_OFFSET:-35}"
EVAL_DRONE_CAMERA_Z_OFFSET="${EVAL_DRONE_CAMERA_Z_OFFSET:--60}"
EVAL_DRONE_FOV="${EVAL_DRONE_FOV:-100}"
EVAL_MAX_CAMERA_SEARCH_CANDIDATES="${EVAL_MAX_CAMERA_SEARCH_CANDIDATES:-12}"
EVAL_SNAP_HEADING="${EVAL_SNAP_HEADING:-0}"
EVAL_FOLLOW_BEHIND="${EVAL_FOLLOW_BEHIND:-1}"
# 无人机训练使用实际执行速度标签；评估时通过 EVAL_DRONE_*_SCALE 映射到底层控制命令。
ACTION_FIELD="${ACTION_FIELD:-auto}"

: '使用评估：
RUN_ORGANIZE=0 RUN_SPLIT=0 RUN_PROCESS=0 RUN_CACHE=0 RUN_TRAIN=0 RUN_EVAL=1 \
CKPT_DIR=/data/hdt/ntv_data/ckpt/New_paths_training_drone_single_frozen_qwen_10ep \
EVAL_ROOT=/data/hdt/ntv_data/sim_data/eval/New_paths_training_drone_single_10to1 \
EVAL_GPU=1 RENDER_GPU=1 \
bash /data/hdt/newtrackvla/sh/run_drone_single_agent_pipeline.sh
'


MANIFEST="${MANIFEST:-${SPLIT_ROOT}/split_manifest.json}"
TRAIN_RAW="${SPLIT_ROOT}/train_raw"
CACHE_ROOT="${TRAIN_DATA_ROOT}/vision_cache"

bool_cli_arg() {
    local name="$1"
    local value="$2"
    case "${value}" in
        1|true|TRUE|yes|YES|on|ON) printf -- "--%s\n" "${name}" ;;
        *) printf -- "--no-%s\n" "${name}" ;;
    esac
}

run_organize() {
    if [[ -d "${RAW_ROOT}" ]] && [[ -f "${RAW_ROOT}/organization_manifest.json" ]]; then
        echo "[organize] reuse ${RAW_ROOT}"
        return
    fi
    "${PYTHON_BIN}" tools/organize_drone_data.py \
        --input-root "${SOURCE_ROOT}" \
        --output-root "${RAW_ROOT}" \
        --seed-name seed_100
}

run_split() {
    if [[ -f "${MANIFEST}" ]]; then
        echo "[split] reuse ${MANIFEST}"
        return
    fi
    "${PYTHON_BIN}" tools/split_unrealzoo_single_agent_data.py \
        --input-root "${RAW_ROOT}" \
        --output-root "${SPLIT_ROOT}" \
        --train-parts 10 \
        --test-parts 1 \
        --seed 42
}

run_process() {
    "${PYTHON_BIN}" -m tools.make_tracking_data \
        --input_root "${TRAIN_RAW}" \
        --output_root "${TRAIN_DATA_ROOT}" \
        --history 31 \
        --horizon 9 \
        --dt 0.1 \
        --agent drone \
        --action_field "${ACTION_FIELD}" \
        --no_aggregate \
        --only_success \
        --min_following_rate 0.5 \
        --exclude_collision \
        --min_total_steps 50
}

run_cache() {
    CUDA_VISIBLE_DEVICES="${CACHE_GPU}" "${PYTHON_BIN}" -m tools.precache_frames \
        --data_root "${TRAIN_DATA_ROOT}" \
        --cache_root "${CACHE_ROOT}" \
        --batch_size 32 \
        --image_size 384
}

run_train() {
    PATH="$(dirname "${PYTHON_BIN}"):${PATH}" \
    TRAIN_JSON="${TRAIN_DATA_ROOT}/jsonl" \
    CACHE_ROOT="${CACHE_ROOT}" \
    OUT_DIR="${CKPT_DIR}" \
    CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" \
    NUM_GPUS="${NUM_GPUS}" \
    BATCH_SIZE="${BATCH_SIZE:-14}" \
    GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-7}" \
    EPOCHS="${EPOCHS:-10}" \
    N_WAYPOINTS=10 \
    HISTORY=31 \
    LR="${LR:-2e-5}" \
    ALPHA_XY=1 \
    BETA_NAV=100 \
    FREEZE_LLM=1 \
    SAVE_TRAJECTORIES=0 \
    bash sh/train_with_estimate.sh
}

run_eval() {
    mkdir -p "${EVAL_ROOT}"
    local video_flag="--no-save-video"
    [[ "${SAVE_EVAL_VIDEO}" == "1" ]] && video_flag="--save-video"
    local distance_flag="--no-require-success-distance"
    [[ "${EVAL_REQUIRE_SUCCESS_DISTANCE}" == "1" ]] && distance_flag="--require-success-distance"
    local face_target_flag="--no-face-target-before-step"
    [[ "${EVAL_FACE_TARGET_BEFORE_STEP}" == "1" ]] && face_target_flag="--face-target-before-step"
    local open_spawn_flag
    open_spawn_flag="$(bool_cli_arg open-spawn "${EVAL_OPEN_SPAWN}")"
    local require_visual_target_flag
    require_visual_target_flag="$(bool_cli_arg require-visual-target "${EVAL_REQUIRE_VISUAL_TARGET}")"
    local require_centered_target_flag
    require_centered_target_flag="$(bool_cli_arg require-centered-target "${EVAL_REQUIRE_CENTERED_TARGET}")"
    local use_mask_visibility_flag
    use_mask_visibility_flag="$(bool_cli_arg use-mask-visibility "${EVAL_USE_MASK_VISIBILITY}")"
    local lock_drone_camera_world_xy_flag
    lock_drone_camera_world_xy_flag="$(bool_cli_arg lock-drone-camera-world-xy "${EVAL_LOCK_DRONE_CAMERA_WORLD_XY}")"
    local snap_heading_flag
    snap_heading_flag="$(bool_cli_arg snap-heading "${EVAL_SNAP_HEADING}")"
    local follow_behind_flag
    follow_behind_flag="$(bool_cli_arg follow-behind "${EVAL_FOLLOW_BEHIND}")"
    local disable_ue_input_flag
    disable_ue_input_flag="$(bool_cli_arg disable-ue-input "${EVAL_DISABLE_UE_INPUT}")"
    local deterministic_step_flag
    deterministic_step_flag="$(bool_cli_arg deterministic-step "${EVAL_DETERMINISTIC_STEP}")"
    local init_from_recorded_agent_pose_flag
    init_from_recorded_agent_pose_flag="$(bool_cli_arg init-from-recorded-agent-pose "${EVAL_INIT_FROM_RECORDED_AGENT_POSE}")"
    local init_followers_behind_target_flag
    init_followers_behind_target_flag="$(bool_cli_arg init-followers-behind-target "${EVAL_INIT_FOLLOWERS_BEHIND_TARGET}")"

    while IFS=$'\t' read -r scene episodes waypoint_command_dt; do
        echo "[eval] scene=${scene} episodes=${episodes} waypoint_command_dt=${waypoint_command_dt}"
        CUDA_VISIBLE_DEVICES="${EVAL_GPU}" "${PYTHON_BIN}" -u eval_unrealzoo_single_agent.py \
            --agent drone \
            --ckpt "${CKPT_DIR}" \
            --test-manifest "${MANIFEST}" \
            --save-path "${EVAL_ROOT}" \
            --env-id "${scene}" \
            --episodes "${episodes}" \
            --render-gpu "${RENDER_GPU}" \
            --max-steps 600 \
            "${disable_ue_input_flag}" \
            --target-replay-mode "${EVAL_TARGET_REPLAY_MODE}" \
            --target-path-min-spacing "${EVAL_TARGET_PATH_MIN_SPACING}" \
            --target-path-reach-distance "${EVAL_TARGET_PATH_REACH_DISTANCE}" \
            --target-goal-reach-distance "${EVAL_TARGET_GOAL_REACH_DISTANCE}" \
            --target-stop-wait-min-steps "${EVAL_TARGET_STOP_WAIT_MIN_STEPS}" \
            --target-stop-wait-max-steps "${EVAL_TARGET_STOP_WAIT_MAX_STEPS}" \
            --waypoint-index "${EVAL_WAYPOINT_INDEX}" \
            --waypoint-horizon-steps "${EVAL_WAYPOINT_HORIZON_STEPS}" \
            --waypoint-command-dt "${waypoint_command_dt}" \
            --dt "${EVAL_DT}" \
            --history-frame-dt "${EVAL_HISTORY_FRAME_DT}" \
            --history-sampling-mode "${EVAL_HISTORY_SAMPLING_MODE}" \
            --fps "${EVAL_FPS}" \
            "${deterministic_step_flag}" \
            "$(bool_cli_arg realtime-waypoint-timing "${EVAL_REALTIME_WAYPOINT_TIMING}")" \
            --realtime-waypoint-min-seconds "${EVAL_REALTIME_WAYPOINT_MIN_SECONDS}" \
            --realtime-waypoint-max-seconds "${EVAL_REALTIME_WAYPOINT_MAX_SECONDS}" \
            --drone-vx-scale "${EVAL_DRONE_VX_SCALE}" \
            --drone-vy-scale "${EVAL_DRONE_VY_SCALE}" \
            --drone-yaw-sign "${EVAL_DRONE_YAW_SIGN}" \
            --drone-yaw-scale "${EVAL_DRONE_YAW_SCALE}" \
            --drone-max-vx "${EVAL_DRONE_MAX_VX}" \
            --drone-max-vy "${EVAL_DRONE_MAX_VY}" \
            --drone-max-yaw-rate "${EVAL_DRONE_MAX_YAW_RATE}" \
            --drone-success-distance "${EVAL_DRONE_SUCCESS_DISTANCE}" \
            --drone-ideal-follow-dist "${EVAL_DRONE_IDEAL_FOLLOW_DIST}" \
            --drone-min-follow-dist "${EVAL_DRONE_MIN_FOLLOW_DIST}" \
            --drone-max-follow-dist "${EVAL_DRONE_MAX_FOLLOW_DIST}" \
            --drone-height "${EVAL_DRONE_HEIGHT}" \
            --drone-max-speed "${EVAL_DRONE_MAX_SPEED}" \
            --max-lost-steps "${EVAL_MAX_LOST_STEPS}" \
            --max-episode-seconds "${EVAL_MAX_EPISODE_SECONDS}" \
            --success-rate-threshold "${EVAL_SUCCESS_RATE_THRESHOLD}" \
            --min-centered-rate "${EVAL_MIN_CENTERED_RATE}" \
            --min-success-steps "${EVAL_MIN_SUCCESS_STEPS}" \
            "${init_from_recorded_agent_pose_flag}" \
            "${init_followers_behind_target_flag}" \
            --human-speed "${EVAL_HUMAN_SPEED}" \
            --human-goal-min-distance "${EVAL_HUMAN_GOAL_MIN_DISTANCE}" \
            --human-goal-max-distance "${EVAL_HUMAN_GOAL_MAX_DISTANCE}" \
            "${open_spawn_flag}" \
            --open-spawn-radius "${EVAL_OPEN_SPAWN_RADIUS}" \
            --min-open-clearance "${EVAL_MIN_OPEN_CLEARANCE}" \
            --open-spawn-candidates "${EVAL_OPEN_SPAWN_CANDIDATES}" \
            --ground-navmesh-tolerance "${EVAL_GROUND_NAVMESH_TOLERANCE}" \
            --drone-navmesh-tolerance "${EVAL_DRONE_NAVMESH_TOLERANCE}" \
            "${require_visual_target_flag}" \
            "${require_centered_target_flag}" \
            "${use_mask_visibility_flag}" \
            --min-visible-ratio "${EVAL_MIN_VISIBLE_RATIO}" \
            --target-center-tolerance "${EVAL_TARGET_CENTER_TOLERANCE}" \
            --drone-camera-fixed-pitch "${EVAL_DRONE_CAMERA_FIXED_PITCH}" \
            --drone-camera-pitches "${EVAL_DRONE_CAMERA_PITCHES}" \
            --drone-camera-fixed-yaw "${EVAL_DRONE_CAMERA_FIXED_YAW}" \
            --drone-camera-yaw-offsets "${EVAL_DRONE_CAMERA_YAW_OFFSETS}" \
            --robotdog-camera-mode "${EVAL_ROBOTDOG_CAMERA_MODE}" \
            --drone-camera-mode "${EVAL_DRONE_CAMERA_MODE}" \
            "${lock_drone_camera_world_xy_flag}" \
            --drone-camera-forward-offset "${EVAL_DRONE_CAMERA_FORWARD_OFFSET}" \
            --drone-camera-z-offset "${EVAL_DRONE_CAMERA_Z_OFFSET}" \
            --drone-fov "${EVAL_DRONE_FOV}" \
            --max-camera-search-candidates "${EVAL_MAX_CAMERA_SEARCH_CANDIDATES}" \
            "${snap_heading_flag}" \
            "${follow_behind_flag}" \
            --launch-retries "${EVAL_LAUNCH_RETRIES}" \
            --ue-sleep-time "${EVAL_UE_SLEEP_TIME}" \
            --settle-steps "${EVAL_SETTLE_STEPS}" \
            "${face_target_flag}" \
            "${distance_flag}" \
            "${video_flag}"
    done < <(
        "${PYTHON_BIN}" - \
            "${MANIFEST}" \
            "${EVAL_SCENES}" \
            "${EVAL_EPISODES}" \
            "${EVAL_WAYPOINT_COMMAND_DT}" \
            "${EVAL_WAYPOINT_COMMAND_DT_BY_SCENE}" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
requested_scenes = [item.strip() for item in sys.argv[2].split(",") if item.strip()]
requested_episodes = sys.argv[3].strip()
default_waypoint_dt = sys.argv[4].strip()
waypoint_dt_by_scene = {}
for assignment in [item.strip() for item in sys.argv[5].split(",") if item.strip()]:
    scene, separator, value = assignment.rpartition("=")
    if not separator or not scene or not value:
        raise ValueError(
            "EVAL_WAYPOINT_COMMAND_DT_BY_SCENE entries must be SCENE=SECONDS"
        )
    waypoint_dt_by_scene[scene] = value
scene_counts = manifest["scene_counts"]

# Python preserves JSON object insertion order; follow the manifest order so
# users can deliberately place slow or unstable scenes at the end.
scenes = requested_scenes or list(scene_counts)
for scene in scenes:
    if scene not in scene_counts:
        raise KeyError(f"Requested eval scene is absent from manifest: {scene}")
    manifest_episodes = int(scene_counts[scene]["test"])
    episodes = (
        manifest_episodes
        if requested_episodes.lower() == "manifest"
        else min(int(requested_episodes), manifest_episodes)
    )
    waypoint_dt = waypoint_dt_by_scene.get(scene, default_waypoint_dt)
    print(f"{scene}\t{episodes}\t{waypoint_dt}")
PY
    )

    "${PYTHON_BIN}" -m tools.calculate_unrealzoo_single_agent_metrics --agent drone --eval-dir "${EVAL_ROOT}"
}

echo "Drone single-agent pipeline"
echo "  source:    ${SOURCE_ROOT}"
echo "  raw:       ${RAW_ROOT}"
echo "  split:     ${SPLIT_ROOT}"
echo "  train:     ${TRAIN_DATA_ROOT}"
echo "  checkpoint:${CKPT_DIR}"
echo "  eval:      ${EVAL_ROOT}"

if [[ "${RUN_ORGANIZE}" != "0" ]]; then run_organize; fi
if [[ "${RUN_SPLIT}" != "0" ]]; then run_split; fi
if [[ "${RUN_PROCESS}" != "0" ]]; then run_process; fi
if [[ "${RUN_CACHE}" != "0" ]]; then run_cache; fi
if [[ "${RUN_TRAIN}" != "0" ]]; then run_train; fi
if [[ "${RUN_EVAL}" != "0" ]]; then run_eval; fi
