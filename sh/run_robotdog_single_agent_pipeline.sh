#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/hdt/miniconda3/envs/omtracknew/bin/python}"
RAW_ROOT="${RAW_ROOT:-/data/hdt/ntv_data/sim_data/robotdog}"
SPLIT_ROOT="${SPLIT_ROOT:-/data/hdt/ntv_data/sim_data/robotdog_split_10to1}"
TRAIN_DATA_ROOT="${TRAIN_DATA_ROOT:-/data/hdt/ntv_data/data/robotdog_single_10to1/train}"
CKPT_DIR="${CKPT_DIR:-/data/hdt/ntv_data/ckpt/robotdog_single_frozen_qwen_10ep}"
EVAL_ROOT="${EVAL_ROOT:-/data/hdt/ntv_data/sim_data/eval/robotdog_single_10to1}"

RUN_SPLIT="${RUN_SPLIT:-1}"
RUN_PROCESS="${RUN_PROCESS:-1}"
RUN_CACHE="${RUN_CACHE:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

TRAIN_GPUS="${TRAIN_GPUS:-1}"
NUM_GPUS="${NUM_GPUS:-1}"
CACHE_GPU="${CACHE_GPU:-1}"

# 使用评估：
# RUN_SPLIT=0 RUN_PROCESS=0 RUN_CACHE=0 RUN_TRAIN=0 RUN_EVAL=1 bash sh/run_robotdog_single_agent_pipeline.sh

# ===== UnrealZoo 机器狗评估参数 =====
# GPU 和输出。
EVAL_GPU="${EVAL_GPU:-1}"
RENDER_GPU="${RENDER_GPU:-1}"
SAVE_EVAL_VIDEO="${SAVE_EVAL_VIDEO:-1}"
EVAL_DEVICE="${EVAL_DEVICE:-}"
EVAL_SEED="${EVAL_SEED:-100}"

# 回合长度和成功判定。
EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-600}"                         # 每个回合最大仿真步数。
EVAL_MAX_LOST_STEPS="${EVAL_MAX_LOST_STEPS:-30}"                 # 连续多少步没跟上就判丢失。
EVAL_MAX_EPISODE_SECONDS="${EVAL_MAX_EPISODE_SECONDS:-600.0}"    # 每个回合的真实时间超时。
EVAL_MIN_SUCCESS_STEPS="${EVAL_MIN_SUCCESS_STEPS:-20}"           # 判成功所需的最少有效步数。
EVAL_SUCCESS_RATE_THRESHOLD="${EVAL_SUCCESS_RATE_THRESHOLD:-0.8}" # 跟踪率达到该值才算成功。
EVAL_MIN_CENTERED_RATE="${EVAL_MIN_CENTERED_RATE:-0.8}"
EVAL_REQUIRE_SUCCESS_DISTANCE="${EVAL_REQUIRE_SUCCESS_DISTANCE:-1}" # 1: 可见且距离达标才算跟上；0: 只看可见。
EVAL_ROBOTDOG_SUCCESS_DISTANCE="${EVAL_ROBOTDOG_SUCCESS_DISTANCE:-8.0}" # 跟踪成功距离阈值，单位 m。
EVAL_ROBOTDOG_LOST_DISTANCE="${EVAL_ROBOTDOG_LOST_DISTANCE:-0.0}" # >0 时仅超过该距离才累计 Lost。

# 模型预测航点到机器狗动作的映射。
EVAL_WAYPOINT_INDEX="${EVAL_WAYPOINT_INDEX:-9}" # 约 0.9 秒后的累计 waypoint，用于覆盖约 1 秒推理周期
EVAL_WAYPOINT_HORIZON_STEPS="${EVAL_WAYPOINT_HORIZON_STEPS:-9}"
EVAL_DETERMINISTIC_STEP="${EVAL_DETERMINISTIC_STEP:-0}"
EVAL_REALTIME_WAYPOINT_TIMING="${EVAL_REALTIME_WAYPOINT_TIMING:-1}"
EVAL_REALTIME_WAYPOINT_MIN_SECONDS="${EVAL_REALTIME_WAYPOINT_MIN_SECONDS:-0.1}"
EVAL_REALTIME_WAYPOINT_MAX_SECONDS="${EVAL_REALTIME_WAYPOINT_MAX_SECONDS:-0.9}"
EVAL_SPEED_GAIN="${EVAL_SPEED_GAIN:-1.15}"                       # 速度增益。
EVAL_ROBOTDOG_MAX_TURN_DEG="${EVAL_ROBOTDOG_MAX_TURN_DEG:-30.0}"
EVAL_ROBOTDOG_YAW_SIGN="${EVAL_ROBOTDOG_YAW_SIGN:-1.0}"
EVAL_ROBOTDOG_YAW_SCALE="${EVAL_ROBOTDOG_YAW_SCALE:-1.0}"
EVAL_ROBOTDOG_MAX_LATERAL_SPEED="${EVAL_ROBOTDOG_MAX_LATERAL_SPEED:-0.45}"
EVAL_ROBOTDOG_MAX_YAW_RATE="${EVAL_ROBOTDOG_MAX_YAW_RATE:-1.0}"
EVAL_DT="${EVAL_DT:-0.1}"
EVAL_HISTORY_FRAME_DT="${EVAL_HISTORY_FRAME_DT:-0.1}"

# 目标人重放和初始状态。
EVAL_TARGET_REPLAY_MODE="${EVAL_TARGET_REPLAY_MODE:-path_goal}"   # path_goal 使用 test 集 target_pose 抽样路径点。
EVAL_TARGET_PATH_MIN_SPACING="${EVAL_TARGET_PATH_MIN_SPACING:-100}"
EVAL_TARGET_PATH_REACH_DISTANCE="${EVAL_TARGET_PATH_REACH_DISTANCE:-120}"
EVAL_TARGET_GOAL_REACH_DISTANCE="${EVAL_TARGET_GOAL_REACH_DISTANCE:-50}"
EVAL_TARGET_STOP_WAIT_MIN_STEPS="${EVAL_TARGET_STOP_WAIT_MIN_STEPS:-5}"
EVAL_TARGET_STOP_WAIT_MAX_STEPS="${EVAL_TARGET_STOP_WAIT_MAX_STEPS:-5}"
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
EVAL_ROBOTDOG_MAX_SPEED="${EVAL_ROBOTDOG_MAX_SPEED:-${DEFAULT_AGENT_MAX_SPEED}}"

EVAL_HUMAN_GOAL_MIN_DISTANCE="${EVAL_HUMAN_GOAL_MIN_DISTANCE:-700.0}"
EVAL_HUMAN_GOAL_MAX_DISTANCE="${EVAL_HUMAN_GOAL_MAX_DISTANCE:-2200.0}"
EVAL_INIT_FROM_RECORDED_AGENT_POSE="${EVAL_INIT_FROM_RECORDED_AGENT_POSE:-0}"
EVAL_INIT_FOLLOWERS_BEHIND_TARGET="${EVAL_INIT_FOLLOWERS_BEHIND_TARGET:-1}"
EVAL_OPEN_SPAWN="${EVAL_OPEN_SPAWN:-1}"
EVAL_OPEN_SPAWN_RADIUS="${EVAL_OPEN_SPAWN_RADIUS:-900.0}"
EVAL_MIN_OPEN_CLEARANCE="${EVAL_MIN_OPEN_CLEARANCE:-300.0}"
EVAL_OPEN_SPAWN_CANDIDATES="${EVAL_OPEN_SPAWN_CANDIDATES:-128}"
EVAL_GROUND_NAVMESH_TOLERANCE="${EVAL_GROUND_NAVMESH_TOLERANCE:-300.0}"

# 观测、渲染和相机搜索。
EVAL_SETTLE_STEPS="${EVAL_SETTLE_STEPS:-5}"
EVAL_FLUSH_INITIAL_OBSERVATION="${EVAL_FLUSH_INITIAL_OBSERVATION:-1}"
EVAL_WIDTH="${EVAL_WIDTH:-640}"
EVAL_HEIGHT="${EVAL_HEIGHT:-480}"
EVAL_FPS="${EVAL_FPS:-10}"
EVAL_OFFSCREEN="${EVAL_OFFSCREEN:-1}"
EVAL_TIME_DILATION="${EVAL_TIME_DILATION:--1}"
EVAL_LAUNCH_RETRIES="${EVAL_LAUNCH_RETRIES:-5}"
EVAL_TRAJECTORY_OVERLAY="${EVAL_TRAJECTORY_OVERLAY:-1}"
EVAL_TRAJECTORY_SCALE="${EVAL_TRAJECTORY_SCALE:-120.0}"
EVAL_REQUIRE_VISUAL_TARGET="${EVAL_REQUIRE_VISUAL_TARGET:-1}"
EVAL_REQUIRE_CENTERED_TARGET="${EVAL_REQUIRE_CENTERED_TARGET:-0}"
EVAL_USE_MASK_VISIBILITY="${EVAL_USE_MASK_VISIBILITY:-1}"
EVAL_MIN_VISIBLE_RATIO="${EVAL_MIN_VISIBLE_RATIO:-0.001}"
EVAL_TARGET_CENTER_TOLERANCE="${EVAL_TARGET_CENTER_TOLERANCE:-0.35}"
EVAL_MAX_CAMERA_SEARCH_CANDIDATES="${EVAL_MAX_CAMERA_SEARCH_CANDIDATES:-12}"
EVAL_SNAP_HEADING="${EVAL_SNAP_HEADING:-0}"
EVAL_FOLLOW_BEHIND="${EVAL_FOLLOW_BEHIND:-1}"
EVAL_TOP_VIEW_HEIGHT="${EVAL_TOP_VIEW_HEIGHT:-}"                 

# 机器狗相机、外观和初始跟随距离。
EVAL_ROBOTDOG_CAMERA_FORWARD="${EVAL_ROBOTDOG_CAMERA_FORWARD:-140.0}"
EVAL_ROBOTDOG_CAMERA_LATERAL="${EVAL_ROBOTDOG_CAMERA_LATERAL:-0.0}"
EVAL_ROBOTDOG_CAMERA_HEIGHT="${EVAL_ROBOTDOG_CAMERA_HEIGHT:-110.0}"
EVAL_ROBOTDOG_CAMERA_MOUNTS="${EVAL_ROBOTDOG_CAMERA_MOUNTS:-140:0:110,170:0:120,110:0:95,0:120:110}"
EVAL_ROBOTDOG_CAMERA_FIXED_PITCH="${EVAL_ROBOTDOG_CAMERA_FIXED_PITCH:-}" # 留空表示自动搜索俯仰角。
EVAL_ROBOTDOG_CAMERA_PITCHES="${EVAL_ROBOTDOG_CAMERA_PITCHES:--15,-8,0,8,15,22,-22}"
EVAL_ROBOTDOG_CAMERA_YAW_OFFSETS="${EVAL_ROBOTDOG_CAMERA_YAW_OFFSETS:-0,-8,8,-15,15}"
EVAL_ROBOTDOG_FOV="${EVAL_ROBOTDOG_FOV:-95.0}"
EVAL_MAX_SELF_VISIBLE_RATIO="${EVAL_MAX_SELF_VISIBLE_RATIO:-0.015}"
EVAL_HUMAN_APPEARANCE_MIN="${EVAL_HUMAN_APPEARANCE_MIN:-1}"
EVAL_HUMAN_APPEARANCE_MAX="${EVAL_HUMAN_APPEARANCE_MAX:-18}"
EVAL_ROBOTDOG_APPEARANCE_MIN="${EVAL_ROBOTDOG_APPEARANCE_MIN:-20}"
EVAL_ROBOTDOG_APPEARANCE_MAX="${EVAL_ROBOTDOG_APPEARANCE_MAX:-33}"
EVAL_ROBOTDOG_IDEAL_FOLLOW_DIST="${EVAL_ROBOTDOG_IDEAL_FOLLOW_DIST:-6.25}"
EVAL_ROBOTDOG_MIN_FOLLOW_DIST="${EVAL_ROBOTDOG_MIN_FOLLOW_DIST:-4.5}"
EVAL_ROBOTDOG_MAX_FOLLOW_DIST="${EVAL_ROBOTDOG_MAX_FOLLOW_DIST:-8.0}"

# 指令文本和少见参数兜底；EVAL_EXTRA_ARGS 按空格拆分。
EVAL_INSTRUCTION="${EVAL_INSTRUCTION:-Follow the target person without collision.}"
EVAL_EXTRA_ARGS="${EVAL_EXTRA_ARGS:-}"
ACTION_FIELD="${ACTION_FIELD:-auto}"

MANIFEST="${MANIFEST:-${SPLIT_ROOT}/split_manifest.json}"
TRAIN_RAW="${SPLIT_ROOT}/train_raw"
CACHE_ROOT="${TRAIN_DATA_ROOT}/vision_cache"

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
    EPOCHS=10 \
    N_WAYPOINTS=10 \
    HISTORY=31 \
    LR="${LR:-2e-5}" \
    ALPHA_XY=1 \
    BETA_NAV=100 \
    FREEZE_LLM=1 \
    SAVE_TRAJECTORIES=0 \
    bash sh/train_with_estimate.sh
}

bool_cli_arg() {
    local name="$1"
    local value="$2"
    case "${value}" in
        1|true|TRUE|yes|YES|on|ON) printf -- "--%s\n" "${name}" ;;
        *) printf -- "--no-%s\n" "${name}" ;;
    esac
}

run_eval() {
    mkdir -p "${EVAL_ROOT}"
    local video_flag="--no-save-video"
    [[ "${SAVE_EVAL_VIDEO}" == "1" ]] && video_flag="--save-video"
    local eval_args=(
        --ckpt "${CKPT_DIR}"
        --agent robotdog
        --test-manifest "${MANIFEST}"
        --save-path "${EVAL_ROOT}"
        --render-gpu "${RENDER_GPU}"
        --max-steps "${EVAL_MAX_STEPS}"
        --seed "${EVAL_SEED}"
        --instruction "${EVAL_INSTRUCTION}"
        --width "${EVAL_WIDTH}"
        --height "${EVAL_HEIGHT}"
        --fps "${EVAL_FPS}"
        --dt "${EVAL_DT}"
        --history-frame-dt "${EVAL_HISTORY_FRAME_DT}"
        "$(bool_cli_arg deterministic-step "${EVAL_DETERMINISTIC_STEP}")"
        "$(bool_cli_arg offscreen "${EVAL_OFFSCREEN}")"
        --time-dilation "${EVAL_TIME_DILATION}"
        --launch-retries "${EVAL_LAUNCH_RETRIES}"
        --settle-steps "${EVAL_SETTLE_STEPS}"
        "$(bool_cli_arg flush-initial-observation "${EVAL_FLUSH_INITIAL_OBSERVATION}")"
        "${video_flag}"
        "$(bool_cli_arg trajectory-overlay "${EVAL_TRAJECTORY_OVERLAY}")"
        --trajectory-scale "${EVAL_TRAJECTORY_SCALE}"
        --waypoint-index "${EVAL_WAYPOINT_INDEX}"
        --waypoint-horizon-steps "${EVAL_WAYPOINT_HORIZON_STEPS}"
        "$(bool_cli_arg realtime-waypoint-timing "${EVAL_REALTIME_WAYPOINT_TIMING}")"
        --realtime-waypoint-min-seconds "${EVAL_REALTIME_WAYPOINT_MIN_SECONDS}"
        --realtime-waypoint-max-seconds "${EVAL_REALTIME_WAYPOINT_MAX_SECONDS}"
        --robotdog-yaw-sign "${EVAL_ROBOTDOG_YAW_SIGN}"
        --robotdog-yaw-scale "${EVAL_ROBOTDOG_YAW_SCALE}"
        --robotdog-speed-gain "${EVAL_SPEED_GAIN}"
        --robotdog-max-speed "${EVAL_ROBOTDOG_MAX_SPEED}"
        --robotdog-max-turn-deg "${EVAL_ROBOTDOG_MAX_TURN_DEG}"
        --robotdog-max-lateral-speed "${EVAL_ROBOTDOG_MAX_LATERAL_SPEED}"
        --robotdog-max-yaw-rate "${EVAL_ROBOTDOG_MAX_YAW_RATE}"
        --robotdog-success-distance "${EVAL_ROBOTDOG_SUCCESS_DISTANCE}"
        --robotdog-lost-distance "${EVAL_ROBOTDOG_LOST_DISTANCE}"
        "$(bool_cli_arg require-success-distance "${EVAL_REQUIRE_SUCCESS_DISTANCE}")"
        --max-lost-steps "${EVAL_MAX_LOST_STEPS}"
        --max-episode-seconds "${EVAL_MAX_EPISODE_SECONDS}"
        --success-rate-threshold "${EVAL_SUCCESS_RATE_THRESHOLD}"
        --min-centered-rate "${EVAL_MIN_CENTERED_RATE}"
        --min-success-steps "${EVAL_MIN_SUCCESS_STEPS}"
        --target-replay-mode "${EVAL_TARGET_REPLAY_MODE}"
        --target-path-min-spacing "${EVAL_TARGET_PATH_MIN_SPACING}"
        --target-path-reach-distance "${EVAL_TARGET_PATH_REACH_DISTANCE}"
        --target-goal-reach-distance "${EVAL_TARGET_GOAL_REACH_DISTANCE}"
        --target-stop-wait-min-steps "${EVAL_TARGET_STOP_WAIT_MIN_STEPS}"
        --target-stop-wait-max-steps "${EVAL_TARGET_STOP_WAIT_MAX_STEPS}"
        "$(bool_cli_arg init-from-recorded-agent-pose "${EVAL_INIT_FROM_RECORDED_AGENT_POSE}")"
        "$(bool_cli_arg init-followers-behind-target "${EVAL_INIT_FOLLOWERS_BEHIND_TARGET}")"
        --human-speed "${EVAL_HUMAN_SPEED}"
        --human-goal-min-distance "${EVAL_HUMAN_GOAL_MIN_DISTANCE}"
        --human-goal-max-distance "${EVAL_HUMAN_GOAL_MAX_DISTANCE}"
        "$(bool_cli_arg open-spawn "${EVAL_OPEN_SPAWN}")"
        --open-spawn-radius "${EVAL_OPEN_SPAWN_RADIUS}"
        --min-open-clearance "${EVAL_MIN_OPEN_CLEARANCE}"
        --open-spawn-candidates "${EVAL_OPEN_SPAWN_CANDIDATES}"
        --ground-navmesh-tolerance "${EVAL_GROUND_NAVMESH_TOLERANCE}"
        "$(bool_cli_arg require-visual-target "${EVAL_REQUIRE_VISUAL_TARGET}")"
        "$(bool_cli_arg require-centered-target "${EVAL_REQUIRE_CENTERED_TARGET}")"
        "$(bool_cli_arg use-mask-visibility "${EVAL_USE_MASK_VISIBILITY}")"
        --min-visible-ratio "${EVAL_MIN_VISIBLE_RATIO}"
        --target-center-tolerance "${EVAL_TARGET_CENTER_TOLERANCE}"
        --human-appearance-min "${EVAL_HUMAN_APPEARANCE_MIN}"
        --human-appearance-max "${EVAL_HUMAN_APPEARANCE_MAX}"
        --robotdog-appearance-min "${EVAL_ROBOTDOG_APPEARANCE_MIN}"
        --robotdog-appearance-max "${EVAL_ROBOTDOG_APPEARANCE_MAX}"
        --robotdog-ideal-follow-dist "${EVAL_ROBOTDOG_IDEAL_FOLLOW_DIST}"
        --robotdog-min-follow-dist "${EVAL_ROBOTDOG_MIN_FOLLOW_DIST}"
        --robotdog-max-follow-dist "${EVAL_ROBOTDOG_MAX_FOLLOW_DIST}"
        --robotdog-camera-forward "${EVAL_ROBOTDOG_CAMERA_FORWARD}"
        --robotdog-camera-lateral "${EVAL_ROBOTDOG_CAMERA_LATERAL}"
        --robotdog-camera-height "${EVAL_ROBOTDOG_CAMERA_HEIGHT}"
        --robotdog-camera-mounts "${EVAL_ROBOTDOG_CAMERA_MOUNTS}"
        "--robotdog-camera-pitches=${EVAL_ROBOTDOG_CAMERA_PITCHES}"
        --robotdog-camera-yaw-offsets "${EVAL_ROBOTDOG_CAMERA_YAW_OFFSETS}"
        --robotdog-fov "${EVAL_ROBOTDOG_FOV}"
        --max-self-visible-ratio "${EVAL_MAX_SELF_VISIBLE_RATIO}"
        --max-camera-search-candidates "${EVAL_MAX_CAMERA_SEARCH_CANDIDATES}"
        "$(bool_cli_arg snap-heading "${EVAL_SNAP_HEADING}")"
        "$(bool_cli_arg follow-behind "${EVAL_FOLLOW_BEHIND}")"
    )
    if [[ -n "${EVAL_DEVICE}" ]]; then eval_args+=(--device "${EVAL_DEVICE}"); fi
    if [[ -n "${EVAL_TOP_VIEW_HEIGHT}" ]]; then eval_args+=(--top-view-height "${EVAL_TOP_VIEW_HEIGHT}"); fi
    if [[ -n "${EVAL_ROBOTDOG_CAMERA_FIXED_PITCH}" ]]; then
        eval_args+=(--robotdog-camera-fixed-pitch "${EVAL_ROBOTDOG_CAMERA_FIXED_PITCH}")
    fi
    if [[ -n "${EVAL_EXTRA_ARGS}" ]]; then
        local extra_eval_args=()
        read -r -a extra_eval_args <<< "${EVAL_EXTRA_ARGS}"
        eval_args+=("${extra_eval_args[@]}")
    fi

    while IFS=$'\t' read -r scene episodes; do
        echo "[eval] scene=${scene} episodes=${episodes}"
        CUDA_VISIBLE_DEVICES="${EVAL_GPU}" "${PYTHON_BIN}" -u eval_unrealzoo_single_agent.py \
            --env-id "${scene}" \
            --episodes "${episodes}" \
            "${eval_args[@]}"
    done < <(
        "${PYTHON_BIN}" - "${MANIFEST}" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
# Preserve the explicit scene order in split_manifest.json.
for scene, counts in manifest["scene_counts"].items():
    print(f"{scene}\t{counts['test']}")
PY
    )

    "${PYTHON_BIN}" -m tools.calculate_unrealzoo_single_agent_metrics --eval-dir "${EVAL_ROOT}"
}

echo "RobotDog single-agent pipeline"
echo "  raw:       ${RAW_ROOT}"
echo "  split:     ${SPLIT_ROOT}"
echo "  train:     ${TRAIN_DATA_ROOT}"
echo "  checkpoint:${CKPT_DIR}"
echo "  eval:      ${EVAL_ROOT}"

if [[ "${RUN_SPLIT}" != "0" ]]; then run_split; fi
if [[ "${RUN_PROCESS}" != "0" ]]; then run_process; fi
if [[ "${RUN_CACHE}" != "0" ]]; then run_cache; fi
if [[ "${RUN_TRAIN}" != "0" ]]; then run_train; fi
if [[ "${RUN_EVAL}" != "0" ]]; then run_eval; fi
