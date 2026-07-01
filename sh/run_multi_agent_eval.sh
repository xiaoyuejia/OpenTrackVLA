#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# UnrealZoo base-concat multi-agent evaluation only.
#
# Defaults are aligned with the single-agent drone and robotdog eval pipelines:
#   sh/run_drone_single_agent_pipeline.sh
#   sh/run_robotdog_single_agent_pipeline.sh
#
# Example:
#   EVAL_GPUS=3 bash sh/run_multi_agent_eval.sh
# -----------------------------------------------------------------------------

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/hdt/miniconda3/envs/omtracknew/bin/python}"
[[ -x "${PYTHON_BIN}" ]] || PYTHON_BIN="python"
EVAL_SCRIPT="${EVAL_SCRIPT:-eval_unrealzoo_multi_agent_base.py}"

CKPT_DIR="${CKPT_DIR:-/data/hdt/ntv_data/ckpt/New_paths_training_multi_agent_base_concat_gpu3_fast_b32}"
SPLIT_ROOT="${SPLIT_ROOT:-/data/hdt/ntv_data/sim_data/New_paths_training_multi_agent_split_10to1}"
TEST_TARGET_MANIFEST="${TEST_TARGET_MANIFEST:-${SPLIT_ROOT}/split_manifest.json}"
EVAL_ROOT="${EVAL_ROOT:-/data/hdt/ntv_data/sim_data/eval/New_paths_training_multi_agent_base_concat_gpu3_fast_b32_10to1}"

# Leave empty to use every scene in TEST_TARGET_MANIFEST. Use comma-separated env ids to restrict.
EVAL_SCENES="${EVAL_SCENES:-}"
# "manifest" means use the test episode count for each scene, matching the single-agent pipelines.
EVAL_EPISODES="${EVAL_EPISODES:-manifest}"
EVAL_GPUS="${EVAL_GPUS:-${EVAL_GPU:-3}}"
RENDER_GPUS="${RENDER_GPUS:-${RENDER_GPU:-${EVAL_GPUS}}}"

# General eval/runtime settings.
EVAL_SEED="${EVAL_SEED:-100}"
EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-600}"
EVAL_MAX_LOST_STEPS="${EVAL_MAX_LOST_STEPS:-20}"
EVAL_MAX_FAILURE_STEPS="${EVAL_MAX_FAILURE_STEPS:-0}"
EVAL_FAILURE_WARMUP_STEPS="${EVAL_FAILURE_WARMUP_STEPS:-20}"
EVAL_MAX_EPISODE_SECONDS="${EVAL_MAX_EPISODE_SECONDS:-0}"
EVAL_SUCCESS_RATE_THRESHOLD="${EVAL_SUCCESS_RATE_THRESHOLD:-0.5}"
EVAL_MIN_SUCCESS_STEPS="${EVAL_MIN_SUCCESS_STEPS:-20}"
EVAL_WIDTH="${EVAL_WIDTH:-640}"
EVAL_HEIGHT="${EVAL_HEIGHT:-480}"
EVAL_FPS="${EVAL_FPS:-10}"
EVAL_DT="${EVAL_DT:-0.1}"
EVAL_HISTORY_FRAME_DT="${EVAL_HISTORY_FRAME_DT:-0.1}"
EVAL_DETERMINISTIC_STEP="${EVAL_DETERMINISTIC_STEP:-1}"
EVAL_REALTIME_WAYPOINT_TIMING="${EVAL_REALTIME_WAYPOINT_TIMING:-0}"
EVAL_REALTIME_WAYPOINT_MIN_SECONDS="${EVAL_REALTIME_WAYPOINT_MIN_SECONDS:-0.1}"
EVAL_REALTIME_WAYPOINT_MAX_SECONDS="${EVAL_REALTIME_WAYPOINT_MAX_SECONDS:-0.9}"
EVAL_OFFSCREEN="${EVAL_OFFSCREEN:-1}"
EVAL_TIME_DILATION="${EVAL_TIME_DILATION:--1}"
EVAL_DISABLE_UE_INPUT="${EVAL_DISABLE_UE_INPUT:-1}"
EVAL_LAUNCH_RETRIES="${EVAL_LAUNCH_RETRIES:-5}"
SAVE_EVAL_VIDEO="${SAVE_EVAL_VIDEO:-1}"
EVAL_WRITE_GLOBAL_VIDEO="${EVAL_WRITE_GLOBAL_VIDEO:-1}"
EVAL_TRAJECTORY_OVERLAY="${EVAL_TRAJECTORY_OVERLAY:-1}"
EVAL_TRAJECTORY_SCALE="${EVAL_TRAJECTORY_SCALE:-120.0}"
EVAL_BBOX_SOURCE="${EVAL_BBOX_SOURCE:-none}"
EVAL_FACE_TARGET_BEFORE_STEP="${EVAL_FACE_TARGET_BEFORE_STEP:-0}"
EVAL_INSTRUCTION="${EVAL_INSTRUCTION:-The aerial drone and the ground robot dog must cooperatively track the same target person. The drone should follow the person from the air, and the robot dog should follow the same person on the ground.}"
EVAL_JOINT_INSTRUCTION="${EVAL_JOINT_INSTRUCTION:-${EVAL_INSTRUCTION}}"
EVAL_AGENT1_INSTRUCTION="${EVAL_AGENT1_INSTRUCTION:-}"
EVAL_AGENT2_INSTRUCTION="${EVAL_AGENT2_INSTRUCTION:-}"
EVAL_PLANNER_DEBUG_STEPS="${EVAL_PLANNER_DEBUG_STEPS:-5}"
EVAL_EXTRA_ARGS="${EVAL_EXTRA_ARGS:-}"

# Waypoint/action mapping. Both single-agent scripts use waypoint index 9.
EVAL_WAYPOINT_INDEX="${EVAL_WAYPOINT_INDEX:-9}"
EVAL_DRONE_WAYPOINT_INDEX="${EVAL_DRONE_WAYPOINT_INDEX:-9}"
EVAL_ROBOTDOG_WAYPOINT_INDEX="${EVAL_ROBOTDOG_WAYPOINT_INDEX:-9}"
EVAL_WAYPOINT_HORIZON_STEPS="${EVAL_WAYPOINT_HORIZON_STEPS:-9}"

# Human replay/spawn settings aligned with the single-agent scripts.
EVAL_TARGET_REPLAY_MODE="${EVAL_TARGET_REPLAY_MODE:-path_goal}"
EVAL_TARGET_PATH_MIN_SPACING="${EVAL_TARGET_PATH_MIN_SPACING:-100}"
EVAL_TARGET_PATH_REACH_DISTANCE="${EVAL_TARGET_PATH_REACH_DISTANCE:-120}"
EVAL_TARGET_GOAL_REACH_DISTANCE="${EVAL_TARGET_GOAL_REACH_DISTANCE:-50}"
EVAL_TARGET_STOP_WAIT_MIN_STEPS="${EVAL_TARGET_STOP_WAIT_MIN_STEPS:-5}"
EVAL_TARGET_STOP_WAIT_MAX_STEPS="${EVAL_TARGET_STOP_WAIT_MAX_STEPS:-15}"
EVAL_HUMAN_SPEED="${EVAL_HUMAN_SPEED:-90}"
EVAL_HUMAN_GOAL_MIN_DISTANCE="${EVAL_HUMAN_GOAL_MIN_DISTANCE:-700.0}"
EVAL_HUMAN_GOAL_MAX_DISTANCE="${EVAL_HUMAN_GOAL_MAX_DISTANCE:-2200.0}"
EVAL_OPEN_SPAWN="${EVAL_OPEN_SPAWN:-1}"
EVAL_OPEN_SPAWN_RADIUS="${EVAL_OPEN_SPAWN_RADIUS:-900.0}"
EVAL_MIN_OPEN_CLEARANCE="${EVAL_MIN_OPEN_CLEARANCE:-300.0}"
EVAL_OPEN_SPAWN_CANDIDATES="${EVAL_OPEN_SPAWN_CANDIDATES:-128}"
EVAL_GROUND_NAVMESH_TOLERANCE="${EVAL_GROUND_NAVMESH_TOLERANCE:-300.0}"
EVAL_DRONE_NAVMESH_TOLERANCE="${EVAL_DRONE_NAVMESH_TOLERANCE:-800.0}"

# Visibility/camera search.
EVAL_REQUIRE_VISUAL_TARGET="${EVAL_REQUIRE_VISUAL_TARGET:-0}"
EVAL_REQUIRE_CENTERED_TARGET="${EVAL_REQUIRE_CENTERED_TARGET:-0}"
EVAL_USE_MASK_VISIBILITY="${EVAL_USE_MASK_VISIBILITY:-1}"
EVAL_MIN_VISIBLE_RATIO="${EVAL_MIN_VISIBLE_RATIO:-0.001}"
EVAL_TARGET_CENTER_TOLERANCE="${EVAL_TARGET_CENTER_TOLERANCE:-0.35}"
EVAL_MAX_CAMERA_SEARCH_CANDIDATES="${EVAL_MAX_CAMERA_SEARCH_CANDIDATES:-12}"
EVAL_SNAP_HEADING="${EVAL_SNAP_HEADING:-1}"
EVAL_FOLLOW_BEHIND="${EVAL_FOLLOW_BEHIND:-1}"
EVAL_TOP_VIEW_HEIGHT="${EVAL_TOP_VIEW_HEIGHT:-}"
EVAL_INIT_FROM_RECORDED_AGENT_POSES="${EVAL_INIT_FROM_RECORDED_AGENT_POSES:-1}"

# Appearance ranges.
EVAL_HUMAN_APPEARANCE_MIN="${EVAL_HUMAN_APPEARANCE_MIN:-1}"
EVAL_HUMAN_APPEARANCE_MAX="${EVAL_HUMAN_APPEARANCE_MAX:-18}"
EVAL_ROBOTDOG_APPEARANCE_MIN="${EVAL_ROBOTDOG_APPEARANCE_MIN:-20}"
EVAL_ROBOTDOG_APPEARANCE_MAX="${EVAL_ROBOTDOG_APPEARANCE_MAX:-33}"

# RobotDog range aligned with the multi-agent training-data distribution.
EVAL_ROBOTDOG_SUCCESS_DISTANCE="${EVAL_ROBOTDOG_SUCCESS_DISTANCE:-8.0}"
EVAL_ROBOTDOG_LOST_DISTANCE="${EVAL_ROBOTDOG_LOST_DISTANCE:-10.0}"
EVAL_ROBOTDOG_YAW_SIGN="${EVAL_ROBOTDOG_YAW_SIGN:-1.0}"
EVAL_ROBOTDOG_YAW_SCALE="${EVAL_ROBOTDOG_YAW_SCALE:-1.0}"
EVAL_ROBOTDOG_SPEED_GAIN="${EVAL_ROBOTDOG_SPEED_GAIN:-1.15}"
EVAL_ROBOTDOG_MAX_SPEED="${EVAL_ROBOTDOG_MAX_SPEED:-1.05}"
EVAL_ROBOTDOG_MAX_TURN_DEG="${EVAL_ROBOTDOG_MAX_TURN_DEG:-30.0}"
EVAL_ROBOTDOG_MAX_LATERAL_SPEED="${EVAL_ROBOTDOG_MAX_LATERAL_SPEED:-0.45}"
EVAL_ROBOTDOG_MAX_YAW_RATE="${EVAL_ROBOTDOG_MAX_YAW_RATE:-1.0}"
EVAL_ROBOTDOG_IDEAL_FOLLOW_DIST="${EVAL_ROBOTDOG_IDEAL_FOLLOW_DIST:-6.25}"
EVAL_ROBOTDOG_MIN_FOLLOW_DIST="${EVAL_ROBOTDOG_MIN_FOLLOW_DIST:-4.5}"
EVAL_ROBOTDOG_MAX_FOLLOW_DIST="${EVAL_ROBOTDOG_MAX_FOLLOW_DIST:-8.0}"
EVAL_ROBOTDOG_CAMERA_FORWARD="${EVAL_ROBOTDOG_CAMERA_FORWARD:-140.0}"
EVAL_ROBOTDOG_CAMERA_LATERAL="${EVAL_ROBOTDOG_CAMERA_LATERAL:-0.0}"
EVAL_ROBOTDOG_CAMERA_HEIGHT="${EVAL_ROBOTDOG_CAMERA_HEIGHT:-110.0}"
EVAL_ROBOTDOG_CAMERA_MOUNTS="${EVAL_ROBOTDOG_CAMERA_MOUNTS:-140:0:110,170:0:120,110:0:95,0:120:110}"
EVAL_ROBOTDOG_CAMERA_FIXED_PITCH="${EVAL_ROBOTDOG_CAMERA_FIXED_PITCH:-}"
EVAL_ROBOTDOG_CAMERA_PITCHES="${EVAL_ROBOTDOG_CAMERA_PITCHES:--15,-8,0,8,15,22,-22}"
EVAL_ROBOTDOG_CAMERA_YAW_OFFSETS="${EVAL_ROBOTDOG_CAMERA_YAW_OFFSETS:-0,-8,8,-15,15}"
EVAL_ROBOTDOG_CAMERA_MODE="${EVAL_ROBOTDOG_CAMERA_MODE:-fixed}"
EVAL_ROBOTDOG_FOV="${EVAL_ROBOTDOG_FOV:-95.0}"
EVAL_MAX_SELF_VISIBLE_RATIO="${EVAL_MAX_SELF_VISIBLE_RATIO:-0.015}"

# Drone range widened slightly around the multi-agent training distribution.
EVAL_DRONE_SUCCESS_DISTANCE="${EVAL_DRONE_SUCCESS_DISTANCE:-6.5}"
EVAL_DRONE_LOST_DISTANCE="${EVAL_DRONE_LOST_DISTANCE:-8.0}"
EVAL_DRONE_IDEAL_FOLLOW_DIST="${EVAL_DRONE_IDEAL_FOLLOW_DIST:-4.5}"
EVAL_DRONE_MIN_FOLLOW_DIST="${EVAL_DRONE_MIN_FOLLOW_DIST:-2.5}"
EVAL_DRONE_MAX_FOLLOW_DIST="${EVAL_DRONE_MAX_FOLLOW_DIST:-6.5}"
EVAL_DRONE_HEIGHT="${EVAL_DRONE_HEIGHT:-600}"
EVAL_DRONE_VX_SCALE="${EVAL_DRONE_VX_SCALE:-0.12}"
EVAL_DRONE_VY_SCALE="${EVAL_DRONE_VY_SCALE:-0.1}"
EVAL_DRONE_YAW_SIGN="${EVAL_DRONE_YAW_SIGN:-1.0}"
EVAL_DRONE_YAW_SCALE="${EVAL_DRONE_YAW_SCALE:-3.0}"
EVAL_DRONE_MAX_SPEED="${EVAL_DRONE_MAX_SPEED:-0.12}"
EVAL_DRONE_MAX_VX="${EVAL_DRONE_MAX_VX:-0.12}"
EVAL_DRONE_MAX_VY="${EVAL_DRONE_MAX_VY:-0.05}"
# Preserve the model's predicted drone yaw while bounding it to a safe rate.
EVAL_DRONE_MAX_YAW_RATE="${EVAL_DRONE_MAX_YAW_RATE:-1.0}"
EVAL_DRONE_CAMERA_FIXED_PITCH="${EVAL_DRONE_CAMERA_FIXED_PITCH:--60}"
EVAL_DRONE_CAMERA_PITCHES="${EVAL_DRONE_CAMERA_PITCHES:--60}"
EVAL_DRONE_CAMERA_FIXED_YAW="${EVAL_DRONE_CAMERA_FIXED_YAW:-0}"
EVAL_DRONE_CAMERA_YAW_OFFSETS="${EVAL_DRONE_CAMERA_YAW_OFFSETS:-0}"
EVAL_DRONE_CAMERA_MODE="${EVAL_DRONE_CAMERA_MODE:-fixed}"
EVAL_LOCK_DRONE_CAMERA_WORLD_XY="${EVAL_LOCK_DRONE_CAMERA_WORLD_XY:-1}"
EVAL_DRONE_CAMERA_Z_OFFSET="${EVAL_DRONE_CAMERA_Z_OFFSET:-0}"
EVAL_DRONE_FOV="${EVAL_DRONE_FOV:-100}"

bool_cli_arg() {
    local name="$1"
    local value="$2"
    case "${value}" in
        1|true|TRUE|yes|YES|on|ON) printf -- "--%s\n" "${name}" ;;
        *) printf -- "--no-%s\n" "${name}" ;;
    esac
}

resolve_cuda_gpu() {
    local gpu="$1"
    local cuda_gpu="${gpu}"
    if [[ "${gpu}" =~ ^[0-9]+$ ]] && command -v nvidia-smi >/dev/null 2>&1; then
        cuda_gpu="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v idx="${gpu}" '$1 == idx {print $2; exit}')"
        cuda_gpu="${cuda_gpu:-${gpu}}"
    fi
    printf '%s\n' "${cuda_gpu}"
}

build_eval_plan() {
    "${PYTHON_BIN}" - "${TEST_TARGET_MANIFEST}" "${EVAL_SCENES}" "${EVAL_EPISODES}" <<'PY'
import json
import sys

manifest_path, requested_scenes, requested_episodes = sys.argv[1:4]
with open(manifest_path, "r", encoding="utf-8") as handle:
    manifest = json.load(handle)

scene_counts = manifest.get("scene_counts", {})
if requested_scenes.strip():
    scenes = [item.strip() for item in requested_scenes.split(",") if item.strip()]
else:
    # dict order follows split_manifest.json, allowing expensive scenes to be
    # intentionally placed last. EVAL_SCENES still overrides this order.
    scenes = list(scene_counts)

for scene in scenes:
    if requested_episodes == "manifest":
        try:
            episodes = int(scene_counts[scene]["test"])
        except KeyError as exc:
            raise SystemExit(f"scene not found in manifest: {scene}") from exc
    else:
        episodes = int(requested_episodes)
    if episodes <= 0:
        raise SystemExit(f"episodes must be positive for {scene}: {episodes}")
    print(f"{scene}\t{episodes}")
PY
}

run_eval_scene() {
    local scene="$1"
    local episodes="$2"
    local gpu="$3"
    local render_gpu="$4"
    local cuda_gpu
    cuda_gpu="$(resolve_cuda_gpu "${gpu}")"
    local save_path="${EVAL_ROOT}/${scene}"
    mkdir -p "${save_path}"

    local eval_args=(
        --ckpt "${CKPT_DIR}"
        --save-path "${save_path}"
        --env-id "${scene}"
        --episodes "${episodes}"
        --recorded-target-dir "${TEST_TARGET_MANIFEST}"
        --max-steps "${EVAL_MAX_STEPS}"
        --max-lost-steps "${EVAL_MAX_LOST_STEPS}"
        --max-failure-steps "${EVAL_MAX_FAILURE_STEPS}"
        --failure-warmup-steps "${EVAL_FAILURE_WARMUP_STEPS}"
        --max-episode-seconds "${EVAL_MAX_EPISODE_SECONDS}"
        --seed "${EVAL_SEED}"
        --render-gpu "${render_gpu}"
        --instruction "${EVAL_INSTRUCTION}"
        --joint-instruction "${EVAL_JOINT_INSTRUCTION}"
        --width "${EVAL_WIDTH}"
        --height "${EVAL_HEIGHT}"
        --fps "${EVAL_FPS}"
        --dt "${EVAL_DT}"
        --history-frame-dt "${EVAL_HISTORY_FRAME_DT}"
        "$(bool_cli_arg deterministic-step "${EVAL_DETERMINISTIC_STEP}")"
        "$(bool_cli_arg realtime-waypoint-timing "${EVAL_REALTIME_WAYPOINT_TIMING}")"
        --realtime-waypoint-min-seconds "${EVAL_REALTIME_WAYPOINT_MIN_SECONDS}"
        --realtime-waypoint-max-seconds "${EVAL_REALTIME_WAYPOINT_MAX_SECONDS}"
        "$(bool_cli_arg offscreen "${EVAL_OFFSCREEN}")"
        "--time-dilation=${EVAL_TIME_DILATION}"
        "$(bool_cli_arg disable-ue-input "${EVAL_DISABLE_UE_INPUT}")"
        --launch-retries "${EVAL_LAUNCH_RETRIES}"
        "$(bool_cli_arg save-video "${SAVE_EVAL_VIDEO}")"
        "$(bool_cli_arg write-global-video "${EVAL_WRITE_GLOBAL_VIDEO}")"
        "$(bool_cli_arg trajectory-overlay "${EVAL_TRAJECTORY_OVERLAY}")"
        --trajectory-scale "${EVAL_TRAJECTORY_SCALE}"
        --planner-debug-steps "${EVAL_PLANNER_DEBUG_STEPS}"
        --bbox-source "${EVAL_BBOX_SOURCE}"
        --waypoint-index "${EVAL_WAYPOINT_INDEX}"
        --drone-waypoint-index "${EVAL_DRONE_WAYPOINT_INDEX}"
        --robotdog-waypoint-index "${EVAL_ROBOTDOG_WAYPOINT_INDEX}"
        --waypoint-horizon-steps "${EVAL_WAYPOINT_HORIZON_STEPS}"
        --target-replay-mode "${EVAL_TARGET_REPLAY_MODE}"
        --target-path-min-spacing "${EVAL_TARGET_PATH_MIN_SPACING}"
        --target-path-reach-distance "${EVAL_TARGET_PATH_REACH_DISTANCE}"
        --target-goal-reach-distance "${EVAL_TARGET_GOAL_REACH_DISTANCE}"
        --target-stop-wait-min-steps "${EVAL_TARGET_STOP_WAIT_MIN_STEPS}"
        --target-stop-wait-max-steps "${EVAL_TARGET_STOP_WAIT_MAX_STEPS}"
        --human-speed "${EVAL_HUMAN_SPEED}"
        --human-goal-min-distance "${EVAL_HUMAN_GOAL_MIN_DISTANCE}"
        --human-goal-max-distance "${EVAL_HUMAN_GOAL_MAX_DISTANCE}"
        "$(bool_cli_arg open-spawn "${EVAL_OPEN_SPAWN}")"
        --open-spawn-radius "${EVAL_OPEN_SPAWN_RADIUS}"
        --min-open-clearance "${EVAL_MIN_OPEN_CLEARANCE}"
        --open-spawn-candidates "${EVAL_OPEN_SPAWN_CANDIDATES}"
        --ground-navmesh-tolerance "${EVAL_GROUND_NAVMESH_TOLERANCE}"
        --drone-navmesh-tolerance "${EVAL_DRONE_NAVMESH_TOLERANCE}"
        "$(bool_cli_arg require-visual-target "${EVAL_REQUIRE_VISUAL_TARGET}")"
        "$(bool_cli_arg require-centered-target "${EVAL_REQUIRE_CENTERED_TARGET}")"
        "$(bool_cli_arg use-mask-visibility "${EVAL_USE_MASK_VISIBILITY}")"
        --min-visible-ratio "${EVAL_MIN_VISIBLE_RATIO}"
        --target-center-tolerance "${EVAL_TARGET_CENTER_TOLERANCE}"
        --human-appearance-min "${EVAL_HUMAN_APPEARANCE_MIN}"
        --human-appearance-max "${EVAL_HUMAN_APPEARANCE_MAX}"
        --robotdog-appearance-min "${EVAL_ROBOTDOG_APPEARANCE_MIN}"
        --robotdog-appearance-max "${EVAL_ROBOTDOG_APPEARANCE_MAX}"
        --robotdog-yaw-sign "${EVAL_ROBOTDOG_YAW_SIGN}"
        --robotdog-yaw-scale "${EVAL_ROBOTDOG_YAW_SCALE}"
        --robotdog-speed-gain "${EVAL_ROBOTDOG_SPEED_GAIN}"
        --robotdog-max-speed "${EVAL_ROBOTDOG_MAX_SPEED}"
        --robotdog-max-turn-deg "${EVAL_ROBOTDOG_MAX_TURN_DEG}"
        --robotdog-max-lateral-speed "${EVAL_ROBOTDOG_MAX_LATERAL_SPEED}"
        --robotdog-max-yaw-rate "${EVAL_ROBOTDOG_MAX_YAW_RATE}"
        --robotdog-success-distance "${EVAL_ROBOTDOG_SUCCESS_DISTANCE}"
        --robotdog-lost-distance "${EVAL_ROBOTDOG_LOST_DISTANCE}"
        --robotdog-ideal-follow-dist "${EVAL_ROBOTDOG_IDEAL_FOLLOW_DIST}"
        --robotdog-min-follow-dist "${EVAL_ROBOTDOG_MIN_FOLLOW_DIST}"
        --robotdog-max-follow-dist "${EVAL_ROBOTDOG_MAX_FOLLOW_DIST}"
        --robotdog-camera-forward "${EVAL_ROBOTDOG_CAMERA_FORWARD}"
        --robotdog-camera-lateral "${EVAL_ROBOTDOG_CAMERA_LATERAL}"
        --robotdog-camera-height "${EVAL_ROBOTDOG_CAMERA_HEIGHT}"
        --robotdog-camera-mounts "${EVAL_ROBOTDOG_CAMERA_MOUNTS}"
        "--robotdog-camera-pitches=${EVAL_ROBOTDOG_CAMERA_PITCHES}"
        --robotdog-camera-yaw-offsets "${EVAL_ROBOTDOG_CAMERA_YAW_OFFSETS}"
        --robotdog-camera-mode "${EVAL_ROBOTDOG_CAMERA_MODE}"
        --robotdog-fov "${EVAL_ROBOTDOG_FOV}"
        --max-self-visible-ratio "${EVAL_MAX_SELF_VISIBLE_RATIO}"
        --drone-vx-scale "${EVAL_DRONE_VX_SCALE}"
        --drone-vy-scale "${EVAL_DRONE_VY_SCALE}"
        --drone-yaw-sign "${EVAL_DRONE_YAW_SIGN}"
        --drone-yaw-scale "${EVAL_DRONE_YAW_SCALE}"
        --drone-max-speed "${EVAL_DRONE_MAX_SPEED}"
        --drone-max-vx "${EVAL_DRONE_MAX_VX}"
        --drone-max-vy "${EVAL_DRONE_MAX_VY}"
        --drone-max-yaw-rate "${EVAL_DRONE_MAX_YAW_RATE}"
        --drone-success-distance "${EVAL_DRONE_SUCCESS_DISTANCE}"
        --drone-lost-distance "${EVAL_DRONE_LOST_DISTANCE}"
        --drone-ideal-follow-dist "${EVAL_DRONE_IDEAL_FOLLOW_DIST}"
        --drone-min-follow-dist "${EVAL_DRONE_MIN_FOLLOW_DIST}"
        --drone-max-follow-dist "${EVAL_DRONE_MAX_FOLLOW_DIST}"
        --drone-height "${EVAL_DRONE_HEIGHT}"
        "--drone-camera-fixed-pitch=${EVAL_DRONE_CAMERA_FIXED_PITCH}"
        "--drone-camera-pitches=${EVAL_DRONE_CAMERA_PITCHES}"
        --drone-camera-fixed-yaw "${EVAL_DRONE_CAMERA_FIXED_YAW}"
        --drone-camera-yaw-offsets "${EVAL_DRONE_CAMERA_YAW_OFFSETS}"
        --drone-camera-mode "${EVAL_DRONE_CAMERA_MODE}"
        "$(bool_cli_arg lock-drone-camera-world-xy "${EVAL_LOCK_DRONE_CAMERA_WORLD_XY}")"
        --drone-camera-z-offset "${EVAL_DRONE_CAMERA_Z_OFFSET}"
        --drone-fov "${EVAL_DRONE_FOV}"
        --max-camera-search-candidates "${EVAL_MAX_CAMERA_SEARCH_CANDIDATES}"
        "$(bool_cli_arg snap-heading "${EVAL_SNAP_HEADING}")"
        "$(bool_cli_arg follow-behind "${EVAL_FOLLOW_BEHIND}")"
        "$(bool_cli_arg init-from-recorded-agent-poses "${EVAL_INIT_FROM_RECORDED_AGENT_POSES}")"
        --success-rate-threshold "${EVAL_SUCCESS_RATE_THRESHOLD}"
        --min-success-steps "${EVAL_MIN_SUCCESS_STEPS}"
    )

    if [[ -n "${EVAL_ROBOTDOG_CAMERA_FIXED_PITCH}" ]]; then
        eval_args+=("--robotdog-camera-fixed-pitch=${EVAL_ROBOTDOG_CAMERA_FIXED_PITCH}")
    fi
    if [[ -n "${EVAL_AGENT1_INSTRUCTION}" ]]; then
        eval_args+=(--agent1-instruction "${EVAL_AGENT1_INSTRUCTION}")
    fi
    if [[ -n "${EVAL_AGENT2_INSTRUCTION}" ]]; then
        eval_args+=(--agent2-instruction "${EVAL_AGENT2_INSTRUCTION}")
    fi
    if [[ -n "${EVAL_TOP_VIEW_HEIGHT}" ]]; then
        eval_args+=(--top-view-height "${EVAL_TOP_VIEW_HEIGHT}")
    fi
    if [[ "${EVAL_FACE_TARGET_BEFORE_STEP}" == "1" ]]; then
        eval_args+=(--face-target-before-step)
    fi
    if [[ -n "${EVAL_EXTRA_ARGS}" ]]; then
        local extra_eval_args=()
        read -r -a extra_eval_args <<< "${EVAL_EXTRA_ARGS}"
        eval_args+=("${extra_eval_args[@]}")
    fi

    echo
    echo "[eval] scene=${scene} episodes=${episodes} gpu=${gpu} cuda=${cuda_gpu} render_gpu=${render_gpu}"
    CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}" \
    CUDA_VISIBLE_DEVICES="${cuda_gpu}" \
    UNREALZOO_FAST_ENV_ID="${scene}" \
    PYTHONPATH="${REPO_ROOT}/unrealzoo-gym:${PYTHONPATH:-}" \
    "${PYTHON_BIN}" -u "${EVAL_SCRIPT}" "${eval_args[@]}"

    "${PYTHON_BIN}" -m tools.calculate_unrealzoo_metrics --eval-dir "${save_path}"
}

if [[ ! -e "${CKPT_DIR}" ]]; then
    echo "[ERROR] checkpoint does not exist: ${CKPT_DIR}" >&2
    exit 1
fi
if [[ ! -f "${TEST_TARGET_MANIFEST}" ]]; then
    echo "[ERROR] test target manifest does not exist: ${TEST_TARGET_MANIFEST}" >&2
    exit 1
fi

echo "==============================================================================="
echo "Base-concat multi-agent UnrealZoo eval"
echo "==============================================================================="
echo "CKPT_DIR=${CKPT_DIR}"
echo "EVAL_SCRIPT=${EVAL_SCRIPT}"
echo "TEST_TARGET_MANIFEST=${TEST_TARGET_MANIFEST}"
echo "EVAL_ROOT=${EVAL_ROOT}"
echo "EVAL_SCENES=${EVAL_SCENES:-<manifest all scenes>}"
echo "EVAL_EPISODES=${EVAL_EPISODES}"
echo "EVAL_GPUS=${EVAL_GPUS}"
echo "RENDER_GPUS=${RENDER_GPUS}"
echo "WAYPOINT_INDEX shared=${EVAL_WAYPOINT_INDEX} drone=${EVAL_DRONE_WAYPOINT_INDEX} robotdog=${EVAL_ROBOTDOG_WAYPOINT_INDEX} horizon_steps=${EVAL_WAYPOINT_HORIZON_STEPS}"
echo "CLOCK deterministic=${EVAL_DETERMINISTIC_STEP} fixed_dt=${EVAL_DT}s"
echo "DRONE_YAW scale=${EVAL_DRONE_YAW_SCALE} sign=${EVAL_DRONE_YAW_SIGN} max_rate=${EVAL_DRONE_MAX_YAW_RATE}"
echo "PLANNER_DEBUG_STEPS=${EVAL_PLANNER_DEBUG_STEPS}"
echo "JOINT_INSTRUCTION=${EVAL_JOINT_INSTRUCTION}"
echo "AGENT1_INSTRUCTION=${EVAL_AGENT1_INSTRUCTION:-<default role marker>}"
echo "AGENT2_INSTRUCTION=${EVAL_AGENT2_INSTRUCTION:-<default role marker>}"
echo "TARGET_REPLAY=${EVAL_TARGET_REPLAY_MODE} spacing=${EVAL_TARGET_PATH_MIN_SPACING} reach=${EVAL_TARGET_PATH_REACH_DISTANCE}"
echo "TARGET_STOP goal_reach=${EVAL_TARGET_GOAL_REACH_DISTANCE} wait_steps=${EVAL_TARGET_STOP_WAIT_MIN_STEPS}-${EVAL_TARGET_STOP_WAIT_MAX_STEPS}"
echo "INIT_FROM_RECORDED_AGENT_POSES=${EVAL_INIT_FROM_RECORDED_AGENT_POSES}"
echo "RobotDog success/lost=${EVAL_ROBOTDOG_SUCCESS_DISTANCE}/${EVAL_ROBOTDOG_LOST_DISTANCE}"
echo "Drone success/lost=${EVAL_DRONE_SUCCESS_DISTANCE}/${EVAL_DRONE_LOST_DISTANCE}"
echo "==============================================================================="

mkdir -p "${EVAL_ROOT}"
IFS=',' read -r -a EVAL_GPU_LIST <<< "${EVAL_GPUS}"
IFS=',' read -r -a RENDER_GPU_LIST <<< "${RENDER_GPUS}"
if [[ "${#EVAL_GPU_LIST[@]}" -eq 0 || -z "${EVAL_GPU_LIST[0]}" ]]; then
    echo "[ERROR] EVAL_GPUS is empty." >&2
    exit 1
fi
if [[ "${#RENDER_GPU_LIST[@]}" -eq 0 || -z "${RENDER_GPU_LIST[0]}" ]]; then
    echo "[ERROR] RENDER_GPUS is empty." >&2
    exit 1
fi

idx=0
while IFS=$'\t' read -r scene episodes; do
    gpu="${EVAL_GPU_LIST[$((idx % ${#EVAL_GPU_LIST[@]}))]}"
    render_gpu="${RENDER_GPU_LIST[$((idx % ${#RENDER_GPU_LIST[@]}))]}"
    run_eval_scene "${scene}" "${episodes}" "${gpu}" "${render_gpu}"
    idx=$((idx + 1))
done < <(build_eval_plan)

echo
echo "[summary] aggregate metrics under ${EVAL_ROOT}"
"${PYTHON_BIN}" -m tools.calculate_unrealzoo_metrics --eval-dir "${EVAL_ROOT}"
echo "[done] eval finished: ${EVAL_ROOT}"
