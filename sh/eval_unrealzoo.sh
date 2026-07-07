#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# UnrealZoo 双 Agent 闭环评估
# -----------------------------------------------------------------------------
# 该脚本只负责 UnrealZoo。原 Habitat / EVT-Bench 评估仍使用：
#   bash sh/eval.sh
#
# 最常需要修改的参数：
#   CKPT       双 Agent checkpoint 文件或目录
#   GPU        使用的物理 GPU 编号
#   ENV_ID     UnrealZoo 场景
#   EPISODES   评估 episode 数量
#   MAX_STEPS  每条 episode 最大步数
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/hdt/miniconda3/envs/omtracknew/bin/python}"
[[ -x "${PYTHON_BIN}" ]] || PYTHON_BIN="python"

CKPT="${CKPT:-/data/hdt/ntv_data/ckpt/New_paths_training_multi_agent_mlp_10ep}"
# GPU 控制 PyTorch 模型推理卡；RENDER_GPU 控制 Unreal Engine Vulkan 渲染卡。
# Unreal Engine 不遵循 CUDA_VISIBLE_DEVICES，因此必须额外传 -graphicsadapter。
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
GPU="${GPU:-0}"
RENDER_GPU="${RENDER_GPU:-${GPU}}"
CUDA_GPU="${CUDA_GPU:-}"
if [ -z "${CUDA_GPU}" ]; then
    if [[ "${GPU}" =~ ^[0-9]+$ ]] && command -v nvidia-smi >/dev/null 2>&1; then
        CUDA_GPU="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v idx="${GPU}" '$1 == idx {print $2; exit}')"
    fi
    CUDA_GPU="${CUDA_GPU:-${GPU}}"
fi
ENV_ID="${ENV_ID:-UnrealTrack-DowntownWest-ContinuousColor-v0}"
EPISODES="${EPISODES:-10}"
MAX_STEPS="${MAX_STEPS:-600}"
# 连续不可见或距离过远多少步后提前结束。
MAX_LOST_STEPS="${MAX_LOST_STEPS:-20}"
# 连续多少步未满足“双 Agent 同时跟踪成功”后判定持续失败。
# 设为 0 可关闭；前 FAILURE_WARMUP_STEPS 步不会触发该规则。
MAX_FAILURE_STEPS="${MAX_FAILURE_STEPS:-50}"
FAILURE_WARMUP_STEPS="${FAILURE_WARMUP_STEPS:-20}"
# 单 episode 最长墙钟时间（秒），避免仿真或推理异常时长期占卡；0 表示关闭。
MAX_EPISODE_SECONDS="${MAX_EPISODE_SECONDS:-600}"
SEED="${SEED:-100}"
SAVE_PATH="${SAVE_PATH:-/data/hdt/ntv_data/sim_data/eval/unrealzoo_multi_agent}"

# 可选混合闭环模式：只从采集 JSON 重放人的 target_pose。
# Drone/RobotDog 动作、RGB、bbox 与后续观测仍由在线 UnrealZoo 闭环产生。
RECORDED_TARGET_DIR="${RECORDED_TARGET_DIR:-}"
RECORDED_TARGET_EPISODES="${RECORDED_TARGET_EPISODES:-}"
# 推荐入口：读取 2:1 split_manifest.json 中的 test episode，并按 ENV_ID 自动筛选。
TEST_TARGET_MANIFEST="${TEST_TARGET_MANIFEST:-}"
if [ -n "${TEST_TARGET_MANIFEST}" ]; then
    RECORDED_TARGET_DIR="${TEST_TARGET_MANIFEST}"
fi

# 模型路点转动作的时间间隔。应与 tools.make_tracking_data --multi_agent 的 --dt 一致。
DT="${DT:-0.1}"

# 参考单独机器狗/无人机评估：机器狗使用第 5 个未来点，无人机使用第 9 个未来点。
# WAYPOINT_INDEX 保留为旧配置兜底；显式的两个索引优先。
WAYPOINT_INDEX="${WAYPOINT_INDEX:-5}"
DRONE_WAYPOINT_INDEX="${DRONE_WAYPOINT_INDEX:-9}"
ROBOTDOG_WAYPOINT_INDEX="${ROBOTDOG_WAYPOINT_INDEX:-5}"

# 参考当前单 Agent 评估的人速设置；各 Agent 自身限制单独控制。
HUMAN_SPEED="${HUMAN_SPEED:-0.9}"  # m/s; allowed: 0.9, 1.0, 1.1, 1.2
agent_speed_for_human() {
    case "$1" in
        0.9|.9) echo "1.20" ;;
        1.0|1) echo "1.35" ;;
        1.1) echo "1.50" ;;
        1.2) echo "1.60" ;;
        *) echo "Unsupported HUMAN_SPEED=$1; choose 0.9, 1.0, 1.1, or 1.2 m/s" >&2; exit 2 ;;
    esac
}
agent_lateral_speed_for_human() {
    case "$1" in
        0.9|.9) echo "0.60" ;;
        1.0|1) echo "0.675" ;;
        1.1) echo "0.75" ;;
        1.2) echo "0.80" ;;
        *) echo "Unsupported HUMAN_SPEED=$1; choose 0.9, 1.0, 1.1, or 1.2 m/s" >&2; exit 2 ;;
    esac
}
DEFAULT_AGENT_MAX_SPEED="$(agent_speed_for_human "${HUMAN_SPEED}")"
DEFAULT_AGENT_LATERAL_SPEED="$(agent_lateral_speed_for_human "${HUMAN_SPEED}")"
ROBOTDOG_MAX_SPEED="${ROBOTDOG_MAX_SPEED:-${DEFAULT_AGENT_MAX_SPEED}}"
ROBOTDOG_MAX_TURN_DEG="${ROBOTDOG_MAX_TURN_DEG:-30}"
ROBOTDOG_SUCCESS_DISTANCE="${ROBOTDOG_SUCCESS_DISTANCE:-8.0}"
ROBOTDOG_LOST_DISTANCE="${ROBOTDOG_LOST_DISTANCE:-8.0}"
ROBOTDOG_YAW_SIGN="${ROBOTDOG_YAW_SIGN:-1.0}"

DRONE_MAX_VX="${DRONE_MAX_VX:-${DEFAULT_AGENT_MAX_SPEED}}"
DRONE_MAX_VY="${DRONE_MAX_VY:-${DEFAULT_AGENT_LATERAL_SPEED}}"
DRONE_MAX_YAW_RATE="${DRONE_MAX_YAW_RATE:-0.4}"
DRONE_VX_SCALE="${DRONE_VX_SCALE:-0.12}"
DRONE_VY_SCALE="${DRONE_VY_SCALE:-0.1}"
DRONE_YAW_SIGN="${DRONE_YAW_SIGN:-1.0}"
DRONE_SUCCESS_DISTANCE="${DRONE_SUCCESS_DISTANCE:-5.5}"
DRONE_LOST_DISTANCE="${DRONE_LOST_DISTANCE:-5.5}"

# 成功规则：联合跟踪率达到阈值、无碰撞且至少运行 MIN_SUCCESS_STEPS。
SUCCESS_RATE_THRESHOLD="${SUCCESS_RATE_THRESHOLD:-0.8}"
MIN_CENTERED_RATE="${MIN_CENTERED_RATE:-0.8}"
MIN_SUCCESS_STEPS="${MIN_SUCCESS_STEPS:-20}"

# 每步在线执行 DINO + SigLIP；关闭视频可以减少编码之外的磁盘开销。
SAVE_VIDEO="${SAVE_VIDEO:-1}"
WRITE_GLOBAL_VIDEO="${WRITE_GLOBAL_VIDEO:-1}"

# model：首帧无框检测，后续仅使用模型上一帧预测框；不会把真值框输入模型。
# ground_truth：使用 UnrealZoo object mask 真值框；none：每一帧都不使用 bbox 先验。
BBOX_SOURCE="${BBOX_SOURCE:-model}"
# 固定扩散噪声，保证相同 checkpoint/seed 的评估结果可复现。
DIFFUSION_DETERMINISTIC_INFERENCE="${DIFFUSION_DETERMINISTIC_INFERENCE:-1}"

# 在保存的无人机/机器狗 RGB 视频中叠加局部预测轨迹。
TRAJECTORY_OVERLAY="${TRAJECTORY_OVERLAY:-1}"
TRAJECTORY_SCALE="${TRAJECTORY_SCALE:-120}"

# 默认不使用目标真值强制旋转 Agent；设为 1 可用于排查仅由朝向导致的问题。
FACE_TARGET_BEFORE_STEP="${FACE_TARGET_BEFORE_STEP:-0}"

# 本地视觉权重路径。
export DINOV3_MODEL_PATH="${DINOV3_MODEL_PATH:-/data/hdt/ntv_data/weights/dinov3}"

bool_flag() {
    local value="$1"
    local positive="$2"
    local negative="$3"
    if [ "${value}" = "1" ]; then
        printf '%s' "${positive}"
    else
        printf '%s' "${negative}"
    fi
}

SAVE_VIDEO_FLAG="$(bool_flag "${SAVE_VIDEO}" --save-video --no-save-video)"
GLOBAL_VIDEO_FLAG="$(bool_flag "${WRITE_GLOBAL_VIDEO}" --write-global-video --no-write-global-video)"
TRAJECTORY_FLAG="$(bool_flag "${TRAJECTORY_OVERLAY}" --trajectory-overlay --no-trajectory-overlay)"
DIFFUSION_FLAG="$(bool_flag "${DIFFUSION_DETERMINISTIC_INFERENCE}" --diffusion-deterministic-inference --no-diffusion-deterministic-inference)"

extra_args=()
if [ "${FACE_TARGET_BEFORE_STEP}" = "1" ]; then
    extra_args+=(--face-target-before-step)
fi
if [ -n "${RECORDED_TARGET_DIR}" ]; then
    extra_args+=(--recorded-target-dir "${RECORDED_TARGET_DIR}")
fi
if [ -n "${RECORDED_TARGET_EPISODES}" ]; then
    extra_args+=(--recorded-target-episodes "${RECORDED_TARGET_EPISODES}")
fi

echo "=============================================="
echo "OpenTrackVLA UnrealZoo multi-agent eval"
echo "=============================================="
echo "Checkpoint:       ${CKPT}"
echo "Model GPU:        ${GPU}"
echo "CUDA visible:     ${CUDA_GPU}"
echo "Render GPU:       ${RENDER_GPU}"
echo "Environment:      ${ENV_ID}"
echo "Episodes:         ${EPISODES}"
echo "Max steps:        ${MAX_STEPS}"
echo "Max lost steps:   ${MAX_LOST_STEPS}"
echo "Max failure steps:${MAX_FAILURE_STEPS} (warmup=${FAILURE_WARMUP_STEPS})"
echo "Episode timeout:  ${MAX_EPISODE_SECONDS}s"
echo "Save path:        ${SAVE_PATH}"
echo "Waypoint index:   shared=${WAYPOINT_INDEX} drone=${DRONE_WAYPOINT_INDEX} robotdog=${ROBOTDOG_WAYPOINT_INDEX}"
echo "DT:               ${DT}"
echo "Human speed:      ${HUMAN_SPEED} m/s"
echo "RobotDog:         max_speed=${ROBOTDOG_MAX_SPEED}m/s max_turn=${ROBOTDOG_MAX_TURN_DEG}deg success=${ROBOTDOG_SUCCESS_DISTANCE}m lost=${ROBOTDOG_LOST_DISTANCE}m yaw_sign=${ROBOTDOG_YAW_SIGN}"
echo "Drone:            max_vx=${DRONE_MAX_VX} max_vy=${DRONE_MAX_VY} max_yaw_rate=${DRONE_MAX_YAW_RATE} vx_scale=${DRONE_VX_SCALE} vy_scale=${DRONE_VY_SCALE} yaw_sign=${DRONE_YAW_SIGN} success=${DRONE_SUCCESS_DISTANCE}m lost=${DRONE_LOST_DISTANCE}m"
echo "BBox source:      ${BBOX_SOURCE}"
echo "Deterministic:    ${DIFFUSION_DETERMINISTIC_INFERENCE}"
echo "Human motion:     $([ -n "${RECORDED_TARGET_DIR}" ] && printf 'recorded target_pose' || printf 'simulator navigation')"
if [ -n "${RECORDED_TARGET_DIR}" ]; then
    echo "Human traj source:${RECORDED_TARGET_DIR}"
    echo "Human episodes:   ${RECORDED_TARGET_EPISODES:-all}"
fi
echo "Trajectory:       ${TRAJECTORY_OVERLAY} (scale=${TRAJECTORY_SCALE})"
echo "=============================================="

CUDA_VISIBLE_DEVICES="${CUDA_GPU}" \
UNREALZOO_FAST_ENV_ID="${ENV_ID}" \
PYTHONPATH="${REPO_ROOT}/unrealzoo-gym:${PYTHONPATH:-}" \
"${PYTHON_BIN}" -u eval_unrealzoo_multi_agent.py \
    --ckpt "${CKPT}" \
    --save-path "${SAVE_PATH}" \
    --env-id "${ENV_ID}" \
    --episodes "${EPISODES}" \
    --max-steps "${MAX_STEPS}" \
    --max-lost-steps "${MAX_LOST_STEPS}" \
    --max-failure-steps "${MAX_FAILURE_STEPS}" \
    --failure-warmup-steps "${FAILURE_WARMUP_STEPS}" \
    --max-episode-seconds "${MAX_EPISODE_SECONDS}" \
    --seed "${SEED}" \
    --render-gpu "${RENDER_GPU}" \
    --dt "${DT}" \
    --waypoint-index "${WAYPOINT_INDEX}" \
    --drone-waypoint-index "${DRONE_WAYPOINT_INDEX}" \
    --robotdog-waypoint-index "${ROBOTDOG_WAYPOINT_INDEX}" \
    --human-speed "${HUMAN_SPEED}" \
    --robotdog-max-speed "${ROBOTDOG_MAX_SPEED}" \
    --robotdog-max-turn-deg "${ROBOTDOG_MAX_TURN_DEG}" \
    --robotdog-success-distance "${ROBOTDOG_SUCCESS_DISTANCE}" \
    --robotdog-lost-distance "${ROBOTDOG_LOST_DISTANCE}" \
    --robotdog-yaw-sign "${ROBOTDOG_YAW_SIGN}" \
    --drone-max-vx "${DRONE_MAX_VX}" \
    --drone-max-vy "${DRONE_MAX_VY}" \
    --drone-max-yaw-rate "${DRONE_MAX_YAW_RATE}" \
    --drone-vx-scale "${DRONE_VX_SCALE}" \
    --drone-vy-scale "${DRONE_VY_SCALE}" \
    --drone-yaw-sign "${DRONE_YAW_SIGN}" \
    --drone-success-distance "${DRONE_SUCCESS_DISTANCE}" \
    --drone-lost-distance "${DRONE_LOST_DISTANCE}" \
    --bbox-source "${BBOX_SOURCE}" \
    --trajectory-scale "${TRAJECTORY_SCALE}" \
    --success-rate-threshold "${SUCCESS_RATE_THRESHOLD}" \
    --min-centered-rate "${MIN_CENTERED_RATE}" \
    --min-success-steps "${MIN_SUCCESS_STEPS}" \
    "${SAVE_VIDEO_FLAG}" \
    "${GLOBAL_VIDEO_FLAG}" \
    "${TRAJECTORY_FLAG}" \
    "${DIFFUSION_FLAG}" \
    "${extra_args[@]}"

"${PYTHON_BIN}" -m tools.calculate_unrealzoo_metrics --eval-dir "${SAVE_PATH}"
