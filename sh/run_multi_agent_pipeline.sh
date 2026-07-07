#!/usr/bin/env bash
# -*- coding: utf-8 -*-
#
# 双 Agent TrackVLA 一键流程脚本。
#
# 功能：
# 1. 将已配对的 UnrealZoo 双 Agent episode 按每个地图 10:1 划分 train/test。
# 2. 从 train_raw 生成训练 JSONL。
# 3. 为 JSONL 中引用的图片预缓存视觉 token。
# 4. 训练前 dry_run 检查数据/cache/shape。
# 5. 启动双 Agent 模型训练。
# 6. 使用 test split 轨迹启动 UnrealZoo 闭环评估。
#
# 使用方式：
#   CUDA_VISIBLE_DEVICES=6,7 RUN_PRECACHE=1 RUN_DRY_RUN=1 RUN_TRAIN=1 RUN_EVAL=1 bash sh/run_multi_agent_pipeline.sh
#
# 新数据双 Agent MLP 完整流程，整理完数据后从视觉缓存开始：
#
#   # split 已存在时：预缓存 -> dry_run -> 训练 -> test split 闭环评估，使用物理 6/7 卡。
#   CUDA_VISIBLE_DEVICES=6,7 \
#   RUN_MAKE_DATA=0 RUN_PRECACHE=1 RUN_DRY_RUN=1 RUN_TRAIN=1 RUN_EVAL=1 \
#   bash sh/run_multi_agent_pipeline.sh
#
#   # 需要重新 10:1 划分时，打开 RUN_SPLIT=1。默认检测到已有 split 会复用。
#   CUDA_VISIBLE_DEVICES=6,7 \
#   RUN_SPLIT=1 RUN_MAKE_DATA=1 RUN_PRECACHE=1 RUN_DRY_RUN=1 RUN_TRAIN=1 RUN_EVAL=1 \
#   bash sh/run_multi_agent_pipeline.sh
#
# 只跑某些步骤：
#   RUN_SPLIT=1 RUN_MAKE_DATA=1 RUN_PRECACHE=0 RUN_DRY_RUN=1 RUN_TRAIN=0 RUN_EVAL=0 bash sh/run_multi_agent_pipeline.sh
#   CUDA_VISIBLE_DEVICES=6,7 RUN_MAKE_DATA=0 RUN_PRECACHE=0 RUN_DRY_RUN=0 RUN_TRAIN=0 RUN_EVAL=1 bash sh/run_multi_agent_pipeline.sh
#
# 注意：
# - 本脚本调用的是主目录文件中的内置双 Agent 分支，即：
#   python -m tools.make_tracking_data --multi_agent
#   python -m tools.precache_frames --multi_agent
#   train.py --multi_agent
# - 不加 --multi_agent 的单 Agent 能力仍然保留在原脚本中，本脚本只封装双 Agent 流程。
# - 本脚本使用 train.py/model.py 中的普通 PlannerHead3L MLP 路点头；
#   不使用 anchor diffusion 规划头。扩散版本在 sh/train_anchor_diffusion.sh。

# =============================================================================
# GPU 配置：统一设置单卡或多卡训练
# =============================================================================

# 默认只使用物理 GPU 1。多卡训练时改为逗号分隔的物理卡编号，例如 "0,1,2,3"。
# 可在执行命令前临时覆盖，例如：
#   CUDA_VISIBLE_DEVICES=0,1 bash sh/run_multi_agent_pipeline.sh
# PyTorch 进程内部会把这些可见卡重新编号为 cuda:0、cuda:1、……。
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

# 默认根据可见 GPU 数量自动决定训练进程数：
#   CUDA_VISIBLE_DEVICES=1       -> NUM_GPUS=1，使用普通 python 启动
#   CUDA_VISIBLE_DEVICES=0,1,2,3 -> NUM_GPUS=4，使用 torchrun 启动
# 也可以在执行命令前显式设置 NUM_GPUS 覆盖自动值。
IFS=',' read -r -a VISIBLE_GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${NUM_GPUS:-${#VISIBLE_GPU_LIST[@]}}"
unset VISIBLE_GPU_LIST

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# =============================================================================
# 0. 步骤开关：每一步都可以独立打开/关闭
# =============================================================================

# 是否重新做 10:1 train/test 划分。
# 已经有 split_manifest.json 时默认复用；需要强制重划分请先手动清理旧 SPLIT_ROOT。
RUN_SPLIT="${RUN_SPLIT:-0}"

# 是否从 UnrealZoo 原始数据重新生成 frames/jsonl/dataset.json。
# 已经生成过数据时，可以设为 0。
RUN_MAKE_DATA="${RUN_MAKE_DATA:-0}"

# 是否预缓存视觉 token。
# 如果 vision_cache 已经完整，可以设为 0。
RUN_PRECACHE="${RUN_PRECACHE:-0}"

# 是否先跑训练 dry_run。
# 强烈建议保持为 1：它不会加载 LLM 训练，只检查 JSONL、图片、cache 和张量 shape。
RUN_DRY_RUN="${RUN_DRY_RUN:-1}"

# 是否正式训练。
# 默认 0，避免误触发长时间训练；确认 dry_run 正常后改为 1。
RUN_TRAIN="${RUN_TRAIN:-0}"

# 是否闭环评估。
# 默认 0，避免误触发 Unreal Engine；需要评估时设为 1。
RUN_EVAL="${RUN_EVAL:-0}"

# =============================================================================
# 1. Python 与路径配置：最常修改
# =============================================================================

# Python 环境。
# 如果你使用 omtracknew 环境，保持默认即可；否则改成自己的 python 路径。
PYTHON_BIN="${PYTHON_BIN:-/home/hdt/miniconda3/envs/omtracknew/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python"
fi

# 已配对的 UnrealZoo 原始双 Agent 数据根目录。
# 里面应该能递归找到：
#   *_drone.mp4
#   *_drone_info.json
#   *_robotdog.mp4
#   *_robotdog_info.json
RAW_ROOT="${RAW_ROOT:-/data/hdt/ntv_data/sim_data/New_paths_training_multi_agent}"

# 每地图 10:1 split 输出目录。
SPLIT_ROOT="${SPLIT_ROOT:-/data/hdt/ntv_data/sim_data/New_paths_training_multi_agent_split_10to1}"
TRAIN_RAW="${TRAIN_RAW:-${SPLIT_ROOT}/train_raw}"
TEST_RAW="${TEST_RAW:-${SPLIT_ROOT}/test_raw}"

# 生成 JSONL 使用的原始数据根目录。默认使用 train_raw。
INPUT_ROOT="${INPUT_ROOT:-${TRAIN_RAW}}"

# 划分参数。
SPLIT_SEED="${SPLIT_SEED:-42}"
SPLIT_TRAIN_PARTS="${SPLIT_TRAIN_PARTS:-10}"
SPLIT_TEST_PARTS="${SPLIT_TEST_PARTS:-1}"
SPLIT_COPY="${SPLIT_COPY:-0}"
REUSE_EXISTING_SPLIT="${REUSE_EXISTING_SPLIT:-1}"

# 处理后的训练数据根目录。
# tools.make_tracking_data --multi_agent 会在这里生成：
#   frames/
#   jsonl/
#   dataset.json
DATA_ROOT="${DATA_ROOT:-/data/hdt/ntv_data/data/New_paths_training_multi_agent_10to1/train}"

# 训练读取的 JSONL 路径。
# 通常保持为 DATA_ROOT/jsonl；也可以改成某个单独 .jsonl 或 DATA_ROOT/dataset.json。
TRAIN_JSON="${TRAIN_JSON:-${DATA_ROOT}/jsonl}"

# 视觉 token 缓存目录。
# tools.precache_frames 会写入这里，train.py 会从这里读取。
CACHE_ROOT="${CACHE_ROOT:-${DATA_ROOT}/vision_cache}"

# checkpoint 输出目录。
OUT_DIR="${OUT_DIR:-/data/hdt/ntv_data/ckpt/New_paths_training_multi_agent_mlp_10ep}"

# =============================================================================
# 2. 数据处理参数：影响样本数量和标签质量
# =============================================================================

# 历史帧数量。
# 训练 shape 中 coarse_tokens 的长度为 HISTORY * 4。
# 默认 HISTORY=31 时，单 Agent coarse token 数为 31*4=124。
HISTORY="${HISTORY:-31}"

# 未来动作积分步数。
# horizon 越大，标签覆盖越远；但数据末尾可用样本会变少。
HORIZON="${HORIZON:-9}"

# 输出 waypoint 数量。
# 模型最终输出 shape 是 (B, 2, N_WAYPOINTS, 3)。
N_WAYPOINTS="${N_WAYPOINTS:-10}"

# 动作积分时间间隔。
# 如果采集频率变化，这里需要同步修改。
DT="${DT:-0.1}"

# 两个 Agent 的顺序。
# agent1/agent2 的顺序会影响模型输出 waypoints[:,0] 和 waypoints[:,1] 的含义。
AGENT1="${AGENT1:-drone}"
AGENT2="${AGENT2:-robotdog}"

# 优先读取的动作字段。
# 默认使用 base_velocity；Habitat 中它是专家命令，UnrealZoo robotdog 中它是实际执行速度。
ACTION_FIELD="${ACTION_FIELD:-auto}"

# 统一训练指令；留空时会优先使用 episode status.json 中的 instruction。
INSTRUCTION="${INSTRUCTION:-}"

# 质量过滤开关。
# ONLY_SUCCESS=1：只保留成功 episode。
# EXCLUDE_COLLISION=1：排除 episode 级碰撞。
# SKIP_COLLISION_STEPS=1：排除 step 级碰撞样本。
# REQUIRE_VISIBLE=1：要求两个 Agent 当前帧都能看到目标。
ONLY_SUCCESS="${ONLY_SUCCESS:-1}"
EXCLUDE_COLLISION="${EXCLUDE_COLLISION:-1}"
SKIP_COLLISION_STEPS="${SKIP_COLLISION_STEPS:-0}"
REQUIRE_VISIBLE="${REQUIRE_VISIBLE:-0}"

# 可见性/跟踪率/步数阈值。
# 数据太少时先保持 0；追求高质量数据时再逐步调高。
MIN_TARGET_VISIBILITY="${MIN_TARGET_VISIBILITY:-0.0}"
MIN_AGENT_FOLLOWING_RATE="${MIN_AGENT_FOLLOWING_RATE:-0.8}"
MIN_TOTAL_STEPS="${MIN_TOTAL_STEPS:-50}"

# 是否允许未来 horizon 不完整的样本。
# 0：丢弃 episode 末尾不足 horizon 的样本，标签更干净。
# 1：保留 partial horizon 样本，样本更多，但 valid_mask 会更稀疏。
ALLOW_PARTIAL_HORIZON="${ALLOW_PARTIAL_HORIZON:-0}"

# 是否复用已抽出的 frames。
# 已经抽过帧时保持 1，可以节省大量时间。
REUSE_EXISTING_FRAMES="${REUSE_EXISTING_FRAMES:-1}"

# 最多处理多少个 episode。
# 0 表示全部；调试时可设成 1 或 2。
MAX_EPISODES="${MAX_EPISODES:-0}"

# =============================================================================
# 3. 视觉缓存参数：影响预缓存速度和显存
# =============================================================================

# DINO/SigLIP 输入图像大小。
# 当前 cache_gridpool 默认使用 384。
IMAGE_SIZE="${IMAGE_SIZE:-384}"

# 预缓存 batch size。
# 显存不足就调小，例如 2 或 4。
PRECACHE_BATCH_SIZE="${PRECACHE_BATCH_SIZE:-8}"

# 预缓存设备。
# 可选：cuda / cpu / 空字符串。空字符串时由 tools.precache_frames 自动选择。
PRECACHE_DEVICE="${PRECACHE_DEVICE:-}"

# 预缓存分片数。
# 默认跟随可见 GPU 数量；CUDA_VISIBLE_DEVICES=6,7 时会启动两个进程分别跑 shard 0/1。
# 如需单进程缓存，可设 PRECACHE_NUM_SHARDS=1。
PRECACHE_NUM_SHARDS="${PRECACHE_NUM_SHARDS:-${NUM_GPUS}}"

# 是否只列出需要缓存的图片，不真正加载视觉模型。
# 调试数据路径时设为 1。
PRECACHE_LIST_ONLY="${PRECACHE_LIST_ONLY:-0}"

# 只预缓存前多少张图。
# 0 表示全部；调试时可设成 20。
PRECACHE_LIMIT="${PRECACHE_LIMIT:-0}"

# =============================================================================
# 4. 训练参数：最常修改
# =============================================================================

# LLM backbone。
# 换模型时，model.py 会自动读取 hidden_size，但显存和训练速度会变化很大。
LLM_NAME="${LLM_NAME:-Qwen/Qwen3-0.6B}"

# 训练轮数。
EPOCHS="${EPOCHS:-10}"

# 单卡 micro batch size。
# 显存不足先调小 BATCH_SIZE，再用 GRAD_ACCUM_STEPS 补有效 batch。
BATCH_SIZE="${BATCH_SIZE:-2}"

# 梯度累积步数。
# 有效 batch size = BATCH_SIZE * GPU数 * GRAD_ACCUM_STEPS。
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"

# 学习率。
# LLM 冻结时一般 2e-5 到 3e-4 都可试；解冻 LLM 时建议更小。
LR="${LR:-2e-5}"

# waypoint 主损失权重。
BETA_NAV="${BETA_NAV:-10}"

# bbox/visibility/relative-pose 辅助损失权重。
# 当前默认打开 grounding 监督，让空间头能参与后续动作规划。
BETA_BBOX="${BETA_BBOX:-1.0}"
BETA_VISIBLE="${BETA_VISIBLE:-0.5}"
# 监督 GND head 输出 agent-centric [dx_m, dy_m, dz_m, sin(d_yaw), cos(d_yaw)]。
BETA_RELATIVE_POSE="${BETA_RELATIVE_POSE:-1.0}"
# 随机整批移除真值 bbox 先验，让 grounding head 同时学习无框绝对检测。
BBOX_DROPOUT_PROB="${BBOX_DROPOUT_PROB:-0.5}"

# 是否冻结 LLM。
# 1：只训练 projector / TVI / planner / grounding head，省显存、稳定。
# 0：微调 LLM，显存更高，也更容易过拟合小数据。
FREEZE_LLM="${FREEZE_LLM:-1}"

# 是否启用 angle TVI。
# 当前双 Agent 数据默认 yaw_hist/yaw_curr 为 0；没有可靠 yaw 时保持 0。
USE_ANGLE_TVI="${USE_ANGLE_TVI:-0}"

# XY 输出缩放。
# 2.0 表示 planner tanh 输出的 XY 会映射到约 ±2m；如需自动估计，设 AUTO_ALPHA_XY=1。
ALPHA_XY="${ALPHA_XY:-2.0}"
AUTO_ALPHA_XY="${AUTO_ALPHA_XY:-0}"

# 是否使用 GND token / grounding head / grounding-to-action feedback。
# 设 USE_GROUNDING=0 可训练“纯双 Agent MLP baseline”：文本 + 双视觉流 + ACT1/ACT2。
USE_GROUNDING="${USE_GROUNDING:-1}"

# 是否把 GT bbox 编成 token 拼入每个 Agent 的视觉流。
# 做最接近单 Agent 的 ablation 时建议和 USE_GROUNDING 一起设为 0。
USE_BBOX_TOKENS="${USE_BBOX_TOKENS:-1}"

# DataLoader worker 数。
# 如果遇到 worker 里 CUDA/编码器问题，先改成 0。
NUM_WORKERS="${NUM_WORKERS:-4}"

# 保存与日志。
LOG_EVERY="${LOG_EVERY:-10}"
SAVE_EVERY="${SAVE_EVERY:-100}"
MAX_CKPTS="${MAX_CKPTS:-0}"

# 是否从 checkpoint 恢复训练。
RESUME="${RESUME:-0}"
RESUME_CKPT="${RESUME_CKPT:-}"

# DDP 是否允许某些可训练参数在部分 iteration 不参与 loss。
# multi-agent grounding/bbox dropout 会切换辅助分支，双卡训练时应保持 1。
DDP_FIND_UNUSED_PARAMETERS="${DDP_FIND_UNUSED_PARAMETERS:-1}"

# =============================================================================
# 5. 评估参数：默认参考单独无人机和单独机器狗评估设置
# =============================================================================

# 评估 checkpoint。默认使用本流水线训练输出目录。
EVAL_CKPT="${EVAL_CKPT:-${OUT_DIR}}"

# 评估读取 test split 的 recorded target_pose，让人按测试集轨迹真实行走/重放。
TEST_TARGET_MANIFEST="${TEST_TARGET_MANIFEST:-${SPLIT_ROOT}/split_manifest.json}"

# 评估输出根目录。每个场景会单独写到 EVAL_ROOT/ENV_ID。
EVAL_ROOT="${EVAL_ROOT:-/data/hdt/ntv_data/sim_data/eval/New_paths_training_multi_agent_mlp_10to1}"

# 默认评估所有已有 test scene；如只测少数场景，用逗号分隔覆盖 EVAL_SCENES。
DEFAULT_EVAL_SCENES="UnrealTrack-ChineseWaterTown_Ver1-ContinuousColor-v0,UnrealTrack-ContainerYard_Night-ContinuousColor-v0,UnrealTrack-Demonstration_Castle-ContinuousColor-v0,UnrealTrack-DowntownWest-ContinuousColor-v0,UnrealTrack-FlexibleRoom-ContinuousColor-v0,UnrealTrack-Greek_Island-ContinuousColor-v0,UnrealTrack-Map_ChemicalPlant_1-ContinuousColor-v0,UnrealTrack-Medieval_Castle-ContinuousColor-v0,UnrealTrack-ModularNeighborhood-ContinuousColor-v0,UnrealTrack-PlanetOutDoor-ContinuousColor-v0,UnrealTrack-PostSoviet_Village-ContinuousColor-v0,UnrealTrack-Real_Landscape-ContinuousColor-v0,UnrealTrack-SnowMap-ContinuousColor-v0,UnrealTrack-Stadium-ContinuousColor-v0,UnrealTrack-StonePineForest-ContinuousColor-v0"
EVAL_SCENES="${EVAL_SCENES:-${DEFAULT_EVAL_SCENES}}"

# 评估使用的物理卡。多个场景会按顺序轮流分配到这些卡上。
EVAL_GPUS="${EVAL_GPUS:-${CUDA_VISIBLE_DEVICES}}"

# 每个场景评估几条 episode。当前 eval 脚本没有 0=全部语义。
EVAL_EPISODES="${EVAL_EPISODES:-3}"
EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-600}"
EVAL_MAX_LOST_STEPS="${EVAL_MAX_LOST_STEPS:-20}"
EVAL_MAX_FAILURE_STEPS="${EVAL_MAX_FAILURE_STEPS:-50}"
EVAL_FAILURE_WARMUP_STEPS="${EVAL_FAILURE_WARMUP_STEPS:-20}"
EVAL_MAX_EPISODE_SECONDS="${EVAL_MAX_EPISODE_SECONDS:-600}"
EVAL_SEED="${EVAL_SEED:-100}"

# 路点选择：机器狗参考单 Agent dog 的第 5 个未来点，无人机参考 drone 的第 9 个未来点。
EVAL_WAYPOINT_INDEX="${EVAL_WAYPOINT_INDEX:-5}"
EVAL_DRONE_WAYPOINT_INDEX="${EVAL_DRONE_WAYPOINT_INDEX:-9}"
EVAL_ROBOTDOG_WAYPOINT_INDEX="${EVAL_ROBOTDOG_WAYPOINT_INDEX:-5}"

# 人与两个 Agent 的执行/判定参数。
EVAL_HUMAN_SPEED="${EVAL_HUMAN_SPEED:-0.9}"  # m/s; allowed: 0.9, 1.0, 1.1, 1.2
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
EVAL_ROBOTDOG_MAX_TURN_DEG="${EVAL_ROBOTDOG_MAX_TURN_DEG:-30}"
EVAL_ROBOTDOG_SUCCESS_DISTANCE="${EVAL_ROBOTDOG_SUCCESS_DISTANCE:-8.0}"
EVAL_ROBOTDOG_LOST_DISTANCE="${EVAL_ROBOTDOG_LOST_DISTANCE:-8.0}"
EVAL_ROBOTDOG_YAW_SIGN="${EVAL_ROBOTDOG_YAW_SIGN:-1.0}"

EVAL_DRONE_MAX_VX="${EVAL_DRONE_MAX_VX:-${DEFAULT_AGENT_MAX_SPEED}}"
EVAL_DRONE_MAX_VY="${EVAL_DRONE_MAX_VY:-${DEFAULT_AGENT_LATERAL_SPEED}}"
EVAL_DRONE_MAX_YAW_RATE="${EVAL_DRONE_MAX_YAW_RATE:-0.4}"
EVAL_DRONE_VX_SCALE="${EVAL_DRONE_VX_SCALE:-0.12}"
EVAL_DRONE_VY_SCALE="${EVAL_DRONE_VY_SCALE:-0.1}"
: "${EVAL_DRONE_YAW_SIGN:=1.0}"
EVAL_DRONE_SUCCESS_DISTANCE="${EVAL_DRONE_SUCCESS_DISTANCE:-5.5}"
EVAL_DRONE_LOST_DISTANCE="${EVAL_DRONE_LOST_DISTANCE:-5.5}"

EVAL_SUCCESS_RATE_THRESHOLD="${EVAL_SUCCESS_RATE_THRESHOLD:-0.8}"
EVAL_MIN_CENTERED_RATE="${EVAL_MIN_CENTERED_RATE:-0.8}"
EVAL_MIN_SUCCESS_STEPS="${EVAL_MIN_SUCCESS_STEPS:-20}"
EVAL_SAVE_VIDEO="${EVAL_SAVE_VIDEO:-1}"
EVAL_WRITE_GLOBAL_VIDEO="${EVAL_WRITE_GLOBAL_VIDEO:-1}"
EVAL_BBOX_SOURCE="${EVAL_BBOX_SOURCE:-model}"
EVAL_TRAJECTORY_OVERLAY="${EVAL_TRAJECTORY_OVERLAY:-1}"
EVAL_TRAJECTORY_SCALE="${EVAL_TRAJECTORY_SCALE:-120}"
EVAL_FACE_TARGET_BEFORE_STEP="${EVAL_FACE_TARGET_BEFORE_STEP:-0}"
EVAL_DIFFUSION_DETERMINISTIC_INFERENCE="${EVAL_DIFFUSION_DETERMINISTIC_INFERENCE:-1}"

# =============================================================================
# 6. 辅助函数
# =============================================================================

bool_arg() {
  local enabled="$1"
  local flag="$2"
  if [[ "${enabled}" == "1" ]]; then
    printf '%s\n' "${flag}"
  fi
}

print_config() {
  cat <<EOF
===============================================================================
双 Agent TrackVLA Pipeline
===============================================================================
REPO_ROOT=${REPO_ROOT}
PYTHON_BIN=${PYTHON_BIN}

[steps]
RUN_SPLIT=${RUN_SPLIT}
RUN_MAKE_DATA=${RUN_MAKE_DATA}
RUN_PRECACHE=${RUN_PRECACHE}
RUN_DRY_RUN=${RUN_DRY_RUN}
RUN_TRAIN=${RUN_TRAIN}
RUN_EVAL=${RUN_EVAL}

[paths]
RAW_ROOT=${RAW_ROOT}
SPLIT_ROOT=${SPLIT_ROOT}
TRAIN_RAW=${TRAIN_RAW}
TEST_RAW=${TEST_RAW}
INPUT_ROOT=${INPUT_ROOT}
DATA_ROOT=${DATA_ROOT}
TRAIN_JSON=${TRAIN_JSON}
CACHE_ROOT=${CACHE_ROOT}
OUT_DIR=${OUT_DIR}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}

[data]
HISTORY=${HISTORY}
HORIZON=${HORIZON}
N_WAYPOINTS=${N_WAYPOINTS}
DT=${DT}
AGENT1=${AGENT1}
AGENT2=${AGENT2}
MAX_EPISODES=${MAX_EPISODES}

[split]
SPLIT_SEED=${SPLIT_SEED}
SPLIT_TRAIN_PARTS=${SPLIT_TRAIN_PARTS}
SPLIT_TEST_PARTS=${SPLIT_TEST_PARTS}
SPLIT_COPY=${SPLIT_COPY}
REUSE_EXISTING_SPLIT=${REUSE_EXISTING_SPLIT}

[train]
LLM_NAME=${LLM_NAME}
EPOCHS=${EPOCHS}
BATCH_SIZE=${BATCH_SIZE}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS}
LR=${LR}
BETA_NAV=${BETA_NAV}
BETA_BBOX=${BETA_BBOX}
BETA_VISIBLE=${BETA_VISIBLE}
BETA_RELATIVE_POSE=${BETA_RELATIVE_POSE}
BBOX_DROPOUT_PROB=${BBOX_DROPOUT_PROB}
USE_GROUNDING=${USE_GROUNDING}
USE_BBOX_TOKENS=${USE_BBOX_TOKENS}
FREEZE_LLM=${FREEZE_LLM}
NUM_GPUS=${NUM_GPUS}
PRECACHE_NUM_SHARDS=${PRECACHE_NUM_SHARDS}
DDP_FIND_UNUSED_PARAMETERS=${DDP_FIND_UNUSED_PARAMETERS}

[eval]
EVAL_CKPT=${EVAL_CKPT}
TEST_TARGET_MANIFEST=${TEST_TARGET_MANIFEST}
EVAL_ROOT=${EVAL_ROOT}
EVAL_GPUS=${EVAL_GPUS}
EVAL_EPISODES=${EVAL_EPISODES}
EVAL_MAX_STEPS=${EVAL_MAX_STEPS}
EVAL_WAYPOINT_INDEX=${EVAL_WAYPOINT_INDEX}
EVAL_DRONE_WAYPOINT_INDEX=${EVAL_DRONE_WAYPOINT_INDEX}
EVAL_ROBOTDOG_WAYPOINT_INDEX=${EVAL_ROBOTDOG_WAYPOINT_INDEX}
EVAL_HUMAN_SPEED=${EVAL_HUMAN_SPEED}
EVAL_DRONE_MAX_VX=${EVAL_DRONE_MAX_VX}
EVAL_DRONE_MAX_VY=${EVAL_DRONE_MAX_VY}
EVAL_DRONE_VX_SCALE=${EVAL_DRONE_VX_SCALE}
EVAL_DRONE_VY_SCALE=${EVAL_DRONE_VY_SCALE}
EVAL_ROBOTDOG_MAX_SPEED=${EVAL_ROBOTDOG_MAX_SPEED}
EVAL_ROBOTDOG_MAX_TURN_DEG=${EVAL_ROBOTDOG_MAX_TURN_DEG}
===============================================================================
EOF
}

run_cmd() {
  echo
  echo ">>> $*"
  "$@"
}

run_eval_scene() {
  local scene="$1"
  local gpu="$2"
  local save_path="${EVAL_ROOT}/${scene}"
  local cuda_gpu="${gpu}"
  if [[ "${gpu}" =~ ^[0-9]+$ ]] && command -v nvidia-smi >/dev/null 2>&1; then
    cuda_gpu="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v idx="${gpu}" '$1 == idx {print $2; exit}')"
    cuda_gpu="${cuda_gpu:-${gpu}}"
  fi

  echo
  echo ">>> eval scene=${scene} gpu=${gpu} cuda_gpu=${cuda_gpu} save_path=${save_path}"
  CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER}" \
  CUDA_VISIBLE_DEVICES="${cuda_gpu}" \
  CKPT="${EVAL_CKPT}" \
  GPU="${cuda_gpu}" \
  RENDER_GPU="${gpu}" \
  ENV_ID="${scene}" \
  EPISODES="${EVAL_EPISODES}" \
  MAX_STEPS="${EVAL_MAX_STEPS}" \
  MAX_LOST_STEPS="${EVAL_MAX_LOST_STEPS}" \
  MAX_FAILURE_STEPS="${EVAL_MAX_FAILURE_STEPS}" \
  FAILURE_WARMUP_STEPS="${EVAL_FAILURE_WARMUP_STEPS}" \
  MAX_EPISODE_SECONDS="${EVAL_MAX_EPISODE_SECONDS}" \
  SEED="${EVAL_SEED}" \
  SAVE_PATH="${save_path}" \
  TEST_TARGET_MANIFEST="${TEST_TARGET_MANIFEST}" \
  DT="${DT}" \
  WAYPOINT_INDEX="${EVAL_WAYPOINT_INDEX}" \
  DRONE_WAYPOINT_INDEX="${EVAL_DRONE_WAYPOINT_INDEX}" \
  ROBOTDOG_WAYPOINT_INDEX="${EVAL_ROBOTDOG_WAYPOINT_INDEX}" \
  HUMAN_SPEED="${EVAL_HUMAN_SPEED}" \
  ROBOTDOG_MAX_SPEED="${EVAL_ROBOTDOG_MAX_SPEED}" \
  ROBOTDOG_MAX_TURN_DEG="${EVAL_ROBOTDOG_MAX_TURN_DEG}" \
  ROBOTDOG_SUCCESS_DISTANCE="${EVAL_ROBOTDOG_SUCCESS_DISTANCE}" \
  ROBOTDOG_LOST_DISTANCE="${EVAL_ROBOTDOG_LOST_DISTANCE}" \
  ROBOTDOG_YAW_SIGN="${EVAL_ROBOTDOG_YAW_SIGN}" \
  DRONE_MAX_VX="${EVAL_DRONE_MAX_VX}" \
  DRONE_MAX_VY="${EVAL_DRONE_MAX_VY}" \
  DRONE_MAX_YAW_RATE="${EVAL_DRONE_MAX_YAW_RATE}" \
  DRONE_VX_SCALE="${EVAL_DRONE_VX_SCALE}" \
  DRONE_VY_SCALE="${EVAL_DRONE_VY_SCALE}" \
  DRONE_YAW_SIGN="${EVAL_DRONE_YAW_SIGN}" \
  DRONE_SUCCESS_DISTANCE="${EVAL_DRONE_SUCCESS_DISTANCE}" \
  DRONE_LOST_DISTANCE="${EVAL_DRONE_LOST_DISTANCE}" \
  SUCCESS_RATE_THRESHOLD="${EVAL_SUCCESS_RATE_THRESHOLD}" \
  MIN_CENTERED_RATE="${EVAL_MIN_CENTERED_RATE}" \
  MIN_SUCCESS_STEPS="${EVAL_MIN_SUCCESS_STEPS}" \
  SAVE_VIDEO="${EVAL_SAVE_VIDEO}" \
  WRITE_GLOBAL_VIDEO="${EVAL_WRITE_GLOBAL_VIDEO}" \
  BBOX_SOURCE="${EVAL_BBOX_SOURCE}" \
  TRAJECTORY_OVERLAY="${EVAL_TRAJECTORY_OVERLAY}" \
  TRAJECTORY_SCALE="${EVAL_TRAJECTORY_SCALE}" \
  FACE_TARGET_BEFORE_STEP="${EVAL_FACE_TARGET_BEFORE_STEP}" \
  DIFFUSION_DETERMINISTIC_INFERENCE="${EVAL_DIFFUSION_DETERMINISTIC_INFERENCE}" \
  bash sh/eval_unrealzoo.sh
}

print_config

# =============================================================================
# 7. Step 1: 每地图 10:1 划分 train/test
# =============================================================================

if [[ "${RUN_SPLIT}" == "1" ]]; then
  if [[ "${REUSE_EXISTING_SPLIT}" == "1" && -f "${SPLIT_ROOT}/split_manifest.json" ]]; then
    echo "[SKIP] 检测到已有 ${SPLIT_ROOT}/split_manifest.json，复用现有 split。"
  else
    SPLIT_ARGS=(
      "${PYTHON_BIN}" tools/split_unrealzoo_multi_agent_data.py
      --input-root "${RAW_ROOT}"
      --output-root "${SPLIT_ROOT}"
      --train-parts "${SPLIT_TRAIN_PARTS}"
      --test-parts "${SPLIT_TEST_PARTS}"
      --seed "${SPLIT_SEED}"
    )
    [[ "${SPLIT_COPY}" == "1" ]] && SPLIT_ARGS+=(--copy)
    run_cmd "${SPLIT_ARGS[@]}"
  fi
else
  echo "[SKIP] RUN_SPLIT=0，跳过 train/test 划分。"
fi

# =============================================================================
# 8. Step 2: 生成双 Agent JSONL
# =============================================================================

if [[ "${RUN_MAKE_DATA}" == "1" ]]; then
  MAKE_ARGS=(
    "${PYTHON_BIN}" -m tools.make_tracking_data
    --multi_agent
    --input_root "${INPUT_ROOT}"
    --output_root "${DATA_ROOT}"
    --history "${HISTORY}"
    --horizon "${HORIZON}"
    --n_waypoints "${N_WAYPOINTS}"
    --dt "${DT}"
    --agent1 "${AGENT1}"
    --agent2 "${AGENT2}"
    --action_field "${ACTION_FIELD}"
    --min_target_visibility "${MIN_TARGET_VISIBILITY}"
    --min_agent_following_rate "${MIN_AGENT_FOLLOWING_RATE}"
    --min_total_steps "${MIN_TOTAL_STEPS}"
    --max_episodes "${MAX_EPISODES}"
  )
  [[ -n "${INSTRUCTION}" ]] && MAKE_ARGS+=(--instruction "${INSTRUCTION}")
  [[ "${ONLY_SUCCESS}" == "1" ]] && MAKE_ARGS+=(--only_success)
  [[ "${EXCLUDE_COLLISION}" == "1" ]] && MAKE_ARGS+=(--exclude_collision)
  [[ "${SKIP_COLLISION_STEPS}" == "1" ]] && MAKE_ARGS+=(--skip_collision_steps)
  [[ "${REQUIRE_VISIBLE}" == "1" ]] && MAKE_ARGS+=(--require_visible)
  [[ "${ALLOW_PARTIAL_HORIZON}" == "1" ]] && MAKE_ARGS+=(--allow_partial_horizon)
  [[ "${REUSE_EXISTING_FRAMES}" == "0" ]] && MAKE_ARGS+=(--no_reuse_existing_frames)

  run_cmd "${MAKE_ARGS[@]}"
else
  echo "[SKIP] RUN_MAKE_DATA=0，跳过数据处理。"
fi

# =============================================================================
# 9. Step 3: 预缓存视觉 token
# =============================================================================

if [[ "${RUN_PRECACHE}" == "1" ]]; then
  if [[ "${PRECACHE_NUM_SHARDS}" -gt 1 ]]; then
    IFS=',' read -r -a PRECACHE_GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
    if [[ "${#PRECACHE_GPU_LIST[@]}" -lt "${PRECACHE_NUM_SHARDS}" ]]; then
      echo "[ERROR] PRECACHE_NUM_SHARDS=${PRECACHE_NUM_SHARDS} 但 CUDA_VISIBLE_DEVICES 只有 ${#PRECACHE_GPU_LIST[@]} 张卡。"
      exit 1
    fi
    pids=()
    for shard_id in $(seq 0 "$((PRECACHE_NUM_SHARDS - 1))"); do
      gpu="${PRECACHE_GPU_LIST[$shard_id]}"
      PRECACHE_ARGS=(
        "${PYTHON_BIN}" -m tools.precache_frames
        --multi_agent
        --data_root "${DATA_ROOT}"
        --cache_root "${CACHE_ROOT}"
        --batch_size "${PRECACHE_BATCH_SIZE}"
        --image_size "${IMAGE_SIZE}"
        --limit "${PRECACHE_LIMIT}"
        --num_shards "${PRECACHE_NUM_SHARDS}"
        --shard_id "${shard_id}"
      )
      [[ -n "${PRECACHE_DEVICE}" ]] && PRECACHE_ARGS+=(--device "${PRECACHE_DEVICE}")
      [[ "${PRECACHE_LIST_ONLY}" == "1" ]] && PRECACHE_ARGS+=(--list_only)
      echo
      echo ">>> CUDA_VISIBLE_DEVICES=${gpu} ${PRECACHE_ARGS[*]}"
      CUDA_VISIBLE_DEVICES="${gpu}" "${PRECACHE_ARGS[@]}" &
      pids+=("$!")
    done
    for pid in "${pids[@]}"; do
      wait "${pid}"
    done
  else
    PRECACHE_ARGS=(
      "${PYTHON_BIN}" -m tools.precache_frames
      --multi_agent
      --data_root "${DATA_ROOT}"
      --cache_root "${CACHE_ROOT}"
      --batch_size "${PRECACHE_BATCH_SIZE}"
      --image_size "${IMAGE_SIZE}"
      --limit "${PRECACHE_LIMIT}"
    )
    [[ -n "${PRECACHE_DEVICE}" ]] && PRECACHE_ARGS+=(--device "${PRECACHE_DEVICE}")
    [[ "${PRECACHE_LIST_ONLY}" == "1" ]] && PRECACHE_ARGS+=(--list_only)

    run_cmd "${PRECACHE_ARGS[@]}"
  fi
else
  echo "[SKIP] RUN_PRECACHE=0，跳过视觉缓存。"
fi

# =============================================================================
# 10. Step 4: 训练前 dry_run 检查
# =============================================================================

if [[ "${RUN_DRY_RUN}" == "1" ]]; then
  DRY_ARGS=(
    "${PYTHON_BIN}" train.py
    --multi_agent
    --train_json "${TRAIN_JSON}"
    --out_dir "${OUT_DIR}"
    --cache_root "${CACHE_ROOT}"
    --llm_name "${LLM_NAME}"
    --n_waypoints "${N_WAYPOINTS}"
    --history "${HISTORY}"
    --batch_size "${BATCH_SIZE}"
    --num_workers 0
    --dry_run
  )
  run_cmd "${DRY_ARGS[@]}"
else
  echo "[SKIP] RUN_DRY_RUN=0，跳过训练前检查。"
fi

# =============================================================================
# 11. Step 5: 正式训练
# =============================================================================

if [[ "${RUN_TRAIN}" == "1" ]]; then
  TRAIN_ARGS=(
    train.py
    --multi_agent
    --train_json "${TRAIN_JSON}"
    --out_dir "${OUT_DIR}"
    --cache_root "${CACHE_ROOT}"
    --llm_name "${LLM_NAME}"
    --n_waypoints "${N_WAYPOINTS}"
    --history "${HISTORY}"
    --epochs "${EPOCHS}"
    --batch_size "${BATCH_SIZE}"
    --grad_accum_steps "${GRAD_ACCUM_STEPS}"
    --lr "${LR}"
    --beta_nav "${BETA_NAV}"
    --beta_bbox "${BETA_BBOX}"
    --beta_visible "${BETA_VISIBLE}"
    --beta_relative_pose "${BETA_RELATIVE_POSE}"
    --bbox_dropout_prob "${BBOX_DROPOUT_PROB}"
    --num_workers "${NUM_WORKERS}"
    --log_every "${LOG_EVERY}"
    --save_every "${SAVE_EVERY}"
    --max_ckpts "${MAX_CKPTS}"
  )
  if [[ "${DDP_FIND_UNUSED_PARAMETERS}" == "1" ]]; then
    TRAIN_ARGS+=(--ddp-find-unused-parameters)
  else
    TRAIN_ARGS+=(--no-ddp-find-unused-parameters)
  fi

  if [[ "${FREEZE_LLM}" == "1" ]]; then
    TRAIN_ARGS+=(--freeze_llm)
  else
    TRAIN_ARGS+=(--no-freeze_llm)
  fi

  [[ "${USE_ANGLE_TVI}" == "1" ]] && TRAIN_ARGS+=(--use_angle_tvi)
  if [[ "${USE_GROUNDING}" == "1" ]]; then
    TRAIN_ARGS+=(--use-grounding)
  else
    TRAIN_ARGS+=(--no-use-grounding)
  fi
  if [[ "${USE_BBOX_TOKENS}" == "1" ]]; then
    TRAIN_ARGS+=(--use-bbox-tokens)
  else
    TRAIN_ARGS+=(--no-use-bbox-tokens)
  fi
  if [[ "${AUTO_ALPHA_XY}" == "1" ]]; then
    TRAIN_ARGS+=(--auto_alpha_xy)
  else
    TRAIN_ARGS+=(--alpha_xy "${ALPHA_XY}")
  fi
  [[ "${RESUME}" == "1" ]] && TRAIN_ARGS+=(--resume)
  [[ -n "${RESUME_CKPT}" ]] && TRAIN_ARGS+=(--resume_ckpt "${RESUME_CKPT}")

  if [[ "${NUM_GPUS}" -gt 1 ]]; then
    run_cmd "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node "${NUM_GPUS}" "${TRAIN_ARGS[@]}" --distributed
  else
    run_cmd "${PYTHON_BIN}" "${TRAIN_ARGS[@]}"
  fi
else
  echo "[SKIP] RUN_TRAIN=0，跳过正式训练。确认 dry_run 正常后可设 RUN_TRAIN=1。"
fi

# =============================================================================
# 12. Step 6: UnrealZoo 闭环评估
# =============================================================================

if [[ "${RUN_EVAL}" == "1" ]]; then
  IFS=',' read -r -a EVAL_SCENE_LIST <<< "${EVAL_SCENES}"
  IFS=',' read -r -a EVAL_GPU_LIST <<< "${EVAL_GPUS}"
  if [[ "${#EVAL_GPU_LIST[@]}" -eq 0 ]]; then
    echo "[ERROR] EVAL_GPUS 为空，无法评估。"
    exit 1
  fi

  for idx in "${!EVAL_SCENE_LIST[@]}"; do
    scene="${EVAL_SCENE_LIST[$idx]}"
    gpu="${EVAL_GPU_LIST[$((idx % ${#EVAL_GPU_LIST[@]}))]}"
    run_eval_scene "${scene}" "${gpu}"
  done
else
  echo "[SKIP] RUN_EVAL=0，跳过 UnrealZoo 闭环评估。需要评估时设 RUN_EVAL=1。"
fi

echo
echo "[DONE] pipeline finished."
