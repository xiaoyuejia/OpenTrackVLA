#!/usr/bin/env bash
# ``model_unrealzoo_anchor_diffusion.py`` 双 Agent Anchor Diffusion 训练入口。
#
# 默认只生成锚点并做数据 dry-run，不会直接开始长时间训练：
#   bash sh/train_anchor_diffusion.sh
#
# 正式训练，只需指定包含 train/ 的最外层数据集目录：
#   DATASET_ROOT=/path/to/dataset RUN_BUILD_ANCHORS=0 RUN_DRY_RUN=0 RUN_TRAIN=1 bash sh/train_anchor_diffusion.sh

# =============================================================================
# GPU 配置：默认使用物理 GPU 1，也支持多卡训练
# =============================================================================

# 单卡示例：CUDA_VISIBLE_DEVICES=1
# 多卡示例：CUDA_VISIBLE_DEVICES=0,1,2,3
# PyTorch 内部会把可见卡重新编号为 cuda:0、cuda:1、……。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
IFS=',' read -r -a VISIBLE_GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${NUM_GPUS:-${#VISIBLE_GPU_LIST[@]}}"
unset VISIBLE_GPU_LIST

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/hdt/miniconda3/envs/omtracknew/bin/python}"
[[ -x "${PYTHON_BIN}" ]] || PYTHON_BIN="python"

# DATASET_ROOT 是推荐入口，其目录结构应为：
#   DATASET_ROOT/train/dataset.json
#   DATASET_ROOT/train/vision_cache/
#   DATASET_ROOT/train/trajectory_anchors/
# 默认使用当前整理好的 2:1 数据集；仍可显式传 DATA_ROOT 使用旧的单目录模式。
DEFAULT_DATASET_ROOT="/data/hdt/ntv_data/data/unrealzoo_aerial_ground_human_hand_multi_2to1"
if [[ -z "${DATASET_ROOT:-}" && -z "${DATA_ROOT:-}" ]]; then
  DATASET_ROOT="${DEFAULT_DATASET_ROOT}"
else
  DATASET_ROOT="${DATASET_ROOT:-}"
fi
if [[ -n "${DATASET_ROOT}" ]]; then
  DATASET_ROOT="$(readlink -f "${DATASET_ROOT}")"
  [[ -f "${DATASET_ROOT}/train/dataset.json" ]] || {
    echo "DATASET_ROOT must contain train/dataset.json: ${DATASET_ROOT}" >&2
    exit 1
  }
  # DATASET_ROOT 模式下强制使用其 train/，避免终端中残留的旧 DATA_ROOT 覆盖当前数据集。
  DATA_ROOT="${DATASET_ROOT}/train"
  DATASET_NAME="$(basename "${DATASET_ROOT}")"
  DEFAULT_OUT_DIR="/data/hdt/ntv_data/ckpt/ckpts_multi_agent_anchor_diffusion_${DATASET_NAME}"
else
  DATA_ROOT="$(readlink -f "${DATA_ROOT}")"
  DEFAULT_OUT_DIR="/data/hdt/ntv_data/ckpt/ckpts_multi_agent_anchor_diffusion"
fi
# 当前数据目录中已有 dataset.json，直接使用它也不要求 jsonl/ 目录存在。
TRAIN_JSON="${TRAIN_JSON:-${DATA_ROOT}/dataset.json}"
CACHE_ROOT="${CACHE_ROOT:-${DATA_ROOT}/vision_cache}"
RUN_VALIDATION="${RUN_VALIDATION:-0}"
if [[ "${RUN_VALIDATION}" == "1" && -n "${DATASET_ROOT}" ]]; then
  VAL_JSON="${VAL_JSON:-${DATASET_ROOT}/test/dataset.json}"
  VAL_CACHE_ROOT="${VAL_CACHE_ROOT:-${DATASET_ROOT}/test/vision_cache}"
else
  VAL_JSON="${VAL_JSON:-}"
  VAL_CACHE_ROOT="${VAL_CACHE_ROOT:-}"
fi
ANCHOR_DIR="${ANCHOR_DIR:-${DATA_ROOT}/trajectory_anchors}"
AGENT1_ANCHORS="${AGENT1_ANCHORS:-${ANCHOR_DIR}/agent1_drone_anchors.npy}"
AGENT2_ANCHORS="${AGENT2_ANCHORS:-${ANCHOR_DIR}/agent2_robotdog_anchors.npy}"
OUT_DIR="${OUT_DIR:-${DEFAULT_OUT_DIR}}"

RUN_BUILD_ANCHORS="${RUN_BUILD_ANCHORS:-1}"
RUN_DRY_RUN="${RUN_DRY_RUN:-1}"
RUN_TRAIN="${RUN_TRAIN:-0}"
# 正式训练前核对 dataset.json 实际引用的每张图片是否都有 fine/coarse token。
VERIFY_CACHE="${VERIFY_CACHE:-1}"

N_WAYPOINTS="${N_WAYPOINTS:-8}"
NUM_ANCHORS="${NUM_ANCHORS:-40}"
ANCHOR_MAX_SAMPLES="${ANCHOR_MAX_SAMPLES:-0}"
ANCHOR_KMEANS_ITERS="${ANCHOR_KMEANS_ITERS:-100}"
SEED="${SEED:-0}"

LLM_NAME="${LLM_NAME:-Qwen/Qwen3-0.6B}"
HISTORY="${HISTORY:-31}"
# 论文附录使用 1 epoch；小数据过拟合实验时再显式增大。
EPOCHS="${EPOCHS:-1}"
# A100 80GB 默认使用较大的 micro-batch，提高 Tensor Core 利用率。
# 若显存仍低于约 60GB，可继续尝试 BATCH_SIZE=96 或 128；OOM 时降至 48。
BATCH_SIZE="${BATCH_SIZE:-64}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
LR="${LR:-2e-5}"
DDP_TIMEOUT_MINUTES="${DDP_TIMEOUT_MINUTES:-120}"
# 本机 GPU 0/1 通过 PXB 互联时，NCCL P2P 会卡在 DDP 首次参数同步。
# 默认关闭 P2P 并改走充足的 /dev/shm；单机训练不需要 InfiniBand。
# 若迁移到 NVLink/P2P 工作正常的机器，可显式设置 NCCL_P2P_DISABLE=0。
NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
# 避免 torchrun 自动降到每个 rank 仅 1 个 OpenMP 线程。
OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
# 每个样本会读取大量视觉 token 小文件；worker 内 LRU 避免反复读取历史 vcoarse。
NUM_WORKERS="${NUM_WORKERS:-32}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
COARSE_CACHE_SIZE="${COARSE_CACHE_SIZE:-4096}"
FREEZE_LLM="${FREEZE_LLM:-1}"

# action_loss 内部已经包含回归损失和候选评分 BCE，默认不再额外放大 10 倍。
BETA_NAV="${BETA_NAV:-1.0}"
BETA_BBOX="${BETA_BBOX:-1.0}"
BETA_VISIBLE="${BETA_VISIBLE:-0.5}"
# 一半样本不提供真值 bbox prior，训练首帧绝对检测；其余样本训练带噪 prior 跟踪修正。
BBOX_DROPOUT_PROB="${BBOX_DROPOUT_PROB:-0.50}"
BBOX_CENTER_JITTER_STD="${BBOX_CENTER_JITTER_STD:-0.25}"
BBOX_SIZE_JITTER_STD="${BBOX_SIZE_JITTER_STD:-0.20}"

DIFFUSION_HIDDEN_DIM="${DIFFUSION_HIDDEN_DIM:-768}"
DIFFUSION_DEPTH="${DIFFUSION_DEPTH:-12}"
DIFFUSION_NUM_HEADS="${DIFFUSION_NUM_HEADS:-12}"
DIFFUSION_SCORE_LOSS_WEIGHT="${DIFFUSION_SCORE_LOSS_WEIGHT:-100.0}"
DIFFUSION_SCORE_LOSS_REDUCTION="${DIFFUSION_SCORE_LOSS_REDUCTION:-mean}"
DIFFUSION_TRAIN_TRUNCATION_STEPS="${DIFFUSION_TRAIN_TRUNCATION_STEPS:-50}"
DIFFUSION_INFERENCE_START_TIMESTEP="${DIFFUSION_INFERENCE_START_TIMESTEP:-10}"
DIFFUSION_INFERENCE_STEPS="${DIFFUSION_INFERENCE_STEPS:-2}"

LOG_EVERY="${LOG_EVERY:-20}"
# 默认关闭按 step 保存；每 5 个 epoch 保存一个常规 checkpoint。
SAVE_EVERY="${SAVE_EVERY:-0}"
SAVE_EVERY_EPOCHS="${SAVE_EVERY_EPOCHS:-5}"
# 最近 5 个常规 checkpoint；model_best_val.pt 独立保留，不计入这里。
MAX_CKPTS="${MAX_CKPTS:-5}"
if [[ "${RUN_VALIDATION}" == "1" ]]; then
  EVAL_EVERY="${EVAL_EVERY:-500}"
else
  EVAL_EVERY="${EVAL_EVERY:-0}"
fi
EVAL_BATCHES="${EVAL_BATCHES:-8}"
VAL_BBOX_SOURCE="${VAL_BBOX_SOURCE:-none}"
RESUME="${RESUME:-0}"
RESUME_CKPT="${RESUME_CKPT:-}"

run_cmd() {
  echo
  echo ">>> $*"
  "$@"
}

echo "TRAIN_JSON=${TRAIN_JSON}"
echo "DATASET_ROOT=${DATASET_ROOT:-<legacy DATA_ROOT mode>}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "CACHE_ROOT=${CACHE_ROOT}"
echo "VAL_JSON=${VAL_JSON:-<disabled>}"
echo "VAL_CACHE_ROOT=${VAL_CACHE_ROOT:-<disabled>}"
echo "AGENT1_ANCHORS=${AGENT1_ANCHORS}"
echo "AGENT2_ANCHORS=${AGENT2_ANCHORS}"
echo "OUT_DIR=${OUT_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NUM_GPUS=${NUM_GPUS}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS}"
echo "EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS * GRAD_ACCUM_STEPS))"
echo "NUM_WORKERS=${NUM_WORKERS}"
echo "TOTAL_DATALOADER_WORKERS=$((NUM_WORKERS * NUM_GPUS))"
echo "PREFETCH_FACTOR=${PREFETCH_FACTOR}"
echo "COARSE_CACHE_SIZE=${COARSE_CACHE_SIZE}"
echo "DDP_TIMEOUT_MINUTES=${DDP_TIMEOUT_MINUTES}"
echo "NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE}"
echo "NCCL_IB_DISABLE=${NCCL_IB_DISABLE}"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "VERIFY_CACHE=${VERIFY_CACHE}"
echo "SAVE_EVERY=${SAVE_EVERY}"
echo "SAVE_EVERY_EPOCHS=${SAVE_EVERY_EPOCHS}"
echo "MAX_CKPTS=${MAX_CKPTS} (+ model_best_val.pt)"

if [[ -n "${VAL_JSON}" && -z "${VAL_CACHE_ROOT}" ]]; then
  echo "VAL_JSON is set but VAL_CACHE_ROOT is empty; refusing to reuse training vision_cache for validation." >&2
  exit 1
fi
if [[ "${RUN_TRAIN}" == "1" && -n "${VAL_JSON}" ]]; then
  [[ -f "${VAL_JSON}" ]] || { echo "Validation dataset does not exist: ${VAL_JSON}" >&2; exit 1; }
  [[ -d "${VAL_CACHE_ROOT}" ]] || {
    echo "Validation vision_cache does not exist: ${VAL_CACHE_ROOT}" >&2
    echo "Run tools.precache_frames for the test split before enabling RUN_VALIDATION=1." >&2
    exit 1
  }
fi

verify_multi_agent_cache() {
  local label="$1"
  local data_root="$2"
  local dataset_json="$3"
  local cache_root="$4"
  local expected fine_count coarse_count

  [[ -f "${dataset_json}" ]] || { echo "${label} dataset does not exist: ${dataset_json}" >&2; exit 1; }
  [[ -d "${cache_root}" ]] || { echo "${label} vision_cache does not exist: ${cache_root}" >&2; exit 1; }
  expected="$("${PYTHON_BIN}" -m tools.precache_frames \
    --multi_agent \
    --data_root "${data_root}" \
    --dataset_json "${dataset_json}" \
    --cache_root "${cache_root}" \
    --list_only | sed -n 's/^Frames to check: \([0-9][0-9]*\).*/\1/p')"
  fine_count="$(find "${cache_root}" -type f -name '*_vfine.pt' | wc -l)"
  coarse_count="$(find "${cache_root}" -type f -name '*_vcoarse.pt' | wc -l)"
  echo "[CACHE][${label}] expected=${expected:-unknown} fine=${fine_count} coarse=${coarse_count}"
  if [[ -z "${expected}" || "${fine_count}" -lt "${expected}" || "${coarse_count}" -lt "${expected}" ]]; then
    echo "${label} vision_cache is incomplete; run tools.precache_frames --multi_agent first." >&2
    exit 1
  fi
}

if [[ "${RUN_TRAIN}" == "1" && "${VERIFY_CACHE}" == "1" ]]; then
  verify_multi_agent_cache "train" "${DATA_ROOT}" "${TRAIN_JSON}" "${CACHE_ROOT}"
  if [[ -n "${VAL_JSON}" ]]; then
    verify_multi_agent_cache "validation" "$(dirname "${VAL_JSON}")" "${VAL_JSON}" "${VAL_CACHE_ROOT}"
  fi
fi

if [[ "${RUN_BUILD_ANCHORS}" == "1" ]]; then
  run_cmd "${PYTHON_BIN}" -m tools.build_unrealzoo_trajectory_anchors \
    --train_json "${TRAIN_JSON}" \
    --out_dir "${ANCHOR_DIR}" \
    --n_waypoints "${N_WAYPOINTS}" \
    --num_anchors "${NUM_ANCHORS}" \
    --num_iters "${ANCHOR_KMEANS_ITERS}" \
    --max_samples "${ANCHOR_MAX_SAMPLES}" \
    --seed "${SEED}"
fi

COMMON_ARGS=(
  --train_json "${TRAIN_JSON}"
  --out_dir "${OUT_DIR}"
  --cache_root "${CACHE_ROOT}"
  --llm_name "${LLM_NAME}"
  --n_waypoints "${N_WAYPOINTS}"
  --history "${HISTORY}"
  --batch_size "${BATCH_SIZE}"
  --diffusion_agent1_anchor_path "${AGENT1_ANCHORS}"
  --diffusion_agent2_anchor_path "${AGENT2_ANCHORS}"
  --diffusion_num_anchors "${NUM_ANCHORS}"
  --diffusion_hidden_dim "${DIFFUSION_HIDDEN_DIM}"
  --diffusion_depth "${DIFFUSION_DEPTH}"
  --diffusion_num_heads "${DIFFUSION_NUM_HEADS}"
  --diffusion_score_loss_weight "${DIFFUSION_SCORE_LOSS_WEIGHT}"
  --diffusion_score_loss_reduction "${DIFFUSION_SCORE_LOSS_REDUCTION}"
  --diffusion_train_truncation_steps "${DIFFUSION_TRAIN_TRUNCATION_STEPS}"
  --diffusion_inference_start_timestep "${DIFFUSION_INFERENCE_START_TIMESTEP}"
  --diffusion_inference_steps "${DIFFUSION_INFERENCE_STEPS}"
)

if [[ "${RUN_DRY_RUN}" == "1" ]]; then
  run_cmd "${PYTHON_BIN}" train_unrealzoo_anchor_diffusion.py "${COMMON_ARGS[@]}" --num_workers 0 --dry_run
fi

if [[ "${RUN_TRAIN}" == "1" ]]; then
  TRAIN_ARGS=(
    train_unrealzoo_anchor_diffusion.py
    "${COMMON_ARGS[@]}"
    --epochs "${EPOCHS}"
    --grad_accum_steps "${GRAD_ACCUM_STEPS}"
    --lr "${LR}"
    --beta_nav "${BETA_NAV}"
    --beta_bbox "${BETA_BBOX}"
    --beta_visible "${BETA_VISIBLE}"
    --bbox_dropout_prob "${BBOX_DROPOUT_PROB}"
    --bbox_center_jitter_std "${BBOX_CENTER_JITTER_STD}"
    --bbox_size_jitter_std "${BBOX_SIZE_JITTER_STD}"
    --num_workers "${NUM_WORKERS}"
    --prefetch_factor "${PREFETCH_FACTOR}"
    --coarse_cache_size "${COARSE_CACHE_SIZE}"
    --ddp_timeout_minutes "${DDP_TIMEOUT_MINUTES}"
    --seed "${SEED}"
    --log_every "${LOG_EVERY}"
    --save_every "${SAVE_EVERY}"
    --save_every_epochs "${SAVE_EVERY_EPOCHS}"
    --max_ckpts "${MAX_CKPTS}"
    --eval_every "${EVAL_EVERY}"
    --eval_batches "${EVAL_BATCHES}"
    --val_bbox_source "${VAL_BBOX_SOURCE}"
  )
  [[ -n "${VAL_JSON}" ]] && TRAIN_ARGS+=(--val_json "${VAL_JSON}")
  [[ -n "${VAL_CACHE_ROOT}" ]] && TRAIN_ARGS+=(--val_cache_root "${VAL_CACHE_ROOT}")
  [[ "${FREEZE_LLM}" == "1" ]] && TRAIN_ARGS+=(--freeze_llm) || TRAIN_ARGS+=(--no-freeze_llm)
  [[ "${RESUME}" == "1" ]] && TRAIN_ARGS+=(--resume)
  [[ -n "${RESUME_CKPT}" ]] && TRAIN_ARGS+=(--resume_ckpt "${RESUME_CKPT}")

  if [[ "${NUM_GPUS}" -gt 1 ]]; then
    export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
    export NCCL_P2P_DISABLE NCCL_IB_DISABLE NCCL_DEBUG OMP_NUM_THREADS
    run_cmd "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node "${NUM_GPUS}" "${TRAIN_ARGS[@]}" --distributed
  else
    run_cmd "${PYTHON_BIN}" "${TRAIN_ARGS[@]}"
  fi
fi

echo
echo "[DONE] anchor diffusion pipeline finished."
