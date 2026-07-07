#!/usr/bin/env bash
set -euo pipefail

# Conservative base-concat experiment for repaired data3047 B-grade data.
# Baseline script is intentionally left untouched.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/hdt/miniconda3/envs/omtracknew/bin/python}"
GPU="${GPU:-0}"

DATA_ROOT="${DATA_ROOT:-/data/hdt/ntv_data/data/data3047_150test_B_repaired}"
SPLIT_ROOT="${SPLIT_ROOT:-${DATA_ROOT}/episode_split_seed42}"
TRAIN_JSON="${TRAIN_JSON:-${SPLIT_ROOT}/train/jsonl}"
VAL_JSON="${VAL_JSON:-${SPLIT_ROOT}/val/jsonl}"
CACHE_ROOT="${CACHE_ROOT:-${SPLIT_ROOT}/train/vision_cache}"
VAL_CACHE_ROOT="${VAL_CACHE_ROOT:-${SPLIT_ROOT}/val/vision_cache}"
OUT_DIR="${OUT_DIR:-/data/hdt/ntv_data/ckpt/data3047_B_repaired_base_concat_conservative_b32_acc4_lr2e-5}"

JOINT_INSTRUCTION="${JOINT_INSTRUCTION:-follow the person}"

# Safe default: run configuration/data/model dry-run only.
RUN_DRY_RUN="${RUN_DRY_RUN:-1}"
RUN_TRAIN="${RUN_TRAIN:-0}"
ALLOW_EXISTING_OUT_DIR="${ALLOW_EXISTING_OUT_DIR:-0}"

SAVE_EVERY="${SAVE_EVERY:-0}"
SAVE_EVERY_EPOCHS="${SAVE_EVERY_EPOCHS:-1}"
MAX_CKPTS="${MAX_CKPTS:-0}"
EVAL_EVERY="${EVAL_EVERY:-500}"
EVAL_BATCHES="${EVAL_BATCHES:-8}"
VAL_BBOX_SOURCE="${VAL_BBOX_SOURCE:-none}"

[[ -x "${PYTHON_BIN}" ]] || { echo "[ERROR] Python not executable: ${PYTHON_BIN}" >&2; exit 1; }
[[ -d "${DATA_ROOT}/train/jsonl" ]] || { echo "[ERROR] repaired data root missing train/jsonl: ${DATA_ROOT}" >&2; exit 1; }
if [[ ! -d "${TRAIN_JSON}" || ! -d "${VAL_JSON}" ]]; then
    echo "[ERROR] Episode split not found under ${SPLIT_ROOT}" >&2
    echo "Create it first, without moving source data:" >&2
    echo "  ${PYTHON_BIN} tools/split_multi_agent_dataset_by_episode.py \\" >&2
    echo "    --jsonl-root ${DATA_ROOT}/train/jsonl \\" >&2
    echo "    --data-root ${DATA_ROOT}/train \\" >&2
    echo "    --out-root ${SPLIT_ROOT} \\" >&2
    echo "    --val-ratio 0.1 --seed 42 --link-mode symlink" >&2
    exit 2
fi
[[ -d "${CACHE_ROOT}" ]] || { echo "[ERROR] train vision cache not found: ${CACHE_ROOT}" >&2; exit 1; }
[[ -d "${VAL_CACHE_ROOT}" ]] || { echo "[ERROR] val vision cache not found: ${VAL_CACHE_ROOT}" >&2; exit 1; }

mkdir -p "${OUT_DIR}"
if [[ "${ALLOW_EXISTING_OUT_DIR}" != "1" ]] && compgen -G "${OUT_DIR}/model*.pt" > /dev/null; then
    echo "[ERROR] Existing checkpoint found in ${OUT_DIR}; choose a new OUT_DIR." >&2
    exit 1
fi

export TOKENIZERS_PARALLELISM=false
export TRACKVLA_USE_MODELSCOPE="${TRACKVLA_USE_MODELSCOPE:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TRAIN_ARGS=(
    train.py
    --multi_agent
    --base_model
    --no-agent-text-markers
    --normalize-agent-loss-weights
    --train_json "${TRAIN_JSON}"
    --cache_root "${CACHE_ROOT}"
    --val_json "${VAL_JSON}"
    --val_cache_root "${VAL_CACHE_ROOT}"
    --out_dir "${OUT_DIR}"
    --llm_name "Qwen/Qwen3-0.6B"
    --history 31
    --n_waypoints 10
    --epochs 10
    --batch_size 32
    --grad_accum_steps 4
    --lr 2e-5
    --lr_scheduler cosine
    --warmup_steps 500
    --min_lr 2e-6
    --weight_decay 0.01
    --grad_clip 1.0
    --beta_nav 100
    --drone_loss_weight 1
    --dog_loss_weight 1
    --nav_loss_type mse
    --yaw_loss_weight 1
    --final_waypoint_loss_weight 0
    --turn_sample_weight 1
    --stop_sample_weight 1
    --alpha_xy 1
    --no_tanh_actions
    --mixed_precision
    --freeze_llm
    --num_workers 8
    --prefetch_factor 2
    --coarse_cache_size 4096
    --log_every 10
    --save_every "${SAVE_EVERY}"
    --save_every_epochs "${SAVE_EVERY_EPOCHS}"
    --max_ckpts "${MAX_CKPTS}"
    --eval_every "${EVAL_EVERY}"
    --eval_batches "${EVAL_BATCHES}"
    --val_bbox_source "${VAL_BBOX_SOURCE}"
    --seed 42
    --no-ddp-find-unused-parameters
    --joint_instruction_override "${JOINT_INSTRUCTION}"
)

echo "==============================================================================="
echo "Conservative multi-agent base concat training"
echo "GPU=${GPU}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "TRAIN_JSON=${TRAIN_JSON}"
echo "VAL_JSON=${VAL_JSON}"
echo "CACHE_ROOT=${CACHE_ROOT}"
echo "VAL_CACHE_ROOT=${VAL_CACHE_ROOT}"
echo "OUT_DIR=${OUT_DIR}"
echo "batch=32 accumulation=4 effective_batch=128"
echo "lr=2e-5 scheduler=cosine warmup_steps=500 min_lr=2e-6 weight_decay=0.01"
echo "grad_clip=1.0 beta_nav=100 nav_loss=mse alpha_xy=1 no_tanh_actions"
echo "RUN_DRY_RUN=${RUN_DRY_RUN} RUN_TRAIN=${RUN_TRAIN}"
echo "==============================================================================="

if [[ "${RUN_DRY_RUN}" == "1" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" "${TRAIN_ARGS[@]}" --dry_run
fi

if [[ "${RUN_TRAIN}" == "1" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU}" \
        "${PYTHON_BIN}" "${TRAIN_ARGS[@]}" \
        2>&1 | tee "${OUT_DIR}/train_stdout.log"
fi
