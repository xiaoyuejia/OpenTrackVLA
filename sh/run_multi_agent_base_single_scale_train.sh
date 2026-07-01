#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/hdt/miniconda3/envs/omtracknew/bin/python}"
TRAIN_JSON="${TRAIN_JSON:-/data/hdt/ntv_data/data/New_paths_training_multi_agent_10to1/train/jsonl}"
CACHE_ROOT="${CACHE_ROOT:-/data/hdt/ntv_data/data/New_paths_training_multi_agent_10to1/train/vision_cache}"
OUT_DIR="${OUT_DIR:-/data/hdt/ntv_data/ckpt/New_paths_training_multi_agent_base_concat_single_scale_b32}"

TRAIN_GPUS="${TRAIN_GPUS:-3}"
NUM_GPUS="${NUM_GPUS:-1}"
LLM_NAME="${LLM_NAME:-Qwen/Qwen3-0.6B}"
HISTORY="${HISTORY:-31}"
N_WAYPOINTS="${N_WAYPOINTS:-10}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-32}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
LR="${LR:-2e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
BETA_NAV="${BETA_NAV:-100}"
ALPHA_XY="${ALPHA_XY:-1}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LOG_EVERY="${LOG_EVERY:-20}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
MAX_CKPTS="${MAX_CKPTS:-3}"
SEED="${SEED:-42}"
RUN_DRY_RUN="${RUN_DRY_RUN:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"

run_cmd() {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
    "$@"
}

if [[ ! -d "${TRAIN_JSON}" ]]; then
    echo "[ERROR] TRAIN_JSON not found: ${TRAIN_JSON}" >&2
    exit 1
fi
if [[ ! -d "${CACHE_ROOT}" ]]; then
    echo "[ERROR] CACHE_ROOT not found: ${CACHE_ROOT}" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}"
export TOKENIZERS_PARALLELISM=false
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"

BASE_ARGS=(
    train_multi_agent_base.py
    --train_json "${TRAIN_JSON}"
    --out_dir "${OUT_DIR}"
    --cache_root "${CACHE_ROOT}"
    --llm_name "${LLM_NAME}"
    --history "${HISTORY}"
    --n_waypoints "${N_WAYPOINTS}"
    --epochs "${EPOCHS}"
    --batch_size "${BATCH_SIZE}"
    --grad_accum_steps "${GRAD_ACCUM_STEPS}"
    --lr "${LR}"
    --weight_decay "${WEIGHT_DECAY}"
    --beta_nav "${BETA_NAV}"
    --alpha_xy "${ALPHA_XY}"
    --no-tanh-actions
    --mixed_precision "${MIXED_PRECISION}"
    --num_workers "${NUM_WORKERS}"
    --log_every "${LOG_EVERY}"
    --save_every "${SAVE_EVERY}"
    --max_ckpts "${MAX_CKPTS}"
    --seed "${SEED}"
)

echo "[CONFIG] train_json=${TRAIN_JSON}"
echo "[CONFIG] cache_root=${CACHE_ROOT}"
echo "[CONFIG] out_dir=${OUT_DIR}"
echo "[CONFIG] gpus=${TRAIN_GPUS} num_gpus=${NUM_GPUS}"
echo "[CONFIG] waypoint scale aligned with single-agent: beta_nav=${BETA_NAV}, alpha_xy=${ALPHA_XY}, tanh_actions=0"

if [[ "${RUN_DRY_RUN}" == "1" ]]; then
    run_cmd "${PYTHON_BIN}" "${BASE_ARGS[@]}" --dry-run
fi

if [[ "${RUN_TRAIN}" == "1" ]]; then
    if [[ "${NUM_GPUS}" -gt 1 ]]; then
        run_cmd "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node "${NUM_GPUS}" "${BASE_ARGS[@]}"
    else
        run_cmd "${PYTHON_BIN}" "${BASE_ARGS[@]}"
    fi
else
    echo "[SKIP] RUN_TRAIN=0, dry-run only."
fi
