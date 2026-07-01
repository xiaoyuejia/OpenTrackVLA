#!/usr/bin/env bash
set -euo pipefail

# Shared-context multi-agent base without per-agent language markers.
# LLM layout: [joint text, drone visual, robotdog visual, ACT1, ACT2].

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/hdt/miniconda3/envs/omtracknew/bin/python}"
GPU="${GPU:-0}"
DATA_ROOT="${DATA_ROOT:-/data/hdt/ntv_data/data/New_paths_training_multi_agent_10to1}"
TRAIN_JSON="${TRAIN_JSON:-${DATA_ROOT}/train/jsonl}"
CACHE_ROOT="${CACHE_ROOT:-${DATA_ROOT}/train/vision_cache}"
OUT_DIR="${OUT_DIR:-/data/hdt/ntv_data/ckpt/New_paths_training_multi_agent_base_concat_no_agent_text_b32_acc4_lr2e-5}"

JOINT_INSTRUCTION="${JOINT_INSTRUCTION:-The aerial drone and the ground robot dog must cooperatively track the same target person. The drone should follow the person from the air, and the robot dog should follow the same person on the ground.}"

RUN_DRY_RUN="${RUN_DRY_RUN:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
ALLOW_EXISTING_OUT_DIR="${ALLOW_EXISTING_OUT_DIR:-0}"

[[ -x "${PYTHON_BIN}" ]] || { echo "[ERROR] Python not executable: ${PYTHON_BIN}" >&2; exit 1; }
[[ -d "${TRAIN_JSON}" ]] || { echo "[ERROR] Training JSONL directory not found: ${TRAIN_JSON}" >&2; exit 1; }
[[ -d "${CACHE_ROOT}" ]] || { echo "[ERROR] Vision cache not found: ${CACHE_ROOT}" >&2; exit 1; }
mkdir -p "${OUT_DIR}"
if [[ "${ALLOW_EXISTING_OUT_DIR}" != "1" ]] && compgen -G "${OUT_DIR}/model*.pt" > /dev/null; then
    echo "[ERROR] Existing checkpoint found in ${OUT_DIR}; choose a new OUT_DIR for fresh training." >&2
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
    --save_every 1000
    --save_every_epochs 1
    --max_ckpts 3
    --seed 42
    --no-ddp-find-unused-parameters
    --joint_instruction_override "${JOINT_INSTRUCTION}"
)

echo "==============================================================================="
echo "Multi-agent base concat training (no agent text markers)"
echo "GPU=${GPU}"
echo "TRAIN_JSON=${TRAIN_JSON}"
echo "CACHE_ROOT=${CACHE_ROOT}"
echo "OUT_DIR=${OUT_DIR}"
echo "layout=[joint_text, drone_visual, robotdog_visual, ACT1, ACT2]"
echo "batch=32 accumulation=4 effective_batch=128"
echo "lr=2e-5 scheduler=cosine warmup_steps=500 min_lr=2e-6"
echo "agent_loss=100 * (MSE_drone + MSE_robotdog) / 2"
echo "==============================================================================="

if [[ "${RUN_DRY_RUN}" == "1" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" "${TRAIN_ARGS[@]}" --dry_run
fi

if [[ "${RUN_TRAIN}" == "1" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU}" \
        "${PYTHON_BIN}" "${TRAIN_ARGS[@]}" \
        2>&1 | tee "${OUT_DIR}/train_stdout.log"
fi
