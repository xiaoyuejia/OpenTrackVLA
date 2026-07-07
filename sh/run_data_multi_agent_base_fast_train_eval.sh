#!/usr/bin/env bash
set -euo pipefail

# Fast base multi-agent training + closed-loop test evaluation.
#
# Defaults:
#   - dataset: /data/hdt/ntv_data/data/data_multi_agent_10to1
#   - split/test manifest: /data/hdt/ntv_data/sim_data/data_multi_agent_split_10to1
#   - GPUs: 0,1
#   - per-GPU batch size: 32, grad accumulation: 4
#   - lr: 2e-5
#
# Examples:
#   bash sh/run_data_multi_agent_base_fast_train_eval.sh
#   RUN_EVAL=0 bash sh/run_data_multi_agent_base_fast_train_eval.sh
#   EPOCHS=3 BATCH_SIZE=32 GRAD_ACCUM_STEPS=4 bash sh/run_data_multi_agent_base_fast_train_eval.sh
#   RUN_TRAIN=0 CKPT_DIR=/data/hdt/ntv_data/ckpt/data_multi_agent_base_fast_b32_acc4_lr2e-5_2gpu bash sh/run_data_multi_agent_base_fast_train_eval.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/hdt/miniconda3/envs/omtracknew/bin/python}"
[[ -x "${PYTHON_BIN}" ]] || PYTHON_BIN="python"

DATA_ROOT="${DATA_ROOT:-/data/hdt/ntv_data/data/data_multi_agent_10to1}"
TRAIN_JSON="${TRAIN_JSON:-${DATA_ROOT}/train/jsonl}"
CACHE_ROOT="${CACHE_ROOT:-${DATA_ROOT}/train/vision_cache}"
SPLIT_ROOT="${SPLIT_ROOT:-/data/hdt/ntv_data/sim_data/data_multi_agent_split_10to1}"
TEST_TARGET_MANIFEST="${TEST_TARGET_MANIFEST:-${SPLIT_ROOT}/split_manifest.json}"

TRAIN_GPUS="${TRAIN_GPUS:-0,1}"
EVAL_GPUS="${EVAL_GPUS:-${TRAIN_GPUS}}"
RENDER_GPUS="${RENDER_GPUS:-${EVAL_GPUS}}"
NUM_GPUS="${NUM_GPUS:-}"

OUT_DIR="${OUT_DIR:-/data/hdt/ntv_data/ckpt/data_multi_agent_base_fast_b32_acc4_lr2e-5_2gpu}"
CKPT_DIR="${CKPT_DIR:-${OUT_DIR}}"
EVAL_ROOT="${EVAL_ROOT:-/data/hdt/ntv_data/sim_data/eval/data_multi_agent_base_fast_b32_acc4_lr2e-5_2gpu_10to1}"

LLM_NAME="${LLM_NAME:-Qwen/Qwen3-0.6B}"
HISTORY="${HISTORY:-31}"
N_WAYPOINTS="${N_WAYPOINTS:-10}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-32}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
LR="${LR:-2e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
BETA_NAV="${BETA_NAV:-100}"
DRONE_LOSS_WEIGHT="${DRONE_LOSS_WEIGHT:-5}"
DOG_LOSS_WEIGHT="${DOG_LOSS_WEIGHT:-1}"
ALPHA_XY="${ALPHA_XY:-1}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
LOG_EVERY="${LOG_EVERY:-10}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
MAX_CKPTS="${MAX_CKPTS:-0}"
MAX_STEPS="${MAX_STEPS:-0}"
SEED="${SEED:-42}"

RUN_DRY_RUN="${RUN_DRY_RUN:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_EVAL_PARALLEL="${RUN_EVAL_PARALLEL:-1}"
MONITOR_GPU="${MONITOR_GPU:-1}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-60}"
TRACKVLA_USE_MODELSCOPE="${TRACKVLA_USE_MODELSCOPE:-0}"
# Same local workaround used by the other training scripts in this repo:
# on this machine GPU 0/1 can hang in the first NCCL all-reduce when P2P is on.
NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

# Eval defaults can still be overridden from the shell.
EVAL_EPISODES="${EVAL_EPISODES:-manifest}"
EVAL_SCENES="${EVAL_SCENES:-}"
EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-600}"
EVAL_SAVE_VIDEO="${EVAL_SAVE_VIDEO:-1}"
EVAL_SCRIPT="${EVAL_SCRIPT:-eval_unrealzoo_multi_agent_base.py}"
EVAL_PLANNER_DEBUG_STEPS="${EVAL_PLANNER_DEBUG_STEPS:-5}"
EVAL_INIT_FROM_RECORDED_AGENT_POSES="${EVAL_INIT_FROM_RECORDED_AGENT_POSES:-1}"

count_csv_items() {
    local value="$1"
    "${PYTHON_BIN}" - "$value" <<'PY'
import sys
items = [x.strip() for x in sys.argv[1].split(",") if x.strip()]
print(len(items))
PY
}

if [[ -z "${NUM_GPUS}" ]]; then
    NUM_GPUS="$(count_csv_items "${TRAIN_GPUS}")"
fi

require_path() {
    local path="$1"
    local label="$2"
    if [[ ! -e "${path}" ]]; then
        echo "[ERROR] ${label} not found: ${path}" >&2
        exit 1
    fi
}

start_gpu_monitor() {
    local log_file="$1"
    if [[ "${MONITOR_GPU}" != "1" ]] || ! command -v nvidia-smi >/dev/null 2>&1; then
        echo ""
        return
    fi
    (
        while true; do
            echo "----- $(date '+%F %T') -----"
            nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader
            sleep "${MONITOR_INTERVAL}"
        done
    ) >> "${log_file}" 2>&1 &
    echo "$!"
}

stop_monitor() {
    local pid="${1:-}"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" >/dev/null 2>&1; then
        kill "${pid}" >/dev/null 2>&1 || true
        wait "${pid}" >/dev/null 2>&1 || true
    fi
}

build_scene_shard() {
    local shard_id="$1"
    local num_shards="$2"
    "${PYTHON_BIN}" - "${TEST_TARGET_MANIFEST}" "${EVAL_SCENES}" "${shard_id}" "${num_shards}" <<'PY'
import json
import sys

manifest_path, requested_scenes, shard_id, num_shards = sys.argv[1:5]
shard_id = int(shard_id)
num_shards = int(num_shards)
with open(manifest_path, "r", encoding="utf-8") as handle:
    manifest = json.load(handle)
scene_counts = manifest.get("scene_counts", {})
if requested_scenes.strip():
    scenes = [item.strip() for item in requested_scenes.split(",") if item.strip()]
else:
    scenes = sorted(scene_counts)
print(",".join(scene for i, scene in enumerate(scenes) if i % num_shards == shard_id))
PY
}

print_config() {
    cat <<EOF
===============================================================================
Fast base multi-agent train/eval
===============================================================================
DATA_ROOT=${DATA_ROOT}
TRAIN_JSON=${TRAIN_JSON}
CACHE_ROOT=${CACHE_ROOT}
SPLIT_ROOT=${SPLIT_ROOT}
TEST_TARGET_MANIFEST=${TEST_TARGET_MANIFEST}
OUT_DIR=${OUT_DIR}
CKPT_DIR=${CKPT_DIR}
EVAL_ROOT=${EVAL_ROOT}
TRAIN_GPUS=${TRAIN_GPUS} NUM_GPUS=${NUM_GPUS}
EVAL_GPUS=${EVAL_GPUS}
NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE} NCCL_IB_DISABLE=${NCCL_IB_DISABLE}
BATCH_SIZE=${BATCH_SIZE}  # per GPU
effective_batch=$((BATCH_SIZE * NUM_GPUS * GRAD_ACCUM_STEPS))
LR=${LR}
EPOCHS=${EPOCHS}
RUN_DRY_RUN=${RUN_DRY_RUN} RUN_TRAIN=${RUN_TRAIN} RUN_EVAL=${RUN_EVAL}
===============================================================================
EOF
}

require_path "${TRAIN_JSON}" "TRAIN_JSON"
require_path "${CACHE_ROOT}" "CACHE_ROOT"
require_path "${TEST_TARGET_MANIFEST}" "TEST_TARGET_MANIFEST"
mkdir -p "${OUT_DIR}" "${EVAL_ROOT}"

RUN_CONFIG_FILE="${OUT_DIR}/run_config.env"
cat > "${RUN_CONFIG_FILE}" <<EOF
created_at=$(date '+%F %T %z')
script=$0
PYTHON_BIN=${PYTHON_BIN}
DATA_ROOT=${DATA_ROOT}
TRAIN_JSON=${TRAIN_JSON}
CACHE_ROOT=${CACHE_ROOT}
SPLIT_ROOT=${SPLIT_ROOT}
TEST_TARGET_MANIFEST=${TEST_TARGET_MANIFEST}
TRAIN_GPUS=${TRAIN_GPUS}
EVAL_GPUS=${EVAL_GPUS}
RENDER_GPUS=${RENDER_GPUS}
NUM_GPUS=${NUM_GPUS}
OUT_DIR=${OUT_DIR}
CKPT_DIR=${CKPT_DIR}
EVAL_ROOT=${EVAL_ROOT}
LLM_NAME=${LLM_NAME}
HISTORY=${HISTORY}
N_WAYPOINTS=${N_WAYPOINTS}
EPOCHS=${EPOCHS}
BATCH_SIZE=${BATCH_SIZE}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS}
EFFECTIVE_BATCH=$((BATCH_SIZE * NUM_GPUS * GRAD_ACCUM_STEPS))
LR=${LR}
WEIGHT_DECAY=${WEIGHT_DECAY}
BETA_NAV=${BETA_NAV}
DRONE_LOSS_WEIGHT=${DRONE_LOSS_WEIGHT}
DOG_LOSS_WEIGHT=${DOG_LOSS_WEIGHT}
ALPHA_XY=${ALPHA_XY}
MIXED_PRECISION=${MIXED_PRECISION}
NUM_WORKERS=${NUM_WORKERS}
PREFETCH_FACTOR=${PREFETCH_FACTOR}
LOG_EVERY=${LOG_EVERY}
SAVE_EVERY=${SAVE_EVERY}
MAX_CKPTS=${MAX_CKPTS}
MAX_STEPS=${MAX_STEPS}
SEED=${SEED}
RUN_DRY_RUN=${RUN_DRY_RUN}
RUN_TRAIN=${RUN_TRAIN}
RUN_EVAL=${RUN_EVAL}
RUN_EVAL_PARALLEL=${RUN_EVAL_PARALLEL}
TRACKVLA_USE_MODELSCOPE=${TRACKVLA_USE_MODELSCOPE}
NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE}
NCCL_IB_DISABLE=${NCCL_IB_DISABLE}
NCCL_DEBUG=${NCCL_DEBUG}
TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING}
OMP_NUM_THREADS=${OMP_NUM_THREADS}
EVAL_EPISODES=${EVAL_EPISODES}
EVAL_SCENES=${EVAL_SCENES}
EVAL_MAX_STEPS=${EVAL_MAX_STEPS}
EVAL_SAVE_VIDEO=${EVAL_SAVE_VIDEO}
EOF

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TRACKVLA_USE_MODELSCOPE="${TRACKVLA_USE_MODELSCOPE}"
export NCCL_P2P_DISABLE NCCL_IB_DISABLE NCCL_DEBUG TORCH_NCCL_ASYNC_ERROR_HANDLING OMP_NUM_THREADS
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"

TRAIN_ARGS=(
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
    --drone_loss_weight "${DRONE_LOSS_WEIGHT}"
    --dog_loss_weight "${DOG_LOSS_WEIGHT}"
    --alpha_xy "${ALPHA_XY}"
    --no-tanh-actions
    --mixed_precision "${MIXED_PRECISION}"
    --num_workers "${NUM_WORKERS}"
    --prefetch_factor "${PREFETCH_FACTOR}"
    --log_every "${LOG_EVERY}"
    --save_every "${SAVE_EVERY}"
    --max_ckpts "${MAX_CKPTS}"
    --max_steps "${MAX_STEPS}"
    --seed "${SEED}"
)

print_config
echo "[config] wrote shell run config: ${RUN_CONFIG_FILE}"

if [[ "${RUN_DRY_RUN}" == "1" ]]; then
    echo "[dry-run] checking one batch/cache path before training..."
    CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" "${PYTHON_BIN}" "${TRAIN_ARGS[@]}" --dry-run
fi

if [[ "${RUN_TRAIN}" == "1" ]]; then
    TRAIN_STDOUT_LOG="${OUT_DIR}/train_stdout.log"
    GPU_MONITOR_LOG="${OUT_DIR}/gpu_monitor.log"
    monitor_pid="$(start_gpu_monitor "${GPU_MONITOR_LOG}")"
    trap 'stop_monitor "${monitor_pid:-}"' EXIT

    echo "[train] starting DDP training, log=${TRAIN_STDOUT_LOG}"
    if [[ "${NUM_GPUS}" -gt 1 ]]; then
        CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" \
        "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node "${NUM_GPUS}" "${TRAIN_ARGS[@]}" \
            2>&1 | tee "${TRAIN_STDOUT_LOG}"
    else
        CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" \
        "${PYTHON_BIN}" "${TRAIN_ARGS[@]}" \
            2>&1 | tee "${TRAIN_STDOUT_LOG}"
    fi
    stop_monitor "${monitor_pid:-}"
    trap - EXIT
else
    echo "[skip] RUN_TRAIN=0"
fi

if [[ "${RUN_EVAL}" == "1" ]]; then
    require_path "${CKPT_DIR}" "CKPT_DIR"
    echo "[eval] using checkpoint dir: ${CKPT_DIR}"

    IFS=',' read -r -a eval_gpu_list <<< "${EVAL_GPUS}"
    if [[ "${RUN_EVAL_PARALLEL}" == "1" && "${#eval_gpu_list[@]}" -gt 1 ]]; then
        echo "[eval] parallel scene shards on GPUs: ${EVAL_GPUS}"
        pids=()
        for shard_id in "${!eval_gpu_list[@]}"; do
            gpu="${eval_gpu_list[$shard_id]}"
            scenes="$(build_scene_shard "${shard_id}" "${#eval_gpu_list[@]}")"
            if [[ -z "${scenes}" ]]; then
                continue
            fi
            log_file="${EVAL_ROOT}/eval_gpu${gpu}.log"
            echo "[eval] shard=${shard_id} gpu=${gpu} scenes=${scenes} log=${log_file}"
            (
                CKPT_DIR="${CKPT_DIR}" \
                SPLIT_ROOT="${SPLIT_ROOT}" \
                TEST_TARGET_MANIFEST="${TEST_TARGET_MANIFEST}" \
                EVAL_ROOT="${EVAL_ROOT}" \
                EVAL_SCENES="${scenes}" \
                EVAL_EPISODES="${EVAL_EPISODES}" \
                EVAL_GPUS="${gpu}" \
                RENDER_GPUS="${gpu}" \
                EVAL_MAX_STEPS="${EVAL_MAX_STEPS}" \
                SAVE_EVAL_VIDEO="${EVAL_SAVE_VIDEO}" \
                EVAL_SCRIPT="${EVAL_SCRIPT}" \
                EVAL_PLANNER_DEBUG_STEPS="${EVAL_PLANNER_DEBUG_STEPS}" \
                EVAL_INIT_FROM_RECORDED_AGENT_POSES="${EVAL_INIT_FROM_RECORDED_AGENT_POSES}" \
                bash sh/run_multi_agent_eval.sh
            ) > "${log_file}" 2>&1 &
            pids+=("$!")
        done
        for pid in "${pids[@]}"; do
            wait "${pid}"
        done
        echo "[eval] final aggregate metrics under ${EVAL_ROOT}"
        "${PYTHON_BIN}" -m tools.calculate_unrealzoo_metrics --eval-dir "${EVAL_ROOT}"
    else
        CKPT_DIR="${CKPT_DIR}" \
        SPLIT_ROOT="${SPLIT_ROOT}" \
        TEST_TARGET_MANIFEST="${TEST_TARGET_MANIFEST}" \
        EVAL_ROOT="${EVAL_ROOT}" \
        EVAL_SCENES="${EVAL_SCENES}" \
        EVAL_EPISODES="${EVAL_EPISODES}" \
        EVAL_GPUS="${EVAL_GPUS}" \
        RENDER_GPUS="${RENDER_GPUS}" \
        EVAL_MAX_STEPS="${EVAL_MAX_STEPS}" \
        SAVE_EVAL_VIDEO="${EVAL_SAVE_VIDEO}" \
        EVAL_SCRIPT="${EVAL_SCRIPT}" \
        EVAL_PLANNER_DEBUG_STEPS="${EVAL_PLANNER_DEBUG_STEPS}" \
        EVAL_INIT_FROM_RECORDED_AGENT_POSES="${EVAL_INIT_FROM_RECORDED_AGENT_POSES}" \
        bash sh/run_multi_agent_eval.sh
    fi
else
    echo "[skip] RUN_EVAL=0"
fi

echo "[done] OUT_DIR=${OUT_DIR}"
echo "[done] EVAL_ROOT=${EVAL_ROOT}"
