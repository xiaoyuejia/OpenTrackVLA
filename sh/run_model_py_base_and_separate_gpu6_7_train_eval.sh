#!/usr/bin/env bash
set -euo pipefail

# Train and evaluate two model.py waypoint-only multi-agent ablations:
#   - GPU 6: shared-context base
#   - GPU 7: separate-context base
#
# Both use /data/hdt/ntv_data/data/data_multi_agent_10to1 by default.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/hdt/miniconda3/envs/omtracknew/bin/python}"
[[ -x "${PYTHON_BIN}" ]] || PYTHON_BIN="python"

DATA_ROOT="${DATA_ROOT:-/data/hdt/ntv_data/data/data_multi_agent_10to1}"
TRAIN_JSON="${TRAIN_JSON:-${DATA_ROOT}/train/jsonl}"
CACHE_ROOT="${CACHE_ROOT:-${DATA_ROOT}/train/vision_cache}"
SPLIT_ROOT="${SPLIT_ROOT:-/data/hdt/ntv_data/sim_data/data_multi_agent_split_10to1}"
TEST_TARGET_MANIFEST="${TEST_TARGET_MANIFEST:-${SPLIT_ROOT}/split_manifest.json}"

BASE_GPU="${BASE_GPU:-6}"
SEPARATE_GPU="${SEPARATE_GPU:-7}"

BASE_OUT_DIR="${BASE_OUT_DIR:-/data/hdt/ntv_data/ckpt/data_multi_agent_model_py_base_marker_b32_acc4_lr2e-5_gpu${BASE_GPU}}"
SEPARATE_OUT_DIR="${SEPARATE_OUT_DIR:-/data/hdt/ntv_data/ckpt/data_multi_agent_model_py_separate_base_marker_b32_acc4_lr2e-5_gpu${SEPARATE_GPU}}"
BASE_EVAL_ROOT="${BASE_EVAL_ROOT:-/data/hdt/ntv_data/sim_data/eval/data_multi_agent_model_py_base_marker_b32_acc4_lr2e-5_gpu${BASE_GPU}_10to1}"
SEPARATE_EVAL_ROOT="${SEPARATE_EVAL_ROOT:-/data/hdt/ntv_data/sim_data/eval/data_multi_agent_model_py_separate_base_marker_b32_acc4_lr2e-5_gpu${SEPARATE_GPU}_10to1}"

LLM_NAME="${LLM_NAME:-Qwen/Qwen3-0.6B}"
HISTORY="${HISTORY:-31}"
N_WAYPOINTS="${N_WAYPOINTS:-10}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-32}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
LR="${LR:-2e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
BETA_NAV="${BETA_NAV:-100}"
DRONE_LOSS_WEIGHT="${DRONE_LOSS_WEIGHT:-2}"
DOG_LOSS_WEIGHT="${DOG_LOSS_WEIGHT:-1}"
ALPHA_XY="${ALPHA_XY:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
COARSE_CACHE_SIZE="${COARSE_CACHE_SIZE:-4096}"
LOG_EVERY="${LOG_EVERY:-10}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
SAVE_EVERY_EPOCHS="${SAVE_EVERY_EPOCHS:-1}"
MAX_CKPTS="${MAX_CKPTS:-3}"
SEED="${SEED:-42}"

RUN_DRY_RUN="${RUN_DRY_RUN:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_EVAL_PARALLEL="${RUN_EVAL_PARALLEL:-1}"

EVAL_EPISODES="${EVAL_EPISODES:-manifest}"
EVAL_SCENES="${EVAL_SCENES:-}"
EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-600}"
EVAL_SAVE_VIDEO="${EVAL_SAVE_VIDEO:-1}"
EVAL_INSTRUCTION="${EVAL_INSTRUCTION:-The aerial drone and the ground robot dog must cooperatively track the same target person. The drone should follow the person from the air, and the robot dog should follow the same person on the ground.}"

TRACKVLA_USE_MODELSCOPE="${TRACKVLA_USE_MODELSCOPE:-0}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

require_path() {
    local path="$1"
    local label="$2"
    if [[ ! -e "${path}" ]]; then
        echo "[ERROR] ${label} not found: ${path}" >&2
        exit 1
    fi
}

write_run_config() {
    local out_dir="$1"
    local variant="$2"
    local gpu="$3"
    mkdir -p "${out_dir}"
    cat > "${out_dir}/run_config.env" <<EOF
created_at=$(date '+%F %T %z')
script=$0
variant=${variant}
gpu=${gpu}
PYTHON_BIN=${PYTHON_BIN}
DATA_ROOT=${DATA_ROOT}
TRAIN_JSON=${TRAIN_JSON}
CACHE_ROOT=${CACHE_ROOT}
SPLIT_ROOT=${SPLIT_ROOT}
TEST_TARGET_MANIFEST=${TEST_TARGET_MANIFEST}
OUT_DIR=${out_dir}
LLM_NAME=${LLM_NAME}
HISTORY=${HISTORY}
N_WAYPOINTS=${N_WAYPOINTS}
EPOCHS=${EPOCHS}
BATCH_SIZE=${BATCH_SIZE}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS}
EFFECTIVE_BATCH=$((BATCH_SIZE * GRAD_ACCUM_STEPS))
LR=${LR}
WEIGHT_DECAY=${WEIGHT_DECAY}
BETA_NAV=${BETA_NAV}
DRONE_LOSS_WEIGHT=${DRONE_LOSS_WEIGHT}
DOG_LOSS_WEIGHT=${DOG_LOSS_WEIGHT}
ALPHA_XY=${ALPHA_XY}
NUM_WORKERS=${NUM_WORKERS}
PREFETCH_FACTOR=${PREFETCH_FACTOR}
COARSE_CACHE_SIZE=${COARSE_CACHE_SIZE}
LOG_EVERY=${LOG_EVERY}
SAVE_EVERY=${SAVE_EVERY}
SAVE_EVERY_EPOCHS=${SAVE_EVERY_EPOCHS}
MAX_CKPTS=${MAX_CKPTS}
SEED=${SEED}
RUN_DRY_RUN=${RUN_DRY_RUN}
RUN_TRAIN=${RUN_TRAIN}
RUN_EVAL=${RUN_EVAL}
EVAL_EPISODES=${EVAL_EPISODES}
EVAL_SCENES=${EVAL_SCENES}
EVAL_MAX_STEPS=${EVAL_MAX_STEPS}
EVAL_SAVE_VIDEO=${EVAL_SAVE_VIDEO}
EVAL_INSTRUCTION=${EVAL_INSTRUCTION}
TRACKVLA_USE_MODELSCOPE=${TRACKVLA_USE_MODELSCOPE}
PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}
OMP_NUM_THREADS=${OMP_NUM_THREADS}
EOF
}

train_variant() {
    local variant="$1"
    local gpu="$2"
    local out_dir="$3"
    local extra_arg="${4:-}"
    local log_file="${out_dir}/train_stdout.log"
    mkdir -p "${out_dir}"
    write_run_config "${out_dir}" "${variant}" "${gpu}"

    local args=(
        train.py
        --multi_agent
        --base_model
        --train_json "${TRAIN_JSON}"
        --out_dir "${out_dir}"
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
        --no_tanh_actions
        --mixed_precision
        --freeze_llm
        --num_workers "${NUM_WORKERS}"
        --prefetch_factor "${PREFETCH_FACTOR}"
        --coarse_cache_size "${COARSE_CACHE_SIZE}"
        --log_every "${LOG_EVERY}"
        --save_every "${SAVE_EVERY}"
        --save_every_epochs "${SAVE_EVERY_EPOCHS}"
        --max_ckpts "${MAX_CKPTS}"
        --seed "${SEED}"
        --no-ddp-find-unused-parameters
    )
    if [[ -n "${extra_arg}" ]]; then
        args+=("${extra_arg}")
    fi

    if [[ "${RUN_DRY_RUN}" == "1" ]]; then
        echo "[dry-run][${variant}] gpu=${gpu}"
        CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${args[@]}" --dry_run
    fi

    if [[ "${RUN_TRAIN}" == "1" ]]; then
        echo "[train][${variant}] gpu=${gpu} out=${out_dir} log=${log_file}"
        CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${args[@]}" 2>&1 | tee "${log_file}"
    else
        echo "[skip][${variant}] RUN_TRAIN=0"
    fi
}

eval_variant() {
    local variant="$1"
    local gpu="$2"
    local ckpt_dir="$3"
    local eval_root="$4"
    local log_file="${eval_root}/eval_stdout.log"
    mkdir -p "${eval_root}"
    echo "[eval][${variant}] gpu=${gpu} ckpt=${ckpt_dir} eval_root=${eval_root} log=${log_file}"
    EVAL_SCRIPT=eval_unrealzoo_multi_agent.py \
    CKPT_DIR="${ckpt_dir}" \
    SPLIT_ROOT="${SPLIT_ROOT}" \
    TEST_TARGET_MANIFEST="${TEST_TARGET_MANIFEST}" \
    EVAL_ROOT="${eval_root}" \
    EVAL_SCENES="${EVAL_SCENES}" \
    EVAL_EPISODES="${EVAL_EPISODES}" \
    EVAL_GPUS="${gpu}" \
    RENDER_GPUS="${gpu}" \
    EVAL_MAX_STEPS="${EVAL_MAX_STEPS}" \
    SAVE_EVAL_VIDEO="${EVAL_SAVE_VIDEO}" \
    EVAL_BBOX_SOURCE=none \
    EVAL_INSTRUCTION="${EVAL_INSTRUCTION}" \
    bash sh/run_multi_agent_eval.sh 2>&1 | tee "${log_file}"
}

wait_all() {
    local status=0
    for pid in "$@"; do
        if ! wait "${pid}"; then
            status=1
        fi
    done
    return "${status}"
}

require_path "${TRAIN_JSON}" "TRAIN_JSON"
require_path "${CACHE_ROOT}" "CACHE_ROOT"
require_path "${TEST_TARGET_MANIFEST}" "TEST_TARGET_MANIFEST"

export TOKENIZERS_PARALLELISM=false
export TRACKVLA_USE_MODELSCOPE
export PYTORCH_CUDA_ALLOC_CONF
export OMP_NUM_THREADS
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"

cat <<EOF
===============================================================================
model.py base vs separate-context train/eval
===============================================================================
DATA_ROOT=${DATA_ROOT}
TRAIN_JSON=${TRAIN_JSON}
CACHE_ROOT=${CACHE_ROOT}
SPLIT_ROOT=${SPLIT_ROOT}
BASE_GPU=${BASE_GPU} BASE_OUT_DIR=${BASE_OUT_DIR}
SEPARATE_GPU=${SEPARATE_GPU} SEPARATE_OUT_DIR=${SEPARATE_OUT_DIR}
BATCH_SIZE=${BATCH_SIZE} GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS} EFFECTIVE_BATCH_PER_RUN=$((BATCH_SIZE * GRAD_ACCUM_STEPS))
LR=${LR} BETA_NAV=${BETA_NAV} DRONE_LOSS_WEIGHT=${DRONE_LOSS_WEIGHT} DOG_LOSS_WEIGHT=${DOG_LOSS_WEIGHT} ALPHA_XY=${ALPHA_XY}
RUN_DRY_RUN=${RUN_DRY_RUN} RUN_TRAIN=${RUN_TRAIN} RUN_EVAL=${RUN_EVAL}
===============================================================================
EOF

train_pids=()
train_variant "shared_base" "${BASE_GPU}" "${BASE_OUT_DIR}" "" &
train_pids+=("$!")
train_variant "separate_base" "${SEPARATE_GPU}" "${SEPARATE_OUT_DIR}" "--separate_agent_context" &
train_pids+=("$!")
wait_all "${train_pids[@]}"

if [[ "${RUN_EVAL}" == "1" ]]; then
    eval_pids=()
    if [[ "${RUN_EVAL_PARALLEL}" == "1" ]]; then
        eval_variant "shared_base" "${BASE_GPU}" "${BASE_OUT_DIR}" "${BASE_EVAL_ROOT}" &
        eval_pids+=("$!")
        eval_variant "separate_base" "${SEPARATE_GPU}" "${SEPARATE_OUT_DIR}" "${SEPARATE_EVAL_ROOT}" &
        eval_pids+=("$!")
        wait_all "${eval_pids[@]}"
    else
        eval_variant "shared_base" "${BASE_GPU}" "${BASE_OUT_DIR}" "${BASE_EVAL_ROOT}"
        eval_variant "separate_base" "${SEPARATE_GPU}" "${SEPARATE_OUT_DIR}" "${SEPARATE_EVAL_ROOT}"
    fi
fi

echo "[done] shared base ckpt: ${BASE_OUT_DIR}"
echo "[done] shared base eval: ${BASE_EVAL_ROOT}"
echo "[done] separate base ckpt: ${SEPARATE_OUT_DIR}"
echo "[done] separate base eval: ${SEPARATE_EVAL_ROOT}"
