#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_IDS="${GPU_IDS:-0,1,3,6}"
BATCH_SIZE="${BATCH_SIZE:-128}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-128}"
MAX_GPU_MEMORY_USED_MIB="${MAX_GPU_MEMORY_USED_MIB:-2048}"
STREAM_OUTPUT="${STREAM_OUTPUT:-1}"
OUTPUT_ROOT="${PROJECT_ROOT}/offline_detection_segmentation/outputs/full_cache"
RUNTIME_ROOT="${PROJECT_ROOT}/offline_detection_segmentation/runtime"

mkdir -p "${OUTPUT_ROOT}" "${RUNTIME_ROOT}/ultralytics"

if [[ ! "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "BATCH_SIZE must be a positive integer: ${BATCH_SIZE}" >&2
    exit 2
fi
if [[ ! "${MAX_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_BATCH_SIZE must be a positive integer: ${MAX_BATCH_SIZE}" >&2
    exit 2
fi
if (( BATCH_SIZE > MAX_BATCH_SIZE )); then
    echo "Refusing unsafe batch size ${BATCH_SIZE}; tested maximum is ${MAX_BATCH_SIZE} per GPU." >&2
    echo "Use BATCH_SIZE=128. Batch 512 caused CUDA launch timeouts on GPU 3 and GPU 6." >&2
    exit 2
fi

# Hold this descriptor for the entire run. Child workers inherit it, so a second
# launcher cannot slip through while the first group is still discovering frames.
LOCK_PATH="${OUTPUT_ROOT}/run_full.lock"
exec 9>"${LOCK_PATH}"
if ! flock -n 9; then
    echo "Refusing to start: another run_full.sh worker group is already active (${LOCK_PATH})." >&2
    exit 2
fi

GPU_IDS_NORMALIZED="${GPU_IDS//,/ }"
read -r -a GPU_ARRAY <<< "${GPU_IDS_NORMALIZED}"
if (( ${#GPU_ARRAY[@]} == 0 )); then
    echo "GPU_IDS must contain at least one physical GPU index." >&2
    exit 2
fi

declare -A SEEN_GPUS=()
for GPU_ID in "${GPU_ARRAY[@]}"; do
    if [[ ! "${GPU_ID}" =~ ^[0-9]+$ ]]; then
        echo "Invalid physical GPU index in GPU_IDS: ${GPU_ID}" >&2
        exit 2
    fi
    if [[ -n "${SEEN_GPUS[${GPU_ID}]:-}" ]]; then
        echo "Duplicate physical GPU index in GPU_IDS: ${GPU_ID}" >&2
        exit 2
    fi
    SEEN_GPUS["${GPU_ID}"]=1

    GPU_MEMORY_USED="$(nvidia-smi -i "${GPU_ID}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
    if (( GPU_MEMORY_USED > MAX_GPU_MEMORY_USED_MIB )); then
        echo "Refusing to start: physical GPU ${GPU_ID} is using ${GPU_MEMORY_USED} MiB (> ${MAX_GPU_MEMORY_USED_MIB} MiB)." >&2
        echo "Stop the old worker or choose idle cards with GPU_IDS=1,3,6." >&2
        exit 2
    fi
done

cd "${PROJECT_ROOT}"
export YOLO_CONFIG_DIR="${RUNTIME_ROOT}/ultralytics"

NUM_SHARDS="${#GPU_ARRAY[@]}"
PIDS=()

terminate_workers() {
    trap - INT TERM
    echo "Stopping ${#PIDS[@]} cache workers..." >&2
    for PID in "${PIDS[@]}"; do
        kill "${PID}" 2>/dev/null || true
    done
    wait || true
    exit 130
}
trap terminate_workers INT TERM

for SHARD_INDEX in "${!GPU_ARRAY[@]}"; do
    GPU_ID="${GPU_ARRAY[${SHARD_INDEX}]}"
    LOG_PATH="${OUTPUT_ROOT}/run_gpu${GPU_ID}_shard${SHARD_INDEX}of${NUM_SHARDS}_batch${BATCH_SIZE}.log"
    echo "[launch] shard=${SHARD_INDEX}/${NUM_SHARDS} physical_gpu=${GPU_ID} batch=${BATCH_SIZE} log=${LOG_PATH}"
    echo "[run] $(date --iso-8601=seconds) shard=${SHARD_INDEX}/${NUM_SHARDS} physical_gpu=${GPU_ID} batch=${BATCH_SIZE}" >> "${LOG_PATH}"
    if [[ "${STREAM_OUTPUT}" == "1" ]]; then
        CUDA_VISIBLE_DEVICES="${GPU_ID}" \
            /home/hdt/miniconda3/envs/omtracknew/bin/python -u \
            -m offline_detection_segmentation.precompute \
            --config "${PROJECT_ROOT}/offline_detection_segmentation/config.yaml" \
            --output-root "${OUTPUT_ROOT}" \
            --device cuda:0 \
            --batch-size "${BATCH_SIZE}" \
            --num-shards "${NUM_SHARDS}" \
            --shard-index "${SHARD_INDEX}" \
            "$@" \
            > >(tee -a "${LOG_PATH}" | sed -u "s/^/[gpu${GPU_ID} shard${SHARD_INDEX}] /") 2>&1 &
    else
        CUDA_VISIBLE_DEVICES="${GPU_ID}" \
            /home/hdt/miniconda3/envs/omtracknew/bin/python -u \
            -m offline_detection_segmentation.precompute \
            --config "${PROJECT_ROOT}/offline_detection_segmentation/config.yaml" \
            --output-root "${OUTPUT_ROOT}" \
            --device cuda:0 \
            --batch-size "${BATCH_SIZE}" \
            --num-shards "${NUM_SHARDS}" \
            --shard-index "${SHARD_INDEX}" \
            "$@" >> "${LOG_PATH}" 2>&1 &
    fi
    PIDS+=("$!")
done

EXIT_STATUS=0
for PID in "${PIDS[@]}"; do
    if ! wait "${PID}"; then
        EXIT_STATUS=1
    fi
done
trap - INT TERM

if (( EXIT_STATUS != 0 )); then
    echo "One or more cache workers failed; inspect the per-GPU logs above." >&2
else
    echo "All ${NUM_SHARDS} cache workers completed successfully."
fi
exit "${EXIT_STATUS}"
