#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/hdt/miniconda3/envs/omtracknew/bin/python}"
DATASET_ROOT="/data/hdt/ntv_data/data/data7_8_camera_m40_pose_fixed_dt_exact_bbox_global_base_split_70_30"
CHECKPOINT="${PROJECT_ROOT}/adapter/saved_ade/models/overlap_100-10_Adapter/step_5/checkpoint-epoch100.pth"
OUTPUT_ROOT="/data/hdt/ntv_data/data/data7_8_camera_m40_pose_fixed_dt_exact_bbox_global_base_split_70_30_adapter_obstacle_step5"
GPU_IDS="3,4"
BATCH_SIZE=16
VISUALIZE_EVERY=60
THRESHOLD=0.5
SPLITS_CSV="train,val"
LIMIT_EPISODES=0
LIMIT_FRAMES_PER_AGENT=0
SAVE_CLASS_LABEL=1
OVERWRITE=0

usage() {
    echo "Usage: bash sh/offline_segment_adapter_obstacles.sh [options]"
    echo
    echo "Options:"
    echo "  --gpu-ids IDS                 Comma-separated physical GPU IDs (default: 3,4)"
    echo "  --batch-size N                Batch size per GPU (default: 16)"
    echo "  --visualize-every N           Save a panel every N frame indices; 0 disables (default: 60)"
    echo "  --threshold P                 Foreground sigmoid threshold (default: 0.5)"
    echo "  --splits train,val            Split manifests to process (default: train,val)"
    echo "  --dataset-root PATH           Split dataset root"
    echo "  --checkpoint PATH             Adapter checkpoint"
    echo "  --output-root PATH            Output dataset root"
    echo "  --no-class-label              Save binary masks only, without ADE class-ID labels"
    echo "  --overwrite                   Recompute files that already exist"
    echo "  --limit-episodes N            Smoke testing only"
    echo "  --limit-frames-per-agent N    Smoke testing only"
    echo "  -h, --help                    Show this help"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu-ids)
            GPU_IDS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --visualize-every)
            VISUALIZE_EVERY="$2"
            shift 2
            ;;
        --threshold)
            THRESHOLD="$2"
            shift 2
            ;;
        --splits)
            SPLITS_CSV="$2"
            shift 2
            ;;
        --dataset-root)
            DATASET_ROOT="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --output-root)
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --no-class-label)
            SAVE_CLASS_LABEL=0
            shift
            ;;
        --overwrite)
            OVERWRITE=1
            shift
            ;;
        --limit-episodes)
            LIMIT_EPISODES="$2"
            shift 2
            ;;
        --limit-frames-per-agent)
            LIMIT_FRAMES_PER_AGENT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 2
fi
if [[ ! -d "${DATASET_ROOT}" ]]; then
    echo "Dataset root not found: ${DATASET_ROOT}" >&2
    exit 2
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "Checkpoint not found: ${CHECKPOINT}" >&2
    exit 2
fi

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
IFS=',' read -r -a SPLIT_ARRAY <<< "${SPLITS_CSV}"
if [[ ${#GPU_ARRAY[@]} -eq 0 ]]; then
    echo "At least one GPU ID is required" >&2
    exit 2
fi
if [[ ${#SPLIT_ARRAY[@]} -eq 0 ]]; then
    echo "At least one split is required" >&2
    exit 2
fi

RUN_ID="adapter_obstacles_$(date +%Y%m%d_%H%M%S)_$$"
LAUNCH_LOG_DIR="${OUTPUT_ROOT}/logs/${RUN_ID}/launcher"
mkdir -p "${LAUNCH_LOG_DIR}"

COMMON_ARGS=(
    "${PROJECT_ROOT}/adapter/offline_segment_dataset_obstacles.py"
    --dataset-root "${DATASET_ROOT}"
    --checkpoint "${CHECKPOINT}"
    --output-root "${OUTPUT_ROOT}"
    --splits "${SPLIT_ARRAY[@]}"
    --batch-size "${BATCH_SIZE}"
    --threshold "${THRESHOLD}"
    --visualize-every "${VISUALIZE_EVERY}"
    --world-size "${#GPU_ARRAY[@]}"
    --run-id "${RUN_ID}"
    --limit-episodes "${LIMIT_EPISODES}"
    --limit-frames-per-agent "${LIMIT_FRAMES_PER_AGENT}"
)
if [[ ${SAVE_CLASS_LABEL} -eq 0 ]]; then
    COMMON_ARGS+=(--no-class-label)
fi
if [[ ${OVERWRITE} -eq 1 ]]; then
    COMMON_ARGS+=(--overwrite)
fi

echo "[config] run_id=${RUN_ID}"
echo "[config] dataset=${DATASET_ROOT}"
echo "[config] checkpoint=${CHECKPOINT}"
echo "[config] output=${OUTPUT_ROOT}"
echo "[config] GPUs=${GPU_IDS} batch_per_gpu=${BATCH_SIZE} visualize_every=${VISUALIZE_EVERY}"
echo "[config] splits=${SPLITS_CSV} resume=$((1 - OVERWRITE)) class_labels=${SAVE_CLASS_LABEL}"

PIDS=()
LOGS=()
cleanup() {
    for pid in "${PIDS[@]:-}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
        fi
    done
}
trap cleanup INT TERM

for rank in "${!GPU_ARRAY[@]}"; do
    gpu="${GPU_ARRAY[$rank]//[[:space:]]/}"
    if [[ -z "${gpu}" ]]; then
        echo "Empty GPU ID in --gpu-ids ${GPU_IDS}" >&2
        cleanup
        exit 2
    fi
    log="${LAUNCH_LOG_DIR}/worker_rank${rank}_gpu${gpu}.log"
    CUDA_VISIBLE_DEVICES="${gpu}" \
        MPLCONFIGDIR="/tmp/adapter-offline-mpl-${USER:-user}-${rank}" \
        "${PYTHON_BIN}" "${COMMON_ARGS[@]}" --rank "${rank}" --device cuda:0 \
        >"${log}" 2>&1 &
    pid=$!
    PIDS+=("${pid}")
    LOGS+=("${log}")
    echo "[worker] rank=${rank} gpu=${gpu} pid=${pid} log=${log}"
done

FAILED=0
for index in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$index]}"; then
        FAILED=1
        echo "[error] worker failed: ${LOGS[$index]}" >&2
        cleanup
    fi
done
trap - INT TERM

if [[ ${FAILED} -ne 0 ]]; then
    echo "[error] one or more workers failed; inspect ${LAUNCH_LOG_DIR}" >&2
    exit 1
fi

echo "[done] all workers completed successfully"
echo "[done] masks=${OUTPUT_ROOT}/masks"
echo "[done] labels=${OUTPUT_ROOT}/labels"
echo "[done] visualizations=${OUTPUT_ROOT}/visualizations"
echo "[done] metadata=${OUTPUT_ROOT}/dataset_info.json"
