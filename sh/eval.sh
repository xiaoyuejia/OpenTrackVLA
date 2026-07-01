#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# 评估脚本
# -----------------------------------------------------------------------------
# 功能：
# 1. 自动加载训练目录中最新的 model_epoch*.pt。
# 2. 并行跑多个 split。
# 3. Ctrl+C 时清理所有后台评估进程。
#
# 常用覆盖示例：
#   CKPT_DIR=/data/hdt/ntv_data/ckpt/ckpt_stt_filtered NUM_PARALLEL=2 GPUS=0,1 ./eval.sh
#   CKPT=/path/to/model_epochxx_stepxxxx.pt ./eval.sh
#   SAVE_VIDEO=0 CHUNKS=10 ./eval.sh

# -----------------------------------------------------------------------------
# 可调参数
# -----------------------------------------------------------------------------
CHUNKS="${CHUNKS:-30}"                         # 将评估集切成多少份
NUM_PARALLEL="${NUM_PARALLEL:-3}"              # 同时运行多少个评估进程
GPUS_CSV="${GPUS:-0,1,2}"                      # 使用哪些 GPU，逗号分隔
SAVE_PATH="${SAVE_PATH:-/data/hdt/ntv_data/sim_data/eval/stt_train}"    # 评估结果保存目录
EXP_CONFIG="${EXP_CONFIG:-habitat-lab/habitat/config/benchmark/nav/track/track_infer_stt.yaml}"

CKPT_DIR="${CKPT_DIR:-/data/hdt/ntv_data/ckpt/ckpt_stt_filtered}"      # 默认从这里找最新训练权重
CKPT="${CKPT:-}"                               # 手动指定权重路径；非空时优先使用
SAVE_VIDEO="${SAVE_VIDEO:-1}"                  # 是否保存评估视频

export DINOV3_MODEL_PATH="${DINOV3_MODEL_PATH:-/data/hdt/ntv_data/weights/dinov3}"

# 默认评估 legacy .pt checkpoint，因此清掉 HF_MODEL_DIR，避免优先加载 open_trackvla_hf。
# 如确实要评估 HuggingFace 目录，设置 USE_HF=1 HF_MODEL_DIR=/path/to/hf_dir。
if [ "${USE_HF:-0}" != "1" ]; then
    unset HF_MODEL_DIR
fi

die() {
    echo "[ERROR] $*" >&2
    exit 1
}

latest_ckpt() {
    find "$1" -maxdepth 1 -type f -name 'model_epoch*.pt' -printf '%T@ %p\n' \
        | sort -n \
        | tail -1 \
        | cut -d' ' -f2-
}

# -----------------------------------------------------------------------------
# 解析权重和 GPU
# -----------------------------------------------------------------------------
if [ -z "${CKPT}" ]; then
    [ -d "${CKPT_DIR}" ] || die "CKPT_DIR does not exist: ${CKPT_DIR}"
    CKPT="$(latest_ckpt "${CKPT_DIR}")"
fi
[ -n "${CKPT}" ] && [ -f "${CKPT}" ] || die "No checkpoint found. Set CKPT=... or check CKPT_DIR=${CKPT_DIR}"
export CKPT

IFS=',' read -r -a GPUS <<< "${GPUS_CSV}"
[ "${#GPUS[@]}" -ge "${NUM_PARALLEL}" ] || die "GPUS must provide at least NUM_PARALLEL devices"

mkdir -p "${SAVE_PATH}"

echo "=============================================="
echo "OpenTrackVLA eval"
echo "=============================================="
echo "Checkpoint:       ${CKPT}"
echo "Chunks:           ${CHUNKS}"
echo "Parallel jobs:    ${NUM_PARALLEL}"
echo "GPUs:             ${GPUS_CSV}"
echo "Save path:        ${SAVE_PATH}"
echo "Save video:       ${SAVE_VIDEO}"
echo "Exp config:       ${EXP_CONFIG}"
echo "DINOv3 path:      ${DINOV3_MODEL_PATH}"
echo "=============================================="

# -----------------------------------------------------------------------------
# Ctrl+C 清理后台子进程
# -----------------------------------------------------------------------------
pids=()
cleanup() {
    echo
    echo "[eval] Stopping ${#pids[@]} child process(es)..."
    if [ "${#pids[@]}" -gt 0 ]; then
        kill "${pids[@]}" 2>/dev/null || true
        sleep 2
        kill -9 "${pids[@]}" 2>/dev/null || true
    fi
}
trap cleanup INT TERM

# -----------------------------------------------------------------------------
# 分批启动评估任务
# -----------------------------------------------------------------------------
IDX=0
while [ "${IDX}" -lt "${CHUNKS}" ]; do
    pids=()
    for ((i = 0; i < NUM_PARALLEL && IDX < CHUNKS; i++)); do
        GPU_ID="${GPUS[$i]}"
        echo "[eval] Launch split ${IDX}/${CHUNKS} on GPU ${GPU_ID}"

        CUDA_VISIBLE_DEVICES="${GPU_ID}" \
        SAVE_VIDEO="${SAVE_VIDEO}" \
        PYTHONPATH="habitat-lab" \
        python eval.py \
            --split-num "${CHUNKS}" \
            --split-id "${IDX}" \
            --exp-config "${EXP_CONFIG}" \
            --run-type eval \
            --save-path "${SAVE_PATH}" &

        pids+=("$!")
        IDX=$((IDX + 1))
    done

    status=0
    for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then
            status=1
        fi
    done
    [ "${status}" -eq 0 ] || die "One or more eval jobs failed"
done

trap - INT TERM
echo "[eval] All splits finished."
