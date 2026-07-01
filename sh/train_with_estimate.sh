#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# 使用示例
# -----------------------------------------------------------------------------
# 默认参数都可以在命令行用环境变量覆盖，例如：
#   NUM_GPUS=2 CUDA_VISIBLE_DEVICES=0,1 ./train_with_estimate.sh
#   EPOCHS=10 BATCH_SIZE=4 RESUME=1 ./train_with_estimate.sh
#   DRY_RUN=1 ./train_with_estimate.sh

# -----------------------------------------------------------------------------
# 参数说明
# -----------------------------------------------------------------------------
# TRAIN_JSON:         训练样本路径；可以是单个 .json/.jsonl，也可以是 jsonl shard 目录。
# CACHE_ROOT:         视觉特征缓存目录；读取 *_vcoarse.pt / *_vfine.pt。
# OUT_DIR:            checkpoint、日志、轨迹 npz 等输出目录。
# NUM_GPUS:           使用的 GPU 数量；大于 1 时启用分布式训练。
# CUDA_DEVICES:       物理 GPU 编号列表；隔离模式下 rank0 用第一个，rank1 用第二个。
# BATCH_SIZE:         每张 GPU 的 micro-batch，不是全局 batch。
# GRAD_ACCUM_STEPS:   梯度累积步数；有效 batch = BATCH_SIZE * NUM_GPUS * GRAD_ACCUM_STEPS。
# EPOCHS:             训练 epoch 数。
# N_WAYPOINTS:        模型预测的未来路点数量 N_w。
# HISTORY:            历史观测帧数。
# LR:                 学习率。
# FREEZE_LLM:         是否冻结 Qwen 主干；1 冻结，0 全参数训练。
# VISION_FEAT_DIM:    视觉 token 特征维度 C；若缓存维度不同，train.py 会自动检测并覆盖。
# ALPHA_XY:           XY 轨迹目标缩放系数 alpha；只作用于 x/y，不缩放 yaw。
# BETA_NAV:           导航损失权重 lambda；最终 loss = BETA_NAV * L_nav。
# NUM_WORKERS:        DataLoader worker 数；过大可能增加 CPU/IO 压力。
# MAX_CKPTS:          最多保留 checkpoint 数量，旧 checkpoint 会自动删除。
# LOG_EVERY:          每多少个 optimizer step 打印一次详细训练日志。
# MIXED_PRECISION:    是否启用混合精度；1 开启，0 关闭。
# CSV_LOGGING:        是否写 OUT_DIR/train_log.csv，用于记录指标和估算 ETA。
# SAVE_TRAJECTORIES:  是否每步保存预测/真值轨迹 npz；会占用磁盘。
# PROGRESS:           是否显示 tqdm 进度条。
# DDP_INIT_SYNC:      DDP 构造时是否做初始参数同步；0 尽量跳过以避免初始化卡住。
# ISOLATE_GPUS:       是否隔离 GPU 启动多进程；1 时每个 rank 只看见一张 GPU。
# MASTER_ADDR:        分布式通信主机地址；单机训练保持 127.0.0.1 即可。
# MASTER_PORT:        分布式通信端口；冲突时换一个空闲端口。
# RESUME:             是否从 checkpoint 恢复训练；1 恢复，0 从头训练。
# RESUME_CKPT:        指定恢复 checkpoint；为空时由 train.py 在 OUT_DIR 中找最新。
# SEC_PER_STEP:       手动指定每个 optimizer step 耗时，用于 ETA；为空时读 train_log.csv。
# LOG_TO_FILE:        是否把终端输出保存到日志文件。
# DRY_RUN:            只打印配置和启动命令，不真正训练。

# -----------------------------------------------------------------------------
# 默认参数
# -----------------------------------------------------------------------------
TRAIN_JSON="${TRAIN_JSON:-/data/hdt/ntv_data/data/stt_filtered/jsonl}"
CACHE_ROOT="${CACHE_ROOT:-/data/hdt/ntv_data/data/stt_filtered/vision_cache}"
OUT_DIR="${OUT_DIR:-ckpt_unrealzoo}"

NUM_GPUS="${NUM_GPUS:-2}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
BATCH_SIZE="${BATCH_SIZE:-14}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-7}"
EPOCHS="${EPOCHS:-4}"
N_WAYPOINTS="${N_WAYPOINTS:-10}"
HISTORY="${HISTORY:-31}"
LR="${LR:-2e-5}"
FREEZE_LLM="${FREEZE_LLM:-0}"
VISION_FEAT_DIM="${VISION_FEAT_DIM:-1408}"
# 实际是1536，但预留一些维度以防后续改进增加特征；train.py 会自动检测缓存特征维度并覆盖这个默认值。
ALPHA_XY="${ALPHA_XY:-1}"
BETA_NAV="${BETA_NAV:-100}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_CKPTS="${MAX_CKPTS:-3}"
LOG_EVERY="${LOG_EVERY:-10}"

MIXED_PRECISION="${MIXED_PRECISION:-1}"
CSV_LOGGING="${CSV_LOGGING:-1}"
SAVE_TRAJECTORIES="${SAVE_TRAJECTORIES:-1}"
PROGRESS="${PROGRESS:-1}"
DDP_INIT_SYNC="${DDP_INIT_SYNC:-0}"
ISOLATE_GPUS="${ISOLATE_GPUS:-1}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
RESUME="${RESUME:-0}"
RESUME_CKPT="${RESUME_CKPT:-}"
SEC_PER_STEP="${SEC_PER_STEP:-}"
LOG_TO_FILE="${LOG_TO_FILE:-1}"
DRY_RUN="${DRY_RUN:-0}"

# -----------------------------------------------------------------------------
# 小工具函数
# -----------------------------------------------------------------------------
die() {
    echo "[ERROR] $*" >&2
    exit 1
}

# enabled=1 时追加一个命令行 flag。
add_flag() {
    local enabled="$1"
    local flag="$2"
    if [ "${enabled}" = "1" ]; then
        ARGS+=("${flag}")
    fi
}

# 统计训练样本数量；支持 jsonl 文件、json 文件、以及包含 jsonl/json 的目录。
count_samples() {
    python - "$1" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])

def count_jsonl(p: Path) -> int:
    with p.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())

if path.is_dir():
    jsonl_files = sorted(path.rglob("*.jsonl"))
    if jsonl_files:
        print(sum(count_jsonl(p) for p in jsonl_files), "jsonl_dir")
    else:
        total = 0
        for p in sorted(path.rglob("*.json")):
            try:
                obj = json.load(p.open("r", encoding="utf-8"))
                total += len(obj) if isinstance(obj, list) else 1
            except Exception:
                pass
        print(total, "json_dir")
elif path.suffix == ".jsonl":
    print(count_jsonl(path), "jsonl_file")
elif path.suffix == ".json":
    obj = json.load(path.open("r", encoding="utf-8"))
    print(len(obj) if isinstance(obj, list) else 1, "json_file")
else:
    print(0, "unknown")
PY
}

# 估算单个 optimizer step 的耗时；优先使用 SEC_PER_STEP，否则读取历史 train_log.csv。
estimate_step_time() {
    if [ -n "${SEC_PER_STEP}" ]; then
        echo "${SEC_PER_STEP} from SEC_PER_STEP"
        return
    fi
    if [ -f "${OUT_DIR}/train_log.csv" ]; then
        python - "${OUT_DIR}/train_log.csv" <<'PY'
import csv
import sys
vals = []
try:
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                vals.append(float(row.get("step_time", "")))
            except Exception:
                pass
except Exception:
    pass
vals = vals[-50:]
if vals:
    print(f"{sum(vals) / len(vals):.6f} from train_log.csv")
PY
    fi
}

# -----------------------------------------------------------------------------
# 预检查和训练规模估算
# -----------------------------------------------------------------------------
[ -e "${TRAIN_JSON}" ] || die "TRAIN_JSON does not exist: ${TRAIN_JSON}"
[ "${NUM_GPUS}" -ge 1 ] || die "NUM_GPUS must be >= 1, got ${NUM_GPUS}"
[ -e "${CACHE_ROOT}" ] || echo "[WARN] CACHE_ROOT does not exist yet: ${CACHE_ROOT}"

read -r NUM_SAMPLES DATASET_KIND <<< "$(count_samples "${TRAIN_JSON}")"
[ "${NUM_SAMPLES}" -gt 0 ] || die "No training samples found under ${TRAIN_JSON}"

EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS * GRAD_ACCUM_STEPS))
STEPS_PER_EPOCH=$(((NUM_SAMPLES + EFFECTIVE_BATCH_SIZE - 1) / EFFECTIVE_BATCH_SIZE))
TOTAL_STEPS=$((STEPS_PER_EPOCH * EPOCHS))
CKPT_SAVES=$((TOTAL_STEPS / 100))

ETA_TEXT="unavailable before first run"
read -r EST_SEC_PER_STEP EST_SOURCE <<< "$(estimate_step_time || true)"
if [ -n "${EST_SEC_PER_STEP:-}" ]; then
    ETA_TEXT="$(
        python - "${EST_SEC_PER_STEP}" "${TOTAL_STEPS}" "${EST_SOURCE:-}" <<'PY'
import sys
sec_per_step = float(sys.argv[1])
steps = int(sys.argv[2])
source = " ".join(sys.argv[3:])
total = int(sec_per_step * steps)
print(f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d} ({sec_per_step:.3f}s/step {source})")
PY
    )"
fi

# -----------------------------------------------------------------------------
# 打印启动前摘要，方便确认本次训练配置
# -----------------------------------------------------------------------------
echo "=============================================="
echo "OpenTrackVLA training pre-flight"
echo "=============================================="
echo "Dataset:          ${TRAIN_JSON} (${DATASET_KIND}, ${NUM_SAMPLES} samples)"
echo "Cache root:       ${CACHE_ROOT}"
echo "Output dir:       ${OUT_DIR}"
echo "GPUs:             ${NUM_GPUS} (${CUDA_DEVICES})"
echo "Isolate GPUs:     ${ISOLATE_GPUS}"
echo "Batch/GPU:        ${BATCH_SIZE}"
echo "Grad accum:       ${GRAD_ACCUM_STEPS}"
echo "Effective batch:  ${EFFECTIVE_BATCH_SIZE}"
echo "Epochs:           ${EPOCHS}"
echo "Steps/epoch:      ${STEPS_PER_EPOCH}"
echo "Total steps:      ${TOTAL_STEPS}"
echo "Expected ckpts:   ${CKPT_SAVES} saves, keep latest ${MAX_CKPTS}"
echo "LR:               ${LR}"
echo "Freeze LLM:       ${FREEZE_LLM}"
echo "Waypoints Nw:     ${N_WAYPOINTS}"
echo "Vision feat dim:  ${VISION_FEAT_DIM}"
echo "Alpha XY:         ${ALPHA_XY}"
echo "Beta/lambda:      ${BETA_NAV}"
echo "Log every:        ${LOG_EVERY}"
echo "Diffusion params: T=1000 T_train=50 T_infer=10 N_step=2 M=40 (not used by current train.py)"
echo "ETA:              ${ETA_TEXT}"
echo "Resume:           $([ "${RESUME}" = "1" ] && echo "enabled ${RESUME_CKPT:-latest}" || echo "disabled")"
echo "Dry run:          ${DRY_RUN}"
echo "=============================================="

# -----------------------------------------------------------------------------
# 组装 train.py 参数
# -----------------------------------------------------------------------------
ARGS=(
    --train_json "${TRAIN_JSON}"
    --cache_root "${CACHE_ROOT}"
    --out_dir "${OUT_DIR}"
    --epochs "${EPOCHS}"
    --batch_size "${BATCH_SIZE}"
    --grad_accum_steps "${GRAD_ACCUM_STEPS}"
    --n_waypoints "${N_WAYPOINTS}"
    --history "${HISTORY}"
    --lr "${LR}"
    --vision_feat_dim "${VISION_FEAT_DIM}"
    --alpha_xy "${ALPHA_XY}"
    --beta_nav "${BETA_NAV}"
    --num_workers "${NUM_WORKERS}"
    --max_ckpts "${MAX_CKPTS}"
    --log_every "${LOG_EVERY}"
)

add_flag "${MIXED_PRECISION}" --mixed_precision
add_flag "${FREEZE_LLM}" --freeze_llm
add_flag "${CSV_LOGGING}" --csv_logging
add_flag "${SAVE_TRAJECTORIES}" --save_trajectories
add_flag "${PROGRESS}" --progress

if [ "${RESUME}" = "1" ]; then
    ARGS+=(--resume)
    [ -z "${RESUME_CKPT}" ] || ARGS+=(--resume_ckpt "${RESUME_CKPT}")
fi

# 根据 GPU 数和启动模式决定 launcher。
if [ "${NUM_GPUS}" -gt 1 ]; then
    ARGS+=(--distributed)
    if [ "${ISOLATE_GPUS}" = "1" ]; then
        LAUNCHER=(python train.py)
    else
        LAUNCHER=(torchrun --standalone --nproc_per_node "${NUM_GPUS}" train.py)
    fi
else
    LAUNCHER=(python train.py)
fi

# -----------------------------------------------------------------------------
# 设置分布式环境变量和输出目录
# -----------------------------------------------------------------------------
mkdir -p "${OUT_DIR}"
export DDP_INIT_SYNC="${DDP_INIT_SYNC}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

echo "[RUN] CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} ${LAUNCHER[*]} ${ARGS[*]}"
[ "${DRY_RUN}" = "0" ] || { echo "[DRY_RUN] Not launching training."; exit 0; }

# -----------------------------------------------------------------------------
# 启动训练
# -----------------------------------------------------------------------------
if [ "${NUM_GPUS}" -gt 1 ] && [ "${ISOLATE_GPUS}" = "1" ]; then
    # 隔离模式：手动启动每个 rank，每个 rank 只看到一张 GPU。
    IFS=',' read -r -a DEVICE_IDS <<< "${CUDA_DEVICES}"
    [ "${#DEVICE_IDS[@]}" -ge "${NUM_GPUS}" ] || die "CUDA_VISIBLE_DEVICES must list at least ${NUM_GPUS} devices"

    LOG_FILE="${OUT_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
    echo "[LOG] ${LOG_FILE}"
    echo "[RUN] isolated DDP: each rank sees exactly one GPU"

    pids=()
    cleanup_children() {
        if [ "${#pids[@]}" -gt 0 ]; then
            kill "${pids[@]}" 2>/dev/null || true
        fi
    }
    trap cleanup_children INT TERM

    for rank in $(seq 0 $((NUM_GPUS - 1))); do
        dev="${DEVICE_IDS[$rank]}"
        (
            export CUDA_VISIBLE_DEVICES="${dev}"
            export RANK="${rank}"
            export LOCAL_RANK=0
            export WORLD_SIZE="${NUM_GPUS}"
            export MASTER_ADDR="${MASTER_ADDR}"
            export MASTER_PORT="${MASTER_PORT}"
            "${LAUNCHER[@]}" "${ARGS[@]}" 2>&1 | sed -u "s/^/[rank${rank}] /"
        ) | tee -a "${LOG_FILE}" &
        pids+=("$!")
    done

    status=0
    for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then
            status=1
        fi
    done
    trap - INT TERM
    exit "${status}"
elif [ "${LOG_TO_FILE}" = "1" ]; then
    # 普通模式：直接运行，并同时输出到终端和日志文件。
    LOG_FILE="${OUT_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
    echo "[LOG] ${LOG_FILE}"
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
    "${LAUNCHER[@]}" "${ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
else
    # 不保存日志，只在终端输出。
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
    "${LAUNCHER[@]}" "${ARGS[@]}"
fi
