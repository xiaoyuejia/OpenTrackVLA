#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# 对 hand/data 的 2:1 训练集完成视觉缓存，然后在物理 GPU 1 训练 100 epoch。

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export DINOV3_MODEL_PATH="${DINOV3_MODEL_PATH:-/data/hdt/ntv_data/weights/dinov3}"
export PYTHONUNBUFFERED=1

PYTHON_BIN="${PYTHON_BIN:-/home/hdt/miniconda3/envs/omtracknew/bin/python}"
DATA_ROOT="${DATA_ROOT:-/data/hdt/ntv_data/data/unrealzoo_aerial_ground_human_hand_multi_2to1/train}"
TRAIN_JSON="${TRAIN_JSON:-${DATA_ROOT}/dataset.json}"
CACHE_ROOT="${CACHE_ROOT:-${DATA_ROOT}/vision_cache}"
ANCHOR_DIR="${ANCHOR_DIR:-${DATA_ROOT}/trajectory_anchors}"
OUT_DIR="${OUT_DIR:-/data/hdt/ntv_data/ckpt/ckpts_multi_agent_anchor_diffusion_hand_2to1_100ep}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs/hand_2to1_anchor_100ep}"

PRECACHE_WORKERS="${PRECACHE_WORKERS:-4}"
PRECACHE_BATCH_SIZE="${PRECACHE_BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-100}"
RUN_TRAIN_AFTER_PREPROCESS="${RUN_TRAIN_AFTER_PREPROCESS:-0}"
RUN_PRECACHE="${RUN_PRECACHE:-0}"

mkdir -p "${CACHE_ROOT}" "${OUT_DIR}" "${LOG_ROOT}"

echo "[pipeline] GPU=${CUDA_VISIBLE_DEVICES}"
echo "[pipeline] train_json=${TRAIN_JSON}"
echo "[pipeline] cache_root=${CACHE_ROOT}"
echo "[pipeline] out_dir=${OUT_DIR}"

if [[ "${RUN_PRECACHE}" == "1" ]]; then
  pids=()
  for ((shard_id=0; shard_id<PRECACHE_WORKERS; shard_id++)); do
    echo "[precache] starting shard ${shard_id}/${PRECACHE_WORKERS}"
    "${PYTHON_BIN}" -m tools.precache_frames \
      --multi_agent \
      --data_root "${DATA_ROOT}" \
      --cache_root "${CACHE_ROOT}" \
      --dataset_json "${TRAIN_JSON}" \
      --batch_size "${PRECACHE_BATCH_SIZE}" \
      --image_size 384 \
      --device cuda \
      --num_shards "${PRECACHE_WORKERS}" \
      --shard_id "${shard_id}" \
      >"${LOG_ROOT}/precache_shard_${shard_id}.log" 2>&1 &
    pids+=("$!")
  done

  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" -ne 0 ]]; then
    echo "[pipeline] at least one precache shard failed; see ${LOG_ROOT}/precache_shard_*.log" >&2
    exit 1
  fi
else
  echo "[precache] RUN_PRECACHE=0, skip completed visual cache generation"
fi

expected="$("${PYTHON_BIN}" -m tools.precache_frames \
  --multi_agent \
  --data_root "${DATA_ROOT}" \
  --dataset_json "${TRAIN_JSON}" \
  --cache_root "${CACHE_ROOT}" \
  --list_only | sed -n 's/^Frames to check: \([0-9][0-9]*\).*/\1/p')"
fine_count="$(find "${CACHE_ROOT}" -name '*_vfine.pt' | wc -l)"
coarse_count="$(find "${CACHE_ROOT}" -name '*_vcoarse.pt' | wc -l)"
echo "[precache] expected=${expected} fine=${fine_count} coarse=${coarse_count}"
if [[ -z "${expected}" || "${fine_count}" -ne "${expected}" || "${coarse_count}" -ne "${expected}" ]]; then
  echo "[pipeline] visual cache is incomplete; refusing to start training" >&2
  exit 1
fi

if [[ "${RUN_TRAIN_AFTER_PREPROCESS}" != "1" ]]; then
  echo "[pipeline] preprocessing complete; paused before training"
  exit 0
fi

echo "[train] cache complete; starting Anchor Diffusion training for ${EPOCHS} epochs"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
DATA_ROOT="${DATA_ROOT}" \
TRAIN_JSON="${TRAIN_JSON}" \
CACHE_ROOT="${CACHE_ROOT}" \
ANCHOR_DIR="${ANCHOR_DIR}" \
OUT_DIR="${OUT_DIR}" \
RUN_BUILD_ANCHORS=0 \
RUN_DRY_RUN=1 \
RUN_TRAIN=1 \
EPOCHS="${EPOCHS}" \
bash sh/train_anchor_diffusion.sh
