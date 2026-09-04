#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/data/hdt/newtrackvla修改/newtrackvla_base_yh_clean"
DATA_ROOT="/data/hdt/ntv_data/data/cyj_data_arr_processed"
STAGE_ROOT="/tmp/cyj_perception_input"
OUTPUT_ROOT="${DATA_ROOT}/perception_cache"
RECORD_LIST="${STAGE_ROOT}/frames.txt"
PYTHON_BIN="/home/hdt/miniconda3/envs/omtracknew/bin/python"
NUM_SHARDS=6
BATCH_SIZE="${BATCH_SIZE:-256}"
# Two independent model workers per requested GPU keep the 80 GiB cards busy
# while the other worker is decoding JPEGs/writing npz files.
GPU_IDS=(0 0 1 1 6 6)

mkdir -p "${STAGE_ROOT}/train/jsonl" "${OUTPUT_ROOT}" "${OUTPUT_ROOT}/logs"
ln -sfn "${DATA_ROOT}/frames" "${STAGE_ROOT}/train/frames"
ln -sfn "${DATA_ROOT}/frames" "${STAGE_ROOT}/frames"

# pathlib.rglob does not recurse through directory symlinks. Link individual
# JSONL files instead, while preserving dt/stt/scene/episode relative paths.
for source_root in dt stt; do
  target_root="${STAGE_ROOT}/train/jsonl/${source_root}"
  rm -rf "${target_root}"
  mkdir -p "${target_root}"
  while IFS= read -r -d '' source_file; do
    relative_file="${source_file#${DATA_ROOT}/jsonl/${source_root}/}"
    target_file="${target_root}/${relative_file}"
    mkdir -p "$(dirname "${target_file}")"
    ln -s "${source_file}" "${target_file}"
  done < <(find "${DATA_ROOT}/jsonl/${source_root}" -type f -name '*.jsonl' -print0)
done

# Build one deterministic list once; workers then shard this list instead of
# reparsing all 2.7M JSONL rows independently.
if [[ ! -s "${RECORD_LIST}" ]]; then
  find "${DATA_ROOT}/frames" -type f -name '*.jpg' -printf 'frames/%P\n' > "${RECORD_LIST}"
fi

LOCK_PATH="${OUTPUT_ROOT}/.run_cyj.lock"
exec 9>"${LOCK_PATH}"
flock -n 9 || { echo "another cyj perception run is active" >&2; exit 2; }
printf 'RUNNING started=%s shards=%s batch=%s\n' "$(date --iso-8601=seconds)" "${NUM_SHARDS}" "${BATCH_SIZE}" > "${OUTPUT_ROOT}/perception_cache.status"

PIDS=()
for shard_index in 0 1 2 3 4 5; do
  gpu_id="${GPU_IDS[$shard_index]}"
  log_path="${OUTPUT_ROOT}/logs/gpu${gpu_id}_shard${shard_index}of${NUM_SHARDS}.log"
  echo "[launch] gpu=${gpu_id} shard=${shard_index}/${NUM_SHARDS} batch=${BATCH_SIZE} log=${log_path}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" \
    "${PYTHON_BIN}" -u -m offline_detection_segmentation.precompute \
      --config "${PROJECT_ROOT}/offline_detection_segmentation/config_cyj_data.yaml" \
      --dataset-root "${DATA_ROOT}" \
      --output-root "${OUTPUT_ROOT}" \
      --splits train \
      --record-list "${RECORD_LIST}" \
      --device cuda:0 \
      --batch-size "${BATCH_SIZE}" \
      --num-shards "${NUM_SHARDS}" \
      --shard-index "${shard_index}" \
      >>"${log_path}" 2>&1 &
  PIDS+=("$!")
done

status=0
for pid in "${PIDS[@]}"; do
  wait "${pid}" || status=1
done

if (( status == 0 )); then
  printf 'ALL_DONE status=0 %s\n' "$(date --iso-8601=seconds)" > "${OUTPUT_ROOT}/perception_cache.status"
else
  printf 'FAILED status=1 %s\n' "$(date --iso-8601=seconds)" > "${OUTPUT_ROOT}/perception_cache.status"
fi
exit "${status}"
