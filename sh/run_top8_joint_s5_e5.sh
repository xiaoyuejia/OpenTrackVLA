#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/data/yh/newtrackvla修改/newtrackvla_base_yh_clean
REPO=${ROOT}/repo
PY=/home/yh/miniconda3/envs/newtrackvla/bin/python
FRAME_ROOT=/data/yh/data/processed/frames/dt
AT_FRAME_ROOT=/data/yh/data/processed/frames/at
CANDIDATE_ROOT=/data/yh/data/processed/perception_cache
VISION_CACHE_ROOT=/data/yh/data/processed/vision_cache
RUN_TAG=${RUN_TAG:-top8_joint_s5_e5_$(date +%Y%m%d_%H%M%S)}
RUN_ROOT=${ROOT}/runs/${RUN_TAG}
OUT_DIR=/data/yh/newtrackvla修改/newtrackvla_base_yh_clean/output/${RUN_TAG}
STATUS=${RUN_ROOT}/status.txt

mkdir -p "${RUN_ROOT}" "${CANDIDATE_ROOT}" "${ROOT}/index_cache"
exec 9>"${ROOT}/runs/.top8_joint_s5_e5.lock"
flock -n 9 || { echo "another top8 preprocess/train job is active" >&2; exit 2; }

busy_pids=''
for _ in {1..12}; do
  busy_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d' || true)
  [[ -z "${busy_pids}" ]] && break
  sleep 5
done
if [[ -n "${busy_pids}" ]]; then
  printf 'BLOCKED busy_gpu_pids=%s time=%s\n' "${busy_pids//$'\n'/,}" "$(date --iso-8601=seconds)" | tee "${STATUS}"
  exit 3
fi

cd "${REPO}"
frame_list=${ROOT}/cache_lists/at_eval_frames.txt
if [[ "${SKIP_PREPROCESS:-0}" != 1 ]]; then
printf 'CANDIDATES_RUNNING started=%s top_k=8 imgsz=640 conf=0.01 shards=8 workers_per_gpu=4\n' "$(date --iso-8601=seconds)" | tee "${STATUS}"
pids=()
gpu_ids=(0 0 0 0 1 1 1 1)
for shard in 0 1 2 3 4 5 6 7; do
  gpu_id=${gpu_ids[$shard]}
  CUDA_VISIBLE_DEVICES=${gpu_id} "${PY}" -u -m tools.precompute_person_candidate_bundles \
    --frame-root "${FRAME_ROOT}" \
    --data-root /data/yh/data/processed \
    --output-root "${CANDIDATE_ROOT}" \
    --device cuda:0 --batch-size 48 --image-size 640 --confidence 0.01 \
    --top-k 8 --num-shards 8 --shard-index "${shard}" \
    >"${RUN_ROOT}/candidates_gpu${gpu_id}_shard${shard}.log" 2>&1 &
  pids+=("$!")
done

candidate_status=0
for pid in "${pids[@]}"; do wait "${pid}" || candidate_status=1; done
if (( candidate_status != 0 )); then
  printf 'CANDIDATES_FAILED time=%s\n' "$(date --iso-8601=seconds)" | tee "${STATUS}"
  exit 4
fi

expected=$(find "${FRAME_ROOT}" -type f -name '*.jpg' -printf '%h\n' | sort -u | wc -l)
actual=$(find "${CANDIDATE_ROOT}/frames/dt" -type f -name '*.candidates.npz' | wc -l)
if [[ "${actual}" -ne "${expected}" ]]; then
  printf 'CANDIDATES_INCOMPLETE expected=%s actual=%s time=%s\n' "${expected}" "${actual}" "$(date --iso-8601=seconds)" | tee "${STATUS}"
  exit 5
fi
printf 'CANDIDATES_DONE expected=%s actual=%s time=%s\n' "${expected}" "${actual}" "$(date --iso-8601=seconds)" | tee "${STATUS}"

if [[ ! -s "${frame_list}" ]]; then
  "${PY}" -m tools.build_eval_at_frame_list --output "${frame_list}"
fi
printf 'AT_CANDIDATES_RUNNING started=%s top_k=8 imgsz=640 conf=0.01 shards=8\n' "$(date --iso-8601=seconds)" | tee "${STATUS}"
pids=()
for shard in 0 1 2 3 4 5 6 7; do
  gpu_id=${gpu_ids[$shard]}
  CUDA_VISIBLE_DEVICES=${gpu_id} "${PY}" -u -m tools.precompute_person_candidate_bundles \
    --frame-root "${AT_FRAME_ROOT}" --frame-list "${frame_list}" --data-root /data/yh/data/processed \
    --output-root "${CANDIDATE_ROOT}" --device cuda:0 --batch-size 48 \
    --image-size 640 --confidence 0.01 --top-k 8 --num-shards 8 --shard-index "${shard}" \
    >"${RUN_ROOT}/at_candidates_gpu${gpu_id}_shard${shard}.log" 2>&1 &
  pids+=("$!")
done
candidate_status=0
for pid in "${pids[@]}"; do wait "${pid}" || candidate_status=1; done
if (( candidate_status != 0 )); then printf 'AT_CANDIDATES_FAILED time=%s\n' "$(date --iso-8601=seconds)" | tee "${STATUS}"; exit 7; fi
if ! "${PY}" -m tools.verify_top8_cache --frame-list "${frame_list}" --candidate-root "${CANDIDATE_ROOT}"; then printf 'AT_CANDIDATES_INCOMPLETE\n' | tee "${STATUS}"; exit 8; fi
printf 'AT_VISION_RUNNING started=%s frames=%s shards=8 workers_per_gpu=4\n' "$(date --iso-8601=seconds)" "$(wc -l <"${frame_list}")" | tee "${STATUS}"
pids=()
vision_gpu_ids=(0 0 0 0 1 1 1 1)
for shard in 0 1 2 3 4 5 6 7; do
  gpu_id=${vision_gpu_ids[$shard]}
  CUDA_VISIBLE_DEVICES=${gpu_id} \
  DINOV3_MODEL_PATH=${ROOT}/models/vision/dinov3 \
  SIGLIP_MODEL_PATH=${ROOT}/models/vision/siglip \
  "${PY}" -u -m tools.precache_frames --multi_agent \
    --data_root /data/yh/data/processed --cache_root "${VISION_CACHE_ROOT}" \
    --frame_list "${frame_list}" --batch_size 64 --device cuda \
    --encoder_amp bfloat16 --image_workers 8 --save_workers 8 \
    --num_shards 8 --shard_id "${shard}" \
    >"${RUN_ROOT}/at_vision_gpu${gpu_id}_shard${shard}.log" 2>&1 &
  pids+=("$!")
done
vision_status=0
for pid in "${pids[@]}"; do wait "${pid}" || vision_status=1; done
if (( vision_status != 0 )); then printf 'AT_VISION_FAILED time=%s\n' "$(date --iso-8601=seconds)" | tee "${STATUS}"; exit 9; fi
if ! "${PY}" -m tools.verify_top8_cache --frame-list "${frame_list}" --candidate-root "${CANDIDATE_ROOT}" --vision-root "${VISION_CACHE_ROOT}"; then printf 'AT_VISION_INCOMPLETE\n' | tee "${STATUS}"; exit 10; fi
printf 'AT_CACHE_DONE vision_frames=%s time=%s\n' "$(wc -l <"${frame_list}")" "$(date --iso-8601=seconds)" | tee "${STATUS}"
else
  expected=$(find "${FRAME_ROOT}" -type f -name '*.jpg' -printf '%h\n' | sort -u | wc -l)
  actual=$(find "${CANDIDATE_ROOT}/frames/dt" -type f -name '*.candidates.npz' | wc -l)
  [[ "${actual}" -eq "${expected}" ]] || { echo "DT candidate cache incomplete: ${actual}/${expected}"; exit 11; }
  "${PY}" -m tools.verify_top8_cache --frame-list "${frame_list}" --candidate-root "${CANDIDATE_ROOT}" --vision-root "${VISION_CACHE_ROOT}"
  printf 'CACHE_REUSED dt_views=%s at_frames=%s time=%s\n' "${actual}" "$(wc -l <"${frame_list}")" "$(date --iso-8601=seconds)" | tee "${STATUS}"
fi

busy_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d' || true)
if [[ -n "${busy_pids}" ]]; then
  printf 'TRAIN_BLOCKED busy_gpu_pids=%s time=%s\n' "${busy_pids//$'\n'/,}" "$(date --iso-8601=seconds)" | tee "${STATUS}"
  exit 6
fi

printf 'TRAIN_RUNNING started=%s out=%s stride=5 epochs=5 batch_per_gpu=40 grad_accum=2 effective_batch=160\n' "$(date --iso-8601=seconds)" "${OUT_DIR}" | tee "${STATUS}"
export AIRGROUND_INDEX_CACHE_ROOT=${ROOT}/index_cache
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set +e
numactl --cpunodebind=1 --membind=1 \
  env PY="${PY}" bash sh/train_airground_coop_v3.sh --gpu-ids 0,1 \
    --config config/airground_cooperative_tracking_v3_yh.yaml \
    --batch-size 40 --grad-accum-steps 2 --lr 3.333e-5 \
    --num-workers 8 --epochs 5 --temporal-stride 5 --out-dir "${OUT_DIR}" \
    2>&1 | tee "${RUN_ROOT}/train.log"
train_status=${PIPESTATUS[0]}
set -e
if (( train_status == 0 )); then state=TRAIN_DONE; else state=TRAIN_FAILED; fi
printf '%s exit=%s time=%s out=%s\n' "${state}" "${train_status}" "$(date --iso-8601=seconds)" "${OUT_DIR}" | tee "${STATUS}"
exit "${train_status}"
