#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/data/yh/newtrackvla修改/newtrackvla_base_yh_clean
DATA=/data/yh/data/processed
PYTHON=/home/yh/miniconda3/envs/newtrackvla/bin/python
FRAME_LIST="${DATA}/vision_cache/dt_replay_frames.txt"
MARKERS="${FRAME_LIST}.markers"
RUN_DIR="${DATA}/vision_cache/logs/dt_replay_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RUN_DIR}"
exec 9>"${DATA}/vision_cache/.dt_replay.lock"
flock -n 9 || { echo "another DT-replay vision cache job is queued/running"; exit 2; }

echo "QUEUED $(date --iso-8601=seconds) waiting_for_frames_and_eval" | tee "${RUN_DIR}/status.txt"
while ! "${PYTHON}" - "${MARKERS}" <<'PY'
import sys
from pathlib import Path
paths = [Path(x) for x in Path(sys.argv[1]).read_text().splitlines() if x]
raise SystemExit(0 if len(paths) == 1000 and all(p.is_file() for p in paths) else 1)
PY
do
  sleep 60
done
echo "FRAMES_READY $(date --iso-8601=seconds)" | tee -a "${RUN_DIR}/status.txt"

# Do not steal CUDA/Vulkan cycles from an evaluation already in progress.
while pgrep -f '/eval_airground_coop_v3(_server)?\.py' >/dev/null; do
  sleep 60
done
echo "CACHE_RUNNING $(date --iso-8601=seconds)" | tee -a "${RUN_DIR}/status.txt"

pids=()
for shard in 0 1; do
  CUDA_VISIBLE_DEVICES="${shard}" \
  DINOV3_MODEL_PATH="${ROOT}/models/vision/dinov3" \
  SIGLIP_MODEL_PATH="${ROOT}/models/vision/siglip" \
  TOKENIZERS_PARALLELISM=false \
  "${PYTHON}" -u "${ROOT}/repo/tools/precache_frames.py" --multi_agent \
    --data_root "${DATA}" --cache_root "${DATA}/vision_cache" \
    --frame_list "${FRAME_LIST}" --num_shards 2 --shard_id "${shard}" \
    --device cuda --batch_size 64 --encoder_amp bfloat16 \
    --image_workers 8 --save_workers 8 --max_pending_saves 256 \
    >"${RUN_DIR}/gpu${shard}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
if (( status == 0 )); then
  echo "ALL_DONE $(date --iso-8601=seconds)" | tee -a "${RUN_DIR}/status.txt"
else
  echo "FAILED $(date --iso-8601=seconds)" | tee -a "${RUN_DIR}/status.txt"
fi
exit "${status}"
