#!/usr/bin/env bash
set -euo pipefail
run_root=${1:-/data/yh/newtrackvla修改/newtrackvla_base_yh_clean/runs/top8_joint_s5_e5_20260903_024407}
cat "${run_root}/status.txt"
state=$(head -1 "${run_root}/status.txt")
if [[ "${state}" == CANDIDATES_RUNNING* ]]; then
  logs=()
  for shard in 0 1 2 3; do logs+=("${run_root}/candidates_gpu0_shard${shard}.log"); done
  for shard in 4 5 6 7; do logs+=("${run_root}/candidates_gpu1_shard${shard}.log"); done
  for log in "${logs[@]}"; do
    last=$(grep -E '\[candidate-bundle\]|\[done\]' "${log}" | tail -1 || true)
    [[ -n "${last}" ]] && printf '%s: %s\n' "$(basename "${log}")" "${last}"
  done
  count=$(find /data/yh/data/processed/perception_cache/dt -type f -name '*.candidates.npz' 2>/dev/null | wc -l)
  printf 'candidate_bundles=%s/4867\n' "${count}"
elif [[ "${state}" == AT_CANDIDATES_RUNNING* ]]; then
  for log in "${run_root}"/at_candidates_gpu*_shard*.log; do
    [[ -f "${log}" ]] || continue
    last=$(grep -E '\[candidate-bundle\]|\[done\]' "${log}" | tail -1 || true)
    [[ -n "${last}" ]] && printf '%s: %s\n' "$(basename "${log}")" "${last}"
  done
elif [[ "${state}" == AT_VISION_RUNNING* ]]; then
  logs=()
  for shard in 0 1 2 3; do logs+=("${run_root}/at_vision_gpu0_shard${shard}.log"); done
  for shard in 4 5 6 7; do logs+=("${run_root}/at_vision_gpu1_shard${shard}.log"); done
  for log in "${logs[@]}"; do
    [[ -f "${log}" ]] || continue
    grep -E 'progress checked|Completed precache' "${log}" | tail -1 || true
  done
elif [[ -f "${run_root}/train.log" ]]; then
  grep -E '\[TRAIN\]|\[VAL\]|\[DONE\]|Traceback' "${run_root}/train.log" | tail -8 || true
fi
