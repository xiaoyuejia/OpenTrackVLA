#!/usr/bin/env bash
set -euo pipefail

# Process data3047 train_raw into the TrackVLA multi-agent training layout,
# precache visual tokens, then train with the current base-concat settings.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/hdt/miniconda3/envs/omtracknew/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python"
fi

SPLIT_ROOT="${SPLIT_ROOT:-/data/hdt/ntv_data/sim_data/data3047_split_150test}"
TRAIN_RAW="${TRAIN_RAW:-${SPLIT_ROOT}/train_raw}"

PROCESSED_ROOT="${PROCESSED_ROOT:-/data/hdt/ntv_data/data/data3047_150test}"
TRAIN_DATA_ROOT="${TRAIN_DATA_ROOT:-${PROCESSED_ROOT}/train}"
TRAIN_JSON="${TRAIN_JSON:-${TRAIN_DATA_ROOT}/jsonl}"
CACHE_ROOT="${CACHE_ROOT:-${TRAIN_DATA_ROOT}/vision_cache}"
OUT_DIR="${OUT_DIR:-/data/hdt/ntv_data/ckpt/data3047_base_concat_no_agent_text_b32_acc4_lr2e-5}"

GPU="${GPU:-0}"
PRECACHE_GPUS="${PRECACHE_GPUS:-${GPU}}"
PRECACHE_BATCH_SIZE="${PRECACHE_BATCH_SIZE:-8}"
PRECACHE_IMAGE_SIZE="${PRECACHE_IMAGE_SIZE:-384}"
PRECACHE_LIMIT="${PRECACHE_LIMIT:-0}"
PRECACHE_DEVICE="${PRECACHE_DEVICE:-}"
PRECACHE_LIST_ONLY="${PRECACHE_LIST_ONLY:-0}"

RUN_MAKE_DATA="${RUN_MAKE_DATA:-1}"
RUN_PRECACHE="${RUN_PRECACHE:-1}"
RUN_DRY_RUN="${RUN_DRY_RUN:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
ALLOW_EXISTING_OUT_DIR="${ALLOW_EXISTING_OUT_DIR:-0}"
SAVE_EVERY="${SAVE_EVERY:-0}"
SAVE_EVERY_EPOCHS="${SAVE_EVERY_EPOCHS:-1}"
MAX_CKPTS="${MAX_CKPTS:-0}"
VAL_JSON="${VAL_JSON:-}"
VAL_CACHE_ROOT="${VAL_CACHE_ROOT:-}"
EVAL_EVERY="${EVAL_EVERY:-0}"
EVAL_BATCHES="${EVAL_BATCHES:-8}"
VAL_BBOX_SOURCE="${VAL_BBOX_SOURCE:-none}"

HISTORY="${HISTORY:-31}"
HORIZON="${HORIZON:-9}"
N_WAYPOINTS="${N_WAYPOINTS:-10}"
DT="${DT:-0.1}"
AGENT1="${AGENT1:-drone}"
AGENT2="${AGENT2:-robotdog}"
ACTION_FIELD="${ACTION_FIELD:-auto}"
INSTRUCTION="${INSTRUCTION:-}"

ONLY_SUCCESS="${ONLY_SUCCESS:-1}"
EXCLUDE_COLLISION="${EXCLUDE_COLLISION:-1}"
SKIP_COLLISION_STEPS="${SKIP_COLLISION_STEPS:-0}"
REQUIRE_VISIBLE="${REQUIRE_VISIBLE:-0}"
MIN_TARGET_VISIBILITY="${MIN_TARGET_VISIBILITY:-0.0}"
MIN_AGENT_FOLLOWING_RATE="${MIN_AGENT_FOLLOWING_RATE:-0.8}"
MIN_TOTAL_STEPS="${MIN_TOTAL_STEPS:-50}"
ALLOW_PARTIAL_HORIZON="${ALLOW_PARTIAL_HORIZON:-0}"
REUSE_EXISTING_FRAMES="${REUSE_EXISTING_FRAMES:-1}"
MAX_EPISODES="${MAX_EPISODES:-0}"
NO_AGGREGATE="${NO_AGGREGATE:-1}"

run_cmd() {
  echo
  echo ">>> $*"
  "$@"
}

print_config() {
  cat <<EOF
===============================================================================
data3047 base-concat pipeline
SPLIT_ROOT=${SPLIT_ROOT}
TRAIN_RAW=${TRAIN_RAW}
TRAIN_DATA_ROOT=${TRAIN_DATA_ROOT}
TRAIN_JSON=${TRAIN_JSON}
CACHE_ROOT=${CACHE_ROOT}
OUT_DIR=${OUT_DIR}
GPU=${GPU}
PRECACHE_GPUS=${PRECACHE_GPUS}
RUN_MAKE_DATA=${RUN_MAKE_DATA}
RUN_PRECACHE=${RUN_PRECACHE}
RUN_DRY_RUN=${RUN_DRY_RUN}
RUN_TRAIN=${RUN_TRAIN}
SAVE_EVERY=${SAVE_EVERY}
SAVE_EVERY_EPOCHS=${SAVE_EVERY_EPOCHS}
MAX_CKPTS=${MAX_CKPTS}
VAL_JSON=${VAL_JSON:-<none>}
EVAL_EVERY=${EVAL_EVERY}
NOTE: test_raw is intentionally not processed.
===============================================================================
EOF
}

[[ -d "${TRAIN_RAW}" ]] || { echo "[ERROR] train_raw not found: ${TRAIN_RAW}" >&2; exit 1; }

print_config

if [[ "${RUN_MAKE_DATA}" == "1" ]]; then
  MAKE_ARGS=(
    "${PYTHON_BIN}" -m tools.make_tracking_data
    --multi_agent
    --input_root "${TRAIN_RAW}"
    --output_root "${TRAIN_DATA_ROOT}"
    --history "${HISTORY}"
    --horizon "${HORIZON}"
    --n_waypoints "${N_WAYPOINTS}"
    --dt "${DT}"
    --agent1 "${AGENT1}"
    --agent2 "${AGENT2}"
    --action_field "${ACTION_FIELD}"
    --min_target_visibility "${MIN_TARGET_VISIBILITY}"
    --min_agent_following_rate "${MIN_AGENT_FOLLOWING_RATE}"
    --min_total_steps "${MIN_TOTAL_STEPS}"
    --max_episodes "${MAX_EPISODES}"
  )
  [[ -n "${INSTRUCTION}" ]] && MAKE_ARGS+=(--instruction "${INSTRUCTION}")
  [[ "${ONLY_SUCCESS}" == "1" ]] && MAKE_ARGS+=(--only_success)
  [[ "${EXCLUDE_COLLISION}" == "1" ]] && MAKE_ARGS+=(--exclude_collision)
  [[ "${SKIP_COLLISION_STEPS}" == "1" ]] && MAKE_ARGS+=(--skip_collision_steps)
  [[ "${REQUIRE_VISIBLE}" == "1" ]] && MAKE_ARGS+=(--require_visible)
  [[ "${ALLOW_PARTIAL_HORIZON}" == "1" ]] && MAKE_ARGS+=(--allow_partial_horizon)
  [[ "${REUSE_EXISTING_FRAMES}" == "0" ]] && MAKE_ARGS+=(--no_reuse_existing_frames)
  [[ "${NO_AGGREGATE}" == "1" ]] && MAKE_ARGS+=(--no_aggregate)
  run_cmd "${MAKE_ARGS[@]}"
else
  echo "[SKIP] RUN_MAKE_DATA=0, skip raw -> frames/jsonl."
fi

if [[ "${RUN_PRECACHE}" == "1" ]]; then
  [[ -d "${TRAIN_JSON}" ]] || { echo "[ERROR] jsonl not found: ${TRAIN_JSON}" >&2; exit 1; }
  mkdir -p "${CACHE_ROOT}"
  IFS=',' read -r -a PRECACHE_GPU_LIST <<< "${PRECACHE_GPUS}"
  PRECACHE_NUM_SHARDS="${PRECACHE_NUM_SHARDS:-${#PRECACHE_GPU_LIST[@]}}"
  if [[ "${PRECACHE_NUM_SHARDS}" -lt 1 ]]; then
    echo "[ERROR] PRECACHE_NUM_SHARDS must be positive." >&2
    exit 1
  fi
  if [[ "${#PRECACHE_GPU_LIST[@]}" -lt "${PRECACHE_NUM_SHARDS}" ]]; then
    echo "[ERROR] PRECACHE_GPUS=${PRECACHE_GPUS} has fewer GPUs than PRECACHE_NUM_SHARDS=${PRECACHE_NUM_SHARDS}." >&2
    exit 1
  fi

  pids=()
  for shard_id in $(seq 0 "$((PRECACHE_NUM_SHARDS - 1))"); do
    gpu_id="${PRECACHE_GPU_LIST[$shard_id]}"
    PRECACHE_ARGS=(
      "${PYTHON_BIN}" -m tools.precache_frames
      --multi_agent
      --data_root "${TRAIN_DATA_ROOT}"
      --cache_root "${CACHE_ROOT}"
      --json_root "${TRAIN_JSON}"
      --batch_size "${PRECACHE_BATCH_SIZE}"
      --image_size "${PRECACHE_IMAGE_SIZE}"
      --limit "${PRECACHE_LIMIT}"
      --num_shards "${PRECACHE_NUM_SHARDS}"
      --shard_id "${shard_id}"
    )
    [[ -n "${PRECACHE_DEVICE}" ]] && PRECACHE_ARGS+=(--device "${PRECACHE_DEVICE}")
    [[ "${PRECACHE_LIST_ONLY}" == "1" ]] && PRECACHE_ARGS+=(--list_only)
    echo
    echo ">>> CUDA_VISIBLE_DEVICES=${gpu_id} ${PRECACHE_ARGS[*]}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${PRECACHE_ARGS[@]}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done
else
  echo "[SKIP] RUN_PRECACHE=0, skip vision cache."
fi

if [[ "${RUN_DRY_RUN}" == "1" || "${RUN_TRAIN}" == "1" ]]; then
  DATA_ROOT="${PROCESSED_ROOT}" \
  TRAIN_JSON="${TRAIN_JSON}" \
  CACHE_ROOT="${CACHE_ROOT}" \
  OUT_DIR="${OUT_DIR}" \
  GPU="${GPU}" \
  RUN_DRY_RUN="${RUN_DRY_RUN}" \
  RUN_TRAIN="${RUN_TRAIN}" \
  ALLOW_EXISTING_OUT_DIR="${ALLOW_EXISTING_OUT_DIR}" \
  SAVE_EVERY="${SAVE_EVERY}" \
  SAVE_EVERY_EPOCHS="${SAVE_EVERY_EPOCHS}" \
  MAX_CKPTS="${MAX_CKPTS}" \
  VAL_JSON="${VAL_JSON}" \
  VAL_CACHE_ROOT="${VAL_CACHE_ROOT}" \
  EVAL_EVERY="${EVAL_EVERY}" \
  EVAL_BATCHES="${EVAL_BATCHES}" \
  VAL_BBOX_SOURCE="${VAL_BBOX_SOURCE}" \
  PYTHON_BIN="${PYTHON_BIN}" \
    bash sh/run_new_paths_multi_agent_base_concat_train.sh
else
  echo "[SKIP] RUN_DRY_RUN=0 and RUN_TRAIN=0, skip base training."
fi
