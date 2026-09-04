#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/hdt/newtrackvla修改/newtrackvla_base_yh_clean"
PYTHON="/home/hdt/miniconda3/envs/omtracknew/bin/python"
PLAN="$ROOT/manifests/data_arr_at_v1/evaluation.json"
SOURCE="/data/hdt/ntv_data/sim_data/data_arr"
OUTPUT="/data/hdt/ntv_data/data_arr_at_v1"
RUNTIMES="/data/hdt/ntv_data/sim_data/unreal_env_workers/at_rerender_gpu016"
LOG_DIR="$OUTPUT/logs_gpu456"
mkdir -p "$LOG_DIR"

launch_worker() {
  local gpu="$1"
  local shard="$2"
  local worker="$3"
  local binary="$RUNTIMES/$worker/Linux/UnrealZoo_UE5_6/Binaries/Linux/UnrealZoo_UE5_6"
  local log="$LOG_DIR/gpu${gpu}.log"
  local pidfile="$LOG_DIR/gpu${gpu}.pid"

  if [[ -f "$pidfile" ]] && kill -0 "$(<"$pidfile")" 2>/dev/null; then
    echo "GPU $gpu worker already running pid=$(<"$pidfile")"
    return
  fi

  env \
    CUDA_VISIBLE_DEVICES="$gpu" \
    UNREALZOO_FAST_ENV_ID="UnrealTrack-DowntownWest-ContinuousColor-v0" \
    UNREALZOO_ENV_BIN="$binary" \
    UNREALZOO_PORT_LOCK="/tmp/at_eval_gpu${gpu}_ports.lock" \
    UNREALZOO_FIXED_TIMESTEP=0.1 \
    PYTHONDONTWRITEBYTECODE=1 \
    nohup "$PYTHON" "$ROOT/tools/replay_data_arr_at.py" \
      --plan "$PLAN" \
      --source-root "$SOURCE" \
      --output-root "$OUTPUT" \
      --render-gpu "$gpu" \
      --shard-index "$shard" \
      --shard-count 3 \
      --episode-retries 2 \
      --snapshot-mode batch \
      --snapshot-attempts 3 \
      --snapshot-render-sync 0.02 \
      --resume \
      >"$log" 2>&1 &
  echo "$!" >"$pidfile"
  echo "started GPU $gpu shard $shard/3 pid=$! log=$log"
}

run_foreground() {
  local gpu="$1"
  local shard worker
  case "$gpu" in
    4) shard=0; worker=worker0 ;;
    5) shard=1; worker=worker1 ;;
    6) shard=2; worker=worker2 ;;
    *) echo "unsupported GPU: $gpu" >&2; return 2 ;;
  esac
  local binary="$RUNTIMES/$worker/Linux/UnrealZoo_UE5_6/Binaries/Linux/UnrealZoo_UE5_6"
  exec env \
    CUDA_VISIBLE_DEVICES="$gpu" \
    UNREALZOO_FAST_ENV_ID="UnrealTrack-DowntownWest-ContinuousColor-v0" \
    UNREALZOO_ENV_BIN="$binary" \
    UNREALZOO_PORT_LOCK="/tmp/at_eval_gpu${gpu}_ports.lock" \
    UNREALZOO_FIXED_TIMESTEP=0.1 \
    PYTHONDONTWRITEBYTECODE=1 \
    "$PYTHON" "$ROOT/tools/replay_data_arr_at.py" \
      --plan "$PLAN" \
      --source-root "$SOURCE" \
      --output-root "$OUTPUT" \
      --render-gpu "$gpu" \
      --shard-index "$shard" \
      --shard-count 3 \
      --episode-retries 2 \
      --snapshot-mode batch \
      --snapshot-attempts 3 \
      --snapshot-render-sync 0.02 \
      --resume
}

if [[ "${1:-}" == "--foreground" ]]; then
  run_foreground "${2:?GPU is required}"
fi

launch_worker 4 0 worker0
launch_worker 5 1 worker1
launch_worker 6 2 worker2
