#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/hdt/newtrackvla修改/newtrackvla_base_yh_clean"
PYTHON="/home/hdt/miniconda3/envs/omtracknew/bin/python"
PLAN="$ROOT/manifests/data_arr_at_v1/evaluation.json"
SOURCE="/data/hdt/ntv_data/sim_data/data_arr"
OUTPUT="/data/hdt/ntv_data/data_arr_at_v1"

case "${1:-}" in
  0) SHARD=0; WORKER=worker3 ;;
  1) SHARD=1; WORKER=worker4 ;;
  4) SHARD=2; WORKER=worker0 ;;
  5) SHARD=3; WORKER=worker1 ;;
  6) SHARD=4; WORKER=worker2 ;;
  *) echo "usage: $0 GPU (one of 0,1,4,5,6)" >&2; exit 2 ;;
esac
GPU="$1"
RUNTIME="/data/hdt/ntv_data/sim_data/unreal_env_workers/at_rerender_gpu016/$WORKER"
exec env CUDA_VISIBLE_DEVICES="$GPU" \
  UNREALZOO_FAST_ENV_ID="UnrealTrack-DowntownWest-ContinuousColor-v0" \
  UNREALZOO_ENV_BIN="$RUNTIME/Linux/UnrealZoo_UE5_6/Binaries/Linux/UnrealZoo_UE5_6" \
  UNREALZOO_PORT_LOCK="/tmp/at_eval_gpu${GPU}_ports_5way.lock" \
  UNREALZOO_FIXED_TIMESTEP=0.1 \
  PYTHONDONTWRITEBYTECODE=1 \
  "$PYTHON" "$ROOT/tools/replay_data_arr_at.py" \
    --plan "$PLAN" \
    --source-root "$SOURCE" \
    --output-root "$OUTPUT" \
    --render-gpu "$GPU" \
    --shard-index "$SHARD" \
    --shard-count 5 \
    --exclude-scenes KoreanPalace \
    --episode-retries 2 \
    --episode-timeout-s 900 \
    --snapshot-mode batch \
    --snapshot-attempts 3 \
    --snapshot-render-sync 0.02 \
    --resume
