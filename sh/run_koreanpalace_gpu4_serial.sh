#!/usr/bin/env bash
set -euo pipefail
ROOT="/data/hdt/newtrackvla修改/newtrackvla_base_yh_clean"
PYTHON="/home/hdt/miniconda3/envs/omtracknew/bin/python"
PLAN="$ROOT/manifests/data_arr_at_v1/evaluation.json"
SOURCE="/data/hdt/ntv_data/sim_data/data_arr"
OUTPUT="/data/hdt/ntv_data/data_arr_at_v1"
BINARY="/data/hdt/ntv_data/sim_data/unreal_env_workers/at_rerender_gpu016/worker0/Linux/UnrealZoo_UE5_6/Binaries/Linux/UnrealZoo_UE5_6"
EP_DIR="$OUTPUT/UnrealTrack-KoreanPalace-ContinuousColor-v0/dt_camera3__7__seed_100"

kill_group() {
  local pid="$1"
  kill -TERM -- "-$pid" 2>/dev/null || true
  sleep 5
  kill -KILL -- "-$pid" 2>/dev/null || true
}

run_attempt() {
  local stem="$1" attempt="$2"
  local progress="$EP_DIR/$stem.progress.json" complete="$EP_DIR/$stem.complete.json"
  rm -f "$progress"
  echo "[koreanpalace-gpu4] start stem=$stem attempt=$attempt $(date -Is)"
  setsid env CUDA_VISIBLE_DEVICES=4 \
    UNREALZOO_FAST_ENV_ID=UnrealTrack-KoreanPalace-ContinuousColor-v0 \
    UNREALZOO_ENV_BIN="$BINARY" \
    UNREALZOO_PORT_LOCK=/tmp/koreanpalace_gpu4_serial.lock \
    UNREALZOO_SKIP_FULL_COLOR_DICT=1 \
    UNREALZOO_REQUEST_TIMEOUT_S=120 \
    UNREALZOO_FIXED_TIMESTEP=0.1 PYTHONDONTWRITEBYTECODE=1 \
    "$PYTHON" -u "$ROOT/tools/replay_data_arr_at.py" \
      --plan "$PLAN" \
      --episode-id "dt/UnrealTrack-KoreanPalace-ContinuousColor-v0/dt_camera3__7__seed_100/$stem" \
      --source-root "$SOURCE" --output-root "$OUTPUT" --render-gpu 4 \
      --episode-retries 0 --startup-timeout-s 86400 --episode-timeout-s 5400 \
      --snapshot-mode batch --snapshot-attempts 3 --snapshot-render-sync 0.02 --resume &
  local pid=$!
  for ((tick=1; tick<=60; tick++)); do
    if [[ -f "$progress" || -f "$complete" ]]; then break; fi
    if ! kill -0 "$pid" 2>/dev/null; then wait "$pid" || true; return 1; fi
    sleep 10
  done
  if [[ ! -f "$progress" && ! -f "$complete" ]]; then
    echo "[koreanpalace-gpu4] startup timeout: stem=$stem frames=0"
    kill_group "$pid"; wait "$pid" 2>/dev/null || true; return 1
  fi
  [[ -f "$progress" ]] && echo "[koreanpalace-gpu4] physical stepping confirmed stem=$stem: $(tr -d '\n' < "$progress")"
  wait "$pid" || return 1
  [[ -f "$complete" ]]
}

for stem in 3 6 7 9; do
  [[ -f "$EP_DIR/$stem.complete.json" ]] && continue
  success=0
  for attempt in 1 2 3; do
    if run_attempt "$stem" "$attempt"; then success=1; break; fi
  done
  [[ "$success" == 1 ]] || { echo "[koreanpalace-gpu4] failed stem=$stem after 3 external attempts" >&2; exit 1; }
  echo "[koreanpalace-gpu4] done stem=$stem $(date -Is)"
done

"$PYTHON" "$ROOT/tools/organize_data_final.py"
echo "[koreanpalace-gpu4] data_final catalog refreshed $(date -Is)"
