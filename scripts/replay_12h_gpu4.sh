#!/usr/bin/env bash
set -uo pipefail

repo=/data/hdt/newtrackvla修改/newtrackvla_base
python=/home/hdt/miniconda3/envs/omtracknew/bin/python
source_root=/data/hdt/ntv_data/cyj/data/cyj/data/keyboard_collect
new7_output=/data/hdt/ntv_data/sim_data/keyboard_collect_inverse_replay_m40_m8_speed100_gpu4_new7
runtime=/data/hdt/ntv_data/sim_data/unreal_env_workers/gpu4

run_batch() {
  local include=$1
  local output=$2
  "$python" -u "$repo/tools/history/batch_replay_hand_inverse_fixed_dt.py" \
    --source-root "$source_root" \
    --output-root "$output" \
    --python-bin "$python" \
    --render-gpu 4 \
    --worker-env-root "$runtime" \
    --workers 1 \
    --include "$include" \
    --worker-timeout-s 900 \
    --max-retries 0 \
    --snapshot-mode sequential \
    --visual-label-mode replay_mask \
    --post-reset-settle-s 3 \
    --drone-camera-pitch -40 \
    --robotdog-camera-pitch -8 \
    --robotdog-camera-mount 170 0 120 \
    --dt 0.1 \
    --ue-interval-ms 100 \
    --max-steps 300 \
    --human-max-speed-mps 100 \
    --dog-max-speed-mps 100 \
    --drone-max-command-mps 100 \
    --drone-max-speed-mps 100 \
    --drone-max-yaw-command-radps 8 \
    --ground-acceleration 10000 \
    --ground-turn-step-gain 0.4 \
    --ground-translation-delay-steps 1 \
    --ground-position-feedback-time-s 1.0 \
    --ground-speed-model legacy_preview \
    --ground-control-mode source_yaw \
    --training-waypoint-horizon 10 \
    --save-video \
    --no-write-global-video \
    --write-training-records \
    --resume
}

run_batch_maps_except_korean_palace() {
  local batch=$1
  local output=$2
  local env_dir map include
  for env_dir in "$source_root/$batch"/seed_*/*; do
    [[ -d "$env_dir" ]] || continue
    map=${env_dir##*/}
    [[ "$map" == *KoreanPalace* || "$map" == *Map_ChemicalPlant_1* || "$map" == *ModularNeighborhood* ]] && continue
    include=${env_dir#"$source_root"/}
    run_batch "$include" "$output" || true
  done
}

cd "$repo" || exit 1
export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/matplotlib-replay-gpu4-12h

run_batch_maps_except_korean_palace new7 "$new7_output"

# One resume pass catches episodes that failed or hit the watchdog once.
run_batch_maps_except_korean_palace new7 "$new7_output"
