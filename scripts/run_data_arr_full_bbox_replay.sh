#!/usr/bin/env bash
set -euo pipefail

gpu="$1"
shard="$2"
slot="$3"
repo="/data/hdt/newtrackvla修改/newtrackvla_base"
source_root="/data/hdt/ntv_data/cyj/data_arr_quarantine_bbox_full_failure_20260821"
selection_root="/data/hdt/newtrackvla修改/newtrackvla_base_yh_clean/reports/data_arr_full_bbox_failure_selection"
output_root="/data/hdt/ntv_data/cyj/data_arr_exact_bbox_replay_gpu${gpu}"
worker_root="/data/hdt/ntv_data/sim_data/unreal_env_workers/gpu${gpu}"
python_bin="/home/hdt/miniconda3/envs/omtracknew/bin/python"

exec "$python_bin" -u "$repo/tools/history/batch_replay_hand_inverse_fixed_dt.py" \
  --source-root "$source_root" \
  --selection-root "$selection_root" \
  --output-root "$output_root" \
  --python-bin "$python_bin" \
  --render-gpu "$gpu" \
  --worker-env-root "$worker_root" \
  --workers 1 \
  --worker-slot-offset "$slot" \
  --num-shards 2 \
  --shard-index "$shard" \
  --worker-timeout-s 420 \
  --max-retries 2 \
  --snapshot-retries 2 \
  --snapshot-mode sequential \
  --snapshot-render-sync-s 0.08 \
  --visual-label-mode replay_mask \
  --post-reset-settle-s 5 \
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
