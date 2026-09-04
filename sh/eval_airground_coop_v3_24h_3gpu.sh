#!/usr/bin/env bash
set -Eeuo pipefail

#   GPU_IDS_24H=0,1,6 \
#   bash sh/eval_airground_coop_v3_24h_3gpu.sh \
#     --manifest /data/hdt/newtrackvla修改/newtrackvla_base_yh_clean/manifests/total.json \
#     --save-path output/eval_total150_24h_20260823

# Throughput profile validated on three GPUs with the V3 fixed-step evaluator.
# User arguments are appended last, so any default below can be overridden.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_IDS_24H="${GPU_IDS_24H:-0,1,6}"

exec bash "${SCRIPT_DIR}/eval_airground_coop_v3.sh" \
  --gpu-ids "${GPU_IDS_24H}" \
  --workers-per-gpu 1 \
  --render-gpu-ids "${GPU_IDS_24H}" \
  --worker-start-stagger-sec 15 \
  --width 640 \
  --height 480 \
  --yolo-half \
  --policy-inference-stride 5 \
  --policy-action-rollout future_segment \
  --skip-rgb-between-policy-steps \
  --metric-mask-stride 1 \
  --mask-image-format png \
  --reuse-post-action-poses \
  --deterministic-pause-check-stride 50 \
  --robotdog-min-follow-dist 1.0 \
  --robotdog-max-follow-dist 6.0 \
  --drone-min-follow-dist 1.0 \
  --drone-max-follow-dist 6.0 \
  --robotdog-ideal-follow-dist 3.25 \
  --drone-ideal-follow-dist 3.25 \
  --manifest manifests/total.json \
  --save-path output/eval_2hz_640x480 \
  "$@"
