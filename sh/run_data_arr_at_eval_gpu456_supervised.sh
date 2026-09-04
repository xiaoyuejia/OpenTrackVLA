#!/usr/bin/env bash
set -euo pipefail
ROOT="/data/hdt/newtrackvla修改/newtrackvla_base_yh_clean"; GPU="${1:?GPU 4,5,6 required}"
case "$GPU" in 4) SHARD=0;WORKER=worker0;;5) SHARD=1;WORKER=worker1;;6) SHARD=2;WORKER=worker2;;*) exit 2;;esac
exec /home/hdt/miniconda3/envs/omtracknew/bin/python -u "$ROOT/tools/supervise_data_arr_at_replay.py" \
 --plan "$ROOT/manifests/data_arr_at_v1/evaluation.json" --source-root /data/hdt/ntv_data/sim_data/data_arr \
 --output-root /data/hdt/ntv_data/data_arr_at_v1 --render-gpu "$GPU" --shard-index "$SHARD" --shard-count 3 \
 --worker-runtime "/data/hdt/ntv_data/sim_data/unreal_env_workers/at_rerender_gpu016/$WORKER" \
 --port-lock "/tmp/at_eval_gpu${GPU}_supervised.lock" --exclude-scenes UnrealTrack-KoreanPalace-ContinuousColor-v0 \
 --startup-timeout-s 600 --heartbeat-timeout-s 300 --active-timeout-s 5400 --attempts 3 \
 --log "/data/hdt/ntv_data/data_arr_at_v1/logs_supervised/gpu${GPU}.log"
