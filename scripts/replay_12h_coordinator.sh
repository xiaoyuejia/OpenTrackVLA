#!/usr/bin/env bash
set -uo pipefail

workspace=/data/hdt/newtrackvla修改/newtrackvla_base_yh_clean
sim_root=/data/hdt/ntv_data/sim_data
main_output="$sim_root/keyboard_collect_inverse_replay_m40_m8_speed100_gpu7"

bash "$workspace/scripts/replay_12h_gpu3.sh" >"$sim_root/replay_12h_gpu3.log" 2>&1 &
gpu3_pid=$!
bash "$workspace/scripts/replay_12h_gpu4.sh" >"$sim_root/replay_12h_gpu4.log" 2>&1 &
gpu4_pid=$!
bash "$workspace/scripts/replay_12h_gpu5.sh" >"$sim_root/replay_12h_gpu5.log" 2>&1 &
gpu5_pid=$!
bash "$workspace/scripts/replay_12h_gpu6.sh" >"$sim_root/replay_12h_gpu6.log" 2>&1 &
gpu6_pid=$!

printf 'gpu3_pid=%s gpu4_pid=%s gpu5_pid=%s gpu6_pid=%s\n' "$gpu3_pid" "$gpu4_pid" "$gpu5_pid" "$gpu6_pid"
wait "$gpu3_pid"
gpu3_status=$?
wait "$gpu4_pid"
gpu4_status=$?
wait "$gpu5_pid"
gpu5_status=$?
wait "$gpu6_pid"
gpu6_status=$?

# Merge only batch directories; do not overwrite the main aggregate files.
rsync -a --ignore-existing \
  "$sim_root/keyboard_collect_inverse_replay_m40_m8_speed100_gpu3_new9/new9/" \
  "$main_output/new9/"
rsync -a --ignore-existing \
  "$sim_root/keyboard_collect_inverse_replay_m40_m8_speed100_gpu4_new7/new7/" \
  "$main_output/new7/"
rsync -a --ignore-existing \
  "$sim_root/keyboard_collect_inverse_replay_m40_m8_speed100_gpu4_new6/new6/" \
  "$main_output/new6/"

printf 'gpu3_status=%s gpu4_status=%s gpu5_status=%s gpu6_status=%s\n' \
  "$gpu3_status" "$gpu4_status" "$gpu5_status" "$gpu6_status"
