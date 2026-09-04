#!/usr/bin/env bash
set -Eeuo pipefail

# By default this evaluates the canonical scene-covered 100-episode V3
# action-replay manifest.
# Multi-GPU, multiple UE workers per GPU:
#   bash sh/eval_airground_coop_v3.sh --gpu-ids 1,3 --workers-per-gpu 2 \
#     --scene-timeout-seconds 12600
# Random single-scene debug remains available explicitly:
#   bash sh/eval_airground_coop_v3.sh --random-episodes --gpu-ids 1 \
#     --workers-per-gpu 1 --env-id SCENE --episodes 2 --max-steps 100
# 
# Each GPU owns one shared model/vision/YOLO server. UE workers keep isolated
# temporal/controller state and communicate with that server over local Unix IPC.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

# 评估必须使用与离线缓存一致的本地视觉权重，禁止静默回退到受限的在线仓库。
export DINOV3_MODEL_PATH="${DINOV3_MODEL_PATH:-${PROJECT_ROOT}/models/vision/dinov3}"
export SIGLIP_MODEL_PATH="${SIGLIP_MODEL_PATH:-${PROJECT_ROOT}/models/vision/siglip}"
[[ -f "${DINOV3_MODEL_PATH}/config.json" ]] || { echo "[ERROR] local DINOv3 model missing: ${DINOV3_MODEL_PATH}" >&2; exit 1; }
[[ -f "${SIGLIP_MODEL_PATH}/config.json" ]] || { echo "[ERROR] local SigLIP model missing: ${SIGLIP_MODEL_PATH}" >&2; exit 1; }

PYTHON_BIN="${PYTHON_BIN:-/home/yh/miniconda3/envs/newtrackvla/bin/python}"
[[ -x "${PYTHON_BIN}" ]] || { echo "[ERROR] Python not executable: ${PYTHON_BIN}" >&2; exit 1; }
EVAL_PROGRAM="${AIRGROUND_V3_EVAL_PROGRAM:-${PROJECT_ROOT}/eval_airground_coop_v3.py}"
SERVER_PROGRAM="${AIRGROUND_V3_SERVER_PROGRAM:-${PROJECT_ROOT}/eval_airground_coop_v3_server.py}"
SERVER_SOCKET_ENV="${AIRGROUND_V3_SERVER_SOCKET_ENV:-EVAL_AIRGROUND_V3_SERVER_SOCKET}"
SERVER_AUTHKEY_ENV="${AIRGROUND_V3_SERVER_AUTHKEY_ENV:-EVAL_AIRGROUND_V3_SERVER_AUTHKEY}"
SERVER_SESSION_ENV="${AIRGROUND_V3_SERVER_SESSION_ENV:-EVAL_AIRGROUND_V3_SESSION}"
RUNTIME_NAMESPACE="${AIRGROUND_V3_RUNTIME_NAMESPACE:-airground_v3}"
MPL_NAMESPACE="${AIRGROUND_V3_MPL_NAMESPACE:-airground-v3}"
AUTHKEY_PREFIX="${AIRGROUND_V3_AUTHKEY_PREFIX:-eval_airground_v3}"
SOCKET_PREFIX="${AIRGROUND_V3_SOCKET_PREFIX:-evalairgroundv3}"
CKPT="${CKPT:-${PROJECT_ROOT}/output/airground_three_stream_cooperative_v3_receiver_target_qwen06b/model_epoch010_step010830_final.pt}"
SAVE_PATH="${SAVE_PATH:-${PROJECT_ROOT}/output/eval_airground_coop_v3_receiver_target_125}"
TEST_MANIFEST="${TEST_TARGET_MANIFEST:-/data/yh/data/manifests/eval_all_2400_recorded.json}"
GPU_IDS="${GPU_IDS:-${GPU:-1}}"
RENDER_GPU_IDS="${RENDER_GPU_IDS:-}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-${EVAL_WORKERS:-1}}"
SERVER_PER_WORKER="${SERVER_PER_WORKER:-0}"
EVAL_NUMA_NODE="${EVAL_NUMA_NODE:-}"
RUN_TAG="${RUN_TAG:-v3_run_$(date +%Y%m%d_%H%M%S)_$$}"
RUNTIME_ROOT="${RUNTIME_ROOT:-}"
WORKER_START_STAGGER_SEC="${WORKER_START_STAGGER_SEC:-${EVAL_WORKER_START_STAGGER_SEC:-15}}"
SCENE_TIMEOUT_SECONDS="${SCENE_TIMEOUT_SECONDS:-${EVAL_SCENE_TIMEOUT_SECONDS:-12600}}"
SCENE_TERM_GRACE_SECONDS="${SCENE_TERM_GRACE_SECONDS:-${EVAL_SCENE_TERM_GRACE_SECONDS:-45}}"
SCENE_RETRIES="${SCENE_RETRIES:-${EVAL_SCENE_RETRIES:-5}}"
DRY_RUN="${DRY_RUN:-0}"
TOTAL_EPISODES="${TOTAL_EPISODES:-}"
SPLIT_SCENES=0
RESUME=1
USER_ARGS=()

# Consume only launcher options here. Everything else is forwarded unchanged
# to eval_airground_coop_v3.py / eval_unrealzoo_multi_agent.py.
while (( $# > 0 )); do
  case "$1" in
    --gpu-ids|--render-gpu-ids|--workers-per-gpu|--total-episodes|--worker-start-stagger-sec|--scene-timeout-seconds|--scene-term-grace-seconds|--scene-retries|--runtime-root|--manifest|--ckpt|--save-path)
      (( $# >= 2 )) || { echo "[ERROR] $1 needs a value" >&2; exit 2; }
      option="$1"
      value="$2"
      shift 2
      case "${option}" in
        --gpu-ids) GPU_IDS="${value}" ;;
        --render-gpu-ids) RENDER_GPU_IDS="${value}" ;;
        --workers-per-gpu) WORKERS_PER_GPU="${value}" ;;
        --total-episodes) TOTAL_EPISODES="${value}" ;;
        --worker-start-stagger-sec) WORKER_START_STAGGER_SEC="${value}" ;;
        --scene-timeout-seconds) SCENE_TIMEOUT_SECONDS="${value}" ;;
        --scene-term-grace-seconds) SCENE_TERM_GRACE_SECONDS="${value}" ;;
        --scene-retries) SCENE_RETRIES="${value}" ;;
        --runtime-root) RUNTIME_ROOT="${value}" ;;
        --manifest) TEST_MANIFEST="${value}" ;;
        --ckpt) CKPT="${value}" ;;
        --save-path) SAVE_PATH="${value}" ;;
      esac
      ;;
    --gpu-ids=*|--render-gpu-ids=*|--workers-per-gpu=*|--total-episodes=*|--worker-start-stagger-sec=*|--scene-timeout-seconds=*|--scene-term-grace-seconds=*|--scene-retries=*|--runtime-root=*|--manifest=*|--ckpt=*|--save-path=*)
      option="${1%%=*}"
      value="${1#*=}"
      shift
      case "${option}" in
        --gpu-ids) GPU_IDS="${value}" ;;
        --render-gpu-ids) RENDER_GPU_IDS="${value}" ;;
        --workers-per-gpu) WORKERS_PER_GPU="${value}" ;;
        --total-episodes) TOTAL_EPISODES="${value}" ;;
        --worker-start-stagger-sec) WORKER_START_STAGGER_SEC="${value}" ;;
        --scene-timeout-seconds) SCENE_TIMEOUT_SECONDS="${value}" ;;
        --scene-term-grace-seconds) SCENE_TERM_GRACE_SECONDS="${value}" ;;
        --scene-retries) SCENE_RETRIES="${value}" ;;
        --runtime-root) RUNTIME_ROOT="${value}" ;;
        --manifest) TEST_MANIFEST="${value}" ;;
        --ckpt) CKPT="${value}" ;;
        --save-path) SAVE_PATH="${value}" ;;
      esac
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --split-scenes)
      SPLIT_SCENES=1
      shift
      ;;
    --no-resume)
      RESUME=0
      shift
      ;;
    --random-episodes)
      TEST_MANIFEST=""
      shift
      ;;
    *)
      USER_ARGS+=("$1")
      shift
      ;;
  esac
done
AUTHKEY="${AUTHKEY_PREFIX}_${RUN_TAG}"

# Different physical GPUs may evaluate concurrently.  Give each GPU set its
# own default UE runtime tree so an evaluation on GPU 6 does not incorrectly
# block an independent run on GPU 5.  An explicit --runtime-root still wins.
if [[ -z "${RUNTIME_ROOT}" ]]; then
  GPU_RUNTIME_TAG="${GPU_IDS//,/_}"
  GPU_RUNTIME_TAG="${GPU_RUNTIME_TAG// /_}"
  RUNTIME_ROOT="/data/yh/newtrackvla修改/newtrackvla_base_yh_clean/output/runtime/${RUNTIME_NAMESPACE}_multi_gpu${GPU_RUNTIME_TAG}"
fi

for token in "${USER_ARGS[@]}"; do
  if [[ "${token}" == "--help" || "${token}" == "-h" ]]; then
    cat <<'EOF'
Launcher options:
  --gpu-ids 1,3               physical GPU indices (default: 1)
  --render-gpu-ids 0,2        UE physical render GPU per worker (required when --gpu-ids contains MIG UUIDs)
  --workers-per-gpu 2         parallel Unreal workers on each GPU (default: 1)
  --manifest PATH             fixed action-replay manifest (default: val100)
  --worker-start-stagger-sec 15
  --scene-timeout-seconds 12600
  --scene-retries 5
  --runtime-root PATH         isolated Unreal runtime directory
  --ckpt PATH
  --save-path PATH
  --robotdog-waypoint-y-mode v3_nonholonomic_projection
  --no-resume                 rerun all manifest entries instead of skipping completed ones
  --random-episodes           disable the fixed manifest for single-scene debugging
  --split-scenes              distribute episode IDs from one scene across workers
  --total-episodes 8          random mode only
  --dry-run                    print the worker plan without launching

All remaining options are forwarded to the evaluation program:
EOF
    CUDA_VISIBLE_DEVICES="${GPU_IDS%%,*}" "${PYTHON_BIN}" "${EVAL_PROGRAM}" --help
    exit 0
  fi
done

[[ -f "${CKPT}" ]] || { echo "[ERROR] checkpoint not found: ${CKPT}" >&2; exit 1; }
[[ "${WORKERS_PER_GPU}" =~ ^[1-9][0-9]*$ ]] || { echo "[ERROR] WORKERS_PER_GPU must be positive" >&2; exit 2; }
[[ "${SERVER_PER_WORKER}" =~ ^[01]$ ]] || { echo "[ERROR] SERVER_PER_WORKER must be 0 or 1" >&2; exit 2; }
[[ -z "${EVAL_NUMA_NODE}" || "${EVAL_NUMA_NODE}" =~ ^[0-9]+$ ]] \
  || { echo "[ERROR] EVAL_NUMA_NODE must be empty or a non-negative integer" >&2; exit 2; }
[[ "${RUN_TAG}" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "[ERROR] invalid RUN_TAG=${RUN_TAG}" >&2; exit 2; }
[[ "${SCENE_TIMEOUT_SECONDS}" =~ ^[0-9]+$ ]] || { echo "[ERROR] SCENE_TIMEOUT_SECONDS must be non-negative" >&2; exit 2; }
[[ "${SCENE_TERM_GRACE_SECONDS}" =~ ^[1-9][0-9]*$ ]] || { echo "[ERROR] SCENE_TERM_GRACE_SECONDS must be positive" >&2; exit 2; }
[[ "${SCENE_RETRIES}" =~ ^[1-9][0-9]*$ ]] || { echo "[ERROR] SCENE_RETRIES must be positive" >&2; exit 2; }
[[ "${WORKER_START_STAGGER_SEC}" =~ ^[0-9]+([.][0-9]+)?$ ]] \
  || { echo "[ERROR] WORKER_START_STAGGER_SEC must be a non-negative number" >&2; exit 2; }

get_arg_value() {
  local option="$1" default_value="$2" i token
  for ((i = 0; i < ${#USER_ARGS[@]}; i++)); do
    token="${USER_ARGS[$i]}"
    if [[ "${token}" == "${option}="* ]]; then
      printf '%s\n' "${token#*=}"
      return
    fi
    if [[ "${token}" == "${option}" ]]; then
      ((i + 1 < ${#USER_ARGS[@]})) || { echo "[ERROR] ${option} needs a value" >&2; exit 2; }
      printf '%s\n' "${USER_ARGS[$((i + 1))]}"
      return
    fi
  done
  printf '%s\n' "${default_value}"
}

GPU_IDS_NORMALIZED="${GPU_IDS//,/ }"
read -r -a GPU_ARRAY <<< "${GPU_IDS_NORMALIZED}"
(( ${#GPU_ARRAY[@]} > 0 )) || { echo "[ERROR] GPU_IDS is empty" >&2; exit 2; }
is_mig_device_id() {
  [[ "$1" =~ ^MIG-[A-Za-z0-9-]+$ ]]
}
declare -A SEEN_GPUS=()
for gpu in "${GPU_ARRAY[@]}"; do
  [[ "${gpu}" =~ ^[0-9]+$ ]] || is_mig_device_id "${gpu}" \
    || { echo "[ERROR] invalid GPU index or MIG UUID: ${gpu}" >&2; exit 2; }
  [[ -z "${SEEN_GPUS[$gpu]:-}" ]] || { echo "[ERROR] duplicate GPU index: ${gpu}" >&2; exit 2; }
  SEEN_GPUS["${gpu}"]=1
  if [[ "${DRY_RUN}" != 1 ]]; then
    if is_mig_device_id "${gpu}"; then
      CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -c \
        'import sys, torch; sys.exit(0 if torch.cuda.is_available() and torch.cuda.device_count() == 1 else 1)' \
        >/dev/null 2>&1 \
        || { echo "[ERROR] MIG device ${gpu} is unavailable to ${PYTHON_BIN}" >&2; exit 1; }
    else
      nvidia-smi -i "${gpu}" --query-gpu=index --format=csv,noheader >/dev/null 2>&1 \
        || { echo "[ERROR] GPU ${gpu} is unavailable" >&2; exit 1; }
    fi
  fi
done

TOTAL_WORKERS=$(( ${#GPU_ARRAY[@]} * WORKERS_PER_GPU ))
RECORDED_TARGET_DIR="$(get_arg_value --recorded-target-dir '')"
RECORDED_TARGET_EPISODES="$(get_arg_value --recorded-target-episodes '')"
if [[ -n "${RECORDED_TARGET_DIR}" ]]; then
  TEST_MANIFEST="${RECORDED_TARGET_DIR}"
fi
FIXED_EVAL=0
[[ -z "${TEST_MANIFEST}" ]] || FIXED_EVAL=1
if (( FIXED_EVAL == 1 )) && [[ -n "${RECORDED_TARGET_EPISODES}" ]]; then
  echo "[ERROR] do not pass --recorded-target-episodes in fixed-manifest mode" >&2
  echo "The launcher reads and partitions all IDs from the manifest itself." >&2
  exit 2
fi

mkdir -p "${SAVE_PATH}"/{logs,pids,workers,plan}
exec 9>"${SAVE_PATH}/.eval_${RUNTIME_NAMESPACE}.lock"
flock -n 9 || { echo "[ERROR] evaluation already running under ${SAVE_PATH}" >&2; exit 2; }
mkdir -p "${RUNTIME_ROOT}"
exec 8>"${RUNTIME_ROOT}/.runtime.lock"
flock -n 8 || { echo "[ERROR] UE worker runtime is already in use: ${RUNTIME_ROOT}" >&2; exit 2; }

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/unrealzoo-gym:${PROJECT_ROOT}/unrealzoo-gym/example/DataRecording"
export TOKENIZERS_PARALLELISM=false
export TRACKVLA_USE_MODELSCOPE=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-${PROJECT_ROOT}/offline_detection_segmentation/runtime/ultralytics}"
PORT_LOCK="/tmp/unrealzoo_${RUNTIME_NAMESPACE}_${RUN_TAG}.lock"

declare -a WORKER_GPU=() WORKER_RENDER_GPU=() WORKER_LOCAL_ID=()
RENDER_GPU_ARRAY=()
if [[ -n "${RENDER_GPU_IDS}" ]]; then
  RENDER_GPU_IDS_NORMALIZED="${RENDER_GPU_IDS//,/ }"
  read -r -a RENDER_GPU_ARRAY <<< "${RENDER_GPU_IDS_NORMALIZED}"
  if (( ${#RENDER_GPU_ARRAY[@]} != 1 && ${#RENDER_GPU_ARRAY[@]} != TOTAL_WORKERS )); then
    echo "[ERROR] --render-gpu-ids needs either 1 ID or ${TOTAL_WORKERS} worker IDs" >&2
    exit 2
  fi
  for gpu in "${RENDER_GPU_ARRAY[@]}"; do
    [[ "${gpu}" =~ ^[0-9]+$ ]] || { echo "[ERROR] invalid render GPU index: ${gpu}" >&2; exit 2; }
    if [[ "${DRY_RUN}" != 1 ]]; then
      nvidia-smi -i "${gpu}" --query-gpu=index --format=csv,noheader >/dev/null 2>&1 \
        || { echo "[ERROR] render GPU ${gpu} is unavailable" >&2; exit 1; }
    fi
  done
fi
if (( ${#RENDER_GPU_ARRAY[@]} == 0 )); then
  for gpu in "${GPU_ARRAY[@]}"; do
    if is_mig_device_id "${gpu}"; then
      echo "[ERROR] MIG CUDA IDs require explicit physical --render-gpu-ids (for Vulkan UE), e.g. 0,0,0,0,1,1,1,1" >&2
      exit 2
    fi
  done
fi
for ((wid = 0; wid < TOTAL_WORKERS; wid++)); do
  gpu_slot=$((wid / WORKERS_PER_GPU))
  WORKER_GPU[$wid]="${GPU_ARRAY[$gpu_slot]}"
  if (( ${#RENDER_GPU_ARRAY[@]} == 0 )); then
    WORKER_RENDER_GPU[$wid]="${WORKER_GPU[$wid]}"
  elif (( ${#RENDER_GPU_ARRAY[@]} == 1 )); then
    WORKER_RENDER_GPU[$wid]="${RENDER_GPU_ARRAY[0]}"
  else
    WORKER_RENDER_GPU[$wid]="${RENDER_GPU_ARRAY[$wid]}"
  fi
  WORKER_LOCAL_ID[$wid]=$((wid % WORKERS_PER_GPU))
done

PLAN_DIR="${SAVE_PATH}/plan/${RUN_TAG}"
mkdir -p "${PLAN_DIR}"
PENDING_EPISODES=0
if (( FIXED_EVAL == 1 )); then
  [[ -f "${TEST_MANIFEST}" ]] || { echo "[ERROR] manifest not found: ${TEST_MANIFEST}" >&2; exit 1; }
  plan_args=(
    --manifest "${TEST_MANIFEST}"
    --eval-root "${SAVE_PATH}"
    --plan-dir "${PLAN_DIR}"
    --workers "${TOTAL_WORKERS}"
  )
  (( SPLIT_SCENES == 1 )) && plan_args+=(--split-scenes)
  (( RESUME == 1 )) || plan_args+=(--no-resume)
  "${PYTHON_BIN}" "${PROJECT_ROOT}/tools/plan_recorded_eval.py" "${plan_args[@]}"
  PLAN_SUMMARY="${PLAN_DIR}/plan.json"
  read -r TOTAL_EPISODES PENDING_EPISODES <<< "$("${PYTHON_BIN}" - "${PLAN_SUMMARY}" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
print(summary["total_episodes"], summary["pending_episodes"])
PY
)"
else
  REQUESTED_EPISODES="$(get_arg_value --episodes 2)"
  TOTAL_EPISODES="${TOTAL_EPISODES:-${REQUESTED_EPISODES}}"
  [[ "${TOTAL_EPISODES}" =~ ^[1-9][0-9]*$ ]] || { echo "[ERROR] TOTAL_EPISODES must be positive" >&2; exit 2; }
  if (( TOTAL_EPISODES < TOTAL_WORKERS )); then
    echo "[ERROR] TOTAL_EPISODES=${TOTAL_EPISODES} is smaller than total workers=${TOTAL_WORKERS}" >&2
    exit 2
  fi
  scene="$(get_arg_value --env-id UnrealTrack-DowntownWest-ContinuousColor-v0)"
  base_count=$((TOTAL_EPISODES / TOTAL_WORKERS))
  remainder=$((TOTAL_EPISODES % TOTAL_WORKERS))
  for ((wid = 0; wid < TOTAL_WORKERS; wid++)); do
    count=$((base_count + (wid < remainder ? 1 : 0)))
    printf '%s\t%s\t\n' "${scene}" "${count}" >"${PLAN_DIR}/worker_${wid}.tsv"
  done
  PENDING_EPISODES="${TOTAL_EPISODES}"
fi

echo "[config] checkpoint=${CKPT}"
echo "[config] GPUs=${GPU_IDS} workers_per_gpu=${WORKERS_PER_GPU} total_workers=${TOTAL_WORKERS} server_per_worker=${SERVER_PER_WORKER} numa_node=${EVAL_NUMA_NODE:-auto}"
echo "[config] fixed_manifest=${FIXED_EVAL} manifest=${TEST_MANIFEST:-none}"
echo "[config] total_episodes=${TOTAL_EPISODES} pending=${PENDING_EPISODES} output=${SAVE_PATH}"
for ((wid = 0; wid < TOTAL_WORKERS; wid++)); do
  count="$(awk -F'\t' '{total += $2} END {print total + 0}' "${PLAN_DIR}/worker_${wid}.tsv")"
  groups="$(awk 'END {print NR + 0}' "${PLAN_DIR}/worker_${wid}.tsv")"
  echo "[plan] worker=${wid} gpu=${WORKER_GPU[$wid]} render_gpu=${WORKER_RENDER_GPU[$wid]} local=${WORKER_LOCAL_ID[$wid]} episodes=${count} scene_groups=${groups}"
done
echo "[plan] ${PLAN_DIR}"

if (( FIXED_EVAL == 1 && PENDING_EPISODES == 0 )); then
  echo "[resume] all ${TOTAL_EPISODES} manifest episodes are already complete"
  "${PYTHON_BIN}" -m tools.calculate_unrealzoo_metrics \
    --eval-dir "${SAVE_PATH}" --expected-episodes "${TOTAL_EPISODES}" --require-exact-episodes \
    --output-csv "${SAVE_PATH}/metrics.csv"
  exit 0
fi

if [[ "${DRY_RUN}" == 1 ]]; then
  echo "[dry-run] plan only; no servers or UE workers started"
  exit 0
fi

NUMA_PREFIX=()
if [[ -n "${EVAL_NUMA_NODE}" ]]; then
  command -v numactl >/dev/null 2>&1 || { echo "[ERROR] numactl is required for EVAL_NUMA_NODE" >&2; exit 1; }
  numactl --hardware 2>/dev/null | grep -q "^node ${EVAL_NUMA_NODE} cpus:" \
    || { echo "[ERROR] NUMA node ${EVAL_NUMA_NODE} is unavailable" >&2; exit 1; }
  NUMA_PREFIX=(numactl --cpunodebind="${EVAL_NUMA_NODE}" --preferred="${EVAL_NUMA_NODE}")
fi

echo "[runtime] preparing ${TOTAL_WORKERS} isolated UE runtimes under ${RUNTIME_ROOT}"
WORKERS="${TOTAL_WORKERS}" RUNTIME_ROOT="${RUNTIME_ROOT}" \
  bash "${PROJECT_ROOT}/sh/prepare_unrealzoo_worker_runtimes.sh" \
  >"${SAVE_PATH}/logs/${RUN_TAG}_prepare_runtime.log" 2>&1

declare -A SERVER_PID=() SERVER_SOCKET=() SERVER_GPU=()
declare -a SERVER_KEYS=() WORKER_SERVER_SOCKET=() WORKER_PID=()
PROGRESS_PID=""

# Build the inference-service topology independently of UE workers.  The
# default keeps one shared model service per CUDA device.  SERVER_PER_WORKER=1
# loads one model replica and creates one socket for every worker, which is
# useful when several workers share a large physical GPU and serialization in
# a shared service is more expensive than concurrent model replicas.
if (( SERVER_PER_WORKER == 1 )); then
  for ((wid = 0; wid < TOTAL_WORKERS; wid++)); do
    key="w${wid}"
    SERVER_KEYS+=("${key}")
    SERVER_GPU["${key}"]="${WORKER_GPU[$wid]}"
  done
else
  for ((gpu_slot = 0; gpu_slot < ${#GPU_ARRAY[@]}; gpu_slot++)); do
    key="g${gpu_slot}"
    SERVER_KEYS+=("${key}")
    SERVER_GPU["${key}"]="${GPU_ARRAY[$gpu_slot]}"
  done
fi

terminate_v3_unreal_processes() {
  local signal="$1" pid command
  # UE may create a process group separate from its Python worker. Restrict
  # cleanup to binaries inside this launcher's locked V3 runtime tree.
  while read -r pid command; do
    [[ "${pid}" =~ ^[1-9][0-9]*$ ]] || continue
    case "${command}" in
      "${RUNTIME_ROOT}"/worker*/Linux/UnrealZoo_UE5_6/Binaries/Linux/UnrealZoo_UE5_6*)
        kill -"${signal}" "${pid}" 2>/dev/null || true
        ;;
    esac
  done < <(ps -eo pid=,args=)
}

cleanup() {
  local active pid key
  for active in "${SAVE_PATH}/pids/${RUN_TAG}_worker"*.scene_pgid; do
    [[ -f "${active}" ]] || continue
    pid="$(<"${active}")"
    if [[ "${pid}" =~ ^[1-9][0-9]*$ ]] && kill -0 "${pid}" 2>/dev/null; then
      kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
    fi
  done
  for pid in "${WORKER_PID[@]:-}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "${pid}" 2>/dev/null || true
    fi
  done
  terminate_v3_unreal_processes TERM
  sleep 1
  for active in "${SAVE_PATH}/pids/${RUN_TAG}_worker"*.scene_pgid; do
    [[ -f "${active}" ]] || continue
    pid="$(<"${active}")"
    if [[ "${pid}" =~ ^[1-9][0-9]*$ ]] && kill -0 "${pid}" 2>/dev/null; then
      kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
    fi
    rm -f "${active}"
  done
  for pid in "${WORKER_PID[@]:-}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill -KILL "${pid}" 2>/dev/null || true
    fi
  done
  terminate_v3_unreal_processes KILL
  for key in "${SERVER_KEYS[@]:-}"; do
    pid="${SERVER_PID[$key]:-}"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
    rm -f "${SERVER_SOCKET[$key]:-}"
  done
  rm -f "${PORT_LOCK}"
  if [[ -n "${PROGRESS_PID}" ]] && kill -0 "${PROGRESS_PID}" 2>/dev/null; then
    kill "${PROGRESS_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

for key in "${SERVER_KEYS[@]}"; do
  gpu="${SERVER_GPU[$key]}"
  socket="/tmp/${SOCKET_PREFIX}_${RUN_TAG}_${key}.sock"
  log="${SAVE_PATH}/logs/${RUN_TAG}_server_${key}_gpu${gpu}.log"
  SERVER_SOCKET["${key}"]="${socket}"
  rm -f "${socket}"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${gpu}" \
    "${NUMA_PREFIX[@]}" "${PYTHON_BIN}" -u "${SERVER_PROGRAM}" \
      --socket "${socket}" --authkey "${AUTHKEY}" >"${log}" 2>&1 &
  SERVER_PID["${key}"]=$!
  echo "${SERVER_PID[$key]}" >"${SAVE_PATH}/pids/${RUN_TAG}_server_${key}_gpu${gpu}.pid"
done

for key in "${SERVER_KEYS[@]}"; do
  gpu="${SERVER_GPU[$key]}"
  socket="${SERVER_SOCKET[$key]}"
  pid="${SERVER_PID[$key]}"
  for _ in $(seq 1 300); do
    [[ -S "${socket}" ]] && break
    kill -0 "${pid}" 2>/dev/null || {
      echo "[ERROR] inference server on GPU ${gpu} exited; see logs" >&2
      exit 1
    }
    sleep 0.1
  done
  [[ -S "${socket}" ]] || { echo "[ERROR] server socket timeout: ${socket}" >&2; exit 1; }
  echo "[server] key=${key} gpu=${gpu} pid=${pid} socket=${socket}"
done

for ((wid = 0; wid < TOTAL_WORKERS; wid++)); do
  if (( SERVER_PER_WORKER == 1 )); then
    key="w${wid}"
  else
    key="g$((wid / WORKERS_PER_GPU))"
  fi
  WORKER_SERVER_SOCKET[$wid]="${SERVER_SOCKET[$key]}"
done

FIXED_PROTOCOL_ARGS=()
if (( FIXED_EVAL == 1 )); then
  FIXED_PROTOCOL_ARGS=(
    --max-steps 600
    --max-lost-steps 400
    --max-failure-steps 0
    --failure-warmup-steps 20
    --instruction "Follow the person without collision."
    --joint-instruction "Use both aerial and ground observations to follow the person without collision."
    --width 640 --height 480 --image-size 384
    --vision-resize-mode letterbox --fps 10
    --dt 0.1 --ue-interval-ms 100
    --history-frame-dt 0.1 --waypoint-source-dt 0.1
    --deterministic-step --no-realtime-waypoint-timing
    --offscreen --time-dilation=-1 --disable-ue-input --launch-retries 5
    --no-save-video --no-write-global-video --no-trajectory-overlay
    --trajectory-scale 120 --planner-debug-steps 5
    --fast-eval-io
    --bbox-source none
    --waypoint-index 1 --drone-waypoint-index 1 --robotdog-waypoint-index 1
    --waypoint-control-mode inverse_fixed_dt --waypoint-horizon-steps 7
    --drone-max-speed 2.5 --robotdog-max-speed 2.5
    --drone-velocity-feedback-gain 0 --drone-yaw-feedback-gain 0
    --robotdog-velocity-feedback-gain 0 --robotdog-yaw-feedback-gain 0
    --target-replay-mode action
    --environment-ground-max-speed-mps 100 --ground-acceleration 10000
    --no-require-visual-target --no-open-spawn
    --init-from-recorded-agent-poses --no-init-followers-behind-target
    --robotdog-camera-forward 170 --robotdog-camera-lateral 0
    --robotdog-camera-height 120 --robotdog-camera-fixed-pitch=-8
    --drone-camera-fixed-pitch=-40
  )
fi

# Keep the V3 full-sequence loss tolerance consistent in random-scene debug
# mode too. An explicit user value is preserved.
if (( FIXED_EVAL == 0 )) && [[ -z "$(get_arg_value --max-lost-steps '')" ]]; then
  USER_ARGS+=(--max-lost-steps 400)
fi

run_scene_attempt() {
  local wid="$1" scene="$2" count="$3" ids="$4" attempt="$5"
  local gpu="${WORKER_GPU[$wid]}" local_id="${WORKER_LOCAL_ID[$wid]}"
  local worker_root="${SAVE_PATH}/workers/${RUN_TAG}/gpu${gpu}_worker${local_id}"
  local worker_bin="${RUNTIME_ROOT}/worker${wid}/Linux/UnrealZoo_UE5_6/Binaries/Linux/UnrealZoo_UE5_6"
  local log="${SAVE_PATH}/logs/${RUN_TAG}_worker${wid}_gpu${gpu}.log"
  local scene_root="${worker_root}/${scene}"
  local active_file="${SAVE_PATH}/pids/${RUN_TAG}_worker${wid}.scene_pgid"
  local args=()
  args+=(--ckpt "${CKPT}" --save-path "${scene_root}" --env-id "${scene}" --episodes "${count}")
  # The server sees exactly one CUDA device, always addressed as cuda:0.
  args+=(--device cuda:0 --seed "$((100 + wid))" --render-gpu "${WORKER_RENDER_GPU[$wid]}")
  if (( FIXED_EVAL == 1 )); then
    args+=(--recorded-target-dir "${TEST_MANIFEST}" --recorded-target-episodes "${ids}")
    args+=("${FIXED_PROTOCOL_ARGS[@]}")
  fi
  # Explicit caller options take precedence over fixed-protocol defaults. This
  # permits controlled speed/ablation benchmarks without editing the launcher.
  args+=("${USER_ARGS[@]}")
  mkdir -p "${scene_root}"
  # Keep CUDA inference and Vulkan rendering namespaces separate.  The model
  # server above is intentionally launched with CUDA_VISIBLE_DEVICES=${gpu},
  # where ${gpu} may be a MIG UUID.  Passing that MIG UUID to the UE process
  # makes the NVIDIA Vulkan ICD disappear on this host and causes UE to fall
  # back to the CPU llvmpipe adapter.  UE must instead inherit the physical
  # render GPU index (0/1); the remote planner does not construct a local
  # model, so this does not move inference off the MIG server.
  local worker_cuda_visible="${WORKER_RENDER_GPU[$wid]}"
  local -a ue_env=(
    CUDA_DEVICE_ORDER=PCI_BUS_ID
    VK_ICD_FILENAMES="${UNREALZOO_VULKAN_ICD:-/usr/share/vulkan/icd.d/nvidia_icd.json}"
  )
  # In MIG mode some NVIDIA drivers fail Vulkan initialization when
  # CUDA_VISIBLE_DEVICES is present, even if it names a physical GPU.  Set
  # UNREALZOO_UE_CUDA_VISIBLE_DEVICES=none to omit it for UE; inference
  # servers retain their independent MIG CUDA visibility.
  if [[ "${UNREALZOO_UE_CUDA_VISIBLE_DEVICES:-}" != "none" ]]; then
    ue_env+=(CUDA_VISIBLE_DEVICES="${UNREALZOO_UE_CUDA_VISIBLE_DEVICES:-${worker_cuda_visible}}")
  fi
  setsid env "${ue_env[@]}" \
    UNREALZOO_ENV_BIN="${worker_bin}" \
    UNREALZOO_PORT_LOCK="${PORT_LOCK}" \
    UNREALZOO_FAST_ENV_ID="${scene}" \
    MPLCONFIGDIR="/tmp/matplotlib-${MPL_NAMESPACE}-${RUN_TAG}-worker${wid}" \
    "${SERVER_SOCKET_ENV}=${WORKER_SERVER_SOCKET[$wid]}" \
    "${SERVER_AUTHKEY_ENV}=${AUTHKEY}" \
    "${SERVER_SESSION_ENV}=${RUN_TAG}_worker${wid}_attempt${attempt}" \
    "${NUMA_PREFIX[@]}" "${PYTHON_BIN}" -u "${EVAL_PROGRAM}" "${args[@]}" \
    >>"${log}" 2>&1 &
  local scene_pid=$!
  printf '%s\n' "${scene_pid}" >"${active_file}"

  local elapsed=0 scene_status=0
  while kill -0 "${scene_pid}" 2>/dev/null; do
    if (( SCENE_TIMEOUT_SECONDS > 0 && elapsed >= SCENE_TIMEOUT_SECONDS )); then
      printf '[worker %s] scene=%s timeout=%ss pgid=%s\n' \
        "${wid}" "${scene}" "${SCENE_TIMEOUT_SECONDS}" "${scene_pid}" >>"${log}"
      kill -TERM -- "-${scene_pid}" 2>/dev/null || kill -TERM "${scene_pid}" 2>/dev/null || true
      for ((grace = 0; grace < SCENE_TERM_GRACE_SECONDS; grace++)); do
        kill -0 "${scene_pid}" 2>/dev/null || break
        sleep 1
      done
      kill -KILL -- "-${scene_pid}" 2>/dev/null || true
      wait "${scene_pid}" 2>/dev/null || true
      rm -f "${active_file}"
      return 124
    fi
    sleep 5
    elapsed=$((elapsed + 5))
  done
  if wait "${scene_pid}"; then
    scene_status=0
  else
    scene_status=$?
  fi
  # `wait` is interrupted when the launcher receives Ctrl-C. The Python
  # leader may exit before its UE child, so always check the whole dedicated
  # process group before deleting the only cleanup marker.
  for _ in $(seq 1 10); do
    kill -0 -- "-${scene_pid}" 2>/dev/null || break
    sleep 0.2
  done
  if kill -0 -- "-${scene_pid}" 2>/dev/null; then
    printf '[worker %s] cleaning residual scene process group pgid=%s\n' \
      "${wid}" "${scene_pid}" >>"${log}"
    kill -TERM -- "-${scene_pid}" 2>/dev/null || true
    sleep 1
    kill -KILL -- "-${scene_pid}" 2>/dev/null || true
  fi
  rm -f "${active_file}"
  return "${scene_status}"
}

run_worker() {
  local wid="$1" gpu="${WORKER_GPU[$1]}" local_id="${WORKER_LOCAL_ID[$1]}"
  local plan="${PLAN_DIR}/worker_${wid}.tsv"
  local log="${SAVE_PATH}/logs/${RUN_TAG}_worker${wid}_gpu${gpu}.log"
  : >"${log}"
  printf '[worker %s] gpu=%s local=%s plan=%s\n' "${wid}" "${gpu}" "${local_id}" "${plan}" >>"${log}"
  while IFS=$'\t' read -r scene count ids; do
    [[ -n "${scene}" ]] || continue
    printf '[worker %s] scene=%s episodes=%s\n' "${wid}" "${scene}" "${count}" >>"${log}"
    local scene_status=1
    for ((attempt = 1; attempt <= SCENE_RETRIES; attempt++)); do
      printf '[worker %s] scene=%s attempt=%s/%s\n' \
        "${wid}" "${scene}" "${attempt}" "${SCENE_RETRIES}" >>"${log}"
      if run_scene_attempt "${wid}" "${scene}" "${count}" "${ids}" "${attempt}"; then
        scene_status=0
        break
      fi
      printf '[worker %s] scene=%s failed attempt=%s\n' \
        "${wid}" "${scene}" "${attempt}" >>"${log}"
      sleep $((attempt * 5))
    done
    if (( scene_status != 0 )); then
      printf '[worker %s] scene=%s failed after %s attempts\n' \
        "${wid}" "${scene}" "${SCENE_RETRIES}" >>"${log}"
      return 1
    fi
  done <"${plan}"
  printf '[worker %s] finished\n' "${wid}" >>"${log}"
}

for ((wid = 0; wid < TOTAL_WORKERS; wid++)); do
  run_worker "${wid}" &
  WORKER_PID[$wid]=$!
  echo "${WORKER_PID[$wid]}" >"${SAVE_PATH}/pids/${RUN_TAG}_worker${wid}.pid"
  echo "[worker] id=${wid} gpu=${WORKER_GPU[$wid]} local=${WORKER_LOCAL_ID[$wid]} pid=${WORKER_PID[$wid]} log=${SAVE_PATH}/logs/${RUN_TAG}_worker${wid}_gpu${WORKER_GPU[$wid]}.log"
  if (( wid + 1 < TOTAL_WORKERS )) && [[ "${WORKER_START_STAGGER_SEC}" != 0 ]]; then
    sleep "${WORKER_START_STAGGER_SEC}"
  fi
done

# Report aggregate completion and ETA while workers run.  A result is counted
# only after both its stat and setup JSON exist, so partially written files do
# not inflate progress.  This monitor is intentionally read-only.
progress_start="$(date +%s)"
(
  while :; do
    done_count="$(find "${SAVE_PATH}/workers/${RUN_TAG}" -type f -name '*_setup.json' 2>/dev/null | while read -r s; do [[ -f "${s%_setup.json}.json" ]] && echo 1; done | wc -l)"
    now="$(date +%s)"
    elapsed=$((now - progress_start))
    remaining=$((TOTAL_EPISODES - done_count))
    eta="unknown"
    if (( done_count > 0 && remaining > 0 && elapsed > 0 )); then
      eta_seconds=$((elapsed * remaining / done_count))
      eta="${eta_seconds}s"
    elif (( remaining <= 0 )); then
      eta="0s"
    fi
    printf '[progress] %s/%s complete elapsed=%ss remaining=%s ETA=%s\n' \
      "${done_count}" "${TOTAL_EPISODES}" "${elapsed}" "${remaining}" "${eta}"
    (( remaining <= 0 )) && break
    sleep 30
  done
) &
PROGRESS_PID=$!

status=0
for ((wid = 0; wid < TOTAL_WORKERS; wid++)); do
  pid="${WORKER_PID[$wid]}"
  wait "${pid}" || status=1
done

if (( status != 0 )); then
  echo "[ERROR] one or more workers failed; inspect ${SAVE_PATH}/logs" >&2
  exit 1
fi
if (( FIXED_EVAL == 1 )); then
  "${PYTHON_BIN}" -m tools.calculate_unrealzoo_metrics \
    --eval-dir "${SAVE_PATH}" --expected-episodes "${TOTAL_EPISODES}" --require-exact-episodes \
    --output-csv "${SAVE_PATH}/metrics.csv"
fi
echo "[done] all ${TOTAL_WORKERS} workers finished; results=${SAVE_PATH}/workers/${RUN_TAG}"
