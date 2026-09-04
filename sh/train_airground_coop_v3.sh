#!/usr/bin/env bash
set -Eeuo pipefail

GPU_IDS="${GPU_IDS:-3,4}"
if [[ "${1:-}" == "--gpu-ids" ]]; then
  if [[ -z "${2:-}" ]]; then
    echo "--gpu-ids requires a comma-separated value, for example 0,1" >&2
    exit 2
  fi
  GPU_IDS="$2"
  shift 2
fi

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${#GPU_ARRAY[@]}}"
PY="${PY:-/home/yh/miniconda3/envs/newtrackvla/bin/python}"
if [[ ! -x "${PY}" ]]; then
  echo "Python executable not found: ${PY}. Set PY=/path/to/your/env/bin/python" >&2
  exit 2
fi

export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export AIRGROUND_INDEX_CACHE_ROOT="${AIRGROUND_INDEX_CACHE_ROOT:-/data/yh/newtrackvla修改/newtrackvla_base_yh_clean/output/index_cache}"
mkdir -p "${AIRGROUND_INDEX_CACHE_ROOT}"

# Keep a TensorBoard server available for every training launch. Reuse an
# existing listener on the default port to avoid duplicate servers.
TB_PORT="${TENSORBOARD_PORT:-6006}"
TB_LOGDIR="${TENSORBOARD_LOGDIR:-/data/yh/newtrackvla修改/newtrackvla_base_yh_clean/output}"
if command -v tensorboard >/dev/null 2>&1; then
  if ! (command -v ss >/dev/null 2>&1 && ss -ltn "sport = :${TB_PORT}" | grep -q LISTEN); then
    mkdir -p "${TB_LOGDIR}"
    nohup tensorboard --logdir "${TB_LOGDIR}" --port "${TB_PORT}" --host 0.0.0.0 \
      >"${TB_LOGDIR}/tensorboard_${TB_PORT}.log" 2>&1 &
    echo "[TENSORBOARD] started port=${TB_PORT} logdir=${TB_LOGDIR}"
  else
    echo "[TENSORBOARD] reused port=${TB_PORT} logdir=${TB_LOGDIR}"
  fi
else
  echo "[TENSORBOARD] command not found; install tensorboard in newtrackvla environment"
fi

CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PY}" -u -m torch.distributed.run \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --local-ranks-filter=0 \
  train_airground_coop_v3.py \
  --distributed \
  "$@"
