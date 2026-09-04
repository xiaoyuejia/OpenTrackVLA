#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/data/yh/newtrackvla修改/newtrackvla_base_yh_clean
REPO=${ROOT}/repo
PY=${PYTHON_BIN:-/home/yh/miniconda3/envs/newtrackvla/bin/python}
CKPT=${CKPT:?set CKPT to the trained top-8 checkpoint}
TAG=${RUN_TAG:-replay_dt_at_$(date +%Y%m%d_%H%M%S)}
BASE=/data/yh/newtrackvla修改/newtrackvla_base_yh_clean/output/${TAG}
mkdir -p "${BASE}"

busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d' || true)
[[ -z "${busy}" ]] || { echo "GPU busy; stop training before replay eval: ${busy}" >&2; exit 2; }

for kind in dt at; do
  echo "[replay-eval] kind=${kind} instruction_source=${kind}"
  SOURCE_RUNTIME=${ROOT}/unrealzoo/Linux/UnrealZoo_UE5_6_Linux_v3.0.0 \
  PYTHON_BIN="${PY}" SERVER_PER_WORKER=1 EVAL_NUMA_NODE=1 \
  bash "${REPO}/sh/eval_airground_coop_v3.sh" \
    --gpu-ids 0,1 --render-gpu-ids 0,1 --workers-per-gpu 2 \
    --manifest "/data/yh/data/manifests/details/eval_${kind}_replay_recorded.json" \
    --ckpt "${CKPT}" \
    --save-path "${BASE}/${kind}" \
    --runtime-root "${BASE}/runtime_${kind}" \
    --no-resume --max-steps 300 --worker-start-stagger-sec 5 \
    --scene-timeout-seconds 12600
done
