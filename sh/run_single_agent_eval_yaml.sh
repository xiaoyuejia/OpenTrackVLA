#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/hdt/miniconda3/envs/omtracknew/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/config/eval_drone_single_realtime_gpu1.yaml}"

exec "${PYTHON_BIN}" tools/run_single_agent_eval_yaml.py \
    --config "${CONFIG_PATH}" \
    "$@"
