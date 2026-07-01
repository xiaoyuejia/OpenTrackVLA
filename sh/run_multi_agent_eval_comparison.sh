#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/hdt/miniconda3/envs/omtracknew/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/config/eval_base_marker_realtime_gpu1.yaml}"

exec "${PYTHON_BIN}" tools/run_multi_agent_eval_comparison.py \
    --config "${CONFIG_PATH}" \
    "$@"
