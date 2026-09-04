#!/usr/bin/env bash
set -Eeuo pipefail

EVAL_DIR="${EVAL_DIR:-output/eval_airground_coop_v3_receiver_target_125}"

PY="/home/hdt/miniconda3/envs/omtracknew/bin/python"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${PROJECT_DIR}"
"${PY}" -m tools.calculate_unrealzoo_metrics \
  --eval-dir "${EVAL_DIR}" \
  --expected-episodes 78 \
  --require-exact-episodes \
  --output-csv "${EVAL_DIR}/metrics.csv"

echo "指标已保存到：${EVAL_DIR}/metrics.csv"
 
