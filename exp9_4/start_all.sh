#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP_ROOT="${ROOT}/exp9_4"
PID_FILE="${EXP_ROOT}/state/pipeline.pid"
PYTHON_BIN="${PYTHON_BIN:-/home/yh/miniconda3/envs/newtrackvla/bin/python}"

# 一次性创建训练、评估、日志、runtime、状态和汇总所需目录。
mkdir -p \
  "${EXP_ROOT}/logs" \
  "${EXP_ROOT}/state" \
  "${EXP_ROOT}/models" \
  "${EXP_ROOT}/results/dt" \
  "${EXP_ROOT}/results/at" \
  "${EXP_ROOT}/results/stt" \
  "${EXP_ROOT}/runtime/dt" \
  "${EXP_ROOT}/runtime/at" \
  "${EXP_ROOT}/runtime/stt" \
  "${EXP_ROOT}/summaries"

if [[ -s "${PID_FILE}" ]]; then
  old_pid="$(cat "${PID_FILE}")"
  if [[ "${old_pid}" =~ ^[0-9]+$ ]] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "[ERROR] exp9_4 pipeline 已在运行：PID=${old_pid}" >&2
    exit 1
  fi
fi

[[ -x "${PYTHON_BIN}" ]] || { echo "[ERROR] Python 不可执行：${PYTHON_BIN}" >&2; exit 1; }

stamp="$(date +%Y%m%d_%H%M%S)"
log="${EXP_ROOT}/logs/pipeline_all_${stamp}.log"
cd "${ROOT}"
nohup env PYTHON_BIN="${PYTHON_BIN}" \
  bash "${EXP_ROOT}/run_pipeline.sh" --stage all \
  >"${log}" 2>&1 &
pid=$!
printf '%s\n' "${pid}" >"${PID_FILE}"
printf '%s\n' "${log}" >"${EXP_ROOT}/state/latest_pipeline_log.txt"
echo "[STARTED] PID=${pid}"
echo "[LOG] ${log}"
echo "[MONITOR] tail -f ${log}"
