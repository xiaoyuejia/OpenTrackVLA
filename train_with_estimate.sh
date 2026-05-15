#!/usr/bin/env bash
set -euo pipefail

# Training wrapper with pre-flight step/ETA estimates.
# It does not change train.py; it only counts samples and then launches train.py.
#
# Examples:
#   ./train_with_estimate.sh
#   EPOCHS=10 BATCH_SIZE=16 OUT_DIR=ckpt_mydata ./train_with_estimate.sh
#   RESUME=1 OUT_DIR=ckpt_mydata ./train_with_estimate.sh
#   SEC_PER_STEP=0.85 EPOCHS=10 ./train_with_estimate.sh
#   DRY_RUN=1 ./train_with_estimate.sh

TRAIN_JSON="${TRAIN_JSON:-data/stt_filtered/jsonl}"
CACHE_ROOT="${CACHE_ROOT:-data/stt_filtered/vision_cache}"
OUT_DIR="${OUT_DIR:-ckpt_stt_filtered}"

EPOCHS="${EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-8}"
N_WAYPOINTS="${N_WAYPOINTS:-8}"
HISTORY="${HISTORY:-31}"
LR="${LR:-2e-5}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_CKPTS="${MAX_CKPTS:-3}"
LOG_EVERY="${LOG_EVERY:-10}"

MIXED_PRECISION="${MIXED_PRECISION:-1}"
CSV_LOGGING="${CSV_LOGGING:-1}"
SAVE_TRAJECTORIES="${SAVE_TRAJECTORIES:-1}"
RESUME="${RESUME:-0}"
RESUME_CKPT="${RESUME_CKPT:-}"
SEC_PER_STEP="${SEC_PER_STEP:-}"
LOG_TO_FILE="${LOG_TO_FILE:-1}"
DRY_RUN="${DRY_RUN:-0}"

if [ ! -e "${TRAIN_JSON}" ]; then
    echo "[ERROR] TRAIN_JSON does not exist: ${TRAIN_JSON}" >&2
    exit 1
fi

if [ ! -e "${CACHE_ROOT}" ]; then
    echo "[WARN] CACHE_ROOT does not exist yet: ${CACHE_ROOT}" >&2
    echo "       train.py may encode/load more slowly or fail if cache is required."
fi

read -r NUM_SAMPLES DATASET_KIND <<< "$(
python - "${TRAIN_JSON}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])

def count_jsonl(p: Path) -> int:
    n = 0
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n

if path.is_dir():
    jsonl_files = sorted(path.rglob("*.jsonl"))
    if jsonl_files:
        print(sum(count_jsonl(p) for p in jsonl_files), "jsonl_dir")
    else:
        json_files = sorted(path.rglob("*.json"))
        total = 0
        for p in json_files:
            try:
                obj = json.load(p.open("r", encoding="utf-8"))
                total += len(obj) if isinstance(obj, list) else 1
            except Exception:
                pass
        print(total, "json_dir")
elif path.suffix == ".jsonl":
    print(count_jsonl(path), "jsonl_file")
elif path.suffix == ".json":
    obj = json.load(path.open("r", encoding="utf-8"))
    print(len(obj) if isinstance(obj, list) else 1, "json_file")
else:
    print(0, "unknown")
PY
)"

if [ "${NUM_SAMPLES}" -le 0 ]; then
    echo "[ERROR] No training samples found under ${TRAIN_JSON}" >&2
    exit 1
fi

STEPS_PER_EPOCH=$(( (NUM_SAMPLES + BATCH_SIZE - 1) / BATCH_SIZE ))
TOTAL_STEPS=$(( STEPS_PER_EPOCH * EPOCHS ))
CKPT_SAVES=$(( TOTAL_STEPS / 100 ))

EST_SEC_PER_STEP="${SEC_PER_STEP}"
EST_SOURCE=""
if [ -z "${EST_SEC_PER_STEP}" ] && [ -f "${OUT_DIR}/train_log.csv" ]; then
    EST_SEC_PER_STEP="$(
    python - "${OUT_DIR}/train_log.csv" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
vals = []
try:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                vals.append(float(row.get("step_time", "")))
            except Exception:
                pass
except Exception:
    pass
vals = vals[-50:]
print(f"{(sum(vals) / len(vals)):.6f}" if vals else "")
PY
    )"
    if [ -n "${EST_SEC_PER_STEP}" ]; then
        EST_SOURCE="from ${OUT_DIR}/train_log.csv"
    fi
elif [ -n "${EST_SEC_PER_STEP}" ]; then
    EST_SOURCE="from SEC_PER_STEP"
fi

echo "=============================================="
echo "OpenTrackVLA training pre-flight"
echo "=============================================="
echo "Dataset:          ${TRAIN_JSON}"
echo "Dataset kind:     ${DATASET_KIND}"
echo "Samples:          ${NUM_SAMPLES}"
echo "Cache root:       ${CACHE_ROOT}"
echo "Output dir:       ${OUT_DIR}"
echo "Epochs:           ${EPOCHS}"
echo "Batch size:       ${BATCH_SIZE}"
echo "Steps/epoch:      ${STEPS_PER_EPOCH}"
echo "Total new steps:  ${TOTAL_STEPS}"
echo "Checkpoint every: 100 steps"
echo "Expected ckpts:   ${CKPT_SAVES} saves, keep latest ${MAX_CKPTS}"
echo "Log every:        ${LOG_EVERY} steps"
echo "LR:               ${LR}"
echo "Mixed precision:  ${MIXED_PRECISION}"
echo "CSV logging:      ${CSV_LOGGING}"
echo "Save traj npz:    ${SAVE_TRAJECTORIES}"

if [ -n "${EST_SEC_PER_STEP}" ]; then
    python - "${EST_SEC_PER_STEP}" "${TOTAL_STEPS}" "${EST_SOURCE}" <<'PY'
import sys
sec_per_step = float(sys.argv[1])
steps = int(sys.argv[2])
source = sys.argv[3]
total = sec_per_step * steps
h = int(total // 3600)
m = int((total % 3600) // 60)
s = int(total % 60)
print(f"ETA:              {h:02d}:{m:02d}:{s:02d} ({sec_per_step:.3f}s/step {source})")
PY
else
    echo "ETA:              unavailable before first run"
    echo "                  Set SEC_PER_STEP=... or enable CSV_LOGGING=1 and rerun after a short pilot."
fi

if [ "${RESUME}" = "1" ]; then
    if [ -n "${RESUME_CKPT}" ]; then
        echo "Resume:           enabled from ${RESUME_CKPT}"
    else
        echo "Resume:           enabled from latest checkpoint in ${OUT_DIR}"
    fi
else
echo "Resume:           disabled"
fi
echo "Dry run:          ${DRY_RUN}"
echo "=============================================="

ARGS=(
    --train_json "${TRAIN_JSON}"
    --cache_root "${CACHE_ROOT}"
    --out_dir "${OUT_DIR}"
    --epochs "${EPOCHS}"
    --batch_size "${BATCH_SIZE}"
    --n_waypoints "${N_WAYPOINTS}"
    --history "${HISTORY}"
    --lr "${LR}"
    --num_workers "${NUM_WORKERS}"
    --max_ckpts "${MAX_CKPTS}"
    --log_every "${LOG_EVERY}"
)

if [ "${MIXED_PRECISION}" = "1" ]; then
    ARGS+=(--mixed_precision)
fi
if [ "${CSV_LOGGING}" = "1" ]; then
    ARGS+=(--csv_logging)
fi
if [ "${SAVE_TRAJECTORIES}" = "1" ]; then
    ARGS+=(--save_trajectories)
fi
if [ "${RESUME}" = "1" ]; then
    ARGS+=(--resume)
    if [ -n "${RESUME_CKPT}" ]; then
        ARGS+=(--resume_ckpt "${RESUME_CKPT}")
    fi
fi

mkdir -p "${OUT_DIR}"

echo "[RUN] python train.py ${ARGS[*]}"
if [ "${DRY_RUN}" = "1" ]; then
    echo "[DRY_RUN] Not launching training."
    exit 0
fi

if [ "${LOG_TO_FILE}" = "1" ]; then
    LOG_FILE="${OUT_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
    echo "[LOG] ${LOG_FILE}"
    python train.py "${ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
else
    python train.py "${ARGS[@]}"
fi
