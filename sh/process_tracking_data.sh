#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/hdt/miniconda3/envs/omtracknew/bin/python}"
INPUT_ROOT="${INPUT_ROOT:-/data/hdt/ntv_data/sim_data/unrealzoo_robotdog_human_debug}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/hdt/ntv_data/data/unrealzoo_robotdog_human_debug}"
HISTORY="${HISTORY:-31}"
HORIZON="${HORIZON:-8}"
DT="${DT:-0.1}"
MIN_FOLLOWING_RATE="${MIN_FOLLOWING_RATE:-0.0}"
MIN_TOTAL_STEPS="${MIN_TOTAL_STEPS:-50}"
ONLY_SUCCESS="${ONLY_SUCCESS:-1}"
EXCLUDE_COLLISION="${EXCLUDE_COLLISION:-1}"
INSTRUCTION="${INSTRUCTION:-Follow the target person without collision.}"

usage() {
    cat <<'EOF'
Process collected simulation data into TrackVLA training samples.

Usage:
  bash process_tracking_data.sh [options]

Examples:
  bash process_tracking_data.sh
  bash process_tracking_data.sh --input /data/hdt/ntv_data/sim_data/sample --output /data/hdt/ntv_data/data/sample_processed
  INPUT_ROOT=/data/hdt/ntv_data/sim_data/sample OUTPUT_ROOT=/data/hdt/ntv_data/data/sample_processed bash process_tracking_data.sh

Options:
  --input PATH                 input sim_data root
  --output PATH                output dataset root
  --history N                  previous frame history length, default 31
  --horizon N                  future action horizon, default 8
  --dt SEC                     action integration timestep, default 0.1
  --min-following-rate FLOAT   minimum following rate, default 0.5
  --min-total-steps N          minimum episode steps, default 50
  --no-only-success            keep episodes even if status is not success
  --include-collision          keep collision episodes
  --instruction TEXT           instruction written into samples
  --python PATH                python executable, default omtracknew env
  -h, --help                   show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input)
            INPUT_ROOT="$2"
            shift 2
            ;;
        --output)
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --history)
            HISTORY="$2"
            shift 2
            ;;
        --horizon)
            HORIZON="$2"
            shift 2
            ;;
        --dt)
            DT="$2"
            shift 2
            ;;
        --min-following-rate)
            MIN_FOLLOWING_RATE="$2"
            shift 2
            ;;
        --min-total-steps)
            MIN_TOTAL_STEPS="$2"
            shift 2
            ;;
        --no-only-success)
            ONLY_SUCCESS=0
            shift
            ;;
        --include-collision)
            EXCLUDE_COLLISION=0
            shift
            ;;
        --instruction)
            INSTRUCTION="$2"
            shift 2
            ;;
        --python)
            PYTHON_BIN="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

[[ -x "$PYTHON_BIN" ]] || {
    echo "[ERROR] Python executable not found or not executable: $PYTHON_BIN" >&2
    exit 1
}

[[ -d "$INPUT_ROOT" ]] || {
    echo "[ERROR] Input root does not exist: $INPUT_ROOT" >&2
    exit 1
}

ARGS=(
    -m tools.make_tracking_data
    --input_root "$INPUT_ROOT"
    --output_root "$OUTPUT_ROOT"
    --history "$HISTORY"
    --horizon "$HORIZON"
    --dt "$DT"
    --min_following_rate "$MIN_FOLLOWING_RATE"
    --min_total_steps "$MIN_TOTAL_STEPS"
    --instruction "$INSTRUCTION"
)

if [[ "$ONLY_SUCCESS" = "1" ]]; then
    ARGS+=(--only_success)
fi

if [[ "$EXCLUDE_COLLISION" = "1" ]]; then
    ARGS+=(--exclude_collision)
fi

echo "=============================================="
echo "Process tracking data"
echo "=============================================="
echo "Python:             $PYTHON_BIN"
echo "Input root:         $INPUT_ROOT"
echo "Output root:        $OUTPUT_ROOT"
echo "History:            $HISTORY"
echo "Horizon:            $HORIZON"
echo "dt:                 $DT"
echo "Only success:       $ONLY_SUCCESS"
echo "Min following rate: $MIN_FOLLOWING_RATE"
echo "Min total steps:    $MIN_TOTAL_STEPS"
echo "Exclude collision:  $EXCLUDE_COLLISION"
echo "Instruction:        $INSTRUCTION"
echo "=============================================="

"$PYTHON_BIN" "${ARGS[@]}"

echo ""
echo "Done. Training JSONL root: $OUTPUT_ROOT/jsonl"
