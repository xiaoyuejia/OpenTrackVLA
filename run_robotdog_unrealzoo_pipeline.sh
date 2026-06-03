#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/hdt/miniconda3/envs/omtracknew/bin/python}"
TOTAL_EPISODES="${TOTAL_EPISODES:-1000}" 
#采集的总 episode 数，采集脚本内部会根据质量过滤掉一部分，因此实际请求的 episode 数是这个数乘以 MAX_ATTEMPTS_PER_EP。
SEED="${SEED:-100}"
MAX_STEPS="${MAX_STEPS:-60}" #每个 episode 的最大步数，超过这个步数会强制终止。
MAX_ATTEMPTS_PER_EP="${MAX_ATTEMPTS_PER_EP:-6}"
SIM_DATA_DIR="${SIM_DATA_DIR:-/data/hdt/newtrackvla/sim_data/unrealzoo_robotdog_human_1k}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/hdt/newtrackvla/data/unrealzoo_robotdog_human_1k}"
SCENES="${SCENES:-}"

MIN_FOLLOWING_RATE="${MIN_FOLLOWING_RATE:-0.5}"
MIN_TOTAL_STEPS="${MIN_TOTAL_STEPS:-50}"
HISTORY="${HISTORY:-31}"
HORIZON="${HORIZON:-8}"
DT="${DT:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-8}"
IMAGE_SIZE="${IMAGE_SIZE:-384}"

RUN_COLLECT="${RUN_COLLECT:-1}"
RUN_PROCESS="${RUN_PROCESS:-1}"
RUN_CACHE="${RUN_CACHE:-1}"
MONITOR="${MONITOR:-0}"
DEBUG_MOTION="${DEBUG_MOTION:-0}"
OFFSCREEN="${OFFSCREEN:-1}"

usage() {
    cat <<'EOF'
Run the full robotdog UnrealZoo data pipeline:
  1. collect multi-scene robotdog-human tracking data
  2. process raw sim_data into TrackVLA training JSONL
  3. precompute DINOv3 + SigLIP visual features

Usage:
  bash run_robotdog_unrealzoo_pipeline.sh [options]

Examples:
  bash run_robotdog_unrealzoo_pipeline.sh

  bash run_robotdog_unrealzoo_pipeline.sh \
    --total 1000 \
    --sim-data /data/hdt/newtrackvla/sim_data/robotdog_1k \
    --output /data/hdt/newtrackvla/data/robotdog_1k

  bash run_robotdog_unrealzoo_pipeline.sh --skip-collect

Options:
  --total N                 total accepted episodes to request, default 1000
  --scenes CSV              scene names without UnrealTrack-/... suffix
  --sim-data PATH           raw sim_data output root
  --output PATH             processed dataset output root
  --seed N                  base seed, default 100
  --max-steps N             max steps per episode, default 60
  --max-attempts-per-ep N   attempts budget multiplier per scene, default 6
  --min-following-rate F    post-process quality threshold, default 0.5
  --min-total-steps N       post-process quality threshold, default 50
  --batch-size N            feature cache batch size, default 8
  --image-size N            feature cache image size, default 384
  --monitor                 show collection monitor windows
  --debug-motion            print robotdog motion debug logs
  --no-offscreen            launch UE with visible rendering
  --skip-collect            only run process/cache on existing sim_data
  --skip-process            skip processing
  --skip-cache              skip feature caching
  --python PATH             python executable, default omtracknew env
  -h, --help                show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --total)
            TOTAL_EPISODES="$2"
            shift 2
            ;;
        --scenes)
            SCENES="$2"
            shift 2
            ;;
        --sim-data)
            SIM_DATA_DIR="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --max-steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --max-attempts-per-ep)
            MAX_ATTEMPTS_PER_EP="$2"
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
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --image-size)
            IMAGE_SIZE="$2"
            shift 2
            ;;
        --monitor)
            MONITOR=1
            shift
            ;;
        --debug-motion)
            DEBUG_MOTION=1
            shift
            ;;
        --no-offscreen)
            OFFSCREEN=0
            shift
            ;;
        --skip-collect)
            RUN_COLLECT=0
            shift
            ;;
        --skip-process)
            RUN_PROCESS=0
            shift
            ;;
        --skip-cache)
            RUN_CACHE=0
            shift
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

require_file() {
    local path="$1"
    [[ -f "$path" ]] || {
        echo "[ERROR] Required script not found: $path" >&2
        exit 1
    }
}

count_jsonl_lines() {
    local root="$1"
    if [[ ! -d "$root" ]]; then
        echo 0
        return
    fi
    find "$root" -name "*.jsonl" -print0 | xargs -0 -r wc -l | awk 'END {print $1 + 0}'
}

require_file "collect_robotdog_unrealzoo_multiscene.sh"
require_file "process_tracking_data.sh"
require_file "precache_tracking_features.sh"

echo "=============================================="
echo "Robotdog UnrealZoo full pipeline"
echo "=============================================="
echo "Python:              $PYTHON_BIN"
echo "Run collect:         $RUN_COLLECT"
echo "Run process:         $RUN_PROCESS"
echo "Run cache:           $RUN_CACHE"
echo "Total episodes:      $TOTAL_EPISODES"
echo "Seed:                $SEED"
echo "Raw sim data:        $SIM_DATA_DIR"
echo "Processed output:    $OUTPUT_DIR"
echo "Min following rate:  $MIN_FOLLOWING_RATE"
echo "Min total steps:     $MIN_TOTAL_STEPS"
echo "Max steps:           $MAX_STEPS"
echo "Attempts multiplier: $MAX_ATTEMPTS_PER_EP"
echo "Batch/image size:    $BATCH_SIZE / $IMAGE_SIZE"
echo "=============================================="

if [[ "$RUN_COLLECT" = "1" ]]; then
    COLLECT_ARGS=(
        --total "$TOTAL_EPISODES"
        --out-dir "$SIM_DATA_DIR"
        --seed "$SEED"
        --max-steps "$MAX_STEPS"
        --max-attempts-per-ep "$MAX_ATTEMPTS_PER_EP"
        --python "$PYTHON_BIN"
    )
    if [[ -n "$SCENES" ]]; then
        COLLECT_ARGS+=(--scenes "$SCENES")
    fi
    if [[ "$MONITOR" = "1" ]]; then
        COLLECT_ARGS+=(--monitor)
    fi
    if [[ "$DEBUG_MOTION" = "1" ]]; then
        COLLECT_ARGS+=(--debug-motion)
    fi
    if [[ "$OFFSCREEN" = "0" ]]; then
        COLLECT_ARGS+=(--no-offscreen)
    fi

    echo ""
    echo "[Step 1/3] Collecting raw UnrealZoo data..."
    bash collect_robotdog_unrealzoo_multiscene.sh "${COLLECT_ARGS[@]}"
else
    echo ""
    echo "[Step 1/3] Skipping collection."
fi

raw_infos=$(find "$SIM_DATA_DIR" -name "*_info.json" 2>/dev/null | wc -l)
raw_stats=$(find "$SIM_DATA_DIR" -name "*.json" ! -name "*_info.json" 2>/dev/null | wc -l)
raw_videos=$(find "$SIM_DATA_DIR" -name "*.mp4" 2>/dev/null | wc -l)
echo "[summary] raw videos=$raw_videos info_json=$raw_infos stat_json=$raw_stats"

if [[ "$RUN_PROCESS" = "1" ]]; then
    echo ""
    echo "[Step 2/3] Processing and filtering training data..."
    bash process_tracking_data.sh \
        --input "$SIM_DATA_DIR" \
        --output "$OUTPUT_DIR" \
        --history "$HISTORY" \
        --horizon "$HORIZON" \
        --dt "$DT" \
        --min-following-rate "$MIN_FOLLOWING_RATE" \
        --min-total-steps "$MIN_TOTAL_STEPS" \
        --python "$PYTHON_BIN"
else
    echo ""
    echo "[Step 2/3] Skipping processing."
fi

jsonl_files=$(find "$OUTPUT_DIR/jsonl" -name "*.jsonl" 2>/dev/null | wc -l)
sample_count=$(count_jsonl_lines "$OUTPUT_DIR/jsonl")
frame_count=$(find "$OUTPUT_DIR/frames" -name "*.jpg" 2>/dev/null | wc -l)
echo "[summary] jsonl_files=$jsonl_files samples=$sample_count frames=$frame_count"

if [[ "$RUN_CACHE" = "1" ]]; then
    echo ""
    echo "[Step 3/3] Precomputing vision features..."
    bash precache_tracking_features.sh \
        --data-root "$OUTPUT_DIR" \
        --batch-size "$BATCH_SIZE" \
        --image-size "$IMAGE_SIZE" \
        --python "$PYTHON_BIN"
else
    echo ""
    echo "[Step 3/3] Skipping feature cache."
fi

vfine_count=$(find "$OUTPUT_DIR/vision_cache" -name "*_vfine.pt" 2>/dev/null | wc -l)
vcoarse_count=$(find "$OUTPUT_DIR/vision_cache" -name "*_vcoarse.pt" 2>/dev/null | wc -l)

echo ""
echo "=============================================="
echo "Pipeline complete"
echo "=============================================="
echo "Raw sim data:       $SIM_DATA_DIR"
echo "Processed output:   $OUTPUT_DIR"
echo "Train JSONL root:   $OUTPUT_DIR/jsonl"
echo "Vision cache root:  $OUTPUT_DIR/vision_cache"
echo "Raw episodes:       videos=$raw_videos info_json=$raw_infos stat_json=$raw_stats"
echo "Training samples:   jsonl_files=$jsonl_files samples=$sample_count frames=$frame_count"
echo "Vision cache files: vfine=$vfine_count vcoarse=$vcoarse_count"
echo "=============================================="
