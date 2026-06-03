#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/hdt/miniconda3/envs/omtracknew/bin/python}"
DATA_ROOT="${DATA_ROOT:-data/unrealzoo_robotdog_human_debug}"
CACHE_ROOT="${CACHE_ROOT:-}"
BATCH_SIZE="${BATCH_SIZE:-8}"
IMAGE_SIZE="${IMAGE_SIZE:-384}"

usage() {
    cat <<'EOF'
Precompute DINOv3 + SigLIP visual features for a processed TrackVLA dataset.

Usage:
  bash precache_tracking_features.sh [options]

Examples:
  bash precache_tracking_features.sh
  bash precache_tracking_features.sh --data-root data/sample_processed
  DATA_ROOT=data/sample_processed bash precache_tracking_features.sh

Options:
  --data-root PATH     processed dataset root containing frames/
  --cache-root PATH    feature cache root, default <data-root>/vision_cache
  --batch-size N       encoder batch size, default 8
  --image-size N       image size, default 384
  --python PATH        python executable, default omtracknew env
  -h, --help           show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --cache-root)
            CACHE_ROOT="$2"
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

if [[ -z "$CACHE_ROOT" ]]; then
    CACHE_ROOT="$DATA_ROOT/vision_cache"
fi

[[ -x "$PYTHON_BIN" ]] || {
    echo "[ERROR] Python executable not found or not executable: $PYTHON_BIN" >&2
    exit 1
}

[[ -d "$DATA_ROOT/frames" ]] || {
    echo "[ERROR] Frames directory does not exist: $DATA_ROOT/frames" >&2
    echo "Run process_tracking_data.sh first." >&2
    exit 1
}

echo "=============================================="
echo "Precache tracking vision features"
echo "=============================================="
echo "Python:     $PYTHON_BIN"
echo "Data root:  $DATA_ROOT"
echo "Cache root: $CACHE_ROOT"
echo "Batch size: $BATCH_SIZE"
echo "Image size: $IMAGE_SIZE"
echo "=============================================="

"$PYTHON_BIN" precache_frames.py \
    --data_root "$DATA_ROOT" \
    --cache_root "$CACHE_ROOT" \
    --batch_size "$BATCH_SIZE" \
    --image_size "$IMAGE_SIZE"

echo ""
echo "Done. Cache root: $CACHE_ROOT"
