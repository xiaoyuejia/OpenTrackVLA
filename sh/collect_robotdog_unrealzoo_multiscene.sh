#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/hdt/miniconda3/envs/omtracknew/bin/python}"
COLLECT_SCRIPT="${COLLECT_SCRIPT:-unrealzoo-gym/example/DataRecording/generate_robotdog_human_tracking_small.py}"
OUT_DIR="${OUT_DIR:-/data/hdt/ntv_data/sim_data/unrealzoo_robotdog_human_1k}"
TOTAL_EPISODES="${TOTAL_EPISODES:-1000}"
SEED="${SEED:-100}"
MAX_STEPS="${MAX_STEPS:-60}"
MAX_ATTEMPTS_PER_EP="${MAX_ATTEMPTS_PER_EP:-6}"
FPS="${FPS:-10}"
WIDTH="${WIDTH:-640}"
HEIGHT="${HEIGHT:-480}"
MONITOR="${MONITOR:-0}"
DEBUG_MOTION="${DEBUG_MOTION:-0}"
OFFSCREEN="${OFFSCREEN:-1}"
LOG_DIR="${LOG_DIR:-logs/robotdog_unrealzoo_1k}"

SCENES="${SCENES:-DowntownWest,SuburbNeighborhood_Day,SuburbNeighborhood_Night,ModularNeighborhood,ModularBuilding,ModularVictorianCity,Old_Factory_01,IndustrialArea,ContainerYard_Day,PlatformFactory}"

usage() {
    cat <<'EOF'
Collect robot-dog-human tracking data uniformly across UnrealZoo Track scenes.

Usage:
  bash collect_robotdog_unrealzoo_multiscene.sh [options]

Examples:
  bash collect_robotdog_unrealzoo_multiscene.sh
  TOTAL_EPISODES=1000 OUT_DIR=/data/hdt/ntv_data/sim_data/robotdog_1k bash collect_robotdog_unrealzoo_multiscene.sh
  bash collect_robotdog_unrealzoo_multiscene.sh --scenes DowntownWest,SuburbNeighborhood_Day --total 100

Options:
  --total N                 total accepted episodes to request, default 1000
  --scenes CSV              scene names without UnrealTrack-/... suffix
  --out-dir PATH            raw sim_data output root
  --seed N                  base seed, default 100
  --max-steps N             max steps per episode, default 60
  --max-attempts-per-ep N   attempts budget multiplier per scene, default 6
  --monitor                 show monitor windows, useful only for debugging
  --debug-motion            print robotdog motion debug logs
  --no-offscreen            launch UE with visible rendering
  --python PATH             python executable, default omtracknew env
  -h, --help                show this help

Notes:
  The collector already rejects low-quality episodes internally. After collection,
  run process_tracking_data.sh with --min-following-rate 0.5 --min-total-steps 50.
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
        --out-dir)
            OUT_DIR="$2"
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
        --python)
            PYTHON_BIN="$2"
            shift 2
            ;;
        -h|--help)
            usagescene_seed=$((SEED + idx))
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

[[ -f "$COLLECT_SCRIPT" ]] || {
    echo "[ERROR] Collector script not found: $COLLECT_SCRIPT" >&2
    exit 1
}

IFS=',' read -r -a SCENE_ARRAY <<< "$SCENES"
NUM_SCENES="${#SCENE_ARRAY[@]}"
[[ "$NUM_SCENES" -gt 0 ]] || {
    echo "[ERROR] No scenes configured." >&2
    exit 1
}

BASE=$((TOTAL_EPISODES / NUM_SCENES))
REMAINDER=$((TOTAL_EPISODES % NUM_SCENES))

mkdir -p "$OUT_DIR" "$LOG_DIR"

echo "=============================================="
echo "Robotdog UnrealZoo multi-scene collection"
echo "=============================================="
echo "Python:              $PYTHON_BIN"
echo "Collector:           $COLLECT_SCRIPT"
echo "Output dir:          $OUT_DIR"
echo "Total episodes:      $TOTAL_EPISODES"
echo "Scenes:              $NUM_SCENES"
echo "Base per scene:      $BASE"
echo "Remainder:           $REMAINDER"
echo "Seed:                $SEED"
echo "Max steps:           $MAX_STEPS"
echo "Attempts multiplier: $MAX_ATTEMPTS_PER_EP"
echo "Offscreen:           $OFFSCREEN"
echo "Monitor:             $MONITOR"
echo "Debug motion:        $DEBUG_MOTION"
echo "=============================================="

for idx in "${!SCENE_ARRAY[@]}"; do
    scene="$(echo "${SCENE_ARRAY[$idx]}" | xargs)"
    [[ -n "$scene" ]] || continue
    
    episodes="$BASE"
    if [[ "$idx" -lt "$REMAINDER" ]]; then
        episodes=$((episodes + 1))
    fi
    [[ "$episodes" -gt 0 ]] || continue

    env_id="UnrealTrack-${scene}-ContinuousColor-v0"
    scene_seed=$((SEED + idx))
    max_attempts=$((episodes * MAX_ATTEMPTS_PER_EP))
    log_file="$LOG_DIR/${idx}_${scene}.log"

    ARGS=(
        "$COLLECT_SCRIPT"
        --env-id "$env_id"
        --episodes "$episodes"
        --max-attempts "$max_attempts"
        --max-steps "$MAX_STEPS"
        --seed "$scene_seed"
        --out-dir "$OUT_DIR"
        --fps "$FPS"
        --width "$WIDTH"
        --height "$HEIGHT"
    )

    if [[ "$OFFSCREEN" = "1" ]]; then
        ARGS+=(--offscreen)
    else
        ARGS+=(--no-offscreen)
    fi
    if [[ "$MONITOR" = "1" ]]; then
        ARGS+=(--monitor)
    fi
    if [[ "$DEBUG_MOTION" = "1" ]]; then
        ARGS+=(--debug-motion)
    fi

    echo ""
    echo "[scene $((idx + 1))/$NUM_SCENES] $env_id episodes=$episodes seed=$scene_seed max_attempts=$max_attempts"
    echo "[scene $((idx + 1))/$NUM_SCENES] log=$log_file"
    "$PYTHON_BIN" "${ARGS[@]}" 2>&1 | tee "$log_file"
done

echo ""
echo "=============================================="
echo "Collection requested for all scenes."
echo "Raw output: $OUT_DIR"
echo "Logs:       $LOG_DIR"
echo "=============================================="
