#!/bin/bash
# =============================================================================
# generate_train_data.sh - OpenTrackVLA 完整训练数据生成流程
# =============================================================================
#
# 这个脚本实现了从HM3D/MP3D场景批量生成训练数据的完整闭环：
#   1. 使用Oracle策略在Habitat仿真器中采集数据
#   2. 将仿真数据转换为训练格式 (支持严格筛选)
#   3. 预计算视觉特征缓存
#
# 用法:
#   bash generate_train_data.sh [选项]
#
# 选项:
#   --mode              运行模式: collect|process|cache|all (默认: all)
#   --config            Habitat配置文件 (默认: track_train_stt.yaml)
#   --num-episodes      采集的episode数量 (默认: 1000)
#   --num-parallel      并行进程数 (默认: 4)
#   --seed              随机种子 (默认: 100)
#   --output            输出目录 (默认: data/generated)
#   --min-following-rate 最小跟踪率阈值 (默认: 0.7, 范围 0-1)
#   --min-total-steps   最小episode步数 (默认: 40)
#   --no-exclude-collision 不排除碰撞episode (默认: 排除碰撞)
#
# 示例:
#   # 完整流程 (使用严格筛选)
#   bash generate_train_data.sh --num-episodes 5000 --num-parallel 8
#
#   # 使用更严格的筛选
#   bash generate_train_data.sh --min-following-rate 0.7 --min-total-steps 40
#
#   # 仅采集数据
#   bash generate_train_data.sh --mode collect --num-episodes 1000
#
#   # 仅处理已采集的数据
#   bash generate_train_data.sh --mode process
#
# =============================================================================

set -e

# 默认参数
MODE="all"
CONFIG="habitat-lab/habitat/config/benchmark/nav/track/track_train_stt.yaml"
NUM_EPISODES=10
NUM_PARALLEL=1
SEED=100
OUTPUT_DIR="data/sample"
SIM_DATA_DIR="sim_data/stt_traintest"
INSTRUCTION="Follow the target person without collision."
# 数据筛选参数 (新增)
MIN_FOLLOWING_RATE=0.5
EXCLUDE_COLLISION=true
MIN_TOTAL_STEPS=50

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --num-episodes)
            NUM_EPISODES="$2"
            shift 2
            ;;
        --num-parallel)
            NUM_PARALLEL="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --sim-data)
            SIM_DATA_DIR="$2"
            shift 2
            ;;
        --instruction)
            INSTRUCTION="$2"
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
        --no-exclude-collision)
            EXCLUDE_COLLISION=false
            shift
            ;;
        -h|--help)
            head -40 "$0" | tail -35
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# 计算每个进程的episode数量
EPISODES_PER_PROCESS=$((NUM_EPISODES / NUM_PARALLEL))

echo "=============================================="
echo "OpenTrackVLA Training Data Generation"
echo "=============================================="
echo "Mode:           $MODE"
echo "Config:         $CONFIG"
echo "Total Episodes: $NUM_EPISODES"
echo "Parallel Jobs:  $NUM_PARALLEL"
echo "Episodes/Job:   $EPISODES_PER_PROCESS"
echo "Seed:           $SEED"
echo "Sim Data Dir:   $SIM_DATA_DIR"
echo "Output Dir:     $OUTPUT_DIR"
echo "Instruction:    $INSTRUCTION"
echo "Min Follow Rate: $MIN_FOLLOWING_RATE"
echo "Min Total Steps: $MIN_TOTAL_STEPS"
echo "Exclude Collision: $EXCLUDE_COLLISION"
echo "=============================================="

# 创建目录
mkdir -p "$SIM_DATA_DIR"
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# =============================================================================
# Step 1: 数据采集 (使用Oracle策略)
# =============================================================================
collect_data() {
    echo ""
    echo "[Step 1/3] Collecting simulation data with Oracle policy..."
    echo "=============================================="

    if [ "$NUM_PARALLEL" -eq 1 ]; then
        # 单进程模式
        echo "Running single collector (seed=$SEED, episodes=$NUM_EPISODES)"

        python collect_sim_data.py \
            --exp-config "$CONFIG" \
            --save-path "$SIM_DATA_DIR" \
            --seed "$SEED" \
            --num-episodes "$NUM_EPISODES" \
            2>&1 | tee "logs/collect.log"

        echo "Collector finished!"
    else
        # 多进程模式 - 所有进程写入同一目录
        PIDS=()

        for ((i=0; i<NUM_PARALLEL; i++)); do
            LOG_FILE="logs/collect_split_${i}.log"

            echo "Starting collector $i (seed=$SEED, split=$i/$NUM_PARALLEL, episodes=$EPISODES_PER_PROCESS)"

            CUDA_VISIBLE_DEVICES=$((i % $(nvidia-smi -L | wc -l))) \
            python collect_sim_data.py \
                --exp-config "$CONFIG" \
                --save-path "$SIM_DATA_DIR" \
                --seed "$SEED" \
                --split-id "$i" \
                --split-num "$NUM_PARALLEL" \
                --num-episodes "$EPISODES_PER_PROCESS" \
                > "$LOG_FILE" 2>&1 &

            PIDS+=($!)
        done

        echo "Waiting for all collectors to finish..."
        echo "Check logs in logs/collect_split_*.log"

        # 等待所有进程完成
        FAILED=0
        for pid in "${PIDS[@]}"; do
            if ! wait "$pid"; then
                FAILED=$((FAILED + 1))
            fi
        done

        if [ $FAILED -gt 0 ]; then
            echo "WARNING: $FAILED collector(s) failed. Check logs for details."
        else
            echo "All collectors finished successfully!"
        fi
    fi

    # 统计采集结果
    TOTAL_VIDEOS=$(find "$SIM_DATA_DIR" -name "*.mp4" | wc -l)
    TOTAL_INFOS=$(find "$SIM_DATA_DIR" -name "*_info.json" | wc -l)
    echo "Collected: $TOTAL_VIDEOS videos, $TOTAL_INFOS info files"
}

# =============================================================================
# Step 2: 数据处理 (转换为训练格式)
# =============================================================================
process_data() {
    echo ""
    echo "[Step 2/3] Processing simulation data to training format..."
    echo "=============================================="

    # 构建筛选参数
    FILTER_ARGS="--only_success --min_following_rate $MIN_FOLLOWING_RATE --min_total_steps $MIN_TOTAL_STEPS"
    if [ "$EXCLUDE_COLLISION" = true ]; then
        FILTER_ARGS="$FILTER_ARGS --exclude_collision"
    fi

    echo "Filter args: $FILTER_ARGS"

    python make_tracking_data.py \
        --input_root "$SIM_DATA_DIR" \
        --output_root "$OUTPUT_DIR" \
        --history 31 \
        --horizon 8 \
        --dt 0.1 \
        $FILTER_ARGS \
        --instruction "$INSTRUCTION"

    # 统计处理结果
    TOTAL_JSONL=$(find "$OUTPUT_DIR/jsonl" -name "*.jsonl" 2>/dev/null | wc -l)
    echo "Generated: $TOTAL_JSONL JSONL files"
}

# =============================================================================
# Step 3: 预计算视觉特征缓存
# =============================================================================
cache_features() {
    echo ""
    echo "[Step 3/3] Pre-caching vision features..."
    echo "=============================================="

    python precache_frames.py \
        --data_root "$OUTPUT_DIR" \
        --cache_root "$OUTPUT_DIR/vision_cache" \
        --batch_size 8 \
        --image_size 384

    # 统计缓存结果
    TOTAL_CACHE=$(find "$OUTPUT_DIR/vision_cache" -name "*.pt" 2>/dev/null | wc -l)
    echo "Cached: $TOTAL_CACHE feature files"
}

# =============================================================================
# 主流程
# =============================================================================
case $MODE in
    collect)
        collect_data
        ;;
    process)
        process_data
        ;;
    cache)
        cache_features
        ;;
    all)
        collect_data
        process_data
        cache_features
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Valid modes: collect, process, cache, all"
        exit 1
        ;;
esac

echo ""
echo "=============================================="
echo "Data generation complete!"
echo "=============================================="
echo ""
echo "Output structure:"
echo "  $OUTPUT_DIR/"
echo "    ├── frames/          # Extracted video frames"
echo "    ├── jsonl/           # Training samples (JSONL)"
echo "    └── vision_cache/    # Pre-computed vision features"
echo ""
echo "To start training:"
echo "  python train.py \\"
echo "      --train_json $OUTPUT_DIR/jsonl \\"
echo "      --cache_root $OUTPUT_DIR/vision_cache \\"
echo "      --out_dir ckpt_sample \\"
echo "      --epochs 2 \\"
echo "      --batch_size 8 \\"
echo "      --n_waypoints 8 \\"
echo "      --history 31 \\"
echo "      --lr 2e-5 \\"
echo "      --mixed_precision \\"
echo "      --save_trajectories"
echo ""
