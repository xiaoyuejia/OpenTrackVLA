CHUNKS=30 #将所有的分成30块来跑
NUM_PARALLEL=3
SAVE_PATH="sim_data/eval/stt"
export HF_MODEL_DIR="/data/hdt/newtrackvla/open_trackvla_hf"
export DINOV3_MODEL_PATH="/data/hdt/newtrackvla/weights/dinov3"

# if [ -n "${HF_MODEL_DIR:-}" ]; then
#     export HF_MODEL_DIR
#     echo "[eval] Using HuggingFace planner weights from ${HF_MODEL_DIR}"
# fi

IDX=0
while [ $IDX -lt $CHUNKS ]; do
    i=0
    while [ $i -lt $NUM_PARALLEL ] && [ $IDX -lt $CHUNKS ]; do
        echo "Launching job IDX=$IDX on GPU=$((IDX % NUM_PARALLEL))"
        CUDA_VISIBLE_DEVICES=$i SAVE_VIDEO=1 PYTHONPATH="habitat-lab" python run_eval.py \
            --split-num $CHUNKS \
            --split-id $IDX \
            --exp-config 'habitat-lab/habitat/config/benchmark/nav/track/track_infer_stt.yaml' \
            --run-type 'eval' \
            --save-path $SAVE_PATH &
        IDX=$((IDX + 1))
        i=$((i + 1))
    done
    wait
done
