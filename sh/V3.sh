# train
bash sh/train_airground_coop_v3.sh

# eval
bash sh/eval_airground_coop_v3.sh --gpu-ids 0 --workers-per-gpu 10 --ckpt output/airground_three_stream_cooperative_v3_receiver_target_qwen06b/best_val.pt --save-path output/eval_airground_coop_v3_receiver_target_fixed100

# metrics 
bash sh/calculate_eval_metrics.sh 