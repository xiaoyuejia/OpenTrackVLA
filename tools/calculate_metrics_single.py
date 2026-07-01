import os
import json
import glob
from collections import defaultdict

def calculate_metrics(save_path):
    # Find all .json files in the save_path directory recursively
    json_files = glob.glob(os.path.join(save_path, '**', '*.json'), recursive=True)
    
    results = []
    for file_path in json_files:
        # Skip _info.json files, only process the main result files
        if '_info.json' in file_path:
            continue
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    if not results:
        print("No result files found.")
        return
    
    # Calculate metrics
    total_episodes = len(results)
    success_count = sum(1 for r in results if r.get('success', False))
    collision_count = sum(1 for r in results if r.get('collision', False))
    tracking_rates = [r.get('following_rate', 0) for r in results if 'following_rate' in r]
    
    success_rate = success_count / total_episodes * 100
    collision_rate = collision_count / total_episodes * 100
    tracking_rate = sum(tracking_rates) / len(tracking_rates) * 100 if tracking_rates else 0
    
    print(f"Total episodes: {total_episodes}")
    print(f"Success Rate (SR): {success_rate:.2f}%")
    print(f"Tracking Rate (TR): {tracking_rate:.2f}%")
    print(f"Collision Rate (CR): {collision_rate:.2f}%")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-path", type=str, default="/data/hdt/ntv_data/sim_data/eval/stt_train", help="Path to the evaluation results")
    args = parser.parse_args()
    calculate_metrics(args.save_path)
    # 修改目录用于结果：--save-path /data/hdt/ntv_data/sim_data/eval/dt
