#!/usr/bin/env python3
"""
计算 EVT-Bench 评估指标: SR (Success Rate), TR (Track Rate), CR (Collision Rate)

使用方法:
    python -m tools.calculate_metrics --eval-dir /data/hdt/ntv_data/sim_data/eval/stt
    python -m tools.calculate_metrics --eval-dir /data/hdt/ntv_data/sim_data/eval/dt
    python -m tools.calculate_metrics --eval-dir /data/hdt/ntv_data/sim_data/eval/at
    python -m tools.calculate_metrics --eval-dir /data/hdt/ntv_data/sim_data/eval  # 计算所有任务类型
"""

import json
import glob
import os
import argparse
from collections import defaultdict


def calculate_metrics_for_dir(eval_dir):
    """
    计算单个目录的 SR, TR, CR 指标

    Args:
        eval_dir: 评估结果目录路径

    Returns:
        dict: 包含 SR, TR, CR 和统计信息
    """
    # 查找所有 JSON 文件（排除 _info.json）
    json_files = glob.glob(os.path.join(eval_dir, "**/*.json"), recursive=True)
    json_files = [f for f in json_files if not f.endswith("_info.json")]

    if not json_files:
        return None

    total_episodes = 0
    success_count = 0
    collision_count = 0
    following_rates = []

    status_counts = defaultdict(int)

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                total_episodes += 1
                success_count += float(data.get('success', 0))
                collision_count += float(data.get('collision', 0))
                following_rates.append(float(data.get('following_rate', 0)))
                status_counts[data.get('status', 'Unknown')] += 1
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Failed to parse {json_file}: {e}")
            continue

    if total_episodes == 0:
        return None

    SR = (success_count / total_episodes) * 100
    TR = (sum(following_rates) / len(following_rates)) * 100
    CR = (collision_count / total_episodes) * 100

    return {
        'total_episodes': total_episodes,
        'SR': SR,
        'TR': TR,
        'CR': CR,
        'success_count': int(success_count),
        'collision_count': int(collision_count),
        'status_counts': dict(status_counts)
    }


def main():
    parser = argparse.ArgumentParser(description='计算 EVT-Bench 评估指标')
    parser.add_argument('--eval-dir', type=str, default='/data/hdt/ntv_data/sim_data/eval',
                        help='评估结果目录路径 (默认: /data/hdt/ntv_data/sim_data/eval)')
    args = parser.parse_args()

    eval_dir = args.eval_dir

    if not os.path.exists(eval_dir):
        print(f"Error: 目录不存在: {eval_dir}")
        return

    # 检查是否是包含多个任务类型的根目录
    task_types = ['stt', 'dt', 'at']
    subdirs = [d for d in task_types if os.path.isdir(os.path.join(eval_dir, d))]

    if subdirs:
        # 多任务类型目录
        print("=" * 70)
        print("EVT-Bench 评估结果汇总")
        print("=" * 70)
        print()
        print(f"{'Task':<10} {'Episodes':<10} {'SR↑':<12} {'TR↑':<12} {'CR↓':<12}")
        print("-" * 56)

        all_results = {}
        for task in subdirs:
            task_dir = os.path.join(eval_dir, task)
            metrics = calculate_metrics_for_dir(task_dir)
            if metrics:
                all_results[task.upper()] = metrics
                print(f"{task.upper():<10} {metrics['total_episodes']:<10} "
                      f"{metrics['SR']:<12.2f} {metrics['TR']:<12.2f} {metrics['CR']:<12.2f}")

        print("-" * 56)
        print()

        # 打印详细统计
        for task, metrics in all_results.items():
            print(f"\n{task} 详细统计:")
            print(f"  总 Episodes: {metrics['total_episodes']}")
            print(f"  成功数: {metrics['success_count']}")
            print(f"  碰撞数: {metrics['collision_count']}")
            print(f"  状态分布: {metrics['status_counts']}")

        # 打印表格格式（便于复制到论文）
        print("\n" + "=" * 70)
        print("表格格式 (SR↑ / TR↑ / CR↓):")
        print("-" * 70)
        for task, metrics in all_results.items():
            print(f"{task}: {metrics['SR']:.1f} / {metrics['TR']:.1f} / {metrics['CR']:.2f}")
    else:
        # 单任务类型目录
        metrics = calculate_metrics_for_dir(eval_dir)
        if metrics:
            print("=" * 50)
            print(f"评估结果: {eval_dir}")
            print("=" * 50)
            print(f"总 Episodes: {metrics['total_episodes']}")
            print(f"SR (Success Rate): {metrics['SR']:.2f}%")
            print(f"TR (Track Rate): {metrics['TR']:.2f}%")
            print(f"CR (Collision Rate): {metrics['CR']:.2f}%")
            print()
            print(f"成功数: {metrics['success_count']}")
            print(f"碰撞数: {metrics['collision_count']}")
            print(f"状态分布: {metrics['status_counts']}")
            print()
            print(f"表格格式: {metrics['SR']:.1f} / {metrics['TR']:.1f} / {metrics['CR']:.2f}")
        else:
            print(f"Error: 在 {eval_dir} 中未找到评估结果文件")


if __name__ == "__main__":
    main()
