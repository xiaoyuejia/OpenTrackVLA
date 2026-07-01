"""Habitat / EVT-Bench 原始单 Agent评估入口。

整体功能：
- 读取 Habitat 实验 YAML 和命令行覆盖参数。
- 按 ``split_id/split_num`` 切分评估数据集，便于多进程并行评估。
- 调用 ``tools.trained_agent.evaluate_agent`` 执行闭环跟踪并保存结果。

关键函数：
- ``main``：解析评估配置、数据切分和输出目录参数。
- ``run_exp``：创建 Habitat 数据集切片并启动评估。

本文件只负责 Habitat 单 Agent链路；UnrealZoo 双 Agent评估使用
``eval_unrealzoo_multi_agent.py``。
"""

# Use cleaned agent by default, fallback to original if needed
from tools.trained_agent import evaluate_agent
import argparse
import habitat
from habitat.datasets import make_dataset
import evt_bench
import numpy as np
import random


# ----------------------- 命令行参数与评估入口 -----------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["eval", "train"],
        required=True,
        help="run type",
    )

    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )

    parser.add_argument(
        "--split-id",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--split-num",
        type=int,
        default=7,
        required=False,
    )

    parser.add_argument(
        "--save-path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )

    args = parser.parse_args()
    run_exp(**vars(args))


# ----------------------- Habitat 数据切分与执行 -----------------------

def run_exp(run_type: str, exp_config: str, split_id: int, split_num: int, save_path: str, opts: None) -> None:
    config = habitat.get_config(exp_config, opts)
    random.seed(config.habitat.simulator.seed)
    np.random.seed(config.habitat.simulator.seed)

    dataset = make_dataset(id_dataset=config.habitat.dataset.type, config=config.habitat.dataset)
    dataset_split = dataset.get_splits(split_num)[split_id]

    if run_type == "eval":
        evaluate_agent(config, dataset_split, save_path)
    else:
        raise ValueError("Not supported now")
    
    return
 

if __name__ == "__main__":
    main()
