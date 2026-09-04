# 2026-08-15：AirGround-Coop V3 唯一版本清理

## 目标

将项目中的 AirGround-Coop 模型版本收敛为唯一的 canonical V3，永久删除其他版本的
模型、训练、评估、服务、配置、脚本、测试、文档、历史副本及对应实验输出。

## 最终 V3 入口

- 模型：`model_airground_coop_v3.py`
- 训练：`train_airground_coop_v3.py`
- 训练公共运行时：`train_airground_v3_common.py`
- 在线 planner：`eval_airground_coop_v3.py`
- 推理公共运行时：`eval_airground_v3_runtime.py`
- 服务：`eval_airground_coop_v3_server.py`
- UnrealZoo 闭环运行时：`eval_unrealzoo_multi_agent.py`
- 配置：`config/airground_cooperative_tracking_v3.yaml`
- 启动脚本：`sh/train_airground_coop_v3.sh`、`sh/eval_airground_coop_v3.sh`
- 回归测试：`tests/test_airground_coop_v3.py`、`tests/test_eval_airground_coop_v3.py`

原先由通用名称承载、但被 V3 直接依赖的训练与推理代码已经改名为显式 V3 runtime，
所有 V3 import、文档命令和 VS Code 调试入口同步更新。

## 删除范围

- 非 V3 的 AirGround-Coop 模型、训练、评估和 server 文件。
- 非 V3 的配置、shell 启动器、测试和版本文档。
- 已废弃的变体、实验性变体和版本历史目录。
- 旧 `change/`、`cundang/` 和 `multi_agent/` 模型/训练归档。
- 与上述版本一一对应的 checkpoint/eval 输出目录。

本次不删除数据集、UnrealZoo 环境、感知/分割工具、数据处理工具和第三方代码；这些是
V3 的运行依赖或独立辅助组件，不属于 AirGround-Coop 版本实现。

## 当前固定协议

- architecture：`airground_three_stream_cooperative_v3`
- 训练输出：`output/airground_three_stream_cooperative_v3_receiver_target_qwen06b`
- 固定评估 manifest：`manifests/eval_manifest_100.json`
- 固定评估输出：`output/eval_airground_coop_v3_receiver_target_fixed100`
- RobotDog waypoint 控制：仅 `v3_nonholonomic_projection`
- checkpoint 必须满足当前 receiver-recovery、ROI curriculum 和 directed relative-pose
  三项语义 contract。

## 验证

清理后结果：

1. V3 Python 文件编译检查通过；
2. V3 shell 脚本 `bash -n` 检查通过；
3. 两个 V3 pytest 回归测试文件共 `53 passed`；
4. V3 配置 dry-run 通过，固定 split 为 train=2142、val=913；
5. 固定评估 manifest 核对为 100 条；
6. canonical V3 范围内未发现失效的旧入口 import。
