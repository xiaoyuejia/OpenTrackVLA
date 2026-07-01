# 主目录代码版本与环境导航

本项目同时保留 Habitat 原始评估、UnrealZoo 双 Agent MLP 版本和 UnrealZoo Anchor Diffusion 版本。
为了避免移动核心入口后破坏 shell、VS Code、内部导入和已有命令，主目录只保留可执行入口与共享模块，
具体环境和版本按下表区分。

本次为避免实验文件名含糊，已完成以下重命名：

| 旧文件名 | 当前文件名 |
|---|---|
| `model_copy.py` | `model_unrealzoo_anchor_diffusion.py` |
| `train_anchor_diffusion.py` | `train_unrealzoo_anchor_diffusion.py` |
| `build_trajectory_anchors.py` | `tools/build_unrealzoo_trajectory_anchors.py` |
| `run_unrealzoo_eval.py` | `eval_unrealzoo_multi_agent.py` |
| `run_eval.py` | `eval.py` |
| `run.py` | `tools/run.py`，使用 `python -m tools.run` 启动 |
| `trained_agent.py` | `tools/trained_agent.py` |
| `baseline_agent.py` | `tools/baseline_agent.py` |
| `requirements-anchor-diffusion.txt` | `docs/requirements-unrealzoo-anchor-diffusion.txt` |
| `md/ANCHOR_DIFFUSION_MODEL_COPY_ZH.md` | `docs/UNREALZOO_ANCHOR_DIFFUSION_MODEL_ZH.md` |

## 1. Habitat 原始单 Agent 版本

用途：运行原始 Habitat / EVT-Bench 数据、训练与闭环评估。

| 类型 | 文件 |
|---|---|
| 模型 | `model.py` 中的 `OpenTrackVLA` |
| 训练入口 | `train.py`，不传 `--multi_agent` |
| Habitat 评估入口 | `eval.py` |
| Habitat 在线 Agent | `tools/trained_agent.py` |
| Habitat 启动脚本 | `sh/eval.sh` |
| 数据采集 | `python -m tools.collect_sim_data` |
| 原始数据处理 | `python -m tools.make_tracking_data` 单 Agent 分支 |
| 视觉缓存 | `python -m tools.precache_frames` 单 Agent 分支 |
| 指标 | `python -m tools.calculate_metrics`、`python -m tools.calculate_metrics_single` |

典型命令：

```bash
bash sh/eval.sh
```

## 2. UnrealZoo 双 Agent MLP 版本

用途：使用无人机和机器狗联合跟踪目标，规划头为普通 MLP。

| 类型 | 文件 |
|---|---|
| 模型 | `model.py` 中的 `MultiAgentOpenTrackVLA` |
| 训练入口 | `train.py --multi_agent` |
| 数据处理 | `python -m tools.make_tracking_data --multi_agent` |
| 视觉缓存 | `python -m tools.precache_frames --multi_agent` |
| 完整训练流程 | `sh/run_multi_agent_pipeline.sh` |
| UnrealZoo 评估 | `eval_unrealzoo_multi_agent.py` |
| UnrealZoo 评估脚本 | `sh/eval_unrealzoo.sh` |
| 指标 | `python -m tools.calculate_unrealzoo_metrics` |

典型训练命令：

```bash
RUN_TRAIN=1 bash sh/run_multi_agent_pipeline.sh
```

`eval_unrealzoo_multi_agent.py` 会读取 checkpoint 并自动识别这是 MLP 版本。

### 2.1 手工采集数据整理

手工采集的 UnrealZoo 双 Agent 数据位于：

```text
sim_data/unrealzoo_aerial_ground_human_small/hand/
```

重新整理数据并汇总 `train.json`：

```bash
/home/hdt/miniconda3/envs/omtracknew/bin/python \
  -m tools.organize_hand_unrealzoo_data \
  --input-root sim_data/unrealzoo_aerial_ground_human_small/hand
```

整理结果位于：

```text
sim_data/unrealzoo_aerial_ground_human_small/hand/organized/
├── train.json
├── manifest.json
├── path_configs/
└── seed_hand/<scene>/0_drone.mp4 ...
```

每个场景内的 episode 使用普通数字 `0、1、2、...` 递增。整理时默认优先使用硬链接，
不会修改原始采集文件。

### 2.2 双 Agent 数据预处理与视觉缓存

推荐使用整理后的 `hand/organized` 作为输入，执行完整的数据预处理、视觉缓存和训练前检查：

```bash
INPUT_ROOT=/data/hdt/newtrackvla/sim_data/unrealzoo_aerial_ground_human_small/hand/organized \
DATA_ROOT=/data/hdt/newtrackvla/data/unrealzoo_aerial_ground_human_hand_multi \
RUN_MAKE_DATA=1 \
RUN_PRECACHE=1 \
RUN_DRY_RUN=1 \
RUN_TRAIN=0 \
CUDA_VISIBLE_DEVICES=1 \
bash sh/run_multi_agent_pipeline.sh
```

该命令依次执行：

| 开关 | 作用 | 主要输出 |
|---|---|---|
| `RUN_MAKE_DATA=1` | 从双视频和双 info 中抽帧并构造训练标签 | `frames/`、`jsonl/`、`dataset.json` |
| `RUN_PRECACHE=1` | 对 RGB 图片预计算视觉 Token | `vision_cache/` |
| `RUN_DRY_RUN=1` | 检查数据、缓存和张量形状 | 终端检查结果，不训练 |
| `RUN_TRAIN=0` | 禁止启动普通双 Agent MLP 训练 | 无 checkpoint |

预处理结果目录：

```text
data/unrealzoo_aerial_ground_human_hand_multi/
├── frames/         # Drone 和 RobotDog 视频抽出的 RGB 帧
├── jsonl/          # 每个 episode 的训练样本
├── dataset.json    # 汇总后的全部训练样本
└── vision_cache/   # 图片对应的视觉 Token 缓存
```

只重新生成数据、不生成视觉缓存：

```bash
INPUT_ROOT=/data/hdt/newtrackvla/sim_data/unrealzoo_aerial_ground_human_small/hand/organized \
DATA_ROOT=/data/hdt/newtrackvla/data/unrealzoo_aerial_ground_human_hand_multi \
RUN_MAKE_DATA=1 RUN_PRECACHE=0 RUN_DRY_RUN=0 RUN_TRAIN=0 \
bash sh/run_multi_agent_pipeline.sh
```

只生成或补全视觉缓存：

```bash
DATA_ROOT=/data/hdt/newtrackvla/data/unrealzoo_aerial_ground_human_hand_multi \
RUN_MAKE_DATA=0 RUN_PRECACHE=1 RUN_DRY_RUN=0 RUN_TRAIN=0 \
CUDA_VISIBLE_DEVICES=1 \
bash sh/run_multi_agent_pipeline.sh
```

注意：`run_multi_agent_pipeline.sh` 中的 `RUN_DRY_RUN=1` 调用的是
`train.py --multi_agent --dry_run`，它检查普通双 Agent 数据链路，不生成扩散轨迹锚点。

## 3. UnrealZoo 双 Agent Anchor Diffusion 版本

用途：使用 K-means 轨迹锚点、DiT 和两步 DDIM 生成多模态候选轨迹。

| 类型 | 文件 |
|---|---|
| 扩散模型 | `model_unrealzoo_anchor_diffusion.py` |
| 扩散训练入口 | `train_unrealzoo_anchor_diffusion.py` |
| 轨迹锚点生成 | `python -m tools.build_unrealzoo_trajectory_anchors` |
| 扩散依赖 | `docs/requirements-unrealzoo-anchor-diffusion.txt` |
| 扩散训练脚本 | `sh/train_anchor_diffusion.sh` |
| UnrealZoo 评估 | `eval_unrealzoo_multi_agent.py`，自动识别锚点 buffer |
| UnrealZoo 评估脚本 | `sh/eval_unrealzoo.sh` |
| 录制人体轨迹闭环评估 | `sh/eval_unrealzoo.sh`，设置 `RECORDED_TARGET_DIR` |
| 录制人体轨迹闭环说明 | `docs/UNREALZOO_RECORDED_TARGET_CLOSED_LOOP_ZH.md` |

### 3.1 生成轨迹锚点与扩散 dry-run

完成双 Agent 数据预处理后，首次运行扩散训练前需要生成轨迹锚点：

```bash
RUN_BUILD_ANCHORS=1 \
RUN_DRY_RUN=1 \
RUN_TRAIN=0 \
CUDA_VISIBLE_DEVICES=1 \
bash sh/train_anchor_diffusion.sh
```

`tools.build_unrealzoo_trajectory_anchors` 会读取：

```text
data/unrealzoo_aerial_ground_human_hand_multi/dataset.json
```

它分别收集 Drone 和 RobotDog 的真实未来 `waypoints`，使用 K-means 聚类得到默认
`40` 个轨迹模式。每个锚点包含 `8` 个 `(x, y, theta)` 路点，因此默认锚点形状为
`(40, 8, 3)`。

生成结果：

```text
data/unrealzoo_aerial_ground_human_hand_multi/trajectory_anchors/
├── agent1_drone_anchors.npy
└── agent2_robotdog_anchors.npy
```

`RUN_DRY_RUN=1` 会调用 `train_unrealzoo_anchor_diffusion.py --dry_run`，检查扩散版本所需的
训练数据、视觉缓存、锚点路径和张量形状，但不会正式训练。

### 3.2 单卡正式训练

`sh/train_anchor_diffusion.sh` 默认使用物理 GPU 1。锚点已生成后，执行：

```bash
DATASET_ROOT=/data/hdt/newtrackvla/data/unrealzoo_aerial_ground_human_hand_multi_2to1 \
RUN_BUILD_ANCHORS=0 \
RUN_DRY_RUN=0 \
RUN_TRAIN=1 \
EPOCHS=100 \
bash sh/train_anchor_diffusion.sh
```

`DATASET_ROOT` 只需指向包含 `train/` 和 `test/` 的最外层目录。脚本会自动使用：

```text
DATASET_ROOT/train/dataset.json
DATASET_ROOT/train/vision_cache/
DATASET_ROOT/train/trajectory_anchors/
```

checkpoint 默认输出到：

```text
ckpt/ckpts_multi_agent_anchor_diffusion_<DATASET_ROOT目录名>/
```

也可以显式指定物理 GPU 1：

```bash
CUDA_VISIBLE_DEVICES=1 \
DATASET_ROOT=/data/hdt/newtrackvla/data/unrealzoo_aerial_ground_human_hand_multi_2to1 \
RUN_BUILD_ANCHORS=0 RUN_DRY_RUN=0 RUN_TRAIN=1 EPOCHS=4 \
bash sh/train_anchor_diffusion.sh
```

### 3.3 多卡正式训练

使用物理 GPU 0 和 GPU 1：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
RUN_BUILD_ANCHORS=0 \
RUN_DRY_RUN=0 \
RUN_TRAIN=1 \
EPOCHS=4 \
bash sh/train_anchor_diffusion.sh
```

脚本会根据 `CUDA_VISIBLE_DEVICES` 自动计算 `NUM_GPUS=2`，并使用：

```text
python -m torch.distributed.run --standalone --nproc_per_node 2
```

训练输出默认写入：

```text
ckpt/ckpts_multi_agent_anchor_diffusion/
├── train_log.csv
├── model_epoch*.pt
└── ...
```

扩散模型训练时，以轨迹锚点加噪后的候选轨迹作为输入，通过 DiT 去噪预测候选轨迹，
同时学习候选轨迹评分。训练损失主要包含最近锚点轨迹回归损失、候选评分 BCE、
bbox 回归损失和目标可见性损失。

### 3.4 恢复训练

从输出目录中的最新 checkpoint 自动恢复：

```bash
RESUME=1 \
RUN_BUILD_ANCHORS=0 RUN_DRY_RUN=0 RUN_TRAIN=1 \
bash sh/train_anchor_diffusion.sh
```

指定 checkpoint 恢复：

```bash
RESUME=1 \
RESUME_CKPT=/data/hdt/newtrackvla/ckpt/ckpts_multi_agent_anchor_diffusion/model_epoch04_step000558_final.pt \
RUN_BUILD_ANCHORS=0 RUN_DRY_RUN=0 RUN_TRAIN=1 \
bash sh/train_anchor_diffusion.sh
```

实际 checkpoint 文件名采用 `model_epoch<epoch>_step<step>.pt` 或
`model_epoch<epoch>_step<step>_final.pt` 格式，请替换为输出目录中真实存在的文件。

### 3.5 典型评估命令

```bash
CKPT=ckpt/ckpts_multi_agent_anchor_diffusion \
  bash sh/eval_unrealzoo.sh
```

## 4. Checkpoint 目录

所有训练权重、训练日志和恢复训练状态统一放在 `ckpt/` 下：

| 目录 | 对应版本 | 默认使用入口 |
|---|---|---|
| `ckpt/ckpt_stt_filtered/` | Habitat 原始单 Agent STT | `sh/eval.sh` |
| `ckpt/ckpt_robotdog_1k/` | 机器狗单 Agent 实验 | 手动传入 `CKPT=...` |
| `ckpt/ckpts_multi_agent/` | UnrealZoo 双 Agent MLP | `train.py --multi_agent` |
| `ckpt/ckpts_multi_agent_anchor_diffusion/` | UnrealZoo 双 Agent Anchor Diffusion | `train_unrealzoo_anchor_diffusion.py`、`sh/eval_unrealzoo.sh` |
| `ckpt/ckpts_multi_agent_anchor_diffusion_debug/` | Anchor Diffusion 调试训练 | VS Code 调试配置 |
| `ckpt/ckpts_qwen4/` | Habitat 原始单 Agent 新训练默认目录 | `train.py` |

评估脚本接受 checkpoint 文件或目录。传目录时会自动选择其中修改时间最新的
`model_epoch*.pt`。

## 5. 共享代码

以下文件被多个环境或版本共同使用，不属于单独模型版本：

| 文件 | 作用 |
|---|---|
| `tools/cache_gridpool.py` | DINO/SigLIP 视觉特征编码与 token pooling |
| `tools/make_tracking_data.py` | 单 Agent 与双 Agent 数据转换 |
| `tools/precache_frames.py` | 单 Agent 与双 Agent 视觉缓存 |
| `tools/analyze_training_data.py` | 训练数据分析 |
| `tools/calculate_unrealzoo_metrics.py` | UnrealZoo 评估汇总 |
| `tools/trained_agent.py` | Habitat 模型闭环评估 Agent |
| `tools/baseline_agent.py` | 基线 Agent |
| `tools/convert_ckpt_to_hf.py` | checkpoint 转换 |
| `tools/view.py` | 可视化辅助工具 |

## 6. 目录职责

| 目录 | 职责 |
|---|---|
| `sh/` | 所有可直接执行的 shell 流程脚本 |
| `docs/` | 当前项目的中文说明、训练与评估文档 |
| `tools/` | 数据处理、视觉缓存、指标计算和 checkpoint 维护工具 |
| `ckpt/` | 按模型版本分类保存 checkpoint 与训练日志 |
| `habitat-lab/` | Habitat 环境与任务实现 |
| `unrealzoo-gym/` | UnrealZoo 环境实现 |
| `evt_bench/` | EVT-Bench 扩展 |
| `multi_agent/` | 早期独立双 Agent 实验实现，当前主流程优先使用主目录入口 |

### tools 运行规则

从仓库根目录运行 `tools/` 中的 Python 工具时，统一使用模块形式：

```bash
python -m tools.make_tracking_data --help
python -m tools.precache_frames --help
python -m tools.build_unrealzoo_trajectory_anchors --help
python -m tools.calculate_unrealzoo_metrics --help
python -m tools.run --help
```

模块形式会把仓库根目录加入 Python 导入路径，因此工具可以稳定导入
`model.py`、`train.py` 和其他 `tools.*` 模块。

## 7. 模型选择规则

训练时模型版本由入口决定：

```text
train.py                     -> model.py
train.py --multi_agent       -> model.py 的双 Agent MLP
train_unrealzoo_anchor_diffusion.py    -> model_unrealzoo_anchor_diffusion.py 的双 Agent Anchor Diffusion
```

UnrealZoo 评估时由 checkpoint 自动决定：

```text
checkpoint 包含 planner_agent1.anchors
    -> model_unrealzoo_anchor_diffusion.py Anchor Diffusion

否则
    -> model.py MLP planner
```

Habitat 评估仍固定使用原始 Habitat 链路，不自动加载双 Agent UnrealZoo checkpoint。

## 8. 后续新增代码的放置规则

- 新的模型版本：使用明确模块名，例如 `model_<version>.py`，不要再次使用带空格文件名。
- 新的训练入口：使用 `train_<version>.py`。
- 新的环境评估入口：使用 `eval_<environment>_<version>.py`。
- 完整命令流程：放入 `sh/`。
- 说明和实验记录：放入 `docs/`。
- 可被多个版本复用的功能：抽到明确命名的共享模块，避免复制到多个模型文件。

按照以上规则，可以同时保留不同实验版本，又不会让主目录入口失去可追踪性。
