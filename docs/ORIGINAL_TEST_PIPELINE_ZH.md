# 原始 OpenTrackVLA 训练后测试流程解析

## 概览

原始代码中，训练后的模型有两类测试方式：

1. **离线测试**
   - 输入已经处理好的 JSON/JSONL 数据和视觉 token cache。
   - 模型直接预测 waypoint。
   - 可计算 waypoint MSE、最终路点误差 EPE 和 hit rate。
   - 主要实现在 `train.py`。

2. **Habitat / EVT-Bench 闭环测试**
   - 模型在 Habitat 仿真环境中逐步接收机器人当前相机图像。
   - 在线编码视觉 token，预测 waypoint，再转成机器人速度动作。
   - 环境执行动作后返回新的观测和跟踪指标。
   - 最终计算 SR、TR、CR。
   - 主要由 `sh/eval.sh -> eval.py -> tools/trained_agent.py -> tools/calculate_metrics.py` 完成。

当前闭环测试代码加载的是原始单 Agent `OpenTrackVLA`。主目录新增的双 Agent 模型还没有接入 `tools/trained_agent.py`。

如果要在新的 UnrealZoo 双 Agent 环境中评估 `MultiAgentOpenTrackVLA`，请看独立文档：

```text
UNREALZOO_EVAL_ZH.md
```

新的 UnrealZoo 入口不会替换原 Habitat 入口：

```text
Habitat:   bash sh/eval.sh
UnrealZoo: bash sh/eval_unrealzoo.sh
```

## 简要说明

并不是每种评估都会调用仿真环境：

- **离线评估不调用仿真环境**：直接读取 JSONL、GT waypoint 和缓存好的视觉特征，让模型预测 waypoint，再计算 MSE、EPE 和 Hit Rate。它速度较快，适合监控训练效果、筛选 checkpoint、比较模型版本和排查轨迹预测问题。
- **闭环评估会调用仿真环境**：每条 episode 在 Habitat 中逐步运行。每一步获取当前 RGB，在线执行 DINO + SigLIP 编码，模型预测 waypoint，将 waypoint 转换为速度动作，再交给仿真环境执行。

```text
离线评估:
缓存视觉特征 + GT waypoint -> 模型预测 -> 与 GT 对比 -> MSE / EPE / Hit Rate

闭环评估:
仿真 RGB -> 在线视觉编码 -> 模型预测 -> 执行动作 -> 下一帧 -> SR / TR / CR
```

离线评估通常明显快于闭环评估，因为它不需要运行 Habitat，也不需要重复执行在线视觉编码和机器人动作。离线预测准确并不代表闭环控制一定稳定，因此推荐先用离线评估筛选 checkpoint，再使用闭环评估检查持续跟踪、丢失和碰撞情况。

---

## 测试代码入口

| 测试方式 | 入口 | 核心函数 | 输出 |
|---|---|---|---|
| 训练中验证集测试 | `train.py` | `train()` 中 periodic evaluation | MSE、EPE、hit rate |
| 单 episode 测试 | `train.py` | `train()` 中 single-episode evaluation | EPE、follow rate |
| 离线推理 | `train.py --infer_json ...` | `_run_inference()` | NPZ、轨迹可视化图片 |
| Habitat 闭环评估 | `sh/eval.sh` | `eval.py -> tools.trained_agent.evaluate_agent()` | episode JSON、step JSON、视频 |
| EVT-Bench 指标汇总 | `tools/calculate_metrics.py` | `calculate_metrics_for_dir()` | SR、TR、CR |

### 关键源码定位

| 文件位置 | 作用 |
|---|---|
| `train.py:420` `JsonTrackingDataset` | 读取离线 JSON/JSONL、历史帧和视觉 cache |
| `train.py:739` `collate_batch()` | 将单条样本拼成 batch |
| `train.py:755` `train()` | 训练、周期验证和单 episode 离线验证 |
| `train.py:1403` `_run_inference()` | checkpoint 离线推理、NPZ 和可视化输出 |
| `sh/eval.sh` | checkpoint 解析、GPU 分配和评估 split 并行调度 |
| `eval.py` `run_exp()` | 加载 Habitat 配置、数据集和指定 split |
| `tools/trained_agent.py:44` `evaluate_agent()` | 执行闭环 episode、记录结果 |
| `tools/trained_agent.py:179` `GTBBoxAgent` | 加载模型、在线编码图像、输出动作 |
| `tools/trained_agent.py:424` `_planner_action()` | waypoint 预测及速度动作转换 |
| `evt_bench/additional_metric.py:130` `HumanCollision` | 判断是否曾接近目标人至 0.5 以内 |
| `evt_bench/additional_metric.py:237` `HumanFollowing` | 判断当前是否正在跟踪 |
| `evt_bench/additional_metric.py:283` `HumanFollowingSuccess` | 判断 stop 时是否满足成功条件 |
| `tools/calculate_metrics.py:19` `calculate_metrics_for_dir()` | 汇总 SR、TR、CR |

---

## 总体测试数据流

```text
训练完成
──────────────────────────────────────────────────────────────────────────────
│                                                                            │
│  checkpoint: model_epoch*.pt                                                │
│                                                                            │
│  {                                                                         │
│    epoch, step,                                                            │
│    model_state,          # 模型参数                                         │
│    optim_state,          # 优化器状态，测试时不使用                          │
│    scaler_state,         # AMP 状态，测试时不使用                            │
│    config                # llm_name/n_waypoints/alpha_xy/...                 │
│  }                                                                         │
│                                                                            │
──────────────────────────────────────────────────────────────────────────────
                         │
          ┌──────────────┴────────────────┐
          ▼                               ▼
离线 JSONL 测试                     Habitat 闭环测试
──────────────────────────          ──────────────────────────────────────────
JsonTrackingDataset                 sh/eval.sh
读取 JSONL + vision_cache                 │
          │                               ▼
          ▼                         eval.py
coarse/fine tokens                  加载 Habitat 配置和评估集
          │                               │
          ▼                               ▼
OpenTrackVLA                        trained_agent.GTBBoxAgent
          │                         加载 checkpoint
          ▼                               │
waypoints: (B,8,3)                       ▼
          │                         当前 RGB 图像
          ├─ 与 GT waypoint 比较           │
          │  MSE / EPE / hit              ▼
          │                         在线视觉编码
          └─ 保存 NPZ / 可视化             │
                                          ▼
                                    OpenTrackVLA waypoint
                                          │
                                          ▼
                                    waypoint -> [vx,vy,wz]
                                          │
                                          ▼
                                    env.step(action)
                                          │
                                          ▼
                                    following/collision/success
                                          │
                                          ▼
                                    episode JSON
                                          │
                                          ▼
                                    tools/calculate_metrics.py
                                    SR / TR / CR
```

---

# 第一部分：离线测试

## 1. 训练中的验证集测试

相关代码位于 `train.py` 的单 Agent `train()` 函数中。

触发条件：

```python
cfg.eval_every > 0
step % cfg.eval_every == 0
cfg.val_json is not None
rank == 0
```

### 1.1 验证数据读取

```python
vds = JsonTrackingDataset(
    DataConfig(
        train_json=cfg.val_json,
        n_waypoints=cfg.n_waypoints,
        history=cfg.history,
        cache_root=cfg.cache_root,
    )
)
```

验证数据和训练数据使用相同的 `JsonTrackingDataset`。

单条样本输出：

```text
coarse_tokens: (H*4, C)      默认 (124,1536)
coarse_tidx:   (H*4)         默认 (124)
fine_tokens:   (64,C)        默认 (64,1536)
fine_tidx:     (64)
yaw_hist:      (H)
yaw_curr:      (1)
waypoints:     (N,3)         默认 (8,3)
valid_mask:    (N)           默认 (8)
instruction:   str
```

经过 `collate_batch()` 后：

```text
coarse_tokens: (B,124,1536)
fine_tokens:   (B,64,1536)
waypoints:     (B,8,3)
valid_mask:    (B,8)
```

### 1.2 模型前向

```python
pred = model_eval(
    coarse_tokens,
    coarse_tidx,
    fine_tokens,
    fine_tidx,
    instr,
    yaw_hist=yaw_hist if cfg.use_angle_tvi else None,
    yaw_curr=yaw_curr if cfg.use_angle_tvi else None,
)
```

输出：

```text
pred: (B, n_waypoints, 3)
默认: (B, 8, 3)
```

每个 waypoint 为：

```text
[x, y, theta]
```

其中 `x/y` 已经在模型内部乘过 `alpha_task`，属于实际尺度。

### 1.3 验证指标

#### Masked MSE

代码先把预测和 GT 的 XY 除以 `alpha_xy`，回到 normalized space：

```python
pred_n[..., 0:2] = pred[..., 0:2] / alpha_xy
gt_n[..., 0:2] = gt[..., 0:2] / alpha_xy
mse = mse_masked(pred_n, gt_n, valid_mask)
```

只对 `valid_mask=True` 的 waypoint 计算 MSE。

#### 最终路点 EPE

```python
pred_xy = pred[:, -1, :2]
gt_xy = gt_wp[:, -1, :2]
epe = torch.linalg.norm(pred_xy - gt_xy, dim=-1)
```

含义：

```text
final_EPE = 最后一个预测 waypoint 与 GT waypoint 的 XY 欧氏距离
```

#### Hit Rate

```python
hit = final_EPE <= final_wp_threshold
```

默认阈值：

```text
final_wp_threshold = 0.2
```

验证日志格式：

```text
[VAL] step ... |
masked_MSE=... |
final_EPE_mean=... |
final_EPE_median=... |
hit@0.2=...
```

### 1.4 当前限制

`TrainConfig` 中存在：

```python
val_json: Optional[str]
```

训练函数也实现了验证集逻辑，但原单 Agent `parse_args()` 当前没有添加 `--val_json` 参数。因此通过原命令行不能直接启用 periodic validation，需要：

- 在代码中构造 `TrainConfig(val_json=...)`；或
- 给原单 Agent `parse_args()` 增加 `--val_json`。

---

## 2. 单 Episode 验证

相关配置：

```text
episode_json
episode_eval_every
episode_threshold
episode_max_frames
```

触发条件：

```python
cfg.episode_eval_every > 0
step % cfg.episode_eval_every == 0
cfg.episode_json is not None
rank == 0
```

### 数据流

```text
单 episode JSON/JSONL
    │
    ▼
JsonTrackingDataset
    │
    ▼
逐帧读取 item
    │
    ▼
OpenTrackVLA
    │
    ▼
最终 waypoint XY EPE
    │
    ▼
EPE <= episode_threshold
    │
    ▼
follow rate
```

计算方式：

```python
follow_rate = 命中阈值的帧数 / 已评估帧数
```

日志：

```text
[EPISODE] step ... |
frames=... |
EPE_mean=... |
EPE_median=... |
follow@0.2=...
```

这仍然是离线 waypoint 准确度测试，不会把动作真正执行进仿真环境。

---

## 3. 离线推理 `_run_inference()`

原代码支持只加载 checkpoint 做离线推理：

```bash
python train.py \
  --train_json <任意占位训练路径> \
  --epochs 0 \
  --infer_json <测试JSON或JSONL目录> \
  --infer_ckpt <checkpoint路径> \
  --infer_out ./infer_out \
  --infer_save_npz \
  --infer_vis
```

注意：原 CLI 仍要求提供 `--train_json`，即使 `epochs=0` 只做推理。

### 3.1 checkpoint 加载

如果没有指定 `--infer_ckpt`，代码会从 `out_dir` 中选择最新修改时间的：

```text
model_epoch*.pt
```

然后读取 checkpoint 内保存的配置：

```python
n_waypoints
use_angle_tvi
no_tanh_actions
vision_feat_dim
alpha_xy
llm_name
beta_nav
```

使用这些配置重新构造 `OpenTrackVLA`，再加载：

```python
checkpoint["model_state"]
```

### 3.2 推理数据流

```text
infer_json
    │
    ▼
JsonTrackingDataset + vision_cache
    │
    ▼
DataLoader / collate_batch
    │
    ▼
OpenTrackVLA.eval()
    │
    ▼
pred waypoint: (B,8,3)
    │
    ├── 保存 NPZ
    └── 在当前帧上绘制预测轨迹
```

### 3.3 NPZ 输出

保存目录：

```text
infer_out/npz/
```

每个 NPZ 包含：

```text
pred:         (8,3)
instruction:  str
current_path: str
```

这里没有保存 GT，也没有自动计算 MSE/EPE。

### 3.4 可视化输出

保存目录：

```text
infer_out/vis/
```

绘制逻辑：

```text
屏幕原点:
x = 图像宽度 / 2
y = 图像高度 * 0.86

轨迹坐标映射:
pixel_x = base_x - y * 120
pixel_y = base_y - x * 120
```

预测轨迹使用青绿色绘制。

---

# 第二部分：Habitat / EVT-Bench 闭环测试

## 4. 闭环测试入口 `sh/eval.sh`

推荐入口：

```bash
CKPT=/data/hdt/newtrackvla/ckpt/ckpts_qwen4/model_epochXX.pt \
GPUS=0,1,2 \
NUM_PARALLEL=3 \
CHUNKS=30 \
SAVE_PATH=sim_data/eval/stt \
EXP_CONFIG=habitat-lab/habitat/config/benchmark/nav/track/track_infer_stt.yaml \
bash sh/eval.sh
```

### 4.1 参数作用

| 参数 | 作用 |
|---|---|
| `CKPT` | 手动指定 checkpoint，优先级最高 |
| `CKPT_DIR` | 未指定 `CKPT` 时，从此目录查找最新 `model_epoch*.pt` |
| `CHUNKS` | 把 Habitat 评估集切成多少份 |
| `NUM_PARALLEL` | 同时运行多少个评估进程 |
| `GPUS` | 并行任务使用的 GPU，例如 `0,1,2` |
| `SAVE_PATH` | episode JSON、逐步信息和视频保存目录 |
| `EXP_CONFIG` | Habitat 任务配置 |
| `USE_HF` / `HF_MODEL_DIR` | 是否使用 HuggingFace 格式模型 |

### 4.2 并行评估流程

```text
完整评估数据集
    │
    ▼
dataset.get_splits(CHUNKS)
    │
    ├── split 0 -> GPU 0 -> eval.py
    ├── split 1 -> GPU 1 -> eval.py
    ├── split 2 -> GPU 2 -> eval.py
    └── ...
```

`eval.sh` 会把 `CKPT` 导出为环境变量：

```bash
export CKPT
```

`tools/trained_agent.py` 后续从这个环境变量加载模型。

---

## 5. Habitat 配置和数据集切分

`eval.py` 负责：

```python
config = habitat.get_config(exp_config, opts)
dataset = make_dataset(
    id_dataset=config.habitat.dataset.type,
    config=config.habitat.dataset,
)
dataset_split = dataset.get_splits(split_num)[split_id]
evaluate_agent(config, dataset_split, save_path)
```

原代码提供三类推理配置：

```text
track_infer_stt.yaml
track_infer_dt.yaml
track_infer_at.yaml
```

分别引用：

```text
track-val-stt
track-val-dt
track-val-at
```

以 STT 配置为例：

```text
max_episode_steps: 300
simulator seed: 100
ctrl_freq: 40
robot: agent_1 spot
target human: agent_0
robot action: agent_1_base_velocity
```

机器人速度限制：

```text
longitudinal_lin_speed: 15.0
lateral_lin_speed: 10.0
ang_speed: 6.28
allow_back: True
enable_lateral_move: True
```

---

## 6. checkpoint 加载流程

核心类：

```python
trained_agent.GTBBoxAgent
```

初始化时执行：

```text
GTBBoxAgent.__init__()
    │
    ├── _resolve_planner_ckpt_once()
    │      ├── HF_MODEL_DIR
    │      ├── HF_MODEL_ID
    │      ├── 环境变量 CKPT
    │      ├── 默认 ckpts/ 中最新 model_epoch*.pt
    │      └── 默认 open_trackvla_hf/
    │
    └── _init_planner_model()
           ├── 从 checkpoint["config"] 恢复模型配置
           ├── 构造 OpenTrackVLA
           ├── 加载 checkpoint["model_state"]
           └── model.eval()
```

Legacy `.pt` checkpoint 会先加载到 CPU：

```python
obj = torch.load(checkpoint_path, map_location="cpu")
```

这样可以避免直接把完整训练 checkpoint 加载到 GPU 时产生较大的显存峰值。

从 checkpoint 恢复的主要配置：

```text
llm_name
n_waypoints
beta_nav
use_angle_tvi
no_tanh_actions
alpha_xy
vision_feat_dim
```

如果权重键带有 DDP 的 `module.` 前缀，会先去掉再加载。

---

## 7. 单步闭环推理数据流

每个 Habitat step 都会调用：

```python
action = robot_config.act(obs, detector, episode_id, instruction)
```

完整数据流：

```text
Habitat 当前观测
    │
    ▼
agent_1_articulated_agent_jaw_rgb
RGB 当前帧: (H,W,3)
    │
    ▼
VisionFeatureCacher
DINO + SigLIP
    │
    ▼
Vcoarse: (4,1536)
Vfine:   (64,1536)
    │
    ├── Vcoarse 放入历史 deque，最大长度 H=31
    │
    └── Vfine 作为当前帧细粒度 token
    │
    ▼
构造模型输入
coarse_tokens: (1,124,1536)
coarse_tidx:   (1,124)
fine_tokens:   (1,64,1536)
fine_tidx:     (1,64)
instruction:   List[str], 长度 1
    │
    ▼
OpenTrackVLA
    │
    ▼
tau / predicted waypoint: (1,8,3)
    │
    ▼
选择 tau[0,1]
    │
    ▼
[x,y,theta] / dt=0.1
    │
    ▼
action = [vx,vy,wz]
    │
    ▼
env.step(action_dict)
    │
    ▼
环境返回下一帧和 metrics
```

### 7.1 在线视觉编码

每一步都在线执行：

```python
tok_dino = DINO(current_rgb)
tok_siglip = SigLIP(current_rgb)
Vt_cat = concat(tok_dino, tok_siglip)
Vfine = grid_pool(..., out_tokens=64)
Vcoarse = grid_pool(..., out_tokens=4)
```

与离线训练不同，闭环测试不读取磁盘上的 `vision_cache`，而是实时编码当前仿真画面。

### 7.2 历史帧构造

```python
self._coarse_hist_tokens = deque(maxlen=31)
```

每一步把当前 `Vcoarse` 加入历史。

历史不足 31 帧时：

```text
使用最早可用帧的 coarse token 左侧补齐
```

最终：

```text
coarse_tokens: (1, 31*4, 1536) = (1,124,1536)
```

### 7.3 waypoint 转控制动作

模型输出：

```text
tau: (1,8,3)
```

当前实现使用：

```python
wp0 = tau[0, 1]
```

也就是使用索引 `1` 的 waypoint，而不是索引 `0`。

然后：

```python
dt = 0.1
vx = x / dt
vy = y / dt
wz = theta / dt
```

输出给 Habitat：

```text
action = [vx, vy, wz]
```

动作被写入：

```python
action_dict["action_args"]["agent_1_base_vel"]
```

---

## 8. Episode 终止与结果保存

闭环评估循环：

```python
while not env.episode_over:
    获取观测
    模型预测动作
    env.step(action)
    读取 metrics
    判断 Lost / Collision
```

### 8.1 Lost 判定

当机器人与目标人的距离超过 `4.0`：

```python
too_far_count += 1
```

连续超过 20 步：

```text
status = "Lost"
结束当前 episode
```

### 8.2 Collision 判定

如果：

```python
info["human_collision"] == 1.0
```

则：

```text
status = "Collision"
结束当前 episode
```

`HumanCollision` 的底层实现是：机器人与目标人的距离小于 `0.5` 时记为碰撞，并保持 episode 内曾碰撞状态。

### 8.3 Human Following 判定

每一步：

```text
human_following = 1
```

要求：

```text
机器人与目标人的距离 <= 3.0
且 detector 判断机器人正面对目标
```

### 8.4 Success 判定

`HumanFollowingSuccess` 的距离要求：

```text
1.0 <= distance_to_target <= 3.0
并且 human_following=True
并且任务调用了 stop
```

`tools/trained_agent.py` 保存 episode 结果时：

```python
if iter_step < 300:
    success = human_following_success and human_following
else:
    success = human_following
```

### 8.5 输出文件结构

```text
SAVE_PATH/
  <scene_name>/
    <episode_id>.json
    <episode_id>_info.json
    <episode_id>.mp4
```

`<episode_id>.json`：

```json
{
  "finish": false,
  "status": "Lost",
  "success": 0.0,
  "following_rate": 0.0,
  "following_step": 0,
  "total_step": 28,
  "collision": 0.0,
  "instruction": "Walk after the woman ..."
}
```

`<episode_id>_info.json` 每一步保存：

```json
{
  "step": 1,
  "dis_to_human": 2.5,
  "facing": 1.0,
  "base_velocity": [vx, vy, wz]
}
```

视频会在 RGB 画面上绘制模型预测轨迹。

---

## 9. EVT-Bench 指标计算

运行：

```bash
python -m tools.calculate_metrics --eval-dir sim_data/eval/stt
```

脚本递归查找：

```text
**/*.json
```

并排除：

```text
*_info.json
```

### 9.1 Success Rate

```text
SR = sum(success) / episode数量 * 100%
```

越高越好。

### 9.2 Track Rate

单 episode：

```text
following_rate = followed_step / total_step
```

整体：

```text
TR = 所有 episode following_rate 的平均值 * 100%
```

越高越好。

### 9.3 Collision Rate

```text
CR = sum(collision) / episode数量 * 100%
```

越低越好。

最终输出格式：

```text
Task       Episodes   SR↑          TR↑          CR↓
STT        ...        ...          ...          ...
DT         ...        ...          ...          ...
AT         ...        ...          ...          ...
```

---

## 原始闭环评估完整命令示例

### STT

```bash
CKPT=/data/hdt/newtrackvla/ckpt/ckpt_stt_filtered/model_epochXX.pt \
GPUS=0,1,2 \
NUM_PARALLEL=3 \
CHUNKS=30 \
SAVE_PATH=sim_data/eval/stt \
EXP_CONFIG=habitat-lab/habitat/config/benchmark/nav/track/track_infer_stt.yaml \
bash sh/eval.sh

python -m tools.calculate_metrics --eval-dir sim_data/eval/stt
```

### DT

```bash
CKPT=/data/hdt/newtrackvla/ckpt_dt/model_epochXX.pt \
SAVE_PATH=sim_data/eval/dt \
EXP_CONFIG=habitat-lab/habitat/config/benchmark/nav/track/track_infer_dt.yaml \
bash sh/eval.sh

python -m tools.calculate_metrics --eval-dir sim_data/eval/dt
```

### AT

```bash
CKPT=/data/hdt/newtrackvla/ckpt_at/model_epochXX.pt \
SAVE_PATH=sim_data/eval/at \
EXP_CONFIG=habitat-lab/habitat/config/benchmark/nav/track/track_infer_at.yaml \
bash sh/eval.sh

python -m tools.calculate_metrics --eval-dir sim_data/eval/at
```

---

## 当前代码中的重要注意事项

### 1. 闭环测试当前只支持原单 Agent 模型

`tools/trained_agent.py` 当前导入：

```python
from model import OpenTrackVLA as PlannerModel
from model import ModelConfig as PlannerConfig
```

它没有使用：

```python
MultiAgentOpenTrackVLA
MultiAgentModelConfig
```

因此双 Agent checkpoint 不能直接交给当前 `sh/eval.sh` 闭环测试。即使 `strict=False` 加载，也会出现大量模型键不匹配，并且输入数据结构不满足双 Agent 模型要求。

### 2. Habitat 当前只控制 agent_1 机器人

当前动作：

```python
"agent_1_base_vel": action
```

没有无人机动作输出，也没有同时执行两个 Agent 的 action。

要测试双 Agent 模型，需要扩展：

- Habitat/UnrealZoo 双 Agent 环境；
- 同步获取 drone/robotdog 两路 RGB；
- 构造 `(B,2,...)` 模型输入；
- 把两个 waypoint 分别转换为两个 Agent 动作；
- 分别统计两个 Agent 跟踪指标。

### 3. detector 参数当前没有被模型使用

闭环循环会计算：

```python
detector = env.task._get_observations(...)
```

并传给：

```python
robot_config.act(obs, detector, ...)
```

但 `GTBBoxAgent.act()` 当前没有使用 `detector` 构造 bbox token，模型只使用 RGB 和 instruction。

### 4. `SAVE_VIDEO` 环境变量当前没有真正控制保存逻辑

`sh/eval.sh` 会传入：

```bash
SAVE_VIDEO=...
```

但 `tools/trained_agent.py` 当前没有读取该环境变量。只要 `rgb_list` 非空，`reset()` 就会调用 `imageio.mimsave()` 保存视频。

### 5. 在线推理速度包含视觉编码成本

每个仿真 step 都会在线运行 DINO + SigLIP。闭环测试速度不只包含 LLM/planner 推理，也包含视觉双塔编码时间。

### 6. 控制动作可能需要限幅

当前直接执行：

```text
[vx,vy,wz] = waypoint / 0.1
```

代码没有在 `_planner_action()` 中显式 clamp。最终限制主要依赖 Habitat action 配置。若 waypoint 较大，可能产生激进动作。

### 7. 原离线推理不会自动计算指标

`_run_inference()` 主要保存预测 NPZ 和可视化图，不会自动与 GT 比较。若需要完整离线测试报告，应额外保存 GT 并计算：

- masked MSE；
- ADE；
- FDE / final EPE；
- hit rate。

---

## 测试方式如何选择

| 目标 | 推荐方式 |
|---|---|
| 快速确认 checkpoint 能正常加载 | `train.py --epochs 0 --infer_json ...` |
| 比较 waypoint 预测准确度 | 验证集 MSE、EPE、hit rate |
| 查看单 episode 轨迹误差 | single-episode evaluation |
| 测试模型执行后能否持续跟踪目标 | Habitat / EVT-Bench 闭环评估 |
| 汇总论文指标 | `tools/calculate_metrics.py` 的 SR/TR/CR |

离线 waypoint 准确并不一定代表闭环跟踪效果好。真正评价跟踪控制能力时，应以闭环 SR/TR/CR 为主，同时用离线 EPE 排查模型预测问题。
