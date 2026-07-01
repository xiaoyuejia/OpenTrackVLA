# UnrealZoo 双 Agent 闭环评估说明

## 1. 本次新增内容

为了在新的 UnrealZoo 环境中测试双 Agent 模型，同时保留原来的 Habitat / EVT-Bench 测试流程，本次新增了三份文件：

| 文件 | 作用 |
|---|---|
| `eval_unrealzoo_multi_agent.py` | 在 UnrealZoo 中运行无人机 + 机器狗 + 行人的双 Agent 闭环评估 |
| `tools/calculate_unrealzoo_metrics.py` | 汇总 UnrealZoo 评估结果，计算 SR、TR、CR 等指标 |
| `sh/eval_unrealzoo.sh` | 一键启动 UnrealZoo 评估的 shell 脚本 |

原来的 Habitat 评估入口保持不变：

```bash
bash sh/eval.sh
```

新的 UnrealZoo 评估入口是：

```bash
bash sh/eval_unrealzoo.sh
```

两套评估互不覆盖：

```text
Habitat / EVT-Bench:
sh/eval.sh -> eval.py -> tools/trained_agent.py -> tools/calculate_metrics.py

UnrealZoo:
sh/eval_unrealzoo.sh -> eval_unrealzoo_multi_agent.py -> tools/calculate_unrealzoo_metrics.py
```

### 是否需要 YAML

UnrealZoo 评估**不需要**像 Habitat 那样额外传入 YAML。

原因是两套环境的配置机制不同：

| 环境 | 配置入口 | 例子 |
|---|---|---|
| Habitat | YAML 文件 | `track_infer_stt.yaml` |
| UnrealZoo | Gym 环境 ID + JSON setting | `UnrealTrack-DowntownWest-ContinuousColor-v0` |

当前 UnrealZoo 环境 ID 会映射到：

```text
unrealzoo-gym/gym_unrealcv/envs/setting/Track/DowntownWest.json
```

因此运行 UnrealZoo 评估时，只需要设置：

```bash
ENV_ID=UnrealTrack-DowntownWest-ContinuousColor-v0
```

---

## 2. UnrealZoo 评估适用对象

当前 `eval_unrealzoo_multi_agent.py` 面向主目录中的双 Agent 模型：

```python
MultiAgentOpenTrackVLA
```

也就是训练时使用：

```bash
python train.py --multi_agent ...
```

得到的 checkpoint。

它不用于原来的单 Agent Habitat checkpoint。单 Agent checkpoint 仍然使用：

```bash
bash sh/eval.sh
```

---

## 3. UnrealZoo 闭环数据流

UnrealZoo 评估是闭环测试，每一步都会调用仿真环境。

```text
UnrealZoo 当前状态
    │
    ├── drone RGB 当前帧:    (H,W,3)
    └── robotdog RGB 当前帧: (H,W,3)
            │
            ▼
在线视觉编码
DINO + SigLIP
            │
            ├── 每个 Agent 当前细粒度 token: (64,1536)
            └── 每个 Agent 历史粗粒度 token: history * 4 = 31*4
            │
            ▼
构造 MultiAgentOpenTrackVLA 输入
            │
            ├── coarse_tokens: (1,2,124,1536)
            ├── coarse_tidx:   (1,2,124)
            ├── fine_tokens:   (1,2,64,1536)
            ├── fine_tidx:     (1,2,64)
            ├── bbox_feat:     (1,2,4)
            └── instruction:   List[str]
            │
            ▼
MultiAgentOpenTrackVLA
            │
            ▼
waypoints: (1,2,8,3)
            │
            ├── agent1=drone    -> [vx, vy, yaw_rate] -> UnrealZoo drone action [vx,vy,0,yaw]
            └── agent2=robotdog -> [vx, vy, yaw_rate] -> UnrealZoo dog action [turn_deg,speed_cm_s]
            │
            ▼
env.step(actions)
            │
            ▼
下一帧 RGB + 距离/可见性/碰撞指标
```

默认 Agent 顺序与训练数据一致：

```text
agent1 = drone
agent2 = robotdog
```

---

## 4. 运行命令

最简单运行：

```bash
CKPT=/data/hdt/newtrackvla/ckpt/ckpts_multi_agent/model_epoch03_step000232_final.pt \
GPU=0 \
EPISODES=10 \
MAX_STEPS=100 \
SAVE_PATH=/data/hdt/newtrackvla/sim_data/eval/unrealzoo_multi_agent \
bash sh/eval_unrealzoo.sh
```

如果 `CKPT` 指向目录，脚本会自动选择其中最新的：

```text
model_epoch*.pt
```

例如：

```bash
CKPT=/data/hdt/newtrackvla/ckpt/ckpts_multi_agent \
GPU=6 \
EPISODES=5 \
bash sh/eval_unrealzoo.sh
```

评估完成后，脚本会自动运行：

```bash
python -m tools.calculate_unrealzoo_metrics --eval-dir "${SAVE_PATH}"
```

---

## 5. 常用参数说明

### 5.1 shell 参数

这些参数在 `sh/eval_unrealzoo.sh` 顶部可以直接修改，也可以在命令行覆盖。

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `CKPT` | `ckpt/ckpts_multi_agent_anchor_diffusion` | 双 Agent checkpoint 文件或目录；可覆盖为 MLP 目录 |
| `GPU` | `0` | 使用哪张 GPU |
| `ENV_ID` | `UnrealTrack-DowntownWest-ContinuousColor-v0` | UnrealZoo 场景 |
| `EPISODES` | `10` | 评估 episode 数量 |
| `MAX_STEPS` | `100` | 每条 episode 最大步数 |
| `SAVE_PATH` | `sim_data/eval/unrealzoo_multi_agent` | 结果保存目录 |
| `DT` | `0.1` | waypoint 转速度时使用的时间间隔 |
| `WAYPOINT_INDEX` | `0` | 使用第几个预测 waypoint 转动作 |
| `SUCCESS_RATE_THRESHOLD` | `0.5` | joint tracking rate 达到多少算成功 |
| `SAVE_VIDEO` | `1` | 是否保存 drone/robotdog 视频 |
| `FACE_TARGET_BEFORE_STEP` | `0` | 是否每步用目标真值强制旋转 Agent 朝向目标 |
| `BBOX_SOURCE` | `model` | `model` 使用模型检测/跟踪框；`ground_truth` 使用仿真真值框；`none` 每帧无框先验 |
| `TRAJECTORY_OVERLAY` | `1` | 是否在 RGB 视频中叠加模型预测轨迹 |
| `TRAJECTORY_SCALE` | `120` | 轨迹可视化的像素/米比例 |

脚本会自动设置：

```bash
UNREALZOO_FAST_ENV_ID="${ENV_ID}"
```

这样 `gym_unrealcv` 只注册当前要用的 UnrealZoo 环境，避免在启动时注册全部场景导致长时间卡住。

推荐先保持：

```bash
WAYPOINT_INDEX=0
DT=0.1
FACE_TARGET_BEFORE_STEP=0
BBOX_SOURCE=model
```

因为训练标签是从当前步开始积分未来速度得到的，`waypoint[0] / DT` 最接近当前控制动作。

### 5.2 Python 关键参数

也可以直接运行：

```bash
python eval_unrealzoo_multi_agent.py \
  --ckpt /data/hdt/newtrackvla/ckpt/ckpts_multi_agent/model_epoch03_step000232_final.pt \
  --save-path /data/hdt/newtrackvla/sim_data/eval/unrealzoo_multi_agent \
  --episodes 10 \
  --max-steps 100 \
  --env-id UnrealTrack-DowntownWest-ContinuousColor-v0 \
  --dt 0.1 \
  --waypoint-index 0 \
  --bbox-source model \
  --trajectory-overlay
```

### 5.3 无真值 bbox 检测/跟踪评估

默认 `--bbox-source model` 不会把 UnrealZoo object mask 生成的真值 bbox 输入模型：

```text
首帧 RGB + 无 bbox 先验 -> 模型预测绝对 bbox
后续 RGB + 上一帧模型 bbox -> 模型修正 bbox
```

object mask 真值框只用于计算 `DroneBBoxIoU`、`RobotDogBBoxIoU` 和可见性准确率。
保存的视频会绘制模型预测框和局部预测轨迹，不绘制真值框。

轨迹叠加现在使用累计弧长参数化的三次样条，并以高分辨率绘制后缩小：

- 输出为一条连续的 TURBO 渐变彩色曲线。
- 不绘制 waypoint 圆点。
- 使用黑色细轮廓保证复杂背景中的可见性。
- `eval_unrealzoo_multi_agent.py` 会根据 checkpoint 中是否包含 `planner_agent1.anchors`
  自动选择旧版 `model.py` MLP 模型或新版 `model_unrealzoo_anchor_diffusion.py` Anchor Diffusion 模型。

扩散 checkpoint 评估示例：

```bash
CKPT=ckpt/ckpts_multi_agent_anchor_diffusion \
EPISODES=1 MAX_STEPS=100 \
  bash sh/eval_unrealzoo.sh
```

需要每次运行得到完全一致的扩散轨迹时，可直接调用：

```bash
python eval_unrealzoo_multi_agent.py \
  --ckpt ckpt/ckpts_multi_agent_anchor_diffusion \
  --diffusion-deterministic-inference \
  --trajectory-overlay
```

现有 checkpoint 如果训练时始终使用真值 bbox 先验，首帧检测能力可能较差。重新训练或微调时推荐：

```bash
BETA_BBOX=1.0 BETA_VISIBLE=0.5 BBOX_DROPOUT_PROB=0.5 \
bash sh/run_multi_agent_pipeline.sh
```

`BBOX_DROPOUT_PROB` 会随机移除完整 bbox 先验，使 grounding head 同时学习无框绝对检测和有框跟踪修正。

动作限幅参数：

| 参数 | 作用 |
|---|---|
| `--drone-max-vx` | 限制无人机前后速度 |
| `--drone-max-vy` | 限制无人机侧向速度 |
| `--drone-max-yaw-rate` | 限制无人机 yaw 速度 |
| `--robotdog-max-speed` | 限制机器狗前进/后退速度 |
| `--robotdog-max-turn-deg` | 限制机器狗单步转向角 |

如果模型动作太激进，优先调小这些限幅。

---

## 6. 输出文件结构

默认保存到：

```text
sim_data/eval/unrealzoo_multi_agent/
  seed_100/
    UnrealTrack-DowntownWest-ContinuousColor-v0/
      0.json
      0_drone_info.json
      0_robotdog_info.json
      0_combined_info.json
      0_setup.json
      0_drone.mp4
      0_robotdog.mp4
      0_global.mp4
```

### 6.1 episode 汇总 `0.json`

主要字段：

```json
{
  "status": "Success",
  "success": 1.0,
  "total_step": 100,
  "collision": 0.0,
  "joint_following_rate": 0.72,
  "drone_following_rate": 0.85,
  "robotdog_following_rate": 0.78,
  "fps": 1.23,
  "ckpt": ".../model_epoch00_step000100_final.pt",
  "env_id": "UnrealTrack-DowntownWest-ContinuousColor-v0"
}
```

### 6.2 单 Agent step 信息

`0_drone_info.json` 和 `0_robotdog_info.json` 每步保存：

```json
{
  "step": 1,
  "dis_to_human": 3.2,
  "target_visible": true,
  "target_bbox": [x, y, w, h],
  "bbox_feat": [cx, cy, w, h],
  "base_velocity": [vx, vy, yaw_rate],
  "following": true,
  "collision": false
}
```

### 6.3 双 Agent 联合信息

`0_combined_info.json` 每步保存：

```json
{
  "step": 1,
  "joint_following": true,
  "drone_following": true,
  "robotdog_following": true,
  "collision": false,
  "visible_score": [0.8, 0.7],
  "refined_bbox": [[...], [...]]
}
```

---

## 7. 指标含义

运行：

```bash
python -m tools.calculate_unrealzoo_metrics \
  --eval-dir /data/hdt/newtrackvla/sim_data/eval/unrealzoo_multi_agent
```

输出指标：

| 指标 | 含义 |
|---|---|
| `SR` | 成功 episode 占比 |
| `JointTR` | 两个 Agent 同时处于跟踪状态的平均比例 |
| `DroneTR` | 无人机单独跟踪率 |
| `RobotDogTR` | 机器狗单独跟踪率 |
| `CR` | 碰撞 episode 占比 |
| `Avg FPS` | 闭环评估平均速度 |

新评估结果会直接保存 `joint_following_rate`。如果使用脚本汇总旧版采集结果，而旧 JSON 没有该字段，脚本会用 `min(drone_following_rate, robotdog_following_rate)` 作为兼容近似值。

当前成功规则：

```text
无碰撞
且没有 Lost
且 total_step >= MIN_SUCCESS_STEPS
且 joint_following_rate >= SUCCESS_RATE_THRESHOLD
```

默认：

```text
MIN_SUCCESS_STEPS = 20
SUCCESS_RATE_THRESHOLD = 0.5
```

---

## 8. 与 Habitat 评估的区别

| 对比项 | Habitat / EVT-Bench | UnrealZoo |
|---|---|---|
| 原入口 | `sh/eval.sh` | `sh/eval_unrealzoo.sh` |
| 模型 | 原单 Agent `OpenTrackVLA` | 新双 Agent `MultiAgentOpenTrackVLA` |
| 环境 | Habitat TrackEnv | UnrealZoo Gym |
| Agent | `agent_1` 机器人 | drone + robotdog |
| 视觉特征 | 在线 DINO + SigLIP | 在线 DINO + SigLIP |
| 输出动作 | `[vx,vy,wz]` | drone `[vx,vy,0,yaw]`，dog `[turn_deg,speed_cm_s]` |
| 指标 | SR / TR / CR | SR / JointTR / DroneTR / RobotDogTR / CR |

Habitat 代码没有被删除或替换，因此仍然可以按原方式评估单 Agent 模型。

---

## 9. 重要注意事项

1. UnrealZoo 评估速度会比离线评估慢很多，因为每一步都要运行仿真环境、在线视觉编码和模型推理。
2. 终端中的 Gym 警告：

```text
Gym has been unmaintained since 2022 ...
```

这是 Gym 旧版本提示，不是程序报错。正常情况下后面应继续打印：

```text
[startup] UnrealZoo imports finished
[init] UnrealZoo env=...
```

3. 如果一直停在 Gym 警告后，通常是 `gym_unrealcv` 正在全量注册大量场景。当前脚本已经通过 `UNREALZOO_FAST_ENV_ID` 开启快速注册；请确认你使用的是更新后的 `sh/eval_unrealzoo.sh`，其中应包含：

```bash
UNREALZOO_FAST_ENV_ID="${ENV_ID}"
python -u eval_unrealzoo_multi_agent.py
```

4. `eval_unrealzoo_multi_agent.py` 默认不使用目标真值强制旋转 Agent。若要排查“只差朝向”的问题，可以设置：

```bash
FACE_TARGET_BEFORE_STEP=1 bash sh/eval_unrealzoo.sh
```

5. 机器狗的 UnrealZoo 动作接口只有 `[turn_deg, speed_cm_s]`，无法直接执行模型预测的侧向速度 `vy`。当前代码会记录 `robotdog_lateral_ignored`，但执行时只使用前进速度和 yaw。
6. 默认 `BBOX_SOURCE=model` 时，object mask bbox 只用于指标计算，不输入模型。设置 `BBOX_SOURCE=ground_truth` 才会恢复真值 bbox 先验评估。
7. 如果模型动作过大，优先调小：

```bash
--drone-max-vx
--drone-max-vy
--drone-max-yaw-rate
--robotdog-max-speed
--robotdog-max-turn-deg
```

8. 如果你只想先检查 checkpoint 能不能加载，不建议直接跑很多 episode。可以先跑：

```bash
CKPT=/path/to/ckpt.pt EPISODES=1 MAX_STEPS=5 SAVE_VIDEO=0 bash sh/eval_unrealzoo.sh
```
