# UnrealZoo 录制人体轨迹闭环评估说明

## 1. 评估目标

该模式只复用手工采集 JSON 中人的世界坐标轨迹，其余内容全部由在线闭环仿真产生：

```text
采集 JSON 中每帧 target_pose
        ↓ 仅控制在线仿真中的人
在线 Drone/RobotDog RGB
        ↓
模型输出 bbox 与未来路点
        ↓
评估代码将路点换算为 Agent 动作
        ↓
UnrealZoo 生成新 Agent 位姿、新 RGB、真值 bbox 和指标
```

旧的录制 RGB 离线推理入口已删除，因为录制视频中的相机运动固定，无法验证模型动作的
真实闭环效果。

## 2. 使用规则

- 只读取 `*_drone_info.json` 中的 `target_pose=[x,y,z,pitch,yaw,roll]`。
- 不读取录制 RGB、Agent 位姿、bbox 或采集动作。
- 人每个仿真步推进到下一条录制位姿。
- Drone 和 RobotDog 只执行由模型预测路点派生的动作。
- 在线 RGB、bbox、可见性、距离和碰撞均来自当前 UnrealZoo 仿真。
- `ENV_ID` 必须与录制轨迹所属场景一致。
- 最大闭环步数为 `min(MAX_STEPS, 录制位姿数 - 1)`。

## 3. 运行命令

```bash
GPU=1 \
ENV_ID=UnrealTrack-DowntownWest-ContinuousColor-v0 \
RECORDED_TARGET_DIR=/data/hdt/newtrackvla/sim_data/unrealzoo_aerial_ground_human_small/hand/seed_hand/UnrealTrack-DowntownWest-ContinuousColor-v0 \
RECORDED_TARGET_EPISODES=0 \
EPISODES=1 \
MAX_STEPS=299 \
SAVE_PATH=/data/hdt/newtrackvla/sim_data/eval/recorded_human_closed_loop \
bash sh/eval_unrealzoo.sh
```

测试目录中的全部人体轨迹：

```bash
GPU=1 \
RECORDED_TARGET_DIR=/data/hdt/newtrackvla/sim_data/unrealzoo_aerial_ground_human_small/hand/seed_hand/UnrealTrack-DowntownWest-ContinuousColor-v0 \
EPISODES=8 \
MAX_STEPS=299 \
bash sh/eval_unrealzoo.sh
```

不设置 `RECORDED_TARGET_DIR` 时，评估保持原行为：人在 UnrealZoo 中随机初始化并自主导航。

### 使用 2:1 划分中 test 集的人体轨迹

推荐直接传入数据集最外层的 `split_manifest.json`。评估程序只读取 manifest 的
`test` 清单，并根据 `ENV_ID` 自动选择相同场景的人体轨迹：

```bash
CKPT=/data/hdt/newtrackvla/ckpt/anchor_diffusion_hand_2to1_fixed_100ep/model_best_val.pt \
GPU=1 RENDER_GPU=1 \
ENV_ID=UnrealTrack-Arctic-ContinuousColor-v0 \
TEST_TARGET_MANIFEST=/data/hdt/newtrackvla/data/unrealzoo_aerial_ground_human_hand_multi_2to1/split_manifest.json \
EPISODES=2 MAX_STEPS=500 \
BBOX_SOURCE=model \
SAVE_PATH=/data/hdt/newtrackvla/sim_data/eval/test_human_closed_loop_arctic \
bash sh/eval_unrealzoo.sh
```

该模式不会使用 test 集中录制的无人机/机器狗 RGB、位姿、bbox 或动作。它只重放
原始 `*_drone_info.json` 中人的 `target_pose`；两个 Agent 的放置、在线 RGB、
模型 bbox、模型轨迹和闭环动作均来自当前仿真。

## 4. 输出确认

`setup.json` 和 episode 统计 JSON 会记录：

```text
target_motion_mode=recorded_pose_replay
recorded_target_source
recorded_target_episode
recorded_target_pose_count
```

每步 `combined_info.json` 会记录：

```text
target_motion_mode
recorded_target_frame
action_debug.action_source=derived_from_model_waypoints
```

这些字段可用于确认：人的运动来自录制轨迹，两个 Agent 的控制来自模型输出。
