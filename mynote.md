小批量测试：
===============================================================================
## 1.数据预处理、视觉缓存和训练前检查：
INPUT_ROOT=/data/hdt/newtrackvla/sim_data/unrealzoo_aerial_ground_human_small/hand/organized \
DATA_ROOT=/data/hdt/newtrackvla/data/unrealzoo_aerial_ground_human_hand_multi \
RUN_MAKE_DATA=1 \
RUN_PRECACHE=1 \
RUN_DRY_RUN=1 \
RUN_TRAIN=0 \
CUDA_VISIBLE_DEVICES=1 \
bash sh/run_multi_agent_pipeline.sh
部分结果：

[paths]
INPUT_ROOT=/data/hdt/newtrackvla/sim_data/unrealzoo_aerial_ground_human_small/hand
DATA_ROOT=/data/hdt/newtrackvla/data/unrealzoo_aerial_ground_human_hand_multi
TRAIN_JSON=/data/hdt/newtrackvla/data/unrealzoo_aerial_ground_human_hand_multi/jsonl
CACHE_ROOT=/data/hdt/newtrackvla/data/unrealzoo_aerial_ground_human_hand_multi/vision_cache
OUT_DIR=/data/hdt/newtrackvla/ckpt/ckpts_multi_agent
CUDA_VISIBLE_DEVICES=1

[data]
HISTORY=31
HORIZON=8
N_WAYPOINTS=8
DT=0.1
AGENT1=drone
AGENT2=robotdog
MAX_EPISODES=0

[train]
LLM_NAME=Qwen/Qwen3-0.6B
EPOCHS=4
BATCH_SIZE=2
GRAD_ACCUM_STEPS=8
LR=2e-5
BETA_NAV=10
BETA_BBOX=1.0
BETA_VISIBLE=0.5
BBOX_DROPOUT_PROB=0.5
FREEZE_LLM=1
NUM_GPUS=1

>>> /home/hdt/miniconda3/envs/omtracknew/bin/python -m tools.make_tracking_data --multi_agent --input_root /data/hdt/newtrackvla/sim_data/unrealzoo_aerial_ground_human_small/hand --output_root /data/hdt/newtrackvla/data/unrealzoo_aerial_ground_human_hand_multi --history 31 --horizon 8 --n_waypoints 8 --dt 0.1 --agent1 drone --agent2 robotdog --action_field base_velocity --min_target_visibility 0.0 --min_agent_following_rate 0.0 --min_total_steps 0 --max_episodes 0
Found paired episodes: 32

Kept episodes: 32
Written JSONL files: 32
Written samples: 8925
Skipped by status: 0
Skipped by load/extract: 0
Skipped empty: 0
Output root: /data/hdt/newtrackvla/data/unrealzoo_aerial_ground_human_hand_multi
Aggregated dataset JSON: /data/hdt/newtrackvla/data/unrealzoo_aerial_ground_human_hand_multi/dataset.json

>>> /home/hdt/miniconda3/envs/omtracknew/bin/python -m tools.precache_frames --multi_agent --data_root /data/hdt/newtrackvla/data/unrealzoo_aerial_ground_human_hand_multi --cache_root /data/hdt/newtrackvla/data/unrealzoo_aerial_ground_human_hand_multi/vision_cache --batch_size 8 --image_size 384 --limit 0
Frames to check: 17850
Cache root: /data/hdt/newtrackvla/data/unrealzoo_aerial_ground_human_hand_multi/vision_cache
===============================================================================

## 2.生成锚点并检查训练：
RUN_BUILD_ANCHORS=1 \
RUN_DRY_RUN=1 \
RUN_TRAIN=0 \
CUDA_VISIBLE_DEVICES=1 \
bash sh/train_anchor_diffusion.sh
结果：
[SANITY] samples_checked=256 xy_std=(0.2888,0.0329) theta_std=0.0956 bbox_mean=0.3138 current_img_ok=512/512 current_cache_ok=512/512
[DRY_RUN][MULTI] first item shapes:
  coarse_tokens: (2, 124, 1536)
  coarse_tidx: (2, 124)
  fine_tokens: (2, 64, 1536)
  fine_tidx: (2, 64)
  bbox_feat: (2, 4)
  waypoints: (2, 8, 3)
  valid_mask: (2, 8)
  instruction: Follow the target person without collision.

## 3.正式训练：
RUN_BUILD_ANCHORS=0 \
RUN_DRY_RUN=0 \
RUN_TRAIN=1 \
EPOCHS=10 \
CUDA_VISIBLE_DEVICES=1 \
bash sh/train_anchor_diffusion.sh


有效 batch = BATCH_SIZE × GRAD_ACCUM_STEPS × GPU 数量
原来：GRAD_ACCUM_STEPS=8 表示进行 8 次前向传播和反向传播后，才更新一次模型参数，称为梯度累积。
在显存不足时模拟更大的 batch
降低梯度波动，使训练更加稳定
减少模型参数更新次数
缺点：GPU 每次只处理较小 batch，利用率可能较低

NUM_WORKERS       = 同时工作的 CPU 数据加载进程数
PREFETCH_FACTOR   = 每个进程提前准备的 batch 数量

CPU 内存 free -h
实时监控 watch -n 1 free -h
共享内存 df -h /dev/shm

BCE 是 Binary Cross Entropy，二元交叉熵损失，用于训练模型判断一个候选是“正样本”还是“负样本”。
在当前 Anchor Diffusion 模型中，BCE 用于训练模型给每条候选轨迹评分。

## 评估：
```bash
读取 Drone 和 RobotDog 当前 RGB
        ↓
在线生成 DINO + SigLIP 视觉 Token
        ↓
结合最近 31 帧历史 Token
        ↓
模型预测两个 Agent 的 bbox、可见性和未来轨迹
        ↓
将预测轨迹转换成控制动作
        ↓
执行 UnrealZoo 环境动作
        ↓
读取执行后的距离、可见性、碰撞等状态
        ↓
进入下一步
```
CKPT=/data/hdt/newtrackvla/ckpt/anchor_diffusion_hand_2to1_fixed_100ep/model_epoch100_step057300_final.pt \
GPU=1 RENDER_GPU=1 \
ENV_ID=UnrealTrack-Arctic-ContinuousColor-v0 \
EPISODES=10 MAX_STEPS=500 \
MAX_LOST_STEPS=20 \
MAX_FAILURE_STEPS=40 \
FAILURE_WARMUP_STEPS=20 \
MAX_EPISODE_SECONDS=300 \
BBOX_SOURCE=model \
DIFFUSION_DETERMINISTIC_INFERENCE=1 \
SAVE_PATH=/data/hdt/newtrackvla/sim_data/eval/anchor_diffusion_closed_loop \
bash sh/eval_unrealzoo.sh

CKPT=/data/hdt/newtrackvla/ckpt/anchor_diffusion_hand_2to1_fixed_100ep/model_epoch100_step057300_final.pt \
GPU=1 RENDER_GPU=1 \
ENV_ID=UnrealTrack-Arctic-ContinuousColor-v0 \
TEST_TARGET_MANIFEST=/data/hdt/newtrackvla/data/unrealzoo_aerial_ground_human_hand_multi_2to1/split_manifest.json \
EPISODES=2 MAX_STEPS=299 \
MAX_LOST_STEPS=20 \
MAX_FAILURE_STEPS=40 \
FAILURE_WARMUP_STEPS=20 \
MAX_EPISODE_SECONDS=300 \
BBOX_SOURCE=model \
DIFFUSION_DETERMINISTIC_INFERENCE=1 \
SAVE_PATH=/data/hdt/newtrackvla/sim_data/eval/test_human_closed_loop_arctic1 \
bash sh/eval_unrealzoo.sh

含义：
final_epe = 0.1398
两个 Agent 第 8 个预测路点与真值最终路点之间的平均 XY 距离误差约为 0.14 个轨迹坐标单位。按照当前数据生成方式，通常可理解为约 0.14 米。

hit = 0.7881
78.81% 的最终预测路点误差不超过 0.2。即约 1614 / 2048 条轨迹命中阈值。

loss = 17.11
联合损失，包含扩散轨迹回归、候选轨迹评分 BCE、bbox 和可见性损失。它不是直接可解释的距离指标。

最优模型出现在 epoch 8 / step 5000

当前实际验证范围
你训练时设置了：
EVAL_BATCHES=32
BATCH_SIZE=32
所以每次验证只评估：
32 × 32 = 1024 个样本
1024 × 2 Agent = 2048 条预测轨迹
并没有评估完整的 17,437 个测试样本。
而且 DataLoader 使用：
shuffle=False
每次都固定读取 dataset.json 开头的 1024 个样本。因此当前验证结果可能偏向文件开头的场景和 episode。

cc

每步跟踪判定
代码位置：[eval_unrealzoo_multi_agent.py (line 977)](/data/hdt/newtrackvla/eval_unrealzoo_multi_agent.py:977)
无人机跟踪成功：
drone_following = drone_visible and drone_dist <= 6.0
机器狗跟踪成功：
dog_following = dog_visible and dog_dist <= 4.0
联合跟踪成功：
joint_following = drone_following and dog_following
其中距离是 Agent 与人的 XY 平面距离：
distance = ||agent_xy - human_xy|| / 100
代码位置：[generate_drone_human_tracking_small.py (line 304)](/data/hdt/newtrackvla/unrealzoo-gym/example/DataRecording/generate_drone_human_tracking_small.py:304)
可见性来自 UnrealZoo 的目标 object mask。只要 mask 中存在目标像素，就认为可见：
visible = visible_pixels > 0
代码位置：[generate_drone_human_tracking_small.py (line 611)](/data/hdt/newtrackvla/unrealzoo-gym/example/DataRecording/generate_drone_human_tracking_small.py:611)

TR 指标
每个 episode 内：
Drone TR = 无人机 following=True 的步数 / episode 总步数
Dog TR   = 机器狗 following=True 的步数 / episode 总步数
Joint TR = 两个 Agent 同时 following=True 的步数 / episode 总步数
代码位置：[eval_unrealzoo_multi_agent.py (line 1165)](/data/hdt/newtrackvla/eval_unrealzoo_multi_agent.py:1165)

SR：成功率
单个 episode 被判定为成功，需要同时满足：
没有碰撞
没有 Lost、PersistentFailure、Timeout
运行步数 >= 20
Joint TR >= 0.5
代码位置：[eval_unrealzoo_multi_agent.py (line 1177)](/data/hdt/newtrackvla/eval_unrealzoo_multi_agent.py:1177)
两个 episode 都是：
status = Lost
因此：
SR = 成功 episode 数 / 总 episode 数
   = 0 / 2
   = 0%
Lost 判定
当无人机或机器狗出现以下任一情况时，lost_count 连续累加：
无人机距离人超过 10m
机器狗距离人超过 8m
任意 Agent 看不见人
连续达到：
MAX_LOST_STEPS=20
episode 提前结束并标记为 Lost。
代码位置：[eval_unrealzoo_multi_agent.py (line 989)](/data/hdt/newtrackvla/eval_unrealzoo_multi_agent.py:989)

CR：碰撞率
CR = 发生碰撞的 episode 数 / 总 episode 数
   = 0 / 2
   = 0%
无人机或机器狗任意一个发生碰撞，整个 episode 都算碰撞。

BBox IoU
每一步将模型预测 bbox 与 UnrealZoo object mask 生成的真值 bbox 比较：
IoU = 预测框与真值框交集面积 / 两框并集面积
代码位置：
IoU 函数：[eval_unrealzoo_multi_agent.py (line 265)](/data/hdt/newtrackvla/eval_unrealzoo_multi_agent.py:265)
每步计算：[eval_unrealzoo_multi_agent.py (line 956)](/data/hdt/newtrackvla/eval_unrealzoo_multi_agent.py:956)
Episode 平均：[eval_unrealzoo_multi_agent.py (line 1170)](/data/hdt/newtrackvla/eval_unrealzoo_multi_agent.py:1170)
你的结果：
Drone bbox IoU = 0.00%
Dog bbox IoU   = 8.89%
表示无人机预测框与真值框几乎完全不重叠；机器狗预测框平均只有约 8.89% 重叠。
这也是闭环很快丢失的重要原因。

Visibility Acc
模型输出两个 Agent 的 visible_score。以 0.5 为阈值，与 UnrealZoo 真值可见性比较：
predicted_visible = visible_score >= 0.5
correct = predicted_visible == target_visible
代码位置：[eval_unrealzoo_multi_agent.py (line 1114)](/data/hdt/newtrackvla/eval_unrealzoo_multi_agent.py:1114)
每个 episode：
Visibility Acc = 两个 Agent 可见性预测正确总数 / (步数 × 2)
最终对 episode 等权平均：
(56.52% + 66.07%) / 2 = 61.30%

跨 Episode 汇总
最终终端中的所有指标由：
[calculate_unrealzoo_metrics.py (line 40)](/data/hdt/newtrackvla/tools/calculate_unrealzoo_metrics.py:40)
计算：
SR       = episode success 平均值
CR       = episode collision 平均值
TR       = 各 episode TR 的平均值
BBox IoU = 各 episode 平均 IoU 的平均值
Avg steps = 各 episode 步数平均值
Avg FPS   = 各 episode FPS 平均值
BBox sources: {'model': 2} 表示两个 episode 都使用模型 bbox，不向模型输入真值 bbox。Avg FPS=0.76 表示每个闭环步骤平均约耗时 1.32 秒。

CUDA_VISIBLE_DEVICES=1 \
/home/hdt/miniconda3/envs/omtracknew/bin/python -u eval_unrealzoo_single_agent.py \
  --ckpt ckpt/robotdog_single_cmd_frozen_qwen_10ep \
  --test-manifest sim_data/robotdog_split_10to1/split_manifest.json \
  --save-path sim_data/eval/robotdog_single_10to1_smoke_fixed111 \
  --env-id UnrealTrack-Arctic-ContinuousColor-v0 \
  --episodes 1 \
  --render-gpu 1 \
  --max-steps 120 \
  --target-replay-mode nav_goal \
  --waypoint-index 9 \
  --robotdog-speed-gain 1.15 \
  --settle-steps 1 \
  --flush-initial-observation \
  --save-video


  在你现在的视觉缓存里，每一帧会同时过两个冻结视觉编码器：

```text
DINOv3
SigLIP
```

然后把它们的 patch token 在最后一维拼接起来：

```text
1536 = 384 + 1152
```

也就是：

```text
DINO 特征:  (patch_num, 384)
SigLIP 特征: (patch_num, 1152)
拼接后:      (patch_num, 1536)
```

**DINO 特征是什么**

DINOv3 是自监督视觉模型。它更擅长提供：

```text
物体形状
边界
局部纹理
空间结构
前景/背景分离
姿态和轮廓
```

在你的 UnrealZoo 狗视角里，DINO 更可能帮助模型理解：

```text
人在哪里
人轮廓大概在哪块区域
地面/墙/建筑结构
画面中可通行区域
目标相对位置变化
```

它不直接输出 “这是人” 这个类别，而是输出每个图像 patch 的视觉表示。

**SigLIP 特征是什么**

SigLIP 是图文对齐模型的视觉塔。它更偏语义，擅长把图像 patch 映射到和语言概念更接近的空间。

它更可能编码：

```text
person / human
dog / robot
street / snow / castle / indoor
object category
场景语义
```

也就是说，SigLIP 更偏“这是什么”，DINO 更偏“它长什么样、在哪里”。

**在代码里怎么拼**

实际逻辑类似：

```python
tok_dino = DINO(image)       # (1, P, 384)
tok_sigl = SigLIP(image)     # (1, P, 1152)
tokens = cat([tok_dino, tok_sigl], dim=-1)
```

得到：

```text
tokens: (1, P, 1536)
```

然后再池化成：

```text
vfine:   (64, 1536)
vcoarse: (4, 1536)
```

**对你的任务的意义**

跟踪人需要两类能力：

```text
1. 看懂目标是什么：SigLIP 更有帮助
2. 看懂目标在哪、怎么动：DINO 更有帮助
```

所以 DINO + SigLIP 的组合是合理的。

但它们只是“特征提取”，不是检测器。它们不会直接给出：

```text
人 bbox
目标 ID
该跟踪哪个人
```
这些需要后面的 Qwen/projector/planner 从 token 里学出来。当前 single-agent 没有 bbox 输入，所以模型只能从这些视觉 token 中自己隐式学“人在哪”。这也是目前可能不稳定的地方。

0.mp4
0_info.json
0.json

frames/.../0/frame_00001.jpg ...
jsonl/.../0.jsonl

{
  "images": [...历史帧...],
  "current": "当前帧",
  "instruction": "Follow the target person without collision.",
  "trajectory": [...未来局部轨迹...],
  "actions": [...未来速度动作...],
  "collision": false,
  "target_distance": 6.3
}

mp4
 -> jpg frames
info.json 的 commanded_base_velocity
 -> actions
actions 按 dt=0.1 积分
 -> trajectory
status json / info json
 -> collision、target_distance、instruction、过滤条件

RGB image: (384, 384, 3)
frame_00001_vcoarse.pt: (4, 1536)
frame_00001_vfine.pt:   (64, 1536)
1536 = DINO 特征 + SigLIP 特征
     = 384 + 1152

vcoarse: 4 个粗粒度 token，给历史帧用
vfine:   64 个细粒度 token，给当前帧用

frame_00001.jpg
  -> DINO patch tokens
  -> SigLIP patch tokens
  -> concat 成 1536 维
  -> grid pool

输出:
  vcoarse: (4, 1536)
  vfine:   (64, 1536)

训练 Dataset 读取 JSONL + vision_cache
coarse_tokens: (124, 1536)
coarse_tidx:   (124,)
fine_tokens:   (64, 1536)
fine_tidx:     (64,)
yaw_hist:      (31,)
yaw_curr:      (1,)
waypoints:     (10, 3)
valid_mask:    (10,)
instruction:   str
current_path:  str
history = 31
每帧 coarse token = 4
31 * 4 = 124

训练时的 waypoints
x = zeros(T)
y = zeros(T)
theta = zeros(T)

for t in range(1, T):
    x[t] = x[t-1] + action[t-1] * dt
数据处理JSONL 的 trajectory
end_index = j + horizon
future_actions = actions[j : end_index + 1]

x, y, theta = 0, 0, 0

for action in future_actions:
    x += ...
    y += ...
    theta += ...
    trajectory.append([x, y, theta])

actions = [None for _ in env.unwrapped.player_list]
actions[robotdog_id] = dog_action
obs, rewards, done, info = common.data_collection_step(env, actions)


commanded_base_velocity = 控制器期望/命令
base_velocity           = 环境中实际执行后的运动
ground_action            = UnrealZoo 实际下发给狗的低层动作 [turn, speed_cm_s]

RUN_ORGANIZE=0 \
RUN_SPLIT=0 \
RUN_PROCESS=0 \
RUN_CACHE=0 \
RUN_TRAIN=0 \
RUN_EVAL=1 \
CKPT_DIR=/data/hdt/newtrackvla/ckpt/drone_single_frozen_qwen_10ep_gpu0 \
EVAL_ROOT=/data/hdt/newtrackvla/sim_data/eval/drone_single_10to1_gpu0 \
EVAL_GPU=1 \
RENDER_GPU=1 \
SAVE_EVAL_VIDEO=1 \
EVAL_REQUIRE_SUCCESS_DISTANCE=1 \
EVAL_DRONE_SUCCESS_DISTANCE=6.0 \
bash /data/hdt/newtrackvla/sh/run_drone_single_agent_pipeline.sh


评估测试：
评估base带marker的模型
cd /data/hdt/newtrackvla

EVAL_SCRIPT=eval_unrealzoo_multi_agent.py \
CKPT_DIR=/data/hdt/newtrackvla/ckpt/data_multi_agent_model_py_base_marker_b32_acc4_lr2e-5_gpu6 \
SPLIT_ROOT=/data/hdt/newtrackvla/sim_data/data_multi_agent_split_10to1 \
TEST_TARGET_MANIFEST=/data/hdt/newtrackvla/sim_data/data_multi_agent_split_10to1/split_manifest.json \
EVAL_ROOT=/data/hdt/newtrackvla/sim_data/eval/data_multi_agent_model_py_base_marker_gpu6_history_yaw_fix_smoke \
EVAL_SCENES=UnrealTrack-DowntownWest-ContinuousColor-v0 \
EVAL_EPISODES=2 \
EVAL_MAX_STEPS=200 \
EVAL_GPUS=3 \
RENDER_GPUS=3 \
EVAL_BBOX_SOURCE=none \
EVAL_INIT_FROM_RECORDED_AGENT_POSES=1 \
EVAL_DRONE_MAX_YAW_RATE=1.0 \
EVAL_PLANNER_DEBUG_STEPS=20 \
SAVE_EVAL_VIDEO=1 \
bash sh/run_multi_agent_eval.sh

评估base-separete模型

当前的训练数据和评估数据：
/data/hdt/ntv_data/data/New_paths_training_multi_agent_10to1/
当前的模型保存位置：
/data/hdt/ntv_data/ckpt/