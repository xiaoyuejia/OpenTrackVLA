# 双 Agent OpenTrackVLA 数据处理与训练说明

本目录用于处理 UnrealZoo 中“无人机 + 机器狗”双 Agent 目标跟踪数据，并训练新的多 Agent VLA 模型。

核心文件：

| 文件 | 作用 |
| --- | --- |
| `make_multi_agent_tracking_data.py` | 从原始 UnrealZoo paired 视频和 info JSON 生成双 Agent JSONL 训练数据 |
| `precache_multi_agent_frames.py` | 为 JSONL 中引用的双 Agent 图片预计算视觉 token |
| `multi_agent_model.py` | 双 Agent VLA 模型结构 |
| `train_multi_agent.py` | 双 Agent 模型训练入口 |
| `model_vs_multi_agent_model.md` | 新旧模型差异对比 |
| `ARCHITECTURE_ZH.md` | 中文模型架构、数据流和代码分析 |
| `CLASS_COMPARISON_ZH.md` | `OpenTrackVLA` 与 `MultiAgentOpenTrackVLA` 的类级代码对比 |

## 1. 原始数据格式

脚本默认处理如下 UnrealZoo 采集结果：

```text
sim_data/unrealzoo_aerial_ground_human_small/
  seed_100/
    UnrealTrack-DowntownWest-ContinuousColor-v0/
      0.json
      0_drone.mp4
      0_drone_info.json
      0_robotdog.mp4
      0_robotdog_info.json
      0_global.mp4
```

说明：

- `*_drone.mp4`：无人机视角视频。
- `*_robotdog.mp4`：机器狗视角视频。
- `*_drone_info.json`：无人机每步动作、bbox、可见性、pose 等信息。
- `*_robotdog_info.json`：机器狗每步动作、bbox、可见性、pose 等信息。
- `*_global.mp4`：全局调试视频，训练默认不使用。
- `<id>.json`：episode 级状态，比如成功、碰撞、跟踪率、总步数。

## 2. 生成双 Agent JSONL

```bash
python multi_agent/make_multi_agent_tracking_data.py \
  --input_root /data/hdt/newtrackvla/sim_data/unrealzoo_aerial_ground_human_small \
  --output_root /data/hdt/newtrackvla/data/unrealzoo_aerial_ground_human_multi \
  --history 31 \
  --horizon 8 \
  --n_waypoints 8 \
  --dt 0.1 \
  --only_success \
  --exclude_collision
```

调试扫描命令：

```bash
python multi_agent/make_multi_agent_tracking_data.py \
  --input_root /data/hdt/newtrackvla/sim_data/unrealzoo_aerial_ground_human_small \
  --output_root /tmp/multi_agent_debug \
  --dry_run
```

默认设置：

- `agent1 = drone`
- `agent2 = robotdog`
- `target_bbox` 从像素坐标 `x, y, w, h` 转为归一化 `cx, cy, w, h`
- 视频帧写入 `<output_root>/frames`
- JSONL 写入 `<output_root>/jsonl`
- 同时写聚合文件 `<output_root>/dataset.json`

单条 JSONL 样本的核心字段：

```text
agent1_images       # agent1 历史帧，不含当前帧
agent1_current      # agent1 当前帧
agent1_bbox         # agent1 当前 bbox，cxcywh_norm
agent1_waypoints    # agent1 未来路点，(N, 3)

agent2_images
agent2_current
agent2_bbox
agent2_waypoints

bbox_feat           # (2, 4)
waypoints           # (2, n_waypoints, 3)
valid_mask          # (2, n_waypoints)
agents              # 结构化调试字段，包含 drone/robotdog 详细信息
```

## 3. 预缓存视觉 Token

```bash
python multi_agent/precache_multi_agent_frames.py \
  --data_root /data/hdt/newtrackvla/data/unrealzoo_aerial_ground_human_multi \
  --batch_size 8 \
  --image_size 384
```

快速检查 JSONL 引用了多少帧，不加载视觉模型：

```bash
python multi_agent/precache_multi_agent_frames.py \
  --data_root /data/hdt/newtrackvla/data/unrealzoo_aerial_ground_human_multi \
  --list_only
```

缓存输出：

```text
<data_root>/vision_cache/frames/.../*_vcoarse.pt
<data_root>/vision_cache/frames/.../*_vfine.pt
```

路径镜像规则：

```text
frames/seed_100/.../drone/frame_00001.jpg
 ->
vision_cache/frames/seed_100/.../drone/frame_00001_vcoarse.pt
vision_cache/frames/seed_100/.../drone/frame_00001_vfine.pt
```

## 4. 训练双 Agent 模型

先检查数据和缓存 shape：

```bash
python multi_agent/train_multi_agent.py \
  --train_json /data/hdt/newtrackvla/data/unrealzoo_aerial_ground_human_multi/jsonl \
  --out_dir /tmp/ckpt_multi_agent_dry \
  --batch_size 2 \
  --num_workers 0 \
  --dry_run
```

正式训练示例：

```bash
python multi_agent/train_multi_agent.py \
  --train_json /data/hdt/newtrackvla/data/unrealzoo_aerial_ground_human_multi/jsonl \
  --out_dir /data/hdt/newtrackvla/ckpt/ckpts_multi_agent \
  --cache_root /data/hdt/newtrackvla/data/unrealzoo_aerial_ground_human_multi/vision_cache \
  --llm_name Qwen/Qwen3-0.6B \
  --n_waypoints 8 \
  --history 31 \
  --batch_size 2 \
  --grad_accum_steps 8 \
  --epochs 4 \
  --lr 2e-5 \
  --beta_nav 10 \
  --freeze_llm
```

训练 batch shape：

```text
coarse_tokens: (B, 2, 124, C)
fine_tokens:   (B, 2, 64, C)
bbox_feat:     (B, 2, 4)
waypoints:     (B, 2, 8, 3)
valid_mask:    (B, 2, 8)
```

## 5. 常用参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--history` | `31` | 每条样本使用多少历史帧 |
| `--horizon` | `8` | 从当前步向未来积分多少动作步 |
| `--n_waypoints` | `8` | 固定输出路点数量 |
| `--dt` | `0.1` | 动作积分时间步长 |
| `--action_field` | `base_velocity` | 生成标签时优先使用的动作字段 |
| `--agent1` | `drone` | 第一个 Agent |
| `--agent2` | `robotdog` | 第二个 Agent |
| `--beta_nav` | `10` | waypoint loss 权重 |
| `--beta_bbox` | `0` | bbox refinement 辅助 loss 权重 |
| `--beta_visible` | `0` | visibility 辅助 loss 权重 |

## 6. 当前实现边界

- 训练主目标是双 Agent 路点规划。
- Grounding head 已实现，但 bbox/visibility loss 默认关闭，可后续打开。
- `token_logits` 输出默认训练不使用，可作为后续 token-level grounding 监督扩展点。
- 如果 vision cache 缺失，训练脚本默认报错；可以传 `--online_encode_missing` 在线补缓存，但速度会慢很多。
