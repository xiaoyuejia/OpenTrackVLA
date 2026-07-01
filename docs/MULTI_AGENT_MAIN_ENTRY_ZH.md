# 主目录脚本的双 Agent 使用方式

现在主目录下这三个脚本已经支持双 Agent 模式：

```text
tools/make_tracking_data.py --multi_agent
tools/precache_frames.py --multi_agent
train.py --multi_agent
```

双 Agent 处理逻辑已经直接写入主目录脚本内部，不再依赖 `multi_agent/` 目录导入。因此你可以在源代码基础上直接查看“单 Agent 旧逻辑”和“无人机 + 机器狗新逻辑”的差异。

## 1. 原始数据示例

输入目录示例：

```text
/data/hdt/newtrackvla/sim_data/unrealzoo_aerial_ground_human_small/
  seed_100/
    UnrealTrack-DowntownWest-ContinuousColor-v0/
      0.json
      0_drone.mp4
      0_drone_info.json
      0_robotdog.mp4
      0_robotdog_info.json
      0_global.mp4
```

训练使用：

- `0_drone.mp4`
- `0_drone_info.json`
- `0_robotdog.mp4`
- `0_robotdog_info.json`
- `0.json`

默认忽略：

- `0_global.mp4`

## 2. 生成双 Agent JSONL

```bash
python -m tools.make_tracking_data \
  --multi_agent \
  --input_root /data/hdt/newtrackvla/sim_data/unrealzoo_aerial_ground_human_small \
  --output_root /data/hdt/newtrackvla/data/unrealzoo_aerial_ground_human_multi \
  --history 31 \
  --horizon 8 \
  --n_waypoints 8 \
  --dt 0.1 \
  --only_success \
  --exclude_collision
```

输出：

```text
data/unrealzoo_aerial_ground_human_multi/
  frames/
  jsonl/
  dataset.json
```

## 3. 预缓存视觉 Token

```bash
python -m tools.precache_frames \
  --multi_agent \
  --data_root /data/hdt/newtrackvla/data/unrealzoo_aerial_ground_human_multi \
  --batch_size 8 \
  --image_size 384
```

只检查 JSONL 引用了多少帧，不加载视觉模型：

```bash
python -m tools.precache_frames \
  --multi_agent \
  --data_root /data/hdt/newtrackvla/data/unrealzoo_aerial_ground_human_multi \
  --list_only
```

输出：

```text
data/unrealzoo_aerial_ground_human_multi/vision_cache/
```

## 4. 训练双 Agent 模型

先检查数据和 cache：

```bash
python train.py \
  --multi_agent \
  --train_json /data/hdt/newtrackvla/data/unrealzoo_aerial_ground_human_multi/jsonl \
  --out_dir /tmp/ckpt_multi_agent_dry \
  --batch_size 2 \
  --num_workers 0 \
  --dry_run
```

正式训练：

```bash
python train.py \
  --multi_agent \
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

## 5. 双 Agent OpenTrackVLA 模型架构

### 5.1 概览

双 Agent OpenTrackVLA 是在原始单 Agent VLA 上扩展出的无人机 + 机器狗联合模型。它把两个 Agent 的视觉历史、当前视角、目标框和自然语言指令放进同一个 LLM 序列中，再分别从两个动作查询 token 中读出两个 Agent 的未来路径点。

**核心组件：**

- 视觉编码器：DINOv3 + SigLIP（双塔结构）
- 数据读取：`MultiAgentJsonDataset`
- 跨模态投影器：`CrossModalityProjector`
- 双 Agent 时间-视角-身份嵌入器：`MultiAgentTVIEmbedder`
- 语言模型骨干：`Qwen/Qwen3-0.6B`
- 双规划头：`planner_agent1`、`planner_agent2`
- 辅助 grounding 头：`GroundingHead`

默认符号：

```text
B = batch size
A = 2                         # agent1=drone, agent2=robotdog
H = history = 31
Pc = 4                         # 每张历史帧 coarse tokens
Pf = 64                        # 当前帧 fine tokens
C = 1536                       # DINO(384) + SigLIP(1152)
D = 1024                       # Qwen/Qwen3-0.6B hidden size
N = n_waypoints = 8
K = action_dims = 3            # [x, y, theta]
Ltxt <= 128
```

如果更换 `--llm_name`，`D` 以对应 LLM 的 `config.hidden_size` 为准。

---

### 5.2 架构图

```text
输入与缓存
──────────────────────────────────────────────────────────────────────────────
│                                                                            │
│  UnrealZoo 双 Agent 原始数据                                                │
│                                                                            │
│  ┌────────────────────────────┐     ┌────────────────────────────┐         │
│  │      Drone 数据             │     │     Robotdog 数据           │         │
│  │  0_drone.mp4                │     │  0_robotdog.mp4             │         │
│  │  0_drone_info.json          │     │  0_robotdog_info.json       │         │
│  └──────────────┬─────────────┘     └──────────────┬─────────────┘         │
│                 │                                  │                       │
│                 └──────────┬───────────────────────┘                       │
│                            ▼                                               │
│       tools/make_tracking_data.py --multi_agent                                  │
│       抽帧 + bbox归一化 + 动作积分 + waypoint重采样                         │
│                            │                                               │
│                            ▼                                               │
│       JSONL 样本                                                            │
│       agent1_images/current: list[str]/str                                  │
│       agent2_images/current: list[str]/str                                  │
│       bbox_feat:  (2, 4)        # cx,cy,w,h in [0,1]                        │
│       waypoints:  (2, 8, 3)     # [x,y,theta]                               │
│       valid_mask: (2, 8)                                                    │
│                            │                                               │
│                            ▼                                               │
│       tools/precache_frames.py --multi_agent                                      │
│       DINOv3(384) + SigLIP(1152) → concat(1536) → GridPool                  │
│                            │                                               │
│                            ▼                                               │
│       vision_cache/                                                         │
│       *_vcoarse.pt: (4, 1536)   # 每张历史帧                                │
│       *_vfine.pt:   (64,1536)   # 当前帧                                    │
│                                                                            │
──────────────────────────────────────────────────────────────────────────────
Dataset 与 Batch
──────────────────────────────────────────────────────────────────────────────
│                                                                            │
│       MultiAgentJsonDataset                                                 │
│       读取 JSONL + vision_cache                                             │
│                                                                            │
│       单样本输出：                                                          │
│       coarse_tokens: (2, 124, 1536)    # 124 = H × Pc = 31 × 4              │
│       coarse_tidx:   (2, 124)                                               │
│       fine_tokens:   (2, 64, 1536)                                          │
│       fine_tidx:     (2, 64)                                                │
│       bbox_feat:     (2, 4)                                                 │
│       waypoints:     (2, 8, 3)                                              │
│       valid_mask:    (2, 8)                                                 │
│                            │                                               │
│                            ▼                                               │
│       collate_multi_agent_batch                                             │
│                                                                            │
│       Batch 输出：                                                          │
│       coarse_tokens: (B, 2, 124, 1536)                                      │
│       coarse_tidx:   (B, 2, 124)                                            │
│       fine_tokens:   (B, 2, 64, 1536)                                       │
│       fine_tidx:     (B, 2, 64)                                             │
│       bbox_feat:     (B, 2, 4)                                              │
│       waypoints:     (B, 2, 8, 3)                                           │
│                                                                            │
──────────────────────────────────────────────────────────────────────────────
模型处理
──────────────────────────────────────────────────────────────────────────────
│                                                                            │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │                     MultiAgentOpenTrackVLA                         │     │
│  │                                                                    │     │
│  │  1. 按 Agent 拆分视觉输入                                           │     │
│  │                                                                    │     │
│  │     agent1/drone:                                                   │     │
│  │       coarse: (B,124,1536), fine: (B,64,1536), bbox: (B,4)          │     │
│  │     agent2/robotdog:                                                │     │
│  │       coarse: (B,124,1536), fine: (B,64,1536), bbox: (B,4)          │     │
│  │                                                                    │     │
│  │  2. CrossModalityProjector                                          │     │
│  │       LayerNorm → Linear(1536→1024) → GELU → Linear(1024→1024)      │     │
│  │       vis_c: (B,124,1024), vis_f: (B,64,1024)                      │     │
│  │                                                                    │     │
│  │  3. MultiAgentTVIEmbedder                                           │     │
│  │       time_emb + view_emb + kind_emb + agent_emb                    │     │
│  │       bbox_proj: (B,4) → (B,1,1024)                                 │     │
│  │                                                                    │     │
│  │       单 Agent 序列（默认不启用 angle TVI）:                         │     │
│  │       历史: 31 × ([TIME] + 4 coarse tokens) = 155 tokens            │     │
│  │       当前: [TIME] + 64 fine tokens = 65 tokens                     │     │
│  │       BBOX: 1 token                                                 │     │
│  │       agent_seq: (B,221,1024)                                       │     │
│  │                                                                    │     │
│  │  4. 文本与查询 token                                                │     │
│  │       txt_emb: (B,Ltxt,1024), Ltxt <= 128                           │     │
│  │       ACT1:    (B,1,1024)                                           │     │
│  │       ACT2:    (B,1,1024)                                           │     │
│  │       GND:     (B,1,1024)                                           │     │
│  │                                                                    │     │
│  │  5. LLM 输入拼接                                                    │     │
│  │       [Text] + [Agent1] + [ACT1] + [Agent2] + [ACT2] + [GND]        │     │
│  │       inputs_embeds: (B, Ltxt + 445, 1024)                          │     │
│  │       attention_mask: (B, Ltxt + 445)                               │     │
│  │                                                                    │     │
│  │  6. Qwen/Qwen3-0.6B AutoModel                                       │     │
│  │       last_hidden_state: (B, Ltxt + 445, 1024)                      │     │
│  │       h_act1: (B,1024), h_act2: (B,1024), h_gnd: (B,1024)           │     │
│  │                                                                    │     │
│  │  7. 输出头                                                          │     │
│  │       planner_agent1: h_act1 → (B,8,3)                              │     │
│  │       planner_agent2: h_act2 → (B,8,3)                              │     │
│  │       GroundingHead:  h_gnd  → bbox/visible                         │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                            │                                               │
│                            ▼                                               │
│       模型输出：                                                            │
│       waypoints:      (B, 2, 8, 3)                                          │
│       refined_bbox:   (B, 2, 4)                                             │
│       visible_logits: (B, 2)                                                │
│                                                                            │
──────────────────────────────────────────────────────────────────────────────
```

---

### 5.3 核心模块详解

#### 1. `MultiAgentJsonDataset`

**作用**：把 JSONL 样本和视觉缓存整理成模型输入。

| 输入字段 | 来源 | 输出张量 | 默认大小 |
|---|---|---|---|
| `agent1_images` | JSONL | `coarse_tokens[0]` | `(124,1536)` |
| `agent2_images` | JSONL | `coarse_tokens[1]` | `(124,1536)` |
| `agent1_current` | JSONL | `fine_tokens[0]` | `(64,1536)` |
| `agent2_current` | JSONL | `fine_tokens[1]` | `(64,1536)` |
| `bbox_feat` | JSONL | `bbox_feat` | `(2,4)` |
| `waypoints` | JSONL | `waypoints` | `(2,8,3)` |

历史帧不足 `history=31` 时，Dataset 会用当前帧的 coarse token 做左侧 padding，保证每条样本长度固定。

#### 2. `CrossModalityProjector`

```python
LayerNorm(1536)
Linear(1536 -> D)
GELU
Linear(D -> D)
```

**作用**：把 DINO + SigLIP 拼接后的视觉 token 映射到 LLM hidden space。当前 `Qwen/Qwen3-0.6B` 下 `D=1024`。

#### 3. `MultiAgentTVIEmbedder`

```python
time_emb   # 帧序号
view_emb   # 视角来源，默认 agent_id 同时作为 view_id
kind_emb   # HISTORY / CURRENT / BBOX / ACT / GND
agent_emb  # drone 与 robotdog 身份区分
bbox_proj  # bbox: (B,4) -> (B,1,D)
```

**作用**：让 LLM 在同一个长序列中区分：

- 这是哪个 Agent 的 token；
- 这是历史帧还是当前帧；
- 这是第几帧；
- 这是视觉 token、bbox token、动作查询 token 还是 grounding 查询 token。

#### 4. `Qwen/Qwen3-0.6B`

**输入**：`inputs_embeds`，不走语言生成，只把 LLM 当作多模态融合骨干。

```text
inputs_embeds:  (B, Ltxt + 445, 1024)
attention_mask: (B, Ltxt + 445)
```

**输出**：

```text
h_act1 = hidden[:, act1_pos, :]  # (B,1024)
h_act2 = hidden[:, act2_pos, :]  # (B,1024)
h_gnd  = hidden[:, gnd_pos,  :]  # (B,1024)
```

#### 5. 双规划头 `planner_agent1 / planner_agent2`

两个 Agent 使用两个独立规划头：

```python
LayerNorm(D)
Linear(D -> 2D) + GELU
Linear(2D -> 2D) + GELU
Linear(2D -> N * K)
reshape -> (B, N, K)
```

默认：

```text
planner_agent1: (B,1024) -> (B,8,3)
planner_agent2: (B,1024) -> (B,8,3)
stack:          (B,2,8,3)
```

独立规划头可以避免无人机和机器狗的运动分布被强行共享。

#### 6. `GroundingHead`

**作用**：用 `GND` 查询位置的隐藏状态预测辅助目标。

```text
h_gnd:          (B,1024)
refined_bbox:   (B,2,4)
visible_logits: (B,2)
```

当前训练默认 `beta_bbox=0`、`beta_visible=0`，也就是先主要训练路径规划；需要辅助目标框/可见性监督时再打开。

---

### 5.4 数据流维度

| 阶段 | 张量 | 类型 | 形状 |
|---|---|---:|---:|
| JSONL | `agent1_images`, `agent2_images` | `list[str]` | 最多 31 条 |
| JSONL | `agent1_current`, `agent2_current` | `str` | 1 条 |
| JSONL | `bbox_feat` | `float` | `(2,4)` |
| JSONL | `waypoints` | `float` | `(2,8,3)` |
| JSONL | `valid_mask` | `bool` | `(2,8)` |
| vision cache | `*_vcoarse.pt` | `float16/float32` | `(4,1536)` |
| vision cache | `*_vfine.pt` | `float16/float32` | `(64,1536)` |
| Dataset 单样本 | `coarse_tokens` | `float32` | `(2,124,1536)` |
| Dataset 单样本 | `fine_tokens` | `float32` | `(2,64,1536)` |
| Batch | `coarse_tokens` | `float32` | `(B,2,124,1536)` |
| Batch | `fine_tokens` | `float32` | `(B,2,64,1536)` |
| Batch | `bbox_feat` | `float32` | `(B,2,4)` |
| Batch | `waypoints` | `float32` | `(B,2,8,3)` |
| 单 Agent 投影后 | `vis_c` | `float/bfloat16` | `(B,124,1024)` |
| 单 Agent 投影后 | `vis_f` | `float/bfloat16` | `(B,64,1024)` |
| 单 Agent 序列 | `agent_seq` | `float/bfloat16` | `(B,221,1024)` |
| LLM 输入 | `inputs_embeds` | `bfloat16/float32` | `(B,Ltxt+445,1024)` |
| LLM 输出 | `h_act1`, `h_act2`, `h_gnd` | `float32` | `(B,1024)` |
| Planner 输出 | `agent*_waypoints` | `float32` | `(B,8,3)` |
| 模型输出 | `waypoints` | `float32` | `(B,2,8,3)` |
| Grounding 输出 | `refined_bbox` | `float32` | `(B,2,4)` |
| Grounding 输出 | `visible_logits` | `float32` | `(B,2)` |

---

### 5.5 LLM 序列长度

默认 `--use_angle_tvi` 关闭时：

```text
历史序列:
H * (1 个 time marker + 4 个 coarse tokens)
= 31 * 5
= 155

当前序列:
1 个 time marker + 64 个 fine tokens
= 65

bbox token:
1

单 Agent 序列长度:
155 + 65 + 1 = 221

完整 LLM 输入长度:
Ltxt + agent1_seq + ACT1 + agent2_seq + ACT2 + GND
= Ltxt + 221 + 1 + 221 + 1 + 1
= Ltxt + 445
<= 128 + 445 = 573
```

打开 `--use_angle_tvi` 时，每帧额外插入 1 个 angle marker：

```text
单 Agent 序列长度:
31 * (1 time + 1 angle + 4 coarse) + (1 time + 1 angle + 64 fine) + 1 bbox
= 31 * 6 + 66 + 1
= 253

完整 LLM 输入长度:
Ltxt + 253 + 1 + 253 + 1 + 1
= Ltxt + 509
<= 637
```

---

### 5.6 损失函数

主损失是双 Agent waypoint masked MSE：

```python
loss_nav = masked_mse_multi_agent(pred_n, gt_n, valid_mask)
loss = beta_nav * loss_nav
```

其中：

```text
pred_n:     (B,2,8,3)
gt_n:       (B,2,8,3)
valid_mask: (B,2,8)
```

可选辅助损失：

```python
loss_bbox = F.mse_loss(refined_bbox, bbox_feat)
loss_visible = F.binary_cross_entropy_with_logits(visible_logits, visible)
loss = beta_nav * loss_nav + beta_bbox * loss_bbox + beta_visible * loss_visible
```

`binary_cross_entropy_with_logits` 是 AMP 安全的写法，可以避免 `binary_cross_entropy` 在 autocast 下的报错。

---

### 5.7 重要说明

**保留单 Agent 能力**

不加 `--multi_agent` 时，`model.py`、`tools/make_tracking_data.py`、`tools/precache_frames.py`、`train.py` 仍然走原始单 Agent 逻辑。

**双 Agent 逻辑已在主目录实现**

加 `--multi_agent` 时，主目录脚本调用同文件内置函数，不再导入 `multi_agent/` 目录。

**此模型仍然不使用 Diffusion**

- 规划头是 MLP 直接回归；
- 没有噪声添加/去噪过程；
- 没有 DDPM/DDIM scheduler；
- 输出是直接的未来 waypoint。

## 6. 与单 Agent 模式的关系

不加 `--multi_agent` 时，三个脚本仍然走原来的单 Agent 逻辑：

```bash
python -m tools.make_tracking_data ...
python -m tools.precache_frames ...
python train.py ...
```

加 `--multi_agent` 时，主目录脚本会调用同文件内置的双 Agent 实现：

```text
tools/make_tracking_data.py -> main_multi_agent()
tools/precache_frames.py    -> main_multi_agent_precache()
train.py              -> train_multi_agent(parse_multi_agent_args())
```

`multi_agent/` 目录下的旧文件仍可作为历史参考，但主目录入口已经不再从那里导入代码。
