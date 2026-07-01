# 双 Agent 模型架构与代码分析

本文档从数据流、模型结构、训练代码和后续扩展四个角度说明 `multi_agent/` 中的新代码。

## 1. 数据处理线框图

```text
UnrealZoo 原始 episode
────────────────────────────────────────────────────────────
0_drone.mp4              0_robotdog.mp4
0_drone_info.json        0_robotdog_info.json
0.json                   0_global.mp4
      │                         │
      │ ffmpeg 抽帧              │ ffmpeg 抽帧
      ▼                         ▼
frames/.../drone/*.jpg   frames/.../robotdog/*.jpg
      │                         │
      │ 读取 target_bbox/action  │ 读取 target_bbox/action
      └──────────────┬──────────┘
                     ▼
        make_multi_agent_tracking_data.py
                     │
                     ▼
jsonl/.../0.jsonl
────────────────────────────────────────────────────────────
每条样本:
agent1_images/current/bbox/waypoints
agent2_images/current/bbox/waypoints
bbox_feat=(2,4), waypoints=(2,N,3), valid_mask=(2,N)
```

关键点：

- `agent1` 默认是无人机，`agent2` 默认是机器狗。
- `target_bbox` 原始格式是像素空间 `x, y, w, h`，预处理后变成 `cx, cy, w, h` 且归一化到 `[0, 1]`。
- 路点标签由每个 Agent 的 `base_velocity` 积分得到，格式为 `[x, y, theta]`。
- 历史帧窗口是 `[j-history, j)`，当前帧是 `j`，未来标签从 `j` 开始积分。

## 2. 视觉缓存线框图

```text
JSONL 中引用的 frame path
      │
      ▼
precache_multi_agent_frames.py
      │
      ├── DINO/SigLIP 编码
      │
      ├── GridPool 粗粒度: 4 tokens/frame
      │
      └── GridPool 细粒度: 64 tokens/current frame
      ▼
vision_cache/frames/.../*_vcoarse.pt
vision_cache/frames/.../*_vfine.pt
```

训练 Dataset 的读取约定：

```text
frames/seed_100/.../drone/frame_00001.jpg
  对应
vision_cache/frames/seed_100/.../drone/frame_00001_vcoarse.pt
vision_cache/frames/seed_100/.../drone/frame_00001_vfine.pt
```

## 3. 模型输入线框图

```text
Batch 输入
────────────────────────────────────────────────────────────
coarse_tokens: (B, 2, 124, C)   # 2 个 Agent，每个 31 帧 * 4 token
fine_tokens:   (B, 2, 64, C)    # 2 个 Agent 当前帧细粒度 token
bbox_feat:     (B, 2, 4)        # 两个 Agent 当前 bbox
instruction:   List[str]
      │
      ▼
MultiAgentOpenTrackVLA
```

其中：

```text
第 2 维 agent index:
0 = Agent-1，默认 drone
1 = Agent-2，默认 robotdog
```

## 4. 模型架构线框图

```text
Agent-1 视觉 token                         Agent-2 视觉 token
────────────────────                       ────────────────────
coarse: (B,124,C)                          coarse: (B,124,C)
fine:   (B,64,C)                           fine:   (B,64,C)
bbox:   (B,4)                               bbox:   (B,4)
      │                                           │
      ▼                                           ▼
CrossModalityProjector                     CrossModalityProjector
Linear(C -> D)                              Linear(C -> D)
      │                                           │
      ▼                                           ▼
MultiAgentTVIEmbedder                      MultiAgentTVIEmbedder
time_emb + view_emb                        time_emb + view_emb
kind_emb + agent_emb                       kind_emb + agent_emb
      │                                           │
      └───────────────┬───────────────────────────┘
                      ▼
LLM 输入序列
────────────────────────────────────────────────────────────
[文本指令]
+ [A1 历史粗视觉 tokens]
+ [A1 当前细视觉 tokens]
+ [A1 BBOX token]
+ [ACT_1]
+ [A2 历史粗视觉 tokens]
+ [A2 当前细视觉 tokens]
+ [A2 BBOX token]
+ [ACT_2]
+ [GND]
                      │
                      ▼
                 Qwen3-0.6B
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
     h_ACT1        h_ACT2         h_GND
        │             │             │
        ▼             ▼             ▼
PlannerHead-1   PlannerHead-2   GroundingHead
        │             │             │
        ▼             ▼             ▼
A1 waypoints    A2 waypoints    bbox/visible/token logits
(B,N,3)         (B,N,3)         (B,2,4)/(B,2)/可选
```

## 5. TVI 嵌入解释

每个视觉 token 会加上：

```text
visual_token
+ time_emb[t]
+ view_emb[v]
+ kind_emb[k]
+ agent_emb[a]
```

五种 kind：

| kind | 含义 |
| --- | --- |
| `KIND_HISTORY` | 历史粗视觉 token |
| `KIND_CURRENT` | 当前细视觉 token |
| `KIND_BBOX` | bbox 条件 token |
| `KIND_ACT` | 动作规划查询 token |
| `KIND_GND` | grounding 查询 token |

这样 LLM 能区分：

- 这是哪个 Agent 的 token。
- 这是历史还是当前帧。
- 这是视觉 token、bbox token，还是查询 token。
- 这个 token 属于哪个时间步。

## 6. 输出与损失

模型输出：

```text
agent1_waypoints: (B, N, 3)
agent2_waypoints: (B, N, 3)
waypoints:        (B, 2, N, 3)
refined_bbox:     (B, 2, 4)
visible_score:    (B, 2)
token_logits:     (B, 2, N_visual)  # 可选
```

训练主损失：

```text
L_nav = masked_mse(pred_waypoints, gt_waypoints, valid_mask)
loss = beta_nav * L_nav
     + beta_bbox * L_bbox
     + beta_visible * L_visible
```

默认：

```text
beta_bbox = 0
beta_visible = 0
```

也就是说当前主要训练双 Agent 未来路点预测；grounding 头作为结构预留，可以后续打开辅助监督。

## 7. 代码文件职责

### `make_multi_agent_tracking_data.py`

职责：

- 查找 paired episode。
- 按 episode 状态过滤数据。
- 用 ffmpeg 抽取无人机和机器狗视角图片。
- 把 bbox 转为 `cxcywh_norm`。
- 把未来动作积分为 waypoint 标签。
- 写出 JSONL 和可选聚合 JSON。

核心函数：

| 函数 | 说明 |
| --- | --- |
| `collect_paired_episodes` | 查找 drone/robotdog 成对文件 |
| `episode_status_ok` | 根据 success/collision/following_rate 过滤 |
| `normalize_bbox_xywh` | pixel xywh 转 normalized cxcywh |
| `integrate_actions` | 速度积分成局部路点 |
| `build_samples_for_episode` | 生成滑动窗口训练样本 |

### `precache_multi_agent_frames.py`

职责：

- 从 JSONL 中收集训练实际引用的帧。
- 对每张图片跑视觉编码器。
- 保存 coarse/fine token cache。
- 支持 `--list_only` 只检查引用帧，不加载模型。

核心函数：

| 函数 | 说明 |
| --- | --- |
| `collect_frame_refs` | 从 JSONL/dataset.json 收集图片路径 |
| `encode_single` | 生成 4-token coarse 和 64-token fine |
| `maybe_make_coarse_from_fine` | 已有 fine 时直接池化 coarse |

### `multi_agent_model.py`

职责：

- 定义双 Agent VLA 模型。
- 处理双 Agent 视觉、bbox、文本融合。
- 输出双 Agent waypoints 和 grounding 结果。

核心模块：

| 模块 | 说明 |
| --- | --- |
| `CrossModalityProjector` | 视觉 token 维度投影 |
| `MultiAgentTVIEmbedder` | 时间、视角、类别、Agent 身份嵌入 |
| `PlannerHead3L` | ACT hidden state -> waypoints |
| `GroundingHead` | GND hidden state -> bbox/visibility/token logits |
| `MultiAgentOpenTrackVLA` | 完整双 Agent 模型 |

### `train_multi_agent.py`

职责：

- 读取双 Agent JSONL 和 vision cache。
- 构造 `(B, 2, ...)` batch。
- 训练 `MultiAgentOpenTrackVLA`。
- 保存 checkpoint 和训练日志。

核心模块：

| 模块 | 说明 |
| --- | --- |
| `MultiAgentJsonDataset` | 双 Agent 数据读取 |
| `collate_batch` | 合并 batch |
| `forward_loss` | 模型 forward 和 loss 计算 |
| `train` | 训练主循环 |
| `evaluate` | 小规模验证 |

## 8. 与原始单 Agent 版本的核心区别

| 维度 | 原始 `model.py` | 新双 Agent 版本 |
| --- | --- | --- |
| 输入 Agent 数 | 1 | 2 |
| bbox 是否入模 | 基本未进入序列 | bbox token 正式进入 LLM 序列 |
| ACT token | 1 个 | 2 个 |
| Grounding token | 无 | 1 个 GND |
| 输出 | `(B,N,3)` | `(B,2,N,3)` |
| Dataset | 单视角当前帧 + 历史帧 | drone/robotdog 双视角当前帧 + 历史帧 |

## 9. 后续可扩展方向

- 打开 `--beta_bbox`，训练 bbox refinement。
- 打开 `--beta_visible`，训练可见性预测。
- 为 `token_logits` 添加 token-level grounding 监督。
- 将 Agent 数从 2 扩展为 N，但需要同步修改模型查询 token 和 Dataset。
- 在推理端接入双 Agent 实时观测，让无人机和机器狗协同规划。

