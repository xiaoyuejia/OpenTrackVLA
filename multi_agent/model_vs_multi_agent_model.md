# `model.py` 与 `multi_agent_model.py` 对比说明

本文档对比原始单 Agent 模型 `model.py` 和新增双 Agent 模型 `multi_agent_model.py` 的主要差异，便于查看新模型相对旧模型改了什么、保留了什么，以及后续接训练代码时需要注意哪些接口变化。

## 1. 总体定位

| 项目 | `model.py` | `multi_agent_model.py` |
| --- | --- | --- |
| 模型类型 | 单 Agent OpenTrackVLA | 双 Agent Multi-Agent OpenTrackVLA |
| 主类 | `OpenTrackVLA` | `MultiAgentOpenTrackVLA` / `OpenTrackVLAMultiAgent` |
| 配置类 | `ModelConfig` | `MultiAgentModelConfig` |
| 输入路数 | 1 路视觉历史 + 当前帧 | 2 路 Agent 视觉历史 + 当前帧 |
| bbox 使用 | `TVIEmbedder` 中有 `bbox_proj`，Dataset 里也尝试产出 `bbox_feat`，但模型 forward 当前没有真正拼入序列 | bbox 是正式条件 token，参与 LLM self-attention |
| 查询 token | 1 个 `ACT` token | `ACT_1`、`ACT_2`、`GND` 三个查询 token |
| 输出 | 单个轨迹 `(B, N, 3)` | 双 Agent 轨迹 + Grounding 输出 |
| 是否包含 Dataset | 包含 `JsonTrackingDataset`、`collate_batch`、推理逻辑 | 只包含模型代码，未包含 Dataset/训练循环 |

## 2. 保留不变的核心设计

两个文件都沿用了 OpenTrackVLA 的主干思路：

- 使用 Qwen/Qwen3-0.6B 作为 LLM backbone。
- 使用 `CrossModalityProjector` 把视觉 token 从 `vision_feat_dim` 投影到 LLM hidden size `D`。
- 使用 TVI/时间 token 给视觉序列提供时间信息。
- 使用 `PlannerHead3L` 从 LLM hidden state 直接回归 waypoints。
- 默认 waypoint 维度为 `(x, y, theta)`，即 `action_dims=3`。
- 支持 `use_tanh_actions` 和 `alpha_xy` 对输出范围进行控制或缩放。

## 3. 配置差异

### 原始 `ModelConfig`

`model.py` 的配置更偏单 Agent：

```python
llm_name
freeze_llm
n_waypoints
max_time
beta_nav
use_angle_tvi
use_tanh_actions
alpha_xy
```

### 新增 `MultiAgentModelConfig`

`multi_agent_model.py` 在旧配置基础上增加了多 Agent 和 Grounding 相关参数：

```python
action_dims
max_views = 2
num_agents = 2
num_kinds = 5
insert_time_tokens
bbox_delta_scale
return_token_logits
text_max_length
```

主要新增含义：

| 参数 | 作用 |
| --- | --- |
| `num_agents=2` | 固定双 Agent 建模 |
| `max_views=2` | 两个 Agent 对应两个 view id |
| `num_kinds=5` | 支持 history/current/bbox/act/gnd 五类 token |
| `insert_time_tokens` | 是否像旧模型一样额外插入时间 marker token |
| `bbox_delta_scale` | Grounding head 对 bbox refinement 的最大修正幅度 |
| `return_token_logits` | 是否输出 token-level grounding logits |

## 4. TVI 设计差异

### 原始 `TVIEmbedder`

原始 TVI 只有：

```python
time_emb
view_emb
kind_emb
angle_proj
bbox_proj
```

其中 `kind_emb` 只有 2 类：

| kind | 含义 |
| --- | --- |
| 0 | 历史粗粒度视觉 token |
| 1 | 当前细粒度视觉 token |

原始 forward 中的序列实际是：

```text
[text] + [history coarse visual with TVI] + [current fine visual with TVI] + [ACT]
```

虽然 `TVIEmbedder` 里定义了 `bbox_proj`，但当前 `OpenTrackVLA.forward(..., bbox_feat=None)` 没有把 bbox token 拼入 LLM 序列。

### 新 `MultiAgentTVIEmbedder`

新 TVI 增加了：

```python
agent_emb
bbox_proj = MLP(4 -> D)
```

并把 token 类型扩展成 5 类：

| 常量 | id | 含义 |
| --- | --- | --- |
| `KIND_HISTORY` | 0 | 历史粗视觉 token |
| `KIND_CURRENT` | 1 | 当前细视觉 token |
| `KIND_BBOX` | 2 | bbox 条件 token |
| `KIND_ACT` | 3 | 动作查询 token |
| `KIND_GND` | 4 | grounding 查询 token |

新模型中的 token embedding 形式变成：

```text
token = visual_or_bbox_or_query_token
      + time_emb[t]
      + view_emb[view_id]
      + kind_emb[kind_id]
      + agent_emb[agent_id]
```

注意：bbox/query token 没有逐帧时间下标，但会加上 `kind_emb + agent_emb + view_emb`。

## 5. 输入接口差异

### 原始模型输入

`OpenTrackVLA.forward` 接收单 Agent 输入：

```python
coarse_tokens: (B, Nc, C)
coarse_tidx:   (B, Nc)
fine_tokens:   (B, Nf, C)
fine_tidx:     (B, Nf)
instructions:  List[str]
yaw_hist:      Optional[(B, H)]
yaw_curr:      Optional[(B, 1)]
bbox_feat:     Optional[(B, 4)]  # 当前未使用
```

典型 shape：

```text
coarse_tokens: (B, 31*4, 1536)
fine_tokens:   (B, 64, 1536)
output:        (B, 8, 3)
```

### 新模型输入

`MultiAgentOpenTrackVLA.forward` 支持两种输入方式。

方式一：堆叠输入：

```python
coarse_tokens: (B, 2, Nc, C)
coarse_tidx:   (B, 2, Nc)
fine_tokens:   (B, 2, Nf, C)
fine_tidx:     (B, 2, Nf)
bbox_feat:     (B, 2, 4)
instructions:  List[str]
```

方式二：分开传两个 Agent：

```python
agent1_coarse_tokens
agent1_coarse_tidx
agent1_fine_tokens
agent1_fine_tidx
agent1_bbox_feat

agent2_coarse_tokens
agent2_coarse_tidx
agent2_fine_tokens
agent2_fine_tidx
agent2_bbox_feat
```

其中 bbox 要求已经归一化到 `[0, 1]`，shape 为：

```text
(cx, cy, w, h) 或其他 4 维归一化 bbox 表示
```

当前实现会拒绝明显未归一化的 bbox：

```python
if bbox.detach().amax() > 1.5:
    raise ValueError(...)
```

## 6. 序列拼接差异

### 原始模型序列

原始模型的 LLM 输入序列较短：

```text
[文本指令]
+ [单 Agent 历史粗视觉 tokens]
+ [单 Agent 当前细视觉 tokens]
+ [ACT]
```

隐状态读取方式：

```python
h_act = out.last_hidden_state[:, -1, :]
```

也就是默认最后一个 token 就是 `ACT`。

### 新模型序列

新模型的 LLM 输入序列变成双 Agent + Grounding：

```text
[文本指令]
+ [Agent-1 历史粗视觉 tokens]
+ [Agent-1 当前细视觉 tokens]
+ [Agent-1 BBOX]
+ [ACT_1]
+ [Agent-2 历史粗视觉 tokens]
+ [Agent-2 当前细视觉 tokens]
+ [Agent-2 BBOX]
+ [ACT_2]
+ [GND]
```

隐状态读取方式不再只取最后一位，而是记录三个查询 token 的位置：

```python
h_act1 = hidden[:, act1_pos, :]
h_act2 = hidden[:, act2_pos, :]
h_gnd  = hidden[:, gnd_pos, :]
```

这样可以分别用于：

| hidden state | 用途 |
| --- | --- |
| `h_act1` | Agent-1 waypoint 规划 |
| `h_act2` | Agent-2 waypoint 规划 |
| `h_gnd` | 双 Agent bbox / visibility / token grounding |

## 7. 规划头差异

### 原始模型

只有一个规划头：

```python
self.planner = PlannerHead3L(...)
```

输出：

```python
tau_pred = self.planner(h_act) * self.alpha_task
```

shape：

```text
(B, n_waypoints, 3)
```

### 新模型

有两个独立规划头：

```python
self.planner_agent1 = PlannerHead3L(...)
self.planner_agent2 = PlannerHead3L(...)
```

输出：

```python
agent1_waypoints = self.planner_agent1(h_act1) * self.alpha_task
agent2_waypoints = self.planner_agent2(h_act2) * self.alpha_task
```

shape：

```text
agent1_waypoints: (B, n_waypoints, 3)
agent2_waypoints: (B, n_waypoints, 3)
waypoints:        (B, 2, n_waypoints, 3)
```

两个 head 参数不共享，因此 Agent-1 和 Agent-2 可以学习不同运动模式。

## 8. Grounding Head 是新增模块

`model.py` 没有 Grounding 输出。

`multi_agent_model.py` 新增：

```python
class GroundingHead(nn.Module)
```

输入：

```text
h_gnd: (B, D)
bbox_feat: Optional[(B, 2, 4)]
visual_tokens: Optional[(B, 2, N_visual, D)]
```

输出：

| 输出字段 | shape | 含义 |
| --- | --- | --- |
| `refined_bbox` | `(B, 2, 4)` | 对两个 Agent 的 bbox 做修正 |
| `visible_score` | `(B, 2)` | 每个 Agent 的目标可见性分数 |
| `token_logits` | `(B, 2, N_visual)` | token-level grounding 分数 |

`refined_bbox` 当前逻辑是：

```python
delta = tanh(MLP(h_gnd)) * bbox_delta_scale
refined_bbox = clamp(bbox_feat + delta, 0, 1)
```

也就是说它不是从零生成 bbox，而是在输入 bbox 附近做 refinement。

## 9. 输出格式差异

### 原始模型输出

原始 forward 直接返回 tensor：

```python
tau_pred
```

shape：

```text
(B, 8, 3)
```

### 新模型输出

默认返回 dict：

```python
{
    "agent1_waypoints": ...,
    "agent2_waypoints": ...,
    "waypoints": ...,
    "refined_bbox": ...,
    "visible_score": ...,
    "token_logits": ...,
}
```

其中：

```text
agent1_waypoints: (B, 8, 3)
agent2_waypoints: (B, 8, 3)
waypoints:        (B, 2, 8, 3)
refined_bbox:     (B, 2, 4)
visible_score:    (B, 2)
token_logits:     (B, 2, N_visual_tokens)
```

如果调用时传入：

```python
return_dict=False
```

则返回：

```python
(agent1_waypoints, agent2_waypoints, grounding)
```

## 10. Dataset 和训练接入差异

这是目前最重要的工程差异。

`model.py` 不只是模型文件，还包含：

- `load_tokens_file`
- `integrate_actions_to_waypoints`
- `JsonTrackingDataset`
- `collate_batch`
- `_run_inference`

而 `multi_agent_model.py` 目前只实现模型，不包含：

- 双 Agent Dataset
- 双 Agent collate
- 双 Agent loss
- 训练脚本接入
- 推理脚本接入

因此当前新模型已经可以被 import 和 forward，但还不能直接替换现有训练流程。要完整训练新模型，需要继续修改或新增：

```text
multi_agent_dataset.py 或改造 train.py
multi_agent_collate_batch
双 Agent waypoint loss
grounding loss，可选
checkpoint 保存和加载逻辑
eval / inference 输出适配
```

## 11. 参数量与计算量变化

相对 `model.py`，新模型增加的参数主要来自：

- `agent_emb`
- 扩展后的 `kind_emb`
- bbox MLP projector
- 第二个 ACT token
- GND token
- 第二个 planner head
- Grounding head

LLM backbone 仍然是同一个 Qwen 模型。主要计算量增加来自：

1. LLM 输入序列约接近翻倍，因为有两个 Agent 的视觉 tokens。
2. 额外的 bbox/ACT/GND token 很少，不是主要瓶颈。
3. 第二个 planner head 和 grounding head 相对 LLM 很小。

## 12. 快速使用示例

堆叠输入方式：

```python
from multi_agent_model import MultiAgentOpenTrackVLA, MultiAgentModelConfig

model = MultiAgentOpenTrackVLA(
    MultiAgentModelConfig(
        llm_name="Qwen/Qwen3-0.6B",
        n_waypoints=8,
        alpha_xy=2.0,
    ),
    vision_feat_dim=1536,
)

out = model(
    coarse_tokens=coarse_tokens,  # (B, 2, 124, 1536)
    coarse_tidx=coarse_tidx,      # (B, 2, 124)
    fine_tokens=fine_tokens,      # (B, 2, 64, 1536)
    fine_tidx=fine_tidx,          # (B, 2, 64)
    bbox_feat=bbox_feat,          # (B, 2, 4), normalized
    instructions=instructions,
)

waypoints = out["waypoints"]          # (B, 2, 8, 3)
refined_bbox = out["refined_bbox"]    # (B, 2, 4)
visible_score = out["visible_score"]  # (B, 2)
```

## 13. 一句话总结

`model.py` 是单目、单 Agent、单 ACT token 的 waypoint 回归模型；`multi_agent_model.py` 把它扩展为双 Agent 输入、显式 bbox 条件、双规划头和 grounding 查询输出的多 Agent VLA 模型，但目前还需要额外的数据集和训练循环适配才能完整训练。

