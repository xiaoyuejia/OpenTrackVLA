# `OpenTrackVLA` 与 `MultiAgentOpenTrackVLA` 类级代码对比

本文档专门对比两个模型主类：

- 原始单 Agent 模型：`model.py` 中的 `class OpenTrackVLA(nn.Module)`
- 新双 Agent 模型：`multi_agent/multi_agent_model.py` 中的 `class MultiAgentOpenTrackVLA(nn.Module)`

重点从代码实现角度解释二者的相同点、不同点、输入输出、序列拼接方式和训练影响。

## 1. 总体结论

一句话概括：

```text
OpenTrackVLA 是单 Agent、单 ACT token、单规划头的路点回归模型；
MultiAgentOpenTrackVLA 是双 Agent、双 ACT token、显式 bbox token、额外 GND token 和 grounding head 的扩展模型。
```

二者都保留了同一条主干思想：

```text
视觉 token -> 投影到 LLM hidden size -> 与文本 token 拼接 -> Qwen LLM 融合 -> 查询 token hidden state -> MLP 输出路点
```

但新模型把输入、token 类型、查询 token 和输出头都扩展到了双 Agent 场景。

## 2. 文件位置和类名

| 项目 | 原始模型 | 新模型 |
| --- | --- | --- |
| 文件 | `model.py` | `multi_agent/multi_agent_model.py` |
| 主类 | `OpenTrackVLA` | `MultiAgentOpenTrackVLA` |
| 配置类 | `ModelConfig` | `MultiAgentModelConfig` |
| 辅助 TVI 类 | `TVIEmbedder` | `MultiAgentTVIEmbedder` |
| 规划头 | `PlannerHead3L` | `PlannerHead3L`，但实例化两个 |
| 新增输出头 | 无 | `GroundingHead` |

## 3. 配置类对比

### `ModelConfig`

原始配置较简单：

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

它只服务单 Agent waypoint 预测，因此没有 Agent 数、bbox 输出、grounding 输出等参数。

### `MultiAgentModelConfig`

新配置增加了多 Agent 和 grounding 相关字段：

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

新增字段含义：

| 字段 | 作用 |
| --- | --- |
| `action_dims` | 输出动作/路点维度，默认 3，即 `[x, y, theta]` |
| `max_views=2` | 两个 Agent 视角 |
| `num_agents=2` | 明确双 Agent |
| `num_kinds=5` | 支持 history/current/bbox/ACT/GND 五类 token |
| `insert_time_tokens` | 是否额外插入显式时间 marker |
| `bbox_delta_scale` | bbox refinement 最大修正幅度 |
| `return_token_logits` | 是否输出 token-level grounding logits |
| `text_max_length` | 文本 tokenizer 最大长度 |

## 4. `__init__` 初始化对比

### 4.1 LLM 加载

二者都加载 Qwen/Qwen3-0.6B：

```python
self.llm = AutoModel.from_pretrained(...)
self.tokenizer = AutoTokenizer.from_pretrained(...)
self.llm.requires_grad_(not cfg.freeze_llm)
```

相同点：

- 都优先尝试 ModelScope，失败后回退 HuggingFace。
- 都支持 `freeze_llm`。
- 都从 `self.llm.config.hidden_size` 得到 LLM hidden size。

不同点：

| 项目 | `OpenTrackVLA` | `MultiAgentOpenTrackVLA` |
| --- | --- | --- |
| dtype 保存 | 直接使用 `self.llm.dtype` | 显式保存 `self.llm_dtype = next(self.llm.parameters()).dtype` |
| 日志 | 简单打印 ModelScope fallback | 打印 rank 和加载耗时，更适合 DDP |
| HF 参数 | 使用 `torch_dtype` | fallback 中使用新版 `dtype` |

### 4.2 视觉投影器

二者都有：

```python
self.proj = CrossModalityProjector(vision_feat_dim, self.D)
```

作用相同：

```text
(B, N, C_vision) -> (B, N, D_llm)
```

区别是输入的 Agent 维度不同：

```text
OpenTrackVLA:
coarse_tokens: (B, Nc, C)
fine_tokens:   (B, Nf, C)

MultiAgentOpenTrackVLA:
coarse_tokens: (B, 2, Nc, C)
fine_tokens:   (B, 2, Nf, C)
```

新模型在 forward 里会先拆成 Agent-1 和 Agent-2，再分别调用同一个 `self.proj`。

### 4.3 TVI 模块

原始模型：

```python
self.tvi = TVIEmbedder(self.D, max_time=cfg.max_time)
```

新模型：

```python
self.tvi = MultiAgentTVIEmbedder(
    self.D,
    max_time=cfg.max_time,
    max_views=cfg.max_views,
    num_agents=cfg.num_agents,
    num_kinds=cfg.num_kinds,
)
```

核心变化：

| 项目 | 原始 `TVIEmbedder` | 新 `MultiAgentTVIEmbedder` |
| --- | --- | --- |
| 时间嵌入 | 有 | 有 |
| 视角嵌入 | 有，默认 1 个 view | 有，默认 2 个 view |
| kind 嵌入 | 2 类：history/current | 5 类：history/current/bbox/ACT/GND |
| agent 嵌入 | 无 | 有，区分 drone/robotdog |
| bbox 编码 | 有 `bbox_proj`，但主 forward 未真正使用 | bbox token 正式进入 LLM 序列 |
| bbox projector | 单 Linear | LayerNorm + Linear + GELU + Linear |

### 4.4 查询 token

原始模型只有一个查询 token：

```python
self.act_token = nn.Parameter(torch.zeros(1, 1, self.D))
```

新模型有三个查询 token：

```python
self.act_token_1 = nn.Parameter(torch.zeros(1, 1, self.D))
self.act_token_2 = nn.Parameter(torch.zeros(1, 1, self.D))
self.gnd_token = nn.Parameter(torch.zeros(1, 1, self.D))
```

含义：

| token | 作用 |
| --- | --- |
| `ACT_1` | Agent-1 规划查询 |
| `ACT_2` | Agent-2 规划查询 |
| `GND` | 双 Agent grounding 查询，用于 bbox/visibility/token logits |

### 4.5 输出头

原始模型：

```python
self.planner = PlannerHead3L(...)
```

新模型：

```python
self.planner_agent1 = PlannerHead3L(...)
self.planner_agent2 = PlannerHead3L(...)
self.grounding_head = GroundingHead(...)
```

区别：

- 原始模型只有一个 planner，输出一个 Agent 的未来路点。
- 新模型有两个独立 planner，分别输出两个 Agent 的未来路点。
- 新模型额外有 grounding head，输出 bbox refinement、visibility 和可选 token logits。

## 5. TVI 处理逻辑对比

### 5.1 原始模型 `_interleave_tvi`

原始模型做法：

```text
每一帧视觉 token 前插入一个 time token
可选再插入 angle token
```

序列类似：

```text
[T0] [frame0 coarse tokens]
[T1] [frame1 coarse tokens]
...
[TH] [current fine tokens]
```

注意：原始模型的 time/view/kind 信息主要通过“额外插入 marker token”提供，视觉 token 本身没有在 `forward` 中逐 token 加上 time embedding。

### 5.2 新模型 `_encode_agent_stream`

新模型有两层 TVI 注入：

第一层：直接加到每个视觉 token 上：

```python
vis_c = self.tvi.add_visual_tvi(vis_c, coarse_tidx, KIND_HISTORY, agent_id, view_id)
vis_f = self.tvi.add_visual_tvi(vis_f, fine_tidx, KIND_CURRENT, agent_id, view_id)
```

也就是：

```text
visual_token
+ time_emb[t]
+ kind_emb[k]
+ agent_emb[a]
+ view_emb[v]
```

第二层：可选额外插入显式 marker：

```python
seq_c = self._interleave_markers(...)
seq_f = self._interleave_markers(...)
```

因此新模型比原始模型的身份信息更强：

- 每个视觉 token 自身知道属于哪个时间。
- 每个视觉 token 自身知道属于哪个 Agent。
- 每个视觉 token 自身知道属于哪个视角。
- 每个视觉 token 自身知道是历史还是当前。

## 6. forward 输入对比

### 6.1 原始模型输入

`OpenTrackVLA.forward`：

```python
forward(
    coarse_tokens,
    coarse_tidx,
    fine_tokens,
    fine_tidx,
    instructions,
    yaw_hist=None,
    yaw_curr=None,
    bbox_feat=None,
)
```

典型 shape：

```text
coarse_tokens: (B, 124, C)
coarse_tidx:   (B, 124)
fine_tokens:   (B, 64, C)
fine_tidx:     (B, 64)
instructions:  List[str]
```

其中 `bbox_feat` 虽然在函数签名中存在，但当前代码没有把它拼进 LLM 序列。

### 6.2 新模型输入

`MultiAgentOpenTrackVLA.forward` 支持两种输入。

第一种：堆叠输入，训练脚本使用这种：

```python
coarse_tokens: (B, 2, Nc, C)
coarse_tidx:   (B, 2, Nc)
fine_tokens:   (B, 2, Nf, C)
fine_tidx:     (B, 2, Nf)
bbox_feat:     (B, 2, 4)
instructions:  List[str]
```

第二种：分开输入，便于单独调试：

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

新模型通过 `_split_stacked_inputs` 把输入统一成：

```text
a1_c, a1_ct, a1_f, a1_ft
a2_c, a2_ct, a2_f, a2_ft
```

## 7. LLM 序列拼接对比

这是两个类最核心的代码差异。

### 7.1 原始模型序列

`OpenTrackVLA.forward` 中：

```python
pieces = [txt_emb] + [vis_c, vis_f, act]
seq = torch.cat(pieces, dim=1)
```

实际序列：

```text
[文本指令]
+ [单 Agent 历史粗视觉 tokens]
+ [单 Agent 当前细视觉 tokens]
+ [ACT]
```

由于 `ACT` 是最后一个 token，所以取 hidden state 很简单：

```python
h_act = out.last_hidden_state[:, -1, :]
```

### 7.2 新模型序列

`MultiAgentOpenTrackVLA.forward` 中：

```python
pieces = [txt_emb, a1_seq, act1, a2_seq, act2, gnd]
seq = torch.cat(pieces, dim=1)
```

实际序列：

```text
[文本指令]
+ [Agent-1 历史粗视觉 tokens]
+ [Agent-1 当前细视觉 tokens]
+ [Agent-1 BBOX token]
+ [ACT_1]
+ [Agent-2 历史粗视觉 tokens]
+ [Agent-2 当前细视觉 tokens]
+ [Agent-2 BBOX token]
+ [ACT_2]
+ [GND]
```

因为 `ACT_1`、`ACT_2`、`GND` 分布在不同位置，新模型必须显式计算位置：

```python
act1_pos = lengths[0] + lengths[1]
act2_pos = lengths[0] + lengths[1] + lengths[2] + lengths[3]
gnd_pos = sum(lengths) - 1
```

然后分别取：

```python
h_act1 = hidden[:, act1_pos, :]
h_act2 = hidden[:, act2_pos, :]
h_gnd  = hidden[:, gnd_pos, :]
```

## 8. LLM 融合能力对比

### 原始模型

原始模型让 LLM 学习：

```text
文本指令 <-> 单 Agent 历史视觉 <-> 单 Agent 当前视觉 <-> ACT
```

它适合单机器人/单视角路径预测。

### 新模型

新模型让 LLM 学习：

```text
文本指令
<-> Agent-1 视觉与 bbox
<-> Agent-2 视觉与 bbox
<-> ACT_1 / ACT_2 / GND 查询
```

因此它额外建模：

- 两个 Agent 对同一目标的观测一致性。
- 无人机俯视视角和机器狗地面视角之间的互补关系。
- bbox 条件与未来运动规划之间的关系。
- grounding 查询与双 Agent 视觉 token 的关系。

## 9. 输出对比

### 9.1 原始模型输出

原始模型：

```python
a_hat = self.planner(h_act)
tau_pred = a_hat * self.alpha_task
return tau_pred
```

输出 shape：

```text
(B, n_waypoints, 3)
```

返回类型是单个 Tensor。

### 9.2 新模型输出

新模型：

```python
agent1_waypoints = self.planner_agent1(h_act1) * self.alpha_task
agent2_waypoints = self.planner_agent2(h_act2) * self.alpha_task
grounding = self.grounding_head(h_gnd, ...)
```

默认返回 dict：

```python
{
    "agent1_waypoints": ...,
    "agent2_waypoints": ...,
    "waypoints": ...,
    "refined_bbox": ...,
    "visible_logits": ...,
    "visible_score": ...,
    "token_logits": ...,
}
```

输出 shape：

```text
agent1_waypoints: (B, n_waypoints, 3)
agent2_waypoints: (B, n_waypoints, 3)
waypoints:        (B, 2, n_waypoints, 3)
refined_bbox:     (B, 2, 4)
visible_logits:   (B, 2)
visible_score:    (B, 2)
token_logits:     (B, 2, N_visual) 或 None
```

如果设置：

```python
return_dict=False
```

则返回：

```python
(agent1_waypoints, agent2_waypoints, grounding)
```

## 10. bbox 处理差异

### 原始模型

原始 `TVIEmbedder` 里定义了：

```python
self.bbox_proj = nn.Linear(4, d_model)
```

也有：

```python
make_bbox_token(...)
```

但是 `OpenTrackVLA.forward` 中没有使用 `bbox_feat`：

```python
extra = []
pieces = [txt_emb] + ([extra[0]] if extra else []) + [vis_c, vis_f, act]
```

因此 bbox 在原始主类中基本是“预留接口”，并没有真正参与 LLM。

### 新模型

新模型显式使用 bbox：

```python
bbox = self._normalize_bbox(bbox_feat, B, device)
bbox_tok = self.tvi.make_bbox_token(bbox_feat, agent_id, view_id)
```

每个 Agent 都有自己的 bbox token：

```text
[A1 BBOX token]
[A2 BBOX token]
```

它们会进入 LLM self-attention，影响：

- `ACT_1`
- `ACT_2`
- `GND`

也就是说 bbox 不只是标签，而是模型输入条件。

## 11. GroundingHead 是新模型独有部分

原始模型没有 grounding 输出。

新模型新增：

```python
self.grounding_head = GroundingHead(...)
```

输入：

```text
h_gnd: (B, D)
bbox_feat: (B, 2, 4)
visual_tokens: (B, 2, N_visual, D) 可选
```

输出：

| 输出 | shape | 说明 |
| --- | --- | --- |
| `refined_bbox` | `(B, 2, 4)` | 对输入 bbox 做修正 |
| `visible_logits` | `(B, 2)` | 可见性 logits，用于 BCEWithLogitsLoss |
| `visible_score` | `(B, 2)` | sigmoid 后的可见性概率 |
| `token_logits` | `(B, 2, N_visual)` | 视觉 token 级 grounding 分数 |

当前训练中，`beta_bbox` 和 `beta_visible` 默认是 0，所以 grounding head 默认不作为主损失训练；它是为后续 bbox refinement、可见性预测和 token-level grounding 预留的扩展接口。

## 12. PlannerHead 使用差异

两个类使用的 `PlannerHead3L` 结构基本相同：

```text
LayerNorm(D)
Linear(D -> 2D) + GELU
Linear(2D -> 2D) + GELU
Linear(2D -> n_waypoints * action_dims)
tanh 可选
reshape -> (B, n_waypoints, action_dims)
```

差异：

| 项目 | 原始模型 | 新模型 |
| --- | --- | --- |
| planner 数量 | 1 | 2 |
| 参数共享 | 不涉及 | Agent-1 和 Agent-2 不共享 planner |
| 输入 hidden | `h_act` | `h_act1`、`h_act2` |
| 输出 | 单 Agent 路点 | 双 Agent 路点 |

为什么新模型用两个 planner？

- 无人机和机器狗运动学不同。
- 无人机 bbox 视角和机器狗 bbox 视角差异大。
- 两个 Agent 的 action 分布可能不同。
- 独立 planner 能让共享 LLM 融合信息，同时保留各自控制头。

## 13. alpha_task 缩放对比

原始模型：

```python
alpha = torch.tensor((1.0, 1.0, 1.0))
```

如果设置 `alpha_xy`，只缩放 x/y：

```python
vec[0] = cfg.alpha_xy
vec[1] = cfg.alpha_xy
```

新模型逻辑相同，只是 `action_dims` 来自配置：

```python
alpha = torch.ones(1, 1, cfg.action_dims)
alpha[..., 0:2] = cfg.alpha_xy
```

二者都是：

```python
output = planner_output * alpha_task
```

区别是新模型输出多一个 Agent 维度：

```text
OpenTrackVLA:
(B, N, D_action)

MultiAgentOpenTrackVLA:
(B, 2, N, D_action)
```

## 14. attention mask 构造对比

原始模型：

```python
attn = torch.cat([
    txt_mask,
    ones(B, vis_c + vis_f + ACT)
], dim=1)
```

新模型：

```python
attn = torch.cat([
    txt_mask,
    ones(B, sum(lengths[1:]))
], dim=1)
```

逻辑相同：

- 文本部分使用 tokenizer 的 attention mask。
- 视觉、bbox、query token 全部是有效 token。

新模型只是在非文本部分包含更多 token：

```text
A1视觉 + A1 bbox + ACT1 + A2视觉 + A2 bbox + ACT2 + GND
```

## 15. 输入长度和显存影响

假设：

```text
history = 31
coarse = 4 tokens/frame -> 124 tokens
fine = 64 tokens/current
```

原始模型单 Agent 视觉 token 约：

```text
124 + 64 = 188 个视觉 token
```

加上每帧 marker：

```text
31 个历史 marker + 1 个当前 marker = 32
```

总非文本 token 约：

```text
188 + 32 + 1 ACT = 221
```

新模型每个 Agent 约：

```text
188 视觉 token + 32 marker + 1 bbox = 221
```

两个 Agent 加查询：

```text
A1 221 + ACT1 1 + A2 221 + ACT2 1 + GND 1 = 445
```

因此新模型 LLM 序列长度约为原始模型的 2 倍。由于 Transformer attention 复杂度近似随序列长度平方增长，新模型显存和计算量会明显增加。

## 16. 训练代码接口影响

原始训练中，模型输出直接是 Tensor：

```python
tau_pred = model(...)
loss = mse_masked(tau_pred, gt_wp, valid_mask)
```

新训练中，模型输出是 dict：

```python
out = model(...)
pred = out["waypoints"]
loss = masked_mse_multi(pred, gt, valid_mask)
```

标签 shape 也改变：

```text
原始:
gt_wp:      (B, N, 3)
valid_mask: (B, N)

新模型:
gt_wp:      (B, 2, N, 3)
valid_mask: (B, 2, N)
```

这也是为什么不能直接用原始 `train.py` 训练新模型，必须使用 `multi_agent/train_multi_agent.py`。

## 17. 代码流程并排示意

### 原始 `OpenTrackVLA.forward`

```text
输入单 Agent tokens
  │
  ├── proj(coarse)
  ├── proj(fine)
  │
  ├── _interleave_tvi(coarse)
  ├── _interleave_tvi(fine)
  │
  ├── embed_text(instruction)
  │
  ├── concat [text, coarse, fine, ACT]
  │
  ├── LLM
  │
  ├── h_act = last_hidden_state[:, -1]
  │
  └── planner(h_act) -> (B, N, 3)
```

### 新 `MultiAgentOpenTrackVLA.forward`

```text
输入双 Agent stacked tokens
  │
  ├── split -> Agent-1 / Agent-2
  │
  ├── normalize bbox -> (B, 2, 4)
  │
  ├── encode Agent-1 stream:
  │     proj -> add_visual_tvi -> interleave markers -> bbox token
  │
  ├── encode Agent-2 stream:
  │     proj -> add_visual_tvi -> interleave markers -> bbox token
  │
  ├── embed_text(instruction)
  │
  ├── make ACT_1 / ACT_2 / GND
  │
  ├── concat [text, A1_seq, ACT1, A2_seq, ACT2, GND]
  │
  ├── LLM
  │
  ├── h_act1, h_act2, h_gnd = hidden[对应位置]
  │
  ├── planner_agent1(h_act1) -> (B, N, 3)
  ├── planner_agent2(h_act2) -> (B, N, 3)
  └── grounding_head(h_gnd) -> bbox/visible/token logits
```

## 18. 主要相同点总结

二者相同点：

- 都继承 `nn.Module`。
- 都使用 Qwen/Qwen3-0.6B 作为 LLM backbone。
- 都用 tokenizer 得到文本 embedding。
- 都用 `inputs_embeds` 而不是让 LLM 自己 embedding 视觉 token。
- 都用 `CrossModalityProjector` 对齐视觉维度到 LLM hidden size。
- 都用 TVI 思想表达时间/视角/类型信息。
- 都用 `PlannerHead3L` 直接回归未来路点。
- 都支持冻结 LLM。
- 都支持 `use_tanh_actions`。
- 都支持 `alpha_xy` 只缩放 x/y。

## 19. 主要不同点总结

二者不同点：

| 方面 | `OpenTrackVLA` | `MultiAgentOpenTrackVLA` |
| --- | --- | --- |
| Agent 数 | 1 | 2 |
| 输入形状 | `(B,N,C)` | `(B,2,N,C)` |
| bbox | 参数存在但未入序列 | 正式 bbox token |
| kind 类型 | 2 类 | 5 类 |
| agent identity | 无 | `agent_emb` |
| query token | 1 个 ACT | ACT_1、ACT_2、GND |
| planner head | 1 个 | 2 个 |
| grounding head | 无 | 有 |
| 输出类型 | Tensor | dict |
| 输出 shape | `(B,N,3)` | `(B,2,N,3)` |
| hidden state 取法 | 取最后一个 token | 按记录位置取 ACT1/ACT2/GND |
| 训练脚本 | `train.py` | `multi_agent/train_multi_agent.py` |

## 20. 适用场景差异

`OpenTrackVLA` 适合：

- 单机器人目标跟踪。
- 单视角输入。
- 只需要输出一个机器人的未来轨迹。

`MultiAgentOpenTrackVLA` 适合：

- 无人机 + 机器狗协同跟踪。
- 两个 Agent 都有自己的视觉观测。
- 两个 Agent 都需要输出各自未来轨迹。
- 希望利用 bbox 显式条件。
- 后续希望扩展 bbox refinement、可见性预测或 token-level grounding。

## 21. 迁移注意事项

如果从旧模型迁移到新模型，需要注意：

1. 数据格式必须变成双 Agent JSONL。
2. vision cache 必须同时覆盖 drone 和 robotdog 两个视角。
3. batch shape 从 `(B,N,C)` 变成 `(B,2,N,C)`。
4. 模型返回值从 Tensor 变成 dict。
5. loss 要读取 `out["waypoints"]`。
6. checkpoint 参数名不同，旧模型权重不能直接 strict load 到新模型。
7. 新模型序列更长，batch size 通常要比旧模型更小。

## 22. 代码级关键差异一句话版

最核心的代码差异可以压缩成下面几行：

```text
OpenTrackVLA:
seq = [text, single_agent_visual, ACT]
h = hidden[-1]
pred = planner(h)

MultiAgentOpenTrackVLA:
seq = [text, agent1_visual+bbox, ACT1, agent2_visual+bbox, ACT2, GND]
h1, h2, hg = hidden[act1_pos], hidden[act2_pos], hidden[gnd_pos]
pred1 = planner_agent1(h1)
pred2 = planner_agent2(h2)
grounding = grounding_head(hg)
```

