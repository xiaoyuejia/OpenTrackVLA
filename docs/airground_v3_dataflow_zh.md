# AirGround-Coop V3 数据流详解（基于代码事实）

> 本文档严格依据仓库当前源码，逐层描述 `AirGroundCooperativeVLAV3` 模型的
> 数据流动，并标注每一处的张量形状、核心名词与因果依赖关系。
> 主要依据文件：
> - `model_airground_coop_v3.py`
> - `train_airground_coop_v3.py`
> - `config/airground_cooperative_tracking_v3.yaml`
> - `docs/airground_coop_v3_method_detailed_zh.md`（方法参考）

---

## 0. 一句话总览

> 无人机与机器狗各自把「共享的 episode 初始目标 reference + 31 帧历史粗粒度 token +
> 当前帧 64 个细粒度 token + YOLO Top-K 人物候选 token」送入**同一个冻结的
> Qwen3-0.6B**，先以一次打包的 **SELF 流**（两条视角隔离的 row 沿 batch 维拼接，
> 每行内 VERIFY 校验查询排在 ACT 动作查询之前）为 Top-8 候选分别输出独立 sigmoid
> 匹配概率，并以单一置信度阈值拒绝弱匹配，随后预测各自的局部 `[x,y,yaw]` 轨迹；随后把**双方视觉、绝对位姿、
> receiver 局部的有向相对位姿、以及从 SELF 流选出的候选上下文**一起送入一次
> **COOP 协同流**，通过带条件记忆的
> **JEPA** 恢复 receiver 可能缺失的当前帧语义、并让每边一个 **K-mode 多模态
> 轨迹解码器**输出 4 条候选恢复轨迹；最后用「YOLO 有效 + LLM target-match 校验」
> 构成的**可见性路由**在 SELF / COOP / BELIEF / SEARCH 之间挑选最终 waypoint，
> 交给 inverse-fixed-dt 控制器转成无人机与机器狗的真实动作。

---

## 1. 关键形状常量

| 符号 | 值 | 含义 |
| --- | --- | --- |
| `B` | batch 大小 | 物理 mini-batch（默认单卡 48） |
| `D` / `d_L` | 1024 | Qwen3-0.6B hidden dim |
| `C_v` | 1536 | DINOv3(384) + SigLIP(1152) 拼接后的视觉维度 |
| `H` | 31 | 历史帧数 |
| `N` | 10 | waypoint 数量 |
| `K` | 8 | Top-K 人物候选框数（`candidate_top_k`） |
| `M` | 4 | 协同轨迹模态数（`num_modes`） |
| `A` | 2 | agent 数（0=drone, 1=robotdog） |

Token 种类（`KIND_*`）：`HISTORY=0, CURRENT=1, DETECTION=2, ACT=3, COOP_ACT=4, POSE=5, TARGET=6, OBSTACLE=7, MISSING=8, VERIFY=9`。

---

## 2. 模型 `forward()` 的输入张量

每个样本同时包含 drone 与 robotdog 的同步数据，逐 agent 维度为 `(B, 2, ...)`。

| 字段 | 形状 | 含义 |
| --- | --- | --- |
| `coarse_tokens` | `(B, 2, 124, 1536)` | 31 帧历史，每帧 2×2=4 个 pooled token |
| `coarse_tidx` | `(B, 2, 124)` | 历史 token 的帧索引 0..30 |
| `fine_tokens` | `(B, 2, 64, 1536)` | 当前帧 8×8=64 个 fine token |
| `fine_tidx` | `(B, 2, 64)` | 当前帧索引 31 |
| `detection_feat` | `(B, 2, 6)` | top-1 检测 `[cx,cy,w,h,score,valid]` |
| `candidate_feat` | `(B, 2, K, 6)` | Top-K 候选框，每框 `[cx,cy,w,h,score,valid]` |
| `candidate_valid` | `(B, 2, K)` | 每框是否有效 |
| `perception_grid` | `(B, 2, 8, 8, 4)` | 每格 `[unknown, free, obstacle, target]` 比例 |
| `agent_poses` | `(B, 2, 4)` | `[x_m, y_m, sinψ, cosψ]` 共享米制坐标 |
| `synthetic_occlusion` | `(B, 2)` | 哪个 agent 被合成遮挡（≤1 个） |
| `coarse_missing_mask` | `(B, 2, 124)` | 哪些历史 token 被 `[MISSING]` 替换 |
| `fine_missing_mask` | `(B, 2, 64)` | 哪些当前 token 被 `[MISSING]` 替换 |
| `instructions` 等 | — | language 指令（tracking / verify / joint） |
| `yaw_hist` / `yaw_curr` | `(B,2,T)` / `(B,2,)` | 历史/当前航向（`use_angle_tvi=false` 不进 TVI） |

**数据来源**：RGB 经 DINOv3 + SigLIP 拼接得到 1536 维特征后，历史帧池化为
4 token、当前帧池化为 64 token；YOLO11m 产生人物候选框与实例 mask 池化成的
8×8×4 场景网格。GT bbox 与 GT target pose 只用于训练期造标签、绝不进入模型输入。

---

## 3. 视觉投影与类型编码（`_encode_agent_streams`）

该方法对 drone / robotdog 各调用一次，产出 `AgentStreamEncoding`。

### 3.1 视觉投影

```
visual_coarse = proj(coarse_tokens)     → (B, 124, 1024)
visual_fine   = proj(fine_tokens)       → (B, 64,  1024)
```

`CrossModalityProjector`（即 `P_v`）= `LayerNorm(1536) → Linear(→1024) → GELU → Linear(→1024)`。

### 3.2 场景网格处理

- 先调用 `_grid_without_detector_target`：把 grid 的 `target` 通道并入 `unknown`，
  并将 `target` 置 0。**原因**：target mask 与候选框来自同一个 YOLO，若保留 target
  通道，VERIFY 头会直接从 YOLO 目标掩码「抄答案」，失去独立校验意义。
- `clean_grid_features = _grid_features(clean_grid)` → `(B, 64, 4)`（把 8×8 grid 对齐到 64 个 fine token）。
- `clean_fine = visual_fine + perception_grid_proj(clean_grid_features)`，用于 **SELF 流**。
- 若当前帧整体缺失（`fine_missing_mask` 全真），`_masked_grid` 把该帧 grid 置为全 unknown；
  否则保留真实背景，得到 `effective_grid`，再投影为 `cooperative_fine`，用于 **COOP 流**。

### 3.3 `[MISSING]` 替换（合成遮挡）

每个 agent 有一个独立的可学习 missing token `masked_visual_tokens[a]`（`1,1,1024`，
`KIND_MISSING`）。在 COOP 流中按 mask 做 token 级替换：

```
cooperative_fine   = where(fine_missing_mask,   [MISSING]^a, cooperative_fine)   (B,64,1024)
cooperative_coarse = where(coarse_missing_mask, [MISSING]^a, visual_coarse)      (B,124,1024)
```

synthetic receiver 的 detection 数值与 candidate ROI 也会被置零（`effective_detection`）。

### 3.4 候选 ROI token（`_candidate_tokens`）

对 Top-K 候选框，不再裁剪单独图像，而是在当前帧 64 个 fine token 上做显式池化：

```
boxes (B*K, 4) → _bbox_grid_mask → masks (B*K, 64)   # 框内 token 置 1，空框取最近 token
pooled = mean over mask(visual_fine)                 (B, K, 1024)
candidate_token = candidate_roi_proj(pooled) + detection_proj(candidate_feat)
                = TVI.make_query(·, KIND_DETECTION, a)   (B, K, 1024)
```

无效框的 candidate token 被 `candidate_valid` 掩码乘 0。该 token 同时携带数值框位置、
置信度和框内 RGB 语义。

### 3.5 TVI 时间/视角/agent/种类编码

`add_visual`：`token + time_emb(tidx) + kind_emb(k) + agent_emb(a) + view_emb(a)`。
`make_query`：`base + kind_emb + agent_emb + view_emb`。

`insert_time_tokens=true` 时，`_interleave_time_markers` 在每个视觉 token 组前插入
一枚显式时间 marker。因此：

```
单 agent 历史序列 = 124 coarse + 31 marker   = 155   (KIND_HISTORY)
单 agent 当前序列 =  64 fine  +  1 marker    =  65   (KIND_CURRENT)
单 agent 候选序列 =  K=8 candidate token     =   8   (KIND_DETECTION)
```

> **重要差异（相对文档 3.5）**：文档中「221」基于旧版「单 detection token」，
> 当前代码已改为 **Top-K=8 候选 token** 并在因果序列中置于 VERIFY/ACT 之前，
> 因此 SELF 单 agent 序列实际长度为 **228 = 155 + 65 + 8**，
> COOP 单 agent 序列实际长度为 **220 = 155 + 65**（COOP 流不含候选段）。

`AgentStreamEncoding` 最终包含：`self_stream(228)`、`self_mask`、
`cooperative_stream(220)`、`cooperative_mask`、`cooperative_fine(64)`、
`teacher_fine(64)`、`target_mask(64)`、`obstacle_tokens(64)`、
`effective_detection(6)`、`candidate_valid(K)`。

其中 `teacher_fine = teacher_proj(fine_tokens)`，来自 EMA 深拷贝投影器
（动量 0.996，stopgrad，不叠加网格），作 JEPA 的 clean teacher。

---

## 4. SELF 双流（`_run_self_flows`，一次 2B forward）

两条 agent 行沿 **batch 维**拼接，而非 sequence 维，因此彼此 **没有任何
attention 连接**——改变机器狗输入不会改变无人机 SELF 轨迹，反之亦然。

### 4.1 序列组装

每行（agent a）序列为：

```
[ text^a ; Z_history^a(155) ; Z_current^a(65) ; candidate_tokens^a(8) ; VERIFY^a(1) ; ACT^a(1) ]
```

- `text` = 角色前缀 + tracking 指令 + target identity 描述 + VERIFY 校验指令。
- 两个 agent 的 `self_stream` 沿 batch 拼成 `streams`（`2B, 228, 1024`）。
- `verify_tokens = TVI(self_verify_tokens[a], KIND_VERIFY, a)`；`act_tokens = TVI(self_act_tokens[a], KIND_ACT, a)`。
- `query_tokens = cat([verify_tokens, act_tokens], dim=1)`（`2B, 2, 1024`），**VERIFY 在前、ACT 在后**。

### 4.2 LLM 前向

```
input_embeds   = cat([text, streams, query_tokens])   (2B, ≤128+228+2, 1024)
attention_mask = cat([text_mask, stream_masks, ones(2)])
hidden = Qwen3-0.6B(inputs_embeds).last_hidden_state  (2B, T, 1024)   # LLM 冻结
```

### 4.3 输出分支

```
verification_hidden[a] = hidden 对应 VERIFY 位置 (B, 1024)
action_hidden[a]        = hidden 对应 ACT 位置    (B, 1024)
candidate_contexts[a]   = hidden 对应候选 token 段 (B, K, 1024)
```

- `self_waypoints[a] = PlannerHead3L(action_hidden[a])` → `(B, 10, 3)`，
  `waypoint[0]` 强制置 `[0,0,0]`（局部原点而非未来点）。
  `PlannerHead3L`：`LN(1024)→Lin(2048)→GELU→Lin(2048)→GELU→Lin(30)→reshape(B,10,3)`，
  `no_tanh_actions=true` 不做 tanh 截断。
- `target_match_logits[a] = target_match_heads[a](verification_hidden[a])` → `(B, 1)`，
  即「YOLO 给出的候选是否是指定跟踪目标」的二分类头。
- `candidate_match_logits[a] = candidate_match_head(candidate_contexts[a])` → `(B, K)`；
  当 `K>1` 时，`candidate_match_logits += target_match_logits[:, None]`，即 VERIFY 提供
  frame 级 accept/reject 偏置，候选 token 提供相对身份排序。

### 4.4 因果依赖（核心设计）

由于 Qwen 使用 causal mask 且 **VERIFY 在 ACT 之前**：

- `h_VERIFY` 看不到后面的 ACT token ⇒ target-match loss 不会通过 ACT 分支回传；
- `h_ACT` 可以看到 VERIFY 及其之前所有证据 ⇒ SELF 动作可利用校验上下文。

这个「VERIFY 在 ACT 前」的排布把 SELF 流从「ACT/VERIFY 各自复制视觉序列」的
4B 行降到 2B 行。

---

## 5. 候选选择（训练 soft / 推理 top-1 gate）

```
masked_candidate_logits = candidate_match_logits.masked_fill(~candidate_valid, -inf)
训练: weights = softmax(masked_candidate_logits)                 (B,2,8)
      selected_candidate_context = Σ_k w · candidate_contexts     (B,2,1024)  # 可微
      selected_detection         = Σ_k w · candidate_feat         (B,2,6)
推理: select_top_candidate(logits, valid, enter=0.50, margin=0.05)
      → 显式 top-1 + 置信度/margin 门控，产出 selected_index/probability/margin/accepted
```

产出 `target_match_probability (B,2)`、`selected_detection (B,2,6)`。

---

## 6. COOP 联合流（`_run_cooperative_flow`，一次 B forward）

每行一整个样本的双视角序列：

```
[ joint_text ;
  Z^D_coop(220) ; Z^G_coop(220) ;                 # 已含 [MISSING] 替换
  selected_candidate_ctx^D(1) ; selected_candidate_ctx^G(1) ;   # 来自 SELF 流
  POSE_D(1) ; POSE_G(1) ;                         # 绝对位姿
  REL_D(1) ; REL_G(1) ;                           # receiver-local 相对位姿
  COOP_ACT_D(1) ; COOP_ACT_G(1) ]
```

### 6.1 位姿 token

- **绝对位姿**：`agent_pose_proj([x/20, y/20, sinψ, cosψ])` → `(B,1,1024)`，`KIND_POSE`。
  训练时施加共享随机 SE(2) 变换，推理时只减去两 agent 中点。投影前除
  `pose_position_scale_m=20`，且不用 LayerNorm（保留度量大小）。
- **有向相对位姿**（`_directed_relative_pose_features`）→ `relative_pose (B,2,5)`，
  `row[:, receiver] = [forward/20, right/20, sinΔyaw, cosΔyaw, dist/20]`，
  即另一个 agent 在 receiver 局部前-右坐标系下的表达，经 `relative_pose_proj` → `(B,1,1024)`。
  该相对 token 对训练 SE(2) 变换保持不变。

### 6.2 LLM 前向与取出

```
hidden = Qwen(inputs_embeds).last_hidden_state   (B, T, 1024)
stream_spans               → 两 agent 视觉 hidden
pose_hidden                (B, 2, 1024)
relative_pose_hidden       (B, 2, 1024)
selected_candidate_hidden  (B, 2, 1024)
coop_contexts[a]           (B, 1024)   # COOP_ACT_a 的 hidden
```

```
cooperative_base_memory = cat([stream_hidden_D, stream_hidden_G,
                               selected_candidate_hidden(2), pose_hidden(2), relative_pose_hidden(2)])
                        = (B, 220+220+2+2+2, 1024) = (B, 446, 1024)
```

> **因果细节**：`COOP_ACT_D` 排在 `COOP_ACT_G` 之前，故 G 可见 D、D 不可见 G，
> 两枚 COOP query 在 causal attention 中不严格对称。

---

## 7. Conditional JEPA（逐 agent）

```
jepa_memory[a] = cat([cooperative_base_memory, coop_contexts[a]], dim=1)  (B, 447, 1024)
prediction_tokens[a] = JEPA[a](jepa_memory[a], cooperative_fine[a])       (B, 64, 1024)
teacher_fine[a]      = teacher_proj(fine_tokens[a])                       (B, 64, 1024)  # EMA/stopgrad
```

`ConditionalJEPAPredictor`：query 由 `query_in(masked_fine) + 可学习 spatial_queries(64)`
构成，经 3 层 8 头 TransformerDecoder 与 memory 做 cross-attention，输出 64 个位置预测。
**JEPA loss 只在 `fine_missing_mask` 置真的位置用 cosine distance 计算**，因此它预测
的是 clean 当前帧 RGB 语义 latent，而非复制 YOLO 标签。

```
pool_weights = softmax(jepa_pool_scores(prediction_tokens))  (B,64)  # 有 mask 时只在遮挡 token 内归一化
pooled = Σ_j w·prediction                                    (B,1024)
target_belief[a] = target_belief_heads[a](pooled)            (B,5)   # [f,r,z,sinΔψ,cosΔψ]
uncertainty[a]   = uncertainty_heads[a](pooled)              (B,1)   # log-var
```

---

## 8. K-mode 协同轨迹解码（逐 agent，`MultimodalTrajectoryDecoder`）

```
decoder_memory[a] = cat([cooperative_base_memory(446), prediction_tokens(64), obstacle_tokens(64)])
                  = (B, 574, 1024)
  obstacle_tokens = obstacle_grid_proj(effective_grid_feat) + obstacle_position(64) + KIND_OBSTACLE  (B,64,1024)
```

`MultimodalTrajectoryDecoder(memory, coop_contexts[a])`：

```
memory_hidden = 1 层 TransformerEncoder(memory)                    (B, 574, 512)
query = mode_queries(K=4) + waypoint_queries(N=10) + context_in(coop_context)  (B,4,10,512)
3 层 8 头 TransformerDecoder → (B,4,10,512)
trajectories = action_out → (B,4,10,3)      # waypoint0 置 0
mode_logits  = mode_out(mean over N) → (B,4)
```

两 agent 堆叠后：

```
cooperative_candidates   (B, 2, 4, 10, 3)
cooperative_mode_logits  (B, 2, 4)
cooperative_waypoints    (B, 2, 10, 3)   # argmax mode 的轨迹
```

---

## 9. 可见性路由（最终 `waypoints`）

```
yolo_visible[a]   = (selected_detection[...,5] > 0.5) & (score ≥ 0.25)
observed_visible[a] = yolo_visible & (target_match_probability ≥ 0.50)
needs_assistance    = ~observed_visible
route_to_cooperative = needs_assistance & observed_visible.flip(1)   # 自己失视且对方可见
both_invisible      = ~observed_visible.any(dim=1)                   # 两边都失视 → BELIEF

routed_waypoints = where(needs_assistance, cooperative_waypoints, self_waypoints)  (B,2,10,3)
```

闭环推理时另由 `AirGroundVisibilityRouter` 做 2 帧滞回（进入/退出双阈值
0.50/0.35、0.35/0.20）稳定可见性状态，并在四状态
SELF / COOPERATIVE / BELIEF / SEARCH 之间路由。

---

## 10. 完整输出字典（形状汇总）

| 输出 | 形状 | 语义 |
| --- | --- | --- |
| `self_waypoints` | `(B,2,10,3)` | 两条 clean SELF 轨迹 |
| `cooperative_candidates` | `(B,2,4,10,3)` | 每个 agent 的 4 条候选 |
| `cooperative_mode_logits` | `(B,2,4)` | 各模态 logit |
| `cooperative_waypoints` | `(B,2,10,3)` | argmax mode 轨迹 |
| `waypoints` | `(B,2,10,3)` | 路由后最终轨迹 |
| `target_match_logits` / `target_match_probability` | `(B,2)` | VERIFY 匹配 logit / 概率 |
| `candidate_match_logits` | `(B,2,8)` | 每候选匹配 logit |
| `candidate_selected_index` | `(B,2)` | 选中候选索引 |
| `target_belief` | `(B,2,5)` | receiver-local 目标 belief |
| `jepa_uncertainty_logit` | `(B,2)` | belief log-var |
| `jepa_prediction_tokens` / `jepa_teacher_tokens` | `(B,2,64,1024)` | JEPA 预测 / EMA teacher latent |
| `jepa_token_mask` | `(B,2,64)` | JEPA 有效区域 |
| `agent_poses` | `(B,2,4)` | 绝对位姿 |
| `directed_relative_pose` | `(B,2,5)` | 有向相对位姿 |
| `coarse_missing_mask` / `fine_missing_mask` | `(B,2,124)` / `(B,2,64)` | 遮挡 mask |
| `routing_mode` | `(B,2)` | SELF/COOP/BELIEF |
| `self_action_context` / `target_verify_context` | `(B,2,1024)` | ACT / VERIFY 的 hidden |

---

## 11. 整体流向图（ASCII）

```
                        DINOv3 + SigLIP (1536)  /  YOLO11m  /  Unreal pose
                                    │
      coarse(2,124,1536)  fine(2,64,1536)  grid(2,8,8,4)  candidate(2,K,6)  pose(2,4)
                                    │ P_v / grid_proj / TVI / [MISSING]
                                    ▼
   ┌───────────────────────────────┐      ┌────────────────────────────────────┐
   │ SELF 流（2B forward, 隔离）    │      │ COOP 流（B forward, 联合）         │
   │ [text,hist155,curr65,cand8,    │      │ [joint_text, D_coop220, G_coop220, │
   │  VERIFY, ACT] ×2 沿 batch 拼   │      │ 选后候选ctx, POSE_D/G, REL_D/G,    │
   │                               │      │  COOP_ACT_D, COOP_ACT_G]           │
   │  ◆ action_hidden→PlannerHead  │      └───────────────┬────────────────────┘
   │    └▶ self_waypoints(2,10,3)  │                      │ base_memory(446,1024)
   │  ◆ verify_hidden→verify_head  │                      ├─◆ JEPA → pred(64) + belief(5) + unc(1)
   │    └▶ target_match_prob(2)    │                      └─◆ coop_context → K-mode decoder
   │  ◆ candidate_ctx→match_head   │                            └▶ coop_waypoints(2,10,3)
   │    └▶ candidate_logits(2,8)   │
   └───────────────────────────────┘
                waypoints = route(observed_visible, self, coop)
                                    ▼
                 routed_waypoints(B,2,10,3) → inverse-fixed-dt 控制
```

---

## 12. 需要牢记的实现边界

1. 视觉前端是 **DINOv3 + SigLIP**，不是 DINOv2。
2. SELF-D / SELF-G 上下文严格隔离但权重共享，且打包为一次 2B forward。
3. VERIFY 是「YOLO 候选是否是指定目标」，不是第二个可见性检测器。
4. GT bbox 与 GT target pose 只在训练期造标签，从不进入推理输入。
5. 合成遮挡只作用于 COOP 流的 receiver row，两条 SELF row 始终 clean。
6. ROI_ONLY 不允许 pose relocation（其背景仍对应 clean receiver pose）。
7. **候选是 Top-K=8**（当前实现），SELF 单 agent 序列为 228，相对文档旧值 221 已变化。

---

## 12.1 当前联合训练合同（2026-09 修订）

- STT、DT、AT 使用**同一个模型、同一个 CandidateTextMatcher、同一个 balanced binary
  candidate loss 和同一组 waypoint heads**；`task_type` 仅用于数据审计，不选择网络分支。
- 三类任务都保留 JSONL 原始 `instruction`：STT 是普通目标规则，DT 是外观描述，
  AT 是对 episode 初始目标的指代描述。角色动作 prompt 不覆盖目标 instruction。
- `effective_visible/self_target` 由 Top-K 中任一候选是否匹配 GT 决定，不再由 YOLO
  top-1 决定；最大 IoU 候选为唯一正样本，其余有效候选为负样本，正负 BCE 分组等权。
- `CandidateTextMatcher` 以已经过同一次 Qwen causal attention 的 `VERIFY hidden`
  查询所有候选 hidden；训练使用 hard-forward/soft-backward 选择，不再调用第二次 Qwen。
- 选中候选 context 经零初始化残差显式条件化 SELF ACT，并进入 COOP；旧 checkpoint
  初始 waypoint 行为因此保持不变。
- STT/DT/AT 共享一枚 episode-start 视觉 reference token 作为时序身份证据。step 0
  不提供 reference，后续训练样本读取初始目标 ROI，并统一做 20% reference dropout，防止
  模型绕过 instruction。在线端连续 3 帧高置信一致后确认，确认后冻结以防漂移，连续拒绝
  后释放并回到 instruction grounding，避免一次误选永久锁死。

以下第 13--17 节保留为**修复前审计与设计推导**；其中“当前没有显式匹配/描述固定”等
结论已经由上述实现取代，不应再视为当前代码状态。

## 13. 修复前 Top-K 候选处理审计（历史）

### 13.1 完整流程（单 agent 视角，`candidate_top_k=8`）

```
  输入（已拆 agent 轴）：
    visual_fine (B,64,1024)   candidate_feat (B,K=8,6)   candidate_valid (B,K=8)

  ① _candidate_tokens：一框一 token
    candidate_feat[...,:4] ─▶ _bbox_grid_mask(boxes,valid,64,expand=1.0) ─▶ masks (B*K,64)
    visual_fine ─掩码均值池化─▶ pooled = Σ(mask·visual_fine)/Σmask  (B,K,1024)  # ROI 视觉
    candidate_token = candidate_roi_proj(pooled) + detection_proj(candidate_feat)
                    = TVI.make_query(·, KIND_DETECTION, a)
    candidate_tokens = candidate_token × candidate_valid          # 无效候选整 token 置 0

  ② 放进 SELF 序列（候选段在 current 之后、VERIFY/ACT 之前）
    [ text(≤128) | hist(155) | curr(65) | candidate_1..K(8) | VERIFY(1) | ACT(1) ]
    ── 一次 Qwen3-0.6B（冻结,causal）──▶ hidden (B,T,1024)

  ③ 取候选段 hidden → 打分
    candidate_contexts = hidden[:, text_len+self_stream_len-K : ...]   (B,K,1024)
    candidate_match_logits = candidate_match_head(candidate_contexts)   (B,K)  # LayerNorm→Linear(1024→1)
    candidate_match_logits += target_match_logits[:,None]               # VERIFY 头 frame 级偏置
    # 仅推理时追加时序一致性：
    candidate_match_logits += candidate_temporal_iou_weight(=2.0)
                              × candidate_iou_with_prior(candidate_box, prior_bbox) × prior_valid

  ④ 选择「我的目标」的特征 token（训练 soft / 推理 hard）
    训练: weights=softmax(masked) ; selected_ctx=Σ_k w_k·ctx_k       (B,1024) 可微加权
    推理: select_top_candidate(logits,valid,enter=0.50,margin=0.05)
          selected_ctx = ctx[index] × accepted                        (B,1024) 硬选或置 0

  ⑤ 流向：selected_candidate_context ─▶ COOP 流 ─▶ cooperative_base_memory
            ─▶ JEPA / K-mode 解码器（它不进 SELF planner）

  ⑥ 路由：observed_visible = yolo_valid & (match_prob ≥ 0.50)
          routed_waypoints = observed ? self : coop
```

### 13.2 关键代码事实

- 候选 token 同时携带 **ROI 视觉**（框内 fine token 均值池化）与 **几何**（6 维 `[cx,cy,w,h,score,valid]` 投影），二者相加后加 TVI 编码。
- 打分头是极简 `LayerNorm(1024) → Linear(1024→1)`，对每个候选**独立**出标量，并非显式对比。
- 训练期用 **softmax 加权平均**融合候选 context；推理期用 **top-1 + 门控**（enter=0.50，top1/top2 margin=0.05）硬选。
- `selected_candidate_context` **只进 COOP 流**，不回流 SELF planner；SELF 的 ACT 仅靠 causal attention 隐式看到候选。
- 推理专属的 `candidate_temporal_iou_weight`（默认 2.0）把「候选框 vs 上一帧选中框的 IoU」作为时序一致性加分，**训练期没有**。

### 13.3 设计问题（详见第 14 节）

1. 匹配头是逐候选独立打分，缺显式「候选 × 目标描述」交叉（`candidate_matching.py` 里的 `CandidateTextMatcher` 已实现但未被主流程调用）。
2. 文本里无真实目标身份描述（`instruction` 字段固定为一句任务指令）。
3. 训练 softmax 加权 vs 推理 top-1 硬选，存在 train/test 分布漂移。
4. ROI 用「框内均值池化」，多人外观区分能力弱。
5. margin 门控（0.05）在训练中无监督。


## 14. 修复前 prompt 覆盖与拼接审计（历史）

### 14.1 数据层

`batch["instruction"]` 来自 JSONL 的 `instruction` 字段（`train_airground_v3_common.py:987`）：

```python
"instruction": ex.get("instruction", "Follow the target person without collision.")
```

实际数据（`dataset.json` 200 条）中该字段**全部是同一固定字符串**
`"Follow the target person without collision."`，数据条目里**没有任何**身份/appearance/description 字段。

### 14.2 config 层四个 override 字段

```yaml
instruction_override:          Follow the person without collision.
agent1_instruction_override:   Aerial drone independently follows the person without collision.
agent2_instruction_override:   Ground robot dog independently follows the person without collision.
joint_instruction_override:    Use both aerial and ground observations to follow the person without collision.
```

`apply_airground_v3_defaults` 还显式关掉了旧 prompt/ROI/bbox 文本开关（`train_airground_coop_v3.py:1620-1631`）：

```python
cfg.use_roi_tokens = False
cfg.use_bbox_tokens = False
cfg.use_bbox_text_prompt = False      # bbox 文本 prompt 被显式关闭
cfg.use_grounding = False
cfg.use_visual_section_markers = False
cfg.beta_bbox = 0.0
cfg.beta_visible = 0.0
...
cfg.bbox_text_dropout_prob = 0.0
```

### 14.3 loss 入口层组装（`forward_airground_v3_loss`）

```python
instructions      = batch["instruction"]                       # 数据原始指令，未被 override
joint             = [joint_instruction_override or instruction_override] * B
if not (joint_instruction_override or instruction_override):
    joint = instructions
drone_instructions = [agent1_instruction_override]*B if ... else None
dog_instructions   = [agent2_instruction_override]*B if ... else None
```

四个文本角色中，只有 `instructions`（SELF 流的 identity 文本）保留数据原始值，
`joint/drone/dog` 三条全部被 config 固定文本替换。

### 14.4 模型内拼接（`_run_self_flows` / `forward`）

SELF 流每 agent 一行：

```python
_role_text(DRONE)    = "Aerial drone: independently follow the target person."
_role_text(ROBOTDOG) = "Ground robot dog: independently follow the target person."
fallback = f"{_role_text(a)} {instruction}" if use_agent_text_markers else instruction

text = (f"Tracking task: {action_text} "
        f"Target identity description: {identity_text} "
        f"Candidate verification task: {verification_prompt}")
```

- `action_text` ← `drone/dog_instructions`（被 override 的固定角色文本）；
- `identity_text` ← `instructions`（即 "Follow the target person without collision."）；
- `verification_prompt` ← `role_prefix + drone/dog_target_verification_prompt`。

joint 流（`forward` 内）额外拼接：

```python
joint_instructions = [f"{joint_text} Target identity description: {identity_text}"
                      for joint_text, identity_text in zip(joint_instructions, instructions)]
```

### 14.5 结论

| 文本角色 | 最终内容来源 | 是否含真实身份描述 |
| --- | --- | --- |
| `instructions`（SELF 的 `identity_text`） | 数据 `instruction` 字段（固定单句） | ❌ 无 |
| `drone/dog_instructions`（`action_text`） | config override | ❌ 无 |
| `joint_instructions` | config override + 拼 `Target identity description: {instructions}` | ❌ 无 |
| `verification_prompt` | config `*_target_verification_prompt` | 描述「判断候选」任务，非身份 |

代码写死了 `"Target identity description: {identity_text}"` 标签，但 `identity_text`
塞进去的其实是任务指令本身，**不是任何外观/身份描述**。训练期「目标身份」在文本层面缺失，
模型区分 DT/AT 多人目标只能依赖：①候选 ROI 外观、②历史一致性、③仅推理期的 prior bbox 时序 IoU。


## 15. 对比选择机制的设计推导（现已按最小方案落地）

### 15.1 现状：当前并没有这套显式机制

- 候选打分是「各打各的独立标量」（`candidate_match_head` 逐候选出分），非对比排序；
- 融合是「softmax 加权平均」，非硬选择；
- waypoint 由分离的 planner/decoder 头从 **query hidden** 读出，非「选中候选」编码；
- prompt 仍是旧单候选措辞（"proposes **one** person candidate … **binary** match decision"），未说「Top-K 挑最像 / 选中者驱动轨迹」。

### 15.2 理想机制（检索式 grounding + 硬选择）

> 给出 K 个候选，Qwen 根据任务描述给每个候选打分（最符合者最高、其余压低），
> 只取**最高分候选的特征编码**作为后续 waypoint 预测的条件，其余候选丢弃/压到低分。

这是多目标跟踪的标配，正规名称：

- 对比/排序式候选打分（softmax 交叉熵监督 top-1）；
- 硬选择 + 直通梯度（hard selection + straight-through）；
- query-conditioned decoding（选中候选 token 作为 decoder 的 memory/condition）。

### 15.3 具体改动点（均在一次 LLM 前向内完成）

方案 A（最小改动）：

1. **改 prompt** 为显式对比选择：「YOLO proposes up to 8 candidates; assign the HIGHEST
   match score to the best-matching candidate, keep others LOW; the selected candidate's
   encoding drives subsequent action/waypoint decoding.」
2. **打分改对比排序**：用 `log_softmax(candidate_match_logits)` 作分布，GT 正候选（IoU≥0.3）
   作 one-hot 标签，训练 NLL loss——显式「给最符合者高分、其余低分」。
3. **融合改硬选 + 直通梯度**：`selected_ctx = candidate_contexts[argmax(logits)]`，梯度经
   soft 权重直通回 logits（Gumbel-Softmax / straight-through），而非 `Σ softmax·ctx` 加权。
4. **选中者编码直接驱动下游**：`planner_input = cat([action_hidden, selected_candidate_context])`
   （或作为 COOP decoder 的额外条件），使 selected 特征真正条件化 waypoint。

方案 B（更彻底，仍在一次 LLM 内）：

- 加可学习 `[TARGET]` query，对 K 个候选 token 做 soft attention（`Q=[TARGET], K/V=candidate_tokens`），
  得到目标聚合表征，直接作为 planner/decoder 的条件输入，并配 softmax 排序 + margin 监督。

这些机制都发生在 LLM 前向**之前**（输入拼接）或**之后**（head 层说明），候选 token 已
在同一次 SELF LLM 前向里被 causal attention 处理过，`candidate_contexts` 已是「读过 text/视觉」
的 hidden，只需改「怎么挑选、怎么喂给下游头」，无需二次过大模型。


## 16. query token（ACT/VERIFY/COOP_ACT）的本质：大模型在做什么

### 16.1 代码事实

```python
self.self_act_tokens    = nn.Parameter(torch.zeros(1,1,1024))  # 每 agent 一个，初始 0
self.self_verify_tokens = nn.Parameter(torch.zeros(1,1,1024))
self.coop_act_tokens    = nn.Parameter(torch.zeros(1,1,1024))
# 送入序列前只经 make_query 加 kind/agent/view 编码，不含任务内容
```

SELF 序列末尾为 `[..., VERIFY(1), ACT(1)]`，`action_hidden = hidden[:, -1]`，
`verification_hidden = hidden[:, -2]`——即这两个空壳 token 经整个 Qwen 前向后，在其位置输出的 1024 维向量。

### 16.2 大模型在这里只做一件事：上下文聚合

Qwen **并没有算出 waypoint**，它做的是把整条异构序列（文本+视觉+检测+位姿）通过
causal self-attention 压缩进 query 那一个位置的表示：

```
h_ACT = Attention(query=ACT空壳, key/value=[text, hist, curr, candidates, ...])
      = 上下文在「ACT 视角」下的加权聚合向量
```

真正解码 waypoint 的是紧随其后的小 MLP：

```python
class PlannerHead3L:  # LN(1024)→Lin(2048)→Lin(2048)→Lin(30)→reshape(B,10,3)
    out = self.net(hidden)       # hidden = action_hidden (B,1024)
    trajectory = out.view(B,10,3)  # 10 个 [x,y,yaw]，waypoint0 锚定为原点
```

### 16.3 分工

| 环节 | 谁在做 | 能力 |
| --- | --- | --- |
| 理解场景/任务，聚合上下文 | Qwen（冻结，attention） | 通用表征/上下文融合 |
| 把上下文映射成轨迹 | PlannerHead3L / K-mode 解码器（可训练） | 输出头拟合 |

### 16.4 为什么空壳 token 能解码出 waypoint（本质：可学习 query / 软 prompt）

1. query 是「提问槽位」，不是「内容」；初始为 0 无关紧要，训练会把 `self_act_tokens`
   调成「最能从下游上下文里读出轨迹」的最优提问向量。
2. Qwen 的 attention 把这个空壳「填满」：同一段上下文，VERIFY 和 ACT 因**提问方向不同**
   （不同可学习向量 + 不同 KIND embedding）产生不同聚合结果——所以同一条视觉序列既能做目标
   匹配（verify）又能做轨迹预测（act）。
3. MLP 头把聚合向量「翻译」成具体标量，这是纯拟合，不含「理解」。

### 16.5 直觉对比

| 传统做法 | 本模型做法 |
| --- | --- |
| 让 LLM 生成文本，再 parse 成 waypoint | 用空壳 query 在末尾「探」出向量，让 MLP 直接回归 waypoint |
| LLM 负责生成数值 | LLM 负责压缩上下文成好向量，MLP 负责读数值 |

waypoint 从来不是 Qwen 生成的语言，而是从 Qwen 输出的上下文向量里由可训练轨迹头回归出的
连续值。Qwen 的贡献是提供「上下文融合后的条件向量」，而不是「会算轨迹」。


## 17. grounding + decoding 闭环设计推导（当前实现见 12.1）

> 针对 DT/AT 多人任务，核心判断：
> **多人跟踪 = 用大模型 attention 做「符合描述的目标 grounding」+ 用 grounding 出来的
> 目标向量做「未来轨迹 decoding」**。改进的关键不是加更多 head，而是让 grounding 与
> decoding **共享同一个目标表征**，并把「身份信息」与「历史轨迹先验」补进来。

### 17.1 现状盘点：现有「信息」与「能力」的利用情况

**信息（代码已存在，但用得不好）：**

| 信息 | 位置 | 现状利用情况 |
| --- | --- | --- |
| 候选框外观 ROI | `candidate_contexts` (B,K,1024) | 只被独立打分，未做目标级精细比较 |
| 候选框几何 | `candidate_feat[...,:4]` | 只进 `detection_proj` 加性融合 |
| 目标身份描述 | text 里的 `Target identity description` | 空壳，塞的是任务指令 |
| 历史 31 帧 | coarse tokens | 已进 SELF 序列（最强时间一致性信号） |
| 上一帧选中框 | `prior_bbox` / `candidate_temporal_iou_weight` | 仅推理用，训练缺失 |
| 双方位姿/相对位姿 | `agent_poses` / `directed_relative_pose` | 已进 COOP 流 |
| 目标历史轨迹 | 无显式 token | 完全未利用 |

**能力（大模型能做而当前未充分调用）：**

1. attention 本身就是「目标 grounding」，但当前只用 `Linear(1024→1)` 独立打分头读结果，丢了匹配结构；
2. 因果上下文记忆（历史一致性）已具备，但目标匹配与轨迹解码之间未共享「上一帧锁定了谁」；
3. 跨模态条件生成已具备，但未做「描述 vs 候选」的显式交叉。

### 17.2 核心设计原则

> 让「目标定位（grounding）」和「轨迹解码（decoding）」共享同一个「目标表征」，
> 而不是像现在这样 grounding 打一个标量分、decoding 读另一个 query hidden。

即：先让大模型 attention 产出「这个目标是谁」的向量，再让该向量作为条件去解码「未来怎么跟」。

### 17.3 三层改进设计（均在一次 LLM 前向内完成）

**第一层：补齐目标身份描述信息（否则 grounding 无的放矢）**

```python
# 数据层：每个样本显式带上目标身份描述（或用 history 首帧框/外观嵌入构造）
target_identity_description = "the person in a red jacket"

# 模型层：把身份文本/token 段放在候选之前
[ text | identity_desc | hist | curr | candidates | VERIFY | ACT ]
```

若无文本身份，则用可学习 `[TARGET]` identity token，通过历史首帧锁定目标外观记忆——
「身份」不靠人工标注，而靠历史一致性学到。

**第二层：用 attention 做「描述→候选」软 grounding（替换独立打分头）**

```python
q_target = target_query                              # (B,1024) 可学习 [TARGET] 或 identity 文本 pooled
weights  = softmax(q_target · candidate_contexts / √D)  # (B,K) attention 权重
target_embedding = Σ_k weights_k · candidate_contexts_k  # (B,1024) 目标 grounding 向量
candidate_match_logits = q_target · candidate_contexts   # attention 打分，天然对比
```

「谁最符合描述」直接由 attention 权重表达，`target_embedding` 即下一步解码条件——
grounding 与 decoding 共享同一 attention 产物。

**第三层：grounding 结果显式条件化轨迹解码（信息闭环）**

```python
# SELF planner 以选中目标为条件
planner_input = cat([action_hidden, target_embedding])         # (B, 2048)
self_waypoints = PlannerHead3L(planner_input)

# COOP decoder 注入目标表征
query = context_in(coop_contexts[a]) + target_proj(target_embedding)

# 复用上一帧选中目标的 waypoint 作为时序先验（从 bbox IoU 升级为轨迹 token）
target_traj_token = trajectory_encoder(prev_selected_waypoint)
planner_input = cat([action_hidden, target_embedding, target_traj_token])
```

### 17.4 一次 LLM 前向内的完整闭环（目标图）

```
              ┌──────────────────────────────────────────────────────┐
              │  一次 Qwen SELF 前向（冻结）                          │
              │  [ text | identity | hist | curr | candidates |       │
              │                    VERIFY | ACT ]                    │
              └───────────────────┬──────────────────────────────────┘
                                  │ candidate_contexts (B,K,1024)
                                  ▼
        ┌──────── 目标 grounding（attention 软选择）────────────────┐
        │  q_target · candidate_contexts  →  weights (B,K)          │
        │  target_embedding = Σ w·ctx        (B,1024)               │
        │  match_logits = q_target·ctx        (B,K)                 │
        └───────────────────────┬────────────────────────────────────┘
                                │ target_embedding
                                ▼
        ┌──────── 轨迹解码（以目标向量为条件）───────────────────────┐
        │  planner_input = [action_hidden; target_embedding]        │
        │    (+ target_traj_token 上一帧轨迹)                        │
        │  self_waypoints (B,10,3)                                   │
        │  COOP decoder 条件化 + K-mode                               │
        └─────────────────────────────────────────────────────────────┘
```

### 17.5 配套损失（让「能力」真正被监督）

1. **对比/排序 loss（对 grounding）**：`NLL(softmax(match_logits), onehot(正候选))`，
   显式训练 attention 给最符合者高分。
2. **目标维持 loss**：跨帧让 `target_embedding` 对同一目标的时序一致性最大化（连续帧应锁定同一人）。
3. **margin loss**：top1/top2 间隔监督，支撑推理门控 0.05。

### 17.6 优先级总结

按优先级补齐三项：

1. **目标身份信息**（文本或历史锁定的可学习 identity）—— 没有它，attention 无从「符合描述」；
2. **attention 软 grounding 替代独立打分头** —— 让「谁最符合」由 attention 权重显式表达；
3. **grounding 向量条件化轨迹解码**（+ 历史轨迹 token）—— 让「定位到的人」直接决定「未来怎么跟」。
