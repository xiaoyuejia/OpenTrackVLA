# USS 对当前双 Agent OpenTrackVLA 的改进启示

## 论文信息

- 论文：USS: Unified Spatial-Semantic Prompts for Embodied Visual Tracking with Latent Dynamics Learning
- arXiv：2606.25880v1，2026-06-24
- 官方页面：https://arxiv.org/abs/2606.25880
- 项目页：https://arescheah.github.io/uss-project-page/
- 本地 PDF：`/data/hdt/newtrackvla/USS_Unified_Spatial_Semantic_Prompts_2606.25880.pdf`
- 官方代码：截至 2026-06-30，项目页仍标记为 `Pending`

## 1. USS 的核心方法

USS 不依赖一个很大的 MLLM 来隐式完成所有事情，而是将目标指定、时序记忆、
动作预测和动态学习拆成四个清晰模块。

### 1.1 统一空间-语义提示

模型支持四种目标提示：

- text：语言描述；
- box：首帧目标框；
- point：首帧目标点；
- mask：首帧目标分割。

Box/point 不是简单地将 4 个坐标映射成一个 token，而是在当前视觉特征网格上做
RoIAlign，再投影成多个目标 prompt token。Mask 则作为首帧 target anchor 持久保存
在 temporal memory 中。

### 1.2 Prompt-conditioned read-write-read fusion

每个视角使用少量 learnable queries。Prompt/query token：

1. 从稠密视觉 token 中读取目标证据；
2. 将目标条件信息写回视觉 token，强化目标、抑制干扰物；
3. 再读取压缩后的目标状态。

每个视角最终只保留 `Kq=10` 个稀疏 target-conditioned token，用于 waypoint 和
visibility 预测。

### 1.3 动作条件 latent world model

训练时将当前稀疏状态 `S_t` 和模型自己预测的 waypoint `a_t` 拼接，预测下一时刻
稀疏状态：

```text
G_t = MLP_a([S_t, repeat(a_t)])
S_hat_{t+1} = F_phi(G_t)
```

下一帧 target 来自 EMA teacher encoder，并使用 LayerNorm 后的 SmoothL1 对齐。
该分支只在训练时存在，推理时删除，因此不增加闭环推理时间。

### 1.4 直接 waypoint + visibility

USS 使用直接 waypoint regression，而不是 diffusion。消融中：

- Full USS：SR 83.6，TR 81.5；
- 无 world model：SR 80.4，TR 79.2；
- 无 temporal memory：SR 72.2，TR 73.3；
- Flow matching：SR 78.8；
- DDIM：SR 75.8。

这说明 temporal memory 的收益最大，world model 提供额外稳定增益，而直接回归比
较慢的生成式 action head 更适合低延迟闭环。

### 1.5 DAgger 闭环再采样

USS 在监督训练后运行模型闭环 rollout，再由 teacher planner 对模型访问到的状态
重新标注。这个阶段专门处理纯 imitation learning 的 covariate shift。

## 2. 与当前模型的直接对照

| 问题 | 当前实现 | USS 思路 |
|---|---|---|
| 目标指定 | base 只使用文本和 role marker | 首帧 text/box/point/mask 统一 prompt |
| bbox 使用 | base 完全关闭；full 仅将 4D bbox 编为一个 token | 在视觉网格上 RoIAlign 得到目标外观 token |
| 目标融合 | 将文本、两个视觉流直接 concat 给冻结 Qwen | prompt/query 与视觉做 read-write-read |
| 历史 | 31 帧粗 token；实时评估中大量重复稀疏帧 | 16 帧视觉 memory + cross-attention |
| 动态学习 | 只监督 waypoint | action-conditioned latent next-state loss |
| 遮挡 | base visibility 固定返回 0.5 | 训练 view-wise presence，遮挡时复用旧轨迹 |
| 闭环分布偏移 | 只训练 teacher 数据 | 监督训练后追加 DAgger |
| 推理控制 | 约 0.8 秒保持一条速度命令 | 高频直接 waypoint，旧轨迹可短时接管 |

当前代码已经具备一些可复用基础：

- `model.py:437` 的 time/agent/view/kind embedding；
- `model.py:537` 的 bbox token；
- `model.py:564` 的 grounding/visibility head；
- `model.py:798` 的两个独立 planner；
- `train.py:2124` 已能读取 visibility 标签；
- `train.py:2135` 已能构造 target relative pose。

但当前 simple/base checkpoint 关闭了 bbox、grounding、visibility 和 relative-pose
监督，所以评估里的 `visible_score` 恒为 0.5、bbox IoU 恒为 0，这是预期占位输出，
不是有效目标跟踪能力。

## 3. 当前失败与 USS 的对应关系

### 3.1 模型学成“恒定巡航”

对当前 base 部分评估的 274 步统计：

- drone 从未后退，63.1% 的命令达到最大前向速度；
- robotdog 从未后退，87.2% 的命令达到最大速度；
- drone 仅 47.4% 帧仍能看到目标；
- 4 个 episode 全部 Lost。

训练集随机抽样 10000 条：

- drone：87.42% 前进，7.88% 后退，尾部持续停止仅 2.38%；
- robotdog：90.08% 前进，5.83% 后退，尾部持续停止仅 1.32%。

当前 `stop_sample_weight` 只对已经存在的稀少停止窗口加权，不能创建模型闭环偏离后
所需的“急停、后退、重新转向”状态。USS 的 DAgger 对这一问题最直接。

### 3.2 实时历史与训练历史不一致

训练输入是连续 10 Hz 的 31 帧历史。当前实时闭环的观测间隔中位数约 0.809 秒，
31 个时间槽内实际只有约 4 张独立图像，其余是重复 token。模型执行推理时，仿真还在
继续执行上一条高速动作。

USS 的 memory bank 不能自动解决采样缺失，但它给出正确方向：memory 应保存真实控制
时刻的状态，并显式编码真实 `delta_t`，而不是将稀疏观测伪装成密集 0.1 秒帧。

### 3.3 文本不足以稳定锁定同一个人

`follow the person` 只能描述类别，不能指定实例。双 Agent 还需要确认 drone 和 dog
持续追踪同一物理目标。首帧 box/mask target anchor 比重复增加文本 marker 更直接。

## 4. 推荐的实现路线

### P0：先修闭环数据和控制，不改大模型结构

1. 收集 model-driven rollout。
2. 使用现有 teacher/目标真值重新生成纠正 waypoint。
3. 强制覆盖以下状态：
   - 距离过近，需要停止或后退；
   - 距离过远，需要追赶；
   - 人突然转弯；
   - 短时遮挡；
   - drone 或 dog 单独丢失目标；
   - 两个 Agent 位于目标不同侧。
4. 按行为分桶采样，避免 90% batch 都是前进。
5. 推理期间不要一直保持单一速度；保存上一条 waypoint trajectory，由 10 Hz
   low-level follower 逐点执行，模型更新后再替换轨迹。

建议至少记录：

```text
distance_bin: too_close / normal / too_far
motion_bin: reverse / stop / cruise / turn
visibility_bin: visible / occluded
agent_failure: none / drone_lost / dog_lost / both_lost
```

### P1：加入首帧空间 prompt 和 target memory

不要在每帧输入 GT bbox，那会形成 oracle。建议协议：

1. episode 首帧为 drone、dog 各提供一次 target bbox；
2. 将当前 64 个 fine token reshape 为 `8 x 8`；
3. 对 bbox 做 RoIAlign，例如输出 `3 x 3 = 9` 个 token；
4. 加上 prompt-kind、agent、view embedding；
5. 将首帧 target prompt token 存入每个 Agent 的 persistent target memory；
6. 后续只依赖视觉和 memory 更新，不再输入 GT bbox。

建议新模块：

```python
class SpatialPromptEncoder(nn.Module):
    # fine_tokens: (B, 64, C) -> (B, C, 8, 8)
    # bbox: (B, 4), cxcywh_norm
    # output: (B, 9, D)
    ...
```

对于同一目标的双 Agent，可进一步加入：

```text
drone_target_queries (B, Kq, D)
dog_target_queries   (B, Kq, D)
        -> cross-agent target fusion
shared_target_state  (B, Kq, D)
        -> planner_agent1 / planner_agent2
```

这样 shared base 的协同信息发生在紧凑 target state 上，而不是让冻结 Qwen 从几百个
视觉 token 中自行找出对应关系。

### P2：训练 visibility head 和遮挡恢复

当前数据已有 `target_visible`。建议：

- 启用真正的 per-agent visibility head；
- `lambda_visible` 可从论文的 `0.05` 开始；
- visibility 低时不继续执行新的高速前进预测；
- 短时遮挡时沿上一条 world-frame trajectory 继续一个有限 horizon；
- 超过 horizon 后减速到 0，而不是无限保持旧速度。

这能直接避免目标已经离开画面时仍持续最大速度。

### P3：加入 action-conditioned latent dynamics

先让 Dataset 返回同 episode 的下一条样本：

```text
current: coarse/fine tokens, waypoint target
next:    next coarse/fine tokens
```

模型 forward 暴露每个 Agent 的稀疏 target state：

```text
S_t: (B, 2, Kq, D)
W_t: (B, 2, 10, 3)
```

世界模型可先采用轻量版本：

```text
action_emb = MLP(flatten(W_t))
S_hat_next = DynamicsTransformer(S_t, action_emb)
S_next_ema = EMAEncoder(next_observation)
L_world = SmoothL1(LN(S_hat_next), stopgrad(LN(S_next_ema)))
```

总损失：

```text
L = L_waypoint
  + 0.05 * L_visibility
  + 0.2  * L_world
  + L_agent_balance
```

双 Agent 可增加同目标一致性，但不能强制两个视角 token 完全相同：

```text
L_shared_target = 1 - cosine(project(S_drone), project(S_dog))
```

只在两个视角都可见时启用，并使用 stop-gradient 或对称 projector，避免表示坍塌。

### P4：使用真实时间和相机几何

当前 time embedding 是离散帧编号。建议额外输入连续时间：

```text
delta_t -> Fourier features -> Linear(D)
```

训练时随机进行 temporal stride/dropout：

```text
stride in {1, 2, 4, 8}
delta_t in {0.1, 0.2, 0.4, 0.8}
```

这比在评估中重复帧更接近真实部署。

无人机和机器狗相机高度、俯仰和运动学差异很大，可参考 USS 的 camera-aware 3D
position encoding，为每个 fine token 加 ray/intrinsic/extrinsic embedding。现有
`agent_emb/view_emb` 只能表示“来自谁”，不能表示该像素对应的空间射线。

## 5. 建议实验矩阵

所有实验保持相同数据 split、相同 evaluator 和相同动力学尺度。

| 实验 | Prompt | Memory | Visibility | World model | DAgger |
|---|---|---:|---:|---:|---:|
| A0 当前 base | text | 31-frame repeat | No | No | No |
| A1 时间对齐 | text | real-dt memory | No | No | No |
| A2 闭环再采样 | text | real-dt memory | No | No | Yes |
| A3 USS-prompt | first-frame bbox | target memory | Yes | No | Yes |
| A4 USS-dynamics | first-frame bbox | target memory | Yes | Yes | Yes |
| A5 协同 target state | bbox per view | cross-agent memory | Yes | Yes | Yes |

必须额外汇报：

- forward/reverse/stop/turn action confusion；
- speed/yaw clipping rate；
- target-visible rate；
- too-close、normal、too-far 三个距离区间的动作正确率；
- drone-only lost、dog-only lost、joint lost；
- control period 和 unique history frames；
- 非 oracle 与 oracle-heading 指标分开报告。

## 6. 优先级结论

对当前项目的推荐顺序：

1. `DAgger + 行为均衡采样`；
2. `真实 delta_t memory + trajectory follower`；
3. `visibility head + 遮挡时旧轨迹/减速策略`；
4. `首帧 bbox RoIAlign prompt + persistent target memory`；
5. `action-conditioned latent world model`；
6. `camera-aware geometry + cross-agent target-state consistency`。

不要优先把 base 换成 diffusion。USS 的消融反而支持继续使用直接 waypoint head，
先解决目标 grounding、历史采样、闭环分布偏移和低层执行。
