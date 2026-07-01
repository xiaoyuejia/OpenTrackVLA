# UnrealZoo Anchor-based Diffusion Action Model 修改说明

训练执行过程、实际训练结果和模型学习本质详见：

```text
docs/ANCHOR_DIFFUSION_TRAINING_PROCESS_ZH.md
```

## 如何跑训练

新扩散模型位于 `model_unrealzoo_anchor_diffusion.py`，原来的 `train.py --multi_agent` 固定导入
`model.py`，不能直接训练新扩散头。现在使用独立入口：

```bash
cd /data/hdt/newtrackvla

# 首次使用扩散版本时安装外部依赖。
/home/hdt/miniconda3/envs/omtracknew/bin/pip install -r docs/requirements-unrealzoo-anchor-diffusion.txt

# 第一次运行：生成 drone/robotdog 两套锚点，并检查数据/cache shape。
bash sh/train_anchor_diffusion.sh

# dry-run 正常后开始正式训练。
RUN_BUILD_ANCHORS=0 RUN_DRY_RUN=0 RUN_TRAIN=1 \
  bash sh/train_anchor_diffusion.sh
```

当前默认读取：

- 训练标签：`data/unrealzoo_aerial_ground_human_multi/dataset.json`
- 视觉缓存：`data/unrealzoo_aerial_ground_human_multi/vision_cache`
- 锚点输出：`data/unrealzoo_aerial_ground_human_multi/trajectory_anchors`
- checkpoint：`ckpt/ckpts_multi_agent_anchor_diffusion`

调试小模型时可以减少锚点数量、DiT 深度和训练样本：

```bash
NUM_ANCHORS=8 ANCHOR_MAX_SAMPLES=200 DIFFUSION_DEPTH=2 \
RUN_BUILD_ANCHORS=1 RUN_DRY_RUN=1 RUN_TRAIN=0 \
  bash sh/train_anchor_diffusion.sh
```

单卡正式训练的常用参数示例：

```bash
BATCH_SIZE=2 GRAD_ACCUM_STEPS=8 EPOCHS=1 LR=2e-5 \
BETA_NAV=1.0 DIFFUSION_SCORE_LOSS_WEIGHT=100 \
RUN_BUILD_ANCHORS=0 RUN_DRY_RUN=0 RUN_TRAIN=1 \
  bash sh/train_anchor_diffusion.sh
```

这里 `BETA_NAV` 推荐从 `1.0` 开始，因为扩散头的 `action_loss` 内部已经包含：

```text
两个 Agent 的 [最近锚轨迹回归损失 + DIFFUSION_SCORE_LOSS_WEIGHT * 候选评分 BCE]
```

论文 Table 5 使用 `lambda=100`、`M=40`。当前稳定默认配置先对全部候选的
BCE 取均值，再乘 `lambda=100`，因此双 Agent 模型在 logits 全为 0 的训练初期，
`loss_nav` 接近 `2 * log(2) * 100 = 138.6`。可设置
`DIFFUSION_SCORE_LOSS_REDUCTION=sum` 恢复公式字面求和，用于消融实验。

训练入口对应关系：

- `tools/build_unrealzoo_trajectory_anchors.py`：读取 `waypoints/valid_mask`，分别聚类两个 Agent 的轨迹。
- `train_unrealzoo_anchor_diffusion.py`：标准导入 `model_unrealzoo_anchor_diffusion.py`，把 GT 路点传给扩散头并使用其 `action_loss`。
- `sh/train_anchor_diffusion.sh`：集中管理锚点生成、dry-run、单卡/多卡训练参数。
- 原 `model.py` 和 `train.py --multi_agent` 的旧 MLP 训练流程保持不变。

## 1. 修改范围

核心模型实现只修改：

```text
/data/hdt/newtrackvla/model_unrealzoo_anchor_diffusion.py
```

另外新增 `tools/build_unrealzoo_trajectory_anchors.py`、`train_unrealzoo_anchor_diffusion.py` 和
`sh/train_anchor_diffusion.sh`，用于生成锚点并接入现有双 Agent 训练循环。

没有替换当前正式运行使用的 `model.py`。扩散训练入口通过
`import model_unrealzoo_anchor_diffusion` 导入独立的新模型实现。

实现目标是将论文 `2505.23189v1.pdf` 中的 Anchor-based Diffusion Action Model
接入现有 `h_act` 规划条件，同时参考 `/data/hdt/DiffusionDrive/` 的截断扩散实现。

默认 `use_anchor_diffusion=False`，因此原有 `PlannerHead3L` 和旧 checkpoint 行为保持
不变。只有显式开启后，模型才使用扩散动作头。

## 2. 已实现模块

### 2.1 轨迹锚点生成与加载

新增：

```python
fit_trajectory_anchors_kmeans(...)
save_trajectory_anchors(...)
_load_trajectory_anchors(...)
```

`fit_trajectory_anchors_kmeans()` 是不依赖 `scikit-learn` 的 K-means++ 实现：

- 输入轨迹 shape：`(S, Nw, D)`。
- 输出锚点 shape：`(M, Nw, D)`。
- 支持 `valid_mask`，partial horizon 的 padding 不参与距离和中心更新。
- 锚点必须与训练标签使用相同的局部坐标系、单位、路点数量和时间语义。

建议单 Agent、无人机、机器狗分别聚类，不要默认共享一套锚点。

### 2.2 基于 diffusers 的截断 DDIM

新增：

```python
TruncatedDDIMScheduler
```

默认参数与 TrackVLA 论文附录一致：

| 参数 | 默认值 |
|---|---:|
| 总扩散步数 | `1000` |
| 训练截断范围 | `[0, 50)` |
| 推理初始加噪 timestep | `10` |
| 推理 DDIM 去噪步数 | `2` |
| prediction type | 直接预测去噪轨迹 `x0` |

实现内部使用与 DiffusionDrive 一致的 `diffusers.schedulers.DDIMScheduler`，
配置为 `beta_schedule="scaled_linear"`、`prediction_type="sample"`。
训练时在锚点附近加入少量高斯噪声；推理时默认使用
`[10, 0]` 两个 timestep 去噪。

### 2.3 Diffusion Transformer

新增：

```python
SinusoidalTimestepEmbedding
AnchorDiTBlock
AnchorDiffusionActionModel
```

数据流：

```text
LLM ACT hidden state h_act: (B, D_llm)
            │
            ├── condition projector
            │
noisy anchors: (B, M, Nw, D_action)
            │
            ├── trajectory embedding
            ├── anchor-id embedding
            ├── waypoint-id embedding
            │
            ▼
M*Nw 个轨迹 token
            │
            ├── DiT self-attention
            ├── AdaLN(h_act + timestep)
            │
            ▼
candidate trajectories: (B, M, Nw, D_action)
candidate logits:       (B, M)
            │
            ▼
top-1 trajectory:       (B, Nw, D_action)
```

当前实现对 `(x, y, theta)` 全部进行扩散。输出和 loss 中对 `theta` 使用 wrapped
angle，避免 `-pi/pi` 边界产生虚假大误差。

### 2.4 最近锚监督与论文式 Tracking Loss

新增：

```python
anchor_diffusion_tracking_loss(...)
```

它实现论文公式：

```text
Ltrack = MSE(最近锚对应的预测轨迹, GT)
       + lambda * BCE(所有锚点分类分数, 最近锚 one-hot 标签)
```

默认 `lambda=100`、候选 BCE reduction 为 `mean`。最近锚分配与轨迹回归均支持
`valid_mask`。

输出包含：

```text
loss
regression_loss
score_loss
nearest_anchor
anchor_distance
```

## 3. 单 Agent 接入

`ModelConfig` 新增扩散配置，核心开关：

```python
ModelConfig(
    use_anchor_diffusion=True,
    diffusion_anchor_path="/path/to/anchors.npy",
    diffusion_num_anchors=40,
    diffusion_hidden_dim=768,
    diffusion_depth=12,
    diffusion_num_heads=12,
    diffusion_num_train_timesteps=1000,
    diffusion_train_truncation_steps=50,
    diffusion_inference_start_timestep=10,
    diffusion_inference_steps=2,
    diffusion_score_loss_weight=100.0,
)
```

开启后：

```text
h_act -> AnchorDiffusionActionModel -> top-1 waypoints
```

默认 forward 仍返回：

```text
(B, Nw, 3)
```

训练或调试时传入：

```python
result = model(
    coarse_tokens,
    coarse_tidx,
    fine_tokens,
    fine_tidx,
    instructions,
    target_waypoints=batch["waypoints"],
    valid_mask=batch["valid_mask"],
    return_action_details=True,
)
loss = result["loss"]
```

此时还能读取 `candidate_trajectories`、`candidate_logits` 和 `nearest_anchor`。

## 4. 双 Agent 接入

`MultiAgentModelConfig` 支持两套独立锚点：

```python
MultiAgentModelConfig(
    use_anchor_diffusion=True,
    diffusion_agent1_anchor_path="/path/to/drone_anchors.npy",
    diffusion_agent2_anchor_path="/path/to/robotdog_anchors.npy",
    diffusion_num_anchors=40,
)
```

模型内部使用两个独立 `AnchorDiffusionActionModel`：

```text
h_act1 -> drone diffusion head
h_act2 -> robotdog diffusion head
```

默认输出仍兼容原接口：

```text
waypoints: (B, 2, Nw, 3)
```

训练时传入：

```python
out = model(
    ...,
    target_waypoints=batch["waypoints"],  # (B, 2, Nw, 3)
    valid_mask=batch["valid_mask"],        # (B, 2, Nw)
)
action_loss = out["action_loss"]
```

候选输出：

```text
candidate_trajectories: (B, 2, M, Nw, 3)
candidate_logits:       (B, 2, M)
candidate_scores:       (B, 2, M)
```

### 4.1 双 GND 与 bbox 训练策略

Grounding 已从一个共享 GND 查询改为两个 Agent-specific 查询：

```text
[文本, Agent1视觉+BBOX, ACT1, Agent2视觉+BBOX, ACT2, GND1, GND2]
                                                        │     │
                                                        │     └─ Agent2 bbox/visibility
                                                        └─────── Agent1 bbox/visibility
```

`GND1/GND2` 都位于完整双视角视觉上下文之后，因此仍能融合另一 Agent 的信息；但
两者分别加入自己的 `agent_emb/view_emb`，并使用独立可学习 query，避免一个共享
hidden state 同时承担两个图像坐标系。Grounding 输出 MLP 在两个 Agent 间共享，
以保持任务定义一致并控制参数量。

Grounding head 内部包含两条 bbox 分支：

```text
有有效 prior: noisy/previous bbox + bounded residual -> refined_bbox
无有效 prior: visual context -> absolute normalized bbox
```

训练不再使用“当前 GT bbox 原样输入，再回归同一个 GT bbox”。现在每个 Agent 独立：

1. 对 GT bbox 中心按框宽高比例加入高斯扰动。
2. 对宽高在 log-space 加入尺度扰动。
3. 按 `bbox_dropout_prob` 丢弃 prior，训练无框绝对检测。
4. 仅对可见且尺寸有效的目标计算 Smooth L1 bbox loss。
5. 两个 GND 输出都参与 bbox/visibility 监督，并分别收到对应 Agent 标签的梯度。

默认训练参数：

```text
BBOX_DROPOUT_PROB=0.50
BBOX_CENTER_JITTER_STD=0.25
BBOX_SIZE_JITTER_STD=0.20
```

这更接近闭环评估中“上一帧预测框存在误差，当前帧需要继续修正”的输入分布。

## 5. 从 DiffusionDrive 的迁移对应

参考代码位于：

```text
/data/hdt/DiffusionDrive/navsim/agents/diffusiondrive/
```

| DiffusionDrive 原实现 | `model_unrealzoo_anchor_diffusion.py` 对应实现 | 迁移与修改 |
|---|---|---|
| `transfuser_model_v2.py::TrajectoryHead` | `AnchorDiffusionActionModel` | 保留 anchor-near truncated diffusion、多模态轨迹和 score；输入条件改为 TrackVLA 的 `h_act` |
| `diffusers.schedulers.DDIMScheduler` | `TruncatedDDIMScheduler` | 直接复用外部 DDIMScheduler，并封装锚点附近截断扩散与统一的两步推理接口 |
| `plan_anchor = np.load(plan_anchor_path)` | `_load_trajectory_anchors()` + `register_buffer("anchors")` | 检查 `(M,Nw,D)` shape，并随 checkpoint 保存，不参与梯度 |
| `forward_train()` 中 `torch.randint(0, 50)` 与 `add_noise()` | `AnchorDiffusionActionModel.forward()` 训练分支 | 保留总步数 1000、训练截断前 50 步的锚点附近加噪 |
| `forward_test()` 中两步 scheduler update | `inference_timesteps()` + `scheduler.step()` | 保留两步 DDIM；使用一致的 `[10,0]` 调度，不复刻原代码初始 t=8 与 roll timestep 不一致的问题 |
| `CustomTransformerDecoderLayer` | `AnchorDiTBlock` | 改成 TrackVLA 论文要求的 DiT；使用 self-attention 与 AdaLN 条件调制 |
| `GridSampleCrossBEVAttention` | 未迁移 | 当前 OpenTrackVLA 没有 BEV feature map，直接迁移会造成坐标语义错误 |
| `ego_query/status_encoding` | `h_act` | 现有 LLM ACT token 隐藏状态已经是文本、视觉和时序融合后的规划条件 |
| `norm_odo()/denorm_odo()` 的汽车硬编码尺度 | 从 anchor bank 自动计算逐维 `action_scale` | 使用真实锚点最大绝对值归一化，适配机器人/无人机当前数据单位 |
| `modules/multimodal_loss.py::LossComputer` | `anchor_diffusion_tracking_loss()` | 保留最近锚正样本分配；使用 MSE + `lambda*BCE`，默认候选均值以稳定梯度，并增加 valid mask 与 wrapped theta |
| DiffusionDrive 的 20 个 XY anchor | 默认 40 个 `(x,y,theta)` anchor | 默认遵循 TrackVLA 表 5；允许配置数量，并满足用户要求的完整轨迹锚点 |

## 6. 与 TrackVLA 论文的对应

| 论文描述 | 当前实现 |
|---|---|
| 从训练轨迹做 K-means 得到 `{tau_i}` | `fit_trajectory_anchors_kmeans()` |
| 锚点加入高斯噪声得到 `{tau_tilde_i}` | `TruncatedDDIMScheduler.add_noise()` |
| `A_theta({tau_tilde_i}, E_pred_T)` | `AnchorDiffusionActionModel(condition=h_act)` |
| 输出 `{score_i, tau_hat_i}` | `candidate_logits` 与 `candidate_trajectories` |
| 最近锚为正，其余为负 | `nearest_anchor` 与 one-hot `score_target` |
| `MSE + lambda BCE` | `anchor_diffusion_tracking_loss()` |
| DiT 去噪 | `AnchorDiTBlock` |
| 推理两步 DDIM | `diffusion_inference_steps=2` |
| 选择 top-1 score 轨迹 | `_gather_top_candidate()` |

论文中的总损失：

```text
L = Ltrack + alpha * Ltext
```

当前 `model_unrealzoo_anchor_diffusion.py` 没有现成的文本生成 loss 分支，因此只输出 `Ltrack`。训练入口需要在
有文本监督时自行组合：

```python
loss = action_output["loss"] + alpha * text_loss
```

## 7. 锚点生成示例

现在文件名符合 Python 模块命名规则，可直接导入锚点工具：

```python
import numpy as np
import model_unrealzoo_anchor_diffusion

# trajectories: (样本数, Nw, 3)，必须是训练集中的局部轨迹
# valid_mask:  (样本数, Nw)
anchors, assignment = model_unrealzoo_anchor_diffusion.fit_trajectory_anchors_kmeans(
    trajectories,
    num_anchors=40,
    valid_mask=valid_mask,
    num_iters=100,
    seed=0,
)

model_unrealzoo_anchor_diffusion.save_trajectory_anchors(
    "anchors/trackvla_40.npy",
    anchors,
    {
        "coordinate_frame": "agent_local",
        "xy_unit": "meter",
        "theta_unit": "radian",
        "source_split": "train",
    },
)
```

双 Agent 应分别生成：

```text
anchors/drone_40.npy
anchors/robotdog_40.npy
```

## 8. 当前限制与后续接入事项

1. 原 `train.py --multi_agent` 仍训练 `model.py`；扩散版本必须使用 `train_unrealzoo_anchor_diffusion.py`。
2. `train_unrealzoo_anchor_diffusion.py` 会把所有 `diffusion_*` 字段及锚点路径写入 checkpoint config。
3. 锚点文件必须在构建模型时可访问；虽然锚点会写入 state dict，但当前构造阶段仍会先校验文件。
4. 当前 DiT 只使用单个 `h_act` 条件，没有 cross-attend 全部 LLM scene tokens。
5. 当前对 theta 也执行高斯扩散，并在输出/loss 中 wrap；如角度跨界样本很多，可改为扩散 `(x,y,sin(theta),cos(theta))`。
6. 双 Agent 当前独立生成候选，没有联合碰撞约束或候选组合打分。
7. 闭环控制仍需确认 waypoint 时间间隔与 waypoint-to-velocity 转换一致。
8. 双 GND 改变了 grounding 参数名；旧 checkpoint 可继续加载规划参数，但新的 GND 与 grounding head 需要重新训练。

新 checkpoint 的 config 会记录：

```text
grounding_architecture=dual_agent_gnd_v2
```

## 9. 已完成验证

已在 `omtracknew` 环境完成：

- `model_unrealzoo_anchor_diffusion.py` Python 语法检查。
- 新增代码 `git diff --check`。
- Anchor Diffusion 训练 forward。
- `Ltrack` 反向传播。
- 两步确定性 DDIM 推理。
- 候选轨迹与分数 shape 检查。
- 无第三方依赖 K-means 聚类测试。
- 当前双 Agent `dataset.json` 的小规模锚点聚类与训练入口 dry-run。
- 新训练 loss 适配器的 GT 路点传参与反向传播。

模块级验证使用的小配置：

```text
B=3, M=4, Nw=8, D_action=3
DiT hidden=64, depth=2, heads=4
```

尚未进行完整数据集训练、Habitat 闭环评估或 UnrealZoo 双 Agent 闭环评估。
