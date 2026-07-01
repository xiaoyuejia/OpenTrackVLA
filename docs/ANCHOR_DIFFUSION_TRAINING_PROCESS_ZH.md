# Anchor Diffusion 训练过程、结果与本质

本文说明执行下面命令后，模型内部实际发生了什么、训练结果保存了什么，以及应当如何理解这些结果：

```bash
RUN_BUILD_ANCHORS=0 RUN_DRY_RUN=0 RUN_TRAIN=1 \
  bash sh/train_anchor_diffusion.sh
```

当前训练入口为：

```text
sh/train_anchor_diffusion.sh
  -> train_unrealzoo_anchor_diffusion.py
  -> train.py::train_multi_agent()
  -> model_unrealzoo_anchor_diffusion.py::MultiAgentOpenTrackVLA
  -> model_unrealzoo_anchor_diffusion.py::AnchorDiffusionActionModel
```

## 1. 一句话理解训练本质

训练不是让模型从零生成任意轨迹，而是让模型学习：

1. 根据无人机和机器狗的历史视觉、当前视觉、目标框和文本指令，理解当前场景。
2. 从 40 种预定义轨迹模式中判断哪种模式最符合当前场景。
3. 将选中的粗略轨迹锚点去噪、修正为更接近真值的轨迹。
4. 同时学习目标框修正和目标可见性预测。

因此，该模型本质上是一个：

```text
场景条件编码器
    + 轨迹模式分类器
    + 基于锚点的轨迹修正器
    + 可选目标 grounding 辅助头
```

## 2. 训练前已经固定的内容

### 2.1 视觉特征缓存

训练不会重新运行 DINO/SigLIP 视觉编码器。图片已经提前转换为固定视觉 token：

```text
历史帧 coarse token: (B, 2, 124, 1536)
当前帧 fine token:   (B, 2, 64, 1536)
```

这些 token 从 `vision_cache` 读取，不参与更新。

### 2.2 轨迹锚点

训练前使用 K-means 从训练集轨迹生成两套锚点：

```text
无人机锚点: (40, 8, 3)
机器狗锚点: (40, 8, 3)
```

每条锚点轨迹包含 8 个 `(x, y, theta)` 路点。锚点代表训练数据中常见的粗略运动模式，例如：

```text
直行、左转、右转、缓慢移动、较快移动等
```

锚点在训练中作为 `buffer` 保存，不通过梯度更新。模型学习的是如何选择和修正它们。

## 3. 一个训练样本包含什么

每个样本同时包含无人机和机器狗的数据：

| 数据 | Shape | 含义 |
|---|---:|---|
| `coarse_tokens` | `(2, 124, 1536)` | 两个 Agent 的历史视觉特征 |
| `fine_tokens` | `(2, 64, 1536)` | 两个 Agent 的当前帧视觉特征 |
| `bbox_feat` | `(2, 4)` | 两个视角中的目标框 `(cx,cy,w,h)` |
| `waypoints` | `(2, 8, 3)` | 两个 Agent 的真值未来轨迹 |
| `valid_mask` | `(2, 8)` | 哪些真值路点有效 |
| `visible` | `(2,)` | 目标在两个视角中是否可见 |
| `instruction` | 字符串 | 例如跟踪目标且避免碰撞 |

DataLoader 将多个样本组成 batch，第一维变为 `B`。

## 4. 一次前向传播发生了什么

### 4.1 构建场景序列

两个 Agent 的视觉 token 先经过可训练的 projector，从 1536 维映射到 LLM hidden size。

随后加入：

- 时间编码：区分历史帧和当前帧。
- Agent 编码：区分无人机和机器狗。
- token 类型编码：区分视觉、bbox、ACT 和 GND。
- bbox token：提供目标所在位置。

最终输入 LLM 的序列近似为：

```text
[文本指令,
 无人机视觉与 bbox, ACT1,
 机器狗视觉与 bbox, ACT2,
 GND]
```

### 4.2 LLM 生成条件表示

模型取出三个位置的隐藏状态：

```text
h_act1: 无人机规划条件
h_act2: 机器狗规划条件
h_gnd:  目标检测与可见性条件
```

当前默认 `freeze_llm=True`，因此 Qwen 主干权重不会更新。

但是梯度仍然可以穿过冻结的 LLM，更新 LLM 输入侧的 projector、TVI、ACT/GND 查询 token，以及后面的扩散头和 grounding head。

### 4.3 锚点附近加噪

训练时，每个 Agent 的 40 条锚点轨迹都会被加入高斯噪声：

```text
总扩散步数 T = 1000
训练采样 timestep = [0, 49]
```

这里只使用靠近原始锚点的低噪声区域，因为任务不是从纯噪声生成轨迹，而是修正已有粗略轨迹。

### 4.4 DiT 输出候选轨迹与评分

每个 Agent 的扩散头接收：

```text
带噪锚点轨迹 + timestep + h_act
```

并输出：

```text
candidate_trajectories: (B, 40, 8, 3)
candidate_logits:       (B, 40)
```

含义分别是：

- 40 条去噪、修正后的候选轨迹。
- 模型认为每条候选轨迹适合当前场景的分数。

注意：训练阶段每个 batch 只随机采样一个 timestep 并执行一次 DiT 解码；推理阶段才执行默认的两步 DDIM 去噪。

## 5. 监督信号与损失

### 5.1 最近锚点分配

对于每个 Agent 的真值轨迹，首先计算它与 40 条原始锚点的 XY 距离：

```text
nearest_anchor = 与真值轨迹最近的锚点编号
```

该锚点被标记为正类，其余 39 条锚点为负类。

### 5.2 轨迹回归损失

只对最近锚点对应的去噪候选轨迹计算 MSE：

```text
regression_loss = MSE(最近锚点的预测轨迹, 真值轨迹)
```

`valid_mask` 会排除无效路点；角度误差会进行 `-pi/pi` wrap。

### 5.3 候选评分损失

模型需要学会给最近锚点较高分、给其余锚点较低分：

```text
score_loss = 对 40 个候选分数计算 BCE
```

默认先对 40 个候选的 BCE 取均值，再使用论文的平衡权重：

```text
score_loss_weight = 100
score_loss_reduction = mean
```

如需复现公式字面上的候选求和，可设置
`DIFFUSION_SCORE_LOSS_REDUCTION=sum`，但它会显著放大分类梯度。

单个 Agent 的扩散损失为：

```text
agent_loss = regression_loss + 100 * score_loss
```

双 Agent 的导航损失为：

```text
action_loss = agent1_loss + agent2_loss
```

### 5.4 Grounding 辅助损失

模型还可以学习：

```text
loss_bbox:    refined_bbox 与真值 bbox 的 MSE
loss_visible: 目标可见性的 BCE
```

最终训练总损失为：

```text
total_loss =
    beta_nav     * action_loss
  + beta_bbox    * loss_bbox
  + beta_visible * loss_visible
```

当前脚本默认：

```text
beta_nav=1.0
beta_bbox=1.0
beta_visible=0.5
```

## 6. 反向传播到底更新了什么

默认 `freeze_llm=True` 时：

| 模块 | 是否更新 |
|---|---|
| 预缓存视觉特征 | 否 |
| K-means 轨迹锚点 | 否 |
| Qwen LLM 主干权重 | 否 |
| 视觉 projector | 是 |
| 时间/Agent/type/bbox 编码模块 | 是 |
| ACT1、ACT2、GND 查询 token | 是 |
| 无人机 Anchor Diffusion DiT | 是 |
| 机器狗 Anchor Diffusion DiT | 是 |
| 候选轨迹评分头 | 是 |
| bbox/visibility grounding head | 是 |

训练通过 AdamW 更新这些可训练参数。梯度累积为 8 时，连续处理 8 个 micro-batch 后才执行一次优化器更新。

## 7. 本次实际训练过程

当前训练数据共有：

```text
样本数 = 1830
batch_size = 2
grad_accum_steps = 8
epochs = 1
```

因此：

```text
每个 epoch 的 micro-batch 数 = 1830 / 2 = 915
优化器更新次数 = ceil(915 / 8) = 115
```

本次训练确实完成了 115 个 optimizer step，并保存了：

```text
ckpt/ckpts_multi_agent_anchor_diffusion/
├── model_epoch00_step000100.pt
├── model_epoch00_step000115_final.pt
└── train_log.csv
```

## 8. 如何理解当前训练日志

日志字段含义：

| 字段 | 含义 |
|---|---|
| `loss` | 加权后的最终总损失 |
| `loss_ema` | 总损失的指数滑动平均 |
| `loss_nav` | 两个 Agent 的 Anchor Diffusion 损失之和 |
| `loss_bbox` | bbox 修正损失 |
| `loss_visible` | 可见性预测损失 |
| `final_epe` | top-1 预测轨迹最后一个 XY 路点与真值的平均距离 |
| `grad_norm` | 梯度裁剪前的总梯度范数 |

修复后的稳定默认配置中，初始 `loss_nav` 通常接近：

```text
2 个 Agent * mean_BCE(logits=0) * 100 ≈ 138.6
```

旧版 `sum` 配置初始值约为 5545，但会让候选评分梯度压过轨迹回归梯度，
不再作为默认训练设置。

`final_epe` 在不同 batch 间波动较大，因为它使用当前 top-1 分数候选，而训练回归监督使用的是“与 GT 最近的固定锚点候选”。评分头尚未稳定时，top-1 候选可能不是被回归监督的候选。

## 9. 训练结果到底是什么

### 9.1 Checkpoint

最终 checkpoint 是训练的主要结果。它保存：

```text
model_state:  模型全部参数和 buffer，包括固定锚点
optim_state:  AdamW 优化器状态
scaler_state: AMP scaler 状态
config:       训练和扩散参数
epoch/step:   恢复训练所需进度
```

当前单个 checkpoint 约为 1.7 GB，因为 `model_state` 中也包含完整的冻结 Qwen 权重，并且还保存了优化器状态。

### 9.2 模型学到的能力

训练完成后，模型学到的不是一条固定轨迹，而是以下映射：

```text
双 Agent 视觉历史 + 当前视觉 + bbox + 指令
    ->
每个 Agent 的 40 条场景条件候选轨迹
    +
候选轨迹评分
    ->
top-1 最终预测轨迹
```

模型还学习了目标框修正和目标可见性预测。

### 9.3 推理输出

推理时不提供真值轨迹，也不计算 `action_loss`。每个 Agent 会：

1. 从固定锚点附近的加噪轨迹开始。
2. 执行两步 DDIM 去噪。
3. 输出 40 条候选轨迹及其分数。
4. 选择分数最高的轨迹作为最终 `waypoints`。

最终可用于控制的是：

```text
waypoints: (B, 2, 8, 3)
```

## 10. 当前结果能证明什么，不能证明什么

当前训练结果可以证明：

- 数据、视觉缓存、锚点和模型 shape 能正确连接。
- 候选评分损失正在下降。
- 模型能够完成前向传播、反向传播和 checkpoint 保存。
- 模型已经开始学习训练集中的轨迹模式与场景条件关系。

当前训练结果不能单独证明：

- 模型在未见场景中具有良好泛化能力。
- top-1 轨迹已经足够准确。
- 模型在 UnrealZoo 或 Habitat 闭环中能稳定跟踪目标。
- 模型学会了避免碰撞；当前 loss 没有显式碰撞约束。
- 模型在没有真值 bbox 输入时具有可靠目标检测能力。

要验证最终能力，需要加载 `model_epoch00_step000115_final.pt`，在不提供真值轨迹和真值 bbox 的条件下运行闭环评估，并统计跟踪成功率、碰撞率、轨迹误差和 bbox 指标。

## 11. 最重要的理解

该训练的核心不是“用扩散模型从纯噪声画出路线”，而是：

```text
使用 K-means 锚点提供可行的粗略轨迹先验，
使用视觉和语言条件选择正确的轨迹模式，
再使用 DiT 将粗略轨迹修正为适合当前场景的轨迹。
```

锚点负责提供运动模式先验，LLM 隐藏状态负责表达场景，DiT 负责条件轨迹修正，评分头负责选择最终轨迹。
