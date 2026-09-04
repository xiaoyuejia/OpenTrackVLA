# AirGround-Coop V3 详细方法

> 本文档以 `sh/train_airground_coop_v3.sh` 为训练入口，以
> `train_airground_coop_v3.py`、`model_airground_coop_v3.py` 和
> `eval_airground_coop_v3.py` 的当前实现为唯一依据。文中既给出可直接写入论文
> Method 章节的定义和公式，也保留了张量形状、训练超参数、数据增强细节与
> 训练/推理差异，便于实验复现和后续修改论文。

## 1. 问题定义与设计动机

我们考虑一个由无人机 (D) 和地面机器狗 (G) 组成的异构空地双 Agent 具身跟踪
系统。两个 Agent 同时观测同一个目标行人，并分别输出局部轨迹以驱动各自的底层
控制器。设 Agent 集合为

\[
\mathcal{A}=\{D,G\}.
\]

在时刻 $t$，Agent $a\in\mathcal A$ 的输入包括当前 RGB 观测 $I^a_t$、过去 $H=31$
帧视觉历史 $\mathcal H^a_t$、YOLO 人物候选框及场景分割网格、两个 follower 的位姿，
以及语言跟踪指令 $l_t$。模型预测长度 $N=10$ 的局部位姿轨迹

\[
\tau^a_t=\left\{\mathbf w^a_{t,n}\right\}_{n=0}^{N-1},\qquad
\mathbf w^a_{t,n}=[x^a_{t,n},y^a_{t,n},\psi^a_{t,n}].
\]

其中 $n=0$ 是当前 Agent 局部坐标系的结构性原点 $[0,0,0]$，不是未来回归点；
真正的未来监督从 $n=1$ 开始。当前数据中标签时域为 $0.1\sim0.9\,\mathrm{s}$，
即原点后的 9 个未来位姿点。

双视角直接拼接的单一策略容易学习两个 Agent 之间的虚假运动绑定：例如机器狗
因障碍停滞时，仍然可正常观测目标的无人机也会降低前进速度。为此，V3 将
“正常跟踪”与“失视协同恢复”显式分解为三条信息流：

1. 无人机隔离的 SELF-D 流；
2. 机器狗隔离的 SELF-G 流；
3. 共同读取双视角和双位姿的 COOP-DG 流。

这一分解强制满足一个关键不变性：当无人机仍可见时，修改机器狗的视觉、检测或位姿
不能改变无人机 SELF 轨迹；反之亦然。只有当某个 Agent 自身失视而另一个
Agent 仍可见时，联合 COOP 流才接管失视端。

## 2. 方法总览

AirGround-Coop V3 由四个逻辑层次构成：

1. **离线/在线视觉前端**：DINOv3 与 SigLIP 产生双路语义特征，YOLO 产生人物候选框
   和类别无关的场景网格。
2. **带类型 token 构造**：将历史、当前帧、候选 ROI、检测值、绝对位姿、有方向的
   相对位姿以及任务 query 投影到冻结 Qwen3-0.6B 的 hidden space。
3. **三信息流策略预测**：两条 SELF row 在一次 $2B$ 前向中打包执行，但彼此没有
   attention 连接；COOP 在另一次 $B$ 前向中融合双视角。
4. **可验证可见性路由与闭环控制**：通过 `YOLO valid/confident AND LLM VERIFY accepted`
   产生带滞回的可见性状态，在 SELF、COOPERATIVE、BELIEF 和 SEARCH 之间路由轨迹，
   最后通过 inverse-fixed-dt 控制器转换为无人机和机器狗动作。

对应的 pipeline 图见
[`docs/figures/airground_v3_pipeline_paper.svg`](figures/airground_v3_pipeline_paper.svg)。

## 3. 输入表示与数据预处理

### 3.1 训练样本组成

每个样本同时包含无人机和机器狗的同步数据。核心张量形状如下，其中 $B$ 是物理
mini-batch 大小，$C_v=1536$ 是 DINOv3 和 SigLIP 拼接后的视觉维度。

| 字段 | 形状 | 含义 |
| --- | --- | --- |
| `coarse_tokens` | $(B,2,124,1536)$ | 31 帧历史，每帧 $2\times2=4$ token |
| `coarse_tidx` | $(B,2,124)$ | 历史时间索引 $0\ldots30$ |
| `fine_tokens` | $(B,2,64,1536)$ | 当前帧 $8\times8$ token |
| `fine_tidx` | $(B,2,64)$ | 当前帧索引 31 |
| `detection_feat` | $(B,2,6)$ | $[c_x,c_y,w,h,s,v]$ |
| `perception_grid` | $(B,2,8,8,4)$ | 每格 `[unknown, free, obstacle, target]` 比例 |
| `agent_poses` | $(B,2,4)$ | $[x_m,y_m,\sin\psi,\cos\psi]$ |
| `waypoints` | $(B,2,10,3)$ | clean 局部未来位姿 |
| `valid_mask` | $(B,2,10)$ | waypoint 监督有效性，索引 0 为 false |
| `target_pose` | $(B,2,5)$ | 目标相对 Agent 的 $[f,r,z,\sin\Delta\psi,\cos\Delta\psi]$ |

历史不足 31 帧时，数据集使用最早的可用历史帧进行左侧填充；若完全没有历史，
则使用当前帧的 coarse token 填满历史。当前配置 `online_encode_missing=false`，因此训练时
强制使用预先生成的 vision cache，缺失 cache 会直接报错，不会在 DataLoader worker 内
临时编码。

waypoint 标签必须显式带有
`waypoint_label_source='recorded_pose_fixed_dt'`，不允许在训练时根据 action 重新积分或
插值构造标签。这保证训练轨迹与后续 fixed-dt 反动力学控制的时间语义一致。

### 3.2 DINOv3-SigLIP 视觉前端

输入图像被 resize/letterbox 到 $384\times384$。对同一图像，视觉前端分别提取：

- DINOv3 patch token，默认 hidden size 为 384；
- `google/siglip-so400m-patch14-384` 的 SigLIP patch token，hidden size 为 1152。

SigLIP token 首先采样到 DINOv3 的 patch grid，然后沿特征维拼接：

\[
\mathbf V=\left[\mathbf V^{\mathrm{DINOv3}};
\mathbf V^{\mathrm{SigLIP}}\right]\in\mathbb R^{P\times1536}.
\]

拼接特征通过空间 average pooling 得到两种尺度：

\[
\mathbf V^{a,h}_{t-i}\in\mathbb R^{4\times1536},\qquad
\mathbf V^{a,f}_{t}\in\mathbb R^{64\times1536}.
\]

历史帧只保留 4 个 coarse token，当前帧保留 64 个 fine token。这种非对称压缩既保留
当前帧的精细定位能力，又将 31 帧历史的 token 数量控制在 124。训练时该视觉
编码结果已离线缓存，DINOv3 和 SigLIP 不参与梯度更新；推理时使用相同编码器
在线提取。

### 3.3 YOLO 候选框与类别无关场景网格

离线 perception cache 使用 YOLO11m instance segmentation。对每幅图像，置信度最高且超过
0.25 的 person instance 被当作人物候选，所有其他超过 0.25 的前景 instance（包括第二个
person）被合并为 obstacle。每个 Agent 获得一个 6 维检测向量

\[
\mathbf d^a=[c_x,c_y,w,h,s,v],
\]

其中前 4 项是归一化 `cxcywh`，$s$ 是置信度，$v\in\{0,1\}$ 表示候选是否有效。
无效时前 5 项全部置零。

实例 mask 被池化为 $8\times8$ 网格。每个网格保存四类像素比例

\[
\mathbf g^a_{ij}=[g_u,g_f,g_o,g_t],
\]

分别表示 unknown、free、obstacle 和 detector target。候选框与 $g_t$ 来自同一 YOLO，
因此 $g_t$ 不能被当作候选是真实跟踪目标的独立证据。在进入 V3 前，实现将 target
质量并入 unknown：

\[
\tilde{\mathbf g}_{ij}
= [\min(1,g_u+g_t),\;g_f,\;g_o,\;0].
\]

这样保留了场景的 unknown/free/obstacle 上下文，同时防止 VERIFY head 从 YOLO target mask
中抄写答案。

### 3.4 检测候选 ROI token

当前实现不单独裁剪一张 ROI 图像，而是在当前帧 $8\times8$ fine token 上按 YOLO 框做
显式池化。代码先以候选框中心为中心、以 $\max(w,h)$ 为边长构造方形区域。设由该区域
覆盖的 token 集合为 $\Omega(\mathbf d^a)$，则

\[
\mathbf r^a
=P_{\mathrm{roi}}\left(
\frac{1}{|\Omega|}\sum_{j\in\Omega}P_v(\mathbf V^{a,f}_{t,j})
\right),
\]

其中 $P_v$ 是跨模态投影器，$P_{\mathrm{roi}}$ 是
`LayerNorm-Linear-GELU-Linear` 投影。即使框小到没有覆盖任何网格中心，代码也会选择距离
框中心最近的一枚 token，保证每个有效候选都有视觉 ROI 表征。

最终的 detection token 为

\[
\mathbf z^a_{det}=E_{\mathrm{TVI}}
\left(P_d(\mathbf d^a)+\mathbf r^a,\;k=\texttt{DETECTION},\;a\right),
\]

其中 $P_d$ 是两层 MLP。这一 token 同时包含数值框位置、检测置信度和框内当前
RGB 语义。

### 3.5 视觉 token 投影与 TVI 类型编码

所有 1536 维视觉 token 通过共享的跨模态投影器映射到 Qwen hidden dimension $d_L$：

\[
P_v(\mathbf v)=W_2\,\mathrm{GELU}(W_1\,\mathrm{LN}(\mathbf v)).
\]

对于当前帧 token，场景网格也通过两层 MLP 投影并加到视觉 token 上：

\[
\mathbf z^{a,f}_{j}=P_v(\mathbf V^{a,f}_{t,j})+P_g(\tilde{\mathbf g}^{a}_{j}).
\]

随后添加 time/view/agent/kind 编码：

\[
\hat{\mathbf z}=\mathbf z+
\mathbf e_{time}(t)+\mathbf e_{view}(a)+\mathbf e_{agent}(a)+\mathbf e_{kind}(k).
\]

V3 对 history、current、detection、self-ACT、coop-ACT、pose、target、obstacle、missing 和 VERIFY
共使用 10 种 kind ID。当前 `use_angle_tvi=false`，不额外加入历史航向 angle marker。

`insert_time_tokens=true` 时，每个视觉帧的 token group 前还会插入一枚显式 time-kind-agent
marker。因此单个 Agent 的默认视觉序列长度为：

\[
(31\times4+31)+(64+1)+1=221,
\]

即 124 个历史视觉 token + 31 个历史 marker + 64 个当前 token + 1 个当前 marker +
1 个 detection/ROI token。

## 4. 双层位姿表示

### 4.1 共享坐标系中的绝对位姿

从 Unreal 位姿记录中取得两个 Agent 的世界坐标与 yaw，并将 Unreal 单位除以 100
转为米：

\[
\mathbf p^a=[x^a_m,y^a_m,\sin\psi^a,\cos\psi^a].
\]

训练时，先以两个 Agent 的中点为原点，再施加一个双 Agent 共享的随机 $SE(2)$ 变换：

\[
\mathbf c=\frac{\mathbf x^D+\mathbf x^G}{2},\qquad
\mathbf x^{a\prime}=R(\theta)(\mathbf x^a-\mathbf c)+\mathbf t,
\qquad
\psi^{a\prime}=\psi^a+\theta.
\]

其中 $\theta\sim U[-180^\circ,180^\circ]$，平移的两个分量独立采样于
$[-20,20]\,\mathrm m$。该增强打破对固定世界坐标原点和朝向的依赖。位置分量在投影前除以
20 m，然后经过无 per-token LayerNorm 的两层 MLP，避免 LayerNorm 破坏度量大小。

推理时不施加随机变换，只将两个 follower 的实时世界 XY 减去两者中点，从而得到
一个共享、米制且与训练分布相容的坐标系。

### 4.2 receiver-local 有方向相对位姿

仅提供两枚绝对位姿 token，意味着冻结 LLM 需要自行学习坐标相减和 $SE(2)$ 旋转。V3 因此
为每个可能的 receiver $r$ 额外构造一枚有方向的相对位姿 token。设 source $s=1-r$，
世界平移差为

\[
\Delta\mathbf x=\mathbf x^s-\mathbf x^r.
\]

将其转到 receiver 的局部前-右坐标系：

\[
\begin{aligned}
f_{s\leftarrow r}&=\cos\psi^r\Delta x+\sin\psi^r\Delta y,\\
r_{s\leftarrow r}&=-\sin\psi^r\Delta x+\cos\psi^r\Delta y,\\
\Delta\psi_{s\leftarrow r}&=\psi^s-\psi^r.
\end{aligned}
\]

最终的相对特征为

\[
\boldsymbol\rho^r=
\left[
\frac{f_{s\leftarrow r}}{20},
\frac{r_{s\leftarrow r}}{20},
\sin\Delta\psi_{s\leftarrow r},
\cos\Delta\psi_{s\leftarrow r},
\frac{\|\Delta\mathbf x\|_2}{20}
\right].
\]

两枚绝对 token 和两枚 receiver-local 相对 token 共同进入 COOP 流。共享 $SE(2)$ 数据增强
会改变绝对 token，但不改变这两枚相对 token，从而同时保留全局布局与局部实体几何。

## 5. 冻结 LLM 上的三条信息流

### 5.1 共享参数与上下文隔离

三条信息流共享同一个 Qwen3-0.6B 的参数，但不共享同一 attention context。
Qwen 权重在当前配置中完全冻结：

\[
\nabla_{\theta_{\mathrm{LLM}}}\mathcal L=0.
\]

训练的是视觉/检测/位姿投影、TVI embedding、可学习 query token、SELF planner、VERIFY head、
JEPA predictor、target-belief/uncertainty head 和两个 cooperative trajectory decoder。

单个物理 batch 会触发两次 Qwen forward：

- forward #1：SELF-D 与 SELF-G 沿 batch 维拼成 $2B$ 条序列；
- forward #2：每个样本一条双视角 COOP-DG 序列，batch 大小为 $B$。

因此，当前实现的计算单位为 $2B+B$，而不是为 ACT 和 VERIFY 分别复制视觉序列的
$4B+B$。

### 5.2 视觉隔离的 SELF-D/SELF-G 流

对 Agent $a$，clean SELF 序列按如下顺序组装：

\[
\mathcal S^a_{self}=
[E(l^a_{track},l^a_{verify});
Z^{a,h}_{clean};
Z^{a,f}_{clean};
z^a_{det};
q^a_{VERIFY};
q^a_{ACT}].
\]

语言中同时包含两个任务：

- tracking prompt：要求对应 Agent 独立跟随目标并避免碰撞；
- verification prompt：要求判断 YOLO 给出的人物候选是否为正在跟踪的目标。

SELF-D 只读取无人机的 clean history/current/detection，SELF-G 只读取机器狗的对应数据。
两条 row 沿 batch 维拼接而不是沿 sequence 维拼接，所以它们没有任何跨视角的
attention 边。

VERIFY 严格位于 ACT 之前。由于 Qwen 采用 causal mask，其梯度和信息关系为：

- $h^a_{VERIFY}$ 看不到后面的 ACT token，因此 target-match loss 不会通过 ACT 分支回传；
- $h^a_{ACT}$ 可以看到 VERIFY token 及其之前的所有证据，因此 SELF action 可以利用目标校验上下文。

#### SELF 轨迹头

每个 Agent 有一个独立的 3-layer planner：

\[
\hat\tau^a_{self}
=\mathrm{reshape}\left(
W_3\,\sigma(W_2\,\sigma(W_1\,\mathrm{LN}(h^a_{ACT})))
\right)\in\mathbb R^{N\times3},
\]

隐藏层宽度为 $2d_L$，激活为 GELU。当前 `no_tanh_actions=true`，输出不做 tanh 截断。
第 0 个 waypoint 被显式替换为零向量，保证轨迹从 Agent 当前原点出发。

#### 目标匹配 VERIFY 头

两个 Agent 各自使用一个 `LayerNorm-Linear-GELU-Linear` 二分类头：

\[
p^a_{match}=\sigma(f^a_{verify}(h^a_{VERIFY})).
\]

该 head 不重复判断“画面中是否有人”，而是回答一个条件问题：“在 YOLO 已给出一个
有效人物候选的前提下，这个人是否是指定跟踪目标？”。

### 5.3 双视角联合 COOP-DG 流

COOP 序列使用独立的联合 instruction，并按以下顺序组装：

\[
\begin{aligned}
\mathcal S_{coop}=[
&E(l_{joint});
Z^D_{coop};Z^G_{coop};\\
&q^D_{abs};q^G_{abs};
q^D_{rel};q^G_{rel};
q^D_{COOP};q^G_{COOP}].
\end{aligned}
\]

其中 $Z^a_{coop}$ 与 SELF 使用相同的 clean token 投影参数，但会根据训练期 receiver
corruption mask 把指定视觉 token 替换为 Agent-specific learned `[MISSING]` token。如果 Agent 被
选为 synthetic receiver，它在 COOP 流中的 detection 数值和 candidate ROI 都被置零；SELF 流始终
保留 clean 输入。

COOP LLM 输出中，实现保留：

- 两个 Agent 的视觉 stream hidden states；
- 两枚绝对 pose hidden states；
- 两枚有方向 relative-pose hidden states；
- 两枚 COOP-ACT query hidden states。

文本 hidden state 不直接进入后续 decoder memory，但文本已经通过 causal attention 影响了上述
视觉和 query hidden。对 Agent $a$，共享基础 memory 写为

\[
M_{base}=[H^D_{stream};H^G_{stream};H_{abs};H_{rel}],
\]

而它自己的 cooperative context 为 $c^a=H(q^a_{COOP})$。

**因果序列实现细节。** 当前代码固定按 `COOP_ACT_D` 后接 `COOP_ACT_G` 的顺序放置两枚
query。在 causal LLM 中，后者可见前者，前者不可见后者。两个轨迹 decoder 的权重仍然独立，
但论文中若要严格宣称 query 级完全对称，需要将两枚 query 改成并行不可见或分别前向。
本文档在其他地方使用“双向协同”，表示训练同时支持 $D\rightarrow G$ 和 $G\rightarrow D$，
不表示两枚 query 在 causal attention 中完全交换对称。

## 6. Receiver corruption curriculum

### 6.1 合成协同样本的启用条件

V3 不对所有样本无条件制造失视。一个样本只有在以下条件全部成立时才是
synthetic-assistance candidate：

1. GT 表明两个 Agent 都可见目标；
2. 两个 YOLO 候选都有效；
3. 两个候选与 GT 目标框的 IoU 均不小于 0.30；
4. 两个 perception cache 均有效；
5. 两个 follower pose 均可解析。

在满足上述条件的样本中，先按当前 curriculum stage 的
`assistance_probability` 决定是否合成 receiver。如果启用，再以互斥的 $0.5/0.5$
条件概率选择无人机或机器狗作为 receiver；同一样本最多只有一个 receiver。
未启用合成辅助时，样本仍保留 clean 双视角 COOP 输入，而不是强行破坏某个视角。

### 6.2 三阶段遮挡课程

默认 10 个 epoch 的课程如下。表中四种 corruption mode 的概率是在“本样本已被选为
synthetic assistance”条件下的概率。

| Epoch | assistance | ROI_ONLY | CURRENT_FULL | RECENT_FULL | ALL_FULL | 位姿扰动概率 | 最大平移 / yaw |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1–3 | 0.70 | 0.70 | 0.30 | 0.00 | 0.00 | 0.00 | 0.25 m / 10°（本阶段实际不启用） |
| 4–7 | 0.85 | 0.40 | 0.30 | 0.25 | 0.05 | 0.50 | 0.50 m / 30° |
| 8–10 | 0.90 | 0.30 | 0.25 | 0.35 | 0.10 | 0.50 | 0.50 m / 30° |

四种模式的具体操作是：

- **ROI_ONLY**：只把当前帧中 YOLO 框扩大 1.5 倍后覆盖的 fine token 替换为
  `[MISSING]`。当前背景、场景网格和全部历史保留。
- **CURRENT_FULL**：屏蔽当前帧全部 64 个 fine token，历史保留。
- **RECENT_FULL**：屏蔽当前帧，同时屏蔽最近 $L$ 帧 coarse history。
  Epoch 4–7 中 $L\in\{2,4\}$，Epoch 8–10 中 $L\in\{2,4,8,16\}$，从集合中均匀抽取。
- **ALL_FULL**：屏蔽当前帧和 31 帧全部历史。

屏蔽作用于 token 级 `fine_missing_mask` 和 `coarse_missing_mask`，而不是先在像素空间
生成一张黑图。每个 Agent 有独立的 learned missing token：

\[
\tilde{\mathbf z}^{a}_{j}=
\begin{cases}
\mathbf m^a, & M^a_j=1,\\
\mathbf z^a_j, & M^a_j=0.
\end{cases}
\]

对于 full-current corruption，场景网格也不再可信，因此被替换为全 unknown 网格。
ROI_ONLY 仍保留真实背景，因此保留 unknown/free/obstacle 网格上下文。不论哪种模式，
synthetic receiver 在 COOP 流中的 detection token 数值和 candidate ROI 都被置零，避免模型
通过当前框直接定位被遮挡目标。

### 6.3 为什么 ROI_ONLY 不能伴随位姿重定位

ROI_ONLY 保留了原始 receiver pose 下拍摄的真实背景。如果同时将 receiver 平移或旋转到
反事实位姿，视觉背景与 pose token 将彼此矛盾。因此 V3 只在 CURRENT_FULL、RECENT_FULL
和 ALL_FULL 三种“当前帧完全缺失”的模式中允许物理位姿重定位。

选中扰动时，先在 receiver 局部坐标系采样

\[
R\sim U(0,R_{max}),\quad
\phi\sim U(-\pi,\pi),\quad
\delta_\psi\sim U(-\psi_{max},\psi_{max}),
\]

\[
\delta_f=R\cos\phi,\qquad \delta_r=R\sin\phi.
\]

注意代码对半径 $R$ 做均匀采样，而不是对圆盘面积做均匀采样。扰动后的 receiver
位姿为

\[
\mathbf x_r^{new}=\mathbf x_r+\delta_f\mathbf e_f(\psi_r)+
\delta_r\mathbf e_r(\psi_r),\qquad
\psi_r^{new}=\psi_r+\delta_\psi.
\]

### 6.4 可执行 receiver-recovery target

物理重定位后，不能简单将 clean trajectory 刚性变换到新 receiver frame 就直接当作监督：
这会让 $n=0$ 仍是原点，但 $n=1$ 瞬间吸收完整的平移/旋转扰动，形成不可执行跳变。

设 clean waypoint 在原 receiver frame 中为
$\mathbf w_i=[\mathbf p_i,\psi_i]$，receiver 扰动为
$\boldsymbol\delta=[\delta_f,\delta_r,\delta_\psi]$。先将 clean path 刚性变换为新 receiver frame 中的
**reference**：

\[
\mathbf p_i^{ref}=R(-\delta_\psi)
(\mathbf p_i-[\delta_f,\delta_r]^\top),\qquad
\psi_i^{ref}=\mathrm{wrap}(\psi_i-\delta_\psi).
\]

随后从 $\hat{\mathbf w}_0=[0,0,0]$ 开始逐步 rollout。两个 Agent 的速度上限均为
$v_{max}=2.5\,\mathrm{m/s}$，yaw rate 上限均为
$\omega_{max}=1.5\,\mathrm{rad/s}$。对每一步，

\[
\Delta s_{max}=v_{max}\Delta t,\qquad
\Delta\psi_{max}=\omega_{max}\Delta t.
\]

对无人机，使用 holonomic rollout：沿当前位置到 $\mathbf p_i^{ref}$ 的方向前进，位移幅度
不超过 $\Delta s_{max}$，yaw 误差被截断到
$[-\Delta\psi_{max},\Delta\psi_{max}]$。

对机器狗，使用 rotate-then-forward 的非完整 rollout。设位置误差方位角为 $b_i$，
当前 yaw 为 $\hat\psi_{i-1}$，则

\[
e_{\psi}=\mathrm{wrap}(b_i-\hat\psi_{i-1}),\qquad
\Delta\hat\psi=\mathrm{clip}(e_\psi,-\Delta\psi_{max},\Delta\psi_{max}),
\]

\[
s_i=\min(\|\mathbf p_i^{ref}-\hat{\mathbf p}_{i-1}\|,\Delta s_{max})
\max(\cos e_\psi,0),
\]

\[
\hat{\mathbf p}_{i}=\hat{\mathbf p}_{i-1}
+s_i
\begin{bmatrix}
\cos(\hat\psi_{i-1}+\frac12\Delta\hat\psi)\\
\sin(\hat\psi_{i-1}+\frac12\Delta\hat\psi)
\end{bmatrix}.
\]

当 reference 位于机器狗身后时，$\max(\cos e_\psi,0)$ 会抑制前进，先进行旋转；
位移方向使用段中点航向，近似一段恒曲率 unicycle arc。

仅被物理重定位的 synthetic receiver 使用上述恢复标签。source、ROI_ONLY、未扰动的
CURRENT/RECENT/ALL 样本，以及所有 SELF 分支，均继续使用原始 clean waypoint 标签。

目标 belief 标签也同步变换到新 receiver frame，以保证 target-belief head 和 recovery trajectory
在同一反事实坐标系中受监督。

## 7. Conditional JEPA 失视表征恢复

### 7.1 Online projector 与 EMA teacher

V3 不在像素空间重建遮挡图像，而是预测 clean current-view 的高层 fine-token latent。
online projector 为前述 $P_v$，teacher projector $\bar P_v$ 初始化为 online projector 的深拷贝，
不参与梯度更新。每次 training forward 开始时执行

\[
\bar\theta\leftarrow m\bar\theta+(1-m)\theta,
\qquad m=0.996.
\]

实现上 EMA 是每个 forward/micro-batch 更新，而不是仅在 optimizer step 后更新。当使用
gradient accumulation 时，这一细节会决定 teacher 的实际时间常数。

teacher target 为

\[
\bar Z^a=\mathrm{stopgrad}(\bar P_v(V^{a,f}_{clean}))\in\mathbb R^{64\times d_L}.
\]

Teacher 不叠加 perception grid，因此 JEPA 预测的是 clean RGB 语义 latent，而不是 YOLO 标签的复制。

### 7.2 Conditional predictor

对 Agent $a$，JEPA memory 是

\[
M^a_{JEPA}=[M_{base};c^a].
\]

predictor 将该 memory 和 receiver 的 64 个 cooperative fine token 投影到 512 维空间，对每个
空间位置加入 learned spatial query，再通过 3 层、8 头 Transformer decoder：

\[
\hat Z^a=J^a(M^a_{JEPA},\tilde Z^{a,f}_{coop})
\in\mathbb R^{64\times d_L}.
\]

predictor 实际上为全部 64 个位置产生预测，但 JEPA loss 只在 `fine_missing_mask`
标记的位置上计算。因此 ROI_ONLY 只恢复目标区域，full-current 模式则恢复全部当前帧
latent。自然失视但没有 synthetic mask 的样本不计算 JEPA loss。

### 7.3 Target belief 与不确定性

对预测 token 学习一个 scalar pooling score。若当前存在遮挡 mask，softmax 只在被遮挡 token
上归一化；否则在 64 个 token 上归一化。得到聚合表征

\[
\mathbf z^a_{pool}=\sum_j\alpha_j^a\hat{\mathbf z}_j^a.
\]

该表征同时输入：

- target-belief head：预测
  $[f_{target},r_{target},z_{target},\sin\Delta\psi,\cos\Delta\psi]$；
- uncertainty head：预测一个 scalar log-variance $s^a$。

这两个头只在对应 Agent 实际是 cooperative receiver 且 target-pose label 有效时受监督。

## 8. Agent-specific 多模态协同轨迹解码器

### 8.1 Decoder memory

对 Agent $a$，cooperative decoder 的 memory 为

\[
M^a_{dec}=[M_{base};\hat Z^a;O^a],
\]

其中 $O^a\in\mathbb R^{64\times d_L}$ 是对 effective perception grid 经过独立 obstacle-grid MLP、
learned $8\times8$ spatial position embedding 和 OBSTACLE kind embedding 后得到的 token。这些 obstacle token
不经过 Qwen，而是直接作为 cooperative decoder memory。

需要强调，当前 obstacle grid 仍然位于图像坐标系，模型可以将它当作语义上下文，
但不能直接将局部 XY waypoint 与其计算几何碰撞 loss。

### 8.2 K-mode trajectory decoding

无人机和机器狗各有一个参数独立的多模态 decoder。每个 decoder 的 hidden size 为 512，
memory 经过 1 层 Transformer encoder，query 经过 3 层、8 头 Transformer decoder。

为了产生 $K=4$ 种轨迹模态，为每个 mode $k$ 学习 mode query $q_k^{mode}$，为每个 waypoint
位置 $n$ 学习 waypoint query $q_n^{wp}$，并加入该 Agent 的 cooperative ACT context：

\[
q^a_{k,n}=q_k^{mode}+q_n^{wp}+P_c(c^a).
\]

将 $K\times N$ 个 query 与 $M^a_{dec}$ 做 cross-attention，得到

\[
\hat{\mathcal T}^a=
\{\hat\tau^a_k\}_{k=1}^{K}
\in\mathbb R^{K\times N\times3}.
\]

每个 candidate 的第 0 个 waypoint 同样被锁定为 $[0,0,0]$。对每个 mode 的 $N$ 个 decoded
hidden 做平均后，再经过 `LayerNorm-Linear-GELU-Linear` 得到 mode logit
$\ell^a_k$。推理时选择

\[
k^{a*}=\arg\max_k \ell^a_k,
\qquad
\hat\tau^a_{coop}=\hat\tau^a_{k^{a*}}.
\]

## 9. 监督信号与训练损失

### 9.1 SELF 和 COOP 监督的有效性掩码

训练中不是所有 Agent/sample 都同时计算所有轨迹 loss。代码先构造以下布尔掩码：

\[
v^a_{det}=\mathbb 1[\text{YOLO valid}],
\]

\[
y^a_{match}=\mathbb 1[
\text{GT visible}\land v^a_{det}
\land \mathrm{IoU}(d^a,b^a_{GT})\ge0.30],
\]

\[
v^a_{eff}=v^a_{det}\land y^a_{match}
\land \text{cache-valid}\land \neg\text{synthetic-occlusion}.
\]

SELF 监督掩码为

\[
m^a_{self}=v^a_{det}\land y^a_{match}\land\text{cache-valid}.
\]

注意 synthetic receiver 的 SELF row 仍使用 clean input，所以它的 $m^a_{self}$ 不因 synthetic occlusion
而清零。协同 receiver 监督掩码为

\[
m^a_{coop}=\neg v^a_{eff}\land v^{1-a}_{eff}
\land \text{both-cache-valid}\land\text{pose-valid}.
\]

该定义同时覆盖合成 receiver 和数据中自然存在的“一边失视、一边可见”样本。
两边都失视时不计算 cooperative waypoint loss，在在线系统中交由 BELIEF/SEARCH 状态处理。

### 9.2 候选框匹配标签与 hard negative

GT bbox **只用于生成训练标签**，不会作为模型输入。当 YOLO 候选有效且 perception
cache 有效时，target-match loss 才有效。IoU 不小于 0.30 为正样本，否则为负样本。

训练数据中自然 YOLO 错配少于 1%，因此以 0.25 概率将部分正确候选替换为 hard
negative。候选池包括：

- YOLO 给出的其他 obstacle boxes；
- 将原始候选框保持尺寸但移到图像四角的合成框。

只保留与 GT 的 IoU 小于 0.30 且宽高合法的候选，随机选一个，并将其置信度下限设为
0.5。这使 VERIFY 必须将数值框与 RGB/history 内容绑定，而不能只学习“高置信度候选
通常为真”的捷径。验证集不注入该假阳性，保持真实 YOLO 分布。

### 9.3 Waypoint regression loss

当前配置使用 MSE。对一条轨迹，仅在 `valid_mask` 为 true 的 waypoint 上计算。
设位置和 yaw 逐点误差为

\[
e^{xy}_{n}=\frac{(\hat x_n-x_n)^2+w_y(\hat y_n-y_n)^2}{1+w_y},
\qquad
e^\psi_n=(\hat\psi_n-\psi_n)^2.
\]

无人机 $w_y=1$，机器狗也显式设置 `dog_lateral_loss_weight=1.0`。因此机器狗的
$y$ 是完整监督的未来位置，不是被忽略的横向速度命令。`yaw_loss_weight=1.0`，
`final_waypoint_loss_weight=0.0`，turn/stop sample weight 均为 1，所以当前总体等价于

\[
L_{wp}(\hat\tau,\tau)=
\frac{1}{|\mathcal V|}\sum_{n\in\mathcal V}
\frac{2e^{xy}_n+e^\psi_n}{3}.
\]

模型接口支持 `alpha_xy` 尺度归一化，当前 `alpha_xy=1.0`，因此数值不变。

SELF loss 只在 $m^a_{self}$ 上聚合：

\[
\mathcal L_{self}=
\sum_{a\in\{D,G\}}\lambda_a
\mathbb E_{m^a_{self}=1}
[L_{wp}(\hat\tau^a_{self},\tau^a_{clean})].
\]

当前 $\lambda_D=\lambda_G=1$，且 `normalize_agent_loss_weights=false`，因此两个 Agent loss 直接
相加，不再除以 2。

### 9.4 Best-of-K cooperative waypoint 与 mode loss

对每个 cooperative receiver，计算 $K=4$ 条候选与 cooperative target 的 waypoint loss：

\[
e^a_k=L_{wp}(\hat\tau^a_k,\tau^a_{coop}),\qquad
k^{a\dagger}=\arg\min_k e^a_k.
\]

回归损失仅优化最佳候选：

\[
\mathcal L_{coop}=\sum_a\lambda_a
\mathbb E_{m^a_{coop}=1}[e^a_{k^{a\dagger}}].
\]

mode classifier 使用停止梯度的 best index 作为类别标签：

\[
\mathcal L_{mode}=\mathbb E_{m^a_{coop}=1}
[-\log\mathrm{softmax}(\boldsymbol\ell^a)_{k^{a\dagger}}].
\]

### 9.5 JEPA loss

对被遮挡的 fine-token 位置集合 $\Omega^a_M$，使用 cosine distance：

\[
\mathcal L_{JEPA}=
\mathbb E_{a,j\in\Omega^a_M}
\left[1-
\frac{\hat{\mathbf z}^a_j\cdot\bar{\mathbf z}^a_j}
{\|\hat{\mathbf z}^a_j\|_2\|\bar{\mathbf z}^a_j\|_2}
\right].
\]

如果当前 batch 没有任何遮挡 token，该 loss 返回与计算图连接的零，避免 DDP 因分支完全
未参与而失去梯度图一致性。

### 9.6 Target-belief 与 uncertainty loss

目标 belief 使用 5 维 Smooth-L1：

\[
e^a_{bel}=\frac{1}{5}
\mathrm{SmoothL1}(\hat{\mathbf b}^a,\mathbf b^a),
\qquad
\mathcal L_{belief}=\mathbb E_{m^a_{coop}\land v^a_{pose}}[e^a_{bel}].
\]

不确定性 head 预测 $s^a\in[-5,5]$ 的 log-variance，并使用异方差形式

\[
\mathcal L_{unc}=\mathbb E
[\exp(-s^a)\,\mathrm{stopgrad}(e^a_{bel})+s^a].
\]

这里 belief error 在 uncertainty loss 中被 detach，因此 uncertainty head 学习拟合误差尺度，
不会反向通过 NLL 改变 belief predictor。

### 9.7 Target-match loss

在有效 YOLO 候选上计算 binary cross entropy：

\[
\mathcal L_{verify}=\mathbb E_{v^a_{det}\land v^a_{cache}}
[\mathrm{BCEWithLogits}(f^a_{verify}(h^a_{VERIFY}),y^a_{match})].
\]

推理时使用 $p^a_{match}\ge0.50$ 作为进入可见状态的 VERIFY 条件，但线上 router 另外
使用 0.35 作为退出阈值，形成滞回。

### 9.8 Smoothness regularization

对所有 cooperative candidate 计算二阶差分：

\[
\mathcal L_{smooth}=\mathbb E
\left[
\|\Delta^2\mathbf p_n\|_2^2+
\mathrm{wrap}(\Delta^2\psi_n)^2
\right].
\]

只有连续三个 waypoint 都有效时才计算该项，然后对 mode、时间步和有效 cooperative
receiver 求平均。

### 9.9 Kinematic regularization

对每个 candidate segment，使用样本自身 $\Delta t$ 计算平移速度和 yaw rate：

\[
v_n=\frac{\|\mathbf p_n-\mathbf p_{n-1}\|_2}{\Delta t},\qquad
\omega_n=\frac{|\mathrm{wrap}(\psi_n-\psi_{n-1})|}{\Delta t}.
\]

超限部分使用 squared hinge：

\[
L_{limit}=\mathrm{ReLU}(v_n-v^a_{max})^2+
\mathrm{ReLU}(\omega_n-\omega^a_{max})^2.
\]

对机器狗还额外约束恒曲率弧的侧向残差。设段中点航向为

\[
\psi_{mid}=\psi_{n-1}+\frac12\mathrm{wrap}(\psi_n-\psi_{n-1}),
\]

则位移在身体横向上的分量为

\[
\delta_{lat}=-\sin\psi_{mid}\Delta x+
\cos\psi_{mid}\Delta y.
\]

超过 0.05 m 容差的部分被惩罚：

\[
L_{nonholo}=\mathrm{ReLU}(|\delta_{lat}|-0.05)^2.
\]

虽然 waypoint 0 在 regression `valid_mask` 中为 false，运动学 loss 会强制将其当作有效端点，
以约束从原点到第一个未来 waypoint 的首段运动。

### 9.10 Diversity regularization

对每对 mode $(k,l)$ 的最终 XY 端点，使用 0.25 m margin 的 squared hinge：

\[
\mathcal L_{div}=\frac{1}{\binom K2}
\sum_{k<l}
\mathrm{ReLU}(m_{div}-
\|\hat{\mathbf p}_{k,N-1}-\hat{\mathbf p}_{l,N-1}\|_2)^2,
\quad m_{div}=0.25\,\mathrm m.
\]

该项用于防止 4 条 candidate 全部坍缩到同一终点。

### 9.11 Obstacle loss 的实现边界

当前 obstacle grid 作为 decoder input token 使用，但

\[
\mathcal L_{obstacle}=0.
\]

配置中 `beta_obstacle=0`。如果未提供经验证的 image-to-local-ground 标定投影，代码会
拒绝将该权重设为正值。这是因为图像空间 mask 与 Agent 局部运动 XY 不在同一坐标系，
直接比较将产生错误的“碰撞”监督。

### 9.12 总损失

总损失为

\[
\begin{aligned}
\mathcal L=&
\beta_{self}\mathcal L_{self}
+\beta_{coop}\mathcal L_{coop}
+\beta_{mode}\mathcal L_{mode}
+\beta_{JEPA}\mathcal L_{JEPA}\\
&+\beta_{bel}\mathcal L_{belief}
+\beta_{verify}\mathcal L_{verify}
+\beta_{unc}\mathcal L_{unc}
+\beta_{smooth}\mathcal L_{smooth}\\
&+\beta_{kin}\mathcal L_{kin}
+\beta_{div}\mathcal L_{div}
+\beta_{obs}\mathcal L_{obstacle}.
\end{aligned}
\]

默认权重为：

| 损失 | 权重 |
| --- | ---: |
| SELF waypoint | 100.0 |
| COOP best-of-K waypoint | 100.0 |
| mode classification | 1.0 |
| JEPA | 1.0 |
| target belief | 1.0 |
| target match / VERIFY | 1.0 |
| uncertainty | 0.1 |
| smoothness | 0.1 |
| kinematics | 0.1 |
| diversity | 0.1 |
| obstacle | 0.0 |

模型还会根据 training-time effective visibility 构造一条 routed waypoint 并计算导航指标，但
`routed_per_agent` 本身不加入总 loss；真正的轨迹优化由上述 SELF 和 best-of-K COOP 两项完成。

## 10. 训练数据组织与时间采样

### 10.1 固定 70:30 episode split

训练启动前会严格验证 episode manifest：

- train：2,142 episodes，623,322 frames；
- validation：913 episodes，265,683 frames；
- episode ID 不重叠；
- JSONL 绝对路径不重叠；
- manifest 列出的文件集必须与对应 train/val 目录实际扫描到的 JSONL 文件集完全一致。

任何数量不符、交叉泄漏或 manifest/目录不一致都会在建模前直接终止训练。

### 10.2 Rotating temporal stride

相邻当前帧高度冗余，因此训练集使用 `temporal_stride=3`。对一个 episode 内的样本序列
$[0,1,2,3,\ldots]$，epoch $e$ 仅选择

\[
\{i\mid i\bmod3=e\bmod3\}.
\]

因此 epoch 1/2/3 依次读取 offset 0/1/2，每 3 个 epoch 完整覆盖所有帧一次。
10 个 epoch 等价于 3 次完整覆盖，再训练一次 offset-0 子集。

采样器保持 episode 边界，先在 episode 内做 stride 采样，再将样本分成 512 大小的局部 block
并随机打乱 block 顺序。同一 block 内的相邻帧仍连续，便于 DataLoader worker 利用容量
8192 的 coarse-token LRU cache。在 DDP 中，采样序列会填充到每个 rank 拥有相同样本数，
从而避免同步步数不一致。

验证集使用 `temporal_stride=1`，不做帧下采样。其 receiver corruption 使用基于
`occlusion_seed:episode_id:step_index` 的 SHA-256 确定性采样，保证多次验证使用同一扰动。
验证集被设为 curriculum 最后阶段，但不注入 synthetic false positive。

## 11. 优化与分布式训练实现

### 11.1 Shell 启动链

`sh/train_airground_coop_v3.sh` 执行以下启动流程：

```text
shell launcher
  -> parse --gpu-ids（仅 launcher 的第一组参数）
  -> CUDA_VISIBLE_DEVICES
  -> python -m torch.distributed.run --standalone
  -> train_airground_coop_v3.py --distributed
  -> load/validate V3 YAML
  -> verify fixed train/val split
  -> install V3 dataset/model/loss callbacks into generic trainer
  -> train_airground_v3_common.py::train_airground_v3
```

默认物理 GPU 为 `0,5`，也可以使用
`bash sh/train_airground_coop_v3.sh --gpu-ids 0,1`覆盖。`NPROC_PER_NODE` 默认等于逗号
分隔的 GPU 数量。启动脚本默认禁用 NCCL P2P 和 InfiniBand，启用 async error handling，
并将 CUDA allocator 设为 `expandable_segments:True`。

配置加载器只接受
`architecture: airground_three_stream_cooperative_v3`。未知 YAML key、非 V3 architecture string 或不兼容的
receiver-target/relative-pose 语义标记都会被拒绝。命令行可覆盖 batch size、epoch、learning rate、
worker 数、输出目录、max steps、cache root、receiver 权重和 temporal stride。

### 11.2 默认优化设置

| 项目 | 设置 |
| --- | --- |
| Optimizer | AdamW |
| Peak learning rate | $4\times10^{-5}$ |
| Minimum learning rate | $4\times10^{-6}$ |
| Weight decay | 0.01 |
| Scheduler | 250-step warmup + cosine decay |
| Epochs | 10 |
| Batch per GPU | 48 |
| GPUs | shell 默认 2 张 |
| Gradient accumulation | 2 |
| Effective global batch | $48\times2\times2=192$ |
| Gradient clipping | global norm 1.0 |
| Mixed precision | BF16 autocast |
| TF32 | 启用 |
| Qwen | frozen/eval mode |
| DataLoader workers | 4/rank |
| Prefetch | 2，persistent workers，pinned memory |
| Random seed | 731，DDP 普通模式下每 rank 加 rank offset |

AdamW 只接收 `requires_grad=True` 的参数，未显式覆盖的 Adam 参数使用 PyTorch 默认值
$(\beta_1,\beta_2)=(0.9,0.999)$、$\epsilon=10^{-8}$。BF16 模式下 GradScaler 不启用；代码仍
保留统一 scaler 接口以支持 FP16 分支。

对 optimizer step $s$，学习率为

\[
\eta(s)=
\begin{cases}
\eta_{min}+(\eta_{max}-\eta_{min})\frac{s}{S_w}, & s\le S_w,\\
\eta_{min}+\frac12(\eta_{max}-\eta_{min})
\left[1+\cos\left(\pi\frac{s-S_w}{S-S_w}\right)\right],&s>S_w,
\end{cases}
\]

其中 $S_w=250$，$S$ 是总 optimizer steps。学习率从 $4\times10^{-6}$ 起步，在第 250
个 optimizer step 到达 $4\times10^{-5}$，随后余弦衰减。

每累积 2 个 micro-batch 更新一次参数。DDP 在非更新 micro-batch 上使用 `no_sync()`，
只在真正 optimizer step 前同步梯度。在 `scaler.step()` 前先 unscale，记录 clip 前后总梯度范数，
然后按 1.0 做 global-norm clipping。

### 11.3 DDP 实现

shell 启动后使用 NCCL process group。每个 rank 先构造模型并移到对应 local CUDA device，然后
用 `DistributedDataParallel` 包装。V3 设置：

- `find_unused_parameters=false`；
- `broadcast_buffers=false`；
- 不手动设置 `_ddp_params_and_buffers_to_ignore`，由 DDP 自行过滤冻结参数。

损失聚合函数对“当前 batch 没有某类 receiver”的情况返回 `values.sum()*0`，而不是
断开该分支，使每个 rank 的计算图保持完整。

### 11.4 验证、日志与 checkpoint

默认每 1,000 个 optimizer step 在 rank 0 上评估 16 个 validation batch。其他 rank 在评估前后
进入 barrier，避免 rank 0 离线验证时其他 rank 进入下一次 all-reduce。默认选择指标为
`val_loss_nav = loss_self + loss_cooperative`，当其降低时写入 `best_val.pt` 和
`best_val_metrics.json`。

按 step 保存默认关闭，每个 epoch 结束时保存一份 final checkpoint。Checkpoint 包含：

- model state；
- optimizer state；
- scheduler state；
- AMP scaler state（若启用）；
- epoch 和 global optimizer step；
- 完整训练配置与 architecture/version metadata。

V3 加载器会检查 self/verify/coop query、两个 self planner、两个 cooperative decoder、JEPA predictor、
teacher projector、target-belief/target-match head、obstacle/detection/ROI/absolute-pose/relative-pose projector 等
关键权重前缀，拒绝非 V3 或不符合当前语义 contract 的 checkpoint。默认 `--resume` 会从输出目录中修改
时间最新的 `model_epoch*.pt` 恢复，而不是自动从 `best_val.pt` 恢复。

## 12. 在线推理与可见性路由

### 12.1 无 oracle 推理协议

闭环推理只读取：

- 无人机 RGB；
- 机器狗 RGB；
- 跟踪 instruction；
- 两个 follower 的实时 pose。

目标 GT pose、GT bbox、GT mask 和 target action 均不输入模型。它们可以由共享仿真运行时
读取，但仅用于评估指标和可视化。

对当前两张 RGB，在线执行同一 DINOv3+SigLIP 编码和 YOLO 分割。每个 Agent 的 coarse
token 按 observation timestamp 放入 deque。构造长度 31 的历史时，对目标历史时刻

\[
t_i=t_{now}-(H-i)\Delta t_h
\]

选择不晚于 $t_i$ 的最近 cache entry，从而在实时帧率不稳定时仍保持与训练一致的
历史时间采样。

推理中 `synthetic_occlusion`、`coarse_missing_mask` 和 `fine_missing_mask` 全部显式置零。
训练期 curriculum 不会泄漏到评估；自然失视时模型仍读取 receiver 的真实背景、真实历史、
无效 YOLO detection 和实时双位姿。

### 12.2 YOLO+VERIFY 滞回

对 Agent $a$，进入可见状态的单帧充要条件为

\[
v^a_{enter}=
[v^a_{YOLO}=1]\land[s^a_{YOLO}\ge0.35]
\land[p^a_{match}\ge0.50].
\]

退出可见状态的单帧充要条件为

\[
v^a_{exit}=
[v^a_{YOLO}=0]\lor[s^a_{YOLO}<0.20]
\lor[p^a_{match}<0.35].
\]

连续 2 帧满足 enter 才将稳定状态置为 visible，连续 2 帧满足 exit 才置为 invisible。
中间区间保持前一状态，避免因单帧检测波动在 SELF 和 COOP 之间频繁切换。

### 12.3 四状态轨迹路由

| 稳定可见性 | Agent 轨迹来源 |
| --- | --- |
| 两边均可见 | 两边都使用自己的 SELF trajectory |
| 只有 $D$ 可见 | $D$ 使用 SELF，$G$ 使用 COOP |
| 只有 $G$ 可见 | $G$ 使用 SELF，$D$ 使用 COOP |
| 两边都不可见且不超过 3 帧 | BELIEF：保留最后一条由至少一个可验证视角支持的轨迹 |
| 两边长时不可见 | SEARCH：以确认丢失时航向为中心做 $\pm30^\circ$ 有界旋转搜索 |

BELIEF 状态最多保持 3 帧。若没有可用的历史 navigation trajectory，则暂时使用当前
cooperative trajectory。

进入双边不可见时立即锁定各 Agent 的 search center yaw，而不是等 BELIEF 结束后才锁定。
SEARCH 轨迹的 $x/y$ 全部为 0，yaw 从 0 线性插值到相对当前航向的目标 yaw error。
每个 Agent 在达到中心 $+30^\circ$ 或 $-30^\circ$ 端点的 $1^\circ$ 容差内时独立反向，在两个端点之间
往返，而不会连续旋转 360°。任一 Agent 恢复可见后清空 search center 并回到常规路由。

## 13. Waypoint 到异构实体动作

### 13.1 机器狗非完整 pose projection

模型对机器狗输出完整的未来局部 pose $[x,y,\psi]$，其 $y$ 在训练中是真实位置监督。
但底层机器狗控制器只接受前进和转向，不接受侧移。V3 因此在推理控制空间对每个
waypoint segment 做恒曲率 unicycle projection。

对段位移先转到上一 pose 的 body frame，得到 $(d_x,d_y)$，yaw 变化为
$\delta=\mathrm{wrap}(\psi_i-\psi_{i-1})$。单位前进弧长在 body frame 中的位移基向量为

\[
\mathbf b(\delta)=
\begin{bmatrix}
\sin\delta/\delta\\
(1-\cos\delta)/\delta
\end{bmatrix},
\]

当 $\delta\rightarrow0$ 时使用稳定极限
$[1-\delta^2/6,\delta/2]$。将模型位移投影到该可行方向：

\[
s_i=\frac{\mathbf b(\delta)^\top[d_x,d_y]^\top}
{\mathbf b(\delta)^\top\mathbf b(\delta)}.
\]

将 $s_i$ 累加成新的 control-waypoint $x$ channel，将 control-waypoint $y$ 置零；同时记录模型
XY 与可行弧之间的带符号侧向残差，用于 debug，但不在推理时修改原始 model waypoint。

模型第 $i$ 个 yaw 是第 $i$ 个未来时域的累计 yaw，而父类 inverse controller 按 `yaw/dt`
解释控制 waypoint。因此 projection 将 yaw 重缩放为

\[
\psi^{ctrl}_i=\psi_i\frac{\Delta t_{control}}
{h_i\Delta t_{source}},
\]

使父类的 `psi_ctrl/dt_control` 等于“未来 yaw / 该 waypoint horizon time”。

### 13.2 Inverse fixed-dt controller

评估协议固定 `dt=0.1 s`，waypoint horizon 为 9 个 source steps。根据选中 waypoint 索引 $i$，
将其映射到 source step

\[
h_i=\max\left(1,\mathrm{round}\left(\frac{9i}{N-1}\right)\right),
\qquad T_i=h_i\Delta t_{source}.
\]

无人机原始期望速度为选中 pose 除以 $T_i$。机器狗前进速度由两个延迟索引间的
projected forward displacement 除以对应时间差得到，yaw rate 从投影后 yaw channel 得到。

原始期望速度经过指数平滑。无人机 XY/yaw 平滑系数默认为 0.20/0.25，机器狗
speed/yaw 为 0.30/0.30。对无人机，使用已标定的一阶响应模型反解当前 command：

\[
u^{xy}_t=\frac{v^{des,xy}_t-a_{xy}v^{pred,xy}_{t-1}}{b_{xy}},
\qquad
u^\psi_t=\frac{\omega^{des}_t-a_\psi\omega^{pred}_{t-1}}{b_\psi}.
\]

机器狗的 yaw rate 经过 degree conversion 和 `ground_yaw_gain` 转为转向动作，前进速度乘以
100 转为 Unreal 世界单位。

### 13.3 LLM-verified bbox residual controller

在 inverse controller 计算原始期望速度后，V3 可使用一个只接受“已经过 LLM VERIFY
和滞回确认”的 YOLO bbox 的因果 residual controller。每个 Agent 在 session reset 后记录第一个
可信 bbox height $h_0$，对 height 和 center-x 使用 $\alpha=0.2$ 的 EMA。至少连续 2 帧有效后
才认为 reliable。

相对高度误差为

\[
e_h=\frac{h_0-\bar h_t}{h_0}.
\]

默认 $\pm20\%$ 是死区，在 20%–50% 之间线性增加响应，超过 50% 饱和。bbox 变小
表示目标变远，前进运动加速；bbox 变大表示目标过近，前进运动减速。当当前命令是后退时，
距离响应反转：大框增大后退幅度，小框抑制继续后退。

center-x 超出 Agent-specific 死区时产生有界 yaw residual。无人机的 center 死区为
$[0.40625,0.59219]$，机器狗为更保守的 $[0.45,0.55]$。机器狗在超出死区时会优先
turn-in-place。

不论 bbox residual 是否启用或是否已 reliable，最终物理速度始终强制被限制为：

- drone translation norm $\le2.5\,\mathrm{m/s}$；
- RobotDog forward speed $\in[-2.5,2.5]\,\mathrm{m/s}$；
- yaw rate 按 Agent-specific 评估参数截断。

## 14. 训练算法伪代码

```text
Input:
  fixed train split, cached DINOv3/SigLIP tokens, cached YOLO perception
  frozen Qwen3-0.6B, trainable projectors/queries/heads/decoders

for epoch = 0 ... 9:
    sampler.set_epoch(epoch)
    # also updates dataset curriculum stage and rotating stride offset

    for paired drone-dog batch:
        1. Load 31x4 history tokens and 64 current tokens for both agents.
        2. Load YOLO [cx,cy,w,h,score,valid] and 8x8x4 scene grids.
        3. Remove detector-target channel from both scene grids.
        4. With p=0.25 on training positives, replace a proposal by a low-IoU
           hard-negative box and form the target-match label from GT IoU.
        5. Apply one shared random SE(2) transform to the two follower poses.
        6. If both views/detections/poses are clean and curriculum enables help:
             choose at most one receiver (drone/dog = 0.5/0.5),
             sample ROI_ONLY / CURRENT_FULL / RECENT_FULL / ALL_FULL,
             construct coarse/fine missing masks,
             optionally relocate a full-current-missing receiver,
             construct a bounded feasible recovery target.
        7. Build clean SELF streams for drone and dog.
        8. Pack the two SELF rows as a 2B Qwen forward:
             [text, own history, own current, own detection+ROI, VERIFY, ACT].
        9. Predict p_match and clean SELF trajectories.
       10. Build one joint COOP sequence with masked receiver tokens,
           two absolute pose tokens and two directed relative-pose tokens.
       11. Run the B-row joint Qwen forward.
       12. Predict receiver clean fine latent with conditional JEPA;
           compute target belief and uncertainty.
       13. Decode K=4 agent-specific cooperative trajectory candidates and logits.
       14. Compute SELF, best-of-K COOP, mode, JEPA, belief, VERIFY,
           uncertainty, smoothness, kinematics and diversity losses.
       15. Backpropagate loss / grad_accum_steps.

        every 2 micro-batches:
            DDP synchronize gradients
            unscale (if needed), clip global grad norm to 1.0
            AdamW.step()
            cosine scheduler.step()
            zero_grad(set_to_none=True)

        every 1000 optimizer steps on rank 0:
            evaluate 16 validation batches
            save best_val.pt if val_loss_nav improves

    save one epoch-final checkpoint
```

## 15. 主要张量形状与输出接口

| 输出 | 形状 | 语义 |
| --- | --- | --- |
| `self_waypoints` | $(B,2,10,3)$ | 两条 clean SELF trajectory |
| `target_match_logits` | $(B,2)$ | YOLO 候选目标匹配 logit |
| `target_match_probability` | $(B,2)$ | sigmoid 匹配概率 |
| `cooperative_candidates` | $(B,2,4,10,3)$ | 两 Agent 各 4 条 COOP candidate |
| `cooperative_mode_logits` | $(B,2,4)$ | 候选模态 logit |
| `cooperative_waypoints` | $(B,2,10,3)$ | argmax-mode COOP trajectory |
| `waypoints` | $(B,2,10,3)$ | 按 route visibility 路由后的轨迹 |
| `jepa_prediction_tokens` | $(B,2,64,d_L)$ | receiver fine latent 预测 |
| `jepa_teacher_tokens` | $(B,2,64,d_L)$ | clean EMA teacher latent |
| `jepa_token_mask` | $(B,2,64)$ | JEPA 有效区域 |
| `target_belief` | $(B,2,5)$ | receiver-local 目标 belief |
| `jepa_uncertainty_logit` | $(B,2)$ | belief log-variance |
| `agent_poses` | $(B,2,4)$ | 两枚绝对 pose |
| `directed_relative_pose` | $(B,2,5)$ | 两枚 receiver-local 相对 pose |
| `routing_mode` | $(B,2)$ | SELF/COOP/BELIEF 神经模型内部状态 |

## 16. 训练与推理的关键差异

| 项目 | 训练 | 推理 |
| --- | --- | --- |
| Vision | 读取离线 DINOv3/SigLIP cache | 在线 DINOv3/SigLIP |
| Perception | 读取离线 YOLO cache | 在线 YOLO |
| GT bbox | 只用于 IoU 标签和合成遮挡区域 | 不使用 |
| GT target pose | 只用于 target-belief 监督 | 不使用 |
| Receiver corruption | curriculum 启用 | 全部关闭 |
| Receiver relocation | full-current missing 模式下可启用 | 不使用 |
| Route visibility | GT-IoU 构造的 effective visibility | YOLO+VERIFY+hysteresis |
| Both invisible | 不计算 COOP waypoint loss | BELIEF 后进入有界 SEARCH |
| RobotDog $y$ | 完整位置监督 | 控制前做 non-holonomic arc projection |
| Bbox residual | 不参与训练 loss | 只对稳定 VERIFY 框做有界控制修正 |

## 17. 实现映射

| 方法组件 | 主要代码位置 |
| --- | --- |
| Shell/DDP 启动 | `sh/train_airground_coop_v3.sh` |
| V3 YAML 与超参数 | `config/airground_cooperative_tracking_v3.yaml` |
| split 验证、curriculum、dataset | `train_airground_coop_v3.py` |
| feasible recovery target | `build_feasible_receiver_recovery_target()` |
| rotating stride sampler | `RotatingTemporalStrideDistributedSampler` |
| 总 loss | `forward_airground_v3_loss()` |
| token 构造 | `AirGroundCooperativeVLAV3._encode_agent_streams()` |
| SELF (2B) 流 | `_run_self_flows()` |
| COOP (B) 流 | `_run_cooperative_flow()` |
| directed relative pose | `_directed_relative_pose_features()` |
| JEPA predictor | `ConditionalJEPAPredictor` |
| K-mode decoder | `MultimodalTrajectoryDecoder` |
| 可见性滞回 | `AirGroundVisibilityRouter` |
| 在线 planner/路由 | `eval_airground_coop_v3.py::AirGroundCoopV3Planner` |
| RobotDog arc projection | `_project_robotdog_waypoints_to_nonholonomic()` |
| bbox residual controller | `BBoxMotionController` |
| inverse fixed-dt control | `eval_airground_v3_runtime.py::AirGroundV3RuntimePlanner.waypoints_to_actions()` |

## 18. 可复现命令

只检查 split、vision cache 和 perception cache，不加载 LLM：

```bash
python train_airground_coop_v3.py --dry-run
```

单卡 2-step 调试：

```bash
CUDA_VISIBLE_DEVICES=0 python train_airground_coop_v3.py \
  --debug-max-steps 2 --batch-size 1
```

使用 shell 的双卡默认训练：

```bash
bash sh/train_airground_coop_v3.sh --gpu-ids 0,1
```

覆盖局部参数：

```bash
bash sh/train_airground_coop_v3.sh --gpu-ids 0,1 \
  --batch-size 48 --epochs 10 --lr 4e-5 \
  --num-workers 4 --temporal-stride 3
```

从同一 V3 输出目录的最新 epoch checkpoint 恢复：

```bash
bash sh/train_airground_coop_v3.sh --gpu-ids 0,1 --resume
```

## 19. 论文表述时必须保留的实现边界

1. 视觉前端的当前实现是 **DINOv3 + SigLIP**，不是 DINOv2。
2. SELF-D/SELF-G 的上下文是严格隔离的，但权重共享，并在 batch 维打包为一次 $2B$ forward。
3. VERIFY 是“YOLO 候选是否为指定目标”，不是第二个 visibility detector。
4. GT bbox 与 GT target pose 只在训练期生成标签，从不进入模型推理输入。
5. Synthetic corruption 只改变 COOP receiver row；两条 SELF row 始终 clean。
6. ROI_ONLY 不允许 pose relocation，因为其背景仍对应 clean receiver pose。
7. 刚性变换后的 clean path 只是 recovery reference，不是直接监督标签。
8. JEPA teacher 只对 clean RGB fine latent 建模，不重建像素也不复制 YOLO target channel。
9. Obstacle grid 参与 decoder attention，但由于缺少标定的图像到局部地面投影，当前 obstacle loss 严格为 0。
10. RobotDog waypoint $y$ 是完整位置监督，仅在推理控制空间通过 unicycle projection 转为前进/转向。
11. 两边同时失视时，神经模型内部可标记 BELIEF，完整的“三帧 hold 后有界 SEARCH”是在有状态在线 planner 中实现的。
12. 当前 COOP query 的固定因果顺序是 Drone 后接 RobotDog，因此应将“对称”限定为“支持两个协同方向且具有 Agent-specific decoder”。

## 20. 可直接用于论文的方法概括

AirGround-Coop V3 将异构空地协同跟踪分解为两条 Agent 隔离的自主跟踪流和一条联合
协同恢复流。冻结的 DINOv3-SigLIP 前端将双视角 RGB 历史编码为多尺度 token，YOLO
则提供人物候选 ROI 与类别无关场景网格。SELF-D 和 SELF-G 沿 batch 维打包进入共享权重
的冻结 Qwen3，但各自仅读取本 Agent 视觉与检测，从结构上消除正常跟踪时的跨 Agent 运动
绑定。每条 SELF row 按因果顺序放置 VERIFY 与 ACT query，分别执行目标候选校验与 clean
局部轨迹预测。COOP 流联合两个视角、两枚绝对位姿和两枚 receiver-local 有方向相对
位姿，通过 conditional JEPA 在 EMA teacher 监督下恢复缺失 receiver 的 clean fine-grid latent，
再由 Agent-specific Transformer decoder 产生四种可行轨迹候选及其 mode score。

为了学习从部分遮挡到完全失视的连续恢复能力，训练期对最多一个 receiver 施加由 ROI-only、
current-full、recent-history-full 到 all-history-full 的分阶段 token corruption。仅当当前观测完全缺失时，
才允许有界的 receiver pose relocation；此时模型不直接回归刚性变换后的跳变轨迹，而是从
新 receiver 原点出发，在 Agent-specific 速度、yaw-rate 和非完整约束下生成逐步追赶 reference
的可执行 recovery target。训练目标综合 clean SELF regression、best-of-K cooperative regression、mode
classification、JEPA latent prediction、target belief、target verification、uncertainty、平滑性、运动学和模态
多样性损失。

在线推理中不施加人工 corruption。带滞回的 YOLO+VERIFY router 在两边可见时保持 SELF，
在一边失视时只对 receiver 启用 COOP，在两边失视时先短时保留最后可信轨迹，随后切换为相对
丢失航向 $\pm30^\circ$ 的旋转搜索。最终 waypoint 经过机器狗 non-holonomic pose projection、
inverse-fixed-dt 动力学反解和只接受 LLM-verified bbox 的有界 residual correction，转换为无人机和机器狗
的实体动作。
