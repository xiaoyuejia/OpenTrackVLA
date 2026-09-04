# Compact paper-style pipeline caption

**Figure X. Overview of AirGround-Coop V3.** A shared frozen visual front-end
encodes aerial-drone and ground-robot video streams into typed visual tokens,
YOLO proposal/ROI tokens, scene-grid tokens, two absolute follower-pose tokens,
and two directed receiver-local relative-pose tokens.
The numbered reading order is: (1) dual-view tokenization, (2) the three
information flows through two shared-weight LLM forwards, (3a) clean self
outputs and (3b) cooperative receiver recovery, and (4) online routing and
control. The SELF-D and SELF-G rows are packed into one 2B forward but retain
separate causal attention contexts; COOP-DG uses a second joint B-row forward.
AirGround-Coop V3 contains two agent-isolated self rows and one joint
cooperative context. In each self row, the causal VERIFY token determines
whether the YOLO person proposal matches the tracked target, while the later
ACT token predicts the agent's clean local trajectory. The cooperative branch
jointly attends to both views and poses. During training, one eligible receiver
is replaced by learned missing-view tokens and receives a pose perturbation; a
conditional JEPA predictor recovers its clean latent representation under an
EMA teacher, and an agent-specific multimodal decoder predicts four trajectory
modes. At inference, no artificial corruption is applied. A stateful
YOLO+VERIFY router selects SELF, COOPERATIVE, short-term BELIEF, or bounded
SEARCH trajectories before the waypoint controller produces embodied drone
and RobotDog actions.

**图 X. AirGround-Coop V3 总体框架。** 冻结视觉前端将无人机和机器狗的视频流
编码为带类型的视觉 token、YOLO 候选/ROI token、场景网格 token、两枚绝对
pose token，以及两枚有方向的 receiver-local relative pose token。V3
包含两条 Agent 隔离的 self row 和一个双视角联合 cooperative
context。self row 内因果顺序在前的 VERIFY token 判断 YOLO 人物候选是否为跟踪
目标，ACT token 则预测各 Agent 的 clean 局部轨迹。训练时选择一个 receiver，
将其 cooperative 视觉替换为 missing token 并扰动 pose；conditional JEPA 在 EMA
teacher 监督下恢复 clean latent，Agent 专属多模态 decoder 输出四种协同轨迹模式。
推理时不加入人工破坏，由带时序迟滞的 YOLO+VERIFY 路由器在 SELF、COOPERATIVE、
短时 BELIEF 和有界 SEARCH 之间选择轨迹，最后转换为无人机与机器狗动作。
