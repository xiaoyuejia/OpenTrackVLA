# AirGround-Coop V3 pipeline figure

**Figure X. Detailed pipeline of AirGround-Coop V3.** A shared frozen visual
front-end converts the aerial-drone and ground-robot observations into coarse
history tokens, fine current-frame tokens, a YOLO proposal/ROI token, and a
semantic scene grid. Two agent-isolated causal rows share Qwen3-0.6B parameters
but not attention context; their VERIFY and ACT queries respectively estimate
whether the YOLO proposal matches the tracked person and predict clean self
trajectories. A separate cooperative forward jointly attends to both streams
and two follower-pose tokens. During training, exactly one eligible receiver is
visually masked and pose-perturbed while the source remains clean. A conditional
JEPA predictor reconstructs the receiver's clean latent representation under an
EMA teacher, and two agent-specific multimodal decoders predict four trajectory
modes. Cooperative waypoint targets preserve the clean future world path but
are expressed in the perturbed receiver frame. At inference, no synthetic
corruption is applied. A per-session YOLO+LLM hysteresis router selects SELF,
COOPERATIVE, short-horizon BELIEF, or bounded yaw SEARCH trajectories before
the embodiment-aware waypoint-to-action controller produces drone and RobotDog
commands.

中文图注建议：

**图 X. AirGround-Coop V3 的详细流程。** 双 Agent 视觉首先编码为历史粗粒度
token、当前帧细粒度 token、YOLO 候选/ROI token 与场景网格。两条共享参数但
attention context 完全隔离的 self row 分别通过 VERIFY 和 ACT query 完成目标候选
校验与自身轨迹预测；独立 cooperative forward 则联合读取双视角和双 follower
pose。训练时只破坏一个 receiver 的视觉与 pose，利用 conditional JEPA 对齐 clean
EMA teacher，并通过 Agent 专属多模态 decoder 预测四模态协同轨迹。推理时不加入
人工破坏，由带迟滞的 YOLO+LLM 路由器在 SELF、COOPERATIVE、BELIEF 和 SEARCH
之间选择轨迹，最后经具身控制适配得到无人机与机器狗动作。
