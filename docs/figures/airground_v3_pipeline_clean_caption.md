# AirGround-Coop V3 clean main-figure caption

**Figure X. AirGround-Coop V3 pipeline.** Dual aerial and ground video streams
are converted into typed visual, YOLO/ROI, instruction, and follower-pose
tokens. The frozen Qwen3 backbone implements two attention-isolated self rows
and one joint cooperative row using shared weights. VERIFY and self action
heads predict target-match probabilities and clean self trajectories. The
cooperative branch recovers a missing receiver representation with conditional
JEPA and predicts four trajectory modes. At inference, a temporally stable
YOLO+VERIFY router selects self, cooperative, belief, or search behavior before
waypoints are converted to embodied actions. The dashed lower band is
training-only: one receiver is visually masked and pose-perturbed, while clean
EMA latents and receiver-frame waypoints provide representation and trajectory
targets.

**图 X. AirGround-Coop V3 流程。** 无人机与机器狗视频流首先转换为带类型的视觉、
YOLO/ROI、指令和双 follower pose token。冻结的 Qwen3 主干以共享权重执行两条
attention 隔离的 self row 和一条双视角联合 cooperative row。VERIFY 与 self action
head 分别输出目标匹配概率和 clean self 轨迹；cooperative 分支利用 conditional JEPA
恢复失视 receiver 表征，并预测四种协同轨迹模式。推理时，带时序迟滞的
YOLO+VERIFY Router 在 self、cooperative、belief 与 search 行为之间选择轨迹，再转换
为具身动作。底部虚线区域只在训练时启用：一个 receiver 的视觉被 mask、pose 被扰动，
clean EMA latent 和 receiver-frame waypoint 分别提供表征与轨迹监督。
