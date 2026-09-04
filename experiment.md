# 空地异构双智能体跟踪：对比实验实施计划

> 文档性质：基于当前代码仓库、`worklog/`、现有训练/评估产物和
> `/data/hdt/code_raw/` 第三方源码重新审计后的实施计划。  
> 当前仓库：`/data/hdt/newtrackvla修改/newtrackvla_base_yh_clean`  
> 审计日期：2026-08-24  
> 本文只确定实验实施顺序与公平性合同，不在此阶段重构或删除现有代码。

---

# 1. 审计后的总判断

本项目不是从零开始搭建双智能体 EVT。当前仓库已经有一条可训练、可闭环评估的
canonical AirGround-Coop V3 主线，并已有完整 checkpoint 和一轮 150 episode 结果。
后续重点应从“重新设计统一项目骨架”改为：

1. 冻结现有 V3 的数据、动作和严格评估协议；
2. 补齐能直接回答论文问题的公平 baseline；
3. 增加与 V3 实际模块一一对应的消融；
4. 修正 evaluator 的指标命名和聚合口径；
5. 在干净的新数据上完成第二阶段训练和泛化实验；
6. 最后再做高成本通用 VLA、RL 和通信延迟扩展。

禁止按原网页方案新建一套平行的 `airground_evt/` 环境、控制器和 evaluator。现有
`eval_unrealzoo_multi_agent.py`、V3 runtime、fixed-dt inverse controller、manifest
launcher 和结果格式应继续作为唯一运行主线，通过 policy adapter 和有限的 evaluator
扩展接入其他方法。

---

# 2. 当前仓库的真实状态

## 2.1 Canonical 模型与运行入口

当前唯一支持的主模型是：

```text
architecture = airground_three_stream_cooperative_v3
```

| 能力 | 当前入口 |
|---|---|
| 模型 | `model_airground_coop_v3.py` |
| 训练 | `train_airground_coop_v3.py` |
| 训练公共运行时 | `train_airground_v3_common.py` |
| 在线 planner / 路由 | `eval_airground_coop_v3.py` |
| 推理公共运行时 | `eval_airground_v3_runtime.py` |
| 模型服务 | `eval_airground_coop_v3_server.py` |
| UnrealZoo 闭环 | `eval_unrealzoo_multi_agent.py` |
| 主配置 | `config/airground_cooperative_tracking_v3.yaml` |
| 训练启动 | `sh/train_airground_coop_v3.sh` |
| 严格评估启动 | `sh/eval_airground_coop_v3.sh` |
| 24 小时速度模式 | `sh/eval_airground_coop_v3_24h_3gpu.sh` |
| 指标聚合 | `tools/calculate_unrealzoo_metrics.py` |

V3 的实际结构不是原计划中的通用“shared memory + cross-agent fusion”抽象，而是：

```text
Drone SELF（只读无人机视觉）
Dog SELF（只读机器狗视觉）
Joint COOPERATIVE（双视觉 + 双 pose + directed relative pose）
YOLO proposal + LLM VERIFY
SELF / COOPERATIVE / BELIEF / SEARCH 在线路由
JEPA masked receiver token prediction
K-mode cooperative trajectory decoder
target-belief / uncertainty heads
```

因此后续消融必须围绕这些已实现模块命名，不能再假定仓库中存在独立的
“Grounding Head / Shared Target Memory / Shared Planner”开关。

## 2.2 当前固定训练合同

```text
train episodes = 2142
val episodes   = 913
history        = 31 frames
dt             = 0.1 s
n_waypoints    = 10
action_dims    = 3
waypoint[0]    = [0, 0, 0]，结构原点，不参与预测损失
waypoint[1:10] = 0.1 s ... 0.9 s 的 9 个未来局部 pose
agent order    = [drone, robotdog]
pose/action    = local [x_m, y_m, delta_yaw_rad]
label source   = recorded_pose_fixed_dt
```

机器狗输出中的 `y` 是未来局部位置监督，不是可执行横向速度。闭环执行前必须经过
`v3_nonholonomic_projection`，最终物理横向速度恒为 0。无人机与机器狗均使用现有
`inverse_fixed_dt` 控制链和 2.5 m/s 上限。

## 2.3 已有 checkpoint 与结果

当前训练目录：

```text
output/airground_three_stream_cooperative_v3_receiver_target_qwen06b/
```

目录中已有 10 个 epoch checkpoint、`best_val.pt`、训练日志和配置快照。当前 150 条
闭环结果实际使用 epoch 10 final checkpoint：

```text
output/eval_airground_coop_v3_receiver_target_125/metrics.csv
```

| 指标 | 数值 |
|---|---:|
| Episodes | 150 |
| Success | 128/150 = 85.33% |
| Joint tracking rate | 81.46% |
| Drone tracking rate | 84.71% |
| RobotDog tracking rate | 94.02% |
| Collision / human collision | 0% / 0% |
| 平均端到端 FPS | 0.568 |

这些结果证明主链可运行，但还不是最终论文主表，因为尚无同 manifest、同 evaluator 的
主要 baseline 结果，也没有 paired confidence interval。

## 2.4 当前 evaluator 的真实指标口径

当前 `tools/calculate_unrealzoo_metrics.py` 计算的是本项目自定义双智能体指标：

```text
SR        = episode success flag 的均值
JointTR   = 各 episode joint_following_rate 的宏平均
DroneTR   = 各 episode drone_following_rate 的宏平均
DogTR     = 各 episode robotdog_following_rate 的宏平均
CR        = episode collision flag 的均值
```

官方 TrackVLA `analyze_results.py` 中的 TR 是
`sum(following_steps) / sum(total_steps)`，即跨 episode 微平均。当前宏平均 JointTR
不能直接命名为官方 EVT-Bench TR。论文和代码中先使用：

```text
AG-SR
Joint-TR-macro / Drone-TR-macro / Dog-TR-macro
AG-CR
```

同时增加对应 `*-TR-micro`。只有在人工小 episode 上与 TrackVLA 官方实现逐项对齐后，
才允许额外报告 `EVT-SR / EVT-TR / EVT-CR`。

`sh/calculate_eval_metrics.sh` 仍硬编码 `--expected-episodes 78`，与当前 100/150 条协议均
不一致。正式实验前必须改为显式参数或直接调用 Python 聚合入口。

## 2.5 三档评估协议

### P0：Smoke

```text
1 episode -> 功能检查
10 episodes / 3 scenes -> 回归和速度检查
```

### P1：论文公平主协议

```text
manifest       = manifests/eval_manifest_100.json
episodes       = 100
scenes         = 26
dt             = 0.1 s
policy stride  = 1（10 Hz）
mask stride    = 1
RGB            = 640 x 480，除非所有方法共同冻结为另一分辨率
target motion  = recorded action replay
initial pose   = recorded follower poses
controller     = inverse_fixed_dt
```

选 100 条作为主协议，是因为现有 CNN-LSTM 和 OpenPI adapter 都围绕 scene-covered
val100 开发，跨方法接入成本最低。

### P2：扩展鲁棒性协议

```text
manifest = manifests/total.json
episodes = 150
scenes   = 26
```

只有同一比较表中的所有方法都完成这 150 条后，才形成扩展表。

`sh/eval_airground_coop_v3_24h_3gpu.sh` 使用 2.5 Hz policy、2 Hz mask 和零阶保持，
属于 throughput profile。它可用于大规模评估和吞吐报告，但不能与 P1 的严格 10 Hz
TR/visibility 结果直接比较。

---

# 3. 数据状态与使用决策

## 3.1 Legacy data7_8：第一阶段公平比较数据

第一阶段所有已经适配或即将适配的学习方法继续使用当前固定 2142/913 split：V3、
CNN-LSTM BC 和 OpenPI adapter 已围绕它训练或开发，可以先补齐 baseline，避免把数据变化
和模型变化混在一起。913 validation 与 100/150 条闭环测试清单不得混称为同一集合。

## 3.2 data_dt：已完整处理，但不是当前默认训练集

```text
raw episodes = 2447
train         = 1713
eval          = 734
scenes        = 23
JSONL/frame/cache/instruction = complete
```

它可作为独立数据扩展实验，但不能无记录地替换当前 split。若使用，必须创建新配置、
checkpoint 目录和 run id。

## 3.3 data_arr：当前不能直接用于正式训练

`worklog/2026-08-22_data_arr_final_reaudit_and_quarantine.md` 记录的当时状态是：

```text
当时审计 episode       = 9802
visible/bbox 冲突隔离  = 355
当时剩余 core          = 9447
```

但这不是当前文件系统的最终状态。隔离目录中的 `rollback_completed.json` 表明，这 355 条
已于 2026-08-22 15:35 全部恢复到 source root。对当前 355 条元数据重新只读统计仍得到：

```text
agent frames                  = 213000
invalid bbox                  = 5505
target_visible + invalid bbox = 5288
```

因此这 355 条当前仍不能进入 target-match/ROI 正式监督。

更早隔离的 376 条 bbox 全失败 episode 后来生成了 fine-tuned YOLO 修复版 info，并通过
`stt_camera2_yolobbox` 视图复用隔离区视频。这 376 条是显式 pseudo-bbox 数据，不等价于
exact GT，必须单独标 `bbox_source=yolo_finetuned` 或独立 source group。

当前 source/processed 数量关系是：

```text
当前主 source（含恢复的 355） = 9802
YOLO-repaired 额外视图         = 376
processed episodes             = 10178
dt                             = 2433
stt                            = 7745
```

所以当前 10,178 条 processed 数据混有 355 条未解决冲突样本和 376 条 pseudo-bbox 样本，
但二者性质不同，不能再简单写成“731 条均已隔离坏数据”。正式可用规模应分两档：

```text
exact/core manifest       = 9447（排除未解决的 355）
core + pseudo-bbox        = 9823（9447 + 376，使用 source/quality mask）
unresolved exclusion list = 355
```

该批 JSONL 使用 `n_waypoints=8`，而当前 V3、checkpoint、控制器和跨方法比较按“原点 +
9 个未来点”的 10 点合同冻结，不能直接混用。

截至本次审计，视觉缓存也未完成：

```text
目标 cache files      = 11,028,000
当前约                = 5,579,478
done chunks           = 24 / 57
活动 precache worker  = 0
```

此前错误生成的 `vision_cache_complete.marker` 不能作为完成证据。

## 3.4 data_arr 正式接入门禁

2026-08-25 已冻结 representation-independent v1 split：

```text
manifest root = manifests/data_arr_7_1_2_v1
core train    = 6598 (69.842%)
core val      =  952 (10.077%)
core test     = 1897 (20.080%)
lostmid       = 305/305 train
loststart     = 913/913 test
pseudo bbox   = 376 auxiliary-train-only
unresolved    = 355 excluded
```

同一 source batch 或同一 scene/target-trajectory signature 不跨 split。现有 8-waypoint
manifest 已落盘；10-waypoint 使用完全相同的 episode key/split，当前标记 pending rebuild。
后续门禁在该固定 split 上继续执行：

1. 不覆盖旧 8-waypoint 产物，生成新的 manifest-driven 10-waypoint processed root；
2. 统一生成 `n_waypoints=10`、`dt=0.1`、`history=31` 的 JSONL；
3. 不再生成 53 GiB 聚合 `dataset.json` 作为训练依赖，使用 JSONL + manifest；
4. 对 future waypoint、bbox/visibility、帧路径和 cache key 做完整性检查；
5. 复用已生成的有效 cache，只为正式 manifest 引用的缺项补齐；
6. 每张 JPG 必须同时存在 `vfine` 和 `vcoarse`，并抽样加载验证；
7. 记录 FP32/BF16 cache 来源，不能在一个正式 run 中静默混用；
8. 通过 1000 sample dataset smoke 和 2-step train smoke 后才启动长训练。

第二阶段先以 9,447 core 为公平主训练集；376 pseudo-bbox 只进入单独的数据增益实验。
所有同表方法必须共同重训或明确标记训练数据不同。

---

# 4. 第三方仓库盘点

所有第三方方法保留在 `/data/hdt/code_raw/`，主仓库只写 adapter。正式实验必须记录
remote、commit、dirty status、权重来源和许可证。

| 方法 | 本地路径 | 当前状态 | 结论 |
|---|---|---|---|
| OmTrackVLA | `/data/hdt/code_raw/OmTrackVLA` | 官方源码完整；主仓库是其深度改造 fork | 高优先级 tracking-VLA baseline |
| Offline EVT | `/data/hdt/code_raw/Offline_RL_Active_Tracking` | 原 CQL 可运行；本任务已实现的是 CNN-LSTM BC | 必须改名，不能把 BC 写成 Offline RL |
| OA-VAT | `/data/hdt/code_raw/oavat/OA-VAT` | 源码及 YOLOE/ORTrack/DINO 权重齐；原控制为 UAV/旧 Gym-UnrealCV | 可做 UAV 模块化强 baseline |
| D-VAT | `/data/hdt/code_raw/d-vat` | 已补下载；UE4.26/Windows、dt=0.05、输出 thrust/angular velocity；无本地权重 | 高风险 optional |
| AD-VAT | `/data/hdt/code_raw/active_tracking_rl` | 这是 AD-VAT，不是 D-VAT | 旧 RL，非主表必做 |
| OpenPI | `/data/hdt/code_raw/openpi` | π0.5 adapter、base 权重、Dog 2-epoch checkpoint 和单条闭环已有 | 可行但尚非完整双 agent baseline |
| OpenVLA | `/data/hdt/code_raw/openvla` | 仅原始代码 | 低优先级 |
| OpenVLA-OFT | `/data/hdt/code_raw/openvla-oft` | 已补下载，尚无本任务 adapter | π0.5 完成后再做 |
| ReferTrack | `/data/hdt/code_raw/referTrack` | 评估代码已发布；forward view、YOLO+ByteTrack、TVBI、Qwen3-4B | 比 OFT 更贴近本任务，优先单体适配 |
| TrackVLA | `/data/hdt/code_raw/TrackVLA` | 已补下载 | 用于官方 EVT 口径核对 |
| HIEVT | `/data/hdt/code_raw/HIEVT` | goal-conditioned offline tracking 代码存在 | recovery 扩展候选 |
| VLM Assistant | `/data/hdt/code_raw/VLM_Assistant` | recovery/reflection 代码和权重存在 | recovery 扩展候选 |

本次审计冻结的第三方源码版本：

| Repository | Commit |
|---|---|
| OmTrackVLA | `e9cb1fbd57f8` |
| Offline_RL_Active_Tracking | `00639410e39e` |
| OA-VAT | `cea1f6b72a9a` |
| D-VAT | `b08161a6e169` |
| AD-VAT | `adcd2d9b1c3c` |
| OpenPI | `15a9616a0094` |
| OpenVLA | `c8f03f48af69` |
| OpenVLA-OFT | `e4287e94541f` |
| TrackVLA | `c69f9d98a73c` |
| ReferTrack | `c12e53e3a5ca` |
| VLM Assistant | `3883c37476b1` |

其中 Offline EVT、OA-VAT、OpenPI 等目录已有本任务改动，是 dirty worktree；上表只记录
upstream 基点。正式 run 还必须额外保存 dirty diff hash，不能只保存 HEAD。

Offline BC 和 OpenPI 的现有 adapter 仍有绝对路径指向旧仓库
`/data/hdt/newtrackvla修改/newtrackvla_base_yh`，而不是当前 `_clean` 根目录。复用前必须
参数化 `NEWTRACK_ROOT`，防止实际运行旧代码。

主配置的 `perception_cache_root` 也指向旧 `_base_yh`。必须先核对该 cache 与当前 JSONL、
resize 和 YOLO 配置完全一致，再显式保留或迁移；不能只按目录存在就视为兼容。

---

# 5. 冻结的科学问题

## Q1：协同分支是否优于两个隔离 SELF 策略？

```text
V3 Full vs V3 Dual SELF-only
```

两者使用同一 checkpoint、YOLO、controller 和 manifest，唯一差异是 receiver 失视时是否
允许路由到 COOPERATIVE。这是当前成本最低、因果最清晰的 cooperation 证据。

## Q2：收益是否只是双视角覆盖？

```text
Drone SELF-only
Dog SELF-only
Dual SELF-only
V3 Full

multi-view coverage gain = Dual SELF-only - Best Single SELF
cooperation gain         = V3 Full - Dual SELF-only
```

单 Agent 结果必须使用 active-agent evaluator，不能继续用“双方最终都 following 才成功”
的 joint success 规则。

## Q3：V3 是否优于经典与轻量学习 baseline？

第一阶段比较 Detector+PID、Independent CNN-LSTM BC、OmTrackVLA/V3 SELF 和 V3 Full。
OA-VAT 完成 adapter 后作为现代模块化非 VLA 强 baseline。

## Q4：V3 是否主要改善单边失视、恢复与交接？

从逐帧 GT mask 和路由日志定义：source visible + receiver lost、进入 COOPERATIVE、
receiver reacquired、返回 SELF。UAV->Dog 与 Dog->UAV 必须分开统计。

## Q5：通用 VLA 或更强单体 tracking VLA 能否解释收益？

优先顺序：ReferTrack -> π0.5/OpenPI -> OpenVLA-OFT。ReferTrack 更贴近动态 tracking，
π0.5 用于“通用 manipulation VLA 能否迁移”的架构对照。

## Q6：跨 Agent 信息缺失或陈旧时是否仍稳定？

当前 V3 是集中式双视角推理，没有显式网络通信模块。在实现可复现 channel 前，只能称为
`cross-agent information dropout/staleness`，不能声称真实通信带宽或分布式部署鲁棒性。

---

# 6. 冻结的实验矩阵

## 6.1 第一阶段论文必做

| ID | 方法 | 平台 | 当前状态 | 角色 |
|---|---|---|---|---|
| B0 | YOLO + PID/IBVS | UAV、Dog | adapter 已实现；UE reset 冒烟待重试 | 经典非学习下界 |
| B1 | Independent CNN-LSTM BC | UAV + Dog | 已训练；val100 中途 SIGKILL | 轻量视觉历史策略 |
| B2 | V3 Drone SELF-only | UAV | 模型已有；需 active-agent evaluator | 单体空中能力 |
| B3 | V3 Dog SELF-only | Dog | 模型已有；需 active-agent evaluator | 单体地面能力 |
| B4 | V3 Dual SELF-only | UAV + Dog | 需 runtime route override | 双视角无协同核心 baseline |
| B5 | OmTrackVLA-0.6B Independent | UAV/Dog | 源码已有；需 adapter | 外部 tracking VLA |
| B6 | OA-VAT-UAV | UAV | 权重齐；需感知/控制 adapter | 模块化强 baseline |
| Ours | AirGround-Coop V3 Full | UAV + Dog | checkpoint 与 150 条结果已有 | 完整协同方法 |

时间有限时最低不可删除集合：`B0 + B1 + B2 + B3 + B4 + Ours`。

## 6.2 第二阶段高价值扩展

| ID | 方法 | 优先级 | 进入条件 |
|---|---|---:|---|
| E0 | ReferTrack single-agent | A | 权重 strict-load、视频推理和 waypoint adapter 通过 |
| E1 | π0.5 RobotDog | A | 当前 checkpoint 完成 val100；不得用 recorded drone 冒充双体结果 |
| E2 | π0.5 UAV + Dog independent | A/B | UAV 分支训练完成，双方均真实闭环控制 |
| E3 | Estimated Pose-Assisted | B | 相机标定和 bbox-ray/ground projection 误差验证完成 |
| E4 | Oracle GT Pose Recovery | 分析上界 | 明确标 Oracle，不进入公平排名 |
| E5 | OpenVLA-OFT | B | π0.5 完成后仍需第二类通用 VLA |

## 6.3 暂不作为主表必做

```text
Original Offline EVT CQL、D-VAT、AD-VAT、HIEVT、VLM Assistant、OpenVLA vanilla
```

当前数据没有原生 RL reward；D-VAT 同时存在环境、动力学和动作语义迁移；HIEVT/VLM
Assistant 更适合 recovery 补充；vanilla OpenVLA 不如 π0.5/OFT 适配连续轨迹。

---

# 7. Baseline 实施合同

## 7.1 B0：YOLO + PID/IBVS

首版复用主仓库当前在线 YOLO proposal：

```text
RGB -> YOLO person proposal -> bbox center/scale error -> PID -> local motion
```

Dog：`yaw_rate <- horizontal error`，`forward <- bbox scale error`，`lateral=0`。
UAV：冻结一种 horizontal yaw/lateral 方案，forward 来自 scale error，vertical 来自
vertical center error。丢失时使用固定有界搜索，不得读取 GT。增益只在开发集调参。
`PID-GT-BBox Oracle` 可作控制上界，但不能进入公平主排名。

当前实现位于 `/data/hdt/code_raw/airground_yolo_pid_ibvs`，已完成：双 Agent 独立 PID、
YOLO11m-seg 在线检测、失视有界搜索、10 点 waypoint adapter、统一 inverse controller、
val100 launcher 和无 GT CLI 门禁。单元/接口测试为 11 passed，val100 dry-run 为 26 scenes /
100 episodes，真实本地 YOLO 权重 smoke 已通过。首次 AncientRuins 闭环尝试在 UE
`set_population()` 阶段发生 UnrealCV socket 断连，尚未执行第一帧 PID，因此状态是
“实现完成、闭环环境待重试”，不能填入正式指标。

## 7.2 B1：Independent CNN-LSTM BC

两套 10-epoch checkpoint 已存在：

```text
/data/hdt/code_raw/Offline_RL_Active_Tracking/
trained_models/data7_8/formal_gpu3/drone_bc/best_val.pt
trained_models/data7_8/formal_gpu3/robotdog_bc/best_val.pt
```

模型只读各自 31 帧 RGB，不读 bbox、文本、pose、对方视角或通信。正式名称必须是
`Independent CNN-LSTM Behavior Cloning`，禁止写成 Offline RL/CQL。当前 val100 在
第 14/26 个场景时 evaluator 被 SIGKILL；先修 root/launcher 资源问题，再 resume，最终
强制恰好 100 个 episode。

## 7.3 B2/B3/B4：V3 SELF 系列

模型 forward 已返回 `self_waypoints` 和 `cooperative_waypoints`，B4 不需重训。给 router
增加可记录的 `route_policy = full_v3 | self_only`。`self_only` 中双方始终执行各自 SELF；
丢失时只允许同一 bounded search，禁止 cooperative trajectory、对方 pose、target belief
或对方视觉。单元测试需证明改变 Dog 输入不改变 Drone SELF action，反向亦然。

B2/B3 中 inactive follower 使用 HOLD，并从 active-agent success、collision 和 TR 分母中
排除；不能用 joint success 评价单体。

## 7.4 B5：外部 OmTrackVLA

1. strict-load 官方 0.6B checkpoint；
2. 从当前 JSONL 构造单 Agent 31 帧 history；
3. 显式映射原 waypoint horizon/尺度到当前 10 点合同；
4. 离线检查 ADE/FDE；
5. 接入相同 inverse controller；
6. 执行 1/10/100 episode。

转换只写在 adapter，不能修改 evaluator 或环境。

## 7.5 B6：OA-VAT-UAV

本地已有 YOLOE、ORTrack、DINOv3 和 MobileCLIP 权重。优先复用 proposal、temporal
tracking、re-ID、confidence-aware KF 和 PID/recovery。原控制器面向 Tello/UAV，首版只做
UAV，不应写成 Dog-first。禁用其独立 Gym-UnrealCV 统计，统一由当前 evaluator 计分。
Diffusion Policy rescue 作为可选变体。

## 7.6 ReferTrack

ReferTrack 使用 forward RGB + YOLO/ByteTrack catalog -> referring target slot -> TVBI bbox
history -> 8 点 waypoint。顺序为：权重 strict-load -> 离线视频 -> 8 点控制时域 adapter ->
Dog smoke -> UAV smoke -> val100。其 Habitat GT bbox 只可用于可视化/evaluator，决策只能走
在线 YOLO+ByteTrack。

## 7.7 π0.5 / OpenPI

当前已有 OpenPI 环境、官方 base 权重转换、task adapter、RobotDog 2-epoch checkpoint 和
单条 recorded-drone 诊断。它尚不能进入双智能体主表。必须先完成 RobotDog active-agent
val100，再训练 UAV 分支，最后双方同时真实闭环。报告两个 3.616B 策略参数之和、两次
前向延迟和 GPU hours，并增加 shared-weight/Siamese 版本或明确额外容量。

首版只用双当前 RGB + pair-centered pose；检测输入版本单独命名，不使用 GT bbox、target
pose、visibility 或 perception cache。

## 7.8 Pose-Assisted

当前没有可信 image-to-ground target projection；`beta_obstacle=0` 也源于图像 mask 与局部
waypoint 尚未标定。Estimated Pose-Assisted 进入主表前必须固定相机内外参、定义 bbox ray
与地面/目标高度模型、在 GT pose 上验证 median/P95 误差，并只发送 estimate + confidence +
timestamp。Oracle GT Pose 必须单列。

---

# 8. Ours 的核心消融

## 8.1 无需重训的 runtime 消融

| ID | 消融 | 唯一变化 | 目的 |
|---|---|---|---|
| A0 | Full V3 | 当前默认 | 完整模型 |
| A1 | Dual SELF-only | 禁止 COOPERATIVE/BELIEF target-belief 路径 | cooperation gain |
| A2 | YOLO-only routing | VERIFY 不参与可见性判定 | LLM target verification |
| A3 | No belief hold | 双失视立即进入同一 bounded search | 短期 belief hold |

输出 setup 必须记录 `ablation_id` 和实际 route policy，不能只靠目录名。

## 8.2 必须重训的消融

| ID | 消融 | 实施方式 | 目的 |
|---|---|---|---|
| A4 | w/o JEPA | `beta_jepa=0`，独立输出目录 | receiver 表征预测作用 |
| A5 | w/o directed relative pose | 移除/置零 directed token 并重训 | receiver-local 几何作用 |
| A6 | w/o ROI temporal curriculum | 不做 receiver corruption，重训 | 合成失视课程作用 |
| A7 | w/o pose relocation | 遮挡保留但 pose perturb=0 | 可执行恢复监督作用 |
| A8 | single-mode decoder | `num_modes=1` 并重训 | K-mode 轨迹作用 |

论文最低集合：`A0, A1, A2, A4, A5, A6`。重训消融保持 episode、stride、epoch、
effective batch、LR schedule、cache 和 seed 一致。若只跑一个训练 seed，报告 paired episode
bootstrap CI，不能伪装成多 seed 均值。

---

# 9. Recovery、handoff 与信息鲁棒性

## 9.1 事件定义

```text
lost_start: receiver GT visibility 由 visible -> invisible
valid_assistance_window: receiver invisible AND source visible
coop_start: receiver route == COOPERATIVE
reacquired: receiver 连续 K=3 帧重新满足可见/跟踪条件
handoff_complete: coop_start 后 T_max 内 reacquired，随后返回 SELF
```

输出 recovery attempts/SR/latency、UAV->Dog 和 Dog->UAV handoff SR、lost duration 和
false reacquisition。阈值与 `K/T_max` 写入 config，不得逐方法调整。

优先从当前 100/150 条逐帧日志筛选自然单边失视，生成固定 recovery manifest。自然事件
不足时，再用已验证的 FlexibleRoom unilateral-loss 采集器生成独立 challenge set。Oracle
recovery 数据只能用于上界或训练。

## 9.2 Cross-agent information channel

先实现确定性输入 channel：

```text
payload = partner visual tokens/detection/pose/timestamp
dropout = 0%, 30%, 60%, 100%
latency = 0, 100, 300 ms
```

dropout 与 latency 分开实验。丢包序列由 `episode_id + experiment_seed` 决定并保存。
100% dropout 与 A1 的信息可用性语义必须核对。在没有 serialization、发送频率和 payload
dtype 前不报告 bytes/s，只称 cross-agent information robustness。

---

# 10. 公平性与统计合同

同表方法统一 manifest、episode order、target replay、initial pose、camera/FOV/resolution、
dt/policy frequency、timeout、speed/yaw limits、inverse controller、collision/following threshold、
success rule 和 metric implementation。

允许方法自带 detector/tracker，但必须报告权重、阈值、额外参数量和延迟。主实验禁止使用
GT target pose、bbox、mask、ID 或 recorded follower action；GT 只用于训练标签、evaluator、
数据审计和显式 Oracle。

每个 run 保存：

```text
resolved config
code root + git commit + dirty diff hash
third-party commit
checkpoint hash
dataset/split/manifest hash
seed
episode JSON + aggregate CSV/JSON
latency + peak memory
failure list
```

主表给 episode-level paired bootstrap 95% CI；方法差值按相同 episode 配对；同时报告 macro
和 micro TR；失败、超时、崩溃不从分母删除。工程失败写 `N/A + 原因`，不能以 smoke 或
oracle 代替正式结果。

---

# 11. 实施阶段与验收

## 当前主线的可执行基准命令

回归测试：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/hdt/miniconda3/envs/omtracknew/bin/python -m pytest -q \
  -p no:cacheprovider \
  tests/test_airground_coop_v3.py tests/test_eval_airground_coop_v3.py
```

训练配置与数据 dry-run：

```bash
/home/hdt/miniconda3/envs/omtracknew/bin/python \
  train_airground_coop_v3.py --dry-run
```

P1 val100 只检查 worker 分配，不启动 UE：

```bash
bash sh/eval_airground_coop_v3.sh \
  --dry-run \
  --gpu-ids 0 \
  --workers-per-gpu 1 \
  --manifest manifests/eval_manifest_100.json \
  --save-path output/eval_p1_val100_dryrun
```

正式结果聚合不要调用当前仍硬编码 78 的 shell 包装器，先直接运行：

```bash
P1_RESULT_DIR=output/eval_airground_v3_p1_val100
/home/hdt/miniconda3/envs/omtracknew/bin/python \
  -m tools.calculate_unrealzoo_metrics \
  --eval-dir "${P1_RESULT_DIR}" \
  --expected-episodes 100 \
  --require-exact-episodes \
  --output-csv "${P1_RESULT_DIR}/metrics.csv"
```

## Phase 0：复现根目录与协议修复

- 参数化主仓库、Offline BC、OpenPI 中的 `NEWTRACK_ROOT`；
- 核对旧 `_base_yh` perception cache 与 `_clean` 数据合同；
- 修复指标 shell 的 78 条硬编码；
- 保存 resolved config、commit、manifest hash；
- 明确 P1/P2/speed-profile 标记；
- 不改模型算法。

验收：V3 pytest 全通过；train dry-run 通过；launcher dry-run 正确分配 100 条；人工 3
episode case 的 macro/micro 指标正确。

## Phase 1：Evaluator 与 V3 SELF baseline

实现 active-agent evaluator、`full_v3/self_only/yolo_only/no_belief_hold`、micro TR 和
recovery/handoff 聚合。按 1 -> 10 -> val100 运行，得到 B2/B3/B4/A1/A2/A3。

## Phase 2：完成 Independent CNN-LSTM BC

adapter 指向 `_clean`；定位现有 SIGKILL；从已有结果 resume；强制收齐 val100；使用统一
聚合器重算。禁止重新标为 Offline RL。

## Phase 3：Detector + PID 与 OA-VAT

先完成简单 YOLO+PID，再接 OA-VAT-UAV 的 ORTrack/DINO/KF/recovery。均执行 1/10/100，
分别报告 detector 和 policy/control 延迟。

## Phase 4：OmTrackVLA 与 ReferTrack

依次做 offline waypoint validation、single active-agent smoke、10 episode、val100。ReferTrack
先做 Dog；不阻塞核心 cooperation table。

## Phase 5：data_arr clean pipeline

按第 3.4 节重建 9,447 条 core、376 条 pseudo 独立可控、10 waypoint、
trajectory-grouped 数据。355 条 unresolved 必须排除。cache 和 split 审计完成前不启动
正式长训练。

## Phase 6：Full V3 与结构消融重训

相同 clean split 上训练 Full 和 A4/A5/A6；A7/A8 视算力追加。先 open-loop validation，
再按同一 P1 manifest 闭环。

## Phase 7：OpenPI

先完成 RobotDog，再训练 UAV。只有双方均真实控制 follower 才进入 joint comparison。
OpenVLA-OFT 在此后评估是否仍有必要。

## Phase 8：Recovery / handoff / information robustness

核心 checkpoint 冻结后生成 recovery manifest，完成双向 handoff 和 dropout/latency，不再
回头改变主模型。

## Phase 9：论文表与图

由 episode JSON/CSV 自动生成 single-agent、cooperation、ablation、recovery/handoff 表，
dropout/latency 曲线、paired difference plot，以及成功与失败 qualitative video。

---

# 12. 最终表格设计

## Table A：单 Agent / 架构比较

| Method | Agent | Input | Params | AG-SR | TR-micro | TR-macro | CR | Latency |
|---|---|---|---:|---:|---:|---:|---:|---:|
| YOLO+PID | UAV/Dog | RGB + online bbox | | | | | | |
| CNN-LSTM BC | UAV/Dog | own RGB history | | | | | | |
| OmTrackVLA | UAV/Dog | own RGB history + text | | | | | | |
| OA-VAT | UAV | detector/tracker | | | | | | |
| ReferTrack* | Dog | RGB + online candidate history | | | | | | |
| π0.5* | Dog | dual current RGB + pose | | | | | | |
| V3 SELF | UAV/Dog | own RGB history + online proposal | | | | | | |

`*` 为第二阶段扩展。单 Agent AG-SR 使用 active-agent success。

## Table B：空地协同核心表

| Method | Drone | Dog | Cross-agent input | AG-SR | Joint-TR-micro | CR | Recovery SR | Handoff SR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Best single SELF | ✓/ | /✓ | No | | | | | |
| Dual CNN-LSTM BC | ✓ | ✓ | No | | | | | |
| V3 Dual SELF-only | ✓ | ✓ | No | | | | | |
| Estimated Pose-Assisted* | ✓ | ✓ | Rule | | | | | |
| AirGround-Coop V3 | ✓ | ✓ | Joint cooperative stream | | | | | |

不要把当前 V3 写成“learned communication”；更准确的列名是 `Cross-agent input`。

## Table C：V3 消融

| Variant | VERIFY | JEPA | Directed pose | Receiver curriculum | COOP route | AG-SR | Joint TR | Recovery SR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full | ✓ | ✓ | ✓ | ✓ | ✓ | | | |
| SELF-only | ✓ | ✓ | ✓ | ✓ | | | | |
| YOLO-only routing | | ✓ | ✓ | ✓ | ✓ | | | |
| w/o JEPA | ✓ | | ✓ | ✓ | ✓ | | | |
| w/o directed pose | ✓ | ✓ | | ✓ | ✓ | | | |
| w/o receiver curriculum | ✓ | ✓ | ✓ | | ✓ | | | |

---

# 13. 当前任务状态表

| 项目 | 状态 | 证据/阻塞 |
|---|---|---|
| Canonical V3 10-epoch train | DONE | checkpoint 与日志完整 |
| V3 150 episode 闭环 | DONE | 128/150 success；仅 Ours 结果 |
| 严格 val100 公平表 | TODO | baseline 未收齐 |
| CNN-LSTM BC train | DONE | Drone/Dog 10 epoch best checkpoints |
| CNN-LSTM BC val100 | PARTIAL | scene 14/26 时 evaluator SIGKILL |
| V3 Dual SELF-only | TODO | 需要 runtime route override |
| Single-agent evaluator | TODO | 当前 success 强制双方最终 following |
| YOLO+PID | IMPLEMENTED / SMOKE BLOCKED | 11 tests passed；UE reset socket 断连，未到首帧策略 |
| OA-VAT adapter | TODO | 源码与权重齐，环境/动作未适配 |
| OmTrackVLA adapter | TODO | 源码齐 |
| ReferTrack adapter | TODO | 评估代码齐，权重未下载 |
| OpenPI RobotDog train | PARTIAL | 2 epoch / step 1624 |
| OpenPI joint baseline | TODO | UAV 未训练，当前闭环用 recorded drone |
| data_arr 历史审计 | DONE | 当时得到 9,447 core；之后 355 被 rollback |
| data_arr 当前 source manifest | DONE | `data_arr_7_1_2_v1`：6598/952/1897，硬约束与防泄漏通过 |
| data_arr 8-waypoint split | DONE | 保留为 horizon 消融；全部 JSONL 路径存在 |
| data_arr 10-waypoint rebuild | TODO | 复用同一 episode assignments，禁止重新划分 |
| data_arr vision cache | PARTIAL | 约 5.58M/11.03M files，24/57 chunks，无活动 worker |
| Recovery/handoff metrics | TODO | 当前只记录路由/debug，未统一聚合 |
| Communication/channel test | TODO | 当前为集中式 joint inference |

---

# 14. Definition of Done

- [ ] P1 val100 的 episode 列表和 manifest hash 对所有主方法一致；
- [ ] B0、B1、B2、B3、B4、Ours 完成；
- [ ] active-agent 和 joint evaluator 均有单元测试；
- [ ] macro/micro TR 口径分开；
- [ ] Full vs SELF-only 有 paired CI；
- [ ] Drone->Dog 与 Dog->UAV recovery/handoff 分开报告；
- [ ] A0/A1/A2/A4/A5/A6 完成；
- [ ] GT 输入只出现在训练标签、evaluator 或显式 Oracle；
- [ ] speed profile 与严格 10 Hz 结果分开；
- [ ] data_arr 公平主训练只用 9,447 core；376 pseudo 单独消融；355 unresolved 排除；
- [ ] data_arr 使用统一 10 waypoint 合同；
- [ ] checkpoint、split、manifest 和第三方 commit 可追溯；
- [ ] 失败/超时/崩溃 episode 不从分母静默删除；
- [ ] 表格由脚本从 episode JSON/CSV 自动生成；
- [ ] 保留 normal、单边/双边失视、recovery、handoff 和失败案例视频。

核心论文结论应建立在：

```text
AirGround-Coop V3 > V3 Dual SELF-only > Best Single SELF
```

且差值主要出现在单边失视、恢复和 handoff episode，而不是来自不同数据、target 轨迹、
controller、评估频率或未标注 GT 输入。只有这条证据链成立，才能可信地主张 V3 的 joint
cooperative stream 带来了独立的空地协同收益。
