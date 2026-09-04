# 三数据流地空协同跟踪 V3

本项目只保留 V3。V3 用于学习“可见 Agent 引导不可见 Agent 转向并重新捕获目标”，训练入口是：

```bash
bash sh/train_airground_coop_v3.sh --gpu-ids 0,1
```

默认实验参数位于 `config/airground_cooperative_tracking_v3.yaml`，默认输出到：

```text
output/airground_three_stream_cooperative_v3_receiver_target_qwen06b
```

## 数据流

```text
drone-only tokens + tracking/verification prompts -- shared LLM row 1
                                                   --> VERIFY query --> target-match head
                                                   --> ACT query --> drone self head
dog-only tokens + tracking/verification prompts   -- shared LLM row 2
                                                   --> VERIFY query --> target-match head
                                                   --> ACT query --> dog self head

visible source tokens + receiver missing tokens
+ POSE_DRONE[x,y,sin(yaw),cos(yaw)]
+ POSE_DOG[x,y,sin(yaw),cos(yaw)]
+ REL_DRONE<-DOG[forward,right,sin(dyaw),cos(dyaw),distance]
+ REL_DOG<-DRONE[forward,right,sin(dyaw),cos(dyaw),distance]
                  -- separate shared-weight LLM call
                  --> drone cooperative K-mode decoder
                  --> dog cooperative K-mode decoder
```

共享的是 LLM 参数，不是两个 Agent 的 token context。改变机器狗输入不会改变无人机 self action，反方向同样成立。同一 Agent 的 VERIFY 位于 ACT 之前：ACT 可以利用校验表征，而 target-match loss 由于因果 mask 不能反传到后面的 ACT token/action head。该优化将 self-flow 从 `4B` 行降为 `2B` 行。

## 常用命令

检查数据 split、vision cache 和 YOLO cache：

```bash
python train_airground_coop_v3.py --dry-run
```

单卡调试两步：

```bash
CUDA_VISIBLE_DEVICES=0 python train_airground_coop_v3.py \
  --debug-max-steps 2 --batch-size 1
```

双卡训练：

```bash
bash sh/train_airground_coop_v3.sh --gpu-ids 0,1
```

覆盖少量参数：

```bash
bash sh/train_airground_coop_v3.sh --gpu-ids 0,1 \
  --batch-size 48 --epochs 10 --lr 4e-5 --num-workers 4 --temporal-stride 3
```

续训使用同一个 V3 输出目录：

```bash
bash sh/train_airground_coop_v3.sh --gpu-ids 0,1 --resume
```

## 重要约束

- 两条 self 流只读取自身 RGB feature、检测和历史，不读取另一视角或 Agent pose。
- 原来的 `visibility_logits` 已替换为 `target_match_logits`：YOLO 先提出人物候选，模型显式池化框内 ROI 视觉特征并与框特征融合送入 LLM，再结合当前 RGB 和历史视觉判断它是否为指定目标。GT bbox 仅用于生成训练标签，推理时不需要 GT。
- 无人机与机器狗分别有可配置的 `drone_target_verification_prompt` / `dog_target_verification_prompt`，校验结果从独立 `VERIFY` query hidden 读出，不再复用 action hidden。
- YOLO 的 target mask 与候选框来自同一检测器，不能作为独立验证证据，因此进入验证流前会移除 target channel；障碍、free、unknown scene context 仍保留。
- 训练集自然错误候选不足 1%，训练时默认以 0.25 概率注入低 IoU 的其他检测框/移位框作为 hard negative；验证集概率为 0，不改变真实 YOLO 评估分布。
- 默认 `train_temporal_stride=3`：每轮在每个 episode 内只取一个 offset，offset 随 epoch 转动。当前数据三轮恰好覆盖全部 623,322 帧一次；最终训练 10 轮，相当于完整覆盖三遍后再训练一次 offset-0 子集。
- synthetic assistance 按 epoch 课程启用：第 1–3 轮概率为 `0.70`，第 4–7 轮为 `0.85`，之后为 `0.90`；未启用时保留双视角 clean cooperative 输入，不强制制造 receiver。启用后以互斥的 `50%/50%` 选择 drone 或 dog receiver，同一样本最多破坏一个 Agent。
- receiver 遮挡分为 `ROI_ONLY`、`CURRENT_FULL`、`RECENT_FULL` 和 `ALL_FULL`。`ROI_ONLY` 只把 YOLO 目标框覆盖的 current fine tokens 换成 missing token，保留真实背景和全部历史；`CURRENT_FULL` 只缺失当前帧；`RECENT_FULL` 额外缺失最近 `2/4/8/16` 帧历史；`ALL_FULL` 才缺失全部历史。模型通过 `fine_missing_mask` / `coarse_missing_mask` 做 token 级替换，不再一律抹掉接收端整段视觉。
- `ROI_ONLY` 保留的是原 receiver pose 下拍摄的真实背景，所以禁止同时做物理位姿重定位。只有完整当前帧已经缺失的三种模式才允许 pose relocation；第 1–3 轮不扰动 pose，后续以 `0.50` 概率施加最多 `0.5 m / 30°` 的局部扰动，不再使用旧的 `2 m / 150°` 大跳变。
- 两个 Agent 分别使用 `[x_m,y_m,sin(yaw),cos(yaw)]` 绝对 pose token，并新增两个有方向的 receiver-local 相对位姿 token：`[forward,right,sin(Δyaw),cos(Δyaw),distance]`。训练时对绝对 pose 使用共享随机 SE(2) 坐标变换，相对 token 对该变换保持不变。
- pose relocation 后，刚性变换得到的 clean future path 仅作为恢复参考，不再直接当标签。receiver target 始终从 `waypoint[0]=[0,0,0]` 出发，逐步朝参考轨迹恢复，并按样本 `dt`、`2.5 m/s` 最大速度和 `1.5 rad/s` 最大角速度裁剪；RobotDog 采用 rotate-then-forward 的非完整约束 rollout，避免第一步瞬移或瞬时大转向。
- source cooperative target 保持 clean；没有 pose relocation 的 ROI/current/history 遮挡也保持 clean target。只有实际重定位的 synthetic receiver 使用可执行恢复 target，其当前 target-belief 标签同步变换到新 receiver frame。
- 协同遮挡和 pose relocation 只作用于 cooperative 流。两个 self row 始终读取各自 clean visual/detection，因此 source 和 synthetic receiver 都继续计算 clean self action 与 VERIFY loss；JEPA 只在真正被遮掉的 receiver tokens 上使用 clean teacher latent。
- cooperative decoder 显式读取障碍 grid token，但 `beta_obstacle` 默认必须为 0。现有 mask 是图像坐标，未标定投影前不能与局部 waypoint 直接计算碰撞损失。
- 当前模型 waypoint 仍保持完整的 `[x, y, yaw]` 未来局部位姿接口。机器狗的
  `y` 是位置监督，不是横向速度；`dog_lateral_loss_weight=1.0`。clean/self target
  仍回归原始 GT，只有被物理重定位的 cooperative receiver 改用非完整约束恢复 target。
- 机器狗不能横向移动，所以 V3 推理在进入共享 inverse-fixed-dt 控制器前独立做
  non-holonomic pose projection：每个 `[dx,dy,d_yaw]` 段按恒曲率 unicycle 的可行弧
  拟合，`x/y` 共同转换为累计前进弧长，未来 `yaw` 按其 horizon 时间转换为角速度，
  物理 action 的横向速度始终为 `0`。原始模型 waypoint 和投影 waypoint 都写入
  每帧 `action_debug`，方便检查 `y` 是否被正确解释。
- V3 训练中的 drone/dog 最大平移速度均为 `2.5 m/s`；V3 推理入口也显式传递并
  硬执行这两个速度上限，即使 bbox correction 未进入 reliable 状态或被关闭也不会
  放开限速。
- 唯一支持的 architecture 是 `airground_three_stream_cooperative_v3`；训练和推理入口会拒绝其他 architecture 的配置与 checkpoint。
- checkpoint 必须同时带有 `receiver_feasible_recovery_v1`、`roi_temporal_curriculum_v1` 和 `directed_receiver_local_v1` 三个 V3 语义标记及 `relative_pose_proj` 权重。不满足当前 receiver-target contract 的 checkpoint 会被 evaluator 拒绝，不能用于 `--resume`。

## 主要输出

模型 `forward()` 返回：

- `self_waypoints`: `(B,2,N,3)`；
- `cooperative_candidates`: `(B,2,K,N,3)`；
- `cooperative_mode_logits`: `(B,2,K)`；
- `cooperative_waypoints`: 每个 Agent 得分最高的候选；
- `waypoints`: 按可见性路由后的轨迹；
- `jepa_prediction_tokens` / `jepa_teacher_tokens` / `jepa_token_mask`；
- `routing_mode`、`route_to_cooperative`、`route_to_belief` 和 `both_invisible`。
- `yolo_visible`、`target_match_logits`、`target_match_probability` 和最终的 `observed_visible`。
- `self_action_context` 和 `target_verify_context`，分别是同一 self row 内 ACT/VERIFY 两个 query 的上下文。
- `agent_poses`: `(B,2,4)` 的两个独立共享坐标 pose token；
- `directed_relative_pose`: `(B,2,5)`，分别表示另一个 Agent 在每个 receiver 局部坐标系下的相对位姿；
- `coarse_missing_mask` / `fine_missing_mask`: cooperative 流实际替换的 history/current token mask；

闭环评估时应为每个环境实例维护独立的 `AirGroundVisibilityRouter`：

```python
state = router.update(
    detection_feat[0],
    output["target_match_probability"][0],
)
```

在线 V3 planner 已同时得到 self/cooperative 全部分支，所以它在同一帧直接按 `state["mode"]` 选择轨迹，无需为了路由再执行第二次 LLM forward。最终可见判断是 `YOLO valid/confident AND LLM target-match passed`，而不是再训练一个重复的可见性检测器。

## 闭环推理

V3 闭环推理入口：

```bash
bash sh/eval_airground_coop_v3.sh --gpu-ids 0 --workers-per-gpu 10
```

V3 默认使用 `--max-lost-steps 400`，固定 100-episode 协议和随机/debug
协议保持一致。因此当 Agent 跟不上目标时，常规录制序列不会因
`Lost` 连续帧阈值而中途终止，会继续运行到该段序列结束。显式传入的
`--max-lost-steps` 仍可覆盖此默认值。

推理时不注入人工 pose 扰动：它使用两个 Agent 当前真实 pose，转成以双方中点为原点的共享米制坐标。receiver 不可见而 source 可见时使用 COOPERATIVE，receiver 重新可见后切回 SELF。两边均不可见时先短暂保留 BELIEF 历史轨迹；超过 belief hold 后进入有界旋转搜索。搜索中心锁定在确认丢失时两个 Agent 各自的 yaw，目标端点为中心左右各 30 度，抵达端点后在两个端点间来回切换；搜索 waypoint 的 `x/y` 始终为 0，不执行平移，也不会连续旋转 360 度。内部数值状态仍为 `ROUTE_SEARCH`，视频/debug 状态名为 `search`，并额外记录 `search_center_yaw_degrees`、`search_target_yaw_degrees` 和 `search_yaw_error_degrees`。

receiver-target V3 已完成 10 epoch 训练，当前最优 checkpoint 为
`output/airground_three_stream_cooperative_v3_receiver_target_qwen06b/best_val.pt`，
对应 epoch 8 / step 9000，`val_loss_nav=0.0424876`。默认评估目录为
`output/eval_airground_coop_v3_receiver_target_fixed100`。推理入口严格检查
三项 receiver-target 语义标记并严格加载全部参数；不满足当前 contract 的 checkpoint
会直接拒绝加载。

推理显式把 `synthetic_occlusion`、`coarse_missing_mask`、`fine_missing_mask` 全部置零，
即训练遮挡课程不会泄漏到在线评测；自然失视时仍输入 receiver 的真实背景与历史、
无效 YOLO detection 和双方实时 pose。RobotDog 默认使用
`v3_nonholonomic_projection`，将预测的完整 `[x,y,yaw]` pose trajectory 投影成可执行的
前进/转向轨迹，与 receiver-target 非完整约束监督一致。V3 不提供绕过该投影的兼容模式。
