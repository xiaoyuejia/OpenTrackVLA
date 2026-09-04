# 代码解释
• 我会按代码实际执行顺序解释：输入记录如何变成参考轨迹、初始化做了什么、每个 0.1s 如何计算人/
  狗/无人机命令、UE tick 如何推进，以及输出 JSON/误差指标各自代表什么。也会明确指出哪些是严格
  复刻，哪些是近似逆动力学。

• Explored
  └ Read replay_hand_realtime_inverse_fixed_dt.py

──────────────────────────────────────────────────────────────────────────────────────────────

• 这个脚本不是简单地把 JSON 中的速度重新发送，而是一个“固定时间步 + 当前状态反馈 + 执行器逆模
  型”的复刻实验。

  代码入口是 tools/replay_hand_realtime_inverse_fixed_dt.py。

  整体流程如下：

  读取 source JSON
        ↓
  只初始化一次人/狗/无人机位姿和相机
        ↓
  暂停 UE
        ↓
  读取当前 replay 实际位姿
        ↓
  根据 source 下一段轨迹反算动作
        ↓
  发送三类 agent action
        ↓
  UE resume → 推进一个 0.1s tick → pause
        ↓
  读取 after-action 实际位姿
        ↓
  计算误差、饱和、预测残差
        ↓
  保存 JSON / CSV / MP4 / 轨迹图

  1. 固定时间步

  脚本启动时设置：

  os.environ.setdefault("UNREALZOO_FIXED_TIMESTEP", args.dt)

  对应代码：tools/replay_hand_realtime_inverse_fixed_dt.py:44

  默认：

  dt = 0.1s
  ue_interval_ms = 100

  初始化后设置：

  unrealcv.set_max_FPS(1.0 / args.dt)
  unrealcv.set_pause()

  每一步由：

  unrealcv.set_resume()
  unrealcv.set_pause()

  推进一个 UE 时间步，对应 tools/replay_hand_realtime_inverse_fixed_dt.py:580。

  暂停期间读取位姿、生成相机画面和计算命令，不推进仿真时间。

  2. 初始化不是每步 teleport

  脚本只在 episode 开始恢复一次：

  restore_pose(env, target_name, first_target)
  restore_pose(env, dog_name, first_dog, ...)
  restore_pose(env, drone_name, first_drone, ...)

  对应 tools/replay_hand_realtime_inverse_fixed_dt.py:486。

  之后没有再调用 set_obj_location 恢复位姿。因此：

  - 人、狗、无人机初始位姿来自 source。
  - 初始相机参数来自 source。
  - 中途位置完全由 BP 动力学产生。
  - 但 BP 内部速度、Movement Component 状态、物理残留状态没有恢复。

  所以这不是完整状态复刻，只是初始外部位姿一致。

  3. 位姿如何转换成速度

  body_xy_velocity() 把两个世界坐标差分转换成当前机体坐标系：

  world_delta = (after_xy - before_xy) / 100
  forward = cos(yaw) * dx + sin(yaw) * dy
  right   = -sin(yaw) * dx + cos(yaw) * dy
  velocity = delta / dt

  代码在 tools/replay_hand_realtime_inverse_fixed_dt.py:79。

  因此返回的是：

  [forward_velocity_mps, right_velocity_mps]

  它不是简单取世界坐标的 x/y 速度，而是按当前 yaw 转换到 agent body frame。

  4. 人和狗当前使用的控制方式

  当前推荐模式是：

  GROUND_CONTROL_MODE=source_yaw
  GROUND_SPEED_MODEL=legacy_preview
  GROUND_TRANSLATION_DELAY_STEPS=1

  核心代码在 tools/replay_hand_realtime_inverse_fixed_dt.py:141。

  地面 source 参考有三个位姿：

  reference_before     = source_pose[t]
  reference_after      = source_pose_after_action[t]
  reference_after_next = source_pose_after_action[t+1]

  当：

  GROUND_TRANSLATION_DELAY_STEPS=1

  脚本使用：

  source_velocity =
      (reference_after_next - reference_after) / 0.1

  也就是认为：

  第 t 步发送的 speed
  影响第 t+1 步的平移

  这就是地面一帧速度延迟补偿，代码在 tools/replay_hand_realtime_inverse_fixed_dt.py:151。

  同时，脚本计算当前位置误差反馈：

  position_feedback =
      (reference_before - actual_before) / feedback_time_s

  代码在 tools/replay_hand_realtime_inverse_fixed_dt.py:158。

  然后：

  desired_forward_speed =
      source_preview_forward_speed
      + position_feedback_forward

  由于狗和人没有 lateral action，横向反馈只被记录为：

  unexecutable_lateral_velocity_mps

  不会真正作为 action 发给 BP。

  最后地面 action 转换成：

  [turn_deg, speed_cm_s]

  其中：

  turn_cmd = yaw_error / 0.4
  speed_cmd = desired_forward_speed * 100

  yaw 关系来自实验标定：

  actual_yaw_delta = 0.4 * turn_cmd

  所以当 source_yaw 模式启用时，狗相机方向保持与 source yaw 对齐，但如果 replay 位置发生横向偏
  差，地面系统无法横移纠正。

  发送给 UE 的人/狗 action 在 tools/replay_hand_realtime_inverse_fixed_dt.py:578。

  5. 人和狗的速度上限、加速度

  初始化阶段：

  set_max_speed(target_name, human_max_speed_mps * 100)
  set_max_speed(dog_name, dog_max_speed_mps * 100)
  set_acceleration(target_name, ground_acceleration)
  set_acceleration(dog_name, ground_acceleration)

  对应 tools/replay_hand_realtime_inverse_fixed_dt.py:497。

  当前常用设置：

  human max speed = 1.2m/s
  dog max speed   = 1.4m/s
  acceleration    = 10000

  注意，10000 只能让达到目标速度更快，不能消除：

  - action 到位姿的一帧延迟；
  - 地面系统没有横向速度；
  - 真实 BP 的速度残留；
  - source 轨迹和 replay 轨迹之间的横向偏差。

  source_yaw 模式下，速度会沿当前 heading 执行，不能保证同时严格恢复 source XY 和 source yaw。

  6. 无人机的逆动力学

  无人机与地面 agent 完全不同。

  当前模型假设：

  v_next = a * v_current + b * u_command

  默认参数：

  forward: a=0.969, b=0.0301
  lateral: a=0.969, b=0.0301
  yaw:     a=0.464, b=0.359

  无人机当前期望速度来自：

  reference_pose_after - actual_pose_before
  ----------------------------------------
                0.1s

  然后变换到无人机 body frame。

  反解：

  u_x = (v_desired_x - a_x * v_current_x) / b_x
  u_y = (v_desired_y - a_y * v_current_y) / b_y

  代码在 tools/replay_hand_realtime_inverse_fixed_dt.py:301。

  之后 action 转换为：

  [dx_m, dy_m, dz_m, yaw_rate_radps]
  =
  [vx_cmd * 0.1, vy_cmd * 0.1, 0, yaw_cmd]

  所以：

  - drone_env_action[0:2] 是每个 0.1s action 的位移型输入。
  - drone_inverse.commanded_body_velocity_mps 才是速度。
  - 不能把 drone_env_action[0:2] 再乘一次 0.1。

  无人机 yaw 也是一阶逆模型：

  omega_next = 0.464 * omega_current + 0.359 * yaw_command

  但无人机 Z 目前始终发送：

  dz = 0

  因此当前无人机高度漂移没有被修正。

  7. 延迟模型的实际含义

  当前并不是所有 agent 都加了延迟：

  人/狗平移：一帧延迟
  人/狗 yaw：本帧生效
  无人机 XY：实验证明使用 0 帧离散延迟
  无人机 yaw：实验证明使用 0 帧离散延迟，但有一阶角速度状态

  因此：

  GROUND_TRANSLATION_DELAY_STEPS=1

  只影响人和狗的平移，不影响无人机。

  这也是为什么旧的 data_cyj_fixed_dt_old 不能靠加一帧延迟修复：旧数据的实际记录间隔本身就在
  0.07-0.20s 之间，且缺少可靠时间戳。

  8. 输出 JSON 的含义

  每一行输出包含：

  target_pose
  robotdog_pose
  drone_pose

  表示当前 action 前的 replay 实际位姿。

  target_pose_after_action
  robotdog_pose_after_action
  drone_pose_after_action

  表示固定 0.1s tick 后的实际位姿。

  source_*_pose_after_action

  表示 source 记录中希望达到的参考位姿。

  *_env_action

  表示真正送给环境的命令：

  target_env_action:   [turn_deg, speed_cm_s]
  robotdog_env_action: [turn_deg, speed_cm_s]
  drone_env_action:    [dx_m, dy_m, dz_m, yaw_rate_radps]

  *_after_error_m

  是实际 after-action 位姿和 source after-action 位姿的三维距离误差。

  *_inverse

  记录逆控制过程，包括：

  - source 前馈速度；
  - 位置反馈速度；
  - 未裁剪命令；
  - 裁剪后的命令；
  - 是否发生速度/转向饱和；
  - 无人机预测下一速度；
  - 当前测得的地面速度；
  - 自适应模型参数。

  输出生成位置在 tools/replay_hand_realtime_inverse_fixed_dt.py:582。

  9. 当前脚本本质上是什么

  也不是：

  完整 UE/BP 状态复原

  以 source pose 作为参考轨迹
  + 只恢复初始外部位姿
  + 使用当前 replay 实际状态
  + 通过执行器模型计算新 action
  + 固定 0.1s 运行
  + 检查这条 source 轨迹是否可由当前 BP 命令空间实现

  因此最准确的定义是：

  固定时间步的闭环逆动力学可达性实验

  当前推荐运行设置仍然是：

  GROUND_CONTROL_MODE=source_yaw
  GROUND_SPEED_MODEL=legacy_preview
  GROUND_TRANSLATION_DELAY_STEPS=1

  vector_2d、vector_2d_stable、vector_2d_lookahead、vector_2d_damped 和 adaptive_inverse 都是
  实验模式，不建议用于保持原始狗相机视图和训练数据对齐。
