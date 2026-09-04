# 无人机固定步长逆动力学标定

本文说明当前 `hand_realtime` replay 使用的无人机速度逆动力学模型。实现位于 `tools/history/replay_hand_realtime_inverse_fixed_dt.py`。

## 适用设置

- 固定时间步：`dt = 0.1 s`
- 控制量：机体坐标系的前向、右向速度命令与 yaw 角速度命令
- 当前全量 replay 限幅：`|u_f|, |u_r| <= 100 m/s`，`|u_yaw| <= 8 rad/s`

这不是关节空间的逆运动学，而是从目标 pose 反解 UE 可执行动作的闭环一阶逆动力学。

## 已标定的动力学

设 `v_f`、`v_r` 分别为无人机机体坐标系中的前向和右向真实速度（m/s），`omega` 为真实 yaw 角速度（rad/s）；`u_f`、`u_r`、`u_yaw` 为反解出的命令。

| 自由度 | 一阶模型 | 标定参数 |
| --- | --- | --- |
| 前向速度 | `v_f[t+1] = 0.969 * v_f[t] + 0.0301 * u_f[t]` | `a_f=0.969`, `b_f=0.0301` |
| 右向速度 | `v_r[t+1] = 0.969 * v_r[t] + 0.0301 * u_r[t]` | `a_r=0.969`, `b_r=0.0301` |
| yaw 角速度 | `omega[t+1] = 0.464 * omega[t] + 0.359 * u_yaw[t]` | `a_yaw=0.464`, `b_yaw=0.359` |

`v_f`、`v_r` 的单位均为 m/s；`omega` 和 `u_yaw` 的单位为 rad/s。

## 逆解

每一步从当前 replay 实际 pose 和 source 的下一 pose 计算期望速度。先把世界坐标 XY 位移旋转到当前机体坐标系：

```text
v_des = R(yaw_t)^T * (p_source[t+1] - p_replay[t]) / dt
omega_des = wrap(yaw_source[t+1] - yaw_replay[t]) / dt
```

然后反解一阶模型：

```text
u_f   = (v_f_des   - 0.969 * v_f_measured) / 0.0301
u_r   = (v_r_des   - 0.969 * v_r_measured) / 0.0301
u_yaw = (omega_des - 0.464 * omega_measured) / 0.359
```

反解得到的命令先记录为 `unconstrained_*`，再按当前上限裁剪；裁剪状态会写入 `translation_saturated` 和 `yaw_saturated`。

## 发给 UE 的 action

当前 replay 将逆解命令转成：

```text
env_action = [u_f * dt, u_r * dt, 0, u_yaw]
```

因此前两个 action 分量是一个固定步内的机体平移量，yaw 分量直接是 yaw 角速度控制量。

## Z 轴说明

当前**没有** `v_z` 的标定模型，也不主动控制高度：

```text
env_action[2] = 0
```

系统会记录 `z_reference_error_m` 和最终三维位置误差，但不会通过垂直速度去追 source 高度。因此当前控制覆盖的是 **XY + yaw**，不是完整的 **XYZ + yaw**。

## 闭环与输出标签

- 每步只恢复初始 pose；后续状态均来自 UE 执行 replay action 后的真实结果。
- 下一步逆解使用的是当前真实 replay 状态，而不是强制写回 source pose。
- `base_velocity` 保存实际执行后测得的速度。
- `commanded_base_velocity` 与 `env_action` 保存逆控制器下发的命令。
- 每步同时保存 source-after 位置/yaw 误差，便于识别饱和或动力学不可达的 transition。
