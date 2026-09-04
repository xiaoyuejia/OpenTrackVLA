# UnrealZoo Motion Model and Inverse Control

## Scope

This document describes the motion behavior measured in the packaged UnrealZoo
environment and the image-and-bbox-only inverse controller used by
`eval_unrealzoo_multi_agent.py`.

The conclusions below apply to the fixed-step protocol:

```text
pause UE -> capture RGB/mask/bbox -> send action -> resume one 0.1 s pulse -> pause
```

`dt = 0.1 s`, `UNREALZOO_FIXED_TIMESTEP=0.1`, and BP interval is `100 ms`.
Wall-clock inference time is outside the simulation pulse and must not change
the control period.

## Coordinate and Command Conventions

| Item | Convention |
|---|---|
| Unreal distance | `100 Unreal units = 1 m` |
| Model waypoint | local to the agent at observation `t`: `[forward_m, right_m, yaw_delta_rad]` |
| Waypoint point 0 | `[0, 0, 0]`, the agent pose at observation `t` |
| Ground BP action | `[turn_deg, speed_cm_s]` |
| Drone BP action | `[x_step_m, y_step_m, z_step_m, yaw_rate_rad_s]` |
| Ground yaw relation | measured `delta_yaw_deg = 0.4 * turn_deg` per fixed pulse |
| Drone XY action | step-like displacement; command velocity is multiplied by `dt` before `set_move_bp` |

The training waypoints are geometric local trajectories. They are not direct BP
commands and cannot safely be passed to `set_move_bp` without a dynamics
adapter.

## Experimental Evidence

### Fixed 0.9 m/s straight-line response

Source: `/data/hdt/ntv_data/sim_data/experment/fixed_step_human_walk_dt01/0.json`
and `/data/hdt/ntv_data/sim_data/experment/fixed_step_dog_drone_walk_dt01/0.json`.

| Agent | Requested distance, 10 s | Measured forward distance | Key response observation |
|---|---:|---:|---|
| Human | 9.000 m | 8.832 m | startup ramp / one-pulse command delay |
| RobotDog | 9.000 m | 8.832 m | startup sequence `0, 0.34, 0.68, 0.90 m/s` |
| Drone | 9.000 m | 6.242 m | long first-order under-response; no step response |

The human and dog use the same two-dimensional ground action space. Giving the
ground agents high requested acceleration removes most of the ramp, but does
not remove the discrete speed-command timing described below.

### RobotDog one-pulse translation delay

Source: `/data/hdt/ntv_data/sim_data/experment/robotdog_fixed_step_command_delay/delay_analysis/robotdog_command_delay_report.json`.

The test sends ten pulses of `0.9 m/s`, followed by ten pulses of `0 m/s`,
with `set_acceleration=10000` requested before the test.

| Alignment of measured `v[t]` | RMSE | Correlation |
|---|---:|---:|
| Current speed command `u[t]` | 0.2862 m/s | 0.7953 |
| Previous speed command `u[t-1]` | 0.0305 m/s | 0.9979 |
| Two-pulse-old command `u[t-2]` | 0.2284 m/s | 0.8700 |

Observed transitions:

```text
step 1: send 0.9 m/s, measured 0.000 m/s
step 2: send 0.9 m/s, measured 0.900 m/s

step 11: send 0.0 m/s, measured 0.900 m/s
step 12: send 0.0 m/s, measured 0.135 m/s
step 13: send 0.0 m/s, measured 0.000 m/s
```

The primary model is therefore:

```text
ground_forward_velocity[t] ~= commanded_speed[t - 1]
```

There is also a small braking residual after a stop command. It is an internal
ground movement state, not a timestamp error in the recording.

### Ground yaw response

Source: `/data/hdt/ntv_data/sim_data/experment/fixed_step_turn_response_dt01/0_robotdog_turn_info.json`
and `0_human_turn_info.json`.

For both human and RobotDog, sending `turn_deg=30` changes yaw by `12 deg`
inside the same `0.1 s` pulse. Thus:

```text
delta_yaw_deg[t] = 0.4 * turn_deg[t]
```

Yaw is current-pulse behavior; it does not share the forward-speed one-pulse
delay. The same experiment measured a first-order yaw response for the drone.

### Drone response

Sources:

- `/data/hdt/ntv_data/sim_data/experment/fixed_step_dog_drone_walk_dt01/0.json`
- `/data/hdt/ntv_data/sim_data/experment/fixed_step_turn_response_dt01/0_turn_response_summary.json`

The drone moves on the current pulse, but its state has strong first-order
inertia. The calibrated discrete approximation used by replay/evaluation is:

```text
v_xy[t+1] = a_xy * v_xy[t] + b_xy * u_xy[t]
omega[t+1] = a_yaw * omega[t] + b_yaw * u_yaw[t]

a_forward = a_lateral = 0.969
b_forward = b_lateral = 0.0301
a_yaw = 0.464
b_yaw = 0.359
```

`u_xy` is in m/s before conversion to the BP step displacement. These
coefficients are empirical fixed-`dt=0.1` coefficients. They are not valid for
a different control period without recalibration.

## Why the Ground Delay Is UE/BP Behavior

The Python wrapper only builds and transmits the command:

```text
vbp <agent> set_move <turn_deg> <speed_cm_s>
```

See `unrealzoo-gym/gym_unrealcv/envs/agent/character.py:set_move_bp`. The
fixed-step driver sends this command before `resume -> pause`; it does not
intentionally delay ground motion. Since the fixed-step experiment still shows
the delay, it is inside the packaged UE BP/character movement execution order:
the movement update integrates an already-held speed, while the BP event
updates the speed used by the next pulse.

The exact Blueprint node graph cannot be named from this repository because it
contains the packaged UE binary, not the editable Unreal project/BP assets.
The measured input-output model is sufficient for execution and is more
reliable than assuming a nonexistent instantaneous response.

## Inverse Waypoint-to-Action Controller

Implemented in `eval_unrealzoo_multi_agent.py` with:

```text
--waypoint-control-mode inverse_fixed_dt
--deterministic-step --dt 0.1
```

### Drone

Choose a future waypoint `W_d[k]` at horizon `T_k` and form desired local
velocity:

```text
v_des = W_d[k, 0:2] / T_k
omega_des = W_d[k, 2] / T_k
```

Maintain an internal actuator estimate `v_hat`, initialized to zero at episode
reset. Invert the measured first-order model:

```text
u_xy = (v_des - a_xy * v_hat_xy) / b_xy
u_yaw = (omega_des - a_yaw * v_hat_yaw) / b_yaw

env_action_drone = [u_x * dt, u_y * dt, 0, u_yaw]
v_hat_next = [a_xy * v_hat_xy + b_xy * u_xy,
              a_yaw * v_hat_yaw + b_yaw * u_yaw]
```

No evaluator-side speed clip is applied in this mode. A large inverse command
means the predicted waypoint requires a large command under the calibrated
drone actuator; it should be logged and handled by model/data improvement, not
silently hidden by a clip.

### Learned-waypoint command smoothing

For learned waypoints, direct inversion amplifies small frame-to-frame changes
by approximately `1 / b_xy = 33`. Evaluation therefore first applies a causal
reference filter and then inverts the actuator:

```text
v_ref[t] = (1 - alpha) * v_ref[t-1] + alpha * v_waypoint[t]
u[t] = inverse_dynamics(v_ref[t], v_hat[t])
```

Default alphas are drone XY `0.20`, drone yaw `0.25`, RobotDog speed `0.30`,
and RobotDog yaw `0.30`. The filter uses only previous model references and
previous commands; it does not read UE pose or realized velocity. Per-step
JSON records raw/reference velocities and command deltas under
`inverse_control.command_smoothing`.

### RobotDog

Let the selected current-pulse target be `W_g[k]`. Yaw is immediate:

```text
turn_deg = degrees(W_g[k, yaw_delta]) / 0.4
```

For `ground_translation_delay_steps=1`, the speed sent at `t` is assigned to
the next geometric segment, not the segment that has already started:

```text
v_forward_cmd = (W_g[k+1, forward] - W_g[k, forward]) / (T[k+1] - T[k])
env_action_dog = [turn_deg, 100 * v_forward_cmd]
```

The predicted lateral dog component is logged but cannot be executed: the BP
ground action is a unicycle-style `[turn, forward_speed]` interface.

At the final available waypoint, where `k+1` is unavailable, the controller
uses the selected waypoint average as an explicit fallback and records that
choice in JSON.

### Human

The evaluation human is a trajectory source, not a model-controlled follower.
When target replay mode is `action`, its recorded action is fed continuously
through the same ground BP action space. It shares the ground timing behavior,
but no human pose is teleported after initial setup.

## No-Cheat Evaluation Boundary

The controller consumes only:

```text
current RGB history + allowed target bbox + model waypoint + internal prior commands
```

It does not consume current/previous UE agent pose, target bearing, recorded
follower actions, oracle heading, or realized velocity readback to generate an
action. UE poses are still read after the pulse for simulation bookkeeping and
metrics such as distance, collision, and success; those values are never fed
back into the planner/inverse controller.

The following must remain disabled for the image+bbox protocol:

```text
--oracle-heading-assist false
--oracle-drone-action-source none
--oracle-robotdog-action-source none
--snap-heading false
--face-target-before-step false
```

Oracle ground-truth bbox/ROI is an explicit evaluation upper-bound input, not
hidden state. Use `--bbox-source none` or a detector/tracker when evaluating
without that upper bound.

## Training and Evaluation Implications

1. Train geometric local waypoints, not raw BP commands, as the primary model
   target. They remain valid descriptions of the demonstrated trajectory even
   when the original realtime action timing was irregular.
2. Use fixed-time waypoint labels at `0.1 s` and mask unavailable future
   horizons. A waypoint is a desired local state, not an assertion that the
   environment has instantaneous velocity control.
3. Run closed-loop evaluation with the inverse controller above. This gives
   the model the same actuator semantics as replay without leaking poses.
4. Treat large inverse commands and persistent image-space tracking error as
   model/data failures. Do not conceal them with evaluator-side clipping.
5. For exact reproduction of old realtime recordings, BP internal state and
   original UE tick ordering would also be needed. Exact raw-action replay is
   a separate reproducibility experiment, not the standard closed-loop model
   evaluation protocol.

## Recalibration Checklist

Re-run the fixed-step tests when any of these changes: UE build, BP asset,
agent pawn, `dt`, interval, frame rate, scene-level physics settings, or
camera/control code that changes tick ordering.

Required tests:

```text
1. Human/dog/drone 0.9 m/s straight response.
2. RobotDog 0.9 -> 0.0 command-delay test.
3. Human/dog/drone positive -> zero -> negative yaw response.
4. Refit drone XY/yaw coefficients from the resulting fixed-step poses.
```
