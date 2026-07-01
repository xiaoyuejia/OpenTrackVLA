# 单 Agent 跟踪评估指标对比：当前 UnrealZoo vs 旧 Habitat

本文整理两套评估代码中 `SR / TR / CR` 的计算方式，并标出关键代码位置。

## 当前 UnrealZoo 狗评估

入口流水线是 [`sh/run_robotdog_single_agent_pipeline.sh`](/data/hdt/newtrackvla/sh/run_robotdog_single_agent_pipeline.sh:94)。当前已在评估命令里显式开启：

```bash
--require-success-distance
```

因此现在狗的逐步跟踪判定不再只看目标是否可见，而是：

```text
following = target_visible && distance_xy <= robotdog_success_distance
```

默认阈值来自 [`eval_unrealzoo_single_agent.py`](/data/hdt/newtrackvla/eval_unrealzoo_single_agent.py:636)：`--robotdog-success-distance 8.0`，单位是米。具体逐步计算在 [`eval_unrealzoo_single_agent.py`](/data/hdt/newtrackvla/eval_unrealzoo_single_agent.py:398)。这里的 `common` 是 [`eval_unrealzoo_multi_agent.py`](/data/hdt/newtrackvla/eval_unrealzoo_multi_agent.py:94)，距离函数复用 [`generate_drone_human_tracking_small.py`](/data/hdt/newtrackvla/unrealzoo-gym/example/DataRecording/generate_drone_human_tracking_small.py:304)：

- 读狗视角和目标 mask 可见性：`common._read_agent_pair(...)`
- 计算狗和人的 XY 平面距离：`common.distance_xy_m(...)`
- `in_success_distance = distance <= success_distance`
- `following = visible and (in_success_distance or not args.require_success_distance)`

显式加上 `--require-success-distance` 后，`not args.require_success_distance` 为假，所以 `following` 必须同时满足可见和距离阈值。

狗的碰撞函数由 [`eval_unrealzoo_multi_agent.py`](/data/hdt/newtrackvla/eval_unrealzoo_multi_agent.py:119) 引入，具体实现见 [`generate_robotdog_human_tracking_small.py`](/data/hdt/newtrackvla/unrealzoo-gym/example/DataRecording/generate_robotdog_human_tracking_small.py:924)：优先读 UnrealZoo `info["metrics"]["collision"]` 矩阵或 `info["collision"]` 字段；如果没有可用信息，则用 `dist_xy < 0.7m && height_gap < 1.0m` 作为兜底碰撞判定。

单 episode 的终止和成功判定在 [`eval_unrealzoo_single_agent.py`](/data/hdt/newtrackvla/eval_unrealzoo_single_agent.py:488) 和 [`eval_unrealzoo_single_agent.py`](/data/hdt/newtrackvla/eval_unrealzoo_single_agent.py:501)：

- 若发生碰撞，`status = "Collision"` 并停止。
- 若连续丢失达到 `--max-lost-steps`，默认 30，`status = "Lost"` 并停止。
- `following_rate = following_steps / total_steps`
- `near_rate = near_steps / total_steps`
- `success = no_collision && status not in {Lost, Collision, Timeout} && total_steps >= min_success_steps && following_rate >= success_rate_threshold`
- 默认 `--min-success-steps 20`，`--success-rate-threshold 0.5`

输出结果字段在 [`eval_unrealzoo_single_agent.py`](/data/hdt/newtrackvla/eval_unrealzoo_single_agent.py:541)，包括：

- `success`: 0/1
- `collision`: 0/1
- `robotdog_following_rate` 和兼容字段 `following_rate`
- `near_rate`
- `success_distance`
- `require_success_distance`

最终汇总脚本是 [`tools/calculate_unrealzoo_single_agent_metrics.py`](/data/hdt/newtrackvla/tools/calculate_unrealzoo_single_agent_metrics.py:12)。它只统计 `model_type == "single_agent_robotdog"` 的 JSON：

```text
SR = mean(success) * 100
TR = mean(robotdog_following_rate) * 100
CR = mean(collision) * 100
```

## 旧 Habitat / EVT-Bench 评估

旧代码根目录是 `/data/hdt/OpenTrackVLA-main/`。入口 [`run_eval.py`](/data/hdt/OpenTrackVLA-main/run_eval.py:56) 做三件事：

- 用 Habitat config 创建 dataset。
- `dataset.get_splits(split_num)[split_id]` 切分评估集。
- 调用 [`trained_agent.evaluate_agent`](/data/hdt/OpenTrackVLA-main/trained_agent.py:41)。

每一步评估在 [`trained_agent.py`](/data/hdt/OpenTrackVLA-main/trained_agent.py:100)：

- 模型输出 `agent_1_base_velocity`。
- `env.step(action_dict)` 推进 Habitat。
- `info = env.get_metrics()` 读取 Habitat measure。
- 若 `info["human_following"] == 1.0`，`followed_step += 1`。
- 若机器人与人 3D 距离大于 4.0 连续超过 20 步，判定 `Lost`。
- 若 `info["human_collision"] == 1.0`，判定 `Collision` 并停止。

单 episode 输出在 [`trained_agent.py`](/data/hdt/OpenTrackVLA-main/trained_agent.py:152)：

```text
if iter_step < 300:
    success = human_following_success and human_following
else:
    success = human_following
following_rate = followed_step / iter_step
collision = human_collision
```

也就是说旧 Habitat 的 `TR` 是按每一步的 `human_following` 统计；`success` 则是 episode 结束时根据最后一次 Habitat measure 写入，且小于 300 步时还要求 `human_following_success`。

这些 Habitat measure 的定义在 [`evt_bench/additional_metric.py`](/data/hdt/OpenTrackVLA-main/evt_bench/additional_metric.py:130)：

- `DistanceToLeader`：机器人 agent 1 与主目标人 agent 0 的 3D 欧氏距离，见 [`additional_metric.py`](/data/hdt/OpenTrackVLA-main/evt_bench/additional_metric.py:163)。
- `HumanCollision`：若 `distance_to_leader < 0.5`，或历史上已碰撞过，则 `human_collision = 1.0`，见 [`additional_metric.py`](/data/hdt/OpenTrackVLA-main/evt_bench/additional_metric.py:151)。
- `HumanFollowing`：若 `distance_to_leader <= success_distance` 且检测器 `agent_1_main_humanoid_detector_sensor["facing"]` 为真，则 `human_following = 1.0`，见 [`additional_metric.py`](/data/hdt/OpenTrackVLA-main/evt_bench/additional_metric.py:237)。
- `HumanFollowingSuccess`：若 stop 被调用，且距离在 `[success_following_distance_lower, success_following_distance_upper]`，并且 `human_following` 为真，则 `human_following_success = 1.0`，见 [`additional_metric.py`](/data/hdt/OpenTrackVLA-main/evt_bench/additional_metric.py:283)。

默认阈值在 [`additional_metric.py`](/data/hdt/OpenTrackVLA-main/evt_bench/additional_metric.py:626)：

```text
human_following.success_distance = 3.0
human_following_success.lower = 1.0
human_following_success.upper = 3.0
```

旧汇总脚本有两个版本：

- [`calculate_metrics.py`](/data/hdt/OpenTrackVLA-main/calculate_metrics.py:6)：递归读取 JSON，跳过 `_info.json`。
- [`calculate_metrics2.py`](/data/hdt/OpenTrackVLA-main/calculate_metrics2.py:18)：支持 `stt / dt / at` 子目录和状态统计。

两者核心公式一致：

```text
SR = success_count / total_episodes * 100
TR = mean(following_rate) * 100
CR = collision_count / total_episodes * 100
```

## 直接对比

| 项目 | 当前 UnrealZoo 狗评估 | 旧 Habitat / EVT-Bench 评估 |
| --- | --- | --- |
| 每步跟踪 `following` | 现在显式要求 `target_visible && XY距离 <= 8.0m` | `human_following == 1.0`，内部是 3D距离 <= 3.0m 且 detector facing |
| 距离类型 | Unreal XY 平面距离，单位换算为米 | Habitat 3D 欧氏距离 |
| 可见性/朝向 | 目标 mask 可见性 `visible` | detector sensor 的 `facing` |
| 碰撞 | 优先读 UnrealZoo collision 信息；兜底为 `dist_xy < 0.7m && height_gap < 1.0m` | `distance_to_leader < 0.5` 后锁定 `human_collision = 1.0` |
| 丢失 | `following` 连续失败达到 `max_lost_steps=30` | 3D 距离 > 4.0 连续超过 20 步 |
| 单 episode 成功 | 不碰撞、不 Lost/Timeout、步数足够、`following_rate >= 0.5` | 若 `iter_step < 300`，要求最终 `human_following_success && human_following`；否则要求最终 `human_following` |
| SR | `mean(success) * 100` | `success_count / total * 100` |
| TR | `mean(robotdog_following_rate) * 100` | `mean(following_rate) * 100` |
| CR | `mean(collision) * 100` | `collision_count / total * 100` |

最重要的差异是：旧 Habitat 的 `human_following` 本身已经同时包含距离和 facing，默认距离阈值 3m；当前 UnrealZoo 之前默认只要目标可见就可算 following，但现在狗流水线已显式开启 `--require-success-distance`，所以会同时要求可见和 8m 内。当前阈值比旧 Habitat 宽，但多了 UnrealZoo mask 可见性约束，并且 success 是按整段 `following_rate >= 0.5` 判定，而旧 Habitat 的 success 更依赖 episode 结束时的 measure 状态。
