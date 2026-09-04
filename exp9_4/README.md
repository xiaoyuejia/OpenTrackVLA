# exp9_4：训练 → DT → AT → STT 闭环评估流水线

本轮设计决策、代码修复、smoke、正式训练快照和后续对比项统一记录在：
[`WORK_LOG.md`](WORK_LOG.md)。

## 1. 流水线顺序

```text
训练（stride=5, epochs=10）
  → 选择 epoch 10 final checkpoint
  → DT 500 条闭环评估 + 视频 + metrics.csv
  → AT 500 条闭环评估 + 视频 + metrics.csv
  → STT 1400 条闭环评估 + 视频 + metrics.csv
  → 汇总三任务 aggregate 指标
```

三组评估严格串行，不会同时启动多个任务。每个任务内部默认使用 2 张 GPU、每卡 1 个 UE worker。

## 2. 目录

```text
exp9_4/
├── pipeline.yaml                 # 流水线、GPU、评估和路径参数
├── run_pipeline.sh               # 主入口
├── config/
│   └── train_s5_e10.yaml         # 实际训练模板
├── manifests/
│   ├── eval_dt_500_recorded.json
│   ├── eval_at_500_recorded.json
│   └── eval_stt_1400_recorded.json
├── models/
│   └── airground_v3_s5_e10/      # checkpoint、train_log.csv、训练配置快照
├── results/
│   ├── dt/                       # DT JSON、视频、metrics.csv、worker 日志
│   ├── at/                       # AT 独立结果
│   └── stt/                      # STT 独立结果
├── summaries/
│   ├── dt_metrics.csv
│   ├── at_metrics.csv
│   ├── stt_metrics.csv
│   ├── metrics_all_tasks.csv
│   └── metrics_summary.md
├── state/                        # effective config、checkpoint 选择、阶段完成标记
├── runtime/                      # 三任务隔离的 UE runtime
└── logs/                         # pipeline 总日志
```

## 3. 关键参数

### 训练

| 参数 | 值 |
|---|---:|
| `train_temporal_stride` | 5（当前帧样本间隔 0.5 s） |
| waypoint 点间隔 | 0.1 s（recorded label 不重采样） |
| `epochs` | 10 |
| `candidate_top_k` | 8 |
| `batch_size` | 48/GPU |
| GPU 数 | 2 |
| `grad_accum_steps` | 2 |
| 有效全局 batch | 192 |
| `lr` | 4e-5 |
| 初始化 | scratch，不加载任何 AirGround checkpoint；Qwen 保持预训练冻结权重 |
| checkpoint | 每 2000 step + 每 epoch final，不裁剪 |
| receiver curriculum | `progressive_linear_10stage_v1`，每 epoch 一个小台阶 |
| target reference | 禁用（训练、validation、正式评估均传零且 valid=false） |

### 均匀渐进难度设置

旧训练在 epoch 4 将 receiver corruption 从 easy stage 一次性切换到 hard stage，导致 cooperative loss 在 step 8017 突然增至约 4.9 倍。exp9_4 将这一跨度拆成 10 个小台阶，每个 epoch 只增加少量难度，第 10 epoch 才达到原 hard stage：

| Epoch | Assistance | ROI | Current | Recent | All | Pose perturb | Max translation | Max yaw |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | .700 | .700 | .300 | .000 | .000 | .000 | .250 m | 10.000° |
| 2 | .717 | .667 | .300 | .028 | .005 | .056 | .278 m | 12.222° |
| 3 | .733 | .633 | .300 | .056 | .011 | .111 | .306 m | 14.444° |
| 4 | .750 | .600 | .300 | .083 | .017 | .167 | .333 m | 16.667° |
| 5 | .767 | .567 | .300 | .111 | .022 | .222 | .361 m | 18.889° |
| 6 | .783 | .533 | .300 | .139 | .028 | .278 | .389 m | 21.111° |
| 7 | .800 | .500 | .300 | .167 | .033 | .333 | .417 m | 23.333° |
| 8 | .817 | .467 | .300 | .194 | .039 | .389 | .444 m | 25.556° |
| 9 | .833 | .433 | .300 | .222 | .045 | .444 | .472 m | 27.778° |
| 10 | .850 | .400 | .300 | .250 | .050 | .500 | .500 m | 30.000° |

每个阶段四种 corruption mode 的概率和均为 1。

候选接受阈值初始固定为 `0.50`。短程 smoke/早期 validation 重点查看 CSV 中的 `candidate_positive_probability`、`candidate_max_negative_probability` 和 `candidate_threshold_target_recall`；只有正负分数明显重叠时才调整阈值。正式评估从 `pipeline.yaml: evaluation.target_match_confidence_threshold` 读取阈值，并通过 `AIRGROUND_V3_TARGET_MATCH_THRESHOLD` 覆盖 checkpoint 配置，因此 smoke 调阈值不需要重新训练。AT 与 bbox 中心偏差本轮暂不改结构，留作后续 ablation。

新训练使用 `init_mode: scratch`，`state/effective_train.yaml` 中强制写入 `runtime.init_ckpt: null`，不加载任何历史 AirGround checkpoint。这里的“从头训练”指所有 AirGround 新增模块从随机初始化开始；冻结的 Qwen3-0.6B 仍使用其正常预训练权重，离线视觉 token 也不重新训练视觉编码器。若 exp9_4 自身训练中断且 model 目录已有 checkpoint，流水线才使用 `--resume` 恢复本实验 optimizer/scheduler。

### 评估

| 参数 | 值 |
|---|---:|
| 顺序 | DT → AT → STT |
| episodes | 500 → 500 → 1400 |
| `policy_inference_stride` | 5（模型调用间隔 0.5 s） |
| `policy_action_rollout` | `future_segment` |
| drone 跟随范围 | 1.0–6.0 m |
| robotdog 跟随范围 | 1.0–6.0 m |
| `dt` | 0.1 s |
| 视频尺寸/FPS | 640×480 / 10 FPS |
| 视频 | drone、robotdog、global，带轨迹 overlay |
| bbox oracle | 禁用，GT 只算指标 |

评估每 5 个环境步（0.5 s）执行一次模型；中间 4 步依次使用预测的 future waypoint segment。必须区分三种时间：policy interval=`5×0.1=0.5 s`，但环境控制步长和预测 waypoint 相邻点的 recorded dt 仍为 `0.1 s`。速度、yaw rate、feasible recovery 和 inverse-fixed-dt 都按相邻 waypoint/action 的 `0.1 s` 计算；若直接改成 `0.5 s`，会把速度缩小 5 倍并破坏训练标签物理含义。

## 4. 视频位置

每个任务的 MP4 位于对应 `results/<task>/workers/...` 场景目录中：

```text
<episode>_drone.mp4
<episode>_robotdog.mp4
<episode>_global.mp4
```

注意：2400 个 episode 全量保存三视角视频会消耗大量磁盘并增加编码时间。若空间不足，可在 `pipeline.yaml` 中把 `write_global_video` 改为 `false`，保留双 follower 视频。

## 5. 使用方式

只检查规划和数据条数，不运行：

```bash
bash exp9_4/run_pipeline.sh --stage plan
```

打印完整命令但不执行：

```bash
bash exp9_4/run_pipeline.sh --stage all --dry-run
```

完整执行：

```bash
nohup bash exp9_4/run_pipeline.sh --stage all \
  > exp9_4/logs/launcher.out 2>&1 &
```

分阶段执行：

```bash
bash exp9_4/run_pipeline.sh --stage train
bash exp9_4/run_pipeline.sh --stage eval-dt
bash exp9_4/run_pipeline.sh --stage eval-at
bash exp9_4/run_pipeline.sh --stage eval-stt
bash exp9_4/run_pipeline.sh --stage summarize
```

只运行完整评估序列：

```bash
bash exp9_4/run_pipeline.sh --stage eval --ckpt /absolute/path/to/checkpoint.pt
```

## 6. 续跑规则

- `state/train.done` 存在：跳过训练。
- `state/eval_<task>.done` 和对应 `metrics.csv` 都存在：跳过该任务。
- 评估 launcher 默认 resume：已完成 episode 不重复执行。
- 每组评估结束后立即生成并保存独立 `metrics.csv`。
- 汇总阶段只读取三个任务各自的 aggregate 行，不把 DT/AT/STT 混成一个总体均值。

## 7. 启动前注意

旧训练已在 step 11802 停止，目前 GPU 已释放。流水线仍会检测其他训练进程并拒绝抢占 GPU；确需并发时可显式设置 `ALLOW_CONCURRENT_TRAIN=1`，但不推荐。
