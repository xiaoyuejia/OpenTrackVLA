# 2026-08-19：150 条闭环评估修复、最终指标与数据盘点

## 1. 背景与问题

`output/eval_airground_coop_v3_receiver_target_125` 的 150 条固定闭环评估首次只完成了
78 条，剩余 72 条没有继续执行。失败运行标识为：

```text
v3_run_20260816_212753_2938302
```

4 个 worker 均在首个待评估场景加载录制人体轨迹时触发 `FileNotFoundError`，重试 5 次
后退出。根因是 `manifests/total.json` 中 5 个条目指向实际不存在的
`/data/hdt/ntv_data/sim_data/data_lost/14/seed_100/` 场景目录：

- `UnrealTrack-ContainerYard_Night-ContinuousColor-v0`
- `UnrealTrack-Real_Landscape-ContinuousColor-v0`
- `UnrealTrack-RussianWinterTownDemo01-ContinuousColor-v0`
- `UnrealTrack-Stadium-ContinuousColor-v0`
- `UnrealTrack-StonePineForest-ContinuousColor-v0`

这不是模型或评估运行时错误，而是评估 manifest 中存在失效数据路径。

## 2. 数据修复

已在 `manifests/total.json` 中将上述 5 个失效条目替换为同场景、格式一致且未与
manifest 其他 episode 重复的有效录制：

- ContainerYard_Night 使用 `data7_8_by_camera/new_effective_dt/.../0`；
- Real_Landscape、RussianWinterTownDemo01、Stadium、StonePineForest 使用
  `data_lost/13/seed_100/.../1`；
- 同时修正了 Desert_ruins 条目的 stem 与 info 文件名不一致导致的 episode key 重复。

修复后的完整性检查：

```text
manifest entries       = 150
unique episode names   = 150
missing drone info     = 0
missing robotdog info  = 0
already complete       = 78
pending                = 72
```

## 3. 续跑与完成状态

使用 GPU 3、4、5、6，各 1 个 worker，从已有 78 条结果断点续跑剩余 72 条：

```text
run tag = v3_run_20260818_171437_246668
worker loads = [16, 19, 19, 18]
```

续跑完成后 `metrics.csv` 包含 150 条 episode 行和 1 条 aggregate 行，严格达到预期
的 150 条评估结果。最终结果文件：

```text
output/eval_airground_coop_v3_receiver_target_125/metrics.csv
```

## 4. 最终 150 条评估指标

| 指标                   |     结果 |
| ---------------------- | -------: |
| Episode                |      150 |
| Success                |      128 |
| TargetStopped          |       22 |
| Success rate           | 85.3333% |
| Collision rate         |  0.0000% |
| Human collision rate   |  0.0000% |
| Average steps          | 309.9533 |
| Average FPS            |  0.56785 |
| Joint tracking rate    | 81.4607% |
| Drone tracking rate    | 84.7131% |
| RobotDog tracking rate | 94.0239% |
| Drone centered rate    | 95.2026% |
| RobotDog centered rate | 93.7915% |
| Visible accuracy       | 95.3302% |
| Drone bbox IoU         | 83.7753% |
| RobotDog bbox IoU      | 86.5439% |

## 5. 当前训练、验证与闭环评估数据量

canonical V3 当前配置
`config/airground_cooperative_tracking_v3.yaml` 使用的数据拆分为：

| Split      | Episode | 帧级训练样本 |
| ---------- | ------: | -----------: |
| Train      |    2142 |       623322 |
| Validation |     913 |       265683 |
| 合计       |    3055 |       889005 |

当前实际完成的固定闭环评估清单 `manifests/total.json` 为 150 条 episode。该 150 条是
独立闭环评估口径，不应与训练配置中的 913 条 validation 混称。

## 6. `data_dt` 与 `data_lost` 原始录制盘点

统计口径：一个 `*_drone_info.json` 代表一个双智能体录制 episode，并检查同 stem 的
`*_robotdog_info.json` 是否存在。

### data_dt

```text
/data/hdt/ntv_data/sim_data/data_dt/
```

| 分组 | Episode |
| ---- | ------: |
| 1    |     222 |
| 2    |     197 |
| 3    |     297 |
| 4    |     317 |
| 5    |     265 |
| 6    |     297 |
| 7    |     239 |
| 8    |     278 |
| 9    |     257 |
| 10   |      78 |
| 合计 |    2447 |

### data_lost

```text
/data/hdt/ntv_data/sim_data/data_lost/
```

| 分组 | Episode |
| ---- | ------: |
| 11   |     297 |
| 12   |      80 |
| 13   |     253 |
| 14   |     283 |
| 合计 |     913 |

两个目录合计：

```text
data_dt + data_lost = 2447 + 913 = 3360 episodes
complete drone/robotdog pairs = 3360
missing robotdog companion = 0
orphan robotdog info = 0
disk bytes = 89054352638（约 89.05 GB / 82.94 GiB）
```

## 7. 结论

- 150 条闭环评估的数据路径问题已修复，剩余 72 条已续跑完成；
- 最终成功率为 85.33%，128/150 成功，无碰撞；
- 当前 V3 训练/验证共 3055 条 episode；
- `data_dt` 与 `data_lost` 原始目录共 3360 条完整双智能体录制，无配对缺失。

## 8. FPS 口径修正与单条实测

`fps` 保留为 episode 端到端速度，包含 UE 截图、UnrealCV 传输、YOLO、
视觉编码、模型、动作发送与后处理。新增的 `model_latency_ms` / `model_fps`
只计算已经准备好的模型输入进入 V3 `forward` 到输出返回的时间。CUDA 计时
前后都调用 `torch.cuda.synchronize()`，避免把异步提交时间误当成真实推理时间。

同一条 ContainerYard_Night stem 32、311 步实测：

| 配置                                      |      端到端 FPS |            模型延迟 |       模型 FPS |
| ----------------------------------------- | --------------: | ------------------: | -------------: |
| 旧 I/O、640x480、保存视频                 |           0.678 |           265.53 ms |           3.77 |
| 快速 I/O、无视频、BF16                    |           1.128 |           262.77 ms |           3.81 |
| 384x288、YOLO FP16、模型/渲染同 GPU       |           1.235 |           263.26 ms |           3.80 |
| 384x288、YOLO FP16、模型 GPU 3 / UE GPU 1 | **1.356** | **193.91 ms** | **5.16** |

将 Vulkan 渲染与 CUDA 模型分卡后，端到端速度相对旧基线提升约 100%，
纯模型前向延迟下降约 26.3%。

## 9. 端到端瓶颈定位

最快单 worker 的每步约 736.8 ms，分项如下：

| 分项                             |   每步耗时 |  占比 |
| -------------------------------- | ---------: | ----: |
| UE snapshot（位姿 + RGB + mask） |   218.2 ms | 29.6% |
| V3 模型前向                      |   193.9 ms | 26.3% |
| 动作后位姿刷新                   |   158.0 ms | 21.4% |
| DINO/SigLIP 编码                 |    84.3 ms | 11.4% |
| YOLO                             |    34.1 ms |  4.6% |
| UE resume/pause                  |    34.6 ms |  4.7% |
| IPC、路由与其他                  | 约 13.8 ms |  1.9% |

因此 CPU 核数不是主瓶颈；最大瓶颈是串行 UE/UnrealCV 渲染与读回。机器为
2 x Xeon Platinum 8368（76 物理核 / 152 线程）、440 GiB 内存、7 x A100-SXM4 80GB；
内存充足。GPU 0--3 属 NUMA 0，GPU 4--6 属 NUMA 1。单卡运行时可用
`taskset -c 0-37,76-113` 将 GPU 0--3 的任务绑定到 NUMA 0，但这只是次要优化。

## 10. 单模型 GPU 的 worker 数实测

固定 GPU 3 运行一份模型服务，384x288、YOLO FP16、快速 I/O、无视频，
每条限制 60 步。模型服务内有全局 `threading.Lock`，所有 worker 的模型请求
实际上串行；增加 worker 的作用是在某路 UE 截图/动作时，用另一路填充模型空闲。

| worker / 渲染配置 | Episode / 步 |     墙钟 |                 批吞吐 |        单条 FPS |  模型延迟 | planner/步 |
| ----------------- | -----------: | -------: | ---------------------: | --------------: | --------: | ---------: |
| 1 / GPU 1         |      4 / 240 | 536.98 s |           0.447 step/s | **1.356** | 188.60 ms |  316.61 ms |
| 2 / 共用 GPU 1    |      4 / 240 | 671.33 s |           0.357 step/s |           0.603 | 183.54 ms |  458.33 ms |
| 2 / GPU 0,1       |      4 / 240 | 430.71 s |           0.557 step/s |           1.031 | 188.79 ms |  466.73 ms |
| 4 / GPU 0,1,4,5   |      4 / 240 | 256.18 s |           0.937 step/s |           1.038 | 198.46 ms |  478.60 ms |
| 5 / GPU 0,1,4,5,6 |      5 / 300 | 266.62 s | **1.125 step/s** |           0.862 | 206.79 ms |  719.62 ms |

注：5-worker 为了让每路都有任务，比前四组多加了一个 Brass_Gardens 场景，
因此墙钟不能直接比较，应比较“总步数 / 墙钟”。5-worker 批吞吐比 4-worker 高约
20.1%，但 planner 含排队延迟显著上升，已接近单模型服务的饱和边界。

结论：

- 一张 GPU 同时承担模型和 UE 渲染：`workers-per-gpu=1` 最佳；
- 单条低延迟：1 worker，约 1.36 FPS；
- 单模型 GPU + 独立渲染 GPU：4 worker 是更稳健的平衡点；
- 只追求 150 条整批尽快完成：当前空闲硬件下 5 worker 吞吐最高，但单条延迟更高；
- 不建议超过 5 worker/模型 GPU，第 6 路预计主要增加模型锁排队，且当前没有第 6 张空闲独立渲染卡。

最大吞吐的单模型卡调用样例：

```bash
bash sh/eval_airground_coop_v3.sh \
  --gpu-ids 3 \
  --workers-per-gpu 5 \
  --render-gpu-ids 0,1,4,5,6 \
  --manifest manifests/total.json \
  --total-episodes 150 \
  --save-path output/<new_eval_name>
```

为支持该配置，`sh/eval_airground_coop_v3.sh` 已新增 `--render-gpu-ids`：可传一个 GPU ID
供所有 worker 共用，或传与总 worker 数相同的 GPU ID 列表实现逐 worker 分配。

## 11. 优化后评估加速结论（2026-08-22 复核）

基于本文第 8--10 节的同条 episode 和多 worker 实测，当前评估优化的
合理结论是：

- 单路端到端评估由旧版 `0.678 FPS` 提升到最快 `1.356 FPS`，
  实测加速为 `1.356 / 0.678 = 2.00x`。
- 相对 150 条旧评估结果的实际平均 `0.56785 FPS`，最快单路配置约为
  `2.39x`；但该数字包含旧运行中场景、I/O 和执行差异，不作为严格的
  同条件加速比。
- 整批任务的建议对外口径为约 `1.5--2.0x`：单路完全分卡可达
  `2.0x`，多 worker 会受模型服务串行锁、UE 启动和场景切换影响。

150 条固定闭环评估的平均步数为 `309.9533`，预计总步数约为：

```text
150 * 309.9533 = 46,493 steps
```

不计入额外的失败重试和异常场景长时间阻塞时，耗时估算为：

| 评估方式                         |     使用的吞吐 | 150 条估算耗时 | 相对旧单路 |
| -------------------------------- | -------------: | -------------: | ---------: |
| 旧版单路同条件基线               |   0.678 step/s |      约 19.0 h |      1.00x |
| 旧 150 条结果的平均 FPS          | 0.56785 step/s |      约 22.7 h |      0.84x |
| 优化后单路、模型/渲染分卡        |   1.356 step/s |       约 9.5 h |      2.00x |
| 单模型 GPU + 5 个独立渲染 worker |   1.125 step/s |      约 11.5 h |      1.66x |

因此，对当前 150 条评估的保守预期为：从旧流程的约 `19--23 h`
缩短至约 `9.5--12 h`。

加速主要来自：

1. `--fast-eval-io`，并关闭评估视频、全局视频和轨迹 overlay 写入；
2. 视觉编码输入使用 `384x288` letterbox；
3. V3 模型使用 BF16，实测快速配置中 YOLO 使用 FP16；
4. 同一模型 GPU 只加载一份共享推理服务；
5. 使用多个 UE worker 填充模型等待期；
6. 通过 `--render-gpu-ids` 将 UE Vulkan 渲染与 CUDA 模型推理分到不同 GPU。

限制与适用条件：

- `2.00x` 是“快速 I/O + 无视频 + BF16 + YOLO FP16 + 384x288 + 模型/渲染分卡”
  的实测结果，不是任意启动参数下都能自动获得的速度。
- 当前固定评估启动器已默认使用快速 I/O、无视频和共享模型服务；
  要复现最快数字，仍需显式指定独立 `--render-gpu-ids`，并确保 YOLO FP16
  等快速参数与实测配置一致。
- 5 worker 相比 4 worker 的实测批吞吐提高约 `20.1%`，但已接近单模型
  服务的饱和点；继续增加 worker 预计主要增加排队和显存占用。

## 12. 3 GPU / 1800 条 / 24 小时优化与 10 条实测（2026-08-22）

### 12.1 目标换算

最终 10 条样本平均为 `310.7 steps/episode`，因此 1800 条约为：

```text
1800 * 310.7 = 559,260 steps
559,260 / 86,400 = 6.473 step/s
```

要在 24 小时内完成，3 张卡的稳态总吞吐必须不低于
`6.473 step/s`，并应给地图切换、启动和偶发重试留出余量。

### 12.2 实现的速度优化

1. 新增 `--policy-inference-stride`，策略不必每个 10 Hz 物理步都重新推理。
2. 新增 `--policy-action-rollout future_segment`，中间步沿上次预测的未来
   轨迹逐段执行，而不是重复第一个动作。
3. 新增 `--metric-mask-stride`，速度模式每 5 步采样一次精确 PNG
   object mask，中间步零阶保持；目标停止等待阶段恢复每步精确
   mask，保证最终成功判定使用当前可见性。
4. 新增 `--reuse-post-action-poses`。确定性评估中，动作后已经获得
   精确位姿，世界随后处于 pause；下一步取图时不再重复查询同一
   组位姿。无 RGB/mask 的中间步连空 UnrealCV 请求和摄像机刷新
   也一并跳过。
5. 新增 `--deterministic-pause-check-stride`。最终配置每 50 步校验一次
   UE pause 状态，而非每步额外往返两次；10 条中所有周期校验均通过。
6. UE 输出使用 `384x288`，YOLO FP16，关闭视频；三张卡每卡一个
   独立模型服务和一个 UE worker。

旧默认保持不变：`policy-inference-stride=1`、`metric-mask-stride=1`、
`reuse-post-action-poses=false`、`deterministic-pause-check-stride=1`。

### 12.3 反优化路线的实测排除

- 同一物理 GPU 上运行两个 UE worker 会使动作步从单 UE 的约
  `223--258 ms` 增加到 `786--910 ms`，总吞吐反而降低。
- object mask 改为 BMP 虽然省去 PNG 压缩，但使本地 socket 传输量
  大幅增加；双 UE 实测 mask snapshot 达 `0.83--0.90 s/step`，已排除。
- 策略 stride=5 虽达到约 `7.76 step/s`，但 10 条 SR 只有 `30%`；
  stride=4 的小样本质量更稳定，因此最终不选 stride=5。

### 12.4 10 条递进测试

| 配置                                       |           墙钟 |                批吞吐 |            三卡稳态吞吐 |            SR |        Joint TR |         碰撞 |
| ------------------------------------------ | -------------: | --------------------: | ----------------------: | ------------: | --------------: | -----------: |
| stride=3，mask 每步，384x288               |          14:38 |           3.54 step/s |          约 4.86 step/s |           50% |           64.5% |           0% |
| stride=5，mask=5，复用位姿                 |           9:52 |           5.25 step/s |             7.76 step/s |           30% |           64.4% |           0% |
| stride=4，mask=4，复用位姿                 |          10:38 |           4.87 step/s |             7.07 step/s |           60% |           71.1% |           0% |
| **stride=4，mask=5，pause check=50** | **9:42** | **5.34 step/s** | **7.9005 step/s** | **50%** | **66.1%** | **0%** |

注：10 条清单由场景调度为 `4/3/3`，所以最后一个 episode 期间只有
GPU 2 工作；小批整体墙钟吞吐不能直接外推均衡的 1800 条任务。
`7.9005 step/s` 是分别以每个 worker 的总 steps / 总 loop seconds 计算后求和：

```text
GPU 2: 1238 steps / 492.647 s = 2.5130 step/s
GPU 3:  936 steps / 342.498 s = 2.7329 step/s
GPU 4:  933 steps / 351.454 s = 2.6547 step/s
total:                            7.9005 step/s
```

最终热路平均分项：

| 分项                          |  每步平均 |
| ----------------------------- | --------: |
| snapshot                      |  59.88 ms |
| planner（已按 stride=4 摊薄） | 118.78 ms |
| action + post-action pose     | 200.59 ms |
| 完整 loop                     | 381.91 ms |

### 12.5 1800 条最终估算

```text
559,260 steps / 7.9005 step/s = 70,789 s = 19.66 h
```

相对于第 11 节旧单路 `0.678 step/s`，三卡稳态总吞吐为
`11.65x`；相对于之前 5 条完整评估的批吞吐 `1.214 step/s`，为
`6.51x`。

按不利的 `10%--15%` 地图切换、调度和短暂波动余量估算，总时间约
`21.6--22.6 h`，仍低于 24 小时。若出现 UE 异常场景长时间重试，仍可能
突破 24 小时，因此正式评估应保留现有 resume 和 scene timeout 保护。

最终入口已固化为：

```bash
GPU_IDS_24H=2,3,4 \
  bash sh/eval_airground_coop_v3_24h_3gpu.sh \
  --manifest <1800-episode-manifest.json> \
  --save-path output/<eval-name>
```

速度模式的口径限制：物理步、动作后位姿、距离与碰撞仍然每步
精确评估；策略视觉推理为 2.5 Hz，GT mask 为 2 Hz 采样加中间步
零阶保持，仅目标停止终止阶段恢复每步精确 mask。因此该输出必须
标注为 `24h speed profile`，不应与 `metric-mask-stride=1` 的严格 10 Hz
可见性/TR 指标不加说明地混合比较。
