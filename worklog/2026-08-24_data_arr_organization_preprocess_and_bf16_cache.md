# 2026-08-24：data_arr 数据整理、训练预处理与三卡 BF16 视觉缓存

## 1. 目标与数据位置

本轮目标是将程昱杰整理的双智能体 `dt/stt` 原始录制按场景归档，并生成后续
TrackVLA 训练所需的抽帧、JSONL、聚合 JSON 和视觉特征缓存。

原始数据：

```text
/data/hdt/ntv_data/cyj/data_arr/
├── dt/
└── stt/
```

整理后的原始数据视图：

```text
/data/hdt/ntv_data/sim_data/data_arr/
├── cyj_dt/
└── cyj_stt/
```

训练预处理输出：

```text
/data/hdt/ntv_data/data/cyj_data_arr_processed/
├── frames/
├── jsonl/
├── dataset.json
├── vision_cache/
├── frame_chunks/
├── logs/
└── precision_compare/
```

## 2. 原始数据按场景整理

参考 `/data/hdt/ntv_data/sim_data/data7_29_dt/` 的场景分类方式，将 `dt` 和
`stt` 分别整理到 `cyj_dt`、`cyj_stt`。

源数据中不同批次和 seed 在同一场景下大量使用相同 episode 文件名，例如
`0.json`、`0_drone.mp4`。因此不能直接将所有文件平铺到场景目录，否则会发生覆盖。
最终使用如下层级：

```text
cyj_stt/<scene>/<source-batch-and-seed>/
```

示例：

```text
cyj_stt/UnrealTrack-DowntownWest-ContinuousColor-v0/
└── stt_camera1_auto__10.6__seed_3971/
```

整理采用软链接，不移动、不修改原始数据，也不额外复制约 226 GB 视频。各批次的
`train.json` 单独保存在 `_manifests/`。

整理结果：

| 类型 | 场景 | `train.json` | 软链接 | 损坏链接 |
| ---- | ---: | -----------: | -----: | -------: |
| dt   |   23 |           10 |  12,346 |        0 |
| stt  |   41 |           98 |  42,482 |        0 |

## 3. 抽帧与训练 JSON 生成

使用仓库现有双智能体预处理入口：

```text
tools/make_tracking_data.py --multi_agent
```

处理合同：

```text
双路 MP4
  -> drone/robotdog JPG 帧

双路 *_info.json
  -> 每 episode JSONL
  -> dataset.json
```

本轮保留所有完整配对 episode，未启用成功率、碰撞或 following-rate 过滤。主要参数：

```text
history=31
n_waypoints=8
dt=0.1
waypoint_label_source=recorded_pose_fixed_dt
ffmpeg_quality=2
min_agent_following_rate=0
```

最终结果：

| 类型 | Episode | 训练样本 |
| ---- | ------: | -------: |
| dt   |   2,433 |  712,869 |
| stt  |   7,745 | 1,972,885 |
| 合计 |  10,178 | 2,685,754 |

完整性状态：

```text
Skipped by status       = 0
Skipped by load/extract = 0
Skipped empty           = 0
JPG frames              = 5,514,000
dataset.json            = 约 53 GiB
```

抽帧和 JSONL 已全部完成。

## 4. 首次视觉缓存启动失败与修复

首次自动启动视觉缓存时，GPU 0、1、6 三个进程均立即退出：

```text
ModuleNotFoundError: No module named 'tools'
```

随后发现最初使用的 `navsim` Python 环境还缺少 `transformers`。最终统一改用：

```text
/home/hdt/miniconda3/envs/omtracknew/bin/python
PYTHONPATH=/data/hdt/newtrackvla修改/newtrackvla_base_yh_clean
```

最初的自动监控脚本没有检查视觉缓存子进程返回码，错误地写入了
`vision_cache_complete.marker`。该标志是无效的，完成状态应以后续分块日志、
缓存文件数量和最终完整性校验为准。

直接从约 53 GiB 的 `dataset.json` 或全部 JSONL 收集帧路径会产生较高启动内存；一次
读取 551.4 万行总清单也不够稳定。为此生成显式帧清单，并进一步切成每块约 10 万帧：

```text
frame_list.txt                  # 5,514,000 行
frame_list_gpu{0,1,6}.txt       # 每卡 1,838,000 行
frame_chunks/gpu{0,1,6}/        # 每卡 19 块
```

分块之间互斥，已存在的 `vfine/vcoarse` 会自动跳过，因此支持安全断点续跑。

## 5. BF16 编码加速

原缓存流程为 FP32 DINOv3/SigLIP 前向，再将结果保存为 FP16。单卡单进程吞吐约
15 frames/s，无法满足 4--6 小时目标。

在 `tools/precache_frames.py` 新增：

```text
--encoder_amp none|bfloat16|float16
```

正式任务使用 A100 原生 BF16 autocast：

```text
encoder_amp=bfloat16
image_size=384
vision_resize_mode=letterbox
batch_size=32
```

缓存文件合同保持不变，每张图片仍生成：

```text
*_vfine.pt
*_vcoarse.pt
```

经过单进程、每卡 4 进程和每卡 8 进程测试，最终使用：

```text
physical GPUs = 0,1,6
workers/GPU   = 8
total workers = 24
```

每卡显存约 34.4 GiB；稳定阶段三卡可达到满 GPU 利用率。实测总吞吐约
320--380 frames/s，预计全量耗时约 4--6 小时。继续增加 worker 会加剧 GPU 调度和
海量小文件写入竞争，不再保证正收益。

## 6. FP32 与 BF16 数值对照

从新数据中抽样 400 张图片，覆盖：

```text
dt drone       100
dt robotdog    100
stt drone      100
stt robotdog   100
```

同一批图片分别生成独立 FP32 和 BF16 缓存，不覆盖正式缓存。结果：

| 指标 | 全部 | vfine | vcoarse |
| ---- | ---: | ----: | ------: |
| 余弦相似度均值 | 0.999552 | 0.999192 | 0.999911 |
| 最低余弦相似度 | 0.997758 | 0.997758 | 0.999763 |
| 平均绝对误差 | 0.017864 | 0.026002 | 0.009726 |
| 相对 MAE | 3.24% | 4.42% | 2.06% |

少量 `vfine` 单元素存在较大离群误差，最大绝对误差为 8.96875；但整体特征方向高度
一致，`vcoarse` 尤其稳定。基于 4--6 小时处理目标，正式任务继续采用 BF16。

对照报告：

```text
/data/hdt/ntv_data/data/cyj_data_arr_processed/precision_compare/comparison.json
```

## 7. 当前进度（日志记录时刻）

视觉缓存目标：

```text
5,514,000 frames
11,028,000 cache files
```

截至本日志更新时：

```text
cache files generated = 3,233,027
progress              = 约 29.3%
missing               = 0
failed                = 0
```

由于每个 worker 当前仍在处理自己的首个大块，尚未产生 `.done` 分块标志；这不表示
任务停滞，应结合 worker 日志中的 `checked/generated/saved` 判断实时进度。

日志位置：

```text
/data/hdt/ntv_data/data/cyj_data_arr_processed/logs/precache_gpu{0,1,6}_w{0..7}.log
```

视觉缓存仍在运行，尚不能标记为全量完成。任务完成后还需要执行最终校验：

1. 缓存文件总数是否恰好为 JPG 数的两倍；
2. 每张 JPG 是否同时存在 `vfine` 和 `vcoarse`；
3. 是否存在损坏或不可加载的 `.pt`；
4. 57 个分块是否全部完成且 worker 无失败；
5. 修正或移除此前错误生成的完成标志。

