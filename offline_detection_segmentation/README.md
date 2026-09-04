# 纯 YOLO 离线人物检测与障碍实例分割

这个目录与现有训练代码完全解耦，只读取 RGB，不使用 JSONL 中的人物真值框。

处理规则非常明确：

1. 用 `YOLO11m-seg` 检测并分割画面中的前景实例；
2. 唯一且最高置信度的人物作为跟踪目标，保存人物框、置信度和 mask；
3. 其余所有检测实例不区分类别，mask 直接取并集作为统一障碍 mask；
4. 没被 YOLO 实例覆盖的像素全部为 unknown，不推断 free，不再使用 SegFormer。

这里特意使用闭集 YOLO，而不是 prompt-free YOLOE。YOLOE 会把天空、地貌或整个遗迹
作为开放词汇实例，全部合并后又会产生大面积假障碍。普通 COCO `YOLO11-seg` 不会做
这种场景级分割，但它也没有通用的树、墙和箱子类别；若这些障碍必须稳定检出，需要
用当前仿真域的实例 mask 微调自定义 YOLO 分割模型。

## 安装

建议使用已有 CUDA PyTorch 的 `omtracknew` 环境：

```bash
/home/hdt/miniconda3/envs/omtracknew/bin/pip install -r offline_detection_segmentation/requirements.txt
```

第一次推理会自动下载 `yolo11m-seg.pt`。

## 检查数据路径

下面的命令不加载模型，也不写文件：

```bash
python -m offline_detection_segmentation.precompute --dry-run --max-frames 16
```

默认读取 JSONL 中的 `agent1_current` 和 `agent2_current`。只有显式添加
`--include-history` 才处理历史图片，重复图片会自动去重。

## 小规模测试

测试结果也统一放在本目录：

```bash
python -m offline_detection_segmentation.precompute \
  --device cuda:4 \
  --max-frames 16 \
  --output-root offline_detection_segmentation/outputs/yolo_smoke
```

程序会自动保存检查图，无需再运行其他脚本：

```text
offline_detection_segmentation/outputs/yolo_smoke/visualizations/frames/.../*.perception.jpg
```

检查图中：

- 绿色：目标人物 mask、人物框和置信度；
- 红色：除目标人物外，所有 YOLOE 检测实例 mask 的并集；
- 原始颜色：未检测背景，也就是 unknown。

要跨多个场景测试，不要只用 `--max-frames` 连续取开头帧。下面会从 train、val
分别均匀选择 4 个不同场景，每个场景选 1 个 episode 和 2 个时刻，总计约 32 张：

```bash
python -m offline_detection_segmentation.precompute \
  --device cuda:4 \
  --scenes-per-split 4 \
  --episodes-per-scene 1 \
  --rows-per-episode 2 \
  --output-root offline_detection_segmentation/outputs/yolo_multiscene_32
```

生成按场景和视角汇总的统计与联系表：

```bash
python -m offline_detection_segmentation.summarize \
  --output-root offline_detection_segmentation/outputs/yolo_multiscene_32
```

它会在该输出目录生成 `evaluation_summary.json` 和 `contact_sheet.jpg`。

如果需要手动重新绘制某个缓存：

```bash
python -m offline_detection_segmentation.visualize \
  --cache offline_detection_segmentation/outputs/yolo_smoke/frames/替换为实际路径.perception.npz \
  --output offline_detection_segmentation/outputs/manual_preview.png
```

## 全量运行

确认小批量效果后，直接运行准备好的脚本：

```bash
bash offline_detection_segmentation/run_full.sh
```

默认在前台汇总显示所有 GPU 的带前缀输出，同时保留每卡日志；按 `Ctrl+C` 会停止全部
worker。只有确实需要脱离终端时才使用下面的 `nohup` 方式。

脚本默认使用物理 GPU 0、1、3、6，每卡 batch size 128，处理 train+val 的所有当前帧。
各卡通过确定性分片处理互不重叠的图片，并分别写入日志、汇总和失败记录。脚本不传
`--overwrite`，所以中断后执行同一个命令会跳过已有缓存并继续。若任一指定卡正在使用，
脚本会拒绝启动，不会挤占其他人的任务。指定其他空闲卡或每卡 batch size：

```bash
GPU_IDS=1,3,6 BATCH_SIZE=64 bash offline_detection_segmentation/run_full.sh
```

后台全量运行并保存日志：

```bash
nohup bash offline_detection_segmentation/run_full.sh \
  > offline_detection_segmentation/outputs/full_run.log 2>&1 &
```

查看进度：

```bash
tail -f offline_detection_segmentation/outputs/full_cache/run_gpu*_shard*of*_batch*.log
```

默认数据集和输出路径已经写在 [config.yaml](config.yaml)。权重、缓存、可视化、运行
统计和测试结果都保存在 `offline_detection_segmentation/` 内。断点重跑会跳过当前版本的
有效缓存；旧版 SegFormer 缓存会因为 schema 不一致而自动重算。只有显式添加
`--overwrite` 才无条件覆盖。

## 缓存字段

每张图片对应一个压缩 `.perception.npz`：

| 字段 | 形状/类型 | 含义 |
| --- | --- | --- |
| `scene_mask` | `H×W uint8` | 0 unknown，1 free（本版本不使用），2 obstacle，3 target |
| `mask_grid` | `8×8×4 float16` | 每格四种状态的像素占比 |
| `person_valid` | `bool` | 是否检测到唯一目标人物 |
| `person_box_xyxy` | `4 float32` | 人物像素框；无目标时全 0 |
| `person_box_cxcywh_norm` | `4 float32` | 归一化人物框 |
| `person_score` | `float32` | 人物检测置信度 |
| `obstacle_boxes_xyxy` | `N×4 float32` | 所有非目标实例框，类别已丢弃 |
| `obstacle_scores` | `N float32` | 非目标实例置信度 |
| `metadata_json` | JSON 字符串 | 原图、模型、阈值和 split/agent 溯源信息 |

当前 schema 是 `offline_perception.yolo_only.v2`。

## 参数位置

通常只需要修改 [config.yaml](config.yaml)：

- `models.yolo_weights`：YOLO 分割权重；
- `models.yolo_image_size`：输入分辨率；
- `thresholds.person_confidence`：人物阈值；
- `thresholds.object_confidence`：其他前景实例阈值；
- `target.person_labels`：哪些 YOLO 标签算人物；
- `runtime.device`、`runtime.batch_size`：GPU 与 batch size。

若画面中出现多个人，当前单目标设定只取置信度最高者，其余人物也进入障碍 mask。
