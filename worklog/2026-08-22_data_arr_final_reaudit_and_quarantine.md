# 2026-08-22：`data_arr` 整理后最终复审与统一隔离

## 1. 任务与口径

对用户重新整理后的 `/data/hdt/ntv_data/cyj/data_arr/` 从零执行全量审计，
不直接沿用整理前的路径和结论。一条 `*_drone_info.json` 代表一个双 Agent
episode。

审计覆盖：

- drone/robotdog info JSON 和 episode JSON 可解析性；
- 两个 Agent 的 JSON 帧数一致性；
- 双路 MP4 可打开性、分辨率和视频/JSON 帧数一致性；
- 黑屏、纯色帧和长时间冻结；
- bbox 非有限值、越界、全程失效和长连续失效；
- `target_visible=True` 但 bbox 无效的监督矛盾；
- `target_visibility` 范围和 step 连续性；
- pose 非有限值、单帧超大跳变；
- Agent 在持续运动命令下长时间静止；
- `stt_camera3_loststart` / `stt_camera3_lostmid` 的人为目标丢失设计与真实异常分开统计。

## 2. 整理后的初始规模

```text
episode                     9,802
drone info                  9,802
robotdog info               9,802
drone MP4                   9,802
robotdog MP4                9,802
Agent-frame / agent     2,644,200
```

帧数分布：6,838 条为 300 帧，2,964 条为 200 帧。所有视频均为
640x480。

## 3. 完整性和视频复审

全部 9,802 条通过以下检查：

```text
JSON parse failure                0
missing paired Agent file         0
Agent JSON frame mismatch         0
video open failure                0
video/JSON frame mismatch         0
resolution mismatch               0
bbox out of 640x480 bounds        0
invalid visibility range          0
step discontinuity                0
black/blank video                 0
long frozen video                 0
```

对 618 条 bbox 长缺失、pose 长静止或其他候选进行了视频抽帧复核，未发现
新的黑屏或冻结视频。

## 4. 发现的监督标签异常

文件和视频结构完整，但 `stt/stt_camera2` 中仍有 355 条存在：

```text
target_visible = True
target_bbox    = [0, 0, 0, 0]  # 或其他无效框
```

矛盾 Agent-frame 数：

| Agent | 矛盾帧 |
| --- | ---: |
| Drone | 2,183 |
| RobotDog | 3,105 |
| 合计 | 5,288 |

该标注会使当前 `target_match` 监督产生错误负样本，因此不应继续留在
可训练主目录。用户授权移动异常数据后，已将这 355 条整体隔离。

## 5. 本次隔离操作

```text
隔离 episode    355
移动文件      1,775
隔离目录      /data/hdt/ntv_data/cyj/隔离/data_arr_reaudit_20260822
隔离大小      约 9.9 GB
```

回滚清单：

```text
/data/hdt/ntv_data/cyj/隔离/data_arr_reaudit_20260822/rollback_manifest.json
```

本次没有删除数据，每个文件的原路径、隔离路径和原因都保存在回滚清单中。

## 6. 统一隔离工作目录

`/data/hdt/ntv_data/cyj/隔离/` 已纳入本次工作目录盘点。当前包含：

| 隔离子目录 | Episode |
| --- | ---: |
| `data_arr_quarantine_bbox_full_failure_20260821` | 376 |
| `data_arr_quarantine_camera_blocked_or_stuck_20260821` | 12 |
| `data_arr_quarantine_final_audit_20260822` | 61 |
| `data_arr_reaudit_20260822` | 355 |
| 合计 | 804 |

统一隔离目录总大小约 22 GB。既有隔离子目录与各自的
`rollback_manifest.json` 保留，未混回主数据目录。

## 7. 隔离后最终状态

```text
/data/hdt/ntv_data/cyj/data_arr episode = 9,447
paired robotdog info                    = 9,447
paired drone video                      = 9,447
paired robotdog video                   = 9,447
target_visible=True + invalid bbox      = 0
bbox out of bounds                      = 0
invalid visibility / discontinuous step = 0
```

剩余 9,447 条中的 `target_visible=False` 但仍有非零 bbox 帧未归为异常：这是
低可见度/遮挡阈值判定，bbox 本身仍在画面内，训练时受 visibility mask 控制。

## 8. 审计产物

- `reports/data_arr_final_audit_20260822.csv`：9,802 条的逐条审计和 355 条隔离标记；
- `reports/data_arr_final_audit_20260822_summary.json`：隔离后摘要；
- `reports/data_arr_final_video_audit_20260822.csv`：视频完整性与抽帧结果；
- `reports/data_arr_bbox_semantic_reaudit_20260822.csv`：bbox/可见性/step 语义检查；
- `scripts/final_audit_data_arr.py`：JSON、bbox、pose 审计入口；
- `scripts/final_audit_data_arr_videos.py`：视频审计入口；
- `scripts/isolate_data_arr_reaudit_visible_bbox_conflicts.py`：本次隔离与回滚清单生成入口。

## 9. 结论与边界

按本次明确检查项，隔离后的 9,447 条未再发现结构错误、长时间卡住、
全程 bbox 缺失、可见性/bbox 矛盾、视频黑屏或冻结。

这一结论不等于逐像素证明所有非零 bbox 都是 exact GT；若要对每帧边界进行
最强保证，仍需 UE object mask 逐帧重算或人工标注复核。
