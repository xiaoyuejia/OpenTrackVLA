# 2026-08-25：data_arr 7:1:2 episode manifests

## 用户冻结约束

- 划分为 train/val/test = 7:1:2；
- `lostmid` 只进入 train，因为作为重放测试没有有效意义；
- `loststart` 全部进入 test；
- 当前 8-waypoint 数据保留用于后续实验；
- 未来 10-waypoint 重建必须复用同一 episode split。

## 数据分层

当前 organized/processed 总计 10,178 条：

| Tier | Episodes | 用途 |
|---|---:|---|
| core_audited | 9,447 | 正式 7:1:2 |
| YOLO pseudo-bbox | 376 | auxiliary train only |
| unresolved bbox/visibility | 355 | excluded |

这里使用 `core_audited` 而不是 `exact`，避免将现有审计结论夸大为逐像素 GT 保证。

## 防泄漏分组

发现 10,178 条 raw episode 与 8-waypoint JSONL 一一对应。对每条 episode 计算目标轨迹
签名，并通过并查集把以下关系连接为不可拆组件：

1. 相同 data kind + source batch；
2. 相同 scene + target trajectory signature。

轨迹签名每 5 帧采样一次 `target_pose_after_action`（缺失时回退 `target_pose`），相对首帧
归一化并按 10 cm / 1° 量化。最终得到 1,906 个 leakage groups。任何 source batch 或
trajectory signature 都没有跨 split。

## 最终数量

```text
train = 6598 (69.8423%)
val   =  952 (10.0773%)
test  = 1897 (20.0804%)
```

硬约束校验：

```text
lostmid train  = 305 / 305
loststart test = 913 / 913
```

测试集进一步保留：

```text
test_loststart = 913
test_standard  = 984
```

建议论文同时报告 full test、standard 和 loststart。仅报告 full test 会让 loststart 占测试权重
约 48.1%，不利于区分普通跟踪与失视恢复能力。

## 场景边界

总计 41 scenes，39 个同时覆盖 train/val/test。`AbandonedDistrict` 只有 1 条，
`InteriorDemo_NEW` 的 11 条受整组约束，二者保留 train-only，没有为追求表面覆盖率破坏
source/trajectory 隔离。

这套划分是 seen-scene + trajectory-disjoint，不是 unseen-scene split。由于 lostmid 和
loststart 在大量相同场景中分别被强制到 train/test，无法同时声称 scene-exclusive。
unseen-scene 应单独建立 challenge manifest。

## 8/10 waypoint

`waypoint8/` 清单已指向现有 JSONL 并逐条验证存在。`waypoint10/` 保存相同 episode key、
split 和期望相对路径，但 `jsonl=null`，明确标记 pending rebuild。重建 10-waypoint 时禁止
重新抽样或重新划分；8-waypoint 保留为 horizon 消融。

## 产物

```text
manifests/data_arr_7_1_2_v1/
tools/build_data_arr_7_1_2_manifests.py
```

独立验证通过：split/quality tier 互斥、全量覆盖、硬约束、source batch/trajectory 防泄漏、
8-waypoint 路径和全部 manifest SHA256。

