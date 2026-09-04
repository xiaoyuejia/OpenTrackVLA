# data_arr 实验数据持续维护表

> 用途：记录 `data_arr` 的数据总量、质量分类、训练/验证/测试划分、特殊场景子集和
> AT 派生数据计划。后续数据处理、重建 waypoint 或 replay 发生变化时，优先编辑本文件，
> 并在“变更记录”追加一行。
> 当前版本：v2  
> 最后更新：2026-08-25
> 对应 split manifest：`manifests/data_arr_7_1_2_v1/split_manifest.json`

## 1. 数据总量与质量分类

| 数据类别                   |              DT |             STT |             合计 | 当前用途/状态                    |
| -------------------------- | --------------: | --------------: | ---------------: | -------------------------------- |
| 原始 organized 数据        |           2,433 |           7,745 |           10,178 | 全部发现的 episode               |
| Core audited               |           2,433 |           7,014 |            9,447 | 正式 7:1:2 划分                  |
| YOLO pseudo-bbox           |               0 |             376 |              376 | Auxiliary train，不进入 val/test |
| Unresolved bbox/visibility |               0 |             355 |              355 | 排除，不进入任何正式 split       |
| **合计**             | **2,433** | **7,745** | **10,178** |                                  |

## 2. Core 数据划分：7:1:2

| Split               |              DT |             STT |            合计 |      Core 占比 |
| ------------------- | --------------: | --------------: | --------------: | -------------: |
| Train               |           1,580 |           5,018 |           6,598 |         69.84% |
| Val                 |             352 |             600 |             952 |         10.08% |
| Test                |             501 |           1,396 |           1,897 |         20.08% |
| **Core 合计** | **2,433** | **7,014** | **9,447** | **100%** |

说明：由于 source batch、目标轨迹签名和相关 leakage group 不可拆分，实际比例不是机械
精确的 70/10/20。当前 split assignment hash：

```text
6098f5a44993429042a6119e786b3a5ada584bf93e25eb8177a1499d46f60622
```

## 3. 特殊场景子集

| 子集                |            DT |             STT |            合计 | 用途                      |
| ------------------- | ------------: | --------------: | --------------: | ------------------------- |
| `train_lostmid`   |             0 |             305 |             305 | 仅训练；不作为测试 replay |
| `test_loststart`  |             0 |             913 |             913 | 失视恢复测试              |
| `test_standard`   |           501 |             483 |             984 | 普通测试                  |
| **Test 合计** | **501** | **1,396** | **1,897** | Full test                 |

报告测试结果时同时给出 `Full Test`、`Standard Test`、`LostStart Test`。

## 4. Waypoint 版本

| 版本        | 当前状态               | Episode assignment              | 用途                             |
| ----------- | ---------------------- | ------------------------------- | -------------------------------- |
| 8 waypoint  | 已有 JSONL，路径已验证 | 与 v1 split 完全一致            | 后续 horizon 消融/兼容实验       |
| 10 waypoint | 尚待重建               | 必须复用 v1 split，不得重新划分 | V3 主训练合同：原点 + 9 个未来点 |

对应文件：

- 8 waypoint：`manifests/data_arr_7_1_2_v1/waypoint8/`
- 10 waypoint：`manifests/data_arr_7_1_2_v1/waypoint10/`

## 5. AT 派生数据计划

当前方案：目标和 Drone/Dog 逐帧 pose 精确重放，干扰者使用固定 seed 重新生成 NavMesh
轨迹，评估 replay 使用 distinct deterministic appearance 并记录映射。AT instruction 参考
OmTrackVLA 的 `AT/train/instructions.json`。

| AT 数据用途               |              DT |         STT |            合计 | 状态                         |
| ------------------------- | --------------: | ----------: | --------------: | ---------------------------- |
| AT train replay           |           1,933 |           0 |           1,933 | AT-language 已构建；AT-full replay 暂未启动 |
| AT evaluation replay      |             500 |           0 |             500 | 仅 DT Test，记录干扰者 pose |
| **AT 当前计划总量** | **2,433** | **0** | **2,433** | train=1933 + eval=500 |

### AT-language train（DT train 指令派生）

已基于 v2 的 1,933 条 DT train 生成与 `dt`、`stt` 并列的：

```text
/data/hdt/ntv_data/data/cyj_data_arr_processed/jsonl/at/
```

该版本只替换 instruction，不复制帧、不修改 bbox、pose、waypoint 或视觉缓存。指令统一表达
“episode 初始化时指定并首先观察到的主要目标”，不将 Drone 全局视角解释为每帧最近邻选择。
每条记录还写入 `target_selection_policy=episode_initial_designated_target`、
`target_identity_policy=fixed_dt_target_pose_identity` 和 `distractors_are_non_target=true`。

对应产物：`at_language_manifest.json`、`at_language_instruction_assignment.json`、
`at_language_summary.json`、`at_language_SHA256SUMS.txt`，以及
`tools/build_data_arr_at_language.py` 构建脚本。
该版本是 `AT-language`（DT 视觉/轨迹 + AT 语言），不是带新增干扰者画面的 `AT-full replay`。

### AT train GPU 分片

| GPU            |         Episode |          估计帧数 | 当前状态   |
| -------------- | --------------: | ----------------: | ---------- |
| GPU 0          |             645 |           193,500 | train 分片已生成 |
| GPU 1          |             644 |           193,200 | train 分片已生成 |
| GPU 6          |             644 |           193,200 | train 分片已生成 |
| **合计** | **1,933** | **579,900** |            |

当前人数策略：

| 场景类型     | Target | Distractors | 总 human 数 |
| ------------ | -----: | ----------: | ----------: |
| 受限/室内    |      1 |           5 |           6 |
| 普通户外     |      1 |           7 |           8 |
| 大型开阔户外 |      1 |           9 |          10 |

GPU 0 的 1 episode smoke 已完成：300 帧耗时 402.98 秒，即 0.7445 steps/s/GPU。
按当前实现，三卡全量约需 59 小时，不能按 3–4 小时目标直接启动全量 replay。

评估 replay 不强制所有人同 appearance。当前 DT 500 条评估数据没有干扰者逐帧 pose，
因此使用 distinct deterministic appearance，并记录 `appearance_map`、干扰者起点/路径
和 `distractor_poses_per_frame`；如果未来源文件已有有效干扰者 pose，则优先直接复用。

## 6. 当前关键产物

| 产物                     | 路径                                                |
| ------------------------ | --------------------------------------------------- |
| 主 split manifest        | `manifests/data_arr_7_1_2_v1/split_manifest.json` |
| split summary            | `manifests/data_arr_7_1_2_v1/summary.json`        |
| AT plan summary          | `manifests/data_arr_at_v1/summary.json`           |
| AT replay plan generator | `tools/build_data_arr_at_replay_plan.py`          |
| AT replay renderer       | `tools/replay_data_arr_at.py`                     |
| AT-language train JSONL  | `/data/hdt/ntv_data/data/cyj_data_arr_processed/jsonl/at/` |
| AT-language manifest     | `/data/hdt/ntv_data/data/cyj_data_arr_processed/at_language_manifest.json` |
| split generator          | `tools/build_data_arr_7_1_2_manifests.py`         |

## 7. 变更记录

| 日期       | 变更                                                | 影响                                       |
| ---------- | --------------------------------------------------- | ------------------------------------------ |
| 2026-08-25 | 建立 data_arr 7:1:2 split                           | Core=9,447；Train/Val/Test=6,598/952/1,897 |
| 2026-08-25 | 固定`lostmid` train-only、`loststart` test-only | 新增特殊子集清单                           |
| 2026-08-25 | 保留 8 waypoint，创建 10 waypoint pending 清单      | 两种 waypoint 共用同一 assignment          |
| 2026-08-25 | 创建 AT train/eval replay plan                      | DT train=1,933；eval=500；GPU=0,1,6       |
| 2026-08-25 | 更新 AT 人数分档为 6/8/10 humans                    | 不低于原 DT 的 6 人配置                    |
| 2026-08-25 | GPU 0 完成 300 帧 AT smoke                          | 0.7445 steps/s/GPU；全量暂不启动           |
| 2026-08-25 | AT eval 改为仅 DT Test 500 条、不强制同 appearance | 记录 appearance 与 distractor pose        |
| 2026-08-27 | 基于 DT train=1,933 生成 AT-language train | `jsonl/at` 与 `dt/stt` 并列；仅 instruction 改写 |
| 2026-08-27 | 建立 `/data/hdt/ntv_data/data_final` 非破坏性整理视图 | raw/processed + train/eval + dt/stt/at；AT eval 295/500 已完成，其余 pending |

## 8. 更新规则

## 8. v2：无 Val、Test=1900

下一轮训练采用 [data_arr_train_test_v2](../manifests/data_arr_train_test_v2/README.md)，不再保留 Val：

| Split | DT | STT | 合计 |
|---|---:|---:|---:|
| Train | 1,933 | 5,614 | 7,547 |
| Test | 500 | 1,400 | 1,900 |
| Val | 0 | 0 | 0 |

STT Test 固定为：`loststart=560`、`standard=840`，比例 2:3。

为使总 Test 精确为 1,900，从 Test 移回 Train 的唯一 DT episode 是：

```text
dt/UnrealTrack-Desert_ruins-ContinuousColor-v0/dt_camera3__5__seed_100/8
```

该 episode 目标轨迹全局唯一，未引入轨迹泄漏。v2 assignment hash：

```text
55b320ba939b2f58703978b9f6818912d234b39915468a637cddaca19aeaa070
```

AT replay plan 已同步使用 v2：AT train=1,933，AT evaluation JSON=500。

每次修改数据后必须同步更新：

1. 对应 manifest 路径和生成时间；
2. DT/STT/合计数量；
3. quality tier 数量；
4. train/val/test 数量和比例；
5. `lostmid/loststart` 子集数量；
6. 8/10 waypoint 状态；
7. AT replay 完成数量、GPU 分片和吞吐；
8. 本表“变更记录”；
9. 对应 `SHA256SUMS.json` 或新的 assignment hash。
