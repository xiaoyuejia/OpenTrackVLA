# data_dt 人物指令与严格 7:3 拆分工作记录

- 日期：2026-08-14
- 状态：已完成全部 2447 条数据的抽帧、JSONL、具体人物指令、视觉缓存和严格 1713/734 拆分；最终完整性校验通过

## 1. 数据位置

原始 2447 条录制数据：

```text
/data/hdt/ntv_data/sim_data/data_dt/
```

已扁平化的 1834 条原始数据：

```text
/data/hdt/ntv_data/sim_data/data7_29_dt/
```

已完成帧抽取、JSONL 生成和视觉特征缓存的 1834 条训练数据：

```text
/data/hdt/ntv_data/data/data7_29_dt_camera_m40_pose_fixed_dt_exact_bbox_global_base_train/
```

在本轮扩展前，该目录是 1834 条的已处理源数据，并不是最终的 1713 条训练 split。

2026-08-15 更新：该已处理源目录现已扩展为完整 2447 条，作为 train/eval 共享的底层数据源。最终严格拆分位于：

```text
/data/hdt/ntv_data/data/data_dt_camera_m40_pose_fixed_dt_exact_bbox_global_base_split_70_30/
```

其中：

```text
train/  # 1713 episodes
eval/   # 734 episodes
val -> eval
```

## 2. 已完成的完整性检查

- `data_dt` 共 2447 个完整 episode，覆盖 23 个场景。
- 每个 episode 都有状态 JSON、drone/robotdog 视频以及两路 info JSON。
- 现有已处理目录中有 1834 个非空 JSONL。
- 1834 个 JSONL 都存在对应的帧目录。
- 现有数据保留了 `frames/`、`jsonl/`、`vision_cache/` 和 `logs/` 数据合同。

## 3. 人物描述标注审计

人物标注格式为每个原始分组/场景目录下的 `human*.json`，包含 `person_id` 和 `appearance`，其中有性别、年龄、身高、体型、头发、服装、鞋、配饰和面部特征。

审计结果：

- 2205 条 episode 有唯一可审计的人物描述。
- 242 条 episode 所在的 20 个“分组×场景”目录没有 `human*.json`。
- 缺标注数据中，40 条属于现有 1834 条，202 条属于新增 613 条。
- 原始来源 `/data/hdt/cyj/multi_people/new_text_target/` 中同样没有这些标注。
- 不能直接复用同分组其他场景的人物描述；已通过目标框和视频帧确认，部分缺标注场景使用了不同人物外观，盲目复用会造成错标。

后续补充情况：

- 分组 1 和分组 3 的 4 个缺失目录已由用户补充原始格式的 `human*.json`。
- 用户授权根据录制视频中的目标框为分组 9/10 建立补充标注。
- 已对分组 9 的 10 个缺标注场景和分组 10 的 6 个缺标注场景逐场景抽取 robotdog 视角下可见率最高的目标框进行人工交叉审查。
- 分组 9 缺标注场景统一为：短深色头发、白色长袖正装衬衫、深灰色西裤、黑色正装鞋的成年男性。
- 分组 10 统一为：短棕色头发、灰色西装外套和同色西裤、棕色正装鞋的成年男性。
- 在上述 16 个目录中写入 `human_visual_derived.json`，使用非伪造的 `VISUAL_DERIVED_GROUP_009/010` person ID。
- 每份补充标注都保留 `annotation_audit.visual_derived=true`、来源分组/场景、证据 episode、视频、info、帧编号、target bbox 和可见率。
- 最终校验：191 个“分组×场景”目录、2447 条 episode 全部具有且只有一份人物标注；16 份 visual-derived 标注的证据路径和 target bbox 均通过校验。

## 4. 确认的严格 7:3 拆分

最终数量：

```text
train = 1713
eval  = 734
total = 2447
```

拆分约束：

1. 新增的 613 条全部进入评估集。
2. 从已处理的 1834 条中移出 121 条进入评估集。
3. 现有数据中缺人物标注的 40 条强制进入评估集。
4. 再从人物标注完整的现有数据中分层选择 81 条进入评估集。
5. 训练集留下的 1713 条全部具有唯一、可核验的 `human*.json`。
6. 训练和评估都覆盖 23 个场景。
7. `ContainerYard` 总共只有 2 条，保留 1 条训练、1 条评估。
8. 在上述硬约束下，尽可能均衡各场景的评估数量。

## 5. 指令生成规则

- `instructions_dt.json` 只用于参考英文动词和句式，不随机抽取人物描述。
- 人物属性必须来自 episode 映射到的原始 `human*.json`。
- 同一 episode 所有 JSONL 行使用完全一致的 instruction。
- 替换 instruction 时不改动轨迹、waypoint、帧路径、目标框或视觉缓存。
- 修改后需校验除 instruction 外的字段逐行不变。

## 6. 待执行项

本轮数据处理已全部完成，无剩余待执行项。

## 7. 2026-08-15 最终处理与校验结果

- 建立 2447 条稳定扁平化视图：`/data/hdt/ntv_data/sim_data/data_dt_all_flat/`。
- 原有 1834 条的场景内编号零变化，新增 613 条仅追加编号。
- 2447 个 JSONL 全部存在且非空，每个 episode 产生 291 个样本，共 712077 个样本。
- 2447 个双路帧目录全部存在。
- 5 条状态为 Success 但 following-rate 低于默认 0.8 的 episode 已明确取消该过滤后补齐，未静默丢弃。
- 使用 GPU 1/3/4/5/6 的 5 个互斥分片完成全量视觉缓存。
- 视觉缓存共检查 1424154 个双路帧引用，新生成 356766 个缺失帧缓存，5 个分片均为 `missing=0 failed=0`。
- `vision_cache/` 最终有 2848308 个文件，恰好是每帧一份 `vfine` 和一份 `vcoarse`。
- 2447 个 JSONL 全部替换为从对应 `human*.json` 生成的具体跟踪指令。
- 每个 JSONL 替换后均校验“移除 instruction 后的内容哈希不变”，全部通过。
- 指令映射审计文件：`instruction_audit.json`，共 2447 个唯一 episode，其中 202 条使用带完整证据链的 visual-derived 人物标注。
- 最终拆分为 train=1713、eval=734，无 episode 重叠，并集恰好为 2447。
- train/eval 均覆盖全部 23 个场景；`ContainerYard` 仅有 2 条，按 train=1/eval=1 保留双侧覆盖。
- train/eval JSONL 链接数分别为 1713/734，无断链；两侧共享的 `frames` 和 `vision_cache` 链接均有效。
