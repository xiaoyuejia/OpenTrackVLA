# 2026-08-24 dt 目标指令标签补全

- 新整理数据：`/data/hdt/ntv_data/data/cyj_data_arr_processed/jsonl/dt`
- 参考元数据：`/data/hdt/ntv_data/sim_data/data_dt`
- 通过 `rel_run_dir` 中的相机目标组和 seed，精确映射到对应的 `human*.json` 外观标注。
- 2,433/2,433 个 dt episode 均找到唯一目标外观文件，无错误。
- 已将每条 dt JSONL 样本的泛化指令替换为目标描述指令，并增加：
  - `target_description`
  - `target_appearance_source`
- 指令包含性别、身高/体型（若有）、上衣、下装、头发和可用配饰，并明确忽略其他人、避免碰撞。
- 目标外观来源包括人工标注和 `human_visual_derived.json` 的视觉复核标注；不可观察字段不会被编入指令。
- 报告：`/data/hdt/ntv_data/data/cyj_data_arr_processed/dt_instruction_enrichment_report.json`
- 原始 `data_dt` 未修改。

## 训练配置注意事项

`train_airground_coop_v3.py` 会优先使用 YAML 中的 `instruction_override`、
`joint_instruction_override`、`agent1_instruction_override` 和
`agent2_instruction_override`。canonical 配置当前仍是泛化指令；使用本数据集
训练时必须将这些 override 设为 `null`，否则 JSONL 中的目标描述不会进入模型文本输入。

## dataset.json 重建

- 已从更新后的 dt/stt JSONL 流式重建 `/data/hdt/ntv_data/data/cyj_data_arr_processed/dataset.json`。
- 新文件大小约 55.35 GB，首条记录已验证包含 `target_description` 和
  `target_appearance_source`，临时文件已清理。
- JSONL 是训练时更推荐的输入形式；dataset.json 与 JSONL 的内容同步。
