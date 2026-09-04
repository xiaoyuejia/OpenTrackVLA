# exp9_4 工作日志

> 用途：记录本轮 AirGround-Coop V3 的设计决策、代码修改、smoke、正式训练状态和后续对比重点。
>
> 最后更新：2026-09-05 04:15 CST

---

## 1. 本轮实验目标

1. 使用同一套冻结 Qwen3-0.6B、CandidateMatcher、grounding head 和 waypoint head 联合训练 STT、DT、AT。
2. AirGround 可训练模块全部 scratch 初始化；不加载任何历史 AirGround checkpoint。
3. 保留 target reference 实现供后续消融，但 exp9_4 的训练、validation 和评估全部禁用 reference。
4. Top-8 候选分别输出独立 sigmoid 匹配概率；允许通过绝对阈值拒绝全部弱候选。
5. 训练 current-row stride=5、10 epoch；闭环 policy stride=5。
6. 保留环境物理步和 waypoint 标签的原始 0.1 s 时间语义。
7. 完成 scratch 训练，然后按 DT 500 → AT 500 → STT 1400 串行闭环评估并汇总。

---

## 2. 固定实验设置

### 2.1 数据

| 项目 | 设置 |
|---|---|
| 训练 episode | STT 5614、DT 1933、AT 1933，合计 9480 |
| validation episode | 2400 |
| 训练 manifest | `/data/yh/data/manifests/train_joint.json` |
| validation manifest | `/data/yh/data/manifests/val_joint.json` |
| vision cache | `/data/yh/data/processed/vision_cache` |
| perception cache | `/data/yh/data/processed/perception_cache` |
| DT 评估 | `exp9_4/manifests/eval_dt_500_recorded.json` |
| AT 评估 | `exp9_4/manifests/eval_at_500_recorded.json` |
| STT 评估 | `exp9_4/manifests/eval_stt_1400_recorded.json` |

DT 保留目标外观 instruction；AT 保留模糊初始目标规则，不注入具体外观答案。`task_type` 只用于审计，不切换专用模型或 loss。

### 2.2 模型与 grounding

| 项目 | 设置 |
|---|---|
| Qwen | Qwen3-0.6B，加载本地预训练权重并冻结 |
| 候选数 | Top-8 |
| grounding 输出 | 8 个独立 sigmoid 概率，不使用第 9 个 NULL 类 |
| 正样本 | 最大 IoU 且 IoU≥0.30 的唯一候选 |
| 负样本 | 同视角其余有效候选 |
| loss | 正负组等权 balanced BCE |
| 训练候选顺序 | 随机打乱，并同步打乱特征、valid、IoU 和标签 |
| 推理接受规则 | 最高概率候选且概率≥0.50 |
| margin | 只记录，不参与接受 |
| reference | `use_target_reference=false` |

训练阶段使用 argmax hard-forward 和 softmax soft-backward，使选中候选上下文显式影响 ACT；推理阶段使用单一硬选择和阈值拒绝。

### 2.3 优化与保存

| 项目 | 设置 |
|---|---:|
| epoch | 10 |
| batch/GPU | 48 |
| GPU | 2 |
| grad accumulation | 2 |
| 有效 batch | 192 |
| 学习率 | 4e-5 |
| warmup | 300 optimizer steps |
| scheduler | cosine |
| grad clip | 1.0 |
| checkpoint | 每 2000 optimizer steps及每 epoch final |
| checkpoint 裁剪 | `max_ckpts=0`，全部保留 |
| receiver curriculum | 10 阶段，每个 epoch 增加一个小台阶 |

训练模板：`exp9_4/config/train_s5_e10.yaml`  
有效运行配置：`exp9_4/state/effective_train.yaml`  
正式输出：`exp9_4/models/airground_v3_s5_e10`

---

## 3. stride=5 时间合同

必须区分 policy 周期、环境物理周期、视觉历史周期和 waypoint 周期：

| 时间量 | 当前值 | 说明 |
|---|---:|---|
| current-row 训练采样间隔 | 0.5 s | 每 5 个 recorded row 选一个训练样本 |
| policy/Qwen 调用间隔 | 0.5 s | 每 5 个环境步运行一次完整 policy |
| environment step dt | 0.1 s | UnrealZoo 每次闭环动作推进一个物理步 |
| target replay dt | 0.1 s | 每个环境步推进一条录制目标动作 |
| history frame dt | 0.1 s | 与训练 JSONL 的连续历史一致 |
| waypoint source dt | 0.1 s | 标签没有重采样，不能改成 0.5 s |
| waypoint horizon | 7 | 8 个 origin-inclusive waypoint：原点 + 0.1～0.7 s |

### 3.1 两次完整 policy 之间的处理

当前不是重复同一图像、也不是重复同一个动作：

1. 第 0、5、10…物理步执行在线 YOLO、DINO+SigLIP、VERIFY/Qwen、路由和 waypoint 解码。
2. 中间 4 个物理步继续读取新 RGB，只执行 DINO+SigLIP coarse 编码并写入 0.1 s 历史；不额外执行 YOLO、VERIFY 或 Qwen。
3. 上一次 policy 输出的轨迹按 `future_segment` 依次执行 offset 1、2、3、4。
4. 每个 offset 都将剩余 ego-frame 轨迹重定位到当前 segment 原点，再交给 0.1 s inverse-fixed-dt 控制器。
5. GT object mask 每 0.1 s 获取一次，只用于指标，不输入模型。

这样既维持 0.5 s 的大模型调用周期，又避免将稀疏的 0.5 s 图像错误填充成训练中的 0.1 s 连续历史。

### 3.2 若以后做“严格五帧才看一次”消融

不能直接把同一 RGB 重复五次。应同时：

- 将训练历史改为每 5 帧采样；
- 将 `history_frame_dt` 改为 0.5 s；
- 明确 31 帧历史将覆盖 15.5 s；
- 评估中完全跳过中间 RGB；
- waypoint/action 仍可维持 0.1 s future segment。

该方案属于新的时间消融，不能与当前 exp9_4 指标直接混用。

---

## 4. 正式训练前发现并修复的问题

### 4.1 独立 sigmoid 初始化饱和

**现象**：最初 scratch smoke 中，正候选和负候选概率均约为 `0.99997`。  
**原因**：原 cosine/temperature 项适合相对 softmax 排序，但直接加入独立 sigmoid logit 时数值过大。  
**修复**：在 `candidate_matching.py` 增加可训练 `cosine_gain`，从 0 初始化。cosine 证据继续保留，但由 balanced BCE 决定何时启用。

### 4.2 全无正候选时出现 NaN

**现象**：单卡 smoke 的一个全无监督 batch 出现 NaN。  
**原因**：全无效候选 gather 到有限最小 logit，`softplus(-finite_min)` 先溢出为 `inf`，之后 `inf*0` 污染 masked mean。  
**修复**：在计算 softplus 前把无监督行替换为 0，并增加全无匹配候选回归测试。

### 4.3 残留 margin gate

独立工具函数 `select_top_candidate()` 仍使用旧 margin gate，虽然正式模型没有调用它，但语义与实验协议不一致。现已改成：只按最高 sigmoid 概率和阈值接受，margin 仅用于诊断。

### 4.4 评估 waypoint horizon 错误

**现象**：评估默认 `waypoint_horizon_steps=9`，但当前模型实际输出 8 个 origin-inclusive waypoint。  
**影响**：尤其会把 RobotDog 相邻 waypoint 的时间差错误解释为 0.2 s，导致速度缩小。  
**修复**：改为 `waypoint_horizon_steps=7`，对应 index 1..7 的 0.1..0.7 s。

### 4.5 stride=5 视觉历史分布偏移

**原问题**：完整 policy 每 0.5 s 调用，若中间不读取 RGB，历史只能用稀疏帧或重复帧，与训练的 0.1 s 连续历史不一致。  
**修复**：增加 `observe()` 轻量路径，中间物理步只编码 coarse 视觉并更新 history，不执行额外 Qwen forward。共享 GPU server 和远程 worker proxy 均支持该操作。

---

## 5. UnrealZoo 评估启动问题及修复

### 5.1 UE runtime 旧绝对路径

原默认路径指向不存在的 `/data/hdt/...`，导致 worker runtime 准备阶段直接失败。现默认使用仓库中的：

`unrealzoo/Linux/UnrealZoo_UE5_6_Linux_v3.0.0`

Engine 和 Content 使用软链接共享，只复制 worker 私有可执行文件和 `unrealcv.ini`。

### 5.2 本地视觉模型未加载

原评估没有设置 `DINOV3_MODEL_PATH`，会尝试访问受限 HuggingFace DINOv3 仓库并得到 401。现 `sh/eval_airground_coop_v3.sh` 强制检查并使用：

- `models/vision/dinov3`
- `models/vision/siglip`

### 5.3 Gym 与新版 setuptools 不兼容

当前 setuptools 84 已移除 `pkg_resources`，旧 Gym 在 `gym.make()` 时失败。现使用等价 importlib entry-point loader；环境快速注册仍由精确 `UNREALZOO_FAST_ENV_ID` 完成。

### 5.4 动态相机缓存 KeyError

`set_population()` 会生成新跟随相机，但 UnrealCV 的 `self.cam` 只在首次连接时初始化。相机数量增长后访问新 ID 会出现 `KeyError`。现 `update_camera_assignments()` 会先补齐新增相机的位置、旋转和 FOV 缓存。

### 5.5 环境注册、颜色和外观检查

- 每个 scene worker 在导入 `gym_unrealcv` 前设置精确 `UNREALZOO_FAST_ENV_ID`。
- `ContinuousColor` 被解析为 `action=Continuous`、`observation=Color`。
- replay 在扩展 population 后重新建立 object color dictionary。
- replay metadata 中的 target/distractor appearance ID 在首次 observation 前恢复。
- UnrealCV/BGR 输入进入视觉模型前显式转换为 RGB，并增加颜色通道回归测试。

---

## 6. Smoke 记录

### 6.1 训练 smoke

- 单 GPU 40 optimizer steps：全部有限；全无监督 step 正确得到零 loss，不再 NaN。
- 双 GPU、正式 `batch_per_gpu=48`、`grad_accum_steps=2`：完成 20 optimizer steps，约 3840 样本。
- 无 DDP unused parameter、NCCL、NaN 或 OOM。
- 峰值显存约 92.4 GiB/GPU。
- grounding 从正/负概率约 `0.522/0.524` 发展到 `0.537/0.484`，确认梯度链路有效。

### 6.2 真实 UnrealZoo 闭环 smoke

使用一份 scratch 1-step 临时 checkpoint，仅验证运行链路，不代表模型质量：

- scene：`UnrealTrack-Brass_Gardens-ContinuousColor-v0`
- task：DT replay
- 物理步：15
- policy stride：5
- 完整 policy step：1、6、11
- 中间 rollout offset：1、2、3、4
- policy interval：0.5 s
- environment/history/waypoint dt：均为 0.1 s
- waypoint horizon：7
- 环境创建、快速注册、appearance 恢复、颜色输入、在线视觉模型、YOLO、Qwen、动作 rollout、指标输出全部成功。

结果目录：`exp9_4/results/smoke_stride5`。

### 6.3 测试

最终相关测试：

```text
pytest -q tests
66 passed
```

全仓库裸跑 `pytest -q` 会额外收集内嵌 habitat-lab，并因当前环境缺少可选依赖 `magnum` 在 collection 阶段停止；这不属于 AirGround V3 测试失败。

---

## 7. 正式训练状态快照

启动时间约：2026-09-05 03:52 CST。  
记录时间：2026-09-05 04:15 CST。  
状态：Epoch 1，optimizer step 约 247，训练仍在运行。

最近 50 optimizer steps 均值：

| 指标 | 均值 | 判断 |
|---|---:|---|
| total loss | 17.421 | 相比初始化约 401 明显下降 |
| navigation loss | 0.152 | 正常下降 |
| SELF loss | 0.058 | 正常 |
| COOP loss | 0.094 | 从初始化约 3.74 明显下降 |
| JEPA loss | 0.580 | 稳定下降 |
| belief loss | 0.172 | 明显下降 |
| mode loss | 1.354 | 略低于 `log(4)=1.386`，仍需后续观察 |
| target-match loss | 0.164 | grounding 正常学习 |
| Top-8 recall | 98.08% | 正常 |
| Top-1 match accuracy | 96.40% | 正常 |
| candidate class accuracy | 93.71% | 正常 |
| 正候选概率 | 0.903 | 已明显高于阈值 |
| 最大负候选概率 | 0.311 | 与正候选已分离 |
| 0.50 阈值目标召回 | 94.21% | 正常改善 |
| no-target false accept | 9.08% | 仍需 validation 标定 |
| final EPE | 0.302 m | 明显优于初始化约 1.03 m |
| grad norm pre-clip | 825.5 | 仍持续触发 clip=1.0 |
| clip scale | 0.00136 | 需继续监控，但当前各 loss 确实在学习 |

截至该快照，全部 CSV 数值有限，无 NaN/Inf。两卡利用率均为 100%，训练稳定。偶发单 batch loss 30～60 属于任务、遮挡与 receiver corruption 混合造成的正常波动，目前没有持续抬升。

`loss_uncertainty` 可以为负，因为它包含可学习 log-variance/NLL 项；只要总 loss 和其他监督项有限且稳定，就不表示数值错误。

---

## 7.1 checkpoint → 评估衔接复核

2026-09-05 对运行中的有效配置和代码路径再次核对：

1. 当前每个 epoch 有 2672 optimizer steps，正常完成 10 epoch 时预计总 step 为 26720。
2. 训练器在第 10 个 epoch 结束后保存 `model_epoch010_step026720_final.pt`；若中途续训导致 step 变化，文件名仍保持 `model_epoch010_step*_final.pt`。
3. pipeline 只匹配 `model_epoch010_step*_final.pt`，不会误选每 2000 step 的普通 checkpoint、较早 epoch final、`best_val.pt` 或 smoke checkpoint。
4. 训练命令必须正常退出且 epoch-10 final 文件存在，pipeline 才写入 `state/train.done` 和 `state/evaluation_checkpoint.txt`；缺失时立即报错，不会开始评估。
5. DT、AT、STT 都从同一个 `evaluation_checkpoint.txt` 读取绝对路径，并在启动前再次检查文件存在。
6. 已用训练器真实生成的 scratch checkpoint 完成严格模型加载和一条 UnrealZoo 闭环 smoke，证明 checkpoint payload、模型 state dict、共享推理 server 和评估 loader 兼容。
7. 当前有效配置确认：`epochs=10`、`save_every=2000`、`save_every_epochs=1`、`max_ckpts=0`、`max_steps=0`、`init_ckpt=null`。

因此，在训练正常完成且存储/硬件没有外部故障的前提下，自动选择 epoch-10 final 并进入 DT → AT → STT 的逻辑闭环完整，不会误选 checkpoint。

---

## 8. 后续重点对比

1. **step 500 左右**：复查 pre-clip grad norm 是否继续下降；若长期保持四位数且 grounding 停止改善，再考虑调整 scratch warmup 或 loss 权重。
2. **epoch 1 final**：重点检查 mode loss 是否开始明显低于随机基线 1.386。
3. **validation**：统计正候选、最大负候选和 no-target max probability 分布，再决定是否覆盖 0.50 阈值。
4. **curriculum epoch 边界**：检查 COOP、JEPA、belief 和 grad norm 是否只有小台阶，不再出现旧 epoch 4 的突变。
5. **epoch 10 final**：确认 checkpoint 同时包含 model、optimizer、scheduler 状态。
6. **闭环评估**：确认 debug 时间字段固定为：
   - `policy_interval_seconds=0.5`
   - `environment_step_dt_seconds=0.1`
   - `history_frame_dt_seconds=0.1`
   - `waypoint_step_dt_seconds=0.1`
   - `waypoint_horizon_steps=7`
7. 后续再做 bbox-only、坐标置零、appearance 置零以及中心/边缘分桶消融。

---

## 9. 常用命令

### 一键启动或断点续跑完整流水线

```bash
cd /data/yh/newtrackvla修改/newtrackvla_base_yh_clean && bash exp9_4/start_all.sh
```

### 查看 pipeline 日志

```bash
tail -f "$(cat exp9_4/state/latest_pipeline_log.txt)"
```

### TensorBoard

```text
http://127.0.0.1:6006/
```

### 流水线预检

```bash
bash exp9_4/run_pipeline.sh --stage plan
```

---

## 10. 本轮关键修改文件

- `candidate_matching.py`
- `model_airground_coop_v3.py`
- `train_airground_coop_v3.py`
- `train_airground_v3_common.py`
- `eval_airground_coop_v3.py`
- `eval_airground_coop_v3_server.py`
- `eval_airground_v3_runtime.py`
- `eval_unrealzoo_multi_agent.py`
- `unrealzoo-gym/gym_unrealcv/envs/base_env.py`
- `sh/eval_airground_coop_v3.sh`
- `sh/prepare_unrealzoo_worker_runtimes.sh`
- `exp9_4/config/train_s5_e10.yaml`
- `exp9_4/pipeline.yaml`
- `exp9_4/run_pipeline.sh`
- `exp9_4/start_all.sh`
- `tests/test_candidate_matching.py`
- `tests/test_airground_coop_v3.py`
- `tests/test_eval_airground_coop_v3.py`

注意：仓库中存在大量本轮之前的修改和未跟踪文件，不执行整体 reset。
