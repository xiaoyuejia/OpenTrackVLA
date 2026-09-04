# AirGround-Coop V3

本仓库只保留 canonical receiver-target AirGround-Coop V3。V3 由两条视角隔离的
SELF 流和一条双视角 COOPERATIVE 流组成：自身可见时独立跟踪，单侧失视时
由另一 Agent 引导重新捕获目标。

> **当前 grounding 更新**：保留训练静态/评估动态 target reference 的代码，但 exp9_4
> 通过 `use_target_reference=false` 在训练、validation 和评估中全部禁用；一个 VERIFY 与
> Top-8 候选逐一匹配，每个候选输出独立 sigmoid 概率。训练使用最大 IoU 唯一正样本、
> 其余有效候选负样本的正负等权 BCE；训练候选顺序随机打乱。推理选择最高概率候选，
> 仅当概率达到 `0.50` 时接受，不设置第九个 `NO_TARGET` 类。本文后部残留的 K+1/NULL
> 说明属于此前版本，应以本说明和 `训练与评估审查9_4.md` 为准。

# 代码审查

## 训练审查

 一句话总结

 训练时不读原始 RGB 图片。真正进入训练 pipeline 的是磁盘上三套离线预处理产物：

1. JSONL 标注文件 —— 每条样本的文本指令 + 标签 + 帧路径索引；
2. 视觉特征缓存 —— 每帧图像预先用 DINO + SigLIP 提取好的 token；
3. 感知缓存 —— 每帧图像预先用 YOLO 跑好的人物候选 + 栅格。

 原始图片只在"离线预处理阶段"被读过一次，训练时直接用缓存。

 ────────────────────────────────────────────────────────────────────────────────

1. JSONL 标注文件（最原始的结构化数据）

 路径：/data/yh/data/processed/train_jsonl/{stt, dt, at}/<场景>/<episode></episode>/<编号>.jsonl

 每个 .jsonl 文件是一个 episode，每一行是一个训练样本（一帧）。一行 JSON 的关键字段：

```text
   schema_version          multi_agent_tracking_v5_global_waypoint                                                                                                              
   episode_id              stt/UnrealTrack-.../14                                                                                                                               
   step_index              0                                                                                                                                                    
   instruction             目标指令（STT/DT/AT 不同）                                                                                                                           
   dt                      0.1                                                                                                                                                  
   history                 31                                                                                                                                                   
   n_waypoints             8                                                                                                                                                    
   waypoint_label_source   recorded_pose_fixed_dt                                                                                                                               
                                                                                                                                                                                
   agent_order             [drone, robotdog]                                                                                                                                    
   agents: {                                                                                                                                                                    
     drone: {                                                                                                                                                                   
       current      frames/.../drone/frame_00001.jpg   ← 当前帧路径                                                                                                             
       images       [...]                              ← 历史帧路径                                                                                                             
       bbox         [cx, cy, w, h]                     ← GT 目标框                                                                                                              
       bbox_valid_mask  true                                                                                                                                                    
       target_visible    true                                                                                                                                                   
       pose / target_pose                              ← 位姿                                                                                                                   
       waypoints    [[x,y,yaw] × 8]                    ← 标签轨迹                                                                                                               
       valid_mask   [bool × 8]                                                                                                                                                  
     },                                                                                                                                                                         
     robotdog: { ...同上... }                                                                                                                                                   
   }
```

- STT 的 instruction：Follow the target person without collision.
- DT 的 instruction：Follow the average-built short man wearing white long-sleeve dress shirt...（具体外观描述）
- AT 的 instruction：Pursue the first person selected at the beginning of the episode.（模糊，指代初始目标；还带有 target_description、target_selection_policy 字段，但训练代码
  不读取它们）

 ────────────────────────────────────────────────────────────────────────────────

2. 视觉特征缓存（图片 → token）

 路径：/data/yh/data/processed/vision_cache/{stt,dt,at}/...

 每张图片对应两个 .pt 文件：

```text
   frame_00001_vcoarse.pt   shape [4, 1536]   float16   ← 每帧 4 个粗 token（历史用）                                                                                           
   frame_00001_vfine.pt     shape [64, 1536]  float16   ← 每帧 64 个细 token（当前用）
```

 这些 token 的生成过程（tools.precache_frames）：

```text
   原始 jpg 图片                                                                                                                                                                
     → DINO 编码器  → tok_dino                                                                                                                                                  
     → SigLIP 编码器 → tok_sigl                                                                                                                                                 
     → 拼接 → [D + S] = 1536 维                                                                                                                                                 
     → grid_pool 池化                                                                                                                                                           
          ├─ out_tokens=4  → vcoarse（历史粗粒度）                                                                                                                              
          └─ out_tokens=64 → vfine（当前细粒度）
```

 所以训练时 model 拿到的 coarse_tokens/fine_tokens 就是这些缓存。

 ────────────────────────────────────────────────────────────────────────────────

3. 感知缓存（YOLO 离线检测结果）

 路径：/data/yh/data/processed/perception_cache/{stt,dt,at}/...

 每帧一个 .perception.npz，包含：

```text
   person_box_cxcywh_norm   YOLO 主检测框 [cx,cy,w,h]                                                                                                                           
   person_score            置信度                                                                                                                                               
   person_valid            是否有效                                                                                                                                             
   mask_grid               [8,8,4] 栅格（unknown/free/obstacle/target）                                                                                                         
   obstacle_boxes_xyxy     障碍物框                                                                                                                                             
   person_candidates_xyxy   Top-K 人物候选框（多人任务）                                                                                                                        
   person_candidate_scores 候选置信度                                                                                                                                           
   metadata_json           图像宽高等
```

 另外每个 agent 还有一个 .candidates.npz（person_candidates.bundle.v1），存 Top-K 候选的 [cx,cy,w,h,score]，这就是模型里的 candidate_feat。

 ────────────────────────────────────────────────────────────────────────────────

4. 完整加载链路（磁盘 → 一条训练样本）

```
   train_json 目录（stt/dt/at 三个子目录）                                                                                                                                      
           │                                                                                                                                                                    
           ▼                                                                                                                                                                    
   JSONL 一行 → get_example(idx) 读原始 dict                                                                                                                                    
           │                                                                                                                                                                    
           ├─ _load_agent_tokens：读 vision_cache                                                                                                                               
           │     ├─ 31 帧历史 × 4 token  → coarse_tokens [2, 124, 1536]                                                                                                         
           │     └─ 当前帧 64 token        → fine_tokens   [2, 64, 1536]                                                                                                        
           │                                                                                                                                                                    
                                                                                                                   
           │                                                                                                                                                                    
           ▼
```

 ────────────────────────────────────────────────────────────────────────────────

5. 关键结论

 ┌───────────────────────┬───────────────────────────────────────────────────────────┐
 │ 问题                  │ 答案                                                      │
 ├───────────────────────┼───────────────────────────────────────────────────────────┤
 │ 训练时读原始 jpg 吗？ │ 否，除非 cache 缺失且开 online_encode_missing             │
 ├───────────────────────┼───────────────────────────────────────────────────────────┤
 │ 视觉信息是什么？      │ DINO+SigLIP 预提取的 1536 维 token（vcoarse/vfine）       │
 ├───────────────────────┼───────────────────────────────────────────────────────────┤
 │ 检测信息是什么？      │ YOLO 预提取的 person box / Top-K 候选 / 栅格              │
 ├───────────────────────┼───────────────────────────────────────────────────────────┤
 │ 标签是什么？          │ JSONL 里的 waypoints + valid_mask + bbox + pose + visible │
 ├───────────────────────┼───────────────────────────────────────────────────────────┤
 │ 指令是什么？          │ JSONL 里的 instruction 字段（STT/DT/AT 各自不同）         │
 └───────────────────────┴───────────────────────────────────────────────────────────┘

 一句话：

 │ 最原始进入训练的是 JSONL 标注 + 预提取的 DINO/SigLIP 视觉 token + 预提取的 YOLO 感知缓存，而不是实时图像。图像在离线预处理阶段就被"翻译"成了特征 token 和检测框。



 每个形状数字的含义

1. waypoints → [2, 8, 3]

```text
   [2,  8,  3]                                                                                                                                                                  
    │   │   │                                                                                                                                                                   
    │   │   └─ 每个 waypoint 3 个动作值：[x, y, yaw]                                                                                                                            
    │   │        x  = 相对当前位置的前进/左右位移（米）                                                                                                                         
    │   │        y  = 相对当前位置的侧向位移（米）                                                                                                                              
    │   │        yaw= 朝向角变化（弧度）                                                                                                                                        
    │   └───── 未来 8 个时刻的 waypoint（n_waypoints=8）                                                                                                                        
    └───────── 两个 agent：drone 和 robotdog                                                                                                                                    
```

 实际例子（STT 第一帧 drone）：

```python
   waypoints[0] = [                                                                                                                                                             
       [0.0, 0.0, 0.0],                    # waypoint 0：锚定为原点，valid_mask=False 不监督                                                                                    
       [-0.0519, -0.0394, 0.1656],         # 第 1 步                                                                                                                            
       [-0.0964, -0.0742, 0.2505],         # 第 2 步                                                                                                                            
       ... 共 8 个                                                                                                                                                              
   ]                                                                                                                                                                            
```

 ────────────────────────────────────────────────────────────────────────────────

2. detection_feat → [2, 6]

```text
   [2,  6]                                                                                                                                                                      
    │   │                                                                                                                                                                       
    │   └─ 6 维 = [cx, cy, w, h, score, valid]                                                                                                                                  
    │        cx    = 目标框中心 x（归一化 0~1）                                                                                                                                 
    │        cy    = 目标框中心 y（归一化 0~1）                                                                                                                                 
    │        w     = 目标框宽度（归一化）                                                                                                                                       
    │        h     = 目标框高度（归一化）                                                                                                                                       
    │        score = YOLO 置信度（0~1）                                                                                                                                         
    │        valid = 该检测是否有效（0/1）                                                                                                                                      
    └───────── 两个 agent                                                                                                                                                       
```

 这是 YOLO 的主检测框（top-1），不是 Top-K。

 ────────────────────────────────────────────────────────────────────────────────

3. perception_grid → [2, 8, 8, 4]

```text
   [2,  8,  8,  4]                                                                                                                                                              
    │   │   │   │                                                                                                                                                               
    │   │   │   └─ 每个格子的 4 个通道：[unknown, free, obstacle, target]                                                                                                       
    │   │   │        unknown  = 未知区域                                                                                                                                        
    │   │   │        free     = 可通行/空闲区域                                                                                                                                 
    │   │   │        obstacle = 障碍物区域                                                                                                                                      
    │   │   │        target   = 检测到目标人物的区域                                                                                                                            
    │   │   └────── 8 列（perception_grid_size=8）                                                                                                                              
    │   └────────── 8 行（8×8 = 64 个格子）                                                                                                                                     
    └────────────── 两个 agent                                                                                                                                                  
```

 即把画面划分成 8×8 栅格，每个格子有 4 个"语义通道"。

 ────────────────────────────────────────────────────────────────────────────────

4. candidate_feat → [2, K, 6]

```text
   [2,  K,  6]                                                                                                                                                                  
    │   │   │                                                                                                                                                                   
    │   │   └─ 6 维 = [cx, cy, w, h, score, valid]（与 detection_feat 相同含义）                                                                                                
    │   │        cx, cy, w, h = 候选框（归一化）                                                                                                                                
    │   │        score        = 候选置信度                                                                                                                                      
    │   │        valid        = 该候选是否有效                                                                                                                                  
    │   └────── K 个候选人物（candidate_top_k=8，最多 8 个）                                                                                                                    
    └────────── 两个 agent                                                                                                                                                      
```

 这是 YOLO 检测出的最多 K 个人物候选，模型要从中选出"谁是目标"。

 ────────────────────────────────────────────────────────────────────────────────

 汇总对照

 ┌─────────────────┬──────────────┬──────────────────────────────────────────────────────────┐
 │ 张量            │ 形状         │ 每个数字含义                                             │
 ├─────────────────┼──────────────┼──────────────────────────────────────────────────────────┤
 │ waypoints       │ [2, 8, 3]    │ agent × 未来时刻 × [x,y,yaw]                             │
 ├─────────────────┼──────────────┼──────────────────────────────────────────────────────────┤
 │ detection_feat  │ [2, 6]       │ agent × [cx,cy,w,h,score,valid]（YOLO 主框）             │
 ├─────────────────┼──────────────┼──────────────────────────────────────────────────────────┤
 │ perception_grid │ [2, 8, 8, 4] │ agent × 栅格行 × 栅格列 × [unknown,free,obstacle,target] │
 ├─────────────────┼──────────────┼──────────────────────────────────────────────────────────┤
 │ candidate_feat  │ [2, K, 6]    │ agent × 候选数 × [cx,cy,w,h,score,valid]（Top-K）        │
 └─────────────────┴──────────────┴──────────────────────────────────────────────────────────┘

 │ 注意：2 在所有形状里都是同一个含义 —— 两个 agent（drone 和 robotdog）。



 关键数字汇总：

- D = 1024（Qwen3-0.6B hidden）
- 视觉特征维度 = 1536（DINO+SigLIP 拼接）
- 历史帧数 = 31，每帧 4 token → 124 token，+31 时间 token = 155
- 当前帧 = 64 fine token，+1 时间 token = 65
- 候选 = 8（K）
- reference = 1
- VERIFY = 1, ACT = 1
- text ≤ 128

 送入 LLM 的 inputs_embeds: (2B, ≤128 + 229 + 2, 1024) = (2B, ≤359, 1024)

 等等，229 = 1 + 155 + 65 + 8 = 229。是的。

 所以单行序列长度 = text_len + 229 + 2 = text_len + 231。

 如果 text_len=128，则总长 = 359。

 让我给出完整清晰的回答。

 送进 LLM 的数据 & 出来什么

0. 先记住几个基础数字

 ┌──────────┬──────────────────┬──────────────────────────┐
 │ 符号     │ 值               │ 含义                     │
 ├──────────┼──────────────────┼──────────────────────────┤
 │ B        │ batch_size       │ 一个 batch 里的样本数    │
 ├──────────┼──────────────────┼──────────────────────────┤
 │ D        │ 1024             │ Qwen3-0.6B 的隐藏维度    │
 ├──────────┼──────────────────┼──────────────────────────┤
 │ 视觉特征 │ 1536             │ DINO + SigLIP 拼接后维度 │
 ├──────────┼──────────────────┼──────────────────────────┤
 │ 历史帧   │ 31 帧            │ 每帧 4 个 coarse token   │
 ├──────────┼──────────────────┼──────────────────────────┤
 │ 当前帧   │ 64 个 fine token │ 8×8 grid                 │
 ├──────────┼──────────────────┼──────────────────────────┤
 │ K        │ 8                │ Top-K 候选人数           │
 └──────────┴──────────────────┴──────────────────────────┘

 ────────────────────────────────────────────────────────────────────────────────

1. 送入 LLM 之前：每个 agent 拼成一条"序列"

 在 _encode_agent_streams 里，先把所有东西投影到 1024 维，再拼成一条 token 序列：

```text
   [ text | reference | 历史coarse | 当前fine | 候选×8 | VERIFY | ACT ]                                                                                                         
```

 单 agent 各段的形状（1024 维）：

 ┌─────────────┬─────────────────┬───────────────────────────────────────────┐
 │ 段          │ 形状            │ 内容                                      │
 ├─────────────┼─────────────────┼───────────────────────────────────────────┤
 │ text        │ (1, ≤128, 1024) │ Qwen tokenizer 编码的文本指令             │
 ├─────────────┼─────────────────┼───────────────────────────────────────────┤
 │ reference   │ (1, 1, 1024)    │ episode 初始目标视觉 token（KIND_TARGET） │
 ├─────────────┼─────────────────┼───────────────────────────────────────────┤
 │ 历史 coarse │ (1, 155, 1024)  │ 31帧×4token + 31个时间token               │
 ├─────────────┼─────────────────┼───────────────────────────────────────────┤
 │ 当前 fine   │ (1, 65, 1024)   │ 64 token + 1个时间token                   │
 ├─────────────┼─────────────────┼───────────────────────────────────────────┤
 │ 候选        │ (1, 8, 1024)    │ Top-K 人物 token（KIND_DETECTION）        │
 ├─────────────┼─────────────────┼───────────────────────────────────────────┤
 │ VERIFY      │ (1, 1, 1024)    │ 可学习 query（KIND_VERIFY）               │
 ├─────────────┼─────────────────┼───────────────────────────────────────────┤
 │ ACT         │ (1, 1, 1024)    │ 可学习 query（KIND_ACT）                  │
 └─────────────┴─────────────────┴───────────────────────────────────────────┘

 单条序列长度 = text_len + 1 + 155 + 65 + 8 + 2 = text_len + 231

 （若 text 满 128，则总共 359 个 token）

 ────────────────────────────────────────────────────────────────────────────────

2. 各 token 的 1024 维向量是怎么来的

 ┌──────────────────┬───────────────────────────────────────────────────────────────────────────────────────┐
 │ token            │ 生成方式                                                                              │
 ├──────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
 │ text             │ tokenizer → input_ids → Qwen embedding 查表 → 1024                                    │
 ├──────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
 │ 视觉 coarse/fine │ 1536 维特征 → proj MLP → 1024                                                         │
 ├──────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
 │ reference        │ 1536 维 → proj → 1024 → KIND_TARGET 编码                                              │
 ├──────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
 │ 候选             │ 框内 fine token 均值池化 → candidate_roi_proj + detection_proj(几何) → KIND_DETECTION │
 ├──────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
 │ 时间 token       │ time_emb(帧id) + type_emb(种类)                                                       │
 ├──────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
 │ VERIFY / ACT     │ 可学习参数（初始 0）+ KIND_VERIFY/KIND_ACT + agent 编码                               │
 └──────────────────┴───────────────────────────────────────────────────────────────────────────────────────┘

 │ 所有 token 都被统一成 1024 维，所以能拼成一条序列喂给 Qwen。

 ────────────────────────────────────────────────────────────────────────────────

3. 两个 agent 怎么同时进 LLM

 不是把 drone 和 robotdog 拼在一条序列里，而是沿 batch 维拼接：

```text
   inputs_embeds: (2B, text_len + 231, 1024)                                                                                                                                    
                   ↑           ↑          ↑                                                                                                                                     
                 前 B 行 drone + 后 B 行 robotdog                                                                                                                               
```

```text
   行 0..B-1    = drone 的  B 条序列                                                                                                                                            
   行 B..2B-1   = robotdog 的 B 条序列                                                                                                                                          
```

 配合 attention_mask，每一行内部 causal attention，行与行之间完全隔离（互相看不到）。

 attention_mask: (2B, text_len + 231)，text 的 padding 位置为 0，其余为 1。

 ────────────────────────────────────────────────────────────────────────────────

4. LLM 内部做了什么

 Qwen 做一次 causal self-attention（参数冻结）：

```text
   inputs_embeds (2B, T, 1024)                                                                                                                                                  
           │                                                                                                                                                                    
           ▼  Qwen 若干层 attention + FFN                                                                                                                                       
   last_hidden_state (2B, T, 1024)                                                                                                                                              
```

 每个位置输出的 1024 维向量 = "看过了它前面所有 token 之后的上下文聚合"。

 因为 VERIFY 排在 ACT 前面：

- VERIFY 位置能看到：text、reference、历史、当前、候选（看不到 ACT）；
- ACT 位置能看到：text、reference、历史、当前、候选、VERIFY。

 ────────────────────────────────────────────────────────────────────────────────

5. LLM 出来什么（取出关键位置）

```text
   last_hidden_state (2B, T, 1024)                                                                                                                                              
           │                                                                                                                                                                    
           ├─ hidden[:, -2]  → VERIFY hidden  (2B, 1024)   ← 目标校验查询结果                                                                                                   
           ├─ hidden[:, -1]  → ACT hidden     (2B, 1024)   ← 动作查询结果                                                                                                       
           └─ 候选段位置      → candidate_contexts (B, 8, 1024)  ← 8 个候选的 hidden                                                                                            
```

 具体取值（_run_self_flows）：

```python
   verification_hidden = hidden[:, -2]      # VERIFY                                                                                                                            
   action_hidden       = hidden[:, -1]      # ACT                                                                                                                               
   candidate_contexts  = hidden[rows, start : start+8]   # 候选段                                                                                                               
```

 ────────────────────────────────────────────────────────────────────────────────

6. 出来后继续流向哪里

```text
   VERIFY hidden (2B,1024)                                                                                                                                                      
       │                                                                                                                                                                        
       ├─ candidate_matcher(候选hidden, VERIFYhidden) → 8 个候选分数                                                                                                            
       ├─ target_match_heads(VERIFYhidden)            → presence（是否存在目标）                                                                                                
       └─ 拼成 K+1 = 9 个 logits（8 候选 + NO_TARGET）                                                                                                                          
                                                                                                                                                                                
   候选 hidden (B,8,1024)                                                                                                                                                       
       │                                                                                                                                                                        
       └─ 按选中的候选加权求和 → selected_candidate_context (B,1024)                                                                                                            
                                                                                                                                                                                
   ACT hidden (B,1024)                                                                                                                                                          
       │                                                                                                                                                                        
       └─ + target_context_projs(selected_candidate_context)                                                                                                                    
            → grounded ACT (B,1024)                                                                                                                                             
            → self_planner → self_waypoints (B,8,3)                                                                                                                             
```

 ────────────────────────────────────────────────────────────────────────────────

7. 完整形状流转图（SELF 流）

```text
                       ┌─────────────────────────────────────────────────┐                                                                                                      
                       │  输入到 LLM（一次 SELF forward）                  │                                                                                                    
                       │  inputs_embeds: (2B, text_len+231, 1024)         │                                                                                                     
                       │  attention_mask: (2B, text_len+231)              │                                                                                                     
                       │                                                 │                                                                                                      
                       │  [text | ref | hist155 | curr65 | cand8 |        │                                                                                                     
                       │              VERIFY | ACT]                       │                                                                                                     
                       └──────────────────┬──────────────────────────────┘                                                                                                      
                                          ▼                                                                                                                                     
                             Qwen3-0.6B（冻结，causal）                                                                                                                         
                                          ▼                                                                                                                                     
                       ┌─────────────────────────────────────────────────┐                                                                                                      
                       │  last_hidden_state: (2B, text_len+231, 1024)     │                                                                                                     
                       └──────────────────┬──────────────────────────────┘                                                                                                      
                                          ▼                                                                                                                                     
           ┌──────────────┬──────────────┴──────────────┐                                                                                                                       
           ▼              ▼                             ▼                                                                                                                       
     VERIFY hidden    候选 hidden                    ACT hidden                                                                                                                 
     (2B,1024)      (B,8,1024)                     (B,1024)                                                                                                                     
           │              │                             │                                                                                                                       
           ▼              ▼                             ▼                                                                                                                       
     K+1 分类打分     selected context             + target 条件                                                                                                                
     (B,9)          (B,1024)                       → self_planner                                                                                                               
                                                       → waypoints (B,8,3)                                                                                                      
```

 ────────────────────────────────────────────────────────────────────────────────

 一句话总结

 │ 送进 LLM 的是：每条样本两行（drone/robotdog）各一条由「文本 + 初始reference + 31帧历史 + 当前帧 + 8个候选 + VERIFY/ACT 查询」拼成的 1024 维 token 序列（形状 (2B, ≤359,
 │ 1024)）。
 │ 出来的是同样长度的 last_hidden_state，然后只取最后两个位置（VERIFY、ACT）和候选段：VERIFY 用来做目标匹配，ACT 用来预测 waypoint。




 三个问题的回答

 问题 1：初始画面没目标，reference 是否有问题？后续会更新吗？

 需要分训练和推理两端看，它们行为不同。

### 训练端：reference 是"静态快照"，只取 step 0，不更新

 _initial_target_reference 的逻辑（关键代码）：

```python
   # 只读 episode 第一条 JSONL 记录（step 0）                                                                                                                                   
   first = json.loads(第一条记录)                                                                                                                                               
   for agent:                                                                                                                                                                   
       bbox = first 的 bbox                                                                                                                                                     
       bbox_valid = first 的 bbox_valid_mask                                                                                                                                    
       if 没有 bbox 或 bbox_valid == False:                                                                                                                                     
           continue           # reference 保持 全零 + valid=False                                                                                                               
       reference = fine_tokens[bbox区域].mean(0)   # 只从 step0 池化一次                                                                                                        
```

 我实际查了数据，目标不可见时：

```text
   target_visible=False  →  bbox_valid_mask=False  →  bbox=[0,0,0,0]                                                                                                            
```

 所以结论：

 ┌──────────────────┬───────────────────────────────────────────────────────────┐ │ 情况             │ 训练端行为                                                │ ├──────────────────┼───────────────────────────────────────────────────────────┤ │ step0 目标可见   │ 正常建立 reference，整个 episode 固定用这一个，不随帧更新 │ ├──────────────────┼───────────────────────────────────────────────────────────┤ │ step0 目标不可见 │ reference 无效（全零 + valid=False），不会向后找后续帧    │ └──────────────────┴───────────────────────────────────────────────────────────┘

- 不会有"错误的 reference"（不会拿背景当目标），因为 bbox 无效时直接跳过；
- 但会"缺失 reference"：step0 不可见 → 整个 episode 都没有 reference → 只能靠 instruction + 候选匹配兜底；
- 训练期不更新，这是设计选择：固定 anchor 防止逐帧漂移。

### 推理端：reference 是"动态记忆"，会更新

 推理时没有 GT bbox，reference 靠模型自己接受的候选建立：

```text
   目标不在画面 → 模型输出 NO_TARGET → 不建立 reference                                                                                                                         
   目标出现后   → 连续 3 帧高置信一致 → 确认建立                                                                                                                                
   确认后       → 冻结（不随每帧漂移）                                                                                                                                          
   连续 12 帧拒绝/NO_TARGET → 释放 → 回到 instruction grounding 重新获取
```

 所以推理端：初始没目标不会错锁（因为还没建立），等目标出现并连续确认后才建立，后续丢失会释放重建。

 │ 一句话：训练端 reference 是"一步到位、终身不变"的静态 GT 快照；推理端是"多帧确认、可失效重建"的动态记忆。

 ────────────────────────────────────────────────────────────────────────────────

 问题 2：text 128 token 会不会截断？

### 机制

 _embed_text 里：

```python
   tokenized = tokenizer(texts, padding=True, truncation=True, max_length=128)                                                                                                  
```

 truncation=True → 会截断，且默认从右边截断。

### 当前实际长度（我刚跑了统计）

 用当前（已缩短的）verification prompt，实测：

 ┌──────┬───────────────────────┬───────────┬────────┐
 │ 任务 │ SELF text             │ COOP text │ 超 128 │
 ├──────┼───────────────────────┼───────────┼────────┤
 │ STT  │ 91                    │ 25        │ 0%     │
 ├──────┼───────────────────────┼───────────┼────────┤
 │ DT   │ 108 ~ 115（均值 112） │ 42 ~ 49   │ 0%     │
 ├──────┼───────────────────────┼───────────┼────────┤
 │ AT   │ 92 ~ 97               │ 26 ~ 31   │ 0%     │
 └──────┴───────────────────────┴───────────┴────────┘

 当前配置下不会截断（DT 最大 115，还剩 13 token 余量）。

### 但存在两个风险点

1. DT 的 instruction 本身较长（如 Follow the short slim-built tied back woman wearing light gray long-sleeve shirt, blue jeans, black boots, neck scarf, utility belt...），如果
   未来出现更长的外观描述，可能逼近 128。
2. SELF text 的拼接顺序决定了截断的受害者：

```text
   "Tracking task: {角色指令} Target rule: {instruction} Grounding and action rule: {校验prompt}"                                                                               
                                                        ↑ 这个校验 prompt 在最后                                                                                                
```

 如果超长被截断，先被截掉的是最后的校验 prompt（Grounding and action rule），而不是目标 instruction。这会破坏"让模型学会选候选"的校验语义。

 （。）

 ────────────────────────────────────────────────────────────────────────────────

 问题 3：self 和 coop 的提示词怎么拼接、怎么区分？

 它们是两次独立的 Qwen forward，输入序列和指令都不同。

### SELF 流（_run_self_flows）

```text
   text = "Tracking task: {action_text} Target rule: {identity_text} Grounding and action rule: {verification_prompt}"                                                          
```

- action_text = 单个 agent 的角色指令（drone 或 dog 各自一条）
- identity_text = 原始 instruction
- verification_prompt = 目标校验规则

 序列（每行一个 agent，两行沿 batch 拼接，互相隔离）：

```text
   [text | ref | hist155 | curr65 | cand8 | VERIFY | ACT]                                                                                                                       
```

 形状 (2B, text_len+231, 1024)，前 B 行 drone、后 B 行 robotdog，行间 attention 隔离。

### COOP 流（_run_cooperative_flow）

```text
   joint_instructions = "{joint_text} Target identity description: {identity_text}"                                                                                             
```

- joint_text = "Use both aerial and ground observations to follow the target person..."
- identity_text = 原始 instruction

 序列（两个 agent 的 stream 拼在同一条序列里，共享 attention）：

```text
   [text | drone coop_stream | dog coop_stream | 选中候选×2 | pose×2 | relpose×2 | COOP_ACT×2]                                                                                  
```

 形状 (B, text_len+450, 1024)，只有 B 行，drone 和 dog 在同一行内互相能看到。

### 核心区别对照

 ┌──────────────┬─────────────────────────────────┬───────────────────────────┐
 │              │ SELF                            │ COOP                      │
 ├──────────────┼─────────────────────────────────┼───────────────────────────┤
 │ 行数         │ 2B 行（每 agent 一行）          │ B 行（两 agent 同一行）   │
 ├──────────────┼─────────────────────────────────┼───────────────────────────┤
 │ agent 间可见 │ ❌ 完全隔离                     │ ✅ 同序列互相可见         │
 ├──────────────┼─────────────────────────────────┼───────────────────────────┤
 │ 指令         │ 单 agent 角色指令 + 校验 prompt │ joint 协同指令 + 目标描述 │
 ├──────────────┼─────────────────────────────────┼───────────────────────────┤
 │ query        │ VERIFY + ACT 两个               │ 只有 COOP_ACT             │
 ├──────────────┼─────────────────────────────────┼───────────────────────────┤
 │ 视觉         │ clean（无遮挡）                 │ 可能被 mask（合成遮挡）   │
 ├──────────────┼─────────────────────────────────┼───────────────────────────┤
 │ 输出用途     │ 目标匹配 + 局部轨迹             │ 协同恢复轨迹              │
 ├──────────────┼─────────────────────────────────┼───────────────────────────┤
 │ 候选段       │ 8 个候选都进                    │ 只有"选中的 1 个"进       │
 └──────────────┴─────────────────────────────────┴───────────────────────────┘

### 为什么这样设计

```text
   SELF：每个 agent 独立判断"我看到的这几个人里谁是我要跟的"，并规划自己的路线。                                                                                                
                                                                                                                                                                                
   COOP：当某个 agent 看不见目标时，把两个 agent 的视觉、位姿、已选目标                                                                                                         
        放进同一条序列，让 Qwen 看到"队友看到的东西"，从而协同恢复路线。                                                                                                        
```

 区分方式不是靠文本标签，而是靠两条不同的序列结构和两次独立的 Qwen forward——SELF 里 agent 隔离、带校验 query；COOP 里 agent 共享、只带动作 query。



 LLM 输出之后的数据流（完整）

 前面讲到 SELF 的 Qwen 输出 last_hidden_state，取出 VERIFY / 候选 / ACT 三个 hidden。从这里开始，数据分三条线往下走，最后在"路由"汇合。

 阶段 1：SELF 输出 → 目标匹配（grounding）

```text
   VERIFY hidden (B,2,1024)        候选 hidden (B,2,8,1024)                                                                                                                     
           │                              │                                                                                                                                     
           ├──────── candidate_matcher ────┤   ← 显式匹配：VERIFY 查询 vs 8 个候选                                                                                              
           │         (B,2,8) 相对分数                                                                                                                                           
           │                                                                                                                                                                    
           └── target_match_heads(VERIFY) → presence_logits (B,2,1)  ← 旧头改当"是否存在目标"偏置                                                                               
                                                                                                                                                                                
   candidate_match_logits = 相对分数 + 0.5 × presence     (B,2,8)                                                                                                               
   null_logits            = -0.5 × presence               (B,2,1)                                                                                                               
                                                                                                                                                                                
   candidate_class_logits = [候选×8, NO_TARGET]           (B,2,9)                                                                                                               
```

 这 9 个 logits 就是"选谁，或者都不选"的完整答案。

 阶段 2：选中目标 → 条件化 ACT → self waypoint

```text
   class_probability = softmax(candidate_class_logits)     (B,2,9)                                                                                                              
                                                                                                                                                                                
   训练：hard 前向 / soft 反向（straight-through）                                                                                                                              
   推理：top-1 + 置信度/间隔/presence 门控                                                                                                                                      
                                                                                                                                                                                
   selected_candidate_context = Σ 选中权重 × 候选hidden + NULL权重 × null_context                                                                                               
                               (B,2,1024)                                                                                                                                       
                                                                                                                                                                                
   ACT hidden (B,2,1024)                                                                                                                                                        
        │                                                                                                                                                                       
        │  + target_context_projs(selected_candidate_context)   ← 零初始化残差                                                                                                  
        ▼                                                                                                                                                                       
   grounded_ACT (B,2,1024)                                                                                                                                                      
        │                                                                                                                                                                       
        ▼                                                                                                                                                                       
   self_planner（每 agent 一个 3 层 MLP）                                                                                                                                       
        │                                                                                                                                                                       
        ▼                                                                                                                                                                       
   self_waypoints (B,2,8,3)   ← 8 个 [x,y,yaw]                                                                                                                                  
```

 │ 关键：self planner 的输入 = ACT hidden + 选中目标特征，所以"选了谁"直接决定"怎么走"。

 阶段 3：COOP 流（第二次 Qwen forward）

 只有"某个 agent 可能看不见目标"时才有意义，但模型每次都会跑（训练时用合成遮挡制造这种场景）。

```text
   COOP 输入序列（B 行，两个 agent 拼在同一条序列里，互相可见）：                                                                                                               
                                                                                                                                                                                
   [ joint文本 | drone coop_stream(221) | dog coop_stream(221)                                                                                                                  
     | 选中候选drone(1) | 选中候选dog(1)                                                                                                                                        
     | pose_drone(1) | pose_dog(1)                                                                                                                                              
     | relpose_drone(1) | relpose_dog(1)                                                                                                                                        
     | COOP_ACT_drone(1) | COOP_ACT_dog(1) ]                                                                                                                                    
```

 总长 ≈ text_len + 450。经过 Qwen 后取出：

```text
   pose_hidden               (B,2,1024)   ← 双方位姿的上下文                                                                                                                    
   relative_pose_hidden      (B,2,1024)   ← 有向相对位姿                                                                                                                        
   selected_candidate_hidden (B,2,1024)   ← 选中目标在 COOP 里的表示                                                                                                            
   coop_contexts             (B,1024) ×2  ← 两个 COOP_ACT query                                                                                                                 
   coop_hidden[stream段]     ← 双方视觉的上下文                                                                                                                                 
```

 阶段 4：JEPA 恢复 + 目标信念

```text
   cooperative_base_memory = cat([                                                                                                                                              
       双方视觉上下文,                                                                                                                                                          
       selected_candidate_hidden,                                                                                                                                               
       pose_hidden,                                                                                                                                                             
       relative_pose_hidden                                                                                                                                                     
   ])                                             (B, ?, 1024)                                                                                                                  
                                                                                                                                                                                
   每个 agent：                                                                                                                                                                 
     jepa_memory = cat([base_memory, coop_contexts[agent]])                                                                                                                     
          │                                                                                                                                                                     
          ▼ ConditionalJEPAPredictor                                                                                                                                            
     prediction_tokens (B,64,1024)   ← 预测"被遮挡一方缺失的当前帧语义"                                                                                                         
          │                                                                                                                                                                     
          ├─ 池化 → target_belief_heads → target_belief (B,5)   ← 目标相对位置信念                                                                                              
          └─ 池化 → uncertainty_heads  → uncertainty  (B,1)     ← 信念不确定度                                                                                                  
```

 JEPA 的教师是 teacher_proj(fine_tokens)（干净视觉），损失只在被 mask 的 token 上算。

 阶段 5：4-mode 协同解码

```text
   decoder_memory = cat([cooperative_base_memory, prediction_tokens, obstacle_tokens])                                                                                          
                                                                                                                                                                                
   coop_decoder[agent](decoder_memory, coop_contexts[agent])                                                                                                                    
          │                                                                                                                                                                     
          ├─ candidates  (B,4,8,3)   ← 4 条候选恢复轨迹                                                                                                                         
          └─ mode_logits (B,4)       ← 每条轨迹的置信度                                                                                                                         
                                                                                                                                                                                
   cooperative_waypoints = 选 mode_logits 最高的那一条   (B,2,8,3)                                                                                                              
```

 阶段 6：路由（决定最终用哪条轨迹）

```text
   observed_visible = 目标可靠可见？   ← 由 Top-K + 匹配 + NO_TARGET 决定                                                                                                       
                                                                                                                                                                                
   needs_assistance = ~observed_visible                                                                                                                                         
   route_to_cooperative = 自己看不见 & 队友看得见                                                                                                                               
   route_to_belief       = 双方都看不见                                                                                                                                         
                                                                                                                                                                                
   routed_waypoints = 自己看得见 ? self_waypoints                                                                                                                               
                     : 队友看得见 ? cooperative_waypoints                                                                                                                       
                     : 双方都看不见 ? cooperative_waypoints(信念模式)                                                                                                           
                     (B,2,8,3)                                                                                                                                                  
```

 阶段 7：损失（训练时）

 所有输出汇到 forward_airground_v3_loss：

```text
   loss = β_nav        × (self waypoint loss + coop waypoint loss)   ← 主轨迹                                                                                                   
        + β_coop       × cooperative waypoint loss（best candidate）                                                                                                            
        + β_mode       × mode 分类 loss                                                                                                                                         
        + β_jepa       × JEPA 恢复 loss（masked token vs clean teacher）                                                                                                        
        + β_belief     × target_belief 回归 loss                                                                                                                                
        + β_target     × K+1 候选分类 loss（含 NO_TARGET）                                                                                                                      
        + β_uncertainty× 不确定性 NLL                                                                                                                                           
        + β_smoothness/kinematics/diversity                                                                                                                                     
```

 ────────────────────────────────────────────────────────────────────────────────

 完整数据流总图

```text
                            ┌─────────────────────────────┐                                                                                                                     
                            │  SELF Qwen forward（1 次）    │                                                                                                                   
                            │  [text|ref|hist|curr|cand|    │                                                                                                                   
                            │        VERIFY|ACT]            │                                                                                                                   
                            └───────┬──────────┬────────────┘                                                                                                                   
                                    │          │                                                                                                                                
                       VERIFY/候选   │          │  ACT                                                                                                                          
                                    ▼          ▼                                                                                                                                
                       候选匹配(K+1)      grounded_ACT                                                                                                                          
                       选中目标context        │                                                                                                                                 
                            │                ▼                                                                                                                                  
                            │           self_waypoints (B,2,8,3)                                                                                                                
                            │                                                                                                                                                   
                            ▼                                                                                                                                                   
                 ┌──────────────────────────────────┐                                                                                                                           
                 │  COOP Qwen forward（第 2 次）      │                                                                                                                         
                 │  [joint|双方视觉|选中目标|位姿|     │                                                                                                                        
                 │       相对位姿|COOP_ACT]          │                                                                                                                          
                 └───────┬──────────┬───────────────┘                                                                                                                           
                         │          │                                                                                                                                           
                 base_memory      coop_contexts                                                                                                                                 
                         │          │                                                                                                                                           
             ┌───────────┴──────────▼────────────┐                                                                                                                              
             │  JEPA 恢复 → target_belief/uncert │                                                                                                                              
             │  4-mode decoder → coop_waypoints  │                                                                                                                              
             └───────────────┬──────────────────┘                                                                                                                               
                             │                                                                                                                                                  
                             ▼                                                                                                                                                  
                 路由: self / coop / belief                                                                                                                                     
                             │                                                                                                                                                  
                             ▼                                                                                                                                                  
                    routed_waypoints (B,2,8,3)                                                                                                                                  
                             │                                                                                                                                                  
                             ▼                                                                                                                                                  
                       loss 汇总（训练）
```

 ────────────────────────────────────────────────────────────────────────────────

 关键形状速查

 ┌────────────────────────────┬─────────────┬─────────────────────────────┐
 │ 中间产物                   │ 形状        │ 含义                        │
 ├────────────────────────────┼─────────────┼─────────────────────────────┤
 │ candidate_class_logits     │ (B,2,9)     │ 8 候选 + NO_TARGET          │
 ├────────────────────────────┼─────────────┼─────────────────────────────┤
 │ selected_candidate_context │ (B,2,1024)  │ 选中目标表征                │
 ├────────────────────────────┼─────────────┼─────────────────────────────┤
 │ self_waypoints             │ (B,2,8,3)   │ 各 agent 局部轨迹           │
 ├────────────────────────────┼─────────────┼─────────────────────────────┤
 │ cooperative_candidates     │ (B,2,4,8,3) │ 每个 agent 4 条候选恢复轨迹 │
 ├────────────────────────────┼─────────────┼─────────────────────────────┤
 │ cooperative_waypoints      │ (B,2,8,3)   │ 选中的协同轨迹              │
 ├────────────────────────────┼─────────────┼─────────────────────────────┤
 │ target_belief              │ (B,2,5)     │ 目标相对位置信念            │
 ├────────────────────────────┼─────────────┼─────────────────────────────┤
 │ routed_waypoints           │ (B,2,8,3)   │ 最终输出轨迹                │
 └────────────────────────────┴─────────────┴─────────────────────────────┘

 一句话总结：SELF 出"目标是谁 + 自己怎么走"，COOP 出"看不见时协同怎么恢复"，最后按可见性路由二选一，训练时所有分支都算损失。



 三个问题的回答

 问题 1：候选与 VERIFY 是怎么"显式匹配"的？每个候选都有 hidden_state 吗？

 是的，每个候选都有一个 hidden_state。

 在 SELF 的 Qwen forward 里，8 个候选 token 各占一个序列位置，经过 Qwen 后每个位置输出一个 1024 维 hidden：

```text
   candidate_context  (B,2,8,1024)   ← 8 个候选，各一个 1024 维 hidden                                                                                                          
   verify_context     (B,2,1024)     ← VERIFY query 位置输出一个 1024 维 hidden                                                                                                 
```

 "显式匹配"就是 candidate_matcher 把 VERIFY 这一个向量 和 每一个候选向量 做交互：

```python
   c = normalize(候选)          # 8 个候选各自归一化                                                                                                                            
   t = normalize(VERIFY).expand # VERIFY 复制 8 份，和每个候选对齐                                                                                                              
                                                                                                                                                                                
   fused = [c, t, c*t, c-t]     # 每个候选拼接成 4×1024 = 4096 维                                                                                                               
           │                                                                                                                                                                    
           ▼ MLP → 每个候选一个标量分数                                                                                                                                         
           + cosine(c,t)/temperature   # 再加余弦相似度                                                                                                                         
```

```text
   VERIFY (1024) ─┬─ 候选1 (1024) → 拼接+交互 → 分数1                                                                                                                           
                  ├─ 候选2 (1024) → 拼接+交互 → 分数2                                                                                                                           
                  ├─ ...                                                                                                                                                        
                  └─ 候选8 (1024) → 拼接+交互 → 分数8                                                                                                                           
```

 对比"独立打分"：旧做法是每个候选自己过一个 Linear(1024→1)，候选之间、候选与 VERIFY 之间没有交互。现在改成 VERIFY 向量与每个候选向量配对交互（拼接 + 点乘 + 差 + cosine），所以是
 "显式匹配"。

 │ 关键：VERIFY 和候选的 hidden 都来自同一次 Qwen causal attention——VERIFY 位置已经"看过"所有候选和文本，候选位置也"看过"文本和视觉，所以这两个 hidden 都是上下文聚合后的产物。

 ────────────────────────────────────────────────────────────────────────────────

 问题 2：都不选（NO_TARGET）后续怎么处理？走协同还是搜索？

 分三层回答：

### 第 1 层：模型内部（选 NULL）

```text
   NULL 概率最高 → selected_accepted = False                                                                                                                                    
                → selected_candidate_context = NULL context（不是任何真人）                                                                                                     
                → self_waypoints 仍然输出，但路由不会采用                                                                                                                       
```

### 第 2 层：模型内路由（forward 里的 routed_waypoints）

```text
   observed_visible = YOLO有效 & 匹配概率≥阈值                                                                                                                                  
                     = False（因为选了 NULL）                                                                                                                                   
                                                                                                                                                                                
   needs_assistance = True（自己"看不见目标"）                                                                                                                                  
                                                                                                                                                                                
   路由结果：                                                                                                                                                                   
     队友可见     → cooperative_waypoints（协同）                                                                                                                               
     双方都不可见 → cooperative_waypoints（信念/JEPA 恢复）                                                                                                                     
```

### 第 3 层：运行层（eval runtime 才有"搜索"）

 模型本身没有 SEARCH 输出。搜索是 runtime 层面的：

```text
   双方都不可见 → BELIEF hold（3 帧，用上一帧导航轨迹/coop 轨迹）                                                                                                               
                → 仍然不可见 → SEARCH（原地左右转圈找目标）                                                                                                                     
```

```text
           ┌─────────────────────────────────────────┐                                                                                                                          
           │ 模型输出只有 3 种轨迹：                   │                                                                                                                        
           │   self / cooperative / (belief用coop)    │                                                                                                                         
           └─────────────────────────────────────────┘                                                                                                                          
                             │                                                                                                                                                  
                             ▼                                                                                                                                                  
           ┌─────────────────────────────────────────┐                                                                                                                          
           │ runtime 路由（带滞回）：                  │                                                                                                                        
           │  可见 → SELF                             │                                                                                                                         
           │  单方不可见 → COOPERATIVE                │                                                                                                                         
           │  双方不可见(短) → BELIEF（hold 3帧）     │                                                                                                                         
           │  双方不可见(长) → SEARCH（转圈找目标）   │                                                                                                                         
           └─────────────────────────────────────────┘                                                                                                                          
```

 所以："都不选"时，模型层面走协同（队友可见）或信念（双方不可见）；"搜索"是 runtime 在双方长时间丢失后才触发的，不是模型输出。

 ────────────────────────────────────────────────────────────────────────────────

 问题 3：COOP 是第二次过大模型，串行还是并行？

 你的理解大部分对，但有一个关键偏差。

### 偏差：不是"先判断要不要 coop，需要才走 coop"

 实际代码里，SELF 和 COOP 无条件都跑，没有 if needs_coop 分支：

```python
   # 1. 先 SELF（第一次 Qwen）                                                                                                                                                  
   self_contexts, verify, candidate = self._run_self_flows(...)                                                                                                                 
                                                                                                                                                                                
   # 2. 候选匹配（用 SELF 输出）                                                                                                                                                
                                                                                                                                                                                
   # 3. 无条件 COOP（第二次 Qwen）← 每次都跑，不判断                                                                                                                            
   coop_hidden, ... = self._run_cooperative_flow(...)                                                                                                                           
                                                                                                                                                                                
   # 4. 路由时才"决定用哪个输出"                                                                                                                                                
   routed = where(needs_assistance, coop_waypoints, self_waypoints)                                                                                                             
```

### 所以正确的理解是：

 ┌───────────┬────────────────────────────┬────────────────────────────────────┐
 │           │ 你的理解                   │ 实际                               │
 ├───────────┼────────────────────────────┼────────────────────────────────────┤
 │ 调用顺序  │ 先 self，判断，需要再 coop │ 先 self，无条件 coop，最后路由才选 │
 ├───────────┼────────────────────────────┼────────────────────────────────────┤
 │ 训练时    │ 都会走？                   │ ✅ 每次都串行跑两次                │
 ├───────────┼────────────────────────────┼────────────────────────────────────┤
 │ 推理时    │ 需要才走 coop？            │ ❌ 也每次都跑两次                  │
 ├───────────┼────────────────────────────┼────────────────────────────────────┤
 │ 并行/串行 │ 串行？                     │ ✅ 串行（且必须串行）              │
 └───────────┴────────────────────────────┴────────────────────────────────────┘

### 为什么必须串行（不能并行）

 COOP 的输入依赖 SELF 的选中目标 context：

```text
   SELF → 候选匹配 → selected_candidate_context（选中了谁）                                                                                                                     
                           │                                                                                                                                                    
                           ▼                                                                                                                                                    
                   COOP 流把这个"选中目标"拼进序列                                                                                                                              
```

 COOP 需要知道"我选中了谁"，才能协同恢复"这个目标"的轨迹。所以 COOP 天然依赖 SELF 的结果，无法并行。

### "判断是否要 coop"发生的位置

 判断发生在最后的路由，而且判断的是"用哪个输出"，不是"要不要调用 coop 模型"：

```text
   SELF forward ──→ self_waypoints ──┐                                                                                                                                          
                                      ├─ 路由二选一 → 最终 waypoint                                                                                                             
   COOP forward ──→ coop_waypoints ──┘                                                                                                                                          
           （两次 forward 都算完了，才选）                                                                                                                                      
```

 ────────────────────────────────────────────────────────────────────────────────

 一句话总结

1. 显式匹配：每个候选有一个 hidden，VERIFY 有一个 hidden，两者配对做"拼接+点乘+差+cosine"交互打分。
2. 都不选：模型输出 NULL → 路由到协同（队友可见）或信念（双方不可见）；搜索是 runtime 在长时间丢失后的兜底，不是模型输出。
3. COOP 串行：是的，但注意是无条件每次串行跑两次 forward，COOP 依赖 SELF 的选中目标所以不能并行；"要不要 coop"是最后路由选输出，不是决定"要不要调用 coop 模型"。



 损失是怎么算的（完整数据流）

0. 先看监督信号（batch 里的 GT 标签）

 ┌───────────────────────┬───────────┬───────────────────────────────────┐
 │ 标签                  │ 形状      │ 含义                              │
 ├───────────────────────┼───────────┼───────────────────────────────────┤
 │ waypoints             │ (B,2,8,3) │ 干净局部轨迹（self 的 GT）        │
 ├───────────────────────┼───────────┼───────────────────────────────────┤
 │ valid_mask            │ (B,2,8)   │ 哪些 waypoint 有效                │
 ├───────────────────────┼───────────┼───────────────────────────────────┤
 │ cooperative_waypoints │ (B,2,8,3) │ 协同恢复轨迹（遮挡+位姿扰动生成） │
 ├───────────────────────┼───────────┼───────────────────────────────────┤
 │ self_target           │ (B,2)     │ 谁参与 self 损失                  │
 ├───────────────────────┼───────────┼───────────────────────────────────┤
 │ cooperative_target    │ (B,2)     │ 谁参与 coop 损失                  │
 ├───────────────────────┼───────────┼───────────────────────────────────┤
 │ candidate_iou         │ (B,2,8)   │ 每个候选与 GT 框的 IoU            │
 ├───────────────────────┼───────────┼───────────────────────────────────┤
 │ target_pose / valid   │ (B,2,5)   │ 目标相对位置（belief 标签）       │
 ├───────────────────────┼───────────┼───────────────────────────────────┤
 │ synthetic_occlusion   │ (B,2)     │ 谁被合成遮挡                      │
 └───────────────────────┴───────────┴───────────────────────────────────┘

 ────────────────────────────────────────────────────────────────────────────────

1. 主轨迹损失：waypoint loss

 核心函数 weighted_multi_agent_waypoint_loss，对每条轨迹算三部分误差：

```text
   ① xy 误差：smooth_l1(pred_xy, gt_xy)，dog 侧向位移额外加权                                                                                                                   
   ② yaw 误差：smooth_l1(pred_yaw, gt_yaw) × yaw_weight                                                                                                                         
   ③ final 误差：最后一个有效 waypoint 额外加权 × final_weight                                                                                                                  
                                                                                                                                                                                
   再加"行为加权" behavior_weight：                                                                                                                                             
     转弯样本（turn）→ 加权 turn_sample_weight                                                                                                                                  
     停止样本（stop）→ 加权 stop_sample_weight                                                                                                                                  
                                                                                                                                                                                
   per_agent = (轨迹误差 + final_weight × final误差) × behavior_weight                                                                                                          
```

### self 损失（loss_self）

```python
   self_per_agent = waypoint_loss(self_waypoints, gt, valid_mask)   # (B,2)                                                                                                     
   loss_self = 只对 self_target 的 agent 求平均                                                                                                                                 
```

 │ 关键：self 行永远干净（synthetic 遮挡只影响 coop 流），所以 self_waypoints 用原始 GT 监督。

### cooperative 损失（loss_cooperative）

 coop decoder 输出 4 条候选轨迹，监督策略是"取 4 条里最好的那个算 loss"（不是平均）：

```python
   candidate_losses = 对 4 条候选分别算 waypoint loss   # (B,2,4)                                                                                                               
   best_loss, best_index = candidate_losses.min(dim=-1)   # 取最小（最好）                                                                                                      
   loss_cooperative = 只对 cooperative_target 的 agent 平均                                                                                                                     
```

```text
           4 条候选轨迹                                                                                                                                                         
           ┌──────────────┐                                                                                                                                                     
           │ 轨迹0  loss=5 │                                                                                                                                                    
           │ 轨迹1  loss=2 │ ← min，只用这个监督                                                                                                                                
           │ 轨迹2  loss=8 │                                                                                                                                                    
           │ 轨迹3  loss=3 │                                                                                                                                                    
           └──────────────┘                                                                                                                                                     
           只惩罚"最好那条"还差多远（min 策略，保证至少有一条是对的）                                                                                                           
```

### mode 损失（loss_mode）

```python
   loss_mode = CE(mode_logits, best_index)   # 用上面最好的那条作为伪标签                                                                                                       
```

 训练 mode head 学会"识别哪条候选最好"。

 ────────────────────────────────────────────────────────────────────────────────

2. 目标匹配损失（K+1 分类，loss_target_match）

```text
   candidate_class_logits (B,2,9) = [8 个候选分数, NO_TARGET]                                                                                                                   
```

 标签怎么造：

```python
   masked_iou = candidate_iou（无效候选置 -1）                                                                                                                                  
   best_iou, best_index = masked_iou.max(-1)                                                                                                                                    
                                                                                                                                                                                
   has_target = 目标可见 & bbox有效 & best_iou >= 0.3                                                                                                                           
   class_label = has_target ? best_index : NULL  # 有目标→最大IoU候选，无→NO_TARGET                                                                                             
                                                                                                                                                                                
   loss = CE(class_logits, class_label)                                                                                                                                         
```

```text
   情况1：目标在候选3（IoU 最大）→ 标签 = 3                                                                                                                                     
   情况2：8 个候选都不是目标      → 标签 = NO_TARGET（第 9 类）                                                                                                                 
```

 │ 这统一了"选谁"和"拒绝"两个问题：正类是最大 IoU 候选，全负时监督 NO_TARGET，避免全负帧被迫选一个干扰者。

 有效 mask：

```python
   class_valid = ((~visible) | bbox_valid) & perception_cache_valid & supervised_agent                                                                                          
```

 ────────────────────────────────────────────────────────────────────────────────

3. JEPA 损失（loss_jepa）

 JEPA 预测"被遮挡一方缺失的当前帧语义"，教师是干净视觉：

```python
   token_cosine = 1 - cosine_similarity(jepa_prediction, jepa_teacher)   # (B,2,64)                                                                                             
   loss_jepa = 只在 masked token & jepa_valid 上平均                                                                                                                            
```

```text
   被遮挡的 receiver：                                                                                                                                                          
     预测 token ← 从 coop 上下文（队友视觉+位姿+选中目标）恢复                                                                                                                  
     教师 token ← teacher_proj(干净 fine_tokens)  ← 无遮挡的真相                                                                                                                
     损失 = 两者余弦距离，只算被 mask 的 token                                                                                                                                  
```

 ────────────────────────────────────────────────────────────────────────────────

4. 目标信念损失（loss_belief + loss_uncertainty）

```python
   belief_per_agent = smooth_l1(target_belief, target_pose)   # 目标相对位置 (B,2,5)                                                                                            
   loss_belief = 只对 cooperative_target 平均                                                                                                                                   
                                                                                                                                                                                
   # 不确定性 NLL（自动加权 belief 误差）                                                                                                                                       
   loss_uncertainty = exp(-log_var) * belief.detach() + log_var                                                                                                                 
```

 ────────────────────────────────────────────────────────────────────────────────

5. 正则损失

```python
   loss_smoothness = 二阶差分（轨迹平滑，抖动惩罚）                                                                                                                             
   loss_kinematics = 超速 + 超 yaw 率 + dog 非完整约束（侧滑惩罚）                                                                                                              
   loss_diversity  = 4 条候选终点不要太近（避免模式塌缩）                                                                                                                       
   loss_obstacle   = 0（未启用，安全投影缺失时禁止）                                                                                                                            
```

 都只对 cooperative_target 的样本算。

 ────────────────────────────────────────────────────────────────────────────────

6. 总损失加权

```python
   loss = β_nav       × loss_self              # 100                                                                                                                            
        + β_coop      × loss_cooperative       # 100                                                                                                                            
        + β_mode      × loss_mode              # 1                                                                                                                              
        + β_jepa      × loss_jepa              # 1                                                                                                                              
        + β_belief    × loss_belief            # 1                                                                                                                              
        + β_target    × loss_target_match      # 1                                                                                                                              
        + β_uncertainty × loss_uncertainty     # 0.1                                                                                                                            
        + β_smoothness × loss_smoothness       # 0.1                                                                                                                            
        + β_kinematics × loss_kinematics       # 0.1                                                                                                                            
        + β_diversity  × loss_diversity        # 0.1                                                                                                                            
        + β_obstacle   × loss_obstacle         # 0（禁用）                                                                                                                      
```

 ────────────────────────────────────────────────────────────────────────────────

7. 完整损失数据流图

```text
   model 输出                          GT 标签                                                                                                                                  
   ─────────                          ────────                                                                                                                                  
   self_waypoints (B,2,8,3) ──┐        waypoints + valid_mask                                                                                                                   
                               ├─→ loss_self（self_target 掩码）                                                                                                                
   cooperative_candidates ─────┤                                                                                                                                                
     (B,2,4,8,3) ──────────────┤        cooperative_waypoints                                                                                                                   
                               ├─→ loss_cooperative（取 4 条最好，coop_target 掩码）                                                                                            
   cooperative_mode_logits ────┘                                                                                                                                                
     (B,2,4) ───────────────────→ loss_mode（伪标签 = best_index）                                                                                                              
                                                                                                                                                                                
   jepa_prediction (B,2,64,1024)┐                                                                                                                                               
   jepa_teacher    (B,2,64,1024)┴→ loss_jepa（只 masked token）                                                                                                                 
                                                                                                                                                                                
   target_belief (B,2,5) ───────→ loss_belief（smooth_l1 vs target_pose）                                                                                                       
   uncertainty   (B,2,1) ───────→ loss_uncertainty（NLL）                                                                                                                       
                                                                                                                                                                                
   candidate_class_logits (B,2,9)┐                                                                                                                                              
   candidate_iou (B,2,8) ─────────┴→ loss_target_match（K+1 分类）                                                                                                              
                                                                                                                                                                                
   cooperative_candidates ───────→ loss_smoothness / kinematics / diversity                                                                                                     
```

 ────────────────────────────────────────────────────────────────────────────────

8. 一个关键设计：mask 如何分离 self 和 coop

```text
                   self_target    cooperative_target                                                                                                                            
   干净 agent      ✅ 有 self 监督   ❌ 无 coop 监督                                                                                                                            
   被遮挡 agent    ✅ 仍有 self 监督  ✅ 有 coop 监督                                                                                                                           
```

- self 行永远有监督（遮挡只作用 coop 流，不破坏 self）
- coop 只监督被遮挡的 receiver（正常 agent 不需要协同恢复）

 这样每个损失都有明确的学习目标：

```text
   self_waypoints       → 学"自己看得见时怎么跟"                                                                                                                                
   cooperative_candidates → 学"看不见时怎么靠队友恢复"                                                                                                                          
   mode_logits          → 学"哪条恢复轨迹最好"                                                                                                                                  
   candidate_class      → 学"谁是目标 / 有没有目标"                                                                                                                             
   jepa                 → 学"补全缺失的视觉"                                                                                                                                    
   target_belief        → 学"目标大概在哪个相对位置"                                                                                                                            
```

 ────────────────────────────────────────────────────────────────────────────────

 一句话总结

 │ 损失分六路：主轨迹（self 用 GT、coop 取 4 候选最好）、mode（学选最好的）、K+1 候选分类（最大 IoU 为正类，全负监督 NO_TARGET）、JEPA（补全遮挡视觉）、belief+uncertainty（目标
 │ 相对位置）、轨迹正则（平滑/运动学/多样性）。每路用不同的 mask 精确指定"谁、在哪、监督什么"，最后按 β 加权求和。




 之前已经讲了「数据 → 前向 → 损失」的主线，但还有几个环节没覆盖：采样（Sampler）→ collate 细节 → 反向传播/优化器/保存。补齐后，下面做端到端完整汇总。

 补充遗漏的环节

### 环节 1：Sampler（决定每个 epoch 读哪些帧）

 RotatingTemporalStrideDistributedSampler 做三件事：

```text
   ① 时间下采样：stride=5 → 每 epoch 只取 1/5 的 current rows，5 个 epoch 轮转覆盖全部时间相位                                                                                                     
      且每个 episode 用独立随机 offset（避免所有 episode 同相位）                                                                                                               
                                                                                                                                                                                
   ② block shuffle：block_size=8 → 把 8 个相邻样本打包，再全局 shuffle 块                                                                                                       
      目的：保持视觉缓存局部性 + 自然混合 STT/DT/AT                                                                                                                             
                                                                                                                                                                                
   ③ DDP 分片：总样本 pad 成 N 的倍数，每个 rank 拿连续的 1/N                                                                                                                   
```

### 环节 2：collate（组装 batch）

 collate_airground_v3_batch：把 B 个 sample 的每个字段 torch.stack：

```text
   coarse_tokens  → (B,2,124,1536)                                                                                                                                              
   fine_tokens    → (B,2,64,1536)                                                                                                                                               
   candidate_feat → (B,2,8,6)                                                                                                                                                   
   waypoints      → (B,2,8,3)                                                                                                                                                   
   ... 共 30+ 字段                                                                                                                                                              
   instruction 等文本 → 保留 list（给 tokenizer）                                                                                                                               
```

### 环节 3：反向传播 + 优化（训练循环）

```text
   loss → loss/grad_accum → backward（梯度累积）                                                                                                                                
        → 梯度裁剪 clip_grad_norm(1.0)                                                                                                                                          
        → optimizer.step + scheduler.step + zero_grad                                                                                                                           
        → 混合精度 AMP（bfloat16 autocast + scaler）                                                                                                                            
```

### 环节 4：日志 + 保存

```text
   每 1 步：CSV + TensorBoard + 终端                                                                                                                                            
   每 2000 步：checkpoint                                                                                                                                                       
   每 epoch：final checkpoint                                                                                                                                                   
   （val_json=null 时不验证）                                                                                                                                                   
```

 ────────────────────────────────────────────────────────────────────────────────

 端到端完整数据流汇总

```text
   ┌─────────────────────────────────────────────────────────────────────────┐                                                                                                  
   │  阶段 0：离线预处理（训练前，只做一次，不在训练循环里）                    │                                                                                               
   │                                                                         │                                                                                                  
   │  原始 jpg ──→ DINO+SigLIP ──→ vision_cache/vcoarse(4) + vfine(64)      │                                                                                                   
   │  原始 jpg ──→ YOLO ────────→ perception_cache/.npz + .candidates.npz   │                                                                                                   
   │  仿真轨迹 ──────────────────→ JSONL 标注（instruction/waypoints/bbox）  │                                                                                                  
   └─────────────────────────────────────────────────────────────────────────┘                                                                                                  
                                       │                                                                                                                                        
                                       ▼                                                                                                                                        
   ┌─────────────────────────────────────────────────────────────────────────┐                                                                                                  
   │  阶段 1：Sampler（选帧）                                                 │                                                                                                 
   │  stride=5 下采样 + 每 episode 独立 offset + block=8 shuffle + DDP 分片   │                                                                                                 
   └─────────────────────────────────────────────────────────────────────────┘                                                                                                  
                                       │                                                                                                                                        
                                       ▼                                                                                                                                        
   ┌─────────────────────────────────────────────────────────────────────────┐                                                                                                  
   │  阶段 2：Dataset.__getitem__（读一个样本）                               │                                                                                                 
   │  JSONL 一行 → 读 vision_cache + perception_cache + GT 标签              │                                                                                                  
   │  → 数据增强（合成遮挡 + 位姿扰动 + reference dropout）                   │                                                                                                 
   │  → sample dict                                                          │                                                                                                  
   └─────────────────────────────────────────────────────────────────────────┘                                                                                                  
                                       │                                                                                                                                        
                                       ▼                                                                                                                                        
   ┌─────────────────────────────────────────────────────────────────────────┐                                                                                                  
   │  阶段 3：collate（组装 batch）                                           │                                                                                                 
   │  B 个 sample → torch.stack → batch dict（30+ 字段）                     │                                                                                                  
   └─────────────────────────────────────────────────────────────────────────┘                                                                                                  
                                       │                                                                                                                                        
                                       ▼                                                                                                                                        
   ┌─────────────────────────────────────────────────────────────────────────┐                                                                                                  
   │  阶段 4：model forward（两次 Qwen 前向）                                 │                                                                                                 
   │                                                                         │                                                                                                  
   │  SELF（第 1 次 Qwen）：                                                  │                                                                                                 
   │   [text|ref|hist|curr|cand8|VERIFY|ACT]                                │                                                                                                   
   │     → 候选匹配 K+1 → 选中目标 → grounded ACT → self_waypoints          │                                                                                                   
   │                                                                         │                                                                                                  
   │  COOP（第 2 次 Qwen，依赖 SELF 的选中目标）：                            │                                                                                                 
   │   [joint|双方视觉|选中目标|位姿|相对位姿|COOP_ACT]                       │                                                                                                 
   │     → JEPA 恢复 → target_belief → 4-mode decoder → cooperative_waypoints│                                                                                                  
   │                                                                         │                                                                                                  
   │  路由：self / coop / belief → routed_waypoints                          │                                                                                                  
   └─────────────────────────────────────────────────────────────────────────┘                                                                                                  
                                       │                                                                                                                                        
                                       ▼                                                                                                                                        
   ┌─────────────────────────────────────────────────────────────────────────┐                                                                                                  
   │  阶段 5：损失计算                                                        │                                                                                                 
   │  loss = β·loss_self + β·loss_coop + β·loss_mode                        │                                                                                                   
   │       + β·loss_jepa + β·loss_belief + β·loss_target_match              │                                                                                                   
   │       + β·loss_uncertainty + β·(smoothness+kinematics+diversity)       │                                                                                                   
   └─────────────────────────────────────────────────────────────────────────┘                                                                                                  
                                       │                                                                                                                                        
                                       ▼                                                                                                                                        
   ┌─────────────────────────────────────────────────────────────────────────┐                                                                                                  
   │  阶段 6：反向传播 + 优化                                                 │                                                                                                 
   │  backward（梯度累积）→ clip_grad_norm → optimizer.step → scheduler.step │                                                                                                  
   └─────────────────────────────────────────────────────────────────────────┘                                                                                                  
                                       │                                                                                                                                        
                                       ▼                                                                                                                                        
   ┌─────────────────────────────────────────────────────────────────────────┐                                                                                                  
   │  阶段 7：日志 + 保存                                                     │                                                                                                 
   │  CSV/TensorBoard/终端 + 每 2000 步 checkpoint + 每 epoch final           │                                                                                                 
   └─────────────────────────────────────────────────────────────────────────┘                                                                                                  
                                       │                                                                                                                                        
                                       ▼                                                                                                                                        
                             下一 batch（循环）
```

 ────────────────────────────────────────────────────────────────────────────────

 一张"从磁盘到梯度"的总图

```text
   磁盘（预处理产物）                                                                                                                                                           
     ├─ JSONL 标注 ────────────────┐                                                                                                                                            
     ├─ vision_cache(vcoarse/vfine)┤                                                                                                                                            
     └─ perception_cache(npz)      │                                                                                                                                            
                                   ▼                                                                                                                                            
                            Sampler 选帧                                                                                                                                        
                                   │                                                                                                                                            
                                   ▼                                                                                                                                            
                            Dataset 读样本                                                                                                                                      
                       （读缓存 + 数据增强）                                                                                                                                    
                                   │                                                                                                                                            
                                   ▼                                                                                                                                            
                            collate 组 batch                                                                                                                                    
                                   │                                                                                                                                            
                                   ▼                                                                                                                                            
                        ┌──────────┴──────────┐                                                                                                                                 
                        ▼                      ▼                                                                                                                                
                  SELF Qwen 前向          （第1次）                                                                                                                             
                  候选匹配→选中目标          │                                                                                                                                  
                        │                    │                                                                                                                                  
                        └───── 选中目标 ──────┘                                                                                                                                 
                                 │                                                                                                                                              
                                 ▼                                                                                                                                              
                           COOP Qwen 前向    （第2次）                                                                                                                          
                           JEPA+belief+4mode                                                                                                                                    
                                 │                                                                                                                                              
                                 ▼                                                                                                                                              
                        路由 → routed_waypoints                                                                                                                                 
                                 │                                                                                                                                              
                                 ▼                                                                                                                                              
                       损失计算（6 路加权）                                                                                                                                     
                                 │                                                                                                                                              
                                 ▼                                                                                                                                              
                       backward → 梯度裁剪 → 优化器                                                                                                                             
                                 │                                                                                                                                              
                                 ▼                                                                                                                                              
                       日志 + checkpoint 保存
```

 ────────────────────────────────────────────────────────────────────────────────

 完整数据流中每个张量的"形状生命线"

 ┌───────────┬───────────────────────┬─────────────────────────┐
 │ 阶段      │ 关键张量              │ 形状                    │
 ├───────────┼───────────────────────┼─────────────────────────┤
 │ 磁盘      │ vcoarse / vfine       │ [4,1536] / [64,1536]    │
 ├───────────┼───────────────────────┼─────────────────────────┤
 │ sample    │ coarse_tokens         │ [2,124,1536]            │
 ├───────────┼───────────────────────┼─────────────────────────┤
 │ sample    │ candidate_feat        │ [2,8,6]                 │
 ├───────────┼───────────────────────┼─────────────────────────┤
 │ batch     │ coarse_tokens         │ [B,2,124,1536]          │
 ├───────────┼───────────────────────┼─────────────────────────┤
 │ 投影后    │ 各视觉 token          │ [B,2,·,1024]            │
 ├───────────┼───────────────────────┼─────────────────────────┤
 │ SELF 输入 │ inputs_embeds         │ [2B, ≤359, 1024]        │
 ├───────────┼───────────────────────┼─────────────────────────┤
 │ SELF 输出 │ hidden / VERIFY / ACT │ [2B,T,1024] / [2B,1024] │
 ├───────────┼───────────────────────┼─────────────────────────┤
 │ 候选匹配  │ class_logits          │ [B,2,9]                 │
 ├───────────┼───────────────────────┼─────────────────────────┤
 │ 轨迹      │ self_waypoints        │ [B,2,8,3]               │
 ├───────────┼───────────────────────┼─────────────────────────┤
 │ COOP 输入 │ inputs_embeds         │ [B, ≤450, 1024]         │
 ├───────────┼───────────────────────┼─────────────────────────┤
 │ COOP 输出 │ candidates / mode     │ [B,2,4,8,3] / [B,2,4]   │
 ├───────────┼───────────────────────┼─────────────────────────┤
 │ 最终      │ routed_waypoints      │ [B,2,8,3]               │
 ├───────────┼───────────────────────┼─────────────────────────┤
 │ 损失      │ loss                  │ 标量                    │
 └───────────┴───────────────────────┴─────────────────────────┘

 ────────────────────────────────────────────────────────────────────────────────

 一句话总结

 │ 完整训练数据流 = 离线预处理（图片→token/检测/标注）→ Sampler 选帧 → Dataset 读缓存+增强 → collate 组 batch → 两次串行 Qwen 前向（SELF 匹配+规划，COOP 协同恢复）→ 路由 → 六路
 │ 损失加权 → 反向传播+梯度裁剪+优化器 → 日志+checkpoint，如此循环直到 5 个 epoch 结束。

##评估审查



 评估（eval）数据流：从启动到指标

1. 启动入口

```text
   eval_airground_coop_v3.py 的 main                                                                                                                                            
      │                                                                                                                                                                         
      └─ 把 UnrealZooMultiAgentPlanner 替换成 AirGroundCoopV3Planner                                                                                                            
         │                                                                                                                                                                      
         └─ 委托 eval_unrealzoo_multi_agent.py 的 main（真正干活）                                                                                                              
```

 eval_airground_coop_v3.py 只负责：解析参数 → 挂载 V3 planner → 交给通用 runtime 跑。

2. 数据读取（不是读训练 JSONL，而是读"人的录制轨迹"）

```text
   load_recorded_target_trajectories(recorded_target_dir, episodes, env_id)                                                                                                     
      │                                                                                                                                                                         
      ├─ 从 split_manifest.json（或原始 episode）读 test 清单                                                                                                                   
      ├─ 按 env_id 过滤场景                                                                                                                                                     
      └─ 每个 episode 得到：                                                                                                                                                    
           episode_name    人的轨迹名字                                                                                                                                         
           poses           人的世界坐标序列（回放用）                                                                                                                           
           instruction     任务指令（STT/DT/AT）                                                                                                                                
           task_type       任务类型                                                                                                                                             
```

 │ 关键区别：训练读的是 vision_cache + perception_cache（离线特征），评估读的是人的世界坐标轨迹，用于在仿真里让"目标人物"按录制路线走。

3. 环境启动

```text
   make_env(args) → 创建 UnrealZoo Gym 环境（UE 引擎）                                                                                                                          
                                                                                                                                                                                
   setup_episode（每个 episode）：                                                                                                                                              
     ├─ reset_env：重置环境                                                                                                                                                     
     ├─ classify_coop_agents：确定 target / drone / robotdog 的 actor ID                                                                                                        
     ├─ 设置外观（appearance）                                                                                                                                                  
     └─ 设置初始位置、相机                                                                                                                                                      
```

4. 闭环循环（run_episode，核心）

 每个 episode 最多 max_steps 步，每步：

```text
   ┌─────────────────────────────────────────────────────────┐                                                                                                                  
   │ ① 读观察（_read_agent_pair）                            │                                                                                                                  
   │    drone/dog 的 RGB 图 + 可见性 mask + GT bbox（指标用）│                                                                                                                  
   └──────────────────────────┬──────────────────────────────┘                                                                                                                  
                              ▼                                                                                                                                                 
   ┌─────────────────────────────────────────────────────────┐                                                                                                                  
   │ ② 模型推理（planner.predict）                           │                                                                                                                  
   │    YOLO 感知 → DINO+SigLIP 编码 → reference 记忆         │                                                                                                                 
   │    → 模型 forward（SELF + COOP + 路由）→ waypoints      │                                                                                                                  
   └──────────────────────────┬──────────────────────────────┘                                                                                                                  
                              ▼                                                                                                                                                 
   ┌─────────────────────────────────────────────────────────┐                                                                                                                  
   │ ③ 动作转换（planner.waypoints_to_actions）              │                                                                                                                  
   │    waypoint [x,y,yaw] → 逆动力学 → 速度/yaw 命令        │                                                                                                                  
   └──────────────────────────┬──────────────────────────────┘                                                                                                                  
                              ▼                                                                                                                                                 
   ┌─────────────────────────────────────────────────────────┐                                                                                                                  
   │ ④ 环境步进（deterministic_data_collection_step）        │                                                                                                                  
   │    目标（人）按录制轨迹走 + drone/dog 执行模型动作       │                                                                                                                 
   │    → 世界更新 → 下一步观察                               │                                                                                                                 
   └──────────────────────────┬──────────────────────────────┘                                                                                                                  
                              ▼                                                                                                                                                 
   ┌─────────────────────────────────────────────────────────┐                                                                                                                  
   │ ⑤ 累积指标：following / bbox IoU / visible / collision │                                                                                                                   
   └──────────────────────────┬──────────────────────────────┘                                                                                                                  
                              ▼                                                                                                                                                 
                       回到 ①（下一帧）                                                                                                                                         
```

 这就是"闭环"：模型输出 → 动作 → 环境 → 新观察 → 模型……循环直到终态。

5. 模型侧 predict 的细节（② 内部）

```text
   drone_frame, dog_frame（RGB）                                                                                                                                                
      │                                                                                                                                                                         
      ├─ YOLO 感知 → Top-K 候选                                                                                                                                                 
      ├─ DINO+SigLIP → coarse/fine 视觉 token                                                                                                                                   
      ├─ reference 记忆（3帧确认 / 12帧释放，动态）                                                                                                                             
      │                                                                                                                                                                         
      ▼                                                                                                                                                                         
   model.forward(...)                                                                                                                                                           
      ├─ SELF：候选匹配 K+1 → 选中目标 → self_waypoints                                                                                                                         
      ├─ COOP：JEPA + 4-mode decoder → cooperative_waypoints                                                                                                                    
      └─ 路由 → waypoints (2,8,3)                                                                                                                                               
      │                                                                                                                                                                         
      ▼                                                                                                                                                                         
   _route_waypoints（带滞回的可见性路由）                                                                                                                                       
      SELF / COOPERATIVE / BELIEF / SEARCH                                                                                                                                      
      │                                                                                                                                                                         
      ▼                                                                                                                                                                         
   waypoints_to_actions（逆动力学）                                                                                                                                             
      → drone/dog 动作                                                                                                                                                          
```

6. 指标计算

 每步累积，episode 结束汇总（stat）：

 ┌────────────────────────────┬───────────────┬──────────────────────────────────────────────────┐
 │ 指标                       │ 含义          │ 计算                                             │
 ├────────────────────────────┼───────────────┼──────────────────────────────────────────────────┤
 │ success（SR）              │ 终态成功      │ 无碰撞 + 状态正常 + 完成 + 最终都在 follow 范围  │
 ├────────────────────────────┼───────────────┼──────────────────────────────────────────────────┤
 │ joint_following_rate       │ 联合跟踪率    │ 每步"两 agent 都可见且距离在 [min,max] 内"的比例 │
 ├────────────────────────────┼───────────────┼──────────────────────────────────────────────────┤
 │ drone_following_rate       │ drone 跟踪率  │ drone 单独 following 比例                        │
 ├────────────────────────────┼───────────────┼──────────────────────────────────────────────────┤
 │ robotdog_following_rate    │ dog 跟踪率    │ dog 单独 following 比例                          │
 ├────────────────────────────┼───────────────┼──────────────────────────────────────────────────┤
 │ drone_centered_rate        │ 目标居中率    │ 目标在视野中心的步数比例                         │
 ├────────────────────────────┼───────────────┼──────────────────────────────────────────────────┤
 │ drone_bbox_iou_mean        │ bbox IoU      │ 模型预测框 vs GT 框的 IoU 均值                   │
 ├────────────────────────────┼───────────────┼──────────────────────────────────────────────────┤
 │ visible_accuracy           │ 可见性准确率  │ 模型可见性预测 vs GT 可见性                      │
 ├────────────────────────────┼───────────────┼──────────────────────────────────────────────────┤
 │ collision                  │ 碰撞          │ UE 物理碰撞 或 距离 < 阈值                       │
 ├────────────────────────────┼───────────────┼──────────────────────────────────────────────────┤
 │ final_distance             │ 最终距离      │ 最后一步到人的距离                               │
 ├────────────────────────────┼───────────────┼──────────────────────────────────────────────────┤
 │ lost_count / failure_count │ 丢失/失败步数 │ 连续丢失触发 early stop                          │
 ├────────────────────────────┼───────────────┼──────────────────────────────────────────────────┤
 │ model_latency_ms / fps     │ 推理速度      │ 纯模型前向耗时                                   │
 └────────────────────────────┴───────────────┴──────────────────────────────────────────────────┘

### 核心指标的定义

```python
   # following（每步）                                                                                                                                                          
   drone_following = drone_visible & (min_dist <= dist <= max_dist)                                                                                                             
                                                                                                                                                                                
   # following_rate = 满足的步数 / 总步数                                                                                                                                       
   drone_rate = sum(following) / total_step                                                                                                                                     
                                                                                                                                                                                
   # success（终态）                                                                                                                                                            
   success = (无碰撞) & (状态正常) & (完成或步数够) & (最终两 agent 都在范围)                                                                                                   
```

7. 结果保存 + 汇总

```text
   write_episode_outputs（每个 episode）                                                                                                                                        
      ├─ 结果 JSON（stat + setup + infos）                                                                                                                                      
      └─ 视频帧（drone/dog/global RGB）                                                                                                                                         
                                                                                                                                                                                
   tools/calculate_unrealzoo_metrics（所有 episode 跑完后）                                                                                                                     
      └─ 汇总 SR / TR / bbox IoU / visible accuracy / collision 等                                                                                                              
```

 ────────────────────────────────────────────────────────────────────────────────

 完整评估数据流总图

```text
   recorded_target_dir（人的世界轨迹）                                                                                                                                          
           │                                                                                                                                                                    
           ▼                                                                                                                                                                    
   load_recorded_target_trajectories ──→ instruction / task_type / poses                                                                                                        
           │                                                                                                                                                                    
           ▼                                                                                                                                                                    
   make_env（UE 引擎）+ setup_episode（重置、设目标/外观）                                                                                                                      
           │                                                                                                                                                                    
           ▼                                                                                                                                                                    
   ┌──── 闭环循环（每 episode，每步）───────────────────┐                                                                                                                       
   │                                                    │                                                                                                                       
   │  RGB 观察 ──→ YOLO+DINO/SigLIP ──→ 模型 forward    │                                                                                                                       
   │                                    │               │                                                                                                                       
   │                                    ▼               │                                                                                                                       
   │                            waypoints (2,8,3)       │                                                                                                                       
   │                                    │               │                                                                                                                       
   │                                    ▼               │                                                                                                                       
   │                          逆动力学 → 动作           │                                                                                                                       
   │                                    │               │                                                                                                                       
   │                                    ▼               │                                                                                                                       
   │                     环境 step（人走录制轨迹，       │                                                                                                                      
   │                     drone/dog 执行模型动作）        │                                                                                                                      
   │                                    │               │                                                                                                                       
   │                                    ▼               │                                                                                                                       
   │                     累积指标（following/IoU/...）   │                                                                                                                      
   │                                    │               │                                                                                                                       
   │                                    └──→ 下一帧 ──┘ │                                                                                                                       
   └────────────────────────────────────────────────────┘                                                                                                                       
           │                                                                                                                                                                    
           ▼                                                                                                                                                                    
   写结果 JSON + 视频 → calculate_unrealzoo_metrics 汇总
```

 ────────────────────────────────────────────────────────────────────────────────

 评估 vs 训练的关键区别

 ┌───────────┬──────────────────────────────┬────────────────────────────────────┐
 │           │ 训练                         │ 评估                               │
 ├───────────┼──────────────────────────────┼────────────────────────────────────┤
 │ 数据      │ 离线 vision/perception cache │ 实时读 RGB + 在线编码              │
 ├───────────┼──────────────────────────────┼────────────────────────────────────┤
 │ 目标运动  │ 固定的 GT 轨迹标签           │ 人在仿真里按录制轨迹走             │
 ├───────────┼──────────────────────────────┼────────────────────────────────────┤
 │ 闭环      │ ❌ 开环（标签固定）          │ ✅ 闭环（动作影响下一步）          │
 ├───────────┼──────────────────────────────┼────────────────────────────────────┤
 │ GT 用途   │ 造损失标签                   │ 只算指标（可见性、bbox IoU、距离） │
 ├───────────┼──────────────────────────────┼────────────────────────────────────┤
 │ reference │ 静态 GT 快照                 │ 动态记忆（3帧确认/12帧释放）       │
 ├───────────┼──────────────────────────────┼────────────────────────────────────┤
 │ 输出      │ loss                         │ waypoints → 动作 → 环境            │
 └───────────┴──────────────────────────────┴────────────────────────────────────┘

 ────────────────────────────────────────────────────────────────────────────────

 一句话总结

 │ 评估数据流 = 读录制的人体轨迹 → 启动 UE 环境 → 闭环循环（读 RGB → 感知+编码 → 模型 forward → 路由 → 逆动力学动作 → 环境步进 → 累积指标）→ episode 结束算 SR/TR/bbox
 │ IoU/visible accuracy/碰撞 → 汇总。其中 GT 可见性和 bbox 只用于算指标，不进模型；模型只有一次机会在闭环里持续跟踪。



 已完成配置修改并验证。

### 已修改文件

- config/airground_cooperative_tracking_v3_yh_stable_joint.yaml
  - train_temporal_stride: 3
  - epochs: 6
- eval_airground_v3_runtime.py
  - 强制 drone/dog 跟随距离均为 1.0–6.0 m
- sh/eval_airground_coop_v3_24h_3gpu.sh
  - drone/dog 跟随距离均改为 1.0–6.0 m

 验证结果：23 passed。

 │ 状态更新：旧训练已在 step 11802 主动停止。下一批不加载 AirGround checkpoint、禁用 reference，使用 stride=5、epochs=10，并把 receiver corruption 拆成每 epoch 一个小台阶的十阶段线性 curriculum。

 新参数表

 一、训练参数

 ┌───────────────────┬──────────────────────────────────────┬────────────┬─────────────────────────────────────────────────────────────────────────┐
 │ 类别              │ 参数                                 │ 新值       │ 作用                                                                    │
 ├───────────────────┼──────────────────────────────────────┼────────────┼─────────────────────────────────────────────────────────────────────────┤
 │ 数据采样          │ train_temporal_stride                │ 3          │ 每个 epoch 采用三分之一时间相位样本；连续 3 个 epoch 可覆盖全部时间位置 │
 ├───────────────────┼──────────────────────────────────────┼────────────┼─────────────────────────────────────────────────────────────────────────┤
 │ 训练周期          │ epochs                               │ 6          │ 总训练 6 轮，相当于时间相位完整覆盖两轮                                 │
 ├───────────────────┼──────────────────────────────────────┼────────────┼─────────────────────────────────────────────────────────────────────────┤
 │ 候选数            │ candidate_top_k                      │ 8          │ 每帧最多输入 8 个候选人物，外加 1 个 NO_TARGET 类                       │
 ├───────────────────┼──────────────────────────────────────┼────────────┼─────────────────────────────────────────────────────────────────────────┤
 │ 历史长度          │ history                              │ 31         │ 每个 agent 使用 31 帧历史视觉特征                                       │
 ├───────────────────┼──────────────────────────────────────┼────────────┼─────────────────────────────────────────────────────────────────────────┤
 │ 轨迹长度          │ n_waypoints                          │ 8          │ 预测未来 8 个 waypoint                                                  │
 ├───────────────────┼──────────────────────────────────────┼────────────┼─────────────────────────────────────────────────────────────────────────┤
 │ 动作维度          │ action_dims                          │ 3          │ 每点表示 [x, y, yaw]                                                    │
 ├───────────────────┼──────────────────────────────────────┼────────────┼─────────────────────────────────────────────────────────────────────────┤
 │ 图像尺寸          │ image_size                           │ 384        │ 离线视觉编码对应的输入尺寸                                              │
 ├───────────────────┼──────────────────────────────────────┼────────────┼─────────────────────────────────────────────────────────────────────────┤
 │ 特征维度          │ vision_feat_dim                      │ 1536       │ DINO+SigLIP 视觉特征维度                                                │
 ├───────────────────┼──────────────────────────────────────┼────────────┼─────────────────────────────────────────────────────────────────────────┤
 │ 匹配阈值          │ target_match_iou_threshold           │ 0.30       │ 候选与 GT 的 IoU 达到该值才作为目标候选标签                             │
 ├───────────────────┼──────────────────────────────────────┼────────────┼─────────────────────────────────────────────────────────────────────────┤
 │ reference dropout │ train_target_reference_dropout_prob  │ 0.20       │ 20% 丢弃初始视觉 reference，防止忽略 instruction                        │
 ├───────────────────┼──────────────────────────────────────┼────────────┼─────────────────────────────────────────────────────────────────────────┤
 │ drone 合成遮挡    │ train_synthetic_drone_occlusion_prob │ 0.50       │ 生成 drone 不可见训练样本                                               │
 ├───────────────────┼──────────────────────────────────────┼────────────┼─────────────────────────────────────────────────────────────────────────┤
 │ dog 合成遮挡      │ train_synthetic_dog_occlusion_prob   │ 0.50       │ 生成 dog 不可见训练样本                                                 │
 ├───────────────────┼──────────────────────────────────────┼────────────┼─────────────────────────────────────────────────────────────────────────┤
 │ LLM               │ llm_name                             │ Qwen3-0.6B │ SELF/COOP 共享语言模型                                                  │
 ├───────────────────┼──────────────────────────────────────┼────────────┼─────────────────────────────────────────────────────────────────────────┤
 │ 冻结 LLM          │ freeze_llm                           │ true       │ 不更新 Qwen 参数                                                        │
 ├───────────────────┼──────────────────────────────────────┼────────────┼─────────────────────────────────────────────────────────────────────────┤
 │ 文本长度          │ text_max_length                      │ 128        │ instruction 和 prompt 的 token 上限                                     │
 ├───────────────────┼──────────────────────────────────────┼────────────┼─────────────────────────────────────────────────────────────────────────┤
 │ 协同模式数        │ num_modes                            │ 4          │ COOP decoder 输出 4 种候选轨迹模式                                      │
 ├───────────────────┼──────────────────────────────────────┼────────────┼─────────────────────────────────────────────────────────────────────────┤
 │ batch/GPU         │ batch_size                           │ 48         │ 每张 GPU 的样本数                                                       │
 ├───────────────────┼──────────────────────────────────────┼────────────┼─────────────────────────────────────────────────────────────────────────┤
 │ GPU 数            │ DDP                                  │ 2          │ 两张 GPU 联合训练                                                       │
 ├───────────────────┼──────────────────────────────────────┼────────────┼─────────────────────────────────────────────────────────────────────────┤
 │ 梯度累积          │ grad_accum_steps                     │ 2          │ 每两次 micro-batch 更新一次                                             │
 ├───────────────────┼──────────────────────────────────────┼────────────┼─────────────────────────────────────────────────────────────────────────┤
 │ 有效全局 batch    │ —                                    │ 192        │ 48 × 2 GPU × 2 accumulation                                             │
 ├───────────────────┼──────────────────────────────────────┼────────────┼─────────────────────────────────────────────────────────────────────────┤
 │ 学习率            │ lr                                   │ 4e-5       │ 峰值学习率                                                              │
 ├───────────────────┼──────────────────────────────────────┼────────────┼─────────────────────────────────────────────────────────────────────────┤
 │ 调度器            │ lr_scheduler                         │ cosine     │ 余弦下降                                                                │
 ├───────────────────┼──────────────────────────────────────┼────────────┼─────────────────────────────────────────────────────────────────────────┤
 │ warmup            │ warmup_steps                         │ 300        │ 前 300 optimizer steps 学习率预热                                       │
 ├───────────────────┼──────────────────────────────────────┼────────────┼─────────────────────────────────────────────────────────────────────────┤
 │ 最低学习率        │ min_lr                               │ 4e-6       │ cosine 最低值                                                           │
 ├───────────────────┼──────────────────────────────────────┼────────────┼─────────────────────────────────────────────────────────────────────────┤
 │ 梯度裁剪          │ grad_clip                            │ 1.0        │ 防止梯度爆炸                                                            │
 ├───────────────────┼──────────────────────────────────────┼────────────┼─────────────────────────────────────────────────────────────────────────┤
 │ 混合精度          │ mixed_precision                      │ true       │ 使用混合精度降低显存和耗时                                              │
 ├───────────────────┼──────────────────────────────────────┼────────────┼─────────────────────────────────────────────────────────────────────────┤
 │ checkpoint 间隔   │ save_every                           │ 2000       │ 每 2000 optimizer steps 保存                                            │
 ├───────────────────┼──────────────────────────────────────┼────────────┼─────────────────────────────────────────────────────────────────────────┤
 │ epoch 保存        │ save_every_epochs                    │ 1          │ 每个 epoch 结束保存                                                     │
 ├───────────────────┼──────────────────────────────────────┼────────────┼─────────────────────────────────────────────────────────────────────────┤
 │ checkpoint 上限   │ max_ckpts                            │ 0          │ 不删除历史 checkpoint                                                   │
 ├───────────────────┼──────────────────────────────────────┼────────────┼─────────────────────────────────────────────────────────────────────────┤
 │ 随机种子          │ seed                                 │ 731        │ 控制采样、遮挡和初始化随机性                                            │
 └───────────────────┴──────────────────────────────────────┴────────────┴─────────────────────────────────────────────────────────────────────────┘

 二、V3 闭环评估参数

 ┌────────────────┬──────────────────────────────┬────────────────────────────┬──────────────────────────────────────────────────┐
 │ 类别           │ 参数                         │ 新值                       │ 作用                                             │
 ├────────────────┼──────────────────────────────┼────────────────────────────┼──────────────────────────────────────────────────┤
 │ drone 跟随距离 │ drone_min/max_follow_dist    │ 1.0–6.0 m                  │ 目标可见且距离处于该范围时，drone 记为 following │
 ├────────────────┼──────────────────────────────┼────────────────────────────┼──────────────────────────────────────────────────┤
 │ dog 跟随距离   │ robotdog_min/max_follow_dist │ 1.0–6.0 m                  │ dog 使用相同的 following 判定范围                │
 ├────────────────┼──────────────────────────────┼────────────────────────────┼──────────────────────────────────────────────────┤
 │ 碰撞距离       │ human_collision_distance     │ 0.5 m                      │ agent 与目标距离小于该值即记人体碰撞             │
 ├────────────────┼──────────────────────────────┼────────────────────────────┼──────────────────────────────────────────────────┤
 │ 仿真步长       │ dt                           │ 0.1 s                      │ 确定性闭环每步推进 0.1 秒                        │
 ├────────────────┼──────────────────────────────┼────────────────────────────┼──────────────────────────────────────────────────┤
 │ 最大步数       │ max_steps                    │ 100                        │ 默认每个 episode 最多运行 100 步                 │
 ├────────────────┼──────────────────────────────┼────────────────────────────┼──────────────────────────────────────────────────┤
 │ 目标速度       │ human_speed                  │ 0.5 m/s                    │ 回放目标人物的速度设置                           │
 ├────────────────┼──────────────────────────────┼────────────────────────────┼──────────────────────────────────────────────────┤
 │ agent 最大速度 │ drone/dog                    │ 2.5 m/s                    │ 限制两类 follower 的控制速度                     │
 ├────────────────┼──────────────────────────────┼────────────────────────────┼──────────────────────────────────────────────────┤
 │ 控制方式       │ waypoint_control_mode        │ inverse_fixed_dt           │ 将预测 waypoint 转换为速度和 yaw 指令            │
 ├────────────────┼──────────────────────────────┼────────────────────────────┼──────────────────────────────────────────────────┤
 │ dog 侧向处理   │ robotdog_waypoint_y_mode     │ v3_nonholonomic_projection │ 适配机器狗非完整运动约束                         │
 ├────────────────┼──────────────────────────────┼────────────────────────────┼──────────────────────────────────────────────────┤
 │ 推理步长       │ policy_inference_stride       │ 3                          │ 每 3 个环境步推理一次，中间回放 waypoint segment │

 ├────────────────┼──────────────────────────────┼────────────────────────────┼──────────────────────────────────────────────────┤
 │ bbox 输入      │ bbox_source                  │ none                       │ GT bbox 不输入模型，只用于指标                   │
 ├────────────────┼──────────────────────────────┼────────────────────────────┼──────────────────────────────────────────────────┤
 │ 检测门槛       │ bbox_motion_min_confidence   │ 0.25                       │ 低置信度 bbox 不用于图像空间辅助控制             │
 ├────────────────┼──────────────────────────────┼────────────────────────────┼──────────────────────────────────────────────────┤
 │ bbox 平滑      │ bbox_motion_ema_alpha        │ 0.20                       │ 对 bbox 运动信号进行 EMA 平滑                    │
 ├────────────────┼──────────────────────────────┼────────────────────────────┼──────────────────────────────────────────────────┤
 │ 丢失容忍       │ max_lost_steps               │ 400                        │ 避免录制轨迹因短期丢失过早结束                   │
 ├────────────────┼──────────────────────────────┼────────────────────────────┼──────────────────────────────────────────────────┤
 │ 最短成功步数   │ min_success_steps            │ 20                         │ 提前终止时至少运行 20 步才可能成功               │
 ├────────────────┼──────────────────────────────┼────────────────────────────┼──────────────────────────────────────────────────┤
 │ GT oracle      │ action/bbox oracle           │ 禁用                       │ 保证结果来自模型闭环控制                         │
 └────────────────┴──────────────────────────────┴────────────────────────────┴──────────────────────────────────────────────────┘

### 关于两个 stride

- `train_temporal_stride=5` 表示 current-row 采样间隔为 `5×0.1=0.5 s`。
- `policy_inference_stride=5` 表示评估模型调用间隔为 0.5 s，两者层级不同、数值一致。
- JSON waypoint 和环境动作仍是相邻 0.1 s；kinematics、feasible recovery 与 inverse control 必须继续使用 0.1 s，不能误改为 0.5 s。
