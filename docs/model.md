# OpenTrackVLA 模型架构

## 概览

OpenTrackVLA 是一个 Vision-Language-Action (VLA) 模型，用于将单目视频和自然语言指令转换为可执行的短期路径点(waypoints)。

**核心组件：**
- 视觉编码器：DINOv2 + SigLIP（双塔结构）
- 语言模型骨干：Qwen3-0.6B（冻结）
- 跨模态投影器：2层 MLP
- 时间-视角嵌入器 (TVI)
- 规划头：3层 MLP（直接回归）

---

## 架构图

```
输入
─────────────────────────────────────────────────────────────────
│                                                               │
│  ┌─────────────────────┐          ┌─────────────────────┐    │
│  │   历史帧 (H=31帧)   │          │      当前帧         │    │
│  │   粗粒度采样        │          │    细粒度采样       │    │
│  └──────────┬──────────┘          └──────────┬──────────┘    │
│             │                                │               │
│             ▼                                ▼               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              VisionFeatureCacher                      │   │
│  │  ┌─────────────┐         ┌─────────────┐             │   │
│  │  │  DINOv2     │         │  SigLIP     │             │   │
│  │  │  ViT-S/16   │         │  SO400M     │             │   │
│  │  │  (384维)    │         │  (1152维)   │             │   │
│  │  └──────┬──────┘         └──────┬──────┘             │   │
│  │         └──────────┬────────────┘                    │   │
│  │                    ▼                                 │   │
│  │              concat → 1536维                         │   │
│  │                    │                                 │   │
│  │                    ▼                                 │   │
│  │         ┌──────────────────────┐                     │   │
│  │         │   GridPool           │                     │   │
│  │         │ 粗: 4 tokens/帧      │                     │   │
│  │         │ 细: 64 tokens/帧     │                     │   │
│  │         └──────────────────────┘                     │   │
│  └──────────────────────────────────────────────────────┘   │
│             │                                │               │
│             ▼                                ▼               │
│      coarse_tokens                    fine_tokens            │
│      (B, H×4, 1536)                  (B, 64, 1536)           │
─────────────────────────────────────────────────────────────────
模型处理
─────────────────────────────────────────────────────────────────
│                                                               │
│  ┌──────────────────────────────────────────────────────┐    │
│  │           CrossModalityProjector                      │    │
│  │  LayerNorm → Linear(1536→D) → GELU → Linear(D→D)     │    │
│  │                    D = LLM隐藏维度                    │    │
│  └──────────────────────┬───────────────────────────────┘    │
│                         │                                    │
│                         ▼                                    │
│  ┌──────────────────────────────────────────────────────┐    │
│  │              TVIEmbedder (时间-视角嵌入)              │    │
│  │                                                       │    │
│  │  为每帧插入时间标记:                                  │    │
│  │  [T0] [粗tokens_0] [T1] [粗tokens_1] ... [TH] [细tok] │    │
│  │                                                       │    │
│  │  时间嵌入 = time_emb[t] + view_emb[v] + kind_emb[k]  │    │
│  │  • time_emb: 位置编码 (max_time=4096)                │    │
│  │  • view_emb: 视角区分 (max_views=1)                  │    │
│  │  • kind_emb: 粗/细类型 (kind=0 粗, kind=1 细)        │    │
│  └──────────────────────┬───────────────────────────────┘    │
│                         │                                    │
│                         ▼                                    │
│  ┌──────────────────────────────────────────────────────┐    │
│  │                   序列拼接                            │    │
│  │                                                       │    │
│  │  [文本指令] + [历史粗视觉] + [当前细视觉] + [ACT]     │    │
│  │                                                       │    │
│  │  • 文本: tokenizer编码后的instruction                │    │
│  │  • ACT: 可学习的动作查询token                        │    │
│  └──────────────────────┬───────────────────────────────┘    │
│                         │                                    │
│                         ▼                                    │
│  ┌──────────────────────────────────────────────────────┐    │
│  │                Qwen3-0.6B LLM                         │    │
│  │                                                       │    │
│  │  • 参数量: 0.6B                                      │    │
│  │  • 隐藏维度 D: 根据Qwen配置                          │    │
│  │  • 训练时冻结 (freeze_llm=True)                      │    │
│  │  • 仅用于特征提取，不做语言生成                      │    │
│  └──────────────────────┬───────────────────────────────┘    │
│                         │                                    │
│                         ▼                                    │
│              取序列最后一个位置的隐状态                      │
│              h_act = last_hidden_state[:, -1, :]             │
│                         │                                    │
│                         ▼                                    │
│  ┌──────────────────────────────────────────────────────┐    │
│  │              PlannerHead3L (规划头)                   │    │
│  │                                                       │    │
│  │  LayerNorm(D)                                        │    │
│  │       ↓                                              │    │
│  │  Linear(D → 2D) + GELU                               │    │
│  │       ↓                                              │    │
│  │  Linear(2D → 2D) + GELU                              │    │
│  │       ↓                                              │    │
│  │  Linear(2D → n_waypoints × action_dims)              │    │
│  │       ↓                                              │    │
│  │  tanh() → 归一化到 [-1, 1]                           │    │
│  │       ↓                                              │    │
│  │  × alpha_task → 缩放到实际单位                       │    │
│  └──────────────────────┬───────────────────────────────┘    │
│                         │                                    │
│                         ▼                                    │
─────────────────────────────────────────────────────────────────
输出
─────────────────────────────────────────────────────────────────
│                                                               │
│                    Waypoints 预测                             │
│              形状: (B, n_waypoints, action_dims)              │
│              默认: (B, 8, 3)                                  │
│                                                               │
│              每个waypoint包含:                                │
│              • x: 前向位移 (米)                               │
│              • y: 侧向位移 (米)                               │
│              • θ: 航向角变化 (弧度)                           │
│                                                               │
─────────────────────────────────────────────────────────────────
```

---

## 核心模块详解

### 1. VisionFeatureCacher

视觉特征提取器，使用双塔结构：

| 编码器 | 模型 | 输出维度 | 用途 |
|--------|------|----------|------|
| DINOv2 | ViT-S/16 | 384 | 自监督视觉特征 |
| SigLIP | SO400M-patch14-384 | 1152 | 视觉-语言对齐特征 |

**输出**: 拼接后 1536 维特征，经 GridPool 压缩为固定数量 tokens

### 2. TVIEmbedder (Time-View Interleaver)

```python
class TVIEmbedder(nn.Module):
    def __init__(self, d_model: int, max_time: int = 4096, max_views: int = 1):
        self.time_emb   = nn.Embedding(max_time, d_model)  # 时间位置
        self.view_emb   = nn.Embedding(max_views, d_model) # 视角
        self.kind_emb   = nn.Embedding(2, d_model)         # 粗/细类型
        self.angle_proj = nn.Linear(2, d_model)            # 可选: 航向角
        self.bbox_proj  = nn.Linear(4, d_model)            # 可选: 边界框
```

**作用**: 在视觉 tokens 序列中插入时间标记，让模型理解时序关系

### 3. CrossModalityProjector

```python
class CrossModalityProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim), nn.GELU(),
            nn.Linear(out_dim, out_dim)
        )
```

**作用**: 将视觉特征投影到 LLM 的嵌入空间

### 4. PlannerHead3L

```python
class PlannerHead3L(nn.Module):
    def __init__(self, d_model, n_waypoints, action_dims, use_tanh=True):
        hid = d_model * 2
        out_dim = n_waypoints * action_dims
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hid), nn.GELU(),
            nn.Linear(hid, hid), nn.GELU(),
            nn.Linear(hid, out_dim)
        )
```

**作用**: 直接从 LLM 隐状态回归出未来路径点

---

## 训练配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `llm_name` | Qwen/Qwen3-0.6B | LLM 骨干 |
| `freeze_llm` | True | 冻结 LLM 参数 |
| `n_waypoints` | 8 | 预测路径点数量 |
| `history` | 31 | 历史帧数量 |
| `action_dims` | 3 | 动作维度 (x, y, θ) |
| `alpha_xy` | 2.0 | XY 方向缩放系数 |
| `use_tanh_actions` | True | 使用 tanh 限制输出范围 |

---

## 数据流维度

假设 batch_size=B, history=31, n_waypoints=8:

| 阶段 | 张量 | 形状 |
|------|------|------|
| 粗视觉输入 | coarse_tokens | (B, 124, 1536) |
| 细视觉输入 | fine_tokens | (B, 64, 1536) |
| 投影后粗视觉 | vis_c | (B, ~155, D) |
| 投影后细视觉 | vis_f | (B, ~65, D) |
| 文本嵌入 | txt_emb | (B, L_txt, D) |
| LLM 输入序列 | seq | (B, L_total, D) |
| ACT 隐状态 | h_act | (B, D) |
| 输出路径点 | waypoints | (B, 8, 3) |

---

## 损失函数

```python
loss = F.mse_loss(pred[valid_mask], target[valid_mask])
```

使用 **MSE 损失**，仅在有效路径点上计算（通过 `valid_mask` 过滤）。

---

## 重要说明

⚠️ **此模型不使用 Diffusion**

- 规划头是纯 MLP 直接回归
- 没有噪声添加/去噪过程
- 没有时间步嵌入
- 没有 DDPM/DDIM scheduler

如需添加 Diffusion Policy，需要将 `PlannerHead3L` 替换为 Diffusion 去噪网络。
