"""UnrealZoo 双 Agent Anchor Diffusion 模型定义。

整体功能：
- 复用 OpenTrackVLA 的视觉、文本、TVI 和 grounding 编码链路。
- 对 K-means 得到的预定义轨迹锚点加噪，再用条件 DiT 和 DDIM 两步去噪。
- 为无人机与机器狗分别生成多条候选轨迹、候选分数和最终选中路点。

核心类：
- ``AnchorDiffusionActionModel``：单个 Agent 的锚点加噪、DiT 去噪、候选评分与选轨。
- ``AnchorDiTBlock`` / ``SinusoidalTimestepEmbedding``：扩散 Transformer 与时间步编码。
- ``MultiAgentOpenTrackVLA``：共享场景编码，调用两个独立 Anchor Diffusion 动作头。
- ``GroundingHead``：从两个 Agent-specific GND state 分别预测 bbox 和可见性。

关键函数：
- ``fit_trajectory_anchors_kmeans`` / ``save_trajectory_anchors``：聚类并保存轨迹锚点。
- ``anchor_diffusion_tracking_loss``：最近锚轨迹回归损失与候选评分 BCE。
- ``_load_trajectory_anchors``：加载锚点文件或创建兜底锚点。
- ``_run_inference`` / ``parse_args``：扩散模型独立推理入口。

主要输出：
``waypoints``、``candidate_trajectories``、``candidate_scores``、
``refined_bbox``、``visible_logits``；训练时额外返回 ``action_loss``。

训练入口为 ``train_unrealzoo_anchor_diffusion.py``，闭环评估入口为
``eval_unrealzoo_multi_agent.py``。
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, Union
import os, json, math, argparse, time
from pathlib import Path
from contextlib import nullcontext
from PIL import Image, ImageDraw

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from diffusers.schedulers import DDIMScheduler
from transformers import AutoTokenizer, AutoModel
from tools.cache_gridpool import VisionFeatureCacher, VisionCacheConfig, grid_pool_tokens, adapt_siglip_grid


# ----------------------- 通用工具与数据加载 -----------------------

# Silence tokenizers fork warnings in dataloader workers
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


# 双 Agent 模型里的 token 类型编号。
# 数据流含义：
# - HISTORY: 历史帧粗粒度视觉 token，用于理解过去运动趋势。
# - CURRENT: 当前帧细粒度视觉 token，用于定位当前目标。
# - BBOX: 当前目标框 token，把检测/跟踪先验注入 LLM。
# - ACT: 每个 Agent 的动作查询 token，LLM 在这个位置输出规划隐藏状态。
# - GND: 每个 Agent 各有一个 grounding 查询 token，输出对应坐标系的 bbox/visibility 状态。
KIND_HISTORY = 0
KIND_CURRENT = 1
KIND_BBOX = 2
KIND_ACT = 3
KIND_GND = 4


def load_tokens_file(path: str) -> torch.Tensor:
    try:
        obj = torch.load(path, map_location='cpu')
    except Exception:
        try:
            obj = torch.load(path, map_location='cpu', weights_only=False)
        except Exception as e:
            raise e
    if isinstance(obj, torch.Tensor):
        return obj.float()
    if isinstance(obj, dict):
        for k in ("V", "Vfine", "Vcoarse", "tokens", "feat", "features"):
            if k in obj and isinstance(obj[k], torch.Tensor):
                t = obj[k]
                if t.dim() == 3 and t.size(0) == 1:
                    t = t[0]
                return t.float()
    raise ValueError(f"Unrecognized token file: {path}")


def integrate_actions_to_waypoints(actions: np.ndarray, n_waypoints: int, dt: float = 0.2) -> np.ndarray:
    a = np.asarray(actions, dtype=np.float32)
    if a.ndim == 1: a = a[None, :]
    T, D = a.shape
    vx = a[:, 0].astype(np.float32)
    vy = a[:, 1].astype(np.float32) if D > 1 else np.zeros_like(vx)
    wz = a[:, 2].astype(np.float32) if D > 2 else np.zeros_like(vx)

    x = np.zeros(T, dtype=np.float32)
    y = np.zeros(T, dtype=np.float32)
    th = np.zeros(T, dtype=np.float32)

    for t in range(1, T):
        th[t] = th[t-1] + wz[t-1] * dt
        c, s = np.cos(th[t-1]), np.sin(th[t-1])
        x[t] = x[t-1] + (c * vx[t-1] - s * vy[t-1]) * dt
        y[t] = y[t-1] + (s * vx[t-1] + c * vy[t-1]) * dt

    traj = np.stack([x, y, th], axis=-1)
    if n_waypoints <= 1: return traj[-1:]
    idx = np.linspace(0, T-1, n_waypoints).round().astype(int)
    return traj[idx]


# Flexible loader for JSON/JSONL datasets and directories

def _read_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    with open(file_path, 'r') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            examples.append(json.loads(s))
    return examples


def load_examples_from_path(train_path: str) -> List[Dict[str, Any]]:
    p = Path(train_path)
    if p.is_dir():
        jsonl_files = sorted(p.rglob('*.jsonl'))
        if len(jsonl_files) == 0:
            raise FileNotFoundError(f"No .jsonl files found under directory: {train_path}")
        all_items: List[Dict[str, Any]] = []
        for fp in jsonl_files:
            all_items.extend(_read_jsonl_file(str(fp)))
        return all_items
    if p.is_file():
        if p.suffix.lower() == '.jsonl':
            return _read_jsonl_file(str(p))
        if p.suffix.lower() == '.json':
            with open(p, 'r') as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"JSON file must contain a list at top-level: {train_path}")
            return data
        raise ValueError(f"Unsupported file type: {train_path}")
    raise FileNotFoundError(f"Path does not exist: {train_path}")


# ----------------------- 单 Agent基础模型组件 -----------------------

class TVIEmbedder(nn.Module):
    def __init__(self, d_model: int, max_time: int = 4096, max_views: int = 1):
        super().__init__()
        self.time_emb   = nn.Embedding(max_time, d_model)
        self.view_emb   = nn.Embedding(max_views, d_model)
        self.kind_emb   = nn.Embedding(2, d_model)
        self.angle_proj = nn.Linear(2, d_model)
        self.bbox_proj  = nn.Linear(4, d_model)

    def make_time_token(self, t_scalar: int, kind_id: int, view_id: int = 0,
                        device: Optional[torch.device] = None) -> torch.Tensor:
        tok = self.time_emb.weight[t_scalar] + self.view_emb.weight[view_id] + self.kind_emb.weight[kind_id]
        return tok.to(device) if device is not None else tok

    def make_angle_token(self, theta: float, kind_id: int, view_id: int = 0,
                         device: Optional[torch.device] = None) -> torch.Tensor:
        theta = (theta + math.pi) % (2*math.pi) - math.pi
        sincos = torch.tensor([math.sin(theta), math.cos(theta)],
                              dtype=self.angle_proj.weight.dtype,
                              device=device)
        ang = F.linear(sincos, self.angle_proj.weight, self.angle_proj.bias)
        tok = ang + self.view_emb.weight[view_id] + self.kind_emb.weight[kind_id]
        return tok

    def make_bbox_token(self, bbox: torch.Tensor, kind_id: int, view_id: int = 1,
                         device: Optional[torch.device] = None) -> torch.Tensor:
        if bbox.dim() == 1:
            bbox = bbox.unsqueeze(0)
        bb = bbox.to(device=device, dtype=self.bbox_proj.weight.dtype)
        emb = F.linear(bb, self.bbox_proj.weight, self.bbox_proj.bias)
        emb = emb + self.view_emb.weight[view_id] + self.kind_emb.weight[kind_id]
        return emb.squeeze(0)

# 功能：跨模态投影器，将视觉token投影到与LLM相同的维度空间  
class CrossModalityProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim), nn.GELU(),
            nn.Linear(out_dim, out_dim)
        )
    def forward(self, x): return self.net(x)

# 功能：规划头部，将LLM输出的动作token转换为具体的路径点
class PlannerHead3L(nn.Module):
    def __init__(self, d_model: int, n_waypoints: int, action_dims: int, use_tanh: bool = True):
        super().__init__()
        hid = d_model * 2
        out_dim = n_waypoints * action_dims
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hid), nn.GELU(),
            nn.Linear(hid, hid), nn.GELU(),
            nn.Linear(hid, out_dim)
        )
        self.nw = n_waypoints
        self.ad = action_dims
        self.use_tanh = use_tanh
    def forward(self, act_h: torch.Tensor) -> torch.Tensor:
        y = self.mlp(act_h)
        if self.use_tanh:
            y = torch.tanh(y)
        return y.view(-1, self.nw, self.ad)


# ----------------------- 轨迹锚点生成与加载 -----------------------
#
# 本节迁移自 DiffusionDrive 的 truncated anchor diffusion 思路，并按 TrackVLA
# 论文配置改造为：轨迹 K-means 锚点 + 锚点附近加噪 + DiT 去噪 + 两步 DDIM
# + 多模态分数预测。DDIM 调度直接使用 DiffusionDrive 同款 diffusers 实现。

def fit_trajectory_anchors_kmeans(
    trajectories: np.ndarray,
    num_anchors: int,
    valid_mask: Optional[np.ndarray] = None,
    num_iters: int = 100,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """不依赖 scikit-learn，对固定长度轨迹执行 K-means 聚类。

    Args:
        trajectories: (S, Nw, D) trajectories in the same local frame and units
            used by the model.
        num_anchors: Number of trajectory modes M.
        valid_mask: Optional (S, Nw) mask for partial horizons.
        num_iters: Maximum K-means iterations.
        seed: Deterministic initialization seed.

    Returns:
        anchors: (M, Nw, D) cluster centers.
        assignment: (S,) nearest anchor index for every sample.
    """
    traj = np.asarray(trajectories, dtype=np.float32)
    if traj.ndim != 3:
        raise ValueError(f"trajectories must have shape (S,Nw,D), got {traj.shape}")
    if traj.shape[0] < num_anchors:
        raise ValueError(f"Need at least {num_anchors} trajectories, got {traj.shape[0]}")
    mask = np.ones(traj.shape[:2], dtype=np.float32) if valid_mask is None else np.asarray(valid_mask, dtype=np.float32)
    if mask.shape != traj.shape[:2]:
        raise ValueError(f"valid_mask must have shape {traj.shape[:2]}, got {mask.shape}")
    mask = (mask > 0).astype(np.float32)
    rng = np.random.default_rng(seed)

    # 使用和最近锚分配完全相同的 masked trajectory distance 做 K-means++ 初始化，
    # 避免 partial horizon 的 padding 影响聚类，也比纯随机初始化稳定。
    centers = [traj[int(rng.integers(0, traj.shape[0]))].copy()]
    for _ in range(1, num_anchors):
        center_arr = np.stack(centers, axis=0)
        diff = traj[:, None] - center_arr[None]
        sq = np.sum(diff * diff, axis=-1)
        dist = np.sum(sq * mask[:, None], axis=-1) / np.maximum(mask.sum(axis=-1, keepdims=True), 1.0)
        nearest = dist.min(axis=1)
        nearest_sum = float(nearest.sum())
        if nearest_sum <= 1e-8:
            centers.append(traj[int(rng.integers(0, traj.shape[0]))].copy())
        else:
            probs = nearest / nearest_sum
            centers.append(traj[int(rng.choice(traj.shape[0], p=probs))].copy())
    centers_arr = np.stack(centers, axis=0)
    assignment = np.full((traj.shape[0],), -1, dtype=np.int64)

    for _ in range(max(1, num_iters)):
        diff = traj[:, None] - centers_arr[None]
        sq = np.sum(diff * diff, axis=-1)
        dist = np.sum(sq * mask[:, None], axis=-1) / np.maximum(mask.sum(axis=-1, keepdims=True), 1.0)
        new_assignment = dist.argmin(axis=1)
        if np.array_equal(new_assignment, assignment):
            break
        assignment = new_assignment
        for anchor_idx in range(num_anchors):
            members = assignment == anchor_idx
            if not np.any(members):
                centers_arr[anchor_idx] = traj[int(np.argmax(dist.min(axis=1)))]
                continue
            member_traj = traj[members]
            member_mask = mask[members, :, None]
            centers_arr[anchor_idx] = np.sum(member_traj * member_mask, axis=0) / np.maximum(
                member_mask.sum(axis=0), 1.0
            )
    return centers_arr.astype(np.float32), assignment


def save_trajectory_anchors(path: Union[str, Path], anchors: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> None:
    """保存 K-means 轨迹锚点，并可在同目录写入描述坐标系/单位的 JSON 元数据。"""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(anchors, dtype=np.float32)
    np.save(output, arr)
    if metadata is not None:
        meta = dict(metadata)
        meta.setdefault("num_anchors", int(arr.shape[0]))
        meta.setdefault("num_waypoints", int(arr.shape[1]))
        meta.setdefault("action_dims", int(arr.shape[2]))
        output.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _load_trajectory_anchors(
    path: Optional[str],
    num_anchors: int,
    num_waypoints: int,
    action_dims: int,
) -> torch.Tensor:
    # 锚点必须来自与训练标签相同的数据分布、坐标系、单位和 waypoint 数量。
    if not path:
        raise ValueError(
            "Anchor diffusion requires an anchor .npy/.pt file. "
            "Set diffusion_anchor_path (single Agent) or the per-Agent anchor paths."
        )
    anchor_path = Path(path).expanduser()
    if not anchor_path.is_file():
        raise FileNotFoundError(f"Trajectory anchor file does not exist: {anchor_path}")
    if anchor_path.suffix.lower() == ".npy":
        anchors = torch.from_numpy(np.load(anchor_path)).float()
    else:
        obj = torch.load(anchor_path, map_location="cpu")
        if isinstance(obj, dict):
            obj = obj.get("anchors", obj.get("trajectory_anchors"))
        anchors = torch.as_tensor(obj, dtype=torch.float32)
    expected = (num_anchors, num_waypoints, action_dims)
    if tuple(anchors.shape) != expected:
        raise ValueError(f"Expected anchor shape {expected}, got {tuple(anchors.shape)} from {anchor_path}")
    return anchors


# ----------------------- 扩散时间编码与 DiT 组件 -----------------------

class SinusoidalTimestepEmbedding(nn.Module):
    """把离散扩散 timestep 编码为正余弦向量，供 DiT 条件调制使用。"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        timestep = timestep.float().view(-1)
        half = self.dim // 2
        frequency = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=timestep.device, dtype=torch.float32) / max(half - 1, 1)
        )
        angles = timestep[:, None] * frequency[None]
        embedding = torch.cat([angles.sin(), angles.cos()], dim=-1)
        if embedding.size(-1) < self.dim:
            embedding = F.pad(embedding, (0, self.dim - embedding.size(-1)))
        return embedding


class TruncatedDDIMScheduler:
    """基于 diffusers.DDIMScheduler 的锚点附近截断扩散调度器。

    对应论文与 DiffusionDrive 的 ``prediction_type="sample"``：模型直接预测
    去噪后的 x0，而不是噪声 epsilon。推理默认从 t=10 开始，只反推两步。
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
    ):
        self.num_train_timesteps = int(num_train_timesteps)
        # 与 DiffusionDrive 保持一致：scaled_linear beta，模型直接预测 x0/sample。
        # clip_sample=False，避免当前轨迹归一化尺度被 diffusers 再次硬裁剪。
        self.scheduler = DDIMScheduler(
            num_train_timesteps=self.num_train_timesteps,
            beta_schedule="scaled_linear",
            prediction_type="sample",
            clip_sample=False,
        )

    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """前向扩散 q(x_t|x_0)，给每条预定义锚点加入少量高斯噪声。"""
        return self.scheduler.add_noise(original_samples=x0, noise=noise, timesteps=timestep.long())

    def step(self, pred_x0: torch.Tensor, sample: torch.Tensor, timestep: int, next_timestep: int) -> torch.Tensor:
        """使用 diffusers 的确定性 DDIM 更新，由当前 x_t 和预测 x0 得到下一步。"""
        if next_timestep < 0:
            return pred_x0
        stride = max(1, int(timestep) - int(next_timestep))
        self.scheduler.num_inference_steps = max(1, self.num_train_timesteps // stride)
        return self.scheduler.step(
            model_output=pred_x0,
            timestep=int(timestep),
            sample=sample,
            eta=0.0,
        ).prev_sample

    @staticmethod
    def inference_timesteps(start_timestep: int, num_steps: int) -> List[int]:
        if num_steps < 1:
            raise ValueError("num_steps must be >= 1")
        return np.linspace(int(start_timestep), 0, num_steps).round().astype(np.int64).tolist()


class AnchorDiTBlock(nn.Module):
    """使用 AdaLN 注入 LLM 动作状态与扩散时间条件的 DiT block。

    当前模型没有 DiffusionDrive 的 BEV feature map，因此不迁移
    GridSampleCrossBEVAttention；所有锚点/路点 token 通过 self-attention 交互，
    LLM 的 ``h_act`` 与 timestep 则通过 AdaLN 调制每一层。
    """

    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm_attn = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.norm_ffn = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        ffn_dim = int(hidden_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    @staticmethod
    def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return x * (1.0 + scale[:, None]) + shift[:, None]

    def forward(self, tokens: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = self.modulation(condition).chunk(6, dim=-1)
        q = self._modulate(self.norm_attn(tokens), shift_a, scale_a)
        attn_out = self.attn(q, q, q, need_weights=False)[0]
        tokens = tokens + gate_a[:, None] * attn_out
        ffn_in = self._modulate(self.norm_ffn(tokens), shift_f, scale_f)
        return tokens + gate_f[:, None] * self.ffn(ffn_in)


# ----------------------- 扩散跟踪损失与动作模型 -----------------------

def anchor_diffusion_tracking_loss(
    candidate_trajectories: torch.Tensor,
    candidate_logits: torch.Tensor,
    anchors: torch.Tensor,
    target: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    score_loss_weight: float = 100.0,
    score_loss_reduction: str = "mean",
) -> Dict[str, torch.Tensor]:
    """论文公式 (3)：最近锚候选做轨迹 MSE，所有候选做二分类 BCE。

    ``valid_mask`` 会同时用于最近锚分配和回归损失，避免无效 padding 路点决定正锚。
    theta 使用 wrapped angle error，防止 -pi/pi 边界产生虚假的大误差。
    默认对 M 个候选 BCE 取均值后乘 λ，避免候选数量再次线性放大梯度；
    ``score_loss_reduction="sum"`` 可恢复公式字面上的求和行为。
    """
    B, M, Nw, D = candidate_trajectories.shape
    if target.shape != (B, Nw, D):
        raise ValueError(f"target must have shape {(B, Nw, D)}, got {tuple(target.shape)}")
    mask = torch.ones(B, Nw, device=target.device, dtype=target.dtype) if valid_mask is None else valid_mask.to(
        device=target.device, dtype=target.dtype
    )
    mask = mask.clamp(0.0, 1.0)
    anchor_xy_error = torch.linalg.vector_norm(target[:, None, :, :2] - anchors[None, :, :, :2], dim=-1)
    anchor_distance = (anchor_xy_error * mask[:, None]).sum(dim=-1) / mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
    # 每个样本只将与 GT 最近的 anchor 标记为正模式，其余模式均为负。
    nearest_anchor = anchor_distance.argmin(dim=-1)

    gather_idx = nearest_anchor[:, None, None, None].expand(-1, 1, Nw, D)
    positive_prediction = candidate_trajectories.gather(1, gather_idx).squeeze(1)
    error = positive_prediction - target
    if D >= 3:
        wrapped_heading = torch.atan2(torch.sin(error[..., 2:3]), torch.cos(error[..., 2:3]))
        error = torch.cat([error[..., :2], wrapped_heading, error[..., 3:]], dim=-1)
    regression_loss = ((error.square().mean(dim=-1) * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)).mean()

    score_target = torch.zeros_like(candidate_logits)
    score_target.scatter_(1, nearest_anchor[:, None], 1.0)
    per_anchor_score_loss = F.binary_cross_entropy_with_logits(candidate_logits, score_target, reduction="none")
    if score_loss_reduction == "mean":
        # λ=100 已负责放大分类项；对候选取均值可避免 M=40 再额外放大梯度。
        score_loss = per_anchor_score_loss.mean()
    elif score_loss_reduction == "sum":
        # 保留论文公式字面上的 sum_i BCE，便于消融和兼容旧实验。
        score_loss = per_anchor_score_loss.sum(dim=-1).mean()
    else:
        raise ValueError(f"Unsupported score_loss_reduction={score_loss_reduction!r}; expected 'mean' or 'sum'.")
    total = regression_loss + float(score_loss_weight) * score_loss
    return {
        "loss": total,
        "regression_loss": regression_loss,
        "score_loss": score_loss,
        "nearest_anchor": nearest_anchor,
        "anchor_distance": anchor_distance.detach(),
    }


class AnchorDiffusionActionModel(nn.Module):
    """参考 TrackVLA 与 DiffusionDrive 实现的基于锚点的扩散动作头。

    锚点与输出都使用当前项目原有的局部轨迹实际单位。内部先按 anchor bank
    的量级归一化，再在锚点附近加噪并用截断 DDIM 去噪。最终选择分数最高的候选，
    因而仍兼容原规划头 ``(B, Nw, D)`` 的 waypoint 输出接口。

    主要张量：
    - anchors / candidate_trajectories: (B, M, Nw, D)
    - candidate_logits: (B, M)
    - condition: (B, D_llm)，即 LLM 在 ACT token 位置的隐藏状态
    """

    def __init__(
        self,
        condition_dim: int,
        anchors: torch.Tensor,
        hidden_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        num_train_timesteps: int = 1000,
        train_truncation_steps: int = 50,
        inference_start_timestep: int = 10,
        inference_steps: int = 2,
        score_loss_weight: float = 100.0,
        score_loss_reduction: str = "mean",
        deterministic_inference: bool = False,
    ):
        super().__init__()
        if anchors.dim() != 3:
            raise ValueError(f"anchors must have shape (M,Nw,D), got {tuple(anchors.shape)}")
        self.num_anchors, self.num_waypoints, self.action_dims = [int(v) for v in anchors.shape]
        self.hidden_dim = int(hidden_dim)
        self.train_truncation_steps = int(train_truncation_steps)
        self.inference_start_timestep = int(inference_start_timestep)
        self.inference_steps = int(inference_steps)
        self.score_loss_weight = float(score_loss_weight)
        self.score_loss_reduction = str(score_loss_reduction)
        self.deterministic_inference = bool(deterministic_inference)

        # 使用 buffer 保存锚点，使其随 checkpoint 保存和迁移设备，但不参与梯度更新。
        self.register_buffer("anchors", anchors.float(), persistent=True)
        # 不照搬 DiffusionDrive 针对汽车轨迹的硬编码 norm_odo；这里从当前 anchor
        # bank 自动得到每个动作维度的尺度，保证训练/推理使用同一归一化方式。
        action_scale = anchors.abs().amax(dim=(0, 1)).clamp_min(1e-3)
        self.register_buffer("action_scale", action_scale.view(1, 1, 1, -1), persistent=True)
        self.scheduler = TruncatedDDIMScheduler(num_train_timesteps=num_train_timesteps)
        self.condition_proj = nn.Sequential(nn.LayerNorm(condition_dim), nn.Linear(condition_dim, hidden_dim))
        self.time_embed = nn.Sequential(
            SinusoidalTimestepEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.trajectory_embed = nn.Linear(self.action_dims, hidden_dim)
        self.anchor_embed = nn.Embedding(self.num_anchors, hidden_dim)
        self.waypoint_embed = nn.Embedding(self.num_waypoints, hidden_dim)
        # 当前采用 base diffusion transformer 配置：12 层、768 隐藏维度、12 个注意力头。
        self.blocks = nn.ModuleList(
            [AnchorDiTBlock(hidden_dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout) for _ in range(depth)]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.trajectory_delta = nn.Linear(hidden_dim, self.action_dims)
        self.score_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))
        nn.init.zeros_(self.trajectory_delta.weight)
        nn.init.zeros_(self.trajectory_delta.bias)
        nn.init.zeros_(self.score_head[-1].weight)
        nn.init.zeros_(self.score_head[-1].bias)

    def _normalize(self, trajectory: torch.Tensor) -> torch.Tensor:
        return trajectory / self.action_scale.to(device=trajectory.device, dtype=trajectory.dtype)

    def _denormalize(self, trajectory: torch.Tensor) -> torch.Tensor:
        return trajectory * self.action_scale.to(device=trajectory.device, dtype=trajectory.dtype)

    def _decode(self, noisy_normalized: torch.Tensor, timestep: torch.Tensor, condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """一次 DiT 去噪预测，同时输出所有锚点模式的 x0 与分类分数。"""
        B, M, Nw, _ = noisy_normalized.shape
        tokens = self.trajectory_embed(noisy_normalized)
        anchor_ids = torch.arange(M, device=tokens.device)
        waypoint_ids = torch.arange(Nw, device=tokens.device)
        tokens = tokens + self.anchor_embed(anchor_ids)[None, :, None] + self.waypoint_embed(waypoint_ids)[None, None]
        # 将 M 条候选轨迹的 Nw 个路点展开为 token 序列，使不同模式也能互相注意。
        tokens = tokens.reshape(B, M * Nw, self.hidden_dim)
        cond = self.condition_proj(condition.float()) + self.time_embed(timestep)
        for block in self.blocks:
            tokens = block(tokens, cond)
        tokens = self.final_norm(tokens).view(B, M, Nw, self.hidden_dim)
        pred_x0 = noisy_normalized + self.trajectory_delta(tokens)
        logits = self.score_head(tokens.mean(dim=2)).squeeze(-1)
        return pred_x0, logits

    @staticmethod
    def _gather_top_candidate(candidates: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        B, _, Nw, D = candidates.shape
        best = logits.argmax(dim=-1)
        return candidates.gather(1, best[:, None, None, None].expand(-1, 1, Nw, D)).squeeze(1)

    def forward(
        self,
        condition: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B = condition.size(0)
        anchors = self.anchors[None].expand(B, -1, -1, -1).to(device=condition.device, dtype=torch.float32)
        anchors_normalized = self._normalize(anchors)
        if self.training:
            # 论文训练阶段只从总计 1000 步中的前 50 步采样，因此噪声始终靠近锚点。
            timestep = torch.randint(
                0,
                min(self.train_truncation_steps, self.scheduler.num_train_timesteps),
                (B,),
                device=condition.device,
            )
            diffusion_noise = torch.randn_like(anchors_normalized) if noise is None else noise.to(anchors_normalized)
            sample = self.scheduler.add_noise(anchors_normalized, diffusion_noise, timestep)
            pred_x0, logits = self._decode(sample, timestep, condition)
        else:
            # 推理阶段从锚点的 t=10 加噪状态开始，默认只做两次 DDIM 去噪。
            diffusion_noise = (
                torch.zeros_like(anchors_normalized)
                if self.deterministic_inference
                else (torch.randn_like(anchors_normalized) if noise is None else noise.to(anchors_normalized))
            )
            start_t = min(self.inference_start_timestep, self.scheduler.num_train_timesteps - 1)
            start = torch.full((B,), start_t, device=condition.device, dtype=torch.long)
            sample = self.scheduler.add_noise(anchors_normalized, diffusion_noise, start)
            timesteps = self.scheduler.inference_timesteps(start_t, self.inference_steps)
            pred_x0 = sample
            logits = torch.zeros(B, self.num_anchors, device=condition.device)
            for idx, timestep_value in enumerate(timesteps):
                timestep = torch.full((B,), int(timestep_value), device=condition.device, dtype=torch.long)
                pred_x0, logits = self._decode(sample, timestep, condition)
                next_timestep = timesteps[idx + 1] if idx + 1 < len(timesteps) else -1
                sample = self.scheduler.step(pred_x0, sample, int(timestep_value), int(next_timestep))

        candidates = self._denormalize(pred_x0)
        if self.action_dims >= 3:
            wrapped_heading = torch.atan2(
                torch.sin(candidates[..., 2:3]),
                torch.cos(candidates[..., 2:3]),
            )
            candidates = torch.cat([candidates[..., :2], wrapped_heading, candidates[..., 3:]], dim=-1)
        # 论文推理规则：选取得分最高的候选轨迹作为最终动作输出。
        trajectory = self._gather_top_candidate(candidates, logits)
        output: Dict[str, torch.Tensor] = {
            "trajectory": trajectory,
            "candidate_trajectories": candidates,
            "candidate_logits": logits,
            "candidate_scores": logits.sigmoid(),
            "anchors": anchors,
        }
        if target is not None:
            output.update(
                anchor_diffusion_tracking_loss(
                    candidates,
                    logits,
                    anchors[0],
                    target.to(device=candidates.device, dtype=candidates.dtype),
                    valid_mask=valid_mask,
                    score_loss_weight=self.score_loss_weight,
                    score_loss_reduction=self.score_loss_reduction,
                )
            )
        return output


# ----------------------- 单 Agent模型配置与主干 -----------------------

@dataclass
class ModelConfig:
    """原始单 Agent OpenTrackVLA 配置。

    数据流相关字段：
    - n_waypoints 控制规划头输出多少个未来路点。
    -  max_time 控制 TVI 时间编码的词表大小，需覆盖 episode 内最大帧数。
    - use_angle_tvi 控制是否把 yaw 信息作为额外 TVI token 注入视觉序列。
    - alpha_xy 用于把模型 normalized XY 输出还原到实际距离尺度，yaw 不缩放。
    """
    llm_name: str = "Qwen/Qwen3-0.6B"
    freeze_llm: bool = True
    n_waypoints: int = 8
    max_time: int = 4096
    beta_nav: float = 10.0
    use_angle_tvi: bool = False
    # Action/target configuration
    use_tanh_actions: bool = True
    alpha_xy: Optional[float] = None
    # 可选的基于锚点扩散动作头。锚点必须使用当前项目的局部轨迹实际单位，
    # shape 固定为 (M, n_waypoints, 3)。
    use_anchor_diffusion: bool = False
    diffusion_anchor_path: Optional[str] = None
    diffusion_num_anchors: int = 40
    diffusion_hidden_dim: int = 768
    diffusion_depth: int = 12
    diffusion_num_heads: int = 12
    diffusion_mlp_ratio: float = 4.0
    diffusion_dropout: float = 0.0
    diffusion_num_train_timesteps: int = 1000
    diffusion_train_truncation_steps: int = 50
    diffusion_inference_start_timestep: int = 10
    diffusion_inference_steps: int = 2
    diffusion_score_loss_weight: float = 100.0
    diffusion_score_loss_reduction: str = "mean"
    diffusion_deterministic_inference: bool = False


class OpenTrackVLA(nn.Module):
    """原始单 Agent OpenTrackVLA。

    单 Agent 数据流：
    1. Dataset 读取 history 个历史帧 coarse token 和当前帧 fine token。
    2. projector 把视觉 token 从视觉特征维度 C 映射到 LLM hidden size D。
    3. TVI 在视觉 token 中注入时间信息，可选注入 yaw/角度信息。
    4. LLM 输入序列为 [文本指令, 历史视觉, 当前视觉, ACT查询token]。
    5. 取最后 ACT token 的隐藏状态，送入 planner head 输出未来路点。

    这个类保持原单 Agent 能力；双 Agent 逻辑在 MultiAgentOpenTrackVLA 中单独实现。
    """

    def __init__(self, cfg: ModelConfig, vision_feat_dim: int):
        """初始化单 Agent 模型组件。"""
        super().__init__()
        self.cfg = cfg
        # Load LLM - try ModelScope first if available, fallback to HuggingFace
        try:
            from modelscope import AutoModel as MSAutoModel
            from modelscope import AutoTokenizer as MSAutoTokenizer
            print(f"[LLM] Loading from ModelScope: {cfg.llm_name}")
            self.llm = MSAutoModel.from_pretrained(
                cfg.llm_name,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None
            )
            self.tokenizer = MSAutoTokenizer.from_pretrained(cfg.llm_name)
        except Exception as e:
            print(f"[LLM] ModelScope failed ({e}), using HuggingFace")
            self.llm = AutoModel.from_pretrained(
                cfg.llm_name,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None
            )
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.llm_name)
        self.llm.requires_grad_(not cfg.freeze_llm)
        self.D = self.llm.config.hidden_size
        self.proj = CrossModalityProjector(vision_feat_dim, self.D)
        self.proj.requires_grad_(True)
        self.tvi = TVIEmbedder(self.D, max_time=cfg.max_time)
        self.act_token = nn.Parameter(torch.zeros(1, 1, self.D))
        nn.init.normal_(self.act_token, std=0.02)
        action_dims = 3
        self.action_dims = action_dims
        if cfg.use_anchor_diffusion:
            # 扩散规划头直接从 LLM 的 ACT hidden state 条件化生成多模态轨迹；
            # 默认不开启，因此旧 checkpoint 和原 MLP 行为保持兼容。
            anchors = _load_trajectory_anchors(
                cfg.diffusion_anchor_path,
                cfg.diffusion_num_anchors,
                cfg.n_waypoints,
                action_dims,
            )
            self.planner = AnchorDiffusionActionModel(
                condition_dim=self.D,
                anchors=anchors,
                hidden_dim=cfg.diffusion_hidden_dim,
                depth=cfg.diffusion_depth,
                num_heads=cfg.diffusion_num_heads,
                mlp_ratio=cfg.diffusion_mlp_ratio,
                dropout=cfg.diffusion_dropout,
                num_train_timesteps=cfg.diffusion_num_train_timesteps,
                train_truncation_steps=cfg.diffusion_train_truncation_steps,
                inference_start_timestep=cfg.diffusion_inference_start_timestep,
                inference_steps=cfg.diffusion_inference_steps,
                score_loss_weight=cfg.diffusion_score_loss_weight,
                score_loss_reduction=cfg.diffusion_score_loss_reduction,
                deterministic_inference=cfg.diffusion_deterministic_inference,
            )
        else:
            self.planner = PlannerHead3L(self.D, cfg.n_waypoints, action_dims, use_tanh=cfg.use_tanh_actions)
        self.planner.requires_grad_(True)
        if not cfg.use_angle_tvi:
            for p in self.tvi.angle_proj.parameters():
                p.requires_grad = False

        if cfg.alpha_xy is not None:
            vec = [1.0] * action_dims
            if action_dims >= 2:
                vec[0] = cfg.alpha_xy
                vec[1] = cfg.alpha_xy
            alpha = torch.tensor(vec, dtype=torch.float32).view(1, 1, -1)
        else:
            alpha = torch.tensor((1.0, 1.0, 1.0), dtype=torch.float32).view(1, 1, -1)
        self.register_buffer("alpha_task", alpha)

    def _embed_text(self, instructions: List[str], device):
        """把批量文本指令转成 LLM 输入 embedding。

        输出：
        - emb: 文本 token embedding，shape (B, Ltxt, D)
        - attention_mask: 文本 attention mask，shape (B, Ltxt)
        """
        tok = self.tokenizer(instructions, return_tensors='pt', padding=True, truncation=True, max_length=128)
        tok = {k: v.to(device) for k, v in tok.items()}
        emb = self.llm.get_input_embeddings()(tok['input_ids'])
        return emb, tok['attention_mask']

    def _interleave_tvi(self, tokens: torch.Tensor, t_idx: torch.Tensor, kind_id: int,
                        yaw_per_frame: Optional[torch.Tensor] = None, use_angle: bool = False) -> torch.Tensor:
        """在单 Agent 视觉 token 中插入时间/角度 marker。

        输入：
        - tokens: (B, N, D)，已经投影到 LLM 维度的视觉 token。
        - t_idx:  (B, N)，每个 token 属于哪一帧。

        输出：
        - (B, N + marker数, D)，每帧视觉块前会加入时间 marker；
          use_angle=True 时还会加入角度 marker。
        """
        B, N, D = tokens.shape
        out_list = []
        for b in range(B):
            tb = t_idx[b]
            xb = tokens[b]
            items = []
            i = 0
            fcount = 0
            while i < N:
                tcur = int(tb[i].item())
                j = i + 1
                while j < N and int(tb[j].item()) == tcur:
                    j += 1
                time_tok = self.tvi.make_time_token(tcur, kind_id, device=xb.device).unsqueeze(0)
                items.append(time_tok)
                if use_angle:
                    theta = 0.0
                    if yaw_per_frame is not None and fcount < yaw_per_frame.size(1):
                        theta = float(yaw_per_frame[b, fcount].item())
                    angle_tok = self.tvi.make_angle_token(theta, kind_id, device=xb.device).unsqueeze(0)
                    items.append(angle_tok)
                items.append(xb[i:j])
                i = j
                fcount += 1
            out_list.append(torch.cat(items, dim=0))
        return torch.stack(out_list, dim=0)

    def forward(self,
                coarse_tokens, coarse_tidx,
                fine_tokens, fine_tidx,
                instructions,
                yaw_hist: Optional[torch.Tensor] = None,
                yaw_curr: Optional[torch.Tensor] = None,
                bbox_feat: Optional[torch.Tensor] = None,
                target_waypoints: Optional[torch.Tensor] = None,
                valid_mask: Optional[torch.Tensor] = None,
                return_action_details: bool = False):
        """单 Agent 前向传播。

        输入：
        - coarse_tokens/coarse_tidx: 历史帧粗 token 与帧编号。
        - fine_tokens/fine_tidx: 当前帧细 token 与帧编号。
        - instructions: 文本任务指令。

        输出：
        - tau_pred: (B, n_waypoints, 3)，未来局部路点 [x, y, theta]。

        bbox_feat 参数保留用于兼容调用接口；单 Agent 当前未使用它。
        """
        device = next(self.parameters()).device
        B = coarse_tokens.size(0)
        vis_c = self.proj(coarse_tokens.to(device))
        vis_f = self.proj(fine_tokens.to(device))
        vis_c = self._interleave_tvi(
            vis_c, coarse_tidx.to(device), kind_id=0,
            yaw_per_frame=yaw_hist, use_angle=self.cfg.use_angle_tvi
        )
        vis_f = self._interleave_tvi(
            vis_f, fine_tidx.to(device), kind_id=1,
            yaw_per_frame=yaw_curr, use_angle=self.cfg.use_angle_tvi
        )
        txt_emb, txt_mask = self._embed_text(instructions, device)
        extra = []        
        act = self.act_token.expand(B, 1, -1)
        pieces = [txt_emb] + ([extra[0]] if extra else []) + [vis_c, vis_f, act]
        seq = torch.cat(pieces, dim=1).to(self.llm.dtype)
        extra_len = (extra[0].size(1) if extra else 0)
        attn = torch.cat([
            txt_mask,
            torch.ones(B, extra_len + vis_c.size(1) + vis_f.size(1) + 1, dtype=torch.long, device=device)
        ], dim=1)
        out = self.llm(inputs_embeds=seq, attention_mask=attn, output_hidden_states=True, use_cache=False)
        h_act = out.last_hidden_state[:, -1, :]
        h_act = h_act.float()
        if self.cfg.use_anchor_diffusion:
            # target_waypoints/valid_mask 只在训练时传入，用于最近锚分配和公式 (3)
            # 的 tracking loss；推理时返回 top-1 候选轨迹。
            action_output = self.planner(h_act, target=target_waypoints, valid_mask=valid_mask)
            return action_output if return_action_details else action_output["trajectory"]
        a_hat = self.planner(h_act)
        tau_pred = a_hat * self.alpha_task
        if return_action_details:
            return {"trajectory": tau_pred}
        return tau_pred


@dataclass
# ----------------------- 双 Agent模型配置与组件 -----------------------

class MultiAgentModelConfig:
    """双 Agent 模型配置。

    数据流相关字段：
    - num_agents 固定为 2：默认 agent1=无人机，agent2=机器狗。
    - max_views/num_kinds 控制 TVI/agent/type embedding 的词表大小。
    - insert_time_tokens=True 时，会在每一帧视觉 token 前插入时间 marker。
    - bbox_delta_scale 控制 grounding head 对输入 bbox 的修正幅度。
    """

    llm_name: str = "Qwen/Qwen3-0.6B"
    freeze_llm: bool = True
    n_waypoints: int = 8
    action_dims: int = 3
    max_time: int = 4096
    max_views: int = 2
    num_agents: int = 2
    num_kinds: int = 5
    use_angle_tvi: bool = False
    insert_time_tokens: bool = True
    use_tanh_actions: bool = True
    alpha_xy: Optional[float] = 2.0
    bbox_delta_scale: float = 0.25
    return_token_logits: bool = False
    text_max_length: int = 128
    # 双 Agent 使用独立 anchor bank，因为无人机和机器狗的动力学与轨迹分布不同。
    use_anchor_diffusion: bool = False
    diffusion_anchor_path: Optional[str] = None
    diffusion_agent1_anchor_path: Optional[str] = None
    diffusion_agent2_anchor_path: Optional[str] = None
    diffusion_num_anchors: int = 40
    diffusion_hidden_dim: int = 768
    diffusion_depth: int = 12
    diffusion_num_heads: int = 12
    diffusion_mlp_ratio: float = 4.0
    diffusion_dropout: float = 0.0
    diffusion_num_train_timesteps: int = 1000
    diffusion_train_truncation_steps: int = 50
    diffusion_inference_start_timestep: int = 10
    diffusion_inference_steps: int = 2
    diffusion_score_loss_weight: float = 100.0
    diffusion_score_loss_reduction: str = "mean"
    diffusion_deterministic_inference: bool = False


class MultiAgentTVIEmbedder(nn.Module):
    """双 Agent 版本的 TVI/类型编码器。

    处理逻辑：
    1. time_emb 表示帧序号，让 LLM 区分历史帧和当前帧的时间位置。
    2. view_emb 表示视角/传感器来源，这里通常 agent0=无人机视角，agent1=机器狗视角。
    3. kind_emb 表示 token 类型，例如历史视觉、当前视觉、bbox、动作查询、grounding 查询。
    4. agent_emb 表示实体身份，避免两个 Agent 的视觉 token 混在同一序列后失去归属。
    """

    def __init__(
        self,
        d_model: int,
        max_time: int = 4096,
        max_views: int = 2,
        num_agents: int = 2,
        num_kinds: int = 5,
    ):
        super().__init__()
        self.time_emb = nn.Embedding(max_time, d_model)
        self.view_emb = nn.Embedding(max_views, d_model)
        self.kind_emb = nn.Embedding(num_kinds, d_model)
        self.agent_emb = nn.Embedding(num_agents, d_model)
        self.angle_proj = nn.Linear(2, d_model)
        self.bbox_proj = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def type_embedding(self, kind_id: int, agent_id: int, view_id: int, device: torch.device) -> torch.Tensor:
        """生成“类型 + Agent 身份 + 视角”的公共 embedding。

        输出 shape: (D,)
        后续会广播加到一整段视觉 token 或查询 token 上。
        """
        kind = torch.tensor(kind_id, dtype=torch.long, device=device)
        agent = torch.tensor(agent_id, dtype=torch.long, device=device)
        view = torch.tensor(view_id, dtype=torch.long, device=device)
        return self.kind_emb(kind) + self.agent_emb(agent) + self.view_emb(view)

    def add_visual_tvi(
        self,
        tokens: torch.Tensor,
        t_idx: torch.Tensor,
        kind_id: int,
        agent_id: int,
        view_id: int,
    ) -> torch.Tensor:
        """给视觉 token 逐 token 加时间/类型/Agent 编码。

        输入：
        - tokens: (B, N, D)，已经投影到 LLM hidden size 的视觉 token。
        - t_idx:  (B, N)，每个视觉 token 属于第几帧。

        输出：
        - (B, N, D)，shape 不变，但每个 token 已带有时间、来源 Agent、视角和 token 类型信息。
        """
        device = tokens.device
        # Do not use clamp_ here. agent1/agent2 t_idx are often views from the same
        # stacked tensor, and in-place mutation breaks autograd version checks.
        t_idx = t_idx.to(device=device, dtype=torch.long).clamp(0, self.time_emb.num_embeddings - 1)
        type_emb = self.type_embedding(kind_id, agent_id, view_id, device)
        return tokens + self.time_emb(t_idx) + type_emb.view(1, 1, -1)

    def make_marker(
        self,
        t_scalar: int,
        kind_id: int,                           
        agent_id: int,
        view_id: int,
        device: torch.device,
    ) -> torch.Tensor:
        """构造插入序列中的时间 marker token。

        这个 token 会放在每帧视觉 token 块之前，显式告诉 LLM：
        “接下来这一段视觉特征来自哪个 Agent 的第 t 帧”。
        """
        t = int(max(0, min(t_scalar, self.time_emb.num_embeddings - 1)))
        return self.time_emb.weight[t] + self.type_embedding(kind_id, agent_id, view_id, device)

    def make_angle_marker(
        self,
        theta: float,
        kind_id: int,
        agent_id: int,
        view_id: int,
        device: torch.device,
    ) -> torch.Tensor:
        """构造可选 yaw/朝向 marker。

        数据流：theta -> [sin(theta), cos(theta)] -> Linear -> 加 Agent/type/view embedding。
        只有 cfg.use_angle_tvi=True 时，模型才会把该 token 插入视觉序列。
        """
        theta = (theta + math.pi) % (2 * math.pi) - math.pi
        sincos = torch.tensor([math.sin(theta), math.cos(theta)], dtype=self.angle_proj.weight.dtype, device=device)
        angle = F.linear(sincos, self.angle_proj.weight, self.angle_proj.bias)
        return angle + self.type_embedding(kind_id, agent_id, view_id, device)

    def make_bbox_token(self, bbox: torch.Tensor, agent_id: int, view_id: int) -> torch.Tensor:
        """把归一化 bbox 编成一个 LLM token。

        输入 bbox:
        - (B, 4) 或 (4,)，格式为 cx, cy, w, h，取值范围 [0, 1]。

        输出：
        - (B, 1, D)，作为当前 Agent 视觉流末尾的 bbox 先验 token。
        """
        if bbox.dim() == 1:
            bbox = bbox.unsqueeze(0)
        bbox = bbox.to(dtype=self.bbox_proj[1].weight.dtype)
        emb = self.bbox_proj(bbox)
        if emb.dim() == 2:
            emb = emb.unsqueeze(1)
        type_emb = self.type_embedding(KIND_BBOX, agent_id, view_id, emb.device)
        return emb + type_emb.view(1, 1, -1)

    def make_query_token(self, base: torch.Tensor, kind_id: int, agent_id: int, view_id: int) -> torch.Tensor:
        """给可学习查询 token 加类型/Agent/视角标识。

        ACT 查询 token 用于规划头；GND 查询 token 用于 bbox/visibility 辅助头。
        """
        type_emb = self.type_embedding(kind_id, agent_id, view_id, base.device)
        return base + type_emb.view(1, 1, -1)


class GroundingHead(nn.Module):
    """逐 Agent 目标 grounding 辅助头。

    每个 Agent 有自己的 GND hidden state，但两个 Agent 共享输出头参数。这样
    GND 查询分别绑定各自坐标系，同时仍能从完整双视角 LLM 上下文读取信息。

    输入：
    - h_gnd: 两个 GND 查询 token 的隐藏状态，shape (B, 2, D)。
    - bbox_feat: bbox 先验，shape (B, 2, 4)，可选。
    - bbox_valid_mask: 每个 Agent 是否有可用 bbox 先验，shape (B, 2)。
    - visual_tokens: 可选视觉 token 池，用于 token-level 对齐打分。

    输出：
    - refined_bbox: 两个 Agent 各自坐标系下的 bbox，shape (B, 2, 4)。
    - visible_logits/visible_score: 两个 Agent 视角下目标是否可见。
    - token_logits: 可选 token grounding 分数，便于后续做更细监督。
    """

    def __init__(self, d_model: int, num_agents: int = 2, bbox_delta_scale: float = 0.25):
        super().__init__()
        self.num_agents = num_agents
        self.bbox_delta_scale = bbox_delta_scale
        hid = d_model * 2
        self.bbox_absolute = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hid),
            nn.GELU(),
            nn.Linear(hid, 4),
        )
        self.bbox_residual = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hid),
            nn.GELU(),
            nn.Linear(hid, 4),
        )
        self.visibility = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hid),
            nn.GELU(),
            nn.Linear(hid, 1),
        )
        self.per_agent_token_query = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )

    def forward(
        self,
        h_gnd: torch.Tensor,
        bbox_feat: Optional[torch.Tensor] = None,
        bbox_valid_mask: Optional[torch.Tensor] = None,
        visual_tokens: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """根据两个 GND 隐藏状态分别预测 bbox 和可见性。

        处理逻辑：
        1. 有有效 bbox prior 的 Agent 使用 residual head 做有界修正。
        2. prior 被丢弃或缺失的 Agent 使用 absolute head 从视觉直接检测。
        3. visibility 和 token grounding 都由对应 Agent 的 GND state 预测。
        """
        if h_gnd.dim() != 3 or h_gnd.size(1) != self.num_agents:
            raise ValueError(
                f"h_gnd must have shape (B, {self.num_agents}, D), got {tuple(h_gnd.shape)}"
            )
        B = h_gnd.size(0)
        absolute_bbox = torch.sigmoid(self.bbox_absolute(h_gnd))
        delta = torch.tanh(self.bbox_residual(h_gnd)) * self.bbox_delta_scale

        if bbox_feat is None:
            bbox_prior = torch.zeros_like(absolute_bbox)
            valid_prior = torch.zeros(B, self.num_agents, dtype=torch.bool, device=h_gnd.device)
        else:
            bbox_prior = bbox_feat.to(device=h_gnd.device, dtype=delta.dtype)
            if bbox_valid_mask is None:
                valid_prior = torch.ones(B, self.num_agents, dtype=torch.bool, device=h_gnd.device)
            else:
                valid_prior = bbox_valid_mask.to(device=h_gnd.device, dtype=torch.bool)
                if valid_prior.shape != (B, self.num_agents):
                    raise ValueError(
                        f"bbox_valid_mask must have shape (B, {self.num_agents}), got {tuple(valid_prior.shape)}"
                    )
        residual_bbox = (bbox_prior + delta).clamp(0.0, 1.0)
        refined_bbox = torch.where(valid_prior.unsqueeze(-1), residual_bbox, absolute_bbox)

        visible_logits = self.visibility(h_gnd).squeeze(-1)
        visible_score = torch.sigmoid(visible_logits)
        token_logits = None
        if visual_tokens is not None:
            queries = self.per_agent_token_query(h_gnd)
            token_logits = torch.einsum("bad,band->ban", queries.float(), visual_tokens.float())
            token_logits = token_logits / math.sqrt(max(1, queries.size(-1)))

        return {
            "refined_bbox": refined_bbox,
            "absolute_bbox": absolute_bbox,
            "bbox_prior_mask": valid_prior,
            "visible_logits": visible_logits,
            "visible_score": visible_score,
            "token_logits": token_logits,
        }


# ----------------------- 双 Agent Anchor Diffusion 模型主干 -----------------------

class MultiAgentOpenTrackVLA(nn.Module):
    """双 Agent OpenTrackVLA 主模型。

    典型输入 shape:
      coarse_tokens: (B, 2, Nc, C)
      coarse_tidx:   (B, 2, Nc)
      fine_tokens:   (B, 2, Nf, C)
      fine_tidx:     (B, 2, Nf)
      bbox_feat:     (B, 2, 4)

    核心数据流：
    1. 每个 Agent 的历史粗 token / 当前细 token 先通过 projector 映射到 LLM hidden size。
    2. TVI 给视觉 token 加上时间、Agent、视角、类型编码，并可插入帧级 marker。
    3. 每个 Agent 的 bbox 被编码成一个 bbox token，拼到对应视觉流末尾。
    4. 序列按 [文本, agent1视觉, agent2视觉, ACT1, ACT2, GND1, GND2] 喂给 LLM，
       使因果 LLM 的两个 ACT 查询都能读取完整双视角上下文。
    5. ACT1/ACT2 位置的隐藏状态分别进入两个 planner head，输出两套路点。
    6. GND1/GND2 分别进入共享逐 Agent grounding head，输出各自 bbox/visibility。
    """

    def __init__(self, cfg: MultiAgentModelConfig, vision_feat_dim: int):
        """初始化模型模块。

        模块组成：
        - llm/tokenizer: 文本和多模态序列的主干。
        - proj: 把 DINO/SigLIP 拼接后的视觉维度 C 投影到 LLM 维度 D。
        - tvi: 注入时间、Agent、视角、token 类型和 bbox 信息。
        - planner_agent1/2: 两个独立规划头，避免无人机/机器狗动作分布被强行共享。
        - grounding_head: 逐 Agent absolute detection / bbox refinement / visibility 辅助头。
        """
        super().__init__()
        if cfg.num_agents != 2:
            raise ValueError("MultiAgentOpenTrackVLA currently expects exactly two agents.")
        self.cfg = cfg
        rank = int(os.environ.get("RANK", "0"))
        t0 = time.time()
        print(f"[MODEL][rank {rank}] loading LLM weights: {cfg.llm_name}", flush=True)
        try:
            from modelscope import AutoModel as MSAutoModel
            from modelscope import AutoTokenizer as MSAutoTokenizer

            self.llm = MSAutoModel.from_pretrained(
                cfg.llm_name,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
            )
            self.tokenizer = MSAutoTokenizer.from_pretrained(cfg.llm_name)
        except Exception:
            load_kwargs = {"dtype": torch.bfloat16} if torch.cuda.is_available() else {}
            self.llm = AutoModel.from_pretrained(cfg.llm_name, **load_kwargs)
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.llm_name)
        print(f"[MODEL][rank {rank}] LLM/tokenizer loaded in {time.time() - t0:.1f}s", flush=True)

        stage_start = time.time()
        self.llm.requires_grad_(not cfg.freeze_llm)
        self.llm_dtype = next(self.llm.parameters()).dtype
        self.D = int(self.llm.config.hidden_size)
        print(
            f"[MODEL][rank {rank}] LLM freeze={cfg.freeze_llm} hidden={self.D} "
            f"configured in {time.time() - stage_start:.1f}s",
            flush=True,
        )

        stage_start = time.time()
        self.proj = CrossModalityProjector(vision_feat_dim, self.D)
        self.tvi = MultiAgentTVIEmbedder(
            self.D,
            max_time=cfg.max_time,
            max_views=cfg.max_views,
            num_agents=cfg.num_agents,
            num_kinds=cfg.num_kinds,
        )

        self.act_token_1 = nn.Parameter(torch.zeros(1, 1, self.D))
        self.act_token_2 = nn.Parameter(torch.zeros(1, 1, self.D))
        self.gnd_token_1 = nn.Parameter(torch.zeros(1, 1, self.D))
        self.gnd_token_2 = nn.Parameter(torch.zeros(1, 1, self.D))
        nn.init.normal_(self.act_token_1, std=0.02)
        nn.init.normal_(self.act_token_2, std=0.02)
        nn.init.normal_(self.gnd_token_1, std=0.02)
        nn.init.normal_(self.gnd_token_2, std=0.02)
        print(
            f"[MODEL][rank {rank}] projector/TVI/query tokens built in {time.time() - stage_start:.1f}s",
            flush=True,
        )

        if cfg.use_anchor_diffusion:
            stage_start = time.time()
            # 如果没有分别指定，则允许两个 Agent 退回共享 diffusion_anchor_path；
            # 实际训练时更推荐分别聚类并提供两套 anchor 文件。
            agent1_anchor_path = cfg.diffusion_agent1_anchor_path or cfg.diffusion_anchor_path
            agent2_anchor_path = cfg.diffusion_agent2_anchor_path or cfg.diffusion_anchor_path
            agent1_anchors = _load_trajectory_anchors(
                agent1_anchor_path,
                cfg.diffusion_num_anchors,
                cfg.n_waypoints,
                cfg.action_dims,
            )
            agent2_anchors = _load_trajectory_anchors(
                agent2_anchor_path,
                cfg.diffusion_num_anchors,
                cfg.n_waypoints,
                cfg.action_dims,
            )
            diffusion_kwargs = dict(
                condition_dim=self.D,
                hidden_dim=cfg.diffusion_hidden_dim,
                depth=cfg.diffusion_depth,
                num_heads=cfg.diffusion_num_heads,
                mlp_ratio=cfg.diffusion_mlp_ratio,
                dropout=cfg.diffusion_dropout,
                num_train_timesteps=cfg.diffusion_num_train_timesteps,
                train_truncation_steps=cfg.diffusion_train_truncation_steps,
                inference_start_timestep=cfg.diffusion_inference_start_timestep,
                inference_steps=cfg.diffusion_inference_steps,
                score_loss_weight=cfg.diffusion_score_loss_weight,
                score_loss_reduction=cfg.diffusion_score_loss_reduction,
                deterministic_inference=cfg.diffusion_deterministic_inference,
            )
            self.planner_agent1 = AnchorDiffusionActionModel(anchors=agent1_anchors, **diffusion_kwargs)
            self.planner_agent2 = AnchorDiffusionActionModel(anchors=agent2_anchors, **diffusion_kwargs)
            print(
                f"[MODEL][rank {rank}] two anchor-diffusion planners built in {time.time() - stage_start:.1f}s",
                flush=True,
            )
        else:
            self.planner_agent1 = PlannerHead3L(self.D, cfg.n_waypoints, cfg.action_dims, cfg.use_tanh_actions)
            self.planner_agent2 = PlannerHead3L(self.D, cfg.n_waypoints, cfg.action_dims, cfg.use_tanh_actions)
        self.grounding_head = GroundingHead(self.D, cfg.num_agents, cfg.bbox_delta_scale)
        if not cfg.return_token_logits:
            # 当前训练不计算 token-level grounding loss；冻结未使用分支，避免 DDP
            # 每次 backward 搜索 unused parameters，也避免该分支被误计入可训练参数。
            self.grounding_head.per_agent_token_query.requires_grad_(False)

        if not cfg.use_angle_tvi:
            for p in self.tvi.angle_proj.parameters():
                p.requires_grad = False

        alpha = torch.ones(1, 1, cfg.action_dims, dtype=torch.float32)
        if cfg.alpha_xy is not None and cfg.action_dims >= 2:
            alpha[..., 0:2] = float(cfg.alpha_xy)
        self.register_buffer("alpha_task", alpha)
        print(f"[MODEL][rank {rank}] multi-agent model construction complete", flush=True)

    def _embed_text(self, instructions: List[str], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """把自然语言指令转成 LLM 输入 embedding。

        输入: List[str]，batch 内每条样本一条 instruction。
        输出:
        - txt_emb:  (B, Ltxt, D)
        - txt_mask: (B, Ltxt)，用于 attention_mask。
        """
        tok = self.tokenizer(
            instructions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.text_max_length,
        )
        tok = {k: v.to(device) for k, v in tok.items()}
        emb = self.llm.get_input_embeddings()(tok["input_ids"])
        return emb, tok["attention_mask"]

    def _normalize_bbox(self, bbox: Optional[torch.Tensor], B: int, device: torch.device) -> torch.Tensor:
        """检查并归一化 bbox 输入。

        支持：
        - None：返回全 0 bbox。
        - (B, 4)：单 bbox 会扩展给两个 Agent。
        - (B, 2, 4)：每个 Agent 一套 bbox。

        输出固定为 (B, 2, 4)，并 clamp 到 [0, 1]。
        """
        if bbox is None:
            return torch.zeros(B, self.cfg.num_agents, 4, dtype=torch.float32, device=device)
        bbox = bbox.to(device=device, dtype=torch.float32)
        if bbox.dim() == 2:
            bbox = bbox.unsqueeze(1).expand(-1, self.cfg.num_agents, -1)
        if bbox.size(1) != self.cfg.num_agents or bbox.size(-1) != 4:
            raise ValueError(f"bbox_feat must have shape (B, 2, 4) or (B, 4), got {tuple(bbox.shape)}")
        if bbox.detach().amax() > 1.5:
            raise ValueError("bbox_feat should be normalized to [0, 1].")
        return bbox.clone().clamp(0.0, 1.0)

    def _interleave_markers(
        self,
        tokens: torch.Tensor,
        t_idx: torch.Tensor,
        kind_id: int,
        agent_id: int,
        view_id: int,
        yaw_per_frame: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """在视觉 token 序列中插入帧级 marker。

        输入 tokens 已经带有逐 token TVI，shape (B, N, D)。
        t_idx 中相同帧号的一段 token 会被视为同一帧块。

        输出示意：
        [time_marker(t0), frame0_tokens, time_marker(t1), frame1_tokens, ...]
        如果 use_angle_tvi=True，则每帧还会额外插入 angle_marker。
        """
        if not self.cfg.insert_time_tokens:
            return tokens
        B, N, _ = tokens.shape
        out_list = []
        for b in range(B):
            tb = t_idx[b]
            xb = tokens[b]
            items = []
            i = 0
            fcount = 0
            while i < N:
                tcur = int(tb[i].item())
                j = i + 1
                while j < N and int(tb[j].item()) == tcur:
                    j += 1
                items.append(self.tvi.make_marker(tcur, kind_id, agent_id, view_id, xb.device).unsqueeze(0))
                if self.cfg.use_angle_tvi:
                    theta = 0.0
                    if yaw_per_frame is not None and fcount < yaw_per_frame.size(1):
                        theta = float(yaw_per_frame[b, fcount].item())
                    items.append(self.tvi.make_angle_marker(theta, kind_id, agent_id, view_id, xb.device).unsqueeze(0))
                items.append(xb[i:j])
                i = j
                fcount += 1
            out_list.append(torch.cat(items, dim=0))
        return torch.stack(out_list, dim=0)

    def _encode_agent_stream(
        self,
        coarse_tokens: torch.Tensor,
        coarse_tidx: torch.Tensor,
        fine_tokens: torch.Tensor,
        fine_tidx: torch.Tensor,
        bbox_feat: torch.Tensor,
        agent_id: int,
        yaw_hist: Optional[torch.Tensor] = None,
        yaw_curr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """编码单个 Agent 的视觉流。

        数据流：
        coarse/fine cache token (C维)
        -> CrossModalityProjector 投影到 D维
        -> add_visual_tvi 加逐 token 时间/Agent/type 信息
        -> _interleave_markers 插入帧级 marker
        -> make_bbox_token 添加当前 bbox token

        返回：
        - agent_seq: 喂给 LLM 的该 Agent 完整视觉序列。
        - visual_for_grounding: 不含 marker/bbox 的视觉 token，用于可选 token grounding。
        """
        view_id = agent_id
        vis_c = self.proj(coarse_tokens)
        vis_f = self.proj(fine_tokens)
        vis_c = self.tvi.add_visual_tvi(vis_c, coarse_tidx, KIND_HISTORY, agent_id, view_id)
        vis_f = self.tvi.add_visual_tvi(vis_f, fine_tidx, KIND_CURRENT, agent_id, view_id)
        visual_for_grounding = torch.cat([vis_c, vis_f], dim=1)
        seq_c = self._interleave_markers(vis_c, coarse_tidx, KIND_HISTORY, agent_id, view_id, yaw_hist)
        seq_f = self._interleave_markers(vis_f, fine_tidx, KIND_CURRENT, agent_id, view_id, yaw_curr)
        bbox_tok = self.tvi.make_bbox_token(bbox_feat, agent_id, view_id)
        return torch.cat([seq_c, seq_f, bbox_tok], dim=1), visual_for_grounding

    def _split_stacked_inputs(
        self,
        coarse_tokens: Optional[torch.Tensor],
        coarse_tidx: Optional[torch.Tensor],
        fine_tokens: Optional[torch.Tensor],
        fine_tidx: Optional[torch.Tensor],
        agent1_coarse_tokens: Optional[torch.Tensor],
        agent1_coarse_tidx: Optional[torch.Tensor],
        agent1_fine_tokens: Optional[torch.Tensor],
        agent1_fine_tidx: Optional[torch.Tensor],
        agent2_coarse_tokens: Optional[torch.Tensor],
        agent2_coarse_tidx: Optional[torch.Tensor],
        agent2_fine_tokens: Optional[torch.Tensor],
        agent2_fine_tidx: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """兼容两种调用方式，把输入拆成 agent1/agent2。

        方式一：训练时推荐的 stacked 输入，形如 (B, 2, N, C)。
        方式二：显式传 agent1_* / agent2_*，便于调试或单独调用。

        输出顺序：
        agent1 coarse/tidx/fine/tidx, agent2 coarse/tidx/fine/tidx。
        """
        if coarse_tokens is not None:
            if coarse_tokens.dim() != 4 or fine_tokens is None or coarse_tidx is None or fine_tidx is None:
                raise ValueError("Stacked inputs require coarse/fine tokens with shape (B, 2, N, C).")
            return (
                # 切片
                coarse_tokens[:, 0], coarse_tidx[:, 0], fine_tokens[:, 0], fine_tidx[:, 0],
                coarse_tokens[:, 1], coarse_tidx[:, 1], fine_tokens[:, 1], fine_tidx[:, 1],
            )
        required = [
            agent1_coarse_tokens, agent1_coarse_tidx, agent1_fine_tokens, agent1_fine_tidx,
            agent2_coarse_tokens, agent2_coarse_tidx, agent2_fine_tokens, agent2_fine_tidx,
        ]
        if any(x is None for x in required):
            raise ValueError("Pass either stacked inputs or all agent1_/agent2_ split inputs.")
        return (
            agent1_coarse_tokens, agent1_coarse_tidx, agent1_fine_tokens, agent1_fine_tidx,
            agent2_coarse_tokens, agent2_coarse_tidx, agent2_fine_tokens, agent2_fine_tidx,
        )

    def forward(
        self,
        coarse_tokens: Optional[torch.Tensor] = None,
        coarse_tidx: Optional[torch.Tensor] = None,
        fine_tokens: Optional[torch.Tensor] = None,
        fine_tidx: Optional[torch.Tensor] = None,
        instructions: Optional[List[str]] = None,
        bbox_feat: Optional[torch.Tensor] = None,
        yaw_hist: Optional[torch.Tensor] = None,
        yaw_curr: Optional[torch.Tensor] = None,
        agent1_coarse_tokens: Optional[torch.Tensor] = None,
        agent1_coarse_tidx: Optional[torch.Tensor] = None,
        agent1_fine_tokens: Optional[torch.Tensor] = None,
        agent1_fine_tidx: Optional[torch.Tensor] = None,
        agent1_bbox_feat: Optional[torch.Tensor] = None,
        agent1_yaw_hist: Optional[torch.Tensor] = None,
        agent1_yaw_curr: Optional[torch.Tensor] = None,
        agent2_coarse_tokens: Optional[torch.Tensor] = None,
        agent2_coarse_tidx: Optional[torch.Tensor] = None,
        agent2_fine_tokens: Optional[torch.Tensor] = None,
        agent2_fine_tidx: Optional[torch.Tensor] = None,
        agent2_bbox_feat: Optional[torch.Tensor] = None,
        bbox_valid_mask: Optional[torch.Tensor] = None,
        agent2_yaw_hist: Optional[torch.Tensor] = None,
        agent2_yaw_curr: Optional[torch.Tensor] = None,
        target_waypoints: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, Dict[str, Optional[torch.Tensor]]]]:
        """双 Agent 前向传播。

        主要输入：
        - coarse_tokens/fine_tokens: 视觉 cache，stacked 形式为 (B, 2, N, C)。
        - bbox_feat: 两个 Agent 的目标框，(B, 2, 4)，格式 cxcywh_norm。
        - instructions: 文本任务指令。

        主要输出：
        - waypoints: (B, 2, n_waypoints, 3)，两个 Agent 的未来局部路点。
        - bbox_valid_mask: (B, 2)，标记每个 Agent 是否有有效 bbox prior。
        - refined_bbox: (B, 2, 4)，逐 Agent bbox 检测或 prior 修正。
        - visible_logits/visible_score: (B, 2)，辅助可见性预测。
        """
        device = next(self.parameters()).device
        a1_c, a1_ct, a1_f, a1_ft, a2_c, a2_ct, a2_f, a2_ft = self._split_stacked_inputs(
            coarse_tokens, coarse_tidx, fine_tokens, fine_tidx,
            agent1_coarse_tokens, agent1_coarse_tidx, agent1_fine_tokens, agent1_fine_tidx,
            agent2_coarse_tokens, agent2_coarse_tidx, agent2_fine_tokens, agent2_fine_tidx,
        )
        a1_c, a1_ct, a1_f, a1_ft = a1_c.to(device), a1_ct.to(device), a1_f.to(device), a1_ft.to(device)
        a2_c, a2_ct, a2_f, a2_ft = a2_c.to(device), a2_ct.to(device), a2_f.to(device), a2_ft.to(device)
        B = a1_c.size(0)
        if instructions is None:
            instructions = ["follow the person"] * B

        if bbox_feat is not None:
            bbox = self._normalize_bbox(bbox_feat, B, device)
            available_prior_mask = torch.ones(B, self.cfg.num_agents, dtype=torch.bool, device=device)
        else:
            b1 = torch.zeros(B, 4, dtype=torch.float32, device=device) if agent1_bbox_feat is None else agent1_bbox_feat.to(device)
            b2 = torch.zeros(B, 4, dtype=torch.float32, device=device) if agent2_bbox_feat is None else agent2_bbox_feat.to(device)
            bbox = torch.stack([b1, b2], dim=1).float().clamp(0.0, 1.0)
            available_prior_mask = torch.tensor(
                [agent1_bbox_feat is not None, agent2_bbox_feat is not None],
                dtype=torch.bool,
                device=device,
            ).view(1, self.cfg.num_agents).expand(B, -1)
        if bbox_valid_mask is None:
            prior_mask = available_prior_mask
        else:
            prior_mask = bbox_valid_mask.to(device=device, dtype=torch.bool)
            if prior_mask.shape != (B, self.cfg.num_agents):
                raise ValueError(
                    f"bbox_valid_mask must have shape (B, {self.cfg.num_agents}), got {tuple(prior_mask.shape)}"
                )
            if torch.any(prior_mask & ~available_prior_mask):
                raise ValueError("bbox_valid_mask marks an Agent valid, but no bbox prior was provided for it.")

        if yaw_hist is not None and yaw_hist.dim() == 3:
            agent1_yaw_hist, agent2_yaw_hist = yaw_hist[:, 0], yaw_hist[:, 1]
        if yaw_curr is not None and yaw_curr.dim() == 3:
            agent1_yaw_curr, agent2_yaw_curr = yaw_curr[:, 0], yaw_curr[:, 1]

        a1_seq, a1_visual = self._encode_agent_stream(
            a1_c, a1_ct, a1_f, a1_ft, bbox[:, 0], 0,
            yaw_hist=agent1_yaw_hist, yaw_curr=agent1_yaw_curr,
        )
        a2_seq, a2_visual = self._encode_agent_stream(
            a2_c, a2_ct, a2_f, a2_ft, bbox[:, 1], 1,
            yaw_hist=agent2_yaw_hist, yaw_curr=agent2_yaw_curr,
        )

        # 1. 文本指令先进入 LLM embedding 空间。
        txt_emb, txt_mask = self._embed_text(instructions, device)
        # 2. 四个可学习查询 token：
        #    ACT1/ACT2 分别读取两个 Agent 的规划状态；GND1/GND2 分别绑定各自
        #    Agent 坐标系，但都位于完整双视角上下文之后。
        act1 = self.tvi.make_query_token(self.act_token_1.expand(B, 1, -1), KIND_ACT, 0, 0)
        act2 = self.tvi.make_query_token(self.act_token_2.expand(B, 1, -1), KIND_ACT, 1, 1)
        gnd1 = self.tvi.make_query_token(self.gnd_token_1.expand(B, 1, -1), KIND_GND, 0, 0)
        gnd2 = self.tvi.make_query_token(self.gnd_token_2.expand(B, 1, -1), KIND_GND, 1, 1)
        # 3. 两个视觉流都放在查询 token 之前，保证因果 LLM 中 ACT1/ACT2 都能读取双视角信息。
        pieces = [txt_emb, a1_seq, a2_seq, act1, act2, gnd1, gnd2]
        lengths = [p.size(1) for p in pieces]
        act1_pos = sum(lengths[:3])
        act2_pos = act1_pos + lengths[3]
        gnd1_pos = sum(lengths[:-2])
        gnd2_pos = sum(lengths) - 1

        seq = torch.cat(pieces, dim=1).to(dtype=self.llm_dtype)
        attn = torch.cat(
            [txt_mask, torch.ones(B, sum(lengths[1:]), dtype=torch.long, device=device)],
            dim=1,
        )
        # 4. LLM 在同一个上下文里融合文本、无人机视觉流、机器狗视觉流和 bbox 先验。
        out = self.llm(inputs_embeds=seq, attention_mask=attn, output_hidden_states=True, use_cache=False)
        hidden = out.last_hidden_state.float()
        h_act1, h_act2 = hidden[:, act1_pos, :], hidden[:, act2_pos, :]
        h_gnd = torch.stack([hidden[:, gnd1_pos, :], hidden[:, gnd2_pos, :]], dim=1)
        # 5. 两个 ACT 隐藏状态分别进入各自 planner head，输出实际单位路点。
        action_output_agent1 = None
        action_output_agent2 = None
        if self.cfg.use_anchor_diffusion:
            # target_waypoints: (B, 2, Nw, D)，分别监督无人机和机器狗的扩散头。
            target1 = target_waypoints[:, 0] if target_waypoints is not None else None
            target2 = target_waypoints[:, 1] if target_waypoints is not None else None
            mask1 = valid_mask[:, 0] if valid_mask is not None else None
            mask2 = valid_mask[:, 1] if valid_mask is not None else None
            action_output_agent1 = self.planner_agent1(h_act1, target=target1, valid_mask=mask1)
            action_output_agent2 = self.planner_agent2(h_act2, target=target2, valid_mask=mask2)
            agent1_waypoints = action_output_agent1["trajectory"]
            agent2_waypoints = action_output_agent2["trajectory"]
        else:
            agent1_waypoints = self.planner_agent1(h_act1) * self.alpha_task
            agent2_waypoints = self.planner_agent2(h_act2) * self.alpha_task
        visual_tokens = torch.stack([a1_visual, a2_visual], dim=1) if self.cfg.return_token_logits and a1_visual.size(1) == a2_visual.size(1) else None
        # Without a bbox prior, predict an absolute normalized box. With a prior,
        # preserve the original behavior and predict a bounded refinement.
        grounding = self.grounding_head(
            h_gnd,
            bbox_feat=bbox,
            bbox_valid_mask=prior_mask,
            visual_tokens=visual_tokens,
        )

        if not return_dict:
            return agent1_waypoints, agent2_waypoints, grounding
        result = {
            "agent1_waypoints": agent1_waypoints,
            "agent2_waypoints": agent2_waypoints,
            "waypoints": torch.stack([agent1_waypoints, agent2_waypoints], dim=1),
            "refined_bbox": grounding["refined_bbox"],
            "absolute_bbox": grounding["absolute_bbox"],
            "bbox_prior_mask": grounding["bbox_prior_mask"],
            "visible_logits": grounding["visible_logits"],
            "visible_score": grounding["visible_score"],
            "token_logits": grounding["token_logits"],
        }
        if action_output_agent1 is not None and action_output_agent2 is not None:
            result.update(
                {
                    "candidate_trajectories": torch.stack(
                        [
                            action_output_agent1["candidate_trajectories"],
                            action_output_agent2["candidate_trajectories"],
                        ],
                        dim=1,
                    ),
                    "candidate_logits": torch.stack(
                        [action_output_agent1["candidate_logits"], action_output_agent2["candidate_logits"]],
                        dim=1,
                    ),
                    "candidate_scores": torch.stack(
                        [action_output_agent1["candidate_scores"], action_output_agent2["candidate_scores"]],
                        dim=1,
                    ),
                    "action_output_agent1": action_output_agent1,
                    "action_output_agent2": action_output_agent2,
                }
            )
            if "loss" in action_output_agent1 and "loss" in action_output_agent2:
                result["action_loss"] = action_output_agent1["loss"] + action_output_agent2["loss"]
                # 与 action_loss 保持一致，记录两个 Agent 的扩散损失分量合计值。
                result["regression_loss"] = (
                    action_output_agent1["regression_loss"] + action_output_agent2["regression_loss"]
                )
                result["score_loss"] = action_output_agent1["score_loss"] + action_output_agent2["score_loss"]
        return result


OpenTrackVLAMultiAgent = MultiAgentOpenTrackVLA


# ----------------------- 单 Agent离线推理数据集 -----------------------

@dataclass
class DataConfig:
    train_json: str
    n_waypoints: int = 8
    history: int = 31
    default_dt: float = 0.1
    cache_root: Optional[str] = None


class JsonTrackingDataset(Dataset):
    def __init__(self, cfg: DataConfig):
        super().__init__()
        self.cfg = cfg
        p = Path(cfg.train_json)
        candidate = p if p.is_dir() else p.parent
        max_up = 4
        while max_up >= 0 and not (candidate / 'frames').exists():
            if candidate.parent == candidate:
                break
            candidate = candidate.parent
            max_up -= 1
        self.base_root = candidate
        self.cache_root = Path(cfg.cache_root) if cfg.cache_root is not None else (self.base_root / "vision_cache")
        self._online_encoder: Optional[VisionFeatureCacher] = None
        self._bbox_processor = None
        self._bbox_model = None
        self._lazy = False
        self._index: Optional[List[Tuple[str, int]]] = None
        self.examples: Optional[List[Dict[str, Any]]] = None
        if p.is_file() and p.suffix.lower() == '.json':
            data = load_examples_from_path(cfg.train_json)
            assert isinstance(data, list) and len(data) > 0
            self.examples = data
        else:
            files: List[Path] = []
            if p.is_file() and p.suffix.lower() == '.jsonl':
                files = [p]
            elif p.is_dir():
                files = sorted(p.rglob('*.jsonl'))
            if len(files) == 0:
                raise FileNotFoundError(f"No .jsonl files found under: {cfg.train_json}")
            self._lazy = True
            self._index = []
            for fp in files:
                with open(fp, 'rb') as f:
                    pos = 0
                    while True:
                        line = f.readline()
                        if not line:
                            break
                        if line.strip():
                            self._index.append((str(fp), pos))
                        pos += len(line)
            if len(self._index) == 0:
                raise RuntimeError(f"No examples indexed from .jsonl sources under: {cfg.train_json}")
        H_target = int(self.cfg.history)
        self.coarse_frames = H_target

    def __len__(self):
        if self.examples is not None:
            return len(self.examples)
        if self._index is not None:
            return len(self._index)
        return 0

    def _load_tokens(self, path: str) -> torch.Tensor:
        return load_tokens_file(path)

    def _get_online_encoder(self) -> VisionFeatureCacher:
        if self._online_encoder is None:
            from torch.utils.data import get_worker_info
            worker_info = get_worker_info()
            use_cuda = torch.cuda.is_available() and (worker_info is None)
            cfg = VisionCacheConfig(image_size=384, batch_size=8, device=('cuda' if use_cuda else 'cpu'))
            self._online_encoder = VisionFeatureCacher(cfg)
            self._online_encoder.eval()
        return self._online_encoder

    def _get_bbox_detector(self):
        if self._bbox_model is None or self._bbox_processor is None:
            from torch.utils.data import get_worker_info
            worker_info = get_worker_info()
            use_cuda = torch.cuda.is_available() and (worker_info is None)
            device = torch.device('cuda' if use_cuda else 'cpu')
            try:
                from transformers import AutoProcessor, OmDetTurboForObjectDetection
                self._bbox_processor = AutoProcessor.from_pretrained("omlab/omdet-turbo-swin-tiny-hf")
                self._bbox_model = OmDetTurboForObjectDetection.from_pretrained("omlab/omdet-turbo-swin-tiny-hf").to(device).eval()
            except Exception as _e:
                print(f"[bbox] failed to init omdet-turbo: {_e}")
                self._bbox_processor = None
                self._bbox_model = None
        return self._bbox_processor, self._bbox_model

    @torch.inference_mode()
    def _encode_image_tokens(self, img_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        enc = self._get_online_encoder()
        pil = Image.open(str(img_path)).convert('RGB')
        tok_dino, Hp, Wp = enc._encode_dino([pil])
        tok_sigl = enc._encode_siglip([pil], out_hw=(Hp, Wp))
        Vt_cat = torch.cat([tok_dino, tok_sigl], dim=-1)
        Vfine = grid_pool_tokens(Vt_cat, Hp, Wp, out_tokens=64)
        Vcoarse = grid_pool_tokens(Vt_cat, Hp, Wp, out_tokens=4)# 设置token数量为4，保持全局信息
        return Vcoarse[0].cpu().float(), Vfine[0].cpu().float()

    def _read_indexed_example(self, idx: int) -> Dict[str, Any]:
        assert self._lazy and self._index is not None
        fp, off = self._index[idx]
        with open(fp, 'rb') as f:
            f.seek(off)
            line = f.readline()
        return json.loads(line.decode('utf-8'))

    def get_example(self, idx: int) -> Dict[str, Any]:
        if self._lazy:
            return self._read_indexed_example(idx)
        assert self.examples is not None
        return self.examples[idx]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.get_example(idx)
        H = self.coarse_frames

        curr_path = Path(ex['current'])
        abs_curr_img = curr_path if curr_path.is_absolute() else (self.base_root / curr_path)
        try:
            rel_curr = abs_curr_img.relative_to(self.base_root)
        except ValueError:
            rel_curr = abs_curr_img
        curr_token_dir = self.cache_root / rel_curr.parent
        curr_token_name = rel_curr.stem + "_vfine.pt"
        curr_tok_path = curr_token_dir / curr_token_name
        try:
            fine_tokens = self._load_tokens(str(curr_tok_path))
        except Exception:
            curr_token_dir.mkdir(parents=True, exist_ok=True)
            vc, vf = self._encode_image_tokens(abs_curr_img)
            try:
                torch.save(vf.half(), str(curr_tok_path))
            except Exception:
                pass
            fine_tokens = vf
        fine_tidx = torch.full((fine_tokens.size(0),), fill_value=H, dtype=torch.long)

        imgs_src = ex.get('images', [])
        imgs_trim = imgs_src[-H:]
        missing = H - len(imgs_trim)
        coarse_list, coarse_tidx = [], []
        first_tok: Optional[torch.Tensor] = None
        current_vc: Optional[torch.Tensor] = None
        for t in range(H):
            if t < missing:
                tok = None
            else:
                img_p = imgs_trim[t - missing]
                rp = Path(img_p)
                abs_img = rp if rp.is_absolute() else (self.base_root / rp)
                try:
                    rel_img = abs_img.relative_to(self.base_root)
                except ValueError:
                    rel_img = abs_img
                token_dir = self.cache_root / rel_img.parent
                token_name = rel_img.stem + "_vcoarse.pt"
                tok_path = token_dir / token_name
                try:
                    tok = self._load_tokens(str(tok_path))
                except Exception:
                    token_dir.mkdir(parents=True, exist_ok=True)
                    vc, vf = self._encode_image_tokens(abs_img)
                    try:
                        torch.save(vc.half(), str(tok_path))
                    except Exception:
                        pass
                    tok = vc
                if first_tok is None:
                    first_tok = tok
            if tok is None:
                if first_tok is not None:
                    tok = first_tok
                else:
                    try:
                        if current_vc is None:
                            cur_coarse_name = rel_curr.stem + "_vcoarse.pt"
                            cur_coarse_path = curr_token_dir / cur_coarse_name
                            try:
                                current_vc = self._load_tokens(str(cur_coarse_path))
                            except Exception:
                                vc_tmp, _ = self._encode_image_tokens(abs_curr_img)
                                current_vc = vc_tmp
                        tok = current_vc
                    except Exception:
                        tok = torch.zeros(4, fine_tokens.size(1), dtype=torch.float32)
            coarse_list.append(tok)
            coarse_tidx.append(torch.full((tok.size(0),), fill_value=t, dtype=torch.long))
        coarse_tokens = torch.cat(coarse_list, dim=0)
        coarse_tidx   = torch.cat(coarse_tidx, dim=0)

        yaw_hist = torch.tensor(ex.get('yaw_hist', [0.0]*H), dtype=torch.float32)
        yaw_curr = torch.tensor(ex.get('yaw_curr', 0.0), dtype=torch.float32).view(1)

        if 'waypoints' in ex:
            wp = torch.tensor(ex['waypoints'], dtype=torch.float32)
        else:
            assert 'actions' in ex, "JSON needs either 'waypoints' or 'actions'"
            dt = float(ex.get('dt', self.cfg.default_dt))
            traj = integrate_actions_to_waypoints(np.asarray(ex['actions'], dtype=np.float32), self.cfg.n_waypoints, dt)
            wp = torch.from_numpy(traj)

        if 'valid_mask' in ex:
            vm = torch.tensor(ex['valid_mask'], dtype=torch.bool)
        elif 'valid_idx' in ex:
            vm = torch.zeros(self.cfg.n_waypoints, dtype=torch.bool)
            vm[torch.tensor(ex['valid_idx'], dtype=torch.long)] = True
        else:
            vm = torch.ones(self.cfg.n_waypoints, dtype=torch.bool)

        item: Dict[str, Any] = {
            'coarse_tokens': coarse_tokens,
            'coarse_tidx':   coarse_tidx,
            'fine_tokens':   fine_tokens,
            'fine_tidx':     fine_tidx,
            'yaw_hist':      yaw_hist,
            'yaw_curr':      yaw_curr,
            'waypoints':     wp,
            'valid_mask':    vm,
            'instruction':   ex.get('instruction', 'follow the person'),
            'current_path':  str(abs_curr_img),
        }
        if 'bbox' in ex:
            bb = np.asarray(ex['bbox'], dtype=np.float32)
            if bb.size == 4:
                x0,y0,x1,y1 = bb.tolist()
                w = max(1e-6, x1 - x0); h = max(1e-6, y1 - y0)
                cx = x0 + 0.5*w; cy = y0 + 0.5*h
                try:
                    with Image.open(str(abs_curr_img)) as _im:
                        W,H = _im.size
                    if max(cx,cy,w,h) > 1.5:
                        cx /= W; cy /= H; w /= W; h /= H
                except Exception:
                    pass
                item['bbox_feat'] = torch.tensor([cx, cy, w, h], dtype=torch.float32)
        elif self.cfg.use_bbox_token:
            try:
                proc, det = self._get_bbox_detector()
                if proc is not None and det is not None:
                    with Image.open(str(abs_curr_img)).convert('RGB') as _im:
                        H, W = _im.height, _im.width
                        labels = ["person"]
                        inputs = proc(_im, text=labels, return_tensors='pt')
                        dev = next(det.parameters()).device
                        inputs = {k: v.to(dev) for k, v in inputs.items()}
                        with torch.inference_mode():
                            outputs = det(**inputs)
                        results = proc.post_process_grounded_object_detection(outputs,
                                                                             target_sizes=[(H, W)],
                                                                             text_labels=labels,
                                                                             threshold=0.25,
                                                                             nms_threshold=0.3)
                        res = results[0]
                        boxes = res.get('boxes', None)
                        scores = res.get('scores', None)
                        tlabels = res.get('text_labels', [])
                        if boxes is not None and scores is not None and len(boxes) > 0:
                            best = -1; best_s = -1.0
                            for i in range(len(boxes)):
                                lbl = tlabels[i] if i < len(tlabels) else ''
                                if lbl in ('person','human') and scores[i].item() > best_s:
                                    best_s = scores[i].item(); best = i
                            if best < 0:
                                best = int(torch.argmax(scores).item())
                            x0,y0,x1,y1 = boxes[best].tolist()
                            w = max(1e-6, x1 - x0); h = max(1e-6, y1 - y0)
                            cx = x0 + 0.5*w; cy = y0 + 0.5*h
                            cx /= W; cy /= H; w /= W; h /= H
                            item['bbox_feat'] = torch.tensor([cx, cy, w, h], dtype=torch.float32)
            except Exception:
                pass
        return item


# ----------------------- 单 Agent batch 组装 -----------------------

def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    instr = [b['instruction'] for b in batch]
    return {
        'coarse_tokens': torch.stack([b['coarse_tokens'] for b in batch], dim=0),
        'coarse_tidx':   torch.stack([b['coarse_tidx']   for b in batch], dim=0),
        'fine_tokens':   torch.stack([b['fine_tokens']   for b in batch], dim=0),
        'fine_tidx':     torch.stack([b['fine_tidx']     for b in batch], dim=0),
        'yaw_hist':      torch.stack([b['yaw_hist']      for b in batch], dim=0),
        'yaw_curr':      torch.stack([b['yaw_curr']      for b in batch], dim=0),
        'waypoints':     torch.stack([b['waypoints']     for b in batch], dim=0),
        'valid_mask':    torch.stack([b['valid_mask']    for b in batch], dim=0),
        'instruction':   instr,
        'current_path':  [b['current_path'] for b in batch],
        'bbox_feat':     torch.stack([b['bbox_feat'] if 'bbox_feat' in b else torch.zeros(4, dtype=torch.float32) for b in batch], dim=0),
    }


# ----------------------- 单 Agent离线推理流程 -----------------------

@torch.inference_mode()
def _run_inference(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = cfg.infer_ckpt
    try:
        if ckpt_path is None:
            from glob import glob
            pts = sorted(glob(os.path.join(cfg.out_dir, 'model_epoch*.pt')), key=lambda p: os.path.getmtime(p))
            ckpt_path = pts[-1] if pts else None
    except Exception:
        ckpt_path = None
    if ckpt_path is None or (not os.path.exists(ckpt_path)):
        raise FileNotFoundError(f"No checkpoint found for inference (looked at --infer_ckpt and {cfg.out_dir})")

    obj = torch.load(ckpt_path, map_location=device)
    ck = obj if isinstance(obj, dict) else {}
    ck_cfg = ck.get('config', {})

    n_waypoints = int(ck_cfg.get('n_waypoints', getattr(cfg, 'n_waypoints', 8)))
    use_angle_tvi = bool(ck_cfg.get('use_angle_tvi', False))
    no_tanh_actions = bool(ck_cfg.get('no_tanh_actions', True))
    vision_feat_dim = int(ck_cfg.get('vision_feat_dim', getattr(cfg, 'vision_feat_dim', 1536)))
    alpha_xy        = ck_cfg.get('alpha_xy', getattr(cfg, 'alpha_xy', None))

    model = OpenTrackVLA(
        ModelConfig(
            llm_name=str(ck_cfg.get('llm_name', "Qwen/Qwen3-0.6B")),
            n_waypoints=n_waypoints,
            beta_nav=float(ck_cfg.get('beta_nav', 10.0)),
            use_angle_tvi=use_angle_tvi,
            use_tanh_actions=(not no_tanh_actions),
            alpha_xy=alpha_xy,
            use_anchor_diffusion=bool(ck_cfg.get('use_anchor_diffusion', getattr(cfg, 'use_anchor_diffusion', False))),
            diffusion_anchor_path=ck_cfg.get('diffusion_anchor_path', getattr(cfg, 'diffusion_anchor_path', None)),
            diffusion_num_anchors=int(ck_cfg.get('diffusion_num_anchors', getattr(cfg, 'diffusion_num_anchors', 40))),
            diffusion_hidden_dim=int(ck_cfg.get('diffusion_hidden_dim', 384)),
            diffusion_depth=int(ck_cfg.get('diffusion_depth', 6)),
            diffusion_num_heads=int(ck_cfg.get('diffusion_num_heads', 4)),
            diffusion_mlp_ratio=float(ck_cfg.get('diffusion_mlp_ratio', 4.0)),
            diffusion_dropout=float(ck_cfg.get('diffusion_dropout', 0.0)),
            diffusion_num_train_timesteps=int(ck_cfg.get('diffusion_num_train_timesteps', 1000)),
            diffusion_train_truncation_steps=int(ck_cfg.get('diffusion_train_truncation_steps', 50)),
            diffusion_inference_start_timestep=int(ck_cfg.get('diffusion_inference_start_timestep', 10)),
            diffusion_inference_steps=int(ck_cfg.get('diffusion_inference_steps', 2)),
            diffusion_score_loss_weight=float(ck_cfg.get('diffusion_score_loss_weight', 100.0)),
            diffusion_score_loss_reduction=str(ck_cfg.get('diffusion_score_loss_reduction', 'mean')),
            diffusion_deterministic_inference=bool(ck_cfg.get('diffusion_deterministic_inference', False)),
        ),
        vision_feat_dim=vision_feat_dim,
    ).to(device).eval()
    msd = ck.get('model_state', None)
    if msd:
        model.load_state_dict(msd, strict=False)

    if cfg.infer_json is None:
        raise ValueError('--infer_json is required for inference')
    vds = JsonTrackingDataset(DataConfig(train_json=cfg.infer_json, n_waypoints=n_waypoints, history=getattr(cfg, 'history', 31), cache_root=getattr(cfg, 'cache_root', None)))
    vdl = DataLoader(vds, batch_size=getattr(cfg, 'batch_size', 2), shuffle=False, num_workers=min(2, getattr(cfg, 'num_workers', 4)), pin_memory=True, collate_fn=collate_batch)

    os.makedirs(cfg.infer_out, exist_ok=True)
    vis_dir = os.path.join(cfg.infer_out, 'vis')
    npz_dir = os.path.join(cfg.infer_out, 'npz')
    if cfg.infer_vis:
        os.makedirs(vis_dir, exist_ok=True)
    if cfg.infer_save_npz:
        os.makedirs(npz_dir, exist_ok=True)

    batches_limit = max(0, int(getattr(cfg, 'infer_batches', 0)))
    bdone = 0
    for bidx, batch in enumerate(vdl):
        coarse_tokens = batch['coarse_tokens'].to(device)
        coarse_tidx   = batch['coarse_tidx'].to(device)
        fine_tokens   = batch['fine_tokens'].to(device)
        fine_tidx     = batch['fine_tidx'].to(device)
        yaw_hist      = batch['yaw_hist'].to(device)
        yaw_curr      = batch['yaw_curr'].to(device)
        instr         = batch['instruction']
        bbox_feat     = batch.get('bbox_feat', None)

        pred = model(
            coarse_tokens, coarse_tidx,
            fine_tokens, fine_tidx,
            instr,
            yaw_hist=yaw_hist if use_angle_tvi else None,
            yaw_curr=yaw_curr if use_angle_tvi else None
        )

        if cfg.infer_save_npz:
            try:
                with torch.no_grad():
                    pred_np = pred.detach().float().cpu().numpy()
                Bcur = pred_np.shape[0]
                for bi in range(Bcur):
                    fpath = os.path.join(npz_dir, f"b{bidx:06d}_i{bi:03d}.npz")
                    np.savez_compressed(
                        fpath,
                        pred=pred_np[bi],
                        instruction=instr[bi],
                        current_path=(batch.get('current_path', [''])[bi] if isinstance(batch.get('current_path', []), list) else '')
                    )
            except Exception:
                pass

        if cfg.infer_vis:
            try:
                with torch.no_grad():
                    pred_draw = pred.detach().float()
                    try:
                        model_inspect = model
                        alpha_vec = getattr(model_inspect, 'alpha_task', None)
                        if alpha_vec is not None and pred_draw.size(-1) >= 2 and alpha_vec.size(-1) >= 2:
                            max_xy = pred_draw[..., 0:2].abs().max().item()
                            if max_xy <= 1.5:
                                ax = alpha_vec[..., 0:2].clamp_min(1e-6).to(pred_draw.device, pred_draw.dtype)
                                pred_draw = pred_draw.clone()
                                pred_draw[..., 0:2] = pred_draw[..., 0:2] * ax
                    except Exception:
                        pass
                    pred_np = pred_draw.cpu().numpy()
                cur_paths = batch.get('current_path', [])
                Bcur = pred_np.shape[0]
                for bi in range(min(Bcur, 4)):
                    cur_path = cur_paths[bi] if isinstance(cur_paths, list) and bi < len(cur_paths) else None
                    if cur_path is None or (not os.path.exists(cur_path)):
                        continue
                    pil_img = Image.open(cur_path).convert('RGB')
                    draw = ImageDraw.Draw(pil_img)
                    w, h = pil_img.size
                    base_x = w // 2
                    base_y = int(h * 0.86)
                    def to_pxxy(traj):
                        pts = []
                        for i in range(min(traj.shape[0], 64)):
                            x, y = float(traj[i, 0]), float(traj[i, 1])
                            px = base_x - int(y * 120)
                            py = base_y - int(x * 120)
                            pts.append((px, py))
                        return pts
                    pts_pred = to_pxxy(pred_np[bi])
                    for i2 in range(1, len(pts_pred)):
                        draw.line([pts_pred[i2-1], pts_pred[i2]], fill=(0, 255, 200), width=6)
                    if pts_pred:
                        r0 = 6
                        sx, sy = pts_pred[0]
                        draw.ellipse([sx-r0, sy-r0, sx+r0, sy+r0], fill=(0,255,0))
                    out_path = os.path.join(vis_dir, f"b{bidx:06d}_i{bi:03d}.jpg")
                    pil_img.save(out_path)
            except Exception:
                pass

        bdone += 1
        if batches_limit and bdone >= batches_limit:
            break

    print(f"[INFER] Done. Outputs under {cfg.infer_out}")


# ----------------------- 命令行参数与程序入口 -----------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--infer_json', type=str, required=True, help='Run inference on this dataset (json/jsonl/dir)')
    ap.add_argument('--infer_ckpt', type=str, default=None, help='Checkpoint to load for inference (defaults to latest in out_dir)')
    ap.add_argument(
        '--out_dir',
        type=str,
        default='/data/hdt/ntv_data/ckpt/ckpts_multi_agent_anchor_diffusion',
        help='Directory where checkpoints are stored (for default lookup)',
    )
    ap.add_argument('--infer_out', type=str, default='./infer_out', help='Output directory for inference results')
    ap.add_argument('--infer_batches', type=int, default=0, help='Limit number of batches to run at inference (0 = all)')
    ap.add_argument('--infer_vis', action='store_true', help='Save visualization images during inference')
    ap.add_argument('--infer_save_npz', action='store_true', help='Save npz predictions during inference')
    ap.add_argument('--n_waypoints', type=int, default=8)
    ap.add_argument('--history', type=int, default=31)
    ap.add_argument('--batch_size', type=int, default=2)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--vision_feat_dim', type=int, default=1536)
    ap.add_argument('--cache_root', type=str, default=None)
    ap.add_argument('--alpha_xy', type=float, default=None)
    ap.add_argument('--use_anchor_diffusion', action='store_true')
    ap.add_argument('--diffusion_anchor_path', type=str, default=None)
    ap.add_argument('--diffusion_num_anchors', type=int, default=40)
    ap.add_argument('--diffusion_score_loss_reduction', choices=['mean', 'sum'], default='mean')
    args = ap.parse_args()
    return args


if __name__ == '__main__':
    cfg = parse_args()
    _run_inference(cfg)
