#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AirGround-Coop V3 训练的共享数据与优化器运行时。

V3 专用入口会注入 dataset、model、loss 和 sampler callback；本模块
只管理双 Agent JSONL/cache、DDP、混合精度、验证、日志与 checkpoint。
"""

from __future__ import annotations
from collections import OrderedDict
from array import array
from dataclasses import dataclass
from datetime import timedelta
from typing import Iterator, List, Tuple, Optional, Dict, Any
import os, sys, json, math, argparse, time, csv, hashlib, fcntl
from pathlib import Path
from contextlib import nullcontext
from PIL import Image, ImageDraw

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
import torch.distributed as dist

from transformers import AutoTokenizer, AutoModel
from tools.cache_gridpool import (
    VisionFeatureCacher,
    VisionCacheConfig,
    build_roi_cache_payload,
    crop_target_roi,
    grid_pool_tokens,
    load_roi_cache,
    adapt_siglip_grid,
)
from tools.bbox_spatial import bbox_prompt_from_spatial, bbox_spatial_fields
from tools.waypoint_inverse_dynamics import (
    INVERSE_CONTROL_VERSION,
    InverseDynamicsConfig,
    inverse_step_torch,
)
from tqdm import tqdm


MULTI_AGENT_COOP_INSTRUCTION = "Follow the person."

ROI_VISUAL_LAYOUT_PROMPT = (
    "Visual layout: GLOBAL_HISTORY and GLOBAL_CURRENT encode scene geometry; "
    "TARGET_ROI encodes the target person's identity and local motion. Combine all three."
)


def prepend_roi_visual_layout_prompt(text: str) -> str:
    """Add explicit visual-layout instructions once for oracle-ROI training/eval."""
    text = str(text or "").strip()
    if ROI_VISUAL_LAYOUT_PROMPT in text:
        return text
    # Keep the task and dynamic bbox fields first so truncation cannot silently
    # remove the behavior instruction while retaining developer-facing layout text.
    return f"Task: {text}\n{ROI_VISUAL_LAYOUT_PROMPT}"


# ----------------------- 通用工具与数据加载 -----------------------

# Silence tokenizers fork warnings in dataloader workers
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

def set_seed(seed: int):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)


def load_tokens_file(path: str) -> torch.Tensor:
    try:
        obj = torch.load(path, map_location='cpu')
    except Exception:
        # PyTorch 2.6 defaults to weights_only=True; retry allowing full unpickling for trusted caches
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


def _cleanup_state_dict_keys(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make checkpoints usable across single-GPU and DDP.

    - If keys are prefixed with 'module.' (DDP/DataParallel), strip that prefix.
    - Otherwise return unchanged.
    """
    if not state_dict:
        return state_dict
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def _compute_total_grad_norm(parameters, norm_type: float = 2.0) -> float:
    parameters = [p for p in parameters if p.grad is not None]
    if not parameters:
        return 0.0
    device = parameters[0].grad.device
    if norm_type == float("inf"):
        total_norm = max(
            p.grad.detach().abs().max().to(device) for p in parameters
        )
        return float(total_norm.item())
    total = torch.zeros([], device=device)
    for parameter in parameters:
        param_norm = parameter.grad.detach().data.norm(norm_type)
        total += param_norm.pow(norm_type)
    return float(total.pow(1.0 / norm_type).item())


def find_multi_agent_base_root(train_path: Path) -> Path:
    """从 train_json 路径向上寻找双 Agent 数据根目录。

    数据流：
    train_json 可以是 <data_root>/jsonl、某个 .jsonl、或 <data_root>/dataset.json。
    Dataset 需要找到 <data_root>/frames 和 <data_root>/vision_cache，所以这里向上查找 frames/。
    """
    candidate = train_path if train_path.is_dir() else train_path.parent
    for _ in range(6):
        if (candidate / "frames").exists():
            return candidate
        if candidate.parent == candidate:
            break
        candidate = candidate.parent
    return train_path if train_path.is_dir() else train_path.parent


def resolve_multi_agent_path(base_root: Path, rel_or_abs: str) -> Path:
    """把 JSON 中的相对图片路径解析为绝对路径。"""
    p = Path(rel_or_abs)
    return p if p.is_absolute() else (base_root / p)


def _jsonl_index_fingerprint(files: List[Path]) -> str:
    digest=hashlib.sha256()
    for path in files:
        stat=path.stat(); digest.update(str(path.resolve()).encode()); digest.update(f':{stat.st_size}:{stat.st_mtime_ns}\n'.encode())
    return digest.hexdigest()


def _build_compact_jsonl_index(files: List[Path]) -> np.ndarray:
    file_ids=array('I'); offsets=array('Q')
    for file_id,path in enumerate(files):
        with path.open('rb') as handle:
            position=0
            while True:
                line=handle.readline()
                if not line: break
                if line.strip(): file_ids.append(file_id); offsets.append(position)
                position+=len(line)
    output=np.empty((len(file_ids),2),dtype=np.int64)
    output[:,0]=np.frombuffer(file_ids,dtype=np.uint32)
    output[:,1]=np.frombuffer(offsets,dtype=np.uint64).astype(np.int64,copy=False)
    return output


def _load_or_build_jsonl_index(files: List[Path], train_path: Path) -> np.ndarray:
    """Create one atomic mmap index shared by all DDP ranks/workers."""
    cache_value=os.environ.get('AIRGROUND_INDEX_CACHE_ROOT','').strip()
    if not cache_value: return _build_compact_jsonl_index(files)
    root=Path(cache_value).resolve(); root.mkdir(parents=True,exist_ok=True)
    key=hashlib.sha256(str(train_path.resolve()).encode()).hexdigest()[:20]
    index_path=root/f'{key}.npy'; metadata_path=root/f'{key}.json'; lock_path=root/f'{key}.lock'
    fingerprint=_jsonl_index_fingerprint(files)
    with lock_path.open('a+b') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        valid=False
        if index_path.is_file() and metadata_path.is_file():
            try:
                metadata=json.loads(metadata_path.read_text(encoding='utf-8'))
                valid=metadata.get('fingerprint')==fingerprint and int(metadata.get('files',-1))==len(files)
            except Exception: valid=False
        if not valid:
            started=time.time(); index=_build_compact_jsonl_index(files)
            temporary=index_path.with_name(f'.{index_path.name}.tmp-{os.getpid()}')
            with temporary.open('wb') as handle: np.save(handle,index,allow_pickle=False)
            os.replace(temporary,index_path)
            metadata_tmp=metadata_path.with_name(f'.{metadata_path.name}.tmp-{os.getpid()}')
            metadata_tmp.write_text(json.dumps({'fingerprint':fingerprint,'files':len(files),'rows':len(index)})+'\n',encoding='utf-8'); os.replace(metadata_tmp,metadata_path)
            print(f'[INDEX] built files={len(files)} rows={len(index)} seconds={time.time()-started:.1f} path={index_path}',flush=True)
        else:
            print(f'[INDEX] reused files={len(files)} rows={metadata["rows"]} path={index_path}',flush=True)
    return np.load(index_path,mmap_mode='r',allow_pickle=False)


def multi_agent_token_paths_for_frame(base_root: Path, cache_root: Path, frame_path: Path) -> Tuple[Path, Path]:
    """把 frame 路径映射到对应视觉 token cache。

    输入：
    data_root/frames/.../frame_00001.jpg

    输出：
    cache_root/frames/.../frame_00001_vcoarse.pt
    cache_root/frames/.../frame_00001_vfine.pt
    """
    try:
        rel = frame_path.absolute().relative_to(base_root.absolute())
    except ValueError:
        try:
            rel = frame_path.resolve().relative_to(base_root.resolve())
        except ValueError:
            parts = frame_path.parts
            if "frames" in parts:
                rel = Path(*parts[parts.index("frames") :])
            else:
                rel = Path(frame_path.name)
    token_dir = cache_root / rel.parent
    coarse = token_dir / f"{rel.stem}_vcoarse.pt"
    fine = token_dir / f"{rel.stem}_vfine.pt"
    # Compact layout stores vision tokens under {kind}/... rather than
    # {frames}/{kind}/....
    if "frames" in rel.parts and (not coarse.exists() or not fine.exists()):
        compact = Path(*rel.parts[rel.parts.index("frames") + 1:])
        compact_dir = cache_root / compact.parent
        coarse, fine = (compact_dir / f"{compact.stem}_vcoarse.pt",
                        compact_dir / f"{compact.stem}_vfine.pt")
    # AT samples are replayed from DT trajectories and intentionally reuse the
    # DT-rendered frames/cache.  Older AT JSONL still names the logical `at`
    # namespace, so fall back only when the AT cache is absent and DT exists.
    if "frames" in rel.parts and "at" in rel.parts:
        at_pos = rel.parts.index("at")
        dt_rel = Path(*rel.parts[:at_pos], "dt", *rel.parts[at_pos + 1 :])
        dt_dir = cache_root / dt_rel.parent
        dt_coarse = dt_dir / f"{rel.stem}_vcoarse.pt"
        dt_fine = dt_dir / f"{rel.stem}_vfine.pt"
        if (not coarse.exists() or not fine.exists()) and dt_coarse.exists() and dt_fine.exists():
            coarse, fine = dt_coarse, dt_fine
    return coarse, fine


def fit_multi_agent_waypoints_and_mask(
    waypoints: Any,
    valid_mask: Any,
    n_waypoints: int,
    action_dims: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """把 JSON 中的单 Agent 路点标签整理成固定形状。

    输入可能不足 n_waypoints；输出会复制最后一个有效路点补齐，
    同时用 valid_mask 告诉 loss 哪些位置是真实标签。

    输出：
    - waypoints:  (n_waypoints, action_dims)
    - valid_mask: (n_waypoints,)
    """
    wp = torch.as_tensor(waypoints, dtype=torch.float32)
    if wp.dim() == 1:
        wp = wp.view(1, -1)
    if wp.dim() != 2:
        raise ValueError(f"Expected per-agent waypoints shape (M,D), got {tuple(wp.shape)}")

    out = torch.zeros(n_waypoints, action_dims, dtype=torch.float32)
    dim = min(action_dims, wp.size(-1))
    length = min(n_waypoints, wp.size(0))
    if length > 0:
        out[:length, :dim] = wp[:length, :dim]
        if length < n_waypoints:
            out[length:, :dim] = wp[length - 1, :dim]

    if valid_mask is None:
        mask = torch.zeros(n_waypoints, dtype=torch.bool)
        mask[:length] = True
    else:
        vm = torch.as_tensor(valid_mask, dtype=torch.bool).view(-1)
        mask = torch.zeros(n_waypoints, dtype=torch.bool)
        mv_len = min(n_waypoints, vm.numel())
        mask[:mv_len] = vm[:mv_len]
        if mv_len == 0:
            mask[:length] = True
    return out, mask


def masked_mse_multi_agent(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """双 Agent masked MSE。

    pred/target: (B, 2, N, D)
    mask:        (B, 2, N)

    只在有效 waypoint 上计算误差，避免 partial horizon 的 padding 污染训练。
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred shape {tuple(pred.shape)} != target shape {tuple(target.shape)}")
    if mask.dim() == 2:
        mask = mask.unsqueeze(1)
    expanded = mask.view(*mask.shape, 1).expand_as(pred)
    se = (pred - target).pow(2)
    selected = se[expanded]
    return selected.mean() if selected.numel() > 0 else pred.new_tensor(0.0)


def weighted_multi_agent_waypoint_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    dt: torch.Tensor,
    *,
    loss_type: str,
    smooth_l1_beta: float,
    yaw_weight: float,
    final_weight: float,
    turn_sample_weight: float,
    turn_rate_threshold: float,
    turn_angle_threshold: float,
    stop_sample_weight: float,
    stop_speed_threshold: float,
    stop_window: int,
    dog_lateral_loss_weight: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute waypoint loss with extra weight on turning and stopping windows."""
    if pred.shape != target.shape:
        raise ValueError(f"pred shape {tuple(pred.shape)} != target shape {tuple(target.shape)}")
    if pred.dim() != 4 or pred.size(-1) < 3:
        raise ValueError(f"Expected (B,A,N,D>=3), got {tuple(pred.shape)}")
    mask = mask.bool()
    if mask.dim() == 2:
        mask = mask.unsqueeze(1)
    if mask.shape != pred.shape[:-1]:
        raise ValueError(f"mask shape {tuple(mask.shape)} != {tuple(pred.shape[:-1])}")

    if loss_type == "smooth_l1":
        point_error = F.smooth_l1_loss(
            pred,
            target,
            reduction="none",
            beta=float(smooth_l1_beta),
        )
    elif loss_type == "mse":
        point_error = (pred - target).pow(2)
    else:
        raise ValueError(f"Unsupported nav_loss_type={loss_type!r}")

    valid = mask.to(dtype=point_error.dtype)
    valid_count = valid.sum(dim=-1).clamp_min(1.0)
    xy_component_weights = torch.ones(
        1, pred.size(1), 1, 2, device=pred.device, dtype=point_error.dtype
    )
    if pred.size(1) >= 2:
        xy_component_weights[:, 1, :, 1] = float(dog_lateral_loss_weight)
    xy_weight_sum = xy_component_weights.sum(dim=-1).clamp_min(1e-6)
    xy_point_error = (
        point_error[..., :2] * xy_component_weights
    ).sum(dim=-1) / xy_weight_sum
    xy_per_agent = (xy_point_error * valid).sum(dim=-1) / valid_count
    yaw_per_agent = (point_error[..., 2] * valid).sum(dim=-1) / valid_count

    indices = torch.arange(mask.size(-1), device=mask.device).view(1, 1, -1)
    last_index = torch.where(mask, indices, torch.zeros_like(indices)).amax(dim=-1)
    gather_index = last_index[..., None, None].expand(-1, -1, 1, point_error.size(-1))
    final_error = point_error.gather(dim=2, index=gather_index).squeeze(2)
    final_target = target.gather(dim=2, index=gather_index).squeeze(2)
    per_agent_xy_count = xy_component_weights.sum(dim=-1).squeeze(2)
    final_xy = (
        final_error[..., :2] * xy_component_weights.squeeze(2)
    ).sum(dim=-1) / per_agent_xy_count.clamp_min(1e-6)
    component_weight_sum = (per_agent_xy_count + float(yaw_weight)).clamp_min(1e-6)
    final_per_agent = (
        per_agent_xy_count * final_xy
        + float(yaw_weight) * final_error[..., 2]
    ) / component_weight_sum

    if dt.dim() == 1:
        dt = dt[:, None, None]
    elif dt.dim() == 2:
        dt = dt[..., None]
    dt = dt.to(device=target.device, dtype=target.dtype).clamp_min(1e-6)
    delta = target[..., 1:, :3] - target[..., :-1, :3]
    delta_valid = mask[..., 1:] & mask[..., :-1]
    delta_valid_f = delta_valid.to(dtype=target.dtype)

    yaw_rate = delta[..., 2].abs() / dt
    masked_yaw_rate = torch.where(delta_valid, yaw_rate, torch.zeros_like(yaw_rate))
    first_theta = target[..., 0, 2]
    final_theta = final_target[..., 2]
    turn_mask = (
        (masked_yaw_rate.amax(dim=-1) >= float(turn_rate_threshold))
        | ((final_theta - first_theta).abs() >= float(turn_angle_threshold))
    )

    speed = torch.linalg.norm(delta[..., :2], dim=-1) / dt
    window = max(1, min(int(stop_window), speed.size(-1)))
    tail_valid_f = delta_valid_f[..., -window:]
    tail_count = tail_valid_f.sum(dim=-1)
    tail_speed = (speed[..., -window:] * tail_valid_f).sum(dim=-1) / tail_count.clamp_min(1.0)
    stop_mask = (tail_count > 0) & (tail_speed <= float(stop_speed_threshold))

    turn_factor = torch.where(
        turn_mask,
        torch.full_like(xy_per_agent, float(turn_sample_weight)),
        torch.ones_like(xy_per_agent),
    )
    stop_factor = torch.where(
        stop_mask,
        torch.full_like(xy_per_agent, float(stop_sample_weight)),
        torch.ones_like(xy_per_agent),
    )
    behavior_weight = torch.maximum(turn_factor, stop_factor)
    trajectory_per_agent = (
        per_agent_xy_count * xy_per_agent + float(yaw_weight) * yaw_per_agent
    ) / component_weight_sum
    per_agent = (
        trajectory_per_agent + float(final_weight) * final_per_agent
    ) * behavior_weight
    return per_agent, {
        "xy_per_agent": xy_per_agent,
        "yaw_per_agent": yaw_per_agent,
        "final_per_agent": final_per_agent,
        "turn_mask": turn_mask,
        "stop_mask": stop_mask,
        "behavior_weight": behavior_weight,
    }


def normalize_multi_agent_xy_by_alpha(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha_task: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """在 loss 中归一化 XY 维度。

    模型 forward 输出已经乘过 alpha_task，是实际单位。
    loss 里把 XY 再除回 normalized space，避免 x/y 数值尺度压过 yaw。
    """
    if alpha_task is None or pred.size(-1) < 2 or alpha_task.size(-1) < 2:
        return pred, target
    alpha_xy = alpha_task[..., 0:2].clamp_min(1e-6).to(device=pred.device, dtype=pred.dtype)
    while alpha_xy.dim() < pred.dim():
        alpha_xy = alpha_xy.unsqueeze(0)
    pred_n = pred.clone()
    target_n = target.clone()
    pred_n[..., 0:2] = pred_n[..., 0:2] / alpha_xy
    target_n[..., 0:2] = target_n[..., 0:2] / alpha_xy
    return pred_n, target_n


UNREAL_UNITS_PER_METER = 100.0


def _wrap_degrees(angle: float) -> float:
    """Wrap degrees to [-180, 180)."""
    return (float(angle) + 180.0) % 360.0 - 180.0


def relative_target_pose_from_unreal_poses(
    agent_pose: Any,
    target_pose: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a spatial grounding label from recorded Unreal poses.

    The label is agent-centric:
    [dx_m, dy_m, dz_m, sin(d_yaw), cos(d_yaw)].
    """
    try:
        a = np.asarray(agent_pose, dtype=np.float32)
        t = np.asarray(target_pose, dtype=np.float32)
        if a.shape[0] < 6 or t.shape[0] < 6:
            raise ValueError("pose must have at least 6 values")
        delta_xy_m = (t[:2] - a[:2]) / UNREAL_UNITS_PER_METER
        dz_m = float((t[2] - a[2]) / UNREAL_UNITS_PER_METER)
        yaw = math.radians(float(a[4]))
        forward = np.asarray([math.cos(yaw), math.sin(yaw)], dtype=np.float32)
        right = np.asarray([-math.sin(yaw), math.cos(yaw)], dtype=np.float32)
        dx_m = float(np.dot(delta_xy_m, forward))
        dy_m = float(np.dot(delta_xy_m, right))
        d_yaw = math.radians(_wrap_degrees(float(t[4]) - float(a[4])))
        rel = torch.tensor(
            [dx_m, dy_m, dz_m, math.sin(d_yaw), math.cos(d_yaw)],
            dtype=torch.float32,
        )
        return rel, torch.tensor(True, dtype=torch.bool)
    except Exception:
        return torch.zeros(5, dtype=torch.float32), torch.tensor(False, dtype=torch.bool)


@dataclass
# ----------------------- 双 Agent训练数据集 -----------------------

class MultiAgentDataConfig:
    """双 Agent Dataset 配置。

    只负责数据读取，不包含优化器或模型超参。
    """

    train_json: str
    n_waypoints: int = 8
    history: int = 31
    cache_root: Optional[str] = None
    action_dims: int = 3
    online_encode_missing: bool = False
    image_size: int = 384
    vision_resize_mode: str = "letterbox"
    use_roi_tokens: bool = False
    roi_token_count: int = 16
    roi_expand_ratio: float = 1.5
    roi_make_square: bool = True
    roi_bbox_source: str = "ground_truth"
    allow_legacy_roi_cache_without_source_bbox: bool = False
    use_visual_section_markers: bool = False
    use_bbox_text_prompt: bool = False
    coarse_cache_size: int = 0
    # Training samplers may rotate through adjacent frames across epochs.
    temporal_stride: int = 1
    global_image_only: bool = False
    require_recorded_waypoints: bool = False


class MultiAgentJsonDataset(Dataset):
    """双 Agent JSONL Dataset。

    每条样本的数据流：
    1. 从 JSONL 读取 agent1/agent2 的历史帧、当前帧和 waypoints。
    2. 历史帧读取 *_vcoarse.pt，并整理成 (2, history*4, C)。
    3. 当前帧读取 *_vfine.pt，并整理成 (2, 64, C)。
    4. waypoints/valid_mask stack 成双 Agent 维度。

    __getitem__ 输出的关键 shape：
    - coarse_tokens: (2, history*4, C)
    - coarse_tidx:   (2, history*4)
    - fine_tokens:   (2, 64, C)
    - fine_tidx:     (2, 64)
    - roi_tokens:    (2, roi_token_count, C), only when use_roi_tokens=True
    - roi_tidx:      (2, roi_token_count), only when use_roi_tokens=True
    - roi_valid:     (2,), only when use_roi_tokens=True
    - waypoints:     (2, n_waypoints, 3)
    """

    def __init__(self, cfg: MultiAgentDataConfig):
        super().__init__()
        self.cfg = cfg
        self.train_path = Path(cfg.train_json).resolve()
        self.base_root = find_multi_agent_base_root(self.train_path)
        self.cache_root = Path(cfg.cache_root).resolve() if cfg.cache_root else (self.base_root / "vision_cache")
        self._online_encoder: Optional[VisionFeatureCacher] = None
        self._lazy = False
        self._index: Any = None
        self._files: Optional[List[str]] = None
        self.examples: Optional[List[Dict[str, Any]]] = None
        # 每个 DataLoader worker 独立维护历史粗 token 缓存，避免相邻样本反复 torch.load。
        # 当前帧 fine token 较大且复用率低，不放入该缓存。
        self._coarse_token_cache: OrderedDict[str, torch.Tensor] = OrderedDict()

        # 支持三种输入：dataset.json、单个 .jsonl、jsonl 目录。
        if self.train_path.is_file() and self.train_path.suffix.lower() == ".json":
            with self.train_path.open("r", encoding="utf-8") as f:
                obj = json.load(f)
            if not isinstance(obj, list):
                raise ValueError(f"JSON file must contain a list: {self.train_path}")
            self.examples = [x for x in obj if isinstance(x, dict)]
        else:
            if self.train_path.is_file() and self.train_path.suffix.lower() == ".jsonl":
                files = [self.train_path]
            elif self.train_path.is_dir():
                files = sorted(self.train_path.rglob("*.jsonl"))
            else:
                raise FileNotFoundError(f"Path does not exist or is unsupported: {self.train_path}")
            if not files:
                raise FileNotFoundError(f"No JSONL files found under: {self.train_path}")
            self._lazy = True
            self._files = [str(path) for path in files]
            self._index = _load_or_build_jsonl_index(files, self.train_path)

    def __len__(self) -> int:
        if self.examples is not None:
            return len(self.examples)
        return 0 if self._index is None else len(self._index)

    def _read_indexed_example(self, idx: int) -> Dict[str, Any]:
        """按 byte offset 懒读取 JSONL 中的一条样本。"""
        assert self._index is not None
        source, offset = self._index[idx]
        fp = self._files[int(source)] if self._files is not None else source
        with open(fp, "rb") as f:
            f.seek(offset)
            line = f.readline()
        return json.loads(line.decode("utf-8"))

    def get_example(self, idx: int) -> Dict[str, Any]:
        """返回原始 JSON dict，sanity check 和 auto alpha 会使用。"""
        if self._lazy:
            return self._read_indexed_example(idx)
        assert self.examples is not None
        return self.examples[idx]

    def _get_online_encoder(self) -> VisionFeatureCacher:
        """懒加载在线视觉编码器。

        正常流程应先运行 `python -m tools.precache_frames --multi_agent`。
        只有 --online_encode_missing=True 且 cache 缺失时才会走这里。
        """
        if self._online_encoder is None:
            use_cuda = torch.cuda.is_available()
            self._online_encoder = VisionFeatureCacher(
                VisionCacheConfig(
                    image_size=self.cfg.image_size,
                    batch_size=8,
                    device="cuda" if use_cuda else "cpu",
                    resize_mode=self.cfg.vision_resize_mode,
                )
            )
            self._online_encoder.eval()
        return self._online_encoder

    @torch.inference_mode()
    def _encode_image_tokens(self, img_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        """在线编码单张图片，输出 coarse/fine token。"""
        enc = self._get_online_encoder()
        pil = Image.open(str(img_path)).convert("RGB")
        tok_dino, hp, wp = enc._encode_dino([pil])
        tok_sigl = enc._encode_siglip([pil], out_hw=(hp, wp))
        tokens = torch.cat([tok_dino, tok_sigl], dim=-1)
        vfine = grid_pool_tokens(tokens, hp, wp, out_tokens=64)
        vcoarse = grid_pool_tokens(tokens, hp, wp, out_tokens=4)
        return vcoarse[0].cpu().float(), vfine[0].cpu().float()

    @torch.inference_mode()
    def _encode_roi_tokens(
        self,
        img_path: Path,
        bbox: Any,
        bbox_format: str,
    ) -> Tuple[torch.Tensor, bool, Tuple[int, int, int, int]]:
        enc = self._get_online_encoder()
        with Image.open(str(img_path)) as img:
            pil = img.convert("RGB")
        roi_img, roi_valid, crop_xyxy = crop_target_roi(
            pil,
            bbox,
            bbox_format,
            expand_ratio=float(self.cfg.roi_expand_ratio),
            make_square=bool(self.cfg.roi_make_square),
        )
        try:
            tokens = enc.encode_pooled_tokens([roi_img], int(self.cfg.roi_token_count))[0].cpu().float()
        finally:
            roi_img.close()
        return tokens, bool(roi_valid), crop_xyxy

    def _load_or_encode_tokens(self, frame_path: Path, fine: bool) -> torch.Tensor:
        """优先读取 vision_cache，必要时在线编码并回写。

        fine=True 读取当前帧 64-token；fine=False 读取历史帧 4-token。
        """
        vc_path, vf_path = multi_agent_token_paths_for_frame(self.base_root, self.cache_root, frame_path)
        wanted = vf_path if fine else vc_path
        cache_key = str(wanted)
        if not fine and self.cfg.coarse_cache_size > 0:
            cached = self._coarse_token_cache.get(cache_key)
            if cached is not None:
                self._coarse_token_cache.move_to_end(cache_key)
                return cached
        try:
            tokens = load_tokens_file(cache_key)
        except Exception:
            if not self.cfg.online_encode_missing:
                raise FileNotFoundError(
                    f"Missing token file: {wanted}. Run `python -m tools.precache_frames --multi_agent` "
                    "or pass --online_encode_missing."
                )
            vc, vf = self._encode_image_tokens(frame_path)
            vc_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                torch.save(vc.half(), str(vc_path))
                torch.save(vf.half(), str(vf_path))
            except Exception:
                torch.save(vc, str(vc_path))
                torch.save(vf, str(vf_path))
            tokens = vf if fine else vc

        if not fine and self.cfg.coarse_cache_size > 0:
            self._coarse_token_cache[cache_key] = tokens
            self._coarse_token_cache.move_to_end(cache_key)
            while len(self._coarse_token_cache) > self.cfg.coarse_cache_size:
                self._coarse_token_cache.popitem(last=False)
        return tokens

    def _roi_cache_path_for_frame(self, frame_path: Path) -> Path:
        _vc_path, vf_path = multi_agent_token_paths_for_frame(self.base_root, self.cache_root, frame_path)
        return vf_path.with_name(f"{vf_path.name[:-len('_vfine.pt')]}_vroi.pt")

    def _agent_roi_bbox(self, ex: Dict[str, Any], agent_idx: int) -> Tuple[Any, str]:
        if str(self.cfg.roi_bbox_source) != "ground_truth":
            raise NotImplementedError(
                f"roi_bbox_source={self.cfg.roi_bbox_source!r} is not implemented; "
                "currently only ground_truth oracle ROI is supported."
            )
        prefix = f"agent{agent_idx}_"
        bbox = ex.get(prefix + "bbox")
        if bbox is None:
            order = ex.get("agent_order", [ex.get("agent1_name", "drone"), ex.get("agent2_name", "robotdog")])
            agents = ex.get("agents", {})
            name = order[agent_idx - 1] if isinstance(order, list) and len(order) >= agent_idx else f"agent{agent_idx}"
            payload = agents.get(name, {}) if isinstance(agents, dict) else {}
            bbox = payload.get("bbox")
        return bbox, "cxcywh_norm"

    def _load_agent_roi_tokens(self, ex: Dict[str, Any], agent_idx: int, current_path: Path) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bbox, bbox_format = self._agent_roi_bbox(ex, agent_idx)
        roi_path = self._roi_cache_path_for_frame(current_path)
        try:
            payload = load_roi_cache(
                roi_path,
                roi_token_count=int(self.cfg.roi_token_count),
                roi_expand_ratio=float(self.cfg.roi_expand_ratio),
                roi_make_square=bool(self.cfg.roi_make_square),
                expected_bbox=bbox,
                allow_legacy_missing_source_bbox=bool(
                    self.cfg.allow_legacy_roi_cache_without_source_bbox
                ),
            )
            tokens = payload["tokens"].float()
            roi_valid = bool(payload.get("roi_valid", False))
        except Exception as exc:
            if not self.cfg.online_encode_missing:
                raise FileNotFoundError(
                    f"Missing or incompatible ROI token cache: {roi_path}. "
                    "Run `python -m tools.precache_frames --multi_agent --with_roi` "
                    "with matching roi_token_count/roi_expand_ratio/roi_make_square."
                ) from exc
            tokens, roi_valid, crop_xyxy = self._encode_roi_tokens(current_path, bbox, bbox_format)
            roi_path.parent.mkdir(parents=True, exist_ok=True)
            payload = build_roi_cache_payload(
                tokens,
                roi_token_count=int(self.cfg.roi_token_count),
                roi_expand_ratio=float(self.cfg.roi_expand_ratio),
                roi_make_square=bool(self.cfg.roi_make_square),
                bbox_format=bbox_format,
                roi_valid=bool(roi_valid),
                crop_xyxy=crop_xyxy,
                source_bbox=bbox,
            )
            torch.save(payload, str(roi_path))
        if tokens.size(0) != int(self.cfg.roi_token_count):
            raise ValueError(
                f"ROI token count mismatch for {roi_path}: got {tokens.size(0)}, "
                f"expected {self.cfg.roi_token_count}"
            )
        roi_tidx = torch.full((tokens.size(0),), fill_value=int(self.cfg.history), dtype=torch.long)
        return tokens.float(), roi_tidx, torch.tensor(bool(roi_valid), dtype=torch.bool)

    def _load_agent_tokens(self, ex: Dict[str, Any], agent_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
        """读取一个 Agent 的历史粗 token 和当前细 token。

        agent_idx=1 -> agent1_* 字段；agent_idx=2 -> agent2_* 字段。
        历史帧不足 history 时，用当前帧 coarse token 做左侧 padding，保持固定长度。
        """
        history = int(self.cfg.history)
        prefix = f"agent{agent_idx}_"
        current_rel = ex.get(prefix + "current")
        if not isinstance(current_rel, str) or not current_rel:
            raise KeyError(f"Missing {prefix}current in sample")
        current_path = resolve_multi_agent_path(self.base_root, current_rel)
        fine_tokens = self._load_or_encode_tokens(current_path, fine=True)
        fine_tidx = torch.full((fine_tokens.size(0),), fill_value=history, dtype=torch.long)

        image_rels = ex.get(prefix + "images", [])
        if not isinstance(image_rels, list):
            image_rels = []
        image_rels = image_rels[-history:]
        real_history: List[torch.Tensor] = []
        for img_rel in image_rels:
            if isinstance(img_rel, str) and img_rel:
                img_path = resolve_multi_agent_path(self.base_root, img_rel)
                real_history.append(self._load_or_encode_tokens(img_path, fine=False))

        if real_history:
            coarse_list = [real_history[0]] * (history - len(real_history)) + real_history
        else:
            current_coarse = self._load_or_encode_tokens(current_path, fine=False)
            coarse_list = [current_coarse] * history
        tidx_list = [
            torch.full((tok.size(0),), fill_value=t, dtype=torch.long)
            for t, tok in enumerate(coarse_list)
        ]

        return (
            torch.cat(coarse_list, dim=0),
            torch.cat(tidx_list, dim=0),
            fine_tokens,
            fine_tidx,
            str(current_path),
        )


    def _load_targets(self, ex: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """读取 JSON 中已记录的两个 Agent waypoint 和 valid_mask 标签。

        global-image base 不允许由 actions 现场积分、插值或差分重建标签；每条
        JSONL 样本必须已经带有完整的 ``recorded_pose_fixed_dt`` waypoint。
        """
        if self.cfg.require_recorded_waypoints:
            if ex.get("waypoint_label_source") != "recorded_pose_fixed_dt":
                raise ValueError(
                    "Global-image base requires JSON waypoints with "
                    "waypoint_label_source='recorded_pose_fixed_dt'."
                )
        if "waypoints" in ex:
            wp_all = ex["waypoints"]
        else:
            wp_all = [ex.get("agent1_waypoints"), ex.get("agent2_waypoints")]
        if "valid_mask" in ex:
            mask_all = ex["valid_mask"]
        else:
            mask_all = [ex.get("agent1_valid_mask"), ex.get("agent2_valid_mask")]

        out_wp = []
        out_mask = []
        for idx in range(2):
            wp_i = wp_all[idx] if isinstance(wp_all, list) and len(wp_all) > idx else None
            vm_i = mask_all[idx] if isinstance(mask_all, list) and len(mask_all) > idx else None
            if wp_i is None:
                raise KeyError(f"Missing waypoints for agent {idx + 1}")
            if self.cfg.require_recorded_waypoints:
                wp_tensor = torch.as_tensor(wp_i)
                vm_tensor = torch.as_tensor(vm_i, dtype=torch.bool) if vm_i is not None else None
                expected = (self.cfg.n_waypoints, self.cfg.action_dims)
                if tuple(wp_tensor.shape) != expected:
                    raise ValueError(
                        f"Global-image base requires recorded waypoint shape {expected}, "
                        f"got {tuple(wp_tensor.shape)} for agent {idx + 1}."
                    )
                if vm_tensor is None or tuple(vm_tensor.shape) != (self.cfg.n_waypoints,):
                    raise ValueError(
                        "Global-image base requires a valid_mask with shape "
                        f"({self.cfg.n_waypoints},) for agent {idx + 1}."
                    )
            wp, vm = fit_multi_agent_waypoints_and_mask(wp_i, vm_i, self.cfg.n_waypoints, self.cfg.action_dims)
            out_wp.append(wp)
            out_mask.append(vm)
        return torch.stack(out_wp, dim=0), torch.stack(out_mask, dim=0)

    def _load_inverse_control(
        self, ex: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load command-rollout state and inverse-control supervision."""
        state = torch.as_tensor(
            ex.get("inverse_control_state_before", [[0.0] * 3, [0.0] * 3]),
            dtype=torch.float32,
        )
        reference = torch.as_tensor(
            ex.get("inverse_control_reference_before", [[0.0] * 3, [0.0] * 3]),
            dtype=torch.float32,
        )
        target = torch.as_tensor(
            ex.get("inverse_control_target", [[0.0] * 3, [0.0] * 3]),
            dtype=torch.float32,
        )
        valid = torch.as_tensor(
            ex.get("inverse_control_valid_mask", [[False] * 3, [False] * 3]),
            dtype=torch.bool,
        )
        for name, value in (
            ("inverse_control_state_before", state),
            ("inverse_control_reference_before", reference),
            ("inverse_control_target", target),
            ("inverse_control_valid_mask", valid),
        ):
            if value.shape != (2, 3):
                raise ValueError(f"Expected {name} shape (2,3), got {tuple(value.shape)}")
        return state, reference, target, valid

    def _load_bbox(self, ex: Dict[str, Any]) -> torch.Tensor:
        """读取 bbox_feat，输出固定 shape (2, 4)。"""
        if "bbox_feat" in ex:
            bbox = ex["bbox_feat"]
        else:
            bbox = [ex.get("agent1_bbox", [0, 0, 0, 0]), ex.get("agent2_bbox", [0, 0, 0, 0])]
        t = torch.as_tensor(bbox, dtype=torch.float32)
        if t.shape != (2, 4):
            raise ValueError(f"Expected bbox_feat shape (2,4), got {tuple(t.shape)}")
        return t.clamp(0.0, 1.0)

    def _load_bbox_valid_mask(self, ex: Dict[str, Any], bbox_feat: torch.Tensor) -> torch.Tensor:
        """Load per-agent GT bbox validity, deriving it for legacy JSONL."""
        value = ex.get("bbox_valid_mask")
        if value is None:
            order = ex.get("agent_order", [ex.get("agent1_name", "drone"), ex.get("agent2_name", "robotdog")])
            agents = ex.get("agents", {})
            if isinstance(agents, dict):
                candidate = []
                for index in range(2):
                    name = order[index] if isinstance(order, list) and len(order) > index else f"agent{index + 1}"
                    payload = agents.get(name, {})
                    if not isinstance(payload, dict) or "bbox_valid_mask" not in payload:
                        candidate = []
                        break
                    candidate.append(bool(payload["bbox_valid_mask"]))
                if len(candidate) == 2:
                    value = candidate
        if value is None:
            value = torch.isfinite(bbox_feat).all(dim=-1) & (bbox_feat[:, 2] > 0.0) & (bbox_feat[:, 3] > 0.0)
        mask = torch.as_tensor(value, dtype=torch.bool)
        if mask.shape != (2,):
            raise ValueError(f"Expected bbox_valid_mask shape (2,), got {tuple(mask.shape)}")
        return mask

    def _load_bbox_prompt_text(self, ex: Dict[str, Any], bbox_feat: torch.Tensor) -> str:
        """Load or synthesize the per-step bbox spatial prompt text."""
        if not self.cfg.use_bbox_text_prompt:
            return ""
        prompt = ex.get("bbox_prompt_text")
        if isinstance(prompt, str) and prompt.strip():
            return prompt.strip()

        order = ex.get("agent_order", [ex.get("agent1_name", "drone"), ex.get("agent2_name", "robotdog")])
        if not isinstance(order, list) or len(order) < 2:
            order = [ex.get("agent1_name", "drone"), ex.get("agent2_name", "robotdog")]

        spatials = ex.get("bbox_spatial")
        if not (isinstance(spatials, list) and len(spatials) >= 2):
            spatials = [ex.get("agent1_bbox_spatial"), ex.get("agent2_bbox_spatial")]
        if not all(isinstance(item, dict) for item in spatials[:2]):
            bbox_list = bbox_feat.detach().cpu().tolist()
            spatials = [bbox_spatial_fields(bbox_list[0]), bbox_spatial_fields(bbox_list[1])]
        return bbox_prompt_from_spatial(spatials[:2], order[:2])

    def _load_visible(self, ex: Dict[str, Any]) -> torch.Tensor:
        """读取两个 Agent 的 target_visible，作为可选 visibility loss 目标。"""
        order = ex.get("agent_order", [ex.get("agent1_name", "drone"), ex.get("agent2_name", "robotdog")])
        agents = ex.get("agents", {})
        vals = []
        for idx in range(2):
            name = order[idx] if isinstance(order, list) and len(order) > idx else f"agent{idx + 1}"
            payload = agents.get(name, {}) if isinstance(agents, dict) else {}
            vals.append(1.0 if bool(payload.get("target_visible", True)) else 0.0)
        return torch.tensor(vals, dtype=torch.float32)

    def _load_relative_pose(self, ex: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """读取 pose/target_pose，输出两个 Agent 的相对目标空间量。"""
        order = ex.get("agent_order", [ex.get("agent1_name", "drone"), ex.get("agent2_name", "robotdog")])
        agents = ex.get("agents", {})
        rels = []
        masks = []
        for idx in range(2):
            name = order[idx] if isinstance(order, list) and len(order) > idx else f"agent{idx + 1}"
            payload = agents.get(name, {}) if isinstance(agents, dict) else {}
            agent_pose = payload.get("pose")
            target_pose = payload.get("target_pose")
            if agent_pose is None:
                agent_pose = ex.get(f"agent{idx + 1}_pose")
            if target_pose is None:
                target_pose = ex.get("target_pose")
            rel, valid = relative_target_pose_from_unreal_poses(agent_pose, target_pose)
            rels.append(rel)
            masks.append(valid)
        return torch.stack(rels, dim=0), torch.stack(masks, dim=0)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """读取并组装一条双 Agent 训练样本。"""
        ex = self.get_example(idx)
        a1 = self._load_agent_tokens(ex, agent_idx=1)
        a2 = self._load_agent_tokens(ex, agent_idx=2)
        waypoints, valid_mask = self._load_targets(ex)
        history = int(self.cfg.history)

        sample = {
            "coarse_tokens": torch.stack([a1[0], a2[0]], dim=0),
            "coarse_tidx": torch.stack([a1[1], a2[1]], dim=0),
            "fine_tokens": torch.stack([a1[2], a2[2]], dim=0),
            "fine_tidx": torch.stack([a1[3], a2[3]], dim=0),
            "yaw_hist": torch.zeros(2, history, dtype=torch.float32),
            "yaw_curr": torch.zeros(2, 1, dtype=torch.float32),
            "waypoints": waypoints,
            "valid_mask": valid_mask,
            "dt": float(ex.get("dt", 0.1)),
            "instruction": ex.get("instruction", "Follow the target person without collision."),
            "episode_id": ex.get("episode_id", ""),
            "step_index": int(ex.get("step_index", idx)),
            "current_path": [a1[4], a2[4]],
        }
        if not self.cfg.global_image_only:
            inverse_state, inverse_reference, inverse_target, inverse_valid = self._load_inverse_control(ex)
            bbox_feat = self._load_bbox(ex)
            bbox_valid_mask = self._load_bbox_valid_mask(ex, bbox_feat)
            relative_pose, relative_pose_valid = self._load_relative_pose(ex)
            sample.update(
                {
                    "bbox_feat": bbox_feat,
                    "bbox_valid_mask": bbox_valid_mask,
                    "bbox_prompt_text": self._load_bbox_prompt_text(ex, bbox_feat),
                    "visible": self._load_visible(ex),
                    "relative_pose": relative_pose,
                    "relative_pose_valid": relative_pose_valid,
                    "inverse_control_state_before": inverse_state,
                    "inverse_control_reference_before": inverse_reference,
                    "inverse_control_target": inverse_target,
                    "inverse_control_valid_mask": inverse_valid,
                }
            )
        if self.cfg.use_roi_tokens:
            r1 = self._load_agent_roi_tokens(ex, 1, Path(a1[4]))
            r2 = self._load_agent_roi_tokens(ex, 2, Path(a2[4]))
            sample.update(
                {
                    "roi_tokens": torch.stack([r1[0], r2[0]], dim=0),
                    "roi_tidx": torch.stack([r1[1], r2[1]], dim=0),
                    "roi_valid": torch.stack([r1[2], r2[2]], dim=0),
                }
            )
        return sample


# ----------------------- 双 Agent batch 与数据检查 -----------------------


class LocalityAwareDistributedSampler(Sampler[int]):
    """Shuffle trajectory blocks while keeping nearby frames adjacent.

    A sample reads up to ``history`` overlapping frame-token files. Shuffling
    individual samples destroys that overlap and turns training into a random
    small-file workload. This sampler shuffles episode-local blocks instead;
    batches inside each block remain sequential, so a DataLoader worker can
    reuse its in-process coarse-token cache.

    The global order is padded and split into equal contiguous rank shards,
    preserving DDP's equal-step requirement without interleaving adjacent
    samples across ranks.
    """

    def __init__(
        self,
        dataset: Dataset,
        block_size: int,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if num_replicas <= 0 or not 0 <= rank < num_replicas:
            raise ValueError(f"Invalid distributed sampler rank={rank}/{num_replicas}")
        self.dataset = dataset
        self.block_size = int(block_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        self.num_samples = math.ceil(len(dataset) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        self._blocks = self._build_blocks()

    def _build_blocks(self) -> List[List[int]]:
        # The lazy index is sorted by JSONL file, with one file per episode.
        # Never merge a block across an episode boundary.
        lazy_index = getattr(self.dataset, "_index", None)
        groups: List[List[int]] = []
        if lazy_index:
            start = 0
            while start < len(lazy_index):
                source = lazy_index[start][0]
                end = start + 1
                while end < len(lazy_index) and lazy_index[end][0] == source:
                    end += 1
                groups.append(list(range(start, end)))
                start = end
        else:
            groups = [list(range(len(self.dataset)))]
        return [
            group[offset : offset + self.block_size]
            for group in groups
            for offset in range(0, len(group), self.block_size)
        ]

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        if self.shuffle:
            order = torch.randperm(len(self._blocks), generator=generator).tolist()
        else:
            order = list(range(len(self._blocks)))
        indices = [index for block_id in order for index in self._blocks[block_id]]
        if len(indices) < self.total_size:
            if not indices:
                return iter(())
            padding = self.total_size - len(indices)
            indices += (indices * math.ceil(padding / len(indices)))[:padding]
        else:
            indices = indices[: self.total_size]
        start = self.rank * self.num_samples
        return iter(indices[start : start + self.num_samples])

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


def collate_multi_agent_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """把双 Agent 单样本合成 batch。

    张量字段 stack；instruction/current_path/episode_id 保留 Python list 给 tokenizer 和日志使用。
    """
    out = {
        "coarse_tokens": torch.stack([b["coarse_tokens"] for b in batch], dim=0),
        "coarse_tidx": torch.stack([b["coarse_tidx"] for b in batch], dim=0),
        "fine_tokens": torch.stack([b["fine_tokens"] for b in batch], dim=0),
        "fine_tidx": torch.stack([b["fine_tidx"] for b in batch], dim=0),
        "yaw_hist": torch.stack([b["yaw_hist"] for b in batch], dim=0),
        "yaw_curr": torch.stack([b["yaw_curr"] for b in batch], dim=0),
        "waypoints": torch.stack([b["waypoints"] for b in batch], dim=0),
        "valid_mask": torch.stack([b["valid_mask"] for b in batch], dim=0),
        "dt": torch.tensor([b["dt"] for b in batch], dtype=torch.float32),
        "instruction": [b["instruction"] for b in batch],
        "episode_id": [b["episode_id"] for b in batch],
        "step_index": torch.tensor([b["step_index"] for b in batch], dtype=torch.long),
        "current_path": [b["current_path"] for b in batch],
    }
    if "bbox_feat" in batch[0]:
        out.update(
            {
                "bbox_feat": torch.stack([b["bbox_feat"] for b in batch], dim=0),
                "bbox_valid_mask": torch.stack([b["bbox_valid_mask"] for b in batch], dim=0),
                "visible": torch.stack([b["visible"] for b in batch], dim=0),
                "relative_pose": torch.stack([b["relative_pose"] for b in batch], dim=0),
                "relative_pose_valid": torch.stack([b["relative_pose_valid"] for b in batch], dim=0),
                "inverse_control_state_before": torch.stack(
                    [b["inverse_control_state_before"] for b in batch], dim=0
                ),
                "inverse_control_reference_before": torch.stack(
                    [b["inverse_control_reference_before"] for b in batch], dim=0
                ),
                "inverse_control_target": torch.stack([b["inverse_control_target"] for b in batch], dim=0),
                "inverse_control_valid_mask": torch.stack(
                    [b["inverse_control_valid_mask"] for b in batch], dim=0
                ),
                "bbox_prompt_text": [b.get("bbox_prompt_text", "") for b in batch],
            }
        )
    if "roi_tokens" in batch[0]:
        out["roi_tokens"] = torch.stack([b["roi_tokens"] for b in batch], dim=0)
        out["roi_tidx"] = torch.stack([b["roi_tidx"] for b in batch], dim=0)
        out["roi_valid"] = torch.stack([b["roi_valid"] for b in batch], dim=0)
    return out


def multi_agent_dataset_sanity_report(ds: MultiAgentJsonDataset, cfg: "MultiAgentTrainConfig", max_items: int = 256) -> None:
    """训练前检查双 Agent 数据分布和 cache 覆盖率。"""
    try:
        n = min(max_items, len(ds))
        # n = max(max_items, len(ds))
        xs, ys, ths = [], [], []
        bbox_vals = []
        img_ok = 0
        cache_ok = 0
        relpose_valid = 0
        for i in range(n):
            ex = ds.get_example(i)
            wp = np.asarray(ex.get("waypoints", []), dtype=np.float32)
            if wp.ndim == 3 and wp.shape[-1] >= 3:
                xs.append(wp[..., 0].reshape(-1))
                ys.append(wp[..., 1].reshape(-1))
                ths.append(wp[..., 2].reshape(-1))
            if not cfg.base_model:
                bbox = np.asarray(ex.get("bbox_feat", []), dtype=np.float32)
                if bbox.shape == (2, 4):
                    bbox_vals.append(bbox.reshape(-1))
                try:
                    _, rel_mask = ds._load_relative_pose(ex)
                    relpose_valid += int(rel_mask.sum().item())
                except Exception:
                    pass
            for key in ("agent1_current", "agent2_current"):
                cur = ex.get(key)
                if isinstance(cur, str):
                    p = resolve_multi_agent_path(ds.base_root, cur)
                    if p.exists():
                        img_ok += 1
                    vc, vf = multi_agent_token_paths_for_frame(ds.base_root, ds.cache_root, p)
                    if vc.exists() and vf.exists():
                        cache_ok += 1
        x_std = float(np.std(np.concatenate(xs))) if xs else float("nan")
        y_std = float(np.std(np.concatenate(ys))) if ys else float("nan")
        th_std = float(np.std(np.concatenate(ths))) if ths else float("nan")
        extra = (
            f"bbox_mean={float(np.mean(np.concatenate(bbox_vals))):.4f} "
            f"relpose_valid={relpose_valid}/{2*n} "
            if not cfg.base_model
            else "global_image_only=True "
        )
        print(
            f"[SANITY] samples_checked={n} xy_std=({x_std:.4f},{y_std:.4f}) theta_std={th_std:.4f} "
            f"{extra}"
            f"current_img_ok={img_ok}/{2*n} current_cache_ok={cache_ok}/{2*n}",
            flush=True,
        )
    except Exception as exc:
        print(f"[SANITY] skipped due to error: {exc}", flush=True)


@dataclass
# ----------------------- 双 Agent训练配置与模型构建 -----------------------

class MultiAgentTrainConfig:
    """V3 双 Agent 训练循环的通用配置字段。"""

    train_json: str
    out_dir: str = "/data/hdt/ntv_data/ckpt/ckpts_multi_agent"
    val_json: Optional[str] = None
    val_cache_root: Optional[str] = None
    llm_name: str = "Qwen/Qwen3-0.6B"
    vision_feat_dim: int = 1536
    n_waypoints: int = 8
    action_dims: int = 3
    history: int = 31
    cache_root: Optional[str] = None
    online_encode_missing: bool = False
    image_size: int = 384
    vision_resize_mode: str = "letterbox"
    use_roi_tokens: bool = False
    roi_token_count: int = 16
    roi_expand_ratio: float = 1.5
    roi_make_square: bool = True
    roi_bbox_source: str = "ground_truth"
    allow_legacy_roi_cache_without_source_bbox: bool = False
    epochs: int = 1
    batch_size: int = 2
    grad_accum_steps: int = 1
    lr: float = 2e-5
    lr_scheduler: str = "constant"
    warmup_steps: int = 0
    min_lr: float = 0.0
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    mixed_precision: bool = True
    seed: int = 0
    num_workers: int = 4
    prefetch_factor: int = 2
    coarse_cache_size: int = 0
    shuffle_block_size: int = 0
    distributed: bool = False
    dist_backend: str = "nccl"
    manual_grad_allreduce: bool = False
    manual_grad_allreduce_cpu: bool = False
    ddp_timeout_minutes: int = 120
    ddp_find_unused_parameters: bool = True
    base_model: bool = False
    # V3 writes its architecture marker into every checkpoint.
    base_model_architecture: str = ""
    separate_agent_context: bool = False
    use_agent_text_markers: bool = True
    use_visual_section_markers: bool = False
    text_max_length: int = 128
    instruction_override: Optional[str] = None
    joint_instruction_override: Optional[str] = None
    agent1_instruction_override: Optional[str] = None
    agent2_instruction_override: Optional[str] = None
    use_bbox_text_prompt: bool = False
    bbox_text_dropout_prob: float = 0.0
    freeze_llm: bool = True
    use_angle_tvi: bool = False
    insert_time_tokens: bool = True
    no_tanh_actions: bool = True
    alpha_xy: Optional[float] = 1.0
    use_grounding: bool = True
    use_bbox_tokens: bool = True
    beta_nav: float = 100.0
    beta_control_drone: float = 0.0
    beta_control_dog: float = 0.0
    control_target_stats: Optional[str] = None
    control_smooth_l1_beta: float = 0.5
    control_scale_drone_forward: float = 6.0
    control_scale_drone_lateral: float = 6.0
    control_scale_drone_yaw: float = 4.0
    control_scale_dog_forward: float = 2.0
    control_scale_dog_yaw: float = 5.0
    inverse_waypoint_index: int = 1
    inverse_ground_translation_delay_steps: int = 1
    inverse_ground_yaw_gain: float = 0.4
    inverse_drone_a_forward: float = 0.969
    inverse_drone_b_forward: float = 0.0301
    inverse_drone_a_lateral: float = 0.969
    inverse_drone_b_lateral: float = 0.0301
    inverse_drone_yaw_a: float = 0.464
    inverse_drone_yaw_b: float = 0.359
    inverse_drone_xy_smoothing_alpha: float = 0.20
    inverse_drone_yaw_smoothing_alpha: float = 0.25
    inverse_robotdog_speed_smoothing_alpha: float = 0.30
    inverse_robotdog_yaw_smoothing_alpha: float = 0.30
    drone_loss_weight: float = 2.0
    dog_loss_weight: float = 1.0
    normalize_agent_loss_weights: bool = False
    nav_loss_type: str = "mse"
    smooth_l1_beta: float = 0.05
    yaw_loss_weight: float = 1.0
    dog_lateral_loss_weight: float = 0.0
    final_waypoint_loss_weight: float = 0.0
    turn_sample_weight: float = 1.0
    turn_rate_threshold: float = 0.1
    turn_angle_threshold: float = 0.08
    stop_sample_weight: float = 1.0
    stop_speed_threshold: float = 0.15
    stop_window: int = 3
    beta_bbox: float = 0.0
    beta_visible: float = 0.0
    beta_relative_pose: float = 0.0
    bbox_dropout_prob: float = 0.0
    return_token_logits: bool = False
    log_every: int = 10
    csv_logging: bool = True
    progress: bool = True
    save_every: int = 100
    save_every_epochs: int = 1
    max_ckpts: int = 0
    max_steps: int = 0
    eval_every: int = 0
    eval_batches: int = 8
    val_bbox_source: str = "none"
    final_wp_threshold: float = 0.2
    resume: bool = False
    resume_ckpt: Optional[str] = None
    # Initialize model weights only; optimizer/scheduler start fresh. Useful
    # when adding small heads to an otherwise compatible architecture.
    init_ckpt: Optional[str] = None
    dry_run: bool = False


def apply_training_defaults(cfg: MultiAgentTrainConfig) -> MultiAgentTrainConfig:
    """Validate runtime fields before the V3 callbacks are installed."""
    if cfg.lr_scheduler not in {"constant", "cosine"}:
        raise ValueError(f"Unsupported lr_scheduler={cfg.lr_scheduler!r}")
    if cfg.lr <= 0.0 or cfg.min_lr < 0.0 or cfg.min_lr > cfg.lr:
        raise ValueError("Require lr>0 and 0<=min_lr<=lr")
    if cfg.warmup_steps < 0 or cfg.max_steps < 0:
        raise ValueError("warmup_steps and max_steps must be non-negative")
    if cfg.normalize_agent_loss_weights and (
        float(cfg.drone_loss_weight) + float(cfg.dog_loss_weight) <= 0.0
    ):
        raise ValueError("Normalized agent loss requires positive weights")
    return cfg


def build_multi_agent_dataset(
    path: str,
    cfg: MultiAgentTrainConfig,
    cache_root: Optional[str] = None,
) -> MultiAgentJsonDataset:
    """根据训练配置创建双 Agent Dataset。"""
    return MultiAgentJsonDataset(
        MultiAgentDataConfig(
            train_json=path,
            n_waypoints=cfg.n_waypoints,
            history=cfg.history,
            cache_root=cfg.cache_root if cache_root is None else cache_root,
            action_dims=cfg.action_dims,
            online_encode_missing=cfg.online_encode_missing,
            image_size=cfg.image_size,
            vision_resize_mode=cfg.vision_resize_mode,
            use_roi_tokens=cfg.use_roi_tokens,
            roi_token_count=cfg.roi_token_count,
            roi_expand_ratio=cfg.roi_expand_ratio,
            roi_make_square=cfg.roi_make_square,
            roi_bbox_source=cfg.roi_bbox_source,
            allow_legacy_roi_cache_without_source_bbox=cfg.allow_legacy_roi_cache_without_source_bbox,
            use_bbox_text_prompt=cfg.use_bbox_text_prompt,
            coarse_cache_size=cfg.coarse_cache_size,
            global_image_only=cfg.base_model,
            require_recorded_waypoints=cfg.base_model,
        )
    )


def build_multi_agent_model(cfg: MultiAgentTrainConfig) -> nn.Module:
    """Callback slot replaced by train_airground_coop_v3 before training."""
    raise RuntimeError("AirGround V3 model callback was not installed")


def forward_multi_agent_loss(
    model: nn.Module,
    batch: Dict[str, Any],
    cfg: MultiAgentTrainConfig,
    device: torch.device,
):
    """Callback slot replaced by train_airground_coop_v3 before training."""
    raise RuntimeError("AirGround V3 loss callback was not installed")


def multi_agent_lr_for_step(
    step: int,
    total_steps: int,
    peak_lr: float,
    min_lr: float,
    warmup_steps: int,
) -> float:
    """Return the LR for a completed optimizer-step index.

    Step 0 is the initial LR. Warmup reaches ``peak_lr`` at ``warmup_steps``;
    the remaining optimizer steps follow a cosine curve to ``min_lr``.
    """
    total_steps = max(1, int(total_steps))
    step = min(max(0, int(step)), total_steps)
    warmup_steps = min(max(0, int(warmup_steps)), total_steps)
    peak_lr = float(peak_lr)
    min_lr = float(min_lr)
    if warmup_steps > 0 and step <= warmup_steps:
        return min_lr + (peak_lr - min_lr) * (float(step) / float(warmup_steps))
    if total_steps <= warmup_steps:
        return peak_lr
    progress = float(step - warmup_steps) / float(total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (peak_lr - min_lr) * cosine


def build_multi_agent_lr_scheduler(
    optim: torch.optim.Optimizer,
    cfg: MultiAgentTrainConfig,
    total_steps: int,
) -> Optional[torch.optim.lr_scheduler.LambdaLR]:
    """Build an optimizer-step scheduler; ``constant`` preserves old training."""
    if cfg.lr_scheduler == "constant":
        return None
    if cfg.lr_scheduler != "cosine":
        raise ValueError(f"Unsupported lr_scheduler={cfg.lr_scheduler!r}")

    def lr_lambda(step: int) -> float:
        return multi_agent_lr_for_step(
            step,
            total_steps,
            cfg.lr,
            cfg.min_lr,
            cfg.warmup_steps,
        ) / float(cfg.lr)

    return torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)


def save_multi_agent_checkpoint(
    path: Path,
    model: nn.Module,
    optim: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    scaler: torch.amp.GradScaler,
    cfg: MultiAgentTrainConfig,
    epoch: int,
    step: int,
) -> None:
    """保存双 Agent 训练状态，兼容 DDP/单卡加载。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    model_to_save = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    config_payload = {
        **cfg.__dict__,
        "model_type": multi_agent_model_type_name(cfg),
        "visual_layout_version": "global_plus_current_roi_v1" if cfg.use_roi_tokens else "global_only_v1",
        "visual_section_marker_version": (
            "global_history_current_target_roi_markers_v1"
            if cfg.use_visual_section_markers
            else None
        ),
        "roi_prompt_version": "roi_visual_layout_prompt_v1" if cfg.use_roi_tokens else None,
        "evaluation_protocol": "oracle_roi_upper_bound" if cfg.use_roi_tokens else None,
        "bbox_usage_note": (
            "roi_bbox is used only for image cropping; bbox_feat numeric model input is disabled."
            if cfg.use_roi_tokens
            else None
        ),
    }
    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "model_state": model_to_save.state_dict(),
            "optim_state": optim.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state": scaler.state_dict() if scaler.is_enabled() else None,
            "config": config_payload,
        },
        str(path),
    )


def multi_agent_model_type_name(cfg: MultiAgentTrainConfig) -> str:
    """用于 checkpoint/config/log 的双 Agent 模型类型名。"""
    if cfg.separate_agent_context:
        return "model_py_multi_agent_separate_base"
    if cfg.base_model:
        return "model_py_multi_agent_base"
    return "model_py_multi_agent"


def write_multi_agent_train_config(out_dir: Path, cfg: MultiAgentTrainConfig) -> None:
    """保存双 Agent 本次训练配置，方便后续复现实验。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {
        **cfg.__dict__,
        "model_type": multi_agent_model_type_name(cfg),
        "visual_layout_version": "global_plus_current_roi_v1" if cfg.use_roi_tokens else "global_only_v1",
        "visual_section_marker_version": (
            "global_history_current_target_roi_markers_v1"
            if cfg.use_visual_section_markers
            else None
        ),
        "roi_prompt_version": "roi_visual_layout_prompt_v1" if cfg.use_roi_tokens else None,
        "evaluation_protocol": "oracle_roi_upper_bound" if cfg.use_roi_tokens else None,
        "bbox_usage_note": (
            "roi_bbox is used only for image cropping; bbox_feat numeric model input is disabled."
            if cfg.use_roi_tokens
            else None
        ),
        "effective_batch_per_rank": int(cfg.batch_size) * max(1, int(cfg.grad_accum_steps)),
    }
    (out_dir / "train_config.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    lines = [
        "AirGround-Coop V3 training config",
        f"model_type: {data['model_type']}",
        f"train_json: {cfg.train_json}",
        f"cache_root: {cfg.cache_root}",
        f"batch_size_per_rank: {cfg.batch_size}",
        f"grad_accum_steps: {cfg.grad_accum_steps}",
        f"effective_batch_per_rank: {int(cfg.batch_size) * max(1, int(cfg.grad_accum_steps))}",
        f"lr: {cfg.lr}",
        f"lr_scheduler: {cfg.lr_scheduler}",
        f"warmup_steps: {cfg.warmup_steps}",
        f"min_lr: {cfg.min_lr}",
        f"beta_nav: {cfg.beta_nav}",
        f"drone_loss_weight: {cfg.drone_loss_weight}",
        f"dog_loss_weight: {cfg.dog_loss_weight}",
        f"normalize_agent_loss_weights: {cfg.normalize_agent_loss_weights}",
        f"nav_loss_type: {cfg.nav_loss_type}",
        f"smooth_l1_beta: {cfg.smooth_l1_beta}",
        f"yaw_loss_weight: {cfg.yaw_loss_weight}",
        f"final_waypoint_loss_weight: {cfg.final_waypoint_loss_weight}",
        f"turn_sample_weight: {cfg.turn_sample_weight}",
        f"turn_rate_threshold: {cfg.turn_rate_threshold}",
        f"turn_angle_threshold: {cfg.turn_angle_threshold}",
        f"stop_sample_weight: {cfg.stop_sample_weight}",
        f"stop_speed_threshold: {cfg.stop_speed_threshold}",
        f"stop_window: {cfg.stop_window}",
        f"instruction_override: {cfg.instruction_override}",
        f"joint_instruction_override: {cfg.joint_instruction_override}",
        f"agent1_instruction_override: {cfg.agent1_instruction_override}",
        f"agent2_instruction_override: {cfg.agent2_instruction_override}",
        f"use_agent_text_markers: {cfg.use_agent_text_markers}",
        f"alpha_xy: {cfg.alpha_xy}",
        f"no_tanh_actions: {cfg.no_tanh_actions}",
        f"use_grounding: {cfg.use_grounding}",
        f"use_bbox_tokens: {cfg.use_bbox_tokens}",
        f"use_roi_tokens: {cfg.use_roi_tokens}",
        f"roi_token_count: {cfg.roi_token_count}",
        f"roi_expand_ratio: {cfg.roi_expand_ratio}",
        f"roi_make_square: {cfg.roi_make_square}",
        f"roi_bbox_source: {cfg.roi_bbox_source}",
        f"use_visual_section_markers: {cfg.use_visual_section_markers}",
        f"visual_layout_version: {data['visual_layout_version']}",
        f"visual_section_marker_version: {data['visual_section_marker_version']}",
        f"roi_prompt_version: {data['roi_prompt_version']}",
        f"evaluation_protocol: {data['evaluation_protocol']}",
        f"bbox_usage_note: {data['bbox_usage_note']}",
    ]
    (out_dir / "train_config.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def prune_multi_agent_checkpoints(out_dir: Path, max_ckpts: int) -> None:
    """只保留最近 max_ckpts 个双 Agent checkpoint。"""
    if max_ckpts <= 0:
        return
    ckpts = sorted(out_dir.glob("model_epoch*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in ckpts[max_ckpts:]:
        try:
            old.unlink()
        except Exception:
            pass



def multi_agent_param_bucket(name: str) -> str:
    if name.startswith("llm."):
        return "LLM"
    if name.startswith("proj."):
        return "projector"
    if name.startswith(("tvi", "agent_embed", "view_embed", "kind_embed", "time_embed")):
        return "TVI embedding"
    if name in {"act_token_1", "act_token_2"}:
        return "ACT tokens"
    if name.startswith("planner_agent1."):
        return "planner_agent1"
    if name.startswith("planner_agent2."):
        return "planner_agent2"
    return "other"


def print_multi_agent_parameter_report(model: nn.Module) -> None:
    model_inspect = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    buckets = {
        "LLM": [0, 0, None],
        "projector": [0, 0, None],
        "TVI embedding": [0, 0, None],
        "ACT tokens": [0, 0, None],
        "planner_agent1": [0, 0, None],
        "planner_agent2": [0, 0, None],
        "other": [0, 0, None],
    }
    for name, parameter in model_inspect.named_parameters():
        bucket = multi_agent_param_bucket(name)
        total = int(parameter.numel())
        trainable = total if parameter.requires_grad else 0
        buckets[bucket][0] += total
        buckets[bucket][1] += trainable
        state = buckets[bucket][2]
        if state is None:
            buckets[bucket][2] = bool(parameter.requires_grad)
        elif state != bool(parameter.requires_grad):
            buckets[bucket][2] = "mixed"
    for bucket, (total, trainable, state) in buckets.items():
        print(
            f"[PARAMS][MULTI][{bucket}] total={int(total):,} trainable={int(trainable):,} "
            f"requires_grad={state}",
            flush=True,
        )
    if hasattr(model_inspect, "llm"):
        print(f"[PARAMS][MULTI][LLM] training={bool(model_inspect.llm.training)}", flush=True)


def manual_allreduce_trainable_grads(model: nn.Module, world_size: int, use_cpu: bool = False) -> None:
    """Average gradients for trainable parameters without wrapping the model in DDP.

    This avoids DDP construction-time
    synchronization of the frozen Qwen backbone, while still keeping optimizer
    updates equivalent to data-parallel training for the trainable heads.
    """
    if world_size <= 1:
        return
    if use_cpu:
        trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not trainable_params:
            return
        flat_parts: List[torch.Tensor] = []
        grad_meta: List[Tuple[torch.nn.Parameter, int, torch.dtype, torch.device]] = []
        for parameter in trainable_params:
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            grad = parameter.grad.detach()
            numel = grad.numel()
            flat_parts.append(grad.float().cpu().reshape(-1))
            grad_meta.append((parameter, numel, grad.dtype, grad.device))
        flat = torch.cat(flat_parts, dim=0)
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(float(world_size))
        offset = 0
        for parameter, numel, dtype, device in grad_meta:
            chunk = flat.narrow(0, offset, numel).view_as(parameter.grad)
            parameter.grad.copy_(chunk.to(device=device, dtype=dtype))
            offset += numel
        return
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        # Every rank must issue collectives in exactly the same order.  Some
        # optional/masked branches can legitimately produce grad=None on one
        # rank but not another; skipping such parameters would desynchronize the
        # NCCL all-reduce sequence and hang the first optimizer step.  Materialize
        # zero gradients so all trainable parameters participate on all ranks.
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(float(world_size))


def manual_broadcast_trainable_params(model: nn.Module, src: int = 0) -> Tuple[int, int]:
    """Broadcast only trainable parameters from rank 0 for manual all-reduce mode."""
    count = 0
    numel = 0
    for parameter in model.parameters():
        if parameter.requires_grad:
            dist.broadcast(parameter.data, src=src)
            count += 1
            numel += parameter.numel()
    return count, numel


def append_multi_agent_epoch_summary(
    path: Path,
    epoch: int,
    step: int,
    grad_preclip: List[float],
    grad_postclip: List[float],
    clip_scales: List[float],
) -> None:
    fields = [
        "epoch",
        "step",
        "clip_ratio",
        "grad_norm_preclip_mean",
        "grad_norm_preclip_p50",
        "grad_norm_preclip_p90",
        "grad_norm_preclip_p95",
        "grad_norm_preclip_max",
        "grad_norm_postclip_mean",
        "clip_scale_mean",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    arr = np.asarray(grad_preclip, dtype=np.float64)
    post = np.asarray(grad_postclip, dtype=np.float64) if grad_postclip else np.asarray([], dtype=np.float64)
    scales = np.asarray(clip_scales, dtype=np.float64) if clip_scales else np.asarray([], dtype=np.float64)
    if arr.size == 0:
        values = [epoch, step, float("nan"), *(float("nan") for _ in range(7))]
    else:
        values = [
            epoch,
            step,
            float(np.mean(arr > 0.0) if scales.size == 0 else np.mean(scales < 0.999999)),
            float(np.mean(arr)),
            float(np.percentile(arr, 50)),
            float(np.percentile(arr, 90)),
            float(np.percentile(arr, 95)),
            float(np.max(arr)),
            float(np.mean(post)) if post.size else float("nan"),
            float(np.mean(scales)) if scales.size else float("nan"),
        ]
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(fields)
        writer.writerow(values)


MULTI_AGENT_LOG_FIELDS = [
    "epoch",
    "step",
    "lr",
    "loss",
    "loss_ema",
    "loss_nav",
    "loss_nav_drone",
    "loss_nav_dog",
    "loss_nav_xy",
    "loss_nav_yaw",
    "loss_nav_final",
    "loss_control_drone",
    "loss_control_dog",
    "turn_fraction",
    "stop_fraction",
    "turn_fraction_drone",
    "turn_fraction_dog",
    "stop_fraction_drone",
    "stop_fraction_dog",
    "behavior_weight_mean",
    "regression_loss",
    "score_loss",
    "loss_bbox",
    "loss_visible",
    "loss_relative_pose",
    "loss_candidate_bce",
    "loss_candidate_rank",
    "candidate_top8_recall",
    "target_match_accuracy",
    "candidate_class_accuracy",
    "candidate_positive_probability",
    "candidate_max_negative_probability",
    "candidate_threshold_target_recall",
    "no_target_false_accept_rate",
    # Explicit V3 losses and curriculum diagnostics. Compatibility aliases
    # above are retained for older model families and historical logs.
    "loss_self",
    "loss_self_drone",
    "loss_self_dog",
    "loss_cooperative",
    "loss_coop_drone",
    "loss_coop_dog",
    "loss_mode",
    "loss_jepa",
    "loss_belief",
    "loss_target_match",
    "loss_uncertainty",
    "loss_smoothness",
    "loss_kinematics",
    "loss_diversity",
    "target_match_positive_fraction",
    "synthetic_drone_fraction",
    "synthetic_dog_fraction",
    "pose_perturbation_fraction",
    "roi_only_fraction",
    "current_full_fraction",
    "recent_full_fraction",
    "all_full_fraction",
    "coop_drone_fraction",
    "coop_dog_fraction",
    "final_epe",
    "final_epe_drone",
    "final_epe_dog",
    "grad_norm_preclip",
    "grad_norm_postclip",
    "was_clipped",
    "clip_scale",
    "grad_norm",
]


def prepare_multi_agent_csv_log(path: Path) -> None:
    """升级旧训练日志表头，为新增损失分量补 ``nan``，避免续训列错位。"""
    if not path.exists():
        return
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys()) if rows else []
    if fieldnames == MULTI_AGENT_LOG_FIELDS:
        return
    for row in rows:
        for key in MULTI_AGENT_LOG_FIELDS:
            row.setdefault(key, "nan")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MULTI_AGENT_LOG_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def evaluate_multi_agent(
    model: nn.Module,
    ds: MultiAgentJsonDataset,
    cfg: MultiAgentTrainConfig,
    device: torch.device,
) -> Dict[str, float]:
    """Validation for waypoint-only multi-agent training."""
    was_training = bool(model.training)
    model_eval = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    model_eval.eval()
    diffusion_planners = [
        planner
        for planner in (getattr(model_eval, "planner_agent1", None), getattr(model_eval, "planner_agent2", None))
        if planner is not None and hasattr(planner, "deterministic_inference")
    ]
    previous_deterministic = [bool(planner.deterministic_inference) for planner in diffusion_planners]
    for planner in diffusion_planners:
        planner.deterministic_inference = True
    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=min(2, cfg.num_workers),
        pin_memory=True,
        collate_fn=collate_multi_agent_batch,
    )
    total_loss = 0.0
    total_count = 0
    metric_sums: Dict[str, float] = {}
    metric_keys = [
        "loss_nav",
        "loss_nav_drone",
        "loss_nav_dog",
        "val_xy_mse_drone",
        "val_xy_mse_robotdog",
        "val_yaw_mse_drone",
        "val_yaw_mse_robotdog",
        "val_final_waypoint_error_drone",
        "val_final_waypoint_error_robotdog",
        "turn_fraction",
        "stop_fraction",
        "loss_candidate_bce",
        "loss_candidate_rank",
        "candidate_top8_recall",
        "target_match_accuracy",
        "candidate_class_accuracy",
        "candidate_positive_probability",
        "candidate_max_negative_probability",
        "candidate_threshold_target_recall",
        "no_target_false_accept_rate",
    ]
    final_errors: List[float] = []
    final_errors_drone: List[float] = []
    final_errors_dog: List[float] = []
    batches = 0
    for batch in dl:
        loss, metrics = forward_multi_agent_loss(model_eval, batch, cfg, device)
        pred = metrics["pred"].float()
        gt = batch["waypoints"].to(device).float()
        epe = torch.linalg.norm(pred[:, :, -1, :2] - gt[:, :, -1, :2], dim=-1)
        final_errors.extend(epe.reshape(-1).cpu().tolist())
        final_errors_drone.extend(epe[:, 0].cpu().tolist())
        final_errors_dog.extend(epe[:, 1].cpu().tolist())
        bs = pred.size(0)
        total_loss += float(loss.item()) * bs
        total_count += bs
        for key in metric_keys:
            if key in metrics:
                metric_sums[key] = metric_sums.get(key, 0.0) + float(metrics[key].item()) * bs
        batches += 1
        if cfg.eval_batches > 0 and batches >= cfg.eval_batches:
            break
    if final_errors:
        final_arr = np.asarray(final_errors, dtype=np.float32)
        hit = float(np.mean(final_arr <= cfg.final_wp_threshold))
        epe_mean = float(np.mean(final_arr))
    else:
        hit = float("nan")
        epe_mean = float("nan")
    stats = {
        "loss": total_loss / max(1, total_count),
        "final_epe": epe_mean,
        "final_epe_drone": float(np.mean(final_errors_drone)) if final_errors_drone else float("nan"),
        "final_epe_robotdog": float(np.mean(final_errors_dog)) if final_errors_dog else float("nan"),
        "hit": hit,
    }
    for key, value in metric_sums.items():
        out_key = f"val_{key}" if not key.startswith("val_") else key
        stats[out_key] = value / max(1, total_count)
    model.train(was_training)
    for planner, previous in zip(diffusion_planners, previous_deterministic):
        planner.deterministic_inference = previous
    return stats


# ----------------------- 双 Agent训练主循环 -----------------------

def train_airground_v3(cfg: MultiAgentTrainConfig) -> None:
    """AirGround-Coop V3 训练主循环。

    入口数据流：
    JSONL/cache -> MultiAgentJsonDataset -> DataLoader
    -> V3 three-stream model -> V3 composite loss
    -> checkpoint/train_log.csv。
    """
    cfg = apply_training_defaults(cfg)
    use_dist = bool(cfg.distributed)
    use_ddp = bool(cfg.distributed and not cfg.manual_grad_allreduce)
    use_manual_grad_allreduce = bool(cfg.distributed and cfg.manual_grad_allreduce)
    if use_dist:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        init_kwargs: Dict[str, Any] = {
            "backend": cfg.dist_backend,
            "init_method": "env://",
            "timeout": timedelta(minutes=max(1, int(cfg.ddp_timeout_minutes))),
        }
        if str(cfg.dist_backend).lower() == "nccl":
            init_kwargs["device_id"] = device
        dist.init_process_group(**init_kwargs)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rank = 0
        world_size = 1
        local_rank = 0

    seed_for_rank = cfg.seed if use_manual_grad_allreduce else (cfg.seed + rank)
    set_seed(seed_for_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_for_rank)
        # A100 对 TF32 有硬件加速；BF16 之外残留的 FP32 matmul 也可获得更高吞吐。
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    if rank == 0:
        mode = "airground-coop-v3"
        print(
            f"[INIT][MULTI] mode={mode} train_json={cfg.train_json} out_dir={cfg.out_dir} "
            f"distributed={use_dist} torch_ddp={use_ddp} manual_grad_allreduce={use_manual_grad_allreduce}",
            flush=True,
        )
    if cfg.val_json and not cfg.val_cache_root:
        raise ValueError(
            "--val_json requires --val_cache_root so validation cannot accidentally read training vision tokens."
        )
    ds = build_multi_agent_dataset(cfg.train_json, cfg)
    if rank == 0 and getattr(ds, "_files", None):
        kind_counts = {kind: 0 for kind in ("stt", "dt", "at")}
        for filename in ds._files:
            parts = Path(filename).parts
            for kind in kind_counts:
                if kind in parts:
                    kind_counts[kind] += 1
                    break
        print(
            f"[DATA][AIRGROUND_V3] joint_files={kind_counts} "
            f"joint_shuffle=True locality_block={cfg.shuffle_block_size}",
            flush=True,
        )
    if float(cfg.beta_control_drone) > 0.0 or float(cfg.beta_control_dog) > 0.0:
        for sample_index in range(min(len(ds), 256)):
            example = ds.get_example(sample_index)
            if example.get("inverse_control_version") != INVERSE_CONTROL_VERSION:
                raise ValueError(
                    "Inverse-control loss requires recorded-pose JSONL generated with "
                    f"{INVERSE_CONTROL_VERSION}; sample {sample_index} has "
                    f"{example.get('inverse_control_version')!r}."
                )
            if example.get("waypoint_label_source") != "recorded_pose_fixed_dt":
                raise ValueError(
                    "Inverse-control loss accepts only recorded_pose_fixed_dt waypoint labels."
                )
            if abs(float(example.get("waypoint_dt_s", example.get("dt", 0.0))) - 0.1) > 1e-8:
                raise ValueError("Inverse-control training requires waypoint_dt_s=0.1.")
    if rank == 0:
        multi_agent_dataset_sanity_report(ds, cfg)

    if cfg.dry_run:
        item = ds[0]
        print("[DRY_RUN][MULTI] first item shapes:")
        for key in (
            "coarse_tokens",
            "coarse_tidx",
            "fine_tokens",
            "fine_tidx",
            "bbox_feat",
            "relative_pose",
            "relative_pose_valid",
            "waypoints",
            "valid_mask",
        ):
            # Base/global-image samples intentionally omit grounding-only
            # labels (bbox_feat and relative_pose).  Keep --dry_run usable
            # for the V3 dry-run path.
            if key in item:
                print(f"  {key}: {tuple(item[key].shape)}")
        effective_instruction = (
            cfg.joint_instruction_override
            or cfg.instruction_override
            or item["instruction"]
        )
        print(f"  raw_instruction: {item['instruction']}")
        print(f"  effective_instruction: {effective_instruction}")
        if cfg.base_model and not cfg.separate_agent_context:
            layout = (
                "[joint_text, agent1_visual, agent2_visual, ACT1, ACT2]"
                if not cfg.use_agent_text_markers
                else "[joint_text, agent1_text, agent1_visual, agent2_text, agent2_visual, ACT1, ACT2]"
            )
            print(f"  shared_context_layout: {layout}")
        print(
            f"  agent_loss: normalize={cfg.normalize_agent_loss_weights} "
            f"drone_weight={cfg.drone_loss_weight} dog_weight={cfg.dog_loss_weight}"
        )
        print(
            f"  lr_schedule: type={cfg.lr_scheduler} peak_lr={cfg.lr} "
            f"warmup_steps={cfg.warmup_steps} min_lr={cfg.min_lr}"
        )
        return

    if cfg.alpha_xy is None:
        vals = []
        for i in range(min(len(ds), 4000)):
            ex = ds.get_example(i)
            try:
                wp = np.asarray(ex["waypoints"], dtype=np.float32)
                if wp.ndim == 3 and wp.shape[-1] >= 2:
                    vals.append(np.linalg.norm(wp[..., :2], axis=-1).reshape(-1))
            except Exception:
                pass
        if vals:
            cfg.alpha_xy = max(float(np.percentile(np.concatenate(vals), 95)), 1e-3)
            if rank == 0:
                print(f"[AUTO_ALPHA][MULTI] alpha_xy={cfg.alpha_xy:.4f}", flush=True)

    try:
        sample = ds[0]
        detected_dim = int(sample["fine_tokens"].shape[-1])
        if detected_dim != cfg.vision_feat_dim:
            if rank == 0:
                print(f"[AUTO_DIM][MULTI] vision_feat_dim {cfg.vision_feat_dim} -> {detected_dim}", flush=True)
            cfg.vision_feat_dim = detected_dim
    except Exception as exc:
        if rank == 0:
            print(f"[AUTO_DIM][MULTI] skipped: {exc}", flush=True)

    if cfg.shuffle_block_size > 0:
        sampler = LocalityAwareDistributedSampler(
            ds,
            block_size=cfg.shuffle_block_size,
            num_replicas=world_size if use_dist else 1,
            rank=rank if use_dist else 0,
            shuffle=True,
            seed=cfg.seed,
        )
    else:
        sampler = (
            torch.utils.data.distributed.DistributedSampler(
                ds, num_replicas=world_size, rank=rank, shuffle=True
            )
            if use_dist
            else None
        )
    loader_kwargs: Dict[str, Any] = {
        "batch_size": cfg.batch_size,
        "shuffle": sampler is None,
        "num_workers": cfg.num_workers,
        "pin_memory": True,
        "collate_fn": collate_multi_agent_batch,
        "sampler": sampler,
    }
    if cfg.num_workers > 0:
        # 每个样本会读取大量视觉 token 小文件；常驻 worker 和预取可减少 GPU 等待。
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = max(1, cfg.prefetch_factor)
    dl = DataLoader(ds, **loader_kwargs)
    cfg.grad_accum_steps = max(1, int(cfg.grad_accum_steps))
    if rank == 0:
        effective_batch = cfg.batch_size * world_size * cfg.grad_accum_steps
        updates_per_epoch = math.ceil(len(dl) / cfg.grad_accum_steps)
        total_optimizer_steps = updates_per_epoch * cfg.epochs
        print(
            f"[INIT][MULTI] samples={len(ds)} batches={len(dl)} batch_per_gpu={cfg.batch_size} "
            f"world_size={world_size} grad_accum_steps={cfg.grad_accum_steps} effective_batch={effective_batch} "
            f"updates_per_epoch={updates_per_epoch} total_optimizer_steps={total_optimizer_steps} "
            f"sampler={type(sampler).__name__ if sampler is not None else 'RandomSampler'}",
            flush=True,
        )
    else:
        updates_per_epoch = math.ceil(len(dl) / cfg.grad_accum_steps)
        total_optimizer_steps = updates_per_epoch * cfg.epochs

    print(f"[INIT][MULTI][rank {rank}] building model", flush=True)
    model = build_multi_agent_model(cfg)
    if cfg.init_ckpt and (not use_dist or rank == 0):
        init_path=Path(cfg.init_ckpt)
        if not init_path.is_file(): raise FileNotFoundError(init_path)
        checkpoint=torch.load(str(init_path),map_location='cpu',weights_only=False)
        state=checkpoint.get('model_state',checkpoint)
        missing,unexpected=model.load_state_dict(_cleanup_state_dict_keys(state),strict=False)
        print(f'[INIT_WEIGHTS][MULTI][rank {rank}] loaded={init_path} missing={len(missing)} unexpected={len(unexpected)} optimizer=fresh',flush=True)
    print(f"[INIT][MULTI][rank {rank}] moving model to {device}", flush=True)
    model = model.to(device)
    print(f"[INIT][MULTI][rank {rank}] model moved to {device}", flush=True)
    if use_manual_grad_allreduce:
        count = sum(1 for parameter in model.parameters() if parameter.requires_grad)
        numel = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        print(
            f"[INIT][MULTI][rank {rank}] using manual gradient all-reduce; "
            f"same_seed_init=True trainable_params={count} trainable_numel={numel:,}",
            flush=True,
        )
    if use_ddp:
        # 这里不要手动设置 _ddp_params_and_buffers_to_ignore。
        # PyTorch DDP 本身只会为 requires_grad=True 的参数建立 reducer；
        # 手动 ignore 冻结 LLM/buffer 在多卡初始化时可能让不同 rank 的
        # 参数校验列表不一致，表现为:
        #   "Rank X has 37 params, while rank Y has inconsistent 0 params"
        # 因此采用保守稳定路径，让 DDP 自己过滤冻结参数。
        trainable_param_count = sum(1 for _, parameter in model.named_parameters() if parameter.requires_grad)
        trainable_numel = sum(parameter.numel() for _, parameter in model.named_parameters() if parameter.requires_grad)
        print(
            f"[INIT][MULTI][rank {rank}] wrapping DDP; "
            f"trainable_params={trainable_param_count} trainable_numel={trainable_numel:,}",
            flush=True,
        )
        ddp_kwargs: Dict[str, Any] = {
            "device_ids": [local_rank],
            "output_device": local_rank,
            "find_unused_parameters": bool(cfg.ddp_find_unused_parameters),
            "broadcast_buffers": False,
        }
        if os.environ.get("DDP_INIT_SYNC", "1") == "0":
            try:
                import inspect

                if "init_sync" in inspect.signature(torch.nn.parallel.DistributedDataParallel).parameters:
                    ddp_kwargs["init_sync"] = False
                    print(f"[INIT][MULTI][rank {rank}] DDP init_sync disabled", flush=True)
                else:
                    print(
                        f"[INIT][MULTI][rank {rank}] DDP_INIT_SYNC=0 requested, "
                        "but this PyTorch version does not expose init_sync",
                        flush=True,
                    )
            except Exception as exc:
                print(
                    f"[INIT][MULTI][rank {rank}] DDP_INIT_SYNC=0 requested, "
                    f"but init_sync capability check failed: {exc}",
                    flush=True,
                )
        ddp_start = time.time()
        model = torch.nn.parallel.DistributedDataParallel(model, **ddp_kwargs)
        print(f"[INIT][MULTI][rank {rank}] DDP ready in {time.time() - ddp_start:.1f}s", flush=True)

    if rank == 0:
        model_inspect = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        total = sum(p.numel() for p in model_inspect.parameters())
        trainable = sum(p.numel() for p in model_inspect.parameters() if p.requires_grad)
        print(f"[PARAMS][MULTI] total={total:,} trainable={trainable:,} ({100.0 * trainable / max(1, total):.2f}%)", flush=True)
        print_multi_agent_parameter_report(model_inspect)

    optim = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if rank == 0:
        print(f"[OPTIMIZER][MULTI] defaults={optim.defaults}", flush=True)
    scheduler = build_multi_agent_lr_scheduler(optim, cfg, total_optimizer_steps)
    if rank == 0:
        print(
            f"[LR][MULTI] scheduler={cfg.lr_scheduler} initial_lr={optim.param_groups[0]['lr']:.8g} "
            f"peak_lr={cfg.lr:.8g} min_lr={cfg.min_lr:.8g} warmup_steps={cfg.warmup_steps}",
            flush=True,
        )
    amp_enabled = bool(cfg.mixed_precision and device.type == "cuda")
    amp_dtype = torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_enabled and amp_dtype == torch.float16))

    start_epoch = 0
    step = 0
    if cfg.resume:
        ckpt_path: Optional[Path]
        if cfg.resume_ckpt:
            ckpt_path = Path(cfg.resume_ckpt)
        else:
            pts = sorted(Path(cfg.out_dir).glob("model_epoch*.pt"), key=lambda p: p.stat().st_mtime)
            ckpt_path = pts[-1] if pts else None
        if ckpt_path is not None and ckpt_path.exists():
            obj = torch.load(str(ckpt_path), map_location=device)
            model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
            missing, unexpected = model_to_load.load_state_dict(
                _cleanup_state_dict_keys(obj.get("model_state", {})),
                strict=False,
            )
            critical_missing = [
                key for key in missing
                if key.startswith((
                    "planner_agent1.",
                    "planner_agent2.",
                    "proj.",
                    "tvi.",
                ))
            ]
            if cfg.use_roi_tokens:
                critical_missing.extend(
                    key for key in missing
                    if key.startswith("roi_proj.")
                )
            if cfg.use_visual_section_markers:
                critical_missing.extend(
                    key for key in missing
                    if key.startswith("visual_section_markers.")
                )
            if not cfg.base_model:
                critical_missing.extend(
                    key for key in missing
                    if key.startswith(("grounding_head.", "grounding_to_act."))
                )
            critical_missing.extend(
                key for key in missing
                if key in {"act_token_1", "act_token_2"}
            )
            if not cfg.base_model:
                critical_missing.extend(
                    key for key in missing
                    if key in {"gnd_token_1", "gnd_token_2", "grounding_act_gate"}
                )
            if critical_missing:
                raise RuntimeError(f"Resume checkpoint is missing critical model weights: {critical_missing[:20]}")
            if rank == 0 and (missing or unexpected):
                print(
                    f"[RESUME][MULTI][WARN] non-critical missing={len(missing)} unexpected={len(unexpected)}",
                    flush=True,
                )
            if obj.get("optim_state") is not None:
                optim.load_state_dict(obj["optim_state"])
            if scheduler is not None:
                if obj.get("scheduler_state") is not None:
                    scheduler.load_state_dict(obj["scheduler_state"])
                else:
                    resumed_lr = multi_agent_lr_for_step(
                        int(obj.get("step", 0)),
                        total_optimizer_steps,
                        cfg.lr,
                        cfg.min_lr,
                        cfg.warmup_steps,
                    )
                    scheduler.last_epoch = int(obj.get("step", 0))
                    scheduler._last_lr = [resumed_lr for _ in optim.param_groups]
                    for param_group in optim.param_groups:
                        param_group["lr"] = resumed_lr
                    if rank == 0:
                        print(
                            "[RESUME][MULTI][WARN] checkpoint has no scheduler state; "
                            f"reconstructed cosine LR at {resumed_lr:.8g}",
                            flush=True,
                        )
            if obj.get("scaler_state") is not None and scaler.is_enabled():
                scaler.load_state_dict(obj["scaler_state"])
            start_epoch = int(obj.get("epoch", 0))
            step = int(obj.get("step", 0))
            if rank == 0:
                print(f"[RESUME][MULTI] loaded {ckpt_path} epoch={start_epoch} step={step}", flush=True)

    out_dir = Path(cfg.out_dir)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_multi_agent_train_config(out_dir, cfg)
    best_val_loss_nav = float("inf")
    best_val_metrics_path = out_dir / "best_val_metrics.json"
    if rank == 0 and best_val_metrics_path.exists():
        try:
            best_val_loss_nav = float(json.loads(best_val_metrics_path.read_text(encoding="utf-8"))["best_val_loss"])
            print(f"[VAL][MULTI] existing best val_loss_nav={best_val_loss_nav:.5f}", flush=True)
        except Exception as exc:
            print(f"[VAL][MULTI][WARN] failed to read {best_val_metrics_path}: {exc}", flush=True)

    val_ds = (
        build_multi_agent_dataset(cfg.val_json, cfg, cache_root=cfg.val_cache_root)
        if (cfg.val_json and rank == 0)
        else None
    )
    ema_loss: Optional[float] = None
    last_log = time.time()

    stop_after_max_steps = bool(cfg.max_steps > 0 and step >= cfg.max_steps)
    if stop_after_max_steps and rank == 0:
        print(f"[DEBUG][MULTI] max_steps={cfg.max_steps} already reached at resumed step={step}", flush=True)

    train_start = time.time()
    tb_writer = None
    if rank == 0:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_writer = SummaryWriter(log_dir=str(Path(cfg.out_dir) / "tensorboard"))
            print(f"[TENSORBOARD] log_dir={Path(cfg.out_dir) / 'tensorboard'}", flush=True)
        except Exception as exc:
            print(f"[TENSORBOARD] disabled: {exc}", flush=True)
    for epoch in range(start_epoch, cfg.epochs):
        if stop_after_max_steps:
            break
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        epoch_start = time.time()
        epoch_grad_preclip: List[float] = []
        epoch_grad_postclip: List[float] = []
        epoch_clip_scales: List[float] = []
        pbar = None
        if rank == 0 and cfg.progress and tqdm is not None:
            pbar = tqdm(total=len(dl), desc=f"multi epoch {epoch + 1}/{cfg.epochs}", dynamic_ncols=True, file=sys.stdout)

        for batch_idx, batch in enumerate(dl, start=1):
            do_step = (batch_idx % max(1, cfg.grad_accum_steps) == 0) or (batch_idx == len(dl))
            sync_ctx = model.no_sync() if use_ddp and hasattr(model, "no_sync") and not do_step else nullcontext()
            amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled) if amp_enabled else nullcontext()

            with sync_ctx:
                with amp_ctx:
                    loss, metrics = forward_multi_agent_loss(model, batch, cfg, device)
                scaler.scale(loss / max(1, cfg.grad_accum_steps)).backward()

            if pbar is not None:
                if bool(getattr(cfg, "compact_terminal_log", False)):
                    memory_gb = (
                        torch.cuda.max_memory_allocated(device) / (1024.0 ** 3)
                        if device.type == "cuda"
                        else 0.0
                    )
                    pbar.set_description_str(
                        f"ep {epoch + 1}/{cfg.epochs} "
                        f"L={float(loss.detach().item()):.3f} "
                        f"N={float(metrics['loss_nav'].item()):.3f} "
                        f"J={float(metrics['loss_bbox'].item()):.3f} "
                        f"M={memory_gb:.1f}G",
                        refresh=False,
                    )
                else:
                    pbar.set_postfix(
                        loss=f"{float(loss.detach().item()):.4f}",
                        nav=f"{float(metrics['loss_nav'].item()):.4f}",
                        refresh=False,
                    )
                pbar.update(1)
            if not do_step:
                continue

            grad_norm_preclip = 0.0
            grad_norm_postclip = 0.0
            was_clipped = 0.0
            clip_scale = 1.0
            if use_manual_grad_allreduce:
                manual_allreduce_trainable_grads(
                    model,
                    world_size,
                    use_cpu=bool(cfg.manual_grad_allreduce_cpu),
                )
            if cfg.grad_clip is not None and cfg.grad_clip > 0:
                scaler.unscale_(optim)
                grad_norm_preclip = _compute_total_grad_norm(model.parameters())
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                grad_norm_postclip = _compute_total_grad_norm(model.parameters())
                was_clipped = 1.0 if grad_norm_preclip > float(cfg.grad_clip) else 0.0
                clip_scale = min(1.0, float(cfg.grad_clip) / (grad_norm_preclip + 1e-12))
            grad_norm = grad_norm_preclip
            epoch_grad_preclip.append(float(grad_norm_preclip))
            epoch_grad_postclip.append(float(grad_norm_postclip))
            epoch_clip_scales.append(float(clip_scale))
            lr_used = float(optim.param_groups[0]["lr"])
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)
            step += 1
            if scheduler is not None:
                scheduler.step()

            if rank == 0 and (step % cfg.log_every == 0):
                now = time.time()
                elapsed = now - last_log
                last_log = now
                loss_val = float(loss.detach().item())
                ema_loss = loss_val if ema_loss is None else (0.98 * ema_loss + 0.02 * loss_val)
                pred = metrics["pred"].float()
                gt = batch["waypoints"].to(device).float()
                final_epe_per_agent = torch.linalg.norm(pred[:, :, -1, :2] - gt[:, :, -1, :2], dim=-1)
                final_epe = final_epe_per_agent.mean().item()
                final_epe_drone = final_epe_per_agent[:, 0].mean().item()
                final_epe_dog = final_epe_per_agent[:, 1].mean().item()
                regression_loss = float(metrics["regression_loss"].item()) if "regression_loss" in metrics else float("nan")
                score_loss = float(metrics["score_loss"].item()) if "score_loss" in metrics else float("nan")
                epoch_eta_seconds = (time.time() - epoch_start) / max(1, batch_idx) * max(0, len(dl) - batch_idx)
                completed_epochs = epoch - start_epoch
                completed_batches = completed_epochs * len(dl) + batch_idx
                total_batches = max(1, (cfg.epochs - start_epoch) * len(dl))
                total_eta_seconds = (time.time() - train_start) / max(1, completed_batches) * max(0, total_batches - completed_batches)
                eta = _format_duration(epoch_eta_seconds)
                total_eta = _format_duration(total_eta_seconds)
                if bool(getattr(cfg, "compact_terminal_log", False)):
                    dog_normal = float(metrics.get("loss_dog_normal", metrics["loss_nav_dog"]).item())
                    dog_guided = float(metrics.get("loss_dog_guided", metrics.get("regression_loss")).item())
                    memory_gb = (
                        torch.cuda.max_memory_allocated(device) / (1024.0 ** 3)
                        if device.type == "cuda"
                        else 0.0
                    )
                    progress_pct = 100.0 * batch_idx / max(1, len(dl))
                    msg = (
                        f"[TRAIN] e{epoch + 1}/{cfg.epochs} b{batch_idx}/{len(dl)} {progress_pct:.2f}% "
                        f"loss={loss_val:.3f} nav={metrics['loss_nav'].item():.3f}"
                        f"(D={metrics['loss_nav_drone'].item():.3f},N={dog_normal:.3f},G={dog_guided:.3f}) "
                        f"jepa={metrics['loss_bbox'].item():.3f} route={score_loss:.3f} "
                        f"match={float(metrics.get('target_match_accuracy', torch.tensor(float('nan')))):.3f} "
                        f"recall8={float(metrics.get('candidate_top8_recall', torch.tensor(float('nan')))):.3f} "
                        f"lr={lr_used:.2g} mem={memory_gb:.1f}G eta={eta} total_eta={total_eta}"
                    )
                else:
                    msg = (
                        f"[TRAIN][MULTI] epoch={epoch + 1}/{cfg.epochs} batch={batch_idx}/{len(dl)} step={step} eta={eta} total_eta={total_eta} "
                        f"lr={lr_used:.8g} "
                        f"loss={loss_val:.5f} ema={ema_loss:.5f} nav={metrics['loss_nav'].item():.5f} "
                        f"nav_drone={metrics['loss_nav_drone'].item():.5f} nav_dog={metrics['loss_nav_dog'].item():.5f} "
                        f"xy={metrics['loss_nav_xy'].item():.5f} yaw={metrics['loss_nav_yaw'].item():.5f} "
                        f"final={metrics['loss_nav_final'].item():.5f} "
                        f"ctrl=(D={metrics['loss_control_drone'].item():.5f},G={metrics['loss_control_dog'].item():.5f}) "
                        f"turn={metrics['turn_fraction'].item():.3f}"
                        f"(D={metrics['turn_fraction_drone'].item():.3f},G={metrics['turn_fraction_dog'].item():.3f}) "
                        f"stop={metrics['stop_fraction'].item():.3f}"
                        f"(D={metrics['stop_fraction_drone'].item():.3f},G={metrics['stop_fraction_dog'].item():.3f}) "
                        f"behavior_w={metrics['behavior_weight_mean'].item():.2f} "
                        f"reg={regression_loss:.5f} score={score_loss:.5f} "
                        f"bbox={metrics['loss_bbox'].item():.5f} vis={metrics['loss_visible'].item():.5f} "
                        f"relpose={metrics['loss_relative_pose'].item():.5f} "
                        f"final_epe={final_epe:.4f} epe_drone={final_epe_drone:.4f} epe_dog={final_epe_dog:.4f} "
                        f"grad_pre={grad_norm_preclip:.3f} grad_post={grad_norm_postclip:.3f} "
                        f"clipped={int(was_clipped)} clip_scale={clip_scale:.3f} dt={elapsed:.2f}s"
                    )
                if pbar is not None:
                    pbar.write(msg)
                else:
                    print(msg, flush=True)
                if tb_writer is not None:
                    tb_writer.add_scalar("train/loss", loss_val, step)
                    for key, value in metrics.items():
                        if torch.is_tensor(value) and value.numel() == 1:
                            tb_writer.add_scalar(f"train/{key}", float(value.detach().cpu()), step)
                    tb_writer.add_scalar("train/lr", lr_used, step)
                    tb_writer.add_scalar("train/epoch", epoch + batch_idx / max(1, len(dl)), step)
                    tb_writer.flush()
                if cfg.csv_logging:
                    csv_path = out_dir / "train_log.csv"
                    prepare_multi_agent_csv_log(csv_path)
                    write_header = not csv_path.exists()
                    with csv_path.open("a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        if write_header:
                            writer.writerow(MULTI_AGENT_LOG_FIELDS)
                        writer.writerow(
                            [
                                epoch,
                                step,
                                lr_used,
                                loss_val,
                                ema_loss,
                                float(metrics["loss_nav"].item()),
                                float(metrics["loss_nav_drone"].item()),
                                float(metrics["loss_nav_dog"].item()),
                                float(metrics["loss_nav_xy"].item()),
                                float(metrics["loss_nav_yaw"].item()),
                                float(metrics["loss_nav_final"].item()),
                                float(metrics["loss_control_drone"].item()),
                                float(metrics["loss_control_dog"].item()),
                                float(metrics["turn_fraction"].item()),
                                float(metrics["stop_fraction"].item()),
                                float(metrics["turn_fraction_drone"].item()),
                                float(metrics["turn_fraction_dog"].item()),
                                float(metrics["stop_fraction_drone"].item()),
                                float(metrics["stop_fraction_dog"].item()),
                                float(metrics["behavior_weight_mean"].item()),
                                regression_loss,
                                score_loss,
                                float(metrics["loss_bbox"].item()),
                                float(metrics["loss_visible"].item()),
                                float(metrics["loss_relative_pose"].item()),
                                float(metrics.get("loss_candidate_bce", torch.tensor(float("nan"))).item()),
                                float(metrics.get("loss_candidate_rank", torch.tensor(float("nan"))).item()),
                                float(metrics.get("candidate_top8_recall", torch.tensor(float("nan"))).item()),
                                float(metrics.get("target_match_accuracy", torch.tensor(float("nan"))).item()),
                                float(metrics.get("candidate_class_accuracy", torch.tensor(float("nan"))).item()),
                                float(metrics.get("candidate_positive_probability", torch.tensor(float("nan"))).item()),
                                float(metrics.get("candidate_max_negative_probability", torch.tensor(float("nan"))).item()),
                                float(metrics.get("candidate_threshold_target_recall", torch.tensor(float("nan"))).item()),
                                float(metrics.get("no_target_false_accept_rate", torch.tensor(float("nan"))).item()),
                                float(metrics.get("loss_self", torch.tensor(float("nan"))).item()),
                                float(metrics.get("loss_self_drone", torch.tensor(float("nan"))).item()),
                                float(metrics.get("loss_self_dog", torch.tensor(float("nan"))).item()),
                                float(metrics.get("loss_cooperative", torch.tensor(float("nan"))).item()),
                                float(metrics.get("loss_coop_drone", torch.tensor(float("nan"))).item()),
                                float(metrics.get("loss_coop_dog", torch.tensor(float("nan"))).item()),
                                float(metrics.get("loss_mode", torch.tensor(float("nan"))).item()),
                                float(metrics.get("loss_jepa", torch.tensor(float("nan"))).item()),
                                float(metrics.get("loss_belief", torch.tensor(float("nan"))).item()),
                                float(metrics.get("loss_target_match", torch.tensor(float("nan"))).item()),
                                float(metrics.get("loss_uncertainty", torch.tensor(float("nan"))).item()),
                                float(metrics.get("loss_smoothness", torch.tensor(float("nan"))).item()),
                                float(metrics.get("loss_kinematics", torch.tensor(float("nan"))).item()),
                                float(metrics.get("loss_diversity", torch.tensor(float("nan"))).item()),
                                float(metrics.get("target_match_positive_fraction", torch.tensor(float("nan"))).item()),
                                float(metrics.get("synthetic_drone_fraction", torch.tensor(float("nan"))).item()),
                                float(metrics.get("synthetic_dog_fraction", torch.tensor(float("nan"))).item()),
                                float(metrics.get("pose_perturbation_fraction", torch.tensor(float("nan"))).item()),
                                float(metrics.get("roi_only_fraction", torch.tensor(float("nan"))).item()),
                                float(metrics.get("current_full_fraction", torch.tensor(float("nan"))).item()),
                                float(metrics.get("recent_full_fraction", torch.tensor(float("nan"))).item()),
                                float(metrics.get("all_full_fraction", torch.tensor(float("nan"))).item()),
                                float(metrics.get("coop_drone_fraction", torch.tensor(float("nan"))).item()),
                                float(metrics.get("coop_dog_fraction", torch.tensor(float("nan"))).item()),
                                final_epe,
                                final_epe_drone,
                                final_epe_dog,
                                grad_norm_preclip,
                                grad_norm_postclip,
                                was_clipped,
                                clip_scale,
                                grad_norm,
                            ]
                        )

            should_save = cfg.save_every > 0 and step % cfg.save_every == 0
            if should_save:
                # rank 0 写入大 checkpoint 时，其余 rank 在 barrier 等待，避免提前进入下一次同步梯度。
                if use_dist:
                    dist.barrier()
                if rank == 0:
                    ckpt = out_dir / f"model_epoch{epoch:02d}_step{step:06d}.pt"
                    save_multi_agent_checkpoint(ckpt, model, optim, scheduler, scaler, cfg, epoch, step)
                    prune_multi_agent_checkpoints(out_dir, cfg.max_ckpts)
                    print(f"[CKPT][MULTI] saved {ckpt}", flush=True)
                if use_dist:
                    dist.barrier()

            if cfg.max_steps > 0 and step >= cfg.max_steps:
                stop_after_max_steps = True
                if rank == 0:
                    print(f"[DEBUG][MULTI] stopping after max_steps={cfg.max_steps}", flush=True)
                break

            should_evaluate = bool(cfg.val_json and cfg.eval_every > 0 and step % cfg.eval_every == 0)
            if should_evaluate:
                # 当前离线验证只在 rank 0 执行；显式同步防止其他 rank 卡在训练 all-reduce。
                if use_dist:
                    dist.barrier()
                if rank == 0:
                    assert val_ds is not None
                    stats = evaluate_multi_agent(model, val_ds, cfg, device)
                    print(
                        f"[VAL][MULTI] step={step} loss={stats['loss']:.5f} "
                        f"loss_nav={stats.get('val_loss_nav', float('nan')):.5f} "
                        f"drone={stats.get('val_loss_nav_drone', float('nan')):.5f} "
                        f"robotdog={stats.get('val_loss_nav_dog', float('nan')):.5f} "
                        f"final_epe={stats['final_epe']:.4f} hit@{cfg.final_wp_threshold}={stats['hit']:.3f}",
                        flush=True,
                    )
                    val_loss_nav = float(stats.get("val_loss_nav", stats["loss"]))
                    if np.isfinite(val_loss_nav) and val_loss_nav < best_val_loss_nav:
                        best_val_loss_nav = val_loss_nav
                        best_ckpt = out_dir / "best_val.pt"
                        save_multi_agent_checkpoint(best_ckpt, model, optim, scheduler, scaler, cfg, epoch, step)
                        payload = {
                            "epoch": epoch,
                            "step": step,
                            "best_epoch": epoch,
                            "best_step": step,
                            "best_val_loss": best_val_loss_nav,
                            "selection_metric": "val_loss_nav",
                            "checkpoint": str(best_ckpt),
                            **{key: float(value) for key, value in stats.items()},
                        }
                        best_val_metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                        print(f"[VAL][MULTI] saved new best checkpoint {best_ckpt}", flush=True)
                if use_dist:
                    dist.barrier()

        if pbar is not None:
            pbar.close()
        completed_epoch = epoch + 1
        if rank == 0:
            append_multi_agent_epoch_summary(
                out_dir / "epoch_summary.csv",
                completed_epoch,
                step,
                epoch_grad_preclip,
                epoch_grad_postclip,
                epoch_clip_scales,
            )
        should_save_epoch = not stop_after_max_steps and (
            (cfg.save_every_epochs > 0 and completed_epoch % cfg.save_every_epochs == 0)
            or completed_epoch == cfg.epochs
        )
        if should_save_epoch:
            if use_dist:
                dist.barrier()
            if rank == 0:
                ckpt = out_dir / f"model_epoch{completed_epoch:03d}_step{step:06d}_final.pt"
                save_multi_agent_checkpoint(ckpt, model, optim, scheduler, scaler, cfg, completed_epoch, step)
                prune_multi_agent_checkpoints(out_dir, cfg.max_ckpts)
                print(f"[CKPT][MULTI] saved epoch {completed_epoch} checkpoint {ckpt}", flush=True)
            if use_dist:
                dist.barrier()

        if stop_after_max_steps:
            break

    if use_dist:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        dist.destroy_process_group()
    if rank == 0:
        if tb_writer is not None:
            tb_writer.close()
        print(f"[DONE][MULTI] training complete step={step}", flush=True)
