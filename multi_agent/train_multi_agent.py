#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""早期独立双 Agent OpenTrackVLA MLP 训练入口。

本脚本参考原始 `train.py`，但把数据结构从单 Agent 扩展为双 Agent：
- JSONL 中每条样本包含 agent1/agent2 的历史帧、当前帧、bbox 和未来路点。
- Dataset 读取 vision_cache 中的 *_vcoarse.pt / *_vfine.pt，并组装成 (B, 2, ...)。
- 模型输出 (B, 2, N, 3)，loss 对两个 Agent 的有效路点共同计算。

推荐流程：
1. 先运行 make_multi_agent_tracking_data.py 生成 JSONL。
2. 再运行 precache_multi_agent_frames.py 生成视觉 token 缓存。
3. 最后运行本脚本训练模型。

核心类与函数：
- ``MultiAgentJsonDataset`` / ``TrainConfig``：读取双 Agent标签与视觉缓存并管理配置。
- ``build_model`` / ``forward_loss``：构建 MLP 模型并计算路点、bbox、可见性损失。
- ``train`` / ``evaluate``：训练、验证、日志和 checkpoint 主循环。
- ``parse_args``：解析命令行配置。

版本边界：
该文件属于 ``multi_agent/`` 早期独立实验实现。当前双 Agent MLP 主流程使用
根目录 ``train.py --multi_agent``；Anchor Diffusion 使用
``train_unrealzoo_anchor_diffusion.py``。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multi_agent_model import MultiAgentModelConfig, MultiAgentOpenTrackVLA  # noqa: E402

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ----------------------- 通用工具与路径解析 -----------------------

def set_seed(seed: int) -> None:
    """固定随机种子，保证训练和 DataLoader shuffle 更可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def cleanup_state_dict_keys(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """兼容 DDP/DataParallel 保存的 `module.` 前缀。"""
    if state_dict and any(k.startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def load_tokens_file(path: str) -> torch.Tensor:
    """读取预缓存视觉 token。

    cache 文件可能直接是 Tensor，也可能是包含 V/Vfine/Vcoarse 等 key 的 dict。
    这里统一返回 float32 Tensor，后续再由 autocast 或模型 dtype 控制精度。
    """
    try:
        obj = torch.load(path, map_location="cpu")
    except Exception:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, torch.Tensor):
        return obj.float()
    if isinstance(obj, dict):
        for key in ("V", "Vfine", "Vcoarse", "tokens", "feat", "features"):
            val = obj.get(key)
            if isinstance(val, torch.Tensor):
                if val.dim() == 3 and val.size(0) == 1:
                    val = val[0]
                return val.float()
    raise ValueError(f"Unrecognized token file: {path}")


def read_jsonl_file(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    out.append(obj)
    return out


def find_base_root(train_path: Path) -> Path:
    """从 train_json 路径向上寻找包含 frames/ 的数据根目录。"""
    candidate = train_path if train_path.is_dir() else train_path.parent
    for _ in range(6):
        if (candidate / "frames").exists():
            return candidate
        if candidate.parent == candidate:
            break
        candidate = candidate.parent
    return train_path if train_path.is_dir() else train_path.parent


def resolve_path(base_root: Path, rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    return p if p.is_absolute() else (base_root / p)


def token_paths_for_frame(base_root: Path, cache_root: Path, frame_path: Path) -> Tuple[Path, Path]:
    """把一张 frame 路径映射到对应的 vcoarse/vfine cache 路径。

    例如:
    frames/seed_100/.../drone/frame_00001.jpg
    ->
    vision_cache/frames/seed_100/.../drone/frame_00001_vcoarse.pt
    vision_cache/frames/seed_100/.../drone/frame_00001_vfine.pt
    """
    try:
        rel = frame_path.resolve().relative_to(base_root.resolve())
    except ValueError:
        parts = frame_path.parts
        if "frames" in parts:
            rel = Path(*parts[parts.index("frames") :])
        else:
            rel = Path(frame_path.name)
    token_dir = cache_root / rel.parent
    return token_dir / f"{rel.stem}_vcoarse.pt", token_dir / f"{rel.stem}_vfine.pt"


# ----------------------- 路点标签与损失工具 -----------------------

def fit_waypoints_and_mask(
    waypoints: Any,
    valid_mask: Any,
    n_waypoints: int,
    action_dims: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """把 JSON 中的变长路点修正成固定形状。

    输出:
    - waypoints:  (n_waypoints, action_dims)
    - valid_mask: (n_waypoints,)

    如果样本不足 n_waypoints，会用最后一个有效路点补齐，但 mask 会标记真实有效范围。
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


def masked_mse_multi(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """双 Agent masked MSE。

    pred/target: (B, 2, N, D)
    mask:        (B, 2, N)
    只在有效 waypoint 上计算误差。
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred shape {tuple(pred.shape)} != target shape {tuple(target.shape)}")
    if mask.dim() == 2:
        mask = mask.unsqueeze(1)
    expanded = mask.view(*mask.shape, 1).expand_as(pred)
    se = (pred - target).pow(2)
    selected = se[expanded]
    return selected.mean() if selected.numel() > 0 else pred.new_tensor(0.0)


def normalize_xy_by_alpha(pred: torch.Tensor, target: torch.Tensor, alpha_task: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """训练 loss 用归一化 XY，避免 x/y 数值尺度支配 yaw。

    模型 forward 输出已经乘过 alpha_task，是实际单位。loss 里再除回去，
    等价于让 x/y 在 normalized space 中训练，而 yaw 保持原尺度。
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


@dataclass
# ----------------------- 双 Agent训练数据集 -----------------------

class MultiAgentDataConfig:
    """Dataset 配置，只包含数据读取相关参数。"""

    train_json: str
    n_waypoints: int = 8
    history: int = 31
    cache_root: Optional[str] = None
    action_dims: int = 3
    online_encode_missing: bool = False
    image_size: int = 384


class MultiAgentJsonDataset(Dataset):
    """双 Agent JSONL Dataset。

    每个 __getitem__ 返回:
    - coarse_tokens: (2, history*4, C)
    - fine_tokens:   (2, 64, C)
    - bbox_feat:     (2, 4)
    - waypoints:     (2, n_waypoints, 3)
    """

    def __init__(self, cfg: MultiAgentDataConfig):
        super().__init__()
        self.cfg = cfg
        self.train_path = Path(cfg.train_json).resolve()
        self.base_root = find_base_root(self.train_path)
        self.cache_root = Path(cfg.cache_root).resolve() if cfg.cache_root else (self.base_root / "vision_cache")
        self._online_encoder = None
        self._lazy = False
        self._index: Optional[List[Tuple[str, int]]] = None
        self.examples: Optional[List[Dict[str, Any]]] = None

        # 支持两类输入：
        # 1. 聚合 JSON 文件 dataset.json
        # 2. JSONL 文件或 JSONL 目录，目录会递归索引所有 .jsonl
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
            self._index = []
            for fp in files:
                with fp.open("rb") as f:
                    pos = 0
                    while True:
                        line = f.readline()
                        if not line:
                            break
                        if line.strip():
                            self._index.append((str(fp), pos))
                        pos += len(line)

    def __len__(self) -> int:
        if self.examples is not None:
            return len(self.examples)
        return len(self._index or [])

    def _read_indexed_example(self, idx: int) -> Dict[str, Any]:
        assert self._index is not None
        fp, offset = self._index[idx]
        with open(fp, "rb") as f:
            f.seek(offset)
            line = f.readline()
        return json.loads(line.decode("utf-8"))

    def get_example(self, idx: int) -> Dict[str, Any]:
        if self._lazy:
            return self._read_indexed_example(idx)
        assert self.examples is not None
        return self.examples[idx]

    def _get_online_encoder(self):
        """懒加载在线视觉编码器。

        正常训练应优先使用 precache 的 token。只有传入 --online_encode_missing 且 cache 缺失时，
        才会走这里，避免训练时反复加载 DINO/SigLIP。
        """
        if self._online_encoder is None:
            try:
                from tools.cache_gridpool import VisionFeatureCacher, VisionCacheConfig
            except Exception as exc:
                raise RuntimeError(
                    "Missing vision cache and failed to import online encoder. "
                    "Run multi_agent/precache_multi_agent_frames.py first, or fix cache_gridpool dependencies."
                ) from exc
            use_cuda = torch.cuda.is_available()
            self._online_encoder = VisionFeatureCacher(
                VisionCacheConfig(
                    image_size=self.cfg.image_size,
                    batch_size=8,
                    device="cuda" if use_cuda else "cpu",
                )
            )
            self._online_encoder.eval()
        return self._online_encoder

    @torch.inference_mode()
    def _encode_image_tokens(self, img_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            from tools.cache_gridpool import grid_pool_tokens
        except Exception as exc:
            raise RuntimeError("Failed to import grid_pool_tokens for online encoding.") from exc
        enc = self._get_online_encoder()
        pil = Image.open(str(img_path)).convert("RGB")
        tok_dino, hp, wp = enc._encode_dino([pil])
        tok_sigl = enc._encode_siglip([pil], out_hw=(hp, wp))
        tokens = torch.cat([tok_dino, tok_sigl], dim=-1)
        vfine = grid_pool_tokens(tokens, hp, wp, out_tokens=64)
        vcoarse = grid_pool_tokens(tokens, hp, wp, out_tokens=4)
        return vcoarse[0].cpu().float(), vfine[0].cpu().float()

    def _load_or_encode_tokens(self, frame_path: Path, fine: bool) -> torch.Tensor:
        """优先从 cache 读取 token，必要时在线编码并回写 cache。"""
        vc_path, vf_path = token_paths_for_frame(self.base_root, self.cache_root, frame_path)
        wanted = vf_path if fine else vc_path
        try:
            return load_tokens_file(str(wanted))
        except Exception:
            if not self.cfg.online_encode_missing:
                raise FileNotFoundError(
                    f"Missing token file: {wanted}. Run multi_agent/precache_multi_agent_frames.py "
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
            return vf if fine else vc

    def _load_agent_tokens(self, ex: Dict[str, Any], agent_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
        """读取单个 Agent 的历史粗 token 和当前细 token。

        agent_idx=1 对应 agent1_* 字段，agent_idx=2 对应 agent2_* 字段。
        历史帧不足 history 时，沿用最早可用帧/当前帧做左侧 padding，与原始 train.py 逻辑一致。
        """
        history = int(self.cfg.history)
        prefix = f"agent{agent_idx}_"
        current_rel = ex.get(prefix + "current")
        if not isinstance(current_rel, str) or not current_rel:
            raise KeyError(f"Missing {prefix}current in sample")
        current_path = resolve_path(self.base_root, current_rel)
        fine_tokens = self._load_or_encode_tokens(current_path, fine=True)
        fine_tidx = torch.full((fine_tokens.size(0),), fill_value=history, dtype=torch.long)

        image_rels = ex.get(prefix + "images", [])
        if not isinstance(image_rels, list):
            image_rels = []
        image_rels = image_rels[-history:]
        missing = history - len(image_rels)

        coarse_list: List[torch.Tensor] = []
        tidx_list: List[torch.Tensor] = []
        first_tok: Optional[torch.Tensor] = None
        current_coarse: Optional[torch.Tensor] = None
        for t in range(history):
            tok: Optional[torch.Tensor] = None
            if t >= missing:
                img_rel = image_rels[t - missing]
                if isinstance(img_rel, str) and img_rel:
                    img_path = resolve_path(self.base_root, img_rel)
                    tok = self._load_or_encode_tokens(img_path, fine=False)
                    if first_tok is None:
                        first_tok = tok
            if tok is None:
                if first_tok is not None:
                    tok = first_tok
                else:
                    if current_coarse is None:
                        current_coarse = self._load_or_encode_tokens(current_path, fine=False)
                    tok = current_coarse
            coarse_list.append(tok)
            tidx_list.append(torch.full((tok.size(0),), fill_value=t, dtype=torch.long))

        coarse_tokens = torch.cat(coarse_list, dim=0)
        coarse_tidx = torch.cat(tidx_list, dim=0)
        return coarse_tokens, coarse_tidx, fine_tokens, fine_tidx, str(current_path)

    def _load_targets(self, ex: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """读取两个 Agent 的 waypoint 标签和 valid_mask。"""
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
            wp, vm = fit_waypoints_and_mask(wp_i, vm_i, self.cfg.n_waypoints, self.cfg.action_dims)
            out_wp.append(wp)
            out_mask.append(vm)
        return torch.stack(out_wp, dim=0), torch.stack(out_mask, dim=0)

    def _load_bbox(self, ex: Dict[str, Any]) -> torch.Tensor:
        """读取两个 Agent 的归一化 bbox，输出 (2, 4)。"""
        if "bbox_feat" in ex:
            bbox = ex["bbox_feat"]
        else:
            bbox = [ex.get("agent1_bbox", [0, 0, 0, 0]), ex.get("agent2_bbox", [0, 0, 0, 0])]
        t = torch.as_tensor(bbox, dtype=torch.float32)
        if t.shape != (2, 4):
            raise ValueError(f"Expected bbox_feat shape (2,4), got {tuple(t.shape)}")
        return t.clamp(0.0, 1.0)

    def _load_visible(self, ex: Dict[str, Any]) -> torch.Tensor:
        """读取可见性监督，给可选 visibility loss 使用。"""
        order = ex.get("agent_order", [ex.get("agent1_name", "drone"), ex.get("agent2_name", "robotdog")])
        agents = ex.get("agents", {})
        vals = []
        for idx in range(2):
            name = order[idx] if isinstance(order, list) and len(order) > idx else f"agent{idx + 1}"
            payload = agents.get(name, {}) if isinstance(agents, dict) else {}
            vals.append(1.0 if bool(payload.get("target_visible", True)) else 0.0)
        return torch.tensor(vals, dtype=torch.float32)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.get_example(idx)
        # 分别读取 agent1/agent2，再 stack 成模型 forward 需要的双 Agent 维度。
        a1 = self._load_agent_tokens(ex, agent_idx=1)
        a2 = self._load_agent_tokens(ex, agent_idx=2)
        waypoints, valid_mask = self._load_targets(ex)
        bbox_feat = self._load_bbox(ex)
        visible = self._load_visible(ex)
        history = int(self.cfg.history)

        coarse_tokens = torch.stack([a1[0], a2[0]], dim=0)
        coarse_tidx = torch.stack([a1[1], a2[1]], dim=0)
        fine_tokens = torch.stack([a1[2], a2[2]], dim=0)
        fine_tidx = torch.stack([a1[3], a2[3]], dim=0)

        return {
            "coarse_tokens": coarse_tokens,
            "coarse_tidx": coarse_tidx,
            "fine_tokens": fine_tokens,
            "fine_tidx": fine_tidx,
            "yaw_hist": torch.zeros(2, history, dtype=torch.float32),
            "yaw_curr": torch.zeros(2, 1, dtype=torch.float32),
            "bbox_feat": bbox_feat,
            "visible": visible,
            "waypoints": waypoints,
            "valid_mask": valid_mask,
            "instruction": ex.get("instruction", "Follow the target person without collision."),
            "episode_id": ex.get("episode_id", ""),
            "step_index": int(ex.get("step_index", idx)),
            "current_path": [a1[4], a2[4]],
        }


# ----------------------- batch 组装与数据检查 -----------------------

def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """把单样本字典合成 batch。

    注意 instruction/current_path/episode_id 保持 Python list，其他张量 stack。
    """
    return {
        "coarse_tokens": torch.stack([b["coarse_tokens"] for b in batch], dim=0),
        "coarse_tidx": torch.stack([b["coarse_tidx"] for b in batch], dim=0),
        "fine_tokens": torch.stack([b["fine_tokens"] for b in batch], dim=0),
        "fine_tidx": torch.stack([b["fine_tidx"] for b in batch], dim=0),
        "yaw_hist": torch.stack([b["yaw_hist"] for b in batch], dim=0),
        "yaw_curr": torch.stack([b["yaw_curr"] for b in batch], dim=0),
        "bbox_feat": torch.stack([b["bbox_feat"] for b in batch], dim=0),
        "visible": torch.stack([b["visible"] for b in batch], dim=0),
        "waypoints": torch.stack([b["waypoints"] for b in batch], dim=0),
        "valid_mask": torch.stack([b["valid_mask"] for b in batch], dim=0),
        "instruction": [b["instruction"] for b in batch],
        "episode_id": [b["episode_id"] for b in batch],
        "step_index": torch.tensor([b["step_index"] for b in batch], dtype=torch.long),
        "current_path": [b["current_path"] for b in batch],
    }


def dataset_sanity_report(ds: MultiAgentJsonDataset, cfg: "TrainConfig", max_items: int = 256) -> None:
    """训练前快速检查数据分布、图片路径和 cache 覆盖率。"""
    try:
        n = min(max_items, len(ds))
        xs, ys, ths = [], [], []
        bbox_vals = []
        img_ok = 0
        cache_ok = 0
        for i in range(n):
            ex = ds.get_example(i)
            wp = np.asarray(ex.get("waypoints", []), dtype=np.float32)
            if wp.ndim == 3 and wp.shape[-1] >= 3:
                xs.append(wp[..., 0].reshape(-1))
                ys.append(wp[..., 1].reshape(-1))
                ths.append(wp[..., 2].reshape(-1))
            bbox = np.asarray(ex.get("bbox_feat", []), dtype=np.float32)
            if bbox.shape == (2, 4):
                bbox_vals.append(bbox.reshape(-1))
            for key in ("agent1_current", "agent2_current"):
                cur = ex.get(key)
                if isinstance(cur, str):
                    p = resolve_path(ds.base_root, cur)
                    if p.exists():
                        img_ok += 1
                    vc, vf = token_paths_for_frame(ds.base_root, ds.cache_root, p)
                    if vc.exists() and vf.exists():
                        cache_ok += 1
        x_std = float(np.std(np.concatenate(xs))) if xs else float("nan")
        y_std = float(np.std(np.concatenate(ys))) if ys else float("nan")
        th_std = float(np.std(np.concatenate(ths))) if ths else float("nan")
        bbox_mean = float(np.mean(np.concatenate(bbox_vals))) if bbox_vals else float("nan")
        print(
            f"[SANITY] samples_checked={n} xy_std=({x_std:.4f},{y_std:.4f}) theta_std={th_std:.4f} "
            f"bbox_mean={bbox_mean:.4f} current_img_ok={img_ok}/{2*n} current_cache_ok={cache_ok}/{2*n}",
            flush=True,
        )
    except Exception as exc:
        print(f"[SANITY] skipped due to error: {exc}", flush=True)


def compute_total_grad_norm(parameters, norm_type: float = 2.0) -> float:
    grads = [p.grad for p in parameters if getattr(p, "grad", None) is not None]
    if not grads:
        return 0.0
    device = grads[0].device
    total = torch.zeros([], device=device)
    for grad in grads:
        total += grad.detach().data.norm(norm_type).pow(norm_type)
    return float(total.pow(1.0 / norm_type).item())


@dataclass
# ----------------------- 训练配置与模型构建 -----------------------

class TrainConfig:
    """训练配置。

    beta_nav 是主路点损失权重；beta_bbox/beta_visible 默认 0，
    表示当前主要训练规划能力，grounding 头先作为可选辅助监督。
    """

    train_json: str
    out_dir: str = "/data/hdt/ntv_data/ckpt/ckpts_multi_agent"
    val_json: Optional[str] = None
    llm_name: str = "Qwen/Qwen3-0.6B"
    vision_feat_dim: int = 1536
    n_waypoints: int = 8
    action_dims: int = 3
    history: int = 31
    cache_root: Optional[str] = None
    online_encode_missing: bool = False
    image_size: int = 384
    epochs: int = 1
    batch_size: int = 2
    grad_accum_steps: int = 1
    lr: float = 2e-5
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    mixed_precision: bool = True
    seed: int = 0
    num_workers: int = 4
    distributed: bool = False
    dist_backend: str = "nccl"
    freeze_llm: bool = True
    use_angle_tvi: bool = False
    insert_time_tokens: bool = True
    no_tanh_actions: bool = True
    alpha_xy: Optional[float] = 2.0
    beta_nav: float = 100.0
    beta_bbox: float = 0.0
    beta_visible: float = 0.0
    return_token_logits: bool = False
    log_every: int = 10
    csv_logging: bool = True
    progress: bool = True
    save_every: int = 100
    max_ckpts: int = 3
    eval_every: int = 0
    eval_batches: int = 8
    final_wp_threshold: float = 0.2
    resume: bool = False
    resume_ckpt: Optional[str] = None
    dry_run: bool = False


def build_dataset(path: str, cfg: TrainConfig) -> MultiAgentJsonDataset:
    """根据 TrainConfig 创建双 Agent Dataset。"""
    return MultiAgentJsonDataset(
        MultiAgentDataConfig(
            train_json=path,
            n_waypoints=cfg.n_waypoints,
            history=cfg.history,
            cache_root=cfg.cache_root,
            action_dims=cfg.action_dims,
            online_encode_missing=cfg.online_encode_missing,
            image_size=cfg.image_size,
        )
    )


def build_model(cfg: TrainConfig) -> MultiAgentOpenTrackVLA:
    """创建 MultiAgentOpenTrackVLA，并把训练参数映射到模型配置。"""
    model_cfg = MultiAgentModelConfig(
        llm_name=cfg.llm_name,
        freeze_llm=cfg.freeze_llm,
        n_waypoints=cfg.n_waypoints,
        action_dims=cfg.action_dims,
        use_angle_tvi=cfg.use_angle_tvi,
        insert_time_tokens=cfg.insert_time_tokens,
        use_tanh_actions=(not cfg.no_tanh_actions),
        alpha_xy=cfg.alpha_xy,
        return_token_logits=cfg.return_token_logits,
    )
    return MultiAgentOpenTrackVLA(model_cfg, vision_feat_dim=cfg.vision_feat_dim)


# ----------------------- checkpoint、损失与验证 -----------------------

def save_checkpoint(
    path: Path,
    model: nn.Module,
    optim: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    cfg: TrainConfig,
    epoch: int,
    step: int,
) -> None:
    """保存训练状态。

    保存底层模型参数，DDP 情况下去掉外层 wrapper，便于单卡/多卡互相加载。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    model_to_save = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "model_state": model_to_save.state_dict(),
            "optim_state": optim.state_dict(),
            "scaler_state": scaler.state_dict() if scaler.is_enabled() else None,
            "config": cfg.__dict__,
        },
        str(path),
    )


def prune_checkpoints(out_dir: Path, max_ckpts: int) -> None:
    """只保留最近 max_ckpts 个 checkpoint，避免磁盘被训练过程占满。"""
    if max_ckpts <= 0:
        return
    ckpts = sorted(out_dir.glob("model_epoch*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in ckpts[max_ckpts:]:
        try:
            old.unlink()
        except Exception:
            pass


def forward_loss(model: nn.Module, batch: Dict[str, Any], cfg: TrainConfig, device: torch.device) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """一次 forward 并计算总损失。

    总损失:
    loss = beta_nav * waypoint_loss
         + beta_bbox * bbox_refine_loss
         + beta_visible * visibility_loss

    默认 beta_bbox/beta_visible 为 0，便于先稳定训练双 Agent 路点规划。
    """
    out = model(
        coarse_tokens=batch["coarse_tokens"].to(device),
        coarse_tidx=batch["coarse_tidx"].to(device),
        fine_tokens=batch["fine_tokens"].to(device),
        fine_tidx=batch["fine_tidx"].to(device),
        instructions=batch["instruction"],
        bbox_feat=batch["bbox_feat"].to(device),
        yaw_hist=batch["yaw_hist"].to(device) if cfg.use_angle_tvi else None,
        yaw_curr=batch["yaw_curr"].to(device) if cfg.use_angle_tvi else None,
    )
    # 模型主输出: (B, 2, N, 3)
    pred = out["waypoints"]
    gt = batch["waypoints"].to(device)
    mask = batch["valid_mask"].to(device)
    model_inspect = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    pred_n, gt_n = normalize_xy_by_alpha(pred, gt, getattr(model_inspect, "alpha_task", None))
    loss_nav = masked_mse_multi(pred_n, gt_n, mask)

    # Grounding 辅助监督：bbox refinement 目标就是输入 bbox，本质上约束 head 不要乱漂。
    refined_bbox = out.get("refined_bbox")
    if cfg.beta_bbox != 0.0 and refined_bbox is not None:
        bbox_target = batch["bbox_feat"].to(device)
        loss_bbox = F.mse_loss(refined_bbox.float(), bbox_target.float())
    else:
        loss_bbox = loss_nav.new_tensor(0.0)

    # 可见性辅助监督：target_visible 来自 UnrealZoo 采集时的 info 字段。
    visible_logits = out.get("visible_logits")
    visible_pred = out.get("visible_score")
    if cfg.beta_visible != 0.0 and visible_logits is not None:
        visible_target = batch["visible"].to(device)
        loss_visible = F.binary_cross_entropy_with_logits(visible_logits.float(), visible_target.float())
    elif cfg.beta_visible != 0.0 and visible_pred is not None:
        visible_target = batch["visible"].to(device)
        # 兼容旧模型输出：BCE 在 AMP autocast 中不安全，因此这里显式关闭 autocast。
        with torch.autocast(device_type=device.type, enabled=False):
            loss_visible = F.binary_cross_entropy(visible_pred.float().clamp(1e-5, 1.0 - 1e-5), visible_target.float())
    else:
        loss_visible = loss_nav.new_tensor(0.0)

    loss = cfg.beta_nav * loss_nav + cfg.beta_bbox * loss_bbox + cfg.beta_visible * loss_visible
    metrics = {
        "loss_nav": loss_nav.detach(),
        "loss_bbox": loss_bbox.detach(),
        "loss_visible": loss_visible.detach(),
        "pred": pred.detach(),
    }
    return loss, metrics


@torch.inference_mode()
def evaluate(model: nn.Module, ds: MultiAgentJsonDataset, cfg: TrainConfig, device: torch.device) -> Dict[str, float]:
    """小规模验证：返回平均 loss、最终路点 EPE 和 hit rate。"""
    model_eval = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    model_eval.eval()
    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=min(2, cfg.num_workers),
        pin_memory=True,
        collate_fn=collate_batch,
    )
    total_loss = 0.0
    total_count = 0
    final_errors: List[float] = []
    batches = 0
    for batch in dl:
        loss, metrics = forward_loss(model_eval, batch, cfg, device)
        pred = metrics["pred"].float()
        gt = batch["waypoints"].to(device).float()
        epe = torch.linalg.norm(pred[:, :, -1, :2] - gt[:, :, -1, :2], dim=-1)
        final_errors.extend(epe.reshape(-1).cpu().tolist())
        bs = pred.size(0)
        total_loss += float(loss.item()) * bs
        total_count += bs
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
    model.train()
    return {
        "loss": total_loss / max(1, total_count),
        "final_epe": epe_mean,
        "hit": hit,
    }


# ----------------------- 训练主循环 -----------------------

def train(cfg: TrainConfig) -> None:
    """训练主函数。"""
    use_ddp = bool(cfg.distributed)
    if use_ddp:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(backend=cfg.dist_backend, init_method="env://", device_id=device)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rank = 0
        world_size = 1
        local_rank = 0

    set_seed(cfg.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed + rank)

    if rank == 0:
        print(f"[INIT] train_json={cfg.train_json} out_dir={cfg.out_dir}", flush=True)
    ds = build_dataset(cfg.train_json, cfg)
    if rank == 0:
        dataset_sanity_report(ds, cfg)

    if cfg.dry_run:
        # dry_run 不加载 LLM，只验证 Dataset/cache/shape，适合处理完数据后先跑一次。
        item = ds[0]
        print("[DRY_RUN] first item shapes:")
        for key in ("coarse_tokens", "coarse_tidx", "fine_tokens", "fine_tidx", "bbox_feat", "waypoints", "valid_mask"):
            val = item[key]
            print(f"  {key}: {tuple(val.shape)}")
        print(f"  instruction: {item['instruction']}")
        return

    if cfg.alpha_xy is None:
        # 根据训练集路点半径的 95 分位自动估计 XY 缩放。
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
                print(f"[AUTO_ALPHA] alpha_xy={cfg.alpha_xy:.4f}", flush=True)

    try:
        # 从 cache token 自动检测 vision_feat_dim，避免 DINO/SigLIP 版本或缓存维度变化导致配置不一致。
        sample = ds[0]
        detected_dim = int(sample["fine_tokens"].shape[-1])
        if detected_dim != cfg.vision_feat_dim:
            if rank == 0:
                print(f"[AUTO_DIM] vision_feat_dim {cfg.vision_feat_dim} -> {detected_dim}", flush=True)
            cfg.vision_feat_dim = detected_dim
    except Exception as exc:
        if rank == 0:
            print(f"[AUTO_DIM] skipped: {exc}", flush=True)

    sampler = torch.utils.data.distributed.DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True) if use_ddp else None
    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=(sampler is None),
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_batch,
        sampler=sampler,
    )
    if rank == 0:
        print(f"[INIT] samples={len(ds)} batches={len(dl)} batch_per_gpu={cfg.batch_size}", flush=True)

    model = build_model(cfg).to(device)
    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
            broadcast_buffers=False,
        )

    if rank == 0:
        model_inspect = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        total = sum(p.numel() for p in model_inspect.parameters())
        trainable = sum(p.numel() for p in model_inspect.parameters() if p.requires_grad)
        print(f"[PARAMS] total={total:,} trainable={trainable:,} ({100.0 * trainable / max(1, total):.2f}%)", flush=True)

    optim = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=cfg.lr, weight_decay=cfg.weight_decay)
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
            model_to_load.load_state_dict(cleanup_state_dict_keys(obj.get("model_state", {})), strict=False)
            if obj.get("optim_state") is not None:
                optim.load_state_dict(obj["optim_state"])
            if obj.get("scaler_state") is not None and scaler.is_enabled():
                scaler.load_state_dict(obj["scaler_state"])
            start_epoch = int(obj.get("epoch", 0))
            step = int(obj.get("step", 0))
            if rank == 0:
                print(f"[RESUME] loaded {ckpt_path} epoch={start_epoch} step={step}", flush=True)

    out_dir = Path(cfg.out_dir)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    val_ds = build_dataset(cfg.val_json, cfg) if (cfg.val_json and rank == 0) else None
    ema_loss: Optional[float] = None
    last_log = time.time()

    for epoch in range(start_epoch, cfg.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        epoch_start = time.time()
        pbar = None
        if rank == 0 and cfg.progress and tqdm is not None:
            pbar = tqdm(total=len(dl), desc=f"epoch {epoch + 1}/{cfg.epochs}", dynamic_ncols=True, file=sys.stdout)

        for batch_idx, batch in enumerate(dl, start=1):
            do_step = (batch_idx % max(1, cfg.grad_accum_steps) == 0) or (batch_idx == len(dl))
            sync_ctx = model.no_sync() if use_ddp and hasattr(model, "no_sync") and not do_step else nullcontext()
            amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled) if amp_enabled else nullcontext()

            with sync_ctx:
                with amp_ctx:
                    loss, metrics = forward_loss(model, batch, cfg, device)
                scaler.scale(loss / max(1, cfg.grad_accum_steps)).backward()

            if pbar is not None:
                pbar.update(1)
            if not do_step:
                continue

            grad_norm = 0.0
            if cfg.grad_clip is not None and cfg.grad_clip > 0:
                # 先 unscale 再裁剪，保证 AMP 下 grad_clip 数值正确。
                scaler.unscale_(optim)
                grad_norm = compute_total_grad_norm(model.parameters())
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)
            step += 1

            if rank == 0 and (step % cfg.log_every == 0):
                # 日志里的 final_epe 是两个 Agent 最终路点 XY 误差的平均值。
                now = time.time()
                elapsed = now - last_log
                last_log = now
                loss_val = float(loss.detach().item())
                ema_loss = loss_val if ema_loss is None else (0.98 * ema_loss + 0.02 * loss_val)
                pred = metrics["pred"].float()
                gt = batch["waypoints"].to(device).float()
                final_epe = torch.linalg.norm(pred[:, :, -1, :2] - gt[:, :, -1, :2], dim=-1).mean().item()
                eta = format_duration((time.time() - epoch_start) / max(1, batch_idx) * max(0, len(dl) - batch_idx))
                msg = (
                    f"[TRAIN] epoch={epoch + 1}/{cfg.epochs} batch={batch_idx}/{len(dl)} step={step} eta={eta} "
                    f"loss={loss_val:.5f} ema={ema_loss:.5f} nav={metrics['loss_nav'].item():.5f} "
                    f"bbox={metrics['loss_bbox'].item():.5f} vis={metrics['loss_visible'].item():.5f} "
                    f"final_epe={final_epe:.4f} grad={grad_norm:.3f} dt={elapsed:.2f}s"
                )
                if pbar is not None:
                    pbar.write(msg)
                else:
                    print(msg, flush=True)
                if cfg.csv_logging:
                    csv_path = out_dir / "train_log.csv"
                    write_header = not csv_path.exists()
                    with csv_path.open("a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        if write_header:
                            writer.writerow(["epoch", "step", "loss", "loss_ema", "loss_nav", "loss_bbox", "loss_visible", "final_epe", "grad_norm"])
                        writer.writerow(
                            [
                                epoch,
                                step,
                                loss_val,
                                ema_loss,
                                float(metrics["loss_nav"].item()),
                                float(metrics["loss_bbox"].item()),
                                float(metrics["loss_visible"].item()),
                                final_epe,
                                grad_norm,
                            ]
                        )

            if rank == 0 and cfg.save_every > 0 and step % cfg.save_every == 0:
                ckpt = out_dir / f"model_epoch{epoch:02d}_step{step:06d}.pt"
                save_checkpoint(ckpt, model, optim, scaler, cfg, epoch, step)
                prune_checkpoints(out_dir, cfg.max_ckpts)
                print(f"[CKPT] saved {ckpt}", flush=True)

            if rank == 0 and val_ds is not None and cfg.eval_every > 0 and step % cfg.eval_every == 0:
                stats = evaluate(model, val_ds, cfg, device)
                print(
                    f"[VAL] step={step} loss={stats['loss']:.5f} final_epe={stats['final_epe']:.4f} "
                    f"hit@{cfg.final_wp_threshold}={stats['hit']:.3f}",
                    flush=True,
                )

        if pbar is not None:
            pbar.close()
        if rank == 0:
            ckpt = out_dir / f"model_epoch{epoch:02d}_step{step:06d}_final.pt"
            save_checkpoint(ckpt, model, optim, scaler, cfg, epoch + 1, step)
            prune_checkpoints(out_dir, cfg.max_ckpts)
            print(f"[CKPT] saved epoch final {ckpt}", flush=True)

    if use_ddp:
        dist.barrier()
        dist.destroy_process_group()
    if rank == 0:
        print(f"[DONE] training complete step={step}", flush=True)


# ----------------------- 命令行参数与程序入口 -----------------------

def parse_args() -> TrainConfig:
    ap = argparse.ArgumentParser(description="Train MultiAgentOpenTrackVLA on two-agent JSONL data.")
    ap.add_argument("--train_json", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="/data/hdt/ntv_data/ckpt/ckpts_multi_agent")
    ap.add_argument("--val_json", type=str, default=None)
    ap.add_argument("--llm_name", type=str, default="Qwen/Qwen3-0.6B")
    ap.add_argument("--vision_feat_dim", type=int, default=1536)
    ap.add_argument("--n_waypoints", type=int, default=8)
    ap.add_argument("--action_dims", type=int, default=3)
    ap.add_argument("--history", type=int, default=31)
    ap.add_argument("--cache_root", type=str, default=None)
    ap.add_argument("--online_encode_missing", action="store_true")
    ap.add_argument("--image_size", type=int, default=384)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum_steps", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--mixed_precision", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--distributed", action="store_true")
    ap.add_argument("--dist_backend", type=str, default="nccl")
    ap.add_argument("--freeze_llm", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--use_angle_tvi", action="store_true")
    ap.add_argument("--insert_time_tokens", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--no_tanh_actions", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--alpha_xy", type=float, default=2.0)
    ap.add_argument("--auto_alpha_xy", action="store_true", help="Set alpha_xy from training waypoint percentile.")
    ap.add_argument("--beta_nav", type=float, default=10.0)
    ap.add_argument("--beta_bbox", type=float, default=0.0)
    ap.add_argument("--beta_visible", type=float, default=0.0)
    ap.add_argument("--return_token_logits", action="store_true")
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--csv_logging", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--save_every", type=int, default=100)
    ap.add_argument("--max_ckpts", type=int, default=3)
    ap.add_argument("--eval_every", type=int, default=0)
    ap.add_argument("--eval_batches", type=int, default=8)
    ap.add_argument("--final_wp_threshold", type=float, default=0.2)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--resume_ckpt", type=str, default=None)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    data = vars(args)
    if data.pop("auto_alpha_xy"):
        data["alpha_xy"] = None
    return TrainConfig(**data)


if __name__ == "__main__":
    train(parse_args())
