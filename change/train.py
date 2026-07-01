#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OpenTrackVLA 通用训练入口，支持单 Agent 与双 Agent 训练模式。

整体功能：
- 读取 JSON/JSONL 标签与预缓存视觉 token，构造训练和验证 DataLoader。
- 完成模型构建、损失计算、梯度累积、混合精度、DDP、日志和 checkpoint 管理。
- 不带 ``--multi_agent`` 时训练 Habitat/UnrealZoo 单 Agent 模型。
- 带 ``--multi_agent`` 时训练 ``model.py`` 中的双 Agent 模型。
- 再加 ``--base_model`` 时训练 model.py-base 对照：只保留双视觉流、TVI、
  Agent/View/Kind 编码、双 ACT 和双 waypoint planner。
- 再加 ``--separate_agent_context`` 时训练另一个 base 对照：两个 Agent 分别
  送入同一个 LLM，上下文不互相拼接，只比较独立 planner 输出。

单 Agent核心：
- ``JsonTrackingDataset`` / ``TrainConfig``：数据读取与训练配置。
- ``train`` / ``_run_inference``：单 Agent训练与离线推理。

双 Agent核心：
- ``MultiAgentJsonDataset`` / ``MultiAgentTrainConfig``：双视角样本与配置。
- ``build_multi_agent_model``：构建 ``model.py::MultiAgentOpenTrackVLA`` 或
  ``model.py::MultiAgentSeparateOpenTrackVLA``。
- ``forward_multi_agent_loss``：计算 waypoint 主损失，可选 bbox/visibility/relative pose 辅助损失。
- ``train_multi_agent`` / ``evaluate_multi_agent``：训练与验证主循环。

主要输入输出：
- 输入为轨迹标签、两个 Agent 的视觉 token、bbox、可见性和文本指令。
- 输出 checkpoint、``train_log.csv``、可选可视化，以及验证损失。

Anchor Diffusion 不从本文件直接启动，应使用
``train_unrealzoo_anchor_diffusion.py``。
"""

from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass
from datetime import timedelta
from typing import List, Tuple, Optional, Dict, Any
import os, sys, json, math, argparse, time, csv
from pathlib import Path
from contextlib import nullcontext
from PIL import Image, ImageDraw

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist

from transformers import AutoTokenizer, AutoModel
from tools.cache_gridpool import VisionFeatureCacher, VisionCacheConfig, grid_pool_tokens, adapt_siglip_grid
from tqdm import tqdm


MULTI_AGENT_COOP_INSTRUCTION = (
    "The aerial drone and the ground robot dog must cooperatively track the same target person. "
    "The drone should follow the person from the air, and the robot dog should follow the same person on the ground."
)


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
    """Load dataset examples from a JSON list, a JSONL file, or a directory (recursively).
    - If directory: recursively loads all .jsonl files in subfolders.
    - If file: supports .json (list of dicts) or .jsonl (one JSON per line).
    """
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

# ----------------------- 单 Agent数据完整性检查 -----------------------

def _dataset_sanity_report(ds: 'JsonTrackingDataset', cfg: 'TrainConfig', max_items: int = 512):
    try:
        import numpy as _np
        n = min(max_items, len(ds))
        xs, ys, thetas = [], [], []
        mask_cov = []
        yaw_hist_present = 0
        yaw_curr_present = 0
        img_ok = 0
        for i in range(n):
            ex = ds.get_example(i)
            # compute target waypoints without triggering token encoding
            if 'waypoints' in ex:
                wp = _np.asarray(ex['waypoints'], dtype=_np.float32)
            elif 'actions' in ex:
                dt = float(ex.get('dt', ds.cfg.default_dt))
                wp = integrate_actions_to_waypoints(_np.asarray(ex['actions'], dtype=_np.float32), cfg.n_waypoints, dt)
            else:
                continue
            xs.append(wp[:, 0])
            if wp.shape[1] >= 2:
                ys.append(wp[:, 1])
            if wp.shape[1] >= 3:
                thetas.append(wp[:, 2])
            # valid mask coverage
            if 'valid_mask' in ex and isinstance(ex['valid_mask'], list):
                mv = _np.asarray(ex['valid_mask'], dtype=bool)
                mask_cov.append(float(mv.mean()))
            elif 'valid_idx' in ex and isinstance(ex['valid_idx'], list):
                mv = _np.zeros(cfg.n_waypoints, dtype=bool)
                idx = _np.asarray(ex['valid_idx'], dtype=int)
                mv[_np.clip(idx, 0, cfg.n_waypoints-1)] = True
                mask_cov.append(float(mv.mean()))
            # yaw fields presence
            if 'yaw_hist' in ex:
                yaw_hist_present += 1
            if 'yaw_curr' in ex:
                yaw_curr_present += 1
            # quick current image existence check (without IO)
            cur_rel = Path(ex.get('current', ''))
            if str(cur_rel):
                cur_abs = cur_rel if cur_rel.is_absolute() else (ds.base_root / cur_rel)
                if cur_abs.exists():
                    img_ok += 1
        if xs:
            x = _np.concatenate(xs)
            x_mu, x_sd = float(_np.mean(x)), float(_np.std(x))
        else:
            x_mu = x_sd = float('nan')
        if ys:
            y = _np.concatenate(ys)
            y_mu, y_sd = float(_np.mean(y)), float(_np.std(y))
        else:
            y_mu = y_sd = float('nan')
        th_sd = float(_np.std(_np.concatenate(thetas))) if thetas else float('nan')
        cov_mu = float(_np.mean(mask_cov)) if mask_cov else float('nan')
        print(f"[SANITY] GT x(mean={x_mu:.3f}, std={x_sd:.3f}) y(mean={y_mu:.3f}, std={y_sd:.3f}) theta(std={th_sd:.3f}) | mask_cov_mean={cov_mu:.3f}")
        print(f"[SANITY] yaw_hist_present={yaw_hist_present}/{n} yaw_curr_present={yaw_curr_present}/{n} | current_img_exists={img_ok}/{n}")
        # simple warnings
        if _np.isfinite(y_sd) and y_sd < 0.05:
            print("[SANITY][warn] GT lateral std is very small; model may learn straight lines.")
        if _np.isfinite(cov_mu) and cov_mu < 0.2:
            print("[SANITY][warn] Many waypoints are invalid; training signal may be sparse.")
    except Exception as _e:
        print(f"[SANITY] skipped due to error: {_e}")


# ----------------------- 单 Agent模型基础组件 -----------------------

class TVIEmbedder(nn.Module):
    """Temporal-Viewpoint Indicator with token insertion.
    - make_time_token(t, kind_id, view_id)
    - make_angle_token(theta, kind_id, view_id) -> uses [sinθ, cosθ] projection
    kind_id: 0 = coarse/history, 1 = fine/current.
    """
    def __init__(self, d_model: int, max_time: int = 4096, max_views: int = 1):
        super().__init__()
        self.time_emb   = nn.Embedding(max_time, d_model)
        self.view_emb   = nn.Embedding(max_views, d_model)
        self.kind_emb   = nn.Embedding(2, d_model)
        self.angle_proj = nn.Linear(2, d_model)

    def make_time_token(self, t_scalar: int, kind_id: int, view_id: int = 0,
                        device: Optional[torch.device] = None) -> torch.Tensor:
        tok = self.time_emb.weight[t_scalar] + self.view_emb.weight[view_id] + self.kind_emb.weight[kind_id]
        return tok.to(device) if device is not None else tok

    def make_angle_token(self, theta: float, kind_id: int, view_id: int = 0,
                         device: Optional[torch.device] = None) -> torch.Tensor:
        # project [sinθ, cosθ] into d_model
        theta = (theta + math.pi) % (2*math.pi) - math.pi
        sincos = torch.tensor([math.sin(theta), math.cos(theta)],
                              dtype=self.angle_proj.weight.dtype,
                              device=device)
        ang = F.linear(sincos, self.angle_proj.weight, self.angle_proj.bias)
        tok = ang + self.view_emb.weight[view_id] + self.kind_emb.weight[kind_id]
        return tok


class CrossModalityProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim), nn.GELU(),
            nn.Linear(out_dim, out_dim)
        )
    def forward(self, x): return self.net(x)


class PlannerHead3L(nn.Module):
    """Three-layer MLP A_θ mapping E_A^T → normalized waypoints â ∈ [-1,1]."""
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
            y = torch.tanh(y)                 # bound to [-1,1]
        return y.view(-1, self.nw, self.ad)   # (B, M, D_action)


# ----------------------- 单 Agent模型配置与主干 -----------------------

@dataclass
class ModelConfig:
    llm_name: str = "Qwen/Qwen3-0.6B"
    freeze_llm: bool = False
    n_waypoints: int = 8
    max_time: int = 4096
    beta_nav: float = 100.0
    use_angle_tvi: bool = False     # single-cam default: off
    # Action/target configuration
    use_tanh_actions: bool = True   # allow removing tanh cap via flag
    alpha_xy: Optional[float] = 1.0  # Optional scalar to scale XY only; yaw stays unscaled


class OpenTrackVLA(nn.Module):
    def __init__(self, cfg: ModelConfig, vision_feat_dim: int):
        super().__init__()
        self.cfg = cfg
        rank = int(os.environ.get("RANK", "0"))
        t0 = time.time()
        print(f"[MODEL][rank {rank}] loading LLM weights: {cfg.llm_name}", flush=True)
        self.llm = AutoModel.from_pretrained(cfg.llm_name, dtype=torch.bfloat16 if torch.cuda.is_available() else None)
        print(f"[MODEL][rank {rank}] LLM weights loaded in {time.time() - t0:.1f}s", flush=True)
        self.llm.requires_grad_(not cfg.freeze_llm)
        t0 = time.time()
        print(f"[MODEL][rank {rank}] loading tokenizer: {cfg.llm_name}", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.llm_name)
        print(f"[MODEL][rank {rank}] tokenizer loaded in {time.time() - t0:.1f}s", flush=True)
        self.D = self.llm.config.hidden_size
        self.proj = CrossModalityProjector(vision_feat_dim, self.D)
        # Always keep projector trainable regardless of LLM freeze
        self.proj.requires_grad_(True)
        self.tvi = TVIEmbedder(self.D, max_time=cfg.max_time)
        self.act_token = nn.Parameter(torch.zeros(1, 1, self.D))
        nn.init.normal_(self.act_token, std=0.02)
        # Determine target/action dimensionality
        action_dims = 3
        self.action_dims = action_dims
        self.planner = PlannerHead3L(self.D, cfg.n_waypoints, action_dims, use_tanh=cfg.use_tanh_actions)
        # Always keep planner trainable regardless of LLM freeze
        self.planner.requires_grad_(True)
        # Freeze unused TVI heads so DDP doesn't expect grads for them
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
            vec = [1.0] * action_dims
            alpha = torch.tensor(vec, dtype=torch.float32).view(1, 1, -1)
        self.register_buffer("alpha_task", alpha)

    def _embed_text(self, instructions: List[str], device):
        tok = self.tokenizer(instructions, return_tensors='pt', padding=True, truncation=True, max_length=128)
        tok = {k: v.to(device) for k, v in tok.items()}
        emb = self.llm.get_input_embeddings()(tok['input_ids'])
        return emb, tok['attention_mask']

    def _interleave_tvi(self, tokens: torch.Tensor, t_idx: torch.Tensor, kind_id: int,
                        yaw_per_frame: Optional[torch.Tensor] = None, use_angle: bool = False) -> torch.Tensor:
        """Insert TVI time (and optional angle) token before each frame's token block.
        tokens: (B, N, D_llm); t_idx: (B, N) frame ids; yaw_per_frame: (B, F) or None.
        Returns: (B, N + (1 or 2)*F, D_llm)
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
                yaw_curr: Optional[torch.Tensor] = None):
        device = next(self.parameters()).device
        B = coarse_tokens.size(0)
        # project to LLM space
        vis_c = self.proj(coarse_tokens.to(device))   # (B, Nc, D)
        vis_f = self.proj(fine_tokens.to(device))     # (B, Nf, D)
        # insert TVI tokens per frame
        vis_c = self._interleave_tvi(
            vis_c, coarse_tidx.to(device), kind_id=0,
            yaw_per_frame=yaw_hist, use_angle=self.cfg.use_angle_tvi
        )
        vis_f = self._interleave_tvi(
            vis_f, fine_tidx.to(device), kind_id=1,
            yaw_per_frame=yaw_curr, use_angle=self.cfg.use_angle_tvi
        )
        txt_emb, txt_mask = self._embed_text(instructions, device)  # (B, Ltxt, D), (B, Ltxt)
        extra = []
        act = self.act_token.expand(B, 1, -1)
        pieces = [txt_emb] + ([extra[0]] if extra else []) + [vis_c, vis_f, act]
        seq = torch.cat(pieces, dim=1).to(self.llm.dtype)
        extra_len = (extra[0].size(1) if extra else 0)
        attn = torch.cat([
            txt_mask,
            torch.ones(B, extra_len + vis_c.size(1) + vis_f.size(1) + 1, dtype=torch.long, device=device)  # +1 for ACT
        ], dim=1)
        out = self.llm(inputs_embeds=seq, attention_mask=attn, output_hidden_states=True, use_cache=False)
        h_act = out.last_hidden_state[:, -1, :]        # E_A^T (ACT is last)
        # Cast to float32 to match planner LayerNorm/Linear parameter dtype
        h_act = h_act.float()
        a_hat = self.planner(h_act)                # normalized [-1,1]
        tau_pred = a_hat * self.alpha_task             # absolute units
        return tau_pred


# ----------------------- 单 Agent训练数据集 -----------------------

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
        # Determine base root to prefix relative paths in JSON examples
        # Robustly ascend until 'frames' directory is found (common layout: <root>/frames/...)
        candidate = p if p.is_dir() else p.parent
        max_up = 4
        while max_up >= 0 and not (candidate / 'frames').exists():
            if candidate.parent == candidate:
                break
            candidate = candidate.parent
            max_up -= 1
        self.base_root = candidate
        # Determine cache root for token files (defaults to <base_root>/vision_cache)
        self.cache_root = Path(cfg.cache_root) if cfg.cache_root is not None else (self.base_root / "vision_cache")
        # Lazy online encoder (created on first use)
        self._online_encoder: Optional[VisionFeatureCacher] = None
        # Dataset storage: either eager JSON list or lazy JSONL/directory index
        self._lazy = False
        self._index: Optional[List[Tuple[str, int]]] = None  # list of (file_path, byte_offset) per example
        self.examples: Optional[List[Dict[str, Any]]] = None
        if p.is_file() and p.suffix.lower() == '.json':
            data = load_examples_from_path(cfg.train_json)
            assert isinstance(data, list) and len(data) > 0, "JSON file must contain a non-empty list"
            self.examples = data
        else:
            # Build lazy index over .jsonl file or directory of .jsonl files
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
                try:
                    with open(fp, 'rb') as f:
                        pos = 0
                        while True:
                            line = f.readline()
                            if not line:
                                break
                            if line.strip():
                                self._index.append((str(fp), pos))
                            pos += len(line)
                except Exception as _e:
                    raise _e
            if len(self._index) == 0:
                raise RuntimeError(f"No examples indexed from .jsonl sources under: {cfg.train_json}")
        # Target history length is configured; per-sample sequences will be padded/trimmed to this length
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
            # Use CPU when running with multiple workers to avoid GPU contention
            from torch.utils.data import get_worker_info
            worker_info = get_worker_info()
            use_cuda = torch.cuda.is_available() and (worker_info is None)
            cfg = VisionCacheConfig(image_size=384, batch_size=8, device=('cuda' if use_cuda else 'cpu'))
            self._online_encoder = VisionFeatureCacher(cfg)
            self._online_encoder.eval()
        return self._online_encoder

    @torch.inference_mode()
    def _encode_image_tokens(self, img_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        enc = self._get_online_encoder()
        pil = Image.open(str(img_path)).convert('RGB')
        tok_dino, Hp, Wp = enc._encode_dino([pil])               # (1, P, C_dino)
        tok_sigl = enc._encode_siglip([pil], out_hw=(Hp, Wp))    # (1, P, C_sigl)
        Vt_cat = torch.cat([tok_dino, tok_sigl], dim=-1)         # (1, P, C_total)
        Vfine = grid_pool_tokens(Vt_cat, Hp, Wp, out_tokens=64)  # (1, 64, C_total)
        Vcoarse = grid_pool_tokens(Vt_cat, Hp, Wp, out_tokens=4) # (1, 4,  C_total)
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

        # fine current (load first to determine feature dimensionality for zero padding)
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
            fine_tokens = self._load_tokens(str(curr_tok_path))  # (64, C)
        except Exception as e:            
            # Fallback: encode online and save for reuse
            curr_token_dir.mkdir(parents=True, exist_ok=True)
            vc, vf = self._encode_image_tokens(abs_curr_img)
            print (curr_tok_path)
            try:
                torch.save(vf.half(), str(curr_tok_path))
            except Exception as e:
                print (str(e))
                pass
            fine_tokens = vf
        fine_tidx = torch.full((fine_tokens.size(0),), fill_value=H, dtype=torch.long)

        # coarse history with left-padding using earliest available token (edge padding)
        imgs_src = ex.get('images', [])
        imgs_trim = imgs_src[-H:]
        missing = H - len(imgs_trim)
        coarse_list, coarse_tidx = [], []
        first_tok: Optional[torch.Tensor] = None
        current_vc: Optional[torch.Tensor] = None
        for t in range(H):
            if t < missing:
                # placeholder; will fill with first_tok after we load at least one real token
                tok = None
            else:
                img_p = imgs_trim[t - missing]
                # Map image path -> corresponding Vcoarse token path within cache_root
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
                except Exception as e:
                    # Fallback: encode online from image and save for reuse
                    token_dir.mkdir(parents=True, exist_ok=True)
                    vc, vf = self._encode_image_tokens(abs_img)
                    try:
                        torch.save(vc.half(), str(tok_path))
                    except Exception:
                        pass
                    tok = vc
                if first_tok is None:
                    first_tok = tok
            # Fill left padding with first available token (edge padding);
            # if none, use current frame's coarse tokens; fallback to zeros if that also fails
            if tok is None:
                if first_tok is not None:
                    tok = first_tok
                else:
                    # Try to obtain current coarse tokens
                    try:
                        if current_vc is None:
                            # Attempt to load cached current coarse tokens alongside fine
                            cur_coarse_name = rel_curr.stem + "_vcoarse.pt"
                            cur_coarse_path = curr_token_dir / cur_coarse_name
                            try:
                                current_vc = self._load_tokens(str(cur_coarse_path))
                            except Exception:
                                # Encode from current image if not cached
                                vc_tmp, _ = self._encode_image_tokens(abs_curr_img)
                                current_vc = vc_tmp
                        tok = current_vc
                    except Exception:
                        tok = torch.zeros(4, fine_tokens.size(1), dtype=torch.float32)
            coarse_list.append(tok)
            coarse_tidx.append(torch.full((tok.size(0),), fill_value=t, dtype=torch.long))
        coarse_tokens = torch.cat(coarse_list, dim=0)      # (H*4, C)
        coarse_tidx   = torch.cat(coarse_tidx, dim=0)      # (H*4,)

        # yaw (optional; used only if angle TVI is enabled)
        yaw_hist = torch.tensor(ex.get('yaw_hist', [0.0]*H), dtype=torch.float32)            # (H,)
        yaw_curr = torch.tensor(ex.get('yaw_curr', 0.0), dtype=torch.float32).view(1)        # (1,)

        # targets
        if 'waypoints' in ex:
            wp = torch.tensor(ex['waypoints'], dtype=torch.float32)
        else:
            assert 'actions' in ex, "JSON needs either 'waypoints' or 'actions'"
            dt = float(ex.get('dt', self.cfg.default_dt))
            traj = integrate_actions_to_waypoints(np.asarray(ex['actions'], dtype=np.float32), self.cfg.n_waypoints, dt)
            wp = torch.from_numpy(traj)

        # validity mask/idx (optional)
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
            'yaw_hist':      yaw_hist,     # (H,)
            'yaw_curr':      yaw_curr,     # (1,)
            'waypoints':     wp,           # (M, D_action)
            'valid_mask':    vm,           # (M,)
            'instruction':   ex.get('instruction', 'follow the person'),
            'current_path':  str(abs_curr_img),
        }
        return item


# ----------------------- 单 Agent损失、配置与训练流程 -----------------------

def mse_masked(pred: torch.Tensor, target: torch.Tensor, mask_waypoints: torch.Tensor) -> torch.Tensor:
    """Mean squared error over selected waypoints (absolute units)."""
    assert pred.shape == target.shape
    B, M, D = pred.shape
    mask = mask_waypoints.view(B, M, 1).expand(B, M, D)
    se = (pred - target).pow(2)
    se = se[mask]
    return se.mean() if se.numel() > 0 else pred.new_tensor(0.0)


def _compute_total_grad_norm(parameters, norm_type: float = 2.0) -> float:
    parameters = [p for p in parameters if p.grad is not None]
    if len(parameters) == 0:
        return 0.0
    device = parameters[0].grad.device
    if norm_type == float('inf'):
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
        return float(total_norm.item())
    total = torch.zeros([], device=device)
    for p in parameters:
        param_norm = p.grad.detach().data.norm(norm_type)
        total += param_norm.pow(norm_type)
    total = total.pow(1.0 / norm_type)
    return float(total.item())


@dataclass
class TrainConfig:
    train_json: str
    out_dir: str = '/data/hdt/ntv_data/ckpt/ckpts_qwen4'
    n_waypoints: int = 8
    history: int = 31
    llm_name: str = "Qwen/Qwen3-0.6B"
    epochs: int = 1
    batch_size: int = 12
    grad_accum_steps: int = 1
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    mixed_precision: bool = True
    vision_feat_dim: int = 1536
    seed: int = 0
    num_workers: int = 4
    # model
    freeze_llm: bool = False
    use_angle_tvi: bool = False
    beta_nav: float = 100.0
    cache_root: Optional[str] = None
    distributed: bool = False
    dist_backend: str = 'nccl'
    alpha_xy: Optional[float] = 1.0
    # logging
    log_every: int = 10
    csv_logging: bool = True
    progress: bool = True
    # trajectory saving
    save_trajectories: bool = False
    traj_subdir: str = 'trajectories'
    # visualization saving
    vis_every: int = 100          # save vis every N steps (0 = disabled)
    vis_samples: int = 4          # number of samples to visualize per batch
    # evaluation
    val_json: Optional[str] = None
    eval_every: int = 0
    eval_batches: int = 8
    final_wp_threshold: float = 0.2
    # single-episode evaluation
    episode_json: Optional[str] = None
    episode_eval_every: int = 0
    episode_threshold: float = 0.2
    episode_max_frames: int = 256
    # modeling options
    no_tanh_actions: bool = True
    # checkpoint retention
    max_ckpts: int = 2
    # resume
    resume: bool = False
    resume_ckpt: Optional[str] = None
    # inference
    infer_json: Optional[str] = None
    infer_ckpt: Optional[str] = None
    infer_out: str = './infer_out'
    infer_batches: int = 0
    infer_vis: bool = False
    infer_save_npz: bool = True


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    instr = [b['instruction'] for b in batch]
    return {
        'coarse_tokens': torch.stack([b['coarse_tokens'] for b in batch], dim=0),
        'coarse_tidx':   torch.stack([b['coarse_tidx']   for b in batch], dim=0),
        'fine_tokens':   torch.stack([b['fine_tokens']   for b in batch], dim=0),
        'fine_tidx':     torch.stack([b['fine_tidx']     for b in batch], dim=0),
        'yaw_hist':      torch.stack([b['yaw_hist']      for b in batch], dim=0),   # (B,H)
        'yaw_curr':      torch.stack([b['yaw_curr']      for b in batch], dim=0),   # (B,1)
        'waypoints':     torch.stack([b['waypoints']     for b in batch], dim=0),
        'valid_mask':    torch.stack([b['valid_mask']    for b in batch], dim=0),
        'instruction':   instr,
        'current_path':  [b['current_path'] for b in batch]
    }


def train(cfg: TrainConfig):
    torch.backends.cudnn.benchmark = True
    # Distributed setup
    use_ddp = bool(cfg.distributed)
    if use_ddp:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        # Set the rank-local CUDA device before any CUDA availability query.
        # Otherwise nonzero ranks can create a small stray context on cuda:0.
        torch.cuda.set_device(local_rank)
        is_cuda = True
        device = torch.device('cuda', local_rank)

        dist.init_process_group(
            backend=cfg.dist_backend,
            init_method='env://',
            device_id=device,  # or device_id=local_rank
        )

        rank = dist.get_rank()
        world_size = dist.get_world_size()

    else:
        is_cuda = torch.cuda.is_available()
        device = torch.device('cuda' if is_cuda else 'cpu')
        local_rank = 0
        rank = 0
        world_size = 1
    set_seed(cfg.seed)
    if is_cuda:
        torch.cuda.manual_seed(cfg.seed)

    print(f"[INIT][rank {rank}] world_size={world_size} device={device} | loading dataset {cfg.train_json}", flush=True)
    ds = JsonTrackingDataset(DataConfig(train_json=cfg.train_json, n_waypoints=cfg.n_waypoints, history=cfg.history, cache_root=cfg.cache_root))
    if rank == 0:
        _dataset_sanity_report(ds, cfg)
    # Auto-compute scalar XY scaling (alpha_xy) from dataset if not provided
    if cfg.alpha_xy is None:
        try:
            import numpy as _np
            vals = []
            sample_n = min(4000, len(ds))
            for i in range(sample_n):
                ex = ds.get_example(i)
                arr = None
                if 'waypoints' in ex:
                    arr = _np.asarray(ex['waypoints'], dtype=_np.float32)
                elif 'actions' in ex:
                    dt = float(ex.get('dt', ds.cfg.default_dt))
                    arr = integrate_actions_to_waypoints(_np.asarray(ex['actions'], dtype=_np.float32), cfg.n_waypoints, dt)
                if arr is None:
                    continue
                if arr.ndim == 1:
                    arr = arr[None, :]
                if arr.shape[1] >= 2:
                    r = _np.linalg.norm(arr[:, :2], axis=-1)
                    vals.append(r)
            if vals:
                allr = _np.concatenate(vals)
                alpha_est = float(_np.percentile(allr, 95))
                alpha_est = max(alpha_est, 1e-3)
                cfg.alpha_xy = alpha_est
                if rank == 0:
                    print(f"[auto_alpha_xy] alpha_xy set to {alpha_est:.3f} from dataset percentiles ({cfg.train_json})")
        except Exception as _e:
            if rank == 0:
                print(f"[auto_alpha_xy] skipped due to error: {_e}")
    # Auto-detect vision_feat_dim on every rank. This avoids an early NCCL
    # broadcast before both ranks have reached model construction.
    try:
        sample_item = ds[0]
        detected_dim = None
        if 'fine_tokens' in sample_item:
            detected_dim = sample_item['fine_tokens'].shape[-1]
        elif 'coarse_tokens' in sample_item:
            detected_dim = sample_item['coarse_tokens'].shape[-1]
        if detected_dim is not None and detected_dim != cfg.vision_feat_dim:
            if rank == 0:
                print(f"[AUTO_DIM] Detected vision_feat_dim={detected_dim} from dataset (config had {cfg.vision_feat_dim}), updating...", flush=True)
            cfg.vision_feat_dim = detected_dim
    except Exception as e:
        if rank == 0:
            print(f"[AUTO_DIM] Failed to auto-detect vision_feat_dim: {e}, using config value {cfg.vision_feat_dim}", flush=True)

    sampler = torch.utils.data.distributed.DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True) if use_ddp else None
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=(sampler is None), num_workers=cfg.num_workers,
                    pin_memory=True, collate_fn=collate_batch, sampler=sampler)
    if rank == 0:
        print(
            f"[INIT] dataloader ready | samples={len(ds)} local_batches/epoch={len(dl)} batch_per_gpu={cfg.batch_size}",
            flush=True,
        )

    print(f"[INIT][rank {rank}] loading model {cfg.llm_name}", flush=True)
    model = OpenTrackVLA(
        ModelConfig(
            llm_name=cfg.llm_name,
            freeze_llm=cfg.freeze_llm,
            n_waypoints=cfg.n_waypoints,
            beta_nav=cfg.beta_nav,
            use_angle_tvi=cfg.use_angle_tvi,
            use_tanh_actions=(not cfg.no_tanh_actions),
            alpha_xy=cfg.alpha_xy,
        ),
        vision_feat_dim=cfg.vision_feat_dim,
    )
    print(f"[INIT][rank {rank}] moving model to {device}", flush=True)
    model = model.to(device)
    print(f"[INIT][rank {rank}] model moved to {device}", flush=True)
    if use_ddp:
        print(f"[INIT][rank {rank}] wrapping model with DDP", flush=True)
        ddp_kwargs = {
            "device_ids": [local_rank],
            "output_device": local_rank,
            "find_unused_parameters": False,
            "broadcast_buffers": False,
        }
        if os.environ.get("DDP_INIT_SYNC", "1") == "0":
            try:
                import inspect
                if "init_sync" in inspect.signature(torch.nn.parallel.DistributedDataParallel).parameters:
                    ddp_kwargs["init_sync"] = False
                    print(f"[INIT][rank {rank}] DDP init_sync disabled", flush=True)
            except Exception:
                pass
        model = torch.nn.parallel.DistributedDataParallel(model, **ddp_kwargs)
        print(f"[INIT][rank {rank}] DDP ready", flush=True)
    print(f"[INIT][rank {rank}] model ready; entering training loop", flush=True)

    # Log trainable vs frozen parameters (rank 0 only)
    if rank == 0:
        try:
            from collections import defaultdict
            model_inspect = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
            total_params = sum(p.numel() for p in model_inspect.parameters())
            trainable_params = sum(p.numel() for p in model_inspect.parameters() if p.requires_grad)
            pct = (trainable_params / max(1, total_params)) * 100.0
            print(f"[PARAMS] total={total_params:,} trainable={trainable_params:,} ({pct:.2f}%)")
            group_counts = defaultdict(lambda: [0, 0])  # [total, trainable]
            for name, p in model_inspect.named_parameters():
                head = name.split('.')[0]
                n = p.numel()
                group_counts[head][0] += n
                if p.requires_grad:
                    group_counts[head][1] += n
            summary = ' '.join([f"{k}:{v[1]}/{v[0]}" for k, v in group_counts.items()])
            print(f"[PARAMS groups] {summary}")
            tn = [n for n, p in model_inspect.named_parameters() if p.requires_grad][:16]
            print(f"[TRAINABLE names (first 16)] {tn}")
        except Exception as _e:
            print(f"[PARAMS] logging skipped due to error: {_e}")

    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=cfg.lr, weight_decay=cfg.weight_decay)
    cfg.grad_accum_steps = max(1, int(cfg.grad_accum_steps))
    if rank == 0:
        eff_batch = cfg.batch_size * world_size * cfg.grad_accum_steps
        print(
            f"[TRAIN] batch_per_gpu={cfg.batch_size} world_size={world_size} "
            f"grad_accum_steps={cfg.grad_accum_steps} effective_batch={eff_batch}",
            flush=True,
        )
    # Mixed precision configuration
    amp_enabled = cfg.mixed_precision and is_cuda
    amp_dtype = torch.bfloat16  # switch to torch.float16 if you want fp16
    scaler = torch.amp.GradScaler('cuda', enabled=(amp_enabled and amp_dtype == torch.float16))

    # Optionally resume from checkpoint
    start_epoch = 0
    step = 0
    if cfg.resume:
        try:
            import glob as _glob
            ckpt_path = cfg.resume_ckpt
            if ckpt_path is None:
                pts = sorted(_glob.glob(os.path.join(cfg.out_dir, 'model_epoch*.pt')), key=lambda p: os.path.getmtime(p))
                ckpt_path = pts[-1] if pts else None
            if ckpt_path and os.path.exists(ckpt_path):
                obj = torch.load(ckpt_path, map_location=device)
                msd = obj.get('model_state', None)
                if msd:
                    msd = _cleanup_state_dict_keys(msd)
                    model_to_load = model.module if isinstance(
                        model, torch.nn.parallel.DistributedDataParallel
                    ) else model
                    model_to_load.load_state_dict(msd, strict=False)
                osd = obj.get('optim_state', None)
                if osd:
                    optim.load_state_dict(osd)
                ssd = obj.get('scaler_state', None)
                if ssd and scaler.is_enabled():
                    scaler.load_state_dict(ssd)
                start_epoch = int(obj.get('epoch', 0))
                step = int(obj.get('step', 0))
                if rank == 0:
                    print(f"[RESUME] Loaded {ckpt_path} | epoch={start_epoch} step={step}")
        except Exception as _e:
            if rank == 0:
                print(f"[RESUME] Skipped due to error: {_e}")

    if rank == 0:
        Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

    # continue step counter if resumed
    ema_loss: Optional[float] = None
    ema_nav: Optional[float] = None
    last_log_time = time.time()
    epoch_start_time = last_log_time
    for epoch in range(cfg.epochs):
        epoch_start_time = time.time()
        model.train()
        if use_ddp and sampler is not None:
            sampler.set_epoch(epoch)
        pbar = None
        if rank == 0 and cfg.progress and tqdm is not None:
            pbar = tqdm(
                total=len(dl),
                desc=f"epoch {epoch + 1}/{cfg.epochs}",
                dynamic_ncols=True,
                leave=True,
                file=sys.stdout,
                disable=False,
            )
        for step_in_epoch, batch in enumerate(dl, start=1):
            do_optim_step = (step_in_epoch % cfg.grad_accum_steps == 0) or (step_in_epoch == len(dl))
            sync_ctx = (
                model.no_sync()
                if use_ddp and hasattr(model, "no_sync") and not do_optim_step
                else nullcontext()
            )
            coarse_tokens = batch['coarse_tokens'].to(device)
            coarse_tidx   = batch['coarse_tidx'].to(device)
            fine_tokens   = batch['fine_tokens'].to(device)
            fine_tidx     = batch['fine_tidx'].to(device)
            yaw_hist      = batch['yaw_hist'].to(device)   # (B,H)
            yaw_curr      = batch['yaw_curr'].to(device)   # (B,1)
            gt_wp         = batch['waypoints'].to(device)
            valid_mask    = batch['valid_mask'].to(device)
            instr         = batch['instruction']

            amp_ctx = torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp_enabled) if amp_enabled else nullcontext()
            with sync_ctx:
                with amp_ctx:
                    tau_pred = model(
                        coarse_tokens, coarse_tidx,
                        fine_tokens, fine_tidx,
                        instr,
                        yaw_hist=yaw_hist if cfg.use_angle_tvi else None,
                        yaw_curr=yaw_curr if cfg.use_angle_tvi else None
                    )
                    # Option A: compute loss in normalized space — divide XY by alpha, keep yaw unscaled
                    # Get alpha vector from model (shape: (1,1,D_action)) and clamp to avoid div-by-zero
                    model_inspect = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
                    alpha_vec = getattr(model_inspect, 'alpha_task', None)
                    if alpha_vec is None:
                        # Fallback: no scaling information; compute loss in absolute space
                        pred_norm = tau_pred
                        gt_norm = gt_wp
                    else:
                        # Normalize only XY dims (0,1); leave others (e.g., yaw) unscaled
                        pred_norm = tau_pred
                        gt_norm = gt_wp
                        if pred_norm.size(-1) >= 2 and alpha_vec.size(-1) >= 2:
                            ax = alpha_vec[..., 0:2].clamp_min(1e-6)
                            pred_norm = pred_norm.clone()
                            gt_norm = gt_norm.clone()
                            pred_norm[..., 0:2] = pred_norm[..., 0:2] / ax
                            gt_norm[..., 0:2] = gt_norm[..., 0:2] / ax
                    L_nav = mse_masked(pred_norm, gt_norm, valid_mask)
                    L_QA = tau_pred.new_tensor(0.0)
                    loss = cfg.beta_nav * L_nav + L_QA
                scaler.scale(loss / cfg.grad_accum_steps).backward()
            # Save trajectories per-step (rank 0)
            if rank == 0 and cfg.save_trajectories:
                try:
                    traj_root = os.path.join(cfg.out_dir, cfg.traj_subdir)
                    os.makedirs(traj_root, exist_ok=True)
                    with torch.no_grad():
                        pred_np = tau_pred.detach().float().cpu().numpy()
                        gt_np = gt_wp.detach().float().cpu().numpy()
                        vm_np = valid_mask.detach().cpu().numpy()
                    Bcur = pred_np.shape[0]
                    for bi in range(Bcur):
                        fpath = os.path.join(traj_root, f"ep{epoch:02d}_st{step+1:06d}_b{bi:03d}.npz")
                        # step+1 so filenames are 1-indexed and align with printed step after increment
                        np.savez_compressed(
                            fpath,
                            pred=pred_np[bi],
                            gt=gt_np[bi],
                            valid_mask=vm_np[bi],
                            instruction=instr[bi],
                            epoch=epoch,
                            step=step+1
                        )
                except Exception:
                    pass
            if pbar is not None:
                pbar.update(1)
            if not do_optim_step:
                continue

            grad_norm_before = 0.0
            if cfg.grad_clip is not None:
                scaler.unscale_(optim)
                grad_norm_before = _compute_total_grad_norm([p for p in model.parameters() if getattr(p, 'grad', None) is not None])
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optim); scaler.update()
            optim.zero_grad(set_to_none=True)

            step += 1
            if rank == 0 and (step % cfg.log_every == 0):
                now = time.time()
                dt = now - last_log_time
                last_log_time = now
                B = coarse_tokens.size(0)
                lr = optim.param_groups[0]['lr']
                with torch.no_grad():
                    tp = tau_pred.detach().float()
                    gwp = gt_wp.detach().float()
                    vm = valid_mask.detach().float()
                    # Scale predictions to absolute units for logging if alpha is available
                    tp_abs = tp
                    try:
                        model_inspect = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
                        alpha_vec = getattr(model_inspect, 'alpha_task', None)
                        if alpha_vec is not None and alpha_vec.size(-1) >= tp.size(-1):
                            av = alpha_vec.to(tp.device, tp.dtype)
                            tp_abs = tp * av
                    except Exception:
                        pass
                    pred_mean = tp_abs.mean().item()
                    pred_std = tp_abs.std().item()
                    pred_absmax = tp_abs.abs().max().item()
                    gt_mean = gwp.mean().item()
                    gt_std = gwp.std().item()
                    mask_cov = vm.mean().item()
                    mse_total = (tp_abs - gwp).pow(2).mean().item()
                    ad = tp_abs.size(-1)
                    per_dim_mse = []
                    for d in range(min(4, ad)):
                        per_dim_mse.append(float((tp_abs[..., d] - gwp[..., d]).pow(2).mean().item()))
                loss_val = float(loss.detach().item())
                nav_val = float(L_nav.detach().item())
                ema_loss = loss_val if ema_loss is None else (0.98 * ema_loss + 0.02 * loss_val)
                ema_nav = nav_val if ema_nav is None else (0.98 * ema_nav + 0.02 * nav_val)
                progress_pct = 100.0 * step_in_epoch / max(1, len(dl))
                elapsed_epoch = now - epoch_start_time
                sec_per_batch = elapsed_epoch / max(1, step_in_epoch)
                eta_txt = _format_duration(sec_per_batch * max(0, len(dl) - step_in_epoch))

                mem_alloc_mb = mem_peak_mb = 0.0
                if torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass
                    mem_alloc_mb = torch.cuda.memory_allocated(device) / (1024**2)
                    mem_peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

                msg_lines = [
                    f"[PROGRESS] epoch {epoch + 1}/{cfg.epochs} batch {min(step_in_epoch, len(dl))}/{len(dl)} ({progress_pct:.1f}%) | global_step={step} | eta={eta_txt}",
                    f"epoch {epoch} step {step} | lr={lr:.2e} | loss={loss_val:.4f} (ema {ema_loss:.4f}) | L_nav={nav_val:.4f} (ema {ema_nav:.4f})",
                    f"  mask_cov={mask_cov:.3f} | grad_norm_preclip={grad_norm_before:.3f} | step_time={dt:.3f}s | throughput={B/dt:.2f} it/s",
                    f"  pred(mean={pred_mean:.3f}, std={pred_std:.3f}, absmax={pred_absmax:.3f}) | gt(mean={gt_mean:.3f}, std={gt_std:.3f})",
                    f"  mse_total={mse_total:.5f} | per_dim_mse={per_dim_mse} | mem_alloc={mem_alloc_mb:.1f}MB peak={mem_peak_mb:.1f}MB"
                ]
                if pbar is not None:
                    pbar.set_postfix(loss=f"{loss_val:.4f}", nav=f"{nav_val:.4f}", s=f"{dt:.2f}")
                    pbar.write("\n".join(msg_lines))
                else:
                    print("\n".join(msg_lines), flush=True)

                # Debug preview: print GT and Pred waypoints (absolute), and normalized XY if alpha is set
                """
                try:
                    import numpy as _np
                    model_inspect = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
                    alpha_vec = getattr(model_inspect, 'alpha_task', None)
                    pred_abs_b0 = tp[0].detach().cpu().numpy()
                    gt_abs_b0 = gwp[0].detach().cpu().numpy()
                    print("[WAYPOINTS abs][b0] pred=", _np.array2string(pred_abs_b0, precision=3, floatmode='fixed'))
                    print("[WAYPOINTS abs][b0]   gt=", _np.array2string(gt_abs_b0, precision=3, floatmode='fixed'))
                    if alpha_vec is not None and pred_abs_b0.shape[1] >= 2 and alpha_vec.size(-1) >= 2:
                        ax = alpha_vec[0, 0, 0:2].clamp_min(1e-6).detach().float().cpu()
                        pred_n_xy = (tp[0, :, 0:2].detach().cpu() / ax).numpy()
                        gt_n_xy = (gwp[0, :, 0:2].detach().cpu() / ax).numpy()
                        print(f"[WAYPOINTS norm][b0] alpha_xy={ax.numpy().tolist()} pred_xy=", _np.array2string(pred_n_xy, precision=3, floatmode='fixed'))
                        print("[WAYPOINTS norm][b0]   gt_xy=", _np.array2string(gt_n_xy, precision=3, floatmode='fixed'))
                except Exception:
                    pass
                """
                if cfg.csv_logging:
                    csv_path = os.path.join(cfg.out_dir, 'train_log.csv')
                    header = [
                        'epoch','step','lr','loss','loss_ema','L_nav','L_nav_ema','mask_cov',
                        'grad_norm_preclip','step_time','throughput_it_per_s','pred_mean','pred_std','pred_absmax',
                        'gt_mean','gt_std','mse_total','mem_alloc_mb','mem_peak_mb'
                    ] + [f'mse_dim_{i}' for i in range(len(per_dim_mse))]
                    write_header = not os.path.exists(csv_path)
                    try:
                        with open(csv_path, 'a', newline='') as f:
                            w = csv.writer(f)
                            if write_header:
                                w.writerow(header)
                            row = [
                                epoch, step, lr, loss_val, ema_loss, nav_val, ema_nav, mask_cov,
                                grad_norm_before, dt, (B/dt), pred_mean, pred_std, pred_absmax,
                                gt_mean, gt_std, mse_total, mem_alloc_mb, mem_peak_mb
                            ] + per_dim_mse
                            w.writerow(row)
                    except Exception:
                        pass

                # Visualize GT vs Pred on current images
                if rank == 0 and cfg.vis_every > 0 and step % cfg.vis_every == 0:
                    try:
                        vis_dir = os.path.join(cfg.out_dir, 'vis')
                        os.makedirs(vis_dir, exist_ok=True)
                        with torch.no_grad():
                            # Ensure predictions are in absolute units for visualization
                            pred_draw = tau_pred.detach().float()
                            pred_np = pred_draw.cpu().numpy()
                            gt_np = gwp.detach().float().cpu().numpy()
                            cur_paths = batch.get('current_path', [])
                        Bcur = pred_np.shape[0]
                        for bi in range(min(Bcur, cfg.vis_samples)):
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
                                    # y is left-positive in robot frame → screen x grows to right ⇒ subtract
                                    px = base_x - int(y * 120)
                                    py = base_y - int(x * 120)
                                    pts.append((px, py))
                                return pts
                            pts_pred = to_pxxy(pred_np[bi])
                            pts_gt   = to_pxxy(gt_np[bi])
                            # outline
                            for seq, color in ((pts_gt, (0,0,0)), (pts_pred, (0,0,0))):
                                for i2 in range(1, len(seq)):
                                    draw.line([seq[i2-1], seq[i2]], fill=color, width=10)
                            # body
                            for i2 in range(1, len(pts_gt)):
                                draw.line([pts_gt[i2-1], pts_gt[i2]], fill=(255, 200, 0), width=6)
                            for i2 in range(1, len(pts_pred)):
                                draw.line([pts_pred[i2-1], pts_pred[i2]], fill=(0, 255, 200), width=6)
                            # start points
                            if pts_gt:
                                r0 = 6
                                sx, sy = pts_gt[0]
                                draw.ellipse([sx-r0, sy-r0, sx+r0, sy+r0], fill=(255,255,255))
                            if pts_pred:
                                r0 = 6
                                sx, sy = pts_pred[0]
                                draw.ellipse([sx-r0, sy-r0, sx+r0, sy+r0], fill=(0,255,0))
                            try:
                                from inspect import isclass
                                model_cfg = (model.module.cfg if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.cfg)
                            except Exception:
                                pass
                            out_path = os.path.join(vis_dir, f"ep{epoch:02d}_st{step:06d}_b{bi:03d}.jpg")
                            pil_img.save(out_path)
                            print(f"[VIS] saved {out_path}")
                    except Exception:
                        pass

            if step % 100 == 0 and rank == 0:
                ckpt = os.path.join(cfg.out_dir, f"model_epoch{epoch:02d}_step{step:06d}.pt")
                # Always save the *underlying* model (so no 'module.' prefix)
                model_to_save = model.module if isinstance(
                    model, torch.nn.parallel.DistributedDataParallel
                ) else model

                torch.save(
                {
                    'epoch': epoch,
                    'model_state': model_to_save.state_dict(),
                    'optim_state': optim.state_dict(),
                    'scaler_state': (scaler.state_dict() if scaler.is_enabled() else None),
                    'config': cfg.__dict__,
                    'step': step,
                },
                ckpt,
                )
                try:
                    from glob import glob
                    pts = sorted(glob(os.path.join(cfg.out_dir, 'model_epoch*.pt')), key=lambda p: os.path.getmtime(p), reverse=True)
                    if cfg.max_ckpts is not None and cfg.max_ckpts > 0 and len(pts) > cfg.max_ckpts:
                        for old in pts[cfg.max_ckpts:]:
                            try:
                                os.remove(old)
                            except Exception:
                                pass
                except Exception:
                    pass
                if torch.cuda.is_available():
                    try:
                        torch.cuda.reset_peak_memory_stats(device)
                    except Exception:
                        pass

            # Periodic evaluation on a small held-out set (rank 0 only)
            if (cfg.eval_every and (step % cfg.eval_every == 0) and rank == 0 and cfg.val_json):
                try:
                    model_eval = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
                    model_eval.eval()
                    with torch.inference_mode():
                        vds = JsonTrackingDataset(DataConfig(train_json=cfg.val_json, n_waypoints=cfg.n_waypoints, history=cfg.history, cache_root=cfg.cache_root))
                        vdl = DataLoader(vds, batch_size=cfg.batch_size, shuffle=False, num_workers=min(2, cfg.num_workers), pin_memory=True, collate_fn=collate_batch)
                        total_mse = 0.0
                        total_count = 0
                        final_errors: List[float] = []
                        hits = 0
                        max_batches = max(1, cfg.eval_batches)
                        bdone = 0
                        for vbatch in vdl:
                            coarse_tokens = vbatch['coarse_tokens'].to(device)
                            coarse_tidx   = vbatch['coarse_tidx'].to(device)
                            fine_tokens   = vbatch['fine_tokens'].to(device)
                            fine_tidx     = vbatch['fine_tidx'].to(device)
                            yaw_hist      = vbatch['yaw_hist'].to(device)
                            yaw_curr      = vbatch['yaw_curr'].to(device)
                            gt_wp         = vbatch['waypoints'].to(device)
                            valid_mask    = vbatch['valid_mask'].to(device)
                            instr         = vbatch['instruction']

                            pred = model_eval(
                                coarse_tokens, coarse_tidx,
                                fine_tokens, fine_tidx,
                                instr,
                                yaw_hist=yaw_hist if cfg.use_angle_tvi else None,
                                yaw_curr=yaw_curr if cfg.use_angle_tvi else None
                            )
                            # masked MSE in normalized space (Option A): divide XY by alpha
                            model_inspect = model_eval
                            alpha_vec = getattr(model_inspect, 'alpha_task', None)
                            if alpha_vec is not None and pred.size(-1) >= 2 and alpha_vec.size(-1) >= 2:
                                ax = alpha_vec[..., 0:2].clamp_min(1e-6)
                                pred_n = pred.clone()
                                gt_n = gt_wp.clone()
                                pred_n[..., 0:2] = pred_n[..., 0:2] / ax
                                gt_n[..., 0:2] = gt_n[..., 0:2] / ax
                                mse = mse_masked(pred_n, gt_n, valid_mask).item()
                            else:
                                mse = mse_masked(pred, gt_wp, valid_mask).item()
                            total_mse += mse * pred.size(0)
                            total_count += pred.size(0)
                            # final waypoint EPE and hit rate (first two dims assumed spatial)
                            pred_xy = pred[:, -1, :2].float()
                            gt_xy = gt_wp[:, -1, :2].float()
                            epe = torch.linalg.norm(pred_xy - gt_xy, dim=-1)  # (B,)
                            final_errors.extend(epe.cpu().tolist())
                            hits += (epe <= cfg.final_wp_threshold).sum().item()

                            bdone += 1
                            if bdone >= max_batches:
                                break
                    mean_mse = total_mse / max(1, total_count)
                    if len(final_errors) > 0:
                        import numpy as _np
                        epe_mean = float(_np.mean(final_errors))
                        epe_median = float(_np.median(final_errors))
                        hit_rate = float(hits / len(final_errors))
                    else:
                        epe_mean = epe_median = hit_rate = float('nan')
                    print(f"[VAL] step {step} | masked_MSE={mean_mse:.5f} | final_EPE_mean={epe_mean:.4f} | final_EPE_median={epe_median:.4f} | hit@{cfg.final_wp_threshold}={hit_rate:.3f}", flush=True)
                except Exception as _e:
                    print(f"[VAL] evaluation skipped due to error: {_e}")
                finally:
                    model.train()

            # Single-episode evaluation (rank 0 only)
            if (cfg.episode_eval_every and (step % cfg.episode_eval_every == 0) and rank == 0 and cfg.episode_json):
                try:
                    model_eval = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
                    model_eval.eval()
                    with torch.inference_mode():
                        ds_tmp = JsonTrackingDataset(DataConfig(train_json=cfg.episode_json, n_waypoints=cfg.n_waypoints, history=cfg.history, cache_root=cfg.cache_root))
                        if len(ds_tmp) == 0:
                            raise RuntimeError('episode_json produced no examples')
                        max_frames = min(cfg.episode_max_frames, len(ds_tmp))
                        epe_list: List[float] = []
                        hits = 0
                        for i in range(max_frames):
                            item = ds_tmp[i]
                            coarse_tokens = item['coarse_tokens'].unsqueeze(0).to(device)
                            coarse_tidx   = item['coarse_tidx'].unsqueeze(0).to(device)
                            fine_tokens   = item['fine_tokens'].unsqueeze(0).to(device)
                            fine_tidx     = item['fine_tidx'].unsqueeze(0).to(device)
                            yaw_hist      = item['yaw_hist'].unsqueeze(0).to(device)
                            yaw_curr      = item['yaw_curr'].unsqueeze(0).to(device)
                            gt_wp         = item['waypoints'].unsqueeze(0).to(device)
                            instr         = [item['instruction']]

                            pred = model_eval(
                                coarse_tokens, coarse_tidx,
                                fine_tokens, fine_tidx,
                                instr,
                                yaw_hist=yaw_hist if cfg.use_angle_tvi else None,
                                yaw_curr=yaw_curr if cfg.use_angle_tvi else None
                            )
                            pred_xy = pred[:, -1, :2].float()
                            gt_xy = gt_wp[:, -1, :2].float()
                            epe = torch.linalg.norm(pred_xy - gt_xy, dim=-1)  # (1,)
                            e = float(epe.item())
                            epe_list.append(e)
                            if e <= cfg.episode_threshold:
                                hits += 1
                        if len(epe_list) > 0:
                            import numpy as _np
                            epe_mean = float(_np.mean(epe_list))
                            epe_median = float(_np.median(epe_list))
                            follow_rate = float(hits / len(epe_list))
                        else:
                            epe_mean = epe_median = follow_rate = float('nan')
                    print(f"[EPISODE] step {step} | frames={len(epe_list)} | EPE_mean={epe_mean:.4f} | EPE_median={epe_median:.4f} | follow@{cfg.episode_threshold}={follow_rate:.3f}", flush=True)
                except Exception as _e:
                    print(f"[EPISODE] evaluation skipped due to error: {_e}")
                finally:
                    model.train()

        if pbar is not None:
            pbar.close()

    # Optional inference after training when requested (rank 0 only)
    if rank == 0 and cfg.infer_json:
        try:
            _run_inference(cfg)
        except Exception as _e:
            print(f"[INFER] failed: {_e}")

    if use_ddp:
        dist.barrier()
        dist.destroy_process_group()

    if rank == 0:
        print(f"[TRAIN] Finished all epochs. last_step={step}")


# ----------------------- 单 Agent离线推理 -----------------------

@torch.inference_mode()
def _run_inference(cfg: TrainConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Resolve checkpoint
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

    # Load checkpoint config
    obj = torch.load(ckpt_path, map_location=device)
    ck = obj if isinstance(obj, dict) else {}
    ck_cfg = ck.get('config', {})

    # Build model config mirroring training
    n_waypoints = int(ck_cfg.get('n_waypoints', cfg.n_waypoints))
    use_angle_tvi = bool(ck_cfg.get('use_angle_tvi', cfg.use_angle_tvi))
    no_tanh_actions = bool(ck_cfg.get('no_tanh_actions', cfg.no_tanh_actions))
    vision_feat_dim = int(ck_cfg.get('vision_feat_dim', cfg.vision_feat_dim))
    alpha_xy        = ck_cfg.get('alpha_xy', getattr(cfg, 'alpha_xy', None))
    llm_name        = str(ck_cfg.get('llm_name', getattr(cfg, 'llm_name', "Qwen/Qwen3-8B")))
    print (llm_name)

    model = OpenTrackVLA(
        ModelConfig(
            llm_name=llm_name,
            n_waypoints=n_waypoints,
            beta_nav=float(ck_cfg.get('beta_nav', cfg.beta_nav)),
            use_angle_tvi=use_angle_tvi,
            use_tanh_actions=(not no_tanh_actions),
            alpha_xy=alpha_xy,
        ),
        vision_feat_dim=vision_feat_dim,
    ).to(device).eval()
    msd = ck.get('model_state', None)
    if msd:
        model.load_state_dict(msd, strict=False)

    # Data
    if cfg.infer_json is None:
        raise ValueError('--infer_json is required for inference')
    vds = JsonTrackingDataset(DataConfig(train_json=cfg.infer_json, n_waypoints=n_waypoints, history=cfg.history, cache_root=cfg.cache_root))
    vdl = DataLoader(vds, batch_size=cfg.batch_size, shuffle=False, num_workers=min(2, cfg.num_workers), pin_memory=True, collate_fn=collate_batch)

    # Output dirs
    os.makedirs(cfg.infer_out, exist_ok=True)
    vis_dir = os.path.join(cfg.infer_out, 'vis')
    npz_dir = os.path.join(cfg.infer_out, 'npz')
    if cfg.infer_vis:
        os.makedirs(vis_dir, exist_ok=True)
    if cfg.infer_save_npz:
        os.makedirs(npz_dir, exist_ok=True)

    batches_limit = max(0, int(cfg.infer_batches))
    bdone = 0
    for bidx, batch in enumerate(vdl):
        coarse_tokens = batch['coarse_tokens'].to(device)
        coarse_tidx   = batch['coarse_tidx'].to(device)
        fine_tokens   = batch['fine_tokens'].to(device)
        fine_tidx     = batch['fine_tidx'].to(device)
        yaw_hist      = batch['yaw_hist'].to(device)
        yaw_curr      = batch['yaw_curr'].to(device)
        instr         = batch['instruction']

        pred = model(
            coarse_tokens, coarse_tidx,
            fine_tokens, fine_tidx,
            instr,
            yaw_hist=yaw_hist if use_angle_tvi else None,
            yaw_curr=yaw_curr if use_angle_tvi else None
        )  # absolute space (alpha applied inside model)

        # Save NPZ
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

        # Visualization in absolute space (consistent with training vis and deployment)
        if cfg.infer_vis:
            try:
                with torch.no_grad():
                    # Ensure predictions are in absolute units for visualization
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


# ----------------------- 双 Agent路径与标签工具 -----------------------

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


def multi_agent_token_paths_for_frame(base_root: Path, cache_root: Path, frame_path: Path) -> Tuple[Path, Path]:
    """把 frame 路径映射到对应视觉 token cache。

    输入：
    data_root/frames/.../frame_00001.jpg

    输出：
    cache_root/frames/.../frame_00001_vcoarse.pt
    cache_root/frames/.../frame_00001_vfine.pt
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
    xy_per_agent = (point_error[..., :2].mean(dim=-1) * valid).sum(dim=-1) / valid_count
    yaw_per_agent = (point_error[..., 2] * valid).sum(dim=-1) / valid_count

    indices = torch.arange(mask.size(-1), device=mask.device).view(1, 1, -1)
    last_index = torch.where(mask, indices, torch.zeros_like(indices)).amax(dim=-1)
    gather_index = last_index[..., None, None].expand(-1, -1, 1, point_error.size(-1))
    final_error = point_error.gather(dim=2, index=gather_index).squeeze(2)
    final_target = target.gather(dim=2, index=gather_index).squeeze(2)
    component_weight_sum = max(2.0 + float(yaw_weight), 1e-6)
    final_per_agent = (
        2.0 * final_error[..., :2].mean(dim=-1)
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
        2.0 * xy_per_agent + float(yaw_weight) * yaw_per_agent
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
    coarse_cache_size: int = 0


class MultiAgentJsonDataset(Dataset):
    """双 Agent JSONL Dataset。

    每条样本的数据流：
    1. 从 JSONL 读取 agent1/agent2 的历史帧、当前帧、bbox、waypoints。
    2. 历史帧读取 *_vcoarse.pt，并整理成 (2, history*4, C)。
    3. 当前帧读取 *_vfine.pt，并整理成 (2, 64, C)。
    4. bbox/waypoints/valid_mask stack 成双 Agent 维度。

    __getitem__ 输出的关键 shape：
    - coarse_tokens: (2, history*4, C)
    - coarse_tidx:   (2, history*4)
    - fine_tokens:   (2, 64, C)
    - fine_tidx:     (2, 64)
    - bbox_feat:     (2, 4)
    - relative_pose: (2, 5)
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
        self._index: Optional[List[Tuple[str, int]]] = None
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
        """按 byte offset 懒读取 JSONL 中的一条样本。"""
        assert self._index is not None
        fp, offset = self._index[idx]
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
                    img_path = resolve_multi_agent_path(self.base_root, img_rel)
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

        return (
            torch.cat(coarse_list, dim=0),
            torch.cat(tidx_list, dim=0),
            fine_tokens,
            fine_tidx,
            str(current_path),
        )

    def _load_targets(self, ex: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """读取两个 Agent 的 waypoint 和 valid_mask 标签。"""
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
            wp, vm = fit_multi_agent_waypoints_and_mask(wp_i, vm_i, self.cfg.n_waypoints, self.cfg.action_dims)
            out_wp.append(wp)
            out_mask.append(vm)
        return torch.stack(out_wp, dim=0), torch.stack(out_mask, dim=0)

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
        bbox_feat = self._load_bbox(ex)
        visible = self._load_visible(ex)
        relative_pose, relative_pose_valid = self._load_relative_pose(ex)
        history = int(self.cfg.history)

        return {
            "coarse_tokens": torch.stack([a1[0], a2[0]], dim=0),
            "coarse_tidx": torch.stack([a1[1], a2[1]], dim=0),
            "fine_tokens": torch.stack([a1[2], a2[2]], dim=0),
            "fine_tidx": torch.stack([a1[3], a2[3]], dim=0),
            "yaw_hist": torch.zeros(2, history, dtype=torch.float32),
            "yaw_curr": torch.zeros(2, 1, dtype=torch.float32),
            "bbox_feat": bbox_feat,
            "visible": visible,
            "relative_pose": relative_pose,
            "relative_pose_valid": relative_pose_valid,
            "waypoints": waypoints,
            "valid_mask": valid_mask,
            "dt": float(ex.get("dt", 0.1)),
            "instruction": ex.get("instruction", "Follow the target person without collision."),
            "episode_id": ex.get("episode_id", ""),
            "step_index": int(ex.get("step_index", idx)),
            "current_path": [a1[4], a2[4]],
        }


# ----------------------- 双 Agent batch 与数据检查 -----------------------

def collate_multi_agent_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """把双 Agent 单样本合成 batch。

    张量字段 stack；instruction/current_path/episode_id 保留 Python list 给 tokenizer 和日志使用。
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
        "relative_pose": torch.stack([b["relative_pose"] for b in batch], dim=0),
        "relative_pose_valid": torch.stack([b["relative_pose_valid"] for b in batch], dim=0),
        "waypoints": torch.stack([b["waypoints"] for b in batch], dim=0),
        "valid_mask": torch.stack([b["valid_mask"] for b in batch], dim=0),
        "dt": torch.tensor([b["dt"] for b in batch], dtype=torch.float32),
        "instruction": [b["instruction"] for b in batch],
        "episode_id": [b["episode_id"] for b in batch],
        "step_index": torch.tensor([b["step_index"] for b in batch], dtype=torch.long),
        "current_path": [b["current_path"] for b in batch],
    }


def multi_agent_dataset_sanity_report(ds: MultiAgentJsonDataset, cfg: "MultiAgentTrainConfig", max_items: int = 256) -> None:
    """训练前检查双 Agent 数据分布和 cache 覆盖率。"""
    try:
        n = min(max_items, len(ds))
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
        bbox_mean = float(np.mean(np.concatenate(bbox_vals))) if bbox_vals else float("nan")
        print(
            f"[SANITY] samples_checked={n} xy_std=({x_std:.4f},{y_std:.4f}) theta_std={th_std:.4f} "
            f"bbox_mean={bbox_mean:.4f} relpose_valid={relpose_valid}/{2*n} "
            f"current_img_ok={img_ok}/{2*n} current_cache_ok={cache_ok}/{2*n}",
            flush=True,
        )
    except Exception as exc:
        print(f"[SANITY] skipped due to error: {exc}", flush=True)


@dataclass
# ----------------------- 双 Agent训练配置与模型构建 -----------------------

class MultiAgentTrainConfig:
    """双 Agent 训练配置。

    默认路径使用 model.py::MultiAgentOpenTrackVLA。设置 base_model=True 时，
    会构建 model.py-base 对照：保留双视觉流、TVI/Agent/View/Kind 编码、
    双 ACT 和双 planner，只关闭 grounding、bbox token、visibility 和 relative pose
    这些辅助分支。设置 separate_agent_context=True 时，会构建另一个 base
    对照：两个 Agent 分别送入 LLM，不共享视觉上下文。
    """

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
    distributed: bool = False
    dist_backend: str = "nccl"
    ddp_timeout_minutes: int = 120
    ddp_find_unused_parameters: bool = True
    base_model: bool = False
    separate_agent_context: bool = False
    use_agent_text_markers: bool = True
    instruction_override: Optional[str] = None
    joint_instruction_override: Optional[str] = None
    agent1_instruction_override: Optional[str] = None
    agent2_instruction_override: Optional[str] = None
    freeze_llm: bool = True
    use_angle_tvi: bool = False
    insert_time_tokens: bool = True
    no_tanh_actions: bool = True
    alpha_xy: Optional[float] = 1.0
    use_grounding: bool = True
    use_bbox_tokens: bool = True
    beta_nav: float = 100.0
    drone_loss_weight: float = 2.0
    dog_loss_weight: float = 1.0
    normalize_agent_loss_weights: bool = False
    nav_loss_type: str = "mse"
    smooth_l1_beta: float = 0.05
    yaw_loss_weight: float = 1.0
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
    max_ckpts: int = 3
    eval_every: int = 0
    eval_batches: int = 8
    val_bbox_source: str = "none"
    final_wp_threshold: float = 0.2
    resume: bool = False
    resume_ckpt: Optional[str] = None
    dry_run: bool = False


def apply_multi_agent_base_defaults(cfg: MultiAgentTrainConfig) -> MultiAgentTrainConfig:
    """把 model.py multi-agent 路径收敛成 waypoint-only base 对照。"""
    if cfg.lr_scheduler not in {"constant", "cosine"}:
        raise ValueError(f"Unsupported lr_scheduler={cfg.lr_scheduler!r}")
    if cfg.lr <= 0.0:
        raise ValueError("--lr must be positive.")
    if cfg.min_lr < 0.0 or cfg.min_lr > cfg.lr:
        raise ValueError("--min_lr must satisfy 0 <= min_lr <= lr.")
    if cfg.warmup_steps < 0:
        raise ValueError("--warmup_steps must be non-negative.")
    if cfg.normalize_agent_loss_weights and (
        float(cfg.drone_loss_weight) + float(cfg.dog_loss_weight) <= 0.0
    ):
        raise ValueError("Normalized agent loss requires a positive sum of agent weights.")
    if cfg.separate_agent_context:
        cfg.base_model = True
        if not cfg.use_agent_text_markers:
            raise ValueError("--no-agent-text-markers is only supported by the shared-context base model.")
    if not cfg.base_model:
        return cfg
    cfg.use_grounding = False
    cfg.use_bbox_tokens = False
    cfg.beta_bbox = 0.0
    cfg.beta_visible = 0.0
    cfg.beta_relative_pose = 0.0
    cfg.bbox_dropout_prob = 0.0
    cfg.return_token_logits = False
    cfg.ddp_find_unused_parameters = False
    cfg.val_bbox_source = "none"
    if cfg.joint_instruction_override is None:
        cfg.joint_instruction_override = cfg.instruction_override
    if cfg.joint_instruction_override is None:
        cfg.joint_instruction_override = MULTI_AGENT_COOP_INSTRUCTION
    if not cfg.use_agent_text_markers and (
        cfg.agent1_instruction_override is not None or cfg.agent2_instruction_override is not None
    ):
        raise ValueError(
            "The no-marker base accepts only --joint_instruction_override; "
            "remove the per-agent instruction overrides."
        )
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
            coarse_cache_size=cfg.coarse_cache_size,
        )
    )


def build_multi_agent_model(cfg: MultiAgentTrainConfig) -> nn.Module:
    """创建主目录 model.py 中的双 Agent 模型。

    cfg.base_model=True 时仍使用同一个 MultiAgentOpenTrackVLA 主干，
    但关闭 grounding/bbox/visible 等辅助分支，只保留 waypoint base。
    cfg.separate_agent_context=True 时改用 MultiAgentSeparateOpenTrackVLA，
    两个 Agent 分别送入同一个 LLM，不把 visual pieces 拼到同一上下文。
    """
    cfg = apply_multi_agent_base_defaults(cfg)
    from model import MultiAgentModelConfig, MultiAgentOpenTrackVLA, MultiAgentSeparateOpenTrackVLA

    model_cfg = MultiAgentModelConfig(
        llm_name=cfg.llm_name,
        freeze_llm=cfg.freeze_llm,
        n_waypoints=cfg.n_waypoints,
        action_dims=cfg.action_dims,
        use_angle_tvi=cfg.use_angle_tvi,
        insert_time_tokens=cfg.insert_time_tokens,
        use_tanh_actions=(not cfg.no_tanh_actions),
        alpha_xy=cfg.alpha_xy,
        use_grounding=cfg.use_grounding,
        use_bbox_tokens=cfg.use_bbox_tokens,
        return_token_logits=cfg.return_token_logits,
        use_agent_text_markers=cfg.use_agent_text_markers,
    )
    model_cls = MultiAgentSeparateOpenTrackVLA if cfg.separate_agent_context else MultiAgentOpenTrackVLA
    return model_cls(model_cfg, vision_feat_dim=cfg.vision_feat_dim)


# ----------------------- 双 Agent损失、学习率与 checkpoint -----------------------

def forward_multi_agent_loss(
    model: nn.Module,
    batch: Dict[str, Any],
    cfg: MultiAgentTrainConfig,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """双 Agent 一次 forward 和 loss 计算。

    model.py-base 对照只使用：
        loss = beta_nav * (drone_loss_weight * drone_waypoint_loss
                           + dog_loss_weight * dog_waypoint_loss)

    shared-context base 和 separate-context base 都走同一个分 Agent loss。
    非 base 模式仍可额外打开 bbox/visibility/relative_pose 辅助监督。
    """
    bbox_target = batch["bbox_feat"].to(device)
    bbox_input = bbox_target
    if model.training and cfg.bbox_dropout_prob > 0.0 and torch.rand((), device=device).item() < cfg.bbox_dropout_prob:
        # Entire batches alternate between bbox refinement and prior-free
        # absolute detection because the model uses a single grounding mode.
        bbox_input = None

    instructions = batch["instruction"]
    joint_instructions = instructions
    if cfg.joint_instruction_override:
        joint_instructions = [cfg.joint_instruction_override] * len(instructions)
    elif cfg.instruction_override:
        joint_instructions = [cfg.instruction_override] * len(instructions)
    agent1_instructions = (
        [cfg.agent1_instruction_override] * len(instructions)
        if cfg.agent1_instruction_override
        else None
    )
    agent2_instructions = (
        [cfg.agent2_instruction_override] * len(instructions)
        if cfg.agent2_instruction_override
        else None
    )

    out = model(
        coarse_tokens=batch["coarse_tokens"].to(device),
        coarse_tidx=batch["coarse_tidx"].to(device),
        fine_tokens=batch["fine_tokens"].to(device),
        fine_tidx=batch["fine_tidx"].to(device),
        instructions=instructions,
        joint_instructions=joint_instructions,
        agent1_instructions=agent1_instructions,
        agent2_instructions=agent2_instructions,
        bbox_feat=bbox_input,
        yaw_hist=batch["yaw_hist"].to(device) if cfg.use_angle_tvi else None,
        yaw_curr=batch["yaw_curr"].to(device) if cfg.use_angle_tvi else None,
    )

    pred = out["waypoints"]
    gt = batch["waypoints"].to(device)
    mask = batch["valid_mask"].to(device)
    model_inspect = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    pred_n, gt_n = normalize_multi_agent_xy_by_alpha(pred, gt, getattr(model_inspect, "alpha_task", None))
    per_agent_loss, waypoint_metrics = weighted_multi_agent_waypoint_loss(
        pred_n,
        gt_n,
        mask,
        batch["dt"].to(device),
        loss_type=cfg.nav_loss_type,
        smooth_l1_beta=cfg.smooth_l1_beta,
        yaw_weight=cfg.yaw_loss_weight,
        final_weight=cfg.final_waypoint_loss_weight,
        turn_sample_weight=cfg.turn_sample_weight,
        turn_rate_threshold=cfg.turn_rate_threshold,
        turn_angle_threshold=cfg.turn_angle_threshold,
        stop_sample_weight=cfg.stop_sample_weight,
        stop_speed_threshold=cfg.stop_speed_threshold,
        stop_window=cfg.stop_window,
    )
    # Convention in the processed dataset: agent index 0 is drone, index 1 is robotdog.
    loss_nav_drone = per_agent_loss[:, 0].mean()
    loss_nav_dog = per_agent_loss[:, 1].mean()
    agent_weight_sum = float(cfg.drone_loss_weight) + float(cfg.dog_loss_weight)
    agent_loss_divisor = agent_weight_sum if cfg.normalize_agent_loss_weights else 1.0
    loss_nav = (
        float(cfg.drone_loss_weight) * loss_nav_drone
        + float(cfg.dog_loss_weight) * loss_nav_dog
    ) / agent_loss_divisor
    xy_per_agent = waypoint_metrics["xy_per_agent"]
    yaw_per_agent = waypoint_metrics["yaw_per_agent"]
    final_per_agent = waypoint_metrics["final_per_agent"]
    loss_nav_xy = (
        float(cfg.drone_loss_weight) * xy_per_agent[:, 0].mean()
        + float(cfg.dog_loss_weight) * xy_per_agent[:, 1].mean()
    ) / agent_loss_divisor
    loss_nav_yaw = (
        float(cfg.drone_loss_weight) * yaw_per_agent[:, 0].mean()
        + float(cfg.dog_loss_weight) * yaw_per_agent[:, 1].mean()
    ) / agent_loss_divisor
    loss_nav_final = (
        float(cfg.drone_loss_weight) * final_per_agent[:, 0].mean()
        + float(cfg.dog_loss_weight) * final_per_agent[:, 1].mean()
    ) / agent_loss_divisor

    refined_bbox = out.get("refined_bbox")
    if cfg.beta_bbox != 0.0 and refined_bbox is not None:
        loss_bbox = F.mse_loss(refined_bbox.float(), bbox_target.float())
    else:
        loss_bbox = loss_nav.new_tensor(0.0)

    visible_logits = out.get("visible_logits")
    if cfg.beta_visible != 0.0 and visible_logits is not None:
        visible_target = batch["visible"].to(device)
        loss_visible = F.binary_cross_entropy_with_logits(visible_logits.float(), visible_target.float())
    else:
        loss_visible = loss_nav.new_tensor(0.0)

    pred_relative_pose = out.get("relative_pose")
    if cfg.beta_relative_pose != 0.0 and pred_relative_pose is not None:
        relative_target = batch["relative_pose"].to(device).float()
        relative_valid = batch["relative_pose_valid"].to(device).bool()
        pose_per_agent = F.smooth_l1_loss(
            pred_relative_pose.float(),
            relative_target,
            reduction="none",
        ).mean(dim=-1)
        if relative_valid.any():
            loss_relative_pose = pose_per_agent[relative_valid].mean()
        else:
            loss_relative_pose = loss_nav.new_tensor(0.0)
    else:
        loss_relative_pose = loss_nav.new_tensor(0.0)

    loss = (
        cfg.beta_nav * loss_nav
        + cfg.beta_bbox * loss_bbox
        + cfg.beta_visible * loss_visible
        + cfg.beta_relative_pose * loss_relative_pose
    )
    return loss, {
        "loss_nav": loss_nav.detach(),
        "loss_nav_drone": loss_nav_drone.detach(),
        "loss_nav_dog": loss_nav_dog.detach(),
        "loss_nav_xy": loss_nav_xy.detach(),
        "loss_nav_yaw": loss_nav_yaw.detach(),
        "loss_nav_final": loss_nav_final.detach(),
        "turn_fraction": waypoint_metrics["turn_mask"].float().mean().detach(),
        "stop_fraction": waypoint_metrics["stop_mask"].float().mean().detach(),
        "turn_fraction_drone": waypoint_metrics["turn_mask"][:, 0].float().mean().detach(),
        "turn_fraction_dog": waypoint_metrics["turn_mask"][:, 1].float().mean().detach(),
        "stop_fraction_drone": waypoint_metrics["stop_mask"][:, 0].float().mean().detach(),
        "stop_fraction_dog": waypoint_metrics["stop_mask"][:, 1].float().mean().detach(),
        "behavior_weight_mean": waypoint_metrics["behavior_weight"].mean().detach(),
        "loss_bbox": loss_bbox.detach(),
        "loss_visible": loss_visible.detach(),
        "loss_relative_pose": loss_relative_pose.detach(),
        "pred": pred.detach(),
    }


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
    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "model_state": model_to_save.state_dict(),
            "optim_state": optim.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state": scaler.state_dict() if scaler.is_enabled() else None,
            "config": {
                **cfg.__dict__,
                "model_type": multi_agent_model_type_name(cfg),
            },
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
        "effective_batch_per_rank": int(cfg.batch_size) * max(1, int(cfg.grad_accum_steps)),
    }
    (out_dir / "train_config.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    lines = [
        "model.py multi-agent training config",
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
    "final_epe",
    "final_epe_drone",
    "final_epe_dog",
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
    """小规模验证双 Agent 规划效果。"""
    model_eval = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    model_eval.eval()
    # 验证时关闭扩散随机噪声，使不同 step/epoch 的 EPE 可以直接比较。
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
    final_errors: List[float] = []
    batches = 0
    for batch in dl:
        loss, metrics = forward_multi_agent_loss(model_eval, batch, cfg, device)
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
    for planner, previous in zip(diffusion_planners, previous_deterministic):
        planner.deterministic_inference = previous
    return {"loss": total_loss / max(1, total_count), "final_epe": epe_mean, "hit": hit}


# ----------------------- 双 Agent训练主循环 -----------------------

def train_multi_agent(cfg: MultiAgentTrainConfig) -> None:
    """双 Agent 训练主函数。

    入口数据流：
    JSONL/cache -> MultiAgentJsonDataset -> DataLoader
    -> MultiAgentOpenTrackVLA -> 双 Agent waypoint loss
    -> 可选 bbox/visibility/relative pose 辅助 loss
    -> checkpoint/train_log.csv。
    """
    cfg = apply_multi_agent_base_defaults(cfg)
    use_ddp = bool(cfg.distributed)
    if use_ddp:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(
            backend=cfg.dist_backend,
            init_method="env://",
            device_id=device,
            timeout=timedelta(minutes=max(1, int(cfg.ddp_timeout_minutes))),
        )
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
        # A100 对 TF32 有硬件加速；BF16 之外残留的 FP32 matmul 也可获得更高吞吐。
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    if rank == 0:
        if cfg.separate_agent_context:
            mode = "model.py-separate-base"
        else:
            mode = "model.py-base" if cfg.base_model else "model.py-full"
        print(f"[INIT][MULTI] mode={mode} train_json={cfg.train_json} out_dir={cfg.out_dir}", flush=True)
    if cfg.val_json and not cfg.val_cache_root:
        raise ValueError(
            "--val_json requires --val_cache_root so validation cannot accidentally read training vision tokens."
        )
    ds = build_multi_agent_dataset(cfg.train_json, cfg)
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

    sampler = torch.utils.data.distributed.DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True) if use_ddp else None
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
            f"updates_per_epoch={updates_per_epoch} total_optimizer_steps={total_optimizer_steps}",
            flush=True,
        )
    else:
        updates_per_epoch = math.ceil(len(dl) / cfg.grad_accum_steps)
        total_optimizer_steps = updates_per_epoch * cfg.epochs

    print(f"[INIT][MULTI][rank {rank}] building model", flush=True)
    model = build_multi_agent_model(cfg)
    print(f"[INIT][MULTI][rank {rank}] moving model to {device}", flush=True)
    model = model.to(device)
    print(f"[INIT][MULTI][rank {rank}] model moved to {device}", flush=True)
    if use_ddp:
        # 冻结的 Qwen 权重已经由每个 rank 从同一 checkpoint 加载，不需要 DDP
        # 再广播一次。排除冻结参数可避免初始化阶段同步整个 LLM 主干。
        ignored = {name for name, parameter in model.named_parameters() if not parameter.requires_grad}
        ignored.update(name for name, _ in model.named_buffers() if name.startswith("llm."))
        model._ddp_params_and_buffers_to_ignore = ignored
        ignored_numel = sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if name in ignored
        )
        print(
            f"[INIT][MULTI][rank {rank}] wrapping DDP; ignored_frozen={len(ignored)} "
            f"ignored_numel={ignored_numel:,}",
            flush=True,
        )
        ddp_kwargs: Dict[str, Any] = {
            "device_ids": [local_rank],
            "output_device": local_rank,
            "find_unused_parameters": bool(cfg.ddp_find_unused_parameters),
            "broadcast_buffers": False,
        }
        ddp_start = time.time()
        model = torch.nn.parallel.DistributedDataParallel(model, **ddp_kwargs)
        print(f"[INIT][MULTI][rank {rank}] DDP ready in {time.time() - ddp_start:.1f}s", flush=True)

    if rank == 0:
        model_inspect = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        total = sum(p.numel() for p in model_inspect.parameters())
        trainable = sum(p.numel() for p in model_inspect.parameters() if p.requires_grad)
        print(f"[PARAMS][MULTI] total={total:,} trainable={trainable:,} ({100.0 * trainable / max(1, total):.2f}%)", flush=True)

    optim = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=cfg.lr, weight_decay=cfg.weight_decay)
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
            ckpt_cfg = obj.get("config", {}) if isinstance(obj, dict) else {}
            if getattr(cfg, "use_anchor_diffusion", False):
                expected_metadata = {
                    "grounding_architecture": "dual_agent_gnd_v2",
                    "multimodal_sequence_layout": "dual_visual_before_queries_v2",
                    "action_scale_version": "anchor_maxabs_v2",
                }
                incompatible = {
                    key: ckpt_cfg.get(key)
                    for key, expected in expected_metadata.items()
                    if ckpt_cfg.get(key) != expected
                }
                if incompatible:
                    raise RuntimeError(
                        "Refusing to resume an incompatible Anchor Diffusion checkpoint. "
                        f"Expected {expected_metadata}, found {incompatible}. Start a fresh output directory."
                    )
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
    best_val_epe = float("inf")
    best_val_metrics_path = out_dir / "best_val_metrics.json"
    if rank == 0 and best_val_metrics_path.exists():
        try:
            best_val_epe = float(json.loads(best_val_metrics_path.read_text(encoding="utf-8"))["final_epe"])
            print(f"[VAL][MULTI] existing best final_epe={best_val_epe:.5f}", flush=True)
        except Exception as exc:
            print(f"[VAL][MULTI][WARN] failed to read {best_val_metrics_path}: {exc}", flush=True)

    val_ds = (
        build_multi_agent_dataset(cfg.val_json, cfg, cache_root=cfg.val_cache_root)
        if (cfg.val_json and rank == 0)
        else None
    )
    ema_loss: Optional[float] = None
    last_log = time.time()

    for epoch in range(start_epoch, cfg.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        epoch_start = time.time()
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
                pbar.update(1)
            if not do_step:
                continue

            grad_norm = 0.0
            if cfg.grad_clip is not None and cfg.grad_clip > 0:
                scaler.unscale_(optim)
                grad_norm = _compute_total_grad_norm(model.parameters())
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
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
                eta = _format_duration((time.time() - epoch_start) / max(1, batch_idx) * max(0, len(dl) - batch_idx))
                msg = (
                    f"[TRAIN][MULTI] epoch={epoch + 1}/{cfg.epochs} batch={batch_idx}/{len(dl)} step={step} eta={eta} "
                    f"lr={lr_used:.8g} "
                    f"loss={loss_val:.5f} ema={ema_loss:.5f} nav={metrics['loss_nav'].item():.5f} "
                    f"nav_drone={metrics['loss_nav_drone'].item():.5f} nav_dog={metrics['loss_nav_dog'].item():.5f} "
                    f"xy={metrics['loss_nav_xy'].item():.5f} yaw={metrics['loss_nav_yaw'].item():.5f} "
                    f"final={metrics['loss_nav_final'].item():.5f} "
                    f"turn={metrics['turn_fraction'].item():.3f}"
                    f"(D={metrics['turn_fraction_drone'].item():.3f},G={metrics['turn_fraction_dog'].item():.3f}) "
                    f"stop={metrics['stop_fraction'].item():.3f}"
                    f"(D={metrics['stop_fraction_drone'].item():.3f},G={metrics['stop_fraction_dog'].item():.3f}) "
                    f"behavior_w={metrics['behavior_weight_mean'].item():.2f} "
                    f"reg={regression_loss:.5f} score={score_loss:.5f} "
                    f"bbox={metrics['loss_bbox'].item():.5f} vis={metrics['loss_visible'].item():.5f} "
                    f"relpose={metrics['loss_relative_pose'].item():.5f} "
                    f"final_epe={final_epe:.4f} epe_drone={final_epe_drone:.4f} epe_dog={final_epe_dog:.4f} "
                    f"grad={grad_norm:.3f} dt={elapsed:.2f}s"
                )
                if pbar is not None:
                    pbar.write(msg)
                else:
                    print(msg, flush=True)
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
                                final_epe,
                                final_epe_drone,
                                final_epe_dog,
                                grad_norm,
                            ]
                        )

            should_save = cfg.save_every > 0 and step % cfg.save_every == 0
            if should_save:
                # rank 0 写入大 checkpoint 时，其余 rank 在 barrier 等待，避免提前进入下一次 DDP backward。
                if use_ddp:
                    dist.barrier()
                if rank == 0:
                    ckpt = out_dir / f"model_epoch{epoch:02d}_step{step:06d}.pt"
                    save_multi_agent_checkpoint(ckpt, model, optim, scheduler, scaler, cfg, epoch, step)
                    prune_multi_agent_checkpoints(out_dir, cfg.max_ckpts)
                    print(f"[CKPT][MULTI] saved {ckpt}", flush=True)
                if use_ddp:
                    dist.barrier()

            should_evaluate = bool(cfg.val_json and cfg.eval_every > 0 and step % cfg.eval_every == 0)
            if should_evaluate:
                # 当前离线验证只在 rank 0 执行；显式同步防止其他 rank 卡在训练 all-reduce。
                if use_ddp:
                    dist.barrier()
                if rank == 0:
                    assert val_ds is not None
                    stats = evaluate_multi_agent(model, val_ds, cfg, device)
                    print(
                        f"[VAL][MULTI] step={step} loss={stats['loss']:.5f} final_epe={stats['final_epe']:.4f} "
                        f"hit@{cfg.final_wp_threshold}={stats['hit']:.3f}",
                        flush=True,
                    )
                    if np.isfinite(stats["final_epe"]) and stats["final_epe"] < best_val_epe:
                        best_val_epe = float(stats["final_epe"])
                        best_ckpt = out_dir / "model_best_val.pt"
                        save_multi_agent_checkpoint(best_ckpt, model, optim, scheduler, scaler, cfg, epoch, step)
                        best_val_metrics_path.write_text(
                            json.dumps(
                                {
                                    "epoch": epoch,
                                    "step": step,
                                    "final_epe": best_val_epe,
                                    "loss": float(stats["loss"]),
                                    "hit": float(stats["hit"]),
                                    "checkpoint": str(best_ckpt),
                                },
                                ensure_ascii=False,
                                indent=2,
                            ),
                            encoding="utf-8",
                        )
                        print(f"[VAL][MULTI] saved new best checkpoint {best_ckpt}", flush=True)
                if use_ddp:
                    dist.barrier()

        if pbar is not None:
            pbar.close()
        completed_epoch = epoch + 1
        should_save_epoch = (
            (cfg.save_every_epochs > 0 and completed_epoch % cfg.save_every_epochs == 0)
            or completed_epoch == cfg.epochs
        )
        if should_save_epoch:
            if use_ddp:
                dist.barrier()
            if rank == 0:
                ckpt = out_dir / f"model_epoch{completed_epoch:03d}_step{step:06d}_final.pt"
                save_multi_agent_checkpoint(ckpt, model, optim, scheduler, scaler, cfg, completed_epoch, step)
                prune_multi_agent_checkpoints(out_dir, cfg.max_ckpts)
                print(f"[CKPT][MULTI] saved epoch {completed_epoch} checkpoint {ckpt}", flush=True)
            if use_ddp:
                dist.barrier()

    if use_ddp:
        dist.barrier()
        dist.destroy_process_group()
    if rank == 0:
        print(f"[DONE][MULTI] training complete step={step}", flush=True)


# ----------------------- 命令行参数与程序入口 -----------------------

def parse_multi_agent_args() -> MultiAgentTrainConfig:
    """双 Agent 训练 CLI。

    原 `parse_args()` 仍服务单 Agent；该 parser 只在命令中包含 --multi_agent 时启用。
    加 --base_model 时训练 model.py-base 对照：无 grounding、无 bbox token、无扩散。
    加 --separate_agent_context 时训练独立上下文 base 对照：两个 Agent 分别送入 LLM。
    """
    ap = argparse.ArgumentParser(description="Train MultiAgentOpenTrackVLA on two-agent JSONL data.")
    ap.add_argument("--train_json", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="/data/hdt/ntv_data/ckpt/ckpts_multi_agent")
    ap.add_argument("--val_json", type=str, default=None)
    ap.add_argument("--val_cache_root", type=str, default=None)
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
    ap.add_argument(
        "--lr_scheduler",
        choices=("constant", "cosine"),
        default="constant",
        help="Learning-rate schedule advanced once per optimizer step.",
    )
    ap.add_argument("--warmup_steps", type=int, default=0)
    ap.add_argument("--min_lr", type=float, default=0.0)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--mixed_precision", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--prefetch_factor", type=int, default=2)
    ap.add_argument(
        "--coarse_cache_size",
        type=int,
        default=0,
        help="Per-worker LRU size for reused historical vcoarse token files; 0 disables it.",
    )
    ap.add_argument("--distributed", action="store_true")
    ap.add_argument("--dist_backend", type=str, default="nccl")
    ap.add_argument("--ddp_timeout_minutes", type=int, default=120)
    ap.add_argument(
        "--ddp-find-unused-parameters",
        dest="ddp_find_unused_parameters",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable DDP unused-parameter detection. Keep this on for multi-agent "
            "grounding/bbox-dropout training, where some auxiliary branches may be skipped."
        ),
    )
    ap.add_argument(
        "--base_model",
        action="store_true",
        help=(
            "Train the model.py-base ablation: dual visual streams + TVI/agent/view/kind tags "
            "+ two ACT planner heads, with grounding/bbox/visibility/relative-pose branches disabled."
        ),
    )
    ap.add_argument(
        "--separate_agent_context",
        action="store_true",
        help=(
            "Train the separate-context base ablation: agent1 and agent2 are forwarded through "
            "the same LLM in two independent [text, one-agent visual, ACT] contexts."
        ),
    )
    ap.add_argument(
        "--agent-text-markers",
        dest="use_agent_text_markers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Add per-agent language markers before each visual stream in shared-context base mode. "
            "Disable for [joint text, agent1 visual, agent2 visual, ACT1, ACT2]."
        ),
    )
    ap.add_argument("--freeze_llm", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--freeze_llm0", dest="freeze_llm", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--use_angle_tvi", action="store_true")
    ap.add_argument("--insert_time_tokens", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--no_tanh_actions", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--alpha_xy", type=float, default=1.0)
    ap.add_argument("--auto_alpha_xy", action="store_true", help="Set alpha_xy from training waypoint percentile.")
    ap.add_argument(
        "--use-grounding",
        dest="use_grounding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use GND tokens, grounding head, and grounding-to-action feedback.",
    )
    ap.add_argument(
        "--use-bbox-tokens",
        dest="use_bbox_tokens",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append bbox prior tokens to each agent visual stream.",
    )
    ap.add_argument("--beta_nav", type=float, default=100.0)
    ap.add_argument("--drone_loss_weight", type=float, default=2.0)
    ap.add_argument("--dog_loss_weight", type=float, default=1.0)
    ap.add_argument(
        "--normalize-agent-loss-weights",
        dest="normalize_agent_loss_weights",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Divide the weighted drone+robotdog loss by the sum of agent weights.",
    )
    ap.add_argument("--nav_loss_type", choices=("mse", "smooth_l1"), default="mse")
    ap.add_argument("--smooth_l1_beta", type=float, default=0.05)
    ap.add_argument("--yaw_loss_weight", type=float, default=1.0)
    ap.add_argument("--final_waypoint_loss_weight", type=float, default=0.0)
    ap.add_argument("--turn_sample_weight", type=float, default=1.0)
    ap.add_argument("--turn_rate_threshold", type=float, default=0.1)
    ap.add_argument("--turn_angle_threshold", type=float, default=0.08)
    ap.add_argument("--stop_sample_weight", type=float, default=1.0)
    ap.add_argument("--stop_speed_threshold", type=float, default=0.15)
    ap.add_argument("--stop_window", type=int, default=3)
    ap.add_argument(
        "--instruction_override",
        type=str,
        default=None,
        help=(
            "Override every JSONL instruction during multi-agent training. "
            "Compatibility alias for --joint_instruction_override."
        ),
    )
    ap.add_argument(
        "--joint_instruction_override",
        type=str,
        default=None,
        help="Override the shared joint task instruction during multi-agent training.",
    )
    ap.add_argument(
        "--agent1_instruction_override",
        type=str,
        default=None,
        help="Override the agent1/drone instruction during multi-agent training.",
    )
    ap.add_argument(
        "--agent2_instruction_override",
        type=str,
        default=None,
        help="Override the agent2/robotdog instruction during multi-agent training.",
    )
    ap.add_argument("--beta_bbox", type=float, default=0.0)
    ap.add_argument("--beta_visible", type=float, default=0.0)
    ap.add_argument(
        "--beta_relative_pose",
        type=float,
        default=0.0,
        help="Weight for spatial grounding loss on [dx, dy, dz, sin(d_yaw), cos(d_yaw)].",
    )
    ap.add_argument(
        "--bbox_dropout_prob",
        type=float,
        default=0.0,
        help="Probability of dropping the full GT bbox prior so the grounding head learns absolute detection.",
    )
    ap.add_argument("--return_token_logits", action="store_true")
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--csv_logging", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--save_every", type=int, default=100)
    ap.add_argument(
        "--save_every_epochs",
        type=int,
        default=1,
        help="Save a regular checkpoint every N completed epochs; 0 only saves the final epoch.",
    )
    ap.add_argument("--max_ckpts", type=int, default=3)
    ap.add_argument("--eval_every", type=int, default=0)
    ap.add_argument("--eval_batches", type=int, default=8)
    ap.add_argument(
        "--val_bbox_source",
        choices=("none", "ground_truth"),
        default="none",
        help="BBox prior used during offline validation; none tests prior-free detection/planning.",
    )
    ap.add_argument("--final_wp_threshold", type=float, default=0.2)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--resume_ckpt", type=str, default=None)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    data = vars(args)
    if data.pop("auto_alpha_xy"):
        data["alpha_xy"] = None
    return apply_multi_agent_base_defaults(MultiAgentTrainConfig(**data))

def parse_args() -> TrainConfig:
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_json', type=str, required=True)
    ap.add_argument('--out_dir', type=str, default='/data/hdt/ntv_data/ckpt/ckpts_qwen4')
    ap.add_argument('--n_waypoints', type=int, default=8)
    ap.add_argument('--history', type=int, default=31)
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--batch_size', type=int, default=2)
    ap.add_argument('--grad_accum_steps', type=int, default=1, help='Accumulate this many micro-batches before each optimizer step')
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--weight_decay', type=float, default=0.01)
    ap.add_argument('--grad_clip', type=float, default=1.0)
    ap.add_argument('--mixed_precision', action='store_true')
    ap.add_argument('--vision_feat_dim', type=int, default=1536)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--cache_root', type=str, default=None, help='Root folder containing vision_cache; defaults to <train_json>/vision_cache')
    ap.add_argument('--distributed', action='store_true')
    ap.add_argument('--dist_backend', type=str, default='nccl')
    ap.add_argument(
        '--freeze_llm',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Freeze the Qwen backbone and train only projector, TVI, ACT token, and planner.',
    )
    ap.add_argument('--use_angle_tvi', action='store_true')  # default False
    ap.add_argument('--alpha_xy', type=float, default=1.0, help='Scalar to scale only XY targets; yaw unscaled')
    ap.add_argument('--beta_nav',      type=float, default=100.0)
    # logging / saving
    ap.add_argument('--log_every', type=int, default=10)
    ap.add_argument('--csv_logging', action='store_true')
    ap.add_argument('--progress', action=argparse.BooleanOptionalAction, default=True, help='Show a tqdm progress bar on rank 0')
    ap.add_argument('--save_trajectories', action='store_true')
    ap.add_argument('--traj_subdir', type=str, default='trajectories')
    ap.add_argument('--vis_every', type=int, default=500, help='Save visualization every N steps (0 = disabled)')
    ap.add_argument('--vis_samples', type=int, default=2, help='Number of samples to visualize per batch')
    # evaluation
    ap.add_argument('--eval_every', type=int, default=0, help='Evaluate every N steps (0 = disabled)')
    ap.add_argument('--eval_batches', type=int, default=8, help='Number of validation batches per eval')
    ap.add_argument('--final_wp_threshold', type=float, default=0.2, help='Hit threshold (units of target XY)')
    # modeling
    ap.add_argument('--no_tanh_actions', action=argparse.BooleanOptionalAction, default=True, help='Remove tanh cap in action head (unbounded outputs)')
    # checkpoint retention
    ap.add_argument('--max_ckpts', type=int, default=3, help='Keep at most this many checkpoints')
    # resume
    ap.add_argument('--resume', action='store_true', help='Resume from the latest checkpoint in out_dir (or --resume_ckpt)')
    ap.add_argument('--resume_ckpt', type=str, default=None, help='Path to a specific checkpoint to resume from')
    # inference
    ap.add_argument('--infer_json', type=str, default=None, help='Run inference on this dataset (json/jsonl/dir)')
    ap.add_argument('--infer_ckpt', type=str, default=None, help='Checkpoint to load for inference (defaults to latest in out_dir)')
    ap.add_argument('--infer_out', type=str, default='./infer_out', help='Output directory for inference results')
    ap.add_argument('--infer_batches', type=int, default=0, help='Limit number of batches to run at inference (0 = all)')
    ap.add_argument('--infer_vis', action='store_true', help='Save visualization images during inference')
    ap.add_argument('--infer_save_npz', action='store_true', help='Save npz predictions during inference')
    # single-episode evaluation
    ap.add_argument('--episode_json', type=str, default=None, help='JSON/JSONL path for a single episode sequence to evaluate')
    ap.add_argument('--episode_eval_every', type=int, default=0, help='Evaluate the episode every N steps (0 = disabled)')
    ap.add_argument('--episode_threshold', type=float, default=0.2, help='Following threshold radius for episode eval')
    ap.add_argument('--episode_max_frames', type=int, default=256, help='Maximum frames to evaluate in the episode')
    ap.add_argument('--llm_name', type=str, default='Qwen/Qwen3-0.6B', help='HuggingFace model id for the LLM backbone')

    args = ap.parse_args()
    return TrainConfig(**vars(args))


if __name__ == '__main__':
    if "--multi_agent" in sys.argv:
        sys.argv.remove("--multi_agent")
        train_multi_agent(parse_multi_agent_args())
    else:
        cfg = parse_args()
        # Inference-only mode: if infer_json is provided and epochs==0, run inference with a loaded model
        if cfg.infer_json and cfg.epochs == 0:
            _run_inference(cfg)
        else:
            train(cfg)
