#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Clean multi-agent base dataset utilities.

This module intentionally keeps only the data needed by the base planner:
two agents' cached visual tokens, text instruction, and waypoint labels.
There is no bbox, grounding, visibility, relative pose, or diffusion logic here.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


def load_tokens_file(path: str | Path) -> torch.Tensor:
    """Load a cached visual-token file written by tools.precache_frames."""
    try:
        obj = torch.load(str(path), map_location="cpu")
    except Exception:
        obj = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(obj, torch.Tensor):
        return obj.float()
    if isinstance(obj, dict):
        for key in ("V", "Vfine", "Vcoarse", "tokens", "feat", "features"):
            value = obj.get(key)
            if isinstance(value, torch.Tensor):
                if value.dim() == 3 and value.size(0) == 1:
                    value = value[0]
                return value.float()
    raise ValueError(f"Unrecognized token cache format: {path}")


def find_data_root(train_path: Path) -> Path:
    """Find the data root containing frames/ from jsonl, jsonl dir, or dataset.json."""
    candidate = train_path if train_path.is_dir() else train_path.parent
    for _ in range(8):
        if (candidate / "frames").exists():
            return candidate
        if candidate.parent == candidate:
            break
        candidate = candidate.parent
    return train_path if train_path.is_dir() else train_path.parent


def resolve_frame_path(data_root: Path, rel_or_abs: str) -> Path:
    path = Path(rel_or_abs)
    return path if path.is_absolute() else data_root / path


def token_paths_for_frame(data_root: Path, cache_root: Path, frame_path: Path) -> Tuple[Path, Path]:
    """Map data_root/frames/.../x.jpg to cache_root/frames/.../x_v*.pt."""
    try:
        rel = frame_path.resolve().relative_to(data_root.resolve())
    except ValueError:
        parts = frame_path.parts
        rel = Path(*parts[parts.index("frames") :]) if "frames" in parts else Path(frame_path.name)
    token_dir = cache_root / rel.parent
    return token_dir / f"{rel.stem}_vcoarse.pt", token_dir / f"{rel.stem}_vfine.pt"


def fit_waypoints(
    waypoints: Any,
    valid_mask: Any,
    n_waypoints: int,
    action_dims: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return fixed-shape waypoint labels and valid mask for one agent."""
    wp = torch.as_tensor(waypoints, dtype=torch.float32)
    if wp.dim() == 1:
        wp = wp.view(1, -1)
    if wp.dim() != 2:
        raise ValueError(f"Expected waypoint shape (M,D), got {tuple(wp.shape)}")

    out = torch.zeros(n_waypoints, action_dims, dtype=torch.float32)
    length = min(n_waypoints, wp.size(0))
    dim = min(action_dims, wp.size(-1))
    if length > 0:
        out[:length, :dim] = wp[:length, :dim]
        if length < n_waypoints:
            out[length:, :dim] = wp[length - 1, :dim]

    mask = torch.zeros(n_waypoints, dtype=torch.bool)
    if valid_mask is None:
        mask[:length] = True
    else:
        vm = torch.as_tensor(valid_mask, dtype=torch.bool).view(-1)
        mv_len = min(n_waypoints, vm.numel())
        mask[:mv_len] = vm[:mv_len]
        if mv_len == 0:
            mask[:length] = True
    return out, mask


@dataclass
class BaseMultiAgentDataConfig:
    train_json: str
    cache_root: Optional[str] = None
    history: int = 31
    n_waypoints: int = 10
    action_dims: int = 3
    coarse_cache_size: int = 4096


class BaseMultiAgentJsonDataset(Dataset):
    """Dataset for the base two-agent planner.

    Output tensor shapes:
    - coarse_tokens: (2, history * 4, C)
    - coarse_tidx:   (2, history * 4)
    - fine_tokens:   (2, 64, C)
    - fine_tidx:     (2, 64)
    - waypoints:     (2, n_waypoints, action_dims)
    - valid_mask:    (2, n_waypoints)
    """

    def __init__(self, cfg: BaseMultiAgentDataConfig):
        super().__init__()
        self.cfg = cfg
        self.train_path = Path(cfg.train_json).resolve()
        self.data_root = find_data_root(self.train_path)
        self.cache_root = Path(cfg.cache_root).resolve() if cfg.cache_root else self.data_root / "vision_cache"
        self._index: List[Tuple[str, int]] = []
        self._examples: Optional[List[Dict[str, Any]]] = None
        self._coarse_cache: OrderedDict[str, torch.Tensor] = OrderedDict()

        if self.train_path.is_file() and self.train_path.suffix.lower() == ".json":
            obj = json.loads(self.train_path.read_text(encoding="utf-8"))
            if not isinstance(obj, list):
                raise ValueError(f"Expected list JSON: {self.train_path}")
            self._examples = [x for x in obj if isinstance(x, dict)]
            return

        if self.train_path.is_file() and self.train_path.suffix.lower() == ".jsonl":
            files = [self.train_path]
        elif self.train_path.is_dir():
            files = sorted(self.train_path.rglob("*.jsonl"))
        else:
            raise FileNotFoundError(f"Unsupported train_json path: {self.train_path}")
        if not files:
            raise FileNotFoundError(f"No jsonl files under {self.train_path}")

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
        return len(self._examples) if self._examples is not None else len(self._index)

    def get_example(self, idx: int) -> Dict[str, Any]:
        if self._examples is not None:
            return self._examples[idx]
        fp, offset = self._index[idx]
        with open(fp, "rb") as f:
            f.seek(offset)
            return json.loads(f.readline().decode("utf-8"))

    def _load_token(self, frame_path: Path, fine: bool) -> torch.Tensor:
        coarse_path, fine_path = token_paths_for_frame(self.data_root, self.cache_root, frame_path)
        token_path = fine_path if fine else coarse_path
        key = str(token_path)
        if not fine and self.cfg.coarse_cache_size > 0 and key in self._coarse_cache:
            value = self._coarse_cache.pop(key)
            self._coarse_cache[key] = value
            return value
        if not token_path.exists():
            raise FileNotFoundError(
                f"Missing token cache: {token_path}. Run tools.precache_frames --multi_agent first."
            )
        value = load_tokens_file(token_path)
        if not fine and self.cfg.coarse_cache_size > 0:
            self._coarse_cache[key] = value
            while len(self._coarse_cache) > self.cfg.coarse_cache_size:
                self._coarse_cache.popitem(last=False)
        return value

    def _load_agent_tokens(self, ex: Dict[str, Any], agent_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
        prefix = f"agent{agent_idx}_"
        current_rel = ex.get(prefix + "current")
        if not isinstance(current_rel, str) or not current_rel:
            raise KeyError(f"Missing {prefix}current")
        current_path = resolve_frame_path(self.data_root, current_rel)
        fine = self._load_token(current_path, fine=True)
        fine_tidx = torch.full((fine.size(0),), int(self.cfg.history), dtype=torch.long)

        image_rels = ex.get(prefix + "images", [])
        if not isinstance(image_rels, list):
            image_rels = []
        image_rels = image_rels[-int(self.cfg.history) :]
        missing = int(self.cfg.history) - len(image_rels)
        current_coarse: Optional[torch.Tensor] = None
        first_loaded: Optional[torch.Tensor] = None
        coarse_items: List[torch.Tensor] = []
        tidx_items: List[torch.Tensor] = []

        for t in range(int(self.cfg.history)):
            token: Optional[torch.Tensor] = None
            if t >= missing:
                rel = image_rels[t - missing]
                if isinstance(rel, str) and rel:
                    token = self._load_token(resolve_frame_path(self.data_root, rel), fine=False)
                    if first_loaded is None:
                        first_loaded = token
            if token is None:
                if first_loaded is not None:
                    token = first_loaded
                else:
                    if current_coarse is None:
                        current_coarse = self._load_token(current_path, fine=False)
                    token = current_coarse
            coarse_items.append(token)
            tidx_items.append(torch.full((token.size(0),), t, dtype=torch.long))

        return (
            torch.cat(coarse_items, dim=0),
            torch.cat(tidx_items, dim=0),
            fine,
            fine_tidx,
            str(current_path),
        )

    def _load_targets(self, ex: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        wp_all = ex.get("waypoints", [ex.get("agent1_waypoints"), ex.get("agent2_waypoints")])
        vm_all = ex.get("valid_mask", [ex.get("agent1_valid_mask"), ex.get("agent2_valid_mask")])
        out_wp: List[torch.Tensor] = []
        out_vm: List[torch.Tensor] = []
        for agent_idx in range(2):
            wp_i = wp_all[agent_idx] if isinstance(wp_all, list) and len(wp_all) > agent_idx else None
            vm_i = vm_all[agent_idx] if isinstance(vm_all, list) and len(vm_all) > agent_idx else None
            if wp_i is None:
                raise KeyError(f"Missing waypoints for agent index {agent_idx}")
            wp, vm = fit_waypoints(wp_i, vm_i, self.cfg.n_waypoints, self.cfg.action_dims)
            out_wp.append(wp)
            out_vm.append(vm)
        return torch.stack(out_wp, dim=0), torch.stack(out_vm, dim=0)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.get_example(idx)
        a1 = self._load_agent_tokens(ex, 1)
        a2 = self._load_agent_tokens(ex, 2)
        waypoints, valid_mask = self._load_targets(ex)
        return {
            "coarse_tokens": torch.stack([a1[0], a2[0]], dim=0),
            "coarse_tidx": torch.stack([a1[1], a2[1]], dim=0),
            "fine_tokens": torch.stack([a1[2], a2[2]], dim=0),
            "fine_tidx": torch.stack([a1[3], a2[3]], dim=0),
            "waypoints": waypoints,
            "valid_mask": valid_mask,
            "instruction": ex.get("instruction", "Follow the target person without collision."),
            "episode_id": ex.get("episode_id", ""),
            "step_index": int(ex.get("step_index", idx)),
            "current_path": [a1[4], a2[4]],
        }


def collate_base_multi_agent(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "coarse_tokens": torch.stack([b["coarse_tokens"] for b in batch], dim=0),
        "coarse_tidx": torch.stack([b["coarse_tidx"] for b in batch], dim=0),
        "fine_tokens": torch.stack([b["fine_tokens"] for b in batch], dim=0),
        "fine_tidx": torch.stack([b["fine_tidx"] for b in batch], dim=0),
        "waypoints": torch.stack([b["waypoints"] for b in batch], dim=0),
        "valid_mask": torch.stack([b["valid_mask"] for b in batch], dim=0),
        "instruction": [b["instruction"] for b in batch],
        "episode_id": [b["episode_id"] for b in batch],
        "step_index": torch.tensor([b["step_index"] for b in batch], dtype=torch.long),
        "current_path": [b["current_path"] for b in batch],
    }
