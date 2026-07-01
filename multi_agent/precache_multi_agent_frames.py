#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为双 Agent JSONL 中引用的图片预计算视觉 token。

该脚本读取 make_multi_agent_tracking_data.py 生成的 JSONL/dataset.json，
只缓存训练样本真正引用到的 agent1/agent2 历史帧和当前帧。

输出路径保持和原始单 Agent 流程一致:
frames/.../frame_00001.jpg
->
vision_cache/frames/.../frame_00001_vcoarse.pt
vision_cache/frames/.../frame_00001_vfine.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, List, Optional, Set, Tuple

import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

VisionFeatureCacher = None
VisionCacheConfig = None
grid_pool_tokens = None


def load_encoder_deps() -> None:
    """懒加载 DINO/SigLIP 相关依赖。

    这样 --help / --list_only 不会提前导入 transformers，便于在环境依赖未完全修好时检查数据。
    """
    global VisionFeatureCacher, VisionCacheConfig, grid_pool_tokens
    if VisionFeatureCacher is not None:
        return
    from tools.cache_gridpool import VisionFeatureCacher as _VisionFeatureCacher
    from tools.cache_gridpool import VisionCacheConfig as _VisionCacheConfig
    from tools.cache_gridpool import grid_pool_tokens as _grid_pool_tokens

    VisionFeatureCacher = _VisionFeatureCacher
    VisionCacheConfig = _VisionCacheConfig
    grid_pool_tokens = _grid_pool_tokens


def list_image_files(root: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts])


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                yield obj


def iter_json(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                yield item
    elif isinstance(obj, dict):
        yield obj


def add_path(paths: Set[str], value: Any) -> None:
    if isinstance(value, str) and value:
        paths.add(value)


def add_paths(paths: Set[str], value: Any) -> None:
    if isinstance(value, list):
        for item in value:
            add_path(paths, item)


def collect_refs_from_sample(paths: Set[str], sample: dict) -> None:
    """从一条 JSONL 样本中收集所有会被训练读取的图片路径。"""
    add_paths(paths, sample.get("agent1_images"))
    add_path(paths, sample.get("agent1_current"))
    add_paths(paths, sample.get("agent2_images"))
    add_path(paths, sample.get("agent2_current"))

    agents = sample.get("agents")
    if isinstance(agents, dict):
        for payload in agents.values():
            if not isinstance(payload, dict):
                continue
            add_paths(paths, payload.get("images"))
            add_path(paths, payload.get("current"))

    # Backward-compatible single-agent keys, useful for quick checks.
    add_paths(paths, sample.get("images"))
    add_path(paths, sample.get("current"))


def collect_frame_refs(data_root: Path, json_root: Optional[Path], dataset_json: Optional[Path], scan_frames: bool) -> List[Path]:
    """从 JSONL/dataset.json 收集图片引用。

    默认只收集 JSON 中引用过的帧；如果传 --scan_frames，则额外扫描 frames/ 下所有图片。
    """
    rel_paths: Set[str] = set()

    json_files: List[Path] = []
    if json_root is not None and json_root.exists():
        if json_root.is_file():
            json_files.append(json_root)
        else:
            json_files.extend(sorted(json_root.rglob("*.jsonl")))
            json_files.extend(sorted(json_root.rglob("*.json")))
    else:
        default_jsonl = data_root / "jsonl"
        if default_jsonl.exists():
            json_files.extend(sorted(default_jsonl.rglob("*.jsonl")))
        if dataset_json is not None and dataset_json.exists():
            json_files.append(dataset_json)
        elif (data_root / "dataset.json").exists():
            json_files.append(data_root / "dataset.json")

    for path in json_files:
        try:
            iterator = iter_jsonl(path) if path.suffix.lower() == ".jsonl" else iter_json(path)
            for sample in iterator:
                collect_refs_from_sample(rel_paths, sample)
        except Exception as exc:
            print(f"[warn] failed to read {path}: {exc}")

    abs_paths: Set[Path] = set()
    for rel in rel_paths:
        p = Path(rel)
        if p.is_absolute():
            abs_paths.add(p)
        else:
            abs_paths.add((data_root / p).resolve())

    if scan_frames or not abs_paths:
        frames_dir = data_root / "frames"
        if frames_dir.exists():
            abs_paths.update(p.resolve() for p in list_image_files(frames_dir))

    return sorted(abs_paths)


@torch.inference_mode()
def encode_single(pil: Image.Image, enc: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """对单张图片编码，生成粗粒度 4 tokens 和细粒度 64 tokens。"""
    assert grid_pool_tokens is not None
    tok_dino, hp, wp = enc._encode_dino([pil])
    tok_sigl = enc._encode_siglip([pil], out_hw=(hp, wp))
    tokens = torch.cat([tok_dino, tok_sigl], dim=-1)
    vfine = grid_pool_tokens(tokens, hp, wp, out_tokens=64)
    vcoarse = grid_pool_tokens(tokens, hp, wp, out_tokens=4)
    return vcoarse[0].cpu().float(), vfine[0].cpu().float()


def rel_to_data_root(path: Path, data_root: Path) -> Path:
    """把绝对图片路径转成相对 data_root 的路径，用于镜像 cache 目录。"""
    try:
        return path.resolve().relative_to(data_root.resolve())
    except ValueError:
        parts = path.parts
        if "frames" in parts:
            idx = parts.index("frames")
            return Path(*parts[idx:])
        return Path(path.name)


def save_tensor(path: Path, tensor: torch.Tensor, save_float32: bool) -> None:
    """保存 token。默认半精度，节省磁盘；必要时可传 --save_float32。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = tensor if save_float32 else tensor.half()
    try:
        torch.save(obj, str(path))
    except Exception:
        torch.save(tensor, str(path))


def maybe_make_coarse_from_fine(vfine_path: Path, vcoarse_path: Path, save_float32: bool) -> bool:
    """如果已有 64-token fine cache，则直接池化生成 4-token coarse cache。

    这能避免重复跑重型视觉编码器。
    """
    if not vfine_path.exists() or vcoarse_path.exists():
        return False
    assert grid_pool_tokens is not None
    try:
        vf = torch.load(str(vfine_path), map_location="cpu")
        if isinstance(vf, dict):
            for key in ("Vfine", "tokens", "feat", "features"):
                if key in vf and isinstance(vf[key], torch.Tensor):
                    vf = vf[key]
                    break
        if not isinstance(vf, torch.Tensor):
            return False
        vf = vf.float()
        if vf.dim() == 3 and vf.size(0) == 1:
            vf = vf[0]
        if vf.dim() != 2 or vf.size(0) != 64:
            return False
        vc = grid_pool_tokens(vf.unsqueeze(0), 8, 8, out_tokens=4)[0].cpu().float()
        save_tensor(vcoarse_path, vc, save_float32)
        return True
    except Exception:
        return False


def parse_args() -> argparse.Namespace:
    """命令行参数。"""
    parser = argparse.ArgumentParser(description="Precache vision tokens for two-agent TrackVLA JSONL frames.")
    parser.add_argument("--data_root", type=str, required=True, help="Training data root produced by make_multi_agent_tracking_data.py.")
    parser.add_argument("--cache_root", type=str, default=None, help="Defaults to <data_root>/vision_cache.")
    parser.add_argument("--json_root", type=str, default=None, help="JSONL directory/file. Defaults to <data_root>/jsonl plus dataset.json.")
    parser.add_argument("--dataset_json", type=str, default=None)
    parser.add_argument("--scan_frames", action="store_true", help="Also scan every image under <data_root>/frames.")
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--save_float32", action="store_true")
    parser.add_argument("--list_only", action="store_true", help="Only list/count frame refs without loading vision encoders.")
    return parser.parse_args()


def main() -> None:
    """脚本入口。"""
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    cache_root = Path(args.cache_root).resolve() if args.cache_root else (data_root / "vision_cache")
    json_root = Path(args.json_root).resolve() if args.json_root else None
    dataset_json = Path(args.dataset_json).resolve() if args.dataset_json else None
    cache_root.mkdir(parents=True, exist_ok=True)

    frame_paths = collect_frame_refs(data_root, json_root, dataset_json, args.scan_frames)
    if args.limit > 0:
        frame_paths = frame_paths[: args.limit]
    print(f"Frames to check: {len(frame_paths)}")
    print(f"Cache root: {cache_root}")
    if args.list_only:
        # 只检查 JSONL 引用了哪些帧，不加载视觉模型。
        for path in frame_paths[:20]:
            print(path)
        if len(frame_paths) > 20:
            print(f"... {len(frame_paths) - 20} more")
        return

    load_encoder_deps()
    assert VisionFeatureCacher is not None
    assert VisionCacheConfig is not None

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = VisionFeatureCacher(
        VisionCacheConfig(
            image_size=args.image_size,
            batch_size=args.batch_size,
            device=device,
        )
    )
    enc.eval()

    checked = 0
    generated = 0
    skipped_missing = 0
    failed = 0
    for img_path in frame_paths:
        checked += 1
        if not img_path.exists():
            skipped_missing += 1
            print(f"[warn] missing frame: {img_path}")
            continue

        rel = rel_to_data_root(img_path, data_root)
        token_dir = cache_root / rel.parent
        vfine_path = token_dir / f"{rel.stem}_vfine.pt"
        vcoarse_path = token_dir / f"{rel.stem}_vcoarse.pt"

        if vfine_path.exists() and vcoarse_path.exists():
            continue
        if maybe_make_coarse_from_fine(vfine_path, vcoarse_path, args.save_float32):
            generated += 1
            continue

        try:
            pil = Image.open(str(img_path)).convert("RGB")
            vc, vf = encode_single(pil, enc)
            if not vfine_path.exists():
                save_tensor(vfine_path, vf, args.save_float32)
            if not vcoarse_path.exists():
                save_tensor(vcoarse_path, vc, args.save_float32)
            generated += 1
        except Exception as exc:
            failed += 1
            print(f"[warn] failed on {img_path}: {exc}")

        if checked % 25 == 0:
            print(f"progress checked={checked} generated={generated} missing={skipped_missing} failed={failed}")

    print(f"Completed precache: checked={checked} generated={generated} missing={skipped_missing} failed={failed}")


if __name__ == "__main__":
    main()
