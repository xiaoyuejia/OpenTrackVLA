#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import os
import argparse
import json
import sys
import time
from contextlib import nullcontext
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import torch
from PIL import Image

from tools.cache_gridpool import (
    VisionFeatureCacher,
    VisionCacheConfig,
    adapt_siglip_grid,
    build_roi_cache_payload,
    crop_target_roi,
    grid_pool_tokens,
    load_roi_cache,
)


def _list_image_files(dir_path: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in exts])


def _detect_layout_and_collect(data_root: Path) -> Tuple[List[List[Path]], List[List[str]], str]:
    """Return (files_by_view, relpaths_by_view, layout_tag).

    - files_by_view: [V][T] absolute Paths
    - relpaths_by_view: [V][T] relative path parts (to mirror under cache)
    - layout_tag: one of {"frames", "views", "flat"}
    """
    root = data_root
    frames_dir = root / "frames"
    if frames_dir.exists() and frames_dir.is_dir():
        # Choose seed
        seed_env = os.getenv("TRACKVLA_SEED", "").strip()
        seed_dirs = sorted([p for p in frames_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")])
        if not seed_dirs:
            # Treat each leaf directory under frames/ as an independent view (video clips, etc.)
            files_by_view = []
            rels_by_view = []

            # Include images directly under frames_dir if present
            root_imgs = _list_image_files(frames_dir)
            if root_imgs:
                files_by_view.append(root_imgs)
                rels_by_view.append([str(Path("frames") / p.name) for p in root_imgs])

            # Walk subdirectories to find image sets
            subdirs = sorted([p for p in frames_dir.rglob('*') if p.is_dir()])
            for subdir in subdirs:
                imgs = _list_image_files(subdir)
                if not imgs:
                    continue
                rel_prefix = Path("frames") / subdir.relative_to(frames_dir)
                files_by_view.append(imgs)
                rels_by_view.append([str(rel_prefix / p.name) for p in imgs])

            if not files_by_view:
                raise RuntimeError(f"No frame images found under {frames_dir}")
            return files_by_view, rels_by_view, "frames"

        if seed_env:
            if seed_env.isdigit():
                seed_choice = f"seed_{seed_env}"
            else:
                seed_choice = seed_env
            seed_dirs = [p for p in seed_dirs if p.name == seed_choice]
            if not seed_dirs:
                raise RuntimeError(f"Requested seed not found: {seed_choice}")

        # Choose scene
        scene_env = os.getenv("TRACKVLA_SCENE", "").strip()
        scenes = []
        for seed_path in seed_dirs:
            for s in sorted([p for p in seed_path.iterdir() if p.is_dir()]):
                if (not scene_env) or (s.name == scene_env):
                    scenes.append((seed_path.name, s))
        if not scenes:
            raise RuntimeError(f"No scene directories found under selected seeds of {frames_dir}")

        # Each camera directory is a view
        files_by_view: List[List[Path]] = []
        rels_by_view: List[List[str]] = []
        for seed_name, scene_path in scenes:
            cam_dirs = sorted([p for p in scene_path.iterdir() if p.is_dir()])
            if not cam_dirs:
                # Fallback: images directly under scene
                imgs = _list_image_files(scene_path)
                if imgs:
                    files_by_view.append(imgs)
                    rels_by_view.append([str(Path("frames")/seed_name/scene_path.name/p.name) for p in imgs])
                continue
            for cam_dir in cam_dirs:
                imgs = _list_image_files(cam_dir)
                if not imgs:
                    continue
                files_by_view.append(imgs)
                rels_by_view.append([str(Path("frames")/seed_name/scene_path.name/cam_dir.name/p.name) for p in imgs])

        if not files_by_view:
            raise RuntimeError("No images found in frames layout")
        return files_by_view, rels_by_view, "frames"

    # Layout 2: <root>/<view>/*
    view_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if view_dirs:
        files_by_view = []
        rels_by_view = []
        for vd in view_dirs:
            imgs = _list_image_files(vd)
            if not imgs:
                continue
            files_by_view.append(imgs)
            rels_by_view.append([str(Path(vd.name)/p.name) for p in imgs])
        if not files_by_view:
            raise RuntimeError(f"No images found under any subdirectory of {root}")
        return files_by_view, rels_by_view, "views"

    # Layout 3: <root>/*
    imgs = _list_image_files(root)
    if not imgs:
        raise RuntimeError(f"No images found in {root}")
    return [imgs], [[p.name for p in imgs]], "flat"


@torch.inference_mode()
def _encode_single(pil: Image.Image, enc: VisionFeatureCacher) -> Tuple[torch.Tensor, torch.Tensor]:
    # Reproduce train_planner encoder contract
    tok_dino, Hp, Wp = enc._encode_dino([pil])
    tok_sigl = enc._encode_siglip([pil], out_hw=(Hp, Wp))
    Vt_cat = torch.cat([tok_dino, tok_sigl], dim=-1)       # (1, P, C)
    Vfine = grid_pool_tokens(Vt_cat, Hp, Wp, out_tokens=64)
    Vcoarse = grid_pool_tokens(Vt_cat, Hp, Wp, out_tokens=4)
    return Vcoarse[0].cpu().float(), Vfine[0].cpu().float()


@torch.inference_mode()
def _encode_batch(
    pils: List[Image.Image],
    enc: VisionFeatureCacher,
    amp_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """批量编码图片，返回 CPU 上的 coarse/fine token。"""
    amp_context = (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if amp_dtype is not None and enc.device.type == "cuda"
        else nullcontext()
    )
    with amp_context:
        tok_dino, hp, wp = enc._encode_dino(pils)
        tok_sigl = enc._encode_siglip(pils, out_hw=(hp, wp))
        tokens = torch.cat([tok_dino, tok_sigl], dim=-1)
        vfine = grid_pool_tokens(tokens, hp, wp, out_tokens=64)
        vcoarse = grid_pool_tokens(tokens, hp, wp, out_tokens=4)
    return vcoarse.cpu().float(), vfine.cpu().float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', type=str, required=True, help='Dataset root to scan for frames')
    ap.add_argument('--cache_root', type=str, default=None, help='Where to mirror cache; defaults to <data_root>/vision_cache')
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--image_size', type=int, default=384)
    ap.add_argument('--vision_resize_mode', choices=("letterbox", "stretch"), default="letterbox")
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    cache_root = Path(args.cache_root).resolve() if args.cache_root else (data_root / 'vision_cache')
    os.makedirs(cache_root, exist_ok=True)

    files_by_view, rels_by_view, _layout = _detect_layout_and_collect(data_root)

    # Initialize encoder (CPU when multi-worker; here single-process heuristic)
    use_cuda = torch.cuda.is_available()
    enc = VisionFeatureCacher(
        VisionCacheConfig(
            image_size=args.image_size,
            batch_size=args.batch_size,
            device=('cuda' if use_cuda else 'cpu'),
            resize_mode=args.vision_resize_mode,
        )
    )
    enc.eval()

    # Iterate views and frames; skip existing token files
    total = 0
    done = 0
    for v_idx in range(len(files_by_view)):
        files = files_by_view[v_idx]
        rels = rels_by_view[v_idx]
        for t_idx in range(len(files)):
            abs_img = files[t_idx]
            rel_path = Path(rels[t_idx])
            token_dir = cache_root / rel_path.parent
            base = rel_path.stem
            vf_path = token_dir / f"{base}_vfine.pt"
            vc_path = token_dir / f"{base}_vcoarse.pt"
            total += 1
            vf_exists = vf_path.exists()
            vc_exists = vc_path.exists()
            if vf_exists and vc_exists:
                continue
            token_dir.mkdir(parents=True, exist_ok=True)
            try:
                # Fast path: if Vfine exists but Vcoarse is missing, derive Vcoarse by pooling
                # 8x8 -> 2x2 without re-encoding heavy towers.
                if vf_exists and (not vc_exists):
                    try:
                        vf = torch.load(str(vf_path), map_location='cpu')  # (64, C)
                        vf = vf.float() if vf.dtype != torch.float32 else vf
                        vc = grid_pool_tokens(vf.unsqueeze(0), 8, 8, out_tokens=4)[0].cpu()  # (4, C)
                        try:
                            torch.save(vc.half(), str(vc_path))
                        except Exception:
                            torch.save(vc, str(vc_path))
                        done += 1
                        print(total, done)
                        continue
                    except Exception as e:
                        # Fall back to full re-encode on any failure
                        pass

                pil = Image.open(str(abs_img)).convert('RGB')
                vc, vf = _encode_single(pil, enc)
                if not vf_exists:
                    try:
                        torch.save(vf.half(), str(vf_path))
                    except Exception:
                        torch.save(vf, str(vf_path))
                if not vc_exists:
                    try:
                        torch.save(vc.half(), str(vc_path))
                    except Exception:
                        torch.save(vc, str(vc_path))
                done += 1
            except Exception as e:
                print(f"[warn] failed on {abs_img}: {e}")
            print (total, done)

    print(f"Completed precache: generated {done} / {total} frame token pairs under {cache_root}")


def iter_multi_agent_jsonl(path: Path) -> Iterable[dict]:
    """逐行读取 JSONL。

    数据流：jsonl 文件 -> dict sample。
    只 yield dict，跳过空行和非 dict 内容。
    """
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                yield obj


def iter_multi_agent_json(path: Path) -> Iterable[dict]:
    """读取聚合 dataset.json。

    支持两种格式：
    - list[dict]：make_tracking_data.py --multi_agent 默认聚合格式。
    - dict：便于临时调试单条样本。
    """
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                yield item
    elif isinstance(obj, dict):
        yield obj


def add_multi_agent_path(paths: Set[str], value: Any) -> None:
    """把一个字符串图片路径加入集合。"""
    if isinstance(value, str) and value:
        paths.add(value)


def add_multi_agent_paths(paths: Set[str], value: Any) -> None:
    """把 list[str] 图片路径加入集合。"""
    if isinstance(value, list):
        for item in value:
            add_multi_agent_path(paths, item)


def collect_multi_agent_refs_from_sample(paths: Set[str], sample: dict) -> None:
    """从一条训练样本中收集图片引用。

    双 Agent JSONL 的主字段：
    - agent1_images / agent1_current
    - agent2_images / agent2_current

    同时兼容 agents 结构化字段和旧单 Agent images/current 字段，方便混合检查。
    """
    add_multi_agent_paths(paths, sample.get("agent1_images"))
    add_multi_agent_path(paths, sample.get("agent1_current"))
    add_multi_agent_paths(paths, sample.get("agent2_images"))
    add_multi_agent_path(paths, sample.get("agent2_current"))

    agents = sample.get("agents")
    if isinstance(agents, dict):
        for payload in agents.values():
            if not isinstance(payload, dict):
                continue
            add_multi_agent_paths(paths, payload.get("images"))
            add_multi_agent_path(paths, payload.get("current"))

    add_multi_agent_paths(paths, sample.get("images"))
    add_multi_agent_path(paths, sample.get("current"))


def _bbox_diff(a: Any, b: Any) -> float:
    if not isinstance(a, list) or not isinstance(b, list) or len(a) < 4 or len(b) < 4:
        return float("inf")
    try:
        return max(abs(float(a[i]) - float(b[i])) for i in range(4))
    except Exception:
        return float("inf")


def _add_roi_ref(
    roi_refs: Dict[str, Dict[str, Any]],
    current: Any,
    bbox: Any,
    agent_name: str,
    bbox_format: str = "cxcywh_norm",
) -> None:
    if not isinstance(current, str) or not current:
        return
    if not isinstance(bbox, list) or len(bbox) < 4:
        return
    try:
        bbox_vals = [float(v) for v in bbox[:4]]
    except Exception:
        return
    payload = {
        "bbox": bbox_vals,
        "bbox_format": bbox_format,
        "agent_name": agent_name,
    }
    existing = roi_refs.get(current)
    if existing is not None:
        diff = _bbox_diff(existing.get("bbox"), payload["bbox"])
        if diff > 1e-3:
            print(
                f"[warn] duplicate ROI bbox differs for {current}: "
                f"first={existing.get('bbox')} new={payload['bbox']} max_diff={diff:.6g}",
                flush=True,
            )
        return
    roi_refs[current] = payload


def collect_multi_agent_roi_refs_from_sample(roi_refs: Dict[str, Dict[str, Any]], sample: dict) -> None:
    _add_roi_ref(roi_refs, sample.get("agent1_current"), sample.get("agent1_bbox"), str(sample.get("agent1_name", "agent1")))
    _add_roi_ref(roi_refs, sample.get("agent2_current"), sample.get("agent2_bbox"), str(sample.get("agent2_name", "agent2")))

    agents = sample.get("agents")
    if isinstance(agents, dict):
        for agent_name, payload in agents.items():
            if not isinstance(payload, dict):
                continue
            _add_roi_ref(roi_refs, payload.get("current"), payload.get("bbox"), str(agent_name))


def list_multi_agent_images(root: Path) -> List[Path]:
    """扫描目录下所有图片，供 --scan_frames 兜底使用。"""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts])


def collect_multi_agent_frame_refs(
    data_root: Path,
    json_root: Optional[Path],
    dataset_json: Optional[Path],
    scan_frames: bool,
) -> List[Path]:
    """从双 Agent JSONL/dataset.json 收集需要缓存的图片。

    推荐数据流：
    tools.make_tracking_data --multi_agent 生成 jsonl
    -> 本函数读取 jsonl 中被训练实际引用的帧
    -> 只缓存这些帧，避免对无用图片重复编码。
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
            iterator = iter_multi_agent_jsonl(path) if path.suffix.lower() == ".jsonl" else iter_multi_agent_json(path)
            for sample in iterator:
                collect_multi_agent_refs_from_sample(rel_paths, sample)
        except Exception as exc:
            print(f"[warn] failed to read {path}: {exc}")

    abs_paths: Set[Path] = set()
    for rel in rel_paths:
        p = Path(rel)
        abs_paths.add(p if p.is_absolute() else (data_root / p).absolute())

    if scan_frames or not abs_paths:
        frames_dir = data_root / "frames"
        if frames_dir.exists():
            abs_paths.update(p.resolve() for p in list_multi_agent_images(frames_dir))
    return sorted(abs_paths)


def collect_multi_agent_roi_refs(
    data_root: Path,
    json_root: Optional[Path],
    dataset_json: Optional[Path],
) -> Dict[Path, Dict[str, Any]]:
    roi_refs: Dict[str, Dict[str, Any]] = {}
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
            iterator = iter_multi_agent_jsonl(path) if path.suffix.lower() == ".jsonl" else iter_multi_agent_json(path)
            for sample in iterator:
                collect_multi_agent_roi_refs_from_sample(roi_refs, sample)
        except Exception as exc:
            print(f"[warn] failed to read ROI refs from {path}: {exc}", flush=True)

    out: Dict[Path, Dict[str, Any]] = {}
    for rel, payload in roi_refs.items():
        p = Path(rel)
        out[p if p.is_absolute() else (data_root / p).absolute()] = payload
    return out


def collect_multi_agent_roi_refs_from_raw(data_root: Path, raw_root: Path) -> Dict[Path, Dict[str, Any]]:
    """Build exact normalized bbox refs directly from raw per-step info JSON.

    Raw ``target_bbox`` is pixel ``xywh`` and the rendered frames are 640x480.
    This avoids reparsing the very large processed JSONL files when the raw
    episode metadata is available.
    """
    out: Dict[Path, Dict[str, Any]] = {}
    frames_root = data_root / "frames"
    for scene_dir in sorted(p for p in frames_root.iterdir() if p.is_dir()):
        raw_scene = raw_root / scene_dir.name
        for episode_dir in sorted(p for p in scene_dir.iterdir() if p.is_dir()):
            for agent in ("drone", "robotdog"):
                info_path = raw_scene / f"{episode_dir.name}_{agent}_info.json"
                frame_dir = episode_dir / agent
                if not info_path.is_file() or not frame_dir.is_dir():
                    continue
                try:
                    rows = json.loads(info_path.read_text(encoding="utf-8"))
                except Exception as exc:
                    print(f"[warn] failed to read raw bbox info {info_path}: {exc}", flush=True)
                    continue
                frame_paths = sorted(frame_dir.glob("frame_*.jpg"))
                for index, frame_path in enumerate(frame_paths):
                    if index >= len(rows) or not isinstance(rows[index], dict):
                        continue
                    bbox = rows[index].get("target_bbox")
                    if not isinstance(bbox, list) or len(bbox) < 4:
                        continue
                    try:
                        x, y, w, h = [float(v) for v in bbox[:4]]
                    except Exception:
                        continue
                    if w <= 0.0 or h <= 0.0:
                        continue
                    out[frame_path.resolve()] = {
                        "bbox": [(x + 0.5 * w) / 640.0, (y + 0.5 * h) / 480.0, w / 640.0, h / 480.0],
                        "bbox_format": "cxcywh_norm",
                        "agent_name": agent,
                    }
    return out


def rel_multi_agent_to_data_root(path: Path, data_root: Path) -> Path:
    """把绝对图片路径映射成相对 data_root 的缓存路径。

    输出用于镜像缓存目录：
    image: data_root/frames/.../frame_00001.jpg
    cache: cache_root/frames/.../frame_00001_vfine.pt
    """
    try:
        return path.absolute().relative_to(data_root.absolute())
    except ValueError:
        try:
            return path.resolve().relative_to(data_root.resolve())
        except ValueError:
            parts = path.parts
            if "frames" in parts:
                idx = parts.index("frames")
                return Path(*parts[idx:])
            return Path(path.name)


def save_multi_agent_tensor(path: Path, tensor: torch.Tensor, save_float32: bool) -> None:
    """保存视觉 token。

    默认保存 half，减少磁盘占用；传 --save_float32 可保留 float32。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = tensor if save_float32 else tensor.half()
    try:
        torch.save(obj, str(path))
    except Exception:
        torch.save(tensor, str(path))


def load_multi_agent_rgb(path: Path) -> Image.Image:
    """Load one frame as RGB and detach it from the file handle."""
    with Image.open(str(path)) as img:
        return img.convert("RGB")


def save_multi_agent_token_pair(
    vfine_path: Path,
    vcoarse_path: Path,
    vf: torch.Tensor,
    vc: torch.Tensor,
    save_float32: bool,
) -> None:
    """Save fine/coarse token files for one frame."""
    if not vfine_path.exists():
        save_multi_agent_tensor(vfine_path, vf, save_float32)
    if not vcoarse_path.exists():
        save_multi_agent_tensor(vcoarse_path, vc, save_float32)


def save_multi_agent_roi_payload(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))


def drain_completed_saves(save_futures: Set[Future], block: bool = False) -> Tuple[int, int]:
    """Collect finished async save tasks.

    Returns:
        (completed, failed)
    """
    completed = 0
    failed = 0
    while save_futures:
        done: Set[Future]
        if block:
            done, _pending = wait(save_futures, return_when=FIRST_COMPLETED)
        else:
            done = {fut for fut in save_futures if fut.done()}
            if not done:
                break
        for fut in done:
            save_futures.remove(fut)
            completed += 1
            try:
                fut.result()
            except Exception as exc:
                failed += 1
                print(f"[warn] async save failed: {exc}", flush=True)
        if not block:
            break
    return completed, failed


def maybe_make_multi_agent_coarse_from_fine(vfine_path: Path, vcoarse_path: Path, save_float32: bool) -> bool:
    """如果已有 64-token fine cache，直接池化生成 4-token coarse cache。

    数据流：vfine(64,C) -> grid_pool(8x8 -> 2x2) -> vcoarse(4,C)。
    这样可以避免重复跑 DINO/SigLIP。
    """
    if not vfine_path.exists() or vcoarse_path.exists():
        return False
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
        save_multi_agent_tensor(vcoarse_path, vc, save_float32)
        return True
    except Exception:
        return False


def roi_cache_is_valid(
    path: Path,
    roi_token_count: int,
    roi_expand_ratio: float,
    roi_make_square: bool,
    expected_bbox: Any = None,
    allow_legacy_missing_source_bbox: bool = False,
) -> bool:
    if not path.exists():
        return False
    try:
        load_roi_cache(
            path,
            roi_token_count,
            roi_expand_ratio,
            roi_make_square,
            expected_bbox=expected_bbox,
            allow_legacy_missing_source_bbox=allow_legacy_missing_source_bbox,
        )
        return True
    except Exception:
        return False


def parse_multi_agent_precache_args() -> argparse.Namespace:
    """双 Agent 预缓存参数。

    原单 Agent 参数不变；该 parser 只在入口包含 --multi_agent 时使用。
    """
    parser = argparse.ArgumentParser(description="Precache vision tokens for two-agent TrackVLA JSONL frames.")
    parser.add_argument("--data_root", type=str, required=True, help="Training data root produced by tools.make_tracking_data --multi_agent.")
    parser.add_argument("--cache_root", type=str, default=None, help="Defaults to <data_root>/vision_cache.")
    parser.add_argument("--json_root", type=str, default=None, help="JSONL directory/file. Defaults to <data_root>/jsonl plus dataset.json.")
    parser.add_argument("--dataset_json", type=str, default=None)
    parser.add_argument(
        "--allow_legacy_roi_cache_without_source_bbox",
        action="store_true",
        help="Reuse legacy ROI payloads that match token/crop settings but predate source_bbox metadata.",
    )
    parser.add_argument(
        "--trust_existing_roi_cache",
        action="store_true",
        help="Treat an existing ROI file as valid without torch.load; useful after an explicit cache audit.",
    )
    parser.add_argument("--scan_frames", action="store_true", help="Also scan every image under <data_root>/frames.")
    parser.add_argument(
        "--frame_list",
        type=str,
        default=None,
        help="Optional newline-delimited image list; avoids slow recursive pathlib scanning.",
    )
    parser.add_argument(
        "--raw_root",
        type=str,
        default=None,
        help="Optional raw episode root for direct target_bbox loading.",
    )
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--vision_resize_mode", choices=("letterbox", "stretch"), default="letterbox")
    parser.add_argument("--with_roi", action="store_true", help="Also generate current-frame target ROI token caches.")
    parser.add_argument("--roi_token_count", type=int, default=16)
    parser.add_argument("--roi_expand_ratio", type=float, default=1.5)
    parser.add_argument("--roi_make_square", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite_roi", action="store_true")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_shards", type=int, default=1, help="Split frame list into this many disjoint shards.")
    parser.add_argument("--shard_id", type=int, default=0, help="Zero-based shard index processed by this worker.")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    parser.add_argument(
        "--encoder_amp",
        choices=("none", "bfloat16", "float16"),
        default="none",
        help="CUDA autocast dtype for DINO/SigLIP encoding; cache files keep the same FP16 format.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--save_float32", action="store_true")
    parser.add_argument("--image_workers", type=int, default=8, help="Threads for parallel image decode in multi-agent precache.")
    parser.add_argument("--save_workers", type=int, default=8, help="Threads for async torch.save in multi-agent precache.")
    parser.add_argument(
        "--max_pending_saves",
        type=int,
        default=256,
        help="Maximum queued async frame-save tasks before the encoder waits.",
    )
    parser.add_argument("--list_only", action="store_true", help="Only list/count frame refs without encoding.")
    return parser.parse_args()


def main_multi_agent_precache() -> None:
    """双 Agent 视觉 token 预缓存入口。

    总数据流：
    JSONL 中的 agent1/agent2 图片路径
    -> 定位 data_root/frames 下的图片
    -> DINO + SigLIP 编码
    -> 保存 fine(64 tokens) 与 coarse(4 tokens) 到 vision_cache。
    """
    args = parse_multi_agent_precache_args()
    data_root = Path(args.data_root).resolve()
    cache_root = Path(args.cache_root).resolve() if args.cache_root else (data_root / "vision_cache")
    json_root = Path(args.json_root).resolve() if args.json_root else None
    dataset_json = Path(args.dataset_json).resolve() if args.dataset_json else None
    cache_root.mkdir(parents=True, exist_ok=True)
    roi_side = int(round(args.roi_token_count ** 0.5))
    if roi_side * roi_side != int(args.roi_token_count):
        raise ValueError("--roi_token_count must be a perfect square")

    if args.frame_list:
        frame_list_path = Path(args.frame_list).expanduser().resolve()
        frame_paths = [Path(line.strip()).resolve() for line in frame_list_path.open("r", encoding="utf-8") if line.strip()]
    else:
        frame_paths = collect_multi_agent_frame_refs(data_root, json_root, dataset_json, args.scan_frames)
    if args.with_roi:
        roi_refs = (
            collect_multi_agent_roi_refs_from_raw(data_root, Path(args.raw_root).resolve())
            if args.raw_root
            else collect_multi_agent_roi_refs(data_root, json_root, dataset_json)
        )
    else:
        roi_refs = {}
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError(f"invalid shard_id={args.shard_id} for num_shards={args.num_shards}")
    all_frame_count = len(frame_paths)
    frame_paths = frame_paths[args.shard_id :: args.num_shards]
    if args.limit > 0:
        frame_paths = frame_paths[: args.limit]
    print(
        f"Frames to check: {len(frame_paths)} "
        f"(shard {args.shard_id}/{args.num_shards}, total before sharding={all_frame_count})"
    )
    print(f"Cache root: {cache_root}")
    print(f"Vision resize: image_size={args.image_size} mode={args.vision_resize_mode}")
    if args.with_roi:
        print(
            f"ROI cache: refs={len(roi_refs)} token_count={args.roi_token_count} "
            f"expand={args.roi_expand_ratio} square={args.roi_make_square} "
            "roi_bbox_source=ground_truth evaluation_protocol=oracle_roi_upper_bound",
            flush=True,
        )
    if args.list_only:
        for path in frame_paths[:20]:
            print(path)
        if len(frame_paths) > 20:
            print(f"... {len(frame_paths) - 20} more")
        return

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    enc = VisionFeatureCacher(
        VisionCacheConfig(
            image_size=args.image_size,
            batch_size=args.batch_size,
            device=device,
            resize_mode=args.vision_resize_mode,
        )
    )
    enc.eval()
    amp_dtype = {
        "none": None,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.encoder_amp]

    checked = 0
    generated = 0
    saved = 0
    skipped_missing = 0
    failed = 0
    batch_size = max(int(args.batch_size), 1)
    image_workers = max(int(args.image_workers), 1)
    save_workers = max(int(args.save_workers), 1)
    max_pending_saves = max(int(args.max_pending_saves), save_workers)
    start_time = time.time()
    last_progress = start_time
    save_futures: Set[Future] = set()

    print(
        f"Precache workers: image_workers={image_workers} save_workers={save_workers} "
        f"max_pending_saves={max_pending_saves}",
        flush=True,
    )

    with ThreadPoolExecutor(max_workers=image_workers) as image_pool, ThreadPoolExecutor(max_workers=save_workers) as save_pool:
        for start in range(0, len(frame_paths), batch_size):
            saved_done, save_failed = drain_completed_saves(save_futures)
            saved += saved_done
            failed += save_failed
            while len(save_futures) >= max_pending_saves:
                saved_done, save_failed = drain_completed_saves(save_futures, block=True)
                saved += saved_done
                failed += save_failed

            batch_paths = frame_paths[start : start + batch_size]
            pending_global: List[Tuple[Path, Path, Path]] = []
            pending_roi: List[Tuple[Path, Path, Dict[str, Any]]] = []
            for img_path in batch_paths:
                checked += 1
                if not img_path.exists():
                    skipped_missing += 1
                    print(f"[warn] missing frame: {img_path}", flush=True)
                    continue

                rel = rel_multi_agent_to_data_root(img_path, data_root)
                token_dir = cache_root / rel.parent
                vfine_path = token_dir / f"{rel.stem}_vfine.pt"
                vcoarse_path = token_dir / f"{rel.stem}_vcoarse.pt"
                if not (vfine_path.exists() and vcoarse_path.exists()):
                    if maybe_make_multi_agent_coarse_from_fine(vfine_path, vcoarse_path, args.save_float32):
                        generated += 1
                        saved += 1
                    else:
                        pending_global.append((img_path, vfine_path, vcoarse_path))

                if args.with_roi and img_path in roi_refs:
                    vroi_path = token_dir / f"{rel.stem}_vroi.pt"
                    existing_roi_valid = (
                        vroi_path.exists()
                        if args.trust_existing_roi_cache
                        else roi_cache_is_valid(
                            vroi_path,
                            args.roi_token_count,
                            args.roi_expand_ratio,
                            bool(args.roi_make_square),
                            expected_bbox=roi_refs[img_path].get("bbox"),
                            allow_legacy_missing_source_bbox=bool(
                                args.allow_legacy_roi_cache_without_source_bbox
                            ),
                        )
                    )
                    if args.overwrite_roi or not existing_roi_valid:
                        pending_roi.append((img_path, vroi_path, roi_refs[img_path]))

            read_futures = [image_pool.submit(load_multi_agent_rgb, item[0]) for item in pending_global]
            pils: List[Image.Image] = []
            valid_pending: List[Tuple[Path, Path, Path]] = []
            for item, fut in zip(pending_global, read_futures):
                try:
                    pils.append(fut.result())
                    valid_pending.append(item)
                except Exception as exc:
                    failed += 1
                    print(f"[warn] failed to read {item[0]}: {exc}", flush=True)
            if pils:
                try:
                    batch_vc, batch_vf = _encode_batch(pils, enc, amp_dtype=amp_dtype)
                    for index, (_img_path, vfine_path, vcoarse_path) in enumerate(valid_pending):
                        save_futures.add(
                            save_pool.submit(
                                save_multi_agent_token_pair,
                                vfine_path,
                                vcoarse_path,
                                batch_vf[index].cpu(),
                                batch_vc[index].cpu(),
                                args.save_float32,
                            )
                        )
                        generated += 1
                except Exception as exc:
                    failed += len(valid_pending)
                    print(f"[warn] failed batch starting at frame {start}: {exc}", flush=True)
                finally:
                    for pil in pils:
                        pil.close()

            roi_read_futures = [image_pool.submit(load_multi_agent_rgb, item[0]) for item in pending_roi]
            roi_pils: List[Image.Image] = []
            valid_roi_pending: List[Tuple[Path, Path, Dict[str, Any], bool, Tuple[int, int, int, int]]] = []
            for item, fut in zip(pending_roi, roi_read_futures):
                try:
                    src = fut.result()
                    meta = item[2]
                    roi_img, roi_valid, crop_xyxy = crop_target_roi(
                        src,
                        meta.get("bbox"),
                        meta.get("bbox_format", "cxcywh_norm"),
                        expand_ratio=float(args.roi_expand_ratio),
                        make_square=bool(args.roi_make_square),
                    )
                    src.close()
                    roi_pils.append(roi_img)
                    valid_roi_pending.append((item[0], item[1], meta, roi_valid, crop_xyxy))
                except Exception as exc:
                    failed += 1
                    print(f"[warn] failed to crop ROI for {item[0]}: {exc}", flush=True)
            if roi_pils:
                try:
                    roi_tokens = enc.encode_pooled_tokens(roi_pils, int(args.roi_token_count)).float().cpu()
                    for index, (_img_path, vroi_path, meta, roi_valid, crop_xyxy) in enumerate(valid_roi_pending):
                        payload = build_roi_cache_payload(
                            roi_tokens[index].cpu(),
                            roi_token_count=int(args.roi_token_count),
                            roi_expand_ratio=float(args.roi_expand_ratio),
                            roi_make_square=bool(args.roi_make_square),
                            bbox_format=str(meta.get("bbox_format", "cxcywh_norm")),
                            roi_valid=bool(roi_valid),
                            crop_xyxy=crop_xyxy,
                            source_bbox=meta.get("bbox"),
                            save_float32=bool(args.save_float32),
                        )
                        save_futures.add(save_pool.submit(save_multi_agent_roi_payload, vroi_path, payload))
                        generated += 1
                except Exception as exc:
                    failed += len(valid_roi_pending)
                    print(f"[warn] failed ROI batch starting at frame {start}: {exc}", flush=True)
                finally:
                    for pil in roi_pils:
                        pil.close()

            now = time.time()
            if checked % 256 == 0 or checked == len(frame_paths) or now - last_progress >= 30:
                elapsed = max(now - start_time, 1e-9)
                rate = checked / elapsed
                remaining = max(len(frame_paths) - checked, 0)
                eta_seconds = remaining / max(rate, 1e-9)
                print(
                    "progress "
                    f"checked={checked}/{len(frame_paths)} generated={generated} saved={saved} "
                    f"pending_saves={len(save_futures)} missing={skipped_missing} failed={failed} "
                    f"rate={rate:.2f} frames/s eta={eta_seconds/3600:.2f}h",
                    flush=True,
                )
                last_progress = now

        while save_futures:
            saved_done, save_failed = drain_completed_saves(save_futures, block=True)
            saved += saved_done
            failed += save_failed

    print(
        f"Completed precache: checked={checked} generated={generated} saved={saved} "
        f"missing={skipped_missing} failed={failed}",
        flush=True,
    )


if __name__ == '__main__':
    if "--multi_agent" in sys.argv:
        sys.argv.remove("--multi_agent")
        main_multi_agent_precache()
    else:
        main()
