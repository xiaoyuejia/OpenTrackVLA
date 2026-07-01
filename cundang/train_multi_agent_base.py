#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train the clean base multi-agent OpenTrackVLA model.

This script is intentionally separate from train.py. It trains only waypoint
regression for two agents and contains no grounding, bbox, visibility, relative
pose, or diffusion branches.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from cundang.data_multi_agent_base import (
    BaseMultiAgentDataConfig,
    BaseMultiAgentJsonDataset,
    collate_base_multi_agent,
)
from cundang.model_multi_agent_base import BaseMultiAgentModelConfig, BaseMultiAgentOpenTrackVLA


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def format_seconds(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


@dataclass
class BaseTrainConfig:
    train_json: str
    out_dir: str
    cache_root: Optional[str] = None
    llm_name: str = "Qwen/Qwen3-0.6B"
    vision_feat_dim: int = 1536
    history: int = 31
    n_waypoints: int = 10
    action_dims: int = 3
    epochs: int = 10
    batch_size: int = 2
    grad_accum_steps: int = 8
    lr: float = 2e-5
    weight_decay: float = 0.01
    beta_nav: float = 100.0
    drone_loss_weight: float = 5.0
    dog_loss_weight: float = 1.0
    alpha_xy: Optional[float] = 1.0
    use_tanh_actions: bool = False
    freeze_llm: bool = True
    num_workers: int = 4
    prefetch_factor: int = 2
    coarse_cache_size: int = 4096
    mixed_precision: str = "bf16"
    log_every: int = 20
    save_every: int = 1000
    max_ckpts: int = 3
    max_steps: int = 0
    seed: int = 42
    dry_run: bool = False
    ddp_find_unused_parameters: bool = False
    use_torch_ddp: bool = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ddp_info() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return distributed, rank, local_rank, world_size


def setup_ddp() -> tuple[bool, int, int, int, torch.device]:
    distributed, rank, local_rank, world_size = ddp_info()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if distributed and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    return distributed, rank, local_rank, world_size, device


def cleanup_ddp() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask.unsqueeze(-1).expand_as(pred)
    selected = (pred - target).pow(2)[expanded]
    return selected.float().mean() if selected.numel() else pred.new_tensor(0.0).float()


def normalize_xy_by_alpha(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha_task: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    if alpha_task is None or pred.size(-1) < 2:
        return pred, target
    alpha = alpha_task.to(device=pred.device, dtype=pred.dtype)
    pred_n = pred.clone()
    target_n = target.clone()
    pred_n[..., :2] = pred_n[..., :2] / alpha[..., :2].clamp_min(1e-6)
    target_n[..., :2] = target_n[..., :2] / alpha[..., :2].clamp_min(1e-6)
    return pred_n, target_n


def final_epe(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid_counts = mask.long().sum(dim=-1).clamp_min(1)
    last_idx = valid_counts - 1
    b = torch.arange(pred.size(0), device=pred.device).view(-1, 1).expand(-1, pred.size(1))
    a = torch.arange(pred.size(1), device=pred.device).view(1, -1).expand(pred.size(0), -1)
    pred_last = pred[b, a, last_idx]
    target_last = target[b, a, last_idx]
    return torch.linalg.norm(pred_last[..., :2] - target_last[..., :2], dim=-1).mean()


def build_model(cfg: BaseTrainConfig) -> BaseMultiAgentOpenTrackVLA:
    model_cfg = BaseMultiAgentModelConfig(
        llm_name=cfg.llm_name,
        freeze_llm=cfg.freeze_llm,
        n_waypoints=cfg.n_waypoints,
        action_dims=cfg.action_dims,
        use_tanh_actions=cfg.use_tanh_actions,
        alpha_xy=cfg.alpha_xy,
    )
    return BaseMultiAgentOpenTrackVLA(model_cfg, vision_feat_dim=cfg.vision_feat_dim)


def forward_loss(model: nn.Module, batch: Dict[str, Any], cfg: BaseTrainConfig, device: torch.device):
    out = model(
        coarse_tokens=batch["coarse_tokens"].to(device, non_blocking=True),
        coarse_tidx=batch["coarse_tidx"].to(device, non_blocking=True),
        fine_tokens=batch["fine_tokens"].to(device, non_blocking=True),
        fine_tidx=batch["fine_tidx"].to(device, non_blocking=True),
        instructions=batch["instruction"],
        return_dict=True,
    )
    pred = out["waypoints"]
    target = batch["waypoints"].to(device, non_blocking=True)
    mask = batch["valid_mask"].to(device, non_blocking=True).bool()
    base_model = model.module if isinstance(model, DDP) else model
    pred_n, target_n = normalize_xy_by_alpha(pred, target, getattr(base_model, "alpha_task", None))
    # Dataset/model convention: agent index 0 is drone, index 1 is robotdog.
    pred_drone, target_drone, mask_drone = pred_n[:, 0:1], target_n[:, 0:1], mask[:,0:1]
    pred_dog, target_dog, mask_dog = pred_n[:, 1:2], target_n[:, 1:2], mask[:,1:2]
    loss_nav_drone = masked_mse(pred_drone, target_drone, mask_drone)
    loss_nav_dog = masked_mse(pred_dog, target_dog, mask_dog)
    loss_nav = float(cfg.drone_loss_weight) * loss_nav_drone + float(cfg.dog_loss_weight) * loss_nav_dog
    loss =float(cfg.beta_nav) * loss_nav
    
    return loss, {
        "loss_nav": loss_nav.detach(),
        "loss_nav_dog": loss_nav_dog.detach(),
        "loss_nav_drone": loss_nav_drone.detach(),
        "final_epe": final_epe(pred.detach(), target, mask).detach(),
        "final_epe_dog": final_epe(pred[:, 1:2].detach(), target[:, 1:2], mask[:, 1:2]).detach(),
        "final_epe_drone": final_epe(pred[:, 0:1].detach(), target[:, 0:1], mask[:, 0:1]).detach(),
    }


def sync_trainable_gradients(model: nn.Module, world_size: int) -> None:
    if world_size <= 1 or not (dist.is_available() and dist.is_initialized()):
        return
    for param in model.parameters():
        if not param.requires_grad:
            continue
        device = param.device
        has_grad = torch.tensor(1 if param.grad is not None else 0, device=device, dtype=torch.int32)
        dist.all_reduce(has_grad, op=dist.ReduceOp.SUM)
        if int(has_grad.item()) == 0:
            continue
        grad = param.grad if param.grad is not None else torch.zeros_like(param)
        dist.all_reduce(grad, op=dist.ReduceOp.SUM)
        grad.div_(float(world_size))
        if param.grad is None:
            param.grad = grad


def save_checkpoint(path: Path, model: nn.Module, optim: torch.optim.Optimizer, cfg: BaseTrainConfig, epoch: int, step: int) -> None:
    base_model = model.module if isinstance(model, DDP) else model
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "model_state": base_model.state_dict(),
            "optim_state": optim.state_dict(),
            "config": {
                **asdict(cfg),
                "model_type": "base_multi_agent_concat",
                "use_grounding": False,
                "use_bbox_tokens": False,
                "use_anchor_diffusion": False,
            },
        },
        path,
    )


def _run_text_command(cmd: list[str], cwd: Optional[Path] = None) -> Optional[str]:
    try:
        proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True, capture_output=True, check=False)
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _path_count(path: Path, pattern: str) -> Optional[int]:
    try:
        if path.is_file():
            return 1 if path.match(pattern) else 0
        if path.is_dir():
            return sum(1 for _ in path.rglob(pattern))
    except Exception:
        return None
    return None


def write_training_metadata(
    out_dir: Path,
    cfg: BaseTrainConfig,
    ds: BaseMultiAgentJsonDataset,
    *,
    distributed: bool,
    rank: int,
    local_rank: int,
    world_size: int,
    device: torch.device,
    steps_per_epoch: Optional[int] = None,
    total_steps: Optional[int] = None,
) -> None:
    effective_batch = int(cfg.batch_size) * int(world_size) * max(1, int(cfg.grad_accum_steps))
    repo_root = Path(__file__).resolve().parent
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    metadata: Dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "command": " ".join(sys.argv),
        "config": asdict(cfg),
        "derived": {
            "effective_batch": effective_batch,
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
            "world_size": world_size,
            "distributed": distributed,
            "rank": rank,
            "local_rank": local_rank,
        },
        "data": {
            "samples": len(ds),
            "train_json": str(ds.train_path),
            "data_root": str(ds.data_root),
            "cache_root": str(ds.cache_root),
            "jsonl_files": _path_count(ds.train_path, "*.jsonl"),
        },
        "runtime": {
            "python": sys.version.replace("\n", " "),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device": str(device),
            "cuda_visible_devices": cuda_visible,
        },
        "git": {
            "commit": _run_text_command(["git", "rev-parse", "HEAD"], repo_root),
            "branch": _run_text_command(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo_root),
            "status_short": _run_text_command(["git", "status", "--short"], repo_root),
        },
    }
    if torch.cuda.is_available():
        try:
            metadata["runtime"]["gpu_name"] = torch.cuda.get_device_name(device)
            metadata["runtime"]["gpu_count_visible"] = torch.cuda.device_count()
        except Exception:
            pass

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "train_config.json"
    txt_path = out_dir / "train_config.txt"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    lines = [
        "Base multi-agent training config",
        f"created_at: {metadata['created_at']}",
        f"command: {metadata['command']}",
        f"train_json: {metadata['data']['train_json']}",
        f"data_root: {metadata['data']['data_root']}",
        f"cache_root: {metadata['data']['cache_root']}",
        f"samples: {metadata['data']['samples']}",
        f"jsonl_files: {metadata['data']['jsonl_files']}",
        f"llm_name: {cfg.llm_name}",
        f"epochs: {cfg.epochs}",
        f"batch_size_per_gpu: {cfg.batch_size}",
        f"world_size: {world_size}",
        f"grad_accum_steps: {cfg.grad_accum_steps}",
        f"effective_batch: {effective_batch}",
        f"lr: {cfg.lr}",
        f"weight_decay: {cfg.weight_decay}",
        f"beta_nav: {cfg.beta_nav}",
        f"drone_loss_weight: {cfg.drone_loss_weight}",
        f"dog_loss_weight: {cfg.dog_loss_weight}",
        f"alpha_xy: {cfg.alpha_xy}",
        f"use_tanh_actions: {cfg.use_tanh_actions}",
        f"mixed_precision: {cfg.mixed_precision}",
        f"num_workers: {cfg.num_workers}",
        f"prefetch_factor: {cfg.prefetch_factor}",
        f"steps_per_epoch: {steps_per_epoch}",
        f"total_steps: {total_steps}",
        f"cuda_visible_devices: {cuda_visible}",
        f"device: {device}",
        f"torch: {torch.__version__}",
        f"git_commit: {metadata['git']['commit']}",
        f"git_branch: {metadata['git']['branch']}",
    ]
    with txt_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def prune_checkpoints(out_dir: Path, max_ckpts: int) -> None:
    if max_ckpts <= 0:
        return
    ckpts = sorted(out_dir.glob("model_epoch*_step*.pt"), key=lambda p: p.stat().st_mtime)
    while len(ckpts) > max_ckpts:
        ckpts.pop(0).unlink(missing_ok=True)


BASE_MULTI_LOG_FIELDS = [
    "epoch",
    "batch",
    "step",
    "lr",
    "loss",
    "loss_ema",
    "loss_nav",
    "loss_nav_drone",
    "loss_nav_dog",
    "final_epe",
    "final_epe_drone",
    "final_epe_dog",
    "grad_norm",
    "step_time",
    "throughput",
    "epoch_eta",
    "total_eta",
    "progress_pct",
]


def write_csv_header(csv_path: Path) -> None:
    if not csv_path.exists():
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(BASE_MULTI_LOG_FIELDS)


def train(cfg: BaseTrainConfig) -> None:
    distributed, rank, local_rank, world_size, device = setup_ddp()
    is_main = rank == 0
    # Keep model initialization identical across ranks. Data shuffling is handled
    # by DistributedSampler; rank-specific model seeds would break manual DDP.
    set_seed(cfg.seed)

    ds = BaseMultiAgentJsonDataset(
        BaseMultiAgentDataConfig(
            train_json=cfg.train_json,
            cache_root=cfg.cache_root,
            history=cfg.history,
            n_waypoints=cfg.n_waypoints,
            action_dims=cfg.action_dims,
            coarse_cache_size=cfg.coarse_cache_size,
        )
    )
    sampler = DistributedSampler(ds, shuffle=True, seed=cfg.seed) if distributed else None
    loader_kwargs = dict(
        dataset=ds,
        batch_size=cfg.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_base_multi_agent,
        drop_last=False,
    )
    if cfg.num_workers > 0:
        loader_kwargs["prefetch_factor"] = cfg.prefetch_factor
        loader_kwargs["persistent_workers"] = True
    loader = DataLoader(**loader_kwargs)

    if is_main:
        print(f"[data] samples={len(ds)} root={ds.data_root} cache={ds.cache_root}", flush=True)
        sample = ds[0]
        print(
            "[data] sample shapes "
            f"coarse={tuple(sample['coarse_tokens'].shape)} fine={tuple(sample['fine_tokens'].shape)} "
            f"waypoints={tuple(sample['waypoints'].shape)}",
            flush=True,
        )

    print(f"[rank {rank}] building model", flush=True)
    model = build_model(cfg)
    print(f"[rank {rank}] moving model to {device}", flush=True)
    model = model.to(device)
    print(f"[rank {rank}] model moved to {device}", flush=True)
    use_torch_ddp = bool(distributed and cfg.use_torch_ddp)
    if use_torch_ddp:
        print(f"[rank {rank}] wrapping model with torch DDP", flush=True)
        model = DDP(
            model,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            find_unused_parameters=cfg.ddp_find_unused_parameters,
            broadcast_buffers=False,
        )
        print(f"[rank {rank}] torch DDP ready", flush=True)
    elif distributed:
        print(f"[rank {rank}] using manual gradient all-reduce for trainable params", flush=True)

    optim = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=cfg.lr, weight_decay=cfg.weight_decay)
    amp_enabled = cfg.mixed_precision.lower() != "none" and device.type == "cuda"
    amp_dtype = torch.bfloat16 if cfg.mixed_precision.lower() == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_enabled and amp_dtype == torch.float16))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    out_dir = Path(cfg.out_dir)
    csv_path = out_dir / "train_log.csv"
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_csv_header(csv_path)
        write_training_metadata(
            out_dir,
            cfg,
            ds,
            distributed=distributed,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            device=device,
        )
        print(
            f"[config] wrote {out_dir / 'train_config.json'} and {out_dir / 'train_config.txt'}",
            flush=True,
        )

    if cfg.dry_run:
        batch = next(iter(loader))
        model.eval()
        with torch.inference_mode():
            loss, metrics = forward_loss(model, batch, cfg, device)
        if is_main:
            print(
                f"[dry-run] loss={float(loss):.6f} nav={metrics['loss_nav'].item():.6f} "
                f"nav_drone={metrics['loss_nav_drone'].item():.6f} nav_dog={metrics['loss_nav_dog'].item():.6f} "
                f"final_epe={metrics['final_epe'].item():.6f} "
                f"epe_drone={metrics['final_epe_drone'].item():.6f} epe_dog={metrics['final_epe_dog'].item():.6f}",
                flush=True,
            )
        cleanup_ddp()
        return

    accum = max(1, int(cfg.grad_accum_steps))
    steps_per_epoch = max(1, math.ceil(len(loader) / accum))
    total_steps = max(1, int(cfg.epochs) * steps_per_epoch)
    step = 0
    ema_loss: Optional[float] = None
    train_start = time.time()
    last_log = train_start
    last_log_step = 0
    if is_main:
        write_training_metadata(
            out_dir,
            cfg,
            ds,
            distributed=distributed,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            device=device,
            steps_per_epoch=steps_per_epoch,
            total_steps=total_steps,
        )
        effective_batch = cfg.batch_size * world_size * accum
        print(
            f"[INIT][BASE_MULTI] samples={len(ds)} batches={len(loader)} batch_per_gpu={cfg.batch_size} "
            f"world_size={world_size} grad_accum_steps={accum} effective_batch={effective_batch} "
            f"steps_per_epoch={steps_per_epoch} total_steps={total_steps} mixed_precision={cfg.mixed_precision}",
            flush=True,
        )
    stop_training = False
    for epoch in range(cfg.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        optim.zero_grad(set_to_none=True)
        epoch_start = time.time()
        for batch_idx, batch in enumerate(loader):
            do_step = ((batch_idx + 1) % accum == 0) or (batch_idx + 1 == len(loader))
            sync_ctx = model.no_sync() if use_torch_ddp and hasattr(model, "no_sync") and not do_step else nullcontext()
            amp_ctx = (
                torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled)
                if device.type == "cuda"
                else nullcontext()
            )
            with sync_ctx:
                with amp_ctx:
                    loss, metrics = forward_loss(model, batch, cfg, device)
                scaler.scale(loss / accum).backward()
            grad_norm = torch.tensor(float("nan"), device=device)
            if do_step:
                if scaler.is_enabled():
                    scaler.unscale_(optim)
                if distributed and not use_torch_ddp:
                    sync_trainable_gradients(model, world_size)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                step += 1

                if is_main and (step % cfg.log_every == 0 or step == 1):
                    now = time.time()
                    steps_since_log = max(1, step - last_log_step)
                    elapsed_since_log = now - last_log
                    step_time = elapsed_since_log / steps_since_log
                    last_log = now
                    last_log_step = step
                    loss_val = float(loss.item())
                    ema_loss = loss_val if ema_loss is None else (0.98 * ema_loss + 0.02 * loss_val)
                    epoch_eta = format_seconds((now - epoch_start) / max(1, batch_idx + 1) * max(0, len(loader) - batch_idx - 1))
                    total_eta = format_seconds((now - train_start) / max(1, step) * max(0, total_steps - step))
                    progress_pct = 100.0 * step / max(1, total_steps)
                    batch_size_now = 0
                    try:
                        batch_size_now = int(batch["waypoints"].size(0)) * world_size
                    except Exception:
                        batch_size_now = cfg.batch_size * world_size
                    throughput = batch_size_now * steps_since_log * accum / max(elapsed_since_log, 1e-9)
                    mem_alloc_gb = mem_reserved_gb = mem_peak_gb = 0.0
                    if torch.cuda.is_available():
                        mem_alloc_gb = torch.cuda.memory_allocated(device) / (1024 ** 3)
                        mem_reserved_gb = torch.cuda.memory_reserved(device) / (1024 ** 3)
                        mem_peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                    print(
                        f"[TRAIN][BASE_MULTI] epoch={epoch + 1}/{cfg.epochs} "
                        f"batch={batch_idx + 1}/{len(loader)} step={step}/{total_steps} "
                        f"eta_epoch={epoch_eta} eta_total={total_eta} progress={progress_pct:.1f}% "
                        f"loss={loss_val:.5f} ema={ema_loss:.5f} "
                        f"nav={metrics['loss_nav'].item():.5f} "
                        f"nav_drone={metrics['loss_nav_drone'].item():.5f} nav_dog={metrics['loss_nav_dog'].item():.5f} "
                        f"final_epe={metrics['final_epe'].item():.4f} "
                        f"epe_drone={metrics['final_epe_drone'].item():.4f} epe_dog={metrics['final_epe_dog'].item():.4f} "
                        f"grad={float(grad_norm):.3f} dt={elapsed_since_log:.2f}s "
                        f"step_time={step_time:.2f}s throughput={throughput:.2f} samples/s "
                        f"mem={mem_alloc_gb:.1f}G/{mem_reserved_gb:.1f}G peak={mem_peak_gb:.1f}G",
                        flush=True,
                    )
                    with csv_path.open("a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow(
                            [
                                epoch,
                                batch_idx + 1,
                                step,
                                cfg.lr,
                                loss_val,
                                float(ema_loss),
                                float(metrics["loss_nav"].item()),
                                float(metrics["loss_nav_drone"].item()),
                                float(metrics["loss_nav_dog"].item()),
                                float(metrics["final_epe"].item()),
                                float(metrics["final_epe_drone"].item()),
                                float(metrics["final_epe_dog"].item()),
                                float(grad_norm),
                                step_time,
                                throughput,
                                epoch_eta,
                                total_eta,
                                progress_pct,
                            ]
                        )

                if is_main and cfg.save_every > 0 and step % cfg.save_every == 0:
                    ckpt = out_dir / f"model_epoch{epoch:02d}_step{step:06d}.pt"
                    save_checkpoint(ckpt, model, optim, cfg, epoch, step)
                    prune_checkpoints(out_dir, cfg.max_ckpts)

                if cfg.max_steps > 0 and step >= cfg.max_steps:
                    stop_training = True
                    break

        if is_main:
            ckpt = out_dir / f"model_epoch{epoch:02d}_step{step:06d}.pt"
            save_checkpoint(ckpt, model, optim, cfg, epoch, step)
            prune_checkpoints(out_dir, cfg.max_ckpts)
        if stop_training:
            break

    if is_main:
        print(f"[train] finished step={step} out={out_dir}", flush=True)
    cleanup_ddp()


def parse_args() -> BaseTrainConfig:
    ap = argparse.ArgumentParser(description="Train clean base multi-agent OpenTrackVLA.")
    ap.add_argument("--train_json", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cache_root", default=None)
    ap.add_argument("--llm_name", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--vision_feat_dim", type=int, default=1536)
    ap.add_argument("--history", type=int, default=31)
    ap.add_argument("--n_waypoints", type=int, default=10)
    ap.add_argument("--action_dims", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum_steps", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--beta_nav", type=float, default=100.0)
    ap.add_argument("--drone_loss_weight", type=float, default=5.0)
    ap.add_argument("--dog_loss_weight", type=float, default=1.0)
    ap.add_argument("--alpha_xy", type=float, default=1.0)
    ap.add_argument(
        "--tanh-actions",
        dest="use_tanh_actions",
        action="store_true",
        default=False,
        help="Enable tanh action cap before alpha scaling. Default is disabled to match single-agent training.",
    )
    ap.add_argument(
        "--no-tanh-actions",
        dest="use_tanh_actions",
        action="store_false",
        help="Disable tanh action cap before alpha scaling. This is the default and matches single-agent training.",
    )
    ap.add_argument("--unfreeze-llm", action="store_true")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--prefetch_factor", type=int, default=2)
    ap.add_argument("--coarse_cache_size", type=int, default=4096)
    ap.add_argument("--mixed_precision", choices=("bf16", "fp16", "none"), default="bf16")
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--max_ckpts", type=int, default=3)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--ddp-find-unused-parameters", action="store_true")
    ap.add_argument(
        "--torch-ddp",
        dest="use_torch_ddp",
        action="store_true",
        default=False,
        help="Use torch DistributedDataParallel. Default is manual gradient all-reduce to avoid syncing frozen LLM weights.",
    )
    args = ap.parse_args()
    return BaseTrainConfig(
        train_json=args.train_json,
        out_dir=args.out_dir,
        cache_root=args.cache_root,
        llm_name=args.llm_name,
        vision_feat_dim=args.vision_feat_dim,
        history=args.history,
        n_waypoints=args.n_waypoints,
        action_dims=args.action_dims,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        beta_nav=args.beta_nav,
        drone_loss_weight=args.drone_loss_weight,
        dog_loss_weight=args.dog_loss_weight,
        alpha_xy=args.alpha_xy,
        use_tanh_actions=args.use_tanh_actions,
        freeze_llm=not args.unfreeze_llm,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        coarse_cache_size=args.coarse_cache_size,
        mixed_precision=args.mixed_precision,
        log_every=args.log_every,
        save_every=args.save_every,
        max_ckpts=args.max_ckpts,
        max_steps=args.max_steps,
        seed=args.seed,
        dry_run=args.dry_run,
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
        use_torch_ddp=args.use_torch_ddp,
    )


if __name__ == "__main__":
    train(parse_args())
