#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""查看 .pt/.pth 文件结构。

这个脚本同时支持：
- 直接保存的 Tensor，例如 vision_cache 里的 *_vcoarse.pt。
- checkpoint dict，例如 train.py 保存的 model_epoch*.pt。
- 嵌套 dict / OrderedDict / list / tuple。

用法：
    python -m tools.view
    python -m tools.view /data/hdt/ntv_data/ckpt/ckpts_multi_agent/model_epoch03_step000232_final.pt
    python -m tools.view /data/hdt/ntv_data/data/.../frame_00001_vcoarse.pt --print-values
    python -m tools.view /data/hdt/ntv_data/ckpt/ckpts_multi_agent/model_epoch03_step000232_final.pt --max-depth 3 --max-items 20
"""

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch


DEFAULT_PATH = "/data/hdt/ntv_data/ckpt/ckpts_multi_agent/model_epoch03_step000232_final.pt"


def fmt_shape(obj: Any) -> str:
    """把 Tensor shape 转成短字符串。"""
    if hasattr(obj, "shape"):
        return str(tuple(obj.shape))
    return "-"


def indent(level: int) -> str:
    return "  " * level


def is_tensor(obj: Any) -> bool:
    return isinstance(obj, torch.Tensor)


def tensor_stats(tensor: torch.Tensor, sample_numel: int = 200000) -> str:
    """返回 Tensor 的数值统计。

    大 Tensor 不直接对全量做统计，而是取前 sample_numel 个元素，避免查看 1.4GB checkpoint 时太慢。
    """
    if tensor.numel() == 0:
        return "empty"
    if not (tensor.is_floating_point() or tensor.is_complex() or tensor.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool)):
        return "非数值张量"

    flat = tensor.detach().reshape(-1)
    if flat.numel() > sample_numel:
        flat = flat[:sample_numel]
        suffix = f" | 统计基于前 {sample_numel} 个元素"
    else:
        suffix = ""

    if tensor.is_complex():
        mag = flat.abs().float()
        return (
            f"abs_min={mag.min().item():.6g}, abs_max={mag.max().item():.6g}, "
            f"abs_mean={mag.mean().item():.6g}{suffix}"
        )

    data = flat.float()
    return (
        f"min={data.min().item():.6g}, max={data.max().item():.6g}, "
        f"mean={data.mean().item():.6g}, std={data.std(unbiased=False).item():.6g}{suffix}"
    )


def describe_tensor(name: str, tensor: torch.Tensor, level: int, print_values: bool) -> None:
    """打印 Tensor 的完整基础信息。"""
    p = indent(level)
    print(f"{p}{name}: Tensor")
    print(f"{p}  shape:  {tuple(tensor.shape)}")
    print(f"{p}  dtype:  {tensor.dtype}")
    print(f"{p}  device: {tensor.device}")
    print(f"{p}  numel:  {tensor.numel():,}")
    print(f"{p}  stats:  {tensor_stats(tensor)}")
    if print_values:
        print(f"{p}  values:")
        print(tensor)


def summarize_state_dict(state: Mapping[str, Any], level: int, max_items: int) -> None:
    """对 model_state/state_dict 做模型权重摘要。"""
    p = indent(level)
    tensor_items = [(k, v) for k, v in state.items() if is_tensor(v)]
    non_tensor_items = [(k, v) for k, v in state.items() if not is_tensor(v)]
    total_params = sum(v.numel() for _, v in tensor_items)
    trainable_like = sum(v.numel() for _, v in tensor_items if v.is_floating_point())

    print(f"{p}模型权重摘要:")
    print(f"{p}  tensor 数量: {len(tensor_items):,}")
    print(f"{p}  非 tensor 数量: {len(non_tensor_items):,}")
    print(f"{p}  参数/元素总数: {total_params:,}")
    print(f"{p}  浮点参数/元素数: {trainable_like:,}")

    dtype_counter = Counter(str(v.dtype) for _, v in tensor_items)
    shape_counter = Counter(tuple(v.shape) for _, v in tensor_items)
    prefix_counter = Counter(k.split(".", 1)[0] for k, _ in tensor_items)

    print(f"{p}  dtype 分布:")
    for key, val in dtype_counter.most_common():
        print(f"{p}    {key}: {val}")

    print(f"{p}  顶层模块分布:")
    for key, val in prefix_counter.most_common():
        print(f"{p}    {key}: {val}")

    print(f"{p}  最常见 shape:")
    for key, val in shape_counter.most_common(10):
        print(f"{p}    {key}: {val}")

    print(f"{p}  前 {min(max_items, len(tensor_items))} 个权重:")
    for k, v in tensor_items[:max_items]:
        print(f"{p}    {k}: shape={tuple(v.shape)}, dtype={v.dtype}, numel={v.numel():,}")


def summarize_optimizer_state(optim_state: Mapping[str, Any], level: int) -> None:
    """对 optim_state 做专门摘要。"""
    p = indent(level)
    print(f"{p}优化器状态摘要:")
    print(f"{p}  顶层键: {list(optim_state.keys())}")

    state = optim_state.get("state")
    param_groups = optim_state.get("param_groups")

    if isinstance(state, Mapping):
        print(f"{p}  state 参数条目数: {len(state):,}")
        tensor_count = 0
        tensor_numel = 0
        state_keys = Counter()
        for value in state.values():
            if isinstance(value, Mapping):
                state_keys.update(value.keys())
                for sub_value in value.values():
                    if is_tensor(sub_value):
                        tensor_count += 1
                        tensor_numel += sub_value.numel()
        print(f"{p}  state 内 tensor 数量: {tensor_count:,}")
        print(f"{p}  state 内 tensor 元素数: {tensor_numel:,}")
        print(f"{p}  state 字段分布: {dict(state_keys.most_common())}")

    if isinstance(param_groups, list):
        print(f"{p}  param_groups 数量: {len(param_groups)}")
        for i, group in enumerate(param_groups[:5]):
            if not isinstance(group, Mapping):
                continue
            keys = [k for k in group.keys() if k != "params"]
            n_params = len(group.get("params", [])) if isinstance(group.get("params"), list) else "?"
            print(f"{p}    group[{i}]: params={n_params}, keys={keys}")
            for key in keys:
                print(f"{p}      {key}: {group[key]}")


def describe_mapping(
    name: str,
    obj: Mapping[str, Any],
    level: int,
    max_depth: int,
    max_items: int,
    print_values: bool,
) -> None:
    """递归打印 dict/OrderedDict 信息。"""
    p = indent(level)
    print(f"{p}{name}: {type(obj).__name__}")
    print(f"{p}  键数量: {len(obj):,}")
    print(f"{p}  键列表: {list(obj.keys())[:max_items]}")
    if len(obj) > max_items:
        print(f"{p}  ... 还有 {len(obj) - max_items:,} 个键未展示")

    lower_name = name.lower()
    if "model_state" in lower_name or "state_dict" in lower_name:
        summarize_state_dict(obj, level + 1, max_items)
    elif "optim_state" in lower_name or "optimizer" in lower_name:
        summarize_optimizer_state(obj, level + 1)

    if level >= max_depth:
        print(f"{p}  达到 max_depth={max_depth}，停止递归展开。")
        return

    print(f"{p}  子项预览:")
    for i, (key, value) in enumerate(obj.items()):
        if i >= max_items:
            print(f"{p}  ... 省略 {len(obj) - max_items:,} 个子项")
            break
        describe_object(str(key), value, level + 1, max_depth, max_items, print_values)


def describe_sequence(
    name: str,
    obj: Iterable[Any],
    level: int,
    max_depth: int,
    max_items: int,
    print_values: bool,
) -> None:
    """递归打印 list/tuple 信息。"""
    p = indent(level)
    seq = list(obj)
    print(f"{p}{name}: {type(obj).__name__}")
    print(f"{p}  长度: {len(seq):,}")
    if level >= max_depth:
        print(f"{p}  达到 max_depth={max_depth}，停止递归展开。")
        return
    for i, value in enumerate(seq[:max_items]):
        describe_object(f"[{i}]", value, level + 1, max_depth, max_items, print_values)
    if len(seq) > max_items:
        print(f"{p}  ... 省略 {len(seq) - max_items:,} 个元素")


def describe_object(
    name: str,
    obj: Any,
    level: int,
    max_depth: int,
    max_items: int,
    print_values: bool,
) -> None:
    """根据对象类型分派打印逻辑。"""
    p = indent(level)
    if is_tensor(obj):
        describe_tensor(name, obj, level, print_values)
    elif isinstance(obj, (dict, OrderedDict)):
        describe_mapping(name, obj, level, max_depth, max_items, print_values)
    elif isinstance(obj, (list, tuple)):
        describe_sequence(name, obj, level, max_depth, max_items, print_values)
    else:
        value_repr = repr(obj)
        if len(value_repr) > 300:
            value_repr = value_repr[:300] + "..."
        print(f"{p}{name}: {type(obj).__name__} = {value_repr}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="查看 torch 保存的 Tensor/dict/checkpoint 文件结构。")
    parser.add_argument("path", nargs="?", default=DEFAULT_PATH, help=f"要查看的 .pt/.pth 文件，默认 {DEFAULT_PATH}")
    parser.add_argument("--max-depth", type=int, default=2, help="递归展开 dict/list 的最大深度。")
    parser.add_argument("--max-items", type=int, default=12, help="每层最多展示多少个键或元素。")
    parser.add_argument("--print-values", action="store_true", help="打印 Tensor 的具体值。大 Tensor 不建议打开。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")

    print(f"文件路径: {path}")
    print(f"文件大小: {path.stat().st_size / (1024 ** 2):.2f} MB")
    print("=" * 80)

    try:
        obj = torch.load(str(path), map_location="cpu")
    except Exception:
        # PyTorch 2.6+ 有时默认 weights_only=True；可信本地 checkpoint 可退回完整反序列化。
        obj = torch.load(str(path), map_location="cpu", weights_only=False)

    print(f"文件类型: {type(obj)}")
    print("=" * 80)
    describe_object("root", obj, level=0, max_depth=args.max_depth, max_items=args.max_items, print_values=args.print_values)


if __name__ == "__main__":
    main()
