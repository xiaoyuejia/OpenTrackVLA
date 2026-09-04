"""Analyze GT target-box distributions for the air-ground tracking dataset.

The action model receives detector boxes at inference, but GT boxes are the
stable source for defining training-time sample weights.  This script scans the
episode manifests without loading images and relates normalized GT boxes to
target distance and recorded future waypoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import numpy as np


AGENTS = ("drone", "robotdog")
COLUMNS = (
    "cx",
    "cy",
    "w",
    "h",
    "area",
    "scale",
    "center_offset_05",
    "edge_margin",
    "target_distance",
    "first_xy",
    "final_xy",
    "final_forward",
    "final_lateral",
    "final_yaw",
    "target_visible",
    "target_centered",
)
COL = {name: index for index, name in enumerate(COLUMNS)}
QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
FOLLOW_BANDS = {
    "drone": {"min": 1.0, "max": 6.5, "lost": 9.0},
    "robotdog": {"min": 1.0, "max": 8.0, "lost": 10.0},
}


def _valid_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        cx, cy, width, height = (float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in (cx, cy, width, height)):
        return None
    if width <= 0.0 or height <= 0.0:
        return None
    return cx, cy, width, height


def _future_motion(agent: dict[str, Any]) -> tuple[float, float, float, float, float]:
    waypoints = agent.get("waypoints") or agent.get("trajectory") or []
    valid_mask = agent.get("valid_mask") or [True] * len(waypoints)
    valid_indices = [
        index
        for index, waypoint in enumerate(waypoints)
        if index < len(valid_mask)
        and bool(valid_mask[index])
        and isinstance(waypoint, (list, tuple))
        and len(waypoint) >= 3
    ]
    if not valid_indices:
        return (math.nan,) * 5
    first = np.asarray(waypoints[valid_indices[0]][:3], dtype=np.float64)
    final = np.asarray(waypoints[valid_indices[-1]][:3], dtype=np.float64)
    if not np.isfinite(first).all() or not np.isfinite(final).all():
        return (math.nan,) * 5
    return (
        float(np.linalg.norm(first[:2])),
        float(np.linalg.norm(final[:2])),
        float(final[0]),
        float(final[1]),
        float(final[2]),
    )


def _scan_episode(task: tuple[str, str, str]) -> tuple[str, str, dict[str, Any]]:
    split, scene, jsonl_path = task
    values: dict[str, list[list[float]]] = {agent: [] for agent in AGENTS}
    counts = {
        agent: {"samples": 0, "bbox_valid": 0, "visible": 0, "visible_bbox_valid": 0}
        for agent in AGENTS
    }
    with Path(jsonl_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            sample = json.loads(line)
            agents = sample.get("agents") or {}
            for agent_name in AGENTS:
                record = agents.get(agent_name) or {}
                counts[agent_name]["samples"] += 1
                visible = bool(record.get("target_visible", False))
                counts[agent_name]["visible"] += int(visible)
                bbox = _valid_bbox(record.get("bbox"))
                if bbox is None:
                    continue
                counts[agent_name]["bbox_valid"] += 1
                counts[agent_name]["visible_bbox_valid"] += int(visible)
                cx, cy, width, height = bbox
                x1, x2 = cx - width / 2.0, cx + width / 2.0
                y1, y2 = cy - height / 2.0, cy + height / 2.0
                distance = record.get("target_distance", math.nan)
                try:
                    distance = float(distance)
                except (TypeError, ValueError):
                    distance = math.nan
                first_xy, final_xy, final_forward, final_lateral, final_yaw = _future_motion(record)
                values[agent_name].append(
                    [
                        cx,
                        cy,
                        width,
                        height,
                        width * height,
                        math.sqrt(width * height),
                        math.hypot(cx - 0.5, cy - 0.5),
                        min(x1, y1, 1.0 - x2, 1.0 - y2),
                        distance,
                        first_xy,
                        final_xy,
                        final_forward,
                        final_lateral,
                        final_yaw,
                        float(visible),
                        float(bool(record.get("target_centered", False))),
                    ]
                )
    arrays = {
        agent: np.asarray(rows, dtype=np.float32).reshape(-1, len(COLUMNS))
        for agent, rows in values.items()
    }
    return split, scene, {"arrays": arrays, "counts": counts}


def _finite(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def _quantile_dict(values: np.ndarray) -> dict[str, float | None]:
    values = _finite(np.asarray(values, dtype=np.float64))
    if values.size == 0:
        return {f"p{int(q * 100):02d}": None for q in QUANTILES}
    result = np.quantile(values, QUANTILES)
    return {f"p{int(q * 100):02d}": float(value) for q, value in zip(QUANTILES, result)}


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3:
        return None
    x, y = left[valid].astype(np.float64), right[valid].astype(np.float64)
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(left: np.ndarray, right: np.ndarray, max_points: int = 250_000) -> float | None:
    valid = np.flatnonzero(np.isfinite(left) & np.isfinite(right))
    if valid.size < 3:
        return None
    if valid.size > max_points:
        valid = valid[np.linspace(0, valid.size - 1, max_points, dtype=np.int64)]
    from scipy.stats import spearmanr

    result = spearmanr(left[valid], right[valid]).statistic
    return float(result) if math.isfinite(float(result)) else None


def _best_far_threshold(scale: np.ndarray, distance: np.ndarray, maximum: float) -> dict[str, float | int | None]:
    valid = np.isfinite(scale) & np.isfinite(distance) & (distance >= 1.0)
    scores = scale[valid].astype(np.float64)
    far = distance[valid] > maximum
    positives = int(far.sum())
    negatives = int((~far).sum())
    if positives == 0 or negatives == 0:
        return {"threshold": None, "balanced_accuracy": None, "tpr": None, "fpr": None,
                "positives": positives, "negatives": negatives}
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_far = far[order]
    tp = np.cumsum(sorted_far)
    fp = np.cumsum(~sorted_far)
    tpr = tp / positives
    fpr = fp / negatives
    candidates = np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    objective = np.where(candidates, tpr - fpr, -np.inf)
    best = int(np.argmax(objective))
    return {
        "threshold": float(sorted_scores[best]),
        "balanced_accuracy": float((tpr[best] + (1.0 - fpr[best])) / 2.0),
        "tpr": float(tpr[best]),
        "fpr": float(fpr[best]),
        "positives": positives,
        "negatives": negatives,
    }


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _format_optional(value: Any, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _create_plots(output_dir: Path, train_arrays: dict[str, np.ndarray]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for agent, array in train_arrays.items():
        visible = array[:, COL["target_visible"]] > 0.5
        data = array[visible]
        if data.size == 0:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].hist(data[:, COL["scale"]], bins=100, color="#3478b8", alpha=0.9)
        axes[0].set(title=f"{agent}: GT bbox scale", xlabel="sqrt(w*h), normalized", ylabel="frames")
        axes[1].hexbin(
            data[:, COL["cx"]], data[:, COL["cy"]], gridsize=60,
            extent=(0, 1, 0, 1), bins="log", mincnt=1, cmap="viridis",
        )
        axes[1].invert_yaxis()
        axes[1].set(title=f"{agent}: GT bbox center", xlabel="cx", ylabel="cy")
        fig.tight_layout()
        fig.savefig(plot_dir / f"{agent}_bbox_distribution.png", dpi=180)
        plt.close(fig)

        finite = np.isfinite(data[:, COL["target_distance"]])
        scatter = data[finite]
        if scatter.shape[0] > 300_000:
            indices = np.linspace(0, scatter.shape[0] - 1, 300_000, dtype=np.int64)
            scatter = scatter[indices]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].hexbin(
            scatter[:, COL["scale"]], scatter[:, COL["target_distance"]],
            gridsize=70, bins="log", mincnt=1, cmap="magma",
        )
        axes[0].set(title=f"{agent}: bbox scale vs target distance", xlabel="sqrt(w*h)", ylabel="distance (m)")
        axes[1].hexbin(
            scatter[:, COL["scale"]], scatter[:, COL["final_xy"]],
            gridsize=70, bins="log", mincnt=1, cmap="plasma",
        )
        axes[1].set(title=f"{agent}: bbox scale vs GT final displacement", xlabel="sqrt(w*h)", ylabel="final XY (m)")
        fig.tight_layout()
        fig.savefig(plot_dir / f"{agent}_scale_relationships.png", dpi=180)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/data/hdt/ntv_data/data/data7_8_camera_m40_pose_fixed_dt_exact_bbox_global_base_split_70_30"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("offline_detection_segmentation/outputs/bbox_distribution_data7_8"),
    )
    parser.add_argument("--workers", type=int, default=min(16, max(1, mp.cpu_count() // 2)))
    parser.add_argument("--max-episodes-per-split", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks: list[tuple[str, str, str]] = []
    manifest_counts: dict[str, dict[str, int]] = {}
    for split in ("train", "val"):
        manifest_path = args.dataset_root / f"{split}_episodes.json"
        episodes = json.loads(manifest_path.read_text(encoding="utf-8"))
        if args.max_episodes_per_split is not None:
            episodes = episodes[: args.max_episodes_per_split]
        manifest_counts[split] = {
            "episodes": len(episodes),
            "declared_samples": int(sum(int(item.get("samples", 0)) for item in episodes)),
        }
        tasks.extend((split, str(item["scene"]), str(item["jsonl"])) for item in episodes)

    split_parts: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    scene_parts: dict[tuple[str, str, str], list[np.ndarray]] = defaultdict(list)
    counts: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    completed = 0
    print(f"[bbox-stats] episodes={len(tasks)} workers={args.workers}", flush=True)
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        for split, scene, result in executor.map(_scan_episode, tasks, chunksize=1):
            completed += 1
            for agent in AGENTS:
                array = result["arrays"][agent]
                if array.size:
                    split_parts[(split, agent)].append(array)
                    scene_parts[(split, scene, agent)].append(array)
                for key, value in result["counts"][agent].items():
                    counts[(split, agent)][key] += int(value)
            if completed % 100 == 0 or completed == len(tasks):
                print(f"[bbox-stats] scanned={completed}/{len(tasks)}", flush=True)

    arrays = {
        key: np.concatenate(parts, axis=0) if parts else np.empty((0, len(COLUMNS)), np.float32)
        for key, parts in split_parts.items()
    }
    train_arrays = {agent: arrays[("train", agent)] for agent in AGENTS}

    summary: dict[str, Any] = {
        "dataset_root": str(args.dataset_root.resolve()),
        "source": "GT bbox from agents.<agent>.bbox (cxcywh_norm)",
        "columns": list(COLUMNS),
        "follow_distance_bands_m": FOLLOW_BANDS,
        "manifests": manifest_counts,
        "splits": {},
        "recommended": {},
    }
    quantile_rows: list[dict[str, Any]] = []
    correlation_rows: list[dict[str, Any]] = []
    distance_rows: list[dict[str, Any]] = []
    size_bin_rows: list[dict[str, Any]] = []
    position_rows: list[dict[str, Any]] = []

    for split in ("train", "val"):
        summary["splits"][split] = {}
        for agent in AGENTS:
            array = arrays[(split, agent)]
            visible = array[:, COL["target_visible"]] > 0.5
            data = array[visible]
            agent_counts = dict(counts[(split, agent)])
            agent_counts["analyzed_visible_valid_bbox"] = int(data.shape[0])
            summary["splits"][split][agent] = {"counts": agent_counts, "quantiles": {}}
            for metric in ("cx", "cy", "w", "h", "area", "scale", "center_offset_05", "edge_margin", "target_distance", "first_xy", "final_xy", "final_forward", "final_lateral", "final_yaw"):
                quantiles = _quantile_dict(data[:, COL[metric]])
                summary["splits"][split][agent]["quantiles"][metric] = quantiles
                quantile_rows.append({"split": split, "agent": agent, "metric": metric, **quantiles})

            pairs = (
                ("scale", "target_distance"),
                ("scale", "final_xy"),
                ("target_distance", "final_xy"),
                ("center_offset_05", "final_xy"),
                ("cx", "final_yaw"),
            )
            center_ref = float(np.nanmedian(data[:, COL["cx"]])) if data.size else math.nan
            center_abs = np.abs(data[:, COL["cx"]] - center_ref) if data.size else np.asarray([])
            yaw_abs = np.abs(data[:, COL["final_yaw"]]) if data.size else np.asarray([])
            for left_name, right_name in pairs:
                left, right = data[:, COL[left_name]], data[:, COL[right_name]]
                correlation_rows.append({
                    "split": split,
                    "agent": agent,
                    "left": left_name,
                    "right": right_name,
                    "pearson": _pearson(left, right),
                    "spearman": _spearman(left, right),
                    "count": int((np.isfinite(left) & np.isfinite(right)).sum()),
                })
            correlation_rows.append({
                "split": split,
                "agent": agent,
                "left": "abs(cx-median_cx)",
                "right": "abs(final_yaw)",
                "pearson": _pearson(center_abs, yaw_abs),
                "spearman": _spearman(center_abs, yaw_abs),
                "count": int((np.isfinite(center_abs) & np.isfinite(yaw_abs)).sum()),
            })

            band = FOLLOW_BANDS[agent]
            distance = data[:, COL["target_distance"]]
            band_masks = {
                "too_close": distance < band["min"],
                "normal": (distance >= band["min"]) & (distance <= band["max"]),
                "over_max": (distance > band["max"]) & (distance <= band["lost"]),
                "lost": distance > band["lost"],
            }
            for name, mask in band_masks.items():
                subset = data[mask & np.isfinite(distance)]
                distance_rows.append({
                    "split": split,
                    "agent": agent,
                    "distance_band": name,
                    "count": int(subset.shape[0]),
                    "fraction": float(subset.shape[0] / max(1, np.isfinite(distance).sum())),
                    "distance_p50": float(np.nanmedian(subset[:, COL["target_distance"]])) if subset.size else None,
                    "scale_p10": _quantile_dict(subset[:, COL["scale"]])["p10"],
                    "scale_p50": _quantile_dict(subset[:, COL["scale"]])["p50"],
                    "scale_p90": _quantile_dict(subset[:, COL["scale"]])["p90"],
                    "final_xy_p50": _quantile_dict(subset[:, COL["final_xy"]])["p50"],
                    "final_xy_mean": float(np.nanmean(subset[:, COL["final_xy"]])) if subset.size else None,
                })

            scale_edges = np.quantile(data[:, COL["scale"]], np.linspace(0.0, 1.0, 6)) if data.size else np.zeros(6)
            for index in range(5):
                lower, upper = scale_edges[index], scale_edges[index + 1]
                mask = (data[:, COL["scale"]] >= lower) & (
                    data[:, COL["scale"]] <= upper if index == 4 else data[:, COL["scale"]] < upper
                )
                subset = data[mask]
                size_bin_rows.append({
                    "split": split,
                    "agent": agent,
                    "size_quintile": index + 1,
                    "scale_min": float(lower),
                    "scale_max": float(upper),
                    "count": int(subset.shape[0]),
                    "distance_mean": float(np.nanmean(subset[:, COL["target_distance"]])),
                    "distance_p50": float(np.nanmedian(subset[:, COL["target_distance"]])),
                    "final_xy_mean": float(np.nanmean(subset[:, COL["final_xy"]])),
                    "final_xy_p50": float(np.nanmedian(subset[:, COL["final_xy"]])),
                    "final_forward_mean": float(np.nanmean(subset[:, COL["final_forward"]])),
                })

            for name, mask in {
                "left": data[:, COL["cx"]] < 0.4,
                "center": (data[:, COL["cx"]] >= 0.4) & (data[:, COL["cx"]] <= 0.6),
                "right": data[:, COL["cx"]] > 0.6,
            }.items():
                subset = data[mask]
                position_rows.append({
                    "split": split,
                    "agent": agent,
                    "position_bin": name,
                    "count": int(subset.shape[0]),
                    "fraction": float(subset.shape[0] / max(1, data.shape[0])),
                    "cx_p50": float(np.nanmedian(subset[:, COL["cx"]])) if subset.size else None,
                    "cy_p50": float(np.nanmedian(subset[:, COL["cy"]])) if subset.size else None,
                    "final_yaw_mean": float(np.nanmean(subset[:, COL["final_yaw"]])) if subset.size else None,
                    "final_yaw_p50": float(np.nanmedian(subset[:, COL["final_yaw"]])) if subset.size else None,
                })

            if split == "train" and data.size:
                quantiles = summary["splits"][split][agent]["quantiles"]
                threshold = _best_far_threshold(
                    data[:, COL["scale"]], data[:, COL["target_distance"]], band["max"]
                )
                summary["recommended"][agent] = {
                    "center_ref": [quantiles["cx"]["p50"], quantiles["cy"]["p50"]],
                    "center_safe_p10_p90": {
                        "cx": [quantiles["cx"]["p10"], quantiles["cx"]["p90"]],
                        "cy": [quantiles["cy"]["p10"], quantiles["cy"]["p90"]],
                    },
                    "scale_distribution_far_p10": quantiles["scale"]["p10"],
                    "scale_distribution_normal_p50": quantiles["scale"]["p50"],
                    "distance_calibrated_far_threshold": threshold,
                    "note": "Use continuous weights; do not turn these values into a hard runtime gate.",
                }

    scene_rows: list[dict[str, Any]] = []
    for (split, scene, agent), parts in sorted(scene_parts.items()):
        data = np.concatenate(parts, axis=0)
        data = data[data[:, COL["target_visible"]] > 0.5]
        scene_rows.append({
            "split": split,
            "scene": scene,
            "agent": agent,
            "count": int(data.shape[0]),
            "cx_p50": _quantile_dict(data[:, COL["cx"]])["p50"],
            "cy_p50": _quantile_dict(data[:, COL["cy"]])["p50"],
            "scale_p10": _quantile_dict(data[:, COL["scale"]])["p10"],
            "scale_p50": _quantile_dict(data[:, COL["scale"]])["p50"],
            "scale_p90": _quantile_dict(data[:, COL["scale"]])["p90"],
            "distance_p50": _quantile_dict(data[:, COL["target_distance"]])["p50"],
            "final_xy_p50": _quantile_dict(data[:, COL["final_xy"]])["p50"],
        })

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(args.output_dir / "quantiles.csv", quantile_rows, list(quantile_rows[0]))
    _write_csv(args.output_dir / "correlations.csv", correlation_rows, list(correlation_rows[0]))
    _write_csv(args.output_dir / "distance_bands.csv", distance_rows, list(distance_rows[0]))
    _write_csv(args.output_dir / "size_quintiles.csv", size_bin_rows, list(size_bin_rows[0]))
    _write_csv(args.output_dir / "position_bins.csv", position_rows, list(position_rows[0]))
    _write_csv(args.output_dir / "scene_summary.csv", scene_rows, list(scene_rows[0]))
    _create_plots(args.output_dir, train_arrays)

    report_lines = [
        "# Data7_8 GT bbox distribution report",
        "",
        f"- Dataset: `{args.dataset_root.resolve()}`",
        f"- Episodes: train {manifest_counts['train']['episodes']}, val {manifest_counts['val']['episodes']}",
        f"- Declared frames: train {manifest_counts['train']['declared_samples']:,}, val {manifest_counts['val']['declared_samples']:,}",
        "- Boxes are normalized `cx, cy, w, h` GT labels. Statistics use visible frames with valid boxes.",
        "",
        "## Recommended training references",
        "",
    ]
    for agent in AGENTS:
        recommendation = summary["recommended"][agent]
        threshold = recommendation["distance_calibrated_far_threshold"]
        report_lines.extend([
            f"### {agent}",
            "",
            f"- Center reference: `{recommendation['center_ref']}`",
            f"- Center safe P10-P90: `{recommendation['center_safe_p10_p90']}`",
            f"- Scale P10 / P50: `{recommendation['scale_distribution_far_p10']:.6f}` / `{recommendation['scale_distribution_normal_p50']:.6f}`",
            "- Distance-calibrated far threshold: "
            f"`{_format_optional(threshold['threshold'])}`; balanced accuracy "
            f"`{_format_optional(threshold['balanced_accuracy'], 4)}`, TPR "
            f"`{_format_optional(threshold['tpr'], 4)}`, FPR "
            f"`{_format_optional(threshold['fpr'], 4)}`",
            "",
        ])
    report_lines.extend([
        "## Output files",
        "",
        "- `summary.json`: counts, quantiles, and recommended continuous-weight references.",
        "- `quantiles.csv`: global distribution percentiles.",
        "- `distance_bands.csv`: bbox scale and GT motion by configured follow-distance band.",
        "- `size_quintiles.csv`: distance and GT motion by bbox-size quintile.",
        "- `position_bins.csv`: GT yaw by horizontal bbox position.",
        "- `correlations.csv`: Pearson and sampled Spearman relationships.",
        "- `scene_summary.csv`: per-scene robustness audit.",
        "- `plots/`: distribution and relationship visualizations.",
        "",
        "Thresholds are analysis references, not hard inference gates. The loss should use smooth weights.",
    ])
    (args.output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"[bbox-stats] wrote {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
