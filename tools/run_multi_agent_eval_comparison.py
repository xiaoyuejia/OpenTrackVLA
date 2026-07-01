#!/usr/bin/env python3
"""Run comparable multi-agent evaluations from one YAML configuration."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.calculate_unrealzoo_metrics import calculate_metrics


METRIC_COLUMNS = (
    "SR",
    "JointTR",
    "DroneTR",
    "RobotDogTR",
    "CR",
    "avg_steps",
    "avg_fps",
)


def as_env_value(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if value is None:
        return ""
    return str(value)


def load_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: YAML root must be a mapping")
    models = data.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError(f"{path}: models must contain at least one entry")
    return data


def validate_model(model: dict[str, Any]) -> None:
    for key in ("name", "checkpoint", "eval_root", "gpu", "render_gpu"):
        if key not in model:
            raise ValueError(f"model entry is missing {key}: {model}")
    checkpoint = Path(str(model["checkpoint"]))
    if not checkpoint.exists():
        raise FileNotFoundError(f"{model['name']}: checkpoint not found: {checkpoint}")


def stream_output(name: str, process: subprocess.Popen[str], log_path: Path) -> None:
    assert process.stdout is not None
    with log_path.open("w", encoding="utf-8") as log:
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(f"[{name}] {line}", end="", flush=True)


def launch_model(
    config: dict[str, Any],
    model: dict[str, Any],
    overwrite: bool,
) -> tuple[subprocess.Popen[str], threading.Thread, Path]:
    validate_model(model)
    name = str(model["name"])
    eval_root = Path(str(model["eval_root"]))
    if eval_root.exists() and any(eval_root.rglob("*.json")):
        if not overwrite:
            raise FileExistsError(
                f"{name}: existing JSON results found under {eval_root}; "
                "use --overwrite to replace them"
            )
        shutil.rmtree(eval_root)
    eval_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update({key: as_env_value(value) for key, value in (config.get("common_env") or {}).items()})
    env.update({key: as_env_value(value) for key, value in (model.get("env") or {}).items()})
    env.update(
        {
            "PYTHON_BIN": str(config.get("python_bin") or sys.executable),
            "CKPT_DIR": str(model["checkpoint"]),
            "EVAL_ROOT": str(eval_root),
            "EVAL_GPUS": str(model["gpu"]),
            "RENDER_GPUS": str(model["render_gpu"]),
        }
    )

    pipeline = REPO_ROOT / str(config.get("pipeline", "sh/run_multi_agent_eval.sh"))
    if not pipeline.is_file():
        raise FileNotFoundError(f"pipeline not found: {pipeline}")
    log_path = eval_root / "pipeline.log"
    print(
        f"[launch] {name}: gpu={model['gpu']} render_gpu={model['render_gpu']} "
        f"checkpoint={model['checkpoint']}",
        flush=True,
    )
    process = subprocess.Popen(
        ["bash", str(pipeline)],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    thread = threading.Thread(
        target=stream_output,
        args=(name, process, log_path),
        daemon=True,
    )
    thread.start()
    return process, thread, log_path


def write_comparison(
    config: dict[str, Any],
    model_metrics: dict[str, dict[str, Any]],
) -> Path:
    comparison_dir = Path(str(config["comparison_dir"]))
    comparison_dir.mkdir(parents=True, exist_ok=True)
    json_path = comparison_dir / "comparison_metrics.json"
    json_path.write_text(
        json.dumps(model_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    csv_path = comparison_dir / "comparison_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("model",) + METRIC_COLUMNS)
        writer.writeheader()
        for name, metrics in model_metrics.items():
            writer.writerow({"model": name, **{key: metrics.get(key, 0.0) for key in METRIC_COLUMNS}})

    print("\n" + "=" * 112)
    print(
        f"{'model':36s} {'SR':>8s} {'JointTR':>9s} {'DroneTR':>9s} "
        f"{'DogTR':>9s} {'CR':>8s} {'steps':>9s} {'FPS':>8s}"
    )
    print("-" * 112)
    for name, metrics in model_metrics.items():
        print(
            f"{name:36.36s} {metrics['SR']:8.2f} {metrics['JointTR']:9.2f} "
            f"{metrics['DroneTR']:9.2f} {metrics['RobotDogTR']:9.2f} "
            f"{metrics['CR']:8.2f} {metrics['avg_steps']:9.1f} {metrics['avg_fps']:8.2f}"
        )
    print("=" * 112)
    print(f"[comparison] JSON: {json_path}")
    print(f"[comparison] CSV:  {csv_path}")
    return json_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing result directories before evaluation.",
    )
    args = parser.parse_args()

    config = load_config(args.config.resolve())
    models = config["models"]
    running: list[tuple[dict[str, Any], subprocess.Popen[str], threading.Thread, Path]] = []
    failures: list[str] = []

    try:
        if bool(config.get("parallel", True)):
            for model in models:
                process, thread, log_path = launch_model(config, model, args.overwrite)
                running.append((model, process, thread, log_path))
        else:
            for model in models:
                process, thread, log_path = launch_model(config, model, args.overwrite)
                process.wait()
                thread.join()
                running.append((model, process, thread, log_path))

        for model, process, thread, log_path in running:
            return_code = process.wait()
            thread.join()
            if return_code != 0:
                failures.append(
                    f"{model['name']} exited with code {return_code}; log={log_path}"
                )
    except KeyboardInterrupt:
        print("\n[stop] terminating evaluation processes...", flush=True)
        for _model, process, _thread, _log_path in running:
            if process.poll() is None:
                process.terminate()
        for _model, process, thread, _log_path in running:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
            thread.join(timeout=2)
        return 130

    if failures:
        print("\n".join(f"[ERROR] {item}" for item in failures), file=sys.stderr)
        return 1

    model_metrics: dict[str, dict[str, Any]] = {}
    for model in models:
        metrics = calculate_metrics(Path(str(model["eval_root"])))
        if metrics is None:
            print(f"[ERROR] no metrics found for {model['name']}", file=sys.stderr)
            return 1
        model_metrics[str(model["name"])] = metrics
    write_comparison(config, model_metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
