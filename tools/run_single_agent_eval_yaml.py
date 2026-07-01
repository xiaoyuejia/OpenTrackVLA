#!/usr/bin/env python3
"""Run one single-agent UnrealZoo evaluation from a YAML environment mapping."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def env_value(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if value is None:
        return ""
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("YAML root must be a mapping")
    agent = str(config.get("agent", "")).strip()
    if agent not in {"drone", "robotdog"}:
        raise ValueError("agent must be drone or robotdog")

    pipeline = REPO_ROOT / str(
        config.get("pipeline") or f"sh/run_{agent}_single_agent_pipeline.sh"
    )
    if not pipeline.is_file():
        raise FileNotFoundError(f"pipeline not found: {pipeline}")

    values = dict(config.get("env") or {})
    for key in ("CKPT_DIR", "MANIFEST", "EVAL_ROOT"):
        if key not in values:
            raise ValueError(f"env is missing required key: {key}")
    checkpoint = Path(str(values["CKPT_DIR"]))
    manifest = Path(str(values["MANIFEST"]))
    eval_root = Path(str(values["EVAL_ROOT"]))
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if not manifest.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest}")
    if args.overwrite and eval_root.exists():
        shutil.rmtree(eval_root)

    env = os.environ.copy()
    env.update({key: env_value(value) for key, value in values.items()})
    env["PYTHON_BIN"] = str(
        config.get("python_bin") or "/home/hdt/miniconda3/envs/omtracknew/bin/python"
    )
    print(
        f"[single-eval] agent={agent} gpu={env.get('EVAL_GPU')} "
        f"checkpoint={checkpoint} output={eval_root}",
        flush=True,
    )
    if args.dry_run:
        print(f"[single-eval] pipeline={pipeline}")
        for key in sorted(values):
            print(f"{key}={env[key]}")
        return 0
    return subprocess.call(["bash", str(pipeline)], cwd=REPO_ROOT, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
