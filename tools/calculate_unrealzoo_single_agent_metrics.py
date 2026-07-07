#!/usr/bin/env python3
"""Summarize single-agent UnrealZoo closed-loop evaluation results."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--agent", choices=["robotdog", "drone"], default="robotdog")
    args = parser.parse_args()
    model_type = f"single_agent_{args.agent}"
    following_key = f"{args.agent}_following_rate"
    rows = []
    for path in sorted(args.eval_dir.rglob("*.json")):
        if path.name.endswith("_info.json"):
            continue
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(row, dict) and row.get("model_type") == model_type:
            rows.append(row)
    if not rows:
        print(f"No {model_type} result JSON found under {args.eval_dir}")
        return 1
    mean = lambda key: sum(float(row.get(key, 0.0)) for row in rows) / len(rows)
    statuses = Counter(str(row.get("status", "Unknown")) for row in rows)
    metrics = {
        "episodes": len(rows),
        "SR": mean("success") * 100.0,
        "TR": mean(following_key) * 100.0,
        "Centered": mean("centered_rate") * 100.0,
        "CR": mean("collision") * 100.0,
        "avg_steps": mean("total_step"),
        "avg_distance": mean("avg_distance"),
        "avg_fps": mean("fps"),
        "agent": args.agent,
        "status_counts": dict(statuses),
    }
    (args.eval_dir / "metrics_single_agent.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
