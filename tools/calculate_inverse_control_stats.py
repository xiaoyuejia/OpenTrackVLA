#!/usr/bin/env python3
"""Calculate robust normalization scales for inverse-control supervision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


COMPONENTS = {
    "control_scale_drone_forward": (0, 0, 0.25),
    "control_scale_drone_lateral": (0, 1, 0.25),
    "control_scale_drone_yaw": (0, 2, 0.25),
    "control_scale_dog_forward": (1, 0, 0.25),
    "control_scale_dog_yaw": (1, 2, 0.25),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--percentile", type=float, default=95.0)
    args = parser.parse_args()

    values: dict[str, list[float]] = {key: [] for key in COMPONENTS}
    samples = 0
    for path in sorted(args.jsonl_root.rglob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                target = np.asarray(row.get("inverse_control_target"), dtype=np.float64)
                valid = np.asarray(row.get("inverse_control_valid_mask"), dtype=bool)
                if target.shape != (2, 3) or valid.shape != (2, 3):
                    continue
                samples += 1
                for key, (agent, component, _floor) in COMPONENTS.items():
                    if valid[agent, component] and np.isfinite(target[agent, component]):
                        values[key].append(abs(float(target[agent, component])))

    recommended: dict[str, float] = {}
    component_stats: dict[str, dict[str, float | int | None]] = {}
    for key, (_agent, _component, floor) in COMPONENTS.items():
        array = np.asarray(values[key], dtype=np.float64)
        p95 = float(np.percentile(array, args.percentile)) if array.size else None
        recommended[key] = max(float(p95 or 0.0), float(floor))
        component_stats[key] = {
            "count": int(array.size),
            "mean_abs": float(array.mean()) if array.size else None,
            "p50_abs": float(np.percentile(array, 50.0)) if array.size else None,
            "p95_abs": p95,
            "max_abs": float(array.max()) if array.size else None,
        }
    payload = {
        "source": str(args.jsonl_root.resolve()),
        "samples": samples,
        "percentile": float(args.percentile),
        "components": component_stats,
        "recommended_scales": recommended,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
