#!/usr/bin/env python3
"""Create the deterministic frame list required by DT-replay vision cache."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=Path("/data/yh/data/manifests/eval_at.json"))
    p.add_argument("--raw-root", type=Path, default=Path("/data/yh/data/raw/at_eval_replay"))
    p.add_argument("--data-root", type=Path, default=Path("/data/yh/data/processed"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    entries = json.loads(args.manifest.read_text(encoding="utf-8"))
    paths = []
    markers = []
    for item in entries:
        parts = str(item["episode_id"]).split("/")
        if len(parts) != 4 or parts[0] != "at":
            raise ValueError(item["episode_id"])
        scene, run, stem = parts[1:]
        for agent in ("drone", "robotdog"):
            info = args.raw_root / scene / run / f"{stem}_{agent}_info.json"
            count = len(json.loads(info.read_text(encoding="utf-8")))
            frame_dir = args.data_root / "frames" / "at" / scene / run / stem / agent
            markers.append(frame_dir / ".complete.json")
            paths.extend(frame_dir / f"frame_{index:05d}.jpg" for index in range(1, count + 1))
    if len(entries) != 500 or len(paths) != 300000:
        raise ValueError(f"expected 500 episodes/300000 frames, got {len(entries)}/{len(paths)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    tmp.write_text("".join(f"{path}\n" for path in paths), encoding="utf-8")
    tmp.replace(args.output)
    marker_file = args.output.with_suffix(args.output.suffix + ".markers")
    marker_file.write_text("".join(f"{path}\n" for path in markers), encoding="utf-8")
    print(f"frames={len(paths)} markers={len(markers)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
