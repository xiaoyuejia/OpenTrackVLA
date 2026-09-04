#!/usr/bin/env python3
"""Build a recorded-evaluation manifest from the data package eval splits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path("/data/yh/data"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--kinds", nargs="+", choices=("dt", "stt", "at"), default=("dt", "stt", "at"))
    args = p.parse_args()
    root = args.data_root.resolve()
    entries = []
    seen = set()
    for kind in args.kinds:
        name = f"eval_{kind}.json"
        rows = json.loads((root / "manifests" / name).read_text(encoding="utf-8"))
        for row in rows:
            episode_id = str(row["episode_id"])
            parts = episode_id.split("/")
            if len(parts) < 4 or parts[0] not in {"dt", "stt", "at"}:
                raise ValueError(f"invalid episode_id: {episode_id}")
            scene = parts[1]
            rel = "/".join(parts[:-1])
            stem = parts[-1]
            replay_root = "at_eval_replay" if parts[0] == "at" else parts[0]
            info = root / "raw" / replay_root / "/".join(parts[1:-1]) / f"{stem}_drone_info.json"
            if not info.is_file():
                raise FileNotFoundError(info)
            key = f"{rel}/{stem}"
            if key in seen:
                raise ValueError(f"duplicate episode: {key}")
            seen.add(key)
            entries.append({"scene": scene, "stem": stem, "relative_dir": rel, "info": str(info)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"test": entries}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(entries)} entries to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
