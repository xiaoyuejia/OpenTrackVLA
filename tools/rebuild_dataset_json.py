#!/usr/bin/env python3
"""Rebuild the combined dataset.json from canonical JSONL records."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    files = sorted(args.jsonl_root.rglob("*.jsonl"))
    if not files:
        raise FileNotFoundError(args.jsonl_root)
    fd, temp_name = tempfile.mkstemp(prefix=f".{args.output.name}.", dir=str(args.output.parent), text=True)
    total = 0
    dt = stt = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            out.write("[\n")
            first = True
            for file_index, path in enumerate(files, 1):
                with path.open("r", encoding="utf-8") as inp:
                    for line in inp:
                        if not line.strip():
                            continue
                        # JSONL was already validated during preprocessing; parse
                        # here to ensure the new combined file contains objects.
                        item = json.loads(line)
                        if not isinstance(item, dict):
                            raise ValueError(f"non-object record: {path}")
                        if not first:
                            out.write(",\n")
                        out.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")))
                        first = False
                        total += 1
                        if str(item.get("episode_id", "")).startswith("dt/"):
                            dt += 1
                        elif str(item.get("episode_id", "")).startswith("stt/"):
                            stt += 1
                if file_index % 250 == 0:
                    out.flush()
                    print(f"[progress] files={file_index}/{len(files)} records={total}", flush=True)
            out.write("\n]\n")
            out.flush()
            os.fsync(out.fileno())
        os.replace(temp_name, args.output)
    except Exception:
        Path(temp_name).unlink(missing_ok=True)
        raise
    print(f"[done] output={args.output} records={total} dt={dt} stt={stt}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
