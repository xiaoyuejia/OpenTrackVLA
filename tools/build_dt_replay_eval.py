#!/usr/bin/env python3
"""Build DT-instruction eval JSONL using the AT replay visual observations."""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path


INSTRUCTION = re.compile(rb'"instruction":\s*"(?:[^"\\]|\\.)*"')


def first_instruction(path: Path) -> str:
    with path.open("r", encoding="utf-8") as handle:
        return str(json.loads(handle.readline())["instruction"])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--joint-root", type=Path, default=Path("/data/yh/data/processed"))
    p.add_argument("--manifest", type=Path, default=Path("/data/yh/data/manifests/eval_dt_replay.json"))
    args = p.parse_args()
    source_root = args.joint_root / "eval_jsonl" / "at"
    dt_root = args.joint_root / "eval_jsonl" / "dt"
    output_root = args.joint_root / "eval_jsonl" / "dt_replay"
    entries = []
    for index, source in enumerate(sorted(source_root.rglob("*.jsonl")), 1):
        relative = source.relative_to(source_root)
        original_dt = dt_root / relative
        if not original_dt.is_file():
            raise FileNotFoundError(original_dt)
        instruction = first_instruction(original_dt)
        replacement = json.dumps("instruction", ensure_ascii=False).encode() + b": " + json.dumps(instruction, ensure_ascii=False).encode()
        destination = output_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + f".tmp.{os.getpid()}")
        rows = 0
        with source.open("rb") as reader, temporary.open("wb") as writer:
            for line in reader:
                line, changed = INSTRUCTION.subn(replacement, line, count=1)
                if changed != 1:
                    raise ValueError(f"missing instruction in {source}, row={rows + 1}")
                line = line.replace(b'"episode_id": "at/', b'"episode_id": "dt_replay/', 1)
                line = line.replace(b'"rel_run_dir": "at/', b'"rel_run_dir": "dt_replay/', 1)
                writer.write(line)
                rows += 1
        if rows == 0:
            raise ValueError(f"empty JSONL: {source}")
        temporary.replace(destination)
        parts = relative.parts
        episode_id = "dt_replay/" + str(relative.with_suffix(""))
        entries.append({
            "episode_id": episode_id,
            "data_kind": "dt",
            "task_type": "dt",
            "jsonl": os.path.relpath(destination, args.manifest.parent),
            "source_jsonl": os.path.relpath(source, args.manifest.parent),
            "source_data_kind": "at_eval_replay",
            "instruction": instruction,
        })
        if index == 1 or index % 25 == 0:
            print(f"[progress] {index}/500", flush=True)
    if len(entries) != 500:
        raise ValueError(f"expected 500 AT replay JSONLs, found {len(entries)}")
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    tmp_manifest = args.manifest.with_suffix(args.manifest.suffix + f".tmp.{os.getpid()}")
    tmp_manifest.write_text(json.dumps(entries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_manifest.replace(args.manifest)
    print(f"[done] episodes=500 output={output_root} manifest={args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
