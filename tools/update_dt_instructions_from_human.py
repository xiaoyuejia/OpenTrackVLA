#!/usr/bin/env python3
"""Replace DT JSONL instructions from the folder's human*.json description."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def make_instruction(human: dict) -> str:
    a = human.get("appearance", {})
    gender = a.get("gender", "person")
    age = a.get("age_group", "")
    height = a.get("height_estimate", "")
    body = a.get("body_type", "")
    hair = a.get("hair", {})
    clothing = a.get("clothing", {})
    subject_parts = []
    if height and height not in {"none", "unknown"}: subject_parts.append(str(height).replace("_", " "))
    if body and body not in {"none", "unknown"}: subject_parts.append(f"{str(body).replace('_', ' ')}-built")
    if hair.get("style") and hair.get("style") not in {"none", "unknown"}: subject_parts.append(str(hair["style"]).replace("_", " "))
    subject_parts.append("man" if gender == "male" else "woman" if gender == "female" else "person")
    subject = " ".join(subject_parts)
    details = []
    for key in ("top", "outerwear", "bottom", "shoes"):
        value = clothing.get(key)
        if value and value not in {"none", "unknown"}:
            details.append(str(value))
    accessories = [str(x) for x in a.get("accessories", []) if x and x not in {"none", "unknown"}]
    details.extend(accessories)
    if details:
        return f"Follow the {subject} wearing {', '.join(details)} without collision."
    return f"Follow the {subject} without collision."


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", type=Path, required=True)
    p.add_argument("--processed-root", type=Path, default=Path("/data/yh/data/processed/train_jsonl/dt"))
    args = p.parse_args()
    humans = sorted(args.raw_dir.glob("human*.json"))
    if not humans:
        raise SystemExit(f"no human*.json under {args.raw_dir}")
    descriptions = [json.loads(path.read_text(encoding="utf-8")) for path in humans]
    instructions = {make_instruction(item) for item in descriptions}
    if len(instructions) != 1:
        raise SystemExit(f"expected one target description per folder, got {len(instructions)}")
    instruction = next(iter(instructions))
    relative = args.raw_dir.resolve().relative_to(Path("/data/yh/data/raw/dt").resolve())
    target = args.processed_root.resolve() / relative
    files = sorted(target.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"no DT JSONL files under {target}")
    changed = 0
    rows = 0
    for path in files:
        temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
        with path.open("r", encoding="utf-8") as reader, temporary.open("w", encoding="utf-8") as writer:
            for line in reader:
                if not line.strip():
                    continue
                obj = json.loads(line)
                if obj.get("instruction") != instruction:
                    changed += 1
                obj["instruction"] = instruction
                obj["instruction_source"] = f"{humans[0].name}:appearance"
                writer.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")
                rows += 1
        temporary.replace(path)
    print(f"updated={changed} rows={rows} files={len(files)}")
    print(f"instruction={instruction}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
