#!/usr/bin/env python3
"""Attach source-derived target appearance instructions to processed dt JSONL."""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from pathlib import Path
from typing import Any, Dict


def _appearance(source_root: Path, example: Dict[str, Any]) -> Dict[str, Any]:
    parts = str(example["rel_run_dir"]).split("/")
    scene = parts[1]
    match = re.fullmatch(r"dt_camera\d+__(\d+)__seed_(\d+)", parts[-1])
    if not match:
        raise ValueError(f"Cannot parse dt rel_run_dir: {example['rel_run_dir']}")
    group, seed = match.groups()
    folder = source_root / group / f"seed_{seed}" / scene
    files = sorted(folder.glob("human*.json"))
    if len(files) != 1:
        raise FileNotFoundError(f"Expected exactly one human*.json under {folder}, got {files}")
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    appearance = payload.get("appearance")
    if not isinstance(appearance, dict):
        raise ValueError(f"Missing appearance in {files[0]}")
    return {"appearance": appearance, "person_id": payload.get("person_id"), "source": str(files[0])}


def _text(appearance: Dict[str, Any]) -> str:
    def clean(value: Any) -> str:
        value = str(value or "").replace("_", " ").strip()
        return "" if value.lower() in {"none", "not observable", "not_observable", "unknown"} else value

    gender = clean(appearance.get("gender"))
    height = clean(appearance.get("height_estimate"))
    body = clean(appearance.get("body_type"))
    hair = appearance.get("hair") or {}
    clothing = appearance.get("clothing") or {}
    hair_color, hair_style = clean(hair.get("color")), clean(hair.get("style"))
    top, bottom = clean(clothing.get("top")), clean(clothing.get("bottom"))
    accessories = [clean(x) for x in (appearance.get("accessories") or [])]
    accessories = [x for x in accessories if x]

    parts = []
    if gender:
        parts.append(gender)
    if height:
        parts.append(height)
    if body:
        parts.append(body)
    if top:
        parts.append("wearing " + top)
    if bottom:
        parts.append("with " + bottom)
    if hair_color and hair_style:
        parts.append(f"{hair_color} {hair_style} hair")
    elif hair_color:
        parts.append(f"{hair_color} hair")
    if accessories:
        parts.append("and " + ", ".join(accessories))
    description = ", ".join(parts)
    return (
        "Track the target person described as " + description
        + "; ignore all other people and avoid collisions."
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    files = sorted(args.jsonl_root.rglob("*.jsonl"))
    if not files:
        raise FileNotFoundError(args.jsonl_root)
    report = {"files": len(files), "episodes": 0, "instructions": {}, "sources": {}, "errors": []}
    for path in files:
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent), text=True)
        try:
            with open(fd, "w", encoding="utf-8") as out, path.open("r", encoding="utf-8") as inp:
                first = True
                instruction = None
                source = None
                for line in inp:
                    if not line.strip():
                        continue
                    example = json.loads(line)
                    if first:
                        if example.get("target_description") and example.get("target_appearance_source"):
                            Path(temp_name).unlink(missing_ok=True)
                            first = False
                            instruction = None
                            source = None
                            break
                        info = _appearance(args.source_root, example)
                        instruction = _text(info["appearance"])
                        source = info["source"]
                        first = False
                        report["episodes"] += 1
                        report["instructions"][instruction] = report["instructions"].get(instruction, 0) + 1
                        report["sources"][source] = report["sources"].get(source, 0) + 1
                    example["instruction"] = instruction
                    example["target_description"] = instruction[len("Track the target person described as ") : -len("; ignore all other people and avoid collisions.")]
                    example["target_appearance_source"] = source
                    out.write(json.dumps(example, ensure_ascii=False, separators=(",", ":")) + "\n")
            if instruction is not None:
                Path(temp_name).replace(path)
        except Exception as exc:
            Path(temp_name).unlink(missing_ok=True)
            report["errors"].append({"file": str(path), "error": repr(exc)})
            raise

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("files", "episodes", "errors")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
