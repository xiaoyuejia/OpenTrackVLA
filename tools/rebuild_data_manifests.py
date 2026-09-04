#!/usr/bin/env python3
"""Rebuild manifests for the compact /data/yh/data layout.

The authoritative inputs are processed/train_jsonl and processed/eval_jsonl;
backup is intentionally never scanned.
"""
from __future__ import annotations
import json
from pathlib import Path

DATA = Path("/data/yh/data")
PROCESSED = DATA / "processed"
OUT = DATA / "manifests"

def files(split: str, kind: str):
    return sorted((PROCESSED / split / kind).rglob("*.jsonl"))

def episode_entry(path: Path, split: str, kind: str) -> dict:
    rel = path.relative_to(PROCESSED / split / kind)
    row = json.loads(path.open(encoding="utf-8").readline())
    # Namespace by split kind: AT JSONL may retain a DT-style source episode_id.
    episode = f"{kind}/{rel.with_suffix('')}"
    return {
        "episode_id": episode,
        "data_kind": kind,
        "task_type": str(row.get("task_type", kind)),
        "jsonl": str(path.resolve()),
    }

def recorded_entry(path: Path, kind: str) -> dict:
    rel = path.relative_to(PROCESSED / "eval_jsonl" / kind)
    row = json.loads(path.open(encoding="utf-8").readline())
    episode = str(row.get("episode_id", f"{kind}/{rel.with_suffix('')}"))
    parts = episode.split("/")
    logical_kind = parts[0]
    scene, run, stem = parts[-3], parts[-2], parts[-1]
    raw_kind = "at_eval_replay" if logical_kind in {"at", "dt_replay"} else logical_kind
    raw_dir = DATA / "raw" / raw_kind / scene / run
    item = {
        "episode_id": episode,
        "data_kind": "dt" if logical_kind == "dt_replay" else kind,
        "task_type": str(row.get("task_type", kind + "_replay")),
        "jsonl": str(path.resolve()),
        "scene": scene,
        "stem": stem,
        # Keep DT and AT replay entries distinct in the planner even though
        # they intentionally share the same underlying replay recording.
        "relative_dir": f"{('dt_replay' if logical_kind == 'dt_replay' else 'at_replay' if logical_kind == 'at' else raw_kind)}/{scene}/{run}",
        "info": str((raw_dir / f"{stem}_drone_info.json").resolve()),
    }
    if raw_kind == "at_eval_replay":
        item.update({
            "replay_meta": str((raw_dir / f"{stem}_at_episode.json").resolve()),
            "source_data_kind": "at_eval_replay",
            "replay_distractors": True,
            "replay_motion_policy": "recorded_distractor_actions",
        })
    if "instruction" in row:
        item["instruction"] = row["instruction"]
    return item

def write(name: str, value) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def main() -> int:
    train, val = [], []
    for kind in ("stt", "dt", "at"):
        train += [episode_entry(p, "train_jsonl", kind) for p in files("train_jsonl", kind)]
        val += [episode_entry(p, "eval_jsonl", kind) for p in files("eval_jsonl", kind)]
        write(f"train_{kind}.json", [x for x in train if x["data_kind"] == kind])
        write(f"eval_{kind}.json", [x for x in val if x["data_kind"] == kind])
    write("train_joint.json", train)
    write("val_joint.json", val)
    dt = [recorded_entry(p, "dt") for p in files("eval_jsonl", "dt")]
    at = [recorded_entry(p, "at") for p in files("eval_jsonl", "at")]
    stt = [recorded_entry(p, "stt") for p in files("eval_jsonl", "stt")]
    write("eval_dt_replay_recorded.json", {"test": dt})
    write("eval_at_replay_recorded.json", {"test": at})
    write("eval_all_2400_recorded.json", {"test": stt + dt + at})
    print(f"train={len(train)} val={len(val)} stt={len(stt)} dt_replay={len(dt)} at_replay={len(at)} all_recorded={len(stt)+len(dt)+len(at)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
