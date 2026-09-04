#!/usr/bin/env python3
"""Derive AT-language training JSONL from the frozen DT training split.

Only instruction/AT provenance fields are changed. Frames, bbox, pose and
waypoint labels continue to reference the existing processed DT data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


AT_INSTRUCTIONS = (
    "Track the person you first observe at the start.",
    "Follow the primary person identified at episode start.",
    "Pursue the first person selected at the beginning of the episode.",
    "Keep following the initially observed target person.",
    "Stay with the main person seen when the episode begins.",
    "Track the designated person first seen at the start.",
    "Follow the initial target person throughout the episode.",
    "Pursue the main target identified in the initial view.",
)


def stable_index(episode_id: str, seed: int, size: int) -> int:
    digest = hashlib.sha256(f"{seed}:{episode_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % size


def atomic_write_jsonl(destination: Path, rows: list[dict]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", type=Path, default=Path("/data/hdt/ntv_data/data/cyj_data_arr_processed"))
    parser.add_argument("--split-manifest", type=Path, default=Path("manifests/data_arr_train_test_v2/split_manifest.json"))
    parser.add_argument("--output-name", default="at")
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()

    processed = args.processed_root.resolve()
    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    train_dt = [item for item in split["train"] if item.get("data_kind") == "dt"]
    if len(train_dt) != 1933:
        raise ValueError(f"expected 1933 DT train episodes, found {len(train_dt)}")

    source_root = processed / "jsonl" / "dt"
    output_root = processed / "jsonl" / args.output_name
    manifest = []
    assignment = []
    instruction_counts = {instruction: 0 for instruction in AT_INSTRUCTIONS}
    missing = []

    for item in sorted(train_dt, key=lambda value: value["episode_id"]):
        source = source_root / item["scene"] / item["source_batch"] / f"{item['stem']}.jsonl"
        if not source.is_file():
            missing.append({"episode_id": item["episode_id"], "source": str(source)})
            continue
        instruction_index = stable_index(item["episode_id"], args.seed, len(AT_INSTRUCTIONS))
        instruction = AT_INSTRUCTIONS[instruction_index]
        destination = output_root / item["scene"] / item["source_batch"] / f"{item['stem']}.jsonl"
        rows = []
        with source.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                row["instruction"] = instruction
                row["task_variant"] = "at_language_derived"
                row["target_selection_policy"] = "episode_initial_designated_target"
                row["target_identity_policy"] = "fixed_dt_target_pose_identity"
                row["distractors_are_non_target"] = True
                row["instruction_source"] = "generated_initial_target_at_v1"
                row["instruction_seed"] = args.seed
                row["instruction_index"] = instruction_index
                rows.append(row)
        atomic_write_jsonl(destination, rows)
        relative = destination.relative_to(processed)
        manifest.append({
            "episode_id": item["episode_id"],
            "data_kind": "at",
            "source_data_kind": "dt",
            "source_episode_id": item["episode_id"],
            "jsonl": str(destination),
            "relative_jsonl": str(relative),
            "instruction": instruction,
            "instruction_index": instruction_index,
            "target_selection_policy": "episode_initial_designated_target",
            "target_identity_policy": "fixed_dt_target_pose_identity",
            "distractors_are_non_target": True,
        })
        assignment.append({"episode_id": item["episode_id"], "instruction": instruction, "instruction_index": instruction_index})
        instruction_counts[instruction] += 1

    if missing:
        raise FileNotFoundError(f"missing {len(missing)} DT JSONL files; first={missing[0]}")

    manifest_path = processed / "at_language_manifest.json"
    assignment_path = processed / "at_language_instruction_assignment.json"
    summary_path = processed / "at_language_summary.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    assignment_path.write_text(json.dumps(assignment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "schema_version": "data_arr_at_language_v1",
        "source_split_manifest": str(args.split_manifest.resolve()),
        "source_jsonl_root": str(source_root),
        "output_jsonl_root": str(output_root),
        "episodes": len(manifest),
        "source_data_kind": "dt",
        "source_split": "train",
        "instruction_count": len(AT_INSTRUCTIONS),
        "instruction_seed": args.seed,
        "instruction_counts": instruction_counts,
        "target_selection_policy": "episode_initial_designated_target",
        "target_identity_policy": "fixed_dt_target_pose_identity",
        "distractors_are_non_target": True,
        "frames_copied": False,
        "trajectory_labels_changed": False,
        "missing": missing,
        "manifest": str(manifest_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"episodes": len(manifest), "output": str(output_root), "instruction_counts": instruction_counts}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
