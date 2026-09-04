#!/usr/bin/env python3
"""Create a non-destructive, split-aware view under /data/hdt/ntv_data/data_final.

Large raw/processed trees are referenced by symlink; episode JSONL files are
linked into train/eval and dt/stt/at views according to the frozen v2 split.
"""
from __future__ import annotations

import argparse, json, os
from pathlib import Path


def link(path: Path, target: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or path.exists():
        if path.is_symlink() and os.path.realpath(path) == str(target):
            return
        raise FileExistsError(f"refusing to overwrite {path}")
    path.symlink_to(target, target_is_directory=target.is_dir())


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--final-root", type=Path, default=Path("/data/hdt/ntv_data/data_final"))
    p.add_argument("--raw-root", type=Path, default=Path("/data/hdt/ntv_data/sim_data/data_arr"))
    p.add_argument("--processed-root", type=Path, default=Path("/data/hdt/ntv_data/data/cyj_data_arr_processed"))
    p.add_argument("--at-eval-root", type=Path, default=Path("/data/hdt/ntv_data/data_arr_at_v1"))
    p.add_argument("--split-manifest", type=Path, default=Path("manifests/data_arr_train_test_v2/split_manifest.json"))
    args = p.parse_args()
    final = args.final_root.resolve(); raw = args.raw_root.resolve(); processed = args.processed_root.resolve()
    final.mkdir(parents=True, exist_ok=True)

    # Source roots are immutable references; no source data is moved/deleted.
    link(final / "raw" / "data_arr", raw)
    if (final / "raw" / "data_arr" / "at").exists():
        # Raw AT is an explicit derived tree, not a symlink: its status JSON
        # carries AT instruction metadata while media/info files link to DT.
        pass
    link(final / "processed" / "source", processed)
    link(final / "processed" / "frames", processed / "frames")
    link(final / "processed" / "vision_cache", processed / "vision_cache")
    link(final / "pending" / "at_eval_replay", args.at_eval_root.resolve())
    (final / "processed" / "eval" / "at").mkdir(parents=True, exist_ok=True)
    (final / "processed" / "eval" / "at" / "PENDING.md").write_text(
        "AT evaluation replay is incomplete. See ../../../../manifests/eval_at_pending.json and "
        "../../../../pending/at_eval_replay. Completed replay outputs are not exposed as processed JSONL "
        "until all planned episodes finish.\n",
        encoding="utf-8",
    )

    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    source_roots = {"dt": processed / "jsonl" / "dt", "stt": processed / "jsonl" / "stt", "at": processed / "jsonl" / "at"}
    # Build train/eval views. AT train is language-derived from DT train;
    # AT eval is replay output and is intentionally represented as pending.
    records = {"train": {"dt": [], "stt": [], "at": []}, "eval": {"dt": [], "stt": [], "at": []}}
    for split_name, out_split in (("train", "train"), ("test", "eval")):
        for item in split[split_name]:
            kind = item["data_kind"]
            if kind not in {"dt", "stt"}: continue
            parts = item["episode_id"].split("/")
            source = source_roots[kind] / parts[1] / parts[2] / f"{parts[3]}.jsonl"
            if not source.is_file(): raise FileNotFoundError(source)
            destination = final / "processed" / out_split / kind / parts[1] / parts[2] / f"{parts[3]}.jsonl"
            link(destination, source)
            records[out_split][kind].append({"episode_id": item["episode_id"], "data_kind": kind, "jsonl": str(destination), "source_jsonl": str(source)})

    # AT-language train view points to the separate derived JSONL tree.
    at_manifest = json.loads((processed / "at_language_manifest.json").read_text(encoding="utf-8"))
    train_dt_ids = {x["episode_id"] for x in split["train"] if x["data_kind"] == "dt"}
    for item in at_manifest:
        if item["episode_id"] not in train_dt_ids: continue
        source = Path(item["jsonl"])
        parts = item["episode_id"].split("/")
        destination = final / "processed" / "train" / "at" / parts[1] / parts[2] / f"{parts[3]}.jsonl"
        link(destination, source)
        records["train"]["at"].append({"episode_id": item["episode_id"], "data_kind": "at", "source_data_kind": "dt", "jsonl": str(destination), "source_jsonl": str(source), "instruction": item["instruction"]})

    # AT eval replay is still incomplete; expose its current tree but do not
    # present it as a processed training directory until every episode is done.
    eval_at = [x for x in json.loads((Path("manifests/data_arr_at_v1/evaluation.json")).read_text(encoding="utf-8"))]
    complete = []
    for item in eval_at:
        marker = args.at_eval_root.resolve() / item["scene"] / item["source_batch"] / f"{item['stem']}.complete.json"
        complete.append(marker.is_file())
    pending = {"data_kind": "at", "planned_episodes": len(eval_at), "completed_episodes": sum(complete), "remaining_episodes": len(eval_at)-sum(complete), "source_root": str(args.at_eval_root.resolve()), "status": "pending_until_all_replay_complete"}

    for split_name, kinds in records.items():
        for kind, entries in kinds.items():
            (final / "manifests").mkdir(parents=True, exist_ok=True)
            (final / "manifests" / f"{split_name}_{kind}.json").write_text(json.dumps(entries, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    (final / "manifests" / "eval_at_pending.json").write_text(json.dumps(pending, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    catalog = {"schema_version": "data_final_layout_v1", "raw": str(raw), "processed": str(processed), "split_manifest": str(args.split_manifest.resolve()), "counts": {s: {k: len(v) for k,v in kinds.items()} for s,kinds in records.items()}, "at_eval": pending}
    (final / "catalog.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    readme = f'''# data_final\n\nThis is a non-destructive, split-aware view of the data_arr corpus. Source trees are symlinked; no raw or processed files were moved or deleted.\n\n## Layout\n\n- `raw/data_arr` -> `{raw}`, containing sibling `dt/`, `stt/`, and `at/` trees.\n- `processed/source` -> `{processed}`\n- `processed/frames` and `processed/vision_cache` expose shared assets for relative JSONL paths.\n- `processed/train/dt`, `processed/train/stt`, `processed/train/at`\n- `processed/eval/dt`, `processed/eval/stt`\n- `pending/at_eval_replay` -> `{args.at_eval_root.resolve()}` (AT eval is incomplete)\n\nAT train is the 1,933-episode DT-derived AT-language set. AT eval remains pending until all 500 replay markers exist.\n'''
    (final / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(catalog, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
