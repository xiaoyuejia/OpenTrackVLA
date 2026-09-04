#!/usr/bin/env python3
"""Build no-validation data_arr split with STT loststart:standard = 2:3."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


SEED = 20260825


def h(value: str) -> str:
    return hashlib.sha256(f"{SEED}:{value}".encode()).hexdigest()


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def proportional_quotas(counts: Counter[str], total: int) -> dict[str, int]:
    source_total = sum(counts.values())
    raw = {key: value * total / source_total for key, value in counts.items()}
    quotas = {key: int(math.floor(value)) for key, value in raw.items()}
    remaining = total - sum(quotas.values())
    order = sorted(counts, key=lambda key: (-(raw[key] - quotas[key]), key))
    for key in order[:remaining]:
        quotas[key] += 1
    return quotas


def choose_loststart(items: list[dict], count: int) -> tuple[list[dict], list[dict]]:
    by_scene: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        by_scene[item["scene"]].append(item)
    quotas = proportional_quotas(Counter({key: len(value) for key, value in by_scene.items()}), count)
    selected, remainder = [], []
    for scene, values in sorted(by_scene.items()):
        values = sorted(values, key=lambda item: h(item["episode_id"]))
        selected.extend(values[: quotas[scene]])
        remainder.extend(values[quotas[scene] :])
    if len(selected) != count:
        raise ValueError(f"loststart selection mismatch: {len(selected)} != {count}")
    return selected, remainder


def choose_standard_groups(
    candidates: list[dict], existing_test: list[dict], add_count: int, desired_total: int
) -> tuple[list[dict], list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for item in candidates:
        groups[item["leakage_group"]].append(item)
    # Exact subset-sum over group sizes, preferring fewer/larger indivisible
    # groups. This dataset resolves 357 as one 341 group plus 16 singleton groups.
    sizes = sorted(groups, key=lambda key: (-len(groups[key]), h(key)))
    reachable: dict[int, tuple[str, ...]] = {0: ()}
    for group_id in sizes:
        size = len(groups[group_id])
        for subtotal, chosen in sorted(list(reachable.items()), reverse=True):
            new = subtotal + size
            if new <= add_count and new not in reachable:
                reachable[new] = (*chosen, group_id)
        if add_count in reachable:
            break
    if add_count not in reachable:
        raise ValueError(f"Cannot select {add_count} standard episodes without splitting leakage groups")
    chosen_ids = set(reachable[add_count])

    # If the exact construction includes singleton freedom, replace selected
    # singleton groups deterministically to improve scene balance.
    fixed = [gid for gid in chosen_ids if len(groups[gid]) > 1]
    fixed_count = sum(len(groups[gid]) for gid in fixed)
    singleton_needed = add_count - fixed_count
    singleton_groups = [gid for gid in groups if len(groups[gid]) == 1]
    all_standard = existing_test + candidates
    desired_scene = proportional_quotas(
        Counter(item["scene"] for item in all_standard), desired_total
    )
    current = Counter(item["scene"] for item in existing_test)
    for gid in fixed:
        current.update(item["scene"] for item in groups[gid])
    singleton_groups.sort(
        key=lambda gid: (
            -(desired_scene[groups[gid][0]["scene"]] - current[groups[gid][0]["scene"]]),
            h(gid),
        )
    )
    selected_singletons = singleton_groups[:singleton_needed]
    selected_ids = set(fixed + selected_singletons)
    selected = [item for gid in selected_ids for item in groups[gid]]
    remainder = [item for gid in groups if gid not in selected_ids for item in groups[gid]]
    if len(selected) != add_count:
        raise ValueError(f"standard selection mismatch: {len(selected)} != {add_count}")
    return selected, remainder


def set_split(items: list[dict], split: str) -> list[dict]:
    return [{**item, "split": split} for item in items]


def processed(items: list[dict], processed_root: Path, waypoint_count: int) -> list[dict]:
    output = []
    for item in sorted(items, key=lambda value: value["episode_id"]):
        relative = item["waypoint8_relative_jsonl"]
        row = {
            "episode_id": item["episode_id"],
            "scene": item["scene"],
            "stem": item["stem"],
            "data_kind": item["data_kind"],
            "source_batch": item["source_batch"],
            "relative_jsonl": relative,
            "waypoint_count": waypoint_count,
            "quality_tier": item["quality_tier"],
            "leakage_group": item["leakage_group"],
        }
        if waypoint_count == 8:
            row["jsonl"] = str((processed_root / relative).resolve())
        else:
            row["jsonl"] = None
            row["build_status"] = "pending_10_waypoint_rebuild"
        output.append(row)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1", type=Path, default=Path("manifests/data_arr_7_1_2_v1/split_manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("manifests/data_arr_train_test_v2"))
    args = parser.parse_args()
    v1 = json.loads(args.v1.read_text(encoding="utf-8"))
    processed_root = Path(v1["waypoint8_processed_root"])

    old_train, old_val, old_test = v1["train"], v1["val"], v1["test"]
    loststart = [item for item in old_test if "loststart" in item["source_batch"].lower()]
    old_standard_test = [item for item in old_test if "loststart" not in item["source_batch"].lower()]
    dt_standard_test_all = [item for item in old_standard_test if item["data_kind"] == "dt"]
    stt_standard_test = [item for item in old_standard_test if item["data_kind"] == "stt"]

    selected_lost, released_lost = choose_loststart(loststart, 560)
    val_stt = [item for item in old_val if item["data_kind"] == "stt"]
    added_standard, remaining_val_stt = choose_standard_groups(
        val_stt, stt_standard_test, add_count=357, desired_total=840
    )
    val_dt = [item for item in old_val if item["data_kind"] == "dt"]

    # The old DT test consists of two indivisible source-batch groups (264/237),
    # so an exact total of 1900 requires one explicit batch exception. Move a
    # globally unique trajectory back to train; trajectory isolation remains hard.
    all_core = old_train + old_val + old_test
    trajectory_frequency = Counter(
        (item["scene"], item["trajectory_signature"]) for item in all_core
    )
    dt_release_candidates = [
        item for item in dt_standard_test_all
        if trajectory_frequency[(item["scene"], item["trajectory_signature"])] == 1
    ]
    if not dt_release_candidates:
        raise ValueError("No unique DT trajectory available for 1900 rounding")
    released_dt = min(dt_release_candidates, key=lambda item: h(f"dt-round:{item['episode_id']}"))
    dt_standard_test = [
        item for item in dt_standard_test_all if item["episode_id"] != released_dt["episode_id"]
    ]

    test_standard = stt_standard_test + added_standard
    test = set_split(dt_standard_test + test_standard + selected_lost, "test")
    train = set_split(old_train + val_dt + remaining_val_stt + released_lost + [released_dt], "train")
    pseudo = [{**item, "split": "auxiliary_train"} for item in v1["auxiliary_pseudo_train"]]
    unresolved = [{**item, "split": "excluded"} for item in v1["excluded_unresolved"]]

    ids = {"train": {x["episode_id"] for x in train}, "test": {x["episode_id"] for x in test}}
    if ids["train"] & ids["test"] or len(ids["train"] | ids["test"]) != 9447:
        raise ValueError("v2 core split overlap or coverage error")
    if len(selected_lost) != 560 or len(test_standard) != 840:
        raise ValueError("STT test 2:3 contract violated")
    if len(train) != 7547 or len(test) != 1900:
        raise ValueError(f"Unexpected v2 sizes: train={len(train)} test={len(test)}")

    # Trajectory signatures remain exclusive. Source-batch exclusivity has one
    # explicit exception: loststart is stratified within its four batches.
    trajectory_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    for split, values in (("train", train), ("test", test)):
        for item in values:
            trajectory_splits[(item["scene"], item["trajectory_signature"])].add(split)
    leaking = [key for key, values in trajectory_splits.items() if len(values) > 1]
    if leaking:
        raise ValueError(f"Trajectory leakage in v2: {leaking[:5]}")

    output = args.output.resolve()
    summary = {
        "schema_version": "data_arr_train_test_no_val_v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_v1": str(args.v1.resolve()),
        "core_total": 9447,
        "train": len(train),
        "test": len(test),
        "val": 0,
        "train_by_kind": dict(Counter(item["data_kind"] for item in train)),
        "test_by_kind": dict(Counter(item["data_kind"] for item in test)),
        "stt_test_loststart": len(selected_lost),
        "stt_test_standard": len(test_standard),
        "stt_test_ratio": "2:3",
        "dt_test_standard": len(dt_standard_test),
        "dt_test_released_to_train_for_rounding": released_dt["episode_id"],
        "loststart_released_to_train": len(released_lost),
        "old_val_moved_to_train_or_standard_test": len(old_val),
        "old_val_added_to_standard_test": len(added_standard),
        "old_val_added_to_train": len(val_dt) + len(remaining_val_stt),
        "pseudo_auxiliary_train": len(pseudo),
        "unresolved_excluded": len(unresolved),
        "source_batch_exception": (
            "loststart batches span train/test for exact 2:3; one DT test episode "
            "moves to train for exact total=1900"
        ),
        "trajectory_leakage": 0,
    }
    assignment_lines = [
        f"{item['episode_id']}\t{split}"
        for split, values in (("train", train), ("test", test))
        for item in values
    ]
    summary["assignment_sha256"] = hashlib.sha256(
        "\n".join(sorted(assignment_lines)).encode()
    ).hexdigest()
    manifest = {
        **summary,
        "waypoint8_processed_root": str(processed_root),
        "waypoint10_processed_root": None,
        "train": sorted(train, key=lambda item: item["episode_id"]),
        "test": sorted(test, key=lambda item: item["episode_id"]),
        "test_stt_loststart": sorted(set_split(selected_lost, "test"), key=lambda item: item["episode_id"]),
        "test_stt_standard": sorted(set_split(test_standard, "test"), key=lambda item: item["episode_id"]),
        "test_dt_standard": sorted(set_split(dt_standard_test, "test"), key=lambda item: item["episode_id"]),
        "train_loststart_remainder": sorted(set_split(released_lost, "train"), key=lambda item: item["episode_id"]),
        "auxiliary_pseudo_train": pseudo,
        "excluded_unresolved": unresolved,
    }
    write(output / "split_manifest.json", manifest)
    write(output / "summary.json", summary)
    for name, values in (
        ("train", train), ("test", test),
        ("test_stt_loststart", selected_lost), ("test_stt_standard", test_standard),
        ("test_dt_standard", dt_standard_test),
        ("train_loststart_remainder", released_lost),
        ("auxiliary_pseudo_train", pseudo), ("excluded_unresolved", unresolved),
    ):
        write(output / f"{name}.json", sorted(values, key=lambda item: item["episode_id"]))
    for split, values in (("train", train), ("test", test)):
        write(output / "waypoint8" / f"{split}_episodes.json", processed(values, processed_root, 8))
        write(output / "waypoint10" / f"{split}_expected.json", processed(values, processed_root, 10))
    checksums = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.json":
            checksums[str(path.relative_to(output))] = hashlib.sha256(path.read_bytes()).hexdigest()
    write(output / "SHA256SUMS.json", checksums)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
