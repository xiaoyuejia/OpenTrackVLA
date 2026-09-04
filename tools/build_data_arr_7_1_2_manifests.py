#!/usr/bin/env python3
"""Build representation-independent 7:1:2 manifests for data_arr.

Hard policy:
- ``lostmid`` episodes and their leakage-linked components are train-only.
- ``loststart`` episodes and their leakage-linked components are test-only.
- fine-tuned-YOLO pseudo-bbox episodes are auxiliary-train-only.
- unresolved ``stt_camera_mix__new3`` bbox/visibility conflicts are excluded.

The split is frozen at the raw episode-key level.  The existing 8-waypoint
JSONLs and future 10-waypoint JSONLs must use the same assignments.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "data_arr_episode_split_7_1_2_v1"
RATIOS = {"train": 0.70, "val": 0.10, "test": 0.20}
SPLITS = tuple(RATIOS)


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def wrap_degrees(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


class DSU:
    def __init__(self, values: Iterable[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, first: str, second: str) -> None:
        a, b = self.find(first), self.find(second)
        if a == b:
            return
        if a > b:
            a, b = b, a
        self.parent[b] = a


@dataclass
class Episode:
    episode_id: str
    data_kind: str
    scene: str
    source_batch: str
    stem: str
    raw_relative_dir: str
    drone_info: str
    robotdog_info: str
    status_json: str
    drone_video: str
    robotdog_video: str
    waypoint8_jsonl: str
    quality_tier: str
    hard_rule: str | None
    trajectory_signature: str = ""
    group_id: str = ""
    split: str = ""

    def rich_item(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "data_kind": self.data_kind,
            "scene": self.scene,
            "source_batch": self.source_batch,
            "stem": self.stem,
            "raw_relative_dir": self.raw_relative_dir,
            "drone_info": self.drone_info,
            "robotdog_info": self.robotdog_info,
            "status_json": self.status_json,
            "drone_video": self.drone_video,
            "robotdog_video": self.robotdog_video,
            "trajectory_signature": self.trajectory_signature,
            "leakage_group": self.group_id,
            "quality_tier": self.quality_tier,
            "hard_rule": self.hard_rule,
            "split": self.split,
            "waypoint8_relative_jsonl": self.waypoint8_jsonl,
            "waypoint10_expected_relative_jsonl": self.waypoint8_jsonl,
        }

    def processed_item(self, processed_root: Path, waypoint_count: int) -> dict[str, Any]:
        relative = self.waypoint8_jsonl
        item = {
            "episode_id": self.episode_id,
            "scene": self.scene,
            "stem": self.stem,
            "data_kind": self.data_kind,
            "source_batch": self.source_batch,
            "relative_jsonl": relative,
            "waypoint_count": waypoint_count,
            "quality_tier": self.quality_tier,
            "leakage_group": self.group_id,
        }
        if waypoint_count == 8:
            item["jsonl"] = str((processed_root / relative).resolve())
        else:
            item["jsonl"] = None
            item["build_status"] = "pending_10_waypoint_rebuild"
        return item


@dataclass
class Component:
    group_id: str
    episodes: list[Episode]
    forced_split: str | None = None
    core_count: int = 0
    core_scene: Counter[str] = field(default_factory=Counter)
    core_kind: Counter[str] = field(default_factory=Counter)


def classify_batch(batch: str) -> tuple[str, str | None]:
    lowered = batch.lower()
    if "stt_camera_mix__new3" in lowered:
        return "unresolved_bbox_visibility", None
    if "yolobbox" in lowered:
        return "pseudo_bbox_yolo_finetuned", "train"
    if "loststart" in lowered:
        return "core_audited", "test"
    if "lostmid" in lowered:
        return "core_audited", "train"
    return "core_audited", None


def discover(source_root: Path, processed_root: Path) -> list[Episode]:
    episodes: list[Episode] = []
    for drone_path in sorted(source_root.rglob("*_drone_info.json"), key=lambda value: str(value)):
        rel = drone_path.relative_to(source_root)
        if len(rel.parts) != 4 or rel.parts[0] not in {"dt", "stt", "cyj_dt", "cyj_stt"}:
            raise ValueError(f"Unexpected organized data_arr layout: {rel}")
        collection, scene, batch, filename = rel.parts
        stem = filename.removesuffix("_drone_info.json")
        kind = "dt" if collection in {"dt", "cyj_dt"} else "stt"
        parent = drone_path.parent
        dog = parent / f"{stem}_robotdog_info.json"
        status = parent / f"{stem}.json"
        drone_video = parent / f"{stem}_drone.mp4"
        dog_video = parent / f"{stem}_robotdog.mp4"
        missing = [path for path in (dog, status, drone_video, dog_video) if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Incomplete episode {rel}: {missing}")
        waypoint8 = Path("jsonl") / kind / scene / batch / f"{stem}.jsonl"
        if not (processed_root / waypoint8).is_file():
            raise FileNotFoundError(f"Missing existing 8-waypoint JSONL: {processed_root / waypoint8}")
        quality, forced = classify_batch(batch)
        episode_id = f"{kind}/{scene}/{batch}/{stem}"
        raw_dir = Path(collection) / scene / batch
        episodes.append(
            Episode(
                episode_id=episode_id,
                data_kind=kind,
                scene=scene,
                source_batch=batch,
                stem=stem,
                raw_relative_dir=str(raw_dir),
                drone_info=str(raw_dir / f"{stem}_drone_info.json"),
                robotdog_info=str(raw_dir / f"{stem}_robotdog_info.json"),
                status_json=str(raw_dir / f"{stem}.json"),
                drone_video=str(raw_dir / f"{stem}_drone.mp4"),
                robotdog_video=str(raw_dir / f"{stem}_robotdog.mp4"),
                waypoint8_jsonl=str(waypoint8),
                quality_tier=quality,
                hard_rule=forced,
            )
        )
    ids = [episode.episode_id for episode in episodes]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate canonical episode IDs")
    return episodes


def trajectory_signature(source_root: Path, episode: Episode) -> tuple[str, str]:
    path = source_root / episode.drone_info
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Empty/non-list info JSON: {path}")
    points: list[list[int]] = []
    origin: tuple[float, float, float, float] | None = None
    for index, row in enumerate(rows):
        if index % 5 != 0 and index != len(rows) - 1:
            continue
        pose = row.get("target_pose_after_action") or row.get("target_pose")
        if not isinstance(pose, list) or len(pose) < 5:
            continue
        values = [float(value) for value in pose[:6]]
        if not all(math.isfinite(value) for value in values):
            continue
        if origin is None:
            origin = (values[0], values[1], values[2], values[4])
        assert origin is not None
        # Translation-invariant within one scene; 10 cm / 1 degree quantization
        # links the same authored target motion despite harmless float jitter.
        points.append(
            [
                round((values[0] - origin[0]) / 10.0),
                round((values[1] - origin[1]) / 10.0),
                round((values[2] - origin[2]) / 10.0),
                round(wrap_degrees(values[4] - origin[3])),
            ]
        )
    if not points:
        raise ValueError(f"No valid target poses: {path}")
    payload = json.dumps(
        {"scene": episode.scene, "frames": len(rows), "points": points},
        separators=(",", ":"),
        sort_keys=True,
    )
    return episode.episode_id, stable_hash(payload)[:24]


def add_trajectory_signatures(source_root: Path, episodes: list[Episode], workers: int) -> None:
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = executor.map(
            lambda item: trajectory_signature(source_root, item),
            episodes,
            chunksize=8,
        )
        lookup = dict(results)
    if len(lookup) != len(episodes):
        raise ValueError("Trajectory signature result count mismatch")
    for episode in episodes:
        episode.trajectory_signature = lookup[episode.episode_id]


def reuse_trajectory_signatures(path: Path, episodes: list[Episode]) -> bool:
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    lookup: dict[str, str] = {}
    for section in ("train", "val", "test", "auxiliary_pseudo_train", "excluded_unresolved"):
        for item in payload.get(section, []):
            episode_id = str(item.get("episode_id", ""))
            signature = str(item.get("trajectory_signature", ""))
            if episode_id and signature:
                lookup[episode_id] = signature
    if len(lookup) != len(episodes) or set(lookup) != {episode.episode_id for episode in episodes}:
        return False
    for episode in episodes:
        episode.trajectory_signature = lookup[episode.episode_id]
    return True


def build_components(episodes: list[Episode]) -> list[Component]:
    eligible = [episode for episode in episodes if episode.quality_tier != "unresolved_bbox_visibility"]
    dsu = DSU(episode.episode_id for episode in eligible)
    by_batch: dict[tuple[str, str], list[str]] = defaultdict(list)
    by_trajectory: dict[tuple[str, str], list[str]] = defaultdict(list)
    for episode in eligible:
        by_batch[(episode.data_kind, episode.source_batch)].append(episode.episode_id)
        by_trajectory[(episode.scene, episode.trajectory_signature)].append(episode.episode_id)
    for values in (*by_batch.values(), *by_trajectory.values()):
        first = values[0]
        for value in values[1:]:
            dsu.union(first, value)
    grouped: dict[str, list[Episode]] = defaultdict(list)
    for episode in eligible:
        grouped[dsu.find(episode.episode_id)].append(episode)

    components: list[Component] = []
    for values in grouped.values():
        values.sort(key=lambda item: item.episode_id)
        group_id = stable_hash("\n".join(item.episode_id for item in values))[:20]
        hard = {item.hard_rule for item in values if item.hard_rule is not None}
        if len(hard) > 1:
            details = Counter(item.hard_rule for item in values if item.hard_rule)
            raise ValueError(f"Conflicting hard rules in leakage group {group_id}: {details}")
        forced = next(iter(hard)) if hard else None
        component = Component(group_id=group_id, episodes=values, forced_split=forced)
        for episode in values:
            episode.group_id = group_id
            if episode.quality_tier == "core_audited":
                component.core_count += 1
                component.core_scene[episode.scene] += 1
                component.core_kind[episode.data_kind] += 1
        components.append(component)
    return components


def rounded_targets(total: int) -> dict[str, int]:
    train = round(total * RATIOS["train"])
    val = round(total * RATIOS["val"])
    return {"train": train, "val": val, "test": total - train - val}


def target_tables(episodes: list[Episode]) -> tuple[dict[str, int], dict[str, dict[str, int]], dict[str, dict[str, int]]]:
    core = [episode for episode in episodes if episode.quality_tier == "core_audited"]
    global_target = rounded_targets(len(core))
    scene_target = {
        key: rounded_targets(value)
        for key, value in Counter(episode.scene for episode in core).items()
    }
    kind_target = {
        key: rounded_targets(value)
        for key, value in Counter(episode.data_kind for episode in core).items()
    }
    return global_target, scene_target, kind_target


def assignment_cost(
    counts: dict[str, int],
    scene_counts: dict[str, Counter[str]],
    kind_counts: dict[str, Counter[str]],
    global_target: dict[str, int],
    scene_target: dict[str, dict[str, int]],
    kind_target: dict[str, dict[str, int]],
) -> float:
    cost = 0.0
    for split in SPLITS:
        scale = max(global_target[split], 1)
        delta = counts[split] - global_target[split]
        cost += 12.0 * delta * delta / scale
    for scene, targets in scene_target.items():
        total = max(sum(targets.values()), 1)
        for split in SPLITS:
            delta = scene_counts[scene][split] - targets[split]
            cost += 1.5 * delta * delta / total
    for kind, targets in kind_target.items():
        total = max(sum(targets.values()), 1)
        for split in SPLITS:
            delta = kind_counts[kind][split] - targets[split]
            cost += 4.0 * delta * delta / total
    return cost


def apply_component(
    component: Component,
    split: str,
    sign: int,
    counts: dict[str, int],
    scene_counts: dict[str, Counter[str]],
    kind_counts: dict[str, Counter[str]],
) -> None:
    counts[split] += sign * component.core_count
    for scene, value in component.core_scene.items():
        scene_counts[scene][split] += sign * value
    for kind, value in component.core_kind.items():
        kind_counts[kind][split] += sign * value


def assign_components(components: list[Component], episodes: list[Episode], seed: int) -> None:
    global_target, scene_target, kind_target = target_tables(episodes)
    counts = {split: 0 for split in SPLITS}
    scene_counts: dict[str, Counter[str]] = defaultdict(Counter)
    kind_counts: dict[str, Counter[str]] = defaultdict(Counter)
    assigned: dict[str, str] = {}

    for component in components:
        if component.forced_split is not None:
            assigned[component.group_id] = component.forced_split
            apply_component(component, component.forced_split, 1, counts, scene_counts, kind_counts)

    unforced = [component for component in components if component.forced_split is None]
    unforced.sort(key=lambda item: (-item.core_count, stable_hash(f"{seed}:{item.group_id}")))
    for component in unforced:
        choices = []
        for split in SPLITS:
            apply_component(component, split, 1, counts, scene_counts, kind_counts)
            cost = assignment_cost(
                counts, scene_counts, kind_counts, global_target, scene_target, kind_target
            )
            apply_component(component, split, -1, counts, scene_counts, kind_counts)
            choices.append((cost, SPLITS.index(split), split))
        split = min(choices)[2]
        assigned[component.group_id] = split
        apply_component(component, split, 1, counts, scene_counts, kind_counts)

    # Deterministic hill climbing moves whole leakage groups only.
    rng = random.Random(seed)
    candidates = unforced.copy()
    current = assignment_cost(
        counts, scene_counts, kind_counts, global_target, scene_target, kind_target
    )
    for _ in range(25000):
        component = candidates[rng.randrange(len(candidates))]
        old = assigned[component.group_id]
        new = SPLITS[rng.randrange(len(SPLITS))]
        if old == new:
            continue
        apply_component(component, old, -1, counts, scene_counts, kind_counts)
        apply_component(component, new, 1, counts, scene_counts, kind_counts)
        proposed = assignment_cost(
            counts, scene_counts, kind_counts, global_target, scene_target, kind_target
        )
        if proposed + 1e-12 < current:
            assigned[component.group_id] = new
            current = proposed
        else:
            apply_component(component, new, -1, counts, scene_counts, kind_counts)
            apply_component(component, old, 1, counts, scene_counts, kind_counts)

    for component in components:
        split = assigned[component.group_id]
        for episode in component.episodes:
            episode.split = split


def validate(episodes: list[Episode]) -> dict[str, Any]:
    unresolved = [episode for episode in episodes if episode.quality_tier == "unresolved_bbox_visibility"]
    eligible = [episode for episode in episodes if episode not in unresolved]
    if any(episode.split not in SPLITS for episode in eligible):
        raise ValueError("Eligible episode without split")
    if any(episode.split for episode in unresolved):
        raise ValueError("Unresolved episode accidentally assigned")
    if any("lostmid" in episode.source_batch.lower() and episode.split != "train" for episode in episodes):
        raise ValueError("lostmid hard rule violated")
    if any("loststart" in episode.source_batch.lower() and episode.split != "test" for episode in episodes):
        raise ValueError("loststart hard rule violated")
    if any(episode.quality_tier == "pseudo_bbox_yolo_finetuned" and episode.split != "train" for episode in episodes):
        raise ValueError("Pseudo-bbox train-only rule violated")
    group_splits: dict[str, set[str]] = defaultdict(set)
    trajectory_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    batch_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    for episode in eligible:
        group_splits[episode.group_id].add(episode.split)
        trajectory_splits[(episode.scene, episode.trajectory_signature)].add(episode.split)
        batch_splits[(episode.data_kind, episode.source_batch)].add(episode.split)
    if any(len(values) != 1 for values in group_splits.values()):
        raise ValueError("Leakage group split across partitions")
    if any(len(values) != 1 for values in trajectory_splits.values()):
        raise ValueError("Trajectory signature split across partitions")
    if any(len(values) != 1 for values in batch_splits.values()):
        raise ValueError("Source batch split across partitions")
    core = [episode for episode in episodes if episode.quality_tier == "core_audited"]
    return {
        "all_discovered": len(episodes),
        "core_total": len(core),
        "pseudo_train": sum(
            episode.quality_tier == "pseudo_bbox_yolo_finetuned" for episode in episodes
        ),
        "unresolved_excluded": len(unresolved),
        "core_split_counts": dict(Counter(episode.split for episode in core)),
        "core_split_ratios": {
            split: Counter(episode.split for episode in core)[split] / max(len(core), 1)
            for split in SPLITS
        },
        "lostmid_train": sum("lostmid" in episode.source_batch.lower() for episode in episodes),
        "loststart_test": sum("loststart" in episode.source_batch.lower() for episode in episodes),
        "leakage_groups": len(group_splits),
        "trajectory_signatures": len(trajectory_splits),
        "source_batches": len(batch_splits),
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_outputs(
    output_root: Path,
    source_root: Path,
    processed_root: Path,
    episodes: list[Episode],
    summary: dict[str, Any],
    seed: int,
) -> None:
    rich = {
        split: sorted(
            [episode.rich_item() for episode in episodes if episode.split == split and episode.quality_tier == "core_audited"],
            key=lambda item: item["episode_id"],
        )
        for split in SPLITS
    }
    pseudo = sorted(
        [episode.rich_item() for episode in episodes if episode.quality_tier == "pseudo_bbox_yolo_finetuned"],
        key=lambda item: item["episode_id"],
    )
    unresolved = sorted(
        [episode.rich_item() for episode in episodes if episode.quality_tier == "unresolved_bbox_visibility"],
        key=lambda item: item["episode_id"],
    )
    policy = {
        "ratios": RATIOS,
        "seed": seed,
        "grouping": [
            "same data_kind/source_batch is indivisible",
            "same scene/quantized target-trajectory signature is indivisible",
        ],
        "hard_rules": {
            "lostmid": "train_only",
            "loststart": "test_only",
            "pseudo_bbox_yolo_finetuned": "auxiliary_train_only",
            "unresolved_bbox_visibility": "excluded",
        },
        "trajectory_signature": {
            "source": "drone target_pose_after_action fallback target_pose",
            "sampling": "every 5 frames plus final frame",
            "normalization": "scene-scoped, relative to first pose",
            "quantization": "10 cm xyz, 1 degree yaw",
        },
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root),
        "waypoint8_processed_root": str(processed_root),
        "waypoint10_processed_root": None,
        "policy": policy,
        "summary": summary,
        **rich,
        "auxiliary_pseudo_train": pseudo,
        "excluded_unresolved": unresolved,
    }
    write_json(output_root / "split_manifest.json", manifest)
    write_json(output_root / "summary.json", {"schema_version": SCHEMA_VERSION, "policy": policy, **summary})
    for split in SPLITS:
        write_json(output_root / f"{split}.json", rich[split])
        write_json(
            output_root / "waypoint8" / f"{split}_episodes.json",
            [
                episode.processed_item(processed_root, 8)
                for episode in sorted(episodes, key=lambda item: item.episode_id)
                if episode.split == split and episode.quality_tier == "core_audited"
            ],
        )
        write_json(
            output_root / "waypoint10" / f"{split}_expected.json",
            [
                episode.processed_item(processed_root, 10)
                for episode in sorted(episodes, key=lambda item: item.episode_id)
                if episode.split == split and episode.quality_tier == "core_audited"
            ],
        )
    write_json(output_root / "auxiliary_pseudo_train.json", pseudo)
    write_json(output_root / "excluded_unresolved.json", unresolved)
    write_json(
        output_root / "train_lostmid.json",
        sorted(
            [
                episode.rich_item()
                for episode in episodes
                if episode.quality_tier == "core_audited"
                and episode.split == "train"
                and "lostmid" in episode.source_batch.lower()
            ],
            key=lambda item: item["episode_id"],
        ),
    )
    write_json(
        output_root / "test_loststart.json",
        sorted(
            [
                episode.rich_item()
                for episode in episodes
                if episode.quality_tier == "core_audited"
                and episode.split == "test"
                and "loststart" in episode.source_batch.lower()
            ],
            key=lambda item: item["episode_id"],
        ),
    )
    write_json(
        output_root / "test_standard.json",
        sorted(
            [
                episode.rich_item()
                for episode in episodes
                if episode.quality_tier == "core_audited"
                and episode.split == "test"
                and "loststart" not in episode.source_batch.lower()
            ],
            key=lambda item: item["episode_id"],
        ),
    )
    write_json(
        output_root / "waypoint8" / "auxiliary_pseudo_train_episodes.json",
        [
            episode.processed_item(processed_root, 8)
            for episode in sorted(episodes, key=lambda item: item.episode_id)
            if episode.quality_tier == "pseudo_bbox_yolo_finetuned"
        ],
    )

    with (output_root / "scene_counts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["scene", "core_total", "train", "val", "test", "lostmid", "loststart"],
        )
        writer.writeheader()
        for scene in sorted({episode.scene for episode in episodes}):
            values = [episode for episode in episodes if episode.scene == scene and episode.quality_tier == "core_audited"]
            writer.writerow(
                {
                    "scene": scene,
                    "core_total": len(values),
                    "train": sum(episode.split == "train" for episode in values),
                    "val": sum(episode.split == "val" for episode in values),
                    "test": sum(episode.split == "test" for episode in values),
                    "lostmid": sum("lostmid" in episode.source_batch.lower() for episode in values),
                    "loststart": sum("loststart" in episode.source_batch.lower() for episode in values),
                }
            )

    checksums = {}
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.json":
            checksums[str(path.relative_to(output_root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(output_root / "SHA256SUMS.json", checksums)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/data/hdt/ntv_data/sim_data/data_arr"),
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("/data/hdt/ntv_data/data/cyj_data_arr_processed"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("manifests/data_arr_7_1_2_v1"),
    )
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    source_root = args.source_root.expanduser().resolve()
    processed_root = args.processed_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    episodes = discover(source_root, processed_root)
    print(f"[discover] episodes={len(episodes)}", flush=True)
    if reuse_trajectory_signatures(output_root / "split_manifest.json", episodes):
        print("[signature] reused existing manifest", flush=True)
    else:
        add_trajectory_signatures(source_root, episodes, max(1, args.workers))
        print("[signature] complete", flush=True)
    components = build_components(episodes)
    print(f"[group] components={len(components)}", flush=True)
    assign_components(components, episodes, args.seed)
    summary = validate(episodes)
    write_outputs(output_root, source_root, processed_root, episodes, summary, args.seed)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"[write] {output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
