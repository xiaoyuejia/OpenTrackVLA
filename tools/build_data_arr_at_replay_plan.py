#!/usr/bin/env python3
"""Build deterministic AT replay/evaluation configs from the frozen DT split."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


INDOOR_MARKERS = (
    "FlexibleRoom", "InteriorDemo", "Demo_Roof", "JapanTrainStation",
    "KoreanPalace", "Brass_Palace", "EnglishCollege", "Medieval_Castle",
)
LARGE_OPEN_MARKERS = (
    "Desert", "Grass_Hills", "SnowMap", "Arctic", "StonePineForest",
    "ForestGasStation", "PlanetOutDoor", "Real_Landscape", "Stadium",
)


def digest(value: str) -> int:
    return int(hashlib.sha256(value.encode()).hexdigest()[:16], 16)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path, default=Path("manifests/data_arr_train_test_v2/split_manifest.json"))
    parser.add_argument("--omtrack-at-instructions", type=Path, default=Path("/data/hdt/code_raw/OmTrackVLA/data/datasets/track/AT/train/instructions.json"))
    parser.add_argument("--output-root", type=Path, default=Path("manifests/data_arr_at_v1"))
    parser.add_argument("--gpus", default="0,1,6")
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()
    source = json.loads(args.split_manifest.read_text())
    instructions = json.loads(args.omtrack_at_instructions.read_text())
    if not isinstance(instructions, list) or not instructions:
        raise ValueError("AT instruction source must be a non-empty list")
    gpus = [int(value) for value in args.gpus.split(",")]
    output = args.output_root.resolve(); output.mkdir(parents=True, exist_ok=True)

    all_items = []
    for split in ("train", "test"):
        for item in source[split]:
            if item["data_kind"] != "dt":
                continue
            key = item["episode_id"]
            indoor = any(marker in item["scene"] for marker in INDOOR_MARKERS)
            # OmTrack train averages ~4 humans and caps at 7. Current mostly-open
            # UnrealZoo scenes use the cap; constrained indoor scenes use 4 total.
            large_open = any(marker in item["scene"] for marker in LARGE_OPEN_MARKERS)
            distractors = 5 if indoor else 9 if large_open else 7
            h = digest(f"{args.seed}:{key}")
            # Separated target-local slots prevent UE character capsules from
            # spawning on top of one another.  The first five are already
            # balanced for indoor episodes; outdoor episodes add the outer
            # slots while remaining within roughly six metres of the target.
            formation_slots = [
                (2.6, -2.2), (2.6, 0.0), (2.6, 2.2),
                (4.2, -1.25), (4.2, 1.25),
                (4.3, -3.35), (4.3, 3.35),
                (5.5, -1.55), (5.5, 1.55),
            ][:distractors]
            common_phase = round((h % 6283) / 1000.0, 3)
            config = {
                **item,
                "source_split": split,
                "task_variant": "at_rerendered_distinct_appearance",
                "instruction": instructions[h % len(instructions)],
                "appearance_id": 1 + (h % 18),
                "appearance_mode": "distinct_deterministic",
                "human_appearance_ids": [1 + ((h + index) % 18) for index in range(1 + distractors)],
                "human_count": 1 + distractors,
                "distractor_count": distractors,
                "distractor_seed": h % (2**31 - 1),
                "target_follower_geometry_source": "recorded_pose_per_frame",
                "distractor_trajectory_source": "target_relative_forward_only_smooth_perturbation",
                "target_actor_slot": 0,
                "robotdog_actor_slot": 1,
                "drone_actor_slot": 2,
                "distractor_actor_slots": list(range(3, 3 + distractors)),
                "all_human_same_appearance": False,
                "waypoint_contracts": [8, 10],
                "distractor_formation": [
                    {
                        "forward_m": formation_slots[index - 1][0],
                        "right_m": formation_slots[index - 1][1],
                        "jitter_forward_m": round((((h >> (index * 3)) % 100) / 100.0 - 0.5) * 0.5, 3),
                        "jitter_right_m": round((((h >> (index * 5 + 7)) % 100) / 100.0 - 0.5) * 0.5, 3),
                        # A shared, large low-frequency drift perturbs the
                        # target path without making neighbouring actors cross.
                        "jitter_forward_amplitude_m": 1.0,
                        "jitter_right_amplitude_m": 1.35,
                        "jitter_forward_secondary_m": 0.45,
                        "jitter_right_secondary_m": 0.55,
                        "jitter_period_s": round(4.8 + 0.55 * ((index - 1) % 6), 3),
                        "jitter_phase_rad": round(common_phase + 0.67 * (index - 1), 3),
                    }
                    for index in range(1, 1 + distractors)
                ],
            }
            all_items.append(config)

    train = [item for item in all_items if item["source_split"] == "train"]
    eval_items = [item for item in all_items if item["source_split"] != "train"]
    shards = {gpu: [] for gpu in gpus}
    loads = {gpu: 0 for gpu in gpus}
    # All current DT episodes are 300 frames, but keep size-aware scheduling.
    for item in sorted(train, key=lambda value: (value["scene"], value["episode_id"])):
        gpu = min(gpus, key=lambda value: (loads[value], value))
        item = {**item, "assigned_gpu": gpu}
        shards[gpu].append(item); loads[gpu] += 300

    for gpu in gpus:
        (output / f"train_gpu{gpu}.json").write_text(json.dumps(shards[gpu], ensure_ascii=False, indent=2)+"\n")
    # Evaluation is also resumable and can be spread across the same workers;
    # unlike training it contains exactly the frozen 500 DT test episodes.
    eval_shards = {gpu: [] for gpu in gpus}
    for index, item in enumerate(sorted(eval_items, key=lambda value: (value["scene"], value["episode_id"]))):
        eval_shards[gpus[index % len(gpus)]].append({**item, "assigned_gpu": gpus[index % len(gpus)]})
    for gpu in gpus:
        (output / f"evaluation_gpu{gpu}.json").write_text(json.dumps(eval_shards[gpu], ensure_ascii=False, indent=2)+"\n")
    (output / "train.json").write_text(json.dumps(train, ensure_ascii=False, indent=2)+"\n")
    (output / "evaluation.json").write_text(json.dumps(eval_items, ensure_ascii=False, indent=2)+"\n")
    summary = {
        "schema_version": "data_arr_at_rerender_plan_v1",
        "source_split_manifest": str(args.split_manifest.resolve()),
        "instruction_source": str(args.omtrack_at_instructions.resolve()),
        "gpus": gpus,
        "train_episodes": len(train),
        "evaluation_json_only_episodes": len(eval_items),
        "train_shards": {str(gpu): len(shards[gpu]) for gpu in gpus},
        "evaluation_shards": {str(gpu): len(eval_shards[gpu]) for gpu in gpus},
        "estimated_frames_per_gpu": {str(gpu): loads[gpu] for gpu in gpus},
        "distractor_count_distribution": dict(Counter(item["distractor_count"] for item in all_items)),
        "note": "Evaluation configs require an AT-aware runtime; editing legacy status JSON alone has no effect.",
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
