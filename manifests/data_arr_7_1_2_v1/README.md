# data_arr 7:1:2 split manifest v1

This directory freezes episode assignments independently of waypoint
representation. The current 8-waypoint JSONLs and the future 10-waypoint
rebuild must use exactly the same `episode_id -> split` mapping.

## Core split

| Split | Episodes | Ratio |
|---|---:|---:|
| Train | 6,598 | 69.842% |
| Validation | 952 | 10.077% |
| Test | 1,897 | 20.080% |
| Core total | 9,447 | 100% |

The ratios are group-constrained rather than row-random, so an exact
`6613/945/1889` split is intentionally not forced.

## Hard scenario rules

- All 305 `lostmid` episodes are in train.
- All 913 `loststart` episodes are in test.
- Test is also exported as:
  - `test_loststart.json`: 913 episodes;
  - `test_standard.json`: 984 episodes.
- The two subsets should be reported separately as well as together. Otherwise
  the aggregate test score gives nearly half of its weight to `loststart`.

## Quality tiers

- `core_audited`: 9,447 episodes used by the formal 7:1:2 split. “Audited”
  means no known structural/semantic issue under the existing checks; it does
  not claim pixel-perfect GT for every bbox.
- `pseudo_bbox_yolo_finetuned`: 376 episodes, exported only as
  `auxiliary_pseudo_train.json`. They are not part of validation/test or the
  core ratio.
- `unresolved_bbox_visibility`: 355 episodes, exported in
  `excluded_unresolved.json` and assigned to no split.

## Leakage prevention

The generator treats each connected component as indivisible when episodes
share either:

1. the same `data_kind/source_batch`; or
2. the same scene-scoped target-trajectory signature.

The signature samples target pose every five frames, expresses motion relative
to the first target pose, and quantizes it at 10 cm / 1 degree. No source batch
or trajectory signature crosses train/val/test.

This is a seen-scene, trajectory-disjoint split. It is not an unseen-scene
benchmark: the `lostmid=train` and `loststart=test` rules place different
episodes from many of the same scenes on opposite sides. Unseen-scene
generalization needs a separate challenge manifest.

Of 41 scenes, 39 occur in all three core splits. `AbandonedDistrict` has only
one core episode, and `InteriorDemo_NEW` has 11 episodes tied by grouping; both
remain train-only rather than breaking leakage groups to manufacture coverage.

## Files

- `split_manifest.json`: full policy, summary, and all quality tiers.
- `train.json`, `val.json`, `test.json`: rich representation-independent core
  entries.
- `train_lostmid.json`, `test_loststart.json`, `test_standard.json`: scenario
  subsets without changing the parent split.
- `waypoint8/*_episodes.json`: existing JSONL paths, verified to exist.
- `waypoint10/*_expected.json`: same assignments with `jsonl=null` and
  `build_status=pending_10_waypoint_rebuild`.
- `scene_counts.csv`: scene-level distribution audit.
- `SHA256SUMS.json`: hashes for every generated manifest file.

Regenerate deterministically with:

```bash
PYTHONDONTWRITEBYTECODE=1 \
  /home/hdt/miniconda3/envs/omtracknew/bin/python \
  tools/build_data_arr_7_1_2_manifests.py \
  --output-root manifests/data_arr_7_1_2_v1 \
  --seed 20260825 --workers 24
```

The generator reads raw metadata and existing 8-waypoint paths but never moves
or modifies source data.

