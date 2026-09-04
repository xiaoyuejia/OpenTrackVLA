# UnrealZoo Existing Dataset Audit

Date: 2026-07-13

## Scope

This audit covers the three existing non-fixed-step collections:

- `/data/hdt/ntv_data/sim_data/data3047`
- `/data/hdt/ntv_data/sim_data/data7_7`
- `/data/hdt/ntv_data/sim_data/data7_8hand`

`data7_8hand_by_source` is a symlink/index view of `data7_8hand`, not an
independent dataset. The consolidated `/data/hdt/ntv_data/sim_data/data7_8`
contains the same 1,813 hand-collected episodes in a flattened layout.

## Important Time Semantics

The following quantities are not interchangeable:

1. JSON `dt` or `training_dt_s`: the nominal training period.
2. `effective_dt_s`: the duration represented by the recorded velocity label.
3. Observation wall interval: host time between adjacent recorded snapshots.
4. Simulated motion between adjacent observations.

For navigation supervision, a row is trustworthy only when its velocity and
effective duration reconstruct the recorded after-action pose. Wall-clock
delay alone does not prove label corruption, because the agent may be stopped
while image queries and file I/O run.

## Summary

| Dataset | Episodes | Rows per agent | Main timing | Bbox validity | Recommended role |
|---|---:|---:|---|---:|---|
| `data3047` | 3,047 | 668,952 | effective dt P50 0.333 s, P95 0.398 s | drone 98.2%, dog 99.5% | visual/ROI pretraining; optional separate time-aware objective |
| `data7_7` | 1,316 | 263,200 | effective dt P50 0.100 s, P95 0.133 s | about 99.75% | primary automatic navigation supervision after filtering |
| `data7_8hand` new format | 918 | 275,200 | effective dt P50 0.100 s, P95 0.133 s | mixed by batch | primary manual navigation supervision after filtering |
| `data7_8hand` old format | 895 | 179,100 | inferred/calibrated dt P50 0.200 s | about 99.0% | visual/ROI only; do not use current navigation labels |

## data3047

- 2,419 episodes use `path_loop`; 628 use `keyboard_wasd`.
- 461,919 rows have both an effective duration and an after-action pose.
- 206,633 old rows have neither reliable effective duration nor after-action
  pose; another 400 rows have an after pose but no effective duration.
- Mean effective duration is 0.317 s. This is low-frequency data, not 10 Hz
  history, even though velocity-times-duration reconstructs the available
  after-action pose.
- Drone and dog visibility rates are 96.8% and 99.1% respectively.

Use this collection to improve global scene encoding, person appearance,
ROI/bbox understanding, visibility recovery, and scene diversity. Do not mix
its row-indexed 31-frame history directly with a 10 Hz temporal model. The old
206,633-row subgroup must have navigation loss disabled entirely.

## data7_7

- All 1,316 episodes are `path_loop`, have 200 rows, and have
  `snap_heading=false`.
- Effective duration: mean 0.1045 s, P05 0.0991 s, P50 0.1000 s,
  P95 0.1333 s, P99 0.1333 s.
- Observation wall interval is slower: mean 0.315 s and P95 0.582 s.
  However, after-action pose to next observation drift is only 2.56 cm mean
  for drone and 2.63 cm for dog. This indicates that most wall delay is not
  equivalent to uncontrolled agent motion.
- `base_velocity * effective_dt_s` reconstructs after-action pose to numerical
  precision.
- Drone speed mean is 0.749 m/s, P95 1.082 m/s; absolute drone yaw-rate P95 is
  0.145 rad/s.
- There are 1,201 unique target trajectories among 1,316 episodes. At least
  115 target trajectories repeat, so data splits must group by trajectory
  signature.

Recommended filter (`dt` 0.09-0.14 s, no collision, both agents' after-action
to-next-observation gap at most 5 cm):

- 255,436 / 263,200 individually valid rows (97.05%).
- 198,196 valid consecutive 9-step future windows.
- 197,234 of those windows have a valid current bbox for both agents.

This is the best existing automatic collection for navigation supervision,
but it should not dominate the final manual fine-tuning stage.

## data7_8hand

### Source split

The source manifests divide the original collection as follows:

| Source | Batches | Episodes | Rows per agent |
|---|---|---:|---:|
| old format | `keyboard_collect`, `new1`, `new2`, old `new3` | 895 | 179,100 |
| effective-dt format | new `new3`, `new4`, `new5`, `new6` | 918 | 275,200 |

The new-format batch counts are: `new3` 157/46,900 rows, `new4` 336/100,800,
`new5` 223/66,900, and `new6` 202/60,600.

### New format

- 874 snap-disabled episodes contain 262,000 rows.
- 44 snap-heading episodes contain 13,200 rows and must be excluded from yaw
  and navigation supervision because heading is corrected outside the label.
- For snap-disabled rows, the recommended timing/continuity filter retains
  260,878 rows (99.57%) and 250,620 consecutive 9-step future windows.
- 139,090 retained windows have a valid current bbox for both agents; 111,530
  do not. Missing bbox is concentrated in known collection batches, not random
  frame corruption.
- New-format `base_velocity * effective_dt_s` reconstructs after-action poses
  to numerical precision.

Use all 250,620 reliable windows for global navigation with an explicit
`roi_valid=false` path. Use the 139,090 bbox-valid windows for oracle-ROI and
bbox-prompt losses. Never convert `[0,0,0,0]` into a fake ground-truth box.
For the missing-bbox subgroup, either keep global-only samples or generate
pseudo boxes with a detector plus temporal tracker and retain only
high-confidence tracks.

### Old format

The old rows now carry calibrated/normalized fields, but those fields do not
make the original command labels reliable:

- Effective-duration median and P95 are both 0.2 s.
- Per-step XY reconstruction error has mean 0.178 m, median 0.198 m, and
  P95 0.300 m.

These 179,100 rows are useful for visual and ROI learning, especially because
bbox validity is about 99%, but navigation/waypoint/yaw loss should be zero
unless labels are rebuilt from trustworthy pose timestamps.

## Recommended Training Use

### Stage A: visual and spatial pretraining

Use all visually valid current frames from all three datasets. Train/fine-tune
the visual projection, ROI branch, bbox-derived spatial fields, visibility,
and target localization objectives. Attach source and quality masks so old or
low-frequency rows do not contribute navigation loss.

### Stage B: filtered navigation training

Use:

- 198,196 reliable future windows from `data7_7`.
- 250,620 reliable snap-disabled new-format windows from `data7_8hand`.

This gives 448,816 candidate 9-step windows before train/validation/test
grouping. Of these, 336,324 have valid current bboxes for both agents. Balance
automatic and manual sources with a sampler rather than allowing row count to
set their importance implicitly.

### Stage C: manual-domain fine-tuning

Fine-tune on bbox-valid, snap-disabled, new-format manual data. Mix a smaller
fraction of bbox-missing manual samples so evaluation remains robust when the
environment bbox is temporarily unavailable.

## Required Preprocessing Changes Before Retraining

1. Add per-sample masks such as `nav_label_valid`, `roi_valid`, `timing_valid`,
   `snap_heading`, and `source_group`.
2. Build future waypoints at fixed cumulative times (0.1, 0.2, ... seconds)
   by interpolating the integrated trajectory. Do not equate nine recorded
   rows with 0.9 seconds when per-row durations vary.
3. Build 31-frame history by cumulative simulated time and nearest/interpolated
   sampling at 0.1-second offsets. The current implementation selects the
   preceding 31 rows, which gives variable temporal coverage.
4. Split by scene and target-trajectory signature. Do not randomly split
   repeated `data7_7` trajectories across train and validation/test.
5. Keep the closed-loop evaluation at deterministic 0.1 s and report results
   separately for bbox-valid oracle ROI and bbox-missing/global-only modes.

The 9-step counts above validate future-label continuity only; they do not by
themselves certify the 31-frame input history. Training effectiveness must be
verified by open-loop reconstruction checks plus held-out deterministic
closed-loop evaluation.

## Implemented Time-Aligned Dataset

The fixed-time preprocessing and pipeline are implemented in:

- `tools/make_tracking_data.py`
- `sh/run_data7_7_8_timealigned_bboxprompt_pipeline.sh`

The generated dataset is:

- Root: `/data/hdt/ntv_data/data/data7_7_8_basevel_timealigned_roi_train`
- Split manifest: `split_manifest_timealigned.json`
- Training episodes: 2,004
- Training samples: 303,257
- `data7_7`: 1,225 episodes / 87,746 samples
- new-format `data7_8hand`: 779 episodes / 215,511 samples
- Held-out comparison test: the existing 100 bbox-zero episodes

Every retained future label is sampled at `[0.0, 0.1, ..., 0.9]` seconds.
History references use the nearest recorded observations at fixed 0.1-second
past-time offsets, with normal edge padding at the beginning of an episode.
The dt, collision, snap-heading, and continuity-gap quality checks cover the
source interval from the selected history through the future action horizon.

Existing extracted frames are reused through source-namespaced symlinks.
Cache keys preserve `frames/data7_7/...` and `frames/data7_8_new/...`, so equal
scene/episode names from the two sources cannot overwrite each other's visual
tokens.

Compatible vision caches are also reused by episode-level symlinks:

- `data7_7_roi_train/vision_cache` supplies the `data7_7` cache namespace.
- `data7_8_basevel_roi_train/vision_cache` is content-hash mapped into the
  `data7_8_new` namespace.

Both old caches use 384-pixel letterbox inputs and ROI settings of 16 tokens,
1.5 expansion, and square crops. Their legacy `roi_tokens_v1` payloads predate
the `source_bbox` metadata field, so the new pipeline enables an explicit
compatibility flag after source JSON identity has been verified. Strict bbox
validation remains the default for all other training runs.
