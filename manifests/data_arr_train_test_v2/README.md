# data_arr train/test split v2 (no validation split)

This v2 replaces the previous 7:1:2 train/val/test use for the next training
round. The old v1 manifest remains unchanged.

| Split | DT | STT | Total |
|---|---:|---:|---:|
| Train | 1,933 | 5,614 | 7,547 |
| Test | 500 | 1,400 | 1,900 |
| Val | 0 | 0 | 0 |
| Core total | 2,433 | 7,014 | 9,447 |

STT test is exactly:

```text
loststart = 560
standard  = 840
ratio     = 2:3
```

All 353 released `loststart` episodes and all former validation episodes not
selected for standard test are in train. The old validation set contributes 357
STT standard episodes to test and 595 episodes to train.

To make the total exactly 1,900, one DT episode is moved from test to train:

```text
dt/UnrealTrack-Desert_ruins-ContinuousColor-v0/dt_camera3__5__seed_100/8
```

The old DT test consists of two indivisible source batches (264 and 237), so
this one-episode adjustment is explicitly recorded as a source-batch exception.
The moved episode has a globally unique trajectory signature; the hard
trajectory leakage check remains zero.

Generated files:

- `split_manifest.json`
- `train.json`, `test.json`
- `test_stt_loststart.json`, `test_stt_standard.json`, `test_dt_standard.json`
- `train_loststart_remainder.json`
- `waypoint8/` and `waypoint10/`
- `summary.json`, `SHA256SUMS.json`

