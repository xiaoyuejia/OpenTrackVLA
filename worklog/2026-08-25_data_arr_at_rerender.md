# 2026-08-25：data_arr DT -> AT same-appearance rerender

## Frozen contract

- Source split: `manifests/data_arr_train_test_v2`.
- Training DT episodes: 1,580.
- GPUs: physical 0, 1, 6; one isolated UE worker per GPU.
- Target, RobotDog and Drone pose/camera: exact per-frame recorded replay.
- Distractors: deterministic regenerated NavMesh starts/routes.
- Evaluation replay uses distinct deterministic human appearance IDs per
  actor; appearance is recorded and restored, not forced identical. RobotDog
  keeps an animal appearance and Drone is unchanged.
- Instruction source: OmTrackVLA AT `instructions.json` (initially observed
  person wording), not DT appearance descriptions.
- Evaluation: JSON-only configs first; an AT-aware evaluator must consume them.

## Distractor policy

OmTrackVLA train uses 2--7 total humans (mean 3.99); validation averages 4.74
and caps at 7. For the mostly outdoor UnrealZoo data:

- constrained indoor: 5 distractors + 1 target = 6 humans (the DT minimum);
- normal outdoor: 7 distractors + 1 target = 8 humans;
- large open outdoor: 9 distractors + 1 target = 10 humans.

The output metadata must mark
`distractor_trajectory_source=deterministic_regenerated_navmesh`; it is true AT
but not a pixel-paired DT/AT replay.

## Replay plan generated

`manifests/data_arr_at_v1` contains:

```text
train episodes = 1580
GPU 0 = 527 episodes / 158100 frames
GPU 1 = 527 episodes / 158100 frames
GPU 6 = 526 episodes / 157800 frames
evaluation JSON configs = 853
```

The plan freezes instruction, appearance, distractor count and distractor seed
per episode. The renderer is implemented in `tools/replay_data_arr_at.py`.

## Source limitation

Only 77/1,580 train episodes can be mapped back to the surviving original
multi-human `train.json`. Per-frame logs omit distractor poses. Therefore the
approved full-data route regenerates distractor trajectories while preserving
the target/follower replay exactly.

## Real smoke and throughput decision

One normal-outdoor Brass_Gardens episode completed on GPU 0 with eight humans:

```text
frames                    = 300
wall time                 = 402.975 s
throughput                = 0.7445 steps/s/GPU
Drone/RobotDog video      = 300 frames each, 640x480
valid target bbox         = 300/300 for both views
```

At this measured rate, three GPUs provide about 2.23 steps/s and the 474,000
training frames require about 59 hours. Even the repository's previously
optimized UE hot-loop rate of roughly 2.6 steps/s/GPU would require about 17
hours. Finishing in four hours requires 32.9 aggregate steps/s, or 11.0 per GPU,
which is not credible for per-frame dual RGB + exact mask + pose/camera mutation.

The full task was not launched. Doing so requires an explicit contract change:
more GPUs/workers, mask-stride/interpolated bbox, temporal subsampling, or an
accepted 17--60 hour runtime. None is silently applied because it changes data
quality or resource scope.

## Evaluation-only update

The v2 split has exactly 500 DT test episodes. Existing DT frame logs contain no
per-frame distractor pose fields, so these 500 episodes require one evaluation
replay. The replay records:

```text
target/Dog/Drone pose = recorded per frame
human appearance      = distinct deterministic IDs
distractor starts/goals= deterministic NavMesh configuration
distractor pose       = distractor_poses_per_frame
RGB/mask/bbox         = regenerated for both follower cameras
```

If a future source file contains a valid distractor pose trace, it should be
consumed directly and regeneration skipped. The current source has no such
trace. A one-episode evaluation smoke completed with 8 humans, 300 frames,
both videos and 300 frames of seven-distractor pose records. The first
NavMesh-only motion was too slow (about 0.19 m/s mean), so the renderer was
changed to explicit DT-style `[turn_deg, speed_cm_s]` human actions with
`speed=90 cm/s`, acceleration `300 UU/s^2`, and target-relative orbit points.
The corrected 30-frame smoke reached about 0.07--0.25 m/s mean per actor and
0.30 m/s peaks, with target distances mostly 5.8--9.8 m. The renderer now uses
target-relative orbit points (3.0--5.1 m), explicit `[turn_deg, speed_cm_s]`
actions, and nearby safe starts/goals; a longer 300-frame validation is still
required before starting the 500-episode run. The observed speed remains below
the nominal 0.9 m/s DT cap during this short acceleration/turning window and is
recorded rather than silently claimed as nominal.
## GPU 6 complete 300-frame validation

The same Brass_Gardens evaluation episode was rerun on GPU 6 after the GPU 0
process was interrupted:

```text
wall time                 = 477.30 s
throughput                = 0.629 steps/s
Drone/RobotDog videos     = 300 frames each, 10 FPS
Drone/RobotDog valid bbox = 300/300 each
distractor pose records   = 300 frames x 7 actors
mean distractor speed     = 0.164 m/s
median speed              = 0.162 m/s
P95 speed                 = 0.300 m/s
speed < 0.01 m/s          = 25.75% of actor-frames
mean distance to target   = 6.43 m
median distance           = 5.83 m
distance < 6 m            = 52.22% of actor-frames
```

The explicit ground-human action path is now effective: most distractors move
near the target instead of remaining at the original slow NavMesh behavior.
However, the trace still contains approximately 25.8% near-zero-speed actor
frames and a maximum measured instantaneous speed of 2.37 m/s. These require a
separate pose-jump/collision audit before treating the motion as a final
kinematic contract. The 500-episode replay remains paused.

## Replay correction: calibrated target motion

The earlier implementation was wrong in two ways: it restored target pose at
every frame and sent a hand-written ground command to the target. The renderer
now restores target/Dog/Drone only at frame zero and calls the repository's
existing `recorded_target_action_for_step()` with calibrated position feedback,
yaw gain and fixed-dt semantics. It passes the resulting target action through
`env.action_mapping()` before `set_move_bp`, matching the previous recorded
replay path.

On a first-frame-moving ModularSciFiVillage episode, 30-frame calibration gave:

```text
recorded target mean speed = 0.956 m/s
replay target mean speed   = 0.668 m/s
replay target max speed    = 0.942 m/s
```

This fixes the prior hard 0.30 m/s cap, but a complete 300-frame first-frame-
moving validation is still required. The earlier Brass_Gardens episode has a
stationary first 30-frame segment and is not sufficient for this check.

## Calibrated 300-frame moving validation

The first-frame-moving ModularSciFiVillage episode was rerun for all 300
frames with the calibrated target action path:

```text
wall time              = 581.58 s
target original path   = 30.35 m
target replay path     = 27.55 m
original mean speed    = 1.015 m/s
replay mean speed      = 0.921 m/s
replay median speed    = 0.967 m/s
replay P95 speed       = 1.067 m/s
target pose error mean = 2.31 m
target pose error P95  = 2.82 m
Drone/RobotDog videos  = 300 frames each
```

This confirms that the target physically walks through the simulator instead
of being restored to a recorded pose every frame. The remaining pose error is
the closed-loop inverse/action replay difference and is reported rather than
hidden by teleportation. The 500-episode evaluation replay remains paused
until the distractor motion audit is also accepted.

## Legacy replay migration

The validated replay core was migrated into the current repository:

```text
tools/history/replay_hand_realtime_inverse_fixed_dt.py
tools/history/replay_hand_step_base_velocity.py
tools/history/replay_unrealzoo_recording.py
tools/legacy_replay_kinematics.py
```

AT rendering now uses the legacy stable sequential snapshot and `fixed_step()`.
Target control uses legacy `source_yaw` calibrated ground control; synthetic
distractors use the legacy `vector_2d_stable` branch with a 0.9 m/s cap.

The first migrated 30-frame smoke showed target speed matching the source:

```text
target source mean speed = 0.9559 m/s
target replay mean speed = 0.9567 m/s
```

After correcting the distractor `max_speed` unit from 100 m/s to 0.9 m/s, the
30-frame distractor smoke showed:

```text
distractor mean speed = 0.239 m/s
distractor max speed  = 0.900 m/s
distance to target    = 2.73--4.76 m
```

This is the first migration smoke that uses the old fixed-step and ground
controller rather than the custom orbit controller. A full 300-frame migrated
smoke is still required before any 500-episode evaluation replay.

## Full 300-frame legacy-migrated smoke

The ModularSciFiVillage episode was run for all 300 frames using the migrated
legacy fixed-step/snapshot/control bridge:

```text
wall time                   = 685.80 s
target source path          = 30.35 m
target replay path          = 30.55 m
target source mean speed    = 1.015 m/s
target replay mean speed    = 1.022 m/s
target pose error mean      = 0.482 m
distractor mean speed       = 0.476 m/s
distractor median speed     = 0.886 m/s
distractor P95 speed        = 0.900 m/s
distractor max observed     = 8.40 m/s
distractor mean target dist = 11.88 m
distractor distance range   = 0.84--27.62 m
```

Target replay is now numerically consistent with the old replay. Distractor
motion still has a pose-jump/outlier and drifts too far from the target, so the
500-episode job remains paused. The next fix must constrain distractor target
references to nearby valid NavMesh paths and reject/repair pose jumps before
any official evaluation data is produced.

## Target-waypoint formation distractor update

The distractor controller was changed from a time-based orbit to a fixed local
formation around the target's source trajectory. Each actor receives a
deterministic local forward/right offset plus a small per-episode perturbation;
the target source displacement is applied to that desired point at every fixed
step. Legacy `vector_2d_stable` inverse control produces the physical action,
with 0.9 m/s speed and 30 degree turn limits.

Full 300-frame ModularSciFiVillage smoke:

```text
mean distractor speed   = 0.589 m/s
median distractor speed = 0.900 m/s
P95/max speed           = 0.900 / 0.900 m/s
mean/median distance    = 5.76 / 4.48 m
distance range          = 1.72--22.40 m
distance < 8 m          = 85.14% actor-frames
distance > 10 m         = 12.42% actor-frames
near-zero speed         = 28.91% actor-frames
```

This removes the high-speed pose jump and keeps most distractors close to the
target. The remaining far/near-zero cases are NavMesh/turning failures; the
500-episode replay stays paused until a reject/repair policy is frozen.

## Dog and distractor motion repair (2026-08-26)

The previous AT renderer left the RobotDog under its default CharacterMovement
ceiling/acceleration while only target and synthetic humans received the old
replay settings. That made Dog appear stuck. The renderer now gives Dog the
same permissive old-replay limits (`max_speed=10000 UU/s`,
`acceleration=10000 UU/s^2`) and drives it with the migrated calibrated
`source_yaw` inverse controller. It also records actual target/Dog/Drone pose
traces in the AT episode JSON.

Distractors previously could receive a behind-vector from `vector_2d_stable`;
clamping its negative speed to zero without changing the heading leaves an
actor permanently stationary. The new forward-only guard turns toward the raw
vector, brakes while the error exceeds 55 degrees, and resumes positive
forward speed after alignment. Negative distractor commands are now audited
per frame.

The first 30-frame GPU 6 check found a separate source of 6.91 m/s jumps: four
actors were spawned on the same safe-start point and collided. Formation
spawns are now target-local, mutually separated (about 2.2--4.4 m apart),
NavMesh-validated, and no longer issue a residual `nav_to_goal` command. The
formation uses deterministic ±0.25 m per-episode offsets plus shared smooth
0.7 m sinusoidal perturbations; distractors have a 1.35 m/s catch-up ceiling,
then follow the target-relative point with the legacy fixed-step controller.
The plan was regenerated for all 1,933 train and 500 evaluation DT episodes.

### 30-frame separated-spawn smoke (GPU 6)

Episode `UnrealTrack-ModularSciFiVillage-ContinuousColor-v0/.../9`:

```text
Dog replay mean speed       = 0.937 m/s (source 0.956 m/s)
Dog mean pose error          = 0.078 m
Drone replay mean speed      = 0.800 m/s (source 0.800 m/s)
distractor negative command  = 0%
distractor max speed         = 1.20 m/s
distractor distance range    = 2.77--6.79 m (100% < 8 m)
all 7 spawn paths            = NavMesh-valid and unique
```

### Full 300-frame separated-spawn smoke (GPU 6)

The first UE instance disconnected during initial snapshot; the renderer now
has an episode-level retry (two retries by default), and the second instance
completed all frames. Output:

```text
wall time                    = 685.12 s (0.438 frames/s)
target path / mean speed     = 320.54 m / 1.072 m/s
target source path / speed   = 303.48 m / 1.015 m/s
target mean pose error       = 0.325 m (P95 0.591 m)
RobotDog path / mean speed   = 291.96 m / 0.976 m/s
Dog source path / speed      = 290.58 m / 0.972 m/s
Drone path / mean speed      = 281.33 m / 0.941 m/s
Drone source path / speed    = 281.04 m / 0.940 m/s
distractor mean/median speed = 1.005 / 1.150 m/s
distractor P95/max speed     = 1.200 / 1.350 m/s
distractor near-zero frames  = 1.14%
distractor distance range    = 1.87--6.80 m (100% < 8 m)
distractor negative commands = 0%
Drone/Dog videos             = 300 frames each at 10 FPS
```

The only approximately 2.1--2.2 m/s actor step occurs at source frame 131 and
is present in the original DT trace itself; it is not a distractor collision
outlier. No 500-episode replay has been launched yet.

## Final perturbation/throughput validation and GPU 6 launch (2026-08-26)

The perturbation was increased to multi-frequency deterministic motion:
forward primary/secondary amplitudes 1.0/0.45 m, lateral primary/secondary
amplitudes 1.35/0.55 m, actor-specific phases, and 4.8--7.55 s periods. The
motion remains target-relative and reproducible from the manifest seed.

The renderer gained `--snapshot-mode`, `--snapshot-attempts`, and
`--snapshot-render-sync`. The evaluation launch uses batch snapshots (3
attempts, 20 ms sync), episode-level retries, and `.complete.json` markers for
resume. A 300-frame GPU 6 validation completed in 579.38 s (0.52 fps):

```text
distractor mean/median/P95/max speed = 1.002/1.200/1.200/1.837 m/s
near-zero speed fraction              = 1.33%
distance to target                   = 1.46--9.79 m
distance <8 m / >10 m                = 95.33% / 0%
negative commands                    = 0%
Drone/Dog video frames               = 300/300 at 10 FPS
```

The full 500-episode evaluation is now launched on GPU 6 only using
`manifests/data_arr_at_v1/evaluation.json`; rerunning the same command resumes
completed episodes and retries transient UE failures.

## GPU 4/5/6 conflict-free redistribution

After 71 episodes completed, the remaining evaluation was redistributed with
a stable modulo partition over the frozen 500-item plan:

```text
GPU 4: shard 0/3, 167 planned items, worker0 runtime
GPU 5: shard 1/3, 167 planned items, worker1 runtime
GPU 6: shard 2/3, 166 planned items, worker2 runtime
pairwise episode-set intersections: 0
union: 500 episodes
```

All workers share the final output root but cannot write the same episode.
Each uses a separate Unreal binary copy and port lock. Existing 71 completion
markers are skipped by `--resume`. The launcher is
`sh/run_data_arr_at_eval_gpu456.sh`; foreground supervised workers are used
because detached children are reclaimed by the execution environment.

### Duplicate GPU 6 process cleanup

An earlier unsharded GPU 6 process (`--shard-count=1`) was still alive when
the three-way run started. It and the new GPU 6 shard both opened episode 7's
temporary videos. Neither had produced a completion marker. Both GPU 6
instances were stopped, only the two conflicted `.7_*.tmp.mp4` files were
removed, and shard 2/3 was restarted. GPU 4/5 were not interrupted. A final
GPU query showed exactly one GPU 6 Python worker and one Unreal process; the
three active episodes again have disjoint output paths.

### GPU 6 dual-instance throughput test

A private `worker3` runtime and port lock were used for a second GPU 6 Unreal
instance. It was restricted to far-tail plan indices 494 and 497, so it could
not overlap the main modulo shards. Single-instance GPU 6 baseline was about
29% utilization and 4.4 GB; dual instance reached 84--96% and about 9.1 GB.

The first complete dual-instance trial episode took 1563.87 s for 300 frames
(0.19 fps), versus the established 579--629 s (0.48--0.52 fps) single-instance
range. It had no episode retry, so the slowdown was contention rather than a
disconnect. Two concurrent workers at this rate provide less aggregate
throughput than one worker. The second trial was stopped, its incomplete
SnowMap episode 7 temporary videos were removed, and the valid completed
SnowMap episode 4 was retained. GPU 4/5 were not expanded. The production run
continues with exactly one Unreal instance on each of GPUs 4, 5, and 6.
### Five-way replay and per-episode watchdog

The remaining plan is partitioned by original evaluation index modulo 5 across
GPUs 0, 1, 4, 5, and 6. Each worker owns 100 plan entries and uses a distinct
UE runtime and port lock; pairwise episode intersections are zero.

Every episode is wrapped by a 900-second (15-minute) SIGALRM watchdog. A
timeout enters the bounded retry path, writes `<stem>.timeout.json`, closes UE
in `finally`, and never writes a completion marker. Existing `.complete.json`
files are skipped, so restarting any worker resumes safely without replacing
successful outputs. The five-way launcher is
`sh/run_data_arr_at_eval_gpu01456.sh`.

## KoreanPalace startup diagnosis and recovery (2026-08-27)

### Symptom and data audit

Four remaining episodes in
`UnrealTrack-KoreanPalace-ContinuousColor-v0/dt_camera3__7__seed_100`
(`3`, `6`, `7`, `9`) repeatedly showed a live UE process but no video,
`progress.json`, or completion marker. Their source files were not damaged:
both follower info arrays contained 300 rows, source status was successful,
and timing fields matched the other KoreanPalace episodes. Eleven sibling
episodes in the same batch had already completed, proving that the four stem
values and their trajectories were not the cause.

### Root causes

1. The replay script imported `gym_unrealcv` before inserting this repository's
   `unrealzoo-gym` into `sys.path`. It therefore loaded the old checkout under
   `/data/hdt/newtrackvla/unrealzoo-gym`. The upstream `init_map()` enumerated
   every object and requested its mask color. KoreanPalace contains over 29k
   annotated actors; logs showed thousands of sequential requests such as
   `vget /object/StaticMeshActor_23600/color`.
2. After the full-map color scan was bypassed, the first KoreanPalace human
   `spawn_from_path` still took about 55.97 seconds. The UnrealCV client request
   timeout was only 30 seconds, so the client disconnected before the valid
   spawn response arrived. Logs consequently looked like a port failure even
   though UE completed the spawn later.
3. An earlier four-GPU special retry also used one shared writable runtime and
   the default `/tmp/unrealzoo_unrealcv_port.lock`. This introduced avoidable
   startup/INI contention. Those attempts produced 0/4 completions and were
   stopped; logs were preserved under
   `/data/hdt/ntv_data/logs/koreanpalace_replay_20260827/`.

### Fixes

- Move `import gym_unrealcv` after the current repository path insertion.
- Add the opt-in `UNREALZOO_SKIP_FULL_COLOR_DICT=1` fast path to
  `Character_API.init_map()`. It initializes cameras but leaves `obj_dict`
  empty; `BaseEnv.init_agents()` subsequently builds colors only for
  `player_list`, which is exactly what target/distractor mask extraction uses.
- Set `UNREALZOO_REQUEST_TIMEOUT_S=120` for KoreanPalace so its ~56-second
  initial actor spawn does not disconnect the client.
- Run the four episodes serially on GPU 4 with private `worker0` runtime and
  `/tmp/koreanpalace_gpu4_serial.lock`.
- Write `<stem>.progress.json` only after a real `legacy_fixed_step()` action
  pulse and refreshed target/Dog/Drone poses. A live UE with no progress file
  is explicitly `frames_completed=0`, not successful replay.
- Use an external process-group watchdog in
  `sh/run_koreanpalace_gpu4_serial.sh`: if no first physical-step heartbeat is
  observed within 600 seconds, send TERM/KILL to the entire Python+UE process
  group and retry. This cannot be swallowed by UnrealCV's internal reconnect
  loop. Only a 300-frame `.complete.json` is accepted.

### Verified recovery

All four previously missing episodes completed with real 300-step traces:

```text
stem 3: 300 frames, 656.11 s, 0.457 fps
stem 6: 300 frames, 608.06 s, 0.493 fps
stem 7: 300 frames, 588.57 s, 0.510 fps
stem 9: 300 frames, 636.55 s, 0.471 fps
```

During validation, `progress.json` advanced from 5 to 189 and finally 300,
including realized target, RobotDog, and Drone poses. This distinguishes true
physics replay from environment startup. The recovered outputs use the same
directory and `.complete.json` contract as earlier AT eval results and are
therefore included automatically by the final catalog refresh.

### Production policy after recovery

The remaining non-KoreanPalace eval set runs on GPUs 4/5/6 using
`tools/supervise_data_arr_at_replay.py` and
`sh/run_data_arr_at_eval_gpu456_supervised.sh`. The original evaluation indices
are partitioned modulo 3 (pairwise intersection 0), KoreanPalace is excluded by
its exact full scene ID, and each episode runs in its own process group. The
external supervisor enforces: 600-second startup-to-first-step deadline,
300-second stale-heartbeat deadline, 5400-second active limit, three attempts,
per-episode temporary-file cleanup, and `.complete.json`-only success. Existing
completed episodes are skipped and never overwritten.
