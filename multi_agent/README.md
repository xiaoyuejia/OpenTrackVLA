# Multi-Agent Data Preprocessing

中文说明见 [README_ZH.md](README_ZH.md)，模型架构与线框图见 [ARCHITECTURE_ZH.md](ARCHITECTURE_ZH.md)。

This folder contains preprocessing scripts for paired UnrealZoo aerial-ground tracking data:

- `make_multi_agent_tracking_data.py`
- `precache_multi_agent_frames.py`

The expected raw episode layout is:

```text
sim_data/unrealzoo_aerial_ground_human_small/
  seed_100/
    UnrealTrack-DowntownWest-ContinuousColor-v0/
      0.json
      0_drone.mp4
      0_drone_info.json
      0_robotdog.mp4
      0_robotdog_info.json
      0_global.mp4
```

`0_global.mp4` is ignored for training. The two model agents are built from `drone` and `robotdog`.

## 1. Build JSONL Data

```bash
python multi_agent/make_multi_agent_tracking_data.py \
  --input_root /data/hdt/newtrackvla/sim_data/unrealzoo_aerial_ground_human_small \
  --output_root /data/hdt/newtrackvla/data/unrealzoo_aerial_ground_human_multi \
  --history 31 \
  --horizon 8 \
  --n_waypoints 8 \
  --dt 0.1 \
  --only_success \
  --exclude_collision
```

Useful debug command:

```bash
python multi_agent/make_multi_agent_tracking_data.py \
  --input_root /data/hdt/newtrackvla/sim_data/unrealzoo_aerial_ground_human_small \
  --output_root /tmp/multi_agent_debug \
  --dry_run
```

By default:

- `agent1 = drone`
- `agent2 = robotdog`
- bbox is normalized to `cx, cy, w, h`
- frame outputs are written under `<output_root>/frames`
- JSONL outputs are written under `<output_root>/jsonl`
- an aggregated `<output_root>/dataset.json` is also written

Each JSONL sample includes both structured and flat fields:

```text
agents.drone.*
agents.robotdog.*
agent1_images / agent1_current / agent1_bbox / agent1_waypoints
agent2_images / agent2_current / agent2_bbox / agent2_waypoints
bbox_feat: (2, 4)
waypoints: (2, n_waypoints, 3)
valid_mask: (2, n_waypoints)
```

## 2. Precache Vision Tokens

```bash
python multi_agent/precache_multi_agent_frames.py \
  --data_root /data/hdt/newtrackvla/data/unrealzoo_aerial_ground_human_multi \
  --batch_size 8 \
  --image_size 384
```

This writes:

```text
<data_root>/vision_cache/frames/.../*_vcoarse.pt
<data_root>/vision_cache/frames/.../*_vfine.pt
```

The cache path mirrors the relative frame path, matching the convention used by the original single-agent dataset code.

## 3. Output Schema Notes

The script treats `target_bbox` from UnrealZoo as pixel-space `x, y, w, h`, then converts it to normalized `cx, cy, w, h`.

Waypoints are produced from the measured `base_velocity` field by default. To prefer another field:

```bash
python multi_agent/make_multi_agent_tracking_data.py \
  ... \
  --action_field commanded_base_velocity
```

For the current generated data, `base_velocity` is recommended because it reflects the measured motion stored in each `*_info.json`.

## 4. Train The Multi-Agent Model

```bash
python multi_agent/train_multi_agent.py \
  --train_json /data/hdt/newtrackvla/data/unrealzoo_aerial_ground_human_multi/jsonl \
  --out_dir /data/hdt/newtrackvla/ckpt/ckpts_multi_agent \
  --cache_root /data/hdt/newtrackvla/data/unrealzoo_aerial_ground_human_multi/vision_cache \
  --llm_name Qwen/Qwen3-0.6B \
  --n_waypoints 8 \
  --history 31 \
  --batch_size 2 \
  --grad_accum_steps 8 \
  --epochs 4 \
  --lr 2e-5 \
  --beta_nav 10 \
  --freeze_llm
```

Sanity-check data and cache shapes without loading the LLM:

```bash
python multi_agent/train_multi_agent.py \
  --train_json /data/hdt/newtrackvla/data/unrealzoo_aerial_ground_human_multi/jsonl \
  --out_dir /tmp/ckpt_multi_agent_dry \
  --batch_size 2 \
  --num_workers 0 \
  --dry_run
```

The training batch shape is:

```text
coarse_tokens: (B, 2, 124, C)
fine_tokens:   (B, 2, 64, C)
bbox_feat:     (B, 2, 4)
waypoints:     (B, 2, 8, 3)
valid_mask:    (B, 2, 8)
```
