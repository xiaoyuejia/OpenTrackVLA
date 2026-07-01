#!/usr/bin/env bash
set -euo pipefail

# Build single-agent drone/robotdog training sets from the latest multi-agent
# dataset, then train on GPU 0/1 and evaluate with single-agent closed-loop
# settings aligned to the current multi-agent evaluation.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/hdt/miniconda3/envs/omtracknew/bin/python}"
[[ -x "${PYTHON_BIN}" ]] || PYTHON_BIN="python"

MULTI_DATA_ROOT="${MULTI_DATA_ROOT:-/data/hdt/ntv_data/data/data_multi_agent_10to1}"
MULTI_TRAIN_ROOT="${MULTI_TRAIN_ROOT:-${MULTI_DATA_ROOT}/train}"
MULTI_TRAIN_JSON="${MULTI_TRAIN_JSON:-${MULTI_TRAIN_ROOT}/jsonl}"
MULTI_CACHE_ROOT="${MULTI_CACHE_ROOT:-${MULTI_TRAIN_ROOT}/vision_cache}"
MULTI_SPLIT_ROOT="${MULTI_SPLIT_ROOT:-/data/hdt/ntv_data/sim_data/data_multi_agent_split_10to1}"
MULTI_MANIFEST="${MULTI_MANIFEST:-${MULTI_SPLIT_ROOT}/split_manifest.json}"

DRONE_DATA_ROOT="${DRONE_DATA_ROOT:-/data/hdt/ntv_data/data/data_multi_agent_drone_single_from_multi_10to1/train}"
ROBOTDOG_DATA_ROOT="${ROBOTDOG_DATA_ROOT:-/data/hdt/ntv_data/data/data_multi_agent_robotdog_single_from_multi_10to1/train}"
DRONE_MANIFEST="${DRONE_MANIFEST:-${MULTI_SPLIT_ROOT}/drone_single_split_manifest.json}"
ROBOTDOG_MANIFEST="${ROBOTDOG_MANIFEST:-${MULTI_SPLIT_ROOT}/robotdog_single_split_manifest.json}"

DRONE_CKPT_DIR="${DRONE_CKPT_DIR:-/data/hdt/ntv_data/ckpt/data_multi_agent_drone_single_from_multi_b32_acc4_lr2e-5_gpu0}"
ROBOTDOG_CKPT_DIR="${ROBOTDOG_CKPT_DIR:-/data/hdt/ntv_data/ckpt/data_multi_agent_robotdog_single_from_multi_b32_acc4_lr2e-5_gpu1}"
DRONE_EVAL_ROOT="${DRONE_EVAL_ROOT:-/data/hdt/ntv_data/sim_data/eval/data_multi_agent_drone_single_from_multi_b32_acc4_lr2e-5_gpu0_10to1}"
ROBOTDOG_EVAL_ROOT="${ROBOTDOG_EVAL_ROOT:-/data/hdt/ntv_data/sim_data/eval/data_multi_agent_robotdog_single_from_multi_b32_acc4_lr2e-5_gpu1_10to1}"

DRONE_GPU="${DRONE_GPU:-0}"
ROBOTDOG_GPU="${ROBOTDOG_GPU:-1}"

BATCH_SIZE="${BATCH_SIZE:-32}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
EPOCHS="${EPOCHS:-10}"
N_WAYPOINTS="${N_WAYPOINTS:-10}"
HISTORY="${HISTORY:-31}"
LR="${LR:-2e-5}"
ALPHA_XY="${ALPHA_XY:-1}"
BETA_NAV="${BETA_NAV:-100}"
FREEZE_LLM="${FREEZE_LLM:-1}"
SAVE_TRAJECTORIES="${SAVE_TRAJECTORIES:-0}"

RUN_PREPARE="${RUN_PREPARE:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_EVAL_PARALLEL="${RUN_EVAL_PARALLEL:-1}"
SAVE_EVAL_VIDEO="${SAVE_EVAL_VIDEO:-1}"
EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-600}"
EVAL_WAYPOINT_INDEX="${EVAL_WAYPOINT_INDEX:-9}"

require_path() {
    local path="$1"
    local label="$2"
    if [[ ! -e "${path}" ]]; then
        echo "[ERROR] ${label} not found: ${path}" >&2
        exit 1
    fi
}

wait_all() {
    local status=0
    for pid in "$@"; do
        if ! wait "${pid}"; then
            status=1
        fi
    done
    return "${status}"
}

prepare_single_agent_data() {
    "${PYTHON_BIN}" - <<'PY'
import json
from pathlib import Path

data_root = Path("/data/hdt/ntv_data")
src_json_root = data_root / "data/data_multi_agent_10to1/train/jsonl"
src_train_root = data_root / "data/data_multi_agent_10to1/train"
manifest_path = data_root / "sim_data/data_multi_agent_split_10to1/split_manifest.json"

out_specs = {
    "drone": {
        "agent_idx": 0,
        "prefix": "agent1",
        "root": data_root / "data/data_multi_agent_drone_single_from_multi_10to1/train",
        "manifest": data_root / "sim_data/data_multi_agent_split_10to1/drone_single_split_manifest.json",
        "info_suffix": "_drone_info.json",
        "instruction": "The aerial drone should follow the target person from the air.",
    },
    "robotdog": {
        "agent_idx": 1,
        "prefix": "agent2",
        "root": data_root / "data/data_multi_agent_robotdog_single_from_multi_10to1/train",
        "manifest": data_root / "sim_data/data_multi_agent_split_10to1/robotdog_single_split_manifest.json",
        "info_suffix": "_robotdog_info.json",
        "instruction": "The ground robot dog should follow the target person on the ground.",
    },
}

def link_dir(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink():
        if dst.resolve() == src.resolve():
            return
        dst.unlink()
    if dst.exists():
        return
    dst.symlink_to(src.resolve())

def pick_agent_sample(ex: dict, spec: dict) -> dict:
    idx = spec["agent_idx"]
    prefix = spec["prefix"]
    waypoints = ex.get("waypoints")
    wp = waypoints[idx] if isinstance(waypoints, list) and len(waypoints) > idx else ex.get(f"{prefix}_waypoints")
    valid_mask = ex.get("valid_mask")
    vm = valid_mask[idx] if isinstance(valid_mask, list) and len(valid_mask) > idx else ex.get(f"{prefix}_valid_mask")
    out = {
        "images": ex.get(f"{prefix}_images", []),
        "current": ex.get(f"{prefix}_current"),
        "waypoints": wp,
        "valid_mask": vm,
        "instruction": spec["instruction"],
        "episode_id": ex.get("episode_id", ""),
        "step_index": ex.get("step_index", 0),
        "dt": ex.get("dt", 0.1),
    }
    if ex.get(f"{prefix}_bbox") is not None:
        out["bbox"] = ex.get(f"{prefix}_bbox")
    return out

for name, spec in out_specs.items():
    out_root = spec["root"]
    out_json_root = out_root / "jsonl"
    out_json_root.mkdir(parents=True, exist_ok=True)
    link_dir(src_train_root / "frames", out_root / "frames")
    link_dir(src_train_root / "vision_cache", out_root / "vision_cache")

    n_files = 0
    n_samples = 0
    for src_file in sorted(src_json_root.rglob("*.jsonl")):
        rel = src_file.relative_to(src_json_root)
        dst_file = out_json_root / rel
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        with src_file.open("r", encoding="utf-8") as fin, dst_file.open("w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)
                out = pick_agent_sample(ex, spec)
                if not out["current"] or out["waypoints"] is None:
                    continue
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                n_samples += 1
        n_files += 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for split_name in ("train", "test"):
        for item in manifest.get(split_name, []):
            item["info"] = f"{item['relative_dir']}/{item['stem']}{spec['info_suffix']}"
            item["video"] = f"{item['relative_dir']}/{item['stem']}_{name}.mp4"
    manifest["single_agent"] = name
    manifest["source_manifest"] = str(manifest_path)
    spec["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{name}: files={n_files} samples={n_samples} root={out_root} manifest={spec['manifest']}", flush=True)
PY
}

train_one() {
    local agent="$1"
    local gpu="$2"
    local data_root="$3"
    local ckpt_dir="$4"
    local log_file="$5"
    mkdir -p "$(dirname "${log_file}")" "${ckpt_dir}"
    echo "[train][${agent}] gpu=${gpu} log=${log_file}"
    TRAIN_JSON="${data_root}/jsonl" \
    CACHE_ROOT="${data_root}/vision_cache" \
    OUT_DIR="${ckpt_dir}" \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    NUM_GPUS=1 \
    BATCH_SIZE="${BATCH_SIZE}" \
    GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS}" \
    EPOCHS="${EPOCHS}" \
    N_WAYPOINTS="${N_WAYPOINTS}" \
    HISTORY="${HISTORY}" \
    LR="${LR}" \
    ALPHA_XY="${ALPHA_XY}" \
    BETA_NAV="${BETA_NAV}" \
    FREEZE_LLM="${FREEZE_LLM}" \
    SAVE_TRAJECTORIES="${SAVE_TRAJECTORIES}" \
    bash sh/train_with_estimate.sh > "${log_file}" 2>&1
}

eval_drone() {
    RUN_ORGANIZE=0 \
    RUN_SPLIT=0 \
    RUN_PROCESS=0 \
    RUN_CACHE=0 \
    RUN_TRAIN=0 \
    RUN_EVAL=1 \
    CKPT_DIR="${DRONE_CKPT_DIR}" \
    SPLIT_ROOT="${MULTI_SPLIT_ROOT}" \
    MANIFEST="${DRONE_MANIFEST}" \
    EVAL_ROOT="${DRONE_EVAL_ROOT}" \
    EVAL_GPU="${DRONE_GPU}" \
    RENDER_GPU="${DRONE_GPU}" \
    EVAL_MAX_STEPS="${EVAL_MAX_STEPS}" \
    EVAL_WAYPOINT_INDEX="${EVAL_WAYPOINT_INDEX}" \
    SAVE_EVAL_VIDEO="${SAVE_EVAL_VIDEO}" \
    bash sh/run_drone_single_agent_pipeline.sh
}

eval_robotdog() {
    RUN_SPLIT=0 \
    RUN_PROCESS=0 \
    RUN_CACHE=0 \
    RUN_TRAIN=0 \
    RUN_EVAL=1 \
    CKPT_DIR="${ROBOTDOG_CKPT_DIR}" \
    SPLIT_ROOT="${MULTI_SPLIT_ROOT}" \
    MANIFEST="${ROBOTDOG_MANIFEST}" \
    EVAL_ROOT="${ROBOTDOG_EVAL_ROOT}" \
    EVAL_GPU="${ROBOTDOG_GPU}" \
    RENDER_GPU="${ROBOTDOG_GPU}" \
    EVAL_MAX_STEPS="${EVAL_MAX_STEPS}" \
    EVAL_WAYPOINT_INDEX="${EVAL_WAYPOINT_INDEX}" \
    SAVE_EVAL_VIDEO="${SAVE_EVAL_VIDEO}" \
    bash sh/run_robotdog_single_agent_pipeline.sh
}

require_path "${MULTI_TRAIN_JSON}" "MULTI_TRAIN_JSON"
require_path "${MULTI_CACHE_ROOT}" "MULTI_CACHE_ROOT"
require_path "${MULTI_MANIFEST}" "MULTI_MANIFEST"

cat <<EOF
===============================================================================
Single-agent drone/robotdog from latest multi-agent data
===============================================================================
MULTI_DATA_ROOT=${MULTI_DATA_ROOT}
DRONE_DATA_ROOT=${DRONE_DATA_ROOT}
ROBOTDOG_DATA_ROOT=${ROBOTDOG_DATA_ROOT}
DRONE_CKPT_DIR=${DRONE_CKPT_DIR}
ROBOTDOG_CKPT_DIR=${ROBOTDOG_CKPT_DIR}
DRONE_GPU=${DRONE_GPU} ROBOTDOG_GPU=${ROBOTDOG_GPU}
BATCH_SIZE=${BATCH_SIZE} GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS} LR=${LR}
BETA_NAV=${BETA_NAV} ALPHA_XY=${ALPHA_XY} EPOCHS=${EPOCHS}
RUN_PREPARE=${RUN_PREPARE} RUN_TRAIN=${RUN_TRAIN} RUN_EVAL=${RUN_EVAL}
===============================================================================
EOF

if [[ "${RUN_PREPARE}" == "1" ]]; then
    prepare_single_agent_data
fi

if [[ "${RUN_TRAIN}" == "1" ]]; then
    pids=()
    train_one "drone" "${DRONE_GPU}" "${DRONE_DATA_ROOT}" "${DRONE_CKPT_DIR}" "/data/hdt/ntv_data/ckpt/train_drone_single_from_multi_gpu0.log" &
    pids+=("$!")
    train_one "robotdog" "${ROBOTDOG_GPU}" "${ROBOTDOG_DATA_ROOT}" "${ROBOTDOG_CKPT_DIR}" "/data/hdt/ntv_data/ckpt/train_robotdog_single_from_multi_gpu1.log" &
    pids+=("$!")
    wait_all "${pids[@]}"
fi

if [[ "${RUN_EVAL}" == "1" ]]; then
    pids=()
    if [[ "${RUN_EVAL_PARALLEL}" == "1" ]]; then
        eval_drone > "${DRONE_EVAL_ROOT}_stdout.log" 2>&1 &
        pids+=("$!")
        eval_robotdog > "${ROBOTDOG_EVAL_ROOT}_stdout.log" 2>&1 &
        pids+=("$!")
        wait_all "${pids[@]}"
    else
        eval_drone
        eval_robotdog
    fi
fi

echo "[done] drone ckpt=${DRONE_CKPT_DIR}"
echo "[done] robotdog ckpt=${ROBOTDOG_CKPT_DIR}"
echo "[done] drone eval=${DRONE_EVAL_ROOT}"
echo "[done] robotdog eval=${ROBOTDOG_EVAL_ROOT}"
