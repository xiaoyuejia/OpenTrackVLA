#!/usr/bin/env bash
set -Eeuo pipefail

EXP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_YAML="${PIPELINE_YAML:-${EXP_ROOT}/pipeline.yaml}"
STAGE="all"
DRY_RUN=0
CKPT_OVERRIDE=""

usage() {
  cat <<'EOF'
Usage:
  bash exp9_4/run_pipeline.sh [options]

Options:
  --stage all|train|eval|eval-dt|eval-at|eval-stt|summarize|plan
  --config PATH        pipeline YAML (default: exp9_4/pipeline.yaml)
  --ckpt PATH          skip automatic final-checkpoint selection for evaluation
  --dry-run            print commands without running training/evaluation
  -h, --help

The all pipeline is strictly sequential:
  train -> DT(500) -> AT(500) -> STT(1400) -> summarize
EOF
}

while (( $# > 0 )); do
  case "$1" in
    --stage) STAGE="${2:?--stage requires a value}"; shift 2 ;;
    --stage=*) STAGE="${1#*=}"; shift ;;
    --config) PIPELINE_YAML="${2:?--config requires a path}"; shift 2 ;;
    --config=*) PIPELINE_YAML="${1#*=}"; shift ;;
    --ckpt) CKPT_OVERRIDE="${2:?--ckpt requires a path}"; shift 2 ;;
    --ckpt=*) CKPT_OVERRIDE="${1#*=}"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "${STAGE}" in
  all|train|eval|eval-dt|eval-at|eval-stt|summarize|plan) ;;
  *) echo "[ERROR] invalid --stage ${STAGE}" >&2; exit 2 ;;
esac

[[ -f "${PIPELINE_YAML}" ]] || { echo "[ERROR] missing pipeline YAML: ${PIPELINE_YAML}" >&2; exit 1; }
PYTHON_BIN="${PYTHON_BIN:-/home/yh/miniconda3/envs/newtrackvla/bin/python}"
[[ -x "${PYTHON_BIN}" ]] || { echo "[ERROR] Python not executable: ${PYTHON_BIN}" >&2; exit 1; }

# Read one dotted YAML key. A supplied default is used only when the key is absent.
yaml_get() {
  local key="$1" default_value="${2-__NO_DEFAULT__}"
  "${PYTHON_BIN}" - "${PIPELINE_YAML}" "${key}" "${default_value}" <<'PY'
import sys, yaml
path, dotted, default = sys.argv[1:]
obj = yaml.safe_load(open(path, encoding="utf-8"))
try:
    value = obj
    for part in dotted.split("."):
        value = value[part]
except (KeyError, TypeError):
    if default == "__NO_DEFAULT__":
        raise SystemExit(f"missing required YAML key: {dotted}")
    value = default
if isinstance(value, bool):
    print("true" if value else "false")
elif value is None:
    print("")
else:
    print(value)
PY
}

PROJECT_ROOT="$(yaml_get pipeline.project_root)"
PYTHON_BIN="${PYTHON_BIN_OVERRIDE:-$(yaml_get pipeline.python "${PYTHON_BIN}")}" 
TRAIN_TEMPLATE="$(yaml_get paths.train_config_template)"
EFFECTIVE_TRAIN_CONFIG="$(yaml_get paths.effective_train_config)"
MODEL_DIR="$(yaml_get paths.model_dir)"
RESULT_ROOT="$(yaml_get paths.result_root)"
RUNTIME_ROOT="$(yaml_get paths.runtime_root)"
LOG_ROOT="$(yaml_get paths.log_root)"
STATE_ROOT="$(yaml_get paths.state_root)"
SUMMARY_ROOT="$(yaml_get paths.summary_root)"
TRAIN_GPUS="$(yaml_get train.gpu_ids)"
EVAL_GPUS="$(yaml_get evaluation.gpu_ids)"
RENDER_GPUS="$(yaml_get evaluation.render_gpu_ids)"
WORKERS_PER_GPU="$(yaml_get evaluation.workers_per_gpu)"
EVAL_STRIDE="$(yaml_get evaluation.policy_inference_stride)"
TARGET_MATCH_THRESHOLD="$(yaml_get evaluation.target_match_confidence_threshold 0.50)"
EXPECTED_FINAL_EPOCH="$(yaml_get train.expected_final_epoch)"
INIT_MODE="$(yaml_get train.init_mode scratch)"
INIT_SEARCH_DIR="$(yaml_get train.init_search_dir "")"
INIT_PATTERN="$(yaml_get train.init_pattern)"
REQUIRE_INIT="$(yaml_get train.require_init_checkpoint true)"

mkdir -p "${MODEL_DIR}" "${RESULT_ROOT}" "${RUNTIME_ROOT}" \
  "${LOG_ROOT}" "${STATE_ROOT}" "${SUMMARY_ROOT}"

exec 9>"${EXP_ROOT}/.pipeline.lock"
flock -n 9 || { echo "[ERROR] another exp9_4 pipeline is active" >&2; exit 2; }

PIPELINE_LOG="${LOG_ROOT}/pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${PIPELINE_LOG}") 2>&1

run_cmd() {
  printf '[CMD]'
  printf ' %q' "$@"
  printf '\n'
  if (( DRY_RUN == 0 )); then
    "$@"
  fi
}

latest_checkpoint() {
  local root="$1" pattern="$2"
  "${PYTHON_BIN}" - "${root}" "${pattern}" <<'PY'
import re, sys
from pathlib import Path
root, pattern = Path(sys.argv[1]), sys.argv[2]
paths = list(root.glob(pattern)) if root.is_dir() else []
def key(path):
    nums = tuple(int(x) for x in re.findall(r"\d+", path.name))
    return nums, path.stat().st_mtime_ns
if paths:
    print(max(paths, key=key).resolve())
PY
}

manifest_count() {
  "${PYTHON_BIN}" - "$1" <<'PY'
import json, sys
obj=json.load(open(sys.argv[1], encoding="utf-8"))
rows=obj.get("test", obj) if isinstance(obj, dict) else obj
print(len(rows))
PY
}

print_plan() {
  local init_candidate=""
  if [[ "${INIT_MODE}" != "scratch" ]]; then
    init_candidate="$(latest_checkpoint "${INIT_SEARCH_DIR}" "${INIT_PATTERN}")"
  fi
  cat <<EOF
[PLAN] config=${PIPELINE_YAML}
[PLAN] project=${PROJECT_ROOT}
[PLAN] train GPUs=${TRAIN_GPUS} stride=$(yaml_get train.temporal_stride) epochs=$(yaml_get train.epochs)
[PLAN] target reference=$(yaml_get train.target_reference) for train/validation/evaluation
[PLAN] train output=${MODEL_DIR}
[PLAN] initialization=${INIT_MODE} candidate=${init_candidate:-NONE}
[PLAN] eval GPUs=${EVAL_GPUS} render GPUs=${RENDER_GPUS} workers/GPU=${WORKERS_PER_GPU}
[PLAN] eval stride=${EVAL_STRIDE}; target-match-threshold=${TARGET_MATCH_THRESHOLD}; videos=$(yaml_get evaluation.save_video); global_video=$(yaml_get evaluation.write_global_video)
[PLAN] order=DT -> AT -> STT
EOF
  local task manifest expected
  for task in dt at stt; do
    manifest="$(yaml_get evaluation.manifests.${task})"
    expected="$(yaml_get evaluation.expected_episodes.${task})"
    printf '[PLAN] eval %-3s manifest=%s actual=%s expected=%s output=%s\n' \
      "${task^^}" "${manifest}" "$(manifest_count "${manifest}")" "${expected}" "${RESULT_ROOT}/${task}"
  done
}

preflight() {
  [[ -d "${PROJECT_ROOT}" ]] || { echo "[ERROR] missing project root ${PROJECT_ROOT}" >&2; exit 1; }
  [[ -f "${TRAIN_TEMPLATE}" ]] || { echo "[ERROR] missing train template ${TRAIN_TEMPLATE}" >&2; exit 1; }
  local task manifest expected actual
  for task in dt at stt; do
    manifest="$(yaml_get evaluation.manifests.${task})"
    expected="$(yaml_get evaluation.expected_episodes.${task})"
    [[ -f "${manifest}" ]] || { echo "[ERROR] missing ${task} manifest: ${manifest}" >&2; exit 1; }
    actual="$(manifest_count "${manifest}")"
    [[ "${actual}" == "${expected}" ]] || {
      echo "[ERROR] ${task} manifest count ${actual}, expected ${expected}" >&2; exit 1;
    }
  done
  [[ "$(yaml_get train.temporal_stride)" == "5" ]] || { echo "[ERROR] training stride must be 5" >&2; exit 1; }
  [[ "$(yaml_get train.sample_interval_seconds)" == "0.5" ]] || { echo "[ERROR] training sample interval must be 0.5 s" >&2; exit 1; }
  [[ "$(yaml_get train.waypoint_step_dt_seconds)" == "0.1" ]] || { echo "[ERROR] recorded waypoint step dt must remain 0.1 s" >&2; exit 1; }
  [[ "$(yaml_get train.epochs)" == "10" ]] || { echo "[ERROR] training epochs must be 10" >&2; exit 1; }
  [[ "$(yaml_get train.receiver_curriculum)" == "progressive_linear_10stage_v1" ]] || {
    echo "[ERROR] exp9_4 requires the ten-stage progressive receiver curriculum" >&2; exit 1;
  }
  [[ "$(yaml_get train.target_reference)" == "disabled" ]] || {
    echo "[ERROR] exp9_4 must train and evaluate without target reference" >&2; exit 1;
  }
  [[ "${INIT_MODE}" == "scratch" && "${REQUIRE_INIT}" == "false" ]] || {
    echo "[ERROR] exp9_4 must start from scratch without an AirGround checkpoint" >&2; exit 1;
  }
  "${PYTHON_BIN}" - "${TRAIN_TEMPLATE}" <<'PY'
import sys, yaml
cfg=yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
assert cfg["data"].get("use_target_reference") is False, \
    "exp9_4 train/validation target reference must be disabled"
stages=cfg["data"].get("receiver_corruption_curriculum") or []
assert len(stages) == 10, "receiver curriculum must contain one stage per epoch"
assert [int(s["end_epoch"]) for s in stages] == list(range(1, 11))
expected = {
    "assistance_probability": [0.700,0.717,0.733,0.750,0.767,0.783,0.800,0.817,0.833,0.850],
    "roi_only_probability": [0.700,0.667,0.633,0.600,0.567,0.533,0.500,0.467,0.433,0.400],
    "current_full_probability": [0.300] * 10,
    "recent_full_probability": [0.000,0.028,0.056,0.083,0.111,0.139,0.167,0.194,0.222,0.250],
    "all_full_probability": [0.000,0.005,0.011,0.017,0.022,0.028,0.033,0.039,0.045,0.050],
    "pose_perturb_probability": [0.000,0.056,0.111,0.167,0.222,0.278,0.333,0.389,0.444,0.500],
    "pose_translation_max_m": [0.250,0.278,0.306,0.333,0.361,0.389,0.417,0.444,0.472,0.500],
    "pose_yaw_max_deg": [10.000,12.222,14.444,16.667,18.889,21.111,23.333,25.556,27.778,30.000],
}
for name, values in expected.items():
    actual=[float(s[name]) for s in stages]
    assert all(abs(a-b) < 1e-8 for a,b in zip(actual, values)), (name, actual)
for stage in stages:
    mode_sum=sum(float(stage[k]) for k in (
        "roi_only_probability", "current_full_probability",
        "recent_full_probability", "all_full_probability"))
    assert abs(mode_sum - 1.0) < 1e-8, mode_sum
print("[PREFLIGHT] ten-stage linear receiver curriculum verified")
PY
  [[ "${EVAL_STRIDE}" == "5" ]] || { echo "[ERROR] evaluation stride must be 5" >&2; exit 1; }
  [[ "$(yaml_get evaluation.policy_interval_seconds)" == "0.5" ]] || { echo "[ERROR] evaluation policy interval must be 0.5 s" >&2; exit 1; }
  [[ "$(yaml_get evaluation.environment_step_dt_seconds)" == "0.1" ]] || { echo "[ERROR] environment control dt must remain 0.1 s" >&2; exit 1; }
  [[ "$(yaml_get evaluation.history_frame_dt_seconds)" == "0.1" ]] || { echo "[ERROR] visual history dt must remain 0.1 s" >&2; exit 1; }
  [[ "$(yaml_get evaluation.waypoint_step_dt_seconds)" == "0.1" ]] || { echo "[ERROR] waypoint source dt must remain 0.1 s" >&2; exit 1; }
  [[ "$(yaml_get evaluation.waypoint_horizon_steps)" == "7" ]] || { echo "[ERROR] 8 origin-inclusive waypoints require horizon step 7" >&2; exit 1; }
  [[ "$(yaml_get evaluation.skip_rgb_between_policy_steps true)" == "false" ]] || { echo "[ERROR] stride-5 evaluation must retain 0.1-s RGB history observations" >&2; exit 1; }
  "${PYTHON_BIN}" - "${TARGET_MATCH_THRESHOLD}" <<'PY'
import sys
value=float(sys.argv[1])
assert 0.0 < value < 1.0, "target-match threshold must be in (0,1)"
PY
  [[ "$(yaml_get evaluation.drone_min_follow_dist_m)" == "1.0" ]] || { echo "[ERROR] drone min distance must be 1.0" >&2; exit 1; }
  [[ "$(yaml_get evaluation.drone_max_follow_dist_m)" == "6.0" ]] || { echo "[ERROR] drone max distance must be 6.0" >&2; exit 1; }
  [[ "$(yaml_get evaluation.robotdog_min_follow_dist_m)" == "1.0" ]] || { echo "[ERROR] dog min distance must be 1.0" >&2; exit 1; }
  [[ "$(yaml_get evaluation.robotdog_max_follow_dist_m)" == "6.0" ]] || { echo "[ERROR] dog max distance must be 6.0" >&2; exit 1; }
}

materialize_train_config() {
  local init_ckpt="$1"
  mkdir -p "$(dirname "${EFFECTIVE_TRAIN_CONFIG}")"
  "${PYTHON_BIN}" - "${TRAIN_TEMPLATE}" "${EFFECTIVE_TRAIN_CONFIG}" "${init_ckpt}" "${MODEL_DIR}" <<'PY'
import sys, yaml
src, dst, init_ckpt, out_dir = sys.argv[1:]
with open(src, encoding="utf-8") as f:
    cfg=yaml.safe_load(f)
cfg["optimization"]["out_dir"] = out_dir
cfg["optimization"]["epochs"] = 10
cfg["data"]["train_temporal_stride"] = 5
cfg["runtime"]["init_ckpt"] = init_ckpt or None
cfg["runtime"]["resume"] = False
with open(dst, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
PY
  printf '%s\n' "${init_ckpt}" >"${STATE_ROOT}/selected_init_checkpoint.txt"
}

run_training() {
  local done_marker="${STATE_ROOT}/train.done" init_ckpt resume_flag=()
  if [[ -f "${done_marker}" ]]; then
    echo "[SKIP] training already complete: $(<"${done_marker}")"
    return
  fi

  if compgen -G "${MODEL_DIR}/*.pt" >/dev/null; then
    [[ -f "${EFFECTIVE_TRAIN_CONFIG}" ]] || {
      echo "[ERROR] partial model directory exists but effective config is missing: ${MODEL_DIR}" >&2
      exit 1
    }
    resume_flag=(--resume)
    echo "[TRAIN] resuming interrupted exp9_4 training"
  else
    if [[ "${INIT_MODE}" == "scratch" ]]; then
      init_ckpt=""
    else
      init_ckpt="$(latest_checkpoint "${INIT_SEARCH_DIR}" "${INIT_PATTERN}")"
      if [[ -z "${init_ckpt}" && "${REQUIRE_INIT}" == "true" ]]; then
        echo "[ERROR] no initialization checkpoint found in ${INIT_SEARCH_DIR}/${INIT_PATTERN}" >&2
        exit 1
      fi
      [[ -z "${init_ckpt}" || -f "${init_ckpt}" ]] || { echo "[ERROR] init checkpoint missing: ${init_ckpt}" >&2; exit 1; }
    fi
    materialize_train_config "${init_ckpt}"
    echo "[TRAIN] fresh run initialized from ${init_ckpt:-scratch}"
  fi

  if (( DRY_RUN == 0 )) && pgrep -f 'train_airground_coop_v3.py' >/dev/null && [[ "${ALLOW_CONCURRENT_TRAIN:-0}" != 1 ]]; then
    echo "[ERROR] another AirGround V3 training process is active. Wait for it or set ALLOW_CONCURRENT_TRAIN=1." >&2
    exit 2
  fi

  cd "${PROJECT_ROOT}"
  run_cmd env PY="${PYTHON_BIN}" bash sh/train_airground_coop_v3.sh \
    --gpu-ids "${TRAIN_GPUS}" --config "${EFFECTIVE_TRAIN_CONFIG}" "${resume_flag[@]}"

  if (( DRY_RUN == 0 )); then
    local final_ckpt
    final_ckpt="$(latest_checkpoint "${MODEL_DIR}" "model_epoch$(printf '%03d' "${EXPECTED_FINAL_EPOCH}")_step*_final.pt")"
    [[ -n "${final_ckpt}" && -f "${final_ckpt}" ]] || {
      echo "[ERROR] expected epoch-${EXPECTED_FINAL_EPOCH} final checkpoint not found" >&2; exit 1;
    }
    printf '%s\n' "${final_ckpt}" | tee "${done_marker}" "${STATE_ROOT}/evaluation_checkpoint.txt"
  fi
}

select_eval_checkpoint() {
  if [[ -n "${CKPT_OVERRIDE}" ]]; then
    printf '%s\n' "${CKPT_OVERRIDE}"
    return
  fi
  if [[ -s "${STATE_ROOT}/evaluation_checkpoint.txt" ]]; then
    cat "${STATE_ROOT}/evaluation_checkpoint.txt"
    return
  fi
  latest_checkpoint "${MODEL_DIR}" "model_epoch$(printf '%03d' "${EXPECTED_FINAL_EPOCH}")_step*_final.pt"
}

run_eval_task() {
  local task="$1" task_upper="${1^^}"
  local manifest expected out runtime done_marker ckpt resume_completed
  manifest="$(yaml_get evaluation.manifests.${task})"
  expected="$(yaml_get evaluation.expected_episodes.${task})"
  out="${RESULT_ROOT}/${task}"
  runtime="${RUNTIME_ROOT}/${task}"
  done_marker="${STATE_ROOT}/eval_${task}.done"
  ckpt="$(select_eval_checkpoint)"
  resume_completed="$(yaml_get evaluation.resume_completed true)"

  if (( DRY_RUN == 1 )) && [[ -z "${ckpt}" || ! -f "${ckpt}" ]]; then
    ckpt="${MODEL_DIR}/model_epoch$(printf '%03d' "${EXPECTED_FINAL_EPOCH}")_step<STEP>_final.pt"
  fi
  if (( DRY_RUN == 0 )); then
    [[ -n "${ckpt}" && -f "${ckpt}" ]] || { echo "[ERROR] evaluation checkpoint not found: ${ckpt:-NONE}" >&2; exit 1; }
  fi
  if [[ -f "${done_marker}" && -f "${out}/metrics.csv" ]]; then
    echo "[SKIP] ${task_upper} evaluation already complete"
    return
  fi
  mkdir -p "${out}" "${runtime}"

  local args=(
    sh/eval_airground_coop_v3.sh
    --gpu-ids "${EVAL_GPUS}"
    --render-gpu-ids "${RENDER_GPUS}"
    --workers-per-gpu "${WORKERS_PER_GPU}"
    --worker-start-stagger-sec "$(yaml_get evaluation.worker_start_stagger_sec)"
    --scene-timeout-seconds "$(yaml_get evaluation.scene_timeout_seconds)"
    --scene-retries "$(yaml_get evaluation.scene_retries)"
    --manifest "${manifest}"
    --ckpt "${ckpt}"
    --save-path "${out}"
    --runtime-root "${runtime}"
    --width "$(yaml_get evaluation.width)"
    --height "$(yaml_get evaluation.height)"
    --fps "$(yaml_get evaluation.fps)"
    --policy-inference-stride "${EVAL_STRIDE}"
    --policy-action-rollout "$(yaml_get evaluation.policy_action_rollout)"
    --history-frame-dt "$(yaml_get evaluation.history_frame_dt_seconds)"
    --waypoint-source-dt "$(yaml_get evaluation.waypoint_step_dt_seconds)"
    --waypoint-horizon-steps "$(yaml_get evaluation.waypoint_horizon_steps)"
    --metric-mask-stride "$(yaml_get evaluation.metric_mask_stride)"
    --drone-min-follow-dist "$(yaml_get evaluation.drone_min_follow_dist_m)"
    --drone-max-follow-dist "$(yaml_get evaluation.drone_max_follow_dist_m)"
    --robotdog-min-follow-dist "$(yaml_get evaluation.robotdog_min_follow_dist_m)"
    --robotdog-max-follow-dist "$(yaml_get evaluation.robotdog_max_follow_dist_m)"
  )
  if [[ "$(yaml_get evaluation.skip_rgb_between_policy_steps true)" == true ]]; then
    args+=(--skip-rgb-between-policy-steps)
  else
    args+=(--no-skip-rgb-between-policy-steps)
  fi
  [[ "$(yaml_get evaluation.save_video true)" == true ]] && args+=(--save-video)
  [[ "$(yaml_get evaluation.write_global_video true)" == true ]] && args+=(--write-global-video)
  [[ "$(yaml_get evaluation.trajectory_overlay true)" == true ]] && args+=(--trajectory-overlay)
  [[ "${resume_completed}" == true ]] || args+=(--no-resume)

  echo "[EVAL] start ${task_upper}: ${expected} episodes, videos enabled, stride=${EVAL_STRIDE}"
  cd "${PROJECT_ROOT}"
  if (( DRY_RUN == 1 )); then
    run_cmd bash "${args[@]}"
  else
    RUN_TAG="exp9_4_${task}_$(date +%Y%m%d_%H%M%S)" \
      PYTHON_BIN="${PYTHON_BIN}" \
      AIRGROUND_V3_TARGET_MATCH_THRESHOLD="${TARGET_MATCH_THRESHOLD}" \
      bash "${args[@]}"
    [[ -s "${out}/metrics.csv" ]] || { echo "[ERROR] ${task_upper} metrics.csv missing" >&2; exit 1; }
    cp -f "${out}/metrics.csv" "${SUMMARY_ROOT}/${task}_metrics.csv"
    printf '%s\n' "${out}/metrics.csv" >"${done_marker}"
    echo "[EVAL] complete ${task_upper}: ${out}/metrics.csv"
  fi
}

summarize_metrics() {
  local output_csv="${SUMMARY_ROOT}/metrics_all_tasks.csv"
  local output_md="${SUMMARY_ROOT}/metrics_summary.md"
  if (( DRY_RUN == 1 )); then
    echo "[DRY-RUN] summarize DT/AT/STT metrics -> ${output_csv}, ${output_md}"
    return
  fi
  "${PYTHON_BIN}" - "${SUMMARY_ROOT}" "${output_csv}" "${output_md}" <<'PY'
import csv, sys
from pathlib import Path
root, output_csv, output_md = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
rows=[]
for task in ("dt", "at", "stt"):
    path=root/f"{task}_metrics.csv"
    if not path.is_file():
        raise SystemExit(f"missing task metrics: {path}")
    with path.open(newline="", encoding="utf-8") as f:
        candidates=list(csv.DictReader(f))
    aggregate=next((r for r in candidates if r.get("row_type")=="aggregate"), None)
    if aggregate is None:
        raise SystemExit(f"aggregate row missing: {path}")
    rows.append({"task":task.upper(), **aggregate})
fields=["task"]+[k for k in rows[0] if k!="task"]
with output_csv.open("w", newline="", encoding="utf-8") as f:
    w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
keys=("episode_count","success_rate_pct","collision_rate_pct","human_collision_rate_pct","joint_tr_pct","drone_tr_pct","robotdog_tr_pct","visible_accuracy_pct","drone_bbox_iou_pct","robotdog_bbox_iou_pct","model_latency_ms","model_fps")
lines=["# exp9_4 DT / AT / STT 指标汇总", "", "| Task | Episodes | SR% | Collision% | Joint TR% | Drone TR% | Dog TR% | Visible Acc% | Model latency(ms) | Model FPS |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
for r in rows:
    lines.append("| {task} | {episode_count} | {success_rate_pct} | {collision_rate_pct} | {joint_tr_pct} | {drone_tr_pct} | {robotdog_tr_pct} | {visible_accuracy_pct} | {model_latency_ms} | {model_fps} |".format(**r))
lines += ["", "完整字段见 `metrics_all_tasks.csv`；每个任务的 episode 明细和 aggregate 行见 `<task>_metrics.csv`。", ""]
output_md.write_text("\n".join(lines), encoding="utf-8")
print(output_csv)
print(output_md)
PY
  printf '%s\n' "${output_csv}" >"${STATE_ROOT}/summarize.done"
}

preflight
print_plan
if [[ "${STAGE}" == plan ]]; then exit 0; fi

case "${STAGE}" in
  all)
    run_training
    run_eval_task dt
    run_eval_task at
    run_eval_task stt
    summarize_metrics
    ;;
  train) run_training ;;
  eval)
    run_eval_task dt
    run_eval_task at
    run_eval_task stt
    summarize_metrics
    ;;
  eval-dt) run_eval_task dt ;;
  eval-at) run_eval_task at ;;
  eval-stt) run_eval_task stt ;;
  summarize) summarize_metrics ;;
esac

echo "[DONE] stage=${STAGE}; log=${PIPELINE_LOG}"
