#!/usr/bin/env python3
"""Final episode-level integrity audit for /data/hdt/ntv_data/cyj/data_arr."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/data/hdt/ntv_data/cyj/data_arr")
REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"
QUARANTINE = Path("/data/hdt/ntv_data/cyj/data_arr_quarantine_final_audit_20260822")


def bbox_valid(value) -> bool:
    try:
        return len(value) >= 4 and all(math.isfinite(float(v)) for v in value[:4]) and float(value[2]) > 0 and float(value[3]) > 0
    except Exception:
        return False


def longest_run(flags: list[bool]) -> tuple[int, int, int]:
    best = (0, -1, -1); start = 0
    for i, flag in enumerate(flags + [False]):
        if flag and (i == 0 or not flags[i - 1]): start = i
        if not flag and i > 0 and flags[i - 1] and i - start > best[0]: best = (i - start, start, i - 1)
    return best


def pose(row: dict, agent: str):
    value = row.get(f"{agent}_pose") or row.get(f"{agent}_pose_after_action")
    try:
        a = np.asarray(value[:3], dtype=float)
        return a if a.size == 3 and np.isfinite(a).all() else None
    except Exception:
        return None


def command_active(row: dict, agent: str) -> bool:
    value = row.get("controller_commanded_base_velocity") or row.get("commanded_base_velocity") or row.get("env_action")
    try:
        a = np.asarray(value, dtype=float).reshape(-1)
        if agent == "robotdog" and "env_action" in row and len(np.asarray(row["env_action"]).reshape(-1)) >= 2:
            return abs(float(np.asarray(row["env_action"]).reshape(-1)[1])) >= 5.0
        return a.size >= 2 and float(np.linalg.norm(a[:2])) >= 0.05
    except Exception:
        return False


def agent_metrics(rows: list[dict], agent: str) -> dict:
    n = len(rows)
    valid = [bbox_valid(r.get("target_bbox")) for r in rows]
    visible = [bool(r.get("target_visible", False)) for r in rows]
    conflicts = [v and not b for v, b in zip(visible, valid)]
    zero_run, zero_start, zero_end = longest_run([not x for x in valid])
    conflict_run, _, _ = longest_run(conflicts)
    poses = [pose(r, agent) for r in rows]
    deltas = []
    for a, b in zip(poses, poses[1:]):
        deltas.append(float(np.linalg.norm(b - a) / 100.0) if a is not None and b is not None else float("nan"))
    static_flags = [False] + [math.isfinite(d) and d < 0.01 for d in deltas]
    static_run, static_start, static_end = longest_run(static_flags)
    active = [command_active(r, agent) for r in rows]
    active_in_static = (sum(active[static_start:static_end + 1]) / static_run) if static_run else 0.0
    finite_delta = [d for d in deltas if math.isfinite(d)]
    return {
        f"{agent}_bbox_valid": sum(valid),
        f"{agent}_bbox_zero_run": zero_run,
        f"{agent}_visible_zero": sum(conflicts),
        f"{agent}_visible_zero_run": conflict_run,
        f"{agent}_pose_missing": sum(p is None for p in poses),
        f"{agent}_path_m": sum(finite_delta),
        f"{agent}_max_step_m": max(finite_delta, default=0.0),
        f"{agent}_static_run": static_run,
        f"{agent}_static_active_ratio": active_in_static,
    }


def video_metrics(path: Path, expected: int) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened(): return {"open": False, "frames": 0, "width": 0, "height": 0, "blank_rate": 1.0, "freeze_run": expected}
    count = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT))); width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sample_indices = sorted(set(range(0, max(count, 1), 10)) | ({count - 1} if count else set()))
    previous = None; freeze_flags=[]; blank=0; decoded=0
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx); ok, frame = cap.read()
        if not ok: continue
        gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 120))
        decoded += 1
        if float(gray.std()) < 2.0 or float(np.mean(gray < 3)) > 0.995 or float(np.mean(gray > 252)) > 0.995: blank += 1
        if previous is not None: freeze_flags.append(float(np.mean(cv2.absdiff(gray, previous))) < 0.15)
        previous = gray
    cap.release()
    freeze_samples = longest_run(freeze_flags)[0]
    return {"open": True, "frames": count, "width": width, "height": height, "blank_rate": blank / decoded if decoded else 1.0, "freeze_run": freeze_samples * 10}


def inspect_episode(drone_path: Path, with_video: bool) -> dict:
    suffix = "_drone_info.json"; ep = drone_path.name[:-len(suffix)]; directory = drone_path.parent
    rel = str(directory.relative_to(ROOT)); result = {"relative_dir": rel, "episode": ep}
    paths = {
        "drone_info": drone_path, "robotdog_info": directory / f"{ep}_robotdog_info.json",
        "episode_json": directory / f"{ep}.json", "drone_video": directory / f"{ep}_drone.mp4", "robotdog_video": directory / f"{ep}_robotdog.mp4",
    }
    result["missing_files"] = ";".join(k for k, p in paths.items() if not p.is_file())
    parse_errors=[]; agents={}
    for agent in ("drone", "robotdog"):
        try:
            value=json.loads(paths[f"{agent}_info"].read_text()); agents[agent]=value if isinstance(value,list) else []
            if not isinstance(value,list): parse_errors.append(f"{agent}_info_not_list")
        except Exception as exc: agents[agent]=[]; parse_errors.append(f"{agent}_info:{type(exc).__name__}")
    try: json.loads(paths["episode_json"].read_text())
    except Exception as exc: parse_errors.append(f"episode_json:{type(exc).__name__}")
    result["parse_errors"]=";".join(parse_errors); result["drone_rows"]=len(agents["drone"]); result["robotdog_rows"]=len(agents["robotdog"])
    n=max(len(agents["drone"]),len(agents["robotdog"])); result["frames"]=n
    for agent in ("drone","robotdog"): result.update(agent_metrics(agents[agent],agent))
    result.update({"drone_video_frames":-1,"robotdog_video_frames":-1,"drone_blank_rate":-1,"robotdog_blank_rate":-1,"drone_freeze_run":-1,"robotdog_freeze_run":-1})
    if with_video:
        for agent in ("drone","robotdog"):
            vm=video_metrics(paths[f"{agent}_video"],len(agents[agent]))
            result[f"{agent}_video_frames"]=vm["frames"]; result[f"{agent}_blank_rate"]=vm["blank_rate"]; result[f"{agent}_freeze_run"]=vm["freeze_run"]
            if not vm["open"]: result["parse_errors"] += (";" if result["parse_errors"] else "") + f"{agent}_video_open"
    reasons=[]
    if result["missing_files"] or result["parse_errors"]: reasons.append("structural_failure")
    if not n or len(agents["drone"]) != len(agents["robotdog"]): reasons.append("agent_frame_count_mismatch")
    if with_video and any(result[f"{a}_video_frames"] != len(agents[a]) for a in ("drone","robotdog")): reasons.append("video_json_frame_count_mismatch")
    for agent in ("drone","robotdog"):
        rows=len(agents[agent]); valid=result[f"{agent}_bbox_valid"]
        if rows and valid == 0: reasons.append(f"{agent}_full_bbox_missing")
        # loststart/lostmid intentionally contain long target-loss intervals.
        # In ordinary collections a >=50-frame continuous bbox gap is a
        # high-confidence annotation/camera failure (visually reviewed).
        path_parts = set(Path(rel).parts)
        intentional_loss = bool(path_parts & {"stt_camera3_loststart", "stt_camera3_lostmid"})
        if not intentional_loss and result[f"{agent}_bbox_zero_run"] >= 50:
            reasons.append(f"{agent}_long_bbox_gap")
        if rows and (result[f"{agent}_visible_zero"] / rows >= 0.5 or result[f"{agent}_visible_zero_run"] >= 150): reasons.append(f"{agent}_severe_visible_bbox_conflict")
        if result[f"{agent}_pose_missing"] > 0: reasons.append(f"{agent}_pose_invalid")
        if result[f"{agent}_max_step_m"] > 10.0: reasons.append(f"{agent}_pose_jump")
        if result[f"{agent}_static_run"] >= 150 and result[f"{agent}_static_active_ratio"] >= 0.5: reasons.append(f"{agent}_long_stuck")
        if with_video and result[f"{agent}_freeze_run"] >= 150: reasons.append(f"{agent}_video_frozen")
        if with_video and result[f"{agent}_blank_rate"] >= 0.5: reasons.append(f"{agent}_video_blank")
    result["isolate"] = bool(reasons); result["reasons"]=";".join(dict.fromkeys(reasons))
    return result


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--execute",action="store_true"); ap.add_argument("--skip-video",action="store_true"); ap.add_argument("--threads",type=int,default=8); args=ap.parse_args()
    episodes=sorted(ROOT.rglob("*_drone_info.json")); print(f"episodes={len(episodes)} video={not args.skip_video}",flush=True)
    if args.skip_video: rows=[inspect_episode(p,False) for p in episodes]
    else:
        with ThreadPoolExecutor(max_workers=args.threads) as pool: rows=list(pool.map(lambda p:inspect_episode(p,True),episodes))
    REPORT_DIR.mkdir(parents=True,exist_ok=True); report=REPORT_DIR/"data_arr_final_audit_20260822.csv"
    with report.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    bad=[r for r in rows if r["isolate"]]
    summary={"created_at":datetime.now().isoformat(),"root":str(ROOT),"episodes":len(rows),"clean":len(rows)-len(bad),"isolate":len(bad),"reason_counts":{reason:sum(reason in r["reasons"].split(";") for r in bad) for reason in sorted({x for r in bad for x in r["reasons"].split(";") if x})},"report":str(report)}
    (REPORT_DIR/"data_arr_final_audit_20260822_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    if not args.execute: print("dry-run; pass --execute after reviewing candidates",flush=True); return
    operations=[]
    for row in bad:
        directory=ROOT/row["relative_dir"]; ep=row["episode"]; files=sorted(directory.glob(f"{ep}_*")); stat=directory/f"{ep}.json"
        if stat.is_file(): files.append(stat)
        for src in files:
            dst=QUARANTINE/row["relative_dir"]/src.name; dst.parent.mkdir(parents=True,exist_ok=True); shutil.move(str(src),str(dst)); operations.append({"source":str(src),"destination":str(dst),"relative_dir":row["relative_dir"],"episode":ep,"reasons":row["reasons"]})
    manifest={"created_at":datetime.now().isoformat(),"source_root":str(ROOT),"quarantine_root":str(QUARANTINE),"episodes":len(bad),"files":operations}
    (QUARANTINE/"rollback_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    print(f"moved_episodes={len(bad)} moved_files={len(operations)} quarantine={QUARANTINE}")


if __name__ == "__main__": main()
