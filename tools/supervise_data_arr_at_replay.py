#!/usr/bin/env python3
"""Externally supervise one conflict-free AT replay shard episode by episode."""
from __future__ import annotations
import argparse, json, os, signal, subprocess, sys, time
from pathlib import Path

def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_name(f'.{path.name}.tmp.{os.getpid()}'); tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n'); tmp.replace(path)

def kill_group(proc: subprocess.Popen) -> None:
    if proc.poll() is not None: return
    try: os.killpg(proc.pid,signal.SIGTERM)
    except ProcessLookupError: return
    try: proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try: os.killpg(proc.pid,signal.SIGKILL)
        except ProcessLookupError: pass
        proc.wait(timeout=10)

def main() -> int:
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--source-root',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);p.add_argument('--render-gpu',type=int,required=True);p.add_argument('--worker-runtime',type=Path,required=True);p.add_argument('--port-lock',type=Path,required=True);p.add_argument('--shard-index',type=int,required=True);p.add_argument('--shard-count',type=int,required=True);p.add_argument('--exclude-scenes',default='UnrealTrack-KoreanPalace-ContinuousColor-v0');p.add_argument('--startup-timeout-s',type=float,default=600);p.add_argument('--heartbeat-timeout-s',type=float,default=300);p.add_argument('--active-timeout-s',type=float,default=5400);p.add_argument('--attempts',type=int,default=3);p.add_argument('--poll-s',type=float,default=10);p.add_argument('--log',type=Path,required=True);args=p.parse_args()
    items=json.loads(args.plan.read_text()); excluded={x.strip() for x in args.exclude_scenes.split(',') if x.strip()}; shard=[x for i,x in enumerate(items) if i%args.shard_count==args.shard_index and x['scene'] not in excluded]
    args.log.parent.mkdir(parents=True,exist_ok=True); binary=args.worker_runtime/'Linux/UnrealZoo_UE5_6/Binaries/Linux/UnrealZoo_UE5_6'; replay=Path(__file__).with_name('replay_data_arr_at.py'); completed=skipped=failed=0
    with args.log.open('a',encoding='utf-8') as log:
      def emit(message): print(message,flush=True); print(message,file=log,flush=True)
      emit(f'[supervisor] gpu={args.render_gpu} shard={args.shard_index}/{args.shard_count} items={len(shard)} excluded={sorted(excluded)}')
      for ordinal,item in enumerate(shard,1):
        out=args.output_root/item['scene']/item['source_batch']; stem=item['stem']; marker=out/f'{stem}.complete.json'; progress=out/f'{stem}.progress.json'
        if marker.is_file(): skipped+=1; continue
        ok=False
        for attempt in range(1,args.attempts+1):
          progress.unlink(missing_ok=True); (out/f'.{stem}_drone.tmp.mp4').unlink(missing_ok=True); (out/f'.{stem}_robotdog.tmp.mp4').unlink(missing_ok=True)
          env=os.environ.copy();env.update({'CUDA_VISIBLE_DEVICES':str(args.render_gpu),'UNREALZOO_FAST_ENV_ID':item['scene'],'UNREALZOO_ENV_BIN':str(binary),'UNREALZOO_PORT_LOCK':str(args.port_lock),'UNREALZOO_FIXED_TIMESTEP':'0.1','UNREALZOO_SKIP_FULL_COLOR_DICT':'1','UNREALZOO_REQUEST_TIMEOUT_S':'120','PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1'})
          cmd=[sys.executable,'-u',str(replay),'--plan',str(args.plan),'--episode-id',item['episode_id'],'--source-root',str(args.source_root),'--output-root',str(args.output_root),'--render-gpu',str(args.render_gpu),'--episode-retries','0','--startup-timeout-s','86400','--episode-timeout-s',str(args.active_timeout_s),'--snapshot-mode','batch','--snapshot-attempts','3','--snapshot-render-sync','0.02','--resume']
          emit(f'[start] gpu={args.render_gpu} {ordinal}/{len(shard)} attempt={attempt} episode={item["episode_id"]}')
          proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True); started=time.time(); first_step=None; reason='process_exit'
          while True:
            if marker.is_file(): ok=True; reason='complete'; break
            rc=proc.poll()
            if rc is not None: reason=f'exit_{rc}'; break
            now=time.time()
            if progress.is_file():
              try: heartbeat=json.loads(progress.read_text()); updated=float(heartbeat['updated_unix_s']); first_step=first_step or now
              except Exception: updated=now
              if now-updated>args.heartbeat_timeout_s: reason=f'heartbeat_stale_{now-updated:.0f}s'; break
              if first_step and now-first_step>args.active_timeout_s: reason='active_timeout'; break
            elif now-started>args.startup_timeout_s: reason='startup_timeout_frames_0'; break
            time.sleep(args.poll_s)
          if not ok: kill_group(proc); atomic_json(out/f'{stem}.supervisor_failure.json',{'episode_id':item['episode_id'],'attempt':attempt,'reason':reason,'unix_s':time.time()}); emit(f'[retry] gpu={args.render_gpu} episode={item["episode_id"]} reason={reason}')
          else:
            try: proc.wait(timeout=30)
            except subprocess.TimeoutExpired: kill_group(proc)
            completed+=1; emit(f'[complete] gpu={args.render_gpu} episode={item["episode_id"]}'); break
        if not ok: failed+=1; emit(f'[failed] gpu={args.render_gpu} episode={item["episode_id"]} attempts={args.attempts}')
      emit(f'[done] gpu={args.render_gpu} completed_now={completed} skipped={skipped} failed={failed}')
    return 0 if failed==0 else 2
if __name__=='__main__': raise SystemExit(main())
