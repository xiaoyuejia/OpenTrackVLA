#!/usr/bin/env python3
"""Validate replay manifest instructions and target/distractor trajectory coverage."""
import argparse,json
from pathlib import Path

def main():
 p=argparse.ArgumentParser(); p.add_argument('--manifest',type=Path,required=True); a=p.parse_args(); entries=json.loads(a.manifest.read_text())['test']; errors=[]; instructions=[]
 for item in entries:
  for key in ('scene','stem','info','replay_meta','instruction'): 
   if not item.get(key): errors.append(f'{item.get("episode_id")}: missing {key}')
  if 'appearance ID' in item.get('instruction',''): errors.append(f'{item.get("episode_id")}: id instruction')
  try:
   drone=json.load(open(item['info'])); meta=json.load(open(item['replay_meta']))
   if len(drone)!=len(meta.get('target_poses_per_frame',[])): errors.append(f'{item["episode_id"]}: target length mismatch')
   if len(meta.get('distractor_poses_per_frame',[]))<len(drone): errors.append(f'{item["episode_id"]}: distractor poses short')
   if len(meta.get('distractor_actions_per_frame',[]))<len(drone): errors.append(f'{item["episode_id"]}: distractor actions short')
   if int(meta.get('distractor_count',0)) != len(meta.get('human_appearance_ids',[]))-1: errors.append(f'{item["episode_id"]}: appearance count mismatch')
  except Exception as exc: errors.append(f'{item.get("episode_id")}: {exc}')
  instructions.append(item['instruction'])
 print(f'episodes={len(entries)} errors={len(errors)} unique_instructions={len(set(instructions))}')
 for e in errors[:20]: print('ERROR',e)
 return 1 if errors else 0
if __name__=='__main__': raise SystemExit(main())
