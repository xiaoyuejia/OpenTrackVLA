#!/usr/bin/env python3
"""Materialize AT replay target appearance metadata and update DT-replay text."""
from __future__ import annotations
import argparse, json, os
from pathlib import Path

def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument('--replay-root',type=Path,default=Path('/data/yh/data/raw/at_eval_replay')); p.add_argument('--dt-replay-root',type=Path,default=Path('/data/yh/data/processed/eval_jsonl/dt')); p.add_argument('--manifest',type=Path,default=Path('/data/yh/data/manifests/eval_dt.json')); args=p.parse_args()
    entries=json.loads(args.manifest.read_text(encoding='utf-8')); changed=0; created=0
    for item in entries:
        eid=str(item['episode_id']).removeprefix('dt_replay/'); parts=eid.split('/')
        if len(parts)!=3: raise ValueError(eid)
        scene,run,stem=parts; d=args.replay_root/scene/run; meta=d/f'{stem}_at_episode.json'
        if not meta.is_file(): raise FileNotFoundError(meta)
        at=json.loads(meta.read_text(encoding='utf-8')); target_id=int(at['appearance_id'])
        human={
            'person_id': f'AT_REPLAY_APPEARANCE_{target_id:02d}',
            'appearance': {'appearance_id': target_id, 'appearance_mode': at.get('appearance_mode'), 'source': str(meta)},
            'replay': {'episode_id': at.get('episode_id'), 'target_actor_slot': at.get('target_actor_slot'), 'human_count': at.get('human_count'), 'distractor_count': at.get('distractor_count'), 'human_appearance_ids': at.get('human_appearance_ids'), 'appearance_map': at.get('appearance_map')},
        }
        hpath=d/f'{stem}_human.json'; tmp=hpath.with_suffix('.json.tmp'); tmp.write_text(json.dumps(human,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); tmp.replace(hpath); created+=1
        instruction=f'Follow the target person with appearance ID {target_id}; ignore all other people and avoid collisions.'
        jp=args.dt_replay_root/scene/run/f'{stem}.jsonl'
        if not jp.is_file(): raise FileNotFoundError(jp)
        tmp=jp.with_suffix('.jsonl.tmp')
        with jp.open(encoding='utf-8') as r,tmp.open('w',encoding='utf-8') as w:
            for line in r:
                if not line.strip(): continue
                obj=json.loads(line); obj['instruction']=instruction; obj['target_appearance_id']=target_id; obj['target_appearance_source']=str(hpath); w.write(json.dumps(obj,ensure_ascii=False,separators=(',',':'))+'\n'); changed+=1
        tmp.replace(jp)
        item['instruction']=instruction; item['target_appearance_id']=target_id; item['target_appearance_source']=str(hpath)
    tmp=args.manifest.with_suffix('.json.tmp'); tmp.write_text(json.dumps(entries,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); tmp.replace(args.manifest)
    print(json.dumps({'episodes':len(entries),'human_json_created':created,'jsonl_rows_updated':changed},ensure_ascii=False))
    return 0
if __name__=='__main__': raise SystemExit(main())
