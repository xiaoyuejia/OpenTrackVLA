#!/usr/bin/env python3
"""Build DT/AT replay manifests with per-episode instruction and distractors."""
from __future__ import annotations
import json
from pathlib import Path

RAW=Path('/data/yh/data/raw/at_eval_replay')
DATA=Path('/data/yh/data/processed')
OUT=Path('/data/yh/data/manifests/details')
CATALOG=Path('/data/yh/newtrackvla修改/newtrackvla_base_yh_clean/repo/tools/appearance_catalog_at_replay.json')

def main() -> int:
    dt_entries=json.loads((Path('/data/yh/data/manifests/details/eval_dt.json')).read_text(encoding='utf-8'))
    at_entries=json.loads((Path('/data/yh/data/manifests/details/eval_at.json')).read_text(encoding='utf-8'))
    catalog=json.loads(CATALOG.read_text(encoding='utf-8'))['appearances']
    at_by_id={str(item['episode_id']).replace('dt/',''): item for item in at_entries}
    outputs={}
    for kind,entries in [('dt',dt_entries),('at',at_entries)]:
        result=[]
        for item in entries:
            episode=str(item['episode_id']).split('/',1)[-1]
            scene,run,stem=episode.split('/')
            raw_dir=RAW/scene/run
            at_meta=raw_dir/f'{stem}_at_episode.json'
            info=raw_dir/f'{stem}_drone_info.json'
            if not at_meta.is_file() or not info.is_file(): raise FileNotFoundError(raw_dir/stem)
            out=dict(item)
            out.update({
                'scene':scene,'stem':stem,'relative_dir':f'at_eval_replay/{scene}/{run}',
                'info':str(info),'replay_meta':str(at_meta),
                'source_data_kind':'at_eval_replay','replay_distractors':True,
                'replay_motion_policy':'recorded_distractor_actions',
            })
            if kind=='at':
                source=DATA/'eval_jsonl/at'/scene/run/f'{stem}.jsonl'
                first=json.loads(source.open(encoding='utf-8').readline())
                # AT is intentionally the language-fuzzy condition. Keep its
                # original instruction so the AT protocol tests visual search
                # without giving the appearance description to the policy.
                out['instruction']=first['instruction']; out['task_type']='at_replay'; out['data_kind']='at'
            else:
                out['instruction']=item['instruction']; out['task_type']='dt_replay'; out['data_kind']='dt'
            result.append(out)
        target=OUT/f'eval_{kind}_replay_recorded.json'; target.write_text(json.dumps({'test':result},ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); outputs[kind]=(str(target),len(result))
    print(json.dumps(outputs,ensure_ascii=False)); return 0
if __name__=='__main__': raise SystemExit(main())
