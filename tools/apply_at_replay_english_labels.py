#!/usr/bin/env python3
"""Apply the reviewed English appearance catalog to AT replay and DT-replay."""
from __future__ import annotations
import argparse, json, os
from pathlib import Path

def atomic_json(path: Path, obj: object) -> None:
    tmp=path.with_name(f'.{path.name}.tmp-{os.getpid()}')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); os.replace(tmp,path)

def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument('--catalog',type=Path,default=Path(__file__).with_name('appearance_catalog_at_replay.json')); p.add_argument('--replay-root',type=Path,default=Path('/data/yh/data/raw/at_eval_replay')); p.add_argument('--dt-replay-root',type=Path,default=Path('/data/yh/data/processed/eval_jsonl/dt')); p.add_argument('--manifest',type=Path,default=Path('/data/yh/data/manifests/eval_dt.json')); a=p.parse_args()
    raw=json.loads(a.catalog.read_text(encoding='utf-8')); catalog=raw['appearances']; desc={str(k):(v['description'] if isinstance(v,dict) else str(v)) for k,v in catalog.items()}
    confidence={str(k):v.get('confidence') for k,v in catalog.items() if isinstance(v,dict)}; evidence={str(k):v.get('evidence') for k,v in catalog.items() if isinstance(v,dict)}
    human=0
    for path in a.replay_root.rglob('*_human.json'):
        obj=json.loads(path.read_text(encoding='utf-8')); aid=str(obj.get('appearance',{}).get('appearance_id'))
        if aid not in desc: continue
        obj.setdefault('appearance',{})['description']=desc[aid]; obj['appearance']['description_language']='en'; obj['appearance']['description_confidence']=confidence.get(aid); obj['appearance']['description_evidence']=evidence.get(aid); obj['appearance']['description_source']=str(a.catalog)
        atomic_json(path,obj); human+=1
    rows=0; files=0
    for path in a.dt_replay_root.rglob('*.jsonl'):
        tmp=path.with_name(f'.{path.name}.tmp-{os.getpid()}'); changed=False
        with path.open(encoding='utf-8') as src,tmp.open('w',encoding='utf-8') as dst:
            for line in src:
                if not line.strip(): continue
                obj=json.loads(line); aid=str(obj.get('target_appearance_id',obj.get('appearance_id','')))
                if aid in desc:
                    old=obj.get('instruction'); obj.setdefault('instruction_id_based',old); obj['instruction']=f'Follow the target person described as {desc[aid]}. Ignore all other people and avoid collisions.'; obj['target_description']=desc[aid]; obj['target_description_language']='en'; changed=True
                dst.write(json.dumps(obj,ensure_ascii=False,separators=(',',':'))+'\n'); rows+=1
        if changed: tmp.replace(path); files+=1
        else: tmp.unlink(missing_ok=True)
    manifest=json.loads(a.manifest.read_text(encoding='utf-8')); manifest_rows=0
    for obj in manifest:
        aid=str(obj.get('target_appearance_id',obj.get('appearance_id','')))
        if aid in desc:
            obj.setdefault('instruction_id_based',obj.get('instruction')); obj['instruction']=f'Follow the target person described as {desc[aid]}. Ignore all other people and avoid collisions.'; obj['target_description']=desc[aid]; obj['target_description_language']='en'; manifest_rows+=1
    atomic_json(a.manifest,manifest)
    print(json.dumps({'human_json_updated':human,'jsonl_files_updated':files,'jsonl_rows':rows,'manifest_rows_updated':manifest_rows,'catalog_entries':len(desc)},ensure_ascii=False)); return 0
if __name__=='__main__': raise SystemExit(main())
