#!/usr/bin/env python3
"""Update DT raw status JSON and processed train/eval JSONL instructions."""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
from update_dt_instructions_from_human import make_instruction

RAW = Path('/data/yh/data/raw/data_arr/dt')
PROC = Path('/data/yh/data/processed')

def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument('--raw-root',type=Path,default=RAW); ap.add_argument('--processed-root',type=Path,default=PROC); args=ap.parse_args()
    raw=args.raw_root.resolve(); processed=args.processed_root.resolve(); descriptions={}
    missing=[]; conflicts=[]
    for d in sorted({p.parent for p in raw.rglob('*_drone_info.json')}):
        hs=sorted(d.glob('human*.json'))
        if not hs: missing.append(d); continue
        ins={make_instruction(json.loads(h.read_text(encoding='utf-8'))) for h in hs}
        if len(ins)!=1: conflicts.append((d,hs)); continue
        descriptions[d.relative_to(raw)] = next(iter(ins))
    raw_changed=0
    for rel, instruction in descriptions.items():
        d=raw/rel
        for path in sorted(d.glob('[0-9]*.json')):
            if path.name.endswith(('_drone_info.json','_robotdog_info.json')): continue
            try: obj=json.loads(path.read_text(encoding='utf-8'))
            except json.JSONDecodeError: continue
            if not isinstance(obj,dict) or 'instruction' not in obj: continue
            if obj.get('instruction') != instruction: raw_changed += 1
            obj['instruction']=instruction; obj['instruction_source']='human_appearance'
            tmp=path.with_suffix(path.suffix+f'.tmp.{os.getpid()}'); tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); tmp.replace(path)
    jsonl_changed=0; jsonl_files=0
    for split in ('train_jsonl','eval_jsonl'):
        root=processed/split/'dt'
        for path in sorted(root.rglob('*.jsonl')):
            rel=path.parent.relative_to(root); instruction=descriptions.get(rel)
            if instruction is None: continue
            tmp=path.with_suffix(path.suffix+f'.tmp.{os.getpid()}'); rows=0
            with path.open(encoding='utf-8') as r,tmp.open('w',encoding='utf-8') as w:
                for line in r:
                    if not line.strip(): continue
                    obj=json.loads(line); rows+=1
                    if obj.get('instruction') != instruction: jsonl_changed+=1
                    obj['instruction']=instruction; obj['instruction_source']='human_appearance'
                    w.write(json.dumps(obj,ensure_ascii=False,separators=(',',':'))+'\n')
            tmp.replace(path); jsonl_files+=1
    print(json.dumps({'human_dirs':len(descriptions),'missing_human_dirs':len(missing),'conflicts':len(conflicts),'raw_status_changed':raw_changed,'processed_jsonl_files':jsonl_files,'processed_rows_changed':jsonl_changed},ensure_ascii=False))
    if missing: print('[missing-human]'); print('\n'.join(str(x) for x in missing))
    if conflicts: print('[conflict-human]'); print('\n'.join(str(x[0]) for x in conflicts))
    return 0
if __name__=='__main__': raise SystemExit(main())
