#!/usr/bin/env python3
"""Create a raw AT view from DT train episodes without modifying DT files."""
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path

INSTRUCTIONS = (
    "Track the person you first observe at the start.",
    "Follow the primary person identified at episode start.",
    "Pursue the first person selected at the beginning of the episode.",
    "Keep following the initially observed target person.",
    "Stay with the main person seen when the episode begins.",
    "Track the designated person first seen at the start.",
    "Follow the initial target person throughout the episode.",
    "Pursue the main target identified in the initial view.",
)

def instruction(ep: str, seed: int) -> tuple[int, str]:
    h = hashlib.sha256(f"{seed}:{ep}".encode()).digest()
    i = int.from_bytes(h[:8], "big") % len(INSTRUCTIONS)
    return i, INSTRUCTIONS[i]

def link(dst: Path, src: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() and os.path.realpath(dst) == os.path.realpath(src): return
        if dst.is_symlink():
            dst.unlink()
        else:
            raise FileExistsError(dst)
    dst.symlink_to(src)

def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument('--final-root',type=Path,default=Path('/data/hdt/ntv_data/data_final')); p.add_argument('--split-manifest',type=Path,default=Path('manifests/data_arr_train_test_v2/split_manifest.json')); p.add_argument('--seed',type=int,default=20260826); args=p.parse_args()
    final=args.final_root.resolve(); dt_root=final/'raw'/'data_arr'/'dt'; out=final/'raw'/'data_arr'/'at'; split=json.loads(args.split_manifest.read_text(encoding='utf-8'))
    train=[x for x in split['train'] if x.get('data_kind')=='dt']
    if len(train)!=1933: raise ValueError(f'expected 1933 DT train episodes, got {len(train)}')
    assignment=[]; missing=[]; linked=0
    for item in sorted(train,key=lambda x:x['episode_id']):
        parts=item['episode_id'].split('/'); scene,batch,stem=parts[1],parts[2],parts[3]; src=dt_root/scene/batch; dst=out/scene/batch; src_status=src/f'{stem}.json'; dst_status=dst/f'{stem}.json'
        if not src_status.is_file(): missing.append(str(src_status)); continue
        payload=json.loads(src_status.read_text(encoding='utf-8')); idx,text=instruction(item['episode_id'],args.seed); payload.update({'instruction':text,'task_variant':'at_raw_language_derived','source_data_kind':'dt','source_split':'train','target_selection_policy':'episode_initial_designated_target','target_identity_policy':'fixed_dt_target_pose_identity','distractors_are_non_target':True,'instruction_source':'generated_initial_target_at_v1','instruction_seed':args.seed,'instruction_index':idx,'source_status_json':str(src_status)})
        dst.mkdir(parents=True,exist_ok=True); dst_status.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        for f in src.iterdir():
            if f.name==f'{stem}.json': continue
            if f.name.startswith(f'{stem}_') or f.name==f'{stem}.mp4': link(dst/f.name,f); linked+=1
        assignment.append({'episode_id':item['episode_id'],'source_dir':str(src),'at_dir':str(dst),'instruction':text,'instruction_index':idx})
    if missing: raise FileNotFoundError(f'missing {len(missing)} DT status files; first={missing[0]}')
    (final/'manifests').mkdir(exist_ok=True); (final/'manifests'/'raw_at_train.json').write_text(json.dumps(assignment,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    summary={'schema_version':'data_final_raw_at_v1','episodes':len(assignment),'output_root':str(out),'source_root':str(dt_root),'linked_files':linked,'instruction_count':len(INSTRUCTIONS),'instruction_seed':args.seed,'raw_media_copied':False,'status_json_rewritten':True,'target_selection_policy':'episode_initial_designated_target','missing':missing}
    (out/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); (out/'README.md').write_text('# Raw AT train\n\nDerived from the 1,933 DT train episodes. Media and info files are symlinks to raw DT; only episode status JSON instruction/provenance fields are rewritten.\n',encoding='utf-8'); print(json.dumps(summary,ensure_ascii=False,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
