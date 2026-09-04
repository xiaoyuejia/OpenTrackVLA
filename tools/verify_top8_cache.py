#!/usr/bin/env python3
"""Verify candidate bundles and/or vision-token pairs for an exact frame list."""
import argparse
from pathlib import Path

def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument('--frame-list',type=Path,required=True); p.add_argument('--data-root',type=Path,default=Path('/data/yh/data/processed')); p.add_argument('--candidate-root',type=Path); p.add_argument('--vision-root',type=Path); a=p.parse_args()
    frames=[Path(line) for line in a.frame_list.read_text(encoding='utf-8').splitlines() if line.strip()]
    missing=[]; views={frame.parent for frame in frames}
    if a.candidate_root:
        for view in views:
            rel=view.resolve().relative_to(a.data_root.resolve()); target=a.candidate_root/rel.parent/f'{rel.name}.candidates.npz'
            if not target.is_file(): missing.append(str(target))
    vision_pairs=0
    if a.vision_root:
        for frame in frames:
            rel=frame.resolve().relative_to(a.data_root.resolve()); directory=a.vision_root/rel.parent
            for suffix in ('_vcoarse.pt','_vfine.pt'):
                target=directory/f'{rel.stem}{suffix}'
                if not target.is_file(): missing.append(str(target))
            vision_pairs+=1
    print(f'frames={len(frames)} views={len(views)} candidate_checked={bool(a.candidate_root)} vision_pairs={vision_pairs} missing={len(missing)}')
    for value in missing[:20]: print('MISSING',value)
    return 1 if missing else 0
if __name__=='__main__': raise SystemExit(main())
