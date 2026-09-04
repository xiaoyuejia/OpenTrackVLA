#!/usr/bin/env python3
"""List exact AT-validation frames from the mirrored eval JSONL tree."""
from pathlib import Path
import argparse, os

def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument('--json-root',type=Path,default=Path('/data/yh/data/processed/eval_jsonl/at')); p.add_argument('--frame-root',type=Path,default=Path('/data/yh/data/processed/frames/at')); p.add_argument('--output',type=Path,default=Path('/data/yh/newtrackvla修改/newtrackvla_base_yh_clean/cache_lists/at_eval_frames.txt')); a=p.parse_args()
    frames=[]
    for jsonl in sorted(a.json_root.rglob('*.jsonl')):
        rel=jsonl.relative_to(a.json_root).with_suffix('')
        episode=a.frame_root/rel
        for view in ('drone','robotdog'):
            frames.extend(sorted((episode/view).glob('*.jpg')))
    if not frames: raise FileNotFoundError(f'No AT validation frames under {a.frame_root}')
    a.output.parent.mkdir(parents=True,exist_ok=True); temporary=a.output.with_name(f'.{a.output.name}.tmp-{os.getpid()}')
    temporary.write_text(''.join(f'{path.resolve()}\n' for path in frames),encoding='utf-8'); os.replace(temporary,a.output)
    print(f'episodes={len(list(a.json_root.rglob("*.jsonl")))} frames={len(frames)} output={a.output}')
    return 0
if __name__=='__main__': raise SystemExit(main())
