#!/usr/bin/env python3
"""Attach reviewed, appearance-id keyed descriptions to AT replay human metadata."""
from __future__ import annotations
import argparse, json
from pathlib import Path

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--replay-root', type=Path, default=Path('/data/yh/data/raw/at_eval_replay'))
    p.add_argument('--catalog', type=Path, default=Path(__file__).with_name('appearance_catalog_at_replay.json'))
    args = p.parse_args()
    cat = json.loads(args.catalog.read_text(encoding='utf-8'))['appearances']
    changed = 0
    for hpath in args.replay_root.rglob('*_human.json'):
        obj = json.loads(hpath.read_text(encoding='utf-8'))
        aid = str(obj.get('appearance', {}).get('appearance_id'))
        if aid not in cat:
            continue
        obj['appearance']['description'] = cat[aid]
        obj['appearance']['description_source'] = str(args.catalog)
        tmp = hpath.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        tmp.replace(hpath)
        changed += 1
    print(json.dumps({'human_json_updated': changed, 'catalog_entries': len(cat)}, ensure_ascii=False))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
