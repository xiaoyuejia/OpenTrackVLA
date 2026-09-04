#!/usr/bin/env python3
"""Replace DT-replay instructions with appearance descriptions, atomically."""
import argparse, json
from pathlib import Path

EN = {
  1:'a dark gray/black long-sleeve top and light gray trousers', 2:'a dark green top and dark trousers',
  3:'all-dark clothing', 4:'a white long-sleeve top and blue jeans', 5:'a dark patterned jacket and dark trousers',
  6:'a black jacket and black trousers', 7:'a purple-blue top and dark trousers', 8:'a brown top or jacket and light gray trousers',
  9:'a blue top or jacket and dark trousers', 10:'a burgundy top and light gray trousers', 11:'a white top and blue trousers',
  12:'a dark gray top and dark trousers', 13:'a brown-gray outer layer over a light top and blue trousers',
  14:'a burgundy top and light gray trousers', 15:'an orange-brown short-sleeve top and blue jeans',
  16:'all-dark clothing', 17:'a light gray/gray-brown top and dark trousers', 18:'all-dark clothing'
}
def main():
 p=argparse.ArgumentParser(); p.add_argument('root',type=Path); p.add_argument('--dry-run',action='store_true'); a=p.parse_args(); n=rows=0
 for f in a.root.rglob('*.jsonl'):
  tmp=f.with_suffix('.jsonl.tmp'); changed=False
  with f.open(encoding='utf-8') as r, tmp.open('w',encoding='utf-8') as w:
   for line in r:
    if not line.strip(): continue
    o=json.loads(line); aid=int(o.get('target_appearance_id',0));
    if aid in EN:
     o.setdefault('instruction_id_based',o.get('instruction'))
     o['instruction']=f'Follow the target person wearing {EN[aid]}; ignore all other people and avoid collisions.'; changed=True
    w.write(json.dumps(o,ensure_ascii=False,separators=(',',':'))+'\n'); rows+=1
  if changed and not a.dry_run: tmp.replace(f); n+=1
  else: tmp.unlink(missing_ok=True)
 print(json.dumps({'files_updated':n,'rows_scanned':rows}))
if __name__=='__main__': main()
