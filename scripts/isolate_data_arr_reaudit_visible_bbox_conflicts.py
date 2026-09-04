#!/usr/bin/env python3
import argparse,csv,json,shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
ROOT=Path('/data/hdt/ntv_data/cyj/data_arr')
QUARANTINE=Path('/data/hdt/ntv_data/cyj/隔离/data_arr_reaudit_20260822')
REPORT=Path(__file__).resolve().parents[1]/'reports/data_arr_final_audit_20260822.csv'
SEMANTIC=Path(__file__).resolve().parents[1]/'reports/data_arr_bbox_semantic_reaudit_20260822.csv'
SUMMARY=Path(__file__).resolve().parents[1]/'reports/data_arr_final_audit_20260822_summary.json'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');ap.add_argument('--isolate-visible-bbox-conflicts',action='store_true',help='Legacy strict mode; normally these frames are retained and masked in training.');args=ap.parse_args()
 rows=list(csv.DictReader(REPORT.open(encoding='utf-8-sig'))); sem={(r['relative_dir'],r['episode']):r for r in csv.DictReader(SEMANTIC.open(encoding='utf-8-sig'))}
 for r in rows:
  s=sem[(r['relative_dir'],r['episode'])];reasons=[x for x in r['reasons'].split(';') if x and x != 'target_visible_true_but_bbox_invalid']
  if args.isolate_visible_bbox_conflicts and (int(s['drone_visible_true_invalid']) or int(s['robotdog_visible_true_invalid'])):reasons.append('target_visible_true_but_bbox_invalid')
  r['reasons']=';'.join(dict.fromkeys(reasons));r['isolate']=str(bool(reasons))
 bad=[r for r in rows if r['isolate']=='True']
 with REPORT.open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 counts=Counter(x for r in bad for x in r['reasons'].split(';') if x)
 summary={'created_at':datetime.now().isoformat(),'root':str(ROOT),'episodes_before':len(rows),'clean_after_isolation':len(rows)-len(bad),'isolate':len(bad),'reason_counts':dict(counts),'report':str(REPORT),'video_report':str(REPORT.parent/'data_arr_final_video_audit_20260822.csv'),'bbox_semantic_report':str(SEMANTIC),'quarantine_root':str(QUARANTINE)}
 SUMMARY.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
 ops=[]
 for r in bad:
  d=ROOT/r['relative_dir'];ep=r['episode'];files=sorted(d.glob(f'{ep}_*'));stat=d/f'{ep}.json'
  if stat.is_file():files.append(stat)
  if not files:raise FileNotFoundError(f'{d}/{ep}')
  for src in files:
   dst=QUARANTINE/r['relative_dir']/src.name
   if dst.exists():raise FileExistsError(dst)
   ops.append({'source':str(src),'destination':str(dst),'relative_dir':r['relative_dir'],'episode':ep,'reasons':r['reasons']})
 print(json.dumps(summary,ensure_ascii=False,indent=2));print('files',len(ops),'execute',args.execute)
 if not args.execute:return
 for op in ops:
  src=Path(op['source']);dst=Path(op['destination']);dst.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(src),str(dst))
 manifest={'created_at':datetime.now().isoformat(),'source_root':str(ROOT),'quarantine_root':str(QUARANTINE),'episodes':len(bad),'files':ops}
 (QUARANTINE/'rollback_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
 print('moved',len(bad),'episodes',len(ops),'files')
if __name__=='__main__':main()
