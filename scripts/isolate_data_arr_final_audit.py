#!/usr/bin/env python3
import argparse,csv,json,shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT=Path('/data/hdt/ntv_data/cyj/data_arr')
QUARANTINE=Path('/data/hdt/ntv_data/cyj/data_arr_quarantine_final_audit_20260822')
REPORT=Path(__file__).resolve().parents[1]/'reports/data_arr_final_audit_20260822.csv'
SUMMARY=Path(__file__).resolve().parents[1]/'reports/data_arr_final_audit_20260822_summary.json'

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');args=ap.parse_args()
 rows=list(csv.DictReader(REPORT.open(encoding='utf-8-sig')))
 for r in rows:
  reasons=[x for x in r['reasons'].split(';') if x]
  intentional_loss=bool(set(Path(r['relative_dir']).parts)&{'stt_camera3_loststart','stt_camera3_lostmid'})
  if not intentional_loss:
   for agent in ('drone','robotdog'):
    if int(r[f'{agent}_bbox_zero_run'])>=50: reasons.append(f'{agent}_long_bbox_gap')
  r['reasons']=';'.join(dict.fromkeys(reasons));r['isolate']=str(bool(reasons))
 bad=[r for r in rows if r['isolate']=='True']
 with REPORT.open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
 counts=Counter(x for r in bad for x in r['reasons'].split(';') if x)
 summary={'created_at':datetime.now().isoformat(),'root':str(ROOT),'episodes':len(rows),'clean':len(rows)-len(bad),'isolate':len(bad),'reason_counts':dict(sorted(counts.items())),'report':str(REPORT),'video_report':str(REPORT.parent/'data_arr_final_video_audit_20260822.csv')}
 SUMMARY.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
 operations=[]
 for r in bad:
  d=ROOT/r['relative_dir'];ep=r['episode'];files=sorted(d.glob(f'{ep}_*'));stat=d/f'{ep}.json'
  if stat.is_file():files.append(stat)
  if not files: raise FileNotFoundError(f'{d}/{ep}')
  for src in files:
   dst=QUARANTINE/r['relative_dir']/src.name
   if dst.exists():raise FileExistsError(dst)
   operations.append({'source':str(src),'destination':str(dst),'relative_dir':r['relative_dir'],'episode':ep,'reasons':r['reasons']})
 print(json.dumps(summary,ensure_ascii=False,indent=2));print(f'operations episodes={len(bad)} files={len(operations)} execute={args.execute}')
 if not args.execute:return
 for op in operations:
  src=Path(op['source']);dst=Path(op['destination']);dst.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(src),str(dst))
 manifest={'created_at':datetime.now().isoformat(),'source_root':str(ROOT),'quarantine_root':str(QUARANTINE),'episodes':len(bad),'files':operations}
 (QUARANTINE/'rollback_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
 print(f'moved episodes={len(bad)} files={len(operations)} rollback={QUARANTINE/"rollback_manifest.json"}')
if __name__=='__main__':main()
