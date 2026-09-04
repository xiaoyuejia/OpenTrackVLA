#!/usr/bin/env python3
"""Video metadata audit for the final data_arr JSON audit report."""
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import cv2
import numpy as np

ROOT=Path('/data/hdt/ntv_data/cyj/data_arr')
INPUT=Path(__file__).resolve().parents[1]/'reports/data_arr_final_audit_20260822.csv'
OUTPUT=Path(__file__).resolve().parents[1]/'reports/data_arr_final_video_audit_20260822.csv'

def one(row):
    cv2.setNumThreads(0)
    candidate=max(int(row['drone_bbox_zero_run']),int(row['robotdog_bbox_zero_run']))>=50 or max(int(row['drone_static_run']),int(row['robotdog_static_run']))>=100 or row['isolate']=='True'
    out={'relative_dir':row['relative_dir'],'episode':row['episode'],'visual_sampled':candidate}
    for agent in ('drone','robotdog'):
        path=ROOT/row['relative_dir']/f"{row['episode']}_{agent}.mp4"
        cap=cv2.VideoCapture(str(path)); opened=cap.isOpened(); count=int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT))) if opened else 0
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if opened else 0; height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if opened else 0
        decoded=0; blank=0; frozen=[]; prev=None
        if opened and candidate:
            for idx in sorted(set(range(0,max(count,1),10))|({count-1} if count else set())):
                cap.set(cv2.CAP_PROP_POS_FRAMES,idx); ok,frame=cap.read()
                if not ok: continue
                gray=cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(160,120)); decoded+=1
                blank += int(float(gray.std())<2 or float(np.mean(gray<3))>.995 or float(np.mean(gray>252))>.995)
                if prev is not None: frozen.append(float(np.mean(cv2.absdiff(gray,prev)))<.15)
                prev=gray
        cap.release()
        best=cur=0
        for flag in frozen:
            cur=cur+1 if flag else 0; best=max(best,cur)
        out.update({f'{agent}_video_open':opened,f'{agent}_video_frames':count,f'{agent}_video_width':width,f'{agent}_video_height':height,f'{agent}_decoded_samples':decoded,f'{agent}_blank_rate':blank/decoded if decoded else 0,f'{agent}_freeze_run_est':best*10})
    out['frame_count_match']=(out['drone_video_frames']==int(row['drone_rows']) and out['robotdog_video_frames']==int(row['robotdog_rows']))
    out['video_issue']=';'.join(x for x in [
      'drone_video_open_or_count' if not out['drone_video_open'] or out['drone_video_frames']!=int(row['drone_rows']) else '',
      'robotdog_video_open_or_count' if not out['robotdog_video_open'] or out['robotdog_video_frames']!=int(row['robotdog_rows']) else '',
      'drone_video_blank' if out['drone_blank_rate']>=.5 else '', 'robotdog_video_blank' if out['robotdog_blank_rate']>=.5 else '',
      'drone_video_frozen' if out['drone_freeze_run_est']>=150 else '', 'robotdog_video_frozen' if out['robotdog_freeze_run_est']>=150 else ''] if x)
    return out

def main():
    rows=list(csv.DictReader(INPUT.open(encoding='utf-8-sig')))
    print('episodes',len(rows),flush=True)
    with ProcessPoolExecutor(max_workers=8) as pool: result=list(pool.map(one,rows,chunksize=8))
    with OUTPUT.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=result[0].keys()); w.writeheader(); w.writerows(result)
    bad=[r for r in result if r['video_issue']]
    print('video_issues',len(bad),flush=True)
    for r in bad[:100]: print(r['relative_dir'],r['episode'],r['video_issue'],flush=True)

if __name__=='__main__': main()
