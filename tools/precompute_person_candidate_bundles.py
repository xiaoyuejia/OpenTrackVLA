#!/usr/bin/env python3
"""Precompute top-8 person boxes into one compact file per episode/view."""
from __future__ import annotations
import argparse, os, time, logging
from pathlib import Path
import numpy as np
from PIL import Image
from offline_detection_segmentation.models import YOLOInstanceSegmenter

SCHEMA="person_candidates.bundle.v1"

def views(root: Path):
    for directory,names,files in os.walk(root):
        images=[name for name in sorted(files) if Path(name).suffix.lower() in {'.jpg','.jpeg','.png'}]
        if images: yield Path(directory),images

def target_path(output_root: Path,data_root: Path,view: Path) -> Path:
    relative=view.resolve().relative_to(data_root.resolve())
    return output_root/relative.parent/f'{relative.name}.candidates.npz'

def duration(value):
    h,r=divmod(max(0,int(value)),3600); m,s=divmod(r,60); return f'{h:02d}:{m:02d}:{s:02d}'

def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument('--frame-root',type=Path,default=Path('/data/yh/data/processed/frames/dt')); p.add_argument('--frame-list',type=Path); p.add_argument('--data-root',type=Path,default=Path('/data/yh/data/processed')); p.add_argument('--output-root',type=Path,default=Path('/data/yh/data/processed/perception_cache')); p.add_argument('--weights',default='/data/yh/newtrackvla修改/newtrackvla_base_yh_clean/repo/offline_detection_segmentation/weights/yolo11m-seg.pt'); p.add_argument('--device',default='cuda:0'); p.add_argument('--batch-size',type=int,default=48); p.add_argument('--image-size',type=int,default=640); p.add_argument('--confidence',type=float,default=0.01); p.add_argument('--top-k',type=int,default=8); p.add_argument('--num-shards',type=int,default=1); p.add_argument('--shard-index',type=int,default=0); p.add_argument('--limit-views',type=int,default=0); a=p.parse_args()
    if a.frame_list:
        grouped={}
        for line in a.frame_list.read_text(encoding='utf-8').splitlines():
            path=Path(line.strip())
            if path.name: grouped.setdefault(path.parent,[]).append(path.name)
        source_views=[(view,sorted(set(names))) for view,names in sorted(grouped.items(),key=lambda item:str(item[0]))]
    else:
        source_views=list(views(a.frame_root))
    all_views=[item for index,item in enumerate(source_views) if index%a.num_shards==a.shard_index]
    if a.limit_views: all_views=all_views[:a.limit_views]
    logging.getLogger('ultralytics').setLevel(logging.ERROR)
    detector=YOLOInstanceSegmenter(a.weights,device=a.device,image_size=a.image_size,confidence=a.confidence,half=True)
    started=time.time(); written=skipped=frames=0
    for view_index,(view,names) in enumerate(all_views,1):
        target=target_path(a.output_root,a.data_root,view)
        if target.is_file(): skipped+=1; continue
        boxes=np.zeros((len(names),a.top_k,4),np.float16); scores=np.zeros((len(names),a.top_k),np.float16); valid=np.zeros((len(names),a.top_k),np.bool_)
        for start in range(0,len(names),a.batch_size):
            batch_names=names[start:start+a.batch_size]; images=[Image.open(view/name).convert('RGB') for name in batch_names]
            try: predictions=detector.predict_person_candidates(images,top_k=a.top_k)
            finally:
                for image in images: image.close()
            for offset,(candidate_boxes,candidate_scores) in enumerate(predictions):
                row=start+offset; count=min(len(candidate_scores),a.top_k)
                if count: boxes[row,:count]=candidate_boxes[:count]; scores[row,:count]=candidate_scores[:count]; valid[row,:count]=True
            frames+=len(batch_names)
        target.parent.mkdir(parents=True,exist_ok=True); temporary=target.with_name(f'.{target.name}.tmp-{os.getpid()}')
        with temporary.open('wb') as handle: np.savez_compressed(handle,schema_version=np.asarray(SCHEMA),frame_stems=np.asarray([Path(name).stem for name in names]),boxes_cxcywh_norm=boxes,scores=scores,valid=valid,top_k=np.asarray(a.top_k,np.int16))
        os.replace(temporary,target); written+=1
        if view_index%10==0 or view_index==len(all_views):
            elapsed=time.time()-started; eta=elapsed/max(1,view_index)*max(0,len(all_views)-view_index)
            print(f'[candidate-bundle] shard={a.shard_index}/{a.num_shards} views={view_index}/{len(all_views)} written={written} skipped={skipped} frames={frames} elapsed={duration(elapsed)} eta={duration(eta)}',flush=True)
    print(f'[done] shard={a.shard_index}/{a.num_shards} views={len(all_views)} written={written} skipped={skipped} frames={frames}',flush=True); return 0
if __name__=='__main__': raise SystemExit(main())
