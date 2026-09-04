#!/usr/bin/env python3
"""One real STT+DT+AT batch through top-8 matching and both existing LLM flows."""
from pathlib import Path
import argparse
import time
import torch

from train_airground_coop_v3 import (
    AirGroundV3DataConfig, AirGroundV3JsonDataset, build_airground_v3_model,
    collate_airground_v3_batch, forward_airground_v3_loss, load_config,
)


def dataset(path: Path, candidate_root: Path) -> AirGroundV3JsonDataset:
    root=Path('/data/yh/data/processed/joint')
    return AirGroundV3JsonDataset(AirGroundV3DataConfig(
        train_json=str(path),n_waypoints=8,history=31,cache_root=str(root/'vision_cache'),
        perception_cache_root=str(root/'perception_cache'),candidate_cache_root=str(candidate_root),
        require_topk_candidate_cache=True,candidate_top_k=8,coarse_cache_size=0,
    ))


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument('--batch-size',type=int,default=3); args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1]
    root=Path('/data/yh/data/processed/joint/train_jsonl')
    shared=Path('UnrealTrack-ModularSciFiVillage-ContinuousColor-v0/dt_camera3__9__seed_100/2.jsonl')
    candidate_root=Path('/data/yh/newtrackvla修改/newtrackvla_base_yh_clean/candidate_cache_smoke_640c001')
    paths=[next((root/'stt').rglob('*.jsonl')),root/'dt'/shared,root/'at'/shared]
    sets=[dataset(path,candidate_root) for path in paths]
    items=[]
    for index in range(args.batch_size):
        source=index%3; row=50+(index//3)%100; items.append(sets[source][min(row,len(sets[source])-1)])
    dt_item=sets[1][50]; at_item=sets[2][50]
    assert dt_item['current_path'][0] == at_item['current_path'][0]
    assert dt_item['instruction'] != at_item['instruction']
    batch=collate_airground_v3_batch(items)
    cfg=load_config(repo/'config/airground_cooperative_tracking_v3_yh.yaml')
    cfg.batch_size=args.batch_size; cfg.num_workers=0
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model=build_airground_v3_model(cfg)
    missing=[]
    if cfg.init_ckpt:
        checkpoint=torch.load(cfg.init_ckpt,map_location='cpu',weights_only=False)
        missing,_=model.load_state_dict(checkpoint['model_state'],strict=False)
    model=model.to(device).train()
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device)
    started=time.time()
    with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=device.type=='cuda'):
        loss,metrics=forward_airground_v3_loss(model,batch,cfg,device)
    loss.backward()
    if device.type=='cuda': torch.cuda.synchronize(device)
    print({
        'loss':float(loss.detach()),
        'candidate_shape':tuple(batch['candidate_feat'].shape),
        'valid_candidates':batch['candidate_valid'].sum(-1).tolist(),
        'top8_target_recall':float(metrics['candidate_top8_recall']),
        'seconds':round(time.time()-started,3),
        'peak_memory_gb':round(torch.cuda.max_memory_allocated(device)/(1024**3),2) if device.type=='cuda' else 0,
        'init_missing_keys':missing,
        'dt_instruction':dt_item['instruction'],
        'at_instruction':at_item['instruction'],
    })
    return 0


if __name__=='__main__': raise SystemExit(main())
