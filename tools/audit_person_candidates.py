#!/usr/bin/env python3
"""Measure detector top-1/top-K target recall against recorded GT boxes."""
from __future__ import annotations
import argparse
from pathlib import Path

from train_airground_coop_v3 import AirGroundV3DataConfig, AirGroundV3JsonDataset


def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument('--jsonl',type=Path,required=True); p.add_argument('--candidate-cache-root',type=Path,required=True); p.add_argument('--limit',type=int,default=0); a=p.parse_args()
    base=Path('/data/yh/data/processed')
    cfg=AirGroundV3DataConfig(train_json=str(a.jsonl),n_waypoints=8,history=31,cache_root=str(base/'vision_cache'),perception_cache_root=str(base/'perception_cache'),candidate_cache_root=str(a.candidate_cache_root),require_topk_candidate_cache=True,candidate_top_k=8,coarse_cache_size=0)
    ds=AirGroundV3JsonDataset(cfg); size=min(len(ds),a.limit) if a.limit else len(ds)
    counts=[]; recall=[]; top1=[]; eligible=[]
    for index in range(size):
        item=ds[index]; counts.extend(item['candidate_valid'].sum(-1).tolist()); recall.extend(item['candidate_match_label'].any(-1).tolist()); top1.extend(item['candidate_match_label'][...,0].tolist()); eligible.extend((item['visible'].bool() & item['bbox_valid_mask'].bool()).tolist())
    eligible_count=sum(eligible)
    print(f'rows={size} views={len(counts)} eligible_visible={eligible_count} mean_candidates={sum(counts)/max(1,len(counts)):.3f} multi_fraction={sum(v>1 for v in counts)/max(1,len(counts)):.4f} top1_recall={sum(v for v,e in zip(top1,eligible) if e)/max(1,eligible_count):.4f} top8_recall={sum(v for v,e in zip(recall,eligible) if e)/max(1,eligible_count):.4f}')
    return 0
if __name__=='__main__': raise SystemExit(main())
