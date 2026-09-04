"""文本条件化的 Top-K 行人候选匹配。

匹配器与检测器、规划器相互独立。它接收紧凑 ROI embedding 和文本 embedding，
为每个候选分别输出一个分数，并应用显式的有效掩码和置信度阈值。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class CandidateSelection:
    index: torch.Tensor
    probability: torch.Tensor
    accepted: torch.Tensor
    margin: torch.Tensor


class CandidateTextMatcher(nn.Module):
    """计算 ``(B,A,K,D)`` 候选特征与 ``(B,A,D)`` 文本特征的匹配分数。"""

    def __init__(self, dim: int, hidden_dim: int = 512) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(0.07).log())
        # cosine/temperature 对比项的数值天然较大，适合相对 softmax 排序，
        # 但从头训练独立 sigmoid 时会让概率立即饱和。残差增益从零开始，
        # 仅在 balanced BCE 认为有用时才逐步启用。
        self.cosine_gain = nn.Parameter(torch.zeros(()))
        self.fusion = nn.Sequential(
            nn.LayerNorm(dim * 4),
            nn.Linear(dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        candidate_features: torch.Tensor,
        text_features: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if candidate_features.ndim != 4:
            raise ValueError("candidate_features must have shape (B,A,K,D)")
        if text_features.shape != candidate_features.shape[:2] + (candidate_features.size(-1),):
            raise ValueError("text_features must have shape (B,A,D)")
        if valid_mask.shape != candidate_features.shape[:3]:
            raise ValueError("valid_mask must have shape (B,A,K)")
        c = F.normalize(candidate_features.float(), dim=-1)
        t = F.normalize(text_features.float(), dim=-1).unsqueeze(2).expand_as(c)
        fused = torch.cat((c, t, c * t, c - t), dim=-1)
        logits = self.fusion(fused).squeeze(-1)
        logits = logits + (
            self.cosine_gain
            * (c * t).sum(dim=-1)
            / self.temperature.exp().clamp_min(1.0e-3)
        )
        return logits.masked_fill(~valid_mask, torch.finfo(logits.dtype).min)


def select_top_candidate(
    logits: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    enter_threshold: float = 0.70,
    margin_threshold: float = 0.15,
    previous_index: Optional[torch.Tensor] = None,
) -> CandidateSelection:
    """按校准后的置信度选择 Top-1；margin 只返回用于诊断，不参与接受判定。"""
    if logits.shape != valid_mask.shape or logits.ndim != 3:
        raise ValueError("logits and valid_mask must both have shape (B,A,K)")
    probs = torch.sigmoid(logits)
    masked = logits.masked_fill(~valid_mask, torch.finfo(logits.dtype).min)
    top_values, top_indices = masked.topk(min(2, masked.size(-1)), dim=-1)
    index = top_indices[..., 0]
    top_prob = probs.gather(-1, index.unsqueeze(-1)).squeeze(-1)
    if top_values.size(-1) > 1:
        second_prob = probs.gather(-1, top_indices[..., 1:2]).squeeze(-1)
    else:
        second_prob = torch.zeros_like(top_prob)
    margin = top_prob - second_prob
    accepted = valid_mask.gather(-1, index.unsqueeze(-1)).squeeze(-1)
    accepted &= top_prob >= float(enter_threshold)
    # 为兼容旧调用保留 margin_threshold 参数，但 margin 本轮只记录、不设门限。
    _ = margin_threshold
    if previous_index is not None:
        if previous_index.shape != index.shape:
            raise ValueError("previous_index must have shape (B,A)")
        keep = previous_index.ge(0) & valid_mask.gather(-1, previous_index.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        index = torch.where(~accepted & keep, previous_index, index)
    return CandidateSelection(index=index, probability=top_prob, accepted=accepted, margin=margin)
