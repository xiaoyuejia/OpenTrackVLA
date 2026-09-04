from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence


BBox = Sequence[float]


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def normalize_bbox_xywh_to_cxcywh(raw_bbox: Any, width: int, height: int) -> List[float]:
    """Convert UnrealZoo xywh bbox to normalized cxcywh.

    If values already look normalized, they are interpreted as xywh_norm.
    Invalid or missing boxes become all zeros.
    """
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) < 4:
        return [0.0, 0.0, 0.0, 0.0]
    try:
        x, y, w, h = [float(v) for v in raw_bbox[:4]]
    except Exception:
        return [0.0, 0.0, 0.0, 0.0]
    vals = [x, y, w, h]
    if not all(math.isfinite(v) for v in vals):
        return [0.0, 0.0, 0.0, 0.0]

    if max(abs(v) for v in vals) <= 1.5:
        return [clamp01(x + 0.5 * w), clamp01(y + 0.5 * h), clamp01(w), clamp01(h)]

    width = max(1, int(width))
    height = max(1, int(height))
    return [
        clamp01((x + 0.5 * w) / width),
        clamp01((y + 0.5 * h) / height),
        clamp01(w / width),
        clamp01(h / height),
    ]


def _coerce_cxcywh_norm(bbox: Any) -> List[float]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return [0.0, 0.0, 0.0, 0.0]
    try:
        vals = [float(v) for v in bbox[:4]]
    except Exception:
        return [0.0, 0.0, 0.0, 0.0]
    if not all(math.isfinite(v) for v in vals):
        return [0.0, 0.0, 0.0, 0.0]
    return [clamp01(v) for v in vals]


def bbox_is_valid_cxcywh(bbox: Any, min_size: float = 1e-4, min_area: float = 1e-6) -> bool:
    cx, cy, w, h = _coerce_cxcywh_norm(bbox)
    if w <= min_size or h <= min_size:
        return False
    if w * h <= min_area:
        return False
    return 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0


def _bin01(x: float, bins: int = 10) -> int:
    return max(0, min(bins - 1, int(clamp01(x) * bins)))


def _horizontal_label(cx: float) -> str:
    if cx < 0.15:
        return "far_left"
    if cx < 0.35:
        return "left"
    if cx < 0.45:
        return "slightly_left"
    if cx <= 0.55:
        return "center"
    if cx <= 0.65:
        return "slightly_right"
    if cx <= 0.85:
        return "right"
    return "far_right"


def _vertical_label(cy: float) -> str:
    if cy < 0.20:
        return "top"
    if cy < 0.40:
        return "upper"
    if cy <= 0.60:
        return "center"
    if cy <= 0.80:
        return "lower"
    return "bottom"


def _size_label(area: float) -> str:
    if area < 0.004:
        return "tiny"
    if area < 0.015:
        return "small"
    if area < 0.050:
        return "medium"
    return "large"


def _distance_hint(area: float) -> str:
    if area < 0.004:
        return "very_far"
    if area < 0.015:
        return "far"
    if area < 0.050:
        return "medium"
    return "near"


def _trend_fields(
    bbox: List[float],
    prev_bbox: Optional[Any],
    center_eps: float = 0.025,
    area_ratio_eps: float = 0.08,
) -> Dict[str, str]:
    if prev_bbox is None or not bbox_is_valid_cxcywh(prev_bbox):
        return {"center_x": "unknown", "center_y": "unknown", "size": "unknown"}
    pcx, pcy, pw, ph = _coerce_cxcywh_norm(prev_bbox)
    cx, cy, w, h = bbox
    dx = cx - pcx
    dy = cy - pcy
    prev_area = max(pw * ph, 1e-6)
    area_ratio = (w * h) / prev_area

    if dx > center_eps:
        center_x = "right"
    elif dx < -center_eps:
        center_x = "left"
    else:
        center_x = "stable"

    if dy > center_eps:
        center_y = "down"
    elif dy < -center_eps:
        center_y = "up"
    else:
        center_y = "stable"

    if area_ratio > 1.0 + area_ratio_eps:
        size = "approaching"
    elif area_ratio < 1.0 - area_ratio_eps:
        size = "receding"
    else:
        size = "stable"
    return {"center_x": center_x, "center_y": center_y, "size": size}


def bbox_spatial_fields(bbox: Any, prev_bbox: Optional[Any] = None, digits: int = 4) -> Dict[str, Any]:
    """Build compact spatial fields from normalized cxcywh bbox."""
    cx, cy, w, h = _coerce_cxcywh_norm(bbox)
    valid = bbox_is_valid_cxcywh([cx, cy, w, h])
    rounded_bbox = [round(v, digits) for v in (cx, cy, w, h)]
    base: Dict[str, Any] = {
        "valid": bool(valid),
        "format": "cxcywh_norm",
        "bbox": rounded_bbox,
        "bin": [_bin01(cx), _bin01(cy), _bin01(w), _bin01(h)],
    }
    if not valid:
        base.update(
            {
                "horizontal": "unknown",
                "vertical": "unknown",
                "size": "unknown",
                "distance_hint": "unknown",
                "area": 0.0,
                "center_offset": [0.0, 0.0],
                "trend": {"center_x": "unknown", "center_y": "unknown", "size": "unknown"},
            }
        )
        return base

    area = w * h
    base.update(
        {
            "horizontal": _horizontal_label(cx),
            "vertical": _vertical_label(cy),
            "size": _size_label(area),
            "distance_hint": _distance_hint(area),
            "area": round(area, digits),
            "center_offset": [round(cx - 0.5, digits), round(cy - 0.5, digits)],
            "trend": _trend_fields([cx, cy, w, h], prev_bbox),
        }
    )
    return base


def _agent_display_name(name: Any) -> str:
    text = str(name or "").strip().lower()
    if text in {"drone", "agent1", "agent0", "aerial_drone"}:
        return "Drone"
    if text in {"robotdog", "robot_dog", "dog", "agent2", "agent1_dog"}:
        return "Robotdog"
    return str(name or "Agent").strip() or "Agent"


def bbox_prompt_from_spatial(
    spatials: Iterable[Dict[str, Any]],
    agent_names: Optional[Sequence[Any]] = None,
    include_hint: bool = True,
) -> str:
    """Render bbox spatial fields into a short dynamic prompt fragment."""
    spatial_list = list(spatials)
    if agent_names is None:
        names = [f"Agent{i + 1}" for i in range(len(spatial_list))]
    else:
        names = list(agent_names)
    parts: List[str] = []
    if include_hint:
        parts.append("BBox prior: norm cxcywh bins 0-9, center bin 5, larger box means closer.")
    for i, spatial in enumerate(spatial_list):
        name = _agent_display_name(names[i] if i < len(names) else f"Agent{i + 1}")
        if not isinstance(spatial, dict) or not spatial.get("valid", False):
            parts.append(f"{name}: valid=0.")
            continue
        bins = spatial.get("bin", [0, 0, 0, 0])
        if not isinstance(bins, (list, tuple)) or len(bins) < 4:
            bins = [0, 0, 0, 0]
        bin_text = ",".join(str(int(v)) for v in bins[:4])
        trend = spatial.get("trend", {})
        if not isinstance(trend, dict):
            trend = {}
        move_x = str(trend.get("center_x", "unknown"))
        scale = str(trend.get("size", "unknown"))
        parts.append(
            f"{name}: valid=1 bin=[{bin_text}] "
            f"h={spatial.get('horizontal', 'unknown')} "
            f"v={spatial.get('vertical', 'unknown')} "
            f"size={spatial.get('size', 'unknown')} "
            f"dist={spatial.get('distance_hint', 'unknown')} "
            f"move={move_x} scale={scale}."
        )
    parts.append("Treat bbox as noisy.")
    return " ".join(parts)
