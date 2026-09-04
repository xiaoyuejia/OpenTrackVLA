"""Offline RGB detection and segmentation cache generation."""

from .core import (
    LABEL_FREE,
    LABEL_OBSTACLE,
    LABEL_TARGET,
    LABEL_UNKNOWN,
    SCHEMA_VERSION,
    FusedPrediction,
    InstancePrediction,
    fuse_instances,
    mask_to_grid,
)

__all__ = [
    "LABEL_UNKNOWN",
    "LABEL_FREE",
    "LABEL_OBSTACLE",
    "LABEL_TARGET",
    "SCHEMA_VERSION",
    "InstancePrediction",
    "FusedPrediction",
    "fuse_instances",
    "mask_to_grid",
]
