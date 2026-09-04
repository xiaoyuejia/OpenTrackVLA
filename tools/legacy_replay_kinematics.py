"""Bridge to the previously validated inverse replay implementation.

The old replay is kept as the numerical reference.  AT rendering adds actors,
but must not fork its fixed-step, ground-control, or stable-snapshot semantics.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


OLD_ROOT = Path("/data/hdt/newtrackvla修改/newtrackvla_base").resolve()
OLD_FILE = OLD_ROOT / "tools/history/replay_hand_realtime_inverse_fixed_dt.py"
MIGRATED_FILE = Path(__file__).with_name("history") / "replay_hand_realtime_inverse_fixed_dt.py"
if not MIGRATED_FILE.is_file():
    raise FileNotFoundError(f"migrated legacy replay is missing: {MIGRATED_FILE}")

_spec = importlib.util.spec_from_file_location("migrated_legacy_inverse_replay", MIGRATED_FILE)
if _spec is None or _spec.loader is None:
    raise ImportError(f"could not load migrated replay module: {MIGRATED_FILE}")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

ground_control = _module.ground_control
capture_color_mask_snapshot_stable = _module.capture_color_mask_snapshot_stable
fixed_step = _module.fixed_step
body_xy_velocity = _module.body_xy_velocity
pose_error_m = _module.pose_error_m
yaw_error_deg = _module.yaw_error_deg
drone_control = _module.drone_control

LEGACY_REPLAY_CONFIG = {
    "source": str(OLD_FILE),
    "ground_speed_model": "legacy_preview",
    "ground_control_mode": "source_yaw",
    "ground_translation_delay_steps": 1,
    "ground_position_feedback_time_s": 1.0,
    "ground_max_forward_feedback_mps": float("inf"),
    "ground_turn_step_gain": 0.4,
    "ground_max_turn_deg": 30.0,
    "ground_acceleration": 10000.0,
    "fixed_dt": 0.1,
    "snapshot_mode": "sequential",
    "snapshot_render_sync_s": 0.08,
}
