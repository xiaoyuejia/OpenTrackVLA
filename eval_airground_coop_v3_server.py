#!/usr/bin/env python3
"""One-GPU shared inference server for V3 evaluation workers."""

from __future__ import annotations

import argparse
import os
import threading
import traceback
from collections import deque
from multiprocessing.connection import Listener
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from eval_airground_coop_v3 import AirGroundCoopV3Planner


SESSION_FIELDS = (
    "histories",
    "inverse_velocity_state",
    "inverse_velocity_reference",
    "last_inverse_command",
    "last_waypoints",
    "last_policy_debug",
    "bbox_motion_controller",
    "visibility_router",
    "last_navigation_waypoints",
    "search_center_yaw_rad",
    "search_target_direction",
)


class SharedAirGroundV3Server:
    """Serialize one V3 GPU model while isolating each UE session state."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.planner: AirGroundCoopV3Planner | None = None
        self.states: dict[str, dict[str, Any]] = {}

    def _metadata(self) -> dict[str, Any]:
        assert self.planner is not None
        names = (
            "history", "history_frame_dt", "n_waypoints", "use_roi_tokens",
            "use_bbox_text_prompt", "use_visual_section_markers", "roi_bbox_source",
            "roi_expand_ratio", "roi_token_count", "roi_make_square",
            "evaluation_protocol", "ckpt_bbox_dropout_prob",
            "supports_intermediate_observation",
        )
        return {name: getattr(self.planner, name) for name in names} | {
            "ckpt_path": str(self.planner.ckpt_path)
        }

    def _blank_state(self) -> dict[str, Any]:
        assert self.planner is not None
        return {
            "histories": [
                deque(maxlen=max(self.planner.history * 4, self.planner.history + 1)),
                deque(maxlen=max(self.planner.history * 4, self.planner.history + 1)),
            ],
            "inverse_velocity_state": None,
            "inverse_velocity_reference": None,
            "last_inverse_command": None,
            "last_waypoints": None,
            "last_policy_debug": {},
            "bbox_motion_controller": self.planner.new_bbox_motion_controller(),
            "visibility_router": self.planner.new_visibility_router(),
            "last_navigation_waypoints": None,
            "search_center_yaw_rad": None,
            "search_target_direction": np.ones(2, dtype=np.float32),
        }

    def _restore(self, session: str) -> None:
        assert self.planner is not None
        state = self.states.setdefault(session, self._blank_state())
        for name, value in state.items():
            setattr(self.planner, name, value)

    def _save(self, session: str) -> None:
        assert self.planner is not None
        self.states[session] = {
            name: getattr(self.planner, name) for name in SESSION_FIELDS
        }

    def request(self, message: dict[str, Any]) -> Any:
        operation = str(message.get("op", ""))
        session = str(message.get("session", ""))
        if not session:
            raise ValueError("missing client session")
        with self.lock:
            if operation == "init":
                supplied = dict(message.get("args") or {})
                if self.planner is None:
                    print(
                        "[airground-v3-server] loading shared "
                        "model/encoders/YOLO once",
                        flush=True,
                    )
                    self.planner = AirGroundCoopV3Planner(
                        SimpleNamespace(**supplied)
                    )
                else:
                    if str(supplied.get("ckpt")) != str(self.planner.args.ckpt):
                        raise ValueError(
                            "all workers on one server must use the same checkpoint"
                        )
                    supplied_y_mode = str(
                        supplied.get(
                            "robotdog_waypoint_y_mode",
                            "v3_nonholonomic_projection",
                        )
                    )
                    planner_y_mode = str(
                        getattr(
                            self.planner.args,
                            "robotdog_waypoint_y_mode",
                            "v3_nonholonomic_projection",
                        )
                    )
                    if supplied_y_mode != planner_y_mode:
                        raise ValueError(
                            "all workers on one server must use the same "
                            "RobotDog waypoint-y control mode"
                        )
                self.states.setdefault(session, self._blank_state())
                return self._metadata()

            if self.planner is None:
                raise RuntimeError("first request must be init")
            self._restore(session)
            if operation == "reset":
                self.planner.reset()
                result: Any = None
            elif operation == "observe":
                result = self.planner.observe(*message["args"], **message["kwargs"])
            elif operation == "predict":
                result = self.planner.predict(*message["args"], **message["kwargs"])
            elif operation == "actions":
                result = self.planner.waypoints_to_actions(
                    *message["args"], **message["kwargs"]
                )
            else:
                raise ValueError(f"unsupported operation {operation!r}")
            self._save(session)
            return result


def serve_connection(connection: Any, server: SharedAirGroundV3Server) -> None:
    try:
        while True:
            try:
                message = connection.recv()
            except EOFError:
                return
            try:
                connection.send({"ok": True, "value": server.request(message)})
            except Exception as exc:
                connection.send(
                    {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                )
                traceback.print_exc()
    finally:
        connection.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True, type=Path)
    parser.add_argument("--authkey", default="eval_airground_v3")
    args = parser.parse_args()
    socket_path = args.socket.resolve()
    if socket_path.parent != Path("/tmp"):
        raise ValueError("shared inference socket must live directly under /tmp")
    if socket_path.exists():
        socket_path.unlink()
    listener = Listener(
        str(socket_path), family="AF_UNIX", authkey=args.authkey.encode("utf-8")
    )
    print(
        f"[airground-v3-server] ready socket={socket_path} pid={os.getpid()}",
        flush=True,
    )
    server = SharedAirGroundV3Server()
    try:
        while True:
            connection = listener.accept()
            threading.Thread(
                target=serve_connection,
                args=(connection, server),
                daemon=True,
            ).start()
    finally:
        listener.close()
        if socket_path.exists():
            socket_path.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
