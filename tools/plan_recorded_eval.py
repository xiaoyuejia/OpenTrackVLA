#!/usr/bin/env python3
"""Build disjoint, scene-aware worker plans for a recorded eval manifest."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any


def episode_name(item: dict[str, Any]) -> str:
    if item.get("key"):
        return str(item["key"])
    if item.get("info"):
        info = Path(str(item["info"]))
        relative_dir = str(item.get("relative_dir", info.parent))
        stem = str(item.get("stem", info.name.removesuffix("_info.json")))
        return f"{relative_dir}/{stem}"
    raise ValueError(f"manifest item has neither key nor info: {item}")


def completed_episode_names(eval_root: Path) -> set[str]:
    completed: set[str] = set()
    if not eval_root.exists():
        return completed
    for setup_path in eval_root.rglob("*_setup.json"):
        stat_path = setup_path.with_name(
            setup_path.name.removesuffix("_setup.json") + ".json"
        )
        if not stat_path.is_file():
            continue
        try:
            setup = json.loads(setup_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        name = setup.get("recorded_target_episode")
        if name:
            completed.add(str(name))
    return completed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--eval-root", required=True, type=Path)
    parser.add_argument("--plan-dir", required=True, type=Path)
    parser.add_argument("--workers", required=True, type=int)
    parser.add_argument("--expected-episodes", type=int, default=None)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Plan every manifest entry even when a completed result already exists.",
    )
    args = parser.parse_args()
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")

    manifest_path = args.manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("test")
    if not isinstance(entries, list) or not entries:
        raise SystemExit("manifest must contain a non-empty 'test' list")
    if args.expected_episodes is not None and len(entries) != args.expected_episodes:
        raise SystemExit(
            f"manifest contains {len(entries)} episodes, expected {args.expected_episodes}"
        )

    items: list[tuple[str, str]] = []
    seen: set[str] = set()
    for raw_item in entries:
        if not isinstance(raw_item, dict):
            raise SystemExit(f"manifest entry is not an object: {raw_item!r}")
        scene = str(raw_item.get("scene", ""))
        name = episode_name(raw_item)
        if not scene:
            raise SystemExit(f"manifest item lacks scene: {name}")
        if name in seen:
            raise SystemExit(f"duplicate recorded target episode: {name}")
        seen.add(name)
        items.append((scene, name))

    completed = set() if args.no_resume else completed_episode_names(args.eval_root)
    completed_in_manifest = {name for _, name in items} & completed
    pending = [(scene, name) for scene, name in items if name not in completed]

    by_scene: OrderedDict[str, list[str]] = OrderedDict()
    for scene, name in pending:
        by_scene.setdefault(scene, []).append(name)

    assignments: list[list[tuple[str, list[str]]]] = [
        [] for _ in range(args.workers)
    ]
    loads = [0] * args.workers
    # Never split one scene across workers. This prevents duplicate output names
    # and avoids launching the same heavyweight UE map more than once.
    for scene, names in sorted(
        by_scene.items(), key=lambda pair: (-len(pair[1]), pair[0])
    ):
        slot = min(range(args.workers), key=lambda index: (loads[index], index))
        assignments[slot].append((scene, names))
        loads[slot] += len(names)

    plan_dir = args.plan_dir.expanduser().resolve()
    plan_dir.mkdir(parents=True, exist_ok=True)
    for slot, groups in enumerate(assignments):
        with (plan_dir / f"worker_{slot}.tsv").open(
            "w", encoding="utf-8"
        ) as handle:
            for scene, names in groups:
                handle.write(f"{scene}\t{len(names)}\t{','.join(names)}\n")

    summary = {
        "manifest": str(manifest_path),
        "total_episodes": len(items),
        "already_complete": len(completed_in_manifest),
        "pending_episodes": len(pending),
        "workers": args.workers,
        "worker_episode_loads": loads,
        "assignments": [
            [
                {"scene": scene, "episodes": names}
                for scene, names in groups
            ]
            for groups in assignments
        ],
    }
    summary_path = plan_dir / "plan.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "[plan] total=%d complete=%d pending=%d loads=%s"
        % (
            summary["total_episodes"],
            summary["already_complete"],
            summary["pending_episodes"],
            loads,
        )
    )
    print(f"[plan] summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
