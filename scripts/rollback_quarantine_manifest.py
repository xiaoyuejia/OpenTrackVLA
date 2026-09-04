#!/usr/bin/env python3
"""Restore files recorded by a quarantine rollback manifest."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("files", [])
    if not isinstance(records, list) or not records:
        raise RuntimeError("manifest has no file records")

    pending = []
    already_restored = []
    for record in records:
        source = Path(record["source"])
        quarantined = Path(record["destination"])
        if source.exists() and quarantined.exists():
            raise FileExistsError(f"both restore target and quarantined file exist: {source}")
        if quarantined.exists():
            pending.append((quarantined, source))
        elif source.exists():
            already_restored.append(source)
        else:
            raise FileNotFoundError(f"neither path exists: {source} / {quarantined}")

    print(
        f"episodes={manifest.get('episodes')} files={len(records)} "
        f"pending={len(pending)} already_restored={len(already_restored)}"
    )
    if not args.execute:
        return

    restored = []
    for quarantined, source in pending:
        source.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(quarantined), str(source))
        restored.append({"from": str(quarantined), "to": str(source)})

    completion = {
        "completed_at": datetime.now().isoformat(),
        "manifest": str(manifest_path),
        "episodes": manifest.get("episodes"),
        "restored_files": len(restored),
        "operations": restored,
    }
    completion_path = manifest_path.with_name("rollback_completed.json")
    completion_path.write_text(
        json.dumps(completion, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"restored_files={len(restored)} completion={completion_path}")


if __name__ == "__main__":
    main()
