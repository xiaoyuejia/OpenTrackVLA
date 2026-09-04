#!/usr/bin/env python3
import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path

SOURCE = Path("/data/hdt/ntv_data/cyj/data_arr")
QUARANTINE = Path("/data/hdt/ntv_data/cyj/data_arr_quarantine_camera_blocked_or_stuck_20260821")
MANIFEST = Path(__file__).resolve().parents[1] / "reports/data_arr_camera_blocked_or_stuck_isolation_manifest.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8-sig")))
    operations = []
    for row in rows:
        relative = Path(row["relative_episode_stem"])
        source_stem = SOURCE / relative
        source_dir, episode = source_stem.parent, source_stem.name
        files = sorted(source_dir.glob(f"{episode}_*"))
        stat = source_dir / f"{episode}.json"
        if stat.is_file():
            files.append(stat)
        if not files:
            raise FileNotFoundError(f"no source files for {source_stem}")
        destination_dir = QUARANTINE / relative.parent
        for source in files:
            destination = destination_dir / source.name
            if destination.exists():
                raise FileExistsError(destination)
            operations.append({"source": str(source), "destination": str(destination), **row})
    print(f"episodes={len(rows)} files={len(operations)} quarantine={QUARANTINE}")
    if not args.execute:
        print("dry-run; pass --execute to move")
        return
    for operation in operations:
        source = Path(operation["source"])
        destination = Path(operation["destination"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
    rollback = {
        "created_at": datetime.now().isoformat(),
        "source_root": str(SOURCE),
        "quarantine_root": str(QUARANTINE),
        "episodes": len(rows),
        "files": operations,
    }
    (QUARANTINE / "rollback_manifest.json").write_text(
        json.dumps(rollback, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"moved={len(operations)} rollback={QUARANTINE / 'rollback_manifest.json'}")


if __name__ == "__main__":
    main()
