#!/usr/bin/env python3
import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

SOURCE = Path("/data/hdt/ntv_data/cyj/data_arr")
QUARANTINE = Path("/data/hdt/ntv_data/cyj/data_arr_quarantine_bbox_full_failure_20260821")
SELECTION = Path(__file__).resolve().parents[1] / "reports/data_arr_full_bbox_failure_selection/selection_manifest.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    operations = []
    for relative_dir, episode in selection["episodes"]:
        source_dir = SOURCE / relative_dir
        files = sorted(source_dir.glob(f"{episode}_*"))
        stat = source_dir / f"{episode}.json"
        if stat.is_file():
            files.append(stat)
        if not files:
            raise FileNotFoundError(f"missing episode: {relative_dir}/{episode}")
        destination_dir = QUARANTINE / relative_dir
        for source in files:
            destination = destination_dir / source.name
            if destination.exists():
                raise FileExistsError(destination)
            operations.append({
                "source": str(source),
                "destination": str(destination),
                "relative_dir": relative_dir,
                "episode": str(episode),
                "reason": "both_agents_full_episode_bbox_missing",
            })
    print(f"episodes={len(selection['episodes'])} files={len(operations)} quarantine={QUARANTINE}")
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
        "episodes": len(selection["episodes"]),
        "files": operations,
    }
    rollback_path = QUARANTINE / "rollback_manifest.json"
    rollback_path.write_text(json.dumps(rollback, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"moved={len(operations)} rollback={rollback_path}")


if __name__ == "__main__":
    main()
