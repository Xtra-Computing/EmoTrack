"""Check the files required by a frozen dataset split."""

import json
from pathlib import Path

INTERVIEW_FILES = ("transcript.jsonl", "ground_truth.json")
CONVERSATION_FILES = (*INTERVIEW_FILES, "self_report.json")


def require_complete_bundle(
    root: Path,
    split_file: Path,
    filenames: tuple[str, ...] = INTERVIEW_FILES,
) -> None:
    manifest = json.loads(split_file.read_text(encoding="utf-8"))
    runs = manifest.get("runs", [])
    if not runs:
        raise ValueError(f"No runs in split manifest: {split_file}")
    missing = []
    for run in runs:
        run_id = str(run["run_id"])
        if Path(run_id).name != run_id or run_id in {".", ".."}:
            raise ValueError(f"Invalid run ID in {split_file}: {run_id!r}")
        for filename in filenames:
            path = root / run_id / "common" / filename
            if not path.is_file() or path.stat().st_size == 0:
                missing.append(str(path))
    if missing:
        raise FileNotFoundError(
            f"Incomplete dataset: {len(missing)} missing or empty files; first: {missing[:5]}. "
            "Restore these files or rerun local preparation."
        )
