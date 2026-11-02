from __future__ import annotations

import json
from pathlib import Path

from EmoTrack.evaluate.core.scores import SessionExampleRow

from EmoTrack.evaluate.utils import (
    load_conversation_artifact,
    render_session_transcript,
)

_SPLIT_ORDER = {"train": 0, "dev": 1, "test": 2}


def iter_daic_woz_run_dirs(root: Path) -> list[Path]:
    resolved_root = Path(root).expanduser().resolve()
    return sorted(
        path
        for path in resolved_root.iterdir()
        if path.is_dir() and (path / "common" / "transcript.jsonl").is_file()
    )


def load_daic_woz_split_manifest(path: Path) -> dict[str, object]:
    resolved = Path(path).expanduser().resolve()
    return json.loads(resolved.read_text(encoding="utf-8"))


def load_daic_woz_rows(
    data_root: Path,
    *,
    transcript_view: str = "full",
    split_file: Path | None = None,
    dataset_name: str = "daic_woz",
) -> list[SessionExampleRow]:
    if dataset_name not in {"daic_woz", "edaic"}:
        raise ValueError(f"Unsupported DAIC-family dataset name: {dataset_name}")
    resolved_root = Path(data_root).expanduser().resolve()
    resolved_split = (
        Path(split_file).expanduser().resolve()
        if split_file is not None
        else resolved_root / "train_val_test_split.json"
    )
    if not resolved_split.is_file():
        raise FileNotFoundError(f"Missing DAIC-WOZ split manifest under {resolved_split}")

    manifest = load_daic_woz_split_manifest(resolved_split)
    split_by_run_id = {
        str(item["run_id"]): str(item["split"])
        for item in manifest.get("runs", [])
        if isinstance(item, dict)
    }

    rows: list[SessionExampleRow] = []
    missing_run_ids: list[str] = []
    for run_dir in iter_daic_woz_run_dirs(resolved_root):
        split = split_by_run_id.get(run_dir.name)
        if split is None:
            missing_run_ids.append(run_dir.name)
            continue

        artifact = load_conversation_artifact(run_dir)
        for session in artifact.sessions:
            rows.append(
                SessionExampleRow(
                    dataset_name=dataset_name,
                    profile_id=artifact.profile_id,
                    run_id=artifact.run_id,
                    trend_name=dataset_name,
                    session_index=session.session_id,
                    subject_id=artifact.profile_id,
                    split=split,
                    transcript_full=render_session_transcript(
                        session,
                        transcript_view=transcript_view,
                    ),
                    target_total=session.latent_total,
                    target_item_scores=dict(session.latent_item_scores),
                    latent_total=session.latent_total,
                    latent_item_scores=dict(session.latent_item_scores),
                    eval_ready_path=None,
                    source_path=run_dir,
                )
            )

    if missing_run_ids:
        preview = ", ".join(sorted(missing_run_ids)[:10])
        raise ValueError(
            f"DAIC-WOZ split manifest {resolved_split} is missing run ids for {len(missing_run_ids)} runs, "
            f"for example: {preview}"
        )

    return sorted(rows, key=lambda row: (_SPLIT_ORDER.get(row.split, 99), int(row.subject_id), row.session_index))
