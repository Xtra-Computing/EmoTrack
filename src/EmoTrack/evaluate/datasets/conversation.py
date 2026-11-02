from __future__ import annotations

import json
from pathlib import Path
import random

from EmoTrack.evaluate.core.scores import SessionExampleRow

from EmoTrack.evaluate.utils import (
    load_conversation_artifact,
    render_session_transcript,
)


def load_conversation_rows(
    root: Path, *, transcript_view: str = "full", include_history: bool = False,
) -> list[SessionExampleRow]:
    rows: list[SessionExampleRow] = []
    for run_dir in iter_conversation_run_dirs(root):
        artifact = load_conversation_artifact(run_dir)

        for idx, session in enumerate(artifact.sessions):
            if session.self_report_total is None:
                continue

            if include_history and idx > 0:
                prior_parts = []
                for prior in artifact.sessions[:idx]:
                    prior_parts.append(
                        f"[Session {prior.session_id}]\n"
                        + render_session_transcript(prior, transcript_view=transcript_view)
                    )
                current_part = (
                    f"[Session {session.session_id} (current)]\n"
                    + render_session_transcript(session, transcript_view=transcript_view)
                )
                transcript = "\n\n".join(prior_parts + [current_part])
            else:
                transcript = render_session_transcript(
                    session, transcript_view=transcript_view,
                )

            rows.append(
                SessionExampleRow(
                    dataset_name="conversation",
                    profile_id=artifact.profile_id,
                    run_id=artifact.run_id,
                    trend_name=artifact.trend_name,
                    session_index=session.session_id,
                    subject_id=artifact.profile_id,
                    split="",
                    transcript_full=transcript,
                    target_total=session.self_report_total,
                    target_item_scores=dict(session.self_report_item_scores),
                    latent_total=session.latent_total,
                    latent_item_scores=dict(session.latent_item_scores),
                    eval_ready_path=None,
                    source_path=run_dir,
                )
            )
    return rows


def iter_conversation_run_dirs(root: Path) -> list[Path]:
    run_dirs: list[Path] = []
    for child in sorted(path for path in root.iterdir() if path.is_dir()):
        if (child / "common" / "transcript.jsonl").is_file():
            run_dirs.append(child)
            continue
        for run_dir in sorted(path for path in child.iterdir() if path.is_dir()):
            if (run_dir / "common" / "transcript.jsonl").is_file():
                run_dirs.append(run_dir)
    return run_dirs


def load_split_manifest(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_conversation_split_manifest(
    rows: list[SessionExampleRow],
    *,
    split_file: Path | None = None,
    ratios: tuple[float, float, float] = (0.8, 0.0, 0.2),
    seed: int = 42,
) -> dict[str, object]:
    if split_file is not None and split_file.is_file():
        return load_split_manifest(split_file)
    run_info_by_split_trend: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        split_trend_name = row.source_path.parent.name or row.trend_name or "default"
        trend_runs = run_info_by_split_trend.setdefault(split_trend_name, {})
        if row.run_id not in trend_runs:
            trend_runs[row.run_id] = {
                "trend_name": split_trend_name,
                "profile_id": row.profile_id,
            }

    rng = random.Random(seed)
    train_ratio, val_ratio, test_ratio = ratios
    manifest_rows: list[dict[str, str]] = []
    trend_counts: dict[str, dict[str, int]] = {}
    for split_trend_name in sorted(run_info_by_split_trend):
        trend_runs = run_info_by_split_trend[split_trend_name]
        run_ids = sorted(trend_runs)
        rng.shuffle(run_ids)

        n = len(run_ids)
        n_train = round(n * train_ratio)
        n_val = round(n * val_ratio)
        n_test = n - n_train - n_val
        trend_counts[split_trend_name] = {
            "train": n_train,
            "val": n_val,
            "test": n_test,
            "total": n,
        }

        splits = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
        for run_id, split in zip(run_ids, splits):
            info = trend_runs[run_id]
            manifest_rows.append(
                {
                    "trend_name": info["trend_name"],
                    "profile_id": info["profile_id"],
                    "run_id": run_id,
                    "split": split,
                }
            )

    n_train = sum(item["train"] for item in trend_counts.values())
    n_val = sum(item["val"] for item in trend_counts.values())
    n_test = sum(item["test"] for item in trend_counts.values())
    n_total = sum(item["total"] for item in trend_counts.values())

    manifest = {
        "dataset_name": "conversation",
        "split_rule": f"per-parent-trend random shuffle (seed={seed}), ratios={ratios}",
        "seed": seed,
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "counts": {"train": n_train, "val": n_val, "test": n_test, "total": n_total},
        "trend_counts": trend_counts,
        "runs": sorted(manifest_rows, key=lambda row: row["run_id"]),
    }

    if split_file is not None:
        split_file.parent.mkdir(parents=True, exist_ok=True)
        split_file.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return manifest
