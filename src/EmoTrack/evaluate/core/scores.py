from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Iterable, Literal

from EmoTrack.evaluate.core.phq8 import DEFAULT_PHQ8_SYMPTOM_ORDER


TranscriptView = Literal["client-only", "full"]
DatasetName = Literal["conversation", "daic_woz"]


@dataclass(slots=True)
class TranscriptTurn:
    turn_id: int
    speaker: str
    text: str


@dataclass(slots=True)
class ConversationSessionArtifact:
    session_id: int
    turns: list[TranscriptTurn]
    latent_item_scores: dict[str, float]
    latent_total: float
    self_report_item_scores: dict[str, float]
    self_report_total: float | None


@dataclass(slots=True)
class ConversationArtifact:
    run_id: str
    trend_name: str
    profile_id: str
    source_dir: Path
    transcript_path: Path
    self_report_path: Path | None
    ground_truth_path: Path | None
    sessions: list[ConversationSessionArtifact]


@dataclass(slots=True)
class SessionExampleRow:
    dataset_name: DatasetName
    profile_id: str
    run_id: str
    trend_name: str
    session_index: int
    subject_id: str
    split: str
    transcript_full: str
    target_total: float | None
    target_item_scores: dict[str, float]
    latent_total: float | None
    latent_item_scores: dict[str, float]
    eval_ready_path: Path | None
    source_path: Path


@dataclass(slots=True)
class DetectorItemScore:
    question_id: int
    symptom_name: str
    question_text: str
    score: float
    score_std: float
    sample_scores: list[int]
    option_label: str
    evidence: str


@dataclass(slots=True)
class DetectorSessionScore:
    session_index: int
    transcript_view: str
    sample_count: int
    predicted_total: float
    predicted_item_scores: dict[str, float] | None = None
    target_item_scores: dict[str, float] | None = None
    latent_item_scores: dict[str, float] | None = None
    item_mean_absolute_error_to_target: float | None = None
    item_mean_absolute_error_to_latent: float | None = None
    per_item_absolute_error_to_target: dict[str, float] | None = None
    per_item_absolute_error_to_latent: dict[str, float] | None = None
    target_total: float | None = None
    latent_total: float | None = None
    self_report_total: float | None = None
    absolute_error_to_target: float | None = None
    absolute_error_to_latent: float | None = None
    absolute_error_to_self_report: float | None = None
    items: list[DetectorItemScore] | None = None


@dataclass(slots=True)
class DetectorRunScore:
    method: str
    scored_at: str
    scoring_model: str
    source_run_path: str
    source_run_id: str
    transcript_view: str
    sample_count: int
    session_count: int
    sessions: list[DetectorSessionScore]
    dataset_name: str = ""
    trend_name: str = ""
    profile_id: str = ""
    subject_id: str = ""
    split: str = ""
    target_label: str = "phq8_total"
    mean_absolute_error_to_target: float | None = None
    mean_absolute_error_to_latent: float | None = None
    mean_absolute_error_to_self_report: float | None = None


def apply_split_manifest(
    rows: list[SessionExampleRow],
    manifest: dict[str, object],
) -> list[SessionExampleRow]:
    run_splits = {
        str(item["run_id"]): str(item["split"])
        for item in manifest.get("runs", [])
        if isinstance(item, dict)
    }
    assigned: list[SessionExampleRow] = []
    for row in rows:
        split = run_splits.get(row.run_id, row.split)
        assigned.append(
            SessionExampleRow(
                dataset_name=row.dataset_name,
                profile_id=row.profile_id,
                run_id=row.run_id,
                trend_name=row.trend_name,
                session_index=row.session_index,
                subject_id=row.subject_id,
                split=split,
                transcript_full=row.transcript_full,
                target_total=row.target_total,
                target_item_scores=dict(row.target_item_scores),
                latent_total=row.latent_total,
                latent_item_scores=dict(row.latent_item_scores),
                eval_ready_path=row.eval_ready_path,
                source_path=row.source_path,
            )
        )
    return assigned


def group_rows_by_run(rows: Iterable[SessionExampleRow]) -> dict[tuple[str, str], list[SessionExampleRow]]:
    grouped: dict[tuple[str, str], list[SessionExampleRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.trend_name, row.run_id)].append(row)
    for key in grouped:
        grouped[key] = sorted(grouped[key], key=lambda row: row.session_index)
    return dict(sorted(grouped.items(), key=lambda item: item[0]))


def item_error_summary(
    predicted_items: dict[str, float] | None,
    reference_items: dict[str, float] | None,
) -> tuple[float | None, dict[str, float] | None]:
    if predicted_items is None or reference_items is None:
        return None, None
    errors: dict[str, float] = {}
    for symptom_name in DEFAULT_PHQ8_SYMPTOM_ORDER:
        if symptom_name not in predicted_items or symptom_name not in reference_items:
            return None, None
        errors[symptom_name] = abs(float(predicted_items[symptom_name]) - float(reference_items[symptom_name]))
    return (
        round(fmean(errors.values()), 4),
        {symptom_name: round(error, 4) for symptom_name, error in errors.items()},
    )


def build_run_score(
    *,
    method_name: str,
    scoring_model: str,
    transcript_view: str,
    rows: list[SessionExampleRow],
    predictions_by_row_key: dict[str, float],
    predicted_items_by_row_key: dict[str, dict[str, float]] | None = None,
) -> DetectorRunScore:
    if not rows:
        raise ValueError("Cannot build a run score with no rows.")

    sessions: list[DetectorSessionScore] = []
    for row in rows:
        row_key = make_row_key(row)
        predicted_total = float(predictions_by_row_key[row_key])
        target_total = float(row.target_total) if row.target_total is not None else None
        latent_total = float(row.latent_total) if row.latent_total is not None else None
        predicted_items = (
            dict(predicted_items_by_row_key[row_key])
            if predicted_items_by_row_key and row_key in predicted_items_by_row_key
            else None
        )
        target_items = dict(row.target_item_scores) if predicted_items is not None else None
        latent_items = dict(row.latent_item_scores) if predicted_items is not None else None
        target_item_mae, target_per_item_errors = item_error_summary(predicted_items, target_items)
        latent_item_mae, latent_per_item_errors = item_error_summary(predicted_items, latent_items)
        sessions.append(
            DetectorSessionScore(
                session_index=row.session_index,
                transcript_view=transcript_view,
                sample_count=1,
                predicted_total=round(predicted_total, 4),
                predicted_item_scores=predicted_items,
                target_item_scores=target_items,
                latent_item_scores=latent_items,
                item_mean_absolute_error_to_target=target_item_mae,
                item_mean_absolute_error_to_latent=latent_item_mae,
                per_item_absolute_error_to_target=target_per_item_errors,
                per_item_absolute_error_to_latent=latent_per_item_errors,
                target_total=target_total,
                latent_total=latent_total,
                self_report_total=target_total if row.dataset_name == "conversation" else None,
                absolute_error_to_target=(
                    round(abs(predicted_total - target_total), 4) if target_total is not None else None
                ),
                absolute_error_to_latent=(
                    round(abs(predicted_total - latent_total), 4) if latent_total is not None else None
                ),
                absolute_error_to_self_report=(
                    round(abs(predicted_total - target_total), 4)
                    if row.dataset_name == "conversation" and target_total is not None
                    else None
                ),
                items=None,
            )
        )

    target_errors = [
        session.absolute_error_to_target
        for session in sessions
        if session.absolute_error_to_target is not None
    ]
    latent_errors = [
        session.absolute_error_to_latent
        for session in sessions
        if session.absolute_error_to_latent is not None
    ]
    self_report_errors = [
        session.absolute_error_to_self_report
        for session in sessions
        if session.absolute_error_to_self_report is not None
    ]
    first_row = rows[0]
    return DetectorRunScore(
        method=method_name,
        scored_at=datetime.now(timezone.utc).isoformat(),
        scoring_model=scoring_model,
        source_run_path=str(first_row.source_path),
        source_run_id=first_row.run_id,
        transcript_view=transcript_view,
        sample_count=1,
        session_count=len(sessions),
        sessions=sessions,
        dataset_name=first_row.dataset_name,
        trend_name=first_row.trend_name,
        profile_id=first_row.profile_id,
        subject_id=first_row.subject_id,
        split=first_row.split,
        target_label="phq8_total",
        mean_absolute_error_to_target=round(fmean(target_errors), 4) if target_errors else None,
        mean_absolute_error_to_latent=round(fmean(latent_errors), 4) if latent_errors else None,
        mean_absolute_error_to_self_report=round(fmean(self_report_errors), 4) if self_report_errors else None,
    )


def build_metrics_payload(run_scores: list[DetectorRunScore]) -> dict[str, object]:
    session_rows: list[dict[str, object]] = []
    for run_score in run_scores:
        for session in run_score.sessions:
            metric_error = _metric_error_for_session(session)
            metric_item_error = _metric_item_error_for_session(session)
            metric_per_item_error = _metric_per_item_error_for_session(session)
            row: dict[str, object] = {
                "trend_name": run_score.trend_name,
                "profile_id": run_score.profile_id,
                "run_id": run_score.source_run_id,
                "session_index": session.session_index,
                "predicted_total": session.predicted_total,
                "target_total": session.target_total,
                "latent_total": session.latent_total,
                "absolute_error_to_target": session.absolute_error_to_target,
                "absolute_error_to_latent": session.absolute_error_to_latent,
                "absolute_error_for_metrics": metric_error,
            }
            if session.predicted_item_scores is not None:
                row.update(
                    {
                        "predicted_item_scores": session.predicted_item_scores,
                        "target_item_scores": session.target_item_scores,
                        "latent_item_scores": session.latent_item_scores,
                        "item_mae_to_target": session.item_mean_absolute_error_to_target,
                        "item_mae_to_latent": session.item_mean_absolute_error_to_latent,
                        "item_mae_for_metrics": metric_item_error,
                        "per_item_absolute_error_to_target": session.per_item_absolute_error_to_target,
                        "per_item_absolute_error_to_latent": session.per_item_absolute_error_to_latent,
                        "per_item_absolute_error_for_metrics": metric_per_item_error,
                    }
                )
            session_rows.append(row)

    valid_metric_errors = [
        float(row["absolute_error_for_metrics"])
        for row in session_rows
        if row["absolute_error_for_metrics"] is not None
    ]
    by_trend: dict[str, list[float]] = defaultdict(list)
    for row in session_rows:
        error = row["absolute_error_for_metrics"]
        if error is None:
            continue
        by_trend[str(row["trend_name"])].append(float(error))

    payload: dict[str, object] = {
        "session_count": len(session_rows),
        "overall_mae": round(fmean(valid_metric_errors), 4) if valid_metric_errors else None,
        "trends": {
            trend_name: {
                "session_count": len(errors),
                "mae": round(fmean(errors), 4) if errors else None,
            }
            for trend_name, errors in sorted(by_trend.items())
        },
        "sessions": session_rows,
    }
    valid_metric_item_errors = [
        float(row["item_mae_for_metrics"])
        for row in session_rows
        if row.get("item_mae_for_metrics") is not None
    ]
    if valid_metric_item_errors:
        payload["overall_item_mae"] = round(fmean(valid_metric_item_errors), 4)
        per_item_values: dict[str, list[float]] = defaultdict(list)
        for row in session_rows:
            per_item_error = row.get("per_item_absolute_error_for_metrics")
            if not isinstance(per_item_error, dict):
                continue
            for symptom_name, error in per_item_error.items():
                per_item_values[str(symptom_name)].append(float(error))
        payload["per_item_mae"] = {
            symptom_name: round(fmean(per_item_values[symptom_name]), 4)
            for symptom_name in DEFAULT_PHQ8_SYMPTOM_ORDER
            if symptom_name in per_item_values
        }
    return payload


def _metric_error_for_session(session: DetectorSessionScore) -> float | None:
    if session.absolute_error_to_latent is not None:
        return float(session.absolute_error_to_latent)
    return (
        float(session.absolute_error_to_target)
        if session.absolute_error_to_target is not None
        else None
    )


def _metric_item_error_for_session(session: DetectorSessionScore) -> float | None:
    if session.item_mean_absolute_error_to_latent is not None:
        return float(session.item_mean_absolute_error_to_latent)
    return (
        float(session.item_mean_absolute_error_to_target)
        if session.item_mean_absolute_error_to_target is not None
        else None
    )


def _metric_per_item_error_for_session(session: DetectorSessionScore) -> dict[str, float] | None:
    if session.per_item_absolute_error_to_latent is not None:
        return dict(session.per_item_absolute_error_to_latent)
    return (
        dict(session.per_item_absolute_error_to_target)
        if session.per_item_absolute_error_to_target is not None
        else None
    )


def make_row_key(row: SessionExampleRow) -> str:
    return f"{row.run_id}::s{row.session_index}"
