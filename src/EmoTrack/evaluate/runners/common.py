from __future__ import annotations

from EmoTrack.evaluate.core.config import EvaluationConfig
from EmoTrack.evaluate.core.scores import DetectorRunScore, SessionExampleRow, group_rows_by_run
from EmoTrack.evaluate.utils import load_detector_artifact, result_detector_path_for_row


def _artifact_namespace(config: EvaluationConfig) -> str:
    return str(config.output.artifact_namespace or "")


def _load_existing_run_score(
    *,
    method_name: str,
    transcript_view: str,
    run_rows: list[SessionExampleRow],
    seed: int = 0,
    namespace: str = "",
) -> DetectorRunScore | None:
    if not run_rows:
        return None
    artifact_path = result_detector_path_for_row(
        run_rows[0],
        method_name=method_name,
        transcript_view=transcript_view,
        seed=seed,
        namespace=namespace,
    )
    if not artifact_path.is_file():
        return None
    return load_detector_artifact(artifact_path)


def _load_complete_existing_run_scores(
    *,
    method_name: str,
    transcript_view: str,
    test_rows: list[SessionExampleRow],
    seed: int = 0,
    namespace: str = "",
) -> list[DetectorRunScore] | None:
    grouped_items = list(group_rows_by_run(test_rows).items())
    if not grouped_items:
        return []
    scores: list[DetectorRunScore] = []
    for _, run_rows in grouped_items:
        existing_score = _load_existing_run_score(
            method_name=method_name,
            transcript_view=transcript_view,
            run_rows=run_rows,
            seed=seed,
            namespace=namespace,
        )
        if existing_score is None:
            return None
        scores.append(existing_score)
    return sorted(scores, key=lambda score: (score.trend_name, score.source_run_id))
