from __future__ import annotations

from pathlib import Path

from EmoTrack.evaluate.core.config import EvaluationConfig
from EmoTrack.evaluate.core.scores import DetectorRunScore, SessionExampleRow, build_run_score, group_rows_by_run
from EmoTrack.evaluate.runners.common import _artifact_namespace, _load_complete_existing_run_scores
from EmoTrack.evaluate.utils import write_result_detector_artifact


async def run_ensembert(
    *,
    dataset_name: str,
    train_rows: list[SessionExampleRow],
    val_rows: list[SessionExampleRow],
    test_rows: list[SessionExampleRow],
    out_root: Path,
    config: EvaluationConfig,
    overwrite: bool = False,
) -> list[DetectorRunScore]:
    if dataset_name == "daic_woz" and config.dataset.include_history:
        raise ValueError("DAIC-WOZ is single-session; include_history must be false for EnsemBERT.")
    from EmoTrack.evaluate.methods.ensembert.runner import config_from_method_config, fit_predict_ensembert

    run_config = config_from_method_config(config.method, include_history=config.dataset.include_history)
    ensembert_method_name = f"{run_config.method_prefix}_{'all' if run_config.include_history else 'now'}"
    if not overwrite:
        existing_scores = _load_complete_existing_run_scores(
            method_name=ensembert_method_name,
            transcript_view=config.dataset.transcript_view,
            test_rows=test_rows,
            seed=run_config.seed,
            namespace=_artifact_namespace(config),
        )
        if existing_scores is not None:
            return existing_scores
    result = fit_predict_ensembert(
        train_rows=train_rows,
        val_rows=val_rows,
        test_rows=test_rows,
        out_root=out_root,
        config=run_config,
        inner_val_fraction=config.method.inner_val_fraction,
    )
    run_scores: list[DetectorRunScore] = []
    for (_trend_name, _run_id), run_rows in group_rows_by_run(test_rows).items():
        score = build_run_score(
            method_name=result.method_name,
            scoring_model=result.scoring_model,
            transcript_view=config.dataset.transcript_view,
            rows=run_rows,
            predictions_by_row_key=result.predictions_by_row_key,
            predicted_items_by_row_key=result.predicted_items_by_row_key,
        )
        write_result_detector_artifact(score, seed=run_config.seed, namespace=_artifact_namespace(config))
        run_scores.append(score)
    return run_scores


async def run_dual_encoder(
    *,
    dataset_name: str,
    train_rows: list[SessionExampleRow],
    val_rows: list[SessionExampleRow],
    test_rows: list[SessionExampleRow],
    out_root: Path,
    config: EvaluationConfig,
    overwrite: bool = False,
) -> list[DetectorRunScore]:
    if dataset_name == "daic_woz" and config.dataset.include_history:
        raise ValueError("DAIC-WOZ is single-session; include_history must be false for Dual Encoder.")
    from EmoTrack.evaluate.methods.dual_encoder import config_from_method_config, fit_predict_dual_encoder

    run_config = config_from_method_config(config.method, include_history=config.dataset.include_history)
    method_name = f"{run_config.method_prefix}_{'all' if run_config.include_history else 'now'}"
    if not overwrite:
        existing_scores = _load_complete_existing_run_scores(
            method_name=method_name,
            transcript_view=config.dataset.transcript_view,
            test_rows=test_rows,
            seed=run_config.seed,
            namespace=_artifact_namespace(config),
        )
        if existing_scores is not None:
            return existing_scores
    result = fit_predict_dual_encoder(
        train_rows=train_rows,
        val_rows=val_rows,
        test_rows=test_rows,
        out_root=out_root,
        config=run_config,
        inner_val_fraction=config.method.inner_val_fraction,
    )
    run_scores: list[DetectorRunScore] = []
    for (_trend_name, _run_id), run_rows in group_rows_by_run(test_rows).items():
        score = build_run_score(
            method_name=result.method_name,
            scoring_model=result.scoring_model,
            transcript_view=config.dataset.transcript_view,
            rows=run_rows,
            predictions_by_row_key=result.predictions_by_row_key,
        )
        write_result_detector_artifact(score, seed=run_config.seed, namespace=_artifact_namespace(config))
        run_scores.append(score)
    return run_scores


async def run_milintsevich(
    *,
    dataset_name: str,
    train_rows: list[SessionExampleRow],
    val_rows: list[SessionExampleRow],
    test_rows: list[SessionExampleRow],
    out_root: Path,
    config: EvaluationConfig,
    overwrite: bool = False,
) -> list[DetectorRunScore]:
    if dataset_name == "daic_woz" and config.dataset.include_history:
        raise ValueError("DAIC-WOZ is single-session; include_history must be false for Milintsevich.")
    from EmoTrack.evaluate.methods.milintsevich import config_from_method_config, fit_predict_milintsevich

    run_config = config_from_method_config(config.method, include_history=config.dataset.include_history)
    method_name = f"{run_config.method_prefix}_{'all' if run_config.include_history else 'now'}"
    if not overwrite:
        existing_scores = _load_complete_existing_run_scores(
            method_name=method_name,
            transcript_view=config.dataset.transcript_view,
            test_rows=test_rows,
            seed=run_config.seed,
            namespace=_artifact_namespace(config),
        )
        if existing_scores is not None:
            return existing_scores
    result = fit_predict_milintsevich(
        train_rows=train_rows,
        val_rows=val_rows,
        test_rows=test_rows,
        out_root=out_root,
        config=run_config,
        inner_val_fraction=config.method.inner_val_fraction,
    )
    run_scores: list[DetectorRunScore] = []
    for (_trend_name, _run_id), run_rows in group_rows_by_run(test_rows).items():
        score = build_run_score(
            method_name=result.method_name,
            scoring_model=result.scoring_model,
            transcript_view=config.dataset.transcript_view,
            rows=run_rows,
            predictions_by_row_key=result.predictions_by_row_key,
            predicted_items_by_row_key=result.predicted_items_by_row_key,
        )
        write_result_detector_artifact(score, seed=run_config.seed, namespace=_artifact_namespace(config))
        run_scores.append(score)
    return run_scores
