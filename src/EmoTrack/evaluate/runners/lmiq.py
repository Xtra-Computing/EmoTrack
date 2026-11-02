from __future__ import annotations

from pathlib import Path

from EmoTrack.evaluate.core.config import EvaluationConfig
from EmoTrack.evaluate.core.context import PreparedLLMWorker
from EmoTrack.evaluate.core.scores import DetectorRunScore, SessionExampleRow, build_run_score, group_rows_by_run
from EmoTrack.evaluate.methods.lmiq import (
    extract_lmiq_features,
    fit_lmiq_model,
    predict_lmiq,
    select_lmiq_hyperparameters,
    write_lmiq_model,
    write_lmiq_model_metadata,
)
from EmoTrack.evaluate.runners.common import _artifact_namespace, _load_complete_existing_run_scores
from EmoTrack.evaluate.utils import make_async_openai_client, write_json, write_result_detector_artifact


async def run_lmiq(
    *,
    dataset_name: str,
    train_rows: list[SessionExampleRow],
    val_rows: list[SessionExampleRow],
    test_rows: list[SessionExampleRow],
    out_root: Path,
    config: EvaluationConfig,
    llm_workers: list[PreparedLLMWorker] | None,
    overwrite: bool = False,
) -> list[DetectorRunScore]:
    is_daic = config.dataset.name == "daic_woz"
    if is_daic and config.dataset.include_history:
        raise ValueError("DAIC-WOZ is single-session; include_history must be false. Use lmiq_now, not lmiq_all.")
    lmiq_suffix = "_all" if config.dataset.include_history else "_now"
    seed = config.method.inference_seed
    lmiq_method_name = f"lmiq{lmiq_suffix}"
    if not overwrite:
        existing_scores = _load_complete_existing_run_scores(
            method_name=lmiq_method_name,
            transcript_view=config.dataset.transcript_view,
            test_rows=test_rows,
            seed=seed,
            namespace=_artifact_namespace(config),
        )
        if existing_scores is not None:
            return existing_scores
    clients = [
        make_async_openai_client(
            base_url=worker.base_url,
            api_key=worker.api_key,
            allow_missing_api_key=config.llm.backend == "local",
        )
        for worker in llm_workers
    ] if llm_workers else []
    model_name = llm_workers[0].model_name if llm_workers else config.llm.model_name
    question_bank_path = config.method.question_bank_path or "configs/evaluate/lmiq_question_bank.json"
    features_by_row_key = await extract_lmiq_features(
        train_rows + val_rows + test_rows,
        llm_clients=clients,
        model_name=model_name,
        cache_path=out_root / "feature_cache" / "lmiq_features.json",
        question_bank_path=question_bank_path,
        max_concurrency=config.method.feature_concurrency,
        inference_seed=seed,
    )

    if dataset_name == "conversation":
        selection_train_rows = train_rows
        validation_rows = val_rows
        if not selection_train_rows:
            raise ValueError("LMIQ conversation evaluation requires non-empty train rows")
        if not validation_rows:
            raise ValueError("LMIQ conversation evaluation requires non-empty validation rows")
        write_json(
            out_root / "model" / "selection_split.json",
            {
                "dataset_name": "conversation",
                "split_rule": "use manifest-provided train/val runs for LMIQ selection",
                "runs": [
                    *[
                        {
                            "trend_name": trend_name,
                            "profile_id": run_rows[0].profile_id,
                            "run_id": run_id,
                            "split": "train",
                        }
                        for (trend_name, run_id), run_rows in group_rows_by_run(selection_train_rows).items()
                    ],
                    *[
                        {
                            "trend_name": trend_name,
                            "profile_id": run_rows[0].profile_id,
                            "run_id": run_id,
                            "split": "val",
                        }
                        for (trend_name, run_id), run_rows in group_rows_by_run(validation_rows).items()
                    ],
                ],
            },
        )
        final_train_rows = train_rows
    else:
        selection_train_rows = [row for row in train_rows if row.split == "train"]
        validation_rows = [row for row in val_rows if row.split in {"val", "dev"}]
        final_train_rows = selection_train_rows

    best_params, selection_trials = select_lmiq_hyperparameters(
        selection_train_rows,
        validation_rows,
        features_by_row_key=features_by_row_key,
        n_estimators_grid=config.method.rf_n_estimators_grid,
        max_depth_grid=config.method.rf_max_depth_grid,
        random_state=seed,
    )
    model = fit_lmiq_model(
        final_train_rows,
        features_by_row_key=features_by_row_key,
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        random_state=seed,
    )
    write_lmiq_model(model, out_root / "model" / "lmiq_model.joblib")
    write_lmiq_model_metadata(
        path=out_root / "model" / "lmiq_model_meta.json",
        question_bank_path=question_bank_path,
        selection_metric="mse",
        best_params=best_params,
        selection_train_size=len(selection_train_rows),
        selection_validation_size=len(validation_rows),
        final_train_size=len(final_train_rows),
        selection_trials=selection_trials,
    )
    predictions = predict_lmiq(test_rows, model=model, features_by_row_key=features_by_row_key)

    run_scores: list[DetectorRunScore] = []
    for (trend_name, run_id), run_rows in group_rows_by_run(test_rows).items():
        score = build_run_score(
            method_name=lmiq_method_name,
            scoring_model=model_name,
            transcript_view=config.dataset.transcript_view,
            rows=run_rows,
            predictions_by_row_key=predictions,
        )
        write_result_detector_artifact(score, seed=seed, namespace=_artifact_namespace(config))
        run_scores.append(score)
    return run_scores
