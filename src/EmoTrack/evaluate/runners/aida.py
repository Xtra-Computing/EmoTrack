from __future__ import annotations

from pathlib import Path

from EmoTrack.evaluate.core.config import EvaluationConfig
from EmoTrack.evaluate.core.context import PreparedLLMWorker
from EmoTrack.evaluate.core.scores import (
    DetectorRunScore,
    SessionExampleRow,
    build_run_score,
    group_rows_by_run,
)
from EmoTrack.evaluate.methods.aida import (
    _EXTRA_GROUPS,
    extract_aida_features,
    fit_aida_model,
    predict_aida,
    write_aida_model,
)
from EmoTrack.evaluate.runners.common import (
    _artifact_namespace,
    _load_complete_existing_run_scores,
)
from EmoTrack.evaluate.utils import (
    make_async_openai_client,
    write_result_detector_artifact,
)


async def run_aida(
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
        raise ValueError("DAIC-WOZ is single-session; include_history must be false. Use aida_now, not aida_all.")
    aida_suffix = "_all" if config.dataset.include_history else "_now"
    seed = config.method.inference_seed
    aida_method_name = f"aida{aida_suffix}"
    clients = [
        make_async_openai_client(
            base_url=worker.base_url,
            api_key=worker.api_key,
            allow_missing_api_key=config.llm.backend == "local",
        )
        for worker in llm_workers
    ] if llm_workers else []
    train_buffer_rows = train_rows + val_rows
    all_rows = train_buffer_rows + test_rows

    aida_model_name = llm_workers[0].model_name if llm_workers else config.llm.model_name
    if config.method.extract_extra_only:
        # Only extract extra features (Neu + DepSev), skip main extraction and model fitting.
        _extra_prompts_kwargs: dict = {}
        if config.method.aida_prompts_path:
            _extra_prompts_kwargs["prompts_path"] = config.method.aida_prompts_path
        await extract_aida_features(
            all_rows,
            llm_clients=clients,
            model_name=aida_model_name,
            cache_path=out_root / "feature_cache" / "aida_extra_features.json",
            max_concurrency=config.method.feature_concurrency,
            groups=_EXTRA_GROUPS,
            inference_seed=seed,
            **_extra_prompts_kwargs,
        )
        return []

    if not overwrite and not config.method.extract_features_only:
        existing_scores = _load_complete_existing_run_scores(
            method_name=aida_method_name,
            transcript_view=config.dataset.transcript_view,
            test_rows=test_rows,
            seed=seed,
            namespace=_artifact_namespace(config),
        )
        if existing_scores is not None:
            return existing_scores
    _aida_prompts_kwargs: dict = {}
    if config.method.aida_prompts_path:
        _aida_prompts_kwargs["prompts_path"] = config.method.aida_prompts_path
    features_by_row_key = await extract_aida_features(
        all_rows,
        llm_clients=clients,
        model_name=aida_model_name,
        cache_path=(
            Path(config.method.aida_cache_path)
            if config.method.aida_cache_path
            else out_root / "feature_cache" / "aida_features.json"
        ),
        max_concurrency=config.method.feature_concurrency,
        inference_seed=seed,
        **_aida_prompts_kwargs,
    )
    if config.method.extract_features_only:
        return []
    # Also extract extra features (Neu + DepSev) into a separate cache.
    await extract_aida_features(
        all_rows,
        llm_clients=clients,
        model_name=aida_model_name,
        cache_path=out_root / "feature_cache" / "aida_extra_features.json",
        max_concurrency=config.method.feature_concurrency,
        groups=_EXTRA_GROUPS,
        inference_seed=seed,
    )
    model = fit_aida_model(train_buffer_rows, features_by_row_key)
    write_aida_model(model, out_root / "model" / "aida_model.json")
    predictions = predict_aida(test_rows, model=model, features_by_row_key=features_by_row_key)

    run_scores: list[DetectorRunScore] = []
    for (trend_name, run_id), run_rows in group_rows_by_run(test_rows).items():
        score = build_run_score(
            method_name=aida_method_name,
            scoring_model=aida_model_name,
            transcript_view=config.dataset.transcript_view,
            rows=run_rows,
            predictions_by_row_key=predictions,
        )
        write_result_detector_artifact(score, seed=seed, namespace=_artifact_namespace(config))
        run_scores.append(score)
    return run_scores
