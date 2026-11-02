from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from EmoTrack.evaluate.core.artifacts import detector_output_path
from EmoTrack.evaluate.core.config import EvaluationConfig
from EmoTrack.evaluate.core.context import PreparedLLMWorker
from EmoTrack.evaluate.core.scores import (
    ConversationArtifact,
    ConversationSessionArtifact,
    DetectorRunScore,
    DetectorSessionScore,
    SessionExampleRow,
    group_rows_by_run,
)
from EmoTrack.evaluate.methods.emotrack import EmoTrackEvaluator
from EmoTrack.evaluate.runners.common import _artifact_namespace, _load_existing_run_score
from EmoTrack.evaluate.utils import (
    OpenAIChatTextClient,
    load_conversation_artifact,
    load_detector_artifact,
    make_async_openai_client,
    write_result_detector_artifact,
)


async def run_emotrack(
    *,
    dataset_name: str,
    test_rows: list[SessionExampleRow],
    out_root: Path,
    config: EvaluationConfig,
    llm_workers: list[PreparedLLMWorker] | None,
    overwrite: bool = False,
    use_vllm: bool = False,
) -> list[DetectorRunScore]:
    if dataset_name == "conversation":
        return await run_emotrack_conversation(
            test_rows=test_rows,
            out_root=out_root,
            config=config,
            llm_workers=llm_workers,
            overwrite=overwrite,
            use_vllm=use_vllm,
        )
    return await run_emotrack_daic(
        test_rows=test_rows,
        out_root=out_root,
        config=config,
        llm_workers=llm_workers,
        overwrite=overwrite,
        use_vllm=use_vllm,
    )


async def run_emotrack_conversation(
    *,
    test_rows: list[SessionExampleRow],
    out_root: Path,
    config: EvaluationConfig,
    llm_workers: list[PreparedLLMWorker] | None,
    overwrite: bool = False,
    use_vllm: bool = False,
) -> list[DetectorRunScore]:
    seed = config.method.inference_seed
    grouped = group_rows_by_run(test_rows)
    run_scores: list[DetectorRunScore] = []
    if config.method.mode == "api":
        if not llm_workers:
            raise RuntimeError("EmoTrack api evaluation requires an llm endpoint.")
        clients = [
            OpenAIChatTextClient(
                client=make_async_openai_client(
                    base_url=worker.base_url,
                    api_key=worker.api_key,
                    allow_missing_api_key=config.llm.backend == "local",
                ),
                model_name=worker.model_name,
            )
            for worker in llm_workers
        ]
        evaluators = [
            EmoTrackEvaluator(
                sample_count=1,
                include_history=config.method.include_history,
                transcript_view=config.dataset.transcript_view,
                use_trend=config.method.use_trend,
                use_confidence=config.method.use_confidence,
                prompt_template_path=config.method.prompt_template or None,
            )
            for _ in llm_workers
        ]
        semaphores = [asyncio.Semaphore(max(1, config.method.eval_concurrency)) for _ in llm_workers]
        grouped_items = list(grouped.items())

        async def _score_one(index: int, item: tuple[tuple[str, str], list[SessionExampleRow]]) -> None:
            worker_index = index % len(llm_workers)
            (trend_name, run_id), _ = item
            artifact = load_conversation_artifact(_root_from_group(trend_name, run_id, test_rows))
            async with semaphores[worker_index]:
                score = await evaluators[worker_index].score_artifact(
                    artifact,
                    clients[worker_index],
                    scoring_model=llm_workers[worker_index].model_name,
                )
            score.split = "test"
            write_result_detector_artifact(score, seed=seed, namespace=_artifact_namespace(config))
            run_scores.append(score)

        await asyncio.gather(*(_score_one(index, item) for index, item in enumerate(grouped_items)))
        return sorted(run_scores, key=lambda score: (score.trend_name, score.source_run_id))

    if not config.method.checkpoint:
        raise ValueError("method.checkpoint is required for emotrack mode=embedding")

    import logging as _logging

    import torch

    _logger = _logging.getLogger(__name__)

    # Detect checkpoint type (AIDA family only)
    ckpt_path = Path(config.method.checkpoint)
    is_aida_attnagg = (ckpt_path / "aida_attnagg_config.json").exists()
    is_aida_gru = (ckpt_path / "aida_gru_config.json").exists()
    is_aida_stackattn = (ckpt_path / "aida_stackattn_config.json").exists()
    is_aida_embed_attnagg = (ckpt_path / "aida_embed_attnagg_config.json").exists()
    is_aida_embed_hybrid = (ckpt_path / "aida_embed_hybrid_config.json").exists()
    if is_aida_embed_hybrid:
        from EmoTrack.train.aida.model import _HYBRID_CONFIG
        method_name = "emotrack"
        config_filename = _HYBRID_CONFIG
    elif is_aida_embed_attnagg:
        method_name = "emotrack_hybrid_embed_attnagg"
        config_filename = "aida_embed_attnagg_config.json"
    elif is_aida_attnagg:
        method_name = "emotrack_aida_attnagg"
        config_filename = "aida_attnagg_config.json"
    elif is_aida_gru:
        method_name = "emotrack_aida_gru"
        config_filename = "aida_gru_config.json"
    elif is_aida_stackattn:
        method_name = "emotrack_aida_stackattn"
        config_filename = "aida_stackattn_config.json"
    else:
        raise ValueError(f"Unrecognised AIDA checkpoint at {ckpt_path}")

    # Override method_name from saved checkpoint config if available
    _ckpt_cfg_path = ckpt_path / config_filename
    if _ckpt_cfg_path.is_file():
        _ckpt_cfg = json.loads(_ckpt_cfg_path.read_text(encoding="utf-8"))
        if "result_name" in _ckpt_cfg:
            method_name = _ckpt_cfg["result_name"]

    pending_keys: list[tuple[str, str]] = []
    completed_scores: list[DetectorRunScore] = []
    for (trend_name, run_id), run_rows in grouped.items():
        if overwrite:
            pending_keys.append((trend_name, run_id))
            continue
        existing = _load_existing_run_score(
            method_name=method_name,
            transcript_view=config.dataset.transcript_view,
            run_rows=run_rows,
            seed=seed,
            namespace=_artifact_namespace(config),
        )
        if existing is not None:
            completed_scores.append(existing)
        else:
            pending_keys.append((trend_name, run_id))

    _logger.info(
        "%s scan: %d completed, %d pending",
        method_name, len(completed_scores), len(pending_keys),
    )
    if pending_keys:
        _logger.info("Pending runs: %s", [f"{t}_{r}" for t, r in pending_keys])
    run_scores.extend(completed_scores)

    if not pending_keys:
        return sorted(run_scores, key=lambda s: (s.trend_name, s.source_run_id))

    # Load model only if there is work to do
    ckpt_config = json.loads((ckpt_path / config_filename).read_text(encoding="utf-8"))
    if True:  # AIDA family only
        from EmoTrack.train.aida.model import (
            load_checkpoint_attnagg as _load_aida_attnagg,
            load_checkpoint_gru as _load_aida_gru,
            load_checkpoint_stackattn as _load_aida_stackattn,
            predict_from_cache_attnagg as _predict_aida_attnagg,
            predict_from_cache_gru as _predict_aida_gru,
            predict_from_cache_stackattn as _predict_aida_stackattn,
        )
        from EmoTrack.train.aida.dataset import load_aida_feature_cache, _merge_extra_cache
        if is_aida_embed_hybrid:
            from EmoTrack.train.aida.model import load_checkpoint_hybrid as _load_hybrid
            model = _load_hybrid(config.method.checkpoint)
        elif is_aida_embed_attnagg:
            from EmoTrack.train.aida.model import load_checkpoint_embed_attnagg as _load_aida_embed
            model = _load_aida_embed(config.method.checkpoint)
        elif is_aida_attnagg:
            model = _load_aida_attnagg(config.method.checkpoint)
        elif is_aida_gru:
            model = _load_aida_gru(config.method.checkpoint)
        else:
            model = _load_aida_stackattn(config.method.checkpoint)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        aida_cache_path = config.method.aida_cache_path or ckpt_config["aida_cache_path"]
        _aida_cache = load_aida_feature_cache(Path(aida_cache_path))
        aida_extra_cache_path = config.method.aida_extra_cache_path or ckpt_config.get("aida_extra_cache_path", "")
        if aida_extra_cache_path:
            _aida_cache = _merge_extra_cache(_aida_cache, aida_extra_cache_path)
    evaluator = EmoTrackEvaluator(
        sample_count=1,
        include_history=config.method.include_history,
        transcript_view=config.dataset.transcript_view,
    )

    # Collect inputs only for pending runs
    run_session_map: list[tuple[tuple[str, str], int, ConversationSessionArtifact]] = []
    artifacts: dict[tuple[str, str], ConversationArtifact] = {}
    pending_set = set(pending_keys)

    for (trend_name, run_id), run_rows in grouped.items():
        if (trend_name, run_id) not in pending_set:
            continue
        artifact = load_conversation_artifact(run_rows[0].source_path)
        artifacts[(trend_name, run_id)] = artifact
        session_by_id = {session.session_id: session for session in artifact.sessions}
        session_pos_by_id = {
            session.session_id: idx for idx, session in enumerate(artifact.sessions)
        }
        for row in run_rows:
            session = session_by_id.get(row.session_index)
            if session is None:
                raise KeyError(
                    f"Missing conversation session {row.run_id}::s{row.session_index} in artifact "
                    f"{run_rows[0].source_path}",
                )
            idx = session_pos_by_id[row.session_index]
            run_session_map.append(((trend_name, run_id), idx, session))

    # Batched inference (AIDA family)
    run_ids_and_sessions = [(run_id, session.session_id) for ((_tn, run_id), _idx, session) in run_session_map]
    if is_aida_embed_hybrid:
        from EmoTrack.train.aida.model import predict_from_cache_hybrid as _predict_hybrid
        _embed_dir = Path(config.method.embed_cache_dir or ckpt_config.get("embed_cache_dir", "out/embed_cache"))
        all_predictions = _predict_hybrid(model, _aida_cache, _embed_dir, run_ids_and_sessions)
    elif is_aida_embed_attnagg:
        from EmoTrack.train.aida.model import predict_from_cache_embed_attnagg as _predict_embed
        _embed_dir = Path(config.method.embed_cache_dir or ckpt_config.get("embed_cache_dir", "out/embed_cache"))
        all_predictions = _predict_embed(model, _aida_cache, _embed_dir, run_ids_and_sessions)
    elif is_aida_attnagg:
        all_predictions = _predict_aida_attnagg(model, _aida_cache, run_ids_and_sessions)
    elif is_aida_gru:
        all_predictions = _predict_aida_gru(model, _aida_cache, run_ids_and_sessions)
    else:
        all_predictions = _predict_aida_stackattn(model, _aida_cache, run_ids_and_sessions)

    # Reassemble per-run scores
    run_sessions: dict[tuple[str, str], list[DetectorSessionScore]] = {}
    for (run_key, _idx, session), predicted_items in zip(run_session_map, all_predictions):
        run_sessions.setdefault(run_key, []).append(
            evaluator.score_session_embedding(
                session=session,
                predicted_items=predicted_items,
            )
        )

    for run_key, session_scores in run_sessions.items():
        artifact = artifacts[run_key]
        target_errors = [
            s.absolute_error_to_target for s in session_scores
            if s.absolute_error_to_target is not None
        ]
        latent_errors = [
            s.absolute_error_to_latent for s in session_scores
            if s.absolute_error_to_latent is not None
        ]
        self_report_errors = [
            s.absolute_error_to_self_report for s in session_scores
            if s.absolute_error_to_self_report is not None
        ]
        score = DetectorRunScore(
            method=method_name,
            scored_at=datetime.now(timezone.utc).isoformat(),
            scoring_model=config.method.checkpoint,
            source_run_path=str(artifact.source_dir),
            source_run_id=artifact.run_id,
            transcript_view=config.dataset.transcript_view,
            sample_count=1,
            session_count=len(session_scores),
            sessions=session_scores,
            dataset_name="conversation",
            trend_name=artifact.trend_name,
            profile_id=artifact.profile_id,
            subject_id=artifact.profile_id,
            split="test",
            target_label="phq8_total",
            mean_absolute_error_to_target=(sum(target_errors) / len(target_errors)) if target_errors else None,
            mean_absolute_error_to_latent=(sum(latent_errors) / len(latent_errors)) if latent_errors else None,
            mean_absolute_error_to_self_report=(
                (sum(self_report_errors) / len(self_report_errors)) if self_report_errors else None
            ),
        )
        write_result_detector_artifact(score, seed=seed, namespace=_artifact_namespace(config))
        run_scores.append(score)
    return run_scores


async def run_emotrack_daic(
    *,
    test_rows: list[SessionExampleRow],
    out_root: Path,
    config: EvaluationConfig,
    llm_workers: list[PreparedLLMWorker] | None,
    overwrite: bool = False,
    use_vllm: bool = False,
) -> list[DetectorRunScore]:
    seed = config.method.inference_seed
    run_scores: list[DetectorRunScore] = []
    artifacts = [_build_daic_artifact(row) for row in test_rows]
    if config.method.mode == "api":
        if not llm_workers:
            raise RuntimeError("EmoTrack api evaluation requires an llm endpoint.")
        clients = [
            OpenAIChatTextClient(
                client=make_async_openai_client(
                    base_url=worker.base_url,
                    api_key=worker.api_key,
                    allow_missing_api_key=config.llm.backend == "local",
                ),
                model_name=worker.model_name,
            )
            for worker in llm_workers
        ]
        evaluators = [
            EmoTrackEvaluator(
                sample_count=1,
                include_history=False,
                transcript_view=config.dataset.transcript_view,
                prompt_template_path=config.method.prompt_template or None,
            )
            for _ in llm_workers
        ]
        semaphores = [asyncio.Semaphore(max(1, config.method.eval_concurrency)) for _ in llm_workers]

        async def _score_one(index: int, artifact: ConversationArtifact) -> None:
            worker_index = index % len(llm_workers)
            async with semaphores[worker_index]:
                score = await evaluators[worker_index].score_artifact(
                    artifact,
                    clients[worker_index],
                    scoring_model=llm_workers[worker_index].model_name,
                )
            score.split = "test"
            score.dataset_name = config.dataset.name
            write_result_detector_artifact(score, seed=seed, namespace=_artifact_namespace(config))
            run_scores.append(score)

        await asyncio.gather(*(_score_one(index, artifact) for index, artifact in enumerate(artifacts)))
        return sorted(run_scores, key=lambda score: score.source_run_id)

    if not config.method.checkpoint:
        raise ValueError("method.checkpoint is required for emotrack mode=embedding")

    import logging as _logging

    import torch

    _logger = _logging.getLogger(__name__)

    ckpt_path = Path(config.method.checkpoint)
    is_aida_attnagg = (ckpt_path / "aida_attnagg_config.json").exists()
    is_aida_gru = (ckpt_path / "aida_gru_config.json").exists()
    is_aida_stackattn = (ckpt_path / "aida_stackattn_config.json").exists()
    is_aida_embed_attnagg = (ckpt_path / "aida_embed_attnagg_config.json").exists()
    is_aida_embed_hybrid = (ckpt_path / "aida_embed_hybrid_config.json").exists()
    if is_aida_embed_hybrid:
        from EmoTrack.train.aida.model import _HYBRID_CONFIG
        method_name = "emotrack"
        config_filename = _HYBRID_CONFIG
    elif is_aida_embed_attnagg:
        method_name = "emotrack_hybrid_embed_attnagg"
        config_filename = "aida_embed_attnagg_config.json"
    elif is_aida_attnagg:
        method_name = "emotrack_aida_attnagg"
        config_filename = "aida_attnagg_config.json"
    elif is_aida_gru:
        method_name = "emotrack_aida_gru"
        config_filename = "aida_gru_config.json"
    elif is_aida_stackattn:
        method_name = "emotrack_aida_stackattn"
        config_filename = "aida_stackattn_config.json"
    else:
        raise ValueError(f"Unrecognised AIDA checkpoint at {ckpt_path}")

    # Override method_name from saved checkpoint config if available
    _ckpt_cfg_path = ckpt_path / config_filename
    if _ckpt_cfg_path.is_file():
        _ckpt_cfg = json.loads(_ckpt_cfg_path.read_text(encoding="utf-8"))
        if "result_name" in _ckpt_cfg:
            method_name = _ckpt_cfg["result_name"]

    # Scan existing results — skip completed participants
    pending_artifacts: list[ConversationArtifact] = []
    for artifact in artifacts:
        if overwrite:
            pending_artifacts.append(artifact)
            continue
        existing_path = detector_output_path(
            artifact.source_dir,
            method_name=method_name,
            transcript_view=config.dataset.transcript_view,
            seed=seed,
            namespace=_artifact_namespace(config),
        )
        if existing_path.is_file():
            run_scores.append(load_detector_artifact(existing_path))
        else:
            pending_artifacts.append(artifact)

    _logger.info(
        "%s daic_woz scan: %d completed, %d pending",
        method_name, len(run_scores), len(pending_artifacts),
    )
    if pending_artifacts:
        _logger.info("Pending participants: %s", [a.run_id for a in pending_artifacts])

    if not pending_artifacts:
        return sorted(run_scores, key=lambda s: s.source_run_id)

    ckpt_config = json.loads((ckpt_path / config_filename).read_text(encoding="utf-8"))
    if True:  # AIDA family only
        from EmoTrack.train.aida.model import (
            load_checkpoint_attnagg as _load_aida_attnagg,
            load_checkpoint_gru as _load_aida_gru,
            load_checkpoint_stackattn as _load_aida_stackattn,
            predict_from_cache_attnagg as _predict_aida_attnagg,
            predict_from_cache_gru as _predict_aida_gru,
            predict_from_cache_stackattn as _predict_aida_stackattn,
        )
        from EmoTrack.train.aida.dataset import load_aida_feature_cache, _merge_extra_cache
        if is_aida_embed_hybrid:
            from EmoTrack.train.aida.model import load_checkpoint_hybrid as _load_hybrid
            model = _load_hybrid(config.method.checkpoint)
        elif is_aida_embed_attnagg:
            from EmoTrack.train.aida.model import load_checkpoint_embed_attnagg as _load_aida_embed
            model = _load_aida_embed(config.method.checkpoint)
        elif is_aida_attnagg:
            model = _load_aida_attnagg(config.method.checkpoint)
        elif is_aida_gru:
            model = _load_aida_gru(config.method.checkpoint)
        else:
            model = _load_aida_stackattn(config.method.checkpoint)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        aida_cache_path = config.method.aida_cache_path or ckpt_config["aida_cache_path"]
        _aida_cache = load_aida_feature_cache(Path(aida_cache_path))
        aida_extra_cache_path = config.method.aida_extra_cache_path or ckpt_config.get("aida_extra_cache_path", "")
        if aida_extra_cache_path:
            _aida_cache = _merge_extra_cache(_aida_cache, aida_extra_cache_path)
    evaluator = EmoTrackEvaluator(sample_count=1, include_history=False, transcript_view=config.dataset.transcript_view)

    # DAIC is single-session — predict from cache
    run_ids_and_sessions = [(a.run_id, 1) for a in pending_artifacts]
    if is_aida_embed_hybrid:
        from EmoTrack.train.aida.model import predict_from_cache_hybrid as _predict_hybrid
        _embed_dir = Path(config.method.embed_cache_dir or ckpt_config.get("embed_cache_dir", "out/embed_cache"))
        all_predictions = _predict_hybrid(model, _aida_cache, _embed_dir, run_ids_and_sessions)
    elif is_aida_embed_attnagg:
        from EmoTrack.train.aida.model import predict_from_cache_embed_attnagg as _predict_embed
        _embed_dir = Path(config.method.embed_cache_dir or ckpt_config.get("embed_cache_dir", "out/embed_cache"))
        all_predictions = _predict_embed(model, _aida_cache, _embed_dir, run_ids_and_sessions)
    elif is_aida_attnagg:
        all_predictions = _predict_aida_attnagg(model, _aida_cache, run_ids_and_sessions)
    elif is_aida_gru:
        all_predictions = _predict_aida_gru(model, _aida_cache, run_ids_and_sessions)
    else:
        all_predictions = _predict_aida_stackattn(model, _aida_cache, run_ids_and_sessions)

    for artifact, predicted_items in zip(pending_artifacts, all_predictions):
        session_score = evaluator.score_session_embedding(session=artifact.sessions[0], predicted_items=predicted_items)
        score = DetectorRunScore(
            method=method_name,
            scored_at=datetime.now(timezone.utc).isoformat(),
            scoring_model=config.method.checkpoint,
            source_run_path=str(artifact.source_dir),
            source_run_id=artifact.run_id,
            transcript_view=config.dataset.transcript_view,
            sample_count=1,
            session_count=1,
            sessions=[session_score],
            dataset_name=config.dataset.name,
            trend_name=artifact.trend_name,
            profile_id=artifact.profile_id,
            subject_id=artifact.profile_id,
            split="test",
            target_label="phq8_total",
            mean_absolute_error_to_target=session_score.absolute_error_to_target,
            mean_absolute_error_to_latent=session_score.absolute_error_to_latent,
            mean_absolute_error_to_self_report=None,
        )
        write_result_detector_artifact(score, seed=seed, namespace=_artifact_namespace(config))
        run_scores.append(score)
    return run_scores


def _root_from_group(trend_name: str, run_id: str, rows: list[SessionExampleRow]) -> Path:
    for row in rows:
        if row.trend_name == trend_name and row.run_id == run_id:
            return row.source_path
    raise FileNotFoundError(f"Cannot resolve source path for {trend_name}/{run_id}")


def _build_daic_artifact(row: SessionExampleRow) -> ConversationArtifact:
    artifact = load_conversation_artifact(row.source_path)
    sessions = [
        ConversationSessionArtifact(
            session_id=session.session_id,
            turns=session.turns,
            latent_item_scores=dict(session.latent_item_scores),
            latent_total=session.latent_total,
            self_report_item_scores={},
            self_report_total=None,
        )
        for session in artifact.sessions
    ]
    return ConversationArtifact(
        run_id=artifact.run_id,
        trend_name="daic_woz",
        profile_id=artifact.profile_id,
        source_dir=artifact.source_dir,
        transcript_path=artifact.transcript_path,
        self_report_path=None,
        ground_truth_path=artifact.ground_truth_path,
        sessions=sessions,
    )
