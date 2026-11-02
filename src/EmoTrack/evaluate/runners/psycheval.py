from __future__ import annotations

import json
from pathlib import Path

from EmoTrack.evaluate.core.config import EvaluationConfig
from EmoTrack.evaluate.core.context import PreparedLLMWorker
from EmoTrack.evaluate.core.scores import DetectorRunScore, SessionExampleRow, build_run_score, group_rows_by_run
from EmoTrack.evaluate.runners.common import _artifact_namespace, _load_existing_run_score
from EmoTrack.evaluate.utils import load_conversation_artifact, run_dynamic_task_queue, write_json, write_result_detector_artifact
from EmoTrack.utils.paths import REPO_ROOT


async def run_psycheval(
    *,
    dataset_name: str,
    test_rows: list[SessionExampleRow],
    out_root: Path,
    config: EvaluationConfig,
    llm_workers: list[PreparedLLMWorker] | None,
) -> list[DetectorRunScore]:
    if not llm_workers:
        raise RuntimeError("PsychEval evaluation requires an llm endpoint.")
    if dataset_name == "conversation":
        return await run_psycheval_conversation(
            test_rows=test_rows,
            out_root=out_root,
            config=config,
            llm_workers=llm_workers,
        )
    return await _run_psycheval_daic(
        test_rows=test_rows,
        out_root=out_root,
        config=config,
        llm_workers=llm_workers,
    )


async def run_psycheval_conversation(
    *,
    test_rows: list[SessionExampleRow],
    out_root: Path,
    config: EvaluationConfig,
    llm_workers: list[PreparedLLMWorker],
) -> list[DetectorRunScore]:
    from EmoTrack.evaluate.methods.psycheval import evaluate_eval_ready_case, extract_session_scores

    seed = config.method.inference_seed
    run_scores_by_key: dict[tuple[str, str], DetectorRunScore] = {}
    grouped_items = list(group_rows_by_run(test_rows).items())
    pending_items: list[tuple[tuple[str, str], list[SessionExampleRow]]] = []
    for key, run_rows in grouped_items:
        existing_score = _load_existing_run_score(
            method_name="psycheval",
            transcript_view=config.dataset.transcript_view,
            run_rows=run_rows,
            seed=seed,
            namespace=_artifact_namespace(config),
        )
        if existing_score is not None:
            run_scores_by_key[key] = existing_score
            continue
        pending_items.append((key, run_rows))

    async def _score_one(
        worker: PreparedLLMWorker,
        item: tuple[tuple[str, str], list[SessionExampleRow]],
    ) -> None:
        (trend_name, run_id), run_rows = item
        eval_ready_path = ensure_conversation_eval_ready_path(row=run_rows[0])
        raw_output_dir = out_root / "raw_psycheval" / trend_name / run_id
        results = await evaluate_eval_ready_case(
            eval_ready_path,
            raw_output_dir,
            sample_count=config.method.sample_count,
            base_url=worker.base_url,
            model=worker.model_name,
            api_key=worker.api_key,
        )
        session_rows = extract_session_scores(results, source_path=eval_ready_path)
        predictions = {
            f"{run_id}::s{int(session_row['session_index'])}": float(session_row["predicted_total"])
            for session_row in session_rows
        }
        predicted_items = {
            f"{run_id}::s{int(session_row['session_index'])}": dict(session_row["predicted_item_scores"])
            for session_row in session_rows
        }
        score = build_run_score(
            method_name="psycheval",
            scoring_model=worker.model_name,
            transcript_view=config.dataset.transcript_view,
            rows=run_rows,
            predictions_by_row_key=predictions,
            predicted_items_by_row_key=predicted_items,
        )
        write_result_detector_artifact(score, seed=seed, namespace=_artifact_namespace(config))
        run_scores_by_key[(trend_name, run_id)] = score

    if pending_items:
        await run_dynamic_task_queue(
            pending_items,
            workers=llm_workers,
            per_worker_concurrency=max(1, config.method.eval_concurrency),
            process_item=_score_one,
        )
    return [run_scores_by_key[key] for key in sorted(run_scores_by_key)]


async def _run_psycheval_daic(
    *,
    test_rows: list[SessionExampleRow],
    out_root: Path,
    config: EvaluationConfig,
    llm_workers: list[PreparedLLMWorker],
) -> list[DetectorRunScore]:
    from EmoTrack.evaluate.methods.psycheval import evaluate_eval_ready_case, extract_session_scores

    seed = config.method.inference_seed
    run_scores_by_id: dict[str, DetectorRunScore] = {}

    async def _score_one(worker: PreparedLLMWorker, row: SessionExampleRow) -> None:
        eval_ready_path = _write_eval_ready_stub(row=row, out_root=out_root)
        results = await evaluate_eval_ready_case(
            eval_ready_path,
            out_root / "raw_psycheval" / row.subject_id,
            sample_count=config.method.sample_count,
            base_url=worker.base_url,
            model=worker.model_name,
            api_key=worker.api_key,
        )
        session_rows = extract_session_scores(results, source_path=eval_ready_path)
        predictions = {
            f"{row.run_id}::s{int(session_row['session_index'])}": float(session_row["predicted_total"])
            for session_row in session_rows
        }
        predicted_items = {
            f"{row.run_id}::s{int(session_row['session_index'])}": dict(session_row["predicted_item_scores"])
            for session_row in session_rows
        }
        score = build_run_score(
            method_name="psycheval",
            scoring_model=worker.model_name,
            transcript_view=config.dataset.transcript_view,
            rows=[row],
            predictions_by_row_key=predictions,
            predicted_items_by_row_key=predicted_items,
        )
        write_result_detector_artifact(score, seed=seed, namespace=_artifact_namespace(config))
        run_scores_by_id[row.run_id] = score

    await run_dynamic_task_queue(
        test_rows,
        workers=llm_workers,
        per_worker_concurrency=max(1, config.method.eval_concurrency),
        process_item=_score_one,
    )
    return [run_scores_by_id[row.run_id] for row in sorted(test_rows, key=lambda item: item.run_id)]


def _write_eval_ready_stub(*, row: SessionExampleRow, out_root: Path) -> Path:
    payload = {
        "client_info": {},
        "sessions": [
            {
                "session_number": 1,
                "dialogue": row.transcript_full,
            }
        ],
    }
    path = out_root / "eval_ready" / f"{row.subject_id}.json"
    return write_json(path, payload)


def ensure_conversation_eval_ready_path(*, row: SessionExampleRow) -> Path:
    if row.eval_ready_path is not None and row.eval_ready_path.is_file():
        return row.eval_ready_path.resolve()

    target_run_dir = _resolve_conversation_eval_ready_run_dir(row.source_path)
    candidate = target_run_dir / "eval_ready_case.json"
    if candidate.is_file():
        return candidate.resolve()

    artifact = load_conversation_artifact(row.source_path)
    payload = {
        "client_id": artifact.run_id,
        "client_info": _load_conversation_client_info(profile_id=row.profile_id),
        "sessions": [
            {
                "session_number": session.session_id,
                "session_dialogue": [
                    {"role": turn.speaker, "text": turn.text}
                    for turn in session.turns
                ],
            }
            for session in artifact.sessions
        ],
    }
    return write_json(candidate, payload)


def _resolve_conversation_eval_ready_run_dir(run_dir: Path) -> Path:
    resolved_run_dir = Path(run_dir).expanduser().resolve()
    mapped_out_dir = _map_result_run_dir_to_out_run_dir(resolved_run_dir)
    if mapped_out_dir is not None:
        return mapped_out_dir
    return resolved_run_dir


def _map_result_run_dir_to_out_run_dir(run_dir: Path) -> Path | None:
    try:
        relative = run_dir.resolve().relative_to((REPO_ROOT / "result").resolve())
    except ValueError:
        return None
    return (REPO_ROOT / "out" / relative).resolve()


def _load_conversation_client_info(*, profile_id: str) -> dict[str, object]:
    for candidate in _tracked_conversation_profile_candidates(profile_id):
        resolved = candidate.resolve()
        if not resolved.is_file():
            continue
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        return _conversation_client_info_from_profile(payload)
    raise FileNotFoundError(
        f"Could not recover client profile for {profile_id} while rebuilding eval_ready_case.json. "
        "Checked tracked source profiles under profiles/<theoretical>/<client_id>.json."
    )


def _tracked_conversation_profile_candidates(profile_id: str) -> list[Path]:
    theoretical, client_id = _split_profile_id(profile_id)
    if not theoretical or not client_id:
        return []
    candidates = [(REPO_ROOT / "profiles" / theoretical / f"{client_id}.json").resolve()]
    if client_id.isdigit():
        normalized = str(int(client_id))
        if normalized != client_id:
            candidates.append((REPO_ROOT / "profiles" / theoretical / f"{normalized}.json").resolve())
    return candidates


def _split_profile_id(profile_id: str) -> tuple[str, str]:
    parts = [part.strip() for part in str(profile_id or "").split("_") if part.strip()]
    if len(parts) == 2:
        return parts[0], parts[1]
    if len(parts) == 3 and parts[0].isdigit():
        return parts[1], parts[2]
    return "", ""


def _conversation_client_info_from_profile(payload: dict[str, object]) -> dict[str, object]:
    client_info = payload.get("client_info")
    if isinstance(client_info, dict) and client_info:
        return dict(client_info)

    intake_profile = payload.get("intake_profile")
    if isinstance(intake_profile, dict) and intake_profile:
        return dict(intake_profile)

    portrait = payload.get("portrait")
    report = payload.get("report")
    if not isinstance(portrait, dict):
        portrait = {}
    if not isinstance(report, dict):
        report = {}

    marital_status = _read_profile_str(portrait, "marital_status")
    if not marital_status:
        marital_status = _read_profile_str(portrait, "martial_status")

    return {
        "static_traits": {
            "age": _read_profile_str(portrait, "age"),
            "name": _read_profile_str(portrait, "name"),
            "gender": _read_profile_str(portrait, "gender"),
            "occupation": _read_profile_str(portrait, "occupation"),
            "educational_background": _read_profile_str(portrait, "educational_background"),
            "marital_status": marital_status,
            "family_status": _read_profile_str(portrait, "family_status"),
            "social_status": _read_profile_str(portrait, "social_status"),
            "medical_history": _read_profile_str(portrait, "medical_history"),
            "language_features": _read_profile_str(portrait, "language_features"),
        },
        "main_problem": _read_profile_str(report, "main_problem") or _read_profile_str(portrait, "symptoms"),
        "topic": _read_profile_str(report, "topic") or _read_profile_str(portrait, "primary_topic"),
        "core_demands": _read_profile_str(report, "core_demands"),
        "growth_experiences": _read_profile_list(report, "growth_experiences"),
        "core_beliefs": _read_profile_list(report, "core_beliefs"),
        "special_situations": _read_profile_list(report, "special_situations"),
    }


def _read_profile_str(mapping: dict[str, object], key: str, default: str = "") -> str:
    value = mapping.get(key, default)
    return str(value).strip() if value is not None else ""


def _read_profile_list(mapping: dict[str, object], key: str) -> list[object]:
    value = mapping.get(key, [])
    return list(value) if isinstance(value, list) else []
