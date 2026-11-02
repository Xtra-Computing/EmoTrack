from __future__ import annotations

import asyncio
import hashlib
import json
import os
from collections import defaultdict
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, TypeVar

from openai import AsyncOpenAI

from EmoTrack.behavior_cue_mapping import PHQ8_RUNTIME_CUE_BANK_PATH
from EmoTrack.evaluate.core.artifacts import (
    ArtifactNamespaceLike,
    detector_output_path as _detector_output_path,
)
from EmoTrack.evaluate.core.phq8 import DEFAULT_PHQ8_SYMPTOM_ORDER
from EmoTrack.evaluate.core.scores import (
    ConversationArtifact,
    ConversationSessionArtifact,
    DetectorItemScore,
    DetectorRunScore,
    DetectorSessionScore,
    SessionExampleRow,
    TranscriptTurn,
    TranscriptView,
)
from EmoTrack.utils.paths import REPO_ROOT


_WorkerT = TypeVar("_WorkerT")
_ItemT = TypeVar("_ItemT")

_GROUND_TRUTH_FILENAME = "ground_truth.json"


class OpenAIChatTextClient:
    def __init__(self, *, client: AsyncOpenAI, model_name: str) -> None:
        self._client = client
        self._model_name = model_name

    async def chat_text(
        self,
        *,
        messages,
        temperature: float = 0.0,
        max_tokens: int = 512,
        response_format=None,
        **_: object,
    ) -> str:
        response = await self._client.chat.completions.create(
            model=self._model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )
        return response.choices[0].message.content or ""


def resolve_api_key(preferred_env: str = "") -> str:
    return resolve_api_key_with_fallback(preferred_env)


def resolve_api_key_with_fallback(
    preferred_env: str = "",
    *,
    explicit_value: str = "",
    allow_missing: bool = False,
) -> str:
    explicit = str(explicit_value or "").strip()
    if explicit:
        return explicit
    candidates: list[str] = []
    if preferred_env:
        candidates.append(preferred_env)
    candidates.extend(
        [
            "CHAT_API_KEY",
            "EMOTRACK_MODEL_API_KEY",
            "LLM_API_KEY",
            "OPENAI_API_KEY",
            "NEBIUS_API_KEY",
        ]
    )
    for name in candidates:
        value = os.getenv(name, "").strip()
        if value:
            return value
    if allow_missing:
        return "EMPTY"
    raise RuntimeError(
        "No API key found. Set one of CHAT_API_KEY, EMOTRACK_MODEL_API_KEY, "
        "LLM_API_KEY, OPENAI_API_KEY, or NEBIUS_API_KEY."
    )


def make_async_openai_client(
    *,
    base_url: str,
    api_key: str = "",
    preferred_key_env: str = "",
    allow_missing_api_key: bool = False,
) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=resolve_api_key_with_fallback(
            preferred_key_env,
            explicit_value=api_key,
            allow_missing=allow_missing_api_key,
        ),
        base_url=base_url,
    )


async def run_dynamic_task_queue(
    items: list[_ItemT],
    *,
    workers: list[_WorkerT],
    per_worker_concurrency: int,
    process_item: Callable[[_WorkerT, _ItemT], Awaitable[None]],
    desc: str = "",
) -> None:
    if not workers:
        raise ValueError("run_dynamic_task_queue requires at least one worker")
    if per_worker_concurrency <= 0:
        raise ValueError("run_dynamic_task_queue requires per_worker_concurrency >= 1")
    if not items:
        return

    from tqdm import tqdm

    pbar = tqdm(total=len(items), desc=desc or "Processing", unit="item")

    queue: asyncio.Queue[_ItemT] = asyncio.Queue()
    for item in items:
        queue.put_nowait(item)

    async def _worker_loop(worker: _WorkerT) -> None:
        while True:
            try:
                item = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                await process_item(worker, item)
            finally:
                queue.task_done()
                pbar.update(1)

    tasks: list[asyncio.Task[None]] = []
    for worker in workers:
        for _ in range(per_worker_concurrency):
            tasks.append(asyncio.create_task(_worker_loop(worker)))
    await asyncio.gather(*tasks)
    pbar.close()


def load_conversation_artifact(path: str | Path) -> ConversationArtifact:
    artifact_dir = _resolve_artifact_dir(path)
    common_dir = artifact_dir / "common"
    transcript_path = common_dir / "transcript.jsonl"
    if not transcript_path.is_file():
        raise FileNotFoundError(f"Missing common/transcript.jsonl in {artifact_dir}")

    self_report_path = common_dir / "self_report.json"
    if not self_report_path.is_file():
        self_report_path = None

    ground_truth_path = common_dir / _GROUND_TRUTH_FILENAME
    if not ground_truth_path.is_file():
        ground_truth_path = None

    rows = _load_transcript_rows(transcript_path)
    self_report_by_session = _load_self_report_by_session(self_report_path)
    ground_truth_by_session = _load_ground_truth_by_session(ground_truth_path)

    turns_by_session: dict[int, list[TranscriptTurn]] = defaultdict(list)
    latent_by_session: dict[int, tuple[dict[str, float], float]] = dict(ground_truth_by_session)
    for row in rows:
        session_id = int(_read_session_id(row))
        turn_id = int(row.get("turn_id") or row.get("message_index") or len(turns_by_session[session_id]) + 1)
        speaker = _normalize_speaker(row.get("speaker") or row.get("role") or "")
        text = str(row.get("text") or row.get("content") or "").strip()
        if not speaker or not text:
            continue
        turns_by_session[session_id].append(TranscriptTurn(turn_id=turn_id, speaker=speaker, text=text))
        if session_id not in latent_by_session:
            latent_by_session[session_id] = _extract_latent_scores(row)

    sessions: list[ConversationSessionArtifact] = []
    for session_id in sorted(turns_by_session):
        latent_item_scores, latent_total = latent_by_session.get(
            session_id,
            ({symptom_name: 0.0 for symptom_name in _load_phq8_symptom_order()}, 0.0),
        )
        self_report_total, self_report_item_scores = self_report_by_session.get(session_id, (None, {}))
        sessions.append(
            ConversationSessionArtifact(
                session_id=session_id,
                turns=sorted(turns_by_session[session_id], key=lambda turn: turn.turn_id),
                latent_item_scores=latent_item_scores,
                latent_total=latent_total,
                self_report_item_scores=self_report_item_scores,
                self_report_total=self_report_total,
            )
        )

    if not sessions:
        raise ValueError(f"No transcript sessions found in {transcript_path}")

    run_id = artifact_dir.name
    trend_name, profile_id = _derive_conversation_run_metadata(
        artifact_dir=artifact_dir,
        run_id=run_id,
    )

    return ConversationArtifact(
        run_id=run_id,
        trend_name=trend_name,
        profile_id=profile_id,
        source_dir=artifact_dir,
        transcript_path=transcript_path,
        self_report_path=self_report_path,
        ground_truth_path=ground_truth_path,
        sessions=sessions,
    )


def render_session_transcript(
    session: ConversationSessionArtifact,
    *,
    transcript_view: TranscriptView,
) -> str:
    lines: list[str] = []
    for turn in session.turns:
        if transcript_view == "client-only" and turn.speaker != "Client":
            continue
        lines.append(f"{turn.speaker}: {turn.text}")
    return "\n".join(lines).strip()


def result_detector_path_for_row(
    row: SessionExampleRow,
    *,
    method_name: str,
    transcript_view: str,
    seed: int = 0,
    namespace: ArtifactNamespaceLike = None,
) -> Path:
    if row.dataset_name in {"conversation", "daic_woz", "edaic"}:
        artifact_dir = _resolve_artifact_dir(row.source_path)
    else:
        raise ValueError(f"Unsupported detector result dataset: {row.dataset_name!r}")
    return _detector_output_path(
        artifact_dir,
        method_name=method_name,
        transcript_view=transcript_view,
        seed=seed,
        namespace=namespace,
    )


def eval_output_root(*, dataset_name: str, experiment_tag: str, method_name: str, seed: int = 0) -> Path:
    return REPO_ROOT / "out" / "eval" / dataset_name / experiment_tag / method_name / f"seed{seed}"


def eval_dataset_result_root(*, dataset_name: str) -> Path:
    return REPO_ROOT / "result" / dataset_name


def eval_result_root(
    *,
    dataset_name: str,
    experiment_tag: str,
    method_name: str,
    model_name: str = "",
    mode: str = "api",
    checkpoint: str = "",
) -> Path:
    safe_tag = _slug_component(experiment_tag) or "default"
    safe_method = _slug_component(method_name) or "method"
    return eval_dataset_result_root(dataset_name=dataset_name) / safe_tag / safe_method


def eval_comparison_result_root(*, dataset_name: str, experiment_tag: str) -> Path:
    safe_tag = _slug_component(experiment_tag) or "default"
    return eval_dataset_result_root(dataset_name=dataset_name) / f"comparison_{safe_tag}"


def eval_run_artifact_dir(
    *,
    dataset_name: str,
    experiment_tag: str,
    method_name: str,
    trend_name: str,
    run_id: str,
    seed: int = 0,
) -> Path:
    return eval_output_root(
        dataset_name=dataset_name,
        experiment_tag=experiment_tag,
        method_name=method_name,
        seed=seed,
    ) / "runs" / trend_name / run_id


def result_detector_artifact_dir(score: DetectorRunScore) -> Path:
    if score.dataset_name in {"conversation", "daic_woz", "edaic"}:
        source_path = Path(score.source_run_path).expanduser().resolve()
        return source_path.parent if source_path.is_file() else source_path
    raise ValueError(f"Unsupported detector result dataset: {score.dataset_name!r}")


def write_result_detector_artifact(
    score: DetectorRunScore,
    *,
    seed: int = 0,
    namespace: ArtifactNamespaceLike = None,
) -> Path:
    return write_detector_artifact(
        score,
        artifact_dir=result_detector_artifact_dir(score),
        seed=seed,
        namespace=namespace,
    )


def _result_namespace(
    *,
    method_name: str,
    model_name: str,
    mode: str,
    checkpoint: str,
) -> str:
    safe_method = _slug_component(method_name) or "method"
    if safe_method == "emotrack" and mode == "embedding":
        checkpoint_label = _slug_component(Path(checkpoint).stem) or "checkpoint"
        return f"{safe_method}_{checkpoint_label}"
    model_label = _model_result_label(model_name)
    if model_label:
        return f"{safe_method}_{model_label}"
    return safe_method


def _model_result_label(model_name: str) -> str:
    raw = str(model_name or "").strip()
    if not raw:
        return ""
    if raw == "Qwen/Qwen3.5-35B-A3B":
        return "qwen_3.5_35B"

    provider = ""
    name = raw
    if "/" in raw:
        provider, name = raw.split("/", 1)
    provider = _slug_component(provider)
    name = _slug_component(name)
    if provider and name.startswith(f"{provider}_"):
        name = name[len(provider) + 1 :]
    if name.endswith("_a3b"):
        name = name[: -len("_a3b")]
    if provider and name:
        return f"{provider}_{name}"
    return provider or name


def _slug_component(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    chars: list[str] = []
    previous_is_sep = False
    for char in text:
        if char.isalnum() or char == ".":
            chars.append(char.lower() if char.isalpha() else char)
            previous_is_sep = False
            continue
        if previous_is_sep:
            continue
        chars.append("_")
        previous_is_sep = True
    return "".join(chars).strip("_")


def write_json(path: str | Path, payload: Any) -> Path:
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps(_to_jsonable(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return resolved


def stable_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_detector_artifact(
    score: DetectorRunScore,
    *,
    artifact_dir: str | Path,
    seed: int = 0,
    namespace: ArtifactNamespaceLike = None,
) -> Path:
    resolved_dir = Path(artifact_dir).expanduser().resolve()
    resolved_dir.mkdir(parents=True, exist_ok=True)
    output_path = _detector_output_path(
        resolved_dir,
        method_name=score.method,
        transcript_view=score.transcript_view,
        seed=seed,
        namespace=namespace,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_to_jsonable(score), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def load_detector_artifact(path: str | Path) -> DetectorRunScore:
    resolved = Path(path).expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Detector artifact must be a JSON object: {resolved}")

    raw_sessions = payload.get("sessions", [])
    if not isinstance(raw_sessions, list):
        raise ValueError(f"Detector artifact sessions must be a list: {resolved}")

    sessions: list[DetectorSessionScore] = []
    for raw_session in raw_sessions:
        if not isinstance(raw_session, dict):
            raise ValueError(f"Detector artifact contains a non-object session row: {resolved}")
        raw_items = raw_session.get("items")
        items = None
        if isinstance(raw_items, list):
            items = [
                DetectorItemScore(
                    question_id=int(item.get("question_id", 0)),
                    symptom_name=str(item.get("symptom_name", "")),
                    question_text=str(item.get("question_text", "")),
                    score=float(item.get("score", 0.0)),
                    score_std=float(item.get("score_std", 0.0)),
                    sample_scores=[int(value) for value in item.get("sample_scores", [])],
                    option_label=str(item.get("option_label", "")),
                    evidence=str(item.get("evidence", "")),
                )
                for item in raw_items
                if isinstance(item, dict)
            ]
        sessions.append(
            DetectorSessionScore(
                session_index=int(raw_session["session_index"]),
                transcript_view=str(raw_session.get("transcript_view", "")),
                sample_count=int(raw_session.get("sample_count", 1)),
                predicted_total=float(raw_session["predicted_total"]),
                predicted_item_scores=(
                    dict(raw_session["predicted_item_scores"])
                    if isinstance(raw_session.get("predicted_item_scores"), dict)
                    else None
                ),
                target_item_scores=(
                    dict(raw_session["target_item_scores"])
                    if isinstance(raw_session.get("target_item_scores"), dict)
                    else None
                ),
                latent_item_scores=(
                    dict(raw_session["latent_item_scores"])
                    if isinstance(raw_session.get("latent_item_scores"), dict)
                    else None
                ),
                item_mean_absolute_error_to_target=(
                    float(raw_session["item_mean_absolute_error_to_target"])
                    if raw_session.get("item_mean_absolute_error_to_target") is not None
                    else None
                ),
                item_mean_absolute_error_to_latent=(
                    float(raw_session["item_mean_absolute_error_to_latent"])
                    if raw_session.get("item_mean_absolute_error_to_latent") is not None
                    else None
                ),
                per_item_absolute_error_to_target=(
                    dict(raw_session["per_item_absolute_error_to_target"])
                    if isinstance(raw_session.get("per_item_absolute_error_to_target"), dict)
                    else None
                ),
                per_item_absolute_error_to_latent=(
                    dict(raw_session["per_item_absolute_error_to_latent"])
                    if isinstance(raw_session.get("per_item_absolute_error_to_latent"), dict)
                    else None
                ),
                target_total=(
                    float(raw_session["target_total"]) if raw_session.get("target_total") is not None else None
                ),
                latent_total=(
                    float(raw_session["latent_total"]) if raw_session.get("latent_total") is not None else None
                ),
                self_report_total=(
                    float(raw_session["self_report_total"])
                    if raw_session.get("self_report_total") is not None
                    else None
                ),
                absolute_error_to_target=(
                    float(raw_session["absolute_error_to_target"])
                    if raw_session.get("absolute_error_to_target") is not None
                    else None
                ),
                absolute_error_to_latent=(
                    float(raw_session["absolute_error_to_latent"])
                    if raw_session.get("absolute_error_to_latent") is not None
                    else None
                ),
                absolute_error_to_self_report=(
                    float(raw_session["absolute_error_to_self_report"])
                    if raw_session.get("absolute_error_to_self_report") is not None
                    else None
                ),
                items=items,
            )
        )

    return DetectorRunScore(
        method=str(payload.get("method", "")),
        scored_at=str(payload.get("scored_at", "")),
        scoring_model=str(payload.get("scoring_model", "")),
        source_run_path=str(payload.get("source_run_path", "")),
        source_run_id=str(payload.get("source_run_id", "")),
        transcript_view=str(payload.get("transcript_view", "")),
        sample_count=int(payload.get("sample_count", 1)),
        session_count=int(payload.get("session_count", len(sessions))),
        sessions=sessions,
        dataset_name=str(payload.get("dataset_name", "")),
        trend_name=str(payload.get("trend_name", "")),
        profile_id=str(payload.get("profile_id", "")),
        subject_id=str(payload.get("subject_id", "")),
        split=str(payload.get("split", "")),
        target_label=str(payload.get("target_label", "phq8_total")),
        mean_absolute_error_to_target=(
            float(payload["mean_absolute_error_to_target"])
            if payload.get("mean_absolute_error_to_target") is not None
            else None
        ),
        mean_absolute_error_to_latent=(
            float(payload["mean_absolute_error_to_latent"])
            if payload.get("mean_absolute_error_to_latent") is not None
            else None
        ),
        mean_absolute_error_to_self_report=(
            float(payload["mean_absolute_error_to_self_report"])
            if payload.get("mean_absolute_error_to_self_report") is not None
            else None
        ),
    )


def _resolve_artifact_dir(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if resolved.is_file():
        if resolved.name == "transcript.jsonl":
            # transcript.jsonl lives in common/ subdirectory
            parent = resolved.parent
            return parent.parent if parent.name == "common" else parent
        raise ValueError(f"Expected a run bundle directory or transcript.jsonl, got {resolved}")
    if resolved.name == "common" and (resolved / "transcript.jsonl").is_file():
        return resolved.parent
    return resolved


def _derive_conversation_run_metadata(*, artifact_dir: Path, run_id: str) -> tuple[str, str]:
    parent_name = artifact_dir.parent.name or "unknown"
    prefix = f"{parent_name}_"
    if run_id.startswith(prefix) and len(run_id) > len(prefix):
        return parent_name, run_id[len(prefix) :]

    parsed = _parse_run_id_with_profile_anchor(run_id)
    if parsed is not None:
        return parsed

    return parent_name, run_id


def _parse_run_id_with_profile_anchor(run_id: str) -> tuple[str, str] | None:
    theoretical_names = {"bt", "cbt", "het", "integrative", "pdt", "pmt"}
    parts = [part.strip() for part in str(run_id or "").split("_") if part.strip()]
    for index, part in enumerate(parts):
        if part not in theoretical_names:
            continue
        trend_name = "_".join(parts[:index]).strip()
        profile_id = "_".join(parts[index:]).strip()
        if trend_name and profile_id:
            return trend_name, profile_id
    return None


def _load_transcript_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _read_session_id(row: dict[str, Any]) -> int:
    for key in ("session_id", "session_number", "session_index"):
        value = row.get(key)
        if value is None:
            continue
        return int(value)
    raise ValueError(f"Transcript row is missing a session identifier: {row}")


def _normalize_speaker(value: Any) -> str:
    speaker = str(value or "").strip().lower()
    if speaker in {"client", "patient", "user"}:
        return "Client"
    if speaker in {"counselor", "therapist", "assistant"}:
        return "Counselor"
    if not speaker:
        return ""
    return str(value).strip()


def _extract_latent_scores(row: dict[str, Any]) -> tuple[dict[str, float], float]:
    hidden_state = row.get("session_hidden_state")
    if not isinstance(hidden_state, dict):
        hidden_state = {}
    latent_state = hidden_state.get("latent_state")
    if not isinstance(latent_state, dict):
        latent_state = {}
    phq8 = latent_state.get("phq8")
    if not isinstance(phq8, dict):
        phq8 = {}
    item_scores = {}
    for symptom_name in _load_phq8_symptom_order():
        raw_value = phq8.get(symptom_name, 0.0)
        if isinstance(raw_value, dict):
            raw_value = raw_value.get("score", 0.0)
        item_scores[symptom_name] = float(raw_value or 0.0)
    macro_labels = hidden_state.get("macro_labels")
    if not isinstance(macro_labels, dict):
        macro_labels = {}
    raw_total = macro_labels.get("phq8_total")
    latent_total = float(raw_total) if raw_total is not None else float(sum(item_scores.values()))
    return item_scores, latent_total


def _load_self_report_by_session(
    path: Path | None,
) -> dict[int, tuple[float | None, dict[str, float]]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_sessions = payload.get("sessions", []) if isinstance(payload, dict) else []
    if not isinstance(raw_sessions, list):
        return {}

    sessions: dict[int, tuple[float | None, dict[str, float]]] = {}
    for raw_session in raw_sessions:
        if not isinstance(raw_session, dict):
            continue
        session_id = raw_session.get("session_id") or raw_session.get("session_number")
        if session_id is None:
            continue
        session_id = int(session_id)
        nested = raw_session.get("self_report_score")
        if not isinstance(nested, dict):
            nested = raw_session

        raw_items = nested.get("symptom_scores") or nested.get("item_scores") or {}
        if not isinstance(raw_items, dict):
            raw_items = {}
        item_scores = {str(symptom_name): float(score) for symptom_name, score in raw_items.items()}
        raw_total = nested.get("overall_score")
        if raw_total is None:
            raw_total = nested.get("total_score")
        total = float(raw_total) if raw_total is not None else None
        if total is None and item_scores:
            total = round(sum(item_scores.values()), 2)
        sessions[session_id] = (total, item_scores)
    return sessions


def _load_ground_truth_by_session(path: Path | None) -> dict[int, tuple[dict[str, float], float]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_sessions = payload.get("sessions", []) if isinstance(payload, dict) else []
    if not isinstance(raw_sessions, list):
        return {}

    sessions: dict[int, tuple[dict[str, float], float]] = {}
    for raw_session in raw_sessions:
        if not isinstance(raw_session, dict):
            continue
        session_id = raw_session.get("session_id") or raw_session.get("session_number")
        if session_id is None:
            continue
        nested = raw_session.get("ground_truth_score")
        if not isinstance(nested, dict):
            nested = raw_session

        raw_items = nested.get("symptom_scores") or nested.get("item_scores") or {}
        if not isinstance(raw_items, dict):
            raw_items = {}
        item_scores = {str(symptom_name): float(score) for symptom_name, score in raw_items.items()}
        raw_total = nested.get("overall_score")
        if raw_total is None:
            raw_total = nested.get("total_score")
        total = float(raw_total) if raw_total is not None else float(sum(item_scores.values()))
        sessions[int(session_id)] = (item_scores, total)
    return sessions


def _load_phq8_symptom_order() -> tuple[str, ...]:
    mapping_path = PHQ8_RUNTIME_CUE_BANK_PATH
    try:
        payload = json.loads(mapping_path.read_text(encoding="utf-8"))
        raw_items = payload["instrument_definitions"]["phq8"]["items"]
        if not isinstance(raw_items, list):
            raise TypeError("PHQ-8 instrument item list must be a list.")
        symptom_order = tuple(str(item) for item in raw_items)
        if set(symptom_order) != set(DEFAULT_PHQ8_SYMPTOM_ORDER):
            raise ValueError("PHQ-8 symptom order in the runtime cue bank is incompatible.")
        return symptom_order
    except Exception:
        return DEFAULT_PHQ8_SYMPTOM_ORDER


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
