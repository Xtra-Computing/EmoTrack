from __future__ import annotations

import atexit
import copy
from dataclasses import dataclass, replace
import json
import os
from queue import Empty, Queue
import shutil
import signal
from string import Template
import subprocess
import threading
import time
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Any, TypeVar
from urllib.parse import urlparse
from uuid import uuid4

import httpx
import yaml
from dotenv import load_dotenv

from EmoTrack.conversation.api import _group_id, run_online_conversation_sessions
from EmoTrack.conversation.model import RetryableGenerationError
from EmoTrack.data import (
    DEFAULT_CONVERSATION_PROFILE_ROOT,
    discover_source_profile_paths,
    ensure_conversation_profile_artifacts,
    prepare_source_profile_manifest,
    profile_id_from_source_path,
)
from EmoTrack.generation.config import CohortGenerationConfig, load_cohort_generation_config
from EmoTrack.utils.env import first_env, first_non_empty
from EmoTrack.utils.paths import REPO_ROOT, repo_relative
from EmoTrack.utils.vertex_auth import (
    VertexAccessTokenProvider,
    is_vertex_base_url,
    normalize_vertex_api_root,
    normalize_vertex_model_name,
)


load_dotenv(REPO_ROOT / ".env", override=False)


RESULT_BUNDLE_VERSION = 4
RESULT_TRANSCRIPT_NAME = "transcript.jsonl"
RESULT_SELF_REPORT_NAME = "self_report.json"
RESULT_GROUND_TRUTH_NAME = "ground_truth.json"
CONVERSATION_TRENDS_DIRNAME = "conversation_trends"
DEFAULT_CONVERSATION_OUTPUT_PARENT = REPO_ROOT / "out"
DEFAULT_CONVERSATION_RESULT_PARENT = REPO_ROOT / "result"
DEFAULT_CONVERSATION_CACHE_PARENT = REPO_ROOT / "cache"
LOCAL_SERVICE_LOG_ROOT = REPO_ROOT / "out" / "start_conversation_launcher" / "services"
DEFAULT_GENERATION_VLLM_ARGS: dict[str, Any] = {
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.90,
    "kv_cache_memory_bytes": None,
    "max_model_len": 8192,
    "max_num_batched_tokens": 2048,
    "max_num_seqs": 256,
    "dtype": "bfloat16",
    "seed": 0,
    "trust_remote_code": False,
    "enforce_eager": True,
    "enable_chunked_prefill": True,
    "enable_prefix_caching": False,
    "gdn_prefill_backend": None,
}
VLLM_VALUE_FLAGS: tuple[tuple[str, str], ...] = (
    ("tensor_parallel_size", "--tensor-parallel-size"),
    ("gpu_memory_utilization", "--gpu-memory-utilization"),
    ("kv_cache_memory_bytes", "--kv-cache-memory-bytes"),
    ("max_model_len", "--max-model-len"),
    ("max_num_batched_tokens", "--max-num-batched-tokens"),
    ("max_num_seqs", "--max-num-seqs"),
    ("dtype", "--dtype"),
    ("seed", "--seed"),
    ("gdn_prefill_backend", "--gdn-prefill-backend"),
)
VLLM_BOOL_FLAGS: tuple[tuple[str, str], ...] = (
    ("trust_remote_code", "--trust-remote-code"),
    ("enforce_eager", "--enforce-eager"),
    ("enable_chunked_prefill", "--enable-chunked-prefill"),
    ("enable_prefix_caching", "--enable-prefix-caching"),
)


@dataclass
class GenerationWorker:
    index: int
    label: str
    visible_gpus: str
    base_url: str
    runtime_visible_gpus: str = ""
    base_raw: dict[str, Any] | None = None
    evermemos_api_url: str = ""


@dataclass(frozen=True)
class PreparedConversationJob:
    index: int
    total: int
    source_path: Path
    source_label: str


@dataclass(frozen=True)
class PreparedConversationTask:
    index: int
    total: int
    profile_index: int
    profile_total: int
    source_path: Path
    source_label: str
    trend_path: Path
    trend_name: str
    transient_attempts: int = 0


@dataclass(frozen=True)
class ResultBundleSyncResult:
    result_dir: Path
    copied_files: tuple[str, ...]
    missing_files: tuple[str, ...]


@dataclass(frozen=True)
class ConversationRunDecision:
    action: str
    existing_group_id: str = ""


QueueItemT = TypeVar("QueueItemT")


def _fill_worker_batch(
    primary_queue: Queue[QueueItemT],
    worker_queues: list[Queue[QueueItemT]],
    batch_size: int,
) -> list[QueueItemT]:
    """Prefer the worker's own queue, then steal pending work from peers."""
    batch: list[QueueItemT] = []
    for _ in range(batch_size):
        try:
            batch.append(primary_queue.get_nowait())
            continue
        except Empty:
            pass

        stolen_item: QueueItemT | None = None
        for queue in worker_queues:
            if queue is primary_queue:
                continue
            try:
                stolen_item = queue.get_nowait()
                break
            except Empty:
                continue
        if stolen_item is None:
            break
        batch.append(stolen_item)
    return batch


def _format_conversation_progress(run_dir: Path) -> str:
    progress_path = run_dir / "progress.json"
    if not progress_path.exists():
        return "status=new sessions=0/0"
    try:
        raw = json.loads(progress_path.read_text(encoding="utf-8"))
    except Exception:
        return f"status=unknown progress_file={progress_path}"
    if not isinstance(raw, dict):
        return f"status=invalid progress_file={progress_path}"

    status = str(raw.get("status", "unknown") or "unknown").strip()
    try:
        completed = int(raw.get("completed_session_count", 0) or 0)
    except (TypeError, ValueError):
        completed = 0
    try:
        total = int(raw.get("num_sessions", 0) or 0)
    except (TypeError, ValueError):
        total = 0

    parts = [f"status={status}", f"sessions={completed}/{total}"]
    if total > 0 and completed < total:
        parts.append(f"next_session={completed + 1}")
    return " ".join(parts)


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _is_valid_json_file(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return True


def _is_valid_jsonl_file(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                json.loads(line)
    except Exception:
        return False
    return True


def _int_or_zero(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _has_complete_self_report_payload(payload: dict[str, Any], *, expected_sessions: int | None) -> bool:
    sessions = payload.get("sessions", [])
    if not isinstance(sessions, list) or not sessions:
        return False
    by_session_id = {
        _int_or_zero(session.get("session_id")): session
        for session in sessions
        if isinstance(session, dict)
    }
    session_ids = range(1, expected_sessions + 1) if expected_sessions is not None else by_session_id.keys()
    for session_id in session_ids:
        session = by_session_id.get(session_id)
        if not isinstance(session, dict):
            return False
        score = session.get("self_report_score")
        if not isinstance(score, dict):
            return False
        samples = score.get("samples")
        if _int_or_zero(score.get("sample_count")) <= 0:
            return False
        if not isinstance(samples, list) or not samples:
            return False
        if not isinstance(score.get("symptom_scores"), dict):
            return False
        if "overall_score" not in score:
            return False
    return True


def _has_expected_sessions(payload: dict[str, Any], *, expected_sessions: int | None) -> bool:
    sessions = payload.get("sessions", [])
    if not isinstance(sessions, list) or not sessions:
        return False
    if expected_sessions is None:
        return True
    session_ids = {
        _int_or_zero(session.get("session_id"))
        for session in sessions
        if isinstance(session, dict)
    }
    return all(session_id in session_ids for session_id in range(1, expected_sessions + 1))


def _completed_out_progress_session_count(run_dir: Path) -> int | None:
    progress = _read_json_object(run_dir / "progress.json")
    if not progress:
        return None
    completed = _int_or_zero(progress.get("completed_session_count"))
    total = _int_or_zero(progress.get("num_sessions"))
    status = str(progress.get("status") or "").strip().lower()
    if total <= 0:
        return None
    if status == "completed" and completed >= total:
        return total
    return None


def _has_completed_conversation_run(*, result_run_dir: Path, conversation_run_dir: Path) -> bool:
    completed_sessions = _completed_out_progress_session_count(conversation_run_dir)
    if completed_sessions is None:
        return False
    return _has_completed_result_bundle(result_run_dir, expected_sessions=completed_sessions)


def _has_existing_out_progress(run_dir: Path) -> bool:
    progress = _read_json_object(run_dir / "progress.json")
    if not progress:
        return False
    return _int_or_zero(progress.get("num_sessions")) > 0


def _has_completed_result_bundle(result_run_dir: Path, *, expected_sessions: int | None = None) -> bool:
    common_dir = result_run_dir / "common"
    if not _is_valid_jsonl_file(common_dir / RESULT_TRANSCRIPT_NAME):
        return False
    self_report = _read_json_object(common_dir / RESULT_SELF_REPORT_NAME)
    if not self_report or not _has_complete_self_report_payload(
        self_report,
        expected_sessions=expected_sessions,
    ):
        return False
    ground_truth = _read_json_object(common_dir / RESULT_GROUND_TRUTH_NAME)
    return bool(ground_truth) and _has_expected_sessions(
        ground_truth,
        expected_sessions=expected_sessions,
    )


def _has_resumable_out_progress(run_dir: Path) -> bool:
    progress = _read_json_object(run_dir / "progress.json")
    if not progress:
        return False
    completed = _int_or_zero(progress.get("completed_session_count"))
    total = _int_or_zero(progress.get("num_sessions"))
    status = str(progress.get("status") or "").strip().lower()
    if total <= 0:
        return False
    return status in {"running", "interrupted", "failed"} and completed < total


def _existing_group_id_from_out_progress(run_dir: Path) -> str:
    progress = _read_json_object(run_dir / "progress.json")
    return str(progress.get("group_id") or "").strip() if progress else ""


def _decide_conversation_run_action(*, result_run_dir: Path, conversation_run_dir: Path) -> ConversationRunDecision:
    completed_sessions = _completed_out_progress_session_count(conversation_run_dir)
    if completed_sessions is not None:
        if _has_completed_result_bundle(result_run_dir, expected_sessions=completed_sessions):
            return ConversationRunDecision(action="skip")
        return ConversationRunDecision(
            action="resume",
            existing_group_id=_existing_group_id_from_out_progress(conversation_run_dir),
        )
    if _has_resumable_out_progress(conversation_run_dir):
        return ConversationRunDecision(
            action="resume",
            existing_group_id=_existing_group_id_from_out_progress(conversation_run_dir),
        )
    if _has_existing_out_progress(conversation_run_dir):
        return ConversationRunDecision(
            action="resume",
            existing_group_id=_existing_group_id_from_out_progress(conversation_run_dir),
        )
    return ConversationRunDecision(action="start")


def _reset_single_run_runtime_paths(*, conversation_run_dir: Path, result_run_dir: Path, cache_run_dir: Path) -> None:
    for stale_dir in (conversation_run_dir, result_run_dir, cache_run_dir):
        if stale_dir.exists():
            shutil.rmtree(stale_dir)


def _sync_result_bundle(source_run_dir: Path, result_run_dir: Path) -> ResultBundleSyncResult:
    result_run_dir.mkdir(parents=True, exist_ok=True)

    copied_files: list[str] = []
    missing_files: list[str] = []
    transcript_present = _copy_bundle_file(
        source_run_dir=source_run_dir,
        result_run_dir=result_run_dir,
        filename=RESULT_TRANSCRIPT_NAME,
        validator=_is_valid_jsonl_file,
        copied_files=copied_files,
        missing_files=missing_files,
    )
    self_report_present = _copy_bundle_file(
        source_run_dir=source_run_dir,
        result_run_dir=result_run_dir,
        filename=RESULT_SELF_REPORT_NAME,
        validator=_is_valid_json_file,
        copied_files=copied_files,
        missing_files=missing_files,
    )
    ground_truth_present = _write_ground_truth_bundle(
        source_run_dir=source_run_dir,
        result_run_dir=result_run_dir,
        copied_files=copied_files,
    )
    if not ground_truth_present:
        missing_files.append(RESULT_GROUND_TRUTH_NAME)
    return ResultBundleSyncResult(
        result_dir=result_run_dir,
        copied_files=tuple(copied_files),
        missing_files=tuple(missing_files),
    )


def _copy_bundle_file(
    *,
    source_run_dir: Path,
    result_run_dir: Path,
    filename: str,
    validator,
    copied_files: list[str],
    missing_files: list[str],
) -> bool:
    source_common = source_run_dir / "common"
    source_path = source_common / filename if source_common.is_dir() else source_run_dir / filename
    common_dir = result_run_dir / "common"
    common_dir.mkdir(parents=True, exist_ok=True)
    target_path = common_dir / filename
    is_present = source_path.exists()
    if is_present and not validator(source_path):
        print(f"[bundle] WARNING: {filename} is corrupted, skipping copy", flush=True)
        is_present = False
    if is_present:
        if source_path.resolve() != target_path.resolve():
            shutil.copy2(source_path, target_path)
        copied_files.append(filename)
    else:
        missing_files.append(filename)
    return is_present


def _write_ground_truth_bundle(
    *,
    source_run_dir: Path,
    result_run_dir: Path,
    copied_files: list[str],
) -> bool:
    source_path = source_run_dir / "session_latent_plan.json"
    common_dir = result_run_dir / "common"
    common_dir.mkdir(parents=True, exist_ok=True)
    target_path = common_dir / RESULT_GROUND_TRUTH_NAME
    if not source_path.is_file():
        return False
    try:
        payload = _build_ground_truth_payload(source_path)
    except Exception:
        print("[bundle] WARNING: session_latent_plan.json is corrupted, skipping ground truth export", flush=True)
        return False
    target_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    copied_files.append(RESULT_GROUND_TRUTH_NAME)
    return True


def _build_ground_truth_payload(session_latent_plan_path: Path) -> dict[str, Any]:
    raw_payload = json.loads(session_latent_plan_path.read_text(encoding="utf-8"))
    if not isinstance(raw_payload, dict):
        raise ValueError("session_latent_plan.json must be a JSON object")
    raw_sessions = raw_payload.get("sessions", [])
    if not isinstance(raw_sessions, list):
        raise ValueError("session_latent_plan.json sessions must be a list")

    sessions: list[dict[str, Any]] = []
    for raw_session in raw_sessions:
        if not isinstance(raw_session, dict):
            continue
        session_id = raw_session.get("session_id")
        if session_id is None:
            continue
        hidden_state = raw_session.get("hidden_state", {})
        if not isinstance(hidden_state, dict):
            hidden_state = {}
        latent_state = hidden_state.get("latent_state", {})
        if not isinstance(latent_state, dict):
            latent_state = {}
        phq8 = latent_state.get("phq8", {})
        if not isinstance(phq8, dict):
            phq8 = {}
        symptom_scores = {
            symptom_name: float(((phq8.get(symptom_name) or {}) if isinstance(phq8.get(symptom_name), dict) else {"score": phq8.get(symptom_name)}).get("score", 0.0) or 0.0)
            for symptom_name in (
                "anhedonia",
                "depressed_mood",
                "sleep",
                "fatigue",
                "appetite",
                "self_worth",
                "concentration",
                "psychomotor",
            )
        }
        macro_labels = hidden_state.get("macro_labels", {})
        if not isinstance(macro_labels, dict):
            macro_labels = {}
        overall_score = macro_labels.get("phq8_total")
        if overall_score is None:
            overall_score = sum(symptom_scores.values())
        sessions.append(
            {
                "session_id": int(session_id),
                "ground_truth_score": {
                    "scale": "PHQ-8",
                    "symptom_scores": symptom_scores,
                    "overall_score": float(overall_score),
                    "severity_band": str(macro_labels.get("phq8_band") or ""),
                },
            }
        )

    return {
        "result_bundle_version": RESULT_BUNDLE_VERSION,
        "run_id": str(raw_payload.get("run_id") or session_latent_plan_path.parent.name),
        "scale": "PHQ-8",
        "sessions": sessions,
    }


def _read_env_assignments(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def _evermemos_enabled(raw: dict[str, Any]) -> bool:
    evermemos_cfg = raw.get("evermemos", {})
    if not isinstance(evermemos_cfg, dict):
        return False
    return bool(evermemos_cfg.get("enabled", False))


def _log_selected_endpoints(configured: dict[str, Any]) -> None:
    models_cfg = configured.get("models", {})
    evermemos_cfg = configured.get("evermemos", {})
    if not isinstance(models_cfg, dict):
        models_cfg = {}
    if not isinstance(evermemos_cfg, dict):
        evermemos_cfg = {}
    evermemos_enabled = bool(evermemos_cfg.get("enabled", False))

    print(
        "[cohort] selected endpoints "
        f"generation={models_cfg.get('base_url', '')} "
        f"model={models_cfg.get('model_name', '')} "
        f"evermemos={'enabled' if evermemos_enabled else 'disabled'} "
        f"vectorize={evermemos_cfg.get('vectorize_base_url', '') if evermemos_enabled else 'disabled'} "
        f"rerank={evermemos_cfg.get('rerank_base_url', '') if evermemos_enabled and evermemos_cfg.get('rerank_enabled', True) else 'disabled'}",
        flush=True,
    )


def _model_directory_suffix(model_name: str) -> str:
    normalized = str(model_name or "").strip()
    if not normalized:
        return "model"

    parts = [part.strip() for part in normalized.split("/") if part.strip()]
    if not parts:
        return "model"

    if len(parts) == 1:
        combined = parts[0]
    else:
        vendor = parts[0]
        leaf = parts[-1]
        if leaf.lower().startswith(vendor.lower()):
            leaf = leaf[len(vendor) :]
        vendor = re.sub(r"^([A-Za-z]+)([0-9])", r"\1_\2", vendor)
        leaf = re.sub(r"^([A-Za-z]+)([0-9])", r"\1_\2", leaf)
        combined = "_".join(part for part in (vendor, leaf) if part)
    combined = re.sub(r"[^A-Za-z0-9.]+", "_", combined)
    combined = re.sub(r"_+", "_", combined).strip("_")
    return combined.lower() or "model"


def _default_model_scoped_root(*, parent: Path, model_name: str) -> Path:
    suffix = _model_directory_suffix(model_name)
    return (parent / f"{CONVERSATION_TRENDS_DIRNAME}_{suffix}").resolve()


def _warn_if_repo_env_mismatch(configured: dict[str, Any]) -> None:
    repo_env = _read_env_assignments(REPO_ROOT / "repos" / "EverMemOS" / ".env")
    if not repo_env:
        return

    models_cfg = configured.get("models", {})
    evermemos_cfg = configured.get("evermemos", {})
    if not isinstance(models_cfg, dict):
        models_cfg = {}
    if not isinstance(evermemos_cfg, dict):
        evermemos_cfg = {}

    expected = {
        "LLM_BASE_URL": str(evermemos_cfg.get("llm_base_url", models_cfg.get("base_url", ""))),
        "LLM_MODEL": str(evermemos_cfg.get("llm_model", models_cfg.get("model_name", ""))),
        "VECTORIZE_BASE_URL": str(evermemos_cfg.get("vectorize_base_url", "")),
        "VECTORIZE_MODEL": str(evermemos_cfg.get("vectorize_model", "")),
    }
    mismatches = [
        f"{key}: repo_env={repo_env.get(key, '')} expected={value}"
        for key, value in expected.items()
        if value and repo_env.get(key, "") and repo_env.get(key, "") != value
    ]
    if mismatches:
        print("[cohort] warning: repos/EverMemOS/.env does not match the selected run config:", flush=True)
        for item in mismatches:
            print(f"[cohort]   {item}", flush=True)
        print(
            "[cohort] if the EverMemOS API was started from repos/EverMemOS/.env, restart it with the new env before running.",
            flush=True,
        )


def _log_generation_workers(workers: list[GenerationWorker]) -> None:
    if not workers:
        return
    summary = ", ".join(
        f"{worker.label} base_url={worker.base_url} evermemos={worker.evermemos_api_url or 'shared'} CUDA_VISIBLE_DEVICES={worker.visible_gpus}"
        for worker in workers
    )
    print(f"[cohort] generation workers: {summary}", flush=True)


def _vllm_base_url(raw_url: str) -> str:
    """Strip trailing path segments like /rerank to get the vLLM base."""
    parsed = urlparse(raw_url.strip().rstrip("/"))
    path = parsed.path
    for suffix in ("/rerank", "/embeddings", "/completions", "/chat/completions"):
        if path.endswith(suffix):
            path = path[: -len(suffix)]
            break
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def _check_endpoint_reachable(base_url: str, label: str) -> None:
    """Verify an endpoint responds. Raises RuntimeError on failure."""
    _check_endpoint_reachable_with_api_key(base_url, label, api_key="")


def _auth_headers(api_key: str) -> dict[str, str]:
    key = str(api_key or "").strip()
    if not key or key.upper() == "EMPTY":
        return {}
    return {"Authorization": f"Bearer {key}"}


def _model_source_name(configured: dict[str, Any]) -> str:
    models = configured.get("models", {})
    if not isinstance(models, dict):
        return "api"
    return str(models.get("source", "api") or "api").strip().lower()


def _auth_headers_for_base_url(base_url: str, *, api_key: str) -> dict[str, str]:
    headers = _auth_headers(api_key)
    if headers:
        return headers
    if not is_vertex_base_url(base_url):
        return {}
    token = VertexAccessTokenProvider(base_url=base_url).current_api_key()
    return _auth_headers(token)


def _check_endpoint_reachable_with_api_key(base_url: str, label: str, *, api_key: str) -> None:
    """Verify an endpoint responds, optionally using bearer auth."""
    normalized = _vllm_base_url(base_url)
    if not normalized:
        raise RuntimeError(f"{label} endpoint is empty in config")
    if is_vertex_base_url(normalized):
        return
    headers = _auth_headers_for_base_url(normalized, api_key=api_key)
    try:
        response = httpx.get(f"{normalized}/models", timeout=5.0, headers=headers or None)
        if response.status_code == 401 and not headers:
            raise RuntimeError(
                f"{label} endpoint {normalized} requires authentication; "
                "configure an API key in config or environment."
            )
        response.raise_for_status()
    except Exception as exc:
        raise RuntimeError(
            f"{label} endpoint {normalized} is not reachable: {exc}"
        ) from exc


def _verify_model_served(base_url: str, model_name: str, label: str) -> None:
    """Verify the configured model is actually served by the endpoint."""
    _verify_model_served_with_api_key(base_url, model_name, label, api_key="")


def _verify_model_served_with_api_key(base_url: str, model_name: str, label: str, *, api_key: str) -> None:
    """Verify the configured model is actually served by the endpoint."""
    normalized = _vllm_base_url(base_url)
    if not normalized or not model_name.strip():
        return
    headers = _auth_headers_for_base_url(normalized, api_key=api_key)
    if is_vertex_base_url(normalized):
        payload = {
            "contents": [{"role": "user", "parts": [{"text": "ping"}]}],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": 1,
            },
        }
        try:
            response = httpx.post(
                (
                    f"{normalize_vertex_api_root(normalized)}/publishers/google/models/"
                    f"{normalize_vertex_model_name(model_name)}:generateContent"
                ),
                timeout=30.0,
                headers={
                    **(headers or {}),
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
        except Exception as exc:
            raise RuntimeError(
                f"{label} endpoint {normalized} failed Vertex Gemini probe for model {model_name}: {exc}"
            ) from exc
        return
    try:
        response = httpx.get(f"{normalized}/models", timeout=5.0, headers=headers or None)
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return
    if not isinstance(payload, dict):
        return
    data = payload.get("data", [])
    if not isinstance(data, list):
        return
    served_ids = [
        str(item.get("id") or "").strip()
        for item in data
        if isinstance(item, dict) and str(item.get("id") or "").strip()
    ]
    if not served_ids:
        return
    if model_name not in served_ids:
        raise RuntimeError(
            f"{label} model '{model_name}' is not served by {normalized}. "
            f"Available: {', '.join(served_ids)}. "
            f"Fix the model name in your config or start the correct vLLM process."
        )


_SPAWNED_SERVERS: list[subprocess.Popen[bytes]] = []
_PROCESS_LOG_PATHS: dict[int, Path] = {}


def _cleanup_servers() -> None:
    for proc in _SPAWNED_SERVERS:
        if proc.poll() is None:
            print(f"[cohort] stopping vLLM server pid={proc.pid}", flush=True)
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


atexit.register(_cleanup_servers)


def _install_signal_cleanup() -> None:
    """Register SIGTERM/SIGINT handlers so spawned vLLM processes are cleaned up
    even when the pipeline is killed by a signal (atexit does not fire on SIGKILL,
    but covers SIGTERM/SIGINT when we re-raise after cleanup)."""

    def _handler(signum: int, _frame: Any) -> None:
        _cleanup_servers()
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _handler)


_install_signal_cleanup()


def _port_from_url(url: str) -> int:
    parsed = urlparse(url)
    if parsed.port is not None:
        return parsed.port
    return 443 if parsed.scheme == "https" else 80


def _replace_url_port(url: str, port: int) -> str:
    parsed = urlparse(url)
    hostname = parsed.hostname
    if not parsed.scheme or not parsed.netloc or not hostname:
        raise ValueError(f"Cannot derive worker URL from base URL: {url}")

    host = hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"

    auth = ""
    if parsed.username:
        auth = parsed.username
        if parsed.password is not None:
            auth += f":{parsed.password}"
        auth += "@"

    netloc = f"{auth}{host}:{port}"
    return parsed._replace(netloc=netloc).geturl()


def _is_endpoint_up(base_url: str) -> bool:
    try:
        r = httpx.get(f"{base_url.rstrip('/')}/models", timeout=3.0)
        return r.status_code == 200
    except Exception:
        return False


def _service_log_path(*, kind: str, label: str, port: int) -> Path:
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("._") or kind
    LOCAL_SERVICE_LOG_ROOT.mkdir(parents=True, exist_ok=True)
    return LOCAL_SERVICE_LOG_ROOT / f"{kind}_{safe_label}_{port}.log"


def _vllm_executable() -> str:
    repo_local = REPO_ROOT / ".venv" / "bin" / "vllm"
    if repo_local.exists():
        return str(repo_local)
    resolved = shutil.which("vllm")
    if resolved:
        return resolved
    raise FileNotFoundError(
        f"vllm executable not found; expected repo-local {repo_local} or PATH entry"
    )


def _process_stderr_tail(proc: subprocess.Popen[bytes], max_chars: int = 4000) -> str:
    log_path = _PROCESS_LOG_PATHS.get(proc.pid or -1)
    if log_path is not None and log_path.exists():
        try:
            payload = log_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            payload = ""
    elif proc.stderr is not None:
        try:
            payload = proc.stderr.read().decode("utf-8", errors="replace")
        except Exception:
            payload = ""
    else:
        return ""
    payload = payload.strip()
    if len(payload) <= max_chars:
        return payload
    return payload[-max_chars:]


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _generation_vllm_args(vllm_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "tensor_parallel_size": int(
            vllm_cfg.get(
                "tensor_parallel_size",
                DEFAULT_GENERATION_VLLM_ARGS["tensor_parallel_size"],
            )
        ),
        "gpu_memory_utilization": float(
            vllm_cfg.get(
                "gpu_memory_utilization",
                DEFAULT_GENERATION_VLLM_ARGS["gpu_memory_utilization"],
            )
        ),
        "kv_cache_memory_bytes": _optional_int(
            vllm_cfg.get(
                "kv_cache_memory_bytes",
                DEFAULT_GENERATION_VLLM_ARGS["kv_cache_memory_bytes"],
            )
        ),
        "max_model_len": int(
            vllm_cfg.get(
                "max_model_len",
                DEFAULT_GENERATION_VLLM_ARGS["max_model_len"],
            )
        ),
        "max_num_batched_tokens": int(
            vllm_cfg.get(
                "max_num_batched_tokens",
                DEFAULT_GENERATION_VLLM_ARGS["max_num_batched_tokens"],
            )
        ),
        "max_num_seqs": int(
            vllm_cfg.get(
                "max_num_seqs",
                DEFAULT_GENERATION_VLLM_ARGS["max_num_seqs"],
            )
        ),
        "dtype": str(vllm_cfg.get("dtype", DEFAULT_GENERATION_VLLM_ARGS["dtype"])),
        "seed": int(vllm_cfg.get("seed", DEFAULT_GENERATION_VLLM_ARGS["seed"])),
        "trust_remote_code": bool(
            vllm_cfg.get(
                "trust_remote_code",
                DEFAULT_GENERATION_VLLM_ARGS["trust_remote_code"],
            )
        ),
        "enforce_eager": bool(
            vllm_cfg.get(
                "enforce_eager",
                DEFAULT_GENERATION_VLLM_ARGS["enforce_eager"],
            )
        ),
        "enable_chunked_prefill": bool(
            vllm_cfg.get(
                "enable_chunked_prefill",
                DEFAULT_GENERATION_VLLM_ARGS["enable_chunked_prefill"],
            )
        ),
        "enable_prefix_caching": bool(
            vllm_cfg.get(
                "enable_prefix_caching",
                DEFAULT_GENERATION_VLLM_ARGS["enable_prefix_caching"],
            )
        ),
        "gdn_prefill_backend": str(vllm_cfg.get("gdn_prefill_backend", "")) or None,
    }


def _wait_for_endpoints(
    pending: list[tuple[str, str, subprocess.Popen[bytes]]], timeout_sec: int = 600
) -> None:
    """Poll until all spawned endpoints are up, or surface startup failure."""
    start = time.time()
    last_msg = 0
    remaining = {i for i in range(len(pending))}
    while time.time() - start < timeout_sec:
        for i in list(remaining):
            base_url, label, proc = pending[i]
            if _is_endpoint_up(base_url):
                print(f"[cohort] {label} server ready at {base_url}", flush=True)
                remaining.discard(i)
                continue
            if proc.poll() is not None:
                stderr_tail = _process_stderr_tail(proc)
                details = f" exited with code {proc.returncode}"
                if stderr_tail:
                    details += f"\n{stderr_tail}"
                raise RuntimeError(f"{label} server failed to start:{details}")
        if not remaining:
            return
        elapsed = int(time.time() - start)
        if elapsed - last_msg >= 10:
            labels = [pending[i][1] for i in remaining]
            print(f"[cohort] waiting for {', '.join(labels)}... ({elapsed}s)", flush=True)
            last_msg = elapsed
        time.sleep(2)
    still_pending = [pending[i][1] for i in remaining]
    raise RuntimeError(
        f"Servers {still_pending} did not start within {timeout_sec}s"
    )


def _spawn_vllm_server(
    model: str,
    port: int,
    base_url: str,
    visible_gpus: str,
    label: str,
    *,
    runner: str | None = None,
    convert: str | None = None,
    vllm_args: dict[str, Any] | None = None,
    reasoning_parser: str | None = None,
) -> tuple[str, str, subprocess.Popen[bytes]] | None:
    """Spawn a vLLM server if not already up."""
    if _is_endpoint_up(base_url):
        print(f"[cohort] {label} already running at {base_url}", flush=True)
        return None

    cmd = [
        _vllm_executable(), "serve", model,
        "--port", str(port),
    ]
    if vllm_args:
        for key, flag in VLLM_VALUE_FLAGS:
            value = vllm_args.get(key)
            if value is None:
                continue
            cmd.extend([flag, str(value)])

        for key, flag in VLLM_BOOL_FLAGS:
            value = vllm_args.get(key)
            if value is None:
                continue
            cmd.append(flag if bool(value) else f"--no-{flag[2:]}")
    if reasoning_parser:
        cmd.extend(["--reasoning-parser", reasoning_parser])
    if runner:
        cmd.extend(["--runner", runner])
    if convert:
        cmd.extend(["--convert", convert])

    env = os.environ.copy()
    if visible_gpus:
        env["CUDA_VISIBLE_DEVICES"] = visible_gpus

    print(f"[cohort] starting {label}: {' '.join(cmd)} (CUDA_VISIBLE_DEVICES={visible_gpus})", flush=True)
    log_path = _service_log_path(kind="vllm", label=label, port=port)
    with log_path.open("ab") as log_handle:
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
    _SPAWNED_SERVERS.append(proc)
    if proc.pid is not None:
        _PROCESS_LOG_PATHS[proc.pid] = log_path
    return (base_url, label, proc)


def _merge_visible_gpus(*values: str) -> str:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        for item in value.split(","):
            gpu_id = item.strip()
            if not gpu_id or gpu_id in seen:
                continue
            seen.add(gpu_id)
            ordered.append(gpu_id)
    return ",".join(ordered)


def _split_visible_gpus(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _validate_generation_workers(visible_gpu_sets: list[str]) -> None:
    seen_gpu_to_worker: dict[str, int] = {}
    for worker_index, visible_gpus in enumerate(visible_gpu_sets, start=1):
        gpu_ids = _split_visible_gpus(visible_gpus)
        if not gpu_ids:
            raise ValueError("Each --llm-gpu entry must resolve to at least one CUDA device")
        for gpu_id in gpu_ids:
            previous_worker = seen_gpu_to_worker.get(gpu_id)
            if previous_worker is None:
                seen_gpu_to_worker[gpu_id] = worker_index
                continue
            raise ValueError(
                f"Generation workers must use disjoint GPUs for profile parallelism; "
                f"GPU {gpu_id} was assigned to worker {previous_worker} and worker {worker_index}"
            )


def _allocate_generation_worker_urls(
    base_url: str,
    num_workers: int,
    *,
    reserved_urls: list[str],
) -> list[str]:
    if num_workers <= 0:
        return []

    urls = [base_url]
    used_ports = {_port_from_url(base_url)}
    used_ports.update(_port_from_url(url) for url in reserved_urls if url)
    next_port = _port_from_url(base_url) + 1

    while len(urls) < num_workers:
        while next_port in used_ports:
            next_port += 1
        urls.append(_replace_url_port(base_url, next_port))
        used_ports.add(next_port)
        next_port += 1
    return urls


def _validate_tensor_parallelism(raw: dict[str, Any], visible_gpu_sets: list[str]) -> None:
    models_cfg = raw.get("models", {})
    if not isinstance(models_cfg, dict):
        models_cfg = {}
    vllm_cfg = models_cfg.get("vllm", {})
    if not isinstance(vllm_cfg, dict):
        vllm_cfg = {}
    tensor_parallel_size = int(vllm_cfg.get("tensor_parallel_size", 1) or 1)
    if tensor_parallel_size <= 0:
        raise ValueError("models.vllm.tensor_parallel_size must be >= 1")

    for worker_index, visible_gpus in enumerate(visible_gpu_sets, start=1):
        gpu_count = len(_split_visible_gpus(visible_gpus))
        if tensor_parallel_size > gpu_count:
            raise ValueError(
                f"models.vllm.tensor_parallel_size={tensor_parallel_size} requires at least "
                f"{tensor_parallel_size} GPUs per worker, but worker {worker_index} only has {gpu_count}: {visible_gpus}"
            )


def _start_model_servers(
    raw: dict[str, Any],
    *,
    llm_visible_gpu_workers: list[str],
    encode_visible_gpus: str,
    rerank_visible_gpus: str = "",
) -> list[GenerationWorker]:
    """Start shared vLLM servers for generation plus encode-side services."""
    models_cfg = raw.get("models", {}) or {}
    evermemos_cfg = raw.get("evermemos", {}) or {}
    evermemos_enabled = _evermemos_enabled(raw)
    vllm_cfg = models_cfg.get("vllm", {}) or {}
    thinking_cfg = models_cfg.get("thinking", {}) or {}

    gen_base_url = str(models_cfg.get("base_url", "")).strip()
    gen_model = str(models_cfg.get("model_name", "")).strip()
    vec_base_url = str(evermemos_cfg.get("vectorize_base_url", "")).strip() if evermemos_enabled else ""
    vec_model = str(evermemos_cfg.get("vectorize_model", "")).strip() if evermemos_enabled else ""
    vec_port = _port_from_url(vec_base_url) if vec_base_url else 0

    rerank_raw_url = str(evermemos_cfg.get("rerank_base_url", "")).strip() if evermemos_enabled else ""
    rerank_base = _vllm_base_url(rerank_raw_url) if rerank_raw_url else ""
    rerank_model = str(evermemos_cfg.get("rerank_model", "")).strip() if evermemos_enabled else ""
    rerank_port = _port_from_url(rerank_raw_url) if rerank_raw_url else 0

    started_ports: set[int] = set()
    pending: list[tuple[str, str, subprocess.Popen[bytes]]] = []
    generation_vllm_args = _generation_vllm_args(vllm_cfg)

    def reasoning_parser_for_model(model: str) -> str | None:
        _ = model
        if not thinking_cfg.get("enabled", True):
            return None
        # For local Qwen API serving, vLLM's server-side reasoning parser can
        # move the entire completion into `reasoning` and leave `content=None`.
        # Keep the raw text in `content` so the client-side pipeline can recover
        # visible output instead of failing the turn outright.
        return None

    def maybe_spawn(
        base_url: str,
        model: str,
        port: int,
        label: str,
        *,
        visible_gpus: str,
        **kwargs: Any,
    ) -> tuple[str, str, subprocess.Popen[bytes]] | None:
        if not base_url or not model:
            return None
        if port in started_ports:
            return None
        result = _spawn_vllm_server(
            model=model, port=port, base_url=base_url,
            visible_gpus=visible_gpus, label=label, **kwargs
        )
        if result:
            pending.append(result)
        started_ports.add(port)
        return result

    worker_urls = _allocate_generation_worker_urls(
        gen_base_url,
        len(llm_visible_gpu_workers),
        reserved_urls=[url for url in (vec_base_url, rerank_raw_url) if url],
    )
    workers: list[GenerationWorker] = []
    for index, (visible_gpus, worker_url) in enumerate(zip(llm_visible_gpu_workers, worker_urls, strict=True), start=1):
        label = "generation" if len(llm_visible_gpu_workers) == 1 else f"generation[{index}]"
        maybe_spawn(
            worker_url,
            gen_model,
            _port_from_url(worker_url),
            label,
            visible_gpus=visible_gpus,
            vllm_args=generation_vllm_args,
            reasoning_parser=reasoning_parser_for_model(gen_model),
        )
        workers.append(
            GenerationWorker(
                index=index,
                label=label,
                visible_gpus=visible_gpus,
                base_url=worker_url,
                runtime_visible_gpus=(
                    _merge_visible_gpus(visible_gpus, encode_visible_gpus, rerank_visible_gpus)
                    if evermemos_enabled
                    else visible_gpus
                ),
            )
        )
    # Spawn encoder and rerank servers before waiting, so all vLLM instances
    # start in parallel instead of serialising LLM → encoder → rerank.
    if evermemos_enabled:
        maybe_spawn(
            vec_base_url,
            vec_model,
            vec_port,
            "vectorize",
            visible_gpus=encode_visible_gpus,
            convert="embed",
            vllm_args={
                "gpu_memory_utilization": float(
                    os.getenv("EMOTRACK_VECTORIZE_GPU_MEM_UTILIZATION")
                    or str(evermemos_cfg.get("vectorize_gpu_memory_utilization", 0.35))
                ),
                "max_model_len": int(os.getenv("EMOTRACK_VECTORIZE_MAX_MODEL_LEN", "4096")),
            },
        )
    if evermemos_enabled and rerank_visible_gpus:
        maybe_spawn(
            rerank_base,
            rerank_model,
            rerank_port,
            "rerank",
            visible_gpus=rerank_visible_gpus,
            convert="classify",
            vllm_args={
                "gpu_memory_utilization": float(
                    os.getenv("EMOTRACK_RERANK_GPU_MEM_UTILIZATION", "0.35")
                ),
                "max_model_len": int(os.getenv("EMOTRACK_RERANK_MAX_MODEL_LEN", "4096")),
            },
        )

    if pending:
        _wait_for_endpoints(pending)
    return workers


def _resolve_trend_path(raw_path: str) -> list[Path]:
    """Resolve a trend path to one or more config files.

    If *raw_path* points to a directory, returns all YAML/JSON files inside it.
    Otherwise returns a single-element list with the resolved file path.
    """
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        if candidate.exists():
            resolved = candidate.resolve()
            if resolved.is_dir():
                return _expand_trend_dir(resolved)
            return [resolved]
        raise FileNotFoundError(f"Trend config not found: {candidate}")

    names = [raw_path]
    if candidate.suffix.lower() not in {".yaml", ".yml", ".json"}:
        names.append(f"{raw_path}.yaml")

    for name in names:
        for base in (
            REPO_ROOT / "configs" / "conversation_trends",
            REPO_ROOT / "configs",
            REPO_ROOT,
        ):
            resolved = (base / name).resolve()
            if resolved.exists():
                if resolved.is_dir():
                    return _expand_trend_dir(resolved)
                return [resolved]
    raise FileNotFoundError(f"Trend config not found: {raw_path}")


def _expand_trend_dir(directory: Path) -> list[Path]:
    """Return sorted YAML/JSON trend files inside *directory*."""
    files = sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in {".yaml", ".yml", ".json"}
    )
    if not files:
        raise FileNotFoundError(f"No trend configs found in directory: {directory}")
    return files


def _resolve_source_profile_path(raw_path: str, *, profiles_root: Path) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
        raise FileNotFoundError(f"Source profile not found: {resolved}")

    for base in (Path.cwd(), REPO_ROOT, profiles_root):
        resolved = (base / candidate).resolve()
        if resolved.exists():
            return resolved
    raise FileNotFoundError(f"Source profile not found: {raw_path}")


def _device_to_visible_gpus(raw_device: str) -> str:
    normalized = raw_device.strip()
    if not normalized or normalized.lower() in {"cpu", "none"}:
        return ""
    gpu_ids: list[str] = []
    seen: set[str] = set()
    for item in normalized.split(","):
        token = item.strip().lower()
        if not token:
            continue
        if token.startswith("cuda:"):
            token = token.split(":", 1)[1].strip()
        if not token.isdigit():
            raise ValueError(f"Unsupported device string: {raw_device}")
        if token in seen:
            raise ValueError(f"Duplicate CUDA device in worker assignment: {raw_device}")
        seen.add(token)
        gpu_ids.append(token)
    if not gpu_ids:
        return ""
    return ",".join(gpu_ids)


def _configure_model_endpoints(
    raw: dict[str, Any],
    visible_gpus: str,
    *,
    rerank_enabled: bool,
    generation_base_url_override: str | None = None,
    llm_base_url_override: str | None = None,
    evermemos_api_base_url_override: str | None = None,
) -> dict[str, Any]:
    """Use model names and endpoints exactly as specified in the config.

    No auto-discovery: the config is the single source of truth for model
    names and URLs.  We only validate that the configured endpoints are
    reachable and actually serve the expected model.
    """
    configured = copy.deepcopy(raw)
    configured.setdefault("models", {})
    configured.setdefault("evermemos", {})
    configured.setdefault("runtime", {})
    evermemos_enabled = _evermemos_enabled(configured)

    generation_base_url = generation_base_url_override or str(configured["models"].get("base_url", "")).strip()
    generation_model = str(configured["models"].get("model_name", "")).strip()
    if not generation_base_url or not generation_model:
        raise RuntimeError(
            "Config must specify models.base_url and models.model_name. "
            "No auto-discovery is performed."
        )

    configured["models"]["source"] = "api"
    configured["models"]["api_key"] = str(configured["models"].get("api_key", "") or "EMPTY")
    configured["models"]["base_url"] = generation_base_url
    configured["evermemos"]["enabled"] = evermemos_enabled

    if evermemos_enabled:
        llm_base_url = (
            llm_base_url_override
            or str(configured["evermemos"].get("llm_base_url", "")).strip()
            or generation_base_url
        )
        llm_model = str(configured["evermemos"].get("llm_model", "")).strip() or generation_model
        configured["evermemos"]["llm_api_key"] = str(configured["evermemos"].get("llm_api_key", "") or "EMPTY")
        configured["evermemos"]["llm_base_url"] = llm_base_url
        configured["evermemos"]["llm_model"] = llm_model

        vectorize_base_url = str(configured["evermemos"].get("vectorize_base_url", "")).strip()
        vectorize_model = str(configured["evermemos"].get("vectorize_model", "")).strip()
        if not vectorize_base_url or not vectorize_model:
            raise RuntimeError(
                "Config must specify evermemos.vectorize_base_url and evermemos.vectorize_model. "
                "No auto-discovery is performed."
            )
        configured["evermemos"]["vectorize_api_key"] = str(configured["evermemos"].get("vectorize_api_key", "") or "EMPTY")

        rerank_base_url = str(configured["evermemos"].get("rerank_base_url", "")).strip()
        rerank_model = str(configured["evermemos"].get("rerank_model", "")).strip() or generation_model
        configured["evermemos"]["rerank_enabled"] = rerank_enabled
        configured["evermemos"]["rerank_api_key"] = str(configured["evermemos"].get("rerank_api_key", "") or "EMPTY")
        configured["evermemos"]["rerank_base_url"] = rerank_base_url if rerank_enabled else ""
        configured["evermemos"]["rerank_model"] = rerank_model if rerank_enabled else ""
    else:
        configured["evermemos"]["rerank_enabled"] = False

    if evermemos_enabled and evermemos_api_base_url_override:
        configured["evermemos"]["api_base_url"] = evermemos_api_base_url_override

    configured["runtime"]["gpus"] = visible_gpus
    configured.setdefault("simulation", {})
    if evermemos_enabled and not rerank_enabled:
        configured["simulation"]["retrieval_method"] = "rrf"

    _check_endpoint_reachable(generation_base_url, "generation")
    _verify_model_served(generation_base_url, generation_model, "generation")
    if evermemos_enabled:
        if llm_base_url != generation_base_url:
            _check_endpoint_reachable(llm_base_url, "evermemos llm")
        _verify_model_served(llm_base_url, llm_model, "evermemos llm")
        _check_endpoint_reachable(vectorize_base_url, "vectorize")
        _verify_model_served(vectorize_base_url, vectorize_model, "vectorize")
        if rerank_enabled:
            _check_endpoint_reachable(rerank_base_url, "rerank")
            _verify_model_served(rerank_base_url, rerank_model, "rerank")

    return configured


def _resolve_generation_api_key(configured: dict[str, Any]) -> str:
    models = configured.get("models", {})
    if not isinstance(models, dict):
        models = {}
    source = _model_source_name(configured)
    configured_key = str(models.get("api_key", ""))
    if source == "vertex":
        return configured_key
    return first_non_empty(
        configured_key,
        first_env("EMOTRACK_MODEL_API_KEY", default=""),
    )


def _resolve_openai_like_key_for_source(source: str, configured_key: str, generation_key: str) -> str:
    if source == "vertex":
        return first_non_empty(
            configured_key,
            generation_key,
            default="",
        )
    return first_non_empty(
        configured_key,
        generation_key,
        first_env("EMOTRACK_MODEL_API_KEY", "OPENAI_API_KEY", default=""),
    )


def _configure_model_endpoints_api(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate hosted API endpoints without mutating local-generation semantics."""
    configured = copy.deepcopy(raw)
    configured.setdefault("models", {})
    configured.setdefault("evermemos", {})
    configured.setdefault("runtime", {})
    evermemos_enabled = _evermemos_enabled(configured)
    model_source = _model_source_name(configured)

    generation_base_url = str(configured["models"].get("base_url", "")).strip()
    generation_model = str(configured["models"].get("model_name", "")).strip()
    if not generation_base_url or not generation_model:
        raise RuntimeError("API backend requires models.base_url and models.model_name.")
    if model_source == "vertex" and not is_vertex_base_url(generation_base_url):
        raise RuntimeError("models.source=vertex requires a Vertex AI base_url.")
    if model_source == "vertex":
        try:
            normalize_vertex_api_root(generation_base_url)
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
    if model_source == "api" and is_vertex_base_url(generation_base_url):
        raise RuntimeError("Vertex AI endpoints require models.source=vertex.")

    # Keep authored keys in the config; API auth resolves from explicit fields or
    # the canonical environment variables at runtime.
    configured["models"]["source"] = model_source
    configured["evermemos"]["enabled"] = evermemos_enabled
    generation_api_key = _resolve_generation_api_key(configured)

    if evermemos_enabled:
        llm_base_url = str(configured["evermemos"].get("llm_base_url", "")).strip() or generation_base_url
        llm_model = str(configured["evermemos"].get("llm_model", "")).strip() or generation_model
        configured["evermemos"]["llm_base_url"] = llm_base_url
        configured["evermemos"]["llm_model"] = llm_model
        llm_api_key = _resolve_openai_like_key_for_source(
            model_source,
            str(configured["evermemos"].get("llm_api_key", "")),
            generation_api_key,
        )

        vectorize_base_url = str(configured["evermemos"].get("vectorize_base_url", "")).strip()
        vectorize_model = str(configured["evermemos"].get("vectorize_model", "")).strip()
        if not vectorize_base_url or not vectorize_model:
            raise RuntimeError("API backend requires evermemos.vectorize_base_url and evermemos.vectorize_model.")
        vectorize_api_key = _resolve_openai_like_key_for_source(
            "api",
            str(configured["evermemos"].get("vectorize_api_key", "")),
            generation_api_key,
        )

        rerank_enabled = bool(configured["evermemos"].get("rerank_enabled", True))
        rerank_base_url = str(configured["evermemos"].get("rerank_base_url", "")).strip()
        rerank_model = str(configured["evermemos"].get("rerank_model", "")).strip() or generation_model
        rerank_api_key = _resolve_openai_like_key_for_source(
            "api",
            str(configured["evermemos"].get("rerank_api_key", "")),
            generation_api_key,
        )
    else:
        configured["evermemos"]["rerank_enabled"] = False

    _check_endpoint_reachable_with_api_key(generation_base_url, "generation", api_key=generation_api_key)
    _verify_model_served_with_api_key(generation_base_url, generation_model, "generation", api_key=generation_api_key)

    if evermemos_enabled and (llm_base_url != generation_base_url or llm_model != generation_model):
        _check_endpoint_reachable_with_api_key(llm_base_url, "evermemos llm", api_key=llm_api_key)
        _verify_model_served_with_api_key(llm_base_url, llm_model, "evermemos llm", api_key=llm_api_key)

    if evermemos_enabled:
        _check_endpoint_reachable_with_api_key(vectorize_base_url, "vectorize", api_key=vectorize_api_key)
        _verify_model_served_with_api_key(vectorize_base_url, vectorize_model, "vectorize", api_key=vectorize_api_key)

        if rerank_enabled:
            _check_endpoint_reachable_with_api_key(rerank_base_url, "rerank", api_key=rerank_api_key)
            _verify_model_served_with_api_key(rerank_base_url, rerank_model, "rerank", api_key=rerank_api_key)
    return configured


def _api_generation_workers(base_raw: dict[str, Any], *, num_workers: int) -> list[GenerationWorker]:
    if num_workers <= 0:
        raise ValueError("--api-workers must be >= 1")
    base_url = str((base_raw.get("models", {}) or {}).get("base_url", "")).strip()
    workers: list[GenerationWorker] = []
    for index in range(1, num_workers + 1):
        label = "api" if num_workers == 1 else f"api[{index}]"
        workers.append(
            GenerationWorker(
                index=index,
                label=label,
                visible_gpus="",
                base_url=base_url,
                runtime_visible_gpus="",
                base_raw=base_raw,
            )
        )
    return workers


def _load_template(path: Path) -> dict[str, Any]:
    text = Template(path.read_text(encoding="utf-8")).safe_substitute(os.environ)
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise TypeError(f"Expected YAML object in {path}")
    return raw


def _build_run_config(
    *,
    base_raw: dict[str, Any],
    prepared_profile: Any,
    trend_path: Path,
    trend_name: str,
    output_root: Path,
    cache_root: Path,
    generated_config_root: Path,
    num_sessions: int,
    target_turns: int,
    visible_gpus: str,
    progress_position: int = 0,
    fresh: bool = False,
    existing_group_id: str = "",
    resume_run: bool | None = None,
) -> Path:
    raw = copy.deepcopy(base_raw)
    run_id = f"{trend_name}_{prepared_profile.profile_id}"
    should_resume = (not fresh) if resume_run is None else bool(resume_run)
    memory_group_id = existing_group_id.strip()
    if not memory_group_id:
        if should_resume:
            memory_group_id = _group_id(run_id)
        else:
            memory_group_id = _group_id(f"{run_id}_{uuid4().hex[:12]}")
    raw.setdefault("run", {})
    raw["run"]["output_root"] = repo_relative(output_root)
    raw["run"]["cache_root"] = repo_relative(cache_root)
    raw["run"]["run_id"] = run_id
    raw["run"]["memory_group_id"] = memory_group_id
    raw["run"]["resume"] = should_resume

    raw.setdefault("simulation", {})
    raw["simulation"]["num_sessions"] = num_sessions
    raw["simulation"]["target_turns"] = target_turns

    raw.setdefault("paths", {})
    raw["paths"]["client_profile_path"] = repo_relative(prepared_profile.client_profile_path)
    raw["paths"]["hidden_state_path"] = repo_relative(prepared_profile.hidden_state_path)
    raw["paths"]["trend_path"] = repo_relative(trend_path)

    raw.setdefault("runtime", {})
    raw["runtime"]["gpus"] = visible_gpus
    raw["runtime"]["generated_env_path"] = repo_relative(cache_root / f"{run_id}.env")
    raw["runtime"]["progress_position"] = max(0, int(progress_position))

    raw.setdefault("evermemos", {})
    prefix = str(raw["evermemos"].get("group_name_prefix", "EmoTrack Conversation")).strip()
    raw["evermemos"]["group_name_prefix"] = f"{prefix} {trend_name}".strip()

    config_path = generated_config_root / f"{prepared_profile.profile_id}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return config_path


def _trend_runtime_paths(
    *,
    trend_name: str,
    data_root: Path,
    output_root_root: Path,
    result_root_root: Path,
    cache_root_root: Path,
) -> tuple[Path, Path, Path, Path]:
    output_root = (output_root_root / trend_name).resolve()
    result_root = (result_root_root / trend_name).resolve()
    cache_root = (cache_root_root / trend_name).resolve()
    generated_config_root = data_root / "generated_configs" / trend_name
    return output_root, result_root, cache_root, generated_config_root


def _reset_trend_runtime_paths(
    *,
    trend_paths: list[Path],
    data_root: Path,
    output_root_root: Path,
    result_root_root: Path,
    cache_root_root: Path,
) -> None:
    seen_trends: set[str] = set()
    for trend_path in trend_paths:
        trend_name = trend_path.stem
        if trend_name in seen_trends:
            continue
        seen_trends.add(trend_name)
        output_root, result_root, cache_root, generated_config_root = _trend_runtime_paths(
            trend_name=trend_name,
            data_root=data_root,
            output_root_root=output_root_root,
            result_root_root=result_root_root,
            cache_root_root=cache_root_root,
        )
        for stale_dir in (generated_config_root, cache_root, output_root, result_root):
            if stale_dir.exists():
                print(f"[cohort] --fresh: removing {stale_dir}", flush=True)
                shutil.rmtree(stale_dir)


def _reset_sampled_runtime_roots(
    *,
    data_root: Path,
    output_root_root: Path,
    result_root_root: Path,
    cache_root_root: Path,
) -> None:
    generated_config_root = data_root / "generated_configs"
    for stale_dir in (generated_config_root, cache_root_root, output_root_root, result_root_root):
        if stale_dir.exists():
            print(f"[cohort] --fresh(sampled): removing {stale_dir}", flush=True)
            shutil.rmtree(stale_dir)


def _run_single_trend(
    *,
    generation_workers: list[GenerationWorker],
    trend_path: Path,
    profiles_root: Path,
    data_root: Path,
    output_root_root: Path,
    result_root_root: Path,
    cache_root_root: Path,
    source_paths: list[Path],
    num_sessions: int,
    target_turns: int,
    force_convert: bool,
    fresh: bool,
    continue_on_error: bool,
    worker_batch_size: int = 1,
) -> bool:
    if not generation_workers:
        raise ValueError("At least one generation worker is required")

    trend_name = trend_path.stem
    output_root, result_root, cache_root, generated_config_root = _trend_runtime_paths(
        trend_name=trend_name,
        data_root=data_root,
        output_root_root=output_root_root,
        result_root_root=result_root_root,
        cache_root_root=cache_root_root,
    )

    if fresh:
        _reset_trend_runtime_paths(
            trend_paths=[trend_path],
            data_root=data_root,
            output_root_root=output_root_root,
            result_root_root=result_root_root,
            cache_root_root=cache_root_root,
        )

    print(f"[cohort] trend={trend_name} profiles={len(source_paths)}", flush=True)

    failures: list[str] = []

    # Round-robin distribute jobs across workers so one fast-failing worker
    # cannot drain the entire shared queue while others are still busy.
    all_jobs = [
        PreparedConversationJob(
            index=idx,
            total=len(source_paths),
            source_path=source_path,
            source_label=f"{source_path.parent.name}/{source_path.name}",
        )
        for idx, source_path in enumerate(source_paths, start=1)
    ]
    worker_queues: list[Queue[PreparedConversationJob]] = [
        Queue() for _ in generation_workers
    ]
    for job_idx, job in enumerate(all_jobs):
        worker_queues[job_idx % len(generation_workers)].put(job)

    stop_event = threading.Event()
    failure_lock = threading.Lock()
    fatal_error: list[Exception] = []

    def _run_single_job(
        worker: GenerationWorker,
        job: PreparedConversationJob,
        worker_base_raw: dict[str, Any],
    ) -> None:
        conversation_run_id = ""
        conversation_run_dir = output_root
        result_run_dir = result_root
        try:
            print(
                f"[cohort] ({job.index}/{job.total}) worker={worker.label} preparing source={job.source_label}",
                flush=True,
            )
            prepared_profile = ensure_conversation_profile_artifacts(
                job.source_path,
                output_root=data_root,
                profiles_root=profiles_root,
                force=force_convert,
            )

            conversation_run_id = f"{trend_name}_{prepared_profile.profile_id}"
            conversation_run_dir = (output_root / conversation_run_id).resolve()
            result_run_dir = (result_root / conversation_run_id).resolve()
            cache_run_dir = (cache_root / conversation_run_id).resolve()
            run_decision = _decide_conversation_run_action(
                result_run_dir=result_run_dir,
                conversation_run_dir=conversation_run_dir,
            )
            if run_decision.action == "skip":
                print(
                    f"[cohort] ({job.index}/{job.total}) worker={worker.label} skipped "
                    f"conversation={conversation_run_id} result_dir={result_run_dir}",
                    flush=True,
                )
                return
            if run_decision.action == "start":
                _reset_single_run_runtime_paths(
                    conversation_run_dir=conversation_run_dir,
                    result_run_dir=result_run_dir,
                    cache_run_dir=cache_run_dir,
                )
            config_path = _build_run_config(
                base_raw=worker_base_raw,
                prepared_profile=prepared_profile,
                trend_path=trend_path,
                trend_name=trend_name,
                output_root=output_root,
                cache_root=cache_root,
                generated_config_root=generated_config_root,
                num_sessions=num_sessions,
                target_turns=target_turns,
                visible_gpus=worker.runtime_visible_gpus,
                progress_position=worker.index - 1,
                fresh=fresh,
                existing_group_id=run_decision.existing_group_id,
                resume_run=run_decision.action == "resume",
            )
            print(
                f"[cohort] ({job.index}/{job.total}) worker={worker.label} "
                f"conversation={conversation_run_id} source={job.source_label} "
                f"{_format_conversation_progress(conversation_run_dir)}",
                flush=True,
            )
            run_dir = run_online_conversation_sessions(config_path=str(config_path))
            bundle = _sync_result_bundle(run_dir, result_run_dir)
            print(
                f"[cohort] ({job.index}/{job.total}) worker={worker.label} completed "
                f"conversation={conversation_run_id} {_format_conversation_progress(run_dir)} "
                f"run_dir={run_dir} result_dir={bundle.result_dir}",
                flush=True,
            )
        except Exception as exc:
            message = f"{job.source_path}: {exc}"
            if conversation_run_id and conversation_run_dir.exists():
                bundle = _sync_result_bundle(conversation_run_dir, result_run_dir)
                print(
                    f"[cohort] ({job.index}/{job.total}) worker={worker.label} synced partial result bundle "
                    f"result_dir={bundle.result_dir}",
                    flush=True,
                )
            with failure_lock:
                failures.append(message)
                if not continue_on_error and not fatal_error:
                    fatal_error.append(exc)
                    stop_event.set()
            print(
                f"[cohort] ({job.index}/{job.total}) worker={worker.label} failed "
                f"conversation={conversation_run_id or 'unprepared'} "
                f"{_format_conversation_progress(conversation_run_dir)} error={exc}",
                flush=True,
            )

    def worker_loop(worker: GenerationWorker, my_queue: Queue[PreparedConversationJob]) -> None:
        worker_base_raw = worker.base_raw
        if worker_base_raw is None:
            with failure_lock:
                exc = RuntimeError(f"Generation worker {worker.label} is missing its configured endpoint")
                failures.append(str(exc))
                if not continue_on_error and not fatal_error:
                    fatal_error.append(exc)
                    stop_event.set()
            return

        while True:
            if stop_event.is_set() and not continue_on_error:
                return

            batch = _fill_worker_batch(my_queue, worker_queues, worker_batch_size)
            if not batch:
                return

            if len(batch) == 1:
                _run_single_job(worker, batch[0], worker_base_raw)
            else:
                sub_threads = [
                    threading.Thread(
                        target=_run_single_job,
                        args=(worker, job, worker_base_raw),
                        name=f"trend-{trend_name}-worker-{worker.index}-job-{job.index}",
                    )
                    for job in batch
                ]
                for t in sub_threads:
                    t.start()
                for t in sub_threads:
                    t.join()

    threads = [
        threading.Thread(
            target=worker_loop,
            args=(worker, worker_queues[i]),
            name=f"trend-{trend_name}-worker-{worker.index}",
        )
        for i, worker in enumerate(generation_workers)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    if fatal_error and not continue_on_error:
        raise fatal_error[0]

    if failures:
        print(f"[cohort] failures for {trend_name}:", flush=True)
        for item in failures:
            print(f"  - {item}", flush=True)
        return False
    return True


def _run_single_profile_across_trends(
    *,
    generation_workers: list[GenerationWorker],
    trend_paths: list[Path],
    profiles_root: Path,
    data_root: Path,
    output_root_root: Path,
    result_root_root: Path,
    cache_root_root: Path,
    source_path: Path,
    num_sessions: int,
    target_turns: int,
    force_convert: bool,
    fresh: bool,
    continue_on_error: bool,
) -> bool:
    if not generation_workers:
        raise ValueError("At least one generation worker is required")
    if not trend_paths:
        return True

    worker_count = min(len(generation_workers), len(trend_paths))
    trend_queue: Queue[Path] = Queue()
    for trend_path in trend_paths:
        trend_queue.put(trend_path)

    failures: list[str] = []
    stop_event = threading.Event()
    failure_lock = threading.Lock()
    fatal_error: list[Exception] = []
    max_transient_retries = max(1, int(os.getenv("EMOTRACK_TRANSIENT_MAX_RETRIES", "50")))
    source_label = f"{source_path.parent.name}/{source_path.name}"

    def worker_loop(worker: GenerationWorker) -> None:
        while True:
            if stop_event.is_set() and not continue_on_error:
                return
            try:
                trend_path = trend_queue.get_nowait()
            except Empty:
                return

            try:
                print(
                    f"[cohort] profile={source_label} worker={worker.label} trend={trend_path.stem} queued",
                    flush=True,
                )
                succeeded = _run_single_trend(
                    generation_workers=[worker],
                    trend_path=trend_path,
                    profiles_root=profiles_root,
                    data_root=data_root,
                    output_root_root=output_root_root,
                    result_root_root=result_root_root,
                    cache_root_root=cache_root_root,
                    source_paths=[source_path],
                    num_sessions=num_sessions,
                    target_turns=target_turns,
                    force_convert=force_convert,
                    fresh=fresh,
                    continue_on_error=continue_on_error,
                )
                if not succeeded:
                    with failure_lock:
                        failures.append(f"{source_label} trend={trend_path.stem} failed")
                        if not continue_on_error:
                            stop_event.set()
            except Exception as exc:
                with failure_lock:
                    failures.append(f"{source_label} trend={trend_path.stem}: {exc}")
                    if not continue_on_error and not fatal_error:
                        fatal_error.append(exc)
                        stop_event.set()
                print(
                    f"[cohort] profile={source_label} worker={worker.label} trend={trend_path.stem} error={exc}",
                    flush=True,
                )
            finally:
                trend_queue.task_done()

    threads = [
        threading.Thread(
            target=worker_loop,
            args=(worker,),
            name=f"profile-{source_path.stem}-worker-{worker.index}",
        )
        for worker in generation_workers[:worker_count]
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    if fatal_error and not continue_on_error:
        raise fatal_error[0]

    if failures:
        print(f"[cohort] profile-first failures for {source_label}:", flush=True)
        for item in failures:
            print(f"  - {item}", flush=True)
        return False
    return True


SAMPLE_TREND_MAP_FILENAME = "sample_trend_map.json"


def _load_or_create_sample_trend_map(
    *,
    source_paths: list[Path],
    trend_paths: list[Path],
    profiles_root: Path,
    cache_root: Path,
    seed: int,
    rounds: int = 1,
) -> dict[str, list[str]]:
    """Return a ``{profile_id: [trend_stem_r0, trend_stem_r1, ...]}`` mapping.

    Each profile is assigned *rounds* distinct trends (no repeats within
    a profile).  Round *r* uses seed *r* (0, 1, 2, ...) so results are
    reproducible.  The mapping is cached to
    ``{cache_root}/sample_trend_map.json``.
    """
    import random as _random

    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / SAMPLE_TREND_MAP_FILENAME

    cached: dict[str, list[str]] = {}
    if cache_path.is_file():
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        # Migrate legacy single-trend format ({pid: stem_str}) to list.
        for k, v in raw.items():
            cached[k] = v if isinstance(v, list) else [v]
        print(f"[batch] loaded sample trend map ({len(cached)} entries) from {cache_path}", flush=True)

    trend_stems = sorted(p.stem for p in trend_paths)
    if rounds > len(trend_stems):
        raise ValueError(
            f"sample_rounds ({rounds}) exceeds available trends ({len(trend_stems)})"
        )

    pids = [profile_id_from_source_path(sp, profiles_root) for sp in source_paths]

    # Build assignments round by round. Within each round, a single RNG
    # (seeded with the round index) walks all profiles in stable order so
    # every profile gets a different draw from the same sequence.
    updated = False
    for r in range(rounds):
        rng = _random.Random(r)
        for pid in pids:
            assignments = cached.setdefault(pid, [])
            if len(assignments) > r:
                # Already assigned for this round (from cache).
                rng.random()  # advance RNG to keep sequence aligned
                continue
            already = set(assignments)
            available = [t for t in trend_stems if t not in already]
            chosen = rng.choice(available)
            assignments.append(chosen)
            updated = True

    if updated:
        cache_path.write_text(
            json.dumps(cached, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"[batch] saved sample trend map ({len(cached)} entries, {rounds} rounds) to {cache_path}", flush=True)

    return cached


def _run_profile_first_queue(
    *,
    generation_workers: list[GenerationWorker],
    trend_paths: list[Path],
    profiles_root: Path,
    data_root: Path,
    output_root_root: Path,
    result_root_root: Path,
    cache_root_root: Path,
    source_paths: list[Path],
    num_sessions: int,
    target_turns: int,
    force_convert: bool,
    fresh: bool,
    continue_on_error: bool,
    worker_batch_size: int = 1,
    trend_first: bool = False,
    sample_trend_per_profile: bool = False,
    sample_seed: int = 42,
    sample_rounds: int = 1,
) -> bool:
    if not generation_workers:
        raise ValueError("At least one generation worker is required")
    if not trend_paths:
        return True

    if fresh:
        _reset_trend_runtime_paths(
            trend_paths=trend_paths,
            data_root=data_root,
            output_root_root=output_root_root,
            result_root_root=result_root_root,
            cache_root_root=cache_root_root,
        )
        if sample_trend_per_profile:
            _reset_sampled_runtime_roots(
                data_root=data_root,
                output_root_root=output_root_root,
                result_root_root=result_root_root,
                cache_root_root=cache_root_root,
            )

    # Round-robin distribute tasks across workers so one fast-failing worker
    # cannot drain the entire shared queue while others are still busy.
    if sample_trend_per_profile:
        # Multi-round mode: each profile gets sample_rounds distinct trends.
        # Order: all profiles round 0, all profiles round 1, ...
        trend_map = _load_or_create_sample_trend_map(
            source_paths=source_paths,
            trend_paths=trend_paths,
            profiles_root=profiles_root,
            cache_root=cache_root_root,
            seed=sample_seed,
            rounds=sample_rounds,
        )
        trend_by_stem: dict[str, Path] = {p.stem: p for p in trend_paths}
        # Pre-compute (source_path, trend_stem) pairs for all rounds.
        pairs: list[tuple[Path, str]] = []
        for round_idx in range(sample_rounds):
            for source_path in source_paths:
                pid = profile_id_from_source_path(source_path, profiles_root)
                round_trends = trend_map.get(pid)
                if round_trends is None or round_idx >= len(round_trends):
                    continue
                trend_stem = round_trends[round_idx]
                if trend_stem not in trend_by_stem:
                    print(
                        f"[batch] warning: trend {trend_stem} not found for profile {pid} round {round_idx}, skipping",
                        flush=True,
                    )
                    continue
                pairs.append((source_path, trend_stem))
        total_tasks = len(pairs)
        all_tasks: list[PreparedConversationTask] = []
        for task_index, (source_path, trend_stem) in enumerate(pairs, start=1):
            source_label = f"{source_path.parent.name}/{source_path.name}"
            all_tasks.append(
                PreparedConversationTask(
                    index=task_index,
                    total=total_tasks,
                    profile_index=task_index,
                    profile_total=len(source_paths),
                    source_path=source_path,
                    source_label=source_label,
                    trend_path=trend_by_stem[trend_stem],
                    trend_name=trend_stem,
                )
            )
    else:
        total_tasks = len(source_paths) * len(trend_paths)
        all_tasks: list[PreparedConversationTask] = []
        task_index = 0
        if trend_first:
            for trend_path in trend_paths:
                for profile_index, source_path in enumerate(source_paths, start=1):
                    task_index += 1
                    source_label = f"{source_path.parent.name}/{source_path.name}"
                    all_tasks.append(
                        PreparedConversationTask(
                            index=task_index,
                            total=total_tasks,
                            profile_index=profile_index,
                            profile_total=len(source_paths),
                            source_path=source_path,
                            source_label=source_label,
                            trend_path=trend_path,
                            trend_name=trend_path.stem,
                        )
                    )
        else:
            for profile_index, source_path in enumerate(source_paths, start=1):
                source_label = f"{source_path.parent.name}/{source_path.name}"
                for trend_path in trend_paths:
                    task_index += 1
                    all_tasks.append(
                        PreparedConversationTask(
                            index=task_index,
                            total=total_tasks,
                            profile_index=profile_index,
                            profile_total=len(source_paths),
                            source_path=source_path,
                            source_label=source_label,
                            trend_path=trend_path,
                            trend_name=trend_path.stem,
                        )
                    )

    # Pre-filter: skip tasks whose result bundles are already complete so that
    # workers only receive genuinely pending work and GPU utilisation stays high.
    pending_tasks: list[PreparedConversationTask] = []
    skipped_count = 0
    for task in all_tasks:
        pid = profile_id_from_source_path(task.source_path, profiles_root)
        conversation_run_id = f"{task.trend_name}_{pid}"
        if sample_trend_per_profile:
            conversation_run_dir = output_root_root / conversation_run_id
            result_run_dir = result_root_root / conversation_run_id
        else:
            conversation_run_dir = output_root_root / task.trend_name / conversation_run_id
            result_run_dir = result_root_root / task.trend_name / conversation_run_id
        if not fresh and _has_completed_conversation_run(
            result_run_dir=result_run_dir, conversation_run_dir=conversation_run_dir,
        ):
            skipped_count += 1
            continue
        pending_tasks.append(task)
    if skipped_count:
        print(
            f"[cohort] pre-filter: {skipped_count} already-complete tasks skipped, "
            f"{len(pending_tasks)} pending",
            flush=True,
        )

    worker_queues: list[Queue[PreparedConversationTask]] = [
        Queue() for _ in generation_workers
    ]
    for task_idx, task in enumerate(pending_tasks):
        worker_queues[task_idx % len(generation_workers)].put(task)

    failures: list[str] = []
    stop_event = threading.Event()
    failure_lock = threading.Lock()
    fatal_error: list[Exception] = []
    max_transient_retries = max(1, int(os.getenv("EMOTRACK_TRANSIENT_MAX_RETRIES", "50")))

    def _run_single_task(
        worker: GenerationWorker,
        task: PreparedConversationTask,
        worker_base_raw: dict[str, Any],
        retry_queue: Queue[PreparedConversationTask],
    ) -> None:
        if sample_trend_per_profile:
            # Flat layout: no trend subfolder.
            output_root = output_root_root.resolve()
            result_root = result_root_root.resolve()
            cache_root = cache_root_root.resolve()
            generated_config_root = data_root / "generated_configs"
        else:
            output_root, result_root, cache_root, generated_config_root = _trend_runtime_paths(
                trend_name=task.trend_name,
                data_root=data_root,
                output_root_root=output_root_root,
                result_root_root=result_root_root,
                cache_root_root=cache_root_root,
            )

        conversation_run_id = ""
        conversation_run_dir = output_root
        result_run_dir = result_root
        try:
            print(
                f"[cohort] ({task.index}/{task.total}) worker={worker.label} preparing "
                f"profile={task.source_label} trend={task.trend_name}",
                flush=True,
            )
            prepared_profile = ensure_conversation_profile_artifacts(
                task.source_path,
                output_root=data_root,
                profiles_root=profiles_root,
                force=force_convert,
            )

            conversation_run_id = f"{task.trend_name}_{prepared_profile.profile_id}"
            conversation_run_dir = (output_root / conversation_run_id).resolve()
            result_run_dir = (result_root / conversation_run_id).resolve()
            cache_run_dir = (cache_root / conversation_run_id).resolve()
            run_decision = _decide_conversation_run_action(
                result_run_dir=result_run_dir,
                conversation_run_dir=conversation_run_dir,
            )
            if run_decision.action == "skip":
                print(
                    f"[cohort] ({task.index}/{task.total}) worker={worker.label} skipped "
                    f"profile={task.source_label} trend={task.trend_name} "
                    f"conversation={conversation_run_id} result_dir={result_run_dir}",
                    flush=True,
                )
                return
            if run_decision.action == "start":
                _reset_single_run_runtime_paths(
                    conversation_run_dir=conversation_run_dir,
                    result_run_dir=result_run_dir,
                    cache_run_dir=cache_run_dir,
                )
            config_path = _build_run_config(
                base_raw=worker_base_raw,
                prepared_profile=prepared_profile,
                trend_path=task.trend_path,
                trend_name=task.trend_name,
                output_root=output_root,
                cache_root=cache_root,
                generated_config_root=generated_config_root,
                num_sessions=num_sessions,
                target_turns=target_turns,
                visible_gpus=worker.runtime_visible_gpus,
                progress_position=worker.index - 1,
                fresh=fresh,
                existing_group_id=run_decision.existing_group_id,
                resume_run=run_decision.action == "resume",
            )
            print(
                f"[cohort] ({task.index}/{task.total}) worker={worker.label} "
                f"profile={task.source_label} trend={task.trend_name} "
                f"conversation={conversation_run_id} {_format_conversation_progress(conversation_run_dir)}",
                flush=True,
            )
            run_dir = run_online_conversation_sessions(config_path=str(config_path))
            bundle = _sync_result_bundle(run_dir, result_run_dir)
            print(
                f"[cohort] ({task.index}/{task.total}) worker={worker.label} completed "
                f"profile={task.source_label} trend={task.trend_name} "
                f"conversation={conversation_run_id} {_format_conversation_progress(run_dir)} "
                f"run_dir={run_dir} result_dir={bundle.result_dir}",
                flush=True,
            )
        except RetryableGenerationError as exc:
            if conversation_run_id and conversation_run_dir.exists():
                bundle = _sync_result_bundle(conversation_run_dir, result_run_dir)
                print(
                    f"[cohort] ({task.index}/{task.total}) worker={worker.label} synced partial result bundle "
                    f"result_dir={bundle.result_dir}",
                    flush=True,
                )
            next_attempt = task.transient_attempts + 1
            if next_attempt <= max_transient_retries:
                retry_after = getattr(exc, "retry_after_seconds", None)
                if retry_after and retry_after > 0:
                    time.sleep(float(retry_after))
                retry_queue.put(replace(task, transient_attempts=next_attempt))
                print(
                    f"[cohort] ({task.index}/{task.total}) worker={worker.label} requeued transient failure "
                    f"profile={task.source_label} trend={task.trend_name} "
                    f"conversation={conversation_run_id or 'unprepared'} "
                    f"attempt={next_attempt}/{max_transient_retries} error={exc}",
                    flush=True,
                )
                return
            message = (
                f"{task.source_path} trend={task.trend_name}: exhausted transient retries "
                f"({max_transient_retries}) after {exc}"
            )
            with failure_lock:
                failures.append(message)
                if not continue_on_error and not fatal_error:
                    fatal_error.append(exc)
                    stop_event.set()
            print(
                f"[cohort] ({task.index}/{task.total}) worker={worker.label} failed "
                f"profile={task.source_label} trend={task.trend_name} "
                f"conversation={conversation_run_id or 'unprepared'} "
                f"{_format_conversation_progress(conversation_run_dir)} error={message}",
                flush=True,
            )
        except Exception as exc:
            message = f"{task.source_path} trend={task.trend_name}: {exc}"
            if conversation_run_id and conversation_run_dir.exists():
                bundle = _sync_result_bundle(conversation_run_dir, result_run_dir)
                print(
                    f"[cohort] ({task.index}/{task.total}) worker={worker.label} synced partial result bundle "
                    f"result_dir={bundle.result_dir}",
                    flush=True,
                )
            with failure_lock:
                failures.append(message)
                if not continue_on_error and not fatal_error:
                    fatal_error.append(exc)
                    stop_event.set()
            print(
                f"[cohort] ({task.index}/{task.total}) worker={worker.label} failed "
                f"profile={task.source_label} trend={task.trend_name} "
                f"conversation={conversation_run_id or 'unprepared'} "
                f"{_format_conversation_progress(conversation_run_dir)} error={exc}",
                flush=True,
            )

    def worker_loop(worker: GenerationWorker, my_queue: Queue[PreparedConversationTask]) -> None:
        worker_base_raw = worker.base_raw
        if worker_base_raw is None:
            with failure_lock:
                exc = RuntimeError(f"Generation worker {worker.label} is missing its configured endpoint")
                failures.append(str(exc))
                if not continue_on_error and not fatal_error:
                    fatal_error.append(exc)
                    stop_event.set()
            return

        while True:
            if stop_event.is_set() and not continue_on_error:
                return

            batch = _fill_worker_batch(my_queue, worker_queues, worker_batch_size)
            if not batch:
                return

            if len(batch) == 1:
                _run_single_task(worker, batch[0], worker_base_raw, my_queue)
            else:
                sub_threads = [
                    threading.Thread(
                        target=_run_single_task,
                        args=(worker, task, worker_base_raw, my_queue),
                        name=f"profile-queue-worker-{worker.index}-task-{task.index}",
                    )
                    for task in batch
                ]
                for t in sub_threads:
                    t.start()
                for t in sub_threads:
                    t.join()

    threads = [
        threading.Thread(
            target=worker_loop,
            args=(worker, worker_queues[i]),
            name=f"profile-queue-worker-{worker.index}",
        )
        for i, worker in enumerate(generation_workers)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    if fatal_error and not continue_on_error:
        raise fatal_error[0]

    if failures:
        print("[cohort] profile-first failures:", flush=True)
        for item in failures:
            print(f"  - {item}", flush=True)
        return False
    return True


def _batched_source_paths(paths: list[Path], batch_size: int) -> list[list[Path]]:
    if batch_size <= 0:
        return [paths]
    return [paths[i : i + batch_size] for i in range(0, len(paths), batch_size)]


def _run_with_options(args: SimpleNamespace) -> int:
    trend_paths = [p for raw_path in args.trends for p in _resolve_trend_path(raw_path)]
    if args.reverse:
        trend_paths = trend_paths[::-1]
    template_path = args.config_template.expanduser().resolve()
    raw_template = _load_template(template_path)
    raw_template.setdefault("evermemos", {})
    raw_template["evermemos"]["enabled"] = bool(getattr(args, "evermemos", False))
    evermemos_enabled = _evermemos_enabled(raw_template)
    if evermemos_enabled:
        raise RuntimeError(
            "EverMemOS automatic startup is not included in this release; "
            "run cohort generation with EverMemOS disabled."
        )

    rerank_enabled = False
    if args.generation_backend == "local":
        if not args.llm_gpu:
            raise ValueError("llm_gpus is required when generation_backend=local")
        if evermemos_enabled and not args.encode_gpu.strip():
            raise ValueError("encode_gpu is required when generation_backend=local")
        if not evermemos_enabled and (args.encode_gpu.strip() or args.rerank_gpu.strip()):
            print(
                "[cohort] warning: --encode-gpu/--rerank-gpu are ignored unless --evermemos is enabled.",
                flush=True,
            )

        llm_visible_gpu_workers = [_device_to_visible_gpus(raw_gpu) for raw_gpu in args.llm_gpu]
        _validate_generation_workers(llm_visible_gpu_workers)
        encode_visible_gpus = _device_to_visible_gpus(args.encode_gpu) if evermemos_enabled and args.encode_gpu.strip() else ""
        rerank_visible_gpus = (
            _device_to_visible_gpus(args.rerank_gpu)
            if evermemos_enabled and args.rerank_gpu.strip()
            else ""
        )
        rerank_enabled = evermemos_enabled and bool(rerank_visible_gpus)

        _validate_tensor_parallelism(raw_template, llm_visible_gpu_workers)
        generation_workers = _start_model_servers(
            raw_template,
            llm_visible_gpu_workers=llm_visible_gpu_workers,
            encode_visible_gpus=encode_visible_gpus,
            rerank_visible_gpus=rerank_visible_gpus,
        )

        for worker in generation_workers:
            worker.base_raw = _configure_model_endpoints(
                raw_template,
                worker.runtime_visible_gpus,
                rerank_enabled=rerank_enabled,
                generation_base_url_override=worker.base_url,
                llm_base_url_override=worker.base_url if evermemos_enabled else None,
                evermemos_api_base_url_override=worker.evermemos_api_url if evermemos_enabled else None,
            )
    else:
        if args.api_workers <= 0:
            raise ValueError("api_workers must be >= 1")
        if args.llm_gpu or args.encode_gpu.strip() or args.rerank_gpu.strip():
            print(
                "[cohort] warning: --llm-gpu/--encode-gpu/--rerank-gpu are ignored when --generation-backend=api.",
                flush=True,
            )
        api_base_raw = _configure_model_endpoints_api(raw_template)
        generation_workers = _api_generation_workers(api_base_raw, num_workers=args.api_workers)

    primary_base_raw = generation_workers[0].base_raw
    if primary_base_raw is None:
        raise RuntimeError("Generation worker initialization failed to produce a configured endpoint")
    _log_selected_endpoints(primary_base_raw)
    _log_generation_workers(generation_workers)
    if args.generation_backend == "local" and evermemos_enabled and not rerank_enabled:
        print("[cohort] rerank is disabled by default for the shared launcher path; using retrieval_method=rrf.", flush=True)
    if not evermemos_enabled:
        print("[cohort] EverMemOS is disabled; generation will use plain LLM calls without memory access.", flush=True)

    profiles_root = args.profiles_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    generation_model_name = str((primary_base_raw.get("models", {}) or {}).get("model_name", "")).strip()
    output_root_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else _default_model_scoped_root(
            parent=DEFAULT_CONVERSATION_OUTPUT_PARENT,
            model_name=generation_model_name,
        )
    )
    result_root_root = (
        args.result_root.expanduser().resolve()
        if args.result_root is not None
        else _default_model_scoped_root(
            parent=DEFAULT_CONVERSATION_RESULT_PARENT,
            model_name=generation_model_name,
        )
    )
    cache_root_root = (
        args.cache_root.expanduser().resolve()
        if args.cache_root is not None
        else _default_model_scoped_root(
            parent=DEFAULT_CONVERSATION_CACHE_PARENT,
            model_name=generation_model_name,
        )
    )
    print(
        f"[cohort] output_root={output_root_root} result_root={result_root_root} cache_root={cache_root_root}",
        flush=True,
    )
    manifest_path = prepare_source_profile_manifest(
        profiles_root=profiles_root,
        output_root=data_root,
    )
    if args.source_profile:
        source_paths = [
            _resolve_source_profile_path(raw_path, profiles_root=profiles_root)
            for raw_path in args.source_profile
        ]
    else:
        source_paths = discover_source_profile_paths(profiles_root)
    if args.reverse:
        source_paths = source_paths[::-1]
    if args.limit is not None:
        source_paths = source_paths[: args.limit]

    print(
        f"[cohort] profiles={len(source_paths)} manifest={manifest_path}"
        + (" (reversed)" if args.reverse else ""),
        flush=True,
    )

    if args.generation_backend == "api" and args.api_batch_size < 0:
        raise ValueError("api_batch_size must be >= 0")
    if args.generation_backend == "api" and args.api_batch_sleep_seconds < 0:
        raise ValueError("api_batch_sleep_seconds must be >= 0")
    if args.generation_backend == "local" and (args.api_batch_size > 0 or args.api_batch_sleep_seconds > 0):
        print(
            "[cohort] warning: --api-batch-size/--api-batch-sleep-seconds are ignored when --generation-backend=local.",
            flush=True,
        )
    if args.queue_mode == "profile-first" and (args.api_batch_size > 0 or args.api_batch_sleep_seconds > 0):
        print(
            "[cohort] warning: --api-batch-size/--api-batch-sleep-seconds are ignored when --queue-mode=profile-first.",
            flush=True,
        )

    all_succeeded = _run_profile_first_queue(
        generation_workers=generation_workers,
        trend_paths=trend_paths,
        profiles_root=profiles_root,
        data_root=data_root,
        output_root_root=output_root_root,
        result_root_root=result_root_root,
        cache_root_root=cache_root_root,
        source_paths=source_paths,
        num_sessions=args.num_sessions,
        target_turns=args.target_turns,
        force_convert=args.force_convert,
        fresh=args.fresh,
        continue_on_error=args.continue_on_error,
        worker_batch_size=args.worker_batch_size,
        trend_first=args.queue_mode == "trend-first",
        sample_trend_per_profile=args.sample_trend_per_profile,
        sample_seed=args.sample_seed,
        sample_rounds=getattr(args, "sample_rounds", 1),
    )

    return 0 if all_succeeded else 1


def _options_from_config(config: CohortGenerationConfig, *, evermemos: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        trends=list(config.trends),
        queue_mode=config.queue_mode,
        generation_backend=config.generation_backend,
        api_workers=config.api_workers,
        llm_gpu=list(config.llm_gpus),
        encode_gpu=config.encode_gpu,
        rerank_gpu=config.rerank_gpu,
        config_template=Path(config.config_template),
        profiles_root=Path(config.profiles_root),
        source_profile=list(config.source_profiles),
        data_root=Path(config.data_root),
        output_root=Path(config.output_root).expanduser() if config.output_root else None,
        result_root=Path(config.result_root).expanduser() if config.result_root else None,
        cache_root=Path(config.cache_root).expanduser() if config.cache_root else None,
        num_sessions=config.num_sessions,
        target_turns=config.target_turns,
        api_batch_size=config.api_batch_size,
        api_batch_sleep_seconds=config.api_batch_sleep_seconds,
        limit=config.limit,
        reverse=config.reverse,
        worker_batch_size=config.worker_batch_size,
        force_convert=config.force_convert,
        fresh=config.fresh,
        continue_on_error=config.continue_on_error,
        sample_trend_per_profile=config.sample_trend_per_profile,
        sample_seed=config.sample_seed,
        sample_rounds=config.sample_rounds,
        evermemos=evermemos,
    )


def run_from_config(config: CohortGenerationConfig, *, evermemos: bool = False) -> int:
    return _run_with_options(_options_from_config(config, evermemos=evermemos))


def run_from_config_path(path: str | Path, *, evermemos: bool = False) -> int:
    return run_from_config(load_cohort_generation_config(path), evermemos=evermemos)
