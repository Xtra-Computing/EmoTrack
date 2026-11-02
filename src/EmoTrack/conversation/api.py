from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Iterable
from urllib.parse import urlparse

import httpx

from EmoTrack.client.evermem_agent import EverMemClientAgent
from EmoTrack.conversation.config import ConversationConfig, load_conversation_config, load_structured_config
from EmoTrack.conversation.model import ReasoningClient
from EmoTrack.conversation.orchestrator import ConversationOrchestrator
from EmoTrack.counselor.agent import CounselorProfile
from EmoTrack.counselor.evermem_agent import EverMemCounselorAgent
from EmoTrack.data.profile import ensure_conversation_profile_artifacts
from EmoTrack.evermemos.client import NullEverMemOSClient
from EmoTrack.latent_variable.trend import LatentTrendPlan
from EmoTrack.utils.paths import REPO_ROOT


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return deduped


def _resolve_config_path(raw_path: str, project_root: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        resolved = path.resolve()
        if resolved.exists():
            return resolved
        raise FileNotFoundError(f"Config path not found: {resolved}")

    for root in _dedupe_paths([Path.cwd(), project_root]):
        candidate = (root / path).resolve()
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Config path not found: {raw_path}")


def _candidate_roots(config_path: Path, project_root: Path) -> list[Path]:
    return _dedupe_paths([config_path.parent, config_path.parent.parent, Path.cwd(), project_root])


def _select_workspace_root(config_path: Path, cfg_paths: list[str], project_root: Path) -> Path:
    candidates = _candidate_roots(config_path, project_root)
    best = candidates[0]
    best_score = -1
    for root in candidates:
        score = 0
        for path_str in cfg_paths:
            if not path_str:
                continue
            path = Path(path_str).expanduser()
            candidate = path if path.is_absolute() else (root / path)
            if candidate.exists():
                score += 1
        if score > best_score:
            best_score = score
            best = root
    return best


def _resolve_data_path(path_str: str, workspace_root: Path, project_root: Path) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        resolved = path.resolve()
        if resolved.exists():
            return resolved
        raise FileNotFoundError(f"Data path not found: {resolved}")

    for root in _dedupe_paths([workspace_root, workspace_root.parent, Path.cwd(), project_root]):
        candidate = (root / path).resolve()
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Data path not found: {path_str}")


def _resolve_optional_data_path(path_str: str | None, workspace_root: Path, project_root: Path) -> Path | None:
    normalized = str(path_str or "").strip()
    if not normalized:
        return None
    return _resolve_data_path(normalized, workspace_root, project_root)


def _infer_source_profiles_root(source_path: Path, project_root: Path) -> Path:
    resolved = source_path.resolve()
    canonical_profiles_root = (project_root / "profiles").resolve()
    if canonical_profiles_root == resolved or canonical_profiles_root in resolved.parents:
        return canonical_profiles_root
    for ancestor in resolved.parents:
        if ancestor.name == "profiles":
            return ancestor
    if len(resolved.parents) >= 2:
        return resolved.parents[1]
    return resolved.parent


def _normalize_profile_tag(raw: str) -> str:
    tag = raw.strip().lower()
    if not tag:
        return ""
    if tag.startswith("client_"):
        tag = tag[len("client_") :]
    for suffix in ("_latent_seed", "_hidden_state", "_seed", "_profile", "_config"):
        if tag.endswith(suffix):
            tag = tag[: -len(suffix)]
            break
    return re.sub(r"[^a-z0-9]+", "_", tag).strip("_")


def _derive_run_id(cfg: ConversationConfig) -> str:
    if cfg.run.run_id:
        return cfg.run.run_id
    for candidate in (
        Path(str(cfg.paths.hidden_state_path or "")).stem,
        Path(str(cfg.paths.client_profile_path or "")).stem,
        Path(str(cfg.paths.source_profile_path or "")).stem,
    ):
        normalized = _normalize_profile_tag(candidate)
        if normalized:
            return normalized
    return "conversation"


def _group_id(run_id: str) -> str:
    return f"emotrack_{run_id}"


def _progress_group_id(progress_path: Path) -> str | None:
    if not progress_path.exists():
        return None
    try:
        raw = json.loads(progress_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(raw, dict):
        return None
    group_id = str(raw.get("group_id") or "").strip()
    return group_id or None


def _resolve_group_id(cfg: ConversationConfig, *, run_name: str, run_dir: Path) -> str:
    progress_group_id = _progress_group_id(run_dir / "progress.json") if cfg.run.resume else None
    if progress_group_id:
        return progress_group_id
    explicit_group_id = str(cfg.run.memory_group_id or "").strip()
    if explicit_group_id:
        return explicit_group_id
    return _group_id(run_name)


def _port_from_url(base_url: str) -> int:
    parsed = urlparse(base_url)
    if parsed.port is not None:
        return parsed.port
    if parsed.scheme == "https":
        return 443
    return 80



def _configure_vllm_devices(cfg: ConversationConfig) -> None:
    if cfg.models.source != "vllm":
        return
    gpu_value = (cfg.runtime.gpus or "").strip()
    if gpu_value:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_value


def _resolve_generated_env_path(cfg: ConversationConfig, project_root: Path) -> Path:
    configured = str(cfg.runtime.generated_env_path or "").strip()
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = (project_root / path).resolve()
        return path
    return project_root / "cache" / "runtime.env"


def _resolve_online_runtime_paths(
    cfg: ConversationConfig,
    *,
    workspace_root: Path,
    project_root: Path,
) -> tuple[Path, Path, Path, Path]:
    source_profile_path = _resolve_optional_data_path(cfg.paths.source_profile_path, workspace_root, project_root)
    counselor_profile_path = _resolve_optional_data_path(cfg.paths.counselor_profile_path, workspace_root, project_root)
    trend_path = _resolve_optional_data_path(cfg.paths.trend_path, workspace_root, project_root)
    client_profile_path = _resolve_optional_data_path(cfg.paths.client_profile_path, workspace_root, project_root)
    hidden_state_path = _resolve_optional_data_path(cfg.paths.hidden_state_path, workspace_root, project_root)

    if counselor_profile_path is None:
        raise ValueError("Online conversation requires paths.counselor_profile_path.")
    if trend_path is None:
        raise ValueError("Online conversation requires paths.trend_path.")

    if source_profile_path is not None:
        prepared = ensure_conversation_profile_artifacts(
            source_profile_path,
            output_root=project_root / "data" / "conversation_profiles",
            profiles_root=_infer_source_profiles_root(source_profile_path, project_root),
        )
        client_profile_path = prepared.client_profile_path.resolve()
        hidden_state_path = prepared.hidden_state_path.resolve()
        cfg.paths.source_profile_path = str(source_profile_path)

    if client_profile_path is None or hidden_state_path is None:
        raise ValueError(
            "Online conversation requires either paths.source_profile_path or both "
            "paths.client_profile_path and paths.hidden_state_path."
        )

    cfg.paths.client_profile_path = str(client_profile_path)
    cfg.paths.counselor_profile_path = str(counselor_profile_path)
    cfg.paths.hidden_state_path = str(hidden_state_path)
    cfg.paths.trend_path = str(trend_path)
    return client_profile_path, counselor_profile_path, hidden_state_path, trend_path


def run_online_conversation_sessions(
    *,
    config_path: str = "configs/longitudinal_local.yaml",
    num_sessions: int | None = None,
    run_id: str | None = None,
    source_profile_path: str | None = None,
    client_profile_path: str | None = None,
    counselor_profile_path: str | None = None,
    hidden_state_path: str | None = None,
    trend_path: str | None = None,
) -> Path:
    project_root = REPO_ROOT
    resolved_config_path = _resolve_config_path(config_path, project_root)
    cfg = load_conversation_config(resolved_config_path, workspace=project_root)
    if num_sessions is not None:
        cfg.simulation.num_sessions = num_sessions
    if run_id is not None:
        cfg.run.run_id = run_id
    if source_profile_path is not None:
        cfg.paths.source_profile_path = source_profile_path
    if client_profile_path is not None:
        cfg.paths.client_profile_path = client_profile_path
    if counselor_profile_path is not None:
        cfg.paths.counselor_profile_path = counselor_profile_path
    if hidden_state_path is not None:
        cfg.paths.hidden_state_path = hidden_state_path
    if trend_path is not None:
        cfg.paths.trend_path = trend_path

    workspace_root = _select_workspace_root(
        resolved_config_path,
        [
            cfg.paths.source_profile_path,
            cfg.paths.client_profile_path,
            cfg.paths.counselor_profile_path,
            cfg.paths.hidden_state_path,
            cfg.paths.trend_path or "",
        ],
        project_root,
    )

    client_profile_path, counselor_profile_path, hidden_state_path, trend_path = _resolve_online_runtime_paths(
        cfg,
        workspace_root=workspace_root,
        project_root=project_root,
    )

    client_profile = load_structured_config(client_profile_path)
    counselor_profile_data = load_structured_config(counselor_profile_path)
    hidden_state = load_structured_config(hidden_state_path)
    trend_plan = LatentTrendPlan.from_path(trend_path)
    _configure_vllm_devices(cfg)

    run_name = _derive_run_id(cfg)
    output_root = Path(cfg.run.output_root).expanduser()
    if not output_root.is_absolute():
        output_root = (project_root / output_root).resolve()
    cache_root = Path(cfg.run.cache_root).expanduser()
    if not cache_root.is_absolute():
        cache_root = (project_root / cache_root).resolve()

    run_dir = (output_root / run_name).resolve()
    cache_dir = (cache_root / run_name).resolve()
    env_path = _resolve_generated_env_path(cfg, project_root)
    evermem_client = NullEverMemOSClient()
    generator = None
    try:
        generator = ReasoningClient(cfg.models)
        client_agent = EverMemClientAgent(
            generator=generator,
            profile_data=client_profile,
            hidden_state=hidden_state,
            trend_plan=trend_plan,
            run_seed=cfg.run.seed,
        )
        counselor_agent = EverMemCounselorAgent(
            generator=generator,
            profile=CounselorProfile.from_dict(counselor_profile_data),
            conversation_profile=client_profile,
            run_seed=cfg.run.seed,
        )
        group_id = _resolve_group_id(cfg, run_name=run_name, run_dir=run_dir)
        orchestrator = ConversationOrchestrator(
            config=cfg,
            run_dir=run_dir,
            cache_dir=cache_dir,
            evermem_env_path=env_path,
            group_id=group_id,
            client_profile=client_profile,
            counselor_profile=counselor_profile_data,
            client_agent=client_agent,
            counselor_agent=counselor_agent,
            evermem_client=evermem_client,
            model_metadata={
                "model_source": cfg.models.source,
                "model_name": cfg.models.model_name,
                "base_url": cfg.models.base_url,
                "thinking_enabled": cfg.models.thinking.enabled,
                "embedding_model": "",
                "gpus": cfg.runtime.gpus,
                "source_profile_path": str(cfg.paths.source_profile_path or ""),
                "trend_path": str(trend_path) if trend_path else "",
                "trend_targets": [target.path for target in trend_plan.targets],
            },
        )
        return orchestrator.run()
    finally:
        if generator is not None:
            generator.close()
        evermem_client.close()


def run_conversation_sessions(
    *,
    config_path: str = "configs/longitudinal_local.yaml",
    num_sessions: int | None = None,
    run_id: str | None = None,
    source_profile_path: str | None = None,
    client_profile_path: str | None = None,
    counselor_profile_path: str | None = None,
    hidden_state_path: str | None = None,
    trend_path: str | None = None,
) -> Path:
    return run_online_conversation_sessions(
        config_path=config_path,
        num_sessions=num_sessions,
        run_id=run_id,
        source_profile_path=source_profile_path,
        client_profile_path=client_profile_path,
        counselor_profile_path=counselor_profile_path,
        hidden_state_path=hidden_state_path,
        trend_path=trend_path,
    )
