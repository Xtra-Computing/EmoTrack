from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from EmoTrack.evaluate.core.config import EvaluationConfig
from EmoTrack.evaluate.core.context import PreparedLLMWorker
from EmoTrack.evaluate.core.scores import SessionExampleRow, apply_split_manifest
from EmoTrack.evaluate.datasets import (
    build_conversation_split_manifest,
    load_conversation_rows,
    load_daic_woz_rows,
)
from EmoTrack.evaluate.utils import resolve_api_key_with_fallback
from EmoTrack.inference.service import build_vllm_server_args, ensure_openai_like_endpoints


def prepare_llm_workers(
    config: EvaluationConfig,
    *,
    label: str | None = None,
) -> list[PreparedLLMWorker] | None:
    if not _method_requires_llm(config):
        return None
    api_key = resolve_api_key_with_fallback(
        config.llm.api_key_env,
        explicit_value=config.llm.api_key,
        allow_missing=config.llm.backend == "local",
    )
    base_urls = ensure_openai_like_endpoints(
        backend=config.llm.backend,
        base_url=config.llm.base_url,
        model_name=config.llm.model_name,
        api_key=api_key,
        label=label or f"evaluation[{config.method.name}]",
        worker_visible_gpus=config.llm.worker_visible_gpus,
        vllm_args=build_vllm_server_args(asdict(config.llm.vllm)),
        startup_timeout_seconds=config.llm.startup_timeout_seconds,
        log_root=Path(config.llm.log_root).expanduser().resolve(),
    )
    return [
        PreparedLLMWorker(
            base_url=base_url,
            model_name=config.llm.model_name,
            api_key=api_key,
            worker_index=index,
        )
        for index, base_url in enumerate(base_urls, start=1)
    ]


def _method_requires_llm(config: EvaluationConfig) -> bool:
    if config.llm.backend == "cache_only":
        return False
    if config.method.name in {"aida", "lmiq", "psycheval"}:
        return True
    return config.method.name == "emotrack" and config.method.mode == "api"


def load_dataset_rows(
    *,
    config: EvaluationConfig,
    input_root: Path,
) -> list[SessionExampleRow]:
    if config.dataset.name == "conversation":
        return load_conversation_rows(
            input_root,
            transcript_view=config.dataset.transcript_view,
            include_history=config.dataset.include_history,
        )
    return load_daic_woz_rows(
        input_root,
        transcript_view=config.dataset.transcript_view,
        split_file=_resolve_dataset_split_file(config),
        dataset_name=config.dataset.name,
    )


def _resolve_dataset_split_file(config: EvaluationConfig) -> Path:
    configured = str(config.dataset.split_file or "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(config.dataset.input_root).expanduser().resolve() / "train_val_test_split.json"


def partition_rows_for_evaluation(
    *,
    config: EvaluationConfig,
    rows: list[SessionExampleRow],
    split_manifest: dict[str, object] | None = None,
) -> tuple[
    list[SessionExampleRow],
    list[SessionExampleRow],
    list[SessionExampleRow],
    list[SessionExampleRow],
    dict[str, object] | None,
]:
    if config.dataset.name == "conversation":
        manifest = split_manifest or build_conversation_split_manifest(
            rows,
            split_file=_resolve_dataset_split_file(config),
        )
        assigned_rows = apply_split_manifest(rows, manifest)
        train_rows = [row for row in assigned_rows if row.split == "train"]
        val_rows = [row for row in assigned_rows if row.split == "val"]
        test_rows = [row for row in assigned_rows if row.split == "test"]
        return assigned_rows, train_rows, val_rows, test_rows, manifest

    train_rows = [row for row in rows if row.split == "train"]
    val_rows = [row for row in rows if row.split in {"dev", "val"}]
    test_rows = [row for row in rows if row.split == "test"]
    return rows, train_rows, val_rows, test_rows, None


def load_rows_for_evaluation(
    *,
    config: EvaluationConfig,
    input_root: Path,
    split_manifest: dict[str, object] | None = None,
) -> tuple[
    list[SessionExampleRow],
    list[SessionExampleRow],
    list[SessionExampleRow],
    list[SessionExampleRow],
    dict[str, object] | None,
]:
    rows = load_dataset_rows(config=config, input_root=input_root)
    return partition_rows_for_evaluation(config=config, rows=rows, split_manifest=split_manifest)
