from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
import os
from pathlib import Path

from EmoTrack.evaluate.orchestration import (
    _method_requires_llm,
    load_dataset_rows,
    partition_rows_for_evaluation,
    prepare_llm_workers,
)
from EmoTrack.evaluate.methods.registry import EVALUATION_METHODS
from EmoTrack.evaluate.core.config import (
    DEFAULT_EXPERIMENT_TAG,
    DEFAULT_EVALUATION_SUITE_CONFIG_PATH,
    EvaluationConfig,
    EvaluationDatasetConfig,
    EvaluationLLMConfig,
    EvaluationLLMVLLMConfig,
    EvaluationMethodConfig,
    EvaluationOutputConfig,
    EvaluationSuiteConfig,
    load_evaluation_suite_config,
    validate_evaluation_config,
)
from EmoTrack.evaluate.runners import run_method
from EmoTrack.evaluate.core.scores import (
    SessionExampleRow,
    build_metrics_payload,
)

from EmoTrack.evaluate.utils import (
    eval_output_root,
    write_json,
)
from EmoTrack.inference.service import cleanup_managed_vllm_servers


DEFAULT_RUN_SUMMARY_NAME = "eval_run_summary.json"


def _prepare_deterministic_environment(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def add_cli_arguments(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    parser.description = (
        "Run selected evaluations with the canonical suite config, one shared split manifest, "
        "and one shared worker pool."
    )
    parser.add_argument(
        "--suite-config",
        type=Path,
        default=None,
        help=(
            f"Evaluation suite config path. Defaults to {DEFAULT_EVALUATION_SUITE_CONFIG_PATH}."
        ),
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help=(
            f"Override dataset.input_root from {DEFAULT_EVALUATION_SUITE_CONFIG_PATH}. "
            "The input dataset should live under result/."
        ),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=sorted(EVALUATION_METHODS),
        default=None,
        help="Methods to run. Defaults to default_methods in the canonical evaluation config.",
    )
    parser.add_argument(
        "--dataset",
        choices=["conversation", "daic_woz"],
        default=None,
        help="Override dataset.name from the canonical evaluation config.",
    )
    parser.add_argument(
        "--experiment-tag",
        default="",
        help=(
            "Override output.experiment_tag from the canonical evaluation config. "
            "If blank, falls back to the input-root directory name and then "
            f"{DEFAULT_EXPERIMENT_TAG!r}."
        ),
    )
    parser.add_argument(
        "--artifact-namespace",
        choices=["detector", "transfer"],
        default=None,
        help=(
            "Override output.artifact_namespace from the canonical evaluation config. "
            "Use detector for the flat seed directory."
        ),
    )
    parser.add_argument(
        "--transcript-view",
        choices=["full", "client-only"],
        default=None,
        help="Override dataset.transcript_view from the canonical evaluation config.",
    )
    parser.add_argument(
        "--backend",
        choices=["local", "api"],
        default=None,
        help="Override llm.backend from the canonical evaluation config.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Override llm.base_url from the canonical evaluation config.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Override llm.model_name from the canonical evaluation config.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Override llm.api_key from the canonical evaluation config.",
    )
    parser.add_argument(
        "--api-key-env",
        default=None,
        help="Override llm.api_key_env from the canonical evaluation config.",
    )
    parser.add_argument(
        "--gpus",
        nargs="+",
        default=None,
        help=(
            "Local worker GPU groups. Each token becomes one worker, for example "
            "--gpus 2 3 5 6. This overrides llm.worker_visible_gpus."
        ),
    )
    parser.add_argument(
        "--startup-timeout-seconds",
        type=int,
        default=None,
        help="Override llm.startup_timeout_seconds from the canonical evaluation config.",
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=None,
        help="Override llm.log_root from the canonical evaluation config.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="Override llm.vllm.tensor_parallel_size from the canonical evaluation config.",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=None,
        help=(
            "Override per-worker request concurrency. Applies to AIDA/LMIQ feature extraction "
            "and PsychEval/EmoTrack evaluation unless overridden by method-specific flags."
        ),
    )
    parser.add_argument(
        "--feature-concurrency",
        type=int,
        default=None,
        help="Override per-worker request concurrency for AIDA/LMIQ.",
    )
    parser.add_argument(
        "--eval-concurrency",
        type=int,
        default=None,
        help="Override per-worker request concurrency for PsychEval and EmoTrack api mode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override method.inference_seed for all selected methods.",
    )
    deterministic_group = parser.add_mutually_exclusive_group()
    deterministic_group.add_argument(
        "--deterministic",
        dest="deterministic",
        action="store_true",
        default=None,
        help=(
            "Enable optional deterministic controls beyond the outer --seed sweep. "
            "All methods still use method.inference_seed/--seed."
        ),
    )
    deterministic_group.add_argument(
        "--no-deterministic",
        dest="deterministic",
        action="store_false",
        help="Disable optional deterministic controls even if enabled in the suite config.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-evaluate methods even when detector artifacts already exist.",
    )
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="Reuse existing per-method metrics files instead of recomputing those methods.",
    )
    return parser


def run_from_parsed_args(args: argparse.Namespace) -> int:
    if bool(getattr(args, "overwrite", False)) and bool(getattr(args, "resume_existing", False)):
        raise ValueError("--overwrite cannot be combined with --resume-existing")
    if getattr(args, "deterministic", None) is True and not bool(getattr(args, "overwrite", False)):
        raise ValueError("--deterministic requires --overwrite so existing detector artifacts are not reused.")
    return asyncio.run(_run_selected_methods_from_parsed_args(args))


async def _run_selected_methods_from_parsed_args(args: argparse.Namespace) -> int:
    suite_config_path = getattr(args, "suite_config", None)
    suite_config = (
        load_evaluation_suite_config(suite_config_path)
        if suite_config_path is not None
        else load_evaluation_suite_config()
    )
    methods = _normalize_methods(args.methods, suite_config=suite_config)
    input_root = _resolve_input_root(args, suite_config=suite_config)
    if not input_root.exists():
        raise FileNotFoundError(f"Evaluation input root does not exist: {input_root}")

    experiment_tag = _resolve_experiment_tag(
        input_root=input_root,
        raw_tag=str(args.experiment_tag or ""),
        suite_tag=suite_config.output.experiment_tag,
    )
    selected_configs = [
        _build_config_from_suite(
            suite_config=suite_config,
            args=args,
            method_name=method_name,
            input_root=input_root,
            experiment_tag=experiment_tag,
        )
        for method_name in methods
    ]
    for config in selected_configs:
        validate_evaluation_config(config)

    deterministic_config = next((config for config in selected_configs if config.method.deterministic), None)
    if deterministic_config is not None:
        if not bool(getattr(args, "overwrite", False)):
            raise ValueError("deterministic evaluation requires --overwrite so existing detector artifacts are not reused.")
        _prepare_deterministic_environment(deterministic_config.method.inference_seed)

    base_config = selected_configs[0]
    rows, train_rows, val_rows, test_rows, split_manifest = _load_rows_with_shared_split(
        config=base_config,
        input_root=input_root,
    )
    if not rows:
        raise ValueError(f"No scorable rows found under {input_root}")
    if not test_rows:
        raise ValueError("Evaluation has no test rows after applying the split policy.")

    shared_root = _shared_eval_root(dataset_name=base_config.dataset.name, experiment_tag=experiment_tag)
    resume_existing = bool(getattr(args, "resume_existing", False))
    method_plan: list[dict[str, object]] = []
    for config in selected_configs:
        out_root = eval_output_root(
            dataset_name=config.dataset.name,
            experiment_tag=config.output.experiment_tag,
            method_name=config.method.name,
            seed=config.method.inference_seed,
        )
        metrics_name = "metrics_test.json" if config.dataset.name == "conversation" else "metrics.json"
        metrics_path = out_root / metrics_name
        metrics: dict[str, object] | None = None
        if resume_existing and metrics_path.exists():
            raw_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            if not isinstance(raw_metrics, dict):
                raise RuntimeError(f"Existing metrics file must contain a JSON object: {metrics_path}")
            metrics = raw_metrics
            print(f"[resume] reusing existing {config.method.name} metrics: {metrics_path}")
        method_plan.append(
            {
                "config": config,
                "out_root": out_root,
                "metrics_name": metrics_name,
                "metrics_path": metrics_path,
                "metrics": metrics,
            }
        )

    llm_workers = None
    worker_count = 0

    method_summaries: list[dict[str, object]] = []
    for plan in method_plan:
        config = plan["config"]
        if not isinstance(config, EvaluationConfig):
            raise TypeError(f"Invalid method plan config: {config!r}")
        out_root = plan["out_root"]
        if not isinstance(out_root, Path):
            raise TypeError(f"Invalid method plan output root: {out_root!r}")
        metrics_path = plan["metrics_path"]
        if not isinstance(metrics_path, Path):
            raise TypeError(f"Invalid method plan metrics path: {metrics_path!r}")
        metrics = plan["metrics"]
        if metrics is None:
            if _method_requires_llm(config):
                if llm_workers is None:
                    llm_workers = prepare_llm_workers(config, label="evaluation[methods]")
                    worker_count = max(worker_count, len(llm_workers or []))
            elif llm_workers is not None:
                cleanup_managed_vllm_servers()
                llm_workers = None
            run_scores = await run_method(
                dataset_name=config.dataset.name,
                train_rows=train_rows,
                val_rows=val_rows,
                test_rows=test_rows,
                out_root=out_root,
                config=config,
                llm_workers=llm_workers,
                overwrite=bool(getattr(args, "overwrite", False)),
            )
            metrics = build_metrics_payload(run_scores)
            metrics_path = write_json(metrics_path, metrics)
        method_summaries.append(
            {
                "method": config.method.name,
                "artifact_namespace": config.output.artifact_namespace,
                "output_root": str(out_root),
                "metrics_path": str(metrics_path),
                "metrics": metrics,
            }
        )

    if llm_workers is not None:
        cleanup_managed_vllm_servers()
        llm_workers = None

    summary = {
        "dataset": base_config.dataset.name,
        "input_root": str(input_root),
        "experiment_tag": experiment_tag,
        "artifact_namespace": base_config.output.artifact_namespace,
        "methods": method_summaries,
        "split_manifest_path": (
            str(_dataset_split_manifest_path(base_config)) if split_manifest is not None else None
        ),
        "worker_count": worker_count,
    }
    write_json(shared_root / DEFAULT_RUN_SUMMARY_NAME, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _resolve_input_root(args: argparse.Namespace, *, suite_config: EvaluationSuiteConfig) -> Path:
    raw_input_root = getattr(args, "input_root", None)
    if raw_input_root is not None:
        return Path(raw_input_root).expanduser().resolve()
    configured = str(suite_config.dataset.input_root or "").strip()
    if not configured:
        raise ValueError(
            f"Evaluation input root is missing. Set dataset.input_root in {DEFAULT_EVALUATION_SUITE_CONFIG_PATH} "
            "or pass --input-root."
        )
    return Path(configured).expanduser().resolve()


def _normalize_methods(
    raw_methods: list[str] | tuple[str, ...] | None,
    *,
    suite_config: EvaluationSuiteConfig,
) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()
    configured_methods = raw_methods or suite_config.default_methods
    for raw_method in configured_methods:
        method_name = str(raw_method or "").strip().lower()
        if not method_name:
            continue
        if method_name not in EVALUATION_METHODS:
            raise ValueError(
                f"Unsupported evaluation method: {method_name}. Expected one of {', '.join(EVALUATION_METHODS)}"
            )
        if method_name not in suite_config.method_configs:
            raise ValueError(
                f"Method {method_name!r} is not configured in {DEFAULT_EVALUATION_SUITE_CONFIG_PATH}"
            )
        if method_name in seen:
            continue
        seen.add(method_name)
        selected.append(method_name)
    if not selected:
        raise ValueError("At least one baseline method must be selected")
    return selected


def _resolve_experiment_tag(*, input_root: Path, raw_tag: str, suite_tag: str) -> str:
    tag = raw_tag.strip()
    if tag:
        return tag
    configured_tag = suite_tag.strip()
    if configured_tag:
        return configured_tag
    fallback = input_root.name.strip()
    return fallback or DEFAULT_EXPERIMENT_TAG


def _build_config_from_suite(
    *,
    suite_config: EvaluationSuiteConfig,
    args: argparse.Namespace,
    method_name: str,
    input_root: Path,
    experiment_tag: str,
) -> EvaluationConfig:
    return EvaluationConfig(
        dataset=_build_dataset_config(
            suite_config=suite_config,
            args=args,
            input_root=input_root,
        ),
        method=_build_method_config(
            suite_config=suite_config,
            args=args,
            method_name=method_name,
        ),
        llm=_build_llm_config(
            suite_config=suite_config,
            args=args,
        ),
        output=EvaluationOutputConfig(
            experiment_tag=experiment_tag,
            artifact_namespace=_resolve_artifact_namespace(args, suite_config=suite_config),
        ),
    )


def _resolve_artifact_namespace(
    args: argparse.Namespace,
    *,
    suite_config: EvaluationSuiteConfig,
) -> str:
    raw_namespace = getattr(args, "artifact_namespace", None)
    if raw_namespace is None:
        return str(suite_config.output.artifact_namespace or "")
    namespace = str(raw_namespace or "").strip()
    return "" if namespace == "detector" else namespace


def _build_dataset_config(
    *,
    suite_config: EvaluationSuiteConfig,
    args: argparse.Namespace,
    input_root: Path,
) -> EvaluationDatasetConfig:
    dataset_name = str(args.dataset or suite_config.dataset.name)
    split_file = ""
    if dataset_name in {"conversation", "daic_woz"}:
        configured = str(suite_config.dataset.split_file or "").strip()
        split_file = configured or str(input_root / "train_val_test_split.json")
    return EvaluationDatasetConfig(
        name=dataset_name,
        input_root=str(input_root),
        transcript_view=str(args.transcript_view or suite_config.dataset.transcript_view),
        split_file=split_file,
        include_history=bool(suite_config.dataset.include_history),
    )


def _build_method_config(
    *,
    suite_config: EvaluationSuiteConfig,
    args: argparse.Namespace,
    method_name: str,
) -> EvaluationMethodConfig:
    configured = suite_config.method_configs.get(method_name)
    if configured is None:
        raise ValueError(f"Method {method_name!r} is not configured in {DEFAULT_EVALUATION_SUITE_CONFIG_PATH}")
    method_config = EvaluationMethodConfig(**asdict(configured))
    if args.parallel is not None:
        if method_name in {"aida", "lmiq"}:
            method_config.feature_concurrency = int(args.parallel)
        else:
            method_config.eval_concurrency = int(args.parallel)
    if args.feature_concurrency is not None and method_name in {"aida", "lmiq"}:
        method_config.feature_concurrency = int(args.feature_concurrency)
    if args.eval_concurrency is not None and method_name in {"psycheval", "emotrack"}:
        method_config.eval_concurrency = int(args.eval_concurrency)
    if args.seed is not None:
        method_config.inference_seed = int(args.seed)
    deterministic = getattr(args, "deterministic", None)
    if deterministic is not None:
        method_config.deterministic = bool(deterministic)
    return method_config


def _build_llm_config(
    *,
    suite_config: EvaluationSuiteConfig,
    args: argparse.Namespace,
) -> EvaluationLLMConfig:
    llm_payload = asdict(suite_config.llm)
    raw_vllm = llm_payload.pop("vllm")
    llm_config = EvaluationLLMConfig(
        vllm=EvaluationLLMVLLMConfig(**raw_vllm),
        **llm_payload,
    )
    if args.backend is not None:
        llm_config.backend = str(args.backend)
    if args.base_url is not None:
        llm_config.base_url = str(args.base_url)
    if args.model_name is not None:
        llm_config.model_name = str(args.model_name)
    if args.api_key is not None:
        llm_config.api_key = str(args.api_key)
    if args.api_key_env is not None:
        llm_config.api_key_env = str(args.api_key_env)
    if args.gpus is not None:
        llm_config.worker_visible_gpus = _normalize_gpu_groups(args.gpus)
    if args.startup_timeout_seconds is not None:
        llm_config.startup_timeout_seconds = int(args.startup_timeout_seconds)
    if args.log_root is not None:
        llm_config.log_root = str(Path(args.log_root).expanduser())
    if args.tensor_parallel_size is not None:
        llm_config.vllm.tensor_parallel_size = int(args.tensor_parallel_size)
    return llm_config


def _normalize_gpu_groups(raw_groups: list[str] | tuple[str, ...] | None) -> list[str]:
    groups: list[str] = []
    if raw_groups is None:
        return groups
    for raw_group in raw_groups:
        tokens = [token.strip() for token in str(raw_group or "").split(",") if token.strip()]
        if not tokens:
            continue
        groups.append(",".join(tokens))
    return groups


def _shared_eval_root(*, dataset_name: str, experiment_tag: str) -> Path:
    return eval_output_root(
        dataset_name=dataset_name,
        experiment_tag=experiment_tag,
        method_name=EVALUATION_METHODS[0],
    ).parents[1]


def _dataset_split_manifest_path(config: EvaluationConfig) -> Path | None:
    configured = str(config.dataset.split_file or "").strip()
    if not configured:
        return None
    return Path(configured).expanduser().resolve()


def _load_rows_with_shared_split(
    *,
    config: EvaluationConfig,
    input_root: Path,
) -> tuple[
    list[SessionExampleRow],
    list[SessionExampleRow],
    list[SessionExampleRow],
    list[SessionExampleRow],
    dict[str, object] | None,
]:
    rows = load_dataset_rows(config=config, input_root=input_root)
    if config.dataset.name != "conversation":
        return partition_rows_for_evaluation(config=config, rows=rows)

    assigned_rows, train_rows, val_rows, test_rows, resolved_manifest = partition_rows_for_evaluation(
        config=config,
        rows=rows,
    )
    manifest_path = _dataset_split_manifest_path(config)
    if resolved_manifest is not None and manifest_path is not None:
        _validate_conversation_manifest(rows=rows, manifest=resolved_manifest, source_path=manifest_path)
    return assigned_rows, train_rows, val_rows, test_rows, resolved_manifest


def _validate_conversation_manifest(
    *,
    rows: list[SessionExampleRow],
    manifest: dict[str, object],
    source_path: Path,
) -> None:
    if str(manifest.get("dataset_name") or "") != "conversation":
        raise ValueError(f"Split manifest {source_path} is not for the conversation dataset")
    raw_runs = manifest.get("runs")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise ValueError(f"Split manifest {source_path} has no runs")

    row_run_ids = {row.run_id for row in rows}
    manifest_run_ids: list[str] = []
    split_values: set[str] = set()
    for item in raw_runs:
        if not isinstance(item, dict):
            raise ValueError(f"Split manifest {source_path} contains a non-object run entry: {item!r}")
        run_id = str(item.get("run_id") or "").strip()
        split = str(item.get("split") or "").strip()
        if not run_id:
            raise ValueError(f"Split manifest {source_path} contains a blank run_id entry")
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Split manifest {source_path} has an invalid split for run {run_id!r}: {split!r}")
        manifest_run_ids.append(run_id)
        split_values.add(split)

    if len(set(manifest_run_ids)) != len(manifest_run_ids):
        raise ValueError(f"Split manifest {source_path} contains duplicate run_id entries")
    if row_run_ids != set(manifest_run_ids):
        missing = sorted(row_run_ids - set(manifest_run_ids))
        extra = sorted(set(manifest_run_ids) - row_run_ids)
        raise ValueError(
            f"Split manifest {source_path} does not match the current dataset runs. "
            f"Missing: {missing}; extra: {extra}"
        )
    if not {"train", "test"}.issubset(split_values):
        raise ValueError(f"Split manifest {source_path} must contain both train and test runs")
