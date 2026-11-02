from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from EmoTrack.evaluate.core.artifacts import artifact_namespace_subdir
from EmoTrack.evaluate.methods.registry import EVALUATION_METHODS
from EmoTrack.utils.paths import REPO_ROOT


DEFAULT_BASE_URL = "http://localhost:8100/v1"
DEFAULT_MODEL = "Qwen/Qwen3.5-35B-A3B"
DEFAULT_EXPERIMENT_TAG = "default"
DEFAULT_EVALUATION_SUITE_CONFIG_PATH = REPO_ROOT / "configs" / "evaluate" / "default.yaml"


@dataclass
class EvaluationDatasetConfig:
    name: str
    input_root: str
    transcript_view: str = "full"
    split_file: str = ""
    include_history: bool = False


@dataclass
class EvaluationMethodConfig:
    name: str
    deterministic: bool = False
    sample_count: int = 5
    feature_concurrency: int = 4
    eval_concurrency: int = 1
    question_bank_path: str = ""
    rf_n_estimators_grid: list[int] = field(default_factory=lambda: [100, 200, 300])
    rf_max_depth_grid: list[int] = field(default_factory=lambda: [10, 20, 30])
    inner_val_fraction: float = 1 / 8
    mode: str = "api"
    checkpoint: str = ""
    extract_features_only: bool = False
    embed_cache_dir: str = ""
    aida_cache_path: str = ""
    aida_extra_cache_path: str = ""
    include_history: bool = True
    use_trend: bool = False
    use_confidence: bool = False
    prompt_template: str = ""
    vllm_enforce_eager: bool | None = None
    vllm_worker_multiproc_method: str = ""
    extract_extra_only: bool = False
    inference_seed: int = 0
    aida_prompts_path: str = ""
    retrieval_model_name: str = "OpenMatch/cocodr-base-msmarco"
    retrieval_batch_size: int = 32
    retrieval_max_length: int = 256
    retrieval_pooling_strategy: str = "mean"
    retrieval_torch_dtype: str = "auto"
    retrieval_trust_remote_code: bool = False
    embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2"
    embedding_batch_size: int = 32
    embedding_max_length: int = 256
    embedding_pooling_strategy: str = "mean"
    embedding_torch_dtype: str = "auto"
    embedding_trust_remote_code: bool = False
    ensembert_max_posts: int = 30
    ensembert_min_words: int = 2
    ensembert_document_source: str = "client_utterances"
    ensembert_dense_units: int = 256
    ensembert_num_heads: int = 2
    ensembert_learning_rate: float = 0.001
    ensembert_weight_decay: float = 0.0001
    ensembert_num_epochs: int = 50
    ensembert_batch_size: int = 64
    ensembert_patience: int = 8
    ensembert_out_loss_weight: float = 1.0
    ensembert_soft_loss_weight: float = 1.0
    ensembert_meta_loss_weight: float = 1.0
    ensembert_prediction_head: str = "meta"
    ensembert_item_description_path: str = "configs/evaluate/ensembert/phq8_item_descriptions.json"
    ensembert_method_prefix: str = "ensembert"
    dual_encoder_repo_root: str = "cache/external/dual-encoder-model"
    dual_encoder_batch_size: int = 2
    dual_encoder_num_epochs: int = 200
    dual_encoder_patience: int = 20
    dual_encoder_learning_rate: float = 3.0e-4
    dual_encoder_weight_decay: float = 0.01
    dual_encoder_transformer_pretrained_id: str = "sentence-transformers/all-mpnet-base-v2"
    dual_encoder_pooling_strategy: str = "mean"
    dual_encoder_torch_dtype: str = "auto"
    dual_encoder_trust_remote_code: bool = False
    dual_encoder_warm_start: bool = True
    dual_encoder_method_prefix: str = "dual_encoder"
    milintsevich_repo_root: str = "cache/external/hierarchical-depression-symptom-classifier"
    milintsevich_python_executable: str = ""
    milintsevich_num_iters: int = 100
    milintsevich_batch_size: int = 8
    milintsevich_early_stopping_patience: int = 20
    milintsevich_method_prefix: str = "milintsevich"


@dataclass
class EvaluationLLMVLLMConfig:
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    kv_cache_memory_bytes: int | None = None
    max_model_len: int = 8192
    max_num_batched_tokens: int = 2048
    max_num_seqs: int = 256
    dtype: str = "bfloat16"
    seed: int = 0
    trust_remote_code: bool = False
    enforce_eager: bool = True
    enable_chunked_prefill: bool = True
    enable_prefix_caching: bool = True
    gdn_prefill_backend: str = ""


@dataclass
class EvaluationLLMConfig:
    backend: str = "local"
    base_url: str = DEFAULT_BASE_URL
    model_name: str = DEFAULT_MODEL
    api_key: str = ""
    api_key_env: str = ""
    worker_visible_gpus: list[str] = field(default_factory=list)
    startup_timeout_seconds: int = 600
    log_root: str = "out/eval/services"
    vllm: EvaluationLLMVLLMConfig = field(default_factory=EvaluationLLMVLLMConfig)


@dataclass
class EvaluationOutputConfig:
    experiment_tag: str = DEFAULT_EXPERIMENT_TAG
    artifact_namespace: str = ""


@dataclass
class EvaluationConfig:
    dataset: EvaluationDatasetConfig
    method: EvaluationMethodConfig
    llm: EvaluationLLMConfig
    output: EvaluationOutputConfig


@dataclass
class EvaluationSuiteConfig:
    dataset: EvaluationDatasetConfig
    llm: EvaluationLLMConfig
    output: EvaluationOutputConfig
    default_methods: list[str] = field(default_factory=list)
    method_configs: dict[str, EvaluationMethodConfig] = field(default_factory=dict)


def load_evaluation_config(path: str | Path) -> EvaluationConfig:
    resolved = Path(path).expanduser().resolve()
    raw: dict[str, Any] = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    dataset = EvaluationDatasetConfig(**_read_section(raw, "dataset"))
    method = EvaluationMethodConfig(**_read_section(raw, "method"))
    llm_raw = _read_section(raw, "llm")
    vllm = EvaluationLLMVLLMConfig(**dict(_read_optional_section(llm_raw, "vllm")))
    llm_payload = dict(llm_raw)
    llm_payload.pop("vllm", None)
    llm = EvaluationLLMConfig(vllm=vllm, **llm_payload)
    output = EvaluationOutputConfig(**dict(_read_optional_section(raw, "output")))
    config = EvaluationConfig(dataset=dataset, method=method, llm=llm, output=output)
    validate_evaluation_config(config)
    return config


def load_evaluation_suite_config(path: str | Path = DEFAULT_EVALUATION_SUITE_CONFIG_PATH) -> EvaluationSuiteConfig:
    resolved = Path(path).expanduser().resolve()
    raw: dict[str, Any] = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    dataset = EvaluationDatasetConfig(**_read_section(raw, "dataset"))
    llm_raw = _read_section(raw, "llm")
    vllm = EvaluationLLMVLLMConfig(**dict(_read_optional_section(llm_raw, "vllm")))
    llm_payload = dict(llm_raw)
    llm_payload.pop("vllm", None)
    llm = EvaluationLLMConfig(vllm=vllm, **llm_payload)
    output = EvaluationOutputConfig(**dict(_read_optional_section(raw, "output")))

    default_methods_raw = raw.get("default_methods", [])
    if default_methods_raw is None:
        default_methods_raw = []
    if not isinstance(default_methods_raw, list):
        raise ValueError("default_methods must be a list")
    default_methods = [str(method_name).strip() for method_name in default_methods_raw if str(method_name).strip()]

    method_configs_raw = _read_section(raw, "method_configs")
    method_configs: dict[str, EvaluationMethodConfig] = {}
    for raw_method_name, method_payload in method_configs_raw.items():
        method_name = str(raw_method_name).strip()
        if not method_name:
            raise ValueError("method_configs cannot contain a blank method name")
        if not isinstance(method_payload, dict):
            raise ValueError(f"method_configs[{method_name!r}] must be a mapping")
        payload = dict(method_payload)
        payload.pop("name", None)
        method_configs[method_name] = EvaluationMethodConfig(name=method_name, **payload)

    if not default_methods:
        default_methods = list(method_configs)
    suite_config = EvaluationSuiteConfig(
        dataset=dataset,
        llm=llm,
        output=output,
        default_methods=default_methods,
        method_configs=method_configs,
    )
    validate_evaluation_suite_config(suite_config)
    return suite_config


def validate_evaluation_config(config: EvaluationConfig) -> EvaluationConfig:
    dataset = config.dataset
    method = config.method
    llm = config.llm
    artifact_namespace_subdir(config.output.artifact_namespace)

    if dataset.name not in {"conversation", "daic_woz", "edaic"}:
        raise ValueError(f"Unsupported evaluation dataset: {dataset.name}")
    if dataset.name == "edaic" and not (method.name == "emotrack" or (method.name == "aida" and method.extract_features_only)):
        raise ValueError("E-DAIC supports frozen EmoTrack evaluation and feature extraction only.")
    if dataset.transcript_view not in {"full", "client-only"}:
        raise ValueError(f"Unsupported transcript_view: {dataset.transcript_view}")
    if method.name not in EVALUATION_METHODS:
        raise ValueError(f"Unsupported evaluation method: {method.name}")
    if method.mode not in {"api", "embedding", "aida"}:
        raise ValueError(f"Unsupported emotrack mode: {method.mode}")
    if method.vllm_worker_multiproc_method not in {"", "fork", "spawn"}:
        raise ValueError("method.vllm_worker_multiproc_method must be one of '', 'fork', or 'spawn'")
    if method.sample_count <= 0:
        raise ValueError("sample_count must be >= 1")
    if method.feature_concurrency <= 0:
        raise ValueError("feature_concurrency must be >= 1")
    if method.eval_concurrency <= 0:
        raise ValueError("eval_concurrency must be >= 1")
    if not 0 < method.inner_val_fraction < 1:
        raise ValueError("inner_val_fraction must be between 0 and 1")
    if method.name == "ensembert":
        if method.ensembert_max_posts <= 0:
            raise ValueError("ensembert_max_posts must be >= 1")
        if method.ensembert_min_words <= 0:
            raise ValueError("ensembert_min_words must be >= 1")
        if method.ensembert_document_source not in {"client_utterances", "all_utterances"}:
            raise ValueError("ensembert_document_source must be client_utterances or all_utterances")
        if method.ensembert_dense_units <= 0:
            raise ValueError("ensembert_dense_units must be >= 1")
        if method.ensembert_num_heads <= 0:
            raise ValueError("ensembert_num_heads must be >= 1")
        if method.ensembert_learning_rate <= 0:
            raise ValueError("ensembert_learning_rate must be > 0")
        if method.ensembert_weight_decay < 0:
            raise ValueError("ensembert_weight_decay must be >= 0")
        if method.ensembert_num_epochs <= 0:
            raise ValueError("ensembert_num_epochs must be >= 1")
        if method.ensembert_batch_size <= 0:
            raise ValueError("ensembert_batch_size must be >= 1")
        if method.ensembert_patience <= 0:
            raise ValueError("ensembert_patience must be >= 1")
        if method.embedding_batch_size <= 0:
            raise ValueError("embedding_batch_size must be >= 1")
        if method.embedding_max_length <= 0:
            raise ValueError("embedding_max_length must be >= 1")
        if method.retrieval_batch_size <= 0:
            raise ValueError("retrieval_batch_size must be >= 1")
        if method.retrieval_max_length <= 0:
            raise ValueError("retrieval_max_length must be >= 1")
        if not method.retrieval_model_name:
            raise ValueError("retrieval_model_name is required for EnsemBERT")
        if not method.embedding_model_name:
            raise ValueError("embedding_model_name is required for EnsemBERT")
        if method.retrieval_pooling_strategy not in {"mean", "last_token"}:
            raise ValueError("retrieval_pooling_strategy must be one of: mean, last_token")
        if method.embedding_pooling_strategy not in {"mean", "last_token"}:
            raise ValueError("embedding_pooling_strategy must be one of: mean, last_token")
        if method.retrieval_torch_dtype not in {"auto", "float16", "float32", "bfloat16"}:
            raise ValueError("retrieval_torch_dtype must be one of: auto, float32, float16, bfloat16")
        if method.embedding_torch_dtype not in {"auto", "float16", "float32", "bfloat16"}:
            raise ValueError("embedding_torch_dtype must be one of: auto, float32, float16, bfloat16")
        if method.ensembert_prediction_head not in {"out", "soft", "meta"}:
            raise ValueError("ensembert_prediction_head must be one of: out, soft, meta")
        if not method.ensembert_method_prefix.strip():
            raise ValueError("ensembert_method_prefix is required for EnsemBERT")
    if method.name == "dual_encoder":
        if not method.dual_encoder_repo_root.strip():
            raise ValueError("dual_encoder_repo_root is required for Dual Encoder")
        if method.dual_encoder_batch_size <= 0:
            raise ValueError("dual_encoder_batch_size must be >= 1")
        if method.dual_encoder_num_epochs <= 0:
            raise ValueError("dual_encoder_num_epochs must be >= 1")
        if method.dual_encoder_patience <= 0:
            raise ValueError("dual_encoder_patience must be >= 1")
        if method.dual_encoder_learning_rate <= 0:
            raise ValueError("dual_encoder_learning_rate must be > 0")
        if method.dual_encoder_weight_decay < 0:
            raise ValueError("dual_encoder_weight_decay must be >= 0")
        if not method.dual_encoder_transformer_pretrained_id.strip():
            raise ValueError("dual_encoder_transformer_pretrained_id is required for Dual Encoder")
        if method.dual_encoder_pooling_strategy not in {"mean", "last_token"}:
            raise ValueError("dual_encoder_pooling_strategy must be one of: mean, last_token")
        if method.dual_encoder_torch_dtype not in {"auto", "float16", "float32", "bfloat16"}:
            raise ValueError("dual_encoder_torch_dtype must be one of: auto, float32, float16, bfloat16")
        if not method.dual_encoder_method_prefix.strip():
            raise ValueError("dual_encoder_method_prefix is required for Dual Encoder")
    if method.name == "milintsevich":
        if not method.milintsevich_repo_root.strip():
            raise ValueError("milintsevich_repo_root is required")
        if method.milintsevich_num_iters <= 0:
            raise ValueError("milintsevich_num_iters must be >= 1")
        if method.milintsevich_batch_size <= 0:
            raise ValueError("milintsevich_batch_size must be >= 1")
        if method.milintsevich_early_stopping_patience < 0:
            raise ValueError("milintsevich_early_stopping_patience must be >= 0")
        if not method.milintsevich_method_prefix.strip():
            raise ValueError("milintsevich_method_prefix is required")
    if not method.rf_n_estimators_grid or any(value <= 0 for value in method.rf_n_estimators_grid):
        raise ValueError("rf_n_estimators_grid must be a non-empty list of positive integers")
    if not method.rf_max_depth_grid or any(value <= 0 for value in method.rf_max_depth_grid):
        raise ValueError("rf_max_depth_grid must be a non-empty list of positive integers")
    if not dataset.input_root:
        raise ValueError("dataset.input_root is required")
    if llm.backend not in {"local", "api", "cache_only"}:
        raise ValueError(f"Unsupported llm backend: {llm.backend}")
    if llm.backend == "cache_only":
        if not llm.model_name:
            raise ValueError("llm.model_name is required with backend=cache_only (needed for cache key matching)")
    elif llm.startup_timeout_seconds <= 0:
        raise ValueError("llm.startup_timeout_seconds must be >= 1")
    if llm.backend not in ("cache_only", "local") and _config_method_requires_llm(method.name, method.mode):
        if not llm.base_url:
            raise ValueError("llm.base_url is required")
    if llm.backend != "cache_only" and _config_method_requires_llm(method.name, method.mode):
        if not llm.model_name:
            raise ValueError("llm.model_name is required")
    if llm.backend == "local" and _config_method_requires_llm(method.name, method.mode):
        if not llm.worker_visible_gpus:
            raise ValueError("llm.worker_visible_gpus is required for local llm-backed evaluation")
        if any(not str(item or "").strip() for item in llm.worker_visible_gpus):
            raise ValueError("llm.worker_visible_gpus entries must be non-empty strings")
    return config


def validate_evaluation_suite_config(config: EvaluationSuiteConfig) -> EvaluationSuiteConfig:
    if not config.method_configs:
        raise ValueError("method_configs must define at least one evaluation method")
    seen_default_methods: set[str] = set()
    for method_name in config.default_methods:
        normalized = str(method_name).strip()
        if not normalized:
            raise ValueError("default_methods cannot contain blank values")
        if normalized in seen_default_methods:
            raise ValueError(f"default_methods contains a duplicate entry: {normalized}")
        seen_default_methods.add(normalized)
        if normalized not in config.method_configs:
            raise ValueError(f"default_methods references an unknown method: {normalized}")
    for method_name, method_config in config.method_configs.items():
        if method_config.name != method_name:
            raise ValueError(
                f"method_configs[{method_name!r}] must not override its own name "
                f"(found {method_config.name!r})"
            )
        validate_evaluation_config(
            EvaluationConfig(
                dataset=config.dataset,
                method=method_config,
                llm=config.llm,
                output=config.output,
            )
        )
    return config


def _read_section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Missing required config section: {key}")
    return dict(value)


def _read_optional_section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{key}' must be a mapping")
    return dict(value)


def _config_method_requires_llm(method_name: str, mode: str) -> bool:
    if method_name in {"aida", "lmiq", "psycheval"}:
        return True
    return method_name == "emotrack" and mode == "api"
