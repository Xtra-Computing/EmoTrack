from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class CohortGenerationConfig:
    trends: list[str]
    queue_mode: str = "trend-first"
    generation_backend: str = "local"
    api_workers: int = 1
    llm_gpus: list[str] = field(default_factory=list)
    encode_gpu: str = ""
    rerank_gpu: str = ""
    config_template: str = "configs/longitudinal_local.yaml"
    profiles_root: str = "profiles"
    source_profiles: list[str] = field(default_factory=list)
    data_root: str = "data/conversation_profiles"
    output_root: str = ""
    result_root: str = ""
    cache_root: str = ""
    num_sessions: int = 6
    target_turns: int = 20
    api_batch_size: int = 0
    api_batch_sleep_seconds: float = 0.0
    limit: int | None = None
    reverse: bool = False
    worker_batch_size: int = 1
    force_convert: bool = False
    fresh: bool = False
    continue_on_error: bool = False
    sample_trend_per_profile: bool = False
    sample_seed: int = 42
    sample_rounds: int = 5


def load_cohort_generation_config(path: str | Path) -> CohortGenerationConfig:
    resolved = Path(path).expanduser().resolve()
    raw: dict[str, Any] = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    config = CohortGenerationConfig(trends=[])
    for key, value in raw.items():
        if not hasattr(config, key):
            raise ValueError(f"Unknown CohortGenerationConfig field: {key}")
        setattr(config, key, value)
    if not config.trends:
        raise ValueError("CohortGenerationConfig.trends must be a non-empty list")
    if config.queue_mode not in {"trend-first", "profile-first"}:
        raise ValueError(f"Unsupported queue_mode: {config.queue_mode}")
    if config.generation_backend not in {"local", "api"}:
        raise ValueError(f"Unsupported generation_backend: {config.generation_backend}")
    if config.api_workers <= 0:
        raise ValueError("api_workers must be >= 1")
    if config.worker_batch_size <= 0:
        raise ValueError("worker_batch_size must be >= 1")
    if config.api_batch_size < 0:
        raise ValueError("api_batch_size must be >= 0")
    if config.api_batch_sleep_seconds < 0:
        raise ValueError("api_batch_sleep_seconds must be >= 0")
    return config
