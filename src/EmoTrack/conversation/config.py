from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from string import Template
from typing import Any, get_args, get_origin, get_type_hints
import json
import os

from dotenv import load_dotenv
import yaml

from EmoTrack.inference.backend import VLLMRuntimeConfig
from EmoTrack.utils.env import first_env


DEFAULT_MODEL_NAME = "Qwen/Qwen3.5-35B-A3B"
DEFAULT_MODEL_BASE_URL = "http://localhost:8100/v1"
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"


def _default_thinking_request_extras() -> dict[str, Any]:
    return {
        "chat_template_kwargs": {"enable_thinking": True},
        "top_p": 0.95,
        "top_k": 20,
        "repetition_penalty": 1.1,
        "frequency_penalty": 0.3,
    }


@dataclass
class RunConfig:
    seed: int = 42
    output_root: str = "out/longitudinal"
    cache_root: str = "cache/conversations"
    run_id: str | None = None
    memory_group_id: str | None = None
    resume: bool = True


@dataclass
class SimulationConfig:
    num_sessions: int = 6
    target_turns: int = 20
    retrieval_top_k: int = 4
    retrieval_threshold: float = 0.12
    retrieval_method: str = "rrf"
    counselor_temperature: float = 0.6
    counselor_max_tokens: int = 512
    client_temperature: float = 0.6
    client_max_tokens: int = 512


@dataclass
class PathsConfig:
    source_profile_path: str = ""
    client_profile_path: str = ""
    counselor_profile_path: str = "profiles/seeds/counselor/default.json"
    hidden_state_path: str = ""
    trend_path: str | None = ""


@dataclass
class ComplaintShiftConfig:
    enabled: bool = False


@dataclass
class FeaturesConfig:
    complaint_shift: ComplaintShiftConfig = field(default_factory=ComplaintShiftConfig)


@dataclass
class ThinkingConfig:
    enabled: bool = True
    budget_tokens: int = 0
    strip_tags: bool = True
    request_extras: dict[str, Any] = field(default_factory=_default_thinking_request_extras)


@dataclass
class ModelConfig:
    source: str = "api"
    api_key: str = ""
    base_url: str = DEFAULT_MODEL_BASE_URL
    model_name: str = DEFAULT_MODEL_NAME
    timeout_seconds: int = 120
    max_retries: int = 3
    thinking: ThinkingConfig = field(default_factory=ThinkingConfig)
    vllm: VLLMRuntimeConfig = field(default_factory=VLLMRuntimeConfig)


@dataclass
class RuntimeConfig:
    gpus: str = ""
    generated_env_path: str = ""
    progress_position: int = 0


@dataclass
class ConversationConfig:
    run: RunConfig
    simulation: SimulationConfig
    paths: PathsConfig
    models: ModelConfig = field(default_factory=ModelConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_structured_file(path: Path) -> dict[str, Any]:
    text = Template(path.read_text(encoding="utf-8")).safe_substitute(os.environ)
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    elif suffix == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"Unsupported file extension: {suffix}")
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}, got {type(data).__name__}")
    return data


def _resolve_nested_dataclass(annotation: Any) -> type[Any] | None:
    if isinstance(annotation, type) and is_dataclass(annotation):
        return annotation

    origin = get_origin(annotation)
    if origin is None:
        return None

    for arg in get_args(annotation):
        if isinstance(arg, type) and is_dataclass(arg):
            return arg
    return None


def _nested_dataclass(cls: type[Any], raw: dict[str, Any] | None) -> Any:
    if raw is None:
        return cls()
    if not isinstance(raw, dict):
        raise TypeError(f"Expected mapping for {cls.__name__}, got {type(raw).__name__}")

    type_hints = get_type_hints(cls)
    field_names = {item.name for item in fields(cls)}
    unknown = sorted(set(raw) - field_names)
    if unknown:
        raise TypeError(f"Unexpected fields for {cls.__name__}: {', '.join(unknown)}")

    payload: dict[str, Any] = {}
    for item in fields(cls):
        if item.name not in raw:
            continue
        value = raw[item.name]
        nested_cls = _resolve_nested_dataclass(type_hints.get(item.name, item.type))
        if nested_cls is not None and isinstance(value, dict):
            payload[item.name] = _nested_dataclass(nested_cls, value)
        else:
            payload[item.name] = value
    return cls(**payload)


def load_structured_config(path: str | Path) -> dict[str, Any]:
    return _load_structured_file(Path(path))


def load_conversation_config(path: str | Path, workspace: str | Path | None = None) -> ConversationConfig:
    root = Path(workspace or os.getenv("EMOTRACK_WORKSPACE") or Path.cwd()).resolve()
    load_dotenv(root / ".env", override=False)

    raw = _load_structured_file(Path(path))
    for section in ("run", "simulation", "paths"):
        if section not in raw:
            raise ValueError(f"Missing required config section: {section}")
        if not isinstance(raw[section], dict):
            raise ValueError(f"Config section '{section}' must be a mapping")

    cfg = ConversationConfig(
        run=_nested_dataclass(RunConfig, raw["run"]),
        simulation=_nested_dataclass(SimulationConfig, raw["simulation"]),
        paths=_nested_dataclass(PathsConfig, raw["paths"]),
        models=_nested_dataclass(ModelConfig, raw.get("models")),
        runtime=_nested_dataclass(RuntimeConfig, raw.get("runtime")),
        features=FeaturesConfig(
            complaint_shift=_nested_dataclass(
                ComplaintShiftConfig,
                (raw.get("features") or {}).get("complaint_shift"),
            ),
        ),
    )

    if cfg.simulation.target_turns <= 0:
        raise ValueError("simulation.target_turns must be > 0")
    cfg.models.source = (cfg.models.source or first_env("EMOTRACK_MODEL_SOURCE", default="api")).strip().lower()
    if cfg.models.source not in {"api", "vertex", "vllm"}:
        raise ValueError(f"Unsupported models.source: {cfg.models.source}")

    if cfg.models.source == "api":
        cfg.models.api_key = cfg.models.api_key or first_env(
            "EMOTRACK_MODEL_API_KEY",
            "EMOTRACK_API_KEY",
            "ANNA_ENGINE_API_KEY",
            "NEBIUS_API_KEY",
            default="",
        )
        cfg.models.base_url = cfg.models.base_url or first_env(
            "EMOTRACK_MODEL_BASE_URL",
            "EMOTRACK_BASE_URL",
            "ANNA_ENGINE_BASE_URL",
            "NEBIUS_BASE_URL",
            default=DEFAULT_MODEL_BASE_URL,
        )
        cfg.models.model_name = cfg.models.model_name or first_env(
            "EMOTRACK_MODEL_NAME",
            "ANNA_ENGINE_MODEL_NAME",
            "NEBIUS_MODEL_NAME",
            default=DEFAULT_MODEL_NAME,
        )
    elif cfg.models.source == "vertex":
        cfg.models.api_key = cfg.models.api_key or first_env(
            "EMOTRACK_VERTEX_API_KEY",
            default="",
        )
        cfg.models.base_url = cfg.models.base_url or first_env(
            "EMOTRACK_VERTEX_BASE_URL",
            default="",
        )
        cfg.models.model_name = cfg.models.model_name or first_env(
            "EMOTRACK_VERTEX_MODEL_NAME",
            default=DEFAULT_MODEL_NAME,
        )
    else:
        cfg.models.model_name = cfg.models.model_name or first_env(
            "EMOTRACK_INFERENCE_MODEL",
            "EMOTRACK_MODEL_NAME",
            "ANNA_ENGINE_MODEL_NAME",
            "NEBIUS_MODEL_NAME",
            default=DEFAULT_MODEL_NAME,
        )
    return cfg
