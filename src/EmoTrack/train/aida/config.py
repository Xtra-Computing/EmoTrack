"""Configuration for AIDA-feature attention models."""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class AidaAttnAggConfig:
    """Config for AIDA-feature attention aggregator."""

    # AIDA feature input
    num_aida_features: int = 23
    aida_groups: list[int] = field(default_factory=lambda: [8, 5, 10])

    # Architecture
    d_model: int = 64
    nhead: int = 4
    dim_feedforward: int = 256
    num_encoder_layers: int = 1
    num_decoder_layers: int = 2
    dropout: float = 0.1
    scalar_mlp_hidden: int = 32
    use_feature_identity: bool = True
    skip_mode: str = "none"
    identity_mode: str = "both"

    # Task
    num_items: int = 8
    head_type: str = "sigmoid"

    # Training
    num_epochs: int = 50
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    patience: int = 8
    ordinal_nll_weight: float = 1.0
    total_huber_weight: float = 0.05
    item_loss_weight: float = 0.0
    deterministic_training: bool = False

    # Data
    aida_cache_path: str = ""
    aida_extra_cache_path: str = ""
    dataset_path: str = ""
    split_file: str = ""
    split_by: str = "run_id"
    eval_split: float = 0.20
    seed: int = 42
    daic_oversample: int = 5

    # Output
    output_dir: str = "out/emotrack_aida_attnagg"
    result_name: str = "emotrack_aida_attnagg"


@dataclass
class AidaGRUConfig:
    """Config for AIDA-feature GRU model."""

    # AIDA feature input
    num_aida_features: int = 23
    aida_groups: list[int] = field(default_factory=lambda: [8, 5, 10])

    # Architecture
    d_model: int = 64
    nhead: int = 4
    dim_feedforward: int = 256
    num_encoder_layers: int = 1
    num_decoder_layers: int = 2
    dropout: float = 0.1
    scalar_mlp_hidden: int = 32
    use_feature_identity: bool = True

    # GRU-specific
    gru_hidden_dim: int = 64
    gru_num_layers: int = 1
    gru_dropout: float = 0.0
    history_dropout: float = 0.2
    max_lag: int = 10

    # Task
    num_items: int = 8
    head_type: str = "sigmoid"

    # Training
    num_epochs: int = 50
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    patience: int = 8
    ordinal_nll_weight: float = 1.0
    total_huber_weight: float = 0.05
    item_loss_weight: float = 0.0
    deterministic_training: bool = False

    # Data
    aida_cache_path: str = ""
    aida_extra_cache_path: str = ""
    dataset_path: str = ""
    split_file: str = ""
    split_by: str = "run_id"
    eval_split: float = 0.20
    seed: int = 42
    daic_oversample: int = 5

    # Warm start
    aida_attnagg_checkpoint: str = ""
    freeze_aggregator: bool = False

    # Output
    output_dir: str = "out/emotrack_aida_gru"
    result_name: str = "emotrack_aida_gru"


@dataclass
class AidaStackAttnConfig:
    """Config for AIDA-feature StackAttn model."""

    # AIDA feature input
    num_aida_features: int = 23
    aida_groups: list[int] = field(default_factory=lambda: [8, 5, 10])

    # Architecture
    d_model: int = 64
    nhead: int = 4
    dim_feedforward: int = 256
    num_encoder_layers: int = 1
    num_decoder_layers: int = 2
    dropout: float = 0.1
    scalar_mlp_hidden: int = 32
    use_feature_identity: bool = True
    skip_mode: str = "none"
    identity_mode: str = "both"

    # StackAttn-specific
    history_num_layers: int = 1
    history_dropout: float = 0.2

    # Task
    num_items: int = 8
    head_type: str = "sigmoid"

    # Training
    num_epochs: int = 50
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    patience: int = 8
    ordinal_nll_weight: float = 1.0
    total_huber_weight: float = 0.05
    item_loss_weight: float = 0.0
    deterministic_training: bool = False

    # Data
    aida_cache_path: str = ""
    aida_extra_cache_path: str = ""
    dataset_path: str = ""
    split_file: str = ""
    split_by: str = "run_id"
    eval_split: float = 0.20
    seed: int = 42
    daic_oversample: int = 5
    inference_seed: int = 0

    # Warm start
    aida_attnagg_checkpoint: str = ""
    freeze_aggregator: bool = False

    # Output
    output_dir: str = "out/emotrack_aida_stackattn"
    result_name: str = "emotrack_aida_stackattn"


@dataclass
class AidaEmbedAttnAggConfig:
    """Config for hybrid AIDA + sentence-embedding attention aggregator."""

    # AIDA feature input
    num_aida_features: int = 23
    aida_groups: list[int] = field(default_factory=lambda: [8, 5, 10])

    # Architecture
    d_model: int = 64
    nhead: int = 4
    dim_feedforward: int = 256
    num_encoder_layers: int = 1
    num_decoder_layers: int = 2
    dropout: float = 0.1
    scalar_mlp_hidden: int = 32
    use_feature_identity: bool = True

    # Sentence embedding
    backbone_embed_dim: int = 4096
    max_sentences: int = 384

    # Task
    num_items: int = 8
    head_type: str = "sigmoid"

    # Training
    num_epochs: int = 50
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    patience: int = 8
    ordinal_nll_weight: float = 1.0
    total_huber_weight: float = 0.05
    item_loss_weight: float = 0.0
    deterministic_training: bool = False

    # Data
    aida_cache_path: str = ""
    aida_extra_cache_path: str = ""
    embed_cache_dir: str = "out/embed_cache"
    dataset_path: str = ""
    split_file: str = ""
    split_by: str = "run_id"
    eval_split: float = 0.20
    seed: int = 42
    daic_oversample: int = 1

    # Output
    output_dir: str = "out/emotrack_hybrid_embed_attnagg"
    result_name: str = "emotrack_hybrid_embed_attnagg"


@dataclass
class AidaEmbedHybridConfig:
    """Config for hybrid AIDA + full sentence-embedding model."""

    # AIDA feature input
    num_aida_features: int = 23
    aida_groups: list[int] = field(default_factory=lambda: [8, 5, 10])

    # Architecture
    d_model: int = 64
    nhead: int = 4
    dim_feedforward: int = 256
    num_encoder_layers: int = 1
    num_decoder_layers: int = 2
    dropout: float = 0.1
    scalar_mlp_hidden: int = 32
    use_feature_identity: bool = True
    skip_mode: str = "none"
    identity_mode: str = "both"
    input_mode: str = "full"
    readout_mode: str = "item_queries"

    # Previous-session sentence memory
    prev_sent_mode: str = "none"  # none, summary_only, summary_retrieval
    ps_num_slots: int = 4  # number of compressed memory slots
    history_dropout: float = 0.1

    # Sentence embedding
    backbone_embed_dim: int = 4096
    max_sentences: int = 384
    speaker_view: str = "client"
    psyche_transcript_root: str = ""
    daic_transcript_root: str = ""

    # Task
    num_items: int = 8
    head_type: str = "sigmoid"

    # Training
    num_epochs: int = 50
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    patience: int = 8
    ordinal_nll_weight: float = 1.0
    total_huber_weight: float = 0.05
    fn_loss_weight: float = 0.0
    fn_threshold: float = 10.0
    item_loss_weight: float = 0.5
    deterministic_training: bool = False

    # Data
    aida_cache_path: str = ""
    aida_extra_cache_path: str = ""
    embed_cache_dir: str = "out/embed_cache"
    dataset_path: str = ""
    split_file: str = ""
    split_by: str = "run_id"
    eval_split: float = 0.20
    seed: int = 42
    daic_oversample: int = 1
    inference_seed: int = 0

    # Warmstart: load compatible weights from a shallower checkpoint (strict=False)
    warmstart_checkpoint: str = ""

    # Output
    output_dir: str = "out/emotrack_hybrid"
    result_name: str = "emotrack_hybrid"


def _load(cls, path: str | Path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    valid = {f.name for f in dataclasses.fields(cls)}
    unknown = set(raw) - valid
    if unknown:
        raise ValueError(
            f"Unknown config keys for {cls.__name__} in {path}: {sorted(unknown)}"
        )
    return cls(**{k: v for k, v in raw.items() if k in valid})


def load_attnagg_config(path: str | Path) -> AidaAttnAggConfig:
    return _load(AidaAttnAggConfig, path)


def load_gru_config(path: str | Path) -> AidaGRUConfig:
    return _load(AidaGRUConfig, path)


def load_stackattn_config(path: str | Path) -> AidaStackAttnConfig:
    return _load(AidaStackAttnConfig, path)


def load_embed_attnagg_config(path: str | Path) -> AidaEmbedAttnAggConfig:
    return _load(AidaEmbedAttnAggConfig, path)


def load_hybrid_config(path: str | Path) -> AidaEmbedHybridConfig:
    config = _load(AidaEmbedHybridConfig, path)
    if not math.isfinite(config.fn_loss_weight) or config.fn_loss_weight < 0:
        raise ValueError("fn_loss_weight must be finite and nonnegative")
    if not math.isfinite(config.fn_threshold) or not 0 <= config.fn_threshold <= 24:
        raise ValueError("fn_threshold must be finite and in [0, 24]")
    return config
