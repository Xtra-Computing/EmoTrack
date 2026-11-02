from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from EmoTrack.evaluate.author_dual_encoder_daic import AuthorDualEncoderConfig, run_author_dual_encoder_daic
from EmoTrack.evaluate.methods._public_daic_export import split_train_validation_rows, write_public_daic_source
from EmoTrack.evaluate.core.config import EvaluationMethodConfig
from EmoTrack.evaluate.core.scores import SessionExampleRow

from EmoTrack.evaluate.utils import write_json


@dataclass(frozen=True)
class DualEncoderRunConfig:
    repo_root: str = "cache/external/dual-encoder-model"
    seed: int = 0
    batch_size: int = 2
    num_epochs: int = 200
    patience: int = 20
    learning_rate: float = 3.0e-4
    weight_decay: float = 0.01
    transformer_pretrained_id: str = "sentence-transformers/all-mpnet-base-v2"
    pooling_strategy: str = "mean"
    torch_dtype: str = "auto"
    trust_remote_code: bool = False
    warm_start: bool = True
    include_history: bool = False
    method_prefix: str = "dual_encoder"
    deterministic: bool = False


@dataclass(frozen=True)
class DualEncoderFitPredictResult:
    method_name: str
    scoring_model: str
    predictions_by_row_key: dict[str, float]
    metadata_path: Path


def config_from_method_config(
    method_config: EvaluationMethodConfig,
    *,
    include_history: bool,
) -> DualEncoderRunConfig:
    return DualEncoderRunConfig(
        repo_root=method_config.dual_encoder_repo_root,
        seed=method_config.inference_seed,
        batch_size=method_config.dual_encoder_batch_size,
        num_epochs=method_config.dual_encoder_num_epochs,
        patience=method_config.dual_encoder_patience,
        learning_rate=method_config.dual_encoder_learning_rate,
        weight_decay=method_config.dual_encoder_weight_decay,
        transformer_pretrained_id=method_config.dual_encoder_transformer_pretrained_id,
        pooling_strategy=method_config.dual_encoder_pooling_strategy,
        torch_dtype=method_config.dual_encoder_torch_dtype,
        trust_remote_code=method_config.dual_encoder_trust_remote_code,
        warm_start=method_config.dual_encoder_warm_start,
        include_history=include_history,
        method_prefix=method_config.dual_encoder_method_prefix,
        deterministic=method_config.deterministic,
    )


def fit_predict_dual_encoder(
    *,
    train_rows: list[SessionExampleRow],
    val_rows: list[SessionExampleRow],
    test_rows: list[SessionExampleRow],
    out_root: Path,
    config: DualEncoderRunConfig,
    inner_val_fraction: float,
) -> DualEncoderFitPredictResult:
    fit_rows, validation_rows, split_summary = split_train_validation_rows(
        train_rows=train_rows,
        val_rows=val_rows,
        inner_val_fraction=inner_val_fraction,
        seed=config.seed,
    )
    runtime_root = out_root / "runtime"
    export = write_public_daic_source(
        train_rows=fit_rows,
        val_rows=validation_rows,
        test_rows=test_rows,
        source_root=runtime_root / "public_daic_source",
    )
    author_config = AuthorDualEncoderConfig(
        seed=config.seed,
        batch_size=config.batch_size,
        num_epoch=config.num_epochs,
        es_patience=config.patience,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        transformer_pretrained_id=config.transformer_pretrained_id,
        pooling_strategy=config.pooling_strategy,
        torch_dtype=config.torch_dtype,
        trust_remote_code=config.trust_remote_code,
        deterministic=config.deterministic,
    )
    summary = run_author_dual_encoder_daic(
        source_root=export.source_root,
        author_repo_root=Path(config.repo_root),
        output_root=runtime_root / "author_dual_encoder",
        config=author_config,
        warm_start=config.warm_start,
    )
    test_metrics = summary.get("test_metrics")
    if not isinstance(test_metrics, dict):
        raise RuntimeError("Dual Encoder author runner did not return test_metrics")
    raw_predictions = test_metrics.get("predictions")
    if not isinstance(raw_predictions, list):
        raise RuntimeError("Dual Encoder author runner did not return test predictions")

    predictions_by_row_key: dict[str, float] = {}
    for raw_prediction in raw_predictions:
        if not isinstance(raw_prediction, dict):
            raise RuntimeError(f"Invalid Dual Encoder prediction row: {raw_prediction!r}")
        participant_id = int(raw_prediction["subject_id"])
        row_key = export.row_key_by_participant_id[participant_id]
        predictions_by_row_key[row_key] = float(raw_prediction["predicted_total"])

    method_name = f"{config.method_prefix}_{'all' if config.include_history else 'now'}"
    scoring_model = f"DualEncoder[{config.transformer_pretrained_id}]"
    metadata_path = write_json(
        out_root / "model" / "dual_encoder_model_meta.json",
        {
            "method_name": method_name,
            "scoring_model": scoring_model,
            "config": asdict(config),
            "author_config": asdict(author_config),
            "split_summary": split_summary,
            "export": {
                "source_root": str(export.source_root),
                "split_counts": export.split_counts,
                "participant_id_by_row_key": export.participant_id_by_row_key,
            },
            "author_summary_path": str(runtime_root / "author_dual_encoder" / "author_run_summary.json"),
        },
    )
    return DualEncoderFitPredictResult(
        method_name=method_name,
        scoring_model=scoring_model,
        predictions_by_row_key=predictions_by_row_key,
        metadata_path=metadata_path,
    )
