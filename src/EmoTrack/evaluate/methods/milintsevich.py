from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import sys

from EmoTrack.evaluate.author_milintsevich_daic import (
    author_checkpoint_path,
    clone_author_repo,
    evaluate_public_test_total_mae,
    patch_author_repo_for_early_stopping,
    patch_author_repo_for_modern_datasets,
    prepare_author_milintsevich_daic_dataset,
    read_best_dev_metrics,
    run_author_training,
    write_author_config,
    write_author_runtime_shims,
)
from EmoTrack.evaluate.methods._public_daic_export import (
    predicted_item_scores_from_sequence,
    split_train_validation_rows,
    write_public_daic_source,
)
from EmoTrack.evaluate.core.config import EvaluationMethodConfig
from EmoTrack.evaluate.core.scores import SessionExampleRow

from EmoTrack.evaluate.utils import write_json


@dataclass(frozen=True)
class MilintsevichRunConfig:
    repo_root: str = "cache/external/hierarchical-depression-symptom-classifier"
    python_executable: str = ""
    seed: int = 0
    num_iters: int = 100
    batch_size: int = 8
    early_stopping_patience: int = 20
    include_history: bool = False
    method_prefix: str = "milintsevich"
    deterministic: bool = False


@dataclass(frozen=True)
class MilintsevichFitPredictResult:
    method_name: str
    scoring_model: str
    predictions_by_row_key: dict[str, float]
    predicted_items_by_row_key: dict[str, dict[str, float]]
    metadata_path: Path


def config_from_method_config(
    method_config: EvaluationMethodConfig,
    *,
    include_history: bool,
) -> MilintsevichRunConfig:
    return MilintsevichRunConfig(
        repo_root=method_config.milintsevich_repo_root,
        python_executable=method_config.milintsevich_python_executable,
        seed=method_config.inference_seed,
        num_iters=method_config.milintsevich_num_iters,
        batch_size=method_config.milintsevich_batch_size,
        early_stopping_patience=method_config.milintsevich_early_stopping_patience,
        include_history=include_history,
        method_prefix=method_config.milintsevich_method_prefix,
        deterministic=method_config.deterministic,
    )


def fit_predict_milintsevich(
    *,
    train_rows: list[SessionExampleRow],
    val_rows: list[SessionExampleRow],
    test_rows: list[SessionExampleRow],
    out_root: Path,
    config: MilintsevichRunConfig,
    inner_val_fraction: float,
) -> MilintsevichFitPredictResult:
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
    author_repo_root = clone_author_repo(repo_root=Path(config.repo_root))
    save_dir = runtime_root / "saved_models"
    write_author_runtime_shims(author_repo_root=author_repo_root)
    patch_author_repo_for_modern_datasets(author_repo_root=author_repo_root)
    patch_author_repo_for_early_stopping(author_repo_root=author_repo_root)
    if config.deterministic:
        from EmoTrack.evaluate.author_milintsevich_daic import patch_author_repo_for_determinism

        patch_author_repo_for_determinism(author_repo_root=author_repo_root)
    dataset_summary = prepare_author_milintsevich_daic_dataset(
        source_root=export.source_root,
        author_repo_root=author_repo_root,
        output_root=runtime_root / "author_milintsevich",
    )
    config_path = write_author_config(
        author_repo_root=author_repo_root,
        save_dir=save_dir,
        seed=config.seed,
        num_iters=config.num_iters,
        batch_size=config.batch_size,
        early_stopping_patience=config.early_stopping_patience,
        deterministic=config.deterministic,
    )
    run_author_training(
        author_repo_root=author_repo_root,
        python_executable=config.python_executable or sys.executable,
        seed=config.seed,
        deterministic=config.deterministic,
    )
    checkpoint = author_checkpoint_path(save_dir=save_dir, seed=config.seed)
    public_test_eval = evaluate_public_test_total_mae(
        author_repo_root=author_repo_root,
        source_root=export.source_root,
        checkpoint_path=checkpoint,
        output_root=runtime_root / "author_milintsevich",
    )
    raw_predictions = public_test_eval.get("predictions")
    if not isinstance(raw_predictions, list):
        raise RuntimeError("Milintsevich runner did not return test predictions")

    predictions_by_row_key: dict[str, float] = {}
    predicted_items_by_row_key: dict[str, dict[str, float]] = {}
    for raw_prediction in raw_predictions:
        if not isinstance(raw_prediction, dict):
            raise RuntimeError(f"Invalid Milintsevich prediction row: {raw_prediction!r}")
        participant_id = int(raw_prediction["participant_id"])
        row_key = export.row_key_by_participant_id[participant_id]
        predictions_by_row_key[row_key] = float(raw_prediction["predicted_total"])
        raw_items = raw_prediction.get("predicted_items")
        if isinstance(raw_items, list):
            predicted_items_by_row_key[row_key] = predicted_item_scores_from_sequence(
                [float(value) for value in raw_items]
            )

    method_name = f"{config.method_prefix}_{'all' if config.include_history else 'now'}"
    scoring_model = "MilintsevichHDSC[sentence-transformers/all-distilroberta-v1]"
    metadata_path = write_json(
        out_root / "model" / "milintsevich_model_meta.json",
        {
            "method_name": method_name,
            "scoring_model": scoring_model,
            "config": asdict(config),
            "split_summary": split_summary,
            "export": {
                "source_root": str(export.source_root),
                "split_counts": export.split_counts,
                "participant_id_by_row_key": export.participant_id_by_row_key,
            },
            "dataset_summary": dataset_summary,
            "config_path": str(config_path),
            "checkpoint_path": str(checkpoint),
            "best_dev_metrics": read_best_dev_metrics(save_dir=save_dir, seed=config.seed),
            "public_test_eval_path": str(runtime_root / "author_milintsevich" / "public_test_total_eval.json"),
        },
    )
    return MilintsevichFitPredictResult(
        method_name=method_name,
        scoring_model=scoring_model,
        predictions_by_row_key=predictions_by_row_key,
        predicted_items_by_row_key=predicted_items_by_row_key,
        metadata_path=metadata_path,
    )
