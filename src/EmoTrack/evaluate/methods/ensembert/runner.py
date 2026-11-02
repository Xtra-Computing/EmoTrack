from __future__ import annotations

import gc
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader, Dataset

from EmoTrack.evaluate.methods.ensembert.dataset import (
    DEFAULT_ITEM_DESCRIPTIONS_PATH,
    EnsemBERTItemExample,
    PHQ8ItemSpec,
    build_labeled_item_examples,
    build_prediction_item_examples,
    extract_document_texts,
    load_phq8_item_specs,
)
from EmoTrack.evaluate.methods.ensembert.embedder import FrozenTextEmbedder
from EmoTrack.evaluate.methods.ensembert.model import (
    EnsemBERTModel,
    EnsemBERTModelConfig,
    EnsemBERTTrainingResult,
    predict_item_classes,
    train_ensembert_model,
)
from EmoTrack.evaluate.methods.ensembert.retrieval import gather_top_k_embeddings
from EmoTrack.evaluate.core.scores import (
    SessionExampleRow,
    make_row_key,
)

from EmoTrack.evaluate.utils import (
    stable_text_hash,
    write_json,
)
from EmoTrack.utils.determinism import make_loader_generator, seed_worker, set_determinism
from EmoTrack.utils.paths import REPO_ROOT


_ITEM_TENSOR_CACHE_VERSION = 1
_CACHE_LOCK_STALE_SECONDS = 43200


@dataclass(frozen=True)
class EnsemBERTRunConfig:
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
    max_posts: int = 20
    min_words: int = 2
    document_source: str = "client_utterances"
    dense_units: int = 128
    num_heads: int = 1
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    num_epochs: int = 50
    batch_size: int = 32
    patience: int = 8
    out_loss_weight: float = 1.0
    soft_loss_weight: float = 1.0
    meta_loss_weight: float = 1.0
    prediction_head: str = "meta"
    item_description_path: str = str(DEFAULT_ITEM_DESCRIPTIONS_PATH)
    method_prefix: str = "ensembert"
    seed: int = 0
    include_history: bool = False
    deterministic: bool = False


@dataclass(frozen=True)
class EnsemBERTFeatureStore:
    metadata: dict[str, object]
    row_texts: dict[str, list[str]]
    row_retrieval_embeddings: dict[str, torch.Tensor]
    row_embeddings: dict[str, torch.Tensor]
    item_retrieval_embeddings: torch.Tensor
    item_embeddings: torch.Tensor
    choice_embeddings: torch.Tensor

    @property
    def embedding_dim(self) -> int:
        return int(self.item_embeddings.size(1))


@dataclass(frozen=True)
class EnsemBERTFitPredictResult:
    method_name: str
    scoring_model: str
    predictions_by_row_key: dict[str, float]
    predicted_items_by_row_key: dict[str, dict[str, float]]
    training_result: EnsemBERTTrainingResult
    metadata_path: Path
    model_path: Path


class EnsemBERTItemDataset(Dataset):
    def __init__(
        self,
        examples: list[EnsemBERTItemExample],
        *,
        feature_store: EnsemBERTFeatureStore,
        max_posts: int,
        cache_dir: Path | None = None,
        cache_name: str = "items",
    ) -> None:
        self._records = _load_or_build_item_records(
            examples,
            feature_store=feature_store,
            max_posts=max_posts,
            cache_dir=cache_dir,
            cache_name=cache_name,
        )

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self._records[idx]


def fit_predict_ensembert(
    *,
    train_rows: list[SessionExampleRow],
    val_rows: list[SessionExampleRow],
    test_rows: list[SessionExampleRow],
    out_root: Path,
    config: EnsemBERTRunConfig,
    inner_val_fraction: float,
) -> EnsemBERTFitPredictResult:
    item_specs = load_phq8_item_specs(descriptions_path=config.item_description_path)
    seed = int(config.seed)
    if config.deterministic:
        set_determinism(seed)
    else:
        _set_seed(seed)
    train_fit_rows, validation_rows = _resolve_training_validation_rows(
        train_rows=train_rows,
        val_rows=val_rows,
        inner_val_fraction=inner_val_fraction,
        seed=seed,
    )
    all_rows = train_rows + val_rows + test_rows
    shared_feature_cache_root = _shared_feature_cache_root(out_root)
    feature_store = build_feature_store(
        all_rows,
        item_specs=item_specs,
        config=config,
        cache_path=shared_feature_cache_root / "ensembert_embeddings.pt",
    )
    train_examples = build_labeled_item_examples(train_fit_rows, item_specs)
    validation_examples = build_labeled_item_examples(validation_rows, item_specs)
    if not train_examples:
        raise ValueError("EnsemBERT requires item-level PHQ-8 labels in train rows")
    if not validation_examples:
        validation_examples = train_examples

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = EnsemBERTModelConfig(
        embedding_dim=feature_store.embedding_dim,
        max_posts=config.max_posts,
        dense_units=config.dense_units,
        num_heads=config.num_heads,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        out_loss_weight=config.out_loss_weight,
        soft_loss_weight=config.soft_loss_weight,
        meta_loss_weight=config.meta_loss_weight,
    )
    model = EnsemBERTModel(model_config)
    train_loader_kwargs: dict[str, object]
    if config.deterministic:
        train_loader_kwargs = {
            "generator": make_loader_generator(seed),
            "worker_init_fn": seed_worker,
        }
    else:
        generator = torch.Generator()
        generator.manual_seed(seed)
        train_loader_kwargs = {"generator": generator}
    train_loader = DataLoader(
        EnsemBERTItemDataset(
            train_examples,
            feature_store=feature_store,
            max_posts=config.max_posts,
            cache_dir=shared_feature_cache_root / "item_tensors",
            cache_name="train",
        ),
        batch_size=config.batch_size,
        shuffle=True,
        **train_loader_kwargs,
    )
    validation_loader = DataLoader(
        EnsemBERTItemDataset(
            validation_examples,
            feature_store=feature_store,
            max_posts=config.max_posts,
            cache_dir=shared_feature_cache_root / "item_tensors",
            cache_name="validation",
        ),
        batch_size=config.batch_size,
        shuffle=False,
    )
    training_result = train_ensembert_model(
        model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        config=model_config,
        num_epochs=config.num_epochs,
        patience=config.patience,
        device=device,
    )
    del train_loader
    del validation_loader
    gc.collect()

    model_dir = out_root / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "ensembert_model.pt"
    torch.save(model.state_dict(), model_path)

    prediction_examples = build_prediction_item_examples(test_rows, item_specs)
    prediction_loader = DataLoader(
        EnsemBERTItemDataset(
            prediction_examples,
            feature_store=feature_store,
            max_posts=config.max_posts,
            cache_dir=shared_feature_cache_root / "item_tensors",
            cache_name="prediction",
        ),
        batch_size=config.batch_size,
        shuffle=False,
    )
    predicted_classes = predict_item_classes(
        model,
        loader=prediction_loader,
        device=device,
        head=config.prediction_head,
    )
    predictions_by_row_key, predicted_items_by_row_key = _assemble_predictions(
        prediction_examples=prediction_examples,
        predicted_classes=predicted_classes,
    )
    metadata_path = write_ensembert_metadata(
        path=model_dir / "ensembert_model_meta.json",
        config=config,
        model_config=model_config,
        item_specs=item_specs,
        training_result=training_result,
        train_fit_rows=train_fit_rows,
        validation_rows=validation_rows,
        test_rows=test_rows,
        feature_store=feature_store,
    )
    method_name = f"{config.method_prefix}_{'all' if config.include_history else 'now'}"
    return EnsemBERTFitPredictResult(
        method_name=method_name,
        scoring_model=f"EnsemBERT[{config.embedding_model_name}]",
        predictions_by_row_key=predictions_by_row_key,
        predicted_items_by_row_key=predicted_items_by_row_key,
        training_result=training_result,
        metadata_path=metadata_path,
        model_path=model_path,
    )


def _shared_feature_cache_root(out_root: Path) -> Path:
    run_root = out_root.parent if out_root.name.startswith("seed") and out_root.name[4:].isdigit() else out_root
    try:
        experiment_tag = run_root.parent.name
        if run_root.parent.parent.parent.name == "eval":
            return REPO_ROOT / "out" / "embed_cache" / experiment_tag
    except IndexError:
        pass
    return REPO_ROOT / "out" / "embed_cache" / run_root.name


def _build_item_records(
    examples: list[EnsemBERTItemExample],
    *,
    feature_store: EnsemBERTFeatureStore,
    max_posts: int,
) -> list[dict[str, torch.Tensor]]:
    return [
        _build_item_record(example, feature_store=feature_store, max_posts=max_posts)
        for example in examples
    ]


def _build_item_record(
    example: EnsemBERTItemExample,
    *,
    feature_store: EnsemBERTFeatureStore,
    max_posts: int,
) -> dict[str, torch.Tensor]:
    item_embedding = feature_store.item_embeddings[example.item_index].float()
    item_retrieval_embedding = feature_store.item_retrieval_embeddings[example.item_index].float()
    document_embeddings = feature_store.row_embeddings[example.row_key].float()
    document_retrieval_embeddings = feature_store.row_retrieval_embeddings[example.row_key].float()
    label = 0 if example.label is None else int(example.label)
    return {
        "reddit_embeddings": gather_top_k_embeddings(
            document_embeddings,
            item_retrieval_embedding,
            k=max_posts,
            retrieval_document_embeddings=document_retrieval_embeddings,
        ),
        "description_embedding": item_embedding,
        "choice_embeddings": feature_store.choice_embeddings[example.item_index].float(),
        "label": torch.tensor(label, dtype=torch.long),
        "item_index": torch.tensor(example.item_index, dtype=torch.long),
    }


def _load_or_build_item_records(
    examples: list[EnsemBERTItemExample],
    *,
    feature_store: EnsemBERTFeatureStore,
    max_posts: int,
    cache_dir: Path | None,
    cache_name: str,
) -> list[dict[str, torch.Tensor]]:
    if cache_dir is None:
        return _build_item_records(examples, feature_store=feature_store, max_posts=max_posts)
    metadata = _item_tensor_cache_metadata(examples, feature_store=feature_store, max_posts=max_posts)
    cache_path = _item_tensor_cache_path(cache_dir=cache_dir, cache_name=cache_name, metadata=metadata)
    cached = _load_item_records(cache_path, expected_metadata=metadata)
    if cached is not None:
        return cached
    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    with _acquire_cache_lock(lock_path):
        cached = _load_item_records(cache_path, expected_metadata=metadata)
        if cached is not None:
            return cached
        records = _build_item_records(examples, feature_store=feature_store, max_posts=max_posts)
        _write_item_records(cache_path, records=records, metadata=metadata)
        return records


def _item_tensor_cache_metadata(
    examples: list[EnsemBERTItemExample],
    *,
    feature_store: EnsemBERTFeatureStore,
    max_posts: int,
) -> dict[str, object]:
    return {
        "version": _ITEM_TENSOR_CACHE_VERSION,
        "max_posts": int(max_posts),
        "feature_metadata": feature_store.metadata,
        "examples": [asdict(example) for example in examples],
        "example_count": len(examples),
    }


def _item_tensor_cache_path(*, cache_dir: Path, cache_name: str, metadata: dict[str, object]) -> Path:
    safe_name = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(cache_name))
    cache_key = stable_text_hash(json.dumps(metadata, sort_keys=True, ensure_ascii=False))
    return cache_dir / f"{safe_name}_{cache_key}.pt"


def _load_item_records(cache_path: Path, *, expected_metadata: dict[str, object]) -> list[dict[str, torch.Tensor]] | None:
    if not cache_path.is_file():
        return None
    try:
        payload = torch.load(cache_path, map_location="cpu")
    except (EOFError, OSError, RuntimeError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("metadata") != expected_metadata:
        return None
    records = payload.get("records")
    if not isinstance(records, list):
        return None
    if len(records) != int(expected_metadata["example_count"]):
        return None
    return records


def _write_item_records(
    cache_path: Path,
    *,
    records: list[dict[str, torch.Tensor]],
    metadata: dict[str, object],
) -> Path:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
    try:
        torch.save({"metadata": metadata, "records": records}, tmp_path)
        tmp_path.replace(cache_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return cache_path


class _CacheLock:
    def __init__(self, path: Path) -> None:
        self.path = path

    def __enter__(self) -> "_CacheLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(f"{os.getpid()}\n")
                return self
            except FileExistsError:
                if _is_stale_lock(self.path):
                    self.path.unlink(missing_ok=True)
                    continue
                time.sleep(5)

    def __exit__(self, *_exc: object) -> None:
        self.path.unlink(missing_ok=True)


def _acquire_cache_lock(lock_path: Path) -> _CacheLock:
    return _CacheLock(lock_path)


def _is_stale_lock(lock_path: Path) -> bool:
    try:
        age_seconds = time.time() - lock_path.stat().st_mtime
        return age_seconds > _CACHE_LOCK_STALE_SECONDS
    except FileNotFoundError:
        return False


def build_feature_store(
    rows: list[SessionExampleRow],
    *,
    item_specs: list[PHQ8ItemSpec],
    config: EnsemBERTRunConfig,
    cache_path: Path,
    embed_texts: Callable[[list[str]], torch.Tensor] | None = None,
) -> EnsemBERTFeatureStore:
    row_texts: dict[str, list[str]] = {}
    for row in rows:
        texts = extract_document_texts(
            row.transcript_full,
            document_source=config.document_source,
            min_words=config.min_words,
        )
        if not texts:
            fallback = row.transcript_full.strip() or "[empty transcript]"
            texts = [fallback]
        row_texts[make_row_key(row)] = texts

    metadata = {
        "retrieval_model_name": config.retrieval_model_name,
        "retrieval_max_length": config.retrieval_max_length,
        "retrieval_pooling_strategy": config.retrieval_pooling_strategy,
        "embedding_model_name": config.embedding_model_name,
        "embedding_max_length": config.embedding_max_length,
        "embedding_pooling_strategy": config.embedding_pooling_strategy,
        "max_posts": config.max_posts,
        "min_words": config.min_words,
        "document_source": config.document_source,
        "item_spec_hash": _item_spec_hash(item_specs),
        "row_hash": stable_text_hash(json.dumps(row_texts, sort_keys=True, ensure_ascii=False)),
    }
    cached = _load_feature_store(cache_path, expected_metadata=metadata)
    if cached is not None:
        return cached
    store = _build_feature_store_uncached(
        row_texts=row_texts,
        item_specs=item_specs,
        config=config,
        metadata=metadata,
        embed_texts=embed_texts,
    )
    _write_feature_store(cache_path, store)
    return store


def _build_feature_store_uncached(
    *,
    row_texts: dict[str, list[str]],
    item_specs: list[PHQ8ItemSpec],
    config: EnsemBERTRunConfig,
    metadata: dict[str, object],
    embed_texts: Callable[[list[str]], torch.Tensor] | None,
) -> EnsemBERTFeatureStore:
    retrieval_texts: list[str] = []
    embedding_texts: list[str] = []
    for spec in item_specs:
        retrieval_texts.append(spec.description)
        embedding_texts.append(spec.description)
        embedding_texts.extend(spec.choice_texts)
    for texts in row_texts.values():
        retrieval_texts.extend(texts)
        embedding_texts.extend(texts)

    unique_retrieval_texts = list(dict.fromkeys(retrieval_texts))
    unique_embedding_texts = list(dict.fromkeys(embedding_texts))
    if embed_texts is None:
        retrieval_embedder = FrozenTextEmbedder(
            model_name=config.retrieval_model_name,
            batch_size=config.retrieval_batch_size,
            max_length=config.retrieval_max_length,
            pooling_strategy=config.retrieval_pooling_strategy,
            torch_dtype=config.retrieval_torch_dtype,
            trust_remote_code=config.retrieval_trust_remote_code,
        )
        embedding_embedder = FrozenTextEmbedder(
            model_name=config.embedding_model_name,
            batch_size=config.embedding_batch_size,
            max_length=config.embedding_max_length,
            pooling_strategy=config.embedding_pooling_strategy,
            torch_dtype=config.embedding_torch_dtype,
            trust_remote_code=config.embedding_trust_remote_code,
        )
        retrieval_tensor = retrieval_embedder.embed_texts(unique_retrieval_texts)
        embedding_tensor = embedding_embedder.embed_texts(unique_embedding_texts)
    else:
        retrieval_tensor = embed_texts(unique_retrieval_texts).float()
        embedding_tensor = embed_texts(unique_embedding_texts).float()

    retrieval_by_text = {text: retrieval_tensor[index] for index, text in enumerate(unique_retrieval_texts)}
    embedding_by_text = {text: embedding_tensor[index] for index, text in enumerate(unique_embedding_texts)}

    row_retrieval_embeddings = {
        row_key: torch.stack([retrieval_by_text[text] for text in texts]).float()
        for row_key, texts in row_texts.items()
    }
    row_embeddings = {
        row_key: torch.stack([embedding_by_text[text] for text in texts]).float()
        for row_key, texts in row_texts.items()
    }
    item_retrieval_embeddings = torch.stack([retrieval_by_text[spec.description] for spec in item_specs]).float()
    item_embeddings = torch.stack([embedding_by_text[spec.description] for spec in item_specs]).float()
    choice_embeddings = torch.stack(
        [
            torch.stack([embedding_by_text[text] for text in spec.choice_texts]).float()
            for spec in item_specs
        ]
    )
    return EnsemBERTFeatureStore(
        metadata=metadata,
        row_texts=row_texts,
        row_retrieval_embeddings=row_retrieval_embeddings,
        row_embeddings=row_embeddings,
        item_retrieval_embeddings=item_retrieval_embeddings,
        item_embeddings=item_embeddings,
        choice_embeddings=choice_embeddings,
    )


def config_from_method_config(method_config: object, *, include_history: bool | None = None) -> EnsemBERTRunConfig:
    return EnsemBERTRunConfig(
        retrieval_model_name=method_config.retrieval_model_name,
        retrieval_batch_size=method_config.retrieval_batch_size,
        retrieval_max_length=method_config.retrieval_max_length,
        retrieval_pooling_strategy=method_config.retrieval_pooling_strategy,
        retrieval_torch_dtype=method_config.retrieval_torch_dtype,
        retrieval_trust_remote_code=method_config.retrieval_trust_remote_code,
        embedding_model_name=method_config.embedding_model_name,
        embedding_batch_size=method_config.embedding_batch_size,
        embedding_max_length=method_config.embedding_max_length,
        embedding_pooling_strategy=method_config.embedding_pooling_strategy,
        embedding_torch_dtype=method_config.embedding_torch_dtype,
        embedding_trust_remote_code=method_config.embedding_trust_remote_code,
        max_posts=method_config.ensembert_max_posts,
        min_words=method_config.ensembert_min_words,
        document_source=method_config.ensembert_document_source,
        dense_units=method_config.ensembert_dense_units,
        num_heads=method_config.ensembert_num_heads,
        learning_rate=method_config.ensembert_learning_rate,
        weight_decay=method_config.ensembert_weight_decay,
        num_epochs=method_config.ensembert_num_epochs,
        batch_size=method_config.ensembert_batch_size,
        patience=method_config.ensembert_patience,
        out_loss_weight=method_config.ensembert_out_loss_weight,
        soft_loss_weight=method_config.ensembert_soft_loss_weight,
        meta_loss_weight=method_config.ensembert_meta_loss_weight,
        prediction_head=method_config.ensembert_prediction_head,
        item_description_path=method_config.ensembert_item_description_path,
        method_prefix=method_config.ensembert_method_prefix,
        seed=method_config.inference_seed,
        include_history=method_config.include_history if include_history is None else include_history,
        deterministic=method_config.deterministic,
    )


def write_ensembert_metadata(
    *,
    path: Path,
    config: EnsemBERTRunConfig,
    model_config: EnsemBERTModelConfig,
    item_specs: list[PHQ8ItemSpec],
    training_result: EnsemBERTTrainingResult,
    train_fit_rows: list[SessionExampleRow],
    validation_rows: list[SessionExampleRow],
    test_rows: list[SessionExampleRow],
    feature_store: EnsemBERTFeatureStore,
) -> Path:
    return write_json(
        path,
        {
            "config": asdict(config),
            "model_config": asdict(model_config),
            "item_specs": [asdict(spec) for spec in item_specs],
            "training_result": asdict(training_result),
            "train_size": len(train_fit_rows),
            "validation_size": len(validation_rows),
            "test_size": len(test_rows),
            "feature_metadata": feature_store.metadata,
            "embedding_dim": feature_store.embedding_dim,
        },
    )


def _resolve_training_validation_rows(
    *,
    train_rows: list[SessionExampleRow],
    val_rows: list[SessionExampleRow],
    inner_val_fraction: float,
    seed: int,
) -> tuple[list[SessionExampleRow], list[SessionExampleRow]]:
    if val_rows:
        return list(train_rows), list(val_rows)
    if len(train_rows) <= 1:
        return list(train_rows), list(train_rows)
    if not 0 < inner_val_fraction < 1:
        raise ValueError("inner_val_fraction must be between 0 and 1")
    grouped: dict[str, list[SessionExampleRow]] = {}
    for row in train_rows:
        grouped.setdefault(row.run_id, []).append(row)
    run_ids = sorted(grouped)
    rng = random.Random(seed)
    rng.shuffle(run_ids)
    validation_count = max(1, round(len(run_ids) * inner_val_fraction))
    validation_ids = set(run_ids[:validation_count])
    fit_rows: list[SessionExampleRow] = []
    validation_rows: list[SessionExampleRow] = []
    for run_id in sorted(grouped):
        target = validation_rows if run_id in validation_ids else fit_rows
        target.extend(grouped[run_id])
    return fit_rows, validation_rows


def _assemble_predictions(
    *,
    prediction_examples: list[EnsemBERTItemExample],
    predicted_classes: list[int],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    if len(prediction_examples) != len(predicted_classes):
        raise ValueError("Prediction count does not match item example count")
    predicted_items: dict[str, dict[str, float]] = {}
    for example, score in zip(prediction_examples, predicted_classes, strict=True):
        predicted_items.setdefault(example.row_key, {})[example.symptom_name] = float(score)
    predictions = {
        row_key: float(sum(item_scores.values()))
        for row_key, item_scores in predicted_items.items()
    }
    return predictions, predicted_items


def _item_spec_hash(item_specs: list[PHQ8ItemSpec]) -> str:
    return stable_text_hash(json.dumps([asdict(spec) for spec in item_specs], sort_keys=True, ensure_ascii=False))


def _load_feature_store(cache_path: Path, *, expected_metadata: dict[str, object]) -> EnsemBERTFeatureStore | None:
    if not cache_path.is_file():
        return None
    payload = torch.load(cache_path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("metadata") != expected_metadata:
        return None
    return EnsemBERTFeatureStore(
        metadata=payload["metadata"],
        row_texts=payload["row_texts"],
        row_retrieval_embeddings=payload["row_retrieval_embeddings"],
        row_embeddings=payload["row_embeddings"],
        item_retrieval_embeddings=payload["item_retrieval_embeddings"],
        item_embeddings=payload["item_embeddings"],
        choice_embeddings=payload["choice_embeddings"],
    )


def _write_feature_store(cache_path: Path, store: EnsemBERTFeatureStore) -> Path:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
    try:
        torch.save(
            {
                "metadata": store.metadata,
                "row_texts": store.row_texts,
                "row_retrieval_embeddings": store.row_retrieval_embeddings,
                "row_embeddings": store.row_embeddings,
                "item_retrieval_embeddings": store.item_retrieval_embeddings,
                "item_embeddings": store.item_embeddings,
                "choice_embeddings": store.choice_embeddings,
            },
            tmp_path,
        )
        tmp_path.replace(cache_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return cache_path


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
