from __future__ import annotations

import csv
import importlib
import json
import os
import random
import re
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer, RobertaModel
from transformers.cache_utils import DynamicCache

from EmoTrack.utils.determinism import make_loader_generator, seed_worker, set_determinism


AUTHOR_REPO_URL = "https://github.com/clintonlau/dual-encoder-model.git"
START_PROMPTS = re.compile(
    "are you okay with this|hi i'm ellie thanks for coming in today|think of me as a friend i don't judge|"
    "and please feel free to tell me anything|i'm not a therapist|i'll ask a few questions|"
    "i was created to talk"
)
CLOSING_PROMPTS = re.compile(
    "okay i think i have asked everything i need to|okay i think i've asked everything i need to|"
    "goodbye|it was great chatting with you|thanks for sharing your thoughts"
)


@dataclass(frozen=True)
class AuthorDualEncoderConfig:
    seed: int = 0
    batch_size: int = 2
    num_epoch: int = 200
    es_patience: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    transformer_pretrained_id: str = "sentence-transformers/all-mpnet-base-v2"
    pooling_strategy: str = "mean"
    torch_dtype: str = "auto"
    trust_remote_code: bool = False
    st_max_token_length: int = 256
    prefix_max_token_length: int = 128
    pre_seq_len: int = 10
    encoding_projection_size: int = 128
    lstm_input_size: int = 128
    lstm_hidden_size: int = 64
    lstm_num_layers: int = 1
    dropout_prob: float = 0.5
    problem_type: str = "regression"
    num_labels: int = 1
    deterministic: bool = False


@dataclass(frozen=True)
class AuthorDualExample:
    subject_id: int
    split: str
    text_input: list[str]
    phq_binary: int
    phq_score: int
    preprocess_mode: str


def clone_author_repo(*, repo_root: Path, clone_url: str = AUTHOR_REPO_URL) -> Path:
    resolved = repo_root.expanduser().resolve()
    if resolved.exists():
        _validate_author_repo(resolved)
        return resolved
    resolved.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", clone_url, str(resolved)], check=True)
    _validate_author_repo(resolved)
    return resolved


def run_author_dual_encoder_daic(
    *,
    source_root: Path,
    author_repo_root: Path,
    output_root: Path,
    config: AuthorDualEncoderConfig = AuthorDualEncoderConfig(),
    prepare_only: bool = False,
    warm_start: bool = False,
) -> dict[str, object]:
    resolved_author_root = clone_author_repo(repo_root=author_repo_root)
    resolved_source_root = source_root.expanduser().resolve()
    resolved_output_root = output_root.expanduser().resolve()
    resolved_output_root.mkdir(parents=True, exist_ok=True)

    examples = load_public_daic_examples(resolved_source_root)
    dataset_summary = write_dataset_summary(
        examples=examples,
        source_root=resolved_source_root,
        author_repo_root=resolved_author_root,
        output_root=resolved_output_root,
    )
    summary: dict[str, object] = {
        "source_root": str(resolved_source_root),
        "author_repo_root": str(resolved_author_root),
        "output_root": str(resolved_output_root),
        "config": asdict(config),
        "dataset_summary": dataset_summary,
        "train_invoked": not prepare_only,
        "warm_start": warm_start,
    }
    if prepare_only:
        _write_json(resolved_output_root / "author_run_summary.json", summary)
        return summary

    set_seed(config.seed, deterministic=config.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df = examples_to_author_dataframe(examples, split="train")
    dev_df = examples_to_author_dataframe(examples, split="dev")
    test_df = examples_to_author_dataframe(examples, split="test")

    with _sys_path(resolved_author_root):
        _patch_author_transformers_compat()
        dataset_module = importlib.import_module("dataset.dataset")
        dual_module = importlib.import_module("model.dual_encoder")
        _patch_author_dual_encoder_runtime(dual_module=dual_module)
        train_dataset = dataset_module.DaicTextDataset(train_df)
        dev_dataset = dataset_module.DaicTextDataset(dev_df)
        test_dataset = dataset_module.DaicTextDataset(test_df)
        collator = _AuthorDualEncoderCollator(config=config)
        hparam = _to_author_hparam(config=config, device=device)

    train_loader_kwargs: dict[str, object] = {}
    if config.deterministic:
        train_loader_kwargs = {
            "generator": make_loader_generator(config.seed),
            "worker_init_fn": seed_worker,
        }
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collator,
        **train_loader_kwargs,
    )
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collator)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collator)

    prefix_summary: dict[str, object] | None = None
    if warm_start:
        with _sys_path(resolved_author_root):
            prefix_model = _init_author_prefix_model(config=config, device=device).to(device)
        prefix_summary = _train_regression_model(
            model=prefix_model,
            model_input="prefix",
            train_loader=train_loader,
            dev_loader=dev_loader,
            device=device,
            config=config,
            stage_name="prefix_only",
            best_path=resolved_output_root / "prefix_only_best_model.pt",
            metrics_path=resolved_output_root / "prefix_only_epoch_metrics.jsonl",
        )
    else:
        prefix_model = None

    with _sys_path(resolved_author_root):
        model = dual_module.init_dual_encoder(hparam).to(device)
    if prefix_model is not None:
        _copy_prefix_only_weights_to_dual(prefix_model=prefix_model, dual_model=model)

    dual_summary = _train_regression_model(
        model=model,
        model_input="dual",
        train_loader=train_loader,
        dev_loader=dev_loader,
        device=device,
        config=config,
        stage_name="dual_encoder",
        best_path=resolved_output_root / "best_model.pt",
        metrics_path=resolved_output_root / "epoch_metrics.jsonl",
    )
    best_path = Path(str(dual_summary["checkpoint_path"]))
    best_epoch = int(dual_summary["best_epoch"])
    best_dev_loss = float(dual_summary["best_dev_loss"])
    best_dev_mae = float(dual_summary["best_dev_mae"])

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = _run_epoch(model=model, model_input="dual", loader=test_loader, device=device, include_predictions=True)

    predictions = [
        {
            "subject_id": int(subject_id),
            "ground_truth_total": float(label),
            "predicted_total": float(pred),
            "absolute_error": abs(float(pred) - float(label)),
        }
        for subject_id, pred, label in zip(test_df["subject_id"].tolist(), test_metrics["predictions"], test_metrics["labels"])
    ]
    test_summary = {
        "checkpoint_path": str(best_path),
        "best_epoch": best_epoch,
        "best_dev_loss": best_dev_loss,
        "best_dev_mae": best_dev_mae,
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
        "test_loss": test_metrics["loss"],
        "test_session_count": len(predictions),
        "predictions": predictions,
    }
    _write_json(resolved_output_root / "public_test_total_eval.json", test_summary)
    summary.update(
        {
            "best_epoch": best_epoch,
            "best_dev_loss": best_dev_loss,
            "best_dev_mae": best_dev_mae,
            "checkpoint_path": str(best_path),
            "epoch_metrics_path": str(dual_summary["metrics_path"]),
            "epochs_completed": dual_summary["epochs_completed"],
            "prefix_only_training": prefix_summary,
            "dual_encoder_training": dual_summary,
            "test_metrics": test_summary,
        }
    )
    _write_json(resolved_output_root / "author_run_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def load_public_daic_examples(source_root: Path) -> list[AuthorDualExample]:
    labels_path = source_root / "labels.csv"
    if not labels_path.is_file():
        raise FileNotFoundError(f"Missing public DAIC labels file: {labels_path}")
    examples: list[AuthorDualExample] = []
    with labels_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            subject_id = int(str(row["participant_id"]).strip())
            split = str(row["split"]).strip().lower()
            transcript_path = source_root / split / f"{subject_id}.txt"
            if not transcript_path.is_file():
                raise FileNotFoundError(f"Missing public DAIC transcript: {transcript_path}")
            text_input, preprocess_mode = _load_author_qr_texts(transcript_path)
            examples.append(
                AuthorDualExample(
                    subject_id=subject_id,
                    split=split,
                    text_input=text_input,
                    phq_binary=int(row["phq_binary"]),
                    phq_score=int(row["phq_score"]),
                    preprocess_mode=preprocess_mode,
                )
            )
    return examples


def examples_to_author_dataframe(examples: list[AuthorDualExample], *, split: str) -> pd.DataFrame:
    rows = [
        {
            "subject_id": example.subject_id,
            "text_input": example.text_input,
            "PHQ8_Binary": example.phq_binary,
            "PHQ8_Score": example.phq_score,
        }
        for example in examples
        if example.split == split
    ]
    if not rows:
        raise ValueError(f"DAIC split {split!r} is empty")
    return pd.DataFrame(rows)


def write_dataset_summary(
    *,
    examples: list[AuthorDualExample],
    source_root: Path,
    author_repo_root: Path,
    output_root: Path,
) -> dict[str, object]:
    counts: dict[str, int] = {}
    turn_lengths: dict[str, dict[str, float]] = {}
    for split in ("train", "dev", "test"):
        split_examples = [example for example in examples if example.split == split]
        counts[split] = len(split_examples)
        lengths = [len(example.text_input) for example in split_examples]
        turn_lengths[split] = {
            "min": float(min(lengths)),
            "mean": float(np.mean(lengths)),
            "max": float(max(lengths)),
        }
    participant_only = [
        {
            "subject_id": example.subject_id,
            "split": example.split,
            "text_count": len(example.text_input),
        }
        for example in examples
        if example.preprocess_mode == "participant_only"
    ]
    summary = {
        "source_root": str(source_root),
        "author_repo_root": str(author_repo_root),
        "counts": counts,
        "qr_text_lengths": turn_lengths,
        "participant_only_transcripts": participant_only,
        "preprocess_note": (
            "Existing public *.txt transcripts are cleaned with the author's rules, then each Ellie "
            "question and following participant response are concatenated into one QR input string. "
            "Transcripts with no Ellie rows are explicitly kept as participant-only utterance sequences."
        ),
    }
    _write_json(output_root / "dataset_summary.json", summary)
    preview = {
        str(example.subject_id): example.text_input[:8]
        for example in sorted(examples, key=lambda item: item.subject_id)[:5]
    }
    _write_json(output_root / "dataset_preview.json", preview)
    return summary


def set_seed(seed: int, *, deterministic: bool = False) -> None:
    if deterministic:
        set_determinism(seed)
        return
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class _AuthorDualEncoderCollator:
    """Author-equivalent collator with explicit tokenizer options for newer embedding backbones."""

    def __init__(self, *, config: AuthorDualEncoderConfig) -> None:
        tokenizer_kwargs = _transformer_config_kwargs(config)
        self.st_tokenizer = AutoTokenizer.from_pretrained(config.transformer_pretrained_id, **tokenizer_kwargs)
        if config.pooling_strategy == "last_token":
            self.st_tokenizer.padding_side = "left"
        if self.st_tokenizer.pad_token is None:
            self.st_tokenizer.pad_token = self.st_tokenizer.eos_token
        self.st_max_token_len = config.st_max_token_length
        self.prefix_tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast=True)
        self.pre_seq_len = config.pre_seq_len
        self.prefix_max_token_len = config.prefix_max_token_length - self.pre_seq_len

    def encode_sample(self, sample: list[str], tokenizer: object, max_token_len: int) -> torch.Tensor:
        encoded_sample = tokenizer(
            sample,
            padding="max_length",
            truncation=True,
            max_length=max_token_len,
            return_tensors="pt",
        )
        input_ids = encoded_sample["input_ids"]
        attention_mask = encoded_sample["attention_mask"]
        return torch.cat((input_ids.unsqueeze(1), attention_mask.unsqueeze(1)), dim=1)

    def __call__(self, batch: list[dict[str, object]]) -> dict[str, torch.Tensor]:
        batch_sentences = [sample["text_input"] for sample in batch]
        batch_interview_length = torch.tensor([len(sentences) for sentences in batch_sentences])
        max_interview_length = int(batch_interview_length.max().item())
        batch_size = len(batch)

        batch_st_input = torch.zeros(
            size=(batch_size, max_interview_length, 2, self.st_max_token_len),
            dtype=torch.long,
        )
        batch_prefix_input = torch.zeros(
            size=(batch_size, max_interview_length, 2, self.prefix_max_token_len),
            dtype=torch.long,
        )

        for sample_idx, sample in enumerate(batch_sentences):
            batch_st_input[sample_idx][: len(sample)] = self.encode_sample(
                sample,
                self.st_tokenizer,
                self.st_max_token_len,
            )
            batch_prefix_input[sample_idx][: len(sample)] = self.encode_sample(
                sample,
                self.prefix_tokenizer,
                self.prefix_max_token_len,
            )

        return {
            "st_inputs": batch_st_input,
            "prefix_inputs": batch_prefix_input,
            "interview_lengths": batch_interview_length,
            "labels": torch.tensor([sample["score_label"] for sample in batch]),
        }


def resolve_public_daic_source_root(repo_root: Path) -> Path:
    candidates = [
        repo_root.expanduser().resolve() / "data" / "daic_woz",
        repo_root.expanduser().resolve().parent / "EmoTrack" / "data" / "daic_woz",
    ]
    for candidate in candidates:
        if (candidate / "labels.csv").is_file():
            return candidate
    attempted = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find public DAIC-WOZ source root. Looked in: {attempted}")


def _run_epoch(
    *,
    model: torch.nn.Module,
    model_input: str,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    include_predictions: bool = False,
) -> dict[str, object]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_count = 0
    predictions: list[float] = []
    labels: list[float] = []

    for batch in loader:
        batch = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            if model_input == "prefix":
                outputs = model(
                    inputs=batch["prefix_inputs"],
                    interview_lengths=batch["interview_lengths"],
                    labels=batch["labels"],
                )
            elif model_input == "dual":
                outputs = model(**batch)
            else:
                raise ValueError(f"Unsupported model_input: {model_input}")
            loss = outputs.loss
            logits = outputs.logits.detach().view(-1).cpu().numpy().astype(float)
            target = batch["labels"].detach().view(-1).cpu().numpy().astype(float)
            if training:
                loss.backward()
                optimizer.step()
        batch_size = int(batch["labels"].numel())
        total_loss += float(loss.detach().cpu()) * batch_size
        total_count += batch_size
        predictions.extend(float(value) for value in logits.tolist())
        labels.extend(float(value) for value in target.tolist())

    pred_arr = np.asarray(predictions, dtype=float)
    label_arr = np.asarray(labels, dtype=float)
    errors = pred_arr - label_arr
    metrics: dict[str, object] = {
        "loss": total_loss / max(total_count, 1),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "count": total_count,
    }
    if include_predictions:
        metrics["predictions"] = predictions
        metrics["labels"] = labels
    return metrics


def _train_regression_model(
    *,
    model: torch.nn.Module,
    model_input: str,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    device: torch.device,
    config: AuthorDualEncoderConfig,
    stage_name: str,
    best_path: Path,
    metrics_path: Path,
) -> dict[str, object]:
    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    best_dev_loss = float("inf")
    best_dev_mae = float("inf")
    best_epoch = -1
    patience_used = 0
    history: list[dict[str, object]] = []

    if metrics_path.exists():
        metrics_path.unlink()

    for epoch in range(config.num_epoch):
        train_metrics = _run_epoch(
            model=model,
            model_input=model_input,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
        )
        dev_metrics = _run_epoch(
            model=model,
            model_input=model_input,
            loader=dev_loader,
            device=device,
        )
        row = {
            "stage": stage_name,
            "epoch": epoch,
            "train": train_metrics,
            "dev": dev_metrics,
        }
        history.append(row)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(
            "{stage} epoch={epoch} train_loss={train_loss:.4f} dev_loss={dev_loss:.4f} dev_mae={dev_mae:.4f}".format(
                stage=stage_name,
                epoch=epoch,
                train_loss=train_metrics["loss"],
                dev_loss=dev_metrics["loss"],
                dev_mae=dev_metrics["mae"],
            ),
            flush=True,
        )

        # The paper selects and early-stops by validation loss.
        if float(dev_metrics["loss"]) < best_dev_loss:
            best_dev_loss = float(dev_metrics["loss"])
            best_dev_mae = float(dev_metrics["mae"])
            best_epoch = epoch
            patience_used = 0
            torch.save(
                {
                    "stage": stage_name,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "config": asdict(config),
                    "dev_metrics": dev_metrics,
                },
                best_path,
            )
        else:
            patience_used += 1
            if patience_used >= config.es_patience:
                break

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return {
        "stage": stage_name,
        "best_epoch": best_epoch,
        "best_dev_loss": best_dev_loss,
        "best_dev_mae": best_dev_mae,
        "checkpoint_path": str(best_path),
        "metrics_path": str(metrics_path),
        "epochs_completed": len(history),
        "early_stopping_patience": config.es_patience,
        "selection_metric": "dev_loss",
    }


def _load_author_qr_texts(path: Path) -> tuple[list[str], str]:
    rows = _parse_transcript_rows(path)
    if rows and not any(speaker == "Ellie" for speaker, _text in rows):
        texts = [text for speaker, text in rows if speaker == "Participant"]
        if not texts:
            raise ValueError(f"Participant-only transcript is empty after cleaning: {path}")
        return texts, "participant_only"
    collapsed = _collapse_participant_responses(rows)
    texts = _to_qr_pair_texts(collapsed)
    if not texts:
        raise ValueError(f"Transcript is empty after author QR preprocessing: {path}")
    return texts, "qr_pairs"


def _parse_transcript_rows(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if ":" not in line:
            continue
        raw_speaker, raw_text = line.split(":", 1)
        speaker = raw_speaker.strip()
        if speaker.lower() not in {"ellie", "participant"}:
            continue
        text = _clean_author_text(raw_text, speaker=speaker)
        if text is None:
            continue
        rows.append(("Ellie" if speaker.lower() == "ellie" else "Participant", text))
    return rows


def _clean_author_text(text: str, *, speaker: str) -> str | None:
    value = text.strip()
    if speaker.lower() == "ellie" and "(" in value:
        value = value.split("(")[-1].strip(")")
    value = value.replace("<laughter>", "*laughter*")
    value = value.replace("[laughter]", "*laughter*")
    value = value.replace("<sigh>", "*sigh*")
    value = re.sub(r"<sync>|<sync\.|<synch>|\[syncing\]|\[sync\]|\[synch\]|\[synching\]", "", value)
    value = re.sub(r"<.*?>", "", value)
    value = re.sub(r"\[.*?\]", "", value)
    value = value.replace("scrubbed_entry", "")
    value = value.replace("xxxx", "")
    value = value.replace("xxx", "")
    value = value.replace("_", "")
    value = re.sub(r" +", " ", value).strip().lower()
    if not value or value in {"laughter", "*laughter*"}:
        return None
    if START_PROMPTS.search(value) or CLOSING_PROMPTS.search(value):
        return None
    return value


def _collapse_participant_responses(rows: list[tuple[str, str]]) -> list[tuple[str, str]]:
    collapsed: list[tuple[str, str]] = []
    for speaker, text in rows:
        if speaker == "Participant" and collapsed and collapsed[-1][0] == "Participant":
            previous_speaker, previous_text = collapsed[-1]
            collapsed[-1] = (previous_speaker, f"{previous_text}, {text}")
        else:
            collapsed.append((speaker, text))
    return collapsed


def _to_qr_pair_texts(rows: list[tuple[str, str]]) -> list[str]:
    ellie_indices = [index for index, (speaker, _text) in enumerate(rows) if speaker == "Ellie"]
    if not ellie_indices:
        return [text for speaker, text in rows if speaker == "Participant"]
    first_ellie_index = ellie_indices[0]
    participant_indices = [
        index
        for index, (speaker, _text) in enumerate(rows)
        if speaker == "Participant" and index > first_ellie_index and index > 0
    ]
    qr_pairs: list[str] = []
    for index in participant_indices:
        question_speaker, question = rows[index - 1]
        if question_speaker != "Ellie":
            continue
        _response_speaker, response = rows[index]
        qr_pairs.append(f"{question} {response}")
    return qr_pairs


def _resolve_torch_dtype(raw_dtype: str) -> torch.dtype | None:
    value = str(raw_dtype or "auto").strip().lower()
    if value == "auto":
        return None
    if value == "float16":
        return torch.float16
    if value == "float32":
        return torch.float32
    if value == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported Dual Encoder torch dtype: {raw_dtype!r}")


def _resolved_config_torch_dtype(config: object) -> torch.dtype | None:
    raw_dtype = getattr(config, "torch_dtype", "auto")
    if raw_dtype is None:
        return None
    if isinstance(raw_dtype, torch.dtype):
        return raw_dtype
    return _resolve_torch_dtype(str(raw_dtype))


def _transformer_config_kwargs(config: object) -> dict[str, object]:
    if bool(getattr(config, "trust_remote_code", False)):
        return {"trust_remote_code": True}
    return {}


def _transformer_model_kwargs(config: object) -> dict[str, object]:
    kwargs = _transformer_config_kwargs(config)
    torch_dtype = _resolved_config_torch_dtype(config)
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    return kwargs


def _last_token_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    batch_size = hidden_states.size(0)
    if attention_mask[:, -1].sum() == batch_size:
        return hidden_states[:, -1]
    seq_lengths = attention_mask.sum(dim=1)
    return hidden_states[torch.arange(batch_size, device=hidden_states.device), seq_lengths - 1]


def _patch_author_dual_encoder_runtime(*, dual_module: object) -> None:
    prefix_module = importlib.import_module("model.prefix_encoder")

    base_cls = prefix_module.InterviewBaseModel
    if not getattr(base_cls, "_emotrack_dynamic_transformer_patched", False):

        def interview_base_init(self: torch.nn.Module, config: object) -> None:
            torch.nn.Module.__init__(self)
            self.config = config
            self.transformer_pretrained_id = config.transformer_pretrained_id
            self.device = config.device
            self.num_labels = config.num_labels

            self.transformer_encoder = AutoModel.from_pretrained(
                self.transformer_pretrained_id,
                config=config,
                **_transformer_model_kwargs(config),
            )
            for param in self.transformer_encoder.parameters():
                param.requires_grad = False

            self.lstm = torch.nn.LSTM(
                input_size=config.lstm_input_size,
                hidden_size=config.lstm_hidden_size,
                num_layers=config.lstm_num_layers,
                dropout=config.dropout_prob,
                batch_first=True,
                bidirectional=True,
            )
            prefix_module.init_rnn(self.lstm)

            self.attention = prefix_module.AttentionLayer(
                device=self.device,
                hidden_size=config.lstm_hidden_size,
                bidirectional=True,
            )

            self.dropout = torch.nn.Dropout(config.dropout_prob)
            self.prediction_head = torch.nn.Linear(config.lstm_hidden_size * 2, config.num_labels)
            self.prediction_head.apply(prefix_module.init_linear_layer)

        base_cls.__init__ = interview_base_init
        base_cls._emotrack_dynamic_transformer_patched = True

    dual_cls = dual_module.DualEncoderModel
    if not getattr(dual_cls, "_emotrack_dynamic_projection_patched", False):

        def dual_init(self: torch.nn.Module, config: object) -> None:
            prefix_module.PrefixModel.__init__(self, config)
            st_hidden_size = int(getattr(config, "hidden_size", 768))
            self.st_projection_layer = torch.nn.Linear(st_hidden_size, self.encoding_projection_size)
            self.st_projection_layer.apply(dual_module.init_linear_layer)

            for param in self.prefix_encoder.parameters():
                param.requires_grad = False

        def pool_text_embeddings(
            self: torch.nn.Module,
            model_output: object,
            attention_mask: torch.Tensor,
        ) -> torch.Tensor:
            token_embeddings = model_output[0]
            strategy = str(getattr(self.config, "dual_encoder_pooling_strategy", "mean"))
            if strategy == "last_token":
                return _last_token_pool(token_embeddings, attention_mask).float()

            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            pooled = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                input_mask_expanded.sum(1),
                min=1e-9,
            )
            return pooled.float()

        dual_cls.__init__ = dual_init
        dual_cls.mean_pooling = pool_text_embeddings
        dual_cls._emotrack_dynamic_projection_patched = True

    if not getattr(dual_module.init_dual_encoder, "_emotrack_config_kwargs_patched", False):

        def init_dual_encoder(hparam: object) -> torch.nn.Module:
            config = AutoConfig.from_pretrained(
                hparam.transformer_pretrained_id,
                **_transformer_config_kwargs(hparam),
            )
            config.transformer_pretrained_id = hparam.transformer_pretrained_id
            config.lstm_input_size = hparam.lstm_input_size
            config.lstm_hidden_size = hparam.lstm_hidden_size
            config.lstm_num_layers = hparam.lstm_num_layers
            config.dropout_prob = hparam.dropout_prob
            config.problem_type = hparam.problem_type
            config.num_labels = hparam.num_labels
            config.encoding_projection_size = hparam.encoding_projection_size
            config.pre_seq_len = hparam.pre_seq_len
            config.device = hparam.device
            config.dual_encoder_pooling_strategy = getattr(hparam, "pooling_strategy", "mean")
            config.torch_dtype = _resolve_torch_dtype(str(getattr(hparam, "torch_dtype", "auto")))
            config.trust_remote_code = getattr(hparam, "trust_remote_code", False)

            model = dual_module.DualEncoderModel(config=config)
            if len(hparam.device_id) > 1:
                model = torch.nn.DataParallel(model)
            return model

        init_dual_encoder._emotrack_config_kwargs_patched = True
        dual_module.init_dual_encoder = init_dual_encoder


def _patch_author_transformers_compat() -> None:
    prefix_module = importlib.import_module("model.prefix_encoder")
    prefix_cls = prefix_module.RobertaPrefixEncoder
    if getattr(prefix_cls, "_emotrack_transformers5_patched", False):
        return
    prefix_cls.all_tied_weights_keys = {}

    def from_pretrained(cls, pretrained_model_name_or_path: str, *args: object, config: object | None = None, **kwargs: object):
        if config is None:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        model = cls(config)
        model.roberta = RobertaModel.from_pretrained(pretrained_model_name_or_path, config=config)
        for param in model.roberta.parameters():
            param.requires_grad = False
        return model

    def get_prompt(self, batch_size: int) -> DynamicCache:
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.roberta.device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd,
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        legacy_cache = [(layer_cache[0], layer_cache[1]) for layer_cache in past_key_values]
        return DynamicCache(ddp_cache_data=legacy_cache)

    prefix_cls.from_pretrained = classmethod(from_pretrained)
    prefix_cls.get_prompt = get_prompt
    prefix_cls._emotrack_transformers5_patched = True


def _init_author_prefix_model(*, config: AuthorDualEncoderConfig, device: torch.device) -> torch.nn.Module:
    prefix_module = importlib.import_module("model.prefix_encoder")
    model_config = AutoConfig.from_pretrained(
        config.transformer_pretrained_id,
        **_transformer_config_kwargs(config),
    )
    model_config.transformer_pretrained_id = config.transformer_pretrained_id
    model_config.lstm_input_size = config.encoding_projection_size
    model_config.lstm_hidden_size = config.lstm_hidden_size
    model_config.lstm_num_layers = config.lstm_num_layers
    model_config.dropout_prob = config.dropout_prob
    model_config.problem_type = config.problem_type
    model_config.num_labels = config.num_labels
    model_config.encoding_projection_size = config.encoding_projection_size
    model_config.pre_seq_len = config.pre_seq_len
    model_config.device = device
    model_config.dual_encoder_pooling_strategy = config.pooling_strategy
    model_config.torch_dtype = _resolved_config_torch_dtype(config)
    model_config.trust_remote_code = config.trust_remote_code
    return prefix_module.PrefixModel(config=model_config)


def _copy_prefix_only_weights_to_dual(*, prefix_model: torch.nn.Module, dual_model: torch.nn.Module) -> None:
    # This follows the paper's warm-start step: initialize the fusion model
    # from a trained prefix-only model before adding the ST branch.
    modules_to_copy = (
        "prefix_encoder",
        "prefix_projection_layer",
        "lstm",
        "attention",
        "prediction_head",
    )
    for module_name in modules_to_copy:
        getattr(dual_model, module_name).load_state_dict(getattr(prefix_model, module_name).state_dict())


def _to_author_hparam(*, config: AuthorDualEncoderConfig, device: torch.device) -> SimpleNamespace:
    return SimpleNamespace(
        seed=config.seed,
        device_id=[0],
        batch_size=config.batch_size,
        problem_type=config.problem_type,
        num_labels=config.num_labels,
        transformer_pretrained_id=config.transformer_pretrained_id,
        st_max_token_length=config.st_max_token_length,
        prefix_max_token_length=config.prefix_max_token_length,
        pre_seq_len=config.pre_seq_len,
        prefix_backbone="roberta",
        fusion_method="avg",
        encoding_projection_size=config.encoding_projection_size,
        lstm_input_size=config.lstm_input_size,
        lstm_hidden_size=config.lstm_hidden_size,
        lstm_num_layers=config.lstm_num_layers,
        dropout_prob=config.dropout_prob,
        optimizer="AdamW",
        learning_rate=config.learning_rate,
        num_epoch=config.num_epoch,
        es_patience=config.es_patience,
        device=device,
        pooling_strategy=config.pooling_strategy,
        torch_dtype=config.torch_dtype,
        trust_remote_code=config.trust_remote_code,
    )


def _validate_author_repo(repo_root: Path) -> None:
    required = [
        repo_root / "dataset" / "dataset.py",
        repo_root / "dataset" / "preprocess.py",
        repo_root / "model" / "dual_encoder.py",
        repo_root / "model" / "prefix_encoder.py",
        repo_root / "config" / "dual_encoder.yaml",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Author dual encoder repo is incomplete under {repo_root}: missing {missing}")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@contextmanager
def _sys_path(path: Path) -> Iterator[None]:
    raw = str(path.expanduser().resolve())
    sys.path.insert(0, raw)
    try:
        yield
    finally:
        try:
            sys.path.remove(raw)
        except ValueError:
            pass
