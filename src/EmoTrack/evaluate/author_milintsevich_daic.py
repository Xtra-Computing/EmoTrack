from __future__ import annotations

import csv
import importlib
import json
import os
import re
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Iterator

import torch
from transformers import AutoTokenizer


AUTHOR_REPO_URL = "https://git.unicaen.fr/kirill.milintsevich/hierarchical-depression-symptom-classifier.git"
_BROKEN_PARTICIPANT_IDS = {"451", "458"}
_LABEL_COLUMNS = (
    "phq8_no_interest",
    "phq8_depressed",
    "phq8_sleep",
    "phq8_tired",
    "phq8_appetite",
    "phq8_failure",
    "phq8_concentrating",
    "phq8_moving",
)


@dataclass(frozen=True)
class PublicDaicExample:
    participant_id: str
    split: str
    gender: int
    phq_binary: int
    phq_score: int
    turns: list[str]
    symptom_scores: list[int] | None


def clone_author_repo(*, repo_root: Path, clone_url: str = AUTHOR_REPO_URL) -> Path:
    resolved = repo_root.expanduser().resolve()
    if resolved.exists():
        _validate_author_repo(resolved)
        return resolved
    resolved.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", clone_url, str(resolved)],
        check=True,
    )
    _validate_author_repo(resolved)
    return resolved


def prepare_author_milintsevich_daic_dataset(
    *,
    source_root: Path,
    author_repo_root: Path,
    output_root: Path,
) -> dict[str, object]:
    rows = _load_public_daic_rows(source_root.expanduser().resolve())
    resolved_author_root = author_repo_root.expanduser().resolve()
    resolved_output_root = output_root.expanduser().resolve()
    data_lines_root = resolved_author_root / "data" / "json" / "lines"
    data_lines_root.mkdir(parents=True, exist_ok=True)
    resolved_output_root.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, list[dict[str, object]]] = {"train": [], "dev": [], "test": []}
    skipped: dict[str, list[dict[str, object]]] = {"train": [], "dev": [], "test": []}
    public_test_totals: list[dict[str, object]] = []

    for row in rows:
        if row.split not in grouped:
            raise ValueError(f"Unsupported public DAIC split {row.split!r} for participant {row.participant_id}")
        payload = {
            "id": int(row.participant_id),
            "turns": list(row.turns),
            "labels": list(row.symptom_scores or []),
        }
        if row.split == "test":
            grouped["test"].append(payload)
            public_test_totals.append(
                {
                    "participant_id": row.participant_id,
                    "phq_score": row.phq_score,
                    "gender": row.gender,
                }
            )
            continue

        if row.participant_id in _BROKEN_PARTICIPANT_IDS:
            skipped[row.split].append(
                {
                    "participant_id": row.participant_id,
                    "reason": "broken_transcript_excluded_by_author_code",
                }
            )
            continue
        if row.symptom_scores is None:
            skipped[row.split].append(
                {
                    "participant_id": row.participant_id,
                    "reason": "missing_symptom_labels",
                }
            )
            continue
        grouped[row.split].append(payload)

    _write_jsonl(data_lines_root / "train.jsonl", grouped["train"])
    _write_jsonl(data_lines_root / "validation.jsonl", grouped["dev"])
    _write_jsonl(data_lines_root / "test.jsonl", grouped["test"])

    _write_label_csv(
        resolved_author_root / "data" / "train_split_Depression_AVEC2017.csv",
        split_rows=grouped["train"],
        split_name="train",
        public_rows_by_id={row.participant_id: row for row in rows},
    )
    _write_label_csv(
        resolved_author_root / "data" / "dev_split_Depression_AVEC2017.csv",
        split_rows=grouped["dev"],
        split_name="dev",
        public_rows_by_id={row.participant_id: row for row in rows},
    )

    public_test_totals_path = resolved_output_root / "public_test_totals.json"
    public_test_totals_path.write_text(
        json.dumps(public_test_totals, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    summary = {
        "source_root": str(source_root.expanduser().resolve()),
        "author_repo_root": str(resolved_author_root),
        "data_lines_root": str(data_lines_root),
        "public_test_totals_path": str(public_test_totals_path),
        "counts": {
            "train": len(grouped["train"]),
            "dev": len(grouped["dev"]),
            "test": len(grouped["test"]),
        },
        "skipped": skipped,
        "preprocess_note": (
            "Public *.txt transcripts are converted with the author prepare_data.py line-cleaning rules: "
            "Ellie prompt identifiers are stripped to parenthesized prompt text, sync/annotation rows are "
            "removed, and no lowercasing or QR pairing is applied."
        ),
    }
    (resolved_output_root / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def write_author_runtime_shims(*, author_repo_root: Path) -> None:
    resolved = author_repo_root.expanduser().resolve()

    (resolved / "tomli.py").write_text(
        "from tomllib import *\n",
        encoding="utf-8",
    )

    torchinfo_root = resolved / "torchinfo"
    torchinfo_root.mkdir(parents=True, exist_ok=True)
    (torchinfo_root / "__init__.py").write_text(
        "def summary(*args, **kwargs):\n"
        "    model = args[0] if args else None\n"
        "    return f\"torchinfo summary shim ({type(model).__name__ if model is not None else 'unknown'})\"\n",
        encoding="utf-8",
    )

    matplotlib_root = resolved / "matplotlib"
    matplotlib_root.mkdir(parents=True, exist_ok=True)
    (matplotlib_root / "__init__.py").write_text("", encoding="utf-8")
    (matplotlib_root / "pyplot.py").write_text(
        "class _NoOpFigure:\n"
        "    pass\n"
        "\n"
        "def figure(*args, **kwargs):\n"
        "    return _NoOpFigure()\n"
        "\n"
        "def __getattr__(name):\n"
        "    def _noop(*args, **kwargs):\n"
        "        return None\n"
        "    return _noop\n",
        encoding="utf-8",
    )
    (matplotlib_root / "lines.py").write_text(
        "class Line2D:\n"
        "    def __init__(self, *args, **kwargs):\n"
        "        pass\n",
        encoding="utf-8",
    )

    torchmetrics_root = resolved / "torchmetrics"
    torchmetrics_root.mkdir(parents=True, exist_ok=True)
    (torchmetrics_root / "__init__.py").write_text("", encoding="utf-8")
    (torchmetrics_root / "functional.py").write_text(
        "from __future__ import annotations\n"
        "\n"
        "import numpy as np\n"
        "import torch\n"
        "from sklearn.metrics import f1_score as _sk_f1_score\n"
        "\n"
        "def _as_tensor(value, *, dtype=torch.float32):\n"
        "    if isinstance(value, torch.Tensor):\n"
        "        return value.detach().to(dtype=dtype)\n"
        "    return torch.as_tensor(value, dtype=dtype)\n"
        "\n"
        "def mean_absolute_error(preds, target):\n"
        "    preds_t = _as_tensor(preds)\n"
        "    target_t = _as_tensor(target)\n"
        "    return torch.mean(torch.abs(preds_t - target_t))\n"
        "\n"
        "def mean_squared_error(preds, target):\n"
        "    preds_t = _as_tensor(preds)\n"
        "    target_t = _as_tensor(target)\n"
        "    return torch.mean(torch.pow(preds_t - target_t, 2))\n"
        "\n"
        "def accuracy(preds, target):\n"
        "    preds_t = torch.as_tensor(preds).detach().cpu()\n"
        "    target_t = torch.as_tensor(target).detach().cpu()\n"
        "    return (preds_t == target_t).to(dtype=torch.float32).mean()\n"
        "\n"
        "def f1_score(preds, target, average='micro', num_classes=None, multiclass=None, task=None, num_labels=None):\n"
        "    preds_t = torch.as_tensor(preds).detach().cpu()\n"
        "    target_t = torch.as_tensor(target).detach().cpu()\n"
        "    if preds_t.ndim > 1 and task != 'multilabel':\n"
        "        preds_t = torch.argmax(preds_t, dim=1)\n"
        "    score = _sk_f1_score(\n"
        "        target_t.numpy().astype(np.int64),\n"
        "        preds_t.numpy().astype(np.int64),\n"
        "        average=average,\n"
        "        zero_division=0,\n"
        "    )\n"
        "    return torch.tensor(float(score), dtype=torch.float32)\n",
        encoding="utf-8",
    )


def patch_author_repo_for_modern_datasets(*, author_repo_root: Path) -> None:
    resolved = author_repo_root.expanduser().resolve()
    train_path = resolved / "train.py"
    raw_train = train_path.read_text(encoding="utf-8")
    needle = '    dataset = load_dataset("daic_woz.py", "lines")\n'
    replacement = (
        '    data_files = {\n'
        '        "train": "data/json/lines/train.jsonl",\n'
        '        "validation": "data/json/lines/validation.jsonl",\n'
        '        "test": "data/json/lines/test.jsonl",\n'
        "    }\n"
        '    dataset = load_dataset("json", data_files=data_files)\n'
    )
    if replacement in raw_train:
        return
    if needle not in raw_train:
        raise RuntimeError(f"Could not find the expected dataset-loader block in {train_path}")
    train_path.write_text(raw_train.replace(needle, replacement), encoding="utf-8")


def patch_author_repo_for_early_stopping(*, author_repo_root: Path) -> None:
    resolved = author_repo_root.expanduser().resolve()
    train_path = resolved / "train.py"
    raw_train = train_path.read_text(encoding="utf-8")
    raw_train = raw_train.replace(
        "                            should_stop = True\n"
        "                            break\n"
        "                                improved = True\n",
        "                            should_stop = True\n"
        "                            break\n",
    )
    if "early_stopping_patience" in raw_train:
        compile(raw_train, str(train_path), "exec")
        train_path.write_text(raw_train, encoding="utf-8")
        return

    raw_train = raw_train.replace(
        '    best_f1 = 0.0\n',
        '    best_f1 = 0.0\n'
        '    early_stopping_patience = int(config.get("early_stopping_patience", 0) or 0)\n'
        '    early_stopping_wait = 0\n'
        '    should_stop = False\n',
    )
    raw_train = raw_train.replace(
        '        for epoch in range(config["num_iters"]):\n',
        '        for epoch in range(config["num_iters"]):\n'
        "            if should_stop:\n"
        "                break\n",
    )
    raw_train = raw_train.replace(
        '                    if config["save_every_epoch"]:\n'
        '                        save_path = save_dir / f"model_{config[\'seed\']}_{epoch}.pt"\n'
        "                        save_model(model, save_path, accelerator)\n"
        "                    else:\n",
        '                    improved = False\n'
        '                    if config["save_every_epoch"]:\n'
        '                        save_path = save_dir / f"model_{config[\'seed\']}_{epoch}.pt"\n'
        "                        save_model(model, save_path, accelerator)\n"
        "                        improved = True\n"
        "                    else:\n",
    )
    raw_train = raw_train.replace(
        "                                save_model(model, save_path, accelerator)\n"
        "                                best_loss = average_dev_loss\n",
        "                                save_model(model, save_path, accelerator)\n"
        "                                best_loss = average_dev_loss\n"
        "                                improved = True\n",
    )
    raw_train = raw_train.replace(
        "                                save_model(model, save_path, accelerator)\n"
        "                                best_loss = mae\n",
        "                                save_model(model, save_path, accelerator)\n"
        "                                best_loss = mae\n"
        "                                improved = True\n",
    )
    raw_train = raw_train.replace(
        "                                save_model(model, save_path, accelerator)\n"
        "                                best_f1 = f1_macro\n",
        "                                save_model(model, save_path, accelerator)\n"
        "                                best_f1 = f1_macro\n"
        "                                improved = True\n",
    )
    raw_train = raw_train.replace(
        "                                best_f1 = f1_macro\n"
        "                                improved = True\n",
        "                                best_f1 = f1_macro\n"
        "                                improved = True\n"
        "                    if improved:\n"
        "                        early_stopping_wait = 0\n"
        "                    elif early_stopping_patience > 0:\n"
        "                        early_stopping_wait += 1\n"
        "                        if early_stopping_wait >= early_stopping_patience:\n"
        '                            accelerator.print(f"Early stopping triggered after {early_stopping_wait} evaluations without improvement.")\n'
        "                            should_stop = True\n"
        "                            break\n",
        1,
    )
    if "early_stopping_patience" not in raw_train:
        raise RuntimeError(f"Could not patch early stopping support into {train_path}")
    compile(raw_train, str(train_path), "exec")
    train_path.write_text(raw_train, encoding="utf-8")


def write_author_config(
    *,
    author_repo_root: Path,
    save_dir: Path,
    seed: int,
    num_iters: int,
    batch_size: int,
    early_stopping_patience: int = 0,
    deterministic: bool = False,
) -> Path:
    resolved_author_root = author_repo_root.expanduser().resolve()
    resolved_save_dir = save_dir.expanduser().resolve()
    resolved_save_dir.mkdir(parents=True, exist_ok=True)
    config_path = resolved_author_root / "config.toml"
    config_path.write_text(
        (
            f'save_dir = "{resolved_save_dir.as_posix()}"\n'
            f"batch_size = {batch_size}\n"
            f"seed = {seed}\n"
            f"deterministic = {str(bool(deterministic)).lower()}\n"
            f"early_stopping_patience = {early_stopping_patience}\n"
            "five_classes = false\n"
            "save_every_epoch = false\n"
            "regularization_loss = false\n"
            "loss_l = 0.1\n"
            "lr = 3e-5\n"
            f"num_iters = {num_iters}\n"
            'encoder_layers_to_freeze = [ ["embeddings"], [0, 1, 2, 3, 4] ]\n'
            "\n"
            "[model]\n"
            'bert_model = "sentence-transformers/all-distilroberta-v1"\n'
            "encoder_hidden_dim = 300\n"
            "encoder_num_layers = 1\n"
            "dropout = 0.5\n"
            "num_classes = 8\n"
            'attention_type = "hierarchical"\n'
            'pooling = "mean"\n'
            "binary_only = true\n"
            "bidirectional = true\n"
            "multilabel = true\n"
            "regression = false\n"
        ),
        encoding="utf-8",
    )
    return config_path


def run_author_training(
    *,
    author_repo_root: Path,
    python_executable: str,
    seed: int | None = None,
    deterministic: bool = False,
) -> subprocess.CompletedProcess[bytes]:
    resolved = author_repo_root.expanduser().resolve()
    env = os.environ.copy()
    if deterministic:
        if seed is None:
            raise ValueError("seed is required when deterministic=True")
        env["PYTHONHASHSEED"] = str(int(seed))
        env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    return subprocess.run(
        [python_executable, "train.py"],
        cwd=resolved,
        check=True,
        env=env,
    )


def evaluate_public_test_total_mae(
    *,
    author_repo_root: Path,
    source_root: Path,
    checkpoint_path: Path,
    output_root: Path,
) -> dict[str, object]:
    resolved_author_root = author_repo_root.expanduser().resolve()
    resolved_output_root = output_root.expanduser().resolve()
    resolved_output_root.mkdir(parents=True, exist_ok=True)
    public_rows = _load_public_daic_rows(source_root.expanduser().resolve())
    test_rows = [row for row in public_rows if row.split == "test"]
    if not test_rows:
        raise ValueError("Public DAIC test split is empty")

    checkpoint = checkpoint_path.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing author checkpoint: {checkpoint}")

    model, bert_model_name = _load_author_model(checkpoint, author_repo_root=resolved_author_root)
    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    device = next(model.parameters()).device

    predictions: list[dict[str, object]] = []
    errors: list[float] = []

    with torch.no_grad():
        for row in test_rows:
            encoded = tokenizer(
                row.turns,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            inputs = {
                "input_ids": encoded["input_ids"].to(device),
                "attention_mask": encoded["attention_mask"].to(device),
                "text_lens": torch.tensor([int(encoded["input_ids"].shape[0])], dtype=torch.long),
            }
            outputs = model(inputs).detach().cpu().squeeze(0)
            predicted_items = [round(float(value), 4) for value in outputs.tolist()]
            predicted_total = round(float(outputs.sum().item()), 4)
            error = abs(predicted_total - float(row.phq_score))
            errors.append(error)
            predictions.append(
                {
                    "participant_id": row.participant_id,
                    "ground_truth_total": row.phq_score,
                    "predicted_total": predicted_total,
                    "absolute_error": round(error, 4),
                    "predicted_items": predicted_items,
                }
            )

    summary = {
        "checkpoint_path": str(checkpoint),
        "test_public_total_mae": float(fmean(errors)),
        "test_session_count": len(predictions),
        "predictions": predictions,
    }
    (resolved_output_root / "public_test_total_eval.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def read_best_dev_metrics(*, save_dir: Path, seed: int) -> dict[str, object] | None:
    model_dir = save_dir.expanduser().resolve() / author_model_name()
    log_path = model_dir / f"log_{author_model_name()}_{seed}.tsv"
    if not log_path.is_file():
        return None
    with log_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        return None
    best = min(rows, key=lambda row: float(row["dev_loss"]))
    return {
        "log_path": str(log_path),
        "best_epoch_by_dev_loss": int(best["epoch"]),
        "best_dev_loss": float(best["dev_loss"]),
        "best_dev_mae": float(best["dev_mae"]),
        "best_dev_mse": float(best["dev_mse"]),
        "best_f1_micro": float(best["f1_micro"]),
        "best_f1_macro": float(best["f1_macro"]),
    }


def author_model_name() -> str:
    return "robert_multilabel_no-regression_"


def author_checkpoint_path(*, save_dir: Path, seed: int) -> Path:
    return save_dir.expanduser().resolve() / author_model_name() / f"model_best_loss_{seed}.pt"


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


def _validate_author_repo(repo_root: Path) -> None:
    required = [
        repo_root / "train.py",
        repo_root / "daic_woz.py",
        repo_root / "hdsc" / "model.py",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Author repo is incomplete under {repo_root}: missing {missing}")


def _load_public_daic_rows(source_root: Path) -> list[PublicDaicExample]:
    labels_path = source_root / "labels.csv"
    if not labels_path.is_file():
        raise FileNotFoundError(f"Missing public DAIC labels file: {labels_path}")
    rows: list[PublicDaicExample] = []
    with labels_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            participant_id = str(row["participant_id"]).strip()
            split = str(row["split"]).strip().lower()
            transcript_path = source_root / split / f"{participant_id}.txt"
            if not transcript_path.is_file():
                raise FileNotFoundError(f"Missing public DAIC transcript: {transcript_path}")
            rows.append(
                PublicDaicExample(
                    participant_id=participant_id,
                    split=split,
                    gender=int(row["gender"]),
                    phq_binary=int(row["phq_binary"]),
                    phq_score=int(row["phq_score"]),
                    turns=_load_turn_texts(transcript_path),
                    symptom_scores=_parse_optional_symptom_scores(row),
                )
            )
    return rows


def _load_turn_texts(path: Path) -> list[str]:
    turns: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if ":" in line:
            raw_speaker, text = line.split(":", 1)
            speaker = raw_speaker.strip()
        else:
            speaker = ""
            text = line
        cleaned = _clean_author_turn_text(text, speaker=speaker)
        if cleaned:
            turns.append(cleaned)
    if not turns:
        raise ValueError(f"Public DAIC transcript is empty after parsing: {path}")
    return turns


def _clean_author_turn_text(text: str, *, speaker: str) -> str | None:
    value = text.strip()
    if speaker == "Ellie" and "(" in value:
        value = value.split("(")[-1].strip(")")
    if re.search(r"<sync>|<sync\.|<synch>|\[syncing\]|\[sync\]|\[synch\]|\[synching\]", value):
        return None
    value = value.replace("<laughter>", "*laughter*")
    value = value.replace("[laughter]", "*laughter*")
    value = value.replace("<sigh>", "*sigh*")
    value = re.sub(r"<.*?>", "", value)
    value = re.sub(r"\[.*?\]", "", value)
    value = value.replace("scrubbed_entry", "")
    value = value.replace("xxxx", "")
    value = value.replace("xxx", "")
    value = re.sub(r" +", " ", value).strip()
    if not value or value in {"laughter", "*laughter*"}:
        return None
    return value


def _parse_optional_symptom_scores(row: dict[str, str]) -> list[int] | None:
    scores: list[int] = []
    for column_name in _LABEL_COLUMNS:
        raw_value = str(row.get(column_name, "")).strip()
        if raw_value == "":
            return None
        scores.append(int(raw_value))
    return scores


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def _write_label_csv(
    path: Path,
    *,
    split_rows: list[dict[str, object]],
    split_name: str,
    public_rows_by_id: dict[str, PublicDaicExample],
) -> None:
    headers = [
        "Participant_ID",
        "PHQ8_Binary",
        "PHQ8_Score",
        "Gender",
        "PHQ8_NoInterest",
        "PHQ8_Depressed",
        "PHQ8_Sleep",
        "PHQ8_Tired",
        "PHQ8_Appetite",
        "PHQ8_Failure",
        "PHQ8_Concentrating",
        "PHQ8_Moving",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for row in split_rows:
            public_row = public_rows_by_id[str(row["id"])]
            if public_row.symptom_scores is None:
                raise ValueError(f"{split_name} row is missing symptom labels for participant {public_row.participant_id}")
            writer.writerow(
                [
                    public_row.participant_id,
                    public_row.phq_binary,
                    public_row.phq_score,
                    public_row.gender,
                    *public_row.symptom_scores,
                ]
            )


def _load_author_model(
    checkpoint_path: Path,
    *,
    author_repo_root: Path,
) -> tuple[torch.nn.Module, str]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    kwargs = dict(checkpoint["kwargs"])
    with _sys_path(author_repo_root):
        module = importlib.import_module("hdsc.model")
        model_cls = getattr(module, "PHQTotalMulticlassAttentionModelBERT")
    model = model_cls(device=device, **kwargs).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, str(kwargs["bert_model"])


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


def patch_author_repo_for_determinism(*, author_repo_root: Path) -> None:
    resolved = author_repo_root.expanduser().resolve()
    train_path = resolved / "train.py"
    raw_train = train_path.read_text(encoding="utf-8")
    if "_emotrack_apply_determinism" in raw_train:
        return

    raw_train = raw_train.replace(
        "import itertools\nimport json\n",
        "import itertools\nimport json\nimport os\nimport random\n",
    )
    raw_train = raw_train.replace(
        "import datasets\nimport tomli\nimport torch\n",
        "import datasets\nimport numpy as np\nimport tomli\nimport torch\n",
    )
    raw_train = raw_train.replace(
        'logger.setLevel("INFO")\n',
        'logger.setLevel("INFO")\n'
        "\n"
        "\n"
        "def _emotrack_seed_worker(worker_id):\n"
        "    worker_seed = torch.initial_seed() % 2**32\n"
        "    np.random.seed(worker_seed)\n"
        "    random.seed(worker_seed)\n"
        "\n"
        "\n"
        "def _emotrack_make_loader_generator(seed):\n"
        "    generator = torch.Generator()\n"
        "    generator.manual_seed(int(seed))\n"
        "    return generator\n"
        "\n"
        "\n"
        "def _emotrack_apply_determinism(config):\n"
        "    seed = int(config.get(\"seed\", 0))\n"
        "    os.environ.setdefault(\"PYTHONHASHSEED\", str(seed))\n"
        "    os.environ.setdefault(\"CUBLAS_WORKSPACE_CONFIG\", \":4096:8\")\n"
        "    random.seed(seed)\n"
        "    np.random.seed(seed)\n"
        "    torch.manual_seed(seed)\n"
        "    if torch.cuda.is_available():\n"
        "        torch.cuda.manual_seed_all(seed)\n"
        "    torch.backends.cudnn.deterministic = True\n"
        "    torch.backends.cudnn.benchmark = False\n"
        "    torch.use_deterministic_algorithms(True, warn_only=True)\n",
    )
    raw_train = raw_train.replace(
        'def main():\n'
        "    accelerator = Accelerator()\n"
        "    device = accelerator.device\n"
        "\n"
        '    with open("config.toml", "rb") as f:\n'
        "        config = tomli.load(f)\n",
        'def main():\n'
        '    with open("config.toml", "rb") as f:\n'
        "        config = tomli.load(f)\n"
        "\n"
        '    if config.get("deterministic", False):\n'
        "        _emotrack_apply_determinism(config)\n"
        "\n"
        "    accelerator = Accelerator()\n"
        "    device = accelerator.device\n",
    )
    raw_train = raw_train.replace(
        "    train_dataloader = torch.utils.data.DataLoader(\n"
        '        encoded_dataset["train"],\n'
        '        batch_size=config["batch_size"],\n'
        "        collate_fn=collate_fn,\n"
        "        shuffle=True,\n"
        "    )\n",
        "    train_dataloader_kwargs = {}\n"
        '    if config.get("deterministic", False):\n'
        "        train_dataloader_kwargs = {\n"
        '            "generator": _emotrack_make_loader_generator(config["seed"]),\n'
        '            "worker_init_fn": _emotrack_seed_worker,\n'
        "        }\n"
        "    train_dataloader = torch.utils.data.DataLoader(\n"
        '        encoded_dataset["train"],\n'
        '        batch_size=config["batch_size"],\n'
        "        collate_fn=collate_fn,\n"
        "        shuffle=True,\n"
        "        **train_dataloader_kwargs,\n"
        "    )\n",
    )
    if "_emotrack_apply_determinism" not in raw_train or "train_dataloader_kwargs" not in raw_train:
        raise RuntimeError(f"Could not patch deterministic support into {train_path}")
    train_path.write_text(raw_train, encoding="utf-8")
