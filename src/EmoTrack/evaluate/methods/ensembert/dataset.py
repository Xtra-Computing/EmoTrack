from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from EmoTrack.evaluate.core.scores import (
    SessionExampleRow,
    make_row_key,
)
from EmoTrack.generative.schema import PHQ8_SYMPTOMS
from EmoTrack.utils.paths import REPO_ROOT


DEFAULT_PHQ8_SCALE_PATH = REPO_ROOT / "configs" / "latent_variable" / "scales" / "phq8.json"
DEFAULT_ITEM_DESCRIPTIONS_PATH = REPO_ROOT / "configs" / "evaluate" / "ensembert" / "phq8_item_descriptions.json"


@dataclass(frozen=True)
class PHQ8ItemSpec:
    item_index: int
    symptom_name: str
    item_text: str
    description: str
    choice_texts: tuple[str, str, str, str]
    response_labels: tuple[str, str, str, str]


@dataclass(frozen=True)
class EnsemBERTItemExample:
    row_key: str
    run_id: str
    session_index: int
    item_index: int
    symptom_name: str
    label: int | None


def load_phq8_item_specs(
    *,
    scale_path: str | Path = DEFAULT_PHQ8_SCALE_PATH,
    descriptions_path: str | Path = DEFAULT_ITEM_DESCRIPTIONS_PATH,
) -> list[PHQ8ItemSpec]:
    resolved_scale_path = _resolve_repo_path(scale_path)
    resolved_descriptions_path = _resolve_repo_path(descriptions_path)
    scale_payload = json.loads(resolved_scale_path.read_text(encoding="utf-8"))
    descriptions_payload = json.loads(resolved_descriptions_path.read_text(encoding="utf-8"))

    raw_items = scale_payload.get("items")
    raw_options = scale_payload.get("response_options")
    timeframe = str(scale_payload.get("instrument", {}).get("timeframe_prompt", "")).strip()
    if not isinstance(raw_items, list) or len(raw_items) != len(PHQ8_SYMPTOMS):
        raise ValueError(f"PHQ-8 scale must define {len(PHQ8_SYMPTOMS)} items: {resolved_scale_path}")
    if not isinstance(raw_options, list) or len(raw_options) != 4:
        raise ValueError(f"PHQ-8 scale must define exactly 4 response options: {resolved_scale_path}")

    if isinstance(descriptions_payload, dict) and isinstance(descriptions_payload.get("items"), list):
        description_by_symptom = {
            str(item.get("symptom", "")).strip(): str(item.get("description", "")).strip()
            for item in descriptions_payload["items"]
            if isinstance(item, dict)
        }
    elif isinstance(descriptions_payload, dict):
        description_by_symptom = {
            str(key).strip(): str(value).strip()
            for key, value in descriptions_payload.items()
        }
    else:
        raise ValueError(f"PHQ-8 item descriptions must be a mapping or item list: {resolved_descriptions_path}")

    missing = [symptom for symptom in PHQ8_SYMPTOMS if not description_by_symptom.get(symptom)]
    if missing:
        raise ValueError(f"Missing PHQ-8 item descriptions for: {', '.join(missing)}")

    response_labels = tuple(str(option.get("label", "")).strip() for option in raw_options)
    specs: list[PHQ8ItemSpec] = []
    for item_index, (symptom_name, raw_item) in enumerate(zip(PHQ8_SYMPTOMS, raw_items, strict=True)):
        if not isinstance(raw_item, dict):
            raise ValueError(f"PHQ-8 item {item_index + 1} must be a mapping")
        item_text = str(raw_item.get("text", "")).strip()
        if not item_text:
            raise ValueError(f"PHQ-8 item {item_index + 1} has no text in {resolved_scale_path}")
        specs.append(
            PHQ8ItemSpec(
                item_index=item_index,
                symptom_name=symptom_name,
                item_text=item_text,
                description=description_by_symptom[symptom_name],
                choice_texts=tuple(
                    _choice_text(timeframe=timeframe, item_text=item_text, response_label=label)
                    for label in response_labels
                ),
                response_labels=response_labels,
            )
        )
    return specs


def extract_document_texts(
    transcript: str,
    *,
    document_source: str = "client_utterances",
    min_words: int = 2,
) -> list[str]:
    if document_source not in {"all_utterances", "client_utterances"}:
        raise ValueError(f"Unsupported EnsemBERT document_source: {document_source}")
    if min_words < 1:
        raise ValueError("min_words must be >= 1")

    texts: list[str] = []
    for raw_line in str(transcript or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("[Session "):
            continue
        speaker = ""
        text = line
        if ":" in line:
            speaker, text = line.split(":", 1)
            speaker = speaker.strip()
            text = text.strip()
        if document_source == "client_utterances" and speaker and speaker.lower() != "client":
            continue
        if not text or len(text.split()) < min_words:
            continue
        texts.append(text)
    return texts


def build_labeled_item_examples(
    rows: Iterable[SessionExampleRow],
    item_specs: list[PHQ8ItemSpec],
) -> list[EnsemBERTItemExample]:
    examples: list[EnsemBERTItemExample] = []
    for row in rows:
        raw_scores = row.target_item_scores
        for spec in item_specs:
            if spec.symptom_name not in raw_scores:
                continue
            examples.append(
                EnsemBERTItemExample(
                    row_key=make_row_key(row),
                    run_id=row.run_id,
                    session_index=row.session_index,
                    item_index=spec.item_index,
                    symptom_name=spec.symptom_name,
                    label=coerce_item_label(raw_scores[spec.symptom_name]),
                )
            )
    return examples


def build_prediction_item_examples(
    rows: Iterable[SessionExampleRow],
    item_specs: list[PHQ8ItemSpec],
) -> list[EnsemBERTItemExample]:
    examples: list[EnsemBERTItemExample] = []
    for row in rows:
        for spec in item_specs:
            examples.append(
                EnsemBERTItemExample(
                    row_key=make_row_key(row),
                    run_id=row.run_id,
                    session_index=row.session_index,
                    item_index=spec.item_index,
                    symptom_name=spec.symptom_name,
                    label=None,
                )
            )
    return examples


def coerce_item_label(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"PHQ-8 item score must be numeric, got {value!r}")
    if not math.isfinite(float(value)):
        raise ValueError(f"PHQ-8 item score must be finite, got {value!r}")
    return max(0, min(3, int(math.floor(float(value) + 0.5))))


def _choice_text(*, timeframe: str, item_text: str, response_label: str) -> str:
    prefix = f"{timeframe} " if timeframe else ""
    return f"{prefix}For the symptom '{item_text}', the selected response is '{response_label}'."


def _resolve_repo_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if resolved.is_absolute():
        return resolved
    return REPO_ROOT / resolved
