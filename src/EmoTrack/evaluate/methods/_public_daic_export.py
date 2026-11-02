from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

from EmoTrack.evaluate.core.scores import (
    SessionExampleRow,
    make_row_key,
)


PHQ8_ITEM_ORDER: tuple[str, ...] = (
    "anhedonia",
    "depressed_mood",
    "sleep",
    "fatigue",
    "appetite",
    "self_worth",
    "concentration",
    "psychomotor",
)
_PUBLIC_LABEL_COLUMNS: tuple[tuple[str, str], ...] = (
    ("anhedonia", "phq8_no_interest"),
    ("depressed_mood", "phq8_depressed"),
    ("sleep", "phq8_sleep"),
    ("fatigue", "phq8_tired"),
    ("appetite", "phq8_appetite"),
    ("self_worth", "phq8_failure"),
    ("concentration", "phq8_concentrating"),
    ("psychomotor", "phq8_moving"),
)
_LABEL_HEADERS = (
    "participant_id",
    "split",
    "gender",
    "phq_binary",
    "phq_score",
    *[column for _item_name, column in _PUBLIC_LABEL_COLUMNS],
)
_AUTHOR_SPEAKER_BY_CANONICAL = {
    "client": "Participant",
    "participant": "Participant",
    "patient": "Participant",
    "seeker": "Participant",
    "user": "Participant",
    "counselor": "Ellie",
    "counsellor": "Ellie",
    "therapist": "Ellie",
    "interviewer": "Ellie",
    "assistant": "Ellie",
    "ellie": "Ellie",
}


@dataclass(frozen=True)
class PublicDaicExport:
    source_root: Path
    row_key_by_participant_id: dict[int, str]
    participant_id_by_row_key: dict[str, int]
    split_counts: dict[str, int]


def split_train_validation_rows(
    *,
    train_rows: list[SessionExampleRow],
    val_rows: list[SessionExampleRow],
    inner_val_fraction: float,
    seed: int,
) -> tuple[list[SessionExampleRow], list[SessionExampleRow], dict[str, object]]:
    if val_rows:
        return (
            list(train_rows),
            list(val_rows),
            {
                "split_rule": "use manifest-provided validation rows",
                "train_size": len(train_rows),
                "validation_size": len(val_rows),
            },
        )
    if len(train_rows) <= 1:
        raise ValueError("This baseline requires validation rows or at least two train rows for an inner split")
    if not 0 < inner_val_fraction < 1:
        raise ValueError("inner_val_fraction must be between 0 and 1")

    grouped: dict[str, list[SessionExampleRow]] = {}
    for row in train_rows:
        grouped.setdefault(row.run_id, []).append(row)
    shuffled = sorted(grouped)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    validation_count = min(len(shuffled) - 1, max(1, round(len(shuffled) * inner_val_fraction)))
    validation_ids = set(shuffled[:validation_count])

    fit_rows: list[SessionExampleRow] = []
    validation_rows: list[SessionExampleRow] = []
    for run_id in sorted(grouped):
        target = validation_rows if run_id in validation_ids else fit_rows
        target.extend(sorted(grouped[run_id], key=lambda row: (row.trend_name, row.run_id, row.session_index)))
    return (
        fit_rows,
        validation_rows,
        {
            "split_rule": f"inner train/validation split because manifest validation is empty (seed={seed})",
            "inner_val_fraction": inner_val_fraction,
            "train_size": len(fit_rows),
            "validation_size": len(validation_rows),
            "validation_run_ids": sorted(validation_ids),
        },
    )


def write_public_daic_source(
    *,
    train_rows: list[SessionExampleRow],
    val_rows: list[SessionExampleRow],
    test_rows: list[SessionExampleRow],
    source_root: Path,
    blank_test_item_labels: bool = True,
) -> PublicDaicExport:
    all_rows = list(train_rows) + list(val_rows) + list(test_rows)
    if not all_rows:
        raise ValueError("Cannot export an empty baseline dataset")
    row_keys = [make_row_key(row) for row in all_rows]
    if len(row_keys) != len(set(row_keys)):
        raise ValueError("Baseline export received duplicate row keys")

    resolved_root = source_root.expanduser().resolve()
    for split_name in ("train", "dev", "test"):
        (resolved_root / split_name).mkdir(parents=True, exist_ok=True)

    participant_id_by_row_key = _assign_participant_ids(all_rows)
    row_key_by_participant_id = {value: key for key, value in participant_id_by_row_key.items()}
    split_rows = {
        "train": list(train_rows),
        "dev": list(val_rows),
        "test": list(test_rows),
    }
    split_counts = {split_name: len(rows) for split_name, rows in split_rows.items()}

    labels_path = resolved_root / "labels.csv"
    with labels_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_LABEL_HEADERS))
        writer.writeheader()
        for split_name, rows in split_rows.items():
            for row in rows:
                participant_id = participant_id_by_row_key[make_row_key(row)]
                transcript = _render_author_transcript(row.transcript_full)
                (resolved_root / split_name / f"{participant_id}.txt").write_text(transcript, encoding="utf-8")
                writer.writerow(
                    _label_row(
                        row,
                        participant_id=participant_id,
                        split=split_name,
                        include_item_labels=not (split_name == "test" and blank_test_item_labels),
                    )
                )

    metadata = {
        "source_root": str(resolved_root),
        "split_counts": split_counts,
        "participant_id_by_row_key": participant_id_by_row_key,
        "row_key_by_participant_id": {str(key): value for key, value in row_key_by_participant_id.items()},
    }
    (resolved_root / "emotrack_export_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return PublicDaicExport(
        source_root=resolved_root,
        row_key_by_participant_id=row_key_by_participant_id,
        participant_id_by_row_key=participant_id_by_row_key,
        split_counts=split_counts,
    )


def predicted_item_scores_from_sequence(values: list[float]) -> dict[str, float]:
    if len(values) != len(PHQ8_ITEM_ORDER):
        raise ValueError(f"Expected {len(PHQ8_ITEM_ORDER)} predicted PHQ-8 item scores, got {len(values)}")
    return {item_name: float(value) for item_name, value in zip(PHQ8_ITEM_ORDER, values, strict=True)}


def _assign_participant_ids(rows: list[SessionExampleRow]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    used_ids: set[int] = set()
    next_synthetic_id = 100000
    for row in rows:
        row_key = make_row_key(row)
        candidates = [row.subject_id, row.run_id]
        numeric_id = None
        if row.dataset_name == "daic_woz":
            numeric_id = _numeric_daic_subject_id(candidates)
        if numeric_id is None:
            while next_synthetic_id in used_ids:
                next_synthetic_id += 1
            numeric_id = next_synthetic_id
            next_synthetic_id += 1
        if numeric_id in used_ids:
            while next_synthetic_id in used_ids:
                next_synthetic_id += 1
            numeric_id = next_synthetic_id
            next_synthetic_id += 1
        used_ids.add(numeric_id)
        mapping[row_key] = numeric_id
    return mapping


def _numeric_daic_subject_id(candidates: list[str]) -> int | None:
    for raw in candidates:
        value = str(raw or "").strip()
        if value.isdigit():
            return int(value)
    return None


def _label_row(
    row: SessionExampleRow,
    *,
    participant_id: int,
    split: str,
    include_item_labels: bool,
) -> dict[str, object]:
    if row.target_total is None:
        raise ValueError(f"Missing target_total for baseline row {make_row_key(row)}")
    total = int(round(float(row.target_total)))
    payload: dict[str, object] = {
        "participant_id": participant_id,
        "split": split,
        "gender": 0,
        "phq_binary": int(total >= 10),
        "phq_score": total,
    }
    for symptom_name, column_name in _PUBLIC_LABEL_COLUMNS:
        if include_item_labels and symptom_name in row.target_item_scores:
            payload[column_name] = int(round(float(row.target_item_scores[symptom_name])))
        else:
            payload[column_name] = ""
    return payload


def _render_author_transcript(transcript: str) -> str:
    lines: list[str] = []
    for raw_line in str(transcript or "").splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        raw_speaker, raw_text = line.split(":", 1)
        speaker = _AUTHOR_SPEAKER_BY_CANONICAL.get(raw_speaker.strip().casefold())
        text = raw_text.strip()
        if not speaker or not text:
            continue
        lines.append(f"{speaker}: {text}")
    if not lines:
        raise ValueError("Baseline transcript export produced no author-readable speaker lines")
    return "\n".join(lines) + "\n"
