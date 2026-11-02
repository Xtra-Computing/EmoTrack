"""E-DAIC autonomous-participant canonical bundle adapter."""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path

from EmoTrack.utils.paths import REPO_ROOT

DEFAULT_SOURCE_ROOT = REPO_ROOT / "data" / "edaic"
DEFAULT_RESULT_ROOT = REPO_ROOT / "result" / "edaic"
DETAILED_LABEL_PATH = Path("labels/detailed_lables.csv")
AUTONOMOUS_ID_MIN = 600
AUTONOMOUS_ID_MAX = 718
EXPECTED_AUTONOMOUS_COUNT = 86

_LABEL_ITEM_COLUMNS = (
    ("PHQ8_1_NoInterest", "anhedonia"),
    ("PHQ8_2_Depressed", "depressed_mood"),
    ("PHQ8_3_Sleep", "sleep"),
    ("PHQ8_4_Tired", "fatigue"),
    ("PHQ8_5_Appetite", "appetite"),
    ("PHQ8_6_Failure", "self_worth"),
    ("PHQ8_7_Concentration", "concentration"),
    ("PHQ8_8_Psychomotor", "psychomotor"),
)


@dataclass(frozen=True)
class EdaicLabel:
    participant_id: int
    phq_total: float
    symptom_scores: dict[str, float]


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").replace("\ufeff", " ")).strip()


def load_autonomous_labels(
    path: Path,
) -> list[EdaicLabel]:
    """Load the newly added, fully autonomous E-DAIC participant labels."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Missing {resolved}. Apply for E-DAIC at https://dcapswoz.ict.usc.edu/ "
            "and place your authorized files as described in README."
        )
    with resolved.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "Participant",
        "Depression_severity",
        *(column for column, _ in _LABEL_ITEM_COLUMNS),
    }
    missing_columns = required - set(rows[0] if rows else ())
    if missing_columns:
        raise ValueError(
            f"E-DAIC labels are missing columns: {sorted(missing_columns)}"
        )

    labels: list[EdaicLabel] = []
    seen: set[int] = set()
    for row in rows:
        participant_id = int(row["Participant"])
        if not AUTONOMOUS_ID_MIN <= participant_id <= AUTONOMOUS_ID_MAX:
            continue
        if participant_id in seen:
            raise ValueError(f"Duplicate E-DAIC participant label: {participant_id}")
        seen.add(participant_id)
        symptom_scores = {
            symptom: float(row[column]) for column, symptom in _LABEL_ITEM_COLUMNS
        }
        if any(score not in {0.0, 1.0, 2.0, 3.0} for score in symptom_scores.values()):
            raise ValueError(
                f"Invalid PHQ-8 item score for E-DAIC participant {participant_id}"
            )
        phq_total = float(row["Depression_severity"])
        if phq_total != sum(symptom_scores.values()):
            raise ValueError(
                f"E-DAIC PHQ total/item mismatch for participant {participant_id}: "
                f"{phq_total} != {sum(symptom_scores.values())}"
            )
        labels.append(
            EdaicLabel(
                participant_id=participant_id,
                phq_total=phq_total,
                symptom_scores=symptom_scores,
            )
        )

    labels.sort(key=lambda item: item.participant_id)
    if len(labels) != EXPECTED_AUTONOMOUS_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_AUTONOMOUS_COUNT} fully autonomous E-DAIC participants, "
            f"found {len(labels)} in {resolved}"
        )
    return labels


def parse_transcript_csv(text: str, *, participant_id: int) -> list[str]:
    reader = csv.DictReader(io.StringIO(text))
    if "Text" not in (reader.fieldnames or ()):
        raise ValueError(
            f"E-DAIC transcript {participant_id} is missing the Text column"
        )
    turns = [normalize_whitespace(row.get("Text", "")) for row in reader]
    turns = [text for text in turns if text]
    if not turns:
        raise ValueError(f"No transcript turns for E-DAIC participant {participant_id}")
    return turns


def build_edaic_result_dataset(
    *,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    output_root: Path = DEFAULT_RESULT_ROOT,
    split_file: Path | None = None,
) -> dict[str, object]:
    """Materialize the autonomous E-DAIC cohort as canonical result bundles.

    All 86 participants are deliberately assigned to one external-test-only split,
    regardless of their AVEC source split. This prevents accidental E-DAIC training
    or hyperparameter selection in the shared evaluation stack.
    """
    source = Path(source_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    labels = load_autonomous_labels(source / DETAILED_LABEL_PATH)

    transcript_root = source / "transcripts"
    missing = [
        label.participant_id
        for label in labels
        if not (transcript_root / f"{label.participant_id}_Transcript.csv").is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} E-DAIC transcripts under {transcript_root}; "
            f"first IDs: {missing[:10]}"
        )

    runs: list[dict[str, object]] = []
    for label in labels:
        transcript_path = transcript_root / f"{label.participant_id}_Transcript.csv"
        turns = parse_transcript_csv(
            transcript_path.read_text(encoding="utf-8-sig"),
            participant_id=label.participant_id,
        )
        common_dir = output / str(label.participant_id) / "common"
        common_dir.mkdir(parents=True, exist_ok=True)
        transcript_rows = [
            json.dumps(
                {
                    "session_id": 1,
                    "turn_id": index,
                    "speaker": "Client",
                    "text": turn,
                },
                ensure_ascii=False,
            )
            for index, turn in enumerate(turns, start=1)
        ]
        (common_dir / "transcript.jsonl").write_text(
            "\n".join(transcript_rows) + "\n",
            encoding="utf-8",
        )
        ground_truth = {
            "sessions": [
                {
                    "session_id": 1,
                    "ground_truth_score": {
                        "overall_score": label.phq_total,
                        "symptom_scores": label.symptom_scores,
                    },
                }
            ]
        }
        (common_dir / "ground_truth.json").write_text(
            json.dumps(ground_truth, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        runs.append(
            {
                "trend_name": "edaic_autonomous",
                "profile_id": str(label.participant_id),
                "run_id": str(label.participant_id),
                "split": "test",
            }
        )

    manifest: dict[str, object] = {
        "dataset_name": "edaic",
        "split_rule": (
            "all newly added fully autonomous participants are external-test-only; "
            "E-DAIC is prohibited for training and hyperparameter selection"
        ),
        "counts": {"train": 0, "dev": 0, "test": len(runs), "total": len(runs)},
        "expected_autonomous_count": EXPECTED_AUTONOMOUS_COUNT,
        "participant_only_transcripts": True,
        "runs": runs,
    }
    split = (
        Path(split_file)
        if split_file is not None
        else output / "train_val_test_split.json"
    )
    split.parent.mkdir(parents=True, exist_ok=True)
    temporary = split.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(split)
    return manifest
