from __future__ import annotations

import argparse
import csv
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from EmoTrack.utils.paths import REPO_ROOT

DAIC_WOZ_BASE_URL = "https://dcapswoz.ict.usc.edu/wwwdaicwoz"
DEFAULT_SOURCE_ROOT = REPO_ROOT / "data" / "daic_woz"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "out" / "daic_woz_prepared"
DEFAULT_RESULT_ROOT = REPO_ROOT / "result" / "daic_woz"
DEFAULT_RESULT_SPLIT_FILE = DEFAULT_RESULT_ROOT / "train_val_test_split.json"
TRAIN_LABEL_FILE = "train_split_Depression_AVEC2017.csv"
DEV_LABEL_FILE = "dev_split_Depression_AVEC2017.csv"
TEST_SPLIT_FILE = "test_split_Depression_AVEC2017.csv"
FULL_TEST_LABEL_FILE = "full_test_split.csv"
SPLIT_ORDER = ("train", "dev", "test")
PHQ8_SYMPTOM_COLUMNS = (
    ("PHQ8_NoInterest", "phq8_no_interest"),
    ("PHQ8_Depressed", "phq8_depressed"),
    ("PHQ8_Sleep", "phq8_sleep"),
    ("PHQ8_Tired", "phq8_tired"),
    ("PHQ8_Appetite", "phq8_appetite"),
    ("PHQ8_Failure", "phq8_failure"),
    ("PHQ8_Concentrating", "phq8_concentrating"),
    ("PHQ8_Moving", "phq8_moving"),
)
ELLIE_AUTOGEN_RE = re.compile(r"^(?P<identifier>[A-Za-z][A-Za-z0-9_.-]*\d+)\s+\((?P<utterance>.+)\)$")
RESULT_ITEM_COLUMN_MAP = {
    "phq8_no_interest": "anhedonia",
    "phq8_depressed": "depressed_mood",
    "phq8_sleep": "sleep",
    "phq8_tired": "fatigue",
    "phq8_appetite": "appetite",
    "phq8_failure": "self_worth",
    "phq8_concentrating": "concentration",
    "phq8_moving": "psychomotor",
}


@dataclass(frozen=True)
class SessionLabel:
    participant_id: int
    split: str
    gender: int | None
    phq_binary: int | None
    phq_score: int | None
    symptom_scores: dict[str, int | None]
    split_source: str
    label_source: str


@dataclass(frozen=True)
class TranscriptUtterance:
    participant_id: int
    split: str
    utterance_index: int
    start_time: float
    stop_time: float
    duration_seconds: float
    speaker: str
    text: str
    text_raw: str
    is_scrubbed: bool


@dataclass(frozen=True)
class SessionRecord:
    participant_id: int
    split: str
    gender: int | None
    phq_binary: int | None
    phq_score: int | None
    symptom_scores: dict[str, int | None]
    split_source: str
    label_source: str
    transcript_source_url: str
    transcript_member: str
    num_utterances: int
    num_participant_utterances: int
    num_ellie_utterances: int
    num_scrubbed_utterances: int
    transcript: str


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\ufeff", " ")).strip()


def normalize_speaker(speaker: str) -> str:
    cleaned = normalize_whitespace(speaker)
    lowered = cleaned.casefold()
    if lowered == "participant":
        return "Participant"
    if lowered == "ellie":
        return "Ellie"
    return cleaned


def clean_transcript_text(text: str, speaker: str) -> str:
    cleaned = normalize_whitespace(text)
    if speaker == "Ellie":
        match = ELLIE_AUTOGEN_RE.match(cleaned)
        if match is not None:
            cleaned = normalize_whitespace(match.group("utterance"))
    return cleaned


def parse_transcript_tsv(tsv_text: str, participant_id: int, split: str) -> list[TranscriptUtterance]:
    reader = csv.DictReader(io.StringIO(tsv_text), delimiter="\t")
    fieldnames = {name.strip() for name in (reader.fieldnames or [])}
    required = {"start_time", "stop_time", "speaker", "value"}
    if not required.issubset(fieldnames):
        raise ValueError(f"Transcript TSV is missing required columns: {sorted(required - fieldnames)}")

    utterances: list[TranscriptUtterance] = []
    utterance_index = 0
    for row in reader:
        raw_text = normalize_whitespace(row.get("value", ""))
        speaker = normalize_speaker(row.get("speaker", ""))
        text = clean_transcript_text(raw_text, speaker)
        if not text:
            continue

        start_time = float(row["start_time"])
        stop_time = float(row["stop_time"])
        utterances.append(
            TranscriptUtterance(
                participant_id=participant_id,
                split=split,
                utterance_index=utterance_index,
                start_time=start_time,
                stop_time=stop_time,
                duration_seconds=round(stop_time - start_time, 3),
                speaker=speaker,
                text=text,
                text_raw=raw_text,
                is_scrubbed="scrubbed_entry" in text.casefold(),
            )
        )
        utterance_index += 1

    if not utterances:
        raise ValueError(f"No utterances were parsed for participant {participant_id}.")
    return utterances


def render_transcript(utterances: list[TranscriptUtterance]) -> str:
    return "\n".join(f"{utterance.speaker}: {utterance.text}" for utterance in utterances)


def build_session_record(label: SessionLabel, utterances: list[TranscriptUtterance]) -> SessionRecord:
    participant_utterances = [row for row in utterances if row.speaker == "Participant"]
    ellie_utterances = [row for row in utterances if row.speaker == "Ellie"]
    return SessionRecord(
        participant_id=label.participant_id,
        split=label.split,
        gender=label.gender,
        phq_binary=label.phq_binary,
        phq_score=label.phq_score,
        symptom_scores=dict(label.symptom_scores),
        split_source=label.split_source,
        label_source=label.label_source,
        transcript_source_url=f"{DAIC_WOZ_BASE_URL}/{label.participant_id}_P.zip",
        transcript_member=f"{label.participant_id}_TRANSCRIPT.csv",
        num_utterances=len(utterances),
        num_participant_utterances=len(participant_utterances),
        num_ellie_utterances=len(ellie_utterances),
        num_scrubbed_utterances=sum(1 for row in utterances if row.is_scrubbed),
        transcript=render_transcript(utterances),
    )


def parse_optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    cleaned = value.strip()
    if cleaned == "":
        return None
    return int(cleaned)


def build_empty_symptom_scores() -> dict[str, int | None]:
    return {field_name: None for _, field_name in PHQ8_SYMPTOM_COLUMNS}


def parse_symptom_scores(row: dict[str, str | None]) -> dict[str, int | None]:
    scores = build_empty_symptom_scores()
    for source_name, field_name in PHQ8_SYMPTOM_COLUMNS:
        if source_name not in row:
            raise ValueError(f"Missing symptom-wise label column {source_name}.")
        scores[field_name] = parse_optional_int(row.get(source_name))
    return scores


def load_split_labels(label_dir: Path) -> dict[str, list[SessionLabel]]:
    train_labels = _load_train_or_dev_labels(label_dir / TRAIN_LABEL_FILE, split="train", score_key="PHQ8_Score", binary_key="PHQ8_Binary")
    dev_labels = _load_train_or_dev_labels(label_dir / DEV_LABEL_FILE, split="dev", score_key="PHQ8_Score", binary_key="PHQ8_Binary")
    test_labels = _load_test_labels(label_dir / TEST_SPLIT_FILE, label_dir / FULL_TEST_LABEL_FILE)
    return {
        "train": train_labels,
        "dev": dev_labels,
        "test": test_labels,
    }


def _load_train_or_dev_labels(
    path: Path,
    *,
    split: str,
    score_key: str,
    binary_key: str,
) -> list[SessionLabel]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        labels = [
            SessionLabel(
                participant_id=int(row["Participant_ID"]),
                split=split,
                gender=parse_optional_int(row.get("Gender")),
                phq_binary=parse_optional_int(row.get(binary_key)),
                phq_score=parse_optional_int(row.get(score_key)),
                symptom_scores=parse_symptom_scores(row),
                split_source=path.name,
                label_source=path.name,
            )
            for row in reader
        ]
    if not labels:
        raise ValueError(f"No labels were parsed from {path}.")
    return labels


def _load_test_labels(split_path: Path, full_label_path: Path) -> list[SessionLabel]:
    with split_path.open("r", encoding="utf-8", newline="") as handle:
        split_reader = csv.DictReader(handle)
        split_rows = {int(row["participant_ID"]): parse_optional_int(row.get("Gender")) for row in split_reader}
    if not split_rows:
        raise ValueError(f"No test split rows were parsed from {split_path}.")

    with full_label_path.open("r", encoding="utf-8", newline="") as handle:
        label_reader = csv.DictReader(handle)
        label_rows = {
            int(row["Participant_ID"]): (
                parse_optional_int(row.get("Gender")),
                parse_optional_int(row.get("PHQ_Binary")),
                parse_optional_int(row.get("PHQ_Score")),
            )
            for row in label_reader
        }
    if not label_rows:
        raise ValueError(f"No test labels were parsed from {full_label_path}.")

    if set(split_rows) != set(label_rows):
        missing_from_scores = sorted(set(split_rows) - set(label_rows))
        missing_from_split = sorted(set(label_rows) - set(split_rows))
        raise ValueError(
            "Test split membership and test labels do not match. "
            f"Missing scores for IDs {missing_from_scores}; missing split entries for IDs {missing_from_split}."
        )

    labels: list[SessionLabel] = []
    for participant_id in sorted(split_rows):
        split_gender = split_rows[participant_id]
        label_gender, phq_binary, phq_score = label_rows[participant_id]
        if split_gender is not None and label_gender is not None and split_gender != label_gender:
            raise ValueError(
                "Gender mismatch between test split and full test labels for "
                f"participant {participant_id}: {split_gender} vs {label_gender}."
            )
        labels.append(
            SessionLabel(
                participant_id=participant_id,
                split="test",
                gender=label_gender if label_gender is not None else split_gender,
                phq_binary=phq_binary,
                phq_score=phq_score,
                symptom_scores=build_empty_symptom_scores(),
                split_source=split_path.name,
                label_source=full_label_path.name,
            )
        )
    return labels


def prepare_daic_woz_dataset(
    *,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    """Convert locally supplied official files; never request interview data over the network."""
    source = source_root.expanduser().resolve()
    output = output_root.expanduser().resolve()
    label_dir = source / "labels"
    required = [
        label_dir / name
        for name in (
            TRAIN_LABEL_FILE,
            DEV_LABEL_FILE,
            TEST_SPLIT_FILE,
            FULL_TEST_LABEL_FILE,
        )
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing DAIC-WOZ files: {missing}. Apply at https://dcapswoz.ict.usc.edu/ "
            f"and place your authorized label CSVs in {label_dir}; see README."
        )
    split_labels = load_split_labels(label_dir)
    transcripts = source / "transcripts"
    missing = [
        str(transcripts / f"{label.participant_id}_TRANSCRIPT.csv")
        for labels in split_labels.values()
        for label in labels
        if not (transcripts / f"{label.participant_id}_TRANSCRIPT.csv").is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing local DAIC-WOZ transcripts; first: {missing[:5]}"
        )
    if source == output:
        raise ValueError("Prepared output must differ from the raw DAIC-WOZ directory")
    rows = []
    for split, labels in split_labels.items():
        records = []
        for label in labels:
            text = (transcripts / f"{label.participant_id}_TRANSCRIPT.csv").read_text(
                encoding="utf-8-sig"
            )
            records.append(
                build_session_record(
                    label, parse_transcript_tsv(text, label.participant_id, split)
                )
            )
        rows.extend(write_split_transcripts(output, records))
    rows.sort(key=lambda row: int(row["participant_id"]))
    _write_csv(output / "labels.csv", rows, LABEL_FIELD_ORDER)
    return {"participants": len(rows), "labels_csv": str(output / "labels.csv")}


def write_split_transcripts(output_root: Path, session_records: list[SessionRecord]) -> list[dict[str, Any]]:
    label_rows: list[dict[str, Any]] = []
    for session in session_records:
        split_dir = output_root / session.split
        split_dir.mkdir(parents=True, exist_ok=True)
        transcript_name = f"{session.participant_id}.txt"
        transcript_path = split_dir / transcript_name
        transcript_path.write_text(session.transcript + "\n", encoding="utf-8")
        label_rows.append(
            {
                "participant_id": session.participant_id,
                "split": session.split,
                "gender": session.gender,
                "phq_binary": session.phq_binary,
                "phq_score": session.phq_score,
                **session.symptom_scores,
            }
        )
    return label_rows


def _write_csv(path: Path, rows: list[dict[str, Any]], field_order: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(field_order))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in field_order})


LABEL_FIELD_ORDER = (
    "participant_id",
    "split",
    "gender",
    "phq_binary",
    "phq_score",
    "phq8_no_interest",
    "phq8_depressed",
    "phq8_sleep",
    "phq8_tired",
    "phq8_appetite",
    "phq8_failure",
    "phq8_concentrating",
    "phq8_moving",
)


def build_daic_woz_result_dataset(
    *,
    source_root: Path,
    output_root: Path = DEFAULT_RESULT_ROOT,
    split_output: Path | None = None,
) -> dict[str, Any]:
    resolved_source = source_root.expanduser().resolve()
    resolved_output = output_root.expanduser().resolve()
    resolved_split = (
        split_output.expanduser().resolve()
        if split_output is not None
        else (resolved_output / "train_val_test_split.json").resolve()
    )
    labels_path = resolved_source / "labels.csv"
    if not labels_path.is_file():
        with TemporaryDirectory(prefix="emotrack-daic-") as temporary:
            prepared = Path(temporary)
            prepare_daic_woz_dataset(source_root=resolved_source, output_root=prepared)
            return build_daic_woz_result_dataset(
                source_root=prepared, output_root=resolved_output, split_output=resolved_split,
            )

    rows = _load_result_label_rows(labels_path)
    split_counts = {split: 0 for split in SPLIT_ORDER}
    manifest_runs: list[dict[str, str]] = []

    for row in rows:
        participant_id = str(row["participant_id"]).strip()
        split = str(row["split"]).strip().lower()
        if split not in split_counts:
            raise ValueError(f"Unsupported DAIC-WOZ split {split!r} for participant {participant_id}")
        transcript_source = resolved_source / split / f"{participant_id}.txt"
        if not transcript_source.is_file():
            raise FileNotFoundError(
                f"Missing DAIC-WOZ transcript for participant {participant_id}: {transcript_source}"
            )

        run_dir = resolved_output / participant_id
        common_dir = run_dir / "common"
        common_dir.mkdir(parents=True, exist_ok=True)
        _write_daic_transcript_bundle(
            transcript_source=transcript_source,
            transcript_output=common_dir / "transcript.jsonl",
        )
        _write_daic_ground_truth_bundle(
            label_row=row,
            ground_truth_output=common_dir / "ground_truth.json",
        )

        split_counts[split] += 1
        manifest_runs.append(
            {
                "trend_name": "daic_woz",
                "profile_id": participant_id,
                "run_id": participant_id,
                "split": split,
            }
        )

    manifest = {
        "dataset_name": "daic_woz",
        "split_rule": "official DAIC-WOZ participant-level train/dev/test split",
        "counts": {
            "train": split_counts["train"],
            "dev": split_counts["dev"],
            "test": split_counts["test"],
            "total": sum(split_counts.values()),
        },
        "runs": sorted(manifest_runs, key=lambda item: int(item["run_id"])),
    }
    resolved_split.parent.mkdir(parents=True, exist_ok=True)
    temporary_split = resolved_split.with_suffix(".json.tmp")
    temporary_split.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_split.replace(resolved_split)
    return manifest


def _load_result_label_rows(labels_path: Path) -> list[dict[str, str]]:
    with labels_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_daic_transcript_bundle(
    *,
    transcript_source: Path,
    transcript_output: Path,
) -> None:
    rows: list[str] = []
    turn_id = 0
    for raw_line in transcript_source.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if ":" in line:
            raw_speaker, text = line.split(":", 1)
        else:
            raw_speaker, text = "Participant", line
        text = text.strip()
        if not text:
            continue
        turn_id += 1
        speaker = _normalize_result_speaker(raw_speaker)
        rows.append(
            json.dumps(
                {
                    "session_id": 1,
                    "turn_id": turn_id,
                    "speaker": speaker,
                    "text": text,
                },
                ensure_ascii=False,
            )
        )

    if not rows:
        raise ValueError(f"DAIC-WOZ transcript is empty after normalization: {transcript_source}")
    transcript_output.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_daic_ground_truth_bundle(
    *,
    label_row: dict[str, str],
    ground_truth_output: Path,
) -> None:
    overall_score = _parse_optional_int(label_row.get("phq_score"))
    symptom_scores = {
        symptom_name: float(value)
        for column_name, symptom_name in RESULT_ITEM_COLUMN_MAP.items()
        if (value := _parse_optional_int(label_row.get(column_name))) is not None
    }
    payload = {
        "sessions": [
            {
                "session_id": 1,
                "ground_truth_score": {
                    "overall_score": float(overall_score) if overall_score is not None else None,
                    "symptom_scores": symptom_scores,
                },
            }
        ]
    }
    ground_truth_output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _normalize_result_speaker(raw_speaker: str) -> str:
    speaker = str(raw_speaker or "").strip().lower()
    if speaker in {"participant", "patient", "client", "user"}:
        return "Client"
    if speaker in {"ellie", "counselor", "therapist", "assistant", "interviewer"}:
        return "Counselor"
    return str(raw_speaker).strip() or "Client"


def _parse_optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return int(float(text))


def build_daic_sft_dataset(
    *,
    result_root: Path = DEFAULT_RESULT_ROOT,
    split_file: Path | None = None,
    transcript_view: str = "full",
    splits: tuple[str, ...] = ("train", "dev"),
) -> list[dict[str, Any]]:
    """Build SFT-format JSONL examples from DAIC-WOZ result bundles.

    Only includes participants in the specified splits (default: train + dev).
    Uses the split manifest as authority to avoid test leakage.
    """
    resolved_root = Path(result_root).expanduser().resolve()
    resolved_split = (
        Path(split_file).expanduser().resolve()
        if split_file is not None
        else resolved_root / "train_val_test_split.json"
    )
    if not resolved_split.is_file():
        raise FileNotFoundError(f"Missing split manifest: {resolved_split}")

    manifest = json.loads(resolved_split.read_text(encoding="utf-8"))
    split_by_id = {
        str(r["run_id"]): str(r["split"])
        for r in manifest.get("runs", [])
    }
    allowed_splits = set(splits)

    import logging as _logging

    _log = _logging.getLogger(__name__)

    examples: list[dict[str, Any]] = []
    skipped: list[str] = []
    for run_id, split in sorted(split_by_id.items(), key=lambda x: int(x[0])):
        if split not in allowed_splits:
            continue

        run_dir = resolved_root / run_id
        common_dir = run_dir / "common"
        transcript_path = common_dir / "transcript.jsonl"
        gt_path = common_dir / "ground_truth.json"
        if not transcript_path.is_file() or not gt_path.is_file():
            raise FileNotFoundError(
                f"Missing result bundle for manifest-listed run {run_id}: {run_dir}"
            )

        # Render transcript
        lines: list[str] = []
        for raw_line in transcript_path.read_text(encoding="utf-8").splitlines():
            if not raw_line.strip():
                continue
            turn = json.loads(raw_line)
            speaker = turn.get("speaker", "")
            text = turn.get("text", "").strip()
            if not text:
                continue
            if transcript_view == "client-only" and speaker != "Client":
                continue
            lines.append(f"{speaker}: {text}")
        transcript_text = "\n".join(lines)
        if not transcript_text:
            raise ValueError(f"Empty transcript for run {run_id} (view={transcript_view})")

        # Load ground truth
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        session_gt = gt["sessions"][0]["ground_truth_score"]
        symptom_scores = session_gt.get("symptom_scores", {})
        required_keys = {"anhedonia", "depressed_mood", "sleep", "fatigue",
                         "appetite", "self_worth", "concentration", "psychomotor"}
        if not symptom_scores or set(symptom_scores.keys()) != required_keys:
            skipped.append(run_id)
            continue

        examples.append({
            "run_id": run_id,
            "trend_name": "daic_woz",
            "session_index": 1,
            "prior_transcripts": [],
            "prior_phq_totals": [],
            "current_transcript": transcript_text,
            "target_phq_items": symptom_scores,
        })

    if skipped:
        _log.warning(
            "Skipped %d runs without per-item symptom scores: %s",
            len(skipped), ", ".join(skipped),
        )

    return examples


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert local, authorized DAIC-WOZ transcripts and labels."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    manifest = prepare_daic_woz_dataset(
        source_root=args.source_root, output_root=args.output_root
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
