"""Reconstruct 5-session PHQ trajectories from quarterly Psyche parquet data."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

QUARTERLY_MONTHS = (3, 6, 9, 12)
PHQ9_MAX = 27
PHQ9_CATEGORIES = {
    (0, 4): "minimal",
    (5, 9): "mild",
    (10, 14): "moderate",
    (15, 19): "moderately_severe",
    (20, 27): "severe",
}


def phq9_category(score: int) -> str:
    for (lo, hi), label in PHQ9_CATEGORIES.items():
        if lo <= score <= hi:
            return label
    return "unknown"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_raw_data(raw_dir: Path) -> pd.DataFrame:
    """Load parquet and parse '{pid}_{quarter}' index into participant_id + month columns."""
    parquet_path = raw_dir / "data_quarterly.parquet"
    df = pd.read_parquet(parquet_path).reset_index()
    df.rename(columns={"index": "raw_id"}, inplace=True)
    parts = df["raw_id"].str.rsplit("_", n=1)
    df["participant_id"] = parts.str[0]
    df["month"] = parts.str[1].astype(int)
    # Keep only rows with PHQ observations (quarters 3, 6, 9, 12)
    df = df.dropna(subset=["phq9_score_start", "phq9_score_end"])
    return df


# ---------------------------------------------------------------------------
# Trajectory reconstruction
# ---------------------------------------------------------------------------


def _extract_participant_sessions(
    rows: pd.DataFrame,
) -> tuple[list[int], dict[int, str]]:
    """Chain quarterly overlapping PHQ windows into 5 anchor sessions.

    Returns (session_phq9_values, mismatch_log).
    """
    month_to_row: dict[int, pd.Series] = {}
    for _, row in rows.iterrows():
        month = int(row["month"])
        month_to_row[month] = row

    sessions: list[int | None] = [None] * 5  # sessions 0..4
    mismatches: dict[int, str] = {}

    # Session 0: phq9_score_start from month-3 row
    if 3 in month_to_row:
        sessions[0] = int(month_to_row[3]["phq9_score_start"])

    # Sessions 1..4 from phq9_score_end of months 3,6,9,12
    for i, month in enumerate(QUARTERLY_MONTHS):
        session_idx = i + 1
        if month in month_to_row:
            sessions[session_idx] = int(month_to_row[month]["phq9_score_end"])

    # Consistency: end of month N should == start of month N+3
    for i in range(len(QUARTERLY_MONTHS) - 1):
        month_cur = QUARTERLY_MONTHS[i]
        month_nxt = QUARTERLY_MONTHS[i + 1]
        if month_cur in month_to_row and month_nxt in month_to_row:
            end_val = int(month_to_row[month_cur]["phq9_score_end"])
            start_val = int(month_to_row[month_nxt]["phq9_score_start"])
            if end_val != start_val:
                avg = round((end_val + start_val) / 2)
                mismatches[i + 1] = (
                    f"month {month_cur} end={end_val} vs month {month_nxt} start={start_val}, "
                    f"using avg={avg}"
                )
                sessions[i + 1] = avg

    return sessions, mismatches  # type: ignore[return-value]


def reconstruct_phq_trajectories(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconstruct per-participant 5-session PHQ-9 trajectories.

    Returns (strict_complete_df, relaxed_partial_df).
    Strict = all 5 sessions present. Relaxed = at least 3 sessions.
    """
    required_cols = {"participant_id", "month", "phq9_score_start", "phq9_score_end"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    records_strict: list[dict[str, Any]] = []
    records_relaxed: list[dict[str, Any]] = []
    total_mismatches = 0

    for pid, group in df.groupby("participant_id"):
        sessions, mismatches = _extract_participant_sessions(group)
        total_mismatches += len(mismatches)

        present = [s for s in sessions if s is not None]
        if len(present) < 3:
            continue

        record: dict[str, Any] = {"participant_id": pid}
        for i, val in enumerate(sessions):
            record[f"session_{i + 1}_phq9"] = val
        record["num_observed"] = len(present)
        record["mismatches"] = mismatches if mismatches else None

        if len(present) == 5:
            records_strict.append(record)
        records_relaxed.append(record)

    strict_df = pd.DataFrame(records_strict)
    relaxed_df = pd.DataFrame(records_relaxed)
    logger.info(
        "Reconstructed trajectories: %d strict (5/5), %d relaxed (3+/5), %d mismatches total",
        len(strict_df),
        len(relaxed_df),
        total_mismatches,
    )
    return strict_df, relaxed_df


# ---------------------------------------------------------------------------
# Wide / long format
# ---------------------------------------------------------------------------


def build_wide_format(traj_df: pd.DataFrame) -> pd.DataFrame:
    """One row per participant with session columns, deltas, and change labels."""
    rows: list[dict[str, Any]] = []
    for _, row in traj_df.iterrows():
        rec: dict[str, Any] = {"participant_id": row["participant_id"]}
        phq_values = []
        for i in range(1, 6):
            val = row.get(f"session_{i}_phq9")
            rec[f"session_{i}_phq9"] = val
            phq_values.append(val)

        # Deltas between consecutive sessions
        for i in range(1, 5):
            v1 = phq_values[i - 1]
            v2 = phq_values[i]
            if v1 is not None and v2 is not None:
                rec[f"delta_{i}_{i + 1}"] = int(v2) - int(v1)
            else:
                rec[f"delta_{i}_{i + 1}"] = None

        # Overall change
        first, last = phq_values[0], phq_values[-1]
        if first is not None and last is not None:
            rec["overall_change"] = int(last) - int(first)
        else:
            rec["overall_change"] = None

        rows.append(rec)
    return pd.DataFrame(rows)


def build_long_format(traj_df: pd.DataFrame) -> pd.DataFrame:
    """One row per participant-session.

    Expects NHANES-sampled ``session_X_phq8`` columns to already be present
    on the input (added by ``annotate_sampled_phq8_totals``).
    """
    months = [0, 3, 6, 9, 12]
    rows: list[dict[str, Any]] = []
    for _, row in traj_df.iterrows():
        pid = row["participant_id"]
        for i in range(1, 6):
            val = row.get(f"session_{i}_phq9")
            phq8_val = row.get(f"session_{i}_phq8")
            if phq8_val is not None and not pd.isna(phq8_val):
                phq8_val = int(phq8_val)
            else:
                phq8_val = None
            rows.append(
                {
                    "participant_id": pid,
                    "session_index": i,
                    "month": months[i - 1],
                    "phq9_score": val,
                    "phq8_score": phq8_val,
                    "phq9_category": phq9_category(int(val)) if val is not None else None,
                    "is_observed": val is not None,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Change labels
# ---------------------------------------------------------------------------


def compute_change_labels(
    wide_df: pd.DataFrame,
    improvement_threshold: int = 5,
    worsening_threshold: int = 5,
) -> pd.DataFrame:
    """Add a change_label column based on overall PHQ-9 change."""
    df = wide_df.copy()
    labels = []
    for change in df["overall_change"]:
        if change is None or pd.isna(change):
            labels.append("unknown")
        elif change <= -improvement_threshold:
            labels.append("improved")
        elif change >= worsening_threshold:
            labels.append("worsened")
        else:
            labels.append("stable")
    df["change_label"] = labels
    return df


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def print_summary(strict_df: pd.DataFrame, relaxed_df: pd.DataFrame) -> str:
    lines = [
        f"Strict complete (5/5 sessions): {len(strict_df)} participants",
        f"Relaxed (3+/5 sessions): {len(relaxed_df)} participants",
    ]
    if not strict_df.empty:
        phq_cols = [f"session_{i}_phq9" for i in range(1, 6)]
        means = strict_df[phq_cols].mean()
        lines.append("Mean PHQ-9 per session (strict):")
        for col, val in means.items():
            lines.append(f"  {col}: {val:.1f}")
    summary = "\n".join(lines)
    logger.info("Summary:\n%s", summary)
    return summary
