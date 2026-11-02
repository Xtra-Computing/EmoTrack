"""Generate per-participant trend YAML and initial hidden-state JSON from Psyche trajectories."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import yaml

from EmoTrack.data.psyche.nhanes_phq_sampler import (
    NhanesPhqConditionalSampler,
    load_default_nhanes_sampler,
)
from EmoTrack.data.session_latent import _band
from EmoTrack.latent_variable.behavior_map import PHQ8_ITEMS

logger = logging.getLogger(__name__)

_PHQ8_ITEM_LIST = list(PHQ8_ITEMS)


def _sample_session(
    phq9_total: int,
    *,
    participant_id: str,
    session_index: int,
    sampler: NhanesPhqConditionalSampler,
    sample_seed: int,
) -> tuple[dict[str, int], int]:
    scores = sampler.sample_phq8_items(
        phq9_total,
        participant_id=participant_id,
        session_index=session_index,
        sample_seed=sample_seed,
    )
    return scores, int(sum(scores.values()))


def generate_psyche_trends(
    wide_df: pd.DataFrame,
    output_dir: Path,
    num_sessions: int = 5,
    *,
    sampler: NhanesPhqConditionalSampler | None = None,
    sample_seed: int = 0,
) -> list[Path]:
    """Generate one YAML trend file per complete-case participant.

    Each file uses NHANES-conditioned PHQ-9 item sampling and retains the
    first 8 PHQ items as the session schedule.
    """
    resolved_sampler = sampler if sampler is not None else load_default_nhanes_sampler()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for _, row in wide_df.iterrows():
        pid = str(row["participant_id"])

        phq9_totals: list[int] = []
        for session_index in range(1, num_sessions + 1):
            phq9_val = row.get(f"session_{session_index}_phq9")
            if phq9_val is None or pd.isna(phq9_val):
                phq9_totals = []
                break
            phq9_totals.append(int(phq9_val))

        if not phq9_totals:
            continue

        item_schedules: dict[str, list[float]] = {item: [] for item in _PHQ8_ITEM_LIST}
        for session_index, phq9_total in enumerate(phq9_totals, start=1):
            scores, _ = _sample_session(
                phq9_total,
                participant_id=pid,
                session_index=session_index,
                sampler=resolved_sampler,
                sample_seed=sample_seed,
            )
            for item in _PHQ8_ITEM_LIST:
                item_schedules[item].append(float(scores[item]))

        targets: dict[str, list[float]] = {}
        for item in _PHQ8_ITEM_LIST:
            path = f"latent_state.phq8.{item}.score"
            targets[path] = item_schedules[item]

        plan = {
            "version": "1.0",
            "description": f"Psyche participant {pid}: {num_sessions}-session PHQ schedule from observed data",
            "targets": targets,
        }

        out_path = output_dir / f"{pid}.yaml"
        out_path.write_text(yaml.dump(plan, sort_keys=False, allow_unicode=True), encoding="utf-8")
        paths.append(out_path)

    logger.info("Generated %d trend files in %s", len(paths), output_dir)
    return paths


def generate_hidden_states(
    wide_df: pd.DataFrame,
    output_dir: Path,
    *,
    sampler: NhanesPhqConditionalSampler | None = None,
    sample_seed: int = 0,
) -> list[Path]:
    """Generate per-participant initial hidden state JSON (session-1 values)."""
    resolved_sampler = sampler if sampler is not None else load_default_nhanes_sampler()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for _, row in wide_df.iterrows():
        pid = str(row["participant_id"])
        phq9_val = row.get("session_1_phq9")
        if phq9_val is None or pd.isna(phq9_val):
            continue

        scores, phq8_total = _sample_session(
            int(phq9_val),
            participant_id=pid,
            session_index=1,
            sampler=resolved_sampler,
            sample_seed=sample_seed,
        )

        phq8_items: dict[str, dict] = {}
        for item in _PHQ8_ITEM_LIST:
            entry: dict = {"score": scores[item]}
            if item == "sleep":
                entry["direction"] = "insomnia"
            elif item == "appetite":
                entry["direction"] = "decrease"
            elif item == "psychomotor":
                entry["polarity"] = "slowed"
            phq8_items[item] = entry

        hidden_state = {
            "macro_labels": {
                "phq8_total": phq8_total,
                "phq8_band": _band(phq8_total, "phq8"),
            },
            "latent_state": {
                "phq8": phq8_items,
            },
        }

        out_path = output_dir / f"{pid}.json"
        out_path.write_text(json.dumps(hidden_state, indent=2, ensure_ascii=False), encoding="utf-8")
        paths.append(out_path)

    logger.info("Generated %d hidden state files in %s", len(paths), output_dir)
    return paths
