"""NHANES-conditioned sampler for PHQ-9 item vectors."""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from EmoTrack.data.psyche.files import require_nhanes_dpq_xpt
from EmoTrack.latent_variable.behavior_map import PHQ8_ITEMS

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_NHANES_DPQ_XPT_PATH = REPO_ROOT / "data" / "DPQ_L.xpt"

PHQ9_ITEM_CODES = (
    "DPQ010",
    "DPQ020",
    "DPQ030",
    "DPQ040",
    "DPQ050",
    "DPQ060",
    "DPQ070",
    "DPQ080",
    "DPQ090",
)
PHQ8_ITEM_CODES = PHQ9_ITEM_CODES[:8]
PHQ8_CODE_TO_SYMPTOM = dict(zip(PHQ8_ITEM_CODES, PHQ8_ITEMS))
_VALID_ITEM_SCORES = frozenset({0, 1, 2, 3})
_ZERO_EPSILON = 1e-12


def _normalize_dpq_score(value: Any) -> int | None:
    if pd.isna(value):
        return None
    numeric = float(value)
    if abs(numeric) < _ZERO_EPSILON:
        return 0
    rounded = round(numeric)
    if abs(numeric - rounded) < _ZERO_EPSILON and rounded in _VALID_ITEM_SCORES:
        return int(rounded)
    if numeric in _VALID_ITEM_SCORES:
        return int(numeric)
    return None


def _stable_seed(base_seed: int, *parts: object) -> int:
    digest = hashlib.sha256()
    digest.update(str(int(base_seed)).encode("utf-8"))
    for part in parts:
        digest.update(b"\0")
        digest.update(str(part).encode("utf-8"))
    return int.from_bytes(digest.digest()[:8], byteorder="big", signed=False)


@dataclass(frozen=True, slots=True)
class NhanesPhqConditionalSampler:
    buckets: dict[int, tuple[tuple[int, ...], ...]]

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "NhanesPhqConditionalSampler":
        missing = set(PHQ9_ITEM_CODES) - set(df.columns)
        if missing:
            joined = ", ".join(sorted(missing))
            raise ValueError(f"Dataframe missing NHANES PHQ-9 columns: {joined}")

        normalized_cols: dict[str, pd.Series] = {}
        for code in PHQ9_ITEM_CODES:
            normalized_cols[code] = df[code].map(_normalize_dpq_score)
        clean = pd.DataFrame(normalized_cols)
        clean = clean[clean.notna().all(axis=1)].astype(int)
        if clean.empty:
            raise ValueError("No complete NHANES PHQ-9 item rows were found.")

        clean["phq9_total"] = clean[list(PHQ9_ITEM_CODES)].sum(axis=1)
        buckets: dict[int, tuple[tuple[int, ...], ...]] = {}
        for total, group in clean.groupby("phq9_total", sort=True):
            vectors = [
                tuple(int(row[code]) for code in PHQ9_ITEM_CODES)
                for _, row in group.iterrows()
            ]
            buckets[int(total)] = tuple(vectors)

        return cls(buckets=buckets)

    @classmethod
    def from_xpt(cls, xpt_path: Path) -> "NhanesPhqConditionalSampler":
        path = require_nhanes_dpq_xpt(Path(xpt_path))
        df = pd.read_sas(path, format="xport")
        return cls.from_dataframe(df)

    @property
    def available_totals(self) -> tuple[int, ...]:
        return tuple(sorted(self.buckets))

    def sample_phq9_vector(
        self,
        phq9_total: int,
        *,
        participant_id: str = "",
        session_index: int = 0,
        sample_seed: int = 0,
    ) -> tuple[int, ...]:
        if phq9_total < 0 or phq9_total > 27:
            raise ValueError(f"PHQ-9 total must be in [0, 27], got {phq9_total}")
        if phq9_total == 0:
            return (0,) * len(PHQ9_ITEM_CODES)
        if phq9_total == 27:
            return (3,) * len(PHQ9_ITEM_CODES)
        if phq9_total not in self.buckets:
            raise ValueError(
                f"No NHANES PHQ-9 bucket available for total {phq9_total}; "
                "the current sampler only supports exact observed totals plus deterministic 0/27."
            )

        rng = random.Random(
            _stable_seed(sample_seed, "nhanes-phq9", participant_id, session_index, phq9_total)
        )
        bucket = self.buckets[phq9_total]
        return tuple(bucket[rng.randrange(len(bucket))])

    def sample_phq8_items(
        self,
        phq9_total: int,
        *,
        participant_id: str = "",
        session_index: int = 0,
        sample_seed: int = 0,
    ) -> dict[str, int]:
        vector = self.sample_phq9_vector(
            phq9_total,
            participant_id=participant_id,
            session_index=session_index,
            sample_seed=sample_seed,
        )
        return {
            PHQ8_CODE_TO_SYMPTOM[code]: int(vector[idx])
            for idx, code in enumerate(PHQ8_ITEM_CODES)
        }


@lru_cache(maxsize=4)
def load_default_nhanes_sampler(
    xpt_path: Path = DEFAULT_NHANES_DPQ_XPT_PATH,
) -> NhanesPhqConditionalSampler:
    return NhanesPhqConditionalSampler.from_xpt(Path(xpt_path))


def annotate_sampled_phq8_totals(
    wide_df: pd.DataFrame,
    *,
    sampler: NhanesPhqConditionalSampler | None = None,
    sample_seed: int = 0,
    num_sessions: int = 5,
) -> pd.DataFrame:
    resolved_sampler = sampler if sampler is not None else load_default_nhanes_sampler()
    annotated = wide_df.copy()

    for row_idx, row in annotated.iterrows():
        participant_id = str(row.get("participant_id", row_idx))
        for session_index in range(1, num_sessions + 1):
            phq9_val = row.get(f"session_{session_index}_phq9")
            if phq9_val is None or pd.isna(phq9_val):
                annotated.at[row_idx, f"session_{session_index}_phq8"] = None
                continue
            item_scores = resolved_sampler.sample_phq8_items(
                int(phq9_val),
                participant_id=participant_id,
                session_index=session_index,
                sample_seed=sample_seed,
            )
            annotated.at[row_idx, f"session_{session_index}_phq8"] = int(sum(item_scores.values()))
    return annotated
