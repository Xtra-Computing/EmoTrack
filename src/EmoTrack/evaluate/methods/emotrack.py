from __future__ import annotations

import asyncio
import inspect
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import jinja2

from EmoTrack.evaluate.core.phq8 import PHQ8_ITEM_TEXT_BY_SYMPTOM
from EmoTrack.evaluate.core.scores import (
    ConversationArtifact,
    ConversationSessionArtifact,
    DetectorItemScore,
    DetectorRunScore,
    DetectorSessionScore,
    TranscriptView,
    item_error_summary,
)

from EmoTrack.evaluate.utils import render_session_transcript

_DEFAULT_TEMPLATE_PATH = Path("configs/train/emotrack_prompt.jinja2")

_PHQ8_SYMPTOMS = list(PHQ8_ITEM_TEXT_BY_SYMPTOM.keys())


def _phq8_severity_band(total: float) -> str:
    if total < 5:
        return "minimal"
    if total < 10:
        return "mild"
    if total < 15:
        return "moderate"
    if total < 20:
        return "moderately_severe"
    return "severe"


class EmoTrackEvaluator:
    """Session-level PHQ-8 evaluator using structured JSON output.

    Works in two modes:
    - Zero-shot: uses a base LLM via API to predict PHQ-8 from transcripts
    - Trained: uses a GRPO-finetuned model (loaded locally or served via API)
    """

    def __init__(
        self,
        *,
        sample_count: int = 1,
        include_history: bool = True,
        transcript_view: TranscriptView = "full",
        use_trend: bool = False,
        use_confidence: bool = False,
        prompt_template_path: str | Path | None = None,
    ) -> None:
        self._sample_count = sample_count
        self._include_history = include_history
        self._transcript_view: TranscriptView = transcript_view
        self._use_trend = use_trend
        self._use_confidence = use_confidence
        template_path = Path(prompt_template_path or _DEFAULT_TEMPLATE_PATH)
        self._template = jinja2.Template(
            template_path.read_text(encoding="utf-8"),
        )

    @property
    def method_name(self) -> str:
        return "emotrack"

    async def score_artifact(
        self,
        artifact: ConversationArtifact,
        llm_client: Any,
        *,
        scoring_model: str = "",
    ) -> DetectorRunScore:
        sessions: list[DetectorSessionScore] = []
        for idx, session in enumerate(artifact.sessions):
            prior_sessions = artifact.sessions[:idx]
            prior_phq_totals = [
                s.self_report_total for s in prior_sessions
                if s.self_report_total is not None
            ]
            session_score = await self.score_session(
                session=session,
                prior_sessions=prior_sessions if self._include_history else [],
                prior_phq_totals=prior_phq_totals if self._include_history else [],
                llm_client=llm_client,
            )
            sessions.append(session_score)

        latent_errors = [s.absolute_error_to_latent for s in sessions]
        self_report_errors = [
            s.absolute_error_to_self_report for s in sessions
            if s.absolute_error_to_self_report is not None
        ]
        return DetectorRunScore(
            method=self.method_name,
            scored_at=datetime.now(timezone.utc).isoformat(),
            scoring_model=scoring_model,
            source_run_path=str(artifact.source_dir),
            source_run_id=artifact.run_id,
            transcript_view=self._transcript_view,
            sample_count=self._sample_count,
            session_count=len(sessions),
            sessions=sessions,
            dataset_name="conversation",
            trend_name=artifact.trend_name,
            profile_id=artifact.profile_id,
            subject_id=artifact.run_id,
            split="",
            target_label="phq8_total",
            mean_absolute_error_to_target=(
                round(mean(self_report_errors), 2) if self_report_errors else None
            ),
            mean_absolute_error_to_latent=round(mean(latent_errors), 2) if latent_errors else 0.0,
            mean_absolute_error_to_self_report=(
                round(mean(self_report_errors), 2) if self_report_errors else None
            ),
        )

    async def score_session(
        self,
        *,
        session: ConversationSessionArtifact,
        prior_sessions: list[ConversationSessionArtifact],
        prior_phq_totals: list[float],
        llm_client: Any,
    ) -> DetectorSessionScore:
        current_transcript = render_session_transcript(
            session, transcript_view=self._transcript_view,
        )
        prior_transcripts = [
            render_session_transcript(s, transcript_view=self._transcript_view)
            for s in prior_sessions
        ]

        all_items: dict[str, list[float]] = {s: [] for s in _PHQ8_SYMPTOMS}
        all_totals: list[float] = []

        for _ in range(self._sample_count):
            result = await self._query_llm(
                current_transcript=current_transcript,
                prior_transcripts=prior_transcripts,
                prior_phq_totals=prior_phq_totals,
                session_index=session.session_id,
                llm_client=llm_client,
            )
            if result is not None:
                items = result.get("items", {})
                if isinstance(items, dict) and all(
                    isinstance(items.get(s), (int, float)) for s in _PHQ8_SYMPTOMS
                ):
                    item_total = sum(
                        float(max(0, min(3, items[s]))) for s in _PHQ8_SYMPTOMS
                    )
                    all_totals.append(item_total)
                    for symptom in _PHQ8_SYMPTOMS:
                        all_items[symptom].append(float(items[symptom]))

        if not all_totals:
            # Fallback: all zeros if LLM failed
            predicted_items = {s: 0.0 for s in _PHQ8_SYMPTOMS}
            predicted_total = 0.0
        else:
            predicted_items = {
                s: float(max(0, min(3, round(mean(all_items[s])))))
                if all_items[s] else 0.0
                for s in _PHQ8_SYMPTOMS
            }
            # Recompute total from clipped items — never trust model's total
            predicted_total = round(sum(predicted_items.values()), 2)

        self_report_total = session.self_report_total
        self_report_items = dict(session.self_report_item_scores)
        latent_items = dict(session.latent_item_scores)
        target_item_mae, target_per_item_errors = item_error_summary(predicted_items, self_report_items)
        latent_item_mae, latent_per_item_errors = item_error_summary(predicted_items, latent_items)
        item_scores = [
            DetectorItemScore(
                question_id=idx + 1,
                symptom_name=symptom,
                question_text=PHQ8_ITEM_TEXT_BY_SYMPTOM[symptom],
                score=predicted_items[symptom],
                score_std=0.0,
                sample_scores=[int(round(predicted_items[symptom]))],
                option_label="",
                evidence="",
            )
            for idx, symptom in enumerate(_PHQ8_SYMPTOMS)
        ]

        return DetectorSessionScore(
            session_index=session.session_id,
            transcript_view=self._transcript_view,
            sample_count=self._sample_count,
            predicted_total=predicted_total,
            predicted_item_scores=predicted_items,
            target_item_scores=self_report_items,
            latent_item_scores=latent_items,
            item_mean_absolute_error_to_target=target_item_mae,
            item_mean_absolute_error_to_latent=latent_item_mae,
            per_item_absolute_error_to_target=target_per_item_errors,
            per_item_absolute_error_to_latent=latent_per_item_errors,
            target_total=self_report_total,
            latent_total=session.latent_total,
            self_report_total=self_report_total,
            absolute_error_to_target=(
                round(abs(predicted_total - self_report_total), 2)
                if self_report_total is not None
                else None
            ),
            absolute_error_to_latent=round(abs(predicted_total - session.latent_total), 2),
            absolute_error_to_self_report=(
                round(abs(predicted_total - self_report_total), 2)
                if self_report_total is not None
                else None
            ),
            items=item_scores,
        )

    async def _query_llm(
        self,
        *,
        current_transcript: str,
        prior_transcripts: list[str],
        prior_phq_totals: list[float],
        session_index: int,
        llm_client: Any,
    ) -> dict[str, Any] | None:
        prompt = self._template.render(
            current_transcript=current_transcript,
            prior_transcripts=prior_transcripts,
            prior_phq_totals=prior_phq_totals,
            session_index=session_index,
            use_trend=self._use_trend,
            use_confidence=self._use_confidence,
            phq8_items=PHQ8_ITEM_TEXT_BY_SYMPTOM,
        )

        messages = [
            {"role": "system", "content": "You are a clinical psychology expert. Return valid JSON only."},
            {"role": "user", "content": prompt},
        ]

        try:
            if hasattr(llm_client, "chat_text"):
                result = llm_client.chat_text(
                    messages=messages,
                    temperature=0.0,
                    max_tokens=512,
                    response_format={"type": "json_object"},
                )
            elif hasattr(llm_client, "complete"):
                result = llm_client.complete(messages, temperature=0.0, max_tokens=512)
            else:
                raise TypeError("llm_client must expose chat_text(...) or complete(...).")

            if inspect.isawaitable(result):
                result = await result

            return _parse_json(str(result))
        except Exception:
            return None


    def score_session_embedding(
        self,
        *,
        session: ConversationSessionArtifact,
        predicted_items: dict[str, float],
    ) -> DetectorSessionScore:
        """Build a DetectorSessionScore from numeric predictions (no LLM/JSON).

        Used by the embedding-based SFT model path where predictions are already
        numeric floats, not parsed from JSON text.
        """
        predicted_total = round(sum(predicted_items.get(s, 0.0) for s in _PHQ8_SYMPTOMS), 2)
        self_report_total = session.self_report_total
        self_report_items = dict(session.self_report_item_scores)
        latent_items = dict(session.latent_item_scores)
        target_item_mae, target_per_item_errors = item_error_summary(predicted_items, self_report_items)
        latent_item_mae, latent_per_item_errors = item_error_summary(predicted_items, latent_items)

        item_scores = [
            DetectorItemScore(
                question_id=idx + 1,
                symptom_name=symptom,
                question_text=PHQ8_ITEM_TEXT_BY_SYMPTOM[symptom],
                score=predicted_items.get(symptom, 0.0),
                score_std=0.0,
                sample_scores=[int(round(predicted_items.get(symptom, 0.0)))],
                option_label="",
                evidence="",
            )
            for idx, symptom in enumerate(_PHQ8_SYMPTOMS)
        ]

        return DetectorSessionScore(
            session_index=session.session_id,
            transcript_view=self._transcript_view,
            sample_count=1,
            predicted_total=predicted_total,
            predicted_item_scores=predicted_items,
            target_item_scores=self_report_items,
            latent_item_scores=latent_items,
            item_mean_absolute_error_to_target=target_item_mae,
            item_mean_absolute_error_to_latent=latent_item_mae,
            per_item_absolute_error_to_target=target_per_item_errors,
            per_item_absolute_error_to_latent=latent_per_item_errors,
            target_total=self_report_total,
            latent_total=session.latent_total,
            self_report_total=self_report_total,
            absolute_error_to_target=(
                round(abs(predicted_total - self_report_total), 2)
                if self_report_total is not None
                else None
            ),
            absolute_error_to_latent=round(abs(predicted_total - session.latent_total), 2),
            absolute_error_to_self_report=(
                round(abs(predicted_total - self_report_total), 2)
                if self_report_total is not None
                else None
            ),
            items=item_scores,
        )


def _parse_json(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    try:
        obj = json.loads(stripped)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end > start:
            try:
                obj = json.loads(stripped[start:end + 1])
                return obj if isinstance(obj, dict) else None
            except json.JSONDecodeError:
                return None
    return None
