"""Build scoring datasets from conversation result bundles."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from EmoTrack.evaluate.core.scores import (
    ConversationArtifact,
    ConversationSessionArtifact,
)

from EmoTrack.evaluate.utils import (
    load_conversation_artifact,
    render_session_transcript,
)
from EmoTrack.generative.schema import phq8_severity_band

logger = logging.getLogger(__name__)


@dataclass
class SessionScoringExample:
    run_id: str
    trend_name: str
    session_index: int  # 1-based
    profile_text: str  # from tracked profile JSON
    prior_transcripts: list[str]  # rendered text for sessions 1..t-1
    prior_phq_totals: list[float]  # SELF-REPORT scores (never latent)
    current_transcript: str  # rendered text for session t
    # Reward targets (NOT in prompt) — from self-report
    target_phq_total: float
    target_phq_items: dict[str, float]
    target_severity_band: str
    # Evaluation-only ground truth (never in training)
    latent_phq_total: float
    latent_phq_items: dict[str, float]


def build_scoring_dataset(
    result_root: Path,
    *,
    transcript_view: str = "full",
) -> list[SessionScoringExample]:
    """Walk result_root, find all run directories with transcript.jsonl + self_report.json,
    load each via load_conversation_artifact, then for each session build a SessionScoringExample.

    For profile_text: resolve the tracked client profile JSON from artifact.profile_id and
    serialize it as a JSON string. If not found, use empty string.

    For prior_phq_totals: use self_report_total from prior sessions (never latent).
    For target: use self_report_total and self_report_item_scores of current session.
    For latent: use latent_total and latent_item_scores (evaluation only).

    Skip sessions where self_report_total is None.
    """
    result_root = Path(result_root).expanduser().resolve()
    examples: list[SessionScoringExample] = []

    for transcript_path in sorted(result_root.rglob("common/transcript.jsonl")):
        if "eval" in transcript_path.parts:
            continue
        artifact_dir = transcript_path.parent.parent
        try:
            artifact = load_conversation_artifact(artifact_dir)
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("Skipping %s: %s", artifact_dir, exc)
            continue

        profile_text = _load_profile_text(artifact_dir)

        for idx, session in enumerate(artifact.sessions):
            if session.self_report_total is None:
                continue

            session_index = idx + 1  # 1-based

            prior_transcripts = [
                render_session_transcript(
                    prior, transcript_view=transcript_view
                )
                for prior in artifact.sessions[:idx]
            ]
            prior_phq_totals = [
                s.self_report_total
                for s in artifact.sessions[:idx]
                if s.self_report_total is not None
            ]

            current_transcript = render_session_transcript(
                session, transcript_view=transcript_view
            )

            examples.append(
                SessionScoringExample(
                    run_id=artifact.run_id,
                    trend_name=artifact.trend_name,
                    session_index=session_index,
                    profile_text=profile_text,
                    prior_transcripts=prior_transcripts,
                    prior_phq_totals=prior_phq_totals,
                    current_transcript=current_transcript,
                    target_phq_total=session.self_report_total,
                    target_phq_items=dict(session.self_report_item_scores),
                    target_severity_band=phq8_severity_band(
                        session.self_report_total
                    ),
                    latent_phq_total=session.latent_total,
                    latent_phq_items=dict(session.latent_item_scores),
                )
            )

    logger.info(
        "Built %d scoring examples from %s", len(examples), result_root
    )
    return examples


def export_jsonl(
    examples: list[SessionScoringExample], output_path: Path
) -> Path:
    """Write examples as one JSON object per line. Returns the output path."""
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(
                json.dumps(asdict(example), ensure_ascii=False) + "\n"
            )
    logger.info("Exported %d examples to %s", len(examples), output_path)
    return output_path


def _load_profile_text(artifact_dir: Path) -> str:
    """Load client_info from tracked profiles using the canonical profile_id."""
    artifact = load_conversation_artifact(artifact_dir)
    for candidate in _tracked_profile_candidates(artifact.profile_id):
        if not candidate.is_file():
            continue
        return _extract_client_info(candidate)
    return ""


def _extract_client_info(profile_path: Path) -> str:
    """Read a tracked profile and return client_info as a JSON string."""
    data = json.loads(profile_path.read_text(encoding="utf-8"))
    client_info = data.get("client_info")
    if client_info is None:
        return ""
    return json.dumps(client_info, ensure_ascii=False)


def _tracked_profile_candidates(profile_id: str) -> list[Path]:
    parts = [part.strip() for part in str(profile_id or "").split("_") if part.strip()]
    if len(parts) != 2:
        return []
    theoretical, client_id = parts
    candidates = [Path("profiles") / theoretical / f"{client_id}.json"]
    if client_id.isdigit():
        normalized = str(int(client_id))
        if normalized != client_id:
            candidates.append(Path("profiles") / theoretical / f"{normalized}.json")
    return [candidate.resolve() for candidate in candidates]
