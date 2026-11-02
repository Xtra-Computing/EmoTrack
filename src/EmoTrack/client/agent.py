from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import json
import re

from EmoTrack.latent_variable import LATENT_MAPPING_PATH
from EmoTrack.latent_variable.behavior_map import PHQ8_ITEMS, build_symptom_behavior_targets

# These PHQ-8 depressive symptom criteria align with DSM-5 depressive-episode hints.
# Keep them as private generation hints only; never expose the criterion labels directly.
_PHQ8_ITEMS = list(PHQ8_ITEMS)
_BLOCKED_SCORING_PATTERN = re.compile(
    r"(on a scale\b|"
    r"rate (your|this|that|it)|"
    r"\bscore\b|"
    r"questionnaire|survey|checklist|"
    r"\b\d+\s*/\s*10\b|"
    r"\b\d+\s*out of\s*10\b|"
    r"\b0\s*(to|-)\s*100\b|"
    r"\b0\s*(to|-)\s*10\b|"
    r"\b0\s*(to|-)\s*4\b|"
    r"\b0\s*(to|-)\s*3\b|"
    r"\bphq-?8\b|\bphq-?9\b)",
    flags=re.IGNORECASE,
)
_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def _extract_item_scores(
    raw: Any,
    item_order: list[str],
    max_score: int,
    *,
    include_zero: bool = False,
) -> dict[str, float]:
    if not isinstance(raw, dict):
        raise TypeError(f"Expected dict latent scores, got {type(raw).__name__}")
    scores: dict[str, float] = {}
    for item in item_order:
        if item not in raw:
            raise KeyError(f"Missing latent item '{item}'")
        payload = raw[item]
        if not isinstance(payload, dict) or "score" not in payload:
            raise ValueError(f"Latent item '{item}' must be an object with key 'score'")
        score = _clamp(_safe_float(payload["score"]), 0.0, float(max_score))
        if include_zero or score > 0.0:
            scores[item] = score
    return scores


def _normalize_hidden_state(hidden_state: dict[str, Any]) -> dict[str, Any]:
    latent_state = hidden_state.get("latent_state")
    if not isinstance(latent_state, dict):
        raise ValueError("Hidden state must include object key 'latent_state'")
    state = latent_state

    if "phq8" not in state:
        raise KeyError("Hidden state missing latent_state.phq8")
    unsupported = sorted(set(state) & {"gad7", "ors", "cssrs"})
    if unsupported:
        joined = ", ".join(unsupported)
        raise ValueError(
            f"Unsupported latent_state keys in PHQ-8-only mode: {joined}. "
            "Remove GAD/ORS/CSSRS from hidden state."
        )

    macro = hidden_state.get("macro_labels")
    if not isinstance(macro, dict):
        raise ValueError("Hidden state must include object key 'macro_labels'")

    if "phq8_total" not in macro:
        raise KeyError("Hidden state missing macro_labels.phq8_total")
    unsupported_macro = sorted(set(macro) & {"gad7_total", "gad7_band", "ors_total", "cssrs_ideation_category"})
    if unsupported_macro:
        joined = ", ".join(unsupported_macro)
        raise ValueError(
            f"Unsupported macro_labels keys in PHQ-8-only mode: {joined}. "
            "Remove GAD/ORS/CSSRS macro labels."
        )

    phq8_scores = _extract_item_scores(state["phq8"], _PHQ8_ITEMS, max_score=3, include_zero=True)

    return {
        "phq8_scores": phq8_scores,
        "phq8_total": _safe_float(macro["phq8_total"]),
    }


def _read_behavior_mapping() -> dict[str, Any]:
    data = json.loads(LATENT_MAPPING_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping JSON object in {LATENT_MAPPING_PATH}")
    return data


def _severity_band(total: float | None, rules: list[dict[str, Any]]) -> str | None:
    if total is None:
        return None
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        lower = _safe_float(rule.get("min"))
        upper = _safe_float(rule.get("max"))
        if lower <= total <= upper:
            name = str(rule.get("band") or "").strip()
            return name or None
    return None


def _latent_behavior_cues(hidden_state: dict[str, Any]) -> tuple[list[str], list[str]]:
    if not hidden_state:
        return [], []

    mapping_doc = _read_behavior_mapping()
    mapping = mapping_doc.get("mapping")
    if not isinstance(mapping, dict):
        raise ValueError(f"Missing 'mapping' section in {LATENT_MAPPING_PATH}")

    normalized = _normalize_hidden_state(hidden_state)
    cues: list[str] = []
    targets = build_symptom_behavior_targets(
        normalized["phq8_scores"],
        mapping_doc,
        symptom_order=tuple(_PHQ8_ITEMS),
    )
    for target in targets:
        cues.extend(target.active_descriptors)

    deduped_cues = list(dict.fromkeys(cue for cue in cues if cue))

    severity_lines: list[str] = []
    severity_rules = mapping_doc.get("severity_bands", {})
    if isinstance(severity_rules, dict):
        phq_band = _severity_band(normalized["phq8_total"], severity_rules.get("phq8", []))
        if phq_band:
            severity_lines.append(f"depressive burden is currently around {phq_band.replace('_', ' ')} range")

    return deduped_cues, severity_lines


def _build_private_latent_prompt(hidden_state: dict[str, Any]) -> str:
    if not hidden_state:
        return ""
    cues, severity_lines = _latent_behavior_cues(hidden_state)
    if not cues and not severity_lines:
        return ""

    lines = [
        "### Private Latent Behavior Guidance",
        "Use these cues naturally in lived experience. Never mention scales, scores, item names, questionnaires, or hidden labels.",
        "Do not mention PHQ/PHQ-8/PHQ-9 or latent variable labels.",
    ]
    for line in severity_lines:
        lines.append(f"- {line}")
    if cues:
        for cue in cues:
            lines.append(f"- {cue}")
    return "\n".join(lines)


def _remove_leaky_scoring_language(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""

    parts = [p.strip() for p in _SENTENCE_SPLIT_PATTERN.split(text) if p.strip()]
    if not parts:
        parts = [text]

    kept = [part for part in parts if not _BLOCKED_SCORING_PATTERN.search(part)]
    cleaned = " ".join(kept).strip()
    if not cleaned:
        raise ValueError("Client output contained blocked scoring/latent-label language only")
    return cleaned


@dataclass
class ClientResponse:
    text: str
    emotion: str
    thinking: str = ""
    raw_message: dict[str, Any] | None = None
