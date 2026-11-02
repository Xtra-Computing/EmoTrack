from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any


PHQ8_ITEMS = (
    "anhedonia",
    "depressed_mood",
    "sleep",
    "fatigue",
    "appetite",
    "self_worth",
    "concentration",
    "psychomotor",
)

BEHAVIOR_CUE_ORDER = (
    "affect",
    "cognition_language",
    "behavior_functioning",
    "speech_interaction",
    "somatic_physiological",
)

PROMPT_ROLE_GUIDANCE = {
    "intact_capacity": (
        "Treat this as an intact area of functioning. It may show up naturally as a strength or "
        "ordinary part of life, but it should not be portrayed as a burden."
    ),
    "background_tint": (
        "Treat this as a faint trace that may color tone under stress. Do not make it a main "
        "complaint or recurring focus."
    ),
    "secondary_burden": (
        "Treat this as a real but not dominant burden. It should support the main session state "
        "without taking over the whole conversation."
    ),
    "dominant_burden": (
        "Treat this as a central burden likely to show up spontaneously in lived details, body "
        "state, and day-to-day functioning."
    ),
}

DEFAULT_CUES_PER_SYMPTOM = 5


@dataclass(frozen=True, slots=True)
class BehaviorCue:
    domain: str
    text: str


@dataclass(frozen=True, slots=True)
class SymptomBehaviorTarget:
    symptom_name: str
    score: float
    lower_level: int
    upper_level: int
    blend_ratio: float
    prompt_role: str
    prompt_guidance: str
    zero_state_descriptors: list[str]
    active_descriptors: list[str]
    active_by_domain: dict[str, list[str]]


def build_symptom_behavior_targets(
    item_scores: dict[str, float],
    mapping_payload: dict[str, Any],
    *,
    symptom_order: tuple[str, ...] = PHQ8_ITEMS,
    rng: random.Random | None = None,
    cues_per_symptom: int = DEFAULT_CUES_PER_SYMPTOM,
) -> list[SymptomBehaviorTarget]:
    if cues_per_symptom <= 0:
        raise ValueError("cues_per_symptom must be positive")

    phq_mapping = mapping_payload["mapping"]["phq8"]
    sampler = rng if rng is not None else random.Random()
    targets: list[SymptomBehaviorTarget] = []
    for symptom in symptom_order:
        score = max(0.0, min(3.0, float(item_scores[symptom])))
        lower_level = max(0, min(3, math.floor(score)))
        upper_level = max(0, min(3, math.ceil(score)))
        raw_blend_ratio = score - lower_level
        blend_ratio = round(raw_blend_ratio, 2)
        lower_entries = _behavior_entries_for_level(phq_mapping, symptom, lower_level)
        upper_entries = _behavior_entries_for_level(phq_mapping, symptom, upper_level)
        active_entries = _blend_behavior_entries(
            lower_level=lower_level,
            upper_level=upper_level,
            blend_ratio=raw_blend_ratio,
            lower_entries=lower_entries,
            upper_entries=upper_entries,
            rng=sampler,
            cues_per_symptom=cues_per_symptom,
        )
        zero_state_descriptors = [entry.text for entry in active_entries] if score <= 0 else []
        active_by_domain = {domain: [] for domain in BEHAVIOR_CUE_ORDER}
        for entry in active_entries:
            active_by_domain.setdefault(entry.domain, []).append(entry.text)
        targets.append(
            SymptomBehaviorTarget(
                symptom_name=symptom,
                score=score,
                lower_level=lower_level,
                upper_level=upper_level,
                blend_ratio=blend_ratio,
                prompt_role=_prompt_role_for_score(score),
                prompt_guidance=_prompt_guidance_for_score(score),
                zero_state_descriptors=zero_state_descriptors,
                active_descriptors=[entry.text for entry in active_entries],
                active_by_domain=active_by_domain,
            )
        )
    return targets


def _behavior_entries_for_level(
    phq_mapping: dict[str, Any],
    symptom: str,
    level: int,
) -> list[BehaviorCue]:
    severity_definition = phq_mapping[symptom]["severity_levels"][str(max(0, min(3, level)))]
    entries: list[BehaviorCue] = []
    if isinstance(severity_definition, list):
        # Flat list format (e.g. phq8_behavior_cue_bank.json):
        # distribute cues round-robin across domains.
        for i, value in enumerate(severity_definition):
            text = str(value or "").strip()
            if text:
                domain = BEHAVIOR_CUE_ORDER[i % len(BEHAVIOR_CUE_ORDER)]
                entries.append(BehaviorCue(domain=domain, text=text))
    else:
        # Dict format (e.g. legacy score_behavior_mapping.json):
        # each domain key maps to a list of cues.
        for domain in BEHAVIOR_CUE_ORDER:
            values = severity_definition.get(domain)
            if not isinstance(values, list):
                continue
            for value in values:
                text = str(value or "").strip()
                if text:
                    entries.append(BehaviorCue(domain=domain, text=text))
    return entries


def _blend_behavior_entries(
    *,
    lower_level: int,
    upper_level: int,
    blend_ratio: float,
    lower_entries: list[BehaviorCue],
    upper_entries: list[BehaviorCue],
    rng: random.Random,
    cues_per_symptom: int = DEFAULT_CUES_PER_SYMPTOM,
) -> list[BehaviorCue]:
    if cues_per_symptom <= 0:
        raise ValueError("cues_per_symptom must be positive")

    if lower_level == upper_level:
        return _sample_behavior_entries(
            lower_entries,
            cue_count=cues_per_symptom,
            rng=rng,
        )

    target_upper_count = max(0.0, min(float(cues_per_symptom), blend_ratio * cues_per_symptom))
    carry_upper = _stochastic_rounded_count(target_upper_count, rng=rng)
    carry_lower = cues_per_symptom - carry_upper
    blended_entries = _sample_behavior_entries(lower_entries, cue_count=carry_lower, rng=rng) + _sample_behavior_entries(
        upper_entries,
        cue_count=carry_upper,
        rng=rng,
    )
    rng.shuffle(blended_entries)
    return blended_entries


def _sample_behavior_entries(
    entries: list[BehaviorCue],
    *,
    cue_count: int,
    rng: random.Random,
) -> list[BehaviorCue]:
    if cue_count < 0:
        raise ValueError("cue_count must be non-negative")
    if cue_count == 0:
        return []
    if cue_count > len(entries):
        raise ValueError(
            f"Behavior cue cell contains only {len(entries)} cues, but {cue_count} were requested"
        )
    return list(rng.sample(entries, k=cue_count))


def _stochastic_rounded_count(target_count: float, *, rng: random.Random) -> int:
    base_count = math.floor(target_count)
    remainder = max(0.0, min(1.0, target_count - base_count))
    return base_count + int(rng.random() < remainder)


def _prompt_role_for_score(score: float) -> str:
    if score <= 0:
        return "intact_capacity"
    if score <= 0.4:
        return "background_tint"
    if score <= 1.4:
        return "secondary_burden"
    return "dominant_burden"


def _prompt_guidance_for_score(score: float) -> str:
    return PROMPT_ROLE_GUIDANCE[_prompt_role_for_score(score)]
