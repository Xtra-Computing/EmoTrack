from __future__ import annotations

import re
from typing import Any

from EmoTrack.latent_variable.behavior_map import PHQ8_ITEMS


STAGE_NAME_EN = {
    "问题概念化与目标设定": "Case Conceptualization and Goal Setting",
    "核心认知与行为干预": "Core Cognitive and Behavioral Intervention",
    "巩固与复发预防": "Consolidation and Relapse Prevention",
}


def extract_session_num(key: str) -> int:
    match = re.search(r"第(\d+)次", str(key))
    return int(match.group(1)) if match else -1


def stage_lookup(sample: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for stage in sample.get("global_plan", []):
        if not isinstance(stage, dict):
            continue
        base = {
            "stage_number": stage.get("stage_number"),
            "stage_name": stage.get("stage_name"),
            "sessions": stage.get("sessions"),
        }
        content = stage.get("content", {})
        if not isinstance(content, dict):
            continue
        for key, session in content.items():
            session_num = extract_session_num(str(key))
            if session_num <= 0:
                continue
            info = dict(base)
            info["plan_session_key"] = key
            info["plan_session_content"] = session
            out[session_num] = info
    return out


def to_stage_en(name: str) -> str:
    return STAGE_NAME_EN.get(str(name), str(name))


def _normalize_space(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _flatten_text_fragments(value: Any) -> list[str]:
    out: list[str] = []
    if isinstance(value, str):
        normalized = _normalize_space(value)
        if normalized:
            out.append(normalized)
        return out
    if isinstance(value, list):
        for item in value:
            out.extend(_flatten_text_fragments(item))
        return out
    if isinstance(value, dict):
        for item in value.values():
            out.extend(_flatten_text_fragments(item))
        return out
    return out


def _band(score: float, scale: str) -> str:
    if scale == "phq8":
        if score <= 4:
            return "minimal"
        if score <= 9:
            return "mild"
        if score <= 14:
            return "moderate"
        if score <= 19:
            return "moderately_severe"
        return "severe"
    return "unknown"


def distribute_phq8_scores(total: int, items: list[str], base: dict[str, int], cap: int = 3) -> dict[str, int]:
    base_scores = {item: int(base.get(item, 0)) for item in items}
    current = sum(base_scores.values())
    while current < total:
        updated = False
        for item in items:
            if base_scores[item] < cap:
                base_scores[item] += 1
                current += 1
                updated = True
                if current >= total:
                    break
        if not updated:
            break
    while current > total:
        updated = False
        for item in reversed(items):
            if base_scores[item] > 0:
                base_scores[item] -= 1
                current -= 1
                updated = True
                if current <= total:
                    break
        if not updated:
            break
    return base_scores


def choose_stage_phq_total(stage_num: int, session_num: int) -> int:
    fixed = {
        1: 13,
        2: 12,
        3: 11,
        4: 10,
        5: 8,
        6: 6,
    }
    if session_num in fixed:
        return fixed[session_num]
    if stage_num == 1:
        return 12
    if stage_num == 2:
        return 10
    return 7


def infer_latent_profile(session: dict[str, Any], stage: dict[str, Any], session_idx: int) -> dict[str, Any]:
    session_num = int(session.get("session_number", session_idx + 1))
    stage_num = int(stage.get("stage_number", 2))
    phq_total = choose_stage_phq_total(stage_num, session_num)

    session_goals = session.get("session_goals", {})
    if not isinstance(session_goals, dict):
        session_goals = {}
    session_focus = session_goals.get("session_focus", {})
    if not isinstance(session_focus, dict):
        session_focus = {}

    focus = " ".join(_flatten_text_fragments(session_focus.get("objective", [])))
    plan_text = " ".join(_flatten_text_fragments(stage.get("plan_session_content", {})))
    text = f"{focus} {plan_text}"

    phq_base = {
        "sleep": 2,
        "fatigue": 2,
        "concentration": 2,
        "depressed_mood": 1,
        "self_worth": 1,
        "anhedonia": 1,
        "appetite": 1,
        "psychomotor": 1,
    }
    if "睡眠" in text or "失眠" in text:
        phq_base["sleep"] += 1
        phq_base["fatigue"] += 1
    if "人际" in text or "宿舍" in text:
        phq_base["self_worth"] += 1
    if "完美" in text or "失败" in text or "被否定" in text:
        phq_base["self_worth"] += 1
        phq_base["depressed_mood"] += 1
    if "回避" in text or "孤独" in text:
        phq_base["anhedonia"] += 1
    if "巩固" in text or "复发预防" in text:
        for key in list(phq_base):
            phq_base[key] = max(0, phq_base[key] - 1)

    phq_scores = distribute_phq8_scores(phq_total, list(PHQ8_ITEMS), phq_base)

    return {
        "macro_labels": {
            "phq8_total": phq_total,
            "phq8_band": _band(phq_total, "phq8"),
        },
        "latent_state": {
            "phq8": {
                **{key: {"score": value} for key, value in phq_scores.items()},
                "sleep": {"score": phq_scores["sleep"], "direction": "insomnia"},
                "appetite": {"score": phq_scores["appetite"], "direction": "decrease"},
                "psychomotor": {"score": phq_scores["psychomotor"], "polarity": "slowed"},
            },
        },
    }


def active_items_from_latent_profile(latent_profile: dict[str, Any]) -> dict[str, list[str]]:
    latent_state = latent_profile.get("latent_state", {}) if isinstance(latent_profile, dict) else {}
    phq8 = latent_state.get("phq8", {}) if isinstance(latent_state, dict) else {}
    return {
        "phq8": [key for key, value in phq8.items() if isinstance(value, dict) and _safe_float(value.get("score")) >= 2.0],
    }
