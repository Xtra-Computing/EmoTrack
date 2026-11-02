"""Canonical PHQ-8 scoring schema shared across generative methods."""
from __future__ import annotations

import json

SYSTEM_PROMPT = "You are a clinical psychology expert. Return valid JSON only."

PHQ8_SYMPTOMS = [
    "anhedonia",
    "depressed_mood",
    "sleep",
    "fatigue",
    "appetite",
    "self_worth",
    "concentration",
    "psychomotor",
]

DEFAULT_TEMPLATE_PATH = "configs/train/emotrack_prompt.jinja2"

# Chat template kwargs for Qwen thinking-tag suppression during training
CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}

# JSON schema for guided decoding (vLLM / OpenAI API)
PHQ8_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "object",
            "properties": {s: {"type": "number"} for s in PHQ8_SYMPTOMS},
            "required": list(PHQ8_SYMPTOMS),
            "additionalProperties": False,
        },
        "total": {"type": "number"},
    },
    "required": ["items", "total"],
    "additionalProperties": False,
}

PHQ8_JSON_SCHEMA_STR = json.dumps(PHQ8_JSON_SCHEMA)

PHQ8_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "PHQ8Score",
        "strict": True,
        "schema": PHQ8_JSON_SCHEMA,
    },
}


def phq8_severity_band(total: float) -> str:
    if total < 5:
        return "minimal"
    if total < 10:
        return "mild"
    if total < 15:
        return "moderate"
    if total < 20:
        return "moderately_severe"
    return "severe"


def format_target_json(target: dict) -> str:
    """Build the canonical JSON completion matching the prompt schema.

    Input: {"phq_total": float, "phq_items": {symptom: score, ...}}
    Output: '{"items": {...}, "total": N}'
    """
    items = target["phq_items"]
    total = sum(float(items[s]) for s in PHQ8_SYMPTOMS)
    return json.dumps(
        {"items": {s: items[s] for s in PHQ8_SYMPTOMS}, "total": total},
        indent=2,
    )
