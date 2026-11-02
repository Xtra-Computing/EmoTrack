from __future__ import annotations

from typing import Any

from EmoTrack.utils.json_utils import extract_json_object

_NESTED_PAYLOAD_KEYS = ("message", "content", "data", "result", "output")


def _normalize_summary_list(value: Any, *, key: str) -> list[str]:
    _ = key
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if not isinstance(value, list):
        text = str(value).strip()
        return [text] if text else []
    normalized: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            normalized.append(text)
    return normalized


def _normalize_summary_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = _normalize_summary_list(value, key="summary")
        return " ".join(parts).strip()
    raise ValueError("Summary output field 'summary' must be a string or list of strings")


def _unwrap_embedded_summary_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    for key in _NESTED_PAYLOAD_KEYS:
        if key not in payload:
            continue
        value = payload[key]
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            text = value.strip()
            if not text:
                continue
            try:
                return extract_json_object(text)
            except ValueError:
                return {
                    "summary": text,
                    "key_facts": [],
                    "goals": [],
                    "homework": [],
                    "risk_flags": [],
                }
    return None


def parse_summary_output(text: str) -> dict[str, Any]:
    parsed = extract_json_object(text)
    required = ["summary", "key_facts", "goals", "homework", "risk_flags"]
    missing = [key for key in required if key not in parsed]
    if missing:
        nested = _unwrap_embedded_summary_payload(parsed)
        if nested is not None:
            parsed = nested
            missing = [key for key in required if key not in parsed]
    if missing:
        raise ValueError(f"Summary output missing required keys: {', '.join(missing)}")
    summary_text = _normalize_summary_text(parsed["summary"])
    if not summary_text:
        raise ValueError("Summary output field 'summary' must be non-empty")
    return {
        "summary": summary_text,
        "key_facts": _normalize_summary_list(parsed["key_facts"], key="key_facts"),
        "goals": _normalize_summary_list(parsed["goals"], key="goals"),
        "homework": _normalize_summary_list(parsed["homework"], key="homework"),
        "risk_flags": _normalize_summary_list(parsed["risk_flags"], key="risk_flags"),
    }
