from __future__ import annotations

import copy
from typing import Any

from EmoTrack.utils.vertex_auth import normalize_vertex_model_name


_CONVERSATION_JSON_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "message": {"type": "STRING"},
        "emotion": {"type": "STRING"},
    },
    "required": ["message"],
    "propertyOrdering": ["message", "emotion"],
}

_REASONING_EFFORT_TO_LEVEL = {
    "low": "LOW",
    "medium": "HIGH",
    "high": "HIGH",
}


def build_vertex_generate_content_payload(
    *,
    messages: list[dict[str, Any]],
    model_name: str,
    temperature: float,
    max_tokens: int,
    response_format: dict[str, Any] | None,
    response_schema: dict[str, Any] | None,
    request_extras: dict[str, Any] | None,
    thinking_enabled: bool,
    thinking_budget_tokens: int,
) -> dict[str, Any]:
    extras = dict(request_extras or {})
    system_parts: list[dict[str, Any]] = []
    contents: list[dict[str, Any]] = []

    for item in messages:
        role = str(item.get("role") or "").strip().lower()
        parts = item.get("parts")
        if isinstance(parts, list):
            normalized_parts = [copy.deepcopy(part) for part in parts if isinstance(part, dict)]
            if not normalized_parts:
                continue
        else:
            content = str(item.get("content") or "").strip()
            if not content:
                continue
            normalized_parts = [{"text": content}]
        if role == "system":
            system_parts.extend(normalized_parts)
            continue
        contents.append(
            {
                "role": "model" if role in {"assistant", "model"} else "user",
                "parts": normalized_parts,
            }
        )

    if not contents:
        raise ValueError("Vertex Gemini requests require at least one non-system message")

    generation_config: dict[str, Any] = {
        "temperature": temperature,
        "maxOutputTokens": max_tokens,
    }

    top_p = extras.pop("top_p", None)
    if top_p is not None:
        generation_config["topP"] = top_p

    top_k = extras.pop("top_k", None)
    if top_k is not None:
        generation_config["topK"] = top_k

    frequency_penalty = extras.pop("frequency_penalty", None)
    if frequency_penalty is not None:
        generation_config["frequencyPenalty"] = frequency_penalty

    thinking_config = _build_thinking_config(
        request_extras=extras,
        enabled=thinking_enabled,
        budget_tokens=thinking_budget_tokens,
    )
    if thinking_config:
        generation_config["thinkingConfig"] = thinking_config

    if _wants_json_object(response_format):
        generation_config["responseMimeType"] = "application/json"
        generation_config["responseSchema"] = response_schema or _CONVERSATION_JSON_SCHEMA

    payload: dict[str, Any] = {
        "contents": contents,
        "generationConfig": generation_config,
    }
    if system_parts:
        payload["systemInstruction"] = {"parts": system_parts}
    return payload


def extract_vertex_generation_result(
    *,
    response_json: dict[str, Any],
    model_name: str,
) -> tuple[str, str, dict[str, Any], int | None]:
    candidates = response_json.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise RuntimeError("Vertex Gemini response has no candidates")
    candidate = candidates[0] if isinstance(candidates[0], dict) else {}
    content = candidate.get("content") if isinstance(candidate, dict) else {}
    parts = content.get("parts") if isinstance(content, dict) else []

    visible_parts: list[str] = []
    thinking_parts: list[str] = []
    thought_signatures: list[str] = []
    if isinstance(parts, list):
        for part in parts:
            if not isinstance(part, dict):
                continue
            text = str(part.get("text") or "").strip()
            if text:
                if part.get("thought") is True:
                    thinking_parts.append(text)
                else:
                    visible_parts.append(text)
            signature = str(part.get("thoughtSignature") or "").strip()
            if signature:
                thought_signatures.append(signature)

    usage = response_json.get("usageMetadata")
    usage_payload = usage if isinstance(usage, dict) else {}
    total_tokens = _safe_int(usage_payload.get("totalTokenCount"))

    raw_message = {
        "provider": "vertex_gemini",
        "model": normalize_vertex_model_name(model_name),
        "finish_reason": str(candidate.get("finishReason") or "").strip(),
        "usage": usage_payload,
        "thought_signatures": thought_signatures,
        "response": response_json,
    }
    return "\n".join(visible_parts).strip(), "\n".join(thinking_parts).strip(), raw_message, total_tokens


def _build_thinking_config(
    *,
    request_extras: dict[str, Any],
    enabled: bool,
    budget_tokens: int,
) -> dict[str, Any]:
    include_thoughts = bool(request_extras.pop("include_thoughts", False))
    thinking_level = str(request_extras.pop("thinking_level", "") or "").strip().upper()
    reasoning_effort = str(request_extras.pop("reasoning_effort", "") or "").strip().lower()

    if not enabled:
        config: dict[str, Any] = {"thinkingBudget": 0}
        if include_thoughts:
            config["includeThoughts"] = False
        return config

    config: dict[str, Any] = {}
    if budget_tokens > 0:
        config["thinkingBudget"] = budget_tokens
    elif thinking_level in {"LOW", "HIGH"}:
        config["thinkingLevel"] = thinking_level
    elif reasoning_effort in _REASONING_EFFORT_TO_LEVEL:
        config["thinkingLevel"] = _REASONING_EFFORT_TO_LEVEL[reasoning_effort]

    if include_thoughts:
        config["includeThoughts"] = True
    return config


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _wants_json_object(response_format: dict[str, Any] | None) -> bool:
    if not isinstance(response_format, dict):
        return False
    return str(response_format.get("type") or "").strip().lower() == "json_object"
