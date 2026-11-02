from __future__ import annotations

import json
from typing import Any


def prefers_max_completion_tokens(model: str) -> bool:
    return (model or "").strip().lower().startswith("gpt-5")


def extract_batch_message_content(result_line: dict[str, Any]) -> str:
    response_obj = result_line.get("response", {})
    if not isinstance(response_obj, dict):
        return ""

    body = response_obj.get("body", response_obj)
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except json.JSONDecodeError:
            return ""
    if not isinstance(body, dict):
        return ""

    choices = body.get("choices", [])
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message", {})
    if not isinstance(message, dict):
        return ""
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [str(item.get("text", "")).strip() for item in content if isinstance(item, dict)]
        return " ".join(part for part in parts if part).strip()
    return ""
