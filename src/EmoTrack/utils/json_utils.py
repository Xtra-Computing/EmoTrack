from __future__ import annotations

from typing import Any
import json
import re


_JSON_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", flags=re.IGNORECASE)


def _find_balanced_object_slice(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
            continue
    return None


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    candidates: list[str] = [text]

    fenced = _JSON_CODE_FENCE_RE.search(text)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())

    balanced = _find_balanced_object_slice(text)
    if balanced:
        candidates.append(balanced.strip())

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if isinstance(parsed, dict):
            return parsed
        last_error = ValueError("Model output JSON is not an object")

    raise ValueError("Model output is not valid JSON object text") from last_error
