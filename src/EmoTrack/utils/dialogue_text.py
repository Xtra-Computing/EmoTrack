from __future__ import annotations

import logging
import re
import copy
from typing import Any

logger = logging.getLogger(__name__)


def history_speaker_label(raw_speaker: str) -> str:
    normalized = (raw_speaker or "").strip().lower()
    if normalized in {"counselor", "assistant", "therapist"}:
        return "counselor"
    if normalized in {"client", "user", "patient"}:
        return "client"
    return normalized or "client"


def normalize_space(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def short_phrase_list(
    items: list[str],
    *,
    phrase_limit: int = 4,
    max_words: int = 8,
    max_chars: int = 72,
) -> list[str]:
    phrases: list[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = normalize_space(item)
        if not normalized:
            continue
        chunks = re.split(r"[.!?;\n]+", normalized)
        for chunk in chunks:
            chunk = normalize_space(chunk)
            if not chunk:
                continue
            words = chunk.split()
            if len(words) > max_words:
                chunk = " ".join(words[:max_words]).rstrip(",:") + "..."
            if len(chunk) > max_chars:
                chunk = chunk[: max_chars - 3].rstrip()
                if " " in chunk:
                    chunk = chunk.rsplit(" ", 1)[0]
                chunk = chunk.rstrip(",:") + "..."
            lowered = chunk.lower()
            if len(words) < 3 or lowered in seen:
                continue
            seen.add(lowered)
            phrases.append(chunk)
            if len(phrases) >= phrase_limit:
                return phrases
    return phrases


def phase_lengths(total_turns: int, phase_count: int) -> list[int]:
    count = max(1, min(phase_count, total_turns))
    base = total_turns // count
    remainder = total_turns % count
    return [base + (1 if idx < remainder else 0) for idx in range(count)]


def build_chat_history_messages(
    session_history: list[dict[str, str]],
    *,
    assistant_speaker: str,
) -> list[dict[str, str]]:
    assistant_label = history_speaker_label(assistant_speaker)
    messages: list[dict[str, str]] = []
    for item in session_history:
        speaker = history_speaker_label(str(item.get("speaker", "")))
        text = normalize_space(item.get("text", ""))
        if speaker not in {"client", "counselor"} or not text:
            continue
        role = "assistant" if speaker == assistant_label else "user"
        if messages and messages[-1]["role"] == role:
            logger.debug("Merging consecutive %s turn into previous message", role)
            messages[-1]["content"] = messages[-1]["content"] + "\n" + text
        else:
            messages.append({"role": role, "content": text})
    return messages


def extract_vertex_history_parts(raw_message: Any) -> list[dict[str, Any]] | None:
    if not isinstance(raw_message, dict):
        return None

    response = raw_message.get("response")
    if isinstance(response, dict):
        candidates = response.get("candidates")
        if isinstance(candidates, list) and candidates and isinstance(candidates[0], dict):
            content = candidates[0].get("content")
            parts = content.get("parts") if isinstance(content, dict) else None
            if isinstance(parts, list) and parts:
                cleaned_parts = [copy.deepcopy(part) for part in parts if isinstance(part, dict)]
                if cleaned_parts:
                    return cleaned_parts

    for key in ("repair_message", "initial_message"):
        nested_parts = extract_vertex_history_parts(raw_message.get(key))
        if nested_parts:
            return nested_parts
    return None


def build_vertex_chat_history_messages(
    session_history: list[dict[str, Any]],
    *,
    assistant_speaker: str,
) -> list[dict[str, Any]]:
    assistant_label = history_speaker_label(assistant_speaker)
    messages: list[dict[str, Any]] = []
    for item in session_history:
        speaker = history_speaker_label(str(item.get("speaker", "")))
        if speaker not in {"client", "counselor"}:
            continue
        role = "assistant" if speaker == assistant_label else "user"

        if role == "assistant":
            parts = item.get("vertex_history_parts")
            if not isinstance(parts, list):
                parts = extract_vertex_history_parts(item.get("raw_message"))
            if isinstance(parts, list) and parts:
                messages.append({"role": role, "parts": copy.deepcopy(parts)})
                continue

        text = normalize_space(item.get("text", ""))
        if not text:
            continue
        if messages and messages[-1].get("role") == role and "content" in messages[-1]:
            logger.debug("Merging consecutive %s turn into previous message", role)
            messages[-1]["content"] = str(messages[-1]["content"]) + "\n" + text
        else:
            messages.append({"role": role, "content": text})
    return messages
