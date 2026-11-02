from __future__ import annotations

import re
from typing import Any

_CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")
_LATIN_WORD_PATTERN = re.compile(r"[A-Za-z]")
_NUMERIC_ONLY_PATTERN = re.compile(r"^[\s\d.,:+\-*/()%]+$")
_WORD_TOKEN_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def describe_invalid_visible_message(value: Any) -> str | None:
    if not isinstance(value, str):
        return "message must be a JSON string"

    cleaned = value.strip()
    if not cleaned:
        return "message is empty"
    if _CJK_PATTERN.search(cleaned):
        return "message must be English only"
    if _NUMERIC_ONLY_PATTERN.fullmatch(cleaned):
        return "message must be natural language, not just numbers"
    if not _LATIN_WORD_PATTERN.search(cleaned):
        return "message must include alphabetic English words"
    return None


def describe_echoing_message(
    value: Any,
    references: list[str],
    *,
    min_tokens: int = 8,
    min_prefix_tokens: int = 6,
    min_overlap_ratio: float = 0.8,
    min_candidate_coverage: float = 0.8,
) -> str | None:
    if not isinstance(value, str):
        return None
    candidate_tokens = _WORD_TOKEN_PATTERN.findall(value.lower())
    if len(candidate_tokens) < min_tokens:
        return None

    candidate_set = set(candidate_tokens)
    for reference in references:
        reference_tokens = _WORD_TOKEN_PATTERN.findall(str(reference or "").lower())
        if len(reference_tokens) < min_tokens:
            continue

        prefix_count = 0
        for candidate_token, reference_token in zip(candidate_tokens, reference_tokens):
            if candidate_token != reference_token:
                break
            prefix_count += 1
        if prefix_count >= min_prefix_tokens:
            return "message echoes prior wording too closely"

        reference_set = set(reference_tokens)
        shared = candidate_set & reference_set
        if not shared:
            continue
        overlap_ratio = len(shared) / max(1, len(candidate_set | reference_set))
        candidate_coverage = len(shared) / max(1, len(candidate_set))
        if overlap_ratio >= min_overlap_ratio and candidate_coverage >= min_candidate_coverage:
            return "message echoes prior wording too closely"
    return None
