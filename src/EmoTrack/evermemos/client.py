"""Stub — EverMemOS support has been removed.

Only NullEverMemOSClient and MemorySearchResult are kept so the
conversation orchestrator can instantiate a no-op memory backend.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemorySearchResult:
    snippets: list[str]
    prompt_snippets: list[str] = field(default_factory=list)
    raw_result: dict[str, Any] = field(default_factory=dict)


class EverMemOSClient:
    """No-op base — real implementation removed."""

    def close(self) -> None:
        pass

    def check_health(self) -> dict[str, Any]:
        return {"status": "disabled"}

    def ensure_conversation_meta(self, **kwargs: Any) -> dict[str, Any]:
        _ = kwargs
        return {}

    def store_message(self, **kwargs: Any) -> dict[str, Any]:
        _ = kwargs
        return {}

    def search_memories(self, **kwargs: Any) -> MemorySearchResult:
        _ = kwargs
        return MemorySearchResult(snippets=[], prompt_snippets=[], raw_result={})


NullEverMemOSClient = EverMemOSClient
