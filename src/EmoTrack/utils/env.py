from __future__ import annotations

import os


def first_non_empty(*values: str) -> str:
    for value in values:
        normalized = str(value or "").strip()
        if normalized:
            return normalized
    return ""


def first_env(*names: str, default: str = "") -> str:
    for name in names:
        value = str(os.getenv(name, "")).strip()
        if value:
            return value
    return default
