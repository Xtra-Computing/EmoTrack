from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationMethodSpec:
    name: str
    requires_llm_by_default: bool = False


METHOD_SPECS: tuple[EvaluationMethodSpec, ...] = (
    EvaluationMethodSpec("psycheval", requires_llm_by_default=True),
    EvaluationMethodSpec("aida", requires_llm_by_default=True),
    EvaluationMethodSpec("lmiq", requires_llm_by_default=True),
    EvaluationMethodSpec("ensembert"),
    EvaluationMethodSpec("dual_encoder"),
    EvaluationMethodSpec("milintsevich"),
    EvaluationMethodSpec("emotrack"),
)

EVALUATION_METHODS: tuple[str, ...] = tuple(spec.name for spec in METHOD_SPECS)


def validate_method_name(method_name: str) -> str:
    normalized = str(method_name or "").strip()
    if normalized not in EVALUATION_METHODS:
        raise ValueError(f"Unsupported evaluation method: {method_name}")
    return normalized
