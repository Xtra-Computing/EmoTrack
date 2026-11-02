from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from math import pi, sin
from pathlib import Path
from typing import Any
import copy
import json

logger = logging.getLogger(__name__)

import yaml

from EmoTrack.behavior_cue_mapping import PHQ8_RUNTIME_CUE_BANK_PATH


LATENT_MAPPING_PATH = PHQ8_RUNTIME_CUE_BANK_PATH

_PHQ8_ITEM_PATHS = {
    "latent_state.phq8.anhedonia.score",
    "latent_state.phq8.depressed_mood.score",
    "latent_state.phq8.sleep.score",
    "latent_state.phq8.fatigue.score",
    "latent_state.phq8.appetite.score",
    "latent_state.phq8.self_worth.score",
    "latent_state.phq8.concentration.score",
    "latent_state.phq8.psychomotor.score",
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp_float(value: float, lower: float | None, upper: float | None) -> float:
    if lower is not None:
        value = max(lower, value)
    if upper is not None:
        value = min(upper, value)
    return value


def _get_by_path(payload: dict[str, Any], path: str) -> Any:
    cursor: Any = payload
    for part in path.split("."):
        if not isinstance(cursor, dict):
            raise KeyError(f"Path '{path}' is not reachable")
        if part not in cursor:
            raise KeyError(f"Path '{path}' is missing at '{part}'")
        cursor = cursor[part]
    return cursor


def _set_by_path(payload: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cursor: Any = payload
    for part in parts[:-1]:
        if not isinstance(cursor, dict) or part not in cursor:
            raise KeyError(f"Path '{path}' is missing at '{part}'")
        cursor = cursor[part]
    if not isinstance(cursor, dict):
        raise KeyError(f"Cannot set path '{path}'")
    cursor[parts[-1]] = value


def _format_number(value: float) -> str:
    if value.is_integer():
        return f"{value:.1f}"
    return f"{value:g}"


def _severity_band_name(metric: str, total: float) -> str | None:
    raw = json.loads(LATENT_MAPPING_PATH.read_text(encoding="utf-8"))
    rules = raw.get("severity_bands", {}).get(metric, [])
    if not isinstance(rules, list):
        return None
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        lower = _safe_float(rule.get("min"))
        upper = _safe_float(rule.get("max"))
        if lower <= total <= upper:
            band = str(rule.get("band") or "").strip()
            return band or None
    return None


def _infer_bounds(path: str, _current_value: Any) -> tuple[float | None, float | None]:
    if path in _PHQ8_ITEM_PATHS:
        return 0.0, 3.0
    if path.endswith("phq8_total"):
        return 0.0, 24.0
    raise ValueError(
        f"Unsupported trend target path '{path}'. PHQ-8-only latent mode accepts only "
        f"'latent_state.phq8.*.score' and 'macro_labels.phq8_total'."
    )


@dataclass(frozen=True)
class LatentTrend:
    name: str
    scale: float = 0.0
    period: float | None = None
    phase: float = 0.0
    schedule: tuple[float, ...] = ()

    @property
    def abbreviation(self) -> str:
        if self.name == "stable":
            return "stable"
        if self.name == "increase":
            return f"inc_{_format_number(abs(self.scale))}"
        if self.name == "decrease":
            return f"dec_{_format_number(abs(self.scale))}"
        if self.name == "fluctuate":
            period = self.period if self.period is not None else 2.0
            return f"fluct_{_format_number(abs(self.scale))}_{_format_number(period)}"
        if self.name == "schedule":
            return f"sched_{len(self.schedule)}"
        raise ValueError(f"Unsupported trend name: {self.name}")

    def absolute(self, session_id: int) -> float | None:
        if not self.schedule:
            return None
        idx = min(max(0, session_id - 1), len(self.schedule) - 1)
        return self.schedule[idx]

    def delta(self, session_id: int) -> float:
        session_index = max(0, session_id - 1)
        if self.name == "stable":
            return 0.0
        if self.name == "increase":
            return abs(self.scale) * session_index
        if self.name == "decrease":
            return -abs(self.scale) * session_index
        if self.name == "fluctuate":
            period = self.period if self.period not in (None, 0.0) else 2.0
            return self.scale * sin((2.0 * pi * session_index / period) + self.phase)
        raise ValueError(f"Unsupported trend name: {self.name}")

    @classmethod
    def from_value(cls, raw: Any) -> "LatentTrend":
        if isinstance(raw, list):
            return cls(name="schedule", schedule=tuple(float(v) for v in raw))
        if isinstance(raw, str):
            text = raw.strip().lower()
            if not text or text == "stable":
                return cls(name="stable", scale=0.0)
            parts = text.split("_")
            head = parts[0]
            if head in {"inc", "increase"}:
                return cls(name="increase", scale=_safe_float(parts[1], 0.0))
            if head in {"dec", "decrease"}:
                return cls(name="decrease", scale=_safe_float(parts[1], 0.0))
            if head in {"fluct", "fluctuate"}:
                scale = _safe_float(parts[1], 0.0)
                period = _safe_float(parts[2], 2.0) if len(parts) > 2 else 2.0
                phase = _safe_float(parts[3], 0.0) if len(parts) > 3 else 0.0
                return cls(name="fluctuate", scale=scale, period=period, phase=phase)
            raise ValueError(f"Unsupported trend abbreviation: {raw}")
        if isinstance(raw, dict):
            name = str(raw.get("name") or "stable").strip().lower()
            if not name:
                name = "stable"
            if name == "schedule":
                values = raw.get("values", [])
                return cls(name="schedule", schedule=tuple(float(v) for v in values))
            if name not in {"stable", "increase", "decrease", "fluctuate"}:
                raise ValueError(f"Unsupported trend name: {name}")
            return cls(
                name=name,
                scale=_safe_float(raw.get("scale"), 0.0),
                period=_safe_float(raw.get("period"), 0.0) if raw.get("period") is not None else None,
                phase=_safe_float(raw.get("phase"), 0.0),
            )
        raise TypeError(f"Unsupported trend payload type: {type(raw).__name__}")


@dataclass(frozen=True)
class TrendTarget:
    path: str
    trend: LatentTrend

    @classmethod
    def from_value(cls, path: str, raw: Any) -> "TrendTarget":
        return cls(path=path, trend=LatentTrend.from_value(raw))


@dataclass(frozen=True)
class ResolvedTrendTarget:
    path: str
    name: str
    abbreviation: str
    scale: float
    period: float | None
    phase: float
    base_value: float
    delta: float
    resolved_value: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResolvedLatentState:
    session_id: int
    hidden_state: dict[str, Any]
    applied_trends: list[ResolvedTrendTarget]

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "hidden_state": self.hidden_state,
            "applied_trends": [item.to_dict() for item in self.applied_trends],
        }

    def summary_lines(self) -> list[str]:
        lines: list[str] = []
        for target in self.applied_trends:
            lines.append(
                f"{target.path} follows {target.name} trend {target.abbreviation}; "
                f"base={_format_number(target.base_value)}, delta={_format_number(target.delta)}, "
                f"current={_format_number(target.resolved_value)}"
            )
        return lines


@dataclass(frozen=True)
class LatentTrendPlan:
    version: str
    description: str
    targets: list[TrendTarget]

    @classmethod
    def empty(cls) -> "LatentTrendPlan":
        return cls(version="1.0", description="No latent trend targets", targets=[])

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "LatentTrendPlan":
        version = str(raw.get("version") or "1.0").strip()
        description = str(raw.get("description") or "").strip()
        targets_raw = raw.get("targets", {})
        if not isinstance(targets_raw, dict):
            raise TypeError("Trend plan field 'targets' must be an object")
        targets = [TrendTarget.from_value(path, value) for path, value in sorted(targets_raw.items())]
        return cls(version=version, description=description, targets=targets)

    @classmethod
    def from_path(cls, path: str | Path | None) -> "LatentTrendPlan":
        if path is None:
            return cls.empty()
        raw_path = Path(path)
        if not raw_path.exists():
            return cls.empty()
        text = raw_path.read_text(encoding="utf-8")
        if raw_path.suffix.lower() in {".yaml", ".yml"}:
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError(f"Expected trend plan object in {raw_path}")
        return cls.from_dict(data)

    def resolve(self, hidden_state: dict[str, Any], session_id: int) -> ResolvedLatentState:
        resolved_hidden_state = copy.deepcopy(hidden_state)
        applied: list[ResolvedTrendTarget] = []

        for target in self.targets:
            if target.trend.schedule and session_id > len(target.trend.schedule):
                logger.warning(
                    "session_id %d exceeds schedule length %d for %s; "
                    "clamping to last scheduled value",
                    session_id, len(target.trend.schedule), target.path,
                )
            # Validate target namespace before path lookup so legacy non-PHQ targets fail clearly.
            lower, upper = _infer_bounds(target.path, None)
            current_value = _get_by_path(resolved_hidden_state, target.path)
            base_value = _safe_float(current_value)
            abs_target = target.trend.absolute(session_id)
            if abs_target is not None:
                resolved_value = _clamp_float(abs_target, lower, upper)
                delta = resolved_value - base_value
            else:
                delta = target.trend.delta(session_id)
                resolved_value = _clamp_float(base_value + delta, lower, upper)
            _set_by_path(resolved_hidden_state, target.path, resolved_value)
            applied.append(
                ResolvedTrendTarget(
                    path=target.path,
                    name=target.trend.name,
                    abbreviation=target.trend.abbreviation,
                    scale=target.trend.scale,
                    period=target.trend.period,
                    phase=target.trend.phase,
                    base_value=base_value,
                    delta=delta,
                    resolved_value=resolved_value,
                )
            )

        self._recompute_macro_labels(resolved_hidden_state)
        return ResolvedLatentState(session_id=session_id, hidden_state=resolved_hidden_state, applied_trends=applied)

    @staticmethod
    def _recompute_macro_labels(hidden_state: dict[str, Any]) -> None:
        latent_state = hidden_state.get("latent_state", {})
        if not isinstance(latent_state, dict):
            return
        unsupported = sorted(set(latent_state) & {"gad7", "ors", "cssrs"})
        if unsupported:
            joined = ", ".join(unsupported)
            raise ValueError(
                f"Unsupported latent_state keys in PHQ-8-only mode: {joined}. "
                "Remove GAD/ORS/CSSRS from hidden state."
            )
        macro = hidden_state.setdefault("macro_labels", {})
        if not isinstance(macro, dict):
            raise TypeError("Hidden state field 'macro_labels' must be an object")

        phq_items = latent_state.get("phq8", {})
        if isinstance(phq_items, dict):
            phq_total = sum(_safe_float((phq_items.get(item) or {}).get("score")) for item in (path.split(".")[2] for path in _PHQ8_ITEM_PATHS))
            macro["phq8_total"] = phq_total
            band = _severity_band_name("phq8", phq_total)
            if band:
                macro["phq8_band"] = band
