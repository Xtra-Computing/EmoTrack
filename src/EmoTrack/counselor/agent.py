from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CounselorTurn:
    message: str
    should_end: bool
    reason: str
    thinking: str = ""
    raw_message: dict[str, Any] | None = None


@dataclass
class CounselorProfile:
    style_prompt: str
    boundaries_prompt: str
    longitudinal_plan_template: str
    crisis_protocol: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CounselorProfile":
        required = [
            "style_prompt",
            "boundaries_prompt",
            "longitudinal_plan_template",
            "crisis_protocol",
        ]
        missing = [key for key in required if key not in data or not str(data[key]).strip()]
        if missing:
            raise ValueError(f"Counselor profile missing required fields: {', '.join(missing)}")
        return cls(
            style_prompt=str(data["style_prompt"]).strip(),
            boundaries_prompt=str(data["boundaries_prompt"]).strip(),
            longitudinal_plan_template=str(data["longitudinal_plan_template"]).strip(),
            crisis_protocol=str(data["crisis_protocol"]).strip(),
        )
