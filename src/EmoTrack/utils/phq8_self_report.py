from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from EmoTrack.latent_variable import PHQ8_SCALE_PATH
from EmoTrack.latent_variable.behavior_map import PHQ8_ITEMS
from EmoTrack.utils.json_utils import extract_json_object


PHQ8_SELF_REPORT_USER_PROMPT = (
    "Complete the PHQ-8 self-report now using the exact option texts listed in <context_and_data>."
)
_NESTED_PAYLOAD_KEYS = ("message", "content", "data", "result", "output")
ITEM_ID_KEY_CANDIDATES = ("item_id", "item", "id", "question_id")
RESPONSE_KEY_CANDIDATES = ("response", "answer", "choice", "option")
REASON_KEY_CANDIDATES = ("reason", "rationale", "why")
_MULTISPACE_RE = re.compile(r"\s+")


def load_json_file(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def load_phq8_scale_payload(path: str | Path = PHQ8_SCALE_PATH) -> dict[str, Any]:
    return load_json_file(path)


def tagged_prompt_block(tag: str, *sections: str) -> str:
    content = "\n\n".join(section.strip() for section in sections if section and section.strip())
    return f"<{tag}>\n{content}\n</{tag}>"


def item_id_to_symptom_map(
    scale_payload: dict[str, Any],
    *,
    symptom_order: tuple[str, ...] = PHQ8_ITEMS,
) -> dict[int, str]:
    raw_items = scale_payload.get("items")
    if not isinstance(raw_items, list) or len(raw_items) != len(symptom_order):
        raise ValueError(f"PHQ-8 scale payload must include {len(symptom_order)} items")
    return {
        int(item["id"]): symptom_name
        for symptom_name, item in zip(symptom_order, raw_items, strict=True)
    }


def build_phq8_self_report_prompt(
    *,
    scale_payload: dict[str, Any],
    behavior_cues: list[str],
    secondary_core_lines: list[str],
    general_profile_lines: list[str] | None = None,
    core_additional_lines: list[str] | None = None,
    extra_core_sections: list[str] | None = None,
    extra_context_sections: list[str] | None = None,
    supplemental_additional_lines: list[str] | None = None,
) -> str:
    timeframe_prompt = str(scale_payload["instrument"]["timeframe_prompt"]).strip()
    response_options = scale_payload["response_options"]
    items = scale_payload["items"]
    option_lines = [f'- "{str(option["label"]).strip()}"' for option in response_options]
    item_lines = [f'{int(item["id"])}. {str(item["text"]).strip()}' for item in items]
    item_count = len(items)
    item_id_list = ", ".join(str(int(item["id"])) for item in items)
    sample_response_label = next(
        (
            str(option["label"]).strip()
            for option in response_options
            if int(option["score"]) == 0
        ),
        str(response_options[0]["label"]).strip(),
    )
    sample_output = json.dumps(
        {
            "answers": [
                {"item_id": int(item["id"]), "response": sample_response_label}
                for item in items
            ]
        },
        ensure_ascii=False,
    )
    selected_scope_line = (
        "- Complete all eight PHQ-8 items exactly once."
        if item_count == 8
        else f"- Complete all {item_count} selected PHQ-8 items exactly once."
    )
    core_lines = [
        "# Core Directive",
        "- You are privately completing a PHQ-8 self-report as a client.",
        "- Your answers must remain 100% anchored to the same client role and the behavior cues below.",
        "- The behavior cues below are the highest-priority guidance for how this client has been feeling, functioning, and showing symptoms over the last two weeks.",
        "- Multiple behavior cues may describe the same area of difficulty. Integrate them as one consistent two-week profile, not as separate people or separate episodes.",
        "- You must strictly follow the behavior cues. If any other role wording or context conflicts with the behavior cues, choose the option texts that match the behavior cues.",
    ]
    for line in core_additional_lines or []:
        if line and line.strip():
            core_lines.append(line.strip())
    core_lines.extend(
        [
            "- If a symptom or domain is not supported by the behavior cues, treat it as ordinary baseline functioning rather than inferring extra burden from nearby symptoms or other context.",
            "- Choose the response option text that this client would honestly select for each symptom over the last two weeks.",
        ]
    )
    behavior_cue_lines = behavior_cues or ["none"]
    core_sections = [
        "\n".join(core_lines),
        "## Behavior Cues\n" + "\n".join(f"- {cue_text}" for cue_text in behavior_cue_lines),
    ]
    core_sections.extend(section for section in (extra_core_sections or []) if section and section.strip())

    context_sections = [
        "\n".join(["# Secondary Core", *[line.strip() for line in secondary_core_lines if line and line.strip()]]),
    ]
    if general_profile_lines:
        context_sections.append(
            "## General Profile\n" + "\n".join(f"- {line.strip()}" for line in general_profile_lines if line and line.strip())
        )
    context_sections.extend(section for section in (extra_context_sections or []) if section and section.strip())
    context_sections.extend(
        [
            f"## PHQ-8 Timeframe\n{timeframe_prompt}",
            "## PHQ-8 Items\n" + "\n".join(item_lines),
            "## Response Options For This Pass\n" + "\n".join(option_lines),
        ]
    )
    supplemental_sections = [
        "\n".join(
            [
                "# Supplemental Guardrails",
                "- Conduct all visible output in English only.",
                *[line.strip() for line in supplemental_additional_lines or [] if line and line.strip()],
                "- Base each choice on how the client would answer today, not on what sounds ideal.",
                "- The randomized question order and option order do not change the meaning of the items or options.",
                "- Do not include explanations, rationale, symptom summaries, or extra prose in the JSON values.",
                "- Do not mention hidden state, latent variables, score numbers, or questionnaire strategy.",
                "- Do not output numbers or letter labels.",
            ]
        )
    ]
    output_sections = [
        "\n".join(
            [
                "# Hard Output Requirements",
                "- Return JSON only with one object key named `answers`.",
                selected_scope_line,
                f'- `"answers"` must be a list of {item_count} objects.',
                '- Each answer object must have exactly two keys: `"item_id"` and `"response"`.',
                f'- `"item_id"` must be each of these integers exactly once: {item_id_list}.',
                '- Each `"response"` must be exactly one of the option texts listed in <context_and_data>.',
                "- Do not paraphrase the option texts.",
                f"Return JSON only: {sample_output}",
            ]
        )
    ]
    final_reminder = (
        "# Final Reminder\n"
        "CRITICAL OVERRIDE: Before proceeding, review the <core_directive>.\n"
        "Step 1: Silently check that every answer stays anchored to the same client role and the behavior cues, not to what sounds ideal.\n"
        "Step 2: Silently check that any symptom not supported by the behavior cues remains at the zero-burden option instead of being inferred from nearby symptoms or other context.\n"
        "Step 3: Output the final JSON only."
    )
    return "\n\n".join(
        [
            tagged_prompt_block("core_directive", *core_sections),
            tagged_prompt_block("context_and_data", *context_sections),
            tagged_prompt_block("supplemental_guardrails", *supplemental_sections),
            tagged_prompt_block("output_contract", *output_sections),
            tagged_prompt_block("final_reminder", final_reminder),
        ]
    )


def build_phq8_self_report_retry_prompt(
    *,
    validation_error: Exception | str,
    scale_payload: dict[str, Any],
) -> str:
    option_texts = [f'"{str(option["label"]).strip()}"' for option in scale_payload["response_options"]]
    item_ids = [int(item["id"]) for item in scale_payload["items"]]
    item_id_list = ", ".join(str(item_id) for item_id in item_ids)
    item_count = len(item_ids)
    return (
        "Your previous PHQ-8 self-report output was invalid. "
        f"Validation error: {validation_error}. "
        f"Return exactly one JSON object with an `answers` list of {item_count} objects now. "
        f"Each answer object must use one of these item_id values exactly once: {item_id_list}. "
        "Each answer object must include `item_id` and `response`. "
        "Each response must be exactly one of these option texts: "
        f'{", ".join(option_texts)}. '
        "Do not paraphrase, abbreviate, or invent synonyms. Retry now."
    )


def build_phq8_answers_response_schema(scale_payload: dict[str, Any]) -> dict[str, Any]:
    item_ids = [int(item["id"]) for item in scale_payload["items"]]
    option_labels = [str(option["label"]).strip() for option in scale_payload["response_options"]]
    return {
        "type": "OBJECT",
        "properties": {
            "answers": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        # Vertex Gemini response schemas reject integer enums here, but the
                        # parser already accepts digit strings and coerces them back to ints.
                        "item_id": {"type": "STRING", "enum": [str(item_id) for item_id in item_ids]},
                        "response": {"type": "STRING", "enum": option_labels},
                    },
                    "required": ["item_id", "response"],
                    "propertyOrdering": ["item_id", "response"],
                },
            }
        },
        "required": ["answers"],
        "propertyOrdering": ["answers"],
    }


def _unwrap_embedded_answers_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    for key in _NESTED_PAYLOAD_KEYS:
        if key not in payload:
            continue
        value = payload[key]
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            text = value.strip()
            if not text:
                continue
            try:
                return extract_json_object(text)
            except ValueError:
                continue
    return None


def parse_phq8_answers_response(
    response_text: str,
    *,
    item_id_to_symptom: dict[int, str],
    symptom_order: tuple[str, ...],
    scale_payload: dict[str, Any],
) -> tuple[dict[str, int], dict[str, str]]:
    payload = extract_json_object(response_text)
    raw_answers = payload.get("answers")
    if not isinstance(raw_answers, list):
        raw_answers = payload.get("items")
    if not isinstance(raw_answers, list):
        nested = _unwrap_embedded_answers_payload(payload)
        if isinstance(nested, dict):
            raw_answers = nested.get("answers")
            if not isinstance(raw_answers, list):
                raw_answers = nested.get("items")
    if not isinstance(raw_answers, list):
        raise ValueError("Model output must contain an 'answers' list")
    label_to_score = {
        str(option["label"]).strip().casefold(): int(option["score"])
        for option in scale_payload["response_options"]
    }
    expected_item_ids = set(item_id_to_symptom)
    parsed_scores: dict[str, int] = {}
    parsed_reasons: dict[str, str] = {}
    seen_item_ids: set[int] = set()
    for raw_answer in raw_answers:
        if not isinstance(raw_answer, dict):
            raise ValueError("Each answer must be a JSON object")
        item_id = _coerce_answer_item_id(raw_answer)
        score = _coerce_answer_response(raw_answer, label_to_score=label_to_score)
        reason = _coerce_answer_reason(raw_answer)
        if item_id not in expected_item_ids:
            raise ValueError(f"Unexpected item_id: {item_id}")
        if item_id in seen_item_ids:
            raise ValueError(f"Duplicate item_id: {item_id}")
        seen_item_ids.add(item_id)
        parsed_scores[item_id_to_symptom[item_id]] = score
        if reason:
            parsed_reasons[item_id_to_symptom[item_id]] = reason
    if seen_item_ids != expected_item_ids:
        missing = sorted(expected_item_ids - seen_item_ids)
        raise ValueError(f"Missing item_ids: {missing}")
    return (
        {symptom: parsed_scores[symptom] for symptom in symptom_order},
        {symptom: parsed_reasons.get(symptom, "") for symptom in symptom_order},
    )


def _coerce_answer_item_id(payload: dict[str, Any]) -> int:
    for key in ITEM_ID_KEY_CANDIDATES:
        if key not in payload:
            continue
        return _coerce_int(payload[key], label=key)
    raise ValueError(f"Answer is missing an item id key. Expected one of: {', '.join(ITEM_ID_KEY_CANDIDATES)}")


def _coerce_answer_response(payload: dict[str, Any], *, label_to_score: dict[str, int]) -> int:
    for key in RESPONSE_KEY_CANDIDATES:
        if key not in payload:
            continue
        value = payload[key]
        if not isinstance(value, str):
            raise ValueError(f"{key} must be a string option label, got {value!r}")
        normalized = value.strip().casefold()
        if normalized in label_to_score:
            return label_to_score[normalized]
        raise ValueError(f"{key} must be one of: {', '.join(sorted(label_to_score))}. Got {value!r}")
    raise ValueError(
        "Answer is missing a response key. "
        f"Expected one of: {', '.join(RESPONSE_KEY_CANDIDATES)}"
    )


def _coerce_answer_reason(payload: dict[str, Any]) -> str:
    for key in REASON_KEY_CANDIDATES:
        if key not in payload:
            continue
        value = payload[key]
        if not isinstance(value, str):
            raise ValueError(f"{key} must be a string reason, got {value!r}")
        normalized = _MULTISPACE_RE.sub(" ", value.strip())
        if normalized:
            return normalized
        raise ValueError(f"{key} must not be empty")
    return ""


def _coerce_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer, got bool")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    raise ValueError(f"{label} must be an integer, got {value!r}")
