from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from EmoTrack.client.agent import (
    ClientResponse,
    _PHQ8_ITEMS,
    _build_private_latent_prompt,
    _normalize_hidden_state,
    _remove_leaky_scoring_language,
)
from EmoTrack.conversation.model import OpenAIReasoningClient
from EmoTrack.conversation.message_validation import describe_echoing_message, describe_invalid_visible_message
from EmoTrack.conversation.realcbt_calibration.surface_control import (
    TurnFormPlan,
    build_realcbt_turn_form_plan,
    format_turn_form_block,
)
from EmoTrack.utils.dialogue_text import (
    build_chat_history_messages as _build_chat_history_messages,
    build_vertex_chat_history_messages as _build_vertex_chat_history_messages,
    history_speaker_label as _history_speaker_label,
    normalize_space as _normalize_space,
    phase_lengths as _phase_lengths,
    short_phrase_list as _short_phrase_list,
)
from EmoTrack.utils.json_utils import extract_json_object
from EmoTrack.utils.phq8_self_report import (
    PHQ8_SELF_REPORT_USER_PROMPT,
    build_phq8_answers_response_schema,
    build_phq8_self_report_prompt,
    build_phq8_self_report_retry_prompt,
    item_id_to_symptom_map,
    load_phq8_scale_payload,
    parse_phq8_answers_response,
)
from EmoTrack.latent_variable.trend import LatentTrendPlan, ResolvedLatentState


def _profile_text(profile_data: dict[str, Any], key: str) -> str:
    value = profile_data.get(key)
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _join_text_parts(parts: list[str]) -> str:
    return "\n".join(part.strip() for part in parts if part and part.strip()).strip()


def _profile_mapping(profile_data: dict[str, Any], key: str) -> dict[str, Any]:
    value = profile_data.get(key, {})
    return value if isinstance(value, dict) else {}


def _profile_value_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
        if items:
            return ", ".join(items)
    if isinstance(value, dict):
        compact = {str(k): v for k, v in value.items() if v not in (None, "", [], {})}
        if compact:
            return json.dumps(compact, ensure_ascii=False, sort_keys=True)
    return str(value).strip()


def _profile_natural_language_block(title: str, mapping: dict[str, Any], intro: str) -> str:
    if not mapping:
        return ""
    _ = title
    lines = [intro]
    for key, value in mapping.items():
        rendered = _profile_value_text(value)
        if not rendered:
            continue
        label = key.replace("_", " ").strip()
        lines.append(f"- {label}: {rendered}")
    return "\n".join(lines).strip()


def _tagged_prompt_block(tag: str, *sections: str) -> str:
    content = "\n\n".join(section.strip() for section in sections if section and section.strip())
    return f"<{tag}>\n{content}\n</{tag}>"


def _closing_window(total_turns: int) -> tuple[int, int]:
    total = max(1, int(total_turns))
    window_turns = max(1, (total + 9) // 10)
    start_turn = max(1, total - window_turns + 1)
    return window_turns, start_turn


def _parse_client_plain_text(text: str) -> tuple[str, str]:
    """Parse plain-text client output with an optional Emotion header.

    JSON-shaped output is treated as invalid for this client path."""
    raw = (text or "").strip()
    if not raw:
        return "", ""
    if raw.lstrip().startswith("{") or raw.lstrip().startswith("["):
        return "", ""

    field_pattern = re.compile(r"^\s*(emotion|complaint)\s*[:：]\s*", flags=re.I)
    role_prefix_pattern = re.compile(r"^\s*(client|counselor)\s*[:：]\s*", flags=re.I)
    emotion = ""
    message_lines: list[str] = []
    for line in raw.split("\n"):
        line = line.rstrip()
        match = field_pattern.match(line)
        if match:
            value = line[match.end() :].strip()
            key = match.group(1).lower()
            if key == "emotion" and value:
                emotion = value
            elif key != "complaint":
                message_lines.append(line)
            continue

        line = role_prefix_pattern.sub("", line).strip()
        if line:
            message_lines.append(line)
    message = "\n".join(message_lines).strip()
    return message, emotion


_CANONICAL_SELF_REPORT_SCALE_PAYLOAD = load_phq8_scale_payload()
_SELF_REPORT_CANONICAL_OPTION_TEXTS = tuple(
    str(option["label"]).strip() for option in _CANONICAL_SELF_REPORT_SCALE_PAYLOAD["response_options"]
)

_CLIENT_STYLE_MODIFIER_GUIDANCE = {
    "plain": "",
    "upset": (
        "In some turns, sound irritated, resistant, or skeptical toward the counselor or the process, "
        "but keep it natural and not constant."
    ),
    "verbose": (
        "In some turns, give fuller and more detailed answers, but stay responsive to the counselor's latest point."
    ),
    "reserved": (
        "In some turns, be guarded, brief, vague, or slow to disclose, and require more prompting before opening up."
    ),
    "tangent": (
        "In some turns, drift into a nearby but less relevant detail or anecdote and require gentle redirection, "
        "but do not derail every turn."
    ),
    "pleasing": (
        "In some turns, minimize your distress, agree too quickly, or seek approval even when your real feelings are mixed."
    ),
}


def _profile_style_modifiers(profile_data: dict[str, Any]) -> list[str]:
    def collect(raw: Any) -> list[str]:
        if isinstance(raw, str):
            chunks = re.split(r"[,;\n]", raw)
            return [chunk.strip().lower() for chunk in chunks if chunk and chunk.strip()]
        if isinstance(raw, list):
            return [str(item).strip().lower() for item in raw if str(item).strip()]
        return []

    candidates: list[str] = []
    for key in ("patient_style_modifiers", "patient_style_modifier", "patient_styles", "patient_style"):
        candidates.extend(collect(profile_data.get(key)))

    meta = profile_data.get("meta", {})
    if isinstance(meta, dict):
        for key in ("patient_style_modifiers", "patient_style_modifier", "patient_styles", "patient_style"):
            candidates.extend(collect(meta.get(key)))

    normalized: list[str] = []
    for item in candidates:
        if item not in _CLIENT_STYLE_MODIFIER_GUIDANCE or item in normalized:
            continue
        normalized.append(item)
    return normalized


def _style_modifier_block(modifiers: list[str]) -> str:
    if not modifiers:
        return ""
    lines = [
        "[Optional patient-style modifiers]",
        "Treat these as soft tendencies, not a caricature.",
        "They can appear in some turns and fade in others as trust and context change.",
        "Do not let a style modifier override the main presenting problem, profile facts, or the counselor's latest question.",
    ]
    for modifier in modifiers:
        guidance = _CLIENT_STYLE_MODIFIER_GUIDANCE.get(modifier, "")
        if guidance:
            lines.append(f"- {modifier}: {guidance}")
    return "\n".join(lines)


def _stable_self_report_order(
    values: tuple[str, ...],
    *,
    kind: str,
    session_id: int,
    sample_index: int,
    complaint: str | None = None,
) -> tuple[str, ...]:
    if sample_index <= 0:
        raise ValueError("sample_index must be >= 1")
    return tuple(
        sorted(
            values,
            key=lambda value: hashlib.sha256(
                (
                    f"self-report|kind={kind}|session={session_id}|sample={sample_index}|"
                    f"complaint={complaint or ''}|value={value}"
                ).encode("utf-8")
            ).hexdigest(),
        )
    )


def _self_report_symptom_order(
    *,
    session_id: int,
    sample_index: int,
    complaint: str | None = None,
) -> tuple[str, ...]:
    return _stable_self_report_order(
        tuple(_PHQ8_ITEMS),
        kind="items",
        session_id=session_id,
        sample_index=sample_index,
        complaint=complaint,
    )


def _self_report_option_order(
    *,
    session_id: int,
    sample_index: int,
    complaint: str | None = None,
) -> tuple[str, str, str, str]:
    return _stable_self_report_order(
        _SELF_REPORT_CANONICAL_OPTION_TEXTS,
        kind="options",
        session_id=session_id,
        sample_index=sample_index,
        complaint=complaint,
    )


def _self_report_scale_payload(
    *,
    session_id: int,
    sample_index: int,
    complaint: str | None = None,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    symptom_order = _self_report_symptom_order(
        session_id=session_id,
        sample_index=sample_index,
        complaint=complaint,
    )
    option_order = _self_report_option_order(
        session_id=session_id,
        sample_index=sample_index,
        complaint=complaint,
    )
    item_by_symptom = {
        symptom_name: dict(item)
        for symptom_name, item in zip(
            _PHQ8_ITEMS,
            _CANONICAL_SELF_REPORT_SCALE_PAYLOAD["items"],
            strict=True,
        )
    }
    option_by_label = {
        str(option["label"]).strip(): dict(option)
        for option in _CANONICAL_SELF_REPORT_SCALE_PAYLOAD["response_options"]
    }
    return (
        {
            **_CANONICAL_SELF_REPORT_SCALE_PAYLOAD,
            "items": [item_by_symptom[symptom_name] for symptom_name in symptom_order],
            "response_options": [option_by_label[label] for label in option_order],
        },
        symptom_order,
    )


def _self_report_behavior_cues(hidden_state: dict[str, Any]) -> list[str]:
    block = _build_private_latent_prompt(hidden_state)
    if not block:
        return []
    return [line[2:].strip() for line in block.splitlines() if line.strip().startswith("- ")]


@dataclass(frozen=True)
class _SessionComplaintPlan:
    session_number: int
    theme: str
    beats: tuple["_ComplaintPhase", ...]


@dataclass(frozen=True)
class _ComplaintPhase:
    title: str
    focus: str
    anchors: tuple[str, ...]


def _dedupe_texts(items: list[str]) -> list[str]:
    deduped: list[str] = []
    for item in items:
        text = str(item or "").strip()
        if not text or text in deduped:
            continue
        deduped.append(text)
    return deduped


def _text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        items: list[str] = []
        for item in value:
            items.extend(_text_list(item))
        return items
    if isinstance(value, dict):
        rendered = _profile_value_text(value)
        return [rendered] if rendered else []
    rendered = str(value).strip()
    return [rendered] if rendered else []


def _target_behavior_entries(profile_data: dict[str, Any]) -> list[dict[str, Any]]:
    client_info = _profile_mapping(profile_data, "client_info")
    raw = client_info.get("target_behavior")
    if isinstance(raw, dict):
        return [raw]
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    return []


_SESSION_KEY_PATTERN = re.compile(r"第\s*(\d+)\s*次")
_CLIENT_MASTER_BEAT_BLUEPRINTS = (
    (
        "Concrete Event Or Story",
        "Stay close to one concrete story, scene, or recent event from this session's source plan instead of speaking only in global feelings.",
    ),
    (
        "Trigger Pattern And Body Alarm",
        "Describe the trigger pattern, body alarm, bracing, and what starts happening before or during the difficult moment.",
    ),
    (
        "Feared Meaning Or Prediction",
        "Expose the feared meaning, catastrophic prediction, or vulnerable interpretation underneath the event.",
    ),
    (
        "Avoidance, Cost, And What Gets Smaller",
        "Show what this pattern costs in daily life, work, routines, relationships, or freedom of movement.",
    ),
    (
        "Desired Help And Next Safe Step",
        "Clarify what kind of help feels useful now, what still blocks change, and what next step would feel barely possible.",
    ),
)

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_source_profile(profile_data: dict[str, Any]) -> dict[str, Any]:
    meta = profile_data.get("meta", {})
    if not isinstance(meta, dict):
        return {}
    raw_path = str(meta.get("source_profile_path") or "").strip()
    if not raw_path:
        return {}

    source_path = Path(raw_path).expanduser()
    if not source_path.is_absolute():
        source_path = (_repo_root() / source_path).resolve()
    try:
        return json.loads(source_path.read_text())
    except Exception:
        return {}


def _chunk_evenly(items: list[str], chunk_count: int) -> list[list[str]]:
    if chunk_count <= 0:
        return []
    cleaned = _dedupe_texts(items)
    if not cleaned:
        return [[] for _ in range(chunk_count)]
    base = len(cleaned) // chunk_count
    remainder = len(cleaned) % chunk_count
    chunks: list[list[str]] = []
    cursor = 0
    for idx in range(chunk_count):
        size = base + (1 if idx < remainder else 0)
        if size <= 0:
            chunks.append([])
            continue
        chunks.append(cleaned[cursor : cursor + size])
        cursor += size
    return chunks


def _source_session_blueprints(source_profile: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(source_profile, dict):
        return []

    merged: dict[int, dict[str, Any]] = {}

    global_plan = source_profile.get("global_plan", [])
    if isinstance(global_plan, list):
        for stage in global_plan:
            if not isinstance(stage, dict):
                continue
            content = stage.get("content", {})
            if not isinstance(content, dict):
                continue
            for session_key, entry in content.items():
                if not isinstance(entry, dict):
                    continue
                match = _SESSION_KEY_PATTERN.search(str(session_key))
                if match is None:
                    continue
                session_number = int(match.group(1))
                target = merged.setdefault(
                    session_number,
                    {
                        "session_number": session_number,
                        "theme": "",
                        "persona_links": [],
                        "case_material": [],
                        "rationale": [],
                        "session_goals": [],
                    },
                )
                if not target["theme"]:
                    target["theme"] = str(entry.get("theme") or "").strip()
                target["persona_links"].extend(_text_list(entry.get("persona_links")))
                target["case_material"].extend(_text_list(entry.get("case_material")))
                target["rationale"].extend(_text_list(entry.get("rationale")))

    sessions = source_profile.get("sessions", [])
    if isinstance(sessions, list):
        for idx, session_entry in enumerate(sessions, start=1):
            if not isinstance(session_entry, dict):
                continue
            raw_number = session_entry.get("session_number")
            if isinstance(raw_number, int):
                session_number = raw_number
            else:
                try:
                    session_number = int(str(raw_number).strip())
                except Exception:
                    session_number = idx
            target = merged.setdefault(
                session_number,
                {
                    "session_number": session_number,
                    "theme": "",
                    "persona_links": [],
                    "case_material": [],
                    "rationale": [],
                    "session_goals": [],
                },
            )
            target["session_goals"].extend(_text_list(session_entry.get("session_goals")))
            summary = session_entry.get("session_summary")
            if isinstance(summary, dict):
                for value in summary.values():
                    target["rationale"].extend(_text_list(value))

    return [merged[key] for key in sorted(merged)]


def _stored_session_master_plans(profile_data: dict[str, Any]) -> list[_SessionComplaintPlan]:
    raw_plans = profile_data.get("client_session_master_plan", [])
    if not isinstance(raw_plans, list):
        return []

    plans: list[_SessionComplaintPlan] = []
    for plan in raw_plans:
        if not isinstance(plan, dict):
            continue
        try:
            session_number = int(plan.get("session_number", 0))
        except Exception:
            continue
        if session_number <= 0:
            continue
        theme = str(plan.get("theme") or "").strip() or f"Session {session_number}"
        beats_raw = plan.get("beats", [])
        if not isinstance(beats_raw, list):
            continue
        beats: list[_ComplaintPhase] = []
        for beat in beats_raw:
            if not isinstance(beat, dict):
                continue
            title = str(beat.get("title") or "").strip()
            focus = str(beat.get("focus") or "").strip()
            anchors = tuple(_dedupe_texts(_text_list(beat.get("anchors"))))
            if not title or not focus:
                continue
            beats.append(_ComplaintPhase(title=title, focus=focus, anchors=anchors))
        if beats:
            plans.append(_SessionComplaintPlan(session_number=session_number, theme=theme, beats=tuple(beats)))
    return plans


def _latent_color_notes(hidden_state: dict[str, Any]) -> list[str]:
    try:
        normalized = _normalize_hidden_state(hidden_state)
    except Exception:
        return []
    scores = normalized.get("phq8_scores", {})
    notes: list[str] = []
    if scores.get("anhedonia", 0.0) >= 2.0:
        notes.append("loss of ease or enjoyment may color how you talk about what life used to feel like")
    if scores.get("sleep", 0.0) >= 2.0 or scores.get("fatigue", 0.0) >= 2.0:
        notes.append("low energy or worn-down pacing may show up subtly")
    if scores.get("concentration", 0.0) >= 2.0:
        notes.append("your attention may narrow quickly around threat or pain")
    if scores.get("self_worth", 0.0) >= 2.0:
        notes.append("embarrassment or self-doubt may surface when talking about other people")
    if scores.get("depressed_mood", 0.0) >= 2.0:
        notes.append("discouragement may color your expectations for recovery")
    if scores.get("psychomotor", 0.0) >= 2.0:
        notes.append("your tone may feel slowed, tense, or effortful")
    return notes[:3]


_PROFILE_PLACEHOLDERS = {
    "",
    "none",
    "n/a",
    "null",
    "unknown",
    "unspecified",
    "未提及",
    "未知",
    "无",
    "暂无",
}


def _meaningful_profile_text(value: Any) -> str:
    text = _normalize_space(value)
    if not text:
        return ""
    if text.lower() in _PROFILE_PLACEHOLDERS or text in _PROFILE_PLACEHOLDERS:
        return ""
    return text


def _stable_identity_line(profile_data: dict[str, Any]) -> str:
    client_info = _profile_mapping(profile_data, "client_info")
    intake_profile = _profile_mapping(profile_data, "intake_profile")
    static_traits = client_info.get("static_traits")
    if not isinstance(static_traits, dict):
        static_traits = intake_profile.get("static_traits")
    if not isinstance(static_traits, dict):
        return ""

    parts: list[str] = []
    for key in ("name", "age", "gender", "occupation"):
        value = _meaningful_profile_text(static_traits.get(key))
        if value:
            parts.append(f"{key}={value}")
    return ", ".join(parts)


def _stable_client_background_block(
    profile_data: dict[str, Any],
    *,
    presenting_problem: str,
) -> str:
    lines: list[str] = []

    identity_line = _stable_identity_line(profile_data)
    if identity_line:
        lines.append(f"- Stable identity: {identity_line}")

    main_problem = _meaningful_profile_text(presenting_problem)
    if main_problem:
        lines.append(f"- Main presenting problem: {main_problem}")

    client_info = _profile_mapping(profile_data, "client_info")
    intake_profile = _profile_mapping(profile_data, "intake_profile")
    profile_problem = _meaningful_profile_text(client_info.get("main_problem") or intake_profile.get("main_problem"))
    if profile_problem and profile_problem != main_problem:
        lines.append(f"- Profile problem context: {profile_problem}")
    growth = _dedupe_texts(_text_list(client_info.get("growth_experiences")))
    if growth:
        lines.append(f"- Key context: {growth[0]}")

    behavior_entries = _target_behavior_entries(profile_data)
    triggers: list[str] = []
    beliefs: list[str] = []
    costs: list[str] = []
    for entry in behavior_entries:
        triggers.extend(_text_list(entry.get("antecedent")))
        beliefs.extend(_text_list(entry.get("core_reason")))
        costs.extend(_text_list(entry.get("consequence")))

    trigger_text = ", ".join(_dedupe_texts(triggers)[:3])
    if trigger_text:
        lines.append(f"- Common trigger contexts: {trigger_text}")

    cost_text = _dedupe_texts(costs)[:1]
    if cost_text:
        lines.append(f"- Main cost: {cost_text[0]}")
    else:
        belief_text = _dedupe_texts(beliefs)[:1]
        if belief_text:
            lines.append(f"- Main fear or belief: {belief_text[0]}")

    topic = _meaningful_profile_text(client_info.get("topic") or intake_profile.get("topic"))
    if topic and len(lines) <= 2:
        lines.append(f"- Main topic: {topic}")

    if not lines:
        prompt_raw = _profile_text(profile_data, "seeker_prompt")
        if prompt_raw:
            lines.append(prompt_raw)
        elif main_problem:
            lines.append(f"- Main presenting problem: {main_problem}")
        else:
            lines.append("- Background details are minimal; stay consistent with the ongoing conversation.")
    return "\n".join(lines)


class EverMemClientAgent:
    MAX_RESPONSE_ATTEMPTS = 5
    MAX_SELF_REPORT_ATTEMPTS = 3
    SELF_REPORT_SAMPLE_COUNT = 5

    def __init__(
        self,
        *,
        generator: OpenAIReasoningClient,
        profile_data: dict[str, Any],
        hidden_state: dict[str, Any],
        trend_plan: LatentTrendPlan | None = None,
        run_seed: int = 0,
        invalid_output_logger: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.generator = generator
        self.profile_data = profile_data
        self.hidden_state = hidden_state
        self.trend_plan = trend_plan or LatentTrendPlan.empty()
        self.run_seed = int(run_seed)
        self.invalid_output_logger = invalid_output_logger
        self._stored_master_plans = _stored_session_master_plans(profile_data)
        self._source_profile = _load_source_profile(profile_data)

    def set_invalid_output_logger(
        self,
        logger: Callable[[dict[str, Any]], None] | None,
    ) -> None:
        self.invalid_output_logger = logger

    def _log_invalid_output(self, payload: dict[str, Any]) -> None:
        if self.invalid_output_logger is None:
            return
        self.invalid_output_logger(payload)

    def resolve_latent_state(self, session_id: int) -> ResolvedLatentState:
        return self.trend_plan.resolve(self.hidden_state, session_id)

    def presenting_problem(self) -> str:
        client_info = _profile_mapping(self.profile_data, "client_info")
        intake_profile = _profile_mapping(self.profile_data, "intake_profile")
        main_problem = _profile_text(client_info, "main_problem")
        if main_problem:
            return main_problem
        intake_problem = _profile_text(intake_profile, "main_problem")
        if intake_problem:
            return intake_problem

        report = self.profile_data.get("report", {})
        portrait = self.profile_data.get("portrait", {})
        if isinstance(report, dict):
            main_problem = _profile_text(report, "main_problem")
            if main_problem:
                return main_problem
        if isinstance(portrait, dict):
            symptoms = _profile_text(portrait, "symptoms")
            if symptoms:
                return symptoms
        return "unspecified"

    def _complaint_chain(self, resolved_latent_state: ResolvedLatentState) -> list[_ComplaintPhase]:
        client_info = _profile_mapping(self.profile_data, "client_info")
        baseline = self.presenting_problem()
        growth = _text_list(client_info.get("growth_experiences"))
        topic = _profile_text(client_info, "topic")
        behavior_entries = _target_behavior_entries(self.profile_data)

        triggers: list[str] = []
        beliefs: list[str] = []
        functions: list[str] = []
        consequences: list[str] = []
        behaviors: list[str] = []
        for entry in behavior_entries:
            behaviors.extend(_text_list(entry.get("behavior")))
            triggers.extend(_text_list(entry.get("antecedent")))
            beliefs.extend(_text_list(entry.get("core_reason")))
            functions.extend(_text_list(entry.get("function")))
            consequences.extend(_text_list(entry.get("consequence")))

        triggers = _dedupe_texts(growth + triggers)
        beliefs = _dedupe_texts(beliefs)
        functions = _dedupe_texts(functions)
        consequences = _dedupe_texts(consequences)
        behaviors = _dedupe_texts(behaviors)
        latent_notes = _latent_color_notes(resolved_latent_state.hidden_state)

        phases = [
            _ComplaintPhase(
                title="Recent Trigger Scene",
                focus=(
                    "Stay close to one recent concrete sound-triggered situation, how the problem started, "
                    "and what made this issue feel urgent enough to seek help."
                ),
                anchors=tuple(_dedupe_texts([baseline, *triggers])[:4]),
            ),
            _ComplaintPhase(
                title="Body Alarm And Anticipation",
                focus=(
                    "Describe the body alarm, bracing, and anticipatory fear that starts before or during "
                    "entering a place where sounds might hurt."
                ),
                anchors=tuple(_dedupe_texts([*behaviors, *triggers, *latent_notes])[:4]),
            ),
            _ComplaintPhase(
                title="Feared Meaning",
                focus=(
                    "Expose the feared meaning underneath the pain and anxiety: what you think it means "
                    "about safety, recovery, embarrassment, or the future."
                ),
                anchors=tuple(_dedupe_texts([*beliefs, *latent_notes])[:4]),
            ),
            _ComplaintPhase(
                title="Avoidance And Cost",
                focus=(
                    "Show how avoidance affects daily functioning, work, going out, relationships, "
                    "and what feels lost or disrupted."
                ),
                anchors=tuple(_dedupe_texts([*functions, *consequences, topic, *latent_notes])[:4]),
            ),
            _ComplaintPhase(
                title="Desired Help And Next Safe Step",
                focus=(
                    "Clarify what kind of help feels useful now, what would make the next step feel safer, "
                    "and what realistic experiment still feels barely possible."
                ),
                anchors=tuple(_dedupe_texts([topic, *consequences, *behaviors, *latent_notes])[:4]),
            ),
        ]
        return phases

    def _session_master_plans(self, resolved_latent_state: ResolvedLatentState) -> list[_SessionComplaintPlan]:
        latent_notes = _latent_color_notes(resolved_latent_state.hidden_state)
        if self._stored_master_plans:
            plans: list[_SessionComplaintPlan] = []
            for plan in self._stored_master_plans:
                beats = []
                for beat in plan.beats:
                    anchors = tuple(_dedupe_texts([*beat.anchors, *latent_notes[:1]])[:5])
                    beats.append(_ComplaintPhase(title=beat.title, focus=beat.focus, anchors=anchors))
                plans.append(_SessionComplaintPlan(session_number=plan.session_number, theme=plan.theme, beats=tuple(beats)))
            return plans

        session_blueprints = _source_session_blueprints(self._source_profile)
        if not session_blueprints:
            return []

        fallback_phases = self._complaint_chain(resolved_latent_state)
        plans: list[_SessionComplaintPlan] = []
        for entry in session_blueprints:
            session_number = int(entry.get("session_number", 0) or 0)
            if session_number <= 0:
                continue
            theme = str(entry.get("theme") or "").strip() or f"Session {session_number}"
            source_items = _dedupe_texts(
                [
                    theme,
                    *_text_list(entry.get("session_goals")),
                    *_text_list(entry.get("case_material")),
                    *_text_list(entry.get("persona_links")),
                    *_text_list(entry.get("rationale")),
                ]
            )
            source_chunks = _chunk_evenly(source_items, len(_CLIENT_MASTER_BEAT_BLUEPRINTS))
            beats: list[_ComplaintPhase] = []
            for idx, (title, focus) in enumerate(_CLIENT_MASTER_BEAT_BLUEPRINTS):
                source_chunk = source_chunks[idx] if idx < len(source_chunks) else []
                fallback_phase = fallback_phases[idx] if idx < len(fallback_phases) else None
                anchors = _dedupe_texts(
                    [
                        *source_chunk,
                        theme,
                        *(list(fallback_phase.anchors) if fallback_phase is not None else []),
                        *latent_notes[:1],
                    ]
                )[:5]
                beats.append(
                    _ComplaintPhase(
                        title=title,
                        focus=f"{focus} Keep it aligned with the current session theme: {theme}.",
                        anchors=tuple(anchors),
                    )
                )
            plans.append(
                _SessionComplaintPlan(
                    session_number=session_number,
                    theme=theme,
                    beats=tuple(beats),
                )
            )
        return plans

    def _complaint_phase_block(
        self,
        *,
        session_id: int,
        turn_id: int,
        target_turns: int | None,
        resolved_latent_state: ResolvedLatentState,
    ) -> str:
        total_turns = max(1, int(target_turns or 20))
        master_plans = self._session_master_plans(resolved_latent_state)
        if master_plans:
            session_index = min(max(1, int(session_id)), len(master_plans)) - 1
            current_plan = master_plans[session_index]
            phases = list(current_plan.beats)
        else:
            current_plan = None
            phases = self._complaint_chain(resolved_latent_state)
        if not phases:
            return ""

        lengths = _phase_lengths(total_turns, len(phases))
        phase_index = 0
        turn_cursor = 0
        for idx, length in enumerate(lengths):
            turn_cursor += length
            if turn_id <= turn_cursor:
                phase_index = idx
                break
        prior_turns = sum(lengths[:phase_index])
        phase_turn_index = max(1, turn_id - prior_turns)
        phase_turn_budget = lengths[phase_index]
        current_phase = phases[phase_index]
        next_phase = phases[phase_index + 1] if phase_index + 1 < len(phases) else None

        chain_lines = [
            "### Hidden Session Beat Plan",
            f"Session {session_id} is organized into {len(phases)} private topic beats.",
            "This is private scaffolding for pacing, not a hard script; never mention it explicitly.",
            "Reply to the live exchange naturally; use this beat plan only as private pacing context.",
        ]
        if current_plan is not None:
            chain_lines.extend(
                [
                    f"Global hidden session plan spans {len(master_plans)} sessions.",
                    "Each session normally has 5 private topic beats, and each beat normally lasts about 4 client turns.",
                    f"Current session theme: {current_plan.theme}",
                ]
            )
        chain_lines.extend(
            [
            f"Current beat: {phase_index + 1}/{len(phases)} - {current_phase.title}",
            f"Current beat turn: {phase_turn_index}/{phase_turn_budget}",
            f"Current beat focus: {current_phase.focus}",
            ]
        )
        if current_phase.anchors:
            chain_lines.append("Current beat anchors:")
            for anchor in current_phase.anchors:
                chain_lines.append(f"- {anchor}")
        if phase_turn_index < phase_turn_budget:
            chain_lines.append(
                "Progression rule for this turn: if this beat fits the live exchange, stay loosely near it. If it does not fit, ignore it and stay with the live exchange."
            )
        elif next_phase is not None:
            chain_lines.append(
                "Progression rule for this turn: if it fits naturally, you may lightly touch the next beat, but do not force a transition."
            )
            chain_lines.append(f"Next beat: {next_phase.title}")
            chain_lines.append(f"Next beat focus: {next_phase.focus}")
        else:
            chain_lines.append(
                "Progression rule for this turn: stay with what feels most alive in this beat. Do not force a next step."
            )
        chain_lines.append(
            "If following the beat would force repetition, self-analysis, planning language, or premature closure, stay in character and follow the live exchange instead."
        )
        return "\n".join(chain_lines)

    @staticmethod
    def _turn_phase(turn_id: int, target_turns: int | None) -> tuple[str, str]:
        total_turns = max(1, int(target_turns or 20))
        if turn_id <= 1:
            return "opening", "State the main problem, the concrete context, and what feels most urgent right now."
        if turn_id >= total_turns:
            return "closing", "Stay natural at the end of the session: respond to the counselor, name what still feels most present, and let any next-step talk stay optional and brief."

        progress = turn_id / total_turns
        if progress <= 0.25:
            return "concretizing", "Add concrete examples, triggers, body reactions, recent incidents, or impact on daily functioning."
        if progress <= 0.55:
            return "meaning-making", "Clarify beliefs, interpretations, fears, patterns, or what feels stuck beneath the surface."
        if progress <= 0.80:
            return "change-work", "React to the counselor's ideas with what fits, what does not, and what practical obstacle shows up."
        return "consolidation", "Focus on what feels different, what remains hard, and what would make the next step feel doable."

    @staticmethod
    def _recent_speaker_turns(
        session_history: list[dict[str, str]],
        *,
        speaker: str,
        limit: int,
    ) -> list[str]:
        speaker_key = speaker.strip().lower()
        matches = [
            str(item.get("text", "")).strip()
            for item in session_history
            if _history_speaker_label(str(item.get("speaker", ""))) == speaker_key
            and str(item.get("text", "")).strip()
        ]
        return matches[-limit:]

    @classmethod
    def _recent_turns_guardrail(
        cls,
        session_history: list[dict[str, str]],
        *,
        speaker: str,
        title: str,
        limit: int = 2,
    ) -> str:
        recent_turns = cls._recent_speaker_turns(session_history, speaker=speaker, limit=limit)
        phrases = _short_phrase_list(recent_turns)
        if not phrases:
            return ""
        lines = [title]
        for phrase in phrases:
            lines.append(f'- "{phrase}"')
        return "\n".join(lines)

    @staticmethod
    def _opening_alignment_note() -> str:
        return (
            "### Private Opening Alignment\n"
            "- On the first client reply of each session, combine stable objective profile facts with the current session latent guidance.\n"
            "- Keep identity, biography, chronology, setting, relationships, and concrete events stable when they are factual.\n"
            "- If source profile wording about symptoms, mood, energy, sleep, motivation, or subjective interpretation conflicts with the latent guidance, rewrite those subjective parts to match the latent guidance.\n"
            "- Do not preserve source-case emotional wording when it conflicts with the latent guidance."
        )

    @staticmethod
    def _session_latent_clue(
        resolved_latent_state: ResolvedLatentState,
    ) -> str:
        latent_prompt = _build_private_latent_prompt(resolved_latent_state.hidden_state)
        if not latent_prompt:
            return ""
        return (
            latent_prompt
            + "\nUse this latent plan only as private background guidance."
            + "\nTreat the cues as the client's internal state, appraisal lens, and avoidance pressure, not as a checklist of things to say aloud."
            + "\nIf a cue sounds like something the client 'says' or 'describes', treat it as material that may surface when naturally triggered or directly asked, not as wording that must be stated in this turn."
            + "\nDefault to partial disclosure: let the cues shape attention, omission, hesitation, body reactions, and concrete details before explicit confession."
        )

    def _turn_surface_plan(
        self,
        *,
        counselor_message: str,
        session_id: int,
        turn_id: int,
        target_turns: int | None,
        session_history: list[dict[str, str]],
    ) -> TurnFormPlan:
        return build_realcbt_turn_form_plan(
            speaker="client",
            session_id=session_id,
            turn_id=turn_id,
            target_turns=target_turns,
            shared_source_text=counselor_message,
            run_seed=self.run_seed,
            prev_utterance_text=counselor_message,
        )

    @classmethod
    def _describe_invalid_client_message(
        cls,
        message_value: Any,
        *,
        counselor_message: str,
        session_history: list[dict[str, str]],
    ) -> str | None:
        invalid_reason = describe_invalid_visible_message(message_value)
        if invalid_reason is not None:
            return invalid_reason

        message = str(message_value).strip()
        echo_references = [counselor_message] + cls._recent_speaker_turns(
            session_history,
            speaker="client",
            limit=2,
        )
        echo_reason = describe_echoing_message(message, echo_references)
        if echo_reason is not None:
            return echo_reason
        return None

    @staticmethod
    def _invalid_client_retry_guidance(invalid_reason: str) -> str:
        normalized = str(invalid_reason or "").strip().lower()
        if "sounds like the counselor" in normalized:
            return (
                "Your last draft sounded like the counselor instead of the client. "
                "Retry as the client only: speak from your own immediate feelings, reactions, uncertainty, resistance, or concrete situation. "
                "Do not open with reflective-validation phrases like 'I hear how', 'it sounds like', 'that sounds', or similar counselor language. "
                "Do not reassure, summarize, interpret, guide, or coach the counselor."
            )
        if "echoes prior wording too closely" in normalized:
            return (
                "Your last draft reused prior wording too closely. "
                "Retry with clearly different wording, sentence structure, and semantic angle. "
                "Answer the counselor's meaning, not their phrasing. "
                "Do not recycle the same semantic point from the counselor's latest message, your own last 2-3 client turns, or the invalid draft with only light rewording. "
                "If the same point was already expressed, switch to a new concrete detail, reaction, obstacle, fear, want, correction, or uncertainty instead of restating it. "
                "If this turn naturally supports a brief or guarded reply, keep it short instead of paraphrasing earlier lines."
            )
        if "blocked scoring/latent-label language" in normalized:
            return (
                "Your last draft exposed score, questionnaire, or hidden-label language instead of natural client speech. "
                "Retry by translating that material into lived experience: what the client feels in their body, avoids, notices, or struggles with in daily life. "
                "Do not mention PHQ, scores, scales, questionnaires, or latent labels."
            )
        if "emotion is empty" in normalized:
            return (
                "Your last draft omitted the emotion field. "
                "Keep `emotion` as a short feeling phrase such as tense, flat, ashamed, scared, or relieved."
            )
        if "message is empty" in normalized:
            return (
                "Your last draft did not produce a usable client reply. "
                "Retry with at least one natural first-person line that answers the counselor in your own words."
            )
        if "message must be english only" in normalized:
            return (
                "Your last draft included non-English text. "
                "Rewrite the visible reply fully in English and translate any source wording silently."
            )
        if "message must be natural language, not just numbers" in normalized or "message must include alphabetic english words" in normalized:
            return (
                "Your last draft was not natural conversational language. "
                "Retry with an English first-person reply, not numbers, labels, or symbol-only fragments."
            )
        if "message must be a json string" in normalized:
            return (
                "Your last draft used the wrong type for `message`. "
                "Keep `message` as normal first-person text and keep `emotion` as a short text phrase."
            )
        return (
            "Your last draft did not satisfy the client-message constraints. "
            "Retry as a natural first-person client reply with fresh wording. "
            "A short natural clause is acceptable, but avoid a bare filler token."
        )

    def build_prompt_messages(
        self,
        *,
        counselor_message: str,
        session_id: int,
        turn_id: int,
        target_turns: int | None = None,
        session_history: list[dict[str, str]],
        retrieved_memories: list[str],
        short_term_note: str,
        resolved_latent_state: ResolvedLatentState,
    ) -> list[dict[str, str]]:
        style_modifiers = _profile_style_modifiers(self.profile_data)
        stable_background_block = _stable_client_background_block(
            self.profile_data,
            presenting_problem=self.presenting_problem(),
        )
        turn_form_plan = self._turn_surface_plan(
            counselor_message=counselor_message,
            session_id=session_id,
            turn_id=turn_id,
            target_turns=target_turns,
            session_history=session_history,
        )
        turn_form_block = format_turn_form_block(turn_form_plan, counterpart_label="counselor's")

        opening_alignment_block = ""
        if turn_id == 1:
            opening_alignment_block = self._opening_alignment_note()
        behavior_cue_block = self._session_latent_clue(resolved_latent_state)
        complaint_phase_block = self._complaint_phase_block(
            session_id=session_id,
            turn_id=turn_id,
            target_turns=target_turns,
            resolved_latent_state=resolved_latent_state,
        )

        memory_text = "\n".join(f"- {item}" for item in retrieved_memories) if retrieved_memories else "- none"
        recent_client_guardrail = self._recent_turns_guardrail(
            session_history,
            speaker="client",
            title="Recent client phrases to avoid mirroring",
        )
        recent_client_guardrail_text = recent_client_guardrail or "Recent client phrases to avoid mirroring\n- none"
        style_modifier_text = _style_modifier_block(style_modifiers)
        total_turns = max(1, int(target_turns or 20))
        closing_window_turns, closing_window_start = _closing_window(total_turns)
        if total_turns <= 1:
            closure_guard = "- This is the only turn, so keep any wrap-up brief and natural."
        elif closing_window_turns == 1:
            closure_guard = "- Do not wrap up the session, say goodbye, or talk as if the session is ending before the final turn."
        else:
            closure_guard = (
                f"- Do not wrap up the session, say goodbye, thank the counselor for the session, or talk as if the "
                f"session is ending before turn {closing_window_start}."
            )
        if turn_id >= total_turns:
            closing_window_note = "- This is the final turn, so a natural ending is allowed."
        elif turn_id >= closing_window_start:
            closing_window_note = (
                f"- This turn is inside the closing window that starts at turn {closing_window_start}. "
                "A light closing tone is fine if it emerges naturally, but do not fully end the session yet."
            )
        else:
            closing_window_note = ""
        progression_rules = (
            "- Use the counselor's latest turn for context, not wording to mirror or summarize back.\n"
            "- Treat the hidden session beat plan below as background pacing only.\n"
            "- If a beat feels forced, drop it and stay with the most immediate feeling, body reaction, memory, obstacle, correction, or uncertainty from your side.\n"
            "- Let the turn form card set how much detail this turn should carry.\n"
        )

        core_sections = [
            "\n".join(
                [
                    "# Core Directive",
                    "- Priority order for this turn: satisfy <output_contract> exactly; stay anchored to the client role and the behavior cues below; obey the turn form block; preserve the same personality and factual life history; treat the beat plan only as secondary scaffolding.",
                    "- You are the client in a real counseling conversation with one counselor.",
                    "- The behavior cues below control how the client feels, interprets events, speaks, and shows symptoms in this turn. If other profile, background, memory, or source wording conflicts with the behavior cues, keep the same facts but rewrite the subjective parts to match the behavior cues.",
                    "- Keep objective facts stable: identity, biography, chronology, setting, relationships, and what concretely happened.",
                    "- Top-level anti-repetition rule: compared with the counselor's latest message and your own last 2-3 client turns, do not recycle the same semantic point, conclusion, fear statement, body-state description, or plan with only light rewording unless a very short acknowledgment is genuinely natural here, such as yes, maybe, or I don't know.",
                    "- Before writing, decide privately what one thing from your side feels most alive right now: a feeling, reaction, memory, obstacle, correction, uncertainty, or want. Center the turn on that one thing. You may add one supporting concrete detail or consequence if it naturally belongs with it. Do not show the private step.",
                    "- It is enough to answer, hesitate, correct one detail, say you do not know, or name one immediate reaction. Do not try to sound helpful or therapist-like.",
                    "- Respond from your own side of the exchange. If the counselor asked something, answer it, resist it, correct it, or say what blocks an answer. If the counselor offered a step, react to it from your side instead of restating it first.",
                    "- Do not proactively analyze yourself, narrate your psychology from the outside, or use therapy jargon, coping-skill labels, case-formulation language, or professional self-analysis.",
                    "- Sound like spoken English. Contractions, brief hesitations, and small discourse markers are fine when natural.",
                    "- If you use more than one sentence, keep the first sentence short and immediate. Do not front-load the whole explanation into the first sentence.",
                    closure_guard,
                    "- Treat the turn form block as a hard surface constraint for sentence count, word count, sentence shape, and question count.",
                    "- Let the behavior cues rewrite how the client feels, speaks, interprets events, and describes symptoms inside the same factual event.",
                    "- Reply as the client in first person.",
                ]
            ),
            opening_alignment_block,
            behavior_cue_block,
        ]
        context_sections = [
            "\n".join(
                [
                    "# Secondary Planning Context",
                    f"- Session: {session_id}",
                    f"- Turn: {turn_id}",
                    "- Preserve the same personality, guardedness, and interpersonal style before following any beat hint.",
                    "- Let the turn form card, not this block, decide the amount of detail for this turn.",
                    progression_rules.strip(),
                    closing_window_note,
                ]
            ),
            f"## Stable Background\n{stable_background_block or '- none'}",
        ]
        if short_term_note:
            context_sections.append(f"## Short-Term Carryover\n- {short_term_note}")
        if retrieved_memories:
            context_sections.append(f"## Retrieved Prior-Session Memories\n{memory_text}")
        if style_modifier_text:
            context_sections.append(f"## Personality Tendencies\n{style_modifier_text}")
        if complaint_phase_block:
            context_sections.append(complaint_phase_block)
        if turn_form_block:
            context_sections.append(turn_form_block)
        supplemental_sections = [
            "# Supplemental Guardrails\n"
            "- Some background records may be written in Chinese because they come from source case files. Translate them silently and never copy Chinese source text into the reply.\n"
            "- Do not turn the behavior cues into a checklist or a new dominant topic, and do not let them replace the counselor's latest point or the concrete event.\n"
            "- Do not sound like the counselor. Do not validate, reassure, appease, guide, summarize, or coach the counselor.\n"
            "- Do not tell the counselor what they should do next.\n"
            "- Do not repeat the counselor's question, suggestion, or wording back to the counselor before reacting to it.\n"
            "- If the counselor offered a step, do not simply echo it with frames like 'i can try ...' or 'i will try ...'. Say what makes it doable, hard, or uncertain from your side.\n"
            "- If the counselor asks a hypothetical or consequence question, answer the consequence, blocker, or uncertainty directly instead of repeating the setup.\n"
            "- If the turn stays brief, use one short natural clause rather than a bare filler token like okay, right, yes, or no.\n"
            "- Avoid coaching phrases such as 'you can ...', 'we will ...', 'let us ...', or 'I want you to ...' unless you are literally asking for help.\n"
            "- Speak from your own side of the exchange: what you feel, notice, fear, resist, want, or remember.\n"
            "- Only reference prior-session memories when clearly relevant; skip them when irrelevant.\n"
            "- Treat the carryover gist and retrieved memories as brief context, not wording to quote back.\n"
            "- Never mention PHQ/PHQ-8/PHQ-9, GAD, ORS, CSSRS, latent variables, hidden labels, or scores.\n"
            "- Do not introduce low-scored symptom topics unless they are already naturally relevant in the main presenting problem, recent events, or current exchange.\n"
            "- Prefer showing symptoms through lived scenes, body state, avoidance, and consequences. Use explicit symptom labels mainly when the counselor directly asks or safety clarity requires it.",
            f"## Anti-Mirroring Guardrail\n{recent_client_guardrail_text}",
        ]
        output_sections = [
            "# Hard Output Requirements\n"
            "- Conduct all visible dialogue in English only.\n"
            "- Return exactly one JSON object with exactly these keys: `message`, `emotion`.\n"
            "- Put the final first-person client reply text in `message`.\n"
            "- Put one short feeling phrase in `emotion`.\n"
            "- Do not output planning notes, labels, field descriptions, or any text outside the object.\n"
            "- Use the example only to understand the JSON shape. Do not copy its wording.\n"
            '- Example shape: {"message":"i am not sure. my body tensed up for a second.","emotion":"uneasy"}'
        ]
        final_reminder = (
            "# Final Reminder\n"
            "CRITICAL OVERRIDE: Satisfy <output_contract> first. Then apply the priority order in <core_directive>. Output only the final JSON object."
        )
        system_sections = [
            _tagged_prompt_block("output_contract", *output_sections),
            _tagged_prompt_block("core_directive", *core_sections),
            _tagged_prompt_block("context_and_data", *context_sections),
            _tagged_prompt_block("supplemental_guardrails", *supplemental_sections),
            _tagged_prompt_block("final_reminder", final_reminder),
        ]
        prompt_messages = [
            {"role": "system", "content": "\n\n".join(section for section in system_sections if section.strip())},
        ]
        if getattr(self.generator, "source", "") == "vertex":
            chat_history = _build_vertex_chat_history_messages(
                session_history, assistant_speaker="client"
            )
        else:
            chat_history = _build_chat_history_messages(session_history, assistant_speaker="client")
        if chat_history:
            prompt_messages.extend(chat_history)
        fallback_counselor_message = _normalize_space(counselor_message)
        if not chat_history and fallback_counselor_message:
            prompt_messages.append({"role": "user", "content": fallback_counselor_message})
        elif prompt_messages[-1]["role"] != "user" and fallback_counselor_message:
            prompt_messages.append({"role": "user", "content": fallback_counselor_message})
        return prompt_messages

    def respond(
        self,
        *,
        counselor_message: str,
        session_id: int,
        turn_id: int,
        target_turns: int | None = None,
        session_history: list[dict[str, str]],
        retrieved_memories: list[str],
        short_term_note: str,
        temperature: float,
        max_tokens: int,
        resolved_latent_state: ResolvedLatentState,
    ) -> ClientResponse:
        base_prompt_messages = self.build_prompt_messages(
            counselor_message=counselor_message,
            session_id=session_id,
            turn_id=turn_id,
            target_turns=target_turns,
            session_history=session_history,
            retrieved_memories=retrieved_memories,
            short_term_note=short_term_note,
            resolved_latent_state=resolved_latent_state,
        )
        prompt_messages = list(base_prompt_messages)
        last_error: Exception | None = None
        total_turns = max(1, int(target_turns or 20))
        _, closing_window_start = _closing_window(total_turns)
        closure_retry_rule = (
            "A natural ending is allowed, because this is the final turn."
            if turn_id >= total_turns
            else (
                f"A light closing tone is allowed only if it fits, because this turn is already inside the closing window "
                f"that starts at turn {closing_window_start}. Do not fully end the session yet."
                if turn_id >= closing_window_start
                else "Do not wrap up, say goodbye, or thank the counselor for the session in this retry."
            )
        )
        for attempt in range(1, self.MAX_RESPONSE_ATTEMPTS + 1):
            raw = self.generator.generate(
                messages=prompt_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            message_value: Any = ""
            invalid_reason: str | None = None
            try:
                parsed = extract_json_object(raw.text)
                message_value = parsed.get("message")
                emotion = str(parsed.get("emotion") or "").strip()
            except Exception:
                message, emotion = _parse_client_plain_text(raw.text)
                message_value = message

            invalid_reason = self._describe_invalid_client_message(
                message_value,
                counselor_message=counselor_message,
                session_history=session_history,
            )
            if invalid_reason is None:
                try:
                    return self._build_response(
                        message=str(message_value).strip(),
                        emotion=emotion or "unspecified",
                        thinking=raw.thinking,
                        raw_message=raw.raw_message,
                    )
                except ValueError as exc:
                    last_error = exc
                else:
                    last_error = None
            else:
                last_error = ValueError(f"Client {invalid_reason}")

            failure_reason = str(invalid_reason or last_error or "invalid output")
            self._log_invalid_output(
                {
                    "output_type": "client_turn",
                    "speaker": "Client",
                    "session_id": session_id,
                    "turn_id": turn_id,
                    "attempt": attempt,
                    "reason": failure_reason,
                    "raw_text": str(raw.text),
                    "raw_message": raw.raw_message if isinstance(raw.raw_message, dict) else str(raw.raw_message),
                }
            )

            if attempt >= self.MAX_RESPONSE_ATTEMPTS:
                break
            targeted_retry_guidance = self._invalid_client_retry_guidance(str(invalid_reason or last_error or ""))
            prompt_messages = base_prompt_messages + [
                {
                    "role": "user",
                    "content": (
                        "Your previous output was invalid. "
                        f"Reason: {invalid_reason or str(last_error or 'invalid output')}.\n"
                        f"{targeted_retry_guidance}\n"
                        "Return exactly one JSON object with keys message and emotion. "
                        "Put the final first-person English reply in message and a short feeling phrase in emotion. "
                        "Do not return Chinese text, numbers, or any non-string value for message. "
                        "Follow this priority order: output format first; behavior cues and turn form guardrails next; personality next; beat notes last. "
                        "Compared with the invalid draft, the counselor's latest wording, and your own last 2-3 client turns, do not recycle the same semantic point with only light rewording unless a minimal reply like yes or I don't know is naturally enough. "
                        "Do not repeat the counselor's question, suggestion, or wording back before reacting from your own side. "
                        "If the counselor offered a step, do not simply echo it with frames like 'i can try ...'; say what makes it doable, hard, or uncertain from your side. "
                        "Do not self-analyze, do not use therapist language, and do not coach or guide the counselor. "
                        f"{closure_retry_rule} "
                        "If the current beat feels forced, stay in character and follow the live exchange instead. "
                        "Use one fresh angle from your side, or keep it short if that is more natural. "
                        "A short natural clause is also acceptable, but avoid a bare filler token. "
                        "Start fresh from the latest conversation state; do not reuse wording or structure from the invalid draft."
                    ),
                }
            ]
        raise ValueError(
            f"{last_error} after {self.MAX_RESPONSE_ATTEMPTS} consecutive attempts"
        ) from last_error

    def build_self_report_messages(
        self,
        *,
        session_id: int,
        sample_index: int,
        session_history: list[dict[str, str]],
        short_term_note: str,
        resolved_latent_state: ResolvedLatentState,
        scale_payload: dict[str, Any],
    ) -> list[dict[str, str]]:
        stable_background_block = _stable_client_background_block(
            self.profile_data,
            presenting_problem=self.presenting_problem(),
        )
        behavior_cues = _self_report_behavior_cues(resolved_latent_state.hidden_state)

        history_text = "\n".join(
            f"{_history_speaker_label(str(item.get('speaker', '')))}: {str(item.get('text', ''))}"
            for item in session_history
        )
        system_prompt = build_phq8_self_report_prompt(
            scale_payload=scale_payload,
            behavior_cues=behavior_cues,
            secondary_core_lines=[
                f"- Session: {session_id}",
                f"- Self-report sample: {sample_index}/{self.SELF_REPORT_SAMPLE_COUNT}",
                "- This self-report happens after the completed counseling session below.",
                "- The question order and option order are randomized for this pass. Judge each option by meaning, not by position.",
                "- Stay consistent with the same client role, behavior cues, stable background, short-term carryover, and completed session history.",
            ],
            general_profile_lines=["You are the same client from the completed counseling session below."],
            core_additional_lines=[
                "- Keep objective facts stable: identity, biography, chronology, setting, relationships, and what concretely happened."
            ],
            extra_context_sections=[
                f"## Stable Background\n{stable_background_block or '- none'}",
                f"## Short-Term Carryover\n- {short_term_note or 'none'}",
                "## Completed Session History\n" + (history_text or "(no dialogue)"),
            ],
            supplemental_additional_lines=[
                "- Some background records may be in Chinese; translate them silently."
            ],
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": PHQ8_SELF_REPORT_USER_PROMPT},
        ]

    def self_report_session(
        self,
        *,
        session_id: int,
        session_history: list[dict[str, str]],
        short_term_note: str,
        max_tokens: int,
        resolved_latent_state: ResolvedLatentState,
    ) -> dict[str, Any]:
        sampled_scores: list[dict[str, int]] = []

        for sample_index in range(1, self.SELF_REPORT_SAMPLE_COUNT + 1):
            sample_scale_payload, sample_symptom_order = _self_report_scale_payload(
                session_id=session_id,
                sample_index=sample_index,
            )
            base_prompt_messages = self.build_self_report_messages(
                session_id=session_id,
                sample_index=sample_index,
                session_history=session_history,
                short_term_note=short_term_note,
                resolved_latent_state=resolved_latent_state,
                scale_payload=sample_scale_payload,
            )
            prompt_messages = list(base_prompt_messages)
            last_error: Exception | None = None
            for attempt in range(1, self.MAX_SELF_REPORT_ATTEMPTS + 1):
                raw = self.generator.generate(
                    messages=prompt_messages,
                    temperature=0.0,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                    vertex_response_schema=build_phq8_answers_response_schema(sample_scale_payload),
                )
                try:
                    parsed_scores, _ = parse_phq8_answers_response(
                        raw.text,
                        item_id_to_symptom=item_id_to_symptom_map(
                            sample_scale_payload,
                            symptom_order=sample_symptom_order,
                        ),
                        symptom_order=tuple(_PHQ8_ITEMS),
                        scale_payload=sample_scale_payload,
                    )
                    sampled_scores.append(parsed_scores)
                    break
                except Exception as exc:
                    last_error = exc
                    self._log_invalid_output(
                        {
                            "output_type": "self_report",
                            "speaker": "Client",
                            "session_id": session_id,
                            "sample_index": sample_index,
                            "attempt": attempt,
                            "reason": str(exc),
                            "raw_text": str(raw.text),
                            "raw_message": raw.raw_message if isinstance(raw.raw_message, dict) else str(raw.raw_message),
                        }
                    )

                if attempt >= self.MAX_SELF_REPORT_ATTEMPTS:
                    break
                prompt_messages = base_prompt_messages + [
                    {
                        "role": "user",
                        "content": build_phq8_self_report_retry_prompt(
                            validation_error=last_error or "unknown validation error",
                            scale_payload=sample_scale_payload,
                        ),
                    }
                ]
            else:
                last_error = ValueError("PHQ-8 self-report sample failed")

            if len(sampled_scores) < sample_index:
                raise ValueError(
                    f"{last_error} after {self.MAX_SELF_REPORT_ATTEMPTS} consecutive PHQ-8 self-report attempts "
                    f"for sample {sample_index}/{self.SELF_REPORT_SAMPLE_COUNT}"
                ) from last_error

        averaged_scores = {
            item: round(sum(sample[item] for sample in sampled_scores) / len(sampled_scores), 2)
            for item in _PHQ8_ITEMS
        }
        samples = [
            {
                "sample_index": sample_index,
                "symptom_scores": {
                    item: int(sample[item])
                    for item in _PHQ8_ITEMS
                },
                "overall_score": int(sum(sample[item] for item in _PHQ8_ITEMS)),
            }
            for sample_index, sample in enumerate(sampled_scores, start=1)
        ]
        return {
            "scale": "PHQ-8",
            "symptom_scores": averaged_scores,
            "overall_score": round(sum(averaged_scores.values()), 2),
            "sample_count": len(sampled_scores),
            "samples": samples,
        }

    def _build_response(
        self,
        *,
        message: str,
        emotion: str,
        thinking: str,
        raw_message: dict[str, Any],
    ) -> ClientResponse:
        cleaned_message = _remove_leaky_scoring_language(message)
        if not cleaned_message:
            raise ValueError("Client message is empty")
        if not emotion:
            raise ValueError("Client emotion is empty")

        return ClientResponse(
            text=cleaned_message,
            emotion=emotion,
            thinking=thinking,
            raw_message=raw_message,
        )
