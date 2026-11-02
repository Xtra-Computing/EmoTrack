from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from EmoTrack.conversation.model import OpenAIReasoningClient
from EmoTrack.conversation.message_validation import describe_echoing_message, describe_invalid_visible_message
from EmoTrack.conversation.realcbt_calibration.surface_control import (
    TurnFormPlan,
    build_realcbt_turn_form_plan,
    format_turn_form_block,
)
from EmoTrack.counselor.agent import CounselorProfile, CounselorTurn
from EmoTrack.counselor.summary_parser import parse_summary_output
from EmoTrack.utils.dialogue_text import (
    build_chat_history_messages as _build_chat_history_messages,
    build_vertex_chat_history_messages as _build_vertex_chat_history_messages,
    history_speaker_label as _history_speaker_label,
    normalize_space as _normalize_space,
    phase_lengths as _phase_lengths,
    short_phrase_list as _short_phrase_list,
)
from EmoTrack.utils.json_utils import extract_json_object


_SESSION_SUMMARY_VERTEX_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "summary": {"type": "STRING"},
        "key_facts": {"type": "ARRAY", "items": {"type": "STRING"}},
        "goals": {"type": "ARRAY", "items": {"type": "STRING"}},
        "homework": {"type": "ARRAY", "items": {"type": "STRING"}},
        "risk_flags": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["summary", "key_facts", "goals", "homework", "risk_flags"],
    "propertyOrdering": ["summary", "key_facts", "goals", "homework", "risk_flags"],
}


def _profile_mapping(profile_data: dict[str, Any] | None, key: str) -> dict[str, Any]:
    if not isinstance(profile_data, dict):
        return {}
    value = profile_data.get(key, {})
    return value if isinstance(value, dict) else {}


def _profile_json(value: dict[str, Any]) -> str:
    if not value:
        return "{}"
    return json.dumps(value, ensure_ascii=False, indent=2)


def _closing_window(total_turns: int) -> tuple[int, int]:
    total = max(1, int(total_turns))
    window_turns = max(1, (total + 9) // 10)
    start_turn = max(1, total - window_turns + 1)
    return window_turns, start_turn


_SESSION_KEY_PATTERN = re.compile(r"第\s*(\d+)\s*次")
_SESSION_RANGE_PATTERN = re.compile(r"(\d+)\s*[–-]\s*(\d+)")
_SOUND_STEP_PROPOSAL_PATTERN = re.compile(
    r"\b("
    r"listen|listening|recording|sound clip|background noise|nature sound|nature sounds|"
    r"fan|gentle hum|white noise|soft noise|audio|practice with noise|noise exposure|"
    r"desensiti[sz]e|just one minute|for a minute|one minute|light sound|gentle background"
    r")\b",
    re.IGNORECASE,
)
_STEP_INVITE_PATTERN = re.compile(
    r"\b("
    r"how about|could you|can you|would you|maybe you could|try|let'?s try|let'?s aim|"
    r"as a next step|it could help|you could"
    r")\b",
    re.IGNORECASE,
)
_STEP_REJECTION_PATTERN = re.compile(
    r"\b("
    r"i can't|i cannot|can't even|too hard|too much|overwhelming|intimidating|terrifying|"
    r"wish i could|don't know how|fills me with dread|makes things worse|just too much|"
    r"hard to think about|hard to imagine|not possible|too intimidating"
    r")\b",
    re.IGNORECASE,
)
_SKILL_PROCESS_TOKENS = (
    "保密",
    "知情同意",
    "权利",
    "义务",
    "资料",
    "一般资料",
    "临床核心资料",
    "病程",
    "就医历程",
    "临床印象",
    "他评",
    "摄入",
    "登记",
    "登记表",
    "规则",
    "流程",
    "协议",
    "设置",
    "费用",
    "频次",
    "时长",
    "预约",
    "督导",
    "研究",
    "量表",
    "测评",
    "测验",
    "填写",
    "机构",
    "系统记录",
    "观察并记录",
    "记录主诉",
)
_SKILL_GENERIC_RELATION_TOKENS = (
    "倾听",
    "共情",
    "尊重",
    "积极关注",
    "安全氛围",
    "信任",
    "合作关系",
    "尽情倾诉",
    "营造氛围",
    "接纳",
)
_SKILL_ACTION_TOKENS = (
    "澄清",
    "描述",
    "回顾",
    "定位",
    "梳理",
    "识别",
    "分析",
    "链接",
    "探索",
    "映射",
    "检验",
    "协商",
    "制定",
    "计划",
    "总结",
    "巩固",
    "复盘",
    "重构",
    "行为实验",
    "暴露",
    "放松",
    "problem",
    "explore",
    "clarify",
    "map",
    "summar",
    "plan",
)


@dataclass(frozen=True)
class _CounselorMove:
    phase_label: str
    move_family: str
    turn_job: str


@dataclass(frozen=True)
class _MoveState:
    phase_index: int
    phase_turn_index: int
    phase_turn_budget: int
    current_move: _CounselorMove
    next_move: _CounselorMove | None


@dataclass(frozen=True)
class _CounselorSkill:
    session_number: int
    skill_id: str
    skill_name: str
    skill_description: str
    when_to_use: str
    trigger: str
    meta_skill: str
    bucket: str
    is_exposure_like: bool
    runtime_usable: bool


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _clip_text(text: str, *, max_chars: int) -> str:
    normalized = _normalize_space(text)
    if not normalized:
        return ""
    if len(normalized) <= max_chars:
        return normalized
    clipped = normalized[: max_chars - 3].rstrip()
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0]
    return clipped.rstrip(",;:") + "..."


def _tagged_prompt_block(tag: str, *sections: str) -> str:
    body = "\n\n".join(section.strip() for section in sections if section and section.strip())
    return f"<{tag}>\n{body}\n</{tag}>"


def _step_proposal_fragment(text: str) -> str:
    normalized = _normalize_space(text)
    if not normalized:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    matched = [
        sentence
        for sentence in sentences
        if _STEP_INVITE_PATTERN.search(sentence) is not None or _SOUND_STEP_PROPOSAL_PATTERN.search(sentence) is not None
    ]
    return " ".join(matched) if matched else normalized


def _dedupe_lines(lines: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for line in lines:
        normalized = _normalize_space(line)
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(normalized)
    return out


def _split_compact_items(text: str, *, limit: int = 3, max_chars: int = 160) -> list[str]:
    if not text:
        return []
    items: list[str] = []
    for part in re.split(r"\s*;\s*", text):
        clipped = _clip_text(part, max_chars=max_chars)
        if clipped:
            items.append(clipped)
        if len(items) >= limit:
            break
    return items


def _parse_carryover_note(note: str) -> dict[str, list[str] | str]:
    sections: dict[str, list[str] | str] = {
        "summary": "",
        "carry_forward": [],
        "key_facts": [],
        "goals": [],
    }
    if not str(note or "").strip():
        return sections

    summary_lines: list[str] = []
    section_keys = {
        "carry forward": "carry_forward",
        "key facts": "key_facts",
        "goals": "goals",
        "homework": "",
        "risk flags": "",
    }
    for raw_line in str(note).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            continue
        line = re.sub(r"^Longitudinal carryover (?:note|gist):\s*", "", line, flags=re.IGNORECASE)
        normalized = line.lower()
        matched_section = False
        for label, section_name in section_keys.items():
            prefix = f"{label}:"
            if not normalized.startswith(prefix):
                continue
            matched_section = True
            if section_name:
                body = line[len(prefix) :].strip()
                items = _split_compact_items(body, limit=3, max_chars=160)
                target = sections.get(section_name, [])
                if isinstance(target, list):
                    target.extend(items)
            break
        if matched_section:
            continue
        clipped = _clip_text(line, max_chars=260)
        if clipped:
            summary_lines.append(clipped)

    sections["summary"] = _clip_text(" ".join(summary_lines), max_chars=260)
    for key in ("carry_forward", "key_facts", "goals"):
        value = sections.get(key, [])
        if isinstance(value, list):
            sections[key] = _dedupe_lines(value)
    return sections


def _load_source_profile(conversation_profile: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(conversation_profile, dict):
        return {}
    meta = conversation_profile.get("meta", {})
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


def _infer_skill_bucket(text: str) -> str:
    normalized = _normalize_space(text).lower()
    if not normalized:
        return "general"
    if any(token in normalized for token in ("总结", "巩固", "回顾", "复盘", "计划", "预防", "迁移", "close", "summar", "next step")):
        return "consolidation"
    if any(token in normalized for token in ("暴露", "放松", "行为实验", "系统脱敏", "技能", "练习", "训练", "应对", "重评", "认知修正", "认知重构", "experiment", "exposure", "relax")):
        return "change"
    if any(token in normalized for token in ("abc", "概念化", "自动想法", "认知", "核心信念", "模式", "澄清", "功能分析", "link", "pattern", "belief")):
        return "formulation"
    if any(token in normalized for token in ("评估", "资料", "主诉", "收集", "观察", "assessment")):
        return "assessment"
    return "general"


def _is_runtime_usable_skill(
    *,
    skill_name: str,
    skill_description: str,
    when_to_use: str,
    trigger: str,
) -> bool:
    public_text = _normalize_space(" ".join([skill_name, skill_description, when_to_use, trigger])).lower()
    action_text = _normalize_space(" ".join([skill_name, skill_description])).lower()
    if not public_text:
        return False
    if any(token.lower() in public_text for token in _SKILL_PROCESS_TOKENS):
        return False
    if any(token.lower() in action_text for token in _SKILL_GENERIC_RELATION_TOKENS) and not any(
        token.lower() in action_text for token in _SKILL_ACTION_TOKENS
    ):
        return False
    return True


def _extract_genre_skill_bank(source_profile: dict[str, Any], *, limit: int = 30) -> list[_CounselorSkill]:
    sessions = source_profile.get("sessions", [])
    if not isinstance(sessions, list):
        return []

    deduped: list[_CounselorSkill] = []
    seen: set[tuple[str, str]] = set()
    for idx, session in enumerate(sessions, start=1):
        if not isinstance(session, dict):
            continue
        raw_session_number = session.get("session_number")
        if isinstance(raw_session_number, int):
            session_number = raw_session_number
        else:
            try:
                session_number = int(str(raw_session_number).strip())
            except Exception:
                session_number = idx
        suggest_skills = session.get("suggest_skills", [])
        if not isinstance(suggest_skills, list):
            continue
        for entry in suggest_skills:
            if not isinstance(entry, dict):
                continue
            meta_skill = _normalize_space(entry.get("meta_skill", ""))
            micro_skills = entry.get("micro_skills", [])
            if not isinstance(micro_skills, list):
                continue
            for micro in micro_skills:
                if not isinstance(micro, dict):
                    continue
                skill_name = _normalize_space(micro.get("skill_name", ""))
                skill_id = _normalize_space(micro.get("skill_id", ""))
                if not skill_name:
                    continue
                skill_description = _normalize_space(micro.get("skill_description", ""))
                when_to_use = _normalize_space(micro.get("when_to_use", ""))
                trigger = _normalize_space(micro.get("trigger", ""))
                runtime_usable = _is_runtime_usable_skill(
                    skill_name=skill_name,
                    skill_description=skill_description,
                    when_to_use=when_to_use,
                    trigger=trigger,
                )
                if not runtime_usable:
                    continue
                key = (skill_id or skill_name, skill_name)
                if key in seen:
                    continue
                bucket = _infer_skill_bucket(" ".join([skill_name, skill_description, when_to_use, trigger]))
                joined = " ".join([skill_name, skill_description, when_to_use, trigger])
                deduped.append(
                    _CounselorSkill(
                        session_number=session_number,
                        skill_id=skill_id,
                        skill_name=skill_name,
                        skill_description=skill_description,
                        when_to_use=when_to_use,
                        trigger=trigger,
                        meta_skill=meta_skill,
                        bucket=bucket,
                        is_exposure_like=_SOUND_STEP_PROPOSAL_PATTERN.search(joined) is not None
                        or any(token in joined for token in ("暴露", "系统脱敏", "声音", "音源", "放松训练")),
                        runtime_usable=runtime_usable,
                    )
                )
                seen.add(key)
                if len(deduped) >= limit:
                    return deduped
    return deduped


def _stored_genre_skill_bank(conversation_profile: dict[str, Any] | None, *, limit: int = 30) -> list[_CounselorSkill]:
    if not isinstance(conversation_profile, dict):
        return []
    raw_bank = conversation_profile.get("counselor_genre_skill_bank", [])
    if not isinstance(raw_bank, list):
        return []

    stored: list[_CounselorSkill] = []
    for entry in raw_bank:
        if not isinstance(entry, dict):
            continue
        try:
            session_number = int(entry.get("session_number", 0))
        except Exception:
            continue
        skill_name = _normalize_space(entry.get("skill_name", ""))
        if session_number <= 0 or not skill_name:
            continue
        runtime_usable = bool(entry.get("runtime_usable", True))
        if not runtime_usable:
            continue
        stored.append(
            _CounselorSkill(
                session_number=session_number,
                skill_id=_normalize_space(entry.get("skill_id", "")),
                skill_name=skill_name,
                skill_description=_normalize_space(entry.get("skill_description", "")),
                when_to_use=_normalize_space(entry.get("when_to_use", "")),
                trigger=_normalize_space(entry.get("trigger", "")),
                meta_skill=_normalize_space(entry.get("meta_skill", "")),
                bucket=_normalize_space(entry.get("bucket", "")) or "general",
                is_exposure_like=bool(entry.get("is_exposure_like", False)),
                runtime_usable=runtime_usable,
            )
        )
        if len(stored) >= limit:
            break
    return stored


class EverMemCounselorAgent:
    MAX_RESPONSE_ATTEMPTS = 3

    def __init__(
        self,
        *,
        generator: OpenAIReasoningClient,
        profile: CounselorProfile,
        conversation_profile: dict[str, Any] | None = None,
        run_seed: int = 0,
        invalid_output_logger: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.generator = generator
        self.profile = profile
        self.conversation_profile = conversation_profile or {}
        self.run_seed = int(run_seed)
        self.invalid_output_logger = invalid_output_logger
        self._genre_skill_bank = _stored_genre_skill_bank(self.conversation_profile)
        self._source_profile = _load_source_profile(self.conversation_profile) if not self._genre_skill_bank else {}
        if not self._genre_skill_bank:
            self._genre_skill_bank = _extract_genre_skill_bank(self._source_profile)

    def set_invalid_output_logger(
        self,
        logger: Callable[[dict[str, Any]], None] | None,
    ) -> None:
        self.invalid_output_logger = logger

    def _log_invalid_output(self, payload: dict[str, Any]) -> None:
        if self.invalid_output_logger is None:
            return
        self.invalid_output_logger(payload)

    @staticmethod
    def _turn_phase(turn_id: int, target_turns: int) -> tuple[str, str]:
        total_turns = max(1, int(target_turns))
        if turn_id <= 1:
            return (
                "opening",
                "Early in the session, avoid stock empathy and simple restatement. "
                "Quickly orient to the client's immediate concrete problem, the most recent triggering situation, "
                "relevant past episodes, and what they want help with today.",
            )
        if turn_id >= total_turns:
            return (
                "closing",
                "Late in the session, briefly summarize what became clearer in emotion handling, daily functioning, "
                "or behavior, then agree on one concrete next step that matches the complaint and the client's goal.",
            )

        progress = turn_id / total_turns
        if progress <= 0.25:
            return (
                "assessment",
                "Stay in early-session exploration. Deepen one recent concrete event, trigger, body response, "
                "avoidance pattern, prior similar experience, or a realistic session goal without defaulting to a recap.",
            )
        if progress <= 0.55:
            return (
                "formulation",
                "Use the middle of the session to deepen self-awareness around thoughts, emotions, key people, "
                "coping patterns, self-view, and available resources, while staying anchored to the presenting complaint "
                "and specific recent situations.",
            )
        if progress <= 0.80:
            return (
                "change-work",
                "Use one concrete intervention tied to the complaint domain and test it against a real recent situation. "
                "Prefer behaviorally specific ideas over generic encouragement or generic social advice.",
            )
        return (
            "consolidation",
            "Keep the late session focused on what fits, what still blocks change, and what concrete next step feels "
            "doable without drifting into a new generic topic.",
        )

    def _turn_surface_plan(
        self,
        *,
        session_id: int,
        turn_id: int,
        target_turns: int,
        session_history: list[dict[str, str]],
        move_family: str,
    ) -> TurnFormPlan:
        client_message = _normalize_space(session_history[-1].get("text", "")) if session_history else ""
        return build_realcbt_turn_form_plan(
            speaker="counselor",
            session_id=session_id,
            turn_id=turn_id,
            target_turns=target_turns,
            shared_source_text=client_message,
            run_seed=self.run_seed,
            prev_utterance_text=client_message,
        )

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

    def _case_session_focus(self, session_id: int) -> str:
        counselor_case_context = _profile_mapping(self.conversation_profile, "counselor_case_context")
        global_plan = counselor_case_context.get("global_plan", [])
        if not isinstance(global_plan, list):
            return ""

        matched_themes: list[str] = []
        stage_fallback = ""
        for stage in global_plan:
            if not isinstance(stage, dict):
                continue
            sessions_text = str(stage.get("sessions") or "").strip()
            stage_name = str(stage.get("stage_name") or "").strip()
            if not stage_fallback and stage_name:
                stage_fallback = stage_name

            if sessions_text:
                range_match = _SESSION_RANGE_PATTERN.search(sessions_text)
                if range_match is not None:
                    start_session = int(range_match.group(1))
                    end_session = int(range_match.group(2))
                    if start_session <= session_id <= end_session and stage_name:
                        stage_fallback = stage_name
                else:
                    numbers = [int(value) for value in re.findall(r"\d+", sessions_text)]
                    if session_id in numbers and stage_name:
                        stage_fallback = stage_name

            session_plan = stage.get("session_plan", [])
            if not isinstance(session_plan, list):
                continue
            for entry in session_plan:
                if not isinstance(entry, dict):
                    continue
                session_key = str(entry.get("session_key") or "").strip()
                key_match = _SESSION_KEY_PATTERN.search(session_key)
                if key_match is None or int(key_match.group(1)) != session_id:
                    continue
                theme = str(entry.get("theme") or "").strip()
                if theme:
                    matched_themes.append(theme)

        if matched_themes:
            return "; ".join(dict.fromkeys(matched_themes))
        return stage_fallback

    @staticmethod
    def _public_client_anchors(session_history: list[dict[str, str]], *, limit: int = 2) -> list[str]:
        anchors = [
            str(item.get("text", "")).strip()
            for item in session_history
            if _history_speaker_label(str(item.get("speaker", ""))) == "client"
            and str(item.get("text", "")).strip()
        ]
        return anchors[-limit:]

    def _presenting_problem_anchor(self) -> str:
        intake_profile = _profile_mapping(self.conversation_profile, "intake_profile")
        client_info = _profile_mapping(self.conversation_profile, "client_info")
        for mapping in (client_info, intake_profile):
            for key in ("main_problem", "core_demands", "topic"):
                text = str(mapping.get(key) or "").strip()
                if text:
                    return text
        return ""

    @staticmethod
    def _move_lattice() -> list[list[_CounselorMove]]:
        return [
            [
                _CounselorMove("opening_assessment", "orient_to_immediate_problem", "Open naturally and orient to the immediate problem without a stock empathy formula."),
                _CounselorMove("opening_assessment", "pin_down_recent_scene", "Pull the client toward one recent concrete scene where the complaint showed up."),
                _CounselorMove("opening_assessment", "map_body_alarm", "Clarify body reactions, bracing, or immediate panic responses in that scene."),
                _CounselorMove("opening_assessment", "clarify_todays_goal", "Clarify what feels most urgent and what the client wants help with today."),
            ],
            [
                _CounselorMove("formulation", "probe_feared_prediction", "Clarify what the client fears the trigger means about safety, embarrassment, recovery, or the future."),
                _CounselorMove("formulation", "map_avoidance_and_cost", "Map how avoidance protects the client in the short term and what it costs in work, daily life, or relationships."),
                _CounselorMove("formulation", "link_pattern", "Link trigger, body alarm, thought, avoidance, and consequence into one coherent pattern."),
                _CounselorMove("formulation", "identify_desired_change", "Clarify what the client most wants to be different if this problem eased."),
            ],
            [
                _CounselorMove("change_work", "choose_target_scene", "Choose one real complaint-linked scene to work on instead of speaking in broad generalities."),
                _CounselorMove("change_work", "propose_testable_step", "Offer one small, behaviorally specific experiment tied to that scene."),
                _CounselorMove("change_work", "troubleshoot_obstacle", "Identify the main obstacle that makes the proposed step feel too hard right now."),
                _CounselorMove("change_work", "refine_step_conditions", "Refine the step so it matches the complaint domain and feels just barely doable."),
            ],
            [
                _CounselorMove("consolidation", "review_what_fits", "Review which idea or formulation fits the client's experience and which part does not."),
                _CounselorMove("consolidation", "identify_remaining_block", "Name the remaining blocker without drifting into a new generic topic."),
                _CounselorMove("consolidation", "turn_learning_into_plan", "Turn what fits into one complaint-linked next step."),
                _CounselorMove("consolidation", "prepare_for_close", "Clarify what still feels unresolved or most useful without starting the close yet."),
            ],
            [
                _CounselorMove("closing", "summarize_shift", "Summarize what became clearer in emotion, behavior, or daily functioning."),
                _CounselorMove("closing", "name_unresolved_piece", "Name what still feels hard without opening a brand new topic."),
                _CounselorMove("closing", "agree_next_step", "Agree on one realistic next step tied to the complaint domain."),
                _CounselorMove("closing", "close_concisely", "Close briefly, naturally, and specifically."),
            ],
        ]

    @classmethod
    def _move_state_for_turn(cls, *, turn_id: int, target_turns: int) -> _MoveState:
        total_turns = max(1, int(target_turns))
        _, closing_window_start = _closing_window(total_turns)
        lattice = cls._move_lattice()
        lengths = _phase_lengths(total_turns, len(lattice))

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
        if phase_turn_budget <= 1:
            slot_index = len(lattice[phase_index]) - 1
        else:
            slot_index = round((phase_turn_index - 1) * (len(lattice[phase_index]) - 1) / (phase_turn_budget - 1))
        current_move = lattice[phase_index][slot_index]
        is_final_turn = turn_id >= total_turns
        if current_move.phase_label == "closing" and turn_id < closing_window_start:
            current_move = lattice[3][slot_index]
        next_move = None
        if phase_turn_index >= phase_turn_budget and phase_index + 1 < len(lattice):
            next_move = lattice[phase_index + 1][0]
        return _MoveState(
            phase_index=phase_index,
            phase_turn_index=phase_turn_index,
            phase_turn_budget=phase_turn_budget,
            current_move=current_move,
            next_move=next_move,
        )

    @staticmethod
    def _eligible_skill_buckets(phase_name: str, move_family: str) -> tuple[str, ...]:
        if phase_name in {"opening", "assessment"}:
            return ("assessment", "formulation", "general")
        if phase_name == "formulation":
            return ("formulation", "assessment", "general")
        if phase_name == "change-work":
            return ("change", "formulation", "general")
        if phase_name in {"consolidation", "closing"}:
            if move_family in {"close_concisely", "summarize_shift", "agree_next_step", "name_unresolved_piece"}:
                return ("consolidation", "change", "formulation", "general")
            return ("consolidation", "formulation", "change", "general")
        return ("general", "assessment", "formulation", "change", "consolidation")

    def _select_scheduled_skill(
        self,
        *,
        session_id: int,
        turn_id: int,
        target_turns: int,
        phase_name: str,
        move_family: str,
        recent_skill_names: set[str],
        rejected_sound_step_count: int,
    ) -> _CounselorSkill | None:
        if not self._genre_skill_bank:
            return None

        eligible_buckets = self._eligible_skill_buckets(phase_name, move_family)
        max_session = max((skill.session_number for skill in self._genre_skill_bank), default=session_id)
        current_session = min(max(1, int(session_id)), max_session)

        def filter_candidates(skills: list[_CounselorSkill], *, strict_session: bool, strict_bucket: bool) -> list[_CounselorSkill]:
            filtered: list[_CounselorSkill] = []
            for skill in skills:
                if not skill.runtime_usable:
                    continue
                if strict_session and skill.session_number > current_session + 1:
                    continue
                if strict_bucket and skill.bucket not in eligible_buckets:
                    continue
                if rejected_sound_step_count > 0 and skill.is_exposure_like:
                    continue
                if skill.skill_name in recent_skill_names:
                    continue
                filtered.append(skill)
            return filtered

        candidate_sets = [
            filter_candidates(self._genre_skill_bank, strict_session=True, strict_bucket=True),
            filter_candidates(self._genre_skill_bank, strict_session=False, strict_bucket=True),
        ]
        candidates = next((items for items in candidate_sets if items), [])
        if not candidates and recent_skill_names:
            candidates = [
                skill
                for skill in self._genre_skill_bank
                if skill.runtime_usable
                and skill.bucket in eligible_buckets
                and (rejected_sound_step_count <= 0 or not skill.is_exposure_like)
            ]
        if not candidates:
            return None

        ranked = sorted(
            candidates,
            key=lambda skill: (
                abs(skill.session_number - current_session),
                hashlib.sha256(
                    (
                        f"genre-skill|genre={self.conversation_profile.get('theoretical','')}|"
                        f"session={session_id}|turn={turn_id}|target={target_turns}|phase={phase_name}|move={move_family}|"
                        f"skill={skill.skill_id or skill.skill_name}"
                    ).encode("utf-8")
                ).hexdigest(),
            ),
        )
        return ranked[0]

    def _scheduled_skill_state(
        self,
        *,
        session_id: int,
        turn_id: int,
        target_turns: int,
        phase_name: str,
        current_move: _CounselorMove,
        rejected_sound_step_count: int,
    ) -> tuple[_CounselorSkill | None, list[str]]:
        recent_skill_names: set[str] = set()
        recent_selected: list[str] = []
        for past_turn in range(max(1, turn_id - 2), turn_id):
            past_phase_name, _ = self._turn_phase(past_turn, target_turns)
            past_move = self._move_state_for_turn(turn_id=past_turn, target_turns=target_turns).current_move
            past_skill = self._select_scheduled_skill(
                session_id=session_id,
                turn_id=past_turn,
                target_turns=target_turns,
                phase_name=past_phase_name,
                move_family=past_move.move_family,
                recent_skill_names=recent_skill_names,
                rejected_sound_step_count=0,
            )
            if past_skill is None:
                continue
            recent_skill_names.add(past_skill.skill_name)
            recent_selected.append(past_skill.skill_name)

        current_skill = self._select_scheduled_skill(
            session_id=session_id,
            turn_id=turn_id,
            target_turns=target_turns,
            phase_name=phase_name,
            move_family=current_move.move_family,
            recent_skill_names=recent_skill_names,
            rejected_sound_step_count=rejected_sound_step_count,
        )
        return current_skill, recent_selected

    @staticmethod
    def _looks_like_step_proposal(text: str) -> bool:
        normalized = str(text or "").strip()
        if not normalized:
            return False
        return _STEP_INVITE_PATTERN.search(normalized) is not None or _SOUND_STEP_PROPOSAL_PATTERN.search(normalized) is not None

    @staticmethod
    def _looks_like_sound_step(text: str) -> bool:
        normalized = str(text or "").strip()
        if not normalized:
            return False
        return _SOUND_STEP_PROPOSAL_PATTERN.search(normalized) is not None

    @staticmethod
    def _looks_like_rejection(text: str) -> bool:
        normalized = str(text or "").strip()
        if not normalized:
            return False
        return _STEP_REJECTION_PATTERN.search(normalized) is not None

    @classmethod
    def _recent_rejected_step_loop(
        cls,
        session_history: list[dict[str, str]],
        *,
        pair_limit: int = 3,
    ) -> tuple[int, int, str]:
        pairs: list[tuple[str, str, bool]] = []
        for idx in range(len(session_history) - 1):
            counselor_turn = session_history[idx]
            client_turn = session_history[idx + 1]
            if _history_speaker_label(str(counselor_turn.get("speaker", ""))) != "counselor":
                continue
            if _history_speaker_label(str(client_turn.get("speaker", ""))) != "client":
                continue
            counselor_text = str(counselor_turn.get("text", "")).strip()
            client_text = str(client_turn.get("text", "")).strip()
            if not counselor_text or not client_text:
                continue
            if not cls._looks_like_step_proposal(counselor_text):
                continue
            if not cls._looks_like_rejection(client_text):
                continue
            pairs.append((counselor_text, client_text, cls._looks_like_sound_step(counselor_text)))

        recent_pairs = pairs[-pair_limit:]
        rejected_count = len(recent_pairs)
        rejected_sound_step_count = sum(1 for _, _, is_sound_step in recent_pairs if is_sound_step)
        latest_sound_step = ""
        for counselor_text, _, is_sound_step in reversed(recent_pairs):
            if is_sound_step:
                latest_sound_step = counselor_text
                break
        return rejected_count, rejected_sound_step_count, latest_sound_step

    @classmethod
    def _prior_step_proposals(
        cls,
        session_history: list[dict[str, str]],
        *,
        limit: int = 6,
    ) -> list[str]:
        proposals: list[str] = []
        seen: set[str] = set()
        for item in session_history:
            if _history_speaker_label(str(item.get("speaker", ""))) != "counselor":
                continue
            text = _normalize_space(item.get("text", ""))
            if not text or not cls._looks_like_step_proposal(text):
                continue
            key = _normalize_space(_step_proposal_fragment(text)).lower()
            if not key or key in seen:
                continue
            seen.add(key)
            proposals.append(text)
        return proposals[-limit:]

    @staticmethod
    def _move_constraints(move_family: str) -> list[str]:
        constraints = {
            "orient_to_immediate_problem": [
                "Open naturally and orient to the immediate complaint or most recent trigger.",
                "Do not give coping advice or homework yet.",
            ],
            "pin_down_recent_scene": [
                "Pull the client toward one recent concrete scene with place, sound, and moment details.",
                "Do not jump to interpretation or treatment advice.",
            ],
            "map_body_alarm": [
                "Stay with body alarm, bracing, and immediate reactions in the recent scene.",
                "Do not propose a new step or homework.",
            ],
            "clarify_todays_goal": [
                "Clarify what feels most urgent and what help the client wants today.",
                "Do not introduce a technique yet.",
            ],
            "probe_feared_prediction": [
                "Ask or reflect on what the trigger means about safety, recovery, embarrassment, or the future.",
                "Do not introduce a coping exercise in this turn.",
            ],
            "map_avoidance_and_cost": [
                "Map short-term relief versus cost in work, daily life, or relationships.",
                "Do not propose a new experiment in this turn.",
            ],
            "link_pattern": [
                "Link trigger, body reaction, thought, avoidance, and cost into one concise pattern.",
                "Do not open a new topic or assign homework.",
            ],
            "identify_desired_change": [
                "Clarify what the client most wants to regain or make easier.",
                "Keep it focused on the complaint domain, not generic life improvement.",
            ],
            "choose_target_scene": [
                "Choose one real complaint-linked scene to work on.",
                "Do not jump ahead to a full intervention plan yet.",
            ],
            "propose_testable_step": [
                "Offer one complaint-linked step only.",
                "Make it concrete in duration, setting, or condition.",
                "Do not bundle multiple techniques together.",
            ],
            "troubleshoot_obstacle": [
                "Stay with why the last idea feels too hard, unsafe, or badly matched.",
                "Do not replace it with another barely reworded version of the same step.",
            ],
            "refine_step_conditions": [
                "If you refine a step, change conditions such as timing, support, distance, duration, or exit option.",
                "Do not invent a fresh step family in this turn.",
            ],
            "review_what_fits": [
                "Review what formulation or idea fits the client's experience so far.",
                "Do not introduce a new technique here.",
            ],
            "identify_remaining_block": [
                "Name the main remaining blocker clearly and concretely.",
                "Do not drift into a new generic intervention topic.",
            ],
            "turn_learning_into_plan": [
                "Turn what fits into one specific complaint-linked next step.",
                "Keep it realistic and tightly scoped.",
            ],
            "prepare_for_close": [
                "Clarify what still feels unresolved or most useful without sounding like the session is ending yet.",
                "Do not reopen broad exploration, introduce a brand new exercise, or say goodbye.",
            ],
            "summarize_shift": [
                "Use the first sentence to summarize what became clearer.",
                "Do not ask a brand-new exploratory question.",
            ],
            "name_unresolved_piece": [
                "Name what still feels hard without opening a new topic.",
                "Keep it short and specific.",
            ],
            "agree_next_step": [
                "Agree on one realistic next step that follows naturally from the session.",
                "Do not introduce a surprising new intervention family at the end.",
            ],
            "close_concisely": [
                "Close briefly and specifically.",
                "Do not open a new question or new intervention.",
            ],
            "identify_remaining_block_after_rejection": [
                "The client has already rejected a similar sound-practice step.",
                "Name the blocker, mismatch, or safety condition that still makes direct sound work feel too hard.",
                "Do not repeat background-noise, fan, hum, recording, nature-sound, or generic sound-practice suggestions.",
            ],
            "summarize_blocker_before_close": [
                "Briefly summarize what became clearer and what blocker still remains.",
                "If you mention a next step, keep it preparatory and different from the rejected sound-practice idea.",
                "Do not repeat background-noise, fan, hum, recording, nature-sound, or generic sound-practice suggestions.",
            ],
        }
        return constraints.get(move_family, ["Keep the turn tightly aligned with the stated move job."])

    def _current_move_state(
        self,
        *,
        session_id: int,
        turn_id: int,
        target_turns: int,
        session_history: list[dict[str, str]],
    ) -> dict[str, str]:
        total_turns = max(1, int(target_turns))
        is_final_turn = turn_id >= total_turns
        _, closing_window_start = _closing_window(total_turns)
        is_penultimate_turn = total_turns > 1 and turn_id == total_turns - 1
        move_state = self._move_state_for_turn(turn_id=turn_id, target_turns=target_turns)
        current_move = move_state.current_move

        presenting_problem = self._presenting_problem_anchor()
        session_focus = self._case_session_focus(session_id)
        rejected_step_count, rejected_sound_step_count, latest_rejected_sound_step = self._recent_rejected_step_loop(
            session_history
        )
        if rejected_sound_step_count > 0 and current_move.move_family in {
            "propose_testable_step",
            "refine_step_conditions",
            "turn_learning_into_plan",
            "agree_next_step",
            "close_concisely",
        }:
            if is_final_turn:
                current_move = _CounselorMove(
                    "closing",
                    "summarize_blocker_before_close",
                    "Summarize what became clearer and name the blocker that still makes direct sound practice feel too hard. "
                    "If you mention a next step, keep it preparatory and different from the rejected sound-practice idea.",
                )
            else:
                current_move = _CounselorMove(
                    "consolidation",
                    "identify_remaining_block_after_rejection",
                    "A similar sound-practice step was already declined. Clarify the blocker, mismatch, or safety condition "
                    "instead of proposing another version of the same idea.",
                )
        move_constraints = self._move_constraints(current_move.move_family)
        primary_constraint = next((rule for rule in move_constraints if rule.lower().startswith("do not ")), "")
        if not primary_constraint and move_constraints:
            primary_constraint = move_constraints[-1]
        if rejected_sound_step_count > 0:
            primary_constraint = (
                "Clarify what still makes direct sound practice feel unsafe, and do not repeat the same sound-practice suggestion."
            )
        elif rejected_step_count > 0 and not primary_constraint:
            primary_constraint = (
                "Stay with the mismatch or blocker behind the last rejected step before offering another plan."
            )

        current_target = _clip_text(current_move.turn_job, max_chars=220)
        if is_penultimate_turn:
            if rejected_sound_step_count > 0:
                current_target = (
                    "Begin the wrap-up by briefly summarizing what became clearer and what blocker still remains, "
                    "then set up one realistic preparatory next step if it fits."
                )
            else:
                current_target = (
                    "Begin the wrap-up by briefly summarizing what became clearer or still feels unresolved, and set "
                    "up one realistic next step."
                )
            primary_constraint = "Do not say goodbye or fully close the session yet."
        elif is_final_turn:
            primary_constraint = "Close briefly and naturally without opening a new topic."
        elif turn_id >= closing_window_start and current_move.move_family == "prepare_for_close":
            primary_constraint = "Start narrowing toward the wrap-up, but do not say goodbye yet."

        return {
            "phase_name": current_move.phase_label,
            "move_family": current_move.move_family,
            "presenting_problem": _clip_text(presenting_problem, max_chars=220),
            "session_focus": _clip_text(session_focus, max_chars=180),
            "current_target": current_target,
            "constraint": _clip_text(primary_constraint, max_chars=220),
        }

    @staticmethod
    def _format_bullet_section(title: str, lines: list[str], *, empty_text: str = "none") -> str:
        cleaned = [line for line in _dedupe_lines(lines) if line]
        if not cleaned:
            cleaned = [empty_text]
        heading = title.rstrip(":")
        return "\n".join([f"## {heading}", *[f"- {line}" for line in cleaned]])

    def _client_summary_lines(self, *, short_term_note: str) -> list[str]:
        intake_profile = _profile_mapping(self.conversation_profile, "intake_profile")
        client_info = _profile_mapping(self.conversation_profile, "client_info")
        counselor_case_context = _profile_mapping(self.conversation_profile, "counselor_case_context")
        carryover = _parse_carryover_note(short_term_note)

        lines: list[str] = []
        static_traits = client_info.get("static_traits", {})
        if isinstance(static_traits, dict):
            identity_parts = [
                str(static_traits.get("name") or "").strip(),
                str(static_traits.get("age") or "").strip(),
                str(static_traits.get("occupation") or "").strip(),
            ]
            identity = ", ".join(part for part in identity_parts if part)
            if identity:
                lines.append(identity)

        main_problem = ""
        for mapping in (client_info, intake_profile):
            main_problem = _clip_text(str(mapping.get("main_problem") or ""), max_chars=220)
            if main_problem:
                break
        if main_problem:
            lines.append(f"Main problem: {main_problem}")

        goal_candidates = []
        parsed_goals = carryover.get("goals", [])
        if isinstance(parsed_goals, list):
            goal_candidates.extend(parsed_goals)
        for mapping in (counselor_case_context, intake_profile, client_info):
            value = str(mapping.get("core_demands") or mapping.get("topic") or "").strip()
            if value:
                goal_candidates.append(value)
        if goal_candidates:
            lines.append(f"Current goal: {_clip_text(goal_candidates[0], max_chars=200)}")

        target_behavior = client_info.get("target_behavior")
        if isinstance(target_behavior, list):
            for entry in target_behavior:
                if not isinstance(entry, dict):
                    continue
                fragments = [
                    _normalize_space(entry.get("behavior", "")),
                    _normalize_space(entry.get("consequence", "")),
                ]
                pattern = "; ".join(fragment for fragment in fragments if fragment)
                if pattern:
                    lines.append(f"Pattern: {_clip_text(pattern, max_chars=220)}")
                    break

        return _dedupe_lines(lines)[:4]

    def _earlier_conversation_summary_lines(
        self,
        *,
        session_history: list[dict[str, str]],
        retrieved_memories: list[str],
        short_term_note: str,
    ) -> list[str]:
        carryover = _parse_carryover_note(short_term_note)
        lines: list[str] = []

        summary_text = carryover.get("summary", "")
        if isinstance(summary_text, str) and summary_text and summary_text.lower() != "none":
            lines.append(f"Prior session: {summary_text}")

        carry_items = carryover.get("carry_forward", [])
        if isinstance(carry_items, list):
            for item in carry_items[:2]:
                lines.append(f"Carry forward: {_clip_text(item, max_chars=180)}")

        key_facts = carryover.get("key_facts", [])
        if isinstance(key_facts, list):
            for item in key_facts[:1]:
                lines.append(f"Relevant earlier detail: {_clip_text(item, max_chars=180)}")

        if not lines:
            for memory in retrieved_memories[:2]:
                clipped = _clip_text(memory, max_chars=180)
                if clipped:
                    lines.append(f"Relevant memory: {clipped}")

        return _dedupe_lines(lines)[:4]

    @staticmethod
    def _last_exchange_lines(session_history: list[dict[str, str]]) -> list[str]:
        if not session_history:
            return ["Session start."]

        lines: list[str] = []
        for item in session_history[-2:]:
            speaker = _history_speaker_label(str(item.get("speaker", "")))
            label = "Counselor" if speaker == "counselor" else "Client"
            text = _clip_text(str(item.get("text", "")), max_chars=420)
            if text:
                lines.append(f"{label}: {text}")
        return lines or ["Session start."]

    @classmethod
    def _describe_invalid_counselor_message(
        cls,
        message_value: Any,
        *,
        session_history: list[dict[str, str]],
    ) -> str | None:
        invalid_reason = describe_invalid_visible_message(message_value)
        if invalid_reason is not None:
            return invalid_reason
        message = str(message_value).strip()
        echo_reason = describe_echoing_message(
            message,
            cls._recent_speaker_turns(session_history, speaker="counselor", limit=2),
        )
        if echo_reason is not None:
            return echo_reason
        return None

    @staticmethod
    def _invalid_counselor_retry_guidance(invalid_reason: str) -> str:
        normalized = str(invalid_reason or "").strip().lower()
        if "not valid json" in normalized:
            return (
                "Your last draft was not valid JSON. "
                "Return only one JSON object with exactly one key named `message` and no extra wrapper text."
            )
        if "echoes prior wording too closely" in normalized:
            return (
                "Your last draft reused recent counselor wording too closely. "
                "Switch both the sentence stem and the counseling move. "
                "Do not restate the same reflection, reassurance, interpretation, summary point, experiment, or question from your own last 2-3 counselor turns with only light rewording. "
                "If the same point was already made, choose a different function, angle, blocker, or concrete focus instead of lightly rewording it. "
                "Respond to the client's state in a fresh way instead of repeating the same validation opener, phrasing, or step suggestion."
            )
        if "message must be a json string" in normalized:
            return (
                "Your last draft used the wrong type for `message`. "
                "Set `message` to normal English counseling text, not a number, list, or nested object."
            )
        if "message is empty" in normalized:
            return (
                "Your last draft did not produce a usable counseling reply. "
                "Retry with one brief, natural counselor utterance that directly responds to the client's latest state."
            )
        if "message must be english only" in normalized:
            return (
                "Your last draft included non-English text. "
                "Rewrite the visible reply fully in English and translate any source wording silently."
            )
        if "message must be natural language, not just numbers" in normalized or "message must include alphabetic english words" in normalized:
            return (
                "Your last draft was not natural conversational language. "
                "Retry with one brief English counseling reply, not numbers, labels, or symbol-only fragments."
            )
        return (
            "Your last draft did not satisfy the counselor-message constraints. "
            "Retry with one brief, natural English counseling reply that stays aligned with the current turn objective."
        )

    def build_prompt_messages(
        self,
        *,
        session_id: int,
        turn_id: int,
        target_turns: int,
        session_history: list[dict[str, str]],
        retrieved_memories: list[str],
        short_term_note: str,
    ) -> list[dict[str, str]]:
        is_final_turn = turn_id >= target_turns
        _, closing_window_start = _closing_window(target_turns)
        in_closing_window = turn_id >= closing_window_start
        is_penultimate_turn = target_turns > 1 and turn_id == target_turns - 1
        if turn_id == 1 and is_final_turn:
            turn_instruction = (
                "This is a single-turn session. Begin warmly, then provide a brief wrap-up "
                "and one concrete next step."
            )
        elif turn_id == 1:
            turn_instruction = "Open the session warmly."
        elif is_final_turn:
            turn_instruction = (
                "This reply will close the current session. End briefly and naturally, with a concise wrap-up and one realistic next step."
            )
        elif is_penultimate_turn:
            turn_instruction = (
                "This is the penultimate turn. Start the wrap-up: briefly summarize what feels clearer or still unresolved, "
                "and set up one realistic next step, but do not say goodbye yet."
            )
        elif in_closing_window:
            turn_instruction = (
                "This turn is inside the closing window. Narrow toward what feels clearer or still unresolved, "
                "but do not say goodbye yet."
            )
        else:
            turn_instruction = "Continue the session naturally. Do not wrap up or say goodbye yet."

        turn_policy = self._current_move_state(
            session_id=session_id,
            turn_id=turn_id,
            target_turns=target_turns,
            session_history=session_history,
        )
        turn_form_plan = self._turn_surface_plan(
            session_id=session_id,
            turn_id=turn_id,
            target_turns=target_turns,
            session_history=session_history,
            move_family=str(turn_policy.get("move_family", "") or ""),
        )
        turn_form_block = format_turn_form_block(turn_form_plan, counterpart_label="client's")
        client_summary_lines = self._client_summary_lines(short_term_note=short_term_note)
        earlier_summary_lines = self._earlier_conversation_summary_lines(
            session_history=session_history,
            retrieved_memories=retrieved_memories,
            short_term_note=short_term_note,
        )
        target_lines = []
        session_focus = turn_policy.get("session_focus", "")
        if session_focus:
            target_lines.append(f"Session focus: {session_focus}")
        current_target = turn_policy.get("current_target", "")
        if current_target:
            target_lines.append(f"Turn objective: {current_target}")
        constraint = turn_policy.get("constraint", "")
        if constraint:
            target_lines.append(f"Constraint: {constraint}")
        prior_step_lines = [_clip_text(item, max_chars=180) for item in self._prior_step_proposals(session_history)]
        retrieved_memory_lines = []
        for item in retrieved_memories[:4]:
            clipped = _clip_text(item, max_chars=180)
            if clipped:
                retrieved_memory_lines.append(clipped)
        closing_window_turns, closing_window_start = _closing_window(target_turns)
        if target_turns <= 1:
            closure_guard = "- This is the only turn, so keep any wrap-up brief and natural."
        elif closing_window_turns == 1:
            closure_guard = "- Do not wrap up the session, say goodbye, or talk as if the session is ending before the final turn."
        else:
            closure_guard = f"- No wrap-up or goodbye before turn {closing_window_start}."
        core_sections = [
            "\n".join(
                [
                    "# Core Directive",
                    "- Priority: satisfy <output_contract>; obey the turn form; stay in role; avoid repetition and premature closure.",
                    "- You are the counselor.",
                    "- Use the turn anchor for topic only; let the turn form block shape the wording.",
                    "- Privately choose one fresh counseling move; do not show that step.",
                    "- Do not paraphrase the client's latest line, compress it into an echo-question, or restate your own last 2-3 counselor turns with light rewording.",
                    "- Move one step forward: narrow the focus, name a blocker or consequence, mark an exception, make one observation, or set one next focus.",
                    "- Let the turn form decide statement versus question.",
                    "- Sound spoken and natural. Contractions and light discourse markers are fine.",
                    "- Keep the first sentence short. Do not use it for a full recap or a filler-only turn.",
                    closure_guard,
                ]
            ),
        ]
        if self.profile.style_prompt:
            core_sections.append(f"## Counseling Style\n- {self.profile.style_prompt}")
        if self.profile.boundaries_prompt:
            core_sections.append(f"## Boundaries\n- {self.profile.boundaries_prompt}")
        if self.profile.crisis_protocol:
            core_sections.append(f"## Crisis Protocol\n- {self.profile.crisis_protocol}")
        core_sections.append(self._format_bullet_section("Current turn anchor:", target_lines))

        context_sections = [
            "\n".join(
                [
                    "# Context",
                    f"- Session: {session_id}. Turn: {turn_id}.",
                    f"- Turn shape: {turn_instruction}",
                ]
            ),
            turn_form_block,
            self._format_bullet_section("Client summary:", client_summary_lines),
            self._format_bullet_section("Earlier conversation summary:", earlier_summary_lines),
        ]
        if retrieved_memory_lines:
            context_sections.append(self._format_bullet_section("Relevant prior-session memories:", retrieved_memory_lines))
        if prior_step_lines:
            context_sections.append(self._format_bullet_section("Prior step proposals already used this session:", prior_step_lines))
        supplemental_sections = [
            "# Supplemental Guardrails\n"
            "- Translate Chinese source text silently.\n"
            "- Use session history for continuity only; do not quote, mirror, or recap the client's wording.\n"
            "- If the turn form allows a question, use it to advance rather than restate.\n"
            "- Do not repeat a step suggestion already used in this session or open a brand-new topic because the plan feels stale.\n"
            "- Do not copy planning labels or placeholders into the reply.\n"
            "- Never output tokens like str, string, message, turn_form, turn_anchor, or output_contract."
        ]
        output_sections = [
            "# Hard Output Requirements\n"
            "- English only. Return one JSON object with key `message`.\n"
            "- Put only the final counselor reply in `message`. No extra text."
        ]
        final_reminder = (
            "# Final Reminder\n"
            "Obey <output_contract> first. Output only the JSON object."
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
                session_history, assistant_speaker="counselor"
            )
        else:
            chat_history = _build_chat_history_messages(session_history, assistant_speaker="counselor")
        if chat_history:
            prompt_messages.extend(chat_history)
        else:
            prompt_messages.append(
                {
                    "role": "user",
                    "content": f"Session {session_id} begins now. {turn_instruction}",
                }
            )
        if prompt_messages[-1]["role"] != "user":
            prompt_messages.append(
                {
                    "role": "user",
                    "content": (
                        "Move one step forward from the client's latest turn. "
                        "Do not restate the client's main claim, image, or wording. "
                        "If the turn form allows a question, use it to advance rather than restate. "
                        "Let the turn form decide statement versus question."
                    ),
                }
            )
        return prompt_messages

    def respond(
        self,
        *,
        session_id: int,
        turn_id: int,
        target_turns: int,
        session_history: list[dict[str, str]],
        retrieved_memories: list[str],
        short_term_note: str,
        temperature: float,
        max_tokens: int,
    ) -> CounselorTurn:
        base_prompt_messages = self.build_prompt_messages(
            session_id=session_id,
            turn_id=turn_id,
            target_turns=target_turns,
            session_history=session_history,
            retrieved_memories=retrieved_memories,
            short_term_note=short_term_note,
        )
        prompt_messages = list(base_prompt_messages)
        last_error: Exception | None = None
        _, closing_window_start = _closing_window(target_turns)
        closure_retry_rule = (
            "A natural ending is allowed, because this is the final turn."
            if turn_id >= target_turns
            else (
                f"A light wrap-up is allowed only if it fits, because this turn is already inside the closing window "
                f"that starts at turn {closing_window_start}. Do not say goodbye yet."
                if turn_id >= closing_window_start
                else "Do not wrap up, say goodbye, or talk as if the session is ending in this retry."
            )
        )

        for attempt in range(1, self.MAX_RESPONSE_ATTEMPTS + 1):
            raw = self.generator.generate(
                messages=prompt_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            message_value: Any = None
            try:
                parsed = extract_json_object(raw.text)
                message_value = parsed.get("message")
            except Exception:
                last_error = ValueError("Counselor response was not valid JSON")
            else:
                invalid_reason = self._describe_invalid_counselor_message(
                    message_value,
                    session_history=session_history,
                )
                if invalid_reason is None:
                    message = str(message_value).strip()
                    should_end = turn_id >= target_turns
                    reason = "hit_target_turns" if should_end else "ongoing"
                    return CounselorTurn(
                        message=message,
                        should_end=should_end,
                        reason=reason,
                        thinking=raw.thinking,
                        raw_message=raw.raw_message,
                    )
                last_error = ValueError(f"Counselor {invalid_reason}")

            self._log_invalid_output(
                {
                    "output_type": "counselor_turn",
                    "speaker": "Counselor",
                    "session_id": session_id,
                    "turn_id": turn_id,
                    "attempt": attempt,
                    "reason": str(last_error or "invalid output"),
                    "raw_text": str(raw.text),
                    "raw_message": raw.raw_message if isinstance(raw.raw_message, dict) else str(raw.raw_message),
                }
            )

            if attempt >= self.MAX_RESPONSE_ATTEMPTS:
                break
            retry_reason = str(last_error or "invalid output")
            targeted_retry_guidance = self._invalid_counselor_retry_guidance(retry_reason)
            prompt_messages = base_prompt_messages + [
                {
                    "role": "user",
                    "content": (
                        "Your previous output was invalid. "
                        f"Reason: {retry_reason}.\n"
                        f"{targeted_retry_guidance}\n"
                        "Return exactly one JSON object with key message. "
                        "Put the final natural English counseling reply in message. "
                        "Do not return Chinese text, numbers, or any non-string value for message. "
                        "Do not output planning labels, field names, or placeholders such as str, string, turn_form, turn_anchor, or output_contract. "
                        "Do not reuse recent counselor wording. "
                        "Follow this priority order: output format first; then obey the turn form block; then avoid repetition, meta talk, and premature closure; then stay within the counselor role and style; then use the turn anchor as a flexible target. "
                        "Do not turn the client's latest line into a recap or echo-question. "
                        "Compared with the invalid draft and your own last 2-3 counselor turns, do not restate the same reflection, reassurance, interpretation, summary point, experiment, or question with only light rewording. "
                        "If the same topic is still active, move one step forward by narrowing it, checking a blocker, naming a consequence, marking an exception, making one brief observation, or setting one next focus instead of replaying it. "
                        "Let the turn form decide statement versus question. "
                        "Do not expose private reasoning. "
                        f"{closure_retry_rule} "
                        "Do not repeat a step suggestion already used in this session; switch to empathy, clarification, reflective listening, blocker exploration, or summary instead. "
                        "Start fresh from the latest conversation state; do not reuse wording or structure from the invalid draft."
                    ),
                }
            ]

        raise ValueError(
            f"{last_error or ValueError('Counselor message is invalid')} after "
            f"{self.MAX_RESPONSE_ATTEMPTS} consecutive attempts"
        ) from last_error

    def summarize_session(
        self,
        *,
        session_id: int | None = None,
        events: list[dict[str, Any]],
        prior_short_term_note: str = "",
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        transcript = "\n".join(f"{event['speaker']}: {event['text']}" for event in events)
        raw = self.generator.generate(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Update the longitudinal continuation note using the prior carryover note and the newly completed session. "
                        "Return JSON with keys: summary, key_facts, goals, homework, risk_flags. "
                        "Preserve still-relevant prior context, revise outdated details, and keep each list concise."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Previous longitudinal note:\n{prior_short_term_note or 'none'}\n\n"
                        f"Completed session transcript:\n{transcript}"
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            vertex_response_schema=_SESSION_SUMMARY_VERTEX_SCHEMA,
        )
        try:
            return parse_summary_output(raw.text)
        except Exception as exc:
            self._log_invalid_output(
                {
                    "output_type": "session_summary",
                    "speaker": "Counselor",
                    "session_id": session_id,
                    "attempt": 1,
                    "reason": str(exc),
                    "raw_text": str(raw.text),
                    "raw_message": raw.raw_message if isinstance(raw.raw_message, dict) else str(raw.raw_message),
                }
            )
            raise
