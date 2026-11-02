from __future__ import annotations

import argparse
import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from EmoTrack.data.session_latent import extract_session_num, infer_latent_profile, stage_lookup, to_stage_en
from EmoTrack.utils.paths import REPO_ROOT, repo_relative


SOURCE_PROFILES_ROOT = REPO_ROOT / "profiles"
DEFAULT_CONVERSATION_PROFILE_ROOT = REPO_ROOT / "data" / "conversation_profiles"
CONVERSATION_PROFILE_SCHEMA_VERSION = 3

CLIENT_MASTER_BEAT_BLUEPRINTS = (
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

SOUND_SKILL_PATTERN = re.compile(
    r"(声音|音源|噪音|喇叭|音乐|耳痛|耳朵|sound|noise|audio|listening|exposure|listen)",
    flags=re.IGNORECASE,
)
LOW_SIGNAL_FRAGMENT_PATTERN = re.compile(
    r"^(?:\d+(?:\.\d+)?|"
    r"(?:name|age|gender|occupation|educational_background|marital_status|family_status|"
    r"social_status|medical_history|language_features|main_problem|core_demands|topic)\s*:)",
    flags=re.IGNORECASE,
)

PROCESS_META_TOKENS = (
    "保密",
    "知情同意",
    "权利义务",
    "权利与义务",
    "咨询设置",
    "咨询事项",
    "规则",
    "流程",
    "登记表",
    "登记",
    "量表",
    "测评",
    "测验",
    "基线",
    "电子版链接",
    "提醒",
    "家庭作业",
    "作业",
    "会后",
    "每周会谈安排",
    "节奏可调",
    "寒暄",
    "关系维护",
    "联盟质量",
    "信任与合作",
    "合作增强",
    "放心",
    "督导",
    "研究",
    "频次",
    "时长",
    "费用",
    "预约",
    "签署",
    "填写",
    "协议",
    "初步评估为主",
    "本次会谈",
    "下次会谈",
    "咨询师",
    "问题概念化",
    "目标设定",
    "心理教育",
    "abc",
    "hq",
    "sas",
    "任务",
    "引导性任务",
    "探索基础信息",
    "明确主诉",
    "会谈收尾",
    "下次铺垫",
    "共同构建",
    "形成模型",
    "行为化探索",
    "在此阶段",
    "为后续",
    "以开放式",
    "合作式方式",
    "功能分析",
    "靶行为",
    "基于行为主义",
)
EVENT_SCENE_TOKENS = (
    "音乐会",
    "超市",
    "商场",
    "教室",
    "课堂",
    "会议",
    "会议室",
    "咖啡店",
    "电影院",
    "会议厅",
    "门口",
    "小区",
    "路边",
    "临街",
    "车道",
    "音箱",
    "播报",
    "喇叭",
    "车流",
    "场景",
    "空间",
    "那次",
    "上次",
    "最近",
)
TRIGGER_BODY_TOKENS = (
    "刺痛",
    "疼痛",
    "痛感",
    "耳鸣",
    "发麻",
    "胸口",
    "心跳",
    "胃",
    "呼吸",
    "紧绷",
    "绷住",
    "惊跳",
    "僵住",
    "捂耳",
    "快步离开",
    "身体",
    "反应",
    "警觉",
    "预警",
    "bracing",
    "panic",
    "ache",
)
FEAR_MEANING_TOKENS = (
    "担心",
    "害怕",
    "会不会",
    "越来越严重",
    "治不好",
    "恢复不了",
    "失控",
    "意味着",
    "灾难",
    "尴尬",
    "未来",
    "职业",
    "工作保不住",
    "安全",
    "伤害",
    "没法恢复",
)
AVOIDANCE_COST_TOKENS = (
    "回避",
    "不能出门",
    "不敢出门",
    "工作受影响",
    "生活受限",
    "错过",
    "限制",
    "离开",
    "逃开",
    "退缩",
    "受影响",
    "没法教学",
    "不去",
    "不参加",
    "关系",
    "社交",
    "自由",
    "routine",
    "cost",
)
HELP_GOAL_TOKENS = (
    "希望",
    "想要",
    "需要帮助",
    "希望通过咨询",
    "改变",
    "目标",
    "下一步",
    "支持",
    "更安全",
    "可行",
    "可控",
    "如何开始",
    "恢复",
    "重新",
)
SKILL_PROCESS_TOKENS = (
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
SKILL_GENERIC_RELATION_TOKENS = (
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
SKILL_ACTION_TOKENS = (
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
class PreparedConversationProfile:
    profile_id: str
    source_path: Path
    client_profile_path: Path
    hidden_state_path: Path
    metadata_path: Path


@dataclass(frozen=True)
class PreparedConversationProfileBatch:
    manifest_path: Path
    prepared_profiles: tuple[PreparedConversationProfile, ...]


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _safe_slug(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(text or "").strip().lower())
    return normalized.strip("_") or "profile"


def _flatten_text_fragments(value: Any) -> list[str]:
    out: list[str] = []
    if isinstance(value, str):
        normalized = _normalize_space(value)
        if normalized:
            out.append(normalized)
        return out
    if isinstance(value, list):
        for item in value:
            out.extend(_flatten_text_fragments(item))
        return out
    if isinstance(value, dict):
        for item in value.values():
            out.extend(_flatten_text_fragments(item))
        return out
    return out


def _dedupe_texts(items: list[str]) -> list[str]:
    deduped: list[str] = []
    for item in items:
        normalized = _normalize_space(item)
        if not normalized or normalized in deduped:
            continue
        deduped.append(normalized)
    return deduped


def _contains_any(text: str, tokens: tuple[str, ...]) -> bool:
    normalized = _normalize_space(text).lower()
    return any(token.lower() in normalized for token in tokens)


def _is_low_signal_fragment(text: str) -> bool:
    normalized = _normalize_space(text)
    if not normalized:
        return True
    return LOW_SIGNAL_FRAGMENT_PATTERN.match(normalized) is not None


def _is_process_meta_fragment(text: str) -> bool:
    normalized = _normalize_space(text)
    if not normalized:
        return True
    if _is_low_signal_fragment(normalized):
        return True
    return _contains_any(normalized, PROCESS_META_TOKENS)


def _client_beat_index_for_fragment(text: str) -> int | None:
    normalized = _normalize_space(text)
    if not normalized or _is_process_meta_fragment(normalized):
        return None

    scores = [
        sum(token.lower() in normalized.lower() for token in EVENT_SCENE_TOKENS),
        sum(token.lower() in normalized.lower() for token in TRIGGER_BODY_TOKENS),
        sum(token.lower() in normalized.lower() for token in FEAR_MEANING_TOKENS),
        sum(token.lower() in normalized.lower() for token in AVOIDANCE_COST_TOKENS),
        sum(token.lower() in normalized.lower() for token in HELP_GOAL_TOKENS),
    ]
    if max(scores) <= 0:
        return None
    return max(range(len(scores)), key=lambda idx: scores[idx])


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


def _maybe_relative(path: Path) -> str:
    return repo_relative(path)


def _profile_sort_key(path: Path) -> tuple[str, int, str]:
    parent = path.parent.name.lower()
    stem = path.stem
    if stem.isdigit():
        return parent, int(stem), ""
    return parent, 10**9, stem.lower()


def discover_source_profile_paths(profiles_root: Path | None = None) -> list[Path]:
    root = (profiles_root or SOURCE_PROFILES_ROOT).resolve()
    candidates = sorted(root.glob("*/*.json"), key=_profile_sort_key)
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen or "seeds" in resolved.parts:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return deduped


def profile_id_from_source_path(source_path: Path, profiles_root: Path | None = None) -> str:
    root = (profiles_root or SOURCE_PROFILES_ROOT).resolve()
    resolved = source_path.resolve()
    relative = resolved.relative_to(root)
    group = _safe_slug(relative.parent.name)
    stem = relative.stem
    sample = f"{int(stem):03d}" if stem.isdigit() else _safe_slug(stem)
    return f"{group}_{sample}"


def prepare_source_profile_manifest(
    *,
    profiles_root: Path | None = None,
    output_root: Path | None = None,
) -> Path:
    root = (profiles_root or SOURCE_PROFILES_ROOT).resolve()
    out_root = (output_root or DEFAULT_CONVERSATION_PROFILE_ROOT).resolve()
    manifest_path = out_root / "manifest.json"
    manifest = {
        "profiles_root": _maybe_relative(root),
        "profile_count": 0,
        "profiles": [],
    }
    for source_path in discover_source_profile_paths(root):
        manifest["profiles"].append(
            {
                "profile_id": profile_id_from_source_path(source_path, root),
                "source_path": _maybe_relative(source_path),
            }
        )
    manifest["profile_count"] = len(manifest["profiles"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _source_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "source_path": _maybe_relative(path),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
    }


def _metadata_is_fresh(metadata: dict[str, Any], source_path: Path) -> bool:
    fingerprint = _source_fingerprint(source_path)
    return (
        metadata.get("schema_version") == CONVERSATION_PROFILE_SCHEMA_VERSION
        and
        metadata.get("source_path") == fingerprint["source_path"]
        and metadata.get("source_size") == fingerprint["source_size"]
        and metadata.get("source_mtime_ns") == fingerprint["source_mtime_ns"]
    )


def _copy_profile_block(sample: dict[str, Any], key: str) -> dict[str, Any]:
    value = sample.get(key, {})
    if not isinstance(value, dict):
        return {}
    return copy.deepcopy(value)


def _strip_counselor_only_fields(profile_block: dict[str, Any]) -> dict[str, Any]:
    cleaned = copy.deepcopy(profile_block) if isinstance(profile_block, dict) else {}
    cleaned.pop("core_demands", None)
    return cleaned


def _target_behavior_entries(client_info: dict[str, Any]) -> list[dict[str, Any]]:
    raw = client_info.get("target_behavior")
    if isinstance(raw, dict):
        return [raw]
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    return []


def _client_fallback_anchor_groups(client_info: dict[str, Any], intake_profile: dict[str, Any]) -> list[list[str]]:
    baseline = _normalize_space(client_info.get("main_problem") or intake_profile.get("main_problem"))
    topic = _normalize_space(client_info.get("topic") or intake_profile.get("topic"))
    growth = _flatten_text_fragments(client_info.get("growth_experiences"))
    beliefs = _flatten_text_fragments(client_info.get("core_beliefs"))
    demands = _flatten_text_fragments(
        [
            client_info.get("core_demands"),
            intake_profile.get("core_demands"),
        ]
    )

    triggers: list[str] = []
    behaviors: list[str] = []
    reasons: list[str] = []
    functions: list[str] = []
    consequences: list[str] = []
    for entry in _target_behavior_entries(client_info):
        behaviors.extend(_flatten_text_fragments(entry.get("behavior")))
        triggers.extend(_flatten_text_fragments(entry.get("antecedent")))
        reasons.extend(_flatten_text_fragments(entry.get("core_reason")))
        functions.extend(_flatten_text_fragments(entry.get("function")))
        consequences.extend(_flatten_text_fragments(entry.get("consequence")))

    return [
        _dedupe_texts([baseline, *growth, *triggers]),
        _dedupe_texts([*behaviors, *triggers, baseline]),
        _dedupe_texts([*reasons, *beliefs, baseline]),
        _dedupe_texts([*functions, *consequences, topic, baseline]),
        _dedupe_texts([*demands, topic, *consequences, baseline]),
    ]


def _session_blueprints(sample: dict[str, Any], *, max_session_num: int = 6) -> list[dict[str, Any]]:
    merged: dict[int, dict[str, Any]] = {}

    for stage in sample.get("global_plan", []):
        if not isinstance(stage, dict):
            continue
        content = stage.get("content", {})
        if not isinstance(content, dict):
            continue
        for session_key, entry in content.items():
            session_num = extract_session_num(str(session_key))
            if session_num <= 0 or session_num > max_session_num or not isinstance(entry, dict):
                continue
            target = merged.setdefault(
                session_num,
                {
                    "session_number": session_num,
                    "theme": "",
                    "persona_links": [],
                    "case_material": [],
                    "rationale": [],
                    "session_goals": [],
                },
            )
            if not target["theme"]:
                target["theme"] = _normalize_space(entry.get("theme"))
            target["persona_links"].extend(_flatten_text_fragments(entry.get("persona_links")))
            target["case_material"].extend(_flatten_text_fragments(entry.get("case_material")))
            target["rationale"].extend(_flatten_text_fragments(entry.get("rationale")))

    sessions = sample.get("sessions", [])
    if isinstance(sessions, list):
        for idx, session_entry in enumerate(sessions, start=1):
            if not isinstance(session_entry, dict):
                continue
            try:
                session_num = int(session_entry.get("session_number", idx))
            except Exception:
                session_num = idx
            if session_num > max_session_num:
                continue
            target = merged.setdefault(
                session_num,
                {
                    "session_number": session_num,
                    "theme": "",
                    "persona_links": [],
                    "case_material": [],
                    "rationale": [],
                    "session_goals": [],
                },
            )
            target["session_goals"].extend(_flatten_text_fragments(session_entry.get("session_goals")))
            summary = session_entry.get("session_summary")
            if isinstance(summary, dict):
                for value in summary.values():
                    target["rationale"].extend(_flatten_text_fragments(value))

    return [merged[key] for key in sorted(merged)]


def _session_source_fragments(entry: dict[str, Any]) -> list[str]:
    if not isinstance(entry, dict):
        return []
    return _dedupe_texts(
        [
            *_flatten_text_fragments(entry.get("session_goals")),
            *_flatten_text_fragments(entry.get("case_material")),
            *_flatten_text_fragments(entry.get("persona_links")),
            *_flatten_text_fragments(entry.get("rationale")),
        ]
    )


def _client_session_anchor_pools(
    entry: dict[str, Any],
    fallback_groups: list[list[str]],
) -> list[list[str]]:
    pools: list[list[str]] = [[] for _ in CLIENT_MASTER_BEAT_BLUEPRINTS]
    for item in _session_source_fragments(entry):
        beat_index = _client_beat_index_for_fragment(item)
        if beat_index is None:
            continue
        pools[beat_index].append(item)

    cleaned: list[list[str]] = []
    for idx in range(len(CLIENT_MASTER_BEAT_BLUEPRINTS)):
        anchors = _dedupe_texts([*pools[idx], *(fallback_groups[idx] if idx < len(fallback_groups) else [])])[:5]
        cleaned.append(anchors)
    return cleaned


def _build_client_session_master_plan(
    sample: dict[str, Any],
    *,
    max_sessions: int = 6,
) -> list[dict[str, Any]]:
    client_info = _copy_profile_block(sample, "client_info")
    intake_profile = _copy_profile_block(sample, "intake_profile")
    fallback_groups = _client_fallback_anchor_groups(client_info, intake_profile)
    plans: list[dict[str, Any]] = []
    for entry in _session_blueprints(sample, max_session_num=max_sessions):
        theme = _normalize_space(entry.get("theme")) or f"Session {entry.get('session_number')}"
        anchor_pools = _client_session_anchor_pools(entry, fallback_groups)
        beats: list[dict[str, Any]] = []
        for idx, (title, focus) in enumerate(CLIENT_MASTER_BEAT_BLUEPRINTS):
            anchors = anchor_pools[idx] if idx < len(anchor_pools) else []
            beats.append(
                {
                    "title": title,
                    "focus": f"{focus} Keep it aligned with the current session theme: {theme}.",
                    "anchors": anchors,
                }
            )
        plans.append(
            {
                "session_number": int(entry.get("session_number", 0) or 0),
                "theme": theme,
                "beats": beats,
            }
        )
    return plans


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
    if _contains_any(public_text, SKILL_PROCESS_TOKENS):
        return False
    if _contains_any(action_text, SKILL_GENERIC_RELATION_TOKENS) and not _contains_any(action_text, SKILL_ACTION_TOKENS):
        return False
    return True


def _skill_record(
    *,
    session_number: int,
    skill_id: str,
    skill_name: str,
    skill_description: str,
    when_to_use: str,
    trigger: str,
    meta_skill: str,
    runtime_usable: bool = True,
) -> dict[str, Any]:
    public_text = " ".join([skill_name, skill_description, when_to_use, trigger])
    return {
        "session_number": session_number,
        "skill_id": skill_id,
        "skill_name": skill_name,
        "skill_description": skill_description,
        "when_to_use": when_to_use,
        "trigger": trigger,
        "meta_skill": meta_skill,
        "bucket": _infer_skill_bucket(public_text),
        "is_exposure_like": SOUND_SKILL_PATTERN.search(public_text) is not None
        or any(token in public_text for token in ("暴露", "系统脱敏", "音源", "放松训练")),
        "runtime_usable": runtime_usable,
    }


def _augment_skill_bank_from_plan(
    sample: dict[str, Any],
    *,
    existing_count: int,
    limit: int,
) -> list[dict[str, Any]]:
    augmented: list[dict[str, Any]] = []
    if existing_count >= limit:
        return augmented
    client_info = _copy_profile_block(sample, "client_info")
    intake_profile = _copy_profile_block(sample, "intake_profile")
    fallback_groups = _client_fallback_anchor_groups(client_info, intake_profile)
    skill_templates = (
        (
            "澄清一个最近的具体触发场景",
            "邀请来访者描述一个最近发生的具体声音场景，补足地点、时刻与发生经过。",
            "assessment",
        ),
        (
            "梳理触发与身体预警",
            "帮助来访者说清触发模式、身体预警与当下的警觉反应。",
            "assessment",
        ),
        (
            "澄清灾难化预测或脆弱解释",
            "探索来访者最担心会发生什么，以及声音痛感背后的灾难化意义。",
            "formulation",
        ),
        (
            "评估回避与功能代价",
            "梳理回避如何短期保护来访者，以及它在工作、生活和关系中的代价。",
            "formulation",
        ),
        (
            "协商一个贴近主诉的安全下一步",
            "把会谈中的理解转成一个紧贴主诉、压力更低的下一步。",
            "change",
        ),
    )
    for entry in _session_blueprints(sample, max_session_num=6):
        session_number = int(entry.get("session_number", 0) or 0)
        theme = _normalize_space(entry.get("theme")) or f"Session {session_number}"
        anchor_pools = _client_session_anchor_pools(entry, fallback_groups)
        for idx, (skill_name, skill_description, bucket) in enumerate(skill_templates, start=1):
            if existing_count + len(augmented) >= limit:
                return augmented
            anchor_candidates = anchor_pools[idx - 1] if idx - 1 < len(anchor_pools) else []
            if not anchor_candidates:
                continue
            anchor = anchor_candidates[0]
            augmented.append(
                _skill_record(
                    session_number=session_number,
                    skill_id=f"plan_{session_number}_{idx}",
                    skill_name=skill_name,
                    skill_description=f"{skill_description} Anchor it to this session material: {anchor}",
                    when_to_use=f"Use when the session theme is '{theme}' and the public transcript is ready for this focus.",
                    trigger=f"Plan-derived anchor: {anchor}",
                    meta_skill=f"Backfilled from session theme: {theme}",
                    runtime_usable=True,
                )
            )
            augmented[-1]["bucket"] = bucket
    return augmented


def _build_counselor_genre_skill_bank(sample: dict[str, Any], *, limit: int = 30) -> list[dict[str, Any]]:
    sessions = sample.get("sessions", [])
    if not isinstance(sessions, list):
        sessions = []

    skill_bank: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for idx, session in enumerate(sessions, start=1):
        if not isinstance(session, dict):
            continue
        try:
            session_number = int(session.get("session_number", idx))
        except Exception:
            session_number = idx
        suggest_skills = session.get("suggest_skills", [])
        if not isinstance(suggest_skills, list):
            continue
        for entry in suggest_skills:
            if not isinstance(entry, dict):
                continue
            meta_skill = _normalize_space(entry.get("meta_skill"))
            micro_skills = entry.get("micro_skills", [])
            if not isinstance(micro_skills, list):
                continue
            for micro in micro_skills:
                if not isinstance(micro, dict):
                    continue
                skill_name = _normalize_space(micro.get("skill_name"))
                skill_id = _normalize_space(micro.get("skill_id"))
                if not skill_name:
                    continue
                skill_description = _normalize_space(micro.get("skill_description"))
                when_to_use = _normalize_space(micro.get("when_to_use"))
                trigger = _normalize_space(micro.get("trigger"))
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
                skill_bank.append(
                    _skill_record(
                        session_number=session_number,
                        skill_id=skill_id,
                        skill_name=skill_name,
                        skill_description=skill_description,
                        when_to_use=when_to_use,
                        trigger=trigger,
                        meta_skill=meta_skill,
                        runtime_usable=runtime_usable,
                    )
                )
                seen.add(key)
                if len(skill_bank) >= limit:
                    return skill_bank

    for skill in _augment_skill_bank_from_plan(sample, existing_count=len(skill_bank), limit=limit):
        key = (str(skill.get("skill_id") or skill.get("skill_name")), str(skill.get("skill_name")))
        if key in seen:
            continue
        skill_bank.append(skill)
        seen.add(key)
        if len(skill_bank) >= limit:
            break
    return skill_bank


def _core_demands(sample: dict[str, Any], intake_profile: dict[str, Any], client_info: dict[str, Any]) -> str:
    candidates = [
        intake_profile.get("core_demands"),
        client_info.get("core_demands"),
        sample.get("core_demands"),
    ]
    for candidate in candidates:
        text = _normalize_space(candidate)
        if text:
            return text
    return ""


def _session_plan_outline(stage_content: dict[str, Any], *, max_session_num: int | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for session_key in sorted(stage_content, key=extract_session_num):
        session_num = extract_session_num(session_key)
        if max_session_num is not None and session_num > max_session_num:
            continue
        item = stage_content.get(session_key, {})
        entry: dict[str, Any] = {"session_key": str(session_key)}
        if isinstance(item, dict):
            theme = _normalize_space(item.get("theme"))
            if theme:
                entry["theme"] = theme
        out.append(entry)
    return out


def _global_plan_outline(sample: dict[str, Any], *, max_session_num: int = 6) -> list[dict[str, Any]]:
    stages = sample.get("global_plan", [])
    if not isinstance(stages, list):
        return []

    out: list[dict[str, Any]] = []
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        entry: dict[str, Any] = {}
        if stage.get("stage_number") is not None:
            entry["stage_number"] = stage.get("stage_number")
        stage_name = _normalize_space(stage.get("stage_name"))
        if stage_name:
            entry["stage_name"] = stage_name
        sessions = _normalize_space(stage.get("sessions"))
        if sessions:
            entry["sessions"] = sessions

        content = stage.get("content", {})
        if isinstance(content, dict):
            session_plan = _session_plan_outline(content, max_session_num=max_session_num)
            if session_plan:
                entry["session_plan"] = session_plan

        if entry.get("session_plan"):
            out.append(entry)
    return out


def _build_counselor_case_context(
    sample: dict[str, Any],
    intake_profile: dict[str, Any],
    client_info: dict[str, Any],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    core_demands = _core_demands(sample, intake_profile, client_info)
    if core_demands:
        out["core_demands"] = core_demands
    global_plan = _global_plan_outline(sample)
    if global_plan:
        out["global_plan"] = global_plan
    return out


def _build_conversation_profile(sample: dict[str, Any], source_path: Path) -> dict[str, Any]:
    raw_client_info = _copy_profile_block(sample, "client_info")
    raw_intake_profile = _copy_profile_block(sample, "intake_profile")
    if not raw_intake_profile and not raw_client_info:
        raise RuntimeError(f"Profile {source_path} does not provide a valid intake_profile or client_info object")

    counselor_case_context = _build_counselor_case_context(sample, raw_intake_profile, raw_client_info)
    client_info = _strip_counselor_only_fields(raw_client_info)
    intake_profile = _strip_counselor_only_fields(raw_intake_profile)
    theoretical = str(sample.get("theoretical", source_path.parent.name) or source_path.parent.name).strip()
    payload = {
        "client_id": sample.get("client_id", ""),
        "theoretical": theoretical,
        "previous_conversations": [],
        "meta": {
            "source_profile_path": _maybe_relative(source_path),
            "source_client_id": sample.get("client_id", ""),
            "source_theoretical": theoretical,
        },
    }
    if intake_profile:
        payload["intake_profile"] = intake_profile
    if client_info:
        payload["client_info"] = client_info
    if counselor_case_context:
        payload["counselor_case_context"] = counselor_case_context
    client_session_master_plan = _build_client_session_master_plan(sample)
    if client_session_master_plan:
        payload["client_session_master_plan"] = client_session_master_plan
    counselor_genre_skill_bank = _build_counselor_genre_skill_bank(sample)
    if counselor_genre_skill_bank:
        payload["counselor_genre_skill_bank"] = counselor_genre_skill_bank
    return payload


def ensure_conversation_profile_artifacts(
    source_path: Path,
    *,
    output_root: Path | None = None,
    profiles_root: Path | None = None,
    force: bool = False,
) -> PreparedConversationProfile:
    source = source_path.resolve()
    root = (profiles_root or SOURCE_PROFILES_ROOT).resolve()
    out_root = (output_root or DEFAULT_CONVERSATION_PROFILE_ROOT).resolve()
    profile_id = profile_id_from_source_path(source, root)

    client_profile_path = out_root / "client" / f"{profile_id}.json"
    hidden_state_path = out_root / "states" / f"{profile_id}.json"
    metadata_path = out_root / "metadata" / f"{profile_id}.json"

    prepared = PreparedConversationProfile(
        profile_id=profile_id,
        source_path=source,
        client_profile_path=client_profile_path,
        hidden_state_path=hidden_state_path,
        metadata_path=metadata_path,
    )

    if not force and metadata_path.exists() and client_profile_path.exists() and hidden_state_path.exists():
        metadata = _load_json(metadata_path)
        if _metadata_is_fresh(metadata, source):
            return prepared

    sample = _load_json(source)
    conversation_profile = _build_conversation_profile(sample, source)

    sessions = sample.get("sessions", [])
    if not isinstance(sessions, list) or not sessions:
        raise RuntimeError(f"Profile {source} does not provide any sessions")
    first_session = sessions[0]
    if not isinstance(first_session, dict):
        raise RuntimeError(f"Profile {source} has an invalid first session payload")

    stage_map = stage_lookup(sample)
    first_session_num = int(first_session.get("session_number", 1))
    first_stage = stage_map.get(
        first_session_num,
        {"stage_number": 2, "stage_name": "Core Intervention", "sessions": "", "plan_session_content": {}},
    )
    hidden_state = infer_latent_profile(first_session, first_stage, 0)

    for path in (client_profile_path, hidden_state_path, metadata_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    client_profile_path.write_text(json.dumps(conversation_profile, ensure_ascii=False, indent=2), encoding="utf-8")
    hidden_state_path.write_text(json.dumps(hidden_state, ensure_ascii=False, indent=2), encoding="utf-8")

    metadata = {
        "profile_id": profile_id,
        "schema_version": CONVERSATION_PROFILE_SCHEMA_VERSION,
        **_source_fingerprint(source),
        "client_profile_path": _maybe_relative(client_profile_path),
        "hidden_state_path": _maybe_relative(hidden_state_path),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return prepared


def ensure_all_conversation_profile_artifacts(
    *,
    profiles_root: Path | None = None,
    output_root: Path | None = None,
    force: bool = False,
    limit: int | None = None,
) -> PreparedConversationProfileBatch:
    root = (profiles_root or SOURCE_PROFILES_ROOT).resolve()
    out_root = (output_root or DEFAULT_CONVERSATION_PROFILE_ROOT).resolve()
    manifest_path = prepare_source_profile_manifest(profiles_root=root, output_root=out_root)
    source_paths = discover_source_profile_paths(root)
    if limit is not None:
        source_paths = source_paths[:limit]

    prepared_profiles = tuple(
        ensure_conversation_profile_artifacts(
            source_path,
            output_root=out_root,
            profiles_root=root,
            force=force,
        )
        for source_path in source_paths
    )
    return PreparedConversationProfileBatch(
        manifest_path=manifest_path,
        prepared_profiles=prepared_profiles,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare generated conversation profiles from source profiles.")
    parser.add_argument("--profiles-root", type=Path, default=SOURCE_PROFILES_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_CONVERSATION_PROFILE_ROOT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    batch = ensure_all_conversation_profile_artifacts(
        profiles_root=args.profiles_root.expanduser().resolve(),
        output_root=args.output_root.expanduser().resolve(),
        force=args.force,
        limit=args.limit,
    )
    print(
        "[profiles] prepared "
        f"{len(batch.prepared_profiles)} profiles manifest={batch.manifest_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
