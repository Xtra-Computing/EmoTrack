from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from EmoTrack.utils.dialogue_text import normalize_space as _normalize_space
from EmoTrack.utils.paths import REPO_ROOT


_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_QUESTION_END_RE = re.compile(r"\?\s*['\")\]]*\s*$")
_STOPWORDS = {
    "a",
    "about",
    "after",
    "all",
    "also",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "for",
    "from",
    "get",
    "got",
    "had",
    "has",
    "have",
    "having",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "not",
    "of",
    "on",
    "or",
    "our",
    "ours",
    "ourselves",
    "out",
    "she",
    "so",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "to",
    "too",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
}
_LOW_VALUE_CONTENT_WORDS = {
    "actually",
    "again",
    "always",
    "anything",
    "better",
    "even",
    "everyone",
    "everything",
    "feels",
    "feel",
    "feeling",
    "felt",
    "going",
    "hard",
    "harder",
    "just",
    "kind",
    "know",
    "like",
    "maybe",
    "normal",
    "people",
    "probably",
    "really",
    "right",
    "something",
    "still",
    "things",
    "think",
    "trying",
    "wrong",
}

SURFACE_STAGE_ORDER = ("early", "mid_early", "mid_late", "late")


@dataclass(frozen=True)
class TurnFormPlan:
    speaker: str
    stage_label: str
    form_label: str
    source_form_id: str
    sentence_word_counts: tuple[int, ...]
    starter: str
    end_with_question: bool
    max_question_count: int
    target_overlap_ratio: float
    reuse_budget_content_words: int
    allowed_shared_words: tuple[str, ...]


def stable_fraction(*parts: Any) -> float:
    seed = "||".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(seed).hexdigest()
    return int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)


def stable_choice(items: tuple[str, ...], *parts: Any) -> str:
    if not items:
        raise ValueError("stable_choice requires at least one item")
    idx = min(len(items) - 1, int(stable_fraction(*parts) * len(items)))
    return str(items[idx])


def _quantile(values: list[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(int(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    q_clamped = max(0.0, min(1.0, float(q)))
    position = q_clamped * (len(ordered) - 1)
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    weight = position - lower
    blended = (ordered[lower] * (1.0 - weight)) + (ordered[upper] * weight)
    return int(round(blended))


def english_word_tokens(text: str) -> list[str]:
    return [token.lower() for token in _WORD_RE.findall(str(text or ""))]


def message_ends_with_question(text: str) -> bool:
    return bool(_QUESTION_END_RE.search(_normalize_space(text)))


def surface_stage_for_turn(turn_id: int, target_turns: int | None) -> str:
    total = max(1, int(target_turns or 20))
    index = max(0, min(total - 1, int(turn_id) - 1))
    stage_idx = min(len(SURFACE_STAGE_ORDER) - 1, int(index * len(SURFACE_STAGE_ORDER) / total))
    return SURFACE_STAGE_ORDER[stage_idx]


def select_allowed_shared_words(text: str, budget: int) -> tuple[str, ...]:
    unique_words: list[str] = []
    seen: set[str] = set()
    fallback_words: list[str] = []
    fallback_seen: set[str] = set()
    for token in english_word_tokens(text):
        if len(token) <= 2 or token in _STOPWORDS:
            continue
        if token not in fallback_seen:
            fallback_words.append(token)
            fallback_seen.add(token)
        if token in _LOW_VALUE_CONTENT_WORDS or token in seen:
            continue
        unique_words.append(token)
        seen.add(token)

    selected = unique_words[: max(0, int(budget))]
    if len(selected) < max(0, int(budget)):
        for token in fallback_words:
            if token in selected:
                continue
            selected.append(token)
            if len(selected) >= max(0, int(budget)):
                break
    return tuple(selected)


def _asset_path() -> Path:
    return REPO_ROOT / "configs" / "realcbt_stats" / "realcbt_surface_control.json"


@lru_cache(maxsize=1)
def load_realcbt_surface_asset() -> dict[str, Any]:
    path = _asset_path()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid RealCBT surface control asset at {path}")
    return data


def _promptv3_reference_path() -> Path:
    return REPO_ROOT / "configs" / "realcbt_stats" / "promptv3_surface_reference.json"


@lru_cache(maxsize=1)
def load_promptv3_surface_reference() -> dict[str, Any]:
    path = _promptv3_reference_path()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid promptv3 surface reference at {path}")
    return data


def _promptv3_reference_key(*, speaker: str, turn_id: int, prev_question: bool) -> str:
    return f"{str(speaker).strip().lower()}|t{int(turn_id):02d}|prevq{int(bool(prev_question))}"


def _speaker_asset(speaker: str, stage_label: str, prev_question: bool) -> dict[str, Any]:
    asset = load_realcbt_surface_asset()
    speaker_key = str(speaker).strip().lower()
    context_key = "prev_q" if prev_question else "prev_nq"
    try:
        return asset["speakers"][speaker_key]["stages"][stage_label]["contexts"][context_key]
    except Exception as exc:  # pragma: no cover - fail fast if asset is malformed
        raise KeyError(
            f"Missing RealCBT surface asset entry for speaker={speaker_key}, stage={stage_label}, context={context_key}"
        ) from exc


def _context_key(prev_question: bool) -> str:
    return "prev_q" if prev_question else "prev_nq"


@lru_cache(maxsize=1)
def _stage_surface_profiles() -> dict[str, dict[str, Any]]:
    asset = load_realcbt_surface_asset()
    profiles: dict[str, dict[str, Any]] = {}
    for stage_label in SURFACE_STAGE_ORDER:
        forms: list[dict[str, Any]] = []
        sentence_counts: list[int] = []
        total_words: list[int] = []
        first_sentence_words: list[int] = []
        later_sentence_words: list[int] = []
        single_sentence_words: list[int] = []
        starters: list[str] = []
        overlap_values: list[float] = []
        for speaker_entry in asset["speakers"].values():
            stage_entry = speaker_entry["stages"][stage_label]
            for context in stage_entry["contexts"].values():
                for form_key in ("statement", "question"):
                    overlap = float(context["mean_overlap_ratio_curr"].get(form_key, 0.0))
                    for form in context["forms"][form_key]:
                        counts = [int(count) for count in form["sentence_word_counts"] if int(count) > 0]
                        if not counts:
                            continue
                        forms.append(form)
                        sentence_counts.append(len(counts))
                        total_words.append(int(form["word_count"]))
                        first_sentence_words.append(int(counts[0]))
                        if len(counts) == 1:
                            single_sentence_words.append(int(counts[0]))
                        else:
                            later_sentence_words.extend(int(count) for count in counts[1:])
                        starter = str(form.get("starter") or "").strip()
                        if starter:
                            starters.append(starter)
                        overlap_values.append(overlap)
        if not forms:
            raise ValueError(f"No RealCBT surface forms found for stage={stage_label}")
        profiles[stage_label] = {
            "total_word_values": tuple(sorted(int(value) for value in total_words)),
            "avg_sentences": mean(sentence_counts),
            "sentence_count_std": pstdev(sentence_counts) if len(sentence_counts) > 1 else 0.0,
            "sentence_count_q10": _quantile(sentence_counts, 0.10),
            "sentence_count_q25": _quantile(sentence_counts, 0.25),
            "sentence_count_q75": _quantile(sentence_counts, 0.75),
            "sentence_count_q90": _quantile(sentence_counts, 0.90),
            "avg_total_words": mean(total_words),
            "total_words_std": pstdev(total_words) if len(total_words) > 1 else 0.0,
            "total_words_q10": _quantile(total_words, 0.10),
            "total_words_q25": _quantile(total_words, 0.25),
            "total_words_q75": _quantile(total_words, 0.75),
            "total_words_q90": _quantile(total_words, 0.90),
            "avg_first_sentence_words": mean(first_sentence_words),
            "first_sentence_std": pstdev(first_sentence_words) if len(first_sentence_words) > 1 else 0.0,
            "first_sentence_q10": _quantile(first_sentence_words, 0.10),
            "first_sentence_q25": _quantile(first_sentence_words, 0.25),
            "first_sentence_q75": _quantile(first_sentence_words, 0.75),
            "first_sentence_q90": _quantile(first_sentence_words, 0.90),
            "avg_later_sentence_words": mean(later_sentence_words) if later_sentence_words else 0.0,
            "later_sentence_std": pstdev(later_sentence_words) if len(later_sentence_words) > 1 else 0.0,
            "later_sentence_q10": _quantile(later_sentence_words, 0.10) if later_sentence_words else 0,
            "later_sentence_q25": _quantile(later_sentence_words, 0.25) if later_sentence_words else 0,
            "later_sentence_q75": _quantile(later_sentence_words, 0.75) if later_sentence_words else 0,
            "later_sentence_q90": _quantile(later_sentence_words, 0.90) if later_sentence_words else 0,
            "avg_single_sentence_words": mean(single_sentence_words) if single_sentence_words else mean(total_words),
            "single_sentence_std": pstdev(single_sentence_words) if len(single_sentence_words) > 1 else pstdev(total_words),
            "single_sentence_q10": _quantile(single_sentence_words, 0.10) if single_sentence_words else _quantile(total_words, 0.10),
            "single_sentence_q25": _quantile(single_sentence_words, 0.25) if single_sentence_words else _quantile(total_words, 0.25),
            "single_sentence_q75": _quantile(single_sentence_words, 0.75) if single_sentence_words else _quantile(total_words, 0.75),
            "single_sentence_q90": _quantile(single_sentence_words, 0.90) if single_sentence_words else _quantile(total_words, 0.90),
            "starter_probability": len(starters) / len(forms),
            "starter_pool": tuple(starters),
            "avg_overlap_ratio_curr": mean(overlap_values) if overlap_values else 0.0,
        }
    return profiles


def _stage_turn_position(stage_label: str, turn_id: int, target_turns: int | None) -> tuple[int, int]:
    total = max(1, int(target_turns or 20))
    stage_turns = [
        current_turn
        for current_turn in range(1, total + 1)
        if surface_stage_for_turn(current_turn, total) == stage_label
    ]
    if not stage_turns:
        return 0, 1
    try:
        return stage_turns.index(int(turn_id)), len(stage_turns)
    except ValueError:
        return 0, len(stage_turns)


def _stage_mean_centered_schedule(*, values: tuple[int, ...], target_mean: float, slot_count: int) -> tuple[int, ...]:
    if slot_count <= 0:
        return ()
    if not values:
        fallback = max(1, int(round(float(target_mean))))
        return tuple(fallback for _ in range(slot_count))

    schedule = [
        float(_quantile(list(values), (slot + 1) / float(slot_count + 1)))
        for slot in range(slot_count)
    ]
    centered_delta = float(target_mean) - (sum(schedule) / float(slot_count))
    adjusted = [max(1.0, value + centered_delta) for value in schedule]
    rounded = [max(1, int(round(value))) for value in adjusted]

    target_total = max(slot_count, int(round(float(target_mean) * float(slot_count))))
    delta = target_total - sum(rounded)
    if delta != 0:
        middle = (slot_count - 1) / 2.0
        order = sorted(range(slot_count), key=lambda index: (abs(index - middle), index))
        if delta < 0:
            order = sorted(order, key=lambda index: (-rounded[index], abs(index - middle), index))
        remaining = abs(delta)
        sign = 1 if delta > 0 else -1
        while remaining > 0:
            progressed = False
            for index in order:
                if sign < 0 and rounded[index] <= 1:
                    continue
                rounded[index] += sign
                remaining -= 1
                progressed = True
                if remaining <= 0:
                    break
            if not progressed:
                break
    return tuple(int(value) for value in rounded)


def _choose_question_end(
    *,
    speaker: str,
    stage_label: str,
    prev_question: bool,
    run_seed: int,
    session_id: int,
    turn_id: int,
    target_turns: int | None,
) -> bool:
    context = _speaker_asset(speaker, stage_label, prev_question)
    probability = float(context.get("question_probability", 0.0))
    if probability <= 0.0:
        return False
    if probability >= 1.0:
        return True
    draw = stable_fraction(
        "surface-question-end",
        str(speaker).strip().lower(),
        stage_label,
        int(prev_question),
        int(run_seed),
        int(session_id),
        int(turn_id),
        int(target_turns or 0),
    )
    return draw < probability


def _target_sentence_count(
    *,
    stage_label: str,
    run_seed: int,
    session_id: int,
    turn_id: int,
    target_turns: int | None,
) -> int:
    profile = _stage_surface_profiles()[stage_label]
    base = max(1.0, float(profile["avg_sentences"]))
    lower = max(1, int(base))
    upper = max(lower, int(base) if float(base).is_integer() else int(base) + 1)
    if lower == upper:
        return lower
    probability_upper = max(0.0, min(1.0, base - lower))
    draw = stable_fraction(
        "surface-sentence-count-choice",
        stage_label,
        int(run_seed),
        int(session_id),
        int(turn_id),
        int(target_turns or 0),
    )
    return upper if draw < probability_upper else lower


def _target_total_words(
    *,
    stage_label: str,
    session_id: int,
    turn_id: int,
    target_turns: int | None,
) -> int:
    profile = _stage_surface_profiles()[stage_label]
    stage_turn_index, stage_turn_count = _stage_turn_position(stage_label, turn_id, target_turns)
    schedule = _stage_mean_centered_schedule(
        values=tuple(int(value) for value in profile["total_word_values"]),
        target_mean=float(profile["avg_total_words"]),
        slot_count=stage_turn_count,
    )
    if not schedule:
        return max(1, int(round(float(profile["avg_total_words"]))))
    return int(schedule[min(stage_turn_index, len(schedule) - 1)])


def _starter_for_stage(
    *,
    stage_label: str,
    run_seed: int,
    session_id: int,
    turn_id: int,
    target_turns: int | None,
) -> str:
    profile = _stage_surface_profiles()[stage_label]
    starter_pool = tuple(str(item) for item in profile["starter_pool"] if str(item).strip())
    if not starter_pool:
        return ""
    keep = stable_fraction(
        "surface-starter-keep",
        stage_label,
        int(run_seed),
        int(session_id),
        int(turn_id),
        int(target_turns or 0),
    ) < float(profile["starter_probability"])
    if not keep:
        return ""
    return stable_choice(
        starter_pool,
        "surface-starter-pick",
        stage_label,
        int(run_seed),
        int(session_id),
        int(turn_id),
        int(target_turns or 0),
    )


def _sentence_word_targets(
    *,
    stage_label: str,
    sentence_count: int,
    run_seed: int,
    session_id: int,
    turn_id: int,
    target_turns: int | None,
) -> tuple[int, ...]:
    profile = _stage_surface_profiles()[stage_label]
    target_total_words = _target_total_words(
        stage_label=stage_label,
        session_id=session_id,
        turn_id=turn_id,
        target_turns=target_turns,
    )
    if sentence_count <= 1:
        return (target_total_words,)

    first_base = max(1.0, float(profile["avg_first_sentence_words"]))
    later_base = max(
        1.0,
        float(
            profile["avg_later_sentence_words"]
            or profile["avg_single_sentence_words"]
            or profile["avg_total_words"]
        ),
    )
    base_counts = [first_base] + [later_base] * max(0, sentence_count - 1)
    base_total = sum(base_counts) or 1.0
    scale = float(target_total_words) / float(base_total)
    counts = [max(1, int(round(base * scale))) for base in base_counts]
    delta = int(target_total_words) - sum(counts)
    if delta != 0:
        order = list(range(len(counts)))
        if delta < 0:
            # Prefer shrinking longer later sentences first so the first sentence keeps its short-opening function.
            order = sorted(order, key=lambda index: (index == 0, -counts[index], index))
        else:
            # Prefer adding words to later sentences first; if there is only one sentence, it naturally absorbs the delta.
            order = sorted(order, key=lambda index: (index == 0, index))
        remaining = abs(delta)
        sign = 1 if delta > 0 else -1
        while remaining > 0:
            progressed = False
            for index in order:
                if sign < 0 and counts[index] <= 1:
                    continue
                counts[index] += sign
                remaining -= 1
                progressed = True
                if remaining <= 0:
                    break
            if not progressed:
                break
    return tuple(int(count) for count in counts)


def build_realcbt_turn_form_plan(
    *,
    speaker: str,
    session_id: int,
    turn_id: int,
    target_turns: int | None,
    run_seed: int,
    shared_source_text: str,
    prev_utterance_text: str,
) -> TurnFormPlan:
    stage_label = surface_stage_for_turn(turn_id, target_turns)
    prev_question = message_ends_with_question(prev_utterance_text)
    context = _speaker_asset(speaker, stage_label, prev_question)
    reference_key = _promptv3_reference_key(
        speaker=speaker,
        turn_id=turn_id,
        prev_question=prev_question,
    )
    reference_entry = load_promptv3_surface_reference().get(reference_key)

    if reference_entry:
        form_label = str(reference_entry["form_label"]).strip()
        sentence_word_counts = tuple(
            int(count) for count in reference_entry["sentence_word_counts"] if int(count) > 0
        )
        if not sentence_word_counts:
            raise ValueError(f"PromptV3 surface reference produced no sentence counts for {reference_key}")
        selected_starter = str(reference_entry.get("starter") or "").strip()
        end_with_question = bool(reference_entry.get("end_with_question", False))
        max_question_count = 1 if end_with_question else 0
        source_form_id = f"promptv3_ref:{reference_key}"
    else:
        end_with_question = _choose_question_end(
            speaker=speaker,
            stage_label=stage_label,
            prev_question=prev_question,
            run_seed=run_seed,
            session_id=session_id,
            turn_id=turn_id,
            target_turns=target_turns,
        )
        form_label = "question" if end_with_question else "statement"
        forms = list(context["forms"][form_label])
        if not forms:
            raise ValueError(
                f"No RealCBT surface forms available for speaker={speaker}, stage={stage_label}, "
                f"context={_context_key(prev_question)}, form={form_label}"
            )
        form_index = min(
            len(forms) - 1,
            int(
                stable_fraction(
                    "surface-observed-form",
                    str(speaker).strip().lower(),
                    stage_label,
                    _context_key(prev_question),
                    form_label,
                    int(run_seed),
                    int(session_id),
                    int(turn_id),
                    int(target_turns or 0),
                )
                * len(forms)
            ),
        )
        selected_form = forms[form_index]
        sentence_word_counts = tuple(int(count) for count in selected_form["sentence_word_counts"] if int(count) > 0)
        if not sentence_word_counts:
            raise ValueError(
                f"Selected RealCBT surface form has no positive sentence counts for "
                f"speaker={speaker}, stage={stage_label}, context={_context_key(prev_question)}, form={form_label}"
            )
        selected_starter = str(selected_form.get("starter") or "").strip()
        max_question_count = 1 if end_with_question else 0
        source_form_id = f"{stage_label}:{_context_key(prev_question)}:{form_label}:{form_index}"

    target_overlap_ratio = float(context["mean_overlap_ratio_curr"].get(form_label, 0.0))
    reuse_budget = max(0, int(round(sum(sentence_word_counts) * target_overlap_ratio)))
    return TurnFormPlan(
        speaker=str(speaker).strip().lower(),
        stage_label=stage_label,
        form_label=form_label,
        source_form_id=source_form_id,
        sentence_word_counts=sentence_word_counts,
        starter=selected_starter,
        end_with_question=end_with_question,
        max_question_count=max_question_count,
        target_overlap_ratio=target_overlap_ratio,
        reuse_budget_content_words=reuse_budget,
        allowed_shared_words=select_allowed_shared_words(shared_source_text, reuse_budget),
    )


def format_turn_form_block(plan: TurnFormPlan, *, counterpart_label: str) -> str:
    lines = [
        "## Turn Form",
        "- Private speaking limits. Use them silently; do not quote or restate them.",
        "- Contractions count as 1 word. Counts are words.",
        f"- Stage: {plan.stage_label}. Form: {plan.form_label}.",
        f"- Sentences: {len(plan.sentence_word_counts)}.",
    ]
    if plan.starter:
        lines.append(f'- Start with this opener if it still sounds natural: "{plan.starter}"')
    for idx, word_count in enumerate(plan.sentence_word_counts, start=1):
        suffix = "; end=?" if plan.end_with_question and idx == len(plan.sentence_word_counts) else ""
        lines.append(f"- Sentence {idx}: {int(word_count)} words{suffix}.")
    lines.append(f"- Questions: {plan.max_question_count}.")
    return "\n".join(lines)
