from __future__ import annotations

import hashlib
import importlib
import itertools
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from jinja2 import Template
from pydantic import BaseModel, ConfigDict, ValidationError

from EmoTrack.utils.paths import REPO_ROOT


PSYCHEVAL_EVAL_ROOT = REPO_ROOT / "repos" / "PsychEval" / "eval"
PSYCHEVAL_SAMPLE_ROOT = PSYCHEVAL_EVAL_ROOT / "data_sample"
PSYCHEVAL_PROMPT_ROOT = PSYCHEVAL_EVAL_ROOT / "prompts_cn"


def _ensure_upstream_import_path() -> None:
    upstream_path = str(PSYCHEVAL_EVAL_ROOT)
    if upstream_path not in sys.path:
        sys.path.insert(0, upstream_path)


def _load_upstream_module(module_name: str) -> Any:
    _ensure_upstream_import_path()
    original_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        return importlib.import_module(module_name)
    finally:
        sys.dont_write_bytecode = original_dont_write_bytecode


EvaluationMethod = _load_upstream_module("manager.base").EvaluationMethod
EvaluationManager = _load_upstream_module("manager.evaluation_multi").EvaluationManager

_UPSTREAM_EXPORTS = {
    "BDI_II": ("methods.client.bdi_ii", "BDI_II"),
    "CCT": ("methods.client.cct", "CCT"),
    "CTRS": ("methods.client.ctrs", "CTRS"),
    "EFT_TFS": ("methods.counselor.eft_tfs", "EFT_TFS"),
    "HTAIS": ("methods.counselor.htais", "HTAIS"),
    "IPO": ("methods.client.ipo", "IPO"),
    "MITI": ("methods.counselor.miti", "MITI"),
    "PANAS": ("methods.client.panas", "PANAS"),
    "PHQ_9": ("methods.client.phq_9", "PHQ_9"),
    "PSC": ("methods.counselor.psc", "PSC"),
    "RRO": ("methods.rro", "RRO"),
    "SCL_90": ("methods.client.scl_90", "SCL_90"),
    "SFBT": ("methods.client.sfbt", "SFBT"),
    "SRS": ("methods.client.srs", "SRS"),
    "STAI": ("methods.client.stai", "STAI"),
    "TES": ("methods.counselor.tes", "TES"),
    "WAI": ("methods.counselor.wai", "WAI"),
    "Custom_Dim": ("methods.counselor.custom_dim", "Custom_Dim"),
}

_PHQ_8_PROMPT = """You are a professional third-party supervisor.
Your core task is to strictly and objectively follow the [Evaluation Criteria], [Response Options], and [Output Format] provided by the user, and score the given counseling dialogue.

Core principles:
1. Objectivity: Evaluate only based on the provided [Input Data].
2. Accuracy: Use the scale definitions as written and make discriminative ratings.
3. Format compliance: Output must be valid JSON only, with no extra text.

### 1. Evaluation Task
Use the PHQ-8 scale to evaluate the client's depression-related presentation over the last two weeks.
This is blind evaluation pass {{ sample_index }}/{{ sample_count }}.

### 2. Response Options
The option letters are randomized for this pass and do not imply severity order.
{{ option_text }}

### 3. Items to Score
1. Little interest or pleasure in doing things
2. Feeling down, depressed, or hopeless
3. Trouble falling or staying asleep, or sleeping too much
4. Feeling tired or having little energy
5. Poor appetite or overeating
6. Feeling bad about yourself, or that you are a failure, or have let yourself or your family down
7. Trouble concentrating on things, such as reading or watching TV
8. Moving or speaking slowly, or the opposite: being restless/fidgety much more than usual

### 4. Output Format
You must output:
{"items": [{"item": "1", "choice": "A"}, {"item": "2", "choice": "B"}, ...]}

Requirements:
- Output exactly one JSON object
- The only top-level key is "items"
- Each entry must be: {"item": "string index", "choice": "single letter"}
- Use each item ID from 1 to 8 exactly once
- Use only the response letters shown above
- Do not explain any hidden mapping or mention any numeric scale

### 5. Input Data
[Client Background]:
{{ intake_form }}

[Counseling Dialogue]:
{{ diag }}
"""

_OPTION_LABELS = ("A", "B", "C", "D")
_OPTION_DESCRIPTIONS = {
    0: "Not at all",
    1: "Several days / infrequent",
    2: "More than half the days",
    3: "Nearly every day",
}
_OPTION_ORDERS = tuple(itertools.permutations(range(len(_OPTION_LABELS))))


class ItemChoice(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    item: str
    choice: str


class Items(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    items: list[ItemChoice]


def _profile_text(profile: dict[str, Any] | None) -> str:
    if not profile:
        return "{}"
    return json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True, default=str)


def _choice_maps(option_order: tuple[int, int, int, int]) -> tuple[dict[int, str], dict[str, int]]:
    score_to_choice = {score: _OPTION_LABELS[index] for index, score in enumerate(option_order)}
    choice_to_score = {choice: score for score, choice in score_to_choice.items()}
    return score_to_choice, choice_to_score


class PHQ_8(EvaluationMethod):
    """Local extension retained because upstream PsychEval only ships PHQ-9."""

    _EXPECTED_IDS = {str(index) for index in range(1, 9)}
    _VALID_CHOICES = set(_OPTION_LABELS)
    _MAX_RETRIES = 3

    def __init__(self, sample_count: int = 5) -> None:
        if sample_count <= 0:
            raise ValueError("sample_count must be positive")
        self.sample_count = sample_count

    @classmethod
    def _validate_item_ids(cls, items: list[ItemChoice]) -> None:
        ids = [item.item for item in items]
        id_set = set(ids)
        if len(items) != 8 or len(id_set) != 8 or id_set != cls._EXPECTED_IDS:
            raise ValueError(f"PHQ_8 items must be exactly 1-8 once each, got: {ids}")
        invalid_choices = sorted({item.choice for item in items} - cls._VALID_CHOICES)
        if invalid_choices:
            raise ValueError(f"PHQ_8 choices must be in {_OPTION_LABELS}, got: {invalid_choices}")

    def _option_order_for_sample(
        self,
        *,
        dialogue: str,
        profile: dict[str, Any] | None,
        sample_index: int,
    ) -> tuple[int, int, int, int]:
        if sample_index <= 0:
            raise ValueError("sample_index must be >= 1")

        seed_source = hashlib.sha256(
            f"psycheval-phq8|profile={_profile_text(profile)}|dialogue={dialogue}".encode("utf-8")
        ).hexdigest()
        ranked = sorted(
            _OPTION_ORDERS,
            key=lambda option_order: hashlib.sha256(
                f"{seed_source}|order={''.join(str(score) for score in option_order)}".encode("utf-8")
            ).hexdigest(),
        )
        return ranked[(sample_index - 1) % len(ranked)]

    async def evaluate(self, gpt_api, dialogue: Any, profile: dict[str, Any] | None = None) -> dict[str, Any]:
        schema = Items.model_json_schema()
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "Items", "strict": True, "schema": schema},
        }

        dialogue_text = str(dialogue)
        sample_totals: list[int] = []
        sample_item_scores: list[dict[str, int]] = []

        for sample_index in range(1, self.sample_count + 1):
            option_order = self._option_order_for_sample(
                dialogue=dialogue_text,
                profile=profile,
                sample_index=sample_index,
            )
            score_to_choice, choice_to_score = _choice_maps(option_order)
            option_text = "\n".join(
                f"- {score_to_choice[score]}: {_OPTION_DESCRIPTIONS[score]}"
                for score in range(len(_OPTION_LABELS))
            )
            prompt = Template(_PHQ_8_PROMPT).render(
                intake_form=_profile_text(profile),
                diag=dialogue_text,
                option_text=option_text,
                sample_index=sample_index,
                sample_count=self.sample_count,
            )

            last_error = ""
            for attempt in range(self._MAX_RETRIES):
                candidate_prompt = prompt
                if attempt > 0:
                    candidate_prompt = (
                        f"{prompt}\n\n"
                        "[Validation Feedback]\n"
                        f"Your previous output was invalid: {last_error}\n"
                        "Re-evaluate invalid items and return the full JSON again.\n"
                        "Hard constraints: item IDs must be exactly 1..8, each once; "
                        "every choice must be one letter in {A, B, C, D}."
                    )

                criteria_output = await self.chat_api(
                    gpt_api,
                    messages=[{"role": "user", "content": candidate_prompt}],
                    response_format=response_format,
                )
                try:
                    parsed = Items.model_validate_json(criteria_output)
                    self._validate_item_ids(parsed.items)
                    item_scores = {item.item: choice_to_score[item.choice] for item in parsed.items}
                    sample_item_scores.append(item_scores)
                    sample_totals.append(sum(item_scores.values()))
                    break
                except (ValidationError, ValueError) as exc:
                    last_error = str(exc)
            else:
                raise RuntimeError(
                    f"PHQ_8 invalid output after {self._MAX_RETRIES} attempts for sample "
                    f"{sample_index}/{self.sample_count}: {last_error}"
                )

        item_ids = sorted(self._EXPECTED_IDS, key=int)
        item_means = {
            item_id: round(statistics.fmean(sample[item_id] for sample in sample_item_scores), 2)
            for item_id in item_ids
        }
        item_std_scores = {
            item_id: round(statistics.pstdev(sample[item_id] for sample in sample_item_scores), 4)
            for item_id in item_ids
        }
        total_mean = round(statistics.fmean(sample_totals), 2)
        total_std = round(statistics.pstdev(sample_totals), 4)

        return {
            "client": total_mean,
            "client_meta": {
                "mean_score": total_mean,
                "std_score": total_std,
                "sample_count": len(sample_totals),
                "samples": sample_totals,
                "item_mean_scores": item_means,
                "item_std_scores": item_std_scores,
            },
        }

    def get_name(self) -> str:
        return "PHQ_8"


def __getattr__(name: str) -> Any:
    if name in _UPSTREAM_EXPORTS:
        module_name, attr_name = _UPSTREAM_EXPORTS[name]
        value = getattr(_load_upstream_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})


def configure_api_env(*, base_url: str, model: str, api_key: str) -> None:
    load_dotenv(Path.cwd() / ".env", override=False)
    os.environ["CHAT_API_KEY"] = api_key
    os.environ["CHAT_API_BASE"] = base_url
    os.environ["CHAT_MODEL_NAME"] = model


def build_evaluation_manager(
    *,
    sample_count: int,
    base_url: str = "",
    model: str = "",
    api_key: str = "",
    max_workers: int = 1,
) -> Any:
    manager = EvaluationManager(max_workers=max_workers)
    if api_key:
        manager.api_key = api_key
    if base_url:
        manager.base_url = base_url
    if model:
        manager.model_name = model
    manager.register(PHQ_8(sample_count=sample_count))
    return manager


async def evaluate_eval_ready_case(
    eval_ready_path: Path,
    output_dir: Path,
    *,
    sample_count: int,
    base_url: str = "",
    model: str = "",
    api_key: str = "",
    max_workers: int = 1,
) -> list[dict[str, object]]:
    prepared_eval_ready_path = _prepare_eval_ready_case_for_evaluation(eval_ready_path, output_dir)
    manager = build_evaluation_manager(
        sample_count=sample_count,
        base_url=base_url,
        model=model,
        api_key=api_key,
        max_workers=max_workers,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    return await manager.evaluate_case(str(prepared_eval_ready_path), str(output_dir))


def extract_session_scores(results: list[dict[str, object]], *, source_path: Path) -> list[dict[str, object]]:
    ordered_rows, _ = _dedupe_session_mapping(
        results,
        source_path=source_path,
        field_name="session_number",
    )
    return [
        {
            "session_index": _session_number(
                row.get("session_number"),
                source_path=source_path,
                field_name="session_number",
            ),
            "predicted_total": _extract_phq8_mean(row, source_path=source_path),
            "predicted_item_scores": _extract_phq8_item_scores(row, source_path=source_path),
        }
        for row in ordered_rows
        if not row.get("error")
    ]


def _prepare_eval_ready_case_for_evaluation(eval_ready_path: Path, output_dir: Path) -> Path:
    payload = json.loads(eval_ready_path.read_text(encoding="utf-8"))
    sessions_raw = payload.get("sessions")
    if not isinstance(sessions_raw, list) or not sessions_raw:
        raise ValueError(f"Missing sessions in {eval_ready_path}")

    deduped_sessions, duplicates_removed = _dedupe_session_mapping(
        sessions_raw,
        source_path=eval_ready_path,
        field_name="session_number",
    )
    original_numbers = [
        _session_number(session.get("session_number"), source_path=eval_ready_path, field_name="session_number")
        for session in sessions_raw
    ]
    deduped_numbers = [
        _session_number(session.get("session_number"), source_path=eval_ready_path, field_name="session_number")
        for session in deduped_sessions
    ]
    if duplicates_removed == 0 and original_numbers == deduped_numbers:
        return eval_ready_path

    payload["sessions"] = deduped_sessions
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared_path = output_dir / "eval_ready_case_deduped.json"
    prepared_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return prepared_path


def _extract_phq8_mean(row: dict[str, object], *, source_path: Path) -> float:
    eval_results = row.get("evaluation_results")
    if not isinstance(eval_results, dict):
        raise TypeError(f"Expected evaluation_results in {source_path}")

    client_scores = eval_results.get("client")
    if not isinstance(client_scores, dict):
        raise TypeError(f"Expected client scores in {source_path}")

    score = client_scores.get("PHQ_8")
    if isinstance(score, (int, float)):
        return float(score)

    if isinstance(score, dict):
        mean_score = score.get("mean_score")
        if isinstance(mean_score, (int, float)):
            return float(mean_score)

    client_meta = eval_results.get("client_meta")
    if isinstance(client_meta, dict):
        meta = client_meta.get("PHQ_8")
        if isinstance(meta, dict):
            mean_score = meta.get("mean_score")
            if isinstance(mean_score, (int, float)):
                return float(mean_score)

    raise TypeError(f"Missing PHQ_8 mean score in {source_path}")


def _extract_phq8_item_scores(
    row: dict[str, object],
    *,
    source_path: Path,
) -> dict[str, float]:
    eval_results = row.get("evaluation_results")
    if not isinstance(eval_results, dict):
        raise TypeError(f"Expected evaluation_results in {source_path}")

    client_meta = eval_results.get("client_meta")
    if not isinstance(client_meta, dict):
        raise TypeError(f"Expected client_meta scores in {source_path}")

    phq8_meta = client_meta.get("PHQ_8")
    if not isinstance(phq8_meta, dict):
        raise TypeError(f"Expected client_meta.PHQ_8 in {source_path}")

    raw_item_scores = phq8_meta.get("item_mean_scores")
    if not isinstance(raw_item_scores, dict):
        raise TypeError(f"Expected item_mean_scores in {source_path}")

    item_scores: dict[str, float] = {}
    for item_index, item_id in enumerate(sorted(raw_item_scores, key=lambda value: int(str(value))), start=1):
        raw_score = raw_item_scores.get(str(item_index), raw_item_scores.get(item_index))
        if not isinstance(raw_score, (int, float)):
            raise TypeError(
                f"Expected numeric item_mean_scores[{item_index}] in {source_path}, got {raw_score!r}"
            )
        item_scores[str(item_index)] = float(raw_score)
    return item_scores


def _session_number(value: object, *, source_path: Path, field_name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"Expected numeric {field_name} in {source_path}, got bool")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    raise TypeError(f"Expected numeric {field_name} in {source_path}, got {value!r}")


def _dedupe_session_mapping(
    rows: list[dict[str, object]],
    *,
    source_path: Path,
    field_name: str,
) -> tuple[list[dict[str, object]], int]:
    deduped: dict[int, dict[str, object]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError(f"Expected object rows in {source_path}")
        session_number = _session_number(row.get(field_name), source_path=source_path, field_name=field_name)
        deduped[session_number] = row

    ordered_numbers = sorted(deduped)
    ordered_rows = [deduped[number] for number in ordered_numbers]
    duplicates_removed = len(rows) - len(ordered_rows)
    return ordered_rows, duplicates_removed


__all__ = [
    "BDI_II",
    "CCT",
    "CTRS",
    "Custom_Dim",
    "build_evaluation_manager",
    "configure_api_env",
    "EFT_TFS",
    "evaluate_eval_ready_case",
    "EvaluationManager",
    "EvaluationMethod",
    "extract_session_scores",
    "HTAIS",
    "IPO",
    "MITI",
    "PANAS",
    "PHQ_8",
    "PHQ_9",
    "PSC",
    "PSYCHEVAL_EVAL_ROOT",
    "PSYCHEVAL_PROMPT_ROOT",
    "PSYCHEVAL_SAMPLE_ROOT",
    "RRO",
    "SCL_90",
    "SFBT",
    "SRS",
    "STAI",
    "TES",
    "WAI",
]
