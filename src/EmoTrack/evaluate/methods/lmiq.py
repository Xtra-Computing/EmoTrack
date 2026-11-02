from __future__ import annotations

import asyncio
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from EmoTrack.evaluate.core.scores import (
    SessionExampleRow,
    make_row_key,
)

from EmoTrack.evaluate.utils import (
    run_dynamic_task_queue,
    stable_text_hash,
    write_json,
)


_DEFAULT_QUESTION_BANK_PATH = Path("configs/evaluate/lmiq_question_bank.json")
# The paper appendix appears to misplace one agreeableness-style question under
# the PHQ-8 section. The canonical bank keeps the published total of 135
# questions and the Table 1 domain counts (70/25/15/25) by assigning that
# question to Agreeableness and keeping the direct block as PHQ-8 + PCL-C.
_EXPECTED_DOMAIN_COUNTS = {
    "Mental Health": 70,
    "Personality": 25,
    "Therapeutic": 15,
    "Direct": 25,
}
_LMIQ_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "LMIQScore",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "score": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5,
                },
            },
            "required": ["score"],
            "additionalProperties": False,
        },
    },
}
_DEFAULT_TEMPERATURE = 0.0
_DEFAULT_TOP_P = 1.0
_DEFAULT_MAX_TOKENS = 64


@dataclass(frozen=True)
class LMIQQuestionSpec:
    feature_name: str
    domain_name: str
    topic_name: str
    question_text: str


@dataclass(frozen=True)
class LMIQFeatureRow:
    row_key: str
    run_id: str
    session_index: int
    transcript_hash: str
    features: list[int]


@dataclass(frozen=True)
class LMIQSelectionTrial:
    n_estimators: int
    max_depth: int
    validation_mse: float
    validation_mae: float
    validation_rmse: float


def load_lmiq_question_bank(path: str | Path = _DEFAULT_QUESTION_BANK_PATH) -> list[LMIQQuestionSpec]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    domains = payload.get("domains")
    if not isinstance(domains, list):
        raise ValueError(f"LMIQ question bank must contain a 'domains' list: {path}")

    specs: list[LMIQQuestionSpec] = []
    domain_counts: dict[str, int] = {}
    feature_index = 1
    for domain in domains:
        if not isinstance(domain, dict):
            raise ValueError(f"Invalid LMIQ domain entry in {path}: {domain!r}")
        domain_name = str(domain.get("domain_name") or "").strip()
        topics = domain.get("topics")
        if not domain_name or not isinstance(topics, list):
            raise ValueError(f"Invalid LMIQ domain in {path}: {domain!r}")
        domain_count = 0
        for topic in topics:
            if not isinstance(topic, dict):
                raise ValueError(f"Invalid LMIQ topic entry in {path}: {topic!r}")
            topic_name = str(topic.get("topic_name") or "").strip()
            questions = topic.get("questions")
            expected_count = 5
            if topic_name == "PHQ-8":
                expected_count = 8
            elif topic_name == "PCL-C":
                expected_count = 17
            if not topic_name or not isinstance(questions, list) or len(questions) != expected_count:
                raise ValueError(f"Invalid LMIQ topic in {path}: {topic!r}")
            for question in questions:
                question_text = str(question or "").strip()
                if not question_text:
                    raise ValueError(f"Blank LMIQ question in {path}: {topic!r}")
                specs.append(
                    LMIQQuestionSpec(
                        feature_name=f"Q{feature_index:03d}",
                        domain_name=domain_name,
                        topic_name=topic_name,
                        question_text=question_text,
                    )
                )
                feature_index += 1
                domain_count += 1
        domain_counts[domain_name] = domain_count

    if domain_counts != _EXPECTED_DOMAIN_COUNTS:
        raise ValueError(
            f"LMIQ domain counts do not match expected values: "
            f"expected={_EXPECTED_DOMAIN_COUNTS}, actual={domain_counts}"
        )
    if len(specs) != 135:
        raise ValueError(f"LMIQ question bank must flatten to 135 features, got {len(specs)}")
    return specs


def load_lmiq_prompt_payload(path: str | Path = _DEFAULT_QUESTION_BANK_PATH) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid LMIQ prompt payload: {path}")
    return payload


async def extract_lmiq_features(
    rows: list[SessionExampleRow],
    *,
    llm_clients: list[Any],
    model_name: str,
    cache_path: Path,
    question_bank_path: str | Path = _DEFAULT_QUESTION_BANK_PATH,
    max_concurrency: int = 4,
    inference_seed: int = 0,
) -> dict[str, LMIQFeatureRow]:


    question_specs = load_lmiq_question_bank(question_bank_path)
    prompt_payload = load_lmiq_prompt_payload(question_bank_path)
    prompt_template = str(prompt_payload.get("prompt_template") or "").strip()
    agreement_range = str(prompt_payload.get("agreement_range") or "").strip()
    if not prompt_template or not agreement_range:
        raise ValueError(f"Missing prompt_template/agreement_range in {question_bank_path}")
    version_str = json.dumps(prompt_payload, ensure_ascii=False, sort_keys=True)
    version_str += f"|seed={inference_seed}"
    prompt_version = stable_text_hash(version_str)
    cache = _load_feature_cache(cache_path)
    results: dict[str, LMIQFeatureRow] = {}
    uncached_rows: list[SessionExampleRow] = []
    cache_write_lock = asyncio.Lock()

    for row in rows:
        row_key = make_row_key(row)
        transcript_hash = stable_text_hash(row.transcript_full)
        cache_key = _cache_key(
            transcript_hash=transcript_hash,
            model_name=model_name,
            prompt_version=prompt_version,
        )
        cached = cache.get(cache_key)
        if cached is None:
            uncached_rows.append(row)
            continue
        results[row_key] = LMIQFeatureRow(
            row_key=row_key,
            run_id=row.run_id,
            session_index=row.session_index,
            transcript_hash=transcript_hash,
            features=[int(value) for value in cached["features"]],
        )

    _cache_write_counter = 0

    async def _extract_one(client: Any, row: SessionExampleRow) -> None:
        nonlocal _cache_write_counter
        row_key = make_row_key(row)
        transcript_hash = stable_text_hash(row.transcript_full)
        features: list[int] = []
        for spec in question_specs:
            prompt = prompt_template.format(
                raw_psyc_transcript=row.transcript_full,
                agreement_range=agreement_range,
                question=spec.question_text,
            )
            prompt = (
                f"{prompt}\n\n"
                'Output a JSON object only in the form {"score": <integer from 1 to 5>}.'
            )
            create_kwargs: dict = dict(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=_DEFAULT_TEMPERATURE,
                top_p=_DEFAULT_TOP_P,
                max_tokens=_DEFAULT_MAX_TOKENS,
                response_format=_LMIQ_RESPONSE_FORMAT,
            )
            create_kwargs["seed"] = inference_seed
            response = await client.chat.completions.create(**create_kwargs)
            features.append(_parse_score(response.choices[0].message.content or ""))

        cache_key = _cache_key(
            transcript_hash=transcript_hash,
            model_name=model_name,
            prompt_version=prompt_version,
        )
        entry = {
            "row_key": row_key,
            "run_id": row.run_id,
            "session_index": row.session_index,
            "transcript_hash": transcript_hash,
            "model_name": model_name,
            "prompt_version": prompt_version,
            "features": features,
        }
        feature_row = LMIQFeatureRow(
            row_key=row_key,
            run_id=row.run_id,
            session_index=row.session_index,
            transcript_hash=transcript_hash,
            features=features,
        )
        async with cache_write_lock:
            cache[cache_key] = entry
            results[row_key] = feature_row
            _cache_write_counter += 1
            if _cache_write_counter % 100 == 0:
                write_json(cache_path, {"entries": list(cache.values())})

    if uncached_rows:
        if not llm_clients:
            raise ValueError(
                f"LMIQ extraction needs an LLM client for {len(uncached_rows)} uncached rows, "
                f"but none was provided. Run with a valid LLM endpoint or populate the cache first."
            )
        await run_dynamic_task_queue(
            uncached_rows,
            workers=llm_clients,
            per_worker_concurrency=max(1, max_concurrency),
            process_item=_extract_one,
            desc=f"LMIQ features ({len(uncached_rows)} sessions × {len(question_specs)} questions)",
        )
        write_json(cache_path, {"entries": list(cache.values())})
    return results


def select_lmiq_hyperparameters(
    train_rows: list[SessionExampleRow],
    validation_rows: list[SessionExampleRow],
    *,
    features_by_row_key: dict[str, LMIQFeatureRow],
    n_estimators_grid: list[int],
    max_depth_grid: list[int],
    random_state: int = 0,
) -> tuple[dict[str, int], list[LMIQSelectionTrial]]:
    train_x, train_y = _matrix_and_targets(train_rows, features_by_row_key=features_by_row_key)
    validation_x, validation_y = _matrix_and_targets(validation_rows, features_by_row_key=features_by_row_key)
    if not train_x or not validation_x:
        raise ValueError("LMIQ hyperparameter selection requires non-empty train and validation rows")

    trials: list[LMIQSelectionTrial] = []
    best_params: dict[str, int] | None = None
    best_key: tuple[float, float, int, int] | None = None
    for n_estimators in n_estimators_grid:
        for max_depth in max_depth_grid:
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=1,
            )
            model.fit(train_x, train_y)
            predicted = model.predict(validation_x)
            mse = float(mean_squared_error(validation_y, predicted))
            mae = float(mean_absolute_error(validation_y, predicted))
            rmse = float(math.sqrt(mse))
            trials.append(
                LMIQSelectionTrial(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    validation_mse=mse,
                    validation_mae=mae,
                    validation_rmse=rmse,
                )
            )
            rank_key = (mse, mae, n_estimators, max_depth)
            if best_key is None or rank_key < best_key:
                best_key = rank_key
                best_params = {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                }

    if best_params is None:
        raise RuntimeError("Failed to select LMIQ hyperparameters")
    return best_params, trials


def fit_lmiq_model(
    train_rows: list[SessionExampleRow],
    *,
    features_by_row_key: dict[str, LMIQFeatureRow],
    n_estimators: int,
    max_depth: int,
    random_state: int = 0,
) -> RandomForestRegressor:
    train_x, train_y = _matrix_and_targets(train_rows, features_by_row_key=features_by_row_key)
    if not train_x:
        raise ValueError("No train rows with target_total available for LMIQ")
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=1,
    )
    model.fit(train_x, train_y)
    return model


def predict_lmiq(
    rows: list[SessionExampleRow],
    *,
    model: RandomForestRegressor,
    features_by_row_key: dict[str, LMIQFeatureRow],
) -> dict[str, float]:
    predictions: dict[str, float] = {}
    for row in rows:
        feature_row = features_by_row_key[make_row_key(row)]
        value = float(model.predict([feature_row.features])[0])
        predictions[make_row_key(row)] = value
    return predictions


def write_lmiq_model(model: RandomForestRegressor, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def write_lmiq_model_metadata(
    *,
    path: Path,
    question_bank_path: str | Path,
    selection_metric: str,
    best_params: dict[str, int],
    selection_train_size: int,
    selection_validation_size: int,
    final_train_size: int,
    selection_trials: list[LMIQSelectionTrial],
) -> Path:
    question_specs = load_lmiq_question_bank(question_bank_path)
    return write_json(
        path,
        {
            "feature_count": len(question_specs),
            "selection_metric": selection_metric,
            "best_params": best_params,
            "selection_train_size": selection_train_size,
            "selection_validation_size": selection_validation_size,
            "final_train_size": final_train_size,
            "features": [
                {
                    "feature_name": spec.feature_name,
                    "domain_name": spec.domain_name,
                    "topic_name": spec.topic_name,
                    "question_text": spec.question_text,
                }
                for spec in question_specs
            ],
            "selection_trials": [
                {
                    "n_estimators": trial.n_estimators,
                    "max_depth": trial.max_depth,
                    "validation_mse": round(trial.validation_mse, 6),
                    "validation_mae": round(trial.validation_mae, 6),
                    "validation_rmse": round(trial.validation_rmse, 6),
                }
                for trial in selection_trials
            ],
        },
    )


def _matrix_and_targets(
    rows: list[SessionExampleRow],
    *,
    features_by_row_key: dict[str, LMIQFeatureRow],
) -> tuple[list[list[int]], list[float]]:
    x: list[list[int]] = []
    y: list[float] = []
    for row in rows:
        if row.target_total is None:
            continue
        feature_row = features_by_row_key[make_row_key(row)]
        x.append(list(feature_row.features))
        y.append(float(row.target_total))
    return x, y


def _cache_key(*, transcript_hash: str, model_name: str, prompt_version: str) -> str:
    return stable_text_hash(f"{transcript_hash}|{model_name}|{prompt_version}")


def _load_feature_cache(path: Path) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    for entry in _iter_feature_cache_entries(path):
        if not isinstance(entry, dict):
            continue
        transcript_hash = str(entry.get("transcript_hash") or "")
        model_name = str(entry.get("model_name") or "")
        prompt_version = str(entry.get("prompt_version") or "")
        if not transcript_hash or not model_name or not prompt_version:
            continue
        entries[_cache_key(
            transcript_hash=transcript_hash,
            model_name=model_name,
            prompt_version=prompt_version,
        )] = entry
    return entries


def _iter_feature_cache_entries(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    entry_dir = _feature_cache_entry_dir(path)
    if entry_dir.is_dir():
        for entry_path in sorted(entry_dir.glob("*.json")):
            payload = json.loads(entry_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                entries.append(payload)
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_entries = payload.get("entries", []) if isinstance(payload, dict) else []
        for entry in raw_entries:
            if isinstance(entry, dict):
                entries.append(entry)
    return entries


def _feature_cache_entry_dir(path: Path) -> Path:
    return path.with_suffix("")


def _write_feature_cache_entry(path: Path, cache_key: str, entry: dict[str, Any]) -> Path:
    return write_json(_feature_cache_entry_dir(path) / f"{cache_key}.json", entry)


def _parse_score(text: str) -> int:
    stripped = str(text or "").strip()
    if not stripped:
        raise ValueError("LMIQ response is empty")
    payload = json.loads(stripped)
    if not isinstance(payload, dict) or "score" not in payload:
        raise ValueError(f"LMIQ response must be a JSON object with 'score': {stripped!r}")
    raw_value = payload["score"]
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
        raise ValueError(f"LMIQ score must be numeric: {stripped!r}")
    rounded = int(round(float(raw_value)))
    if abs(float(raw_value) - rounded) > 1e-6:
        raise ValueError(f"LMIQ score must be an integer: {stripped!r}")
    if rounded < 1 or rounded > 5:
        raise ValueError(f"LMIQ score must be between 1 and 5: {stripped!r}")
    return rounded


__all__ = [
    "LMIQFeatureRow",
    "LMIQQuestionSpec",
    "LMIQSelectionTrial",
    "extract_lmiq_features",
    "fit_lmiq_model",
    "load_lmiq_prompt_payload",
    "load_lmiq_question_bank",
    "predict_lmiq",
    "select_lmiq_hyperparameters",
    "write_lmiq_model",
    "write_lmiq_model_metadata",
]
