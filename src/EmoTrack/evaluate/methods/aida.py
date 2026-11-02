from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from EmoTrack.evaluate.core.scores import SessionExampleRow

from EmoTrack.evaluate.utils import (
    run_dynamic_task_queue,
    stable_text_hash,
    write_json,
)


_DEFAULT_PROMPTS_PATH = Path("configs/evaluate/aida_prompts.json")
_MAIN_GROUPS = ("PHQ8", "Ling", "CDS")
_EXTRA_GROUPS = ("Neu", "DepSev")
_GREEDY_TEMPERATURE = 0.0
_GREEDY_TOP_P = 1.0
_GREEDY_MAX_TOKENS = 64
_PARSE_MAX_RETRIES = 3
_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{[^{}]*\}")
_AIDA_SCORE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "AIDAScore",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "score": {"type": "number"},
            },
            "required": ["score"],
            "additionalProperties": False,
        },
    },
}


@dataclass(frozen=True)
class AIDAFeatureRow:
    row_key: str
    run_id: str
    session_index: int
    transcript_hash: str
    features: list[float]


@dataclass(frozen=True)
class AIDAModel:
    feature_names: list[str]
    scaler_mean: list[float]
    scaler_std: list[float]
    intercept: float
    coefficients: list[float]
    train_size: int


def load_aida_questions(
    path: str | Path = _DEFAULT_PROMPTS_PATH,
    groups: tuple[str, ...] = _MAIN_GROUPS,
) -> list[dict[str, str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    questions: list[dict[str, str]] = []
    qid = 1
    for group_name in groups:
        group = payload[group_name]
        prompt = str(group["Prompt"])
        for question in group["Question"]:
            questions.append(
                {
                    "feature_name": f"Q{qid}",
                    "group_name": group_name,
                    "prompt": prompt,
                    "question": str(question),
                }
            )
            qid += 1
    return questions


async def extract_aida_features(
    rows: list[SessionExampleRow],
    *,
    llm_clients: list[Any],
    model_name: str,
    cache_path: Path,
    prompts_path: str | Path = _DEFAULT_PROMPTS_PATH,
    max_concurrency: int = 4,
    groups: tuple[str, ...] = _MAIN_GROUPS,
    inference_seed: int = 0,
) -> dict[str, AIDAFeatureRow]:
    questions = load_aida_questions(prompts_path, groups=groups)
    prompt_payload = json.loads(Path(prompts_path).read_text(encoding="utf-8"))
    version_str = (
        json.dumps(prompt_payload, ensure_ascii=False, sort_keys=True)
        + "|groups=" + ",".join(groups)
    )
    version_str += f"|seed={inference_seed}"
    prompt_version = stable_text_hash(version_str)
    cache = _load_feature_cache(cache_path)
    results: dict[str, AIDAFeatureRow] = {}
    uncached_rows: list[SessionExampleRow] = []
    cache_write_lock = asyncio.Lock()

    for row in rows:
        row_key = _row_key(row)
        transcript_hash = stable_text_hash(row.transcript_full)
        cache_key = _cache_key(
            transcript_hash=transcript_hash,
            model_name=model_name,
            prompt_version=prompt_version,
        )
        cached = cache.get(cache_key)
        if cached is not None:
            results[row_key] = AIDAFeatureRow(
                row_key=row_key,
                run_id=row.run_id,
                session_index=row.session_index,
                transcript_hash=transcript_hash,
                features=list(cached["features"]),
            )
            continue
        uncached_rows.append(row)

    _cache_write_counter = 0

    async def _extract_one(client: Any, row: SessionExampleRow) -> None:
        nonlocal _cache_write_counter
        row_key = _row_key(row)
        transcript_hash = stable_text_hash(row.transcript_full)
        features: list[float] = []
        for spec in questions:
            base_prompt = (
                spec["prompt"]
                .replace("[INSERT_INTERVIEW]", row.transcript_full)
                .replace("[INSERT_QUESTION]", spec["question"])
            )
            last_err: Exception | None = None
            for attempt in range(_PARSE_MAX_RETRIES):
                suffix = (
                    'Output a JSON object only in the form {"score": <number from 0 to 10>}.'
                    if attempt == 0
                    else 'Return only {"score": N} where N is 0-10. No thinking tags, no prose, no markdown.'
                )
                prompt = f"{base_prompt}\n\n{suffix}"
                create_kwargs: dict = dict(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=_GREEDY_TEMPERATURE,
                    top_p=_GREEDY_TOP_P,
                    max_tokens=_GREEDY_MAX_TOKENS,
                    response_format=_AIDA_SCORE_RESPONSE_FORMAT,
                )
                create_kwargs["seed"] = inference_seed
                response = await client.chat.completions.create(**create_kwargs)
                try:
                    text = response.choices[0].message.content or ""
                    score = _parse_score(text)
                    last_err = None
                    break
                except (ValueError, json.JSONDecodeError) as exc:
                    last_err = exc
            if last_err is not None:
                raise last_err
            features.append(score)

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
        async with cache_write_lock:
            cache[cache_key] = entry
            results[row_key] = AIDAFeatureRow(
                row_key=row_key,
                run_id=row.run_id,
                session_index=row.session_index,
                transcript_hash=transcript_hash,
                features=features,
            )
            _cache_write_counter += 1
            if _cache_write_counter % 100 == 0:
                write_json(cache_path, {"entries": list(cache.values())})

    if uncached_rows:
        if not llm_clients:
            raise ValueError(
                f"AIDA extraction needs an LLM client for {len(uncached_rows)} uncached rows, "
                f"but none was provided. Run with a valid LLM endpoint or populate the cache first."
            )
        await run_dynamic_task_queue(
            uncached_rows,
            workers=llm_clients,
            per_worker_concurrency=max(1, max_concurrency),
            process_item=_extract_one,
            desc=f"AIDA features ({len(uncached_rows)} sessions × {len(questions)} questions)",
        )
        write_json(cache_path, {"entries": list(cache.values())})
    return results


def fit_aida_model(
    train_rows: list[SessionExampleRow],
    features_by_row_key: dict[str, AIDAFeatureRow],
    *,
    prompts_path: str | Path = _DEFAULT_PROMPTS_PATH,
) -> AIDAModel:
    feature_names = [spec["feature_name"] for spec in load_aida_questions(prompts_path)]
    train_pairs: list[tuple[list[float], float]] = []
    for row in train_rows:
        if row.target_total is None:
            continue
        feature_row = features_by_row_key[_row_key(row)]
        train_pairs.append((feature_row.features, float(row.target_total)))
    if not train_pairs:
        raise ValueError("No train rows with target_total available for AIDA.")

    x = torch.tensor([pair[0] for pair in train_pairs], dtype=torch.float64)
    y = torch.tensor([pair[1] for pair in train_pairs], dtype=torch.float64)
    scaler_mean = x.mean(dim=0)
    scaler_std = x.std(dim=0, unbiased=False)
    scaler_std = torch.where(scaler_std == 0, torch.ones_like(scaler_std), scaler_std)
    x_scaled = (x - scaler_mean) / scaler_std
    x_aug = torch.cat(
        [torch.ones((x_scaled.shape[0], 1), dtype=torch.float64), x_scaled],
        dim=1,
    )
    solution = torch.linalg.lstsq(x_aug, y.unsqueeze(1)).solution.squeeze(1)
    intercept = float(solution[0].item())
    coefficients = [float(value) for value in solution[1:].tolist()]
    return AIDAModel(
        feature_names=feature_names,
        scaler_mean=[float(value) for value in scaler_mean.tolist()],
        scaler_std=[float(value) for value in scaler_std.tolist()],
        intercept=intercept,
        coefficients=coefficients,
        train_size=len(train_pairs),
    )


def predict_aida(
    rows: list[SessionExampleRow],
    *,
    model: AIDAModel,
    features_by_row_key: dict[str, AIDAFeatureRow],
) -> dict[str, float]:
    scaler_mean = torch.tensor(model.scaler_mean, dtype=torch.float64)
    scaler_std = torch.tensor(model.scaler_std, dtype=torch.float64)
    coefficients = torch.tensor(model.coefficients, dtype=torch.float64)
    predictions: dict[str, float] = {}
    for row in rows:
        feature_row = features_by_row_key[_row_key(row)]
        features = torch.tensor(feature_row.features, dtype=torch.float64)
        scaled = (features - scaler_mean) / scaler_std
        prediction = float(model.intercept + torch.dot(scaled, coefficients).item())
        predictions[_row_key(row)] = prediction
    return predictions


def write_aida_model(model: AIDAModel, path: Path) -> Path:
    return write_json(path, model)


def _row_key(row: SessionExampleRow) -> str:
    return f"{row.run_id}::s{row.session_index}"


def _cache_key(*, transcript_hash: str, model_name: str, prompt_version: str) -> str:
    return stable_text_hash(f"{transcript_hash}|{model_name}|{prompt_version}")


def _load_feature_cache(path: Path) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}

    def _ingest(entry: dict[str, Any]) -> None:
        transcript_hash = str(entry.get("transcript_hash") or "")
        model_name = str(entry.get("model_name") or "")
        prompt_version = str(entry.get("prompt_version") or "")
        if not transcript_hash or not model_name or not prompt_version:
            return
        entries[_cache_key(
            transcript_hash=transcript_hash,
            model_name=model_name,
            prompt_version=prompt_version,
        )] = entry

    # Load bulk cache file
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        for entry in (payload.get("entries", []) if isinstance(payload, dict) else []):
            if isinstance(entry, dict):
                _ingest(entry)

    # Load per-entry incremental cache files
    entry_dir = path.with_suffix("")
    if entry_dir.is_dir():
        for entry_file in entry_dir.glob("*.json"):
            try:
                entry = json.loads(entry_file.read_text(encoding="utf-8"))
                if isinstance(entry, dict):
                    _ingest(entry)
            except (json.JSONDecodeError, OSError):
                continue

    return entries


def _parse_score(text: str) -> float:
    raw = str(text or "").strip()
    if not raw:
        raise ValueError("AIDA response is empty.")
    stripped = _THINK_TAG_RE.sub("", raw).strip()
    if not stripped:
        raise ValueError(f"AIDA response contains only think tags: {raw[:120]!r}")
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        match = _JSON_OBJECT_RE.search(stripped)
        if match is None:
            raise ValueError(f"AIDA response has no parseable JSON: {stripped[:120]!r}")
        payload = json.loads(match.group())
    if not isinstance(payload, dict) or "score" not in payload:
        raise ValueError(f"AIDA response must be a JSON object with 'score': {stripped[:120]!r}")
    raw_value = payload["score"]
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
        raise ValueError(f"AIDA score must be numeric, got {raw_value!r}")
    value = float(raw_value)
    if value < 0 or value > 10:
        raise ValueError(f"AIDA score out of range: {value}")
    return value
