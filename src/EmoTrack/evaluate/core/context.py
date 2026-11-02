from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

from EmoTrack.evaluate.core.config import EvaluationConfig
from EmoTrack.evaluate.core.scores import DetectorRunScore, SessionExampleRow


@dataclass(frozen=True)
class PreparedLLMWorker:
    base_url: str
    model_name: str
    api_key: str
    worker_index: int


@dataclass(frozen=True)
class MethodRunContext:
    dataset_name: str
    train_rows: list[SessionExampleRow]
    val_rows: list[SessionExampleRow]
    test_rows: list[SessionExampleRow]
    out_root: Path
    config: EvaluationConfig
    llm_workers: list[PreparedLLMWorker] | None
    overwrite: bool = False
    use_vllm: bool = False


MethodRunner = Callable[[MethodRunContext], Awaitable[list[DetectorRunScore]]]
