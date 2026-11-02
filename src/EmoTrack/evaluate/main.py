from __future__ import annotations

import asyncio
import json
from pathlib import Path

from EmoTrack.evaluate.core.config import EvaluationConfig, load_evaluation_config
from EmoTrack.evaluate.core.scores import build_metrics_payload
from EmoTrack.evaluate.orchestration import load_rows_for_evaluation, prepare_llm_workers
from EmoTrack.evaluate.runners import run_method
from EmoTrack.evaluate.utils import (
    eval_output_root,
    write_json,
)


def run_from_config_path(path: str | Path, *, overwrite: bool = False, use_vllm: bool = False) -> int:
    return asyncio.run(_run(load_evaluation_config(path), overwrite=overwrite, use_vllm=use_vllm))


def run_from_config(config: EvaluationConfig, *, overwrite: bool = False, use_vllm: bool = False) -> int:
    return asyncio.run(_run(config, overwrite=overwrite, use_vllm=use_vllm))


async def _run(config: EvaluationConfig, *, overwrite: bool = False, use_vllm: bool = False) -> int:
    dataset_name = config.dataset.name
    input_root = Path(config.dataset.input_root).expanduser().resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"Evaluation input root does not exist: {input_root}")
    llm_workers = prepare_llm_workers(config)

    rows, train_rows, val_rows, test_rows, _split_manifest = load_rows_for_evaluation(
        config=config,
        input_root=input_root,
    )
    if not rows:
        raise ValueError(f"No scorable rows found under {input_root}")
    if not test_rows:
        raise ValueError("Evaluation has no test rows after applying the split policy.")

    seed = config.method.inference_seed
    out_root = eval_output_root(
        dataset_name=dataset_name,
        experiment_tag=config.output.experiment_tag,
        method_name=config.method.name,
        seed=seed,
    )

    run_scores = await run_method(
        dataset_name=dataset_name,
        train_rows=train_rows,
        val_rows=val_rows,
        test_rows=test_rows,
        out_root=out_root,
        config=config,
        llm_workers=llm_workers,
        overwrite=overwrite,
        use_vllm=use_vllm,
    )

    metrics = build_metrics_payload(run_scores)
    metrics_name = "metrics_test.json" if dataset_name == "conversation" else "metrics.json"
    write_json(out_root / metrics_name, metrics)
    print(
        json.dumps(
            {
                "dataset": dataset_name,
                "method": config.method.name,
                "experiment_tag": config.output.experiment_tag,
                "metrics": metrics,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0
