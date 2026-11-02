from __future__ import annotations

from pathlib import Path

from EmoTrack.evaluate.core.config import EvaluationConfig
from EmoTrack.evaluate.core.context import MethodRunContext, MethodRunner, PreparedLLMWorker
from EmoTrack.evaluate.core.scores import DetectorRunScore, SessionExampleRow
from EmoTrack.evaluate.runners.aida import run_aida
from EmoTrack.evaluate.runners.baselines import (
    run_dual_encoder,
    run_ensembert,
    run_milintsevich,
)
from EmoTrack.evaluate.runners.emotrack import run_emotrack
from EmoTrack.evaluate.runners.lmiq import run_lmiq
from EmoTrack.evaluate.runners.psycheval import run_psycheval


async def run_method(
    *,
    dataset_name: str,
    train_rows: list[SessionExampleRow],
    val_rows: list[SessionExampleRow],
    test_rows: list[SessionExampleRow],
    out_root: Path,
    config: EvaluationConfig,
    llm_workers: list[PreparedLLMWorker] | None,
    overwrite: bool = False,
    use_vllm: bool = False,
) -> list[DetectorRunScore]:
    context = MethodRunContext(
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
    return await _method_runner(config.method.name)(context)


async def _run_aida_registered(context: MethodRunContext) -> list[DetectorRunScore]:
    return await run_aida(
        dataset_name=context.dataset_name,
        train_rows=context.train_rows,
        val_rows=context.val_rows,
        test_rows=context.test_rows,
        out_root=context.out_root,
        config=context.config,
        llm_workers=context.llm_workers,
        overwrite=context.overwrite,
    )


async def _run_lmiq_registered(context: MethodRunContext) -> list[DetectorRunScore]:
    return await run_lmiq(
        dataset_name=context.dataset_name,
        train_rows=context.train_rows,
        val_rows=context.val_rows,
        test_rows=context.test_rows,
        out_root=context.out_root,
        config=context.config,
        llm_workers=context.llm_workers,
        overwrite=context.overwrite,
    )


async def _run_ensembert_registered(context: MethodRunContext) -> list[DetectorRunScore]:
    return await run_ensembert(
        dataset_name=context.dataset_name,
        train_rows=context.train_rows,
        val_rows=context.val_rows,
        test_rows=context.test_rows,
        out_root=context.out_root,
        config=context.config,
        overwrite=context.overwrite,
    )


async def _run_dual_encoder_registered(context: MethodRunContext) -> list[DetectorRunScore]:
    return await run_dual_encoder(
        dataset_name=context.dataset_name,
        train_rows=context.train_rows,
        val_rows=context.val_rows,
        test_rows=context.test_rows,
        out_root=context.out_root,
        config=context.config,
        overwrite=context.overwrite,
    )


async def _run_milintsevich_registered(context: MethodRunContext) -> list[DetectorRunScore]:
    return await run_milintsevich(
        dataset_name=context.dataset_name,
        train_rows=context.train_rows,
        val_rows=context.val_rows,
        test_rows=context.test_rows,
        out_root=context.out_root,
        config=context.config,
        overwrite=context.overwrite,
    )


async def _run_psycheval_registered(context: MethodRunContext) -> list[DetectorRunScore]:
    return await run_psycheval(
        dataset_name=context.dataset_name,
        test_rows=context.test_rows,
        out_root=context.out_root,
        config=context.config,
        llm_workers=context.llm_workers,
    )


async def _run_emotrack_registered(context: MethodRunContext) -> list[DetectorRunScore]:
    return await run_emotrack(
        dataset_name=context.dataset_name,
        test_rows=context.test_rows,
        out_root=context.out_root,
        config=context.config,
        llm_workers=context.llm_workers,
        overwrite=context.overwrite,
        use_vllm=context.use_vllm,
    )


_METHOD_RUNNERS: dict[str, MethodRunner] = {
    "aida": _run_aida_registered,
    "lmiq": _run_lmiq_registered,
    "ensembert": _run_ensembert_registered,
    "dual_encoder": _run_dual_encoder_registered,
    "milintsevich": _run_milintsevich_registered,
    "psycheval": _run_psycheval_registered,
    "emotrack": _run_emotrack_registered,
}


def _method_runner(method_name: str) -> MethodRunner:
    try:
        return _METHOD_RUNNERS[method_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported method: {method_name}") from exc
