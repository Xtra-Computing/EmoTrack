"""CLI entrypoint for EmoTrack evaluation pipeline."""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def _prepare_deterministic_environment(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def _checkpoint_for_seed(raw_checkpoint: str, seed: int) -> str:
    if not raw_checkpoint:
        return raw_checkpoint
    path = Path(raw_checkpoint)
    parts = list(path.parts)
    for index, part in enumerate(parts):
        if part.startswith("seed") and part[4:].isdigit():
            parts[index] = f"seed{seed}"
            return str(Path(*parts))
    return raw_checkpoint


def _apply_dataset_overrides(config, *, input_root: Path | None, split_file: Path | None):
    if input_root is not None:
        config.dataset.input_root = str(input_root)
    if split_file is not None:
        config.dataset.split_file = str(split_file)
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description="Run EmoTrack evaluation.")
    parser.add_argument("config", type=Path, help="Path to evaluation config YAML")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-evaluate all runs, ignoring cached results",
    )
    parser.add_argument(
        "--vllm",
        action="store_true",
        help="Use vLLM for generative inference (sft_causal checkpoints)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override method.inference_seed from the evaluation config",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help="Override dataset.input_root from the evaluation config.",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        default=None,
        help="Override dataset.split_file from the evaluation config.",
    )
    deterministic_group = parser.add_mutually_exclusive_group()
    deterministic_group.add_argument(
        "--deterministic",
        dest="deterministic",
        action="store_true",
        default=None,
        help=(
            "Enable optional deterministic controls beyond the outer --seed sweep. "
            "All methods still use method.inference_seed/--seed."
        ),
    )
    deterministic_group.add_argument(
        "--no-deterministic",
        dest="deterministic",
        action="store_false",
        help="Disable optional deterministic controls even if enabled in the suite config.",
    )
    args = parser.parse_args()
    if args.deterministic is True and not args.overwrite:
        parser.error("--deterministic requires --overwrite so existing detector artifacts are not reused.")

    from EmoTrack.evaluate.main import run_from_config
    from EmoTrack.evaluate.core.config import load_evaluation_config

    config = load_evaluation_config(args.config)
    _apply_dataset_overrides(config, input_root=args.input_root, split_file=args.split_file)
    if args.seed is not None:
        config.method.inference_seed = int(args.seed)
        config.method.checkpoint = _checkpoint_for_seed(config.method.checkpoint, int(args.seed))
    if args.deterministic is not None:
        config.method.deterministic = bool(args.deterministic)
    if config.method.deterministic and not args.overwrite:
        parser.error("deterministic evaluation requires --overwrite so existing detector artifacts are not reused.")
    if config.method.deterministic:
        _prepare_deterministic_environment(config.method.inference_seed)
    return run_from_config(config, overwrite=args.overwrite, use_vllm=args.vllm)


if __name__ == "__main__":
    raise SystemExit(main())
