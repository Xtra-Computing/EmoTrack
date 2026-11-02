"""CLI entrypoint for EmoTrack training pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

_DEFAULT_CONFIGS = {
    "emotrack": Path("configs/train/emotrack.yaml"),
    "emotrack_daic": Path("configs/train/emotrack_daic.yaml"),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or manage the EmoTrack PHQ-8 evaluator.",
    )
    parser.add_argument(
        "--method",
        choices=["emotrack", "emotrack_daic"],
        default="emotrack",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to training config YAML (defaults to configs/train/emotrack.yaml)",
    )
    parser.add_argument(
        "--build-dataset-only",
        action="store_true",
        help="Only build the JSONL dataset, do not train",
    )
    parser.add_argument(
        "--build-split-only",
        action="store_true",
        help="Only build a train/val/test split manifest, do not train",
    )
    parser.add_argument(
        "--split-ratios",
        type=str,
        default="0.7,0.1,0.2",
        help="Comma-separated train,val,test ratios (default: 0.7,0.1,0.2)",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Random seed for split (default: 42)",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Clean conversation bundle root for dataset building, for example result/longcounsel-8",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for dataset JSONL or split manifest",
    )
    parser.add_argument(
        "--transcript-view",
        choices=["full", "client-only"],
        default="full",
        help="Transcript view for dataset building",
    )
    parser.add_argument(
        "--build-daic-dataset",
        action="store_true",
        help="Build SFT-format JSONL from DAIC-WOZ result bundles (train+dev splits)",
    )
    parser.add_argument(
        "--inference-seed",
        type=int,
        default=None,
        help="Override training seed and suffix output paths for multi-seed runs.",
    )
    parser.add_argument(
        "--deterministic-training",
        action="store_true",
        help="Fix Python/NumPy/Torch/DataLoader randomness for reproducible training.",
    )
    parser.add_argument(
        "--embed-cache",
        action="store_true",
        help="Build embedding cache (requires --root and --output)",
    )
    parser.add_argument(
        "--dataset-name",
        choices=["longcounsel8", "daic"],
        default="longcounsel8",
        help="Dataset name for embedding cache (default: longcounsel8)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if args.embed_cache:
        from EmoTrack.train.aida.embed_cache import build_embedding_cache

        if args.root is None:
            print("Error: --root is required with --embed-cache")
            return 1
        output = args.output or Path("out/embed_cache")
        print(f"Building {args.dataset_name} embedding cache from {args.root} -> {output}")
        build_embedding_cache(
            source_root=args.root,
            cache_dir=output,
            dataset_name=args.dataset_name,
        )
        return 0

    build_flags = sum([args.build_daic_dataset, args.build_dataset_only, args.build_split_only])
    if build_flags > 1:
        print("Error: --build-daic-dataset, --build-dataset-only, and --build-split-only are mutually exclusive")
        return 1

    if args.build_daic_dataset:
        from EmoTrack.data.daic_woz import build_daic_sft_dataset

        root = args.root or Path("result/daic_woz")
        output = args.output or Path("out/emotrack_sft_daic/daic_dataset.jsonl")
        print(f"Building DAIC-WOZ SFT dataset from {root} ...")
        examples = build_daic_sft_dataset(
            result_root=root,
            transcript_view=args.transcript_view,
        )
        print(f"  {len(examples)} examples")
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"  Written to {output}")
        return 0

    if args.build_split_only:
        from EmoTrack.evaluate.datasets.conversation import (
            build_conversation_split_manifest,
            load_conversation_rows,
        )

        if args.root is None:
            print("Error: --root is required with --build-split-only")
            return 1
        output = args.output or (args.root / "train_val_test_split.json")
        ratios = tuple(float(x) for x in args.split_ratios.split(","))
        if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
            print(f"Error: --split-ratios must be 3 values summing to 1.0, got {ratios}")
            return 1
        print(f"Building split from {args.root} ...")
        rows = load_conversation_rows(args.root)
        manifest = build_conversation_split_manifest(
            rows, split_file=output, ratios=ratios, seed=args.split_seed,
        )
        print(f"  {manifest['counts']}")
        print(f"  Written to {output}")
        return 0

    if args.build_dataset_only:
        from EmoTrack.data.scoring_dataset import build_scoring_dataset, export_jsonl

        if args.root is None:
            print("Error: --root is required with --build-dataset-only")
            return 1
        output = args.output or Path("out/emotrack_dataset.jsonl")
        print(f"Building dataset from {args.root} ...")
        examples = build_scoring_dataset(args.root, transcript_view=args.transcript_view)
        print(f"  {len(examples)} examples")
        export_jsonl(examples, output)
        print(f"  Written to {output}")
        return 0

    config_path = args.config or _DEFAULT_CONFIGS.get(args.method)
    if config_path is None:
        print(f"Error: no default config for method {args.method}, use --config")
        return 1

    from EmoTrack.train.aida.config import load_hybrid_config
    from EmoTrack.train.aida.trainer import train_hybrid

    config = load_hybrid_config(config_path)
    if args.inference_seed is not None:
        config.inference_seed = args.inference_seed
        config.seed = args.inference_seed
    if args.deterministic_training:
        config.deterministic_training = True
    print(f"EmoTrack training with config: {config_path}")
    print(f"  AIDA cache: {config.aida_cache_path}")
    print(f"  Embed cache: {config.embed_cache_dir}")
    print(f"  Output: {config.output_dir}")
    if config.inference_seed != 0:
        print(f"  Inference seed: {config.inference_seed}")
    if config.deterministic_training:
        print("  Deterministic training: enabled")
    checkpoint_path = train_hybrid(config)
    print(f"Best checkpoint saved to {checkpoint_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
