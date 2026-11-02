"""CLI entrypoint for Psyche dataset reconstruction, and trend generation."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Use locally supplied Psyche dataset assets to generate EmoTrack-compatible PHQ trend files."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/psyche/raw"),
        help="Directory for Psyche raw files (default: data/psyche/raw)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/psyche/processed"),
        help="Directory for processed outputs (default: data/psyche/processed)",
    )
    parser.add_argument(
        "--trend-dir",
        type=Path,
        default=Path("configs/conversation_trends/psyche"),
        help="Directory for generated trend YAML files (default: configs/conversation_trends/psyche)",
    )
    parser.add_argument(
        "--nhanes-xpt",
        type=Path,
        default=Path("data/DPQ_L.xpt"),
        help="Destination path for the NHANES DPQ_L XPT used to sample PHQ item profiles (default: data/DPQ_L.xpt)",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=0,
        help="Deterministic seed for NHANES-conditioned PHQ item sampling (default: 0)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from EmoTrack.data.psyche.files import require_nhanes_dpq_xpt, require_psyche_files
    from EmoTrack.data.psyche.generate_trends import (
        generate_hidden_states,
        generate_psyche_trends,
    )
    from EmoTrack.data.psyche.nhanes_phq_sampler import (
        annotate_sampled_phq8_totals,
        load_default_nhanes_sampler,
    )
    from EmoTrack.data.psyche.reconstruct import (
        build_long_format,
        build_wide_format,
        compute_change_labels,
        load_raw_data,
        print_summary,
        reconstruct_phq_trajectories,
    )

    logger.info("Checking local Psyche data under %s", args.raw_dir)
    require_psyche_files(args.raw_dir)
    logger.info("Checking local NHANES DPQ_L XPT at %s", args.nhanes_xpt)
    nhanes_xpt_path = require_nhanes_dpq_xpt(args.nhanes_xpt)

    raw_df = load_raw_data(args.raw_dir)
    strict_df, relaxed_df = reconstruct_phq_trajectories(raw_df)

    if strict_df.empty:
        logger.error("No complete-case participants found. Aborting.")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sampler = load_default_nhanes_sampler(nhanes_xpt_path)

    wide_df = build_wide_format(strict_df)
    wide_df = compute_change_labels(wide_df)
    wide_df = annotate_sampled_phq8_totals(
        wide_df,
        sampler=sampler,
        sample_seed=args.sample_seed,
    )
    wide_df.to_csv(args.output_dir / "wide.csv", index=False)

    long_df = build_long_format(wide_df)
    long_df.to_csv(args.output_dir / "long.csv", index=False)

    summary_text = print_summary(strict_df, relaxed_df)
    (args.output_dir / "summary.txt").write_text(summary_text, encoding="utf-8")

    trend_paths = generate_psyche_trends(
        wide_df,
        args.trend_dir,
        sampler=sampler,
        sample_seed=args.sample_seed,
    )
    state_dir = args.output_dir / "hidden_states"
    state_paths = generate_hidden_states(
        wide_df,
        state_dir,
        sampler=sampler,
        sample_seed=args.sample_seed,
    )

    logger.info(
        "Done. %d trend files, %d hidden states, outputs in %s",
        len(trend_paths),
        len(state_paths),
        args.output_dir,
    )


if __name__ == "__main__":
    main()
