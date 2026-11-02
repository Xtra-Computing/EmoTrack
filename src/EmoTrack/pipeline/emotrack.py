from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from EmoTrack.utils.paths import REPO_ROOT


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified EmoTrack wrapper for generation, evaluation, and data conversion.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Run cohort conversation generation from a YAML config")
    generate.add_argument("--config", type=Path, required=True, help="Path to generation config YAML")
    generate.add_argument("--detach", action="store_true", help="Run generation in the background and write a launcher log")

    evaluate = subparsers.add_parser("eval", help="Run evaluation with the canonical config and selected methods")
    try:
        from EmoTrack.evaluate.run_all_baselines import add_cli_arguments as add_baseline_cli_arguments
    except ModuleNotFoundError as exc:
        evaluate.add_argument("passthrough_args", nargs=argparse.REMAINDER)
        evaluate.set_defaults(_baseline_cli_import_error=exc)
    else:
        add_baseline_cli_arguments(evaluate)

    prepare_daic = subparsers.add_parser(
        "prepare-daic-woz",
        help="Convert raw data/daic_woz into canonical result/daic_woz run bundles",
    )
    prepare_daic.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="Optional raw DAIC-WOZ root override. Defaults to data/daic_woz.",
    )
    prepare_daic.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional canonical result root override. Defaults to result/daic_woz.",
    )
    prepare_daic.add_argument(
        "--split-output",
        type=Path,
        default=None,
        help="Optional split manifest output path. Defaults to <output-root>/train_val_test_split.json.",
    )

    summarize_metrics = subparsers.add_parser(
        "summarize-metrics",
        help="Aggregate generated evaluation metrics into CSV and Markdown tables",
    )
    summarize_metrics.add_argument(
        "--eval-root",
        type=Path,
        default=REPO_ROOT / "out" / "eval",
        help="Root containing per-method evaluation outputs. Defaults to out/eval.",
    )
    summarize_metrics.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Directory for aggregate tables. Defaults to result/metrics.",
    )

    dev_single = subparsers.add_parser("dev-generate-one", help="Dev-only single-run online generation")
    dev_single.add_argument("--config", type=Path, required=True, help="Path to a single-run conversation config YAML")
    return parser.parse_args()


def _run_detached(config_path: Path) -> int:
    log_root = REPO_ROOT / "out" / "start_conversation_launcher"
    log_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = log_root / f"{stamp}_emotrack_generate.log"
    pid_path = log_root / f"{stamp}_emotrack_generate.pid"
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "emotrack.py"),
        "generate",
        "--config",
        str(config_path.expanduser().resolve()),
    ]
    environment = os.environ.copy()
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=environment,
        )
    pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
    print(f"Detached generation started: pid={process.pid}")
    print(f"PID file: {pid_path}")
    print(f"Log: {log_path}")
    print(f"Follow: tail -f {log_path}")
    return 0


def main() -> int:
    args = _parse_args()
    if args.command == "generate":
        if args.detach:
            return _run_detached(args.config)
        from EmoTrack.pipeline.start_conversation import run_from_config_path

        return run_from_config_path(args.config)

    if args.command == "eval":
        if hasattr(args, "_baseline_cli_import_error"):
            raise args._baseline_cli_import_error
        from EmoTrack.evaluate.run_all_baselines import run_from_parsed_args

        return run_from_parsed_args(args)

    if args.command == "prepare-daic-woz":
        from EmoTrack.pipeline.prepare_daic_woz_result import run as prepare_daic_run

        kwargs: dict[str, object] = {}
        if args.source_root is not None:
            kwargs["source_root"] = args.source_root
        if args.output_root is not None:
            kwargs["output_root"] = args.output_root
        if args.split_output is not None:
            kwargs["split_output"] = args.split_output
        return prepare_daic_run(**kwargs)

    if args.command == "summarize-metrics":
        from EmoTrack.evaluate.aggregate_metrics import aggregate_metrics

        outputs = aggregate_metrics(eval_root=args.eval_root, output_root=args.output_root)
        print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))
        return 0

    from EmoTrack.conversation.api import run_online_conversation_sessions

    output = run_online_conversation_sessions(config_path=str(args.config))
    print(f"Run completed: {output}")
    print(f"Transcript: {output / 'transcript.jsonl'}")
    print(f"Thinking: {output / 'thinking.jsonl'}")
    print(f"Eval-ready: {output / 'eval_ready_case.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
