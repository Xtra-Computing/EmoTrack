from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from EmoTrack.utils.paths import REPO_ROOT


DEFAULT_AGGREGATE_OUTPUT_ROOT = REPO_ROOT / "result" / "metrics"
CORE_COLUMNS = (
    "dataset",
    "experiment_tag",
    "method",
    "seed",
    "session_count",
    "overall_mae",
    "overall_item_mae",
    "metrics_path",
)
SUMMARY_CORE_COLUMNS = (
    "dataset",
    "experiment_tag",
    "method",
    "seed_count",
    "seeds",
    "session_count_min",
    "session_count_max",
    "overall_mae_mean",
    "overall_mae_std",
    "overall_item_mae_mean",
    "overall_item_mae_std",
)
MARKDOWN_COLUMNS = ("Dataset", "Method", "Seed", "Overall MAE", "Symptom MAE")
METRIC_FILE_NAMES = {"metrics.json", "metrics_test.json"}


def aggregate_metrics(
    *,
    eval_root: Path = REPO_ROOT / "out" / "eval",
    output_root: Path | None = None,
) -> dict[str, Path]:
    resolved_eval_root = eval_root.expanduser().resolve()
    resolved_output_root = (
        output_root.expanduser().resolve()
        if output_root is not None
        else DEFAULT_AGGREGATE_OUTPUT_ROOT.expanduser().resolve()
    )
    records = discover_metric_records(resolved_eval_root)
    summary_rows = build_summary_rows(records)

    resolved_output_root.mkdir(parents=True, exist_ok=True)
    by_seed_path = resolved_output_root / "reproduction_metrics_by_seed.csv"
    summary_csv_path = resolved_output_root / "reproduction_metrics_summary.csv"
    summary_md_path = resolved_output_root / "reproduction_metrics_summary.md"
    json_path = resolved_output_root / "reproduction_metrics.json"

    write_csv(by_seed_path, records, core_columns=CORE_COLUMNS)
    write_csv(summary_csv_path, summary_rows, core_columns=SUMMARY_CORE_COLUMNS)
    write_markdown(summary_md_path, summary_rows)
    json_path.write_text(
        json.dumps(
            {
                "eval_root": str(resolved_eval_root),
                "record_count": len(records),
                "summary_count": len(summary_rows),
                "by_seed": records,
                "summary": summary_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "by_seed_csv": by_seed_path,
        "summary_csv": summary_csv_path,
        "summary_markdown": summary_md_path,
        "summary_json": json_path,
    }


def discover_metric_records(eval_root: Path) -> list[dict[str, Any]]:
    if not eval_root.exists():
        return []

    records: list[dict[str, Any]] = []
    for metrics_path in sorted(path for path in eval_root.rglob("metrics*.json") if path.name in METRIC_FILE_NAMES):
        relative_parts = metrics_path.relative_to(eval_root).parts
        if len(relative_parts) < 5:
            continue
        dataset, experiment_tag, method, seed_part = relative_parts[:4]
        if not seed_part.startswith("seed"):
            continue
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        record: dict[str, Any] = {
            "dataset": dataset,
            "experiment_tag": experiment_tag,
            "method": method,
            "seed": _parse_seed(seed_part),
            "session_count": payload.get("session_count"),
            "overall_mae": payload.get("overall_mae"),
            "overall_item_mae": payload.get("overall_item_mae"),
            "metrics_path": str(metrics_path),
        }
        for trend_name, trend_payload in sorted(_mapping(payload.get("trends")).items()):
            trend_metrics = _mapping(trend_payload)
            record[f"trend_mae.{trend_name}"] = trend_metrics.get("mae")
            record[f"trend_session_count.{trend_name}"] = trend_metrics.get("session_count")
        for item_name, item_mae in sorted(_mapping(payload.get("per_item_mae")).items()):
            record[f"per_item_mae.{item_name}"] = item_mae
        records.append(record)

    return sorted(
        records,
        key=lambda row: (
            str(row.get("dataset", "")),
            str(row.get("experiment_tag", "")),
            str(row.get("method", "")),
            int(row.get("seed", -1)),
            str(row.get("metrics_path", "")),
        ),
    )


def build_summary_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for record in records:
        key = (
            str(record["dataset"]),
            str(record["experiment_tag"]),
            str(record["method"]),
        )
        groups.setdefault(key, []).append(record)

    summary_rows: list[dict[str, Any]] = []
    for (dataset, experiment_tag, method), group_records in sorted(groups.items()):
        seeds = sorted({int(record["seed"]) for record in group_records if record.get("seed") is not None})
        session_counts = [
            int(record["session_count"])
            for record in group_records
            if _is_number(record.get("session_count"))
        ]
        row: dict[str, Any] = {
            "dataset": dataset,
            "experiment_tag": experiment_tag,
            "method": method,
            "seed_count": len(seeds),
            "seeds": " ".join(str(seed) for seed in seeds),
            "session_count_min": min(session_counts) if session_counts else "",
            "session_count_max": max(session_counts) if session_counts else "",
            "overall_mae_mean": _mean_metric(group_records, "overall_mae"),
            "overall_mae_std": _std_metric(group_records, "overall_mae"),
            "overall_item_mae_mean": _mean_metric(group_records, "overall_item_mae"),
            "overall_item_mae_std": _std_metric(group_records, "overall_item_mae"),
        }

        dynamic_keys = sorted(
            {
                key
                for record in group_records
                for key in record
                if key.startswith(("per_item_mae.", "trend_mae."))
            }
        )
        for key in dynamic_keys:
            row[f"{key}_mean"] = _mean_metric(group_records, key)
            row[f"{key}_std"] = _std_metric(group_records, key)
        summary_rows.append(row)

    return summary_rows


def write_csv(path: Path, rows: list[dict[str, Any]], *, core_columns: tuple[str, ...]) -> None:
    dynamic_columns = sorted(
        {
            key
            for row in rows
            for key in row
            if key not in core_columns
        }
    )
    fieldnames = list(core_columns) + dynamic_columns
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fieldnames})


def write_markdown(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    lines = ["# Reproduction Metrics Summary", ""]
    if not summary_rows:
        lines.append("No metrics files were found.")
    else:
        lines.append("| " + " | ".join(MARKDOWN_COLUMNS) + " |")
        lines.append("| " + " | ".join("---" for _ in MARKDOWN_COLUMNS) + " |")
        for row in summary_rows:
            values = (
                _dataset_label(row.get("dataset", "")),
                _method_label(row),
                _seed_cell(row.get("seeds", "")),
                _mean_std_cell(row, mean_key="overall_mae_mean", std_key="overall_mae_std"),
                _mean_std_cell(row, mean_key="overall_item_mae_mean", std_key="overall_item_mae_std"),
            )
            lines.append("| " + " | ".join(_markdown_value(value) for value in values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_seed(seed_part: str) -> int:
    raw = seed_part[4:]
    return int(raw) if raw.isdigit() else -1


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _metric_values(records: list[dict[str, Any]], key: str) -> list[float]:
    return [float(record[key]) for record in records if _is_number(record.get(key))]


def _mean_metric(records: list[dict[str, Any]], key: str) -> float | str:
    values = _metric_values(records, key)
    return round(mean(values), 4) if values else ""


def _std_metric(records: list[dict[str, Any]], key: str) -> float | str:
    values = _metric_values(records, key)
    if not values:
        return ""
    if len(values) == 1:
        return 0.0
    return round(stdev(values), 4)


def _dataset_label(dataset: Any) -> str:
    labels = {
        "conversation": "LongCounsel-8",
        "daic_woz": "DAIC-WOZ",
    }
    return labels.get(str(dataset), str(dataset))


def _method_label(row: dict[str, Any]) -> str:
    method = str(row.get("method", ""))
    experiment_tag = str(row.get("experiment_tag", ""))
    dataset = str(row.get("dataset", ""))
    if method == "emotrack" and dataset == "daic_woz":
        if "transfer" in experiment_tag:
            return "emotrack_transfer"
        if "daic" in experiment_tag:
            return "emotrack_daic"
    return method


def _seed_cell(seeds: Any) -> str:
    return ", ".join(str(seeds).split())


def _mean_std_cell(row: dict[str, Any], *, mean_key: str, std_key: str) -> str:
    mean_value = row.get(mean_key)
    if not _is_number(mean_value):
        return ""
    std_value = row.get(std_key)
    if not _is_number(std_value):
        std_value = 0.0
    return f"{float(mean_value):.4f} \u00b1 {float(std_value):.4f}"


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _csv_value(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.4f}"
    return value


def _markdown_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("|", "\\|")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate generated EmoTrack evaluation metrics.")
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=REPO_ROOT / "out" / "eval",
        help="Root containing per-method evaluation outputs (default: out/eval).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Directory for aggregate tables (default: result/metrics).",
    )
    args = parser.parse_args(argv)
    outputs = aggregate_metrics(eval_root=args.eval_root, output_root=args.output_root)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
