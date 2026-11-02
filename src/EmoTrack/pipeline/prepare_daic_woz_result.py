from __future__ import annotations

import argparse
import json
from pathlib import Path

from EmoTrack.data.daic_woz import (
    DEFAULT_RESULT_ROOT,
    DEFAULT_SOURCE_ROOT,
    build_daic_woz_result_dataset,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize canonical DAIC-WOZ run bundles under result/daic_woz "
            "from the raw transcript-only dataset under data/daic_woz."
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help=f"Local DAIC-WOZ root with labels/ and transcripts/, or prepared labels.csv (default: {DEFAULT_SOURCE_ROOT})",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_RESULT_ROOT,
        help=f"Canonical result bundle root (default: {DEFAULT_RESULT_ROOT})",
    )
    parser.add_argument(
        "--split-output",
        type=Path,
        default=None,
        help="Optional explicit path for the split manifest JSON. Defaults to <output-root>/train_val_test_split.json.",
    )
    return parser.parse_args()


def run(
    *,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    output_root: Path = DEFAULT_RESULT_ROOT,
    split_output: Path | None = None,
) -> int:
    manifest = build_daic_woz_result_dataset(
        source_root=source_root,
        output_root=output_root,
        split_output=split_output,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    args = _parse_args()
    return run(
        source_root=args.source_root,
        output_root=args.output_root,
        split_output=args.split_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
