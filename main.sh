#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

uv_run() {
  if [[ -f "${ENV_FILE:-.env}" ]]; then
    uv run --env-file "${ENV_FILE:-.env}" "$@"
  else
    uv run "$@"
  fi
}

case "${1:-help}" in
  setup) uv sync --locked ;;
  run-main)
    shift
    dataset="${1:-counselarc}"
    if [[ $# -gt 0 ]]; then shift; fi
    uv_run emotrack-reproduce "$dataset" "$@"
    ;;
  help|-h|--help)
    cat <<'HELP'
Usage:
  bash main.sh setup
  bash main.sh run-main [counselarc|luna|mini|daic|edaic|mini-fn] [options]

Options:
  --seeds 0              Run one seed (default: 0 1 2 3 4)
  --stage prepare        Run download, prepare, features, train, evaluate, or all
  --dry-run              Show commands and configurations without running them
  --overwrite            Recompute evaluation predictions

Examples:
  bash main.sh run-main counselarc --seeds 0
  bash main.sh run-main luna
  bash main.sh run-main edaic --stage evaluate
HELP
    ;;
  *) printf 'Unknown command: %s. Run bash main.sh --help.\n' "$1" >&2; exit 2 ;;
esac
