"""Reproduce the paper's selected EmoTrack configurations."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

from EmoTrack.data.bundles import (
    CONVERSATION_FILES,
    INTERVIEW_FILES,
    require_complete_bundle,
)
from EmoTrack.utils.paths import REPO_ROOT, resolve_seed_cache_path

HF_DATASETS = {
    "counselarc": "longcounsel-8-qwen-3.5-35b-a3b",
    "luna": "longcounsel-8-gpt-5.6-luna-low",
    "mini": "longcounsel-8-gpt-5.4-mini-low",
}
CONFIGS = {
    "counselarc": "emotrack.yaml",
    "luna": "emotrack_luna.yaml",
    "mini": "emotrack_mini.yaml",
    "mini-fn": "emotrack_mini_fn.yaml",
    "daic": "emotrack_daic.yaml",
}


def build_configs(name: str, seed: int) -> tuple[dict | None, dict, dict]:
    config_name = "daic" if name == "edaic" else name
    training = yaml.safe_load(
        (REPO_ROOT / "configs/train" / CONFIGS[config_name]).read_text()
    )
    # The original run numbered zero used the seed stored in its training YAML.
    training.update(inference_seed=seed, seed=training["seed"] if seed == 0 else seed)
    checkpoint = str(Path(training["output_dir"]) / f"seed{seed}" / "best")
    if name == "edaic":
        evaluation = yaml.safe_load(
            (REPO_ROOT / "configs/evaluate/edaic_emotrack.yaml").read_text()
        )
        training = None
    else:
        evaluation = {
            "dataset": {
                "name": "daic_woz" if name == "daic" else "conversation",
                "input_root": str(Path(training["split_file"]).parent),
                "split_file": training["split_file"],
                "transcript_view": "full",
            },
            "method": {
                "aida_cache_path": training["aida_cache_path"],
                "embed_cache_dir": training["embed_cache_dir"],
            },
        }
    method = evaluation["method"]
    method["aida_cache_path"] = resolve_seed_cache_path(method["aida_cache_path"], seed)
    method.update(
        name="emotrack",
        mode="aida",
        inference_seed=seed,
        checkpoint=checkpoint,
        include_history=name not in {"daic", "edaic"},
    )
    if training is not None:
        training["aida_cache_path"] = method["aida_cache_path"]
    suite = yaml.safe_load((REPO_ROOT / "configs/evaluate/default.yaml").read_text())
    llm = suite["llm"]
    llm["base_url"] = os.environ.get("AIDA_BASE_URL", llm["base_url"])
    if "AIDA_GPUS" in os.environ:
        llm["worker_visible_gpus"] = [os.environ["AIDA_GPUS"]]
    llm["vllm"]["seed"] = seed
    data_name = "mini" if name == "mini-fn" else name
    features = {
        "dataset": evaluation["dataset"],
        "method": {
            "name": "aida",
            "extract_features_only": True,
            "inference_seed": seed,
            "aida_cache_path": method["aida_cache_path"],
        },
        "llm": llm,
        "output": {"experiment_tag": f"{data_name}_features"},
    }
    evaluation["llm"] = {
        "backend": "cache_only",
        "model_name": llm["model_name"],
        "base_url": "",
    }
    evaluation["output"] = {"experiment_tag": name.replace("-", "_")}
    return training, features, evaluation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=[*CONFIGS, "edaic"])
    parser.add_argument(
        "--stage",
        choices=["download", "prepare", "features", "train", "evaluate", "all"],
        default="all",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved commands and configs without writing or running anything",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute detector predictions; feature caches are reused",
    )
    args = parser.parse_args()
    if any(seed < 0 for seed in args.seeds):
        parser.error("seeds must be nonnegative")
    if args.dataset == "edaic" and args.stage == "train":
        parser.error("E-DAIC is external evaluation only; train DAIC first")
    name = args.dataset
    data_name = "mini" if name == "mini-fn" else name
    if args.stage == "download" and data_name not in HF_DATASETS:
        parser.error(
            "Obtain access at https://dcapswoz.ict.usc.edu/ and place your files under data/; see README"
        )
    os.chdir(REPO_ROOT)
    training, features, evaluation = build_configs(name, args.seeds[0])
    root = Path(evaluation["dataset"]["input_root"])
    split_file = Path(evaluation["dataset"]["split_file"])
    required_files = CONVERSATION_FILES if data_name in HF_DATASETS else INTERVIEW_FILES
    stages = (
        ["prepare", "features", "train", "evaluate"]
        if args.stage == "all"
        else [args.stage]
    )

    def run(module: str, *arguments: str) -> None:
        command = [sys.executable, "-m", module, *map(str, arguments)]
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)

    if args.stage == "download" or (
        args.stage == "all"
        and data_name in HF_DATASETS
        and (not root.exists() or (root / "download.json").is_file())
    ):
        print(f"Download/verify {HF_DATASETS[data_name]} from Hugging Face", flush=True)
        if not args.dry_run:
            from EmoTrack.data.download import download_dataset

            download_dataset(HF_DATASETS[data_name], destination=root)
    if "prepare" in stages:
        try:
            require_complete_bundle(root, split_file, required_files)
        except (FileNotFoundError, json.JSONDecodeError):
            if name == "daic":
                run(
                    "EmoTrack.pipeline.prepare_daic_woz_result",
                    "--source-root",
                    "data/daic_woz",
                    "--output-root",
                    str(root),
                    "--split-output",
                    str(split_file),
                )
            elif name == "edaic":
                print("Build E-DAIC bundles from local data/edaic", flush=True)
                if not args.dry_run:
                    from EmoTrack.data.edaic import build_edaic_result_dataset

                    build_edaic_result_dataset(
                        source_root=Path("data/edaic"),
                        output_root=root,
                        split_file=split_file,
                    )
            elif not args.dry_run:
                raise
        if not args.dry_run:
            require_complete_bundle(root, split_file, required_files)
        run(
            "EmoTrack.pipeline.train_emotrack",
            "--embed-cache",
            "--dataset-name",
            "daic" if name in {"daic", "edaic"} else "longcounsel8",
            "--root",
            str(root),
            "--output",
            evaluation["method"]["embed_cache_dir"],
        )
        if training is not None:
            run(
                "EmoTrack.pipeline.train_emotrack",
                "--build-daic-dataset" if name == "daic" else "--build-dataset-only",
                "--root",
                str(root),
                "--output",
                training["dataset_path"],
            )
    if not args.dry_run and any(stage in stages for stage in ("features", "evaluate")):
        require_complete_bundle(root, split_file, required_files)
    for seed in args.seeds:
        training, features, evaluation = build_configs(name, seed)
        for stage, config in [
            ("features", features),
            ("train", training),
            ("evaluate", evaluation),
        ]:
            if stage not in stages or config is None:
                continue
            path = Path(f"out/reproduce/{name}/seed{seed}/{stage}.yaml")
            if args.dry_run:
                print(json.dumps({"config": str(path), "settings": config}, indent=2))
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(yaml.safe_dump(config, sort_keys=False))
            if stage == "train":
                run("EmoTrack.pipeline.train_emotrack", "--config", str(path))
            else:
                run(
                    "EmoTrack.pipeline.evaluate",
                    str(path),
                    *(["--overwrite"] if args.overwrite else []),
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
