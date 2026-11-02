"""Regression tests for local data preparation and shared run configuration."""

import csv
import hashlib
import io
import json
import os
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml
import zstandard

from EmoTrack.data.download import download_dataset
from EmoTrack.pipeline.reproduce import build_configs, main


class RefinementTests(unittest.TestCase):
    def test_daic_raw_files_convert_offline_without_changing_splits(self):
        from EmoTrack.data.daic_woz import (
            DEV_LABEL_FILE,
            FULL_TEST_LABEL_FILE,
            PHQ8_SYMPTOM_COLUMNS,
            TEST_SPLIT_FILE,
            TRAIN_LABEL_FILE,
            build_daic_woz_result_dataset,
        )

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "raw"
            (source / "labels").mkdir(parents=True)
            (source / "transcripts").mkdir()
            columns = [
                "Participant_ID",
                "Gender",
                "PHQ8_Binary",
                "PHQ8_Score",
                *[column for column, _ in PHQ8_SYMPTOM_COLUMNS],
            ]
            for filename, participant in [
                (TRAIN_LABEL_FILE, 300),
                (DEV_LABEL_FILE, 301),
            ]:
                with (source / "labels" / filename).open("w") as handle:
                    writer = csv.DictWriter(handle, fieldnames=columns)
                    writer.writeheader()
                    writer.writerow(
                        dict.fromkeys(columns, 1)
                        | {
                            "Participant_ID": participant,
                            "PHQ8_Binary": 0,
                            "PHQ8_Score": 8,
                        }
                    )
            (source / "labels" / TEST_SPLIT_FILE).write_text(
                "participant_ID,Gender\n302,1\n"
            )
            (source / "labels" / FULL_TEST_LABEL_FILE).write_text(
                "Participant_ID,Gender,PHQ_Binary,PHQ_Score\n302,1,0,8\n"
            )
            for participant in range(300, 303):
                (source / f"transcripts/{participant}_TRANSCRIPT.csv").write_text(
                    "start_time\tstop_time\tspeaker\tvalue\n0\t1\tEllie\tyeah3 (yeah)\n1\t2\tParticipant\texample\n"
                )
            output = Path(directory) / "result"
            with patch(
                "socket.socket.connect",
                side_effect=AssertionError("Network use is forbidden"),
            ):
                manifest = build_daic_woz_result_dataset(
                    source_root=source, output_root=output
                )
            self.assertEqual(
                manifest["counts"], {"train": 1, "dev": 1, "test": 1, "total": 3}
            )
            transcript = [
                json.loads(line)
                for line in (output / "300/common/transcript.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(
                [(row["speaker"], row["text"]) for row in transcript],
                [("Counselor", "yeah"), ("Client", "example")],
            )
            self.assertTrue((source / "transcripts/300_TRANSCRIPT.csv").is_file())

    def test_external_download_stage_is_rejected(self):
        for dataset in ("daic", "edaic"):
            with (
                patch("sys.argv", ["reproduce", dataset, "--stage", "download"]),
                patch(
                    "socket.socket.connect",
                    side_effect=AssertionError("Network use is forbidden"),
                ),
            ):
                with self.assertRaises(SystemExit) as raised:
                    main()
                self.assertEqual(raised.exception.code, 2)

    def test_training_paths_are_shared_by_features_and_evaluation(self):
        original_load = yaml.safe_load

        def load_modified(text):
            config = original_load(text)
            if isinstance(config, dict) and "output_dir" in config:
                config.update(
                    output_dir="out/custom",
                    embed_cache_dir="out/custom_embeddings",
                    split_file="result/custom/split.json",
                    aida_cache_path="out/custom_features/seed0/aida.json",
                )
            return config

        with patch(
            "EmoTrack.pipeline.reproduce.yaml.safe_load", side_effect=load_modified
        ):
            train, features, evaluation = build_configs("mini", 2)
            self.assertEqual(
                evaluation["method"]["checkpoint"], "out/custom/seed2/best"
            )
            self.assertEqual(features["dataset"]["split_file"], train["split_file"])
            self.assertEqual(
                features["method"]["aida_cache_path"],
                "out/custom_features/seed2/aida.json",
            )
            self.assertEqual(
                evaluation["method"]["embed_cache_dir"], train["embed_cache_dir"]
            )
            self.assertEqual(
                build_configs("edaic", 2)[2]["method"]["checkpoint"],
                "out/custom/seed2/best",
            )

    def test_completed_download_checks_core_files_and_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split = root / "split.json"
            split.write_text(
                json.dumps({"runs": [{"run_id": "run1", "split": "test"}]})
            )
            data = io.BytesIO()
            with tarfile.open(fileobj=data, mode="w") as archive:
                for filename in (
                    "transcript.jsonl",
                    "ground_truth.json",
                    "self_report.json",
                ):
                    member = tarfile.TarInfo(f"dataset/run1/common/{filename}")
                    member.size = 3
                    archive.addfile(member, io.BytesIO(b"{}\n"))
            packed = root / "data.tar.zst"
            packed.write_bytes(zstandard.ZstdCompressor().compress(data.getvalue()))
            entry = {
                "archive": packed.name,
                "archive_root": "dataset",
                "core_files": 3,
                "archive_sha256": hashlib.sha256(packed.read_bytes()).hexdigest(),
                "split_file": split.name,
                "split_sha256": hashlib.sha256(split.read_bytes()).hexdigest(),
            }
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps({"schema_version": 3, "datasets": {"dataset": entry}})
            )

            def fetch(repo_id, filename, **kwargs):
                return str(root / filename)

            with patch("EmoTrack.data.download.hf_hub_download", side_effect=fetch):
                destination = download_dataset("dataset", destination=root / "custom")
                self.assertEqual(
                    download_dataset("dataset", destination=destination), destination
                )
                core = destination / "run1/common/self_report.json"
                core.unlink()
                with self.assertRaises(FileNotFoundError):
                    download_dataset("dataset", destination=destination)
                core.write_text("{}\n")
                (destination / "train_val_test_split.json").write_text("{}")
                with self.assertRaises(ValueError):
                    download_dataset("dataset", destination=destination)

    def test_missing_features_stop_training(self):
        from EmoTrack.train.aida.dataset import _require_cached_examples

        examples = [{"run_id": "run1", "session_index": 1}]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "run1"):
                _require_cached_examples(
                    examples, Path(directory), {}, require_embed_cache=False
                )
            with self.assertRaisesRegex(FileNotFoundError, "session_1.pt"):
                _require_cached_examples(
                    examples,
                    Path(directory),
                    {"run1::s1": [0.0] * 23},
                    require_embed_cache=True,
                )

    def test_edaic_preparation_repairs_partial_output_using_text_only(self):
        from EmoTrack.data.edaic import _LABEL_ITEM_COLUMNS

        previous_cwd = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "data/edaic"
                (source / "labels").mkdir(parents=True)
                (source / "transcripts").mkdir()
                columns = [
                    "Participant",
                    "Depression_severity",
                    "split",
                    *[name for name, _ in _LABEL_ITEM_COLUMNS],
                ]
                with (source / "labels/detailed_lables.csv").open("w") as handle:
                    writer = csv.DictWriter(handle, fieldnames=columns)
                    writer.writeheader()
                    for participant in range(600, 686):
                        writer.writerow(
                            dict.fromkeys(columns, 0)
                            | {"Participant": participant, "split": "train"}
                        )
                        (
                            source / f"transcripts/{participant}_Transcript.csv"
                        ).write_text("Text\nexample\n")
                output = root / "result/edaic"
                (output / "600/common").mkdir(parents=True)
                configs = build_configs("edaic", 0)
                for config in configs[1:]:
                    config["dataset"]["input_root"] = str(output)
                    config["dataset"]["split_file"] = str(
                        output / "train_val_test_split.json"
                    )
                with (
                    patch("EmoTrack.pipeline.reproduce.REPO_ROOT", root),
                    patch(
                        "EmoTrack.pipeline.reproduce.build_configs",
                        return_value=configs,
                    ),
                    patch("sys.argv", ["reproduce", "edaic", "--stage", "prepare"]),
                    patch("subprocess.run"),
                ):
                    self.assertEqual(main(), 0)
                manifest = json.loads(
                    (output / "train_val_test_split.json").read_text()
                )
                self.assertEqual(len(manifest["runs"]), 86)
                self.assertEqual({row["split"] for row in manifest["runs"]}, {"test"})
                row = json.loads((output / "600/common/transcript.jsonl").read_text())
                self.assertEqual(set(row), {"session_id", "turn_id", "speaker", "text"})
        finally:
            os.chdir(previous_cwd)

    def test_missing_external_files_fail_locally(self):
        from EmoTrack.data.daic_woz import prepare_daic_woz_dataset
        from EmoTrack.data.psyche.files import (
            require_nhanes_dpq_xpt,
            require_psyche_files,
        )

        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "socket.socket.connect",
                side_effect=AssertionError("Network use is forbidden"),
            ),
        ):
            root = Path(directory)
            for operation in [
                lambda: prepare_daic_woz_dataset(
                    source_root=root / "raw", output_root=root / "prepared"
                ),
                lambda: require_psyche_files(root / "psyche"),
                lambda: require_nhanes_dpq_xpt(root / "DPQ_L.xpt"),
            ]:
                with self.assertRaises(FileNotFoundError):
                    operation()


if __name__ == "__main__":
    unittest.main()
