"""Small CPU checks for the public reproduction entry points."""

import asyncio
import io
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import torch
import yaml
import zstandard

from EmoTrack.data.download import extract_archive
from EmoTrack.evaluate.core.config import load_evaluation_config
from EmoTrack.evaluate.runners.aida import run_aida
from EmoTrack.evaluate.utils import (
    result_detector_artifact_dir,
    result_detector_path_for_row,
)
from EmoTrack.pipeline.reproduce import build_configs
from EmoTrack.train.aida.config import load_hybrid_config
from EmoTrack.train.aida.model import predict_from_cache_hybrid
from EmoTrack.train.aida.trainer import _false_negative_loss


class ReleaseTests(unittest.TestCase):
    def test_inference_respects_checkpoint_utterance_cap(self):
        class RecordingModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))
                self.config = SimpleNamespace(max_sentences=2)
                self.seen = None

            def forward(self, features, embeddings, mask):
                self.seen = embeddings.shape[1]
                return torch.zeros(1, 8), None

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "daic" / "600"
            cache.mkdir(parents=True)
            torch.save({"client_embeddings": torch.zeros(5, 4)}, cache / "session_1.pt")
            model = RecordingModel()
            predict_from_cache_hybrid(
                model, {"600::s1": [0.0] * 23}, root, [("600", 1)]
            )
            self.assertEqual(model.seen, 2)

    def test_interview_artifacts_stay_with_the_selected_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for dataset in ("daic_woz", "edaic"):
                row = SimpleNamespace(
                    dataset_name=dataset, source_path=root, run_id="600"
                )
                score = SimpleNamespace(dataset_name=dataset, source_run_path=str(root))
                self.assertEqual(result_detector_artifact_dir(score), root)
                self.assertEqual(
                    result_detector_path_for_row(
                        row, method_name="emotrack", transcript_view="full"
                    ),
                    root / "seed0" / "emotrack.json",
                )

    def test_external_feature_extraction_does_not_fit_a_regressor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_file = root / "config.yaml"
            config_file.write_text(yaml.safe_dump(build_configs("edaic", 0)[1]))
            config = load_evaluation_config(config_file)
            with (
                patch(
                    "EmoTrack.evaluate.runners.aida.extract_aida_features",
                    new_callable=AsyncMock,
                    return_value={},
                ) as extract,
                patch("EmoTrack.evaluate.runners.aida.fit_aida_model") as fit,
            ):
                scores = asyncio.run(
                    run_aida(
                        dataset_name="edaic",
                        train_rows=[],
                        val_rows=[],
                        test_rows=[],
                        out_root=root,
                        config=config,
                        llm_workers=None,
                    )
                )
                self.assertEqual(scores, [])
                extract.assert_awaited_once()
                self.assertEqual(
                    extract.call_args.kwargs["cache_path"],
                    Path(config.method.aida_cache_path),
                )
                fit.assert_not_called()

    def test_all_resolved_configs_are_accepted_by_their_entry_points(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("counselarc", "luna", "mini", "mini-fn", "daic", "edaic"):
                for index, config in enumerate(build_configs(name, 0)):
                    if config is None:
                        continue
                    path = Path(directory) / "config.yaml"
                    path.write_text(yaml.safe_dump(config))
                    loader = (
                        load_hybrid_config if index == 0 else load_evaluation_config
                    )
                    loader(path)

    def test_false_negative_penalty_uses_full_batch(self):
        predictions = torch.tensor([8.0, 12.0, 4.0], requires_grad=True)
        targets = torch.tensor([10.0, 15.0, 5.0])
        loss = _false_negative_loss(predictions, targets, 10.0)
        self.assertAlmostEqual(loss.item(), 2 / 3)
        loss.backward()
        torch.testing.assert_close(predictions.grad, torch.tensor([-1 / 3, 0.0, 0.0]))

    def test_datasets_have_separate_caches_and_splits(self):
        roots = set()
        for name in ("counselarc", "luna", "mini", "daic"):
            train, features, evaluation = build_configs(name, 1)
            self.assertEqual(train["seed"], 1)
            self.assertEqual(train["inference_seed"], 1)
            self.assertEqual(train["split_file"], features["dataset"]["split_file"])
            self.assertEqual(evaluation["dataset"], features["dataset"])
            roots.add(train["embed_cache_dir"])
        self.assertEqual(len(roots), 4)

    def test_original_training_seed_is_distinct_from_run_id(self):
        self.assertEqual(build_configs("daic", 0)[0]["seed"], 42)
        self.assertEqual(build_configs("luna", 0)[0]["seed"], 0)

    def test_edaic_uses_daic_checkpoint_and_external_features(self):
        train, features, evaluation = build_configs("edaic", 3)
        self.assertIsNone(train)
        self.assertTrue(features["method"]["extract_features_only"])
        self.assertEqual(
            evaluation["method"]["checkpoint"], "out/emotrack_daic/seed3/best"
        )
        self.assertIn("edaic", evaluation["method"]["aida_cache_path"])
        self.assertIn("edaic", evaluation["method"]["embed_cache_dir"])

    def test_archive_rejects_traversal_and_links(self):
        for name, kind in [
            ("../outside", tarfile.REGTYPE),
            ("dataset/link", tarfile.SYMTYPE),
        ]:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                data = io.BytesIO()
                with tarfile.open(fileobj=data, mode="w") as archive:
                    member = tarfile.TarInfo(name)
                    member.type = kind
                    member.linkname = "/tmp/outside"
                    archive.addfile(member)
                path = root / "data.tar.zst"
                path.write_bytes(zstandard.ZstdCompressor().compress(data.getvalue()))
                with self.assertRaises(ValueError):
                    extract_archive(path, root / "result", "dataset")
                self.assertFalse((root / "outside").exists())


if __name__ == "__main__":
    unittest.main()
