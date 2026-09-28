"""Safe numeric model exchange and independent parity checks on invented data."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))

from build_content_features import load_model_anime  # noqa: E402
from export_model_web import build_payload, export_model  # noqa: E402
from model_artifact import load_numeric_model, metadata_path, save_numeric_model  # noqa: E402
from model_parity_reference import build_parity_report, parse_parity_spec  # noqa: E402
from recommend_graph_mf import load_model, recommend  # noqa: E402
from train_graph_mf import Dataset, Split, save_model  # noqa: E402


INPUT = ROOT / "fixtures" / "synthetic-model-parity-input.json"
SPEC = ROOT / "fixtures" / "synthetic-model-parity-spec.json"


class NumericModelArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.model_path = Path(self.temporary.name) / "model.npz"
        self.source = json.loads(INPUT.read_text(encoding="utf-8"))["model"]
        self.save()

    def save(self) -> None:
        source = self.source
        save_numeric_model(
            self.model_path,
            p=np.asarray(source["userEmbeddings"], dtype=np.float32),
            q=np.asarray(source["embeddings"], dtype=np.float32),
            bu=np.asarray(source["userBiases"], dtype=np.float32),
            bi=np.asarray(source["biases"], dtype=np.float32),
            global_mean=source["globalMean"], user_ids=source["userIds"],
            anime_ids=source["animeIds"], anime_titles=source["titles"],
            train_user_items=source["trainUserItems"],
        )

    def test_numeric_roundtrip_export_and_local_consumers(self) -> None:
        loaded = load_numeric_model(self.model_path)
        self.assertEqual(loaded.q.shape, (8, 2))
        self.assertEqual(loaded.train_user_items, [{0, 2}])
        self.assertEqual(loaded.anime_titles[0], "Invented Alpha")
        with np.load(self.model_path, allow_pickle=False) as raw:
            self.assertTrue(all(raw[name].dtype.kind in "fi" for name in raw.files))
            self.assertEqual(len(raw.files), 8)
        compact = build_payload(self.model_path, "compact", 8)
        legacy = build_payload(self.model_path, "legacy", 8)
        self.assertEqual(compact["sourceModelSha256"], loaded.archive_sha256)
        self.assertEqual(legacy["sourceModelSha256"], loaded.archive_sha256)
        self.assertEqual(compact["sourceModel"], "model.npz")
        self.assertEqual(legacy["sourceModel"], "model.npz")
        self.assertEqual(compact["embeddings"][0], [1.0, 0.0])
        self.assertEqual(legacy["anime"][0]["title"], "Invented Alpha")
        compact_path = Path(self.temporary.name) / "model-mf-web.compact.json"
        export_model(self.model_path, compact_path, "compact", 8)
        self.assertEqual(json.loads(compact_path.read_text(encoding="utf-8"))["animeIds"],
                         loaded.anime_ids)
        self.assertEqual(load_model_anime(self.model_path)[0], (301, "Invented Alpha"))
        result = recommend(load_model(self.model_path), "", [(301, 1.0)], 2, 0,
                           float("-inf"), {}, 0.0)
        self.assertNotIn(301, [item["animeId"] for item in result["recommendations"]])

    def test_web_export_cli_consumes_sidecar_and_writes_digest(self) -> None:
        output = Path(self.temporary.name) / "cli-model.json"
        completed = subprocess.run(
            [sys.executable, str(ROOT / "ml" / "export_model_web.py"),
             "--model", str(self.model_path), "--out", str(output),
             "--round", "8"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        )
        exported = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(exported["format"], "model-mf-compact-v1")
        self.assertEqual(exported["sourceModelSha256"],
                         load_numeric_model(self.model_path).archive_sha256)
        self.assertEqual(exported["sourceModel"], "model.npz")
        self.assertNotIn("invented-fit-user", completed.stdout + completed.stderr)

    def test_missing_sidecar_and_changed_archive_fail_before_numeric_use(self) -> None:
        sidecar = metadata_path(self.model_path)
        sidecar.unlink()
        with self.assertRaisesRegex(ValueError, "sidecar missing"):
            load_numeric_model(self.model_path)
        self.save()
        self.model_path.write_bytes(self.model_path.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "SHA-256 does not match"):
            load_numeric_model(self.model_path)

    def test_metadata_shapes_duplicates_and_object_array_are_rejected(self) -> None:
        sidecar = metadata_path(self.model_path)
        original = json.loads(sidecar.read_text(encoding="utf-8"))
        wrong = dict(original, factors=3)
        sidecar.write_text(json.dumps(wrong), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "dimensions disagree"):
            load_numeric_model(self.model_path)
        wrong = dict(original, animeTitles=["short"])
        sidecar.write_text(json.dumps(wrong), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "animeTitles"):
            load_numeric_model(self.model_path)
        sidecar.write_text('{"format":"first","format":"second"}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate field format"):
            load_numeric_model(self.model_path)

        with np.load(self.model_path, allow_pickle=False) as raw:
            arrays = {name: raw[name] for name in raw.files}
        arrays["P"] = arrays["P"].astype(object)
        np.savez_compressed(self.model_path, **arrays)
        original["archiveSha256"] = hashlib.sha256(self.model_path.read_bytes()).hexdigest()
        sidecar.write_text(json.dumps(original), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "object arrays|pickle loading"):
            load_numeric_model(self.model_path)

    def test_writer_rejects_nonfinite_duplicate_and_invalid_index_inputs(self) -> None:
        source = self.source
        options = {
            "p": np.asarray(source["userEmbeddings"], dtype=np.float32),
            "q": np.asarray(source["embeddings"], dtype=np.float32),
            "bu": np.asarray(source["userBiases"], dtype=np.float32),
            "bi": np.asarray(source["biases"], dtype=np.float32),
            "global_mean": source["globalMean"], "user_ids": source["userIds"],
            "anime_ids": source["animeIds"], "anime_titles": source["titles"],
            "train_user_items": source["trainUserItems"],
        }
        wrong = dict(options, global_mean=float("nan"))
        with self.assertRaisesRegex(ValueError, "global_mean.*nonfinite"):
            save_numeric_model(self.model_path, **wrong)
        wrong = dict(options, anime_ids=[301, 301, *source["animeIds"][2:]])
        with self.assertRaisesRegex(ValueError, "anime_ids.*unique"):
            save_numeric_model(self.model_path, **wrong)
        wrong = dict(options, train_user_items=[[8]])
        with self.assertRaisesRegex(ValueError, "invalid index"):
            save_numeric_model(self.model_path, **wrong)

    def test_existing_trainer_writer_emits_safe_pair(self) -> None:
        dataset = Dataset(["invented-fit-user"], [301, 302],
                          ["Invented Alpha", "Invented Beta"],
                          [(0, 0, 9.0), (0, 1, 8.0)])
        split = Split(np.array([0], dtype=np.int32), np.array([0], dtype=np.int32),
                      np.array([9.0], dtype=np.float32), [{0}], {}, 0)
        model = {
            "P": np.array([[0.25, 0.5]], dtype=np.float32),
            "Q": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            "bu": np.zeros(1, dtype=np.float32), "bi": np.zeros(2, dtype=np.float32),
            "global_mean": 5.0,
        }
        metrics = {key: 0.0 for key in (
            "evaluated_users", "precision_at_k", "recall_at_k", "hit_rate_at_k", "ndcg_at_k"
        )}
        args = argparse.Namespace(
            ratings="invented", graph="invented", factors=2, epochs=1, lr=0.01,
            reg=0.01, reg_bias=0.01, graph_lambda=0.0, graph_min_abs_weight=0.0,
            graph_sample_rate=1.0, test_ratio=0.2, min_ratings_for_test=2,
            positive_threshold=7, top_k=2, seed=1,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            save_model(Path(self.temporary.name), dataset, split, model, metrics, args, 0)
        self.assertEqual(load_numeric_model(self.model_path).anime_ids, [301, 302])


class ParityReferenceTests(unittest.TestCase):
    def test_pinned_fixture_scores_have_hand_computed_values(self) -> None:
        report = build_parity_report(INPUT, SPEC)
        self.assertEqual([case["id"] for case in report["cases"]],
                         ["signed-and-excluded", "metadata-and-allowlist", "seen-only"])
        self.assertEqual(report["cases"][0]["topKIds"], [302, 307])
        first = next(item for item in report["cases"][0]["eligible"]
                     if item["animeId"] == 302)
        self.assertAlmostEqual(first["score"], 5.625)
        second = report["cases"][1]["eligible"][0]
        self.assertEqual(second["animeId"], 307)
        self.assertAlmostEqual(second["score"], 5.4375)
        self.assertEqual(report["cases"][2]["raw"], [])
        self.assertEqual(report["archiveSha256"],
                         report["model"]["sourceModelSha256"])

    def test_changed_fixture_or_tolerance_refuses_stale_protocol(self) -> None:
        spec = json.loads(SPEC.read_text(encoding="utf-8"))
        changed = dict(spec, scoreAbsoluteTolerance=0.5)
        with self.assertRaisesRegex(ValueError, "Invalid or stale"):
            parse_parity_spec(changed, INPUT)
        with tempfile.TemporaryDirectory() as temporary:
            input_path = Path(temporary) / "input.json"
            input_path.write_text(INPUT.read_text(encoding="utf-8").replace(
                "Invented Alpha", "Invented Changed"), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid or stale"):
                parse_parity_spec(spec, input_path)


if __name__ == "__main__":
    unittest.main()
