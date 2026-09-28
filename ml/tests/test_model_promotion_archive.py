"""Invented private archive gate tests; no provider data or public model write."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))

from export_model_web import build_payload  # noqa: E402
from model_artifact import load_numeric_model, metadata_path, save_numeric_model  # noqa: E402
from prepare_graph_dataset_bridge import prepare  # noqa: E402
from raw_interaction_split import build_split_manifest, load_raw_snapshot  # noqa: E402
from split_first_graph_mf import model_fingerprint  # noqa: E402
from verify_model_promotion_archive import verify_private_archive  # noqa: E402


class PrivatePromotionArchiveTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.archive = self.root / "model.npz"
        self.web_path = self.root / "model-mf-web.compact.json"
        self._save(0.5)
        self._write_evidence()

    def _save(self, second_embedding: float, user_id: str = "invented-fit-user") -> None:
        save_numeric_model(
            self.archive,
            p=np.asarray([[0.25]], dtype=np.float32),
            q=np.asarray([[1.0], [second_embedding]], dtype=np.float32),
            bu=np.asarray([0.125], dtype=np.float32),
            bi=np.asarray([0.0, 0.25], dtype=np.float32),
            global_mean=5.0, user_ids=[user_id], anime_ids=[301, 302],
            anime_titles=["Invented Alpha", "Invented Beta"], train_user_items=[{0, 1}],
        )

    def _write_evidence(self) -> None:
        loaded = load_numeric_model(self.archive)
        self.refit = {
            "numericArchiveSha256": loaded.archive_sha256,
            "numericMetadataSha256": hashlib.sha256(metadata_path(self.archive).read_bytes()).hexdigest(),
            "refitModelSha256": model_fingerprint({
                "P": loaded.p, "Q": loaded.q, "bu": loaded.bu, "bi": loaded.bi,
                "global_mean": loaded.global_mean,
            }),
            "refitRows": 2,
        }
        self._put("refit-record.json", self.refit)
        self._put("serving-cohort.json", {"trainingUsers": ["invented-fit-user"]})
        self._put("model-mf-web.compact.json", build_payload(self.archive, "compact", 8))

    def _put(self, filename: str, value: object) -> None:
        (self.root / filename).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def test_valid_numeric_archive_matches_refit_and_item_export(self) -> None:
        verify_private_archive(self.root, self.web_path)

    def test_changed_bytes_and_missing_sidecar_fail_before_numeric_use(self) -> None:
        self.archive.write_bytes(self.archive.read_bytes() + b"invented tamper")
        with self.assertRaisesRegex(ValueError, "SHA-256 does not match"):
            verify_private_archive(self.root, self.web_path)
        self._save(0.5)
        metadata_path(self.archive).unlink()
        with self.assertRaisesRegex(ValueError, "sidecar missing"):
            verify_private_archive(self.root, self.web_path)

    def test_rehashed_changed_arrays_cannot_keep_the_old_refit_fingerprint(self) -> None:
        original_fingerprint = self.refit["refitModelSha256"]
        self._save(0.75)
        self._write_evidence()
        self.refit["refitModelSha256"] = original_fingerprint
        self._put("refit-record.json", self.refit)
        with self.assertRaisesRegex(ValueError, "refit-record.refitModelSha256"):
            verify_private_archive(self.root, self.web_path)

    def test_rehashed_archive_must_match_the_exported_item_values(self) -> None:
        web = json.loads(self.web_path.read_text(encoding="utf-8"))
        web["embeddings"][1][0] += 0.125
        self._put("model-mf-web.compact.json", web)
        with self.assertRaisesRegex(ValueError, "model-mf-web.compact.json.embeddings"):
            verify_private_archive(self.root, self.web_path)

    def test_fitted_users_and_packed_rows_must_match_private_evidence(self) -> None:
        self._save(0.5, "invented-other-fit-user")
        self._write_evidence()
        with self.assertRaisesRegex(ValueError, "serving-cohort.trainingUsers"):
            verify_private_archive(self.root, self.web_path)
        self._put("serving-cohort.json", {"trainingUsers": ["invented-other-fit-user"]})
        self.refit["refitRows"] = 3
        self._put("refit-record.json", self.refit)
        with self.assertRaisesRegex(ValueError, "refit-record.refitRows"):
            verify_private_archive(self.root, self.web_path)

    def test_packed_fit_items_must_match_validated_raw_rows_at_fixed_count(self) -> None:
        source = self.root / "source"
        source.mkdir()
        shutil.copyfile(ROOT / "fixtures/synthetic-split-input.json",
                        source / "raw-ratings.json")
        shutil.copyfile(ROOT / "fixtures/synthetic-anime-metadata.json",
                        source / "anime-metadata.json")
        manifest = build_split_manifest(load_raw_snapshot(source / "raw-ratings.json"),
                                        policy="seeded", seed=8)
        (source / "split-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        prepared = prepare(source / "raw-ratings.json", source / "split-manifest.json",
                           source / "anime-metadata.json")
        user_ids = sorted({row["userId"] for row in prepared["rows"]})
        anime_ids = list(range(101, 110))
        expected = [{anime_ids.index(row["animeId"]) for row in prepared["rows"]
                     if row["userId"] == user_id} for user_id in user_ids]

        def save(items: list[set[int]]) -> None:
            save_numeric_model(
                self.archive, p=np.zeros((len(user_ids), 1), dtype=np.float32),
                q=np.ones((len(anime_ids), 1), dtype=np.float32),
                bu=np.zeros(len(user_ids), dtype=np.float32),
                bi=np.zeros(len(anime_ids), dtype=np.float32), global_mean=5.0,
                user_ids=user_ids, anime_ids=anime_ids,
                anime_titles=[f"Invented {anime_id}" for anime_id in anime_ids],
                train_user_items=items,
            )
            self._write_evidence()
            self._put("serving-cohort.json", {"trainingUsers": user_ids})
            self.refit["refitRows"] = sum(map(len, items))
            self._put("refit-record.json", self.refit)

        save(expected)
        verify_private_archive(self.root, self.web_path, source)
        changed = [set(items) for items in expected]
        user_index = next(index for index, items in enumerate(changed)
                          if 0 < len(items) < len(anime_ids))
        replacement = next(index for index in range(len(anime_ids))
                           if index not in changed[user_index])
        changed[user_index].remove(min(changed[user_index]))
        changed[user_index].add(replacement)
        save(changed)
        with self.assertRaisesRegex(ValueError, "packed fit items differ from validated raw fit rows"):
            verify_private_archive(self.root, self.web_path, source)


if __name__ == "__main__":
    unittest.main()
