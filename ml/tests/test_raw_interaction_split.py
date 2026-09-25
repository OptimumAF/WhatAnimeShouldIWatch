import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))
from raw_interaction_split import (  # noqa: E402
    build_split_manifest,
    interaction_id,
    parse_raw_snapshot,
    partition_snapshot,
)


class RawInteractionSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.input_path = ROOT / "fixtures" / "synthetic-split-input.json"
        cls.golden_path = ROOT / "fixtures" / "synthetic-split-manifest.json"
        cls.input = json.loads(cls.input_path.read_text(encoding="utf-8"))

    def test_pinned_fixture_deduplicates_before_three_way_allocation(self):
        snapshot = parse_raw_snapshot(self.input)
        manifest = build_split_manifest(snapshot)
        golden = json.loads(self.golden_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest, golden)
        self.assertEqual(manifest["uniqueInteractions"], 13)
        self.assertEqual(manifest["duplicateRowsDropped"], 1)
        self.assertEqual(manifest["usersWithHoldout"], 3)
        self.assertNotIn("invented-", json.dumps(manifest))
        self.assertEqual([len(manifest[key]) for key in ("trainIds", "validationIds", "testIds")],
                         [7, 3, 3])
        train, validation, test = [set(manifest[key]) for key in
                                   ("trainIds", "validationIds", "testIds")]
        self.assertFalse(train & validation or train & test or validation & test)
        self.assertIn(interaction_id("invented-sparse", 109), train)
        partitions = partition_snapshot(snapshot, manifest)
        self.assertEqual([len(partitions.train), len(partitions.validation), len(partitions.test)],
                         [7, 3, 3])
        self.assertTrue(all(isinstance(row.raw_score, float) for row in partitions.train))
        tampered = copy.deepcopy(manifest)
        tampered["testIds"][0] = tampered["trainIds"][0]
        with self.assertRaisesRegex(ValueError, "does not match"):
            partition_snapshot(snapshot, tampered)

    def test_order_and_scores_cannot_change_membership(self):
        baseline = build_split_manifest(parse_raw_snapshot(self.input))
        reordered = copy.deepcopy(self.input)
        reordered["interactions"].reverse()
        self.assertEqual(build_split_manifest(parse_raw_snapshot(reordered)), baseline)
        changed = copy.deepcopy(self.input)
        changed["interactions"][1]["rawScore"] = 3
        changed_manifest = build_split_manifest(parse_raw_snapshot(changed))
        for key in ("trainIds", "validationIds", "testIds", "identitySha256"):
            self.assertEqual(changed_manifest[key], baseline[key])
        self.assertNotEqual(changed_manifest["rawContentSha256"], baseline["rawContentSha256"])
        with self.assertRaisesRegex(ValueError, "does not match"):
            partition_snapshot(parse_raw_snapshot(changed), baseline)

    def test_existing_exports_use_raw_score_and_ignore_centered_fields(self):
        legacy = {"users": [
            {"userId": "invented-a", "ratings": [
                {"animeId": 101, "rawScore": 9, "normalizedScore": -100},
                {"animeId": 102, "rawScore": 8, "normalizedScore": 100},
                {"animeId": 103, "rawScore": 3, "normalizedScore": 200},
            ]},
        ]}
        compact = {"format": "ratings-compact-v1",
                   "anime": [[101, "A"], [102, "B"], [103, "C"]],
                   "users": [["invented-a", [[0, 9, 999], [1, 8, -999], [2, 3, 0]]]]}
        a = build_split_manifest(parse_raw_snapshot(legacy))
        b = build_split_manifest(parse_raw_snapshot(compact))
        for key in ("trainIds", "validationIds", "testIds", "identitySha256",
                    "rawContentSha256"):
            self.assertEqual(a[key], b[key])
        compact["users"][0][1][0][2] = -700
        self.assertEqual(build_split_manifest(parse_raw_snapshot(compact)), b)

    def test_conflicts_and_malformed_raw_rows_fail_closed(self):
        conflicting = copy.deepcopy(self.input)
        conflicting["interactions"][-1]["rawScore"] = 1
        with self.assertRaisesRegex(ValueError, "Conflicting duplicate"):
            parse_raw_snapshot(conflicting)
        for value, message in (
            ({"users": [{"userId": "u", "ratings": [
                {"animeId": 1, "normalizedScore": 2}]}]}, "rawScore"),
            ({"format": "ratings-compact-v1", "anime": [[1, "A"]],
              "users": [["u", [[2, 9, 1]]]]}, "anime index"),
            ({"format": "other", "users": []}, "Unsupported"),
            ({"format": "raw-interactions-v1", "timestampBasis": "none",
              "interactions": [{"userId": "u", "animeId": 1, "rawScore": float("nan")}]},
             "rawScore"),
            ({"format": "raw-interactions-v1", "timestampBasis": "none",
              "interactions": [{"userId": "u", "animeId": 1, "rawScore": 10**400}]},
             "rawScore"),
            ({"format": "raw-interactions-v1", "timestampBasis": "none",
              "interactions": [{"userId": "\ud800", "animeId": 1, "rawScore": 8}]},
             "userId"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                parse_raw_snapshot(value)

    def test_temporal_requires_verified_event_time_and_respects_order(self):
        rows = [{"userId": "invented-time", "animeId": item, "rawScore": 10 - item,
                 "eventAt": f"2026-01-0{item}T12:00:00Z"}
                for item in range(1, 6)]
        verified = {"format": "raw-interactions-v1",
                    "timestampBasis": "verified-rating-or-viewing", "interactions": rows}
        manifest = build_split_manifest(parse_raw_snapshot(verified))
        self.assertEqual(manifest["policy"], "verified-event-time-v1")
        self.assertEqual(manifest["seed"], None)
        self.assertIn(interaction_id("invented-time", 5), manifest["testIds"])
        self.assertIn(interaction_id("invented-time", 4), manifest["validationIds"])
        self.assertEqual(set(manifest["trainIds"]),
                         {interaction_id("invented-time", item) for item in (1, 2, 3)})
        unverified = copy.deepcopy(verified)
        unverified["timestampBasis"] = "none"
        for row in unverified["interactions"]:
            row["fetchedAt"] = row["eventAt"]
        self.assertEqual(build_split_manifest(parse_raw_snapshot(unverified))["policy"],
                         "seeded-per-user-v1")
        with self.assertRaisesRegex(ValueError, "verified"):
            build_split_manifest(parse_raw_snapshot(unverified), policy="temporal")
        incomplete = copy.deepcopy(verified)
        incomplete["interactions"][0].pop("eventAt")
        with self.assertRaisesRegex(ValueError, "eventAt"):
            parse_raw_snapshot(incomplete)
        tied = copy.deepcopy(verified)
        tied["interactions"][-1]["eventAt"] = tied["interactions"][-2]["eventAt"]
        with self.assertRaisesRegex(ValueError, "tie"):
            build_split_manifest(parse_raw_snapshot(tied))
        tied_validation = copy.deepcopy(verified)
        tied_validation["interactions"][-2]["eventAt"] = tied_validation["interactions"][-3]["eventAt"]
        with self.assertRaisesRegex(ValueError, "tie"):
            build_split_manifest(parse_raw_snapshot(tied_validation))

    def test_cli_never_overwrites_and_check_is_read_only(self):
        script = ROOT / "ml" / "raw_interaction_split.py"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "split.json"
            args = [sys.executable, str(script), "--input", str(self.input_path),
                    "--out", str(output)]
            first = subprocess.run(args, capture_output=True, text=True, check=False)
            self.assertEqual(first.returncode, 0, first.stderr)
            original = output.read_bytes()
            second = subprocess.run(args, capture_output=True, text=True, check=False)
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual(output.read_bytes(), original)
            checked = subprocess.run(args + ["--check"], capture_output=True, text=True,
                                     check=False)
            self.assertEqual(checked.returncode, 0, checked.stderr)
            self.assertEqual(output.read_bytes(), original)
            output.write_text("{}\n", encoding="utf-8")
            stale = subprocess.run(args + ["--check"], capture_output=True, text=True,
                                   check=False)
            self.assertNotEqual(stale.returncode, 0)
            invalid = Path(directory) / "invalid.json"
            invalid.write_text('{"format":"raw-interactions-v1","format":"other"}', encoding="utf-8")
            no_output = Path(directory) / "no-output.json"
            failed = subprocess.run([sys.executable, str(script), "--input", str(invalid),
                                     "--out", str(no_output)], capture_output=True, text=True,
                                    check=False)
            self.assertNotEqual(failed.returncode, 0)
            self.assertFalse(no_output.exists())
            public_output = ROOT / "web" / "public" / "data" / "split-m5-test.json"
            public = subprocess.run([sys.executable, str(script), "--input", str(self.input_path),
                                     "--out", str(public_output)], capture_output=True, text=True,
                                    check=False)
            self.assertNotEqual(public.returncode, 0)
            self.assertFalse(public_output.exists())


if __name__ == "__main__":
    unittest.main()
