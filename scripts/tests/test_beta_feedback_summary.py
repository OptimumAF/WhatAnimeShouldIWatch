"""Invented beta records exercise aggregate counts and private-field refusal."""

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from scripts import beta_feedback_summary as beta


def invented_session(case="sparse-history", device="desktop", code="INVENTED01"):
    return {
        "sessionCode": code,
        "consent": {"notesConsent": True, "date": "2026-10-05", "quoteOptIn": False},
        "case": case,
        "device": device,
        "startingPath": beta.STARTING_PATHS[case],
        "requestedEngine": "none" if case == "new-to-anime" else "graph",
        "actualEngine": "explore-community" if case == "new-to-anime" else "graph",
        "prompts": {name: {"outcome": "unassisted", "minutes": 1} for name in beta.PROMPTS},
        "previewBeforeApply": case == "broad-history",
        "explanationPass": True,
        "savedUnseen": True,
        "savedReturn": "persisted",
        "firstEligibleShown": 10,
        "plausibleCount": 3,
        "appMarkedLeaks": 0,
        "recalledUnrecorded": 0,
        "feedbackCodes": [],
        "stopCodes": [],
        "defectRefs": [],
    }


def invented_root(sessions):
    return {
        "format": "beta-session-records-v1",
        "protocolVersion": 1,
        "buildId": "invented-build",
        "bundleId": "invented-bundle",
        "sessions": sessions,
    }


def encoded(root):
    return json.dumps(root).encode("utf-8")


class BetaFeedbackSummaryTests(unittest.TestCase):
    def test_eight_case_device_records_have_hand_counted_aggregate(self):
        sessions = [invented_session(case, device, f"INVENTED{index:02d}")
                    for index, (case, device) in enumerate(
                        ((case, device) for case in beta.CASES for device in beta.DEVICES), 1)]
        sessions[1]["prompts"]["filter"]["outcome"] = "assisted"
        sessions[1]["explanationPass"] = False
        sessions[2]["firstEligibleShown"] = 8
        sessions[2]["plausibleCount"] = 2
        sessions[3]["firstEligibleShown"] = 8
        sessions[3]["appMarkedLeaks"] = 1
        sessions[3]["recalledUnrecorded"] = 1
        sessions[3]["feedbackCodes"] = ["already-seen", "confusing"]
        sessions[3]["defectRefs"] = ["BETA-001"]
        sessions[4]["feedbackCodes"] = ["disliked", "irrelevant", "prerequisite-missing", "metadata-wrong"]
        root, validated = beta.validate_records(encoded(invented_root(sessions)))
        report = beta.summarize(root, validated)
        self.assertEqual(report["sessionCount"], 8)
        self.assertEqual(report["caseCounts"], {case: 2 for case in beta.CASES})
        self.assertEqual(report["deviceCounts"], {"desktop": 4, "mobile": 4})
        self.assertEqual(report["caseDeviceCounts"]["broad-history"], {"desktop": 1, "mobile": 1})
        self.assertEqual(report["coreJourneyPassCount"], 7)
        self.assertEqual(report["coreJourneyPassByCase"]["sparse-history"], 1)
        self.assertEqual(report["coreJourneyPassByDevice"], {"desktop": 4, "mobile": 3})
        self.assertEqual(report["explanationPassCount"], 7)
        self.assertEqual(report["explanationPassByCase"]["sparse-history"], 1)
        self.assertEqual(report["assessableTenCount"], 6)
        self.assertEqual(report["plausibleThreeOfTenCount"], 6)
        self.assertEqual(report["plausibleThreeOfTenByCase"]["broad-history"], 0)
        self.assertEqual(report["requestedEngineCounts"],
                         {"graph": 6, "model": 0, "hybrid": 0, "none": 2})
        self.assertEqual(report["appMarkedLeakCount"], 1)
        self.assertEqual(report["recalledUnrecordedCount"], 1)
        self.assertEqual(report["assistedPromptCounts"]["filter"], 1)
        self.assertEqual(report["feedbackCounts"]["confusing"], 1)
        self.assertEqual(report["restrictedDefectReferenceCount"], 1)
        self.assertNotIn("INVENTED", json.dumps(report))
        self.assertNotIn("BETA-001", json.dumps(report))
        self.assertTrue(report["releaseDecision"].startswith("not-made"))

    def test_private_fields_duplicate_keys_and_invalid_counts_fail_closed(self):
        clean = invented_root([invented_session()])
        bad_cases = []
        extra = copy.deepcopy(clean)
        extra["sessions"][0]["rawTitle"] = "Invented Private Title"
        bad_cases.append(extra)
        no_consent = copy.deepcopy(clean)
        no_consent["sessions"][0]["consent"]["notesConsent"] = False
        bad_cases.append(no_consent)
        invalid_count = copy.deepcopy(clean)
        invalid_count["sessions"][0]["plausibleCount"] = 11
        bad_cases.append(invalid_count)
        unknown_feedback = copy.deepcopy(clean)
        unknown_feedback["sessions"][0]["feedbackCodes"] = ["Invented Private Title"]
        bad_cases.append(unknown_feedback)
        missing_repro = copy.deepcopy(clean)
        missing_repro["sessions"][0]["firstEligibleShown"] = 9
        missing_repro["sessions"][0]["appMarkedLeaks"] = 1
        bad_cases.append(missing_repro)
        false_ten = copy.deepcopy(clean)
        false_ten["sessions"][0]["appMarkedLeaks"] = 1
        false_ten["sessions"][0]["defectRefs"] = ["BETA-001"]
        bad_cases.append(false_ten)
        lost_without_stop = copy.deepcopy(clean)
        lost_without_stop["sessions"][0]["savedReturn"] = "lost"
        bad_cases.append(lost_without_stop)
        for item in bad_cases:
            with self.subTest(item=item["sessions"][0]), self.assertRaises(beta.RecordError):
                beta.validate_records(encoded(item))
        with self.assertRaisesRegex(beta.RecordError, "duplicate JSON field"):
            beta.validate_records(b'{"format":"x","format":"y"}')
        with self.assertRaisesRegex(beta.RecordError, "nonfinite JSON number"):
            beta.validate_records(encoded(clean).replace(b'"plausibleCount": 3', b'"plausibleCount": NaN'))
        duplicate = invented_root([invented_session(), invented_session()])
        with self.assertRaisesRegex(beta.RecordError, "sessionCode: duplicate"):
            beta.validate_records(encoded(duplicate))

    def test_cli_writes_only_aggregate_outside_checkout_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "records.json"
            target = root / "aggregate.json"
            source.write_bytes(encoded(invented_root([invented_session()])))
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(beta.main(["--input", str(source), "--output", str(target)]), 0)
            output = target.read_text(encoding="utf-8")
            self.assertEqual(json.loads(output)["sessionCount"], 1)
            self.assertNotIn("sessionCode", output)
            self.assertNotIn("consent", output)
            self.assertNotIn("INVENTED01", output)
            self.assertIn("1 sessions", stdout.getvalue())
            with redirect_stderr(io.StringIO()):
                self.assertEqual(beta.main(["--input", str(source), "--output", str(target)]), 1)
            self.assertEqual(json.loads(target.read_text(encoding="utf-8"))["sessionCount"], 1)
            bad_source = root / "bad-records.json"
            bad = invented_root([invented_session()])
            bad["sessions"][0]["rawTitle"] = "Invented Private Title"
            bad_source.write_bytes(encoded(bad))
            rejected = io.StringIO()
            with redirect_stderr(rejected):
                self.assertEqual(beta.main(["--input", str(bad_source),
                                            "--output", str(root / "bad-output.json")]), 1)
            self.assertNotIn("Invented Private Title", rejected.getvalue())
            self.assertFalse((root / "bad-output.json").exists())
        with self.assertRaisesRegex(beta.RecordError, "outside the checkout"):
            beta.outside_checkout(beta.REPO_ROOT / "docs" / "BETA_PROTOCOL.md")
        with self.assertRaisesRegex(beta.RecordError, "outside the checkout"):
            beta.outside_checkout(beta.REPO_ROOT / "docs" / "beta-aggregate.json")

    def test_lost_state_is_counted_as_a_stop_without_exporting_reproduction_ref(self):
        session = invented_session()
        session["savedReturn"] = "lost"
        session["stopCodes"] = ["lost-state"]
        session["defectRefs"] = ["BETA-042"]
        root, sessions = beta.validate_records(encoded(invented_root([session])))
        report = beta.summarize(root, sessions)
        self.assertEqual(report["coreJourneyPassCount"], 0)
        self.assertEqual(report["stopCounts"]["lost-state"], 1)
        self.assertEqual(report["savedReturnPersistedCount"], 0)
        self.assertNotIn("BETA-042", json.dumps(report))


if __name__ == "__main__":
    unittest.main()
