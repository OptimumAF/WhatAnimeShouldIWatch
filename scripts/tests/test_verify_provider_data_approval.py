import tempfile
import unittest
from pathlib import Path

from scripts.verify_provider_data_approval import ApprovalError, validate_approval

APPROVAL_REF = "https://github.com/OptimumAF/WhatAnimeShouldIWatch/issues/123"


class ProviderDataApprovalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        decision = self.root / "docs" / "decisions" / "0002-approved-use.md"
        decision.parent.mkdir(parents=True)
        decision.write_text("# Test approval\n", encoding="utf-8")
        self.entry = {
            "approved": True,
            "sources": ["Synthetic test source"],
            "sourceBasis": "Synthetic test license",
            "use": "Synthetic test use",
            "owner": "Test owner",
            "approvedAt": "2026-09-24",
            "decisionRef": "docs/decisions/0002-approved-use.md",
            "approvalRef": APPROVAL_REF,
        }

    def manifest(self, scope="training"):
        return {"schemaVersion": 1, "approvals": {scope: self.entry.copy()}}

    def test_empty_record_blocks_every_scope(self):
        for scope in ("training", "publication", "deployment"):
            with self.subTest(scope=scope), self.assertRaises(ApprovalError):
                validate_approval({"schemaVersion": 1, "approvals": {}}, scope, APPROVAL_REF, self.root)

    def test_complete_record_enables_only_its_scope(self):
        for scope in ("training", "publication", "deployment"):
            with self.subTest(scope=scope):
                validate_approval(self.manifest(scope), scope, APPROVAL_REF, self.root)
                other = next(item for item in ("training", "publication", "deployment") if item != scope)
                with self.assertRaises(ApprovalError):
                    validate_approval(self.manifest(scope), other, APPROVAL_REF, self.root)

    def test_missing_or_mismatched_record_blocks(self):
        invalid = [
            ("approved", False),
            ("sources", []),
            ("sourceBasis", ""),
            ("use", ""),
            ("owner", ""),
            ("approvedAt", "not-a-date"),
            ("decisionRef", "../outside.md"),
            ("approvalRef", "http://example.com/approval"),
        ]
        for field, value in invalid:
            with self.subTest(field=field), self.assertRaises(ApprovalError):
                manifest = self.manifest()
                manifest["approvals"]["training"][field] = value
                validate_approval(manifest, "training", APPROVAL_REF, self.root)
        with self.assertRaises(ApprovalError):
            validate_approval(self.manifest(), "training", "https://example.com/other", self.root)
        with self.assertRaises(ApprovalError):
            validate_approval(self.manifest(), "training", "", self.root)


if __name__ == "__main__":
    unittest.main()
