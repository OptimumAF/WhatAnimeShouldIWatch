import re
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
GUARDED = {
    "ml-retrain.yml": ("training", {"retrain"}),
    "publish-data-release.yml": ("publication", {"publish"}),
    "deploy-web.yml": ("deployment", {"build", "deploy"}),
}


class ProviderWorkflowConditionTests(unittest.TestCase):
    def test_every_provider_data_job_has_separate_fail_closed_gate(self):
        for filename, (scope, expected_jobs) in GUARDED.items():
            workflow = yaml.safe_load((WORKFLOWS / filename).read_text(encoding="utf-8"))
            jobs = workflow["jobs"]
            self.assertEqual(set(jobs), expected_jobs, filename)
            prefix = f"PROVIDER_DATA_{scope.upper()}"
            expected_flag = f"{prefix}_APPROVED"
            expected_ref = f"{prefix}_APPROVAL_REF"
            for job_name, job in jobs.items():
                with self.subTest(workflow=filename, job=job_name):
                    expression = job.get("if", "")
                    match = re.fullmatch(
                        r"\$\{\{\s*vars\.([A-Z_]+)\s*==\s*'true'\s*&&\s*vars\.([A-Z_]+)\s*!=\s*''\s*\}\}",
                        expression,
                    )
                    self.assertIsNotNone(match, expression)
                    flag, ref = match.groups()
                    self.assertEqual((flag, ref), (expected_flag, expected_ref))
                    for variables, should_run in (
                        ({}, False),
                        ({flag: "false", ref: "https://example.com/approval"}, False),
                        ({flag: "true"}, False),
                        ({flag: "true", ref: "https://example.com/approval"}, True),
                    ):
                        actual = variables.get(flag, "") == "true" and variables.get(ref, "") != ""
                        self.assertEqual(actual, should_run)

                    if job_name == "deploy":
                        self.assertEqual(job.get("needs"), "build")
                        continue
                    steps = job["steps"]
                    self.assertEqual(steps[0].get("uses"), "actions/checkout@v4")
                    self.assertEqual(
                        steps[1].get("run"), f"python3 scripts/verify_provider_data_approval.py {scope}"
                    )
                    self.assertEqual(
                        steps[1].get("env", {}).get("PROVIDER_DATA_APPROVAL_REF"),
                        f"${{{{ vars.{expected_ref} }}}}",
                    )

    def test_fixture_pr_checks_remain_independent(self):
        workflow = yaml.safe_load((WORKFLOWS / "ci.yml").read_text(encoding="utf-8"))
        job = workflow["jobs"]["verify"]
        self.assertNotIn("if", job)
        runs = "\n".join(step.get("run", "") for step in job["steps"])
        self.assertIn("npm run data:fixture:check", runs)
        self.assertNotIn("data:fetch:release", runs)


if __name__ == "__main__":
    unittest.main()
