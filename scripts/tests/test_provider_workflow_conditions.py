import re
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
GUARDED = {
    "ml-retrain.yml": ("training", {"retrain"}),
    "publish-data-release.yml": ("publication", {"verify", "publish"}),
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
                    if filename == "publish-data-release.yml":
                        self.assertIn("github.ref == 'refs/heads/master' &&", expression)
                        expression = expression.replace(
                            "github.ref == 'refs/heads/master' && ", ""
                        )
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

    def test_immutable_data_release_is_exact_and_approval_gated(self):
        workflow = yaml.safe_load((WORKFLOWS / "publish-data-release.yml").read_text(
            encoding="utf-8"
        ))
        self.assertEqual(workflow["permissions"], {"contents": "read"})
        self.assertEqual(workflow["concurrency"]["cancel-in-progress"], False)
        verify = workflow["jobs"]["verify"]
        publish = workflow["jobs"]["publish"]
        self.assertEqual(verify["permissions"], {"contents": "read", "actions": "read"})
        self.assertEqual(publish["permissions"], {"contents": "write", "actions": "read"})
        self.assertEqual(publish["needs"], "verify")
        self.assertFalse(workflow[True]["workflow_dispatch"]["inputs"]["previous_tag"]["required"])
        verify_steps = verify["steps"]
        publish_steps = publish["steps"]
        verify_names = [step.get("name") for step in verify_steps]
        publish_names = [step.get("name") for step in publish_steps]
        self.assertLess(verify_names.index("Require immutable releases and unused tag"),
                        verify_names.index("Download exact candidate and named prior"))
        self.assertLess(verify_names.index("Recompute audited package and exact approvals"),
                        verify_names.index("Pass only five verified assets to publication job"))
        for steps, download_name, verify_name in [
            (verify_steps, "Download exact candidate and named prior",
             "Recompute audited package and exact approvals"),
            (publish_steps, "Download exact named prior", "Reverify package before publication"),
        ]:
            self.assertIn('if [[ -n "$PRIOR_TAG" ]]',
                          steps[[step.get("name") for step in steps].index(download_name)]["run"])
            verifier = steps[[step.get("name") for step in steps].index(verify_name)]["run"]
            self.assertIn('if [[ -n "$PRIOR_TAG" ]]', verifier)
            self.assertIn('args+=(--previous prior --prior-tag "$PRIOR_TAG")', verifier)
        self.assertLess(publish_names.index("Reverify package before publication"),
                        publish_names.index("Recheck immutable setting and absent target"))
        self.assertLess(publish_names.index("Recheck immutable setting and absent target"),
                        publish_names.index("Create new release with exact audited assets"))
        self.assertLess(publish_names.index("Create new release with exact audited assets"),
                        publish_names.index("Verify immutable published release and asset hashes"))
        self.assertEqual(verify_steps[verify_names.index(
            "Require immutable releases and unused tag")]["env"][
                "RELEASE_IMMUTABILITY_READ_TOKEN"],
            "${{ secrets.RELEASE_IMMUTABILITY_READ_TOKEN }}")
        self.assertEqual(publish_steps[publish_names.index(
            "Recheck immutable setting and absent target")]["env"][
                "RELEASE_IMMUTABILITY_READ_TOKEN"],
            "${{ secrets.RELEASE_IMMUTABILITY_READ_TOKEN }}")
        upload = verify_steps[verify_names.index("Pass only five verified assets to publication job")]
        release = publish_steps[publish_names.index("Create new release with exact audited assets")]
        expected = {"release-manifest.json", "graph.compact.json",
                    "graph-explorer.compact.json", "catalog.identity.json",
                    "publication-audit.json"}
        self.assertEqual({line.removeprefix("candidate/") for line in
                          upload["with"]["path"].splitlines()}, expected)
        self.assertEqual({name for name in expected if f"candidate/{name}" in release["run"]},
                         expected)
        self.assertIn("--target \"$GITHUB_SHA\"", release["run"])
        self.assertNotIn("*", release["run"])
        self.assertNotIn("--clobber", str(workflow))
        self.assertNotIn("anonymized-ratings", str(workflow))
        self.assertNotIn("data-latest", str(workflow))


if __name__ == "__main__":
    unittest.main()
