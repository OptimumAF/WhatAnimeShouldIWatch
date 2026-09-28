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

    def test_immutable_model_release_transfers_only_six_public_files(self):
        workflow = yaml.safe_load((WORKFLOWS / "publish-model-release.yml").read_text(
            encoding="utf-8"
        ))
        self.assertEqual(workflow["permissions"], {"contents": "read"})
        self.assertFalse(workflow["concurrency"]["cancel-in-progress"])
        inputs = workflow[True]["workflow_dispatch"]["inputs"]
        self.assertEqual(set(inputs), {"tag", "source_run_id", "artifact_name", "base_tag"})
        self.assertTrue(all(item["required"] for item in inputs.values()))
        verify = workflow["jobs"]["verify"]
        publish = workflow["jobs"]["publish"]
        self.assertEqual(verify["permissions"], {"contents": "read", "actions": "read"})
        self.assertEqual(publish["permissions"], {"contents": "write", "actions": "read"})
        self.assertEqual(publish["needs"], "verify")
        gate_parts = ["github.ref == 'refs/heads/master'"]
        for scope in ("TRAINING", "PUBLICATION", "DEPLOYMENT"):
            gate_parts.extend([
                f"vars.PROVIDER_DATA_{scope}_APPROVED == 'true'",
                f"vars.PROVIDER_DATA_{scope}_APPROVAL_REF != ''",
            ])
        gate_parts.extend(["vars.MODEL_PROMOTION_APPROVED == 'true'",
                           "vars.MODEL_PROMOTION_APPROVAL_REF != ''"])
        expected_gate = "${{ " + " && ".join(gate_parts) + " }}"
        for job in (verify, publish):
            gate = job["if"]
            self.assertEqual(gate, expected_gate)
            self.assertIn("github.ref == 'refs/heads/master'", gate)
            for scope in ("TRAINING", "PUBLICATION", "DEPLOYMENT"):
                self.assertIn(f"vars.PROVIDER_DATA_{scope}_APPROVED == 'true'", gate)
                self.assertIn(f"vars.PROVIDER_DATA_{scope}_APPROVAL_REF != ''", gate)
            self.assertIn("vars.MODEL_PROMOTION_APPROVED == 'true'", gate)
            self.assertIn("vars.MODEL_PROMOTION_APPROVAL_REF != ''", gate)
            steps = job["steps"]
            self.assertEqual(steps[0].get("uses"), "actions/checkout@v4")
            self.assertEqual(steps[0].get("with", {}).get("fetch-depth"), 0)
            for scope in ("training", "publication", "deployment"):
                self.assertIn(f"scripts/verify_provider_data_approval.py {scope}",
                              steps[1]["run"])
        verify_steps = verify["steps"]
        publish_steps = publish["steps"]
        verify_names = [step.get("name") for step in verify_steps]
        publish_names = [step.get("name") for step in publish_steps]
        self.assertLess(verify_names.index("Require immutable releases and unused target"),
                        verify_names.index("Download exact public candidate and named data base"))
        self.assertLess(verify_names.index("Verify immutable published data base"),
                        verify_names.index("Verify exact public package and committed approvals"))
        self.assertLess(verify_names.index("Verify exact public package and committed approvals"),
                        verify_names.index("Pass only six verified public assets to write-permission job"))
        self.assertLess(publish_names.index("Reverify package and approvals before publication"),
                        publish_names.index("Recheck immutable setting and unused target"))
        self.assertLess(publish_names.index("Recheck immutable setting and unused target"),
                        publish_names.index("Create new release with six exact audited assets"))
        self.assertLess(publish_names.index("Create new release with six exact audited assets"),
                        publish_names.index("Reverify local approved bytes after release creation"))
        self.assertLess(publish_names.index("Reverify local approved bytes after release creation"),
                        publish_names.index("Verify immutable published release and exact remote hashes"))
        allowed = {"release-manifest.json", "graph.compact.json",
                   "graph-explorer.compact.json", "catalog.identity.json",
                   "model-mf-web.compact.json", "model-promotion-audit.json"}
        upload = verify_steps[verify_names.index(
            "Pass only six verified public assets to write-permission job")]
        self.assertEqual({line.removeprefix("candidate/") for line in
                          upload["with"]["path"].splitlines()}, allowed)
        create = publish_steps[publish_names.index(
            "Create new release with six exact audited assets")]["run"]
        self.assertEqual({name for name in allowed if f"candidate/{name}" in create}, allowed)
        self.assertIn('--target "$GITHUB_SHA"', create)
        for text in ("model.npz", "raw-ratings", "serving-final", "data-latest",
                     "--clobber"):
            self.assertNotIn(text, str(workflow))
        self.assertNotIn("*", create)


if __name__ == "__main__":
    unittest.main()
