import re
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
GUARDED = {
    "ml-retrain.yml": ("training", {"retrain"}),
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
        trigger = workflow["jobs"]["trigger_pages"]
        self.assertEqual(set(workflow["jobs"]), {"verify", "publish", "trigger_pages"})
        self.assertEqual(verify["permissions"], {"contents": "read", "actions": "read"})
        self.assertEqual(publish["permissions"], {"contents": "write", "actions": "read"})
        self.assertEqual(publish["needs"], "verify")
        self.assertEqual(trigger["needs"], "publish")
        self.assertEqual(trigger["permissions"], {"contents": "read", "actions": "write"})
        publication_gate = "${{ github.ref == 'refs/heads/master' && vars.PROVIDER_DATA_PUBLICATION_APPROVED == 'true' && vars.PROVIDER_DATA_PUBLICATION_APPROVAL_REF != '' }}"
        self.assertEqual(verify["if"], publication_gate)
        self.assertEqual(publish["if"], publication_gate)
        self.assertIn("vars.PROVIDER_DATA_DEPLOYMENT_APPROVED == 'true'", trigger["if"])
        self.assertIn("vars.PROVIDER_DATA_DEPLOYMENT_APPROVAL_REF != ''", trigger["if"])
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
        trigger_steps = trigger["steps"]
        self.assertEqual(trigger_steps[0]["uses"], "actions/checkout@v4")
        self.assertIn("verify_provider_data_approval.py deployment", trigger_steps[1]["run"])
        trigger_run = trigger_steps[2]["run"]
        self.assertLess(trigger_run.index("verify_immutable_data_release.py published"),
                        trigger_run.index("gh workflow run deploy-web.yml"))
        for field in ("kind=data", "tag=$RELEASE_TAG", "manifest_sha256=$MANIFEST_SHA256",
                      "source_run_id=$SOURCE_RUN_ID", "artifact_name=$ARTIFACT_NAME",
                      "previous_tag=$PREVIOUS_TAG"):
            self.assertIn(field, trigger_run)

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
        trigger = workflow["jobs"]["trigger_pages"]
        self.assertEqual(set(workflow["jobs"]), {"verify", "publish", "trigger_pages"})
        self.assertEqual(verify["permissions"], {"contents": "read", "actions": "read"})
        self.assertEqual(publish["permissions"], {"contents": "write", "actions": "read"})
        self.assertEqual(publish["needs"], "verify")
        self.assertEqual(trigger["needs"], "publish")
        self.assertEqual(trigger["permissions"], {"contents": "read", "actions": "write"})
        gate_parts = ["github.ref == 'refs/heads/master'"]
        for scope in ("TRAINING", "PUBLICATION", "DEPLOYMENT"):
            gate_parts.extend([
                f"vars.PROVIDER_DATA_{scope}_APPROVED == 'true'",
                f"vars.PROVIDER_DATA_{scope}_APPROVAL_REF != ''",
            ])
        gate_parts.extend(["vars.MODEL_PROMOTION_APPROVED == 'true'",
                           "vars.MODEL_PROMOTION_APPROVAL_REF != ''"])
        expected_gate = "${{ " + " && ".join(gate_parts) + " }}"
        for job in (verify, publish, trigger):
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
            if job is not trigger:
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
        trigger_run = trigger["steps"][2]["run"]
        self.assertLess(trigger_run.index("verify_immutable_model_release published"),
                        trigger_run.index("gh workflow run deploy-web.yml"))
        for field in ("kind=model", "tag=$RELEASE_TAG", "manifest_sha256=$MANIFEST_SHA256",
                      "source_run_id=$SOURCE_RUN_ID", "artifact_name=$ARTIFACT_NAME",
                      "previous_tag=$BASE_TAG"):
            self.assertIn(field, trigger_run)

    def test_pages_deployment_uses_only_approved_exact_release(self):
        workflow = yaml.safe_load((WORKFLOWS / "deploy-web.yml").read_text(
            encoding="utf-8"))
        self.assertEqual(set(workflow[True]), {"workflow_dispatch"})
        self.assertEqual(set(workflow["jobs"]), {"build", "deploy", "verify_hosted"})
        self.assertEqual(workflow["permissions"], {"contents": "read"})
        self.assertFalse(workflow["concurrency"]["cancel-in-progress"])
        inputs = workflow[True]["workflow_dispatch"]["inputs"]
        self.assertEqual(set(inputs), {"kind", "tag", "manifest_sha256",
                                       "source_run_id", "artifact_name", "previous_tag"})
        self.assertEqual(inputs["kind"]["options"], ["data", "model"])
        build, deploy = workflow["jobs"]["build"], workflow["jobs"]["deploy"]
        hosted = workflow["jobs"]["verify_hosted"]
        self.assertEqual(build["if"], deploy["if"])
        self.assertEqual(build["if"], hosted["if"])
        for key in ("DEPLOYMENT", "PUBLICATION"):
            self.assertIn(f"vars.PROVIDER_DATA_{key}_APPROVED == 'true'", build["if"])
            self.assertIn(f"vars.PROVIDER_DATA_{key}_APPROVAL_REF != ''", build["if"])
        self.assertIn("vars.PROVIDER_DATA_TRAINING_APPROVED == 'true'", build["if"])
        self.assertIn("vars.MODEL_PROMOTION_APPROVED == 'true'", build["if"])
        self.assertEqual(build["permissions"], {"contents": "read"})
        self.assertEqual(deploy["permissions"], {"pages": "write", "id-token": "write"})
        self.assertEqual(deploy["needs"], "build")
        self.assertEqual(hosted["permissions"], {"contents": "read"})
        self.assertEqual(hosted["needs"], "deploy")
        self.assertEqual(hosted["steps"][0]["uses"], "actions/checkout@v4")
        self.assertIn("verify_provider_data_approval.py deployment", hosted["steps"][1]["run"])
        self.assertIn("verify_hosted_pages", hosted["steps"][2]["run"])
        steps = build["steps"]
        names = [step.get("name") for step in steps]
        self.assertEqual(steps[0]["uses"], "actions/checkout@v4")
        self.assertEqual(steps[0]["with"]["fetch-depth"], 0)
        for scope in ("deployment", "publication", "training"):
            self.assertIn(f"verify_provider_data_approval.py {scope}", steps[1]["run"])
        self.assertLess(names.index("Verify source-use approvals before release access"),
                        names.index("Download only named public release assets"))
        self.assertLess(names.index("Install pinned dependencies and run offline gates"),
                        names.index("Download only named public release assets"))
        self.assertLess(names.index("Verify published immutable state and remote asset hashes"),
                        names.index("Recompute approvals and atomically install the exact public bundle"))
        self.assertLess(names.index("Build exact project-path Pages output and verify direct entry"),
                        next(index for index, step in enumerate(steps) if
                             step.get("uses") == "actions/upload-pages-artifact@v3"))
        runs = "\n".join(step.get("run", "") for step in steps)
        for command in ("npm ci", "npm run data:fixture:check", "npm run typecheck",
                        "npm test", "npm run test:python", "npm run test:e2e",
                        "npm run test:pages", "prepare-pages-bundle.ts",
                        "finalize-pages-build.ts"):
            self.assertIn(command, runs)
        for disallowed in ("data:fetch:release", "sync:web", "data-latest",
                           "--clobber", "anonymized-ratings", "model.npz"):
            self.assertNotIn(disallowed, str(workflow))


if __name__ == "__main__":
    unittest.main()
