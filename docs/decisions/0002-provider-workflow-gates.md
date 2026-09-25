# 0002 — Provider-data workflow approval gates

**Status:** Proposed for merge, 2026-09-24. No provider-derived training, publication, or deployment use is approved by this record. See [decision 0001](0001-provider-data-permissions.md) for the current source/use holds and the [public artifact audit](../audits/public-artifacts-2026-09-24.md) for the existing exposure.

## Gate

Each provider-derived job has a job-level condition. Both repository variables for that use must be set before a runner starts. An unset variable resolves to an empty string, so the job is skipped. The first step after checkout checks a committed, use-specific approval entry in [`docs/approvals/provider-data.json`](../approvals/provider-data.json). The entry must cite an existing decision file and match the repository approval reference. The current manifest has no entries and fails closed even if someone sets the variables prematurely.

| Use | Jobs | Repository variables required |
|---|---|---|
| Training | `ml-retrain.yml` / `retrain` | `PROVIDER_DATA_TRAINING_APPROVED=true`, `PROVIDER_DATA_TRAINING_APPROVAL_REF=<HTTPS approval URL>` |
| Release publication | `publish-data-release.yml` / `publish` | `PROVIDER_DATA_PUBLICATION_APPROVED=true`, `PROVIDER_DATA_PUBLICATION_APPROVAL_REF=<HTTPS approval URL>` |
| Pages build and deployment | `deploy-web.yml` / `build` and `deploy` | `PROVIDER_DATA_DEPLOYMENT_APPROVED=true`, `PROVIDER_DATA_DEPLOYMENT_APPROVAL_REF=<HTTPS approval URL>` |

The approval entry for the relevant use must contain `approved: true`, nonempty `sources`, `sourceBasis`, `use`, and `owner`, an ISO `approvedAt` date, a `decisionRef` inside `docs/decisions/` that exists in the checkout, and an HTTPS `approvalRef` exactly equal to that use's repository variable. The decision should record permitted fields, cache/retention, attribution, redistribution or training scope, deletion/correction, and the source's applicable terms or consent. The public record may summarize private licensing evidence without copying private documents or personal histories into the repository. A maintainer must review that record and the actual artifact contents before an owner enables the variables. A variable or a syntactically valid record alone is not proof of provider permission.

These are separate authorizations. A training clearance does not enable publication or deployment. The release workflow still contains a path that republishes user-linked ratings and graph edges, and the Pages workflow still copies graph edges. Their approval cannot be inferred from a training permission or from the data being publicly accessible. M8.4 still owns model evaluation, compatibility, and promotion controls.

## Limits and operation

- The repository variables are intentionally unset. Do not set them or dispatch these jobs for development checks. The fixture PR workflow remains independent of the gates.
- A skipped job can appear as a successful workflow run; inspect the job result before treating a run as a completed retrain, release, or deployment. After enabling a use, the verification step must pass before data fetch, training, or publication.
- The gates apply to these GitHub Actions jobs after the change is merged into the branch that runs them. They do not revoke the existing public `data-latest` release or deployed Pages graph, prevent someone from running local commands, or authorize deleting or republishing any current asset. Resolving existing exposure remains an owner decision.
- The checks validate that a reviewed record is present and linked. They cannot establish the legal validity of a source license or the privacy of an output. A separate source/use review and output inspection remain mandatory.

## Safe verification

`python -m unittest discover -s scripts/tests` tests empty, incomplete, wrong-scope, mismatched-reference, and synthetic complete approval records without touching provider endpoints. `python scripts/verify_provider_data_approval.py training` must exit 1 against the committed empty manifest. Workflow syntax and job conditions can be checked with `actionlint` and a local condition matrix; do not dispatch production jobs as a test. The [GitHub Actions variables context](https://docs.github.com/en/actions/reference/workflows-and-actions/contexts#vars-context) and [job conditions](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/control-jobs-with-conditions) document the skip behavior.
