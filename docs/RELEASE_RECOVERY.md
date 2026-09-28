# Release and recovery playbook

This playbook describes the reviewed routes and the invented staging rehearsal. Decisions [0001](decisions/0001-provider-data-permissions.md), [0002](decisions/0002-provider-workflow-gates.md), [0031](decisions/0031-immutable-publication-workflow.md), [0036](decisions/0036-immutable-model-release-route.md), [0037](decisions/0037-exact-tag-pages-deployment.md), and [0039](decisions/0039-local-rollback-rehearsal.md) remain authoritative for their gates. The committed approval registries currently authorize no real release.

## Three identities to record

| Identity | Meaning | Where to verify it |
| --- | --- | --- |
| Source build | Root package version and commit used to build the app | Build/PR commit and the local diagnostics App label |
| Data release | Immutable `data-v...` tag, content-derived bundle ID, manifest-byte SHA-256, graph format, and catalog identity | Release manifest, `active.json`, immutable release assets, and local diagnostics Data label |
| Optional model | Model format and asset SHA-256 declared by that data release, plus the browser's loaded state | Release manifest and local diagnostics Model label |

A model-bearing bundle has its own data-release tag and manifest. The app can be rebuilt from a different source commit while serving the same data bundle. A data-only bundle has no model file; it cannot borrow one from an earlier release. Browser profiles are local state, not a fourth release asset.

## Refresh data

1. Record a reviewed provider source/use decision and the required use-specific repository approvals. Keep raw interactions, split manifests, user factors, salts, and browser histories out of public packages.
2. Produce the strict aggregate-only v3 graph, explorer, and identity catalog from a permitted snapshot. Audit exact public bytes, provenance, redistribution permission, compatibility, and quality before the owner signs the package approval. Use a separately reviewed first-release approval only for genesis; ordinary releases name a fully verified data-only predecessor.
3. Publish the exact allowlisted data-only package under a new immutable tag. Recheck remote asset SHA-256 values and immutability. A failed check leaves the previous release intact.
4. Dispatch the exact-tag Pages workflow from `master` with the committed approval and the release's original tag, manifest digest, source run, artifact name, and predecessor tag. Review offline gates, the project-path build, and the hosted report for direct navigation, active pointer, exact assets, optional-model absence, gzip observation, and reported security headers. M8.5 remains open until this actual hosted evidence exists.

## Promote a model

1. Use only a permitted, split-first evaluation with the frozen new-user serving policy, one-use final labels, train-plus-validation refit, and independently reviewed quality and lineage evidence. Keep raw rows, user factors, cohorts, and reports in private inputs.
2. Bind the six-file public package to an approved data-only v3 base. The graph, explorer, and catalog bytes stay exact; the model is item-only. Recompute the private archive and public-byte checks, then record exact owner, provider-use, and model-promotion approvals.
3. Publish a new immutable model-bearing tag, verify every remote asset digest, and dispatch Pages with that exact tag and its approved data-only predecessor. Review the hosted model bytes and browser load state. A weekly retrain upload alone does not promote or deploy a model.

## Rehearse a local restore

The routine rehearsal uses only invented assets and a temporary store:

```text
npm run data:fixture:check
node --import tsx --test pipeline/test/install-release-bundle.test.ts
npm run test:e2e --workspace web -- tests/release-rollback.spec.ts
```

The installer test creates a staging store, installs two compatible invented bundles, invokes the local CLI with both exact identities, checks the restored pointer and retained candidate, and reapplies that candidate. It also checks corrupt bytes, stale identities, unexpected files, a lock, an interruption, and a missing older predecessor. The browser test serves that staged store through a local mocked route, migrates invented v1 and v4 profiles on the candidate, restores the data-only prior, and reloads to check the migrated profiles and actual fallback label.

For a reviewed staging store, the same local operation can be run explicitly:

```text
node --import tsx pipeline/src/restore-release-bundle.ts --store <staging-store> --current-tag <tag> --current-bundle-id <64-hex> --current-manifest-sha256 <64-hex> --previous-tag <tag> --previous-bundle-id <64-hex> --previous-manifest-sha256 <64-hex>
```

Read those six values from independently verified manifests and the active pointer. The target must be the active release's immediate predecessor. Keep the target's own named predecessor available when it has one. The command changes only this local store and sends no network request. If verification fails, investigate the named field; do not hand-edit `active.json` or a bundle directory.

## Recover a hosted release

1. Record the active source commit, hosted pointer, release tag, bundle ID, manifest digest, and optional model digest. Check the last hosted verification report and identify the exact previous approved release and that release's own predecessor. Do not copy or upload browser storage, usernames, or histories as part of recovery.
2. Verify the target's immutable release and exact original approval records. If its own predecessor is missing or no longer verifiable, stop before deployment. A corrupt current pointer can be bypassed only by a clean build from the approved target and its verified predecessor, not by editing hosted files.
3. With the required source/use and owner approvals still valid, run `deploy-web.yml` on `master` for that **previous approved tag**, supplying its original manifest SHA-256, source run ID, artifact name, kind, and its own `previous_tag`. Review the workflow's gates and the resulting hosted report. This is a new clean Pages deployment; the local restore CLI is not used on Pages.
4. Reopen the same hosted origin and verify the source/data/model labels, direct navigation, exact asset hashes, model absence or load state, and representative profile behavior. Use invented test profiles for a migration check; do not inspect or collect a real user's browser storage. Record the rollback event and the follow-up release decision separately.

The local rehearsal establishes recovery mechanics and browser compatibility only. It does not prove a real Pages rollback, provider permission, production data quality, CDN cache behavior, gzip delivery, or security headers. Those remain part of the reviewed hosted gate.
