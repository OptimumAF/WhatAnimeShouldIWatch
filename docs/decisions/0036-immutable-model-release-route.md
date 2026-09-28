# 0036 — Publish a model only from an exact approved public package

**Status:** M8.4 synthetic and mocked workflow, 2026-09-28. This decision does not approve any provider use, owner review, model, release, or deployment. The model, plan, provider, and data publication approval registries remain empty for real promotion.

## Two verification boundaries

The private `model:release:package:check` gate must first recompute the candidate's raw-split graph, frozen MF parameters, one-use reports, and browser serving scores from restricted inputs. An owner reviews that result and the permitted source/use basis before adding an exact model entry to the committed registry. A Git approval record cannot prove the absence of an unrecorded experiment or replace that private review. No private raw rows, labels, numeric archive, user factors, or evidence directory belongs in the release artifact.

The new `model:release:public:check` gate is deliberately narrower. It accepts only the six-file public package and the separately published five-file data-only base. It verifies exact bounded inventories, candidate/base artifact semantics and byte identity, every public asset hash in the model audit, the approved immutable base's audit hash, provider source/use entries, the exact owner model entry, and the strict earlier plan commit. The dispatch pins the same-repository source run, exact public artifact name, candidate tag, data-base tag, and separate owner approval reference. This recheck guards publication against changed public bytes or an unapproved dispatch; it does not recompute private quality evidence.

## Remote publication order

`publish-model-release.yml` runs only on `master` when training, publication, deployment, and distinct model-promotion repository variables and references are present. Both jobs validate the committed source/use records after checkout. The read-permission job checks repository release immutability and absence of the candidate tag and release before downloading an exact named six-file Actions artifact. It fetches only five named assets from the versioned data base, verifies that remote release is published and immutable with exact SHA-256 asset digests, then checks the public model package and approvals. It passes only six explicit public files to the write-permission job.

Both jobs fetch the full reviewed Git history so the verifier can inspect the strict prior plan-approval commit. A shallow checkout cannot satisfy that chronology check.

The write-permission job repeats the source/use, data-base, public-package, and remote preflight checks. It creates a new release with six explicit paths, then rechecks local approved bytes and GitHub's reported published immutable release, exact asset names, sizes, states, and SHA-256 digests. It never uses `data-latest`, an asset glob, `--clobber`, or automatic recovery of a partial draft/tag. A failed remote mutation requires an owner recovery review while the prior data-only base remains available. The repository's immutable-release setting still needs an owner-provisioned administration-read credential; an absent or denied credential fails before mutation. No repository variable or secret is set by this change.

GitHub's [immutable release setting endpoint](https://docs.github.com/en/rest/repos/repos#check-if-immutable-releases-are-enabled-for-a-repository) requires repository administration read permission. [Release asset metadata](https://docs.github.com/en/rest/releases/releases) includes the `immutable` release field and per-asset SHA-256 digest. [`gh release create`](https://cli.github.com/manual/gh_release_create) uploads through a draft before publishing, so the post-publication check is mandatory. These are mocked in routine tests; do not dispatch this workflow to test a gate.

## Remaining hold

The publication route stops after verified model release. [Decision 0037](0037-exact-tag-pages-deployment.md) adds a separate exact-tag Pages trigger, installer, and mocked hosted check; no real deployment or hosted result is established. Rollback practice remains for M8.7. The current provider permissions and invented serving labels do not support a real model approval or quality claim. M8.4 and M8 remain unchecked.
