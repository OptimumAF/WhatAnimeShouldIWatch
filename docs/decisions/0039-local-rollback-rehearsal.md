# 0039 — Verify and rehearse recovery before changing a release pointer

**Status:** M8.7 invented local staging rehearsal, 2026-09-28. This decision grants no provider use, publication, model promotion, Pages deployment, or access to a real profile.

## Local restore contract

The forward installer deliberately requires a candidate's `lastKnownGood` to identify the active release. Recovery therefore has a separate local operation. The operator supplies the exact tag, bundle ID, and manifest-byte SHA-256 for both the active bundle and its immediate predecessor. Under the install lock, the restore operation verifies the active pointer and complete bundle, the predecessor's exact files and manifest, and the predecessor's own named prior when one exists. A missing older prior, stale identity, corrupt byte, unexpected stored file, or concurrent install fails before activation. Only then does it atomically replace `active.json`; it retains the candidate directory so a corrected candidate can be reinstalled. It makes no HTTP request and reads or writes no browser profile.

The local CLI is `node --import tsx pipeline/src/restore-release-bundle.ts ...`. It is for an existing staging store whose exact identities have been reviewed. Reusing an old tag with changed bytes or moving a pointer around validation is not a rollback. The installer now requires an exact stored-file inventory for the active bundle, so a stray private file cannot be served from that directory as part of a recovery attempt.

## Pages and profile boundary

GitHub Pages receives a fresh static build; its deployed files cannot be recovered by changing a local staging pointer. Once source/use and owner approvals exist, hosted recovery must run the existing exact-tag Pages workflow on `master` for the previously approved immutable release, using that release's original manifest digest, source run, artifact name, and its own recorded predecessor. The workflow rechecks approvals, downloads only its declared public files, constructs a clean store, and runs hosted verification. A model-to-data rollback deliberately removes the model. The local CLI does not dispatch that workflow, waive its gates, or prove CDN behavior.

Profiles belong to browser storage on the same origin and are independent of the release pointer. The rehearsal loads invented v1 and v4 named profiles on a candidate bundle, checks v5 migration and exact untouched raw backups, restores the prior data-only bundle, reloads, and checks profile state and the labeled fallback engine. It tests compatibility on invented assets, not private users or real provider histories. The source build revision, data tag/bundle digest, and optional model digest are separate identities; rolling back data does not imply rolling back source code or browser storage.

The [release and recovery playbook](../RELEASE_RECOVERY.md) records the data refresh, model promotion, recovery, and hosted review sequence. M8.5 and the M8 exit gate still require an approved real release and actual hosted evidence.
