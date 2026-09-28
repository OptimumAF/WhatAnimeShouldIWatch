# 0028 — Public release audit and per-user-data boundary

**Status:** M8.3 safety contract recorded on 2026-09-28. It authorizes synthetic fixtures and mocked checks only. Decisions 0001 and 0002 still hold provider-derived publication; this record is not a source license or an owner approval.

## Publication boundary

The current `data-latest` publishers are legacy and must not be used for a new release. They can upload `anonymized-ratings.compact.json.gz`, and the current compact v2 graph carries `userIds` and per-user `ua` edges. A pseudonymous ID does not make those edges safe public history. Disable the legacy local and workflow upload routes before they can stage or mutate a release. Keep the existing public release untouched; its remediation requires an owner decision.

A future data-only release must use a distinct immutable `data-v...` tag and a fully verified decision-0026 bundle. Its public asset allowlist is exactly `release-manifest.json`, `catalog.identity.json`, `graph.compact.json`, `graph-explorer.compact.json`, and one publication audit file. The two graph files must contain no per-user identifiers, per-user edges, raw scores, or training-user factors; the candidate may not include a model, ratings file, SQLite file, salt, hidden file, symbolic link, or nested path. Copy only verified allowlisted bytes to a fresh staging directory. Never use a glob over a source directory to choose public assets. A release tag that already exists cannot be updated or clobbered by the data-only path.

The audit file must bind the exact tag, bundle ID, manifest-byte SHA-256, and asset hashes and sizes. It must identify the source snapshot and derivation, an owner-reviewed decision and publication approval reference, the permitted public fields and redistribution basis, attribution and deletion/correction handling, changes since the named prior bundle, graph/catalog compatibility, and recorded quality checks with evidence and denominators. A boolean `approved` field or a syntactically valid record cannot substitute for the reviewed source/use decision, repository approval gate, or inspection of actual release bytes. The source and audit records may summarize private license evidence but must not contain credentials, usernames, histories, or salts.

## Dependency exposed by current graph contract

`graph-compact-v2` equates retained recommendation `ua` edges with `truncation.selectedRatings`. Erasing user rows from a graph with pair statistics while leaving that count unchanged fails its runtime contract; changing the count would misstate which ratings produced the pairs. M8.3 therefore needs a separately versioned aggregate-only recommendation/export contract and matching manifest/browser checks before a useful public candidate can pass. Do not relabel a v2 graph or use an empty-user dummy to claim that publication quality is ready. M8.3 remains unchecked until this dependency, an auditable package, negative privacy cases, workflow integration, and clean-checkout tests pass.

All work before source/use clearance uses invented data and mocked GitHub responses. No workflow dispatch, production asset download, release mutation, or deployment is an M8.3 test.
