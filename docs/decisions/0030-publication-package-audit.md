# 0030 — Audited data-only publication package

**Status:** M8.3 local packaging contract fixed on 2026-09-28 before implementation. Only invented candidate bundles and mocked approval evidence may be used in routine checks. The legacy upload job stays disabled and provider-derived publication remains held by decisions 0001, 0002, and 0028.

## Input and output boundary

The packager accepts one fully verified decision-0026 bundle with `graph-compact-v3` recommendation and visualization files, `model: null`, and an exact source directory inventory: `release-manifest.json`, `graph.compact.json`, `graph-explorer.compact.json`, `catalog.identity.json`. It rejects any extra entry, nested directory, symlink, ratings file, SQLite file, salt, model, training-user factor, or hidden file. A real candidate must name and verify a separate prior bundle; only a marked invented fixture may use genesis. It checks the existing compressed/plain size limits even though the package contains plain JSON.

The packager copies only those four verified artifact files into a new temporary output, then writes `publication-audit.json` and atomically moves the complete output into an unused destination. It never globs the source, overwrites a destination, updates a GitHub release, or treats an existing mutable tag as a candidate. The output inventory is exactly five files. The audit lists exact path, byte length, and SHA-256 for each of the four other files, including the manifest. It is itself an immutable release asset but cannot self-hash.

## Review and computed audit

A separate strict review input binds the tag, bundle ID, manifest-byte hash, source name and snapshot digest, derivation, decision reference, redistribution status/basis, publication approval reference and owner where applicable, permitted public fields, attribution and deletion/correction handling, a change summary and named prior, and evidence references for structural/privacy/quality checks. The packager compares these bindings with the verified bundle and derives compatibility and counts directly from the bytes: dataset ID, graph IDs, catalog map digest, anime count, pair count, and minimum co-rater support. It refuses a missing, stale, contradictory, or unrecognized field. Free-text review claims and approval references are recorded for human audit; their presence alone never proves legal permission or scientific quality.

An invented fixture review has `redistribution.status: synthetic-only`, no approval reference, and produces `publishable: false`. A real `reviewed-allowed` review must match a separately validated publication approval entry and its decision reference. The committed approval manifest is empty, so no current package is release-ready. A mocked approval object may exercise the real-mode comparison in tests, but that does not change the committed hold. M8.3 still needs an immutable, approval-gated upload workflow and inspected rights/quality evidence before its checkbox can pass.

Use adversarial invented cases for extra files, links, ratings/model/factor names, malformed v3 payloads, stale manifest/prior/review bindings, changed asset bytes, missing quality evidence, approval mismatch, and output recovery. Do not dispatch a release job or copy a synthetic package into public assets.
