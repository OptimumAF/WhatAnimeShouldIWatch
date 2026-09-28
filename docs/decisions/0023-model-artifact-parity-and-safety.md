# 0023 — Safe model exchange and Python/browser parity

**Status:** M5.8 protocol fixed on 2026-09-28 before its synthetic export or parity run. M5.8 and the M5 exit gate remain open.

## Restricted parity contract

`fixtures/synthetic-model-parity-spec.json` pins the newline-normalized SHA-256 of an invented eight-title, two-factor input fixture, compact web format, eight-decimal export precision, absolute score tolerance of `1e-5`, tie rule, and three cases. Save its model as a numeric-only NPZ plus a JSON sidecar, export compact web JSON from that validated pair, then score the same exported item arrays independently in Python and through the browser's `parseCompactModel`, preference-vector scorer, central eligibility policy, and final selector. Compare every pre-policy candidate ID and score, every eligible candidate ID and score, displayed top-K IDs, and exclusion reasons/sets. Cover positive and negative preferences, Seen/history/exclude/Include Only, metadata filtering, equal scores, and a Seen-only no-vector case. Report an absolute maximum score delta and fail above the declared tolerance. No validation/test labels or provider data enter this check.

The Python reference may mirror the browser equation, but it must consume the exported numeric values rather than the TypeScript output. Keep rank ordering deterministic on score, support count, strongest contribution, then model source order. Do not change production ranking semantics to make the fixture pass. Inspect the actual browser implementation if the independent result differs.

## Artifact boundary

New MF and LightGCN NPZ archives contain only finite numeric arrays: `P`, `Q`, `bu`, `bi`, `global_mean`, `anime_ids`, packed train-item indices, and offsets. Put user IDs and anime titles in a versioned JSON sidecar with array counts, factor count, and SHA-256 of the NPZ bytes. A reader must verify the hash, exact array membership/types/shapes, index bounds, and metadata lengths before use. Read NPZ with `allow_pickle=False` and reject old object-array archives by default with an actionable error; do not load pickles as an implicit compatibility fallback. Update the MF web exporter, local recommender, and content-feature title reader to use this boundary. Add the sidecar to the existing approval-gated retraining artifact upload without dispatching that workflow.

The browser's existing compact and legacy JSON readers stay valid. A new optional `sourceModelSha256` field in exported JSON identifies the validated source archive; the runtime parser checks its syntax and still checks dimensions and finite values. The browser cannot verify source bytes it does not receive, so this field alone is not an integrity guarantee. The restricted parity harness checks that the digest matches the temporary NPZ. Fail present malformed web artifacts with their file/field named.

Run hand-computed and tampered-artifact tests, the full synthetic/mock gate, and fresh PR CI before checking M5.8. The legacy full-graph training and LightGCN quality metrics remain invalid M5 evidence. The synthetic parity check establishes implementation agreement and artifact handling, not ranking quality or permission to train/release provider data.
