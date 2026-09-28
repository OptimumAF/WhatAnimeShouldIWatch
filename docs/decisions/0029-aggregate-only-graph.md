# 0029 — Aggregate-only compact graph for a public candidate

**Status:** M8.3 engineering contract fixed on 2026-09-28 before implementation. Invented fixture inputs and mocked browser transport only. It does not approve provider-derived publication.

## Format and derivation

`graph-compact-v3` retains v2 anime-pair semantics, dataset identity, configuration, and exact pair-selection/truncation statistics. Its additional `projection.policy` is `omit-user-anime-v1`. Both recommendation and visualization roles require `userIds: []`, `ua: []`, `userCount: 0`, and node/edge counts for retained anime and pair edges only. `truncation.selectedRatings` still counts the ratings used to produce the pairs; it is **not** redefined as the number of public user-anime edges. The recommendation graph requires `truncation.selectedPairs === aa.length`. The explorer keeps the same bounded pair-selection policy and must name the v3 recommendation graph ID; its excluded user-edge count is zero. Every v3 pair retains signed weight and co-rater support.

A local projector accepts only a validated v2 recommendation graph, copies its anime-pair tuples and provenance/statistics without rounding or recomputation, removes all per-user IDs/edges, and computes a new v3 graph ID over the public fields. It derives a linked v3 explorer from the projected graph. The v3 graph ID does not hash the removed per-user arrays or expose the private v2 graph ID. This is a projection of pair evidence, not a new similarity algorithm or a quality claim. The source rights hold still covers derived pair weights and dataset identity.

The v3 parser requires exact top-level and nested fields, rejects nonempty user arrays, hidden extra fields, malformed pair support, and inconsistent counts. The manifest continues to bind exact bytes and declares the v3 format for both graph roles; it rejects mixed v2/v3 roles. Existing v1/v2 graph and release readers remain valid. Browser ranking may use the pair edges, but the sample-based popularity proxy has no user-anime edges in a v3 release and must be disclosed as unavailable rather than presented as zero global popularity.

## Verification boundary

Use invented source ratings to prove unchanged pair tuples and selection statistics, no per-user IDs/edges in either output, deterministic IDs, manifest/catalog/explorer compatibility, parser refusal for extra or reintroduced history fields, and normal-mode browser loading with mocked transport. Do not publish the invented artifact, read a real history, or infer real recommendation quality. M8.3 still needs the publication audit, rights/quality review, exact output allowlist, immutable upload workflow, and clean-checkout review after this graph contract passes.
