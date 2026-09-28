# 0034 — Verify the aggregate graph against a validated raw split

**Status:** Synthetic-only M8.4 engineering path, 2026-09-28. No provider-derived input, source/use permission, model promotion, or release is approved.

## Private lineage boundary

The public v3 graph's dataset digest is a hash of the **fit membership** rating content, not the split manifest's full-snapshot `rawContentSha256`. Keep both identities. Validate the raw snapshot and exact split manifest together, then take train plus validation for the final refit graph. Exclude every test row. Center each fit user's raw scores with the existing train-only fitter and bind its `refitTrainSha256`, `refitFitSha256`, metadata hash, row counts, and split identity to the private refit record. The fixed metadata snapshot supplies titles; it cannot contain user or rating-derived fields.

Rebuild the recommendation graph with the existing TypeScript pair aggregation, dataset identity, selection statistics, and v3 aggregate projection. Compare the **entire** candidate v3 graph with the recomputed result, including signed weights, co-rater support, configuration, truncation counts, anime IDs and titles, graph ID, and empty user arrays. An invented fit-only SQLite run of the existing graph producer must agree exactly with the projection. A refreshed manifest after a test-only score edit changes `rawContentSha256` but leaves all fit rows and the graph unchanged; a stale manifest or changed fit score/title fails.

The read-only `model:dataset-bridge:check` command accepts private raw ratings, split manifest, fixed metadata, candidate v3 graph, and private refit record. Its CLI allows the exact checked-in invented raw/split/metadata path and pinned bytes (ignoring line-ending conversion); any other inputs require the committed training source/use approval and matching reference before the row preparer runs. Its Python child sends user-bearing fit rows only through a local process pipe, and direct CLI use of that child refuses to print them. The verifier emits only hashes, counts, and graph identity. Keep its inputs and detailed fit rows out of web assets, release assets, logs, and commits. The prepared row pipe is bounded; a larger permitted snapshot needs an explicit streaming/resource review rather than a silent limit increase.

## Promotion hold

This verifier is independent of the present `model-dataset-bridge-v1` assertion in the local package. A future package step must require these private inputs and invoke the verifier before nonfixture staging, bind the verified result to its exact review, and check sidecar fit-user membership against the split fit users. The current invented archive's rows are constructed for tamper tests, not proof of a real fit. A successful synthetic bridge does not establish provider rights, predeclared evaluation timing, an actual model's fit lineage, quality, owner approval, immutable publication, deployment, or rollback. M8.4 remains unchecked.
