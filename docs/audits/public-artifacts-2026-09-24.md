# Public data artifact audit — 2026-09-24

**Scope:** Read-only inspection of the existing [`data-latest` GitHub release](https://github.com/OptimumAF/WhatAnimeShouldIWatch/releases/tag/data-latest) and [deployed GitHub Pages site](https://optimumaf.github.io/WhatAnimeShouldIWatch/). Three `.gz` release assets were downloaded to a temporary directory outside the repository. The audit computed schemas, counts, and checksums without printing or committing any user ID, title list, rating row, or model vector. No asset, release, page, workflow, or user account was changed.

## Verified release inventory

The release manifest is dated 2026-03-06. All downloaded compressed sizes and SHA-256 checksums matched `data-manifest.json` and `SHA256SUMS.txt` in that release. The table describes the JSON **inside** each gzip file.

| Asset | Compressed bytes; SHA-256 prefix | Format and aggregate contents | Public boundary finding |
|---|---|---|---|
| `anonymized-ratings.compact.json.gz` | 257,261; `26a24f19c68e4df3` | `ratings-compact-v1`: 200 stable 24-character pseudonymous user IDs, 6,487 anime items, and 48,558 user rating tuples. Each tuple has an item index, raw score, and normalized score. Per-user list lengths range from 17 to 1,398. | **Direct per-user histories are public.** Hashing names does not remove the linked rating history or establish anonymity. |
| `graph.compact.json.gz` | 8,429,887; `804803fb9e5e1fd1` | `graph-compact-v1`: the same 200 user IDs, 6,487 anime items, 48,558 `ua` user-anime edges, and 2,000,000 `aa` anime-pair tuples. All `ua` user indexes are valid. `aa` tuples have three values and no pair-support field in this published version. | **Full user-linked graph is public.** Removing the ratings file from a web build alone cannot remove these relationships. The pair tuples cannot be thresholded by observed support from this file. |
| `model-mf-web.compact.json.gz` | 3,595,589; `c0b28305f4536a59` | `model-mf-compact-v1`: 19,053 anime IDs/titles, 19,053 item biases, and 19,053 item embeddings of width 64. No user-named field or user-factor array was found in the top-level artifact. Its generation time predates the ratings/graph files. | No direct per-user rows were observed in this export, but model provenance, training permission, compatibility, and privacy review are unproven. Item-only shape is not by itself publication approval. |

The graph's 200 user IDs exactly match the ratings export's IDs. Mapping tuple indexes back to in-memory user and anime IDs produced 48,558 distinct `(user, anime)` pairs in each file with zero pair differences and zero normalized-weight differences above `0.0001`. This establishes that both public files expose the same per-user relationships without disclosing any member or row in this report.

## Deployed site and workflow paths

- `GET /WhatAnimeShouldIWatch/data/graph.compact.json.gz` returned HTTP 200 and the same decompressed JSON bytes as the release graph, despite different gzip bytes. The deployed full graph therefore exposes the same 200 IDs and 48,558 user-anime edges.
- `GET /WhatAnimeShouldIWatch/data/graph-explorer.compact.json.gz` returned HTTP 200. The sampled explorer file still contains 168 user IDs and 2,500 user-anime edges, plus 8,000 item pairs.
- `GET /WhatAnimeShouldIWatch/data/anonymized-ratings.compact.json.gz` returned HTTP 404 at audit time. This is consistent with `pipeline/src/sync-web.ts` excluding the ratings export from the web build by default; the separate public GitHub release still serves that export.
- `.github/workflows/publish-data-release.yml` explicitly includes ratings, graph, and optional model. `.github/workflows/deploy-web.yml` fetches the release and syncs the graph into the site. `.github/workflows/ml-retrain.yml` is scheduled weekly and fetches the public ratings export to train an artifact. These are existing paths, not actions this audit ran.

## Proposed minimum public payload

Until source permissions and historical release handling are resolved, publish **only synthetic fixture data** for development. A future production bundle should have an explicit allowlist and a separate review for each field:

1. **Catalog:** anime/item IDs and the smallest source-cleared title, year, format, genre, and filter fields needed by the UI, with provenance, version, license/attribution requirements, and a permitted image reference policy. No names, stable user IDs, scores, or per-user rows.
2. **Recommendation neighbors, if approved:** item-to-item aggregates only, after provider/data-use clearance and a documented support/disclosure threshold and graph-semantic review. Do not carry `userIds`, `ua`, user nodes, or an explorer sample of user edges. The current published `aa` tuples lack support, so they cannot simply be copied into a privacy-filtered replacement.
3. **Optional item model, if approved:** item-only parameters with training provenance, allowed source, evaluation, compatibility, and privacy review. Exclude user embeddings/factors and individual histories. Do not promote the current file merely because its top-level schema has no user array.

The replacement build and release workflows should reject any asset outside this allowlist, inspect parsed schema rather than trusting filenames, verify hashes and counts, and test that the deployed site cannot serve restricted files. M8.3/M8.4/M8.5 own that implementation after M2.1 permissions and M3/M4 artifact semantics are settled.

## Remediation decision still needed

The current release and Pages graph contain user-linked records. The repository owner should decide how to handle the existing public release, deployed graph, CDN/cache copies, and affected-data communication after reviewing permissions and privacy implications. Do not delete, overwrite, republish, or deploy an altered production bundle as a by-product of this audit. Before any later promotion, prevent automatic retraining and release/deploy workflows from consuming uncleared assets, and verify the entire replacement path. Local fixture work remains available.

## Reproduction notes and limits

Read-only commands used: `gh release view data-latest --json assets`; `gh release download data-latest --pattern '*.gz'` plus manifest and checksum files to a temporary directory; `gh api repos/OptimumAF/WhatAnimeShouldIWatch/pages`; and `curl` GET/HEAD checks of the three site paths. A local Python `gzip`/`json`/`hashlib` inspection verified the manifest sizes/checksums, top-level keys, tuple widths, counts, valid user indexes, release/site raw-JSON equality, and cross-file user/rating-edge equality. It printed only aggregate statistics. This audit does not determine whether a user is reidentifiable, settle provider rights, prove model privacy, or inventory third-party mirrors and historical caches.
