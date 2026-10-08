# 0045 — Scoped TV first-airing candidate and private provenance

**Status:** Selected synthetic contract/audit design, 2026-10-07. Real scope/type mappings and application to source-derived catalog years remain proposals under decisions [0043](0043-catalog-source-candidate.md) and [0044](0044-catalog-field-readiness.md). No collection, retention extension, public catalog, model, release or deployment permission. M2.7 remains incomplete.

## Meaning and scope

The product needs a useful year filter, but P580 is a general start time. A format label alone cannot prove that the date is a TV work's first airing. Freeze a separate item declaration and exact type table before interpreting the candidate. The declaration names an unambiguous source item, numeric catalog ID, **series or season**, and `whole-work-first-airing` meaning. An episode, part, franchise, broadcast rerun, localized release or publication edition is outside this candidate. No title heuristic, earliest/latest shortcut, qualifier ignore list, or source-property presence proves the meaning.

The explicit `tv-first-airing-policy-v1` contract names P577 primary, P580 fallback, whole-work first-airing scope, and exact source type IDs mapped to series/season. Every configured type must also map to TV in the independently supplied existing metadata mapping policy. Both that resolved format and **all nondeprecated P31 claims** must agree with the declared work kind and be unqualified known item values. Qualified, missing, conflicting, unmapped or mixed-kind identities remain unknown. A series declaration cannot convert a season's year into a franchise start year.

The separate `declared-tv-first-airing-scope-v1` bytes bind actual source bytes, canonical metadata mapping policy, exact date-policy bytes, and the fixed universe hash. Its sorted unique item entries contain only `animeId`, `sourceItemId`, `kind` and `dateScope`. Unlisted IDs and missing identities remain in the full universe's denominator. These checks prove byte consistency and declared membership; **a declaration is not reviewed semantic truth, owner approval, source rights, or coverage evidence**. Real declarations/type tables still need independent source-scope and use review. They must stay private.

## Resolution rules

| Evidence within declared TV scope | Candidate result |
|---|---|
| P577 has no statements, including an absent property or empty array; usable P580 agrees | Select P580. |
| P577 has usable live evidence; P580 has no statements | Select P577. |
| Both have usable evidence, agreeing on year | Select P577; record its weakest selected precision. No full-date accuracy claim. |
| Invalid, unknown, qualified or conflicting live P577 | Unknown; P580 cannot rescue it. |
| Only deprecated P577 statements remain | Unknown; historical presence is not an absent property for fallback. |
| Present unusable, conflicting or deprecated-only P580 | Unknown, even with usable P577, because the declared comparison cannot be resolved. |
| Both usable properties disagree on year | Unknown; never select an earlier, later or preferred property to conceal it. |
| Neither has usable evidence | Unknown. |

Unlike the preserved v1 best-rank scalar, this **separate candidate** examines all nondeprecated date claims, including normal claims when preferred ones exist. A preferred claim cannot hide a different live release year or unresolved qualifier scope. Identical duplicate evidence and agreeing years at different precision remain usable; deprecated claims do not compete with live claims. This conservatism is an explicit design refinement for first-airing scope, not a change to v1 or its completed-work history.

Only Gregorian, zero-timezone, zero-uncertainty dates are usable. Precision 9 requires year with zero month/day; precision 10 requires a valid month with zero day; precision 11 requires a real calendar day, including leap-year checks. Accept years 1800–3000. Other calendars, precision, noncanonical placeholders, time-of-day, uncertainty, extra date/datavalue fields, invalid dates and unknown snaks remain unknown. The candidate does not infer airing time from fetch, update, import, generation or review timestamps. The stricter precision/placeholders apply only here; v1's existing decoder stays unchanged.

## Private audit boundary

`pipeline/src/tv-date-candidate.ts` validates bounded local bytes and reuses the existing mapper's identity quarantine and metadata validation. It returns `private-tv-first-airing-audit-v1`, with source/mapping/date-policy/scope/universe hashes, fixed-denominator counts, aggregate refusal codes, and private per-ID rows. Each usable row names P577 or P580, weakest selected precision, and an order-independent SHA-256 of canonical selected statements. Rejected rows have null year/property/precision/statement digest and a fixed reason. One refusal per unavailable row reconciles with the full universe count. Raw time strings, qualifier payloads and references are not returned.

Rows retain catalog/source IDs, so **the entire audit is private**, even though raw statements are absent. The function has no transport, file writer, retained cache, acquisition, release, installer, Pages or browser-ranking entry point. `publicationAuthorized` is always false. It does not emit or mutate catalog years, produce readiness success, or manufacture source approval. Independently checking source bytes prevents an authored digest from replacing identity checks, but does not establish first-airing meaning or rights.

The public `anime-metadata-catalog-v1` remains unchanged: one mapped year or null, no private provenance rows. Its strict runtime parser rejects an added `yearProvenance` field. V1/v2 offline mapping and browser year filtering continue to ignore P580, including missing P577 and conflicting P580 examples. The completed pilot did not retain this new property and is not reinterpreted or reprocessed; no acquisition adapter or approval record is extended.

## Verification and next dependency

Invented tests cover primary/fallback absence versus unusability, both-property conflicts, preferred/normal and deprecated ranks, permutation and digest stability, series/season mismatches, unknown/qualified types and dates, calendars/precision/leap days, source/policy/scope/universe tampering, fixed denominators, byte/statement limits and redacted errors. Browser compatibility checks prove this private design does not silently change the existing public year filter. These authored cases establish engineering behavior only.

## Explicit synthetic integration — 2026-10-07

`mapSyntheticScopedMetadata` now accepts only the exact `synthetic-scoped-metadata-candidate-v1` envelope with `purpose: "fixture-only"` and an invented/synthetic source-name prefix. These markers prevent an accidental default call; they cannot prove invention, source meaning or rights and grant no approval. The existing v1/v2 mapper, private date audit, acquisition adapter, installer and publication paths are unchanged.

The wrapper recomputes the base metadata and byte-bound private date audit from the supplied local inputs. Confirmed TV items use only the scoped candidate year; undeclared/unresolved TV scope leaves null even where the base mapper had a known P577 year. Unknown media format also leaves year null, because the wrapper cannot choose the TV or other-format rule. Known Movie/OVA/ONA/Special items retain their existing P577 publication year. Other catalog fields and source-byte identity are preserved. These stricter year choices apply only to the explicit synthetic wrapper, not existing v1/v2 exports or consumers.

The result contains strict catalog `snapshot` and exact UTF-8 `metadataBytes`, with a **separate private** `private-scoped-metadata-audit-v1`. Its metadata SHA-256 binds those actual bytes. The original report is clearly named `baseMappingAudit`, retaining original field issues and counts; final `coverage` is recomputed after year mapping. `yearBasis` counts scoped TV, legacy publication and unknown years among emitted items, while missing identities remain counted in full-universe coverage and the date audit. Do not cite the base mapper's old year counts as final coverage. No accepted identity returns null snapshot/bytes/digest instead of a fabricated empty catalog export.

Neither date rows, selected property/precision/digest, scope bindings, certificate references nor the private audit enter metadata bytes. The strict runtime contract rejects an added private audit or provenance field. Invented tests cover fallback/unknown/conflict mapping, byte identity, unchanged nonyear fields, v2 certificate composition, stale scope refusal, quarantine and empty output. Mocked bundle/browser cases verify positive fallback filtering and conflict/undeclared/unknown-format exclusion plus reversible recovery, without provider requests.

There is no transport, file writer, release/package/installer/Pages entry point or workflow approval. `publicationAuthorized` stays false. This completes the planned synthetic integration seam, not M2.7, real date semantics, permitted catalog completeness, or publication.

**Single next task:** review a concrete new bounded feasibility scope and its exact acquisition/mapping gates against decision 0044, including complete referenced-definition accounting, new selected-property inventory and cleanup prerequisite, before seeking a new owner source/use record. The closed pilot cannot be extended or reinterpreted. No source request begins without that separate authorization; public catalog publication remains separately held.

**Scope review successor (2026-10-07):** [Decision 0046](0046-bounded-followup-study-gates.md) now supplies that concrete unapproved review. No acquisition or real application of this date candidate is authorized. The next technical task is separate synthetic preflight and complete-inventory gate implementation; verified cleanup and a new exact owner record remain prerequisites to execution.
