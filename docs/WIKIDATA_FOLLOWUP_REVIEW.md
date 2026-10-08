# Bounded Wikidata follow-up — owner review

**Status: not approved.** This is the concrete one-use proposal for decision [0046](decisions/0046-bounded-followup-study-gates.md). The completed pilot's approval and the cleanup authorization do not authorize this study. M2.7 and the M2 exit remain open.

## Decision to review

Permit one private feasibility study of **fixed public anime IDs 101–200**, independent of personal history, only after the prior pilot is absent and the private path passes inspection. Use official read-only Wikidata endpoints; do not contact MAL, AniList, Jikan or a proxy, follow external-ID links, or expand relationships/hierarchies. No usernames, ratings, viewing history, images or community scores enter this study.

| Scope | Exact proposed bound |
|---|---|
| Identity | `wikidata-followup-101-200-v1`; no second start after reservation, failure or interruption |
| Scope digest | `312e2c78d48b29b41ec552d70a1a98637a8f67b4a6960536bf6a94244b397b7b` |
| Endpoints | `https://query.wikidata.org/sparql` and `https://www.wikidata.org/w/api.php`; identified GET, no credentials/redirects |
| Lookup | Ten batches of ten IDs; LIMIT 201 and refusal above 200 rows per batch; at most 100 unique anime items |
| Main properties | P4086, P31, P136, P577, P580, P1113, P2047, P155, P156, P2756 |
| Terms | Anime en/ja labels/aliases; definitions en labels only; no descriptions/reference payloads |
| Definitions | Complete inventory of at most 100 items in batches of twenty; include live main values, units/calendars and all live qualifier item/unit/calendar roles; fail before definition reads at 101, never truncate |
| Requests/bytes | At most 40 attempts including retries; 4 MiB exposed body per attempt and 16 MiB total, including discarded errors |
| Timing | Serial starts at least two seconds apart; thirty-second whole-fetch/body bound or earlier approval expiry; at most one retry per logical request, maximum sixty-second wait; maxlag pauses at least five seconds; marked cache timeouts fail without retry |
| Private location | `C:\Users\Avery\Documents\ChatGPT\Anime-private\wikidata-followup-101-200-v1` |
| Consumption | Separate protected `wikidata-followup-101-200-v1.reserved.json`; no deletion, repair or overwrite to retry |
| Private source output | Only source projection, English definition labels, inventory, receipt and completion; partial completion/output may remain after failure |
| Retention | Earlier of start plus seven days and the owner's absolute approval expiry; proposed absolute ceiling **October 14, 2026 at 2:43 PM Pacific** (`2026-10-14T21:43:00.000Z`) |
| Reporting/use | Private feasibility review and aggregate counts only; no product cache, real rule promotion, training, public catalog/data/model/package, release or deployment permission |

Readiness still uses the entire declared 100-ID universe: identity 100%, joint genre/year/format 95%, and applicable independently declared detail/classification strata 90%. Missing identities, absent fields and unassessed strata stay visible. Label retrieval is not semantic mapping. Dates, film identity/board/value tables and the exceptions in decisions 0044–0045 still require separate meaning review; the synthetic scoped wrapper cannot disguise source data as invented. A successful small biased study cannot establish a complete catalog or public-use permission.

## Exact execution gates

1. Independently verify that the old pilot directory is absent, without reading its contents. Automatic approval review blocked its exact authorized deletion; manual removal remains due October 14 at 2:43 PM Pacific. Do not retry through another tool. An affirmative authorization is not cleanup evidence.
2. Obtain a **direct, exact human owner source/use approval** for this proposal and expiry. Review its trusted provenance separately from the JSON record, then bind the complete strict record's canonical digest to the verified approval. Set its expiry no later than the proposed absolute ceiling above. The injected verifier is a trust boundary, with **no default production implementation**; an authored digest, declared `approved` flag, fake callback or fixture pass is insufficient. The approval time is the actual consent time, never a prefilled test time. Keep publication/workflow approvals and variables unchanged.
3. The authorized operator may then perform the fixed read-only private-parent/path/ACL probe. Unknown or unsafe access, symlinks/reparse points, prior/target/marker existence or changed approval stop the study. No existing ACL is fixed automatically; any needed actual permission change requires separate exact review. Pinned ancestors, explicit protected current-owner creation and exclusive reservation precede the first source request.
4. Supply explicit official fetch, the tested Node timing ports and the private Windows writer to the reviewed coordinator. There is no routine acquisition npm command, CLI or scheduled job. A verifier/refusal/abort timeout makes no source request. Recheck approval across verification, transport and output; preserve expiry even after failure.
5. On success, flush and read back every protected private file before completion is committed last. On failure, emit only a fixed private phase/outcome, discard in-memory results, and retain consumption. `not-attempted` means this call did not attempt reservation; it does not prove the study is unused. A reservation exception is uncertain; do not invent its persisted start/expiry or retry it. Independently inspect minimal consumption evidence before any retention claim or approved cleanup.
6. Aborting/timing out output may leave partial or completed files, so report `attempted` unless completion was verified. Do not overwrite or repair them. The source outputs must be removed by their deadline; if automatic cleanup is blocked, the owner must remove them manually. Minimal consumption metadata remains independent of source cleanup. No publication or semantic readiness is inferred from a completion marker.

## Reviewable evidence and remaining holds

The gate/inventory, mocked transport and protected writer have separate synthetic tests and reviewed PRs. The coordinator additionally tests rejected owner checks before I/O, late/expired/changed approval, known versus uncertain consumption, failure/no retry, actual Node timer cancellation and a real deadline against a mocked stalled body. The Windows journey composes invented source transport, fake owner verification and actual temporary private ports. These tests prove engineering behavior, not human consent, real parent readiness, real source lineage, coverage or permission.

Current approval remains absent; the old pilot is closed and its cleanup unresolved. Do not execute this proposal until final-head technical review, verified cleanup, exact owner consent and actual private readiness pass. Public catalog/promotion/release decisions remain separate later work. See [PROGRESS.md](PROGRESS.md) for commands, actual counts, commit/PR evidence and checks not run.
