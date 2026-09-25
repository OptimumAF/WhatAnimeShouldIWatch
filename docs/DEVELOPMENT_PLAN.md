# WhatAnimeShouldIWatch — Living Development Plan

- Prepared: September 24, 2026
- Repository: `OptimumAF/WhatAnimeShouldIWatch`
- Reviewed branch: `master`
- Reviewed commit: `dea9a40e87d9864a3ae991c759a193ee43addebc`
- Recommended repository location: `docs/DEVELOPMENT_PLAN.md`
Status: Active living plan; only verified implementation tasks are checked.

## 1. Mission and scope

Turn the existing recommendation lab into a dependable anime-discovery application: a user adds or imports their viewing history, expresses what they liked and disliked, gets useful unseen recommendations with honest explanations, saves a shortlist, and returns without losing their preferences.

Keep the network explorer as an advanced explanation and exploration tool. Preserve the current TypeScript/SQLite pipeline, Vite/TypeScript web application, Python recommendation experiments, and Rust/Dioxus desktop project unless measured constraints justify a change. Do not start with a React/Next.js rewrite, a backend migration, a larger crawler, or a more complex model.

Default delivery scope is the web experience on the existing static-hosting architecture. Desktop is a maintained, separately gated deliverable, not a requirement for every web improvement. This prioritization is a planning assumption, not a previously approved product decision; record a change if the repository owner chooses otherwise.

### The first useful release

A first-time user can select a few genuinely liked titles or import a supported list, see a ranked set of eligible unseen anime, understand the strongest reasons, apply reliable filters, save a watchlist, reload safely, and recover when a provider or optional model is unavailable. Core recommendations must not require an account, an LLM, or live third-party metadata requests.

### Explicit non-goals for the first release

Do not add social feeds, chat, subscriptions, streaming playback, cloud accounts, recommendation chatbots, distributed training, or production GNN infrastructure. Do not prioritize a native desktop feature-parity rewrite. Keep these in a clearly separate backlog until the core recommendation loop and release gates are demonstrated.

## 2. Instructions for every implementing agent

**This is a living execution document, not a one-time proposal. Keep it accurate as work proceeds.**

1. At the beginning of each session, read repository-level and directory-level `AGENTS.md` instructions, this plan, the latest handoff, and relevant decisions. Inspect the actual branch, current commit, local changes, and applicable issues/PRs before editing. The reviewed commit above is a reference point, not a requirement to reset newer work.
2. Compare each task against current code. Preserve working implementations. When a feature already exists, verify it against the task's acceptance criteria rather than rebuild it. Source presence alone is not evidence of runtime correctness.
3. Select the smallest dependency-ready task or coherent slice. Mark it in progress in the progress table; do not start unrelated rewrites or exhaust the whole roadmap in one patch.
4. Add or update the relevant test, implement the change, run verification, inspect the diff, and update documentation. Use synthetic fixtures for routine tests rather than live scraping or production-sized data.
5. **Immediately change a task from `[ ]` to `[x]` once its acceptance criteria actually pass.** Record the verification commands, result, date, affected files, and commit/PR reference when available. Keep implementation and progress documentation in the same change set. A milestone is complete only when its required tasks and exit gate pass.
6. Do not check off work merely because code compiles, a function exists, or one happy-path example works. Partial, untested, unavailable-environment, and externally blocked work remains unchecked, with a precise explanation.
7. Adjust this plan when evidence changes the best approach. Tasks may be split, reordered, added, clarified, or deferred. Preserve task IDs; do not reuse them. Record the reason, evidence, impact, replacement task, and changed dependencies in the change log. Do not quietly weaken acceptance criteria to make a failing task pass.
8. Mark obsolete tasks `superseded` or `deferred`, not falsely complete. Keep them discoverable and exclude them explicitly from active completion totals. Previously completed tasks that regress must be reopened or linked to a new regression task.
9. Ask for approval before material product-direction changes, paid infrastructure, production deployments, deleting datasets, public data/model publication, new external processing of user data, or bulk collection. This planning request is not authorization for those actions. For ordinary reversible implementation details, choose a reasonable default and document it instead of stalling.
10. Preserve unrelated user changes. Never overwrite an existing `AGENTS.md`; merge only the relevant workflow guidance. Do not include usernames, raw personal lists, salts, credentials, or private data in logs, fixtures, screenshots, commits, or public artifacts.
11. End each session with completed task IDs, verification evidence, unfinished work, blockers, plan revisions, and the single next recommended task. Never claim tests were run when they were not.

### Required planning records

Use this document as the canonical checklist. Keep a concise `docs/PROGRESS.md` for execution evidence and handoffs, and `docs/decisions/` for consequential architectural/data decisions. Keep the number of planning files small; do not create competing copies of the roadmap.

Suggested progress record:

```markdown
## Session YYYY-MM-DD
Branch / starting commit:
Tasks worked:
Completed:
Verification commands and actual results:
Manual checks / artifacts:
Files changed:
Commit or PR, when available:
Blockers / checks not run:
Plan revisions and reasons:
Next task:
```

Suggested task-change record:

```markdown
Date | Task ID | Previous scope/status | New scope/status | Evidence/reason | Dependency impact
```

Status vocabulary: `pending`, `in progress`, `blocked`, `complete`, `deferred`, `superseded`. Only verified `complete` tasks receive `[x]`.

## 3. Source-review baseline

This plan is based on repository source and configuration inspection, not a successful local build, live-site test, desktop launch, or model-training run. The implementation agent must establish those baselines. Some large files were inspected in targeted sections rather than audited exhaustively.

### Existing capabilities to retain and verify

- `pipeline/`: MAL collection/network expansion, SQLite storage, per-user normalization, graph construction, compact exports, release-data tooling, and web-data synchronization.
- `web/src/main.ts`: Graph/Model/Hybrid ranking, watched-title weights, manual and username imports, local profiles, include/exclude candidate controls, metadata filters, recommendation explanations, seasonal suggestions, keyboard commands, themes, high-contrast controls, and SVG network exploration.
- `ml/`: graph-regularized matrix factorization, model export, Optuna search, content-feature experiments, CLI recommendations, and a LightGCN evaluation script.
- `desktop/`: a Dioxus graph viewer with local legacy-dataset loading and an embedded sample fallback.
- `.github/workflows/`: web deployment, desktop builds/releases, data publishing, and scheduled model retraining.

### Findings driving priority

**Graph aggregation is order-dependent.** `pipeline/src/build-graph.ts` updates an existing pair weight using `(current + userPairScore) / 2`, not a count-weighted mean. For observations 1, 3, 5, the result is 3.5; reversed, it is 2.5; the arithmetic mean is 3. The edge-count cap also keeps first-encountered pair keys rather than an explicitly selected best-neighbor set. Fix determinism first, then evaluate whether the pair statistic is useful at all.

**The evaluation path can leak held-out information.** The graph builder normalizes over a user's entire list, the training loader consumes `normalizedScore`, and the trainer splits those interactions while loading an independently prebuilt graph. In the documented full-dataset workflow, holdout information can affect both normalization and graph regularization. Treat historical metrics as provisional until split-first evaluation is rebuilt.

**Training evaluation and browser inference are different.** The trainer evaluates learned user factors. `buildModelRecommendations` constructs a new user's vector by averaging selected item embeddings. Good metrics for trained-user factors do not establish that this browser path works well. Evaluate both explicitly.

**Signed graph semantics need a decision.** The training graph loader takes the absolute value of edge weights and uses them as attractive regularization. This erases the distinction between negative and positive input edges. First define what the edges mean; do not assume a negative co-rating statistic means either similarity or dissimilarity.

**Metadata availability affects eligibility.** Browser recommendations prefetch a limited leading subset of candidates, while active filters reject candidates without cached metadata. This can hide valid candidates solely because their metadata was not loaded. Move filter-critical metadata into a validated, permitted catalog artifact or use an explicitly complete bounded retrieval strategy.

**Imports discard meaningful distinctions.** The inspected username-import paths retain positive scored entries mapped into the graph but do not retain full viewing statuses or unknown-catalog items. The browser MAL path also silently falls back to `r.jina.ai`. Preserve watch history separately from positive preference, preserve unmapped items, and make all external routing explicit.

**The web entry point is large and tightly coupled.** The inspected tree reports `web/src/main.ts` at 171,639 bytes, with application state, UI, providers, scoring, loading, and graph rendering together. Extract tested boundaries incrementally, without changing frameworks merely for structure.

**Artifact and desktop behavior need explicit contracts.** `sync-web.ts` writes gzip by default and can remove plain JSON, whereas the browser depends on `DecompressionStream` before its plain-JSON fallback. The desktop reader currently tries legacy rating-file paths and otherwise loads a sample. Scheduled retraining uploads workflow artifacts; that alone does not promote a compatible model to the deployed website.

**Test entry points need establishing.** The inspected root and web package scripts do not define a test command; pipeline defines typechecking. This is not a claim that every possible test location has been exhaustively searched. Inventory first, then provide a dependable offline test suite.

## 4. Architecture and data boundaries

Preserve the existing stacks. Introduce boundaries only as needed for the next tested change.

Proposed TypeScript organization:

```text
web/src/
  app/                    # boot, navigation, orchestration
  domain/                 # catalog, watch-history, profile types
  data/                   # manifests, loaders, validation, migrations
  providers/              # provider adapters, shared request scheduling
  recommendations/        # graph, model, hybrid, eligibility, explanations
  features/               # onboarding, results, profiles, watchlist, filters
  network/                # subgraph selection and SVG interactions
  workers/                # measured CPU-heavy work
  ui/                     # small reusable accessible UI pieces
pipeline/src/
  cli/                    # command entry points as extraction becomes useful
  core/                   # pure normalization, aggregation, validation
  providers/              # ingestion adapters and scheduling
  artifacts/              # release manifests and public/private export rules
schemas/                  # language-neutral artifact/profile contracts
fixtures/                 # small synthetic, redistributable fixtures
ml/                       # retain Python; separate reusable evaluation modules
```

These are proposed paths, not claims about files that already exist. Avoid a large directory-only refactor. Start with pure ranking and graph-building logic, then import adapters and storage.

### Canonical data model

Use a stable internal `animeId` with explicit provider-ID mappings; initially retaining MAL IDs as the primary ID is reasonable because existing artifacts use them. Preserve provider IDs that have no mapping instead of coercing them to another title.

Separate: catalog metadata; viewing status/progress; rating/preference; hard exclusions; explanation evidence; recommendation model; and local application state. A watched title is not automatically a liked title, a dropped title is not automatically a negative rating, and an unrated item is not a zero score.

Version public data, recommendation models, and local profile schemas independently. Public artifacts should carry `schemaVersion`, `datasetVersion`, generation/source timestamps, builder version/commit, item-map identity, row/count summaries, hashes, compressed/uncompressed sizes, and provenance. Models additionally need split/training identities, preprocessing configuration, factor dimensions, evaluation results, and compatible catalog identities.

Separate public item-level recommendation artifacts from private/restricted training records. Stable pseudonymous user IDs plus full rating histories are not a promise of anonymity. Do not distribute individual histories simply because the names were hashed. Review current provider permissions before collecting, caching, training on, or redistributing their data.

## 5. Milestones and checkable tasks

Task IDs are stable. Each task inherits the general definition of done in Section 7 and its milestone's exit gate. The agent may split larger items while preserving their parent ID and recording replacement subtasks.

### M0 — Establish a reproducible baseline

**Priority:** P0. **Dependencies:** None. **Outcome:** The next agent can run a safe, useful local instance and distinguish current failures from regressions.

- [x] **M0.1 — Inspect the actual working state.** Read instructions, branch/commit, uncommitted changes, recent history, open issues/PRs, manifests, release-data scripts, and workflows. Record differences from this reviewed snapshot. Do not reset to the reviewed SHA.
- [x] **M0.2 — Record baseline commands and outcomes.** Run dependency installation, pipeline typecheck, web production build, and available checks. Attempt desktop checks only in a compatible environment. Record exact versions, failures, and environment limitations separately.
- [x] **M0.3 — Introduce a synthetic fixture dataset.** Include overlapping tastes, opposite preferences, equal scores, sparse users, an empty user, isolated items, duplicated input, unknown IDs, and non-ASCII titles. Keep it small and free of real user lists.
- [x] **M0.4 — Add a fixture-backed development path.** Generate a demo graph/catalog/model fixture and run the web experience without crawling or downloading production data. Distinguish demo data visually. Never silently replace real-data failures with a demo.
- [x] **M0.5 — Establish automated test entry points.** Reuse existing tests where found; otherwise add TypeScript unit/integration tests, a browser smoke test, and Python tests around parsing and ranking. Keep routine tests offline and deterministic.
- [x] **M0.6 — Add pull-request CI.** Run install, pipeline typecheck, web typecheck/build, synthetic-fixture validation, and fast tests without secrets or provider requests. Keep expensive training and production-data jobs separate.
- [x] **M0.7 — Capture the existing user journeys.** Record expected behavior for manual selection, graph/model/hybrid mode, imports, profile persistence, filters, and network inspection. A failing baseline is documented rather than disguised as a passing test.

**Exit gate:** A fresh checkout can serve a fixture-backed site and run the documented fast checks. Missing production datasets or provider access do not block ordinary UI and recommendation development.

### M1 — Extract testable contracts and repair immediate safety failures

**Priority:** P0/P1. **Dependencies:** M0 fixtures and baseline checks. **Outcome:** Business logic can be tested independently of page setup.

- [x] **M1.1 — Extract artifact and domain types.** Define runtime validation in addition to TypeScript types. Reject unsupported versions, non-finite weights, duplicate IDs, malformed tuples, invalid references, and model dimension mismatches with actionable errors.
- [x] **M1.2 — Extract pure recommendation functions.** Move graph scoring, model scoring, blending, eligibility, and explanations out of `main.ts` behind typed interfaces. Preserve existing behavior with characterization fixtures before intentional changes.
- [x] **M1.3 — Extract provider and persistence adapters.** Separate network calls and local storage from UI event handlers. Make clocks, storage, transport, and random seeds injectable for tests.
- [x] **M1.4 — Implement versioned profile migrations.** Migrate the existing `wasiw.*.v1` state and named profiles without silently discarding selections, weights, exclusions, or unknown IDs. Back up before migration and recover from corrupt/quota-rejected storage.
- [x] **M1.5 — Define asynchronous cancellation and state rules.** Cancel obsolete imports/metadata work; ensure old responses cannot overwrite a newer profile. Distinguish loading, empty, unavailable, failed, stale, and demo states.
- [x] **M1.6 — Scope browser cleanup correctly.** Audit the origin-wide service-worker unregistration at startup. Remove obsolete behavior or restrict cleanup to this application's known scope; test that other apps on the same origin are unaffected.
- [x] **M1.7 — Audit UI rendering and dependencies.** Preserve escaping for external text, validate external URL schemes, render imported data as text, and test malicious strings. Remove obsolete renderer dependencies only after confirming they are unused.

**Exit gate:** Recommendation logic and provider failures are testable without booting the entire DOM. Existing profiles migrate safely and the tested baseline journeys still work.

### M2 — Make ingestion, metadata, and privacy dependable

**Priority:** P0/P1. **Dependencies:** M0 plus M1's artifact/provider contracts; unrelated tasks may proceed in parallel. **Outcome:** Repeatable data updates and transparent imports.

- [x] **M2.1 — Document data permissions and public/private boundaries.** Review current MAL/AniList/Jikan documentation and terms for each proposed use. Record allowed collection, caching, attribution, redistribution, training, and deletion handling; unresolved permissions block that provider-dependent activity, not fixture development.
- [x] **M2.2 — Audit existing public artifacts.** Check release assets and graph user-anime edges as well as raw ratings exports. Propose a minimal item-level public payload; do not delete or republish existing public releases without approval.
- [x] **M2.3 — Remove silent third-party proxy fallback.** Make an authorized supported provider route or explicit file import the default. Any optional third-party routing must disclose the destination and transferred data before use. Do not bypass access restrictions or rate controls.
- [ ] **M2.4 — Create provider-wide request scheduling.** Coordinate metadata, seasonal lists, imports, and crawler requests; honor runtime limits and `Retry-After`, use bounded exponential backoff with jitter, timeouts, cancellation, and retry budgets. Concurrency limits alone are not a complete rate policy.
- [ ] **M2.5 — Make collection resumable and idempotent.** Track pages/checkpoints and per-user fetch outcomes. Validate a complete snapshot before atomically replacing a user's list; never delete previous ratings because a page timed out or a response was empty/private. Test changed scores and removed entries.
- [ ] **M2.6 — Add schema migrations and provenance.** Record provider/source IDs, fetched time, genuine provider update times when available, import-run status, and normalized-field version. Do not label crawl timestamps as viewing/rating timestamps.
- [ ] **M2.7 — Publish a permitted catalog snapshot.** Include filter-critical genres, year, format, episode count/runtime, content classification, aliases, and available franchise relationships. Represent missing/unknown values explicitly. Separate permitted cached images or remote image references from the metadata contract.
- [ ] **M2.8 — Preserve imported history faithfully.** Retain status, progress, score scale, unscored seen items, and unmapped entries. Deduplicate by provider identity; show import-preview counts and merge/replace choices. Add file-size limits and safe parsers for each chosen export format.
- [x] **M2.9 — Quarantine uncleared provider-derived jobs.** Gate scheduled retraining and release/deploy paths that consume or publish provider-derived data until the source, use, and owner approval are recorded. Verify blocked and explicitly enabled paths without provider calls or production publication; preserve fixture PR checks. Do not delete or overwrite existing public assets as a side effect.

**Exit gate:** Re-importing the same list is idempotent; interrupted imports preserve the last complete data; filters can operate on the bundled catalog; network calls and public data exposure are documented and intentional.

### M3 — Correct graph mathematics and bound graph-building cost

**Priority:** P0. **Dependencies:** M0 fixtures and the minimum M1 pure-function extraction. **Outcome:** Deterministic, interpretable item relationships.

- [x] **M3.1 — Reproduce the aggregation defect in a unit test.** Use at least three observations per pair and shuffle users and ratings. Show that the existing recursive average changes with order. Preserve the fixture as a regression test.
- [x] **M3.2 — Introduce proper sufficient statistics.** Track pair sum and count for an order-independent mean as the compatibility correction. Carry true pair support into exports. Use stable accumulation/order and numeric tolerances so determinism claims are meaningful.
- [ ] **M3.3 — Decide and document edge semantics.** Distinguish pair preference, correlation, co-occurrence, and item similarity. Benchmark a corrected legacy statistic against support-shrunk item similarity; do not call the legacy mean of two centered scores a correlation.
- [ ] **M3.4 — Test positive/negative/neutral behavior.** Cover opposite tastes, universally disliked items, uniformly high/low raters, no variance, one observation, and no overlap. Define how negative evidence is used separately by visualization, candidate generation, and model regularization.
- [ ] **M3.5 — Replace first-encounter cap bias.** Evaluate minimum support, deterministic top-neighbor selection, and bounded candidate-generation strategies. Ensure an output-edge cap also has a separate input-work/memory budget; postprocessing top-K alone does not eliminate quadratic pair enumeration.
- [ ] **M3.6 — Make approximation explicit and reproducible.** For any sampling or per-user cap, use a recorded seed and documented selection policy, not lowest-ID slices. Report skipped data, approximation level, runtime, peak memory, and resulting coverage.
- [ ] **M3.7 — Version and migrate the graph format.** Export support, semantics, configuration, and dataset identity. Maintain a deliberate compatibility path or reject old incompatible versions clearly. Keep recommendation neighborhoods separate from graph visualization samples.

**Exit gate:** Reordering equivalent input leaves relationships and selected neighborhoods unchanged within a declared tolerance. Hand-computed fixtures match outputs; truncation is visible; no NaNs or broken references survive export.

### M4 — Align recommendation behavior with actual user intent

**Priority:** P0/P1. **Dependencies:** M1 domain contracts and M3 corrected graph; metadata-dependent cases use M2. **Outcome:** Watching, liking, excluding, and recommending are distinct operations.

- [ ] **M4.1 — Introduce explicit preference semantics.** Separate seen/unrated, liked, disliked, and confidence/importance. Map provider scores deliberately rather than treating every positive numeric score as positive preference. Migrate existing watch weights conservatively and explain changed behavior.
- [ ] **M4.2 — Centralize eligibility.** Apply seen status, user exclusions, include-only candidate sets, content preferences, catalog availability, and required filters consistently across graph, model, hybrid, and fallback paths. Exclusions win over inclusion. Define the meaning of the existing include control clearly.
- [ ] **M4.3 — Make missing model data nonfatal.** Fall back from hybrid/model to an eligible graph or baseline ranking, show the actual engine in use, and preserve settings. Test absent, corrupt, incompatible, and partially mapped model artifacts.
- [ ] **M4.4 — Implement a useful cold-start path.** With no preferences, offer transparent popularity/quality and genre-based exploration; with a few titles, use a supported content/neighbor baseline. Seasonal starter suggestions must not silently become the user's watched/liked history.
- [ ] **M4.5 — Evaluate user-vector construction.** Compare the current item-embedding average with a regularized fold-in fit using observed preferences and item biases. Test the method deployed in the browser, not just stored training-user vectors. Retain the simplest method that satisfies quality and latency goals.
- [ ] **M4.6 — Validate hybrid blending.** Compare existing min-max blending with a robust alternative such as rank fusion on validation data. Define missing-component behavior, equal scores, endpoint blend weights, negative scores, and deterministic tie-breaking. Do not present an uncalibrated score as a probability.
- [ ] **M4.7 — Produce faithful explanations.** Include actual source titles, distinct evidence counts, normalization and blend effects, biases/prior contribution, and uncertainty. Test that displayed numeric contributions reconcile with the score where claimed; qualitative reasons must be labelled as such.
- [ ] **M4.8 — Add diversity and franchise-aware eligibility.** Avoid filling the list with sequels or near-duplicate franchise entries; provide an option to allow them. Do not imply a title has no prerequisites when relationship data is unknown. Validate relevance/diversity tradeoffs rather than hardcoding genre quotas.

**Exit gate:** Low-rated or merely seen titles are not treated as favorites by default; seen/excluded titles never leak into any ranking mode; missing optional assets still yield an honest useful result; reasons match the implemented ranking.

### M5 — Rebuild trustworthy ML evaluation before tuning

**Priority:** P0 for evaluation integrity; P2 for expensive experiments. **Dependencies:** M3 graph logic, M4 serving contract, and raw-score/provenance contracts. **Outcome:** A defensible decision about which model to ship.

- [ ] **M5.1 — Split raw interactions first.** Define immutable train/validation/test IDs and deduplicate before splitting. Use true temporal splits only where reliable rating/viewing timestamps exist; otherwise use seeded per-user holdouts and disclose the limitation.
- [ ] **M5.2 — Fit all learned preprocessing on training data only.** Compute user baselines, graph edges/support, popularity statistics, and any learned feature transforms from the training partition. Isolate validation/test labels from these steps. Define an allowed metadata snapshot separately.
- [ ] **M5.3 — Add explicit leakage tests.** Modify held-out scores or hidden interactions and assert that training normalization, graph statistics, fitted parameters, and training hashes do not change. Fixed split IDs are essential to make this test meaningful.
- [ ] **M5.4 — Separate tuning from final reporting.** Use validation for Optuna, early stopping, blend parameters, regularization, thresholds, and candidate-selection choices. Keep test data untouched until a candidate configuration is selected; do not repeatedly optimize against the same test report.
- [ ] **M5.5 — Evaluate the production new-user path.** Test held-out users or simulated new-user histories with 1, 3, 5, and 10 supplied preferences. Run the same mapping, vector construction, candidate pool, filters, and exclusions used by the browser. Report warm-user performance separately.
- [ ] **M5.6 — Establish fair baselines and ablations.** Compare popularity/quality, supported item similarity, content-only, plain MF, graph-regularized MF, and hybrid. Keep splits and eligible candidate sets identical. Audit the current absolute-value graph regularizer and evaluate it against alternatives.
- [ ] **M5.7 — Report more than aggregate ranking accuracy.** Include Recall@10/20, NDCG@10/20, coverage, diversity, popularity bias, sparse-history and low-support slices, eligible-user counts, and latency. Include seed variability or bootstrap uncertainty and test metric edge cases with hand-computed examples.
- [ ] **M5.8 — Verify cross-language parity and artifact safety.** Export a tiny model and compare Python versus TypeScript candidate IDs, scores, top-K, and exclusions within a declared tolerance. Validate dimensions and hashes. Audit object-array/pickle loading; prefer safe numeric arrays plus JSON metadata for exchange.
- [ ] **M5.9 — Separate experimentation, release training, and promotion.** Keep LightGCN/content-feature experiments behind the same evaluation harness. After selection, record any final refit on train+validation or the permitted production snapshot separately from test metrics. Promote only a compatible approved release model; preserve the previous version for rollback.

**Exit gate:** The selected engine satisfies predeclared quality/latency/coverage gates against the best simple baseline. All deployed inference paths have relevant evaluation. Complexity is rejected when evidence does not justify it.

### M6 — Finish the user-facing discovery loop

**Priority:** P1. **Dependencies:** M2 catalog/import contracts and M4 stable ranking; visual iteration can begin earlier on fixtures. **Outcome:** A person can choose something to watch without understanding model internals.

- [ ] **M6.1 — Simplify first-use onboarding.** Offer manual favorites, explicit list import, or nonpersonalized browsing. Ask for a small useful set of inputs; move blend sliders and graph internals into advanced controls. Keep existing power-user functionality accessible.
- [ ] **M6.2 — Improve title resolution.** Search canonical titles and known aliases; show year/format/cover for disambiguation. Never silently map an ambiguous substring to the first match. Preserve unknown imports for later resolution.
- [ ] **M6.3 — Make result cards actionable.** Show title, relevant metadata, a brief reason, confidence/evidence language, and save/seen/not-interested actions. Separate personal ranking scores from provider community scores. Include meaningful image and metadata error states.
- [ ] **M6.4 — Fix filter completeness.** Apply catalog-based filters across the complete eligible candidate pool before taking the display top-K. Define unknown-metadata behavior, loading/retry, and no-match explanations. Test a valid item outside the old prefetched leading subset.
- [ ] **M6.5 — Add a local watchlist and feedback loop.** Support plan-to-watch, watching, completed, on-hold, and dropped states plus explicit ratings. Keep state independent of dataset version and model changes; changing status is not implicit permission to train on a user's history.
- [ ] **M6.6 — Complete profile backup and recovery.** Provide versioned export/import, merge preview, reset, and recovery from invalid local state. Profiles should preserve unmapped items and migrate once without repeated destructive changes. Large state may move to IndexedDB when justified.
- [ ] **M6.7 — Verify mobile and accessible interaction.** Test keyboard-only flows, labels, focus restoration, announcements, touch targets, zoom, reduced motion, normal/high-contrast themes, and narrow layouts. Avoid an enormous sequence of tabbable graph nodes; provide an equivalent accessible list.
- [ ] **M6.8 — Add end-to-end journey tests.** Cover manual onboarding to saved result and reload, imported history to filters, profile switching during requests, missing provider/model, and corrupt-state recovery. Use mocked provider responses and semantic assertions, not screenshot-only approval.

**Exit gate:** A new tester can get useful recommendations and save a shortlist without assistance. Core workflows function on desktop and mobile without live provider availability.

### M7 — Reduce data cost and make the network explorer useful

**Priority:** P1. **Dependencies:** M1 boundaries and M3 artifact semantics. **Outcome:** Recommendations stay responsive without loading or rendering the entire research graph.

- [ ] **M7.1 — Measure before optimizing.** Record transferred/compressed/uncompressed bytes, parse/index time, ranking latency, main-thread stalls, graph-render time, and memory on declared desktop and mobile profiles. Establish both cold and warm measurements.
- [ ] **M7.2 — Separate initial recommendation assets.** Load catalog and compact item neighborhoods first; lazy-load model and graph-explorer data when needed. Do not send user-rating histories or full bipartite graphs merely to rank anime.
- [ ] **M7.3 — Make graph samples honest.** Label sample/truncation limits and graph/model versions. A sampled visualization must not imply that absent edges or missing users prove no relationship. Use item-only explanation graphs by default unless public user-level data is explicitly justified.
- [ ] **M7.4 — Implement bounded local exploration.** Focus on one anime and its strongest relevant neighbors with an explicit node/edge budget, keyboard/list alternative, focus preservation, selection, zoom/pan as useful, and reset. Keep the current SVG renderer unless profiling demonstrates a need to replace it.
- [ ] **M7.5 — Move measured heavy work off the main thread.** Use a worker for large parse/index/rank/layout operations only where measurements warrant it. Bound queues, cancel stale work, and transfer buffers rather than duplicating full objects unnecessarily.
- [ ] **M7.6 — Make compressed delivery robust.** Test gzip-only assets, plain JSON, unsupported decompression, browser-decoded responses, malformed gzip, HTTP errors, stale caches, and oversized/decompression-bomb payloads. Provide a compatible fallback or explicit supported-browser requirement.
- [ ] **M7.7 — Set and enforce initial performance budgets.** Proposed starting budgets are a usable fixture-backed first view within 3 seconds on a declared throttled profile and warm recommendation updates under 200 ms at p95 on a declared representative dataset. Measure real conditions, refine with recorded rationale, and document any missed budget rather than asserting achievement.

**Exit gate:** Recommendations do not wait for full network rendering. Performance measurements are reproducible; stale work can be cancelled; payload and rendering growth are bounded.

### M8 — Make data/model delivery and deployment reproducible

**Priority:** P0/P1. **Dependencies:** Validated artifact contracts and passing core tests. **Outcome:** A deployment identifies exactly which compatible data and model it serves.

- [ ] **M8.1 — Introduce an immutable release manifest.** Pin dataset/catalog/neighborhood/model hashes and schema versions. Reject incompatible item maps and stale cross-version combinations. Preserve a named last-known-good bundle.
- [ ] **M8.2 — Validate downloads and atomic installation.** Download to temporary locations, enforce compressed/uncompressed size limits, verify hashes and schema/counts, and only then replace the active local bundle. Test partial/corrupt/missing files and failure recovery.
- [ ] **M8.3 — Make release-data publication auditable.** Include provenance, allowed redistribution, changes, compatibility, quality checks, and an explicit asset allowlist. Prevent accidental SQLite files, salts, user histories, or training-user factors from entering public artifacts.
- [ ] **M8.4 — Connect model promotion deliberately.** Keep retraining artifacts separate from production release assets. Promotion must include evaluation, mapping compatibility, required approval, and a deployment trigger; a successful weekly upload is not equivalent to a deployed model update.
- [ ] **M8.5 — Expand deployment verification.** Run offline tests and build checks before publication, then test the real GitHub Pages base path, direct navigation, data loading, gzip behavior, security headers where controllable, and optional-model fallback. Use least-privilege workflow permissions and validate workflow inputs.
- [ ] **M8.6 — Add privacy-conscious diagnostics.** Surface app/data/model versions and actionable error codes; redact usernames and raw histories. Prefer local diagnostics for the first release. External telemetry requires a documented purpose, consent/controls as applicable, and owner approval.
- [ ] **M8.7 — Practice rollback and recovery.** Restore a prior compatible release in a test/staging context, verify old profile migrations still work, and document data refresh, promotion, recovery, and the distinction between source, data, and model versions.

**Exit gate:** A release is a reproducible versioned bundle, not whatever `data-latest` happens to contain during a build. Invalid updates fail safely and the prior working release can be restored.

### M9 — Stabilize desktop as a separate deliverable

**Priority:** P2 unless the owner explicitly changes release priority. **Dependencies:** Stable public artifact contracts. **Outcome:** Desktop behavior is honest and maintainable, without blocking web delivery.

- [ ] **M9.1 — Record the desktop product decision.** Choose a maintained graph companion versus a full recommendation client. Default to the smaller maintained companion; do not promise web parity before approving its scope and duplication cost.
- [ ] **M9.2 — Support deliberate data selection.** Read the approved compact/manifest format, offer explicit file selection, show dataset version/counts, and distinguish real data, no data, error, and demo mode. Do not silently fall back to sample data after a failed real-data load.
- [ ] **M9.3 — Eliminate divergent graph rebuilding.** Prefer consuming the shared precomputed graph semantics; if Rust must rebuild, use the same specification and cross-language golden tests, including the fixed aggregation behavior.
- [ ] **M9.4 — Keep loading/rendering bounded.** Move expensive file work away from reactive rendering, cap visible data, preserve responsiveness, and test malformed files and Unicode. Avoid rebuilding the whole dataset on each UI update.
- [ ] **M9.5 — Add Rust verification and packaging checks.** Format, lint, test, and build on supported platforms. Pin the toolchain/lockfile strategy. Verify required runtime dependencies and usable distribution packaging rather than assuming an EXE alone is sufficient.
- [ ] **M9.6 — Test outside the repository.** Launch a packaged application in a clean supported environment with no checkout or relative `data/` directory. Verify demo labels, file loading, missing assets, and documented limitations before approving release.

**Exit gate:** The supported desktop scope runs outside a development checkout, uses compatible data, and never misrepresents a sample as the user's real dataset. This gate is separate from the web release gate.

### M10 — Run a small beta and close the loop

**Priority:** P1. **Dependencies:** M0–M8 web release gates; M9 is independent. **Outcome:** Release decisions include observed human usefulness.

- [ ] **M10.1 — Prepare a repeatable beta script.** Use representative sparse-history, broad-history, niche-taste, and new-to-anime cases. Ask testers to import/select, inspect, filter, save, and return without coaching. Obtain consent before collecting feedback data.
- [ ] **M10.2 — Define success criteria before sessions.** Proposed targets: testers complete the core journey unassisted, understand why recommendations appear, identify several plausible unseen choices, and encounter no seen/excluded items. Use a practical starting target such as 3 plausible choices among the top 10, not a claimed measured outcome.
- [ ] **M10.3 — Collect diagnostic feedback without default surveillance.** Distinguish known/already-seen, disliked, confusing, irrelevant, prerequisite-missing, and metadata-wrong reports. Record local reproduction details and aggregate observations without exporting raw personal lists by default.
- [ ] **M10.4 — Fix blockers and re-test.** Prioritize crashes, data loss, incorrect imports, exclusion failures, misleading explanations, and unusable mobile flows over new features. Link each beta defect to a regression test and a plan task.
- [ ] **M10.5 — Release and record the next evidence-based backlog.** Publish accurate setup, privacy/data-source, model-limitations, supported-platform, and recovery documentation. Record fulfilled gates, known limitations, release versions, and the next highest-impact task; do not describe the entire long-term product as finished.

**Exit gate:** Both technical gates and observed core-user usefulness are recorded. Remaining limitations are explicit, and optional experiments are not confused with release blockers.

## 6. Execution order and parallel work

Recommended critical path:

```text
M0 baseline/fixtures
  -> minimal M1 extraction/contracts
  -> M3 graph correction + M2 data/import/catalog work
  -> M4 serving semantics + M5 trustworthy evaluation
  -> M6 complete discovery loop
  -> M7 measured performance + M8 release reliability
  -> M10 beta and release

M9 desktop follows stable artifact contracts on its own track.
```

This is a dependency guide, not an instruction to finish every item in one milestone before any work elsewhere. In particular, privacy/proxy and destructive-import risks may move ahead of lower-risk refactors, and metadata completeness can be fixed independently of ML experiments. Break conflicts into small PRs; avoid concurrent edits to the large `main.ts` until the relevant modules are extracted.

### First implementation slice

Start with M0.1–M0.6 and the minimum extraction required for M3.1. Produce a synthetic fixture and a failing aggregation-order regression test. Implement the sum/count compatibility fix in M3.2, document the semantic limitation of that statistic, and verify shuffled-input equality. Do not immediately regenerate or publish production artifacts: format compatibility and evaluation must be addressed first.

In the next slice, remove silent external fallback routing, define viewing/preference semantics, and establish split-first evaluation. Only after those foundations should a larger model or major UI redesign compete for priority.

### Existing baseline commands to verify

These commands are present in the inspected repository. They are not a claim of successful execution here. Do not use data-publishing, network-expansion, or scheduled-production actions as ordinary test commands.

```bash
npm ci
npm run typecheck --workspace pipeline
npm run build:web

# Requires suitable local/release data; not the offline fixture path to be added.
npm run data:fetch:release
npm run sync:web
npm run dev:web

# Python environment setup; establish a lock/version policy before CI expansion.
python -m pip install -r ml/requirements.txt

# Requires a compatible Rust/Dioxus desktop environment.
cargo check --manifest-path desktop/Cargo.toml
```

Proposed commands to add, not assumed to exist: root `typecheck`, `test`, `test:e2e`, `data:fixture`, and `dev:demo`, plus a documented small Python evaluation-smoke command. Keep invocation names discoverable in the README and package manifests.

## 7. Definition of done and release gates

A task is done only when its specified behavior works, relevant automated tests pass, pertinent failure paths are tested, data/profile compatibility is handled, documentation is updated, no sensitive data enters outputs, and verification evidence is recorded next to the completed task ID. Large algorithm changes additionally need benchmark/evaluation evidence; UI changes need the relevant browser/manual checks.

Proposed minimum web release gates:

- Clean fixture-based installation, typechecking, unit/integration tests, browser journey tests, and production build pass in CI.
- Graph generation is order-independent under the declared tolerance and release artifacts pass schema/reference/hash validation.
- Held-out interactions do not influence training preprocessing or graph construction; model claims refer to the actual deployed inference path.
- Seen/excluded items never appear across tested ranking modes, missing optional assets degrade honestly, and metadata filters search the intended eligible catalog.
- Profile migration and export/import preserve user state; provider failure and partial imports cannot erase prior valid history.
- Performance budgets are measured on declared devices/profiles, public data exposure is reviewed, and no silent proxy routing remains.
- Versioned deployment and rollback are tested, and beta findings support the core discovery loop.

No arbitrary overall test-coverage percentage substitutes for these behavioral gates. Quality targets must be declared before tuning or beta evaluation; later changes require documented evidence and cannot be used to conceal regression.

## 8. Deferred opportunities

After a useful release, reconsider richer mood discovery, better watch-order relationships, a binge-time planner, opt-in cross-device sync, group recommendations, improved semantic/content features, and optional advanced graph experiments. Each should begin with a user problem, measurable success criteria, a data/privacy plan, and the smallest useful implementation. Social feeds, monetization, and chatbot features remain separate proposals rather than automatic roadmap commitments.

## 9. Progress and change log

| Area | Status | Evidence / next action |
|---|---|---|
| Source review | Complete for initial starting state | Initial `master` matched the reviewed commit; draft PRs now carry subsequent work. See `docs/PROGRESS.md` for current branches and checks. |
| M0 | Complete | M0.1–M0.7 and the exit gate passed: a fresh PR checkout generated and served the fixture site, then completed fast checks in [run 36088138480](https://github.com/OptimumAF/WhatAnimeShouldIWatch/actions/runs/36088138480). |
| M3 | In progress | M3.1–M3.2 verified on synthetic inputs; M3.3–M3.7 and milestone exit gate remain open. |
| M2 | In progress | M2.1–M2.3 and M2.9 verified; M2.9 fixture PR [CI passed](https://github.com/OptimumAF/WhatAnimeShouldIWatch/actions/runs/36091467488). [Decision 0001](decisions/0001-provider-data-permissions.md) records provider holds; the [M2.2 audit](audits/public-artifacts-2026-09-24.md) confirms public per-user ratings and graph edges. [Decision 0002](decisions/0002-provider-workflow-gates.md) gates future jobs once merged. M2.4–M2.8 and the exit gate remain open. |
| M1 | Complete: exit gate passed | M1.1–M1.7 verified. Pure recommendation/provider tests run without the DOM; version 1–3 profile migration, corrupt/quota recovery, and baseline UI journeys pass. After M1.7, a clean install, fixture checks, typecheck, 4 pipeline and 28 web unit tests, 2 ML and 5 workflow tests, web build, and 28 mocked/demo browser tests passed. `npm ci` reports zero advisories; draft [PR #11 fixture CI](https://github.com/OptimumAF/WhatAnimeShouldIWatch/actions/runs/36098562980) passed on the implementation commit. |
| M4–M8, M10 | Pending | Follow dependencies and safety priorities; no claim of completion. |
| M9 desktop | Pending, separate track | Decide scope after artifact contracts stabilize |

Initial plan entry: 2026-09-24 — Created from source review. Defaults to preserving the existing stack, a web-first dependable recommendation loop, and explicitly evidence-gated graph/model work. Future agents must append material changes and update completion state as they execute.

| Date | Task ID | Previous scope/status | New scope/status | Evidence/reason | Dependency impact |
|---|---|---|---|---|---|
| 2026-09-24 | M0.4 | Proposed fixture-backed path | Explicit Vite demo mode with generated local graph, catalog, and tiny synthetic model | Existing web loaders required release files and live Jikan metadata; browser smoke now runs without production data or provider requests | Enables M0.5, M0.6, and graph work on safe fixtures |
| 2026-09-24 | M3.2 | Pair weight stored as recursive average, no support export | Sum/count mean with true support on legacy edges and optional fourth compact tuple value | Three observations produced 3.5 under the old formula; regression and CLI export checks now pass | M3.3 semantics, M3.5 cap policy, and M3.7 format versioning remain required |
| 2026-09-24 | M0.6 | Pending | Complete | PR [run 36088138480](https://github.com/OptimumAF/WhatAnimeShouldIWatch/actions/runs/36088138480) passed install, fixture checks, TypeScript/Python tests, web build, and browser smoke without provider requests | M0 exit gate passed; M1 and independent safety work are dependency-ready |
| 2026-09-24 | M2.3 | Planned after M1 provider contracts | Next isolated safety slice, still pending | Source inspection found the existing MAL username path can silently route through `r.jina.ai`; mocked-provider tests can verify removal without a broader adapter refactor | M1.3 remains open; no provider collection or external routing is authorized |
| 2026-09-24 | M2.3 | Silent Jina fallback; no file picker | Complete: direct MAL failures stop with a clear message; local `.txt` import is the recommended first route and requires review before applying | Browser regression first observed the proxy request under a mocked MAL 403; afterward mocked 403, transport failure, direct success, and local-file cases passed. AniList's [API terms](https://docs.anilist.co/guide/terms-of-use) require a separate product-use review, so the local route avoids assuming that review is complete | M2.1 provider permissions and M2.8 complete export handling remain open; no new provider access or data publication |
| 2026-09-24 | M0.6 | PR CI targeted only `master` | PR CI also accepts `codex/**` base branches | M2.3 is a separate reviewable PR stacked on the passing M0 branch; the same fixture checks should run on that PR | No new deploy or data-publishing trigger; M0 exit gate remains passed |
| 2026-09-24 | M2.1 | Provider permissions and public/private boundary unrecorded | Complete documentation with per-source, per-use holds and an explicit local/restricted/public boundary in [decision 0001](decisions/0001-provider-data-permissions.md) | Current MAL terms restrict site extraction and aggregation; AniList's competing-service status is unresolved; Jikan defers to MAL and its linked terms page returned 404. Existing public release asset names and scheduled retraining make follow-up urgent | M2.2 actual asset audit next; M8.4 workflow/promotion controls move ahead of further provider-dependent work. No provider permissions were assumed or granted |
| 2026-09-24 | M2.2 | Public artifact contents not inspected | Complete read-only [release and Pages audit](audits/public-artifacts-2026-09-24.md) with an item-level replacement allowlist proposal | Release and deployed graph both contain the same 200 pseudonymous IDs and 48,558 user-anime relationships; the release separately publishes 48,558 raw/normalized rating rows. Checksums and cross-file pairs matched; no rows were logged or copied into the repo | Existing publication requires owner decision; M2.9 immediate workflow quarantine is dependency-ready before any further provider-derived jobs |
| 2026-09-24 | M2.9 | No explicit containment task | Added pending task for approval-gated provider-data jobs | M2.1 found unclear source rights; M2.2 confirmed current release and Pages graph expose user-linked data, while scheduled retraining consumes release ratings | Prioritize M2.9 before provider-dependent collection/training/deployment; M8.4 still owns full model-promotion design |
| 2026-09-24 | M2.9 | Scheduled retrain and release/Pages jobs ran without source/use approval gates | Complete implementation: separate training, publication, and deployment repository flags and approval references; a committed use-specific record is validated before data access | Five local gate tests covered missing, wrong-scope, incomplete, and synthetic approved records; `actionlint` passed on four workflows; the full fixture/typecheck/test/build/browser suite passed. No repository approval variables are set. See [decision 0002](decisions/0002-provider-workflow-gates.md) and `docs/PROGRESS.md` | Gate changes take effect only after review and merge. Existing public assets remain; no provider right or owner approval is inferred. M8.4 remains open |
| 2026-09-24 | M1.6 | Startup unregisters every service worker whose scope starts with the origin | Complete: remove the cleanup because this repository never registers a service worker | New browser test observed one unregister call for `/other-app/` before the change and zero after; all six browser tests, typecheck, and web build passed | Prevents this app from disturbing another app on the same origin; no new registration or caching behavior introduced. M1 exit gate remains open |
| 2026-09-24 | M1.1 | Artifact shapes trusted after a format-string check; malformed model rows were silently skipped | Complete: extracted browser artifact/domain types and validated compact/legacy graph, demo catalog, and compact/legacy model JSON before use | Synthetic unit cases reject unsupported versions, non-finite weights, duplicate IDs/relationships, malformed tuples, broken references/counts, and model dimension/array mismatches. Browser cases show named failures and preserve valid legacy loading; the full fast suite passed | A present invalid compact artifact now fails instead of silently using legacy data. Missing compact files still use legacy fallback. M1.2 can extract pure scoring against these contracts; M1 exit gate remains open |
| 2026-09-24 | M1.2 | Graph/model/hybrid scoring, candidate and metadata eligibility, and contributor explanations depended on `main.ts` UI state | Complete: moved pure indexes, ranking, filters, and explanation text to typed `web/src/recommendations.ts`; the UI supplies selection, metadata, and mode state | Exact synthetic browser rankings, scores, and explanation text passed before and after extraction. Eleven web unit tests and 13 browser tests pass, including negative-edge order characterization and escaped invented markup | Existing selection-order negative evidence and min-max blend endpoint behavior are documented, not changed. M3.4 and M4.6 own semantic fixes; M1.3 provider/persistence adapters are next. M1 exit gate remains open |
| 2026-09-24 | M1.3 | Network fetches, localStorage parsing/writes, timers, and random calls lived in `main.ts` | Complete: artifact, provider, and persistence adapters receive typed runtime ports for transport, storage, clock/scheduling, and random source; `main.ts` supplies UI state and uses adapter methods | Baseline synthetic profile journey passed before and after extraction; injected-port unit cases cover direct provider success/failure, seeded retry timing, denied storage, namespaced legacy keys, demo artifacts, and missing-compact fallback. Full local fixture/typecheck/test/build/browser suite passed | M1.4 owns versioned migration, backup, and quota recovery; M1.5 owns stale-response cancellation; M2.4 owns provider-wide rate and retry policy. No existing permission hold was cleared; M1 exit gate remains open |
| 2026-09-24 | M1.4 | Version 1 browser keys held state versions 1–3 and an unversioned profile array; loading and saving silently filtered catalog-missing IDs, and failed writes still reported success | Complete: version 4 state and profile envelope with raw pre-migration backups, previous-current backups, corrupt-copy preservation, and visible storage warnings; old keys stay untouched and unknown selections/overrides round-trip | Six direct synthetic migration/recovery/quota tests and four profile browser journeys passed; full fixture/typecheck/test/build/browser suite passed (see `docs/PROGRESS.md`) | M1.5 cancellation and M1.7 rendering audit are next; M6.6 retains full export/import/reset/recovery UI. M1 exit gate and provider permission holds remain open |
| 2026-09-24 | M1.5 | A delayed username import could apply after profile load; superseded metadata could still enter the cache; local file reads and seasonal refresh had no stale-result guard | Complete: abort signals and generation checks for provider imports, retries, metadata, local files, and seasonal results, with distinct visible async states | Mocked delayed-import, provider-backoff, metadata, file, and status tests passed; full fixture/typecheck/test/build/browser suite passed (see `docs/PROGRESS.md`) | M1.7 rendering and URL audit remains before M1 exit gate. M2.4 still owns provider-wide rate scheduling, timeouts, `Retry-After`, and retry budgets |
| 2026-09-24 | M1.7 | External catalog/provider image URLs were HTML-escaped but accepted without scheme checks; Sigma and ForceAtlas2 were direct unused dependencies; npm audit found five advisories | Complete: allow parsed HTTP(S) image URLs only, omit unsafe covers, retain literal rendering of untrusted text, remove confirmed unused renderers, and refresh compatible lockfile dependencies to clear the advisories | Two new browser cases first failed on unsafe image sources; after the fix, malicious catalog/provider text, graph labels, imports, and profiles stayed literal. A clean `npm ci` reported zero advisories, and the full local fixture/typecheck/unit/Python/build/28-browser suite passed | M1 exit gate now passes locally: pure recommendation and provider failure tests, safe profile migration, and baseline journeys remain green. M2.4 is the next dependency-ready slice; provider permission holds remain |

## 10. Evidence references

Repository source references below use the reviewed commit. Re-read the current checkout before implementation; line numbers in source may change. These sources support the baseline observations, not a claim that the application was executed.

- Repository identity/history: `https://github.com/OptimumAF/WhatAnimeShouldIWatch/commit/dea9a40e87d9864a3ae991c759a193ee43addebc`
- Project description and workflows: `README.md`, `package.json`, `web/package.json`, `pipeline/package.json`.
- Graph semantics and cap: `pipeline/src/build-graph.ts`; SQLite normalization/schema: `pipeline/src/db.ts`.
- Browser imports: `web/src/main.ts`, reviewed approximately lines 2200–2500.
- Browser ranking/filter behavior: `web/src/main.ts`, reviewed approximately lines 2800–3100 and 3500–3860; startup/types/UI declaration also inspected.
- Compression and SVG graph handling: `web/src/main.ts`, reviewed approximately lines 4400–4850; `pipeline/src/sync-web.ts`.
- ML implementation: `ml/train_graph_mf.py`, reviewed lines 1–270 and 340–end; `ml/README.md`.
- Desktop loading/rendering: `desktop/src/main.rs`, reviewed lines 1–200.
- Automation: `.github/workflows/ml-retrain.yml`, `.github/workflows/deploy-web.yml`; additional workflow names established from the repository tree.

Canonical source URL pattern: `https://github.com/OptimumAF/WhatAnimeShouldIWatch/blob/dea9a40e87d9864a3ae991c759a193ee43addebc/<path>`.

External provider documentation consulted through search on September 24, 2026:

- Jikan official documentation: `https://docs.api.jikan.moe/`. Search-visible documentation reports per-second and per-minute limits; implement runtime-aware scheduling rather than treating this plan as a permanent limit specification.
- AniList rate-limit documentation: `https://docs.anilist.co/guide/rate-limiting`.
- AniList terms: `https://docs.anilist.co/guide/terms-of-use`.
- Jikan service scope: `https://jikan.moe/`.

Some direct provider documentation requests were unavailable during planning. No definitive legal permission determination is claimed. Reverification is a task before provider-dependent collection, redistribution, or expansion.
