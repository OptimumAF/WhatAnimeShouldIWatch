# Execution progress and handoffs

## Session 2026-09-24 (America/Los_Angeles)

**Branch / starting commit:** Local workspace was an empty, unborn `master` with no remote or files. Fetched `OptimumAF/WhatAnimeShouldIWatch` at `dea9a40e87d9864a3ae991c759a193ee43addebc` (the plan's reviewed SHA), confirmed it was current remote `master`, and created `codex/m0-fixtures-graph`. The working tree was clean immediately after checkout. No repository or ancestor `AGENTS.md`, `docs/DEVELOPMENT_PLAN.md`, open GitHub issues, or open PRs existed. Inspected recent commits, manifests, release-data fetch/publish scripts, `sync-web.ts`, and workflow purposes; no newer implementation to preserve was found. The exact supplied `_DEVELOPMENT/_PLAN.md` path was absent; the matching `WhatAnimeShouldIWatch_DEVELOPMENT_PLAN.md` in Downloads was saved as the canonical plan.

**Tasks completed:** M0.1–M0.7, M3.1, M3.2. M0 exit gate passed: a fresh PR checkout generated and served the fixture-backed site and completed the documented fast checks without production data or provider access. M3 remains open.

**Commands and actual results:**

| Command | Result |
|---|---|
| `node --version`; `npm --version`; `python --version`; `rustc --version`; `cargo --version` | v23.10.0; 11.2.0; 3.9.12; 1.93.1; 1.93.1 on Windows. CI uses Node 22 and Python 3.12. |
| `npm ci` | Passed: 68 packages at baseline; after adding Playwright, passed with 71 packages. One intermediate rerun hit a Windows Rollup file lock while Vite was open; closing the temporary server resolved it. npm reported 5 dependency audit findings (1 low, 4 high); no automatic dependency upgrades applied. |
| `npm run typecheck --workspace pipeline`; `npm run build:web`; `python -m compileall -q ml` | Passed before changes. Vite 7.3.1 built 9 modules. |
| `cargo check --manifest-path desktop/Cargo.toml` | Passed on Windows with one existing dead-code warning for `Node.id` and `Node.label`; generated `desktop/Cargo.lock` for reproducibility. Desktop UI was not launched. |
| `node --import tsx --test pipeline/test/pair-aggregation.test.ts` | **Failed as intended before fix:** recursive pair average yielded 3.5 for three observations with expected mean 3. |
| `npm run data:fixture`; `npm run data:fixture:check` | Passed: deterministic local graph/catalog/model; 7 synthetic users, 8 invented anime, 11 pairs, 1 duplicate rating and 1 unknown ID handled. |
| `npm run typecheck`; `npm test`; `npm run test:python`; `npm run build:web` | Passed after clean install: both TypeScript projects; 4 pipeline unit/CLI tests; 2 Python tests; Vite production build. |
| `npx playwright install chromium`; `npm run test:e2e` | Chromium installed. Browser smoke passed: synthetic banner, manual selection, graph/model/hybrid results, local metadata filter, SVG network, and zero production-data/provider requests. Initial browser failure was a missing Chromium install; a later overly broad assertion matched the explanation's source title and was corrected. |
| `gh run watch 36088138480 --exit-status` | Passed: PR `verify` job completed in 48 seconds on Ubuntu. `npm ci`, fixture generation/check, TypeScript typechecks, 4 pipeline tests, 2 Python tests, web build, and Playwright browser smoke all passed. [Run evidence](https://github.com/OptimumAF/WhatAnimeShouldIWatch/actions/runs/36088138480). |

**Baseline journey inventory (M0.7):** Expected behavior is from source inspection; only the marked demo paths were exercised in a browser.

| Journey | Current expected behavior | Observed / known limitation |
|---|---|---|
| Manual selection | Search a graph title, add it to watched, adjust weight or remove it; state persists locally. | Demo smoke added invented `Copper Comet` and received unseen graph results. Watched and liked remain conflated (M4.1). |
| Graph / model / hybrid | Graph uses anime edges; model loads an optional artifact; hybrid blends both. Missing model shows an unavailable message. | All three modes produced demo results. The demo model is synthetic and has no quality claim. Production compatibility untested. |
| Imports | Bulk input accepts IDs/titles and optional scores. Username import queries AniList or MAL for rated mapped entries. | Demo disables username import; neither live import was run. The normal MAL route can fall back to `r.jina.ai` without prior disclosure (M2.3); statuses/unmapped entries are not retained (M2.8). No browser account history entered fixtures. |
| Profiles | Local storage saves selections, weights, mode, blend and candidate overrides in current or named profiles, then restores them. | Source inspected; full save/reload/migration and corrupt-state paths are untested (M1.4/M6.6). |
| Filters | Genre, year range and minimum community score filter candidates using metadata. | Demo smoke filtered a local catalog result. Limited normal-path metadata prefetch can hide eligible candidates (M6.4). |
| Network inspection | Open SVG explorer, toggle user/anime edges, search/select a node, and view weighted connections. | Demo smoke rendered SVG; manual snapshot showed 5 nodes and 4 edges at the default threshold. Keyboard and truncation claims need later tests (M7.3–M7.4). |

**Files changed:** `AGENTS.md`, `.gitignore`, `.github/workflows/ci.yml`, `README.md`, `docs/DEVELOPMENT_PLAN.md`, `docs/PROGRESS.md`, `desktop/Cargo.lock`, `fixtures/synthetic-input.json`, root and workspace package files, `pipeline/src/build-graph.ts`, `pipeline/src/core/pair-aggregation.ts`, `pipeline/src/generate-fixture.ts`, `pipeline/src/sync-web.ts`, `pipeline/src/types.ts`, `pipeline/test/`, `ml/tests/`, `web/.env.demo`, `web/playwright.config.ts`, `web/src/main.ts`, `web/src/style.css`, and `web/tests/demo.spec.ts`.

**Commit / PR:** Implementation commit `7b57e60ef6cef369960ff26c9506d26ca50dd7ac`; draft PR [#1](https://github.com/OptimumAF/WhatAnimeShouldIWatch/pull/1). PR CI [run 36088138480](https://github.com/OptimumAF/WhatAnimeShouldIWatch/actions/runs/36088138480) passed.

**Checks not run / blockers:** No production dataset fetch, crawl, sync, model training, release publish, deployment, live username import, or desktop UI launch was performed. Full graph cap determinism, input-work budget, semantic validity, and format versioning remain M3.3–M3.7; no production artifacts were regenerated. Live import/proxy behavior and profile migration have only source-review evidence.

**Plan revisions and reasons:** Added explicit Vite demo mode with ignored generated files, separate local storage keys, and a local catalog/model so ordinary work needs no release download or contamination of normal dev preferences. Added optional fourth compact anime-pair tuple value for support while preserving three-value readers; formal format/semantic migration remains M3.7. First-encounter edge and first-N per-user caps remain M3.5–M3.6. M0.6 was checked only after its PR run passed.

**Single next task:** M2.3 — remove the silent MAL proxy fallback with a mocked-provider regression test and an explicit user-facing import error. This privacy fix is independent of broader M1 extraction and was elevated ahead of lower-risk refactors as allowed by the plan.

## Session 2026-09-24 — M2.3

**Branch / starting commit:** `codex/mal-import-proxy` from `9218864df1b5d89b7d76f6e716960894789e0704`; clean worktree. Draft PR #1 for the base slice is open with passing checks.

**Completed:** M2.3. A mocked MAL 403 test failed before the fix and recorded the prior `r.jina.ai` request. The fallback and its text parser were removed. Direct MAL failure now gives an actionable error without changing the current watched list. The recommended first import route is a local `.txt` file (128 KiB maximum) or pasted text; selecting a file fills the existing bulk-import area for review before applying it. The UI identifies where a username goes when a provider is selected. Existing direct MAL success remains covered.

**Verification commands and actual results:** `npx playwright test tests/mal-import.spec.ts` — 4 passed; `npm run data:fixture:check` — verified 7 synthetic users, 8 anime, 11 pairs; `npm run typecheck` — both workspaces passed; `npm test` — 4 pipeline tests passed; `npm run test:python` — 2 passed; `npm run build:web` — passed; `npm run test:e2e` — 5 passed (demo smoke plus 4 mocked-provider/import tests); `git diff --check` — passed; `rg -n 'r\.jina|fetchTextWithRetries|parseJinaMarkdownJson' web/src` — no matches. The browser tests cover direct success, HTTP 403, transport failure, local file size rejection, review before apply, and no proxy/provider request during file import.

**Files changed:** `.github/workflows/ci.yml`, `README.md`, `docs/DEVELOPMENT_PLAN.md`, `docs/PROGRESS.md`, `web/playwright.config.ts`, `web/src/main.ts`, `web/src/style.css`, `web/tests/mal-import.spec.ts`.

**Checks not run / blockers:** No live MAL or AniList username was used; browser provider responses were mocked. No production dataset, crawl, release, deploy, or public artifact change. M2.1 still needs a provider-by-provider permissions review, especially AniList's restriction on competing list/tracker services; M2.8 still needs supported export formats, complete status/progress/score retention, unmapped-entry handling, and preview/merge semantics. M2 milestone exit gate is not met.

**Plan revisions and reasons:** M2.3 is checked because the recommended import route is now an explicit local file and the silent proxy path is removed and tested. The file parser deliberately reuses the existing one-line bulk format; it is not claimed to parse Crunchyroll or MAL exports. PR CI now includes `codex/**` base branches so this isolated slice can be checked while stacked on the M0 branch. The AniList route remains existing optional functionality, with product-use permission pending M2.1.

**Single next task:** M2.1 — document current provider permissions and the public/private boundary before expanding any provider-dependent activity.
