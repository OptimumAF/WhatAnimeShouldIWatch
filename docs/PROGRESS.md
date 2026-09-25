# Execution progress and handoffs

## Session 2026-09-24 (America/Los_Angeles)

**Branch / starting commit:** Local workspace was an empty, unborn `master` with no remote or files. Fetched `OptimumAF/WhatAnimeShouldIWatch` at `dea9a40e87d9864a3ae991c759a193ee43addebc` (the plan's reviewed SHA), confirmed it was current remote `master`, and created `codex/m0-fixtures-graph`. The working tree was clean immediately after checkout. No repository or ancestor `AGENTS.md`, `docs/DEVELOPMENT_PLAN.md`, open GitHub issues, or open PRs existed. Inspected recent commits, manifests, release-data fetch/publish scripts, `sync-web.ts`, and workflow purposes; no newer implementation to preserve was found. The exact supplied `_DEVELOPMENT/_PLAN.md` path was absent; the matching `WhatAnimeShouldIWatch_DEVELOPMENT_PLAN.md` in Downloads was saved as the canonical plan.

**Tasks completed:** M0.1, M0.2, M0.3, M0.4, M0.5, M0.7, M3.1, M3.2. M0.6 has a local PR workflow, but remains unchecked until an actual pull-request run passes. M0 as a milestone remains open.

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

**Commit / PR:** Pending final review and commit. Record identifiers after creation.

**Checks not run / blockers:** The new GitHub pull-request workflow has not run yet, so M0.6 remains open. No production dataset fetch, crawl, sync, model training, release publish, deployment, live username import, or desktop UI launch was performed. Full graph cap determinism, input-work budget, semantic validity, and format versioning remain M3.3–M3.7; no production artifacts were regenerated.

**Plan revisions and reasons:** Added explicit Vite demo mode with ignored generated files, separate local storage keys, and a local catalog/model so ordinary work needs no release download or contamination of normal dev preferences. Added optional fourth compact anime-pair tuple value for support while preserving three-value readers; formal format/semantic migration remains M3.7. First-encounter edge and first-N per-user caps remain M3.5–M3.6. Added PR workflow but kept M0.6 open pending an actual run.

**Single next task:** Verify M0.6 on a pull request and address any CI-only failures; then update this record and its checkbox.
