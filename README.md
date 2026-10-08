# WhatAnimeShouldIWatch

Recommendation and discovery project with an offline synthetic development path. The repository contains:

- TypeScript collection, SQLite, normalization, and graph-building pipelines.
- A Vite/TypeScript web app for recommendations, local preferences and watchlists, and graph exploration.
- Python model experiments and split-first evaluation tools.
- An optional Rust/Dioxus local graph companion with a checked Windows x64 candidate package and an invented-data clean-host GUI check. Desktop publication remains held.

New provider-derived collection, publication, and deployment remain held pending the source/use and release approvals in [the living plan](docs/DEVELOPMENT_PLAN.md). Use the invented demo below for routine development.

## Repo Layout

- `pipeline/`: ingestion, anonymization, normalization, graph generation.
- `data/`: SQLite + exported JSON data.
- `web/`: Vite/TypeScript discovery and recommendation app with a network explorer (GitHub Pages compatible).
- `desktop/`: Rust/Dioxus local graph companion; [scope decision](docs/decisions/0042-desktop-companion-scope.md) and [candidate limits](desktop/README.md).

## Install

```bash
npm install
```

For a reproducible checkout, use `npm ci` with the committed lockfile.

## Offline synthetic demo and fast checks

The fixture contains invented user IDs, titles, ratings, and metadata. It covers overlapping and opposite tastes, equal scores, sparse and empty users, an isolated anime, a duplicate rating, an unknown ID, and non-ASCII titles. It does not contain a real viewing history.

```bash
npm ci
npm run dev:demo
```

Open `http://127.0.0.1:5173/`. The banner says **SYNTHETIC DEMO DATA**. This mode generates a catalog, aggregate-only v3 recommendation graph, linked v3 explorer, and tiny synthetic model in ignored `web/public/demo-data/`. The default view loads the catalog and aggregate graph; the model and explorer load when opened. A separate invented v2 graph remains there for compatibility tests and fixture manifest construction; the demo never requests it for ranking. Sample-based popularity is unavailable on v3, so browsing starts with invented community scores. Demo mode makes no live metadata or seasonal requests. Preferences use separate local storage keys, and username import is disabled. A missing fixture is an error; regular `dev:web` still requires its normal data files.

The web fonts are bundled with the site, so the demo's first view needs no external font service. Their redistribution notices are in [`web/public/licenses/`](web/public/licenses/).

`npm run metadata:fixture:check` exercises the bounded offline Wikibase statement mapper and acquisition adapter with invented bytes and mocked transport. It checks exact IDs, ranks, qualifiers, units, conflicts, unknowns, provenance, coverage, scope/retry/byte limits, and the closed-study refusal. It performs no real lookup or publication. The delegated 2026-10-07 local pilot is completed and its exact approval record is closed; it is not a standing collection or public-catalog permission. See [decision 0043](docs/decisions/0043-catalog-source-candidate.md) for aggregate findings and the remaining review.

It also checks the proposed follow-up's separate offline preflight/reservation and complete-definition inventory with invented data. Mocked path/access facts and in-memory reservations do not verify actual Windows ACLs or durable one-use storage; no new live adapter or approval record exists. See [decision 0046](docs/decisions/0046-bounded-followup-study-gates.md). Routine Node unit suites use two workers so the complete suite does not spawn an unbounded number of memory-consuming processes.

`npm run metadata:windows:fixture:check` separately tests the actual Windows private-path/ACL and durable-reservation ports in newly created invented temporary directories. It makes no source request and changes no real pilot or parent permissions. A Linux run checks platform refusal and skips the Windows-only cases; fixture CI has a narrow Windows job. This verifies local OS mechanics, not acquisition or source-use approval.

`npm run metadata:fixture:check` also checks the local field-readiness audit with invented candidates and a fixed plan. It counts joint field availability, retained missing-ID denominators, and format/classification scope; a passing local target never authorizes publication. [Decision 0044](docs/decisions/0044-catalog-field-readiness.md) records the proposed date, genre, certificate-qualifier rules and a separately held follow-up scope.

An explicit offline v2 policy now tests the certificate proposal with invented film ratings and references. It accepts only the exact candidate rule, keeps unsupported or conflicting evidence unknown, and leaves certificate values out of catalog and aggregate audit output. V1 still rejects every qualified classification. This technical candidate does not approve real mapping tables, reinterpret the completed pilot, or authorize a new source request.

The metadata fixture command also tests [decision 0045](docs/decisions/0045-tv-date-scope-and-provenance.md)'s private TV date audit. Byte-bound series/season declarations, exact type scope, date precision and all live primary/fallback evidence are checked with invented data. Its per-ID provenance remains private and it does not change catalog years; current v1/v2 mapping and browser filters continue to ignore P580. A successful local audit is not approval of first-airing meaning, source use or publication.

An explicit synthetic-only wrapper now exercises scoped years in strict catalog bytes and mocked browser bundles. It returns its hash-bound private audit separately, recomputes final coverage, leaves ambiguous TV/unknown-format years null, and preserves known non-TV publication years. It has no transport, writer, installer or publication entry point; v1/v2 behavior stays unchanged. Fixture markers and passing checks do not approve real mappings or source use.

Run the fast checks separately:

```bash
npm run data:fixture
npm run data:fixture:check
npm run metadata:fixture:check
npm run split:fixture:check
npm run eval:new-user:fixture
npm run typecheck
npm test
npm run test:python
python -m unittest discover -s scripts/tests
npm run build:web
npx playwright install chromium
npm run test:e2e
npm run eval:user-vector:fixture
npm run eval:hybrid-blend:fixture
npm run eval:franchise-diversity:fixture
```

The Python smoke and workflow tests require NumPy, PyYAML, and Optuna (`python -m pip install "numpy>=2,<3" "PyYAML>=6,<7" "optuna>=4,<5"`). The Playwright install is needed once per machine. Browser tests use the synthetic demo or a normal-mode app with mocked providers; no real usernames are queried. Pull-request CI runs these checks without fetching production data or using provider credentials. See `docs/DEVELOPMENT_PLAN.md` and `docs/PROGRESS.md` for acceptance criteria and evidence.

To measure the built browser app without provider traffic, run `npm run bench:browser:fixture` followed by `npm run bench:browser:scale`, then `npm run bench:browser:budgets:check` on a clean checkout. The scale command derives a deterministic invented 3,000-title/3,000-user aggregate v3 graph from local ratings, builds a separate 8,000-pair explorer and 16-factor model, and replaces only ignored `web/dist/demo-data/` files. Raw samples stay in ignored `web/test-results/performance-*.json`. The budget check requires reports from the current clean commit, complete zero-external-request samples, three cold/warm runs, and 24 warm updates per mode on the declared mobile-emulated Chromium profile, then checks 3,000 ms cold first-view and 200 ms p95 warm-update limits. Re-run both benchmarks after changing commits. The generated shape is a repeatable scale probe; it is not evidence of the size of an approved release or performance on a physical phone. The full M7.7 gate remains open pending a justified representative dataset and real-condition measurements.

The scale report also separates recommendation scoring, eligibility, franchise selection, card/DOM work, and network selection, construction, layout, keyboard-list preparation, control updates, and SVG drawing. It reports first-render long tasks separately from later interactions; one task can overlap several nested phases. Large network overviews group signed edges and, above 500 nodes, node circles into SVG paths while the keyboard list still exposes every node. Large Graphology and SVG builds yield between batches and phase boundaries, cancel stale renders, and reuse one matching overview graph and SVG. For a local one-run diagnostic after generating the scale build, run `node web/bench/browser-performance.mjs --scenario=scale --runs=1 --updates=8 --graph-renders=4 --trace-graph`. It writes ignored `web/test-results/performance-graph-trace.json` and `performance-scale-trace.json` without replacing the three-run budget report. The invented trace located a remaining long task in the mobile controls reveal after the first SVG render, not in an edge-geometry batch. CPU work remains on the main thread, and M7.5 is still open.

For an opt-in cold-load trace on that same built invented scale, replace `--trace-graph` with `--trace-cold`. It writes ignored `web/test-results/performance-scale-cold-trace.json` and `performance-scale-cold-trace-summary.json` without replacing the three-run budget report. The trace separates parse/validation, graph indexing, title/network suggestion creation, and first recommendation work on the marked renderer thread. Startup yields and 500-option DOM batches keep the full sorted suggestion lists while allowing a browser task between phases. This is a local diagnostic, not a representative-device result or a reason by itself to add a worker.

To distinguish native pointer activation from the app's controls callback, repeat that one-run command with `--trace-graph-dom-click` or `--trace-graph-touch` (one input flag at a time). The first opens the same panel with `element.click()`; the second enables Chromium touch emulation and taps it. They write separate ignored `performance-graph-trace-<input>.json` and `performance-scale-trace-<input>.json` reports, summarizing tasks on the marked renderer thread, including major GC. The DOM-click route bypasses pointer down and focus handling, so it is only a diagnostic comparison, not a user-interaction or performance-budget substitute. On the clean invented scale, mouse and emulated touch traces had later style/layout tasks; the DOM-click trace also had a layout task and a separate major-GC task. The app click callback was under 1 ms in the touch trace. These short local samples do not establish physical-device behavior or a speedup.

Large candidate rankings yield to the browser between scoring, eligibility, and result rendering. A newer preference or setting change aborts a queued continuation. The synthetic scale probe measures this scheduling path; the network has its own abortable build and drawing path.

The user-vector diagnostic calls the same new-user scorer used by the browser and compares it with an evaluation-only ridge fold-in reference on invented held-out ratings. It reports Hit@3, reciprocal rank, and local latency; it is not a production ranking-quality result. [Decision 0010](docs/decisions/0010-user-vector-evaluation.md) records the fixed protocol, results, and why the simpler browser average remains selected.

The [split-first new-user fixture](docs/decisions/0018-new-user-split-first-evaluation.md) fits item factors from an invented training-only snapshot and scores four separate invented users with 1, 3, 5, and 10 supplied ratings through the browser preference mapper, model scorer, and eligibility policy. `npm run eval:new-user:fixture` checks its pinned split and prints validation-only model ranks, displayed fallback engines, and a separate warm-user validation report. It does not inspect the fit-user test partition or claim production ranking quality.

The [split-first hybrid selector](docs/decisions/0020-hybrid-validation-boundary.md) evaluates five predeclared blend weights on that browser new-user path, after eligibility and final-list selection. Run `npm run eval:hybrid:split:select` once to freeze an ignored local validation record. It hashes the separate invented final cohort without scoring it; the final-report command is reserved until the later baseline and reporting protocol is fixed. The selected synthetic weight is not the product default.

The pinned [raw-interaction split protocol](docs/decisions/0014-raw-interaction-splits.md) creates train/validation/test IDs from invented raw scores before any centering or graph work. `npm run split:fixture:check` verifies its immutable 13-interaction manifest. The current exported ratings contain no verified rating/viewing times, so the fixture uses a seeded per-user holdout. The legacy MF training and LightGCN scripts still use a full-graph evaluation path and their metrics are not M5 release evidence; `ml:search` now uses the split-first Optuna validation route.

The web app validates graph, demo catalog, and model JSON before scoring or rendering. New graph exports use compact or legacy v2 with explicit semantics, support, build configuration, truncation counts, and dataset identity. The browser loads a separate, identified v2 explorer sample while recommendations use the recommendation graph. Existing compact v1 and unversioned legacy graphs remain readable, including three- or four-value compact anime-pair tuples; unknown or mixed formats fail with a named artifact and field. If the optional model is missing, invalid, or cannot score eligible candidates, model/hybrid mode serves graph results or a labeled catalog coverage baseline; the selected mode stays saved. See [decision 0009](docs/decisions/0009-graph-v2-contract.md) for identities and compatibility limits.

## Local watched-list import

In **Import & Profiles**, choose a local `.txt` file up to 128 KiB, a decompressed MAL-style `.xml` anime list up to 2 MiB, or paste text. Text lines use `animeId[, score[, status[, episodes]]]` or a title in place of the ID (the synthetic demo accepts `101, 9, Completed, 12`). The XML reader accepts anime IDs, titles, `my_status`, `my_watched_episodes`, and `my_score`; it rejects DTDs and malformed fields. Files are read only in the browser. Preview counts, including unscored and unmapped entries, then choose **Merge** or **Replace** and apply. Replace also resets current preferences; the preview counts those removals. Re-importing the same file merges by source identity. Imported history, status, episode progress, native score scale, and unmatched titles persist with local recommendation state and named profiles. This implementation was verified with synthetic XML and text, not a real account export; no Crunchyroll-native export parser is claimed.

Adding a title manually defaults to **Seen / Unrated**, which excludes it from suggestions without treating it as a favorite. Explicit **Liked** titles seed positive pair-graph suggestions; the existing model also uses **Disliked** as negative evidence. Importance is user-controlled emphasis, while confidence records how strongly an imported score supports its sentiment. After converting from its native scale, an imported score at or below 4/10 is Disliked, above 4 and below 7 is Seen, and 7 or above is Liked; unscored watches are Seen and planned titles add no preference. A manual choice takes precedence over a later import. Earlier saved watch weights migrate to v5 state/profile keys: their numeric value becomes importance, linked history scores determine sentiment, ambiguous default-weight watches stay Seen, and higher unlinked weights become half-confidence likes. The original v1/v4 storage and raw backups stay intact, and the app displays a migration notice. Browsing seasonal ideas does not add them as watched.

**Include Only Candidates** narrows the current ranking to listed titles; it does not add a title to graph or model rankings. When catalog fallback is active, it narrows that catalog list. Seen titles, imported watched history, explicit exclusions, and titles absent from the current graph catalog stay out even if included. Exclusion wins when a title is in both lists. Genre, year, and minimum community score are required filters when selected; candidates without metadata needed to check them are skipped. Hybrid ranking filters each source before weighted rank fusion so ineligible titles cannot change visible ranks. A source with no eligible candidates yields its weight to the available source, and the 0%/100% endpoints keep every candidate from the selected source. Hybrid rank points express relative ordering, not a probability or community rating. Catalog coverage fallback orders titles by their count of positive graph connections, then anime ID, and labels that count instead of calling it a personalized score.

Recommendation cards name the actual source titles and show how many distinct titles contributed. Open **How the score was calculated** to see graph edge sums, or the model's global mean, candidate bias, normalized signed title terms, and mapped-signal count. Hybrid cards show effective graph/model weights and the rank points from each source, with their underlying score equations. Any display-rounding adjustment is a separate term so the shown sum matches the shown score. Cards disclose that these scores have no calibrated confidence interval or probability; a reason without score provenance is labelled qualitative. [Decision 0012](docs/decisions/0012-score-explanations.md) describes the equations and limits.

The final list defaults to **Prefer variety**: it withholds a candidate with a known immediate prequel absent from watched history and shows at most one title per known or conservatively title-suggested series. **Allow related titles and known sequels** restores the original eligible ranking and scores; this choice is saved with local state and profiles. Cards show known prequel status or say prerequisites are unverified. Missing, empty, or one-sided relation data may miss a connection, and title similarities can be wrong. The selector uses existing candidate metadata without additional provider requests and never overrides watched, exclusion, Include Only, or metadata filters. [Decision 0013](docs/decisions/0013-franchise-diversity.md) records the invented relevance/diversity comparison; it does not establish production quality.

With no preference signal, **Automatic** discovery uses sampled rating counts for legacy/v1/v2 recommendation graphs that include user-anime edge rows. The count is a proxy for that graph sample, not global popularity. Aggregate-only v3 graphs have no such rows, so Automatic uses **Community score** exploration instead. That view ranks only titles with known score metadata and can show partial coverage or no confirmed match. The genre filter explores titles with known genre metadata. **Shared genres** compares candidates with explicitly Liked titles using exact genre overlap weighted by preference importance and confidence. A sparse Liked history with no eligible graph result can use that content baseline automatically when metadata supports it. All discovery views apply the same watched, exclusion, Include Only, and required-filter rules; browsing never adds a watched or liked title or changes the selected recommendation engine. Seasonal ideas likewise remain browsing-only.

The synthetic demo contains local metadata and makes no provider requests. In normal mode, an untouched Automatic discovery view uses only metadata already loaded. Choosing **Browse** or deliberately selecting **Community score** or **Shared genres**, pressing the next-catalog-check control, or falling back from a sparse Liked title may send explicitly Liked source anime IDs and a batch of up to 12 eligible catalog anime IDs through the existing Jikan metadata adapter. The page reports known score/genre coverage and offers bounded follow-up checks across the eligible catalog. A required filter excludes titles whose needed metadata is unknown; a partial scan is labelled rather than reported as a complete no-match. Provider-dependent product use still requires the permissions decision below.

Username import is optional. The entered username goes directly to the selected AniList or MAL provider to read a public list into the same preview. Nothing is applied until the user chooses merge or replace. If direct MAL access fails, the app reports the failure and keeps the current history; it does not try a proxy. A provider field unavailable in a response stays unknown rather than being guessed. AniList history retains the user's declared native score format; sad, neutral, and happy three-point smileys map to Disliked, Seen, and Liked. The M2.1 permissions review still holds provider-dependent product use; fixture tests are not approval for live import, storage, or deployment.

## 1) Collect MAL Data into Anonymized SQLite

**Permission hold:** This is an existing pipeline path, not an authorized routine setup step. Review [provider data permissions](docs/decisions/0001-provider-data-permissions.md) and obtain the recorded source/use clearance before running it or the network-expansion, Jikan feature-harvesting, training, or release-publication commands below. Use the synthetic fixture commands above for development.

This uses public MAL lists from:
`https://myanimelist.net/animelist/{username}/load.json`

```bash
npm run collect -- "username1,username2" "your-private-salt" 800 "data/anime.sqlite"
```

Positional arguments:

- `1:` MAL usernames (comma-separated)
- `2:` anonymization salt
- `3:` delay in ms between page requests
- `4:` SQLite output path

Notes:

- Keep the anonymization salt private and stable if you want deterministic IDs over time.
- If you change salt, remove `data/anime.sqlite` first (or use a new DB path), otherwise the same user may be imported as a new anonymized user ID.
- Only rated entries (`score > 0`) enter the committed ratings table. The collector validates every returned entry, stages full 300-entry pages, and resumes from the last committed page checkpoint after a failure or page cap. `--max-pages-per-user` limits pages **per run**; a full capped page does not replace ratings.
- A nonempty short page closes the snapshot. The collector then replaces that user's ratings and normalized scores in one SQLite transaction, including changed scores and removed entries. Empty/private, malformed, failed, and interrupted responses keep the last complete ratings. Because an empty response cannot prove whether a list is genuinely empty, a list with exactly a multiple of 300 entries remains unresolved on this site route.
- `collection_status` records the latest per-user outcome and last completion under an anonymized ID; `collection_pages` and `collection_staged_entries` hold only incomplete pages. Incomplete runs exit nonzero. The site endpoint has no snapshot version, so a list changing during offset pagination can still produce a mixed response; source approval and a versioned provider route are separate decisions.
- SQLite schema version 3 migrates older databases transactionally. New rows record the MAL source route and anime ID, local page-fetch time, import-run outcome, and normalization formula version. Existing rows keep unknown provenance; no timestamp is fabricated. The current site adapter supplies no verified provider update time, so `provider_updated_at` stays empty. See [decision 0004](docs/decisions/0004-collection-provenance.md). These fields and the stored user key remain restricted data under the permission hold above.

### Grow the Network Automatically

This crawls outward from seed users and discovers more users through shared anime activity:

```bash
npm run expand:network -- "invented-user,example-neighbor" "your-private-salt" 150 "data/anime.sqlite"
```

Positional arguments:

- `1:` seed usernames (comma-separated)
- `2:` anonymization salt
- `3:` target total user count in DB
- `4:` SQLite path

Useful env/config flags:

- `--discovery-anime-per-user` (default `8`)
- `--updates-pages-per-anime` (default `1`)
- `--fallback-users-pages` (default `2`)
- `--min-scored-anime` (default `30`)
- `--max-mal-pages-per-user` (default `0`, unlimited; a full capped page is checkpointed for a later run)

### One-Command 100x Scale-Up

Automatically scales target users to `current_users * 100` (with sane crawl defaults):

```bash
npm run expand:100x
```

Optional overrides:

- `--target-total-users <count>` for an explicit target
- `--max-mal-pages-per-user <count>` to cap MAL pages per user per run; capped users remain deferred until a later run completes their snapshot

## 2) Build Dataset + Graph JSON

```bash
npm run build:graph -- "data/anime.sqlite" "data/anonymized-ratings.json" "data/graph.json" 0
```

Compact-only shortcut:

```bash
npm run build:graph:compact
```

Outputs:

- `data/anonymized-ratings.compact.json` (compact)
- `data/graph.compact.json` (compact)
- `data/anonymized-ratings.json` (legacy, unless `--compact-only`)
- `data/graph.json` (legacy, unless `--compact-only`)
- `data/graph.json.report.json` (build selection/coverage/resource report; compact-only uses `data/graph.compact.json.report.json`)

Both graph formats now carry `graphId`, a canonical `dataset.sha256`, current pair-preference semantics, selection settings, and truncation counts. The report repeats both identities. The compact and legacy v2 recommendation exports share an ID. `sync:web` writes a separate bounded `graph-explorer.compact.json.gz` for visualization from either export; it is not used for recommendation scoring. Its source ID must match the recommendation graph. Existing v1 graphs remain a deliberate read-only compatibility path. No provider-derived data or model regeneration was performed for this format change.

Note:
- JSON exports are minified by default to reduce disk size.
- Use `--pretty-json` if you need human-readable formatting.
- Use `--compact-only` to write only compact files.
- `--max-anime-anime-edges` limits the selected output to 2,000,000 pairs by default (`0` means no output cap). Exact candidates are ranked by co-rater support, absolute pair mean, then numeric IDs; `--max-neighbors-per-anime` optionally limits selected degree (`0` by default).
- Separate fail-closed input limits are `--max-pair-visits 20000000` and `--max-pair-candidates 2500000`. If either is exceeded, graph building fails before writing new artifacts instead of returning a biased partial graph. `--min-pair-support` defaults to 1.
- `--max-ratings-per-user N` (default `0`, unlimited) selects up to N ratings per user by the smallest SHA-256 ranks using `--pair-selection-seed` (default `0`). The same subset supplies user-anime and anime-anime graph edges; the ratings dataset export remains complete. `--out-report` overrides the build report path. The report records the seed, policy, skipped ratings and pair observations, retained anime/ratings/observation fractions, output truncation, elapsed build time, and peak process RSS. Exact candidate-pair recall is unknown for an ordinary capped build; the report records `null` for it.
- Run `npm run benchmark:pair-cap` for a reproducible, invented-data comparison of exact and capped pair coverage, runtime, and peak RSS in separate processes. No provider or production data is read.

Graph rules implemented:

- User node and anime node for each entity.
- `user -> anime` edge weight = normalized score (`raw - user_avg`).
- For each user, every selected rated anime pair gets an `anime <-> anime` edge with pair score:
  `(anime_a_normalized + anime_b_normalized) / 2`
- For each anime pair, keep a sum and observation count. The edge weight is the
  arithmetic mean of its users' pair scores; `support` is the observation count.
  Legacy graph edges carry `support`, and compact anime-anime tuples can carry it
  as a fourth value: `[leftIndex, rightIndex, weight, support]`.

This is a corrected compatibility statistic, not a correlation or a validated
similarity measure. The output cap now selects deterministically from exact
candidate statistics within its independent input budgets. A per-user cap now
uses a recorded seed and an input-order independent hash sample, documented in
[decision 0008](docs/decisions/0008-seeded-per-user-selection.md). Sampling
changes support and pair weights relative to the full input; the report's
coverage counts do not establish recommendation quality. [Decision 0005](docs/decisions/0005-graph-edge-semantics.md)
compares this pair preference with a support-shrunk, user-centered item cosine
on an invented fixture (`node --import tsx --test pipeline/test/edge-semantics.benchmark.test.ts`).
The proposed similarity is not wired into v1 exports, browser ranking, or training.
[Decision 0006](docs/decisions/0006-signed-graph-evidence.md) tests signed,
neutral, sparse, flat-rater, and missing-overlap cases. In the v1 browser,
nonpositive pair-preference edges are shown with their sign in the network but
do not seed or penalize recommendations. The legacy trainer now uses positive
v1 edges only for attractive regularization; `--graph-min-abs-weight` retains
its CLI spelling but thresholds those positive weights. Existing model files
were not retrained. A signed-similarity recommendation path remains gated on
split-first evaluation and a deliberately versioned graph-semantic migration.

## 3) Run Web App

```bash
npm run data:fetch:release
npm run sync:web
npm run dev:web
```

`sync:web` now writes compressed files (`*.json.gz`) to `web/public/data` by default.
For the legacy normal-mode path, the web app tries `.json.gz` first. A 404 selects the
plain `.json` file; a browser without `DecompressionStream` can also use a plain
sibling. A gzip-only legacy host therefore requires that browser capability. A
present malformed gzip file or HTTP error fails with the named asset instead of
silently using a different copy. Browser-decoded gzip bodies are accepted as JSON.
Legacy requests bypass the browser cache and are bounded to 64 MiB compressed and
256 MiB decoded/plain bytes. A pinned versioned bundle checks exact manifest and
asset hashes, then retries a mismatched cached response once without cache before
failing closed. These are transport limits, not a claim that provider-derived
assets are approved for publication.

Optional ML recommendation engine for the web app:

```bash
npm run ml:train
npm run ml:export:web
npm run sync:web
```

Optional GNN ranking baseline evaluation:

```bash
npm run ml:gnn:eval -- --epochs 12 --layers 3
```

Scheduled model retraining workflow (weekly + manual trigger):

- `.github/workflows/ml-retrain.yml`

The provider-derived retrain, data-release, and Pages jobs are held by the [use-specific workflow approval gates](docs/decisions/0002-provider-workflow-gates.md). The current approval manifest is empty. These commands and workflows are reference paths, not routine development steps; use `npm run dev:demo` for fixture work.

Optional `sync:web` flags:
- `SYNC_WEB_INCLUDE_DATASET=1` to also copy anonymized ratings for web (`*.compact.json` preferred).
- `SYNC_WEB_KEEP_JSON=1` to keep plain `.json` next to `.json.gz`.

## Data Artifacts In Releases

Large JSON artifacts are no longer intended to be versioned in git.
Use GitHub release assets (`data-*` tags) as the source of truth.

Download into local `data/`:

```bash
npm run data:fetch:release
```

Publish/update data release assets from your local `data/*.compact.json` files:

```bash
npm run data:publish:release
```

This packages the local ignored compact data files and uploads them to the
`data-latest` release by default.

The workflow `.github/workflows/publish-data-release.yml` is still useful when
you want to publish from:

- checked-in repo data (`source_mode=repo`)
- a previous workflow artifact (`source_mode=artifact`)
- the existing release payload (`source_mode=release`)

## 4) Build Static Site for GitHub Pages

```bash
npm run sync:web
npm run build:web
```

GitHub Actions workflow included:
`.github/workflows/deploy-web.yml`

## 5) Desktop graph companion track

[Decision 0042](docs/decisions/0042-desktop-companion-scope.md) keeps desktop as an optional local graph companion; the web app owns recommendations. The Rust app now starts with no data, offers an explicit invented demo or local v3 manifest picker, and displays precomputed signed pair evidence without reading raw ratings. See [desktop/README.md](desktop/README.md) for the synthetic fixture commands and the limits of local verification.

## 6) Desktop release hold

The old tag-triggered EXE publisher has been removed. M9.4 has a synthetic bounded-loading check. M9.5 creates and verifies a Windows x64 candidate ZIP with a pinned Rust toolchain and explicit runtime prerequisites. M9.6's invented kit now passes four GUI states in an interactive Windows x64 runner after its checkout is emptied. CI uploads no package. Any desktop publication still needs a separately reviewed release route and permitted graph source.
