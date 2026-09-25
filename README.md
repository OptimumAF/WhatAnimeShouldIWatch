# WhatAnimeShouldIWatch

End-to-end project for:

- Pulling public MyAnimeList user scores.
- Storing anonymized users + ratings in SQLite.
- Normalizing scores per user by each user's mean score.
- Building a weighted bipartite-plus-anime graph.
- Visualizing the graph in a static GitHub Pages web app.
- Running a matching local desktop app in Rust + Dioxus.

## Repo Layout

- `pipeline/`: ingestion, anonymization, normalization, graph generation.
- `data/`: SQLite + exported JSON data.
- `web/`: Vite TypeScript network graph viewer (GitHub Pages compatible).
- `desktop/`: Rust/Dioxus desktop graph app.

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

Open `http://127.0.0.1:5173/`. The banner says **SYNTHETIC DEMO DATA**. This mode generates local graph, catalog, and tiny synthetic model files in ignored `web/public/demo-data/`, uses those files for graph/model/filter views, and does not make live metadata or seasonal requests. Demo preferences use separate local storage keys, and username import is disabled. A missing fixture is an error; regular `dev:web` still requires its normal data files.

Run the fast checks separately:

```bash
npm run data:fixture
npm run data:fixture:check
npm run split:fixture:check
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

The Python smoke and workflow tests require NumPy and PyYAML (`python -m pip install "numpy>=2,<3" "PyYAML>=6,<7"`). The Playwright install is needed once per machine. Browser tests use the synthetic demo or a normal-mode app with mocked providers; no real usernames are queried. Pull-request CI runs these checks without fetching production data or using provider credentials. See `docs/DEVELOPMENT_PLAN.md` and `docs/PROGRESS.md` for acceptance criteria and evidence.

The user-vector diagnostic calls the same new-user scorer used by the browser and compares it with an evaluation-only ridge fold-in reference on invented held-out ratings. It reports Hit@3, reciprocal rank, and local latency; it is not a production ranking-quality result. [Decision 0010](docs/decisions/0010-user-vector-evaluation.md) records the fixed protocol, results, and why the simpler browser average remains selected.

The pinned [raw-interaction split protocol](docs/decisions/0014-raw-interaction-splits.md) creates train/validation/test IDs from invented raw scores before any centering or graph work. `npm run split:fixture:check` verifies its immutable 13-interaction manifest. The current exported ratings contain no verified rating/viewing times, so the fixture uses a seeded per-user holdout. Existing ML training/search/GNN scripts still use their older full-graph evaluation path; their metrics are not M5 release evidence until train-only preprocessing and leakage checks are wired in.

The web app validates graph, demo catalog, and model JSON before scoring or rendering. New graph exports use compact or legacy v2 with explicit semantics, support, build configuration, truncation counts, and dataset identity. The browser loads a separate, identified v2 explorer sample while recommendations use the recommendation graph. Existing compact v1 and unversioned legacy graphs remain readable, including three- or four-value compact anime-pair tuples; unknown or mixed formats fail with a named artifact and field. If the optional model is missing, invalid, or cannot score eligible candidates, model/hybrid mode serves graph results or a labeled catalog coverage baseline; the selected mode stays saved. See [decision 0009](docs/decisions/0009-graph-v2-contract.md) for identities and compatibility limits.

## Local watched-list import

In **Import & Profiles**, choose a local `.txt` file up to 128 KiB, a decompressed MAL-style `.xml` anime list up to 2 MiB, or paste text. Text lines use `animeId[, score[, status[, episodes]]]` or a title in place of the ID (the synthetic demo accepts `101, 9, Completed, 12`). The XML reader accepts anime IDs, titles, `my_status`, `my_watched_episodes`, and `my_score`; it rejects DTDs and malformed fields. Files are read only in the browser. Preview counts, including unscored and unmapped entries, then choose **Merge** or **Replace** and apply. Replace also resets current preferences; the preview counts those removals. Re-importing the same file merges by source identity. Imported history, status, episode progress, native score scale, and unmatched titles persist with local recommendation state and named profiles. This implementation was verified with synthetic XML and text, not a real account export; no Crunchyroll-native export parser is claimed.

Adding a title manually defaults to **Seen / Unrated**, which excludes it from suggestions without treating it as a favorite. Explicit **Liked** titles seed positive pair-graph suggestions; the existing model also uses **Disliked** as negative evidence. Importance is user-controlled emphasis, while confidence records how strongly an imported score supports its sentiment. After converting from its native scale, an imported score at or below 4/10 is Disliked, above 4 and below 7 is Seen, and 7 or above is Liked; unscored watches are Seen and planned titles add no preference. A manual choice takes precedence over a later import. Earlier saved watch weights migrate to v5 state/profile keys: their numeric value becomes importance, linked history scores determine sentiment, ambiguous default-weight watches stay Seen, and higher unlinked weights become half-confidence likes. The original v1/v4 storage and raw backups stay intact, and the app displays a migration notice. Browsing seasonal ideas does not add them as watched.

**Include Only Candidates** narrows the current ranking to listed titles; it does not add a title to graph or model rankings. When catalog fallback is active, it narrows that catalog list. Seen titles, imported watched history, explicit exclusions, and titles absent from the current graph catalog stay out even if included. Exclusion wins when a title is in both lists. Genre, year, and minimum community score are required filters when selected; candidates without metadata needed to check them are skipped. Hybrid ranking filters each source before weighted rank fusion so ineligible titles cannot change visible ranks. A source with no eligible candidates yields its weight to the available source, and the 0%/100% endpoints keep every candidate from the selected source. Hybrid rank points express relative ordering, not a probability or community rating. Catalog coverage fallback orders titles by their count of positive graph connections, then anime ID, and labels that count instead of calling it a personalized score.

Recommendation cards name the actual source titles and show how many distinct titles contributed. Open **How the score was calculated** to see graph edge sums, or the model's global mean, candidate bias, normalized signed title terms, and mapped-signal count. Hybrid cards show effective graph/model weights and the rank points from each source, with their underlying score equations. Any display-rounding adjustment is a separate term so the shown sum matches the shown score. Cards disclose that these scores have no calibrated confidence interval or probability; a reason without score provenance is labelled qualitative. [Decision 0012](docs/decisions/0012-score-explanations.md) describes the equations and limits.

The final list defaults to **Prefer variety**: it withholds a candidate with a known immediate prequel absent from watched history and shows at most one title per known or conservatively title-suggested series. **Allow related titles and known sequels** restores the original eligible ranking and scores; this choice is saved with local state and profiles. Cards show known prequel status or say prerequisites are unverified. Missing, empty, or one-sided relation data may miss a connection, and title similarities can be wrong. The selector uses existing candidate metadata without additional provider requests and never overrides watched, exclusion, Include Only, or metadata filters. [Decision 0013](docs/decisions/0013-franchise-diversity.md) records the invented relevance/diversity comparison; it does not establish production quality.

With no preference signal, **Automatic** discovery lists catalog titles by the number of user-anime rating edges retained in the loaded recommendation graph. This is a sample-based popularity proxy, not a global audience count. The **Community score** view ranks only titles whose metadata has a known score; the genre filter explores titles with known genre metadata. **Shared genres** compares candidates with explicitly Liked titles using exact genre overlap weighted by preference importance and confidence. A sparse Liked history with no eligible graph result can use that content baseline automatically when metadata supports it. All discovery views apply the same watched, exclusion, Include Only, and required-filter rules; browsing never adds a watched or liked title or changes the selected recommendation engine. Seasonal ideas likewise remain browsing-only.

The synthetic demo contains local metadata and makes no provider requests. In normal mode, the no-preference popularity view makes no anime-metadata request. Selecting a metadata-based view, pressing **Check 12 more catalog titles**, or falling back from a sparse Liked title may send the Liked title's anime ID and up to 12 eligible catalog anime IDs through the existing Jikan metadata adapter. The page reports known score/genre coverage. Its bounded check can miss lower-ranked matching titles; M6.4 owns broader catalog completeness.

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
The web app loads gzip first, then falls back to plain JSON.

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

## 5) Run Rust/Dioxus Desktop App

Install Rust first, then:

```bash
cd desktop
cargo run
```

## 6) Publish Desktop EXE Release

Pushing a version tag (`v*`) triggers a workflow that builds the Windows EXE and attaches:

- `anime_graph_desktop.exe`
- `anime_graph_desktop.exe.sha256`

Example:

```bash
git tag v0.1.0
git push origin v0.1.0
```
