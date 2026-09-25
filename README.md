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
npm run typecheck
npm test
npm run test:python
npm run build:web
npx playwright install chromium
npm run test:e2e
```

The Python smoke tests require NumPy (`python -m pip install "numpy>=2,<3"`). The Playwright install is needed once per machine. The browser smoke test starts the synthetic demo and blocks external requests. Pull-request CI runs these checks without fetching production data or using provider credentials. See `docs/DEVELOPMENT_PLAN.md` and `docs/PROGRESS.md` for acceptance criteria and evidence.

## 1) Collect MAL Data into Anonymized SQLite

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
- Only rated entries (`score > 0`) are imported.

### Grow the Network Automatically

This crawls outward from seed users and discovers more users through shared anime activity:

```bash
npm run expand:network -- "Gigguk,TheAnimeMan" "your-private-salt" 150 "data/anime.sqlite"
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
- `--max-mal-pages-per-user` (default `0`, unlimited)

### One-Command 100x Scale-Up

Automatically scales target users to `current_users * 100` (with sane crawl defaults):

```bash
npm run expand:100x
```

Optional overrides:

- `--target-total-users <count>` for an explicit target
- `--max-mal-pages-per-user <count>` to cap MAL pages per user for faster, lighter expansion

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

Note:
- JSON exports are minified by default to reduce disk size.
- Use `--pretty-json` if you need human-readable formatting.
- Use `--compact-only` to write only compact files.
- A safety guard now limits unique anime-anime edges during build to prevent memory blowups on very large datasets. Default cap: `2,000,000` (set `--max-anime-anime-edges 0` for unlimited).

Graph rules implemented:

- User node and anime node for each entity.
- `user -> anime` edge weight = normalized score (`raw - user_avg`).
- For each user, every rated anime pair gets an `anime <-> anime` edge with pair score:
  `(anime_a_normalized + anime_b_normalized) / 2`
- For each anime pair, keep a sum and observation count. The edge weight is the
  arithmetic mean of its users' pair scores; `support` is the observation count.
  Legacy graph edges carry `support`, and compact anime-anime tuples can carry it
  as a fourth value: `[leftIndex, rightIndex, weight, support]`.

This is a corrected compatibility statistic, not a correlation or a validated
similarity measure. The unique-pair cap still keeps first-encountered pairs and
the per-user rating cap still takes the first N ratings. Their selection bias and
work budget are tracked in M3.5–M3.6; do not treat capped graph output as order
independent yet.

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
