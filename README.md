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

## 2) Build Dataset + Graph JSON

```bash
npm run build:graph -- "data/anime.sqlite" "data/anonymized-ratings.json" "data/graph.json" 0
```

Outputs:

- `data/anonymized-ratings.json`
- `data/graph.json`

Graph rules implemented:

- User node and anime node for each entity.
- `user -> anime` edge weight = normalized score (`raw - user_avg`).
- For each user, every rated anime pair gets an `anime <-> anime` edge with pair score:
  `(anime_a_normalized + anime_b_normalized) / 2`
- If an anime pair edge already exists, update with:
  `(existing_weight + pair_score) / 2`

## 3) Run Web App

```bash
npm run sync:web
npm run dev:web
```

Optional ML recommendation engine for the web app:

```bash
npm run ml:train
npm run ml:export:web
npm run sync:web
```

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
