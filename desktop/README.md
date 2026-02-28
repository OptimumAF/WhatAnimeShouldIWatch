# Desktop App (Rust + Dioxus)

This app loads `../data/anonymized-ratings.json`, normalizes user scores by each user's mean score, and generates the same graph model used by the web app:

- `user -> anime` edges weighted by normalized score.
- `anime <-> anime` edges formed from all co-rated anime pairs.
- Existing anime-pair edge weights are updated as `(existing + new_pair_score) / 2`.

## Run

1. Install Rust (stable) and cargo.
2. From `desktop/`, run:

```bash
cargo run
```
