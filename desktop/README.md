# Desktop graph companion (Rust + Dioxus)

[Decision 0042](../docs/decisions/0042-desktop-companion-scope.md) keeps this as a small local graph companion. The web app owns recommendations and imports. Desktop does not contact a provider or send local files to a service.

The app opens in **No data**. Choose **Open invented demo** to inspect the embedded synthetic v3 graph, or **Select local manifest** to choose a `release-manifest-v1` JSON file. The selected directory must also contain its declared `graph.compact.json` recommendation graph. The loader checks the manifest structure, v3 aggregate-only graph fields, graph byte length and SHA-256, dataset identity, graph ID declaration, and counts. A failed selection shows an error and clears the old graph; it never opens the demo automatically. The graph shows signed pair-preference weights and co-rater support, with a labeled 300-title/1,400-pair overview limit. These weights are not similarity scores or recommendations.

The native picker is asynchronous. The graph read, parse, and bounded overview preparation run on a background worker; Clear, Demo, or a newer selection invalidates older work. The loader reads in 64-KiB chunks and limits the selected manifest to 256 KiB and recommendation graph to 256 MiB. The overview scans pair evidence with a bounded heap, keeps at most 300 titles and 1,400 pairs, and shortens long visible labels. It does not rebuild pair weights, retain the full graph in reactive state, or verify the rest of the bundle.

For reproducible invented data from the repository root:

```bash
npm run desktop:fixture:check
cargo fmt --manifest-path desktop/Cargo.toml -- --check
cargo clippy --locked --manifest-path desktop/Cargo.toml --all-targets -- -D warnings
cargo test --locked --manifest-path desktop/Cargo.toml
cargo run --locked --manifest-path desktop/Cargo.toml
```

In the file picker, select `desktop/fixtures/release-manifest.json` to exercise the local-file state. The embedded demo and this file have the same invented pair evidence but distinct status labels. Regenerate the checked-in fixture with `npm run desktop:fixture` after a reviewed change to `fixtures/synthetic-input.json` or its graph contract.

For a larger local shape probe, run `npm run desktop:stress:fixture`; it creates a TypeScript-verified invented v3 bundle under ignored `desktop/target/` with 12,000 titles and 60,000 pairs by default. Run `node --import tsx desktop/fixtures/generate-stress.mts --anime 50000 --pairs 300000 --title-chars 1500` for a roughly 95-MB graph. Its `invented-shape-probe` dataset digest is an internal shape label, not a raw-rating snapshot or permission evidence. Select the printed manifest path in the desktop picker. Use synthetic output only.

This local graph check does not verify the other manifest assets, a named predecessor, source rights, or a release approval. M9.5–M9.6 must check packaging and an outside-repository launch. There is no supported desktop release yet, and private or provider-derived data must not be used without the recorded permissions.
