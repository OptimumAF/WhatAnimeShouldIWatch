# Desktop graph companion (Rust + Dioxus)

[Decision 0042](../docs/decisions/0042-desktop-companion-scope.md) selects a small, optional graph companion. The web app remains the recommendation client. Desktop data selection, v3 bundle loading, bounded exploration, packaging, and outside-checkout verification are still M9.2–M9.6 work; there is no supported desktop release yet.

The current Rust binary is a **legacy prototype**. It reads relative `data/anonymized-ratings.json` paths, recomputes pairs with order-dependent averaging, and silently uses an embedded sample if a read or parse fails. That behavior is not the approved v3 graph contract and must not be used to inspect private or production ratings. Do not describe its display as a verified release dataset.

For a synthetic-only development baseline from the repository root:

```bash
cargo check --locked --manifest-path desktop/Cargo.toml
cargo test --locked --manifest-path desktop/Cargo.toml
```

The current test target has no Rust cases. These commands check compilation; they do not verify data loading, graph semantics, UI behavior, package dependencies, or a clean external launch. Do not publish or distribute its EXE until M9.5–M9.6 pass.
