# 0042 — Maintain desktop as a local graph companion

**Status:** M9.1 product scope selected on 2026-10-05. This decision does not approve a desktop release, provider-derived data use, or the current Rust implementation.

## Context

The web application already owns recommendation ranking, eligibility, explanations, imports, profiles, and recovery. Rebuilding those rules in Rust would duplicate fast-changing behavior and require independent parity, privacy, and assisted-use evidence. The current Dioxus prototype instead reads legacy per-user ratings from relative paths, recomputes anime pairs with order-dependent averaging, and silently substitutes an embedded sample when a read or parse fails. Its README calls that graph equivalent to the web graph, which the implementation does not support. The existing `v*` desktop workflow can publish this prototype as an EXE without M9.5 packaging or M9.6 outside-checkout verification.

## Decision

The maintained desktop scope is an **optional local graph companion**, separate from the web recommendation client and lower priority than web release gates. Preserve Rust and Dioxus. Desktop may inspect bounded, signed anime-pair evidence from an explicitly selected, locally verified aggregate-only `graph-compact-v3` bundle with its `release-manifest-v1` identity. Show the selected dataset version, graph role, counts, truncation, and evidence limits. Treat the source label in a file as a claim, not proof of redistribution rights or owner approval.

Desktop will not implement a second recommendation engine, provider import, profile store, model trainer, cloud account, or web feature-parity promise under M9. The web app remains the supported recommendation path. A later scope expansion requires a new product decision, a duplication-cost review, and its own tests. No real per-user ratings, user-anime edges, or private history belong in the supported desktop load path.

Startup must distinguish **No data** from an explicitly entered **Demo** mode. M9.2 must provide explicit file selection and visible errors; an invalid selected bundle must never switch to sample data. V1/v2 or raw-rating files are not silently rebuilt into v3. The existing prototype remains unsupported until M9.2–M9.4 replace that path and verify the graph/list behavior with invented bundles and malformed inputs. M9.3 should consume the precomputed pair tuples rather than repair a duplicate Rust aggregator.

M9.5–M9.6 remain release gates: pin and check the Rust toolchain, dependencies, package contents, runtime prerequisites, and an outside-repository launch before any desktop publication. The tag-triggered EXE publisher is retired now because it has no such gate. Restore a publication route only through a separately reviewed workflow with explicit approval and a verified package. The ordinary read-only desktop build job may continue as an engineering check; its uploaded CI artifact is not a public desktop release.

## Consequences and verification boundary

This scope avoids maintaining two recommendation policies and keeps user-linked rows out of a future desktop bundle. It deliberately offers less than web parity. M9.1 is a recorded product decision, not evidence that the present binary obeys it. M9.2–M9.6 and the M9 exit gate stay open until the implementation, packaging, and outside-checkout behavior pass. Routine development uses invented v3 bundles and no live provider or private history. See `docs/PROGRESS.md` for the reviewed baseline and change evidence.
