# 0027 — Verified versioned release installation

**Status:** M8.2 implementation contract fixed on 2026-09-28 before changing the downloader or browser loader. This authorizes invented fixtures and mocked transport only. Provider source/use, publication, model promotion, and deployment remain held by decisions 0001, 0002, and 0024.

## Local layout and activation

Install under `web/public/data/`: each complete plain-JSON release lives in `bundles/<bundleId>/`, named by the full lowercase SHA-256 bundle ID from decision 0026. `active.json` is an `active-release-bundle-v1` pointer containing the versioned tag, bundle ID, and exact manifest-byte SHA-256. The installer writes candidate files to a private sibling staging directory, verifies the full M8.1 contract and named prior bundle, moves the complete directory to its immutable name, then atomically renames a temporary active-pointer file over `active.json`. The prior bundle remains available. A failed download, parse, hash, compatibility, move, or pointer swap leaves the prior pointer intact; a complete unactivated directory may be verified and reused on retry. An exclusive local installation lock prevents concurrent pointer writers. Recursive cleanup is confined to the verified staging path.

The candidate's `lastKnownGood` must identify the currently active and fully verified bundle, including its exact manifest bytes. An empty store permits only explicit synthetic fixture bootstrap with a genesis manifest. The live GitHub transport and workflows must not bootstrap or fetch provider-derived data during routine development. A repeated active bundle is a verified no-op; a conflicting pre-existing bundle ID is a failure.

## Download and consumer checks

Fetch the manifest first, require its tag to match an explicit versioned request, and use only its fixed asset names. Accept one plain or gzip transport representation per declared artifact, with the manifest hash/length defined over **plain** bytes. Reject a model asset when `model` is null, or a missing declared model. Read response bodies incrementally and stop at 256 KiB for the manifest, 64 MiB compressed or 256 MiB plain per asset, and 512 MiB total plain JSON. Reject oversized declarations before asset download; cap gzip output during decompression. These are provisional engineering limits on the synthetic route, not a measured production sizing claim; changing them requires a recorded payload measurement and review. Reject missing, partial, corrupt, and unsupported files before activation. Do not download raw ratings or other undeclared assets.

The normal browser loader checks `./data/active.json` once per session. A 404 deliberately uses the existing legacy compact/JSON fallback. A present malformed or unavailable pointer fails closed. With a pointer, the browser validates the pinned manifest-byte hash, reads all graph/explorer/model files only from its versioned directory, checks each exact plain-byte hash and length before parsing, and never falls back to a legacy file on bundle failure. The identity catalog is verified at installation; the browser does not need it for current ranking. The versioned path and per-file check prevent a stale cache from silently mixing releases. The existing gzip legacy loader stays intact; M7.6 owns its broader compressed-delivery behavior.

## Scope boundary

This makes synthetic installation and local normal-mode consumption reproducible. It does not publish a release, change the held `data-latest` workflows, approve provider rights, or prove a deployed Pages build uses an approved bundle. M8.3–M8.7 retain publication, promotion, hosted verification, diagnostics, and practiced rollback gates.
