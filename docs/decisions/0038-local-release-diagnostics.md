# 0038 — Keep first-release diagnostics local and free of private inputs

**Status:** M8.6 synthetic and mocked implementation, 2026-09-28. This decision adds no external telemetry, provider use, release approval, or deployment permission.

## Version identity

The app reports the root package version and the source commit injected at build time. A local build says so rather than inventing a release revision. A verified active release reports its versioned data tag, graph format, and short bundle digest. Its optional model reports the declared format and short asset digest, plus whether it has actually loaded. A data-only release says the model was not included. Demo and legacy paths explicitly say synthetic or unversioned; they do not claim an immutable data or model identity. These are public artifact fields, not recommendation scores or quality claims.

## Error contract and privacy

The in-page diagnostics panel keeps one current fixed code and a fixed next action for data, model, import, storage, explorer, render, and seasonal failures. It reads no profile, username, watch list, raw score, provider response body, or imported file content. It writes no diagnostics to browser storage and sends none to a server. App and artifact labels are validated or reduced to short public digests before display, and all panel text uses `textContent`.

Provider-supplied error messages and request URLs are not echoed into import status. File names and arbitrary file-read exceptions are not echoed into local import status; only typed parser-authored messages with fixed field names and entry numbers retain their useful detail. Artifact-validation and loader-authored messages continue to identify the file and field, while arbitrary artifact transport exceptions receive fixed fallback text; the diagnostics panel records only a fixed code and action. Browser console messages no longer attach raw exception objects from these paths. This avoids turning an entered username, file name, provider error, or raw history into a diagnostic string. The deliberate direct username request to the selected provider, after the user initiates import, remains the existing separate provider path.

The first release uses this local panel. External telemetry would need a separate documented purpose, data inventory, consent and controls as applicable, and owner approval before implementation. Synthetic unit/browser tests inject invented private-looking strings to verify the display and error boundary; they do not establish provider permission, production data quality, or a deployed release.
