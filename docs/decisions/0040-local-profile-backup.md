# 0040 — Keep profile backup local, versioned, and recoverable

## Context

Recommendation state and named profiles use separate version 5 browser keys. Earlier version 1 and 4 sources and raw backups are deliberately retained. A portable backup must include the active state, imported history, candidate overrides, and named profiles without depending on a graph, catalog, or model release. A failed write between the two current keys could otherwise leave an import half applied.

## Decision

Use a strict `wasiw-profile-backup` JSON document at backup format version 1. It contains the complete version 5 active state and named profiles plus an export time. A local `.json` file is bounded to 8 MiB and parsed before any storage write. Unknown catalog IDs, source identities, episode progress, native scores, importance, and overrides remain intact. Unknown fields or versions are refused rather than silently discarded. The file is prepared in the browser for a user download; no provider, proxy, telemetry, or server receives it. The interface warns that the file contains private watch history.

Every import shows a preview. **Merge** retains local preference and history values on matching identities, retains local named profiles on matching names, adds missing imported entries, extends candidate lists, and keeps the local engine and settings; exclusions retain their existing precedence. **Replace** adopts the imported active state and profiles and reports removed and changed local counts first. A preview becomes stale when current in-memory or stored data changes. Local file reads carry a generation guard.

The app displays a loading status and withholds controls until required local artifacts and event handlers are ready. An early local file selection previously could fire before its handler existed and be silently ignored; a delayed-graph browser regression covers this startup boundary. Required-data failures still reveal their fixed local diagnostic message.

Before writing the two version 5 keys, the adapter records the exact original bytes of every current, backup, and corrupt key the operation may touch in a local pending journal. On a failed or interrupted write it restores those bytes, including absent keys, and on startup it attempts the same recovery. If storage refuses rollback, reads use the journal's original current bytes and further writes are blocked until recovery succeeds. An unreadable journal may be archived only during an explicit previewed reset. Reset writes valid empty version 5 state and profiles, retaining older source keys and raw backups so an old migration does not silently repopulate the list. A readable backup can be activated through the repair control; unreadable current bytes are preserved in `.corrupt` storage when the write succeeds. Storage rejection is shown as failure, never as a saved import.

The backup contract covers the state fields that exist now. M6.5 must extend and version this contract, with round-trip and recovery tests, when it introduces a local watchlist and status/ratings state. No size evidence yet justifies IndexedDB; the bounded local JSON and existing storage remain the current path. The synthetic and mocked tests do not establish compatibility with a real provider export or permission to read one.
