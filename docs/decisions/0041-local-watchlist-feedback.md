# 0041 — Keep watch plans separate from imported history and use ratings only for local feedback

## Context

Imported history already retains provider status, progress, native score, and source identity. The recommendation page had no independent shortlist or way to update a title's local watch status and rating. Reusing imported history for a user-edited watchlist would let a later import overwrite local choices. M6.5 also requires the portable backup to cover the new state.

## Decision

Add an optional `watchlist` array to the existing version 5 recommendation state and named profiles. Absence in earlier version 5 or migrated v1/v4 state becomes an empty list without rewriting untouched legacy source bytes. Each entry uses a stable positive anime ID, a retained title snapshot, one of plan-to-watch, watching, completed, on-hold, or dropped, and an optional explicit integer rating from 1–10. Validate unique IDs, bounded fields, and exact entry keys. Keep catalog-missing entries visible and editable after a data/model release changes.

All five states exclude the title from new recommendation candidates; a planned title is a saved choice, not watched evidence. Watching, completed, on-hold, and dropped count as watched for franchise prerequisite display. An explicit local plan-to-watch status takes precedence over stale imported or selected watched evidence for that prerequisite check. Status changes and unrated titles never create preference evidence. For local browser scoring only, a rated nonplanned title maps 1–4 to Disliked, 5–6 to Seen, and 7–10 to Liked using the existing score-confidence curve. An explicitly selected manual preference takes precedence; a local rating takes precedence over an imported or legacy preference. A planned title supplies no ranking preference even if a saved preference or rating exists. No watchlist entry is sent to a trainer, provider, proxy, or telemetry service by this feature.

Export backup format version 2 with a required watchlist in the active state and each named profile. Strict version 1 backups remain readable and upgrade to empty watchlists. Merge keeps local watchlist entries on matching anime IDs and adds missing imported IDs; replace previews changed and removed watchlist counts. Reset, stale-preview checks, and the existing exact-key recovery journal cover the extended version 5 state. This extends [decision 0040](0040-local-profile-backup.md); its version 1 files remain supported, while new downloads use version 2.

The manual add control requires an exact catalog title or anime ID, and recommendation cards can save a title directly. Ambiguous and partial names are not silently resolved. This local path does not resolve M6.2's broader alias and disambiguation work. Synthetic tests establish storage, ranking, and UI behavior, not production recommendation quality or permission to train on a person's history.
