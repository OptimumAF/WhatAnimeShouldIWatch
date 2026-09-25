# 0004 — Collection schema and provenance

**Status:** Implemented for synthetic development on 2026-09-24. Provider collection and downstream use remain held by [decision 0001](0001-provider-data-permissions.md).

## Schema migration

`openDatabase` applies SQLite `PRAGMA user_version` migrations in one transaction. Version 1 contains `users`, `anime`, and `ratings`; version 2 adds the M2.5 per-user checkpoint tables; version 3 adds the provenance fields, `import_runs`, and `provider_anime_ids`. An unversioned database may contain only version 1 tables or also the M2.5 tables. `CREATE TABLE IF NOT EXISTS` and guarded column additions preserve both layouts. Required columns and foreign keys are checked; a malformed or newer unsupported version is rejected without a partial migration.

Legacy ratings retain their raw and normalized values. Their source provider, source item ID, fetch time, provider update time, and fetch run stay `NULL`; `normalized_field_version = 0` means the saved formula has not been established by this migration. Existing staged pages also keep unknown provenance. No historical timestamp or provider identity is inferred during upgrade.

## New collection records

New MAL site-route ratings record `source_provider = mal`, the specific `load.json` route, the response anime ID as `source_anime_id`, the local page-response time as `fetched_at`, the fetching run ID, and `normalized_field_version = 1` for `raw score − mean of that user's scored snapshot`. `provider_anime_ids` records the mapping to the internal anime ID. An ordinary `upsertRating` without source evidence clears old provenance and sets the normalization version back to 0; explicit recomputation sets the version to 1.

Each invocation writes an `import_runs` row with an anonymized user ID, source route, start/end offsets, page and entry counts, start/finish times, and a terminal outcome. A run superseded by a later attempt becomes `interrupted`; its delayed result cannot commit. Pages from an earlier paused run retain their original fetch run ID and response time if a later run completes the snapshot. A database error leaves the run `running` until a later attempt records the interruption, while the rating replacement transaction rolls back.

`fetched_at`, `started_at`, `finished_at`, and `last_completed_at` are local pipeline times. They are **not** viewing, rating, or provider update times. `provider_updated_at` is populated only when a source adapter supplies an explicit, validated, provider-documented per-entry update time through `VerifiedProviderPage`. The current MAL site adapter supplies no verified update-time mapping, so its value remains `NULL` even if an unrecognized response field looks like a timestamp. A dataset with legacy or mixed source rows uses `source = mixed-or-unverified` instead of attributing every row to the MAL route.

The database stores the existing salted user key, not a raw provider username. That key and its associated ratings remain restricted data. These schema fields do not clear collection, retention, training, redistribution, or deployment permissions. The current offset-paged site route also has no source snapshot version, so concurrent changes at the provider can still yield a mixed response without duplicate IDs; M2.5's terminal-page validation cannot prove a frozen provider snapshot.
