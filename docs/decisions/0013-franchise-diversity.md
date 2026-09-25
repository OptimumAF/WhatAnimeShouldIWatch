# 0013 — Known-franchise diversity and prerequisite protocol

**Status:** Protocol fixed before implementation or synthetic evaluation. M4.8 remains unchecked until the runtime behavior, user option, uncertainty copy, and fixture/browser gates pass.

## Inputs and limits

Use only the existing eligible ranking, its already requested candidate metadata, selected/seen history, and optional relationships in the synthetic catalog or the existing `/anime/{id}/full` response. Do not request more metadata, crawl related IDs, infer provider permission, or change graph/model scores. Jikan's own [maintainer description](https://github.com/jikan-me/jikan-rest/issues/461) says its relation list is scraped from the requested title and can be incomplete or asymmetric. Absence of a link can never establish that a title has no prerequisites.

Recognize anime-to-anime `Prequel`, `Sequel`, `Alternative Version`, `Side Story`, and `Spin-Off` links as evidence that entries share a known franchise. Only `Prequel` and `Sequel` establish a directed immediate predecessor: a `Prequel` entry precedes the requested title, while a `Sequel` entry follows it. Ignore manga and other relationship types for this selector. An exact normalized title match or a conservative `Season N`/`Part N` suffix may suggest a near duplicate for diversity only; it cannot establish viewing order. Preserve each candidate's original score and ordering within the remaining list.

Default to **Prefer variety**: withhold a candidate whose known immediate prequel is absent from seen/watched history, then show at most one candidate per known or title-suggested franchise. Keep unknown-relationship candidates eligible, make the limited coverage visible, and backfill from the existing eligible list without exceeding the existing metadata-request budget. **Allow related titles** returns the original eligible order, including known sequels, and shows known predecessor warnings. The option persists with recommendation state and named profiles. Seen, exclusion, Include Only, catalog, and required metadata filters still run first; the selector may only remove or reorder an eligible result, never insert one. Apply the same final-list rule to graph, model, hybrid, fallback, and discovery views. Unchecked or incomplete relationship data must be labelled as such on cards; never say that a title has no prerequisites.

## Predeclared invented evaluation

Use an eight-title, invented scored list with two related entries near the top, four unrelated entries, an unknown-relationship entry, and a title resembling a different franchise. Declare relevance gains independently of the selector. For the top three, compare the unmodified eligible order to **Prefer variety** by distinct known/suggested franchise count and NDCG@3. In the watched-prequel scenario, variety must increase distinct franchise count by at least one while losing no more than 0.10 NDCG@3. Report the actual positions and gains, not only a pass flag. Separately, an unwatched known prequel must suppress its sequel by default, while **Allow related titles** must exactly restore the input order and scores. Test missing, empty, malformed, one-sided, and transitive relationship evidence; repeated names, false title cues, history mapping, filter precedence, persistence, and HTML escaping. Do not hardcode genre quotas or use the invented result as a production ranking-quality claim.

## Result

Pending implementation and measurement.
