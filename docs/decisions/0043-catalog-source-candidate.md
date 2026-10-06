# 0043 — Wikidata structured-data catalog candidate

**Status:** Proposed M2.7 source/use route, 2026-10-05. No source collection, cache, publication, provider-use approval, or release is authorized by this proposal.

## Why this candidate

The current immutable bundle's `anime-catalog-v1` contains only ID/title pairs. The browser gets filter and discovery metadata from its synthetic demo catalog or bounded optional Jikan requests. This cannot meet M2.7's bundled genres, year, format, episode/runtime, classification, aliases, and relationship requirement in normal mode. AniList media data remains within its unresolved competing-service restriction, while Jikan does not grant rights to MAL-derived content ([decision 0001](0001-provider-data-permissions.md)).

[Wikidata's structured data is published under CC0](https://www.wikidata.org/wiki/Wikidata:Licensing), and its [data-access guide](https://www.wikidata.org/wiki/Help:Linked_Data_Interface) describes external reuse. A Wikidata-only metadata snapshot is therefore a candidate to review without querying MAL, AniList, or Jikan. This is a source-license finding, not a conclusion that coverage, upstream statement quality, this product's source/use review, or its existing graph and model are cleared. The owner must approve the exact collection and use before a real snapshot is fetched or retained.

## Proposed bounded source and fields

Take only structured entity statements, labels, and aliases for a predeclared, bounded set of anime identities. Use [MyAnimeList anime ID (P4086)](https://www.wikidata.org/wiki/Property:P4086) as an external **identifier** compatible with current numeric anime IDs; do not follow its formatter URL or read MAL content. Require one unambiguous Wikidata entity per ID, report duplicates or season-part qualifiers for review, and never silently merge them. Do not request user entities, lists, ratings, reviews, or account data.

| Catalog field | Candidate Wikidata statement | Rule before any real export |
|---|---|---|
| Canonical title and aliases | Entity labels/aliases | Declare a language order and length/count bounds; retain source entity ID and do not treat an alias as a unique identity. |
| Genres | [P136](https://www.wikidata.org/wiki/Property:P136) | Map only explicit genre entities with a reviewed label table; missing or unmapped values stay unknown. |
| Year | [P577](https://www.wikidata.org/wiki/Property:P577) or reviewed start-time rule | Use a stated release/air date, not fetch time; conflicting dates stay unresolved. |
| Media format | P31 instance-of | Map only reviewed TV/film/OVA/ONA/special types; ambiguous or unrecognized values stay unknown. |
| Episode count and runtime | [P1113](https://www.wikidata.org/wiki/Property:P1113), [P2047](https://www.wikidata.org/wiki/Property:P2047) | Check integer count, duration unit, scope, and dated qualifiers. Do not infer runtime from episode count. |
| Content classification | Jurisdiction-specific rating statements, such as [EIRIN P2756](https://www.wikidata.org/wiki/Property:P2756) for films | Store rating system and jurisdiction with the value; no single universal classification is established. Unknown must remain explicit, especially for TV titles. |
| Franchise relationships | [P179](https://www.wikidata.org/wiki/Property:P179), [P155](https://www.wikidata.org/wiki/Property:P155), [P156](https://www.wikidata.org/wiki/Property:P156) | Only an explicit, directed immediate predecessor/successor mapped to another catalog ID may become a prequel/sequel hint. A shared series alone is not a viewing prerequisite; missing or one-sided statements do not prove safety. Follow decision 0013. |
| Community score | No proposed Wikidata field | Explicitly unknown. A required score filter excludes unknown values; community-score exploration must report its coverage rather than manufacture a score. |
| Cover image | Excluded | No image bytes or P18 URL in this catalog. [Commons files have individual licenses](https://commons.wikimedia.org/wiki/Commons:Reusing_content_outside_Wikimedia/licenses); a later image route needs its own file-level review. |

The field mapping is a proposal, not evidence that the statements exist for enough of this product's IDs. A real candidate must report exact universe size and per-field known/unknown/conflicting counts, mapping failures, alias ambiguity, and directed-relation coverage **before** a field is represented as complete. The owner must set useful coverage criteria before interpreting that report. Required filters continue to exclude unknown metadata; empty relationships never establish a safe starting title. No source-derived values or example anime are committed for this proposal.

## Access and publication boundary

The proposed first feasibility run would examine **at most 100 owner-supplied anime IDs**, with a recorded, non-personal selection rule and a bounded exact P4086 lookup/entity read. It would save the candidate only in a restricted local path and report aggregate field coverage, conflicts, and missing mappings; it would not create a web/release asset. Without an approved ID list and method, there is no real run. Before any real request or dump processing, record that scope, access method, purpose, retention period, public fields, notice/attribution choice, correction and removal process, and owner. The [Wikidata Query Service manual](https://www.mediawiki.org/wiki/Wikidata_Query_Service/User_Manual) documents a 60-second query deadline, client processing/error limits, HTTP 429 and `Retry-After`, and the need for an identifying user agent; the [Wikimedia user-agent policy](https://foundation.wikimedia.org/wiki/Policy:Wikimedia_Foundation_User-Agent_Policy/en) gives the contact format. Do not run an unbounded query or treat a successful request as source/use approval.

Keep the metadata source identity and snapshot digest distinct from the ratings-derived `datasetSha256`. The existing `release-manifest-v1` and five-asset publication package bind a two-column identity catalog, not this metadata. A new versioned metadata asset/manifest and browser loader need strict byte/field checks, synthetic unit/browser tests, and a deliberate v1 compatibility path before deployment. Do not smuggle extra fields into `anime-catalog-v1`, call Jikan to fill gaps, or change the current approval registries to test this proposal. Graph, model, provider-derived publication, Pages, and existing `data-latest` remediation remain separate decisions.

## Owner decision required

1. Approve or reject **a bounded Wikidata structured-data feasibility snapshot** for local coverage review, with at most 100 owner-supplied, non-personal anime IDs, an exact selection rule, and an access method. This does not approve public redistribution.
2. After coverage and statement-quality review, decide whether Wikidata alone can support M2.7's filters and relationship behavior, which fields may enter a public catalog, the minimum usable coverage, and the correction/removal process. If it cannot, choose another source with separately reviewed collection and redistribution rights.
3. Approve any public catalog package and deployment separately under decisions 0001, 0002, and 0028–0032. A permitted catalog does not clear the graph/model lineage or the existing public release.

Until the first decision is recorded, continue only with invented fixtures, mocked transport, source-neutral contracts and browser integration, and read-only documentation review. M2.7 and the M2 exit gate remain unchecked.

## Synthetic contract preparation (2026-10-05)

`web/src/artifacts.ts` parses a separate `anime-metadata-catalog-v1` **candidate** with sorted unique anime IDs, unique source item IDs, bounded titles/aliases/genres/relationships, explicit nulls for unknown fields, jurisdiction-specific classification, and no image or user-row fields. Its source name, UTC time, and SHA-256 are declared provenance only; parsing does not verify an approved source snapshot. A pure coverage function counts known and usable fields, missing IDs, directed relationship evidence, and directed targets outside the examined universe without printing titles or IDs. Invented unit and browser tests exercise valid unknowns and reject hidden fields. The candidate is outside `release-manifest-v1` and public packages.

## Browser-only bundle candidate (2026-10-05)

The browser now accepts an invented or mocked `release-manifest-v2` with a separately hash-bound `catalog.metadata.json`. It verifies the manifest and asset bytes, the v1 identity catalog's item-map digest and exact graph mapping, the metadata source-digest declaration and item count, and every metadata ID against that identity catalog. It projects year, format, genres, aliases, score, episode/runtime, classification, and directed relations to browser metadata. Unknown fields stay explicit in the parsed asset; a missing catalog item is unavailable and does not trigger Jikan detail enrichment. Card text is escaped. A present bad asset fails with its file and field or hash error instead of falling back. Existing v1 reads remain unchanged.

This verifies a technical browser seam only. The declared source digest is not a proof that approved source bytes exist or match it. No source entity was fetched, and the exact public asset allowlists still accept only v1. The local producer and public-byte verifier below do not change that hold. Before a real release, measure source coverage, decide permitted fields/use, install a versioned bundle with invented assets, and review the publication route separately. The owner decisions above and M2.7 remain open.

## Local producer and public-byte check (2026-10-05)

`pipeline/src/metadata-release-bundle.ts` builds a data-only v2 manifest from an exact aggregate v3 graph/explorer/identity catalog and strict metadata file. Building requires separate local source bytes whose SHA-256 equals the snapshot's declared source digest; those bytes do not enter the five-file bundle. The writer rejects undeclared files and overwrites, and fixture genesis requires a `data-vsynthetic-` tag. The public-byte verifier recomputes the manifest, bundle ID, every declared asset hash, and the named v1 or v2 data-only predecessor's public bytes. It cannot recheck the private source stream or grant rights. Invented tests reject changed source bytes, hidden fields, unknown IDs, a graph with user rows, changed public bytes, extra files, and predecessor drift.

This is an offline candidate, not an installer or publisher. Existing v1 package inventories, installer, release workflows, and approval registries are unchanged and still reject v2. The next synthetic technical check is installation from a mocked exact inventory while keeping publication disabled. Any real source collection still requires the owner decision above; any real release requires independently reviewed coverage, source/use, publication, and deployment evidence.
