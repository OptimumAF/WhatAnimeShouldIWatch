# 0003 — Provider request scheduling

**Status:** Implemented for existing request paths, 2026-09-24. This is a rate and failure policy, not permission to collect, retain, train on, or publish provider data. The holds in [decision 0001](0001-provider-data-permissions.md) remain in force.

## Scope and source limits

| Source | Local policy | Evidence and boundary |
|---|---|---|
| Jikan | At most 3 starts per rolling second and 60 per rolling minute; at most 3 in flight in TypeScript. The Python content-feature builder uses the same rolling windows and runs sequentially. | [Jikan v4 OpenAPI description](https://raw.githubusercontent.com/jikan-me/jikan-rest/master/storage/api-docs/api-docs.json) lists 3/second and 60/minute and warns that upstream MAL can rate-limit Jikan too. |
| AniList | At most 30 starts per rolling minute, spaced by at least 2 seconds, with one in flight. A lower valid `X-RateLimit-Limit` header tightens the minute window; `X-RateLimit-Remaining: 0` and `X-RateLimit-Reset` block subsequent starts. | [AniList rate-limiting guide](https://docs.anilist.co/guide/rate-limiting) currently warns that its normal 90/minute quota is temporarily 30/minute and documents these headers, `Retry-After`, and a separate burst limiter. The 2-second spacing is this project's conservative choice for the unspecified burst limit. |
| MAL site `load.json` | One start per rolling second in each runtime, with one in flight; a crawler `--delay-ms` may only increase the interval. | Decision 0001 found no documented application quota or permission for this site route. The floor limits local pressure; it does not make collection authorized or guarantee avoidance of a server-side limit. |

The browser adapter shares one TypeScript scheduler among AniList and MAL imports and Jikan metadata and seasonal reads. The TypeScript crawler shares one scheduler between MAL list pages and Jikan discovery calls. The Python feature builder has a separate Jikan scheduler because it is a separate process. Provider buckets are independent, but every retry consumes a fresh slot in its provider bucket. Local artifact reads and GitHub release-asset transfer are outside these provider buckets; M8.2 owns release download policy.

These are **per runtime** limits: one browser tab, one TypeScript crawler process, or one Python builder process. They cannot enforce a quota shared by unrelated tabs, processes, other clients on an IP address, or a future server fleet. A deployment that adds parallel workers needs a shared coordinator and a renewed provider review before collection is allowed. No provider endpoints were contacted to test the policy.

## Failure and cancellation rules

- Only 408, 425, 429, and 5xx responses and transport failures retry. 400, 403, 404, and 405 stop immediately; a 404 remains an unavailable/missing result in existing callers. MAL's previous retry of 405 was removed because it means the route rejected the method.
- `Retry-After` seconds or HTTP dates establish a provider cooldown. A longer instruction than a request's retry budget is not shortened to force a retry; that operation fails, and later calls still observe the cooldown. AniList's reset header can extend its cooldown.
- Each retry uses capped exponential backoff plus jitter. Browser requests have a 10-second attempt timeout and a 30–45-second request budget; TypeScript crawler calls can use up to 6 attempts and a 120-second budget. Python Jikan calls have a 30-second `urlopen` timeout, capped retries, and a 120-second budget. The existing crawler delay flags can raise, but never lower, the policy floor.
- Browser-provided abort signals cancel queued, waiting, backoff, and in-flight requests through the scheduler. TypeScript crawler SIGINT/SIGTERM handling propagates abort signals and closes its database in `finally`. Python SIGINT/SIGTERM stops before the next call or during pacing/backoff; a blocking `urllib` call can take up to its 30-second timeout to return. This limitation is explicit until a future transport supports interrupting an in-flight Python request.

## Verification and maintenance

The synthetic scheduler tests inject a clock, random source, response headers, and transport. They cover shared Jikan windows across metadata/seasonal and crawler calls, AniList reset headers, provider isolation, `Retry-After`, retry budgets, queued cancellation, in-flight timeout, and crawler cancellation. Python tests inject an opener and clock for the same Jikan windows, backoff, nonretry statuses, and cancellation during waits. Browser journeys continue to use intercepted provider responses. Recheck both source limits and the permission decision before changing policy defaults or enabling any held workflow.
