# 0011 — Hybrid blend comparison protocol

**Status:** Complete local M4.6 decision; protocol committed as `620853f` before the synthetic comparison. This is a local ranking-engineering decision, not a production quality or provider-data approval.

## Question and held-out inputs

The current hybrid engine min–max scales each eligible graph/model list and discards zero-scaled items. In particular, it drops the minimum item even at a 0% or 100% blend endpoint. Compare that deployed baseline with weighted reciprocal-rank fusion (RRF) at the existing 50% model setting, then define and test every endpoint and missing-component case.

Use only `fixtures/synthetic-input.json` and its generated compact demo model. Deduplicate each invented user's ratings by anime ID with the last score winning and omit unknown catalog IDs. For each validation user, rebuild the v1 pair-preference graph from **the other users only**: center each training user's deduplicated scores on that user's training mean, aggregate with the repository's pair function, and keep the complete eight-title catalog. The validation user's ratings must not enter the graph, normalization, support, or candidate selection. The compact model's two-dimensional item vectors/biases are fixed authored fixture values; they are not retrained for a fold and were designed alongside the invented tastes, so this is still weak external evidence.

For each validation user, enumerate every positive holdout (native score at least 7/10) and every one- or three-rating subset of the remaining ratings. Convert supplied ratings with the browser's local ten-point preference mapper; skip a subset without Liked or Disliked evidence. The holdout and other unobserved ratings never enter the preference, seen/excluded, or fitting inputs. Both methods call the same browser graph/model scorers and `rankEligibleCandidates` with the same catalog and eligibility policy. The 1/3-rating slices overlap and must be reported separately. Keep the 50% weight, graph/model scores, RRF constant, and selection rule fixed before inspecting results.

## Alternative and selection rule

RRF ranks each eligible component by descending finite raw score. Equal raw scores share the competition rank; anime ID ascending breaks ordering ties. With `k = 60`, a candidate's fusion **rank points** are `1000 × (graphWeight / (k + graphRank) + modelWeight / (k + modelRank))`; a missing candidate contributes zero for that component. At 0% or 100%, use only the selected component's full eligible list, including its lowest-scored and negative-scored items. When an entire component is empty, give the available component 100% effective weight regardless of slider setting; when both are empty, return empty. Source ranks and anime ID make all ties deterministic. Do not use signed score magnitude as a probability or treat negative model values as absent. Filter watched/excluded/catalog/required-metadata candidates before assigning ranks. The UI must identify rank points as relative ordering evidence, never probability or calibrated preference.

Report case counts, Hit@3, mean reciprocal rank, and each held-out rank for baseline min–max and RRF separately in the one- and three-observation slices. Count a missing held-out result as a miss. RRF may replace min–max only if each slice has at least five cases, loses no more than one Hit@3 case in either slice, loses at most 0.05 mean reciprocal rank in either slice, and has p95 at most 50 ms for 200 warmed fusion calls over two deterministic 1,000-item scored lists on this host. It must also pass endpoint, missing-component, equal/negative-score, deterministic tie, eligibility, and browser-display tests. If this gate fails, leave M4.6 unchecked and retain the current deployed method while recording the failure; a later method decision needs a new, explicit protocol.

Even a passing local gate cannot establish a production ranking lift: the tiny invented users overlap across cases, the fixed model is hand-authored, and a real split-first evaluation on permitted inputs remains in M5.1–M5.7 before calibration, tuning, or promotion claims.

## Result and decision

`npm run eval:hybrid-blend:fixture` checked the generated fixture, rebuilt each validation user's graph from the other users, and asserted that changing the validation user's ratings does not change that graph. Each scored case used the browser graph/model scorers and eligibility path. The runtime RRF candidate IDs and scores matched an independently written reference on every case. The old min–max implementation remains an evaluation-only reference outside the web bundle. The host was Windows x64, Node 23.10.0, Intel i7-12700K.

| Observed ratings | Cases | Min–max Hit@3 / MRR | Rank fusion Hit@3 / MRR |
|---|---:|---:|---:|
| 1 | 29 | 20/29 = 0.690 / 0.449 | 20/29 = 0.690 / 0.481 |
| 3 | 7 | 5/7 = 0.714 / 0.560 | 5/7 = 0.714 / 0.588 |

The final local 1,000-item, 200-call p95 was 0.466 ms for min–max, 0.693 ms for the independent RRF reference, and 0.750 ms for deployed RRF. Timing varied across reruns and is not a production latency estimate. Both predeclared quality and deployed latency gates passed. Unit and browser tests passed for complete endpoints, an empty component at every slider setting, tied and negative scores, anime-ID tie breaking, required filters before ranks, and relative-rank labels. Browser cards state source ranks and qualitative title evidence; the numeric result is labelled **rank points**, not probability or a sum of raw source contributions.

**Decision:** Use weighted RRF in the browser, with `k = 60` and 1,000 rank-point scaling. A candidate absent from one nonempty component gets zero from that component. An entirely empty component transfers full effective weight to the other; the UI reports that condition. The 0%/100% endpoints preserve the full active component, including its bottom and negative-scored candidates. The two methods tied on Hit@3 in both slices, while RRF met the MRR and latency gates and removed the endpoint omission. This does not establish a production ranking lift: cases share a tiny invented catalog and the fixed item vectors were authored alongside the ratings. M5's permitted split-first evaluation remains required for later tuning, calibration, or promotion claims; M4.7 owns deeper explanation reconciliation.
