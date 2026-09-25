# 0012 — Recommendation score explanations

**Status:** M4.7 local implementation and fresh-checkout fixture CI passed. This defines what the browser can attribute to its current scorers; it is not confidence calibration or a ranking-quality result.

## Explanation contract

Graph recommendations sum each retained **positive v1 pair-preference** edge from an explicitly Liked source, multiplied by that preference's importance and confidence. The graph score has no global mean, item bias, or further normalization. Count contributing edges separately from distinct source anime IDs. A legacy v1 graph may contain multiple edges for one source/candidate pair, so an explanation must retain and sum every edge. Pair preference is not item similarity.

The browser model averages signed source item vectors by the sum of absolute mapped preference weights. Its candidate score is the model's global mean plus candidate item bias plus the dot product with that averaged vector. Explain each actual mapped source title as its signed, weighted source/candidate dot product divided by that same denominator. Show how many distinct supplied signals mapped; unseen, unmapped, or merely Seen titles do not gain a score contribution. Browser `Float32Array` arithmetic can leave a small residual against double-precision per-source arithmetic. Show a separate float32 term only when it reaches the displayed precision; otherwise the display-rounding adjustment covers any difference visible after rounding.

Hybrid rank points come from the eligible graph and model **ranks**, with `1000 × effective weight ÷ (60 + rank)` per present component. A candidate absent from a component contributes zero; an entirely empty component yields full effective weight to the other. Explain source ranks, effective weights, both rank-point terms, and the underlying source score equations where provenance exists. Raw graph and model scores determine ranks; their magnitudes are never added to rank points. Count an observed title present in both components once. Rank points are relative and are not a probability.

The UI shows actual source titles and the distinct count, with a full score equation behind **How the score was calculated**. Each term rounds independently to the displayed three decimals for graph/model or two for fusion. An explicit `display rounding adjustment` term appears only when needed so that the displayed terms sum exactly to the displayed score without altering a real source term. Results without scorer provenance may offer named evidence only if clearly labelled qualitative; they must not invent a numeric decomposition. Displayed uncertainty states that no calibrated confidence interval or probability exists and calls out sparse evidence or sample/model/pool limits.

## Verification and limits

Invented hand-computed tests cover duplicate graph edges, distinct titles, normalized liked/disliked model terms, global mean, item bias, partial mapping, rank fusion, rounding, and exact displayed-unit reconciliation. Mocked and demo browser tests cover graph/model/hybrid cards, unchanged ranks and scores, three explicit signals, an absent hybrid component, and HTML escaping of a source title inside the equation. The full local fixture gate and CI evidence are recorded in `docs/PROGRESS.md`.

These equations explain the implemented browser score for a shown candidate. They do not prove causality outside the scorer, quantify confidence, establish model quality, or clear decision 0001's provider-use holds. M5 retains split-first evaluation on permitted inputs, and M4.8 retains diversity/prerequisite behavior.
