import assert from "node:assert/strict";
import { selectFranchiseDiverseRecommendations } from "../src/franchise-diversity.ts";
import { franchiseGains, franchiseIdentity, franchiseMetadata, franchiseResults,
  franchiseTitles } from "./franchise-diversity-fixture.ts";

const topK = 3;
const ids = (items: readonly { anime: { animeId: number } }[]): number[] =>
  items.map((item) => item.anime.animeId);
const dcg = (ordered: readonly number[]): number => ordered.slice(0, topK).reduce((sum, id, index) =>
  sum + (franchiseGains.get(id) ?? 0) / Math.log2(index + 2), 0);
const ideal = dcg([...franchiseGains.keys()].sort((a, b) =>
  (franchiseGains.get(b) ?? 0) - (franchiseGains.get(a) ?? 0)));
const coverage = (ordered: readonly number[]): number => new Set(ordered.slice(0, topK)
  .map((id) => franchiseIdentity.get(id))).size;
const baseline = ids(franchiseResults);
const defaultSelection = selectFranchiseDiverseRecommendations(
  franchiseResults, franchiseMetadata, new Set([200]), false, franchiseTitles);
const varied = ids(defaultSelection.recommendations);
const allowed = selectFranchiseDiverseRecommendations(
  franchiseResults, franchiseMetadata, new Set([200]), true, franchiseTitles);
const baselineNdcg = dcg(baseline) / ideal;
const variedNdcg = dcg(varied) / ideal;
const relevanceLoss = baselineNdcg - variedNdcg;
const unwatched = selectFranchiseDiverseRecommendations(
  franchiseResults, franchiseMetadata, new Set(), false, franchiseTitles);

assert.deepEqual(ids(allowed.recommendations), baseline);
assert.deepEqual(allowed.recommendations.map((item) => item.score),
  franchiseResults.map((item) => item.score));
assert.ok(!ids(unwatched.recommendations).includes(201));
assert.ok(coverage(varied) >= coverage(baseline) + 1);
assert.ok(relevanceLoss <= 0.10);

process.stdout.write(JSON.stringify({ protocol: "docs/decisions/0013-franchise-diversity.md",
  topK, baseline: { ids: baseline.slice(0, topK), gains: baseline.slice(0, topK)
    .map((id) => franchiseGains.get(id)), distinctFranchises: coverage(baseline), ndcg: baselineNdcg },
  preferVariety: { ids: varied.slice(0, topK), gains: varied.slice(0, topK)
    .map((id) => franchiseGains.get(id)), distinctFranchises: coverage(varied), ndcg: variedNdcg,
    knownPrequelHidden: defaultSelection.knownPrequelHidden,
    repeatedFranchiseHidden: defaultSelection.repeatedFranchiseHidden },
  relevanceLoss, allowRelatedRestoresEligibleOrder: true,
  unwatchedPrequelHidesKnownSequel: true, localGate: true }, null, 2) + "\n");
