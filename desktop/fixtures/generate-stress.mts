/** Invented graph shape for local desktop responsiveness checks; no real ratings are read. */
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { aggregateRecommendationGraphId } from "../../pipeline/src/core/graph-contract.ts";
import { buildExplorerGraph } from "../../pipeline/src/core/explorer-graph.ts";
import { buildReleaseManifest, verifyReleaseBundle } from "../../pipeline/src/release-manifest.ts";
import type { CompactGraphDataV3 } from "../../pipeline/src/types.ts";

const directory = path.dirname(fileURLToPath(import.meta.url));
function count(flag: string, fallback: number, maximum: number): number {
  const index = process.argv.indexOf(flag);
  if (index < 0) return fallback;
  const value = Number(process.argv[index + 1]);
  if (!Number.isSafeInteger(value) || value < 1 || value > maximum) {
    throw new Error(`${flag} must be an integer from 1 to ${maximum}.`);
  }
  return value;
}
const animeCount = count("--anime", 12_000, 50_000);
const pairCount = count("--pairs", 60_000, 300_000);
const titleChars = count("--title-chars", 40, 2_000);
if (pairCount > animeCount * 8 || animeCount < 12) {
  throw new Error("The invented shape requires at least 12 anime and at most 8 pairs per anime.");
}
if (animeCount * titleChars > 100_000_000) {
  throw new Error("The invented title payload exceeds the local 100-million-character bound.");
}
const output = path.resolve(directory, `../target/m9-scale-${animeCount}-${pairCount}-${titleChars}`);
const seed = JSON.parse(fs.readFileSync(path.join(directory, "graph.compact.json"), "utf8")) as CompactGraphDataV3;
const anime: CompactGraphDataV3["anime"] = Array.from({ length: animeCount }, (_, index) =>
  [10_000_000 + index, `星の航路 ${index} · invented scale title 🌌`.padEnd(titleChars, ".")]);
const aa: CompactGraphDataV3["aa"] = Array.from({ length: pairCount }, (_, index) => {
  const left = index % animeCount;
  const right = (left + 1 + Math.floor(index / animeCount)) % animeCount;
  return [left, right, index % 3 === 0 ? -0.75 : 0.5, 1 + index % 5];
});
const dataset = {
  sha256: crypto.createHash("sha256").update(JSON.stringify(["invented-shape", animeCount, pairCount])).digest("hex"),
  scope: "anonymized-ratings-content-v1" as const,
  source: "invented-shape-probe",
};
const { graphId: _oldGraphId, ...seedWithoutId } = seed;
const withoutId: Omit<CompactGraphDataV3, "graphId"> = {
  ...seedWithoutId,
  dataset,
  config: {
    ...seed.config,
    maxPairVisits: Math.max(seed.config.maxPairVisits, pairCount * 4),
    maxPairCandidates: Math.max(seed.config.maxPairCandidates, pairCount * 2),
  },
  truncation: {
    inputRatings: animeCount * 10,
    selectedRatings: animeCount * 10,
    ratingsSkipped: 0,
    potentialPairVisits: pairCount * 2,
    pairVisits: pairCount * 2,
    pairVisitsSkipped: 0,
    candidatePairs: pairCount,
    eligiblePairs: pairCount,
    selectedPairs: pairCount,
    excludedBySupport: 0,
    excludedByNeighborLimit: 0,
    excludedByOutputLimit: 0,
  },
  anime,
  aa,
  animeCount,
  nodeCount: animeCount,
  edgeCount: pairCount,
};
const recommendation: CompactGraphDataV3 = {
  ...withoutId, graphId: aggregateRecommendationGraphId(withoutId),
};
const explorer = buildExplorerGraph(recommendation, Math.min(pairCount, 1_400), 0);
const catalog = { format: "anime-catalog-v1", datasetSha256: dataset.sha256, anime };
const json = (value: unknown): Buffer => Buffer.from(`${JSON.stringify(value, null, 2)}\n`);
const graphBytes = json(recommendation);
const explorerBytes = json(explorer);
const catalogBytes = json(catalog);
const manifest = buildReleaseManifest({
  neighborhood: graphBytes, explorer: explorerBytes, catalog: catalogBytes,
}, { tag: `data-vsynthetic-scale-${animeCount}-${pairCount}`, fixtureGenesis: true });
fs.mkdirSync(output, { recursive: true });
for (const [file, bytes] of [
  ["graph.compact.json", graphBytes],
  ["graph-explorer.compact.json", explorerBytes],
  ["catalog.identity.json", catalogBytes],
  ["release-manifest.json", json(manifest)],
] as const) fs.writeFileSync(path.join(output, file), bytes);
verifyReleaseBundle(output, undefined, true);
process.stdout.write(`Verified invented desktop shape: ${animeCount} titles, ${pairCount} pairs, ` +
  `${graphBytes.length} graph bytes; ${path.join(output, "release-manifest.json")}\n`);
