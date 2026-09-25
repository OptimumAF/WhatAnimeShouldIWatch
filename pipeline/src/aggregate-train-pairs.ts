/** Local, stdin-only bridge from the validated raw-split fitter to the existing pair core. */
import {
  aggregateAnimePairs,
  DEFAULT_MAX_CANDIDATE_PAIRS,
  DEFAULT_MAX_PAIR_VISITS,
  DEFAULT_PAIR_SELECTION_SEED,
  type PairUser,
} from "./core/pair-aggregation.js";

const chunks: Buffer[] = [];
for await (const chunk of process.stdin) chunks.push(Buffer.from(chunk));
const input: unknown = JSON.parse(Buffer.concat(chunks).toString("utf8"));
if (!input || typeof input !== "object" || Array.isArray(input)) {
  throw new Error("Train-pair input must be an object.");
}
const record = input as Record<string, unknown>;
if (record.format !== "train-centered-pairs-v1" || !Array.isArray(record.users)) {
  throw new Error("Train-pair input requires format and users.");
}
// The core validates user IDs, unique anime IDs, numeric values, and work budgets.
const result = aggregateAnimePairs(record.users as PairUser[], 0, 0);
process.stdout.write(JSON.stringify({
  format: "train-pairs-v1",
  config: {
    maxRatingsPerUser: 0,
    maxAnimeAnimeEdges: 0,
    maxPairVisits: DEFAULT_MAX_PAIR_VISITS,
    maxCandidatePairs: DEFAULT_MAX_CANDIDATE_PAIRS,
    minSupport: 1,
    maxNeighborsPerAnime: 0,
    selectionSeed: DEFAULT_PAIR_SELECTION_SEED,
  },
  pairs: [...result.pairs.entries()].map(([key, value]) => {
    const [low, high] = key.split(":").map(Number);
    return [low, high, value.weight, value.support];
  }),
  stats: result.stats,
}));
