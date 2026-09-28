/** Offline M5.5 evaluation through the browser's preference, model, and eligibility code. */
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseCompactModel } from "../src/artifacts.ts";
import type { AnimeMetadata, GraphData } from "../src/artifacts.ts";
import type { ModelRecommendationIndex, RecommendationResult } from "../src/domain.ts";
import { selectFranchiseDiverseRecommendations } from "../src/franchise-diversity.ts";
import type { HistoryEntry } from "../src/import-history.ts";
import { preferenceFromHistory } from "../src/preferences.ts";
import type { AnimePreference } from "../src/preferences.ts";
import {
  buildCatalogCoverageRecommendations, buildGraphRecommendationsForPreferences,
  buildModelRecommendationsForPreferences, buildRecommendationIndex,
  createCandidateEligibilityPolicy, rankEligibleCandidates,
} from "../src/recommendations.ts";
import type { RecommendationFilters } from "../src/recommendations.ts";

const ROOT = fileURLToPath(new URL("../../", import.meta.url));
const COUNTS = [1, 3, 5, 10] as const;
type RawRating = { animeId: number; rawScore: number };
type CandidateMetadata = { animeId: number; year: number; score: number; genres: string[] };
type EvalUser = {
  userId: string; observed: RawRating[]; validation: RawRating[];
  historySeen: number[]; exclude: number[]; includeOnly: number[];
  filters: RecommendationFilters;
};
export type EvalFixture = {
  seed: number; topK: number; positiveRawScoreMin: number;
  candidateMetadata: CandidateMetadata[]; users: EvalUser[];
};
type EvalBundle = {
  candidateId: string; trainSha256: string; fitSha256: string; modelSha256: string;
  metadataSha256: string; fitUserCount: number; trainRowCount: number;
  warmValidation: { eligibleUsers: number; positiveLabels: number; hitsAtK: number;
    ndcgAtK: number; recallAtK: number };
  catalog: { animeId: number; title: string }[];
  positivePairs: { leftAnimeId: number; rightAnimeId: number; weight: number; support: number }[];
  model: ReturnType<typeof parseCompactModel>;
};
export type UserResult = {
  userNumber: number; suppliedCount: number; vectorSignalCount: number;
  seenCount: number; mappedSignalCount: number;
  candidateCount: number; eligibleCandidateCount: number;
  positiveLabels: number; eligiblePositiveLabels: number; excludedPositiveLabels: number;
  excludedPositiveReasons: { watchedOrExcluded: number; includeOnly: number; metadataFilter: number };
  ranks: (number | null)[];
  ranked: { animeId: number; score: number }[];
  displayedEngine: "model" | "graph" | "coverage";
  displayedCandidateCount: number; displayedRanks: (number | null)[];
  hitAtK: number; recallAtK: number; ndcgAtK: number; reciprocalRank: number;
};

function fail(location: string): never { throw new Error(`Invalid new-user evaluation ${location}.`); }
function record(value: unknown, location: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) fail(location);
  return value as Record<string, unknown>;
}
function fields(value: Record<string, unknown>, expected: readonly string[], location: string): void {
  if (Object.keys(value).sort().join("|") !== [...expected].sort().join("|")) fail(location + " fields");
}
function array(value: unknown, location: string): unknown[] {
  if (!Array.isArray(value)) fail(location);
  return value;
}
function integer(value: unknown, location: string, min = 0): number {
  if (!Number.isSafeInteger(value) || typeof value !== "number" || value < min) fail(location);
  return value;
}
function number(value: unknown, location: string, min: number, max: number): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < min || value > max) fail(location);
  return value;
}
function idList(value: unknown, location: string, catalogIds: ReadonlySet<number>): number[] {
  const ids = array(value, location).map((entry, i) => integer(entry, `${location}[${i}]`, 1));
  if (new Set(ids).size !== ids.length || ids.some((id) => !catalogIds.has(id))) fail(location);
  return ids;
}
function ratings(value: unknown, location: string, catalogIds: ReadonlySet<number>): RawRating[] {
  return array(value, location).map((entry, i) => {
    const item = record(entry, `${location}[${i}]`);
    fields(item, ["animeId", "rawScore"], `${location}[${i}]`);
    const animeId = integer(item.animeId, `${location}[${i}].animeId`, 1);
    if (!catalogIds.has(animeId)) fail(`${location}[${i}].animeId absent from train catalog`);
    return { animeId, rawScore: number(item.rawScore, `${location}[${i}].rawScore`, 1, 10) };
  });
}

/** The Python adapter validates the raw split and emits no user rows or user factors. */
export function loadEvalBundle(): EvalBundle {
  const output = execFileSync("python", [
    "ml/split_first_new_user_export.py",
    "--raw-ratings", "fixtures/synthetic-new-user-fit.json",
    "--split-manifest", "fixtures/synthetic-new-user-fit-manifest.json",
    "--metadata", "fixtures/synthetic-new-user-anime-metadata.json",
    "--candidates", "fixtures/synthetic-mf-candidates.json",
    "--candidate-id", "graph-two-epochs",
  ], { cwd: ROOT, encoding: "utf8", maxBuffer: 8 * 1024 * 1024 });
  return parseEvalBundle(JSON.parse(output));
}

export function parseEvalBundle(value: unknown): EvalBundle {
  const root = record(value, "bundle");
  fields(root, ["format", "candidateId", "trainSha256", "fitSha256", "modelSha256",
    "metadataSha256", "fitUserCount", "trainRowCount", "warmValidation",
    "catalog", "positivePairs", "model"], "bundle");
  if (root.format !== "split-first-new-user-bundle-v1" ||
      root.candidateId !== "graph-two-epochs") fail("bundle format/candidate");
  for (const key of ["trainSha256", "fitSha256", "modelSha256", "metadataSha256"]) {
    if (typeof root[key] !== "string" || !/^[a-f0-9]{64}$/.test(root[key])) fail(`bundle.${key}`);
  }
  const model = parseCompactModel(root.model, "split-first local model");
  const modelIds = new Set(model.animeIds);
  const catalog = array(root.catalog, "catalog").map((entry, i) => {
    const item = record(entry, `catalog[${i}]`);
    fields(item, ["animeId", "title"], `catalog[${i}]`);
    const animeId = integer(item.animeId, `catalog[${i}].animeId`, 1);
    if (!modelIds.has(animeId) || typeof item.title !== "string" || !item.title.trim()) fail(`catalog[${i}]`);
    return { animeId, title: item.title };
  });
  if (!catalog.length || new Set(catalog.map((item) => item.animeId)).size !== catalog.length) fail("catalog IDs");
  const catalogIds = new Set(catalog.map((item) => item.animeId));
  const positivePairs = array(root.positivePairs, "positivePairs").map((entry, i) => {
    const item = record(entry, `positivePairs[${i}]`);
    fields(item, ["leftAnimeId", "rightAnimeId", "weight", "support"], `positivePairs[${i}]`);
    const leftAnimeId = integer(item.leftAnimeId, `positivePairs[${i}].leftAnimeId`, 1);
    const rightAnimeId = integer(item.rightAnimeId, `positivePairs[${i}].rightAnimeId`, 1);
    if (!catalogIds.has(leftAnimeId) || !catalogIds.has(rightAnimeId) || leftAnimeId >= rightAnimeId) {
      fail(`positivePairs[${i}] catalog/order`);
    }
    return { leftAnimeId, rightAnimeId,
      weight: number(item.weight, `positivePairs[${i}].weight`, Number.MIN_VALUE, Infinity),
      support: integer(item.support, `positivePairs[${i}].support`, 1) };
  });
  if (new Set(positivePairs.map((item) => `${item.leftAnimeId}:${item.rightAnimeId}`)).size !== positivePairs.length) {
    fail("duplicate positivePairs");
  }
  const warm = record(root.warmValidation, "warmValidation");
  fields(warm, ["eligibleUsers", "positiveLabels", "hitsAtK", "ndcgAtK", "recallAtK"], "warmValidation");
  const warmValidation = { eligibleUsers: integer(warm.eligibleUsers, "warmValidation.eligibleUsers", 1),
    positiveLabels: integer(warm.positiveLabels, "warmValidation.positiveLabels", 1),
    hitsAtK: integer(warm.hitsAtK, "warmValidation.hitsAtK"),
    ndcgAtK: number(warm.ndcgAtK, "warmValidation.ndcgAtK", 0, 1),
    recallAtK: number(warm.recallAtK, "warmValidation.recallAtK", 0, 1) };
  return { candidateId: root.candidateId, trainSha256: root.trainSha256,
    fitSha256: root.fitSha256, modelSha256: root.modelSha256,
    metadataSha256: root.metadataSha256,
    fitUserCount: integer(root.fitUserCount, "fitUserCount", 1),
    trainRowCount: integer(root.trainRowCount, "trainRowCount", 1),
    warmValidation, catalog, positivePairs, model } as EvalBundle;
}

export function fitUserIds(value: unknown): Set<string> {
  const root = record(value, "fit snapshot");
  if (root.format !== "raw-interactions-v1" || root.timestampBasis !== "none") fail("fit snapshot format");
  const users = new Set<string>();
  for (const [i, entry] of array(root.interactions, "fit interactions").entries()) {
    const item = record(entry, `fit interactions[${i}]`);
    if (typeof item.userId !== "string" || !item.userId) fail(`fit interactions[${i}].userId`);
    users.add(item.userId);
  }
  return users;
}

export function parseEvalFixture(value: unknown, bundle: EvalBundle, fitUsers: ReadonlySet<string>): EvalFixture {
  const root = record(value, "fixture");
  fields(root, ["format", "seed", "positiveRawScoreMin", "topK", "candidateMetadata", "users"], "fixture");
  if (root.format !== "new-user-validation-fixture-v1" || root.seed !== 42 ||
      root.positiveRawScoreMin !== 7 || root.topK !== 10) fail("fixture protocol");
  const catalogIds = new Set(bundle.catalog.map((item) => item.animeId));
  const candidateMetadata = array(root.candidateMetadata, "candidateMetadata").map((entry, i) => {
    const item = record(entry, `candidateMetadata[${i}]`);
    fields(item, ["animeId", "year", "score", "genres"], `candidateMetadata[${i}]`);
    const animeId = integer(item.animeId, `candidateMetadata[${i}].animeId`, 1);
    const genres = array(item.genres, `candidateMetadata[${i}].genres`);
    if (!catalogIds.has(animeId) || !genres.length ||
        genres.some((genre) => typeof genre !== "string" || !genre.trim())) fail(`candidateMetadata[${i}]`);
    return { animeId, year: integer(item.year, `candidateMetadata[${i}].year`, 1),
      score: number(item.score, `candidateMetadata[${i}].score`, 0, 10),
      genres: genres as string[] };
  });
  if (candidateMetadata.length !== catalogIds.size ||
      new Set(candidateMetadata.map((item) => item.animeId)).size !== catalogIds.size) {
    fail("candidateMetadata catalog coverage");
  }
  const seenUsers = new Set<string>();
  const users = array(root.users, "users").map((entry, i) => {
    const item = record(entry, `users[${i}]`);
    fields(item, ["userId", "observed", "validation", "historySeen", "exclude",
      "includeOnly", "filters"], `users[${i}]`);
    const userId = item.userId;
    if (typeof userId !== "string" || !userId || fitUsers.has(userId) || seenUsers.has(userId)) {
      fail(`users[${i}].userId overlaps fit or evaluation cohort`);
    }
    seenUsers.add(userId);
    const observed = ratings(item.observed, `users[${i}].observed`, catalogIds);
    const validation = ratings(item.validation, `users[${i}].validation`, catalogIds);
    const historySeen = idList(item.historySeen, `users[${i}].historySeen`, catalogIds);
    const exclude = idList(item.exclude, `users[${i}].exclude`, catalogIds);
    const includeOnly = idList(item.includeOnly, `users[${i}].includeOnly`, catalogIds);
    const distinct = [...observed, ...validation].map((rating) => rating.animeId)
      .concat(historySeen, exclude);
    if (observed.length !== 10 || validation.length !== 2 ||
        new Set(distinct).size !== distinct.length ||
        !validation.some((rating) => rating.rawScore >= 7)) fail(`users[${i}] ratings/overlap`);
    const rawFilters = record(item.filters, `users[${i}].filters`);
    fields(rawFilters, ["genre", "minYear", "maxYear", "minScore"], `users[${i}].filters`);
    if (typeof rawFilters.genre !== "string" || rawFilters.genre.length > 100) fail(`users[${i}].filters.genre`);
    const year = (field: "minYear" | "maxYear") => rawFilters[field] === null ? null
      : integer(rawFilters[field], `users[${i}].filters.${field}`, 1);
    const filters: RecommendationFilters = { genre: rawFilters.genre, minYear: year("minYear"),
      maxYear: year("maxYear"),
      minScore: rawFilters.minScore === null ? null
        : number(rawFilters.minScore, `users[${i}].filters.minScore`, 0, 10) };
    return { userId, observed, validation, historySeen, exclude, includeOnly, filters };
  });
  if (!users.length) fail("users empty");
  return { seed: 42, positiveRawScoreMin: 7, topK: 10, candidateMetadata, users };
}

/** Hash order is independent of scores and validation labels; slices are nested. */
export function orderedObserved(user: EvalUser, seed: number): RawRating[] {
  const key = (rating: RawRating) => createHash("sha256")
    .update("wasiw-new-user-observed-v1\n" + JSON.stringify([seed, user.userId, rating.animeId]))
    .digest("hex");
  return [...user.observed].sort((left, right) =>
    key(left).localeCompare(key(right)) || left.animeId - right.animeId);
}

function localHistory(rating: RawRating, title: string): HistoryEntry {
  return { provider: "local", sourceId: String(rating.animeId), title, animeId: rating.animeId,
    status: "completed", sourceStatus: "completed", progressEpisodes: null,
    score: rating.rawScore, scoreScale: "local-10" };
}
function seenHistory(animeId: number, title: string): HistoryEntry {
  return { ...localHistory({ animeId, rawScore: 5 }, title), score: null };
}

export function metricsForRanks(ranks: readonly (number | null)[], positiveCount: number,
                                topK: number) {
  if (positiveCount < 1 || ranks.length !== positiveCount || topK < 1 ||
      ranks.some((rank) => rank !== null && (!Number.isSafeInteger(rank) || rank < 1))) {
    fail("metric labels");
  }
  const hits = ranks.filter((rank): rank is number => rank !== null && rank <= topK);
  const found = ranks.filter((rank): rank is number => rank !== null);
  const dcg = hits.reduce((sum, rank) => sum + 1 / Math.log2(rank + 1), 0);
  const ideal = Array.from({ length: Math.min(positiveCount, topK) }, (_, i) =>
    1 / Math.log2(i + 2)).reduce((sum, value) => sum + value, 0);
  return { hitAtK: hits.length ? 1 : 0, recallAtK: hits.length / positiveCount,
    ndcgAtK: dcg / ideal,
    reciprocalRank: found.length ? 1 / Math.min(...found) : 0 };
}

export function evaluateNewUsers(bundle: EvalBundle, fixture: EvalFixture): UserResult[] {
  const nodes: GraphData["nodes"] = bundle.catalog.map((anime) =>
    ({ id: `anime:${anime.animeId}`, label: anime.title, nodeType: "anime" }));
  const edges: GraphData["edges"] = bundle.positivePairs.map((pair) => ({
    id: `aa:${pair.leftAnimeId}:${pair.rightAnimeId}`,
    source: `anime:${pair.leftAnimeId}`, target: `anime:${pair.rightAnimeId}`,
    edgeType: "anime-anime", weight: pair.weight, support: pair.support,
  }));
  const index = buildRecommendationIndex({ generatedAt: bundle.model.generatedAt,
    userCount: bundle.fitUserCount, animeCount: nodes.length, nodeCount: nodes.length,
    edgeCount: edges.length, nodes, edges });
  const model: ModelRecommendationIndex = { generatedAt: bundle.model.generatedAt,
    factors: bundle.model.factors, globalMean: bundle.model.globalMean,
    animeByAnimeId: new Map(bundle.model.animeIds.map((animeId, i) =>
      [animeId, { animeId, title: bundle.model.titles[i], bias: bundle.model.biases[i],
        embedding: bundle.model.embeddings[i] }])) };
  const metadata = new Map<number, AnimeMetadata>(fixture.candidateMetadata.map((item) =>
    [item.animeId, { ...item, studios: [], synopsis: "", imageUrl: "", season: null }]));
  const titleByAnimeId = new Map(bundle.catalog.map((item) => [item.animeId, item.title]));
  const results: UserResult[] = [];
  for (const [userIndex, user] of fixture.users.entries()) {
    const ordered = orderedObserved(user, fixture.seed);
    const history = user.historySeen.map((animeId) =>
      seenHistory(animeId, index.animeByAnimeId.get(animeId)!.label));
    for (const suppliedCount of COUNTS) {
      const preferences: AnimePreference[] = ordered.slice(0, suppliedCount).map((rating) =>
        preferenceFromHistory(localHistory(rating, index.animeByAnimeId.get(rating.animeId)!.label),
          `anime:${rating.animeId}`)!);
      const policy = createCandidateEligibilityPolicy({ index, preferences, history,
        includeOnlyNodeIds: user.includeOnly.map((id) => `anime:${id}`),
        excludeNodeIds: user.exclude.map((id) => `anime:${id}`), filters: user.filters });
      const modelResults = buildModelRecommendationsForPreferences(preferences, index, model);
      const ranked = rankEligibleCandidates("model", { model: modelResults }, policy, metadata).recommendations;
      let displayedEngine: UserResult["displayedEngine"] = "model";
      let displayed = ranked;
      if (!displayed.length) {
        const graph = buildGraphRecommendationsForPreferences(preferences, index);
        displayed = rankEligibleCandidates("graph", { graph }, policy, metadata).recommendations;
        displayedEngine = "graph";
      }
      if (!displayed.length) {
        const fallback = buildCatalogCoverageRecommendations(index);
        displayed = rankEligibleCandidates("fallback", { fallback }, policy, metadata).recommendations;
        displayedEngine = "coverage";
      }
      const watchedIds = new Set([...preferences.map((item) =>
        index.animeByNodeId.get(item.nodeId)?.animeId ?? -1), ...user.historySeen]);
      displayed = selectFranchiseDiverseRecommendations(displayed, metadata, watchedIds,
        false, titleByAnimeId).recommendations;
      const positives = user.validation.filter((rating) =>
        rating.rawScore >= fixture.positiveRawScoreMin);
      const eligible: RawRating[] = [];
      const excludedPositiveReasons = { watchedOrExcluded: 0, includeOnly: 0, metadataFilter: 0 };
      for (const rating of positives) {
        const anime = index.animeByAnimeId.get(rating.animeId)!;
        const probe: RecommendationResult = { anime, score: 0, strongest: 0,
          supportCount: 0, contributions: [] };
        if (policy.evaluate([probe], metadata).recommendations.length === 1) {
          eligible.push(rating);
        } else if (preferences.some((item) => item.nodeId === anime.nodeId) ||
                   user.historySeen.includes(rating.animeId) || user.exclude.includes(rating.animeId)) {
          excludedPositiveReasons.watchedOrExcluded += 1;
        } else if (user.includeOnly.length && !user.includeOnly.includes(rating.animeId)) {
          excludedPositiveReasons.includeOnly += 1;
        } else {
          excludedPositiveReasons.metadataFilter += 1;
        }
      }
      const ranks = eligible.map((rating) => {
        const position = ranked.findIndex((item) => item.anime.animeId === rating.animeId);
        return position < 0 ? null : position + 1;
      });
      const displayedRanks = eligible.map((rating) => {
        const position = displayed.findIndex((item) => item.anime.animeId === rating.animeId);
        return position < 0 ? null : position + 1;
      });
      const metric = eligible.length ? metricsForRanks(ranks, eligible.length, fixture.topK)
        : { hitAtK: 0, recallAtK: 0, ndcgAtK: 0, reciprocalRank: 0 };
      const mappedSignalCount = preferences.filter((item) => item.sentiment !== "seen" &&
        model.animeByAnimeId.has(index.animeByNodeId.get(item.nodeId)?.animeId ?? -1)).length;
      const vectorSignalCount = preferences.filter((item) => item.sentiment !== "seen").length;
      results.push({ userNumber: userIndex + 1, suppliedCount, vectorSignalCount,
        seenCount: suppliedCount - vectorSignalCount, mappedSignalCount,
        candidateCount: modelResults.length, eligibleCandidateCount: ranked.length,
        positiveLabels: positives.length, eligiblePositiveLabels: eligible.length,
        excludedPositiveLabels: positives.length - eligible.length,
        excludedPositiveReasons, ranks,
        ranked: ranked.map((item) => ({ animeId: item.anime.animeId, score: item.score })),
        displayedEngine, displayedCandidateCount: displayed.length, displayedRanks,
        ...metric });
    }
  }
  return results;
}

export function summarizeNewUsers(bundle: EvalBundle, results: readonly UserResult[]) {
  return { format: "split-first-new-user-validation-v1", candidateId: bundle.candidateId,
    trainSha256: bundle.trainSha256, fitSha256: bundle.fitSha256,
    modelSha256: bundle.modelSha256, fitUserCount: bundle.fitUserCount,
    trainRowCount: bundle.trainRowCount, catalogCount: bundle.catalog.length,
    warmValidation: { ...bundle.warmValidation, candidatePolicy: "train-items-masked-no-browser-filters" },
    newUserValidation: COUNTS.map((suppliedCount) => {
      const rows = results.filter((item) => item.suppliedCount === suppliedCount);
      const measurable = rows.filter((item) => item.eligiblePositiveLabels > 0);
      const mean = (key: "hitAtK" | "recallAtK" | "ndcgAtK" | "reciprocalRank") =>
        measurable.length ? measurable.reduce((sum, item) => sum + item[key], 0) / measurable.length : 0;
      return { suppliedCount, users: rows.length, eligibleUsers: measurable.length,
        usersWithoutEligiblePositive: rows.length - measurable.length,
        suppliedRatings: rows.length * suppliedCount,
        vectorSignals: rows.reduce((sum, item) => sum + item.vectorSignalCount, 0),
        seenOnlyRatings: rows.reduce((sum, item) => sum + item.seenCount, 0),
        mappedSignals: rows.reduce((sum, item) => sum + item.mappedSignalCount, 0),
        eligibleCandidateCounts: rows.map((item) => item.eligibleCandidateCount),
        displayedEngineCounts: { model: rows.filter((item) => item.displayedEngine === "model").length,
          graph: rows.filter((item) => item.displayedEngine === "graph").length,
          coverage: rows.filter((item) => item.displayedEngine === "coverage").length },
        positiveLabels: rows.reduce((sum, item) => sum + item.positiveLabels, 0),
        eligiblePositiveLabels: rows.reduce((sum, item) => sum + item.eligiblePositiveLabels, 0),
        excludedPositiveLabels: rows.reduce((sum, item) => sum + item.excludedPositiveLabels, 0),
        excludedPositiveReasons: {
          watchedOrExcluded: rows.reduce((sum, item) =>
            sum + item.excludedPositiveReasons.watchedOrExcluded, 0),
          includeOnly: rows.reduce((sum, item) =>
            sum + item.excludedPositiveReasons.includeOnly, 0),
          metadataFilter: rows.reduce((sum, item) =>
            sum + item.excludedPositiveReasons.metadataFilter, 0),
        },
        hitAt10: mean("hitAtK"), recallAt10: mean("recallAtK"),
        ndcgAt10: mean("ndcgAtK"), reciprocalRank: mean("reciprocalRank"),
        userRanks: rows.map((item) => ({ userNumber: item.userNumber, ranks: item.ranks,
          displayedEngine: item.displayedEngine, displayedRanks: item.displayedRanks })) };
    }) };
}

function readJson(relative: string): unknown {
  return JSON.parse(readFileSync(new URL("../../" + relative, import.meta.url), "utf8"));
}

if (process.argv[1] && fileURLToPath(import.meta.url) === resolve(process.argv[1])) {
  const bundle = loadEvalBundle();
  const fixture = parseEvalFixture(readJson("fixtures/synthetic-new-user-validation.json"),
    bundle, fitUserIds(readJson("fixtures/synthetic-new-user-fit.json")));
  const result = summarizeNewUsers(bundle, evaluateNewUsers(bundle, fixture));
  process.stdout.write(JSON.stringify(result, null, 2) + "\n");
}
