import { createHash } from "node:crypto";

export interface PairRating {
  animeId: number;
  normalizedScore: number;
}

export interface PairUser {
  userId: string;
  ratings: PairRating[];
}

export interface PairAggregate {
  weight: number;
  support: number;
}

export interface PairSelectionOptions {
  /** Fail before pair enumeration when this exact observation bound is exceeded. */
  maxPairVisits?: number;
  /** Fail instead of evicting or silently omitting candidate statistics. */
  maxCandidatePairs?: number;
  minSupport?: number;
  /** Greedy degree limit after ranking by support, magnitude, then numeric IDs. */
  maxNeighborsPerAnime?: number;
  /** Unsigned 32-bit seed for the stable per-user hash sample when capped. */
  selectionSeed?: number;
}

export interface PairSelectionStats {
  inputUsers: number;
  usersCapped: number;
  inputRatings: number;
  selectedRatings: number;
  inputAnimeCount: number;
  selectedAnimeCount: number;
  potentialPairVisits: number;
  pairVisits: number;
  pairVisitsSkippedByUserCap: number;
  candidatePairs: number;
  eligiblePairs: number;
  selectedPairs: number;
  excludedBySupport: number;
  excludedByNeighborLimit: number;
  excludedByOutputLimit: number;
  ratingsSkippedByUserCap: number;
}

export const DEFAULT_MAX_PAIR_VISITS = 20_000_000;
export const DEFAULT_MAX_CANDIDATE_PAIRS = 2_500_000;
export const DEFAULT_PAIR_SELECTION_SEED = 0;
export const PAIR_CAP_POLICY = "sha256-bottom-k-v1";

interface CandidatePair {
  key: string;
  low: number;
  high: number;
  sum: number;
  count: number;
  weight: number;
}

function requireSafeInteger(name: string, value: number, minimum: number): void {
  if (!Number.isSafeInteger(value) || value < minimum) {
    throw new Error(`${name} must be a ${minimum === 0 ? "nonnegative" : "positive"} safe integer.`);
  }
}

/** Exact v1 pair means, with independent work/key budgets and deterministic output selection. */
export function aggregateAnimePairs<T extends PairUser>(
  users: T[],
  maxRatingsPerUser: number,
  maxAnimeAnimeEdges: number,
  options: PairSelectionOptions = {},
): { pairs: Map<string, PairAggregate>; stats: PairSelectionStats; selectedUsers: T[] } {
  const maxPairVisits = options.maxPairVisits ?? DEFAULT_MAX_PAIR_VISITS;
  const maxCandidatePairs = options.maxCandidatePairs ?? DEFAULT_MAX_CANDIDATE_PAIRS;
  const minSupport = options.minSupport ?? 1;
  const maxNeighborsPerAnime = options.maxNeighborsPerAnime ?? 0;
  const selectionSeed = options.selectionSeed ?? DEFAULT_PAIR_SELECTION_SEED;
  requireSafeInteger("maxRatingsPerUser", maxRatingsPerUser, 0);
  requireSafeInteger("maxAnimeAnimeEdges", maxAnimeAnimeEdges, 0);
  requireSafeInteger("maxPairVisits", maxPairVisits, 1);
  requireSafeInteger("maxCandidatePairs", maxCandidatePairs, 1);
  requireSafeInteger("minSupport", minSupport, 1);
  requireSafeInteger("maxNeighborsPerAnime", maxNeighborsPerAnime, 0);
  if (!Number.isInteger(selectionSeed) || selectionSeed < 0 || selectionSeed > 0xffffffff) {
    throw new Error("selectionSeed must be an unsigned 32-bit integer.");
  }

  // Count full and selected pair work before building keys. Fail closed if the
  // selected observations exceed the work budget, even with an output cap.
  let potentialPairVisits = 0;
  let pairVisits = 0;
  let inputRatings = 0;
  let ratingsSkippedByUserCap = 0;
  for (const user of users) {
    inputRatings += user.ratings.length;
    potentialPairVisits += user.ratings.length * (user.ratings.length - 1) / 2;
    if (!Number.isSafeInteger(inputRatings) || !Number.isSafeInteger(potentialPairVisits)) {
      throw new Error("Input rating or potential pair count exceeds the safe integer range.");
    }
    const selectedCount = maxRatingsPerUser > 0
      ? Math.min(user.ratings.length, maxRatingsPerUser) : user.ratings.length;
    ratingsSkippedByUserCap += user.ratings.length - selectedCount;
    pairVisits += selectedCount * (selectedCount - 1) / 2;
    if (!Number.isSafeInteger(pairVisits) || pairVisits > maxPairVisits) {
      throw new Error(`Pair-visit budget exceeded: ${pairVisits} observations exceed limit ${maxPairVisits}. Raise --max-pair-visits or use a deliberate per-user selection policy.`);
    }
  }

  const inputAnimeIds = new Set<number>();
  const selectedAnimeIds = new Set<number>();
  const sortedUsers = [...users].sort((a, b) =>
    a.userId < b.userId ? -1 : a.userId > b.userId ? 1 : 0,
  );
  const selectedUsers: T[] = [];
  let usersCapped = 0;
  let previousUserId: string | null = null;
  for (const user of sortedUsers) {
    if (typeof user.userId !== "string" || user.userId.length === 0) {
      throw new Error("Invalid user ID in pair input.");
    }
    if (user.userId === previousUserId) {
      throw new Error("Duplicate user ID in pair input.");
    }
    previousUserId = user.userId;
    const ratings = [...user.ratings].sort((a, b) => a.animeId - b.animeId);
    for (let i = 0; i < ratings.length; i += 1) {
      const rating = ratings[i];
      if (!Number.isSafeInteger(rating.animeId) || rating.animeId <= 0 ||
          !Number.isFinite(rating.normalizedScore)) {
        throw new Error("Invalid anime ID or centered score in pair input.");
      }
      if (i > 0 && ratings[i - 1].animeId === rating.animeId) {
        throw new Error(`Duplicate anime ID ${rating.animeId} in one user's pair input.`);
      }
      inputAnimeIds.add(rating.animeId);
    }
    const isCapped = maxRatingsPerUser > 0 && ratings.length > maxRatingsPerUser;
    let selected = ratings;
    if (isCapped) {
      usersCapped += 1;
      selected = ratings.map((rating) => ({
        rating,
        rank: createHash("sha256")
          .update(JSON.stringify([PAIR_CAP_POLICY, selectionSeed, user.userId, rating.animeId]))
          .digest("hex"),
      })).sort((a, b) =>
        a.rank < b.rank ? -1 : a.rank > b.rank ? 1 : a.rating.animeId - b.rating.animeId,
      ).slice(0, maxRatingsPerUser).map(({ rating }) => rating)
        .sort((a, b) => a.animeId - b.animeId);
    }
    for (const rating of selected) {
      selectedAnimeIds.add(rating.animeId);
    }
    selectedUsers.push(isCapped ? { ...user, ratings: selected } as T : user);
  }

  const candidates = new Map<string, CandidatePair>();
  for (const user of selectedUsers) {
    const ratings = [...user.ratings].sort((a, b) => a.animeId - b.animeId);
    for (let i = 0; i < ratings.length; i += 1) {
      for (let j = i + 1; j < ratings.length; j += 1) {
        const low = ratings[i].animeId;
        const high = ratings[j].animeId;
        const key = `${low}:${high}`;
        let current = candidates.get(key);
        if (!current) {
          if (candidates.size >= maxCandidatePairs) {
            throw new Error(`Candidate-key budget exceeded: limit ${maxCandidatePairs}. No partial pair graph was produced.`);
          }
          current = { key, low, high, sum: 0, count: 0, weight: 0 };
          candidates.set(key, current);
        }
        const pairScore = ratings[i].normalizedScore / 2 + ratings[j].normalizedScore / 2;
        current.sum += pairScore;
        current.count += 1;
        if (!Number.isFinite(current.sum)) {
          throw new Error(`Non-finite pair sum for ${key}.`);
        }
      }
    }
  }

  const ranked: CandidatePair[] = [];
  for (const pair of candidates.values()) {
    if (pair.count >= minSupport) {
      pair.weight = pair.sum / pair.count;
      ranked.push(pair);
    }
  }
  ranked.sort((a, b) =>
    b.count - a.count || Math.abs(b.weight) - Math.abs(a.weight) ||
    a.low - b.low || a.high - b.high,
  );

  const degree = new Map<number, number>();
  const selected: CandidatePair[] = [];
  let excludedByNeighborLimit = 0;
  let excludedByOutputLimit = 0;
  for (const pair of ranked) {
    if (maxNeighborsPerAnime > 0 &&
        ((degree.get(pair.low) ?? 0) >= maxNeighborsPerAnime ||
         (degree.get(pair.high) ?? 0) >= maxNeighborsPerAnime)) {
      excludedByNeighborLimit += 1;
      continue;
    }
    if (maxAnimeAnimeEdges > 0 && selected.length >= maxAnimeAnimeEdges) {
      excludedByOutputLimit += 1;
      continue;
    }
    selected.push(pair);
    degree.set(pair.low, (degree.get(pair.low) ?? 0) + 1);
    degree.set(pair.high, (degree.get(pair.high) ?? 0) + 1);
  }
  selected.sort((a, b) => a.low - b.low || a.high - b.high);
  const pairs = new Map<string, PairAggregate>(selected.map((pair) => [pair.key, {
    weight: pair.weight,
    support: pair.count,
  }]));
  return {
    pairs,
    selectedUsers,
    stats: {
      inputUsers: users.length,
      usersCapped,
      inputRatings,
      selectedRatings: inputRatings - ratingsSkippedByUserCap,
      inputAnimeCount: inputAnimeIds.size,
      selectedAnimeCount: selectedAnimeIds.size,
      potentialPairVisits,
      pairVisits,
      pairVisitsSkippedByUserCap: potentialPairVisits - pairVisits,
      candidatePairs: candidates.size,
      eligiblePairs: ranked.length,
      selectedPairs: selected.length,
      excludedBySupport: candidates.size - ranked.length,
      excludedByNeighborLimit,
      excludedByOutputLimit,
      ratingsSkippedByUserCap,
    },
  };
}
