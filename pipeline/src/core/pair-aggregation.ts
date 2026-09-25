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

export function aggregateAnimePairs(
  users: PairUser[],
  maxRatingsPerUser: number,
  maxAnimeAnimeEdges: number,
): { pairs: Map<string, PairAggregate>; skippedNewPairs: number } {
  const pairStats = new Map<string, { sum: number; count: number }>();
  let skippedNewPairs = 0;

  // Keep the existing first-N cap policy, then make traversal independent of
  // input ordering for uncapped data and of user ordering for capped data.
  const sortedUsers = [...users].sort((a, b) =>
    a.userId < b.userId ? -1 : a.userId > b.userId ? 1 : 0,
  );
  for (const user of sortedUsers) {
    const selected = maxRatingsPerUser > 0
      ? user.ratings.slice(0, maxRatingsPerUser)
      : user.ratings;
    const ratings = [...selected].sort((a, b) => a.animeId - b.animeId);

    for (let i = 0; i < ratings.length; i += 1) {
      for (let j = i + 1; j < ratings.length; j += 1) {
        const low = ratings[i].animeId;
        const high = ratings[j].animeId;
        const key = `${low}:${high}`;
        const userPairScore =
          (ratings[i].normalizedScore + ratings[j].normalizedScore) / 2;
        const current = pairStats.get(key);
        if (
          current === undefined &&
          maxAnimeAnimeEdges > 0 &&
          pairStats.size >= maxAnimeAnimeEdges
        ) {
          skippedNewPairs += 1;
          continue;
        }
        pairStats.set(key, {
          sum: (current?.sum ?? 0) + userPairScore,
          count: (current?.count ?? 0) + 1,
        });
      }
    }
  }

  const sortedPairs = [...pairStats].sort(([a], [b]) => {
      const [aLow, aHigh] = a.split(":").map(Number);
      const [bLow, bHigh] = b.split(":").map(Number);
      return aLow - bLow || aHigh - bHigh;
  });
  const pairs = new Map<string, PairAggregate>(sortedPairs.map(([key, stats]) => [key, {
      weight: stats.sum / stats.count,
      support: stats.count,
  }]));
  return {
    pairs,
    skippedNewPairs,
  };
}
