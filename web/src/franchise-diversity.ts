import type { AnimeMetadata, AnimeRelation } from "./artifacts";
import type { RecommendationResult } from "./domain";
import { normalizeTitle } from "./title";

export interface FranchiseSelection {
  recommendations: RecommendationResult[];
  notesByAnimeId: ReadonlyMap<number, string>;
  knownPrequelHidden: number;
  repeatedFranchiseHidden: number;
  relationshipPayloadCount: number;
  uncheckedCount: number;
}

function normalizedSeriesTitle(title: string): string {
  return normalizeTitle(title).replace(/\s*[:–—-]\s*/g, " ").trim();
}

function hasSerialSuffix(title: string): boolean {
  return /\s+(?:season|part)\s+[1-9]\d*$/i.test(normalizedSeriesTitle(title));
}

function titleFamilyKey(title: string): string {
  const normalized = normalizedSeriesTitle(title);
  return normalized.replace(/\s+(?:season|part)\s+[1-9]\d*$/i, "").trim();
}

/** Final-list selector. It never adds a candidate or changes a source score/rank. */
export function selectFranchiseDiverseRecommendations(
  eligible: readonly RecommendationResult[],
  metadataByAnimeId: ReadonlyMap<number, AnimeMetadata>,
  seenAnimeIds: ReadonlySet<number>,
  allowRelatedTitles: boolean,
  titleByAnimeId: ReadonlyMap<number, string>,
): FranchiseSelection {
  const parent = new Map<number, number>();
  const find = (id: number): number => {
    const current = parent.get(id);
    if (current === undefined) {
      parent.set(id, id);
      return id;
    }
    if (current === id) return id;
    const root = find(current);
    parent.set(id, root);
    return root;
  };
  const unite = (left: number, right: number): void => {
    const a = find(left);
    const b = find(right);
    if (a !== b) parent.set(Math.max(a, b), Math.min(a, b));
  };
  const prequels = new Map<number, Map<number, string>>();
  const addPrequel = (candidateId: number, prequelId: number, title: string): void => {
    const known = prequels.get(candidateId) ?? new Map<number, string>();
    known.set(prequelId, title);
    prequels.set(candidateId, known);
  };
  const relationKinds = new Set<AnimeRelation["kind"]>([
    "prequel", "sequel", "alternative-version", "side-story", "spin-off",
  ]);
  for (const [animeId, metadata] of metadataByAnimeId) {
    for (const relation of metadata.relations ?? []) {
      if (!relationKinds.has(relation.kind)) continue;
      unite(animeId, relation.animeId);
      if (relation.kind === "prequel") {
        addPrequel(animeId, relation.animeId, relation.title);
      } else if (relation.kind === "sequel") {
        addPrequel(relation.animeId, animeId, titleByAnimeId.get(animeId) ?? `Anime ${animeId}`);
      }
    }
  }

  const firstByTitleFamily = new Map<string, number>();
  const bareByTitleFamily = new Map<string, number>();
  const titleFamilyCounts = new Map<string, number>();
  for (const item of eligible) {
    const key = titleFamilyKey(item.anime.label);
    titleFamilyCounts.set(key, (titleFamilyCounts.get(key) ?? 0) + 1);
    if (!hasSerialSuffix(item.anime.label) && !bareByTitleFamily.has(key)) {
      bareByTitleFamily.set(key, item.anime.animeId);
    }
    const first = firstByTitleFamily.get(key);
    if (first === undefined) firstByTitleFamily.set(key, item.anime.animeId);
    else unite(first, item.anime.animeId);
  }

  let knownPrequelHidden = 0;
  let repeatedFranchiseHidden = 0;
  let relationshipPayloadCount = 0;
  const notesByAnimeId = new Map<number, string>();
  const selectedGroups = new Set<number>();
  const recommendations: RecommendationResult[] = [];
  for (const item of eligible) {
    const id = item.anime.animeId;
    const metadata = metadataByAnimeId.get(id);
    if (metadata?.relations !== undefined && metadata.relations !== null) relationshipPayloadCount += 1;
    const directPrequels = [...(prequels.get(id)?.entries() ?? [])]
      .sort(([left], [right]) => left - right);
    const unseen = directPrequels.filter(([prequelId]) => !seenAnimeIds.has(prequelId));
    const knownNote = directPrequels.length > 0
      ? `Known immediate prequel${directPrequels.length === 1 ? "" : "s"}: ` +
        directPrequels.map(([prequelId, title]) =>
          `${title} (${seenAnimeIds.has(prequelId) ? "in watched history" : "not in watched history"})`).join(", ") +
        ". Other prerequisites may be unlisted."
      : "Prerequisites unverified; relationship data may be incomplete.";
    const titleCue = titleFamilyCounts.get(titleFamilyKey(item.anime.label)) ?? 0;
    notesByAnimeId.set(id, knownNote + (titleCue > 1 ? " Possible same-series title match." : ""));
    if (!allowRelatedTitles && unseen.length > 0) {
      knownPrequelHidden += 1;
      continue;
    }
    if (!allowRelatedTitles && hasSerialSuffix(item.anime.label) &&
        bareByTitleFamily.has(titleFamilyKey(item.anime.label))) {
      repeatedFranchiseHidden += 1;
      continue;
    }
    const group = find(id);
    if (!allowRelatedTitles && selectedGroups.has(group)) {
      repeatedFranchiseHidden += 1;
      continue;
    }
    recommendations.push(item);
    selectedGroups.add(group);
  }
  return {
    recommendations, notesByAnimeId, knownPrequelHidden, repeatedFranchiseHidden,
    relationshipPayloadCount, uncheckedCount: eligible.length - relationshipPayloadCount,
  };
}
