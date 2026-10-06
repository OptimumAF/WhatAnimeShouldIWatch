/** Source-neutral projection into the existing browser recommendation metadata shape. */
import type { AnimeMetadata, CatalogMetadataItemV1 } from "./artifacts";

export function projectCatalogMetadata(item: CatalogMetadataItemV1): AnimeMetadata {
  // A source's canonical label may differ from the stable graph/identity label.
  const aliases = new Map<string, string>();
  for (const name of [item.title, ...(item.aliases ?? [])]) {
    const key = name.trim().toLowerCase();
    if (!aliases.has(key)) aliases.set(key, name);
  }
  return {
    animeId: item.animeId,
    aliases: [...aliases.values()],
    mediaFormat: item.mediaFormat,
    year: item.year,
    score: item.communityScore,
    genres: item.genres ?? [],
    studios: [],
    synopsis: "",
    imageUrl: "",
    season: null,
    relations: item.relations,
    episodeCount: item.episodeCount,
    runtimeMinutes: item.runtimeMinutes,
    contentClassification: item.contentClassification,
  };
}
