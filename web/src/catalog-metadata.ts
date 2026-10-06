/** Source-neutral projection into the existing browser recommendation metadata shape. */
import type { AnimeMetadata, CatalogMetadataItemV1 } from "./artifacts";

export function projectCatalogMetadata(item: CatalogMetadataItemV1): AnimeMetadata {
  return {
    animeId: item.animeId,
    aliases: item.aliases ?? [],
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
