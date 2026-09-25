/** Artifact transport and format selection. Validation remains in artifacts.ts. */
import {
  parseCompactGraph,
  parseCompactModel,
  parseDemoCatalog,
  parseLegacyGraph,
  parseLegacyModel,
} from "./artifacts";
import type { DemoCatalogItem, GraphV2Metadata, LoadedGraphData, ModelRecommendationAnime } from "./artifacts";
import type { ModelRecommendationIndex } from "./domain";
import type { RuntimePorts } from "./runtime";

export function createArtifactLoader(runtime: RuntimePorts, demoMode: boolean) {
  async function fetchGraph(): Promise<LoadedGraphData> {
    if (demoMode) {
      const graph = await fetchPlainJson(
        "./demo-data/graph.compact.json", "synthetic demo graph",
      );
      return parseCompactGraph(graph, "synthetic demo graph", "recommendation");
    }
    const compactData = await fetchJsonWithGzipFallback({
      path: "./data/graph.compact.json",
      required: false,
      label: "graph.compact.json",
    });
    if (compactData !== null) {
      return parseCompactGraph(compactData.value, "graph.compact.json", "recommendation");
    }
    const legacyData = await fetchJsonWithGzipFallback({
      path: "./data/graph.json",
      required: true,
      label: "graph.json",
    });
    if (!legacyData) {
      throw new Error("Unable to load required graph data.");
    }
    return parseLegacyGraph(legacyData.value, "graph.json");
  }

  async function fetchExplorerGraph(graphData: LoadedGraphData): Promise<LoadedGraphData> {
    if (demoMode) {
      const value = await fetchPlainJson("./demo-data/graph-explorer.compact.json", "synthetic demo explorer graph");
      const explorer = parseCompactGraph(value, "synthetic demo explorer graph", "visualization");
      assertExplorerMatches(graphData, explorer);
      return explorer;
    }
    const needsV2Explorer = isV2Graph(graphData);
    const explorerCompactData = await fetchJsonWithGzipFallback({
      path: "./data/graph-explorer.compact.json",
      required: needsV2Explorer,
      label: "graph-explorer.compact.json",
    });
    if (explorerCompactData !== null) {
      const explorer = parseCompactGraph(explorerCompactData.value, "graph-explorer.compact.json", "visualization");
      assertExplorerMatches(graphData, explorer);
      return explorer;
    }
    return graphData;
  }

  async function fetchModelRecommendationIndex(): Promise<ModelRecommendationIndex | null> {
    const rawCompact = demoMode
      ? { value: await fetchPlainJson(
          "./demo-data/model-mf-web.compact.json", "synthetic demo model",
        ) }
      : await fetchJsonWithGzipFallback({
          path: "./data/model-mf-web.compact.json",
          required: false,
          label: "model-mf-web.compact.json",
        });
    const rawLegacy = rawCompact !== null || demoMode
      ? null
      : await fetchJsonWithGzipFallback({
          path: "./data/model-mf-web.json",
          required: false,
          label: "model-mf-web.json",
        });

    const animeByAnimeId = new Map<number, ModelRecommendationAnime>();
    let generatedAt = "";
    let factors = 0;
    let globalMean = 0;

    if (rawCompact !== null) {
      const model = parseCompactModel(
        rawCompact.value, demoMode ? "synthetic demo model" : "model-mf-web.compact.json",
      );
      generatedAt = model.generatedAt;
      factors = model.factors;
      globalMean = model.globalMean;
      for (let index = 0; index < model.animeIds.length; index += 1) {
        const animeId = model.animeIds[index];
        animeByAnimeId.set(animeId, {
          animeId,
          title: model.titles[index],
          bias: model.biases[index],
          embedding: model.embeddings[index],
        });
      }
    } else if (rawLegacy !== null) {
      const model = parseLegacyModel(rawLegacy.value, "model-mf-web.json");
      generatedAt = model.generatedAt;
      factors = model.factors;
      globalMean = model.globalMean;
      for (const anime of model.anime) {
        animeByAnimeId.set(anime.animeId, anime);
      }
    }

    if (animeByAnimeId.size === 0) {
      return null;
    }

    return {
      generatedAt,
      factors,
      globalMean,
      animeByAnimeId,
    };
  }

  async function fetchDemoCatalog(): Promise<DemoCatalogItem[]> {
    const raw = await fetchPlainJson(
      "./demo-data/catalog.json", "synthetic demo catalog",
    );
    return parseDemoCatalog(raw, "synthetic demo catalog");
  }

  async function fetchPlainJson(url: string, label: string): Promise<unknown> {
    const response = await runtime.fetch(url);
    if (!response.ok) throw new Error(`Unable to load ${label} (${response.status}). Run npm run data:fixture.`);
    try {
      return await response.json();
    } catch {
      throw new Error(`${label}: invalid JSON. Rebuild or replace this artifact.`);
    }
  }

  async function fetchJsonWithGzipFallback({
    path,
    required,
    label,
  }: {
    path: string;
    required: boolean;
    label: string;
  }): Promise<{ value: unknown } | null> {
    const gzPath = `${path}.gz`;
    let gzStatus: number | null = null;

    try {
      const gzResponse = await runtime.fetch(gzPath);
      gzStatus = gzResponse.status;
      if (gzResponse.ok) {
        try {
          return { value: await parseGzipJsonResponse(gzResponse, label) };
        } catch {
          console.warn(`Failed to parse ${label}.gz; falling back to JSON`);
        }
      } else if (gzResponse.status !== 404) {
        console.warn(`Unable to load ${label}.gz (${gzResponse.status})`);
      }
    } catch (error) {
      console.warn(`Fetch failed for ${label}.gz`, error);
    }

    const response = await runtime.fetch(path);
    if (!response.ok) {
      if (!required && response.status === 404 && (gzStatus === 404 || gzStatus === null)) {
        return null;
      }
      throw new Error(`Unable to load ${label} (${response.status})`);
    }
    try {
      return { value: await response.json() };
    } catch {
      throw new Error(`${label}: invalid JSON. Rebuild or replace this artifact.`);
    }
  }

  async function parseGzipJsonResponse(
    response: Response,
    label: string,
  ): Promise<unknown> {
    if (typeof DecompressionStream === "undefined") {
      throw new Error(
        `This browser does not support DecompressionStream for ${label}.gz`,
      );
    }
    if (!response.body) {
      throw new Error(`Missing response body for ${label}.gz`);
    }
    const stream = response.body.pipeThrough(new DecompressionStream("gzip"));
    const text = await new Response(stream).text();
    return JSON.parse(text) as unknown;
  }

  return { fetchGraph, fetchExplorerGraph, fetchModelRecommendationIndex, fetchDemoCatalog };
}

function isV2Graph(graph: LoadedGraphData): graph is LoadedGraphData & GraphV2Metadata {
  return "format" in graph &&
    (graph.format === "graph-compact-v2" || graph.format === "graph-legacy-v2");
}

function assertExplorerMatches(main: LoadedGraphData, explorer: LoadedGraphData): void {
  if (!isV2Graph(main)) {
    if (isV2Graph(explorer)) {
      throw new Error("graph-explorer.compact.json: v2 explorer requires a v2 recommendation graph.");
    }
    return;
  }
  if (!isV2Graph(explorer) || explorer.role !== "visualization" ||
      explorer.sourceGraphId !== main.graphId ||
      explorer.dataset.sha256 !== main.dataset.sha256 ||
      explorer.dataset.scope !== main.dataset.scope ||
      explorer.dataset.source !== main.dataset.source ||
      !sameFields(explorer.semantics, main.semantics,
        ["pairWeight", "support", "recommendationUse"]) ||
      !sameFields(explorer.config, main.config,
        ["ratingSelectionPolicy", "seed", "maxRatingsPerUser", "maxAnimeAnimeEdges",
          "maxPairVisits", "maxPairCandidates", "minPairSupport", "maxNeighborsPerAnime"])) {
    throw new Error("graph-explorer.compact.json: sourceGraphId or graph provenance does not match the recommendation graph. Rebuild both artifacts.");
  }
}

function sameFields<T extends object>(left: T, right: T, fields: (keyof T)[]): boolean {
  return fields.every((field) => left[field] === right[field]);
}
