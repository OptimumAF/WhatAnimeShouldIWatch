/** Artifact transport and format selection. Validation remains in artifacts.ts. */
import {
  parseCompactGraph,
  parseCompactModel,
  parseActiveReleaseBundle,
  parseDemoCatalog,
  parseLegacyGraph,
  parseLegacyModel,
  parseReleaseManifest,
  RELEASE_BUNDLE_LIMITS,
} from "./artifacts";
import type {
  CompactGraphDataV2, CompactGraphDataV3, DemoCatalogItem, GraphDataV2,
  LoadedGraphData, ModelRecommendationAnime, ReleaseManifestAsset, ReleaseManifestV1,
} from "./artifacts";
import type { ModelRecommendationIndex } from "./domain";
import type { RuntimePorts } from "./runtime";

export function createArtifactLoader(runtime: RuntimePorts, demoMode: boolean) {
  let activeBundlePromise: Promise<{ manifest: ReleaseManifestV1; basePath: string } | null> | null = null;

  async function getActiveBundle(): Promise<{ manifest: ReleaseManifestV1; basePath: string } | null> {
    if (demoMode) return null;
    activeBundlePromise ??= loadActiveBundle();
    return activeBundlePromise;
  }

  async function loadActiveBundle(): Promise<{ manifest: ReleaseManifestV1; basePath: string } | null> {
    const response = await runtime.fetch("./data/active.json", {
      headers: { Accept: "application/json" }, cache: "no-store",
    });
    if (response.status === 404) return null;
    if (!response.ok) throw new Error(`active.json: unable to load (${response.status}).`);
    const pointer = parseActiveReleaseBundle(parseBoundedJson(await readBoundedResponse(response,
      RELEASE_BUNDLE_LIMITS.activePointerBytes, "active.json"), "active.json"), "active.json");
    const basePath = `./data/bundles/${pointer.bundleId}/`;
    const manifestResponse = await runtime.fetch(`${basePath}release-manifest.json`, {
      headers: { Accept: "application/json" },
    });
    if (!manifestResponse.ok) {
      throw new Error(`release-manifest.json: unable to load (${manifestResponse.status}).`);
    }
    const bytes = await readBoundedResponse(manifestResponse,
      RELEASE_BUNDLE_LIMITS.manifestBytes, "release-manifest.json");
    if (await sha256Hex(bytes) !== pointer.manifestSha256) {
      throw new Error("release-manifest.json: SHA-256 differs from active.json.manifestSha256.");
    }
    const manifest = parseReleaseManifest(parseBoundedJson(bytes, "release-manifest.json"),
      "release-manifest.json");
    if (manifest.tag !== pointer.tag || manifest.bundleId !== pointer.bundleId) {
      throw new Error("release-manifest.json: tag or bundleId differs from active.json.");
    }
    const entries = [manifest.neighborhood, manifest.explorer, manifest.catalog,
      ...(manifest.model ? [manifest.model] : [])];
    if (entries.some((entry) => entry.bytes > RELEASE_BUNDLE_LIMITS.plainAssetBytes) ||
        entries.reduce((sum, entry) => sum + entry.bytes, 0) > RELEASE_BUNDLE_LIMITS.totalPlainBytes) {
      throw new Error("release-manifest.json: declared asset bytes exceed bundle limits.");
    }
    return { manifest, basePath };
  }

  async function fetchBundleAsset(bundle: { manifest: ReleaseManifestV1; basePath: string },
    entry: ReleaseManifestAsset): Promise<unknown> {
    const response = await runtime.fetch(`${bundle.basePath}${entry.path}`, {
      headers: { Accept: "application/json" },
    });
    if (!response.ok) throw new Error(`${entry.path}: unable to load (${response.status}).`);
    const bytes = await readBoundedResponse(response, entry.bytes, entry.path);
    if (bytes.byteLength !== entry.bytes || await sha256Hex(bytes) !== entry.sha256) {
      throw new Error(`${entry.path}: byte length or SHA-256 differs from release-manifest.json.`);
    }
    return parseBoundedJson(bytes, entry.path);
  }

  async function fetchGraph(): Promise<LoadedGraphData> {
    if (demoMode) {
      const graph = await fetchPlainJson(
        "./demo-data/graph.compact.json", "synthetic demo graph",
      );
      return parseCompactGraph(graph, "synthetic demo graph", "recommendation");
    }
    const bundle = await getActiveBundle();
    if (bundle) {
      const graph = parseCompactGraph(await fetchBundleAsset(bundle, bundle.manifest.neighborhood),
        bundle.manifest.neighborhood.path, "recommendation");
      if (graph.format !== bundle.manifest.neighborhood.format) {
        throw new Error(`${bundle.manifest.neighborhood.path}: format differs from release-manifest.json.neighborhood.format.`);
      }
      return graph;
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
    const bundle = await getActiveBundle();
    if (bundle) {
      const explorer = parseCompactGraph(await fetchBundleAsset(bundle, bundle.manifest.explorer),
        bundle.manifest.explorer.path, "visualization");
      if (explorer.format !== bundle.manifest.explorer.format) {
        throw new Error(`${bundle.manifest.explorer.path}: format differs from release-manifest.json.explorer.format.`);
      }
      assertExplorerMatches(graphData, explorer);
      return explorer;
    }
    const needsVersionedExplorer = isVersionedGraph(graphData);
    const explorerCompactData = await fetchJsonWithGzipFallback({
      path: "./data/graph-explorer.compact.json",
      required: needsVersionedExplorer,
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
    const bundle = await getActiveBundle();
    const rawCompact = bundle
      ? bundle.manifest.model
        ? { value: await fetchBundleAsset(bundle, bundle.manifest.model) } : null
      : demoMode
      ? await fetchOptionalPlainJson("./demo-data/model-mf-web.compact.json", "synthetic demo model")
      : await fetchJsonWithGzipFallback({
          path: "./data/model-mf-web.compact.json",
          required: false,
          label: "model-mf-web.compact.json",
        });
    const rawLegacy = rawCompact !== null || demoMode || bundle
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

  async function fetchOptionalPlainJson(url: string, label: string): Promise<{ value: unknown } | null> {
    const response = await runtime.fetch(url);
    if (response.status === 404) return null;
    if (!response.ok) throw new Error(`Unable to load ${label} (${response.status}).`);
    try {
      return { value: await response.json() as unknown };
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

function isVersionedGraph(graph: LoadedGraphData):
  graph is GraphDataV2 | CompactGraphDataV2 | CompactGraphDataV3 {
  return "format" in graph &&
    (graph.format === "graph-compact-v2" || graph.format === "graph-compact-v3" ||
      graph.format === "graph-legacy-v2");
}

function assertExplorerMatches(main: LoadedGraphData, explorer: LoadedGraphData): void {
  if (!isVersionedGraph(main)) {
    if (isVersionedGraph(explorer)) {
      const version = explorer.format === "graph-compact-v3" ? "v3" : "v2";
      throw new Error(`graph-explorer.compact.json: ${version} explorer requires a ${version} recommendation graph.`);
    }
    return;
  }
  if (!isVersionedGraph(explorer) || explorer.role !== "visualization" ||
      explorer.format !== main.format ||
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
  if (main.format === "graph-compact-v3" && explorer.format === "graph-compact-v3" &&
      explorer.projection.policy !== main.projection.policy) {
    throw new Error("graph-explorer.compact.json: projection does not match the recommendation graph.");
  }
}

function sameFields<T extends object>(left: T, right: T, fields: (keyof T)[]): boolean {
  return fields.every((field) => left[field] === right[field]);
}

async function readBoundedResponse(response: Response, maximum: number, label: string): Promise<Uint8Array> {
  const advertised = response.headers.get("content-length");
  if (advertised !== null && /^\d+$/.test(advertised) && Number(advertised) > maximum) {
    throw new Error(`${label}: advertised byte length exceeds bundle limit.`);
  }
  if (!response.body) throw new Error(`${label}: response body is missing.`);
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let length = 0;
  let completed = false;
  try {
    while (true) {
      const next = await reader.read();
      if (next.done) { completed = true; break; }
      length += next.value.byteLength;
      if (length > maximum) throw new Error(`${label}: byte length exceeds bundle limit.`);
      chunks.push(next.value);
    }
  } finally {
    if (!completed) await reader.cancel().catch(() => undefined);
    reader.releaseLock();
  }
  const bytes = new Uint8Array(length);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return bytes;
}

function parseBoundedJson(bytes: Uint8Array, label: string): unknown {
  try {
    return JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes)) as unknown;
  } catch {
    throw new Error(`${label}: invalid JSON or UTF-8. Rebuild or replace this artifact.`);
  }
}

async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes.buffer as ArrayBuffer));
  return [...digest].map((value) => value.toString(16).padStart(2, "0")).join("");
}
