/** Artifact transport and format selection. Validation remains in artifacts.ts. */
import {
  parseAggregateDemoGraph,
  isCompactGraphData,
  parseBrowserReleaseManifest,
  parseCatalogMetadataSnapshot,
  parseCompactGraph,
  parseCompactModel,
  parseActiveReleaseBundle,
  parseDemoCatalog,
  parseLegacyGraph,
  parseLegacyModel,
  parseReleaseIdentityCatalog,
  RELEASE_BUNDLE_LIMITS,
} from "./artifacts";
import type {
  BrowserReleaseManifest, CatalogMetadataSnapshotV1, CompactGraphDataV2,
  CompactGraphDataV3, DemoCatalogItem, GraphDataV2, LoadedGraphData,
  ModelRecommendationAnime, ReleaseManifestAsset,
} from "./artifacts";
import type { ModelRecommendationIndex } from "./domain";
import type { RuntimePorts } from "./runtime";
import { measureLocal, measureLocalAsync } from "./local-performance";
import type { LocalPerformanceMetric } from "./local-performance";

/** Loader-authored field/path message; never a network or provider exception body. */
export class ArtifactLoadError extends Error {
  readonly name = "ArtifactLoadError";
}

class UnsupportedGzipError extends ArtifactLoadError {}
class ArtifactBodyReadError extends ArtifactLoadError {}

export interface LegacyJsonLimits {
  compressedBytes: number;
  plainBytes: number;
}

function legacyLimit(value: number | undefined, maximum: number, field: string): number {
  if (value === undefined) return maximum;
  if (!Number.isSafeInteger(value) || value < 1 || value > maximum) {
    throw new ArtifactLoadError(`${field}: legacy transport limit must be within 1 and ${maximum}.`);
  }
  return value;
}

export function createArtifactLoader(runtime: RuntimePorts, demoMode: boolean,
  publicBasePath = "./", legacyLimits: Partial<LegacyJsonLimits> = {}) {
  if (!/^(?:\.\/|\/|\/[A-Za-z0-9._-]+\/)$/.test(publicBasePath)) {
    throw new ArtifactLoadError("Artifact base path must be ./, /, or one absolute project path.");
  }
  const compressedLimit = legacyLimit(legacyLimits.compressedBytes,
    RELEASE_BUNDLE_LIMITS.compressedAssetBytes, "compressedBytes");
  const plainLimit = legacyLimit(legacyLimits.plainBytes,
    RELEASE_BUNDLE_LIMITS.plainAssetBytes, "plainBytes");
  const publicPath = (relative: string) => `${publicBasePath}${relative.replace(/^\.\//, "")}`;
  let activeBundlePromise: Promise<{ manifest: BrowserReleaseManifest; basePath: string } | null> | null = null;
  let loadedModelFormat: string | null = null;

  async function getActiveBundle(): Promise<{ manifest: BrowserReleaseManifest; basePath: string } | null> {
    if (demoMode) return null;
    activeBundlePromise ??= loadActiveBundle();
    return activeBundlePromise;
  }

  async function loadActiveBundle(): Promise<{ manifest: BrowserReleaseManifest; basePath: string } | null> {
    const response = await runtime.fetch(publicPath("./data/active.json"), {
      headers: { Accept: "application/json" }, cache: "no-store",
    });
    if (response.status === 404) return null;
    if (!response.ok) throw new ArtifactLoadError(`active.json: unable to load (${response.status}).`);
    const pointer = parseActiveReleaseBundle(parseBoundedJson(await readBoundedResponse(response,
      RELEASE_BUNDLE_LIMITS.activePointerBytes, "active.json"), "active.json"), "active.json");
    const basePath = publicPath(`./data/bundles/${pointer.bundleId}/`);
    const bytes = await fetchVerifiedBundleBytes(`${basePath}release-manifest.json`,
      "release-manifest.json", RELEASE_BUNDLE_LIMITS.manifestBytes,
      pointer.manifestSha256, undefined,
      "SHA-256 differs from active.json.manifestSha256.");
    const manifest = parseBrowserReleaseManifest(parseBoundedJson(bytes, "release-manifest.json"),
      "release-manifest.json");
    if (manifest.tag !== pointer.tag || manifest.bundleId !== pointer.bundleId) {
      throw new ArtifactLoadError("release-manifest.json: tag or bundleId differs from active.json.");
    }
    const entries = [manifest.neighborhood, manifest.explorer, manifest.catalog,
      ...(manifest.format === "release-manifest-v2" ? [manifest.metadata] : []),
      ...(manifest.model ? [manifest.model] : [])];
    if (entries.some((entry) => entry.bytes > RELEASE_BUNDLE_LIMITS.plainAssetBytes) ||
        entries.reduce((sum, entry) => sum + entry.bytes, 0) > RELEASE_BUNDLE_LIMITS.totalPlainBytes) {
      throw new ArtifactLoadError("release-manifest.json: declared asset bytes exceed bundle limits.");
    }
    return { manifest, basePath };
  }

  async function fetchBundleAsset(bundle: { manifest: BrowserReleaseManifest; basePath: string },
    entry: ReleaseManifestAsset): Promise<unknown> {
    const bytes = await fetchVerifiedBundleBytes(`${bundle.basePath}${entry.path}`, entry.path,
      entry.bytes, entry.sha256, entry.bytes,
      "byte length or SHA-256 differs from release-manifest.json.");
    return parseBoundedJson(bytes, entry.path);
  }

  async function fetchVerifiedBundleBytes(url: string, label: string, maximum: number,
    expectedSha256: string, expectedBytes: number | undefined, mismatch: string): Promise<Uint8Array> {
    for (let attempt = 0; attempt < 2; attempt += 1) {
      let response: Response;
      try {
        response = await runtime.fetch(url, {
          headers: { Accept: "application/json" },
          ...(attempt === 1 ? { cache: "no-store" as const } : {}),
        });
      } catch {
        throw new Error("Artifact transport failed.");
      }
      if (!response.ok) throw new ArtifactLoadError(`${label}: unable to load (${response.status}).`);
      let bytes: Uint8Array;
      try {
        bytes = await readBoundedResponse(response, maximum, label);
      } catch (error) {
        if (attempt === 0) continue;
        throw error;
      }
      if ((expectedBytes === undefined || bytes.byteLength === expectedBytes) &&
          await sha256Hex(bytes) === expectedSha256) return bytes;
      if (attempt === 1) throw new ArtifactLoadError(`${label}: ${mismatch}`);
    }
    throw new ArtifactLoadError(`${label}: ${mismatch}`);
  }

  async function fetchGraph(): Promise<LoadedGraphData> {
    if (demoMode) {
      const graph = await fetchPlainJson(
        "./demo-data/graph.aggregate.compact.json", "synthetic demo graph", "wasiw:json:graph",
      );
      return measureLocal("wasiw:schema:graph", () =>
        parseAggregateDemoGraph(graph, "synthetic demo graph", "recommendation"));
    }
    const bundle = await getActiveBundle();
    if (bundle) {
      const graph = parseCompactGraph(await fetchBundleAsset(bundle, bundle.manifest.neighborhood),
        bundle.manifest.neighborhood.path, "recommendation");
      if (graph.format !== bundle.manifest.neighborhood.format) {
        throw new ArtifactLoadError(`${bundle.manifest.neighborhood.path}: format differs from release-manifest.json.neighborhood.format.`);
      }
      return graph;
    }
    const compactData = await fetchJsonWithGzipFallback({
      path: "./data/graph.compact.json",
      required: false,
      label: "graph.compact.json",
    });
    if (compactData !== null) {
      return parseCompactGraph(compactData.value, compactData.label, "recommendation");
    }
    const legacyData = await fetchJsonWithGzipFallback({
      path: "./data/graph.json",
      required: true,
      label: "graph.json",
    });
    if (!legacyData) {
      throw new ArtifactLoadError("Unable to load required graph data.");
    }
    return parseLegacyGraph(legacyData.value, legacyData.label);
  }

  /** The v2 candidate is browser-readable but not accepted by release publishers/installers. */
  async function fetchCatalogMetadata(graphData: LoadedGraphData): Promise<CatalogMetadataSnapshotV1 | null> {
    const bundle = await getActiveBundle();
    if (!bundle || bundle.manifest.format !== "release-manifest-v2") return null;
    const { catalog: catalogEntry, metadata: metadataEntry } = bundle.manifest;
    const catalog = parseReleaseIdentityCatalog(await fetchBundleAsset(bundle, catalogEntry),
      catalogEntry.path);
    if (catalog.datasetSha256 !== bundle.manifest.dataset.sha256 ||
        catalog.anime.length !== catalogEntry.animeCount) {
      throw new ArtifactLoadError(`${catalogEntry.path}: datasetSha256 or anime count differs from release-manifest.json.catalog.`);
    }
    const mapDigest = await sha256Hex(new TextEncoder().encode(JSON.stringify(catalog.anime)));
    if (mapDigest !== catalogEntry.itemMapSha256) {
      throw new ArtifactLoadError(`${catalogEntry.path}: item map differs from release-manifest.json.catalog.itemMapSha256.`);
    }
    if (!isCompactGraphData(graphData) || graphData.format !== bundle.manifest.neighborhood.format ||
        JSON.stringify(graphData.anime) !== JSON.stringify(catalog.anime)) {
      throw new ArtifactLoadError(`${catalogEntry.path}: anime IDs or titles differ from graph.compact.json.`);
    }
    const snapshot = parseCatalogMetadataSnapshot(await fetchBundleAsset(bundle, metadataEntry),
      metadataEntry.path);
    if (snapshot.anime.length !== metadataEntry.animeCount ||
        snapshot.source.snapshotSha256 !== metadataEntry.sourceSnapshotSha256) {
      throw new ArtifactLoadError(`${metadataEntry.path}: anime count or source.snapshotSha256 differs from release-manifest.json.metadata.`);
    }
    const knownIds = new Set(catalog.anime.map(([id]) => id));
    for (let index = 0; index < snapshot.anime.length; index += 1) {
      if (!knownIds.has(snapshot.anime[index].animeId)) {
        throw new ArtifactLoadError(`${metadataEntry.path}: anime[${index}].animeId is outside catalog.identity.json.`);
      }
    }
    return snapshot;
  }

  async function fetchExplorerGraph(graphData: LoadedGraphData): Promise<LoadedGraphData> {
    if (demoMode) {
      const value = await fetchPlainJson("./demo-data/graph-explorer.aggregate.compact.json",
        "synthetic demo explorer graph", "wasiw:json:explorer");
      const explorer = measureLocal("wasiw:schema:explorer", () =>
        parseAggregateDemoGraph(value, "synthetic demo explorer graph", "visualization"));
      assertExplorerMatches(graphData, explorer);
      return explorer;
    }
    const bundle = await getActiveBundle();
    if (bundle) {
      const explorer = parseCompactGraph(await fetchBundleAsset(bundle, bundle.manifest.explorer),
        bundle.manifest.explorer.path, "visualization");
      if (explorer.format !== bundle.manifest.explorer.format) {
        throw new ArtifactLoadError(`${bundle.manifest.explorer.path}: format differs from release-manifest.json.explorer.format.`);
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
      const explorer = parseCompactGraph(explorerCompactData.value,
        explorerCompactData.label, "visualization");
      assertExplorerMatches(graphData, explorer);
      return explorer;
    }
    return graphData;
  }

  async function fetchModelRecommendationIndex(): Promise<ModelRecommendationIndex | null> {
    const bundle = await getActiveBundle();
    const rawCompact = bundle
      ? bundle.manifest.model
        ? { value: await fetchBundleAsset(bundle, bundle.manifest.model),
          label: bundle.manifest.model.path } : null
      : demoMode
      ? await fetchOptionalPlainJson("./demo-data/model-mf-web.compact.json",
          "synthetic demo model", "wasiw:json:model")
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
      const model = measureLocal("wasiw:schema:model", () => parseCompactModel(
        rawCompact.value, rawCompact.label,
      ));
      loadedModelFormat = model.format;
      generatedAt = model.generatedAt;
      factors = model.factors;
      globalMean = model.globalMean;
      measureLocal("wasiw:index:model", () => {
        for (let index = 0; index < model.animeIds.length; index += 1) {
          const animeId = model.animeIds[index];
          animeByAnimeId.set(animeId, {
            animeId,
            title: model.titles[index],
            bias: model.biases[index],
            embedding: model.embeddings[index],
          });
        }
      });
    } else if (rawLegacy !== null) {
      const model = parseLegacyModel(rawLegacy.value, rawLegacy.label);
      loadedModelFormat = "legacy-model";
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
      "./demo-data/catalog.json", "synthetic demo catalog", "wasiw:json:catalog",
    );
    return measureLocal("wasiw:schema:catalog", () => parseDemoCatalog(raw, "synthetic demo catalog"));
  }

  async function fetchPlainJson(url: string, label: string,
    metric: LocalPerformanceMetric): Promise<unknown> {
    const response = await runtime.fetch(publicPath(url));
    if (!response.ok) throw new ArtifactLoadError(`Unable to load ${label} (${response.status}). Run npm run data:fixture.`);
    try {
      return await measureLocalAsync(metric, () => response.json());
    } catch {
      throw new ArtifactLoadError(`${label}: invalid JSON. Rebuild or replace this artifact.`);
    }
  }

  async function fetchOptionalPlainJson(url: string, label: string,
    metric: LocalPerformanceMetric): Promise<{ value: unknown; label: string } | null> {
    const response = await runtime.fetch(publicPath(url));
    if (response.status === 404) return null;
    if (!response.ok) throw new ArtifactLoadError(`Unable to load ${label} (${response.status}).`);
    try {
      return { value: await measureLocalAsync(metric, () => response.json()) as unknown,
        label };
    } catch {
      throw new ArtifactLoadError(`${label}: invalid JSON. Rebuild or replace this artifact.`);
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
  }): Promise<{ value: unknown; label: string } | null> {
    const gzPath = publicPath(`${path}.gz`);
    let gzResponse: Response;
    try {
      gzResponse = await runtime.fetch(gzPath, { cache: "no-store" });
    } catch {
      throw new Error("Artifact transport failed.");
    }
    if (gzResponse.status !== 404) {
      if (!gzResponse.ok) throw new ArtifactLoadError(`${label}.gz: unable to load (${gzResponse.status}).`);
      try {
        return { value: await parseGzipJsonResponse(gzResponse, label), label: `${label}.gz` };
      } catch (error) {
        if (!(error instanceof UnsupportedGzipError)) throw error;
      }
    }

    let response: Response;
    try {
      response = await runtime.fetch(publicPath(path), { cache: "no-store" });
    } catch {
      throw new Error("Artifact transport failed.");
    }
    if (!response.ok) {
      if (!required && response.status === 404 && gzResponse.status === 404) {
        return null;
      }
      if (response.status === 404 && gzResponse.ok) {
        throw new ArtifactLoadError(`${label}.gz: this browser cannot decompress gzip and ${label} is missing.`);
      }
      throw new ArtifactLoadError(`Unable to load ${label} (${response.status}).`);
    }
    const bytes = await readBoundedResponse(response, plainLimit, label);
    return { value: parseBoundedJson(bytes, label), label };
  }

  async function parseGzipJsonResponse(
    response: Response,
    label: string,
  ): Promise<unknown> {
    const source = `${label}.gz`;
    const bytes = await readBoundedResponse(response, plainLimit, source, compressedLimit);
    if (bytes[0] !== 0x1f || bytes[1] !== 0x8b) {
      try {
        return parseBoundedJson(bytes, source);
      } catch {
        throw new ArtifactLoadError(`${source}: expected gzip bytes or browser-decoded JSON.`);
      }
    }
    if (typeof DecompressionStream === "undefined") {
      throw new UnsupportedGzipError(`${source}: this browser does not support gzip decompression.`);
    }
    let decompressor: DecompressionStream;
    try {
      decompressor = new DecompressionStream("gzip");
    } catch {
      throw new UnsupportedGzipError(`${source}: this browser does not support gzip decompression.`);
    }
    let plain: Uint8Array;
    try {
      const stream = new Response(bytes.buffer as ArrayBuffer).body!.pipeThrough(decompressor);
      plain = await readBoundedResponse(new Response(stream), plainLimit, source);
    } catch (error) {
      if (error instanceof ArtifactLoadError && !(error instanceof ArtifactBodyReadError)) throw error;
      throw new ArtifactLoadError(`${source}: invalid gzip stream.`);
    }
    return parseBoundedJson(plain, source);
  }

  return { fetchGraph, fetchExplorerGraph, fetchModelRecommendationIndex, fetchDemoCatalog,
    fetchCatalogMetadata,
    getActiveReleaseManifest: async () => (await getActiveBundle())?.manifest ?? null,
    getLoadedModelFormat: () => loadedModelFormat };
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
      throw new ArtifactLoadError(`graph-explorer.compact.json: ${version} explorer requires a ${version} recommendation graph.`);
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
    throw new ArtifactLoadError("graph-explorer.compact.json: sourceGraphId or graph provenance does not match the recommendation graph. Rebuild both artifacts.");
  }
  if (main.format === "graph-compact-v3" && explorer.format === "graph-compact-v3" &&
      explorer.projection.policy !== main.projection.policy) {
    throw new ArtifactLoadError("graph-explorer.compact.json: projection does not match the recommendation graph.");
  }
}

function sameFields<T extends object>(left: T, right: T, fields: (keyof T)[]): boolean {
  return fields.every((field) => left[field] === right[field]);
}

async function readBoundedResponse(response: Response, maximum: number, label: string,
  compressedMaximum?: number): Promise<Uint8Array> {
  const advertised = response.headers.get("content-length");
  if (advertised !== null && /^\d+$/.test(advertised) && Number(advertised) > maximum) {
    throw new ArtifactLoadError(`${label}: advertised byte length exceeds transport limit.`);
  }
  if (!response.body) throw new ArtifactLoadError(`${label}: response body is missing.`);
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let length = 0;
  let completed = false;
  let firstByte: number | undefined;
  let secondByte: number | undefined;
  try {
    while (true) {
      const next = await reader.read();
      if (next.done) { completed = true; break; }
      let prefixIndex = 0;
      if (firstByte === undefined && next.value.byteLength > prefixIndex) {
        firstByte = next.value[prefixIndex++];
      }
      if (secondByte === undefined && next.value.byteLength > prefixIndex) {
        secondByte = next.value[prefixIndex];
      }
      length += next.value.byteLength;
      const limit = firstByte === 0x1f && secondByte === 0x8b && compressedMaximum !== undefined
        ? compressedMaximum : maximum;
      if (length > limit) throw new ArtifactLoadError(`${label}: byte length exceeds transport limit.`);
      chunks.push(next.value);
    }
  } catch (error) {
    if (error instanceof ArtifactLoadError) throw error;
    throw new ArtifactBodyReadError(`${label}: response body could not be read.`);
  } finally {
    if (!completed) void reader.cancel().catch(() => undefined);
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
    throw new ArtifactLoadError(`${label}: invalid JSON or UTF-8. Rebuild or replace this artifact.`);
  }
}

async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes.buffer as ArrayBuffer));
  return [...digest].map((value) => value.toString(16).padStart(2, "0")).join("");
}
