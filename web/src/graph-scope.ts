/** Human-readable boundaries for the loaded recommendation graph and explorer sample. */
import type { CompactGraphDataV2, CompactGraphDataV3, GraphDataV2,
  LoadedGraphData } from "./artifacts";

export interface GraphScope {
  versions: string;
  selection: string;
  explorer: string;
  limits: string;
  caveat: string;
  userRowsOmitted: boolean;
}

function versioned(graph: LoadedGraphData):
  graph is GraphDataV2 | CompactGraphDataV2 | CompactGraphDataV3 {
  return "format" in graph && graph.format !== "graph-compact-v1";
}

function format(graph: LoadedGraphData): string {
  return "format" in graph ? graph.format : "unversioned legacy graph";
}

export function describeGraphScope(recommendation: LoadedGraphData, explorer: LoadedGraphData,
  maxRenderedPairs: number, maxRenderedUserEdges: number): GraphScope {
  const userRowsOmitted = "format" in recommendation && recommendation.format === "graph-compact-v3";
  const versions = `Recommendation graph: ${format(recommendation)}. Explorer: ${format(explorer)}${
    explorer === recommendation ? " (same loaded graph)" : " (separate loaded asset)"}.`;
  const limits = `Local drawing limits: ${maxRenderedPairs} pair edges and ${maxRenderedUserEdges} user-anime edges; the current edge filter may show fewer.`;
  const caveat = "Missing nodes or edges may reflect source selection, pair limits, explorer sampling, filters, or drawing limits. Absence in this view does not prove no relationship.";

  if (!versioned(recommendation)) {
    return {
      versions,
      selection: "This legacy graph does not declare source rating or pair-selection coverage.",
      explorer: "Explorer sample coverage cannot be verified from this legacy graph.",
      limits, caveat, userRowsOmitted: false,
    };
  }

  const source = recommendation.truncation;
  const selection = `${source.selectedRatings}/${source.inputRatings} source ratings selected for pair computation; ` +
    `${source.ratingsSkipped} omitted before aggregation (${recommendation.config.ratingSelectionPolicy}). ` +
    `${source.selectedPairs}/${source.eligiblePairs} eligible pair edges retained; ` +
    `${source.excludedBySupport} candidate pairs lacked minimum support, ` +
    `${source.excludedByNeighborLimit} were omitted by the neighbor limit, and ` +
    `${source.excludedByOutputLimit} by the output limit.`;

  if (!versioned(explorer) || explorer.role !== "visualization" || !explorer.visualization) {
    return { versions, selection,
      explorer: "Explorer sampling counts are unavailable; only loaded edges can be described.",
      limits, caveat, userRowsOmitted };
  }

  const sample = explorer.visualization;
  const shownPairs = explorer.aa.length;
  const shownUserEdges = explorer.ua.length;
  const explorerPairs = `Explorer sample: ${shownPairs}/${source.selectedPairs} retained pair edges ` +
    `(${sample.excludedAnimeAnimeEdges} omitted; sample cap ${sample.maxAnimeAnimeEdges}).`;
  const explorerUsers = userRowsOmitted
    ? " User rows are deliberately omitted from v3; selected-rating counts describe pair inputs, not visible users or global popularity."
    : ` ${shownUserEdges}/${source.selectedRatings} retained user-anime edges sampled ` +
      `(${sample.excludedUserAnimeEdges} omitted; sample cap ${sample.maxUserAnimeEdges}); user edges are hidden by default.`;
  return { versions, selection, explorer: explorerPairs + explorerUsers,
    limits, caveat, userRowsOmitted };
}
