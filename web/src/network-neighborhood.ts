import { isCompactGraphData } from "./artifacts";
import type { GraphEdge, GraphNode, LoadedGraphData } from "./artifacts";

export interface AnimeNeighborhood {
  center: GraphNode;
  nodes: GraphNode[];
  /** Strongest distinct one-hop pair edges, ordered by absolute signed preference. */
  edges: GraphEdge[];
  eligiblePairEdges: number;
  omittedByBudget: number;
}

function neighborId(edge: GraphEdge, centerId: string): string {
  return edge.source === centerId ? edge.target : edge.source;
}

function compareEvidence(left: GraphEdge, right: GraphEdge, centerId: string): number {
  const leftNeighbor = neighborId(left, centerId).slice(6);
  const rightNeighbor = neighborId(right, centerId).slice(6);
  const leftNumber = Number(leftNeighbor);
  const rightNumber = Number(rightNeighbor);
  const neighborOrder = Number.isSafeInteger(leftNumber) && Number.isSafeInteger(rightNumber)
    ? leftNumber - rightNumber : leftNeighbor.localeCompare(rightNeighbor);
  return Math.abs(right.weight) - Math.abs(left.weight) ||
    (right.support ?? 0) - (left.support ?? 0) ||
    neighborOrder ||
    left.id.localeCompare(right.id);
}

/** Selects a bounded, exact one-hop view; pair preference is never renamed similarity. */
export function selectAnimeNeighborhood(graph: LoadedGraphData, centerId: string,
  maxNodes: number, maxEdges: number, minAbsoluteWeight = 0): AnimeNeighborhood | null {
  if (!Number.isInteger(maxNodes) || maxNodes < 1 ||
      !Number.isInteger(maxEdges) || maxEdges < 0 ||
      !Number.isFinite(minAbsoluteWeight) || minAbsoluteWeight < 0) {
    throw new RangeError("Network neighborhood budgets and weight threshold must be finite and nonnegative.");
  }
  const edgeBudget = Math.min(maxNodes - 1, maxEdges);
  const selected = new Map<string, GraphEdge>();
  let eligiblePairEdges = 0;
  const offer = (edge: GraphEdge) => {
    if (!Number.isFinite(edge.weight) || Math.abs(edge.weight) < minAbsoluteWeight) return;
    eligiblePairEdges += 1;
    if (edgeBudget === 0) return;
    const neighbor = neighborId(edge, centerId);
    const prior = selected.get(neighbor);
    if (prior) {
      if (compareEvidence(edge, prior, centerId) < 0) selected.set(neighbor, edge);
      return;
    }
    if (selected.size < edgeBudget) {
      selected.set(neighbor, edge);
      return;
    }
    let weakestNeighbor = "";
    let weakestEdge: GraphEdge | null = null;
    for (const [id, retained] of selected) {
      if (!weakestEdge || compareEvidence(retained, weakestEdge, centerId) > 0) {
        weakestNeighbor = id;
        weakestEdge = retained;
      }
    }
    if (weakestEdge && compareEvidence(edge, weakestEdge, centerId) < 0) {
      selected.delete(weakestNeighbor);
      selected.set(neighbor, edge);
    }
  };

  let center: GraphNode | null = null;
  if (isCompactGraphData(graph)) {
    const centerIndex = graph.anime.findIndex(([animeId]) => `anime:${animeId}` === centerId);
    if (centerIndex < 0) return null;
    const [animeId, title] = graph.anime[centerIndex];
    center = { id: `anime:${animeId}`, label: title, nodeType: "anime" };
    for (const [leftIndex, rightIndex, weight, support] of graph.aa) {
      if (leftIndex !== centerIndex && rightIndex !== centerIndex) continue;
      const left = graph.anime[leftIndex];
      const right = graph.anime[rightIndex];
      if (!left || !right) continue;
      offer({ id: `aa:${left[0]}:${right[0]}`, source: `anime:${left[0]}`,
        target: `anime:${right[0]}`, edgeType: "anime-anime", weight, support });
    }
  } else {
    center = graph.nodes.find((node) => node.id === centerId && node.nodeType === "anime") ?? null;
    if (!center) return null;
    for (const edge of graph.edges) {
      if (edge.edgeType !== "anime-anime" ||
          (edge.source !== centerId && edge.target !== centerId)) continue;
      offer(edge);
    }
  }
  const edges = [...selected.values()].sort((left, right) =>
    compareEvidence(left, right, centerId));
  if (!center) return null;
  const wanted = new Set(edges.map((edge) => neighborId(edge, centerId)));
  const neighbors = new Map<string, GraphNode>();
  if (isCompactGraphData(graph)) {
    for (const [animeId, title] of graph.anime) {
      const id = `anime:${animeId}`;
      if (wanted.has(id)) neighbors.set(id, { id, label: title, nodeType: "anime" });
    }
  } else {
    for (const node of graph.nodes) {
      if (wanted.has(node.id) && node.nodeType === "anime") neighbors.set(node.id, node);
    }
  }
  const nodes = [center, ...edges.flatMap((edge) => {
    const node = neighbors.get(neighborId(edge, centerId));
    return node ? [node] : [];
  })];
  return { center, nodes, edges, eligiblePairEdges,
    omittedByBudget: eligiblePairEdges - edges.length };
}
