import Graph from "graphology";
import forceAtlas2 from "graphology-layout-forceatlas2";
import Sigma from "sigma";
import "./style.css";

type NodeType = "user" | "anime";
type EdgeType = "user-anime" | "anime-anime";
type AppView = "recommendations" | "network";

interface GraphNode {
  id: string;
  label: string;
  nodeType: NodeType;
}

interface GraphEdge {
  id: string;
  source: string;
  target: string;
  edgeType: EdgeType;
  weight: number;
}

interface GraphData {
  generatedAt: string;
  userCount: number;
  animeCount: number;
  nodeCount: number;
  edgeCount: number;
  nodes: GraphNode[];
  edges: GraphEdge[];
}

interface ConnectedItem {
  nodeId: string;
  label: string;
  nodeType: NodeType;
  edgeType: EdgeType;
  weight: number;
}

interface AnimeInfo {
  nodeId: string;
  animeId: number;
  label: string;
}

interface RecommendationResult {
  anime: AnimeInfo;
  score: number;
  strongest: number;
  supportCount: number;
  contributions: RecommendationContribution[];
}

interface RecommendationContribution {
  watched: AnimeInfo;
  edgeWeight: number;
  weightFactor: number;
  weightedScore: number;
}

interface RecommendationIndex {
  animeList: AnimeInfo[];
  animeByNodeId: Map<string, AnimeInfo>;
  titleLookup: Map<string, AnimeInfo[]>;
  adjacency: Map<string, { otherNodeId: string; weight: number }[]>;
}

const FORCE_ATLAS_MAX_EDGES = 45000;
const FORCE_ATLAS_ITERATIONS = 180;
const INSPECT_MAX_ITEMS = 250;
const MAX_RECOMMENDATIONS = 40;
const MIN_WATCH_WEIGHT = 0.2;
const MAX_WATCH_WEIGHT = 3;
const WATCH_WEIGHT_STEP = 0.1;

const app = document.querySelector<HTMLDivElement>("#app");
if (!app) {
  throw new Error("Missing #app container");
}

app.innerHTML = `
  <div class="app-shell">
    <header class="topbar">
      <div>
        <h1>What Anime Should I Watch</h1>
        <p>Recommendation-first anime discovery powered by the rating network.</p>
      </div>
      <nav class="topnav" aria-label="Primary">
        <button id="nav-recommendations" class="nav-btn" type="button">Recommendations</button>
        <button id="nav-network" class="nav-btn" type="button">Network Explorer</button>
      </nav>
    </header>

    <main>
      <section id="view-recommendations" class="view">
        <div class="recommend-layout">
          <section class="card">
            <h2>Find Your Next Anime</h2>
            <p class="muted">Add anime you just watched, then get the best next picks from anime-to-anime edge strength.</p>

            <form id="add-anime-form" class="add-form">
              <input id="anime-input" type="text" list="anime-options" autocomplete="off" placeholder="Type an anime title" />
              <button type="submit">Add</button>
            </form>
            <datalist id="anime-options"></datalist>

            <p id="rec-message" class="rec-message"></p>

            <div class="selected-head">
              <h3>Watched List</h3>
              <button id="clear-watched" type="button" class="ghost-btn">Clear All</button>
            </div>
            <div id="selected-anime" class="selected-anime"></div>
          </section>

          <section class="card">
            <h2>Top Recommendations</h2>
            <p id="rec-summary" class="muted">Add at least one anime to start.</p>
            <ol id="rec-results" class="rec-results"></ol>
          </section>
        </div>
      </section>

      <section id="view-network" class="view" hidden>
        <div class="network-layout">
          <aside class="panel">
            <h2>Network Explorer</h2>
            <p class="muted">Interact with the graph, inspect node connections, and filter visible edges.</p>

            <div class="stats" id="stats"></div>

            <label class="control">
              <span>Min absolute edge weight</span>
              <input id="min-weight" type="range" min="0" max="4" value="0" step="0.05" />
              <output id="min-weight-value">0.00</output>
            </label>

            <label class="checkbox">
              <input id="toggle-anime-edges" type="checkbox" checked />
              <span>Show anime-to-anime edges</span>
            </label>

            <label class="checkbox">
              <input id="toggle-users" type="checkbox" />
              <span>Show user nodes + user-anime edges</span>
            </label>

            <section class="inspect">
              <div class="inspect-head">
                <h3>Inspect Node</h3>
                <button id="clear-selection" type="button" class="ghost-btn">Clear</button>
              </div>
              <p id="inspect-empty" class="inspect-empty">
                Click a node to see connected items sorted by weight.
              </p>
              <div id="inspect-content" class="inspect-content" hidden>
                <div id="inspect-meta" class="inspect-meta"></div>
                <div id="inspect-count" class="inspect-count"></div>
                <div id="inspect-values" class="inspect-values"></div>
                <ul id="inspect-list" class="inspect-list"></ul>
              </div>
            </section>
          </aside>

          <section class="graph-shell">
            <div id="graph"></div>
          </section>
        </div>
      </section>
    </main>
  </div>
`;

const navRecommendationsBtn = mustElement<HTMLButtonElement>("#nav-recommendations");
const navNetworkBtn = mustElement<HTMLButtonElement>("#nav-network");
const viewRecommendations = mustElement<HTMLElement>("#view-recommendations");
const viewNetwork = mustElement<HTMLElement>("#view-network");

const addAnimeForm = mustElement<HTMLFormElement>("#add-anime-form");
const animeInput = mustElement<HTMLInputElement>("#anime-input");
const animeOptions = mustElement<HTMLDataListElement>("#anime-options");
const recMessageEl = mustElement<HTMLParagraphElement>("#rec-message");
const selectedAnimeEl = mustElement<HTMLDivElement>("#selected-anime");
const clearWatchedBtn = mustElement<HTMLButtonElement>("#clear-watched");
const recSummaryEl = mustElement<HTMLParagraphElement>("#rec-summary");
const recResultsEl = mustElement<HTMLOListElement>("#rec-results");

const statsEl = mustElement<HTMLDivElement>("#stats");
const graphContainer = mustElement<HTMLDivElement>("#graph");
const minWeightInput = mustElement<HTMLInputElement>("#min-weight");
const minWeightValue = mustElement<HTMLOutputElement>("#min-weight-value");
const toggleAnimeEdges = mustElement<HTMLInputElement>("#toggle-anime-edges");
const toggleUsers = mustElement<HTMLInputElement>("#toggle-users");
const inspectEmptyEl = mustElement<HTMLParagraphElement>("#inspect-empty");
const inspectContentEl = mustElement<HTMLDivElement>("#inspect-content");
const inspectMetaEl = mustElement<HTMLDivElement>("#inspect-meta");
const inspectCountEl = mustElement<HTMLDivElement>("#inspect-count");
const inspectValuesEl = mustElement<HTMLDivElement>("#inspect-values");
const inspectListEl = mustElement<HTMLUListElement>("#inspect-list");
const clearSelectionBtn = mustElement<HTMLButtonElement>("#clear-selection");

let renderer: Sigma | null = null;
let selectedNodeId: string | null = null;
let currentGraph: Graph | null = null;
let activeView: AppView = "recommendations";

const graphData = await fetchGraph();
const recommendationIndex = buildRecommendationIndex(graphData);
const selectedAnimeNodeIds: string[] = [];
const selectedAnimeWeights = new Map<string, number>();

populateAnimeOptions(recommendationIndex.animeList, animeOptions);

const defaultMinWeight = getDefaultMinAnimeAnimeWeight(graphData, minWeightInput);
minWeightInput.value = defaultMinWeight.toFixed(2);
minWeightValue.textContent = defaultMinWeight.toFixed(2);

setActiveView(viewFromHash(), true);
updateRecommendations();

window.addEventListener("hashchange", () => {
  setActiveView(viewFromHash(), true);
});

navRecommendationsBtn.addEventListener("click", () => {
  setActiveView("recommendations", false);
});

navNetworkBtn.addEventListener("click", () => {
  setActiveView("network", false);
});

addAnimeForm.addEventListener("submit", (event) => {
  event.preventDefault();
  addAnimeFromInput();
});

animeInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    addAnimeFromInput();
  }
});

selectedAnimeEl.addEventListener("click", (event) => {
  const target = event.target;
  if (!(target instanceof HTMLElement)) {
    return;
  }
  const button = target.closest<HTMLButtonElement>("button[data-node-id]");
  if (!button) {
    return;
  }
  const nodeId = button.dataset.nodeId;
  if (!nodeId) {
    return;
  }
  removeSelectedAnime(nodeId);
});

selectedAnimeEl.addEventListener("input", (event) => {
  const target = event.target;
  if (!(target instanceof HTMLInputElement)) {
    return;
  }
  if (target.dataset.weightNodeId === undefined) {
    return;
  }

  const nodeId = target.dataset.weightNodeId;
  if (!nodeId) {
    return;
  }

  const parsed = Number.parseFloat(target.value);
  const clamped = clampWatchWeight(parsed);
  selectedAnimeWeights.set(nodeId, clamped);
  target.value = clamped.toFixed(1);

  const chip = target.closest(".chip-weighted");
  const valueEl = chip?.querySelector<HTMLOutputElement>(".chip-weight-value");
  if (valueEl) {
    valueEl.textContent = `${clamped.toFixed(1)}x`;
  }

  updateRecommendations();
});

clearWatchedBtn.addEventListener("click", () => {
  selectedAnimeNodeIds.splice(0, selectedAnimeNodeIds.length);
  selectedAnimeWeights.clear();
  recMessageEl.textContent = "";
  renderSelectedAnime();
  updateRecommendations();
});

clearSelectionBtn.addEventListener("click", () => {
  selectedNodeId = null;
  renderInspectPanel(null);
});

minWeightInput.addEventListener("input", () => {
  minWeightValue.textContent = Number.parseFloat(minWeightInput.value).toFixed(2);
  if (activeView === "network") {
    rerenderGraph();
  }
});

toggleAnimeEdges.addEventListener("change", () => {
  if (activeView === "network") {
    rerenderGraph();
  }
});

toggleUsers.addEventListener("change", () => {
  if (activeView === "network") {
    rerenderGraph();
  }
});

function setActiveView(view: AppView, fromHash: boolean): void {
  activeView = view;
  viewRecommendations.hidden = view !== "recommendations";
  viewNetwork.hidden = view !== "network";

  navRecommendationsBtn.classList.toggle("active", view === "recommendations");
  navNetworkBtn.classList.toggle("active", view === "network");

  if (!fromHash) {
    const nextHash = view === "network" ? "network" : "recommendations";
    if (window.location.hash !== `#${nextHash}`) {
      window.location.hash = nextHash;
    }
  }

  if (view === "network") {
    window.requestAnimationFrame(() => {
      rerenderGraph();
    });
  }
}

function viewFromHash(): AppView {
  return window.location.hash.toLowerCase() === "#network"
    ? "network"
    : "recommendations";
}

function addAnimeFromInput(): void {
  const raw = animeInput.value.trim();
  if (!raw) {
    recMessageEl.textContent = "Enter an anime title first.";
    return;
  }

  const anime = resolveAnimeInput(raw, recommendationIndex);
  if (!anime) {
    recMessageEl.textContent = `No anime match found for "${raw}".`;
    return;
  }

  if (selectedAnimeNodeIds.includes(anime.nodeId)) {
    recMessageEl.textContent = `${anime.label} is already in your watched list.`;
    animeInput.value = "";
    return;
  }

  selectedAnimeNodeIds.push(anime.nodeId);
  selectedAnimeWeights.set(anime.nodeId, 1);
  animeInput.value = "";
  recMessageEl.textContent = `Added: ${anime.label}`;
  renderSelectedAnime();
  updateRecommendations();
}

function removeSelectedAnime(nodeId: string): void {
  const index = selectedAnimeNodeIds.indexOf(nodeId);
  if (index < 0) {
    return;
  }
  selectedAnimeNodeIds.splice(index, 1);
  selectedAnimeWeights.delete(nodeId);
  recMessageEl.textContent = "";
  renderSelectedAnime();
  updateRecommendations();
}

function renderSelectedAnime(): void {
  if (selectedAnimeNodeIds.length === 0) {
    selectedAnimeEl.innerHTML = `<p class="muted">No anime added yet.</p>`;
    return;
  }

  const html = selectedAnimeNodeIds
    .map((nodeId) => recommendationIndex.animeByNodeId.get(nodeId))
    .filter((anime): anime is AnimeInfo => Boolean(anime))
    .map((anime) => {
      const weight = clampWatchWeight(selectedAnimeWeights.get(anime.nodeId) ?? 1);
      return `
      <div class="chip chip-weighted">
        <span class="chip-title">${escapeHtml(anime.label)}</span>
        <label class="chip-weight-control">
          <span>Weight</span>
          <input
            type="range"
            min="${MIN_WATCH_WEIGHT}"
            max="${MAX_WATCH_WEIGHT}"
            step="${WATCH_WEIGHT_STEP}"
            value="${weight.toFixed(1)}"
            data-weight-node-id="${anime.nodeId}"
            aria-label="Weight for ${escapeHtml(anime.label)}"
          />
          <output class="chip-weight-value">${weight.toFixed(1)}x</output>
        </label>
        <button type="button" data-node-id="${anime.nodeId}" aria-label="Remove ${escapeHtml(anime.label)}">x</button>
      </div>
    `;
    })
    .join("");

  selectedAnimeEl.innerHTML = html;
}

function updateRecommendations(): void {
  if (selectedAnimeNodeIds.length === 0) {
    recSummaryEl.textContent = "Add at least one anime to start.";
    recResultsEl.innerHTML = "";
    renderSelectedAnime();
    return;
  }

  const recommendations = buildRecommendations(
    selectedAnimeNodeIds,
    selectedAnimeWeights,
    recommendationIndex,
  );

  if (recommendations.length === 0) {
    recSummaryEl.textContent =
      "No positive recommendations found from the current watched list. Try adding more anime.";
    recResultsEl.innerHTML = "";
    return;
  }

  recSummaryEl.textContent = `Showing top ${Math.min(MAX_RECOMMENDATIONS, recommendations.length)} recommendations from ${recommendations.length} candidates.`;

  recResultsEl.innerHTML = recommendations
    .slice(0, MAX_RECOMMENDATIONS)
    .map(
      (item) => `
      <li class="rec-item">
        <div>
          <div class="rec-title">${escapeHtml(item.anime.label)}</div>
          <div class="rec-meta">Support edges: ${item.supportCount} | Strongest: ${formatWeight(item.strongest)}</div>
          <div class="rec-why">${escapeHtml(formatRecommendationWhy(item))}</div>
        </div>
        <div class="rec-score">${formatWeight(item.score)}</div>
      </li>
    `,
    )
    .join("");
}

function buildRecommendations(
  selectedNodeIds: string[],
  selectedWeights: Map<string, number>,
  index: RecommendationIndex,
): RecommendationResult[] {
  const selected = new Set(selectedNodeIds);
  const scored = new Map<
    string,
    {
      score: number;
      strongest: number;
      supportCount: number;
      sourceMap: Map<
        string,
        { edgeWeight: number; weightFactor: number; weightedScore: number }
      >;
    }
  >();

  for (const selectedNodeId of selectedNodeIds) {
    const weightFactor = clampWatchWeight(selectedWeights.get(selectedNodeId) ?? 1);
    const neighbors = index.adjacency.get(selectedNodeId) ?? [];
    for (const neighbor of neighbors) {
      if (selected.has(neighbor.otherNodeId)) {
        continue;
      }
      if (neighbor.weight <= 0) {
        continue;
      }

      const weightedScore = neighbor.weight * weightFactor;

      const current = scored.get(neighbor.otherNodeId);
      if (!current) {
        scored.set(neighbor.otherNodeId, {
          score: weightedScore,
          strongest: weightedScore,
          supportCount: 1,
          sourceMap: new Map([
            [
              selectedNodeId,
              {
                edgeWeight: neighbor.weight,
                weightFactor,
                weightedScore,
              },
            ],
          ]),
        });
        continue;
      }

      current.score += weightedScore;
      current.strongest = Math.max(current.strongest, weightedScore);
      current.supportCount += 1;
      current.sourceMap.set(selectedNodeId, {
        edgeWeight: neighbor.weight,
        weightFactor,
        weightedScore,
      });
    }
  }

  return [...scored.entries()]
    .map(([nodeId, aggregate]) => {
      const anime = index.animeByNodeId.get(nodeId);
      if (!anime) {
        return null;
      }
      const contributions = [...aggregate.sourceMap.entries()]
        .map(([watchedNodeId, source]) => {
          const watched = index.animeByNodeId.get(watchedNodeId);
          if (!watched) {
            return null;
          }
          return {
            watched,
            edgeWeight: source.edgeWeight,
            weightFactor: source.weightFactor,
            weightedScore: source.weightedScore,
          } satisfies RecommendationContribution;
        })
        .filter((value): value is RecommendationContribution => value !== null)
        .sort((left, right) => right.weightedScore - left.weightedScore);

      return {
        anime,
        score: aggregate.score,
        strongest: aggregate.strongest,
        supportCount: aggregate.supportCount,
        contributions,
      } satisfies RecommendationResult;
    })
    .filter((value): value is RecommendationResult => value !== null)
    .sort((left, right) => {
      if (right.score !== left.score) {
        return right.score - left.score;
      }
      if (right.supportCount !== left.supportCount) {
        return right.supportCount - left.supportCount;
      }
      return right.strongest - left.strongest;
    });
}

function resolveAnimeInput(
  raw: string,
  index: RecommendationIndex,
): AnimeInfo | null {
  const normalized = normalizeTitle(raw);
  if (!normalized) {
    return null;
  }

  const exact = index.titleLookup.get(normalized);
  if (exact && exact.length > 0) {
    return exact[0];
  }

  if (/^anime:\d+$/i.test(raw)) {
    const byNodeId = index.animeByNodeId.get(raw.toLowerCase());
    if (byNodeId) {
      return byNodeId;
    }
  }

  if (/^\d+$/.test(raw)) {
    const byAnimeId = index.animeList.find((item) => item.animeId === Number.parseInt(raw, 10));
    if (byAnimeId) {
      return byAnimeId;
    }
  }

  return index.animeList.find((item) => normalizeTitle(item.label).includes(normalized)) ?? null;
}

function populateAnimeOptions(animeList: AnimeInfo[], datalist: HTMLDataListElement): void {
  const sorted = [...animeList].sort((left, right) => left.label.localeCompare(right.label));
  datalist.innerHTML = sorted
    .map((anime) => `<option value="${escapeHtml(anime.label)}"></option>`)
    .join("");
}

function buildRecommendationIndex(graphDataValue: GraphData): RecommendationIndex {
  const animeList: AnimeInfo[] = [];
  const animeByNodeId = new Map<string, AnimeInfo>();
  const titleLookup = new Map<string, AnimeInfo[]>();
  const adjacency = new Map<string, { otherNodeId: string; weight: number }[]>();

  for (const node of graphDataValue.nodes) {
    if (node.nodeType !== "anime") {
      continue;
    }

    const animeId = parseAnimeId(node.id);
    const anime: AnimeInfo = {
      nodeId: node.id,
      animeId,
      label: node.label,
    };

    animeList.push(anime);
    animeByNodeId.set(node.id, anime);

    const normalizedTitle = normalizeTitle(node.label);
    const existing = titleLookup.get(normalizedTitle);
    if (!existing) {
      titleLookup.set(normalizedTitle, [anime]);
    } else {
      existing.push(anime);
    }
  }

  for (const edge of graphDataValue.edges) {
    if (edge.edgeType !== "anime-anime") {
      continue;
    }
    if (!animeByNodeId.has(edge.source) || !animeByNodeId.has(edge.target)) {
      continue;
    }

    pushAdjacency(adjacency, edge.source, edge.target, edge.weight);
    pushAdjacency(adjacency, edge.target, edge.source, edge.weight);
  }

  return {
    animeList,
    animeByNodeId,
    titleLookup,
    adjacency,
  };
}

function pushAdjacency(
  adjacency: Map<string, { otherNodeId: string; weight: number }[]>,
  source: string,
  target: string,
  weight: number,
): void {
  const list = adjacency.get(source);
  if (!list) {
    adjacency.set(source, [{ otherNodeId: target, weight }]);
    return;
  }
  list.push({ otherNodeId: target, weight });
}

function parseAnimeId(nodeId: string): number {
  const value = nodeId.startsWith("anime:") ? nodeId.slice("anime:".length) : nodeId;
  const parsed = Number.parseInt(value, 10);
  return Number.isNaN(parsed) ? -1 : parsed;
}

function normalizeTitle(value: string): string {
  return value.trim().toLowerCase().replace(/\s+/g, " ");
}

function rerenderGraph(): void {
  const minWeight = Number.parseFloat(minWeightInput.value);
  minWeightValue.textContent = minWeight.toFixed(2);
  renderGraph(
    graphData,
    minWeight,
    toggleAnimeEdges.checked,
    toggleUsers.checked,
  );
}

function renderGraph(
  graphDataValue: GraphData,
  minAbsoluteWeight: number,
  showAnimeAnimeEdges: boolean,
  showUsers: boolean,
): void {
  const graph = new Graph({ multi: true, type: "undirected" });

  for (const node of graphDataValue.nodes) {
    if (!showUsers && node.nodeType === "user") {
      continue;
    }

    const isUser = node.nodeType === "user";
    graph.addNode(node.id, {
      label: node.label,
      nodeType: node.nodeType,
      size: isUser ? 5.2 : 2.8,
      color: isUser ? "#ff8a00" : "#0f8b8d",
      x: Math.random(),
      y: Math.random(),
    });
  }

  for (const edge of graphDataValue.edges) {
    if (!showUsers && edge.edgeType === "user-anime") {
      continue;
    }

    const edgePassesTypeFilter =
      showAnimeAnimeEdges || edge.edgeType !== "anime-anime";
    const edgePassesWeightFilter =
      Math.abs(edge.weight) >= minAbsoluteWeight || edge.edgeType === "user-anime";

    if (!edgePassesTypeFilter || !edgePassesWeightFilter) {
      continue;
    }

    if (!graph.hasNode(edge.source) || !graph.hasNode(edge.target)) {
      continue;
    }

    graph.addEdgeWithKey(edge.id, edge.source, edge.target, {
      size: edge.edgeType === "user-anime" ? 1.4 : 0.7,
      color: edge.edgeType === "user-anime" ? "#f4d35e88" : "#6fffe988",
      weight: Math.max(Math.abs(edge.weight), 0.01),
      signedWeight: edge.weight,
      edgeType: edge.edgeType,
    });
  }

  applyLayout(graph);
  currentGraph = graph;

  if (renderer) {
    renderer.kill();
  }

  renderer = new Sigma(graph, graphContainer, {
    renderEdgeLabels: false,
    labelRenderedSizeThreshold: 14,
    allowInvalidContainer: false,
  });

  renderer.on("clickNode", (event) => {
    selectedNodeId = event.node;
    renderInspectPanel(event.node);
  });

  renderer.on("clickStage", () => {
    selectedNodeId = null;
    renderInspectPanel(null);
  });

  const visibleUsers = countNodesByType(graph, "user");
  const visibleAnime = graph.order - visibleUsers;

  statsEl.innerHTML = [
    statLine("Generated", new Date(graphDataValue.generatedAt).toLocaleString()),
    statLine("Visible users", `${visibleUsers} / ${graphDataValue.userCount}`),
    statLine("Visible anime", `${visibleAnime} / ${graphDataValue.animeCount}`),
    statLine("Visible nodes", String(graph.order)),
    statLine("Visible edges", String(graph.size)),
  ].join("");

  if (selectedNodeId && graph.hasNode(selectedNodeId)) {
    renderInspectPanel(selectedNodeId);
  } else {
    selectedNodeId = null;
    renderInspectPanel(null);
  }
}

function statLine(label: string, value: string): string {
  return `<div class="stat-row"><span>${label}</span><strong>${value}</strong></div>`;
}

async function fetchGraph(): Promise<GraphData> {
  const response = await fetch("./data/graph.json");
  if (!response.ok) {
    throw new Error(`Unable to load graph.json (${response.status})`);
  }
  return (await response.json()) as GraphData;
}

function mustElement<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (!element) {
    throw new Error(`Missing element ${selector}`);
  }
  return element;
}

function applyLayout(graph: Graph): void {
  if (graph.order === 0) {
    return;
  }

  try {
    if (graph.size > FORCE_ATLAS_MAX_EDGES) {
      assignRingLayout(graph);
    } else {
      forceAtlas2.assign(graph, {
        iterations: FORCE_ATLAS_ITERATIONS,
        settings: forceAtlas2.inferSettings(graph),
      });
    }
  } catch (error) {
    console.warn("Graph layout failed; using fallback ring layout.", error);
    assignRingLayout(graph);
  }

  sanitizeCoordinates(graph);
}

function assignRingLayout(graph: Graph): void {
  const userNodes: string[] = [];
  const animeNodes: string[] = [];

  graph.forEachNode((node, attributes) => {
    if (attributes.nodeType === "user") {
      userNodes.push(node);
    } else {
      animeNodes.push(node);
    }
  });

  for (let i = 0; i < userNodes.length; i += 1) {
    const node = userNodes[i];
    const angle = ((i + 1) / Math.max(userNodes.length, 1)) * Math.PI * 2;
    graph.mergeNodeAttributes(node, {
      x: Math.cos(angle) * 1.25,
      y: Math.sin(angle) * 1.25,
    });
  }

  for (let i = 0; i < animeNodes.length; i += 1) {
    const node = animeNodes[i];
    const jitter = hashToUnit(node);
    const angle = (i / Math.max(animeNodes.length, 1)) * Math.PI * 2 + jitter * 0.18;
    const radius = 0.72 + jitter * 0.28;
    graph.mergeNodeAttributes(node, {
      x: Math.cos(angle) * radius,
      y: Math.sin(angle) * radius,
    });
  }
}

function sanitizeCoordinates(graph: Graph): void {
  let index = 0;
  graph.forEachNode((node, attributes) => {
    const x = attributes.x as number | undefined;
    const y = attributes.y as number | undefined;
    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      const angle = (index / Math.max(graph.order, 1)) * Math.PI * 2;
      graph.mergeNodeAttributes(node, {
        x: Math.cos(angle) * 0.8,
        y: Math.sin(angle) * 0.8,
      });
    }
    index += 1;
  });
}

function hashToUnit(value: string): number {
  let hash = 2166136261;
  for (let i = 0; i < value.length; i += 1) {
    hash ^= value.charCodeAt(i);
    hash += (hash << 1) + (hash << 4) + (hash << 7) + (hash << 8) + (hash << 24);
  }
  return ((hash >>> 0) % 10000) / 10000;
}

function renderInspectPanel(nodeId: string | null): void {
  if (!currentGraph || !nodeId || !currentGraph.hasNode(nodeId)) {
    inspectEmptyEl.hidden = false;
    inspectContentEl.hidden = true;
    inspectMetaEl.textContent = "";
    inspectCountEl.textContent = "";
    inspectValuesEl.innerHTML = "";
    inspectListEl.innerHTML = "";
    return;
  }

  const nodeAttrs = currentGraph.getNodeAttributes(nodeId) as Record<string, unknown>;
  const label = typeof nodeAttrs.label === "string" ? nodeAttrs.label : nodeId;
  const nodeType = nodeAttrs.nodeType === "user" ? "user" : ("anime" as NodeType);

  const connections = getConnectedItems(currentGraph, nodeId).sort(
    (left, right) => right.weight - left.weight,
  );

  inspectEmptyEl.hidden = true;
  inspectContentEl.hidden = false;
  inspectMetaEl.innerHTML = `
    <div class="inspect-title">${escapeHtml(label)}</div>
    <div class="inspect-sub">${nodeType} | ${escapeHtml(nodeId)}</div>
  `;

  const connectionCount = connections.length;
  const userConnections = connections.filter((item) => item.nodeType === "user").length;
  const animeConnections = connectionCount - userConnections;
  const positiveConnections = connections.filter((item) => item.weight > 0).length;
  const negativeConnections = connections.filter((item) => item.weight < 0).length;
  const sumWeight = connections.reduce((sum, item) => sum + item.weight, 0);
  const avgWeight = connectionCount > 0 ? sumWeight / connectionCount : 0;
  const strongestWeight = connectionCount > 0 ? connections[0].weight : 0;
  const weakestWeight = connectionCount > 0 ? connections[connectionCount - 1].weight : 0;

  inspectCountEl.textContent = `Connected items: ${connectionCount} (sorted by weight desc)`;
  inspectValuesEl.innerHTML = [
    valueRow("Connected users", String(userConnections)),
    valueRow("Connected anime", String(animeConnections)),
    valueRow("Positive edges", String(positiveConnections)),
    valueRow("Negative edges", String(negativeConnections)),
    valueRow("Average weight", formatWeight(avgWeight)),
    valueRow("Strongest edge", formatWeight(strongestWeight)),
    valueRow("Weakest edge", formatWeight(weakestWeight)),
  ].join("");

  const visible = connections.slice(0, INSPECT_MAX_ITEMS);
  const truncationNotice =
    visible.length < connectionCount
      ? `<li class="inspect-trunc">Showing top ${visible.length} by weight</li>`
      : "";

  const listHtml = visible
    .map((item) => {
      const weight = formatWeight(item.weight);
      return `
        <li class="inspect-item">
          <div class="inspect-item-main">
            <span class="inspect-item-label">${escapeHtml(item.label)}</span>
            <span class="inspect-item-type">${item.nodeType}</span>
          </div>
          <div class="inspect-item-meta">
            <span>${item.edgeType}</span>
            <strong>${weight}</strong>
          </div>
        </li>
      `;
    })
    .join("");

  inspectListEl.innerHTML = truncationNotice + listHtml;
}

function getConnectedItems(graph: Graph, nodeId: string): ConnectedItem[] {
  const items: ConnectedItem[] = [];

  graph.forEachEdge(
    nodeId,
    (
      _edgeKey,
      attributes,
      source,
      target,
      sourceAttributes,
      targetAttributes,
    ) => {
      const otherNode = source === nodeId ? target : source;
      const otherAttributes = (
        source === nodeId ? targetAttributes : sourceAttributes
      ) as Record<string, unknown>;
      const edgeAttributes = attributes as Record<string, unknown>;

      const label =
        typeof otherAttributes.label === "string"
          ? otherAttributes.label
          : otherNode;
      const nodeType =
        otherAttributes.nodeType === "user"
          ? "user"
          : ("anime" as NodeType);
      const edgeType =
        edgeAttributes.edgeType === "anime-anime"
          ? "anime-anime"
          : ("user-anime" as EdgeType);

      const signedWeight = edgeAttributes.signedWeight;
      const fallbackWeight = edgeAttributes.weight;
      const weight =
        typeof signedWeight === "number"
          ? signedWeight
          : typeof fallbackWeight === "number"
            ? fallbackWeight
            : 0;

      items.push({
        nodeId: otherNode,
        label,
        nodeType,
        edgeType,
        weight,
      });
    },
  );

  return items;
}

function escapeHtml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function formatWeight(value: number): string {
  const normalized = Math.abs(value) < 0.0005 ? 0 : value;
  const rounded = normalized.toFixed(3);
  return normalized > 0 ? `+${rounded}` : rounded;
}

function formatRecommendationWhy(result: RecommendationResult): string {
  if (result.contributions.length === 0) {
    return "Why: no direct positive contributing anime found.";
  }

  const top = result.contributions.slice(0, 3);
  const summary = top
    .map(
      (item) =>
        `${item.watched.label} (${formatWeight(item.weightedScore)} from ${formatWeight(item.edgeWeight)} x ${item.weightFactor.toFixed(1)})`,
    )
    .join(" | ");

  const truncated =
    result.contributions.length > top.length
      ? ` | +${result.contributions.length - top.length} more`
      : "";

  return `Why: ${summary}${truncated}`;
}

function clampWatchWeight(value: number): number {
  if (!Number.isFinite(value)) {
    return 1;
  }
  return Math.min(Math.max(value, MIN_WATCH_WEIGHT), MAX_WATCH_WEIGHT);
}

function valueRow(label: string, value: string): string {
  return `<div class="inspect-value-row"><span>${label}</span><strong>${value}</strong></div>`;
}

function countNodesByType(graph: Graph, type: NodeType): number {
  let count = 0;
  graph.forEachNode((_node, attributes) => {
    if (attributes.nodeType === type) {
      count += 1;
    }
  });
  return count;
}

function getDefaultMinAnimeAnimeWeight(
  graphDataValue: GraphData,
  input: HTMLInputElement,
): number {
  let sumAbsWeight = 0;
  let count = 0;

  for (const edge of graphDataValue.edges) {
    if (edge.edgeType !== "anime-anime") {
      continue;
    }
    if (!Number.isFinite(edge.weight)) {
      continue;
    }
    sumAbsWeight += Math.abs(edge.weight);
    count += 1;
  }

  const min = Number.parseFloat(input.min || "0");
  const max = Number.parseFloat(input.max || "4");
  const step = Number.parseFloat(input.step || "0");

  if (count === 0) {
    return min;
  }

  let value = sumAbsWeight / count;
  if (Number.isNaN(value) || !Number.isFinite(value)) {
    value = min;
  }

  value = Math.min(Math.max(value, min), max);
  if (step > 0 && Number.isFinite(step)) {
    value = Math.round(value / step) * step;
  }

  return Number(value.toFixed(2));
}
