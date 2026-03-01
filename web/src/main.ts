import Graph from "graphology";
import forceAtlas2 from "graphology-layout-forceatlas2";
import Sigma from "sigma";
import "./style.css";

type NodeType = "user" | "anime";
type EdgeType = "user-anime" | "anime-anime";
type AppView = "recommendations" | "network";
type RecommendationMode = "graph" | "model" | "hybrid";

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

type CompactAnimeEntry = [animeId: number, title: string];
type CompactUserAnimeEdge = [userIndex: number, animeIndex: number, weight: number];
type CompactAnimeAnimeEdge = [
  leftAnimeIndex: number,
  rightAnimeIndex: number,
  weight: number,
];

interface CompactGraphData {
  format: "graph-compact-v1";
  generatedAt: string;
  userIds: string[];
  anime: CompactAnimeEntry[];
  ua: CompactUserAnimeEdge[];
  aa: CompactAnimeAnimeEdge[];
  userCount: number;
  animeCount: number;
  nodeCount: number;
  edgeCount: number;
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
  animeByAnimeId: Map<number, AnimeInfo>;
  titleLookup: Map<string, AnimeInfo[]>;
  adjacency: Map<string, { otherNodeId: string; weight: number }[]>;
}

interface ModelRecommendationAnime {
  animeId: number;
  title: string;
  bias: number;
  embedding: number[];
}

interface ModelRecommendationData {
  generatedAt: string;
  globalMean: number;
  factors: number;
  anime: ModelRecommendationAnime[];
}

interface CompactModelRecommendationData {
  format: "model-mf-compact-v1";
  generatedAt: string;
  globalMean: number;
  factors: number;
  animeIds: number[];
  titles: string[];
  biases: number[];
  embeddings: number[][];
}

interface ModelRecommendationIndex {
  generatedAt: string;
  factors: number;
  globalMean: number;
  animeByAnimeId: Map<number, ModelRecommendationAnime>;
}

const FORCE_ATLAS_MAX_EDGES = 45000;
const FORCE_ATLAS_ITERATIONS = 180;
const INSPECT_MAX_ITEMS = 250;
const MAX_RECOMMENDATIONS = 40;
const MIN_WATCH_WEIGHT = 0.2;
const MAX_WATCH_WEIGHT = 3;
const WATCH_WEIGHT_STEP = 0.1;
const MIN_MODEL_BLEND_WEIGHT = 0;
const MAX_MODEL_BLEND_WEIGHT = 1;
const MODEL_BLEND_WEIGHT_STEP = 0.05;
const RECOMMENDATION_STATE_STORAGE_KEY = "wasiw.recommendationState.v1";

interface StoredRecommendationState {
  version: number;
  mode: RecommendationMode;
  selected: { nodeId: string; weight: number }[];
  modelBlendWeight?: number;
  includeCandidates?: string[];
  excludeCandidates?: string[];
}

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
            <p class="muted">Add anime you just watched, then rank next picks with either graph edges or the trained ML model.</p>

            <label class="rec-engine-control" for="rec-method">
              <span>Recommendation engine</span>
              <select id="rec-method">
                <option value="graph" selected>Graph (anime-to-anime edges)</option>
                <option value="model">ML Model (matrix factorization)</option>
                <option value="hybrid">Hybrid (blend graph + ML)</option>
              </select>
            </label>
            <label id="rec-blend-control" class="rec-blend-control" for="rec-blend" hidden>
              <span>Model blend weight</span>
              <input id="rec-blend" type="range" min="${MIN_MODEL_BLEND_WEIGHT}" max="${MAX_MODEL_BLEND_WEIGHT}" step="${MODEL_BLEND_WEIGHT_STEP}" value="0.50" />
              <output id="rec-blend-value">50% model / 50% graph</output>
            </label>
            <p id="rec-engine-status" class="rec-engine-status">Using graph recommendations.</p>

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

            <div class="selected-head">
              <h3>Include Candidates</h3>
              <button id="clear-include" type="button" class="ghost-btn">Clear</button>
            </div>
            <form id="add-include-form" class="add-form add-form-compact">
              <input id="include-input" type="text" list="anime-options" autocomplete="off" placeholder="Add anime to force-include" />
              <button type="submit">Add</button>
            </form>
            <div id="include-anime" class="selected-anime"></div>

            <div class="selected-head">
              <h3>Exclude Candidates</h3>
              <button id="clear-exclude" type="button" class="ghost-btn">Clear</button>
            </div>
            <form id="add-exclude-form" class="add-form add-form-compact">
              <input id="exclude-input" type="text" list="anime-options" autocomplete="off" placeholder="Add anime to exclude" />
              <button type="submit">Add</button>
            </form>
            <div id="exclude-anime" class="selected-anime"></div>
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

            <section class="network-search">
              <h3>Search In Graph</h3>
              <form id="network-search-form" class="network-search-form">
                <input id="network-search-input" type="text" list="network-node-options" placeholder="Title, anime:ID, or user:ID" />
                <button type="submit">Find</button>
              </form>
              <datalist id="network-node-options"></datalist>
              <p id="network-search-message" class="network-search-message"></p>
            </section>

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
const recMethodSelect = mustElement<HTMLSelectElement>("#rec-method");
const recBlendControl = mustElement<HTMLLabelElement>("#rec-blend-control");
const recBlendInput = mustElement<HTMLInputElement>("#rec-blend");
const recBlendValueEl = mustElement<HTMLOutputElement>("#rec-blend-value");
const recEngineStatusEl = mustElement<HTMLParagraphElement>("#rec-engine-status");
const recMessageEl = mustElement<HTMLParagraphElement>("#rec-message");
const selectedAnimeEl = mustElement<HTMLDivElement>("#selected-anime");
const clearWatchedBtn = mustElement<HTMLButtonElement>("#clear-watched");
const addIncludeForm = mustElement<HTMLFormElement>("#add-include-form");
const includeInput = mustElement<HTMLInputElement>("#include-input");
const includeAnimeEl = mustElement<HTMLDivElement>("#include-anime");
const clearIncludeBtn = mustElement<HTMLButtonElement>("#clear-include");
const addExcludeForm = mustElement<HTMLFormElement>("#add-exclude-form");
const excludeInput = mustElement<HTMLInputElement>("#exclude-input");
const excludeAnimeEl = mustElement<HTMLDivElement>("#exclude-anime");
const clearExcludeBtn = mustElement<HTMLButtonElement>("#clear-exclude");
const recSummaryEl = mustElement<HTMLParagraphElement>("#rec-summary");
const recResultsEl = mustElement<HTMLOListElement>("#rec-results");

const statsEl = mustElement<HTMLDivElement>("#stats");
const graphContainer = mustElement<HTMLDivElement>("#graph");
const minWeightInput = mustElement<HTMLInputElement>("#min-weight");
const minWeightValue = mustElement<HTMLOutputElement>("#min-weight-value");
const toggleAnimeEdges = mustElement<HTMLInputElement>("#toggle-anime-edges");
const toggleUsers = mustElement<HTMLInputElement>("#toggle-users");
const networkSearchForm = mustElement<HTMLFormElement>("#network-search-form");
const networkSearchInput = mustElement<HTMLInputElement>("#network-search-input");
const networkNodeOptions = mustElement<HTMLDataListElement>("#network-node-options");
const networkSearchMessage = mustElement<HTMLParagraphElement>("#network-search-message");
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
let recommendationMode: RecommendationMode = "graph";
let modelBlendWeight = 0.5;
let recommendationRunId = 0;
let modelRecommendationIndexPromise: Promise<ModelRecommendationIndex | null> | null = null;

const graphData = await fetchGraph();
const recommendationIndex = buildRecommendationIndex(graphData);
const selectedAnimeNodeIds: string[] = [];
const selectedAnimeWeights = new Map<string, number>();
const includeCandidateNodeIds: string[] = [];
const excludeCandidateNodeIds: string[] = [];
const persistedState = loadRecommendationState(recommendationIndex);

for (const entry of persistedState.selected) {
  selectedAnimeNodeIds.push(entry.nodeId);
  selectedAnimeWeights.set(entry.nodeId, clampWatchWeight(entry.weight));
}
for (const nodeId of persistedState.includeCandidates) {
  includeCandidateNodeIds.push(nodeId);
}
for (const nodeId of persistedState.excludeCandidates) {
  excludeCandidateNodeIds.push(nodeId);
}
recommendationMode = persistedState.mode;
modelBlendWeight = clampModelBlendWeight(persistedState.modelBlendWeight ?? 0.5);
recMethodSelect.value = recommendationMode;
recBlendInput.value = modelBlendWeight.toFixed(2);
renderModelBlendValue();
setBlendControlVisibility();

populateAnimeOptions(recommendationIndex.animeList, animeOptions);
populateNetworkNodeOptions(graphData.nodes, networkNodeOptions);
renderSelectedAnime();
renderIncludeCandidates();
renderExcludeCandidates();

const defaultMinWeight = getDefaultMinAnimeAnimeWeight(graphData, minWeightInput);
minWeightInput.value = defaultMinWeight.toFixed(2);
minWeightValue.textContent = defaultMinWeight.toFixed(2);

setActiveView(viewFromHash(), true);
void updateRecommendations();

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

addIncludeForm.addEventListener("submit", (event) => {
  event.preventDefault();
  addCandidateFromInput(includeInput, includeCandidateNodeIds, "include");
});

addExcludeForm.addEventListener("submit", (event) => {
  event.preventDefault();
  addCandidateFromInput(excludeInput, excludeCandidateNodeIds, "exclude");
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

includeAnimeEl.addEventListener("click", (event) => {
  const target = event.target;
  if (!(target instanceof HTMLElement)) {
    return;
  }
  const button = target.closest<HTMLButtonElement>("button[data-include-node-id]");
  if (!button) {
    return;
  }
  const nodeId = button.dataset.includeNodeId;
  if (!nodeId) {
    return;
  }
  removeCandidateNodeId(nodeId, includeCandidateNodeIds, "include");
});

excludeAnimeEl.addEventListener("click", (event) => {
  const target = event.target;
  if (!(target instanceof HTMLElement)) {
    return;
  }
  const button = target.closest<HTMLButtonElement>("button[data-exclude-node-id]");
  if (!button) {
    return;
  }
  const nodeId = button.dataset.excludeNodeId;
  if (!nodeId) {
    return;
  }
  removeCandidateNodeId(nodeId, excludeCandidateNodeIds, "exclude");
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
  persistRecommendationState();
  target.value = clamped.toFixed(1);

  const chip = target.closest(".chip-weighted");
  const valueEl = chip?.querySelector<HTMLOutputElement>(".chip-weight-value");
  if (valueEl) {
    valueEl.textContent = `${clamped.toFixed(1)}x`;
  }

  void updateRecommendations();
});

clearWatchedBtn.addEventListener("click", () => {
  selectedAnimeNodeIds.splice(0, selectedAnimeNodeIds.length);
  selectedAnimeWeights.clear();
  persistRecommendationState();
  recMessageEl.textContent = "";
  renderSelectedAnime();
  void updateRecommendations();
});

clearIncludeBtn.addEventListener("click", () => {
  includeCandidateNodeIds.splice(0, includeCandidateNodeIds.length);
  persistRecommendationState();
  renderIncludeCandidates();
  void updateRecommendations();
});

clearExcludeBtn.addEventListener("click", () => {
  excludeCandidateNodeIds.splice(0, excludeCandidateNodeIds.length);
  persistRecommendationState();
  renderExcludeCandidates();
  void updateRecommendations();
});

recMethodSelect.addEventListener("change", () => {
  recommendationMode = parseRecommendationMode(recMethodSelect.value);
  setBlendControlVisibility();
  persistRecommendationState();
  void updateRecommendations();
});

recBlendInput.addEventListener("input", () => {
  modelBlendWeight = clampModelBlendWeight(Number.parseFloat(recBlendInput.value));
  recBlendInput.value = modelBlendWeight.toFixed(2);
  renderModelBlendValue();
  persistRecommendationState();
  if (recommendationMode === "hybrid") {
    void updateRecommendations();
  }
});

clearSelectionBtn.addEventListener("click", () => {
  selectedNodeId = null;
  networkSearchMessage.textContent = "";
  renderInspectPanel(null);
  if (renderer) {
    renderer.refresh();
  }
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

networkSearchForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const query = networkSearchInput.value.trim();
  if (!query) {
    networkSearchMessage.textContent = "Enter a node query first.";
    return;
  }
  const match = resolveNetworkNodeQuery(query, graphData);
  if (!match) {
    networkSearchMessage.textContent = `No node match found for "${query}".`;
    return;
  }
  networkSearchMessage.textContent = `Focused: ${match.label} (${match.id})`;
  selectNodeAndFocus(match.id);
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
  persistRecommendationState();
  animeInput.value = "";
  recMessageEl.textContent = `Added: ${anime.label}`;
  renderSelectedAnime();
  void updateRecommendations();
}

function removeSelectedAnime(nodeId: string): void {
  const index = selectedAnimeNodeIds.indexOf(nodeId);
  if (index < 0) {
    return;
  }
  selectedAnimeNodeIds.splice(index, 1);
  selectedAnimeWeights.delete(nodeId);
  persistRecommendationState();
  recMessageEl.textContent = "";
  renderSelectedAnime();
  void updateRecommendations();
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

function addCandidateFromInput(
  input: HTMLInputElement,
  targetList: string[],
  mode: "include" | "exclude",
): void {
  const raw = input.value.trim();
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
    input.value = "";
    return;
  }
  if (targetList.includes(anime.nodeId)) {
    recMessageEl.textContent = `${anime.label} is already in your ${mode} list.`;
    input.value = "";
    return;
  }

  targetList.push(anime.nodeId);
  if (mode === "include") {
    const index = excludeCandidateNodeIds.indexOf(anime.nodeId);
    if (index >= 0) {
      excludeCandidateNodeIds.splice(index, 1);
    }
  } else {
    const index = includeCandidateNodeIds.indexOf(anime.nodeId);
    if (index >= 0) {
      includeCandidateNodeIds.splice(index, 1);
    }
  }

  persistRecommendationState();
  input.value = "";
  recMessageEl.textContent = `Added ${anime.label} to ${mode} list.`;
  renderIncludeCandidates();
  renderExcludeCandidates();
  void updateRecommendations();
}

function removeCandidateNodeId(
  nodeId: string,
  targetList: string[],
  mode: "include" | "exclude",
): void {
  const index = targetList.indexOf(nodeId);
  if (index < 0) {
    return;
  }
  targetList.splice(index, 1);
  persistRecommendationState();
  recMessageEl.textContent = "";
  if (mode === "include") {
    renderIncludeCandidates();
  } else {
    renderExcludeCandidates();
  }
  void updateRecommendations();
}

function renderIncludeCandidates(): void {
  renderCandidateChips(includeAnimeEl, includeCandidateNodeIds, "include");
}

function renderExcludeCandidates(): void {
  renderCandidateChips(excludeAnimeEl, excludeCandidateNodeIds, "exclude");
}

function renderCandidateChips(
  container: HTMLDivElement,
  nodeIds: string[],
  mode: "include" | "exclude",
): void {
  if (nodeIds.length === 0) {
    container.innerHTML = `<p class="muted">No anime in ${mode} list.</p>`;
    return;
  }

  const dataAttrName = mode === "include" ? "data-include-node-id" : "data-exclude-node-id";
  const html = nodeIds
    .map((nodeId) => recommendationIndex.animeByNodeId.get(nodeId))
    .filter((anime): anime is AnimeInfo => Boolean(anime))
    .map(
      (anime) => `
      <div class="chip">
        <span class="chip-title">${escapeHtml(anime.label)}</span>
        <button type="button" ${dataAttrName}="${anime.nodeId}" aria-label="Remove ${escapeHtml(anime.label)}">x</button>
      </div>
    `,
    )
    .join("");
  container.innerHTML = html;
}

async function updateRecommendations(): Promise<void> {
  const runId = ++recommendationRunId;

  if (selectedAnimeNodeIds.length === 0) {
    recSummaryEl.textContent = "Add at least one anime to start.";
    recEngineStatusEl.textContent =
      recommendationMode === "graph"
        ? "Using graph recommendations."
        : recommendationMode === "model"
          ? "Using ML model recommendations."
          : `Using hybrid recommendations (${Math.round(modelBlendWeight * 100)}% model).`;
    recResultsEl.innerHTML = "";
    renderSelectedAnime();
    return;
  }

  let recommendations: RecommendationResult[] = [];

  if (recommendationMode === "graph") {
    recEngineStatusEl.textContent = "Using graph recommendations.";
    recommendations = buildGraphRecommendations(
      selectedAnimeNodeIds,
      selectedAnimeWeights,
      recommendationIndex,
    );
  } else if (recommendationMode === "model") {
    recEngineStatusEl.textContent = "Loading ML model recommendations...";
    const modelIndex = await ensureModelRecommendationIndex();
    if (runId !== recommendationRunId) {
      return;
    }
    if (!modelIndex) {
      recEngineStatusEl.textContent =
        "ML model data not found (expected model-mf-web.compact.json(.gz) or model-mf-web.json(.gz)).";
      recSummaryEl.textContent =
        "Model recommendations are unavailable until model data is exported to the web data folder.";
      recResultsEl.innerHTML = "";
      return;
    }
    recEngineStatusEl.textContent = `Using ML model recommendations (${modelIndex.factors} factors).`;
    recommendations = buildModelRecommendations(
      selectedAnimeNodeIds,
      selectedAnimeWeights,
      recommendationIndex,
      modelIndex,
    );
  } else {
    recEngineStatusEl.textContent = "Loading hybrid recommendations...";
    const modelIndex = await ensureModelRecommendationIndex();
    if (runId !== recommendationRunId) {
      return;
    }
    if (!modelIndex) {
      recEngineStatusEl.textContent =
        "ML model data not found for hybrid mode (expected model-mf-web.compact.json(.gz) or model-mf-web.json(.gz)).";
      recSummaryEl.textContent =
        "Hybrid mode needs model data. Export model data or switch to graph-only mode.";
      recResultsEl.innerHTML = "";
      return;
    }
    const graphRecommendations = buildGraphRecommendations(
      selectedAnimeNodeIds,
      selectedAnimeWeights,
      recommendationIndex,
    );
    const modelRecommendations = buildModelRecommendations(
      selectedAnimeNodeIds,
      selectedAnimeWeights,
      recommendationIndex,
      modelIndex,
    );
    recommendations = combineHybridRecommendations(
      graphRecommendations,
      modelRecommendations,
      modelBlendWeight,
    );
    recEngineStatusEl.textContent =
      `Using hybrid recommendations (${Math.round(modelBlendWeight * 100)}% model, ${Math.round((1 - modelBlendWeight) * 100)}% graph).`;
  }

  const includeSet = new Set(includeCandidateNodeIds);
  const excludeSet = new Set(excludeCandidateNodeIds);
  recommendations = recommendations.filter((item) => {
    if (excludeSet.has(item.anime.nodeId)) {
      return false;
    }
    if (includeSet.size > 0 && !includeSet.has(item.anime.nodeId)) {
      return false;
    }
    return true;
  });

  if (recommendations.length === 0) {
    recSummaryEl.textContent =
      "No positive recommendations found from the current watched list. Try adding more anime.";
    recResultsEl.innerHTML = "";
    return;
  }

  const methodLabel =
    recommendationMode === "graph"
      ? "graph edge ranking"
      : recommendationMode === "model"
        ? "ML model ranking"
        : "hybrid graph+ML ranking";
  recSummaryEl.textContent = `Showing top ${Math.min(MAX_RECOMMENDATIONS, recommendations.length)} recommendations from ${recommendations.length} candidates (${methodLabel}).`;

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

function buildGraphRecommendations(
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

function buildModelRecommendations(
  selectedNodeIds: string[],
  selectedWeights: Map<string, number>,
  index: RecommendationIndex,
  modelIndex: ModelRecommendationIndex,
): RecommendationResult[] {
  const watchedEntries = selectedNodeIds
    .map((nodeId) => {
      const anime = index.animeByNodeId.get(nodeId);
      if (!anime) {
        return null;
      }
      const modelAnime = modelIndex.animeByAnimeId.get(anime.animeId);
      if (!modelAnime) {
        return null;
      }
      return {
        anime,
        modelAnime,
        weight: clampWatchWeight(selectedWeights.get(nodeId) ?? 1),
      };
    })
    .filter(
      (
        value,
      ): value is {
        anime: AnimeInfo;
        modelAnime: ModelRecommendationAnime;
        weight: number;
      } => value !== null,
    );

  if (watchedEntries.length === 0) {
    return [];
  }

  const factors = watchedEntries[0].modelAnime.embedding.length;
  if (factors === 0) {
    return [];
  }

  const watchedAnimeIds = new Set(watchedEntries.map((entry) => entry.anime.animeId));
  const userVector = new Float32Array(factors);
  let denominator = 0;

  for (const watched of watchedEntries) {
    denominator += Math.abs(watched.weight);
    for (let i = 0; i < factors; i += 1) {
      userVector[i] += watched.modelAnime.embedding[i] * watched.weight;
    }
  }

  if (denominator <= 0) {
    denominator = watchedEntries.length;
  }
  for (let i = 0; i < factors; i += 1) {
    userVector[i] /= denominator;
  }

  const scored: RecommendationResult[] = [];
  for (const [animeId, modelAnime] of modelIndex.animeByAnimeId.entries()) {
    if (watchedAnimeIds.has(animeId)) {
      continue;
    }
    if (modelAnime.embedding.length !== factors) {
      continue;
    }

    let score = modelAnime.bias + modelIndex.globalMean;
    for (let i = 0; i < factors; i += 1) {
      score += userVector[i] * modelAnime.embedding[i];
    }

    const contributions = watchedEntries
      .map((watched) => {
        let similarity = 0;
        for (let i = 0; i < factors; i += 1) {
          similarity += watched.modelAnime.embedding[i] * modelAnime.embedding[i];
        }
        const weightedScore = similarity * watched.weight;
        return {
          watched: watched.anime,
          edgeWeight: similarity,
          weightFactor: watched.weight,
          weightedScore,
        } satisfies RecommendationContribution;
      })
      .sort((left, right) => right.weightedScore - left.weightedScore);

    const anime =
      index.animeByAnimeId.get(animeId) ??
      ({
        nodeId: `anime:${animeId}`,
        animeId,
        label: modelAnime.title,
      } satisfies AnimeInfo);

    const strongest = contributions.length > 0 ? contributions[0].weightedScore : 0;
    const supportCount = contributions.filter((item) => item.weightedScore > 0).length;

    scored.push({
      anime,
      score,
      strongest,
      supportCount,
      contributions,
    });
  }

  return scored.sort((left, right) => {
    if (right.score !== left.score) {
      return right.score - left.score;
    }
    if (right.supportCount !== left.supportCount) {
      return right.supportCount - left.supportCount;
    }
    return right.strongest - left.strongest;
  });
}

function combineHybridRecommendations(
  graphRecommendations: RecommendationResult[],
  modelRecommendations: RecommendationResult[],
  modelWeight: number,
): RecommendationResult[] {
  const clampedModelWeight = clampModelBlendWeight(modelWeight);
  const graphWeight = 1 - clampedModelWeight;

  const graphScale = createScoreScale(graphRecommendations);
  const modelScale = createScoreScale(modelRecommendations);

  const byAnime = new Map<
    number,
    {
      anime: AnimeInfo;
      score: number;
      strongest: number;
      supportCount: number;
      contributions: RecommendationContribution[];
    }
  >();

  for (const item of graphRecommendations) {
    const normalized = graphScale(item.score);
    const weighted = normalized * graphWeight;
    if (weighted <= 0) {
      continue;
    }
    byAnime.set(item.anime.animeId, {
      anime: item.anime,
      score: weighted,
      strongest: item.strongest * graphWeight,
      supportCount: item.supportCount,
      contributions: scaleContributions(item.contributions, graphWeight),
    });
  }

  for (const item of modelRecommendations) {
    const normalized = modelScale(item.score);
    const weighted = normalized * clampedModelWeight;
    if (weighted <= 0) {
      continue;
    }
    const current = byAnime.get(item.anime.animeId);
    if (!current) {
      byAnime.set(item.anime.animeId, {
        anime: item.anime,
        score: weighted,
        strongest: item.strongest * clampedModelWeight,
        supportCount: item.supportCount,
        contributions: scaleContributions(item.contributions, clampedModelWeight),
      });
      continue;
    }

    current.score += weighted;
    current.strongest = Math.max(
      current.strongest,
      item.strongest * clampedModelWeight,
    );
    current.supportCount += item.supportCount;
    current.contributions = [...current.contributions]
      .concat(scaleContributions(item.contributions, clampedModelWeight))
      .sort((left, right) => right.weightedScore - left.weightedScore)
      .slice(0, 10);
  }

  return [...byAnime.values()]
    .map((value) => ({
      anime: value.anime,
      score: value.score,
      strongest: value.strongest,
      supportCount: value.supportCount,
      contributions: value.contributions,
    }))
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

function createScoreScale(
  recommendations: RecommendationResult[],
): (score: number) => number {
  if (recommendations.length === 0) {
    return () => 0;
  }
  let min = recommendations[0].score;
  let max = recommendations[0].score;
  for (const item of recommendations) {
    if (item.score < min) {
      min = item.score;
    }
    if (item.score > max) {
      max = item.score;
    }
  }

  if (!Number.isFinite(min) || !Number.isFinite(max) || max <= min) {
    return (score: number) => (Number.isFinite(score) ? 1 : 0);
  }

  const denominator = max - min;
  return (score: number) => {
    if (!Number.isFinite(score)) {
      return 0;
    }
    const normalized = (score - min) / denominator;
    return Math.min(Math.max(normalized, 0), 1);
  };
}

function scaleContributions(
  contributions: RecommendationContribution[],
  factor: number,
): RecommendationContribution[] {
  return contributions.map((item) => ({
    watched: item.watched,
    edgeWeight: item.edgeWeight,
    weightFactor: item.weightFactor,
    weightedScore: item.weightedScore * factor,
  }));
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

function populateNetworkNodeOptions(nodes: GraphNode[], datalist: HTMLDataListElement): void {
  const values: string[] = [];
  for (const node of nodes) {
    values.push(node.label);
    values.push(node.id);
  }
  const uniqueSorted = [...new Set(values)].sort((left, right) => left.localeCompare(right));
  datalist.innerHTML = uniqueSorted
    .slice(0, 8000)
    .map((value) => `<option value="${escapeHtml(value)}"></option>`)
    .join("");
}

function resolveNetworkNodeQuery(query: string, graphDataValue: GraphData): GraphNode | null {
  const raw = query.trim();
  if (!raw) {
    return null;
  }

  const rawLower = raw.toLowerCase();
  const normalized = normalizeTitle(raw);

  const byExactId = graphDataValue.nodes.find((node) => node.id.toLowerCase() === rawLower);
  if (byExactId) {
    return byExactId;
  }

  if (/^\d+$/.test(raw)) {
    const animeNodeId = `anime:${Number.parseInt(raw, 10)}`;
    const byAnimeId = graphDataValue.nodes.find((node) => node.id === animeNodeId);
    if (byAnimeId) {
      return byAnimeId;
    }
  }

  const byExactLabel = graphDataValue.nodes.find(
    (node) => normalizeTitle(node.label) === normalized,
  );
  if (byExactLabel) {
    return byExactLabel;
  }

  const byStartsWith = graphDataValue.nodes.find((node) =>
    normalizeTitle(node.label).startsWith(normalized),
  );
  if (byStartsWith) {
    return byStartsWith;
  }

  return (
    graphDataValue.nodes.find((node) =>
      normalizeTitle(`${node.label} ${node.id}`).includes(normalized),
    ) ?? null
  );
}

function selectNodeAndFocus(nodeId: string): void {
  selectedNodeId = nodeId;
  if (activeView !== "network") {
    setActiveView("network", false);
    window.requestAnimationFrame(() => {
      window.requestAnimationFrame(() => {
        focusNodeInRenderer(nodeId);
      });
    });
    return;
  }
  if (!currentGraph || !currentGraph.hasNode(nodeId)) {
    rerenderGraph();
    window.requestAnimationFrame(() => {
      focusNodeInRenderer(nodeId);
    });
    return;
  }
  focusNodeInRenderer(nodeId);
}

function focusNodeInRenderer(nodeId: string): void {
  if (!currentGraph || !renderer || !currentGraph.hasNode(nodeId)) {
    return;
  }
  renderInspectPanel(nodeId);

  const attrs = currentGraph.getNodeAttributes(nodeId) as Record<string, unknown>;
  const x = Number(attrs.x);
  const y = Number(attrs.y);
  if (Number.isFinite(x) && Number.isFinite(y)) {
    const camera = renderer.getCamera();
    camera.animate({ x, y, ratio: 0.32 }, { duration: 360 });
  }
  renderer.refresh();
}

function buildRecommendationIndex(graphDataValue: GraphData): RecommendationIndex {
  const animeList: AnimeInfo[] = [];
  const animeByNodeId = new Map<string, AnimeInfo>();
  const animeByAnimeId = new Map<number, AnimeInfo>();
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
    animeByAnimeId.set(animeId, anime);

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
    animeByAnimeId,
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
  applySelectionReducers();

  renderer.on("clickNode", (event) => {
    selectedNodeId = event.node;
    renderInspectPanel(event.node);
    networkSearchMessage.textContent = `Focused: ${event.node}`;
    if (renderer) {
      renderer.refresh();
    }
  });

  renderer.on("clickStage", () => {
    selectedNodeId = null;
    networkSearchMessage.textContent = "";
    renderInspectPanel(null);
    if (renderer) {
      renderer.refresh();
    }
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
    focusNodeInRenderer(selectedNodeId);
  } else {
    selectedNodeId = null;
    renderInspectPanel(null);
  }
}

function statLine(label: string, value: string): string {
  return `<div class="stat-row"><span>${label}</span><strong>${value}</strong></div>`;
}

async function fetchGraph(): Promise<GraphData> {
  const compactData = await fetchJsonWithGzipFallback<CompactGraphData>({
    path: "./data/graph.compact.json",
    required: false,
    label: "graph.compact.json",
  });
  if (compactData && isCompactGraphData(compactData)) {
    return expandCompactGraphData(compactData);
  }

  const legacyData = await fetchJsonWithGzipFallback<GraphData>({
    path: "./data/graph.json",
    required: true,
    label: "graph.json",
  });
  if (!legacyData) {
    throw new Error("Unable to load required graph data.");
  }
  return legacyData;
}

async function ensureModelRecommendationIndex(): Promise<ModelRecommendationIndex | null> {
  if (!modelRecommendationIndexPromise) {
    modelRecommendationIndexPromise = fetchModelRecommendationIndex();
  }
  return modelRecommendationIndexPromise;
}

async function fetchModelRecommendationIndex(): Promise<ModelRecommendationIndex | null> {
  const rawCompact = await fetchJsonWithGzipFallback<CompactModelRecommendationData>({
    path: "./data/model-mf-web.compact.json",
    required: false,
    label: "model-mf-web.compact.json",
  });
  const rawLegacy = rawCompact
    ? null
    : await fetchJsonWithGzipFallback<ModelRecommendationData>({
        path: "./data/model-mf-web.json",
        required: false,
        label: "model-mf-web.json",
      });

  const animeByAnimeId = new Map<number, ModelRecommendationAnime>();
  let generatedAt = "";
  let factors = 0;
  let globalMean = 0;

  if (rawCompact && isCompactModelRecommendationData(rawCompact)) {
    generatedAt = rawCompact.generatedAt;
    factors = rawCompact.factors;
    globalMean = Number.isFinite(rawCompact.globalMean) ? rawCompact.globalMean : 0;

    const count = Math.min(
      rawCompact.animeIds.length,
      rawCompact.titles.length,
      rawCompact.biases.length,
      rawCompact.embeddings.length,
    );
    for (let index = 0; index < count; index += 1) {
      const animeId = rawCompact.animeIds[index];
      const title = rawCompact.titles[index];
      const bias = rawCompact.biases[index];
      const embedding = rawCompact.embeddings[index];
      if (!Number.isFinite(animeId) || !Number.isFinite(bias)) {
        continue;
      }
      if (!Array.isArray(embedding) || embedding.length === 0) {
        continue;
      }
      if (!embedding.every((value) => Number.isFinite(value))) {
        continue;
      }
      animeByAnimeId.set(animeId, {
        animeId,
        title: String(title),
        bias,
        embedding,
      });
    }
  } else if (rawLegacy) {
    generatedAt = rawLegacy.generatedAt;
    factors = rawLegacy.factors;
    globalMean = Number.isFinite(rawLegacy.globalMean) ? rawLegacy.globalMean : 0;
    for (const anime of rawLegacy.anime) {
      if (!Number.isFinite(anime.animeId) || !Number.isFinite(anime.bias)) {
        continue;
      }
      if (!Array.isArray(anime.embedding) || anime.embedding.length === 0) {
        continue;
      }
      if (!anime.embedding.every((value) => Number.isFinite(value))) {
        continue;
      }
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

function isCompactGraphData(value: unknown): value is CompactGraphData {
  if (!value || typeof value !== "object") {
    return false;
  }
  const maybe = value as Record<string, unknown>;
  return maybe.format === "graph-compact-v1";
}

function expandCompactGraphData(compact: CompactGraphData): GraphData {
  const nodes: GraphNode[] = [];
  const edges: GraphEdge[] = [];

  const userNodeIds: string[] = [];
  for (let i = 0; i < compact.userIds.length; i += 1) {
    const userId = compact.userIds[i];
    const nodeId = `user:${userId}`;
    userNodeIds.push(nodeId);
    nodes.push({
      id: nodeId,
      label: `User ${String(userId).slice(0, 8)}`,
      nodeType: "user",
    });
  }

  const animeNodeIds: string[] = [];
  for (let i = 0; i < compact.anime.length; i += 1) {
    const animeEntry = compact.anime[i];
    const animeId = animeEntry[0];
    const title = animeEntry[1];
    const nodeId = `anime:${animeId}`;
    animeNodeIds.push(nodeId);
    nodes.push({
      id: nodeId,
      label: String(title),
      nodeType: "anime",
    });
  }

  for (const [userIndex, animeIndex, weight] of compact.ua) {
    const source = userNodeIds[userIndex];
    const target = animeNodeIds[animeIndex];
    if (!source || !target || !Number.isFinite(weight)) {
      continue;
    }
    edges.push({
      id: `ua:${compact.userIds[userIndex]}:${compact.anime[animeIndex][0]}`,
      source,
      target,
      edgeType: "user-anime",
      weight,
    });
  }

  for (const [leftAnimeIndex, rightAnimeIndex, weight] of compact.aa) {
    const source = animeNodeIds[leftAnimeIndex];
    const target = animeNodeIds[rightAnimeIndex];
    if (!source || !target || !Number.isFinite(weight)) {
      continue;
    }
    edges.push({
      id: `aa:${compact.anime[leftAnimeIndex][0]}:${compact.anime[rightAnimeIndex][0]}`,
      source,
      target,
      edgeType: "anime-anime",
      weight,
    });
  }

  return {
    generatedAt: compact.generatedAt,
    userCount: compact.userCount,
    animeCount: compact.animeCount,
    nodeCount: nodes.length,
    edgeCount: edges.length,
    nodes,
    edges,
  };
}

function isCompactModelRecommendationData(
  value: unknown,
): value is CompactModelRecommendationData {
  if (!value || typeof value !== "object") {
    return false;
  }
  const maybe = value as Record<string, unknown>;
  return maybe.format === "model-mf-compact-v1";
}

async function fetchJsonWithGzipFallback<T>({
  path,
  required,
  label,
}: {
  path: string;
  required: boolean;
  label: string;
}): Promise<T | null> {
  const gzPath = `${path}.gz`;
  let gzStatus: number | null = null;

  try {
    const gzResponse = await fetch(gzPath);
    gzStatus = gzResponse.status;
    if (gzResponse.ok) {
      try {
        return await parseGzipJsonResponse<T>(gzResponse, label);
      } catch (error) {
        console.warn(`Failed to parse ${label}.gz; falling back to JSON`, error);
      }
    } else if (gzResponse.status !== 404) {
      console.warn(`Unable to load ${label}.gz (${gzResponse.status})`);
    }
  } catch (error) {
    console.warn(`Fetch failed for ${label}.gz`, error);
  }

  const response = await fetch(path);
  if (!response.ok) {
    if (!required && response.status === 404 && (gzStatus === 404 || gzStatus === null)) {
      return null;
    }
    throw new Error(`Unable to load ${label} (${response.status})`);
  }
  return (await response.json()) as T;
}

async function parseGzipJsonResponse<T>(
  response: Response,
  label: string,
): Promise<T> {
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
  return JSON.parse(text) as T;
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

function applySelectionReducers(): void {
  if (!renderer || !currentGraph) {
    return;
  }

  renderer.setSetting("nodeReducer", (node, data) => {
    if (!selectedNodeId || !currentGraph || !currentGraph.hasNode(selectedNodeId)) {
      return data;
    }

    const selected = selectedNodeId;
    if (node === selected) {
      return {
        ...data,
        zIndex: 2,
        size: ((data.size as number) ?? 1) * 1.35,
        color: "#ffd166",
      };
    }

    const connected =
      currentGraph.hasEdge(node, selected) || currentGraph.hasEdge(selected, node);
    if (connected) {
      return {
        ...data,
        zIndex: 1,
      };
    }

    return {
      ...data,
      color: "#4f607388",
      label: "",
    };
  });

  renderer.setSetting("edgeReducer", (edge, data) => {
    if (!selectedNodeId || !currentGraph || !currentGraph.hasNode(selectedNodeId)) {
      return data;
    }
    const source = currentGraph.source(edge);
    const target = currentGraph.target(edge);
    if (source === selectedNodeId || target === selectedNodeId) {
      return {
        ...data,
        color: "#ffd166bb",
        size: ((data.size as number) ?? 1) * 1.3,
      };
    }
    return {
      ...data,
      color: "#30415655",
    };
  });
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

function parseRecommendationMode(value: string): RecommendationMode {
  if (value === "model") {
    return "model";
  }
  if (value === "hybrid") {
    return "hybrid";
  }
  return "graph";
}

function clampWatchWeight(value: number): number {
  if (!Number.isFinite(value)) {
    return 1;
  }
  return Math.min(Math.max(value, MIN_WATCH_WEIGHT), MAX_WATCH_WEIGHT);
}

function clampModelBlendWeight(value: number): number {
  if (!Number.isFinite(value)) {
    return 0.5;
  }
  return Math.min(Math.max(value, MIN_MODEL_BLEND_WEIGHT), MAX_MODEL_BLEND_WEIGHT);
}

function renderModelBlendValue(): void {
  const modelPercent = Math.round(modelBlendWeight * 100);
  const graphPercent = 100 - modelPercent;
  recBlendValueEl.textContent = `${modelPercent}% model / ${graphPercent}% graph`;
}

function setBlendControlVisibility(): void {
  recBlendControl.hidden = recommendationMode !== "hybrid";
}

function loadRecommendationState(index: RecommendationIndex): {
  mode: RecommendationMode;
  selected: { nodeId: string; weight: number }[];
  modelBlendWeight: number;
  includeCandidates: string[];
  excludeCandidates: string[];
} {
  try {
    const raw = window.localStorage.getItem(RECOMMENDATION_STATE_STORAGE_KEY);
    if (!raw) {
      return {
        mode: "graph",
        selected: [],
        modelBlendWeight: 0.5,
        includeCandidates: [],
        excludeCandidates: [],
      };
    }
    const parsed = JSON.parse(raw) as StoredRecommendationState;
    if (!parsed || (parsed.version !== 1 && parsed.version !== 2 && parsed.version !== 3)) {
      return {
        mode: "graph",
        selected: [],
        modelBlendWeight: 0.5,
        includeCandidates: [],
        excludeCandidates: [],
      };
    }
    const mode = parseRecommendationMode(parsed.mode);
    const modelBlendWeight = clampModelBlendWeight(
      Number(parsed.modelBlendWeight ?? 0.5),
    );
    const selected = Array.isArray(parsed.selected)
      ? parsed.selected
          .filter(
            (entry) =>
              entry &&
              typeof entry.nodeId === "string" &&
              index.animeByNodeId.has(entry.nodeId),
          )
          .map((entry) => ({
            nodeId: entry.nodeId,
            weight: clampWatchWeight(Number(entry.weight)),
          }))
      : [];

    const includeCandidates = Array.isArray(parsed.includeCandidates)
      ? parsed.includeCandidates
          .filter((entry) => typeof entry === "string" && index.animeByNodeId.has(entry))
      : [];
    const excludeCandidates = Array.isArray(parsed.excludeCandidates)
      ? parsed.excludeCandidates
          .filter((entry) => typeof entry === "string" && index.animeByNodeId.has(entry))
      : [];

    return {
      mode,
      selected,
      modelBlendWeight,
      includeCandidates,
      excludeCandidates,
    };
  } catch (error) {
    console.warn("Unable to load saved recommendation state.", error);
    return {
      mode: "graph",
      selected: [],
      modelBlendWeight: 0.5,
      includeCandidates: [],
      excludeCandidates: [],
    };
  }
}

function persistRecommendationState(): void {
  try {
    const selected = selectedAnimeNodeIds
      .filter((nodeId) => recommendationIndex.animeByNodeId.has(nodeId))
      .map((nodeId) => ({
        nodeId,
        weight: clampWatchWeight(selectedAnimeWeights.get(nodeId) ?? 1),
      }));
    const payload: StoredRecommendationState = {
      version: 3,
      mode: recommendationMode,
      selected,
      modelBlendWeight: clampModelBlendWeight(modelBlendWeight),
      includeCandidates: includeCandidateNodeIds.filter((nodeId) =>
        recommendationIndex.animeByNodeId.has(nodeId),
      ),
      excludeCandidates: excludeCandidateNodeIds.filter((nodeId) =>
        recommendationIndex.animeByNodeId.has(nodeId),
      ),
    };
    window.localStorage.setItem(
      RECOMMENDATION_STATE_STORAGE_KEY,
      JSON.stringify(payload),
    );
  } catch (error) {
    console.warn("Unable to persist recommendation state.", error);
  }
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
