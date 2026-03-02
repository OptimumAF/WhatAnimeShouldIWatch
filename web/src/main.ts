import Graph from "graphology";
import forceAtlas2 from "graphology-layout-forceatlas2";
import Sigma from "sigma";
import "./style.css";

type NodeType = "user" | "anime";
type EdgeType = "user-anime" | "anime-anime";
type AppView = "recommendations" | "network";
type RecommendationMode = "graph" | "model" | "hybrid";
type UsernameImportProvider = "anilist" | "mal";
type ThemeMode = "dark" | "light";

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

interface AnimeMetadata {
  animeId: number;
  year: number | null;
  score: number | null;
  genres: string[];
  studios: string[];
  synopsis: string;
  imageUrl: string;
  season: string | null;
}

interface RecommendationFilters {
  genre: string;
  minYear: number | null;
  maxYear: number | null;
  minScore: number | null;
}

interface SeasonalAnimeItem {
  animeId: number;
  title: string;
  score: number | null;
  year: number | null;
  season: string | null;
  imageUrl: string;
}

interface ImportedWatchedEntry {
  anime: AnimeInfo;
  weight: number;
}

interface UsernameImportResult {
  entries: ImportedWatchedEntry[];
  ratedCount: number;
  unmappedCount: number;
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
const USERNAME_IMPORT_MAX_RETRIES = 3;
const USERNAME_IMPORT_PAGE_SIZE = 300;
const USERNAME_IMPORT_PAGE_DELAY_MS = 350;
const METADATA_MAX_RETRIES = 2;
const METADATA_PREFETCH_LIMIT = 100;
const METADATA_PREFETCH_WITH_FILTER_LIMIT = 30;
const METADATA_PREFETCH_CONCURRENCY = 3;
const METADATA_SCORE_STEP = 0.1;
const SEASONAL_LIST_LIMIT = 12;
const RECOMMENDATION_STATE_STORAGE_KEY = "wasiw.recommendationState.v1";
const RECOMMENDATION_PROFILES_STORAGE_KEY = "wasiw.recommendationProfiles.v1";
const THEME_STORAGE_KEY = "wasiw.theme.v1";

interface StoredRecommendationState {
  version: number;
  mode: RecommendationMode;
  selected: { nodeId: string; weight: number }[];
  modelBlendWeight?: number;
  includeCandidates?: string[];
  excludeCandidates?: string[];
}

interface RecommendationProfileRecord {
  name: string;
  updatedAt: string;
  state: StoredRecommendationState;
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
        <button id="theme-toggle" class="nav-btn theme-btn" type="button" aria-label="Toggle theme">Theme</button>
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

            <p id="rec-message" class="rec-message" role="status" aria-live="polite"></p>

            <div class="selected-head">
              <h3>Watched List</h3>
              <button id="clear-watched" type="button" class="ghost-btn">Clear All</button>
            </div>
            <div id="selected-anime" class="selected-anime"></div>

            <section class="bulk-import">
              <h3>Bulk Import</h3>
              <p class="muted">One entry per line: <code>animeId[, score]</code>, <code>anime:ID[, score]</code>, or <code>title[, score]</code>.</p>
              <form id="bulk-import-form" class="bulk-import-form">
                <textarea id="bulk-import-input" rows="6" placeholder="5114, 9&#10;anime:9253, 7.5&#10;Steins;Gate"></textarea>
                <button type="submit">Import Watched Entries</button>
              </form>
            </section>

            <section class="username-import">
              <h3>Import From Username</h3>
              <p class="muted">Load rated anime from a public AniList or MAL profile.</p>
              <form id="username-import-form" class="username-import-form">
                <select id="username-import-provider" aria-label="Import provider">
                  <option value="anilist" selected>AniList</option>
                  <option value="mal">MyAnimeList (MAL)</option>
                </select>
                <input id="username-import-input" type="text" autocomplete="off" placeholder="Enter username" />
                <button id="username-import-submit" type="submit">Import User List</button>
              </form>
            </section>

            <section class="profiles">
              <h3>Saved Profiles</h3>
              <p class="muted">Save and reload named recommendation setups.</p>
              <form id="profile-save-form" class="profile-save-form">
                <input id="profile-name-input" type="text" autocomplete="off" placeholder="Profile name" />
                <button id="profile-save-submit" type="submit">Save Profile</button>
              </form>
              <div class="profile-load-row">
                <select id="profile-select" aria-label="Saved profile"></select>
                <button id="profile-load-btn" type="button">Load</button>
                <button id="profile-delete-btn" type="button" class="ghost-btn">Delete</button>
              </div>
            </section>

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
            <section class="rec-filters">
              <h3>Filters</h3>
              <div class="rec-filter-grid">
                <label class="control-inline" for="filter-genre">
                  <span>Genre</span>
                  <select id="filter-genre">
                    <option value="">Any</option>
                  </select>
                </label>
                <label class="control-inline" for="filter-year-min">
                  <span>Year from</span>
                  <input id="filter-year-min" type="number" min="1900" max="2100" step="1" placeholder="Any" />
                </label>
                <label class="control-inline" for="filter-year-max">
                  <span>Year to</span>
                  <input id="filter-year-max" type="number" min="1900" max="2100" step="1" placeholder="Any" />
                </label>
              </div>
              <label class="control-inline control-inline-score" for="filter-min-score">
                <span>Minimum MAL score</span>
                <input id="filter-min-score" type="range" min="0" max="10" step="${METADATA_SCORE_STEP}" value="0" />
                <output id="filter-min-score-value">Any</output>
              </label>
              <div class="rec-filter-actions">
                <button id="clear-rec-filters" type="button" class="ghost-btn">Clear Filters</button>
                <span id="metadata-status" class="metadata-status" role="status" aria-live="polite">Metadata: idle</span>
              </div>
            </section>
            <p id="rec-summary" class="muted" role="status" aria-live="polite">Add at least one anime to start.</p>
            <ol id="rec-results" class="rec-results"></ol>

            <section class="seasonal">
              <div class="seasonal-head">
                <h3>Seasonal Trending</h3>
                <button id="refresh-seasonal" type="button" class="ghost-btn">Refresh</button>
              </div>
              <p id="seasonal-status" class="muted" role="status" aria-live="polite">Loading current season...</p>
              <ul id="seasonal-list" class="seasonal-list"></ul>
            </section>
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
              <p id="network-search-message" class="network-search-message" role="status" aria-live="polite"></p>
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
const themeToggleBtn = mustElement<HTMLButtonElement>("#theme-toggle");
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
const bulkImportForm = mustElement<HTMLFormElement>("#bulk-import-form");
const bulkImportInput = mustElement<HTMLTextAreaElement>("#bulk-import-input");
const usernameImportForm = mustElement<HTMLFormElement>("#username-import-form");
const usernameImportProvider = mustElement<HTMLSelectElement>("#username-import-provider");
const usernameImportInput = mustElement<HTMLInputElement>("#username-import-input");
const usernameImportSubmit = mustElement<HTMLButtonElement>("#username-import-submit");
const profileSaveForm = mustElement<HTMLFormElement>("#profile-save-form");
const profileNameInput = mustElement<HTMLInputElement>("#profile-name-input");
const profileSelect = mustElement<HTMLSelectElement>("#profile-select");
const profileLoadBtn = mustElement<HTMLButtonElement>("#profile-load-btn");
const profileDeleteBtn = mustElement<HTMLButtonElement>("#profile-delete-btn");
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
const filterGenreSelect = mustElement<HTMLSelectElement>("#filter-genre");
const filterYearMinInput = mustElement<HTMLInputElement>("#filter-year-min");
const filterYearMaxInput = mustElement<HTMLInputElement>("#filter-year-max");
const filterMinScoreInput = mustElement<HTMLInputElement>("#filter-min-score");
const filterMinScoreValue = mustElement<HTMLOutputElement>("#filter-min-score-value");
const clearRecFiltersBtn = mustElement<HTMLButtonElement>("#clear-rec-filters");
const metadataStatusEl = mustElement<HTMLSpanElement>("#metadata-status");
const seasonalStatusEl = mustElement<HTMLParagraphElement>("#seasonal-status");
const seasonalListEl = mustElement<HTMLUListElement>("#seasonal-list");
const refreshSeasonalBtn = mustElement<HTMLButtonElement>("#refresh-seasonal");

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
const recommendationFilters: RecommendationFilters = {
  genre: "",
  minYear: null,
  maxYear: null,
  minScore: null,
};
const animeMetadataCache = new Map<number, AnimeMetadata>();
const animeMetadataUnavailable = new Set<number>();
const animeMetadataInFlight = new Map<number, Promise<AnimeMetadata | null>>();
let seasonalItems: SeasonalAnimeItem[] = [];
let seasonalLoadingPromise: Promise<void> | null = null;
let activeTheme: ThemeMode = loadThemeModePreference();
const reduceMotionMediaQuery = window.matchMedia("(prefers-reduced-motion: reduce)");

applyTheme(activeTheme);

const graphData = await fetchGraph();
const recommendationIndex = buildRecommendationIndex(graphData);
const selectedAnimeNodeIds: string[] = [];
const selectedAnimeWeights = new Map<string, number>();
const includeCandidateNodeIds: string[] = [];
const excludeCandidateNodeIds: string[] = [];
const persistedState = loadRecommendationState(recommendationIndex);
const savedProfiles = loadRecommendationProfiles();

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
renderProfileOptions(savedProfiles);
renderFilterControls();
renderSeasonalList();

const defaultMinWeight = getDefaultMinAnimeAnimeWeight(graphData, minWeightInput);
minWeightInput.value = defaultMinWeight.toFixed(2);
minWeightValue.textContent = defaultMinWeight.toFixed(2);

setActiveView(viewFromHash(), true);
void updateRecommendations();
void loadSeasonalTrending(false);

window.addEventListener("hashchange", () => {
  setActiveView(viewFromHash(), true);
});

navRecommendationsBtn.addEventListener("click", () => {
  setActiveView("recommendations", false);
});

navNetworkBtn.addEventListener("click", () => {
  setActiveView("network", false);
});

themeToggleBtn.addEventListener("click", () => {
  activeTheme = activeTheme === "dark" ? "light" : "dark";
  applyTheme(activeTheme);
  persistThemeModePreference(activeTheme);
});

addAnimeForm.addEventListener("submit", (event) => {
  event.preventDefault();
  addAnimeFromInput();
});

bulkImportForm.addEventListener("submit", (event) => {
  event.preventDefault();
  importWatchedFromBulkInput();
});

usernameImportForm.addEventListener("submit", (event) => {
  event.preventDefault();
  void importWatchedFromUsername();
});

profileSaveForm.addEventListener("submit", (event) => {
  event.preventDefault();
  saveCurrentProfile();
});

profileLoadBtn.addEventListener("click", () => {
  loadSelectedProfile();
});

profileDeleteBtn.addEventListener("click", () => {
  deleteSelectedProfile();
});

filterGenreSelect.addEventListener("change", () => {
  recommendationFilters.genre = filterGenreSelect.value.trim().toLowerCase();
  void updateRecommendations();
});

filterYearMinInput.addEventListener("change", () => {
  recommendationFilters.minYear = parseYearFilterValue(filterYearMinInput.value);
  void updateRecommendations();
});

filterYearMaxInput.addEventListener("change", () => {
  recommendationFilters.maxYear = parseYearFilterValue(filterYearMaxInput.value);
  void updateRecommendations();
});

filterMinScoreInput.addEventListener("input", () => {
  const scoreValue = Number.parseFloat(filterMinScoreInput.value);
  recommendationFilters.minScore =
    Number.isFinite(scoreValue) && scoreValue > 0 ? scoreValue : null;
  renderMinScoreFilterLabel();
  void updateRecommendations();
});

clearRecFiltersBtn.addEventListener("click", () => {
  recommendationFilters.genre = "";
  recommendationFilters.minYear = null;
  recommendationFilters.maxYear = null;
  recommendationFilters.minScore = null;
  renderFilterControls();
  void updateRecommendations();
});

refreshSeasonalBtn.addEventListener("click", () => {
  void loadSeasonalTrending(true);
});

seasonalListEl.addEventListener("click", (event) => {
  const target = event.target;
  if (!(target instanceof HTMLElement)) {
    return;
  }
  const button = target.closest<HTMLButtonElement>("button[data-seasonal-anime-id]");
  if (!button) {
    return;
  }
  const animeIdRaw = button.dataset.seasonalAnimeId;
  if (!animeIdRaw) {
    return;
  }
  const animeId = Number.parseInt(animeIdRaw, 10);
  if (!Number.isFinite(animeId)) {
    return;
  }
  addAnimeById(animeId);
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

function parseYearFilterValue(raw: string): number | null {
  const value = Number.parseInt(raw.trim(), 10);
  if (!Number.isFinite(value)) {
    return null;
  }
  return Math.min(Math.max(value, 1900), 2100);
}

function renderFilterControls(): void {
  filterGenreSelect.value = recommendationFilters.genre;
  filterYearMinInput.value =
    recommendationFilters.minYear === null
      ? ""
      : String(recommendationFilters.minYear);
  filterYearMaxInput.value =
    recommendationFilters.maxYear === null
      ? ""
      : String(recommendationFilters.maxYear);
  filterMinScoreInput.value = (recommendationFilters.minScore ?? 0).toFixed(1);
  renderMinScoreFilterLabel();
}

function renderMinScoreFilterLabel(): void {
  if (recommendationFilters.minScore === null || recommendationFilters.minScore <= 0) {
    filterMinScoreValue.textContent = "Any";
    return;
  }
  filterMinScoreValue.textContent = recommendationFilters.minScore.toFixed(1);
}

function addAnimeById(animeId: number): void {
  const anime = recommendationIndex.animeByAnimeId.get(animeId);
  if (!anime) {
    recMessageEl.textContent = `Anime ${animeId} is not in the current graph dataset.`;
    return;
  }
  addAnimeToWatchedList(anime, "Added");
}

function addAnimeToWatchedList(anime: AnimeInfo, prefix: string): void {
  if (selectedAnimeNodeIds.includes(anime.nodeId)) {
    recMessageEl.textContent = `${anime.label} is already in your watched list.`;
    return;
  }

  selectedAnimeNodeIds.push(anime.nodeId);
  selectedAnimeWeights.set(anime.nodeId, 1);
  persistRecommendationState();
  recMessageEl.textContent = `${prefix}: ${anime.label}`;
  renderSelectedAnime();
  void updateRecommendations();
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

  animeInput.value = "";
  addAnimeToWatchedList(anime, "Added");
}

function importWatchedFromBulkInput(): void {
  const raw = bulkImportInput.value.trim();
  if (!raw) {
    recMessageEl.textContent = "Paste at least one line to import.";
    return;
  }

  const lines = raw
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0);

  if (lines.length === 0) {
    recMessageEl.textContent = "Paste at least one line to import.";
    return;
  }

  let added = 0;
  let updated = 0;
  let skipped = 0;
  let unresolved = 0;
  const unresolvedLines: string[] = [];

  for (const line of lines) {
    const parsed = parseBulkWatchedLine(line, recommendationIndex);
    if (!parsed) {
      unresolved += 1;
      if (unresolvedLines.length < 3) {
        unresolvedLines.push(line);
      }
      continue;
    }

    const { anime, weight } = parsed;
    const existingIndex = selectedAnimeNodeIds.indexOf(anime.nodeId);
    if (existingIndex < 0) {
      selectedAnimeNodeIds.push(anime.nodeId);
      selectedAnimeWeights.set(anime.nodeId, weight ?? 1);
      added += 1;
      continue;
    }

    if (weight === undefined) {
      skipped += 1;
      continue;
    }

    selectedAnimeWeights.set(anime.nodeId, weight);
    updated += 1;
  }

  if (added === 0 && updated === 0 && skipped === 0 && unresolved === 0) {
    recMessageEl.textContent = "No entries were imported.";
    return;
  }

  persistRecommendationState();
  renderSelectedAnime();
  void updateRecommendations();

  const unresolvedNote =
    unresolved > 0
      ? ` Unresolved: ${unresolved}${unresolvedLines.length > 0 ? ` (${unresolvedLines.join("; ")})` : ""}.`
      : "";
  recMessageEl.textContent =
    `Import complete. Added: ${added}, Updated: ${updated}, Skipped: ${skipped}.` +
    unresolvedNote;
}

async function importWatchedFromUsername(): Promise<void> {
  const provider = parseUsernameImportProvider(usernameImportProvider.value);
  const username = usernameImportInput.value.trim();
  if (!username) {
    recMessageEl.textContent = "Enter a username first.";
    return;
  }

  setUsernameImportLoading(true, provider);
  recMessageEl.textContent = `Importing rated anime from ${providerLabel(provider)} user "${username}"...`;

  try {
    const result =
      provider === "anilist"
        ? await fetchAniListUsernameImport(username, recommendationIndex)
        : await fetchMalUsernameImport(username, recommendationIndex);

    const summary = upsertImportedWatchedEntries(result.entries);
    persistRecommendationState();
    renderSelectedAnime();
    void updateRecommendations();

    recMessageEl.textContent =
      `Imported ${providerLabel(provider)} user "${username}". ` +
      `Rated: ${result.ratedCount}, matched in graph: ${result.entries.length}, ` +
      `added: ${summary.added}, updated: ${summary.updated}, skipped: ${summary.skipped}, ` +
      `unmapped: ${result.unmappedCount}.`;
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "Unable to import this username.";
    recMessageEl.textContent = `Username import failed: ${message}`;
  } finally {
    setUsernameImportLoading(false, provider);
  }
}

function parseUsernameImportProvider(raw: string): UsernameImportProvider {
  if (raw === "mal") {
    return "mal";
  }
  return "anilist";
}

function providerLabel(provider: UsernameImportProvider): string {
  return provider === "mal" ? "MAL" : "AniList";
}

function setUsernameImportLoading(
  loading: boolean,
  provider: UsernameImportProvider,
): void {
  usernameImportProvider.disabled = loading;
  usernameImportInput.disabled = loading;
  usernameImportSubmit.disabled = loading;
  usernameImportSubmit.textContent = loading
    ? `Importing ${providerLabel(provider)}...`
    : "Import User List";
}

async function fetchAniListUsernameImport(
  username: string,
  index: RecommendationIndex,
): Promise<UsernameImportResult> {
  const query = `
    query ($userName: String) {
      MediaListCollection(userName: $userName, type: ANIME) {
        lists {
          entries {
            score(format: POINT_10_DECIMAL)
            media {
              idMal
            }
          }
        }
      }
    }
  `;

  const payload = await fetchJsonWithRetries<{
    data?: {
      MediaListCollection?: {
        lists?: Array<{
          entries?: Array<{
            score?: number;
            media?: {
              idMal?: number | null;
            } | null;
          }>;
        }>;
      } | null;
    };
    errors?: Array<{ message?: string }>;
  }>("https://graphql.anilist.co", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
    },
    body: JSON.stringify({
      query,
      variables: { userName: username },
    }),
  });

  if (payload.errors && payload.errors.length > 0) {
    const message = payload.errors[0]?.message ?? "AniList API returned an error.";
    throw new Error(message);
  }

  const lists = payload.data?.MediaListCollection?.lists ?? [];
  if (lists.length === 0) {
    throw new Error(
      "No AniList anime list found for that user (private, empty, or not found).",
    );
  }

  let ratedCount = 0;
  let unmappedCount = 0;
  const byNodeId = new Map<string, ImportedWatchedEntry>();

  for (const list of lists) {
    const entries = list.entries ?? [];
    for (const entry of entries) {
      const score = Number(entry.score ?? 0);
      const malId = Number(entry.media?.idMal ?? 0);
      if (!Number.isFinite(score) || score <= 0 || !Number.isFinite(malId) || malId <= 0) {
        continue;
      }

      ratedCount += 1;
      const anime = index.animeByAnimeId.get(Math.trunc(malId));
      if (!anime) {
        unmappedCount += 1;
        continue;
      }

      const weight = normalizeImportedScoreToWeight(score);
      byNodeId.set(anime.nodeId, { anime, weight });
    }
  }

  return {
    entries: [...byNodeId.values()],
    ratedCount,
    unmappedCount,
  };
}

async function fetchMalUsernameImport(
  username: string,
  index: RecommendationIndex,
): Promise<UsernameImportResult> {
  const allEntries: Array<{ anime_id?: number; score?: number }> = [];
  let offset = 0;

  while (true) {
    const page = await fetchMalUsernamePage(username, offset);
    if (!Array.isArray(page) || page.length === 0) {
      break;
    }

    allEntries.push(...page);
    if (page.length < USERNAME_IMPORT_PAGE_SIZE) {
      break;
    }

    offset += page.length;
    await sleep(USERNAME_IMPORT_PAGE_DELAY_MS);
  }

  if (allEntries.length === 0) {
    throw new Error("No rated MAL entries found for that user (private, empty, or not found).");
  }

  let ratedCount = 0;
  let unmappedCount = 0;
  const byNodeId = new Map<string, ImportedWatchedEntry>();

  for (const entry of allEntries) {
    const score = Number(entry.score ?? 0);
    const animeId = Number(entry.anime_id ?? 0);
    if (!Number.isFinite(score) || score <= 0 || !Number.isFinite(animeId) || animeId <= 0) {
      continue;
    }

    ratedCount += 1;
    const anime = index.animeByAnimeId.get(Math.trunc(animeId));
    if (!anime) {
      unmappedCount += 1;
      continue;
    }

    const weight = normalizeImportedScoreToWeight(score);
    byNodeId.set(anime.nodeId, { anime, weight });
  }

  return {
    entries: [...byNodeId.values()],
    ratedCount,
    unmappedCount,
  };
}

async function fetchMalUsernamePage(
  username: string,
  offset: number,
): Promise<Array<{ anime_id?: number; score?: number }>> {
  const malUrl = new URL(
    `https://myanimelist.net/animelist/${encodeURIComponent(username)}/load.json`,
  );
  malUrl.searchParams.set("status", "7");
  malUrl.searchParams.set("offset", String(offset));

  try {
    return await fetchJsonWithRetries<Array<{ anime_id?: number; score?: number }>>(
      malUrl.toString(),
      {
        headers: {
          Accept: "application/json",
        },
      },
      USERNAME_IMPORT_MAX_RETRIES,
    );
  } catch {
    const jinaUrl = `https://r.jina.ai/http://${malUrl.host}${malUrl.pathname}${malUrl.search}`;
    const text = await fetchTextWithRetries(jinaUrl);
    const parsed = parseJinaMarkdownJson<Array<{ anime_id?: number; score?: number }>>(text);
    if (!Array.isArray(parsed)) {
      throw new Error(
        "Unable to read MAL list data. The profile may be private or blocked by CORS/rate limits.",
      );
    }
    return parsed;
  }
}

function parseJinaMarkdownJson<T>(content: string): T | null {
  const marker = "Markdown Content:";
  const markerIndex = content.indexOf(marker);
  if (markerIndex < 0) {
    return null;
  }
  const jsonText = content.slice(markerIndex + marker.length).trim();
  if (!jsonText) {
    return null;
  }
  try {
    return JSON.parse(jsonText) as T;
  } catch {
    return null;
  }
}

function upsertImportedWatchedEntries(entries: ImportedWatchedEntry[]): {
  added: number;
  updated: number;
  skipped: number;
} {
  let added = 0;
  let updated = 0;
  let skipped = 0;

  for (const entry of entries) {
    const existingIndex = selectedAnimeNodeIds.indexOf(entry.anime.nodeId);
    if (existingIndex < 0) {
      selectedAnimeNodeIds.push(entry.anime.nodeId);
      selectedAnimeWeights.set(entry.anime.nodeId, entry.weight);
      added += 1;
      continue;
    }

    const current = clampWatchWeight(selectedAnimeWeights.get(entry.anime.nodeId) ?? 1);
    if (Math.abs(current - entry.weight) < 0.0001) {
      skipped += 1;
      continue;
    }

    selectedAnimeWeights.set(entry.anime.nodeId, entry.weight);
    updated += 1;
  }

  return { added, updated, skipped };
}

async function fetchJsonWithRetries<T>(
  url: string,
  init?: RequestInit,
  maxRetries = USERNAME_IMPORT_MAX_RETRIES,
): Promise<T> {
  for (let attempt = 0; attempt <= maxRetries; attempt += 1) {
    const response = await fetch(url, init);
    if (response.ok) {
      return (await response.json()) as T;
    }

    if (!isRetryableStatus(response.status) || attempt >= maxRetries) {
      throw new Error(`Request failed (${response.status}) for ${url}`);
    }

    await sleep(Math.min(1000 * 2 ** attempt, 5000) + Math.floor(Math.random() * 200));
  }

  throw new Error("Request retries exhausted.");
}

async function fetchTextWithRetries(
  url: string,
  maxRetries = USERNAME_IMPORT_MAX_RETRIES,
): Promise<string> {
  for (let attempt = 0; attempt <= maxRetries; attempt += 1) {
    const response = await fetch(url, {
      headers: { Accept: "text/plain" },
    });
    if (response.ok) {
      return await response.text();
    }

    if (!isRetryableStatus(response.status) || attempt >= maxRetries) {
      throw new Error(`Request failed (${response.status}) for ${url}`);
    }

    await sleep(Math.min(1000 * 2 ** attempt, 5000) + Math.floor(Math.random() * 200));
  }

  throw new Error("Request retries exhausted.");
}

function isRetryableStatus(status: number): boolean {
  return status === 408 || status === 425 || status === 429 || status >= 500;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function parseBulkWatchedLine(
  line: string,
  index: RecommendationIndex,
): { anime: AnimeInfo; weight?: number } | null {
  const parts = line
    .split(/[,\t|]/)
    .map((part) => part.trim())
    .filter((part) => part.length > 0);
  if (parts.length === 0) {
    return null;
  }

  const animeToken = parts[0];
  const anime = resolveAnimeInput(animeToken, index);
  if (!anime) {
    return null;
  }

  const scoreToken = parts[1];
  if (!scoreToken) {
    return { anime };
  }

  const scoreValue = Number.parseFloat(scoreToken);
  if (!Number.isFinite(scoreValue)) {
    return { anime };
  }

  return {
    anime,
    weight: normalizeImportedScoreToWeight(scoreValue),
  };
}

function normalizeImportedScoreToWeight(value: number): number {
  if (!Number.isFinite(value)) {
    return 1;
  }

  if (value <= MAX_WATCH_WEIGHT) {
    return clampWatchWeight(value);
  }

  const mapped = 1 + (value - 5) / 5;
  return clampWatchWeight(mapped);
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
    setMetadataStatus("Metadata: add anime to begin.");
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
    setMetadataStatus("Metadata: no candidates.");
    return;
  }

  const metadataCandidateAnimeIds = recommendations
    .slice(0, METADATA_PREFETCH_LIMIT)
    .map((item) => item.anime.animeId)
    .filter((animeId) => Number.isFinite(animeId) && animeId > 0);
  const metadataPrefetchLimit = hasActiveRecommendationFilters()
    ? METADATA_PREFETCH_WITH_FILTER_LIMIT
    : Math.min(12, metadataCandidateAnimeIds.length);
  if (metadataPrefetchLimit > 0) {
    await hydrateMetadataForAnimeIds(metadataCandidateAnimeIds, metadataPrefetchLimit);
    if (runId !== recommendationRunId) {
      return;
    }
  }

  updateGenreFilterOptions(recommendations);
  const filterResult = applyRecommendationFilters(recommendations);
  const filteredRecommendations = filterResult.recommendations;
  if (filteredRecommendations.length === 0) {
    recSummaryEl.textContent = hasActiveRecommendationFilters()
      ? "No recommendations match your current metadata filters."
      : "No positive recommendations found from the current watched list.";
    recResultsEl.innerHTML = "";
    setMetadataStatus(
      hasActiveRecommendationFilters() && filterResult.missingMetadataCount > 0
        ? `Metadata: ${filterResult.missingMetadataCount} candidate(s) skipped due to missing metadata.`
        : "Metadata: no matches after filters.",
    );
    return;
  }

  const methodLabel =
    recommendationMode === "graph"
      ? "graph edge ranking"
      : recommendationMode === "model"
        ? "ML model ranking"
        : "hybrid graph+ML ranking";
  const filterSummary = formatActiveFilterSummary();
  recSummaryEl.textContent = `Showing top ${Math.min(MAX_RECOMMENDATIONS, filteredRecommendations.length)} recommendations from ${filteredRecommendations.length} candidates (${methodLabel})${filterSummary}.`;

  const visibleRecommendations = filteredRecommendations
    .slice(0, MAX_RECOMMENDATIONS)
    .map((item) => renderRecommendationCard(item));
  recResultsEl.innerHTML = visibleRecommendations.join("");

  const visibleWithMetadata = filteredRecommendations
    .slice(0, MAX_RECOMMENDATIONS)
    .filter((item) => animeMetadataCache.has(item.anime.animeId)).length;
  const visibleMissingMetadata = filteredRecommendations
    .slice(0, MAX_RECOMMENDATIONS)
    .filter(
      (item) =>
        !animeMetadataCache.has(item.anime.animeId) &&
        !animeMetadataUnavailable.has(item.anime.animeId),
    ).length;
  const missingNote =
    filterResult.missingMetadataCount > 0
      ? ` | skipped (missing metadata): ${filterResult.missingMetadataCount}`
      : "";
  setMetadataStatus(
    `Metadata loaded for ${visibleWithMetadata}/${Math.min(MAX_RECOMMENDATIONS, filteredRecommendations.length)} visible recommendations` +
      `${visibleMissingMetadata > 0 ? ` | pending: ${visibleMissingMetadata}` : ""}` +
      missingNote,
  );
}

function renderRecommendationCard(item: RecommendationResult): string {
  const metadata = animeMetadataCache.get(item.anime.animeId) ?? null;
  const coverHtml =
    metadata && metadata.imageUrl
      ? `<img class="rec-cover" src="${escapeHtml(metadata.imageUrl)}" alt="Cover for ${escapeHtml(item.anime.label)}" loading="lazy" />`
      : `<div class="rec-cover rec-cover-placeholder" aria-hidden="true">No image</div>`;
  const metadataMeta = formatRecommendationMetadataMeta(metadata);
  const synopsisText =
    metadata && metadata.synopsis
      ? truncateText(metadata.synopsis, 180)
      : "Metadata loading...";
  const synopsisClass =
    metadata && metadata.synopsis
      ? "rec-synopsis"
      : "rec-synopsis rec-synopsis-muted";

  return `
      <li class="rec-item">
        <div class="rec-main">
          ${coverHtml}
          <div class="rec-copy">
            <div class="rec-title">${escapeHtml(item.anime.label)}</div>
            <div class="rec-meta">Support edges: ${item.supportCount} | Strongest: ${formatWeight(item.strongest)}</div>
            <div class="rec-meta rec-meta-details">${escapeHtml(metadataMeta)}</div>
            <p class="${synopsisClass}">${escapeHtml(synopsisText)}</p>
            <div class="rec-why">${formatRecommendationWhyHtml(item)}</div>
          </div>
        </div>
        <div class="rec-score">${formatWeight(item.score)}</div>
      </li>
    `;
}

function formatRecommendationMetadataMeta(metadata: AnimeMetadata | null): string {
  if (!metadata) {
    return "Metadata pending";
  }

  const parts: string[] = [];
  if (metadata.year !== null) {
    parts.push(String(metadata.year));
  }
  if (metadata.score !== null) {
    parts.push(`MAL ${metadata.score.toFixed(2)}`);
  }
  if (metadata.studios.length > 0) {
    parts.push(metadata.studios.slice(0, 2).join(", "));
  }
  if (metadata.genres.length > 0) {
    parts.push(metadata.genres.slice(0, 3).join(", "));
  }
  if (parts.length === 0) {
    return "Metadata available";
  }
  return parts.join(" | ");
}

function setMetadataStatus(message: string): void {
  metadataStatusEl.textContent = message;
}

function hasActiveRecommendationFilters(): boolean {
  return (
    recommendationFilters.genre.length > 0 ||
    recommendationFilters.minYear !== null ||
    recommendationFilters.maxYear !== null ||
    (recommendationFilters.minScore ?? 0) > 0
  );
}

function formatActiveFilterSummary(): string {
  const parts: string[] = [];
  if (recommendationFilters.genre) {
    parts.push(`genre=${recommendationFilters.genre}`);
  }
  if (recommendationFilters.minYear !== null) {
    parts.push(`year>=${recommendationFilters.minYear}`);
  }
  if (recommendationFilters.maxYear !== null) {
    parts.push(`year<=${recommendationFilters.maxYear}`);
  }
  if ((recommendationFilters.minScore ?? 0) > 0) {
    parts.push(`score>=${(recommendationFilters.minScore ?? 0).toFixed(1)}`);
  }
  return parts.length > 0 ? ` | filters: ${parts.join(", ")}` : "";
}

function applyRecommendationFilters(recommendations: RecommendationResult[]): {
  recommendations: RecommendationResult[];
  missingMetadataCount: number;
} {
  if (!hasActiveRecommendationFilters()) {
    return {
      recommendations,
      missingMetadataCount: 0,
    };
  }

  const minYearRaw = recommendationFilters.minYear;
  const maxYearRaw = recommendationFilters.maxYear;
  const lowerYear =
    minYearRaw !== null && maxYearRaw !== null
      ? Math.min(minYearRaw, maxYearRaw)
      : minYearRaw;
  const upperYear =
    minYearRaw !== null && maxYearRaw !== null
      ? Math.max(minYearRaw, maxYearRaw)
      : maxYearRaw;
  const minScore = recommendationFilters.minScore ?? 0;
  const genreFilter = recommendationFilters.genre.trim().toLowerCase();

  let missingMetadataCount = 0;
  const filtered = recommendations.filter((item) => {
    const metadata = animeMetadataCache.get(item.anime.animeId);
    if (!metadata) {
      missingMetadataCount += 1;
      return false;
    }

    if (genreFilter) {
      const hasGenre = metadata.genres.some(
        (genre) => genre.trim().toLowerCase() === genreFilter,
      );
      if (!hasGenre) {
        return false;
      }
    }

    if (lowerYear !== null) {
      if (metadata.year === null || metadata.year < lowerYear) {
        return false;
      }
    }

    if (upperYear !== null) {
      if (metadata.year === null || metadata.year > upperYear) {
        return false;
      }
    }

    if (minScore > 0) {
      if (metadata.score === null || metadata.score < minScore) {
        return false;
      }
    }

    return true;
  });

  return {
    recommendations: filtered,
    missingMetadataCount,
  };
}

function updateGenreFilterOptions(recommendations: RecommendationResult[]): void {
  const byNormalized = new Map<string, string>();
  for (const item of recommendations.slice(0, METADATA_PREFETCH_LIMIT)) {
    const metadata = animeMetadataCache.get(item.anime.animeId);
    if (!metadata) {
      continue;
    }
    for (const genre of metadata.genres) {
      const normalized = genre.trim().toLowerCase();
      if (!normalized || byNormalized.has(normalized)) {
        continue;
      }
      byNormalized.set(normalized, genre);
    }
  }

  if (recommendationFilters.genre && !byNormalized.has(recommendationFilters.genre)) {
    byNormalized.set(recommendationFilters.genre, recommendationFilters.genre);
  }

  const sortedGenres = [...byNormalized.entries()].sort((left, right) =>
    left[1].localeCompare(right[1]),
  );
  filterGenreSelect.innerHTML = [
    `<option value="">Any</option>`,
    ...sortedGenres.map(
      ([normalized, label]) =>
        `<option value="${escapeHtml(normalized)}">${escapeHtml(label)}</option>`,
    ),
  ].join("");
  filterGenreSelect.value = recommendationFilters.genre;
}

async function hydrateMetadataForAnimeIds(
  animeIds: number[],
  maxToFetch: number,
): Promise<void> {
  const uniqueTargets = [...new Set(animeIds)]
    .filter(
      (animeId) =>
        animeId > 0 &&
        !animeMetadataCache.has(animeId) &&
        !animeMetadataUnavailable.has(animeId),
    )
    .slice(0, maxToFetch);
  if (uniqueTargets.length === 0) {
    return;
  }

  const queue = [...uniqueTargets];
  const workerCount = Math.min(METADATA_PREFETCH_CONCURRENCY, queue.length);
  const workers: Promise<void>[] = [];

  for (let i = 0; i < workerCount; i += 1) {
    workers.push(
      (async () => {
        while (queue.length > 0) {
          const animeId = queue.shift();
          if (animeId === undefined) {
            return;
          }
          await ensureAnimeMetadata(animeId);
        }
      })(),
    );
  }

  await Promise.all(workers);
}

async function ensureAnimeMetadata(animeId: number): Promise<AnimeMetadata | null> {
  const cached = animeMetadataCache.get(animeId);
  if (cached) {
    return cached;
  }
  if (animeMetadataUnavailable.has(animeId)) {
    return null;
  }
  const inflight = animeMetadataInFlight.get(animeId);
  if (inflight) {
    return inflight;
  }

  const promise = fetchAnimeMetadataFromJikan(animeId)
    .then((metadata) => {
      if (!metadata) {
        animeMetadataUnavailable.add(animeId);
        return null;
      }
      animeMetadataCache.set(animeId, metadata);
      return metadata;
    })
    .finally(() => {
      animeMetadataInFlight.delete(animeId);
    });

  animeMetadataInFlight.set(animeId, promise);
  return await promise;
}

async function fetchAnimeMetadataFromJikan(
  animeId: number,
): Promise<AnimeMetadata | null> {
  const url = `https://api.jikan.moe/v4/anime/${animeId}/full`;

  for (let attempt = 0; attempt <= METADATA_MAX_RETRIES; attempt += 1) {
    try {
      const response = await fetch(url, {
        headers: {
          Accept: "application/json",
        },
      });
      if (response.ok) {
        const payload = (await response.json()) as {
          data?: Record<string, unknown>;
        };
        return parseAnimeMetadataPayload(animeId, payload.data);
      }
      if (response.status === 404) {
        return null;
      }
      if (!isRetryableStatus(response.status) || attempt >= METADATA_MAX_RETRIES) {
        return null;
      }
    } catch {
      if (attempt >= METADATA_MAX_RETRIES) {
        return null;
      }
    }

    await sleep(Math.min(800 * 2 ** attempt, 5000) + Math.floor(Math.random() * 220));
  }

  return null;
}

function parseAnimeMetadataPayload(
  animeId: number,
  raw: Record<string, unknown> | undefined,
): AnimeMetadata | null {
  if (!raw || typeof raw !== "object") {
    return null;
  }

  const year =
    typeof raw.year === "number" && Number.isFinite(raw.year)
      ? Math.trunc(raw.year)
      : null;
  const score =
    typeof raw.score === "number" && Number.isFinite(raw.score)
      ? Number(raw.score)
      : null;
  const synopsis = typeof raw.synopsis === "string" ? raw.synopsis.trim() : "";
  const season = typeof raw.season === "string" ? raw.season : null;
  const genres = parseNameList(raw.genres);
  const studios = parseNameList(raw.studios);

  const images = raw.images as Record<string, unknown> | undefined;
  const webp = images?.webp as Record<string, unknown> | undefined;
  const jpg = images?.jpg as Record<string, unknown> | undefined;
  const imageUrl = [
    webp?.large_image_url,
    webp?.image_url,
    jpg?.large_image_url,
    jpg?.image_url,
  ].find((value): value is string => typeof value === "string" && value.length > 0);

  return {
    animeId,
    year,
    score,
    genres,
    studios,
    synopsis,
    imageUrl: imageUrl ?? "",
    season,
  };
}

function parseNameList(raw: unknown): string[] {
  if (!Array.isArray(raw)) {
    return [];
  }
  const values: string[] = [];
  for (const entry of raw) {
    if (!entry || typeof entry !== "object") {
      continue;
    }
    const name = (entry as Record<string, unknown>).name;
    if (typeof name !== "string") {
      continue;
    }
    const trimmed = name.trim();
    if (!trimmed || values.includes(trimmed)) {
      continue;
    }
    values.push(trimmed);
  }
  return values;
}

function truncateText(value: string, maxLength: number): string {
  const normalized = value.trim().replace(/\s+/g, " ");
  if (normalized.length <= maxLength) {
    return normalized;
  }
  return `${normalized.slice(0, maxLength - 3)}...`;
}

async function loadSeasonalTrending(force: boolean): Promise<void> {
  if (seasonalLoadingPromise && !force) {
    return seasonalLoadingPromise;
  }

  refreshSeasonalBtn.disabled = true;
  seasonalStatusEl.textContent = "Loading current season...";

  const promise = (async () => {
    try {
      const payload = await fetchJsonWithRetries<{
        data?: Array<Record<string, unknown>>;
      }>(
        `https://api.jikan.moe/v4/seasons/now?limit=${SEASONAL_LIST_LIMIT}`,
        {
          headers: {
            Accept: "application/json",
          },
        },
      );
      const incoming = Array.isArray(payload.data) ? payload.data : [];
      seasonalItems = incoming
        .map((entry) => {
          const animeId = Number((entry as Record<string, unknown>).mal_id ?? 0);
          const title = String((entry as Record<string, unknown>).title ?? "").trim();
          if (!Number.isFinite(animeId) || animeId <= 0 || !title) {
            return null;
          }
          const scoreRaw = (entry as Record<string, unknown>).score;
          const score =
            typeof scoreRaw === "number" && Number.isFinite(scoreRaw)
              ? scoreRaw
              : null;
          const yearRaw = (entry as Record<string, unknown>).year;
          const year =
            typeof yearRaw === "number" && Number.isFinite(yearRaw)
              ? Math.trunc(yearRaw)
              : null;
          const seasonRaw = (entry as Record<string, unknown>).season;
          const season = typeof seasonRaw === "string" ? seasonRaw : null;
          const imageUrl = parseAnimeMetadataPayload(animeId, entry)?.imageUrl ?? "";

          return {
            animeId,
            title,
            score,
            year,
            season,
            imageUrl,
          } satisfies SeasonalAnimeItem;
        })
        .filter((entry): entry is SeasonalAnimeItem => entry !== null);
      seasonalStatusEl.textContent =
        seasonalItems.length > 0
          ? `Loaded ${seasonalItems.length} seasonal anime from Jikan.`
          : "No seasonal anime returned.";
      renderSeasonalList();
    } catch (error) {
      const message = error instanceof Error ? error.message : "Request failed.";
      seasonalStatusEl.textContent = `Unable to load seasonal anime: ${message}`;
      seasonalItems = [];
      renderSeasonalList();
    } finally {
      refreshSeasonalBtn.disabled = false;
      seasonalLoadingPromise = null;
    }
  })();

  seasonalLoadingPromise = promise;
  await promise;
}

function renderSeasonalList(): void {
  if (seasonalItems.length === 0) {
    seasonalListEl.innerHTML = `<li class="seasonal-empty">No seasonal data yet.</li>`;
    return;
  }

  seasonalListEl.innerHTML = seasonalItems
    .map((item) => {
      const inGraph = recommendationIndex.animeByAnimeId.has(item.animeId);
      const subtitleParts: string[] = [];
      if (item.season) {
        subtitleParts.push(item.season);
      }
      if (item.year !== null) {
        subtitleParts.push(String(item.year));
      }
      if (item.score !== null) {
        subtitleParts.push(`MAL ${item.score.toFixed(2)}`);
      }
      const subtitle = subtitleParts.length > 0 ? subtitleParts.join(" | ") : "No stats";
      const coverHtml = item.imageUrl
        ? `<img class="seasonal-cover" src="${escapeHtml(item.imageUrl)}" alt="Cover for ${escapeHtml(item.title)}" loading="lazy" />`
        : `<div class="seasonal-cover seasonal-cover-placeholder" aria-hidden="true">No image</div>`;
      return `
        <li class="seasonal-item">
          ${coverHtml}
          <div class="seasonal-copy">
            <div class="seasonal-title">${escapeHtml(item.title)}</div>
            <div class="seasonal-meta">${escapeHtml(subtitle)}</div>
          </div>
          <button
            type="button"
            data-seasonal-anime-id="${item.animeId}"
            ${inGraph ? "" : "disabled"}
            title="${inGraph ? "Add to watched list" : "Not available in current graph"}"
          >
            ${inGraph ? "Add" : "N/A"}
          </button>
        </li>
      `;
    })
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
      const weightedScore = neighbor.weight * weightFactor;

      const current = scored.get(neighbor.otherNodeId);
      if (neighbor.weight <= 0) {
        if (current) {
          current.sourceMap.set(selectedNodeId, {
            edgeWeight: neighbor.weight,
            weightFactor,
            weightedScore,
          });
        }
        continue;
      }

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
    camera.animate(
      { x, y, ratio: 0.32 },
      { duration: prefersReducedMotion() ? 0 : 360 },
    );
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

function formatRecommendationWhyHtml(result: RecommendationResult): string {
  if (result.contributions.length === 0) {
    return `<div class="rec-why-line">Why: no direct contributing anime found.</div>`;
  }

  const positives = result.contributions
    .filter((item) => item.weightedScore > 0)
    .sort((left, right) => right.weightedScore - left.weightedScore)
    .slice(0, 2);
  const negatives = result.contributions
    .filter((item) => item.weightedScore < 0)
    .sort((left, right) => left.weightedScore - right.weightedScore)
    .slice(0, 2);

  const positiveLine =
    positives.length > 0
      ? `Why+: ${positives
          .map(
            (item) =>
              `${escapeHtml(item.watched.label)} (${formatWeight(item.weightedScore)})`,
          )
          .join(" | ")}`
      : "Why+: no strong positive contributors.";

  const negativeLine =
    negatives.length > 0
      ? `Why-: ${negatives
          .map(
            (item) =>
              `${escapeHtml(item.watched.label)} (${formatWeight(item.weightedScore)})`,
          )
          .join(" | ")}`
      : "Why-: no notable negative contributors.";

  return [
    `<div class="rec-why-line rec-why-pos">${positiveLine}</div>`,
    `<div class="rec-why-line rec-why-neg">${negativeLine}</div>`,
  ].join("");
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

function loadThemeModePreference(): ThemeMode {
  try {
    const raw = window.localStorage.getItem(THEME_STORAGE_KEY);
    if (raw === "dark" || raw === "light") {
      return raw;
    }
  } catch (error) {
    console.warn("Unable to read saved theme preference.", error);
  }

  if (window.matchMedia("(prefers-color-scheme: light)").matches) {
    return "light";
  }
  return "dark";
}

function persistThemeModePreference(theme: ThemeMode): void {
  try {
    window.localStorage.setItem(THEME_STORAGE_KEY, theme);
  } catch (error) {
    console.warn("Unable to persist theme preference.", error);
  }
}

function applyTheme(theme: ThemeMode): void {
  document.documentElement.dataset.theme = theme;
  themeToggleBtn.textContent = theme === "dark" ? "Theme: Dark" : "Theme: Light";
  themeToggleBtn.setAttribute(
    "aria-label",
    theme === "dark" ? "Switch to light theme" : "Switch to dark theme",
  );
}

function prefersReducedMotion(): boolean {
  return reduceMotionMediaQuery.matches;
}

function loadRecommendationState(index: RecommendationIndex): {
  mode: RecommendationMode;
  selected: { nodeId: string; weight: number }[];
  modelBlendWeight: number;
  includeCandidates: string[];
  excludeCandidates: string[];
} {
  const emptyState = {
    mode: "graph" as RecommendationMode,
    selected: [],
    modelBlendWeight: 0.5,
    includeCandidates: [],
    excludeCandidates: [],
  };

  try {
    const raw = window.localStorage.getItem(RECOMMENDATION_STATE_STORAGE_KEY);
    if (!raw) {
      return emptyState;
    }
    const parsed = JSON.parse(raw) as StoredRecommendationState;
    return sanitizeRecommendationState(parsed, index);
  } catch (error) {
    console.warn("Unable to load saved recommendation state.", error);
    return emptyState;
  }
}

function persistRecommendationState(): void {
  try {
    const payload = buildCurrentRecommendationState();
    window.localStorage.setItem(
      RECOMMENDATION_STATE_STORAGE_KEY,
      JSON.stringify(payload),
    );
  } catch (error) {
    console.warn("Unable to persist recommendation state.", error);
  }
}

function buildCurrentRecommendationState(): StoredRecommendationState {
  const selected = selectedAnimeNodeIds
    .filter((nodeId) => recommendationIndex.animeByNodeId.has(nodeId))
    .map((nodeId) => ({
      nodeId,
      weight: clampWatchWeight(selectedAnimeWeights.get(nodeId) ?? 1),
    }));

  return {
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
}

function sanitizeRecommendationState(
  parsed: StoredRecommendationState | null | undefined,
  index: RecommendationIndex,
): {
  mode: RecommendationMode;
  selected: { nodeId: string; weight: number }[];
  modelBlendWeight: number;
  includeCandidates: string[];
  excludeCandidates: string[];
} {
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
    ? parsed.includeCandidates.filter(
        (entry) => typeof entry === "string" && index.animeByNodeId.has(entry),
      )
    : [];
  const excludeCandidates = Array.isArray(parsed.excludeCandidates)
    ? parsed.excludeCandidates.filter(
        (entry) => typeof entry === "string" && index.animeByNodeId.has(entry),
      )
    : [];

  return {
    mode,
    selected,
    modelBlendWeight,
    includeCandidates,
    excludeCandidates,
  };
}

function loadRecommendationProfiles(): Map<string, RecommendationProfileRecord> {
  const profiles = new Map<string, RecommendationProfileRecord>();
  try {
    const raw = window.localStorage.getItem(RECOMMENDATION_PROFILES_STORAGE_KEY);
    if (!raw) {
      return profiles;
    }
    const parsed = JSON.parse(raw) as RecommendationProfileRecord[];
    if (!Array.isArray(parsed)) {
      return profiles;
    }

    for (const record of parsed) {
      if (!record || typeof record.name !== "string" || !record.state) {
        continue;
      }
      const name = record.name.trim();
      if (!name) {
        continue;
      }
      const sanitized = sanitizeRecommendationState(
        record.state,
        recommendationIndex,
      );
      profiles.set(name, {
        name,
        updatedAt:
          typeof record.updatedAt === "string" ? record.updatedAt : new Date().toISOString(),
        state: {
          version: 3,
          mode: sanitized.mode,
          selected: sanitized.selected,
          modelBlendWeight: sanitized.modelBlendWeight,
          includeCandidates: sanitized.includeCandidates,
          excludeCandidates: sanitized.excludeCandidates,
        },
      });
    }
  } catch (error) {
    console.warn("Unable to load saved profiles.", error);
  }
  return profiles;
}

function persistRecommendationProfiles(
  profiles: Map<string, RecommendationProfileRecord>,
): void {
  try {
    const payload = [...profiles.values()]
      .sort((left, right) => left.name.localeCompare(right.name))
      .map((profile) => ({
        name: profile.name,
        updatedAt: profile.updatedAt,
        state: profile.state,
      }));
    window.localStorage.setItem(
      RECOMMENDATION_PROFILES_STORAGE_KEY,
      JSON.stringify(payload),
    );
  } catch (error) {
    console.warn("Unable to persist saved profiles.", error);
  }
}

function renderProfileOptions(profiles: Map<string, RecommendationProfileRecord>): void {
  const names = [...profiles.keys()].sort((left, right) => left.localeCompare(right));
  if (names.length === 0) {
    profileSelect.innerHTML = `<option value="">No saved profiles</option>`;
    profileSelect.disabled = true;
    profileLoadBtn.disabled = true;
    profileDeleteBtn.disabled = true;
    return;
  }

  profileSelect.innerHTML = names
    .map((name) => `<option value="${escapeHtml(name)}">${escapeHtml(name)}</option>`)
    .join("");
  profileSelect.disabled = false;
  profileLoadBtn.disabled = false;
  profileDeleteBtn.disabled = false;
}

function saveCurrentProfile(): void {
  const profileName = profileNameInput.value.trim();
  if (!profileName) {
    recMessageEl.textContent = "Enter a profile name first.";
    return;
  }

  const state = buildCurrentRecommendationState();
  const now = new Date().toISOString();
  savedProfiles.set(profileName, {
    name: profileName,
    updatedAt: now,
    state,
  });
  persistRecommendationProfiles(savedProfiles);
  renderProfileOptions(savedProfiles);
  profileSelect.value = profileName;
  profileNameInput.value = "";
  recMessageEl.textContent = `Saved profile "${profileName}".`;
}

function loadSelectedProfile(): void {
  const name = profileSelect.value.trim();
  if (!name) {
    recMessageEl.textContent = "Select a saved profile first.";
    return;
  }
  const profile = savedProfiles.get(name);
  if (!profile) {
    recMessageEl.textContent = `Saved profile "${name}" was not found.`;
    return;
  }

  const sanitized = sanitizeRecommendationState(profile.state, recommendationIndex);
  applyRecommendationState(sanitized);
  persistRecommendationState();
  renderSelectedAnime();
  renderIncludeCandidates();
  renderExcludeCandidates();
  void updateRecommendations();
  recMessageEl.textContent = `Loaded profile "${name}".`;
}

function deleteSelectedProfile(): void {
  const name = profileSelect.value.trim();
  if (!name) {
    recMessageEl.textContent = "Select a saved profile first.";
    return;
  }
  if (!savedProfiles.has(name)) {
    recMessageEl.textContent = `Saved profile "${name}" was not found.`;
    return;
  }

  savedProfiles.delete(name);
  persistRecommendationProfiles(savedProfiles);
  renderProfileOptions(savedProfiles);
  recMessageEl.textContent = `Deleted profile "${name}".`;
}

function applyRecommendationState(state: {
  mode: RecommendationMode;
  selected: { nodeId: string; weight: number }[];
  modelBlendWeight: number;
  includeCandidates: string[];
  excludeCandidates: string[];
}): void {
  selectedAnimeNodeIds.splice(0, selectedAnimeNodeIds.length);
  selectedAnimeWeights.clear();
  includeCandidateNodeIds.splice(0, includeCandidateNodeIds.length);
  excludeCandidateNodeIds.splice(0, excludeCandidateNodeIds.length);

  for (const entry of state.selected) {
    if (!recommendationIndex.animeByNodeId.has(entry.nodeId)) {
      continue;
    }
    selectedAnimeNodeIds.push(entry.nodeId);
    selectedAnimeWeights.set(entry.nodeId, clampWatchWeight(entry.weight));
  }
  for (const nodeId of state.includeCandidates) {
    if (recommendationIndex.animeByNodeId.has(nodeId)) {
      includeCandidateNodeIds.push(nodeId);
    }
  }
  for (const nodeId of state.excludeCandidates) {
    if (recommendationIndex.animeByNodeId.has(nodeId)) {
      excludeCandidateNodeIds.push(nodeId);
    }
  }

  recommendationMode = state.mode;
  modelBlendWeight = clampModelBlendWeight(state.modelBlendWeight);
  recMethodSelect.value = recommendationMode;
  recBlendInput.value = modelBlendWeight.toFixed(2);
  renderModelBlendValue();
  setBlendControlVisibility();
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
