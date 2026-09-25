import Graph from "graphology";
import { isCompactGraphData } from "./artifacts";
import type {
  AnimeMetadata,
  EdgeType,
  GraphEdge,
  GraphNode,
  LoadedGraphData,
  NodeType,
} from "./artifacts";
import type {
  AnimeInfo,
  ConnectedItem,
  ImportedPreferenceEntry,
  ModelRecommendationIndex,
  RecommendationIndex,
  RecommendationResult,
  SeasonalAnimeItem,
} from "./domain";
import {
  MAX_MODEL_BLEND_WEIGHT,
  MIN_MODEL_BLEND_WEIGHT,
  buildCatalogCoverageRecommendations,
  buildGraphRecommendationsForPreferences,
  buildModelRecommendationsForPreferences,
  buildRecommendationIndex,
  buildRecommendationIndexFromCompact,
  clampModelBlendWeight,
  createCandidateEligibilityPolicy,
  explainRecommendation,
  formatWeight,
  hasActiveRecommendationFilters,
  normalizeTitle,
  rankEligibleCandidates,
} from "./recommendations";
import type { EligibilityRankingMode, RecommendationFilters } from "./recommendations";
import { ProviderUnavailableError, createProviderAdapter } from "./providers";
import { createArtifactLoader } from "./artifact-loader";
import {
  MAX_MAL_XML_IMPORT_BYTES,
  MAX_TEXT_IMPORT_BYTES,
  mergeHistory,
  parseMalXmlHistory,
  parseTextHistory,
  previewHistory,
  resolveHistoryAnime,
} from "./import-history";
import type { HistoryEntry, ImportMode, ParsedHistory } from "./import-history";
import { MAX_IMPORTANCE, MIN_IMPORTANCE, clampImportance, manualPreference, preferenceFromHistory } from "./preferences";
import type { AnimePreference, PreferenceSentiment } from "./preferences";
import {
  COMMAND_HISTORY_LIMIT,
  COMMAND_PINNED_LIMIT,
  RECOMMENDATION_STORAGE_VERSION,
  createPersistenceAdapter,
} from "./persistence";
import type {
  ContrastMode,
  RecommendationMode,
  RecommendationProfileRecord,
  StoredRecommendationState,
  ThemeMode,
} from "./persistence";
import { createBrowserRuntime, isAbortError } from "./runtime";
import { safeExternalImageUrl } from "./safe-url";
import "./style.css";

const runtime = createBrowserRuntime();
const providerAdapter = createProviderAdapter(runtime);
const demoMode = import.meta.env.VITE_DEMO_MODE === "true";
const artifactLoader = createArtifactLoader(runtime, demoMode);
const persistence = createPersistenceAdapter(runtime, demoMode ? "wasiw.demo" : "wasiw");

type AppView = "recommendations" | "network";
type UsernameImportProvider = "anilist" | "mal";
type AsyncUiState = "idle" | "loading" | "ready" | "empty" | "unavailable" | "failed" | "stale" | "demo";
type CommandMatchReason =
  | "pinned"
  | "exact"
  | "prefix"
  | "word"
  | "contains"
  | "fuzzy"
  | "recent";

const MAX_RENDERED_ANIME_ANIME_EDGES = 12000;
const MAX_RENDERED_USER_ANIME_EDGES = 4000;
const INSPECT_MAX_ITEMS = 250;
const MAX_RECOMMENDATIONS = 40;
const IMPORTANCE_STEP = 0.1;
const MODEL_BLEND_WEIGHT_STEP = 0.05;
const METADATA_PREFETCH_LIMIT = 100;
const METADATA_PREFETCH_WITH_FILTER_LIMIT = 30;
const METADATA_PREFETCH_CONCURRENCY = 3;
const METADATA_SCORE_STEP = 0.1;
const SEASONAL_LIST_LIMIT = 12;
const NETWORK_MOBILE_COMPACT_MAX_WIDTH = 980;
const SVG_NS = "http://www.w3.org/2000/svg";

type CommandGroup = "Navigation" | "Display" | "Utilities";

interface CommandAction {
  id: string;
  label: string;
  group: CommandGroup;
  shortcutLabel?: string;
  keywords: string[];
  run: () => void;
}

interface CommandSection {
  group: string;
  actions: CommandAction[];
}

interface CommandTokenMatch {
  score: number;
  reason: Exclude<CommandMatchReason, "recent" | "pinned">;
}

const app = document.querySelector<HTMLDivElement>("#app");
if (!app) {
  throw new Error("Missing #app container");
}

app.innerHTML = `
  <div class="app-shell">
    <header class="topbar">
      <div class="brand">
        <p class="eyebrow">Graph + ML Recommendation Lab</p>
        <h1>What Anime Should I Watch</h1>
        ${demoMode ? '<p class="demo-banner" role="status">SYNTHETIC DEMO DATA · Invented titles and ratings · No live metadata requests</p>' : ""}
        <p>Recommendation-first anime discovery powered by the rating network.</p>
        <p class="shortcut-hint">Shortcuts: Alt+1 Recommendations, Alt+2 Network, Ctrl/Cmd+K Commands, Alt+/ Focus</p>
      </div>
      <nav class="topnav" aria-label="Primary">
        <button id="nav-recommendations" class="nav-btn" type="button" aria-label="Open recommendations page" aria-keyshortcuts="Alt+1">
          <span class="nav-kicker">01</span>
          <span>Recommendations</span>
        </button>
        <button id="nav-network" class="nav-btn" type="button" aria-label="Open network explorer page" aria-keyshortcuts="Alt+2">
          <span class="nav-kicker">02</span>
          <span>Network Explorer</span>
        </button>
        <button id="theme-toggle" class="nav-btn theme-btn" type="button" aria-label="Toggle theme" aria-keyshortcuts="Alt+T">
          <span class="icon icon-theme" aria-hidden="true"></span>
          <span id="theme-toggle-label">Theme</span>
        </button>
        <button id="contrast-toggle" class="nav-btn contrast-btn" type="button" aria-label="Toggle high contrast mode" aria-keyshortcuts="Alt+C">
          <span class="icon icon-contrast" aria-hidden="true"></span>
          <span id="contrast-toggle-label">Contrast: Normal</span>
        </button>
        <button id="tips-toggle" class="nav-btn tips-btn" type="button" aria-label="Hide help tips" aria-pressed="true" aria-keyshortcuts="Alt+H">
          <span class="icon icon-help" aria-hidden="true"></span>
          <span id="tips-toggle-label">Hide Tips</span>
        </button>
        <button id="commands-toggle" class="nav-btn commands-btn" type="button" aria-label="Open command palette" aria-keyshortcuts="Control+K Meta+K">
          <span class="icon icon-command" aria-hidden="true"></span>
          <span>Commands</span>
        </button>
      </nav>
    </header>

    <div id="command-palette" class="command-palette" hidden aria-hidden="true">
      <div class="command-backdrop" data-command-close="true"></div>
      <section class="command-dialog" role="dialog" aria-modal="true" aria-labelledby="command-title">
        <div class="command-head">
          <h2 id="command-title">Quick Actions</h2>
          <button id="command-close" type="button" class="ghost-btn" aria-label="Close command palette">Close</button>
        </div>
        <input id="command-input" type="text" autocomplete="off" placeholder="Type an action (e.g. network, theme, import)" />
        <p id="command-hint" class="muted command-hint">Enter to run selected action. Keys 1-9 run visible commands instantly. Esc closes. Fuzzy search enabled. Use the star to pin favorites and drag the grip to reorder.</p>
        <ul id="command-list" class="command-list"></ul>
      </section>
    </div>

    <main>
      <section id="view-recommendations" class="view">
        <section id="tips-recommendations" class="panel contextual-tip" hidden>
          <div class="contextual-tip-head">
            <h3>Quick Tips</h3>
            <button id="tips-dismiss-recommendations" class="ghost-btn" type="button">Dismiss Tips</button>
          </div>
          <ul class="contextual-tip-list">
            <li>Mark titles you liked or disliked; Seen alone does not become a favorite.</li>
            <li>Use <strong>Hybrid</strong> mode when model data is available for best balance.</li>
            <li>Use Include Only to limit scored candidates; exclusions and watched titles always win.</li>
            <li>Keyboard: <strong>Alt+/</strong> focuses the main anime input instantly.</li>
          </ul>
        </section>

        <section class="panel intro-panel">
          <div class="intro-copy">
            <h2>Start in 3 Steps</h2>
            <p class="muted">
              Add what you watched, state how you felt, then review ranked picks with explainable reasons.
            </p>
          </div>
          <ol class="intro-steps">
            <li><strong>Step 1:</strong> Add a title and mark it Liked, Disliked, or Seen/Unrated.</li>
            <li><strong>Step 2:</strong> Pick Graph, Model, or Hybrid ranking.</li>
            <li><strong>Step 3:</strong> Use filters and inspect recommendation reasons.</li>
          </ol>
          <div class="intro-actions">
            <button id="quickstart-seasonal" type="button" class="primary-btn">${demoMode ? "Browse Demo Ideas" : "Browse Seasonal Ideas"}</button>
            <button id="quickstart-network" type="button" class="ghost-btn">Open Network Explorer</button>
          </div>
        </section>

        <div class="recommend-layout">
          <section class="card">
            <h2>Find Your Next Anime</h2>
            <p class="muted">${demoMode ? "Choose invented anime, then compare graph edges with a tiny synthetic model." : "Add anime you have seen, then mark your preference to rank next picks."}</p>

            <label class="rec-engine-control" for="rec-method">
              <span>Recommendation engine</span>
              <select id="rec-method">
                <option value="graph" selected>Graph (anime-to-anime edges)</option>
                <option value="model">${demoMode ? "Synthetic Model" : "ML Model (matrix factorization)"}</option>
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
              <input id="anime-input" type="text" list="anime-options" autocomplete="off" placeholder="Type an anime title" aria-label="Anime title input" />
              <select id="add-preference" aria-label="Preference for added anime">
                <option value="seen" selected>Seen / Unrated</option>
                <option value="liked">Liked</option>
                <option value="disliked">Disliked</option>
              </select>
              <button type="submit" class="primary-btn">Add</button>
            </form>
            <datalist id="anime-options"></datalist>
            <p class="muted" id="preference-guidance">Seen titles are excluded from picks without acting as likes. Liked titles seed graph suggestions; the model also uses dislikes as negative evidence. Importance is your emphasis; confidence records how certain an imported score is. Your manual choice takes precedence over later imports.</p>
            <p id="preference-migration-notice" class="muted" role="status" hidden></p>

            <p id="rec-message" class="rec-message" role="status" aria-live="polite"></p>
            <p id="storage-status" class="storage-status" role="alert" aria-live="assertive"></p>

            <div class="selected-head">
              <h3>Watched &amp; Preferences <span id="watched-count" class="count-pill">0</span></h3>
              <button id="clear-watched" type="button" class="ghost-btn">Clear List</button>
            </div>
            <div id="selected-anime" class="selected-anime"></div>

            <details class="accordion">
              <summary>Import & Profiles</summary>
              <section class="bulk-import">
                <h3>Import From Local File or Text (Recommended)</h3>
                <p class="muted">Choose a local MAL-style XML export, a plain-text list, or paste text below. Text lines use <code>animeId[, score[, status[, episodes]]]</code> or a title in place of the ID. Files stay in this browser. Review counts and choose merge or replace before applying.</p>
                <form id="bulk-import-form" class="bulk-import-form">
                  <label for="bulk-import-file">Local .txt (up to 128 KiB) or .xml (up to 2 MiB)</label>
                  <input id="bulk-import-file" type="file" accept=".txt,.xml,text/plain,application/xml,text/xml" />
                  <textarea id="bulk-import-input" rows="6" placeholder="5114, 9&#10;anime:9253, 7.5&#10;Steins;Gate"></textarea>
                  <button type="submit">
                    <span class="icon icon-import" aria-hidden="true"></span>
                    <span>Preview Text Import</span>
                  </button>
                </form>
                <p id="bulk-import-status" class="muted" role="status" aria-live="polite" data-state="idle"></p>
              </section>

              <section class="username-import">
                <h3>Import From Username</h3>
                <p class="muted">${demoMode ? "Username imports are unavailable in the offline demo." : "Your entered username is sent directly to the selected provider to read its public anime list. MAL may block browser requests; no third-party proxy is used. The local file or text import above needs no provider username."}</p>
                <form id="username-import-form" class="username-import-form">
                  <select id="username-import-provider" aria-label="Import provider">
                    <option value="anilist" selected>AniList</option>
                    <option value="mal">MyAnimeList (MAL, direct)</option>
                  </select>
                  <input id="username-import-input" type="text" autocomplete="off" placeholder="Enter username" />
                  <button id="username-import-submit" type="submit">
                    <span class="icon icon-import" aria-hidden="true"></span>
                    <span id="username-import-submit-label">Import User List</span>
                  </button>
                </form>
                <p id="username-import-status" class="muted" role="status" aria-live="polite" data-state="idle"></p>
              </section>

              <section id="history-import-preview" class="import-preview" hidden>
                <h3>Import Preview</h3>
                <p id="history-import-summary" role="status" aria-live="polite"></p>
                <p id="history-import-unmapped" class="muted"></p>
                <label for="history-import-mode">Apply as</label>
                <select id="history-import-mode" aria-label="Import mode">
                  <option value="merge">Merge with current history and watched picks</option>
                  <option value="replace">Replace current history and watched picks</option>
                </select>
                <button id="history-import-apply" type="button" class="primary-btn">Apply Import</button>
              </section>

              <section class="imported-history">
                <h3>Imported History <span id="history-count" class="count-pill">0</span></h3>
                <p class="muted">Statuses, episode progress, original score scale, and titles missing from this catalog are kept with your local recommendation state and saved profiles.</p>
                <ul id="history-list"></ul>
              </section>

              <section class="profiles">
                <h3>Saved Profiles</h3>
                <p class="muted">Save and reload named recommendation setups.</p>
                <form id="profile-save-form" class="profile-save-form">
                  <input id="profile-name-input" type="text" autocomplete="off" placeholder="Profile name" />
                  <button id="profile-save-submit" type="submit">
                    <span class="icon icon-save" aria-hidden="true"></span>
                    <span>Save Profile</span>
                  </button>
                </form>
                <div class="profile-load-row">
                  <select id="profile-select" aria-label="Saved profile"></select>
                  <button id="profile-load-btn" type="button">
                    <span class="icon icon-load" aria-hidden="true"></span>
                    <span>Load</span>
                  </button>
                  <button id="profile-delete-btn" type="button" class="ghost-btn">Delete</button>
                </div>
              </section>
            </details>

            <details class="accordion">
              <summary>Candidate Overrides</summary>
              <p class="muted">Include Only narrows the current ranking to listed titles. It does not add a title to graph or model rankings or override watched titles, exclusions, catalog availability, or required filters. If catalog fallback is active, it narrows that catalog list. Exclusions always win.</p>
              <div class="selected-head">
                <h3>Include Only Candidates</h3>
                <button id="clear-include" type="button" class="ghost-btn">Clear</button>
              </div>
              <form id="add-include-form" class="add-form add-form-compact">
                <input id="include-input" type="text" list="anime-options" autocomplete="off" placeholder="Limit results to this anime" />
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
            </details>
          </section>

          <section class="card">
            <h2>Top Recommendations</h2>
            <details class="accordion" open>
              <summary>Filters</summary>
              <section class="rec-filters rec-filters-inline">
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
                  <span>Minimum ${demoMode ? "demo" : "MAL"} score</span>
                  <input id="filter-min-score" type="range" min="0" max="10" step="${METADATA_SCORE_STEP}" value="0" />
                  <output id="filter-min-score-value">Any</output>
                </label>
                <div class="rec-filter-actions">
                  <button id="clear-rec-filters" type="button" class="ghost-btn">Clear Filters</button>
                  <span id="metadata-status" class="metadata-status" role="status" aria-live="polite" data-state="idle">Metadata: idle</span>
                </div>
              </section>
            </details>
            <p id="rec-summary" class="muted" role="status" aria-live="polite">Add at least one anime to start.</p>
            <ol id="rec-results" class="rec-results"></ol>

            <section class="seasonal">
              <div class="seasonal-head">
                <h3>${demoMode ? "Demo Suggestions" : "Seasonal Trending"}</h3>
                <button id="refresh-seasonal" type="button" class="ghost-btn">Refresh</button>
              </div>
              <p id="seasonal-status" class="muted" role="status" aria-live="polite" data-state="idle">Seasonal data: idle.</p>
              <ul id="seasonal-list" class="seasonal-list"></ul>
            </section>
          </section>
        </div>
      </section>

      <section id="view-network" class="view" hidden>
        <section id="tips-network" class="panel contextual-tip" hidden>
          <div class="contextual-tip-head">
            <h3>Network Tips</h3>
            <button id="tips-dismiss-network" class="ghost-btn" type="button">Dismiss Tips</button>
          </div>
          <ul class="contextual-tip-list">
            <li>Start with anime-only edges to reduce graph noise, then add users as needed.</li>
            <li>Raise minimum absolute edge weight to highlight stronger pair-preference signals.</li>
            <li>Click a node to inspect top weighted neighbors and compare context.</li>
            <li>Keyboard: <strong>Alt+/</strong> jumps to graph search from anywhere.</li>
          </ul>
        </section>

        <button id="network-mobile-toggle" type="button" class="network-mobile-toggle" hidden aria-expanded="false">
          Show Controls
        </button>
        <div id="network-layout" class="network-layout">
          <aside id="network-panel" class="panel">
            <h2>Network Explorer</h2>
            <p class="muted">Interact with the graph, inspect node connections, and filter visible edges.</p>

            <details class="accordion network-controls" open>
              <summary>Visibility Controls</summary>
              <div class="stats" id="stats"></div>
              <p id="network-render-status" class="network-render-status" role="status" aria-live="polite">
                Render status: ready.
              </p>

              <label class="control" for="min-weight">
                <span>Min absolute edge weight</span>
                <input id="min-weight" type="range" min="0" max="4" value="0" step="0.05" aria-label="Minimum absolute edge weight" />
                <output id="min-weight-value">0.00</output>
              </label>
              <p id="network-edge-legend" class="muted">Anime edges show v1 pair preference: teal is positive, coral is negative, and dashed grey is zero. A negative pair mean does not prove opposite tastes.</p>

              <label class="checkbox">
                <input id="toggle-anime-edges" type="checkbox" checked />
                <span>Show anime-to-anime edges</span>
              </label>

              <label class="checkbox">
                <input id="toggle-users" type="checkbox" />
                <span>Show user nodes + user-anime edges</span>
              </label>
            </details>

            <section class="network-search">
              <h3>Search In Graph</h3>
              <form id="network-search-form" class="network-search-form">
                <input id="network-search-input" type="text" list="network-node-options" placeholder="Title, anime:ID, or user:ID" aria-label="Search for anime or user node" />
                <button type="submit" class="primary-btn">
                  <span class="icon icon-search" aria-hidden="true"></span>
                  <span>Find</span>
                </button>
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

          <section id="graph-shell" class="graph-shell">
            <div id="graph-loading" class="graph-loading" hidden aria-hidden="true">
              <div class="graph-spinner"></div>
              <p id="graph-loading-message">Rendering network...</p>
            </div>
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
const themeToggleLabelEl = mustElement<HTMLSpanElement>("#theme-toggle-label");
const contrastToggleBtn = mustElement<HTMLButtonElement>("#contrast-toggle");
const contrastToggleLabelEl = mustElement<HTMLSpanElement>("#contrast-toggle-label");
const tipsToggleBtn = mustElement<HTMLButtonElement>("#tips-toggle");
const tipsToggleLabelEl = mustElement<HTMLSpanElement>("#tips-toggle-label");
const commandsToggleBtn = mustElement<HTMLButtonElement>("#commands-toggle");
const commandPaletteEl = mustElement<HTMLDivElement>("#command-palette");
const commandInput = mustElement<HTMLInputElement>("#command-input");
const commandListEl = mustElement<HTMLUListElement>("#command-list");
const commandCloseBtn = mustElement<HTMLButtonElement>("#command-close");
const viewRecommendations = mustElement<HTMLElement>("#view-recommendations");
const viewNetwork = mustElement<HTMLElement>("#view-network");
const tipsRecommendationsEl = mustElement<HTMLElement>("#tips-recommendations");
const tipsNetworkEl = mustElement<HTMLElement>("#tips-network");
const tipsDismissRecommendationsBtn = mustElement<HTMLButtonElement>(
  "#tips-dismiss-recommendations",
);
const tipsDismissNetworkBtn = mustElement<HTMLButtonElement>("#tips-dismiss-network");
const quickstartSeasonalBtn = mustElement<HTMLButtonElement>("#quickstart-seasonal");
const quickstartNetworkBtn = mustElement<HTMLButtonElement>("#quickstart-network");

const addAnimeForm = mustElement<HTMLFormElement>("#add-anime-form");
const animeInput = mustElement<HTMLInputElement>("#anime-input");
const addPreferenceSelect = mustElement<HTMLSelectElement>("#add-preference");
const animeOptions = mustElement<HTMLDataListElement>("#anime-options");
const recMethodSelect = mustElement<HTMLSelectElement>("#rec-method");
const recBlendControl = mustElement<HTMLLabelElement>("#rec-blend-control");
const recBlendInput = mustElement<HTMLInputElement>("#rec-blend");
const recBlendValueEl = mustElement<HTMLOutputElement>("#rec-blend-value");
const recEngineStatusEl = mustElement<HTMLParagraphElement>("#rec-engine-status");
const recMessageEl = mustElement<HTMLParagraphElement>("#rec-message");
const storageStatusEl = mustElement<HTMLParagraphElement>("#storage-status");
const preferenceMigrationNoticeEl = mustElement<HTMLParagraphElement>("#preference-migration-notice");
const selectedAnimeEl = mustElement<HTMLDivElement>("#selected-anime");
const watchedCountEl = mustElement<HTMLSpanElement>("#watched-count");
const clearWatchedBtn = mustElement<HTMLButtonElement>("#clear-watched");
const bulkImportForm = mustElement<HTMLFormElement>("#bulk-import-form");
const bulkImportFile = mustElement<HTMLInputElement>("#bulk-import-file");
const bulkImportInput = mustElement<HTMLTextAreaElement>("#bulk-import-input");
const bulkImportStatusEl = mustElement<HTMLParagraphElement>("#bulk-import-status");
const usernameImportForm = mustElement<HTMLFormElement>("#username-import-form");
const usernameImportProvider = mustElement<HTMLSelectElement>("#username-import-provider");
const usernameImportInput = mustElement<HTMLInputElement>("#username-import-input");
const usernameImportSubmit = mustElement<HTMLButtonElement>("#username-import-submit");
const usernameImportSubmitLabel = mustElement<HTMLSpanElement>("#username-import-submit-label");
const usernameImportStatusEl = mustElement<HTMLParagraphElement>("#username-import-status");
const historyImportPreviewEl = mustElement<HTMLElement>("#history-import-preview");
const historyImportSummaryEl = mustElement<HTMLParagraphElement>("#history-import-summary");
const historyImportUnmappedEl = mustElement<HTMLParagraphElement>("#history-import-unmapped");
const historyImportModeEl = mustElement<HTMLSelectElement>("#history-import-mode");
const historyImportApplyBtn = mustElement<HTMLButtonElement>("#history-import-apply");
const historyCountEl = mustElement<HTMLSpanElement>("#history-count");
const historyListEl = mustElement<HTMLUListElement>("#history-list");
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
const networkMobileToggleBtn = mustElement<HTMLButtonElement>("#network-mobile-toggle");
const networkLayoutEl = mustElement<HTMLDivElement>("#network-layout");
const networkPanelEl = mustElement<HTMLElement>("#network-panel");
const networkRenderStatusEl = mustElement<HTMLParagraphElement>("#network-render-status");
const graphShell = mustElement<HTMLElement>("#graph-shell");
const graphLoadingEl = mustElement<HTMLDivElement>("#graph-loading");
const graphLoadingMessageEl = mustElement<HTMLParagraphElement>("#graph-loading-message");
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

let selectedNodeId: string | null = null;
let currentGraph: Graph | null = null;
let explorerGraphData: LoadedGraphData | null = null;
let explorerGraphDataPromise: Promise<LoadedGraphData> | null = null;
let activeView: AppView = "recommendations";
let recommendationMode: RecommendationMode = "graph";
let modelBlendWeight = 0.5;
let recommendationRunId = 0;
let recommendationController: AbortController | null = null;
let activeUsernameImport: { controller: AbortController; provider: UsernameImportProvider } | null = null;
let bulkFileLoadId = 0;
let activeBulkFileLoadId: number | null = null;
let pendingHistoryImport: { parsed: ParsedHistory; origin: "local" | "username" } | null = null;
let graphRenderRunId = 0;
let modelRecommendationIndexPromise: Promise<ModelRecommendationIndex | null> | null = null;
let modelLoadError: string | null = null;
const recommendationFilters: RecommendationFilters = {
  genre: "",
  minYear: null,
  maxYear: null,
  minScore: null,
};
const animeMetadataCache = new Map<number, AnimeMetadata>();
const animeMetadataUnavailable = new Set<number>();
const animeMetadataFailed = new Set<number>();
const demoCatalog = demoMode ? await loadRequiredArtifact(artifactLoader.fetchDemoCatalog()) : null;
if (demoCatalog) {
  for (const item of demoCatalog) animeMetadataCache.set(item.animeId, item);
  usernameImportProvider.disabled = true;
  usernameImportInput.disabled = true;
  usernameImportSubmit.disabled = true;
  setAsyncStatus(usernameImportStatusEl, "demo", "Username imports are unavailable in the offline demo.");
}
let seasonalItems: SeasonalAnimeItem[] = [];
let seasonalLoadingPromise: Promise<void> | null = null;
let seasonalController: AbortController | null = null;
let activeTheme: ThemeMode = persistence.loadThemeModePreference(
  () => window.matchMedia("(prefers-color-scheme: light)").matches,
);
let activeContrast: ContrastMode = persistence.loadContrastModePreference();
let helpTipsDismissed = persistence.loadHelpTipsDismissed();
let commandPaletteOpen = false;
let commandSelectionIndex = 0;
let commandFilteredActions: CommandAction[] = [];
let commandMatchReasons = new Map<string, CommandMatchReason[]>();
let commandPinnedIds = persistence.loadCommandPinnedIds();
let commandHistoryIds = persistence.loadCommandHistoryIds();
let draggedPinnedCommandId: string | null = null;
const reduceMotionMediaQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
const networkCompactMediaQuery = window.matchMedia(
  `(max-width: ${NETWORK_MOBILE_COMPACT_MAX_WIDTH}px)`,
);
let networkControlsHiddenOnMobile = true;

applyTheme(activeTheme);
applyContrast(activeContrast);

const graphData = await loadRequiredArtifact(artifactLoader.fetchGraph());
const graphNodes = getGraphNodes(graphData);
const recommendationIndex = isCompactGraphData(graphData)
  ? buildRecommendationIndexFromCompact(graphData)
  : buildRecommendationIndex(graphData);
const selectedAnimeNodeIds: string[] = [];
const selectedAnimePreferences = new Map<string, AnimePreference>();
const historyEntries: HistoryEntry[] = [];
const includeCandidateNodeIds: string[] = [];
const excludeCandidateNodeIds: string[] = [];
const persistedState = persistence.loadRecommendationState();
const savedProfiles = persistence.loadRecommendationProfiles();
const commandActions = buildCommandActions();

for (const entry of persistedState.preferences) {
  selectedAnimeNodeIds.push(entry.nodeId);
  selectedAnimePreferences.set(entry.nodeId, entry);
}
historyEntries.push(...persistedState.history);
preferenceMigrationNoticeEl.textContent = persistence.getMigrationNotice() ??
  (persistedState.preferences.some((item) => item.source === "legacy")
    ? "Earlier watch weights were migrated conservatively. Unrated watches stay Seen; low scores are Disliked; clear likes are Liked. Review preferences and importance below."
    : "");
preferenceMigrationNoticeEl.hidden = !preferenceMigrationNoticeEl.textContent;
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
populateNetworkNodeOptions(graphNodes, networkNodeOptions);
renderSelectedAnime();
renderImportedHistory();
renderIncludeCandidates();
renderExcludeCandidates();
renderProfileOptions(savedProfiles);
renderStorageWarnings();
renderFilterControls();
renderSeasonalList();
renderContextualTips();
syncNetworkCompactMode();
renderCommandPaletteList();

const defaultMinWeight = getDefaultMinAnimeAnimeWeight(graphData, minWeightInput);
minWeightInput.value = defaultMinWeight.toFixed(2);
minWeightValue.textContent = defaultMinWeight.toFixed(2);

setActiveView(viewFromHash(), true);
void updateRecommendations();
void loadSeasonalTrending(false);

window.addEventListener("hashchange", () => {
  setActiveView(viewFromHash(), true);
});

window.addEventListener("keydown", (event) => {
  handleGlobalShortcut(event);
});

commandsToggleBtn.addEventListener("click", () => {
  toggleCommandPalette();
});

commandCloseBtn.addEventListener("click", () => {
  closeCommandPalette();
});

commandPaletteEl.addEventListener("click", (event) => {
  const target = event.target;
  if (!(target instanceof HTMLElement)) {
    return;
  }
  if (target.dataset.commandClose === "true") {
    closeCommandPalette();
  }
});

commandInput.addEventListener("input", () => {
  commandSelectionIndex = 0;
  renderCommandPaletteList();
});

commandInput.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    event.preventDefault();
    closeCommandPalette();
    return;
  }
  if (
    !event.altKey &&
    !event.ctrlKey &&
    !event.metaKey &&
    !event.shiftKey &&
    /^[1-9]$/.test(event.key)
  ) {
    const index = Number.parseInt(event.key, 10) - 1;
    const command = commandFilteredActions[index];
    if (command) {
      event.preventDefault();
      executeCommand(command);
    }
    return;
  }
  if (event.key === "ArrowDown") {
    event.preventDefault();
    moveCommandSelection(1);
    return;
  }
  if (event.key === "ArrowUp") {
    event.preventDefault();
    moveCommandSelection(-1);
    return;
  }
  if (event.key === "Enter") {
    event.preventDefault();
    const selected = commandFilteredActions[commandSelectionIndex];
    if (selected) {
      executeCommand(selected);
    }
  }
});

commandListEl.addEventListener("click", (event) => {
  const target = event.target;
  if (!(target instanceof HTMLElement)) {
    return;
  }
  const dragHandle = target.closest<HTMLButtonElement>("button[data-command-drag-id]");
  if (dragHandle) {
    return;
  }
  const pinButton = target.closest<HTMLButtonElement>("button[data-command-pin-id]");
  if (pinButton) {
    const commandId = pinButton.dataset.commandPinId;
    if (!commandId) {
      return;
    }
    toggleCommandPinned(commandId);
    renderCommandPaletteList();
    return;
  }
  const button = target.closest<HTMLButtonElement>("button[data-command-id]");
  if (!button) {
    return;
  }
  const commandId = button.dataset.commandId;
  if (!commandId) {
    return;
  }
  const command = commandActions.find((item) => item.id === commandId);
  if (!command) {
    return;
  }
  executeCommand(command);
});

commandListEl.addEventListener("dragstart", (event) => {
  if (!(event instanceof DragEvent)) {
    return;
  }
  const target = event.target;
  if (!(target instanceof HTMLElement)) {
    return;
  }
  const dragHandle = target.closest<HTMLButtonElement>("button[data-command-drag-id]");
  const commandId = dragHandle?.dataset.commandDragId;
  if (!commandId || !isCommandPinned(commandId)) {
    return;
  }
  draggedPinnedCommandId = commandId;
  dragHandle.classList.add("active");
  if (event.dataTransfer) {
    event.dataTransfer.effectAllowed = "move";
    event.dataTransfer.setData("text/plain", commandId);
  }
});

commandListEl.addEventListener("dragover", (event) => {
  if (!(event instanceof DragEvent) || !draggedPinnedCommandId) {
    return;
  }
  const target = event.target;
  if (!(target instanceof HTMLElement)) {
    return;
  }
  const row = target.closest<HTMLElement>(".command-row[data-command-row-id]");
  if (!row) {
    return;
  }
  const targetId = row.dataset.commandRowId;
  if (!targetId || !isCommandPinned(targetId)) {
    return;
  }
  event.preventDefault();
  clearCommandDragOverState();
  row.classList.add("drag-over");
});

commandListEl.addEventListener("drop", (event) => {
  if (!(event instanceof DragEvent) || !draggedPinnedCommandId) {
    return;
  }
  const target = event.target;
  if (!(target instanceof HTMLElement)) {
    return;
  }
  const row = target.closest<HTMLElement>(".command-row[data-command-row-id]");
  if (!row) {
    clearCommandDragState();
    return;
  }
  const targetId = row.dataset.commandRowId;
  if (!targetId || !isCommandPinned(targetId)) {
    clearCommandDragState();
    return;
  }
  event.preventDefault();
  movePinnedCommandBefore(draggedPinnedCommandId, targetId);
  clearCommandDragState();
  renderCommandPaletteList();
});

commandListEl.addEventListener("dragend", () => {
  clearCommandDragState();
});

navRecommendationsBtn.addEventListener("click", () => {
  setActiveView("recommendations", false);
});

navNetworkBtn.addEventListener("click", () => {
  setActiveView("network", false);
});

quickstartNetworkBtn.addEventListener("click", () => {
  setActiveView("network", false);
});

quickstartSeasonalBtn.addEventListener("click", () => {
  showSeasonalIdeas();
});

themeToggleBtn.addEventListener("click", () => {
  activeTheme = activeTheme === "dark" ? "light" : "dark";
  applyTheme(activeTheme);
  persistence.persistThemeModePreference(activeTheme);
});

contrastToggleBtn.addEventListener("click", () => {
  activeContrast = activeContrast === "normal" ? "high" : "normal";
  applyContrast(activeContrast);
  persistence.persistContrastModePreference(activeContrast);
});

networkMobileToggleBtn.addEventListener("click", () => {
  networkControlsHiddenOnMobile = !networkControlsHiddenOnMobile;
  applyNetworkCompactControlsState();
  if (!networkControlsHiddenOnMobile) {
    networkPanelEl.scrollIntoView({
      block: "start",
      behavior: prefersReducedMotion() ? "auto" : "smooth",
    });
  }
});

onMediaQueryChange(networkCompactMediaQuery, () => {
  syncNetworkCompactMode();
});

tipsToggleBtn.addEventListener("click", () => {
  helpTipsDismissed = !helpTipsDismissed;
  persistence.persistHelpTipsDismissed(helpTipsDismissed);
  renderContextualTips();
});

tipsDismissRecommendationsBtn.addEventListener("click", () => {
  helpTipsDismissed = true;
  persistence.persistHelpTipsDismissed(helpTipsDismissed);
  renderContextualTips();
});

tipsDismissNetworkBtn.addEventListener("click", () => {
  helpTipsDismissed = true;
  persistence.persistHelpTipsDismissed(helpTipsDismissed);
  renderContextualTips();
});

addAnimeForm.addEventListener("submit", (event) => {
  event.preventDefault();
  addAnimeFromInput();
});

bulkImportForm.addEventListener("submit", (event) => {
  event.preventDefault();
  importWatchedFromBulkInput();
});

bulkImportFile.addEventListener("change", () => {
  void loadBulkImportFile();
});

bulkImportInput.addEventListener("input", () => {
  cancelBulkFileLoad();
  cancelActiveUsernameImport();
  clearPendingHistoryImport();
});

usernameImportForm.addEventListener("submit", (event) => {
  event.preventDefault();
  void importWatchedFromUsername();
});

historyImportModeEl.addEventListener("change", renderImportPreview);
historyImportApplyBtn.addEventListener("click", applyPendingHistoryImport);
usernameImportProvider.addEventListener("change", () => {
  cancelActiveUsernameImport();
  clearPendingHistoryImport();
});
usernameImportInput.addEventListener("input", () => {
  cancelActiveUsernameImport();
  clearPendingHistoryImport();
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
  if (target.dataset.importanceNodeId === undefined) {
    return;
  }

  const nodeId = target.dataset.importanceNodeId;
  if (!nodeId) {
    return;
  }

  const parsed = Number.parseFloat(target.value);
  const clamped = clampImportance(parsed);
  const current = selectedAnimePreferences.get(nodeId);
  if (!current) return;
  selectedAnimePreferences.set(nodeId, { ...current, importance: clamped });
  persistRecommendationState();
  target.value = clamped.toFixed(1);

  const chip = target.closest(".chip-weighted");
  const valueEl = chip?.querySelector<HTMLOutputElement>(".chip-importance-value");
  if (valueEl) {
    valueEl.textContent = `${clamped.toFixed(1)}x`;
  }

  void updateRecommendations();
});

selectedAnimeEl.addEventListener("change", (event) => {
  const target = event.target;
  if (!(target instanceof HTMLSelectElement)) return;
  const nodeId = target.dataset.preferenceNodeId;
  if (!nodeId || !selectedAnimePreferences.has(nodeId)) return;
  const sentiment = target.value as PreferenceSentiment;
  if (!["seen", "liked", "disliked"].includes(sentiment)) return;
  const prior = selectedAnimePreferences.get(nodeId)!;
  selectedAnimePreferences.set(nodeId, manualPreference(nodeId, sentiment, prior.importance));
  persistRecommendationState();
  renderSelectedAnime();
  void updateRecommendations();
});

clearWatchedBtn.addEventListener("click", () => {
  selectedAnimeNodeIds.splice(0, selectedAnimeNodeIds.length);
  selectedAnimePreferences.clear();
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
  recommendationMode = persistence.parseRecommendationMode(recMethodSelect.value);
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
  if (currentGraph) {
    renderSvgGraph(currentGraph);
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
  renderContextualTips();
  applyNetworkCompactControlsState();

  navRecommendationsBtn.classList.toggle("active", view === "recommendations");
  navNetworkBtn.classList.toggle("active", view === "network");

  if (!fromHash) {
    const nextHash = view === "network" ? "network" : "recommendations";
    if (window.location.hash !== `#${nextHash}`) {
      window.location.hash = nextHash;
    }
  }

  if (view === "network") {
    setGraphLoadingState(true, "Render status: loading explorer data...");
    void ensureExplorerGraphData()
      .then(() => {
        runtime.frame(() => {
          rerenderGraph();
        });
      })
      .catch((error) => {
        const errorMessage = error instanceof Error ? error.message : "unknown error";
        setGraphLoadingState(
          false,
          `Render status: failed to load explorer data (${errorMessage}).`,
        );
        console.error("Explorer graph load failed.", error);
      });
  } else {
    graphRenderRunId += 1;
    setGraphLoadingState(false, "Render status: ready.");
  }
}

function syncNetworkCompactMode(): void {
  if (!networkCompactMediaQuery.matches) {
    networkControlsHiddenOnMobile = false;
  } else if (!networkLayoutEl.classList.contains("mobile-compact")) {
    networkControlsHiddenOnMobile = true;
  }
  applyNetworkCompactControlsState();
}

function applyNetworkCompactControlsState(): void {
  const compact = networkCompactMediaQuery.matches;
  networkLayoutEl.classList.toggle("mobile-compact", compact);
  networkMobileToggleBtn.hidden = !compact || activeView !== "network";

  const hideControls = compact && networkControlsHiddenOnMobile;
  networkLayoutEl.classList.toggle("controls-hidden", hideControls);
  networkMobileToggleBtn.setAttribute("aria-expanded", hideControls ? "false" : "true");
  networkMobileToggleBtn.textContent = hideControls ? "Show Controls" : "Hide Controls";
}

function viewFromHash(): AppView {
  return window.location.hash.toLowerCase() === "#network"
    ? "network"
    : "recommendations";
}

function handleGlobalShortcut(event: KeyboardEvent): void {
  const key = event.key.toLowerCase();
  const commandModifier = event.ctrlKey || event.metaKey;

  if (commandModifier && !event.altKey && key === "k") {
    event.preventDefault();
    toggleCommandPalette();
    return;
  }

  if (commandPaletteOpen && key === "escape") {
    event.preventDefault();
    closeCommandPalette();
    return;
  }

  if (commandPaletteOpen) {
    return;
  }
  if (isTypingTarget(event.target)) {
    return;
  }
  if (!event.altKey || event.metaKey || event.ctrlKey) {
    return;
  }

  switch (key) {
    case "1":
      event.preventDefault();
      runCommandById("open-recommendations");
      break;
    case "2":
      event.preventDefault();
      runCommandById("open-network");
      break;
    case "t":
      event.preventDefault();
      runCommandById("toggle-theme");
      break;
    case "c":
      event.preventDefault();
      runCommandById("toggle-contrast");
      break;
    case "h":
      event.preventDefault();
      runCommandById("toggle-tips");
      break;
    case "/":
    case "?":
      event.preventDefault();
      runCommandById("focus-input");
      break;
    default:
      break;
  }
}

function isTypingTarget(target: EventTarget | null): boolean {
  if (!(target instanceof Element)) {
    return false;
  }
  if (target.closest("textarea, input, select, [contenteditable='true']")) {
    return true;
  }
  return target instanceof HTMLElement && target.isContentEditable;
}

function focusPrimaryInputForActiveView(): void {
  if (activeView === "network") {
    if (networkCompactMediaQuery.matches && networkControlsHiddenOnMobile) {
      networkControlsHiddenOnMobile = false;
      applyNetworkCompactControlsState();
    }
    networkSearchInput.focus();
    networkSearchInput.select();
    return;
  }

  animeInput.focus();
  animeInput.select();
}

function runCommandById(id: string): void {
  const command = commandActions.find((item) => item.id === id);
  if (!command) {
    return;
  }
  recordCommandHistory(id);
  command.run();
}

function buildCommandActions(): CommandAction[] {
  return [
    {
      id: "open-recommendations",
      label: "Open Recommendations",
      group: "Navigation",
      shortcutLabel: "Alt+1",
      keywords: ["recommend", "home", "view"],
      run: () => setActiveView("recommendations", false),
    },
    {
      id: "open-network",
      label: "Open Network Explorer",
      group: "Navigation",
      shortcutLabel: "Alt+2",
      keywords: ["network", "graph", "view"],
      run: () => setActiveView("network", false),
    },
    {
      id: "focus-input",
      label: "Focus Main Input",
      group: "Navigation",
      shortcutLabel: "Alt+/",
      keywords: ["focus", "search", "anime", "input"],
      run: () => focusPrimaryInputForActiveView(),
    },
    {
      id: "toggle-theme",
      label: "Toggle Theme",
      group: "Display",
      shortcutLabel: "Alt+T",
      keywords: ["theme", "dark", "light"],
      run: () => themeToggleBtn.click(),
    },
    {
      id: "toggle-contrast",
      label: "Toggle Contrast",
      group: "Display",
      shortcutLabel: "Alt+C",
      keywords: ["contrast", "accessibility", "readability"],
      run: () => contrastToggleBtn.click(),
    },
    {
      id: "toggle-tips",
      label: "Toggle Help Tips",
      group: "Display",
      shortcutLabel: "Alt+H",
      keywords: ["tips", "help", "onboarding"],
      run: () => tipsToggleBtn.click(),
    },
    {
      id: "seasonal-starter",
      label: "Browse Seasonal Ideas",
      group: "Utilities",
      keywords: ["seasonal", "starter", "quickstart"],
      run: () => showSeasonalIdeas(),
    },
    {
      id: "toggle-network-controls",
      label: "Toggle Mobile Network Controls",
      group: "Utilities",
      keywords: ["mobile", "network", "controls", "panel"],
      run: () => {
        if (activeView !== "network") {
          setActiveView("network", false);
        }
        if (networkCompactMediaQuery.matches) {
          networkControlsHiddenOnMobile = !networkControlsHiddenOnMobile;
          applyNetworkCompactControlsState();
        } else {
          focusPrimaryInputForActiveView();
        }
      },
    },
  ];
}

function toggleCommandPalette(): void {
  if (commandPaletteOpen) {
    closeCommandPalette();
    return;
  }
  openCommandPalette();
}

function openCommandPalette(): void {
  commandPaletteOpen = true;
  commandPaletteEl.hidden = false;
  commandPaletteEl.setAttribute("aria-hidden", "false");
  commandInput.value = "";
  commandSelectionIndex = 0;
  renderCommandPaletteList();
  document.body.classList.add("palette-open");
  runtime.frame(() => {
    commandInput.focus();
  });
}

function closeCommandPalette(): void {
  commandPaletteOpen = false;
  commandPaletteEl.hidden = true;
  commandPaletteEl.setAttribute("aria-hidden", "true");
  document.body.classList.remove("palette-open");
  clearCommandDragState();
  commandsToggleBtn.focus();
}

function moveCommandSelection(delta: number): void {
  if (commandFilteredActions.length === 0) {
    return;
  }
  commandSelectionIndex =
    (commandSelectionIndex + delta + commandFilteredActions.length) %
    commandFilteredActions.length;
  renderCommandPaletteList();
}

function executeCommand(command: CommandAction): void {
  recordCommandHistory(command.id);
  closeCommandPalette();
  command.run();
}

function renderCommandPaletteList(): void {
  const query = normalizeTitle(commandInput.value);
  const filtered = filterCommandActions(query);
  const sections = buildCommandSections(query, filtered);
  commandFilteredActions = sections.flatMap((section) => section.actions);
  if (commandFilteredActions.length === 0) {
    commandSelectionIndex = 0;
    commandListEl.innerHTML = `<li class="command-empty">No command matches "${escapeHtml(query)}".</li>`;
    return;
  }

  if (commandSelectionIndex >= commandFilteredActions.length) {
    commandSelectionIndex = 0;
  }

  let flatIndex = 0;
  commandListEl.innerHTML = sections
    .map((section) => {
      const items = section.actions
        .map((command) => {
          const selected = flatIndex === commandSelectionIndex;
          const quickIndex = flatIndex + 1;
          const quickIndexLabel = quickIndex <= 9 ? String(quickIndex) : "";
          const highlightedLabel = highlightCommandLabel(command.label, query);
          const reasonBadges = renderCommandReasonBadges(command.id, section.group, query);
          const pinned = isCommandPinned(command.id);
          const draggablePinned = section.group === "Pinned" && pinned;
          flatIndex += 1;
          return `
            <li class="command-row" data-command-row-id="${escapeHtml(command.id)}">
              ${
                draggablePinned
                  ? `<button
                      type="button"
                      class="command-drag-handle"
                      data-command-drag-id="${escapeHtml(command.id)}"
                      draggable="true"
                      aria-label="Drag to reorder pinned command"
                      title="Drag to reorder"
                    >⋮⋮</button>`
                  : ""
              }
              <button type="button" class="command-item${selected ? " selected" : ""}" data-command-id="${escapeHtml(command.id)}">
                <span class="command-item-prefix">
                  ${
                    quickIndexLabel
                      ? `<span class="command-item-index">${escapeHtml(quickIndexLabel)}</span>`
                      : ""
                  }
                  <span class="command-item-label">${highlightedLabel}</span>
                </span>
                <span class="command-item-meta">
                  ${reasonBadges}
                  ${
                    command.shortcutLabel
                      ? `<span class="command-item-shortcut">${escapeHtml(command.shortcutLabel)}</span>`
                      : ""
                  }
                </span>
              </button>
              <button
                type="button"
                class="command-pin-btn${pinned ? " pinned" : ""}"
                data-command-pin-id="${escapeHtml(command.id)}"
                aria-pressed="${pinned ? "true" : "false"}"
                aria-label="${pinned ? "Unpin command" : "Pin command"}"
                title="${pinned ? "Unpin command" : "Pin command"}"
              >
                ★
              </button>
            </li>
          `;
        })
        .join("");
      const sectionClass =
        section.group === "Pinned"
          ? "command-group pinned"
          : section.group === "Recent"
            ? "command-group recent"
            : "command-group";
      return `<li class="${sectionClass}">${escapeHtml(section.group)}</li>${items}`;
    })
    .join("");
}

function filterCommandActions(query: string): CommandAction[] {
  if (!query) {
    commandMatchReasons = new Map();
    return [...commandActions];
  }
  const queryTokens = query.split(/\s+/).filter((token) => token.length > 0);
  const matchReasons = new Map<string, Set<CommandMatchReason>>();
  const scoredMatches = commandActions
    .map((command) => {
      const label = normalizeTitle(command.label);
      const haystack = normalizeTitle([command.label, ...command.keywords].join(" "));
      let score = 0;
      const reasons = new Set<CommandMatchReason>();
      for (const token of queryTokens) {
        const tokenMatch = scoreCommandTokenMatch(token, label, haystack);
        if (!tokenMatch) {
          return null;
        }
        score += tokenMatch.score;
        reasons.add(tokenMatch.reason);
      }
      if (isCommandPinned(command.id)) {
        score += 18;
        reasons.add("pinned");
      }
      const historyIndex = commandHistoryIds.indexOf(command.id);
      if (historyIndex >= 0) {
        score += Math.max(0, 12 - historyIndex * 2);
        reasons.add("recent");
      }
      matchReasons.set(command.id, reasons);
      return { command, score, reasons };
    })
    .filter(
      (
        entry,
      ): entry is { command: CommandAction; score: number; reasons: Set<CommandMatchReason> } =>
        entry !== null,
    );

  scoredMatches.sort((left, right) => {
    if (right.score !== left.score) {
      return right.score - left.score;
    }
    return left.command.label.localeCompare(right.command.label);
  });

  commandMatchReasons = new Map(
    scoredMatches.map((entry) => [
      entry.command.id,
      normalizeCommandReasons(entry.reasons),
    ]),
  );
  return scoredMatches.map((entry) => entry.command);
}

function scoreCommandTokenMatch(
  token: string,
  label: string,
  haystack: string,
): CommandTokenMatch | null {
  if (label === token) {
    return { score: 180, reason: "exact" };
  }
  if (label.startsWith(token)) {
    return { score: 140, reason: "prefix" };
  }

  const haystackWords = haystack.split(/\s+/);
  if (haystackWords.some((word) => word.startsWith(token))) {
    return { score: 110, reason: "word" };
  }
  if (haystack.includes(token)) {
    return { score: 80, reason: "contains" };
  }

  const fuzzyScore = scoreSubsequenceMatch(token, haystack);
  if (fuzzyScore > 0) {
    return { score: fuzzyScore, reason: "fuzzy" };
  }
  return null;
}

function scoreSubsequenceMatch(token: string, haystack: string): number {
  let tokenIndex = 0;
  const matchedPositions: number[] = [];
  for (let haystackIndex = 0; haystackIndex < haystack.length; haystackIndex += 1) {
    if (haystack[haystackIndex] === token[tokenIndex]) {
      matchedPositions.push(haystackIndex);
      tokenIndex += 1;
      if (tokenIndex === token.length) {
        break;
      }
    }
  }
  if (tokenIndex !== token.length || matchedPositions.length === 0) {
    return 0;
  }

  const span = matchedPositions[matchedPositions.length - 1] - matchedPositions[0] + 1;
  const gapPenalty = Math.max(0, span - token.length);
  const positionBonus = matchedPositions[0] === 0 ? 6 : matchedPositions[0] <= 2 ? 3 : 0;
  return Math.max(12, 54 - Math.min(gapPenalty, 34) + positionBonus);
}

function normalizeCommandReasons(
  reasons: Set<CommandMatchReason>,
): CommandMatchReason[] {
  const order: CommandMatchReason[] = [
    "pinned",
    "exact",
    "prefix",
    "word",
    "contains",
    "fuzzy",
    "recent",
  ];
  return order.filter((reason) => reasons.has(reason));
}

function renderCommandReasonBadges(
  commandId: string,
  sectionGroup: string,
  query: string,
): string {
  const reasons = new Set(commandMatchReasons.get(commandId) ?? []);
  if (!query && sectionGroup === "Pinned") {
    reasons.add("pinned");
  }
  if (!query && sectionGroup === "Recent") {
    reasons.add("recent");
  }
  if (reasons.size === 0) {
    return "";
  }
  return normalizeCommandReasons(reasons)
    .slice(0, 3)
    .map(
      (reason) =>
        `<span class="command-reason command-reason-${reason}">${escapeHtml(
          commandReasonLabel(reason),
        )}</span>`,
    )
    .join("");
}

function commandReasonLabel(reason: CommandMatchReason): string {
  if (reason === "word") {
    return "keyword";
  }
  return reason;
}

function highlightCommandLabel(label: string, normalizedQuery: string): string {
  if (!normalizedQuery) {
    return escapeHtml(label);
  }

  const tokens = normalizedQuery.split(/\s+/).filter((token) => token.length > 0);
  if (tokens.length === 0) {
    return escapeHtml(label);
  }

  const lowerLabel = label.toLowerCase();
  const scores = new Array<number>(label.length).fill(0);

  for (const token of tokens) {
    const contiguousStart = lowerLabel.indexOf(token);
    if (contiguousStart >= 0) {
      for (
        let index = contiguousStart;
        index < contiguousStart + token.length && index < scores.length;
        index += 1
      ) {
        scores[index] += 2;
      }
      continue;
    }

    const positions = findSubsequencePositions(token, lowerLabel);
    if (positions.length === token.length) {
      for (const position of positions) {
        if (position >= 0 && position < scores.length) {
          scores[position] += 1;
        }
      }
    }
  }

  let html = "";
  let start = 0;
  while (start < label.length) {
    const highlighted = scores[start] > 0;
    let end = start + 1;
    while (end < label.length && (scores[end] > 0) === highlighted) {
      end += 1;
    }
    const segment = escapeHtml(label.slice(start, end));
    html += highlighted ? `<mark class="command-match">${segment}</mark>` : segment;
    start = end;
  }
  return html;
}

function findSubsequencePositions(token: string, value: string): number[] {
  const positions: number[] = [];
  let searchFrom = 0;
  for (const char of token) {
    const position = value.indexOf(char, searchFrom);
    if (position < 0) {
      return [];
    }
    positions.push(position);
    searchFrom = position + 1;
  }
  return positions;
}

function buildCommandSections(
  query: string,
  filtered: CommandAction[],
): CommandSection[] {
  const sections: CommandSection[] = [];
  const pinnedActions = orderCommandsByIds(commandPinnedIds, filtered);
  if (pinnedActions.length > 0) {
    sections.push({
      group: "Pinned",
      actions: pinnedActions,
    });
  }

  const pinnedIds = new Set(pinnedActions.map((command) => command.id));
  let remaining = filtered.filter((command) => !pinnedIds.has(command.id));
  if (!query && commandHistoryIds.length > 0) {
    const recentActions = orderCommandsByIds(commandHistoryIds, remaining);
    if (recentActions.length > 0) {
      sections.push({
        group: "Recent",
        actions: recentActions,
      });
      const recentIds = new Set(recentActions.map((command) => command.id));
      remaining = remaining.filter((command) => !recentIds.has(command.id));
    }
  }

  sections.push(...groupCommandActions(remaining));
  return sections;
}

function orderCommandsByIds(
  ids: string[],
  commands: CommandAction[],
): CommandAction[] {
  const byId = new Map(commands.map((command) => [command.id, command]));
  const ordered: CommandAction[] = [];
  for (const id of ids) {
    const command = byId.get(id);
    if (!command) {
      continue;
    }
    ordered.push(command);
  }
  return ordered;
}

function groupCommandActions(
  actions: CommandAction[],
): CommandSection[] {
  const sections = new Map<CommandGroup, CommandAction[]>([
    ["Navigation", []],
    ["Display", []],
    ["Utilities", []],
  ]);
  for (const action of actions) {
    sections.get(action.group)?.push(action);
  }
  return [...sections.entries()]
    .filter((entry) => entry[1].length > 0)
    .map(([group, groupedActions]) => ({
      group,
      actions: groupedActions,
    }));
}

function isCommandPinned(commandId: string): boolean {
  return commandPinnedIds.includes(commandId);
}

function toggleCommandPinned(commandId: string): void {
  if (isCommandPinned(commandId)) {
    commandPinnedIds = commandPinnedIds.filter((id) => id !== commandId);
  } else {
    commandPinnedIds = [commandId, ...commandPinnedIds].slice(0, COMMAND_PINNED_LIMIT);
  }
  persistence.persistCommandPinnedIds(commandPinnedIds);
}

function movePinnedCommandBefore(draggedId: string, targetId: string): void {
  if (draggedId === targetId) {
    return;
  }
  if (!isCommandPinned(draggedId) || !isCommandPinned(targetId)) {
    return;
  }

  const orderedPinned = commandPinnedIds.filter((id) => isCommandPinned(id));
  const withoutDragged = orderedPinned.filter((id) => id !== draggedId);
  const targetIndex = withoutDragged.indexOf(targetId);
  if (targetIndex < 0) {
    return;
  }
  withoutDragged.splice(targetIndex, 0, draggedId);
  commandPinnedIds = withoutDragged.slice(0, COMMAND_PINNED_LIMIT);
  persistence.persistCommandPinnedIds(commandPinnedIds);
}

function clearCommandDragOverState(): void {
  commandListEl
    .querySelectorAll<HTMLElement>(".command-row.drag-over")
    .forEach((row) => row.classList.remove("drag-over"));
}

function clearCommandDragState(): void {
  draggedPinnedCommandId = null;
  clearCommandDragOverState();
  commandListEl
    .querySelectorAll<HTMLElement>(".command-drag-handle.active")
    .forEach((handle) => handle.classList.remove("active"));
}

function recordCommandHistory(commandId: string): void {
  commandHistoryIds = [commandId, ...commandHistoryIds.filter((id) => id !== commandId)].slice(
    0,
    COMMAND_HISTORY_LIMIT,
  );
  persistence.persistCommandHistoryIds(commandHistoryIds);
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
  addAnimeToWatchedList(anime, "Added", "seen");
}

function addAnimeToWatchedList(anime: AnimeInfo, prefix: string, sentiment: PreferenceSentiment): void {
  if (selectedAnimeNodeIds.includes(anime.nodeId)) {
    recMessageEl.textContent = `${anime.label} is already in your watched list.`;
    return;
  }

  selectedAnimeNodeIds.push(anime.nodeId);
  selectedAnimePreferences.set(anime.nodeId, manualPreference(anime.nodeId, sentiment));
  persistRecommendationState();
  recMessageEl.textContent = `${prefix}: ${anime.label} (${sentiment === "seen" ? "Seen / Unrated" : sentiment}).`;
  renderSelectedAnime();
  void updateRecommendations();
}

function showSeasonalIdeas(): void {
  if (seasonalItems.length === 0) {
    recMessageEl.textContent =
      "Seasonal ideas are not ready yet. Try again in a moment.";
    return;
  }
  seasonalListEl.scrollIntoView({ behavior: "smooth", block: "start" });
  recMessageEl.textContent = "Browse seasonal ideas below. Add a title only if you have seen it; choose its preference explicitly.";
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
  const sentiment = addPreferenceSelect.value as PreferenceSentiment;
  addPreferenceSelect.value = "seen";
  addAnimeToWatchedList(anime, "Added", ["seen", "liked", "disliked"].includes(sentiment) ? sentiment : "seen");
}

async function loadBulkImportFile(): Promise<void> {
  const file = bulkImportFile.files?.[0];
  if (!file) {
    return;
  }
  cancelActiveUsernameImport();
  const loadId = ++bulkFileLoadId;
  activeBulkFileLoadId = loadId;
  bulkImportFile.value = "";
  clearPendingHistoryImport();
  const name = file.name.toLowerCase();
  const isText = name.endsWith(".txt");
  const isXml = name.endsWith(".xml");
  if ((!isText && !isXml) || file.size > (isText ? MAX_TEXT_IMPORT_BYTES : MAX_MAL_XML_IMPORT_BYTES)) {
    activeBulkFileLoadId = null;
    setAsyncStatus(bulkImportStatusEl, "failed", "Choose a .txt file up to 128 KiB or .xml file up to 2 MiB.");
    recMessageEl.textContent = "Choose a .txt file up to 128 KiB or .xml file up to 2 MiB.";
    return;
  }
  setAsyncStatus(bulkImportStatusEl, "loading", "Reading local file...");
  try {
    const content = await file.text();
    if (activeBulkFileLoadId !== loadId) return;
    if (isXml) {
      const parsed = parseMalXmlHistory(content);
      bulkImportInput.value = "";
      if (parsed.entries.length > 0) {
        setPendingHistoryImport(parsed, "local");
        setAsyncStatus(bulkImportStatusEl, "ready", "Local XML parsed. Review the import preview before applying.");
        recMessageEl.textContent = `Loaded local XML file "${file.name}". Review the import preview.`;
      } else {
        setAsyncStatus(bulkImportStatusEl, "empty", "The local XML has no anime entries; current history is intact.");
      }
    } else {
      bulkImportInput.value = content;
      if (content.trim()) {
        setAsyncStatus(bulkImportStatusEl, "ready", "Local text loaded. Preview the entries before applying.");
        recMessageEl.textContent = `Loaded local file "${file.name}". Select Preview Text Import.`;
      } else {
        setAsyncStatus(bulkImportStatusEl, "empty", "The local file has no entries to import.");
        recMessageEl.textContent = `Local file "${file.name}" is empty.`;
      }
    }
    activeBulkFileLoadId = null;
  } catch (error) {
    if (activeBulkFileLoadId !== loadId) return;
    activeBulkFileLoadId = null;
    clearPendingHistoryImport();
    setAsyncStatus(bulkImportStatusEl, "failed", "Unable to parse that local file; current history is intact.");
    recMessageEl.textContent = error instanceof Error ? error.message : "Unable to read that local file.";
  }
}

function importWatchedFromBulkInput(): void {
  if (activeBulkFileLoadId !== null) {
    recMessageEl.textContent = "Wait for the local file to finish loading before importing.";
    return;
  }
  cancelActiveUsernameImport();
  const raw = bulkImportInput.value.trim();
  if (!raw) {
    recMessageEl.textContent = "Paste at least one line to import.";
    return;
  }

  try {
    const parsed = parseTextHistory(raw);
    if (parsed.entries.length === 0) {
      setAsyncStatus(bulkImportStatusEl, "empty", "No entries found; current history is intact.");
      return;
    }
    setPendingHistoryImport(parsed, "local");
    setAsyncStatus(bulkImportStatusEl, "ready", "Text parsed. Review the import preview before applying.");
    recMessageEl.textContent = "Review the import counts and choose merge or replace before applying.";
  } catch (error) {
    clearPendingHistoryImport();
    setAsyncStatus(bulkImportStatusEl, "failed", "Text import is invalid; current history is intact.");
    recMessageEl.textContent = error instanceof Error ? error.message : "Unable to parse text import.";
  }
}

async function importWatchedFromUsername(): Promise<void> {
  if (demoMode) {
    setAsyncStatus(usernameImportStatusEl, "demo", "Username imports are unavailable in the offline demo.");
    recMessageEl.textContent = "Username imports are unavailable in the offline demo.";
    return;
  }
  const provider = parseUsernameImportProvider(usernameImportProvider.value);
  const username = usernameImportInput.value.trim();
  if (!username) {
    recMessageEl.textContent = "Enter a username first.";
    return;
  }

  cancelActiveUsernameImport();
  clearPendingHistoryImport();
  const controller = new AbortController();
  activeUsernameImport = { controller, provider };
  setUsernameImportLoading(true, provider);
  setAsyncStatus(usernameImportStatusEl, "loading", `Importing from ${providerLabel(provider)}...`);
  recMessageEl.textContent = `Reading anime history from ${providerLabel(provider)} for import preview...`;

  try {
    const result =
      provider === "anilist"
        ? await providerAdapter.fetchAniListUsernameImport(username, recommendationIndex, controller.signal)
        : await providerAdapter.fetchMalUsernameImport(username, recommendationIndex, controller.signal);

    if (controller.signal.aborted || activeUsernameImport?.controller !== controller) return;
    activeUsernameImport = null;
    setUsernameImportLoading(false, provider);
    if (result.history.length === 0) {
      setAsyncStatus(usernameImportStatusEl, "empty", `No entries returned by ${providerLabel(provider)}.`);
      recMessageEl.textContent = `No entries returned by ${providerLabel(provider)}; the watched list was left intact.`;
      return;
    }

    setPendingHistoryImport({ entries: result.history, duplicates: result.duplicateCount }, "username");
    recMessageEl.textContent = `Review ${providerLabel(provider)} import counts and choose merge or replace before applying.`;
    setAsyncStatus(usernameImportStatusEl, "ready", `${result.history.length} entries ready for import preview.`);
  } catch (error) {
    if (controller.signal.aborted || activeUsernameImport?.controller !== controller) return;
    if (isAbortError(error)) {
      setAsyncStatus(usernameImportStatusEl, "stale", "Username import request was canceled.");
      recMessageEl.textContent = "Username import request was canceled; the watched list was left intact.";
      return;
    }
    setAsyncStatus(
      usernameImportStatusEl,
      error instanceof ProviderUnavailableError ? "unavailable" : "failed",
      error instanceof ProviderUnavailableError ? `${providerLabel(provider)} import is unavailable.` : `${providerLabel(provider)} import failed.`,
    );
    const message =
      error instanceof Error ? error.message : "Unable to import this username.";
    recMessageEl.textContent = `Username import failed: ${message}`;
  } finally {
    if (activeUsernameImport?.controller === controller) {
      activeUsernameImport = null;
      setUsernameImportLoading(false, provider);
    }
  }
}

function clearPendingHistoryImport(): void {
  pendingHistoryImport = null;
  historyImportPreviewEl.hidden = true;
  historyImportSummaryEl.textContent = "";
  historyImportUnmappedEl.textContent = "";
}

function setPendingHistoryImport(parsed: ParsedHistory, origin: "local" | "username"): void {
  pendingHistoryImport = { parsed, origin };
  historyImportModeEl.value = "merge";
  historyImportPreviewEl.hidden = false;
  renderImportPreview();
}

function renderImportPreview(): void {
  const pending = pendingHistoryImport;
  if (!pending) return;
  const mode: ImportMode = historyImportModeEl.value === "replace" ? "replace" : "merge";
  const counts = previewHistory(pending.parsed, historyEntries, recommendationIndex, mode);
  const incomingPicks = new Set(pending.parsed.entries
    .map((entry) => {
      const nodeId = resolveHistoryAnime(entry, recommendationIndex)?.nodeId;
      return nodeId && preferenceFromHistory(entry, nodeId) ? nodeId : undefined;
    })
    .filter((nodeId): nodeId is string => nodeId !== undefined));
  const picksRemoved = mode === "replace"
    ? selectedAnimeNodeIds.filter((nodeId) => !incomingPicks.has(nodeId)).length : 0;
  const plannedCleared = mode === "merge" ? new Set(pending.parsed.entries
    .filter((entry) => entry.status === "plan_to_watch")
    .map((entry) => resolveHistoryAnime(entry, recommendationIndex)?.nodeId)
    .filter((nodeId): nodeId is string => nodeId !== undefined &&
      selectedAnimePreferences.get(nodeId)?.source === "import")).size : 0;
  historyImportSummaryEl.textContent =
    `${counts.total} entries (${counts.duplicates} duplicate identities collapsed). ` +
    `Mapped: ${counts.mapped}; unmapped: ${counts.unmapped}; unscored: ${counts.unscored}; ` +
    `seen: ${counts.seen}; planned: ${counts.planned}. ` +
    `History new: ${counts.added}; updated: ${counts.updated}; unchanged: ${counts.unchanged}; ` +
    `removed by replace: ${counts.removed}. Preferences removed by replace: ${picksRemoved}; ` +
    `cleared by planned status: ${plannedCleared}.`;
  const unmapped = pending.parsed.entries
    .filter((entry) => !resolveHistoryAnime(entry, recommendationIndex))
    .slice(0, 3).map((entry) => entry.title);
  historyImportUnmappedEl.textContent = unmapped.length > 0
    ? `Unmapped examples kept for later matching: ${unmapped.join("; ")}.` : "All entries match the current graph.";
}

function applyPendingHistoryImport(): void {
  const pending = pendingHistoryImport;
  if (!pending) return;
  const mode: ImportMode = historyImportModeEl.value === "replace" ? "replace" : "merge";
  const previousHistory = [...historyEntries];
  const previousSelected = [...selectedAnimeNodeIds];
  const previousPreferences = new Map(selectedAnimePreferences);
  try {
    const nextHistory = mergeHistory(historyEntries, pending.parsed.entries, mode);
    if (mode === "replace") {
      selectedAnimeNodeIds.splice(0, selectedAnimeNodeIds.length);
      selectedAnimePreferences.clear();
    }
    const mapped: ImportedPreferenceEntry[] = [];
    for (const entry of pending.parsed.entries) {
      const anime = resolveHistoryAnime(entry, recommendationIndex);
      if (!anime) continue;
      const preference = preferenceFromHistory(entry, anime.nodeId);
      if (preference) mapped.push({ anime, preference });
      else if (selectedAnimePreferences.get(anime.nodeId)?.source === "import") {
        selectedAnimePreferences.delete(anime.nodeId);
        selectedAnimeNodeIds.splice(selectedAnimeNodeIds.indexOf(anime.nodeId), 1);
      }
    }
    const selectionSummary = upsertImportedPreferences(mapped);
    historyEntries.splice(0, historyEntries.length, ...nextHistory);
    if (!persistRecommendationState()) {
      throw new Error("Browser storage rejected the import; prior history is intact.");
    }
    renderSelectedAnime();
    renderImportedHistory();
    void updateRecommendations();
    setAsyncStatus(pending.origin === "local" ? bulkImportStatusEl : usernameImportStatusEl,
      "ready", "Import applied and saved in this browser.");
    recMessageEl.textContent =
      `Import applied: ${pending.parsed.entries.length} history entries; ` +
      `preferences added: ${selectionSummary.added}, updated: ${selectionSummary.updated}, ` +
      `unchanged: ${selectionSummary.skipped}.`;
  } catch (error) {
    historyEntries.splice(0, historyEntries.length, ...previousHistory);
    selectedAnimeNodeIds.splice(0, selectedAnimeNodeIds.length, ...previousSelected);
    selectedAnimePreferences.clear();
    for (const [nodeId, preference] of previousPreferences) selectedAnimePreferences.set(nodeId, preference);
    renderSelectedAnime();
    renderImportedHistory();
    setAsyncStatus(pending.origin === "local" ? bulkImportStatusEl : usernameImportStatusEl,
      "failed", "Import was not applied; prior history is intact.");
    recMessageEl.textContent = error instanceof Error ? error.message : "Import was not applied.";
  }
}

function renderImportedHistory(): void {
  historyCountEl.textContent = String(historyEntries.length);
  historyListEl.replaceChildren();
  if (historyEntries.length === 0) {
    const item = document.createElement("li");
    item.textContent = "No imported history yet.";
    historyListEl.append(item);
    return;
  }
  const ordered = [...historyEntries].sort((left, right) =>
    Number(Boolean(resolveHistoryAnime(left, recommendationIndex))) -
    Number(Boolean(resolveHistoryAnime(right, recommendationIndex))));
  for (const entry of ordered.slice(0, 50)) {
    const item = document.createElement("li");
    const mappedAnime = resolveHistoryAnime(entry, recommendationIndex);
    const mapped = mappedAnime !== null;
    item.textContent = `${mappedAnime?.label ?? entry.title} — ${entry.sourceStatus || entry.status}; ` +
      `episodes: ${entry.progressEpisodes ?? "unknown"}; ` +
      `score: ${entry.score ?? "unscored"} (${entry.scoreScale}); ` +
      `${mapped ? "in catalog" : "unmapped, kept"}` +
      (mapped && entry.title !== mappedAnime.label ? `; source title: ${entry.title}` : "");
    historyListEl.append(item);
  }
  if (historyEntries.length > 50) {
    const more = document.createElement("li");
    more.textContent = `${historyEntries.length - 50} more entries saved locally.`;
    historyListEl.append(more);
  }
}

function setAsyncStatus(element: HTMLElement, state: AsyncUiState, message: string): void {
  element.dataset.state = state;
  element.textContent = message;
}

function cancelActiveUsernameImport(): void {
  const active = activeUsernameImport;
  if (!active) return;
  activeUsernameImport = null;
  active.controller.abort();
  setUsernameImportLoading(false, active.provider);
  setAsyncStatus(usernameImportStatusEl, "stale", "Previous username import canceled after recommendation state changed.");
}

function cancelBulkFileLoad(): void {
  if (activeBulkFileLoadId === null) return;
  ++bulkFileLoadId;
  activeBulkFileLoadId = null;
  setAsyncStatus(bulkImportStatusEl, "stale", "Previous local file read was superseded.");
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
  usernameImportSubmitLabel.textContent = loading
    ? `Importing ${providerLabel(provider)}...`
    : "Import User List";
}

function upsertImportedPreferences(entries: ImportedPreferenceEntry[]): {
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
      selectedAnimePreferences.set(entry.anime.nodeId, entry.preference);
      added += 1;
      continue;
    }

    const current = selectedAnimePreferences.get(entry.anime.nodeId);
    if (current?.source === "manual") {
      skipped += 1;
      continue;
    }
    const next = { ...entry.preference, importance: current?.importance ?? 1 };
    if (current && JSON.stringify(current) === JSON.stringify(next)) {
      skipped += 1;
      continue;
    }
    selectedAnimePreferences.set(entry.anime.nodeId, next);
    updated += 1;
  }

  return { added, updated, skipped };
}

function removeSelectedAnime(nodeId: string): void {
  const index = selectedAnimeNodeIds.indexOf(nodeId);
  if (index < 0) {
    return;
  }
  selectedAnimeNodeIds.splice(index, 1);
  selectedAnimePreferences.delete(nodeId);
  persistRecommendationState();
  recMessageEl.textContent = "";
  renderSelectedAnime();
  void updateRecommendations();
}

function renderSelectedAnime(): void {
  watchedCountEl.textContent = String(selectedAnimeNodeIds.length);
  if (selectedAnimeNodeIds.length === 0) {
    selectedAnimeEl.innerHTML = `<p class="muted">No anime added yet.</p>`;
    return;
  }

  const html = selectedAnimeNodeIds
    .map((nodeId) => {
      const anime = recommendationIndex.animeByNodeId.get(nodeId);
      const label = anime?.label ?? nodeId;
      const missingNote = anime ? "" : `<span class="chip-missing-note">Unavailable in this catalog; kept in your saved list</span>`;
      const preference = selectedAnimePreferences.get(nodeId) ?? manualPreference(nodeId);
      const importance = clampImportance(preference.importance);
      const confidence = Math.round(preference.confidence * 100);
      return `
      <div class="chip chip-weighted">
        <span class="chip-title">${escapeHtml(label)}${missingNote}</span>
        <label class="chip-weight-control">
          <span>Preference</span>
          <select data-preference-node-id="${escapeHtml(nodeId)}" aria-label="Preference for ${escapeHtml(label)}">
            <option value="seen"${preference.sentiment === "seen" ? " selected" : ""}>Seen / Unrated</option>
            <option value="liked"${preference.sentiment === "liked" ? " selected" : ""}>Liked</option>
            <option value="disliked"${preference.sentiment === "disliked" ? " selected" : ""}>Disliked</option>
          </select>
        </label>
        <label class="chip-weight-control">
          <span>Importance</span>
          <input
            type="range"
            min="${MIN_IMPORTANCE}"
            max="${MAX_IMPORTANCE}"
            step="${IMPORTANCE_STEP}"
            value="${importance.toFixed(1)}"
            data-importance-node-id="${escapeHtml(nodeId)}"
            aria-label="Importance for ${escapeHtml(label)}"
            ${preference.sentiment === "seen" ? "disabled" : ""}
          />
          <output class="chip-importance-value">${importance.toFixed(1)}x</output>
        </label>
        <span class="chip-confidence">${preference.sentiment === "seen" ? "No preference signal" : `${confidence}% confidence`} · ${escapeHtml(preference.source)}</span>
        <button type="button" data-node-id="${escapeHtml(nodeId)}" aria-label="Remove ${escapeHtml(label)}">x</button>
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
  persistRecommendationState();
  input.value = "";
  recMessageEl.textContent = mode === "include"
    ? `Limited results to ${anime.label} and other Include Only titles; exclusions still win.`
    : `Excluded ${anime.label}; exclusions win over Include Only.`;
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
    .map((nodeId) => {
      const anime = recommendationIndex.animeByNodeId.get(nodeId);
      const label = anime?.label ?? nodeId;
      const missingNote = anime ? "" : `<span class="chip-missing-note">Unavailable in this catalog; kept in your saved list</span>`;
      return `
      <div class="chip">
        <span class="chip-title">${escapeHtml(label)}${missingNote}</span>
        <button type="button" ${dataAttrName}="${escapeHtml(nodeId)}" aria-label="Remove ${escapeHtml(label)}">x</button>
      </div>
    `;
    })
    .join("");
  container.innerHTML = html;
}

async function updateRecommendations(): Promise<void> {
  recommendationController?.abort();
  // A later user action can retry transient metadata failures; unavailable IDs stay known.
  animeMetadataFailed.clear();
  const controller = new AbortController();
  recommendationController = controller;
  const runId = ++recommendationRunId;
  const preferences = selectedAnimeNodeIds
    .map((nodeId) => selectedAnimePreferences.get(nodeId))
    .filter((item): item is AnimePreference => item !== undefined);

  if (selectedAnimeNodeIds.length === 0) {
    recSummaryEl.textContent = "Add a title and mark your preference to start.";
    recEngineStatusEl.textContent =
      recommendationMode === "graph"
        ? "Using graph recommendations."
        : recommendationMode === "model"
          ? "Using ML model recommendations."
          : `Using hybrid recommendations (${Math.round(modelBlendWeight * 100)}% model).`;
    recResultsEl.innerHTML = "";
    setMetadataStatus("empty", "Metadata: add anime to begin.");
    renderSelectedAnime();
    return;
  }

  if (!(recommendationMode === "graph"
    ? preferences.some((item) => item.sentiment === "liked")
    : preferences.some((item) => item.sentiment !== "seen"))) {
    recEngineStatusEl.textContent = recommendationMode === "graph"
      ? "Graph waiting for a Liked title."
      : "Model waiting for a Liked or Disliked title.";
    recSummaryEl.textContent = recommendationMode === "graph"
      ? "Seen and disliked titles are excluded. Mark at least one title Liked for graph suggestions."
      : "Seen titles are excluded. Mark a title Liked or Disliked for model suggestions.";
    recResultsEl.innerHTML = "";
    setMetadataStatus("empty", "Metadata: no preference signal yet.");
    renderSelectedAnime();
    return;
  }

  const eligibilityPolicy = createCandidateEligibilityPolicy({
    index: recommendationIndex,
    preferences,
    history: historyEntries,
    includeOnlyNodeIds: includeCandidateNodeIds,
    excludeNodeIds: excludeCandidateNodeIds,
    filters: recommendationFilters,
  });
  const graphRecommendations = buildGraphRecommendationsForPreferences(preferences, recommendationIndex);
  let modelRecommendations: RecommendationResult[] = [];
  let activeRankingMode: EligibilityRankingMode = recommendationMode;
  let usingCatalogFallback = false;
  let fallbackReason: string | null = null;
  let modelFactors: number | null = null;
  let modelCoverageNote = "";
  const sources: {
    graph: RecommendationResult[];
    model: RecommendationResult[];
    fallback: RecommendationResult[];
  } = { graph: graphRecommendations, model: modelRecommendations, fallback: [] };
  function showActiveEngine(): void {
    if (activeRankingMode === "fallback") {
      recEngineStatusEl.textContent =
        `Using catalog coverage baseline (positive graph connections, then anime ID). ${fallbackReason ?? ""}`;
    } else if (activeRankingMode === "graph") {
      recEngineStatusEl.textContent = fallbackReason
        ? `Using graph fallback. ${fallbackReason}`
        : "Using graph recommendations.";
    } else if (activeRankingMode === "model") {
      recEngineStatusEl.textContent = `Using ML model recommendations (${modelFactors} factors)${modelCoverageNote}.`;
    } else {
      recEngineStatusEl.textContent =
        `Using hybrid recommendations (${Math.round(modelBlendWeight * 100)}% model, ${Math.round((1 - modelBlendWeight) * 100)}% graph)${modelCoverageNote}.`;
    }
  }
  function selectCatalogBaseline(): void {
    activeRankingMode = "fallback";
    usingCatalogFallback = true;
    sources.fallback = buildCatalogCoverageRecommendations(recommendationIndex);
    showActiveEngine();
  }
  setMetadataStatus(demoMode ? "demo" : "loading", demoMode ? "Metadata: synthetic demo catalog." : "Metadata: loading recommendations...");

  if (recommendationMode !== "graph") {
    recEngineStatusEl.textContent = "Loading ML model recommendations...";
    const modelIndex = await ensureModelRecommendationIndex();
    if (runId !== recommendationRunId) {
      return;
    }
    if (!modelIndex) {
      fallbackReason = modelLoadError
        ? `Invalid ML model artifact: ${modelLoadError}`
        : "ML model data not found (expected model-mf-web.compact.json(.gz) or model-mf-web.json(.gz)).";
      activeRankingMode = "graph";
    } else {
      modelFactors = modelIndex.factors;
      const signals = preferences.filter((item) => item.sentiment !== "seen");
      const mappedSignals = signals.filter((item) => {
        const anime = recommendationIndex.animeByNodeId.get(item.nodeId);
        return anime !== undefined && modelIndex.animeByAnimeId.has(anime.animeId);
      }).length;
      if (mappedSignals < signals.length) {
        modelCoverageNote = `; model mapped ${mappedSignals}/${signals.length} preference signals`;
      }
      modelRecommendations = buildModelRecommendationsForPreferences(preferences, recommendationIndex, modelIndex);
      sources.model = modelRecommendations;
      if (eligibilityPolicy.evaluate(modelRecommendations, animeMetadataCache).structurallyEligible.length === 0) {
        fallbackReason = mappedSignals === 0
          ? "ML model maps no selected preference signals."
          : "ML model has no candidates in the current catalog and eligibility set.";
        activeRankingMode = "graph";
      }
    }
  }
  showActiveEngine();

  let initialEligibility = rankEligibleCandidates(
    activeRankingMode, sources, eligibilityPolicy, animeMetadataCache, modelBlendWeight,
  );
  if (recommendationMode !== "graph" && activeRankingMode !== "graph" &&
      initialEligibility.structurallyEligible.length === 0) {
    activeRankingMode = "graph";
    fallbackReason = "No ML candidates passed catalog and eligibility rules.";
    showActiveEngine();
    initialEligibility = rankEligibleCandidates(
      activeRankingMode, sources, eligibilityPolicy, animeMetadataCache, modelBlendWeight,
    );
  }
  if (fallbackReason && initialEligibility.structurallyEligible.length === 0) {
    selectCatalogBaseline();
    initialEligibility = rankEligibleCandidates(
      activeRankingMode, sources, eligibilityPolicy, animeMetadataCache, modelBlendWeight,
    );
  }
  let structuralCandidates = initialEligibility.structurallyEligible;
  if (structuralCandidates.length === 0) {
    recSummaryEl.textContent = includeCandidateNodeIds.length > 0
      ? "No scored candidates match Include Only after watched and excluded titles are removed."
      : "No eligible recommendations found from these preferences. Try another liked title or review filters.";
    recResultsEl.innerHTML = "";
    setMetadataStatus("empty", "Metadata: no candidates.");
    return;
  }

  let metadataCandidateAnimeIds: number[] = [];
  async function hydrateRankedCandidates(candidates: RecommendationResult[]): Promise<boolean> {
    metadataCandidateAnimeIds = candidates
      .slice(0, METADATA_PREFETCH_LIMIT)
      .map((item) => item.anime.animeId)
      .filter((animeId) => Number.isFinite(animeId) && animeId > 0);
    const metadataPrefetchLimit = hasActiveRecommendationFilters(recommendationFilters)
      ? METADATA_PREFETCH_WITH_FILTER_LIMIT
      : Math.min(12, metadataCandidateAnimeIds.length);
    if (metadataPrefetchLimit > 0) {
      await hydrateMetadataForAnimeIds(metadataCandidateAnimeIds, metadataPrefetchLimit, controller.signal);
    }
    return runId === recommendationRunId && !controller.signal.aborted;
  }
  if (!await hydrateRankedCandidates(structuralCandidates)) {
    return;
  }

  let finalEligibility = rankEligibleCandidates(
    activeRankingMode, sources, eligibilityPolicy, animeMetadataCache, modelBlendWeight,
  );
  if (recommendationMode !== "graph" && !usingCatalogFallback &&
      finalEligibility.recommendations.length === 0) {
    if (activeRankingMode !== "graph") {
      activeRankingMode = "graph";
      fallbackReason = "No ML candidates passed catalog and required filters.";
      showActiveEngine();
      structuralCandidates = rankEligibleCandidates(
        activeRankingMode, sources, eligibilityPolicy, animeMetadataCache, modelBlendWeight,
      ).structurallyEligible;
      if (!await hydrateRankedCandidates(structuralCandidates)) {
        return;
      }
      finalEligibility = rankEligibleCandidates(
        activeRankingMode, sources, eligibilityPolicy, animeMetadataCache, modelBlendWeight,
      );
    }
    if (finalEligibility.recommendations.length === 0) {
      selectCatalogBaseline();
      structuralCandidates = rankEligibleCandidates(
        activeRankingMode, sources, eligibilityPolicy, animeMetadataCache, modelBlendWeight,
      ).structurallyEligible;
      if (!await hydrateRankedCandidates(structuralCandidates)) {
        return;
      }
      finalEligibility = rankEligibleCandidates(
        activeRankingMode, sources, eligibilityPolicy, animeMetadataCache, modelBlendWeight,
      );
    }
  }
  updateGenreFilterOptions(structuralCandidates);
  const filteredRecommendations = finalEligibility.recommendations;
  if (filteredRecommendations.length === 0) {
    recSummaryEl.textContent = hasActiveRecommendationFilters(recommendationFilters)
      ? "No recommendations match your current metadata filters."
      : "No eligible recommendations found from these preferences.";
    recResultsEl.innerHTML = "";
    const metadataState: AsyncUiState = demoMode ? "demo"
      : metadataCandidateAnimeIds.some((id) => animeMetadataFailed.has(id)) ? "failed"
        : metadataCandidateAnimeIds.some((id) => animeMetadataUnavailable.has(id)) ? "unavailable" : "empty";
    setMetadataStatus(metadataState,
      metadataState === "failed" ? "Metadata request failed; some candidates could not be checked."
        : metadataState === "unavailable" ? "Metadata is unavailable for some candidates."
          : hasActiveRecommendationFilters(recommendationFilters) && finalEligibility.missingMetadataCount > 0
            ? `Metadata: ${finalEligibility.missingMetadataCount} candidate(s) skipped due to missing metadata.`
            : "Metadata: no matches after filters.",
    );
    return;
  }

  const methodLabel = usingCatalogFallback ? "catalog coverage baseline"
    : activeRankingMode === "graph" ? fallbackReason ? "graph edge fallback" : "graph edge ranking"
      : activeRankingMode === "model" ? "ML model ranking" : "hybrid graph+ML ranking";
  const filterSummary = formatActiveFilterSummary();
  recSummaryEl.textContent = `Showing top ${Math.min(MAX_RECOMMENDATIONS, filteredRecommendations.length)} recommendations from ${filteredRecommendations.length} candidates (${methodLabel})${filterSummary}.`;

  const visibleRecommendations = filteredRecommendations
    .slice(0, MAX_RECOMMENDATIONS)
    .map((item) => renderRecommendationCard(item, usingCatalogFallback));
  recResultsEl.innerHTML = visibleRecommendations.join("");

  const visibleWithMetadata = filteredRecommendations
    .slice(0, MAX_RECOMMENDATIONS)
    .filter((item) => animeMetadataCache.has(item.anime.animeId)).length;
  const visibleMissingMetadata = filteredRecommendations
    .slice(0, MAX_RECOMMENDATIONS)
    .filter(
      (item) =>
        !animeMetadataCache.has(item.anime.animeId) &&
        !animeMetadataUnavailable.has(item.anime.animeId) &&
        !animeMetadataFailed.has(item.anime.animeId),
    ).length;
  const missingNote =
    finalEligibility.missingMetadataCount > 0
      ? ` | skipped (missing metadata): ${finalEligibility.missingMetadataCount}`
      : "";
  const visibleIds = filteredRecommendations.slice(0, MAX_RECOMMENDATIONS).map((item) => item.anime.animeId);
  const metadataState: AsyncUiState = demoMode ? "demo"
    : visibleIds.some((id) => animeMetadataFailed.has(id)) ? "failed"
      : visibleIds.some((id) => animeMetadataUnavailable.has(id)) ? "unavailable" : "ready";
  const providerNote = metadataState === "failed" ? " | some metadata requests failed"
    : metadataState === "unavailable" ? " | some metadata unavailable"
      : metadataState === "demo" ? " | synthetic demo data" : "";
  setMetadataStatus(metadataState,
    `Metadata loaded for ${visibleWithMetadata}/${Math.min(MAX_RECOMMENDATIONS, filteredRecommendations.length)} visible recommendations` +
      `${visibleMissingMetadata > 0 ? ` | pending: ${visibleMissingMetadata}` : ""}` +
      missingNote + providerNote,
  );
}

function renderRecommendationCard(item: RecommendationResult, catalogFallback = false): string {
  const metadata = animeMetadataCache.get(item.anime.animeId) ?? null;
  const imageUrl = safeExternalImageUrl(metadata?.imageUrl ?? "");
  const coverHtml =
    imageUrl
      ? `<img class="rec-cover" src="${escapeHtml(imageUrl)}" alt="Cover for ${escapeHtml(item.anime.label)}" loading="lazy" referrerpolicy="no-referrer" />`
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
  const reason = catalogFallback
    ? `Catalog coverage: ${item.supportCount} positive graph connections; personal preference evidence is unavailable.`
    : formatRecommendationWhyHtml(item);
  const score = catalogFallback ? `${item.supportCount} connections` : formatWeight(item.score);
  const supportLine = catalogFallback
    ? `Positive graph connections: ${item.supportCount}`
    : `Support edges: ${item.supportCount} | Strongest: ${formatWeight(item.strongest)}`;

  return `
      <li class="rec-item">
        <div class="rec-main">
          ${coverHtml}
          <div class="rec-copy">
            <div class="rec-title">${escapeHtml(item.anime.label)}</div>
            <div class="rec-meta">${supportLine}</div>
            <div class="rec-meta rec-meta-details">${escapeHtml(metadataMeta)}</div>
            <p class="${synopsisClass}">${escapeHtml(synopsisText)}</p>
            <div class="rec-why">${reason}</div>
          </div>
        </div>
        <div class="rec-score">${score}</div>
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
    parts.push(`${demoMode ? "Demo" : "MAL"} ${metadata.score.toFixed(2)}`);
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

function setMetadataStatus(state: AsyncUiState, message: string): void {
  setAsyncStatus(metadataStatusEl, state, message);
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
  signal: AbortSignal,
): Promise<void> {
  const uniqueTargets = [...new Set(animeIds)]
    .filter(
      (animeId) =>
        animeId > 0 &&
        !animeMetadataCache.has(animeId) &&
        !animeMetadataUnavailable.has(animeId) &&
        !animeMetadataFailed.has(animeId),
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
        while (queue.length > 0 && !signal.aborted) {
          const animeId = queue.shift();
          if (animeId === undefined) {
            return;
          }
          await ensureAnimeMetadata(animeId, signal);
        }
      })(),
    );
  }

  await Promise.all(workers);
}

async function ensureAnimeMetadata(animeId: number, signal: AbortSignal): Promise<AnimeMetadata | null> {
  const cached = animeMetadataCache.get(animeId);
  if (cached) {
    return cached;
  }
  if (demoMode) return null;
  if (animeMetadataUnavailable.has(animeId)) {
    return null;
  }
  if (animeMetadataFailed.has(animeId) || signal.aborted) return null;
  try {
    const outcome = await providerAdapter.fetchAnimeMetadataFromJikan(animeId, signal);
    if (signal.aborted) return null;
    if (outcome.state === "unavailable") {
      animeMetadataUnavailable.add(animeId);
      return null;
    }
    if (outcome.state === "failed") {
      animeMetadataFailed.add(animeId);
      return null;
    }
    animeMetadataCache.set(animeId, outcome.metadata);
    return outcome.metadata;
  } catch (error) {
    if (!signal.aborted) animeMetadataFailed.add(animeId);
    return null;
  }
}

function truncateText(value: string, maxLength: number): string {
  const normalized = value.trim().replace(/\s+/g, " ");
  if (normalized.length <= maxLength) {
    return normalized;
  }
  return `${normalized.slice(0, maxLength - 3)}...`;
}

async function loadSeasonalTrending(force: boolean): Promise<void> {
  if (demoCatalog) {
    seasonalItems = demoCatalog.slice(0, 3).map((item) => ({
      animeId: item.animeId, title: item.title, score: item.score,
      year: item.year, season: null, imageUrl: "",
    }));
    setAsyncStatus(seasonalStatusEl, "demo", "Invented titles from the local demo catalog.");
    renderSeasonalList();
    return;
  }
  if (seasonalLoadingPromise && !force) {
    return seasonalLoadingPromise;
  }

  seasonalController?.abort();
  const controller = new AbortController();
  seasonalController = controller;
  refreshSeasonalBtn.disabled = true;
  setAsyncStatus(seasonalStatusEl, "loading", "Loading current season...");

  const promise = (async () => {
    try {
      const items = await providerAdapter.fetchSeasonalAnime(SEASONAL_LIST_LIMIT, controller.signal);
      if (controller.signal.aborted || seasonalController !== controller) return;
      seasonalItems = items;
      setAsyncStatus(seasonalStatusEl, items.length > 0 ? "ready" : "empty",
        seasonalItems.length > 0
          ? `Loaded ${seasonalItems.length} seasonal anime from Jikan.`
          : "No seasonal anime returned.");
      renderSeasonalList();
    } catch (error) {
      if (controller.signal.aborted || seasonalController !== controller) return;
      if (isAbortError(error)) {
        setAsyncStatus(seasonalStatusEl, "stale", "Seasonal request was canceled.");
        return;
      }
      const message = error instanceof Error ? error.message : "Request failed.";
      setAsyncStatus(seasonalStatusEl, error instanceof ProviderUnavailableError ? "unavailable" : "failed",
        `Unable to load seasonal anime: ${message}`);
      seasonalItems = [];
      renderSeasonalList();
    } finally {
      if (seasonalController === controller) {
        refreshSeasonalBtn.disabled = false;
        seasonalController = null;
        seasonalLoadingPromise = null;
      }
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
        subtitleParts.push(`${demoMode ? "Demo" : "MAL"} ${item.score.toFixed(2)}`);
      }
      const subtitle = subtitleParts.length > 0 ? subtitleParts.join(" | ") : "No stats";
      const imageUrl = safeExternalImageUrl(item.imageUrl);
      const coverHtml = imageUrl
        ? `<img class="seasonal-cover" src="${escapeHtml(imageUrl)}" alt="Cover for ${escapeHtml(item.title)}" loading="lazy" referrerpolicy="no-referrer" />`
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
            title="${inGraph ? "Mark as seen" : "Not available in current graph"}"
          >
            ${inGraph ? "Mark Seen" : "N/A"}
          </button>
        </li>
      `;
    })
    .join("");
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

function resolveNetworkNodeQuery(
  query: string,
  graphDataValue: LoadedGraphData,
): GraphNode | null {
  const raw = query.trim();
  if (!raw) {
    return null;
  }

  const rawLower = raw.toLowerCase();
  const normalized = normalizeTitle(raw);
  const nodes = getGraphNodes(graphDataValue);

  const byExactId = nodes.find((node) => node.id.toLowerCase() === rawLower);
  if (byExactId) {
    return byExactId;
  }

  if (/^\d+$/.test(raw)) {
    const animeNodeId = `anime:${Number.parseInt(raw, 10)}`;
    const byAnimeId = nodes.find((node) => node.id === animeNodeId);
    if (byAnimeId) {
      return byAnimeId;
    }
  }

  const byExactLabel = nodes.find(
    (node) => normalizeTitle(node.label) === normalized,
  );
  if (byExactLabel) {
    return byExactLabel;
  }

  const byStartsWith = nodes.find((node) =>
    normalizeTitle(node.label).startsWith(normalized),
  );
  if (byStartsWith) {
    return byStartsWith;
  }

  return (
    nodes.find((node) =>
      normalizeTitle(`${node.label} ${node.id}`).includes(normalized),
    ) ?? null
  );
}

function selectNodeAndFocus(nodeId: string): void {
  selectedNodeId = nodeId;
  if (activeView !== "network") {
    setActiveView("network", false);
    runtime.frame(() => {
      runtime.frame(() => {
        focusNodeInRenderer(nodeId);
      });
    });
    return;
  }
  if (!currentGraph || !currentGraph.hasNode(nodeId)) {
    rerenderGraph();
    runtime.frame(() => {
      focusNodeInRenderer(nodeId);
    });
    return;
  }
  focusNodeInRenderer(nodeId);
}

function focusNodeInRenderer(nodeId: string): void {
  if (!currentGraph || !currentGraph.hasNode(nodeId)) {
    return;
  }
  selectedNodeId = nodeId;
  renderInspectPanel(nodeId);
  renderSvgGraph(currentGraph);
  const target = graphContainer.querySelector<SVGElement>(
    `[data-node-id="${cssEscapeAttributeValue(nodeId)}"]`,
  );
  target?.scrollIntoView({
    block: "center",
    inline: "center",
    behavior: prefersReducedMotion() ? "auto" : "smooth",
  });
}

function rerenderGraph(): void {
  const minWeight = Number.parseFloat(minWeightInput.value);
  minWeightValue.textContent = minWeight.toFixed(2);
  const showAnimeAnimeEdges = toggleAnimeEdges.checked;
  const showUsers = toggleUsers.checked;
  const runId = ++graphRenderRunId;
  const renderSource = explorerGraphData ?? graphData;

  setGraphLoadingState(true, "Render status: rendering network...");

  runtime.schedule(() => {
    if (runId !== graphRenderRunId) {
      return;
    }

    if (activeView !== "network") {
      setGraphLoadingState(false, "Render status: ready.");
      return;
    }

    const startedAt = runtime.monotonicNow();
    try {
      const renderResult = renderGraph(
        renderSource,
        minWeight,
        showAnimeAnimeEdges,
        showUsers,
      );
      if (runId !== graphRenderRunId) {
        return;
      }
      const elapsedMs = Math.max(1, Math.round(runtime.monotonicNow() - startedAt));
      const visibleNodes = currentGraph ? currentGraph.order : 0;
      const visibleEdges = renderResult.renderedEdgeCount;
      const limitSuffix = renderResult.edgeLimitHit
        ? `, capped from ${renderResult.totalEligibleEdgeCount.toLocaleString()} matching edges`
        : "";
      setGraphLoadingState(
        false,
        `Render status: ${visibleNodes.toLocaleString()} nodes, ${visibleEdges.toLocaleString()} edges (${elapsedMs} ms${limitSuffix}).`,
      );
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "unknown error";
      setGraphLoadingState(false, `Render status: failed (${errorMessage}).`);
      console.error("Graph render failed.", error);
    }
  }, 0);
}

function setGraphLoadingState(loading: boolean, statusMessage: string): void {
  graphLoadingEl.hidden = !loading;
  graphLoadingEl.setAttribute("aria-hidden", loading ? "false" : "true");
  graphShell.setAttribute("aria-busy", loading ? "true" : "false");
  graphLoadingMessageEl.textContent = loading ? "Rendering network..." : "";
  networkRenderStatusEl.textContent = statusMessage;
}

function renderGraph(
  graphDataValue: LoadedGraphData,
  minAbsoluteWeight: number,
  showAnimeAnimeEdges: boolean,
  showUsers: boolean,
): { renderedEdgeCount: number; totalEligibleEdgeCount: number; edgeLimitHit: boolean } {
  const graph = new Graph({ multi: true, type: "undirected" });
  const selectedEdges = selectRenderableEdges(
    graphDataValue,
    minAbsoluteWeight,
    showAnimeAnimeEdges,
    showUsers,
  );
  const activeNodeIds = new Set<string>();

  for (const edge of selectedEdges.edges) {
    activeNodeIds.add(edge.source);
    activeNodeIds.add(edge.target);
  }

  for (const node of getGraphNodes(graphDataValue)) {
    if (!showUsers && node.nodeType === "user") {
      continue;
    }
    if (!activeNodeIds.has(node.id)) {
      continue;
    }

    const isUser = node.nodeType === "user";
    graph.addNode(node.id, {
      label: node.label,
      nodeType: node.nodeType,
      size: isUser ? 5.2 : 2.8,
      color: isUser ? "#ff8a00" : "#0f8b8d",
      x: runtime.random(),
      y: runtime.random(),
    });
  }

  for (const edge of selectedEdges.edges) {
    if (!graph.hasNode(edge.source) || !graph.hasNode(edge.target)) {
      continue;
    }

    const sign = edge.weight > 0 ? "positive" : edge.weight < 0 ? "negative" : "neutral";
    const color = sign === "neutral" ? "#aab4c088"
      : edge.edgeType === "user-anime"
        ? sign === "positive" ? "#f4d35eaa" : "#eaa0d6aa"
        : sign === "positive" ? "#6fffe988" : "#ff8f7a99";
    graph.addEdgeWithKey(edge.id, edge.source, edge.target, {
      size: edge.edgeType === "user-anime" ? 1.4 : 0.7,
      color,
      weight: Math.max(Math.abs(edge.weight), 0.01),
      signedWeight: edge.weight,
      sign,
      edgeType: edge.edgeType,
    });
  }

  applyLayout(graph);
  currentGraph = graph;
  renderSvgGraph(graph);

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

  return {
    renderedEdgeCount: graph.size,
    totalEligibleEdgeCount: selectedEdges.totalEligibleEdgeCount,
    edgeLimitHit: selectedEdges.edgeLimitHit,
  };
}

function selectRenderableEdges(
  graphDataValue: LoadedGraphData,
  minAbsoluteWeight: number,
  showAnimeAnimeEdges: boolean,
  showUsers: boolean,
): {
  edges: GraphEdge[];
  totalEligibleEdgeCount: number;
  edgeLimitHit: boolean;
} {
  const selectedAnimeAnimeEdges: GraphEdge[] = [];
  const selectedUserAnimeEdges: GraphEdge[] = [];
  let eligibleAnimeAnimeCount = 0;
  let eligibleUserAnimeCount = 0;

  if (isCompactGraphData(graphDataValue)) {
    for (const [userIndex, animeIndex, weight] of graphDataValue.ua) {
      if (!showUsers || !Number.isFinite(weight)) {
        continue;
      }
      eligibleUserAnimeCount += 1;
      if (selectedUserAnimeEdges.length >= MAX_RENDERED_USER_ANIME_EDGES) {
        continue;
      }
      const userId = graphDataValue.userIds[userIndex];
      const animeEntry = graphDataValue.anime[animeIndex];
      if (!userId || !animeEntry) {
        continue;
      }
      selectedUserAnimeEdges.push({
        id: `ua:${userId}:${animeEntry[0]}`,
        source: `user:${userId}`,
        target: `anime:${animeEntry[0]}`,
        edgeType: "user-anime",
        weight,
      });
    }

    if (showAnimeAnimeEdges) {
      for (const [leftAnimeIndex, rightAnimeIndex, weight] of graphDataValue.aa) {
        if (!Number.isFinite(weight) || Math.abs(weight) < minAbsoluteWeight) {
          continue;
        }
        eligibleAnimeAnimeCount += 1;
        if (selectedAnimeAnimeEdges.length >= MAX_RENDERED_ANIME_ANIME_EDGES) {
          continue;
        }
        const leftAnime = graphDataValue.anime[leftAnimeIndex];
        const rightAnime = graphDataValue.anime[rightAnimeIndex];
        if (!leftAnime || !rightAnime) {
          continue;
        }
        selectedAnimeAnimeEdges.push({
          id: `aa:${leftAnime[0]}:${rightAnime[0]}`,
          source: `anime:${leftAnime[0]}`,
          target: `anime:${rightAnime[0]}`,
          edgeType: "anime-anime",
          weight,
        });
      }
    }
  } else {
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

      if (edge.edgeType === "anime-anime") {
        eligibleAnimeAnimeCount += 1;
        if (selectedAnimeAnimeEdges.length < MAX_RENDERED_ANIME_ANIME_EDGES) {
          selectedAnimeAnimeEdges.push(edge);
        }
      } else {
        eligibleUserAnimeCount += 1;
        if (selectedUserAnimeEdges.length < MAX_RENDERED_USER_ANIME_EDGES) {
          selectedUserAnimeEdges.push(edge);
        }
      }
    }
  }
  const totalEligibleEdgeCount =
    eligibleAnimeAnimeCount + eligibleUserAnimeCount;
  const renderedEdgeCount =
    selectedAnimeAnimeEdges.length + selectedUserAnimeEdges.length;

  return {
    edges: [...selectedUserAnimeEdges, ...selectedAnimeAnimeEdges],
    totalEligibleEdgeCount,
    edgeLimitHit: renderedEdgeCount < totalEligibleEdgeCount,
  };
}

function statLine(label: string, value: string): string {
  return `<div class="stat-row"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`;
}

async function ensureExplorerGraphData(): Promise<LoadedGraphData> {
  if (explorerGraphData) {
    return explorerGraphData;
  }
  if (!explorerGraphDataPromise) {
    explorerGraphDataPromise = artifactLoader.fetchExplorerGraph(graphData);
  }
  explorerGraphData = await explorerGraphDataPromise;
  return explorerGraphData;
}

async function ensureModelRecommendationIndex(): Promise<ModelRecommendationIndex | null> {
  if (!modelRecommendationIndexPromise) {
    modelRecommendationIndexPromise = artifactLoader.fetchModelRecommendationIndex().catch((error: unknown) => {
      modelLoadError = error instanceof Error ? error.message : "unknown validation error";
      console.error("Model artifact load failed.", error);
      return null;
    });
  }
  return modelRecommendationIndexPromise;
}

async function loadRequiredArtifact<T>(operation: Promise<T>): Promise<T> {
  try {
    return await operation;
  } catch (error) {
    const detail = error instanceof Error ? error.message : "unknown artifact error";
    recMessageEl.textContent = `Data unavailable: ${detail}`;
    recMessageEl.setAttribute("role", "alert");
    recEngineStatusEl.textContent = "Recommendations unavailable until the data artifact is repaired.";
    throw error;
  }
}

function getGraphNodes(graphDataValue: LoadedGraphData): GraphNode[] {
  if (!isCompactGraphData(graphDataValue)) {
    return graphDataValue.nodes;
  }

  const nodes: GraphNode[] = [];
  for (let i = 0; i < graphDataValue.userIds.length; i += 1) {
    const userId = graphDataValue.userIds[i];
    nodes.push({
      id: `user:${userId}`,
      label: `User ${String(userId).slice(0, 8)}`,
      nodeType: "user",
    });
  }

  for (let i = 0; i < graphDataValue.anime.length; i += 1) {
    const animeEntry = graphDataValue.anime[i];
    const animeId = animeEntry[0];
    const title = animeEntry[1];
    nodes.push({
      id: `anime:${animeId}`,
      label: String(title),
      nodeType: "anime",
    });
  }

  return nodes;
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
    assignRingLayout(graph);
  } catch (error) {
    console.warn("Graph layout failed; using fallback ring layout.", error);
    assignRingLayout(graph);
  }

  sanitizeCoordinates(graph);
}

function renderSvgGraph(graph: Graph): void {
  graphContainer.replaceChildren();
  if (graph.order === 0) {
    return;
  }

  const width = 1200;
  const height = 900;
  const padding = 56;
  const coords = new Map<string, { x: number; y: number }>();
  let minX = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;

  graph.forEachNode((node, attributes) => {
    const x = Number(attributes.x);
    const y = Number(attributes.y);
    minX = Math.min(minX, x);
    maxX = Math.max(maxX, x);
    minY = Math.min(minY, y);
    maxY = Math.max(maxY, y);
    coords.set(node, { x, y });
  });

  const spanX = Math.max(maxX - minX, 0.001);
  const spanY = Math.max(maxY - minY, 0.001);
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("class", "graph-svg");
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", "Anime recommendation network");

  const edgeLayer = document.createElementNS(SVG_NS, "g");
  edgeLayer.setAttribute("class", "graph-edge-layer");
  const nodeLayer = document.createElementNS(SVG_NS, "g");
  nodeLayer.setAttribute("class", "graph-node-layer");
  const labelLayer = document.createElementNS(SVG_NS, "g");
  labelLayer.setAttribute("class", "graph-label-layer");

  const selected = selectedNodeId;
  const connectedToSelected = new Set<string>();
  if (selected && graph.hasNode(selected)) {
    graph.forEachNeighbor(selected, (neighbor) => {
      connectedToSelected.add(neighbor);
    });
  }

  graph.forEachEdge((_edgeKey, attributes, source, target) => {
    const sourceCoord = coords.get(source);
    const targetCoord = coords.get(target);
    if (!sourceCoord || !targetCoord) {
      return;
    }

    const line = document.createElementNS(SVG_NS, "line");
    line.setAttribute("x1", String(scaleGraphCoordinate(sourceCoord.x, minX, spanX, padding, width)));
    line.setAttribute("y1", String(scaleGraphCoordinate(sourceCoord.y, minY, spanY, padding, height)));
    line.setAttribute("x2", String(scaleGraphCoordinate(targetCoord.x, minX, spanX, padding, width)));
    line.setAttribute("y2", String(scaleGraphCoordinate(targetCoord.y, minY, spanY, padding, height)));

    const edgeAttrs = attributes as Record<string, unknown>;
    const baseColor =
      typeof edgeAttrs.color === "string" ? edgeAttrs.color : "#6fffe944";
    const baseSize = Number(edgeAttrs.size) || 1;
    const highlighted =
      selected !== null && (source === selected || target === selected);
    line.setAttribute("stroke", selected && !highlighted ? "#30415655" : baseColor);
    line.setAttribute("stroke-width", String(highlighted ? baseSize * 1.3 : baseSize));
    line.setAttribute("stroke-linecap", "round");
    const sign = edgeAttrs.sign;
    if (sign === "positive" || sign === "negative" || sign === "neutral") {
      line.setAttribute("data-edge-sign", sign);
      if (sign === "neutral") line.setAttribute("stroke-dasharray", "3 3");
    }
    edgeLayer.appendChild(line);
  });

  graph.forEachNode((node, attributes) => {
    const coord = coords.get(node);
    if (!coord) {
      return;
    }

    const x = scaleGraphCoordinate(coord.x, minX, spanX, padding, width);
    const y = scaleGraphCoordinate(coord.y, minY, spanY, padding, height);
    const nodeAttrs = attributes as Record<string, unknown>;
    const baseColor = typeof nodeAttrs.color === "string" ? nodeAttrs.color : "#0f8b8d";
    const baseSize = Number(nodeAttrs.size) || 3;
    const isSelected = selected === node;
    const isConnected = selected !== null && connectedToSelected.has(node);
    const dimmed = selected !== null && !isSelected && !isConnected;

    const circle = document.createElementNS(SVG_NS, "circle");
    circle.setAttribute("cx", String(x));
    circle.setAttribute("cy", String(y));
    circle.setAttribute("r", String(isSelected ? baseSize * 1.5 : baseSize));
    circle.setAttribute("fill", isSelected ? "#ffd166" : dimmed ? "#4f607388" : baseColor);
    circle.setAttribute("data-node-id", node);
    circle.setAttribute("tabindex", "0");
    circle.setAttribute("role", "button");
    circle.setAttribute("aria-label", String(nodeAttrs.label ?? node));
    circle.addEventListener("click", () => {
      selectedNodeId = node;
      networkSearchMessage.textContent = `Focused: ${node}`;
      renderInspectPanel(node);
      renderSvgGraph(graph);
    });
    circle.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectedNodeId = node;
        networkSearchMessage.textContent = `Focused: ${node}`;
        renderInspectPanel(node);
        renderSvgGraph(graph);
      }
    });
    nodeLayer.appendChild(circle);

    if (isSelected || graph.order <= 180) {
      const label = document.createElementNS(SVG_NS, "text");
      label.setAttribute("x", String(x + 8));
      label.setAttribute("y", String(y - 8));
      label.setAttribute("fill", dimmed ? "#6d7b8c" : "#f5f7fa");
      label.setAttribute("font-size", isSelected ? "18" : "12");
      label.setAttribute("font-family", "IBM Plex Mono, monospace");
      label.textContent = String(nodeAttrs.label ?? node);
      labelLayer.appendChild(label);
    }
  });

  svg.addEventListener("click", (event) => {
    if (event.target === svg || event.target === edgeLayer) {
      selectedNodeId = null;
      networkSearchMessage.textContent = "";
      renderInspectPanel(null);
      renderSvgGraph(graph);
    }
  });

  svg.append(edgeLayer, nodeLayer, labelLayer);
  graphContainer.appendChild(svg);
}

function scaleGraphCoordinate(
  value: number,
  min: number,
  span: number,
  padding: number,
  extent: number,
): number {
  return padding + ((value - min) / span) * (extent - padding * 2);
}

function cssEscapeAttributeValue(value: string): string {
  if (typeof CSS !== "undefined" && typeof CSS.escape === "function") {
    return CSS.escape(value);
  }
  return value.replaceAll("\\", "\\\\").replaceAll('"', '\\"');
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

function formatRecommendationWhyHtml(result: RecommendationResult): string {
  const explanation = explainRecommendation(result);
  if (explanation.kind === "none") {
    return `<div class="rec-why-line">Why: no direct contributing anime found.</div>`;
  }

  return [
    `<div class="rec-why-line rec-why-pos">${escapeHtml(explanation.positiveLine)}</div>`,
    `<div class="rec-why-line rec-why-neg">${escapeHtml(explanation.negativeLine)}</div>`,
  ].join("");
}

function renderModelBlendValue(): void {
  const modelPercent = Math.round(modelBlendWeight * 100);
  const graphPercent = 100 - modelPercent;
  recBlendValueEl.textContent = `${modelPercent}% model / ${graphPercent}% graph`;
}

function setBlendControlVisibility(): void {
  recBlendControl.hidden = recommendationMode !== "hybrid";
}

function applyTheme(theme: ThemeMode): void {
  document.documentElement.dataset.theme = theme;
  themeToggleBtn.dataset.mode = theme;
  themeToggleLabelEl.textContent = theme === "dark" ? "Theme: Dark" : "Theme: Light";
  themeToggleBtn.setAttribute(
    "aria-label",
    theme === "dark" ? "Switch to light theme" : "Switch to dark theme",
  );
}

function applyContrast(contrast: ContrastMode): void {
  document.documentElement.dataset.contrast = contrast;
  contrastToggleBtn.dataset.mode = contrast;
  contrastToggleLabelEl.textContent =
    contrast === "high" ? "Contrast: High" : "Contrast: Normal";
  contrastToggleBtn.setAttribute(
    "aria-label",
    contrast === "high" ? "Switch to normal contrast mode" : "Switch to high contrast mode",
  );
}

function renderContextualTips(): void {
  const showTips = !helpTipsDismissed;
  tipsRecommendationsEl.hidden = !showTips || activeView !== "recommendations";
  tipsNetworkEl.hidden = !showTips || activeView !== "network";
  tipsToggleBtn.setAttribute("aria-pressed", showTips ? "true" : "false");
  tipsToggleBtn.setAttribute(
    "aria-label",
    showTips ? "Hide help tips" : "Show help tips",
  );
  tipsToggleLabelEl.textContent = showTips ? "Hide Tips" : "Show Tips";
}

function prefersReducedMotion(): boolean {
  return reduceMotionMediaQuery.matches;
}

function onMediaQueryChange(
  mediaQuery: MediaQueryList,
  listener: () => void,
): void {
  try {
    mediaQuery.addEventListener("change", listener);
    return;
  } catch {
    const legacyMediaQuery = mediaQuery as MediaQueryList & {
      addListener?: (
        callback: (this: MediaQueryList, event: MediaQueryListEvent) => void,
      ) => void;
    };
    legacyMediaQuery.addListener?.(() => {
      listener();
    });
  }
}

function renderStorageWarnings(): void {
  storageStatusEl.textContent = persistence.getStorageWarnings().join(" ");
}

function persistRecommendationState(): boolean {
  cancelActiveUsernameImport();
  cancelBulkFileLoad();
  clearPendingHistoryImport();
  recommendationController?.abort();
  ++recommendationRunId;
  const saved = persistence.persistRecommendationState(buildCurrentRecommendationState());
  renderStorageWarnings();
  return saved;
}

function buildCurrentRecommendationState(): StoredRecommendationState {
  const preferences = selectedAnimeNodeIds
    .map((nodeId) => selectedAnimePreferences.get(nodeId))
    .filter((item): item is AnimePreference => item !== undefined);

  return {
    version: RECOMMENDATION_STORAGE_VERSION,
    mode: recommendationMode,
    preferences,
    modelBlendWeight: clampModelBlendWeight(modelBlendWeight),
    includeCandidates: [...includeCandidateNodeIds],
    excludeCandidates: [...excludeCandidateNodeIds],
    history: [...historyEntries],
  };
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
  const now = runtime.now().toISOString();
  const previous = savedProfiles.get(profileName);
  savedProfiles.set(profileName, {
    name: profileName,
    updatedAt: now,
    state,
  });
  if (!persistence.persistRecommendationProfiles(savedProfiles)) {
    if (previous) savedProfiles.set(profileName, previous);
    else savedProfiles.delete(profileName);
    renderStorageWarnings();
    recMessageEl.textContent = `Could not save profile "${profileName}".`;
    return;
  }
  renderStorageWarnings();
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

  applyRecommendationState({
    ...profile.state,
    modelBlendWeight: profile.state.modelBlendWeight ?? 0.5,
    includeCandidates: profile.state.includeCandidates ?? [],
    excludeCandidates: profile.state.excludeCandidates ?? [],
    history: profile.state.history ?? [],
  });
  const saved = persistRecommendationState();
  renderSelectedAnime();
  renderImportedHistory();
  renderIncludeCandidates();
  renderExcludeCandidates();
  void updateRecommendations();
  recMessageEl.textContent = saved
    ? `Loaded profile "${name}".`
    : `Loaded profile "${name}" for this session; browser storage rejected the change.`;
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

  const previous = savedProfiles.get(name);
  savedProfiles.delete(name);
  if (!persistence.persistRecommendationProfiles(savedProfiles)) {
    if (previous) savedProfiles.set(name, previous);
    renderStorageWarnings();
    recMessageEl.textContent = `Could not delete profile "${name}".`;
    return;
  }
  renderStorageWarnings();
  renderProfileOptions(savedProfiles);
  recMessageEl.textContent = `Deleted profile "${name}".`;
}

function applyRecommendationState(state: {
  mode: RecommendationMode;
  preferences: AnimePreference[];
  modelBlendWeight: number;
  includeCandidates: string[];
  excludeCandidates: string[];
  history?: HistoryEntry[];
}): void {
  selectedAnimeNodeIds.splice(0, selectedAnimeNodeIds.length);
  selectedAnimePreferences.clear();
  includeCandidateNodeIds.splice(0, includeCandidateNodeIds.length);
  excludeCandidateNodeIds.splice(0, excludeCandidateNodeIds.length);
  historyEntries.splice(0, historyEntries.length, ...(state.history ?? []));
  clearPendingHistoryImport();

  for (const entry of state.preferences) {
    selectedAnimeNodeIds.push(entry.nodeId);
    selectedAnimePreferences.set(entry.nodeId, entry);
  }
  for (const nodeId of state.includeCandidates) {
    includeCandidateNodeIds.push(nodeId);
  }
  for (const nodeId of state.excludeCandidates) {
    excludeCandidateNodeIds.push(nodeId);
  }

  recommendationMode = state.mode;
  modelBlendWeight = clampModelBlendWeight(state.modelBlendWeight);
  recMethodSelect.value = recommendationMode;
  recBlendInput.value = modelBlendWeight.toFixed(2);
  renderModelBlendValue();
  setBlendControlVisibility();
}

function valueRow(label: string, value: string): string {
  return `<div class="inspect-value-row"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`;
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
  graphDataValue: LoadedGraphData,
  input: HTMLInputElement,
): number {
  let sumAbsWeight = 0;
  let count = 0;

  if (isCompactGraphData(graphDataValue)) {
    for (const edge of graphDataValue.aa) {
      const weight = edge[2];
      if (!Number.isFinite(weight)) {
        continue;
      }
      sumAbsWeight += Math.abs(weight);
      count += 1;
    }
  } else {
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
    // Round down so a sparse selected graph still shows at least its
    // strongest edge at the initial threshold.
    value = min + Math.floor((value - min) / step) * step;
  }

  return Number(value.toFixed(2));
}
