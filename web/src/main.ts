import Graph from "graphology";
import { ArtifactValidationError, isCompactGraphData } from "./artifacts";
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
  RecommendationResult,
  SeasonalAnimeItem,
} from "./domain";
import {
  MAX_MODEL_BLEND_WEIGHT,
  MIN_MODEL_BLEND_WEIGHT,
  buildCatalogCoverageRecommendations,
  buildCommunityQualityExploration,
  buildGenreOverlapExploration,
  buildGraphRecommendationsForPreferences,
  buildModelRecommendationsForPreferences,
  buildRecommendationIndex,
  buildRecommendationIndexFromCompact,
  buildSamplePopularityExploration,
  clampModelBlendWeight,
  createCandidateEligibilityPolicy,
  explainRecommendation,
  formatScoreEquation,
  formatWeight,
  hasActiveRecommendationFilters,
  normalizeTitle,
  rankEligibleCandidates,
  summarizeCandidateMetadataCoverage,
} from "./recommendations";
import type { CandidateEligibilityPolicy, CandidateMetadataCoverage, EligibilityRankingMode, GenreOverlapRecommendation, RecommendationExplanation, RecommendationFilters } from "./recommendations";
import { ProviderUnavailableError, createProviderAdapter } from "./providers";
import { selectFranchiseDiverseRecommendations } from "./franchise-diversity";
import type { FranchiseSelection } from "./franchise-diversity";
import { ArtifactLoadError, createArtifactLoader } from "./artifact-loader";
import { projectCatalogMetadata } from "./catalog-metadata";
import { measureLocal, measureLocalAsync, recordLocalDuration } from "./local-performance";
import { describeGraphScope } from "./graph-scope";
import { selectAnimeNeighborhood } from "./network-neighborhood";
import { appVersionLabel, dataVersionLabel, diagnosticIssue, modelVersionLabel } from "./diagnostics";
import type { DiagnosticCode } from "./diagnostics";
import {
  MAX_MAL_XML_IMPORT_BYTES,
  MAX_TEXT_IMPORT_BYTES,
  HistoryImportValidationError,
  mergeHistory,
  parseMalXmlHistory,
  parseTextHistory,
  previewHistory,
  resolveHistoryAnime,
  seenHistoryAnimeIds,
} from "./import-history";
import type { HistoryEntry, ImportMode, ParsedHistory } from "./import-history";
import { MAX_IMPORTANCE, MIN_IMPORTANCE, clampImportance, manualPreference, preferenceFromHistory } from "./preferences";
import type { AnimePreference, PreferenceSentiment } from "./preferences";
import {
  COMMAND_HISTORY_LIMIT,
  COMMAND_PINNED_LIMIT,
  MAX_PROFILE_BACKUP_BYTES,
  RECOMMENDATION_STORAGE_VERSION,
  createPersistenceAdapter,
  emptyRecommendationState,
} from "./persistence";
import type {
  ContrastMode,
  ProfileBackupDocument,
  ProfileBackupImportMode,
  RecommendationMode,
  RecommendationProfileRecord,
  StoredRecommendationState,
  ThemeMode,
} from "./persistence";
import { createBrowserRuntime, isAbortError, throwIfAborted } from "./runtime";
import { safeExternalImageUrl } from "./safe-url";
import { searchAnimeTitles } from "./title-search";
import type { TitleSearchResult } from "./title-search";
import { mergeWatchlistFeedback, validateWatchlist, watchedWatchlistAnimeIds } from "./watchlist";
import type { WatchlistEntry, WatchlistStatus } from "./watchlist";
import "./style.css";

const runtime = createBrowserRuntime();
const providerAdapter = createProviderAdapter(runtime);
const demoMode = import.meta.env.VITE_DEMO_MODE === "true";
const artifactLoader = createArtifactLoader(runtime, demoMode, import.meta.env.BASE_URL);
const persistence = createPersistenceAdapter(runtime, demoMode ? "wasiw.demo" : "wasiw");

type AppView = "recommendations" | "network";
type UsernameImportProvider = "anilist" | "mal";
type AsyncUiState = "idle" | "loading" | "partial" | "ready" | "empty" | "unavailable" | "failed" | "stale" | "demo";
type DiscoveryView = "auto" | "popularity" | "quality" | "related";
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
const BATCHED_SVG_EDGE_THRESHOLD = 500;
const BATCHED_SVG_NODE_THRESHOLD = 500;
const LARGE_RECOMMENDATION_YIELD_CANDIDATES = 500;
const CHUNKED_NETWORK_EDGE_THRESHOLD = 500;
const NETWORK_BUILD_BATCH_SIZE = 1000;
const NETWORK_SVG_BATCH_SIZE = 1000;
const NEIGHBORHOOD_MAX_NODES = 25;
const NEIGHBORHOOD_MAX_EDGES = 24;
const GRAPH_VIEW_WIDTH = 1200;
const GRAPH_VIEW_HEIGHT = 900;
const NETWORK_NODE_LIST_PAGE_SIZE = 15;
const INSPECT_MAX_ITEMS = 250;
const MAX_RECOMMENDATIONS = 40;
const IMPORTANCE_STEP = 0.1;
const MODEL_BLEND_WEIGHT_STEP = 0.05;
const METADATA_PREFETCH_LIMIT = 100;
const METADATA_PREFETCH_WITH_FILTER_LIMIT = 30;
const METADATA_PREFETCH_CONCURRENCY = 3;
const MAX_SPARSE_CONTENT_SEEDS = 3;
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
const startupStatusEl = document.querySelector<HTMLElement>("#startup-status");

function revealApp(): void {
  app!.hidden = false;
  startupStatusEl?.remove();
}

app.innerHTML = `
  <div class="app-shell">
    <header class="topbar">
      <div class="brand">
        <p class="eyebrow">ANIME DISCOVERY</p>
        <h1>What Anime Should I Watch</h1>
        ${demoMode ? '<p class="demo-banner" role="status">SYNTHETIC DEMO DATA · Invented titles and ratings · No live metadata requests</p>' : ""}
        <p>Browse ideas or shape your picks with favorites.</p>
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
        <input id="command-input" type="text" autocomplete="off" aria-label="Search quick actions" placeholder="Type an action (e.g. network, theme, import)" />
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
            <li>Use Discovery view to compare sampled popularity, known community scores, and shared genres.</li>
            <li>Use Include Only to limit scored candidates; exclusions and watched titles always win.</li>
            <li>Keyboard: <strong>Alt+/</strong> focuses the main anime input instantly.</li>
          </ul>
        </section>

        <section class="panel intro-panel" aria-labelledby="intro-title">
          <div class="intro-copy">
            <p class="intro-eyebrow">GET STARTED</p>
            <h2 id="intro-title">How would you like to find your next anime?</h2>
            <p class="muted">Choose one path to start. Saved picks stay in this browser; local files are previewed before import.</p>
          </div>
          <div class="intro-paths">
            <button id="quickstart-favorites" class="intro-path" type="button" aria-controls="add-anime-form">
              <strong>Add favorites</strong>
              <span>Add one or two titles you liked to shape your picks.</span>
            </button>
            <button id="quickstart-import" class="intro-path" type="button" aria-controls="import-profiles">
              <strong>Import a list</strong>
              <span>Choose a local file or paste text, then review before applying.</span>
            </button>
            <button id="quickstart-browse" class="intro-path" type="button" aria-controls="rec-results">
              <strong>Browse without a list</strong>
              <span>Explore community scores, or sampled counts when available, without adding history.</span>
            </button>
          </div>
          <div class="intro-actions">
            <button id="quickstart-seasonal" type="button" class="ghost-btn">${demoMode ? "See demo ideas" : "See seasonal ideas"}</button>
            <button id="quickstart-network" type="button" class="ghost-btn">Open network explorer</button>
          </div>
        </section>

        <div class="recommend-layout">
          <section class="card">
            <h2>Your starting point</h2>
            <p class="muted">${demoMode ? "Try the invented catalog. Mark a favorite or browse without adding anything." : "Mark a favorite or browse the loaded catalog without adding anything."}</p>

            <label class="rec-engine-control" for="discovery-view">
              <span>Browse by</span>
              <select id="discovery-view">
                <option value="auto">Automatic: explore until a ranking has a preference signal</option>
                <option value="popularity">Popularity proxy: ratings in loaded recommendation graph</option>
                <option value="quality">Community score among checked titles</option>
                <option value="related">Shared genres with Liked titles</option>
              </select>
            </label>
            <p id="rec-engine-status" class="rec-engine-status">Using graph recommendations.</p>

            <h3 id="manual-entry-heading" class="manual-entry-heading">Add a title you know</h3>
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
            <p class="muted" id="preference-guidance">Choose Liked for a favorite. Seen keeps a title out of your picks without treating it as a like. You can add more later.</p>
            <p id="preference-migration-notice" class="muted" role="status" hidden></p>

            <details id="advanced-recommendation-settings" class="accordion advanced-settings">
              <summary>Advanced recommendation settings</summary>
              <p class="muted">Compare ranking engines and adjust how related titles appear. Your manual choices take precedence over later imports.</p>
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
              <label class="rec-diversity-control" for="allow-related-titles">
                <input id="allow-related-titles" type="checkbox" />
                <span>Allow related titles and known sequels</span>
              </label>
              <p class="muted rec-diversity-help">By default, show one title per known or title-suggested series and withhold known sequels whose immediate prequel is not in watched history. Relationship checks are incomplete; switch this on to see the original eligible order with warnings.</p>
              <p class="muted rec-diversity-help">Liked titles seed graph suggestions. The model also uses dislikes as negative evidence. Importance is your emphasis; confidence records how certain an imported score is.</p>
            </details>

            <p id="rec-message" class="rec-message" role="status" aria-live="polite"></p>
            <p id="storage-status" class="storage-status" role="alert" aria-live="assertive"></p>

            <div class="selected-head">
              <h3>Watched &amp; Preferences <span id="watched-count" class="count-pill">0</span></h3>
              <button id="clear-watched" type="button" class="ghost-btn">Clear List</button>
            </div>
            <div id="selected-anime" class="selected-anime"></div>

            <section class="local-watchlist" aria-labelledby="watchlist-title">
              <h3 id="watchlist-title">My Watchlist <span id="watchlist-count" class="count-pill">0</span></h3>
              <p class="muted">Save a shortlist, set a watch status, and rate titles from 1–10. Status alone never becomes a preference or training permission. An explicit rating can refine suggestions in this browser; a manually chosen preference takes precedence. Planned titles do not supply preference evidence.</p>
              <form id="watchlist-form" class="add-form add-form-compact">
                <input id="watchlist-input" type="text" list="anime-options" autocomplete="off" aria-label="Anime to add to watchlist" placeholder="Title, alias, or anime ID" />
                <button type="submit">Plan to Watch</button>
              </form>
              <p id="watchlist-status" class="muted" role="status" aria-live="polite"></p>
              <ul id="watchlist-list" class="watchlist-list"></ul>
            </section>

            <details id="import-profiles" class="accordion">
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
              <section class="profiles profile-backup">
                <h3>Backup &amp; Recovery</h3>
                <p class="muted">Download your current preferences, imported history, local watchlist, candidate overrides, and named profiles as a versioned JSON file. It stays on your device; store it privately because it contains your watch history. A backup does not depend on the current catalog or model release.</p>
                <div class="profile-backup-actions">
                  <button id="profile-export-btn" type="button">Download Profile Backup</button>
                  <label for="profile-backup-file">Import a local .json backup (up to 8 MiB)</label>
                  <input id="profile-backup-file" type="file" accept=".json,application/json" />
                  <button id="profile-reset-preview-btn" type="button" class="ghost-btn">Preview Local Reset</button>
                  <button id="profile-repair-btn" type="button" class="ghost-btn">Recover Loaded Copy</button>
                </div>
                <p id="profile-backup-status" class="muted" role="status" aria-live="polite"></p>
                <section id="profile-backup-preview" class="import-preview" hidden>
                  <h3 id="profile-backup-preview-title">Backup Preview</h3>
                  <p id="profile-backup-summary"></p>
                  <p id="profile-backup-unmapped" class="muted"></p>
                  <label id="profile-backup-mode-row" for="profile-backup-mode">Apply as
                    <select id="profile-backup-mode">
                      <option value="merge">Merge; keep local values on conflicts</option>
                      <option value="replace">Replace current state and named profiles</option>
                    </select>
                  </label>
                  <div class="profile-backup-actions">
                    <button id="profile-backup-apply" type="button" class="primary-btn">Apply Backup</button>
                    <button id="profile-backup-cancel" type="button" class="ghost-btn">Cancel</button>
                  </div>
                </section>
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
                  <span>Minimum ${demoMode ? "demo" : "community"} score</span>
                  <input id="filter-min-score" type="range" min="0" max="10" step="${METADATA_SCORE_STEP}" value="0" />
                  <output id="filter-min-score-value">Any</output>
                </label>
                <div class="rec-filter-actions">
                  <button id="clear-rec-filters" type="button" class="ghost-btn">Clear Filters</button>
                  <span id="metadata-status" class="metadata-status" role="status" aria-live="polite" data-state="idle">Metadata: idle</span>
                </div>
                <div id="filter-metadata-controls" class="metadata-coverage-controls" hidden>
                  <p id="filter-metadata-note" class="muted" role="status"></p>
                  <button id="filter-metadata-more" type="button" class="ghost-btn">Check more candidate metadata</button>
                </div>
              </section>
            </details>
            <p id="rec-summary" class="muted" role="status" aria-live="polite">Add at least one anime to start.</p>
            <div id="discovery-metadata-controls" class="discovery-metadata-controls" hidden>
              <p id="discovery-metadata-note" class="muted" role="status"></p>
              <button id="discovery-load-metadata" type="button" class="ghost-btn">Check 12 more catalog titles for quality and genres</button>
            </div>
            <p id="rec-action-status" class="rec-action-status" role="status" aria-live="polite" tabindex="-1"></p>
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
            <li>Search for an anime to focus on its strongest retained pair evidence; Reset returns to the explorer sample.</li>
            <li>Raise minimum absolute edge weight to highlight stronger pair-preference signals.</li>
            <li>Click a node or use the node list to inspect it, then focus that anime if useful.</li>
            <li>Keyboard: <strong>Alt+/</strong> jumps to graph search from anywhere.</li>
          </ul>
        </section>

        <section class="network-scope" aria-label="Graph versions and coverage">
          <p id="network-versions">Checking graph and model versions...</p>
          <p id="network-selection">Checking pair-selection coverage...</p>
          <p id="network-explorer-sample">Checking explorer sample...</p>
          <p id="network-drawing-limits"></p>
          <p id="network-scope-caveat"></p>
        </section>

        <button id="network-mobile-toggle" type="button" class="network-mobile-toggle" hidden aria-expanded="false" aria-controls="network-panel">
          Show Controls
        </button>
        <div id="network-layout" class="network-layout">
          <aside id="network-panel" class="panel">
            <h2>Network Explorer</h2>
            <p class="muted">Inspect the loaded pair evidence and filter visible edges.</p>

            <section class="network-exploration" aria-label="Local graph exploration">
              <p id="network-mode-status" role="status" aria-live="polite">Explorer sample overview.</p>
              <p class="muted">Focused view uses at most 25 anime nodes and 24 signed pair edges from the loaded recommendation graph. Strongest means absolute v1 pair preference, with support breaking ties; it is not item similarity.</p>
              <div class="network-exploration-actions">
                <button id="focus-neighborhood" type="button" class="ghost-btn" disabled>Focus selected anime</button>
                <button id="reset-neighborhood" type="button" class="ghost-btn">Reset overview</button>
              </div>
            </section>

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
                <span id="toggle-users-label">Show user nodes + user-anime edges</span>
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

            <details id="network-node-list" class="network-node-list">
              <summary>Visible nodes as a list</summary>
              <label for="network-node-filter">Filter visible nodes by title or ID</label>
              <input id="network-node-filter" type="search" autocomplete="off" />
              <p id="network-node-list-status" class="muted" role="status" aria-live="polite">Open the network to browse visible nodes.</p>
              <ul id="network-node-list-results"></ul>
              <div class="network-node-list-pages">
                <button id="network-node-prev" type="button" class="ghost-btn">Previous</button>
                <button id="network-node-next" type="button" class="ghost-btn">Next</button>
              </div>
            </details>

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

          <section id="graph-shell" class="graph-shell" tabindex="0" aria-label="Network plot. Arrow keys pan, plus and minus zoom, and zero resets the view.">
            <div class="graph-viewport-controls" role="group" aria-label="Network viewport controls">
              <button id="graph-zoom-in" type="button" aria-label="Zoom in network">+</button>
              <button id="graph-zoom-out" type="button" aria-label="Zoom out network" disabled>−</button>
              <button id="graph-zoom-reset" type="button" aria-label="Fit network view">Fit</button>
            </div>
            <div id="graph-loading" class="graph-loading" hidden aria-hidden="true">
              <div class="graph-spinner"></div>
              <p id="graph-loading-message">Rendering network...</p>
            </div>
            <div id="graph"></div>
          </section>
        </div>
      </section>
      <details id="local-diagnostics" class="panel local-diagnostics">
        <summary>Local diagnostics</summary>
        <p class="muted">This panel shows only public release identifiers and fixed error codes. It does not include or send usernames or watch history.</p>
        <dl class="diagnostic-versions">
          <div><dt>App</dt><dd id="diagnostic-app">Checking build...</dd></div>
          <div><dt>Data</dt><dd id="diagnostic-data">Checking data...</dd></div>
          <div><dt>Model</dt><dd id="diagnostic-model">Not checked</dd></div>
        </dl>
        <p id="diagnostic-code" role="status" aria-live="polite">No error code recorded.</p>
        <p id="diagnostic-action" class="muted"></p>
      </details>
    </main>
    <dialog id="title-search-dialog" class="title-search-dialog" aria-labelledby="title-search-heading" aria-describedby="title-search-summary">
      <div class="title-search-head">
        <h2 id="title-search-heading">Choose a catalog title</h2>
        <button id="title-search-close" class="ghost-btn" type="button">Cancel</button>
      </div>
      <p id="title-search-summary" class="muted"></p>
      <ul id="title-search-results" class="title-search-results"></ul>
    </dialog>
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
const appTopbarEl = mustElement<HTMLElement>(".topbar");
const appMainEl = mustElement<HTMLElement>("main");
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
const quickstartFavoritesBtn = mustElement<HTMLButtonElement>("#quickstart-favorites");
const quickstartImportBtn = mustElement<HTMLButtonElement>("#quickstart-import");
const quickstartBrowseBtn = mustElement<HTMLButtonElement>("#quickstart-browse");
let favoriteQuickstartActive = false;

const addAnimeForm = mustElement<HTMLFormElement>("#add-anime-form");
const animeInput = mustElement<HTMLInputElement>("#anime-input");
const addPreferenceSelect = mustElement<HTMLSelectElement>("#add-preference");
const animeOptions = mustElement<HTMLDataListElement>("#anime-options");
const titleSearchDialog = mustElement<HTMLDialogElement>("#title-search-dialog");
const titleSearchSummary = mustElement<HTMLParagraphElement>("#title-search-summary");
const titleSearchResults = mustElement<HTMLUListElement>("#title-search-results");
const titleSearchClose = mustElement<HTMLButtonElement>("#title-search-close");
const recMethodSelect = mustElement<HTMLSelectElement>("#rec-method");
const discoveryViewSelect = mustElement<HTMLSelectElement>("#discovery-view");
const allowRelatedInput = mustElement<HTMLInputElement>("#allow-related-titles");
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
const watchlistForm = mustElement<HTMLFormElement>("#watchlist-form");
const watchlistInput = mustElement<HTMLInputElement>("#watchlist-input");
const watchlistCountEl = mustElement<HTMLSpanElement>("#watchlist-count");
const watchlistListEl = mustElement<HTMLUListElement>("#watchlist-list");
const watchlistStatusEl = mustElement<HTMLParagraphElement>("#watchlist-status");
const bulkImportForm = mustElement<HTMLFormElement>("#bulk-import-form");
const importProfilesDetails = mustElement<HTMLDetailsElement>("#import-profiles");
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
const profileExportBtn = mustElement<HTMLButtonElement>("#profile-export-btn");
const profileBackupFile = mustElement<HTMLInputElement>("#profile-backup-file");
const profileResetPreviewBtn = mustElement<HTMLButtonElement>("#profile-reset-preview-btn");
const profileRepairBtn = mustElement<HTMLButtonElement>("#profile-repair-btn");
const profileBackupStatusEl = mustElement<HTMLParagraphElement>("#profile-backup-status");
const profileBackupPreviewEl = mustElement<HTMLElement>("#profile-backup-preview");
const profileBackupPreviewTitleEl = mustElement<HTMLElement>("#profile-backup-preview-title");
const profileBackupSummaryEl = mustElement<HTMLParagraphElement>("#profile-backup-summary");
const profileBackupUnmappedEl = mustElement<HTMLParagraphElement>("#profile-backup-unmapped");
const profileBackupModeRowEl = mustElement<HTMLLabelElement>("#profile-backup-mode-row");
const profileBackupModeEl = mustElement<HTMLSelectElement>("#profile-backup-mode");
const profileBackupApplyBtn = mustElement<HTMLButtonElement>("#profile-backup-apply");
const profileBackupCancelBtn = mustElement<HTMLButtonElement>("#profile-backup-cancel");
const addIncludeForm = mustElement<HTMLFormElement>("#add-include-form");
const includeInput = mustElement<HTMLInputElement>("#include-input");
const includeAnimeEl = mustElement<HTMLDivElement>("#include-anime");
const clearIncludeBtn = mustElement<HTMLButtonElement>("#clear-include");
const addExcludeForm = mustElement<HTMLFormElement>("#add-exclude-form");
const excludeInput = mustElement<HTMLInputElement>("#exclude-input");
const excludeAnimeEl = mustElement<HTMLDivElement>("#exclude-anime");
const clearExcludeBtn = mustElement<HTMLButtonElement>("#clear-exclude");
const recSummaryEl = mustElement<HTMLParagraphElement>("#rec-summary");
const recActionStatusEl = mustElement<HTMLParagraphElement>("#rec-action-status");
const discoveryMetadataControlsEl = mustElement<HTMLDivElement>("#discovery-metadata-controls");
const discoveryMetadataNoteEl = mustElement<HTMLParagraphElement>("#discovery-metadata-note");
const discoveryLoadMetadataBtn = mustElement<HTMLButtonElement>("#discovery-load-metadata");
const recResultsEl = mustElement<HTMLOListElement>("#rec-results");
const filterGenreSelect = mustElement<HTMLSelectElement>("#filter-genre");
const filterYearMinInput = mustElement<HTMLInputElement>("#filter-year-min");
const filterYearMaxInput = mustElement<HTMLInputElement>("#filter-year-max");
const filterMinScoreInput = mustElement<HTMLInputElement>("#filter-min-score");
const filterMinScoreValue = mustElement<HTMLOutputElement>("#filter-min-score-value");
const clearRecFiltersBtn = mustElement<HTMLButtonElement>("#clear-rec-filters");
const metadataStatusEl = mustElement<HTMLSpanElement>("#metadata-status");
const filterMetadataControlsEl = mustElement<HTMLDivElement>("#filter-metadata-controls");
const filterMetadataNoteEl = mustElement<HTMLParagraphElement>("#filter-metadata-note");
const filterMetadataMoreBtn = mustElement<HTMLButtonElement>("#filter-metadata-more");
const seasonalStatusEl = mustElement<HTMLParagraphElement>("#seasonal-status");
const seasonalListEl = mustElement<HTMLUListElement>("#seasonal-list");
const refreshSeasonalBtn = mustElement<HTMLButtonElement>("#refresh-seasonal");

const statsEl = mustElement<HTMLDivElement>("#stats");
const networkMobileToggleBtn = mustElement<HTMLButtonElement>("#network-mobile-toggle");
const networkLayoutEl = mustElement<HTMLDivElement>("#network-layout");
const networkPanelEl = mustElement<HTMLElement>("#network-panel");
const networkRenderStatusEl = mustElement<HTMLParagraphElement>("#network-render-status");
const networkModeStatusEl = mustElement<HTMLParagraphElement>("#network-mode-status");
const focusNeighborhoodBtn = mustElement<HTMLButtonElement>("#focus-neighborhood");
const resetNeighborhoodBtn = mustElement<HTMLButtonElement>("#reset-neighborhood");
const networkVersionsEl = mustElement<HTMLParagraphElement>("#network-versions");
const networkSelectionEl = mustElement<HTMLParagraphElement>("#network-selection");
const networkExplorerSampleEl = mustElement<HTMLParagraphElement>("#network-explorer-sample");
const networkDrawingLimitsEl = mustElement<HTMLParagraphElement>("#network-drawing-limits");
const networkScopeCaveatEl = mustElement<HTMLParagraphElement>("#network-scope-caveat");
const networkNodeFilterEl = mustElement<HTMLInputElement>("#network-node-filter");
const networkNodeListStatusEl = mustElement<HTMLParagraphElement>("#network-node-list-status");
const networkNodeListResultsEl = mustElement<HTMLUListElement>("#network-node-list-results");
const networkNodePrevBtn = mustElement<HTMLButtonElement>("#network-node-prev");
const networkNodeNextBtn = mustElement<HTMLButtonElement>("#network-node-next");
const graphShell = mustElement<HTMLElement>("#graph-shell");
const graphZoomInBtn = mustElement<HTMLButtonElement>("#graph-zoom-in");
const graphZoomOutBtn = mustElement<HTMLButtonElement>("#graph-zoom-out");
const graphZoomResetBtn = mustElement<HTMLButtonElement>("#graph-zoom-reset");
const graphLoadingEl = mustElement<HTMLDivElement>("#graph-loading");
const graphLoadingMessageEl = mustElement<HTMLParagraphElement>("#graph-loading-message");
const graphContainer = mustElement<HTMLDivElement>("#graph");
const minWeightInput = mustElement<HTMLInputElement>("#min-weight");
const minWeightValue = mustElement<HTMLOutputElement>("#min-weight-value");
const toggleAnimeEdges = mustElement<HTMLInputElement>("#toggle-anime-edges");
const toggleUsers = mustElement<HTMLInputElement>("#toggle-users");
const toggleUsersLabelEl = mustElement<HTMLSpanElement>("#toggle-users-label");
const networkSearchForm = mustElement<HTMLFormElement>("#network-search-form");
const networkSearchInput = mustElement<HTMLInputElement>("#network-search-input");
const networkNodeOptions = mustElement<HTMLDataListElement>("#network-node-options");
const diagnosticAppEl = mustElement<HTMLElement>("#diagnostic-app");
const diagnosticDataEl = mustElement<HTMLElement>("#diagnostic-data");
const diagnosticModelEl = mustElement<HTMLElement>("#diagnostic-model");
const diagnosticCodeEl = mustElement<HTMLParagraphElement>("#diagnostic-code");
const diagnosticActionEl = mustElement<HTMLParagraphElement>("#diagnostic-action");
const networkSearchMessage = mustElement<HTMLParagraphElement>("#network-search-message");
const inspectEmptyEl = mustElement<HTMLParagraphElement>("#inspect-empty");
const inspectContentEl = mustElement<HTMLDivElement>("#inspect-content");
const inspectMetaEl = mustElement<HTMLDivElement>("#inspect-meta");
const inspectCountEl = mustElement<HTMLDivElement>("#inspect-count");
const inspectValuesEl = mustElement<HTMLDivElement>("#inspect-values");
const inspectListEl = mustElement<HTMLUListElement>("#inspect-list");
const clearSelectionBtn = mustElement<HTMLButtonElement>("#clear-selection");

diagnosticAppEl.textContent = appVersionLabel(__WASIW_APP_VERSION__, __WASIW_SOURCE_REVISION__);
diagnosticModelEl.textContent = modelVersionLabel(null, null, "unchecked", demoMode);

let selectedNodeId: string | null = null;
let neighborhoodFocusId: string | null = null;
let currentGraph: Graph | null = null;
let graphViewport = { x: 0, y: 0, width: GRAPH_VIEW_WIDTH, height: GRAPH_VIEW_HEIGHT };
let visibleGraphNodes: { id: string; label: string; nodeType: string; searchText: string;
  evidence?: { weight: number; support?: number } }[] = [];
let networkNodeListPage = 0;
let explorerGraphData: LoadedGraphData | null = null;
let explorerGraphDataPromise: Promise<LoadedGraphData> | null = null;
let activeView: AppView = "recommendations";
let recommendationMode: RecommendationMode = "graph";
let discoveryView: DiscoveryView = "auto";
let discoveryMetadataBatchRequested = false;
let modelBlendWeight = 0.5;
let allowRelatedTitles = false;
let recommendationRunId = 0;
let recommendationController: AbortController | null = null;
let activeUsernameImport: { controller: AbortController; provider: UsernameImportProvider } | null = null;
let bulkFileLoadId = 0;
let activeBulkFileLoadId: number | null = null;
let pendingHistoryImport: { parsed: ParsedHistory; origin: "local" | "username" } | null = null;
let profileBackupFileLoadId = 0;
let activeProfileBackupFileLoadId: number | null = null;
let pendingProfileBackup: {
  kind: "import" | "reset";
  document?: ProfileBackupDocument;
  storageRevision: string;
  memoryRevision: string;
} | null = null;
let graphRenderRunId = 0;
let graphRenderController: AbortController | null = null;
let svgRenderRunId = 0;
let overviewGraphCache: {
  source: LoadedGraphData;
  minAbsoluteWeight: number;
  showAnimeAnimeEdges: boolean;
  showUsers: boolean;
  graph: Graph;
  svg: SVGSVGElement | null;
  totalEligibleEdgeCount: number;
  edgeLimitHit: boolean;
} | null = null;
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
let commandPaletteReturnFocus: HTMLElement | null = null;
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
const bundledMetadata = await loadRequiredArtifact(artifactLoader.fetchCatalogMetadata(graphData));
const activeReleaseManifest = await artifactLoader.getActiveReleaseManifest();
diagnosticDataEl.textContent = dataVersionLabel(activeReleaseManifest,
  isCompactGraphData(graphData) ? graphData.format : "legacy-graph", demoMode);
diagnosticModelEl.textContent = modelVersionLabel(activeReleaseManifest, null, "unchecked", demoMode);
const samplePopularityAvailable = !isCompactGraphData(graphData) ||
  graphData.format !== "graph-compact-v3";
const graphNodes = getGraphNodes(graphData);
const recommendationIndex = measureLocal("wasiw:index:graph", () =>
  isCompactGraphData(graphData)
    ? buildRecommendationIndexFromCompact(graphData)
    : buildRecommendationIndex(graphData));
if (bundledMetadata) {
  for (const item of bundledMetadata.anime) {
    animeMetadataCache.set(item.animeId, projectCatalogMetadata(item));
  }
  for (const anime of recommendationIndex.animeList) {
    if (!animeMetadataCache.has(anime.animeId)) animeMetadataUnavailable.add(anime.animeId);
  }
}
const selectedAnimeNodeIds: string[] = [];
const selectedAnimePreferences = new Map<string, AnimePreference>();
const historyEntries: HistoryEntry[] = [];
const watchlistEntries: WatchlistEntry[] = [];
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
watchlistEntries.push(...persistedState.watchlist);
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
allowRelatedTitles = persistedState.allowRelatedTitles;
allowRelatedInput.checked = allowRelatedTitles;
recMethodSelect.value = recommendationMode;
recBlendInput.value = modelBlendWeight.toFixed(2);
renderModelBlendValue();
setBlendControlVisibility();

populateAnimeOptions(recommendationIndex.animeList, animeMetadataCache, animeOptions);
populateNetworkNodeOptions(graphNodes, networkNodeOptions);
renderSelectedAnime();
renderImportedHistory();
renderWatchlist();
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

quickstartFavoritesBtn.addEventListener("click", () => {
  favoriteQuickstartActive = true;
  addPreferenceSelect.value = "liked";
  animeInput.scrollIntoView({ block: "center", behavior: prefersReducedMotion() ? "auto" : "smooth" });
  animeInput.focus({ preventScroll: true });
  recMessageEl.textContent = "Enter a favorite title, then Add. Liked is selected.";
});

quickstartImportBtn.addEventListener("click", () => {
  favoriteQuickstartActive = false;
  addPreferenceSelect.value = "seen";
  importProfilesDetails.open = true;
  bulkImportInput.scrollIntoView({ block: "center", behavior: prefersReducedMotion() ? "auto" : "smooth" });
  bulkImportInput.focus({ preventScroll: true });
});

quickstartBrowseBtn.addEventListener("click", () => {
  favoriteQuickstartActive = false;
  addPreferenceSelect.value = "seen";
  discoveryViewSelect.value = samplePopularityAvailable ? "popularity" : "quality";
  discoveryViewSelect.dispatchEvent(new Event("change", { bubbles: true }));
  recResultsEl.scrollIntoView({ block: "start", behavior: prefersReducedMotion() ? "auto" : "smooth" });
});

addPreferenceSelect.addEventListener("change", () => {
  favoriteQuickstartActive = false;
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
    runtime.schedule(() => {
      if (activeView !== "network" || !networkCompactMediaQuery.matches ||
          networkControlsHiddenOnMobile) return;
      networkPanelEl.scrollIntoView({
        block: "start",
        behavior: prefersReducedMotion() ? "auto" : "smooth",
      });
    }, 0);
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

titleSearchClose.addEventListener("click", () => titleSearchDialog.close());

watchlistForm.addEventListener("submit", (event) => {
  event.preventDefault();
  addWatchlistFromInput();
});

watchlistListEl.addEventListener("change", (event) => {
  const target = event.target;
  if (!(target instanceof HTMLSelectElement)) return;
  const animeId = Number(target.dataset.watchlistStatusId ?? target.dataset.watchlistRatingId);
  if (!Number.isSafeInteger(animeId) || animeId <= 0) return;
  const existing = watchlistEntries.find((item) => item.animeId === animeId);
  if (!existing) return;
  if (target.dataset.watchlistStatusId !== undefined) {
    const status = target.value as WatchlistStatus;
    if (!["plan_to_watch", "watching", "completed", "on_hold", "dropped"].includes(status)) return;
    changeWatchlistEntry({ ...existing, status });
  } else {
    const rating = target.value === "" ? null : Number(target.value);
    if (rating !== null && (!Number.isInteger(rating) || rating < 1 || rating > 10)) return;
    changeWatchlistEntry({ ...existing, rating });
  }
});

commandPaletteEl.addEventListener("keydown", (event) => {
  if (event.key !== "Tab") return;
  const focusable = [...commandPaletteEl.querySelectorAll<HTMLElement>(
    "button:not(:disabled), input:not(:disabled)",
  )].filter((element) => element.getClientRects().length > 0);
  if (focusable.length === 0) return;
  const first = focusable[0];
  const last = focusable[focusable.length - 1];
  if (event.shiftKey && (document.activeElement === first ||
      !commandPaletteEl.contains(document.activeElement))) {
    event.preventDefault();
    last.focus();
  } else if (!event.shiftKey && (document.activeElement === last ||
      !commandPaletteEl.contains(document.activeElement))) {
    event.preventDefault();
    first.focus();
  }
});

watchlistListEl.addEventListener("click", (event) => {
  const target = event.target;
  if (!(target instanceof HTMLElement)) return;
  const button = target.closest<HTMLButtonElement>("button[data-watchlist-remove-id]");
  if (!button) return;
  const animeId = Number(button.dataset.watchlistRemoveId);
  if (!Number.isSafeInteger(animeId) || animeId <= 0) return;
  applyWatchlistChange(watchlistEntries.filter((item) => item.animeId !== animeId), "Removed from your watchlist.");
});

recResultsEl.addEventListener("click", (event) => {
  const target = event.target;
  if (!(target instanceof HTMLElement)) return;
  const button = target.closest<HTMLButtonElement>("button[data-card-action]");
  if (!button) return;
  const animeId = Number(button.dataset.animeId);
  if (!Number.isSafeInteger(animeId) || animeId <= 0) return;
  const anime = recommendationIndex.animeByAnimeId.get(animeId);
  if (!anime) return;
  if (button.dataset.cardAction === "save") {
    addToWatchlist(anime);
    recActionStatusEl.textContent = watchlistStatusEl.textContent;
    recActionStatusEl.focus({ preventScroll: true });
  } else if (button.dataset.cardAction === "seen") {
    markResultSeen(anime);
  } else if (button.dataset.cardAction === "hide") {
    hideResult(anime);
  } else if (button.dataset.cardAction === "retry-metadata") {
    void retryResultMetadata(anime, button);
  }
});

recResultsEl.addEventListener("error", (event) => {
  const image = event.target;
  if (!(image instanceof HTMLImageElement) || !image.classList.contains("rec-cover")) return;
  const placeholder = document.createElement("div");
  placeholder.className = "rec-cover rec-cover-placeholder rec-cover-error";
  placeholder.setAttribute("role", "img");
  const title = image.closest(".rec-item")?.querySelector(".rec-title")?.textContent ?? "this title";
  placeholder.setAttribute("aria-label", `Cover could not load for ${title}`);
  placeholder.textContent = "Cover could not load";
  image.replaceWith(placeholder);
}, true);

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

profileExportBtn.addEventListener("click", downloadProfileBackup);
profileBackupFile.addEventListener("change", () => { void loadProfileBackupFile(); });
profileBackupModeEl.addEventListener("change", renderProfileBackupPreview);
profileBackupApplyBtn.addEventListener("click", applyPendingProfileBackup);
profileBackupCancelBtn.addEventListener("click", () => {
  clearPendingProfileBackup();
  profileBackupStatusEl.textContent = "Backup operation canceled; local data is unchanged.";
});
profileResetPreviewBtn.addEventListener("click", previewProfileReset);
profileRepairBtn.addEventListener("click", recoverLoadedProfileCopy);

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

filterMetadataMoreBtn.addEventListener("click", () => {
  void updateRecommendations(true);
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

discoveryViewSelect.addEventListener("change", () => {
  const value = discoveryViewSelect.value;
  if (value !== "auto" && value !== "popularity" && value !== "quality" && value !== "related") return;
  discoveryView = value;
  discoveryMetadataBatchRequested = !demoMode && (value === "quality" || value === "related");
  void updateRecommendations();
});

discoveryLoadMetadataBtn.addEventListener("click", () => {
  discoveryMetadataBatchRequested = true;
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

allowRelatedInput.addEventListener("change", () => {
  allowRelatedTitles = allowRelatedInput.checked;
  persistRecommendationState();
  void updateRecommendations();
});

clearSelectionBtn.addEventListener("click", () => {
  selectedNodeId = null;
  networkSearchMessage.textContent = "";
  renderInspectPanel(null);
  if (currentGraph) {
    renderSvgGraph(currentGraph);
  }
});

focusNeighborhoodBtn.addEventListener("click", () => {
  if (selectedNodeId) enterFocusedNeighborhood(selectedNodeId);
});

resetNeighborhoodBtn.addEventListener("click", () => {
  neighborhoodFocusId = null;
  selectedNodeId = null;
  networkSearchInput.value = "";
  networkSearchMessage.textContent = "";
  networkNodeFilterEl.value = "";
  networkNodeListPage = 0;
  minWeightInput.value = "0";
  toggleAnimeEdges.checked = true;
  toggleUsers.checked = false;
  resetGraphViewport();
  rerenderGraph();
});

graphZoomInBtn.addEventListener("click", () => zoomGraphViewport(0.8));
graphZoomOutBtn.addEventListener("click", () => zoomGraphViewport(1.25));
graphZoomResetBtn.addEventListener("click", resetGraphViewport);
graphShell.addEventListener("keydown", (event) => {
  if (event.target !== graphShell) return;
  const panSteps: Record<string, [number, number]> = {
    ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1],
  };
  if (event.key in panSteps) {
    const [horizontal, vertical] = panSteps[event.key];
    panGraphViewport(horizontal * graphViewport.width * 0.12,
      vertical * graphViewport.height * 0.12);
  } else if (event.key === "+" || event.key === "=") {
    zoomGraphViewport(0.8);
  } else if (event.key === "-") {
    zoomGraphViewport(1.25);
  } else if (event.key === "0") {
    resetGraphViewport();
  } else {
    return;
  }
  event.preventDefault();
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
  const found = resolveNetworkNodeQuery(query, graphData);
  if (found.ambiguous) {
    networkSearchMessage.textContent = "Several graph nodes match. Enter the exact node ID to choose one.";
    return;
  }
  const match = found.match;
  if (!match) {
    networkSearchMessage.textContent = "No match in the loaded recommendation graph. This does not prove no relationship in omitted source data.";
    return;
  }
  networkSearchMessage.textContent = `Focused: ${match.label} (${match.id})`;
  if (match.nodeType === "anime") {
    enterFocusedNeighborhood(match.id);
  } else {
    selectNodeAndFocus(match.id);
  }
});

networkNodeFilterEl.addEventListener("input", () => {
  networkNodeListPage = 0;
  renderVisibleNodeList();
});

networkNodePrevBtn.addEventListener("click", () => {
  networkNodeListPage = Math.max(0, networkNodeListPage - 1);
  renderVisibleNodeList();
});

networkNodeNextBtn.addEventListener("click", () => {
  networkNodeListPage += 1;
  renderVisibleNodeList();
});

networkNodeListResultsEl.addEventListener("click", (event) => {
  const target = event.target;
  if (!(target instanceof Element) || !currentGraph) return;
  const button = target.closest<HTMLButtonElement>("button[data-node-id]");
  const nodeId = button?.dataset.nodeId;
  if (!nodeId || !currentGraph.hasNode(nodeId)) return;
  const attributes = currentGraph.getNodeAttributes(nodeId) as Record<string, unknown>;
  const label = typeof attributes.label === "string" ? attributes.label : nodeId;
  selectedNodeId = nodeId;
  networkSearchMessage.textContent = `Focused: ${label} (${nodeId})`;
  renderInspectPanel(nodeId);
  renderSvgGraph(currentGraph);
});

revealApp();

function setActiveView(view: AppView, fromHash: boolean): void {
  const leavingView = view === "network" ? viewRecommendations : viewNetwork;
  const focusWasInLeavingView = leavingView.contains(document.activeElement);
  activeView = view;
  viewRecommendations.hidden = view !== "recommendations";
  viewNetwork.hidden = view !== "network";
  renderContextualTips();
  applyNetworkCompactControlsState();

  navRecommendationsBtn.classList.toggle("active", view === "recommendations");
  navNetworkBtn.classList.toggle("active", view === "network");
  if (view === "recommendations") {
    navRecommendationsBtn.setAttribute("aria-current", "page");
    navNetworkBtn.removeAttribute("aria-current");
  } else {
    navNetworkBtn.setAttribute("aria-current", "page");
    navRecommendationsBtn.removeAttribute("aria-current");
  }
  if (focusWasInLeavingView) {
    (view === "network" ? navNetworkBtn : navRecommendationsBtn).focus({ preventScroll: true });
  }

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
        const errorMessage = error instanceof ArtifactValidationError || error instanceof ArtifactLoadError
          ? error.message : "unable to load or verify the optional asset";
        setGraphLoadingState(
          false,
          `Render status: failed to load explorer data (${errorMessage}).`,
        );
        recordDiagnosticIssue("EXPLORER-001");
        console.error("Explorer graph load failed [EXPLORER-001].");
      });
  } else {
    graphRenderRunId += 1;
    graphRenderController?.abort();
    graphRenderController = null;
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
  if (activeView === "network" && hideControls && networkPanelEl.contains(document.activeElement)) {
    networkMobileToggleBtn.focus({ preventScroll: true });
  }
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
  const opener = document.activeElement;
  commandPaletteReturnFocus = opener instanceof HTMLElement && opener !== document.body
    ? opener : commandsToggleBtn;
  commandPaletteOpen = true;
  commandPaletteEl.hidden = false;
  commandPaletteEl.setAttribute("aria-hidden", "false");
  appTopbarEl.inert = true;
  appMainEl.inert = true;
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
  appTopbarEl.inert = false;
  appMainEl.inert = false;
  document.body.classList.remove("palette-open");
  clearCommandDragState();
  const restoreTo = commandPaletteReturnFocus;
  commandPaletteReturnFocus = null;
  (restoreTo?.isConnected && restoreTo.getClientRects().length > 0 &&
      !restoreTo.closest("[hidden]") && !restoreTo.matches(":disabled")
    ? restoreTo : commandsToggleBtn).focus();
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
  chooseAnimeFromInput(animeInput, recMessageEl, (anime) => {
    if (selectedAnimeNodeIds.includes(anime.nodeId)) {
      recMessageEl.textContent = `${anime.label} is already in your watched list.`;
      animeInput.value = "";
      return;
    }

    animeInput.value = "";
    const sentiment = addPreferenceSelect.value as PreferenceSentiment;
    addPreferenceSelect.value = favoriteQuickstartActive && sentiment === "liked" ? "liked" : "seen";
    addAnimeToWatchedList(anime, "Added", ["seen", "liked", "disliked"].includes(sentiment) ? sentiment : "seen");
  });
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
        recMessageEl.textContent = "Loaded local XML file. Review the import preview.";
      } else {
        setAsyncStatus(bulkImportStatusEl, "empty", "The local XML has no anime entries; current history is intact.");
      }
    } else {
      bulkImportInput.value = content;
      if (content.trim()) {
        setAsyncStatus(bulkImportStatusEl, "ready", "Local text loaded. Preview the entries before applying.");
        recMessageEl.textContent = "Loaded local text file. Select Preview Text Import.";
      } else {
        setAsyncStatus(bulkImportStatusEl, "empty", "The local file has no entries to import.");
        recMessageEl.textContent = "Local file is empty.";
      }
    }
    activeBulkFileLoadId = null;
  } catch (error) {
    if (activeBulkFileLoadId !== loadId) return;
    activeBulkFileLoadId = null;
    clearPendingHistoryImport();
    setAsyncStatus(bulkImportStatusEl, "failed", "Unable to parse that local file; current history is intact.");
    recMessageEl.textContent = error instanceof HistoryImportValidationError ? error.message
      : "Unable to read or parse that local file. Check its .txt or .xml format.";
    recordDiagnosticIssue("IMPORT-002");
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
    recMessageEl.textContent = error instanceof HistoryImportValidationError ? error.message
      : "Text import is invalid. Check the documented line format and preview again.";
    recordDiagnosticIssue("IMPORT-002");
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
    recMessageEl.textContent = provider === "mal"
      ? "Direct MAL import failed. Browser access may be blocked, the profile may be private, or MAL may be rate limiting. No proxy was contacted. Try the local file or text import above, or AniList."
      : "AniList import failed. Check provider availability and try the local file or text import. Current history is intact.";
    recordDiagnosticIssue("IMPORT-001");
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
  } catch {
    historyEntries.splice(0, historyEntries.length, ...previousHistory);
    selectedAnimeNodeIds.splice(0, selectedAnimeNodeIds.length, ...previousSelected);
    selectedAnimePreferences.clear();
    for (const [nodeId, preference] of previousPreferences) selectedAnimePreferences.set(nodeId, preference);
    renderSelectedAnime();
    renderImportedHistory();
    setAsyncStatus(pending.origin === "local" ? bulkImportStatusEl : usernameImportStatusEl,
      "failed", "Import was not applied; prior history is intact.");
    recMessageEl.textContent = "Import was not applied. Check browser storage and try the preview again.";
    recordDiagnosticIssue("STORAGE-001");
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

function recordDiagnosticIssue(code: DiagnosticCode): void {
  const issue = diagnosticIssue(code);
  diagnosticCodeEl.textContent = `${issue.code} · ${issue.title}`;
  diagnosticActionEl.textContent = issue.action;
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

function addWatchlistFromInput(): void {
  chooseAnimeFromInput(watchlistInput, watchlistStatusEl, (anime) => {
    if (addToWatchlist(anime)) watchlistInput.value = "";
  });
}

function addToWatchlist(anime: AnimeInfo): boolean {
  if (watchlistEntries.some((item) => item.animeId === anime.animeId)) {
    watchlistStatusEl.textContent = `${anime.label} is already on your watchlist.`;
    return false;
  }
  return applyWatchlistChange([...watchlistEntries, {
    animeId: anime.animeId, title: anime.label, status: "plan_to_watch", rating: null,
  }], `Saved ${anime.label} to Plan to Watch.`);
}

function markResultSeen(anime: AnimeInfo): void {
  if (selectedAnimeNodeIds.includes(anime.nodeId)) return;
  selectedAnimeNodeIds.push(anime.nodeId);
  selectedAnimePreferences.set(anime.nodeId, manualPreference(anime.nodeId, "seen"));
  if (!persistRecommendationState()) {
    selectedAnimeNodeIds.pop();
    selectedAnimePreferences.delete(anime.nodeId);
    recActionStatusEl.textContent = "Browser storage rejected Seen; no preference was saved.";
    recActionStatusEl.focus({ preventScroll: true });
    void updateRecommendations();
    return;
  }
  renderSelectedAnime();
  recActionStatusEl.textContent = `Marked ${anime.label} Seen. This hides it without treating it as Liked. Remove it from Watched & Preferences to undo.`;
  recActionStatusEl.focus({ preventScroll: true });
  void updateRecommendations();
}

function hideResult(anime: AnimeInfo): void {
  if (excludeCandidateNodeIds.includes(anime.nodeId)) return;
  excludeCandidateNodeIds.push(anime.nodeId);
  if (!persistRecommendationState()) {
    excludeCandidateNodeIds.pop();
    recActionStatusEl.textContent = "Browser storage rejected Not interested; the title was not hidden.";
    recActionStatusEl.focus({ preventScroll: true });
    void updateRecommendations();
    return;
  }
  renderExcludeCandidates();
  recActionStatusEl.textContent = `Hidden ${anime.label}. Not interested excludes this title; it does not create a Disliked model signal. Remove it under Candidate Overrides to undo.`;
  recActionStatusEl.focus({ preventScroll: true });
  void updateRecommendations();
}

async function retryResultMetadata(anime: AnimeInfo, button: HTMLButtonElement): Promise<void> {
  const signal = recommendationController?.signal;
  if (!signal || signal.aborted || demoMode) return;
  animeMetadataFailed.delete(anime.animeId);
  button.disabled = true;
  button.textContent = "Retrying details...";
  recActionStatusEl.textContent = `Retrying details for ${anime.label}.`;
  const metadata = await ensureAnimeMetadata(anime.animeId, signal);
  if (signal.aborted) return;
  if (metadata || animeMetadataUnavailable.has(anime.animeId)) {
    recActionStatusEl.textContent = metadata
      ? `Details loaded for ${anime.label}.`
      : `Details remain unavailable for ${anime.label}.`;
    void updateRecommendations();
  } else {
    button.disabled = false;
    button.textContent = "Retry details";
    recActionStatusEl.textContent = `Details still failed for ${anime.label}. You can retry.`;
  }
}

function changeWatchlistEntry(nextEntry: WatchlistEntry): void {
  const current = watchlistEntries.find((item) => item.animeId === nextEntry.animeId);
  if (!current || JSON.stringify(current) === JSON.stringify(nextEntry)) return;
  const field = current.status !== nextEntry.status ? "status" : "rating";
  applyWatchlistChange(watchlistEntries.map((item) =>
    item.animeId === nextEntry.animeId ? nextEntry : item),
  `Saved local ${field} for ${nextEntry.title}.`,
  field === "status" ? `select[data-watchlist-status-id="${nextEntry.animeId}"]`
    : `select[data-watchlist-rating-id="${nextEntry.animeId}"]`);
}

function applyWatchlistChange(next: WatchlistEntry[], message: string, focusSelector?: string): boolean {
  let checked: WatchlistEntry[];
  try { checked = validateWatchlist(next); } catch {
    watchlistStatusEl.textContent = "Invalid watchlist change; your list was not changed.";
    renderWatchlist();
    return false;
  }
  const previous = [...watchlistEntries];
  watchlistEntries.splice(0, watchlistEntries.length, ...checked);
  if (!persistRecommendationState()) {
    watchlistEntries.splice(0, watchlistEntries.length, ...previous);
    renderWatchlist();
    watchlistStatusEl.textContent = "Browser storage rejected the watchlist change; your previous list is intact.";
    return false;
  }
  renderWatchlist();
  if (focusSelector) watchlistListEl.querySelector<HTMLElement>(focusSelector)?.focus({ preventScroll: true });
  watchlistStatusEl.textContent = message;
  void updateRecommendations();
  return true;
}

function renderWatchlist(): void {
  watchlistCountEl.textContent = String(watchlistEntries.length);
  watchlistListEl.replaceChildren();
  if (watchlistEntries.length === 0) {
    const empty = document.createElement("li");
    empty.className = "muted";
    empty.textContent = "No saved titles yet. Save a recommendation or enter an exact catalog title.";
    watchlistListEl.append(empty);
    return;
  }
  const statusOptions: [WatchlistStatus, string][] = [
    ["plan_to_watch", "Plan to Watch"], ["watching", "Watching"],
    ["completed", "Completed"], ["on_hold", "On Hold"], ["dropped", "Dropped"],
  ];
  for (const entry of watchlistEntries) {
    const row = document.createElement("li");
    row.className = "watchlist-item";
    const title = document.createElement("span");
    title.className = "watchlist-item-title";
    const known = recommendationIndex.animeByAnimeId.get(entry.animeId);
    title.textContent = known?.label ?? entry.title;
    row.append(title);
    if (!known) {
      const missing = document.createElement("span");
      missing.className = "chip-missing-note";
      missing.textContent = "Unavailable in this catalog; saved title and ID retained";
      row.append(missing);
    }
    const controls = document.createElement("div");
    controls.className = "watchlist-item-controls";
    const statusLabel = document.createElement("label");
    statusLabel.textContent = "Status ";
    const status = document.createElement("select");
    status.dataset.watchlistStatusId = String(entry.animeId);
    status.setAttribute("aria-label", `Watch status for ${known?.label ?? entry.title}`);
    for (const [value, label] of statusOptions) status.add(new Option(label, value));
    status.value = entry.status;
    statusLabel.append(status);
    controls.append(statusLabel);
    const ratingLabel = document.createElement("label");
    ratingLabel.textContent = "My rating ";
    const rating = document.createElement("select");
    rating.dataset.watchlistRatingId = String(entry.animeId);
    rating.setAttribute("aria-label", `My rating for ${known?.label ?? entry.title}`);
    rating.add(new Option("Unrated", ""));
    for (let score = 1; score <= 10; score += 1) rating.add(new Option(`${score}/10`, String(score)));
    rating.value = entry.rating === null ? "" : String(entry.rating);
    ratingLabel.append(rating);
    controls.append(ratingLabel);
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "ghost-btn";
    remove.dataset.watchlistRemoveId = String(entry.animeId);
    remove.setAttribute("aria-label", `Remove ${known?.label ?? entry.title} from watchlist`);
    remove.textContent = "Remove";
    controls.append(remove);
    row.append(controls);
    watchlistListEl.append(row);
  }
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
  chooseAnimeFromInput(input, recMessageEl, (anime) => {
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
  });
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

function selectVisibleFranchises(eligible: readonly RecommendationResult[]): FranchiseSelection {
  const planned = new Set(watchlistEntries.filter((entry) => entry.status === "plan_to_watch")
    .map((entry) => entry.animeId));
  const seen = new Set(seenHistoryAnimeIds(historyEntries, recommendationIndex)
    .filter((animeId) => !planned.has(animeId)));
  for (const animeId of watchedWatchlistAnimeIds(watchlistEntries)) seen.add(animeId);
  for (const nodeId of selectedAnimeNodeIds) {
    const rawId = /^anime:([1-9]\d*)$/.exec(nodeId)?.[1];
    const animeId = recommendationIndex.animeByNodeId.get(nodeId)?.animeId ??
      (rawId ? Number(rawId) : undefined);
    if (animeId !== undefined && Number.isSafeInteger(animeId) && !planned.has(animeId)) seen.add(animeId);
  }
  const titles = new Map(recommendationIndex.animeList.map((item) => [item.animeId, item.label]));
  return selectFranchiseDiverseRecommendations(
    eligible, animeMetadataCache, seen, allowRelatedTitles, titles);
}

function franchiseSelectionSummary(selection: FranchiseSelection, eligibleCount: number): string {
  const coverage = `Relationship entries checked for ${selection.relationshipPayloadCount}/${eligibleCount} eligible titles; missing or empty entries do not prove there are no prerequisites.`;
  if (allowRelatedTitles) return `Related titles allowed in original eligible order. ${coverage}`;
  return `Prefer variety withheld ${selection.knownPrequelHidden} known-unwatched sequel(s) and ` +
    `${selection.repeatedFranchiseHidden} repeated known/title-suggested series entry(ies). ${coverage}`;
}

function updateRecommendations(checkMoreFilterMetadata = false): Promise<void> {
  return measureLocalAsync("wasiw:recommendation:update", () =>
    updateRecommendationsBody(checkMoreFilterMetadata));
}

async function updateRecommendationsBody(checkMoreFilterMetadata = false): Promise<void> {
  recommendationController?.abort();
  filterMetadataMoreBtn.disabled = true;
  filterMetadataControlsEl.hidden = true;
  const controller = new AbortController();
  recommendationController = controller;
  const runId = ++recommendationRunId;
  async function yieldForLatestRecommendation(): Promise<boolean> {
    try {
      await measureLocalAsync("wasiw:recommendation:yield", () =>
        runtime.sleep(0, controller.signal));
    } catch (error) {
      if (isAbortError(error)) return false;
      throw error;
    }
    return runId === recommendationRunId && !controller.signal.aborted;
  }
  const filtersActive = hasActiveRecommendationFilters(recommendationFilters);
  const selectedPreferences = selectedAnimeNodeIds
    .map((nodeId) => selectedAnimePreferences.get(nodeId))
    .filter((item): item is AnimePreference => item !== undefined);
  const preferences = mergeWatchlistFeedback(selectedPreferences, watchlistEntries, recommendationIndex);
  const hasEngineSignal = recommendationMode === "graph"
    ? preferences.some((item) => item.sentiment === "liked")
    : preferences.some((item) => item.sentiment !== "seen");
  const eligibilityPolicy = createCandidateEligibilityPolicy({
    index: recommendationIndex,
    preferences,
    history: historyEntries,
    watchlist: watchlistEntries,
    includeOnlyNodeIds: includeCandidateNodeIds,
    excludeNodeIds: excludeCandidateNodeIds,
    filters: recommendationFilters,
  });
  const activeDiscoveryView = discoveryView === "auto"
    ? hasEngineSignal ? null : samplePopularityAvailable ? "popularity" : "quality"
    : discoveryView;
  if (activeDiscoveryView) {
    await renderCatalogExploration(activeDiscoveryView, preferences, eligibilityPolicy, runId, controller.signal);
    return;
  }
  discoveryMetadataControlsEl.hidden = true;

  const graphRecommendations = measureLocal("wasiw:recommendation:graph-score", () =>
    buildGraphRecommendationsForPreferences(preferences, recommendationIndex));
  let modelRecommendations: RecommendationResult[] = [];
  let activeRankingMode: EligibilityRankingMode = recommendationMode;
  let usingFallback = false;
  let fallbackDisplay: "coverage" | "related" = "coverage";
  let fallbackReason: string | null = null;
  let modelFactors: number | null = null;
  let modelCoverageNote = "";
  const sources: {
    graph: RecommendationResult[];
    model: RecommendationResult[];
    fallback: RecommendationResult[];
  } = { graph: graphRecommendations, model: modelRecommendations, fallback: [] };
  const rankCurrentCandidates = () => measureLocal("wasiw:recommendation:eligibility", () =>
    rankEligibleCandidates(activeRankingMode, sources, eligibilityPolicy,
      animeMetadataCache, modelBlendWeight));
  function showActiveEngine(): void {
    if (activeRankingMode === "fallback") {
      recEngineStatusEl.textContent = fallbackDisplay === "related"
        ? `Using shared-genre content baseline from Liked titles. ${fallbackReason ?? ""}`
        : `Using catalog coverage baseline (positive graph connections, then anime ID). ${fallbackReason ?? ""}`;
    } else if (activeRankingMode === "graph") {
      recEngineStatusEl.textContent = fallbackReason
        ? `Using graph fallback. ${fallbackReason}`
        : "Using graph recommendations.";
    } else if (activeRankingMode === "model") {
      recEngineStatusEl.textContent = `Using ML model recommendations (${modelFactors} factors)${modelCoverageNote}.`;
    } else {
      recEngineStatusEl.textContent =
        `Using hybrid recommendations via rank fusion (${Math.round(modelBlendWeight * 100)}% model, ${Math.round((1 - modelBlendWeight) * 100)}% graph). Rank points are relative, not probabilities${modelCoverageNote}.`;
    }
  }
  function selectContentBaseline(): boolean {
    const likedCount = preferences.filter((item) => item.sentiment === "liked").length;
    if (likedCount === 0 || likedCount > MAX_SPARSE_CONTENT_SEEDS) return false;
    const content = buildGenreOverlapExploration(preferences, recommendationIndex, animeMetadataCache);
    const hasConfirmedContent = eligibilityPolicy.evaluate(content, animeMetadataCache).recommendations.length > 0;
    const catalog = filtersActive && !demoMode ? eligibilityPolicy.evaluate(
      buildSamplePopularityExploration(recommendationIndex), animeMetadataCache,
    ).structurallyEligible : [];
    const catalogMayStillMatch = catalog.length > 0 &&
      unresolvedMetadataCount(candidateMetadataCoverage(catalog)) > 0;
    if (!hasConfirmedContent && !catalogMayStillMatch) return false;
    activeRankingMode = "fallback";
    usingFallback = true;
    fallbackDisplay = "related";
    sources.fallback = content;
    showActiveEngine();
    return true;
  }
  function selectCatalogBaseline(): void {
    if (selectContentBaseline()) return;
    activeRankingMode = "fallback";
    usingFallback = true;
    fallbackDisplay = "coverage";
    sources.fallback = buildCatalogCoverageRecommendations(recommendationIndex);
    showActiveEngine();
  }
  function currentFallbackDisplay(): "coverage" | "related" {
    return fallbackDisplay;
  }
  let sparseContentMetadataPrimed = false;
  async function primeSparseContentMetadata(): Promise<boolean> {
    if (demoMode || sparseContentMetadataPrimed) return runId === recommendationRunId && !controller.signal.aborted;
    const liked = preferences.filter((item) => item.sentiment === "liked");
    if (liked.length === 0 || liked.length > MAX_SPARSE_CONTENT_SEEDS) return true;
    sparseContentMetadataPrimed = true;
    const seedIds = liked.map((item) => recommendationIndex.animeByNodeId.get(item.nodeId)?.animeId)
      .filter((animeId): animeId is number => animeId !== undefined);
    const candidateIds = eligibilityPolicy.evaluate(
      buildSamplePopularityExploration(recommendationIndex), animeMetadataCache,
    ).structurallyEligible.slice(0, METADATA_PREFETCH_LIMIT).map((item) => item.anime.animeId);
    setMetadataStatus("loading", "Metadata: checking a bounded catalog sample for shared genres...");
    await hydrateMetadataForAnimeIds(seedIds, seedIds.length, controller.signal);
    if (runId !== recommendationRunId || controller.signal.aborted) return false;
    await hydrateMetadataForAnimeIds(candidateIds, 12, controller.signal);
    return runId === recommendationRunId && !controller.signal.aborted;
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
      modelRecommendations = measureLocal("wasiw:recommendation:model-score", () =>
        buildModelRecommendationsForPreferences(preferences, recommendationIndex, modelIndex));
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

  const largeRanking = Math.max(graphRecommendations.length, modelRecommendations.length) >
    LARGE_RECOMMENDATION_YIELD_CANDIDATES;
  if (largeRanking && !await yieldForLatestRecommendation()) return;

  let initialEligibility = rankCurrentCandidates();
  if (recommendationMode !== "graph" && activeRankingMode !== "graph" &&
      initialEligibility.structurallyEligible.length === 0) {
    activeRankingMode = "graph";
    fallbackReason = "No ML candidates passed catalog and eligibility rules.";
    showActiveEngine();
    initialEligibility = rankCurrentCandidates();
  }
  if (fallbackReason && initialEligibility.structurallyEligible.length === 0) {
    if (!await primeSparseContentMetadata()) return;
    selectCatalogBaseline();
    initialEligibility = rankCurrentCandidates();
  }
  if (recommendationMode === "graph" && initialEligibility.structurallyEligible.length === 0) {
    if (!await primeSparseContentMetadata()) return;
    if (selectContentBaseline()) {
      initialEligibility = rankCurrentCandidates();
    }
  }
  let structuralCandidates = initialEligibility.structurallyEligible;
  if (structuralCandidates.length === 0 &&
      !(filtersActive && usingFallback && currentFallbackDisplay() === "related")) {
    recSummaryEl.textContent = includeCandidateNodeIds.length > 0
      ? "No scored candidates match Include Only after watched and excluded titles are removed."
      : "No eligible recommendations found from these preferences. Try another liked title or review filters.";
    recResultsEl.innerHTML = "";
    setMetadataStatus("empty", "Metadata: no candidates.");
    return;
  }

  let metadataCandidateAnimeIds: number[] = [];
  let filterFetchBudget = METADATA_PREFETCH_WITH_FILTER_LIMIT;
  async function hydrateRankedCandidates(candidates: RecommendationResult[]): Promise<boolean> {
    metadataCandidateAnimeIds = candidates
      .map((item) => item.anime.animeId)
      .filter((animeId) => Number.isFinite(animeId) && animeId > 0);
    if (filtersActive && !demoMode) {
      const scope = checkMoreFilterMetadata ? candidates
        : candidates.slice(0, METADATA_PREFETCH_WITH_FILTER_LIMIT);
      const coverage = candidateMetadataCoverage(scope);
      let batch = coverage.uncheckedAnimeIds.slice(0, filterFetchBudget);
      if (batch.length === 0 && checkMoreFilterMetadata) {
        batch = coverage.failedAnimeIds.slice(0, filterFetchBudget);
        for (const animeId of batch) animeMetadataFailed.delete(animeId);
      }
      if (batch.length > 0) {
        setMetadataStatus("loading", `Checking ${batch.length} candidate metadata records for required filters...`);
        filterFetchBudget -= batch.length;
        await hydrateMetadataForAnimeIds(batch, batch.length, controller.signal);
      }
    } else {
      await hydrateMetadataForAnimeIds(metadataCandidateAnimeIds.slice(0, METADATA_PREFETCH_LIMIT),
        Math.min(12, metadataCandidateAnimeIds.length), controller.signal);
    }
    return runId === recommendationRunId && !controller.signal.aborted;
  }
  if (!await hydrateRankedCandidates(structuralCandidates)) {
    return;
  }

  let finalEligibility = rankCurrentCandidates();
  if (recommendationMode !== "graph" && !usingFallback &&
      finalEligibility.recommendations.length === 0 &&
      (!filtersActive || unresolvedMetadataCount(candidateMetadataCoverage(
        finalEligibility.structurallyEligible)) === 0)) {
    if (activeRankingMode !== "graph") {
      activeRankingMode = "graph";
      fallbackReason = "No ML candidates passed catalog and required filters.";
      showActiveEngine();
      structuralCandidates = rankCurrentCandidates().structurallyEligible;
      if (!await hydrateRankedCandidates(structuralCandidates)) {
        return;
      }
      finalEligibility = rankCurrentCandidates();
    }
    if (finalEligibility.recommendations.length === 0 &&
        (!filtersActive || unresolvedMetadataCount(candidateMetadataCoverage(
          finalEligibility.structurallyEligible)) === 0)) {
      if (!await primeSparseContentMetadata()) return;
      selectCatalogBaseline();
      structuralCandidates = rankCurrentCandidates().structurallyEligible;
      if (!await hydrateRankedCandidates(structuralCandidates)) {
        return;
      }
      finalEligibility = rankCurrentCandidates();
    }
  }
  if (recommendationMode === "graph" && finalEligibility.recommendations.length === 0 && !usingFallback &&
      (!filtersActive || unresolvedMetadataCount(candidateMetadataCoverage(
        finalEligibility.structurallyEligible)) === 0)) {
    if (!await primeSparseContentMetadata()) return;
    if (selectContentBaseline()) {
      structuralCandidates = rankCurrentCandidates().structurallyEligible;
      finalEligibility = rankCurrentCandidates();
    }
  }
  if (largeRanking) {
    if (!await yieldForLatestRecommendation()) return;
  }
  let filterCoverageCandidates = finalEligibility.structurallyEligible;
  if (filtersActive && usingFallback && currentFallbackDisplay() === "related" && !demoMode) {
    const eligibleCatalog = eligibilityPolicy.evaluate(
      buildSamplePopularityExploration(recommendationIndex), animeMetadataCache,
    ).structurallyEligible;
    if (checkMoreFilterMetadata) {
      if (!await hydrateRankedCandidates(eligibleCatalog)) return;
      sources.fallback = buildGenreOverlapExploration(preferences, recommendationIndex, animeMetadataCache);
      finalEligibility = rankCurrentCandidates();
      if (recommendationMode !== "graph" && finalEligibility.recommendations.length === 0 &&
          unresolvedMetadataCount(candidateMetadataCoverage(eligibleCatalog)) === 0) {
        selectCatalogBaseline();
        finalEligibility = rankCurrentCandidates();
      }
    }
    filterCoverageCandidates = usingFallback && currentFallbackDisplay() === "related"
      ? eligibleCatalog : finalEligibility.structurallyEligible;
  }
  const filterCoverage = measureLocal("wasiw:recommendation:filter-ui", () => {
    updateGenreFilterOptions(filterCoverageCandidates);
    return renderRankedFilterMetadataCoverage(filterCoverageCandidates);
  });
  const filterScanIncomplete = filtersActive && unresolvedMetadataCount(filterCoverage) > 0;
  const franchiseSelection = measureLocal("wasiw:recommendation:franchise", () =>
    selectVisibleFranchises(finalEligibility.recommendations));
  const filteredRecommendations = franchiseSelection.recommendations;
  const effectiveFusion = activeRankingMode === "hybrid" ? filteredRecommendations[0]?.fusion : undefined;
  if (effectiveFusion &&
      (effectiveFusion.modelWeight !== modelBlendWeight ||
       effectiveFusion.graphWeight !== 1 - modelBlendWeight)) {
    const available = effectiveFusion.modelWeight === 1 ? "model" : "graph";
    const missing = available === "model" ? "graph" : "model";
    recEngineStatusEl.textContent = `Using hybrid recommendations via ${available}-only rank fusion ` +
      `(${missing} has no eligible candidates; full weight goes to ${available}). ` +
      "Rank points are relative, not probabilities.";
  }
  if (filteredRecommendations.length === 0) {
    recSummaryEl.textContent = finalEligibility.recommendations.length > 0
      ? `Known immediate prequels block all currently eligible titles. Turn on Allow related titles to inspect them. ${franchiseSelectionSummary(franchiseSelection, finalEligibility.recommendations.length)}`
      : filtersActive
        ? filterScanIncomplete
          ? `No confirmed matches yet. Filter scan is partial: ${metadataCoverageText(filterCoverage)} Check more candidate metadata before treating this as no match.`
          : `No candidates meet the required filters after checking all ${filterCoverage.total} eligible titles.` +
            (filterCoverage.unavailable > 0
              ? ` ${filterCoverage.unavailable} title(s) had unavailable metadata and could not pass.` : "")
        : "No eligible recommendations found from these preferences.";
    recResultsEl.innerHTML = "";
    const metadataState: AsyncUiState = demoMode ? "demo"
      : filtersActive ? filterCoverage.failed > 0 ? "failed"
        : filterScanIncomplete ? "partial" : filterCoverage.unavailable > 0 ? "unavailable" : "empty"
        : metadataCandidateAnimeIds.some((id) => animeMetadataFailed.has(id)) ? "failed"
          : metadataCandidateAnimeIds.some((id) => animeMetadataUnavailable.has(id)) ? "unavailable" : "empty";
    setMetadataStatus(metadataState, filtersActive
      ? `Filter metadata: ${metadataCoverageText(filterCoverage)}`
      : metadataState === "failed" ? "Metadata request failed; some candidates could not be checked."
        : metadataState === "unavailable" ? "Metadata is unavailable for some candidates."
          : "Metadata: no matches after filters.");
    return;
  }

  const methodLabel = usingFallback ? currentFallbackDisplay() === "related"
    ? "shared-genre content baseline" : "catalog coverage baseline"
    : activeRankingMode === "graph" ? fallbackReason ? "graph edge fallback" : "graph edge ranking"
      : activeRankingMode === "model" ? "ML model ranking" : "hybrid graph+ML rank fusion";
  const filterSummary = formatActiveFilterSummary();
  const fallbackScopeNote = usingFallback && currentFallbackDisplay() === "related" && !demoMode
    ? samplePopularityAvailable
      ? " Shared genres use available metadata; the automatic check covers at most 12 eligible catalog titles by sampled rating count."
      : " Shared genres use available metadata; the automatic check covers at most 12 eligible catalog titles in catalog order."
    : "";
  recSummaryEl.textContent = `Showing top ${Math.min(MAX_RECOMMENDATIONS, filteredRecommendations.length)} recommendations from ${finalEligibility.recommendations.length} eligible candidates (${methodLabel})${filterSummary}. ${franchiseSelectionSummary(franchiseSelection, finalEligibility.recommendations.length)}${fallbackScopeNote}` +
    (filtersActive ? ` Filter scan ${filterScanIncomplete ? "partial" : "complete"}: ${metadataCoverageText(filterCoverage)}` : "");

  const visibleRecommendations = measureLocal("wasiw:recommendation:cards", () =>
    filteredRecommendations.slice(0, MAX_RECOMMENDATIONS)
      .map((item) => renderRecommendationCard(item,
        usingFallback ? currentFallbackDisplay() : "ranking",
        franchiseSelection.notesByAnimeId.get(item.anime.animeId))));
  measureLocal("wasiw:recommendation:dom", () => {
    recResultsEl.innerHTML = visibleRecommendations.join("");
  });

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
    : filtersActive ? filterCoverage.failed > 0 ? "failed"
      : filterScanIncomplete ? "partial" : filterCoverage.unavailable > 0 ? "unavailable" : "ready"
      : visibleIds.some((id) => animeMetadataFailed.has(id)) ? "failed"
        : visibleIds.some((id) => animeMetadataUnavailable.has(id)) ? "unavailable" : "ready";
  const providerNote = metadataState === "failed" ? " | some metadata requests failed"
    : metadataState === "unavailable" ? " | some metadata unavailable"
      : metadataState === "demo" ? " | synthetic demo data" : "";
  setMetadataStatus(metadataState,
    `Metadata loaded for ${visibleWithMetadata}/${Math.min(MAX_RECOMMENDATIONS, filteredRecommendations.length)} visible recommendations` +
      `${visibleMissingMetadata > 0 ? ` | pending: ${visibleMissingMetadata}` : ""}` +
      missingNote + providerNote + (filtersActive ? ` | ${metadataCoverageText(filterCoverage)}` : ""),
  );
}

async function renderCatalogExploration(
  view: Exclude<DiscoveryView, "auto">,
  preferences: readonly AnimePreference[],
  policy: CandidateEligibilityPolicy,
  runId: number,
  signal: AbortSignal,
): Promise<void> {
  if (view === "popularity" && !samplePopularityAvailable) {
    recEngineStatusEl.textContent = "Popularity proxy unavailable in this aggregate-only graph.";
    recSummaryEl.textContent = "This release omits user-anime edges, so sampled rating counts are unavailable. Try community scores or shared genres.";
    recResultsEl.innerHTML = "";
    discoveryMetadataControlsEl.hidden = true;
    setMetadataStatus("unavailable", "Sample-based popularity is unavailable in this graph.");
    return;
  }
  const popularity = buildSamplePopularityExploration(recommendationIndex);
  const eligibleCatalog = policy.evaluate(popularity, animeMetadataCache).structurallyEligible;
  const metadataScopeIds = eligibleCatalog.map((item) => item.anime.animeId);
  const hasLikedSource = preferences.some((item) => item.sentiment === "liked");
  if (view === "related" && !hasLikedSource) discoveryMetadataBatchRequested = false;
  if (discoveryMetadataBatchRequested && !demoMode) {
    discoveryMetadataBatchRequested = false;
    discoveryLoadMetadataBtn.disabled = true;
    recEngineStatusEl.textContent = "Checking catalog metadata for exploration...";
    setMetadataStatus("loading", "Metadata: checking a bounded catalog batch...");
    if (view === "related") {
      const seedIds = preferences.filter((item) => item.sentiment === "liked")
        .map((item) => recommendationIndex.animeByNodeId.get(item.nodeId)?.animeId)
        .filter((animeId): animeId is number => animeId !== undefined);
      await hydrateMetadataForAnimeIds(seedIds, seedIds.length, signal);
      if (runId !== recommendationRunId || signal.aborted) return;
    }
    const coverage = candidateMetadataCoverage(eligibleCatalog);
    let batch = coverage.uncheckedAnimeIds.slice(0, 12);
    if (batch.length === 0) {
      batch = coverage.failedAnimeIds.slice(0, 12);
      for (const animeId of batch) animeMetadataFailed.delete(animeId);
    }
    await hydrateMetadataForAnimeIds(batch, batch.length, signal);
    if (runId !== recommendationRunId || signal.aborted) return;
  }

  updateGenreFilterOptions(eligibleCatalog);
  const coverage = candidateMetadataCoverage(eligibleCatalog);
  const incompleteMetadata = unresolvedMetadataCount(coverage) > 0;
  const metadataMatters = hasActiveRecommendationFilters(recommendationFilters) || view !== "popularity";
  const knownInScope = metadataScopeIds.filter((id) => animeMetadataCache.has(id)).length;
  const scoredInScope = metadataScopeIds.filter((id) => animeMetadataCache.get(id)?.score !== null &&
    animeMetadataCache.get(id)?.score !== undefined).length;
  const genresInScope = metadataScopeIds.filter((id) => (animeMetadataCache.get(id)?.genres.length ?? 0) > 0).length;
  discoveryMetadataControlsEl.hidden = demoMode;
  if (!demoMode) {
    discoveryMetadataNoteEl.textContent =
      `Metadata across the full ${metadataScopeIds.length}-title eligible catalog: ${metadataCoverageText(coverage)} ` +
      `${scoredInScope} have a community score and ` +
      `${genresInScope} have genres. Each check sends at most 12 catalog IDs through the existing metadata provider.`;
    discoveryLoadMetadataBtn.disabled = metadataScopeIds.length === 0 ||
      !incompleteMetadata || (view === "related" && !hasLikedSource);
    discoveryLoadMetadataBtn.textContent = coverage.uncheckedAnimeIds.length > 0
      ? `Check next ${Math.min(12, coverage.uncheckedAnimeIds.length)} eligible titles`
      : `Retry ${Math.min(12, coverage.failed)} failed metadata checks`;
  }

  const explorationResults = view === "popularity" ? popularity
    : view === "quality" ? buildCommunityQualityExploration(recommendationIndex, animeMetadataCache)
      : buildGenreOverlapExploration(preferences, recommendationIndex, animeMetadataCache);
  const finalEligibility = rankEligibleCandidates("fallback", { fallback: explorationResults },
    policy, animeMetadataCache);
  const franchiseSelection = selectVisibleFranchises(finalEligibility.recommendations);
  const results = franchiseSelection.recommendations;
  const heading = view === "popularity" ? "catalog popularity proxy"
    : view === "quality" ? "community-score exploration"
      : "shared-genre content baseline";
  recEngineStatusEl.textContent = `Using ${heading}; selected recommendation engine remains ${recommendationMode}.`;
  if (results.length === 0) {
    recSummaryEl.textContent = finalEligibility.recommendations.length > 0
      ? `Known immediate prequels block all currently eligible titles. Turn on Allow related titles to inspect them. ${franchiseSelectionSummary(franchiseSelection, finalEligibility.recommendations.length)}`
      : view === "related" && !hasLikedSource
      ? "Mark a title Liked to explore shared genres. Seen and Disliked titles are still excluded."
      : metadataMatters && incompleteMetadata
        ? `No confirmed matches yet. Metadata scan is partial: ${metadataCoverageText(coverage)} Check more eligible titles before treating this as no match.`
      : view === "quality"
        ? samplePopularityAvailable
          ? "No eligible catalog titles with a known community score match these filters. You can explore sampled rating counts."
          : "No eligible catalog titles with a known community score match these filters."
        : view === "related"
          ? "No eligible catalog titles with verified shared genres match these filters and Liked titles. Try another Liked title."
          : "No eligible catalog titles match these filters and overrides.";
    if (metadataMatters && !incompleteMetadata && !(view === "related" && !hasLikedSource)) {
      recSummaryEl.textContent += ` Metadata scan complete: ${metadataCoverageText(coverage)}`;
    }
    recResultsEl.innerHTML = "";
  } else {
    const scopeNote = view === "popularity"
      ? "Counts cover only user-anime edges retained in the loaded recommendation graph; they are not global popularity."
      : view === "quality"
        ? `Community scores are known for ${buildCommunityQualityExploration(recommendationIndex, animeMetadataCache).length}/${recommendationIndex.animeList.length} catalog titles.`
        : "Scores sum exact shared genres with Liked titles, weighted by importance and confidence.";
    recSummaryEl.textContent =
      `Showing top ${Math.min(MAX_RECOMMENDATIONS, results.length)} of ${results.length} eligible titles ` +
      `(${heading})${formatActiveFilterSummary()}. ${scopeNote} ` +
      franchiseSelectionSummary(franchiseSelection, finalEligibility.recommendations.length) +
      (metadataMatters ? ` Metadata scan ${incompleteMetadata ? "partial" : "complete"}: ${metadataCoverageText(coverage)}` : "");
    recResultsEl.innerHTML = results.slice(0, MAX_RECOMMENDATIONS)
      .map((item) => renderRecommendationCard(item, view,
        franchiseSelection.notesByAnimeId.get(item.anime.animeId))).join("");
  }
  const metadataState: AsyncUiState = demoMode ? "demo"
    : coverage.failed > 0 ? "failed"
      : metadataMatters && incompleteMetadata ? "partial"
        : coverage.unavailable > 0 ? "unavailable" : knownInScope > 0 ? "ready" : "empty";
  setMetadataStatus(metadataState,
    `Metadata available for ${knownInScope}/${metadataScopeIds.length} eligible titles in bounded exploration batches` +
      (finalEligibility.missingMetadataCount > 0
        ? ` | ${finalEligibility.missingMetadataCount} skipped by required filters until metadata is known`
        : "") + (demoMode ? " | synthetic demo catalog" : ""),
  );
}

function renderRecommendationCard(
  item: RecommendationResult,
  display: "ranking" | "coverage" | "popularity" | "quality" | "related" = "ranking",
  relationshipNote?: string,
): string {
  const metadata = animeMetadataCache.get(item.anime.animeId) ?? null;
  const metadataState = metadata ? "ready"
    : animeMetadataFailed.has(item.anime.animeId) ? "failed"
      : animeMetadataUnavailable.has(item.anime.animeId) ? "unavailable" : "not-loaded";
  const rawImageUrl = metadata?.imageUrl ?? "";
  const imageUrl = safeExternalImageUrl(rawImageUrl);
  const coverLabel = rawImageUrl && !imageUrl ? "Cover blocked"
    : metadataState === "unavailable" ? "Cover unavailable"
      : metadataState === "ready" ? "No cover available" : "Cover not loaded";
  const coverHtml =
    imageUrl
      ? `<img class="rec-cover" src="${escapeHtml(imageUrl)}" alt="Cover for ${escapeHtml(item.anime.label)}" loading="lazy" referrerpolicy="no-referrer" />`
      : `<div class="rec-cover rec-cover-placeholder" role="img" aria-label="${escapeHtml(`${coverLabel} for ${item.anime.label}`)}">${coverLabel}</div>`;
  const metadataMeta = formatRecommendationMetadataMeta(metadata, metadataState);
  const synopsisText =
    metadata && metadata.synopsis
      ? truncateText(metadata.synopsis, 180)
      : metadataState === "not-loaded" ? "Description not loaded."
        : metadataState === "failed" ? "Description failed to load."
          : "No description available.";
  const synopsisClass =
    metadata && metadata.synopsis
      ? "rec-synopsis"
      : "rec-synopsis rec-synopsis-muted";
  const related = display === "related" ? item as GenreOverlapRecommendation : null;
  const explanation = display === "ranking" ? explainRecommendation(item) : null;
  const reasonHeadline = display === "ranking" ? ""
    : display === "coverage"
      ? `Catalog coverage: ${item.supportCount} positive graph connections; personal preference evidence is unavailable.`
      : display === "popularity"
        ? `${item.supportCount} user-anime edges are retained for this title in the loaded recommendation graph.`
        : display === "quality"
          ? samplePopularityAvailable
            ? `Community score ${item.score.toFixed(2)}/10 is from available catalog metadata; ${item.supportCount} sampled rating edges were retained.`
            : `Community score ${item.score.toFixed(2)}/10 is from available catalog metadata; sampled rating counts are unavailable.`
          : `Shares ${related!.sharedGenres.join(", ")} with Liked title(s) ${related!.matchingLikedTitles.join(", ")}. ` +
            `Weighted genre-overlap sum: ${item.score.toFixed(2)}.`;
  const evidenceNote = display === "coverage" ? "Coverage only; this is not a personal ranking or quality score."
    : display === "popularity" ? "Sampled graph count only; not global popularity or personal fit."
      : display === "quality" ? bundledMetadata
        ? "Bundled catalog community opinion; not a personal prediction."
        : "Provider community opinion; not a personal prediction."
        : "Exact shared genres from Liked titles; no calibrated confidence interval.";
  const reason = explanation ? formatRecommendationWhyHtml(explanation)
    : `<div class="rec-why-line">${escapeHtml(reasonHeadline)}</div>` +
      `<div class="rec-why-line rec-evidence">${escapeHtml(evidenceNote)}</div>`;
  const score = display === "coverage" ? `${item.supportCount} connections`
    : display === "popularity" ? `${item.supportCount} sampled ratings`
      : display === "quality" ? `${item.score.toFixed(2)} / 10`
        : display === "related" ? `${item.score.toFixed(2)} overlap`
          : item.fusion ? `${item.score.toFixed(2)} rank points` : formatWeight(item.score);
  const scoreLabel = display === "coverage" ? "Catalog connections"
    : display === "popularity" ? "Sample count"
      : display === "quality" ? "Community score"
        : display === "related" ? "Genre overlap"
          : item.fusion ? "Hybrid ranking"
            : item.scoreSource?.kind === "model" ? "Model ranking" : "Graph ranking";
  const communityScore = metadata?.score !== null && metadata?.score !== undefined && display !== "quality"
    ? `<div class="rec-community-score">${demoMode ? "Demo" : bundledMetadata ? "Catalog" : "MAL"} community score: ${metadata.score.toFixed(2)}/10</div>`
    : "";
  const supportLine = display === "coverage" ? `Positive graph connections: ${item.supportCount}`
    : display === "popularity" || display === "quality"
      ? samplePopularityAvailable
        ? `Rating edges in loaded recommendation graph: ${item.supportCount}`
        : "Sampled rating counts unavailable in aggregate-only graph"
      : display === "related"
        ? `Shared genres: ${related!.sharedGenres.length} across ${related!.supportCount} liked titles`
        : item.fusion
          ? `Graph ${item.fusion.graphWeight === 0 ? "inactive" : item.fusion.graphRank === null
            ? "no candidate" : `rank #${item.fusion.graphRank}`} | Model ${item.fusion.modelWeight === 0
            ? "inactive" : item.fusion.modelRank === null ? "no candidate"
              : `rank #${item.fusion.modelRank}`} | ${explanation?.kind === "score" ? explanation.distinctSourceCount : 0} distinct source titles`
          : item.scoreSource?.kind === "graph" && explanation?.kind === "score"
            ? `${explanation.distinctSourceCount} distinct Liked sources | ${item.scoreSource.contributingEdges} positive pair edges`
          : item.scoreSource?.kind === "model" && explanation?.kind === "score"
            ? `${explanation.distinctSourceCount} distinct mapped titles | ${item.scoreSource.mappedSignals}/${item.scoreSource.suppliedSignals} signals mapped`
          : `Support edges: ${item.supportCount} | Strongest: ${formatWeight(item.strongest)}`;

  return `
      <li class="rec-item" data-metadata-state="${metadataState}">
        <div class="rec-main">
          ${coverHtml}
          <div class="rec-copy">
            <div class="rec-heading-row">
              <div class="rec-title">${escapeHtml(item.anime.label)}</div>
              <div class="rec-score-wrap"><span class="rec-score-label">${scoreLabel}</span><div class="rec-score">${score}</div></div>
            </div>
            <div class="rec-meta rec-meta-details">${escapeHtml(metadataMeta)}${metadataState === "failed"
              ? ` <button type="button" class="rec-metadata-retry" data-card-action="retry-metadata" data-anime-id="${item.anime.animeId}">Retry details</button>` : ""}</div>
            ${communityScore}
            <div class="rec-why">${reason}</div>
            <div class="rec-meta rec-support">${escapeHtml(supportLine)}</div>
            <p class="${synopsisClass}">${escapeHtml(synopsisText)}</p>
            <div class="rec-relationship">${escapeHtml(relationshipNote ?? "Prerequisites unverified; relationship data may be incomplete.")}</div>
            <div class="rec-actions" role="group" aria-label="Actions for ${escapeHtml(item.anime.label)}">
              <button type="button" class="ghost-btn rec-watchlist-save" data-card-action="save" data-anime-id="${item.anime.animeId}" data-watchlist-save-id="${item.anime.animeId}">Plan to Watch</button>
              <button type="button" class="ghost-btn" data-card-action="seen" data-anime-id="${item.anime.animeId}" aria-label="Mark ${escapeHtml(item.anime.label)} Seen">Seen</button>
              <button type="button" class="ghost-btn" data-card-action="hide" data-anime-id="${item.anime.animeId}" aria-label="Not interested in ${escapeHtml(item.anime.label)}">Not interested</button>
            </div>
          </div>
        </div>
      </li>
    `;
}

function formatRecommendationMetadataMeta(
  metadata: AnimeMetadata | null,
  state: "ready" | "not-loaded" | "failed" | "unavailable",
): string {
  if (!metadata) {
    return state === "failed" ? "Details failed to load"
      : state === "unavailable" ? bundledMetadata ? "Details unavailable in bundled catalog"
        : "Details unavailable from provider" : "Details not loaded";
  }

  const parts: string[] = [];
  if (metadata.year !== null) {
    parts.push(String(metadata.year));
  }
  if (metadata.mediaFormat) {
    parts.push(metadata.mediaFormat);
  }
  if (metadata.episodeCount !== null && metadata.episodeCount !== undefined) {
    parts.push(`${metadata.episodeCount} episodes`);
  }
  if (metadata.runtimeMinutes !== null && metadata.runtimeMinutes !== undefined) {
    parts.push(`${metadata.runtimeMinutes} min`);
  }
  if (metadata.contentClassification) {
    const rating = metadata.contentClassification;
    parts.push(`${rating.system} ${rating.value} (${rating.jurisdiction})`);
  }
  if (metadata.studios.length > 0) {
    parts.push(metadata.studios.slice(0, 2).join(", "));
  }
  if (metadata.genres.length > 0) {
    parts.push(metadata.genres.slice(0, 3).join(", "));
  }
  if (parts.length === 0) {
    return "Year, format, studios, and genres unavailable";
  }
  return parts.join(" | ");
}

function setMetadataStatus(state: AsyncUiState, message: string): void {
  setAsyncStatus(metadataStatusEl, state, message);
}

function candidateMetadataCoverage(candidates: readonly RecommendationResult[]): CandidateMetadataCoverage {
  return summarizeCandidateMetadataCoverage(
    candidates.map((item) => item.anime.animeId),
    animeMetadataCache, animeMetadataUnavailable, animeMetadataFailed,
  );
}

function unresolvedMetadataCount(coverage: CandidateMetadataCoverage): number {
  return coverage.uncheckedAnimeIds.length + coverage.failed;
}

function metadataCoverageText(coverage: CandidateMetadataCoverage): string {
  return `${coverage.ready + coverage.unavailable}/${coverage.total} resolved; ` +
    `${coverage.uncheckedAnimeIds.length} unchecked, ${coverage.failed} failed, ` +
    `${coverage.unavailable} unavailable.` +
    (hasActiveRecommendationFilters(recommendationFilters)
      ? " Required filters exclude titles without verified fields." : "");
}

function renderRankedFilterMetadataCoverage(candidates: readonly RecommendationResult[]): CandidateMetadataCoverage {
  const coverage = candidateMetadataCoverage(candidates);
  filterMetadataControlsEl.hidden = demoMode || !hasActiveRecommendationFilters(recommendationFilters) ||
    coverage.total === 0;
  if (filterMetadataControlsEl.hidden) return coverage;
  filterMetadataNoteEl.textContent = `Filter metadata: ${metadataCoverageText(coverage)}`;
  filterMetadataMoreBtn.hidden = unresolvedMetadataCount(coverage) === 0;
  filterMetadataMoreBtn.disabled = false;
  filterMetadataMoreBtn.textContent = coverage.uncheckedAnimeIds.length > 0
    ? `Check next ${Math.min(METADATA_PREFETCH_WITH_FILTER_LIMIT, coverage.uncheckedAnimeIds.length)} unchecked candidates`
    : `Retry ${Math.min(METADATA_PREFETCH_WITH_FILTER_LIMIT, coverage.failed)} failed metadata checks`;
  return coverage;
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
  for (const item of recommendations) {
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
  if (demoMode || bundledMetadata) return null;
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
    populateAnimeOptions(recommendationIndex.animeList, animeMetadataCache, animeOptions);
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
      setAsyncStatus(seasonalStatusEl, error instanceof ProviderUnavailableError ? "unavailable" : "failed",
        "Unable to load seasonal anime. Retry later; existing recommendations are unchanged.");
      recordDiagnosticIssue("SEASONAL-001");
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

function chooseAnimeFromInput(
  input: HTMLInputElement,
  status: HTMLElement,
  onChoose: (anime: AnimeInfo) => void,
): void {
  const raw = input.value.trim();
  if (!raw) {
    status.textContent = "Enter an anime title first.";
    return;
  }
  const found = searchAnimeTitles(raw, recommendationIndex, animeMetadataCache);
  if (found.automatic) {
    onChoose(found.automatic);
  } else if (!found.total) {
    status.textContent = `No catalog title or known alias found for "${raw}".`;
  } else {
    showTitleSearchChoices(raw, found, input, onChoose);
  }
}

function showTitleSearchChoices(
  raw: string,
  found: TitleSearchResult,
  input: HTMLInputElement,
  onChoose: (anime: AnimeInfo) => void,
): void {
  titleSearchResults.replaceChildren();
  titleSearchSummary.textContent = found.total > found.matches.length
    ? `Showing ${found.matches.length} of ${found.total} matches for "${raw}". Choose one or refine your search.`
    : found.total === 1
      ? `Confirm the partial match for "${raw}" before adding it.`
      : `${found.total} titles match "${raw}". Choose the intended title.`;
  for (const { anime, matchedAlias } of found.matches) {
    const metadata = animeMetadataCache.get(anime.animeId);
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.className = "title-search-choice";
    const imageUrl = safeExternalImageUrl(metadata?.imageUrl ?? "");
    if (imageUrl) {
      const cover = document.createElement("img");
      cover.src = imageUrl;
      cover.alt = "";
      cover.referrerPolicy = "no-referrer";
      cover.loading = "lazy";
      button.append(cover);
    } else {
      const placeholder = document.createElement("span");
      placeholder.className = "title-search-cover-placeholder";
      placeholder.textContent = "No cover";
      button.append(placeholder);
    }
    const copy = document.createElement("span");
    copy.className = "title-search-choice-copy";
    const heading = document.createElement("strong");
    heading.textContent = anime.label;
    copy.append(heading);
    if (matchedAlias) {
      const alias = document.createElement("span");
      alias.textContent = `Matched alias: ${matchedAlias}`;
      copy.append(alias);
    }
    const detail = document.createElement("span");
    detail.textContent = `Anime ${anime.animeId} · ${metadata?.year ?? "Year unknown"} · ${metadata?.mediaFormat ?? "Format unknown"}`;
    copy.append(detail);
    button.append(copy);
    button.addEventListener("click", () => {
      titleSearchDialog.close();
      input.focus({ preventScroll: true });
      onChoose(anime);
    });
    item.append(button);
    titleSearchResults.append(item);
  }
  titleSearchDialog.showModal();
  titleSearchResults.querySelector<HTMLButtonElement>("button")?.focus();
}

function populateAnimeOptions(
  animeList: AnimeInfo[],
  metadata: ReadonlyMap<number, AnimeMetadata>,
  datalist: HTMLDataListElement,
): void {
  const values = new Set<string>();
  for (const anime of animeList) {
    values.add(anime.label);
    for (const alias of metadata.get(anime.animeId)?.aliases ?? []) values.add(alias);
  }
  datalist.replaceChildren(...[...values].sort((left, right) => left.localeCompare(right))
    .slice(0, 8000).map((value) => {
      const option = document.createElement("option");
      option.value = value;
      return option;
    }));
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
): { match: GraphNode | null; ambiguous: boolean } {
  const raw = query.trim();
  if (!raw) {
    return { match: null, ambiguous: false };
  }

  const rawLower = raw.toLowerCase();
  const normalized = normalizeTitle(raw);
  const nodes = getGraphNodes(graphDataValue);

  const byExactId = nodes.find((node) => node.id.toLowerCase() === rawLower);
  if (byExactId) {
    return { match: byExactId, ambiguous: false };
  }

  if (/^\d+$/.test(raw)) {
    const animeNodeId = `anime:${Number.parseInt(raw, 10)}`;
    const byAnimeId = nodes.find((node) => node.id === animeNodeId);
    if (byAnimeId) {
      return { match: byAnimeId, ambiguous: false };
    }
  }
  const exact = nodes.filter((node) => normalizeTitle(node.label) === normalized);
  const prefix = exact.length ? [] : nodes.filter((node) => normalizeTitle(node.label).startsWith(normalized));
  const partial = exact.length || prefix.length ? [] : nodes.filter((node) =>
    normalizeTitle(`${node.label} ${node.id}`).includes(normalized));
  const candidates = exact.length ? exact : prefix.length ? prefix : partial;
  return { match: candidates.length === 1 ? candidates[0] : null, ambiguous: candidates.length > 1 };
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

function enterFocusedNeighborhood(nodeId: string): void {
  let centerId: string;
  let centerLabel: string;
  if (isCompactGraphData(graphData)) {
    const entry = graphData.anime.find(([animeId]) => `anime:${animeId}` === nodeId);
    if (!entry) return;
    centerId = `anime:${entry[0]}`;
    centerLabel = entry[1];
  } else {
    const entry = graphData.nodes.find((node) => node.id === nodeId && node.nodeType === "anime");
    if (!entry) return;
    centerId = entry.id;
    centerLabel = entry.label;
  }
  neighborhoodFocusId = centerId;
  selectedNodeId = centerId;
  toggleUsers.checked = false;
  toggleAnimeEdges.checked = true;
  resetGraphViewport();
  networkSearchMessage.textContent = `Focused neighborhood: ${centerLabel} (${centerId}).`;
  if (activeView !== "network") {
    setActiveView("network", false);
  } else {
    rerenderGraph();
  }
}

function focusNodeInRenderer(nodeId: string): void {
  if (!currentGraph || !currentGraph.hasNode(nodeId)) {
    return;
  }
  selectedNodeId = nodeId;
  renderInspectPanel(nodeId);
  renderSvgGraph(currentGraph);
  scrollGraphNodeIntoView(nodeId);
}

function scrollGraphNodeIntoView(nodeId: string): void {
  const target = graphContainer.querySelector<SVGElement>(
    `[data-node-id="${cssEscapeAttributeValue(nodeId)}"]`,
  );
  (target ?? graphContainer).scrollIntoView({
    block: "center",
    inline: "center",
    behavior: prefersReducedMotion() ? "auto" : "smooth",
  });
}

function rerenderGraph(): void {
  const minWeight = Number.parseFloat(minWeightInput.value);
  minWeightValue.textContent = minWeight.toFixed(2);
  const showAnimeAnimeEdges = toggleAnimeEdges.checked;
  if (neighborhoodFocusId || (isCompactGraphData(graphData) && graphData.format === "graph-compact-v3")) {
    toggleUsers.checked = false;
  }
  const showUsers = toggleUsers.checked;
  graphRenderController?.abort();
  const controller = new AbortController();
  graphRenderController = controller;
  const runId = ++graphRenderRunId;
  const renderSource = explorerGraphData ?? graphData;

  setGraphLoadingState(true, "Render status: rendering network...");

  runtime.schedule(() => {
    if (runId !== graphRenderRunId || controller.signal.aborted) return;
    if (activeView !== "network") {
      if (graphRenderController === controller) graphRenderController = null;
      setGraphLoadingState(false, "Render status: ready.");
      return;
    }

    void (async () => {
      const startedAt = runtime.monotonicNow();
      try {
        const renderResult = await measureLocalAsync("wasiw:network:render", () => renderGraph(
          renderSource, minWeight, showAnimeAnimeEdges, showUsers, controller.signal,
        ));
        if (runId !== graphRenderRunId || controller.signal.aborted || activeView !== "network") return;
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
        if (isAbortError(error) || runId !== graphRenderRunId || controller.signal.aborted) return;
        setGraphLoadingState(false, "Render status: failed. Reload and reduce visible edges.");
        recordDiagnosticIssue("RENDER-001");
        console.error("Graph render failed [RENDER-001].");
      } finally {
        if (graphRenderController === controller) graphRenderController = null;
      }
    })();
  }, 0);
}

function setGraphLoadingState(loading: boolean, statusMessage: string): void {
  graphLoadingEl.hidden = !loading;
  graphLoadingEl.setAttribute("aria-hidden", loading ? "false" : "true");
  graphShell.setAttribute("aria-busy", loading ? "true" : "false");
  graphLoadingMessageEl.textContent = loading ? "Rendering network..." : "";
  networkRenderStatusEl.textContent = statusMessage;
}

function renderVisibleNodeList(): void {
  const query = normalizeTitle(networkNodeFilterEl.value);
  const matching = query ? visibleGraphNodes.filter((node) => node.searchText.includes(query))
    : visibleGraphNodes;
  const pageCount = Math.max(1, Math.ceil(matching.length / NETWORK_NODE_LIST_PAGE_SIZE));
  networkNodeListPage = Math.min(networkNodeListPage, pageCount - 1);
  const start = networkNodeListPage * NETWORK_NODE_LIST_PAGE_SIZE;
  const visible = matching.slice(start, start + NETWORK_NODE_LIST_PAGE_SIZE);
  networkNodePrevBtn.disabled = networkNodeListPage === 0;
  networkNodeNextBtn.disabled = networkNodeListPage >= pageCount - 1;
  networkNodeListStatusEl.textContent = matching.length === 0
    ? `No nodes match this filter among ${visibleGraphNodes.length} visible nodes.`
    : `Showing ${start + 1}–${start + visible.length} of ${matching.length} matching nodes (${visibleGraphNodes.length} visible nodes).`;
  networkNodeListResultsEl.replaceChildren(...visible.map((node) => {
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.className = "network-node-choice";
    button.dataset.nodeId = node.id;
    const evidence = node.evidence;
    const evidenceText = evidence
      ? ` · ${formatWeight(evidence.weight)} pair preference · support ${evidence.support ?? "undeclared"}` : "";
    button.textContent = node.label + evidenceText;
    button.setAttribute("aria-label", `${node.label} (${node.nodeType}, ${node.id})${evidenceText}`);
    button.setAttribute("aria-pressed", node.id === selectedNodeId ? "true" : "false");
    item.append(button);
    return item;
  }));
}

function syncVisibleNodeSelection(): void {
  for (const button of networkNodeListResultsEl.querySelectorAll<HTMLButtonElement>("button[data-node-id]")) {
    button.setAttribute("aria-pressed", button.dataset.nodeId === selectedNodeId ? "true" : "false");
  }
}

async function renderGraph(
  graphDataValue: LoadedGraphData,
  minAbsoluteWeight: number,
  showAnimeAnimeEdges: boolean,
  showUsers: boolean,
  signal: AbortSignal,
): Promise<{ renderedEdgeCount: number; totalEligibleEdgeCount: number; edgeLimitHit: boolean }> {
  throwIfAborted(signal);
  const focused = neighborhoodFocusId
    ? selectAnimeNeighborhood(graphData, neighborhoodFocusId,
      NEIGHBORHOOD_MAX_NODES, NEIGHBORHOOD_MAX_EDGES, minAbsoluteWeight)
    : null;
  const cached = !focused && overviewGraphCache?.source === graphDataValue &&
    overviewGraphCache.minAbsoluteWeight === minAbsoluteWeight &&
    overviewGraphCache.showAnimeAnimeEdges === showAnimeAnimeEdges &&
    overviewGraphCache.showUsers === showUsers ? overviewGraphCache : null;
  const graph = cached?.graph ?? new Graph({ multi: true, type: "undirected" });
  const selectedEdges = cached
    ? { edges: [] as GraphEdge[], totalEligibleEdgeCount: cached.totalEligibleEdgeCount,
      edgeLimitHit: cached.edgeLimitHit }
    : measureLocal("wasiw:network:select", () => focused
      ? { edges: showAnimeAnimeEdges ? focused.edges : [],
        totalEligibleEdgeCount: showAnimeAnimeEdges ? focused.eligiblePairEdges : 0,
        edgeLimitHit: showAnimeAnimeEdges && focused.omittedByBudget > 0 }
      : selectRenderableEdges(graphDataValue, minAbsoluteWeight,
        showAnimeAnimeEdges, showUsers));
  const chunkBuild = selectedEdges.edges.length > CHUNKED_NETWORK_EDGE_THRESHOLD;
  const activeNodeIds = new Set<string>();
  if (!cached) await measureLocalAsync("wasiw:network:construct", async () => {
    if (focused) activeNodeIds.add(focused.center.id);

    for (const edge of selectedEdges.edges) {
      activeNodeIds.add(edge.source);
      activeNodeIds.add(edge.target);
    }

    const nodes = focused ? focused.nodes : getGraphNodes(graphDataValue);
    let nodeBatchStartedAt = performance.now();
    for (let index = 0; index < nodes.length; index += 1) {
      const node = nodes[index];
      if ((showUsers || node.nodeType !== "user") && activeNodeIds.has(node.id)) {
        const isUser = node.nodeType === "user";
        graph.addNode(node.id, {
          label: node.label,
          nodeType: node.nodeType,
          size: isUser ? 5.2 : focused ? node.id === focused.center.id ? 8 : 5 : 2.8,
          color: isUser ? "#ff8a00" : "#0f8b8d",
          x: runtime.random(),
          y: runtime.random(),
        });
      }
      if (chunkBuild && (index + 1) % NETWORK_BUILD_BATCH_SIZE === 0 && index + 1 < nodes.length) {
        recordLocalDuration("wasiw:network:construct-node-batch", nodeBatchStartedAt);
        await runtime.yieldMainThread(signal);
        nodeBatchStartedAt = performance.now();
      }
    }
    recordLocalDuration("wasiw:network:construct-node-batch", nodeBatchStartedAt);

    let edgeBatchStartedAt = performance.now();
    for (let index = 0; index < selectedEdges.edges.length; index += 1) {
      const edge = selectedEdges.edges[index];
      if (graph.hasNode(edge.source) && graph.hasNode(edge.target)) {
        const sign = edge.weight > 0 ? "positive" : edge.weight < 0 ? "negative" : "neutral";
        const color = sign === "neutral" ? "var(--graph-neutral-edge)"
          : edge.edgeType === "user-anime"
            ? sign === "positive" ? "var(--graph-user-positive-edge)" : "var(--graph-user-negative-edge)"
            : sign === "positive" ? "var(--graph-positive-edge)" : "var(--graph-negative-edge)";
        graph.addEdgeWithKey(edge.id, edge.source, edge.target, {
          size: focused ? 1.6 : edge.edgeType === "user-anime" ? 1.4 : 0.7,
          color,
          weight: Math.max(Math.abs(edge.weight), 0.01),
          signedWeight: edge.weight,
          sign,
          edgeType: edge.edgeType,
        });
      }
      if (chunkBuild && (index + 1) % NETWORK_BUILD_BATCH_SIZE === 0 &&
          index + 1 < selectedEdges.edges.length) {
        recordLocalDuration("wasiw:network:construct-edge-batch", edgeBatchStartedAt);
        await runtime.yieldMainThread(signal);
        edgeBatchStartedAt = performance.now();
      }
    }
    recordLocalDuration("wasiw:network:construct-edge-batch", edgeBatchStartedAt);
  });

  throwIfAborted(signal);
  if (chunkBuild) {
    await runtime.yieldMainThread(signal);
    throwIfAborted(signal);
  }
  if (!cached) {
    measureLocal("wasiw:network:layout", () => applyLayout(graph, focused?.center.id));
    if (!focused && graph.size > CHUNKED_NETWORK_EDGE_THRESHOLD) {
      overviewGraphCache = { source: graphDataValue, minAbsoluteWeight,
        showAnimeAnimeEdges, showUsers, graph, svg: null,
        totalEligibleEdgeCount: selectedEdges.totalEligibleEdgeCount,
        edgeLimitHit: selectedEdges.edgeLimitHit };
    }
  }
  const nextVisibleGraphNodes: typeof visibleGraphNodes = [];
  const focusedEvidence = new Map<string, { weight: number; support?: number }>();
  const focusedOrder = new Map<string, number>();
  if (focused) {
    focusedOrder.set(focused.center.id, 0);
    focused.edges.forEach((edge, index) => {
      const neighbor = edge.source === focused.center.id ? edge.target : edge.source;
      focusedEvidence.set(neighbor, { weight: edge.weight, support: edge.support });
      focusedOrder.set(neighbor, index + 1);
    });
  }
  measureLocal("wasiw:network:visible-node-map", () => graph.forEachNode((id, attributes) => {
    const label = typeof attributes.label === "string" ? attributes.label : id;
    nextVisibleGraphNodes.push({
      id,
      label,
      nodeType: attributes.nodeType === "user" ? "user" : "anime",
      searchText: normalizeTitle(`${label} ${id}`),
      evidence: focusedEvidence.get(id),
    });
  }));
  measureLocal("wasiw:network:visible-node-sort", () => nextVisibleGraphNodes.sort((left, right) =>
    (focused ? (focusedOrder.get(left.id) ?? Number.MAX_SAFE_INTEGER) -
      (focusedOrder.get(right.id) ?? Number.MAX_SAFE_INTEGER) : 0) ||
    left.label.localeCompare(right.label) || left.id.localeCompare(right.id)));
  if (chunkBuild) await runtime.yieldMainThread(signal);
  throwIfAborted(signal);
  currentGraph = graph;
  visibleGraphNodes = nextVisibleGraphNodes;
  measureLocal("wasiw:network:visible-node-list", () => renderVisibleNodeList());

  measureLocal("wasiw:network:control-update", () => {
    const visibleUsers = countNodesByType(graph, "user");
    const visibleAnime = graph.order - visibleUsers;
    const scope = measureLocal("wasiw:network:scope", () =>
      updateNetworkScope(explorerGraphData ?? graphData));
    const countSource = focused ? graphData : graphDataValue;
    if (focused) {
      const budgetNote = focused.omittedByBudget > 0
        ? ` ${focused.omittedByBudget} matching pair edges omitted by the 25-node/24-edge focus budget.`
        : "";
      const noEvidenceNote = focused.eligiblePairEdges === 0
        ? " No retained pair edge meets this filter; source selection can omit other relationships."
        : "";
      networkModeStatusEl.textContent = `Focused on ${focused.center.label} (${focused.center.id}): ` +
        `${graph.size}/${focused.eligiblePairEdges} retained signed pair edges shown from the recommendation graph.` +
        budgetNote + noEvidenceNote +
        (showAnimeAnimeEdges ? "" : " Pair edges are hidden by the visibility control.");
    } else {
      networkModeStatusEl.textContent = `Explorer sample overview: ${graph.order} nodes and ${graph.size} edges visible. ` +
        "Search or select an anime to inspect its bounded recommendation-graph neighborhood.";
    }

    statsEl.innerHTML = [
      statLine("Generated", new Date(countSource.generatedAt).toLocaleString()),
      statLine(scope.userRowsOmitted ? "User rows" : "Visible users",
        scope.userRowsOmitted ? "omitted from v3" : `${visibleUsers} / ${countSource.userCount}`),
      statLine("Visible anime", `${visibleAnime} / ${countSource.animeCount}`),
      statLine("Visible nodes", String(graph.order)),
      statLine("Visible edges", String(graph.size)),
    ].join("");

    if (selectedNodeId && graph.hasNode(selectedNodeId)) {
      renderInspectPanel(selectedNodeId);
    } else {
      if (selectedNodeId) {
        networkSearchMessage.textContent = "That node is in the recommendation graph but not in the current explorer sample or filter. Its absence here does not prove no relationship.";
      }
      selectedNodeId = null;
      renderInspectPanel(null);
    }
  });
  if (chunkBuild) await runtime.yieldMainThread(signal);
  throwIfAborted(signal);
  await measureLocalAsync("wasiw:network:svg", () => renderSvgGraph(graph, signal));

  return {
    renderedEdgeCount: graph.size,
    totalEligibleEdgeCount: selectedEdges.totalEligibleEdgeCount,
    edgeLimitHit: selectedEdges.edgeLimitHit,
  };
}

function updateNetworkScope(explorer: LoadedGraphData) {
  const scope = describeGraphScope(graphData, explorer,
    MAX_RENDERED_ANIME_ANIME_EDGES, MAX_RENDERED_USER_ANIME_EDGES);
  const modelStatus = diagnosticModelEl.textContent || "Model status unavailable";
  const expectedDemoModel = demoMode && modelStatus === "Synthetic model not checked"
    ? " (expected format model-mf-compact-v1 when requested)" : "";
  networkVersionsEl.textContent = `${scope.versions} Model: ${modelStatus}${expectedDemoModel}.`;
  networkSelectionEl.textContent = scope.selection;
  networkExplorerSampleEl.textContent = scope.explorer;
  networkDrawingLimitsEl.textContent = scope.limits;
  networkScopeCaveatEl.textContent = scope.caveat;
  toggleUsers.disabled = scope.userRowsOmitted || neighborhoodFocusId !== null;
  toggleUsersLabelEl.textContent = neighborhoodFocusId !== null
    ? "User edges are available only in the explorer overview"
    : scope.userRowsOmitted
    ? "User rows omitted from this aggregate-only graph"
    : "Show sampled user nodes + user-anime edges";
  networkSearchInput.placeholder = scope.userRowsOmitted
    ? "Title or anime:ID" : "Title, anime:ID, or user:ID";
  networkSearchInput.setAttribute("aria-label", scope.userRowsOmitted
    ? "Search loaded anime nodes" : "Search loaded anime or user nodes");
  return scope;
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
    modelRecommendationIndexPromise = artifactLoader.fetchModelRecommendationIndex()
      .then((model) => {
        diagnosticModelEl.textContent = modelVersionLabel(activeReleaseManifest,
          artifactLoader.getLoadedModelFormat(), model ? "loaded" : "absent", demoMode);
        if (explorerGraphData) updateNetworkScope(explorerGraphData);
        if (!model && (!activeReleaseManifest || activeReleaseManifest.model)) {
          recordDiagnosticIssue("MODEL-001");
        }
        return model;
      })
      .catch((error: unknown) => {
        modelLoadError = error instanceof ArtifactValidationError || error instanceof ArtifactLoadError
          ? error.message
          : "model-mf-web.compact.json: unable to load or verify the optional asset.";
        diagnosticModelEl.textContent = modelVersionLabel(activeReleaseManifest, null, "failed", demoMode);
        if (explorerGraphData) updateNetworkScope(explorerGraphData);
        recordDiagnosticIssue("MODEL-001");
        console.error("Model artifact load failed [MODEL-001].");
        return null;
      });
  }
  return modelRecommendationIndexPromise;
}

async function loadRequiredArtifact<T>(operation: Promise<T>): Promise<T> {
  try {
    return await operation;
  } catch (error) {
    revealApp();
    const authoredError = error instanceof ArtifactValidationError || error instanceof ArtifactLoadError;
    const detail = authoredError ? error.message : "unable to load or verify the required asset";
    recMessageEl.textContent = `Data unavailable: ${detail}`;
    recMessageEl.setAttribute("role", "alert");
    recEngineStatusEl.textContent = "Recommendations unavailable until the data artifact is repaired.";
    diagnosticDataEl.textContent = "Unavailable; no verified data identity";
    recordDiagnosticIssue(error instanceof ArtifactValidationError || /SHA-256|byte length/.test(detail)
      ? "DATA-002" : "DATA-001");
    throw authoredError ? error : new Error("Required data artifact unavailable.");
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

function applyLayout(graph: Graph, centerId?: string): void {
  if (graph.order === 0) {
    return;
  }

  try {
    if (centerId && graph.hasNode(centerId)) {
      assignNeighborhoodLayout(graph, centerId);
    } else {
      assignRingLayout(graph);
    }
  } catch {
    console.warn("Graph layout failed; using fallback ring layout.");
    assignRingLayout(graph);
  }

  sanitizeCoordinates(graph);
}

function setGraphViewport(x: number, y: number, width: number): void {
  const boundedWidth = Math.max(300, Math.min(GRAPH_VIEW_WIDTH, width));
  const height = boundedWidth * GRAPH_VIEW_HEIGHT / GRAPH_VIEW_WIDTH;
  graphViewport = {
    x: Math.max(0, Math.min(GRAPH_VIEW_WIDTH - boundedWidth, x)),
    y: Math.max(0, Math.min(GRAPH_VIEW_HEIGHT - height, y)),
    width: boundedWidth, height,
  };
  const svg = graphContainer.querySelector<SVGSVGElement>("svg.graph-svg");
  if (svg) {
    svg.setAttribute("viewBox", `${graphViewport.x} ${graphViewport.y} ${graphViewport.width} ${graphViewport.height}`);
    svg.style.touchAction = graphViewport.width < GRAPH_VIEW_WIDTH ? "none" : "auto";
  }
  graphZoomInBtn.disabled = graphViewport.width <= 300;
  graphZoomOutBtn.disabled = graphViewport.width >= GRAPH_VIEW_WIDTH;
}

function resetGraphViewport(): void {
  setGraphViewport(0, 0, GRAPH_VIEW_WIDTH);
}

function zoomGraphViewport(factor: number): void {
  const centerX = graphViewport.x + graphViewport.width / 2;
  const centerY = graphViewport.y + graphViewport.height / 2;
  const width = Math.max(300, Math.min(GRAPH_VIEW_WIDTH, graphViewport.width * factor));
  const height = width * GRAPH_VIEW_HEIGHT / GRAPH_VIEW_WIDTH;
  setGraphViewport(centerX - width / 2, centerY - height / 2, width);
}

function panGraphViewport(dx: number, dy: number): void {
  setGraphViewport(graphViewport.x + dx, graphViewport.y + dy, graphViewport.width);
}

async function renderSvgGraph(graph: Graph, signal?: AbortSignal): Promise<void> {
  const svgRunId = ++svgRenderRunId;
  throwIfAborted(signal);
  if (graph.order === 0) {
    graphContainer.replaceChildren();
    syncVisibleNodeSelection();
    return;
  }

  const cachedSvg = selectedNodeId === null && !neighborhoodFocusId &&
    overviewGraphCache?.graph === graph ? overviewGraphCache.svg : null;
  if (cachedSvg) {
    cachedSvg.setAttribute("viewBox", `${graphViewport.x} ${graphViewport.y} ` +
      `${graphViewport.width} ${graphViewport.height}`);
    cachedSvg.style.touchAction = graphViewport.width < GRAPH_VIEW_WIDTH ? "none" : "auto";
    graphContainer.replaceChildren(cachedSvg);
    syncVisibleNodeSelection();
    return;
  }

  const width = GRAPH_VIEW_WIDTH;
  const height = GRAPH_VIEW_HEIGHT;
  const padding = 56;
  const coords = new Map<string, { x: number; y: number }>();
  let minX = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;

  measureLocal("wasiw:network:svg-coordinates", () => graph.forEachNode((node, attributes) => {
    const x = Number(attributes.x);
    const y = Number(attributes.y);
    minX = Math.min(minX, x);
    maxX = Math.max(maxX, x);
    minY = Math.min(minY, y);
    maxY = Math.max(maxY, y);
    coords.set(node, { x, y });
  }));

  const spanX = maxX - minX;
  const spanY = maxY - minY;
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("viewBox", `${graphViewport.x} ${graphViewport.y} ${graphViewport.width} ${graphViewport.height}`);
  svg.style.touchAction = graphViewport.width < GRAPH_VIEW_WIDTH ? "none" : "auto";
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

  const batchEdges = graph.size > BATCHED_SVG_EDGE_THRESHOLD;
  const batchNodes = graph.order > BATCHED_SVG_NODE_THRESHOLD && !neighborhoodFocusId;
  const edgePaths = new Map<string, {
    color: string; width: number; sign: string; dashed: boolean;
    segments: string[];
  }>();
  const roundedPathCoordinate = (value: number) => Math.round(value * 10) / 10;
  const edgeKeys = graph.edges();
  const yieldSvgBuild = signal !== undefined && batchEdges;
  let edgeBatchStartedAt = performance.now();
  for (let edgeIndex = 0; edgeIndex < edgeKeys.length; edgeIndex += 1) {
    if (yieldSvgBuild && edgeIndex > 0 && edgeIndex % NETWORK_SVG_BATCH_SIZE === 0) {
      recordLocalDuration("wasiw:network:svg-edge-batch", edgeBatchStartedAt);
      await runtime.yieldMainThread(signal);
      throwIfAborted(signal);
      if (svgRunId !== svgRenderRunId) return;
      edgeBatchStartedAt = performance.now();
    }
    const edgeKey = edgeKeys[edgeIndex];
    const attributes = graph.getEdgeAttributes(edgeKey);
    const [source, target] = graph.extremities(edgeKey);
    const sourceCoord = coords.get(source);
    const targetCoord = coords.get(target);
    if (!sourceCoord || !targetCoord) {
      continue;
    }
    const x1 = scaleGraphCoordinate(sourceCoord.x, minX, spanX, padding, width);
    const y1 = scaleGraphCoordinate(sourceCoord.y, minY, spanY, padding, height);
    const x2 = scaleGraphCoordinate(targetCoord.x, minX, spanX, padding, width);
    const y2 = scaleGraphCoordinate(targetCoord.y, minY, spanY, padding, height);
    const edgeAttrs = attributes as Record<string, unknown>;
    const baseColor =
      typeof edgeAttrs.color === "string" ? edgeAttrs.color : "#6fffe944";
    const baseSize = Number(edgeAttrs.size) || 1;
    const highlighted =
      selected !== null && (source === selected || target === selected);
    const color = selected && !highlighted ? "var(--graph-dim-edge)" : baseColor;
    const strokeWidth = highlighted ? baseSize * 1.3 : baseSize;
    const sign = typeof edgeAttrs.sign === "string" ? edgeAttrs.sign : "";
    const dashed = sign === "neutral";
    if (batchEdges) {
      const key = JSON.stringify([color, strokeWidth, sign, dashed]);
      let group = edgePaths.get(key);
      if (!group) {
        group = { color, width: strokeWidth, sign, dashed, segments: [] };
        edgePaths.set(key, group);
      }
      // A tenth of a viewBox unit is below one screen pixel even at phone width.
      group.segments.push(`M${roundedPathCoordinate(x1)} ${roundedPathCoordinate(y1)}` +
        `L${roundedPathCoordinate(x2)} ${roundedPathCoordinate(y2)}`);
      continue;
    }

    const line = document.createElementNS(SVG_NS, "line");
    line.setAttribute("x1", String(x1));
    line.setAttribute("y1", String(y1));
    line.setAttribute("x2", String(x2));
    line.setAttribute("y2", String(y2));
    line.setAttribute("stroke", color);
    line.setAttribute("stroke-width", String(strokeWidth));
    line.setAttribute("stroke-linecap", "round");
    if (sign === "positive" || sign === "negative" || sign === "neutral") {
      line.setAttribute("data-edge-sign", sign);
      if (dashed) line.setAttribute("stroke-dasharray", "3 3");
    }
    edgeLayer.appendChild(line);
  }
  recordLocalDuration("wasiw:network:svg-edge-batch", edgeBatchStartedAt);
  if (batchEdges) {
    measureLocal("wasiw:network:svg-paths", () => {
      edgeLayer.setAttribute("data-render-mode", "batched-paths");
      for (const group of edgePaths.values()) {
        const path = document.createElementNS(SVG_NS, "path");
        path.setAttribute("d", group.segments.join(""));
        path.setAttribute("fill", "none");
        path.setAttribute("stroke", group.color);
        path.setAttribute("stroke-width", String(group.width));
        path.setAttribute("stroke-linecap", "round");
        path.setAttribute("data-edge-count", String(group.segments.length));
        if (group.sign === "positive" || group.sign === "negative" || group.sign === "neutral") {
          path.setAttribute("data-edge-sign", group.sign);
        }
        if (group.dashed) path.setAttribute("stroke-dasharray", "3 3");
        edgeLayer.appendChild(path);
      }
    });
  }

  const nodeKeys = graph.nodes();
  const nodePaths = new Map<string, {
    fill: string; priority: number; segments: string[];
  }>();
  const nodeHitTargets: Array<{ id: string; x: number; y: number; radius: number }> = [];
  let nodeBatchStartedAt = performance.now();
  for (let nodeIndex = 0; nodeIndex < nodeKeys.length; nodeIndex += 1) {
    if (yieldSvgBuild && nodeIndex > 0 && nodeIndex % NETWORK_SVG_BATCH_SIZE === 0) {
      recordLocalDuration("wasiw:network:svg-node-batch", nodeBatchStartedAt);
      await runtime.yieldMainThread(signal);
      throwIfAborted(signal);
      if (svgRunId !== svgRenderRunId) return;
      nodeBatchStartedAt = performance.now();
    }
    const node = nodeKeys[nodeIndex];
    const attributes = graph.getNodeAttributes(node);
    const coord = coords.get(node);
    if (!coord) {
      continue;
    }

    const x = scaleGraphCoordinate(coord.x, minX, spanX, padding, width);
    const y = scaleGraphCoordinate(coord.y, minY, spanY, padding, height);
    const nodeAttrs = attributes as Record<string, unknown>;
    const baseColor = typeof nodeAttrs.color === "string" ? nodeAttrs.color : "#0f8b8d";
    const baseSize = Number(nodeAttrs.size) || 3;
    const isSelected = selected === node;
    const isConnected = selected !== null && connectedToSelected.has(node);
    const dimmed = selected !== null && !isSelected && !isConnected;

    const radius = isSelected ? baseSize * 1.5 : baseSize;
    const fill = isSelected ? "var(--graph-selected-node)" : dimmed ? "var(--graph-dim-node)" : baseColor;
    if (batchNodes) {
      nodeHitTargets.push({ id: node, x, y, radius });
      const priority = isSelected ? 2 : isConnected ? 1 : 0;
      const key = JSON.stringify([fill, radius, priority]);
      let group = nodePaths.get(key);
      if (!group) {
        group = { fill, priority, segments: [] };
        nodePaths.set(key, group);
      }
      const circleDiameter = roundedPathCoordinate(radius * 2);
      group.segments.push(`M${roundedPathCoordinate(x - radius)} ${roundedPathCoordinate(y)}` +
        `a${roundedPathCoordinate(radius)} ${roundedPathCoordinate(radius)} 0 1 0 ${circleDiameter} 0` +
        `a${roundedPathCoordinate(radius)} ${roundedPathCoordinate(radius)} 0 1 0 -${circleDiameter} 0`);
    } else {
      const circle = document.createElementNS(SVG_NS, "circle");
      circle.setAttribute("cx", String(x));
      circle.setAttribute("cy", String(y));
      circle.setAttribute("r", String(radius));
      circle.setAttribute("fill", fill);
      circle.setAttribute("data-node-id", node);
      circle.setAttribute("aria-hidden", "true");
      circle.addEventListener("click", () => {
        selectedNodeId = node;
        networkSearchMessage.textContent = `Focused: ${String(nodeAttrs.label ?? node)} (${node})`;
        renderInspectPanel(node);
        renderSvgGraph(graph);
      });
      nodeLayer.appendChild(circle);
    }

    if (isSelected || (neighborhoodFocusId ? nodeIndex < 13 : graph.order <= 180)) {
      const label = document.createElementNS(SVG_NS, "text");
      label.setAttribute("x", String(x + 8));
      label.setAttribute("y", String(y - 8));
      label.setAttribute("fill", dimmed ? "var(--muted)" : "var(--text)");
      label.setAttribute("font-size", isSelected ? "19" : neighborhoodFocusId ? "15" : "12");
      label.setAttribute("font-family", "IBM Plex Mono, monospace");
      label.textContent = String(nodeAttrs.label ?? node);
      labelLayer.appendChild(label);
    }
  }
  recordLocalDuration("wasiw:network:svg-node-batch", nodeBatchStartedAt);
  if (batchNodes) {
    measureLocal("wasiw:network:svg-node-paths", () => {
      nodeLayer.setAttribute("data-render-mode", "batched-paths");
      for (const group of [...nodePaths.values()].sort((left, right) => left.priority - right.priority)) {
        const path = document.createElementNS(SVG_NS, "path");
        path.setAttribute("d", group.segments.join(""));
        path.setAttribute("fill", group.fill);
        path.setAttribute("data-node-count", String(group.segments.length));
        path.setAttribute("aria-hidden", "true");
        nodeLayer.appendChild(path);
      }
    });
  }

  let drag: { pointerId: number; clientX: number; clientY: number;
    x: number; y: number; width: number; height: number; moved: boolean } | null = null;
  let suppressClearClick = false;
  svg.addEventListener("pointerdown", (event) => {
    if (event.button !== 0 ||
        (event.pointerType === "touch" && graphViewport.width >= GRAPH_VIEW_WIDTH) ||
        (event.target instanceof Element && event.target.closest("circle[data-node-id]"))) return;
    suppressClearClick = false;
    drag = { pointerId: event.pointerId, clientX: event.clientX, clientY: event.clientY,
      ...graphViewport, moved: false };
    svg.setPointerCapture(event.pointerId);
  });
  svg.addEventListener("pointermove", (event) => {
    if (!drag || event.pointerId !== drag.pointerId) return;
    const rect = svg.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;
    const pixelX = event.clientX - drag.clientX;
    const pixelY = event.clientY - drag.clientY;
    if (Math.abs(pixelX) + Math.abs(pixelY) > 4) drag.moved = true;
    setGraphViewport(drag.x - pixelX * drag.width / rect.width,
      drag.y - pixelY * drag.height / rect.height, drag.width);
  });
  const finishDrag = (event: PointerEvent) => {
    if (!drag || event.pointerId !== drag.pointerId) return;
    suppressClearClick = drag.moved;
    drag = null;
    if (svg.hasPointerCapture(event.pointerId)) svg.releasePointerCapture(event.pointerId);
  };
  svg.addEventListener("pointerup", finishDrag);
  svg.addEventListener("pointercancel", finishDrag);
  svg.addEventListener("click", (event) => {
    if (suppressClearClick) {
      suppressClearClick = false;
      return;
    }
    if (batchNodes) {
      const matrix = svg.getScreenCTM();
      if (matrix) {
        const point = svg.createSVGPoint();
        point.x = event.clientX;
        point.y = event.clientY;
        const local = point.matrixTransform(matrix.inverse());
        let nearest: { id: string; distanceSquared: number } | null = null;
        for (const candidate of nodeHitTargets) {
          const dx = local.x - candidate.x;
          const dy = local.y - candidate.y;
          const distanceSquared = dx * dx + dy * dy;
          const hitRadius = Math.max(14, candidate.radius + 8);
          if (distanceSquared <= hitRadius * hitRadius &&
              (!nearest || distanceSquared < nearest.distanceSquared)) {
            nearest = { id: candidate.id, distanceSquared };
          }
        }
        if (nearest) {
          selectedNodeId = nearest.id;
          const attrs = graph.getNodeAttributes(nearest.id) as Record<string, unknown>;
          networkSearchMessage.textContent = `Focused: ${String(attrs.label ?? nearest.id)} (${nearest.id})`;
          renderInspectPanel(nearest.id);
          renderSvgGraph(graph);
          return;
        }
      }
      selectedNodeId = null;
      networkSearchMessage.textContent = "";
      renderInspectPanel(null);
      renderSvgGraph(graph);
      return;
    }
    if (event.target === svg || event.target === edgeLayer) {
      selectedNodeId = null;
      networkSearchMessage.textContent = "";
      renderInspectPanel(null);
      renderSvgGraph(graph);
    }
  });

  throwIfAborted(signal);
  if (svgRunId !== svgRenderRunId) return;
  measureLocal("wasiw:network:svg-dom-commit", () => {
    svg.append(edgeLayer, nodeLayer, labelLayer);
    graphContainer.replaceChildren(svg);
    if (selectedNodeId === null && !neighborhoodFocusId && overviewGraphCache?.graph === graph) {
      overviewGraphCache.svg = svg;
    }
    syncVisibleNodeSelection();
  });
}

function scaleGraphCoordinate(
  value: number,
  min: number,
  span: number,
  padding: number,
  extent: number,
): number {
  if (span <= 0) return extent / 2;
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

function assignNeighborhoodLayout(graph: Graph, centerId: string): void {
  graph.mergeNodeAttributes(centerId, { x: 0, y: 0 });
  const neighbors = graph.neighbors(centerId);
  neighbors.forEach((node, index) => {
    const angle = (index / Math.max(neighbors.length, 1)) * Math.PI * 2 - Math.PI / 2;
    graph.mergeNodeAttributes(node, { x: Math.cos(angle), y: Math.sin(angle) });
  });
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
    focusNeighborhoodBtn.disabled = true;
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
  focusNeighborhoodBtn.disabled = nodeType !== "anime";

  const connections = getConnectedItems(currentGraph, nodeId).sort(
    (left, right) => neighborhoodFocusId
      ? Math.abs(right.weight) - Math.abs(left.weight) || right.weight - left.weight
      : right.weight - left.weight,
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

  inspectCountEl.textContent = `Connected items: ${connectionCount} ` +
    (neighborhoodFocusId ? "(sorted by absolute signed weight desc)" : "(sorted by weight desc)");
  inspectValuesEl.innerHTML = [
    valueRow("Connected users", String(userConnections)),
    valueRow("Connected anime", String(animeConnections)),
    valueRow("Positive edges", String(positiveConnections)),
    valueRow("Negative edges", String(negativeConnections)),
    valueRow("Average weight", formatWeight(avgWeight)),
    valueRow(neighborhoodFocusId ? "Largest absolute edge" : "Strongest edge", formatWeight(strongestWeight)),
    valueRow(neighborhoodFocusId ? "Smallest absolute edge" : "Weakest edge", formatWeight(weakestWeight)),
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

function formatRecommendationWhyHtml(explanation: RecommendationExplanation): string {
  if (explanation.kind === "none") {
    return `<div class="rec-why-line">${escapeHtml(explanation.headline)}</div>`;
  }
  if (explanation.kind === "qualitative") {
    return `<div class="rec-why-line">${escapeHtml(explanation.headline)}</div>` +
      `<div class="rec-why-line">${escapeHtml(explanation.uncertainty)}</div>`;
  }
  const equation = formatScoreEquation(explanation);
  return `<div class="rec-why-line">${escapeHtml(explanation.headline)}</div>` +
    `<div class="rec-why-line">${escapeHtml(explanation.uncertainty)}</div>` +
    `<details class="rec-score-breakdown"><summary>How the score was calculated</summary>` +
    `<div class="rec-score-equation">${escapeHtml(equation.line)}</div>` +
    explanation.detailLines.map((line) => `<div class="rec-score-detail">${escapeHtml(line)}</div>`).join("") +
    (equation.roundingAdjustmentUnits !== 0
      ? `<div class="rec-score-detail">The display rounding adjustment reconciles independently rounded terms with the shown score.</div>`
      : "") +
    `</details>`;
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
  const warnings = persistence.getStorageWarnings();
  storageStatusEl.textContent = warnings.join(" ");
  if (warnings.length > 0) recordDiagnosticIssue("STORAGE-001");
}

function persistRecommendationState(): boolean {
  recActionStatusEl.textContent = "";
  cancelActiveUsernameImport();
  cancelBulkFileLoad();
  clearPendingHistoryImport();
  invalidatePendingProfileBackup();
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
    allowRelatedTitles,
    includeCandidates: [...includeCandidateNodeIds],
    excludeCandidates: [...excludeCandidateNodeIds],
    history: [...historyEntries],
    watchlist: [...watchlistEntries],
  };
}

function profileMemoryRevision(): string {
  return JSON.stringify({
    state: buildCurrentRecommendationState(),
    profiles: [...savedProfiles.entries()],
  });
}

function clearPendingProfileBackup(): void {
  ++profileBackupFileLoadId;
  activeProfileBackupFileLoadId = null;
  pendingProfileBackup = null;
  profileBackupFile.value = "";
  profileBackupPreviewEl.hidden = true;
}

function invalidatePendingProfileBackup(): void {
  if (pendingProfileBackup === null && activeProfileBackupFileLoadId === null) return;
  clearPendingProfileBackup();
  profileBackupStatusEl.textContent = "Profile data changed. Choose the backup again to review a fresh preview.";
}

function downloadProfileBackup(): void {
  try {
    const raw = persistence.createProfileBackup(buildCurrentRecommendationState(), savedProfiles);
    const url = URL.createObjectURL(new Blob([raw], { type: "application/json" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = `wasiw-profile-backup-${runtime.now().toISOString().slice(0, 10)}.json`;
    document.body.append(link);
    link.click();
    link.remove();
    runtime.schedule(() => URL.revokeObjectURL(url), 1000);
    profileBackupStatusEl.textContent = "Profile backup prepared for download. Keep the file private; it contains watch history.";
  } catch {
    profileBackupStatusEl.textContent = "Unable to prepare this profile backup. Local data was not changed.";
  }
}

async function loadProfileBackupFile(): Promise<void> {
  const file = profileBackupFile.files?.[0];
  if (!file) return;
  clearPendingProfileBackup();
  const loadId = ++profileBackupFileLoadId;
  activeProfileBackupFileLoadId = loadId;
  if (!file.name.toLowerCase().endsWith(".json") || file.size > MAX_PROFILE_BACKUP_BYTES) {
    activeProfileBackupFileLoadId = null;
    profileBackupStatusEl.textContent = "Choose a .json profile backup no larger than 8 MiB. Local data is unchanged.";
    return;
  }
  profileBackupStatusEl.textContent = "Reading local profile backup...";
  try {
    const raw = await file.text();
    if (activeProfileBackupFileLoadId !== loadId) return;
    const document = persistence.parseProfileBackup(raw);
    const storageRevision = persistence.getProfileStorageRevision();
    if (storageRevision === null) throw new Error("Storage unavailable");
    activeProfileBackupFileLoadId = null;
    pendingProfileBackup = {
      kind: "import", document, storageRevision, memoryRevision: profileMemoryRevision(),
    };
    profileBackupModeEl.value = "merge";
    renderProfileBackupPreview();
    if (pendingProfileBackup !== null) {
      profileBackupStatusEl.textContent = "Local backup parsed. Review the counts and choose merge or replace before applying.";
    }
  } catch {
    if (activeProfileBackupFileLoadId !== loadId) return;
    clearPendingProfileBackup();
    profileBackupStatusEl.textContent = "Profile backup is unreadable, unsupported, or invalid. Local data is unchanged.";
  }
}

function renderProfileBackupPreview(): void {
  const pending = pendingProfileBackup;
  if (!pending) return;
  profileBackupPreviewEl.hidden = false;
  if (pending.kind === "reset") {
    profileBackupPreviewTitleEl.textContent = "Local Reset Preview";
    profileBackupModeRowEl.hidden = true;
    profileBackupApplyBtn.textContent = "Reset Local Profiles";
    profileBackupSummaryEl.textContent =
      `Reset will clear ${selectedAnimeNodeIds.length} current preferences, ${historyEntries.length} imported history entries, ${watchlistEntries.length} watchlist titles, ` +
      `${savedProfiles.size} named profiles, and current candidate overrides. Older source keys and raw backups remain in browser storage.`;
    profileBackupUnmappedEl.textContent = persistence.getStorageWarnings().length > 0
      ? "Unreadable browser copies may also exist. Reset archives them when storage permits. Download a readable backup first if possible."
      : "Download a profile backup first if you may want to restore these choices.";
    return;
  }
  if (!pending.document) return;
  const mode: ProfileBackupImportMode = profileBackupModeEl.value === "replace" ? "replace" : "merge";
  let plan;
  try {
    plan = persistence.planProfileBackupImport(
      pending.document, buildCurrentRecommendationState(), savedProfiles, mode,
    );
  } catch {
    clearPendingProfileBackup();
    profileBackupStatusEl.textContent = "This backup cannot be combined with current browser data. Local data is unchanged.";
    return;
  }
  const c = plan.counts;
  profileBackupPreviewTitleEl.textContent = "Profile Backup Preview";
  profileBackupModeRowEl.hidden = false;
  profileBackupApplyBtn.textContent = "Apply Profile Backup";
  profileBackupSummaryEl.textContent = mode === "merge"
    ? `Backup has ${c.importedPreferences} preferences, ${c.importedHistory} history entries, ${c.importedWatchlist} watchlist titles, and ${c.importedProfiles} named profiles. ` +
      `Merge adds ${c.addedPreferences} preferences, ${c.addedHistory} history entries, ${c.addedWatchlist} watchlist titles, and ${c.addedProfiles} profiles; ` +
      `keeps your local values for ${c.keptPreferences}, ${c.keptHistory}, ${c.keptWatchlist}, and ${c.keptProfiles} matching identities. ` +
      "Your current engine and settings stay; candidate lists are extended and exclusions still win."
    : `Backup has ${c.importedPreferences} preferences, ${c.importedHistory} history entries, ${c.importedWatchlist} watchlist titles, and ${c.importedProfiles} named profiles. ` +
      `Replace removes ${c.removedPreferences} local preferences, ${c.removedHistory} history entries, ${c.removedWatchlist} watchlist titles, and ${c.removedProfiles} profiles absent from the file; ` +
      `overwrites ${c.replacedPreferences} preferences, ${c.replacedHistory} history entries, ${c.replacedWatchlist} watchlist titles, and ${c.replacedProfiles} matching profiles. ` +
      "It also replaces the engine, settings, and candidate overrides.";
  const unmappedPreferences = plan.state.preferences.filter((item) =>
    !recommendationIndex.animeByNodeId.has(item.nodeId)).length;
  const unmappedHistory = (plan.state.history ?? []).filter((item) =>
    item.animeId === null || !recommendationIndex.animeByAnimeId.has(item.animeId)).length;
  const unmappedWatchlist = (plan.state.watchlist ?? []).filter((item) =>
    !recommendationIndex.animeByAnimeId.has(item.animeId)).length;
  profileBackupUnmappedEl.textContent =
    `${unmappedPreferences} resulting preferences, ${unmappedHistory} history entries, and ${unmappedWatchlist} watchlist titles are unavailable in this catalog; their original identities are retained.`;
}

function previewProfileReset(): void {
  clearPendingProfileBackup();
  const storageRevision = persistence.getProfileStorageRevision();
  if (storageRevision === null) {
    profileBackupStatusEl.textContent = "Browser storage could not be read. Reset was not prepared.";
    renderStorageWarnings();
    return;
  }
  pendingProfileBackup = { kind: "reset", storageRevision, memoryRevision: profileMemoryRevision() };
  renderProfileBackupPreview();
  profileBackupStatusEl.textContent = "Review the local reset counts before applying.";
}

function applyCommittedProfileCollection(
  state: StoredRecommendationState, profiles: Map<string, RecommendationProfileRecord>,
): void {
  cancelActiveUsernameImport();
  cancelBulkFileLoad();
  clearPendingHistoryImport();
  recommendationController?.abort();
  ++recommendationRunId;
  clearPendingProfileBackup();
  savedProfiles.clear();
  for (const [name, profile] of profiles) savedProfiles.set(name, profile);
  applyRecommendationState({
    ...state,
    modelBlendWeight: state.modelBlendWeight ?? 0.5,
    allowRelatedTitles: state.allowRelatedTitles ?? false,
    includeCandidates: state.includeCandidates ?? [],
    excludeCandidates: state.excludeCandidates ?? [],
    history: state.history ?? [],
    watchlist: state.watchlist ?? [],
  });
  renderProfileOptions(savedProfiles);
  renderSelectedAnime();
  renderImportedHistory();
  renderWatchlist();
  renderIncludeCandidates();
  renderExcludeCandidates();
  renderStorageWarnings();
  void updateRecommendations();
}

function applyPendingProfileBackup(): void {
  const pending = pendingProfileBackup;
  if (!pending) return;
  if (profileMemoryRevision() !== pending.memoryRevision ||
    persistence.getProfileStorageRevision() !== pending.storageRevision) {
    clearPendingProfileBackup();
    profileBackupStatusEl.textContent = "Profile data changed since the preview. Review a fresh preview before applying.";
    renderStorageWarnings();
    return;
  }
  const isReset = pending.kind === "reset";
  let plan;
  try {
    plan = !isReset && pending.document
      ? persistence.planProfileBackupImport(pending.document, buildCurrentRecommendationState(), savedProfiles,
        profileBackupModeEl.value === "replace" ? "replace" : "merge")
      : null;
  } catch {
    clearPendingProfileBackup();
    profileBackupStatusEl.textContent = "This backup cannot be applied to current browser data. Local data is unchanged.";
    return;
  }
  const state = plan?.state ?? emptyRecommendationState();
  const profiles = plan?.profiles ?? new Map<string, RecommendationProfileRecord>();
  const saved = isReset
    ? persistence.resetProfileCollection(pending.storageRevision)
    : persistence.commitProfileCollection(state, profiles, pending.storageRevision);
  if (!saved) {
    clearPendingProfileBackup();
    profileBackupStatusEl.textContent = "Browser storage rejected the change. The original data is retained or available for recovery.";
    renderStorageWarnings();
    return;
  }
  applyCommittedProfileCollection(state, profiles);
  profileBackupStatusEl.textContent = isReset
    ? "Local recommendation state and named profiles were reset. Older source keys and backups were retained."
    : "Profile backup was applied and saved locally.";
}

function recoverLoadedProfileCopy(): void {
  clearPendingProfileBackup();
  if (!persistence.recoverInterruptedProfileWrite()) {
    profileBackupStatusEl.textContent = "Browser storage still rejects recovery. Original data remains in the recovery journal.";
    renderStorageWarnings();
    return;
  }
  const state = persistence.loadRecommendationState();
  const profiles = persistence.loadRecommendationProfiles();
  if (persistence.getStorageWarnings().some((warning) => warning.includes("could not be read"))) {
    profileBackupStatusEl.textContent = "No readable copy was found for part of this profile data. Preview reset or import a valid backup.";
    renderStorageWarnings();
    return;
  }
  const revision = persistence.getProfileStorageRevision();
  if (revision === null || !persistence.commitProfileCollection({
    version: RECOMMENDATION_STORAGE_VERSION, ...state,
  }, profiles, revision)) {
    profileBackupStatusEl.textContent = "Browser storage rejected repair. The original and backup copies remain available.";
    renderStorageWarnings();
    return;
  }
  applyCommittedProfileCollection({ version: RECOMMENDATION_STORAGE_VERSION, ...state }, profiles);
  profileBackupStatusEl.textContent = "Readable profile data was restored to current browser storage.";
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

  invalidatePendingProfileBackup();
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
    allowRelatedTitles: profile.state.allowRelatedTitles ?? false,
    includeCandidates: profile.state.includeCandidates ?? [],
    excludeCandidates: profile.state.excludeCandidates ?? [],
    history: profile.state.history ?? [],
    watchlist: profile.state.watchlist ?? [],
  });
  const saved = persistRecommendationState();
  renderSelectedAnime();
  renderImportedHistory();
  renderWatchlist();
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

  invalidatePendingProfileBackup();
  renderStorageWarnings();
  renderProfileOptions(savedProfiles);
  recMessageEl.textContent = `Deleted profile "${name}".`;
}

function applyRecommendationState(state: {
  mode: RecommendationMode;
  preferences: AnimePreference[];
  modelBlendWeight: number;
  allowRelatedTitles: boolean;
  includeCandidates: string[];
  excludeCandidates: string[];
  history?: HistoryEntry[];
  watchlist?: WatchlistEntry[];
}): void {
  selectedAnimeNodeIds.splice(0, selectedAnimeNodeIds.length);
  selectedAnimePreferences.clear();
  includeCandidateNodeIds.splice(0, includeCandidateNodeIds.length);
  excludeCandidateNodeIds.splice(0, excludeCandidateNodeIds.length);
  historyEntries.splice(0, historyEntries.length, ...(state.history ?? []));
  watchlistEntries.splice(0, watchlistEntries.length, ...(state.watchlist ?? []));
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
  allowRelatedTitles = state.allowRelatedTitles;
  allowRelatedInput.checked = allowRelatedTitles;
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
