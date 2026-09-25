/** Local browser state and legacy profile parsing behind injected storage and clock. */
import type { RecommendationIndex } from "./domain";
import { clampModelBlendWeight, clampWatchWeight } from "./recommendations";
import type { RuntimePorts } from "./runtime";

export type RecommendationMode = "graph" | "model" | "hybrid";
export type ThemeMode = "dark" | "light";
export type ContrastMode = "normal" | "high";
export interface StoredRecommendationState {
  version: number;
  mode: RecommendationMode;
  selected: { nodeId: string; weight: number }[];
  modelBlendWeight?: number;
  includeCandidates?: string[];
  excludeCandidates?: string[];
}

export interface RecommendationProfileRecord {
  name: string;
  updatedAt: string;
  state: StoredRecommendationState;
}

interface StoredHelpTipsState {
  version: number;
  dismissed: boolean;
}

export const COMMAND_HISTORY_LIMIT = 6;
export const COMMAND_PINNED_LIMIT = 8;

export function createPersistenceAdapter(runtime: RuntimePorts, storagePrefix: string) {
  const RECOMMENDATION_STATE_STORAGE_KEY = `${storagePrefix}.recommendationState.v1`;
  const RECOMMENDATION_PROFILES_STORAGE_KEY = `${storagePrefix}.recommendationProfiles.v1`;
  const THEME_STORAGE_KEY = `${storagePrefix}.theme.v1`;
  const CONTRAST_STORAGE_KEY = `${storagePrefix}.contrast.v1`;
  const HELP_TIPS_STORAGE_KEY = `${storagePrefix}.helpTips.v1`;
  const HELP_TIPS_VERSION = 1;
  const COMMAND_HISTORY_STORAGE_KEY = `${storagePrefix}.commandHistory.v1`;
  const COMMAND_PINNED_STORAGE_KEY = `${storagePrefix}.commandPinned.v1`;

  function parseRecommendationMode(value: string): RecommendationMode {
    if (value === "model") {
      return "model";
    }
    if (value === "hybrid") {
      return "hybrid";
    }
    return "graph";
  }

  function loadCommandPinnedIds(): string[] {
    try {
      const raw = runtime.storage.getItem(COMMAND_PINNED_STORAGE_KEY);
      if (!raw) {
        return [];
      }
      const parsed = JSON.parse(raw) as unknown;
      if (!Array.isArray(parsed)) {
        return [];
      }
      return parsed
        .filter((entry): entry is string => typeof entry === "string" && entry.length > 0)
        .slice(0, COMMAND_PINNED_LIMIT);
    } catch (error) {
      console.warn("Unable to load pinned commands.", error);
      return [];
    }
  }

  function persistCommandPinnedIds(ids: string[]): void {
    try {
      runtime.storage.setItem(COMMAND_PINNED_STORAGE_KEY, JSON.stringify(ids));
    } catch (error) {
      console.warn("Unable to persist pinned commands.", error);
    }
  }

  function loadCommandHistoryIds(): string[] {
    try {
      const raw = runtime.storage.getItem(COMMAND_HISTORY_STORAGE_KEY);
      if (!raw) {
        return [];
      }
      const parsed = JSON.parse(raw) as unknown;
      if (!Array.isArray(parsed)) {
        return [];
      }
      return parsed
        .filter((entry): entry is string => typeof entry === "string" && entry.length > 0)
        .slice(0, COMMAND_HISTORY_LIMIT);
    } catch (error) {
      console.warn("Unable to load command history.", error);
      return [];
    }
  }

  function persistCommandHistoryIds(history: string[]): void {
    try {
      runtime.storage.setItem(COMMAND_HISTORY_STORAGE_KEY, JSON.stringify(history));
    } catch (error) {
      console.warn("Unable to persist command history.", error);
    }
  }

  function loadThemeModePreference(prefersLight: () => boolean): ThemeMode {
    try {
      const raw = runtime.storage.getItem(THEME_STORAGE_KEY);
      if (raw === "dark" || raw === "light") {
        return raw;
      }
    } catch (error) {
      console.warn("Unable to read saved theme preference.", error);
    }

    if (prefersLight()) {
      return "light";
    }
    return "dark";
  }

  function loadContrastModePreference(): ContrastMode {
    try {
      const raw = runtime.storage.getItem(CONTRAST_STORAGE_KEY);
      if (raw === "normal" || raw === "high") {
        return raw;
      }
    } catch (error) {
      console.warn("Unable to read saved contrast preference.", error);
    }
    return "normal";
  }

  function persistThemeModePreference(theme: ThemeMode): void {
    try {
      runtime.storage.setItem(THEME_STORAGE_KEY, theme);
    } catch (error) {
      console.warn("Unable to persist theme preference.", error);
    }
  }

  function persistContrastModePreference(contrast: ContrastMode): void {
    try {
      runtime.storage.setItem(CONTRAST_STORAGE_KEY, contrast);
    } catch (error) {
      console.warn("Unable to persist contrast preference.", error);
    }
  }

  function loadHelpTipsDismissed(): boolean {
    try {
      const raw = runtime.storage.getItem(HELP_TIPS_STORAGE_KEY);
      if (!raw) {
        return false;
      }
      const parsed = JSON.parse(raw) as StoredHelpTipsState | null;
      if (
        parsed &&
        typeof parsed === "object" &&
        Number(parsed.version) === HELP_TIPS_VERSION &&
        typeof parsed.dismissed === "boolean"
      ) {
        return parsed.dismissed;
      }
    } catch (error) {
      console.warn("Unable to load help tips preference.", error);
    }
    return false;
  }

  function persistHelpTipsDismissed(dismissed: boolean): void {
    try {
      const payload: StoredHelpTipsState = {
        version: HELP_TIPS_VERSION,
        dismissed,
      };
      runtime.storage.setItem(HELP_TIPS_STORAGE_KEY, JSON.stringify(payload));
    } catch (error) {
      console.warn("Unable to persist help tips preference.", error);
    }
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
      const raw = runtime.storage.getItem(RECOMMENDATION_STATE_STORAGE_KEY);
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

  function persistRecommendationState(payload: StoredRecommendationState): void {
    try {
      runtime.storage.setItem(
        RECOMMENDATION_STATE_STORAGE_KEY,
        JSON.stringify(payload),
      );
    } catch (error) {
      console.warn("Unable to persist recommendation state.", error);
    }
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

  function loadRecommendationProfiles(index: RecommendationIndex): Map<string, RecommendationProfileRecord> {
    const profiles = new Map<string, RecommendationProfileRecord>();
    try {
      const raw = runtime.storage.getItem(RECOMMENDATION_PROFILES_STORAGE_KEY);
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
          index,
        );
        profiles.set(name, {
          name,
          updatedAt:
            typeof record.updatedAt === "string" ? record.updatedAt : runtime.now().toISOString(),
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
      runtime.storage.setItem(
        RECOMMENDATION_PROFILES_STORAGE_KEY,
        JSON.stringify(payload),
      );
    } catch (error) {
      console.warn("Unable to persist saved profiles.", error);
    }
  }

  return {
    parseRecommendationMode,
    loadCommandPinnedIds,
    persistCommandPinnedIds,
    loadCommandHistoryIds,
    persistCommandHistoryIds,
    loadThemeModePreference,
    loadContrastModePreference,
    persistThemeModePreference,
    persistContrastModePreference,
    loadHelpTipsDismissed,
    persistHelpTipsDismissed,
    loadRecommendationState,
    persistRecommendationState,
    sanitizeRecommendationState,
    loadRecommendationProfiles,
    persistRecommendationProfiles,
  };
}
