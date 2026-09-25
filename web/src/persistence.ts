/** Local browser state and legacy profile parsing behind injected storage and clock. */
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

export type RecommendationState = Omit<StoredRecommendationState, "version"> & {
  modelBlendWeight: number;
  includeCandidates: string[];
  excludeCandidates: string[];
};

export const RECOMMENDATION_STORAGE_VERSION = 4;

interface StoredHelpTipsState {
  version: number;
  dismissed: boolean;
}

export const COMMAND_HISTORY_LIMIT = 6;
export const COMMAND_PINNED_LIMIT = 8;

export function createPersistenceAdapter(runtime: RuntimePorts, storagePrefix: string) {
  const LEGACY_STATE_KEY = `${storagePrefix}.recommendationState.v1`;
  const LEGACY_PROFILES_KEY = `${storagePrefix}.recommendationProfiles.v1`;
  const STATE_KEY = `${storagePrefix}.recommendationState.v4`;
  const PROFILES_KEY = `${storagePrefix}.recommendationProfiles.v4`;
  const storageWarnings = new Map<"state" | "profiles", string>();
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

  function isRecord(value: unknown): value is Record<string, unknown> {
    return value !== null && typeof value === "object" && !Array.isArray(value);
  }

  function parseState(value: unknown, versions: number[]): StoredRecommendationState {
    if (!isRecord(value) || !versions.includes(value.version as number) ||
      (value.mode !== "graph" && value.mode !== "model" && value.mode !== "hybrid") ||
      !Array.isArray(value.selected)) {
      throw new Error("Invalid recommendation state shape or version");
    }
    const selected = value.selected.map((entry: unknown) => {
      if (!isRecord(entry) || typeof entry.nodeId !== "string" || !entry.nodeId ||
        typeof entry.weight !== "number" || !Number.isFinite(entry.weight) ||
        clampWatchWeight(entry.weight) !== entry.weight) {
        throw new Error("Invalid saved selection");
      }
      return { nodeId: entry.nodeId, weight: entry.weight };
    });
    const candidateIds = (field: "includeCandidates" | "excludeCandidates"): string[] => {
      const entries = value[field] === undefined ? [] : value[field];
      if (!Array.isArray(entries) || entries.some((entry) => typeof entry !== "string" || !entry)) {
        throw new Error(`Invalid ${field}`);
      }
      return [...entries] as string[];
    };
    const blend = value.modelBlendWeight === undefined ? 0.5 : value.modelBlendWeight;
    if (typeof blend !== "number" || !Number.isFinite(blend) ||
      clampModelBlendWeight(blend) !== blend) {
      throw new Error("Invalid saved blend weight");
    }
    return {
      version: RECOMMENDATION_STORAGE_VERSION,
      mode: value.mode,
      selected,
      modelBlendWeight: blend,
      includeCandidates: candidateIds("includeCandidates"),
      excludeCandidates: candidateIds("excludeCandidates"),
    };
  }

  function parseProfiles(value: unknown, current: boolean): Map<string, RecommendationProfileRecord> {
    const records = current && isRecord(value) && value.version === RECOMMENDATION_STORAGE_VERSION
      ? value.profiles : !current ? value : null;
    if (!Array.isArray(records)) throw new Error("Invalid saved profiles shape or version");
    const profiles = new Map<string, RecommendationProfileRecord>();
    for (const entry of records) {
      if (!isRecord(entry) || typeof entry.name !== "string" || !entry.name.trim()) {
        throw new Error("Invalid saved profile name");
      }
      const name = entry.name.trim();
      if (profiles.has(name)) throw new Error("Duplicate saved profile name");
      profiles.set(name, {
        name,
        updatedAt: typeof entry.updatedAt === "string" ? entry.updatedAt : runtime.now().toISOString(),
        state: parseState(entry.state, current ? [RECOMMENDATION_STORAGE_VERSION] : [1, 2, 3]),
      });
    }
    return profiles;
  }

  function encodeProfiles(profiles: Map<string, RecommendationProfileRecord>): string {
    return JSON.stringify({
      version: RECOMMENDATION_STORAGE_VERSION,
      profiles: [...profiles.values()]
        .sort((left, right) => left.name.localeCompare(right.name))
        .map((profile) => ({
          name: profile.name,
          updatedAt: profile.updatedAt,
          state: parseState(profile.state, [RECOMMENDATION_STORAGE_VERSION]),
        })),
    });
  }

  function backUpRaw(key: string, raw: string): void {
    const backupKey = `${key}.backup`;
    const existing = runtime.storage.getItem(backupKey);
    // Preserve an older distinct backup; the untouched source key still holds this version.
    if (existing === null) runtime.storage.setItem(backupKey, raw);
    else if (existing !== raw && key.endsWith(".v4")) runtime.storage.setItem(backupKey, raw);
  }

  function loadVersioned<T>(
    kind: "state" | "profiles",
    currentKey: string,
    legacyKey: string,
    parseCurrent: (raw: string) => T,
    parseLegacy: (raw: string) => T,
    encode: (value: T) => string,
  ): T | null {
    const label = kind === "state" ? "recommendation state" : "saved profiles";
    try {
      const currentRaw = runtime.storage.getItem(currentKey);
      if (currentRaw !== null) {
        try {
          const value = parseCurrent(currentRaw);
          storageWarnings.delete(kind);
          return value;
        } catch {
          const currentBackup = runtime.storage.getItem(`${currentKey}.backup`);
          if (currentBackup !== null) {
            try {
              const value = parseCurrent(currentBackup);
              storageWarnings.set(kind, `Recovered ${label} from a backup. The unreadable current copy remains in browser storage.`);
              return value;
            } catch { /* Try the untouched legacy source below. */ }
          }
        }
      }

      const legacyRaw = runtime.storage.getItem(legacyKey);
      const legacyBackup = runtime.storage.getItem(`${legacyKey}.backup`);
      let value: T | null = null;
      let sourceRaw: string | null = null;
      if (legacyRaw !== null) {
        try { value = parseLegacy(legacyRaw); sourceRaw = legacyRaw; } catch { /* Try backup. */ }
      }
      if (value === null && legacyBackup !== null) {
        try { value = parseLegacy(legacyBackup); sourceRaw = legacyBackup; } catch { /* Report below. */ }
      }
      if (value === null || sourceRaw === null) {
        if (currentRaw !== null || legacyRaw !== null || legacyBackup !== null) {
          storageWarnings.set(kind, `Stored ${label} could not be read. Original browser storage was left intact.`);
        }
        return null;
      }
      if (currentRaw !== null) {
        storageWarnings.set(kind, `Recovered ${label} from older storage. The unreadable current copy remains in browser storage.`);
        return value;
      }
      try {
        backUpRaw(legacyKey, sourceRaw);
        runtime.storage.setItem(currentKey, encode(value));
        storageWarnings.delete(kind);
      } catch {
        storageWarnings.set(kind, `Browser storage rejected the ${label} backup or migration. Older data is intact, but changes may be lost on reload.`);
      }
      return value;
    } catch {
      storageWarnings.set(kind, `Browser storage could not read ${label}. Changes may be lost on reload.`);
      return null;
    }
  }

  function persistVersioned(
    kind: "state" | "profiles", currentKey: string, legacyKey: string,
    raw: string, validateCurrent: (raw: string) => unknown,
  ): boolean {
    const label = kind === "state" ? "recommendation state" : "saved profiles";
    try {
      validateCurrent(raw);
      const existing = runtime.storage.getItem(currentKey);
      if (existing === raw) {
        storageWarnings.delete(kind);
        return true;
      }
      if (existing !== null) {
        let validExisting = true;
        try { validateCurrent(existing); } catch { validExisting = false; }
        if (validExisting) {
          backUpRaw(currentKey, existing);
        } else {
          const corruptKey = `${currentKey}.corrupt`;
          const preserved = runtime.storage.getItem(corruptKey);
          if (preserved !== null && preserved !== existing) throw new Error("Corrupt storage copy already exists");
          if (preserved === null) runtime.storage.setItem(corruptKey, existing);
        }
      } else {
        const legacy = runtime.storage.getItem(legacyKey);
        if (legacy !== null) backUpRaw(legacyKey, legacy);
      }
      runtime.storage.setItem(currentKey, raw);
      storageWarnings.delete(kind);
      return true;
    } catch {
      storageWarnings.set(kind, `Browser storage rejected changes to ${label}. Current edits are not saved for reload; existing data remains intact.`);
      return false;
    }
  }

  function loadRecommendationState(): RecommendationState {
    const stored = loadVersioned(
      "state", STATE_KEY, LEGACY_STATE_KEY,
      (raw) => parseState(JSON.parse(raw) as unknown, [RECOMMENDATION_STORAGE_VERSION]),
      (raw) => parseState(JSON.parse(raw) as unknown, [1, 2, 3]),
      (state) => JSON.stringify(state),
    );
    if (stored !== null) {
      return {
        mode: stored.mode, selected: stored.selected,
        modelBlendWeight: stored.modelBlendWeight ?? 0.5,
        includeCandidates: stored.includeCandidates ?? [],
        excludeCandidates: stored.excludeCandidates ?? [],
      };
    }
    return {
      mode: "graph", selected: [], modelBlendWeight: 0.5,
      includeCandidates: [], excludeCandidates: [],
    };
  }

  function persistRecommendationState(payload: StoredRecommendationState): boolean {
    return persistVersioned(
      "state", STATE_KEY, LEGACY_STATE_KEY, JSON.stringify(payload),
      (raw) => parseState(JSON.parse(raw) as unknown, [RECOMMENDATION_STORAGE_VERSION]),
    );
  }

  function loadRecommendationProfiles(): Map<string, RecommendationProfileRecord> {
    return loadVersioned(
      "profiles", PROFILES_KEY, LEGACY_PROFILES_KEY,
      (raw) => parseProfiles(JSON.parse(raw) as unknown, true),
      (raw) => parseProfiles(JSON.parse(raw) as unknown, false),
      encodeProfiles,
    ) ?? new Map();
  }

  function persistRecommendationProfiles(profiles: Map<string, RecommendationProfileRecord>): boolean {
    try {
      return persistVersioned(
        "profiles", PROFILES_KEY, LEGACY_PROFILES_KEY, encodeProfiles(profiles),
        (raw) => parseProfiles(JSON.parse(raw) as unknown, true),
      );
    } catch {
      storageWarnings.set("profiles", "Saved profiles could not be encoded; existing browser data remains intact.");
      return false;
    }
  }

  function getStorageWarnings(): string[] {
    return [...storageWarnings.values()];
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
    loadRecommendationProfiles,
    persistRecommendationProfiles,
    getStorageWarnings,
  };
}
