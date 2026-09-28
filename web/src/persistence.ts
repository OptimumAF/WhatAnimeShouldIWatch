/** Local browser state and legacy profile parsing behind injected storage and clock. */
import { clampModelBlendWeight, clampWatchWeight } from "./recommendations";
import { historyIdentity, validateHistoryEntries } from "./import-history";
import type { HistoryEntry } from "./import-history";
import { migrateLegacyPreferences, validatePreferences } from "./preferences";
import type { AnimePreference } from "./preferences";
import type { RuntimePorts } from "./runtime";

export type RecommendationMode = "graph" | "model" | "hybrid";
export type ThemeMode = "dark" | "light";
export type ContrastMode = "normal" | "high";
export interface StoredRecommendationState {
  version: number;
  mode: RecommendationMode;
  preferences: AnimePreference[];
  modelBlendWeight?: number;
  allowRelatedTitles?: boolean;
  includeCandidates?: string[];
  excludeCandidates?: string[];
  history?: HistoryEntry[];
}

export interface RecommendationProfileRecord {
  name: string;
  updatedAt: string;
  state: StoredRecommendationState;
}

export type RecommendationState = Omit<StoredRecommendationState, "version"> & {
  modelBlendWeight: number;
  allowRelatedTitles: boolean;
  includeCandidates: string[];
  excludeCandidates: string[];
  history: HistoryEntry[];
};

export const RECOMMENDATION_STORAGE_VERSION = 5;
export function emptyRecommendationState(): StoredRecommendationState {
  return {
    version: RECOMMENDATION_STORAGE_VERSION,
    mode: "graph", preferences: [], modelBlendWeight: 0.5, allowRelatedTitles: false,
    includeCandidates: [], excludeCandidates: [], history: [],
  };
}
export const PROFILE_BACKUP_FORMAT = "wasiw-profile-backup";
export const PROFILE_BACKUP_VERSION = 1;
export const MAX_PROFILE_BACKUP_BYTES = 8 * 1024 * 1024;

export type ProfileBackupImportMode = "merge" | "replace";
export interface ProfileBackupDocument {
  format: typeof PROFILE_BACKUP_FORMAT;
  version: typeof PROFILE_BACKUP_VERSION;
  exportedAt: string;
  state: StoredRecommendationState;
  profiles: RecommendationProfileRecord[];
}

export interface ProfileBackupImportPlan {
  mode: ProfileBackupImportMode;
  state: StoredRecommendationState;
  profiles: Map<string, RecommendationProfileRecord>;
  counts: {
    importedPreferences: number;
    addedPreferences: number;
    keptPreferences: number;
    replacedPreferences: number;
    removedPreferences: number;
    importedHistory: number;
    addedHistory: number;
    keptHistory: number;
    replacedHistory: number;
    removedHistory: number;
    importedProfiles: number;
    addedProfiles: number;
    keptProfiles: number;
    replacedProfiles: number;
    removedProfiles: number;
  };
}

interface ProfileWriteJournal {
  version: 1;
  originals: Record<string, string | null>;
}

interface StoredHelpTipsState {
  version: number;
  dismissed: boolean;
}

export const COMMAND_HISTORY_LIMIT = 6;
export const COMMAND_PINNED_LIMIT = 8;

export function createPersistenceAdapter(runtime: RuntimePorts, storagePrefix: string) {
  const LEGACY_STATE_KEY = `${storagePrefix}.recommendationState.v1`;
  const LEGACY_PROFILES_KEY = `${storagePrefix}.recommendationProfiles.v1`;
  const V4_STATE_KEY = `${storagePrefix}.recommendationState.v4`;
  const V4_PROFILES_KEY = `${storagePrefix}.recommendationProfiles.v4`;
  const STATE_KEY = `${storagePrefix}.recommendationState.v5`;
  const PROFILES_KEY = `${storagePrefix}.recommendationProfiles.v5`;
  const storageWarnings = new Map<"state" | "profiles" | "recovery", string>();
  let migrationNotice: string | null = null;
  const PROFILE_WRITE_JOURNAL_KEY = `${storagePrefix}.profileWrite.v1.pending`;
  const PROFILE_WRITE_KEYS = [
    STATE_KEY, PROFILES_KEY,
    `${STATE_KEY}.backup`, `${STATE_KEY}.corrupt`,
    `${PROFILES_KEY}.backup`, `${PROFILES_KEY}.corrupt`,
    `${V4_STATE_KEY}.backup`, `${LEGACY_STATE_KEY}.backup`,
    `${V4_PROFILES_KEY}.backup`, `${LEGACY_PROFILES_KEY}.backup`,
  ];
  let pendingProfileWrite: ProfileWriteJournal | null = null;
  let invalidProfileWriteJournal = false;
  let profileWriteInProgress = false;
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
    } catch {
      console.warn("Unable to load pinned commands.");
      return [];
    }
  }

  function persistCommandPinnedIds(ids: string[]): void {
    try {
      runtime.storage.setItem(COMMAND_PINNED_STORAGE_KEY, JSON.stringify(ids));
    } catch {
      console.warn("Unable to persist pinned commands.");
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
    } catch {
      console.warn("Unable to load command history.");
      return [];
    }
  }

  function persistCommandHistoryIds(history: string[]): void {
    try {
      runtime.storage.setItem(COMMAND_HISTORY_STORAGE_KEY, JSON.stringify(history));
    } catch {
      console.warn("Unable to persist command history.");
    }
  }

  function loadThemeModePreference(prefersLight: () => boolean): ThemeMode {
    try {
      const raw = runtime.storage.getItem(THEME_STORAGE_KEY);
      if (raw === "dark" || raw === "light") {
        return raw;
      }
    } catch {
      console.warn("Unable to read saved theme preference.");
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
    } catch {
      console.warn("Unable to read saved contrast preference.");
    }
    return "normal";
  }

  function persistThemeModePreference(theme: ThemeMode): void {
    try {
      runtime.storage.setItem(THEME_STORAGE_KEY, theme);
    } catch {
      console.warn("Unable to persist theme preference.");
    }
  }

  function persistContrastModePreference(contrast: ContrastMode): void {
    try {
      runtime.storage.setItem(CONTRAST_STORAGE_KEY, contrast);
    } catch {
      console.warn("Unable to persist contrast preference.");
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
    } catch {
      console.warn("Unable to load help tips preference.");
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
    } catch {
      console.warn("Unable to persist help tips preference.");
    }
  }

  function isRecord(value: unknown): value is Record<string, unknown> {
    return value !== null && typeof value === "object" && !Array.isArray(value);
  }

  function parseProfileWriteJournal(raw: string): ProfileWriteJournal {
    const value = JSON.parse(raw) as unknown;
    if (!isRecord(value) || value.version !== 1 ||
      !Object.keys(value).every((key) => ["version", "originals"].includes(key)) ||
      !isRecord(value.originals)) {
      throw new Error("Invalid profile write journal");
    }
    const originals = value.originals;
    if (Object.keys(originals).length !== PROFILE_WRITE_KEYS.length ||
      PROFILE_WRITE_KEYS.some((key) => !(key in originals) ||
        !(originals[key] === null || typeof originals[key] === "string"))) {
      throw new Error("Invalid profile write journal");
    }
    return value as unknown as ProfileWriteJournal;
  }

  function restoreRaw(key: string, raw: string | null): void {
    if (runtime.storage.getItem(key) === raw) return;
    if (raw === null) runtime.storage.removeItem(key);
    else runtime.storage.setItem(key, raw);
  }

  function recoverInterruptedProfileWrite(): boolean {
    let raw: string | null;
    try {
      raw = runtime.storage.getItem(PROFILE_WRITE_JOURNAL_KEY);
    } catch {
      invalidProfileWriteJournal = true;
      storageWarnings.set("recovery", "Browser storage cannot read profile recovery data. New changes are blocked.");
      return false;
    }
    try {
      if (raw === null) {
        pendingProfileWrite = null;
        invalidProfileWriteJournal = false;
        return true;
      }
      const journal = parseProfileWriteJournal(raw);
      pendingProfileWrite = journal;
      for (const key of PROFILE_WRITE_KEYS) restoreRaw(key, journal.originals[key]);
      if (PROFILE_WRITE_KEYS.some((key) => runtime.storage.getItem(key) !== journal.originals[key])) {
        throw new Error("Profile write recovery did not match original bytes");
      }
      runtime.storage.removeItem(PROFILE_WRITE_JOURNAL_KEY);
      pendingProfileWrite = null;
      invalidProfileWriteJournal = false;
      storageWarnings.set("recovery", "An interrupted profile change was rolled back to the original browser data.");
      return true;
    } catch {
      if (pendingProfileWrite === null) invalidProfileWriteJournal = true;
      storageWarnings.set("recovery", "An interrupted profile change needs recovery. Original data is retained; new changes are blocked.");
      return false;
    }
  }

  function parseState(value: unknown, versions: number[]): StoredRecommendationState {
    if (!isRecord(value) || !versions.includes(value.version as number) ||
      (value.mode !== "graph" && value.mode !== "model" && value.mode !== "hybrid")) {
      throw new Error("Invalid recommendation state shape or version");
    }
    const history = value.history === undefined ? [] : validateHistoryEntries(value.history);
    let preferences: AnimePreference[];
    if (value.version === RECOMMENDATION_STORAGE_VERSION) {
      preferences = validatePreferences(value.preferences);
    } else {
      if (!Array.isArray(value.selected)) throw new Error("Invalid saved selection");
      const selected = value.selected.map((entry: unknown) => {
        if (!isRecord(entry) || typeof entry.nodeId !== "string" || !entry.nodeId ||
          typeof entry.weight !== "number" || !Number.isFinite(entry.weight) ||
          clampWatchWeight(entry.weight) !== entry.weight) {
          throw new Error("Invalid saved selection");
        }
        return { nodeId: entry.nodeId, weight: entry.weight };
      });
      preferences = migrateLegacyPreferences(selected, history);
    }
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
    if (value.allowRelatedTitles !== undefined && typeof value.allowRelatedTitles !== "boolean") {
      throw new Error("Invalid related-title preference");
    }
    return {
      version: RECOMMENDATION_STORAGE_VERSION,
      mode: value.mode,
      preferences,
      modelBlendWeight: blend,
      allowRelatedTitles: value.allowRelatedTitles ?? false,
      includeCandidates: candidateIds("includeCandidates"),
      excludeCandidates: candidateIds("excludeCandidates"),
      history,
    };
  }

  function parseProfiles(value: unknown, sourceVersion: 1 | 4 | 5): Map<string, RecommendationProfileRecord> {
    const records = sourceVersion === 1 ? value :
      isRecord(value) && value.version === sourceVersion ? value.profiles : null;
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
        state: parseState(entry.state, sourceVersion === 1 ? [1, 2, 3] : [sourceVersion]),
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
    else if (existing !== raw && key.endsWith(".v5")) runtime.storage.setItem(backupKey, raw);
  }

  function loadVersioned<T>(
    kind: "state" | "profiles",
    currentKey: string,
    legacySources: { key: string; parse: (raw: string) => T }[],
    parseCurrent: (raw: string) => T,
    encode: (value: T) => string,
  ): T | null {
    const label = kind === "state" ? "recommendation state" : "saved profiles";
    if (pendingProfileWrite !== null || invalidProfileWriteJournal) {
      if (invalidProfileWriteJournal) return null;
      try {
        const originalRaw = pendingProfileWrite!.originals[kind === "state" ? STATE_KEY : PROFILES_KEY];
        if (originalRaw !== null) return parseCurrent(originalRaw);
        for (const source of legacySources) {
          for (const candidate of [runtime.storage.getItem(source.key), runtime.storage.getItem(`${source.key}.backup`)]) {
            if (candidate === null) continue;
            try { return source.parse(candidate); } catch { /* Try the next original copy. */ }
          }
        }
      } catch { /* The recovery warning is already visible. */ }
      return null;
    }
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

      let value: T | null = null;
      let sourceRaw: string | null = null;
      let sourceKey: string | null = null;
      let foundOlder = false;
      for (const source of legacySources) {
        const raw = runtime.storage.getItem(source.key);
        const backup = runtime.storage.getItem(`${source.key}.backup`);
        if (raw !== null || backup !== null) foundOlder = true;
        for (const candidate of [raw, backup]) {
          if (candidate === null) continue;
          try {
            value = source.parse(candidate);
            sourceRaw = candidate;
            sourceKey = source.key;
            break;
          } catch { /* Try the next intact copy. */ }
        }
        if (value !== null) break;
      }
      if (value === null || sourceRaw === null || sourceKey === null) {
        if (currentRaw !== null || foundOlder) {
          storageWarnings.set(kind, `Stored ${label} could not be read. Original browser storage was left intact.`);
        }
        return null;
      }
      if (currentRaw !== null) {
        storageWarnings.set(kind, `Recovered ${label} from older storage. The unreadable current copy remains in browser storage.`);
        return value;
      }
      try {
        backUpRaw(sourceKey, sourceRaw);
        runtime.storage.setItem(currentKey, encode(value));
        storageWarnings.delete(kind);
        migrationNotice = "Earlier watch weights were migrated conservatively. Unrated watches stay Seen; low scores are Disliked; clear likes are Liked. Review preferences and importance below.";
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
    kind: "state" | "profiles", currentKey: string, legacyKeys: string[],
    raw: string, validateCurrent: (raw: string) => unknown,
  ): boolean {
    const label = kind === "state" ? "recommendation state" : "saved profiles";
    if (!profileWriteInProgress && (pendingProfileWrite !== null || invalidProfileWriteJournal)) {
      storageWarnings.set(kind, `Browser storage needs profile recovery before saving ${label}.`);
      return false;
    }
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
        for (const key of legacyKeys) {
          const legacy = runtime.storage.getItem(key);
          if (legacy !== null) {
            backUpRaw(key, legacy);
            break;
          }
        }
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
      "state", STATE_KEY, [
        { key: V4_STATE_KEY, parse: (raw) => parseState(JSON.parse(raw) as unknown, [4]) },
        { key: LEGACY_STATE_KEY, parse: (raw) => parseState(JSON.parse(raw) as unknown, [1, 2, 3]) },
      ],
      (raw) => parseState(JSON.parse(raw) as unknown, [RECOMMENDATION_STORAGE_VERSION]),
      (state) => JSON.stringify(state),
    );
    if (stored !== null) {
      return {
        mode: stored.mode, preferences: stored.preferences,
        modelBlendWeight: stored.modelBlendWeight ?? 0.5,
        allowRelatedTitles: stored.allowRelatedTitles ?? false,
        includeCandidates: stored.includeCandidates ?? [],
        excludeCandidates: stored.excludeCandidates ?? [],
        history: stored.history ?? [],
      };
    }
    const state = emptyRecommendationState();
    return {
      mode: state.mode, preferences: state.preferences,
      modelBlendWeight: state.modelBlendWeight ?? 0.5,
      allowRelatedTitles: state.allowRelatedTitles ?? false,
      includeCandidates: state.includeCandidates ?? [],
      excludeCandidates: state.excludeCandidates ?? [], history: state.history ?? [],
    };
  }

  function persistRecommendationState(payload: StoredRecommendationState): boolean {
    return persistVersioned(
      "state", STATE_KEY, [V4_STATE_KEY, LEGACY_STATE_KEY], JSON.stringify(payload),
      (raw) => parseState(JSON.parse(raw) as unknown, [RECOMMENDATION_STORAGE_VERSION]),
    );
  }

  function loadRecommendationProfiles(): Map<string, RecommendationProfileRecord> {
    return loadVersioned(
      "profiles", PROFILES_KEY, [
        { key: V4_PROFILES_KEY, parse: (raw) => parseProfiles(JSON.parse(raw) as unknown, 4) },
        { key: LEGACY_PROFILES_KEY, parse: (raw) => parseProfiles(JSON.parse(raw) as unknown, 1) },
      ],
      (raw) => parseProfiles(JSON.parse(raw) as unknown, 5),
      encodeProfiles,
    ) ?? new Map();
  }

  function persistRecommendationProfiles(profiles: Map<string, RecommendationProfileRecord>): boolean {
    try {
      return persistVersioned(
        "profiles", PROFILES_KEY, [V4_PROFILES_KEY, LEGACY_PROFILES_KEY], encodeProfiles(profiles),
        (raw) => parseProfiles(JSON.parse(raw) as unknown, 5),
      );
    } catch {
      storageWarnings.set("profiles", "Saved profiles could not be encoded; existing browser data remains intact.");
      return false;
    }
  }

  function onlyFields(value: unknown, fields: readonly string[], label: string): asserts value is Record<string, unknown> {
    if (!isRecord(value) || Object.keys(value).some((key) => !fields.includes(key))) {
      throw new Error(`Invalid ${label} fields`);
    }
  }

  function parseProfileBackup(raw: string): ProfileBackupDocument {
    if (new TextEncoder().encode(raw).byteLength > MAX_PROFILE_BACKUP_BYTES) {
      throw new Error("Profile backup exceeds 8 MiB");
    }
    const value = JSON.parse(raw) as unknown;
    onlyFields(value, ["format", "version", "exportedAt", "state", "profiles"], "profile backup");
    if (value.format !== PROFILE_BACKUP_FORMAT || value.version !== PROFILE_BACKUP_VERSION ||
      typeof value.exportedAt !== "string" || !Number.isFinite(Date.parse(value.exportedAt))) {
      throw new Error("Unsupported profile backup format or version");
    }
    onlyFields(value.state, ["version", "mode", "preferences", "modelBlendWeight", "allowRelatedTitles",
      "includeCandidates", "excludeCandidates", "history"], "profile backup state");
    if (!Array.isArray(value.state.preferences) || !Array.isArray(value.state.history)) {
      throw new Error("Invalid profile backup state");
    }
    for (const item of value.state.preferences) {
      onlyFields(item, ["nodeId", "sentiment", "importance", "confidence", "source"], "preference");
    }
    for (const item of value.state.history) {
      onlyFields(item, ["provider", "sourceId", "title", "animeId", "status", "sourceStatus",
        "progressEpisodes", "score", "scoreScale"], "history");
    }
    if (!Array.isArray(value.profiles) || value.profiles.length > 1000) {
      throw new Error("Invalid profile backup profile count");
    }
    for (const profile of value.profiles) {
      onlyFields(profile, ["name", "updatedAt", "state"], "profile");
      if (typeof profile.name !== "string" || profile.name !== profile.name.trim() ||
        profile.name.length > 160 || typeof profile.updatedAt !== "string") {
        throw new Error("Invalid profile backup profile name or date");
      }
      onlyFields(profile.state, ["version", "mode", "preferences", "modelBlendWeight", "allowRelatedTitles",
        "includeCandidates", "excludeCandidates", "history"], "profile state");
      if (!Array.isArray(profile.state.preferences) || !Array.isArray(profile.state.history)) {
        throw new Error("Invalid profile state");
      }
      for (const item of profile.state.preferences) {
        onlyFields(item, ["nodeId", "sentiment", "importance", "confidence", "source"], "profile preference");
      }
      for (const item of profile.state.history) {
        onlyFields(item, ["provider", "sourceId", "title", "animeId", "status", "sourceStatus",
          "progressEpisodes", "score", "scoreScale"], "profile history");
      }
    }
    const state = parseState(value.state, [RECOMMENDATION_STORAGE_VERSION]);
    const profiles = parseProfiles({ version: RECOMMENDATION_STORAGE_VERSION, profiles: value.profiles }, 5);
    return {
      format: PROFILE_BACKUP_FORMAT,
      version: PROFILE_BACKUP_VERSION,
      exportedAt: value.exportedAt,
      state,
      profiles: [...profiles.values()],
    };
  }

  function createProfileBackup(
    state: StoredRecommendationState, profiles: Map<string, RecommendationProfileRecord>,
  ): string {
    const document: ProfileBackupDocument = {
      format: PROFILE_BACKUP_FORMAT,
      version: PROFILE_BACKUP_VERSION,
      exportedAt: runtime.now().toISOString(),
      state: parseState(state, [RECOMMENDATION_STORAGE_VERSION]),
      profiles: [...parseProfiles(JSON.parse(encodeProfiles(profiles)) as unknown, 5).values()],
    };
    const raw = JSON.stringify(document, null, 2);
    parseProfileBackup(raw);
    return `${raw}\n`;
  }

  function planProfileBackupImport(
    backup: ProfileBackupDocument, currentState: StoredRecommendationState,
    currentProfiles: Map<string, RecommendationProfileRecord>, mode: ProfileBackupImportMode,
  ): ProfileBackupImportPlan {
    if (mode !== "merge" && mode !== "replace") throw new Error("Invalid profile backup import mode");
    const imported = parseProfileBackup(JSON.stringify(backup));
    const current = parseState(currentState, [RECOMMENDATION_STORAGE_VERSION]);
    const localProfiles = parseProfiles(JSON.parse(encodeProfiles(currentProfiles)) as unknown, 5);
    const importedPreferences = new Map(imported.state.preferences.map((item) => [item.nodeId, item]));
    const currentPreferences = new Map(current.preferences.map((item) => [item.nodeId, item]));
    const importedHistory = new Map((imported.state.history ?? []).map((item) => [historyIdentity(item), item]));
    const currentHistory = new Map((current.history ?? []).map((item) => [historyIdentity(item), item]));
    const importedProfiles = new Map(imported.profiles.map((profile) => [profile.name, profile]));
    const overlap = <T>(left: Map<string, T>, right: Map<string, T>): number =>
      [...left.keys()].filter((key) => right.has(key)).length;
    const changed = <T>(left: Map<string, T>, right: Map<string, T>): number =>
      [...left].filter(([key, value]) => right.has(key) && JSON.stringify(value) !== JSON.stringify(right.get(key))).length;
    const counts = {
      importedPreferences: importedPreferences.size,
      addedPreferences: importedPreferences.size - overlap(importedPreferences, currentPreferences),
      keptPreferences: mode === "merge" ? overlap(importedPreferences, currentPreferences) : 0,
      replacedPreferences: mode === "replace" ? changed(importedPreferences, currentPreferences) : 0,
      removedPreferences: mode === "replace" ? currentPreferences.size - overlap(currentPreferences, importedPreferences) : 0,
      importedHistory: importedHistory.size,
      addedHistory: importedHistory.size - overlap(importedHistory, currentHistory),
      keptHistory: mode === "merge" ? overlap(importedHistory, currentHistory) : 0,
      replacedHistory: mode === "replace" ? changed(importedHistory, currentHistory) : 0,
      removedHistory: mode === "replace" ? currentHistory.size - overlap(currentHistory, importedHistory) : 0,
      importedProfiles: importedProfiles.size,
      addedProfiles: importedProfiles.size - overlap(importedProfiles, localProfiles),
      keptProfiles: mode === "merge" ? overlap(importedProfiles, localProfiles) : 0,
      replacedProfiles: mode === "replace" ? changed(importedProfiles, localProfiles) : 0,
      removedProfiles: mode === "replace" ? localProfiles.size - overlap(localProfiles, importedProfiles) : 0,
    };
    if (mode === "replace") {
      return { mode, state: imported.state, profiles: importedProfiles, counts };
    }
    const preferences = [...current.preferences];
    for (const item of imported.state.preferences) {
      if (!currentPreferences.has(item.nodeId)) preferences.push(item);
    }
    const history = [...(current.history ?? [])];
    for (const item of imported.state.history ?? []) {
      if (!currentHistory.has(historyIdentity(item))) history.push(item);
    }
    const addMissing = (local: string[], incoming: string[]): string[] => {
      const next = [...local];
      const seen = new Set(next);
      for (const item of incoming) if (!seen.has(item)) { next.push(item); seen.add(item); }
      return next;
    };
    const profiles = new Map(localProfiles);
    for (const [name, profile] of importedProfiles) if (!profiles.has(name)) profiles.set(name, profile);
    const state = parseState({
      ...current,
      preferences,
      history,
      includeCandidates: addMissing(current.includeCandidates ?? [], imported.state.includeCandidates ?? []),
      excludeCandidates: addMissing(current.excludeCandidates ?? [], imported.state.excludeCandidates ?? []),
    }, [RECOMMENDATION_STORAGE_VERSION]);
    return { mode, state, profiles, counts };
  }

  function getProfileStorageRevision(): string | null {
    try {
      return JSON.stringify([runtime.storage.getItem(STATE_KEY), runtime.storage.getItem(PROFILES_KEY)]);
    } catch {
      storageWarnings.set("recovery", "Browser storage could not read profile data. Import and reset are blocked.");
      return null;
    }
  }

  function commitProfileCollection(
    state: StoredRecommendationState, profiles: Map<string, RecommendationProfileRecord>,
    expectedRevision: string,
  ): boolean {
    try {
      const checkedState = parseState(state, [RECOMMENDATION_STORAGE_VERSION]);
      const checkedProfiles = parseProfiles(JSON.parse(encodeProfiles(profiles)) as unknown, 5);
      if (!recoverInterruptedProfileWrite() || getProfileStorageRevision() !== expectedRevision) {
        storageWarnings.set("recovery", "Profile data changed since preview or needs recovery. Review it again before applying.");
        return false;
      }
      const journal: ProfileWriteJournal = {
        version: 1,
        originals: Object.fromEntries(PROFILE_WRITE_KEYS.map((key) => [key, runtime.storage.getItem(key)])),
      };
      runtime.storage.setItem(PROFILE_WRITE_JOURNAL_KEY, JSON.stringify(journal));
      pendingProfileWrite = journal;
      profileWriteInProgress = true;
      if (!persistRecommendationState(checkedState) || !persistRecommendationProfiles(checkedProfiles)) {
        throw new Error("Profile collection write rejected");
      }
      runtime.storage.removeItem(PROFILE_WRITE_JOURNAL_KEY);
      pendingProfileWrite = null;
      storageWarnings.delete("recovery");
      return true;
    } catch {
      recoverInterruptedProfileWrite();
      storageWarnings.set("recovery", "Profile change was not applied. Original browser data was retained or is available for recovery.");
      return false;
    } finally {
      profileWriteInProgress = false;
    }
  }

  function resetProfileCollection(expectedRevision: string): boolean {
    let archivedInvalidJournal: string | null = null;
    if (invalidProfileWriteJournal) {
      try {
        const raw = runtime.storage.getItem(PROFILE_WRITE_JOURNAL_KEY);
        if (raw !== null) {
          const corruptKey = `${PROFILE_WRITE_JOURNAL_KEY}.corrupt`;
          const preserved = runtime.storage.getItem(corruptKey);
          if (preserved !== null && preserved !== raw) throw new Error("Different recovery journal already archived");
          if (preserved === null) runtime.storage.setItem(corruptKey, raw);
          runtime.storage.removeItem(PROFILE_WRITE_JOURNAL_KEY);
          archivedInvalidJournal = raw;
        }
        invalidProfileWriteJournal = false;
      } catch {
        storageWarnings.set("recovery", "Unreadable recovery data could not be preserved. Local reset was not applied.");
        return false;
      }
    }
    const saved = commitProfileCollection(emptyRecommendationState(), new Map(), expectedRevision);
    if (!saved && archivedInvalidJournal !== null) {
      try {
        if (runtime.storage.getItem(PROFILE_WRITE_JOURNAL_KEY) === null) {
          runtime.storage.setItem(PROFILE_WRITE_JOURNAL_KEY, archivedInvalidJournal);
          invalidProfileWriteJournal = true;
        }
      } catch {
        storageWarnings.set("recovery", "Reset failed; the unreadable journal remains in its exact corrupt archive.");
      }
    }
    return saved;
  }

  function getStorageWarnings(): string[] {
    return [...storageWarnings.values()];
  }

  function getMigrationNotice(): string | null {
    return migrationNotice;
  }

  recoverInterruptedProfileWrite();

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
    createProfileBackup,
    parseProfileBackup,
    planProfileBackupImport,
    getProfileStorageRevision,
    commitProfileCollection,
    resetProfileCollection,
    recoverInterruptedProfileWrite,
    getStorageWarnings,
    getMigrationNotice,
  };
}
