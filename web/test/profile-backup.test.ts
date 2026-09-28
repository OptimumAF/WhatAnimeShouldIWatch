import assert from "node:assert/strict";
import { test } from "node:test";
import {
  PROFILE_BACKUP_FORMAT, PROFILE_BACKUP_VERSION, createPersistenceAdapter, emptyRecommendationState,
} from "../src/persistence.ts";
import type { RecommendationProfileRecord, StoredRecommendationState } from "../src/persistence.ts";
import type { RuntimePorts, StoragePort } from "../src/runtime.ts";

const prefix = "wasiw.demo";
const stateKey = `${prefix}.recommendationState.v5`;
const profilesKey = `${prefix}.recommendationProfiles.v5`;
const journalKey = `${prefix}.profileWrite.v1.pending`;

function fakeRuntime() {
  const values = new Map<string, string>();
  const rejected = new Set<string>();
  const storage: StoragePort = {
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => {
      if (rejected.has(key)) throw new Error("invented quota rejection");
      values.set(key, value);
    },
    removeItem: (key) => { values.delete(key); },
  };
  const runtime = { storage, now: () => new Date("2026-09-28T12:00:00.000Z") } as RuntimePorts;
  return { values, rejected, runtime };
}

function state(nodeId: string, importance = 1.7): StoredRecommendationState {
  return {
    ...emptyRecommendationState(), mode: "hybrid", modelBlendWeight: 0.35,
    preferences: [{ nodeId, sentiment: "liked", importance, confidence: 0.5, source: "manual" }],
    includeCandidates: ["anime:998"], excludeCandidates: ["anime:997"],
    history: [{ provider: "local", sourceId: "anime:999", title: "Invented Unknown",
      animeId: 999, status: "completed", sourceStatus: "Completed", progressEpisodes: 12,
      score: 9, scoreScale: "local-10" }],
    watchlist: [{ animeId: 999, title: "<Invented Unknown>", status: "on_hold", rating: 4 }],
  };
}

function profile(name: string, value: StoredRecommendationState): RecommendationProfileRecord {
  return { name, updatedAt: "2026-09-28T12:00:00.000Z", state: value };
}

test("versioned JSON backup round-trips unknown IDs, native history, profiles, and exact importance", () => {
  const fake = fakeRuntime();
  const adapter = createPersistenceAdapter(fake.runtime, prefix);
  const current = state("anime:999", 2.4);
  const profiles = new Map([["<Invented> Profile", profile("<Invented> Profile", current)]]);
  const raw = adapter.createProfileBackup(current, profiles);
  const parsed = adapter.parseProfileBackup(raw);
  assert.equal(parsed.format, PROFILE_BACKUP_FORMAT);
  assert.equal(parsed.version, PROFILE_BACKUP_VERSION);
  assert.equal(parsed.exportedAt, "2026-09-28T12:00:00.000Z");
  assert.deepEqual(parsed.state, current);
  assert.deepEqual(parsed.profiles, [...profiles.values()]);
  assert.equal(fake.values.size, 0);

  for (const bad of [
    { ...parsed, version: 3 },
    { ...parsed, extra: "future field" },
    { ...parsed, state: { ...parsed.state, version: 6 } },
    { ...parsed, state: { ...parsed.state, preferences: [{ ...parsed.state.preferences[0], extra: 1 }] } },
    { ...parsed, profiles: [...parsed.profiles, parsed.profiles[0]] },
  ]) {
    assert.throws(() => adapter.parseProfileBackup(JSON.stringify(bad)));
  }
});

test("v2 backup preserves local watchlist and upgrades strict v1 documents without inventing entries", () => {
  const adapter = createPersistenceAdapter(fakeRuntime().runtime, prefix);
  const current = state("anime:101");
  const raw = adapter.createProfileBackup(current, new Map([["Invented", profile("Invented", current)]]));
  const parsed = adapter.parseProfileBackup(raw);
  assert.equal(parsed.version, 2);
  assert.deepEqual(parsed.state.watchlist, current.watchlist);
  assert.deepEqual(parsed.profiles[0].state.watchlist, current.watchlist);
  const { watchlist: _ignored, ...oldState } = current;
  const legacy = { format: PROFILE_BACKUP_FORMAT, version: 1,
    exportedAt: "2026-09-28T12:00:00.000Z", state: oldState,
    profiles: [profile("Invented", oldState)] };
  const upgraded = adapter.parseProfileBackup(JSON.stringify(legacy));
  assert.equal(upgraded.version, 2);
  assert.deepEqual(upgraded.state.watchlist, []);
  assert.deepEqual(upgraded.profiles[0].state.watchlist, []);
  assert.throws(() => adapter.parseProfileBackup(JSON.stringify({ ...legacy, state: current })));
  assert.throws(() => adapter.parseProfileBackup(JSON.stringify({ ...parsed,
    state: { ...parsed.state, watchlist: undefined } })));
  assert.throws(() => adapter.parseProfileBackup(JSON.stringify({ ...parsed,
    state: { ...parsed.state, watchlist: [{ ...current.watchlist![0], extra: true }] } })));
});

test("merge keeps local conflicting intent and adds imported unknown identities; replace previews losses", () => {
  const adapter = createPersistenceAdapter(fakeRuntime().runtime, prefix);
  const local = state("anime:101", 1.7);
  const imported = state("anime:101", 2.4);
  imported.preferences.push({ nodeId: "anime:999", sentiment: "disliked", importance: 2.1,
    confidence: 0.75, source: "import" });
  imported.history = [{ ...local.history![0], sourceId: "anime:999", score: 2 },
    { ...local.history![0], sourceId: "anime:998", animeId: 998, title: "Invented Other" }];
  imported.includeCandidates = ["anime:996"];
  imported.watchlist = [
    { animeId: 999, title: "Changed invented title", status: "completed", rating: 9 },
    { animeId: 998, title: "Imported invented title", status: "plan_to_watch", rating: null },
  ];
  const localProfiles = new Map([["Shared", profile("Shared", local)], ["Local", profile("Local", local)]]);
  const importedProfiles = new Map([["Shared", profile("Shared", imported)], ["Imported", profile("Imported", imported)]]);
  const document = adapter.parseProfileBackup(adapter.createProfileBackup(imported, importedProfiles));

  const merged = adapter.planProfileBackupImport(document, local, localProfiles, "merge");
  assert.equal(merged.state.mode, "hybrid");
  assert.equal(merged.state.preferences[0].importance, 1.7);
  assert.equal(merged.state.preferences[1].nodeId, "anime:999");
  assert.equal(merged.state.history?.[0].score, 9);
  assert.equal(merged.state.history?.[1].sourceId, "anime:998");
  assert.deepEqual(merged.state.includeCandidates, ["anime:998", "anime:996"]);
  assert.deepEqual(merged.state.watchlist, [local.watchlist![0], imported.watchlist[1]]);
  assert.equal(merged.profiles.get("Shared")?.state.preferences[0].importance, 1.7);
  assert.equal(merged.profiles.get("Imported")?.state.preferences[0].importance, 2.4);
  assert.deepEqual([merged.counts.addedPreferences, merged.counts.keptPreferences,
    merged.counts.addedHistory, merged.counts.keptHistory,
    merged.counts.addedProfiles, merged.counts.keptProfiles], [1, 1, 1, 1, 1, 1]);

  const replaced = adapter.planProfileBackupImport(document, local, localProfiles, "replace");
  assert.equal(replaced.state.preferences[0].importance, 2.4);
  assert.equal(replaced.profiles.has("Local"), false);
  assert.equal(replaced.counts.replacedPreferences, 1);
  assert.equal(replaced.counts.replacedHistory, 1);
  assert.equal(replaced.counts.replacedWatchlist, 1);
  assert.equal(merged.counts.addedWatchlist, 1);
  assert.equal(merged.counts.keptWatchlist, 1);
  assert.equal(replaced.counts.replacedProfiles, 1);
  assert.equal(replaced.counts.removedProfiles, 1);
});

test("two-key import rolls back both current keys if the second write is rejected", () => {
  const fake = fakeRuntime();
  const adapter = createPersistenceAdapter(fake.runtime, prefix);
  const beforeState = state("anime:101");
  const beforeProfiles = new Map([["Before", profile("Before", beforeState)]]);
  assert.equal(adapter.persistRecommendationState(beforeState), true);
  assert.equal(adapter.persistRecommendationProfiles(beforeProfiles), true);
  const beforeRaw = fake.values.get(stateKey);
  const beforeProfilesRaw = fake.values.get(profilesKey);
  const olderStateBackup = JSON.stringify(state("anime:777"));
  const olderProfilesBackup = JSON.stringify({ version: 5, profiles: [] });
  fake.values.set(`${stateKey}.backup`, olderStateBackup);
  fake.values.set(`${profilesKey}.backup`, olderProfilesBackup);
  const revision = adapter.getProfileStorageRevision();
  assert.ok(revision);
  fake.rejected.add(profilesKey);
  assert.equal(adapter.commitProfileCollection(state("anime:999"), new Map(), revision), false);
  assert.equal(fake.values.get(stateKey), beforeRaw);
  assert.equal(fake.values.get(profilesKey), beforeProfilesRaw);
  assert.equal(fake.values.get(`${stateKey}.backup`), olderStateBackup);
  assert.equal(fake.values.get(`${profilesKey}.backup`), olderProfilesBackup);
  assert.equal(fake.values.has(journalKey), false);
  assert.match(adapter.getStorageWarnings().join(" "), /not applied/i);
  fake.rejected.clear();
  assert.equal(adapter.commitProfileCollection(state("anime:999"), new Map(), revision), true);
  assert.equal(adapter.loadRecommendationState().preferences[0].nodeId, "anime:999");
});

test("interrupted import is rolled back on reopen; a denied rollback reads original journal bytes", () => {
  const fake = fakeRuntime();
  const beforeState = JSON.stringify(state("anime:101"));
  const beforeProfiles = JSON.stringify({ version: 5, profiles: [profile("Before", state("anime:101"))] });
  const journalKeys = [stateKey, profilesKey,
    `${stateKey}.backup`, `${stateKey}.corrupt`, `${profilesKey}.backup`, `${profilesKey}.corrupt`,
    `${prefix}.recommendationState.v4.backup`, `${prefix}.recommendationState.v1.backup`,
    `${prefix}.recommendationProfiles.v4.backup`, `${prefix}.recommendationProfiles.v1.backup`];
  const originals = Object.fromEntries(journalKeys.map((key) => [key, null]));
  originals[stateKey] = beforeState;
  originals[profilesKey] = beforeProfiles;
  fake.values.set(journalKey, JSON.stringify({ version: 1, originals }));
  fake.values.set(stateKey, JSON.stringify(state("anime:999")));
  fake.values.set(profilesKey, JSON.stringify({ version: 5, profiles: [] }));
  fake.rejected.add(stateKey);
  const blocked = createPersistenceAdapter(fake.runtime, prefix);
  assert.equal(blocked.loadRecommendationState().preferences[0].nodeId, "anime:101");
  assert.equal(blocked.loadRecommendationProfiles().has("Before"), true);
  assert.equal(blocked.persistRecommendationState(state("anime:998")), false);
  assert.equal(fake.values.has(journalKey), true);
  fake.rejected.clear();
  assert.equal(blocked.recoverInterruptedProfileWrite(), true);
  assert.equal(fake.values.get(stateKey), beforeState);
  assert.equal(fake.values.get(profilesKey), beforeProfiles);
  assert.equal(fake.values.has(journalKey), false);
  assert.equal(createPersistenceAdapter(fake.runtime, prefix).loadRecommendationState().preferences[0].nodeId,
    "anime:101");
});

test("reset keeps exact older sources and backups, and a stale preview cannot write", () => {
  const fake = fakeRuntime();
  const adapter = createPersistenceAdapter(fake.runtime, prefix);
  const oldRaw = JSON.stringify({ version: 3, mode: "graph", selected: [{ nodeId: "anime:999", weight: 2.4 }] });
  fake.values.set(`${prefix}.recommendationState.v1`, oldRaw);
  assert.equal(adapter.persistRecommendationState(state("anime:101")), true);
  const revision = adapter.getProfileStorageRevision();
  assert.ok(revision);
  assert.equal(adapter.persistRecommendationProfiles(new Map([["New", profile("New", state("anime:101"))]])), true);
  assert.equal(adapter.resetProfileCollection(revision), false);
  assert.equal(adapter.loadRecommendationState().preferences[0].nodeId, "anime:101");

  const freshRevision = adapter.getProfileStorageRevision();
  assert.ok(freshRevision);
  assert.equal(adapter.resetProfileCollection(freshRevision), true);
  assert.deepEqual(adapter.loadRecommendationState().preferences, []);
  assert.deepEqual(adapter.loadRecommendationProfiles(), new Map());
  assert.equal(fake.values.get(`${prefix}.recommendationState.v1`), oldRaw);
  assert.equal(fake.values.get(`${prefix}.recommendationState.v1.backup`), oldRaw);
  assert.equal(createPersistenceAdapter(fake.runtime, prefix).loadRecommendationState().preferences.length, 0);
});

test("repair may replace an invalid current copy while preserving its exact raw bytes", () => {
  const fake = fakeRuntime();
  fake.values.set(stateKey, "{invalid current");
  fake.values.set(`${stateKey}.backup`, JSON.stringify(state("anime:999", 2.4)));
  const adapter = createPersistenceAdapter(fake.runtime, prefix);
  const recovered = adapter.loadRecommendationState();
  const revision = adapter.getProfileStorageRevision();
  assert.ok(revision);
  assert.equal(adapter.commitProfileCollection({ version: 5, ...recovered }, new Map(), revision), true);
  assert.equal(fake.values.get(`${stateKey}.corrupt`), "{invalid current");
  assert.equal(adapter.loadRecommendationState().preferences[0].importance, 2.4);
});

test("explicit reset archives an unreadable interrupted-write journal and clears partial current keys", () => {
  const fake = fakeRuntime();
  fake.values.set(journalKey, "{unreadable journal");
  fake.values.set(stateKey, JSON.stringify(state("anime:999")));
  fake.values.set(profilesKey, JSON.stringify({ version: 5, profiles: [] }));
  const adapter = createPersistenceAdapter(fake.runtime, prefix);
  assert.deepEqual(adapter.loadRecommendationState().preferences, []);
  assert.equal(adapter.persistRecommendationState(state("anime:101")), false);
  const revision = adapter.getProfileStorageRevision();
  assert.ok(revision);
  assert.equal(adapter.resetProfileCollection(revision), true);
  assert.equal(fake.values.get(`${journalKey}.corrupt`), "{unreadable journal");
  assert.equal(fake.values.has(journalKey), false);
  assert.deepEqual(adapter.loadRecommendationState().preferences, []);
  assert.deepEqual(adapter.loadRecommendationProfiles(), new Map());
});

test("rejected reset restores an unreadable journal when no new recovery transaction remains", () => {
  const fake = fakeRuntime();
  fake.values.set(journalKey, "{unreadable journal");
  fake.values.set(stateKey, JSON.stringify(state("anime:999")));
  const adapter = createPersistenceAdapter(fake.runtime, prefix);
  const revision = adapter.getProfileStorageRevision();
  assert.ok(revision);
  fake.rejected.add(stateKey);
  assert.equal(adapter.resetProfileCollection(revision), false);
  assert.equal(fake.values.get(journalKey), "{unreadable journal");
  assert.equal(fake.values.get(`${journalKey}.corrupt`), "{unreadable journal");
  assert.equal(adapter.persistRecommendationState(state("anime:101")), false);
});
