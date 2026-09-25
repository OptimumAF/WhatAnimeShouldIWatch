import assert from "node:assert/strict";
import { test } from "node:test";
import { createPersistenceAdapter } from "../src/persistence.ts";
import { parseTextHistory } from "../src/import-history.ts";
import { migrateLegacyPreferences } from "../src/preferences.ts";
import type { RuntimePorts, StoragePort } from "../src/runtime.ts";

function storageRuntime() {
  const values = new Map<string, string>();
  const rejected = new Set<string>();
  const storage: StoragePort = {
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => {
      if (rejected.has(key)) throw new Error("synthetic quota exceeded");
      values.set(key, value);
    },
    removeItem: (key) => { values.delete(key); },
  };
  const runtime = {
    storage,
    now: () => new Date("2026-09-24T12:34:56.000Z"),
  } as RuntimePorts;
  return { values, rejected, runtime };
}

const oldState = {
  version: 3,
  mode: "hybrid",
  selected: [
    { nodeId: "anime:101", weight: 1.7 },
    { nodeId: "anime:999", weight: 2.4 },
  ],
  modelBlendWeight: 0.35,
  includeCandidates: ["anime:102", "anime:998"],
  excludeCandidates: ["anime:105", "anime:997"],
};
const migrated = migrateLegacyPreferences(oldState.selected, []);
const v5State = {
  version: 5, mode: oldState.mode, preferences: migrated,
  modelBlendWeight: oldState.modelBlendWeight,
  allowRelatedTitles: false,
  includeCandidates: oldState.includeCandidates,
  excludeCandidates: oldState.excludeCandidates,
  history: [],
};

test("legacy state and named profiles migrate with exact raw backups and unknown IDs", () => {
  const fake = storageRuntime();
  const stateRaw = JSON.stringify(oldState);
  const profilesRaw = JSON.stringify([{ name: "Fixture Profile", updatedAt: "2026-01-01T00:00:00Z", state: oldState }]);
  fake.values.set("wasiw.demo.recommendationState.v1", stateRaw);
  fake.values.set("wasiw.demo.recommendationProfiles.v1", profilesRaw);
  const persistence = createPersistenceAdapter(fake.runtime, "wasiw.demo");

  assert.deepEqual(persistence.loadRecommendationState(), {
    mode: "hybrid", preferences: migrated, modelBlendWeight: 0.35, allowRelatedTitles: false,
    includeCandidates: oldState.includeCandidates, excludeCandidates: oldState.excludeCandidates,
    history: [],
  });
  assert.deepEqual(persistence.loadRecommendationProfiles().get("Fixture Profile")?.state, {
    ...v5State,
  });
  assert.equal(fake.values.get("wasiw.demo.recommendationState.v1.backup"), stateRaw);
  assert.equal(fake.values.get("wasiw.demo.recommendationProfiles.v1.backup"), profilesRaw);
  assert.equal(fake.values.get("wasiw.demo.recommendationState.v1"), stateRaw);
  assert.equal(fake.values.get("wasiw.demo.recommendationProfiles.v1"), profilesRaw);
  assert.deepEqual(JSON.parse(fake.values.get("wasiw.demo.recommendationState.v5") ?? "null"), v5State);
  assert.deepEqual(JSON.parse(fake.values.get("wasiw.demo.recommendationProfiles.v5") ?? "null"), {
    version: 5,
    profiles: [{ name: "Fixture Profile", updatedAt: "2026-01-01T00:00:00Z", state: v5State }],
  });
  assert.match(persistence.getMigrationNotice() ?? "", /migrated conservatively/i);
  assert.deepEqual(persistence.getStorageWarnings(), []);

  persistence.loadRecommendationState();
  persistence.loadRecommendationProfiles();
  assert.equal(fake.values.get("wasiw.demo.recommendationState.v1.backup"), stateRaw);
  assert.equal(fake.values.get("wasiw.demo.recommendationProfiles.v1.backup"), profilesRaw);
});

test("v1 and v2 state defaults migrate without dropping watched weights", () => {
  for (const version of [1, 2]) {
    const fake = storageRuntime();
    fake.values.set("wasiw.demo.recommendationState.v1", JSON.stringify({
      version, mode: "model", selected: [{ nodeId: "anime:999", weight: 2.2 }],
    }));
    const state = createPersistenceAdapter(fake.runtime, "wasiw.demo").loadRecommendationState();
    assert.deepEqual(state, {
      mode: "model", preferences: [{ nodeId: "anime:999", sentiment: "liked", importance: 2.2,
        confidence: 0.5, source: "legacy" }],
      modelBlendWeight: 0.5, allowRelatedTitles: false,
      includeCandidates: [], excludeCandidates: [], history: [],
    });
  }
});

test("related-title option persists in state and profiles without a storage-version change", () => {
  const fake = storageRuntime();
  const persistence = createPersistenceAdapter(fake.runtime, "wasiw.demo");
  const state = { ...v5State, mode: "hybrid" as const, allowRelatedTitles: true };
  assert.equal(persistence.persistRecommendationState(state), true);
  assert.equal(persistence.persistRecommendationProfiles(new Map([["Variety override", {
    name: "Variety override", updatedAt: "2026-09-25T00:00:00Z", state,
  }]])), true);
  const reopened = createPersistenceAdapter(fake.runtime, "wasiw.demo");
  assert.equal(reopened.loadRecommendationState().allowRelatedTitles, true);
  assert.equal(reopened.loadRecommendationProfiles().get("Variety override")?.state.allowRelatedTitles, true);
  assert.equal(persistence.persistRecommendationState({ ...state, allowRelatedTitles: "yes" } as any), false);
});

test("full imported history survives state and profile reload without dropping unknown identities", () => {
  const fake = storageRuntime();
  const history = parseTextHistory("101, 9, Completed, 12\n99999, 0, Plan to Watch, 0").entries;
  const persistence = createPersistenceAdapter(fake.runtime, "wasiw.demo");
  const state = {
    version: 5, mode: "graph" as const,
    preferences: [{ nodeId: "anime:101", sentiment: "liked" as const,
      importance: 1.8, confidence: 0.75, source: "import" as const }],
    history,
  };
  assert.equal(persistence.persistRecommendationState(state), true);
  const profile = new Map([["Fixture history", {
    name: "Fixture history", updatedAt: "2026-09-25T00:00:00.000Z", state,
  }]]);
  assert.equal(persistence.persistRecommendationProfiles(profile), true);

  const reopened = createPersistenceAdapter(fake.runtime, "wasiw.demo");
  assert.deepEqual(reopened.loadRecommendationState().history, history);
  assert.deepEqual(reopened.loadRecommendationProfiles().get("Fixture history")?.state.history, history);
  const raw = fake.values.get("wasiw.demo.recommendationState.v5");
  assert.ok(raw);
  assert.equal(reopened.persistRecommendationState({ ...state, history: [...history, history[0]] }), false);
  assert.equal(fake.values.get("wasiw.demo.recommendationState.v5"), raw);
});

test("corrupt current state and profiles recover from valid backups without overwriting them", () => {
  const fake = storageRuntime();
  fake.values.set("wasiw.demo.recommendationState.v5", "{broken");
  fake.values.set("wasiw.demo.recommendationState.v5.backup", JSON.stringify(v5State));
  fake.values.set("wasiw.demo.recommendationProfiles.v5", "{broken");
  const profileBackup = JSON.stringify({ version: 5, profiles: [
    { name: "Fixture Profile", updatedAt: "2026-01-01T00:00:00Z", state: v5State },
  ] });
  fake.values.set("wasiw.demo.recommendationProfiles.v5.backup", profileBackup);
  const persistence = createPersistenceAdapter(fake.runtime, "wasiw.demo");

  assert.equal(persistence.loadRecommendationState().preferences[1]?.nodeId, "anime:999");
  assert.equal(persistence.loadRecommendationProfiles().get("Fixture Profile")?.state.excludeCandidates?.[1], "anime:997");
  assert.equal(fake.values.get("wasiw.demo.recommendationState.v5"), "{broken");
  assert.equal(fake.values.get("wasiw.demo.recommendationProfiles.v5.backup"), profileBackup);
  assert.match(persistence.getStorageWarnings().join(" "), /backup/i);
});

test("a corrupt legacy copy recovers from its backup and a malformed profile set is not partly loaded", () => {
  const fake = storageRuntime();
  fake.values.set("wasiw.demo.recommendationState.v1", "{broken");
  fake.values.set("wasiw.demo.recommendationState.v1.backup", JSON.stringify(oldState));
  fake.values.set("wasiw.demo.recommendationProfiles.v1", JSON.stringify([
    { name: "Good Profile", state: oldState },
    { name: "Broken Profile", state: { ...oldState, excludeCandidates: [null] } },
  ]));
  const persistence = createPersistenceAdapter(fake.runtime, "wasiw.demo");
  assert.equal(persistence.loadRecommendationState().preferences[1]?.nodeId, "anime:999");
  assert.equal(fake.values.get("wasiw.demo.recommendationState.v1"), "{broken");
  assert.deepEqual(persistence.loadRecommendationProfiles(), new Map());
  assert.equal(fake.values.has("wasiw.demo.recommendationProfiles.v5"), false);
  assert.match(persistence.getStorageWarnings().join(" "), /profiles.*could not be read/i);
});

test("quota rejection leaves legacy source and current bytes intact and reports unsaved state", () => {
  const fake = storageRuntime();
  const raw = JSON.stringify(oldState);
  fake.values.set("wasiw.demo.recommendationState.v1", raw);
  fake.rejected.add("wasiw.demo.recommendationState.v1.backup");
  const persistence = createPersistenceAdapter(fake.runtime, "wasiw.demo");
  assert.equal(persistence.loadRecommendationState().preferences[1]?.nodeId, "anime:999");
  assert.equal(fake.values.has("wasiw.demo.recommendationState.v5"), false);
  assert.equal(fake.values.get("wasiw.demo.recommendationState.v1"), raw);
  assert.match(persistence.getStorageWarnings().join(" "), /storage|backup/i);

  fake.rejected.clear();
  assert.equal(persistence.persistRecommendationState(v5State), true);
  const current = fake.values.get("wasiw.demo.recommendationState.v5");
  fake.rejected.add("wasiw.demo.recommendationState.v5");
  assert.equal(persistence.persistRecommendationState({ ...v5State, mode: "graph" }), false);
  assert.equal(fake.values.get("wasiw.demo.recommendationState.v5"), current);
  assert.match(persistence.getStorageWarnings().join(" "), /storage|saved/i);
});

test("failed backup of an existing current version prevents an overwrite", () => {
  const fake = storageRuntime();
  const current = JSON.stringify(v5State);
  fake.values.set("wasiw.demo.recommendationState.v5", current);
  fake.rejected.add("wasiw.demo.recommendationState.v5.backup");
  const persistence = createPersistenceAdapter(fake.runtime, "wasiw.demo");
  assert.equal(persistence.persistRecommendationState({ ...v5State, mode: "graph" }), false);
  assert.equal(fake.values.get("wasiw.demo.recommendationState.v5"), current);
});

test("v4 scores migrate low, neutral, high, and unknown entries without losing importance", () => {
  const fake = storageRuntime();
  const history = parseTextHistory("101, 2, Completed, 12\n102, 6, Watching, 3\n103, 9, Completed, 12").entries;
  const old = { version: 4, mode: "graph", selected: [
    { nodeId: "anime:101", weight: 2 }, { nodeId: "anime:102", weight: 1.8 },
    { nodeId: "anime:103", weight: 1 }, { nodeId: "anime:999", weight: 1 },
  ], history };
  const raw = JSON.stringify(old);
  fake.values.set("wasiw.demo.recommendationState.v4", raw);
  const state = createPersistenceAdapter(fake.runtime, "wasiw.demo").loadRecommendationState();
  assert.deepEqual(state.preferences.map(({ nodeId, sentiment, importance, confidence }) =>
    [nodeId, sentiment, importance, confidence]), [
    ["anime:101", "disliked", 2, 0.75], ["anime:102", "seen", 1.8, 0],
    ["anime:103", "liked", 1, 0.75], ["anime:999", "seen", 1, 0],
  ]);
  assert.deepEqual(state.history, history);
  assert.equal(fake.values.get("wasiw.demo.recommendationState.v4"), raw);
  assert.equal(fake.values.get("wasiw.demo.recommendationState.v4.backup"), raw);
});
