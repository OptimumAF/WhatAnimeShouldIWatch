import assert from "node:assert/strict";
import { test } from "node:test";
import { createPersistenceAdapter } from "../src/persistence.ts";
import { parseTextHistory } from "../src/import-history.ts";
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

test("legacy state and named profiles migrate with exact raw backups and unknown IDs", () => {
  const fake = storageRuntime();
  const stateRaw = JSON.stringify(oldState);
  const profilesRaw = JSON.stringify([{ name: "Fixture Profile", updatedAt: "2026-01-01T00:00:00Z", state: oldState }]);
  fake.values.set("wasiw.demo.recommendationState.v1", stateRaw);
  fake.values.set("wasiw.demo.recommendationProfiles.v1", profilesRaw);
  const persistence = createPersistenceAdapter(fake.runtime, "wasiw.demo");

  assert.deepEqual(persistence.loadRecommendationState(), {
    mode: "hybrid", selected: oldState.selected, modelBlendWeight: 0.35,
    includeCandidates: oldState.includeCandidates, excludeCandidates: oldState.excludeCandidates,
    history: [],
  });
  assert.deepEqual(persistence.loadRecommendationProfiles().get("Fixture Profile")?.state, {
    ...oldState, version: 4, history: [],
  });
  assert.equal(fake.values.get("wasiw.demo.recommendationState.v1.backup"), stateRaw);
  assert.equal(fake.values.get("wasiw.demo.recommendationProfiles.v1.backup"), profilesRaw);
  assert.equal(fake.values.get("wasiw.demo.recommendationState.v1"), stateRaw);
  assert.equal(fake.values.get("wasiw.demo.recommendationProfiles.v1"), profilesRaw);
  assert.deepEqual(JSON.parse(fake.values.get("wasiw.demo.recommendationState.v4") ?? "null"), {
    ...oldState, version: 4, history: [],
  });
  assert.deepEqual(JSON.parse(fake.values.get("wasiw.demo.recommendationProfiles.v4") ?? "null"), {
    version: 4,
    profiles: [{ name: "Fixture Profile", updatedAt: "2026-01-01T00:00:00Z", state: { ...oldState, version: 4, history: [] } }],
  });
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
      mode: "model", selected: [{ nodeId: "anime:999", weight: 2.2 }],
      modelBlendWeight: 0.5, includeCandidates: [], excludeCandidates: [], history: [],
    });
  }
});

test("full imported history survives state and profile reload without dropping unknown identities", () => {
  const fake = storageRuntime();
  const history = parseTextHistory("101, 9, Completed, 12\n99999, 0, Plan to Watch, 0").entries;
  const persistence = createPersistenceAdapter(fake.runtime, "wasiw.demo");
  const state = {
    version: 4, mode: "graph" as const,
    selected: [{ nodeId: "anime:101", weight: 1.8 }],
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
  const raw = fake.values.get("wasiw.demo.recommendationState.v4");
  assert.ok(raw);
  assert.equal(reopened.persistRecommendationState({ ...state, history: [...history, history[0]] }), false);
  assert.equal(fake.values.get("wasiw.demo.recommendationState.v4"), raw);
});

test("corrupt current state and profiles recover from valid backups without overwriting them", () => {
  const fake = storageRuntime();
  fake.values.set("wasiw.demo.recommendationState.v4", "{broken");
  fake.values.set("wasiw.demo.recommendationState.v4.backup", JSON.stringify({ ...oldState, version: 4 }));
  fake.values.set("wasiw.demo.recommendationProfiles.v4", "{broken");
  const profileBackup = JSON.stringify({ version: 4, profiles: [
    { name: "Fixture Profile", updatedAt: "2026-01-01T00:00:00Z", state: { ...oldState, version: 4 } },
  ] });
  fake.values.set("wasiw.demo.recommendationProfiles.v4.backup", profileBackup);
  const persistence = createPersistenceAdapter(fake.runtime, "wasiw.demo");

  assert.equal(persistence.loadRecommendationState().selected[1]?.nodeId, "anime:999");
  assert.equal(persistence.loadRecommendationProfiles().get("Fixture Profile")?.state.excludeCandidates?.[1], "anime:997");
  assert.equal(fake.values.get("wasiw.demo.recommendationState.v4"), "{broken");
  assert.equal(fake.values.get("wasiw.demo.recommendationProfiles.v4.backup"), profileBackup);
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
  assert.equal(persistence.loadRecommendationState().selected[1]?.nodeId, "anime:999");
  assert.equal(fake.values.get("wasiw.demo.recommendationState.v1"), "{broken");
  assert.deepEqual(persistence.loadRecommendationProfiles(), new Map());
  assert.equal(fake.values.has("wasiw.demo.recommendationProfiles.v4"), false);
  assert.match(persistence.getStorageWarnings().join(" "), /profiles.*could not be read/i);
});

test("quota rejection leaves legacy source and current bytes intact and reports unsaved state", () => {
  const fake = storageRuntime();
  const raw = JSON.stringify(oldState);
  fake.values.set("wasiw.demo.recommendationState.v1", raw);
  fake.rejected.add("wasiw.demo.recommendationState.v1.backup");
  const persistence = createPersistenceAdapter(fake.runtime, "wasiw.demo");
  assert.equal(persistence.loadRecommendationState().selected[1]?.nodeId, "anime:999");
  assert.equal(fake.values.has("wasiw.demo.recommendationState.v4"), false);
  assert.equal(fake.values.get("wasiw.demo.recommendationState.v1"), raw);
  assert.match(persistence.getStorageWarnings().join(" "), /storage|backup/i);

  fake.rejected.clear();
  assert.equal(persistence.persistRecommendationState({ ...oldState, version: 4 }), true);
  const current = fake.values.get("wasiw.demo.recommendationState.v4");
  fake.rejected.add("wasiw.demo.recommendationState.v4");
  assert.equal(persistence.persistRecommendationState({ ...oldState, version: 4, mode: "graph" }), false);
  assert.equal(fake.values.get("wasiw.demo.recommendationState.v4"), current);
  assert.match(persistence.getStorageWarnings().join(" "), /storage|saved/i);
});

test("failed backup of an existing current version prevents an overwrite", () => {
  const fake = storageRuntime();
  const current = JSON.stringify({ ...oldState, version: 4 });
  fake.values.set("wasiw.demo.recommendationState.v4", current);
  fake.rejected.add("wasiw.demo.recommendationState.v4.backup");
  const persistence = createPersistenceAdapter(fake.runtime, "wasiw.demo");
  assert.equal(persistence.persistRecommendationState({ ...oldState, version: 4, mode: "graph" }), false);
  assert.equal(fake.values.get("wasiw.demo.recommendationState.v4"), current);
});
