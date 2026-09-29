import { readFile } from "node:fs/promises";
import { expect, test } from "@playwright/test";

const backupState = {
  version: 5, mode: "hybrid", modelBlendWeight: 0.35, allowRelatedTitles: true,
  preferences: [
    { nodeId: "anime:101", sentiment: "liked", importance: 2.4, confidence: 1, source: "manual" },
    { nodeId: "anime:999", sentiment: "disliked", importance: 1.8, confidence: 0.75, source: "import" },
  ],
  includeCandidates: ["anime:998"], excludeCandidates: ["anime:997"],
  history: [{ provider: "local", sourceId: "anime:999", title: "<Invented Unknown>", animeId: 999,
    status: "completed", sourceStatus: "Completed", progressEpisodes: 12, score: 2, scoreScale: "local-10" }],
};

const localState = {
  ...backupState, mode: "graph", modelBlendWeight: 0.5, allowRelatedTitles: false,
  preferences: [{ nodeId: "anime:101", sentiment: "liked", importance: 1.2, confidence: 1, source: "manual" }],
  includeCandidates: [], excludeCandidates: [], history: [],
};

const profile = (name: string, state: object) => ({ name, updatedAt: "2026-09-28T00:00:00Z", state });

test("local backup download, merge preview, and reload keep unknown identities and local conflicts", async ({ page }) => {
  await page.goto("/");
  await page.evaluate(({ backupState }) => {
    localStorage.setItem("wasiw.demo.recommendationState.v5", JSON.stringify(backupState));
    localStorage.setItem("wasiw.demo.recommendationProfiles.v5", JSON.stringify({ version: 5,
      profiles: [{ name: "<Shared> Profile", updatedAt: "2026-09-28T00:00:00Z", state: backupState }] }));
  }, { backupState });
  await page.reload();
  await page.locator("summary").filter({ hasText: "Import & Profiles" }).click();
  const downloadPromise = page.waitForEvent("download");
  await page.locator("#profile-export-btn").click();
  const download = await downloadPromise;
  const raw = await readFile(await download.path(), "utf8");
  const exported = JSON.parse(raw);
  expect(exported).toMatchObject({ format: "wasiw-profile-backup", version: 1, state: backupState });
  expect(exported.profiles[0].name).toBe("<Shared> Profile");

  await page.evaluate(({ localState }) => {
    localStorage.setItem("wasiw.demo.recommendationState.v5", JSON.stringify(localState));
    localStorage.setItem("wasiw.demo.recommendationProfiles.v5", JSON.stringify({ version: 5,
      profiles: [{ name: "<Shared> Profile", updatedAt: "2026-09-28T00:00:00Z", state: localState },
        { name: "Local Only", updatedAt: "2026-09-28T00:00:00Z", state: localState }] }));
  }, { localState });
  await page.reload();
  await page.locator("summary").filter({ hasText: "Import & Profiles" }).click();
  await page.locator("#profile-backup-file").setInputFiles({
    name: "invented-backup.json", mimeType: "application/json", buffer: Buffer.from(raw),
  });
  await expect(page.locator("#profile-backup-preview")).toBeVisible();
  await expect(page.locator("#profile-backup-summary")).toContainText("Merge adds 1 preferences, 1 history entries");
  await expect(page.locator("#profile-backup-unmapped")).toContainText("identities are retained");
  await expect(page.locator("#profile-backup-preview")).not.toContainText("<script>");
  await page.locator("#profile-backup-apply").click();
  await expect(page.locator("#profile-backup-status")).toContainText("applied and saved locally");
  await page.reload();
  const saved = await page.evaluate(() => ({
    state: JSON.parse(localStorage.getItem("wasiw.demo.recommendationState.v5") ?? "null"),
    profiles: JSON.parse(localStorage.getItem("wasiw.demo.recommendationProfiles.v5") ?? "null"),
  }));
  expect(saved.state.mode).toBe("graph");
  expect(saved.state.preferences).toEqual([localState.preferences[0], backupState.preferences[1]]);
  expect(saved.state.history).toEqual(backupState.history);
  expect(saved.state.includeCandidates).toEqual(["anime:998"]);
  expect(saved.profiles.profiles).toHaveLength(2);
  expect(saved.profiles.profiles.find((entry: { name: string }) => entry.name === "<Shared> Profile").state)
    .toEqual(localState);
  await expect(page.locator("#selected-anime")).toContainText("anime:999");
});

test("replace and reset previews show losses; reset keeps legacy source and current backup", async ({ page }) => {
  await page.goto("/");
  await page.evaluate(({ localState }) => {
    localStorage.setItem("wasiw.demo.recommendationState.v1", "invented untouched legacy bytes");
    localStorage.setItem("wasiw.demo.recommendationState.v5", JSON.stringify(localState));
    localStorage.setItem("wasiw.demo.recommendationProfiles.v5", JSON.stringify({ version: 5,
      profiles: [
        { name: "Local Only", updatedAt: "2026-09-28T00:00:00Z", state: localState },
      ] }));
  }, { localState });
  await page.reload();
  await page.locator("summary").filter({ hasText: "Import & Profiles" }).click();
  const raw = JSON.stringify({ format: "wasiw-profile-backup", version: 1,
    exportedAt: "2026-09-28T00:00:00Z", state: backupState,
    profiles: [profile("Imported Only", backupState)] });
  await page.locator("#profile-backup-file").setInputFiles({
    name: "invented-backup.json", mimeType: "application/json", buffer: Buffer.from(raw),
  });
  await page.locator("#profile-backup-mode").selectOption("replace");
  await expect(page.locator("#profile-backup-summary")).toContainText("Replace removes");
  await expect(page.locator("#profile-backup-summary")).toContainText("1 profiles absent from the file");
  await page.locator("#profile-backup-apply").click();
  await expect(page.locator("#profile-select")).toContainText("Imported Only");
  await expect(page.locator("#profile-select")).not.toContainText("Local Only");

  await page.locator("#profile-reset-preview-btn").click();
  await expect(page.locator("#profile-backup-summary")).toContainText("2 current preferences");
  await expect(page.locator("#profile-backup-mode-row")).toBeHidden();
  await page.locator("#profile-backup-apply").click();
  await expect(page.locator("#watched-count")).toHaveText("0");
  await expect(page.locator("#profile-select")).toBeDisabled();
  await page.reload();
  const rawStorage = await page.evaluate(() => ({
    state: JSON.parse(localStorage.getItem("wasiw.demo.recommendationState.v5") ?? "null"),
    profiles: JSON.parse(localStorage.getItem("wasiw.demo.recommendationProfiles.v5") ?? "null"),
    old: localStorage.getItem("wasiw.demo.recommendationState.v1"),
    prior: JSON.parse(localStorage.getItem("wasiw.demo.recommendationState.v5.backup") ?? "null"),
  }));
  expect(rawStorage.state.preferences).toEqual([]);
  expect(rawStorage.profiles.profiles).toEqual([]);
  expect(rawStorage.old).toBe("invented untouched legacy bytes");
  expect(rawStorage.prior.preferences).toEqual(backupState.preferences);
});

test("invalid file, rejected storage, and corrupt-current repair leave reviewable local evidence", async ({ page }) => {
  await page.goto("/");
  await page.evaluate(({ localState, backupState }) => {
    localStorage.setItem("wasiw.demo.recommendationState.v5", JSON.stringify(localState));
    localStorage.setItem("wasiw.demo.recommendationProfiles.v5", JSON.stringify({ version: 5, profiles: [] }));
    localStorage.setItem("wasiw.demo.recommendationState.v5.backup", JSON.stringify(backupState));
  }, { localState, backupState });
  await page.reload();
  await page.locator("summary").filter({ hasText: "Import & Profiles" }).click();
  await page.locator("#profile-backup-file").setInputFiles({
    name: "invalid.json", mimeType: "application/json", buffer: Buffer.from('{"version":99}'),
  });
  await expect(page.locator("#profile-backup-status")).toContainText("unsupported, or invalid");
  await expect(page.locator("#profile-backup-preview")).toBeHidden();

  await page.evaluate(() => {
    const original = Storage.prototype.setItem;
    Storage.prototype.setItem = function (key, value) {
      if (key === "wasiw.demo.recommendationProfiles.v5") {
        throw new DOMException("Invented quota rejection", "QuotaExceededError");
      }
      return original.call(this, key, value);
    };
  });
  const raw = JSON.stringify({ format: "wasiw-profile-backup", version: 1,
    exportedAt: "2026-09-28T00:00:00Z", state: backupState,
    profiles: [profile("Imported", backupState)] });
  await page.locator("#profile-backup-file").setInputFiles({
    name: "invented-backup.json", mimeType: "application/json", buffer: Buffer.from(raw),
  });
  await page.locator("#profile-backup-apply").click();
  await expect(page.locator("#profile-backup-status")).toContainText("storage rejected the change");
  expect(await page.evaluate(() => JSON.parse(localStorage.getItem("wasiw.demo.recommendationState.v5") ?? "null")))
    .toEqual(localState);
  expect(await page.evaluate(() => localStorage.getItem("wasiw.demo.profileWrite.v1.pending"))).toBeNull();

  await page.reload();
  await page.evaluate(({ backupState }) => {
    localStorage.setItem("wasiw.demo.recommendationState.v5", "{corrupt invented bytes");
    localStorage.setItem("wasiw.demo.recommendationState.v5.backup", JSON.stringify(backupState));
    localStorage.setItem("wasiw.demo.recommendationProfiles.v5", "{corrupt invented profiles");
    localStorage.setItem("wasiw.demo.recommendationProfiles.v5.backup", JSON.stringify({ version: 5,
      profiles: [{ name: "Recovered Invented", updatedAt: "2026-09-28T00:00:00Z", state: backupState }] }));
  }, { backupState });
  await page.reload();
  await expect(page.locator("#storage-status")).toContainText("Recovered recommendation state from a backup");
  await page.locator("summary").filter({ hasText: "Import & Profiles" }).click();
  await expect(page.locator("#profile-select")).toContainText("Recovered Invented");
  await page.locator("#profile-repair-btn").click();
  await expect(page.locator("#profile-backup-status")).toContainText("restored to current browser storage");
  expect(await page.evaluate(() => localStorage.getItem("wasiw.demo.recommendationState.v5.corrupt")))
    .toBe("{corrupt invented bytes");
  expect(await page.evaluate(() => localStorage.getItem("wasiw.demo.recommendationProfiles.v5.corrupt")))
    .toBe("{corrupt invented profiles");
  expect(await page.evaluate(() => JSON.parse(localStorage.getItem("wasiw.demo.recommendationState.v5") ?? "null")))
    .toEqual(backupState);
});

test("a delayed local backup read cannot preview after recommendation state changes", async ({ page }) => {
  await page.addInitScript(() => {
    const original = File.prototype.text;
    File.prototype.text = function () {
      return new Promise<string>((resolve, reject) => {
        setTimeout(() => original.call(this).then(resolve, reject), 300);
      });
    };
  });
  await page.goto("/");
  await page.locator("#anime-input").fill("Copper Comet");
  await page.locator("#add-anime-form button").click();
  await page.locator("summary").filter({ hasText: "Import & Profiles" }).click();
  const raw = JSON.stringify({ format: "wasiw-profile-backup", version: 1,
    exportedAt: "2026-09-28T00:00:00Z", state: backupState, profiles: [] });
  await page.locator("#profile-backup-file").setInputFiles({
    name: "invented-backup.json", mimeType: "application/json", buffer: Buffer.from(raw),
  });
  await expect(page.locator("#profile-backup-status")).toContainText("Reading local profile backup");
  await page.locator("#clear-watched").click();
  await page.waitForTimeout(400);
  await expect(page.locator("#profile-backup-preview")).toBeHidden();
  await expect(page.locator("#profile-backup-status")).toContainText("data changed");
  await expect(page.locator("#watched-count")).toHaveText("0");
});

test("legacy unknown profile migrates once, survives backup reset and import, and keeps raw sources", async ({ page }) => {
  await page.goto("/");
  const legacyState = { version: 3, mode: "hybrid", modelBlendWeight: 0.35,
    selected: [{ nodeId: "anime:999", weight: 2.4 }],
    includeCandidates: ["anime:998"], excludeCandidates: ["anime:997"] };
  const legacyRaw = JSON.stringify(legacyState);
  const legacyProfilesRaw = JSON.stringify([{ name: "Legacy Unknown",
    updatedAt: "2026-01-01T00:00:00Z", state: legacyState }]);
  await page.evaluate(({ legacyRaw, legacyProfilesRaw }) => {
    localStorage.setItem("wasiw.demo.recommendationState.v1", legacyRaw);
    localStorage.setItem("wasiw.demo.recommendationProfiles.v1", legacyProfilesRaw);
  }, { legacyRaw, legacyProfilesRaw });
  await page.reload();
  await page.locator("summary").filter({ hasText: "Import & Profiles" }).click();
  const downloadPromise = page.waitForEvent("download");
  await page.locator("#profile-export-btn").click();
  const raw = await readFile(await (await downloadPromise).path(), "utf8");
  const exported = JSON.parse(raw);
  expect(exported.state.preferences[0]).toMatchObject({ nodeId: "anime:999", importance: 2.4 });
  expect(exported.profiles[0].state.preferences[0].importance).toBe(2.4);

  await page.locator("#profile-reset-preview-btn").click();
  await page.locator("#profile-backup-apply").click();
  await page.locator("#profile-backup-file").setInputFiles({
    name: "legacy-invented.json", mimeType: "application/json", buffer: Buffer.from(raw),
  });
  await page.locator("#profile-backup-mode").selectOption("replace");
  await page.locator("#profile-backup-apply").click();
  await page.reload();
  const first = await page.evaluate(() => ({
    state: localStorage.getItem("wasiw.demo.recommendationState.v5"),
    profiles: localStorage.getItem("wasiw.demo.recommendationProfiles.v5"),
    oldState: localStorage.getItem("wasiw.demo.recommendationState.v1"),
    oldProfiles: localStorage.getItem("wasiw.demo.recommendationProfiles.v1"),
    stateBackup: localStorage.getItem("wasiw.demo.recommendationState.v1.backup"),
    profilesBackup: localStorage.getItem("wasiw.demo.recommendationProfiles.v1.backup"),
  }));
  expect(JSON.parse(first.state ?? "null").preferences[0].importance).toBe(2.4);
  expect(JSON.parse(first.profiles ?? "null").profiles[0].name).toBe("Legacy Unknown");
  expect(first.oldState).toBe(legacyRaw);
  expect(first.oldProfiles).toBe(legacyProfilesRaw);
  expect(first.stateBackup).toBe(legacyRaw);
  expect(first.profilesBackup).toBe(legacyProfilesRaw);
  await page.reload();
  const second = await page.evaluate(() => ({
    state: localStorage.getItem("wasiw.demo.recommendationState.v5"),
    profiles: localStorage.getItem("wasiw.demo.recommendationProfiles.v5"),
  }));
  expect(second.state).toBe(first.state);
  expect(second.profiles).toBe(first.profiles);
  await expect(page.locator("#selected-anime")).toContainText("anime:999");
});

test("profile and file controls stay unavailable until the local catalog and handlers are ready", async ({ page }) => {
  let releaseGraph!: () => void;
  const graphGate = new Promise<void>((resolve) => { releaseGraph = resolve; });
  await page.route("**/demo-data/graph.compact.json", async (route) => {
    await graphGate;
    await route.continue();
  });
  const navigation = page.goto("/");
  await expect(page.locator("#startup-status")).toBeVisible();
  await expect(page.locator("#app")).toBeHidden();
  releaseGraph();
  await navigation;
  await expect(page.locator("#app")).toBeVisible();
  await expect(page.locator("#startup-status")).toHaveCount(0);
  await page.locator("summary").filter({ hasText: "Import & Profiles" }).click();
  await page.locator("#profile-reset-preview-btn").click();
  await expect(page.locator("#profile-backup-preview")).toBeVisible();
});
