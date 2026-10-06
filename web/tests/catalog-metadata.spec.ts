import { expect, test } from "@playwright/test";

test("browser metadata candidate contract keeps unknown values and rejects hidden fields", async ({ page }) => {
  await page.goto("/");
  const result = await page.evaluate(async () => {
    const { catalogMetadataCoverage, parseCatalogMetadataSnapshot } =
      await import("/src/artifacts.ts");
    const invented = {
      format: "anime-metadata-catalog-v1",
      source: { name: "invented-browser-test", snapshotAt: "2026-09-24T00:00:00.000Z",
        snapshotSha256: "a".repeat(64) },
      anime: [{ animeId: 101, sourceItemId: "invented:101", title: "Copper Comet",
        aliases: null, genres: ["Adventure"], year: 2021, mediaFormat: "TV",
        episodeCount: 12, runtimeMinutes: 24, contentClassification: null,
        communityScore: null, relations: null }],
    };
    const parsed = parseCatalogMetadataSnapshot(invented, "catalog.metadata.json");
    const coverage = catalogMetadataCoverage(parsed, [101, 102]);
    let error = "";
    try {
      parseCatalogMetadataSnapshot({ ...invented, userRows: [{ userId: "invented" }] },
        "catalog.metadata.json");
    } catch (caught) {
      error = caught instanceof Error ? caught.message : "unknown error";
    }
    return { coverage, error };
  });
  expect(result.coverage).toMatchObject({ total: 2, missingItems: 1,
    known: { genres: 1, aliases: 0, communityScore: 0 },
    usable: { genres: 1, aliases: 0 } });
  expect(result.error).toContain("catalog.metadata.json: root.userRows is unsupported");
  expect(result.error).not.toContain("userId");
});
