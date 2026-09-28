import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { parseCompactGraph, parseDemoCatalog } from "../src/artifacts.ts";
import { buildRecommendationIndexFromCompact } from "../src/recommendations.ts";
import { searchAnimeTitles } from "../src/title-search.ts";

const fixture = (name: string): unknown => JSON.parse(readFileSync(
  new URL(`../public/demo-data/${name}`, import.meta.url), "utf8"));
const index = buildRecommendationIndexFromCompact(parseCompactGraph(fixture("graph.compact.json"), "graph fixture"));
const metadata = new Map(parseDemoCatalog(fixture("catalog.json"), "catalog fixture")
  .map((item) => [item.animeId, item] as const));

test("exact catalog IDs, unique canonical titles, and known aliases resolve explicitly", () => {
  for (const query of ["101", "anime:101", "Copper Comet"]) {
    assert.equal(searchAnimeTitles(query, index, metadata).automatic?.animeId, 101);
  }
  assert.equal(searchAnimeTitles("Moonlit Studio", index, metadata).automatic?.animeId, 102);
});

test("ambiguous aliases and every partial match require a choice", () => {
  const alias = searchAnimeTitles("Galaxy Route", index, metadata);
  assert.equal(alias.automatic, null);
  assert.deepEqual(alias.matches.map((item) => item.anime.animeId), [101, 104]);
  assert.equal(searchAnimeTitles("Copper", index, metadata).automatic, null);
  assert.equal(searchAnimeTitles("Copper", index, metadata).total, 1);
  const broad = searchAnimeTitles("o", index, metadata, 2);
  assert.equal(broad.matches.length, 2);
  assert.ok(broad.total > 2);
  assert.equal(broad.automatic, null);
});

test("an exact canonical title colliding with another title's alias is not auto-selected", () => {
  const changed = new Map(metadata);
  changed.set(104, { ...metadata.get(104)!, aliases: ["Copper Comet"] });
  const found = searchAnimeTitles("Copper Comet", index, changed);
  assert.equal(found.automatic, null);
  assert.deepEqual(found.matches.map((item) => item.anime.animeId), [101, 104]);
  assert.equal(found.matches[1].matchedAlias, "Copper Comet");
});

test("unknown title and ID do not invent a catalog mapping", () => {
  assert.equal(searchAnimeTitles("Unmapped Invented Title", index, metadata).total, 0);
  assert.equal(searchAnimeTitles("anime:999", index, metadata).automatic, null);
});
