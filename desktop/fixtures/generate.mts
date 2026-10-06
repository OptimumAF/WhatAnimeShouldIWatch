import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { buildReleaseManifest, verifyReleaseBundle } from "../../pipeline/src/release-manifest.ts";

const outputDir = path.dirname(fileURLToPath(import.meta.url));
const inputDir = path.resolve(outputDir, "../../web/public/demo-data");
const outputs = new Map<string, Buffer>([
  ["graph.compact.json", fs.readFileSync(path.join(inputDir, "graph.aggregate.compact.json"))],
  ["graph-explorer.compact.json", fs.readFileSync(path.join(inputDir,
    "graph-explorer.aggregate.compact.json"))],
  ["catalog.identity.json", fs.readFileSync(path.join(inputDir, "catalog.identity.json"))],
]);
const manifest = buildReleaseManifest({
  neighborhood: outputs.get("graph.compact.json")!,
  explorer: outputs.get("graph-explorer.compact.json")!,
  catalog: outputs.get("catalog.identity.json")!,
}, { tag: "data-vsynthetic-desktop-v1", fixtureGenesis: true });
outputs.set("release-manifest.json", Buffer.from(`${JSON.stringify(manifest, null, 2)}\n`));

const check = process.argv.includes("--check");
for (const [filename, expected] of outputs) {
  const target = path.join(outputDir, filename);
  if (check) {
    if (!fs.existsSync(target) || !fs.readFileSync(target).equals(expected)) {
      throw new Error(`Synthetic desktop fixture is missing or stale: ${filename}.`);
    }
  } else {
    fs.writeFileSync(target, expected);
  }
}
verifyReleaseBundle(outputDir, undefined, true);
process.stdout.write(`${check ? "Verified" : "Generated"} invented desktop v3 bundle: ` +
  `${manifest.catalog.animeCount} anime, ${manifest.tag}.\n`);
