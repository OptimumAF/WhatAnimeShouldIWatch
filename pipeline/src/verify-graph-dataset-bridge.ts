/** Read-only private bridge check; prints only aggregate hashes and counts. */
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Command } from "commander";
import { verifyGraphDatasetBridgeFiles } from "./core/split-graph-bridge.js";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const fixtureInputs = [
  ["fixtures/synthetic-split-input.json", "cf6c53386175613610363b90ace3deb473bc93edbc348e2b099d91634bc23b18"],
  ["fixtures/synthetic-split-manifest.json", "3b21b267a7a9eaa0b677ca7ad1a20ae208b3df2ad7b2826fefeccdfe78bd6b76"],
  ["fixtures/synthetic-anime-metadata.json", "2a2ec17221c63a46d44d85699e72f193f31c1cf5c7f7741a6151b462776f0694"],
] as const;
const command = new Command()
  .requiredOption("--raw-ratings <path>")
  .requiredOption("--split-manifest <path>")
  .requiredOption("--metadata <path>")
  .requiredOption("--graph <path>")
  .requiredOption("--refit-record <path>")
  .requiredOption("--source-name <name>")
  .option("--training-approval-ref <url>");
command.parse();
const options = command.opts();
try {
  const fixture = [options.rawRatings, options.splitManifest, options.metadata]
    .every((name: string, index: number) => {
      const [expectedPath, expectedHash] = fixtureInputs[index];
      if (fs.realpathSync(name) !== fs.realpathSync(path.join(root, expectedPath))) return false;
      const normalized = fs.readFileSync(name, "utf8").replace(/\r\n/g, "\n");
      return createHash("sha256").update(normalized).digest("hex") === expectedHash;
    });
  if (!fixture) {
    if (typeof options.trainingApprovalRef !== "string" || !options.trainingApprovalRef) {
      throw new Error("Graph dataset bridge private inputs require a recorded training approval");
    }
    const approved = spawnSync("python", [path.join(root, "scripts/verify_provider_data_approval.py"),
      "training"], { cwd: root, encoding: "utf8", env: {
        ...process.env, PROVIDER_DATA_APPROVAL_REF: options.trainingApprovalRef,
      } });
    if (approved.error || approved.status !== 0) {
      throw new Error("Graph dataset bridge training source/use approval is absent or invalid");
    }
  }
  process.stdout.write(`${JSON.stringify(verifyGraphDatasetBridgeFiles({
    rawRatings: options.rawRatings, splitManifest: options.splitManifest,
    metadata: options.metadata, graph: options.graph,
    refitRecord: options.refitRecord, sourceName: options.sourceName,
  }))}\n`);
} catch (error) {
  process.stderr.write(`${error instanceof Error ? error.message : "Graph bridge refused"}\n`);
  process.exitCode = 1;
}
