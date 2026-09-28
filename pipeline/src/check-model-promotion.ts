// Read-only model promotion preflight; it never publishes an artifact.

import path from "node:path";
import { Command } from "commander";
import { getRepoRoot } from "./paths.js";
import { verifyModelPromotion } from "./model-promotion-preflight.js";

const repoRoot = getRepoRoot(import.meta.url);
const program = new Command()
  .name("check-model-promotion")
  .description("Check a private model bundle, committed review, and rollback bundle.")
  .requiredOption("--candidate-dir <path>")
  .requiredOption("--rollback-dir <path>")
  .parse(process.argv);
const options = program.opts<{ candidateDir: string; rollbackDir: string }>();
const result = verifyModelPromotion({
  candidateDir: path.resolve(options.candidateDir),
  rollbackDir: path.resolve(options.rollbackDir),
  approvalsPath: path.join(repoRoot, "docs/approvals/model-promotions.json"),
  repoRoot,
});
process.stdout.write(`${JSON.stringify(result)}\n`);
