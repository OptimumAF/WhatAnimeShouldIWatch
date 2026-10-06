/** Local synthetic budget gate; real-data/device acceptance remains a separate review. */
import { execFileSync } from "node:child_process";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { checkPerformanceBudgets } from "./performance-budget-rules.mjs";

const reportRoot = resolve(fileURLToPath(new URL("../test-results/", import.meta.url)));
const repoRoot = resolve(reportRoot, "../..");
const reports = await Promise.all(["performance-baseline.json", "performance-scale.json"]
  .map(async (name) => {
    try {
      return JSON.parse(await readFile(resolve(reportRoot, name), "utf8"));
    } catch (error) {
      if (error?.code === "ENOENT") {
        process.stderr.write(`Missing ${name}; run both browser benchmark commands first.\n`);
        process.exit(1);
      }
      throw error;
    }
  }));
const currentRevision = execFileSync("git", ["rev-parse", "HEAD"], {
  cwd: repoRoot, encoding: "utf8",
}).trim();
const currentWorkingTreeDirty = execFileSync("git", ["status", "--porcelain"], {
  cwd: repoRoot, encoding: "utf8",
}).trim().length > 0;
const result = checkPerformanceBudgets(reports[0], reports[1], {
  currentRevision, currentWorkingTreeDirty,
});
process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
if (result.failures.length) process.exitCode = 1;
