/** Offline local-store recovery; caller must separately review any later deployment. */
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Command } from "commander";
import { restorePreviousRelease } from "./install-release-bundle.js";

function main(): void {
  const command = new Command();
  command.requiredOption("--store <path>")
    .requiredOption("--current-tag <tag>")
    .requiredOption("--current-bundle-id <digest>")
    .requiredOption("--current-manifest-sha256 <digest>")
    .requiredOption("--previous-tag <tag>")
    .requiredOption("--previous-bundle-id <digest>")
    .requiredOption("--previous-manifest-sha256 <digest>");
  command.parse(process.argv);
  const flags = command.opts();
  const result = restorePreviousRelease({
    storeDir: flags.store,
    expectedCurrent: { tag: flags.currentTag, bundleId: flags.currentBundleId,
      manifestSha256: flags.currentManifestSha256 },
    expectedPrevious: { tag: flags.previousTag, bundleId: flags.previousBundleId,
      manifestSha256: flags.previousManifestSha256 },
  });
  process.stdout.write(`Restored verified local release ${result.manifest.tag} ${result.manifest.bundleId}.\n`);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try { main(); } catch (error) {
    process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
    process.exitCode = 1;
  }
}
