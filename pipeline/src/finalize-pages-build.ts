/** Offline gate for a Pages build whose active bundle was already approved and installed. */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Command } from "commander";
import { parseActiveReleaseBundle, parseReleaseManifest } from "../../web/src/artifacts.js";
import { RELEASE_FILES, releaseSha256, verifyReleaseBundle } from "./release-manifest.js";

export interface PagesBuildOptions {
  distDir: string;
  tag: string;
  manifestSha256: string;
  basePath: string;
}

function fail(field: string, reason: string): never {
  throw new Error(`Pages build ${field}: ${reason}`);
}

function exact(directory: string, names: readonly string[], field: string): void {
  if (!fs.existsSync(directory) || !fs.lstatSync(directory).isDirectory() ||
      fs.lstatSync(directory).isSymbolicLink()) fail(field, "must be a real directory");
  const entries = fs.readdirSync(directory).sort();
  if (JSON.stringify(entries) !== JSON.stringify([...names].sort())) {
    fail(field, "contains missing or unsupported files");
  }
  for (const name of names) {
    if (fs.lstatSync(path.join(directory, name)).isSymbolicLink()) {
      fail(`${field}.${name}`, "must not be a symbolic link");
    }
  }
}

/** Copy index to 404 for direct Pages navigation, then verify exact pinned public data. */
export function finalizePagesBuild(options: PagesBuildOptions): string {
  if (!/^data-v[A-Za-z0-9][A-Za-z0-9._-]*$/.test(options.tag) ||
      !/^[a-f0-9]{64}$/.test(options.manifestSha256) ||
      !/^\/[A-Za-z0-9._-]+\/$/.test(options.basePath)) {
    fail("inputs", "requires a versioned tag, lowercase manifest digest, and project base path");
  }
  const dist = path.resolve(options.distDir);
  const indexPath = path.join(dist, "index.html");
  const index = fs.readFileSync(indexPath);
  const html = index.toString("utf8");
  if (index.length < 1 || !html.includes(`${options.basePath}assets/`) ||
      !html.includes(`href="${options.basePath}favicon.svg"`) ||
      !html.includes(`href="${options.basePath}manifest.webmanifest"`) ||
      !html.includes(`href="${options.basePath}icons/apple-touch-icon.png"`) ||
      html.includes('src="./assets/')) {
    fail("index.html", "does not use the expected absolute Pages asset base");
  }
  const notFoundPath = path.join(dist, "404.html");
  if (fs.existsSync(notFoundPath) && !fs.readFileSync(notFoundPath).equals(index)) {
    fail("404.html", "differs from the direct-navigation app entry");
  }
  const data = path.join(dist, "data");
  const rootNames = fs.readdirSync(data).filter((name) => name !== ".gitkeep");
  if (JSON.stringify(rootNames.sort()) !== JSON.stringify(["active.json", "bundles"])) {
    fail("data", "contains legacy or unsupported public files");
  }
  const pointer = parseActiveReleaseBundle(JSON.parse(fs.readFileSync(path.join(data,
    "active.json"), "utf8")), "active.json");
  const current = path.join(data, "bundles", pointer.bundleId);
  const manifestBytes = fs.readFileSync(path.join(current, RELEASE_FILES.manifest));
  const manifest = parseReleaseManifest(JSON.parse(manifestBytes.toString("utf8")),
    RELEASE_FILES.manifest);
  if (pointer.tag !== options.tag || pointer.manifestSha256 !== options.manifestSha256 ||
      releaseSha256(manifestBytes) !== options.manifestSha256 ||
      manifest.tag !== pointer.tag || manifest.bundleId !== pointer.bundleId) {
    fail("active.json", "differs from the dispatched immutable release");
  }
  const prior = manifest.lastKnownGood
    ? path.join(data, "bundles", manifest.lastKnownGood.bundleId) : undefined;
  const expectedBundles = [manifest.bundleId,
    ...(manifest.lastKnownGood ? [manifest.lastKnownGood.bundleId] : [])];
  exact(path.join(data, "bundles"), expectedBundles, "data.bundles");
  const bundleFiles = (model: boolean) => [RELEASE_FILES.manifest,
    RELEASE_FILES.neighborhood, RELEASE_FILES.explorer, RELEASE_FILES.catalog,
    ...(model ? [RELEASE_FILES.model] : [])];
  exact(current, bundleFiles(manifest.model !== null), "data.current");
  if (prior) {
    const priorManifest = parseReleaseManifest(JSON.parse(fs.readFileSync(path.join(prior,
      RELEASE_FILES.manifest), "utf8")), RELEASE_FILES.manifest);
    exact(prior, bundleFiles(priorManifest.model !== null), "data.prior");
  }
  verifyReleaseBundle(current, prior, false, !prior);
  if (!fs.existsSync(notFoundPath)) fs.writeFileSync(notFoundPath, index, { flag: "wx" });
  return manifest.bundleId;
}

function main(): void {
  const command = new Command();
  command.requiredOption("--dist <path>").requiredOption("--tag <tag>")
    .requiredOption("--manifest-sha256 <digest>").requiredOption("--base-path <path>");
  command.parse(process.argv);
  const flags = command.opts();
  const bundleId = finalizePagesBuild({ distDir: flags.dist, tag: flags.tag,
    manifestSha256: flags.manifestSha256, basePath: flags.basePath });
  process.stdout.write(`Verified Pages build for ${flags.tag} ${bundleId}.\n`);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try { main(); } catch (error) {
    process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
    process.exitCode = 1;
  }
}
