/** Install one exact approved public release into a clean Pages build tree. */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Command } from "commander";
import { parseReleaseManifest, type ReleaseManifestV1 } from "../../web/src/artifacts.js";
import { installReleaseBundle, type ReleaseAssetTransport } from "./install-release-bundle.js";
import { getRepoRoot } from "./paths.js";
import { RELEASE_FILES, releaseSha256 } from "./release-manifest.js";
import { verifyModelPublicRelease } from "./verify-model-release-public.js";
import { verifyPublicationPackage } from "./verify-publication-package.js";

const repoRoot = getRepoRoot(import.meta.url);
const TAG = /^data-v[A-Za-z0-9][A-Za-z0-9._-]*$/;
const DIGEST = /^[a-f0-9]{64}$/;
const ARTIFACT = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;
const HTTPS = /^https:\/\/[^\s/]+\/\S+$/;

export interface PreparePagesOptions {
  kind: "data" | "model";
  packageDir: string;
  previousDir?: string;
  previousTag?: string;
  storeDir: string;
  tag: string;
  manifestSha256: string;
  sourceRunId: number;
  artifactName: string;
  publicationApprovalRef: string;
  deploymentApprovalRef: string;
  ownerApprovalRef?: string;
  providerApprovals: unknown;
  publicationApprovals: unknown;
  modelApprovals: unknown;
  planApprovals: unknown;
  approvalRepoDir?: string;
}

type RecordValue = Record<string, unknown>;

function fail(field: string, reason: string): never {
  throw new Error(`Pages bundle ${field}: ${reason}`);
}

function record(value: unknown, field: string): RecordValue {
  if (!value || typeof value !== "object" || Array.isArray(value)) fail(field, "must be an object");
  return value as RecordValue;
}

function match(value: unknown, pattern: RegExp, field: string): string {
  if (typeof value !== "string" || !pattern.test(value)) fail(field, "is invalid");
  return value;
}

function jsonFile(filename: string): unknown {
  return JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(fs.readFileSync(filename)));
}

function checkDeploymentApproval(options: PreparePagesOptions, source: string,
  owner: string): void {
  const root = record(options.providerApprovals, "providerApprovals");
  if (root.schemaVersion !== 1) fail("providerApprovals.schemaVersion", "is unsupported");
  const approvals = record(root.approvals, "providerApprovals.approvals");
  const deployment = record(approvals.deployment, "providerApprovals.approvals.deployment");
  if (deployment.approved !== true || !Array.isArray(deployment.sources) ||
      !deployment.sources.includes(source) || deployment.owner !== owner ||
      deployment.approvalRef !== options.deploymentApprovalRef) {
    fail("providerApprovals.approvals.deployment", "does not approve this source, owner, and reference");
  }
}

function localTransport(directory: string, manifest: ReleaseManifestV1): ReleaseAssetTransport {
  const names: string[] = [RELEASE_FILES.manifest, RELEASE_FILES.neighborhood, RELEASE_FILES.explorer,
    RELEASE_FILES.catalog, ...(manifest.model ? [RELEASE_FILES.model] : [])];
  return { tag: manifest.tag, assets: names,
    async fetchAsset(name: string): Promise<Response> {
      if (!names.includes(name)) fail("asset", `unsupported ${name}`);
      return new Response(new Uint8Array(fs.readFileSync(path.join(directory, name))),
        { status: 200, headers: { "Content-Type": "application/json" } });
    } };
}

/** Approval verifiers run before the installer receives reviewed-genesis or prior inputs. */
export async function preparePagesBundle(options: PreparePagesOptions): Promise<ReleaseManifestV1> {
  if (options.kind !== "data" && options.kind !== "model") fail("kind", "is unsupported");
  match(options.tag, TAG, "tag");
  match(options.manifestSha256, DIGEST, "manifestSha256");
  match(options.artifactName, ARTIFACT, "artifactName");
  match(options.publicationApprovalRef, HTTPS, "publicationApprovalRef");
  match(options.deploymentApprovalRef, HTTPS, "deploymentApprovalRef");
  if (!Number.isSafeInteger(options.sourceRunId) || options.sourceRunId < 1) {
    fail("sourceRunId", "must be a positive safe integer");
  }
  if (options.previousTag !== undefined) match(options.previousTag, TAG, "previousTag");
  if ((options.previousTag === undefined) !== (options.previousDir === undefined) ||
      options.previousTag === options.tag ||
      (options.kind === "model" && !options.previousTag)) {
    fail("previousTag", "must name exactly one distinct data-only predecessor when present");
  }
  const directory = path.resolve(options.packageDir);
  const previous = options.previousDir ? path.resolve(options.previousDir) : undefined;
  const store = path.resolve(options.storeDir);
  if (store === directory || store === previous || directory.startsWith(`${store}${path.sep}`) ||
      (previous && previous.startsWith(`${store}${path.sep}`))) {
    fail("storeDir", "must be separate from the downloaded packages");
  }
  if (fs.existsSync(store)) {
    const entries = fs.readdirSync(store);
    if (entries.some((entry) => entry !== ".gitkeep")) {
      fail("storeDir", "must be clean before a Pages build");
    }
  }
  const manifestBytes = fs.readFileSync(path.join(directory, RELEASE_FILES.manifest));
  if (releaseSha256(manifestBytes) !== options.manifestSha256) {
    fail("manifestSha256", "differs from the exact dispatched release manifest bytes");
  }
  const manifest = parseReleaseManifest(JSON.parse(new TextDecoder("utf-8", { fatal: true })
    .decode(manifestBytes)), RELEASE_FILES.manifest);
  if (manifest.tag !== options.tag ||
      (manifest.lastKnownGood?.tag ?? undefined) !== options.previousTag) {
    fail("release-manifest.json", "tag or predecessor differs from dispatch");
  }
  let owner: string;
  if (options.kind === "data") {
    const audit = verifyPublicationPackage({ packageDir: directory, previousDir: previous,
      dispatch: { tag: options.tag, sourceRunId: options.sourceRunId,
        artifactName: options.artifactName, previousTag: options.previousTag ?? null,
        approvalRef: options.publicationApprovalRef },
      providerApprovals: options.providerApprovals, packageApprovals: options.publicationApprovals });
    owner = audit.redistribution.owner;
  } else {
    match(options.ownerApprovalRef, HTTPS, "ownerApprovalRef");
    const audit = verifyModelPublicRelease({ packageDir: directory, baseDir: previous!,
      dispatch: { tag: options.tag, baseTag: options.previousTag!,
        sourceRunId: options.sourceRunId, artifactName: options.artifactName,
        ownerApprovalRef: options.ownerApprovalRef! },
      providerApprovals: options.providerApprovals,
      publicationApprovals: options.publicationApprovals,
      modelApprovals: options.modelApprovals, planApprovals: options.planApprovals,
      approvalRepoDir: options.approvalRepoDir });
    owner = audit.approvals.owner;
    if (audit.approvals.publicationApprovalRef !== options.publicationApprovalRef) {
      fail("publicationApprovalRef", "differs from the approved model package");
    }
  }
  checkDeploymentApproval(options, manifest.dataset.source, owner);
  const result = await installReleaseBundle({ storeDir: store, tag: options.tag,
    transport: localTransport(directory, manifest),
    ...(previous ? { verifiedPriorDir: previous } : {
      reviewedGenesis: { tag: manifest.tag, bundleId: manifest.bundleId,
        manifestSha256: options.manifestSha256 },
    }) });
  if (result.manifest.bundleId !== manifest.bundleId || !result.changed) {
    fail("install", "did not activate the exact reviewed bundle");
  }
  return result.manifest;
}

async function main(): Promise<void> {
  const command = new Command();
  command.requiredOption("--kind <data|model>").requiredOption("--package <path>")
    .requiredOption("--store <path>").requiredOption("--tag <tag>")
    .requiredOption("--manifest-sha256 <digest>").requiredOption("--run-id <number>")
    .requiredOption("--artifact-name <name>").requiredOption("--publication-approval-ref <url>")
    .requiredOption("--deployment-approval-ref <url>")
    .option("--previous <path>").option("--previous-tag <tag>")
    .option("--owner-approval-ref <url>");
  command.parse(process.argv);
  const flags = command.opts();
  const manifest = await preparePagesBundle({ kind: flags.kind,
    packageDir: flags.package, previousDir: flags.previous,
    previousTag: flags.previousTag, storeDir: flags.store, tag: flags.tag,
    manifestSha256: flags.manifestSha256, sourceRunId: Number(flags.runId),
    artifactName: flags.artifactName,
    publicationApprovalRef: flags.publicationApprovalRef,
    deploymentApprovalRef: flags.deploymentApprovalRef,
    ownerApprovalRef: flags.ownerApprovalRef,
    providerApprovals: jsonFile(path.join(repoRoot, "docs/approvals/provider-data.json")),
    publicationApprovals: jsonFile(path.join(repoRoot,
      "docs/approvals/publication-bundles.json")),
    modelApprovals: jsonFile(path.join(repoRoot,
      "docs/approvals/model-release-bundles.json")),
    planApprovals: jsonFile(path.join(repoRoot,
      "docs/approvals/model-evaluation-plans.json")) });
  process.stdout.write(`Prepared exact Pages bundle ${manifest.tag} ${manifest.bundleId}.\n`);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch((error: unknown) => {
    process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
    process.exitCode = 1;
  });
}
