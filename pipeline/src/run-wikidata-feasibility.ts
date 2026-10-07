/** Explicit local pilot only. No routine npm command, provider workflow, or release integration. */
import { existsSync, mkdirSync, readFileSync, realpathSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { acquireWikidataPilot, PILOT_IDS, verifyPilotApproval } from "./wikidata-feasibility.js";
import { mapWikibaseMetadata, type WikibaseMappingPolicy } from "./wikibase-metadata.js";

const repoRoot = realpathSync(fileURLToPath(new URL("../../", import.meta.url)));
const args = process.argv.slice(2);
if (args.length !== 3 || args[0] !== "--approved-study" || args[1] !== "wikidata-pilot-2026-10-07" || !path.isAbsolute(args[2])) {
  throw new Error("Usage: --approved-study wikidata-pilot-2026-10-07 ABSOLUTE_PRIVATE_DIRECTORY");
}
const approval = JSON.parse(readFileSync(path.join(repoRoot, "docs/approvals/wikidata-feasibility.json"), "utf8"));
verifyPilotApproval(approval, Date.now());
const out = path.resolve(args[2]), parent = realpathSync(path.dirname(out));
const resolved = path.join(parent, path.basename(out));
const approvedOutput = path.resolve(repoRoot, "..", "Anime-private", "wikidata-pilot-2026-10-07");
const insideRepo = (candidate: string) => candidate === repoRoot || (!path.relative(repoRoot, candidate).startsWith("..") && !path.isAbsolute(path.relative(repoRoot, candidate)));
if (out !== approvedOutput || insideRepo(resolved) || path.basename(out) !== "wikidata-pilot-2026-10-07" || existsSync(out)) {
  throw new Error("Pilot requires a new named private directory outside the checkout.");
}
const expiresAt = new Date(Math.min(Date.now() + 7 * 86400000, Date.parse(approval.expiresAt))).toISOString();
mkdirSync(out, { mode: 0o700 });
const write = (name: string, value: unknown) => writeFileSync(path.join(out, name), `${JSON.stringify(value, null, 2)}\n`, { flag: "wx", mode: 0o600 });
write("study.json", { studyId: approval.studyId, startedAt: new Date().toISOString(), expiresAt,
  selection: approval.selection, publicArtifacts: false, oneUse: true });
try {
  const acquired = await acquireWikidataPilot(approval, { fetch, now: Date.now,
    sleep: async (milliseconds) => { if (milliseconds) await new Promise((resolve) => setTimeout(resolve, milliseconds)); } });
  const policy: WikibaseMappingPolicy = { format: "wikibase-metadata-policy-v1", languageOrder: ["en", "ja"],
    genreLabels: {}, mediaFormats: {}, durationUnits: {}, classification: null };
  const mapped = mapWikibaseMetadata(acquired.sourceBytes, PILOT_IDS, policy, acquired.source);
  writeFileSync(path.join(out, "source-projection.json"), acquired.sourceBytes, { flag: "wx", mode: 0o600 });
  writeFileSync(path.join(out, "definition-labels.json"), acquired.definitionBytes, { flag: "wx", mode: 0o600 });
  write("receipt.json", acquired.receipt); write("policy.json", policy);
  write("audit.json", mapped.report); if (mapped.snapshot) write("candidate.json", mapped.snapshot);
  write("completed.json", { completedAt: new Date().toISOString(), expiresAt });
  console.log(JSON.stringify({ status: "local-pilot-complete", entities: acquired.receipt.animeEntities,
    definitions: acquired.receipt.definitionEntities, omittedDefinitions: acquired.receipt.omittedDefinitionEntities,
    attempts: acquired.receipt.attempts, statementPresence: acquired.receipt.statementPresence,
    coverage: mapped.report.coverage }));
} catch (error) {
  // Retain only a fixed local failure description, never provider payloads or exception objects.
  const message = error instanceof Error && /^Wikidata pilot [a-zA-Z. ]+: [a-zA-Z0-9 -]+\.$/.test(error.message)
    ? error.message : "Wikidata pilot failed validation or transport.";
  write("failed.json", { code: "pilot-incomplete", message, expiresAt });
  console.error(message); process.exitCode = 1;
}
