/** Private Windows OS ports only; no acquisition, approval registry, CLI or output-directory creation. */
import { execFile } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { FOLLOWUP_SCOPE_SHA256, type FollowupGatePorts, type FollowupReservation } from "./wikidata-followup-gates.js";

const script = fileURLToPath(new URL("./windows-followup-private.ps1", import.meta.url));
const stages = new Set(["initialization", "path-syntax", "path-normalization", "path-component", "derived-path", "ancestor-enumeration",
  "directory-pin", "directory-attributes", "directory-resolution", "private-access",
  "reservation-preflight", "reservation-record", "reservation-create", "reservation-access", "reservation-flush"]);
const failure = (stdout?: string) => {
  let stage = "";
  try {
    const value = JSON.parse(stdout ?? "");
    if (value && typeof value === "object" && !Array.isArray(value) && Object.keys(value).length === 2 &&
        value.code === "windows-private-preflight-failed" && typeof value.stage === "string" && stages.has(value.stage)) stage = ` (${value.stage})`;
  } catch { /* Arbitrary helper output remains redacted. */ }
  return new Error(`Wikidata follow-up preflight: Windows private inspection or reservation failed${stage}.`);
};
function fail(): never { throw failure(); }
export function createWindowsFollowupPorts(repoRoot: string, now: () => number = Date.now): FollowupGatePorts {
  if (process.platform !== "win32") fail();
  const executable = path.win32.join(process.env.SystemRoot ?? "C:\\Windows", "System32", "WindowsPowerShell", "v1.0", "powershell.exe");
  const run = (action: "Inspect" | "Reserve", record?: FollowupReservation): Promise<unknown> => new Promise((resolve, reject) => {
    const args = ["-NoLogo", "-NoProfile", "-NonInteractive", "-File", script, "-Action", action,
      "-RepositoryRoot", repoRoot, "-ScopeSha256", FOLLOWUP_SCOPE_SHA256];
    if (record) args.push("-RecordBase64", Buffer.from(JSON.stringify(record)).toString("base64"));
    const env = { ...process.env, PSModulePath: path.win32.join(process.env.SystemRoot ?? "C:\\Windows", "System32", "WindowsPowerShell", "v1.0", "Modules") };
    execFile(executable, args, { env, windowsHide: true, timeout: 30000, maxBuffer: 16384 }, (error, stdout) => {
      if (error) { reject(failure(stdout)); return; }
      try { resolve(JSON.parse(stdout)); } catch { reject(failure()); }
    });
  });
  return { now, inspect: () => run("Inspect"), reserveAtomic: async (record) => {
    const result = await run("Reserve", record);
    if (!result || typeof result !== "object" || Array.isArray(result) || Object.keys(result).length !== 1 ||
        !("reserved" in result) || typeof result.reserved !== "boolean") fail();
    return result.reserved;
  } };
}
