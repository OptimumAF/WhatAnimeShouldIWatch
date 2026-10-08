/** Private Windows OS ports only; no acquisition, approval registry or CLI. */
import { execFile } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { FOLLOWUP_SCOPE_SHA256, type FollowupGatePorts, type FollowupReservation } from "./wikidata-followup-gates.js";
import type { FollowupOutputPort, FollowupOutputPayload } from "./wikidata-followup-output.js";

const script = fileURLToPath(new URL("./windows-followup-private.ps1", import.meta.url));
const stages = new Set(["initialization", "path-syntax", "path-normalization", "path-component", "derived-path", "ancestor-enumeration",
  "directory-pin", "directory-attributes", "directory-resolution", "private-access",
  "reservation-preflight", "reservation-record", "reservation-security", "reservation-create", "reservation-access", "reservation-flush",
  "output-preflight", "output-marker", "output-payload", "output-create", "output-access", "output-files", "output-complete"]);
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
function windowsRunner(repoRoot: string) {
  if (process.platform !== "win32") fail();
  const executable = path.win32.join(process.env.SystemRoot ?? "C:\\Windows", "System32", "WindowsPowerShell", "v1.0", "powershell.exe");
  return (action: "Inspect" | "Reserve" | "Save", record?: FollowupReservation, payload?: FollowupOutputPayload): Promise<unknown> => new Promise((resolve, reject) => {
    let input: string, recordBase64: string | undefined;
    try {
      input = payload ? JSON.stringify(payload) : "";
      const recordBytes = record ? Buffer.from(JSON.stringify(record)) : undefined;
      if (Buffer.byteLength(input) > 12 * 1024 * 1024 || recordBytes && recordBytes.length > 4096) { reject(failure()); return; }
      recordBase64 = recordBytes?.toString("base64");
    } catch { reject(failure()); return; }
    const args = ["-NoLogo", "-NoProfile", "-NonInteractive", "-File", script, "-Action", action,
      "-RepositoryRoot", repoRoot, "-ScopeSha256", FOLLOWUP_SCOPE_SHA256];
    if (recordBase64) args.push("-RecordBase64", recordBase64);
    const env = { ...process.env, PSModulePath: path.win32.join(process.env.SystemRoot ?? "C:\\Windows", "System32", "WindowsPowerShell", "v1.0", "Modules") };
    const child = execFile(executable, args, { env, windowsHide: true, timeout: 30000, maxBuffer: 16384 }, (error, stdout) => {
      if (error) { reject(failure(stdout)); return; }
      try { resolve(JSON.parse(stdout)); } catch { reject(failure()); }
    });
    child.stdin?.on("error", () => reject(failure()));
    child.stdin?.end(input);
  });
}
export function createWindowsFollowupPorts(repoRoot: string, now: () => number = Date.now): FollowupGatePorts {
  const run = windowsRunner(repoRoot);
  return { now, inspect: () => run("Inspect"), reserveAtomic: async (record) => {
    const result = await run("Reserve", record);
    if (!result || typeof result !== "object" || Array.isArray(result) || Object.keys(result).length !== 1 ||
        !("reserved" in result) || typeof result.reserved !== "boolean") fail();
    return result.reserved;
  } };
}
/** Caller must first recompute the private payload with saveFollowupOutput; no real invocation is approved by this factory. */
export function createWindowsFollowupOutputPort(repoRoot: string): FollowupOutputPort {
  const run = windowsRunner(repoRoot);
  return { save: async (payload) => {
    let record: FollowupReservation;
    try { record = payload.reservation; } catch { fail(); }
    return run("Save", record, payload);
  } };
}
