import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { test, type TestContext } from "node:test";
import { createWindowsFollowupPorts } from "../src/windows-followup-ports.js";
import { FOLLOWUP_STUDY_ID, FOLLOWUP_SCOPE_SHA256, reserveFollowupStudy } from "../src/wikidata-followup-gates.js";

const windows = process.platform === "win32";
const executable = path.join(process.env.SystemRoot ?? "C:\\Windows", "System32", "WindowsPowerShell", "v1.0", "powershell.exe");
const approval = () => ({ format: "wikidata-followup-approval-v1", studyId: FOLLOWUP_STUDY_ID,
  state: "approved", approved: true, owner: "Avery", authority: "Invented OS test approval only",
  decisionRef: "docs/decisions/0046-bounded-followup-study-gates.md", scopeSha256: FOLLOWUP_SCOPE_SHA256,
  approvedAt: new Date(Date.now() - 60000).toISOString(), expiresAt: new Date(Date.now() + 3600000).toISOString(),
  use: "one-local-feasibility-study-only", publicArtifacts: false, training: false, deployment: false, productCache: false });
function setup(t: TestContext, broad = false, unprotected = false) {
  // TEMP may use an 8.3 alias on a hosted Windows runner. The production port deliberately
  // rejects aliases, so prepare our invented workspace using the canonical path.
  const temporary = fs.realpathSync.native(os.tmpdir());
  t.diagnostic(`Invented fixture temporary-path alias: ${temporary.toLowerCase() !== os.tmpdir().toLowerCase()}`);
  const root = fs.realpathSync.native(fs.mkdtempSync(path.join(temporary, "invented-followup-ports-")));
  const repo = path.join(root, "Anime"), parent = path.join(root, "Anime-private");
  t.diagnostic(`Invented fixture path facts: ${JSON.stringify({ driveRoot: /^[A-Za-z]:\\/.test(repo),
    bounded: repo.length <= 240, forbiddenSyntax: /[:*?"<>|]/.test(repo.slice(2)), extendedPrefix: repo.startsWith("\\\\?\\") })}`);
  fs.mkdirSync(repo); fs.mkdirSync(parent);
  t.after(() => {
    const actual = fs.realpathSync.native(root), temporary = fs.realpathSync.native(os.tmpdir());
    if (!actual.startsWith(temporary + path.sep) || !path.basename(actual).startsWith("invented-followup-ports-")) throw new Error("Refuse cleanup outside invented temporary root");
    fs.rmSync(actual, { recursive: true, force: true });
  });
  // Fixed test-only PowerShell body; the literal path travels as data via an environment variable.
  const command = `$sid = [Security.Principal.WindowsIdentity]::GetCurrent().User
$acl = [Security.AccessControl.DirectorySecurity]::new()
$acl.SetOwner($sid)
$acl.SetAccessRuleProtection(($env:WASIW_FIXTURE_UNPROTECTED -ne '1'), $false)
$inherit = [Security.AccessControl.InheritanceFlags]::ContainerInherit -bor [Security.AccessControl.InheritanceFlags]::ObjectInherit
$acl.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new($sid, 'FullControl', $inherit, 'None', 'Allow'))
if ($env:WASIW_FIXTURE_BROAD_ACL -eq '1') {
  $acl.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new([Security.Principal.SecurityIdentifier]::new('S-1-1-0'), 'Read', $inherit, 'None', 'Allow'))
}
Set-Acl -LiteralPath $env:WASIW_FIXTURE_ACL_PATH -AclObject $acl`;
  execFileSync(executable, ["-NoLogo", "-NoProfile", "-NonInteractive", "-Command", command], {
    env: { ...process.env, PSModulePath: path.join(process.env.SystemRoot ?? "C:\\Windows", "System32", "WindowsPowerShell", "v1.0", "Modules"),
      WASIW_FIXTURE_ACL_PATH: parent, WASIW_FIXTURE_BROAD_ACL: broad ? "1" : "0", WASIW_FIXTURE_UNPROTECTED: unprotected ? "1" : "0" },
    windowsHide: true, stdio: "pipe", timeout: 30000,
  });
  return { root, repo, parent, ports: createWindowsFollowupPorts(repo),
    marker: path.join(parent, `${FOLLOWUP_STUDY_ID}.reserved.json`), target: path.join(parent, FOLLOWUP_STUDY_ID) };
}

test("non-Windows factory fails closed without probing or reserving", { skip: windows }, () => {
  assert.throws(() => createWindowsFollowupPorts("invented"), /Windows private/);
});
test("Windows OS facts and flushed exclusive reservation survive new process and output removal", { skip: !windows }, async (t) => {
  const context = setup(t), facts: any = await context.ports.inspect();
  const short = execFileSync(executable, ["-NoLogo", "-NoProfile", "-NonInteractive", "-Command",
    "(New-Object -ComObject Scripting.FileSystemObject).GetFolder($env:WASIW_FIXTURE_SHORT_PATH).ShortPath"], {
    env: { ...process.env, PSModulePath: path.join(process.env.SystemRoot ?? "C:\\Windows", "System32", "WindowsPowerShell", "v1.0", "Modules"),
      WASIW_FIXTURE_SHORT_PATH: context.root }, windowsHide: true, timeout: 30000, encoding: "utf8",
  }).trim();
  const shortAlias = short.toLowerCase() !== context.root.toLowerCase();
  t.diagnostic(`Invented fixture 8.3 alias available: ${shortAlias}`);
  if (shortAlias) {
    assert.ok(fs.realpathSync(short).toLowerCase() !== context.root.toLowerCase(), "JS resolver preserves the invented short alias");
    assert.ok(fs.realpathSync.native(short).toLowerCase() === context.root.toLowerCase(), "Native resolver expands the invented short alias");
    await assert.rejects(createWindowsFollowupPorts(path.join(short, "Anime")).inspect(),
      (error: any) => /Windows private.*\(path-normalization\)/.test(error.message) && !error.message.includes(short));
    assert.equal(fs.existsSync(context.marker), false);
  }
  assert.equal(facts.access, "verified-owner-only"); assert.equal(facts.priorPilotExists, false);
  const record = await reserveFollowupStudy(approval(), context.ports);
  assert.deepEqual(JSON.parse(fs.readFileSync(context.marker, "utf8")), record);
  assert.equal(fs.existsSync(context.target), false); // The port creates no study output directory.
  fs.mkdirSync(context.target); fs.rmdirSync(context.target); // Removing only invented empty output leaves consumption intact.
  const fresh = createWindowsFollowupPorts(context.repo);
  await assert.rejects(reserveFollowupStudy(approval(), fresh), /preflight/);
  assert.equal((await fresh.inspect() as any).reservationExists, true);
});
test("Windows overly broad ACL, prior pilot or target refuse without a marker", { skip: !windows }, async (t) => {
  const broad = setup(t, true);
  assert.equal((await broad.ports.inspect() as any).access, "unknown");
  await assert.rejects(reserveFollowupStudy(approval(), broad.ports), /preflight/);
  assert.equal(fs.existsSync(broad.marker), false);
  const unprotected = setup(t, false, true);
  assert.equal((await unprotected.ports.inspect() as any).access, "unknown");
  await assert.rejects(reserveFollowupStudy(approval(), unprotected.ports), /preflight/);
  assert.equal(fs.existsSync(unprotected.marker), false);
  for (const name of ["wikidata-pilot-2026-10-07", FOLLOWUP_STUDY_ID]) {
    const context = setup(t); fs.mkdirSync(path.join(context.parent, name));
    await assert.rejects(reserveFollowupStudy(approval(), context.ports), /preflight/);
    assert.equal(fs.existsSync(context.marker), false);
  }
});
test("Windows atomic reservation permits one concurrent winner and preserves corrupt markers", { skip: !windows }, async (t) => {
  const context = setup(t), results = await Promise.allSettled([
    reserveFollowupStudy(approval(), createWindowsFollowupPorts(context.repo)),
    reserveFollowupStudy(approval(), createWindowsFollowupPorts(context.repo)),
  ]);
  assert.equal(results.filter((result) => result.status === "fulfilled").length, 1,
    results.flatMap((result) => result.status === "rejected" ? [String(result.reason?.message)] : []).join("; "));
  const before = fs.readFileSync(context.marker);
  await assert.rejects(reserveFollowupStudy(approval(), context.ports), /preflight/);
  assert.deepEqual(fs.readFileSync(context.marker), before);
  const corrupt = setup(t); fs.writeFileSync(corrupt.marker, "invented interrupted write");
  await assert.rejects(reserveFollowupStudy(approval(), corrupt.ports), /preflight/);
  assert.equal(fs.readFileSync(corrupt.marker, "utf8"), "invented interrupted write");
});
test("Windows fresh reservation probe detects intervening paths and junction ancestors", { skip: !windows }, async (t) => {
  const context = setup(t), inspect = context.ports.inspect;
  await assert.rejects(createWindowsFollowupPorts(context.repo + "\\.").inspect(),
    (error: any) => /Windows private.*\(path-normalization\)/.test(error.message) && !error.message.includes(context.root));
  assert.equal(fs.existsSync(context.marker), false);
  context.ports.inspect = async () => { const facts = await inspect(); fs.mkdirSync(context.target); return facts; };
  await assert.rejects(reserveFollowupStudy(approval(), context.ports), /preflight/);
  assert.equal(fs.existsSync(context.marker), false);
  const linked = setup(t), moved = path.join(linked.root, "invented-private-target");
  fs.renameSync(linked.parent, moved); fs.symlinkSync(moved, linked.parent, "junction");
  await assert.rejects(linked.ports.inspect(), (error: any) => /Windows private/.test(error.message) && !error.message.includes(moved));
  assert.equal(fs.existsSync(path.join(moved, `${FOLLOWUP_STUDY_ID}.reserved.json`)), false);
  const switched = setup(t), original = switched.ports.inspect, relocated = path.join(switched.root, "invented-switched-parent");
  switched.ports.inspect = async () => { const facts = await original();
    fs.renameSync(switched.parent, relocated); fs.symlinkSync(relocated, switched.parent, "junction"); return facts; };
  await assert.rejects(reserveFollowupStudy(approval(), switched.ports), /preflight/);
  assert.equal(fs.existsSync(path.join(relocated, `${FOLLOWUP_STUDY_ID}.reserved.json`)), false);
});
