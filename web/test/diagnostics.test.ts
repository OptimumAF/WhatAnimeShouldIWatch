import assert from "node:assert/strict";
import test from "node:test";
import type { ReleaseManifestV1 } from "../src/artifacts";
import { appVersionLabel, dataVersionLabel, diagnosticIssue, modelVersionLabel } from "../src/diagnostics";

const digest = "a".repeat(64);
const manifest = {
  tag: "data-vinvented-diagnostics", bundleId: digest,
  model: { format: "model-mf-compact-v1", sha256: "b".repeat(64) },
} as ReleaseManifestV1;

test("local identities distinguish app source, release data, and optional model", () => {
  assert.equal(appVersionLabel("0.1.0", "c".repeat(40)), "0.1.0 · source cccccccccccc");
  assert.equal(dataVersionLabel(manifest, "graph-compact-v3", false),
    "data-vinvented-diagnostics · graph-compact-v3 · bundle aaaaaaaaaaaa");
  assert.equal(modelVersionLabel(manifest, null, "unchecked", false),
    "model-mf-compact-v1 · sha256 bbbbbbbbbbbb · declared; not loaded");
  assert.match(modelVersionLabel(manifest, "model-mf-compact-v1", "loaded", false), /loaded$/);
  assert.equal(modelVersionLabel({ ...manifest, model: null }, null, "absent", false),
    "Not included in this data release");
  assert.equal(dataVersionLabel(null, "graph-compact-v2", false),
    "Legacy unversioned data · graph-compact-v2");
  assert.equal(modelVersionLabel(null, "model-mf-compact-v1", "loaded", true),
    "model-mf-compact-v1 · synthetic fixture");
});

test("diagnostics use fixed codes and reject untrusted version strings", () => {
  const privateText = "private-user raw-history=1,2,3";
  const output = [appVersionLabel(privateText, privateText),
    dataVersionLabel({ ...manifest, tag: privateText, bundleId: privateText }, privateText, false),
    modelVersionLabel({ ...manifest, model: { ...manifest.model!, format: privateText,
      sha256: privateText } }, null, "unchecked", false),
    JSON.stringify(diagnosticIssue("IMPORT-001"))].join(" ");
  assert.doesNotMatch(output, /private-user|raw-history/);
  assert.match(output, /IMPORT-001/);
  assert.match(output, /local file or text import/);
});
