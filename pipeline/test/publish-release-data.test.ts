import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");

test("local release publisher refuses an optional model before upload", () => {
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), "invented-data-release-"));
  try {
    const source = path.join(temporary, "source");
    const work = path.join(temporary, "work");
    fs.mkdirSync(source);
    fs.writeFileSync(path.join(source, "graph.compact.json"), "{}\n");
    fs.writeFileSync(path.join(source, "anonymized-ratings.compact.json"), "{}\n");
    fs.writeFileSync(path.join(source, "model-mf-web.compact.json"), "{}\n");
    const result = spawnSync(process.execPath, [
      "--import", "tsx", path.join(root, "pipeline/src/publish-release-data.ts"),
      "--source-dir", source, "--work-dir", work,
    ], { cwd: root, encoding: "utf8" });
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /cannot publish model-mf-web\.compact\.json/);
    assert.equal(fs.existsSync(work), false);
  } finally {
    fs.rmSync(temporary, { recursive: true, force: true });
  }
});
