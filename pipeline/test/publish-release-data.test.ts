import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");

test("legacy publisher refuses missing or user-linked inputs before creating a release workdir", () => {
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), "invented-data-release-"));
  try {
    const source = path.join(temporary, "source");
    const work = path.join(temporary, "work");
    fs.mkdirSync(source);
    for (const populated of [false, true]) {
      if (populated) {
        fs.writeFileSync(path.join(source, "graph.compact.json"), "{}\n");
        fs.writeFileSync(path.join(source, "anonymized-ratings.compact.json"), "{}\n");
        fs.writeFileSync(path.join(source, "model-mf-web.compact.json"), "{}\n");
      }
      const result = spawnSync(process.execPath, [
        "--import", "tsx", path.join(root, "pipeline/src/publish-release-data.ts"),
        "--source-dir", source, "--work-dir", work,
      ], { cwd: root, encoding: "utf8", env: { ...process.env,
        GH_TOKEN: "invented-token", DATA_RELEASE_TAG: "data-vinvented" } });
      assert.notEqual(result.status, 0);
      assert.match(result.stderr, /Legacy data release publication is disabled by decision 0028/);
      assert.equal(fs.existsSync(work), false);
    }
  } finally {
    const resolved = fs.realpathSync(temporary);
    if (!resolved.startsWith(path.resolve(os.tmpdir()) + path.sep)) {
      throw new Error("Refusing cleanup outside the temporary directory");
    }
    fs.rmSync(resolved, { recursive: true, force: true });
  }
});
