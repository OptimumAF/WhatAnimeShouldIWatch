import { defineConfig } from "vite";
import { readFileSync } from "node:fs";

const rootPackage = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8")) as { version: string };
const sourceRevision = process.env.GITHUB_SHA;

const pagesBasePath = process.env.PAGES_BASE_PATH;
if (pagesBasePath && !/^\/[A-Za-z0-9._-]+\/$/.test(pagesBasePath)) {
  throw new Error("PAGES_BASE_PATH must be a single absolute project path ending in /");
}

export default defineConfig({
  base: pagesBasePath ?? "./",
  define: {
    __WASIW_APP_VERSION__: JSON.stringify(rootPackage.version),
    __WASIW_SOURCE_REVISION__: JSON.stringify(sourceRevision && /^[a-f0-9]{40}$/.test(sourceRevision)
      ? sourceRevision : "local"),
  },
});
