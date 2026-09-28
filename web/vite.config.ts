import { defineConfig } from "vite";

const pagesBasePath = process.env.PAGES_BASE_PATH;
if (pagesBasePath && !/^\/[A-Za-z0-9._-]+\/$/.test(pagesBasePath)) {
  throw new Error("PAGES_BASE_PATH must be a single absolute project path ending in /");
}

export default defineConfig({
  base: pagesBasePath ?? "./",
});
