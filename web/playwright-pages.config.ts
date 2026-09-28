import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./pages-tests",
  timeout: 30_000,
  use: { ...devices["Desktop Chrome"],
    baseURL: "http://127.0.0.1:5175/WhatAnimeShouldIWatch/" },
  webServer: {
    command: "npm run build:web && npm run preview --workspace web -- --host 127.0.0.1 --port 5175 --strictPort",
    cwd: "..",
    env: { PAGES_BASE_PATH: "/WhatAnimeShouldIWatch/" },
    url: "http://127.0.0.1:5175/WhatAnimeShouldIWatch/",
    reuseExistingServer: !process.env.CI,
    timeout: 60_000,
  },
});
