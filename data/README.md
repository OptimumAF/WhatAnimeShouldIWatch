# Data Artifacts

This directory stores generated data artifacts locally:

- `graph.compact.json`
- `anonymized-ratings.compact.json`
- `model-mf-web.compact.json` (optional)

These files are intentionally not tracked in git. Use one of:

1. Generate locally via pipeline (`npm run collect`, `npm run build:graph`, `npm run ml:export:web`).
2. Download the latest published release assets:

```bash
npm run data:fetch:release
```

To publish/update release assets from local files, run the workflow:

- `.github/workflows/publish-data-release.yml`
