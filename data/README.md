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

To publish/update release assets from local files, use:

```bash
npm run data:publish:release
```

The GitHub workflow `.github/workflows/publish-data-release.yml` is still available,
but it is meant for:

1. Republish from checked-in repo data (`source_mode=repo`)
2. Publish from a prior workflow artifact (`source_mode=artifact`)
3. Refresh metadata from the current release payload (`source_mode=release`)
