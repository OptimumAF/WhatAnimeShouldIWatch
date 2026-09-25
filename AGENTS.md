# Repository workflow

- Read `docs/DEVELOPMENT_PLAN.md` and the latest `docs/PROGRESS.md` handoff before changing code. Inspect the actual branch, commit, local changes, and applicable instructions; the reviewed SHA in the plan is a reference only.
- Work in small, reviewable slices. Verify the relevant behavior, inspect the diff, and update the plan and progress record in the same change. Check a task only after its stated acceptance criteria pass. Preserve task IDs and log material scope or dependency changes.
- Run routine development with `fixtures/synthetic-input.json`, `npm run dev:demo`, and mocked providers. Never use real usernames, viewing histories, private ratings, salts, or credentials in fixtures, logs, screenshots, tests, or commits.
- Do not fetch or regenerate production datasets, expand collection, publish data or models, route imports through a new third party, deploy, or add paid infrastructure without the relevant authorization. Keep the existing TypeScript, Python, and Rust stacks unless evidence and an approved product decision justify a change.
- The compact graph's optional fourth anime-pair tuple value is pair support. Existing three-value readers remain valid; graph semantics and format versioning are still open tasks in M3.
