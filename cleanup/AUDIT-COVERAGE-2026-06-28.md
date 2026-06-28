# Repository cleanup audit coverage — 2026-06-28

## Scope

- Git-tracked files inspected: 652
- Code and configuration files classified: 518
- Python: 320 files
- TypeScript: 55 files
- TSX: 88 files
- JavaScript/MJS: 2 files
- JSON: 40 files
- SQL: 3 files
- YAML: 4 files
- Shell: 6 files

Tracked assets under `backups/`, `data/`, and `system_architecture/` were inventoried but not rewritten. They are data, recovery material, or design inputs rather than safe cleanup targets.

## Method

Every tracked code/configuration path was included in extension and root inventories. The audit then searched production paths for debug output, temporary/fallback behavior, broad exception handling, hard-coded local endpoints, TODO/FIXME markers, oversized modules, duplicated dependency declarations, and apparently unreferenced files.

Convention-loaded files were not treated as dead solely because no static import was found. This includes Next.js `page.tsx`/`route.ts`, FastAPI routers, migrations, Modal/RunPod workers, and executable scripts.

## Applied in pass 2

- Removed browser logging of token prefixes, token length, and token validity.
- Removed post content, hashtag, file, team, user, and API response logging from the post creation flow.
- Removed post-detail object logging.
- Removed image-generation proxy request-body logging.
- Removed render-time `console.log` expressions from the chat avatar tree.
- Removed a gallery URL debug log.
- Removed raw tone-generation response output and an ad-hoc fine-tuning debug message.

These changes delete observation-only side effects. Request payloads, return values, state transitions, and error handling remain unchanged.

## Deferred deliberately

- Authentication/authorization bypasses require product-policy decisions and endpoint regression coverage.
- Sync-session fallback from an async dependency may change transaction and concurrency behavior.
- Fake API responses and fallback content require explicit availability/error contracts.
- Broad `except Exception` cleanup needs service-specific failure tests before narrowing.
- Large page and endpoint decomposition affects ownership boundaries and should be performed incrementally.
- Dependency consolidation requires environment lockfiles and deployment verification.

See [GitHub Issues](https://github.com/GTU9/AIMEX/issues) for bounded follow-up work.

## Verification baseline

- `python -m unittest tests.test_data_mapping -v`: 4 passed
- `python -m compileall -q app`: passed
- `npx.cmd tsc --noEmit --pretty false --incremental false`: passed

The repository does not currently provide a non-interactive configured frontend lint command; `next lint` enters setup. Full backend pytest execution is also unavailable in the current system interpreter because the project runtime dependencies are not installed there.
