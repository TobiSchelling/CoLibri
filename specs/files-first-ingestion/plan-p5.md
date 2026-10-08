# Implementation Plan: P5 (fetchers)

> Spec: `requirements.md` REQ-025..REQ-030.
> Branch: `feat/ffi-p5-fetchers`. Written alongside the implementation; records the decisions taken.

## Decisions

- **A fetcher is part of a mirror.** A mirror with `fetch:` runs the fetcher first, then the normal reconcile. There is no separate command; `colibri update <name>` covers it. `--dry-run` skips fetchers (status `skipped`, problem `fetch_skipped`), because a fetch writes files.
- **Folder:** `path:` is optional for fetched mirrors and defaults to `<mirrors_dir>/<name>` (`mirrors_dir` defaults to `~/Documents/CoLibri/mirrors`, visible to DEVONthink).
- **Ownership:** a fetcher creates a new folder, adopts an empty one (ignoring `.DS_Store`), or uses one with the `.colibri-mirror` marker. Any other folder is refused before anything is written. Zephyr deletes only files listed in its own `.fetch-state.json`, so files placed into the folder by hand stay.
- **Writes:** atomic (temp file + rename) and skipped when the bytes are identical, so DEVONthink sees stable modification times.
- **Completeness:** `FetchReport.complete` is true only after a full listing (Zephyr) or exit code 0 (command). An incomplete fetch blocks pruning in the reconcile of the same run ("the fetch was incomplete; deletions wait for a complete fetch"). A fetch that fails outright makes the mirror status `error` and `update` exit non-zero; partial problems make it `partial`.
- **Pagination** (`api.rs::next_page`, pure): `isLast: true` ends; a short page without `isLast` ends; otherwise follow `next` unless it equals the current URL, else continue by `startAt` offset. The final page is complete only if the collected items reach `total`. Folders, statuses and priorities must list completely (`paginate_all`), otherwise the fetch fails.
- **Incremental Zephyr fetch:** fingerprint = SHA-256 of the list item JSON plus resolved folder path, status and priority names and the include flags. Unchanged fingerprint and a full refresh younger than `full_refresh_days` (default 7) mean no step or link requests and no write. A failed step or link request keeps the existing file and the old state entry, so the case is retried next time.
- **Frontmatter** is built with `serde_yaml` (title, key, name, project, folder, status, priority, owner, labels, tags, links, created_on, updated_on), so special characters in names cannot break YAML.
- **Script fetchers:** argv with `{path}` substitution, stdin closed, last 3 stderr lines in the `fetch_failed` problem.
- **Removed:** `connectors/`, `envelope.rs`, `ingest_envelopes`, `colibri sync`, `colibri connectors`, `AppConfig.connector_jobs`. A `connectors:` key fails config loading with the replacement YAML in the message.
- **Migration:** document ids change from `zephyr-ctslab:CTSLAB-T1` to `zephyr-ctslab:CTSLAB-T1.md`; the first update needs `--allow-mass-prune` once (documented in README, "Upgrading from 0.15").

### After code review

- `.fetch-state.json` starts as a copy of the previous state and is saved every 50 written test cases, so an interrupted first fetch keeps its progress and still owns (and can later delete) every file it wrote.
- A test case whose steps/links fetch or file write fails gets an empty fingerprint: it stays owned and is fetched again on the next run, also after a failed full refresh.
- The fetcher applies the mirror prune limit to file deletions (`max(min_count, max_fraction × known cases)`); above it nothing is deleted, the fetch counts as incomplete and `--allow-mass-prune` lifts it. This protects against a complete but empty listing (token without permission, changed `folder_path`).
- Script fetchers stop after 60 minutes (shared `run_with_timeout` with the converters).
- Pagination stops after 10,000 pages and reports the listing as incomplete.
- A fetch problem never downgrades a reconcile `error` to `partial` (the worse status wins).
- A fetched mirror's folder may not equal, contain or lie inside another mirror's folder; plain mirrors may still nest.

## Tasks

1. `fetch/mod.rs`: `MARKER`, `FetchReport`, `prepare_folder`, `write_if_changed`; tests (AC-027.1, AC-027.2 write part).
2. Move `connectors/zephyr_scale/{api,folders,html_to_md,render}.rs` to `fetch/zephyr/`; pure `next_page` with `total` check; `(Vec, complete)` from `paginate`; tests (AC-026.1).
3. `render.rs`: `serde_yaml` frontmatter; test with special characters.
4. `fetch/zephyr/mod.rs`: `ZephyrSource` trait, `ApiSource`, `.fetch-state.json`, `fetch`; tests with `FakeZephyr` (AC-025.1, AC-026.1 deletion part, AC-026.2, AC-027.2, AC-028.1, full refresh) and a reconcile test (AC-025.2).
5. `fetch/command.rs`; test.
6. `config.rs`: `mirrors_dir`, optional `path`, `fetch:` (tagged, `deny_unknown_fields`), `connectors:` error; tests (AC-030.1).
7. `ingest/update.rs` + `ingest/mirror.rs`: run fetch before reconcile, `fetch_incomplete` blocks prune, merge fetch problems, dry run skips fetch.
8. Remove connectors, envelope, `sync`, `connectors` commands; doctor checks token env var or command on PATH; `instructions` lists mirrors.
9. `tests/cli_fetch.rs`: fake Zephyr HTTP server (two pages without `isLast`, frontmatter search, second run makes no step requests) and script fetcher (AC-029.1).
10. README/CLAUDE.md; code review; `/verify` for REQ-025..030.
