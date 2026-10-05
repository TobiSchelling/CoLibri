# CoLibri ingestion redesign: library + mirrors, files-first

## Context

CoLibri currently returns nothing: the SQLite `documents` table was wiped (the `ensure_metadata_db` fallback in `src/config.rs:559-590` renames the DB on any bootstrap error, and every `colibri serve` writes to SQLite at startup), while LanceDB still holds 333k orphan chunks. Beyond that bug, the ingestion model does not fit how the user works:

- Connectors re-fetch and re-convert everything on every sync (docling on every PDF, every Zephyr test case plus steps and links). There is no cursor and no conversion cache.
- Nothing is ever tombstoned. Deleted or moved files stay searchable forever. The doc_id includes a hash of the root path, so moving a root duplicates everything (the 73 books were indexed twice; 807 of 825 vault notes are stale after the vault restructure).
- Books need different semantics from living sources. A book is added once, should never be re-converted, and should survive a source move. Zephyr, the vault, architecture docs and later requirements/Confluence content need incremental updates where deletions propagate.

The intended outcome is two simple verbs: `colibri add` for books (explicit file or sweep of the calibre library, skips anything already known), and `colibri update` for everything that changes. One `colibri status` screen shows what is in, what changed and what has problems. The user agreed to drop all data and rebuild, so the schema can change freely and no migration code is needed.

## User experience (target)

```
colibri add                       # sweep configured library roots (calibre); new books only
colibri add <file|folder>...      # add specific books; already-known ones are skipped
colibri add <book> --reconvert    # redo a bad conversion (e.g. marker instead of docling)
colibri list [--books]            # what is in the library
colibri remove <title|id>         # sticky: a later sweep will not re-add it
colibri update [name...]          # fetch remote sources, reconcile mirrors, sweep library, index
colibri update --dry-run          # show adds / changes / prunes before applying
colibri status                    # per collection: counts, last run, pending index, problems
colibri search ... / colibri serve  # unchanged surface, read-only
colibri reset                     # drop all data (typed confirmation)
```

## Concept: one engine, two policies, everything is a file

CoLibri indexes local folders only. A folder belongs to one of two policies:

| | Library (books) | Mirror (living sources) |
|---|---|---|
| Identity | calibre uuid from sibling `metadata.opf`, else sha256 of the file | `<mirror name>:<relative path>` |
| Missing at source | keep (book survives moves; source path updated when the uuid reappears) | prune, guarded (complete walk, root present, mass-delete threshold) |
| Changed at source | EPUB: reconvert, re-embed only if markdown hash changed. PDF: flag in status, reconvert only on `--reconvert` | reconvert/re-read, re-embed if hash changed |
| Metadata | OPF title, authors, language, tags into SQLite and frontmatter filters | YAML frontmatter into `frontmatter_json` |
| Removal | explicit `remove`, sticky | follows the source |

Remote sources (Zephyr now, Confluence later) become fetchers that write markdown with frontmatter into a mirror folder. The mirror reconcile then ingests them like any other folder. Fetchers support a built-in `zephyr_scale` type and a generic `command` type (any script that writes files), so new sources need no Rust.

Fetched mirror folders live in a visible location for DEVONthink: `mirrors_dir: ~/Documents/CoLibri/mirrors` (not iCloud-synced, checked), one subfolder per fetched mirror, overridable per mirror with `path:`. The user indexes `~/Documents/CoLibri/mirrors` in DEVONthink. Rules for these folders: CoLibri owns them (each carries a `.colibri-mirror` marker, and fetchers only delete inside marked folders); edits made in DEVONthink get overwritten on the next fetch; unchanged files are never rewritten, so DEVONthink sees stable mtimes. They must stay outside `~/PKM` (the vault mirror would index them twice) and outside OneDrive.

Decided (2026-10-05): Variant B (files-first), scope P0 now then P1-P5, fetched mirrors in a DEVONthink-visible folder. Variant A (in-process mirrors) was rejected because every new source would need Rust code, there would be two change-detection engines, and fetched content would stay invisible to DEVONthink.

## Phases (each ships, passes `make test lint`, and is usable)

### P0: Stop the wipe (S), ship as v0.14.2 first
- `src/config.rs`: remove the rename-to-`legacy-json.bak` fallback in `ensure_metadata_db`; return the error.
- `src/cli/serve.rs`, `src/cli/search.rs`, `src/cli/instructions.rs`: use `load_config_no_bootstrap()` so read paths never write.
- `src/metadata_store.rs`: add `-cmd ".timeout 5000"` to `exec_script` and `query_json` (interim until P1).
- Tests: a corrupt `metadata.db` errors and stays byte-identical in place; the serve config path does not create or modify `metadata.db`.

### P1: Storage foundation (M)
- Replace the `sqlite3` CLI executor in `src/metadata_store.rs` with `rusqlite` (bundled): WAL, `busy_timeout`, bound parameters, read-only connections for serve/search, `PRAGMA user_version = 7`, schema mismatch errors with "run `colibri reset`" (never rename or delete).
- New schema: `documents` (doc_id, collection, key, source_path, title, authors_json, tags_json, language, frontmatter_json, doc_type, format, format_pinned, converter, source_size, source_mtime, source_sha256, content_hash, markdown_path, status active/removed/missing, removed_reason, indexed_hash, chunk_count, indexed_at, timestamps), `conversions` (source_sha256 to markdown, converter), `collections` (last run and report), `problems`, `meta`.
- New `src/lock.rs`: exclusive write lock for add/update/index/remove/reset.
- New `src/cli/reset.rs`.
- Canonical path becomes `canonical/<collection>/<sha(doc_id)[..24]>.md`.
- Delete: `document_blobs`, `index_generations`, `document_index_state`, `embedding_profiles`, `routing_policy`, `migration_log`, `schema_versions`, `src/cli/migrate.rs`, `manifest.json` and unused dirs (`state/`, `backups/`, `logs/`).

### P2: Indexer and query simplification (M)
- `src/indexer.rs`: one LanceDB table at `<home>/index/lancedb`; index state on the `documents` row; commit per document so an interrupted run keeps finished work; build FTS once at the end of a run, then compact; new `purge_orphans()` that diffs LanceDB doc_ids against live docs. Reuse `split_text`, `rows_to_batch`, `chunks_schema`, `embed_texts_with_progress` (`src/embedding.rs`).
- `src/config.rs`: drop embedding profiles, routing, classification, generations; config becomes `embedding: {endpoint, model}` (keep `ollama:` as alias).
- `src/query.rs`, `src/mcp.rs`, `src/serve_ready.rs`: single backend; `classification` filter becomes `collection`; `list_books` returns title, authors, source path, chunk count.
- Delete `src/cli/profiles.rs` and classification everywhere.

### P3: Mirrors, `update`, `status` (M)
- New `src/ingest/convert.rs` (moved from `src/connectors/filesystem.rs`: `convert_pdf`, `convert_with_pandoc`, `convert_pptx`, `which_exists`), one pandoc mode (`-t gfm --wrap=none`), goes through the `conversions` cache, `Converter` trait for test fakes.
- New `src/ingest/walk.rs` (moved `discover_files`, `walk_dir`, `should_exclude`): a directory error aborts the walk; skips OneDrive online-only (dataless) files and records a problem.
- New `src/ingest/frontmatter.rs` (moved `parse_frontmatter`, PlantUML enrichment).
- New `src/ingest/mirror.rs`: `reconcile -> Plan{add, change, unchanged, prune}` then `apply`. Size+mtime fast path for converted formats. `.yaml/.yml` indexed as fenced blocks (currently silently skipped). Prune only after a complete walk, root present, below `prune.max_fraction` / `min_count`, otherwise a problem entry; `--allow-mass-prune` overrides.
- New `src/cli/update.rs`, `src/cli/status.rs`; typed `mirrors:` config with `deny_unknown_fields`.

### P4: Library (M-L)
- New `src/ingest/calibre.rs`: `parse_opf` (roxmltree), `scan_book_dir` choosing a format by `prefer_formats` (EPUB before PDF), ignoring `.acsm` (DRM-only books become a problem entry), `.original_epub`, `cover.jpg`, `.caltrash`, `.calnotes`.
- New `src/ingest/library.rs`: `add_paths` and `sweep` on the shared engine; identity resolution, format pinning, sticky removal, source path update on moves without conversion, conversion persisted before embedding.
- New `src/cli/add.rs`, `src/cli/list.rs`, `src/cli/remove.rs`. Delete `src/cli/import.rs` (keep `import` as hidden alias for one release).

### P5: Zephyr fetcher writing files (M)
- Move `src/connectors/zephyr_scale/{api,folders,render,html_to_md}.rs` to `src/fetch/zephyr_scale/`; new fetcher writes `<KEY>.md` with frontmatter, keeps a `.fetch-state.json` fingerprint per test case, refetches steps/links only when the fingerprint changes (plus a periodic full refresh), writes atomically and skips identical bytes, deletes files only on a complete listing and only in folders carrying a `.colibri-mirror` marker.
- Fix `paginate()` in `api.rs` (missing `isLast` currently ends after page 1; check `total`). Step/link fetch errors keep the existing file instead of writing a degraded one.
- New `src/fetch/command.rs` for script-based fetchers.
- Delete `src/connectors/`, `src/envelope.rs`, `ingest_envelopes`, `src/cli/sync.rs`, `src/cli/connectors.rs`; an old `connectors:` config key becomes a clear error pointing to `mirrors:`.

### P6: Polish (S-M)
- `bootstrap` becomes `init` writing the new config; rewrite README, `CLAUDE.md`, `docs/user/*`, `tour`, `instructions`; new ADR "files-first ingestion", mark ADR 0001/0002 superseded; delete `plugins/`, obsolete `docs/plans/2026-02-2*`.
- Optional: rename reuse (a pruned and a new doc with equal content hash move chunks instead of re-embedding).

## Target config

```yaml
data:
  directory: ~/.local/share/colibri
mirrors_dir: ~/Documents/CoLibri/mirrors   # fetched mirrors default to <mirrors_dir>/<name>; indexed in DEVONthink
embedding: { endpoint: http://localhost:11434, model: bge-m3 }
chunking: { chunk_size: 512, chunk_overlap: 64 }
retrieval: { top_k: 10, similarity_threshold: 0.3 }

library:
  doc_type: book
  prefer_formats: [epub, pdf]
  pdf_converter: docling
  roots:
    - path: "~/Library/CloudStorage/OneDrive-Hilti/00 My Workflow/101 Bibliothek/eBooks - calibre"
      layout: calibre

mirrors:
  - name: vault
    path: ~/PKM
    doc_type: note
    include: ["**/*.md"]
    exclude: [".obsidian/**", ".claude/**", ".github/**", ".git/**", "**/.trash/**",
              "0999_SYSTEM/TEMPLATES/**", "**/node_modules/**", "**/.venv*/**"]
  - name: architecture
    path: ~/GIT_ROOT/GIT_LAB/ONTrack/architecture-artifacts
    doc_type: architecture
    include: ["**/*.md", "**/*.yaml", "**/*.yml"]
    plantuml_summaries: true
  - name: zephyr-ctslab
    doc_type: test_case
    fetch: { type: zephyr_scale, project_key: CTSLAB, token_env: ZEPHYR_API_TOKEN, full_refresh_days: 7 }

prune: { max_fraction: 0.2, min_count: 25 }
```

## Execution order and rebuild
1. P0: implement, test, release v0.14.2 (Homebrew), so running MCP servers stop being a wipe risk.
2. Before P1: capture this plan as a spec with `/specify` in `specs/files-first-ingestion/` (requirements with acceptance criteria per phase), so `/verify` has a contract to check against.
3. P1 to P5 in order, each on its own branch with tests and `/verify`; no release in between unless needed.
4. Rebuild once after P5: start Ollama, stop running `colibri serve` processes, move `~/.local/share/colibri` to a dated backup, write the new config, run `colibri update` (Zephyr fetch into `~/Documents/CoLibri/mirrors/zephyr-ctslab`, reconcile vault and architecture, calibre sweep, index). I will time one docling PDF first and report the estimate before the full run.
5. Release the redesigned version, then the user adds `~/Documents/CoLibri/mirrors` as an indexed folder in DEVONthink.
6. Delete the backup after the user confirms.

## Verification
- Per phase: `make format && make lint && make test`; new unit tests listed per phase use tempdir fixtures and a fake converter (reuse helpers `metadata_store.rs` `temp_db_path`/`bootstrap_store`, `mcp.rs` `test_config`, `filesystem.rs` `make_test_dir`).
- Key behavioural tests: second `add` sweep makes 0 converter calls; removed book not re-added by sweep; same calibre uuid at new path updates path without conversion; mirror delete is pruned, missing root prunes nothing, mass guard trips; unchanged second `update` makes 0 writes and 0 embeds; interrupted index run resumes; orphan purge; Zephyr pagination completeness.
- End to end on the real setup: `colibri status` clean; `colibri list --books` shows about 96 calibre titles with authors; `colibri search --mode keyword "relative estimation"` and a semantic and hybrid query return hits whose paths exist; MCP `list_books` and `search_books` work from a fresh Claude session; delete a test note in the vault, `colibri update`, confirm it is pruned; `colibri add` again reports 0 new books.
- Run `/verify` before each commit; release via the 10-step process in `CLAUDE.md` after P0 and after the phase that completes the rebuild.
