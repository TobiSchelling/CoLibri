# Implementation Plan: P1 + P2 (storage foundation, index and search)

> Spec: `requirements.md` REQ-001..REQ-008. Design: `design.md` P1, P2.
> Branch: `feat/ffi-p1-storage`. P1 and P2 are implemented together: dropping `classification`
> and `document_index_state` from the schema (P1) forces the indexer and query onto the new
> model (P2), so a separate P1 would need a throwaway compatibility layer.

## Decisions

- **SQLite via `rusqlite` (bundled)**, rollback journal (not WAL) plus `busy_timeout(5s)`.
  Rollback mode lets read-only connections read without creating `-wal`/`-shm` files, which
  REQ-002 requires. Writers keep transactions short.
- **Schema v7** is identified by `PRAGMA user_version = 7`. A DB with `user_version = 0` that
  already has tables (every pre-v7 DB, created by the `sqlite3` CLI) is treated as foreign:
  error with "run `colibri reset`", file untouched.
- **Document identity during the transition (until P5):** `doc_id = "<collection>:<key>"`.
  `sync` uses the connector job id as collection and the envelope `external_id` as key;
  `import` uses collection `books`. The envelope's `doc_id`, `plugin_id` and `classification`
  are ignored by ingest.
- **Status model:** `status IN ('active','removed')` plus `source_missing` flag. Search returns
  `active` documents only (source-missing books stay searchable, REQ-022 later).
- **Write lock:** `<home>/write.lock` via `std::fs::File::try_lock` (Rust 1.89+), PID written
  into the file. Taken by `sync`, `index`, `import`, `reset` before any write.
- **Index:** one LanceDB table at `<home>/index/lancedb` (`chunks`: doc_id, text, vector),
  `index_meta.json` beside it with `schema_version = 7` and `embedding_model`. Model or schema
  mismatch triggers a full rebuild. Writes are committed per batch of documents (default 256
  chunks, configurable for tests), then FTS is rebuilt once and the table compacted and pruned.
- **Embedder trait** (`async fn embed(&self, texts) -> Vec<Vec<f32>>`) so tests use a fake.
- `load_config()` has no side effects. Write commands call `config.open_for_write()` which
  creates directories, takes the lock and opens the store read-write.

## Tasks

1. **Dependencies:** add `rusqlite = { version = "0.37", features = ["bundled"] }` (or the
   latest compatible), run `cargo check`. Remove nothing yet.
2. **`metadata_store.rs` rewrite (TDD).**
   - `open_rw(path)` (create schema on empty file, verify version), `open_ro(path)`
     (`SQLITE_OPEN_READ_ONLY`, verify version, error if missing).
   - `DocumentRecord` (all v7 columns), `upsert_document`, `get_document`, `list_documents`,
     `get_documents_by_ids`, `set_status`, `mark_indexed`, `clear_index_state`,
     `clear_all_index_state`, `meta_get/meta_set`, `transaction(|tx| ...)`.
   - Tables `conversions`, `collections`, `problems` created now (used from P3).
   - Tests: AC-001.1 (non-SQLite bytes, byte-identical), AC-001.2 (old schema / other
     user_version), round trip, read while a write transaction is open (AC-003.2).
3. **`lock.rs` (TDD):** `WriteLock::acquire(home)`, error with holder PID (AC-003.1).
4. **`config.rs` simplification.** Remove embedding profiles, routing, classification,
   generations, manifest, migrations, `state/backups/logs` dirs, `index.directory`.
   New: `embedding: {endpoint, model}` with `ollama:` fallback and the existing env overrides;
   paths `index_dir = <home>/index/lancedb`, `lock_path`. `open_for_write()`.
   Delete config tests for routing/generations; keep path/env tests and P0 tests (adapted).
5. **`canonical_store.rs`:** `ingest_envelopes(config, store, collection, envelopes, dry_run)`
   writes `canonical/<collection>/<sha256(doc_id)[..24]>.md` and upserts `DocumentRecord`
   (key, source_path from uri/external id, frontmatter, tags). One SQLite transaction per batch.
6. **`indexer.rs` rewrite of orchestration** (keep `split_text`, `chunks_schema`,
   `rows_to_batch`): deletions for non-active indexed docs, batched embed/write/mark,
   orphan purge (select `doc_id` column, delete ids not live), FTS once, compact + prune,
   index meta. Tests with a fake embedder and temp LanceDB: AC-005.1, AC-006.1.
7. **`query.rs`, `serve_ready.rs`, `mcp.rs`:** single backend; `SearchFilter.collection`
   replaces `classification`; results carry `collection`; `list_books` returns
   `title, authors, source_path, chunks`; `browse_topics(collection)`. MCP schemas: add
   `collection` to search tools, `browse_topics.collection` replaces `classification`.
   Tests: AC-007.1 (filter), AC-008.1 (tool names/params), AC-008.2 (list_books shape).
8. **CLI:** `sync`/`index`/`import` take the lock and pass collection; `search --collection`
   replaces `--classification`; new `reset [--yes]` (AC-004.1/.2, also removes legacy
   `indexes/`, `manifest.json`, `state/`, `backups/`, `logs/`, `plugins/`,
   `metadata.legacy-json.bak`); delete `profiles` and `migrate`; `doctor` reports the single
   index; `bootstrap` drops the sqlite3 check and `--classification`.
9. **Read-only guarantee:** search/serve/instructions use `open_ro`; test AC-002.1/.2
   (no file created or modified in the data dir).
10. **Cleanup and docs:** grep gate AC-007.2; README/CLAUDE.md sections on data layout,
    `reset`, config `embedding:`; `make format lint test`.
11. **Integration test** `tests/cli_storage.rs`: temp home + tiny fake Ollama HTTP server
    (std `TcpListener`, `/api/embed` returns fixed-dimension vectors); `sync` a temp folder,
    `search --mode keyword`, `reset --yes`; asserts on outputs and files.
12. **Review and verify:** code review of the branch diff, `/verify` for REQ-001..008.
