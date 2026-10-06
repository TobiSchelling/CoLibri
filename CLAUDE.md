# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
make build     # Build release binary
make check     # Type-check (fast)
make test      # Run tests
make lint      # Run clippy
make format    # Format code
```

Run a single test: `cargo test <test_name>` (e.g., `cargo test interrupted_run_resumes`). Integration tests: `cargo test --test cli_storage`.

Build prerequisite: `brew install protobuf` (required by LanceDB/Arrow transitive deps). Toolchain is pinned in `rust-toolchain.toml` (1.93.0); it overrides the toolchain the CI workflows install.

## Architecture

CoLibri is a local RAG system that indexes markdown content into LanceDB for semantic search, exposed via CLI and MCP server.

### Data Flow

```
Connectors → DocumentEnvelope[] → Canonical Store → Indexer → LanceDB
                                   (markdown + SQLite)    ↑
                                                   Ollama /api/embed
                                                  (batches of 32)

SearchEngine ← LanceDB ← MCP Server / CLI
     ↑              ↑
Ollama embed    FTS index (BM25)
(semantic)      (keyword)
     └──── hybrid mode: both + RRF fusion ────┘
```

### Key Types & Boundaries

- **`AppConfig`** (`config.rs`): Resolved config (YAML + env overrides). `load_config()` has no side effects; write commands call `config.open_for_write()` (creates layout, takes the write lock, opens the metadata DB read-write), read paths call `config.open_read()`.
- **`MetadataStore`** (`metadata_store.rs`): rusqlite (bundled), schema v8 via `PRAGMA user_version`, rollback journal + 5 s busy timeout. `open_ro` never creates files. A DB with another schema is never modified; the error points to `colibri reset`.
- **`DocumentRecord`** (`metadata_store.rs`): one row of `documents`. `doc_id = "<collection>:<key>"`; `status` is `active`/`removed` (only `active` is searchable); `indexed_hash` is the content hash the index chunks were built from.
- **`WriteLock`** (`lock.rs`): OS file lock on `<home>/write.lock` holding the PID; second writer fails fast.
- **Mirrors** (`ingest/mirror.rs`): `reconcile_mirror` plans (walk + hash/stat, read-only) then applies (convert via cache, write canonical + rows, guarded prune, record run + problems). `ingest/walk.rs` (include/exclude globs, completeness, online-only detection), `ingest/convert.rs` (`Converter` trait, `ExternalConverter`, SHA-256 keyed `convert_cached`), `ingest/update.rs` (`run_update`: mirrors then index), `cli/status.rs` (read-only report).
- **Library** (`ingest/library.rs`, collection `books`): `sweep` (configured roots) and `add_paths` (explicit, pins format, restores removed books) build candidates per folder: `metadata.opf` with uuid → key `calibre:<uuid>`, otherwise `sha:<sha256[..16]>`; `process` decides add / metadata update / convert / flag (`source_changed` for PDFs) / skip (removed by user, reason `user`). Never deletes; unseen books get `source_missing`. `ingest/calibre.rs` parses OPF (roxmltree). Conversions use the content-addressed cache with negative caching (`--retry-failed`, missing tools are not cached).
- **`Connector` trait** (`connectors/mod.rs`): only `ZephyrScaleConnector` remains (via `colibri sync`) until it becomes a fetcher in P5. The connector `id` is the collection name.
- **`ingest_envelopes`** (`canonical_store.rs`): writes `canonical/<collection>/<sha256(doc_id)[..24]>.md` and upserts `DocumentRecord`s in one transaction.
- **`index_library`** (`indexer.rs`): drops chunks of non-searchable docs, embeds docs whose `indexed_hash != content_hash` in committed batches (resumable), purges orphan chunks, rebuilds FTS once, compacts. Takes an `Embedder` (`embedding.rs`; `OllamaEmbedder` in production, a fake in tests).
- **`SearchEngine`** (`query.rs`): one LanceDB table. Modes: `hybrid` (BM25 + vector via native RRF, default; falls back to keyword if embedding fails), `semantic` (L2 distance → `exp(-distance)`, `similarity_threshold` applies), `keyword`. Filters (`collection`, `doc_type`, path, frontmatter, `since`) are applied after the SQLite join.
- **`ColibriError`** (`error.rs`): `thiserror` enum with domain variants. CLI commands return `anyhow::Result` at the boundary.

### LanceDB Schema

Single table `"chunks"` at `<home>/index/lancedb` with columns `doc_id`, `text`, `vector` (FixedSizeList<Float32>, dimension from the model). All other metadata comes from SQLite by `doc_id`. `index_meta.json` records `schema_version` (`SCHEMA_VERSION` in `config.rs`) and `embedding_model`; a mismatch makes the index not queryable and forces a full rebuild on the next index run.

### MCP Server

JSON-RPC over stdio (`mcp.rs`). Checks index readiness (`serve_ready.rs`) and opens `SearchEngine` at startup; refuses to start if not ready. Tools: `search_library`, `search_books`, `list_books`, `browse_topics`. Search tools accept `mode`, `collection` (library only), path/frontmatter/since filters and `group_by_doc`.

### Key Patterns

- **Collections**: every document belongs to one (mirror name, connector id for `sync`, `books` for the library). Ids are stable across root moves.
- **Content-hash deduplication**: unchanged content is neither rewritten nor re-embedded.
- **Document conversion**: FilesystemConnector converts non-markdown formats via external tools (docling for PDF, pandoc for DOCX/EPUB, markitdown for PPTX).
- **PlantUML enrichment**: PlantUML blocks are parsed and entity/relation summaries inserted as HTML comments for searchability.
- **Embedding batching**: Ollama requests batched at 32 texts, 120 s timeout per batch; the indexer commits every 256 chunks.
- **Tests**: unit tests use `AppConfig::for_test(dir)`, `indexer::tests::FakeEmbedder` and `ingest::convert::fakes::CountingConverter`; `tests/cli_*.rs` drive the real binary against a temp home with a `notes` mirror and a fake Ollama HTTP server (`tests/common`).

## Config Structure

```yaml
data:
  directory: ~/.local/share/colibri
embedding:                     # legacy alias: ollama: {base_url, embedding_model}
  endpoint: http://localhost:11434
  model: bge-m3
mirrors:
  - name: vault                # = collection name
    path: ~/PKM
    include: ["**/*.md"]       # default
    exclude: [".obsidian/**"]
    doc_type: note             # default
    plantuml_summaries: true   # default
prune: {max_fraction: 0.2, min_count: 25}
library:                       # collection `books`
  prefer_formats: [epub, pdf, docx]
  roots: [{path: "~/…/eBooks - calibre"}]
connectors:                    # Zephyr only, until P5
  - type: zephyr_scale
    id: zephyr-ctslab
    project_key: CTSLAB
```

Env var overrides: `COLIBRI_HOME` / `COLIBRI_DATA_DIR`, `COLIBRI_CONFIG_PATH`, `OLLAMA_BASE_URL`, `COLIBRI_EMBEDDING_MODEL`.

## Data Locations

- `~/.config/colibri/config.yaml`: configuration
- `~/.local/share/colibri/` (COLIBRI_HOME): data directory, fully rebuildable via `colibri reset` + `colibri update`
- `metadata.db`: SQLite metadata (schema v8)
- `canonical/<collection>/`: canonical markdown
- `conversions/`: converted-file cache keyed by source SHA-256
- `index/lancedb/`: vector + FTS index and `index_meta.json`
- `write.lock`: write lock

Design and requirements for the ongoing ingestion redesign: `specs/files-first-ingestion/`.

## CI

GitHub Actions on macOS: `cargo fmt --check` → `cargo clippy -- -D warnings` → `cargo build --release` → `cargo test`. Release workflow triggers on `v*` tags, builds macOS ARM64 binary.

## Style

- Rust 2021 edition, stable toolchain
- Clippy with `-D warnings`
- Conventional commits

## Releasing

When user says "release version X.Y.Z", follow these steps:

1. **Update version** in `Cargo.toml`
2. **Commit**: `git add Cargo.toml Cargo.lock && git commit -m "chore: Bump version to X.Y.Z"`
3. **Push**: `git push`
4. **Tag and push**: `git tag vX.Y.Z && git push origin vX.Y.Z`
5. **Monitor release workflow**: `gh run list --limit 1` (wait for success)
6. **Get SHA256**: `gh release download vX.Y.Z --pattern "*.sha256" --output -`
7. **Update Homebrew formula** in `packaging/homebrew/colibri.rb`:
   - Update `version "X.Y.Z"`
   - Update `sha256 "..."`
8. **Commit formula**: `git add packaging/homebrew/colibri.rb && git commit -m "chore: Update Homebrew formula for vX.Y.Z" && git push`
9. **Update tap repo**:
   ```bash
   cd /tmp && rm -rf homebrew-tap && gh repo clone TobiSchelling/homebrew-tap
   cp packaging/homebrew/colibri.rb /tmp/homebrew-tap/Formula/
   cd /tmp/homebrew-tap && git add -A && git commit -m "Update colibri to vX.Y.Z" && git push
   ```
10. **Verify**: `brew update && brew upgrade colibri && colibri --version`
