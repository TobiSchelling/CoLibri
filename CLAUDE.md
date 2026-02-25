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

Run a single test: `cargo test <test_name>` (e.g., `cargo test test_navigation`).

Build prerequisite: `brew install protobuf` (required by LanceDB/Arrow transitive deps).

## Architecture

CoLibri is a local RAG system that indexes markdown content into LanceDB for semantic search, exposed via CLI and MCP server.

### Data Flow

```
Connectors → DocumentEnvelope[] → Canonical Store → Indexer → LanceDB
                                   (markdown + SQLite)    ↑
                                                   Ollama /api/embed
                                                  (batches of 32)

SearchEngine ← LanceDB ← MCP Server / CLI
     ↑
Ollama /api/embed (single query)
```

### Key Types & Boundaries

- **`AppConfig`** (`config.rs`): Resolved config loaded from YAML + env var overrides. Central to all operations.
- **`Connector` trait** (`connectors/mod.rs`): `async fn sync(&self) -> Result<Vec<DocumentEnvelope>>`. Only implementation: `FilesystemConnector`.
- **`DocumentEnvelope`** (`envelope.rs`): Canonical document representation with source, document, and metadata sections. Produced by connectors, consumed by canonical store.
- **`ConnectorJob`** (`connectors/mod.rs`): Resolved connector config with `id`, `connector_type`, `enabled`, and `config: serde_json::Value`.
- **`FilesystemConnector`** (`connectors/filesystem.rs`): Walks directories, reads/converts files (MD, PDF, DOCX, EPUB, PPTX), enriches PlantUML blocks, produces envelopes.
- **`SearchEngine`** (`query.rs`): Wraps LanceDB table. L2 distance → similarity via `exp(-distance)`. Filters by `similarity_threshold`.
- **`ColibriError`** (`error.rs`): `thiserror` enum with domain variants. CLI commands return `anyhow::Result` at the boundary.

### LanceDB Schema

Single table `"chunks"` with columns: `text`, `source_file`, `title`, `doc_type`, `folder`, `source_name`, `source_type`, `tags`, `vector` (FixedSizeList<Float32>). Vector dimension is determined by the embedding model at runtime.

### MCP Server

JSON-RPC over stdio (`mcp.rs`). Lazily initializes `SearchEngine` on first tool call. Exposes four tools: `search_library`, `search_books`, `list_books`, `browse_topics`.

### Key Patterns

- **Connector-based ingestion**: Each connector produces `Vec<DocumentEnvelope>`. Envelopes are persisted to canonical markdown store and tracked in SQLite metadata DB.
- **Content-hash deduplication**: Canonical store uses SHA-256 content hashes to skip unchanged documents.
- **Document conversion**: FilesystemConnector converts non-markdown formats via external tools (docling for PDF, pandoc for DOCX/EPUB, markitdown for PPTX).
- **PlantUML enrichment**: PlantUML blocks in markdown are parsed and entity/relation summaries inserted as HTML comments for searchability.
- **Schema versioning**: `SCHEMA_VERSION` in `config.rs`. Mismatch triggers automatic full rebuild.
- **Embedding batching**: Ollama requests batched at 32 texts, 120s timeout per batch.

## Config Structure

```yaml
data:
  directory: ~/.local/share/colibri
ollama:
  base_url: http://localhost:11434
  embedding_model: bge-m3
connectors:
  - type: filesystem
    id: books
    enabled: true
    root_path: ~/Library/Books
    include_extensions: [.md, .markdown]
    exclude_globs: ["**/node_modules/**"]
    doc_type: book
    classification: internal
    plantuml_summaries: true   # default: true
sources:                       # legacy folder profiles for indexing
  - name: Books
    path: ~/Library/Books
    doc_type: book
    mode: static               # static | incremental
```

Env var overrides: `COLIBRI_DATA_DIR`, `OLLAMA_BASE_URL`, `COLIBRI_EMBEDDING_MODEL`.

## Data Locations

- `~/.config/colibri/config.yaml` — Configuration
- `~/.local/share/colibri/` — Data directory (COLIBRI_HOME)
- `~/.local/share/colibri/canonical/` — Canonical markdown store (connector output)
- `~/.local/share/colibri/metadata.db` — SQLite metadata database
- `~/.local/share/colibri/lancedb/` — Vector index
- `~/.local/share/colibri/manifest.json` — Change tracking (legacy sources)
- `~/.local/share/colibri/index_meta.json` — Schema version and stats

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
