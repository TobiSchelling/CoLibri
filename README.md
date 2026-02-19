# CoLibri

Local RAG system for semantic search over markdown content. Indexes markdown files into LanceDB and exposes search via CLI and MCP server.

## Installation

### Homebrew (macOS)

```bash
brew tap TobiSchelling/tap
brew install colibri
```

### From Source

Requires Rust toolchain and protobuf compiler:

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install protoc (macOS)
brew install protobuf

# Build
cargo build --release

# Binary at target/release/colibri
```

## Prerequisites

CoLibri uses Ollama for local embeddings:

```bash
brew install ollama
ollama serve
ollama pull nomic-embed-text
```

## Commands

```bash
# Health check
colibri doctor
colibri doctor --strict
colibri doctor --json
colibri doctor --json --strict

# Migration lifecycle
colibri migrate --dry-run
colibri migrate

# Profile/routing status
colibri profiles
colibri profiles --json

# Plugin runtime (ingestion skeleton)
colibri plugins run --manifest plugins/examples/filesystem_markdown/plugin_manifest.json \
  --config-json '{"root_path":"/path/to/markdown","classification":"internal"}'

# Plugin ingestion into canonical store
colibri plugins ingest --manifest plugins/examples/filesystem_markdown/plugin_manifest.json \
  --config-json '{"root_path":"/path/to/markdown","classification":"internal"}'
colibri plugins ingest --manifest plugins/examples/filesystem_markdown/plugin_manifest.json \
  --config-json '{"root_path":"/path/to/markdown","classification":"internal"}' --dry-run

# Plugin incremental sync with persisted cursor state
colibri plugins sync --manifest plugins/examples/filesystem_markdown/plugin_manifest.json \
  --config-json '{"root_path":"/path/to/markdown","classification":"internal"}'
colibri plugins jobs --validate-manifests
colibri plugins sync-all
colibri plugins sync-all --job fs_docs --dry-run --json
colibri plugins sync-all --index-canonical --index-force
colibri plugins sync-all --index-canonical --generation gen_2026_02_18_candidate --activate

# Plugin sync-state operations
colibri plugins state list
colibri plugins state show --manifest plugins/examples/filesystem_markdown/plugin_manifest.json \
  --config-json '{"root_path":"/path/to/markdown","classification":"internal"}'
colibri plugins state reset --manifest plugins/examples/filesystem_markdown/plugin_manifest.json \
  --config-json '{"root_path":"/path/to/markdown","classification":"internal"}' --yes

# Generation lifecycle
colibri generations create gen_2026_02_18_candidate
colibri generations create gen_2026_02_18_candidate --activate
colibri generations list
colibri generations list --json
colibri generations activate gen_2026_02_18_candidate
colibri generations activate gen_2026_02_18_candidate --allow-unready
colibri generations delete gen_2026_02_18_candidate --confirm gen_2026_02_18_candidate

# Index markdown corpus
colibri index
colibri index --generation gen_2026_02_18_candidate --force
colibri index --generation gen_2026_02_18_candidate --force --activate
colibri index --folder Books --force
colibri index --canonical --force

# Semantic search
colibri search "microservices patterns"
colibri search "clean architecture" --json --limit 10

# MCP server (for Claude integration)
colibri serve --check
colibri serve --check --json
colibri serve
```

`colibri index --generation <id>` builds that generation without switching active traffic.  
Switch explicitly with `colibri generations activate <id>`.
Activation is blocked when a generation has zero serve-ready profiles unless `--allow-unready` is used.
`colibri generations list` shows per-profile lifecycle state (`prepared`, `building`, `ready`, `error`) from `metadata.db`.
`colibri profiles` shows `serve_ready` per profile for the active generation.
`colibri doctor` reports serving-alignment drift (missing generation metadata, non-ready lifecycle state, or model mismatch).
`colibri serve` performs the same checks at startup and refuses to start when no profile is queryable.

Use `--activate` to switch automatically after a successful index run.
Deleting the active generation requires both `--force` and `--confirm <generation>`.

## Configuration

CoLibri reads configuration from `~/.config/colibri/config.yaml`:

```yaml
sources:
  - name: Books
    path: ~/Library/Books
    doc_type: book
    mode: incremental
    classification: confidential

ollama:
  base_url: http://localhost:11434
  embedding_model: bge-m3

embeddings:
  default_profile: local_secure
  profiles:
    - id: local_secure
      provider: tei
      endpoint: http://localhost:8080
      model: bge-m3
      locality: local
    - id: cloud_fast
      provider: openai_compatible
      endpoint: https://api.example.com
      model: text-embedding-3-large
      locality: cloud

routing:
  classification_profiles:
    restricted: local_secure
    confidential: local_secure
    internal: cloud_fast
    public: cloud_fast

plugins:
  jobs:
    - id: fs_docs
      manifest: ~/GIT_ROOT/GIT_HUB/CoLibri/plugins/examples/filesystem_markdown/plugin_manifest.json
      enabled: true
      config:
        root_path: ~/Documents/knowledge
        classification: internal
```

`colibri search` and `colibri serve` federate results across all indexed embedding profiles, so clients use one interface regardless of local/cloud profile routing.
Only profiles marked `ready` in `metadata.db.index_generations` for the active generation are queried; model/index mismatches are rejected to prevent serving-plane drift.

## Data Directory

Runtime data is stored under `COLIBRI_HOME` (default: `~/.local/share/colibri/`):

- `canonical/<classification>/...` — Canonical markdown storage grouped by classification
- `indexes/<generation>/lancedb/` — Vector index per generation
- `metadata.db` — Application-managed metadata store
- `metadata.db.index_generations` — Per-generation/profile lifecycle and activation timestamps
- `metadata.db.document_blobs` — Markdown blob metadata (`path`, `size`, `checksum`)
- `metadata.db.document_index_state` — Per-document index state by generation/profile (`indexed`, `error`, `deleted`)
- `metadata.db.migration_log` — Applied migration records for audit/debug
- `manifest.json` — Indexing state + active generation pointer
- `state/`, `backups/`, `logs/` — Runtime state, backups, logs

You can relocate all data by setting:

```bash
export COLIBRI_HOME=/path/to/portable/colibri
```

`colibri plugins ingest` writes validated plugin envelopes to `canonical/` and records document metadata/tombstones in `metadata.db`.
`colibri plugins sync` additionally persists and reuses a per-plugin cursor in `metadata.db.sync_state` for incremental ingestion.
`colibri plugins sync-all` executes all configured `plugins.jobs` in one run.
Use `--index-canonical` to chain indexing after a successful sync-all run.
Use `colibri plugins state` to inspect or reset cursor entries.
Use `colibri index --canonical` to index that managed corpus.

## Development

```bash
make check    # Type-check (fast)
make test     # Run tests
make lint     # Clippy linter
make format   # Format code
```

## License

MIT
