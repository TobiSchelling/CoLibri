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
ollama pull bge-m3
```

## Commands

```bash
# In-app tour / concepts
colibri tour
colibri tour concepts
colibri tour config
colibri tour use-cases

# First-time setup wizard
colibri bootstrap

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

# Ingest configured sources into canonical store (and index by default)
colibri sync
colibri sync --job fs_docs --dry-run --json
colibri sync --force

# Index markdown corpus
colibri index
colibri index --force

# Semantic / hybrid / keyword search
colibri search "microservices patterns"
colibri search "clean architecture" --json --limit 10
colibri search "architecture decisions" --classification internal

# Filter by document path (substring match — repeatable)
colibri search "stakeholder commitments" --path-includes 03_MY_PROJECTS

# Filter by parsed YAML frontmatter (repeatable KEY=VALUE)
colibri search "test plan" --frontmatter area=SIT --frontmatter status=active

# Time-bound queries
colibri search "decisions" --since 2026-04-01T00:00:00Z

# Document-level grouping (best chunk per file + chunk_count + frontmatter)
colibri search "Heimdall" --group-by-doc --limit 5

# MCP server (for Claude integration)
colibri serve --check
colibri serve --check --json
colibri serve
```

`colibri profiles` shows index readiness per embedding profile.
`colibri doctor` reports serving-alignment drift (missing index metadata or model mismatch).
`colibri serve` performs the same checks at startup and refuses to start when no profile is queryable.

## Configuration

CoLibri reads configuration from `~/.config/colibri/config.yaml`:

```yaml
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
      manifest: ~/.local/share/colibri/plugins/bundled/filesystem_documents/plugin_manifest.json
      enabled: true
      config:
        root_path: ~/Documents/knowledge
        classification: internal
        include_extensions: [".md", ".markdown"]
```

`colibri search` and `colibri serve` federate results across all indexed embedding profiles, so clients use one interface regardless of local/cloud profile routing.
Only serve-ready profile indexes are queried; model/schema mismatches are rejected to prevent serving-plane drift.

## Data Directory

Runtime data is stored under `COLIBRI_HOME` (default: `~/.local/share/colibri/`):

- `canonical/<classification>/...` — Canonical markdown storage grouped by classification
- `indexes/<generation>/<embedding_profile_id>/lancedb/` — Vector index per generation/profile (default)
- `metadata.db` — Application-managed metadata store
- `metadata.db.document_blobs` — Markdown blob metadata (`path`, `size`, `checksum`)
- `metadata.db.document_index_state` — Per-document index state by generation/profile (`indexed`, `error`, `deleted`)
- `metadata.db.migration_log` — Applied migration records for audit/debug
- `manifest.json` — Active generation pointer (internal)
- `state/`, `backups/`, `logs/` — Runtime state, backups, logs

You can relocate all data by setting:

```bash
export COLIBRI_HOME=/path/to/portable/colibri
```

`colibri sync` executes all configured `plugins.jobs`, writes canonical Markdown under `canonical/`, and records document metadata/tombstones in `metadata.db`.
For advanced debugging, hidden subcommands exist under `colibri plugins` (cursor state, direct plugin runs).
Use `colibri index` to (re)build the vector index from the managed corpus.

## User Docs

Start here:

- `docs/user/getting-started.md`
- `docs/user/concepts.md`
- `docs/user/configuration.md`
- `docs/user/use-cases.md`
- `docs/user/troubleshooting.md`

## Development

```bash
make check    # Type-check (fast)
make test     # Run tests
make lint     # Clippy linter
make format   # Format code
```

## License

MIT
