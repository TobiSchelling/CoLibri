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

# First-time setup wizard
colibri bootstrap

# Health check (config, connectors, metadata DB, Ollama, index readiness)
colibri doctor
colibri doctor --json --strict

# Ingest configured sources into the canonical store (and index by default)
colibri sync
colibri sync --connector vault --dry-run --json
colibri sync --force          # re-embed everything

# Bring the index in line with the metadata DB
colibri index
colibri index --force

# Import a single PDF or EPUB as a book
colibri import ~/Downloads/book.epub --reindex

# Hybrid (default) / semantic / keyword search
colibri search "microservices patterns"
colibri search "clean architecture" --json --limit 10
colibri search "architecture decisions" --collection vault --mode keyword

# Filter by source path (substring match, repeatable)
colibri search "stakeholder commitments" --path-includes 0300_PROJECTS

# Filter by parsed YAML frontmatter (repeatable KEY=VALUE)
colibri search "test plan" --frontmatter area=SIT --frontmatter status=active

# Time-bound queries
colibri search "decisions" --since 2026-04-01T00:00:00Z

# Document-level grouping (best chunk per file + chunk_count + frontmatter)
colibri search "Heimdall" --group-by-doc --limit 5

# MCP server (for Claude integration)
colibri serve --check
colibri serve

# Delete all CoLibri data and rebuild from sources
colibri reset
colibri sync
```

Every ingested document belongs to a collection: the connector `id` for `sync`, and `books` for `colibri import`. Search results carry the collection, and `--collection` (CLI) or `collection` (MCP) restricts a search to one.

Commands that change data (`sync`, `index`, `import`, `reset`) take a write lock, so only one runs at a time; a second one fails immediately and names the process holding the lock. `search` and `serve` only read and never block writers.

`colibri serve --check` and `colibri doctor` report whether the index matches the configured embedding model; `colibri serve` refuses to start when it does not.

## Configuration

CoLibri reads configuration from `~/.config/colibri/config.yaml`:

```yaml
data:
  directory: ~/.local/share/colibri
embedding:
  endpoint: http://localhost:11434   # Ollama
  model: bge-m3
chunking:
  chunk_size: 512
  chunk_overlap: 64
retrieval:
  top_k: 10
  similarity_threshold: 0.3
connectors:
  - type: filesystem
    id: vault                        # also the collection name
    root_path: ~/PKM
    include_extensions: [".md"]
    exclude_globs: [".obsidian/**", "**/.trash/**"]
    doc_type: note
```

The older `ollama: {base_url, embedding_model}` section is still read when `embedding:` is absent. `OLLAMA_BASE_URL` and `COLIBRI_EMBEDDING_MODEL` override both. Changing the model triggers a full re-embed on the next index run.

## Data Directory

Runtime data is stored under `COLIBRI_HOME` (default: `~/.local/share/colibri/`):

- `metadata.db`: SQLite metadata (schema v7): documents, collections, problems, conversions
- `canonical/<collection>/`: canonical markdown of every document
- `index/lancedb/`: the vector and keyword index (one `chunks` table) plus `index_meta.json`
- `write.lock`: lock file held by the command currently changing data

Everything in this directory can be rebuilt from the sources with `colibri reset` followed by `colibri sync`. A metadata DB from an older CoLibri version is never modified; commands report it and point to `colibri reset`.

You can relocate all data by setting:

```bash
export COLIBRI_HOME=/path/to/portable/colibri
```

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
