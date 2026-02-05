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

# Index markdown corpus
colibri index
colibri index --folder Books --force

# Semantic search
colibri search "microservices patterns"
colibri search "clean architecture" --json --limit 10

# MCP server (for Claude integration)
colibri serve
```

## Configuration

CoLibri reads configuration from `~/.config/colibri/config.yaml`:

```yaml
embedding_model: nomic-embed-text
ollama_base_url: http://localhost:11434
top_k: 20
similarity_threshold: 0.3

sources:
  - name: Books
    path: ~/Library/Books
    doc_type: book
    mode: incremental
```

## Data Directory

Index and metadata are stored in `~/.local/share/colibri/`:

- `lancedb/` — Vector index
- `manifest.json` — Change tracking for incremental indexing
- `index_meta.json` — Schema version and stats

## Development

```bash
make check    # Type-check (fast)
make test     # Run tests
make lint     # Clippy linter
make format   # Format code
```

## License

MIT
