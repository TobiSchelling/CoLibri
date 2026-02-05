# CoLibri

Local RAG system for semantic search over markdown content. Indexes markdown files into LanceDB and exposes search via CLI and MCP server.

## Prerequisites

### Rust Toolchain

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Protocol Buffers Compiler

LanceDB requires `protoc` for building:

```bash
# macOS
brew install protobuf

# Ubuntu/Debian
sudo apt install protobuf-compiler

# Verify
protoc --version
```

### Ollama

CoLibri uses Ollama for local embeddings:

```bash
# macOS
brew install ollama

# Start the server
ollama serve

# Pull the embedding model
ollama pull nomic-embed-text
```

## Building

```bash
# Development build
cargo build

# Release build (optimized)
cargo build --release

# Or use Make
make build
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
