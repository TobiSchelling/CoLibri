# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Commands

```bash
make build     # Build release binary
make check     # Type-check (fast)
make test      # Run tests
make lint      # Run clippy
make format    # Format code
```

## Architecture

CoLibri is a local RAG system that indexes markdown content into LanceDB for semantic search, exposed via CLI and MCP server.

### Data Flow

```
Markdown Sources → Indexer → LanceDB ← Query Engine ← MCP Server / CLI
                      ↑                       ↑
                Embedding (Ollama)      Embedding (Ollama)
```

### Module Structure

```
src/
├── main.rs           # Entry point, clap CLI
├── cli/              # Command implementations
│   ├── doctor.rs     # Health check
│   ├── index.rs      # Index command
│   ├── search.rs     # Search command
│   └── serve.rs      # MCP serve command
├── config.rs         # YAML config loading
├── embedding.rs      # Ollama HTTP client
├── indexer.rs        # Chunking + LanceDB indexing
├── query.rs          # SearchEngine
├── manifest.rs       # Change tracking for incremental indexing
├── index_meta.rs     # Schema version tracking
├── mcp.rs            # MCP stdio server (JSON-RPC)
├── sources/          # Content source abstraction
│   ├── mod.rs        # ContentSource trait
│   └── markdown.rs   # MarkdownFolderSource
└── error.rs          # Error types
```

### Key Patterns

- **Per-source profiles**: Each content directory is a `FolderProfile` with its own indexing mode, doc_type, chunk settings. Defined in config YAML.
- **Incremental indexing**: `Manifest` tracks file state (mtime + SHA-256). Only changed files are re-embedded.
- **Schema versioning**: `SCHEMA_VERSION` in `config.rs`. Mismatch triggers rebuild prompt.

## Data Compatibility

Reads/writes standard locations:
- `~/.config/colibri/config.yaml` — Configuration
- `~/.local/share/colibri/lancedb/` — Vector index
- `~/.local/share/colibri/manifest.json` — Change tracking

## Style

- Rust 2021 edition, stable toolchain
- Clippy with `-D warnings`
- Conventional commits
