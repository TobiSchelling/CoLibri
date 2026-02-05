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
