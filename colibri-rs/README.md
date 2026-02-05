# CoLibri Rust Implementation

Rust port of the CoLibri RAG system. Reads the same config and LanceDB index as the Python version.

## Prerequisites

### Rust Toolchain

```bash
# Install via rustup (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Protocol Buffers Compiler

LanceDB requires `protoc` for building:

```bash
# macOS
brew install protobuf

# Ubuntu/Debian
sudo apt install protobuf-compiler

# Fedora
sudo dnf install protobuf-compiler

# Verify installation
protoc --version
```

## Building

```bash
# Development build
cargo build

# Release build (optimized)
cargo build --release

# Smallest binary (slower build)
cargo build --profile release-small
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

## Development

```bash
# Check without building
cargo check

# Run tests
cargo test

# Lint
cargo clippy -- -D warnings

# Format
cargo fmt
```

## Data Compatibility

This binary reads/writes the same files as the Python version:

- `~/.config/colibri/config.yaml` — shared configuration
- `~/.local/share/colibri/lancedb/` — shared vector index
- `~/.local/share/colibri/manifest.json` — shared change tracking
