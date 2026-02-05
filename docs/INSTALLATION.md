# CoLibri Installation & Usage Guide

Complete lifecycle reference: install, configure, use, update, uninstall.

---

## Prerequisites

| Requirement | Version | Purpose |
|-------------|---------|---------|
| Python | ≥3.11 | Runtime |
| Ollama | Latest | Local embedding generation |
| Git | Any | Clone the repository |

### Installing Ollama

**macOS:**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

After installing, pull the embedding model:
```bash
ollama pull nomic-embed-text
```

---

## Installation

### Install from Git (Recommended)

```bash
pip install git+https://gitlab.com/Tobias.Schelling/CoLibri.git
```

This installs CoLibri and all dependencies into your current Python environment. The `colibri` CLI command becomes available immediately.

### Install via Homebrew (macOS)

CoLibri can be distributed via a Homebrew tap (recommended for macOS users who want `brew upgrade` workflows).

```bash
brew tap TobiSchelling/tap
brew install colibri
```

Notes:

- This requires a tap repository (typically named `homebrew-colibri`) containing `Formula/colibri.rb`.
- If you maintain multiple formulae in one tap, you can always use the fully-qualified name:

```bash
brew install TobiSchelling/tap/colibri
```

- After installation, you still need Ollama + the embedding model (see Prerequisites above).

### Install into an Isolated Environment

To avoid polluting your global Python, use `pipx` or `uv`:

```bash
# With pipx
pipx install git+https://gitlab.com/Tobias.Schelling/CoLibri.git

# With uv
uv tool install git+https://gitlab.com/Tobias.Schelling/CoLibri.git
```

### Install for Development

```bash
git clone https://gitlab.com/Tobias.Schelling/CoLibri.git
cd CoLibri

# Install with uv (recommended)
uv sync

# Or with pip in editable mode
pip install -e .

# Include dev dependencies (linting, testing)
uv sync --all-extras
```

---

## Configuration

### Interactive Setup (Recommended)

```bash
colibri setup
```

The wizard checks prerequisites, configures the library path, and optionally sets up Claude Code integration.

### Manual Configuration

Create `~/.config/colibri/config.yaml`:

```yaml
library:
  path: ~/Documents/CoLibri       # Path to your markdown folder
  books_folder: Books              # Subfolder for imported books

index:
  directory: lancedb

ollama:
  base_url: http://localhost:11434
  embedding_model: nomic-embed-text

retrieval:
  top_k: 10
  similarity_threshold: 0.3

chunking:
  chunk_size: 1024
  chunk_overlap: 128
```

### Environment Variable Overrides

| Variable | Overrides |
|----------|-----------|
| `COLIBRI_LIBRARY_PATH` | `library.path` |
| `OLLAMA_BASE_URL` | `ollama.base_url` |
| `COLIBRI_EMBEDDING_MODEL` | `ollama.embedding_model` |

### Claude Code Integration (MCP)

The setup wizard configures this automatically. To set up manually, add to `~/.mcp.json`:

```json
{
  "mcpServers": {
    "colibri": {
      "command": "<python-path>",
      "args": ["-m", "colibri.mcp_server"]
    }
  }
}
```

Find `<python-path>` with:
```bash
python -c "import sys; print(sys.executable)"
```

---

## Usage

### Quick Start

```bash
colibri doctor                            # Verify prerequisites
colibri import ~/Downloads/book.pdf       # Import a book
colibri index                             # Build search index
colibri search "clean architecture"       # Search
```

For configuration details (sources, modes, data directory), see `docs/CONFIGURATION.md`.

### Commands

| Command | Description |
|---------|-------------|
| `colibri setup` | Interactive setup wizard |
| `colibri doctor` | Check system health |
| `colibri import <file>` | Import PDF or EPUB into library |
| `colibri index` | Build/rebuild search index |
| `colibri search "<query>"` | Search indexed content |
| `colibri books` | List indexed books |
| `colibri formats` | List supported import formats |
| `colibri config` | Show current configuration |
| `colibri status` | Check Ollama, library, and index status |
| `colibri capabilities` | Machine-readable description of the indexed corpus |
| `colibri changes` | Show what changed in the indexed corpus since a revision |
| `colibri agent-guide` | Print copy/paste instructions for CLI-native coding agents |
| `colibri serve` | Start MCP server (for Claude) |
| `colibri api` | Start REST API server |

### Import Options

```bash
colibri import book.pdf                   # Auto-detect format
colibri import book.pdf --title "Custom"  # Override title
colibri import book.epub --library ~/Notes  # Different library
```

### Search Options

```bash
colibri search "query"                    # Search all indexed content
colibri search "query" --books-only       # Books folder only
colibri search "query" -n 3              # Limit results
colibri search "query" -n 10 --json       # Machine-readable output
```

### CLI-Native Coding Agents

If your clients are coding-focused agents that work well with CLI tools, use the `colibri` CLI directly.

```bash
colibri agent-guide
colibri status --json
colibri capabilities --json
colibri changes --since 0 --json
colibri books --json
```

---

## Updating

```bash
# If installed from git
pip install --upgrade git+https://gitlab.com/Tobias.Schelling/CoLibri.git

# If installed with pipx
pipx upgrade colibri

# If installed for development
cd CoLibri && git pull && uv sync
```

---

## Uninstallation

### 1. Remove the Package

```bash
# Matches your install method
pip uninstall colibri
pipx uninstall colibri
uv tool uninstall colibri
```

### 2. Remove Configuration

```bash
rm -rf ~/.config/colibri
```

### 3. Remove MCP Integration

Edit `~/.mcp.json` and remove the `"colibri"` entry.

### 4. Remove Index Data

```bash
# Default location: inside your library
rm -rf <library-path>/.colibri
```

### 5. Remove Imported Books (Optional)

Imported books are plain markdown files. Remove individually or:
```bash
rm -rf <library-path>/Books
```

### 6. Remove Ollama Model (Optional)

```bash
ollama rm nomic-embed-text
```

### Cleanup Summary

| What | Location | Command |
|------|----------|---------|
| Package | Python environment | `pip uninstall colibri` |
| Config | `~/.config/colibri/` | `rm -rf ~/.config/colibri` |
| MCP config | `~/.mcp.json` | Edit file, remove `colibri` entry |
| Index | `<library>/.colibri/` | `rm -rf <library>/.colibri` |
| Books | `<library>/Books/*.md` | Remove individual files |
| Embedding model | Ollama | `ollama rm nomic-embed-text` |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `colibri: command not found` | Ensure pip's bin directory is in `$PATH` |
| Ollama not running | `ollama serve` |
| Embedding model not found | `ollama pull nomic-embed-text` |
| Index not found | `colibri index` |
| Config not found | `colibri setup` |
| General diagnostics | `colibri doctor` |
