# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
make dev              # Install with dev dependencies (uv sync --all-extras)
make test             # Run all tests (uv run pytest tests -v)
make lint             # Lint + typecheck (ruff check + mypy)
make format           # Auto-format (ruff format + ruff check --fix)

# Run a single test file or test class
uv run pytest tests/test_indexer_incremental.py -v
uv run pytest tests/test_indexer_incremental.py::TestIndexFolder::test_first_run_indexes_everything -v
```

## Architecture

CoLibri is a local RAG system that indexes markdown content into LanceDB for semantic search, exposed via MCP (Claude) and REST API interfaces.

### Data flow

```
Content Sources → Indexer → LanceDB ← Query Engine ← MCP Server / REST API
                    ↑                                        ↑
              Embedding (Ollama)                        Embedding (Ollama)
```

### Module layers

**Infrastructure (shared by indexer + query):**
- `embedding.py` — Ollama `/api/embed` HTTP client with batching
- `index_meta.py` — Schema version tracking (`SCHEMA_VERSION`); bump to force rebuild
- `source_factory.py` — Creates `MarkdownFolderSource` instances from `FolderProfile` config, handling nested source exclusions
- `manifest.py` — JSON-based change tracking (mtime + content hash) for incremental indexing

**Core pipeline:**
- `config.py` — YAML config from `~/.config/colibri/config.yaml`, module-level constants (`SOURCES`, `LANCEDB_DIR`, etc.). Config is loaded at import time.
- `indexer.py` — Chunking, per-folder indexing with four modes (static/incremental/append_only/disabled), LanceDB table management
- `query.py` — `SearchEngine` class: semantic search, document retrieval, book listing, topic browsing. Singleton via `get_engine()`.

**Import pipeline (`processors/`):**
- Registry pattern: processors auto-register via side-effect imports in `__init__.py`
- `ProcessorRegistry.get_processor(path)` returns the right processor by file extension
- PDF (`pymupdf4llm`) and EPUB (`ebooklib`) processors produce `ExtractedContent` → written to library as markdown with YAML frontmatter

**Content sources (`sources/`):**
- `ContentSource` ABC with `list_documents()` and `read_document()` methods
- `MarkdownFolderSource` — primary source type, supports configurable extensions and path exclusions
- `ObsidianSource` — Obsidian-aware variant with wiki-link resolution

**Interfaces:**
- `cli.py` — Click CLI (`colibri` command). Import commands use `_import_document()` internally.
- `mcp_server.py` — MCP stdio server for Claude integration (7 tools)
- `api_server.py` — FastAPI REST server for HTTP/Copilot integration

### Key patterns

- **Per-source profiles**: Each content directory is a `FolderProfile` with its own indexing mode, doc_type, chunk settings, and file extensions. Defined in config YAML under `sources:`.
- **Incremental indexing**: `Manifest` tracks file state (mtime + SHA-256). Only changed files are re-embedded. Schema version mismatch triggers automatic full rebuild.
- **Tests mock embeddings**: Integration tests use `_fake_embed` returning fixed-dimension vectors and patch `colibri.indexer.embed_texts` (patch the consumer module, not `colibri.embedding`).

## Style

- Python 3.11+, type hints required
- Ruff rules: `E, F, I, UP, B, SIM` at 100-char line length
- Conventional commits
