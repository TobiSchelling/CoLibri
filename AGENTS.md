# CoLibri Agent Guide

This repo is **CoLibri**: a local-first RAG system that imports PDF/EPUB → Markdown, indexes content into LanceDB, and exposes search via **CLI**, **MCP** (Claude integration), and a **REST API**.

If you’re an automated coding agent, this file is your “how to be effective here” map: where things live, how to run the suite, and the sharp edges to avoid.

## Quick Start

```bash
make dev          # uv sync --all-extras
make test         # uv run pytest tests -v
make lint         # uv run ruff check + uv run mypy
make format       # uv run ruff format + ruff --fix
```

The canonical command runner is `uv` (see `Makefile`).

## Repository Map

Core code:

- `src/colibri/cli.py`: Click CLI (`colibri ...`)
- `src/colibri/indexer.py`: indexing pipeline (chunking, modes, LanceDB tables)
- `src/colibri/query.py`: search/query engine (`SearchEngine`)
- `src/colibri/embedding.py`: Ollama embedding client (`/api/embed`)
- `src/colibri/manifest.py`: incremental indexing change tracking (mtime + SHA-256)
- `src/colibri/sources/`: content sources (markdown folders, Obsidian)
- `src/colibri/processors/`: import pipeline (PDF/EPUB → `ExtractedContent`)
- `src/colibri/mcp_server.py`: MCP stdio server (Claude tools)
- `src/colibri/api_server.py`: FastAPI server (HTTP integration)
- `src/colibri/setup.py`: interactive setup (writes config + optional MCP config)

Docs:

- `README.md`: user-facing overview + CLI usage
- `docs/INSTALLATION.md`: install/config/use/uninstall
- `docs/ARCHITECTURE.md`: deep architecture diagrams + flows
- `docs/MAINTENANCE.md`: maintainer decisions and upgrade guidance
- `CLAUDE.md`: contributor quick commands + key design notes (worth reading first)

## Running Locally (Without Foot-Guns)

CoLibri reads/writes user-level data by design. As an agent, avoid “surprising” writes.

Safe defaults for local experimentation:

```bash
export COLIBRI_DATA_DIR="$(pwd)/.tmp/colibri-data"
export COLIBRI_LIBRARY_PATH="$(pwd)/.tmp/library"
mkdir -p "$COLIBRI_DATA_DIR" "$COLIBRI_LIBRARY_PATH"
```

Notes:

- Config loads at import-time from `~/.config/colibri/config.yaml` (see `src/colibri/config.py`).
- `colibri setup` may write to `~/.config/colibri/config.yaml` and `~/.mcp.json`. Only run it if explicitly requested, or after sandboxing with env vars / patched paths.
- This repo contains a repo-local MCP config example: `.mcp.json`. Treat it as development convenience, not a source of truth for user machines.

## Configuration & Env Overrides

Supported env overrides (non-exhaustive, but the “usual suspects”):

- `COLIBRI_LIBRARY_PATH`: override derived library root
- `COLIBRI_DATA_DIR`: override data dir (default uses XDG data home)
- `XDG_DATA_HOME`: impacts default data dir resolution
- `OLLAMA_BASE_URL`: Ollama endpoint (default `http://localhost:11434`)
- `COLIBRI_EMBEDDING_MODEL`: embedding model name (default `nomic-embed-text`)

When you need to make code testable, prefer dependency injection or patching constants/paths in `colibri.config` rather than reading the real user config.

## Tests: Expectations And Patterns

```bash
uv run pytest tests -v
```

Testing conventions you should follow:

- Tests generally patch file paths/constants (example: patch `colibri.config.CONFIG_FILE`/`CONFIG_DIR`) to avoid touching real `~/.config`.
- Embedding calls are mocked in integration-ish tests; patch the **consumer module** (e.g. `colibri.indexer.embed_texts`) rather than `colibri.embedding` (see `CLAUDE.md` guidance).
- Network/process interactions are mocked (example: translation tests patch `httpx.*` and `subprocess.run`).

If you add a feature that touches the filesystem, add tests that use `tmp_path` and patch config paths so the suite stays hermetic.

## Style & Quality Bar

- Python `>=3.11`, type hints required
- `ruff` formatting + linting (line length `100`)
- `mypy` runs in `strict` mode (see `pyproject.toml`)

Before calling something “done”, run:

```bash
make format
make lint
make test
```

## Common Pitfalls

- Config is loaded at import time: changes to env vars after importing `colibri.config` will not retroactively apply unless you reload modules.
- Index schema/versioning: if you change vector dimensions or schema-related fields, you may need to bump schema versioning (see `src/colibri/index_meta.py`) and ensure rebuild behavior is correct.
- Don’t assume Ollama is running during tests. Most tests are written to be offline and deterministic.

## What To Update When You Make Changes

- CLI surface area changes: update `README.md` (and potentially `docs/INSTALLATION.md`)
- Architecture changes: update `docs/ARCHITECTURE.md` if diagrams/flows are impacted
- Dependency/tooling changes: update `docs/MAINTENANCE.md` if it affects long-term decisions

