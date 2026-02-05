# CoLibri Maintenance Guide

This document is for project maintainers. It records technology decisions, lists alternatives worth evaluating, and defines processes for keeping the stack current.

---

## Table of Contents

- [Technology Decisions](#technology-decisions)
- [Evaluation Candidates](#evaluation-candidates)
- [Dependency Review Process](#dependency-review-process)
- [Model Selection](#model-selection)
- [Known Limitations](#known-limitations)

---

## Technology Decisions

Each entry records what we use, why we chose it, what alternatives exist, and when to reconsider.

### Embedding Model

| | Current |
|---|---|
| **Choice** | `nomic-embed-text` via Ollama |
| **Why** | Runs locally, no API keys, 768-dim vectors, 8192 token context |
| **Config** | `ollama.embedding_model` in config.yaml |
| **Files** | `config.py`, `indexer.py`, `query.py` |

**Alternatives to evaluate:**

| Model | Dimensions | Context | Notes |
|-------|-----------|---------|-------|
| `nomic-embed-text` | 768 | 8192 | Current choice. Good general-purpose. |
| `mxbai-embed-large` | 1024 | 512 | Higher dimensional but shorter context. |
| `snowflake-arctic-embed` | 1024 | 8192 | Strong benchmark scores, larger vectors. |
| `bge-m3` | 1024 | 8192 | Multilingual. Evaluate if non-English books needed. |
| `nomic-embed-text-v1.5` | 768 | 8192 | Check for newer nomic releases. |

**When to reconsider:** When Ollama ships new embedding models, or if retrieval quality becomes a bottleneck. Run `ollama list` to check available models.

**How to evaluate:** Change `embedding_model` in config, rebuild the index with `colibri index`, and compare search quality on known queries.

> Note: Changing the embedding model requires a full reindex. Vector dimensions must match the LanceDB schema.

### Embedding Provider

| | Current |
|---|---|
| **Choice** | Ollama (local) |
| **Why** | Privacy (books never leave the machine), no API costs, no rate limits |
| **Config** | `ollama.base_url` in config.yaml |
| **Files** | `config.py`, `indexer.py`, `query.py`, `setup.py` |

**Alternatives:**

| Provider | Latency | Cost | Privacy | Notes |
|----------|---------|------|---------|-------|
| Ollama (local) | ~50ms | Free | Full | Current. Requires local GPU for speed. |
| OpenAI `text-embedding-3-small` | ~100ms | $0.02/1M tokens | Sends data to API | High quality, requires API key. |
| Voyage AI | ~100ms | Pay-per-use | Sends data to API | Strong code/technical text embeddings. |
| HuggingFace local | ~50ms | Free | Full | More setup, but no Ollama dependency. |

**When to reconsider:** If users need cloud deployment, or if a local model significantly outperforms nomic-embed-text.

### Vector Store

| | Current |
|---|---|
| **Choice** | LanceDB (embedded) |
| **Why** | No server process, file-based, works offline, fast enough for ~100 books |
| **Config** | `index.directory` in config.yaml |
| **Files** | `indexer.py`, `query.py` |

**Alternatives:**

| Store | Type | Notes |
|-------|------|-------|
| LanceDB | Embedded | Current. Zero-infra, good for local use. |
| ChromaDB | Embedded/Server | Similar to LanceDB, larger community. Evaluate if LanceDB has issues. |
| Qdrant | Server | Better for multi-user or remote deployment. Overkill for local. |
| FAISS | Library | Facebook's library. Fast, but no metadata filtering. |
| SQLite-VSS | Embedded | SQLite extension. Interesting for simplicity. |

**When to reconsider:** If index size exceeds ~1GB, if we need multi-user access, or if LanceDB introduces breaking changes.

### RAG Framework

| | Current |
|---|---|
| **Choice** | None — direct LanceDB SDK + Ollama HTTP API |
| **Why** | Minimal dependencies, full control over chunking/embedding/retrieval pipeline |
| **Files** | `indexer.py`, `query.py` |

**Alternatives:**

| Framework | Notes |
|-----------|-------|
| Direct integration | Current. Simple, few dependencies. |
| LlamaIndex | Full-featured RAG framework. Was used previously, removed to reduce dependency surface. |
| LangChain | Similar scope to LlamaIndex. Heavier. |
| Haystack | Modular pipeline approach. Good for complex retrieval. |

**When to reconsider:** If the retrieval pipeline becomes significantly more complex (e.g., hybrid search, reranking, multi-step retrieval), a framework could reduce boilerplate.

### PDF Extraction

| | Current |
|---|---|
| **Choice** | pymupdf4llm |
| **Why** | Best Markdown output from PDFs, handles tables and formatting |
| **Files** | `processors/pdf.py` |

**Alternatives:**

| Library | Notes |
|---------|-------|
| pymupdf4llm | Current. Purpose-built for LLM consumption. Good table handling. |
| PyMuPDF (fitz) | Lower-level. pymupdf4llm builds on this. |
| pdfplumber | Good table extraction. Less Markdown-focused. |
| marker-pdf | AI-assisted PDF-to-Markdown. Higher quality but slower. |
| docling (IBM) | Document understanding. Handles complex layouts. Heavy. |

**When to reconsider:** If PDF output quality is poor for specific book layouts (multi-column, heavy math, scanned pages).

### EPUB Extraction

| | Current |
|---|---|
| **Choice** | ebooklib + BeautifulSoup + markdownify |
| **Why** | Standard EPUB parsing + HTML-to-Markdown pipeline |
| **Files** | `processors/epub.py` |

**Alternatives:**

| Library | Notes |
|---------|-------|
| ebooklib + markdownify | Current. Works well, three-library chain. |
| pandoc (via pypandoc) | Single tool for EPUB-to-Markdown. External binary dependency. |
| calibre (via CLI) | Handles more formats (MOBI, AZW). Heavy dependency. |

**When to reconsider:** If we need more ebook formats (MOBI, AZW3), calibre becomes attractive as a single conversion tool.

### CLI Framework

| | Current |
|---|---|
| **Choice** | Click + Rich |
| **Why** | Click for argument parsing, Rich for formatted output |
| **Files** | `cli.py`, `setup.py` |

**Alternatives:**

| Library | Notes |
|---------|-------|
| Click + Rich | Current. Mature, well-documented. |
| Typer + Rich | Typer is built on Click with type-hint-based API. Less boilerplate. |
| cyclopts | Newer Click alternative with better type support. |

**When to reconsider:** If adding many new commands. Typer reduces boilerplate for typed CLIs.

### REST API Framework

| | Current |
|---|---|
| **Choice** | FastAPI + Uvicorn |
| **Why** | Automatic OpenAPI schema, async support, Copilot integration |
| **Files** | `api_server.py` |

No strong reason to change. FastAPI is the standard for Python APIs.

### MCP Integration

| | Current |
|---|---|
| **Choice** | mcp SDK (official Anthropic) |
| **Why** | Required for Claude Code/Desktop integration |
| **Files** | `mcp_server.py` |

Follow upstream MCP SDK releases. This is the canonical implementation.

---

## Evaluation Candidates

Technologies worth watching that could improve CoLibri:

### Near-Term (Evaluate Next)

| Candidate | Replaces | Why Evaluate |
|-----------|----------|--------------|
| `nomic-embed-text` v1.5+ | Current model | Better retrieval quality |
| marker-pdf | pymupdf4llm | Better PDF conversion for complex layouts |
| Typer | Click | Less boilerplate for CLI commands |

### Medium-Term (Watch)

| Candidate | Replaces | Why Watch |
|-----------|----------|-----------|
| SQLite-vec | LanceDB | If SQLite ecosystem grows for vector search |
| Ollama structured outputs | Custom parsing | Cleaner metadata extraction |
| Hybrid search (BM25 + vector) | Pure vector search | Better keyword matching |
| Cross-encoder reranking | Raw similarity scores | Better result ordering |

### Longer-Term (Track)

| Candidate | Area | Why Track |
|-----------|------|-----------|
| WebAssembly embedding models | Ollama | Eliminate Ollama dependency entirely |
| MCP evolution | Claude integration | New capabilities, tools, resources |
| Local LLM for summarization | New feature | Book summaries without cloud API |

---

## Dependency Review Process

### Routine Review (Quarterly)

1. **Check for updates:**
   ```bash
   # List outdated dependencies
   uv pip list --outdated

   # Check for security advisories
   pip audit
   ```

2. **Review Ollama models:**
   ```bash
   # List available embedding models
   ollama list

   # Check for new models
   ollama search embed
   ```

3. **Run tests after updating:**
   ```bash
   uv sync
   make test
   make lint
   ```

### Before Upgrading a Dependency

1. Read the changelog for breaking changes
2. Run the full test suite
4. Test with a real library (import a book, rebuild index, search)

### Adding a New Dependency

Before adding a dependency, check:

- [ ] Is there a stdlib or existing dependency that already does this?
- [ ] Is the library actively maintained (commits in last 6 months)?
- [ ] Does it add significant transitive dependencies?
- [ ] Is it compatible with Python 3.11+?
- [ ] Does it work on macOS and Linux?

---

## Model Selection

### Current Embedding Model Profile

| Property | Value |
|----------|-------|
| Model | `nomic-embed-text` |
| Dimensions | 768 |
| Context window | 8192 tokens |
| Max safe chunk size | ~16,000 chars (conservative) |
| Provider | Ollama (local) |
| Memory usage | ~300MB |

### Evaluating a New Model

1. **Change config:**
   ```yaml
   ollama:
     embedding_model: new-model-name
   ```

2. **Pull the model:**
   ```bash
   ollama pull new-model-name
   ```

3. **Rebuild the index:**
   ```bash
   colibri index
   ```

4. **Test with benchmark queries:**
   ```bash
   colibri search "dependency injection patterns"
   colibri search "testing strategies for microservices"
   colibri search "SOLID principles"
   ```

5. **Compare results** against the previous model on the same queries.

> Important: If the new model has different vector dimensions, LanceDB will reject the new embeddings against the old schema. A full reindex is always required.

### Chunking Parameters

Current settings in `config.yaml`:

```yaml
chunking:
  chunk_size: 3000    # Characters per chunk (character-based splitting)
  chunk_overlap: 200  # Overlap between adjacent chunks
```

Chunking is character-based, splitting on paragraph and sentence boundaries. These values interact with the embedding model's context window: 3000 chars ≈ 750–1000 tokens, well within `nomic-embed-text`'s 8192-token limit. If switching to a model with a smaller context (e.g., 512 tokens), reduce `chunk_size` accordingly. The safety limit is defined in `indexer.py` as `MAX_CHUNK_CHARS = 16000`.

---

## Known Limitations

| Limitation | Impact | Potential Fix |
|------------|--------|---------------|
| Single embedding model | Can't mix models per source | Support model-per-source in config |
| Full reindex on model change | Downtime for large libraries | Incremental migration strategy |
| No hybrid search | Keyword-exact matches may rank low | Add BM25 alongside vector search |
| No reranking | Top-K results may not be optimally ordered | Add cross-encoder reranking step |
| `watchdog` imported but unused | Dead dependency | Remove or implement file watching |
| `pandas` used only for `list_books` | Heavy dependency for one function | Replace with direct LanceDB arrow query |
| Ollama required at runtime | Can't run without Ollama service | Support alternative embedding backends |

---

## File Reference

Quick reference for where each technology is integrated:

| Component | Files | Dependencies |
|-----------|-------|--------------|
| Embedding | `indexer.py`, `query.py` | httpx (Ollama HTTP API) |
| Vector store | `indexer.py`, `query.py` | lancedb |
| Chunking | `indexer.py` | None (built-in character-based splitter) |
| PDF import | `processors/pdf.py` | pymupdf4llm |
| EPUB import | `processors/epub.py` | ebooklib, beautifulsoup4, markdownify |
| Frontmatter | `sources/obsidian.py`, `processors/utils.py` | python-frontmatter, pyyaml |
| CLI | `cli.py`, `setup.py` | click, rich |
| REST API | `api_server.py` | fastapi, uvicorn |
| MCP | `mcp_server.py` | mcp |
| HTTP client | `cli.py`, `setup.py` | httpx |
| Book stats | `query.py` | pandas |
