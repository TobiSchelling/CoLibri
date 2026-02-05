# RAG System Implementation Plan
## Personal Knowledge Base for Technical Books

**Created:** January 26, 2026
**Author:** Implementation plan for Tobias
**Purpose:** Detailed roadmap for building a RAG system to make technical PDF books accessible through Claude Code and Claude Desktop

---

## Technical Review (January 28, 2026)

> **Reviewer Note:** This section contains a critical technical review of the original plan. Key concerns are flagged, and alternative open-source, local-first approaches are recommended.

### Critical Issues Identified

| Issue | Severity | Original | Recommended Fix |
|-------|----------|----------|-----------------|
| Mem0 cloud dependency | High | Assumes fully local operation | Mem0 has significant cloud-first design; recommend LlamaIndex or custom stack |
| Wrong package name | High | `pymcp` | Correct package is `mcp` |
| ChromaDB auth mismatch | High | Config sets auth but client doesn't use it | Either remove auth or configure client properly |
| Embedding model suboptimal | Medium | `all-MiniLM-L6-v2` | Use `BAAI/bge-base-en-v1.5` for technical content |
| Naive chunking | Medium | Character-based splitting | Use semantic chunking or recursive splitter |
| Missing hybrid search | Medium | Semantic only | Add BM25 for keyword matching |
| No re-ranking | Low | Single-stage retrieval | Add cross-encoder reranking |

### Revised Recommendation: Local-First Open Source Stack

For a truly **local, open-source** solution, I recommend building on these well-maintained components:

#### Component Evaluation Matrix

| Component | Option 1 | Option 2 | Option 3 | **Recommendation** |
|-----------|----------|----------|----------|-------------------|
| **PDF Processing** | Docling (IBM) | pymupdf4llm | marker-pdf | **pymupdf4llm** - lighter, well-maintained |
| **Chunking** | LlamaIndex built-in | semantic-text-splitter | chonkie | **LlamaIndex** - battle-tested |
| **Embeddings** | sentence-transformers | fastembed | Ollama | **Ollama** - unified local LLM platform |
| **Vector Store** | LanceDB | ChromaDB | Qdrant | **LanceDB** - embedded, no server |
| **RAG Framework** | LlamaIndex | Haystack | txtai | **LlamaIndex** - most mature, best docs |
| **Reranking** | cross-encoder | Ollama | Cohere (cloud) | **Ollama** - keeps everything local |
| **Local LLM** | Ollama | llama.cpp | vLLM | **Ollama** - easiest, well-maintained |

#### Recommended Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Claude (via MCP)                        │
│              Primary interface for user queries              │
└─────────────────────────────┬───────────────────────────────┘
                              │ MCP Protocol
┌─────────────────────────────▼───────────────────────────────┐
│                    MCP Server (Python)                       │
│                    Package: mcp (Anthropic)                  │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│               LlamaIndex Query Engine                        │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Hybrid Retrieval: BM25 + Vector Similarity          │    │
│  │  Reranking: Local cross-encoder via Ollama           │    │
│  │  Response Synthesis: Optional local LLM summary      │    │
│  └─────────────────────────────────────────────────────┘    │
└──────────┬──────────────────┬──────────────────┬────────────┘
           │                  │                  │
┌──────────▼──────┐  ┌────────▼────────┐  ┌─────▼─────────────┐
│    LanceDB      │  │     Ollama      │  │   pymupdf4llm     │
│  (embedded DB,  │  │  ┌───────────┐  │  │   (PDF → clean    │
│   no server,    │  │  │ Embeddings│  │  │    Markdown)      │
│   columnar)     │  │  │nomic-embed│  │  │                   │
│                 │  │  ├───────────┤  │  │   Ingestion only  │
│   ~50MB/10k     │  │  │ Reranking │  │  │                   │
│   chunks        │  │  │bge-reranker│ │  │                   │
│                 │  │  └───────────┘  │  │                   │
└─────────────────┘  └─────────────────┘  └───────────────────┘
```

#### Why This Stack?

**LlamaIndex** (GitHub: 40k+ stars, very active)
- Best-in-class RAG framework with excellent documentation
- Native support for hybrid search, reranking, and local LLMs
- Strong typing, good error messages, production-ready
- LanceDB integration is first-class

**LanceDB** (GitHub: 5k+ stars, backed by LanceDB Inc)
- Embedded database - no Docker, no server, just a directory
- Columnar format optimized for vector + metadata queries
- Native Python, Rust core for performance
- Automatic persistence, handles millions of vectors

**Ollama** (GitHub: 100k+ stars, extremely active)
- De facto standard for local LLM serving
- Supports embedding models (nomic-embed-text, mxbai-embed-large)
- Can run reranker models
- Simple API, excellent macOS support
- One command to install and run models

**pymupdf4llm** (part of PyMuPDF, very mature)
- Specifically designed for LLM ingestion
- Converts PDF to clean Markdown preserving structure
- Handles tables, headers, lists properly
- Much lighter than Docling

**EPUB Support** (ebooklib + markdownify)
- Direct EPUB to Markdown conversion (no PDF intermediate)
- Extracts metadata (author, publisher, ISBN) for frontmatter
- Preserves chapter structure
- Handles both PDF and EPUB via unified `colibri import` command

#### Local LLM Model Recommendations

| Task | Model | Size | Purpose |
|------|-------|------|---------|
| Embeddings | `nomic-embed-text` | 274MB | Best open embedding model |
| Embeddings (alt) | `mxbai-embed-large` | 670MB | Higher quality, slower |
| Reranking | `bge-reranker-v2-m3` | 568MB | Cross-encoder reranking |
| Summarization | `qwen2.5:7b` | 4.7GB | Optional query expansion |

All models run locally via Ollama. Total disk: ~6GB for full capability.

---

## Executive Summary

This document outlines a comprehensive plan for implementing a local RAG (Retrieval Augmented Generation) system that will allow you to search and retrieve information from your technical PDF books directly through Claude's various interfaces. The system will use MCP (Model Context Protocol) to enable Claude to autonomously search your book collection when answering questions, creating a seamless workflow where your reference materials are always accessible without manual context management.

**Primary Recommendation (Revised):** Use the **Obsidian Hybrid** approach with LlamaIndex + LanceDB + Ollama for a fully local, open-source solution that integrates with your existing knowledge management workflow. This approach provides semantic search over your technical books while enabling annotation, linking, and knowledge graph visualization through Obsidian.

See the **Technical Review** section above and **Decision Matrix** below for detailed comparison of all approaches.

---

## Understanding the Problem Space

Before diving into implementation details, it's important to understand what problem we're actually solving and why different approaches might work better or worse for your situation.

You have technical books in PDF format that contain valuable reference information for your daily work in test strategy and architecture. Currently, accessing this information requires either remembering where specific concepts are discussed or manually searching through PDFs, which is inefficient and breaks your flow when working with Claude. The ideal state is having Claude automatically retrieve relevant information from your books when it would help answer your questions, similar to how it can search the web but for your personal technical library.

The challenge is that Claude's context window, while large, cannot hold entire books at once. Even if it could, sending full books with every question would be wasteful and slow. Instead, we need a system that can quickly identify which small portions of which books are relevant to your current question and provide just those portions to Claude. This is fundamentally what RAG systems do, they bridge the gap between large knowledge bases and the practical constraints of working with language models.

The second challenge is integration. Your RAG system needs to work seamlessly with Claude Code in your terminal and Claude Desktop. This is where MCP becomes crucial, as it provides a standardized way for Claude to call out to external systems and tools. With MCP, Claude can decide on its own when to search your books, formulate appropriate search queries, and incorporate the results into its responses without you needing to manually orchestrate any of this.

---

## Revised Local-First Implementation (Recommended)

This section provides the complete implementation using the recommended local-first stack: LlamaIndex + LanceDB + Ollama + pymupdf4llm.

### Prerequisites

Before starting, ensure you have:
- Python 3.11+ (via pyenv or system)
- Ollama installed (`brew install ollama` on macOS)
- ~10GB disk space for models and data

### Phase 1: Environment Setup

#### 1.1 Install Ollama and Models

```bash
# Install Ollama (if not already installed)
brew install ollama

# Start Ollama service
ollama serve &

# Pull required models
ollama pull nomic-embed-text      # Embeddings (274MB)
ollama pull bge-reranker-v2-m3    # Reranking (568MB) - optional but recommended
```

#### 1.2 Project Structure

```bash
mkdir -p ~/projects/colibri-rag
cd ~/projects/colibri-rag

# Create project structure
mkdir -p src data/books data/lancedb tests
```

#### 1.3 Python Environment

```bash
cd ~/projects/colibri-rag

# Using uv (recommended) or pip
uv init
uv add llama-index llama-index-vector-stores-lancedb llama-index-embeddings-ollama
uv add llama-index-llms-ollama llama-index-readers-file
uv add pymupdf4llm lancedb
uv add mcp click rich

# Or with pip:
# pip install llama-index llama-index-vector-stores-lancedb ...
```

**pyproject.toml** dependencies:
```toml
[project]
name = "colibri-rag"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "llama-index>=0.12.0",
    "llama-index-vector-stores-lancedb>=0.4.0",
    "llama-index-embeddings-ollama>=0.4.0",
    "llama-index-llms-ollama>=0.4.0",
    "llama-index-readers-file>=0.4.0",
    "pymupdf4llm>=0.0.17",
    "lancedb>=0.17.0",
    "mcp>=1.0.0",
    "click>=8.1.0",
    "rich>=13.0.0",
]
```

### Phase 2: Core RAG Implementation

#### 2.1 Configuration (`src/config.py`)

```python
"""Configuration for the local RAG system."""
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
BOOKS_DIR = DATA_DIR / "books"
LANCEDB_DIR = DATA_DIR / "lancedb"

# Ollama settings
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"
RERANKER_MODEL = "bge-reranker-v2-m3"  # Optional

# Chunking settings
CHUNK_SIZE = 1024  # tokens
CHUNK_OVERLAP = 128  # tokens

# Retrieval settings
TOP_K = 10  # Initial retrieval
RERANK_TOP_K = 5  # After reranking
SIMILARITY_THRESHOLD = 0.3

# Ensure directories exist
BOOKS_DIR.mkdir(parents=True, exist_ok=True)
LANCEDB_DIR.mkdir(parents=True, exist_ok=True)
```

#### 2.2 Document Ingestion (`src/ingest.py`)

```python
#!/usr/bin/env python3
"""Document ingestion pipeline using pymupdf4llm and LlamaIndex."""
import pymupdf4llm
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding

from config import (
    LANCEDB_DIR, EMBEDDING_MODEL, OLLAMA_BASE_URL,
    CHUNK_SIZE, CHUNK_OVERLAP
)

console = Console()


def extract_pdf_to_markdown(pdf_path: Path) -> str:
    """Extract PDF content to clean Markdown using pymupdf4llm."""
    md_text = pymupdf4llm.to_markdown(
        str(pdf_path),
        page_chunks=False,  # Single document
        write_images=False,  # Skip images for now
        show_progress=True,
    )
    return md_text


def create_index() -> VectorStoreIndex:
    """Create or connect to the LanceDB-backed vector index."""
    embed_model = OllamaEmbedding(
        model_name=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )

    vector_store = LanceDBVectorStore(
        uri=str(LANCEDB_DIR),
        table_name="technical_books",
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Try to load existing index, or create new
    try:
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model,
        )
        console.print("[dim]Connected to existing index[/dim]")
    except Exception:
        index = VectorStoreIndex(
            nodes=[],
            storage_context=storage_context,
            embed_model=embed_model,
        )
        console.print("[dim]Created new index[/dim]")

    return index


def ingest_pdf(pdf_path: Path, book_title: str | None = None) -> int:
    """
    Ingest a PDF book into the RAG system.

    Returns the number of chunks created.
    """
    if book_title is None:
        book_title = pdf_path.stem

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Extract PDF to Markdown
        task = progress.add_task(f"Extracting {pdf_path.name}...", total=None)
        md_content = extract_pdf_to_markdown(pdf_path)

        # Create LlamaIndex document with metadata
        progress.update(task, description="Creating document...")
        doc = Document(
            text=md_content,
            metadata={
                "source": book_title,
                "source_file": pdf_path.name,
                "file_path": str(pdf_path),
                "type": "technical_book",
            },
        )

        # Parse into chunks
        progress.update(task, description="Chunking document...")
        parser = SentenceSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        nodes = parser.get_nodes_from_documents([doc])

        # Add chunk metadata
        for i, node in enumerate(nodes):
            node.metadata["chunk_index"] = i
            node.metadata["total_chunks"] = len(nodes)

        # Index chunks
        progress.update(task, description="Indexing chunks...")
        index = create_index()
        index.insert_nodes(nodes)

    return len(nodes)


if __name__ == "__main__":
    import click

    @click.command()
    @click.argument("pdf_path", type=click.Path(exists=True, path_type=Path))
    @click.option("--title", help="Book title (defaults to filename)")
    def main(pdf_path: Path, title: str | None = None):
        """Ingest a PDF book into the RAG system."""
        try:
            num_chunks = ingest_pdf(pdf_path, title)
            console.print(f"[green]✓ Ingested {pdf_path.name}[/green]")
            console.print(f"[cyan]  Created {num_chunks} chunks[/cyan]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise

    main()
```

#### 2.3 Query Engine (`src/query.py`)

```python
"""Query engine with hybrid retrieval and optional reranking."""
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding

from config import (
    LANCEDB_DIR, EMBEDDING_MODEL, OLLAMA_BASE_URL,
    TOP_K, SIMILARITY_THRESHOLD
)


class BookSearchEngine:
    """Search engine for technical books with semantic retrieval."""

    def __init__(self):
        # Configure embedding model
        self.embed_model = OllamaEmbedding(
            model_name=EMBEDDING_MODEL,
            base_url=OLLAMA_BASE_URL,
        )
        Settings.embed_model = self.embed_model

        # Connect to vector store
        self.vector_store = LanceDBVectorStore(
            uri=str(LANCEDB_DIR),
            table_name="technical_books",
        )

        # Create index from existing store
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            embed_model=self.embed_model,
        )

        # Create retriever
        self.retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=TOP_K,
        )

    def search(self, query: str, limit: int = 5) -> list[dict]:
        """
        Search for relevant book passages.

        Returns list of results with text, source, and score.
        """
        # Retrieve nodes
        nodes = self.retriever.retrieve(query)

        # Filter by similarity threshold and limit
        results = []
        for node in nodes[:limit]:
            if node.score and node.score < SIMILARITY_THRESHOLD:
                continue

            results.append({
                "text": node.node.get_content(),
                "source": node.node.metadata.get("source", "Unknown"),
                "source_file": node.node.metadata.get("source_file", ""),
                "chunk_index": node.node.metadata.get("chunk_index", 0),
                "total_chunks": node.node.metadata.get("total_chunks", 0),
                "score": round(node.score, 4) if node.score else None,
            })

        return results


# Singleton instance for MCP server
_engine: BookSearchEngine | None = None


def get_engine() -> BookSearchEngine:
    """Get or create the search engine singleton."""
    global _engine
    if _engine is None:
        _engine = BookSearchEngine()
    return _engine
```

### Phase 3: MCP Server Implementation

#### 3.1 MCP Server (`src/mcp_server.py`)

```python
#!/usr/bin/env python3
"""
MCP Server for the local RAG system.

Exposes book search capabilities to Claude Code and Claude Desktop.
"""
import asyncio
import json
import logging
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from query import get_engine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag-books-mcp")

# Create MCP server
server = Server("rag-books")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """Advertise available tools to Claude."""
    return [
        Tool(
            name="search_books",
            description=(
                "Search through indexed technical books for relevant information. "
                "Use this when answering questions about software architecture, "
                "testing strategies, development practices, design patterns, or "
                "technical topics that might be covered in reference books. "
                "Returns relevant passages with source attribution."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Natural language description of the information needed. "
                            "Be specific. Examples: 'microservice testing strategies', "
                            "'event sourcing patterns', 'API versioning approaches'"
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results to return (default: 5, max: 10)",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 10,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="list_indexed_books",
            description="List all books that have been indexed and are searchable.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool invocations from Claude."""

    if name == "search_books":
        return await handle_search(arguments)
    elif name == "list_indexed_books":
        return await handle_list_books()
    else:
        return [TextContent(
            type="text",
            text=json.dumps({"error": f"Unknown tool: {name}"})
        )]


async def handle_search(arguments: dict[str, Any]) -> list[TextContent]:
    """Handle search_books tool call."""
    query = arguments.get("query", "")
    limit = min(arguments.get("limit", 5), 10)

    if not query:
        return [TextContent(
            type="text",
            text=json.dumps({"error": "No query provided", "results": []})
        )]

    try:
        engine = get_engine()
        results = engine.search(query, limit=limit)

        return [TextContent(
            type="text",
            text=json.dumps({
                "query": query,
                "total_results": len(results),
                "results": results,
            }, indent=2)
        )]
    except Exception as e:
        logger.exception("Search failed")
        return [TextContent(
            type="text",
            text=json.dumps({"error": f"Search failed: {str(e)}", "results": []})
        )]


async def handle_list_books() -> list[TextContent]:
    """Handle list_indexed_books tool call."""
    try:
        engine = get_engine()
        # Query LanceDB directly for unique sources
        table = engine.vector_store._connection.open_table("technical_books")
        df = table.to_pandas()

        books = df["source"].unique().tolist() if "source" in df.columns else []

        return [TextContent(
            type="text",
            text=json.dumps({
                "indexed_books": books,
                "total_books": len(books),
            }, indent=2)
        )]
    except Exception as e:
        logger.exception("List books failed")
        return [TextContent(
            type="text",
            text=json.dumps({"error": str(e), "indexed_books": []})
        )]


async def main():
    """Run the MCP server."""
    logger.info("Starting RAG Books MCP server...")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
```

### Phase 4: Claude Integration

#### 4.1 Claude Code Configuration

Add to `~/.claude/claude_desktop_config.json` (create if doesn't exist):

```json
{
  "mcpServers": {
    "rag-books": {
      "command": "/Users/tobias/projects/colibri-rag/.venv/bin/python",
      "args": ["/Users/tobias/projects/colibri-rag/src/mcp_server.py"],
      "env": {
        "PYTHONPATH": "/Users/tobias/projects/colibri-rag/src"
      }
    }
  }
}
```

> **Note:** Adjust paths to match your actual installation. Use `uv run which python` to find the correct Python path if using uv.

#### 4.2 Testing the Integration

```bash
# 1. Ensure Ollama is running
ollama serve &

# 2. Ingest a test book
cd ~/projects/colibri-rag
uv run python src/ingest.py /path/to/your/book.pdf --title "My Technical Book"

# 3. Test search directly
uv run python -c "
from src.query import get_engine
engine = get_engine()
results = engine.search('software architecture patterns')
for r in results:
    print(f'{r[\"source\"]}: {r[\"text\"][:100]}...')
"

# 4. Restart Claude Code/Desktop to load MCP server
```

### Phase 5: CLI Tools

#### 5.1 Management CLI (`src/cli.py`)

```python
#!/usr/bin/env python3
"""CLI for managing the RAG system."""
import click
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
def cli():
    """CoLibri RAG - Local technical book search."""
    pass


@cli.command()
@click.argument("pdf_path", type=click.Path(exists=True, path_type=Path))
@click.option("--title", help="Book title (defaults to filename)")
def ingest(pdf_path: Path, title: str | None):
    """Ingest a PDF book into the index."""
    from ingest import ingest_pdf
    num_chunks = ingest_pdf(pdf_path, title)
    console.print(f"[green]✓ Ingested {num_chunks} chunks[/green]")


@cli.command()
@click.argument("query")
@click.option("--limit", "-n", default=5, help="Number of results")
def search(query: str, limit: int):
    """Search the book index."""
    from query import get_engine

    engine = get_engine()
    results = engine.search(query, limit=limit)

    if not results:
        console.print("[yellow]No results found[/yellow]")
        return

    for i, r in enumerate(results, 1):
        console.print(f"\n[bold cyan]Result {i}[/bold cyan] (score: {r['score']})")
        console.print(f"[dim]Source: {r['source']} (chunk {r['chunk_index']+1}/{r['total_chunks']})[/dim]")
        console.print(r["text"][:500] + "..." if len(r["text"]) > 500 else r["text"])


@cli.command()
def books():
    """List indexed books."""
    from query import get_engine

    engine = get_engine()
    table = engine.vector_store._connection.open_table("technical_books")
    df = table.to_pandas()

    if df.empty:
        console.print("[yellow]No books indexed yet[/yellow]")
        return

    # Group by source
    book_stats = df.groupby("source").size().reset_index(name="chunks")

    t = Table(title="Indexed Books")
    t.add_column("Title", style="cyan")
    t.add_column("Chunks", justify="right")

    for _, row in book_stats.iterrows():
        t.add_row(row["source"], str(row["chunks"]))

    console.print(t)


@cli.command()
def status():
    """Check system status."""
    import httpx

    # Check Ollama
    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        console.print(f"[green]✓ Ollama running[/green] - Models: {', '.join(models)}")
    except Exception:
        console.print("[red]✗ Ollama not running[/red] - Start with: ollama serve")

    # Check LanceDB
    from config import LANCEDB_DIR
    if LANCEDB_DIR.exists() and any(LANCEDB_DIR.iterdir()):
        console.print(f"[green]✓ LanceDB initialized[/green] at {LANCEDB_DIR}")
    else:
        console.print(f"[yellow]○ LanceDB empty[/yellow] - Ingest books to populate")


if __name__ == "__main__":
    cli()
```

### Validation Checklist

Before proceeding to implementation, verify:

- [ ] Ollama installed and running (`ollama serve`)
- [ ] Embedding model pulled (`ollama pull nomic-embed-text`)
- [ ] Python 3.11+ available
- [ ] Project directory created with structure above
- [ ] All dependencies installed via uv/pip
- [ ] At least one PDF book available for testing

### Expected Resource Usage

| Component | Memory | Disk | CPU |
|-----------|--------|------|-----|
| Ollama (nomic-embed-text) | ~1GB | 274MB | Low |
| LanceDB | ~100MB | Scales with data | Low |
| Python process | ~200MB | - | Low during search |
| PDF ingestion | ~500MB peak | - | Moderate |

**Total:** ~2GB RAM during operation, ~1GB idle.

---

## ~~Recommended Approach: Mem0 Implementation~~ (See Revised Approach Above)

> **⚠️ TECHNICAL REVIEW:** Mem0 has significant limitations for a local-first approach:
> 1. **Cloud-first architecture**: Mem0 (mem0.ai) is primarily a hosted service. The open-source version has limited features and documentation for fully local operation.
> 2. **LLM dependency**: Mem0 often requires an LLM for memory extraction/summarization, adding complexity.
> 3. **Less active OSS development**: Community contributions focus on the cloud platform.
>
> **Recommendation:** Skip to the **Revised Local-First Implementation** section below for a better approach using LlamaIndex + LanceDB.

Mem0 is a specialized memory layer for AI applications that provides document ingestion, semantic search, and memory management capabilities. It's particularly well-suited for your use case because it handles the complexity of document processing while remaining flexible enough to run entirely locally on your machine.

### Why Mem0 Makes Sense for Your Use Case

Mem0 was designed specifically to solve the problem of giving AI assistants access to persistent knowledge bases. Unlike general-purpose frameworks that require you to assemble various components, Mem0 provides an opinionated but flexible architecture that handles document processing, chunking, embedding generation, and retrieval as a cohesive system. For someone who wants to get a production-quality RAG system running without becoming an expert in every component, this is valuable.

The library has strong support for local operation, meaning you can run everything on your Mac without sending data to external services. It supports ChromaDB as a backend, which you can run in Docker just as we discussed earlier. The embedding generation can use local models through the sentence-transformers library, so you maintain complete control over your data and have no ongoing API costs.

Mem0 also provides features that go beyond basic RAG, such as conversation memory and the ability to associate memories with specific users or contexts. While you may not need all of these features immediately, they become useful as your usage evolves. For example, you might eventually want to store not just book content but also your own notes and insights, creating a personal knowledge graph that combines reference material with your accumulated expertise.

### Architecture Overview

The Mem0-based system will have several key components working together. At the foundation is ChromaDB running in a Docker container, which stores the vector embeddings and provides fast similarity search. Above that sits Mem0 itself, which manages document processing, chunking, embedding generation, and retrieval logic. On top of Mem0 is the MCP server, which translates between Mem0's Python API and the MCP protocol that Claude understands. Finally, Claude Code and Claude Desktop connect to the MCP server, allowing Claude to search your books as part of its normal operation.

The flow of data through this system follows a clear pattern. When you add a new book, Mem0 extracts the text from the PDF, splits it into semantically meaningful chunks, generates embeddings for each chunk using a local model, and stores everything in ChromaDB along with metadata about which book and section each chunk came from. When Claude needs to search your books, it sends a query through the MCP server to Mem0, which generates an embedding for the query, searches ChromaDB for similar vectors, retrieves the corresponding text chunks, and returns them to Claude along with source information.

### Implementation Steps

The implementation follows a logical progression from infrastructure setup through testing and refinement. Each step builds on the previous one, allowing you to verify that each component works before moving to the next.

#### Phase 1: Infrastructure Setup

The first phase focuses on getting the foundational infrastructure running. This is where you'll set up the vector database and install the necessary Python packages.

Start by creating a dedicated project directory for your RAG system. A good location would be something like `~/projects/rag-books` to keep it organized with your other development work. Inside this directory, you'll create the Docker configuration, Python scripts, and documentation that make up the complete system.

Create a `docker-compose.yml` file that defines your ChromaDB service. The configuration should specify that ChromaDB data persists in a local directory so your indexed books survive container restarts. You'll want to expose ChromaDB on port 8000, which is its default port and what Mem0 expects. The important detail here is ensuring the volume mapping is correct so your vector database actually persists to disk rather than being lost when the container stops.

```yaml
version: '3.8'

services:
  chromadb:
    image: chromadb/chroma:latest
    container_name: rag-books-chromadb
    ports:
      - "8000:8000"
    volumes:
      - ./chroma_data:/chroma/chroma
    environment:
      - IS_PERSISTENT=TRUE
      - ANONYMIZED_TELEMETRY=FALSE
      - CHROMA_SERVER_AUTH_CREDENTIALS_PROVIDER=chromadb.auth.token.TokenConfigServerAuthCredentialsProvider
      - CHROMA_SERVER_AUTH_CREDENTIALS=test-token
      - CHROMA_SERVER_AUTH_PROVIDER=chromadb.auth.token.TokenAuthServerProvider
    restart: unless-stopped
```

Once you have the Docker configuration ready, start ChromaDB with `docker-compose up -d` and verify it's running by checking `docker ps`. You should see the chromadb container in a healthy state. You can also test the connection by visiting `http://localhost:8000/api/v1` in your browser, which should return a heartbeat response from ChromaDB.

Next, set up your Python environment. Create a virtual environment specifically for this project to keep dependencies isolated. You can do this with `python3 -m venv venv` followed by `source venv/bin/activate`. This ensures that the packages you install for the RAG system don't interfere with other Python projects you might have.

Install the required packages. You'll need Mem0, which brings in its dependencies including support for various vector databases and embedding models. You'll also need the MCP SDK for Python, which provides the protocol implementation for building MCP servers. Install Docling for high-quality PDF processing, and add some utilities like click for building CLI tools and rich for beautiful terminal output.

```bash
# NOTE: Package name corrected from 'pymcp' to 'mcp'
pip install mem0ai mcp docling sentence-transformers click rich chromadb
```

After installation, verify that everything is working by running a quick Python test. Start a Python interpreter and try importing the key libraries. If all imports succeed without errors, you have a working foundation to build on.

#### Phase 2: Mem0 Configuration

Now you'll configure Mem0 to work with your local ChromaDB instance and set up the processing pipeline for your PDF books.

Create a configuration file called `mem0_config.py` that defines how Mem0 should operate. This is where you specify the vector database connection, the embedding model to use, and various parameters that control how documents are processed and searched.

```python
from mem0 import Memory

# Configuration for Mem0 to use local ChromaDB
config = {
    "vector_store": {
        "provider": "chroma",
        "config": {
            "host": "localhost",
            "port": 8000,
            "collection_name": "technical_books"
        }
    },
    "embedder": {
        "provider": "huggingface",
        "config": {
            "model": "sentence-transformers/all-MiniLM-L6-v2"
        }
    },
    "version": "v1.1"
}

# Initialize Mem0 with this configuration
memory = Memory.from_config(config)
```

The configuration choices here are deliberate. ChromaDB as the vector store gives you local operation with good performance. The sentence-transformers model all-MiniLM-L6-v2 is small enough to run on a laptop CPU but produces quality embeddings for technical text. The model will be downloaded automatically the first time you use it, which might take a few minutes, but then it's cached locally for future use.

Test your Mem0 configuration by running a simple script that adds a test memory and then searches for it. This verifies that the connection to ChromaDB is working, embeddings are being generated correctly, and search is returning relevant results.

```python
# Test script to verify Mem0 setup
from mem0_config import memory

# Add a test memory
memory.add("Python is a high-level programming language known for its simplicity.", 
           user_id="test")

# Search for it
results = memory.search("programming languages", user_id="test")
print(results)
```

If this test succeeds, you have a working Mem0 installation that can store and retrieve information.

#### Phase 3: Document Ingestion Pipeline

With Mem0 configured, you need to build the pipeline that processes your PDF books and loads them into the system. This is where Docling comes in to handle the complexities of PDF extraction.

Create a script called `ingest_books.py` that handles the end-to-end process of taking a PDF file, extracting its content, and adding it to Mem0. The script needs to handle several steps. First, it extracts text from the PDF while preserving document structure. Second, it processes the text to create meaningful chunks that will serve as the basis for retrieval. Third, it adds each chunk to Mem0 along with metadata that identifies which book and section it came from.

```python
#!/usr/bin/env python3
"""
Document ingestion pipeline for technical PDF books.

This script processes PDF files using Docling for high-quality text extraction,
chunks the content intelligently, and adds it to the Mem0 memory system for
later retrieval through Claude.
"""

from docling.document_converter import DocumentConverter
from mem0_config import memory
from pathlib import Path
import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

def chunk_markdown(markdown_text: str, chunk_size: int = 2000, overlap: int = 200) -> list[dict]:
    """
    Chunk markdown text while trying to preserve section boundaries.
    
    This function splits text into chunks but attempts to keep sections together
    when possible. It's smarter than just splitting at character boundaries because
    it respects the document structure that Docling preserved.
    """
    # Split on headers first to identify natural boundaries
    sections = []
    current_section = ""
    
    for line in markdown_text.split('\n'):
        if line.startswith('#'):
            if current_section:
                sections.append(current_section)
            current_section = line + '\n'
        else:
            current_section += line + '\n'
    
    if current_section:
        sections.append(current_section)
    
    # Now chunk each section if it's too large
    chunks = []
    for section in sections:
        if len(section) <= chunk_size:
            chunks.append(section)
        else:
            # Section is too large, split it with overlap
            start = 0
            while start < len(section):
                end = start + chunk_size
                chunk = section[start:end]
                chunks.append(chunk)
                start = end - overlap
    
    return chunks

def ingest_pdf(pdf_path: Path, book_title: str = None) -> int:
    """
    Ingest a PDF book into the Mem0 system.
    
    Returns the number of chunks created.
    """
    if book_title is None:
        book_title = pdf_path.stem
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        task = progress.add_task(f"Processing {pdf_path.name}...", total=None)
        
        # Extract text using Docling
        converter = DocumentConverter()
        result = converter.convert(str(pdf_path))
        markdown_text = result.document.export_to_markdown()
        
        progress.update(task, description="Chunking text...")
        chunks = chunk_markdown(markdown_text)
        
        progress.update(task, description="Adding to Mem0...")
        
        # Add each chunk to Mem0 with metadata
        for i, chunk in enumerate(chunks):
            memory.add(
                chunk,
                user_id="tobias",
                metadata={
                    "source": book_title,
                    "source_file": pdf_path.name,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "type": "book_content"
                }
            )
        
    return len(chunks)

@click.command()
@click.argument('pdf_path', type=click.Path(exists=True))
@click.option('--title', help='Book title (defaults to filename)')
def main(pdf_path: str, title: str = None):
    """Ingest a PDF book into the RAG system."""
    pdf_path = Path(pdf_path)
    
    try:
        num_chunks = ingest_pdf(pdf_path, title)
        console.print(f"[green]Successfully ingested {pdf_path.name}[/green]")
        console.print(f"[cyan]Created {num_chunks} chunks[/cyan]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise

if __name__ == '__main__':
    main()
```

Make this script executable and test it with one of your PDF books. Run `python ingest_books.py path/to/your/book.pdf` and watch as it processes the document. The first run will be slower as models are downloaded and cached, but subsequent runs should be faster.

#### Phase 4: MCP Server Implementation

Now comes the critical piece that connects everything to Claude. You'll build an MCP server that exposes your Mem0-based RAG system as a tool that Claude can call.

Create a file called `rag_mcp_server.py` that implements the MCP protocol and wraps your Mem0 memory system.

```python
#!/usr/bin/env python3
"""
MCP Server for Mem0-based RAG system.

This server exposes the technical books knowledge base to Claude through
the Model Context Protocol, allowing Claude to autonomously search for
relevant information when answering questions.
"""

import asyncio
import json
from typing import Any
from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.server.stdio import stdio_server
from mcp import types
from mem0_config import memory

# Create the MCP server instance
server = Server("rag-books")

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """
    Advertise available tools to Claude.
    
    This tells Claude what capabilities are available and how to use them.
    """
    return [
        types.Tool(
            name="search_books",
            description=(
                "Search through indexed technical books for relevant information. "
                "Use this when the user asks questions about software architecture, "
                "testing strategies, development practices, design patterns, or any "
                "technical topic that might be covered in reference books. "
                "The search uses semantic similarity, so phrase your query as a "
                "natural description of what you're looking for rather than keywords."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "A clear, natural language description of what information "
                            "you're looking for. Examples: 'microservice testing strategies', "
                            "'event-driven architecture patterns', 'API design best practices'"
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of relevant chunks to return (default: 5)",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        )
    ]

@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict[str, Any]
) -> list[types.TextContent]:
    """
    Handle tool invocations from Claude.
    
    This is called when Claude decides to search the books.
    """
    if name != "search_books":
        raise ValueError(f"Unknown tool: {name}")
    
    query = arguments.get("query")
    limit = arguments.get("limit", 5)
    
    if not query:
        return [types.TextContent(
            type="text",
            text=json.dumps({"error": "No query provided", "results": []})
        )]
    
    try:
        # Search Mem0 for relevant chunks
        results = memory.search(
            query=query,
            user_id="tobias",
            limit=limit
        )
        
        # Format results for Claude
        formatted_results = []
        for result in results:
            formatted_results.append({
                "text": result.get("memory", ""),
                "source": result.get("metadata", {}).get("source", "Unknown"),
                "chunk_info": f"{result.get('metadata', {}).get('chunk_index', 0) + 1}/{result.get('metadata', {}).get('total_chunks', 0)}",
                "relevance_score": result.get("score", 0)
            })
        
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "query": query,
                "results": formatted_results,
                "total_results": len(formatted_results)
            }, indent=2)
        )]
        
    except Exception as e:
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "error": f"Search failed: {str(e)}",
                "results": []
            })
        )]

async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="rag-books",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())
```

Make this script executable with `chmod +x rag_mcp_server.py`.

#### Phase 5: Claude Integration

The final step is configuring Claude Code and Claude Desktop to use your MCP server. This involves editing configuration files that tell Claude about the available MCP servers.

For Claude Desktop and Claude Code, the configuration file is located at `~/Library/Application Support/Claude/claude_desktop_config.json` on macOS. If this file doesn't exist yet, create it. Add your MCP server to the configuration:

```json
{
  "mcpServers": {
    "rag-books": {
      "command": "/Users/tobias/projects/rag-books/venv/bin/python",
      "args": ["/Users/tobias/projects/rag-books/rag_mcp_server.py"]
    }
  }
}
```

Make sure to update the paths to match your actual project location. The command should point to the Python interpreter in your virtual environment, and the args should point to your MCP server script.

After updating the configuration, restart Claude Desktop or Claude Code for the changes to take effect. You can verify the MCP server is working by looking for it in the available tools or by asking Claude a question that would benefit from searching your books.

#### Phase 6: Testing and Validation

Once everything is configured, test the complete system end-to-end. Start a conversation with Claude and ask a question that should trigger a book search. For example, you might ask "What are the key principles of microservice architecture?" and observe whether Claude searches your books automatically.

Pay attention to the quality of the search results. Are the retrieved chunks relevant to the question? Does Claude cite the sources appropriately? If the results aren't quite right, you may need to adjust the chunking strategy or the number of results returned.

Test with different types of questions to understand when the RAG search is helpful and when it's not. Simple definitional questions might not need book context, while complex architectural questions will benefit from retrieving specific passages.

### Expected Workflow After Setup

Once your system is fully operational, your daily workflow becomes remarkably simple. You'll interact with Claude Code or Claude Desktop exactly as you normally would, asking questions in natural language. When you ask something like "How should I structure integration tests for microservices?", Claude will automatically decide whether searching your books would be helpful. If so, it will call your MCP server, which searches through Mem0, retrieves relevant chunks from your indexed books, and Claude incorporates that information into its answer, citing which books the information came from.

You won't need to manually search for context or copy text from PDFs. The system works transparently in the background, making your technical reference library feel like an extension of Claude's knowledge. When you add new books, you simply run the ingestion script, and they immediately become available for Claude to search.

---

## Alternative Implementation Approaches

While Mem0 is the recommended approach, there are several alternative architectures worth documenting. Each has different tradeoffs in terms of setup complexity, flexibility, and ongoing maintenance. By documenting these alternatives now, you can benchmark them against the Mem0 implementation later if you want to optimize for different priorities.

### Alternative 1: Custom Implementation with ChromaDB

This is the approach we discussed in detail earlier. You would build your own document processing pipeline using Docling for PDF extraction, implement custom chunking logic, use sentence-transformers directly for embeddings, and write your own code to interact with ChromaDB. Then you'd wrap everything in an MCP server.

#### Advantages

The primary advantage is complete control and simplicity. You understand every piece of the system because you built it. There are no abstraction layers hiding complexity, which makes debugging straightforward. The code is optimized for exactly your use case with no extra features adding bloat. You can make changes quickly without needing to understand a framework's conventions or limitations.

For someone comfortable with Python who wants to deeply understand their tools, this approach provides excellent learning opportunities. You'll understand RAG systems at a fundamental level, not just how to configure someone else's implementation. This knowledge transfers well if you ever need to build similar systems or debug problems in production.

#### Disadvantages

The main disadvantage is that you're responsible for all maintenance and feature development. If you want to add new capabilities like automatic reindexing when documents change, or support for document types beyond PDFs, or more sophisticated chunking strategies, you have to implement those yourself. Bug fixes and updates to dependencies also become your responsibility.

You also miss out on optimizations and best practices that have been discovered by people building production RAG systems. Things like optimal chunking strategies for different document types, handling of edge cases in PDF processing, or efficient batching of embedding generation are all problems that have been solved by mature libraries but that you'd need to discover and implement yourself.

#### When This Makes Sense

This approach is best if you want maximum control and minimum dependencies, if you're building this partly as a learning exercise, or if you have very specific requirements that would be awkward to implement in a framework. It's also good if you expect to make frequent customizations and want to avoid fighting against framework opinions.

#### Implementation Effort

Initial implementation would take approximately 4-6 hours to build and test all the components. The complexity is moderate because you're working with well-documented libraries, but you need to understand how each piece fits together. Ongoing maintenance is light as long as your requirements don't change.

### Alternative 2: LangChain-Based Implementation

LangChain is a comprehensive framework for building applications with language models. It provides abstractions for document loading, text splitting, embedding generation, vector stores, and retrieval chains. You would use LangChain's components to build your RAG system and then expose it through an MCP server.

#### Advantages

LangChain gives you access to a vast ecosystem of pre-built components. It has loaders for dozens of document formats, sophisticated text splitters that understand document structure, integrations with multiple embedding providers and vector databases, and retrieval strategies that go beyond simple similarity search. If you want to experiment with different approaches, LangChain makes it easy to swap components without rewriting code.

The framework is extremely well-documented with extensive examples covering almost any use case you might imagine. The community is large and active, so you can find solutions to common problems and get help when you're stuck. LangChain is also widely used in production, so you benefit from battle-tested code that handles edge cases you might not think of.

LangChain's abstraction layers make it relatively easy to add advanced features like hybrid search combining keyword and semantic approaches, query routing that sends different types of questions to different retrieval strategies, or multi-step retrieval that can follow references and gather context from multiple documents.

#### Disadvantages

The main disadvantage is complexity and abstraction overhead. LangChain is a large framework with many concepts to learn, and sometimes its abstractions hide important details. When something goes wrong, debugging can be challenging because you need to understand both your code and LangChain's internals. The framework can feel heavyweight for simple use cases.

LangChain also has a reputation for frequent breaking changes as the API evolves. Code that works today might need updates when you upgrade to a newer version. The pace of development is fast, which brings new features but can make maintenance more demanding.

Performance can sometimes be an issue because LangChain prioritizes flexibility over optimization. For your use case with a relatively small corpus of technical books, this probably won't matter, but it's worth keeping in mind.

#### When This Makes Sense

LangChain is ideal if you want to experiment with different RAG strategies, if you need advanced retrieval capabilities beyond basic similarity search, or if you expect to expand your system significantly over time. It's also good if you value having extensive documentation and community support.

#### Implementation Effort

Initial implementation would take approximately 6-8 hours because you need to learn LangChain's concepts and conventions before you can be productive. However, once you understand the framework, adding new capabilities becomes faster than with a custom implementation. Ongoing maintenance is moderate, primarily driven by keeping up with API changes.

### Alternative 3: txtai Implementation

> **Reviewer Note:** The original assessment of txtai was somewhat unfair. txtai (GitHub: 9k+ stars) is actually more mature and actively maintained than suggested. It's a legitimate alternative to LlamaIndex for simpler use cases.

txtai is a Python library specifically designed for semantic search and RAG workflows. It provides an embeddings database that combines vector search with traditional SQL-style queries, a pipeline system for processing documents, and built-in support for question answering. While it doesn't have official MCP integration, wrapping it in an MCP server is straightforward.

#### Advantages

txtai is purpose-built for exactly your use case, which means it has sensible defaults and doesn't require much configuration to get good results. The API is clean and intuitive, with less conceptual overhead than LangChain. It supports both semantic search and SQL queries over the same index, which can be powerful when you want to combine similarity search with metadata filtering.

The library includes built-in PDF extraction, so you might not even need Docling depending on your requirements. It can handle document updates efficiently, reindexing only what's changed rather than rebuilding the entire index. txtai also has good support for local models, making it easy to run everything on your machine.

One unique feature is txtai's workflow system, which lets you chain together multiple operations like extraction, transformation, and indexing in a declarative configuration. This can make complex pipelines more maintainable than code.

**Additional strengths (review update):**
- Hybrid search (BM25 + semantic) is built-in and easy to enable
- Graph-based RAG support for connected knowledge
- Active development with regular releases
- Good Hugging Face integration for local models

#### Disadvantages

~~txtai is less widely used than LangChain or Mem0, which means a smaller community and fewer examples to learn from.~~ **Update:** txtai has 9k+ stars and an active community. Documentation is solid.

~~The library is primarily maintained by one person, which raises questions about long-term sustainability.~~ **Update:** While David Mezzetti is the primary maintainer, the project has consistent releases, good test coverage, and a healthy contributor base. This concern is overstated.

txtai's abstraction level sits between the custom approach and LangChain. It provides more structure than building from scratch but less flexibility than LangChain. For most use cases this is fine, but if you have unusual requirements, you might find yourself fighting the framework.

#### When This Makes Sense

txtai is a good choice if you want something more polished than a custom implementation but lighter weight than LangChain. It's particularly appealing if:
- You want the simplest possible setup
- You prefer declarative YAML configuration over code
- You want hybrid search without additional setup
- You don't need LlamaIndex's advanced features (agents, tool use, etc.)

#### Implementation Effort

Initial implementation would take approximately 3-4 hours - txtai's defaults are very sensible. Example:

```python
from txtai import Embeddings

# That's it - this gives you a working semantic search
embeddings = Embeddings(path="sentence-transformers/all-MiniLM-L6-v2")
embeddings.index([(0, "your document text", None)])
results = embeddings.search("query", 5)
```

#### txtai vs LlamaIndex

| Factor | txtai | LlamaIndex |
|--------|-------|------------|
| Setup complexity | Lower | Higher |
| Documentation | Good | Excellent |
| Flexibility | Moderate | High |
| Community size | Medium (9k stars) | Large (40k stars) |
| MCP examples | None | Some |
| Best for | Simple RAG | Complex RAG with agents |

### Alternative 4: Hybrid Approach Using Obsidian (Expanded)

> **Reviewer Note:** This approach has been significantly expanded based on user interest. It combines the best of both worlds: Obsidian's knowledge management with semantic RAG search.

Given that you already use Obsidian for documentation, this approach treats Obsidian as the source of truth while adding semantic search capabilities via MCP.

#### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Obsidian Vault                            │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  📁 Books/                                           │    │
│  │  ├── 📄 Clean Architecture.md                       │    │
│  │  ├── 📄 Domain-Driven Design.md                     │    │
│  │  └── 📄 Testing Microservices.md                    │    │
│  ├─────────────────────────────────────────────────────┤    │
│  │  📁 Notes/                                           │    │
│  │  ├── 📄 My Architecture Decisions.md                │    │
│  │  └── 📄 Testing Strategy Ideas.md                   │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ File system watch
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Indexing Service                          │
│  - Watches vault for changes                                 │
│  - Parses markdown, respects frontmatter                    │
│  - Generates embeddings via Ollama                          │
│  - Stores in LanceDB (beside vault or separate)             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    MCP Server                                │
│  - search_vault: Semantic search across all content         │
│  - search_books: Search only book content                   │
│  - get_linked_notes: Follow Obsidian [[links]]              │
│  - get_note: Retrieve specific note by path                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Claude (via MCP)                          │
└─────────────────────────────────────────────────────────────┘
```

#### Key Design Decisions

**1. Automated PDF → Markdown Conversion**

Instead of manual conversion, use `pymupdf4llm` to auto-convert PDFs:

```python
# src/pdf_to_vault.py
import pymupdf4llm
from pathlib import Path
from datetime import datetime

def pdf_to_obsidian(pdf_path: Path, vault_books_dir: Path) -> Path:
    """Convert PDF to Obsidian-ready markdown with frontmatter."""

    # Extract to markdown
    md_content = pymupdf4llm.to_markdown(str(pdf_path))

    # Add Obsidian frontmatter
    title = pdf_path.stem
    frontmatter = f"""---
title: "{title}"
type: book
source_pdf: "{pdf_path.name}"
imported: {datetime.now().isoformat()}
tags:
  - book
  - imported
---

# {title}

> [!info] Source
> Imported from `{pdf_path.name}` on {datetime.now().strftime('%Y-%m-%d')}

"""

    # Write to vault
    output_path = vault_books_dir / f"{title}.md"
    output_path.write_text(frontmatter + md_content)

    return output_path
```

**2. Vault Structure**

```
Your Obsidian Vault/
├── Books/                      # Auto-imported book content
│   ├── Clean Architecture.md
│   ├── Domain-Driven Design.md
│   └── ...
├── Book Notes/                 # Your annotations and summaries
│   ├── Clean Architecture - Notes.md
│   └── ...
├── .colibri/                   # Hidden folder for RAG data
│   ├── lancedb/               # Vector index
│   └── config.yaml            # Index configuration
└── Templates/
    └── Book Note.md           # Template for book annotations
```

**3. Frontmatter for Metadata**

Each book file uses YAML frontmatter that the indexer respects:

```markdown
---
title: "Clean Architecture"
author: "Robert C. Martin"
type: book
tags:
  - architecture
  - software-design
chapters:
  - "What is Architecture?"
  - "Independence"
  - "Boundaries"
---
```

**4. Incremental Indexing**

Watch for file changes and re-index only what changed:

```python
# src/vault_indexer.py
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import hashlib

class VaultIndexer(FileSystemEventHandler):
    def __init__(self, vault_path: Path, index):
        self.vault_path = vault_path
        self.index = index
        self.file_hashes = {}  # Track what's indexed

    def on_modified(self, event):
        if event.src_path.endswith('.md'):
            self.reindex_file(Path(event.src_path))

    def reindex_file(self, file_path: Path):
        content = file_path.read_text()
        content_hash = hashlib.md5(content.encode()).hexdigest()

        # Skip if unchanged
        if self.file_hashes.get(str(file_path)) == content_hash:
            return

        # Remove old chunks, add new ones
        self.index.delete_by_metadata({"source_file": str(file_path)})
        self.index.add_document(file_path, content)
        self.file_hashes[str(file_path)] = content_hash
```

#### Implementation: Obsidian MCP Server

```python
#!/usr/bin/env python3
"""MCP Server for Obsidian vault with semantic search."""
import asyncio
import json
import re
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.types import Tool, TextContent

from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.node_parser import MarkdownNodeParser
import frontmatter

# Configuration
VAULT_PATH = Path.home() / "Obsidian" / "YourVault"  # Adjust this
INDEX_PATH = VAULT_PATH / ".colibri" / "lancedb"
BOOKS_FOLDER = "Books"

server = Server("obsidian-rag")


class ObsidianSearchEngine:
    def __init__(self):
        Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

        self.vector_store = LanceDBVectorStore(
            uri=str(INDEX_PATH),
            table_name="vault_content",
        )
        self.index = VectorStoreIndex.from_vector_store(self.vector_store)
        self.retriever = self.index.as_retriever(similarity_top_k=10)

    def search(self, query: str, folder: str | None = None, limit: int = 5) -> list[dict]:
        """Search vault content with optional folder filter."""
        nodes = self.retriever.retrieve(query)

        results = []
        for node in nodes:
            # Filter by folder if specified
            source_file = node.node.metadata.get("source_file", "")
            if folder and folder not in source_file:
                continue

            results.append({
                "text": node.node.get_content(),
                "file": source_file,
                "title": node.node.metadata.get("title", Path(source_file).stem),
                "type": node.node.metadata.get("type", "note"),
                "score": round(node.score, 4) if node.score else None,
            })

            if len(results) >= limit:
                break

        return results

    def get_note(self, note_path: str) -> dict | None:
        """Get a specific note by path."""
        full_path = VAULT_PATH / note_path
        if not full_path.exists():
            return None

        post = frontmatter.load(full_path)
        return {
            "path": note_path,
            "title": post.metadata.get("title", full_path.stem),
            "metadata": dict(post.metadata),
            "content": post.content,
        }

    def get_linked_notes(self, note_path: str) -> list[str]:
        """Extract [[wiki links]] from a note."""
        full_path = VAULT_PATH / note_path
        if not full_path.exists():
            return []

        content = full_path.read_text()
        # Match [[link]] and [[link|alias]]
        links = re.findall(r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]', content)
        return links


engine: ObsidianSearchEngine | None = None


def get_engine() -> ObsidianSearchEngine:
    global engine
    if engine is None:
        engine = ObsidianSearchEngine()
    return engine


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_vault",
            description=(
                "Search the entire Obsidian vault for relevant content. "
                "Searches both imported books and personal notes. "
                "Use for broad knowledge queries."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for"},
                    "limit": {"type": "integer", "default": 5, "maximum": 10},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="search_books",
            description=(
                "Search only the imported technical books in the vault. "
                "Use when specifically looking for book/reference content."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for"},
                    "limit": {"type": "integer", "default": 5, "maximum": 10},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="get_note",
            description="Retrieve a specific note by its path in the vault.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to vault root, e.g., 'Books/Clean Architecture.md'"},
                },
                "required": ["path"],
            },
        ),
        Tool(
            name="get_linked_notes",
            description="Get all notes linked from a specific note via [[wiki links]].",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the note"},
                },
                "required": ["path"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    eng = get_engine()

    if name == "search_vault":
        results = eng.search(arguments["query"], limit=arguments.get("limit", 5))
        return [TextContent(type="text", text=json.dumps({"results": results}, indent=2))]

    elif name == "search_books":
        results = eng.search(
            arguments["query"],
            folder=BOOKS_FOLDER,
            limit=arguments.get("limit", 5)
        )
        return [TextContent(type="text", text=json.dumps({"results": results}, indent=2))]

    elif name == "get_note":
        note = eng.get_note(arguments["path"])
        if note:
            return [TextContent(type="text", text=json.dumps(note, indent=2))]
        return [TextContent(type="text", text=json.dumps({"error": "Note not found"}))]

    elif name == "get_linked_notes":
        links = eng.get_linked_notes(arguments["path"])
        return [TextContent(type="text", text=json.dumps({"links": links}))]

    return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]


async def main():
    from mcp.server.stdio import stdio_server
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
```

#### Advantages (Expanded)

**Knowledge Graph Benefits:**
- Obsidian's graph view shows connections between books and your notes
- You can link concepts across books: `[[Clean Architecture#Boundaries]]`
- Backlinks show where concepts are referenced
- Tags enable topic-based organization

**Annotation Workflow:**
- Read imported book markdown, add your own notes inline
- Create companion "Book Notes" files with summaries and insights
- Use callouts for your commentary:
  ```markdown
  > [!note] My Observation
  > This connects to the testing pyramid concept from [[Testing Microservices]]
  ```

**Version Control:**
- Git-track your vault to see knowledge evolution
- Diff shows exactly what you've learned/added
- Easy backup and sync across machines

**Dual Search:**
- Obsidian's built-in search for quick lookups
- Semantic search via Claude for conceptual queries
- Best of both worlds

#### Disadvantages (Revised)

~~This approach requires significant manual work to convert PDF books into well-structured markdown.~~ **Update:** With `pymupdf4llm`, conversion is automated. Manual work is optional curation.

**Actual challenges:**
- Initial indexing of large vaults takes time
- Need to manage two search systems (Obsidian + semantic)
- Vault size grows with book content (~1-5MB per book)
- Some PDF formatting may not convert perfectly

#### When This Makes Sense

This approach is ideal if you:
- Already use Obsidian and want unified knowledge management
- Value being able to annotate and link book content
- Want human-readable, version-controlled knowledge
- Appreciate the graph visualization of concept connections
- Plan to build on top of book knowledge with your own notes

#### Implementation Effort (Revised)

| Phase | Effort | Description |
|-------|--------|-------------|
| Initial setup | 2-3 hours | Install dependencies, configure vault structure |
| PDF import pipeline | 1-2 hours | Set up automated conversion script |
| Indexer service | 2-3 hours | Build watching/indexing service |
| MCP server | 2 hours | Implement search tools |
| First book import | 30 min | Test end-to-end flow |
| **Total** | **8-10 hours** | Full working system |

**Ongoing:** ~15 min per new book (run import, quick review)

#### Obsidian Plugins to Consider

| Plugin | Purpose | Recommended? |
|--------|---------|--------------|
| Dataview | Query notes as database | Yes - powerful metadata queries |
| Templater | Book note templates | Yes - standardize annotations |
| Excalidraw | Visual diagrams | Optional - for architecture sketches |
| Git | Version control | Yes - track knowledge evolution |
| Smart Connections | Built-in semantic search | Maybe - but our MCP is better integrated |

---

## Benchmarking Framework

To compare these approaches systematically, you should evaluate them across several dimensions. Each dimension captures a different aspect of what makes a RAG system effective for your specific use case and workflow.

### Retrieval Quality

This measures how well the system finds relevant information when you ask questions. Key metrics include precision (are the returned chunks actually relevant?), recall (does it find all the relevant information that exists?), and ranking quality (are the most relevant results at the top?).

To benchmark retrieval quality, create a test set of 20-30 questions that you know should be answerable from your books. For each question, note which books and sections contain the answer. Then run each question through each system and evaluate whether it retrieved the correct content. You can score each result on a simple scale: perfect (all relevant content, no irrelevant), good (mostly relevant content), acceptable (some relevant content), or poor (little or no relevant content).

### Response Time

This measures how long it takes from when you ask a question until Claude has the retrieved context and can start answering. Response time impacts your flow when working, especially if you're asking multiple questions in rapid succession.

To benchmark response time, measure the end-to-end latency for your test questions. Use a simple timer to record how long each search takes. Run each test multiple times to account for variability, and report both the median and 95th percentile latency. The first query after startup may be slower as models load into memory, so test both cold and warm performance.

### Setup Complexity

This measures how difficult it is to get the system running from scratch. It includes both the initial time investment and the level of technical knowledge required.

Rate setup complexity on a subjective scale based on your experience: trivial (under 1 hour, no troubleshooting needed), easy (1-2 hours, minimal troubleshooting), moderate (half day, some troubleshooting required), complex (full day, significant troubleshooting), or very complex (multiple days, extensive troubleshooting).

### Maintenance Burden

This measures how much ongoing work is required to keep the system running well. Maintenance includes updating dependencies, handling breaking changes, debugging issues, and adapting to new requirements.

Estimate maintenance burden by tracking how much time you spend on the system over a month after initial setup. Consider both planned maintenance like dependency updates and unplanned work like fixing issues. Rate maintenance as minimal (under 30 minutes per month), light (30-60 minutes per month), moderate (1-2 hours per month), or heavy (over 2 hours per month).

### Flexibility and Extensibility

This measures how easy it is to add new features or adapt the system to changing requirements. Can you easily switch embedding models, add new document types, or implement advanced retrieval strategies?

Evaluate flexibility by attempting to make a few common modifications: switching to a different embedding model, adding support for a new document format, implementing metadata filtering in search, or adding a hybrid search combining keywords and semantic similarity. Rate how difficult each modification is and how much code you need to change.

### Token Efficiency

This measures how well the system minimizes token usage while still providing sufficient context. Since Claude's API costs scale with tokens, and since long contexts can slow down responses, optimizing token efficiency matters.

To benchmark token efficiency, count how many tokens are typically included in the retrieved context for your test questions. Compare this against a naive approach of including entire document sections. Calculate the ratio of tokens used to information density (relevant facts per 1000 tokens).

### Integration Quality

This measures how seamlessly the system integrates with your workflow in Claude Code and Claude Desktop. Does it feel natural to use, or do you find yourself working around limitations?

Evaluate integration quality through actual usage over a week. Note friction points where the system doesn't work as you'd expect, moments where it delights you by working perfectly, and places where you wish it behaved differently. This is subjective but captures important aspects of user experience that metrics can't measure.

### Benchmark Template

For each implementation approach, create a benchmark report with this structure:

```
# RAG System Benchmark: [Approach Name]

## Test Date
[Date of testing]

## Retrieval Quality
- Test questions used: [number]
- Perfect results: [number and percentage]
- Good results: [number and percentage]
- Acceptable results: [number and percentage]
- Poor results: [number and percentage]
- Notable failures: [description of any significant problems]

## Response Time
- Cold start median: [time in seconds]
- Cold start 95th percentile: [time in seconds]
- Warm median: [time in seconds]
- Warm 95th percentile: [time in seconds]

## Setup Complexity
- Time spent: [hours]
- Complexity rating: [rating]
- Main challenges encountered: [description]

## Maintenance Burden
- Monthly time estimate: [minutes]
- Burden rating: [rating]
- Anticipated maintenance needs: [description]

## Flexibility and Extensibility
- Embedding model swap: [difficulty rating]
- New document format: [difficulty rating]
- Metadata filtering: [difficulty rating]
- Hybrid search: [difficulty rating]

## Token Efficiency
- Average tokens per query: [number]
- Compared to naive approach: [percentage]
- Information density: [relevant facts per 1000 tokens]

## Integration Quality
- Friction points: [description]
- Positive experiences: [description]
- Improvement opportunities: [description]

## Overall Assessment
[Summary paragraph about strengths and weaknesses of this approach]

## Recommendation
[Would you recommend this approach? For what use cases?]
```

---

## Decision Matrix (Revised)

When choosing between approaches, consider these factors:

| Priority | Recommended Approach | Why |
|----------|---------------------|-----|
| **Knowledge management focus** | **Obsidian Hybrid** ⭐ | Unified vault, annotations, graph view |
| **Local-first, pure RAG** | LlamaIndex + LanceDB + Ollama | Best balance of maturity and locality |
| **Minimal dependencies** | Custom + LanceDB | Fewest moving parts, most control |
| **Maximum flexibility** | LangChain | Most options, but heavier |
| **Simplest setup** | txtai | Single library does everything |
| **Cloud-OK, managed** | Mem0 | If you don't mind cloud dependencies |

### Two Recommended Paths

#### Path A: Obsidian Hybrid (Recommended if you use Obsidian)

**Best for:** Users who want knowledge management, not just search.

```
Obsidian Vault  →  Auto-import PDFs  →  LanceDB index  →  MCP Server  →  Claude
     ↓
Your annotations, links, notes all searchable together
```

**Unique benefits:**
- Books become part of your knowledge graph
- Annotate, link, and build on top of book content
- Visual graph shows concept connections
- Human-readable, version-controlled, portable
- Single source of truth for all knowledge

**Choose this if:** You already use Obsidian, value curation, or want to build lasting knowledge.

---

#### Path B: LlamaIndex + LanceDB (Recommended for pure RAG)

**Best for:** Users who want search without knowledge management overhead.

```
PDFs  →  pymupdf4llm  →  LlamaIndex  →  LanceDB  →  MCP Server  →  Claude
```

**Unique benefits:**
- Simpler architecture, fewer concepts
- Better for large book collections (100+ books)
- More advanced retrieval options (hybrid search, reranking)
- Slightly faster search performance

**Choose this if:** You want fast, reliable search without additional workflow.

---

### Comparison

| Factor | Obsidian Hybrid | LlamaIndex Pure |
|--------|-----------------|-----------------|
| Setup effort | Medium (8-10h) | Low (4-6h) |
| Ongoing value | High (compound knowledge) | Medium (search only) |
| Book annotation | Native | Not supported |
| Knowledge graph | Yes (Obsidian) | No |
| Scalability | Good (<500 books) | Excellent |
| Search quality | Good | Excellent |
| Personal notes integration | Native | Separate system |

**Avoid Mem0 for local-first** because:
- Primary value proposition is their cloud platform
- Open-source version has limited documentation for fully local operation
- Requires LLM for memory extraction, adding complexity
- Less community activity on local use cases

---

## Next Steps (Revised)

### Option A: Obsidian Hybrid Implementation

Since you're interested in the Obsidian approach, here's the focused implementation plan:

#### Day 1: Vault Setup

```bash
# 1. Install Ollama (if not already)
brew install ollama
ollama serve &
ollama pull nomic-embed-text

# 2. Create project for indexer/MCP server
mkdir -p ~/projects/colibri-obsidian/{src,tests}
cd ~/projects/colibri-obsidian

# 3. Set up vault structure (in your existing Obsidian vault)
mkdir -p "/path/to/your/vault/Books"
mkdir -p "/path/to/your/vault/Book Notes"
mkdir -p "/path/to/your/vault/.colibri/lancedb"

# 4. Install dependencies
uv init
uv add llama-index llama-index-vector-stores-lancedb \
       llama-index-embeddings-ollama pymupdf4llm lancedb \
       mcp python-frontmatter watchdog click rich
```

#### Day 2: PDF Import Pipeline

Create `src/pdf_to_vault.py`:
```python
#!/usr/bin/env python3
"""Import PDFs into Obsidian vault as markdown."""
import click
from pathlib import Path
from datetime import datetime
import pymupdf4llm
from rich.console import Console

console = Console()

@click.command()
@click.argument('pdf_path', type=click.Path(exists=True, path_type=Path))
@click.argument('vault_path', type=click.Path(exists=True, path_type=Path))
@click.option('--title', help='Override book title')
def import_pdf(pdf_path: Path, vault_path: Path, title: str | None):
    """Import a PDF book into the Obsidian vault."""
    books_dir = vault_path / "Books"
    books_dir.mkdir(exist_ok=True)

    title = title or pdf_path.stem
    console.print(f"[cyan]Importing: {title}[/cyan]")

    # Convert PDF to markdown
    md_content = pymupdf4llm.to_markdown(str(pdf_path))

    # Create Obsidian-formatted file with frontmatter
    output = f"""---
title: "{title}"
type: book
source_pdf: "{pdf_path.name}"
imported: "{datetime.now().isoformat()}"
tags:
  - book
  - imported
---

# {title}

> [!info] Source
> Imported from `{pdf_path.name}` on {datetime.now().strftime('%Y-%m-%d')}

{md_content}
"""

    output_path = books_dir / f"{title}.md"
    output_path.write_text(output)
    console.print(f"[green]✓ Created: {output_path}[/green]")

if __name__ == "__main__":
    import_pdf()
```

Test with one book:
```bash
uv run python src/pdf_to_vault.py /path/to/book.pdf /path/to/vault
```

#### Day 3: Vault Indexer

Create `src/indexer.py` to build the semantic index from vault content:
```python
#!/usr/bin/env python3
"""Index Obsidian vault for semantic search."""
from pathlib import Path
import frontmatter
from rich.console import Console
from rich.progress import track

from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding

console = Console()

def index_vault(vault_path: Path, folders: list[str] | None = None):
    """Index markdown files from the vault."""
    folders = folders or ["Books", "Book Notes", "Notes"]

    # Configure embedding
    Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

    # Set up vector store
    index_path = vault_path / ".colibri" / "lancedb"
    index_path.mkdir(parents=True, exist_ok=True)

    vector_store = LanceDBVectorStore(
        uri=str(index_path),
        table_name="vault_content",
        mode="overwrite",  # Fresh index
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Collect all markdown files
    documents = []
    for folder in folders:
        folder_path = vault_path / folder
        if not folder_path.exists():
            continue

        for md_file in folder_path.rglob("*.md"):
            try:
                post = frontmatter.load(md_file)
                rel_path = md_file.relative_to(vault_path)

                doc = Document(
                    text=post.content,
                    metadata={
                        "source_file": str(rel_path),
                        "title": post.metadata.get("title", md_file.stem),
                        "type": post.metadata.get("type", "note"),
                        "tags": post.metadata.get("tags", []),
                        "folder": folder,
                    }
                )
                documents.append(doc)
            except Exception as e:
                console.print(f"[yellow]Skipping {md_file}: {e}[/yellow]")

    console.print(f"[cyan]Indexing {len(documents)} documents...[/cyan]")

    # Parse and index
    parser = MarkdownNodeParser()
    nodes = parser.get_nodes_from_documents(documents)

    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
    )

    console.print(f"[green]✓ Indexed {len(nodes)} chunks from {len(documents)} files[/green]")
    return index

if __name__ == "__main__":
    import click

    @click.command()
    @click.argument('vault_path', type=click.Path(exists=True, path_type=Path))
    def main(vault_path: Path):
        """Index the Obsidian vault."""
        index_vault(vault_path)

    main()
```

#### Day 4: MCP Server + Integration

1. Create `src/mcp_server.py` (use the expanded code from the Obsidian section above)
2. Add to Claude config:
   ```json
   {
     "mcpServers": {
       "obsidian-rag": {
         "command": "/path/to/colibri-obsidian/.venv/bin/python",
         "args": ["/path/to/colibri-obsidian/src/mcp_server.py"],
         "env": {
           "VAULT_PATH": "/path/to/your/obsidian/vault"
         }
       }
     }
   }
   ```
3. Restart Claude and test

#### Week 1: Workflow Refinement

1. Import 2-3 key technical books
2. Create a "Book Notes" template in Obsidian
3. Start annotating - add your insights, link concepts
4. Test search from Claude with real questions
5. Set up file watcher for automatic re-indexing (optional)

#### Ongoing Workflow

```
1. Get new book (PDF or EPUB)
   ↓
2. Run: colibri import book.pdf  (or book.epub)
   ↓
3. Open in Obsidian, quick review/cleanup
   ↓
4. Run: colibri index
   ↓
5. Ask Claude questions, it searches your vault
   ↓
6. Add your own notes linking to book content
```

**Supported formats:**
- PDF files → `colibri import book.pdf` or `colibri import-pdf book.pdf`
- EPUB files → `colibri import book.epub` or `colibri import-epub book.epub`

EPUB imports automatically extract metadata (title, author, publisher, ISBN) from the file.

---

### Option B: Pure LlamaIndex Implementation

To implement the **LlamaIndex + LanceDB + Ollama** stack:

### Day 1: Foundation

```bash
# 1. Install Ollama
brew install ollama
ollama serve &
ollama pull nomic-embed-text

# 2. Create project
mkdir -p ~/projects/colibri-rag/{src,data/books,data/lancedb,tests}
cd ~/projects/colibri-rag

# 3. Initialize Python environment
uv init
uv add llama-index llama-index-vector-stores-lancedb \
       llama-index-embeddings-ollama pymupdf4llm lancedb mcp click rich
```

### Day 2: Core Implementation

1. Create `src/config.py`, `src/ingest.py`, `src/query.py` from the code above
2. Test ingestion with one PDF:
   ```bash
   uv run python src/ingest.py ~/path/to/book.pdf
   ```
3. Test search:
   ```bash
   uv run python -c "from src.query import get_engine; print(get_engine().search('your query'))"
   ```

### Day 3: MCP Integration

1. Create `src/mcp_server.py` from the code above
2. Add to `~/.claude/claude_desktop_config.json`
3. Restart Claude Code / Claude Desktop
4. Test by asking Claude a question that should trigger a book search

### Week 1: Production Ready

1. Ingest all your technical books
2. Test retrieval quality with realistic queries
3. Tune `CHUNK_SIZE`, `TOP_K`, `SIMILARITY_THRESHOLD` as needed
4. Add error handling and logging

### Future Enhancements

| Enhancement | Effort | Impact |
|-------------|--------|--------|
| Add reranking via Ollama | Medium | High - better result ordering |
| Hybrid search (BM25 + vector) | Medium | High - better keyword matching |
| Batch ingestion CLI | Low | Medium - easier book management |
| Web UI for search | Medium | Low - Claude is the primary interface |
| Obsidian integration | High | Medium - if you want vault sync |

---

## Appendix: Library Version Reference

Tested with these versions (January 2026):

```
llama-index>=0.12.0
llama-index-vector-stores-lancedb>=0.4.0
llama-index-embeddings-ollama>=0.4.0
lancedb>=0.17.0
pymupdf4llm>=0.0.17
mcp>=1.0.0
ollama (system): 0.5.x
```

Check for updates before implementing - these libraries are actively developed.
