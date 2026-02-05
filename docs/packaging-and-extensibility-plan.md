# CoLibri: Packaging, Vault Flexibility & Document Processor Modularization

**Planning Document v1.1**
**Date:** 2026-01-29

---

## Status Overview

| Workstream | Status | Notes |
|------------|--------|-------|
| 1. Packaging & Installation | ✅ **Completed** | Implemented 2026-01-29 |
| 2. Vault Flexibility | ✅ **Completed** | Implemented 2026-01-29 |
| 3. Document Processor Modularization | ✅ **Completed** | Implemented 2026-01-29 |

---

## Executive Summary

This document outlines three independent workstreams to evolve CoLibri from a development prototype to a distributable, flexible knowledge management tool:

1. **Packaging & Installation** — Make CoLibri easy to install and run for non-developers
2. **Vault Flexibility** — Support any markdown folder structure, not just Obsidian
3. **Document Processor Modularization** — Create a plugin architecture for format handlers ✅

Each section can be implemented independently, though they share some infrastructure.

---

## 1. Packaging & Installation

### Current State

```
Installation today:
1. Clone repository
2. Install uv
3. Run `uv sync`
4. Configure ~/.config/colibri/config.yaml manually
5. Set up MCP configuration manually
6. Ensure Ollama is running + model pulled
```

This requires Python knowledge and multiple manual steps.

### Goals

| Goal | Priority | Rationale |
|------|----------|-----------|
| Single-command installation | High | Remove barrier to entry |
| Automatic dependency setup | High | Ollama model pulling, etc. |
| Cross-platform support | Medium | macOS, Linux, (Windows) |
| Zero-config quick start | High | Works immediately after install |
| MCP/API auto-configuration | Medium | Seamless Claude/Copilot integration |

### Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Distribution Options                         │
├─────────────────────┬─────────────────────┬─────────────────────┤
│    PyPI Package     │   pipx / uv tool    │   Homebrew Formula  │
│    `pip install`    │   `pipx install`    │   `brew install`    │
└─────────────────────┴─────────────────────┴─────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Bootstrap Command                           │
│                     `colibri setup`                              │
├─────────────────────────────────────────────────────────────────┤
│  1. Check Python version (≥3.11)                                │
│  2. Detect/install Ollama if missing                            │
│  3. Pull embedding model (nomic-embed-text)                     │
│  4. Create default config.yaml with prompts                     │
│  5. Configure MCP integration (Claude Code)                     │
│  6. Run initial health check                                    │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation Tasks

#### 1.1 PyPI Distribution (Priority: High)

**File changes:**
- `pyproject.toml` — Add metadata for PyPI publication
- `README.md` — Update with pip install instructions

**New fields needed:**
```toml
[project]
description = "Local RAG system for technical books with Claude/Copilot integration"
readme = "README.md"
license = { text = "MIT" }
authors = [{ name = "Your Name", email = "you@example.com" }]
keywords = ["rag", "obsidian", "llm", "knowledge-management"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Environment :: Console",
    "Topic :: Text Processing :: Indexing",
]

[project.urls]
Homepage = "https://github.com/..."
Documentation = "https://..."
```

**Deliverables:**
- [ ] Complete pyproject.toml metadata
- [ ] Create `LICENSE` file
- [ ] GitHub Actions workflow for PyPI publish
- [ ] Test installation via `pip install colibri-rag` (or chosen name)

#### 1.2 Bootstrap Command (Priority: High)

**New file:** `src/colibri/setup.py`

**Subcommand:** `colibri setup`

**Interactive flow:**
```
$ colibri setup

CoLibri Setup Wizard
====================

Checking prerequisites...
  ✓ Python 3.14.0
  ✗ Ollama not found

Would you like to install Ollama? [Y/n]: y
  → Installing Ollama via brew...
  → Pulling nomic-embed-text model...
  ✓ Ollama ready

Configure your vault:
  Vault path [~/Documents/CoLibri]: ~/Obsidian/MainVault
  Books folder [Books]:
  ✓ Configuration saved to ~/.config/colibri/config.yaml

Would you like to enable Claude Code integration? [Y/n]: y
  ✓ MCP configuration written to ~/.mcp.json

Setup complete! Try these commands:
  colibri import ~/Downloads/book.pdf
  colibri index
  colibri search "clean architecture"
```

**Deliverables:**
- [ ] `colibri setup` command with interactive wizard
- [ ] Ollama detection and guided installation
- [ ] Model pulling with progress indicator
- [ ] Config file generation with validation
- [ ] MCP configuration helper

#### 1.3 Homebrew Formula (Priority: Low)

For macOS users who prefer Homebrew:

```ruby
class Colibri < Formula
  include Language::Python::Virtualenv

  desc "Local RAG system for technical books"
  homepage "https://github.com/..."
  url "https://files.pythonhosted.org/..."
  sha256 "..."
  license "MIT"

  depends_on "python@3.11"

  def install
    virtualenv_install_with_resources
  end

  def caveats
    <<~EOS
      To complete setup, run:
        colibri setup

      Ollama is required for embeddings:
        brew install ollama
        ollama pull nomic-embed-text
    EOS
  end
end
```

**Deliverables:**
- [ ] Homebrew formula in separate tap repository
- [ ] Installation documentation

#### 1.4 Shell Completion (Priority: Low)

Enable tab completion for better UX:

```python
# cli.py addition
@cli.command()
@click.option('--shell', type=click.Choice(['bash', 'zsh', 'fish']))
def completion(shell):
    """Generate shell completion script."""
    # Output appropriate completion script
```

**Deliverables:**
- [ ] Completion scripts for bash/zsh/fish
- [ ] Installation instructions in README

### Testing Strategy

| Test | Method |
|------|--------|
| Fresh install | Docker container with clean Python |
| Upgrade path | Install v0.1, upgrade to v0.2 |
| Config migration | Old config → new config format |
| Cross-platform | GitHub Actions matrix (ubuntu, macos) |

---

## 2. Vault Flexibility

### Current State

The system assumes an Obsidian vault structure:
- Hardcoded `Books/` folder convention
- Obsidian-specific frontmatter assumptions
- `.colibri/` index directory inside vault
- Wiki-link parsing (`[[Note Name]]`)

### Goals

| Goal | Priority | Rationale |
|------|----------|-----------|
| Support plain markdown folders | High | Not everyone uses Obsidian |
| Multiple vault/folder sources | Medium | Aggregate content from different locations |
| Portable index location | Medium | Index doesn't need to live in vault |
| Optional Obsidian features | Low | Wiki links as opt-in |

### Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Content Source Abstraction                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐          │
│   │  Obsidian   │   │   Plain     │   │   Custom    │          │
│   │   Vault     │   │  Markdown   │   │  Adapter    │          │
│   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘          │
│          │                 │                  │                  │
│          └─────────────────┼──────────────────┘                  │
│                            ▼                                     │
│              ┌─────────────────────────┐                        │
│              │   ContentSource ABC     │                        │
│              │   - list_documents()    │                        │
│              │   - read_document()     │                        │
│              │   - get_metadata()      │                        │
│              │   - resolve_link()      │                        │
│              └─────────────────────────┘                        │
│                            │                                     │
│                            ▼                                     │
│              ┌─────────────────────────┐                        │
│              │       Indexer           │                        │
│              │   (source-agnostic)     │                        │
│              └─────────────────────────┘                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation Tasks

#### 2.1 Content Source Abstraction (Priority: High)

**New file:** `src/colibri/sources/base.py`

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

@dataclass
class Document:
    """Normalized document representation."""
    path: Path
    content: str
    title: str
    metadata: dict[str, Any]
    source_type: str  # "obsidian", "markdown", etc.

class ContentSource(ABC):
    """Abstract base for content sources."""

    @abstractmethod
    def list_documents(self, folders: list[str] | None = None) -> Iterator[Path]:
        """Yield paths to indexable documents."""
        ...

    @abstractmethod
    def read_document(self, path: Path) -> Document:
        """Read and parse a document."""
        ...

    @abstractmethod
    def resolve_link(self, link: str, from_doc: Path) -> Path | None:
        """Resolve internal links (wiki links, relative paths, etc.)."""
        ...

    @property
    @abstractmethod
    def root_path(self) -> Path:
        """Base path for this content source."""
        ...
```

**Deliverables:**
- [ ] `ContentSource` abstract base class
- [ ] `Document` dataclass for normalized representation
- [ ] Unit tests for interface contract

#### 2.2 Obsidian Source Adapter (Priority: High)

**New file:** `src/colibri/sources/obsidian.py`

Extracts current Obsidian-specific logic:
- YAML frontmatter parsing
- Wiki link resolution (`[[Note Name]]` → path)
- Tag extraction from frontmatter
- `.obsidian/` folder exclusion

```python
class ObsidianSource(ContentSource):
    """Obsidian vault adapter with wiki-link support."""

    def __init__(self, vault_path: Path, books_folder: str = "Books"):
        self.vault_path = vault_path
        self.books_folder = books_folder
        self._build_link_index()

    def resolve_link(self, link: str, from_doc: Path) -> Path | None:
        # Handle [[Wiki Links]] and [[Link|Alias]]
        ...
```

**Deliverables:**
- [ ] `ObsidianSource` implementation
- [ ] Migrate existing vault logic to this adapter
- [ ] Tests with sample Obsidian vault structure

#### 2.3 Plain Markdown Source Adapter (Priority: High)

**New file:** `src/colibri/sources/markdown.py`

Simpler adapter for non-Obsidian folders:
- Optional frontmatter (graceful fallback)
- Standard markdown link resolution
- Filename-based title extraction

```python
class MarkdownFolderSource(ContentSource):
    """Plain markdown folder adapter."""

    def __init__(self, folder_path: Path, recursive: bool = True):
        self.folder_path = folder_path
        self.recursive = recursive

    def read_document(self, path: Path) -> Document:
        content = path.read_text()

        # Try frontmatter, fall back to filename
        if content.startswith("---"):
            metadata, content = parse_frontmatter(content)
            title = metadata.get("title", path.stem)
        else:
            metadata = {}
            title = path.stem.replace("-", " ").replace("_", " ").title()

        return Document(
            path=path,
            content=content,
            title=title,
            metadata=metadata,
            source_type="markdown"
        )
```

**Deliverables:**
- [ ] `MarkdownFolderSource` implementation
- [ ] Frontmatter-optional parsing
- [ ] Tests with plain markdown folders

#### 2.4 Multi-Source Configuration (Priority: Medium)

Update config.yaml to support multiple sources:

```yaml
# New configuration format
sources:
  - type: obsidian
    path: ~/Obsidian/MainVault
    folders: [Books, Notes]

  - type: markdown
    path: ~/Documents/TechNotes
    recursive: true

  - type: markdown
    path: ~/Projects/docs
    recursive: false

index:
  # Index can now be anywhere
  directory: ~/.local/share/colibri/index
```

**Backward compatibility:**
```yaml
# Old format still works (detected and migrated)
vault:
  path: ~/Obsidian/Vault
  books_folder: Books
```

**Deliverables:**
- [ ] New config schema with `sources` list
- [ ] Auto-migration from old `vault` format
- [ ] Multi-source indexer support
- [ ] CLI updates for source management

#### 2.5 Portable Index Location (Priority: Medium)

Decouple index from vault:

```python
# config.py
def get_index_path(config: dict) -> Path:
    """Determine index storage location."""
    if explicit := config.get("index", {}).get("directory"):
        return Path(explicit).expanduser()

    # Default: XDG data directory
    xdg_data = os.environ.get(
        "XDG_DATA_HOME",
        Path.home() / ".local" / "share"
    )
    return Path(xdg_data) / "colibri" / "index"
```

**Benefits:**
- Index survives vault reorganization
- Can share vault via cloud sync without index bloat
- Cleaner separation of concerns

**Deliverables:**
- [ ] XDG-compliant default index location
- [ ] Config option for custom index path
- [ ] Migration tool for existing `.colibri/` indices

### Testing Strategy

| Test | Method |
|------|--------|
| Obsidian vault | Sample vault with wiki links, tags |
| Plain markdown | Flat folder, nested folder |
| Mixed frontmatter | Some files with, some without |
| Multi-source | Two sources merged in search results |
| Link resolution | Wiki links vs relative paths |

---

## 3. Document Processor Modularization

### Current State

Two processors exist:
- `pdf_to_vault.py` — Uses pymupdf4llm
- `epub_to_vault.py` — Uses ebooklib + markdownify

Both have:
- Similar structure but duplicated code
- Hardcoded output format (Obsidian markdown)
- No extensibility mechanism

### Goals

| Goal | Priority | Rationale |
|------|----------|-----------|
| Plugin architecture | High | Easy to add new formats |
| Shared utilities | High | DRY for common operations |
| Format auto-detection | Medium | `colibri import file.xyz` just works |
| Output format options | Low | Not everyone wants Obsidian format |

### Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Processor Plugin System                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                  ProcessorRegistry                       │   │
│   │   - register(processor)                                  │   │
│   │   - get_processor(file_path) -> Processor               │   │
│   │   - list_supported_formats() -> list[str]               │   │
│   └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│              ┌───────────────┼───────────────┐                  │
│              ▼               ▼               ▼                  │
│   ┌──────────────┐ ┌──────────────┐ ┌──────────────┐           │
│   │ PDFProcessor │ │EPUBProcessor │ │ Future...    │           │
│   │              │ │              │ │ - MOBI       │           │
│   │ extensions:  │ │ extensions:  │ │ - DJVU       │           │
│   │ [.pdf]       │ │ [.epub]      │ │ - HTML       │           │
│   └──────────────┘ └──────────────┘ └──────────────┘           │
│              │               │               │                  │
│              └───────────────┼───────────────┘                  │
│                              ▼                                   │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                  DocumentProcessor ABC                   │   │
│   │   - extensions: list[str]                               │   │
│   │   - extract(path) -> ExtractedContent                   │   │
│   │   - can_handle(path) -> bool                            │   │
│   └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                  OutputFormatter                         │   │
│   │   - format_obsidian(content) -> str                     │   │
│   │   - format_plain(content) -> str                        │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation Tasks

#### 3.1 Processor Base Class (Priority: High)

**New file:** `src/colibri/processors/base.py`

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

@dataclass
class ExtractedContent:
    """Normalized extraction result."""
    title: str
    content: str  # Markdown text
    metadata: dict[str, Any] = field(default_factory=dict)
    source_path: Path | None = None
    extracted_at: datetime = field(default_factory=datetime.now)

    # Optional rich metadata
    author: str | None = None
    publisher: str | None = None
    language: str | None = None
    isbn: str | None = None
    description: str | None = None

class DocumentProcessor(ABC):
    """Base class for document format processors."""

    # Subclasses define supported extensions
    extensions: list[str] = []

    @abstractmethod
    def extract(self, path: Path) -> ExtractedContent:
        """Extract content from document."""
        ...

    def can_handle(self, path: Path) -> bool:
        """Check if this processor can handle the file."""
        return path.suffix.lower() in self.extensions
```

**Deliverables:**
- [ ] `ExtractedContent` dataclass
- [ ] `DocumentProcessor` ABC
- [ ] Type hints and documentation

#### 3.2 Shared Utilities Module (Priority: High)

**New file:** `src/colibri/processors/utils.py`

Consolidate duplicated logic:

```python
def sanitize_filename(title: str) -> str:
    """Convert title to safe filename."""
    safe = re.sub(r'[^\w\s\-]', '', title)
    safe = re.sub(r'\s+', ' ', safe).strip()
    return safe[:100]  # Length limit

def clean_text(text: str) -> str:
    """Normalize unicode and clean whitespace."""
    # Smart quotes → straight quotes
    replacements = {
        '\u2018': "'", '\u2019': "'",  # Single quotes
        '\u201c': '"', '\u201d': '"',  # Double quotes
        '\u2013': '-', '\u2014': '--', # Dashes
        '\u2026': '...', '\xa0': ' ',  # Ellipsis, NBSP
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Remove control characters
    text = ''.join(c for c in text if c >= ' ' or c in '\n\t')

    return text

def format_frontmatter(metadata: dict) -> str:
    """Generate YAML frontmatter block."""
    import yaml
    return f"---\n{yaml.dump(metadata, default_flow_style=False)}---\n"

def generate_markdown_document(
    content: ExtractedContent,
    include_source_callout: bool = True
) -> str:
    """Format ExtractedContent as Obsidian-style markdown."""
    ...
```

**Deliverables:**
- [ ] `sanitize_filename()`
- [ ] `clean_text()` with unicode normalization
- [ ] `format_frontmatter()`
- [ ] `generate_markdown_document()`
- [ ] Unit tests for all utilities

#### 3.3 Processor Registry (Priority: High)

**New file:** `src/colibri/processors/registry.py`

```python
from pathlib import Path
from typing import Type

class ProcessorRegistry:
    """Central registry for document processors."""

    _processors: list[Type[DocumentProcessor]] = []

    @classmethod
    def register(cls, processor: Type[DocumentProcessor]) -> Type[DocumentProcessor]:
        """Register a processor (can be used as decorator)."""
        cls._processors.append(processor)
        return processor

    @classmethod
    def get_processor(cls, path: Path) -> DocumentProcessor | None:
        """Find processor that can handle the file."""
        for processor_cls in cls._processors:
            processor = processor_cls()
            if processor.can_handle(path):
                return processor
        return None

    @classmethod
    def supported_extensions(cls) -> list[str]:
        """List all supported file extensions."""
        extensions = []
        for processor_cls in cls._processors:
            extensions.extend(processor_cls.extensions)
        return extensions
```

**Usage:**
```python
# In pdf.py
@ProcessorRegistry.register
class PDFProcessor(DocumentProcessor):
    extensions = ['.pdf']
    ...

# In cli.py
processor = ProcessorRegistry.get_processor(Path("book.pdf"))
if processor:
    content = processor.extract(path)
```

**Deliverables:**
- [ ] `ProcessorRegistry` with auto-discovery
- [ ] Decorator-based registration
- [ ] Format listing for help text

#### 3.4 Refactored PDF Processor (Priority: High)

**New file:** `src/colibri/processors/pdf.py`

```python
import pymupdf4llm
from .base import DocumentProcessor, ExtractedContent
from .utils import clean_text, sanitize_filename
from .registry import ProcessorRegistry

@ProcessorRegistry.register
class PDFProcessor(DocumentProcessor):
    """Extract content from PDF documents."""

    extensions = ['.pdf']

    def extract(self, path: Path) -> ExtractedContent:
        # Extract with pymupdf4llm
        raw_md = pymupdf4llm.to_markdown(str(path))

        # Clean and normalize
        content = clean_text(raw_md)

        # Derive title from filename
        title = sanitize_filename(path.stem)

        return ExtractedContent(
            title=title,
            content=content,
            source_path=path,
            metadata={
                "type": "book",
                "source_format": "pdf",
                "source_file": path.name,
            }
        )
```

**Deliverables:**
- [ ] Refactor `pdf_to_vault.py` → `processors/pdf.py`
- [ ] Use shared utilities
- [ ] Maintain backward compatibility in CLI

#### 3.5 Refactored EPUB Processor (Priority: High)

**New file:** `src/colibri/processors/epub.py`

```python
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from markdownify import markdownify

from .base import DocumentProcessor, ExtractedContent
from .utils import clean_text
from .registry import ProcessorRegistry

@ProcessorRegistry.register
class EPUBProcessor(DocumentProcessor):
    """Extract content from EPUB documents."""

    extensions = ['.epub']

    def extract(self, path: Path) -> ExtractedContent:
        book = epub.read_epub(str(path))

        # Extract metadata
        title = self._get_metadata(book, 'title') or path.stem
        author = self._get_metadata(book, 'creator')
        publisher = self._get_metadata(book, 'publisher')
        language = self._get_metadata(book, 'language')

        # Process chapters
        chapters = []
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            html = item.get_content().decode('utf-8')
            soup = BeautifulSoup(html, 'html.parser')

            # Remove non-content elements
            for tag in soup(['script', 'style', 'nav']):
                tag.decompose()

            md = markdownify(str(soup), heading_style="ATX")
            chapters.append(clean_text(md))

        return ExtractedContent(
            title=title,
            content="\n\n---\n\n".join(chapters),
            author=author,
            publisher=publisher,
            language=language,
            source_path=path,
            metadata={
                "type": "book",
                "source_format": "epub",
            }
        )

    def _get_metadata(self, book: epub.EpubBook, field: str) -> str | None:
        values = book.get_metadata('DC', field)
        return values[0][0] if values else None
```

**Deliverables:**
- [ ] Refactor `epub_to_vault.py` → `processors/epub.py`
- [ ] Richer metadata extraction
- [ ] Use shared utilities

#### 3.6 Updated CLI Import Command (Priority: High)

**Modify:** `src/colibri/cli.py`

```python
from colibri.processors.registry import ProcessorRegistry
from colibri.processors.utils import generate_markdown_document

@cli.command()
@click.argument('files', nargs=-1, type=click.Path(exists=True))
@click.option('--output', '-o', help='Output directory (default: vault books folder)')
def import_(files, output):
    """Import documents into the vault.

    Supported formats: PDF, EPUB
    """
    for file_path in files:
        path = Path(file_path)

        processor = ProcessorRegistry.get_processor(path)
        if not processor:
            console.print(f"[red]Unsupported format: {path.suffix}[/red]")
            console.print(f"Supported: {', '.join(ProcessorRegistry.supported_extensions())}")
            continue

        with console.status(f"Processing {path.name}..."):
            content = processor.extract(path)
            markdown = generate_markdown_document(content)

            # Write to vault
            output_path = determine_output_path(content.title, output)
            output_path.write_text(markdown)

        console.print(f"[green]✓[/green] Imported: {content.title}")
```

**Deliverables:**
- [ ] Unified `import` command using registry
- [ ] Auto-detection of format
- [ ] Better error messages for unsupported formats

#### 3.7 Future Processor Stubs (Priority: Low)

Prepare extension points:

```python
# src/colibri/processors/mobi.py (stub)
class MOBIProcessor(DocumentProcessor):
    """MOBI/AZW format support (requires calibre)."""
    extensions = ['.mobi', '.azw', '.azw3']

    def extract(self, path: Path) -> ExtractedContent:
        raise NotImplementedError(
            "MOBI support requires calibre. "
            "Install with: brew install calibre"
        )

# src/colibri/processors/html.py (stub)
class HTMLProcessor(DocumentProcessor):
    """HTML/web page support."""
    extensions = ['.html', '.htm']
    # Implementation...
```

**Deliverables:**
- [ ] MOBI processor stub with calibre integration path
- [ ] HTML processor for web clippings
- [ ] Documentation for adding new processors

### Package Structure After Refactor

```
src/colibri/
├── __init__.py
├── cli.py                    # Updated for new processors
├── config.py
├── indexer.py
├── query.py
├── mcp_server.py
├── api_server.py
├── processors/               # NEW: modular processors
│   ├── __init__.py          # Export registry
│   ├── base.py              # ABC and dataclasses
│   ├── registry.py          # Processor registry
│   ├── utils.py             # Shared utilities
│   ├── pdf.py               # PDF processor
│   └── epub.py              # EPUB processor
└── sources/                  # NEW: content source adapters
    ├── __init__.py
    ├── base.py              # ABC
    ├── obsidian.py          # Obsidian adapter
    └── markdown.py          # Plain markdown adapter
```

### Testing Strategy

| Test | Method |
|------|--------|
| PDF extraction | Sample PDFs with various layouts |
| EPUB extraction | Sample EPUBs with metadata |
| Registry lookup | Extension matching |
| Utility functions | Unit tests with edge cases |
| CLI integration | End-to-end import tests |

---

## Implementation Roadmap

### Phase 1: Foundation (Week 1-2)

Focus on enabling independent work on all three tracks:

```
┌─────────────────────────────────────────────────────────────────┐
│ Shared Foundation                                                │
├─────────────────────────────────────────────────────────────────┤
│ □ Create processors/ package with base classes                  │
│ □ Create sources/ package with base classes                     │
│ □ Extract shared utilities                                       │
│ □ Add comprehensive tests                                        │
└─────────────────────────────────────────────────────────────────┘
```

### Phase 2: Parallel Tracks (Week 2-4)

```
Track A: Packaging          Track B: Vault Flex       Track C: Processors
─────────────────────       ─────────────────────     ─────────────────────
□ PyPI metadata             □ ObsidianSource          □ PDF processor
□ `colibri setup`           □ MarkdownSource          □ EPUB processor
□ Ollama bootstrap          □ Multi-source config     □ Registry + CLI
□ MCP auto-config           □ Index relocation        □ Format detection
```

### Phase 3: Integration (Week 4-5)

```
┌─────────────────────────────────────────────────────────────────┐
│ Integration & Polish                                             │
├─────────────────────────────────────────────────────────────────┤
│ □ End-to-end testing                                            │
│ □ Documentation updates                                          │
│ □ Migration guides                                               │
│ □ Release v0.2.0                                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Dependency Graph

```
                    ┌─────────────────────┐
                    │   Shared Utils      │
                    │   (processors/      │
                    │    utils.py)        │
                    └──────────┬──────────┘
                               │
           ┌───────────────────┼───────────────────┐
           │                   │                   │
           ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ 1. Packaging    │ │ 2. Vault Flex   │ │ 3. Processors   │
│                 │ │                 │ │                 │
│ - setup command │ │ - ContentSource │ │ - Registry      │
│ - PyPI publish  │ │ - Multi-source  │ │ - PDF/EPUB      │
│ - Homebrew      │ │ - Index path    │ │ - Future fmts   │
└─────────────────┘ └─────────────────┘ └─────────────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │
                             ▼
                    ┌─────────────────────┐
                    │   CLI Integration   │
                    │   (cli.py updates)  │
                    └─────────────────────┘
```

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Breaking config changes | Medium | High | Auto-migration + deprecation warnings |
| PyPI name collision | Low | Medium | Check availability early, have alternatives |
| Multi-source performance | Low | Medium | Lazy loading, parallel indexing |
| Homebrew approval delays | Medium | Low | Optional, PyPI is primary |

---

## Success Criteria

### Packaging
- [ ] `pip install colibri-rag && colibri setup` works on fresh machine
- [ ] Setup wizard handles missing Ollama gracefully
- [ ] Upgrade from dev install preserves configuration

### Vault Flexibility
- [ ] Plain markdown folder indexes without Obsidian
- [ ] Multiple sources searchable simultaneously
- [ ] Existing Obsidian users see no behavior change

### Processors
- [ ] `colibri import book.pdf` auto-detects format
- [ ] Adding new processor requires only one file
- [ ] All existing PDF/EPUB functionality preserved

---

## Next Steps

1. **Review this plan** and identify any missing requirements
2. **Choose starting track** based on immediate needs
3. **Create feature branches** for parallel development
4. **Set up CI/CD** for PyPI publishing (Track A prerequisite)

---

*Document maintained in `/docs/packaging-and-extensibility-plan.md`*
