# CoLibri Architecture

This document describes the architecture of CoLibri (COntext LIBRary), a local RAG system for technical books and notes.

## Table of Contents

- [System Overview](#system-overview)
- [Component Architecture](#component-architecture)
- [Data Flow](#data-flow)
- [Key Processes](#key-processes)
- [Data Model](#data-model)
- [Deployment Architecture](#deployment-architecture)
- [Technology Stack](#technology-stack)

## System Overview

CoLibri is a local-first RAG (Retrieval-Augmented Generation) system for searching a personal library of technical books and notes. The system converts PDF and EPUB files to Markdown, indexes content for semantic search, and exposes retrieval via:

- **CLI** (primary; great for coding-focused “tool-using” agents)
- **MCP server** (for Claude Desktop/Code integrations)
- **REST API** (for non-MCP automation)

### Design Principles

1. **Local-First** - All processing happens locally; no cloud dependencies
2. **Source Agnostic** - Works with plain markdown folders and Obsidian vaults
3. **Minimal Infrastructure** - No external databases or services required
4. **Multiple Interfaces** - CLI-first, with optional MCP and REST
5. **Incremental by Default** - Only re-index what has changed

### High-Level Architecture

```plantuml
@startuml
!theme plain
skinparam componentStyle rectangle

title CoLibri System Overview

cloud "Claude Code/Desktop" as claude {
  [Claude LLM]
}

cloud "CLI / Agents" as agents {
  [CLI Clients]
}

package "CoLibri" {
  [CLI] as cli
  [MCP Server] as mcp
  [REST API] as api
  [Query Engine] as query
  [Indexer] as indexer
  [Import Pipeline] as import
  [Manifest] as manifest
}

database "Library" as library {
  folder "Books (Markdown)" as books
}

database "Data Directory" as datadir {
  folder "lancedb (Index)" as index
}

node "Ollama" as ollama {
  [nomic-embed-text]
}

file "PDF/EPUB Files" as files

[Claude LLM] <--> mcp : MCP Protocol
agents --> cli : shell
api --> query
cli --> import
cli --> indexer
cli --> query
mcp --> query
query --> index : Vector Search
query --> ollama : Generate Embeddings
indexer --> books : Read Markdown
indexer --> index : Write Vectors
indexer --> ollama : Generate Embeddings
indexer --> manifest : Track Changes
manifest --> datadir : Store Manifest
import --> files : Read
import --> books : Write Markdown

@enduml
```

## Component Architecture

### Core Components

```plantuml
@startuml
!theme plain
skinparam packageStyle rectangle

title CoLibri Component Architecture

package "colibri" {

  package "Interface Layer" {
    [cli.py] as cli
    [mcp_server.py] as mcp
    [api_server.py] as api
  }

  package "Core Layer" {
    [query.py] as query
    [indexer.py] as indexer
    [manifest.py] as manifest
  }

  package "Processors" {
    [ProcessorRegistry] as registry
    [PDFProcessor] as pdf
    [EPUBProcessor] as epub
    [utils.py] as utils
  }

  package "Sources" {
    [ContentSource] as source_base
    [ObsidianSource] as obsidian
    [MarkdownFolderSource] as markdown_src
  }

  package "Configuration" {
    [config.py] as config
    [IndexMode] as mode
    [FolderProfile] as profile
    file "~/.config/colibri/config.yaml" as yaml
  }
}

package "External Dependencies" {
  [LanceDB] as lance
  [Ollama] as ollama
  [pymupdf4llm] as pymupdf
  [ebooklib] as ebook
}

cli --> query
cli --> indexer
cli --> registry
cli --> config

registry --> pdf
registry --> epub
pdf --> utils
epub --> utils

indexer --> source_base
indexer --> manifest
indexer --> config
query --> source_base
source_base <|-- obsidian
source_base <|-- markdown_src

mcp --> query
mcp --> config

query --> lance
query --> ollama

indexer --> lance
indexer --> ollama

pdf --> pymupdf
epub --> ebook

config --> yaml
config --> mode
config --> profile

@enduml
```

### Component Responsibilities

| Component | File(s) | Responsibility |
|-----------|---------|----------------|
| **CLI** | `cli.py` | Command-line interface, user interaction |
| **MCP Server** | `mcp_server.py` | Claude integration via Model Context Protocol |
| **REST API** | `api_server.py` | HTTP API for Copilot and external clients |
| **Query Engine** | `query.py` | Semantic search, result ranking |
| **Indexer** | `indexer.py` | Document chunking, embedding generation, incremental index management |
| **Manifest** | `manifest.py` | Change tracking via mtime + SHA-256 hash |
| **Config** | `config.py` | Configuration loading, `IndexMode` enum, `FolderProfile` dataclass |
| **Processor Registry** | `processors/registry.py` | Auto-discovery and routing of format handlers |
| **PDF Processor** | `processors/pdf.py` | PDF to Markdown extraction |
| **EPUB Processor** | `processors/epub.py` | EPUB to Markdown extraction |
| **Processor Utils** | `processors/utils.py` | Shared text cleaning, frontmatter generation |
| **Content Source** | `sources/base.py` | Abstract interface for content sources |
| **Obsidian Source** | `sources/obsidian.py` | Obsidian-compatible source adapter (wiki links, frontmatter) |
| **Markdown Source** | `sources/markdown.py` | Plain markdown folder adapter |

## Data Flow

### Import Flow

```plantuml
@startuml
!theme plain

title Book Import Flow

actor User
participant "CLI" as cli
participant "Import Pipeline" as import
participant "pymupdf4llm/ebooklib" as parser
database "Library" as library

User -> cli : colibri import book.pdf
activate cli

cli -> import : _import_document(path)
activate import

import -> parser : Extract content
activate parser
parser --> import : Markdown text
deactivate parser

import -> import : Extract metadata\n(title, author, ISBN)

import -> import : Generate frontmatter

import -> library : Write markdown file
library --> import : File path

import --> cli : Success
deactivate import

cli --> User : "Created: Books/Title.md"
deactivate cli

@enduml
```

### Indexing Flow

```plantuml
@startuml
!theme plain

title Incremental Indexing Flow

actor User
participant "CLI" as cli
participant "Indexer" as indexer
participant "Manifest" as manifest
participant "Ollama" as ollama
database "Data Dir (LanceDB)" as lance
database "Library/Books" as books

User -> cli : colibri index
activate cli

cli -> indexer : index_library()
activate indexer

indexer -> manifest : Load manifest.json

loop For each source profile
  indexer -> books : Scan folder files
  books --> indexer : File list

  indexer -> manifest : Classify files\n(new/changed/unchanged)

  note right of indexer
    **Mode logic:**
    static → skip known files
    incremental → check mtime + hash
    append_only → skip known files
    disabled → skip folder
  end note

  indexer -> books : Read changed files
  books --> indexer : Markdown with frontmatter

  indexer -> indexer : Chunk documents\n(SentenceSplitter)

  loop For each batch of chunks
    indexer -> ollama : Generate embeddings
    ollama --> indexer : Vector embeddings
  end

  indexer -> lance : Store vectors + metadata
  lance --> indexer : Indexed

  indexer -> manifest : Record indexed files
end

indexer -> manifest : Save manifest.json

indexer --> cli : IndexResult\n(indexed, skipped, deleted)
deactivate indexer

cli --> User : Summary
deactivate cli

@enduml
```

### Search Flow (MCP)

```plantuml
@startuml
!theme plain

title MCP Search Flow

actor "Claude" as claude
participant "MCP Server" as mcp
participant "Query Engine" as query
participant "Ollama" as ollama
database "LanceDB" as lance

claude -> mcp : search_books("testing strategies")
activate mcp

mcp -> query : search(query, folder="Books")
activate query

query -> ollama : Embed query text
ollama --> query : Query vector

query -> lance : Vector similarity search
lance --> query : Top-K results

query -> query : Filter by threshold\nExtract metadata

query --> mcp : Results list
deactivate query

mcp --> claude : JSON response\n(text, source, score)
deactivate mcp

note right of claude
  Claude synthesizes
  results into answer
  with citations
end note

@enduml
```

### REST API Search Flow (Copilot)

```plantuml
@startuml
!theme plain

title REST API Search Flow

actor "Copilot/Client" as client
participant "FastAPI Server" as api
participant "Query Engine" as query
participant "Ollama" as ollama
database "LanceDB" as lance

client -> api : GET /api/search/books?q=testing
activate api

api -> query : search_books("testing")
activate query

query -> ollama : Embed query text
ollama --> query : Query vector

query -> lance : Vector similarity search
lance --> query : Top-K results

query -> query : Filter by threshold\nExtract metadata

query --> api : Results list
deactivate query

api --> client : JSON response\n(SearchResponse)
deactivate api

@enduml
```

### Complete Request Flow

```plantuml
@startuml
!theme plain

title End-to-End Request Flow

actor "User" as user
participant "Claude" as claude
participant "MCP Server" as mcp
participant "Query Engine" as query
database "LanceDB" as db

user -> claude : "What does Bach say\nabout testing?"

claude -> claude : Determine if book\nsearch needed

claude -> mcp : tool_call: search_books\n{query: "Bach testing"}
activate mcp

mcp -> query : search_books()
query -> db : Vector search
db --> query : Matching chunks
query --> mcp : Results

mcp --> claude : Tool result (JSON)
deactivate mcp

claude -> claude : Synthesize answer\nfrom retrieved chunks

claude --> user : "According to 'Taking\nTesting Seriously'..."

@enduml
```

## Data Model

### Markdown Document Format

```plantuml
@startuml
!theme plain

title Imported Book Structure

class "Markdown File" as md {
  + path: str
  + frontmatter: YAML
  + content: str
}

class "Frontmatter" as fm {
  + title: str
  + type: "book"
  + source_pdf/epub: str
  + imported: datetime
  + author: str?
  + publisher: str?
  + isbn: str?
  + tags: list[str]
}

class "Content" as content {
  + headers: str
  + paragraphs: str
  + code_blocks: str
  + callouts: str
}

md *-- fm
md *-- content

@enduml
```

### Vector Index Schema

```plantuml
@startuml
!theme plain

title LanceDB Index Schema

entity "chunks" as table {
  * vector : float[768]
  --
  * text : str
  * source_file : str
  * title : str
  * doc_type : str
  * folder : str
  * source_name : str
  * source_type : str
  * tags : str
}

@enduml
```

### Manifest Schema

```plantuml
@startuml
!theme plain

title Manifest Structure (manifest.json in data dir)

	class "Manifest" as manifest {
	  + version: int = 2
	  + indexed_at: str (ISO 8601)
	  + files: dict[str, FileEntry]  # key: "<source_id>:<rel_path>"
	  --
	  + load(path) : Manifest
	  + save(path)
	  + is_file_changed(key, abs_path) : bool
	  + is_file_known(key) : bool
	  + record_file(key, abs_path, chunk_count)
	  + remove_file(key)
	  + get_folder_files(folder) : dict
	}

class "FileEntry" as entry {
  + mtime: float
  + content_hash: str (sha256:...)
  + chunk_count: int
  + indexed_at: str (ISO 8601)
}

manifest "1" *-- "0..*" entry

@enduml
```

### Configuration Schema

```plantuml
@startuml
!theme plain

title Configuration Structure

class "IndexMode" as mode <<enum>> {
  STATIC
  INCREMENTAL
  APPEND_ONLY
  DISABLED
}

class "FolderProfile" as profile <<frozen>> {
  + path: str (absolute)
  + mode: IndexMode
  + doc_type: str
  + chunk_size: int?
  + chunk_overlap: int?
  + extensions: tuple[str, ...]
  + name: str?
  --
  + effective_chunk_size(default) : int
  + effective_chunk_overlap(default) : int
}

object "config.yaml" as config {
  sources = [FolderProfile...]
  data.directory = null (XDG default)
  index.directory = "lancedb"
  ollama.base_url = "http://localhost:11434"
  ollama.embedding_model = "nomic-embed-text"
  retrieval.top_k = 10
  retrieval.similarity_threshold = 0.3
  chunking.chunk_size = 3000
  chunking.chunk_overlap = 200
  translation.model = null
}

profile --> mode
config --> profile

@enduml
```

## Deployment Architecture

### Local Deployment

```plantuml
@startuml
!theme plain

title Local Deployment Architecture

node "User Machine" {

  package "Applications" {
    [Claude Code] as cc
    [Claude Desktop] as cd
    [Obsidian] as obs
  }

  package "CoLibri Runtime" {
    [Python 3.11+] as py
    [Virtual Environment] as venv
    [MCP Server Process] as mcp
    [REST API Server] as restapi
  }

  package "Services" {
    [Ollama Service] as ollama
  }

	  package "File System" {
	    folder "~/.config/colibri" as conf
	    folder "~/.local/share/colibri" as datadir {
	      folder "lancedb/" as index {
	        file "index_meta.json" as indexmeta
	      }
	      file "manifest.json" as manifest
	      file "doc_catalog.json" as catalog
	      file "index_changes.jsonl" as changes
	    }
	    folder "Library" as library {
	      folder "Books/" as books
	    }
	    folder "~/.mcp.json" as mcpconf
	  }
	}

cc --> mcp : stdio
cd --> mcp : stdio
mcp --> venv
venv --> py
mcp --> ollama : HTTP :11434
mcp --> index : Read vectors
mcp --> conf : Read config
obs --> books : Edit markdown
	restapi --> datadir : Read index state\n(manifest/meta/catalog)
restapi --> ollama : HTTP :11434
restapi --> index : Read vectors

@enduml
```

### Process Architecture

```plantuml
@startuml
!theme plain

title Process Architecture

rectangle "Claude Code Process" as claude {
}

rectangle "MCP Server Process" as mcp {
  component "colibri.mcp_server"
  component "LanceDB (embedded)"
}

rectangle "Ollama Process" as ollama {
  component "nomic-embed-text model"
}

claude <--> mcp : stdin/stdout\n(JSON-RPC)
mcp --> ollama : HTTP API\n(embeddings)

note bottom of mcp
  Spawned by Claude
  on startup via
  ~/.mcp.json config
end note

@enduml
```

## Technology Stack

### Dependencies

```plantuml
@startuml
!theme plain

title Technology Stack

package "Vector Storage" {
  [LanceDB] as lance
  note right: Embedded vector DB\nNo server required
}

package "Embeddings" {
  [Ollama] as ollama
  [nomic-embed-text] as nomic
  ollama --> nomic
  note right: Local embedding\ngeneration
}

package "Document Processing" {
  [pymupdf4llm] as pdf
  [ebooklib] as epub
  [markdownify] as md
  note right: PDF/EPUB to\nMarkdown conversion
}

package "Interface" {
  [MCP SDK] as mcp
  [FastAPI] as fastapi
  [Click] as click
  [Rich] as rich
  note right: Claude integration\nREST API\nCLI interface
}

package "Configuration" {
  [PyYAML] as yaml
  [python-frontmatter] as fm
  note right: YAML config\nMarkdown metadata
}

@enduml
```

### Version Requirements

| Component | Version | Purpose |
|-----------|---------|---------|
| Python | >=3.11 | Runtime |
| LanceDB | >=0.17.0 | Vector storage |
| Ollama | >=0.5.x | Embedding service |
| MCP SDK | >=1.0.0 | Claude integration |
| FastAPI | >=0.115.0 | REST API server |

## Security Considerations

### Data Privacy

```plantuml
@startuml
!theme plain

title Data Flow - Privacy Perspective

rectangle "Local Machine" {
  [Books (PDF/EPUB)] as src
  [Markdown Files] as md
  [Vector Index] as vec
  [Ollama] as ollama
  [Claude Code] as claude
}

cloud "External" {
  [Anthropic API] as api
}

src --> md : Local conversion
md --> vec : Local indexing
md --> ollama : Local embeddings
claude --> api : Queries + Context\n(retrieved chunks)

note bottom of api
  Only retrieved text chunks
  are sent to Anthropic API
  as part of Claude context
end note

note bottom of vec
  Vectors and full text
  remain local
end note

@enduml
```

### Key Security Points

1. **Local Processing** - All document processing and embedding generation happens locally
2. **No Cloud Storage** - Book content and vectors are stored only on local filesystem
3. **Selective Context** - Only relevant chunks (not full books) are sent to Claude API
4. **Separate Data** - Index and manifest are stored in a separate data directory (`~/.local/share/colibri/`), not inside the library

## Future Considerations

### Potential Enhancements

```plantuml
@startuml
!theme plain

title Potential Future Architecture

package "Current" {
  [Semantic Search]
  [Incremental Indexing]
  [Per-Folder Profiles]
}

package "Enhancements" {
  [Hybrid Search\n(BM25 + Vector)] as hybrid
  [Cross-encoder\nReranking] as rerank
  [File Watcher\n(Auto-reindex)] as watch
  [Web UI / TUI\nConfig Interface] as ui
}

[Semantic Search] --> hybrid : Add keyword matching
hybrid --> rerank : Improve ranking
[Incremental Indexing] --> watch : Auto-detect changes
[Per-Folder Profiles] --> ui : Visual configuration

@enduml
```

| Enhancement | Complexity | Impact |
|-------------|------------|--------|
| Hybrid Search (BM25) | Medium | Better keyword matching |
| Cross-encoder Reranking | Medium | Improved result ordering |
| File Watcher | Medium | Auto-reindex on changes |
| Web UI / TUI Config | Medium | Visual profile management |
