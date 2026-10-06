# CoLibri

Local RAG system for semantic search over markdown content. Indexes markdown files into LanceDB and exposes search via CLI and MCP server.

## Installation

### Homebrew (macOS)

```bash
brew tap TobiSchelling/tap
brew install colibri
```

### From Source

Requires Rust toolchain and protobuf compiler:

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install protoc (macOS)
brew install protobuf

# Build
cargo build --release

# Binary at target/release/colibri
```

## Prerequisites

CoLibri uses Ollama for local embeddings:

```bash
brew install ollama
ollama serve
ollama pull bge-m3
```

## Commands

```bash
# In-app tour / concepts
colibri tour

# First-time setup wizard
colibri bootstrap

# Health check (config, connectors, metadata DB, Ollama, index readiness)
colibri doctor
colibri doctor --json --strict

# Bring mirrors (folders) up to date: new, changed and deleted files, then index
colibri update
colibri update vault --dry-run          # show what would change, write nothing
colibri update --allow-mass-prune       # accept deletions above the safety limit
colibri update --retry-failed           # retry files whose conversion failed before

# What is in CoLibri, what is pending, what needs attention
colibri status
colibri status --json

# Remote connectors (Zephyr Scale) until they become fetchers
colibri sync --connector zephyr-ctslab

# Bring the index in line with the metadata DB
colibri index
colibri index --force

# Books: sweep the configured library folders (only new or changed books)
colibri add
colibri add ~/Downloads/book.epub          # one book, from anywhere
colibri add book.pdf --reconvert           # redo a poor conversion
colibri list
colibri remove "Pro Git"                   # later sweeps will not add it again

# Hybrid (default) / semantic / keyword search
colibri search "microservices patterns"
colibri search "clean architecture" --json --limit 10
colibri search "architecture decisions" --collection vault --mode keyword

# Filter by source path (substring match, repeatable)
colibri search "stakeholder commitments" --path-includes 0300_PROJECTS

# Filter by parsed YAML frontmatter (repeatable KEY=VALUE)
colibri search "test plan" --frontmatter area=SIT --frontmatter status=active

# Time-bound queries
colibri search "decisions" --since 2026-04-01T00:00:00Z

# Document-level grouping (best chunk per file + chunk_count + frontmatter)
colibri search "Heimdall" --group-by-doc --limit 5

# MCP server (for Claude integration)
colibri serve --check
colibri serve

# Delete all CoLibri data and rebuild from sources
colibri reset
colibri update
```

Every ingested document belongs to a collection: the mirror `name`, the connector `id` for `sync`, and `books` for the library. Search results carry the collection, and `--collection` (CLI) or `collection` (MCP) restricts a search to one.

### Library

The library holds books. A folder with a calibre `metadata.opf` is one book: it is identified by calibre's book UUID (renaming or moving it in calibre does not create a duplicate), its title, authors, language and tags come from the OPF, and one format is used (`prefer_formats`, EPUB before PDF by default). Any other EPUB, PDF or DOCX is a book identified by its content. `.acsm` files are Adobe DRM links, not books; they are reported in `colibri status`.

Books are never deleted automatically. A book whose file disappears stays searchable and shows as "source missing"; `colibri remove` takes a book out, and sweeps respect that until you add the file again explicitly. A changed EPUB is converted again (and re-embedded only if its text changed); a changed PDF is only reported, because re-running docling is slow, so update it with `colibri add <file> --reconvert`. A file whose conversion failed is not retried on every run; use `--retry-failed` (installing a missing converter needs no flag).

`colibri update` sweeps the library too, so one command keeps everything current.

### Mirrors

A mirror is a folder CoLibri keeps in sync. `colibri update` ingests new files, re-ingests changed ones and removes documents whose files are gone. Unchanged files cost a hash (markdown, YAML) or a size/mtime check (PDF, EPUB, DOCX, PPTX); a converted file is never converted twice for the same bytes.

Deletions are guarded. Nothing is pruned when the mirror folder is missing or any part of it could not be read, and a run that would delete more than `max(prune.min_count, prune.max_fraction × documents)` stops and reports instead (override with `--allow-mass-prune`). Narrowing `include`/`exclude` counts as deleting, so review `colibri update --dry-run` first. Files that exist only in the cloud (OneDrive/iCloud placeholders) are skipped and reported, never pruned.

Commands that change data (`update`, `add`, `remove`, `sync`, `index`, `reset`) take a write lock, so only one runs at a time; a second one fails immediately and names the process holding the lock. `search` and `serve` only read and never block writers.

`colibri serve --check` and `colibri doctor` report whether the index matches the configured embedding model; `colibri serve` refuses to start when it does not.

## Configuration

CoLibri reads configuration from `~/.config/colibri/config.yaml`:

```yaml
data:
  directory: ~/.local/share/colibri
embedding:
  endpoint: http://localhost:11434   # Ollama
  model: bge-m3
chunking:
  chunk_size: 512
  chunk_overlap: 64
retrieval:
  top_k: 10
  similarity_threshold: 0.3
mirrors:
  - name: vault                      # collection name, prefix of document ids
    path: ~/PKM
    include: ["**/*.md"]             # default
    exclude: [".obsidian/**", "**/.trash/**"]
    doc_type: note                   # default
  - name: architecture
    path: ~/GIT_ROOT/GIT_LAB/ONTrack/architecture-artifacts
    include: ["**/*.md", "**/*.yaml", "**/*.yml"]
    doc_type: architecture
    plantuml_summaries: true         # default
prune:
  max_fraction: 0.2                  # defaults
  min_count: 25
library:
  doc_type: book                     # default
  prefer_formats: [epub, pdf, docx]  # default
  roots:
    - path: "~/Library/CloudStorage/OneDrive-Hilti/00 My Workflow/101 Bibliothek/eBooks - calibre"
```

Unknown keys inside `mirrors`, `prune` and `library` are errors, so typos do not go unnoticed. `type: filesystem` connectors are no longer supported; move them to `mirrors:`.

The older `ollama: {base_url, embedding_model}` section is still read when `embedding:` is absent. `OLLAMA_BASE_URL` and `COLIBRI_EMBEDDING_MODEL` override both. Changing the model triggers a full re-embed on the next index run.

## Data Directory

Runtime data is stored under `COLIBRI_HOME` (default: `~/.local/share/colibri/`):

- `metadata.db`: SQLite metadata (schema v8): documents, collections, problems, conversions
- `canonical/<collection>/`: canonical markdown of every document
- `conversions/`: cached markdown of converted files, keyed by the source file's SHA-256
- `index/lancedb/`: the vector and keyword index (one `chunks` table) plus `index_meta.json`
- `write.lock`: lock file held by the command currently changing data

Everything in this directory can be rebuilt from the sources with `colibri reset` followed by `colibri update`. A metadata DB from an older CoLibri version is never modified; commands report it and point to `colibri reset`.

You can relocate all data by setting:

```bash
export COLIBRI_HOME=/path/to/portable/colibri
```

## User Docs

Start here:

- `docs/user/getting-started.md`
- `docs/user/concepts.md`
- `docs/user/configuration.md`
- `docs/user/use-cases.md`
- `docs/user/troubleshooting.md`

## Development

```bash
make check    # Type-check (fast)
make test     # Run tests
make lint     # Clippy linter
make format   # Format code
```

## License

MIT
