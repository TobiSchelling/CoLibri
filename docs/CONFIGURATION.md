# Configuration

This document describes CoLibri's configuration file, data directory layout, and the on-disk state used for indexing and change tracking.

## Config File Location

- Config file: `~/.config/colibri/config.yaml`
- Show path: `colibri config --path`
- Edit config: `colibri config --edit`
- Manage sources via TUI: `colibri config --tui`

## Configuration Schema

CoLibri uses a top-level `sources:` list. Each source is a folder that will be indexed according to its own mode and settings.

Example (`~/.config/colibri/config.yaml`):

```yaml
# Content sources (per-folder profiles)
sources:
  - path: /Users/you/Documents/CoLibri/Books
    mode: static          # index once, skip unless --force
    doc_type: book
    name: Books
    extensions: [".md"]
  - path: /Users/you/Documents/Notes
    mode: incremental     # track changes, re-index modified/new
    doc_type: note
    name: Notes
    extensions: [".md", ".yaml", ".yml"]
  - path: /Users/you/Documents/DailyNotes
    mode: append_only     # only index new files, never re-check
    doc_type: journal
    name: DailyNotes
  - path: /Users/you/Documents/Drafts
    mode: disabled        # excluded from indexing
    doc_type: note
    name: Drafts

# Data directory for index and state files (default: ~/.local/share/colibri)
data:
  directory: null  # null = XDG default

# Vector index subdirectory (relative to data directory)
index:
  directory: lancedb

# Ollama settings
ollama:
  base_url: http://localhost:11434
  embedding_model: nomic-embed-text

# Retrieval settings
retrieval:
  top_k: 10
  similarity_threshold: 0.3

# Global chunking defaults (can be overridden per source)
chunking:
  chunk_size: 3000
  chunk_overlap: 200

# Translation settings (optional)
translation:
  model: null
```

### Source Profile Fields

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `path` | Yes | - | Absolute path to a source root directory |
| `mode` | No | `incremental` | Indexing mode (`static`, `incremental`, `append_only`, `disabled`) |
| `doc_type` | No | `note` | Default document type (overridden by frontmatter `type` field) |
| `extensions` | No | `[".md"]` | File extensions to index (e.g. `[".md", ".yaml"]`) |
| `name` | No | basename of path | Display name used by `colibri index --folder <name>` |
| `chunk_size` | No | global default | Per-source chunk size override |
| `chunk_overlap` | No | global default | Per-source chunk overlap override |

## Environment Variables

Environment variables override config file settings:

| Variable | Description |
|----------|-------------|
| `COLIBRI_LIBRARY_PATH` | Override derived library root (for legacy setups) |
| `COLIBRI_DATA_DIR` | Data directory override (index, manifest, catalog, change journal) |
| `XDG_DATA_HOME` | Impacts default data dir if `COLIBRI_DATA_DIR` is not set |
| `OLLAMA_BASE_URL` | Ollama API URL |
| `COLIBRI_EMBEDDING_MODEL` | Embedding model name |

## Data Directory Layout

Default: `~/.local/share/colibri/` (or `COLIBRI_DATA_DIR`)

```
~/.local/share/colibri/
├── lancedb/
│   └── index_meta.json       # Index metadata (schema/revision/digest + summary)
├── manifest.json             # Change tracking manifest (v2 keys are namespaced)
├── index_changes.jsonl       # Change journal (revisions; used by `colibri changes`)
└── doc_catalog.json          # Per-document catalog (for fast `capabilities`)
```

### Notes On `manifest.json` (v2)

Manifest keys are namespaced per source to avoid cross-source collisions:

- key format: `<source_id>:<rel_path>`
- example: `cc043087f110:SomeBook.md`

This is important when multiple sources contain files with the same relative names.

## Imported Book Format (Markdown)

Imported books are converted to markdown with YAML frontmatter:

```markdown
---
title: "Software Testing with Generative AI"
type: book
source_epub: "original-filename.epub"
imported: "2026-01-28T12:00:00"
author: "Mark Winteringham"
publisher: "Manning"
tags:
  - book
  - imported
  - epub
---

# Software Testing with Generative AI

> [!info] Source
> Imported from `original-filename.epub` on 2026-01-28

[Book content in markdown...]
```

