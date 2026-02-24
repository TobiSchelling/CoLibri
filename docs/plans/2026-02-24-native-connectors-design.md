# Native Connectors — Design Document

**Date:** 2026-02-24
**Status:** Approved

## Problem

CoLibri's plugin system uses subprocess spawning, JSONL protocol, Python SDK, and
manifest files to ingest content. All plugins ship from the same repo. The
indirection adds ~2,000 lines of infrastructure without real isolation benefit.

## Decision

Replace the plugin system with **native Rust connectors**. Remove the plugin
infrastructure entirely. Keep the canonical store and indexer pipeline unchanged.

## Architecture

```
config.yaml (connectors[])
       |
       v
Connector trait  ──→  Vec<DocumentEnvelope>  ──→  canonical_store  ──→  indexer  ──→  LanceDB
```

Each connector is a Rust module implementing a trait. No subprocesses, no JSONL,
no manifests.

```rust
pub trait Connector: Send + Sync {
    fn id(&self) -> &str;
    async fn sync(&self) -> Result<Vec<DocumentEnvelope>>;
}
```

### Initial Connectors

- **`FilesystemConnector`** — scans directories, converts documents to Markdown
- **Zephyr** — separate follow-up (not part of this migration)

## Config Change

### Before (plugins)

```yaml
plugins:
  jobs:
    - id: books
      manifest: /path/to/plugin_manifest.json
      enabled: true
      config:
        root_path: ~/Books
        include_extensions: [".pdf", ".md"]
        classification: internal
```

### After (connectors)

```yaml
connectors:
  - type: filesystem
    id: books
    enabled: true
    root_path: ~/Books
    include_extensions: [".pdf", ".md"]
    exclude_globs: ["**/.git/**"]
    mode: incremental
    doc_type: book
    classification: internal
```

Flat config — no nested `config:` block, no `manifest:` path. The `type` field
dispatches to the right Rust connector.

## Filesystem Connector

### Conversion Pipeline

| Extension         | Primary      | Fallback  | External Tool             |
|-------------------|--------------|-----------|---------------------------|
| `.md`, `.markdown`| Read directly| —         | None                      |
| `.pdf`            | `docling`    | —         | `docling` (pipx)          |
| `.epub`           | `pandoc`     | —         | `pandoc` (brew)           |
| `.docx`           | `pandoc`     | —         | `pandoc` (brew)           |
| `.pptx`           | `markitdown` | `pandoc`  | `markitdown` (pipx) / `pandoc` (brew) |

No `soffice`/LibreOffice in the pipeline.

### Incremental Mode

Snapshot-only at the trait level (no cursor persistence). The filesystem
connector uses **mtime checks** internally — compares file mtimes against
last-known state in metadata.db to skip unchanged files.

### PlantUML Enrichment

The Python plugin enriches Markdown files containing PlantUML blocks with
searchable text summaries. The Rust connector ports this logic: regex-based
extraction of entities and relations from PlantUML blocks, inserted as HTML
comments.

### Error Handling

Conversion failures for individual files are logged to stderr and skipped. The
connector returns all successfully converted documents.

## What Gets Deleted (~2,500 lines)

| File/Directory                  | Lines  | Reason                                    |
|---------------------------------|--------|-------------------------------------------|
| `src/plugin_host.rs`            | ~1,028 | Subprocess spawning, JSONL, manifest       |
| `src/cli/plugins.rs`            | ~1,105 | All plugin CLI commands                    |
| `src/bundled_plugins.rs`        | ~167   | Bundled plugin extraction                  |
| `plugins/` directory            | entire | Python plugins, SDK, JSON specs            |
| `src/cli/mod.rs` PluginCommands | ~70    | Plugin command enum                        |
| `src/main.rs` plugin routing    | ~80    | Plugin command dispatch                    |
| `config.rs` plugin sections     | ~150   | `PluginsConfig`, `PluginJob`, resolvers    |
| `metadata_store.rs` sync_state  | ~120   | `sync_state` table + cursor methods        |

## What Gets Kept (refactored)

| Component              | Current Location   | New Location      | Change                |
|------------------------|--------------------|-------------------|-----------------------|
| `DocumentEnvelope`     | `plugin_host.rs`   | `src/envelope.rs` | Extracted to own module|
| Envelope validation    | `plugin_host.rs`   | `src/envelope.rs` | Extracted with structs |
| `canonical_store.rs`   | stays              | stays             | Import path change only|
| `metadata_store.rs`    | stays              | stays             | Remove sync_state table|
| `indexer.rs`           | stays              | stays             | Unchanged             |
| `mcp.rs`               | stays              | stays             | Unchanged             |

## What Gets Created

| File                          | Purpose                                           |
|-------------------------------|---------------------------------------------------|
| `src/envelope.rs`             | DocumentEnvelope structs + validation              |
| `src/connectors/mod.rs`       | Connector trait, ConnectorConfig enum, dispatch     |
| `src/connectors/filesystem.rs`| Filesystem connector (Rust port of plugin.py)      |
| `src/cli/connectors.rs`       | CLI: `colibri connectors list`, sync integration   |

## CLI Changes

### Removed

All `colibri plugins *` commands (run, ingest, sync, sync-all, jobs, configure,
state list/show/reset).

### New / Changed

| Command                                     | Purpose                            |
|---------------------------------------------|------------------------------------|
| `colibri connectors list`                   | Show configured connectors         |
| `colibri sync [--connector <id>] [--dry-run] [--index]` | Run connectors + ingest |

`colibri sync` becomes a first-class command rather than a hidden wrapper.

## Scope Boundaries

This design does NOT include:

- Zephyr connector (follow-up task)
- Cursor/incremental sync at the trait level
- Plugin system preservation for future third-party use
- LLM-based document aggregation
- Configure hooks (interactive setup)
- New Rust dependencies beyond what's needed for file I/O and subprocess calls
