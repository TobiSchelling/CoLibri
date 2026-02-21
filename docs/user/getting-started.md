# Getting Started

This guide explains the smallest “happy path” to get from **raw documents** to **search results**.

## Mental model (3 steps)

1. **Ingest**: bring content in and convert it to canonical Markdown (usually via plugins).
2. **Index**: chunk + embed canonical Markdown into the vector index.
3. **Serve/Search**: query the index via CLI or MCP.

## 1) Install prerequisites

- SQLite CLI is required (`sqlite3` on PATH).
- A local embedding runtime is recommended (example: Ollama).

Verify:

```bash
colibri doctor
```

## Optional: run the bootstrap wizard

`colibri bootstrap` can write a starter config, initialize storage, and print the exact commands
needed to install missing tools (including plugin converters).

```bash
colibri bootstrap
```

Notes:
- `colibri bootstrap` installs bundled plugin manifests into `~/.local/share/colibri/plugins/` (or your configured `COLIBRI_HOME`).
- The config examples below assume the default `COLIBRI_HOME`.

## 2) Configure your first source

CoLibri reads `~/.config/colibri/config.yaml`.

Start with a single plugin job (recommended path):

```yaml
plugins:
  jobs:
    - id: my_docs
      manifest: ~/.local/share/colibri/plugins/bundled/filesystem_documents/plugin_manifest.json
      enabled: true
      config:
        root_path: ~/Documents/knowledge
        classification: internal
        include_extensions: [".md", ".markdown"]
```

## 3) Ingest + index + search

Run incremental ingestion for all enabled plugin jobs, then index canonical store:

```bash
colibri sync --force
colibri search "what is this repo about?"
```
