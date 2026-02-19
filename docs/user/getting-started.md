# Getting Started

This guide explains the smallest “happy path” to get from **raw documents** to **search results**.

## Mental model (3 steps)

1. **Ingest**: bring content in and convert it to canonical Markdown (usually via plugins).
2. **Index**: chunk + embed canonical Markdown into an index generation.
3. **Serve/Search**: query that active generation via CLI or MCP.

## 1) Install prerequisites

- SQLite CLI is required (`sqlite3` on PATH).
- A local embedding runtime is recommended (example: Ollama).

Verify:

```bash
colibri doctor
```

## 2) Configure your first source

CoLibri reads `~/.config/colibri/config.yaml`.

Start with a single plugin job (recommended path):

```yaml
plugins:
  jobs:
    - id: my_docs
      manifest: ~/GIT_ROOT/GIT_HUB/CoLibri/plugins/examples/filesystem_markdown/plugin_manifest.json
      enabled: true
      config:
        root_path: ~/Documents/knowledge
        classification: internal
```

## 3) Ingest + index + search

Run incremental ingestion for all enabled plugin jobs, then index canonical store:

```bash
colibri plugins sync-all --index-canonical --index-force
colibri search "what is this repo about?"
```

## Optional: generations (safe cutovers)

Create and build a new generation without switching traffic:

```bash
colibri generations create gen_candidate
colibri index --canonical --generation gen_candidate --force
colibri generations activate gen_candidate
```

