# Use Cases

## 1) A Markdown Git repository (changes daily)

Goal: continuously ingest a local git checkout into canonical store and re-index safely.

Config:

```yaml
plugins:
  jobs:
    - id: arch_repo
      manifest: ~/.local/share/colibri/plugins/bundled/filesystem_documents/plugin_manifest.json
      enabled: true
      config:
        root_path: ~/repos/architecture-docs
        classification: internal
        mode: incremental
        include_extensions: [".md", ".markdown"]
```

Daily workflow:

```bash
cd ~/repos/architecture-docs && git pull
colibri sync
```

## 2) A local library of EPUB/PDF books (mostly static)

Goal: ingest a book folder (EPUB/PDF) with high-quality conversion to Markdown.

Config:

```yaml
plugins:
  jobs:
    - id: library
      manifest: ~/.local/share/colibri/plugins/bundled/filesystem_documents/plugin_manifest.json
      enabled: true
      config:
        root_path: ~/Library/Books
        classification: confidential
        include_extensions: [".epub", ".pdf"]
```

Initial build:

```bash
colibri sync --force
```

## 3) Single PPTX/DOCX files in a shared folder (mixed quality, frequently added)

Goal: treat a “drop zone” folder as an ingestion source.

Config:

```yaml
plugins:
  jobs:
    - id: drop_zone
      manifest: ~/.local/share/colibri/plugins/bundled/filesystem_documents/plugin_manifest.json
      enabled: true
      config:
        root_path: ~/Documents/drop-zone
        classification: internal
        include_extensions: [".pptx", ".docx", ".pdf", ".md"]
        pptx_backend: soffice_pdf_docling
```

## 4) Zephyr Scale test cases (structured, API-backed)

Goal: export test cases to Markdown (frontmatter + steps), then ingest into canonical store.

Config:

```yaml
plugins:
  jobs:
    - id: zephyr_ctslab
      manifest: ~/GIT_ROOT/GIT_HUB/CoLibri/plugins/bundled/zephyr_scale/plugin_manifest.json
      enabled: true
      config:
        project_key: CTSLAB
        classification: internal
        zephyr_export_cmd: /path/to/zephyr-export
        token_env: ZEPHYR_API_TOKEN
```

Run:

```bash
export ZEPHYR_API_TOKEN="..."
colibri sync
```
