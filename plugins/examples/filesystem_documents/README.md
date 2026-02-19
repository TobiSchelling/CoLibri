# filesystem_documents Plugin

Ingests a local folder containing mixed document formats and emits canonical CoLibri document envelopes as JSONL.

Supported extensions:
- Markdown: `.md`, `.markdown`
- Documents: `.pdf`, `.epub`, `.docx`, `.pptx`

Quality-first conversion backends (uses locally-installed tools):
- PDF: `docling` (CLI)
- EPUB/DOCX/PPTX: `pandoc` (CLI)

Optional enrichment:
- Extracts a small text summary from fenced `plantuml` blocks to improve semantic retrieval.

## Run

```bash
colibri plugins run \
  --manifest plugins/examples/filesystem_documents/plugin_manifest.json \
  --config-json '{"root_path":"/path/to/folder","classification":"internal"}'
```

## Ingest into canonical store

```bash
colibri plugins ingest \
  --manifest plugins/examples/filesystem_documents/plugin_manifest.json \
  --config-json '{"root_path":"/path/to/folder","classification":"internal"}'
```

Then index:

```bash
colibri index --canonical --force
```

## Tooling prerequisites

Install the converters if missing:

```bash
brew install pandoc
pipx install docling  # or: python3 -m pip install --user docling
```

