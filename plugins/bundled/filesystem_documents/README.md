# filesystem_documents Plugin

Ingests a local folder containing mixed document formats and emits canonical CoLibri document envelopes as JSONL.

Supported extensions:
- Markdown: `.md`, `.markdown`
- Documents: `.pdf`, `.epub`, `.docx`, `.pptx`

Quality-first conversion backends (uses locally-installed tools):
- PDF: `docling` (CLI)
- EPUB/DOCX: `pandoc` (CLI)
- PPTX (configurable via `pptx_backend`):
  - `soffice_pdf_docling` (default): `soffice` → PDF → `docling`
  - `pandoc`: `pandoc` directly
  - `python_pptx`: `python-pptx` (text-only, fast, no layout)
  - `markitdown`: `markitdown` (quality varies by deck)

Optional enrichment:
- Extracts a small text summary from fenced `plantuml` blocks to improve semantic retrieval.

## Run

Bootstrap the local venv (needed for optional Python backends `python_pptx` and `markitdown`):

```bash
plugins/bundled/filesystem_documents/bootstrap.sh
```

```bash
colibri plugins run \
  --manifest plugins/bundled/filesystem_documents/plugin_manifest.json \
  --config-json '{"root_path":"/path/to/folder","classification":"internal"}'
```

## Ingest into canonical store

```bash
colibri plugins ingest \
  --manifest plugins/bundled/filesystem_documents/plugin_manifest.json \
  --config-json '{"root_path":"/path/to/folder","classification":"internal"}'
```

Then index:

```bash
colibri index --force
```

## Tooling prerequisites

Install the converters if missing:

```bash
brew install pandoc
brew install poppler
brew install --cask libreoffice   # provides `soffice` used by the default PPTX backend

# docling CLI (must be on PATH, install method up to you)
pipx install docling
```
