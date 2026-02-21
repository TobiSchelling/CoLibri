# Configuration

CoLibri reads `~/.config/colibri/config.yaml` by default.

Override for experiments or multi-profile setups:

```bash
export COLIBRI_CONFIG_PATH=/path/to/config.yaml
```

## Canonical corpus

CoLibri indexes a **managed canonical markdown store** under `COLIBRI_HOME/canonical`.

To bring content in, configure ingestion **plugin jobs** under `plugins.jobs` and run `colibri sync`.

## Embeddings + routing

Embedding profiles define how embeddings are produced; routing maps classification to a profile.

```yaml
embeddings:
  default_profile: local_secure
  profiles:
    - id: local_secure
      provider: ollama
      endpoint: http://localhost:11434
      model: bge-m3
      locality: local

routing:
  classification_profiles:
    restricted: local_secure
    confidential: local_secure
    internal: local_secure
    public: local_secure
```

## Sync jobs (recommended ingestion path)

Sync jobs are configured under `plugins.jobs` (each job references a plugin manifest).

Notes:
- If `id` is omitted, CoLibri assigns `job_1`, `job_2`, ...
- `manifest` may be absolute or relative to the directory containing `config.yaml`.
- `config` must be an object and is passed to the plugin unchanged.
- Bundled plugin manifests are installed under `COLIBRI_HOME/plugins/` when running `colibri bootstrap` (default `COLIBRI_HOME` is `~/.local/share/colibri`).

Example:

```yaml
plugins:
  jobs:
    - id: fs_repo
      manifest: ~/.local/share/colibri/plugins/bundled/filesystem_documents/plugin_manifest.json
      enabled: true
      config:
        root_path: ~/repos/architecture-docs
        classification: internal
        include_extensions: [".md", ".markdown"]

    - id: books
      manifest: ~/.local/share/colibri/plugins/bundled/filesystem_documents/plugin_manifest.json
      enabled: true
      config:
        root_path: ~/Library/Books
        classification: confidential
        include_extensions: [".pdf", ".epub"]
```

Inspect jobs:

```bash
colibri sync --dry-run --json
colibri doctor --json
```

## Secrets

Prefer passing secrets via environment variables referenced by plugin config (for example `token_env`),
not inline `config.yaml`.
