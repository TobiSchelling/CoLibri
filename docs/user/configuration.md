# Configuration

CoLibri reads `~/.config/colibri/config.yaml` by default.

Override for experiments or multi-profile setups:

```bash
export COLIBRI_CONFIG_PATH=/path/to/config.yaml
```

## Canonical corpus

CoLibri indexes a **managed canonical markdown store** under `COLIBRI_HOME/canonical`.

To bring content in, configure ingestion **plugin jobs** under `plugins.jobs` and run `colibri plugins sync-all`.

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

## Plugins (recommended ingestion path)

Plugin jobs are configured under `plugins.jobs`.

Notes:
- If `id` is omitted, CoLibri assigns `job_1`, `job_2`, ...
- `manifest` may be absolute or relative to the directory containing `config.yaml`.
- `config` must be an object and is passed to the plugin unchanged.

Example:

```yaml
plugins:
  jobs:
    - id: fs_repo
      manifest: ~/GIT_ROOT/GIT_HUB/CoLibri/plugins/examples/filesystem_markdown/plugin_manifest.json
      enabled: true
      config:
        root_path: ~/repos/architecture-docs
        classification: internal

    - id: books
      manifest: ~/GIT_ROOT/GIT_HUB/CoLibri/plugins/examples/filesystem_documents/plugin_manifest.json
      enabled: true
      config:
        root_path: ~/Library/Books
        classification: confidential
        include_extensions: [".pdf", ".epub"]
```

Inspect jobs:

```bash
colibri plugins jobs
colibri plugins jobs --validate-manifests
```

## Secrets

Prefer passing secrets via environment variables referenced by plugin config (for example `token_env`),
not inline `config.yaml`.
