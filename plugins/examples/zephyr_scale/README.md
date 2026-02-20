# zephyr_scale Plugin (Zephyr Scale Cloud → Markdown → Canonical Store)

Ingests Zephyr Scale Cloud test cases by calling the `zephyr-export` CLI (proof-of-concept exporter),
then wraps the generated Markdown into CoLibri document envelopes.

This plugin is intentionally **markdown-only**: it does not store original payloads, only the derived
markdown representation (which already contains rich YAML frontmatter metadata).

## Prerequisites

- `zephyr-export` CLI installed and reachable via `--config-json.zephyr_export_cmd` (default: `zephyr-export`)
- A Zephyr Scale API token available via environment variable (default: `ZEPHYR_API_TOKEN`)

## Run (one-shot)

```bash
export ZEPHYR_API_TOKEN="…"

colibri plugins run \
  --manifest plugins/examples/zephyr_scale/plugin_manifest.json \
  --config-json '{"project_key":"CTSLAB","classification":"internal"}'
```

## Ingest into canonical store

```bash
export ZEPHYR_API_TOKEN="…"

colibri plugins ingest \
  --manifest plugins/examples/zephyr_scale/plugin_manifest.json \
  --config-json '{"project_key":"CTSLAB","classification":"internal"}'
```

Then index:

```bash
colibri index --force
```

## Config notes

- `folder` scopes export to a specific Zephyr folder path. When set, tombstone emission is disabled
  automatically to avoid deleting content outside the selected scope.
- `emit_tombstones` only applies to full-project exports (no `folder`).
- Prefer `token_env` over `token` to avoid storing secrets in config files.
