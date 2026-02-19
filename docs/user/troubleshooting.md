# Troubleshooting

## `colibri doctor` fails

- Ensure `~/.config/colibri/config.yaml` exists and is readable.
- Run `colibri doctor --json` to see the specific error.

## Plugins fail with “manifest missing” or “invalid manifest”

- Run `colibri plugins jobs --validate-manifests`.
- Confirm `plugins.jobs[].manifest` paths (absolute or relative to your `config.yaml` directory).

## Converters missing (pandoc/docling/soffice)

- `filesystem_documents` needs external tools depending on which formats you ingest.
- `colibri doctor` warns if it detects missing tools for configured jobs.

## Nothing is queryable / serve refuses to start

- Run `colibri profiles` and `colibri generations list`.
- Ensure the active generation has at least one profile in a serve-ready lifecycle state.

