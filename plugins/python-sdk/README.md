# CoLibri Python Plugin SDK (Skeleton)

This folder is the starter SDK layout for ingestion plugins.

Current contract:
- Plugins read request JSON from stdin: `{ "config": {...}, "cursor": <json|null> }`.
- Plugins emit one JSON document-envelope per line (JSONL) on stdout.
- Optional cursor checkpoint lines are supported: `{ "type": "cursor", "cursor": {...} }`.
- Envelope format: `plugins/spec/document-envelope.schema.json`.
- Plugin manifest format: `plugins/spec/plugin-manifest.schema.json`.

Minimal structure:
- `colibri_plugin_sdk/` — shared helpers/types.
- `examples/` — runnable examples used for host integration testing.

This skeleton intentionally keeps dependencies minimal; add runtime-specific libraries per plugin.
