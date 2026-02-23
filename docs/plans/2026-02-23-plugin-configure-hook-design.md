# Plugin Configure Hook — Design Document

**Date:** 2026-02-23
**Status:** Approved

## Problem

Plugins like Zephyr Scale require interactive setup (token validation, project
discovery, folder browsing) before they can sync. Today, users must manually
write JSON config into `config.yaml`. There is no mechanism for a plugin to
expose its own interactive configuration flow.

## Decision

Add an optional **configure hook** to the plugin manifest. CoLibri delegates
interactive setup to the plugin's own TUI via a **config file exchange**
protocol.

## Protocol

```
colibri plugins configure <job_id>
       |
       +-- 1. Resolve job from config.yaml (manifest path + current config)
       +-- 2. Load manifest -> check for "configure.entrypoint"
       +-- 3. Write current config to temp file: /tmp/colibri-cfg-XXXX.json
       +-- 4. Spawn: python3 configure.py /tmp/colibri-cfg-XXXX.json
       |      (stdin/stdout/stderr: INHERITED -- full TTY access)
       +-- 5. Wait for exit
       +-- 6. Read /tmp/colibri-cfg-XXXX.json (plugin overwrites it)
       +-- 7. Validate: must be a JSON object
       +-- 8. Merge into plugins.jobs[<job_id>].config in config.yaml
       +-- 9. Clean up temp file
```

### Exit Codes

| Code | Meaning                        |
|------|--------------------------------|
| 0    | Success — read the file        |
| 1    | User cancelled — no changes    |
| 2+   | Error                          |

### What the Configure Entrypoint Does

1. Read the JSON file (current config, may be `{}` on first run)
2. Run its own TUI/interactive flow
3. Write the resulting config back to the same file path
4. Exit 0

Secrets (API tokens) are **never written** to the config file. The plugin
should only persist the env var name (e.g., `token_env: "ZEPHYR_API_TOKEN"`).

## Manifest Schema Change

Add an optional `configure` field:

```json
{
  "schema_version": 1,
  "plugin_id": "zephyr_scale",
  "version": "0.1.0",
  "runtime": "python",
  "entrypoint": "plugin.py",
  "capabilities": { "snapshot": true, "incremental": true, "webhook": false },
  "configure": {
    "entrypoint": "configure.py"
  },
  "requirements": { ... }
}
```

The `configure.entrypoint` is resolved relative to the manifest directory
(same rules as the main `entrypoint`).

## CoLibri-Side Changes

### Files to Modify

| File | Change |
|------|--------|
| `plugins/spec/plugin-manifest.schema.json` | Add optional `configure` property |
| `src/plugin_host.rs` | Add `configure` to `PluginManifest`, add `run_plugin_configure()` |
| `src/cli/mod.rs` | Add `Configure` variant to `PluginCommands` |
| `src/main.rs` | Route `Configure` to handler |
| `src/cli/plugins.rs` | New `configure()` function |
| `src/config.rs` | Add `update_plugin_job_config()` to write back to YAML |

### New CLI Command

```bash
colibri plugins configure <job_id> [--json]
```

- `<job_id>` — matches `plugins.jobs[].id` in config.yaml
- `--json` — output result summary as JSON

### `run_plugin_configure()` vs `run_plugin_manifest()`

The configure runner differs from the data runner:

| Aspect | Data pipeline | Configure hook |
|--------|--------------|----------------|
| stdin | Piped (JSON request) | Inherited (TTY) |
| stdout | Piped (JSONL envelopes) | Inherited (TTY) |
| stderr | Piped (captured) | Inherited (TTY) |
| Arguments | None | `[config_file_path]` |
| Output | Parsed envelopes | Read file on exit |

### Config Write-Back

Use `serde_yaml` to:

1. Read config.yaml as `serde_yaml::Value`
2. Navigate to `plugins.jobs[idx].config`
3. Replace with the new config (JSON -> YAML value)
4. Serialize and write back

Comments are not preserved (acceptable since `config` is machine-generated).

## ZephyrFrontend-Side Changes

New file: `plugins/bundled/zephyr_scale/configure.py`

1. Accept file path as `sys.argv[1]`
2. Read current config from file
3. Run existing TUI screens (LoginScreen, ProjectSelectScreen, FolderBrowseScreen)
4. Write resulting config to file (project_key, folder, classification, etc.)
5. Never write secrets — only `token_env` reference

The existing Textual TUI screens can be reused. Main change: redirect config
persistence from `~/.config/zephyr-export/config.yaml` to the provided path.

## Scope Boundaries

This design does NOT include:

- Generic config schema or form rendering
- Secret storage or keychain integration
- Bidirectional IPC between CoLibri and the plugin
- New Rust dependencies
- Breaking changes to existing plugins (configure is optional)
