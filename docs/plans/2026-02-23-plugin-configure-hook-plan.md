# Plugin Configure Hook — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Allow plugins to expose interactive TUI-based configuration via a new `colibri plugins configure <job_id>` command using a config file exchange protocol.

**Architecture:** CoLibri writes the current job config to a temp JSON file, spawns the plugin's configure entrypoint with inherited TTY and the file path as argument, then reads the updated config back and merges it into config.yaml.

**Tech Stack:** Rust (clap CLI, serde_yaml, serde_json, tokio::process), Python (plugin-side TUI)

**Design doc:** `docs/plans/2026-02-23-plugin-configure-hook-design.md`

---

### Task 1: Add `configure` to Plugin Manifest Schema

Extend the JSON schema and Rust struct to accept an optional `configure` field.

**Files:**
- Modify: `plugins/spec/plugin-manifest.schema.json`
- Modify: `src/plugin_host.rs:60-71`

**Step 1: Update the JSON schema**

In `plugins/spec/plugin-manifest.schema.json`, add a `configure` property to the `properties` object (after `requirements`):

```json
    "configure": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "entrypoint": {
          "type": "string",
          "minLength": 1
        }
      },
      "required": ["entrypoint"]
    }
```

**Step 2: Add `configure` field to PluginManifest struct**

In `src/plugin_host.rs`, add a new struct and field to `PluginManifest`:

```rust
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PluginConfigureHook {
    pub entrypoint: String,
}
```

Add to the `PluginManifest` struct (after `requirements`):

```rust
    pub configure: Option<PluginConfigureHook>,
```

**Step 3: Verify existing tests still pass**

Run: `cargo test manifest_tests -v`
Expected: All existing manifest tests pass (the new field is optional so `deny_unknown_fields` doesn't break anything).

**Step 4: Commit**

```bash
git add plugins/spec/plugin-manifest.schema.json src/plugin_host.rs
git commit -m "feat(plugins): add optional configure hook to manifest schema"
```

---

### Task 2: Add `run_plugin_configure()` to Plugin Host

Create the function that spawns the configure entrypoint with inherited TTY.

**Files:**
- Modify: `src/plugin_host.rs`

**Step 1: Write a test for the configure runner**

Add at the bottom of the `tests` module in `src/plugin_host.rs`:

```rust
#[cfg(test)]
mod configure_tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn resolve_configure_entrypoint_relative() {
        let manifest_dir = Path::new("/tmp/plugins/myplugin");
        let resolved = resolve_entrypoint(manifest_dir, "configure.py");
        assert_eq!(resolved, PathBuf::from("/tmp/plugins/myplugin/configure.py"));
    }

    #[test]
    fn resolve_configure_entrypoint_absolute() {
        let manifest_dir = Path::new("/tmp/plugins/myplugin");
        let resolved = resolve_entrypoint(manifest_dir, "/usr/local/bin/configure");
        assert_eq!(resolved, PathBuf::from("/usr/local/bin/configure"));
    }
}
```

Run: `cargo test configure_tests -v`
Expected: PASS (resolve_entrypoint is already implemented and works for both cases).

**Step 2: Write `run_plugin_configure()`**

Add this public function to `src/plugin_host.rs` (after `run_plugin_manifest()`):

```rust
/// Result of running a plugin's configure hook.
#[derive(Debug)]
pub struct PluginConfigureResult {
    pub plugin_id: String,
    pub exit_code: i32,
    pub cancelled: bool,
}

/// Run a plugin's configure entrypoint with inherited TTY.
///
/// The plugin receives the config file path as its first CLI argument and has
/// full terminal access (stdin/stdout/stderr inherited). On exit 0, the caller
/// reads back the file. Exit 1 means user cancelled.
pub async fn run_plugin_configure(
    manifest_path: &Path,
    config_file_path: &Path,
) -> Result<PluginConfigureResult, ColibriError> {
    let manifest = load_plugin_manifest(manifest_path)?;

    let hook = manifest.configure.ok_or_else(|| {
        ColibriError::Config(format!(
            "Plugin '{}' does not declare a configure hook",
            manifest.plugin_id
        ))
    })?;

    let manifest_dir = manifest_path.parent().unwrap_or(Path::new("."));
    let entrypoint_path = resolve_entrypoint(manifest_dir, &hook.entrypoint);

    if !entrypoint_path.exists() {
        return Err(ColibriError::Config(format!(
            "Configure entrypoint not found: {}",
            entrypoint_path.display()
        )));
    }

    let mut cmd = build_command(&manifest.runtime, &entrypoint_path)?;
    cmd.arg(config_file_path);
    cmd.stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());

    let status = cmd.spawn()
        .map_err(|e| {
            ColibriError::Config(format!(
                "Failed to launch configure hook for '{}': {e}",
                manifest.plugin_id
            ))
        })?
        .wait()
        .await
        .map_err(|e| {
            ColibriError::Config(format!(
                "Failed waiting for configure hook '{}': {e}",
                manifest.plugin_id
            ))
        })?;

    let exit_code = status.code().unwrap_or(-1);

    if exit_code >= 2 {
        return Err(ColibriError::Config(format!(
            "Plugin '{}' configure hook failed (exit code {exit_code})",
            manifest.plugin_id
        )));
    }

    Ok(PluginConfigureResult {
        plugin_id: manifest.plugin_id,
        exit_code,
        cancelled: exit_code == 1,
    })
}
```

**Step 3: Run all plugin host tests**

Run: `cargo test plugin_host -v && cargo test manifest_tests -v && cargo test configure_tests -v`
Expected: All PASS.

**Step 4: Commit**

```bash
git add src/plugin_host.rs
git commit -m "feat(plugins): add run_plugin_configure() with inherited TTY"
```

---

### Task 3: Add `update_plugin_job_config()` to Config

Write the function that updates a specific job's config in config.yaml.

**Files:**
- Modify: `src/config.rs`

**Step 1: Write a test for config update**

Add a test module at the bottom of `src/config.rs`:

```rust
#[cfg(test)]
mod config_update_tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn update_plugin_job_config_replaces_config_block() {
        let yaml_content = r#"
sources: []
plugins:
  jobs:
    - id: myjob
      manifest: /tmp/manifest.json
      enabled: true
      config:
        old_key: old_value
"#;
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), yaml_content).unwrap();

        let new_config = serde_json::json!({"new_key": "new_value"});
        update_plugin_job_config(tmp.path(), "myjob", &new_config).unwrap();

        let updated = std::fs::read_to_string(tmp.path()).unwrap();
        let doc: serde_yaml::Value = serde_yaml::from_str(&updated).unwrap();
        let jobs = doc["plugins"]["jobs"].as_sequence().unwrap();
        let job = &jobs[0];
        assert_eq!(job["config"]["new_key"].as_str().unwrap(), "new_value");
        assert!(job["config"].get("old_key").is_none());
    }

    #[test]
    fn update_plugin_job_config_errors_for_unknown_job() {
        let yaml_content = r#"
plugins:
  jobs:
    - id: myjob
      manifest: /tmp/manifest.json
      config: {}
"#;
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), yaml_content).unwrap();

        let new_config = serde_json::json!({"key": "value"});
        let err = update_plugin_job_config(tmp.path(), "nonexistent", &new_config);
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("nonexistent"));
    }
}
```

**Step 2: Add `tempfile` dependency**

Run: `cargo add tempfile --dev`

(tempfile is only needed for tests; the main code uses `std::env::temp_dir()` + random filename or `tempfile` at the call site in plugins.rs.)

Actually, check if `tempfile` is already a dev dependency. If not:

```bash
# In Cargo.toml, add under [dev-dependencies]:
tempfile = "3"
```

**Step 3: Write `update_plugin_job_config()`**

Add this public function to `src/config.rs`:

```rust
/// Update a plugin job's config in the YAML config file.
///
/// Reads the config file, finds the job by ID, replaces its `config` block
/// with the new JSON value, and writes back. Does not preserve YAML comments.
pub fn update_plugin_job_config(
    config_path: &Path,
    job_id: &str,
    new_config: &serde_json::Value,
) -> anyhow::Result<()> {
    let text = std::fs::read_to_string(config_path)
        .with_context(|| format!("Failed to read config file: {}", config_path.display()))?;

    let mut doc: serde_yaml::Value = serde_yaml::from_str(&text)
        .with_context(|| format!("Failed to parse config YAML: {}", config_path.display()))?;

    // Navigate to plugins.jobs array
    let jobs = doc
        .get_mut("plugins")
        .and_then(|p| p.get_mut("jobs"))
        .and_then(|j| j.as_sequence_mut())
        .ok_or_else(|| anyhow::anyhow!("No plugins.jobs array found in config"))?;

    // Find the job by id
    let job = jobs
        .iter_mut()
        .find(|j| {
            j.get("id")
                .and_then(|id| id.as_str())
                .map(|id| id == job_id)
                .unwrap_or(false)
        })
        .ok_or_else(|| {
            anyhow::anyhow!("Plugin job '{}' not found in config", job_id)
        })?;

    // Convert serde_json::Value -> serde_yaml::Value
    let yaml_config: serde_yaml::Value =
        serde_yaml::to_value(new_config).context("Failed to convert config to YAML")?;

    // Replace the config block
    if let serde_yaml::Value::Mapping(ref mut map) = job {
        map.insert(
            serde_yaml::Value::String("config".into()),
            yaml_config,
        );
    } else {
        anyhow::bail!("Plugin job entry is not a YAML mapping");
    }

    let output = serde_yaml::to_string(&doc).context("Failed to serialize updated config")?;
    std::fs::write(config_path, output)
        .with_context(|| format!("Failed to write config file: {}", config_path.display()))?;

    Ok(())
}
```

**Step 4: Run the tests**

Run: `cargo test config_update_tests -v`
Expected: Both tests PASS.

**Step 5: Commit**

```bash
git add src/config.rs Cargo.toml Cargo.lock
git commit -m "feat(config): add update_plugin_job_config() for YAML write-back"
```

---

### Task 4: Add `Configure` CLI Command

Wire up the new `colibri plugins configure <job_id>` command.

**Files:**
- Modify: `src/cli/mod.rs:229-347` (PluginCommands enum)
- Modify: `src/main.rs:66-130` (command router)
- Modify: `src/cli/plugins.rs` (new handler function)

**Step 1: Add the `Configure` variant to `PluginCommands`**

In `src/cli/mod.rs`, add a new variant to the `PluginCommands` enum (after `Jobs` and before `State`):

```rust
    /// Run a plugin's interactive configuration wizard
    Configure {
        /// Plugin job id (from plugins.jobs[].id in config.yaml)
        job_id: String,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },
```

**Step 2: Route the command in `src/main.rs`**

In the `cli::Commands::Plugins { command } => match command {` block, add a new arm (before the `State` arm):

```rust
            cli::PluginCommands::Configure { job_id, json } => {
                cli::plugins::configure(job_id, json).await
            }
```

**Step 3: Write the `configure()` handler in `src/cli/plugins.rs`**

Add this function:

```rust
pub async fn configure(job_id: String, json: bool) -> anyhow::Result<()> {
    let app_config = load_config_no_bootstrap()?;

    // Find the job
    let job = app_config
        .plugin_jobs
        .iter()
        .find(|j| j.id == job_id)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Plugin job '{}' not found. Run `colibri plugins jobs` to list configured jobs.",
                job_id
            )
        })?;

    // Load manifest to check for configure hook
    let manifest = crate::plugin_host::load_plugin_manifest(&job.manifest)
        .with_context(|| format!("Failed to load manifest for job '{}'", job_id))?;

    if manifest.configure.is_none() {
        anyhow::bail!(
            "Plugin '{}' does not support interactive configuration (no configure hook in manifest).",
            manifest.plugin_id
        );
    }

    // Write current config to temp file
    let tmp_dir = std::env::temp_dir();
    let config_file_path = tmp_dir.join(format!("colibri-cfg-{}.json", job_id));
    let config_json = serde_json::to_string_pretty(&job.config)
        .context("Failed to serialize current config")?;
    std::fs::write(&config_file_path, &config_json)
        .with_context(|| {
            format!(
                "Failed to write temp config file: {}",
                config_file_path.display()
            )
        })?;

    // Run the configure hook with inherited TTY
    let result = crate::plugin_host::run_plugin_configure(&job.manifest, &config_file_path).await;

    // Read back and clean up regardless of outcome
    let readback = match &result {
        Ok(r) if !r.cancelled => {
            let text = std::fs::read_to_string(&config_file_path).ok();
            let _ = std::fs::remove_file(&config_file_path);
            text
        }
        _ => {
            let _ = std::fs::remove_file(&config_file_path);
            None
        }
    };

    let result = result?;

    if result.cancelled {
        if json {
            let payload = serde_json::json!({
                "job_id": job_id,
                "plugin_id": result.plugin_id,
                "status": "cancelled"
            });
            println!("{}", serde_json::to_string_pretty(&payload)?);
        } else {
            eprintln!("Configuration cancelled. No changes were made.");
        }
        return Ok(());
    }

    // Parse and validate new config
    let new_config_text = readback.ok_or_else(|| {
        anyhow::anyhow!("Failed to read back config file after plugin configure hook")
    })?;
    let new_config: Value = serde_json::from_str(&new_config_text)
        .context("Plugin wrote invalid JSON to config file")?;
    if !new_config.is_object() {
        anyhow::bail!("Plugin config must be a JSON object, got: {}", new_config);
    }

    // Write back to config.yaml
    let config_path = crate::config::AppConfig::config_path();
    crate::config::update_plugin_job_config(&config_path, &job_id, &new_config)?;

    if json {
        let payload = serde_json::json!({
            "job_id": job_id,
            "plugin_id": result.plugin_id,
            "status": "ok",
            "config": new_config
        });
        println!("{}", serde_json::to_string_pretty(&payload)?);
    } else {
        eprintln!("Plugin Configured");
        eprintln!("=================");
        eprintln!("Job: {}", job_id);
        eprintln!("Plugin: {}", result.plugin_id);
        eprintln!("Config updated in: {}", config_path.display());
    }

    Ok(())
}
```

**Step 4: Verify it compiles**

Run: `cargo check`
Expected: Success with no errors.

**Step 5: Run all tests**

Run: `cargo test`
Expected: All tests pass (no behavioral changes to existing commands).

**Step 6: Commit**

```bash
git add src/cli/mod.rs src/main.rs src/cli/plugins.rs
git commit -m "feat(plugins): add 'colibri plugins configure <job_id>' command"
```

---

### Task 5: Update Zephyr Scale Plugin Manifest

Declare the configure hook in the Zephyr plugin manifest.

**Files:**
- Modify: `plugins/bundled/zephyr_scale/plugin_manifest.json`

**Step 1: Add configure field to manifest**

In `plugins/bundled/zephyr_scale/plugin_manifest.json`, add after the `capabilities` block:

```json
  "configure": {
    "entrypoint": "configure.py"
  },
```

**Step 2: Verify manifest loads**

Run: `cargo test manifest_tests -v`
Expected: PASS (manifest parsing still works with the new optional field).

**Step 3: Commit**

```bash
git add plugins/bundled/zephyr_scale/plugin_manifest.json
git commit -m "feat(zephyr): declare configure hook in plugin manifest"
```

---

### Task 6: Create Zephyr Scale `configure.py`

Build the configure entrypoint that reuses ZephyrFrontend's TUI.

**Files:**
- Create: `plugins/bundled/zephyr_scale/configure.py`

**Context:** The ZephyrFrontend project at `/Users/tobias/GIT_ROOT/GIT_LAB/Tobias.Schelling/ZephyrFrontend` has a full Textual TUI with screens for login, project selection, and folder browsing. The configure.py script needs to:

1. Read the config JSON file from `sys.argv[1]`
2. Launch the TUI with pre-populated values from the config
3. On completion, write the updated config back to the same file
4. Exit 0 on success, 1 on cancel

**Step 1: Write `configure.py`**

Create `plugins/bundled/zephyr_scale/configure.py`:

```python
#!/usr/bin/env python3
"""Interactive configuration for the Zephyr Scale plugin.

Protocol: receives a JSON config file path as argv[1].
Reads current config, runs TUI, writes updated config back.

Exit codes:
  0  success (config file updated)
  1  user cancelled (config file unchanged)
  2+ error
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def load_config_file(path: Path) -> dict:
    """Read the JSON config file provided by CoLibri."""
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    return json.loads(text)


def save_config_file(path: Path, config: dict) -> None:
    """Write the updated config back to the exchange file."""
    path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run_interactive(current_config: dict) -> dict | None:
    """Run the interactive TUI and return updated config, or None if cancelled.

    This imports from the zephyr_export package (ZephyrFrontend project).
    If not installed, falls back to a simple CLI prompt flow.
    """
    try:
        from zephyr_export.api.client import ZephyrClient
        from zephyr_export.services.project_service import ProjectService
        from zephyr_export.services.folder_service import FolderService
    except ImportError:
        return _run_simple_prompts(current_config)

    return _run_tui(current_config)


def _run_tui(current_config: dict) -> dict | None:
    """Full TUI flow using ZephyrFrontend's Textual screens."""
    # Deferred to ZephyrFrontend integration — for now use simple prompts.
    # TODO: Import and launch the Textual app with config pre-population.
    return _run_simple_prompts(current_config)


def _run_simple_prompts(current_config: dict) -> dict | None:
    """Minimal CLI fallback when TUI is not available."""
    print("\n  Zephyr Scale — Plugin Configuration")
    print("  " + "=" * 38 + "\n")

    # Token env var
    token_env = current_config.get("token_env", "ZEPHYR_API_TOKEN")
    token = os.environ.get(token_env, "")
    if token:
        print(f"  API Token ({token_env}): ***set***")
    else:
        print(f"  WARNING: {token_env} is not set in environment.")
        print(f"  Set it before running sync: export {token_env}=...")

    # Project key
    current_project = current_config.get("project_key", "")
    prompt = f"  Project key [{current_project}]: " if current_project else "  Project key: "
    try:
        project_key = input(prompt).strip() or current_project
    except (EOFError, KeyboardInterrupt):
        print("\n  Cancelled.")
        return None

    if not project_key:
        print("  Error: project_key is required.")
        return None

    # Folder (optional)
    current_folder = current_config.get("folder", "")
    prompt = f"  Folder (optional) [{current_folder}]: "
    try:
        folder = input(prompt).strip() or current_folder
    except (EOFError, KeyboardInterrupt):
        print("\n  Cancelled.")
        return None

    # Build updated config
    config = dict(current_config)
    config["project_key"] = project_key
    config["token_env"] = token_env
    if folder:
        config["folder"] = folder
    elif "folder" in config:
        del config["folder"]

    # Set defaults for fields not yet configured
    config.setdefault("classification", "internal")
    config.setdefault("doc_type", "test_case")

    print(f"\n  Configuration:")
    print(f"    project_key: {config['project_key']}")
    print(f"    folder: {config.get('folder', '(all)')}")
    print(f"    token_env: {config['token_env']}")
    print(f"    classification: {config['classification']}")
    print()

    return config


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: configure.py <config_file_path>", file=sys.stderr)
        raise SystemExit(2)

    config_path = Path(sys.argv[1])
    current_config = load_config_file(config_path)

    result = run_interactive(current_config)

    if result is None:
        raise SystemExit(1)

    save_config_file(config_path, result)


if __name__ == "__main__":
    main()
```

**Step 2: Test it manually**

```bash
echo '{}' > /tmp/test-cfg.json
python3 plugins/bundled/zephyr_scale/configure.py /tmp/test-cfg.json
# Enter: CTSLAB for project, leave folder empty
cat /tmp/test-cfg.json
# Should show: {"project_key": "CTSLAB", "token_env": "ZEPHYR_API_TOKEN", ...}
```

**Step 3: Test cancel behavior**

```bash
echo '{}' > /tmp/test-cfg.json
python3 plugins/bundled/zephyr_scale/configure.py /tmp/test-cfg.json
# Press Ctrl+C at the project prompt
echo $?
# Should be: 1
cat /tmp/test-cfg.json
# Should still be: {}
```

**Step 4: Commit**

```bash
git add plugins/bundled/zephyr_scale/configure.py
git commit -m "feat(zephyr): add configure.py with simple CLI prompts"
```

---

### Task 7: End-to-End Integration Test

Test the full flow: `colibri plugins configure <job_id>` with a real manifest.

**Files:**
- No new files — manual testing against existing infrastructure

**Step 1: Ensure a Zephyr job is configured**

Check that `~/.config/colibri/config.yaml` has a Zephyr job. If not, add one:

```yaml
plugins:
  jobs:
    - id: zephyr_test
      manifest: plugins/bundled/zephyr_scale/plugin_manifest.json
      enabled: true
      config: {}
```

**Step 2: Run configure command**

```bash
cargo run -- plugins configure zephyr_test
```

Expected: The simple CLI prompts appear. Enter a project key, optionally a folder.

**Step 3: Verify config.yaml was updated**

```bash
grep -A5 "zephyr_test" ~/.config/colibri/config.yaml
```

Expected: The `config:` block contains the values entered interactively.

**Step 4: Run configure again (reconfigure)**

```bash
cargo run -- plugins configure zephyr_test
```

Expected: Previous values shown as defaults. Can change or keep them.

**Step 5: Test error cases**

```bash
# Unknown job
cargo run -- plugins configure nonexistent
# Expected: Error "Plugin job 'nonexistent' not found"

# Plugin without configure hook (filesystem_documents)
# First ensure there's a filesystem job configured, then:
cargo run -- plugins configure fs_docs
# Expected: Error "does not support interactive configuration"
```

**Step 6: Run full test suite**

Run: `cargo test`
Expected: All tests PASS.

**Step 7: Commit any fixes**

If any fixes were needed during integration testing, commit them.

---

### Task 8: Update Documentation

Update the plugin README and use-cases documentation.

**Files:**
- Modify: `plugins/bundled/zephyr_scale/README.md`

**Step 1: Add configure section to Zephyr README**

Add after the "Config notes" section:

```markdown
## Interactive setup

```bash
colibri plugins configure zephyr_ctslab
```

Walks through project selection and folder scoping. Updates `config.yaml` automatically.
Requires `ZEPHYR_API_TOKEN` to be set in the environment.

To reconfigure later, run the same command again.
```

**Step 2: Commit**

```bash
git add plugins/bundled/zephyr_scale/README.md
git commit -m "docs(zephyr): add configure hook usage to README"
```
