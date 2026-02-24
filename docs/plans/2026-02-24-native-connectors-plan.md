# Native Connectors Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the plugin system with native Rust connectors, removing ~2,500 lines of plugin infrastructure while preserving the canonical store and indexer pipeline.

**Architecture:** A `Connector` trait produces `Vec<DocumentEnvelope>` directly from Rust code. The filesystem connector replaces the Python `filesystem_documents` plugin. All plugin infrastructure (subprocess spawning, JSONL, manifests, bundled plugin extraction) is deleted.

**Tech Stack:** Rust (tokio, serde, sha2, chrono, clap), external tools via `std::process::Command` (docling, pandoc, markitdown)

**Design doc:** `docs/plans/2026-02-24-native-connectors-design.md`

---

### Task 1: Extract DocumentEnvelope into its own module

Move the envelope structs and validation out of `plugin_host.rs` into a standalone `envelope.rs` module. This is the foundation everything else builds on.

**Files:**
- Create: `src/envelope.rs`
- Modify: `src/main.rs`
- Modify: `src/canonical_store.rs` (update import path)
- Modify: `src/metadata_store.rs` (if it imports from plugin_host)

**Step 1: Create `src/envelope.rs`**

Extract these structs and functions from `src/plugin_host.rs`:
- `EnvelopeSource` (line 81)
- `EnvelopeDocument` (line 88)
- `EnvelopeMetadata` (line 98)
- `DocumentEnvelope` (line 107)
- `validate_envelope()` (line 828) — rename to `pub fn validate()`
- `CONTENT_HASH_RE` static (line 19)
- The necessary imports: `chrono::DateTime`, `regex::Regex`, `serde`, `std::sync::OnceLock`
- Import `ColibriError` from `crate::error`

Also add a helper to build envelopes (replaces the Python SDK's `build_envelope`):

```rust
use sha2::{Digest, Sha256};

pub fn content_hash(markdown: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(markdown.as_bytes());
    format!("sha256:{:x}", hasher.finalize())
}
```

**Step 2: Register the module in `src/main.rs`**

Add `mod envelope;` and remove `mod plugin_host;` (don't delete the file yet — that comes in Task 6).

**Step 3: Update imports in `src/canonical_store.rs`**

Change `use crate::plugin_host::DocumentEnvelope;` to `use crate::envelope::DocumentEnvelope;`.

**Step 4: Update any other files importing from `plugin_host`**

Check all files that import `DocumentEnvelope` or envelope structs from `plugin_host` and redirect to `envelope`. This includes `src/cli/plugins.rs` — but don't fix it yet since we'll delete it in Task 6. Just ensure the non-plugin files compile.

**Step 5: Run tests**

```bash
cargo test
```

All existing tests should still pass (plugin_host.rs still exists, envelope.rs duplicates the structs for now).

**Step 6: Commit**

```bash
git add src/envelope.rs src/main.rs src/canonical_store.rs
git commit -m "refactor: extract DocumentEnvelope into standalone envelope module"
```

---

### Task 2: Create the Connector trait and config types

Define the connector abstraction and YAML config structures.

**Files:**
- Create: `src/connectors/mod.rs`
- Modify: `src/config.rs`
- Modify: `src/main.rs`

**Step 1: Write the connector test**

In `src/connectors/mod.rs`, add a test that verifies connector config deserialization:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialize_filesystem_connector_config() {
        let yaml = r#"
type: filesystem
id: books
enabled: true
root_path: /tmp/books
include_extensions:
  - .md
  - .pdf
exclude_globs:
  - "**/.git/**"
mode: incremental
doc_type: book
classification: internal
"#;
        let raw: ConnectorRawConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(raw.connector_type, "filesystem");
        assert_eq!(raw.id, "books");
        assert!(raw.enabled);
    }

    #[test]
    fn deserialize_disabled_connector() {
        let yaml = r#"
type: filesystem
id: test
enabled: false
root_path: /tmp/test
"#;
        let raw: ConnectorRawConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(!raw.enabled);
    }
}
```

**Step 2: Run tests — expect failure**

```bash
cargo test deserialize_filesystem_connector_config
```

Expected: compile error — `ConnectorRawConfig` doesn't exist.

**Step 3: Implement the connector module**

```rust
//! Native connector framework.

pub mod filesystem;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::Value;

use crate::envelope::DocumentEnvelope;
use crate::error::ColibriError;

/// A native content connector.
#[async_trait]
pub trait Connector: Send + Sync {
    /// Unique identifier for this connector instance.
    fn id(&self) -> &str;

    /// Fetch all documents from this source.
    async fn sync(&self) -> Result<Vec<DocumentEnvelope>, ColibriError>;
}

/// Raw YAML config for a connector entry.
#[derive(Debug, Deserialize)]
pub struct ConnectorRawConfig {
    #[serde(rename = "type")]
    pub connector_type: String,
    pub id: String,
    #[serde(default = "default_enabled")]
    pub enabled: bool,
    /// All remaining fields passed to the connector as config.
    #[serde(flatten)]
    pub config: Value,
}

fn default_enabled() -> bool {
    true
}

/// Resolved connector ready to sync.
#[derive(Debug, Clone)]
pub struct ConnectorJob {
    pub id: String,
    pub connector_type: String,
    pub enabled: bool,
    pub config: Value,
}
```

Note: add `async-trait = "0.1"` to `Cargo.toml` dependencies.

**Step 4: Add connectors config to `src/config.rs`**

In `RawConfig`, add alongside the existing `plugins` field:

```rust
#[serde(default)]
connectors: Vec<connectors::ConnectorRawConfig>,
```

In `AppConfig`, add:

```rust
pub connector_jobs: Vec<connectors::ConnectorJob>,
```

In `load_config_inner()`, resolve connectors:

```rust
let connector_jobs: Vec<connectors::ConnectorJob> = raw
    .connectors
    .iter()
    .enumerate()
    .map(|(idx, c)| {
        let id = if c.id.trim().is_empty() {
            format!("connector_{}", idx + 1)
        } else {
            c.id.trim().to_string()
        };
        connectors::ConnectorJob {
            id,
            connector_type: c.connector_type.clone(),
            enabled: c.enabled,
            config: c.config.clone(),
        }
    })
    .collect();
```

Add `connector_jobs` to the `AppConfig` struct initialization.

**Step 5: Register the module**

In `src/main.rs`, add `mod connectors;`.

Create empty `src/connectors/filesystem.rs` with just a comment:

```rust
//! Filesystem document connector.
```

**Step 6: Run tests**

```bash
cargo test
```

**Step 7: Commit**

```bash
git add src/connectors/ src/config.rs src/main.rs Cargo.toml Cargo.lock
git commit -m "feat: add Connector trait and connector config parsing"
```

---

### Task 3: Implement the FilesystemConnector (Markdown-only first)

Start with just Markdown file reading — no external tool conversion yet. This gets the core scanning, hashing, and envelope generation right.

**Files:**
- Modify: `src/connectors/filesystem.rs`
- Modify: `src/connectors/mod.rs`

**Step 1: Write failing tests**

In `src/connectors/filesystem.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn make_test_dir() -> TempDir {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("hello.md"), "# Hello\nWorld").unwrap();
        fs::create_dir(dir.path().join("sub")).unwrap();
        fs::write(dir.path().join("sub/nested.md"), "# Nested").unwrap();
        fs::write(dir.path().join("ignore.txt"), "not markdown").unwrap();
        dir
    }

    #[tokio::test]
    async fn sync_discovers_markdown_files() {
        let dir = make_test_dir();
        let connector = FilesystemConnector {
            id: "test".into(),
            root_path: dir.path().to_path_buf(),
            include_extensions: vec![".md".into()],
            exclude_globs: vec![],
            doc_type: "note".into(),
            classification: "internal".into(),
            plantuml_summaries: false,
        };
        let envelopes = connector.sync().await.unwrap();
        assert_eq!(envelopes.len(), 2);
        // Verify envelope fields
        let titles: Vec<&str> = envelopes.iter().map(|e| e.document.title.as_str()).collect();
        assert!(titles.contains(&"hello"));
        assert!(titles.contains(&"nested"));
    }

    #[tokio::test]
    async fn sync_respects_exclude_globs() {
        let dir = make_test_dir();
        let connector = FilesystemConnector {
            id: "test".into(),
            root_path: dir.path().to_path_buf(),
            include_extensions: vec![".md".into()],
            exclude_globs: vec!["sub/**".into()],
            doc_type: "note".into(),
            classification: "internal".into(),
            plantuml_summaries: false,
        };
        let envelopes = connector.sync().await.unwrap();
        assert_eq!(envelopes.len(), 1);
        assert_eq!(envelopes[0].document.title, "hello");
    }

    #[tokio::test]
    async fn sync_produces_valid_content_hash() {
        let dir = make_test_dir();
        let connector = FilesystemConnector {
            id: "test".into(),
            root_path: dir.path().to_path_buf(),
            include_extensions: vec![".md".into()],
            exclude_globs: vec![],
            doc_type: "note".into(),
            classification: "internal".into(),
            plantuml_summaries: false,
        };
        let envelopes = connector.sync().await.unwrap();
        for env in &envelopes {
            assert!(env.document.content_hash.starts_with("sha256:"));
            assert_eq!(env.document.content_hash.len(), 71); // "sha256:" + 64 hex
        }
    }

    #[tokio::test]
    async fn sync_empty_dir_returns_empty() {
        let dir = TempDir::new().unwrap();
        let connector = FilesystemConnector {
            id: "test".into(),
            root_path: dir.path().to_path_buf(),
            include_extensions: vec![".md".into()],
            exclude_globs: vec![],
            doc_type: "note".into(),
            classification: "internal".into(),
            plantuml_summaries: false,
        };
        let envelopes = connector.sync().await.unwrap();
        assert!(envelopes.is_empty());
    }
}
```

**Step 2: Run tests — expect compile failure**

```bash
cargo test sync_discovers_markdown
```

**Step 3: Implement FilesystemConnector**

```rust
//! Filesystem document connector.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use sha2::{Digest, Sha256};

use super::Connector;
use crate::envelope::{
    content_hash, DocumentEnvelope, EnvelopeDocument, EnvelopeMetadata, EnvelopeSource,
};
use crate::error::ColibriError;

const PLUGIN_ID: &str = "filesystem_documents";

pub struct FilesystemConnector {
    pub id: String,
    pub root_path: PathBuf,
    pub include_extensions: Vec<String>,
    pub exclude_globs: Vec<String>,
    pub doc_type: String,
    pub classification: String,
    pub plantuml_summaries: bool,
}

#[async_trait]
impl Connector for FilesystemConnector {
    fn id(&self) -> &str {
        &self.id
    }

    async fn sync(&self) -> Result<Vec<DocumentEnvelope>, ColibriError> {
        let root = self.root_path.canonicalize().map_err(|e| {
            ColibriError::Config(format!(
                "Cannot resolve root_path {}: {e}",
                self.root_path.display()
            ))
        })?;

        let source_id = stable_source_id(&root);
        let files = discover_files(&root, &self.include_extensions, &self.exclude_globs)?;

        let mut envelopes = Vec::new();
        for file_path in &files {
            let rel = file_path
                .strip_prefix(&root)
                .unwrap_or(file_path)
                .to_string_lossy()
                .to_string();
            let ext = file_path
                .extension()
                .map(|e| format!(".{}", e.to_string_lossy().to_lowercase()))
                .unwrap_or_default();

            let markdown = match ext.as_str() {
                ".md" | ".markdown" => {
                    match std::fs::read_to_string(file_path) {
                        Ok(text) => text,
                        Err(e) => {
                            eprintln!("Failed reading {rel}: {e}");
                            continue;
                        }
                    }
                }
                _ => continue, // Non-markdown handled in Task 4
            };

            let mtime = file_mtime(file_path);
            let title = default_title(file_path);
            let doc_id = format!("{PLUGIN_ID}:{source_id}:{rel}");

            envelopes.push(DocumentEnvelope {
                schema_version: 1,
                source: EnvelopeSource {
                    plugin_id: PLUGIN_ID.into(),
                    connector_instance: root.display().to_string(),
                    external_id: rel,
                    uri: Some(file_path.display().to_string()),
                },
                document: EnvelopeDocument {
                    doc_id,
                    title,
                    content_hash: content_hash(&markdown),
                    markdown,
                    source_updated_at: mtime,
                    deleted: false,
                },
                metadata: EnvelopeMetadata {
                    doc_type: self.doc_type.clone(),
                    classification: self.classification.clone(),
                    tags: None,
                    language: None,
                    acl_tags: None,
                },
            });
        }

        Ok(envelopes)
    }
}

fn stable_source_id(root: &Path) -> String {
    let mut hasher = Sha256::new();
    hasher.update(root.to_string_lossy().as_bytes());
    let digest = hasher.finalize();
    format!("{:x}", digest)[..12].to_string()
}

fn discover_files(
    root: &Path,
    include_extensions: &[String],
    exclude_globs: &[String],
) -> Result<Vec<PathBuf>, ColibriError> {
    let exts: std::collections::HashSet<String> =
        include_extensions.iter().map(|e| e.to_lowercase()).collect();
    let mut files = Vec::new();

    fn walk(
        dir: &Path,
        root: &Path,
        exts: &std::collections::HashSet<String>,
        exclude_globs: &[String],
        files: &mut Vec<PathBuf>,
    ) -> Result<(), ColibriError> {
        let entries = std::fs::read_dir(dir).map_err(|e| {
            ColibriError::Config(format!("Cannot read directory {}: {e}", dir.display()))
        })?;
        for entry in entries {
            let entry = entry.map_err(|e| {
                ColibriError::Config(format!("Error reading entry in {}: {e}", dir.display()))
            })?;
            let path = entry.path();
            let rel = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .to_string_lossy()
                .to_string();

            if should_exclude(&rel, exclude_globs) {
                continue;
            }

            if path.is_dir() {
                walk(&path, root, exts, exclude_globs, files)?;
            } else if path.is_file() {
                let ext = path
                    .extension()
                    .map(|e| format!(".{}", e.to_string_lossy().to_lowercase()))
                    .unwrap_or_default();
                if exts.contains(&ext) {
                    files.push(path);
                }
            }
        }
        Ok(())
    }

    walk(root, root, &exts, exclude_globs, &mut files)?;
    files.sort();
    Ok(files)
}

fn should_exclude(rel_path: &str, exclude_globs: &[String]) -> bool {
    for pattern in exclude_globs {
        if glob_match(pattern, rel_path) {
            return true;
        }
    }
    false
}

/// Simple glob matching supporting `*` and `**`.
fn glob_match(pattern: &str, path: &str) -> bool {
    // Use the glob crate's Pattern for proper matching
    glob::Pattern::new(pattern)
        .map(|p| p.matches(path))
        .unwrap_or(false)
}

fn file_mtime(path: &Path) -> String {
    std::fs::metadata(path)
        .ok()
        .and_then(|m| m.modified().ok())
        .map(|t| {
            let dt: DateTime<Utc> = t.into();
            dt.to_rfc3339()
        })
        .unwrap_or_else(|| Utc::now().to_rfc3339())
}

fn default_title(path: &Path) -> String {
    path.file_stem()
        .map(|s| {
            s.to_string_lossy()
                .replace('_', " ")
                .replace('-', " ")
                .trim()
                .to_string()
        })
        .unwrap_or_else(|| "untitled".into())
}
```

Note: add `glob = "0.3"` to `Cargo.toml` dependencies, and `tempfile = "3"` to `[dev-dependencies]` if not already there.

**Step 4: Run tests**

```bash
cargo test -p colibri -- filesystem
```

All 4 tests should pass.

**Step 5: Commit**

```bash
git add src/connectors/filesystem.rs src/connectors/mod.rs Cargo.toml Cargo.lock
git commit -m "feat: implement FilesystemConnector with Markdown scanning"
```

---

### Task 4: Add document conversion (PDF, DOCX, EPUB, PPTX)

Add external tool conversion for non-Markdown formats.

**Files:**
- Modify: `src/connectors/filesystem.rs`

**Step 1: Write failing tests**

```rust
#[tokio::test]
async fn sync_skips_unsupported_extension_gracefully() {
    let dir = TempDir::new().unwrap();
    fs::write(dir.path().join("data.csv"), "a,b,c").unwrap();
    let connector = FilesystemConnector {
        id: "test".into(),
        root_path: dir.path().to_path_buf(),
        include_extensions: vec![".csv".into()],
        exclude_globs: vec![],
        doc_type: "note".into(),
        classification: "internal".into(),
        plantuml_summaries: false,
    };
    let envelopes = connector.sync().await.unwrap();
    assert!(envelopes.is_empty()); // .csv has no converter
}

#[test]
fn test_convert_command_for_pdf() {
    let cmd = conversion_command(".pdf", Path::new("/tmp/test.pdf"));
    assert!(cmd.is_some());
    let (program, args) = cmd.unwrap();
    assert_eq!(program, "docling");
}

#[test]
fn test_convert_command_for_docx() {
    let cmd = conversion_command(".docx", Path::new("/tmp/test.docx"));
    assert!(cmd.is_some());
    let (program, args) = cmd.unwrap();
    assert_eq!(program, "pandoc");
}

#[test]
fn test_convert_command_for_pptx_prefers_markitdown() {
    let cmd = conversion_command(".pptx", Path::new("/tmp/test.pptx"));
    assert!(cmd.is_some());
    // Primary is markitdown, fallback is pandoc
    let (program, _) = cmd.unwrap();
    assert!(program == "markitdown" || program == "pandoc");
}
```

**Step 2: Run tests — expect failure**

```bash
cargo test test_convert_command
```

**Step 3: Implement conversion**

Add to `src/connectors/filesystem.rs`:

```rust
use std::process::Command;

/// Run an external conversion tool and return the Markdown output.
fn convert_to_markdown(ext: &str, file_path: &Path) -> Result<String, String> {
    match ext {
        ".pdf" => convert_pdf(file_path),
        ".epub" => convert_with_pandoc(file_path, "epub"),
        ".docx" => convert_with_pandoc(file_path, "docx"),
        ".pptx" => convert_pptx(file_path),
        _ => Err(format!("No converter for extension: {ext}")),
    }
}

fn convert_pdf(file_path: &Path) -> Result<String, String> {
    // docling <input> --to md --image-export-mode placeholder --output <tmpdir>
    let tmp = tempfile::tempdir().map_err(|e| format!("tempdir: {e}"))?;
    let status = Command::new("docling")
        .arg(file_path)
        .args(["--to", "md", "--image-export-mode", "placeholder", "--output"])
        .arg(tmp.path())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .status()
        .map_err(|e| format!("Failed to run docling: {e}"))?;

    if !status.success() {
        return Err(format!("docling exited with {status}"));
    }

    let stem = file_path.file_stem().unwrap_or_default().to_string_lossy();
    let out_md = tmp.path().join(format!("{stem}.md"));
    if !out_md.exists() {
        // docling sometimes uses different naming
        let candidates: Vec<_> = std::fs::read_dir(tmp.path())
            .map_err(|e| e.to_string())?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map(|x| x == "md").unwrap_or(false))
            .collect();
        if let Some(entry) = candidates.first() {
            return std::fs::read_to_string(entry.path()).map_err(|e| e.to_string());
        }
        return Err("docling produced no .md output".into());
    }
    std::fs::read_to_string(out_md).map_err(|e| e.to_string())
}

fn convert_with_pandoc(file_path: &Path, from_format: &str) -> Result<String, String> {
    let output = Command::new("pandoc")
        .args(["-f", from_format, "-t", "gfm", "--wrap=none"])
        .arg(file_path)
        .output()
        .map_err(|e| format!("Failed to run pandoc: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("pandoc failed: {stderr}"));
    }
    String::from_utf8(output.stdout).map_err(|e| format!("pandoc output not UTF-8: {e}"))
}

fn convert_pptx(file_path: &Path) -> Result<String, String> {
    // Try markitdown first, fall back to pandoc
    if which_exists("markitdown") {
        let output = Command::new("markitdown")
            .arg(file_path)
            .output()
            .map_err(|e| format!("Failed to run markitdown: {e}"))?;

        if output.status.success() {
            let text = String::from_utf8(output.stdout)
                .map_err(|e| format!("markitdown output not UTF-8: {e}"))?;
            if !text.trim().is_empty() {
                return Ok(text);
            }
        }
    }
    // Fallback to pandoc
    convert_with_pandoc(file_path, "pptx")
}

fn which_exists(tool: &str) -> bool {
    Command::new("which")
        .arg(tool)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}
```

Update the `sync()` method's match arm to handle non-markdown extensions:

```rust
".md" | ".markdown" => { /* existing code */ }
".pdf" | ".epub" | ".docx" | ".pptx" => {
    match convert_to_markdown(&ext, file_path) {
        Ok(text) => text,
        Err(e) => {
            eprintln!("Failed converting {rel}: {e}");
            continue;
        }
    }
}
_ => continue,
```

**Step 4: Run tests**

```bash
cargo test -p colibri -- filesystem
```

**Step 5: Commit**

```bash
git add src/connectors/filesystem.rs
git commit -m "feat: add PDF/DOCX/EPUB/PPTX conversion to filesystem connector"
```

---

### Task 5: Add PlantUML enrichment

Port the PlantUML summary enrichment from `plugin.py`.

**Files:**
- Modify: `src/connectors/filesystem.rs`

**Step 1: Write failing tests**

```rust
#[test]
fn test_plantuml_enrichment_adds_summary() {
    let input = r#"# Doc

```plantuml
actor User
User -> Server: request
Server -> DB: query
```

More text.
"#;
    let enriched = enrich_plantuml_blocks(input);
    assert!(enriched.contains("colibri:plantuml-summary:start"));
    assert!(enriched.contains("User"));
    assert!(enriched.contains("Server"));
    assert!(enriched.contains("DB"));
}

#[test]
fn test_plantuml_enrichment_no_plantuml() {
    let input = "# Just markdown\n\nNo diagrams here.\n";
    let enriched = enrich_plantuml_blocks(input);
    assert!(!enriched.contains("plantuml-summary"));
    assert_eq!(enriched.trim(), input.trim());
}

#[test]
fn test_plantuml_strips_existing_summaries() {
    let input = r#"# Doc

```plantuml
A -> B
```

<!-- colibri:plantuml-summary:start -->
old summary
<!-- colibri:plantuml-summary:end -->

More text.
"#;
    let enriched = enrich_plantuml_blocks(input);
    // Should not contain "old summary" but should have new summary
    assert!(!enriched.contains("old summary"));
    assert!(enriched.contains("colibri:plantuml-summary:start"));
}
```

**Step 2: Run tests — expect failure**

```bash
cargo test test_plantuml_enrichment
```

**Step 3: Implement PlantUML enrichment**

Port the Python logic from `plugin.py` lines 248-372. This is regex-based parsing of PlantUML blocks to extract entity names and relations, inserted as HTML comments after each block.

```rust
fn enrich_plantuml_blocks(md: &str) -> String {
    let stripped = strip_existing_plantuml_summaries(md);
    let lines: Vec<&str> = stripped.lines().collect();
    let mut out: Vec<String> = Vec::new();
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i];
        out.push(line.to_string());

        if line.trim() == "```plantuml" || line.trim() == "```puml" {
            i += 1;
            let mut block = Vec::new();
            while i < lines.len() && lines[i].trim() != "```" {
                block.push(lines[i].to_string());
                out.push(lines[i].to_string());
                i += 1;
            }
            if i < lines.len() {
                out.push(lines[i].to_string()); // closing fence
            }
            let summary = parse_plantuml(&block.join("\n"));
            if !summary.entities.is_empty() || !summary.relations.is_empty() {
                out.push(String::new());
                out.push("<!-- colibri:plantuml-summary:start -->".into());
                out.push("[CoLibri PlantUML summary]".into());
                if !summary.entities.is_empty() {
                    out.push(format!("Entities: {}", summary.entities.join(", ")));
                }
                if !summary.relations.is_empty() {
                    out.push("Relations:".into());
                    for r in summary.relations.iter().take(50) {
                        out.push(format!("- {r}"));
                    }
                }
                out.push("<!-- colibri:plantuml-summary:end -->".into());
                out.push(String::new());
            }
        }
        i += 1;
    }

    let result = out.join("\n");
    if result.ends_with('\n') {
        result
    } else {
        result + "\n"
    }
}

fn strip_existing_plantuml_summaries(md: &str) -> String {
    let start = "<!-- colibri:plantuml-summary:start -->";
    let end = "<!-- colibri:plantuml-summary:end -->";
    let mut out = Vec::new();
    let mut in_block = false;
    for line in md.lines() {
        if line.trim() == start {
            in_block = true;
            continue;
        }
        if line.trim() == end {
            in_block = false;
            continue;
        }
        if !in_block {
            out.push(line);
        }
    }
    out.join("\n")
}

struct PlantUmlSummary {
    entities: Vec<String>,
    relations: Vec<String>,
}

fn parse_plantuml(text: &str) -> PlantUmlSummary {
    // Port of plugin.py parse_plantuml():
    // - Track aliases ("X" as Y)
    // - Detect participant/actor/component declarations
    // - Detect arrow relations (A -> B : label)
    let mut entities = std::collections::BTreeSet::new();
    let mut relations = Vec::new();
    let mut alias_map: std::collections::HashMap<String, String> = std::collections::HashMap::new();

    let arrows = ["<->", "<-->", "-->", "->", "<-", "<--", "..>", ".>", "--|>", "-|>"];
    let decl_keywords = [
        "participant", "actor", "boundary", "control", "entity",
        "database", "component", "interface", "class", "object", "usecase",
    ];

    for raw_line in text.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('\'') {
            continue;
        }

        // Alias: "User" as U
        if line.contains(" as ") {
            let parts: Vec<&str> = line.splitn(2, " as ").collect();
            if parts.len() == 2 {
                let left = norm(parts[0].split_whitespace().last().unwrap_or(""));
                let right = norm(parts[1].split_whitespace().next().unwrap_or(""));
                if !left.is_empty() && !right.is_empty() {
                    alias_map.insert(right.clone(), left.clone());
                    entities.insert(left);
                }
            }
        }

        // Declarations
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if !tokens.is_empty() && decl_keywords.contains(&tokens[0]) && tokens.len() >= 2 {
            let name = norm(tokens[1]);
            if !name.is_empty() {
                entities.insert(name);
            }
        }

        // Arrow relations
        for arrow in &arrows {
            if line.contains(arrow) {
                let parts: Vec<&str> = line.splitn(2, arrow).collect();
                if parts.len() == 2 {
                    let left = norm(parts[0].split_whitespace().last().unwrap_or(""));
                    let right_part = parts[1].trim();
                    let right = norm(right_part.split_whitespace().next().unwrap_or(""));
                    if left.is_empty() || right.is_empty() {
                        continue;
                    }
                    let left = alias_map.get(&left).cloned().unwrap_or(left);
                    let right = alias_map.get(&right).cloned().unwrap_or(right);
                    entities.insert(left.clone());
                    entities.insert(right.clone());

                    let label = if right_part.contains(':') {
                        right_part.splitn(2, ':').nth(1).unwrap_or("").trim().to_string()
                    } else {
                        String::new()
                    };
                    let rel = if label.is_empty() {
                        format!("{left} {arrow} {right}")
                    } else {
                        format!("{left} {arrow} {right}: {label}")
                    };
                    relations.push(rel);
                }
                break;
            }
        }
    }

    PlantUmlSummary {
        entities: entities.into_iter().collect(),
        relations,
    }
}

fn norm(s: &str) -> String {
    s.trim().trim_matches('"').trim_matches('\'').to_string()
}
```

In the `sync()` method, after reading markdown, apply enrichment when `self.plantuml_summaries` is true:

```rust
let markdown = if self.plantuml_summaries {
    enrich_plantuml_blocks(&markdown)
} else {
    markdown
};
```

**Step 4: Run tests**

```bash
cargo test -p colibri -- plantuml
```

**Step 5: Commit**

```bash
git add src/connectors/filesystem.rs
git commit -m "feat: add PlantUML enrichment to filesystem connector"
```

---

### Task 6: Wire up CLI and `colibri sync`

Create the new CLI handler for connectors and rewire `colibri sync` to use connectors instead of plugins.

**Files:**
- Create: `src/cli/connectors.rs`
- Modify: `src/cli/mod.rs`
- Modify: `src/cli/sync.rs`
- Modify: `src/main.rs`

**Step 1: Implement `src/cli/connectors.rs`**

```rust
//! CLI handlers for native connectors.

use serde::Serialize;

use crate::canonical_store::{ingest_envelopes, CanonicalIngestReport};
use crate::config::{load_config, AppConfig};
use crate::connectors::{ConnectorJob, filesystem::FilesystemConnector, Connector};
use crate::error::ColibriError;
use crate::indexer::index_library;

#[derive(Debug, Serialize)]
struct ConnectorListEntry {
    id: String,
    connector_type: String,
    enabled: bool,
}

pub async fn list(json: bool) -> anyhow::Result<()> {
    let config = load_config().map_err(|e| anyhow::anyhow!(e.to_string()))?;

    let entries: Vec<ConnectorListEntry> = config
        .connector_jobs
        .iter()
        .map(|j| ConnectorListEntry {
            id: j.id.clone(),
            connector_type: j.connector_type.clone(),
            enabled: j.enabled,
        })
        .collect();

    if json {
        println!("{}", serde_json::to_string_pretty(&entries)?);
        return Ok(());
    }

    if entries.is_empty() {
        eprintln!("No connectors configured.");
        return Ok(());
    }

    for entry in &entries {
        let label = if entry.enabled { "enabled" } else { "disabled" };
        eprintln!("  {} [{}] ({})", entry.id, entry.connector_type, label);
    }
    Ok(())
}

/// Build a Connector from a ConnectorJob.
fn build_connector(job: &ConnectorJob) -> Result<Box<dyn Connector>, ColibriError> {
    match job.connector_type.as_str() {
        "filesystem" => {
            let root_path = job.config.get("root_path")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ColibriError::Config(
                    format!("Connector '{}': missing root_path", job.id)
                ))?;
            let include_extensions: Vec<String> = job.config.get("include_extensions")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                .unwrap_or_else(|| vec![".md".into(), ".markdown".into()]);
            let exclude_globs: Vec<String> = job.config.get("exclude_globs")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                .unwrap_or_default();
            let doc_type = job.config.get("doc_type")
                .and_then(|v| v.as_str())
                .unwrap_or("note")
                .to_string();
            let classification = job.config.get("classification")
                .and_then(|v| v.as_str())
                .unwrap_or("internal")
                .to_string();
            let plantuml_summaries = job.config.get("plantuml_summaries")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);

            // Expand ~ in root_path
            let expanded = if root_path.starts_with("~/") {
                dirs::home_dir()
                    .unwrap_or_default()
                    .join(&root_path[2..])
            } else {
                std::path::PathBuf::from(root_path)
            };

            Ok(Box::new(FilesystemConnector {
                id: job.id.clone(),
                root_path: expanded,
                include_extensions,
                exclude_globs,
                doc_type,
                classification,
                plantuml_summaries,
            }))
        }
        other => Err(ColibriError::Config(
            format!("Unknown connector type: '{other}'. Available types: filesystem")
        )),
    }
}

#[derive(Debug, Serialize)]
pub struct SyncReport {
    pub connector_id: String,
    pub connector_type: String,
    pub status: String,
    pub envelope_count: usize,
    pub ingest: Option<CanonicalIngestReport>,
    pub error: Option<String>,
}

pub async fn sync_all(
    config: &AppConfig,
    requested_connectors: &[String],
    include_disabled: bool,
    fail_fast: bool,
    dry_run: bool,
    json: bool,
) -> anyhow::Result<Vec<SyncReport>> {
    let jobs: Vec<&ConnectorJob> = config
        .connector_jobs
        .iter()
        .filter(|j| {
            if !requested_connectors.is_empty() {
                return requested_connectors.contains(&j.id);
            }
            include_disabled || j.enabled
        })
        .collect();

    let mut reports = Vec::new();

    for job in &jobs {
        if !json {
            eprintln!("Syncing connector: {} [{}]", job.id, job.connector_type);
        }

        let connector = match build_connector(job) {
            Ok(c) => c,
            Err(e) => {
                let report = SyncReport {
                    connector_id: job.id.clone(),
                    connector_type: job.connector_type.clone(),
                    status: "error".into(),
                    envelope_count: 0,
                    ingest: None,
                    error: Some(e.to_string()),
                };
                if !json {
                    eprintln!("  ERROR: {e}");
                }
                reports.push(report);
                if fail_fast {
                    break;
                }
                continue;
            }
        };

        match connector.sync().await {
            Ok(envelopes) => {
                let count = envelopes.len();
                let ingest_report = ingest_envelopes(config, &envelopes, dry_run)?;
                if !json {
                    eprintln!(
                        "  OK: {} envelopes, {} written, {} unchanged",
                        count, ingest_report.written, ingest_report.unchanged
                    );
                }
                reports.push(SyncReport {
                    connector_id: job.id.clone(),
                    connector_type: job.connector_type.clone(),
                    status: "ok".into(),
                    envelope_count: count,
                    ingest: Some(ingest_report),
                    error: None,
                });
            }
            Err(e) => {
                if !json {
                    eprintln!("  ERROR: {e}");
                }
                reports.push(SyncReport {
                    connector_id: job.id.clone(),
                    connector_type: job.connector_type.clone(),
                    status: "error".into(),
                    envelope_count: 0,
                    ingest: None,
                    error: Some(e.to_string()),
                });
                if fail_fast {
                    break;
                }
            }
        }
    }

    Ok(reports)
}
```

**Step 2: Create `Connectors` subcommand in `src/cli/mod.rs`**

Add to the `Commands` enum:

```rust
/// Manage content connectors
Connectors {
    #[command(subcommand)]
    command: ConnectorCommands,
},
```

Add a new enum:

```rust
#[derive(Subcommand)]
pub enum ConnectorCommands {
    /// List configured connectors
    List {
        #[arg(long)]
        json: bool,
    },
}
```

Add `pub mod connectors;` to the module declarations at the top.

**Step 3: Rewire `src/cli/sync.rs`**

Replace the current implementation that delegates to `plugins::sync_all`:

```rust
//! `colibri sync` — ingest configured sources into the canonical store.

use crate::cli::connectors;
use crate::config::load_config;
use crate::indexer::index_library;

pub async fn run(
    requested: Vec<String>,
    include_disabled: bool,
    fail_fast: bool,
    no_index: bool,
    force: bool,
    dry_run: bool,
    json: bool,
) -> anyhow::Result<()> {
    let config = load_config().map_err(|e| anyhow::anyhow!(e.to_string()))?;

    let reports = connectors::sync_all(
        &config,
        &requested,
        include_disabled,
        fail_fast,
        dry_run,
        json,
    )
    .await?;

    if json {
        println!("{}", serde_json::to_string_pretty(&reports)?);
    }

    let any_failed = reports.iter().any(|r| r.status == "error");

    if !no_index && !dry_run && !any_failed {
        if !json {
            eprintln!("\nIndexing...");
        }
        index_library(&config, force).await?;
    }

    if any_failed {
        anyhow::bail!("One or more connectors failed");
    }

    Ok(())
}
```

**Step 4: Update `src/main.rs` command routing**

Add the `Connectors` command routing alongside the existing ones:

```rust
cli::Commands::Connectors { command } => match command {
    cli::ConnectorCommands::List { json } => cli::connectors::list(json).await,
},
```

Update the `Sync` command args to rename `jobs` to `connector`:

In `src/cli/mod.rs`, change:
```rust
Sync {
    /// Restrict to specific connector id(s); may be repeated
    #[arg(long = "connector")]
    connectors: Vec<String>,
    // ... rest stays the same but remove include_disabled
```

**Step 5: Run tests**

```bash
cargo test
```

**Step 6: Commit**

```bash
git add src/cli/connectors.rs src/cli/mod.rs src/cli/sync.rs src/main.rs
git commit -m "feat: wire up connector CLI and rewrite colibri sync"
```

---

### Task 7: Delete plugin infrastructure

Remove all plugin-related code and files.

**Files:**
- Delete: `src/plugin_host.rs`
- Delete: `src/cli/plugins.rs`
- Delete: `src/bundled_plugins.rs`
- Delete: `src/plugin_requirements.rs`
- Delete: `plugins/` directory (entire)
- Modify: `src/main.rs` (remove plugin module declarations and routing)
- Modify: `src/cli/mod.rs` (remove PluginCommands, plugin module)
- Modify: `src/cli/bootstrap.rs` (remove plugin references)
- Modify: `src/cli/doctor.rs` (remove plugin health checks)
- Modify: `src/config.rs` (remove PluginsConfig, PluginJob, resolve_plugin_jobs, update_plugin_job_config)
- Modify: `src/metadata_store.rs` (remove sync_state table and methods)

**Step 1: Delete plugin files**

```bash
rm src/plugin_host.rs src/cli/plugins.rs src/bundled_plugins.rs src/plugin_requirements.rs
rm -rf plugins/
```

**Step 2: Clean up `src/main.rs`**

Remove these lines:
- `mod bundled_plugins;`
- `mod plugin_host;`
- `mod plugin_requirements;`
- The entire `cli::Commands::Plugins { command } => match command { ... }` arm

**Step 3: Clean up `src/cli/mod.rs`**

Remove:
- `pub mod plugins;`
- The `Plugins` variant from `Commands`
- The `PluginCommands` enum
- The `PluginStateCommands` enum

**Step 4: Clean up `src/config.rs`**

Remove:
- `PluginsConfig` struct
- `PluginJobRawConfig` struct and its `Default` impl
- `PluginJob` struct
- `resolve_plugin_jobs()` function
- `update_plugin_job_config()` function
- The `plugins` field from `RawConfig`
- The `plugin_jobs` field from `AppConfig`
- References to `plugin_jobs` in `load_config_inner()`
- The `config_update_tests` module

**Step 5: Clean up `src/cli/bootstrap.rs`**

This is a significant rewrite. Remove all plugin-related logic:
- Remove `use crate::bundled_plugins;`
- Remove `use crate::plugin_host::{load_plugin_manifest, RequiredEnvVar, RequiredTool};`
- Remove `YamlPluginJob`, `YamlPluginsConfig`
- Remove `gather_plugin_requirements()` function
- Remove the `plugins:` section from `YamlConfig`
- Remove `bundled_plugins_ok`, `bundled_plugins_error`, `plugin_tools_missing`, `plugin_env_missing` from `BootstrapReport`
- Instead, bootstrap should write a config with `connectors:` section if `--init-path` is given
- The connector dependency check (pandoc, docling) can be handled by checking tool availability based on the configured extensions

**Step 6: Clean up `src/cli/doctor.rs`**

Remove plugin health check section:
- Remove `use crate::plugin_host::load_plugin_manifest;`
- Remove `use crate::plugin_requirements::tool_checks_relevant_for_job;`
- Remove `DoctorPluginJobStatus` struct
- Remove `plugin_jobs` from `DoctorReport`
- Remove the entire "1b. Plugins" section (lines 148-351)
- Add a "Connectors" health check section that validates configured connectors exist and have required tools

**Step 7: Clean up `src/metadata_store.rs`**

Remove sync_state related code:
- Remove `SyncStateEntry` struct
- Remove `get_sync_cursor()`, `upsert_sync_success()`, `upsert_sync_error()`, `list_sync_entries()`, `get_sync_entry()`, `delete_sync_entry()` methods
- Remove the `sync_state` CREATE TABLE from `bootstrap()`

**Step 8: Update `src/canonical_store.rs`**

Change import: `use crate::plugin_host::DocumentEnvelope;` → `use crate::envelope::DocumentEnvelope;`

Update doc comment on `ingest_envelopes()`: "Persist validated plugin envelopes" → "Persist validated connector envelopes"

**Step 9: Run tests**

```bash
cargo test
```

Fix any remaining compilation errors from removed types.

**Step 10: Run clippy and format**

```bash
cargo clippy -- -D warnings
cargo fmt
```

**Step 11: Commit**

```bash
git add -A
git commit -m "refactor: remove plugin infrastructure, complete connector migration

Remove plugin_host.rs, cli/plugins.rs, bundled_plugins.rs,
plugin_requirements.rs, and plugins/ directory. Update bootstrap,
doctor, config, and metadata_store to work without plugins.
All content ingestion now flows through native Rust connectors."
```

---

### Task 8: Update bootstrap and doctor for connectors

Rewrite bootstrap to generate connector config and doctor to validate connectors.

**Files:**
- Modify: `src/cli/bootstrap.rs`
- Modify: `src/cli/doctor.rs`

**Step 1: Rewrite bootstrap config generation**

When `--init-path` is provided, generate a connector entry instead of a plugin job:

```yaml
data:
  directory: ~/.local/share/colibri
ollama:
  base_url: http://localhost:11434
  embedding_model: bge-m3
connectors:
  - type: filesystem
    id: docs
    enabled: true
    root_path: ~/Documents
    include_extensions:
      - .md
      - .markdown
    classification: internal
```

Replace `YamlPluginJob` / `YamlPluginsConfig` with:

```rust
#[derive(Debug, Serialize)]
struct YamlConnectorEntry {
    #[serde(rename = "type")]
    connector_type: String,
    id: String,
    enabled: bool,
    root_path: String,
    include_extensions: Vec<String>,
    classification: String,
}

#[derive(Debug, Serialize)]
struct YamlConfig {
    data: YamlDataConfig,
    ollama: YamlOllamaConfig,
    connectors: Vec<YamlConnectorEntry>,
}
```

For dependency checking, replace `gather_plugin_requirements()` with a simpler function that checks what external tools the configured connectors need based on their `include_extensions`:

```rust
fn check_connector_tools(connectors: &[connectors::ConnectorJob]) -> Vec<String> {
    let mut missing = Vec::new();
    for job in connectors {
        if job.connector_type != "filesystem" { continue; }
        let exts: Vec<String> = job.config.get("include_extensions")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(|s| s.to_lowercase())).collect())
            .unwrap_or_default();

        if exts.iter().any(|e| e == ".pdf") && !tool_on_path("docling") {
            missing.push("docling (pipx install docling)".into());
        }
        if exts.iter().any(|e| e == ".docx" || e == ".epub") && !tool_on_path("pandoc") {
            missing.push("pandoc (brew install pandoc)".into());
        }
        if exts.iter().any(|e| e == ".pptx") {
            if !tool_on_path("markitdown") && !tool_on_path("pandoc") {
                missing.push("markitdown (pipx install markitdown) or pandoc (brew install pandoc)".into());
            }
        }
    }
    missing
}
```

**Step 2: Rewrite doctor connector section**

Replace the plugin health check with a connector health check:

```rust
#[derive(Debug, Serialize)]
struct DoctorConnectorStatus {
    id: String,
    connector_type: String,
    enabled: bool,
    status: String,
    issues: Vec<String>,
}
```

For filesystem connectors, check:
- root_path exists
- Required tools available for configured extensions
- Environment variables set (if any)

**Step 3: Run tests**

```bash
cargo test
cargo clippy -- -D warnings
```

**Step 4: Commit**

```bash
git add src/cli/bootstrap.rs src/cli/doctor.rs
git commit -m "feat: update bootstrap and doctor for native connectors"
```

---

### Task 9: End-to-end testing

Verify the complete flow works with a real filesystem connector.

**Files:**
- No new files

**Step 1: Build**

```bash
make build
```

**Step 2: Test `colibri connectors list`**

With the user's existing config updated to use connectors:

```bash
cargo run -- connectors list
```

Expected: lists configured connectors.

**Step 3: Test `colibri sync` with a Markdown-only connector**

```bash
cargo run -- sync --connector books --dry-run
```

Expected: scans the configured directory, reports envelopes found, no writes (dry run).

**Step 4: Test `colibri sync` for real**

```bash
cargo run -- sync --connector books
```

Expected: envelopes written to canonical store, index updated.

**Step 5: Test `colibri doctor`**

```bash
cargo run -- doctor
```

Expected: shows connector health, no plugin references.

**Step 6: Test `colibri bootstrap`**

```bash
cargo run -- bootstrap --non-interactive --init-path /tmp/test-docs --json
```

Expected: generates config.yaml with `connectors:` section.

**Step 7: Run full test suite**

```bash
cargo test
cargo clippy -- -D warnings
cargo fmt --check
```

**Step 8: Commit any fixes**

```bash
git add -A
git commit -m "test: verify end-to-end connector pipeline"
```

---

### Task 10: Update CLAUDE.md and config documentation

Update project documentation to reflect the architecture change.

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update CLAUDE.md**

Key changes:
- Replace all "plugin" references with "connector" in architecture docs
- Update config structure example to show `connectors:` instead of `plugins.jobs:`
- Update data flow diagram
- Remove references to Python SDK, JSONL protocol, plugin manifests
- Update CLI command reference (remove `colibri plugins *`, add `colibri connectors list`)
- Update the "Key Types" section to include `Connector` trait

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for native connector architecture"
```
