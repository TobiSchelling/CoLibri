//! Filesystem document connector.
//!
//! Walks a directory tree, filters by extension and exclude globs,
//! reads Markdown files, and produces `DocumentEnvelope`s.
//! Non-markdown conversion (PDF, DOCX, etc.) is deferred to Task 4.

// Wired into production code paths in Task 6 (CLI sync).
#![allow(dead_code)]

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use sha2::{Digest, Sha256};

use super::Connector;
use crate::envelope::{
    content_hash, DocumentEnvelope, EnvelopeDocument, EnvelopeMetadata, EnvelopeSource,
};
use crate::error::ColibriError;

// Matches the Python plugin_id for data compatibility with the canonical store.
const PLUGIN_ID: &str = "filesystem_documents";

/// A connector that reads documents from the local filesystem.
pub struct FilesystemConnector {
    pub id: String,
    pub root_path: PathBuf,
    pub include_extensions: Vec<String>,
    pub exclude_globs: Vec<String>,
    pub doc_type: String,
    pub classification: String,
    /// Whether to enrich PlantUML code blocks with text summaries (Task 5).
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
                ".md" | ".markdown" => match std::fs::read_to_string(file_path) {
                    Ok(text) => text,
                    Err(e) => {
                        eprintln!("Failed reading {rel}: {e}");
                        continue;
                    }
                },
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

/// Compute a stable 12-hex-char source identifier from the canonicalized root path.
///
/// Matches the Python plugin's `stable_source_id()` for data compatibility.
fn stable_source_id(root: &Path) -> String {
    let mut hasher = Sha256::new();
    hasher.update(root.to_string_lossy().as_bytes());
    let digest = hasher.finalize();
    format!("{:x}", digest)[..12].to_string()
}

/// Recursively discover files under `root` matching `include_extensions`,
/// skipping paths that match any `exclude_globs` pattern.
fn discover_files(
    root: &Path,
    include_extensions: &[String],
    exclude_globs: &[String],
) -> Result<Vec<PathBuf>, ColibriError> {
    let exts: HashSet<String> = include_extensions
        .iter()
        .map(|e| e.to_lowercase())
        .collect();
    let mut files = Vec::new();

    walk_dir(root, root, &exts, exclude_globs, &mut files)?;
    files.sort();
    Ok(files)
}

/// Recursive directory walker.
fn walk_dir(
    dir: &Path,
    root: &Path,
    exts: &HashSet<String>,
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
            walk_dir(&path, root, exts, exclude_globs, files)?;
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

/// Check whether a relative path should be excluded by any glob pattern.
fn should_exclude(rel_path: &str, exclude_globs: &[String]) -> bool {
    for pattern in exclude_globs {
        if let Ok(pat) = glob::Pattern::new(pattern) {
            if pat.matches(rel_path) {
                return true;
            }
        }
    }
    false
}

/// Extract the file modification time as an RFC 3339 string, falling back to now.
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

/// Derive a human-friendly title from the file stem.
///
/// Replaces underscores and hyphens with spaces, matching the Python plugin's behavior.
fn default_title(path: &Path) -> String {
    path.file_stem()
        .map(|s| {
            s.to_string_lossy()
                .replace(['_', '-'], " ")
                .trim()
                .to_string()
        })
        .unwrap_or_else(|| "untitled".into())
}

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
        let titles: Vec<&str> = envelopes
            .iter()
            .map(|e| e.document.title.as_str())
            .collect();
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

    #[test]
    fn stable_source_id_is_deterministic() {
        let id1 = stable_source_id(Path::new("/tmp/test"));
        let id2 = stable_source_id(Path::new("/tmp/test"));
        assert_eq!(id1, id2);
        assert_eq!(id1.len(), 12);
    }

    #[test]
    fn stable_source_id_differs_for_different_paths() {
        let id1 = stable_source_id(Path::new("/tmp/a"));
        let id2 = stable_source_id(Path::new("/tmp/b"));
        assert_ne!(id1, id2);
    }

    #[test]
    fn default_title_replaces_separators() {
        assert_eq!(default_title(Path::new("my_file-name.md")), "my file name");
    }

    #[test]
    fn should_exclude_matches_glob() {
        assert!(should_exclude("sub/nested.md", &["sub/**".into()]));
        assert!(!should_exclude("hello.md", &["sub/**".into()]));
    }

    #[tokio::test]
    async fn sync_sets_correct_doc_id_format() {
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
        let doc_id = &envelopes[0].document.doc_id;
        assert!(doc_id.starts_with("filesystem_documents:"));
        // Format: "filesystem_documents:{12-hex}:{rel_path}"
        let parts: Vec<&str> = doc_id.splitn(3, ':').collect();
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0], "filesystem_documents");
        assert_eq!(parts[1].len(), 12);
        assert_eq!(parts[2], "hello.md");
    }

    #[tokio::test]
    async fn sync_envelope_passes_validation() {
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
            crate::envelope::validate(env, PLUGIN_ID).unwrap();
        }
    }
}
