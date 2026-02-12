//! Plain markdown folder content source.
//!
//! Handles simple markdown folders with optional YAML frontmatter.
//! Mirrors the Python `MarkdownFolderSource`.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use regex::Regex;

use crate::error::ColibriError;
use crate::sources::{ContentSource, SourceDocument};

/// Content source for plain markdown folders.
pub struct MarkdownFolderSource {
    folder_path: PathBuf,
    recursive: bool,
    display_name: String,
    extensions: Vec<String>,
    exclude_paths: Vec<String>,
}

impl MarkdownFolderSource {
    pub fn new(
        folder_path: impl Into<PathBuf>,
        recursive: bool,
        name: Option<String>,
        extensions: Vec<String>,
        exclude_paths: Vec<String>,
    ) -> Self {
        let folder_path = folder_path.into();
        let folder_path = if folder_path.starts_with("~") {
            if let Some(home) = dirs::home_dir() {
                home.join(folder_path.strip_prefix("~").unwrap())
            } else {
                folder_path
            }
        } else {
            folder_path
        };
        let folder_path = folder_path
            .canonicalize()
            .unwrap_or_else(|_| folder_path.clone());

        let display_name = name.unwrap_or_else(|| {
            folder_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown")
                .to_string()
        });

        // Resolve exclude paths to absolute
        let exclude_paths = exclude_paths
            .into_iter()
            .map(|p| {
                let pb = PathBuf::from(&p);
                pb.canonicalize()
                    .unwrap_or(pb)
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();

        Self {
            folder_path,
            recursive,
            display_name,
            extensions,
            exclude_paths,
        }
    }
}

impl ContentSource for MarkdownFolderSource {
    fn name(&self) -> &str {
        &self.display_name
    }

    fn root_path(&self) -> &Path {
        &self.folder_path
    }

    fn source_type(&self) -> &str {
        "markdown"
    }

    fn list_documents(&self) -> Vec<PathBuf> {
        let mut seen = HashSet::new();
        let mut results = Vec::new();

        if !self.folder_path.exists() {
            return results;
        }

        for ext in &self.extensions {
            let pattern = if self.recursive {
                format!("{}/**/*{ext}", self.folder_path.display())
            } else {
                format!("{}/*{ext}", self.folder_path.display())
            };

            let entries = match glob::glob(&pattern) {
                Ok(entries) => entries,
                Err(_) => continue,
            };

            for entry in entries.flatten() {
                let rel = match entry.strip_prefix(&self.folder_path) {
                    Ok(r) => r.to_path_buf(),
                    Err(_) => continue,
                };

                // Skip duplicates
                if seen.contains(&rel) {
                    continue;
                }

                // Skip hidden files/directories
                if rel
                    .components()
                    .any(|c| c.as_os_str().to_str().is_some_and(|s| s.starts_with('.')))
                {
                    continue;
                }

                // Skip files under excluded paths
                if !self.exclude_paths.is_empty() {
                    let abs_str = entry.to_string_lossy();
                    if self.exclude_paths.iter().any(|ex| {
                        abs_str.starts_with(&format!("{ex}/")) || abs_str.as_ref() == ex.as_str()
                    }) {
                        continue;
                    }
                }

                seen.insert(rel.clone());
                results.push(rel);
            }
        }

        results.sort();
        results
    }

    fn read_document(&self, path: &Path) -> Result<SourceDocument, ColibriError> {
        let full_path = self.folder_path.join(path);

        if !full_path.exists() {
            return Err(ColibriError::Source(format!(
                "Document not found: {}",
                path.display()
            )));
        }

        let suffix = full_path.extension().and_then(|e| e.to_str()).unwrap_or("");

        match suffix {
            "yaml" | "yml" => self.read_yaml(path, &full_path),
            _ => self.read_markdown(path, &full_path),
        }
    }
}

impl MarkdownFolderSource {
    /// Read a markdown file with optional YAML frontmatter.
    fn read_markdown(&self, path: &Path, full_path: &Path) -> Result<SourceDocument, ColibriError> {
        let content = read_lossy(full_path)?;

        let (metadata, doc_content) = parse_frontmatter(&content);

        // Extract title: frontmatter > first H1 > filename
        let title = metadata
            .get("title")
            .and_then(|v| v.as_str())
            .map(String::from)
            .or_else(|| extract_first_heading(&doc_content))
            .unwrap_or_else(|| title_from_filename(path));

        let folder = path
            .components()
            .next()
            .and_then(|c| c.as_os_str().to_str())
            .filter(|_| path.components().count() > 1)
            .unwrap_or("")
            .to_string();

        // Extract tags
        let tags = extract_tags(&metadata);

        // Extract doc_type from frontmatter "type" field
        let doc_type = metadata
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("note")
            .to_string();

        Ok(SourceDocument {
            path: path.to_path_buf(),
            content: doc_content,
            title,
            doc_type,
            source_name: self.display_name.clone(),
            source_type: "markdown".into(),
            folder,
            metadata,
            tags,
        })
    }

    /// Read a YAML file (e.g., OpenAPI spec) as a document.
    fn read_yaml(&self, path: &Path, full_path: &Path) -> Result<SourceDocument, ColibriError> {
        let content = read_lossy(full_path)?;

        let title = extract_yaml_title(&content).unwrap_or_else(|| title_from_filename(path));

        let folder = path
            .components()
            .next()
            .and_then(|c| c.as_os_str().to_str())
            .filter(|_| path.components().count() > 1)
            .unwrap_or("")
            .to_string();

        Ok(SourceDocument {
            path: path.to_path_buf(),
            content,
            title,
            doc_type: "note".into(),
            source_name: self.display_name.clone(),
            source_type: "markdown".into(),
            folder,
            metadata: serde_json::Map::new(),
            tags: vec![],
        })
    }
}

/// Parse YAML frontmatter delimited by `---`.
///
/// Returns `(metadata, content_without_frontmatter)`.
fn parse_frontmatter(text: &str) -> (serde_json::Map<String, serde_json::Value>, String) {
    let empty = (serde_json::Map::new(), text.to_string());

    if !text.starts_with("---") {
        return empty;
    }

    // Find closing ---
    let rest = &text[3..];
    let end = match rest.find("\n---") {
        Some(pos) => pos,
        None => return empty,
    };

    let yaml_str = &rest[..end];
    let content_start = 3 + end + 4; // skip opening ---, yaml, \n---
    let doc_content = if content_start < text.len() {
        text[content_start..].trim_start_matches('\n').to_string()
    } else {
        String::new()
    };

    // Parse YAML into a JSON map
    match serde_yaml::from_str::<serde_json::Value>(yaml_str) {
        Ok(serde_json::Value::Object(map)) => (map, doc_content),
        _ => (serde_json::Map::new(), text.to_string()),
    }
}

/// Extract the first H1 heading from markdown content.
fn extract_first_heading(content: &str) -> Option<String> {
    let re = Regex::new(r"(?m)^#\s+(.+)$").ok()?;
    re.captures(content).map(|cap| cap[1].trim().to_string())
}

/// Convert filename stem to a readable title.
fn title_from_filename(path: &Path) -> String {
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("Untitled");
    stem.replace(['-', '_'], " ")
}

/// Extract tags from frontmatter metadata.
fn extract_tags(metadata: &serde_json::Map<String, serde_json::Value>) -> Vec<String> {
    match metadata.get("tags") {
        Some(serde_json::Value::Array(arr)) => arr
            .iter()
            .filter_map(|v| v.as_str().map(|s| s.trim().to_string()))
            .filter(|s| !s.is_empty())
            .collect(),
        Some(serde_json::Value::String(s)) => s
            .split(',')
            .map(|t| t.trim().to_string())
            .filter(|t| !t.is_empty())
            .collect(),
        _ => vec![],
    }
}

/// Read a file as UTF-8, replacing invalid bytes with the Unicode replacement character.
fn read_lossy(path: &Path) -> Result<String, ColibriError> {
    let bytes = std::fs::read(path).map_err(|e| {
        ColibriError::Source(format!("Failed to read {}: {e}", path.display()))
    })?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

/// Extract title from a YAML document (e.g., OpenAPI spec).
fn extract_yaml_title(content: &str) -> Option<String> {
    // Try info.title (OpenAPI convention)
    let info_re = Regex::new(r"(?m)^info:\s*\n(?:[ \t]+\S.*\n)*?[ \t]+title:\s*(.+)").ok()?;
    if let Some(cap) = info_re.captures(content) {
        let title = cap[1].trim().trim_matches(|c| c == '\'' || c == '"');
        if !title.is_empty() {
            return Some(title.to_string());
        }
    }

    // Fallback: top-level title key
    let title_re = Regex::new(r"(?m)^title:\s*(.+)").ok()?;
    title_re.captures(content).map(|cap| {
        cap[1]
            .trim()
            .trim_matches(|c| c == '\'' || c == '"')
            .to_string()
    })
}
