//! Content source abstraction.
//!
//! Defines the `ContentSource` trait and the `SourceDocument` struct that
//! all source implementations must use.

pub mod markdown;

use std::path::{Path, PathBuf};

/// Normalized document representation from any source.
///
/// Mirrors the Python `SourceDocument` dataclass — consistent metadata
/// regardless of the source type.
#[derive(Debug, Clone)]
pub struct SourceDocument {
    /// Path to the document (relative to source root).
    pub path: PathBuf,
    /// Document text content (without frontmatter).
    pub content: String,
    /// Document title.
    pub title: String,
    /// Document type (e.g., "book", "note").
    pub doc_type: String,
    /// Name of the content source.
    pub source_name: String,
    /// Type of source ("markdown", "obsidian", etc.).
    pub source_type: String,
    /// Folder within the source (first path component).
    pub folder: String,
    /// Additional metadata from frontmatter.
    pub metadata: serde_json::Map<String, serde_json::Value>,
    /// Tags if available.
    pub tags: Vec<String>,
}

/// Abstract interface for content sources.
///
/// Mirrors the Python `ContentSource` ABC. Implementations handle
/// different folder structures and conventions.
#[allow(dead_code)]
pub trait ContentSource {
    /// Human-readable name for this source.
    fn name(&self) -> &str;

    /// Base path for this content source.
    fn root_path(&self) -> &Path;

    /// Source type identifier (e.g., "markdown").
    fn source_type(&self) -> &str;

    /// List all indexable documents.
    ///
    /// Returns paths relative to `root_path()`.
    fn list_documents(&self) -> Vec<PathBuf>;

    /// Read and parse a document.
    ///
    /// `path` is relative to `root_path()`.
    fn read_document(&self, path: &Path) -> Result<SourceDocument, crate::error::ColibriError>;
}
