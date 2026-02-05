//! Manifest-based change tracking for incremental indexing.
//!
//! Tracks which files have been indexed, their modification times, and
//! content hashes. The manifest is stored as JSON in the data directory.
//! Mirrors the Python `manifest.py` module.

use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};

use chrono::Utc;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::ColibriError;

/// Tracking state for a single indexed file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileEntry {
    pub mtime: f64,
    pub content_hash: String,
    pub chunk_count: usize,
    pub indexed_at: String,
}

/// Tracks which files have been indexed and their state.
#[derive(Debug, Serialize, Deserialize)]
pub struct Manifest {
    pub version: u32,
    pub indexed_at: String,
    pub files: HashMap<String, FileEntry>,
}

impl Default for Manifest {
    fn default() -> Self {
        Self {
            version: 2,
            indexed_at: String::new(),
            files: HashMap::new(),
        }
    }
}

impl Manifest {
    /// Create an empty v2 manifest.
    pub fn new() -> Self {
        Self::default()
    }

    /// Load manifest from disk, or return an empty one if missing.
    pub fn load(manifest_path: &Path) -> Result<Self, ColibriError> {
        if !manifest_path.exists() {
            return Ok(Self::new());
        }
        let text = std::fs::read_to_string(manifest_path)?;
        let manifest: Self = serde_json::from_str(&text)?;
        Ok(manifest)
    }

    /// Persist manifest to disk.
    pub fn save(&mut self, manifest_path: &Path) -> Result<(), ColibriError> {
        if let Some(parent) = manifest_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let indexed_at = Utc::now().to_rfc3339();
        self.indexed_at = indexed_at;
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(manifest_path, json)?;
        Ok(())
    }

    /// Check if a file has changed since last indexing.
    ///
    /// Compares mtime first (fast), then content hash if mtime differs.
    pub fn is_file_changed(&self, key: &str, abs_path: &Path) -> bool {
        let entry = match self.files.get(key) {
            Some(e) => e,
            None => return true, // new file
        };

        let stat = match abs_path.metadata() {
            Ok(m) => m,
            Err(_) => return true, // can't stat — treat as changed
        };

        let mtime = file_mtime(&stat);
        if (mtime - entry.mtime).abs() > f64::EPSILON {
            // mtime changed — check content hash
            match compute_hash(abs_path) {
                Ok(hash) => hash != entry.content_hash,
                Err(_) => true,
            }
        } else {
            false
        }
    }

    /// Return `true` if the file path exists in the manifest.
    pub fn is_file_known(&self, key: &str) -> bool {
        self.files.contains_key(key)
    }

    /// Record a file as indexed (or update an existing entry).
    pub fn record_file(
        &mut self,
        key: &str,
        abs_path: &Path,
        chunk_count: usize,
    ) -> Result<(), ColibriError> {
        let stat = abs_path.metadata()?;
        let mtime = file_mtime(&stat);
        let content_hash = compute_hash(abs_path)?;

        self.files.insert(
            key.to_string(),
            FileEntry {
                mtime,
                content_hash,
                chunk_count,
                indexed_at: Utc::now().to_rfc3339(),
            },
        );
        Ok(())
    }

    /// Remove a file entry from the manifest.
    pub fn remove_file(&mut self, key: &str) {
        self.files.remove(key);
    }
}

/// Compute SHA-256 hash of file contents (streaming).
///
/// Returns format: `"sha256:<hexdigest>"`.
pub fn compute_hash(path: &Path) -> Result<String, ColibriError> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 8192];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let hex = format!("{:x}", hasher.finalize());
    Ok(format!("sha256:{hex}"))
}

/// Get manifest file path.
pub fn get_manifest_path(data_dir: &Path) -> PathBuf {
    data_dir.join("manifest.json")
}

// ---------------------------------------------------------------------------
// Key helpers (v2): namespace keys by source_id
// ---------------------------------------------------------------------------

/// Stable 12-hex source identifier for a given source root.
///
/// Matches the Python `source_id_for_root()` exactly.
pub fn source_id_for_root(root: &Path) -> String {
    let resolved = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    let mut hasher = Sha256::new();
    hasher.update(resolved.to_string_lossy().as_bytes());
    let hex = format!("{:x}", hasher.finalize());
    hex[..12].to_string()
}

/// Create a namespaced manifest key.
pub fn make_key(source_id: &str, rel_path: &str) -> String {
    format!("{source_id}:{rel_path}")
}

/// Split a namespaced manifest key.
pub fn split_key(key: &str) -> (&str, &str) {
    key.split_once(':').unwrap_or(("", key))
}

/// Check if a key is namespaced (v2 format).
#[allow(dead_code)]
pub fn is_namespaced_key(key: &str) -> bool {
    // 12 hex chars followed by colon
    key.len() > 13 && key.as_bytes()[12] == b':' && key[..12].chars().all(|c| c.is_ascii_hexdigit())
}

/// Build a manifest signature for change tracking.
///
/// Returns `{rel_path: (content_hash, chunk_count)}` for all files.
pub fn manifest_signature(manifest: &Manifest) -> HashMap<String, (String, usize)> {
    manifest
        .files
        .iter()
        .map(|(k, e)| (k.clone(), (e.content_hash.clone(), e.chunk_count)))
        .collect()
}

/// Extract mtime as f64 from file metadata.
#[cfg(unix)]
fn file_mtime(metadata: &std::fs::Metadata) -> f64 {
    use std::os::unix::fs::MetadataExt;
    metadata.mtime() as f64 + (metadata.mtime_nsec() as f64 / 1_000_000_000.0)
}

#[cfg(not(unix))]
fn file_mtime(metadata: &std::fs::Metadata) -> f64 {
    metadata
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}
