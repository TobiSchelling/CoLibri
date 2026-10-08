//! Canonical markdown store: where each document's markdown lives, plus
//! the identity and hashing helpers shared by mirrors and the library.

use std::path::PathBuf;

use sha2::{Digest, Sha256};

use crate::metadata_store::DocumentRecord;

/// SHA-256 content hash in the stored format (`sha256:{hex}`).
pub fn content_hash(markdown: &str) -> String {
    format!("sha256:{}", sha256_hex(markdown))
}

fn sha256_hex(input: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn short_hash(input: &str, len: usize) -> String {
    let hex = sha256_hex(input);
    let n = len.min(hex.len());
    hex[..n].to_string()
}

fn safe_component(input: &str, max_len: usize) -> String {
    let mut out = String::new();
    let mut prev_sep = false;
    for ch in input.chars() {
        let normalized = if ch.is_ascii_alphanumeric() {
            Some(ch.to_ascii_lowercase())
        } else if ch == '-' || ch == '_' || ch == '.' {
            Some(ch)
        } else if ch.is_whitespace() || ch == '/' || ch == '\\' {
            Some('-')
        } else {
            None
        };

        let Some(c) = normalized else {
            continue;
        };
        if c == '-' {
            if prev_sep || out.is_empty() {
                continue;
            }
            prev_sep = true;
            out.push(c);
        } else {
            prev_sep = false;
            out.push(c);
        }

        if out.len() >= max_len {
            break;
        }
    }

    // Dots are trimmed too, so `.` or `..` can never become a path component.
    let trimmed = out.trim_matches(|c| c == '-' || c == '.');
    if trimmed.is_empty() {
        "unnamed".into()
    } else {
        trimmed.to_string()
    }
}

/// True when `b` only differs from `a` in its update/seen timestamps.
pub(crate) fn same_except_timestamps(a: &DocumentRecord, b: &DocumentRecord) -> bool {
    let mut b = b.clone();
    b.updated_at.clone_from(&a.updated_at);
    b.last_seen_at.clone_from(&a.last_seen_at);
    *a == b
}

/// Document id within CoLibri: `<collection>:<key>`.
pub fn doc_id_for(collection: &str, key: &str) -> String {
    format!("{collection}:{key}")
}

/// Canonical markdown location relative to the canonical dir:
/// `<collection>/<sha256(doc_id)[..24]>.md`.
pub fn canonical_rel_path(collection: &str, doc_id: &str) -> PathBuf {
    PathBuf::from(safe_component(collection, 48)).join(format!("{}.md", short_hash(doc_id, 24)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safe_component_normalizes_input() {
        assert_eq!(safe_component(" My Folder  / Docs ", 64), "my-folder-docs");
        assert_eq!(safe_component("___", 64), "___");
        assert_eq!(safe_component("..", 64), "unnamed");
    }

    #[test]
    fn canonical_path_is_scoped_by_collection() {
        let path = canonical_rel_path("vault", "vault:docs/readme.md");
        let path_str = path.to_string_lossy();
        assert!(path_str.starts_with("vault/"));
        assert!(path_str.ends_with(".md"));
        assert_eq!(path, canonical_rel_path("vault", "vault:docs/readme.md"));
    }

    #[test]
    fn content_hash_format_and_timestamp_insensitive_comparison() {
        let h = content_hash("# Hi");
        assert!(h.starts_with("sha256:") && h.len() == 71);
        let mut a = DocumentRecord::new("d", "c", "k");
        let mut b = a.clone();
        b.updated_at = "later".into();
        b.last_seen_at = Some("later".into());
        assert!(same_except_timestamps(&a, &b));
        a.title = "changed".into();
        assert!(!same_except_timestamps(&a, &b));
    }
}
