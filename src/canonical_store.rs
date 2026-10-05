//! Canonical markdown persistence for connector ingestion.

use std::collections::HashSet;
use std::path::PathBuf;

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::config::AppConfig;
use crate::envelope::DocumentEnvelope;
use crate::error::ColibriError;
use crate::metadata_store::{DocStatus, DocumentRecord, MetadataStore};

/// Summary of a canonical ingest run.
#[derive(Debug, Clone, Serialize)]
pub struct CanonicalIngestReport {
    pub processed: usize,
    pub written: usize,
    pub unchanged: usize,
    pub tombstoned: usize,
    pub deleted_files: usize,
    pub duplicate_doc_ids: usize,
    pub dry_run: bool,
    pub canonical_dir: String,
    pub metadata_db_path: String,
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

/// Rows written per SQLite transaction during ingest.
const COMMIT_EVERY: usize = 200;

/// True when `b` only differs from `a` in its update/seen timestamps.
fn same_except_timestamps(a: &DocumentRecord, b: &DocumentRecord) -> bool {
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

fn source_path_from_uri(uri: Option<&str>) -> Option<String> {
    uri.map(|u| u.strip_prefix("file://").unwrap_or(u).to_string())
}

fn build_record(
    envelope: &DocumentEnvelope,
    collection: &str,
    doc_id: &str,
    markdown_rel_path: &str,
    existing: Option<&DocumentRecord>,
) -> Result<DocumentRecord, ColibriError> {
    let mut doc = match existing {
        // Keep identity, creation time and index state of the existing row.
        Some(prev) => prev.clone(),
        None => DocumentRecord::new(doc_id, collection, &envelope.source.external_id),
    };
    doc.collection = collection.to_string();
    doc.key = envelope.source.external_id.clone();
    doc.source_path = source_path_from_uri(envelope.source.uri.as_deref());
    doc.title = envelope.document.title.clone();
    doc.tags_json = serde_json::to_string(&envelope.metadata.tags.clone().unwrap_or_default())?;
    doc.language = envelope.metadata.language.clone();
    doc.frontmatter_json = match &envelope.metadata.frontmatter {
        Some(map) => serde_json::to_string(map)?,
        None => "{}".to_string(),
    };
    doc.doc_type = envelope.metadata.doc_type.clone();
    doc.source_updated_at = Some(envelope.document.source_updated_at.clone());
    doc.content_hash = envelope.document.content_hash.clone();
    doc.markdown_path = markdown_rel_path.to_string();
    doc.updated_at = chrono::Utc::now().to_rfc3339();
    doc.last_seen_at = Some(doc.updated_at.clone());
    if envelope.document.deleted {
        doc.status = DocStatus::Removed;
        doc.removed_reason = Some("source_deleted".into());
    } else {
        doc.status = DocStatus::Active;
        doc.removed_reason = None;
    }
    Ok(doc)
}

/// Persist connector envelopes into the canonical store and metadata DB as
/// documents of `collection`. With `dry_run`, only report what would change;
/// `store` may then be `None` when no data exists yet.
pub fn ingest_envelopes(
    config: &AppConfig,
    store: Option<&MetadataStore>,
    collection: &str,
    envelopes: &[DocumentEnvelope],
    dry_run: bool,
) -> Result<CanonicalIngestReport, ColibriError> {
    if !dry_run && store.is_none() {
        return Err(ColibriError::Config(
            "ingest needs a writable metadata DB".into(),
        ));
    }

    let mut report = CanonicalIngestReport {
        processed: envelopes.len(),
        written: 0,
        unchanged: 0,
        tombstoned: 0,
        deleted_files: 0,
        duplicate_doc_ids: 0,
        dry_run,
        canonical_dir: config.canonical_dir.display().to_string(),
        metadata_db_path: config.metadata_db_path.display().to_string(),
    };

    let writable = if dry_run { None } else { store };
    let mut tx = writable.map(MetadataStore::begin).transpose()?;
    let mut pending_writes = 0usize;
    let mut seen_doc_ids = HashSet::new();

    for envelope in envelopes {
        let doc_id = doc_id_for(collection, &envelope.source.external_id);
        if !seen_doc_ids.insert(doc_id.clone()) {
            report.duplicate_doc_ids += 1;
        }

        let existing = match store {
            Some(s) => s.get_document(&doc_id)?,
            None => None,
        };
        let rel_path = canonical_rel_path(collection, &doc_id)
            .to_string_lossy()
            .to_string();
        let abs_path = config.canonical_dir.join(&rel_path);

        if envelope.document.deleted {
            report.tombstoned += 1;
            if abs_path.exists() {
                report.deleted_files += 1;
                if !dry_run {
                    std::fs::remove_file(&abs_path)?;
                }
            }
        } else {
            let unchanged = existing
                .as_ref()
                .is_some_and(|prev| prev.content_hash == envelope.document.content_hash)
                && abs_path.exists();
            if unchanged {
                report.unchanged += 1;
            } else {
                report.written += 1;
                if !dry_run {
                    if let Some(parent) = abs_path.parent() {
                        std::fs::create_dir_all(parent)?;
                    }
                    std::fs::write(&abs_path, &envelope.document.markdown)?;
                }
            }
        }

        if let Some(s) = writable {
            let record = build_record(envelope, collection, &doc_id, &rel_path, existing.as_ref())?;
            if existing
                .as_ref()
                .is_some_and(|prev| same_except_timestamps(prev, &record))
            {
                continue;
            }
            s.upsert_document(&record)?;
            // Commit regularly so readers never wait long on the write lock.
            pending_writes += 1;
            if pending_writes == COMMIT_EVERY {
                if let Some(t) = tx.take() {
                    t.commit()?;
                }
                tx = Some(s.begin()?);
                pending_writes = 0;
            }
        }
    }

    if let Some(tx) = tx {
        tx.commit()?;
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::envelope::{EnvelopeDocument, EnvelopeMetadata, EnvelopeSource};

    pub(crate) fn sample_envelope(external_id: &str, markdown: &str) -> DocumentEnvelope {
        DocumentEnvelope {
            schema_version: 1,
            source: EnvelopeSource {
                plugin_id: "filesystem_documents".into(),
                connector_instance: "/tmp/My Folder".into(),
                external_id: external_id.into(),
                uri: Some(format!("/tmp/My Folder/{external_id}")),
            },
            document: EnvelopeDocument {
                doc_id: format!("filesystem_documents:{external_id}"),
                title: "Readme".into(),
                markdown: markdown.into(),
                content_hash: crate::envelope::content_hash(markdown),
                source_updated_at: "2026-02-18T08:00:00Z".into(),
                deleted: false,
            },
            metadata: EnvelopeMetadata {
                doc_type: "note".into(),
                tags: Some(vec!["a".into()]),
                language: None,
                acl_tags: None,
                frontmatter: None,
            },
        }
    }

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
    fn ingest_writes_documents_and_detects_unchanged() {
        let dir = tempfile::TempDir::new().unwrap();
        let config = AppConfig::for_test(dir.path());
        let (_lock, store) = config.open_for_write().unwrap();
        let envs = vec![sample_envelope("docs/readme.md", "# Hi")];

        let first = ingest_envelopes(&config, Some(&store), "vault", &envs, false).unwrap();
        assert_eq!((first.written, first.unchanged), (1, 0));
        let doc = store.get_document("vault:docs/readme.md").unwrap().unwrap();
        assert_eq!(doc.key, "docs/readme.md");
        assert_eq!(
            doc.source_path.as_deref(),
            Some("/tmp/My Folder/docs/readme.md")
        );
        assert_eq!(doc.tags_json, r#"["a"]"#);
        assert!(config.canonical_dir.join(&doc.markdown_path).exists());

        store
            .mark_indexed(&doc.doc_id, &doc.content_hash, 3)
            .unwrap();
        let second = ingest_envelopes(&config, Some(&store), "vault", &envs, false).unwrap();
        assert_eq!((second.written, second.unchanged), (0, 1));
        let after = store.get_document("vault:docs/readme.md").unwrap().unwrap();
        assert!(after.is_index_current(), "re-ingest keeps index state");
        assert_eq!(
            after.updated_at, doc.updated_at,
            "unchanged rows are not rewritten"
        );
    }

    #[test]
    fn dry_run_writes_nothing() {
        let dir = tempfile::TempDir::new().unwrap();
        let config = AppConfig::for_test(dir.path());
        let envs = vec![sample_envelope("a.md", "# A")];
        let report = ingest_envelopes(&config, None, "vault", &envs, true).unwrap();
        assert_eq!(report.written, 1);
        assert!(!config.canonical_dir.exists());
        assert!(!config.metadata_db_path.exists());
    }
}
