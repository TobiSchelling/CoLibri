//! Canonical markdown persistence for plugin ingestion.

use std::collections::HashSet;
use std::path::PathBuf;

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::config::AppConfig;
use crate::error::ColibriError;
use crate::metadata_store::{DocumentUpsert, MetadataStore};
use crate::plugin_host::DocumentEnvelope;

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

    let trimmed = out.trim_matches('-');
    if trimmed.is_empty() {
        "unnamed".into()
    } else {
        trimmed.to_string()
    }
}

fn canonical_rel_path(envelope: &DocumentEnvelope) -> PathBuf {
    let classification = safe_component(&envelope.metadata.classification, 24);
    let plugin = safe_component(&envelope.source.plugin_id, 48);
    let connector_slug = safe_component(&envelope.source.connector_instance, 32);
    let connector_hash = short_hash(&envelope.source.connector_instance, 12);
    let connector = if connector_slug == "unnamed" {
        connector_hash
    } else {
        format!("{connector_slug}-{connector_hash}")
    };
    let file_hash = short_hash(&envelope.document.doc_id, 24);

    PathBuf::from(classification)
        .join(plugin)
        .join(connector)
        .join(format!("{file_hash}.md"))
}

fn build_document_upsert(
    envelope: &DocumentEnvelope,
    markdown_rel_path: &str,
    existing_created_at: Option<String>,
) -> Result<DocumentUpsert, ColibriError> {
    let tags_json = serde_json::to_string(&envelope.metadata.tags.clone().unwrap_or_default())?;
    let acl_tags_json =
        serde_json::to_string(&envelope.metadata.acl_tags.clone().unwrap_or_default())?;

    Ok(DocumentUpsert {
        doc_id: envelope.document.doc_id.clone(),
        plugin_id: envelope.source.plugin_id.clone(),
        connector_instance: envelope.source.connector_instance.clone(),
        external_id: envelope.source.external_id.clone(),
        title: envelope.document.title.clone(),
        content_hash: envelope.document.content_hash.clone(),
        source_updated_at: envelope.document.source_updated_at.clone(),
        deleted: envelope.document.deleted,
        classification: envelope.metadata.classification.clone(),
        doc_type: envelope.metadata.doc_type.clone(),
        markdown_path: markdown_rel_path.to_string(),
        uri: envelope.source.uri.clone(),
        tags_json,
        acl_tags_json,
        language: envelope.metadata.language.clone(),
        created_at: existing_created_at,
    })
}

/// Persist validated plugin envelopes into canonical storage and metadata DB.
pub fn ingest_envelopes(
    config: &AppConfig,
    envelopes: &[DocumentEnvelope],
    dry_run: bool,
) -> Result<CanonicalIngestReport, ColibriError> {
    if !dry_run {
        config.ensure_storage_layout()?;
    }

    let store = if config.metadata_db_path.exists() {
        match MetadataStore::open(&config.metadata_db_path) {
            Ok(s) => Some(s),
            Err(_) if dry_run => None,
            Err(e) => return Err(e),
        }
    } else {
        None
    };
    if !dry_run && store.is_none() {
        return Err(ColibriError::Config(format!(
            "Metadata DB missing after bootstrap: {}",
            config.metadata_db_path.display()
        )));
    }

    let mut seen_doc_ids = HashSet::new();

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

    for envelope in envelopes {
        if !seen_doc_ids.insert(envelope.document.doc_id.clone()) {
            report.duplicate_doc_ids += 1;
        }

        let existing = store
            .as_ref()
            .map(|s| s.get_document_state(&envelope.document.doc_id))
            .transpose()?
            .flatten();
        let existing_hash = existing.as_ref().map(|x| x.content_hash.clone());
        let existing_created_at = existing.as_ref().and_then(|x| x.created_at.clone());

        let rel_path = existing
            .as_ref()
            .map(|x| x.markdown_path.clone())
            .unwrap_or_else(|| canonical_rel_path(envelope).to_string_lossy().to_string());

        let abs_path = config.canonical_dir.join(&rel_path);

        if envelope.document.deleted {
            report.tombstoned += 1;
            if abs_path.exists() {
                report.deleted_files += 1;
                if !dry_run {
                    std::fs::remove_file(&abs_path)?;
                }
            }

            if !dry_run {
                if let Some(store) = &store {
                    let row = build_document_upsert(envelope, &rel_path, existing_created_at)?;
                    store.upsert_document(&row)?;
                    store.delete_document_blob(&envelope.document.doc_id)?;
                }
            }
            continue;
        }

        let unchanged = existing_hash
            .as_deref()
            .is_some_and(|hash| hash == envelope.document.content_hash)
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

        if !dry_run {
            if let Some(store) = &store {
                let row = build_document_upsert(envelope, &rel_path, existing_created_at)?;
                store.upsert_document(&row)?;
                store.upsert_document_blob(
                    &envelope.document.doc_id,
                    &rel_path,
                    envelope.document.markdown.len() as u64,
                    &envelope.document.content_hash,
                )?;
            }
        }
    }

    if !dry_run {
        if let Some(store) = &store {
            let _ = store.document_count()?;
            store.touch_updated_at()?;
        }
    }

    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::{canonical_rel_path, safe_component};
    use crate::plugin_host::{
        DocumentEnvelope, EnvelopeDocument, EnvelopeMetadata, EnvelopeSource,
    };

    fn sample_envelope() -> DocumentEnvelope {
        DocumentEnvelope {
            schema_version: 1,
            source: EnvelopeSource {
                plugin_id: "filesystem_documents".into(),
                connector_instance: "/tmp/My Folder".into(),
                external_id: "docs/readme.md".into(),
                uri: None,
            },
            document: EnvelopeDocument {
                doc_id: "filesystem_documents:docs/readme.md".into(),
                title: "Readme".into(),
                markdown: "# Hi".into(),
                content_hash:
                    "sha256:0000000000000000000000000000000000000000000000000000000000000000".into(),
                source_updated_at: "2026-02-18T08:00:00Z".into(),
                deleted: false,
            },
            metadata: EnvelopeMetadata {
                doc_type: "note".into(),
                classification: "internal".into(),
                tags: None,
                language: None,
                acl_tags: None,
            },
        }
    }

    #[test]
    fn safe_component_normalizes_input() {
        assert_eq!(safe_component(" My Folder  / Docs ", 64), "my-folder-docs");
        assert_eq!(safe_component("___", 64), "___");
    }

    #[test]
    fn canonical_path_is_stable_and_scoped() {
        let path = canonical_rel_path(&sample_envelope());
        let path_str = path.to_string_lossy();
        assert!(path_str.starts_with("internal/filesystem_documents/"));
        assert!(path_str.ends_with(".md"));
    }
}
