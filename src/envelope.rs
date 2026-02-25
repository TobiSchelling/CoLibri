//! Document envelope types and validation.
//!
//! The `DocumentEnvelope` is the canonical exchange format between content
//! producers (connectors, plugins, CLI import) and the storage layer
//! (canonical store, metadata DB, indexer).

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[cfg(test)]
use crate::error::ColibriError;
#[cfg(test)]
use chrono::DateTime;
#[cfg(test)]
use regex::Regex;
#[cfg(test)]
use std::sync::OnceLock;

#[cfg(test)]
pub(crate) static CONTENT_HASH_RE: OnceLock<Regex> = OnceLock::new();

/// Returns the compiled regex for validating content hash format.
#[cfg(test)]
pub(crate) fn content_hash_regex() -> &'static Regex {
    CONTENT_HASH_RE.get_or_init(|| {
        Regex::new(r"^sha256:[a-f0-9]{64}$").expect("content hash regex must compile")
    })
}

#[derive(Debug, Deserialize, Serialize)]
pub struct EnvelopeSource {
    pub plugin_id: String,
    pub connector_instance: String,
    pub external_id: String,
    pub uri: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct EnvelopeDocument {
    pub doc_id: String,
    pub title: String,
    pub markdown: String,
    pub content_hash: String,
    pub source_updated_at: String,
    pub deleted: bool,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct EnvelopeMetadata {
    pub doc_type: String,
    pub classification: String,
    pub tags: Option<Vec<String>>,
    pub language: Option<String>,
    pub acl_tags: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct DocumentEnvelope {
    pub schema_version: u32,
    pub source: EnvelopeSource,
    pub document: EnvelopeDocument,
    pub metadata: EnvelopeMetadata,
}

/// Compute a SHA-256 content hash in the envelope format (`sha256:{hex}`).
pub fn content_hash(markdown: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(markdown.as_bytes());
    format!("sha256:{:x}", hasher.finalize())
}

/// Validate a deserialized `DocumentEnvelope` for structural correctness.
///
/// Checks schema version, plugin_id match, required fields, content_hash
/// format, RFC 3339 timestamp, and classification values.
#[cfg(test)]
pub fn validate(envelope: &DocumentEnvelope, expected_plugin_id: &str) -> Result<(), ColibriError> {
    if envelope.schema_version != 1 {
        return Err(ColibriError::Config(format!(
            "Invalid envelope schema_version {} (expected 1)",
            envelope.schema_version
        )));
    }
    if envelope.source.plugin_id != expected_plugin_id {
        return Err(ColibriError::Config(format!(
            "Envelope plugin_id mismatch: expected '{}', got '{}'",
            expected_plugin_id, envelope.source.plugin_id
        )));
    }
    if envelope.document.doc_id.trim().is_empty() {
        return Err(ColibriError::Config(
            "Envelope document.doc_id cannot be empty".into(),
        ));
    }
    if envelope.document.content_hash.trim().is_empty() {
        return Err(ColibriError::Config(
            "Envelope document.content_hash cannot be empty".into(),
        ));
    }
    if !content_hash_regex().is_match(&envelope.document.content_hash) {
        return Err(ColibriError::Config(format!(
            "Envelope content_hash has invalid format: {}",
            envelope.document.content_hash
        )));
    }

    if DateTime::parse_from_rfc3339(&envelope.document.source_updated_at).is_err() {
        return Err(ColibriError::Config(format!(
            "Envelope source_updated_at is not RFC3339: {}",
            envelope.document.source_updated_at
        )));
    }

    if envelope.metadata.doc_type.trim().is_empty() {
        return Err(ColibriError::Config(
            "Envelope metadata.doc_type cannot be empty".into(),
        ));
    }

    match envelope.metadata.classification.as_str() {
        "restricted" | "confidential" | "internal" | "public" => {}
        other => {
            return Err(ColibriError::Config(format!(
                "Envelope classification must be one of restricted/confidential/internal/public, got '{other}'"
            )))
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_envelope() -> DocumentEnvelope {
        DocumentEnvelope {
            schema_version: 1,
            source: EnvelopeSource {
                plugin_id: "test-plugin".into(),
                connector_instance: "instance-1".into(),
                external_id: "ext-1".into(),
                uri: None,
            },
            document: EnvelopeDocument {
                doc_id: "doc-1".into(),
                title: "Test Document".into(),
                markdown: "# Hello".into(),
                content_hash: content_hash("# Hello"),
                source_updated_at: "2026-02-18T00:00:00Z".into(),
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
    fn content_hash_produces_valid_format() {
        let hash = content_hash("hello world");
        assert!(hash.starts_with("sha256:"));
        assert_eq!(hash.len(), 7 + 64); // "sha256:" + 64 hex chars
    }

    #[test]
    fn content_hash_is_deterministic() {
        assert_eq!(content_hash("same input"), content_hash("same input"));
    }

    #[test]
    fn content_hash_differs_for_different_input() {
        assert_ne!(content_hash("aaa"), content_hash("bbb"));
    }

    #[test]
    fn validate_accepts_valid_envelope() {
        let env = sample_envelope();
        validate(&env, "test-plugin").unwrap();
    }

    #[test]
    fn validate_rejects_wrong_schema_version() {
        let mut env = sample_envelope();
        env.schema_version = 99;
        let err = validate(&env, "test-plugin").unwrap_err().to_string();
        assert!(err.contains("schema_version"));
    }

    #[test]
    fn validate_rejects_plugin_id_mismatch() {
        let env = sample_envelope();
        let err = validate(&env, "wrong-plugin").unwrap_err().to_string();
        assert!(err.contains("mismatch"));
    }

    #[test]
    fn validate_rejects_empty_doc_id() {
        let mut env = sample_envelope();
        env.document.doc_id = "  ".into();
        let err = validate(&env, "test-plugin").unwrap_err().to_string();
        assert!(err.contains("doc_id"));
    }

    #[test]
    fn validate_rejects_invalid_content_hash() {
        let mut env = sample_envelope();
        env.document.content_hash = "md5:abc".into();
        let err = validate(&env, "test-plugin").unwrap_err().to_string();
        assert!(err.contains("content_hash"));
    }

    #[test]
    fn validate_rejects_non_rfc3339_timestamp() {
        let mut env = sample_envelope();
        env.document.source_updated_at = "not-a-date".into();
        let err = validate(&env, "test-plugin").unwrap_err().to_string();
        assert!(err.contains("RFC3339"));
    }

    #[test]
    fn validate_rejects_empty_doc_type() {
        let mut env = sample_envelope();
        env.metadata.doc_type = "".into();
        let err = validate(&env, "test-plugin").unwrap_err().to_string();
        assert!(err.contains("doc_type"));
    }

    #[test]
    fn validate_rejects_invalid_classification() {
        let mut env = sample_envelope();
        env.metadata.classification = "secret".into();
        let err = validate(&env, "test-plugin").unwrap_err().to_string();
        assert!(err.contains("classification"));
    }

    #[test]
    fn validate_accepts_all_valid_classifications() {
        for cls in &["restricted", "confidential", "internal", "public"] {
            let mut env = sample_envelope();
            env.metadata.classification = cls.to_string();
            validate(&env, "test-plugin").unwrap();
        }
    }

    #[test]
    fn validate_rejects_empty_content_hash() {
        let mut env = sample_envelope();
        env.document.content_hash = "".into();
        let err = validate(&env, "test-plugin").unwrap_err().to_string();
        assert!(err.contains("content_hash"));
    }
}
