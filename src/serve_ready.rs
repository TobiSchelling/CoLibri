//! Is the index ready to serve searches with the configured embedding model?

use serde::Serialize;

use crate::config::{AppConfig, SCHEMA_VERSION};
use crate::error::ColibriError;
use crate::index_meta::read_index_meta;

#[derive(Debug, Clone, Serialize)]
pub struct ServeReadyCheck {
    pub queryable: bool,
    pub issues: Vec<String>,
    pub schema_version: Option<u32>,
    pub chunk_count: Option<u64>,
    pub embedding_model: Option<String>,
    pub index_path: String,
}

/// Read-only check of the index metadata against the current config.
pub fn check(config: &AppConfig) -> Result<ServeReadyCheck, ColibriError> {
    let (meta, unreadable) = match read_index_meta(&config.index_dir) {
        Ok(meta) => (meta, None),
        Err(e) => (serde_json::Map::new(), Some(e)),
    };
    let schema_version = meta
        .get("schema_version")
        .and_then(|v| v.as_u64())
        .map(|v| v as u32);
    let chunk_count = meta.get("chunk_count").and_then(|v| v.as_u64());
    let embedding_model = meta
        .get("embedding_model")
        .and_then(|v| v.as_str())
        .map(ToOwned::to_owned);

    let issue = if let Some(e) = unreadable {
        Some(format!("index metadata unreadable ({e}); reindex needed"))
    } else if meta.is_empty() {
        Some(format!(
            "no index yet at {} (ingest content first)",
            config.index_dir.display()
        ))
    } else if schema_version != Some(SCHEMA_VERSION) {
        Some(format!(
            "index schema v{} but this colibri needs v{SCHEMA_VERSION} (reindex needed)",
            schema_version.unwrap_or(0)
        ))
    } else if embedding_model.as_deref() != Some(config.embedding_model.as_str()) {
        Some(format!(
            "index built with model '{}', config uses '{}' (reindex needed)",
            embedding_model.clone().unwrap_or_default(),
            config.embedding_model
        ))
    } else if !config.index_dir.join("chunks.lance").exists() {
        Some("index is empty (nothing has been embedded yet)".into())
    } else {
        None
    };

    Ok(ServeReadyCheck {
        queryable: issue.is_none(),
        issues: issue.into_iter().collect(),
        schema_version,
        chunk_count,
        embedding_model,
        index_path: config.index_dir.display().to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index_meta::write_index_meta;

    #[test]
    fn missing_index_is_not_queryable() {
        let dir = tempfile::TempDir::new().unwrap();
        let cfg = AppConfig::for_test(dir.path());
        let c = check(&cfg).unwrap();
        assert!(!c.queryable);
        assert!(c.issues[0].contains("no index yet"));
    }

    #[test]
    fn matching_index_is_queryable_and_model_mismatch_is_not() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut cfg = AppConfig::for_test(dir.path());
        std::fs::create_dir_all(&cfg.index_dir).unwrap();
        write_index_meta(
            &cfg.index_dir,
            &cfg.embedding_model,
            &serde_json::Map::new(),
        )
        .unwrap();
        assert!(check(&cfg).unwrap().issues[0].contains("index is empty"));
        std::fs::create_dir_all(cfg.index_dir.join("chunks.lance")).unwrap();
        assert!(check(&cfg).unwrap().queryable);

        cfg.embedding_model = "other".into();
        let c = check(&cfg).unwrap();
        assert!(!c.queryable);
        assert!(c.issues[0].contains("reindex needed"));
    }
}
