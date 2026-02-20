//! Shared serving-alignment checks for embedding profiles.

use crate::config::{AppConfig, SCHEMA_VERSION};
use crate::error::ColibriError;
use crate::index_meta::read_index_meta;

#[derive(Debug, Clone)]
pub struct ServeReadyProfileCheck {
    pub profile_id: String,
    pub queryable: bool,
    pub issues: Vec<String>,
    pub schema_version: Option<u32>,
    pub file_count: Option<u64>,
    pub chunk_count: Option<u64>,
    pub index_path: String,
}

pub fn profile_checks(config: &AppConfig) -> Result<Vec<ServeReadyProfileCheck>, ColibriError> {
    let mut profile_ids: Vec<String> = config.embedding_profiles.keys().cloned().collect();
    profile_ids.sort();

    let mut out = Vec::new();

    for profile_id in profile_ids {
        let profile = config.embedding_profile(&profile_id)?;
        let mut issues = Vec::new();

        let index_path = config.lancedb_dir_for_profile(&profile_id);
        let meta = read_index_meta(&index_path)?;
        let schema_version = meta
            .get("schema_version")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32);
        let file_count = meta.get("file_count").and_then(|v| v.as_u64());
        let chunk_count = meta.get("chunk_count").and_then(|v| v.as_u64());
        let embedding_model = meta
            .get("embedding_model")
            .and_then(|v| v.as_str())
            .map(ToOwned::to_owned);

        if meta.is_empty() {
            issues.push(format!(
                "index metadata missing at {}",
                index_path.display()
            ));
        } else if schema_version != Some(SCHEMA_VERSION) {
            issues.push(format!(
                "schema outdated (index v{}, expected v{})",
                schema_version.unwrap_or(0),
                SCHEMA_VERSION
            ));
        } else if embedding_model.as_deref().unwrap_or("").is_empty() {
            issues.push("embedding_model missing in index metadata".into());
        } else if embedding_model.as_deref() != Some(profile.model.as_str()) {
            issues.push(format!(
                "embedding model mismatch (index='{}', config='{}')",
                embedding_model.clone().unwrap_or_default(),
                profile.model
            ));
        }

        out.push(ServeReadyProfileCheck {
            profile_id,
            queryable: issues.is_empty(),
            issues,
            schema_version,
            file_count,
            chunk_count,
            index_path: index_path.display().to_string(),
        });
    }

    Ok(out)
}
