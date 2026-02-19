//! Shared serving-alignment checks for embedding profiles.

use std::collections::HashMap;

use crate::config::{AppConfig, SCHEMA_VERSION};
use crate::error::ColibriError;
use crate::index_meta::read_index_meta;
use crate::metadata_store::MetadataStore;

#[derive(Debug, Clone)]
pub struct ServeReadyProfileCheck {
    pub profile_id: String,
    pub queryable: bool,
    pub issues: Vec<String>,
    pub lifecycle_status: Option<String>,
    pub schema_version: Option<u32>,
    pub file_count: Option<u64>,
    pub chunk_count: Option<u64>,
    pub index_path: String,
}

pub fn profile_checks(config: &AppConfig) -> Result<Vec<ServeReadyProfileCheck>, ColibriError> {
    let generation_status = load_generation_status(config)?;
    let mut profile_ids: Vec<String> = config.embedding_profiles.keys().cloned().collect();
    profile_ids.sort();

    let mut out = Vec::new();

    for profile_id in profile_ids {
        let profile = config.embedding_profile(&profile_id)?;
        let mut issues = Vec::new();

        let lifecycle_status = generation_status.get(&profile_id).cloned();
        if !generation_status.is_empty() {
            match lifecycle_status.as_deref() {
                Some("ready") => {}
                Some(status) => issues.push(format!(
                    "lifecycle status for generation '{}' is '{}'",
                    config.active_generation, status
                )),
                None => issues.push(format!(
                    "missing generation metadata row for '{}'",
                    config.active_generation
                )),
            }
        }

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
            lifecycle_status,
            schema_version,
            file_count,
            chunk_count,
            index_path: index_path.display().to_string(),
        });
    }

    Ok(out)
}

fn load_generation_status(config: &AppConfig) -> Result<HashMap<String, String>, ColibriError> {
    if !config.metadata_db_path.exists() {
        return Ok(HashMap::new());
    }
    let store = MetadataStore::open(&config.metadata_db_path)?;
    let rows = store.list_generation_entries()?;
    let mut out = HashMap::new();
    for row in rows {
        if row.generation_id == config.active_generation {
            out.insert(row.embedding_profile_id, row.status);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::load_generation_status;
    use crate::config::{AppConfig, EmbeddingLocality, EmbeddingProfile};
    use std::collections::HashMap;
    use std::path::PathBuf;

    #[test]
    fn generation_status_empty_when_db_missing() {
        let cfg = AppConfig {
            sources: Vec::new(),
            plugin_jobs: Vec::new(),
            colibri_home: PathBuf::from("/tmp/colibri-test"),
            data_dir: PathBuf::from("/tmp/colibri-test"),
            canonical_dir: PathBuf::from("/tmp/colibri-test/canonical"),
            indexes_dir: PathBuf::from("/tmp/colibri-test/indexes"),
            state_dir: PathBuf::from("/tmp/colibri-test/state"),
            backups_dir: PathBuf::from("/tmp/colibri-test/backups"),
            logs_dir: PathBuf::from("/tmp/colibri-test/logs"),
            metadata_db_path: PathBuf::from("/tmp/colibri-test/metadata.db.DOES_NOT_EXIST"),
            active_generation: "gen_default".into(),
            index_dir_name: "lancedb".into(),
            embedding_profiles: {
                let mut m = HashMap::new();
                m.insert(
                    "local_default".into(),
                    EmbeddingProfile {
                        id: "local_default".into(),
                        provider: "ollama".into(),
                        endpoint: "http://localhost:11434".into(),
                        model: "bge-m3".into(),
                        locality: EmbeddingLocality::Local,
                    },
                );
                m
            },
            routing_policy: HashMap::new(),
            default_embedding_profile: "local_default".into(),
            lancedb_dir: PathBuf::from(
                "/tmp/colibri-test/indexes/gen_default/local_default/lancedb",
            ),
            ollama_base_url: "http://localhost:11434".into(),
            embedding_model: "bge-m3".into(),
            top_k: 10,
            similarity_threshold: 0.3,
            chunk_size: 3000,
            chunk_overlap: 200,
        };

        let m = load_generation_status(&cfg).expect("load status");
        assert!(m.is_empty());
    }
}
