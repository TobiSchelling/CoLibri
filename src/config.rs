//! Configuration loading from `~/.config/colibri/config.yaml`.
//!
//! Mirrors the Python `config.py` module: loads YAML config with defaults,
//! supports env var overrides, and produces the same derived paths.

use std::collections::HashMap;
use std::env;
use std::path::{Path, PathBuf};

use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::error::ColibriError;
use crate::metadata_store::{
    read_metadata_versions as read_metadata_versions_sqlite, MetadataStore,
};

/// Schema version — must match Python's `SCHEMA_VERSION` for cross-compat.
pub const SCHEMA_VERSION: u32 = 6;

/// Canonical storage schema version.
pub const CANONICAL_SCHEMA_VERSION: u32 = 1;

/// Indexing pipeline schema version.
pub const PIPELINE_SCHEMA_VERSION: u32 = 1;

/// Serving schema version.
pub const SERVING_SCHEMA_VERSION: u32 = 1;

/// Root metadata DB format version.
pub const METADATA_DB_FORMAT_VERSION: u32 = 1;

/// Default active index generation identifier.
pub const DEFAULT_ACTIVE_GENERATION: &str = "gen_default";

/// Root manifest schema version.
pub const ROOT_MANIFEST_VERSION: u32 = 4;

/// Migration check row.
#[derive(Debug, Clone, Serialize)]
pub struct MigrationCheck {
    pub component: String,
    pub current: Option<u32>,
    pub target: u32,
    pub status: String,
}

/// Migration inspection result.
#[derive(Debug, Clone, Serialize)]
pub struct MigrationReport {
    pub colibri_home: String,
    pub up_to_date: bool,
    pub checks: Vec<MigrationCheck>,
}

/// Raw YAML config structure.
#[derive(Debug, Deserialize)]
struct RawConfig {
    #[serde(default)]
    data: DataConfig,

    #[serde(default)]
    index: IndexConfig,

    #[serde(default)]
    ollama: OllamaConfig,

    #[serde(default)]
    retrieval: RetrievalConfig,

    #[serde(default)]
    chunking: ChunkingConfig,

    #[serde(default)]
    embeddings: EmbeddingsConfig,

    #[serde(default)]
    routing: RoutingConfig,

    #[serde(default)]
    plugins: PluginsConfig,
}

#[derive(Debug, Default, Deserialize)]
struct PluginsConfig {
    #[serde(default)]
    jobs: Vec<PluginJobRawConfig>,
}

#[derive(Debug, Deserialize)]
#[serde(default)]
struct PluginJobRawConfig {
    id: Option<String>,
    manifest: String,
    enabled: bool,
    config: Value,
}

impl Default for PluginJobRawConfig {
    fn default() -> Self {
        Self {
            id: None,
            manifest: String::new(),
            enabled: true,
            config: Value::Object(Map::new()),
        }
    }
}

/// A configured plugin sync job.
#[derive(Debug, Clone, Serialize)]
pub struct PluginJob {
    pub id: String,
    pub manifest: PathBuf,
    pub enabled: bool,
    pub config: Value,
}

#[derive(Debug, Default, Deserialize)]
struct DataConfig {
    directory: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct EmbeddingsConfig {
    #[serde(default)]
    profiles: Vec<EmbeddingProfileConfig>,
    default_profile: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(default)]
struct EmbeddingProfileConfig {
    id: String,
    provider: String,
    endpoint: String,
    model: String,
    locality: EmbeddingLocality,
}

impl Default for EmbeddingProfileConfig {
    fn default() -> Self {
        Self {
            id: String::new(),
            provider: default_embedding_provider(),
            endpoint: default_ollama_url(),
            model: default_embedding_model(),
            locality: EmbeddingLocality::Local,
        }
    }
}

fn default_embedding_provider() -> String {
    "ollama".into()
}

#[derive(Debug, Default, Deserialize)]
struct RoutingConfig {
    #[serde(default)]
    classification_profiles: HashMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct IndexConfig {
    #[serde(default = "default_index_dir")]
    directory: String,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            directory: default_index_dir(),
        }
    }
}

fn default_index_dir() -> String {
    "lancedb".into()
}

#[derive(Debug, Deserialize)]
struct OllamaConfig {
    #[serde(default = "default_ollama_url")]
    base_url: String,

    #[serde(default = "default_embedding_model")]
    embedding_model: String,
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            base_url: default_ollama_url(),
            embedding_model: default_embedding_model(),
        }
    }
}

fn default_ollama_url() -> String {
    "http://localhost:11434".into()
}

fn default_embedding_model() -> String {
    "bge-m3".into()
}

#[derive(Debug, Deserialize)]
struct RetrievalConfig {
    #[serde(default = "default_top_k")]
    top_k: usize,

    #[serde(default = "default_similarity_threshold")]
    similarity_threshold: f64,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            top_k: default_top_k(),
            similarity_threshold: default_similarity_threshold(),
        }
    }
}

fn default_top_k() -> usize {
    10
}

fn default_similarity_threshold() -> f64 {
    0.3
}

#[derive(Debug, Deserialize)]
struct ChunkingConfig {
    #[serde(default = "default_chunk_size")]
    chunk_size: usize,

    #[serde(default = "default_chunk_overlap")]
    chunk_overlap: usize,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            chunk_size: default_chunk_size(),
            chunk_overlap: default_chunk_overlap(),
        }
    }
}

fn default_chunk_size() -> usize {
    3000
}

fn default_chunk_overlap() -> usize {
    200
}

/// Embedding deployment location.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingLocality {
    #[default]
    Local,
    Cloud,
}

/// A resolved embedding profile used by indexing and query paths.
#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingProfile {
    pub id: String,
    pub provider: String,
    pub endpoint: String,
    pub model: String,
    pub locality: EmbeddingLocality,
}

fn default_colibri_home() -> PathBuf {
    let xdg = env::var("XDG_DATA_HOME").unwrap_or_else(|_| {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".local")
            .join("share")
            .to_string_lossy()
            .into_owned()
    });
    PathBuf::from(xdg).join("colibri")
}

fn resolve_colibri_home(raw_data_dir: Option<&str>) -> PathBuf {
    if let Ok(val) = env::var("COLIBRI_HOME") {
        return PathBuf::from(val);
    }
    if let Ok(val) = env::var("COLIBRI_DATA_DIR") {
        return PathBuf::from(val);
    }
    if let Some(dir) = raw_data_dir {
        return PathBuf::from(dir);
    }
    default_colibri_home()
}

fn normalize_generation_id(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return DEFAULT_ACTIVE_GENERATION.to_string();
    }
    if trimmed
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.')
    {
        return trimmed.to_string();
    }
    DEFAULT_ACTIVE_GENERATION.to_string()
}

fn active_generation_from_manifest(colibri_home: &std::path::Path) -> String {
    let manifest_path = colibri_home.join("manifest.json");
    let text = match std::fs::read_to_string(manifest_path) {
        Ok(t) => t,
        Err(_) => return DEFAULT_ACTIVE_GENERATION.to_string(),
    };
    let parsed: Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(_) => return DEFAULT_ACTIVE_GENERATION.to_string(),
    };
    parsed
        .get("active_generation")
        .and_then(|v| v.as_str())
        .map(normalize_generation_id)
        .unwrap_or_else(|| DEFAULT_ACTIVE_GENERATION.to_string())
}

fn normalize_classification(raw: &str) -> String {
    let normalized = raw.trim().to_ascii_lowercase();
    if normalized.is_empty() {
        "internal".into()
    } else {
        normalized
    }
}

fn resolve_embedding_profiles(
    raw: &RawConfig,
) -> Result<(HashMap<String, EmbeddingProfile>, String), ColibriError> {
    let mut profiles: HashMap<String, EmbeddingProfile> = HashMap::new();

    if raw.embeddings.profiles.is_empty() {
        let id = "local_default".to_string();
        profiles.insert(
            id.clone(),
            EmbeddingProfile {
                id: id.clone(),
                provider: default_embedding_provider(),
                endpoint: raw.ollama.base_url.clone(),
                model: raw.ollama.embedding_model.clone(),
                locality: EmbeddingLocality::Local,
            },
        );
    } else {
        for profile in &raw.embeddings.profiles {
            let id = profile.id.trim();
            if id.is_empty() {
                return Err(ColibriError::Config(
                    "Embedding profile id cannot be empty".into(),
                ));
            }
            profiles.insert(
                id.to_string(),
                EmbeddingProfile {
                    id: id.to_string(),
                    provider: if profile.provider.trim().is_empty() {
                        default_embedding_provider()
                    } else {
                        profile.provider.trim().to_string()
                    },
                    endpoint: if profile.endpoint.trim().is_empty() {
                        raw.ollama.base_url.clone()
                    } else {
                        profile.endpoint.trim().to_string()
                    },
                    model: if profile.model.trim().is_empty() {
                        raw.ollama.embedding_model.clone()
                    } else {
                        profile.model.trim().to_string()
                    },
                    locality: profile.locality,
                },
            );
        }
    }

    let default_profile = raw
        .embeddings
        .default_profile
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| {
            profiles
                .keys()
                .next()
                .cloned()
                .unwrap_or_else(|| "local_default".into())
        });

    if !profiles.contains_key(&default_profile) {
        return Err(ColibriError::Config(format!(
            "Default embedding profile '{default_profile}' not found in embeddings.profiles"
        )));
    }

    Ok((profiles, default_profile))
}

fn resolve_routing_policy(
    raw: &RawConfig,
    default_profile: &str,
    known_profiles: &HashMap<String, EmbeddingProfile>,
) -> Result<HashMap<String, String>, ColibriError> {
    let mut policy = HashMap::new();
    for class in ["restricted", "confidential", "internal", "public"] {
        policy.insert(class.to_string(), default_profile.to_string());
    }

    for (class, profile) in &raw.routing.classification_profiles {
        let class = normalize_classification(class);
        let profile = profile.trim().to_string();
        if !known_profiles.contains_key(&profile) {
            return Err(ColibriError::Config(format!(
                "Routing references unknown embedding profile '{profile}' for classification '{class}'"
            )));
        }
        policy.insert(class, profile);
    }

    enforce_local_only_routing(&policy, known_profiles)?;
    Ok(policy)
}

fn enforce_local_only_routing(
    policy: &HashMap<String, String>,
    known_profiles: &HashMap<String, EmbeddingProfile>,
) -> Result<(), ColibriError> {
    for class in ["restricted", "confidential"] {
        let Some(profile_id) = policy.get(class) else {
            continue;
        };
        let Some(profile) = known_profiles.get(profile_id) else {
            continue;
        };
        if profile.locality != EmbeddingLocality::Local {
            return Err(ColibriError::Config(format!(
                "Safety policy violation: classification '{class}' must route to a local embedding profile, but '{profile_id}' is cloud."
            )));
        }
    }
    Ok(())
}

fn resolve_plugin_jobs(
    raw: &RawConfig,
    config_path: &Path,
) -> Result<Vec<PluginJob>, ColibriError> {
    let config_dir = config_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));

    let mut jobs = Vec::new();
    for (idx, job) in raw.plugins.jobs.iter().enumerate() {
        let manifest_raw = job.manifest.trim();
        if manifest_raw.is_empty() {
            return Err(ColibriError::Config(format!(
                "plugins.jobs[{}].manifest cannot be empty",
                idx
            )));
        }

        let manifest_path = {
            let candidate = PathBuf::from(manifest_raw);
            if candidate.is_absolute() {
                candidate
            } else {
                config_dir.join(candidate)
            }
        };

        let id = job.id.as_deref().map(str::trim).filter(|s| !s.is_empty());
        let id = id
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| format!("job_{}", idx + 1));

        if !job.config.is_object() {
            return Err(ColibriError::Config(format!(
                "plugins.jobs[{}].config must be a JSON/YAML object",
                idx
            )));
        }

        jobs.push(PluginJob {
            id,
            manifest: manifest_path,
            enabled: job.enabled,
            config: job.config.clone(),
        });
    }

    Ok(jobs)
}

fn read_root_manifest_version(colibri_home: &std::path::Path) -> Result<Option<u32>, ColibriError> {
    let manifest_path = colibri_home.join("manifest.json");
    if !manifest_path.exists() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(manifest_path)?;
    let val: Value = serde_json::from_str(&text)?;
    Ok(val
        .get("version")
        .and_then(|v| v.as_u64())
        .map(|v| v as u32))
}

fn read_metadata_versions(
    metadata_db_path: &std::path::Path,
) -> Result<(Option<u32>, HashMap<String, u32>), ColibriError> {
    read_metadata_versions_sqlite(metadata_db_path)
}

fn check_status(current: Option<u32>, target: u32) -> String {
    match current {
        Some(v) if v >= target => "ok".to_string(),
        Some(v) => format!("pending(v{v} -> v{target})"),
        None => format!("pending(missing -> v{target})"),
    }
}

fn build_migration_report(
    colibri_home: &std::path::Path,
    manifest_version: Option<u32>,
    metadata_format_version: Option<u32>,
    schema_versions: &HashMap<String, u32>,
) -> MigrationReport {
    let canonical_current = schema_versions.get("canonical").copied();
    let pipeline_current = schema_versions.get("pipeline").copied();
    let serving_current = schema_versions.get("serving").copied();

    let checks = vec![
        MigrationCheck {
            component: "root_manifest".into(),
            current: manifest_version,
            target: ROOT_MANIFEST_VERSION,
            status: check_status(manifest_version, ROOT_MANIFEST_VERSION),
        },
        MigrationCheck {
            component: "metadata_format".into(),
            current: metadata_format_version,
            target: METADATA_DB_FORMAT_VERSION,
            status: check_status(metadata_format_version, METADATA_DB_FORMAT_VERSION),
        },
        MigrationCheck {
            component: "canonical_schema".into(),
            current: canonical_current,
            target: CANONICAL_SCHEMA_VERSION,
            status: check_status(canonical_current, CANONICAL_SCHEMA_VERSION),
        },
        MigrationCheck {
            component: "pipeline_schema".into(),
            current: pipeline_current,
            target: PIPELINE_SCHEMA_VERSION,
            status: check_status(pipeline_current, PIPELINE_SCHEMA_VERSION),
        },
        MigrationCheck {
            component: "serving_schema".into(),
            current: serving_current,
            target: SERVING_SCHEMA_VERSION,
            status: check_status(serving_current, SERVING_SCHEMA_VERSION),
        },
    ];

    let up_to_date = checks.iter().all(|c| c.status == "ok");

    MigrationReport {
        colibri_home: colibri_home.display().to_string(),
        up_to_date,
        checks,
    }
}

fn ensure_root_manifest(
    colibri_home: &std::path::Path,
    active_generation: &str,
) -> Result<(), ColibriError> {
    let manifest_path = colibri_home.join("manifest.json");
    let mut changed = false;

    let mut obj = if manifest_path.exists() {
        let text = std::fs::read_to_string(&manifest_path)?;
        match serde_json::from_str::<Value>(&text) {
            Ok(Value::Object(map)) => map,
            _ => Map::new(),
        }
    } else {
        Map::new()
    };

    let needs_version = match obj.get("version").and_then(|v| v.as_u64()) {
        Some(v) => v < ROOT_MANIFEST_VERSION as u64,
        None => true,
    };
    if needs_version {
        obj.insert("version".into(), Value::from(ROOT_MANIFEST_VERSION as u64));
        changed = true;
    }
    // Legacy keys from the old manifest-based indexer; keep the root manifest minimal.
    if obj.remove("indexed_at").is_some() {
        changed = true;
    }
    if obj.remove("files").is_some() {
        changed = true;
    }

    let normalized = normalize_generation_id(active_generation);
    if obj
        .get("active_generation")
        .and_then(|v| v.as_str())
        .map(normalize_generation_id)
        .as_deref()
        != Some(normalized.as_str())
    {
        obj.insert("active_generation".into(), Value::String(normalized));
        changed = true;
    }

    if changed || !manifest_path.exists() {
        let json = serde_json::to_string_pretty(&Value::Object(obj))?;
        std::fs::write(manifest_path, json)?;
    }
    Ok(())
}

fn ensure_metadata_db(
    metadata_db_path: &std::path::Path,
    embedding_profiles: &HashMap<String, EmbeddingProfile>,
    routing_policy: &HashMap<String, String>,
    default_embedding_profile: &str,
) -> Result<(), ColibriError> {
    if let Some(parent) = metadata_db_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let mut store = MetadataStore::open(metadata_db_path)?;
    match store.bootstrap(
        embedding_profiles,
        routing_policy,
        default_embedding_profile,
    ) {
        Ok(()) => Ok(()),
        Err(first_err) => {
            if metadata_db_path.exists() {
                let backup = metadata_db_path.with_extension("legacy-json.bak");
                let _ = std::fs::rename(metadata_db_path, backup);
                let mut store = MetadataStore::open(metadata_db_path)?;
                store.bootstrap(
                    embedding_profiles,
                    routing_policy,
                    default_embedding_profile,
                )?;
                Ok(())
            } else {
                Err(first_err)
            }
        }
    }
}

/// Resolved application configuration.
#[derive(Debug, Clone)]
pub struct AppConfig {
    pub plugin_jobs: Vec<PluginJob>,
    pub colibri_home: PathBuf,
    pub canonical_dir: PathBuf,
    pub indexes_dir: PathBuf,
    pub state_dir: PathBuf,
    pub backups_dir: PathBuf,
    pub logs_dir: PathBuf,
    pub metadata_db_path: PathBuf,
    pub active_generation: String,
    pub index_dir_name: String,
    pub embedding_profiles: HashMap<String, EmbeddingProfile>,
    pub routing_policy: HashMap<String, String>,
    pub default_embedding_profile: String,
    pub lancedb_dir: PathBuf,
    pub ollama_base_url: String,
    pub embedding_model: String,
    pub top_k: usize,
    pub similarity_threshold: f64,
    pub chunk_size: usize,
    pub chunk_overlap: usize,
}

impl AppConfig {
    /// Config file path.
    pub fn config_path() -> PathBuf {
        if let Ok(p) = env::var("COLIBRI_CONFIG_PATH") {
            let trimmed = p.trim();
            if !trimmed.is_empty() {
                return PathBuf::from(trimmed);
            }
        }
        if let Ok(p) = env::var("COLIBRI_CONFIG") {
            let trimmed = p.trim();
            if !trimmed.is_empty() {
                return PathBuf::from(trimmed);
            }
        }
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".config")
            .join("colibri")
            .join("config.yaml")
    }

    /// Resolve the configured vector index directory for an embedding profile.
    pub fn lancedb_dir_for_profile(&self, profile_id: &str) -> PathBuf {
        self.lancedb_dir_for_generation_profile(&self.active_generation, profile_id)
    }

    /// Resolve the configured vector index directory for a specific generation/profile.
    pub fn lancedb_dir_for_generation_profile(
        &self,
        generation_id: &str,
        profile_id: &str,
    ) -> PathBuf {
        self.indexes_dir
            .join(generation_id)
            .join(profile_id)
            .join(&self.index_dir_name)
    }

    /// Resolve embedding profile id for a data classification.
    pub fn resolve_embedding_profile_id(&self, classification: &str) -> String {
        let class = normalize_classification(classification);
        self.routing_policy
            .get(&class)
            .cloned()
            .unwrap_or_else(|| self.default_embedding_profile.clone())
    }

    /// Get embedding profile by id.
    pub fn embedding_profile(&self, profile_id: &str) -> Result<&EmbeddingProfile, ColibriError> {
        self.embedding_profiles
            .get(profile_id)
            .ok_or_else(|| ColibriError::Config(format!("Unknown embedding profile: {profile_id}")))
    }

    /// Ensure storage directories exist without mutating active generation pointer.
    pub fn ensure_storage_layout(&self) -> Result<(), ColibriError> {
        std::fs::create_dir_all(&self.colibri_home)?;
        std::fs::create_dir_all(&self.canonical_dir)?;
        std::fs::create_dir_all(&self.indexes_dir)?;
        std::fs::create_dir_all(&self.state_dir)?;
        std::fs::create_dir_all(&self.backups_dir)?;
        std::fs::create_dir_all(&self.logs_dir)?;
        std::fs::create_dir_all(&self.lancedb_dir)?;
        ensure_metadata_db(
            &self.metadata_db_path,
            &self.embedding_profiles,
            &self.routing_policy,
            &self.default_embedding_profile,
        )
    }

    /// Ensure storage directories and bootstrap files exist.
    pub fn ensure_directories(&self) -> Result<(), ColibriError> {
        self.ensure_storage_layout()?;
        ensure_root_manifest(&self.colibri_home, &self.active_generation)?;
        Ok(())
    }

    /// Inspect storage schema migration state without applying changes.
    pub fn inspect_migrations(&self) -> Result<MigrationReport, ColibriError> {
        let manifest_version = read_root_manifest_version(&self.colibri_home)?;
        let (metadata_format_version, schema_versions) =
            read_metadata_versions(&self.metadata_db_path)?;
        Ok(build_migration_report(
            &self.colibri_home,
            manifest_version,
            metadata_format_version,
            &schema_versions,
        ))
    }

    /// Apply migrations and return updated migration report.
    ///
    /// Creates a lightweight backup of migration-controlled files before changes.
    pub fn apply_migrations(&self) -> Result<MigrationReport, ColibriError> {
        let before = self.inspect_migrations()?;
        if !before.up_to_date {
            let backup_root = self
                .backups_dir
                .join(format!("migration-{}", Utc::now().format("%Y%m%dT%H%M%SZ")));
            std::fs::create_dir_all(&backup_root)?;

            let manifest = self.colibri_home.join("manifest.json");
            if manifest.exists() {
                std::fs::copy(&manifest, backup_root.join("manifest.json"))?;
            }
            if self.metadata_db_path.exists() {
                std::fs::copy(&self.metadata_db_path, backup_root.join("metadata.db"))?;
            }
        }

        self.ensure_directories()?;
        let after = self.inspect_migrations()?;

        if !before.up_to_date && self.metadata_db_path.exists() {
            let store = MetadataStore::open(&self.metadata_db_path)?;
            for check in &before.checks {
                let before_version = check.current.unwrap_or(0);
                if before_version >= check.target {
                    continue;
                }
                let after_version = after
                    .checks
                    .iter()
                    .find(|c| c.component == check.component)
                    .and_then(|c| c.current)
                    .unwrap_or(0);
                let success = after_version >= check.target;
                let notes = if success {
                    "migration applied"
                } else {
                    "migration still pending"
                };
                store.append_migration_log(
                    &check.component,
                    check.current,
                    check.target,
                    success,
                    Some(notes),
                )?;
            }
            store.touch_updated_at()?;
        }

        Ok(after)
    }
}

/// Update a plugin job's config in the YAML config file.
///
/// Reads the config file, finds the job by ID, replaces its `config` block
/// with the new JSON value, and writes back. Does not preserve YAML comments.
pub fn update_plugin_job_config(
    config_path: &Path,
    job_id: &str,
    new_config: &serde_json::Value,
) -> anyhow::Result<()> {
    use anyhow::Context;

    let text = std::fs::read_to_string(config_path)
        .with_context(|| format!("Failed to read config file: {}", config_path.display()))?;

    let mut doc: serde_yaml::Value = serde_yaml::from_str(&text)
        .with_context(|| format!("Failed to parse config YAML: {}", config_path.display()))?;

    // Navigate to plugins.jobs array
    let jobs = doc
        .get_mut("plugins")
        .and_then(|p| p.get_mut("jobs"))
        .and_then(|j| j.as_sequence_mut())
        .ok_or_else(|| anyhow::anyhow!("No plugins.jobs array found in config"))?;

    // Find the job by id
    let job = jobs
        .iter_mut()
        .find(|j| {
            j.get("id")
                .and_then(|id| id.as_str())
                .map(|id| id == job_id)
                .unwrap_or(false)
        })
        .ok_or_else(|| anyhow::anyhow!("Plugin job '{}' not found in config", job_id))?;

    // Convert serde_json::Value -> serde_yaml::Value
    let yaml_config: serde_yaml::Value =
        serde_yaml::to_value(new_config).context("Failed to convert config to YAML")?;

    // Replace the config block
    if let serde_yaml::Value::Mapping(ref mut map) = job {
        map.insert(serde_yaml::Value::String("config".into()), yaml_config);
    } else {
        anyhow::bail!("Plugin job entry is not a YAML mapping");
    }

    let output = serde_yaml::to_string(&doc).context("Failed to serialize updated config")?;
    std::fs::write(config_path, output)
        .with_context(|| format!("Failed to write config file: {}", config_path.display()))?;

    Ok(())
}

/// Load configuration from YAML file + env var overrides.
///
/// Matches the Python `load_config()` behaviour: reads
/// `~/.config/colibri/config.yaml`, falls back to defaults,
/// and applies environment variable overrides.
fn load_config_inner(bootstrap: bool) -> Result<AppConfig, ColibriError> {
    let config_path = AppConfig::config_path();

    let raw: RawConfig = if config_path.exists() {
        let text = std::fs::read_to_string(&config_path).map_err(|e| {
            ColibriError::Config(format!("Failed to read {}: {e}", config_path.display()))
        })?;
        serde_yaml::from_str(&text).map_err(|e| {
            ColibriError::Config(format!("Failed to parse {}: {e}", config_path.display()))
        })?
    } else {
        // No config file — use defaults
        RawConfig {
            data: DataConfig::default(),
            index: IndexConfig::default(),
            ollama: OllamaConfig::default(),
            retrieval: RetrievalConfig::default(),
            chunking: ChunkingConfig::default(),
            embeddings: EmbeddingsConfig::default(),
            routing: RoutingConfig::default(),
            plugins: PluginsConfig::default(),
        }
    };

    let plugin_jobs = resolve_plugin_jobs(&raw, &config_path)?;

    // Resolve app root: COLIBRI_HOME > COLIBRI_DATA_DIR > config data.directory > XDG default
    let colibri_home = resolve_colibri_home(raw.data.directory.as_deref());

    // Read active generation pointer from manifest if available.
    let active_generation = active_generation_from_manifest(&colibri_home);

    // Resolve embedding profiles and routing policy.
    let (mut embedding_profiles, default_embedding_profile) = resolve_embedding_profiles(&raw)?;
    let routing_policy =
        resolve_routing_policy(&raw, &default_embedding_profile, &embedding_profiles)?;

    let canonical_dir = colibri_home.join("canonical");
    let indexes_dir = colibri_home.join("indexes");
    let state_dir = colibri_home.join("state");
    let backups_dir = colibri_home.join("backups");
    let logs_dir = colibri_home.join("logs");
    let metadata_db_path = colibri_home.join("metadata.db");
    let index_dir_name = raw.index.directory.clone();

    // Env var overrides for default embedding runtime settings.
    let env_ollama_base_url = env::var("OLLAMA_BASE_URL").ok();
    let env_embedding_model = env::var("COLIBRI_EMBEDDING_MODEL").ok();
    if let Some(profile) = embedding_profiles.get_mut(&default_embedding_profile) {
        if let Some(url) = env_ollama_base_url.as_deref() {
            profile.endpoint = url.to_string();
        }
        if let Some(model) = env_embedding_model.as_deref() {
            profile.model = model.to_string();
        }
    }

    let default_profile = embedding_profiles
        .get(&default_embedding_profile)
        .ok_or_else(|| {
            ColibriError::Config(format!(
                "Default embedding profile '{}' missing after resolution",
                default_embedding_profile
            ))
        })?;
    let lancedb_dir = indexes_dir
        .join(&active_generation)
        .join(&default_embedding_profile)
        .join(&index_dir_name);

    // Legacy fields retained for compatibility in current query/index APIs.
    let ollama_base_url = default_profile.endpoint.clone();
    let embedding_model = default_profile.model.clone();

    let config = AppConfig {
        plugin_jobs,
        colibri_home,
        canonical_dir,
        indexes_dir,
        state_dir,
        backups_dir,
        logs_dir,
        metadata_db_path,
        active_generation,
        index_dir_name,
        embedding_profiles,
        routing_policy,
        default_embedding_profile,
        lancedb_dir,
        ollama_base_url,
        embedding_model,
        top_k: raw.retrieval.top_k,
        similarity_threshold: raw.retrieval.similarity_threshold,
        chunk_size: raw.chunking.chunk_size,
        chunk_overlap: raw.chunking.chunk_overlap,
    };
    if bootstrap {
        config.ensure_directories()?;
    }
    Ok(config)
}

/// Load configuration and ensure storage bootstrap is applied.
pub fn load_config() -> Result<AppConfig, ColibriError> {
    load_config_inner(true)
}

/// Load configuration without applying storage bootstrap.
pub fn load_config_no_bootstrap() -> Result<AppConfig, ColibriError> {
    load_config_inner(false)
}

#[cfg(test)]
mod tests {
    use super::{
        enforce_local_only_routing, load_config, AppConfig, EmbeddingLocality, EmbeddingProfile,
    };
    use std::path::{Path, PathBuf};
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    struct EnvSnapshot {
        home: Option<String>,
        colibri_home: Option<String>,
        xdg_data_home: Option<String>,
        colibri_data_dir: Option<String>,
        colibri_config_path: Option<String>,
        colibri_config: Option<String>,
    }

    impl EnvSnapshot {
        fn capture() -> Self {
            Self {
                home: std::env::var("HOME").ok(),
                colibri_home: std::env::var("COLIBRI_HOME").ok(),
                xdg_data_home: std::env::var("XDG_DATA_HOME").ok(),
                colibri_data_dir: std::env::var("COLIBRI_DATA_DIR").ok(),
                colibri_config_path: std::env::var("COLIBRI_CONFIG_PATH").ok(),
                colibri_config: std::env::var("COLIBRI_CONFIG").ok(),
            }
        }

        fn restore(self) {
            set_env_opt("HOME", self.home.as_deref());
            set_env_opt("COLIBRI_HOME", self.colibri_home.as_deref());
            set_env_opt("XDG_DATA_HOME", self.xdg_data_home.as_deref());
            set_env_opt("COLIBRI_DATA_DIR", self.colibri_data_dir.as_deref());
            set_env_opt("COLIBRI_CONFIG_PATH", self.colibri_config_path.as_deref());
            set_env_opt("COLIBRI_CONFIG", self.colibri_config.as_deref());
        }
    }

    fn set_env_opt(key: &str, val: Option<&str>) {
        match val {
            Some(v) => std::env::set_var(key, v),
            None => std::env::remove_var(key),
        }
    }

    fn copy_dir_all(src: &Path, dst: &Path) -> std::io::Result<()> {
        std::fs::create_dir_all(dst)?;
        for entry in std::fs::read_dir(src)? {
            let entry = entry?;
            let ty = entry.file_type()?;
            let src_path = entry.path();
            let dst_path = dst.join(entry.file_name());
            if ty.is_dir() {
                copy_dir_all(&src_path, &dst_path)?;
            } else if ty.is_file() {
                std::fs::copy(&src_path, &dst_path)?;
            }
        }
        Ok(())
    }

    fn unique_tmp_dir(prefix: &str) -> PathBuf {
        let n = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!("{prefix}-{n}"))
    }

    #[test]
    fn restricted_must_not_route_to_cloud_profile() {
        let mut profiles = std::collections::HashMap::new();
        profiles.insert(
            "local_secure".to_string(),
            EmbeddingProfile {
                id: "local_secure".to_string(),
                provider: "tei".to_string(),
                endpoint: "http://localhost:8080".to_string(),
                model: "bge-m3".to_string(),
                locality: EmbeddingLocality::Local,
            },
        );
        profiles.insert(
            "cloud_fast".to_string(),
            EmbeddingProfile {
                id: "cloud_fast".to_string(),
                provider: "openai_compatible".to_string(),
                endpoint: "https://api.example.com".to_string(),
                model: "text-embedding-3-large".to_string(),
                locality: EmbeddingLocality::Cloud,
            },
        );

        let mut policy = std::collections::HashMap::new();
        policy.insert("restricted".to_string(), "cloud_fast".to_string());
        policy.insert("confidential".to_string(), "local_secure".to_string());
        let err = enforce_local_only_routing(&policy, &profiles)
            .unwrap_err()
            .to_string();
        assert!(err.contains("restricted"));
        assert!(err.contains("local"));
    }

    #[test]
    fn portable_colibri_home_copy_preserves_active_generation() {
        let _guard = ENV_LOCK.lock().unwrap();
        let snap = EnvSnapshot::capture();

        let root = unique_tmp_dir("colibri-portable");
        let home = root.join("home");
        let colibri_a = root.join("colibri_a");
        let colibri_b = root.join("colibri_b");

        let cfg_dir = home.join(".config").join("colibri");
        std::fs::create_dir_all(&cfg_dir).expect("create config dir");
        std::fs::write(cfg_dir.join("config.yaml"), "{}\n").expect("write config");

        set_env_opt("HOME", Some(home.to_string_lossy().as_ref()));
        set_env_opt("XDG_DATA_HOME", None);
        set_env_opt("COLIBRI_DATA_DIR", None);

        set_env_opt("COLIBRI_HOME", Some(colibri_a.to_string_lossy().as_ref()));
        let gen = "gen_portable_test";
        std::fs::create_dir_all(&colibri_a).expect("create colibri home");
        super::ensure_root_manifest(&colibri_a, gen).expect("write root manifest");

        copy_dir_all(&colibri_a, &colibri_b).expect("copy colibri_home");

        set_env_opt("COLIBRI_HOME", Some(colibri_b.to_string_lossy().as_ref()));
        let cfg2 = load_config().expect("load config from copied home");
        assert_eq!(cfg2.active_generation, gen);

        snap.restore();
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn config_path_is_under_home() {
        let _guard = ENV_LOCK.lock().unwrap();
        let snap = EnvSnapshot::capture();

        let root = unique_tmp_dir("colibri-home");
        let home = root.join("home");
        std::fs::create_dir_all(&home).expect("create home");

        set_env_opt("COLIBRI_CONFIG_PATH", None);
        set_env_opt("COLIBRI_CONFIG", None);
        set_env_opt("HOME", Some(home.to_string_lossy().as_ref()));
        set_env_opt("XDG_DATA_HOME", None);

        let p = AppConfig::config_path();
        assert!(p.starts_with(&home));

        snap.restore();
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn config_path_respects_env_override() {
        let _guard = ENV_LOCK.lock().unwrap();
        let snap = EnvSnapshot::capture();

        let root = unique_tmp_dir("colibri-config");
        let cfg = root.join("config.yaml");
        std::fs::create_dir_all(&root).expect("create tmp root");
        std::fs::write(&cfg, "{}\n").expect("write config");

        set_env_opt("COLIBRI_CONFIG_PATH", Some(cfg.to_string_lossy().as_ref()));
        set_env_opt("COLIBRI_CONFIG", None);
        let p = AppConfig::config_path();
        assert_eq!(p, cfg);

        set_env_opt("COLIBRI_CONFIG_PATH", None);
        set_env_opt("COLIBRI_CONFIG", Some(cfg.to_string_lossy().as_ref()));
        let p2 = AppConfig::config_path();
        assert_eq!(p2, cfg);

        snap.restore();
        let _ = std::fs::remove_dir_all(root);
    }
}

#[cfg(test)]
mod config_update_tests {
    use super::*;

    #[test]
    fn update_plugin_job_config_replaces_config_block() {
        let yaml_content = r#"
sources: []
plugins:
  jobs:
    - id: myjob
      manifest: /tmp/manifest.json
      enabled: true
      config:
        old_key: old_value
"#;
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), yaml_content).unwrap();

        let new_config = serde_json::json!({"new_key": "new_value"});
        update_plugin_job_config(tmp.path(), "myjob", &new_config).unwrap();

        let updated = std::fs::read_to_string(tmp.path()).unwrap();
        let doc: serde_yaml::Value = serde_yaml::from_str(&updated).unwrap();
        let jobs = doc["plugins"]["jobs"].as_sequence().unwrap();
        let job = &jobs[0];
        assert_eq!(job["config"]["new_key"].as_str().unwrap(), "new_value");
        assert!(job["config"].get("old_key").is_none());
    }

    #[test]
    fn update_plugin_job_config_errors_for_unknown_job() {
        let yaml_content = r#"
plugins:
  jobs:
    - id: myjob
      manifest: /tmp/manifest.json
      config: {}
"#;
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), yaml_content).unwrap();

        let new_config = serde_json::json!({"key": "value"});
        let err = update_plugin_job_config(tmp.path(), "nonexistent", &new_config);
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("nonexistent"));
    }
}
