//! Configuration loading from `~/.config/colibri/config.yaml`.
//!
//! Loading the config has no side effects. Commands that change data call
//! [`AppConfig::open_for_write`], which creates the data layout, takes the
//! write lock and opens the metadata DB read-write.

use std::env;
use std::path::PathBuf;

use serde::Deserialize;

use crate::error::ColibriError;
use crate::lock::WriteLock;
use crate::metadata_store::MetadataStore;

/// LanceDB index layout version. A mismatch triggers a full index rebuild.
pub const SCHEMA_VERSION: u32 = 7;

/// Raw YAML config structure. Unknown top-level keys are ignored.
#[derive(Debug, Default, Deserialize)]
struct RawConfig {
    #[serde(default)]
    data: DataConfig,

    #[serde(default)]
    embedding: EmbeddingConfig,

    /// Legacy name of the embedding section, used when `embedding:` is absent.
    #[serde(default)]
    ollama: OllamaConfig,

    #[serde(default)]
    retrieval: RetrievalConfig,

    #[serde(default)]
    chunking: ChunkingConfig,

    #[serde(default)]
    connectors: Vec<crate::connectors::ConnectorRawConfig>,
}

#[derive(Debug, Default, Deserialize)]
struct DataConfig {
    directory: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct EmbeddingConfig {
    endpoint: Option<String>,
    model: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct OllamaConfig {
    base_url: Option<String>,
    embedding_model: Option<String>,
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
    25
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

/// Resolved application configuration.
#[derive(Debug, Clone)]
pub struct AppConfig {
    pub connector_jobs: Vec<crate::connectors::ConnectorJob>,
    pub colibri_home: PathBuf,
    pub canonical_dir: PathBuf,
    /// LanceDB directory holding the single `chunks` table.
    pub index_dir: PathBuf,
    pub metadata_db_path: PathBuf,
    pub lock_path: PathBuf,
    pub embedding_endpoint: String,
    pub embedding_model: String,
    pub top_k: usize,
    pub similarity_threshold: f64,
    pub chunk_size: usize,
    pub chunk_overlap: usize,
}

impl AppConfig {
    /// Config file path.
    pub fn config_path() -> PathBuf {
        for var in ["COLIBRI_CONFIG_PATH", "COLIBRI_CONFIG"] {
            if let Ok(p) = env::var(var) {
                let trimmed = p.trim();
                if !trimmed.is_empty() {
                    return PathBuf::from(trimmed);
                }
            }
        }
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".config")
            .join("colibri")
            .join("config.yaml")
    }

    /// Create the data layout, take the write lock and open the metadata DB
    /// read-write. Keep the returned lock alive for the whole command.
    pub fn open_for_write(&self) -> Result<(WriteLock, MetadataStore), ColibriError> {
        std::fs::create_dir_all(&self.colibri_home)?;
        let lock = WriteLock::acquire(&self.lock_path)?;
        std::fs::create_dir_all(&self.canonical_dir)?;
        std::fs::create_dir_all(&self.index_dir)?;
        let store = MetadataStore::open_rw(&self.metadata_db_path)?;
        Ok((lock, store))
    }

    /// Open the metadata DB read-only. Creates nothing.
    pub fn open_read(&self) -> Result<MetadataStore, ColibriError> {
        MetadataStore::open_ro(&self.metadata_db_path)
    }
}

#[cfg(test)]
impl AppConfig {
    /// A config rooted at `home` with a fake embedding endpoint, for tests.
    pub(crate) fn for_test(home: &std::path::Path) -> Self {
        Self {
            connector_jobs: Vec::new(),
            colibri_home: home.to_path_buf(),
            canonical_dir: home.join("canonical"),
            index_dir: home.join("index").join("lancedb"),
            metadata_db_path: home.join("metadata.db"),
            lock_path: home.join("write.lock"),
            embedding_endpoint: "http://127.0.0.1:9".into(),
            embedding_model: "fake".into(),
            top_k: 10,
            similarity_threshold: 0.0,
            chunk_size: 200,
            chunk_overlap: 20,
        }
    }
}

/// Load configuration from the YAML file plus environment overrides.
/// Never touches the data directory.
pub fn load_config() -> Result<AppConfig, ColibriError> {
    let config_path = AppConfig::config_path();

    let raw: RawConfig = if config_path.exists() {
        let text = std::fs::read_to_string(&config_path).map_err(|e| {
            ColibriError::Config(format!("Failed to read {}: {e}", config_path.display()))
        })?;
        serde_yaml::from_str(&text).map_err(|e| {
            ColibriError::Config(format!("Failed to parse {}: {e}", config_path.display()))
        })?
    } else {
        RawConfig::default()
    };

    let connector_jobs: Vec<crate::connectors::ConnectorJob> = raw
        .connectors
        .iter()
        .enumerate()
        .map(|(idx, c)| {
            let id = if c.id.trim().is_empty() {
                format!("connector_{}", idx + 1)
            } else {
                c.id.trim().to_string()
            };
            crate::connectors::ConnectorJob {
                id,
                connector_type: c.connector_type.clone(),
                enabled: c.enabled,
                config: c.config.clone(),
            }
        })
        .collect();

    // Resolve app root: COLIBRI_HOME > COLIBRI_DATA_DIR > config data.directory > XDG default
    let colibri_home = resolve_colibri_home(raw.data.directory.as_deref());

    // Embedding runtime: env var > `embedding:` > legacy `ollama:` > default.
    let embedding_endpoint = env::var("OLLAMA_BASE_URL")
        .ok()
        .or(raw.embedding.endpoint)
        .or(raw.ollama.base_url)
        .unwrap_or_else(default_ollama_url);
    let embedding_model = env::var("COLIBRI_EMBEDDING_MODEL")
        .ok()
        .or(raw.embedding.model)
        .or(raw.ollama.embedding_model)
        .unwrap_or_else(default_embedding_model);

    Ok(AppConfig {
        connector_jobs,
        canonical_dir: colibri_home.join("canonical"),
        index_dir: colibri_home.join("index").join("lancedb"),
        metadata_db_path: colibri_home.join("metadata.db"),
        lock_path: colibri_home.join("write.lock"),
        colibri_home,
        embedding_endpoint,
        embedding_model,
        top_k: raw.retrieval.top_k,
        similarity_threshold: raw.retrieval.similarity_threshold,
        chunk_size: raw.chunking.chunk_size,
        chunk_overlap: raw.chunking.chunk_overlap,
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::{load_config, AppConfig};
    use std::path::PathBuf;
    use std::sync::Mutex;

    /// Serializes tests that change process-wide env vars.
    pub(crate) static ENV_LOCK: Mutex<()> = Mutex::new(());

    const VARS: [&str; 8] = [
        "HOME",
        "COLIBRI_HOME",
        "XDG_DATA_HOME",
        "COLIBRI_DATA_DIR",
        "COLIBRI_CONFIG_PATH",
        "COLIBRI_CONFIG",
        "OLLAMA_BASE_URL",
        "COLIBRI_EMBEDDING_MODEL",
    ];

    pub(crate) struct EnvSnapshot(Vec<(&'static str, Option<String>)>);

    impl EnvSnapshot {
        pub(crate) fn capture() -> Self {
            Self(VARS.iter().map(|k| (*k, std::env::var(k).ok())).collect())
        }

        pub(crate) fn restore(self) {
            for (k, v) in self.0 {
                set_env_opt(k, v.as_deref());
            }
        }
    }

    pub(crate) fn set_env_opt(key: &str, val: Option<&str>) {
        match val {
            Some(v) => std::env::set_var(key, v),
            None => std::env::remove_var(key),
        }
    }

    /// Point config + COLIBRI_HOME at a fresh temp dir with the given config text.
    /// Returns (root, colibri_home).
    pub(crate) fn isolated_home(config_yaml: &str) -> (tempfile::TempDir, PathBuf) {
        let root = tempfile::TempDir::new().expect("tmp root");
        let cfg = root.path().join("config.yaml");
        let colibri_home = root.path().join("colibri");
        std::fs::write(&cfg, config_yaml).expect("write config");
        set_env_opt("COLIBRI_CONFIG_PATH", Some(cfg.to_string_lossy().as_ref()));
        set_env_opt("COLIBRI_CONFIG", None);
        set_env_opt(
            "COLIBRI_HOME",
            Some(colibri_home.to_string_lossy().as_ref()),
        );
        set_env_opt("OLLAMA_BASE_URL", None);
        set_env_opt("COLIBRI_EMBEDDING_MODEL", None);
        (root, colibri_home)
    }

    #[test]
    fn config_path_respects_env_override() {
        let _guard = ENV_LOCK.lock().unwrap();
        let snap = EnvSnapshot::capture();
        let root = tempfile::TempDir::new().unwrap();
        let cfg = root.path().join("config.yaml");

        set_env_opt("COLIBRI_CONFIG_PATH", Some(cfg.to_string_lossy().as_ref()));
        set_env_opt("COLIBRI_CONFIG", None);
        assert_eq!(AppConfig::config_path(), cfg);

        set_env_opt("COLIBRI_CONFIG_PATH", None);
        set_env_opt("COLIBRI_CONFIG", Some(cfg.to_string_lossy().as_ref()));
        assert_eq!(AppConfig::config_path(), cfg);

        set_env_opt("COLIBRI_CONFIG", None);
        set_env_opt("HOME", Some(root.path().to_string_lossy().as_ref()));
        assert!(AppConfig::config_path().starts_with(root.path()));

        snap.restore();
    }

    #[test]
    fn embedding_section_overrides_legacy_ollama_and_env_overrides_both() {
        let _guard = ENV_LOCK.lock().unwrap();
        let snap = EnvSnapshot::capture();
        let (_root, _home) = isolated_home(
            "ollama: {base_url: 'http://old:1', embedding_model: old}\n\
             embedding: {endpoint: 'http://new:2', model: new}\n",
        );
        let cfg = load_config().unwrap();
        assert_eq!(cfg.embedding_endpoint, "http://new:2");
        assert_eq!(cfg.embedding_model, "new");

        set_env_opt("COLIBRI_EMBEDDING_MODEL", Some("from-env"));
        assert_eq!(load_config().unwrap().embedding_model, "from-env");

        let (_root2, _home2) = isolated_home("ollama: {embedding_model: legacy}\n");
        assert_eq!(load_config().unwrap().embedding_model, "legacy");
        snap.restore();
    }

    #[test]
    fn layout_paths_are_under_colibri_home() {
        let _guard = ENV_LOCK.lock().unwrap();
        let snap = EnvSnapshot::capture();
        let (_root, home) = isolated_home("{}\n");
        let cfg = load_config().unwrap();
        assert_eq!(cfg.metadata_db_path, home.join("metadata.db"));
        assert_eq!(cfg.index_dir, home.join("index").join("lancedb"));
        assert_eq!(cfg.canonical_dir, home.join("canonical"));
        assert_eq!(cfg.lock_path, home.join("write.lock"));
        snap.restore();
    }

    // AC-002.2 (config part): loading config never touches the data dir.
    #[test]
    fn load_config_creates_nothing() {
        let _guard = ENV_LOCK.lock().unwrap();
        let snap = EnvSnapshot::capture();
        let (_root, home) = isolated_home("{}\n");
        let cfg = load_config().unwrap();
        assert!(!home.exists());
        assert!(cfg.open_read().is_err());
        assert!(!home.exists());
        snap.restore();
    }

    // AC-001.1 at the config level: a corrupt DB stays byte-identical.
    #[test]
    fn open_for_write_leaves_corrupt_metadata_db_untouched() {
        let _guard = ENV_LOCK.lock().unwrap();
        let snap = EnvSnapshot::capture();
        let (_root, home) = isolated_home("{}\n");
        std::fs::create_dir_all(&home).unwrap();
        let db = home.join("metadata.db");
        let original = b"definitely not a sqlite database\n".repeat(100);
        std::fs::write(&db, &original).unwrap();

        let err = load_config().unwrap().open_for_write().err().unwrap();
        assert!(err.to_string().contains("colibri reset"), "{err}");
        assert_eq!(std::fs::read(&db).unwrap(), original);
        assert!(!home.join("metadata.legacy-json.bak").exists());
        snap.restore();
    }
}
