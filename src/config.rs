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

    #[serde(default)]
    mirrors: Vec<MirrorRaw>,

    #[serde(default)]
    prune: PruneRaw,
}

/// A folder CoLibri keeps in sync: new and changed files are ingested,
/// deleted files are pruned (guarded).
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MirrorRaw {
    name: String,
    path: String,
    #[serde(default = "default_mirror_doc_type")]
    doc_type: String,
    #[serde(default = "default_mirror_include")]
    include: Vec<String>,
    #[serde(default)]
    exclude: Vec<String>,
    #[serde(default = "default_true")]
    plantuml_summaries: bool,
}

fn default_mirror_doc_type() -> String {
    "note".into()
}

fn default_mirror_include() -> Vec<String> {
    vec!["**/*.md".into()]
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PruneRaw {
    #[serde(default = "default_prune_max_fraction")]
    max_fraction: f64,
    #[serde(default = "default_prune_min_count")]
    min_count: usize,
}

impl Default for PruneRaw {
    fn default() -> Self {
        Self {
            max_fraction: default_prune_max_fraction(),
            min_count: default_prune_min_count(),
        }
    }
}

fn default_prune_max_fraction() -> f64 {
    0.2
}

fn default_prune_min_count() -> usize {
    25
}

/// A resolved mirror definition.
#[derive(Debug, Clone)]
pub struct MirrorConfig {
    /// Collection name; also the prefix of every document id.
    pub name: String,
    pub path: PathBuf,
    pub doc_type: String,
    pub include: Vec<String>,
    pub exclude: Vec<String>,
    pub plantuml_summaries: bool,
}

/// Limits for deletions in one update: a mirror prunes at most
/// `max(min_count, max_fraction × active documents)` unless overridden.
#[derive(Debug, Clone, Copy)]
pub struct PruneConfig {
    pub max_fraction: f64,
    pub min_count: usize,
}

impl PruneConfig {
    pub fn limit(&self, active: usize) -> usize {
        self.min_count
            .max((self.max_fraction * active as f64).floor() as usize)
    }
}

/// Expand a leading `~` to the home directory.
pub fn expand_tilde(path: &str) -> PathBuf {
    if path == "~" {
        if let Some(home) = dirs::home_dir() {
            return home;
        }
    } else if let Some(rest) = path.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.join(rest);
        }
    }
    PathBuf::from(path)
}

fn valid_collection_name(name: &str) -> bool {
    !name.is_empty()
        && !name.starts_with('.')
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.')
}

fn resolve_mirrors(
    raw: Vec<MirrorRaw>,
    connector_ids: &[String],
) -> Result<Vec<MirrorConfig>, ColibriError> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for m in raw {
        let name = m.name.trim().to_string();
        if !valid_collection_name(&name) {
            return Err(ColibriError::Config(format!(
                "mirrors: invalid name '{name}' (use letters, digits, '-', '_', '.')"
            )));
        }
        if !seen.insert(name.clone()) || connector_ids.contains(&name) {
            return Err(ColibriError::Config(format!(
                "mirrors: name '{name}' is used twice (mirror and connector names must be unique)"
            )));
        }
        for pattern in m.include.iter().chain(m.exclude.iter()) {
            glob::Pattern::new(pattern).map_err(|e| {
                ColibriError::Config(format!(
                    "mirrors.{name}: invalid glob pattern '{pattern}': {e}"
                ))
            })?;
        }
        let path = expand_tilde(m.path.trim());
        if !path.is_absolute() {
            return Err(ColibriError::Config(format!(
                "mirrors.{name}: path must be absolute or start with ~ (got '{}')",
                m.path
            )));
        }
        out.push(MirrorConfig {
            path,
            doc_type: m.doc_type,
            include: m.include,
            exclude: m.exclude,
            plantuml_summaries: m.plantuml_summaries,
            name,
        });
    }
    Ok(out)
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
    pub mirrors: Vec<MirrorConfig>,
    pub prune: PruneConfig,
    pub colibri_home: PathBuf,
    pub canonical_dir: PathBuf,
    /// LanceDB directory holding the single `chunks` table.
    pub index_dir: PathBuf,
    /// Cached markdown of converted sources, keyed by source SHA-256.
    pub conversions_dir: PathBuf,
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
            mirrors: Vec::new(),
            prune: PruneConfig {
                max_fraction: default_prune_max_fraction(),
                min_count: default_prune_min_count(),
            },
            colibri_home: home.to_path_buf(),
            canonical_dir: home.join("canonical"),
            index_dir: home.join("index").join("lancedb"),
            conversions_dir: home.join("conversions"),
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

    let connector_ids: Vec<String> = connector_jobs.iter().map(|j| j.id.clone()).collect();
    let mirrors = resolve_mirrors(raw.mirrors, &connector_ids)?;
    let prune = PruneConfig {
        max_fraction: raw.prune.max_fraction,
        min_count: raw.prune.min_count,
    };

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
        mirrors,
        prune,
        canonical_dir: colibri_home.join("canonical"),
        index_dir: colibri_home.join("index").join("lancedb"),
        conversions_dir: colibri_home.join("conversions"),
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

    #[test]
    fn mirrors_are_resolved_and_validated() {
        let _guard = ENV_LOCK.lock().unwrap();
        let snap = EnvSnapshot::capture();

        let (_r, _h) = isolated_home(
            "mirrors:\n  - {name: vault, path: ~/PKM, exclude: ['.obsidian/**']}\nprune: {min_count: 5}\n",
        );
        let cfg = load_config().unwrap();
        assert_eq!(cfg.mirrors.len(), 1);
        let m = &cfg.mirrors[0];
        assert_eq!(m.include, ["**/*.md"]);
        assert_eq!(m.doc_type, "note");
        assert!(!m.path.to_string_lossy().starts_with('~'));
        assert_eq!(cfg.prune.min_count, 5);
        assert_eq!(cfg.prune.limit(100), 20);

        for (yaml, needle) in [
            ("mirrors:\n  - {name: v, path: /x, exlude: []}\n", "exlude"),
            (
                "mirrors:\n  - {name: v, path: /x}\n  - {name: v, path: /y}\n",
                "used twice",
            ),
            ("mirrors:\n  - {name: '../x', path: /x}\n", "invalid name"),
            (
                "mirrors:\n  - {name: v, path: /x, include: ['[']}\n",
                "invalid glob",
            ),
            ("mirrors:\n  - {name: v, path: docs}\n", "must be absolute"),
        ] {
            let (_r, _h) = isolated_home(yaml);
            let err = load_config().unwrap_err().to_string();
            assert!(err.contains(needle), "{yaml}: {err}");
        }
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
