//! Configuration loading from `~/.config/colibri/config.yaml`.
//!
//! Mirrors the Python `config.py` module: loads YAML config with defaults,
//! supports env var overrides, and produces the same derived paths.

use std::env;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::error::ColibriError;

/// Schema version — must match Python's `SCHEMA_VERSION` for cross-compat.
pub const SCHEMA_VERSION: u32 = 5;

/// How a folder should be indexed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum IndexMode {
    /// Content assumed stable — skip known files, but detect deletions.
    Static,
    /// Track changes via mtime + SHA-256 — re-index modified files, detect deletions.
    #[default]
    Incremental,
}

/// Per-source indexing configuration (mirrors Python `FolderProfile`).
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct FolderProfile {
    /// Absolute path to the source directory (required).
    pub path: String,

    /// Indexing mode.
    #[serde(default)]
    pub mode: IndexMode,

    /// Document type label (e.g., "book", "note").
    #[serde(default = "default_doc_type")]
    pub doc_type: String,

    /// Per-folder chunk size override.
    pub chunk_size: Option<usize>,

    /// Per-folder chunk overlap override.
    pub chunk_overlap: Option<usize>,

    /// File extensions to index.
    #[serde(default = "default_extensions")]
    pub extensions: Vec<String>,

    /// Display name (defaults to path basename).
    pub name: Option<String>,
}

fn default_doc_type() -> String {
    "note".into()
}

fn default_extensions() -> Vec<String> {
    vec![".md".into()]
}

impl Default for FolderProfile {
    fn default() -> Self {
        Self {
            path: String::new(),
            mode: IndexMode::default(),
            doc_type: default_doc_type(),
            chunk_size: None,
            chunk_overlap: None,
            extensions: default_extensions(),
            name: None,
        }
    }
}

impl FolderProfile {
    /// Human-readable display name.
    pub fn display_name(&self) -> &str {
        self.name.as_deref().unwrap_or_else(|| {
            std::path::Path::new(&self.path)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or(&self.path)
        })
    }

    /// Effective chunk size (per-folder override or global default).
    pub fn effective_chunk_size(&self, default: usize) -> usize {
        self.chunk_size.unwrap_or(default)
    }

    /// Effective chunk overlap (per-folder override or global default).
    pub fn effective_chunk_overlap(&self, default: usize) -> usize {
        self.chunk_overlap.unwrap_or(default)
    }
}

/// Raw YAML config structure.
#[derive(Debug, Deserialize)]
struct RawConfig {
    #[serde(default)]
    sources: Vec<FolderProfile>,

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
}

#[derive(Debug, Default, Deserialize)]
struct DataConfig {
    directory: Option<String>,
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

/// Resolved application configuration.
#[derive(Debug, Clone)]
pub struct AppConfig {
    pub sources: Vec<FolderProfile>,
    pub data_dir: PathBuf,
    pub lancedb_dir: PathBuf,
    pub ollama_base_url: String,
    pub embedding_model: String,
    pub top_k: usize,
    pub similarity_threshold: f64,
    pub chunk_size: usize,
    pub chunk_overlap: usize,
}

impl AppConfig {
    /// Return the first source with `doc_type == "book"`, if any.
    #[allow(dead_code)]
    pub fn books_source(&self) -> Option<&FolderProfile> {
        self.sources.iter().find(|s| s.doc_type == "book")
    }

    /// Config file path.
    pub fn config_path() -> PathBuf {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".config")
            .join("colibri")
            .join("config.yaml")
    }

    /// Ensure data and LanceDB directories exist.
    pub fn ensure_directories(&self) -> std::io::Result<()> {
        std::fs::create_dir_all(&self.data_dir)?;
        std::fs::create_dir_all(&self.lancedb_dir)?;
        Ok(())
    }
}

/// Load configuration from YAML file + env var overrides.
///
/// Matches the Python `load_config()` behaviour: reads
/// `~/.config/colibri/config.yaml`, falls back to defaults,
/// and applies environment variable overrides.
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
        // No config file — use defaults
        RawConfig {
            sources: vec![FolderProfile {
                path: dirs::home_dir()
                    .unwrap_or_else(|| PathBuf::from("."))
                    .join("Documents")
                    .join("CoLibri")
                    .join("Books")
                    .to_string_lossy()
                    .into_owned(),
                mode: IndexMode::Static,
                doc_type: "book".into(),
                chunk_size: None,
                chunk_overlap: None,
                extensions: default_extensions(),
                name: None,
            }],
            data: DataConfig::default(),
            index: IndexConfig::default(),
            ollama: OllamaConfig::default(),
            retrieval: RetrievalConfig::default(),
            chunking: ChunkingConfig::default(),
        }
    };

    // Resolve data directory: env > config > XDG default
    let data_dir = if let Ok(val) = env::var("COLIBRI_DATA_DIR") {
        PathBuf::from(val)
    } else if let Some(ref dir) = raw.data.directory {
        PathBuf::from(dir)
    } else {
        let xdg = env::var("XDG_DATA_HOME").unwrap_or_else(|_| {
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".local")
                .join("share")
                .to_string_lossy()
                .into_owned()
        });
        PathBuf::from(xdg).join("colibri")
    };

    let lancedb_dir = data_dir.join(&raw.index.directory);

    // Env var overrides for Ollama settings
    let ollama_base_url =
        env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| raw.ollama.base_url.clone());
    let embedding_model =
        env::var("COLIBRI_EMBEDDING_MODEL").unwrap_or_else(|_| raw.ollama.embedding_model.clone());

    Ok(AppConfig {
        sources: raw.sources,
        data_dir,
        lancedb_dir,
        ollama_base_url,
        embedding_model,
        top_k: raw.retrieval.top_k,
        similarity_threshold: raw.retrieval.similarity_threshold,
        chunk_size: raw.chunking.chunk_size,
        chunk_overlap: raw.chunking.chunk_overlap,
    })
}

/// Structure for saving config back to YAML.
/// Only includes sources since other settings may be from env vars.
#[derive(Debug, Serialize)]
struct SaveConfig {
    sources: Vec<FolderProfile>,
}

/// Save configuration sources to the config file.
///
/// Only writes the `sources` section to preserve user's other settings.
/// If the config file exists, it merges sources into the existing file.
pub fn save_config(config: &AppConfig) -> Result<(), ColibriError> {
    let config_path = AppConfig::config_path();

    // Ensure config directory exists
    if let Some(parent) = config_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Try to preserve existing config structure
    let yaml_content = if config_path.exists() {
        let existing_text = std::fs::read_to_string(&config_path).map_err(|e| {
            ColibriError::Config(format!("Failed to read {}: {e}", config_path.display()))
        })?;

        // Parse existing YAML as a generic value to preserve other sections
        let mut existing: serde_yaml::Value = serde_yaml::from_str(&existing_text)
            .unwrap_or(serde_yaml::Value::Mapping(serde_yaml::Mapping::new()));

        // Update sources section
        let sources_value = serde_yaml::to_value(&config.sources)
            .map_err(|e| ColibriError::Config(format!("Failed to serialize sources: {e}")))?;

        if let serde_yaml::Value::Mapping(ref mut map) = existing {
            map.insert(serde_yaml::Value::String("sources".into()), sources_value);
        }

        serde_yaml::to_string(&existing)
            .map_err(|e| ColibriError::Config(format!("Failed to serialize config: {e}")))?
    } else {
        // Create new config with just sources
        let save_config = SaveConfig {
            sources: config.sources.clone(),
        };
        serde_yaml::to_string(&save_config)
            .map_err(|e| ColibriError::Config(format!("Failed to serialize config: {e}")))?
    };

    std::fs::write(&config_path, yaml_content).map_err(|e| {
        ColibriError::Config(format!("Failed to write {}: {e}", config_path.display()))
    })?;

    Ok(())
}
