//! Error types for the CoLibri application.

use thiserror::Error;

/// Application-level errors returned by library functions.
#[derive(Debug, Error)]
pub enum ColibriError {
    #[error("Config error: {0}")]
    Config(String),

    #[error("Embedding error: {0}")]
    Embedding(String),

    #[error("Index error: {0}")]
    Index(String),

    #[error("Query error: {0}")]
    Query(String),

    #[error("Manifest error: {0}")]
    #[allow(dead_code)]
    Manifest(String),

    #[error("Source error: {0}")]
    Source(String),

    #[error("MCP error: {0}")]
    Mcp(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("YAML error: {0}")]
    Yaml(#[from] serde_yaml::Error),

    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("LanceDB error: {0}")]
    Lance(#[from] lancedb::Error),
}
