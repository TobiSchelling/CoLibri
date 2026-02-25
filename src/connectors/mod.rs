//! Native connector framework.

pub mod filesystem;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::Value;

use crate::envelope::DocumentEnvelope;
use crate::error::ColibriError;

/// A native content connector.
#[async_trait]
pub trait Connector: Send + Sync {
    /// Unique identifier for this connector instance.
    fn id(&self) -> &str;

    /// Fetch all documents from this source.
    async fn sync(&self) -> Result<Vec<DocumentEnvelope>, ColibriError>;
}

/// Raw YAML config for a connector entry.
#[derive(Debug, Deserialize)]
pub struct ConnectorRawConfig {
    #[serde(rename = "type")]
    pub connector_type: String,
    pub id: String,
    #[serde(default = "default_enabled")]
    pub enabled: bool,
    /// All remaining fields passed to the connector as config.
    #[serde(flatten)]
    pub config: Value,
}

fn default_enabled() -> bool {
    true
}

/// Resolved connector ready to sync.
#[derive(Debug, Clone)]
pub struct ConnectorJob {
    pub id: String,
    pub connector_type: String,
    pub enabled: bool,
    pub config: Value,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialize_filesystem_connector_config() {
        let yaml = r#"
type: filesystem
id: books
enabled: true
root_path: /tmp/books
include_extensions:
  - .md
  - .pdf
exclude_globs:
  - "**/.git/**"
mode: incremental
doc_type: book
classification: internal
"#;
        let raw: ConnectorRawConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(raw.connector_type, "filesystem");
        assert_eq!(raw.id, "books");
        assert!(raw.enabled);
    }

    #[test]
    fn deserialize_disabled_connector() {
        let yaml = r#"
type: filesystem
id: test
enabled: false
root_path: /tmp/test
"#;
        let raw: ConnectorRawConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(!raw.enabled);
    }
}
