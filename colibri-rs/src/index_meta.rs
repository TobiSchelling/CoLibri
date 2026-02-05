//! Index metadata (schema version tracking).
//!
//! Tracks the schema version of the LanceDB index so that layout changes
//! trigger an automatic rebuild. Mirrors the Python `index_meta.py`.

use std::path::Path;

use chrono::Utc;
use serde_json::Value;

use crate::config::SCHEMA_VERSION;
use crate::error::ColibriError;

/// Read index metadata from `<data_dir>/index_meta.json`.
pub fn read_index_meta(data_dir: &Path) -> Result<serde_json::Map<String, Value>, ColibriError> {
    let meta_path = data_dir.join("index_meta.json");
    if !meta_path.exists() {
        return Ok(serde_json::Map::new());
    }
    let text = std::fs::read_to_string(&meta_path)?;
    let val: Value = serde_json::from_str(&text)?;
    match val {
        Value::Object(map) => Ok(map),
        _ => Ok(serde_json::Map::new()),
    }
}

/// Write index metadata to `<data_dir>/index_meta.json`.
///
/// Preserves `created_at` from existing metadata. Merges `extra` fields.
pub fn write_index_meta(
    data_dir: &Path,
    embedding_model: &str,
    extra: &serde_json::Map<String, Value>,
) -> Result<(), ColibriError> {
    let existing = read_index_meta(data_dir)?;

    let created_at = existing
        .get("created_at")
        .and_then(|v| v.as_str())
        .unwrap_or(&Utc::now().to_rfc3339())
        .to_string();

    let mut meta = serde_json::Map::new();
    meta.insert(
        "schema_version".into(),
        Value::Number(SCHEMA_VERSION.into()),
    );
    meta.insert("created_at".into(), Value::String(created_at));
    meta.insert("updated_at".into(), Value::String(Utc::now().to_rfc3339()));
    meta.insert(
        "embedding_model".into(),
        Value::String(embedding_model.to_string()),
    );

    // Merge extra fields
    for (k, v) in extra {
        meta.insert(k.clone(), v.clone());
    }

    let meta_path = data_dir.join("index_meta.json");
    let json = serde_json::to_string_pretty(&Value::Object(meta))?;
    std::fs::write(meta_path, json)?;
    Ok(())
}
