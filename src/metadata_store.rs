//! SQLite-backed metadata store for canonical docs and sync state.
//!
//! Uses the system `sqlite3` binary to avoid adding new Rust crate dependencies.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::Command;

use chrono::Utc;
use serde_json::Value;

use crate::config::{EmbeddingLocality, EmbeddingProfile};
use crate::config::{
    CANONICAL_SCHEMA_VERSION, METADATA_DB_FORMAT_VERSION, PIPELINE_SCHEMA_VERSION,
    SERVING_SCHEMA_VERSION,
};
use crate::error::ColibriError;

const FORMAT_NAME: &str = "colibri-metadata-db";

pub fn require_sqlite3() -> Result<(), ColibriError> {
    let output = match Command::new("sqlite3").arg("-version").output() {
        Ok(o) => o,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(ColibriError::Config(
                "sqlite3 not found on PATH. Install SQLite (sqlite3 CLI) to use metadata.db."
                    .into(),
            ));
        }
        Err(e) => return Err(ColibriError::Io(e)),
    };
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(ColibriError::Config(format!(
            "sqlite3 failed to run: {}",
            if stderr.is_empty() {
                "unknown sqlite3 error".to_string()
            } else {
                stderr
            }
        )));
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub struct DocumentState {
    pub content_hash: String,
    pub markdown_path: String,
    pub created_at: Option<String>,
}

#[derive(Debug, Clone)]
pub struct DocumentUpsert {
    pub doc_id: String,
    pub plugin_id: String,
    pub connector_instance: String,
    pub external_id: String,
    pub title: String,
    pub content_hash: String,
    pub source_updated_at: String,
    pub deleted: bool,
    pub classification: String,
    pub doc_type: String,
    pub markdown_path: String,
    pub uri: Option<String>,
    pub tags_json: String,
    pub acl_tags_json: String,
    pub language: Option<String>,
    pub created_at: Option<String>,
}

#[derive(Debug, Clone)]
pub struct DocumentRow {
    pub doc_id: String,
    pub title: String,
    pub content_hash: String,
    pub doc_type: String,
    pub classification: String,
    pub markdown_path: String,
    pub tags_json: String,
    pub deleted: bool,
}

#[derive(Debug, Clone)]
pub struct DocumentIndexStateRow {
    pub doc_id: String,
    pub embedding_profile_id: String,
    pub status: String,
    pub indexed_content_hash: Option<String>,
    pub indexed_markdown_path: Option<String>,
}

#[derive(Debug, Clone)]
pub struct SyncStateEntry {
    pub sync_key: String,
    pub manifest_path: String,
    pub config_hash: String,
    pub plugin_id: Option<String>,
    pub cursor: Option<Value>,
    pub last_success_at: Option<String>,
    pub last_error_at: Option<String>,
    pub last_error: Option<String>,
    pub envelope_count_last_run: Option<u64>,
    pub updated_at: String,
}

pub struct MetadataStore {
    path: PathBuf,
}

impl MetadataStore {
    pub fn open(path: &Path) -> Result<Self, ColibriError> {
        Ok(Self {
            path: path.to_path_buf(),
        })
    }

    pub fn bootstrap(
        &mut self,
        embedding_profiles: &HashMap<String, EmbeddingProfile>,
        routing_policy: &HashMap<String, String>,
        default_embedding_profile: &str,
    ) -> Result<(), ColibriError> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let now = Utc::now().to_rfc3339();
        let mut sql = String::from(
            r#"
BEGIN;
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS schema_versions (
    component TEXT PRIMARY KEY,
    version INTEGER NOT NULL,
    applied_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS embedding_profiles (
    profile_id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    model TEXT NOT NULL,
    locality TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS routing_policy (
    classification TEXT PRIMARY KEY,
    profile_id TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    plugin_id TEXT NOT NULL,
    connector_instance TEXT NOT NULL,
    external_id TEXT NOT NULL,
    title TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    source_updated_at TEXT NOT NULL,
    deleted INTEGER NOT NULL,
    classification TEXT NOT NULL,
    doc_type TEXT NOT NULL,
    markdown_path TEXT NOT NULL,
    uri TEXT,
    tags_json TEXT NOT NULL,
    acl_tags_json TEXT NOT NULL,
    language TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS document_blobs (
    doc_id TEXT PRIMARY KEY,
    markdown_path TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    checksum TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sync_state (
    sync_key TEXT PRIMARY KEY,
    manifest_path TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    plugin_id TEXT,
    cursor_json TEXT,
    last_success_at TEXT,
    last_error_at TEXT,
    last_error TEXT,
    envelope_count_last_run INTEGER,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS index_generations (
    generation_id TEXT NOT NULL,
    embedding_profile_id TEXT NOT NULL,
    pipeline_version_json TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    activated_at TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (generation_id, embedding_profile_id)
);
CREATE TABLE IF NOT EXISTS document_index_state (
    doc_id TEXT NOT NULL,
    generation_id TEXT NOT NULL,
    embedding_profile_id TEXT NOT NULL,
    status TEXT NOT NULL,
    chunk_count INTEGER,
    indexed_content_hash TEXT,
    indexed_markdown_path TEXT,
    embedded_at TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (doc_id, generation_id, embedding_profile_id)
);
CREATE TABLE IF NOT EXISTS migration_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    component TEXT NOT NULL,
    from_version INTEGER,
    to_version INTEGER,
    applied_at TEXT NOT NULL,
    success INTEGER NOT NULL,
    notes TEXT
);
"#,
        );

        sql.push_str(&format!(
            "INSERT INTO meta(key, value) VALUES ('format', {}) ON CONFLICT(key) DO UPDATE SET value=excluded.value;\n",
            q(FORMAT_NAME)
        ));
        sql.push_str(&format!(
            "INSERT INTO meta(key, value) VALUES ('format_version', {}) ON CONFLICT(key) DO UPDATE SET value=excluded.value;\n",
            q(&METADATA_DB_FORMAT_VERSION.to_string())
        ));
        sql.push_str(&format!(
            "INSERT INTO meta(key, value) VALUES ('default_embedding_profile', {}) ON CONFLICT(key) DO UPDATE SET value=excluded.value;\n",
            q(default_embedding_profile)
        ));
        sql.push_str(&format!(
            "INSERT INTO meta(key, value) VALUES ('updated_at', {}) ON CONFLICT(key) DO UPDATE SET value=excluded.value;\n",
            q(&now)
        ));

        for (component, version) in [
            ("canonical", CANONICAL_SCHEMA_VERSION),
            ("pipeline", PIPELINE_SCHEMA_VERSION),
            ("serving", SERVING_SCHEMA_VERSION),
        ] {
            sql.push_str(&format!(
                "INSERT INTO schema_versions(component, version, applied_at) VALUES ({}, {}, {}) \
                 ON CONFLICT(component) DO UPDATE SET \
                   version = CASE WHEN excluded.version > schema_versions.version THEN excluded.version ELSE schema_versions.version END, \
                   applied_at = excluded.applied_at;\n",
                q(component),
                version as i64,
                q(&now)
            ));
        }

        sql.push_str("DELETE FROM embedding_profiles;\n");
        let mut ids: Vec<&String> = embedding_profiles.keys().collect();
        ids.sort();
        for id in ids {
            let Some(profile) = embedding_profiles.get(id) else {
                continue;
            };
            let locality = match profile.locality {
                EmbeddingLocality::Local => "local",
                EmbeddingLocality::Cloud => "cloud",
            };
            sql.push_str(&format!(
                "INSERT INTO embedding_profiles(profile_id, provider, endpoint, model, locality, updated_at) \
                 VALUES ({}, {}, {}, {}, {}, {});\n",
                q(&profile.id),
                q(&profile.provider),
                q(&profile.endpoint),
                q(&profile.model),
                q(locality),
                q(&now)
            ));
        }

        sql.push_str("DELETE FROM routing_policy;\n");
        let mut classes: Vec<&String> = routing_policy.keys().collect();
        classes.sort();
        for class in classes {
            if let Some(profile) = routing_policy.get(class) {
                sql.push_str(&format!(
                    "INSERT INTO routing_policy(classification, profile_id, updated_at) VALUES ({}, {}, {});\n",
                    q(class),
                    q(profile),
                    q(&now)
                ));
            }
        }

        sql.push_str("COMMIT;\n");
        self.exec_script(&sql)?;
        self.ensure_document_index_state_columns()?;
        Ok(())
    }

    fn ensure_document_index_state_columns(&self) -> Result<(), ColibriError> {
        self.ensure_columns(
            "document_index_state",
            &[
                ("indexed_content_hash", "TEXT"),
                ("indexed_markdown_path", "TEXT"),
            ],
        )
    }

    fn ensure_columns(&self, table: &str, columns: &[(&str, &str)]) -> Result<(), ColibriError> {
        let existing = self.table_columns(table)?;
        let mut sql = String::new();
        for (name, ty) in columns {
            if !existing.contains(*name) {
                sql.push_str(&format!("ALTER TABLE {table} ADD COLUMN {name} {ty};\n"));
            }
        }
        if sql.trim().is_empty() {
            return Ok(());
        }
        self.exec_script(&sql)
    }

    fn table_columns(&self, table: &str) -> Result<HashSet<String>, ColibriError> {
        // Table name is internal/constant; do not accept user-controlled input here.
        let rows = self.query_json(&format!("PRAGMA table_info({table});"))?;
        let mut out = HashSet::new();
        for row in rows {
            let Some(name) = row.get("name").and_then(Value::as_str) else {
                continue;
            };
            out.insert(name.to_string());
        }
        Ok(out)
    }

    pub fn read_versions(&self) -> Result<(Option<u32>, HashMap<String, u32>), ColibriError> {
        let format_rows =
            self.query_json("SELECT value FROM meta WHERE key='format_version' LIMIT 1")?;
        let format_version = format_rows
            .first()
            .and_then(|v| v.get("value"))
            .and_then(Value::as_str)
            .and_then(|s| s.parse::<u32>().ok());

        let rows = self.query_json("SELECT component, version FROM schema_versions")?;
        let mut components = HashMap::new();
        for row in rows {
            let Some(component) = row.get("component").and_then(Value::as_str) else {
                continue;
            };
            let Some(version) = row.get("version").and_then(Value::as_i64) else {
                continue;
            };
            if version >= 0 {
                components.insert(component.to_string(), version as u32);
            }
        }
        Ok((format_version, components))
    }

    pub fn get_document_state(&self, doc_id: &str) -> Result<Option<DocumentState>, ColibriError> {
        let rows = self.query_json(&format!(
            "SELECT content_hash, markdown_path, created_at FROM documents WHERE doc_id={} LIMIT 1",
            q(doc_id)
        ))?;
        let Some(row) = rows.first() else {
            return Ok(None);
        };
        Ok(Some(DocumentState {
            content_hash: row
                .get("content_hash")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            markdown_path: row
                .get("markdown_path")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            created_at: row
                .get("created_at")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned),
        }))
    }

    pub fn upsert_document(&self, row: &DocumentUpsert) -> Result<(), ColibriError> {
        let now = Utc::now().to_rfc3339();
        let created_at = row.created_at.clone().unwrap_or_else(|| now.clone());
        let sql = format!(
            "INSERT INTO documents(\
                doc_id, plugin_id, connector_instance, external_id, title, content_hash,\
                source_updated_at, deleted, classification, doc_type, markdown_path,\
                uri, tags_json, acl_tags_json, language, created_at, updated_at\
            ) VALUES ({doc_id}, {plugin_id}, {connector_instance}, {external_id}, {title}, {content_hash}, {source_updated_at}, {deleted}, {classification}, {doc_type}, {markdown_path}, {uri}, {tags_json}, {acl_tags_json}, {language}, {created_at}, {updated_at}) \
            ON CONFLICT(doc_id) DO UPDATE SET \
                plugin_id=excluded.plugin_id,\
                connector_instance=excluded.connector_instance,\
                external_id=excluded.external_id,\
                title=excluded.title,\
                content_hash=excluded.content_hash,\
                source_updated_at=excluded.source_updated_at,\
                deleted=excluded.deleted,\
                classification=excluded.classification,\
                doc_type=excluded.doc_type,\
                markdown_path=excluded.markdown_path,\
                uri=excluded.uri,\
                tags_json=excluded.tags_json,\
                acl_tags_json=excluded.acl_tags_json,\
                language=excluded.language,\
                created_at=documents.created_at,\
                updated_at=excluded.updated_at;",
            doc_id = q(&row.doc_id),
            plugin_id = q(&row.plugin_id),
            connector_instance = q(&row.connector_instance),
            external_id = q(&row.external_id),
            title = q(&row.title),
            content_hash = q(&row.content_hash),
            source_updated_at = q(&row.source_updated_at),
            deleted = if row.deleted { 1 } else { 0 },
            classification = q(&row.classification),
            doc_type = q(&row.doc_type),
            markdown_path = q(&row.markdown_path),
            uri = q_opt(row.uri.as_deref()),
            tags_json = q(&row.tags_json),
            acl_tags_json = q(&row.acl_tags_json),
            language = q_opt(row.language.as_deref()),
            created_at = q(&created_at),
            updated_at = q(&now),
        );
        self.exec_script(&sql)
    }

    pub fn document_count(&self) -> Result<usize, ColibriError> {
        let rows = self.query_json("SELECT COUNT(*) AS c FROM documents")?;
        let count = rows
            .first()
            .and_then(|row| row.get("c"))
            .and_then(Value::as_i64)
            .unwrap_or(0);
        Ok(if count < 0 { 0 } else { count as usize })
    }

    pub fn upsert_document_blob(
        &self,
        doc_id: &str,
        markdown_path: &str,
        size_bytes: u64,
        checksum: &str,
    ) -> Result<(), ColibriError> {
        let now = Utc::now().to_rfc3339();
        let sql = format!(
            "INSERT INTO document_blobs(doc_id, markdown_path, size_bytes, checksum, updated_at) \
             VALUES ({doc_id}, {markdown_path}, {size_bytes}, {checksum}, {updated_at}) \
             ON CONFLICT(doc_id) DO UPDATE SET \
                markdown_path=excluded.markdown_path,\
                size_bytes=excluded.size_bytes,\
                checksum=excluded.checksum,\
                updated_at=excluded.updated_at;",
            doc_id = q(doc_id),
            markdown_path = q(markdown_path),
            size_bytes = size_bytes as i64,
            checksum = q(checksum),
            updated_at = q(&now),
        );
        self.exec_script(&sql)
    }

    pub fn delete_document_blob(&self, doc_id: &str) -> Result<(), ColibriError> {
        self.exec_script(&format!(
            "DELETE FROM document_blobs WHERE doc_id={};",
            q(doc_id)
        ))
    }

    pub fn list_documents(&self) -> Result<Vec<DocumentRow>, ColibriError> {
        let rows = self.query_json(
            "SELECT doc_id, title, content_hash, doc_type, classification, markdown_path, tags_json, deleted \
             FROM documents ORDER BY doc_id",
        )?;
        let mut out = Vec::new();
        for row in rows {
            let Some(obj) = row.as_object() else {
                continue;
            };
            let Some(doc_id) = obj.get("doc_id").and_then(Value::as_str) else {
                continue;
            };
            let Some(title) = obj.get("title").and_then(Value::as_str) else {
                continue;
            };
            let Some(content_hash) = obj.get("content_hash").and_then(Value::as_str) else {
                continue;
            };
            let Some(doc_type) = obj.get("doc_type").and_then(Value::as_str) else {
                continue;
            };
            let Some(classification) = obj.get("classification").and_then(Value::as_str) else {
                continue;
            };
            let Some(markdown_path) = obj.get("markdown_path").and_then(Value::as_str) else {
                continue;
            };
            let tags_json = obj
                .get("tags_json")
                .and_then(Value::as_str)
                .unwrap_or("[]")
                .to_string();
            let deleted = obj.get("deleted").and_then(Value::as_i64).unwrap_or(0) != 0;

            out.push(DocumentRow {
                doc_id: doc_id.to_string(),
                title: title.to_string(),
                content_hash: content_hash.to_string(),
                doc_type: doc_type.to_string(),
                classification: classification.to_string(),
                markdown_path: markdown_path.to_string(),
                tags_json,
                deleted,
            });
        }
        Ok(out)
    }

    pub fn get_documents_by_ids(
        &self,
        doc_ids: &[String],
    ) -> Result<HashMap<String, DocumentRow>, ColibriError> {
        if doc_ids.is_empty() {
            return Ok(HashMap::new());
        }
        let in_list = doc_ids.iter().map(|id| q(id)).collect::<Vec<_>>().join(",");
        let rows = self.query_json(&format!(
            "SELECT doc_id, title, content_hash, doc_type, classification, markdown_path, tags_json, deleted \
             FROM documents WHERE doc_id IN ({in_list})"
        ))?;

        let mut out = HashMap::new();
        for row in rows {
            let Some(obj) = row.as_object() else {
                continue;
            };
            let Some(doc_id) = obj.get("doc_id").and_then(Value::as_str) else {
                continue;
            };
            let Some(title) = obj.get("title").and_then(Value::as_str) else {
                continue;
            };
            let Some(content_hash) = obj.get("content_hash").and_then(Value::as_str) else {
                continue;
            };
            let Some(doc_type) = obj.get("doc_type").and_then(Value::as_str) else {
                continue;
            };
            let Some(classification) = obj.get("classification").and_then(Value::as_str) else {
                continue;
            };
            let Some(markdown_path) = obj.get("markdown_path").and_then(Value::as_str) else {
                continue;
            };
            let tags_json = obj
                .get("tags_json")
                .and_then(Value::as_str)
                .unwrap_or("[]")
                .to_string();
            let deleted = obj.get("deleted").and_then(Value::as_i64).unwrap_or(0) != 0;

            out.insert(
                doc_id.to_string(),
                DocumentRow {
                    doc_id: doc_id.to_string(),
                    title: title.to_string(),
                    content_hash: content_hash.to_string(),
                    doc_type: doc_type.to_string(),
                    classification: classification.to_string(),
                    markdown_path: markdown_path.to_string(),
                    tags_json,
                    deleted,
                },
            );
        }
        Ok(out)
    }

    pub fn indexed_chunk_counts_for_generation(
        &self,
        generation_id: &str,
    ) -> Result<HashMap<(String, String), u64>, ColibriError> {
        let rows = self.query_json(&format!(
            "SELECT doc_id, embedding_profile_id, chunk_count \
             FROM document_index_state \
             WHERE generation_id={} AND status='indexed' AND chunk_count IS NOT NULL",
            q(generation_id)
        ))?;
        let mut out = HashMap::new();
        for row in rows {
            let Some(obj) = row.as_object() else {
                continue;
            };
            let Some(doc_id) = obj.get("doc_id").and_then(Value::as_str) else {
                continue;
            };
            let Some(profile_id) = obj.get("embedding_profile_id").and_then(Value::as_str) else {
                continue;
            };
            let Some(chunk_count) = obj.get("chunk_count").and_then(Value::as_i64) else {
                continue;
            };
            if chunk_count >= 0 {
                out.insert(
                    (doc_id.to_string(), profile_id.to_string()),
                    chunk_count as u64,
                );
            }
        }
        Ok(out)
    }

    pub fn list_document_index_state_for_generation(
        &self,
        generation_id: &str,
    ) -> Result<Vec<DocumentIndexStateRow>, ColibriError> {
        let rows = self.query_json(&format!(
            "SELECT doc_id, embedding_profile_id, status, indexed_content_hash, indexed_markdown_path \
             FROM document_index_state WHERE generation_id={} \
             ORDER BY doc_id, embedding_profile_id",
            q(generation_id)
        ))?;
        let mut out = Vec::new();
        for row in rows {
            let Some(obj) = row.as_object() else {
                continue;
            };
            let Some(doc_id) = obj.get("doc_id").and_then(Value::as_str) else {
                continue;
            };
            let Some(embedding_profile_id) =
                obj.get("embedding_profile_id").and_then(Value::as_str)
            else {
                continue;
            };
            let Some(status) = obj.get("status").and_then(Value::as_str) else {
                continue;
            };
            out.push(DocumentIndexStateRow {
                doc_id: doc_id.to_string(),
                embedding_profile_id: embedding_profile_id.to_string(),
                status: status.to_string(),
                indexed_content_hash: obj
                    .get("indexed_content_hash")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned),
                indexed_markdown_path: obj
                    .get("indexed_markdown_path")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned),
            });
        }
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn upsert_document_index_state(
        &self,
        doc_id: &str,
        generation_id: &str,
        embedding_profile_id: &str,
        status: &str,
        chunk_count: Option<u64>,
        indexed_content_hash: Option<&str>,
        indexed_markdown_path: Option<&str>,
    ) -> Result<(), ColibriError> {
        let now = Utc::now().to_rfc3339();
        let sql = format!(
            "INSERT INTO document_index_state(\
                doc_id, generation_id, embedding_profile_id, status, chunk_count, indexed_content_hash, indexed_markdown_path, embedded_at, updated_at\
             ) VALUES ({doc_id}, {generation_id}, {embedding_profile_id}, {status}, {chunk_count}, {indexed_content_hash}, {indexed_markdown_path}, {embedded_at}, {updated_at}) \
             ON CONFLICT(doc_id, generation_id, embedding_profile_id) DO UPDATE SET \
                status=excluded.status,\
                chunk_count=excluded.chunk_count,\
                indexed_content_hash=excluded.indexed_content_hash,\
                indexed_markdown_path=excluded.indexed_markdown_path,\
                embedded_at=excluded.embedded_at,\
                updated_at=excluded.updated_at;",
            doc_id = q(doc_id),
            generation_id = q(generation_id),
            embedding_profile_id = q(embedding_profile_id),
            status = q(status),
            chunk_count = chunk_count
                .map(|n| (n as i64).to_string())
                .unwrap_or_else(|| "NULL".into()),
            indexed_content_hash = q_opt(indexed_content_hash),
            indexed_markdown_path = q_opt(indexed_markdown_path),
            embedded_at = q(&now),
            updated_at = q(&now),
        );
        self.exec_script(&sql)
    }

    pub fn append_migration_log(
        &self,
        component: &str,
        from_version: Option<u32>,
        to_version: u32,
        success: bool,
        notes: Option<&str>,
    ) -> Result<(), ColibriError> {
        let sql = format!(
            "INSERT INTO migration_log(component, from_version, to_version, applied_at, success, notes) \
             VALUES ({component}, {from_version}, {to_version}, {applied_at}, {success}, {notes});",
            component = q(component),
            from_version = from_version
                .map(|v| (v as i64).to_string())
                .unwrap_or_else(|| "NULL".into()),
            to_version = to_version as i64,
            applied_at = q(&Utc::now().to_rfc3339()),
            success = if success { 1 } else { 0 },
            notes = q_opt(notes),
        );
        self.exec_script(&sql)
    }

    pub fn touch_updated_at(&self) -> Result<(), ColibriError> {
        self.exec_script(&format!(
            "INSERT INTO meta(key, value) VALUES ('updated_at', {}) \
             ON CONFLICT(key) DO UPDATE SET value=excluded.value;",
            q(&Utc::now().to_rfc3339())
        ))
    }

    pub fn get_sync_cursor(&self, sync_key: &str) -> Result<Option<Value>, ColibriError> {
        let rows = self.query_json(&format!(
            "SELECT cursor_json FROM sync_state WHERE sync_key={} LIMIT 1",
            q(sync_key)
        ))?;
        let Some(row) = rows.first() else {
            return Ok(None);
        };
        let raw = row.get("cursor_json").and_then(Value::as_str).unwrap_or("");
        if raw.trim().is_empty() {
            return Ok(None);
        }
        Ok(serde_json::from_str(raw).ok())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn upsert_sync_success(
        &self,
        sync_key: &str,
        manifest_path: &str,
        config_hash: &str,
        plugin_id: &str,
        cursor: Option<Value>,
        envelope_count_last_run: usize,
    ) -> Result<(), ColibriError> {
        let now = Utc::now().to_rfc3339();
        let cursor_json = cursor.map(|v| serde_json::to_string(&v)).transpose()?;
        let sql = format!(
            "INSERT INTO sync_state(\
                sync_key, manifest_path, config_hash, plugin_id, cursor_json,\
                last_success_at, last_error_at, last_error, envelope_count_last_run, updated_at\
             ) VALUES ({sync_key}, {manifest_path}, {config_hash}, {plugin_id}, {cursor_json}, {last_success_at}, NULL, NULL, {envelope_count}, {updated_at}) \
             ON CONFLICT(sync_key) DO UPDATE SET \
                manifest_path=excluded.manifest_path,\
                config_hash=excluded.config_hash,\
                plugin_id=excluded.plugin_id,\
                cursor_json=excluded.cursor_json,\
                last_success_at=excluded.last_success_at,\
                last_error_at=NULL,\
                last_error=NULL,\
                envelope_count_last_run=excluded.envelope_count_last_run,\
                updated_at=excluded.updated_at;",
            sync_key = q(sync_key),
            manifest_path = q(manifest_path),
            config_hash = q(config_hash),
            plugin_id = q(plugin_id),
            cursor_json = q_opt(cursor_json.as_deref()),
            last_success_at = q(&now),
            envelope_count = envelope_count_last_run as i64,
            updated_at = q(&Utc::now().to_rfc3339()),
        );
        self.exec_script(&sql)
    }

    pub fn upsert_sync_error(
        &self,
        sync_key: &str,
        manifest_path: &str,
        config_hash: &str,
        error: &str,
    ) -> Result<(), ColibriError> {
        let now = Utc::now().to_rfc3339();
        let sql = format!(
            "INSERT INTO sync_state(\
                sync_key, manifest_path, config_hash, plugin_id, cursor_json,\
                last_success_at, last_error_at, last_error, envelope_count_last_run, updated_at\
             ) VALUES ({sync_key}, {manifest_path}, {config_hash}, NULL, NULL, NULL, {last_error_at}, {last_error}, NULL, {updated_at}) \
             ON CONFLICT(sync_key) DO UPDATE SET \
                manifest_path=excluded.manifest_path,\
                config_hash=excluded.config_hash,\
                last_error_at=excluded.last_error_at,\
                last_error=excluded.last_error,\
                updated_at=excluded.updated_at;",
            sync_key = q(sync_key),
            manifest_path = q(manifest_path),
            config_hash = q(config_hash),
            last_error_at = q(&now),
            last_error = q(error),
            updated_at = q(&Utc::now().to_rfc3339()),
        );
        self.exec_script(&sql)
    }

    pub fn list_sync_entries(&self) -> Result<Vec<SyncStateEntry>, ColibriError> {
        let rows = self.query_json(
            "SELECT sync_key, manifest_path, config_hash, plugin_id, cursor_json,\
                    last_success_at, last_error_at, last_error, envelope_count_last_run, updated_at \
             FROM sync_state ORDER BY sync_key",
        )?;
        rows.into_iter().map(parse_sync_row).collect()
    }

    pub fn get_sync_entry(&self, sync_key: &str) -> Result<Option<SyncStateEntry>, ColibriError> {
        let rows = self.query_json(&format!(
            "SELECT sync_key, manifest_path, config_hash, plugin_id, cursor_json,\
                    last_success_at, last_error_at, last_error, envelope_count_last_run, updated_at \
             FROM sync_state WHERE sync_key={} LIMIT 1",
            q(sync_key)
        ))?;
        let Some(row) = rows.into_iter().next() else {
            return Ok(None);
        };
        Ok(Some(parse_sync_row(row)?))
    }

    pub fn delete_sync_entry(&self, sync_key: &str) -> Result<bool, ColibriError> {
        let rows = self.query_json(&format!(
            "DELETE FROM sync_state WHERE sync_key={}; SELECT changes() AS affected;",
            q(sync_key)
        ))?;
        let affected = rows
            .first()
            .and_then(|r| r.get("affected"))
            .and_then(Value::as_i64)
            .unwrap_or(0);
        Ok(affected > 0)
    }

    fn exec_script(&self, sql: &str) -> Result<(), ColibriError> {
        require_sqlite3()?;
        let output = Command::new("sqlite3").arg(&self.path).arg(sql).output()?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
            return Err(ColibriError::Config(format!(
                "sqlite3 execution failed for {}: {}",
                self.path.display(),
                if stderr.is_empty() {
                    "unknown sqlite3 error".to_string()
                } else {
                    stderr
                }
            )));
        }
        Ok(())
    }

    fn query_json(&self, sql: &str) -> Result<Vec<Value>, ColibriError> {
        require_sqlite3()?;
        let output = Command::new("sqlite3")
            .arg("-json")
            .arg(&self.path)
            .arg(sql)
            .output()?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
            return Err(ColibriError::Config(format!(
                "sqlite3 query failed for {}: {}",
                self.path.display(),
                if stderr.is_empty() {
                    "unknown sqlite3 error".to_string()
                } else {
                    stderr
                }
            )));
        }
        let stdout = String::from_utf8_lossy(&output.stdout);
        let trimmed = stdout.trim();
        if trimmed.is_empty() {
            return Ok(Vec::new());
        }
        let parsed: Value = serde_json::from_str(trimmed)?;
        let rows = parsed.as_array().cloned().unwrap_or_default();
        Ok(rows)
    }
}

fn parse_sync_row(row: Value) -> Result<SyncStateEntry, ColibriError> {
    let obj = row
        .as_object()
        .ok_or_else(|| ColibriError::Config("Invalid sync_state row shape".into()))?;
    let cursor = obj
        .get("cursor_json")
        .and_then(Value::as_str)
        .filter(|s| !s.trim().is_empty())
        .and_then(|s| serde_json::from_str::<Value>(s).ok());
    let envelope_count_last_run = obj
        .get("envelope_count_last_run")
        .and_then(Value::as_i64)
        .and_then(|v| if v >= 0 { Some(v as u64) } else { None });

    Ok(SyncStateEntry {
        sync_key: obj
            .get("sync_key")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        manifest_path: obj
            .get("manifest_path")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        config_hash: obj
            .get("config_hash")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        plugin_id: obj
            .get("plugin_id")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        cursor,
        last_success_at: obj
            .get("last_success_at")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        last_error_at: obj
            .get("last_error_at")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        last_error: obj
            .get("last_error")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        envelope_count_last_run,
        updated_at: obj
            .get("updated_at")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
    })
}

fn q(raw: &str) -> String {
    format!("'{}'", raw.replace('\'', "''"))
}

fn q_opt(raw: Option<&str>) -> String {
    raw.map(q).unwrap_or_else(|| "NULL".to_string())
}

pub fn read_metadata_versions(
    metadata_db_path: &Path,
) -> Result<(Option<u32>, HashMap<String, u32>), ColibriError> {
    if !metadata_db_path.exists() {
        return Ok((None, HashMap::new()));
    }
    let store = MetadataStore::open(metadata_db_path)?;
    store.read_versions()
}

#[cfg(test)]
mod tests {
    use super::{DocumentUpsert, MetadataStore};
    use crate::config::{EmbeddingLocality, EmbeddingProfile};
    use serde_json::json;
    use std::collections::HashMap;
    use std::path::PathBuf;

    fn temp_db_path() -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "colibri-metadata-store-test-{}-{}.db",
            std::process::id(),
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0)
        ));
        path
    }

    fn bootstrap_store(path: &std::path::Path) -> MetadataStore {
        let mut profiles = HashMap::new();
        profiles.insert(
            "local_default".into(),
            EmbeddingProfile {
                id: "local_default".into(),
                provider: "ollama".into(),
                endpoint: "http://localhost:11434".into(),
                model: "bge-m3".into(),
                locality: EmbeddingLocality::Local,
            },
        );
        let mut routing = HashMap::new();
        routing.insert("internal".into(), "local_default".into());
        routing.insert("restricted".into(), "local_default".into());
        routing.insert("confidential".into(), "local_default".into());
        routing.insert("public".into(), "local_default".into());

        let mut store = MetadataStore::open(path).expect("open metadata store");
        store
            .bootstrap(&profiles, &routing, "local_default")
            .expect("bootstrap metadata store");
        store
    }

    #[test]
    fn document_index_state_roundtrip_for_live_document() {
        let db_path = temp_db_path();
        let store = bootstrap_store(&db_path);

        let row = DocumentUpsert {
            doc_id: "doc_1".into(),
            plugin_id: "filesystem_documents".into(),
            connector_instance: "local".into(),
            external_id: "docs/a.md".into(),
            title: "Doc A".into(),
            content_hash: "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                .into(),
            source_updated_at: "2026-02-18T00:00:00Z".into(),
            deleted: false,
            classification: "internal".into(),
            doc_type: "note".into(),
            markdown_path: "internal/filesystem_documents/local/doc-a.md".into(),
            uri: None,
            tags_json: "[]".into(),
            acl_tags_json: "[]".into(),
            language: Some("en".into()),
            created_at: None,
        };
        store.upsert_document(&row).expect("upsert document");

        let docs = store.list_documents().expect("list documents");
        let found = docs
            .iter()
            .find(|d| d.doc_id == "doc_1")
            .expect("doc exists");
        assert!(!found.deleted);
        assert_eq!(
            found.markdown_path,
            "internal/filesystem_documents/local/doc-a.md"
        );

        store
            .upsert_document_index_state(
                "doc_1",
                "gen_a",
                "local_default",
                "indexed",
                Some(7),
                Some("sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
                Some("internal/filesystem_documents/local/doc-a.md"),
            )
            .expect("upsert indexed state");
        store
            .upsert_document_index_state(
                "doc_1",
                "gen_a",
                "local_default",
                "deleted",
                None,
                None,
                Some("internal/filesystem_documents/local/doc-a.md"),
            )
            .expect("upsert deleted state");

        let rows = store
            .query_json(
                "SELECT status, chunk_count, indexed_markdown_path FROM document_index_state \
                 WHERE doc_id='doc_1' AND generation_id='gen_a' AND embedding_profile_id='local_default' \
                 LIMIT 1",
            )
            .expect("query document_index_state row");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["status"], json!("deleted"));
        assert!(rows[0]["chunk_count"].is_null());
        assert_eq!(
            rows[0]["indexed_markdown_path"],
            json!("internal/filesystem_documents/local/doc-a.md")
        );

        let _ = std::fs::remove_file(db_path);
    }
}
