//! SQLite metadata store (schema v7): documents, conversions, collections, problems.
//!
//! Uses the rollback journal (not WAL) so read-only connections never create
//! or touch files next to the database. Writers wait up to [`BUSY_TIMEOUT`]
//! for locks held by other colibri processes.

use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;

use chrono::Utc;
use rusqlite::{params, params_from_iter, Connection, OpenFlags, OptionalExtension, Row};

use crate::error::ColibriError;

/// Schema version stored in `PRAGMA user_version`.
pub const METADATA_SCHEMA_VERSION: i64 = 7;

/// How long a connection waits for a lock held by another colibri process.
const BUSY_TIMEOUT: Duration = Duration::from_secs(5);

/// SQLite limits the number of bound parameters; stay well below it.
const MAX_IDS_PER_QUERY: usize = 500;

const SCHEMA_SQL: &str = r#"
CREATE TABLE documents (
    doc_id TEXT PRIMARY KEY,
    collection TEXT NOT NULL,
    key TEXT NOT NULL,
    source_path TEXT,
    title TEXT NOT NULL,
    authors_json TEXT NOT NULL DEFAULT '[]',
    tags_json TEXT NOT NULL DEFAULT '[]',
    language TEXT,
    frontmatter_json TEXT NOT NULL DEFAULT '{}',
    doc_type TEXT NOT NULL,
    format TEXT,
    format_pinned INTEGER NOT NULL DEFAULT 0,
    converter TEXT,
    source_size INTEGER,
    source_mtime TEXT,
    source_sha256 TEXT,
    source_updated_at TEXT,
    content_hash TEXT NOT NULL,
    markdown_path TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'removed')),
    source_missing INTEGER NOT NULL DEFAULT 0,
    removed_reason TEXT,
    indexed_hash TEXT,
    chunk_count INTEGER,
    indexed_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_seen_at TEXT
);
CREATE INDEX documents_collection ON documents (collection);
CREATE TABLE conversions (
    source_sha256 TEXT PRIMARY KEY,
    converter TEXT NOT NULL,
    markdown_path TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE collections (
    name TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    root TEXT,
    last_run_at TEXT,
    last_run_status TEXT,
    last_run_report_json TEXT
);
CREATE TABLE problems (
    collection TEXT NOT NULL,
    source_path TEXT NOT NULL,
    kind TEXT NOT NULL,
    message TEXT NOT NULL,
    seen_at TEXT NOT NULL,
    PRIMARY KEY (collection, source_path)
);
CREATE TABLE meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"#;

const DOCUMENT_COLUMNS: &str = "doc_id, collection, key, source_path, title, authors_json, \
     tags_json, language, frontmatter_json, doc_type, format, format_pinned, converter, \
     source_size, source_mtime, source_sha256, source_updated_at, content_hash, markdown_path, \
     status, source_missing, removed_reason, indexed_hash, chunk_count, indexed_at, created_at, \
     updated_at, last_seen_at";

/// Lifecycle state of a document. Only `Active` documents are searchable.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DocStatus {
    #[default]
    Active,
    /// Removed by the user (library) or pruned because the source is gone (mirror).
    Removed,
}

impl DocStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            DocStatus::Active => "active",
            DocStatus::Removed => "removed",
        }
    }

    fn parse(raw: &str) -> Self {
        match raw {
            "removed" => DocStatus::Removed,
            _ => DocStatus::Active,
        }
    }
}

/// One row of the `documents` table.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DocumentRecord {
    pub doc_id: String,
    pub collection: String,
    /// Identity within the collection (relative path, upstream key, book identity).
    pub key: String,
    /// User-facing location of the source (absolute path or URI).
    pub source_path: Option<String>,
    pub title: String,
    pub authors_json: String,
    pub tags_json: String,
    pub language: Option<String>,
    pub frontmatter_json: String,
    pub doc_type: String,
    pub format: Option<String>,
    pub format_pinned: bool,
    pub converter: Option<String>,
    pub source_size: Option<i64>,
    pub source_mtime: Option<String>,
    pub source_sha256: Option<String>,
    /// Last-modified timestamp reported by the source (RFC 3339).
    pub source_updated_at: Option<String>,
    pub content_hash: String,
    /// Path of the canonical markdown, relative to the canonical dir.
    pub markdown_path: String,
    pub status: DocStatus,
    pub source_missing: bool,
    pub removed_reason: Option<String>,
    /// Content hash the current index chunks were built from.
    pub indexed_hash: Option<String>,
    pub chunk_count: Option<i64>,
    pub indexed_at: Option<String>,
    pub created_at: String,
    pub updated_at: String,
    pub last_seen_at: Option<String>,
}

impl DocumentRecord {
    /// A new active document with empty JSON fields and fresh timestamps.
    pub fn new(
        doc_id: impl Into<String>,
        collection: impl Into<String>,
        key: impl Into<String>,
    ) -> Self {
        let now = Utc::now().to_rfc3339();
        Self {
            doc_id: doc_id.into(),
            collection: collection.into(),
            key: key.into(),
            authors_json: "[]".into(),
            tags_json: "[]".into(),
            frontmatter_json: "{}".into(),
            created_at: now.clone(),
            updated_at: now,
            ..Default::default()
        }
    }

    pub fn is_searchable(&self) -> bool {
        self.status == DocStatus::Active
    }

    /// True when the index holds chunks built from the current content.
    pub fn is_index_current(&self) -> bool {
        self.indexed_hash.as_deref() == Some(self.content_hash.as_str())
    }

    fn from_row(row: &Row<'_>) -> rusqlite::Result<Self> {
        Ok(Self {
            doc_id: row.get(0)?,
            collection: row.get(1)?,
            key: row.get(2)?,
            source_path: row.get(3)?,
            title: row.get(4)?,
            authors_json: row.get(5)?,
            tags_json: row.get(6)?,
            language: row.get(7)?,
            frontmatter_json: row.get(8)?,
            doc_type: row.get(9)?,
            format: row.get(10)?,
            format_pinned: row.get(11)?,
            converter: row.get(12)?,
            source_size: row.get(13)?,
            source_mtime: row.get(14)?,
            source_sha256: row.get(15)?,
            source_updated_at: row.get(16)?,
            content_hash: row.get(17)?,
            markdown_path: row.get(18)?,
            status: DocStatus::parse(&row.get::<_, String>(19)?),
            source_missing: row.get(20)?,
            removed_reason: row.get(21)?,
            indexed_hash: row.get(22)?,
            chunk_count: row.get(23)?,
            indexed_at: row.get(24)?,
            created_at: row.get(25)?,
            updated_at: row.get(26)?,
            last_seen_at: row.get(27)?,
        })
    }
}

/// Handle to the metadata DB. Open with [`MetadataStore::open_rw`] for write
/// commands (under the write lock) or [`MetadataStore::open_ro`] for reads.
pub struct MetadataStore {
    conn: Connection,
}

fn reset_hint(path: &Path, detail: &str) -> ColibriError {
    ColibriError::Config(format!(
        "Metadata DB {} cannot be used ({detail}). It was left unchanged. \
         Run `colibri reset` to rebuild CoLibri's data (your sources are not touched).",
        path.display()
    ))
}

impl MetadataStore {
    /// Open for reading and writing, creating the schema in a new or empty file.
    ///
    /// A file that is not a v7 CoLibri database is never modified: the call
    /// fails with a hint to run `colibri reset`.
    pub fn open_rw(path: &Path) -> Result<Self, ColibriError> {
        let conn = Connection::open(path)?;
        conn.busy_timeout(BUSY_TIMEOUT)?;
        let store = Self { conn };
        match store.schema_state()? {
            SchemaState::Current => {}
            SchemaState::Empty => store.create_schema()?,
            SchemaState::Foreign(detail) => return Err(reset_hint(path, &detail)),
        }
        Ok(store)
    }

    /// Open read-only. Fails if the DB does not exist or has another schema.
    pub fn open_ro(path: &Path) -> Result<Self, ColibriError> {
        if !path.exists() {
            return Err(ColibriError::Config(format!(
                "No CoLibri data yet ({} does not exist). Ingest content first.",
                path.display()
            )));
        }
        let flags = OpenFlags::SQLITE_OPEN_READ_ONLY
            | OpenFlags::SQLITE_OPEN_URI
            | OpenFlags::SQLITE_OPEN_NO_MUTEX;
        let conn = Connection::open_with_flags(path, flags)?;
        conn.busy_timeout(BUSY_TIMEOUT)?;
        let store = Self { conn };
        match store.schema_state()? {
            SchemaState::Current => Ok(store),
            SchemaState::Empty => Err(reset_hint(path, "empty database")),
            SchemaState::Foreign(detail) => Err(reset_hint(path, &detail)),
        }
    }

    /// Start a transaction. Store methods called while it is open run inside it.
    pub fn begin(&self) -> Result<rusqlite::Transaction<'_>, ColibriError> {
        Ok(self.conn.unchecked_transaction()?)
    }

    fn schema_state(&self) -> Result<SchemaState, ColibriError> {
        let version: i64 = match self.conn.query_row("PRAGMA user_version", [], |r| r.get(0)) {
            Ok(v) => v,
            Err(e) => return classify_open_error(e),
        };
        if version == METADATA_SCHEMA_VERSION {
            return Ok(SchemaState::Current);
        }
        if version != 0 {
            return Ok(SchemaState::Foreign(format!(
                "schema v{version}, this colibri needs v{METADATA_SCHEMA_VERSION}"
            )));
        }
        let tables: i64 = self.conn.query_row(
            "SELECT count(*) FROM sqlite_master WHERE type = 'table'",
            [],
            |r| r.get(0),
        )?;
        if tables == 0 {
            Ok(SchemaState::Empty)
        } else {
            Ok(SchemaState::Foreign(format!(
                "pre-v{METADATA_SCHEMA_VERSION} schema"
            )))
        }
    }

    fn create_schema(&self) -> Result<(), ColibriError> {
        let tx = self.begin()?;
        tx.execute_batch(SCHEMA_SQL)?;
        tx.pragma_update(None, "user_version", METADATA_SCHEMA_VERSION)?;
        tx.commit()?;
        Ok(())
    }

    // -- documents ---------------------------------------------------------

    /// Insert or fully replace a document row (`created_at` of an existing row is kept).
    pub fn upsert_document(&self, doc: &DocumentRecord) -> Result<(), ColibriError> {
        self.conn.execute(
            &format!(
                "INSERT INTO documents ({DOCUMENT_COLUMNS}) VALUES \
                 (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, \
                  ?18, ?19, ?20, ?21, ?22, ?23, ?24, ?25, ?26, ?27, ?28)
                 ON CONFLICT(doc_id) DO UPDATE SET
                    collection = excluded.collection, key = excluded.key,
                    source_path = excluded.source_path, title = excluded.title,
                    authors_json = excluded.authors_json, tags_json = excluded.tags_json,
                    language = excluded.language, frontmatter_json = excluded.frontmatter_json,
                    doc_type = excluded.doc_type, format = excluded.format,
                    format_pinned = excluded.format_pinned, converter = excluded.converter,
                    source_size = excluded.source_size, source_mtime = excluded.source_mtime,
                    source_sha256 = excluded.source_sha256,
                    source_updated_at = excluded.source_updated_at,
                    content_hash = excluded.content_hash, markdown_path = excluded.markdown_path,
                    status = excluded.status, source_missing = excluded.source_missing,
                    removed_reason = excluded.removed_reason, indexed_hash = excluded.indexed_hash,
                    chunk_count = excluded.chunk_count, indexed_at = excluded.indexed_at,
                    updated_at = excluded.updated_at, last_seen_at = excluded.last_seen_at"
            ),
            params![
                doc.doc_id,
                doc.collection,
                doc.key,
                doc.source_path,
                doc.title,
                doc.authors_json,
                doc.tags_json,
                doc.language,
                doc.frontmatter_json,
                doc.doc_type,
                doc.format,
                doc.format_pinned,
                doc.converter,
                doc.source_size,
                doc.source_mtime,
                doc.source_sha256,
                doc.source_updated_at,
                doc.content_hash,
                doc.markdown_path,
                doc.status.as_str(),
                doc.source_missing,
                doc.removed_reason,
                doc.indexed_hash,
                doc.chunk_count,
                doc.indexed_at,
                doc.created_at,
                doc.updated_at,
                doc.last_seen_at,
            ],
        )?;
        Ok(())
    }

    pub fn get_document(&self, doc_id: &str) -> Result<Option<DocumentRecord>, ColibriError> {
        Ok(self
            .conn
            .query_row(
                &format!("SELECT {DOCUMENT_COLUMNS} FROM documents WHERE doc_id = ?1"),
                [doc_id],
                DocumentRecord::from_row,
            )
            .optional()?)
    }

    /// All documents (any status), ordered by `doc_id`.
    pub fn list_documents(&self) -> Result<Vec<DocumentRecord>, ColibriError> {
        let mut stmt = self.conn.prepare(&format!(
            "SELECT {DOCUMENT_COLUMNS} FROM documents ORDER BY doc_id"
        ))?;
        let rows = stmt.query_map([], DocumentRecord::from_row)?;
        Ok(rows.collect::<Result<Vec<_>, _>>()?)
    }

    pub fn get_documents_by_ids(
        &self,
        doc_ids: &[String],
    ) -> Result<HashMap<String, DocumentRecord>, ColibriError> {
        let mut out = HashMap::new();
        for batch in doc_ids.chunks(MAX_IDS_PER_QUERY) {
            let placeholders = vec!["?"; batch.len()].join(", ");
            let mut stmt = self.conn.prepare(&format!(
                "SELECT {DOCUMENT_COLUMNS} FROM documents WHERE doc_id IN ({placeholders})"
            ))?;
            let rows = stmt.query_map(params_from_iter(batch.iter()), DocumentRecord::from_row)?;
            for row in rows {
                let row = row?;
                out.insert(row.doc_id.clone(), row);
            }
        }
        Ok(out)
    }

    pub fn document_count(&self) -> Result<usize, ColibriError> {
        let n: i64 = self
            .conn
            .query_row("SELECT count(*) FROM documents", [], |r| r.get(0))?;
        Ok(n as usize)
    }

    pub fn set_status(
        &self,
        doc_id: &str,
        status: DocStatus,
        removed_reason: Option<&str>,
    ) -> Result<(), ColibriError> {
        self.conn.execute(
            "UPDATE documents SET status = ?2, removed_reason = ?3, updated_at = ?4 \
             WHERE doc_id = ?1",
            params![doc_id, status.as_str(), removed_reason, now()],
        )?;
        Ok(())
    }

    /// Record that the index holds `chunk_count` chunks built from `indexed_hash`.
    pub fn mark_indexed(
        &self,
        doc_id: &str,
        indexed_hash: &str,
        chunk_count: usize,
    ) -> Result<(), ColibriError> {
        self.conn.execute(
            "UPDATE documents SET indexed_hash = ?2, chunk_count = ?3, indexed_at = ?4 \
             WHERE doc_id = ?1",
            params![doc_id, indexed_hash, chunk_count as i64, now()],
        )?;
        Ok(())
    }

    /// Record that the index holds no chunks for this document.
    pub fn clear_index_state(&self, doc_id: &str) -> Result<(), ColibriError> {
        self.conn.execute(
            "UPDATE documents SET indexed_hash = NULL, chunk_count = NULL, indexed_at = NULL \
             WHERE doc_id = ?1",
            [doc_id],
        )?;
        Ok(())
    }

    /// Forget all index state (before a full index rebuild).
    pub fn clear_all_index_state(&self) -> Result<(), ColibriError> {
        self.conn.execute(
            "UPDATE documents SET indexed_hash = NULL, chunk_count = NULL, indexed_at = NULL",
            [],
        )?;
        Ok(())
    }
}

/// A problem found during the latest run of a collection.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct ProblemRow {
    pub collection: String,
    pub source_path: String,
    pub kind: String,
    pub message: String,
    pub seen_at: String,
}

/// Outcome of the latest run of a collection.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct CollectionRun {
    pub name: String,
    pub kind: String,
    pub root: Option<String>,
    pub last_run_at: Option<String>,
    pub last_run_status: Option<String>,
}

impl MetadataStore {
    /// All documents of one collection (any status).
    pub fn list_documents_in(&self, collection: &str) -> Result<Vec<DocumentRecord>, ColibriError> {
        let mut stmt = self.conn.prepare(&format!(
            "SELECT {DOCUMENT_COLUMNS} FROM documents WHERE collection = ?1 ORDER BY doc_id"
        ))?;
        let rows = stmt.query_map([collection], DocumentRecord::from_row)?;
        Ok(rows.collect::<Result<Vec<_>, _>>()?)
    }

    pub fn record_collection_run(
        &self,
        name: &str,
        kind: &str,
        root: Option<&str>,
        status: &str,
        report_json: &str,
    ) -> Result<(), ColibriError> {
        self.conn.execute(
            "INSERT INTO collections (name, kind, root, last_run_at, last_run_status, last_run_report_json)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)
             ON CONFLICT(name) DO UPDATE SET kind = excluded.kind, root = excluded.root,
                last_run_at = excluded.last_run_at, last_run_status = excluded.last_run_status,
                last_run_report_json = excluded.last_run_report_json",
            params![name, kind, root, now(), status, report_json],
        )?;
        Ok(())
    }

    pub fn list_collection_runs(&self) -> Result<Vec<CollectionRun>, ColibriError> {
        let mut stmt = self.conn.prepare(
            "SELECT name, kind, root, last_run_at, last_run_status FROM collections ORDER BY name",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(CollectionRun {
                name: r.get(0)?,
                kind: r.get(1)?,
                root: r.get(2)?,
                last_run_at: r.get(3)?,
                last_run_status: r.get(4)?,
            })
        })?;
        Ok(rows.collect::<Result<Vec<_>, _>>()?)
    }

    /// Replace the problems of a collection with those of the latest run.
    pub fn replace_problems(
        &self,
        collection: &str,
        problems: &[(String, String, String)],
    ) -> Result<(), ColibriError> {
        self.conn
            .execute("DELETE FROM problems WHERE collection = ?1", [collection])?;
        let seen_at = now();
        for (source_path, kind, message) in problems {
            self.conn.execute(
                "INSERT OR REPLACE INTO problems (collection, source_path, kind, message, seen_at)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![collection, source_path, kind, message, seen_at],
            )?;
        }
        Ok(())
    }

    pub fn list_problems(&self) -> Result<Vec<ProblemRow>, ColibriError> {
        let mut stmt = self.conn.prepare(
            "SELECT collection, source_path, kind, message, seen_at FROM problems
             ORDER BY collection, source_path",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(ProblemRow {
                collection: r.get(0)?,
                source_path: r.get(1)?,
                kind: r.get(2)?,
                message: r.get(3)?,
                seen_at: r.get(4)?,
            })
        })?;
        Ok(rows.collect::<Result<Vec<_>, _>>()?)
    }

    /// Cached conversion for these source bytes: (converter, markdown path
    /// relative to the conversions dir).
    pub fn get_conversion(
        &self,
        source_sha256: &str,
    ) -> Result<Option<(String, String)>, ColibriError> {
        Ok(self
            .conn
            .query_row(
                "SELECT converter, markdown_path FROM conversions WHERE source_sha256 = ?1",
                [source_sha256],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()?)
    }

    pub fn put_conversion(
        &self,
        source_sha256: &str,
        converter: &str,
        markdown_path: &str,
    ) -> Result<(), ColibriError> {
        self.conn.execute(
            "INSERT INTO conversions (source_sha256, converter, markdown_path, created_at)
             VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(source_sha256) DO UPDATE SET converter = excluded.converter,
                markdown_path = excluded.markdown_path, created_at = excluded.created_at",
            params![source_sha256, converter, markdown_path, now()],
        )?;
        Ok(())
    }
}

enum SchemaState {
    Current,
    Empty,
    Foreign(String),
}

/// Only "this file is not a database" means foreign. Busy, I/O and
/// permission errors are reported as they are, so users are never told to
/// reset a valid database because of a transient problem.
fn classify_open_error(e: rusqlite::Error) -> Result<SchemaState, ColibriError> {
    match e.sqlite_error_code() {
        Some(rusqlite::ErrorCode::NotADatabase) => Ok(SchemaState::Foreign(e.to_string())),
        _ => Err(e.into()),
    }
}

fn now() -> String {
    Utc::now().to_rfc3339()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn sample(doc_id: &str) -> DocumentRecord {
        let mut doc = DocumentRecord::new(doc_id, "notes", "a/b.md");
        doc.title = "B".into();
        doc.doc_type = "note".into();
        doc.content_hash = "sha256:abc".into();
        doc.markdown_path = "notes/abc.md".into();
        doc.source_path = Some("/src/a/b.md".into());
        doc
    }

    #[test]
    fn new_db_gets_current_schema_and_round_trips_documents() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("metadata.db");
        let store = MetadataStore::open_rw(&path).unwrap();
        let doc = sample("notes:a/b.md");
        store.upsert_document(&doc).unwrap();

        let ro = MetadataStore::open_ro(&path).unwrap();
        assert_eq!(ro.get_document("notes:a/b.md").unwrap(), Some(doc.clone()));
        assert_eq!(ro.list_documents().unwrap(), vec![doc]);
    }

    #[test]
    fn upsert_keeps_created_at_and_updates_fields() {
        let dir = TempDir::new().unwrap();
        let store = MetadataStore::open_rw(&dir.path().join("m.db")).unwrap();
        let mut doc = sample("d1");
        doc.created_at = "2020-01-01T00:00:00Z".into();
        store.upsert_document(&doc).unwrap();

        let mut changed = sample("d1");
        changed.title = "Changed".into();
        store.upsert_document(&changed).unwrap();

        let got = store.get_document("d1").unwrap().unwrap();
        assert_eq!(got.title, "Changed");
        assert_eq!(got.created_at, "2020-01-01T00:00:00Z");
    }

    #[test]
    fn index_state_and_status_updates() {
        let dir = TempDir::new().unwrap();
        let store = MetadataStore::open_rw(&dir.path().join("m.db")).unwrap();
        store.upsert_document(&sample("d1")).unwrap();

        store.mark_indexed("d1", "sha256:abc", 7).unwrap();
        let got = store.get_document("d1").unwrap().unwrap();
        assert!(got.is_index_current());
        assert_eq!(got.chunk_count, Some(7));

        store
            .set_status("d1", DocStatus::Removed, Some("pruned"))
            .unwrap();
        store.clear_index_state("d1").unwrap();
        let got = store.get_document("d1").unwrap().unwrap();
        assert!(!got.is_searchable());
        assert_eq!(got.removed_reason.as_deref(), Some("pruned"));
        assert_eq!(got.indexed_hash, None);
    }

    #[test]
    fn get_documents_by_ids_spans_multiple_batches() {
        let dir = TempDir::new().unwrap();
        let store = MetadataStore::open_rw(&dir.path().join("m.db")).unwrap();
        let tx = store.begin().unwrap();
        let ids: Vec<String> = (0..1203).map(|i| format!("d{i}")).collect();
        for id in &ids {
            store.upsert_document(&sample(id)).unwrap();
        }
        tx.commit().unwrap();

        let found = store.get_documents_by_ids(&ids).unwrap();
        assert_eq!(found.len(), 1203);
    }

    // AC-001.1: a non-SQLite file is reported and left byte-identical.
    #[test]
    fn non_sqlite_file_is_rejected_and_untouched() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("metadata.db");
        let bytes = b"definitely not a sqlite database, just some bytes\n".repeat(200);
        std::fs::write(&path, &bytes).unwrap();

        for result in [
            MetadataStore::open_rw(&path).map(|_| ()),
            MetadataStore::open_ro(&path).map(|_| ()),
        ] {
            let err = result.unwrap_err().to_string();
            assert!(err.contains("colibri reset"), "{err}");
        }
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
    }

    // AC-001.2: a pre-v7 database is reported and left byte-identical.
    #[test]
    fn old_schema_is_rejected_and_untouched() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("metadata.db");
        {
            let conn = Connection::open(&path).unwrap();
            conn.execute_batch("CREATE TABLE documents (doc_id TEXT PRIMARY KEY);")
                .unwrap();
        }
        let before = std::fs::read(&path).unwrap();

        let err = MetadataStore::open_rw(&path).err().unwrap().to_string();
        assert!(err.contains("colibri reset"), "{err}");
        assert!(err.contains("pre-v7"), "{err}");
        assert_eq!(std::fs::read(&path).unwrap(), before);

        {
            let conn = Connection::open(&path).unwrap();
            conn.pragma_update(None, "user_version", 99).unwrap();
        }
        let err = MetadataStore::open_rw(&path).err().unwrap().to_string();
        assert!(err.contains("schema v99"), "{err}");
    }

    #[test]
    fn open_ro_requires_existing_db_and_creates_nothing() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("metadata.db");
        assert!(MetadataStore::open_ro(&path).is_err());
        assert!(!path.exists());
    }

    #[test]
    fn only_not_a_database_counts_as_foreign() {
        let err = |code| rusqlite::Error::SqliteFailure(rusqlite::ffi::Error::new(code), None);
        assert!(classify_open_error(err(rusqlite::ffi::SQLITE_BUSY)).is_err());
        assert!(matches!(
            classify_open_error(err(rusqlite::ffi::SQLITE_NOTADB)),
            Ok(SchemaState::Foreign(_))
        ));
    }

    #[test]
    fn collections_problems_and_conversions_round_trip() {
        let dir = TempDir::new().unwrap();
        let store = MetadataStore::open_rw(&dir.path().join("m.db")).unwrap();
        store
            .record_collection_run("vault", "mirror", Some("/pkm"), "ok", "{}")
            .unwrap();
        store
            .record_collection_run("vault", "mirror", Some("/pkm"), "partial", "{}")
            .unwrap();
        let runs = store.list_collection_runs().unwrap();
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].last_run_status.as_deref(), Some("partial"));

        let p = |path: &str| {
            (
                path.to_string(),
                "unreadable".to_string(),
                "denied".to_string(),
            )
        };
        store.replace_problems("vault", &[p("a"), p("b")]).unwrap();
        store.replace_problems("vault", &[p("c")]).unwrap();
        let problems = store.list_problems().unwrap();
        assert_eq!(problems.len(), 1);
        assert_eq!(problems[0].source_path, "c");

        assert_eq!(store.get_conversion("abc").unwrap(), None);
        store.put_conversion("abc", "pandoc", "ab/abc.md").unwrap();
        assert_eq!(
            store.get_conversion("abc").unwrap(),
            Some(("pandoc".into(), "ab/abc.md".into()))
        );

        let mut other = sample("other:x");
        other.collection = "other".into();
        store.upsert_document(&sample("notes:y")).unwrap();
        store.upsert_document(&other).unwrap();
        assert_eq!(store.list_documents_in("other").unwrap().len(), 1);
    }

    // AC-003.2: readers are not blocked by an open write transaction.
    #[test]
    fn read_succeeds_while_write_transaction_is_open() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("metadata.db");
        let writer = MetadataStore::open_rw(&path).unwrap();
        writer.upsert_document(&sample("d1")).unwrap();

        let tx = writer.begin().unwrap();
        writer.upsert_document(&sample("d2")).unwrap();

        let reader = MetadataStore::open_ro(&path).unwrap();
        let docs = reader.list_documents().unwrap();
        assert_eq!(docs.len(), 1, "reader sees the last committed state");

        tx.commit().unwrap();
        assert_eq!(reader.list_documents().unwrap().len(), 2);
    }
}
