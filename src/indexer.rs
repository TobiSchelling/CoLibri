//! Index the canonical corpus for semantic and keyword search.
//!
//! The metadata DB is the source of truth for what should be searchable;
//! the single LanceDB table `chunks` (doc_id, text, vector) follows it.
//! Each document records the content hash its chunks were built from
//! (`indexed_hash`), so an interrupted run resumes where it stopped.

use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator,
    StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use lancedb::index::Index;
use lancedb::query::{ExecutableQuery, QueryBase, Select};
use lancedb::table::{CompactionOptions, Duration, OptimizeAction};

use crate::config::{AppConfig, SCHEMA_VERSION};
use crate::embedding::Embedder;
use crate::error::ColibriError;
use crate::index_meta::{read_index_meta, write_index_meta};
use crate::metadata_store::{DocumentRecord, MetadataStore};

/// Progress events emitted during indexing.
#[derive(Debug, Clone)]
pub enum IndexEvent {
    /// Work planned for this run.
    Start {
        to_index: usize,
        unchanged: usize,
        removed: usize,
    },
    /// Documents embedded and committed so far.
    Progress {
        docs_done: usize,
        chunks_done: usize,
    },
    /// Rebuilding the keyword index and compacting the table.
    Finalizing,
    /// A non-fatal problem (e.g. unreadable canonical file).
    Warning { message: String },
}

/// Safety limit for chunk text length (bge-m3 context window ≈ 8192 tokens).
const MAX_CHUNK_CHARS: usize = 16000;

/// LanceDB table name.
const TABLE_NAME: &str = "chunks";

/// Doc ids per LanceDB delete predicate.
const DELETE_BATCH: usize = 200;

/// Old table versions younger than this are kept for concurrent readers.
const PRUNE_GRACE_MINUTES: i64 = 10;

/// Options for an indexing run.
#[derive(Debug, Clone)]
pub struct IndexOptions {
    /// Drop the table and re-embed every searchable document.
    pub force: bool,
    /// Embed and commit once this many chunks are pending.
    pub batch_chunks: usize,
}

impl Default for IndexOptions {
    fn default() -> Self {
        Self {
            force: false,
            batch_chunks: 256,
        }
    }
}

/// Summary of an indexing run.
#[derive(Debug, Default, Clone)]
pub struct IndexResult {
    pub total_chunks: usize,
    pub files_indexed: usize,
    pub files_skipped: usize,
    pub files_deleted: usize,
    /// Doc ids found in the index without a searchable document.
    pub orphans_removed: usize,
    pub errors: usize,
}

// ---------------------------------------------------------------------------
// Text chunking
// ---------------------------------------------------------------------------

/// Find the nearest valid UTF-8 char boundary at or before the given byte index.
fn floor_char_boundary(s: &str, mut idx: usize) -> usize {
    if idx >= s.len() {
        return s.len();
    }
    while idx > 0 && !s.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

/// Find the nearest valid UTF-8 char boundary at or after the given byte index.
fn ceil_char_boundary(s: &str, mut idx: usize) -> usize {
    if idx >= s.len() {
        return s.len();
    }
    while idx < s.len() && !s.is_char_boundary(idx) {
        idx += 1;
    }
    idx
}

/// Split text into overlapping chunks on natural boundaries.
///
/// Tries paragraph boundaries first, then sentence boundaries,
/// and finally hard-breaks at `chunk_size`.
pub fn split_text(text: &str, chunk_size: usize, chunk_overlap: usize) -> Vec<String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return vec![];
    }

    if text.len() <= chunk_size {
        return vec![text.to_string()];
    }

    let text_len = text.len();
    let mut chunks = Vec::new();
    let mut start = 0;

    while start < text_len {
        // Ensure end is on a valid char boundary
        let mut end = floor_char_boundary(text, start + chunk_size);

        if end >= text_len {
            let chunk = text[start..].trim();
            if !chunk.is_empty() {
                chunks.push(chunk.to_string());
            }
            break;
        }

        let segment = &text[start..end];

        // Try paragraph boundary (\n\n)
        if let Some(para_break) = segment.rfind("\n\n") {
            if para_break > chunk_size / 4 {
                end = start + para_break + 2; // include \n\n
            } else {
                end = try_sentence_break(segment, start, chunk_size, end);
            }
        } else {
            end = try_sentence_break(segment, start, chunk_size, end);
        }

        let chunk = text[start..end].trim();
        if !chunk.is_empty() {
            chunks.push(chunk.to_string());
        }

        // Advance with overlap, ensuring we land on a char boundary
        let next = end.saturating_sub(chunk_overlap);
        start = ceil_char_boundary(text, std::cmp::max(start + 1, next));
    }

    chunks
}

fn try_sentence_break(segment: &str, start: usize, chunk_size: usize, default_end: usize) -> usize {
    let separators = [". ", ".\n", "? ", "!\n", "! ", "?\n"];
    for sep in &separators {
        if let Some(pos) = segment.rfind(sep) {
            if pos > chunk_size / 4 {
                return start + pos + sep.len();
            }
        }
    }
    default_end
}

// ---------------------------------------------------------------------------
// Arrow / LanceDB helpers
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct ChunkRow {
    doc_id: String,
    text: String,
}

fn chunks_schema(vector_dim: usize) -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("doc_id", DataType::Utf8, false),
        Field::new("text", DataType::Utf8, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                vector_dim as i32,
            ),
            true,
        ),
    ]))
}

fn rows_to_batch(
    rows: &[ChunkRow],
    vectors: &[Vec<f32>],
    vector_dim: usize,
) -> Result<RecordBatch, ColibriError> {
    let schema = chunks_schema(vector_dim);

    let doc_id_arr: ArrayRef = Arc::new(StringArray::from(
        rows.iter().map(|r| r.doc_id.as_str()).collect::<Vec<_>>(),
    ));
    let text_arr: ArrayRef = Arc::new(StringArray::from(
        rows.iter().map(|r| r.text.as_str()).collect::<Vec<_>>(),
    ));

    let flat_values: Vec<f32> = vectors.iter().flat_map(|v| v.iter().copied()).collect();
    let values_arr: ArrayRef = Arc::new(Float32Array::from(flat_values));
    let field = Arc::new(Field::new("item", DataType::Float32, true));
    let vector_arr: ArrayRef = Arc::new(
        FixedSizeListArray::try_new(field, vector_dim as i32, values_arr, None)
            .map_err(|e| ColibriError::Index(format!("Failed to build vector array: {e}")))?,
    );

    RecordBatch::try_new(schema, vec![doc_id_arr, text_arr, vector_arr])
        .map_err(|e| ColibriError::Index(format!("Failed to build RecordBatch: {e}")))
}

// ---------------------------------------------------------------------------
// Index orchestration
// ---------------------------------------------------------------------------

fn read_lossy(path: &Path) -> Result<String, ColibriError> {
    let bytes = std::fs::read(path)?;
    Ok(String::from_utf8_lossy(&bytes).to_string())
}

fn chunk_document(text: &str, config: &AppConfig) -> Vec<String> {
    split_text(text, config.chunk_size, config.chunk_overlap)
        .into_iter()
        .map(|mut chunk| {
            if chunk.len() > MAX_CHUNK_CHARS {
                let safe = floor_char_boundary(&chunk, MAX_CHUNK_CHARS);
                chunk.truncate(safe);
                chunk.push_str("...");
            }
            chunk
        })
        .collect()
}

fn id_list_predicate(ids: &[String]) -> String {
    let quoted: Vec<String> = ids
        .iter()
        .map(|id| format!("'{}'", id.replace('\'', "''")))
        .collect();
    format!("doc_id IN ({})", quoted.join(", "))
}

async fn delete_doc_chunks(table: &lancedb::Table, ids: &[String]) -> Result<(), ColibriError> {
    for batch in ids.chunks(DELETE_BATCH) {
        table.delete(&id_list_predicate(batch)).await?;
    }
    Ok(())
}

/// Distinct doc ids present in the table.
async fn indexed_doc_ids(table: &lancedb::Table) -> Result<HashSet<String>, ColibriError> {
    let batches: Vec<RecordBatch> = table
        .query()
        .select(Select::Columns(vec!["doc_id".into()]))
        .execute()
        .await?
        .try_collect()
        .await?;
    let mut ids = HashSet::new();
    for batch in &batches {
        let Some(col) = batch
            .column_by_name("doc_id")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>())
        else {
            continue;
        };
        for i in 0..col.len() {
            if col.is_valid(i) {
                ids.insert(col.value(i).to_string());
            }
        }
    }
    Ok(ids)
}

/// Documents whose chunks are waiting to be embedded and committed.
#[derive(Default)]
struct PendingBatch {
    rows: Vec<ChunkRow>,
    /// (doc_id, content_hash, chunk_count)
    docs: Vec<(String, String, usize)>,
}

struct Writer<'a, E: Embedder> {
    db: lancedb::Connection,
    table: Option<lancedb::Table>,
    store: &'a MetadataStore,
    embedder: &'a E,
    wrote: bool,
}

impl<E: Embedder> Writer<'_, E> {
    /// Embed the pending chunks, replace the documents' chunks, mark them indexed.
    async fn flush(&mut self, batch: &mut PendingBatch) -> Result<usize, ColibriError> {
        if batch.docs.is_empty() {
            return Ok(0);
        }
        let texts: Vec<String> = batch.rows.iter().map(|r| r.text.clone()).collect();
        let vectors = if texts.is_empty() {
            Vec::new()
        } else {
            self.embedder.embed(&texts).await?
        };
        if vectors.len() != texts.len() {
            return Err(ColibriError::Embedding(format!(
                "Embedding provider returned {} vectors for {} chunks",
                vectors.len(),
                texts.len()
            )));
        }

        let doc_ids: Vec<String> = batch.docs.iter().map(|(id, _, _)| id.clone()).collect();
        if let Some(table) = &self.table {
            delete_doc_chunks(table, &doc_ids).await?;
            self.wrote = true;
        }
        if !batch.rows.is_empty() {
            let dim = vectors[0].len();
            let schema = chunks_schema(dim);
            let record_batch = rows_to_batch(&batch.rows, &vectors, dim)?;
            let reader = RecordBatchIterator::new(vec![Ok(record_batch)], schema);
            match &self.table {
                Some(table) => {
                    table.add(Box::new(reader)).execute().await?;
                }
                None => {
                    let table = self
                        .db
                        .create_table(TABLE_NAME, Box::new(reader))
                        .execute()
                        .await?;
                    self.table = Some(table);
                }
            }
            self.wrote = true;
        }

        let tx = self.store.begin()?;
        for (doc_id, hash, chunk_count) in &batch.docs {
            self.store.mark_indexed(doc_id, hash, *chunk_count)?;
        }
        tx.commit()?;

        let chunks = batch.rows.len();
        *batch = PendingBatch::default();
        Ok(chunks)
    }
}

/// Open the existing table if it can be extended, otherwise start fresh.
///
/// Starting fresh clears all recorded index state *before* dropping the
/// table, and the index metadata is written before anything is embedded, so
/// an interrupted run is resumed by the next one instead of rebuilt again.
async fn prepare_table(
    config: &AppConfig,
    store: &MetadataStore,
    db: &lancedb::Connection,
    force: bool,
    on_progress: &impl Fn(IndexEvent),
) -> Result<(Option<lancedb::Table>, bool), ColibriError> {
    let meta = read_index_meta(&config.index_dir).unwrap_or_else(|e| {
        on_progress(IndexEvent::Warning {
            message: format!("Ignoring unreadable index metadata: {e}"),
        });
        serde_json::Map::new()
    });
    let built_compatibly = meta.get("schema_version").and_then(|v| v.as_u64())
        == Some(SCHEMA_VERSION as u64)
        && meta.get("embedding_model").and_then(|v| v.as_str())
            == Some(config.embedding_model.as_str());
    let table = match db.open_table(TABLE_NAME).execute().await {
        Ok(t) => Some(t),
        Err(lancedb::Error::TableNotFound { .. }) => None,
        Err(e) => return Err(e.into()),
    };

    if table.is_some() && built_compatibly && !force {
        return Ok((table, false));
    }
    store.clear_all_index_state()?;
    if table.is_some() {
        db.drop_table(TABLE_NAME, &[]).await?;
    }
    write_index_meta(
        &config.index_dir,
        &config.embedding_model,
        &serde_json::Map::new(),
    )?;
    Ok((None, true))
}

async fn has_keyword_index(table: &lancedb::Table) -> bool {
    table
        .list_indices()
        .await
        .map(|indices| {
            indices.iter().any(|i| {
                matches!(i.index_type, lancedb::index::IndexType::FTS)
                    && i.columns.iter().any(|c| c == "text")
            })
        })
        .unwrap_or(false)
}

/// Bring the index in line with the metadata DB: drop chunks of documents
/// that are no longer searchable or unknown, and embed documents whose
/// content changed since they were indexed.
pub async fn index_library<E: Embedder>(
    config: &AppConfig,
    store: &MetadataStore,
    embedder: &E,
    opts: &IndexOptions,
    on_progress: impl Fn(IndexEvent),
) -> Result<IndexResult, ColibriError> {
    std::fs::create_dir_all(&config.index_dir)?;
    let db = lancedb::connect(config.index_dir.to_string_lossy().as_ref())
        .execute()
        .await?;

    let (table, rebuild) = prepare_table(config, store, &db, opts.force, &on_progress).await?;

    let docs = store.list_documents()?;
    let mut result = IndexResult::default();

    // 1. Chunks of documents that are no longer searchable.
    let stale: Vec<String> = docs
        .iter()
        .filter(|d| !d.is_searchable() && d.indexed_hash.is_some())
        .map(|d| d.doc_id.clone())
        .collect();
    if !stale.is_empty() {
        if let Some(t) = &table {
            delete_doc_chunks(t, &stale).await?;
        }
        let tx = store.begin()?;
        for id in &stale {
            store.clear_index_state(id)?;
        }
        tx.commit()?;
    }
    result.files_deleted = stale.len();

    // 2. Searchable documents whose chunks are missing or outdated.
    let to_index: Vec<&DocumentRecord> = docs
        .iter()
        .filter(|d| d.is_searchable() && !d.is_index_current())
        .collect();
    result.files_skipped = docs
        .iter()
        .filter(|d| d.is_searchable())
        .count()
        .saturating_sub(to_index.len());
    on_progress(IndexEvent::Start {
        to_index: to_index.len(),
        unchanged: result.files_skipped,
        removed: result.files_deleted,
    });

    let mut writer = Writer {
        db: db.clone(),
        table,
        store,
        embedder,
        wrote: !stale.is_empty(),
    };
    let embed_outcome = embed_documents(
        config,
        &to_index,
        &mut writer,
        opts,
        &mut result,
        &on_progress,
    )
    .await;

    // 3. Orphans: chunks whose doc_id has no searchable, indexed document.
    if embed_outcome.is_ok() {
        if let Some(t) = &writer.table {
            let live: HashSet<String> = store
                .list_documents()?
                .into_iter()
                .filter(|d| d.is_searchable() && d.indexed_hash.is_some())
                .map(|d| d.doc_id)
                .collect();
            let orphans: Vec<String> = indexed_doc_ids(t)
                .await?
                .into_iter()
                .filter(|id| !live.contains(id))
                .collect();
            if !orphans.is_empty() {
                delete_doc_chunks(t, &orphans).await?;
                writer.wrote = true;
            }
            result.orphans_removed = orphans.len();
        }
    }

    // 4. Keyword index and compaction, also after a failed run so that what
    //    was committed stays searchable.
    if let Some(t) = &writer.table {
        if writer.wrote || rebuild || !has_keyword_index(t).await {
            on_progress(IndexEvent::Finalizing);
            if let Err(e) = t
                .create_index(&["text"], Index::FTS(Default::default()))
                .replace(true)
                .execute()
                .await
            {
                on_progress(IndexEvent::Warning {
                    message: format!("Keyword index creation failed: {e}"),
                });
            }
            compact(t, &on_progress).await;
        }
        let mut extra = serde_json::Map::new();
        extra.insert(
            "chunk_count".into(),
            serde_json::Value::from(t.count_rows(None).await?),
        );
        write_index_meta(&config.index_dir, &config.embedding_model, &extra)?;
    }

    embed_outcome?;
    Ok(result)
}

async fn embed_documents<E: Embedder>(
    config: &AppConfig,
    to_index: &[&DocumentRecord],
    writer: &mut Writer<'_, E>,
    opts: &IndexOptions,
    result: &mut IndexResult,
    on_progress: &impl Fn(IndexEvent),
) -> Result<(), ColibriError> {
    let mut batch = PendingBatch::default();
    let mut docs_done = 0usize;
    for doc in to_index {
        let path = config.canonical_dir.join(&doc.markdown_path);
        let content = match read_lossy(&path) {
            Ok(c) => c,
            Err(e) => {
                result.errors += 1;
                on_progress(IndexEvent::Warning {
                    message: format!("Failed to read {} ({}): {e}", doc.doc_id, path.display()),
                });
                continue;
            }
        };
        let chunks = chunk_document(&content, config);
        batch
            .docs
            .push((doc.doc_id.clone(), doc.content_hash.clone(), chunks.len()));
        batch.rows.extend(chunks.into_iter().map(|text| ChunkRow {
            doc_id: doc.doc_id.clone(),
            text,
        }));

        if batch.rows.len() >= opts.batch_chunks.max(1) {
            let n = batch.docs.len();
            result.total_chunks += writer.flush(&mut batch).await?;
            result.files_indexed += n;
            docs_done += n;
            on_progress(IndexEvent::Progress {
                docs_done,
                chunks_done: result.total_chunks,
            });
        }
    }
    let n = batch.docs.len();
    result.total_chunks += writer.flush(&mut batch).await?;
    result.files_indexed += n;
    if n > 0 {
        on_progress(IndexEvent::Progress {
            docs_done: docs_done + n,
            chunks_done: result.total_chunks,
        });
    }
    Ok(())
}

/// Merge small data files and drop table versions older than
/// [`PRUNE_GRACE_MINUTES`]. The grace window keeps the files that a search
/// running concurrently in `colibri serve` may still be reading.
async fn compact(table: &lancedb::Table, on_progress: &impl Fn(IndexEvent)) {
    let steps = [
        OptimizeAction::Compact {
            options: CompactionOptions::default(),
            remap_options: None,
        },
        OptimizeAction::Prune {
            older_than: Some(Duration::minutes(PRUNE_GRACE_MINUTES)),
            delete_unverified: Some(true),
            error_if_tagged_old_versions: Some(false),
        },
    ];
    for step in steps {
        if let Err(e) = table.optimize(step).await {
            on_progress(IndexEvent::Warning {
                message: format!("Index compaction step failed: {e}"),
            });
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::metadata_store::DocStatus;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Deterministic 4-dim vectors; optionally fails on the n-th call.
    pub(crate) struct FakeEmbedder {
        pub calls: AtomicUsize,
        pub fail_on_call: Option<usize>,
    }

    impl FakeEmbedder {
        pub(crate) fn new() -> Self {
            Self {
                calls: AtomicUsize::new(0),
                fail_on_call: None,
            }
        }
    }

    impl Embedder for FakeEmbedder {
        fn embed(
            &self,
            texts: &[String],
        ) -> impl std::future::Future<Output = Result<Vec<Vec<f32>>, ColibriError>> + Send {
            let call = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
            let fail = self.fail_on_call == Some(call);
            let out: Vec<Vec<f32>> = texts
                .iter()
                .map(|t| vec![t.len() as f32, 1.0, 0.5, 0.25])
                .collect();
            async move {
                if fail {
                    Err(ColibriError::Embedding("fake failure".into()))
                } else {
                    Ok(out)
                }
            }
        }
    }

    fn add_doc(config: &AppConfig, store: &MetadataStore, id: &str, text: &str) {
        let rel = format!("t/{id}.md");
        let path = config.canonical_dir.join(&rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, text).unwrap();
        let mut doc = DocumentRecord::new(id, "t", id);
        doc.title = id.into();
        doc.doc_type = "note".into();
        doc.content_hash = crate::envelope::content_hash(text);
        doc.markdown_path = rel;
        store.upsert_document(&doc).unwrap();
    }

    async fn table_ids(config: &AppConfig) -> HashSet<String> {
        let db = lancedb::connect(config.index_dir.to_string_lossy().as_ref())
            .execute()
            .await
            .unwrap();
        let t = db.open_table(TABLE_NAME).execute().await.unwrap();
        indexed_doc_ids(&t).await.unwrap()
    }

    fn opts_per_doc() -> IndexOptions {
        IndexOptions {
            force: false,
            batch_chunks: 1,
        }
    }

    #[tokio::test]
    async fn indexes_changed_docs_only_and_removes_unsearchable() {
        let dir = tempfile::TempDir::new().unwrap();
        let config = AppConfig::for_test(dir.path());
        let (_lock, store) = config.open_for_write().unwrap();
        add_doc(&config, &store, "a", "alpha text");
        add_doc(&config, &store, "b", "beta text");
        let fake = FakeEmbedder::new();

        let r = index_library(&config, &store, &fake, &opts_per_doc(), |_| {})
            .await
            .unwrap();
        assert_eq!((r.files_indexed, r.files_skipped), (2, 0));
        assert_eq!(table_ids(&config).await.len(), 2);

        let r = index_library(&config, &store, &fake, &opts_per_doc(), |_| {})
            .await
            .unwrap();
        assert_eq!((r.files_indexed, r.files_skipped), (0, 2));

        store
            .set_status("b", DocStatus::Removed, Some("user"))
            .unwrap();
        let r = index_library(&config, &store, &fake, &opts_per_doc(), |_| {})
            .await
            .unwrap();
        assert_eq!(r.files_deleted, 1);
        assert_eq!(table_ids(&config).await, HashSet::from(["a".to_string()]));
        assert_eq!(store.get_document("b").unwrap().unwrap().indexed_hash, None);
    }

    // AC-005.1: a failure on document 3 keeps documents 1-2; the rerun embeds 3-5 only.
    #[tokio::test]
    async fn interrupted_run_resumes_with_remaining_documents() {
        let dir = tempfile::TempDir::new().unwrap();
        let config = AppConfig::for_test(dir.path());
        let (_lock, store) = config.open_for_write().unwrap();
        for id in ["d1", "d2", "d3", "d4", "d5"] {
            add_doc(&config, &store, id, &format!("text of {id}"));
        }

        let failing = FakeEmbedder {
            calls: AtomicUsize::new(0),
            fail_on_call: Some(3),
        };
        assert!(
            index_library(&config, &store, &failing, &opts_per_doc(), |_| {})
                .await
                .is_err()
        );
        let indexed: Vec<String> = store
            .list_documents()
            .unwrap()
            .into_iter()
            .filter(|d| d.is_index_current())
            .map(|d| d.doc_id)
            .collect();
        assert_eq!(indexed, vec!["d1", "d2"]);

        let ok = FakeEmbedder::new();
        let r = index_library(&config, &store, &ok, &opts_per_doc(), |_| {})
            .await
            .unwrap();
        assert_eq!((r.files_indexed, r.files_skipped), (3, 2));
        assert_eq!(ok.calls.load(Ordering::SeqCst), 3);
    }

    // AC-006.1: chunks of unknown doc ids are purged; live ones stay.
    #[tokio::test]
    async fn orphan_chunks_are_purged() {
        let dir = tempfile::TempDir::new().unwrap();
        let config = AppConfig::for_test(dir.path());
        let (_lock, store) = config.open_for_write().unwrap();
        add_doc(&config, &store, "keep", "kept text");
        add_doc(&config, &store, "ghost", "ghost text");
        let fake = FakeEmbedder::new();
        index_library(&config, &store, &fake, &opts_per_doc(), |_| {})
            .await
            .unwrap();

        // Simulate lost metadata: the row disappears, its chunks stay.
        let conn = rusqlite::Connection::open(&config.metadata_db_path).unwrap();
        conn.execute("DELETE FROM documents WHERE doc_id = 'ghost'", [])
            .unwrap();
        drop(conn);

        let r = index_library(&config, &store, &fake, &opts_per_doc(), |_| {})
            .await
            .unwrap();
        assert_eq!(r.orphans_removed, 1);
        assert_eq!(
            table_ids(&config).await,
            HashSet::from(["keep".to_string()])
        );
    }

    #[tokio::test]
    async fn model_change_triggers_full_rebuild() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut config = AppConfig::for_test(dir.path());
        let (_lock, store) = config.open_for_write().unwrap();
        add_doc(&config, &store, "a", "alpha");
        let fake = FakeEmbedder::new();
        index_library(&config, &store, &fake, &opts_per_doc(), |_| {})
            .await
            .unwrap();

        config.embedding_model = "other".into();
        let r = index_library(&config, &store, &fake, &opts_per_doc(), |_| {})
            .await
            .unwrap();
        assert_eq!(r.files_indexed, 1);
        let meta = read_index_meta(&config.index_dir).unwrap();
        assert_eq!(
            meta.get("embedding_model").and_then(|v| v.as_str()),
            Some("other")
        );
    }

    // Review fix: a run killed after its first commits must be resumed, not
    // rebuilt, so the metadata has to exist before anything is embedded.
    #[tokio::test]
    async fn index_meta_is_written_before_embedding() {
        let dir = tempfile::TempDir::new().unwrap();
        let config = AppConfig::for_test(dir.path());
        let (_lock, store) = config.open_for_write().unwrap();
        add_doc(&config, &store, "a", "alpha");
        let failing = FakeEmbedder {
            calls: AtomicUsize::new(0),
            fail_on_call: Some(1),
        };
        assert!(
            index_library(&config, &store, &failing, &opts_per_doc(), |_| {})
                .await
                .is_err()
        );
        let meta = read_index_meta(&config.index_dir).unwrap();
        assert_eq!(
            meta.get("embedding_model").and_then(|v| v.as_str()),
            Some("fake")
        );
        assert_eq!(
            meta.get("schema_version").and_then(|v| v.as_u64()),
            Some(SCHEMA_VERSION as u64)
        );
    }

    #[tokio::test]
    async fn missing_table_is_rebuilt_even_if_documents_claim_indexed() {
        let dir = tempfile::TempDir::new().unwrap();
        let config = AppConfig::for_test(dir.path());
        let (_lock, store) = config.open_for_write().unwrap();
        add_doc(&config, &store, "a", "alpha");
        add_doc(&config, &store, "b", "beta");
        let fake = FakeEmbedder::new();
        index_library(&config, &store, &fake, &opts_per_doc(), |_| {})
            .await
            .unwrap();

        std::fs::remove_dir_all(&config.index_dir).unwrap();
        let r = index_library(&config, &store, &fake, &opts_per_doc(), |_| {})
            .await
            .unwrap();
        assert_eq!(r.files_indexed, 2);
        assert_eq!(table_ids(&config).await.len(), 2);
    }

    #[tokio::test]
    async fn missing_keyword_index_is_repaired_without_changes() {
        let dir = tempfile::TempDir::new().unwrap();
        let config = AppConfig::for_test(dir.path());
        let (_lock, store) = config.open_for_write().unwrap();
        add_doc(&config, &store, "a", "alpha");
        let fake = FakeEmbedder::new();
        index_library(&config, &store, &fake, &opts_per_doc(), |_| {})
            .await
            .unwrap();

        let db = lancedb::connect(config.index_dir.to_string_lossy().as_ref())
            .execute()
            .await
            .unwrap();
        let t = db.open_table(TABLE_NAME).execute().await.unwrap();
        for idx in t.list_indices().await.unwrap() {
            t.drop_index(&idx.name).await.unwrap();
        }
        assert!(!has_keyword_index(&t).await);

        let r = index_library(&config, &store, &fake, &opts_per_doc(), |_| {})
            .await
            .unwrap();
        assert_eq!(r.files_indexed, 0);
        t.checkout_latest().await.unwrap();
        assert!(has_keyword_index(&t).await);
    }

    #[test]
    fn id_predicate_escapes_quotes() {
        assert_eq!(
            id_list_predicate(&["a".into(), "it's".into()]),
            "doc_id IN ('a', 'it''s')"
        );
    }
}
