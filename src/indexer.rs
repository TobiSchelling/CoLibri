//! Index the canonical corpus for semantic search.
//!
//! CoLibri ingests content into a managed canonical markdown store (`COLIBRI_HOME/canonical`)
//! and indexes *only* that store into LanceDB. Direct indexing of arbitrary user folders is not
//! supported.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::{
    ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator, StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use lancedb::database::CreateTableMode;
use tracing::info;

use crate::config::{AppConfig, SCHEMA_VERSION};
use crate::embedding::embed_texts_with_progress;
use crate::error::ColibriError;
use crate::index_meta::{read_index_meta, write_index_meta};
use crate::metadata_store::{DocumentIndexStateRow, DocumentRow, MetadataStore};

// ---------------------------------------------------------------------------
// Progress events
// ---------------------------------------------------------------------------

/// Events emitted during indexing for progress reporting.
#[derive(Debug, Clone)]
pub enum IndexEvent {
    /// A profile is about to be indexed.
    SourceStart { name: String },
    /// File reading progress within the current profile.
    Reading { done: usize, total: usize },
    /// Embedding progress (chunks processed so far).
    Embedding {
        chunks_done: usize,
        total_chunks: usize,
    },
    /// Writing embedded chunks to LanceDB.
    Writing,
    /// A profile completed successfully.
    SourceComplete { name: String, result: IndexResult },
    /// A profile had no changes (all files skipped).
    SourceUnchanged {
        name: String,
        skipped: usize,
        deleted: usize,
    },
    /// A non-fatal warning (e.g. unreadable file).
    Warning { message: String },
}

/// Safety limit for chunk text length (bge-m3 context window ≈ 8192 tokens).
const MAX_CHUNK_CHARS: usize = 16000;

/// LanceDB table name.
const TABLE_NAME: &str = "chunks";

/// Summary of an indexing operation.
#[derive(Debug, Default, Clone)]
pub struct IndexResult {
    pub total_chunks: usize,
    pub files_indexed: usize,
    pub files_skipped: usize,
    pub files_deleted: usize,
    pub errors: usize,
}

impl IndexResult {
    fn accumulate(&mut self, other: &IndexResult) {
        self.total_chunks += other.total_chunks;
        self.files_indexed += other.files_indexed;
        self.files_skipped += other.files_skipped;
        self.files_deleted += other.files_deleted;
        self.errors += other.errors;
    }
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
// Canonical document helpers
// ---------------------------------------------------------------------------

fn read_lossy(path: &Path) -> Result<String, ColibriError> {
    let bytes = std::fs::read(path)?;
    Ok(String::from_utf8_lossy(&bytes).to_string())
}

fn abs_markdown_path(config: &AppConfig, markdown_path: &str) -> PathBuf {
    let rel = PathBuf::from(markdown_path);
    config.canonical_dir.join(rel)
}

// ---------------------------------------------------------------------------
// Per-profile indexing
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct IndexDoc<'a> {
    row: &'a DocumentRow,
    abs_path: PathBuf,
}

#[allow(clippy::too_many_arguments)]
async fn index_profile(
    profile_id: &str,
    embedding_profile: &crate::config::EmbeddingProfile,
    docs: &[IndexDoc<'_>],
    config: &AppConfig,
    metadata_store: &MetadataStore,
    generation_id: &str,
    profile_force: bool,
    profile_full_rebuild: bool,
    docs_to_delete: &[(String, String)],
    index_state: &HashMap<String, DocumentIndexStateRow>,
    on_progress: &(impl Fn(IndexEvent) + Send + Sync),
) -> Result<IndexResult, ColibriError> {
    let name = format!("profile:{profile_id}");
    on_progress(IndexEvent::SourceStart { name: name.clone() });

    let db = lancedb::connect(
        config
            .lancedb_dir_for_profile(profile_id)
            .to_string_lossy()
            .as_ref(),
    )
    .execute()
    .await?;

    let table: Option<lancedb::Table> = if profile_full_rebuild {
        None
    } else {
        db.open_table(TABLE_NAME).execute().await.ok()
    };

    // If we are not rebuilding the table, apply deletions first.
    let mut deleted = 0usize;
    if !profile_full_rebuild && !docs_to_delete.is_empty() {
        if let Some(tbl) = table.as_ref() {
            for (doc_id, markdown_path) in docs_to_delete {
                let escaped = doc_id.replace('\'', "''");
                if let Err(e) = tbl.delete(&format!("doc_id = '{escaped}'")).await {
                    on_progress(IndexEvent::Warning {
                        message: format!("Delete failed for doc {doc_id} ({markdown_path}): {e}"),
                    });
                } else {
                    deleted += 1;
                }
            }
        }
    }

    // We always update index state for deletions, even if the table is being rebuilt.
    for (doc_id, markdown_path) in docs_to_delete {
        metadata_store.upsert_document_index_state(
            doc_id,
            generation_id,
            profile_id,
            "deleted",
            None,
            None,
            Some(markdown_path.as_str()),
        )?;
    }

    // Decide which documents to index.
    let mut to_index: Vec<&IndexDoc<'_>> = Vec::new();
    let mut skipped = 0usize;
    for doc in docs {
        let row = doc.row;
        if row.deleted {
            continue;
        }
        let state = index_state.get(&row.doc_id);
        let already_indexed = state
            .map(|s| s.status.as_str() == "indexed")
            .unwrap_or(false);
        let indexed_hash = state.and_then(|s| s.indexed_content_hash.as_deref());
        let indexed_path = state.and_then(|s| s.indexed_markdown_path.as_deref());

        let changed = profile_force
            || !already_indexed
            || indexed_hash != Some(row.content_hash.as_str())
            || indexed_path != Some(row.markdown_path.as_str());

        if changed {
            to_index.push(doc);
        } else {
            skipped += 1;
        }
    }

    if to_index.is_empty() {
        on_progress(IndexEvent::SourceUnchanged {
            name,
            skipped,
            deleted,
        });
        return Ok(IndexResult {
            total_chunks: 0,
            files_indexed: 0,
            files_skipped: skipped,
            files_deleted: deleted,
            errors: 0,
        });
    }

    // Read + chunk
    let total_files = to_index.len();
    let mut rows = Vec::new();
    let mut file_chunk_counts: HashMap<String, usize> = HashMap::new();
    let mut errors = 0usize;

    for (i, doc) in to_index.iter().enumerate() {
        on_progress(IndexEvent::Reading {
            done: i + 1,
            total: total_files,
        });

        let row = doc.row;
        let content = match read_lossy(&doc.abs_path) {
            Ok(c) => c,
            Err(e) => {
                errors += 1;
                on_progress(IndexEvent::Warning {
                    message: format!("Failed to read {}: {e}", row.markdown_path),
                });
                continue;
            }
        };

        let chunks = split_text(&content, config.chunk_size, config.chunk_overlap);
        file_chunk_counts.insert(row.markdown_path.clone(), chunks.len());

        for mut chunk_text in chunks {
            if chunk_text.len() > MAX_CHUNK_CHARS {
                let safe = floor_char_boundary(&chunk_text, MAX_CHUNK_CHARS);
                chunk_text.truncate(safe);
                chunk_text.push_str("...");
            }
            rows.push(ChunkRow {
                doc_id: row.doc_id.clone(),
                text: chunk_text,
            });
        }
    }

    if rows.is_empty() {
        on_progress(IndexEvent::SourceComplete {
            name,
            result: IndexResult {
                total_chunks: 0,
                files_indexed: 0,
                files_skipped: skipped,
                files_deleted: deleted,
                errors,
            },
        });
        return Ok(IndexResult {
            total_chunks: 0,
            files_indexed: 0,
            files_skipped: skipped,
            files_deleted: deleted,
            errors,
        });
    }

    // Embed
    let total_chunks = rows.len();
    on_progress(IndexEvent::Embedding {
        chunks_done: 0,
        total_chunks,
    });

    let texts: Vec<String> = rows.iter().map(|r| r.text.clone()).collect();
    let vectors = embed_texts_with_progress(
        &texts,
        &embedding_profile.model,
        &embedding_profile.endpoint,
        |done, _total| {
            on_progress(IndexEvent::Embedding {
                chunks_done: done,
                total_chunks,
            });
        },
    )
    .await?;

    if vectors.is_empty() {
        return Err(ColibriError::Embedding(
            "Embedding provider returned no embeddings".into(),
        ));
    }

    let vector_dim = vectors[0].len();
    let batch = rows_to_batch(&rows, &vectors, vector_dim)?;
    let schema = chunks_schema(vector_dim);

    // Write
    on_progress(IndexEvent::Writing);

    let profile_index_dir = config.lancedb_dir_for_profile(profile_id);
    std::fs::create_dir_all(&profile_index_dir)?;

    if profile_full_rebuild {
        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let _new_table = db
            .create_table(TABLE_NAME, Box::new(batches))
            .mode(CreateTableMode::Overwrite)
            .execute()
            .await?;
    } else if let Some(tbl) = table.as_ref() {
        let indexed_docs: HashSet<&str> = rows.iter().map(|r| r.doc_id.as_str()).collect();
        for doc_id in &indexed_docs {
            let escaped = doc_id.replace('\'', "''");
            tbl.delete(&format!("doc_id = '{escaped}'")).await?;
        }
        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);
        tbl.add(Box::new(batches)).execute().await?;
    } else {
        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let _new_table = db
            .create_table(TABLE_NAME, Box::new(batches))
            .execute()
            .await?;
    }

    // Update manifest + index state.
    let mut files_indexed = 0usize;
    for doc in &to_index {
        let row = doc.row;
        if row.deleted {
            continue;
        }
        let Some(chunk_count) = file_chunk_counts.get(&row.markdown_path).copied() else {
            continue;
        };
        metadata_store.upsert_document_index_state(
            &row.doc_id,
            generation_id,
            profile_id,
            "indexed",
            Some(chunk_count as u64),
            Some(row.content_hash.as_str()),
            Some(row.markdown_path.as_str()),
        )?;
        files_indexed += 1;
    }

    let result = IndexResult {
        total_chunks,
        files_indexed,
        files_skipped: skipped,
        files_deleted: deleted,
        errors,
    };

    on_progress(IndexEvent::SourceComplete {
        name,
        result: result.clone(),
    });

    Ok(result)
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Index the managed canonical store into per-profile LanceDB indexes.
pub async fn index_library(
    config: &AppConfig,
    force: bool,
    on_progress: impl Fn(IndexEvent) + Send + Sync,
) -> Result<IndexResult, ColibriError> {
    config.ensure_storage_layout()?;
    let metadata_store = MetadataStore::open(&config.metadata_db_path)?;
    let docs = metadata_store.list_documents()?;

    // Build desired profile for each live doc based on classification routing.
    let mut desired_profile: HashMap<String, String> = HashMap::new();
    let mut doc_by_id: HashMap<String, &DocumentRow> = HashMap::new();
    for row in &docs {
        doc_by_id.insert(row.doc_id.clone(), row);
        if row.deleted {
            continue;
        }
        let pid = config.resolve_embedding_profile_id(&row.classification);
        config.embedding_profile(&pid)?;
        desired_profile.insert(row.doc_id.clone(), pid);
    }

    // Index state for move/deletion reconciliation.
    let index_state_rows =
        metadata_store.list_document_index_state_for_generation(&config.active_generation)?;

    // Per-profile deletions to apply before indexing.
    let mut deletions_by_profile: HashMap<String, Vec<(String, String)>> = HashMap::new();
    for state in &index_state_rows {
        if state.status != "indexed" {
            continue;
        }
        let current = doc_by_id.get(&state.doc_id).copied();
        let should_remove = match current {
            None => true,
            Some(doc) if doc.deleted => true,
            Some(doc) => {
                let desired_profile_id = config.resolve_embedding_profile_id(&doc.classification);
                if desired_profile_id != state.embedding_profile_id {
                    true
                } else {
                    state
                        .indexed_markdown_path
                        .as_deref()
                        .is_some_and(|prev| prev != doc.markdown_path)
                }
            }
        };

        if should_remove {
            let delete_path = state
                .indexed_markdown_path
                .clone()
                .or_else(|| current.map(|r| r.markdown_path.clone()))
                .unwrap_or_default();
            if !delete_path.is_empty() {
                deletions_by_profile
                    .entry(state.embedding_profile_id.clone())
                    .or_default()
                    .push((state.doc_id.clone(), delete_path));
            }
        }
    }

    // Group live docs by embedding profile.
    let mut grouped: BTreeMap<String, Vec<IndexDoc<'_>>> = BTreeMap::new();
    for row in &docs {
        if row.deleted {
            continue;
        }
        let Some(profile_id) = desired_profile.get(&row.doc_id) else {
            continue;
        };
        let abs_path = abs_markdown_path(config, &row.markdown_path);
        grouped
            .entry(profile_id.clone())
            .or_default()
            .push(IndexDoc { row, abs_path });
    }

    let mut index_state_by_profile: HashMap<String, HashMap<String, DocumentIndexStateRow>> =
        HashMap::new();
    for row in &index_state_rows {
        index_state_by_profile
            .entry(row.embedding_profile_id.clone())
            .or_default()
            .insert(row.doc_id.clone(), row.clone());
    }

    let mut aggregate = IndexResult::default();
    let empty_state: HashMap<String, DocumentIndexStateRow> = HashMap::new();

    for (profile_id, profile_docs) in grouped {
        let embedding_profile = config.embedding_profile(&profile_id)?.clone();

        let profile_index_dir = config.lancedb_dir_for_profile(&profile_id);
        std::fs::create_dir_all(&profile_index_dir)?;

        // Decide per-profile rebuild if schema is outdated.
        let mut profile_force = force;
        let meta = read_index_meta(&profile_index_dir)?;
        let stored_version = meta
            .get("schema_version")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;
        if stored_version != SCHEMA_VERSION && !profile_force {
            info!(
                "Index schema outdated for profile '{}' (v{} -> v{}). Forcing rebuild for this profile.",
                profile_id, stored_version, SCHEMA_VERSION
            );
            profile_force = true;
        }
        let profile_full_rebuild = profile_force;

        let docs_to_delete = deletions_by_profile
            .get(&profile_id)
            .cloned()
            .unwrap_or_default();
        let state_for_profile = index_state_by_profile
            .get(&profile_id)
            .unwrap_or(&empty_state);

        let profile_result = match index_profile(
            &profile_id,
            &embedding_profile,
            &profile_docs,
            config,
            &metadata_store,
            &config.active_generation,
            profile_force,
            profile_full_rebuild,
            &docs_to_delete,
            state_for_profile,
            &on_progress,
        )
        .await
        {
            Ok(r) => r,
            Err(e) => return Err(e),
        };

        aggregate.accumulate(&profile_result);

        // Write index meta for this profile.
        let mut profile_extra = serde_json::Map::new();
        profile_extra.insert(
            "embedding_profile_id".into(),
            serde_json::Value::String(profile_id.clone()),
        );
        profile_extra.insert(
            "embedding_provider".into(),
            serde_json::Value::String(embedding_profile.provider.clone()),
        );
        profile_extra.insert(
            "embedding_locality".into(),
            serde_json::Value::String(
                match embedding_profile.locality {
                    crate::config::EmbeddingLocality::Local => "local",
                    crate::config::EmbeddingLocality::Cloud => "cloud",
                }
                .to_string(),
            ),
        );
        profile_extra.insert(
            "files_indexed_last_run".into(),
            serde_json::Value::Number(profile_result.files_indexed.into()),
        );
        profile_extra.insert(
            "file_count".into(),
            serde_json::Value::Number(
                (profile_result.files_indexed + profile_result.files_skipped).into(),
            ),
        );
        profile_extra.insert(
            "chunk_count".into(),
            serde_json::Value::Number(profile_result.total_chunks.into()),
        );
        profile_extra.insert(
            "files_skipped_last_run".into(),
            serde_json::Value::Number(profile_result.files_skipped.into()),
        );
        profile_extra.insert(
            "files_deleted_last_run".into(),
            serde_json::Value::Number(profile_result.files_deleted.into()),
        );
        profile_extra.insert(
            "errors_last_run".into(),
            serde_json::Value::Number(profile_result.errors.into()),
        );
        write_index_meta(&profile_index_dir, &embedding_profile.model, &profile_extra)?;
    }

    Ok(aggregate)
}
