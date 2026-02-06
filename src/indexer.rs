//! Index content sources for semantic search.
//!
//! Uses LanceDB Rust SDK and Ollama HTTP API. Supports four per-folder
//! indexing modes: static, incremental, append_only, disabled.
//! Mirrors the Python `indexer.py` module.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

use arrow_array::{
    ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator, StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use lancedb::database::CreateTableMode;
use tracing::{info, warn};

use crate::config::{AppConfig, FolderProfile, IndexMode, SCHEMA_VERSION};
use crate::embedding::embed_texts;
use crate::error::ColibriError;
use crate::index_meta::{read_index_meta, write_index_meta};
use crate::manifest::{
    get_manifest_path, make_key, manifest_signature, source_id_for_root, Manifest,
};
use crate::sources::markdown::MarkdownFolderSource;
use crate::sources::ContentSource;

/// Safety limit for chunk text length (nomic-embed-text context window).
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

/// Split text into overlapping chunks on natural boundaries.
///
/// Tries paragraph boundaries first, then sentence boundaries,
/// and finally hard-breaks at `chunk_size`. Exact port of Python `_split_text()`.
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
        let mut end = start + chunk_size;

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
                // Try sentence boundaries
                end = try_sentence_break(segment, start, chunk_size, end);
            }
        } else {
            // Try sentence boundaries
            end = try_sentence_break(segment, start, chunk_size, end);
        }

        let chunk = text[start..end].trim();
        if !chunk.is_empty() {
            chunks.push(chunk.to_string());
        }

        // Advance with overlap
        let next = end.saturating_sub(chunk_overlap);
        start = std::cmp::max(start + 1, next);
    }

    chunks
}

/// Try to find a sentence boundary in the segment.
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
// Source factory
// ---------------------------------------------------------------------------

/// Create a content source for a folder profile, with nested exclusions.
pub fn create_source_for_profile(
    profile: &FolderProfile,
    all_profiles: &[FolderProfile],
) -> MarkdownFolderSource {
    // Compute nested exclusions: any other profile whose path is under this profile's path
    let exclude_paths: Vec<String> = all_profiles
        .iter()
        .filter(|other| other.path != profile.path && other.path.starts_with(&profile.path))
        .map(|other| other.path.clone())
        .collect();

    MarkdownFolderSource::new(
        &profile.path,
        true,
        profile.name.clone(),
        profile.extensions.clone(),
        exclude_paths,
    )
}

// ---------------------------------------------------------------------------
// Row building
// ---------------------------------------------------------------------------

/// Build chunk rows for a single document (without vectors).
fn build_rows_for_doc(
    source: &dyn ContentSource,
    doc_path: &Path,
    profile: &FolderProfile,
    chunk_size: usize,
    chunk_overlap: usize,
) -> Result<Vec<ChunkRow>, ColibriError> {
    let doc = source.read_document(doc_path)?;

    // Frontmatter type takes precedence; profile doc_type is fallback
    let doc_type = if doc.metadata.contains_key("type") {
        doc.doc_type.clone()
    } else {
        profile.doc_type.clone()
    };

    let tags_str = if doc.tags.is_empty() {
        String::new()
    } else {
        doc.tags.join(",")
    };

    let chunks = split_text(&doc.content, chunk_size, chunk_overlap);
    let mut rows = Vec::with_capacity(chunks.len());

    for mut chunk_text in chunks {
        if chunk_text.len() > MAX_CHUNK_CHARS {
            chunk_text.truncate(MAX_CHUNK_CHARS);
            chunk_text.push_str("...");
        }
        rows.push(ChunkRow {
            text: chunk_text,
            source_file: doc.path.to_string_lossy().into_owned(),
            title: doc.title.clone(),
            doc_type: doc_type.clone(),
            folder: doc.folder.clone(),
            source_name: doc.source_name.clone(),
            source_type: doc.source_type.clone(),
            tags: tags_str.clone(),
        });
    }

    Ok(rows)
}

/// Intermediate chunk row (before embedding).
#[derive(Debug, Clone)]
struct ChunkRow {
    text: String,
    source_file: String,
    title: String,
    doc_type: String,
    folder: String,
    source_name: String,
    source_type: String,
    tags: String,
}

// ---------------------------------------------------------------------------
// Arrow / LanceDB helpers
// ---------------------------------------------------------------------------

/// Build an Arrow schema for the chunks table.
fn chunks_schema(vector_dim: usize) -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("text", DataType::Utf8, false),
        Field::new("source_file", DataType::Utf8, false),
        Field::new("title", DataType::Utf8, false),
        Field::new("doc_type", DataType::Utf8, false),
        Field::new("folder", DataType::Utf8, false),
        Field::new("source_name", DataType::Utf8, false),
        Field::new("source_type", DataType::Utf8, false),
        Field::new("tags", DataType::Utf8, false),
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

/// Convert chunk rows + vectors into an Arrow RecordBatch.
fn rows_to_batch(
    rows: &[ChunkRow],
    vectors: &[Vec<f32>],
    vector_dim: usize,
) -> Result<RecordBatch, ColibriError> {
    let schema = chunks_schema(vector_dim);

    let text_arr: ArrayRef = Arc::new(StringArray::from(
        rows.iter().map(|r| r.text.as_str()).collect::<Vec<_>>(),
    ));
    let source_file_arr: ArrayRef = Arc::new(StringArray::from(
        rows.iter()
            .map(|r| r.source_file.as_str())
            .collect::<Vec<_>>(),
    ));
    let title_arr: ArrayRef = Arc::new(StringArray::from(
        rows.iter().map(|r| r.title.as_str()).collect::<Vec<_>>(),
    ));
    let doc_type_arr: ArrayRef = Arc::new(StringArray::from(
        rows.iter().map(|r| r.doc_type.as_str()).collect::<Vec<_>>(),
    ));
    let folder_arr: ArrayRef = Arc::new(StringArray::from(
        rows.iter().map(|r| r.folder.as_str()).collect::<Vec<_>>(),
    ));
    let source_name_arr: ArrayRef = Arc::new(StringArray::from(
        rows.iter()
            .map(|r| r.source_name.as_str())
            .collect::<Vec<_>>(),
    ));
    let source_type_arr: ArrayRef = Arc::new(StringArray::from(
        rows.iter()
            .map(|r| r.source_type.as_str())
            .collect::<Vec<_>>(),
    ));
    let tags_arr: ArrayRef = Arc::new(StringArray::from(
        rows.iter().map(|r| r.tags.as_str()).collect::<Vec<_>>(),
    ));

    // Build FixedSizeList of Float32 for vectors
    let flat_values: Vec<f32> = vectors.iter().flat_map(|v| v.iter().copied()).collect();
    let values_arr: ArrayRef = Arc::new(Float32Array::from(flat_values));
    let field = Arc::new(Field::new("item", DataType::Float32, true));
    let vector_arr: ArrayRef = Arc::new(
        FixedSizeListArray::try_new(field, vector_dim as i32, values_arr, None)
            .map_err(|e| ColibriError::Index(format!("Failed to build vector array: {e}")))?,
    );

    RecordBatch::try_new(
        schema,
        vec![
            text_arr,
            source_file_arr,
            title_arr,
            doc_type_arr,
            folder_arr,
            source_name_arr,
            source_type_arr,
            tags_arr,
            vector_arr,
        ],
    )
    .map_err(|e| ColibriError::Index(format!("Failed to build RecordBatch: {e}")))
}

// ---------------------------------------------------------------------------
// File classification
// ---------------------------------------------------------------------------

/// Classify files: which need indexing, which are skipped.
fn classify_files(
    source: &dyn ContentSource,
    profile: &FolderProfile,
    manifest: &Manifest,
    force: bool,
    source_id: &str,
) -> (Vec<std::path::PathBuf>, usize) {
    let all_files = source.list_documents();

    if force {
        return (all_files, 0);
    }

    let mut to_index = Vec::new();
    let mut skipped = 0;

    for fp in &all_files {
        let rel = fp.to_string_lossy();
        let key = make_key(source_id, &rel);

        match profile.mode {
            IndexMode::Static | IndexMode::AppendOnly => {
                if manifest.is_file_known(&key) {
                    skipped += 1;
                } else {
                    to_index.push(fp.clone());
                }
            }
            IndexMode::Incremental => {
                let abs_path = source.root_path().join(fp);
                if manifest.is_file_changed(&key, &abs_path) {
                    to_index.push(fp.clone());
                } else {
                    skipped += 1;
                }
            }
            IndexMode::Disabled => {
                skipped += 1;
            }
        }
    }

    (to_index, skipped)
}

// ---------------------------------------------------------------------------
// Deleted-file detection
// ---------------------------------------------------------------------------

/// Remove index chunks and manifest entries for files no longer on disk.
async fn detect_deleted_files(
    manifest: &mut Manifest,
    known_keys: &HashSet<String>,
    current_keys: &HashSet<String>,
    table: &lancedb::Table,
) -> Result<usize, ColibriError> {
    let mut deleted = 0;

    for key in known_keys {
        if !current_keys.contains(key) {
            let (_, rel_path) = crate::manifest::split_key(key);
            let escaped = rel_path.replace('\'', "''");
            table.delete(&format!("source_file = '{escaped}'")).await?;
            manifest.remove_file(key);
            deleted += 1;
        }
    }

    Ok(deleted)
}

// ---------------------------------------------------------------------------
// Per-folder indexing
// ---------------------------------------------------------------------------

/// Index a single folder according to its profile.
#[allow(clippy::too_many_arguments)]
async fn index_folder(
    source: &dyn ContentSource,
    profile: &FolderProfile,
    manifest: &mut Manifest,
    db: &lancedb::Connection,
    table: &mut Option<lancedb::Table>,
    config: &AppConfig,
    chunk_size: usize,
    chunk_overlap: usize,
    force: bool,
    overwrite_first: bool,
) -> Result<IndexResult, ColibriError> {
    let source_label = profile.display_name().to_string();
    let src_id = source_id_for_root(source.root_path());

    let all_files = source.list_documents();
    let all_rel_paths: HashSet<String> = all_files
        .iter()
        .map(|p| p.to_string_lossy().into_owned())
        .collect();

    let (files_to_index, files_skipped) = classify_files(source, profile, manifest, force, &src_id);

    let prefix = format!("{src_id}:");
    let manifest_keys_for_source: HashSet<String> = manifest
        .files
        .keys()
        .filter(|k| k.starts_with(&prefix))
        .cloned()
        .collect();
    let current_keys_for_source: HashSet<String> = all_rel_paths
        .iter()
        .map(|rel| make_key(&src_id, rel))
        .collect();

    if files_to_index.is_empty() {
        // Still detect deletions for incremental mode
        let mut deleted = 0;
        if profile.mode == IndexMode::Incremental && !force {
            if let Some(tbl) = table.as_ref() {
                deleted = detect_deleted_files(
                    manifest,
                    &manifest_keys_for_source,
                    &current_keys_for_source,
                    tbl,
                )
                .await?;
            }
        }
        if files_skipped > 0 || deleted > 0 {
            info!(
                "{source_label}: {files_skipped} unchanged{}",
                if deleted > 0 {
                    format!(", {deleted} removed")
                } else {
                    String::new()
                }
            );
        }
        return Ok(IndexResult {
            files_skipped,
            files_deleted: deleted,
            ..Default::default()
        });
    }

    // Read and chunk all files
    let mut rows: Vec<ChunkRow> = Vec::new();
    let mut files_indexed = 0;
    let mut errors = 0;
    let mut file_chunk_counts: HashMap<String, usize> = HashMap::new();

    for doc_path in &files_to_index {
        match build_rows_for_doc(source, doc_path, profile, chunk_size, chunk_overlap) {
            Ok(doc_rows) => {
                file_chunk_counts.insert(doc_path.to_string_lossy().into_owned(), doc_rows.len());
                rows.extend(doc_rows);
                files_indexed += 1;
            }
            Err(e) => {
                warn!(
                    "Skipping {}: {e}",
                    doc_path.file_name().unwrap_or_default().to_string_lossy()
                );
                errors += 1;
            }
        }
    }

    if rows.is_empty() {
        return Ok(IndexResult {
            files_skipped,
            errors,
            ..Default::default()
        });
    }

    // Embed all chunks
    info!(
        "Embedding {} chunks from {files_indexed} files...",
        rows.len()
    );
    let texts: Vec<String> = rows.iter().map(|r| r.text.clone()).collect();
    let vectors = embed_texts(&texts, &config.embedding_model, &config.ollama_base_url).await?;

    if vectors.is_empty() {
        return Err(ColibriError::Embedding(
            "Ollama returned no embeddings".into(),
        ));
    }

    let vector_dim = vectors[0].len();
    let batch = rows_to_batch(&rows, &vectors, vector_dim)?;
    let schema = chunks_schema(vector_dim);

    // Write to LanceDB
    if overwrite_first && table.is_none() {
        // First folder in a full rebuild — create with overwrite
        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let new_table = db
            .create_table(TABLE_NAME, Box::new(batches))
            .mode(CreateTableMode::Overwrite)
            .execute()
            .await?;
        *table = Some(new_table);
    } else if let Some(tbl) = table.as_ref() {
        // Delete old chunks for files we're re-indexing
        let indexed_files: HashSet<&str> = rows.iter().map(|r| r.source_file.as_str()).collect();
        for sf in &indexed_files {
            let escaped = sf.replace('\'', "''");
            tbl.delete(&format!("source_file = '{escaped}'")).await?;
        }
        // Add new rows
        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);
        tbl.add(Box::new(batches)).execute().await?;
    } else {
        // Table doesn't exist yet — create it
        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let new_table = db
            .create_table(TABLE_NAME, Box::new(batches))
            .execute()
            .await?;
        *table = Some(new_table);
    }

    // Update manifest entries
    for doc_path in &files_to_index {
        let rel = doc_path.to_string_lossy();
        let key = make_key(&src_id, &rel);
        let abs_path = source.root_path().join(doc_path);
        if abs_path.exists() {
            if let Some(&count) = file_chunk_counts.get(rel.as_ref()) {
                manifest.record_file(&key, &abs_path, count)?;
            }
        }
    }

    // Detect deleted files (incremental mode)
    let mut deleted = 0;
    if profile.mode == IndexMode::Incremental && !force {
        if let Some(tbl) = table.as_ref() {
            deleted = detect_deleted_files(
                manifest,
                &manifest_keys_for_source,
                &current_keys_for_source,
                tbl,
            )
            .await?;
        }
    }

    info!(
        "{source_label}: indexed {files_indexed} files ({} chunks)",
        rows.len()
    );

    Ok(IndexResult {
        total_chunks: rows.len(),
        files_indexed,
        files_skipped,
        files_deleted: deleted,
        errors,
    })
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Index the library according to the loaded configuration.
pub async fn index_library(
    config: &AppConfig,
    folder_filter: Option<&str>,
    force: bool,
) -> Result<IndexResult, ColibriError> {
    config.ensure_directories()?;

    // Resolve profiles
    let mut profiles: Vec<&FolderProfile> = if let Some(filter) = folder_filter {
        let matches: Vec<&FolderProfile> = config
            .sources
            .iter()
            .filter(|p| p.display_name() == filter)
            .collect();
        if matches.is_empty() {
            let names: Vec<&str> = config.sources.iter().map(|s| s.display_name()).collect();
            return Err(ColibriError::Config(format!(
                "Unknown source: {filter}. Configured: {}",
                names.join(", ")
            )));
        }
        matches
    } else {
        config
            .sources
            .iter()
            .filter(|p| p.mode != IndexMode::Disabled)
            .collect()
    };

    let mut force = force;

    // Load manifest
    let manifest_path = get_manifest_path(&config.data_dir);
    let mut manifest = Manifest::load(&manifest_path)?;

    // Check schema version — force rebuild if outdated
    let meta = read_index_meta(&config.lancedb_dir)?;
    let stored_version = meta
        .get("schema_version")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;

    if stored_version != SCHEMA_VERSION && !force {
        info!(
            "Index schema outdated (v{stored_version} -> v{SCHEMA_VERSION}). Forcing full rebuild."
        );
        force = true;
        // Re-resolve all non-disabled profiles
        profiles = config
            .sources
            .iter()
            .filter(|p| p.mode != IndexMode::Disabled)
            .collect();
    }

    let full_rebuild = force && folder_filter.is_none();

    // Print header
    let sources_str: Vec<String> = profiles
        .iter()
        .map(|p| format!("{} ({})", p.display_name(), format_mode(p.mode)))
        .collect();
    info!("Indexing sources: {}", sources_str.join(", "));
    if full_rebuild {
        info!("Mode: full rebuild");
    }

    if full_rebuild {
        manifest = Manifest::new();
    }

    // Connect to LanceDB
    std::fs::create_dir_all(&config.lancedb_dir)?;
    let db = lancedb::connect(config.lancedb_dir.to_string_lossy().as_ref())
        .execute()
        .await?;

    let mut table: Option<lancedb::Table> = if full_rebuild {
        None
    } else {
        db.open_table(TABLE_NAME).execute().await.ok()
    };

    let mut aggregate = IndexResult::default();

    for (i, profile) in profiles.iter().enumerate() {
        let source = create_source_for_profile(profile, &config.sources);
        let chunk_size = profile.effective_chunk_size(config.chunk_size);
        let chunk_overlap = profile.effective_chunk_overlap(config.chunk_overlap);

        let result = index_folder(
            &source,
            profile,
            &mut manifest,
            &db,
            &mut table,
            config,
            chunk_size,
            chunk_overlap,
            force,
            full_rebuild && i == 0,
        )
        .await?;

        aggregate.accumulate(&result);
    }

    // Save manifest
    manifest.save(&manifest_path)?;

    // Write index metadata
    let after_sig = manifest_signature(&manifest);
    let total_chunks: usize = after_sig.values().map(|(_, count)| count).sum();

    let mut extra = serde_json::Map::new();
    extra.insert(
        "last_indexed_at".into(),
        serde_json::Value::String(manifest.indexed_at.clone()),
    );
    extra.insert(
        "file_count".into(),
        serde_json::Value::Number(after_sig.len().into()),
    );
    extra.insert(
        "chunk_count".into(),
        serde_json::Value::Number(total_chunks.into()),
    );
    extra.insert(
        "files_indexed_last_run".into(),
        serde_json::Value::Number(aggregate.files_indexed.into()),
    );
    extra.insert(
        "files_skipped_last_run".into(),
        serde_json::Value::Number(aggregate.files_skipped.into()),
    );
    extra.insert(
        "files_deleted_last_run".into(),
        serde_json::Value::Number(aggregate.files_deleted.into()),
    );
    extra.insert(
        "errors_last_run".into(),
        serde_json::Value::Number(aggregate.errors.into()),
    );

    write_index_meta(&config.lancedb_dir, &config.embedding_model, &extra)?;

    // Print summary
    eprintln!(
        "Done: {} files indexed ({} chunks)",
        aggregate.files_indexed, aggregate.total_chunks
    );
    if aggregate.files_skipped > 0 {
        eprintln!("{} unchanged files skipped", aggregate.files_skipped);
    }
    if aggregate.files_deleted > 0 {
        eprintln!(
            "{} deleted files removed from index",
            aggregate.files_deleted
        );
    }
    if aggregate.errors > 0 {
        eprintln!("{} files had errors", aggregate.errors);
    }

    Ok(aggregate)
}

fn format_mode(mode: IndexMode) -> &'static str {
    match mode {
        IndexMode::Static => "static",
        IndexMode::Incremental => "incremental",
        IndexMode::AppendOnly => "append_only",
        IndexMode::Disabled => "disabled",
    }
}
