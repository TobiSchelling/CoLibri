//! Index content sources for semantic search.
//!
//! Uses LanceDB Rust SDK and Ollama HTTP API. Supports two per-folder
//! indexing modes: static and incremental.
//! Mirrors the Python `indexer.py` module.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

use arrow_array::{
    ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator, StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use lancedb::database::CreateTableMode;
use tracing::info;

use crate::config::{AppConfig, FolderProfile, IndexMode, PIPELINE_SCHEMA_VERSION, SCHEMA_VERSION};
use crate::embedding::embed_texts_with_progress;
use crate::error::ColibriError;
use crate::index_meta::{read_index_meta, write_index_meta};
use crate::manifest::{get_manifest_path, make_key, source_id_for_root, Manifest};
use crate::metadata_store::MetadataStore;
use crate::sources::markdown::MarkdownFolderSource;
use crate::sources::ContentSource;

// ---------------------------------------------------------------------------
// Progress events
// ---------------------------------------------------------------------------

/// Events emitted during indexing for progress reporting.
///
/// Each call site (CLI, TUI) provides its own handler to render these events
/// appropriately — CLI uses indicatif progress bars, TUI sends them over a
/// channel to update its ratatui gauge.
#[derive(Debug, Clone)]
pub enum IndexEvent {
    /// A source folder is about to be indexed.
    SourceStart { name: String },
    /// File reading progress within the current source.
    Reading { done: usize, total: usize },
    /// Embedding progress (chunks processed so far).
    Embedding {
        chunks_done: usize,
        total_chunks: usize,
    },
    /// Writing embedded chunks to LanceDB.
    Writing,
    /// A source completed successfully.
    SourceComplete { name: String, result: IndexResult },
    /// A source had no changes (all files skipped).
    SourceUnchanged {
        name: String,
        skipped: usize,
        deleted: usize,
    },
    /// A non-fatal warning (e.g. unreadable file).
    Warning { message: String },
    /// Indexing finished (final event, carries the aggregate result or error).
    Done(Result<IndexResult, String>),
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

        // Advance with overlap, ensuring we land on a char boundary
        let next = end.saturating_sub(chunk_overlap);
        start = ceil_char_boundary(text, std::cmp::max(start + 1, next));
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

fn canonical_markdown_path_for_doc(
    config: &AppConfig,
    source_root: &Path,
    rel_path: &Path,
) -> Option<String> {
    let class_root = source_root.strip_prefix(&config.canonical_dir).ok()?;
    let joined = class_root.join(rel_path);
    Some(joined.to_string_lossy().replace('\\', "/"))
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
            IndexMode::Static => {
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
) -> Result<Vec<String>, ColibriError> {
    let mut deleted_rel_paths = Vec::new();

    for key in known_keys {
        if !current_keys.contains(key) {
            let (_, rel_path) = crate::manifest::split_key(key);
            let escaped = rel_path.replace('\'', "''");
            table.delete(&format!("source_file = '{escaped}'")).await?;
            manifest.remove_file(key);
            deleted_rel_paths.push(rel_path.to_string());
        }
    }

    Ok(deleted_rel_paths)
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
    metadata_store: &MetadataStore,
    generation_id: &str,
    embedding_profile_id: &str,
    canonical_doc_ids: &HashMap<String, String>,
    chunk_size: usize,
    chunk_overlap: usize,
    force: bool,
    overwrite_first: bool,
    on_progress: &(impl Fn(IndexEvent) + Send + Sync),
) -> Result<IndexResult, ColibriError> {
    let source_label = profile.display_name().to_string();
    let src_id = source_id_for_root(source.root_path());
    let source_root = source.root_path();

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
        // Detect deletions for all modes
        let mut deleted = 0;
        if !force {
            if let Some(tbl) = table.as_ref() {
                let deleted_rel_paths = detect_deleted_files(
                    manifest,
                    &manifest_keys_for_source,
                    &current_keys_for_source,
                    tbl,
                )
                .await?;
                deleted = deleted_rel_paths.len();
                for rel in deleted_rel_paths {
                    let rel_path = Path::new(&rel);
                    if let Some(canonical_path) =
                        canonical_markdown_path_for_doc(config, source_root, rel_path)
                    {
                        if let Some(doc_id) = canonical_doc_ids.get(&canonical_path) {
                            metadata_store.upsert_document_index_state(
                                doc_id,
                                generation_id,
                                embedding_profile_id,
                                "deleted",
                                None,
                            )?;
                        }
                    }
                }
            }
        }
        if files_skipped > 0 || deleted > 0 {
            on_progress(IndexEvent::SourceUnchanged {
                name: source_label,
                skipped: files_skipped,
                deleted,
            });
        }
        return Ok(IndexResult {
            files_skipped,
            files_deleted: deleted,
            ..Default::default()
        });
    }

    on_progress(IndexEvent::SourceStart {
        name: source_label.clone(),
    });

    // Read and chunk all files
    let mut rows: Vec<ChunkRow> = Vec::new();
    let mut files_indexed = 0;
    let mut errors = 0;
    let mut file_chunk_counts: HashMap<String, usize> = HashMap::new();
    let total_files = files_to_index.len();

    on_progress(IndexEvent::Reading {
        done: 0,
        total: total_files,
    });

    for (i, doc_path) in files_to_index.iter().enumerate() {
        match build_rows_for_doc(source, doc_path, profile, chunk_size, chunk_overlap) {
            Ok(doc_rows) => {
                file_chunk_counts.insert(doc_path.to_string_lossy().into_owned(), doc_rows.len());
                rows.extend(doc_rows);
                files_indexed += 1;
            }
            Err(e) => {
                if let Some(canonical_path) =
                    canonical_markdown_path_for_doc(config, source_root, doc_path)
                {
                    if let Some(doc_id) = canonical_doc_ids.get(&canonical_path) {
                        metadata_store.upsert_document_index_state(
                            doc_id,
                            generation_id,
                            embedding_profile_id,
                            "error",
                            None,
                        )?;
                    }
                }
                on_progress(IndexEvent::Warning {
                    message: format!(
                        "Skipping {}: {e}",
                        doc_path.file_name().unwrap_or_default().to_string_lossy()
                    ),
                });
                errors += 1;
            }
        }
        on_progress(IndexEvent::Reading {
            done: i + 1,
            total: total_files,
        });
    }

    if rows.is_empty() {
        return Ok(IndexResult {
            files_skipped,
            errors,
            ..Default::default()
        });
    }

    // Embed all chunks
    let total_chunks = rows.len();

    on_progress(IndexEvent::Embedding {
        chunks_done: 0,
        total_chunks,
    });

    let texts: Vec<String> = rows.iter().map(|r| r.text.clone()).collect();
    let vectors = embed_texts_with_progress(
        &texts,
        &config.embedding_model,
        &config.ollama_base_url,
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
            "Ollama returned no embeddings".into(),
        ));
    }

    let vector_dim = vectors[0].len();
    let batch = rows_to_batch(&rows, &vectors, vector_dim)?;
    let schema = chunks_schema(vector_dim);

    // Write to LanceDB
    on_progress(IndexEvent::Writing);

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
                if let Some(canonical_path) =
                    canonical_markdown_path_for_doc(config, source_root, doc_path)
                {
                    if let Some(doc_id) = canonical_doc_ids.get(&canonical_path) {
                        metadata_store.upsert_document_index_state(
                            doc_id,
                            generation_id,
                            embedding_profile_id,
                            "indexed",
                            Some(count as u64),
                        )?;
                    }
                }
            }
        }
    }

    // Detect deleted files (all modes)
    let mut deleted = 0;
    if !force {
        if let Some(tbl) = table.as_ref() {
            let deleted_rel_paths = detect_deleted_files(
                manifest,
                &manifest_keys_for_source,
                &current_keys_for_source,
                tbl,
            )
            .await?;
            deleted = deleted_rel_paths.len();
            for rel in deleted_rel_paths {
                let rel_path = Path::new(&rel);
                if let Some(canonical_path) =
                    canonical_markdown_path_for_doc(config, source_root, rel_path)
                {
                    if let Some(doc_id) = canonical_doc_ids.get(&canonical_path) {
                        metadata_store.upsert_document_index_state(
                            doc_id,
                            generation_id,
                            embedding_profile_id,
                            "deleted",
                            None,
                        )?;
                    }
                }
            }
        }
    }

    let result = IndexResult {
        total_chunks: rows.len(),
        files_indexed,
        files_skipped,
        files_deleted: deleted,
        errors,
    };

    on_progress(IndexEvent::SourceComplete {
        name: source_label,
        result: result.clone(),
    });

    Ok(result)
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Index the library according to the loaded configuration.
pub async fn index_library(
    config: &AppConfig,
    folder_filter: Option<&str>,
    force: bool,
    on_progress: impl Fn(IndexEvent) + Send + Sync,
) -> Result<IndexResult, ColibriError> {
    // Indexing should never mutate the global active_generation pointer.
    config.ensure_storage_layout()?;
    let metadata_store = MetadataStore::open(&config.metadata_db_path)?;
    let canonical_doc_ids = metadata_store.list_live_document_paths()?;

    // Resolve source profiles for this run
    let selected_profiles: Vec<&FolderProfile> = if let Some(filter) = folder_filter {
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
        config.sources.iter().collect()
    };

    // Group sources by embedding profile resolved from classification routing.
    let mut grouped_profiles: BTreeMap<String, Vec<&FolderProfile>> = BTreeMap::new();
    for profile in &selected_profiles {
        let embedding_profile_id = config.resolve_embedding_profile_id(&profile.classification);
        config.embedding_profile(&embedding_profile_id)?;
        grouped_profiles
            .entry(embedding_profile_id)
            .or_default()
            .push(*profile);
    }

    // Load manifest
    let manifest_path = get_manifest_path(&config.data_dir);
    let mut manifest = Manifest::load(&manifest_path)?;

    let full_rebuild = force && folder_filter.is_none();

    if full_rebuild {
        manifest = Manifest::new_with_active_generation(&config.active_generation);
    }

    let mut aggregate = IndexResult::default();

    for (embedding_profile_id, profiles) in grouped_profiles {
        let embedding_profile = config.embedding_profile(&embedding_profile_id)?.clone();
        let profile_config = config.with_embedding_profile(&embedding_profile);
        let pipeline_version = serde_json::json!({
            "pipeline_schema_version": PIPELINE_SCHEMA_VERSION,
            "index_schema_version": SCHEMA_VERSION,
            "embedding_profile_id": embedding_profile.id.clone(),
            "embedding_provider": embedding_profile.provider.clone(),
            "embedding_model": embedding_profile.model.clone(),
            "chunk_size_default": config.chunk_size,
            "chunk_overlap_default": config.chunk_overlap
        });
        metadata_store.upsert_generation_status(
            &config.active_generation,
            &embedding_profile_id,
            &pipeline_version,
            "building",
        )?;
        std::fs::create_dir_all(&profile_config.lancedb_dir)?;

        // Check schema version per profile-specific index and decide local rebuild mode.
        let mut profile_force = force;
        let meta = read_index_meta(&profile_config.lancedb_dir)?;
        let stored_version = meta
            .get("schema_version")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;
        if stored_version != SCHEMA_VERSION && !profile_force {
            info!(
                "Index schema outdated for profile '{}' (v{} -> v{}). Forcing rebuild for this profile.",
                embedding_profile_id, stored_version, SCHEMA_VERSION
            );
            profile_force = true;
        }

        let profile_full_rebuild = full_rebuild || (profile_force && folder_filter.is_none());

        let profile_outcome: Result<IndexResult, ColibriError> = async {
            let db = lancedb::connect(profile_config.lancedb_dir.to_string_lossy().as_ref())
                .execute()
                .await?;

            let mut table: Option<lancedb::Table> = if profile_full_rebuild {
                None
            } else {
                db.open_table(TABLE_NAME).execute().await.ok()
            };

            let mut profile_aggregate = IndexResult::default();

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
                    &profile_config,
                    &metadata_store,
                    &config.active_generation,
                    &embedding_profile_id,
                    &canonical_doc_ids,
                    chunk_size,
                    chunk_overlap,
                    profile_force,
                    profile_full_rebuild && i == 0,
                    &on_progress,
                )
                .await?;

                profile_aggregate.accumulate(&result);
                aggregate.accumulate(&result);
            }
            Ok(profile_aggregate)
        }
        .await;

        let profile_aggregate = match profile_outcome {
            Ok(v) => v,
            Err(e) => {
                let _ = metadata_store.upsert_generation_status(
                    &config.active_generation,
                    &embedding_profile_id,
                    &pipeline_version,
                    "error",
                );
                return Err(e);
            }
        };

        let mut profile_extra = serde_json::Map::new();
        profile_extra.insert(
            "embedding_profile_id".into(),
            serde_json::Value::String(embedding_profile_id.clone()),
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
            serde_json::Value::Number(profile_aggregate.files_indexed.into()),
        );
        profile_extra.insert(
            "file_count".into(),
            serde_json::Value::Number(
                (profile_aggregate.files_indexed + profile_aggregate.files_skipped).into(),
            ),
        );
        profile_extra.insert(
            "chunk_count".into(),
            serde_json::Value::Number(profile_aggregate.total_chunks.into()),
        );
        profile_extra.insert(
            "files_skipped_last_run".into(),
            serde_json::Value::Number(profile_aggregate.files_skipped.into()),
        );
        profile_extra.insert(
            "files_deleted_last_run".into(),
            serde_json::Value::Number(profile_aggregate.files_deleted.into()),
        );
        profile_extra.insert(
            "errors_last_run".into(),
            serde_json::Value::Number(profile_aggregate.errors.into()),
        );
        write_index_meta(
            &profile_config.lancedb_dir,
            &embedding_profile.model,
            &profile_extra,
        )?;

        metadata_store.upsert_generation_status(
            &config.active_generation,
            &embedding_profile_id,
            &pipeline_version,
            if profile_aggregate.errors > 0 {
                "error"
            } else {
                "ready"
            },
        )?;
    }

    // Save manifest
    manifest.save(&manifest_path)?;

    Ok(aggregate)
}
