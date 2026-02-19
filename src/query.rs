//! Query engine for semantic search over content sources.
//!
//! Mirrors the Python `query.py` module. Uses LanceDB vector search
//! with L2 distance converted to similarity score via `exp(-distance)`.

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::path::Path;

use arrow_array::{Float32Array, RecordBatch, StringArray};
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase, Select};
use serde::Serialize;
use tracing::warn;

use crate::config::{AppConfig, SCHEMA_VERSION};
use crate::embedding::embed_texts;
use crate::error::ColibriError;
use crate::index_meta::read_index_meta;
use crate::metadata_store::MetadataStore;

/// A single search result.
#[derive(Debug, Clone, Serialize)]
pub struct SearchResult {
    pub text: String,
    pub file: String,
    pub title: String,
    #[serde(rename = "type")]
    pub doc_type: String,
    pub folder: String,
    pub score: f64,
}

/// Book entry from the index.
#[derive(Debug, Clone, Serialize)]
pub struct BookEntry {
    pub title: String,
    pub chunks: usize,
    pub file: String,
}

/// Topic entry with document count.
#[derive(Debug, Clone, Serialize)]
pub struct TopicEntry {
    pub tag: String,
    pub document_count: usize,
}

/// Search engine backed by LanceDB.
pub struct SearchEngine {
    backends: Vec<ProfileBackend>,
    config: AppConfig,
}

struct ProfileBackend {
    profile_id: String,
    embedding_model: String,
    embedding_endpoint: String,
    table: lancedb::Table,
}

impl SearchEngine {
    /// Create a new search engine, verifying schema version.
    pub async fn new(config: &AppConfig) -> Result<Self, ColibriError> {
        let mut backends = Vec::new();
        let generation_status = load_generation_status(config)?;
        let mut profile_ids: Vec<String> = config.embedding_profiles.keys().cloned().collect();
        profile_ids.sort();

        for profile_id in profile_ids {
            let profile = config.embedding_profile(&profile_id)?;
            if !profile_enabled_for_active_generation(&profile_id, &generation_status) {
                continue;
            }
            let lancedb_dir = config.lancedb_dir_for_profile(&profile_id);
            let meta = read_index_meta(&lancedb_dir)?;

            if meta.is_empty() {
                continue;
            }

            let stored_version = meta
                .get("schema_version")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32;
            if stored_version != SCHEMA_VERSION {
                return Err(ColibriError::Query(format!(
                    "Index schema outdated for profile '{profile_id}' (v{stored_version}, need v{SCHEMA_VERSION}). \
                     Run `colibri index --force` to rebuild."
                )));
            }
            validate_embedding_model_alignment(config, &profile_id, &meta, &profile.model)?;

            let db = match lancedb::connect(lancedb_dir.to_string_lossy().as_ref())
                .execute()
                .await
            {
                Ok(db) => db,
                Err(e) => {
                    warn!("Skipping profile '{profile_id}' (connect failed): {e}");
                    continue;
                }
            };

            let table = match db.open_table("chunks").execute().await {
                Ok(table) => table,
                Err(e) => {
                    warn!("Skipping profile '{profile_id}' (open table failed): {e}");
                    continue;
                }
            };

            backends.push(ProfileBackend {
                profile_id,
                embedding_model: profile.model.clone(),
                embedding_endpoint: profile.endpoint.clone(),
                table,
            });
        }

        if backends.is_empty() {
            return Err(ColibriError::Query(
                "No indexed embedding profiles found. Run `colibri index` first.".into(),
            ));
        }

        Ok(Self {
            backends,
            config: config.clone(),
        })
    }

    /// Semantic search with optional folder/doc_type filters.
    pub async fn search(
        &self,
        query: &str,
        folder: Option<&str>,
        doc_type: Option<&str>,
        limit: usize,
    ) -> Result<Vec<SearchResult>, ColibriError> {
        let mut merged = Vec::new();
        let mut succeeded = 0usize;

        for backend in &self.backends {
            let query_vector = match embed_texts(
                &[query.to_string()],
                &backend.embedding_model,
                &backend.embedding_endpoint,
            )
            .await
            {
                Ok(v) if !v.is_empty() => v,
                Ok(_) => {
                    warn!(
                        "Skipping profile '{}' for query: embedding returned empty vector",
                        backend.profile_id
                    );
                    continue;
                }
                Err(e) => {
                    warn!(
                        "Skipping profile '{}' for query embed error: {e}",
                        backend.profile_id
                    );
                    continue;
                }
            };

            let mut search = backend
                .table
                .vector_search(query_vector[0].clone())
                .map_err(|e| ColibriError::Query(format!("Vector search failed: {e}")))?
                .limit(self.config.top_k);

            if let Some(folder) = folder {
                let escaped = folder.replace('\'', "''");
                search = search.only_if(format!(
                    "folder = '{escaped}' OR source_file LIKE '{escaped}/%'"
                ));
            }
            if let Some(dt) = doc_type {
                let escaped = dt.replace('\'', "''");
                search = search.only_if(format!("doc_type = '{escaped}'"));
            }

            let raw_results = match search.execute().await {
                Ok(r) => r,
                Err(e) => {
                    warn!(
                        "Skipping profile '{}' for query execution error: {e}",
                        backend.profile_id
                    );
                    continue;
                }
            };

            let batches: Vec<RecordBatch> = match raw_results.try_collect().await {
                Ok(b) => b,
                Err(e) => {
                    warn!(
                        "Skipping profile '{}' while collecting results: {e}",
                        backend.profile_id
                    );
                    continue;
                }
            };

            succeeded += 1;
            collect_search_results(&batches, self.config.similarity_threshold, &mut merged);
        }

        if succeeded == 0 {
            return Err(ColibriError::Query(
                "No searchable embedding profile is currently available.".into(),
            ));
        }

        merged.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));

        let mut deduped = Vec::new();
        let mut seen = HashSet::new();
        for result in merged {
            let key = format!("{}:{}", result.file, result.text);
            if seen.insert(key) {
                deduped.push(result);
            }
            if deduped.len() >= limit {
                break;
            }
        }

        Ok(deduped)
    }

    /// Search only books.
    pub async fn search_books(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<SearchResult>, ColibriError> {
        self.search(query, None, Some("book"), limit).await
    }

    /// Search the entire library.
    pub async fn search_library(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<SearchResult>, ColibriError> {
        self.search(query, None, None, limit).await
    }

    /// List all indexed books with metadata.
    pub async fn list_books(&self) -> Result<Vec<BookEntry>, ColibriError> {
        let mut book_map: HashMap<String, (usize, String)> = HashMap::new();
        let mut succeeded = 0usize;

        for backend in &self.backends {
            let results = match backend
                .table
                .query()
                .only_if("doc_type = 'book'")
                .select(Select::columns(&["title", "source_file"]))
                .execute()
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    warn!(
                        "Skipping profile '{}' for list_books query error: {e}",
                        backend.profile_id
                    );
                    continue;
                }
            };

            let batches: Vec<RecordBatch> = match results.try_collect().await {
                Ok(b) => b,
                Err(e) => {
                    warn!(
                        "Skipping profile '{}' while collecting list_books: {e}",
                        backend.profile_id
                    );
                    continue;
                }
            };
            succeeded += 1;

            for batch in &batches {
                let title_col = batch
                    .column_by_name("title")
                    .and_then(|c| c.as_any().downcast_ref::<StringArray>());
                let file_col = batch
                    .column_by_name("source_file")
                    .and_then(|c| c.as_any().downcast_ref::<StringArray>());

                for i in 0..batch.num_rows() {
                    let title = title_col
                        .map(|c| c.value(i).to_string())
                        .unwrap_or_default();
                    let file = file_col.map(|c| c.value(i).to_string()).unwrap_or_default();
                    let entry = book_map.entry(title).or_insert((0, file));
                    entry.0 += 1;
                }
            }
        }

        if succeeded == 0 {
            return Err(ColibriError::Query(
                "No searchable embedding profile is currently available.".into(),
            ));
        }

        let mut books: Vec<BookEntry> = book_map
            .into_iter()
            .map(|(title, (chunks, file))| BookEntry {
                title,
                chunks,
                file,
            })
            .collect();

        books.sort_by(|a, b| a.title.cmp(&b.title));
        Ok(books)
    }

    /// List all topics (tags) with document counts.
    pub async fn browse_topics(
        &self,
        folder: Option<&str>,
    ) -> Result<Vec<TopicEntry>, ColibriError> {
        // Deduplicate by source_file, then count tags
        let mut seen_files: HashSet<String> = HashSet::new();
        let mut tag_counter: HashMap<String, usize> = HashMap::new();
        let mut succeeded = 0usize;

        for backend in &self.backends {
            let query = if let Some(folder) = folder {
                let escaped = folder.replace('\'', "''");
                backend
                    .table
                    .query()
                    .only_if(format!("folder = '{escaped}'"))
                    .select(Select::columns(&["source_file", "tags"]))
                    .execute()
                    .await
            } else {
                backend
                    .table
                    .query()
                    .select(Select::columns(&["source_file", "tags"]))
                    .execute()
                    .await
            };

            let results = match query {
                Ok(r) => r,
                Err(e) => {
                    warn!(
                        "Skipping profile '{}' for browse_topics query error: {e}",
                        backend.profile_id
                    );
                    continue;
                }
            };

            let batches: Vec<RecordBatch> = match results.try_collect().await {
                Ok(b) => b,
                Err(e) => {
                    warn!(
                        "Skipping profile '{}' while collecting browse_topics: {e}",
                        backend.profile_id
                    );
                    continue;
                }
            };
            succeeded += 1;

            for batch in &batches {
                let file_col = batch
                    .column_by_name("source_file")
                    .and_then(|c| c.as_any().downcast_ref::<StringArray>());
                let tags_col = batch
                    .column_by_name("tags")
                    .and_then(|c| c.as_any().downcast_ref::<StringArray>());

                for i in 0..batch.num_rows() {
                    let file = file_col.map(|c| c.value(i).to_string()).unwrap_or_default();
                    if !seen_files.insert(file) {
                        continue; // already counted this file
                    }
                    let tags_str = tags_col.map(|c| c.value(i).to_string()).unwrap_or_default();
                    if tags_str.is_empty() {
                        continue;
                    }
                    for tag in tags_str.split(',') {
                        let tag = tag.trim();
                        if !tag.is_empty() {
                            *tag_counter.entry(tag.to_string()).or_default() += 1;
                        }
                    }
                }
            }
        }

        if succeeded == 0 {
            return Err(ColibriError::Query(
                "No searchable embedding profile is currently available.".into(),
            ));
        }

        let mut topics: Vec<TopicEntry> = tag_counter
            .into_iter()
            .map(|(tag, count)| TopicEntry {
                tag,
                document_count: count,
            })
            .collect();

        // Sort by count descending
        topics.sort_by(|a, b| b.document_count.cmp(&a.document_count));
        Ok(topics)
    }
}

fn load_generation_status(config: &AppConfig) -> Result<HashMap<String, String>, ColibriError> {
    if !config.metadata_db_path.exists() {
        return Ok(HashMap::new());
    }
    let store = MetadataStore::open(&config.metadata_db_path)?;
    let rows = store.list_generation_entries()?;
    let mut out = HashMap::new();
    for row in rows {
        if row.generation_id == config.active_generation {
            out.insert(row.embedding_profile_id, row.status);
        }
    }
    Ok(out)
}

fn profile_enabled_for_active_generation(
    profile_id: &str,
    generation_status: &HashMap<String, String>,
) -> bool {
    if generation_status.is_empty() {
        return true;
    }
    match generation_status.get(profile_id) {
        Some(status) if status == "ready" => true,
        Some(status) => {
            warn!(
                "Skipping profile '{}' for active generation (status={})",
                profile_id, status
            );
            false
        }
        None => {
            warn!(
                "Skipping profile '{}' for active generation (no generation metadata row)",
                profile_id
            );
            false
        }
    }
}

fn validate_embedding_model_alignment(
    config: &AppConfig,
    profile_id: &str,
    meta: &serde_json::Map<String, serde_json::Value>,
    expected_model: &str,
) -> Result<(), ColibriError> {
    let Some(index_model) = meta.get("embedding_model").and_then(|v| v.as_str()) else {
        return Err(ColibriError::Query(format!(
            "Index metadata missing embedding_model for profile '{}' in generation '{}'. \
             Rebuild with `colibri index --generation {} --force`.",
            profile_id, config.active_generation, config.active_generation
        )));
    };

    if index_model != expected_model {
        return Err(ColibriError::Query(format!(
            "Embedding model mismatch for profile '{}' in generation '{}': index='{}', config='{}'. \
             Rebuild with `colibri index --generation {} --force`.",
            profile_id,
            config.active_generation,
            index_model,
            expected_model,
            config.active_generation
        )));
    }

    Ok(())
}

fn collect_search_results(
    batches: &[RecordBatch],
    similarity_threshold: f64,
    out: &mut Vec<SearchResult>,
) {
    for batch in batches {
        let num_rows = batch.num_rows();

        let text_col = batch
            .column_by_name("text")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let file_col = batch
            .column_by_name("source_file")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let title_col = batch
            .column_by_name("title")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let type_col = batch
            .column_by_name("doc_type")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let folder_col = batch
            .column_by_name("folder")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let dist_col = batch
            .column_by_name("_distance")
            .and_then(|c| c.as_any().downcast_ref::<Float32Array>());

        for i in 0..num_rows {
            let distance = dist_col.map(|c| c.value(i) as f64).unwrap_or(0.0);
            let score = (-distance).exp();
            if score < similarity_threshold {
                continue;
            }

            let source_file = file_col.map(|c| c.value(i)).unwrap_or("");
            let title = title_col.map(|c| c.value(i)).unwrap_or_else(|| {
                Path::new(source_file)
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("")
            });

            out.push(SearchResult {
                text: text_col.map(|c| c.value(i).to_string()).unwrap_or_default(),
                file: source_file.to_string(),
                title: title.to_string(),
                doc_type: type_col
                    .map(|c| c.value(i).to_string())
                    .unwrap_or_else(|| "note".to_string()),
                folder: folder_col
                    .map(|c| c.value(i).to_string())
                    .unwrap_or_default(),
                score: (score * 10000.0).round() / 10000.0,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::profile_enabled_for_active_generation;
    use std::collections::HashMap;

    #[test]
    fn profile_enabled_when_generation_state_empty() {
        let m = HashMap::new();
        assert!(profile_enabled_for_active_generation("local_default", &m));
    }

    #[test]
    fn profile_enabled_only_when_ready() {
        let mut m = HashMap::new();
        m.insert("local_default".to_string(), "building".to_string());
        assert!(!profile_enabled_for_active_generation("local_default", &m));

        m.insert("local_default".to_string(), "ready".to_string());
        assert!(profile_enabled_for_active_generation("local_default", &m));
    }
}
