//! Query engine for semantic search over content sources.
//!
//! Mirrors the Python `query.py` module. Uses LanceDB vector search
//! with L2 distance converted to similarity score via `exp(-distance)`.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use arrow_array::{Float32Array, RecordBatch, StringArray};
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase, Select};
use serde::Serialize;

use crate::config::{AppConfig, SCHEMA_VERSION};
use crate::embedding::embed_texts;
use crate::error::ColibriError;
use crate::index_meta::read_index_meta;

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
    table: lancedb::Table,
    config: AppConfig,
}

impl SearchEngine {
    /// Create a new search engine, verifying schema version.
    pub async fn new(config: &AppConfig) -> Result<Self, ColibriError> {
        // Check schema version
        let meta = read_index_meta(&config.lancedb_dir)?;
        let stored_version = meta
            .get("schema_version")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;

        if stored_version != SCHEMA_VERSION {
            return Err(ColibriError::Query(format!(
                "Index schema outdated (v{stored_version}, need v{SCHEMA_VERSION}). \
                 Run `colibri index --force` to rebuild."
            )));
        }

        let db = lancedb::connect(config.lancedb_dir.to_string_lossy().as_ref())
            .execute()
            .await?;

        let table = db.open_table("chunks").execute().await.map_err(|e| {
            ColibriError::Query(format!(
                "Failed to open 'chunks' table. Run `colibri index` first. ({e})"
            ))
        })?;

        Ok(Self {
            table,
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
        // Embed the query
        let query_vector = embed_texts(
            &[query.to_string()],
            &self.config.embedding_model,
            &self.config.ollama_base_url,
        )
        .await?;

        if query_vector.is_empty() {
            return Err(ColibriError::Embedding(
                "Query embedding returned no results".into(),
            ));
        }

        // Build search
        let mut search = self
            .table
            .vector_search(query_vector[0].clone())
            .map_err(|e| ColibriError::Query(format!("Vector search failed: {e}")))?
            .limit(self.config.top_k);

        // Apply filters
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

        let raw_results = search
            .execute()
            .await
            .map_err(|e| ColibriError::Query(format!("Search execution failed: {e}")))?;

        // Collect results from the stream
        let batches: Vec<RecordBatch> = raw_results
            .try_collect()
            .await
            .map_err(|e| ColibriError::Query(format!("Failed to collect results: {e}")))?;

        let mut results = Vec::new();

        for batch in &batches {
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

                if score < self.config.similarity_threshold {
                    continue;
                }

                let source_file = file_col.map(|c| c.value(i)).unwrap_or("");
                let title = title_col.map(|c| c.value(i)).unwrap_or_else(|| {
                    Path::new(source_file)
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("")
                });

                results.push(SearchResult {
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

                if results.len() >= limit {
                    return Ok(results);
                }
            }
        }

        Ok(results)
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
        // Query all book rows
        let results = self
            .table
            .query()
            .only_if("doc_type = 'book'")
            .select(Select::columns(&["title", "source_file"]))
            .execute()
            .await
            .map_err(|e| ColibriError::Query(format!("list_books query failed: {e}")))?;

        let batches: Vec<RecordBatch> = results
            .try_collect()
            .await
            .map_err(|e| ColibriError::Query(format!("Failed to collect: {e}")))?;

        // Aggregate by title
        let mut book_map: HashMap<String, (usize, String)> = HashMap::new();

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
        let query = if let Some(folder) = folder {
            let escaped = folder.replace('\'', "''");
            self.table
                .query()
                .only_if(format!("folder = '{escaped}'"))
                .select(Select::columns(&["source_file", "tags"]))
                .execute()
                .await
        } else {
            self.table
                .query()
                .select(Select::columns(&["source_file", "tags"]))
                .execute()
                .await
        };

        let results =
            query.map_err(|e| ColibriError::Query(format!("browse_topics failed: {e}")))?;

        let batches: Vec<RecordBatch> = results
            .try_collect()
            .await
            .map_err(|e| ColibriError::Query(format!("Failed to collect: {e}")))?;

        // Deduplicate by source_file, then count tags
        let mut seen_files: HashSet<String> = HashSet::new();
        let mut tag_counter: HashMap<String, usize> = HashMap::new();

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
