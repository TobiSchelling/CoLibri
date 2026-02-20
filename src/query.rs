//! Query engine for semantic search over content sources.
//!
//! Mirrors the Python `query.py` module. Uses LanceDB vector search
//! with L2 distance converted to similarity score via `exp(-distance)`.

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use arrow_array::{Float32Array, RecordBatch, StringArray};
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use serde::Serialize;
use tracing::warn;

use crate::config::AppConfig;
use crate::embedding::embed_texts;
use crate::error::ColibriError;
use crate::metadata_store::MetadataStore;

/// A single search result.
#[derive(Debug, Clone, Serialize)]
pub struct SearchResult {
    pub text: String,
    pub file: String,
    pub title: String,
    #[serde(rename = "type")]
    pub doc_type: String,
    pub classification: String,
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

#[derive(Debug, Clone)]
struct SearchHit {
    doc_id: String,
    text: String,
    score: f64,
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
        let checks = crate::serve_ready::profile_checks(config)?;

        for check in checks {
            if !check.queryable {
                warn!(
                    "Skipping profile '{}' for active generation (not serve-ready): {}",
                    check.profile_id,
                    check.issues.join("; ")
                );
                continue;
            }

            let profile_id = check.profile_id;
            let profile = config.embedding_profile(&profile_id)?;
            let lancedb_dir = config.lancedb_dir_for_profile(&profile_id);

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
                "No searchable embedding profile is currently available. Run `colibri doctor`, then `colibri index --force`.".into(),
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
        classification: Option<&str>,
        doc_type: Option<&str>,
        limit: usize,
    ) -> Result<Vec<SearchResult>, ColibriError> {
        let mut merged: Vec<SearchHit> = Vec::new();
        let mut succeeded = 0usize;
        let per_backend_limit = self
            .config
            .top_k
            .saturating_mul(5)
            .max(limit.saturating_mul(5))
            .min(500);

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

            let search = backend
                .table
                .vector_search(query_vector[0].clone())
                .map_err(|e| ColibriError::Query(format!("Vector search failed: {e}")))?
                .limit(per_backend_limit);

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
            collect_search_hits(&batches, self.config.similarity_threshold, &mut merged);
        }

        if succeeded == 0 {
            return Err(ColibriError::Query(
                "No searchable embedding profile is currently available.".into(),
            ));
        }

        merged.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));

        let mut doc_ids_set = HashSet::new();
        for hit in &merged {
            doc_ids_set.insert(hit.doc_id.clone());
        }
        let doc_ids: Vec<String> = doc_ids_set.into_iter().collect();
        let store = MetadataStore::open(&self.config.metadata_db_path)?;
        let docs_by_id = store.get_documents_by_ids(&doc_ids)?;

        let mut deduped = Vec::new();
        let mut seen = HashSet::new();
        for hit in merged {
            let Some(doc) = docs_by_id.get(&hit.doc_id) else {
                continue;
            };
            if doc.deleted {
                continue;
            }

            if let Some(dt) = doc_type {
                if doc.doc_type != dt {
                    continue;
                }
            }
            if let Some(classification) = classification {
                if doc.classification != classification {
                    continue;
                }
            }

            let result = SearchResult {
                text: hit.text,
                file: doc.markdown_path.clone(),
                title: doc.title.clone(),
                doc_type: doc.doc_type.clone(),
                classification: doc.classification.clone(),
                score: hit.score,
            };

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
        let store = MetadataStore::open(&self.config.metadata_db_path)?;
        let docs = store.list_documents()?;
        let chunk_counts =
            store.indexed_chunk_counts_for_generation(&self.config.active_generation)?;

        let mut books = Vec::new();
        for doc in docs {
            if doc.deleted || doc.doc_type != "book" {
                continue;
            }
            let profile_id = self
                .config
                .resolve_embedding_profile_id(&doc.classification);
            let chunks = chunk_counts
                .get(&(doc.doc_id.clone(), profile_id))
                .copied()
                .unwrap_or(0) as usize;
            books.push(BookEntry {
                title: doc.title,
                chunks,
                file: doc.markdown_path,
            });
        }

        books.sort_by(|a, b| a.title.cmp(&b.title));
        Ok(books)
    }

    /// List all topics (tags) with document counts.
    pub async fn browse_topics(
        &self,
        classification: Option<&str>,
    ) -> Result<Vec<TopicEntry>, ColibriError> {
        let store = MetadataStore::open(&self.config.metadata_db_path)?;
        let docs = store.list_documents()?;
        let mut tag_counter: HashMap<String, usize> = HashMap::new();

        for doc in docs {
            if doc.deleted {
                continue;
            }
            if let Some(classification) = classification {
                if doc.classification != classification {
                    continue;
                }
            }

            let tags: Result<Vec<String>, _> = serde_json::from_str(&doc.tags_json);
            let tags = match tags {
                Ok(t) => t,
                Err(_) => continue,
            };
            let mut seen = HashSet::new();
            for tag in tags {
                let tag = tag.trim().to_string();
                if tag.is_empty() {
                    continue;
                }
                if seen.insert(tag.clone()) {
                    *tag_counter.entry(tag).or_default() += 1;
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

fn collect_search_hits(
    batches: &[RecordBatch],
    similarity_threshold: f64,
    out: &mut Vec<SearchHit>,
) {
    for batch in batches {
        let num_rows = batch.num_rows();

        let doc_id_col = batch
            .column_by_name("doc_id")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let text_col = batch
            .column_by_name("text")
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

            let doc_id = doc_id_col.map(|c| c.value(i)).unwrap_or("");
            out.push(SearchHit {
                doc_id: doc_id.to_string(),
                text: text_col.map(|c| c.value(i).to_string()).unwrap_or_default(),
                score: (score * 10000.0).round() / 10000.0,
            });
        }
    }
}
