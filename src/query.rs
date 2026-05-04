//! Query engine for semantic search over content sources.
//!
//! Mirrors the Python `query.py` module. Uses LanceDB vector search
//! with L2 distance converted to similarity score via `exp(-distance)`.

use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::str::FromStr;

use arrow_array::{Float32Array, RecordBatch, StringArray};
use chrono::{DateTime, Utc};
use futures::TryStreamExt;
use lancedb::index::scalar::FullTextSearchQuery;
use lancedb::query::{ExecutableQuery, QueryBase};
use serde::Serialize;
use serde_json::{Map as JsonMap, Value as JsonValue};
use tracing::warn;

use crate::config::AppConfig;
use crate::embedding::embed_texts;
use crate::error::ColibriError;
use crate::metadata_store::{DocumentRow, MetadataStore};

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
    pub search_mode: SearchMode,
    /// When `group_by_doc` was true: how many chunks of this document
    /// matched the underlying search.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunk_count: Option<usize>,
    /// When `group_by_doc` was true: the document's parsed frontmatter
    /// (populated from metadata_store).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frontmatter: Option<JsonMap<String, JsonValue>>,
}

/// Optional filters applied to a search.
///
/// All fields default to "no filter". `Default::default()` yields the
/// pre-feature behavior (returns all matching chunks).
#[derive(Debug, Clone, Default)]
pub struct SearchFilter {
    pub classification: Option<String>,
    pub doc_type: Option<String>,
    /// Return only docs whose `markdown_path` contains *any* listed substring.
    pub path_includes: Vec<String>,
    /// Drop docs whose `markdown_path` contains *any* listed substring.
    pub path_excludes: Vec<String>,
    /// Equality match on parsed frontmatter fields. Multiple keys combine with AND.
    pub frontmatter: BTreeMap<String, String>,
    /// Drop docs with `source_updated_at` strictly before this timestamp.
    pub since: Option<DateTime<Utc>>,
}

impl SearchFilter {
    /// Convenience constructor preserving the pre-feature `(classification, doc_type)` shape.
    pub fn legacy(classification: Option<&str>, doc_type: Option<&str>) -> Self {
        Self {
            classification: classification.map(|s| s.to_string()),
            doc_type: doc_type.map(|s| s.to_string()),
            ..Default::default()
        }
    }
}

/// Controls how search queries are executed against LanceDB.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, serde::Deserialize)]
pub enum SearchMode {
    /// BM25 + vector combined via LanceDB native RRF.
    #[default]
    Hybrid,
    /// Vector-only search (original behavior).
    Semantic,
    /// BM25 full-text search only.
    Keyword,
}

impl fmt::Display for SearchMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SearchMode::Hybrid => write!(f, "hybrid"),
            SearchMode::Semantic => write!(f, "semantic"),
            SearchMode::Keyword => write!(f, "keyword"),
        }
    }
}

impl FromStr for SearchMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "hybrid" => Ok(SearchMode::Hybrid),
            "semantic" => Ok(SearchMode::Semantic),
            "keyword" => Ok(SearchMode::Keyword),
            other => Err(format!(
                "Invalid search mode '{other}'. Must be one of: hybrid, semantic, keyword"
            )),
        }
    }
}

impl clap::ValueEnum for SearchMode {
    fn value_variants<'a>() -> &'a [Self] {
        &[
            SearchMode::Hybrid,
            SearchMode::Semantic,
            SearchMode::Keyword,
        ]
    }

    fn to_possible_value(&self) -> Option<clap::builder::PossibleValue> {
        match self {
            SearchMode::Hybrid => Some(clap::builder::PossibleValue::new("hybrid")),
            SearchMode::Semantic => Some(clap::builder::PossibleValue::new("semantic")),
            SearchMode::Keyword => Some(clap::builder::PossibleValue::new("keyword")),
        }
    }
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

    /// Search with optional filters and grouping.
    ///
    /// `filter` allows constraining results by classification, doc_type,
    /// path-includes/excludes, frontmatter equality, and `since` timestamp.
    /// `group_by_doc = true` returns one result per document (best matching
    /// chunk + chunk_count + frontmatter); `false` returns chunk-level results
    /// (legacy behaviour).
    pub async fn search(
        &self,
        query: &str,
        filter: &SearchFilter,
        group_by_doc: bool,
        limit: usize,
        mode: SearchMode,
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
            // Refresh to latest LanceDB version so externally-rebuilt indexes
            // (including FTS indexes) are visible to this table handle.
            if let Err(e) = backend.table.checkout_latest().await {
                warn!(
                    "Failed to refresh table for profile '{}': {e}",
                    backend.profile_id
                );
            }

            let batches: Vec<RecordBatch> = match mode {
                SearchMode::Semantic => {
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
                                "Skipping profile '{}': embedding returned empty vector",
                                backend.profile_id
                            );
                            continue;
                        }
                        Err(e) => {
                            warn!(
                                "Skipping profile '{}': embed error: {e}",
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

                    match search.execute().await {
                        Ok(stream) => match stream.try_collect().await {
                            Ok(b) => b,
                            Err(e) => {
                                warn!(
                                    "Skipping profile '{}': collect error: {e}",
                                    backend.profile_id
                                );
                                continue;
                            }
                        },
                        Err(e) => {
                            warn!(
                                "Skipping profile '{}': query error: {e}",
                                backend.profile_id
                            );
                            continue;
                        }
                    }
                }
                SearchMode::Keyword => {
                    let fts_query = FullTextSearchQuery::new(query.to_string());
                    let search = backend
                        .table
                        .query()
                        .full_text_search(fts_query)
                        .limit(per_backend_limit);

                    match search.execute().await {
                        Ok(stream) => match stream.try_collect().await {
                            Ok(b) => b,
                            Err(e) => {
                                warn!(
                                    "Skipping profile '{}': FTS collect error: {e}",
                                    backend.profile_id
                                );
                                continue;
                            }
                        },
                        Err(e) => {
                            warn!(
                                "Skipping profile '{}': FTS query error: {e}",
                                backend.profile_id
                            );
                            continue;
                        }
                    }
                }
                SearchMode::Hybrid => {
                    match embed_texts(
                        &[query.to_string()],
                        &backend.embedding_model,
                        &backend.embedding_endpoint,
                    )
                    .await
                    {
                        Ok(v) if !v.is_empty() => {
                            let fts_query = FullTextSearchQuery::new(query.to_string());
                            let search = backend
                                .table
                                .query()
                                .full_text_search(fts_query)
                                .nearest_to(v[0].as_slice())
                                .map_err(|e| {
                                    ColibriError::Query(format!("Hybrid search failed: {e}"))
                                })?
                                .limit(per_backend_limit);

                            match search.execute().await {
                                Ok(stream) => match stream.try_collect().await {
                                    Ok(b) => b,
                                    Err(e) => {
                                        warn!(
                                            "Skipping profile '{}': hybrid collect error: {e}",
                                            backend.profile_id
                                        );
                                        continue;
                                    }
                                },
                                Err(e) => {
                                    warn!(
                                        "Skipping profile '{}': hybrid query error: {e}",
                                        backend.profile_id
                                    );
                                    continue;
                                }
                            }
                        }
                        Ok(_) => {
                            warn!(
                                "Skipping profile '{}': embedding returned empty vector",
                                backend.profile_id
                            );
                            continue;
                        }
                        Err(e) => {
                            // REQ-008: fallback to keyword on embedding failure
                            warn!(
                                "Embedding failed for profile '{}', falling back to keyword search: {e}",
                                backend.profile_id
                            );
                            let fts_query = FullTextSearchQuery::new(query.to_string());
                            let search = backend
                                .table
                                .query()
                                .full_text_search(fts_query)
                                .limit(per_backend_limit);

                            match search.execute().await {
                                Ok(stream) => match stream.try_collect().await {
                                    Ok(b) => b,
                                    Err(e) => {
                                        warn!(
                                            "Skipping profile '{}': FTS fallback error: {e}",
                                            backend.profile_id
                                        );
                                        continue;
                                    }
                                },
                                Err(e) => {
                                    warn!(
                                        "Skipping profile '{}': FTS fallback query error: {e}",
                                        backend.profile_id
                                    );
                                    continue;
                                }
                            }
                        }
                    }
                }
            };

            succeeded += 1;
            collect_search_hits(
                &batches,
                self.config.similarity_threshold,
                mode,
                &mut merged,
            );
        }

        if succeeded == 0 {
            return Err(ColibriError::Query(
                "No searchable embedding profile is currently available.".into(),
            ));
        }

        merged.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));

        // Look up document metadata once for all hit doc_ids — needed for
        // every filter predicate plus (when grouping) the result frontmatter.
        let mut doc_ids_set = HashSet::new();
        for hit in &merged {
            doc_ids_set.insert(hit.doc_id.clone());
        }
        let doc_ids: Vec<String> = doc_ids_set.into_iter().collect();
        let store = MetadataStore::open(&self.config.metadata_db_path)?;
        let docs_by_id = store.get_documents_by_ids(&doc_ids)?;

        // Stage 1: filter hits using metadata join. Drops anything failing
        // any predicate; preserves chunk-level granularity.
        let filtered: Vec<(SearchHit, &DocumentRow)> = merged
            .into_iter()
            .filter_map(|hit| {
                let doc = docs_by_id.get(&hit.doc_id)?;
                if doc.deleted {
                    return None;
                }
                if !document_matches_filter(doc, filter) {
                    return None;
                }
                Some((hit, doc))
            })
            .collect();

        // Stage 2 + 3: group_by_doc OR chunk-level dedup, then truncate to limit.
        let results = if group_by_doc {
            collapse_to_doc_results(filtered, mode, limit)
        } else {
            chunk_level_results(filtered, mode, limit)
        };

        Ok(results)
    }

    /// Search only books. Returns chunk-level results (legacy behavior).
    /// Convenience wrapper around `search()`.
    #[allow(dead_code)] // Public API — kept for external callers/tests after MCP refactor.
    pub async fn search_books(
        &self,
        query: &str,
        limit: usize,
        mode: SearchMode,
    ) -> Result<Vec<SearchResult>, ColibriError> {
        self.search(
            query,
            &SearchFilter::legacy(None, Some("book")),
            false,
            limit,
            mode,
        )
        .await
    }

    /// Search the entire library. Returns chunk-level results (legacy behavior).
    /// Convenience wrapper around `search()`.
    #[allow(dead_code)] // Public API — kept for external callers/tests after MCP refactor.
    pub async fn search_library(
        &self,
        query: &str,
        limit: usize,
        mode: SearchMode,
    ) -> Result<Vec<SearchResult>, ColibriError> {
        self.search(query, &SearchFilter::default(), false, limit, mode)
            .await
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

/// Predicate: does this document satisfy every active filter?
fn document_matches_filter(doc: &DocumentRow, filter: &SearchFilter) -> bool {
    if let Some(dt) = &filter.doc_type {
        if doc.doc_type != *dt {
            return false;
        }
    }
    if let Some(c) = &filter.classification {
        if doc.classification != *c {
            return false;
        }
    }
    if !filter.path_includes.is_empty() {
        let any = filter
            .path_includes
            .iter()
            .any(|needle| doc.markdown_path.contains(needle));
        if !any {
            return false;
        }
    }
    if filter
        .path_excludes
        .iter()
        .any(|needle| doc.markdown_path.contains(needle))
    {
        return false;
    }
    if let Some(since) = &filter.since {
        // doc.source_updated_at is RFC 3339 (validated upstream by envelope::validate).
        // If parsing fails, conservatively drop the doc.
        match DateTime::parse_from_rfc3339(&doc.source_updated_at) {
            Ok(t) => {
                if t.with_timezone(&Utc) < *since {
                    return false;
                }
            }
            Err(_) => return false,
        }
    }
    if !filter.frontmatter.is_empty() {
        // Parse the doc's frontmatter_json once. Empty/missing → no match for any key.
        let parsed: JsonValue = serde_json::from_str(&doc.frontmatter_json)
            .unwrap_or(JsonValue::Object(JsonMap::new()));
        let map = match parsed.as_object() {
            Some(m) => m,
            None => return false,
        };
        for (key, expected) in &filter.frontmatter {
            match map.get(key) {
                Some(JsonValue::String(s)) if s == expected => {}
                Some(JsonValue::Number(n)) if &n.to_string() == expected => {}
                Some(JsonValue::Bool(b)) if &b.to_string() == expected => {}
                _ => return false,
            }
        }
    }
    true
}

/// Convert filtered (hit, doc) pairs into chunk-level results.
/// Preserves the legacy text-dedup behavior: skip exact-text duplicates.
fn chunk_level_results(
    filtered: Vec<(SearchHit, &DocumentRow)>,
    mode: SearchMode,
    limit: usize,
) -> Vec<SearchResult> {
    let mut deduped = Vec::new();
    let mut seen = HashSet::new();
    for (hit, doc) in filtered {
        let result = SearchResult {
            text: hit.text,
            file: doc.markdown_path.clone(),
            title: doc.title.clone(),
            doc_type: doc.doc_type.clone(),
            classification: doc.classification.clone(),
            score: hit.score,
            search_mode: mode,
            chunk_count: None,
            frontmatter: None,
        };
        let key = format!("{}:{}", result.file, result.text);
        if seen.insert(key) {
            deduped.push(result);
        }
        if deduped.len() >= limit {
            break;
        }
    }
    deduped
}

/// Collapse filtered (hit, doc) pairs into one result per document.
/// Keeps the highest-scoring chunk; counts how many chunks of that doc matched.
/// Attaches the document's frontmatter map.
fn collapse_to_doc_results(
    filtered: Vec<(SearchHit, &DocumentRow)>,
    mode: SearchMode,
    limit: usize,
) -> Vec<SearchResult> {
    // Bucket by doc_id; for each, keep best-scoring hit and count.
    let mut best: HashMap<String, (SearchHit, &DocumentRow, usize)> = HashMap::new();
    for (hit, doc) in filtered {
        let entry = best
            .entry(hit.doc_id.clone())
            .or_insert_with(|| (hit.clone(), doc, 0));
        entry.2 += 1;
        if hit.score > entry.0.score {
            entry.0 = hit;
            entry.1 = doc;
        }
    }
    let mut results: Vec<SearchResult> = best
        .into_values()
        .map(|(hit, doc, count)| {
            let frontmatter = serde_json::from_str::<JsonValue>(&doc.frontmatter_json)
                .ok()
                .and_then(|v| v.as_object().cloned());
            SearchResult {
                text: hit.text,
                file: doc.markdown_path.clone(),
                title: doc.title.clone(),
                doc_type: doc.doc_type.clone(),
                classification: doc.classification.clone(),
                score: hit.score,
                search_mode: mode,
                chunk_count: Some(count),
                frontmatter,
            }
        })
        .collect();
    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
    results.truncate(limit);
    results
}

fn collect_search_hits(
    batches: &[RecordBatch],
    similarity_threshold: f64,
    mode: SearchMode,
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

        let score_col = batch
            .column_by_name("_score")
            .and_then(|c| c.as_any().downcast_ref::<Float32Array>());
        let dist_col = batch
            .column_by_name("_distance")
            .and_then(|c| c.as_any().downcast_ref::<Float32Array>());

        for i in 0..num_rows {
            let score = if let Some(sc) = score_col {
                sc.value(i) as f64
            } else {
                let distance = dist_col.map(|c| c.value(i) as f64).unwrap_or(0.0);
                (-distance).exp()
            };

            if mode == SearchMode::Semantic && score < similarity_threshold {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_mode_default_is_hybrid() {
        assert_eq!(SearchMode::default(), SearchMode::Hybrid);
    }

    #[test]
    fn search_mode_from_str_valid() {
        assert_eq!("hybrid".parse::<SearchMode>().unwrap(), SearchMode::Hybrid);
        assert_eq!(
            "semantic".parse::<SearchMode>().unwrap(),
            SearchMode::Semantic
        );
        assert_eq!(
            "keyword".parse::<SearchMode>().unwrap(),
            SearchMode::Keyword
        );
        assert_eq!("HYBRID".parse::<SearchMode>().unwrap(), SearchMode::Hybrid);
        assert_eq!(
            "Semantic".parse::<SearchMode>().unwrap(),
            SearchMode::Semantic
        );
    }

    #[test]
    fn search_mode_from_str_invalid() {
        let err = "fuzzy".parse::<SearchMode>().unwrap_err();
        assert!(err.contains("Invalid search mode"));
        assert!(err.contains("fuzzy"));
    }

    #[test]
    fn search_mode_display() {
        assert_eq!(SearchMode::Hybrid.to_string(), "hybrid");
        assert_eq!(SearchMode::Semantic.to_string(), "semantic");
        assert_eq!(SearchMode::Keyword.to_string(), "keyword");
    }

    // ----- SearchFilter + helpers (Wave 2 Cluster E) -----

    fn make_doc(
        doc_id: &str,
        path: &str,
        doc_type: &str,
        classification: &str,
        frontmatter_json: &str,
        source_updated_at: &str,
    ) -> DocumentRow {
        DocumentRow {
            doc_id: doc_id.into(),
            title: doc_id.into(),
            content_hash: "sha256:0000000000000000000000000000000000000000000000000000000000000000"
                .into(),
            doc_type: doc_type.into(),
            classification: classification.into(),
            markdown_path: path.into(),
            tags_json: "[]".into(),
            deleted: false,
            frontmatter_json: frontmatter_json.into(),
            source_updated_at: source_updated_at.into(),
        }
    }

    #[test]
    fn filter_default_matches_everything() {
        let doc = make_doc(
            "d1",
            "internal/a.md",
            "note",
            "internal",
            "{}",
            "2026-01-01T00:00:00Z",
        );
        assert!(document_matches_filter(&doc, &SearchFilter::default()));
    }

    #[test]
    fn filter_classification_and_doc_type() {
        let doc = make_doc(
            "d1",
            "internal/a.md",
            "note",
            "internal",
            "{}",
            "2026-01-01T00:00:00Z",
        );
        let mut f = SearchFilter::default();
        f.classification = Some("internal".into());
        assert!(document_matches_filter(&doc, &f));
        f.classification = Some("public".into());
        assert!(!document_matches_filter(&doc, &f));

        let mut f = SearchFilter::default();
        f.doc_type = Some("note".into());
        assert!(document_matches_filter(&doc, &f));
        f.doc_type = Some("book".into());
        assert!(!document_matches_filter(&doc, &f));
    }

    #[test]
    fn filter_path_includes() {
        let doc = make_doc(
            "d1",
            "03_MY_PROJECTS/02_HEIMDALL/foo.md",
            "note",
            "internal",
            "{}",
            "2026-01-01T00:00:00Z",
        );
        let mut f = SearchFilter::default();
        f.path_includes = vec!["02_HEIMDALL".into()];
        assert!(document_matches_filter(&doc, &f));
        f.path_includes = vec!["03_GO_AI".into()];
        assert!(!document_matches_filter(&doc, &f));
        // Multiple substrings: ANY match wins
        f.path_includes = vec!["03_GO_AI".into(), "HEIMDALL".into()];
        assert!(document_matches_filter(&doc, &f));
    }

    #[test]
    fn filter_path_excludes() {
        let doc = make_doc(
            "d1",
            "06_ARCHIVE/old.md",
            "note",
            "internal",
            "{}",
            "2026-01-01T00:00:00Z",
        );
        let mut f = SearchFilter::default();
        f.path_excludes = vec!["06_ARCHIVE".into()];
        assert!(!document_matches_filter(&doc, &f));
    }

    #[test]
    fn filter_frontmatter_string() {
        let doc = make_doc(
            "d1",
            "internal/a.md",
            "note",
            "internal",
            r#"{"area":"SIT","status":"active"}"#,
            "2026-01-01T00:00:00Z",
        );
        let mut f = SearchFilter::default();
        f.frontmatter.insert("area".into(), "SIT".into());
        assert!(document_matches_filter(&doc, &f));
        f.frontmatter.insert("status".into(), "draft".into());
        assert!(!document_matches_filter(&doc, &f));
    }

    #[test]
    fn filter_frontmatter_missing_key_excludes() {
        let doc = make_doc(
            "d1",
            "internal/a.md",
            "note",
            "internal",
            "{}",
            "2026-01-01T00:00:00Z",
        );
        let mut f = SearchFilter::default();
        f.frontmatter.insert("area".into(), "SIT".into());
        assert!(!document_matches_filter(&doc, &f));
    }

    #[test]
    fn filter_since() {
        let doc = make_doc(
            "d1",
            "internal/a.md",
            "note",
            "internal",
            "{}",
            "2026-04-15T00:00:00Z",
        );
        let mut f = SearchFilter::default();
        f.since = Some(
            DateTime::parse_from_rfc3339("2026-04-01T00:00:00Z")
                .unwrap()
                .with_timezone(&Utc),
        );
        assert!(document_matches_filter(&doc, &f));
        f.since = Some(
            DateTime::parse_from_rfc3339("2026-05-01T00:00:00Z")
                .unwrap()
                .with_timezone(&Utc),
        );
        assert!(!document_matches_filter(&doc, &f));
    }

    #[test]
    fn collapse_to_doc_keeps_best_chunk_and_counts() {
        let docs = vec![
            make_doc(
                "d1",
                "p1.md",
                "note",
                "internal",
                "{}",
                "2026-01-01T00:00:00Z",
            ),
            make_doc(
                "d2",
                "p2.md",
                "note",
                "internal",
                "{}",
                "2026-01-01T00:00:00Z",
            ),
        ];
        // Three hits across two docs; d1 has chunks at 0.5 and 0.9, d2 at 0.7
        let filtered: Vec<(SearchHit, &DocumentRow)> = vec![
            (
                SearchHit {
                    doc_id: "d1".into(),
                    text: "low".into(),
                    score: 0.5,
                },
                &docs[0],
            ),
            (
                SearchHit {
                    doc_id: "d1".into(),
                    text: "high".into(),
                    score: 0.9,
                },
                &docs[0],
            ),
            (
                SearchHit {
                    doc_id: "d2".into(),
                    text: "mid".into(),
                    score: 0.7,
                },
                &docs[1],
            ),
        ];
        let results = collapse_to_doc_results(filtered, SearchMode::Hybrid, 10);
        assert_eq!(results.len(), 2);
        // Best (d1@0.9) ranks first; chunk_count=2.
        assert_eq!(results[0].file, "p1.md");
        assert_eq!(results[0].text, "high");
        assert_eq!(results[0].chunk_count, Some(2));
        // d2 ranks second with chunk_count=1.
        assert_eq!(results[1].file, "p2.md");
        assert_eq!(results[1].chunk_count, Some(1));
    }

    #[test]
    fn collapse_to_doc_attaches_frontmatter() {
        let docs = vec![make_doc(
            "d1",
            "p1.md",
            "note",
            "internal",
            r#"{"area":"SIT"}"#,
            "2026-01-01T00:00:00Z",
        )];
        let filtered = vec![(
            SearchHit {
                doc_id: "d1".into(),
                text: "x".into(),
                score: 0.9,
            },
            &docs[0],
        )];
        let results = collapse_to_doc_results(filtered, SearchMode::Hybrid, 10);
        assert_eq!(
            results[0]
                .frontmatter
                .as_ref()
                .and_then(|m| m.get("area"))
                .and_then(JsonValue::as_str),
            Some("SIT")
        );
    }

    #[test]
    fn collapse_truncates_to_limit() {
        let docs: Vec<DocumentRow> = (0..5)
            .map(|i| {
                make_doc(
                    &format!("d{i}"),
                    &format!("p{i}.md"),
                    "note",
                    "internal",
                    "{}",
                    "2026-01-01T00:00:00Z",
                )
            })
            .collect();
        let filtered: Vec<(SearchHit, &DocumentRow)> = docs
            .iter()
            .enumerate()
            .map(|(i, d)| {
                (
                    SearchHit {
                        doc_id: format!("d{i}"),
                        text: format!("t{i}"),
                        score: 0.9 - (i as f64) * 0.01,
                    },
                    d,
                )
            })
            .collect();
        let results = collapse_to_doc_results(filtered, SearchMode::Hybrid, 3);
        assert_eq!(results.len(), 3);
    }
}
