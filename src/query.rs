//! Query engine: hybrid (BM25 + vector), semantic and keyword search over the
//! single LanceDB `chunks` table, joined with document metadata from SQLite.

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
use crate::metadata_store::DocumentRecord;

/// LanceDB table name (shared with the indexer).
const TABLE_NAME: &str = "chunks";

/// A single search result.
#[derive(Debug, Clone, Serialize)]
pub struct SearchResult {
    pub text: String,
    pub file: String,
    pub title: String,
    #[serde(rename = "type")]
    pub doc_type: String,
    pub collection: String,
    pub score: f64,
    pub search_mode: SearchMode,
    /// When `group_by_doc` was true: how many chunks of this document
    /// matched the underlying search.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunk_count: Option<usize>,
    /// When `group_by_doc` was true: the document's parsed frontmatter.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frontmatter: Option<JsonMap<String, JsonValue>>,
}

/// Optional filters applied to a search. `Default` means no filter.
#[derive(Debug, Clone, Default)]
pub struct SearchFilter {
    pub collection: Option<String>,
    pub doc_type: Option<String>,
    /// Keep only docs whose source key (path relative to the source root)
    /// contains *any* listed substring.
    pub path_includes: Vec<String>,
    /// Drop docs whose source key contains *any* listed substring.
    pub path_excludes: Vec<String>,
    /// Equality match on parsed frontmatter fields. Multiple keys combine with AND.
    pub frontmatter: BTreeMap<String, String>,
    /// Drop docs with `source_updated_at` strictly before this timestamp.
    pub since: Option<DateTime<Utc>>,
}

/// Controls how search queries are executed against LanceDB.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, serde::Deserialize)]
pub enum SearchMode {
    /// BM25 + vector combined via LanceDB native RRF.
    #[default]
    Hybrid,
    /// Vector-only search.
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

/// Book entry for `list_books`.
#[derive(Debug, Clone, Serialize)]
pub struct BookEntry {
    pub title: String,
    pub authors: Vec<String>,
    pub source_path: Option<String>,
    pub chunks: usize,
}

/// Topic entry with document count.
#[derive(Debug, Clone, Serialize)]
pub struct TopicEntry {
    pub tag: String,
    pub document_count: usize,
}

/// Search engine backed by the LanceDB `chunks` table.
pub struct SearchEngine {
    table: lancedb::Table,
    config: AppConfig,
}

#[derive(Debug, Clone)]
struct SearchHit {
    doc_id: String,
    text: String,
    score: f64,
}

impl SearchEngine {
    /// Open the index read-only after checking it matches the config.
    pub async fn new(config: &AppConfig) -> Result<Self, ColibriError> {
        let ready = crate::serve_ready::check(config)?;
        if !ready.queryable {
            return Err(ColibriError::Query(format!(
                "Index not ready: {}",
                ready.issues.join("; ")
            )));
        }
        let db = lancedb::connect(config.index_dir.to_string_lossy().as_ref())
            .execute()
            .await?;
        let table = db.open_table(TABLE_NAME).execute().await?;
        Ok(Self {
            table,
            config: config.clone(),
        })
    }

    async fn embed_query(&self, query: &str) -> Result<Vec<f32>, ColibriError> {
        embed_texts(
            &[query.to_string()],
            &self.config.embedding_model,
            &self.config.embedding_endpoint,
        )
        .await?
        .into_iter()
        .next()
        .ok_or_else(|| ColibriError::Embedding("embedding returned no vector".into()))
    }

    async fn fetch(
        &self,
        query: &str,
        mode: SearchMode,
        limit: usize,
    ) -> Result<Vec<RecordBatch>, ColibriError> {
        let keyword = || {
            self.table
                .query()
                .full_text_search(FullTextSearchQuery::new(query.to_string()))
                .limit(limit)
        };
        let batches = match mode {
            SearchMode::Keyword => keyword().execute().await?.try_collect().await?,
            SearchMode::Semantic => {
                let vector = self.embed_query(query).await?;
                self.table
                    .vector_search(vector)?
                    .limit(limit)
                    .execute()
                    .await?
                    .try_collect()
                    .await?
            }
            SearchMode::Hybrid => match self.embed_query(query).await {
                Ok(vector) => {
                    keyword()
                        .nearest_to(vector.as_slice())?
                        .execute()
                        .await?
                        .try_collect()
                        .await?
                }
                Err(e) => {
                    warn!("Embedding failed, falling back to keyword search: {e}");
                    keyword().execute().await?.try_collect().await?
                }
            },
        };
        Ok(batches)
    }

    /// Search with optional filters. `group_by_doc = true` returns one result
    /// per document (best chunk + chunk_count + frontmatter).
    pub async fn search(
        &self,
        query: &str,
        filter: &SearchFilter,
        group_by_doc: bool,
        limit: usize,
        mode: SearchMode,
    ) -> Result<Vec<SearchResult>, ColibriError> {
        // Refresh so an index rebuilt by another process is visible.
        if let Err(e) = self.table.checkout_latest().await {
            warn!("Failed to refresh index table: {e}");
        }
        let candidates = self
            .config
            .top_k
            .saturating_mul(5)
            .max(limit.saturating_mul(5))
            .min(500);
        let batches = self.fetch(query, mode, candidates).await?;

        let mut hits = Vec::new();
        collect_search_hits(&batches, self.config.similarity_threshold, mode, &mut hits);
        hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));

        let doc_ids: Vec<String> = hits
            .iter()
            .map(|h| h.doc_id.clone())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        let docs_by_id = self.config.open_read()?.get_documents_by_ids(&doc_ids)?;

        let filtered: Vec<(SearchHit, &DocumentRecord)> = hits
            .into_iter()
            .filter_map(|hit| {
                let doc = docs_by_id.get(&hit.doc_id)?;
                (doc.is_searchable() && document_matches_filter(doc, filter)).then_some((hit, doc))
            })
            .collect();

        Ok(if group_by_doc {
            collapse_to_doc_results(filtered, mode, limit, &self.config.canonical_dir)
        } else {
            chunk_level_results(filtered, mode, limit, &self.config.canonical_dir)
        })
    }

    /// All searchable books, sorted by title.
    pub async fn list_books(&self) -> Result<Vec<BookEntry>, ColibriError> {
        let mut books: Vec<BookEntry> = self
            .config
            .open_read()?
            .list_documents()?
            .into_iter()
            .filter(|d| d.is_searchable() && d.doc_type == "book")
            .map(|d| BookEntry {
                authors: serde_json::from_str(&d.authors_json).unwrap_or_default(),
                title: d.title,
                source_path: d.source_path,
                chunks: d.chunk_count.unwrap_or(0).max(0) as usize,
            })
            .collect();
        books.sort_by(|a, b| a.title.cmp(&b.title));
        Ok(books)
    }

    /// Tags with document counts, optionally limited to one collection.
    pub async fn browse_topics(
        &self,
        collection: Option<&str>,
    ) -> Result<Vec<TopicEntry>, ColibriError> {
        let mut tag_counter: HashMap<String, usize> = HashMap::new();
        for doc in self.config.open_read()?.list_documents()? {
            if !doc.is_searchable() || collection.is_some_and(|c| doc.collection != c) {
                continue;
            }
            let tags: Vec<String> = serde_json::from_str(&doc.tags_json).unwrap_or_default();
            let unique: HashSet<String> = tags
                .into_iter()
                .map(|t| t.trim().to_string())
                .filter(|t| !t.is_empty())
                .collect();
            for tag in unique {
                *tag_counter.entry(tag).or_default() += 1;
            }
        }
        let mut topics: Vec<TopicEntry> = tag_counter
            .into_iter()
            .map(|(tag, document_count)| TopicEntry {
                tag,
                document_count,
            })
            .collect();
        topics.sort_by(|a, b| {
            b.document_count
                .cmp(&a.document_count)
                .then_with(|| a.tag.cmp(&b.tag))
        });
        Ok(topics)
    }
}

/// Predicate: does this document satisfy every active filter?
fn document_matches_filter(doc: &DocumentRecord, filter: &SearchFilter) -> bool {
    if filter
        .collection
        .as_ref()
        .is_some_and(|c| doc.collection != *c)
    {
        return false;
    }
    if filter.doc_type.as_ref().is_some_and(|t| doc.doc_type != *t) {
        return false;
    }
    // Path filters target the source key (e.g. `03_PROJECTS/HEIMDALL/foo.md`),
    // not the internal canonical path.
    if !filter.path_includes.is_empty()
        && !filter
            .path_includes
            .iter()
            .any(|needle| doc.key.contains(needle))
    {
        return false;
    }
    if filter
        .path_excludes
        .iter()
        .any(|needle| doc.key.contains(needle))
    {
        return false;
    }
    if let Some(since) = &filter.since {
        // Docs without a parseable source timestamp are dropped.
        match doc
            .source_updated_at
            .as_deref()
            .map(DateTime::parse_from_rfc3339)
        {
            Some(Ok(t)) if t.with_timezone(&Utc) >= *since => {}
            _ => return false,
        }
    }
    if !filter.frontmatter.is_empty() {
        let parsed: JsonValue = serde_json::from_str(&doc.frontmatter_json)
            .unwrap_or(JsonValue::Object(JsonMap::new()));
        let Some(map) = parsed.as_object() else {
            return false;
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

/// `file` is the source file, or the absolute canonical markdown for
/// documents without one (e.g. fetched test cases), so it can be opened.
fn result_for(
    hit: SearchHit,
    doc: &DocumentRecord,
    mode: SearchMode,
    canonical_dir: &std::path::Path,
) -> SearchResult {
    SearchResult {
        text: hit.text,
        file: doc
            .source_path
            .clone()
            .unwrap_or_else(|| canonical_dir.join(&doc.markdown_path).display().to_string()),
        title: doc.title.clone(),
        doc_type: doc.doc_type.clone(),
        collection: doc.collection.clone(),
        score: hit.score,
        search_mode: mode,
        chunk_count: None,
        frontmatter: None,
    }
}

/// Chunk-level results, skipping exact-text duplicates within a file.
fn chunk_level_results(
    filtered: Vec<(SearchHit, &DocumentRecord)>,
    mode: SearchMode,
    limit: usize,
    canonical_dir: &std::path::Path,
) -> Vec<SearchResult> {
    let mut deduped = Vec::new();
    let mut seen = HashSet::new();
    for (hit, doc) in filtered {
        let result = result_for(hit, doc, mode, canonical_dir);
        if seen.insert(format!("{}:{}", result.file, result.text)) {
            deduped.push(result);
        }
        if deduped.len() >= limit {
            break;
        }
    }
    deduped
}

/// One result per document: its best chunk, the number of matching chunks
/// and its frontmatter.
fn collapse_to_doc_results(
    filtered: Vec<(SearchHit, &DocumentRecord)>,
    mode: SearchMode,
    limit: usize,
    canonical_dir: &std::path::Path,
) -> Vec<SearchResult> {
    let mut best: HashMap<String, (SearchHit, &DocumentRecord, usize)> = HashMap::new();
    for (hit, doc) in filtered {
        let entry = best
            .entry(hit.doc_id.clone())
            .or_insert_with(|| (hit.clone(), doc, 0));
        entry.2 += 1;
        if hit.score > entry.0.score {
            entry.0 = hit;
        }
    }
    let mut results: Vec<SearchResult> = best
        .into_values()
        .map(|(hit, doc, count)| {
            let mut result = result_for(hit, doc, mode, canonical_dir);
            result.chunk_count = Some(count);
            result.frontmatter = serde_json::from_str::<JsonValue>(&doc.frontmatter_json)
                .ok()
                .and_then(|v| v.as_object().cloned());
            result
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
    fn search_mode_parsing_and_display() {
        assert_eq!(SearchMode::default(), SearchMode::Hybrid);
        assert_eq!("HYBRID".parse::<SearchMode>().unwrap(), SearchMode::Hybrid);
        assert_eq!(
            "Semantic".parse::<SearchMode>().unwrap(),
            SearchMode::Semantic
        );
        assert_eq!(
            "keyword".parse::<SearchMode>().unwrap(),
            SearchMode::Keyword
        );
        let err = "fuzzy".parse::<SearchMode>().unwrap_err();
        assert!(err.contains("Invalid search mode") && err.contains("fuzzy"));
        assert_eq!(SearchMode::Semantic.to_string(), "semantic");
    }

    fn make_doc(
        doc_id: &str,
        key: &str,
        collection: &str,
        frontmatter_json: &str,
    ) -> DocumentRecord {
        let mut doc = DocumentRecord::new(doc_id, collection, key);
        doc.title = doc_id.into();
        doc.doc_type = "note".into();
        doc.content_hash = "sha256:00".into();
        doc.markdown_path = format!("{collection}/{doc_id}.md");
        doc.frontmatter_json = frontmatter_json.into();
        doc.source_updated_at = Some("2026-04-15T00:00:00Z".into());
        doc
    }

    fn hit(doc_id: &str, text: &str, score: f64) -> SearchHit {
        SearchHit {
            doc_id: doc_id.into(),
            text: text.into(),
            score,
        }
    }

    #[test]
    fn filter_default_matches_everything() {
        let doc = make_doc("d1", "a.md", "vault", "{}");
        assert!(document_matches_filter(&doc, &SearchFilter::default()));
    }

    // AC-007.1 (filter part): collection and doc_type filters.
    #[test]
    fn filter_collection_and_doc_type() {
        let doc = make_doc("d1", "a.md", "books", "{}");
        let mut f = SearchFilter {
            collection: Some("books".into()),
            ..Default::default()
        };
        assert!(document_matches_filter(&doc, &f));
        f.collection = Some("vault".into());
        assert!(!document_matches_filter(&doc, &f));

        let mut f = SearchFilter {
            doc_type: Some("note".into()),
            ..Default::default()
        };
        assert!(document_matches_filter(&doc, &f));
        f.doc_type = Some("book".into());
        assert!(!document_matches_filter(&doc, &f));
    }

    #[test]
    fn filter_paths_target_source_key() {
        let doc = make_doc("d1", "03_MY_PROJECTS/02_HEIMDALL/foo.md", "vault", "{}");
        let mut f = SearchFilter {
            path_includes: vec!["02_HEIMDALL".into()],
            ..Default::default()
        };
        assert!(document_matches_filter(&doc, &f));
        f.path_includes = vec!["03_GO_AI".into()];
        assert!(!document_matches_filter(&doc, &f));
        f.path_includes = vec!["03_GO_AI".into(), "HEIMDALL".into()];
        assert!(document_matches_filter(&doc, &f));

        let f = SearchFilter {
            path_excludes: vec!["03_MY".into()],
            ..Default::default()
        };
        assert!(!document_matches_filter(&doc, &f));
    }

    #[test]
    fn filter_frontmatter_and_since() {
        let doc = make_doc("d1", "a.md", "vault", r#"{"area":"SIT","n":3,"ok":true}"#);
        let mut f = SearchFilter::default();
        f.frontmatter.insert("area".into(), "SIT".into());
        f.frontmatter.insert("n".into(), "3".into());
        f.frontmatter.insert("ok".into(), "true".into());
        assert!(document_matches_filter(&doc, &f));
        f.frontmatter.insert("status".into(), "draft".into());
        assert!(!document_matches_filter(&doc, &f));

        let at = |s: &str| DateTime::parse_from_rfc3339(s).unwrap().with_timezone(&Utc);
        let mut f = SearchFilter {
            since: Some(at("2026-04-01T00:00:00Z")),
            ..Default::default()
        };
        assert!(document_matches_filter(&doc, &f));
        f.since = Some(at("2026-05-01T00:00:00Z"));
        assert!(!document_matches_filter(&doc, &f));
        let mut undated = doc.clone();
        undated.source_updated_at = None;
        assert!(!document_matches_filter(&undated, &f));
    }

    #[test]
    fn collapse_to_doc_keeps_best_chunk_counts_and_frontmatter() {
        let mut d1 = make_doc("d1", "p1.md", "vault", r#"{"area":"SIT"}"#);
        d1.source_path = Some("/src/p1.md".into());
        let d2 = make_doc("d2", "p2.md", "vault", "{}");
        let filtered = vec![
            (hit("d1", "low", 0.5), &d1),
            (hit("d1", "high", 0.9), &d1),
            (hit("d2", "mid", 0.7), &d2),
        ];
        let results = collapse_to_doc_results(filtered, SearchMode::Hybrid, 10, Path::new("/c"));
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].file, "/src/p1.md");
        assert_eq!(results[0].text, "high");
        assert_eq!(results[0].chunk_count, Some(2));
        assert_eq!(results[0].collection, "vault");
        assert_eq!(
            results[0]
                .frontmatter
                .as_ref()
                .and_then(|m| m.get("area"))
                .and_then(JsonValue::as_str),
            Some("SIT")
        );
        // No source path: the absolute canonical file.
        assert_eq!(results[1].file, "/c/vault/d2.md");
        assert_eq!(results[1].chunk_count, Some(1));
    }

    #[test]
    fn collapse_truncates_and_chunk_level_dedups() {
        let docs: Vec<DocumentRecord> = (0..5)
            .map(|i| make_doc(&format!("d{i}"), &format!("p{i}.md"), "vault", "{}"))
            .collect();
        let filtered: Vec<(SearchHit, &DocumentRecord)> = docs
            .iter()
            .enumerate()
            .map(|(i, d)| (hit(&format!("d{i}"), "same", 0.9 - i as f64 * 0.01), d))
            .collect();
        assert_eq!(
            collapse_to_doc_results(filtered, SearchMode::Hybrid, 3, Path::new("/c")).len(),
            3
        );

        let d = &docs[0];
        let dupes = vec![(hit("d0", "x", 0.9), d), (hit("d0", "x", 0.8), d)];
        assert_eq!(
            chunk_level_results(dupes, SearchMode::Keyword, 10, Path::new("/c")).len(),
            1
        );
    }

    // -- end-to-end over a real (temp) index --------------------------------

    use crate::indexer::tests::FakeEmbedder;
    use crate::indexer::{index_library, IndexOptions};
    use std::collections::BTreeMap as Snapshot;
    use std::path::{Path, PathBuf};

    fn snapshot(root: &Path) -> Snapshot<PathBuf, (u64, std::time::SystemTime)> {
        let mut out = Snapshot::new();
        let mut stack = vec![root.to_path_buf()];
        while let Some(dir) = stack.pop() {
            for entry in std::fs::read_dir(&dir).unwrap() {
                let path = entry.unwrap().path();
                let meta = std::fs::metadata(&path).unwrap();
                if meta.is_dir() {
                    stack.push(path.clone());
                }
                out.insert(path, (meta.len(), meta.modified().unwrap()));
            }
        }
        out
    }

    async fn build_index(config: &AppConfig) {
        let (_lock, store) = config.open_for_write().unwrap();
        let docs = [
            (
                "books",
                "book",
                "Pro Git",
                r#"["Scott Chacon","Ben Straub"]"#,
                "alpha branching and merging",
            ),
            ("vault", "note", "Note", "[]", "alpha meeting notes"),
        ];
        for (collection, doc_type, title, authors, text) in docs {
            let key = format!("{title}.md");
            let mut doc = DocumentRecord::new(format!("{collection}:{key}"), collection, &key);
            doc.title = title.into();
            doc.doc_type = doc_type.into();
            doc.authors_json = authors.into();
            doc.tags_json = r#"["t1"]"#.into();
            doc.source_path = Some(format!("/src/{key}"));
            doc.content_hash = crate::canonical_store::content_hash(text);
            doc.markdown_path = format!("{collection}/{title}.md");
            let path = config.canonical_dir.join(&doc.markdown_path);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, text).unwrap();
            store.upsert_document(&doc).unwrap();
        }
        index_library(
            config,
            &store,
            &FakeEmbedder::new(),
            &IndexOptions::default(),
            |_| {},
        )
        .await
        .unwrap();
    }

    // AC-002.1, AC-007.1, AC-008.2
    #[tokio::test]
    async fn read_paths_filter_by_collection_and_leave_data_dir_untouched() {
        let dir = tempfile::TempDir::new().unwrap();
        let config = AppConfig::for_test(dir.path());
        build_index(&config).await;
        let before = snapshot(dir.path());

        let engine = SearchEngine::new(&config).await.unwrap();
        let all = engine
            .search(
                "alpha",
                &SearchFilter::default(),
                true,
                10,
                SearchMode::Keyword,
            )
            .await
            .unwrap();
        assert_eq!(all.len(), 2);

        let books_only = SearchFilter {
            collection: Some("books".into()),
            ..Default::default()
        };
        let hits = engine
            .search("alpha", &books_only, true, 10, SearchMode::Keyword)
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].collection, "books");
        assert_eq!(hits[0].file, "/src/Pro Git.md");

        let books = engine.list_books().await.unwrap();
        assert_eq!(books.len(), 1);
        assert_eq!(books[0].title, "Pro Git");
        assert_eq!(books[0].authors, ["Scott Chacon", "Ben Straub"]);
        assert_eq!(books[0].source_path.as_deref(), Some("/src/Pro Git.md"));
        assert_eq!(books[0].chunks, 1);
        let json = serde_json::to_value(&books[0]).unwrap();
        let mut keys: Vec<&String> = json.as_object().unwrap().keys().collect();
        keys.sort();
        assert_eq!(keys, ["authors", "chunks", "source_path", "title"]);

        let topics = engine.browse_topics(Some("vault")).await.unwrap();
        assert_eq!(topics.len(), 1);
        assert_eq!(topics[0].document_count, 1);

        assert_eq!(snapshot(dir.path()), before, "read paths must not write");
    }
}
