//! Library: books are added once and kept.
//!
//! A folder holding `metadata.opf` is one calibre book, identified by the
//! calibre UUID; one format is chosen by preference. Any other book file is
//! identified by the SHA-256 of its bytes. Books are never deleted
//! automatically: a book whose file disappears stays searchable and is
//! marked `source_missing`; only `colibri remove` takes a book out, and a
//! sweep does not bring a removed book back.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

use chrono::Utc;
use serde::Serialize;
use serde_json::{json, Map, Value};

use crate::canonical_store::{canonical_rel_path, doc_id_for, same_except_timestamps};
use crate::config::{AppConfig, LibraryConfig, BOOK_FORMATS, LIBRARY_COLLECTION};
use crate::envelope::content_hash;
use crate::error::ColibriError;
use crate::ingest::calibre::{parse_opf, BookMeta};
use crate::ingest::convert::{clear_cached, convert_cached, file_sha256, Converter};
use crate::ingest::mirror::{default_title, rfc3339, Problem};
use crate::ingest::walk::{walk, Filter, FoundFile};
use crate::metadata_store::{DocStatus, DocumentRecord, MetadataStore};

/// `removed_reason` for books taken out with `colibri remove`.
pub const REMOVED_BY_USER: &str = "user";

#[derive(Debug, Clone, Copy, Default)]
pub struct LibraryOptions {
    pub dry_run: bool,
    /// Convert again even if a cached conversion exists.
    pub reconvert: bool,
    /// Retry conversions that failed in an earlier run.
    pub retry_failed: bool,
}

/// What happened to one book.
#[derive(Debug, Clone, Serialize)]
pub struct BookOutcome {
    pub title: String,
    pub path: String,
    /// added | updated | restored | already_known | skipped_removed | problem
    pub outcome: String,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct LibraryReport {
    pub name: String,
    /// `ok`, `partial` (problems) or `error` (a root could not be read at all).
    pub status: String,
    pub dry_run: bool,
    pub added: usize,
    pub updated: usize,
    pub unchanged: usize,
    pub restored: usize,
    pub skipped_removed: usize,
    /// Books whose source file is gone (still searchable).
    pub source_missing: usize,
    /// Every book that was not unchanged.
    pub books: Vec<BookOutcome>,
    pub problems: Vec<Problem>,
}

/// One book found on disk.
struct Candidate {
    /// `calibre:<uuid>` or `sha:<first 16 hex of SHA-256>`.
    key: String,
    file: FoundFile,
    /// Lowercase extension with dot.
    ext: String,
    /// From `metadata.opf`; `None` for loose files (keep stored metadata).
    meta: Option<BookMeta>,
    sha256: Option<String>,
    /// The user named this exact file: keep this format.
    pinned: bool,
    explicit: bool,
}

fn ext_of(path: &Path) -> String {
    path.extension()
        .map(|e| format!(".{}", e.to_string_lossy().to_lowercase()))
        .unwrap_or_default()
}

fn is_book_ext(ext: &str) -> bool {
    BOOK_FORMATS.contains(&ext.trim_start_matches('.'))
}

fn book_filter() -> Filter {
    let include: Vec<String> = BOOK_FORMATS
        .iter()
        .map(|f| format!("**/*.{f}"))
        .chain(["**/*.acsm".to_string()])
        .collect();
    // calibre keeps `.caltrash` and `.calnotes` next to the books.
    Filter::new(&include, &[".*/**".into(), "**/.*/**".into()])
}

fn found_file(path: &Path, rel: String) -> Result<FoundFile, String> {
    let meta = std::fs::metadata(path).map_err(|e| e.to_string())?;
    if !meta.is_file() {
        return Err("not a file".into());
    }
    Ok(FoundFile {
        rel,
        abs: path.to_path_buf(),
        size: meta.len(),
        mtime: meta.modified().unwrap_or(std::time::SystemTime::UNIX_EPOCH),
        online_only: false,
    })
}

/// Build candidates from files grouped by folder.
struct Collector<'a> {
    library: &'a LibraryConfig,
    store: Option<&'a MetadataStore>,
    /// Existing books by source path, to reuse identities without hashing.
    by_path: HashMap<String, DocumentRecord>,
    problems: Vec<Problem>,
}

impl Collector<'_> {
    /// Identity of a file outside a calibre book folder. A file already
    /// tracked at this path keeps its identity even if its bytes changed, so
    /// edits go through the normal change rules instead of creating a second
    /// book. New files are identified by content (which means reading them).
    fn loose_key(&mut self, file: &FoundFile) -> Option<(String, Option<String>)> {
        if let Some(d) = self.by_path.get(&*file.abs.to_string_lossy()) {
            return Some((d.key.clone(), None));
        }
        if file.online_only {
            self.problems.push(Problem::new(
                file.abs.display().to_string(),
                "online_only",
                "file is only in the cloud; make it available offline to add it",
            ));
            return None;
        }
        match file_sha256(&file.abs) {
            Ok(sha) => Some((format!("sha:{}", &sha[..16]), Some(sha))),
            Err(e) => {
                self.problems.push(Problem::new(
                    file.abs.display().to_string(),
                    "unreadable",
                    e.to_string(),
                ));
                None
            }
        }
    }

    fn preferred_format(&self, key: &str, available: &[&FoundFile]) -> Option<usize> {
        // Keep a format the user pinned with an explicit add.
        if let Some(store) = self.store {
            if let Ok(Some(d)) = store.get_document(&doc_id_for(LIBRARY_COLLECTION, key)) {
                if d.format_pinned {
                    if let Some(i) = available.iter().position(|f| {
                        Some(ext_of(&f.abs).trim_start_matches('.')) == d.format.as_deref()
                    }) {
                        return Some(i);
                    }
                }
            }
        }
        for pref in &self.library.prefer_formats {
            if let Some(i) = available
                .iter()
                .position(|f| ext_of(&f.abs).trim_start_matches('.') == pref)
            {
                return Some(i);
            }
        }
        (!available.is_empty()).then_some(0)
    }

    /// `explicit_file`: the user named this file (it becomes the chosen, pinned format).
    fn candidates_in(
        &mut self,
        dir: &Path,
        files: Vec<FoundFile>,
        explicit: bool,
        explicit_file: Option<&Path>,
    ) -> Vec<Candidate> {
        let has_acsm = files.iter().any(|f| ext_of(&f.abs) == ".acsm");
        let books: Vec<FoundFile> = files
            .into_iter()
            .filter(|f| is_book_ext(&ext_of(&f.abs)))
            .collect();

        let opf = dir.join("metadata.opf");
        let meta = if opf.is_file() {
            // A calibre folder whose metadata cannot be used is skipped as a
            // whole: treating its formats as loose books would split the book.
            match parse_opf(&opf) {
                Ok(m) if m.uuid.is_some() => Some(m),
                Ok(_) => {
                    self.problems.push(Problem::new(
                        opf.display().to_string(),
                        "bad_metadata",
                        "metadata.opf has no calibre uuid; folder skipped",
                    ));
                    return Vec::new();
                }
                Err(e) => {
                    self.problems.push(Problem::new(
                        opf.display().to_string(),
                        "bad_metadata",
                        format!("{e}; folder skipped"),
                    ));
                    return Vec::new();
                }
            }
        } else {
            None
        };

        if let Some(meta) = meta {
            let key = format!("calibre:{}", meta.uuid.clone().unwrap_or_default());
            if books.is_empty() {
                if has_acsm {
                    self.problems.push(Problem::new(
                        dir.display().to_string(),
                        "drm_placeholder",
                        "only an Adobe DRM .acsm link, no readable book file",
                    ));
                }
                return Vec::new();
            }
            let refs: Vec<&FoundFile> = books.iter().collect();
            let chosen = match explicit_file {
                Some(named) => refs.iter().position(|f| f.abs == named),
                None => self.preferred_format(&key, &refs),
            };
            let Some(i) = chosen else { return Vec::new() };
            let file = books[i].clone();
            return vec![Candidate {
                key,
                ext: ext_of(&file.abs),
                file,
                meta: Some(meta),
                sha256: None,
                pinned: explicit_file.is_some(),
                explicit,
            }];
        }

        if has_acsm && books.is_empty() {
            self.problems.push(Problem::new(
                dir.display().to_string(),
                "drm_placeholder",
                "only an Adobe DRM .acsm link, no readable book file",
            ));
        }
        let mut out = Vec::new();
        for file in books {
            if explicit_file.is_some_and(|named| named != file.abs) {
                continue;
            }
            if let Some((key, sha)) = self.loose_key(&file) {
                out.push(Candidate {
                    key,
                    ext: ext_of(&file.abs),
                    meta: None,
                    sha256: sha,
                    pinned: explicit_file.is_some(),
                    explicit,
                    file,
                });
            }
        }
        out
    }
}

/// Same file system object, ignoring symlinks and `/var` vs `/private/var`.
fn same_path(a: &Path, b: &Path) -> bool {
    a == b || matches!((a.canonicalize(), b.canonicalize()), (Ok(x), Ok(y)) if x == y)
}

fn group_by_dir(files: Vec<FoundFile>) -> BTreeMap<PathBuf, Vec<FoundFile>> {
    let mut groups: BTreeMap<PathBuf, Vec<FoundFile>> = BTreeMap::new();
    for f in files {
        let dir = f.abs.parent().map(Path::to_path_buf).unwrap_or_default();
        groups.entry(dir).or_default().push(f);
    }
    groups
}

/// Frontmatter-style metadata so `--frontmatter author=...` style filters work on books.
fn book_frontmatter(meta: &BookMeta, format: &str) -> String {
    let mut m = Map::new();
    let mut put = |k: &str, v: Value| {
        if !(v.is_null() || v.as_array().is_some_and(|a| a.is_empty())) {
            m.insert(k.into(), v);
        }
    };
    put("authors", json!(meta.authors));
    put("tags", json!(meta.tags));
    put("language", json!(meta.language));
    put("publisher", json!(meta.publisher));
    put("date", json!(meta.date));
    put("isbn", json!(meta.isbn));
    put("calibre_id", json!(meta.calibre_id));
    put("format", json!(format));
    Value::Object(m).to_string()
}

struct Ctx<'a> {
    config: &'a AppConfig,
    library: &'a LibraryConfig,
    store: Option<&'a MetadataStore>,
    converter: &'a dyn Converter,
    opts: LibraryOptions,
    /// Cache entries already cleared by `--reconvert` in this run.
    cleared: std::cell::RefCell<HashSet<String>>,
}

fn outcome(report: &mut LibraryReport, title: &str, path: &Path, what: &str) {
    report.books.push(BookOutcome {
        title: title.to_string(),
        path: path.display().to_string(),
        outcome: what.to_string(),
    });
}

/// Ingest or update one book.
fn process(ctx: &Ctx<'_>, c: Candidate, report: &mut LibraryReport) -> Result<(), ColibriError> {
    let doc_id = doc_id_for(LIBRARY_COLLECTION, &c.key);
    let prev = match ctx.store {
        Some(s) => s.get_document(&doc_id)?,
        None => None,
    };
    let title = c
        .meta
        .as_ref()
        .and_then(|m| m.title.clone())
        .or_else(|| prev.as_ref().map(|d| d.title.clone()))
        .unwrap_or_else(|| default_title(&c.file.abs));
    let removed_by_user = prev.as_ref().is_some_and(|d| {
        !d.is_searchable() && d.removed_reason.as_deref() == Some(REMOVED_BY_USER)
    });
    if removed_by_user && !c.explicit {
        report.skipped_removed += 1;
        return Ok(());
    }

    // The same book in another folder while the original still exists:
    // keep the original as source. Formats of one calibre book share a folder.
    if let Some(d) = prev.as_ref().filter(|d| d.is_searchable()) {
        let original = d
            .source_path
            .as_deref()
            .map(Path::new)
            .filter(|p| p.exists() && !same_path(p, &c.file.abs))
            .filter(|p| {
                !(c.key.starts_with("calibre:")
                    && same_path(p.parent().unwrap_or(p), c.file.abs.parent().unwrap_or(p)))
            });
        if let Some(original) = original {
            if c.explicit {
                report.unchanged += 1;
                outcome(report, &title, &c.file.abs, "already_known");
            } else {
                report.problems.push(Problem::new(
                    c.file.abs.display().to_string(),
                    "duplicate",
                    format!("same book as {}", original.display()),
                ));
            }
            return Ok(());
        }
    }

    let format = c.ext.trim_start_matches('.').to_string();
    let mtime = rfc3339(c.file.mtime);
    let same_stat = prev.as_ref().is_some_and(|d| {
        d.source_size == Some(c.file.size as i64)
            && d.source_mtime.as_deref() == Some(mtime.as_str())
            && d.format.as_deref() == Some(format.as_str())
    });
    let canonical_missing = prev.as_ref().is_some_and(|d| {
        d.markdown_path.is_empty() || !ctx.config.canonical_dir.join(&d.markdown_path).exists()
    });
    let needs_read = prev.is_none() || ctx.opts.reconvert || !same_stat || canonical_missing;
    if c.file.online_only && needs_read {
        report.problems.push(Problem::new(
            c.file.abs.display().to_string(),
            "online_only",
            "file is only in the cloud; make it available offline to add it",
        ));
        return Ok(());
    }

    // Decide whether the content has to be (re)converted.
    let mut sha = c.sha256.clone();
    let mut convert = prev.is_none() || ctx.opts.reconvert || canonical_missing;
    if let (Some(d), false) = (&prev, convert || same_stat) {
        if sha.is_none() {
            sha = Some(file_sha256(&c.file.abs)?);
        }
        let same_bytes = sha.as_deref() == d.source_sha256.as_deref()
            && d.format.as_deref() == Some(format.as_str());
        if !same_bytes {
            if d.format.as_deref() == Some(format.as_str()) && c.ext == ".pdf" {
                // Re-running docling is slow and not deterministic; ask first.
                report.problems.push(Problem::new(
                    c.file.abs.display().to_string(),
                    "source_changed",
                    "the PDF changed; run `colibri add <file> --reconvert` to update it",
                ));
                return Ok(());
            }
            convert = true;
        }
    }

    let mut markdown = None;
    let mut tool = prev.as_ref().and_then(|d| d.converter.clone());
    if convert && !ctx.opts.dry_run {
        let store = ctx.store.expect("writes need a store");
        let sha_value = match sha.clone() {
            Some(s) => s,
            None => file_sha256(&c.file.abs)?,
        };
        if ctx.opts.reconvert && ctx.cleared.borrow_mut().insert(sha_value.clone()) {
            clear_cached(store, &ctx.config.conversions_dir, &sha_value)
                .map_err(ColibriError::Index)?;
        }
        match convert_cached(
            store,
            &ctx.config.conversions_dir,
            ctx.converter,
            &c.ext,
            &c.file.abs,
            &sha_value,
            ctx.opts.retry_failed || ctx.opts.reconvert,
        ) {
            Ok((md, t)) => {
                markdown = Some(md);
                tool = Some(t);
            }
            Err(e) => {
                report.problems.push(Problem::new(
                    c.file.abs.display().to_string(),
                    "conversion_failed",
                    e,
                ));
                outcome(report, &title, &c.file.abs, "problem");
                return Ok(());
            }
        }
        sha = Some(sha_value);
    }

    let mut doc = prev
        .clone()
        .unwrap_or_else(|| DocumentRecord::new(&doc_id, LIBRARY_COLLECTION, &c.key));
    doc.key = c.key.clone();
    doc.title = title.clone();
    if let Some(meta) = &c.meta {
        doc.authors_json = serde_json::to_string(&meta.authors)?;
        doc.tags_json = serde_json::to_string(&meta.tags)?;
        doc.language = meta.language.clone();
        doc.frontmatter_json = book_frontmatter(meta, &format);
    } else if prev.is_none() {
        doc.frontmatter_json = book_frontmatter(&BookMeta::default(), &format);
    }
    doc.doc_type = ctx.library.doc_type.clone();
    doc.format = Some(format.clone());
    if c.pinned {
        doc.format_pinned = true;
    }
    doc.converter = tool;
    // Keep the stored path string when it names the same file (stable output).
    let keep_path = prev
        .as_ref()
        .and_then(|d| d.source_path.as_deref())
        .is_some_and(|p| same_path(Path::new(p), &c.file.abs));
    if !keep_path {
        doc.source_path = Some(c.file.abs.display().to_string());
    }
    doc.source_size = Some(c.file.size as i64);
    doc.source_mtime = Some(mtime.clone());
    if sha.is_some() {
        doc.source_sha256 = sha;
    }
    doc.source_updated_at = Some(mtime);
    doc.status = DocStatus::Active;
    doc.removed_reason = None;
    doc.source_missing = false;
    if let Some(md) = &markdown {
        let hash = content_hash(md);
        if canonical_missing
            || prev.as_ref().map(|d| d.content_hash.as_str()) != Some(hash.as_str())
        {
            let rel = canonical_rel_path(LIBRARY_COLLECTION, &doc_id)
                .to_string_lossy()
                .to_string();
            let abs = ctx.config.canonical_dir.join(&rel);
            std::fs::create_dir_all(abs.parent().unwrap_or(&ctx.config.canonical_dir))?;
            std::fs::write(&abs, md)?;
            doc.content_hash = hash;
            doc.markdown_path = rel;
        }
    }

    let what = match &prev {
        None => "added",
        Some(_) if removed_by_user => "restored",
        Some(_) if convert && ctx.opts.dry_run => "updated",
        Some(p) if same_except_timestamps(p, &doc) => {
            if c.explicit {
                "already_known"
            } else {
                report.unchanged += 1;
                return Ok(());
            }
        }
        Some(_) => "updated",
    };
    if what != "already_known" && !ctx.opts.dry_run {
        let now = Utc::now().to_rfc3339();
        doc.updated_at = now.clone();
        doc.last_seen_at = Some(now);
        ctx.store
            .expect("writes need a store")
            .upsert_document(&doc)?;
    }
    count(report, what);
    outcome(report, &title, &c.file.abs, what);
    Ok(())
}

fn count(report: &mut LibraryReport, what: &str) {
    match what {
        "added" => report.added += 1,
        "restored" => report.restored += 1,
        "updated" => report.updated += 1,
        _ => report.unchanged += 1,
    }
}

fn existing_books(store: Option<&MetadataStore>) -> Result<Vec<DocumentRecord>, ColibriError> {
    match store {
        Some(s) => s.list_documents_in(LIBRARY_COLLECTION),
        None => Ok(Vec::new()),
    }
}

/// Sweep the configured roots: add new books, update changed ones, mark
/// books whose file is gone as source-missing. Never removes a book.
pub fn sweep(
    config: &AppConfig,
    library: &LibraryConfig,
    store: Option<&MetadataStore>,
    converter: &dyn Converter,
    opts: LibraryOptions,
) -> Result<LibraryReport, ColibriError> {
    let mut report = LibraryReport {
        name: LIBRARY_COLLECTION.into(),
        dry_run: opts.dry_run,
        ..Default::default()
    };
    if !opts.dry_run && store.is_none() {
        return Err(ColibriError::Config(
            "library sweep needs a writable metadata DB".into(),
        ));
    }
    let existing = existing_books(store)?;
    let mut collector = Collector {
        library,
        store,
        by_path: existing
            .iter()
            .filter_map(|d| d.source_path.clone().map(|p| (p, d.clone())))
            .collect(),
        problems: Vec::new(),
    };
    let ctx = Ctx {
        config,
        library,
        store,
        converter,
        opts,
        cleared: Default::default(),
    };

    let mut root_error = false;
    let mut seen: HashSet<String> = HashSet::new();
    // Missing roots and unreadable folders: their books are not judged missing.
    let mut unavailable: Vec<PathBuf> = Vec::new();
    for root in &library.roots {
        if !root.is_dir() {
            root_error = true;
            unavailable.push(root.clone());
            report.problems.push(Problem::new(
                root.display().to_string(),
                "root_missing",
                "library folder does not exist; books from it stay searchable",
            ));
            continue;
        }
        let walked = walk(root, &book_filter());
        unavailable.extend(walked.unreadable.iter().map(|(rel, _)| root.join(rel)));
        for (rel, message) in walked.unreadable.iter().chain(walked.notices.iter()) {
            report.problems.push(Problem::new(
                root.join(rel).display().to_string(),
                "unreadable",
                message.clone(),
            ));
        }
        for (dir, files) in group_by_dir(walked.files) {
            for candidate in collector.candidates_in(&dir, files, false, None) {
                seen.insert(candidate.key.clone());
                process(&ctx, candidate, &mut report)?;
            }
        }
    }
    report.problems.append(&mut collector.problems);

    // Books not found in this sweep: still searchable, flagged if their file is gone.
    for mut doc in existing.into_iter().filter(|d| d.is_searchable()) {
        let in_unavailable = doc
            .source_path
            .as_deref()
            .is_some_and(|p| unavailable.iter().any(|u| Path::new(p).starts_with(u)));
        if seen.contains(&doc.key) || in_unavailable {
            continue;
        }
        let missing = doc
            .source_path
            .as_deref()
            .is_none_or(|p| !Path::new(p).exists());
        if missing != doc.source_missing && !opts.dry_run {
            doc.source_missing = missing;
            doc.updated_at = Utc::now().to_rfc3339();
            store.expect("checked").upsert_document(&doc)?;
        }
        if missing {
            report.source_missing += 1;
        }
    }

    report.status = if root_error {
        "error".into()
    } else if report.problems.is_empty() {
        "ok".into()
    } else {
        "partial".into()
    };
    if let (Some(store), false) = (store, opts.dry_run) {
        let problems: Vec<(String, String, String)> = report
            .problems
            .iter()
            .map(|p| (p.source_path.clone(), p.kind.clone(), p.message.clone()))
            .collect();
        store.replace_problems(LIBRARY_COLLECTION, &problems)?;
        let roots: Vec<String> = library
            .roots
            .iter()
            .map(|r| r.display().to_string())
            .collect();
        store.record_collection_run(
            LIBRARY_COLLECTION,
            "library",
            Some(&roots.join(", ")),
            &report.status,
            &serde_json::to_string(&report)?,
        )?;
    }
    Ok(report)
}

/// Add the given book files or folders. Restores books the user removed.
pub fn add_paths(
    config: &AppConfig,
    library: &LibraryConfig,
    store: Option<&MetadataStore>,
    converter: &dyn Converter,
    paths: &[PathBuf],
    opts: LibraryOptions,
) -> Result<LibraryReport, ColibriError> {
    let mut report = LibraryReport {
        name: LIBRARY_COLLECTION.into(),
        dry_run: opts.dry_run,
        ..Default::default()
    };
    let existing = existing_books(store)?;
    let mut collector = Collector {
        library,
        store,
        by_path: existing
            .iter()
            .filter_map(|d| d.source_path.clone().map(|p| (p, d.clone())))
            .collect(),
        problems: Vec::new(),
    };
    let ctx = Ctx {
        config,
        library,
        store,
        converter,
        opts,
        cleared: Default::default(),
    };

    for path in paths {
        // Absolute but not symlink-resolved, so paths match what a sweep stores.
        let path = std::path::absolute(path).unwrap_or_else(|_| path.clone());
        if path.is_dir() {
            let walked = walk(&path, &book_filter());
            for (dir, files) in group_by_dir(walked.files) {
                for c in collector.candidates_in(&dir, files, true, None) {
                    process(&ctx, c, &mut report)?;
                }
            }
            continue;
        }
        let ext = ext_of(&path);
        if !is_book_ext(&ext) {
            report.problems.push(Problem::new(
                path.display().to_string(),
                "unsupported_type",
                format!("books must be one of: {}", BOOK_FORMATS.join(", ")),
            ));
            continue;
        }
        let dir = path.parent().map(Path::to_path_buf).unwrap_or_default();
        let named = match found_file(&path, path.display().to_string()) {
            Ok(f) => f,
            Err(e) => {
                report
                    .problems
                    .push(Problem::new(path.display().to_string(), "not_found", e));
                continue;
            }
        };
        for c in collector.candidates_in(&dir, vec![named], true, Some(&path)) {
            process(&ctx, c, &mut report)?;
        }
    }
    report.problems.append(&mut collector.problems);
    report.status = if report.problems.is_empty() {
        "ok"
    } else {
        "partial"
    }
    .into();
    Ok(report)
}

/// Searchable books matching an id, key or title substring (case-insensitive).
pub fn find_books(store: &MetadataStore, query: &str) -> Result<Vec<DocumentRecord>, ColibriError> {
    let q = query.trim().to_lowercase();
    let books: Vec<DocumentRecord> = store
        .list_documents_in(LIBRARY_COLLECTION)?
        .into_iter()
        .filter(|d| d.is_searchable())
        .collect();
    if let Some(exact) = books.iter().find(|d| d.doc_id == query || d.key == query) {
        return Ok(vec![exact.clone()]);
    }
    Ok(books
        .into_iter()
        .filter(|d| d.title.to_lowercase().contains(&q))
        .collect())
}

/// Take a book out of search. Sweeps will not add it again; an explicit
/// `colibri add <file>` restores it (without converting again).
pub fn remove_book(store: &MetadataStore, doc_id: &str) -> Result<(), ColibriError> {
    store.set_status(doc_id, DocStatus::Removed, Some(REMOVED_BY_USER))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ingest::convert::fakes::CountingConverter;

    struct Lib {
        _dir: tempfile::TempDir,
        root: PathBuf,
        config: AppConfig,
        library: LibraryConfig,
    }

    fn lib() -> Lib {
        let dir = tempfile::TempDir::new().unwrap();
        let root = dir.path().join("calibre");
        std::fs::create_dir_all(&root).unwrap();
        let config = AppConfig::for_test(&dir.path().join("home"));
        let library = LibraryConfig {
            doc_type: "book".into(),
            prefer_formats: vec!["epub".into(), "pdf".into(), "docx".into()],
            roots: vec![root.clone()],
        };
        Lib {
            _dir: dir,
            root,
            config,
            library,
        }
    }

    fn opf(uuid: &str, title: &str, authors: &[&str]) -> String {
        let creators: String = authors
            .iter()
            .map(|a| format!("<dc:creator opf:role=\"aut\">{a}</dc:creator>"))
            .collect();
        format!(
            r#"<?xml version='1.0' encoding='utf-8'?>
<package xmlns="http://www.idpf.org/2007/opf" version="2.0">
<metadata xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:opf="http://www.idpf.org/2007/opf">
<dc:identifier opf:scheme="uuid">{uuid}</dc:identifier><dc:title>{title}</dc:title>{creators}
<dc:language>eng</dc:language><dc:subject>Git</dc:subject>
</metadata></package>"#
        )
    }

    impl Lib {
        /// calibre-style folder `<root>/<rel>/` with metadata.opf and book files.
        fn book(&self, rel: &str, uuid: &str, title: &str, files: &[(&str, &str)]) -> PathBuf {
            let dir = self.root.join(rel);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(
                dir.join("metadata.opf"),
                opf(uuid, title, &["Scott Chacon", "Ben Straub"]),
            )
            .unwrap();
            std::fs::write(dir.join("cover.jpg"), "jpg").unwrap();
            for (name, bytes) in files {
                std::fs::write(dir.join(name), bytes).unwrap();
            }
            dir
        }

        fn sweep(&self, store: &MetadataStore, conv: &CountingConverter) -> LibraryReport {
            sweep(
                &self.config,
                &self.library,
                Some(store),
                conv,
                LibraryOptions::default(),
            )
            .unwrap()
        }

        fn add(
            &self,
            store: &MetadataStore,
            conv: &CountingConverter,
            paths: &[PathBuf],
            opts: LibraryOptions,
        ) -> LibraryReport {
            add_paths(&self.config, &self.library, Some(store), conv, paths, opts).unwrap()
        }
    }

    // AC-017.1, AC-020.1
    #[test]
    fn sweep_adds_new_books_once_with_calibre_metadata() {
        let l = lib();
        let (_lock, store) = l.config.open_for_write().unwrap();
        let conv = CountingConverter::default();
        l.book(
            "Scott Chacon/Pro Git (90)",
            "uuid-progit",
            "Pro Git",
            &[("Pro Git.epub", "epub bytes")],
        );

        let r = l.sweep(&store, &conv);
        assert_eq!((r.added, r.unchanged), (1, 0));
        let doc = store
            .get_document("books:calibre:uuid-progit")
            .unwrap()
            .unwrap();
        assert_eq!(doc.title, "Pro Git");
        assert_eq!(doc.authors_json, r#"["Scott Chacon","Ben Straub"]"#);
        assert_eq!(doc.language.as_deref(), Some("eng"));
        assert_eq!(doc.doc_type, "book");
        assert!(doc.frontmatter_json.contains("\"language\":\"eng\""));

        let r = l.sweep(&store, &conv);
        assert_eq!((r.added, r.updated, r.unchanged), (0, 0, 1));
        assert_eq!(conv.calls(), 1);
    }

    // AC-017.2
    #[test]
    fn moved_calibre_book_keeps_identity_without_conversion() {
        let l = lib();
        let (_lock, store) = l.config.open_for_write().unwrap();
        let conv = CountingConverter::default();
        let dir = l.book("A/Book (1)", "uuid-1", "Book", &[("Book.epub", "bytes")]);
        l.sweep(&store, &conv);
        store
            .mark_indexed(
                "books:calibre:uuid-1",
                &store
                    .get_document("books:calibre:uuid-1")
                    .unwrap()
                    .unwrap()
                    .content_hash,
                1,
            )
            .unwrap();

        std::fs::create_dir_all(l.root.join("B")).unwrap();
        std::fs::rename(&dir, l.root.join("B/Book (1)")).unwrap();
        let r = l.sweep(&store, &conv);
        assert_eq!((r.added, r.updated), (0, 1));
        let doc = store.get_document("books:calibre:uuid-1").unwrap().unwrap();
        assert!(doc.source_path.as_deref().unwrap().contains("/B/Book (1)/"));
        assert!(doc.is_index_current(), "no re-embedding");
        assert_eq!(conv.calls(), 1);
    }

    // AC-017.3
    #[test]
    fn loose_files_are_identified_by_content() {
        let l = lib();
        let (_lock, store) = l.config.open_for_write().unwrap();
        let conv = CountingConverter::default();
        std::fs::write(l.root.join("loose.epub"), "loose bytes").unwrap();
        l.sweep(&store, &conv);
        let copy = l.root.parent().unwrap().join("copy.epub");
        std::fs::write(&copy, "loose bytes").unwrap();

        let r = l.add(&store, &conv, &[copy], LibraryOptions::default());
        assert_eq!(r.books[0].outcome, "already_known");
        assert_eq!(store.list_documents_in("books").unwrap().len(), 1);
        assert_eq!(conv.calls(), 1);
    }

    // AC-019.1, AC-019.2
    #[test]
    fn one_format_per_book_and_drm_only_books_are_problems() {
        let l = lib();
        let (_lock, store) = l.config.open_for_write().unwrap();
        let conv = CountingConverter::default();
        l.book(
            "A/Both (1)",
            "uuid-both",
            "Both",
            &[
                ("Both.pdf", "pdf"),
                ("Both.epub", "epub"),
                ("Both.original_epub", "orig"),
            ],
        );
        l.book(
            "A/Drm (2)",
            "uuid-drm",
            "Drm",
            &[("Drm.acsm", "<fulfillmentToken/>")],
        );

        let r = l.sweep(&store, &conv);
        assert_eq!(r.added, 1);
        let doc = store
            .get_document("books:calibre:uuid-both")
            .unwrap()
            .unwrap();
        assert_eq!(doc.format.as_deref(), Some("epub"));
        assert!(store
            .get_document("books:calibre:uuid-drm")
            .unwrap()
            .is_none());
        assert!(r.problems.iter().any(|p| p.kind == "drm_placeholder"));
        assert_eq!(store.list_problems().unwrap()[0].kind, "drm_placeholder");
    }

    // AC-021.1
    #[test]
    fn removed_books_stay_removed_until_added_explicitly() {
        let l = lib();
        let (_lock, store) = l.config.open_for_write().unwrap();
        let conv = CountingConverter::default();
        let dir = l.book("A/Book (1)", "uuid-1", "Book", &[("Book.epub", "bytes")]);
        l.sweep(&store, &conv);

        let found = find_books(&store, "book").unwrap();
        assert_eq!(found.len(), 1);
        remove_book(&store, &found[0].doc_id).unwrap();
        assert!(!store
            .get_document("books:calibre:uuid-1")
            .unwrap()
            .unwrap()
            .is_searchable());

        let r = l.sweep(&store, &conv);
        assert_eq!((r.added, r.skipped_removed), (0, 1));

        let r = l.add(
            &store,
            &conv,
            &[dir.join("Book.epub")],
            LibraryOptions::default(),
        );
        assert_eq!(r.restored, 1);
        assert!(store
            .get_document("books:calibre:uuid-1")
            .unwrap()
            .unwrap()
            .is_searchable());
        assert_eq!(conv.calls(), 1, "restoring does not convert again");
    }

    // AC-022.1
    #[test]
    fn books_whose_file_disappears_stay_searchable() {
        let l = lib();
        let (_lock, store) = l.config.open_for_write().unwrap();
        let conv = CountingConverter::default();
        let dir = l.book("A/Book (1)", "uuid-1", "Book", &[("Book.epub", "bytes")]);
        l.sweep(&store, &conv);

        std::fs::remove_dir_all(&dir).unwrap();
        let r = l.sweep(&store, &conv);
        assert_eq!(r.source_missing, 1);
        let doc = store.get_document("books:calibre:uuid-1").unwrap().unwrap();
        assert!(doc.is_searchable());
        assert!(doc.source_missing);
    }

    // AC-023.1
    #[test]
    fn same_bytes_convert_once_and_reconvert_converts_exactly_once() {
        let l = lib();
        let (_lock, store) = l.config.open_for_write().unwrap();
        let conv = CountingConverter::default();
        let a = l.book("A/One (1)", "uuid-a", "One", &[("One.epub", "identical")]);
        l.book("A/Two (2)", "uuid-b", "Two", &[("Two.epub", "identical")]);
        let r = l.sweep(&store, &conv);
        assert_eq!(r.added, 2);
        assert_eq!(
            conv.calls(),
            1,
            "same bytes under another identity reuse the conversion"
        );

        let r = l.add(
            &store,
            &conv,
            &[a.join("One.epub")],
            LibraryOptions {
                reconvert: true,
                ..Default::default()
            },
        );
        assert_eq!(conv.calls(), 2);
        assert_eq!(r.problems.len(), 0);
    }

    // AC-024.1, AC-024.2
    #[test]
    fn changed_epub_is_reconverted_but_changed_pdf_is_flagged() {
        let l = lib();
        let (_lock, store) = l.config.open_for_write().unwrap();
        let conv = CountingConverter {
            fixed_output: Some("# same markdown"),
            ..Default::default()
        };
        let e = l.book("A/E (1)", "uuid-e", "E", &[("E.epub", "epub v1")]);
        let p = l.book("A/P (2)", "uuid-p", "P", &[("P.pdf", "pdf v1")]);
        l.sweep(&store, &conv);
        let before = store.get_document("books:calibre:uuid-e").unwrap().unwrap();
        store
            .mark_indexed(&before.doc_id, &before.content_hash, 1)
            .unwrap();

        std::fs::write(e.join("E.epub"), "epub v2 with other bytes").unwrap();
        std::fs::write(p.join("P.pdf"), "pdf v2 with other bytes").unwrap();
        let r = l.sweep(&store, &conv);
        assert_eq!(conv.calls(), 3, "epub reconverted once, pdf not");
        let after = store.get_document("books:calibre:uuid-e").unwrap().unwrap();
        assert!(
            after.is_index_current(),
            "identical markdown means no re-embedding"
        );
        assert!(r
            .problems
            .iter()
            .any(|x| x.kind == "source_changed" && x.source_path.ends_with("P.pdf")));
    }

    #[test]
    fn pdf_to_epub_upgrade_and_pinned_format() {
        let l = lib();
        let (_lock, store) = l.config.open_for_write().unwrap();
        let conv = CountingConverter::default();
        let dir = l.book("A/B (1)", "uuid-1", "B", &[("B.pdf", "pdf")]);
        l.sweep(&store, &conv);
        std::fs::write(dir.join("B.epub"), "epub").unwrap();
        l.sweep(&store, &conv);
        let doc = store.get_document("books:calibre:uuid-1").unwrap().unwrap();
        assert_eq!(doc.format.as_deref(), Some("epub"));

        // Explicitly adding the PDF pins it; sweeps keep it.
        l.add(
            &store,
            &conv,
            &[dir.join("B.pdf")],
            LibraryOptions::default(),
        );
        l.sweep(&store, &conv);
        let doc = store.get_document("books:calibre:uuid-1").unwrap().unwrap();
        assert_eq!(doc.format.as_deref(), Some("pdf"));
        assert!(doc.format_pinned);
    }

    // Review fix 1: an edited loose book keeps its identity.
    #[test]
    fn edited_loose_books_keep_their_identity() {
        let l = lib();
        let (_lock, store) = l.config.open_for_write().unwrap();
        let conv = CountingConverter::default();
        std::fs::write(l.root.join("loose.pdf"), "pdf v1").unwrap();
        std::fs::write(l.root.join("loose.epub"), "epub v1").unwrap();
        l.sweep(&store, &conv);
        assert_eq!(conv.calls(), 2);

        std::fs::write(l.root.join("loose.pdf"), "pdf v2, longer").unwrap();
        std::fs::write(l.root.join("loose.epub"), "epub v2, longer").unwrap();
        let r = l.sweep(&store, &conv);
        assert_eq!(
            store.list_documents_in("books").unwrap().len(),
            2,
            "no second document"
        );
        assert_eq!(conv.calls(), 3, "epub reconverted, pdf only flagged");
        assert!(r.problems.iter().any(|p| p.kind == "source_changed"));
        assert_eq!(r.updated, 1);
    }

    // Review fix 2: an unusable OPF skips the folder instead of splitting the book.
    #[test]
    fn broken_opf_skips_the_folder_and_keeps_the_book() {
        let l = lib();
        let (_lock, store) = l.config.open_for_write().unwrap();
        let conv = CountingConverter::default();
        let dir = l.book(
            "A/B (1)",
            "uuid-1",
            "Real Title",
            &[("B.epub", "e"), ("B.pdf", "p")],
        );
        l.sweep(&store, &conv);
        let good = opf("uuid-1", "Real Title", &["Scott Chacon", "Ben Straub"]);

        std::fs::write(dir.join("metadata.opf"), "<package><broken").unwrap();
        let r = l.sweep(&store, &conv);
        assert!(r.problems.iter().any(|p| p.kind == "bad_metadata"));
        std::fs::write(dir.join("metadata.opf"), good).unwrap();
        l.sweep(&store, &conv);

        let docs = store.list_documents_in("books").unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].title, "Real Title");
        assert!(!docs[0].source_missing);
    }

    // Review fix 3: adding a folder explicitly keeps a pinned format.
    #[test]
    fn folder_add_keeps_pinned_format() {
        let l = lib();
        let (_lock, store) = l.config.open_for_write().unwrap();
        let conv = CountingConverter::default();
        let dir = l.book(
            "A/B (1)",
            "uuid-1",
            "B",
            &[("B.pdf", "pdf"), ("B.epub", "epub")],
        );
        l.add(
            &store,
            &conv,
            &[dir.join("B.pdf")],
            LibraryOptions::default(),
        );
        l.add(
            &store,
            &conv,
            std::slice::from_ref(&dir),
            LibraryOptions::default(),
        );
        l.sweep(&store, &conv);
        let doc = store.get_document("books:calibre:uuid-1").unwrap().unwrap();
        assert_eq!(doc.format.as_deref(), Some("pdf"));
        assert!(doc.format_pinned);
    }

    // Review fix 4: identical loose copies in one folder do not flip-flop.
    #[test]
    fn identical_loose_copies_are_one_book_and_stay_stable() {
        let l = lib();
        let (_lock, store) = l.config.open_for_write().unwrap();
        let conv = CountingConverter::default();
        std::fs::write(l.root.join("book.epub"), "same").unwrap();
        std::fs::write(l.root.join("book (1).epub"), "same").unwrap();
        l.sweep(&store, &conv);
        let r = l.sweep(&store, &conv);
        assert_eq!(store.list_documents_in("books").unwrap().len(), 1);
        assert_eq!((r.added, r.updated), (0, 0));
        assert!(r.problems.iter().any(|p| p.kind == "duplicate"));
    }

    // Review fix 5: a new online-only loose file is not read to identify it.
    #[test]
    fn new_online_only_loose_file_is_not_hashed() {
        let l = lib();
        let mut collector = Collector {
            library: &l.library,
            store: None,
            by_path: HashMap::new(),
            problems: Vec::new(),
        };
        std::fs::write(l.root.join("x.epub"), "x").unwrap();
        let mut file = found_file(&l.root.join("x.epub"), "x.epub".into()).unwrap();
        file.online_only = true;
        // Remove the bytes: hashing would now fail with "unreadable".
        std::fs::remove_file(&file.abs).unwrap();
        let out = collector.candidates_in(&l.root, vec![file], false, None);
        assert!(out.is_empty());
        assert_eq!(collector.problems[0].kind, "online_only");
    }

    // Review fix 6: an unavailable root does not flag its books as missing.
    #[test]
    fn unavailable_root_does_not_flag_books_missing() {
        let mut l = lib();
        let (_lock, store) = l.config.open_for_write().unwrap();
        let conv = CountingConverter::default();
        l.book("A/B (1)", "uuid-1", "B", &[("B.epub", "x")]);
        l.sweep(&store, &conv);

        let offline = l.root.with_extension("offline");
        std::fs::rename(&l.root, &offline).unwrap();
        let r = l.sweep(&store, &conv);
        assert_eq!(r.status, "error");
        assert_eq!(r.source_missing, 0);
        assert!(
            !store
                .get_document("books:calibre:uuid-1")
                .unwrap()
                .unwrap()
                .source_missing
        );
        std::fs::rename(&offline, &l.root).unwrap();
        l.library.roots = vec![l.root.clone()];
        let r = l.sweep(&store, &conv);
        assert_eq!((r.updated, r.unchanged), (0, 1));
    }

    #[test]
    fn restore_rewrites_missing_canonical_markdown_from_cache() {
        let l = lib();
        let (_lock, store) = l.config.open_for_write().unwrap();
        let conv = CountingConverter::default();
        let dir = l.book("A/B (1)", "uuid-1", "B", &[("B.epub", "x")]);
        l.sweep(&store, &conv);
        let doc = store.get_document("books:calibre:uuid-1").unwrap().unwrap();
        remove_book(&store, &doc.doc_id).unwrap();
        std::fs::remove_file(l.config.canonical_dir.join(&doc.markdown_path)).unwrap();

        let r = l.add(
            &store,
            &conv,
            &[dir.join("B.epub")],
            LibraryOptions::default(),
        );
        assert_eq!(r.restored, 1);
        assert!(l.config.canonical_dir.join(&doc.markdown_path).exists());
        assert_eq!(conv.calls(), 1, "restored from the conversion cache");
    }

    #[test]
    fn reconvert_converts_identical_bytes_once_per_run() {
        let l = lib();
        let (_lock, store) = l.config.open_for_write().unwrap();
        let conv = CountingConverter::default();
        l.book("A/One (1)", "uuid-a", "One", &[("One.epub", "identical")]);
        l.book("A/Two (2)", "uuid-b", "Two", &[("Two.epub", "identical")]);
        l.sweep(&store, &conv);
        sweep(
            &l.config,
            &l.library,
            Some(&store),
            &conv,
            LibraryOptions {
                reconvert: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(conv.calls(), 2);
    }

    #[test]
    fn dry_run_and_missing_root() {
        let mut l = lib();
        l.book("A/B (1)", "uuid-1", "B", &[("B.epub", "x")]);
        let conv = CountingConverter::default();
        let r = sweep(
            &l.config,
            &l.library,
            None,
            &conv,
            LibraryOptions {
                dry_run: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(r.added, 1);
        assert_eq!(conv.calls(), 0);
        assert!(!l.config.colibri_home.exists());

        l.library.roots = vec![l.root.join("nope")];
        let (_lock, store) = l.config.open_for_write().unwrap();
        let r = l.sweep(&store, &conv);
        assert_eq!(r.status, "error");
        assert_eq!(r.problems[0].kind, "root_missing");
    }
}
