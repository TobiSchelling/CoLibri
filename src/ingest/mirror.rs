//! Mirrors: folders whose content CoLibri keeps in sync.
//!
//! A reconcile first plans (read-only: hashes text files, compares size and
//! mtime of converted formats) and then applies the plan: converts where
//! needed, writes canonical markdown and document rows, and prunes documents
//! whose files are gone. Pruning only happens after a complete walk and
//! below the configured mass-deletion limit.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use chrono::{DateTime, Utc};
use serde::Serialize;

use crate::canonical_store::content_hash;
use crate::canonical_store::{canonical_rel_path, doc_id_for};
use crate::config::{AppConfig, MirrorConfig};
use crate::error::ColibriError;
use crate::ingest::convert::{convert_cached, file_sha256, is_convertible, Converter};
use crate::ingest::frontmatter::parse_frontmatter;
use crate::ingest::plantuml::enrich_plantuml_blocks;
use crate::ingest::walk::{walk, Filter, FoundFile, WalkResult};
use crate::metadata_store::{DocStatus, DocumentRecord, MetadataStore};

/// Rows written per SQLite transaction.
const COMMIT_EVERY: usize = 200;

/// Something that needs the user's attention after a run.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Problem {
    pub source_path: String,
    pub kind: String,
    pub message: String,
}

impl Problem {
    pub(crate) fn new(
        source_path: impl Into<String>,
        kind: &str,
        message: impl Into<String>,
    ) -> Self {
        Self {
            source_path: source_path.into(),
            kind: kind.into(),
            message: message.into(),
        }
    }
}

/// Result of reconciling one mirror.
#[derive(Debug, Clone, Default, Serialize)]
pub struct MirrorReport {
    pub name: String,
    /// `ok`, `partial` (problems or blocked prune) or `error` (nothing done).
    pub status: String,
    pub dry_run: bool,
    pub added: usize,
    pub changed: usize,
    pub unchanged: usize,
    pub pruned: usize,
    /// Deletions detected but not applied, with the reason.
    pub prune_blocked: Option<String>,
    pub prune_candidates: usize,
    pub problems: Vec<Problem>,
    /// Result of the fetch that filled the folder, if the mirror has one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fetch: Option<crate::fetch::FetchReport>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ReconcileOptions {
    pub dry_run: bool,
    pub allow_mass_prune: bool,
    /// Retry conversions that failed in an earlier run.
    pub retry_failed: bool,
    /// The fetch before this reconcile was incomplete: do not prune.
    pub fetch_incomplete: bool,
}

/// Markdown plus metadata ready to be stored.
#[derive(Debug, Clone)]
struct Prepared {
    title: String,
    markdown: String,
    tags: Vec<String>,
    frontmatter_json: String,
}

enum Content {
    Ready(Prepared),
    /// Convertible source with these bytes; converted during apply.
    Convert {
        sha256: String,
    },
    /// Same content as stored; only location, size/mtime or doc_type changed.
    Meta,
}

struct Planned {
    file: FoundFile,
    ext: String,
    is_new: bool,
    content: Content,
}

pub(crate) fn rfc3339(t: std::time::SystemTime) -> String {
    DateTime::<Utc>::from(t).to_rfc3339()
}

fn extension(rel: &str) -> String {
    Path::new(rel)
        .extension()
        .map(|e| format!(".{}", e.to_string_lossy().to_lowercase()))
        .unwrap_or_default()
}

/// Title from the file stem with `_`/`-` turned into spaces.
pub fn default_title(path: &Path) -> String {
    path.file_stem()
        .map(|s| {
            s.to_string_lossy()
                .replace(['_', '-'], " ")
                .trim()
                .to_string()
        })
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "untitled".into())
}

fn prepare_text(mirror: &MirrorConfig, file: &FoundFile, ext: &str, text: &str) -> Prepared {
    let fallback_title = default_title(Path::new(&file.rel));
    match ext {
        ".yaml" | ".yml" => Prepared {
            markdown: format!("# {fallback_title}\n\n```yaml\n{text}\n```\n"),
            title: fallback_title,
            tags: Vec::new(),
            frontmatter_json: "{}".into(),
        },
        ".md" | ".markdown" => {
            let (tags, frontmatter, body) = parse_frontmatter(text, &file.rel);
            let title = frontmatter
                .as_ref()
                .and_then(|fm| fm.get("title"))
                .and_then(|t| t.as_str())
                .map(|t| t.trim().to_string())
                .filter(|t| !t.is_empty())
                .unwrap_or(fallback_title);
            let markdown = if mirror.plantuml_summaries {
                enrich_plantuml_blocks(body)
            } else {
                body.to_string()
            };
            Prepared {
                title,
                markdown,
                tags: tags.unwrap_or_default(),
                frontmatter_json: frontmatter
                    .map(|m| serde_json::Value::Object(m).to_string())
                    .unwrap_or_else(|| "{}".into()),
            }
        }
        _ => Prepared {
            title: fallback_title,
            markdown: text.to_string(),
            tags: Vec::new(),
            frontmatter_json: "{}".into(),
        },
    }
}

/// Formats read as text. `.feature` (Gherkin) and `.txt` are indexed as is.
fn is_text(ext: &str) -> bool {
    matches!(
        ext,
        ".md" | ".markdown" | ".txt" | ".feature" | ".yaml" | ".yml"
    )
}

/// Stored location, stat and doc_type match the file and mirror.
fn meta_matches(d: &DocumentRecord, mirror: &MirrorConfig, file: &FoundFile, mtime: &str) -> bool {
    d.doc_type == mirror.doc_type
        && d.source_path.as_deref() == Some(&*file.abs.to_string_lossy())
        && d.source_size == Some(file.size as i64)
        && d.source_mtime.as_deref() == Some(mtime)
}

/// Decide what to do with one file. `Ok(None)` means unchanged.
fn plan_file(
    mirror: &MirrorConfig,
    file: &FoundFile,
    existing: Option<&DocumentRecord>,
) -> Result<Option<Planned>, Problem> {
    let ext = extension(&file.rel);
    let mtime = rfc3339(file.mtime);
    let active = existing.filter(|d| d.is_searchable());
    let planned = |content: Content| Planned {
        file: file.clone(),
        ext: ext.clone(),
        is_new: existing.is_none(),
        content,
    };
    let meta_only = |d: &DocumentRecord| {
        if meta_matches(d, mirror, file, &mtime) {
            None
        } else {
            Some(planned(Content::Meta))
        }
    };

    if is_text(&ext) {
        let bytes = std::fs::read(&file.abs)
            .map_err(|e| Problem::new(&file.rel, "unreadable", e.to_string()))?;
        let prepared = prepare_text(mirror, file, &ext, &String::from_utf8_lossy(&bytes));
        return Ok(match active {
            Some(d)
                if d.content_hash == content_hash(&prepared.markdown)
                    && d.title == prepared.title
                    && d.frontmatter_json == prepared.frontmatter_json =>
            {
                meta_only(d)
            }
            _ => Some(planned(Content::Ready(prepared))),
        });
    }

    if is_convertible(&ext) {
        // Same size and mtime: trust the stored conversion without hashing.
        if let Some(d) = active.filter(|d| {
            d.source_size == Some(file.size as i64)
                && d.source_mtime.as_deref() == Some(mtime.as_str())
        }) {
            return Ok(meta_only(d));
        }
        let sha256 = file_sha256(&file.abs)
            .map_err(|e| Problem::new(&file.rel, "unreadable", e.to_string()))?;
        if active.is_some_and(|d| d.source_sha256.as_deref() == Some(sha256.as_str())) {
            return Ok(Some(planned(Content::Meta)));
        }
        return Ok(Some(planned(Content::Convert { sha256 })));
    }

    Err(Problem::new(
        &file.rel,
        "unsupported_type",
        format!("no converter for '{ext}' files"),
    ))
}

/// Bring one mirror's documents in line with its folder.
pub fn reconcile_mirror(
    config: &AppConfig,
    store: Option<&MetadataStore>,
    mirror: &MirrorConfig,
    converter: &dyn Converter,
    opts: ReconcileOptions,
) -> Result<MirrorReport, ColibriError> {
    let mut report = MirrorReport {
        name: mirror.name.clone(),
        dry_run: opts.dry_run,
        ..Default::default()
    };
    if !opts.dry_run && store.is_none() {
        return Err(ColibriError::Config(
            "reconcile needs a writable metadata DB".into(),
        ));
    }

    if !mirror.path.is_dir() {
        report.status = "error".into();
        report.problems.push(Problem::new(
            mirror.path.display().to_string(),
            "root_missing",
            "mirror folder does not exist or is not a directory; nothing was changed",
        ));
        record_run(store, mirror, &report, opts)?;
        return Ok(report);
    }

    let walked = walk(&mirror.path, &Filter::new(&mirror.include, &mirror.exclude));
    reconcile_walked(config, store, mirror, converter, opts, walked, report)
}

/// Reconcile against an already walked folder (separate for tests).
fn reconcile_walked(
    config: &AppConfig,
    store: Option<&MetadataStore>,
    mirror: &MirrorConfig,
    converter: &dyn Converter,
    opts: ReconcileOptions,
    walked: WalkResult,
    mut report: MirrorReport,
) -> Result<MirrorReport, ColibriError> {
    let existing: HashMap<String, DocumentRecord> = match store {
        Some(s) => s
            .list_documents_in(&mirror.name)?
            .into_iter()
            .map(|d| (d.key.clone(), d))
            .collect(),
        None => HashMap::new(),
    };

    // 1. Plan (read-only).
    for (rel, message) in &walked.notices {
        report
            .problems
            .push(Problem::new(rel.clone(), "skipped", message.clone()));
    }
    for (rel, message) in &walked.unreadable {
        report
            .problems
            .push(Problem::new(rel.clone(), "unreadable", message.clone()));
    }
    let mut seen: HashSet<&str> = HashSet::new();
    let mut plan = Vec::new();
    for file in &walked.files {
        seen.insert(file.rel.as_str());
        if file.online_only {
            report.problems.push(Problem::new(
                &file.rel,
                "online_only",
                "file is only in the cloud; make it available offline to ingest it",
            ));
            continue;
        }
        match plan_file(mirror, file, existing.get(&file.rel)) {
            Ok(Some(p)) => plan.push(p),
            Ok(None) => report.unchanged += 1,
            Err(problem) => report.problems.push(problem),
        }
    }

    let active_count = existing.values().filter(|d| d.is_searchable()).count();
    let to_prune: Vec<&DocumentRecord> = existing
        .values()
        .filter(|d| d.is_searchable() && !seen.contains(d.key.as_str()))
        .collect();
    report.prune_candidates = to_prune.len();
    let limit = config.prune.limit(active_count);
    report.prune_blocked = if to_prune.is_empty() {
        None
    } else if opts.fetch_incomplete {
        Some("the fetch was incomplete; deletions wait for a complete fetch".into())
    } else if !walked.complete {
        Some("the folder could not be read completely".into())
    } else if walked.files.is_empty() && !opts.allow_mass_prune {
        Some(format!(
            "the folder looks empty but {active_count} documents came from it (unmounted or signed-out cloud folder?); rerun with --allow-mass-prune if it really is empty"
        ))
    } else if to_prune.len() > limit && !opts.allow_mass_prune {
        Some(format!(
            "{} deletions exceed the limit of {limit}; rerun with --allow-mass-prune if they are intended",
            to_prune.len()
        ))
    } else {
        None
    };
    if let Some(reason) = &report.prune_blocked {
        report.problems.push(Problem::new(
            mirror.path.display().to_string(),
            "prune_blocked",
            reason.clone(),
        ));
    }

    // 2. Apply.
    if opts.dry_run {
        for p in &plan {
            match (&p.content, p.is_new) {
                (Content::Meta, _) => report.unchanged += 1,
                (_, true) => report.added += 1,
                (_, false) => report.changed += 1,
            }
        }
        report.pruned = if report.prune_blocked.is_some() {
            0
        } else {
            to_prune.len()
        };
        report.status = status_of(&report);
        return Ok(report);
    }

    let store = store.expect("checked above");
    let mut tx = Some(store.begin()?);
    let mut pending = 0usize;
    let mut commit_point = |pending: &mut usize| -> Result<(), ColibriError> {
        *pending += 1;
        if *pending >= COMMIT_EVERY {
            if let Some(t) = tx.take() {
                t.commit()?;
            }
            tx = Some(store.begin()?);
            *pending = 0;
        }
        Ok(())
    };

    for p in plan {
        let doc_id = doc_id_for(&mirror.name, &p.file.rel);
        let prev = existing.get(&p.file.rel);
        let mtime = rfc3339(p.file.mtime);
        let (prepared, tool, sha) = match p.content {
            Content::Meta => {
                let mut doc = prev.cloned().expect("meta update implies an existing row");
                doc.doc_type = mirror.doc_type.clone();
                doc.source_path = Some(p.file.abs.display().to_string());
                doc.source_size = Some(p.file.size as i64);
                doc.source_mtime = Some(mtime.clone());
                doc.source_updated_at = Some(mtime);
                doc.updated_at = Utc::now().to_rfc3339();
                store.upsert_document(&doc)?;
                commit_point(&mut pending)?;
                report.unchanged += 1;
                continue;
            }
            Content::Ready(prepared) => (prepared, None, None),
            Content::Convert { sha256 } => {
                match convert_cached(
                    store,
                    &config.conversions_dir,
                    converter,
                    &p.ext,
                    &p.file.abs,
                    &sha256,
                    opts.retry_failed,
                ) {
                    Ok((markdown, tool)) => (
                        Prepared {
                            title: default_title(Path::new(&p.file.rel)),
                            markdown,
                            tags: Vec::new(),
                            frontmatter_json: "{}".into(),
                        },
                        Some(tool),
                        Some(sha256),
                    ),
                    Err(e) => {
                        report
                            .problems
                            .push(Problem::new(&p.file.rel, "conversion_failed", e));
                        continue;
                    }
                }
            }
        };

        let rel_path = canonical_rel_path(&mirror.name, &doc_id)
            .to_string_lossy()
            .to_string();
        let abs = config.canonical_dir.join(&rel_path);
        std::fs::create_dir_all(abs.parent().unwrap_or(&config.canonical_dir))?;
        std::fs::write(&abs, &prepared.markdown)?;

        let now = Utc::now().to_rfc3339();
        let mut doc = prev
            .cloned()
            .unwrap_or_else(|| DocumentRecord::new(&doc_id, &mirror.name, &p.file.rel));
        doc.source_path = Some(p.file.abs.display().to_string());
        doc.title = prepared.title;
        doc.tags_json = serde_json::to_string(&prepared.tags)?;
        doc.frontmatter_json = prepared.frontmatter_json;
        doc.doc_type = mirror.doc_type.clone();
        doc.format = Some(p.ext.trim_start_matches('.').to_string());
        doc.converter = tool;
        doc.source_size = Some(p.file.size as i64);
        doc.source_mtime = Some(mtime.clone());
        doc.source_sha256 = sha;
        doc.source_updated_at = Some(mtime);
        doc.content_hash = content_hash(&prepared.markdown);
        doc.markdown_path = rel_path;
        doc.status = DocStatus::Active;
        doc.removed_reason = None;
        doc.source_missing = false;
        doc.updated_at = now.clone();
        doc.last_seen_at = Some(now);
        store.upsert_document(&doc)?;
        commit_point(&mut pending)?;
        if p.is_new {
            report.added += 1;
        } else {
            report.changed += 1;
        }
    }

    let mut pruned_files = Vec::new();
    if report.prune_blocked.is_none() {
        for doc in &to_prune {
            // A file that still exists was excluded by the include/exclude patterns.
            let reason = if mirror.path.join(&doc.key).exists() {
                "excluded"
            } else {
                "source_deleted"
            };
            store.set_status(&doc.doc_id, DocStatus::Removed, Some(reason))?;
            pruned_files.push(config.canonical_dir.join(&doc.markdown_path));
            commit_point(&mut pending)?;
            report.pruned += 1;
        }
    }

    if let Some(t) = tx.take() {
        t.commit()?;
    }
    // Only after the rows are committed, so a rollback never leaves an
    // active document without its canonical markdown.
    for path in pruned_files {
        let _ = std::fs::remove_file(path);
    }
    report.status = status_of(&report);
    record_run(Some(store), mirror, &report, opts)?;
    Ok(report)
}

fn status_of(report: &MirrorReport) -> String {
    if report.problems.is_empty() {
        "ok".into()
    } else {
        "partial".into()
    }
}

fn record_run(
    store: Option<&MetadataStore>,
    mirror: &MirrorConfig,
    report: &MirrorReport,
    opts: ReconcileOptions,
) -> Result<(), ColibriError> {
    let Some(store) = store.filter(|_| !opts.dry_run) else {
        return Ok(());
    };
    let problems: Vec<(String, String, String)> = report
        .problems
        .iter()
        .map(|p| (p.source_path.clone(), p.kind.clone(), p.message.clone()))
        .collect();
    store.replace_problems(&mirror.name, &problems)?;
    store.record_collection_run(
        &mirror.name,
        "mirror",
        Some(&mirror.path.display().to_string()),
        &report.status,
        &serde_json::to_string(report)?,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ingest::convert::fakes::CountingConverter;

    struct Fixture {
        _dir: tempfile::TempDir,
        root: std::path::PathBuf,
        config: AppConfig,
        mirror: MirrorConfig,
    }

    fn fixture() -> Fixture {
        let dir = tempfile::TempDir::new().unwrap();
        let root = dir.path().join("src");
        std::fs::create_dir_all(&root).unwrap();
        let mut config = AppConfig::for_test(&dir.path().join("home"));
        config.prune.min_count = 25;
        config.prune.max_fraction = 0.2;
        let mirror = MirrorConfig {
            name: "vault".into(),
            path: root.clone(),
            doc_type: "note".into(),
            include: vec!["**/*.md".into(), "**/*.yaml".into(), "**/*.epub".into()],
            exclude: vec![],
            plantuml_summaries: true,
            fetch: None,
        };
        Fixture {
            _dir: dir,
            root,
            config,
            mirror,
        }
    }

    impl Fixture {
        fn write(&self, rel: &str, text: &str) {
            let p = self.root.join(rel);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, text).unwrap();
        }

        fn run(
            &self,
            store: &MetadataStore,
            conv: &CountingConverter,
            opts: ReconcileOptions,
        ) -> MirrorReport {
            reconcile_mirror(&self.config, Some(store), &self.mirror, conv, opts).unwrap()
        }
    }

    fn counts(r: &MirrorReport) -> (usize, usize, usize, usize) {
        (r.added, r.changed, r.unchanged, r.pruned)
    }

    // AC-009.1 / AC-009.2 (reconcile part)
    #[test]
    fn adds_changes_and_leaves_unchanged_files_alone() {
        let f = fixture();
        let (_lock, store) = f.config.open_for_write().unwrap();
        let conv = CountingConverter::default();
        f.write("a.md", "# A");
        f.write(
            "sub/b.md",
            "---\ntitle: Bee\nstatus: active\ntags: [x]\n---\nbody b",
        );
        f.write("book.epub", "epub bytes");

        let r = f.run(&store, &conv, ReconcileOptions::default());
        assert_eq!(counts(&r), (3, 0, 0, 0));
        assert_eq!(r.status, "ok");
        let b = store.get_document("vault:sub/b.md").unwrap().unwrap();
        assert_eq!(b.title, "Bee");
        assert_eq!(b.tags_json, r#"["x"]"#);
        assert!(b.frontmatter_json.contains("\"status\":\"active\""));
        assert_eq!(conv.calls(), 1);

        f.write("a.md", "# A changed");
        let r = f.run(&store, &conv, ReconcileOptions::default());
        assert_eq!(counts(&r), (0, 1, 2, 0));

        let before = store.list_documents().unwrap();
        let canonical_mtimes: Vec<_> = before
            .iter()
            .map(|d| {
                std::fs::metadata(f.config.canonical_dir.join(&d.markdown_path))
                    .unwrap()
                    .modified()
                    .unwrap()
            })
            .collect();
        let r = f.run(&store, &conv, ReconcileOptions::default());
        assert_eq!(counts(&r), (0, 0, 3, 0));
        assert_eq!(conv.calls(), 1, "unchanged epub is never converted again");
        // AC-009.2: no row and no canonical file was rewritten.
        assert_eq!(store.list_documents().unwrap(), before);
        let after_mtimes: Vec<_> = before
            .iter()
            .map(|d| {
                std::fs::metadata(f.config.canonical_dir.join(&d.markdown_path))
                    .unwrap()
                    .modified()
                    .unwrap()
            })
            .collect();
        assert_eq!(after_mtimes, canonical_mtimes);
    }

    #[test]
    fn doc_type_change_on_converted_files_is_applied_without_reconversion() {
        let mut f = fixture();
        let (_lock, store) = f.config.open_for_write().unwrap();
        let conv = CountingConverter::default();
        f.write("book.epub", "epub bytes");
        f.run(&store, &conv, ReconcileOptions::default());

        f.mirror.doc_type = "book".into();
        let r = reconcile_mirror(
            &f.config,
            Some(&store),
            &f.mirror,
            &conv,
            ReconcileOptions::default(),
        )
        .unwrap();
        assert_eq!(counts(&r), (0, 0, 1, 0));
        assert_eq!(
            store
                .get_document("vault:book.epub")
                .unwrap()
                .unwrap()
                .doc_type,
            "book"
        );
        assert_eq!(conv.calls(), 1);

        // Settled: the next run has nothing to write.
        let before = store.get_document("vault:book.epub").unwrap();
        reconcile_mirror(
            &f.config,
            Some(&store),
            &f.mirror,
            &conv,
            ReconcileOptions::default(),
        )
        .unwrap();
        assert_eq!(store.get_document("vault:book.epub").unwrap(), before);
    }

    // AC-016.1 (reconcile part): online-only files count as seen.
    #[test]
    fn online_only_files_are_reported_and_never_pruned() {
        let f = fixture();
        let (_lock, store) = f.config.open_for_write().unwrap();
        let conv = CountingConverter::default();
        f.write("a.md", "# A");
        f.run(&store, &conv, ReconcileOptions::default());

        let mut walked = walk(&f.root, &Filter::new(&f.mirror.include, &f.mirror.exclude));
        walked.files[0].online_only = true;
        let report = MirrorReport {
            name: f.mirror.name.clone(),
            ..Default::default()
        };
        let r = reconcile_walked(
            &f.config,
            Some(&store),
            &f.mirror,
            &conv,
            ReconcileOptions::default(),
            walked,
            report,
        )
        .unwrap();
        assert_eq!(r.pruned, 0);
        assert_eq!(r.problems[0].kind, "online_only");
        assert!(store
            .get_document("vault:a.md")
            .unwrap()
            .unwrap()
            .is_searchable());
    }

    #[test]
    fn narrowed_include_prunes_with_reason_excluded() {
        let mut f = fixture();
        let (_lock, store) = f.config.open_for_write().unwrap();
        let conv = CountingConverter::default();
        f.write("a.md", "# A");
        f.write("arch/x.yaml", "k: v");
        f.run(&store, &conv, ReconcileOptions::default());

        f.mirror.include = vec!["**/*.md".into()];
        let r = reconcile_mirror(
            &f.config,
            Some(&store),
            &f.mirror,
            &conv,
            ReconcileOptions::default(),
        )
        .unwrap();
        assert_eq!(r.pruned, 1);
        let x = store.get_document("vault:arch/x.yaml").unwrap().unwrap();
        assert_eq!(x.removed_reason.as_deref(), Some("excluded"));
    }

    #[test]
    fn empty_folder_blocks_prune() {
        let f = fixture();
        let (_lock, store) = f.config.open_for_write().unwrap();
        let conv = CountingConverter::default();
        f.write("a.md", "# A");
        f.write("b.md", "# B");
        f.run(&store, &conv, ReconcileOptions::default());

        for name in ["a.md", "b.md"] {
            std::fs::remove_file(f.root.join(name)).unwrap();
        }
        let r = f.run(&store, &conv, ReconcileOptions::default());
        assert_eq!(r.pruned, 0);
        assert!(r.prune_blocked.as_deref().unwrap().contains("looks empty"));

        let r = f.run(
            &store,
            &conv,
            ReconcileOptions {
                allow_mass_prune: true,
                ..Default::default()
            },
        );
        assert_eq!(r.pruned, 2);
    }

    #[test]
    fn touched_convertible_file_with_same_bytes_is_not_reconverted() {
        let f = fixture();
        let (_lock, store) = f.config.open_for_write().unwrap();
        let conv = CountingConverter::default();
        f.write("book.epub", "epub bytes");
        f.run(&store, &conv, ReconcileOptions::default());

        let path = f.root.join("book.epub");
        let later = std::time::SystemTime::now() + std::time::Duration::from_secs(60);
        std::fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_modified(later)
            .unwrap();
        let r = f.run(&store, &conv, ReconcileOptions::default());
        assert_eq!(counts(&r), (0, 0, 1, 0));
        assert_eq!(conv.calls(), 1);
        let doc = store.get_document("vault:book.epub").unwrap().unwrap();
        assert_eq!(doc.source_mtime.as_deref(), Some(rfc3339(later).as_str()));
    }

    // AC-010.1 (reconcile part)
    #[test]
    fn deleted_file_is_pruned_and_reappearing_file_is_reactivated() {
        let f = fixture();
        let (_lock, store) = f.config.open_for_write().unwrap();
        let conv = CountingConverter::default();
        f.write("a.md", "# A");
        f.write("b.md", "# B");
        f.run(&store, &conv, ReconcileOptions::default());

        std::fs::remove_file(f.root.join("b.md")).unwrap();
        let r = f.run(&store, &conv, ReconcileOptions::default());
        assert_eq!(r.pruned, 1);
        let b = store.get_document("vault:b.md").unwrap().unwrap();
        assert!(!b.is_searchable());
        assert_eq!(b.removed_reason.as_deref(), Some("source_deleted"));
        assert!(!f.config.canonical_dir.join(&b.markdown_path).exists());

        f.write("b.md", "# B");
        let r = f.run(&store, &conv, ReconcileOptions::default());
        assert_eq!(r.changed, 1);
        assert!(store
            .get_document("vault:b.md")
            .unwrap()
            .unwrap()
            .is_searchable());
    }

    // AC-011.1
    #[test]
    fn missing_root_prunes_nothing_and_reports_a_problem() {
        let f = fixture();
        let (_lock, store) = f.config.open_for_write().unwrap();
        let conv = CountingConverter::default();
        f.write("a.md", "# A");
        f.run(&store, &conv, ReconcileOptions::default());

        std::fs::rename(&f.root, f.root.with_extension("moved")).unwrap();
        let r = f.run(&store, &conv, ReconcileOptions::default());
        assert_eq!(r.status, "error");
        assert_eq!(r.pruned, 0);
        assert!(store
            .get_document("vault:a.md")
            .unwrap()
            .unwrap()
            .is_searchable());
        let problems = store.list_problems().unwrap();
        assert_eq!(problems[0].kind, "root_missing");
    }

    // AC-011.2
    #[test]
    fn unreadable_subdirectory_blocks_prune() {
        use std::os::unix::fs::PermissionsExt;
        let f = fixture();
        let (_lock, store) = f.config.open_for_write().unwrap();
        let conv = CountingConverter::default();
        f.write("a.md", "# A");
        f.write("locked/b.md", "# B");
        f.run(&store, &conv, ReconcileOptions::default());

        let locked = f.root.join("locked");
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000)).unwrap();
        let r = f.run(&store, &conv, ReconcileOptions::default());
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o755)).unwrap();
        assert_eq!(r.pruned, 0);
        assert_eq!(r.prune_candidates, 1);
        assert!(r.prune_blocked.is_some());
        assert_eq!(r.status, "partial");
    }

    // AC-011.3
    #[test]
    fn mass_deletion_is_blocked_unless_allowed() {
        let f = fixture();
        let (_lock, store) = f.config.open_for_write().unwrap();
        let conv = CountingConverter::default();
        for i in 0..40 {
            f.write(&format!("n{i:02}.md"), &format!("# Note {i}"));
        }
        f.run(&store, &conv, ReconcileOptions::default());
        for i in 0..30 {
            std::fs::remove_file(f.root.join(format!("n{i:02}.md"))).unwrap();
        }

        let r = f.run(&store, &conv, ReconcileOptions::default());
        assert_eq!(r.pruned, 0);
        assert!(r
            .prune_blocked
            .as_deref()
            .unwrap()
            .contains("exceed the limit of 25"));
        assert!(store
            .list_problems()
            .unwrap()
            .iter()
            .any(|p| p.kind == "prune_blocked"));

        let r = f.run(
            &store,
            &conv,
            ReconcileOptions {
                allow_mass_prune: true,
                ..Default::default()
            },
        );
        assert_eq!(r.pruned, 30);
        assert!(store.list_problems().unwrap().is_empty());
    }

    // AC-012.1
    #[test]
    fn moving_the_mirror_folder_keeps_document_ids() {
        let mut f = fixture();
        let (_lock, store) = f.config.open_for_write().unwrap();
        let conv = CountingConverter::default();
        f.write("a.md", "# A");
        f.write("x/b.md", "# B");
        f.run(&store, &conv, ReconcileOptions::default());

        let moved = f.root.with_extension("moved");
        std::fs::rename(&f.root, &moved).unwrap();
        f.mirror.path = moved.clone();
        let r = reconcile_mirror(
            &f.config,
            Some(&store),
            &f.mirror,
            &conv,
            ReconcileOptions::default(),
        )
        .unwrap();
        assert_eq!((r.added, r.changed, r.unchanged, r.pruned), (0, 0, 2, 0));
        let b = store.get_document("vault:x/b.md").unwrap().unwrap();
        assert_eq!(
            b.source_path,
            Some(moved.join("x/b.md").display().to_string())
        );
        assert!(
            b.is_index_current() || b.indexed_hash.is_none(),
            "no content change"
        );
    }

    // AC-013.1 (reconcile part)
    #[test]
    fn dry_run_reports_without_writing() {
        let f = fixture();
        let conv = CountingConverter::default();
        f.write("a.md", "# A");
        f.write("book.epub", "bytes");
        let r = reconcile_mirror(
            &f.config,
            None,
            &f.mirror,
            &conv,
            ReconcileOptions {
                dry_run: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(r.added, 2);
        assert_eq!(conv.calls(), 0);
        assert!(!f.config.colibri_home.exists());
    }

    #[test]
    fn yaml_is_fenced_and_unsupported_types_are_problems() {
        let mut f = fixture();
        f.mirror.include.push("**/*.bin".into());
        let (_lock, store) = f.config.open_for_write().unwrap();
        let conv = CountingConverter::default();
        f.write("arch/service-map.yaml", "service: billing\n");
        f.write("blob.bin", "x");
        let r = f.run(&store, &conv, ReconcileOptions::default());
        assert_eq!(r.added, 1);
        assert_eq!(r.problems[0].kind, "unsupported_type");
        let doc = store
            .get_document("vault:arch/service-map.yaml")
            .unwrap()
            .unwrap();
        assert_eq!(doc.title, "service map");
        let md = std::fs::read_to_string(f.config.canonical_dir.join(&doc.markdown_path)).unwrap();
        assert!(md.contains("```yaml\nservice: billing"));
    }

    #[test]
    fn gherkin_feature_files_are_indexed_as_text() {
        let mut f = fixture();
        f.mirror.include = vec!["**/*.feature".into()];
        let (_lock, store) = f.config.open_for_write().unwrap();
        let conv = CountingConverter::default();
        f.write(
            "features/asset_transfer.feature",
            "Feature: Asset transfer\n  Scenario: Transfer between jobsites\n",
        );
        let r = f.run(&store, &conv, ReconcileOptions::default());
        assert_eq!(r.added, 1);
        assert!(r.problems.is_empty());
        let doc = store
            .get_document("vault:features/asset_transfer.feature")
            .unwrap()
            .unwrap();
        assert_eq!(doc.title, "asset transfer");
        assert_eq!(conv.calls(), 0);
    }

    #[test]
    fn default_title_replaces_separators() {
        assert_eq!(default_title(Path::new("my_file-name.md")), "my file name");
    }
}
