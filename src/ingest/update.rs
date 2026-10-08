//! `colibri update`: reconcile mirrors, sweep the library, then bring the
//! index up to date.

use serde::Serialize;

use std::path::Path;

use crate::config::{AppConfig, FetchConfig, PruneConfig, LIBRARY_COLLECTION};
use crate::embedding::Embedder;
use crate::error::ColibriError;
use crate::fetch::FetchReport;
use crate::indexer::{index_library, IndexEvent, IndexOptions};
use crate::ingest::convert::Converter;
use crate::ingest::library::{sweep, LibraryOptions, LibraryReport};
use crate::ingest::mirror::{reconcile_mirror, MirrorReport, Problem, ReconcileOptions};
use crate::metadata_store::MetadataStore;

#[derive(Debug, Clone, Default)]
pub struct UpdateOptions {
    /// Mirror names (or `books` for the library) to update; empty means all.
    pub names: Vec<String>,
    pub dry_run: bool,
    pub allow_mass_prune: bool,
    /// Retry conversions that failed in an earlier run.
    pub retry_failed: bool,
    pub no_index: bool,
    /// Re-embed everything.
    pub force_index: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct IndexSummary {
    pub indexed: usize,
    pub unchanged: usize,
    pub removed: usize,
    pub orphans_removed: usize,
    pub chunks: usize,
    pub errors: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct UpdateReport {
    pub dry_run: bool,
    pub mirrors: Vec<MirrorReport>,
    pub library: Option<LibraryReport>,
    pub index: Option<IndexSummary>,
}

impl UpdateReport {
    /// A mirror could not be processed at all, or indexing had errors.
    pub fn has_errors(&self) -> bool {
        self.mirrors.iter().any(|m| m.status == "error")
            || self.library.as_ref().is_some_and(|l| l.status == "error")
            || self.index.as_ref().is_some_and(|i| i.errors > 0)
    }
}

/// Reconcile the selected mirrors and (unless disabled) index. `store` is
/// read-only or `None` for dry runs.
pub async fn run_update<E: Embedder>(
    config: &AppConfig,
    store: Option<&MetadataStore>,
    embedder: &E,
    converter: &dyn Converter,
    opts: &UpdateOptions,
    on_progress: impl Fn(IndexEvent),
) -> Result<UpdateReport, ColibriError> {
    if config.mirrors.is_empty() && config.library.is_none() {
        return Err(ColibriError::Config(
            "Nothing to update. Add `mirrors:` or `library:` to config.yaml.".into(),
        ));
    }
    for name in &opts.names {
        let is_library = name == LIBRARY_COLLECTION && config.library.is_some();
        if !is_library && !config.mirrors.iter().any(|m| &m.name == name) {
            return Err(ColibriError::Config(format!("Unknown mirror '{name}'")));
        }
    }
    let selected = config
        .mirrors
        .iter()
        .filter(|m| opts.names.is_empty() || opts.names.contains(&m.name));

    let reconcile_opts = ReconcileOptions {
        dry_run: opts.dry_run,
        allow_mass_prune: opts.allow_mass_prune,
        retry_failed: opts.retry_failed,
        fetch_incomplete: false,
    };
    let mut mirrors = Vec::new();
    for mirror in selected {
        let Some(fetch) = &mirror.fetch else {
            mirrors.push(reconcile_mirror(
                config,
                store,
                mirror,
                converter,
                reconcile_opts,
            )?);
            continue;
        };
        if opts.dry_run {
            mirrors.push(MirrorReport {
                name: mirror.name.clone(),
                status: "skipped".into(),
                dry_run: true,
                problems: vec![Problem::new(
                    mirror.path.display().to_string(),
                    "fetch_skipped",
                    "dry run: the fetch was not run, so changes at the source are not shown",
                )],
                ..Default::default()
            });
            continue;
        }
        let prune = (!opts.allow_mass_prune).then_some(config.prune);
        let fetched = run_fetch(fetch, &mirror.path, prune).await;
        let mut report = reconcile_mirror(
            config,
            store,
            mirror,
            converter,
            ReconcileOptions {
                fetch_incomplete: !fetched.complete,
                ..reconcile_opts
            },
        )?;
        if !fetched.problems.is_empty() {
            let failed = !fetched.complete && fetched.listed == 0;
            let fetch_status = if failed { "error" } else { "partial" };
            report.status = worse_status(&report.status, fetch_status).into();
            report.problems.extend(fetched.problems.iter().cloned());
            if let Some(store) = store {
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
                    &serde_json::to_string(&report)?,
                )?;
            }
        }
        report.fetch = Some(fetched);
        mirrors.push(report);
    }

    let wants_library = opts.names.is_empty() || opts.names.iter().any(|n| n == LIBRARY_COLLECTION);
    let library = match &config.library {
        Some(library) if wants_library => Some(sweep(
            config,
            library,
            store,
            converter,
            LibraryOptions {
                dry_run: opts.dry_run,
                retry_failed: opts.retry_failed,
                ..Default::default()
            },
        )?),
        _ => None,
    };

    let index = match store {
        Some(store) if !opts.dry_run && !opts.no_index => {
            let index_opts = IndexOptions {
                force: opts.force_index,
                ..Default::default()
            };
            let r = index_library(config, store, embedder, &index_opts, on_progress).await?;
            Some(IndexSummary {
                indexed: r.files_indexed,
                unchanged: r.files_skipped,
                removed: r.files_deleted,
                orphans_removed: r.orphans_removed,
                chunks: r.total_chunks,
                errors: r.errors,
            })
        }
        _ => None,
    };

    Ok(UpdateReport {
        dry_run: opts.dry_run,
        mirrors,
        library,
        index,
    })
}

/// The more severe of two mirror statuses (`error` > `partial` > others).
fn worse_status<'a>(a: &'a str, b: &'a str) -> &'a str {
    let rank = |s: &str| match s {
        "error" => 2,
        "partial" => 1,
        _ => 0,
    };
    if rank(b) > rank(a) {
        b
    } else {
        a
    }
}

/// Fill a mirror folder from its source. `prune` limits deletions by the
/// fetcher (`None` with `--allow-mass-prune`).
async fn run_fetch(fetch: &FetchConfig, dir: &Path, prune: Option<PruneConfig>) -> FetchReport {
    match fetch {
        FetchConfig::ZephyrScale(cfg) => match crate::fetch::zephyr::ApiSource::from_config(cfg) {
            Ok(source) => {
                crate::fetch::zephyr::fetch(&source, cfg, dir, chrono::Utc::now(), prune).await
            }
            Err(e) => FetchReport {
                problems: vec![Problem::new("zephyr", "fetch_failed", e)],
                ..Default::default()
            },
        },
        FetchConfig::Command { run } => crate::fetch::command::fetch(run, dir),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::MirrorConfig;
    use crate::indexer::tests::FakeEmbedder;
    use crate::ingest::convert::fakes::CountingConverter;

    fn setup() -> (tempfile::TempDir, AppConfig, std::path::PathBuf) {
        let dir = tempfile::TempDir::new().unwrap();
        let root = dir.path().join("src");
        std::fs::create_dir_all(&root).unwrap();
        let mut config = AppConfig::for_test(&dir.path().join("home"));
        config.mirrors = vec![MirrorConfig {
            name: "notes".into(),
            path: root.clone(),
            doc_type: "note".into(),
            include: vec!["**/*.md".into()],
            exclude: vec![],
            plantuml_summaries: false,
            fetch: None,
        }];
        (dir, config, root)
    }

    // AC-009.1, AC-009.2, AC-010.1 at the update level (reconcile + index).
    #[tokio::test]
    async fn update_embeds_only_changes_and_drops_deleted_documents() {
        let (_dir, config, root) = setup();
        for name in ["a", "b", "c"] {
            std::fs::write(
                root.join(format!("{name}.md")),
                format!("# {name}\ntext {name}"),
            )
            .unwrap();
        }
        let (_lock, store) = config.open_for_write().unwrap();
        let conv = CountingConverter::default();
        let opts = UpdateOptions::default();

        let r = run_update(
            &config,
            Some(&store),
            &FakeEmbedder::new(),
            &conv,
            &opts,
            |_| {},
        )
        .await
        .unwrap();
        assert_eq!(r.mirrors[0].added, 3);
        assert_eq!(r.index.as_ref().unwrap().indexed, 3);

        std::fs::write(root.join("b.md"), "# b\nchanged").unwrap();
        let r = run_update(
            &config,
            Some(&store),
            &FakeEmbedder::new(),
            &conv,
            &opts,
            |_| {},
        )
        .await
        .unwrap();
        assert_eq!(r.mirrors[0].changed, 1);
        let idx = r.index.unwrap();
        assert_eq!((idx.indexed, idx.unchanged), (1, 2));

        let embedder = FakeEmbedder::new();
        let r = run_update(&config, Some(&store), &embedder, &conv, &opts, |_| {})
            .await
            .unwrap();
        assert_eq!(r.index.unwrap().indexed, 0);
        assert_eq!(embedder.calls.load(std::sync::atomic::Ordering::SeqCst), 0);

        std::fs::remove_file(root.join("c.md")).unwrap();
        let r = run_update(
            &config,
            Some(&store),
            &FakeEmbedder::new(),
            &conv,
            &opts,
            |_| {},
        )
        .await
        .unwrap();
        assert_eq!(r.mirrors[0].pruned, 1);
        assert_eq!(r.index.unwrap().removed, 1);
        let c = store.get_document("notes:c.md").unwrap().unwrap();
        assert!(!c.is_searchable());
        assert_eq!(c.indexed_hash, None, "chunks of the deleted file are gone");
    }

    #[tokio::test]
    async fn unknown_mirror_name_is_an_error() {
        let (_dir, config, _root) = setup();
        let opts = UpdateOptions {
            names: vec!["nope".into()],
            dry_run: true,
            ..Default::default()
        };
        let err = run_update(
            &config,
            None,
            &FakeEmbedder::new(),
            &CountingConverter::default(),
            &opts,
            |_| {},
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("Unknown mirror 'nope'"));
    }
}
