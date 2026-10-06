//! `colibri update`: reconcile mirrors, sweep the library, then bring the
//! index up to date.

use serde::Serialize;

use crate::config::{AppConfig, LIBRARY_COLLECTION};
use crate::embedding::Embedder;
use crate::error::ColibriError;
use crate::indexer::{index_library, IndexEvent, IndexOptions};
use crate::ingest::convert::Converter;
use crate::ingest::library::{sweep, LibraryOptions, LibraryReport};
use crate::ingest::mirror::{reconcile_mirror, MirrorReport, ReconcileOptions};
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
    };
    let mut mirrors = Vec::new();
    for mirror in selected {
        mirrors.push(reconcile_mirror(
            config,
            store,
            mirror,
            converter,
            reconcile_opts,
        )?);
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
