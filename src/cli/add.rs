//! `colibri add` — add books (sweep the configured library or given files).

use std::path::PathBuf;

use crate::config::load_config;
use crate::embedding::OllamaEmbedder;
use crate::indexer::{index_library, IndexOptions};
use crate::ingest::convert::ExternalConverter;
use crate::ingest::library::{add_paths, sweep, LibraryOptions, LibraryReport};

pub struct AddOptions {
    pub paths: Vec<PathBuf>,
    pub reconvert: bool,
    pub retry_failed: bool,
    pub dry_run: bool,
    pub no_index: bool,
    pub json: bool,
}

pub async fn run(o: AddOptions) -> anyhow::Result<()> {
    let config = load_config()?;
    let library = config.library.clone().unwrap_or_default();
    if o.paths.is_empty() && library.roots.is_empty() {
        anyhow::bail!(
            "No library folders configured. Add `library: {{roots: [{{path: ...}}]}}` to config.yaml, or pass book files."
        );
    }
    let lib_opts = LibraryOptions {
        dry_run: o.dry_run,
        reconvert: o.reconvert,
        retry_failed: o.retry_failed,
    };
    let converter = ExternalConverter;

    let report = if o.dry_run {
        let store = if config.metadata_db_path.exists() {
            Some(config.open_read()?)
        } else {
            None
        };
        run_library(
            &config,
            &library,
            store.as_ref(),
            &converter,
            &o.paths,
            lib_opts,
        )?
    } else {
        let (_lock, store) = config.open_for_write()?;
        let report = run_library(
            &config,
            &library,
            Some(&store),
            &converter,
            &o.paths,
            lib_opts,
        )?;
        if !o.no_index {
            let progress = crate::cli::index::CliProgress::new();
            index_library(
                &config,
                &store,
                &OllamaEmbedder::from_config(&config),
                &IndexOptions::default(),
                |e| {
                    if !o.json {
                        progress.handle(e)
                    }
                },
            )
            .await?;
        }
        report
    };

    if o.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print_report(&report);
    }
    if report.status == "error" {
        std::process::exit(1);
    }
    Ok(())
}

fn run_library(
    config: &crate::config::AppConfig,
    library: &crate::config::LibraryConfig,
    store: Option<&crate::metadata_store::MetadataStore>,
    converter: &ExternalConverter,
    paths: &[PathBuf],
    opts: LibraryOptions,
) -> anyhow::Result<LibraryReport> {
    Ok(if paths.is_empty() {
        sweep(config, library, store, converter, opts)?
    } else {
        add_paths(config, library, store, converter, paths, opts)?
    })
}

fn print_report(r: &LibraryReport) {
    if r.dry_run {
        eprintln!("Dry run: nothing was changed.");
    }
    for b in &r.books {
        let label = match b.outcome.as_str() {
            "already_known" => "already in library",
            other => other,
        };
        eprintln!("{label}: {} ({})", b.title, b.path);
    }
    for p in &r.problems {
        eprintln!("{} {}: {}", p.kind, p.source_path, p.message);
    }
    eprintln!(
        "Books: {} added, {} updated, {} restored, {} unchanged, {} removed by you (skipped), {} with missing source",
        r.added, r.updated, r.restored, r.unchanged, r.skipped_removed, r.source_missing
    );
}
