//! `colibri update` — reconcile mirrors and update the index.

use crate::config::load_config;
use crate::embedding::OllamaEmbedder;
use crate::ingest::convert::ExternalConverter;
use crate::ingest::update::{run_update, UpdateOptions, UpdateReport};

pub async fn run(opts: UpdateOptions, json: bool) -> anyhow::Result<()> {
    let config = load_config()?;
    let embedder = OllamaEmbedder::from_config(&config);
    let progress = crate::cli::index::CliProgress::new();
    let on_progress = |e| {
        if !json {
            progress.handle(e);
        }
    };

    let report = if opts.dry_run {
        // Read existing state (if any) without the lock; write nothing.
        let store = if config.metadata_db_path.exists() {
            Some(config.open_read()?)
        } else {
            None
        };
        run_update(
            &config,
            store.as_ref(),
            &embedder,
            &ExternalConverter,
            &opts,
            on_progress,
        )
        .await?
    } else {
        let (_lock, store) = config.open_for_write()?;
        let _awake = crate::power::keep_awake();
        run_update(
            &config,
            Some(&store),
            &embedder,
            &ExternalConverter,
            &opts,
            on_progress,
        )
        .await?
    };

    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print_report(&report);
    }
    if report.has_errors() {
        std::process::exit(1);
    }
    Ok(())
}

fn print_report(report: &UpdateReport) {
    if report.dry_run {
        eprintln!("Dry run: nothing was changed.");
    }
    for m in &report.mirrors {
        eprintln!(
            "{} [{}] added={} changed={} unchanged={} pruned={}",
            m.name, m.status, m.added, m.changed, m.unchanged, m.pruned
        );
        if let Some(reason) = &m.prune_blocked {
            eprintln!(
                "  prune skipped ({} candidates): {reason}",
                m.prune_candidates
            );
        }
        for p in m
            .problems
            .iter()
            .filter(|p| p.kind != "prune_blocked")
            .take(10)
        {
            eprintln!("  {} {}: {}", p.kind, p.source_path, p.message);
        }
        if m.problems.len() > 10 {
            eprintln!(
                "  ... {} more problem(s), see `colibri status`",
                m.problems.len() - 10
            );
        }
    }
    if let Some(l) = &report.library {
        eprintln!(
            "{} [{}] added={} updated={} unchanged={} restored={} source-missing={}",
            l.name, l.status, l.added, l.updated, l.unchanged, l.restored, l.source_missing
        );
        for b in l
            .books
            .iter()
            .filter(|b| b.outcome != "already_known")
            .take(20)
        {
            eprintln!("  {}: {}", b.outcome, b.title);
        }
        for p in l.problems.iter().take(10) {
            eprintln!("  {} {}: {}", p.kind, p.source_path, p.message);
        }
    }
    if let Some(i) = &report.index {
        eprintln!(
            "Index: {} embedded ({} chunks), {} unchanged, {} removed, {} orphaned purged, {} errors",
            i.indexed, i.chunks, i.unchanged, i.removed, i.orphans_removed, i.errors
        );
    }
}
