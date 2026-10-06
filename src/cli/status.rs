//! `colibri status` — what is in CoLibri, what changed, what needs attention.

use std::collections::{BTreeMap, HashSet};

use serde::Serialize;

use crate::config::{load_config, AppConfig};
use crate::metadata_store::ProblemRow;

#[derive(Debug, Default, Serialize)]
pub struct CollectionStatus {
    pub name: String,
    pub kind: Option<String>,
    pub active: usize,
    pub removed: usize,
    pub source_missing: usize,
    /// Searchable documents whose chunks are missing or outdated.
    pub pending_index: usize,
    pub last_run_at: Option<String>,
    pub last_run_status: Option<String>,
    pub problems: Vec<ProblemRow>,
}

#[derive(Debug, Serialize)]
pub struct StatusReport {
    pub data_dir: String,
    pub index_ready: bool,
    pub index_issues: Vec<String>,
    /// Index chunks whose document is not searchable (should be 0 after an update).
    pub orphan_chunks: usize,
    pub total_chunks: usize,
    pub collections: Vec<CollectionStatus>,
}

/// Read-only snapshot of all collections.
pub async fn collect(config: &AppConfig) -> anyhow::Result<StatusReport> {
    let store = config.open_read()?;
    let mut by_name: BTreeMap<String, CollectionStatus> = BTreeMap::new();
    let entry = |map: &mut BTreeMap<String, CollectionStatus>, name: &str| {
        map.entry(name.to_string())
            .or_insert_with(|| CollectionStatus {
                name: name.to_string(),
                ..Default::default()
            });
    };
    for m in &config.mirrors {
        entry(&mut by_name, &m.name);
        by_name.get_mut(&m.name).unwrap().kind = Some("mirror".into());
    }
    if config.library.is_some() {
        let name = crate::config::LIBRARY_COLLECTION;
        entry(&mut by_name, name);
        by_name.get_mut(name).unwrap().kind = Some("library".into());
    }

    let docs = store.list_documents()?;
    let mut live: HashSet<&str> = HashSet::new();
    for d in &docs {
        entry(&mut by_name, &d.collection);
        let c = by_name.get_mut(&d.collection).unwrap();
        if d.is_searchable() {
            // Same definition as the indexer's orphan purge: chunks of a
            // searchable document are never orphans, even if outdated.
            live.insert(d.doc_id.as_str());
            c.active += 1;
            if d.source_missing {
                c.source_missing += 1;
            }
            if !d.is_index_current() {
                c.pending_index += 1;
            }
        } else {
            c.removed += 1;
        }
    }
    for run in store.list_collection_runs()? {
        entry(&mut by_name, &run.name);
        let c = by_name.get_mut(&run.name).unwrap();
        c.kind = Some(run.kind);
        c.last_run_at = run.last_run_at;
        c.last_run_status = run.last_run_status;
    }
    for p in store.list_problems()? {
        entry(&mut by_name, &p.collection);
        by_name.get_mut(&p.collection).unwrap().problems.push(p);
    }

    let ready = crate::serve_ready::check(config)?;
    let counts = crate::indexer::chunk_counts(config)
        .await?
        .unwrap_or_default();
    let total_chunks = counts.values().sum();
    let orphan_chunks = counts
        .iter()
        .filter(|(id, _)| !live.contains(id.as_str()))
        .map(|(_, n)| n)
        .sum();

    Ok(StatusReport {
        data_dir: config.colibri_home.display().to_string(),
        index_ready: ready.queryable,
        index_issues: ready.issues,
        orphan_chunks,
        total_chunks,
        collections: by_name.into_values().collect(),
    })
}

pub async fn run(json: bool) -> anyhow::Result<()> {
    let config = load_config()?;
    let report = collect(&config).await?;
    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }

    println!("CoLibri status ({})", report.data_dir);
    println!(
        "Index: {} ({} chunks, orphan chunks: {})",
        if report.index_ready {
            "ready"
        } else {
            "NOT READY"
        },
        report.total_chunks,
        report.orphan_chunks
    );
    for issue in &report.index_issues {
        println!("  - {issue}");
    }
    println!();
    for c in &report.collections {
        println!(
            "{} [{}] active={} removed={} source-missing={} pending-index={}",
            c.name,
            c.kind.as_deref().unwrap_or("collection"),
            c.active,
            c.removed,
            c.source_missing,
            c.pending_index
        );
        match (&c.last_run_at, &c.last_run_status) {
            (Some(at), Some(status)) => println!("  last run: {at} ({status})"),
            _ => println!("  last run: never"),
        }
        for p in c.problems.iter().take(10) {
            println!("  ! {} {}: {}", p.kind, p.source_path, p.message);
        }
        if c.problems.len() > 10 {
            println!("  ! ... {} more", c.problems.len() - 10);
        }
    }
    Ok(())
}
