//! `colibri index` — bring the vector/keyword index in line with the metadata DB.

use std::sync::Mutex;

use indicatif::{ProgressBar, ProgressStyle};

use crate::config::load_config;
use crate::embedding::OllamaEmbedder;
use crate::indexer::{index_library, IndexEvent, IndexOptions, IndexResult};

/// CLI progress handler that drives an indicatif bar from `IndexEvent`s.
pub struct CliProgress {
    bar: Mutex<Option<ProgressBar>>,
}

impl CliProgress {
    pub fn new() -> Self {
        Self {
            bar: Mutex::new(None),
        }
    }

    pub fn handle(&self, event: IndexEvent) {
        let mut bar = self.bar.lock().unwrap();
        match event {
            IndexEvent::Start {
                to_index,
                unchanged,
                removed,
            } => {
                eprintln!("Index: {to_index} to embed, {unchanged} unchanged, {removed} to remove");
                if to_index > 0 {
                    let pb = ProgressBar::new(to_index as u64);
                    pb.set_style(
                        ProgressStyle::with_template(
                            "  Embedding [{bar:28}] {pos}/{len} docs  {msg}",
                        )
                        .unwrap()
                        .progress_chars("##-"),
                    );
                    *bar = Some(pb);
                }
            }
            IndexEvent::Progress {
                docs_done,
                chunks_done,
            } => {
                if let Some(pb) = bar.as_ref() {
                    pb.set_position(docs_done as u64);
                    pb.set_message(format!("{chunks_done} chunks"));
                }
            }
            IndexEvent::Finalizing => {
                if let Some(pb) = bar.take() {
                    pb.finish_and_clear();
                }
                eprintln!("  Updating keyword index and compacting...");
            }
            IndexEvent::Warning { message } => match bar.as_ref() {
                Some(pb) => pb.println(format!("  ⚠ {message}")),
                None => eprintln!("  ⚠ {message}"),
            },
        }
    }
}

/// One-line summary of an indexing run.
pub fn summarize(result: &IndexResult) -> String {
    let mut parts = vec![format!(
        "{} docs indexed ({} chunks)",
        result.files_indexed, result.total_chunks
    )];
    if result.files_skipped > 0 {
        parts.push(format!("{} unchanged", result.files_skipped));
    }
    if result.files_deleted > 0 {
        parts.push(format!("{} removed", result.files_deleted));
    }
    if result.orphans_removed > 0 {
        parts.push(format!("{} orphaned docs purged", result.orphans_removed));
    }
    if result.errors > 0 {
        parts.push(format!("{} errors", result.errors));
    }
    parts.join(", ")
}

pub async fn run(force: bool) -> anyhow::Result<()> {
    let config = load_config()?;
    let (_lock, store) = config.open_for_write()?;
    let _awake = crate::power::keep_awake();
    if force {
        eprintln!("Mode: full rebuild");
    }

    let progress = CliProgress::new();
    let result = index_library(
        &config,
        &store,
        &OllamaEmbedder::from_config(&config),
        &IndexOptions {
            force,
            ..Default::default()
        },
        |e| progress.handle(e),
    )
    .await?;

    eprintln!("\nDone: {}", summarize(&result));
    if result.errors > 0 {
        std::process::exit(1);
    }
    Ok(())
}
