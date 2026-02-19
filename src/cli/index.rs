//! `colibri index` — index markdown corpus command.

use std::sync::Mutex;

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

use crate::config::{load_config, FolderProfile, IndexMode};
use crate::embedding::EMBED_BATCH_SIZE;
use crate::indexer::{index_library, IndexEvent};

/// CLI progress handler that drives indicatif bars from `IndexEvent`s.
pub struct CliProgress {
    mp: MultiProgress,
    state: Mutex<CliProgressState>,
}

#[derive(Default)]
struct CliProgressState {
    read_pb: Option<ProgressBar>,
    embed_pb: Option<ProgressBar>,
    write_pb: Option<ProgressBar>,
    total_batches: usize,
}

impl CliProgress {
    fn new() -> Self {
        Self {
            mp: MultiProgress::new(),
            state: Mutex::new(CliProgressState::default()),
        }
    }

    fn handle(&self, event: IndexEvent) {
        let mut st = self.state.lock().unwrap();
        match event {
            IndexEvent::SourceStart { name } => {
                let _ = self.mp.println(format!("\n{name}"));
            }
            IndexEvent::Reading { done, total } => {
                if st.read_pb.is_none() {
                    let style = ProgressStyle::with_template("  Reading    [{bar:28}] {pos}/{len}")
                        .unwrap()
                        .progress_chars("##-");
                    let pb = self.mp.add(ProgressBar::new(total as u64));
                    pb.set_style(style);
                    st.read_pb = Some(pb);
                }
                if let Some(pb) = &st.read_pb {
                    pb.set_position(done as u64);
                }
            }
            IndexEvent::Embedding {
                chunks_done,
                total_chunks,
            } => {
                // Finish reading bar on first embedding event
                if let Some(pb) = st.read_pb.take() {
                    pb.finish_and_clear();
                }
                if st.embed_pb.is_none() {
                    let total_batches = total_chunks.div_ceil(EMBED_BATCH_SIZE);
                    st.total_batches = total_batches;
                    let style = ProgressStyle::with_template(
                        "  Embedding  [{bar:28}] {pos}/{len} chunks  batch {msg}",
                    )
                    .unwrap()
                    .progress_chars("##-");
                    let pb = self.mp.add(ProgressBar::new(total_chunks as u64));
                    pb.set_style(style);
                    pb.set_message(format!("0/{total_batches}"));
                    st.embed_pb = Some(pb);
                }
                if let Some(pb) = &st.embed_pb {
                    pb.set_position(chunks_done as u64);
                    let batch_num = chunks_done.div_ceil(EMBED_BATCH_SIZE);
                    pb.set_message(format!("{batch_num}/{}", st.total_batches));
                }
            }
            IndexEvent::Writing => {
                // Finish embedding bar
                if let Some(pb) = st.embed_pb.take() {
                    pb.finish_and_clear();
                }
                let pb = self.mp.add(ProgressBar::new_spinner());
                pb.set_style(
                    ProgressStyle::with_template("  Writing    {spinner} {msg}")
                        .unwrap()
                        .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏", "✓"]),
                );
                pb.set_message("Updating index...");
                pb.enable_steady_tick(std::time::Duration::from_millis(80));
                st.write_pb = Some(pb);
            }
            IndexEvent::SourceComplete { name, result } => {
                // Finish any active bar
                if let Some(pb) = st.write_pb.take() {
                    pb.finish_and_clear();
                }
                let _ = self.mp.println(format!(
                    "  {name}: {} files indexed ({} chunks)",
                    result.files_indexed, result.total_chunks
                ));
            }
            IndexEvent::SourceUnchanged {
                name,
                skipped,
                deleted,
            } => {
                let deleted_str = if deleted > 0 {
                    format!(", {deleted} removed")
                } else {
                    String::new()
                };
                let _ = self
                    .mp
                    .println(format!("  {name}: {skipped} unchanged{deleted_str}"));
            }
            IndexEvent::Warning { message } => {
                let _ = self.mp.println(format!("  ⚠ {message}"));
            }
            IndexEvent::Done(_) => {}
        }
    }
}

pub async fn run(
    folder: Option<String>,
    canonical: bool,
    generation: Option<String>,
    activate: bool,
    force: bool,
) -> anyhow::Result<()> {
    let base_config = load_config()?;
    if activate && generation.is_none() {
        anyhow::bail!("`--activate` requires `--generation <id>`");
    }
    let mut config = if let Some(generation) = generation {
        base_config.with_generation(&generation)?
    } else {
        base_config.clone()
    };

    if canonical {
        let canonical_sources = canonical_source_profiles(&config);
        if canonical_sources.is_empty() {
            anyhow::bail!(
                "No canonical markdown sources found under {}",
                config.canonical_dir.display()
            );
        }
        config.sources = canonical_sources;
    }

    // Print header
    let profiles: Vec<_> = if let Some(ref filter) = folder {
        config
            .sources
            .iter()
            .filter(|p| p.display_name() == filter)
            .collect()
    } else {
        config.sources.iter().collect()
    };
    let sources_str: Vec<String> = profiles
        .iter()
        .map(|p| {
            format!(
                "{} ({})",
                p.display_name(),
                match p.mode {
                    crate::config::IndexMode::Static => "static",
                    crate::config::IndexMode::Incremental => "incremental",
                }
            )
        })
        .collect();
    if sources_str.is_empty() {
        anyhow::bail!("No matching sources selected for indexing.");
    }
    eprintln!("Indexing: {}", sources_str.join(", "));
    if canonical {
        eprintln!("Source mode: canonical");
    }
    eprintln!("Generation: {}", config.active_generation);
    if force && folder.is_none() {
        eprintln!("Mode: full rebuild");
    }
    if activate {
        eprintln!("Will activate generation on success");
    }

    let progress = CliProgress::new();

    let result = index_library(&config, folder.as_deref(), force, |e| {
        progress.handle(e);
    })
    .await?;

    if activate {
        base_config.set_active_generation(&config.active_generation)?;
        eprintln!("Activated generation: {}", config.active_generation);
    }

    // Print summary
    let mut summary_parts = vec![format!(
        "{} files indexed ({} chunks)",
        result.files_indexed, result.total_chunks
    )];
    if result.files_skipped > 0 {
        summary_parts.push(format!("{} skipped", result.files_skipped));
    }
    if result.files_deleted > 0 {
        summary_parts.push(format!("{} removed", result.files_deleted));
    }
    if result.errors > 0 {
        summary_parts.push(format!("{} errors", result.errors));
    }
    eprintln!("\nDone: {}", summary_parts.join(", "));

    // Exit with non-zero if there were errors
    if result.errors > 0 {
        std::process::exit(1);
    }

    Ok(())
}

fn canonical_source_profiles(config: &crate::config::AppConfig) -> Vec<FolderProfile> {
    let mut profiles = Vec::new();
    for class in ["restricted", "confidential", "internal", "public"] {
        let class_dir = config.canonical_dir.join(class);
        if !class_dir.exists() || !class_dir.is_dir() {
            continue;
        }
        profiles.push(FolderProfile {
            path: class_dir.to_string_lossy().to_string(),
            mode: IndexMode::Incremental,
            doc_type: "note".into(),
            chunk_size: None,
            chunk_overlap: None,
            extensions: vec![".md".into()],
            name: Some(format!("canonical-{class}")),
            classification: class.to_string(),
        });
    }
    profiles
}
