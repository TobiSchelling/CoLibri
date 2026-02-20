//! `colibri instructions` — generate LLM instructions for using colibri.

use std::path::PathBuf;

use crate::config::{load_config, AppConfig};
use crate::metadata_store::MetadataStore;
use crate::plugin_host::load_plugin_manifest;

/// Run the instructions command.
pub async fn run(output: Option<PathBuf>) -> anyhow::Result<()> {
    let config = load_config()?;
    let instructions = generate_instructions(&config)?;

    let output_path = output.unwrap_or_else(|| {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("COLIBRI_INSTRUCTIONS.md")
    });

    std::fs::write(&output_path, &instructions)?;
    eprintln!("Instructions written to: {}", output_path.display());
    eprintln!("\nYou can include this in your CLAUDE.md or project instructions.");

    Ok(())
}

fn generate_instructions(config: &AppConfig) -> anyhow::Result<String> {
    let mut md = String::new();

    // Header
    md.push_str("# Knowledge Base Search\n\n");
    md.push_str("Use `colibri search` to retrieve information from the local knowledge base.\n\n");

    // Search command
    md.push_str("## Search\n\n");
    md.push_str("```bash\n");
    md.push_str("colibri search \"<query>\"\n");
    md.push_str("colibri search \"<query>\" --doc-type <type>\n");
    md.push_str("colibri search \"<query>\" --limit <n>\n");
    md.push_str("```\n\n");

    // Available content
    md.push_str("## Available Content\n\n");
    md.push_str("Content is ingested into CoLibri's managed canonical markdown store and indexed from there.\n\n");

    if config.metadata_db_path.exists() {
        if let Ok(store) = MetadataStore::open(&config.metadata_db_path) {
            if let Ok(rows) = store.list_documents() {
                let mut total_live = 0usize;
                let mut by_type: std::collections::BTreeMap<String, usize> =
                    std::collections::BTreeMap::new();
                for row in rows {
                    if row.deleted {
                        continue;
                    }
                    total_live += 1;
                    *by_type.entry(row.doc_type).or_default() += 1;
                }

                md.push_str(&format!("- Total documents: {total_live}\n"));
                if !by_type.is_empty() {
                    md.push_str("- By type:\n");
                    for (doc_type, n) in by_type {
                        md.push_str(&format!("  - {doc_type}: {n}\n"));
                    }
                }
                md.push('\n');
            }
        }
    }

    if config.plugin_jobs.is_empty() {
        md.push_str("- Ingestion jobs: none configured\n");
        md.push_str(
            "  - Configure `plugins.jobs` in `config.yaml` and run `colibri plugins sync-all`.\n",
        );
    } else {
        md.push_str("- Ingestion jobs (plugins):\n");
        for job in &config.plugin_jobs {
            let plugin_id = load_plugin_manifest(&job.manifest)
                .ok()
                .map(|m| m.plugin_id)
                .unwrap_or_else(|| "unknown".into());
            let enabled = if job.enabled { "enabled" } else { "disabled" };
            md.push_str(&format!("  - {} ({plugin_id}) [{enabled}]\n", job.id));
        }
    }

    Ok(md)
}
