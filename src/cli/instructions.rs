//! `colibri instructions` — generate LLM instructions for using colibri.

use std::path::PathBuf;

use crate::config::{load_config, AppConfig, IndexMode};
use crate::manifest::{get_manifest_path, Manifest};

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

    let manifest_path = get_manifest_path(&config.data_dir);
    let manifest = Manifest::load(&manifest_path).ok();

    for source in &config.sources {
        if source.mode == IndexMode::Disabled {
            continue;
        }

        let (file_count, _) = if let Some(ref m) = manifest {
            count_source_files(m, &source.path)
        } else {
            (0, 0)
        };

        let description = get_source_description(&source.doc_type);

        md.push_str(&format!(
            "- **{}** (`--doc-type {}`): {} ({} files)\n",
            source.display_name(),
            source.doc_type,
            description,
            file_count
        ));
    }

    Ok(md)
}

fn count_source_files(manifest: &Manifest, source_path: &str) -> (usize, usize) {
    let source_id = crate::manifest::source_id_for_root(std::path::Path::new(source_path));
    let prefix = format!("{source_id}:");

    let mut file_count = 0;
    let mut chunk_count = 0;

    for (key, entry) in &manifest.files {
        if key.starts_with(&prefix) {
            file_count += 1;
            chunk_count += entry.chunk_count;
        }
    }

    (file_count, chunk_count)
}

fn get_source_description(doc_type: &str) -> &'static str {
    match doc_type.to_lowercase().as_str() {
        "book" => "Technical books and reference materials",
        "note" | "notes" => "Personal notes and documentation",
        "article" => "Articles and blog posts",
        "docs" | "documentation" => "Project documentation",
        "reference" => "Reference materials",
        "manual" => "User manuals and guides",
        "paper" => "Academic papers and research",
        "transcript" => "Meeting transcripts and recordings",
        _ => "General content",
    }
}
