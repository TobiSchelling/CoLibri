//! CLI command definitions and handlers.

pub mod add;
pub mod bootstrap;
pub mod connectors;
pub mod doctor;
pub mod index;
pub mod instructions;
pub mod list;
pub mod remove;
pub mod reset;
pub mod search;
pub mod serve;
pub mod status;
pub mod sync;
pub mod tour;
pub mod update;

use std::path::PathBuf;
use std::process::Command;

use clap::Subcommand;

use crate::query::SearchMode;

/// Check whether a tool is available on `$PATH` via `which`.
pub(crate) fn tool_on_path(tool: &str) -> bool {
    Command::new("which")
        .arg(tool)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// External tools missing for the file types a mirror includes.
pub(crate) fn missing_tools(include: &[String]) -> Vec<String> {
    let wants = |ext: &str| include.iter().any(|p| p.to_lowercase().contains(ext));
    let mut missing = Vec::new();
    if wants(".pdf") && !tool_on_path("docling") {
        missing.push("docling (pipx install docling), needed for PDF".to_string());
    }
    if (wants(".epub") || wants(".docx")) && !tool_on_path("pandoc") {
        missing.push("pandoc (brew install pandoc), needed for EPUB/DOCX".to_string());
    }
    if wants(".pptx") && !tool_on_path("markitdown") && !tool_on_path("pandoc") {
        missing.push("markitdown or pandoc, needed for PPTX".to_string());
    }
    missing
}

/// Extract a non-empty trimmed string value from a JSON object by key.
pub(crate) fn config_string(config: &serde_json::Value, key: &str) -> Option<String> {
    config
        .get(key)
        .and_then(|v| v.as_str())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

#[derive(Subcommand)]
pub enum Commands {
    /// First-time setup: write config, check dependencies, and initialize storage
    Bootstrap {
        /// Path to write config.yaml (default: ~/.config/colibri/config.yaml)
        #[arg(long)]
        config_path: Option<PathBuf>,

        /// CoLibri data directory (writes to config as data.directory)
        #[arg(long)]
        data_dir: Option<PathBuf>,

        /// Initialize a filesystem_documents job scanning this path (defaults to Markdown only)
        #[arg(long = "init-path", alias = "init-filesystem-markdown")]
        init_path: Option<PathBuf>,

        /// Do not prompt; require flags for paths/init and only print actions
        #[arg(long)]
        non_interactive: bool,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Check system health (Ollama, config, index)
    Doctor {
        /// Exit non-zero when serving alignment has any issue
        #[arg(long)]
        strict: bool,

        /// Output health report as JSON
        #[arg(long)]
        json: bool,
    },

    /// Delete all CoLibri data (metadata DB, canonical store, index) to rebuild from sources
    Reset {
        /// Skip the confirmation prompt
        #[arg(long)]
        yes: bool,
    },

    /// Manage native connectors
    Connectors {
        #[command(subcommand)]
        command: ConnectorCommands,
    },

    /// Sync configured sources into canonical store (and optionally index)
    Sync {
        /// Restrict to specific connector id(s); may be repeated
        #[arg(long = "connector")]
        connectors: Vec<String>,

        /// Also run jobs marked as disabled in config
        #[arg(long)]
        include_disabled: bool,

        /// Stop on first failed job
        #[arg(long)]
        fail_fast: bool,

        /// Skip indexing step (default is to index after a successful sync)
        #[arg(long)]
        no_index: bool,

        /// Force full rebuild for index step
        #[arg(long)]
        force: bool,

        /// Validate and report writes without mutating canonical storage/state
        #[arg(long)]
        dry_run: bool,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Reconcile mirrors (new, changed and deleted files) and update the index
    Update {
        /// Only these mirrors (default: all)
        names: Vec<String>,

        /// Show what would change without writing anything
        #[arg(long)]
        dry_run: bool,

        /// Apply deletions even above the configured mass-deletion limit
        #[arg(long)]
        allow_mass_prune: bool,

        /// Retry conversions that failed in an earlier run
        #[arg(long)]
        retry_failed: bool,

        /// Skip the indexing step
        #[arg(long)]
        no_index: bool,

        /// Re-embed everything
        #[arg(long)]
        force: bool,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Show collections, pending work, problems and index health
    Status {
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Index markdown corpus into LanceDB
    Index {
        /// Drop the index and re-embed everything
        #[arg(long)]
        force: bool,
    },

    /// Generate LLM instructions for using colibri
    Instructions {
        /// Output file path (default: ~/COLIBRI_INSTRUCTIONS.md)
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Explain core concepts and workflows
    Tour {
        /// Topic to show (run without this to list topics)
        topic: Option<String>,
    },

    /// Search the indexed library
    Search {
        /// Search query
        query: String,

        /// Maximum results to return
        #[arg(short, long, default_value_t = 5)]
        limit: usize,

        /// Output as JSON
        #[arg(long)]
        json: bool,

        /// Filter by document type
        #[arg(long)]
        doc_type: Option<String>,

        /// Restrict to one collection (connector id, e.g. `vault`, or `books`)
        #[arg(long)]
        collection: Option<String>,

        /// Search mode: hybrid (default), semantic, or keyword
        #[arg(long, value_enum, default_value_t = SearchMode::Hybrid)]
        mode: SearchMode,

        /// Restrict results to documents whose path contains any of these
        /// substrings. Repeatable: `--path-includes 03_PROJECTS --path-includes MEETINGS`.
        #[arg(long, value_name = "SUBSTRING")]
        path_includes: Vec<String>,

        /// Drop documents whose path contains any of these substrings.
        #[arg(long, value_name = "SUBSTRING")]
        path_excludes: Vec<String>,

        /// Equality filter on parsed frontmatter fields. KEY=VALUE, repeatable.
        /// e.g. `--frontmatter area=SIT --frontmatter status=active`.
        #[arg(long, value_name = "KEY=VALUE")]
        frontmatter: Vec<String>,

        /// Only return documents updated on or after this RFC 3339 timestamp.
        /// e.g. `--since 2026-04-01T00:00:00Z`.
        #[arg(long, value_name = "RFC3339")]
        since: Option<String>,

        /// Group results by document — best-matching chunk per file plus
        /// chunk_count. Default: false (chunk-level results).
        #[arg(long)]
        group_by_doc: bool,
    },

    /// Start MCP stdio server
    Serve {
        /// Run startup readiness checks only (do not start server)
        #[arg(long)]
        check: bool,

        /// Output check report as JSON (requires --check)
        #[arg(long)]
        json: bool,
    },

    /// Add books: sweep the configured library folders, or add the given files/folders
    #[command(alias = "import")]
    Add {
        /// Book files or folders (default: sweep `library.roots`)
        paths: Vec<PathBuf>,

        /// Convert again even if a cached conversion exists
        #[arg(long)]
        reconvert: bool,

        /// Retry conversions that failed in an earlier run
        #[arg(long)]
        retry_failed: bool,

        /// Show what would be added without writing anything
        #[arg(long)]
        dry_run: bool,

        /// Skip the indexing step
        #[arg(long)]
        no_index: bool,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// List the books in the library
    List {
        /// Output as JSON
        #[arg(long)]
        json: bool,

        /// Include books you removed
        #[arg(long)]
        removed: bool,
    },

    /// Remove a book (by id or title); later sweeps will not add it again
    Remove {
        /// Book id (see `colibri list --json`) or part of its title
        query: String,
    },
}

#[derive(Subcommand)]
pub enum ConnectorCommands {
    /// List configured connectors
    List {
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },
}
