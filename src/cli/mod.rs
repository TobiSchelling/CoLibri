//! CLI command definitions and handlers.

pub mod config_tui;
pub mod doctor;
pub mod index;
pub mod instructions;
pub mod search;
pub mod serve;

use std::path::PathBuf;

use clap::Subcommand;

#[derive(Subcommand)]
pub enum Commands {
    /// Interactive TUI for managing configuration
    Config,

    /// Check system health (Ollama, config, index)
    Doctor,

    /// Index markdown corpus into LanceDB
    Index {
        /// Only index the source matching this name
        #[arg(long)]
        folder: Option<String>,

        /// Force full re-index regardless of mode
        #[arg(long)]
        force: bool,
    },

    /// Generate LLM instructions for using colibri
    Instructions {
        /// Output file path (default: ~/COLIBRI_INSTRUCTIONS.md)
        #[arg(short, long)]
        output: Option<PathBuf>,
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

        /// Filter by folder
        #[arg(long)]
        folder: Option<String>,
    },

    /// Start MCP stdio server
    Serve,
}
