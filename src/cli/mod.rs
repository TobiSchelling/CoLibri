//! CLI command definitions and handlers.

pub mod bootstrap;
pub mod connectors;
pub mod doctor;
pub mod import;
pub mod index;
pub mod instructions;
pub mod migrate;
pub mod plugins;
pub mod profiles;
pub mod search;
pub mod serve;
pub mod sync;
pub mod tour;

use std::path::PathBuf;
use std::process::Command;

use clap::Subcommand;

/// Check whether a tool is available on `$PATH` via `which`.
pub(crate) fn tool_on_path(tool: &str) -> bool {
    Command::new("which")
        .arg(tool)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Check whether a tool spec (absolute path, relative path, or bare name) resolves to something
/// that exists on disk or on `$PATH`.
pub(crate) fn tool_available(spec: &str) -> bool {
    let trimmed = spec.trim();
    if trimmed.is_empty() {
        return false;
    }
    let path = PathBuf::from(trimmed);
    if path.is_absolute() || trimmed.contains('/') {
        return path.exists();
    }
    tool_on_path(trimmed)
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

        /// Classification for the initialized job (default: internal)
        #[arg(long, default_value = "internal")]
        classification: String,

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

    /// Inspect or apply storage migrations
    Migrate {
        /// Show pending/applied migrations without changing files
        #[arg(long)]
        dry_run: bool,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Show embedding profiles, routing policy, and index readiness
    Profiles {
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Run plugin commands
    #[command(hide = true)]
    Plugins {
        #[command(subcommand)]
        command: PluginCommands,
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

    /// Index markdown corpus into LanceDB
    Index {
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

        /// Filter by classification
        #[arg(long)]
        classification: Option<String>,
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

    /// Import PDF or EPUB files into the library as markdown
    Import {
        /// Input file path (PDF or EPUB)
        input: PathBuf,

        /// Working directory for conversion tools (defaults to a temp dir)
        #[arg(short, long)]
        output_dir: Option<PathBuf>,

        /// PDF converter: docling (quality) or marker (speed)
        #[arg(long, default_value = "docling", value_parser = clap::value_parser!(import::PdfConverter))]
        converter: import::PdfConverter,

        /// Image handling: placeholder (default, no images), referenced (separate files), embedded (base64)
        #[arg(long, default_value = "placeholder", value_parser = clap::value_parser!(import::ImageMode))]
        image_mode: import::ImageMode,

        /// Directory for extracted images (not supported by `colibri import`)
        #[arg(long)]
        attachments_dir: Option<PathBuf>,

        /// Index the canonical corpus after import
        #[arg(long)]
        reindex: bool,
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

#[derive(Subcommand)]
pub enum PluginCommands {
    /// Execute a plugin manifest and validate emitted envelopes
    Run {
        /// Path to plugin_manifest.json
        #[arg(long)]
        manifest: PathBuf,

        /// JSON object passed as plugin config (mutually exclusive with --config-file)
        #[arg(long)]
        config_json: Option<String>,

        /// Path to JSON file used as plugin config (mutually exclusive with --config-json)
        #[arg(long)]
        config_file: Option<PathBuf>,

        /// Include envelope payloads in output (default: summary only)
        #[arg(long)]
        include_envelopes: bool,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Execute a plugin and persist envelopes into canonical storage
    Ingest {
        /// Path to plugin_manifest.json
        #[arg(long)]
        manifest: PathBuf,

        /// JSON object passed as plugin config (mutually exclusive with --config-file)
        #[arg(long)]
        config_json: Option<String>,

        /// Path to JSON file used as plugin config (mutually exclusive with --config-json)
        #[arg(long)]
        config_file: Option<PathBuf>,

        /// Validate and report writes without mutating canonical storage
        #[arg(long)]
        dry_run: bool,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Execute a plugin incrementally using persisted sync cursor state
    Sync {
        /// Path to plugin_manifest.json
        #[arg(long)]
        manifest: PathBuf,

        /// JSON object passed as plugin config (mutually exclusive with --config-file)
        #[arg(long)]
        config_json: Option<String>,

        /// Path to JSON file used as plugin config (mutually exclusive with --config-json)
        #[arg(long)]
        config_file: Option<PathBuf>,

        /// Validate and report writes without mutating canonical storage/state
        #[arg(long)]
        dry_run: bool,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Execute all configured plugin sync jobs from config.yaml
    SyncAll {
        /// Restrict to specific job id(s); may be repeated
        #[arg(long = "job")]
        jobs: Vec<String>,

        /// Also run jobs marked as disabled in config
        #[arg(long)]
        include_disabled: bool,

        /// Stop on first failed job
        #[arg(long)]
        fail_fast: bool,

        /// Index canonical corpus after successful sync run
        #[arg(long)]
        index: bool,

        /// Force full rebuild for optional index step
        #[arg(long)]
        index_force: bool,

        /// Validate and report writes without mutating canonical storage/state
        #[arg(long)]
        dry_run: bool,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Show configured plugin jobs from config.yaml
    Jobs {
        /// Output as JSON
        #[arg(long)]
        json: bool,

        /// Include filesystem validation status for manifest paths
        #[arg(long)]
        validate_manifests: bool,
    },

    /// Run a plugin's interactive configuration wizard
    Configure {
        /// Plugin job id (from plugins.jobs[].id in config.yaml)
        job_id: String,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Inspect or reset persisted plugin sync state
    State {
        #[command(subcommand)]
        command: PluginStateCommands,
    },
}

#[derive(Subcommand)]
pub enum PluginStateCommands {
    /// List all persisted plugin sync entries
    List {
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Show sync entry for a specific manifest + config pair
    Show {
        /// Path to plugin_manifest.json
        #[arg(long)]
        manifest: PathBuf,

        /// JSON object passed as plugin config (mutually exclusive with --config-file)
        #[arg(long)]
        config_json: Option<String>,

        /// Path to JSON file used as plugin config (mutually exclusive with --config-json)
        #[arg(long)]
        config_file: Option<PathBuf>,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Remove sync entry for a specific manifest + config pair
    Reset {
        /// Path to plugin_manifest.json
        #[arg(long)]
        manifest: PathBuf,

        /// JSON object passed as plugin config (mutually exclusive with --config-file)
        #[arg(long)]
        config_json: Option<String>,

        /// Path to JSON file used as plugin config (mutually exclusive with --config-json)
        #[arg(long)]
        config_file: Option<PathBuf>,

        /// Required safety flag to confirm reset action
        #[arg(long)]
        yes: bool,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },
}
