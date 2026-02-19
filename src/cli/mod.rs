//! CLI command definitions and handlers.

pub mod bootstrap;
pub mod config_tui;
pub mod doctor;
pub mod generations;
pub mod import;
pub mod index;
pub mod instructions;
pub mod migrate;
pub mod plugins;
pub mod profiles;
pub mod search;
pub mod serve;
pub mod tour;

use std::path::PathBuf;

use clap::Subcommand;

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

        /// Initialize a filesystem_markdown plugin job with this root_path
        #[arg(long)]
        init_filesystem_markdown: Option<PathBuf>,

        /// Classification for the initialized job (default: internal)
        #[arg(long, default_value = "internal")]
        classification: String,

        /// Do not prompt; require flags for paths/init and only print actions
        #[arg(long)]
        non_interactive: bool,

        /// Attempt to install missing tools (brew/pipx) and pull Ollama model
        #[arg(long)]
        install: bool,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Interactive TUI for managing configuration
    Config,

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
    Plugins {
        #[command(subcommand)]
        command: PluginCommands,
    },

    /// Manage index generations
    Generations {
        #[command(subcommand)]
        command: GenerationCommands,
    },

    /// Index markdown corpus into LanceDB
    Index {
        /// Only index the source matching this name
        #[arg(long)]
        folder: Option<String>,

        /// Index from managed canonical store (`COLIBRI_HOME/canonical`)
        #[arg(long)]
        canonical: bool,

        /// Target generation id (defaults to active generation)
        #[arg(long)]
        generation: Option<String>,

        /// Activate target generation after successful indexing
        #[arg(long)]
        activate: bool,

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

        /// Filter by folder
        #[arg(long)]
        folder: Option<String>,
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

        /// Output directory (defaults to books source from config)
        #[arg(short, long)]
        output_dir: Option<PathBuf>,

        /// PDF converter: docling (quality) or marker (speed)
        #[arg(long, default_value = "docling", value_parser = clap::value_parser!(import::PdfConverter))]
        converter: import::PdfConverter,

        /// Image handling: placeholder (default, no images), referenced (separate files), embedded (base64)
        #[arg(long, default_value = "placeholder", value_parser = clap::value_parser!(import::ImageMode))]
        image_mode: import::ImageMode,

        /// Directory for extracted images (only used with --image-mode=referenced)
        #[arg(long)]
        attachments_dir: Option<PathBuf>,

        /// Re-index the books folder after import
        #[arg(long)]
        reindex: bool,
    },
}

#[derive(Subcommand)]
pub enum GenerationCommands {
    /// Create a generation layout for all embedding profiles
    Create {
        /// Generation id (example: gen_2026_02_18_bge-m3_v1)
        generation: String,

        /// Activate generation after creation
        #[arg(long)]
        activate: bool,
    },

    /// List known generations and profile index status
    List {
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Activate (or create) an index generation
    Activate {
        /// Generation id (example: gen_2026_02_18_bge-m3_v1)
        generation: String,

        /// Allow activating a generation even if no profile is serve-ready
        #[arg(long)]
        allow_unready: bool,
    },

    /// Delete a generation directory
    Delete {
        /// Generation id to delete
        generation: String,

        /// Required confirmation token; must exactly match generation id
        #[arg(long)]
        confirm: String,

        /// Allow deleting currently active generation
        #[arg(long)]
        force: bool,
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
        index_canonical: bool,

        /// Target generation id for optional index step
        #[arg(long)]
        generation: Option<String>,

        /// Activate target generation after successful optional index step
        #[arg(long)]
        activate: bool,

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
