//! CoLibri — Local RAG system for semantic search over markdown content.

// Bump beyond default 128 — `lance` v1.0.1 macros otherwise overflow the depth
// limit on macos-14 + rust 1.85.0 + newer cargo, causing release builds to fail
// with "queries overflow the depth limit!". See past CI fix attempts in commit
// log (3193899, d9b9cad) which pinned the runner/toolchain but didn't help.
#![recursion_limit = "512"]

mod canonical_store;
mod cli;
mod config;
mod embedding;
mod error;
mod fetch;
mod index_meta;
mod indexer;
mod ingest;
mod lock;
mod mcp;
mod metadata_store;
mod power;
mod query;
mod serve_ready;

use clap::Parser;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(
    name = "colibri",
    version,
    about = "Local RAG system for semantic search"
)]
struct Cli {
    #[command(subcommand)]
    command: cli::Commands,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing with RUST_LOG env filter
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_target(false)
        // IMPORTANT: keep stdout reserved for command output / protocols (e.g. MCP stdio).
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();

    match cli.command {
        cli::Commands::Bootstrap {
            config_path,
            data_dir,
            init_path,
            non_interactive,
            json,
        } => {
            cli::bootstrap::run(cli::bootstrap::BootstrapOptions {
                config_path,
                data_dir,
                init_path,
                non_interactive,
                json,
            })
            .await
        }
        cli::Commands::Doctor { strict, json } => cli::doctor::run(strict, json).await,
        cli::Commands::Reset { yes } => cli::reset::run(yes).await,
        cli::Commands::Update {
            names,
            dry_run,
            allow_mass_prune,
            retry_failed,
            no_index,
            force,
            json,
        } => {
            cli::update::run(
                crate::ingest::update::UpdateOptions {
                    names,
                    dry_run,
                    allow_mass_prune,
                    retry_failed,
                    no_index,
                    force_index: force,
                },
                json,
            )
            .await
        }
        cli::Commands::Status { json } => cli::status::run(json).await,
        cli::Commands::Index { force } => cli::index::run(force).await,
        cli::Commands::Instructions { output } => cli::instructions::run(output).await,
        cli::Commands::Tour { topic } => cli::tour::run(topic).await,
        cli::Commands::Search {
            query,
            limit,
            json,
            doc_type,
            collection,
            mode,
            path_includes,
            path_excludes,
            frontmatter,
            since,
            group_by_doc,
        } => {
            cli::search::run(
                query,
                limit,
                json,
                doc_type,
                collection,
                mode,
                path_includes,
                path_excludes,
                frontmatter,
                since,
                group_by_doc,
            )
            .await
        }
        cli::Commands::Serve { check, json } => cli::serve::run(check, json).await,
        cli::Commands::Add {
            paths,
            reconvert,
            retry_failed,
            dry_run,
            no_index,
            json,
        } => {
            cli::add::run(cli::add::AddOptions {
                paths,
                reconvert,
                retry_failed,
                dry_run,
                no_index,
                json,
            })
            .await
        }
        cli::Commands::List { json, removed } => cli::list::run(json, removed).await,
        cli::Commands::Remove { query } => cli::remove::run(query).await,
    }
}
