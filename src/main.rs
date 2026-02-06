//! CoLibri — Local RAG system for semantic search over markdown content.

mod cli;
mod config;
mod embedding;
mod error;
mod index_meta;
mod indexer;
mod manifest;
mod mcp;
mod query;
mod sources;

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
        .init();

    let cli = Cli::parse();

    match cli.command {
        cli::Commands::Config => cli::config_tui::run().await,
        cli::Commands::Doctor => cli::doctor::run().await,
        cli::Commands::Index { folder, force } => cli::index::run(folder, force).await,
        cli::Commands::Instructions { output } => cli::instructions::run(output).await,
        cli::Commands::Search {
            query,
            limit,
            json,
            doc_type,
            folder,
        } => cli::search::run(query, limit, json, doc_type, folder).await,
        cli::Commands::Serve => cli::serve::run().await,
        cli::Commands::Import {
            input,
            output_dir,
            converter,
            reindex,
        } => cli::import::run(input, output_dir, converter, reindex).await,
    }
}
