//! CoLibri — Local RAG system for semantic search over markdown content.

mod canonical_store;
mod cli;
mod config;
mod embedding;
mod error;
mod index_meta;
mod indexer;
mod manifest;
mod mcp;
mod metadata_store;
mod plugin_host;
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
        cli::Commands::Doctor { strict, json } => cli::doctor::run(strict, json).await,
        cli::Commands::Migrate { dry_run, json } => cli::migrate::run(dry_run, json).await,
        cli::Commands::Profiles { json } => cli::profiles::run(json).await,
        cli::Commands::Plugins { command } => match command {
            cli::PluginCommands::Run {
                manifest,
                config_json,
                config_file,
                include_envelopes,
                json,
            } => {
                cli::plugins::run(manifest, config_json, config_file, include_envelopes, json).await
            }
            cli::PluginCommands::Ingest {
                manifest,
                config_json,
                config_file,
                dry_run,
                json,
            } => cli::plugins::ingest(manifest, config_json, config_file, dry_run, json).await,
            cli::PluginCommands::Sync {
                manifest,
                config_json,
                config_file,
                dry_run,
                json,
            } => cli::plugins::sync(manifest, config_json, config_file, dry_run, json).await,
            cli::PluginCommands::SyncAll {
                jobs,
                include_disabled,
                fail_fast,
                index_canonical,
                generation,
                activate,
                index_force,
                dry_run,
                json,
            } => {
                cli::plugins::sync_all(cli::plugins::SyncAllOptions {
                    requested_jobs: jobs,
                    include_disabled,
                    fail_fast,
                    index_canonical,
                    generation,
                    activate,
                    index_force,
                    dry_run,
                    json,
                })
                .await
            }
            cli::PluginCommands::Jobs {
                json,
                validate_manifests,
            } => cli::plugins::jobs(json, validate_manifests).await,
            cli::PluginCommands::State { command } => match command {
                cli::PluginStateCommands::List { json } => cli::plugins::state_list(json).await,
                cli::PluginStateCommands::Show {
                    manifest,
                    config_json,
                    config_file,
                    json,
                } => cli::plugins::state_show(manifest, config_json, config_file, json).await,
                cli::PluginStateCommands::Reset {
                    manifest,
                    config_json,
                    config_file,
                    yes,
                    json,
                } => cli::plugins::state_reset(manifest, config_json, config_file, yes, json).await,
            },
        },
        cli::Commands::Generations { command } => match command {
            cli::GenerationCommands::Create {
                generation,
                activate,
            } => cli::generations::create(generation, activate).await,
            cli::GenerationCommands::List { json } => cli::generations::list(json).await,
            cli::GenerationCommands::Activate {
                generation,
                allow_unready,
            } => cli::generations::activate(generation, allow_unready).await,
            cli::GenerationCommands::Delete {
                generation,
                confirm,
                force,
            } => cli::generations::delete(generation, confirm, force).await,
        },
        cli::Commands::Index {
            folder,
            canonical,
            generation,
            activate,
            force,
        } => cli::index::run(folder, canonical, generation, activate, force).await,
        cli::Commands::Instructions { output } => cli::instructions::run(output).await,
        cli::Commands::Search {
            query,
            limit,
            json,
            doc_type,
            folder,
        } => cli::search::run(query, limit, json, doc_type, folder).await,
        cli::Commands::Serve { check, json } => cli::serve::run(check, json).await,
        cli::Commands::Import {
            input,
            output_dir,
            converter,
            image_mode,
            attachments_dir,
            reindex,
        } => {
            cli::import::run(
                input,
                output_dir,
                converter,
                image_mode,
                attachments_dir,
                reindex,
            )
            .await
        }
    }
}
