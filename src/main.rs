//! CoLibri — Local RAG system for semantic search over markdown content.

mod bundled_plugins;
mod canonical_store;
mod cli;
mod config;
mod embedding;
mod envelope;
mod error;
mod index_meta;
mod indexer;
mod mcp;
mod metadata_store;
mod plugin_host;
mod plugin_requirements;
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
            classification,
            non_interactive,
            json,
        } => {
            cli::bootstrap::run(cli::bootstrap::BootstrapOptions {
                config_path,
                data_dir,
                init_path,
                classification,
                non_interactive,
                json,
            })
            .await
        }
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
                index,
                index_force,
                dry_run,
                json,
            } => {
                cli::plugins::sync_all(cli::plugins::SyncAllOptions {
                    requested_jobs: jobs,
                    include_disabled,
                    fail_fast,
                    index,
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
            cli::PluginCommands::Configure { job_id, json } => {
                cli::plugins::configure(job_id, json).await
            }
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
        cli::Commands::Index { force } => cli::index::run(force).await,
        cli::Commands::Sync {
            jobs,
            include_disabled,
            fail_fast,
            no_index,
            force,
            dry_run,
            json,
        } => {
            cli::sync::run(
                jobs,
                include_disabled,
                fail_fast,
                no_index,
                force,
                dry_run,
                json,
            )
            .await
        }
        cli::Commands::Instructions { output } => cli::instructions::run(output).await,
        cli::Commands::Tour { topic } => cli::tour::run(topic).await,
        cli::Commands::Search {
            query,
            limit,
            json,
            doc_type,
            classification,
        } => cli::search::run(query, limit, json, doc_type, classification).await,
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
