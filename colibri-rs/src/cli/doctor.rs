//! `colibri doctor` — health check command.

use crate::config::{load_config, SCHEMA_VERSION};
use crate::embedding::check_ollama;
use crate::index_meta::read_index_meta;

pub async fn run() -> anyhow::Result<()> {
    eprintln!("CoLibri Doctor");
    eprintln!("==============\n");

    // 1. Config
    eprint!("Config ... ");
    match load_config() {
        Ok(config) => {
            eprintln!("OK ({})", config::AppConfig::config_path().display());
            eprintln!(
                "  Sources: {}",
                config
                    .sources
                    .iter()
                    .map(|s| s.display_name().to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            eprintln!("  Data dir: {}", config.data_dir.display());
            eprintln!("  LanceDB dir: {}", config.lancedb_dir.display());

            // 2. Ollama
            eprint!("\nOllama ... ");
            match check_ollama(&config.ollama_base_url).await {
                Ok(true) => {
                    eprintln!("OK ({})", config.ollama_base_url);
                    eprintln!("  Model: {}", config.embedding_model);
                }
                Ok(false) => {
                    eprintln!("UNREACHABLE ({})", config.ollama_base_url);
                    eprintln!("  Ollama is not running or not reachable.");
                }
                Err(e) => {
                    eprintln!("ERROR: {e}");
                }
            }

            // 3. Index
            eprint!("\nIndex ... ");
            match read_index_meta(&config.lancedb_dir) {
                Ok(meta) if meta.is_empty() => {
                    eprintln!("NOT FOUND");
                    eprintln!("  Run `colibri index` to create the index.");
                }
                Ok(meta) => {
                    let version = meta
                        .get("schema_version")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    let file_count = meta.get("file_count").and_then(|v| v.as_u64()).unwrap_or(0);
                    let chunk_count = meta
                        .get("chunk_count")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    let model = meta
                        .get("embedding_model")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown");

                    if version == SCHEMA_VERSION as u64 {
                        eprintln!("OK (schema v{version})");
                    } else {
                        eprintln!("OUTDATED (v{version}, need v{SCHEMA_VERSION})");
                        eprintln!("  Run `colibri index --force` to rebuild.");
                    }
                    eprintln!("  Files: {file_count}");
                    eprintln!("  Chunks: {chunk_count}");
                    eprintln!("  Model: {model}");

                    if let Some(last) = meta.get("last_indexed_at").and_then(|v| v.as_str()) {
                        eprintln!("  Last indexed: {last}");
                    }
                }
                Err(e) => {
                    eprintln!("ERROR: {e}");
                }
            }

            // 4. Source directories
            eprintln!("\nSources:");
            for source in &config.sources {
                let path = std::path::Path::new(&source.path);
                let status = if path.exists() { "OK" } else { "MISSING" };
                eprintln!("  {} ({}) ... {status}", source.display_name(), source.path);
            }
        }
        Err(e) => {
            eprintln!("FAILED: {e}");
        }
    }

    eprintln!();
    Ok(())
}

// Bring AppConfig into scope for the config_path() call
use crate::config;
