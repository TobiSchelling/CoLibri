//! `colibri doctor` — health check command.

use serde::Serialize;

use crate::config::{self, load_config, SCHEMA_VERSION};
use crate::embedding::check_ollama;
use crate::index_meta::read_index_meta;
use crate::mcp;

#[derive(Debug, Serialize)]
struct DoctorSourceStatus {
    name: String,
    path: String,
    status: String,
}

#[derive(Debug, Serialize)]
struct DoctorReport {
    strict: bool,
    strict_violation: bool,
    config_path: String,
    config_ok: bool,
    config_error: Option<String>,
    colibri_home: Option<String>,
    active_generation: Option<String>,
    migration_up_to_date: Option<bool>,
    generation_count: Option<usize>,
    ollama_reachable: Option<bool>,
    ollama_error: Option<String>,
    index_status: Option<String>,
    index_schema_version: Option<u32>,
    index_file_count: Option<u64>,
    index_chunk_count: Option<u64>,
    index_model: Option<String>,
    serving_queryable_profiles: Option<usize>,
    serving_total_profiles: Option<usize>,
    serving_issues: Vec<String>,
    sources: Vec<DoctorSourceStatus>,
}

impl DoctorReport {
    fn new(strict: bool) -> Self {
        Self {
            strict,
            strict_violation: false,
            config_path: config::AppConfig::config_path().display().to_string(),
            config_ok: false,
            config_error: None,
            colibri_home: None,
            active_generation: None,
            migration_up_to_date: None,
            generation_count: None,
            ollama_reachable: None,
            ollama_error: None,
            index_status: None,
            index_schema_version: None,
            index_file_count: None,
            index_chunk_count: None,
            index_model: None,
            serving_queryable_profiles: None,
            serving_total_profiles: None,
            serving_issues: Vec::new(),
            sources: Vec::new(),
        }
    }
}

pub async fn run(strict: bool, json: bool) -> anyhow::Result<()> {
    if !json {
        eprintln!("CoLibri Doctor");
        eprintln!("==============\n");
    }

    let mut report = DoctorReport::new(strict);
    let mut strict_violation = false;

    // 1. Config
    if !json {
        eprint!("Config ... ");
    }
    match load_config() {
        Ok(config) => {
            report.config_ok = true;
            report.colibri_home = Some(config.colibri_home.display().to_string());
            report.active_generation = Some(config.active_generation.clone());
            if !json {
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
                eprintln!("  CoLibri home: {}", config.colibri_home.display());
                eprintln!("  Data dir: {}", config.data_dir.display());
                eprintln!("  Canonical dir: {}", config.canonical_dir.display());
                eprintln!("  Metadata DB: {}", config.metadata_db_path.display());
                eprintln!("  Active generation: {}", config.active_generation);
                eprintln!(
                    "  Default embedding profile: {}",
                    config.default_embedding_profile
                );
                eprintln!("  LanceDB dir: {}", config.lancedb_dir.display());
            }

            // 1b. Migrations
            if !json {
                eprint!("\nMigrations ... ");
            }
            match config.inspect_migrations() {
                Ok(migration_report) if migration_report.up_to_date => {
                    report.migration_up_to_date = Some(true);
                    if !json {
                        eprintln!("OK");
                    }
                }
                Ok(migration_report) => {
                    report.migration_up_to_date = Some(false);
                    if !json {
                        eprintln!("PENDING");
                    }
                    for check in migration_report.checks {
                        if check.status != "ok" && !json {
                            let current = check
                                .current
                                .map(|v| format!("v{v}"))
                                .unwrap_or_else(|| "missing".to_string());
                            eprintln!(
                                "  {}: {} (current={}, target=v{})",
                                check.component, check.status, current, check.target
                            );
                        }
                    }
                    if !json {
                        eprintln!("  Run `colibri migrate` to apply pending migrations.");
                    }
                }
                Err(e) => {
                    if !json {
                        eprintln!("ERROR: {e}");
                    }
                }
            }

            // 1c. Generations
            if !json {
                eprint!("\nGenerations ... ");
            }
            match config.list_generations() {
                Ok(gens) => {
                    report.generation_count = Some(gens.len());
                    if !json {
                        eprintln!("OK ({})", gens.len());
                    }
                    if !json && !gens.is_empty() {
                        eprintln!("  {}", gens.join(", "));
                    }
                }
                Err(e) => {
                    if !json {
                        eprintln!("ERROR: {e}");
                    }
                }
            }

            // 2. Ollama
            if !json {
                eprint!("\nOllama ... ");
            }
            match check_ollama(&config.ollama_base_url).await {
                Ok(true) => {
                    report.ollama_reachable = Some(true);
                    if !json {
                        eprintln!("OK ({})", config.ollama_base_url);
                        eprintln!("  Model: {}", config.embedding_model);
                    }
                }
                Ok(false) => {
                    report.ollama_reachable = Some(false);
                    if !json {
                        eprintln!("UNREACHABLE ({})", config.ollama_base_url);
                        eprintln!("  Ollama is not running or not reachable.");
                    }
                }
                Err(e) => {
                    report.ollama_error = Some(e.to_string());
                    if !json {
                        eprintln!("ERROR: {e}");
                    }
                }
            }

            // 3. Index
            if !json {
                eprint!("\nIndex ... ");
            }
            match read_index_meta(&config.lancedb_dir) {
                Ok(meta) if meta.is_empty() => {
                    report.index_status = Some("not_found".into());
                    if !json {
                        eprintln!("NOT FOUND");
                        eprintln!("  Run `colibri index` to create the index.");
                    }
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
                    report.index_schema_version = Some(version as u32);
                    report.index_file_count = Some(file_count);
                    report.index_chunk_count = Some(chunk_count);
                    report.index_model = Some(model.to_string());

                    if version == SCHEMA_VERSION as u64 {
                        report.index_status = Some("ok".into());
                        if !json {
                            eprintln!("OK (schema v{version})");
                        }
                    } else {
                        report.index_status = Some("outdated".into());
                        if !json {
                            eprintln!("OUTDATED (v{version}, need v{SCHEMA_VERSION})");
                            eprintln!("  Run `colibri index --force` to rebuild.");
                        }
                    }
                    if !json {
                        eprintln!("  Files: {file_count}");
                        eprintln!("  Chunks: {chunk_count}");
                        eprintln!("  Model: {model}");
                    }

                    if !json {
                        if let Some(last) = meta.get("last_indexed_at").and_then(|v| v.as_str()) {
                            eprintln!("  Last indexed: {last}");
                        }
                    }
                }
                Err(e) => {
                    report.index_status = Some("error".into());
                    if !json {
                        eprintln!("ERROR: {e}");
                    }
                }
            }

            // 3b. Serving alignment across active generation/profile indexes
            if !json {
                eprint!("\nServing alignment ... ");
            }
            match mcp::startup_report(&config) {
                Ok(serve_report) if serve_report.issues.is_empty() => {
                    report.serving_queryable_profiles = Some(serve_report.queryable_profiles);
                    report.serving_total_profiles = Some(serve_report.total_profiles);
                    if !json {
                        eprintln!("OK");
                    }
                }
                Ok(serve_report) => {
                    report.serving_queryable_profiles = Some(serve_report.queryable_profiles);
                    report.serving_total_profiles = Some(serve_report.total_profiles);
                    report.serving_issues = serve_report.issues.clone();
                    if !json {
                        eprintln!("WARN ({} issue(s))", report.serving_issues.len());
                        for issue in &report.serving_issues {
                            eprintln!("  - {}", issue);
                        }
                    }
                    if strict {
                        strict_violation = true;
                    }
                }
                Err(e) => {
                    report.serving_issues = vec![e.to_string()];
                    if !json {
                        eprintln!("ERROR: {e}");
                    }
                    if strict {
                        strict_violation = true;
                    }
                }
            }

            // 4. Source directories
            if !json {
                eprintln!("\nSources:");
            }
            for source in &config.sources {
                let path = std::path::Path::new(&source.path);
                let status = if path.exists() { "OK" } else { "MISSING" };
                report.sources.push(DoctorSourceStatus {
                    name: source.display_name().to_string(),
                    path: source.path.clone(),
                    status: status.to_string(),
                });
                if !json {
                    eprintln!("  {} ({}) ... {status}", source.display_name(), source.path);
                }
            }
        }
        Err(e) => {
            report.config_error = Some(e.to_string());
            if !json {
                eprintln!("FAILED: {e}");
            }
            if strict {
                strict_violation = true;
            }
        }
    }

    report.strict_violation = strict_violation;
    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    }

    if strict && strict_violation {
        anyhow::bail!("doctor strict mode failed: serving alignment issues detected");
    }

    if !json {
        eprintln!();
    }
    Ok(())
}
