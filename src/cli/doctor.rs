//! `colibri doctor` — health check command.

use std::path::PathBuf;
use std::process::Command;

use serde::Serialize;

use crate::config::{self, load_config, SCHEMA_VERSION};
use crate::embedding::check_ollama;
use crate::index_meta::read_index_meta;
use crate::mcp;
use crate::plugin_host::load_plugin_manifest;

#[derive(Debug, Serialize)]
struct DoctorSourceStatus {
    name: String,
    path: String,
    status: String,
}

#[derive(Debug, Serialize)]
struct DoctorPluginJobStatus {
    id: String,
    enabled: bool,
    manifest: String,
    plugin_id: Option<String>,
    runtime: Option<String>,
    entrypoint: Option<String>,
    status: String,
    issues: Vec<String>,
}

#[derive(Debug, Serialize)]
struct DoctorReport {
    strict: bool,
    strict_violation: bool,
    config_path: String,
    sqlite3_available: Option<bool>,
    sqlite3_error: Option<String>,
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
    plugin_jobs: Vec<DoctorPluginJobStatus>,
}

impl DoctorReport {
    fn new(strict: bool) -> Self {
        Self {
            strict,
            strict_violation: false,
            config_path: config::AppConfig::config_path().display().to_string(),
            sqlite3_available: None,
            sqlite3_error: None,
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
            plugin_jobs: Vec::new(),
        }
    }
}

fn resolve_entrypoint(manifest_path: &PathBuf, entrypoint: &str) -> PathBuf {
    let manifest_dir = manifest_path
        .parent()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    let candidate = PathBuf::from(entrypoint);
    if candidate.is_absolute() {
        candidate
    } else {
        manifest_dir.join(candidate)
    }
}

fn tool_on_path(tool: &str) -> bool {
    Command::new("which")
        .arg(tool)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn tool_available(spec: &str) -> bool {
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

fn config_string(config: &serde_json::Value, key: &str) -> Option<String> {
    config
        .get(key)
        .and_then(|v| v.as_str())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

pub async fn run(strict: bool, json: bool) -> anyhow::Result<()> {
    if !json {
        eprintln!("CoLibri Doctor");
        eprintln!("==============\n");
    }

    let mut report = DoctorReport::new(strict);
    let mut strict_violation = false;

    // 0. Runtime prerequisites
    if !json {
        eprint!("SQLite ... ");
    }
    match crate::metadata_store::require_sqlite3() {
        Ok(_) => {
            report.sqlite3_available = Some(true);
            if !json {
                eprintln!("OK");
            }
        }
        Err(e) => {
            report.sqlite3_available = Some(false);
            report.sqlite3_error = Some(e.to_string());
            if !json {
                eprintln!("ERROR: {e}");
            }
            if strict {
                strict_violation = true;
            }
        }
    }

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

            // 1b. Plugins
            if !json {
                eprint!("\nPlugins ... ");
            }
            if config.plugin_jobs.is_empty() {
                if !json {
                    eprintln!("OK (no plugin jobs configured)");
                }
            } else {
                if !json {
                    eprintln!("OK ({})", config.plugin_jobs.len());
                }
                for job in &config.plugin_jobs {
                    let mut issues: Vec<String> = Vec::new();
                    let mut status = "ok".to_string();

                    if !job.manifest.exists() {
                        status = "missing_manifest".into();
                        issues.push("manifest file not found".into());
                        if strict && job.enabled {
                            strict_violation = true;
                        }
                        report.plugin_jobs.push(DoctorPluginJobStatus {
                            id: job.id.clone(),
                            enabled: job.enabled,
                            manifest: job.manifest.display().to_string(),
                            plugin_id: None,
                            runtime: None,
                            entrypoint: None,
                            status,
                            issues,
                        });
                        continue;
                    }

                    let manifest = match load_plugin_manifest(&job.manifest) {
                        Ok(m) => m,
                        Err(e) => {
                            status = "invalid_manifest".into();
                            issues.push(e.to_string());
                            if strict && job.enabled {
                                strict_violation = true;
                            }
                            report.plugin_jobs.push(DoctorPluginJobStatus {
                                id: job.id.clone(),
                                enabled: job.enabled,
                                manifest: job.manifest.display().to_string(),
                                plugin_id: None,
                                runtime: None,
                                entrypoint: None,
                                status,
                                issues,
                            });
                            continue;
                        }
                    };

                    let entrypoint_path = resolve_entrypoint(&job.manifest, &manifest.entrypoint);
                    if !entrypoint_path.exists() {
                        status = "warn".into();
                        issues.push(format!(
                            "entrypoint not found: {}",
                            entrypoint_path.display()
                        ));
                    }

                    match manifest.runtime.as_str() {
                        "python" => {
                            if !tool_on_path("python3") {
                                status = "warn".into();
                                issues.push("python3 not found on PATH".into());
                            }
                        }
                        "external" | "rust" => {
                            if entrypoint_path.exists() {
                                if let Ok(meta) = std::fs::metadata(&entrypoint_path) {
                                    #[cfg(unix)]
                                    {
                                        use std::os::unix::fs::PermissionsExt;
                                        if meta.permissions().mode() & 0o111 == 0 {
                                            status = "warn".into();
                                            issues.push(format!(
                                                "entrypoint is not executable: {}",
                                                entrypoint_path.display()
                                            ));
                                        }
                                    }
                                }
                            }
                        }
                        "wasm" => {
                            status = "warn".into();
                            issues.push(
                                "wasm runtime declared, but no wasm runner is implemented".into(),
                            );
                        }
                        other => {
                            status = "warn".into();
                            issues.push(format!("unsupported runtime: {other}"));
                        }
                    }

                    if let Some(req) = &manifest.requirements {
                        // Tools
                        if let Some(tools) = &req.tools {
                            for tool in tools {
                                let optional = tool.optional.unwrap_or(false);
                                let spec = if let Some(key) = tool.check_from_config.as_deref() {
                                    config_string(&job.config, key).or_else(|| tool.default.clone())
                                } else {
                                    tool.check.clone()
                                };

                                let Some(spec) = spec else {
                                    continue;
                                };
                                if !tool_available(&spec) {
                                    status = "warn".into();
                                    let mut msg = format!("missing tool: {spec}");
                                    if let Some(hint) = tool.install_hint.as_deref() {
                                        msg.push_str(&format!(" ({hint})"));
                                    } else if let Some(formula) = tool.brew.as_deref() {
                                        msg.push_str(&format!(" (brew install {formula})"));
                                    } else if let Some(cask) = tool.brew_cask.as_deref() {
                                        msg.push_str(&format!(" (brew install --cask {cask})"));
                                    } else if let Some(pkg) = tool.pipx.as_deref() {
                                        msg.push_str(&format!(" (pipx install {pkg})"));
                                    }
                                    if optional {
                                        msg = format!("optional {msg}");
                                    }
                                    issues.push(msg);
                                }
                            }
                        }

                        // Env vars
                        if let Some(env) = &req.env {
                            for var in env {
                                let required = var.required;
                                let name = if let Some(key) = var.name_from_config.as_deref() {
                                    config_string(&job.config, key).or_else(|| var.default.clone())
                                } else {
                                    var.name.clone()
                                };
                                let Some(name) = name else {
                                    continue;
                                };
                                let present =
                                    std::env::var(&name).ok().is_some_and(|v| !v.is_empty());
                                if required && !present {
                                    status = "warn".into();
                                    let mut msg = format!("required env var not set: {name}");
                                    if let Some(hint) = var.hint.as_deref() {
                                        msg.push_str(&format!(" ({hint})"));
                                    }
                                    issues.push(msg);
                                }
                            }
                        }
                    }

                    if !json {
                        let label = if job.enabled { "enabled" } else { "disabled" };
                        if issues.is_empty() {
                            eprintln!("  - {} ({}) [{}] OK", job.id, manifest.plugin_id, label);
                        } else {
                            eprintln!(
                                "  - {} ({}) [{}] {}: {}",
                                job.id,
                                manifest.plugin_id,
                                label,
                                status.to_uppercase(),
                                issues.join("; ")
                            );
                        }
                    }

                    report.plugin_jobs.push(DoctorPluginJobStatus {
                        id: job.id.clone(),
                        enabled: job.enabled,
                        manifest: job.manifest.display().to_string(),
                        plugin_id: Some(manifest.plugin_id),
                        runtime: Some(manifest.runtime),
                        entrypoint: Some(manifest.entrypoint),
                        status,
                        issues,
                    });
                }
            }

            // 1c. Migrations
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
        anyhow::bail!("doctor strict mode failed");
    }

    if !json {
        eprintln!();
    }
    Ok(())
}
