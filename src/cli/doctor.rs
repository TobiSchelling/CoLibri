//! `colibri doctor` — health check command.

use serde::Serialize;

use super::tool_on_path;
use crate::config::{self, load_config, SCHEMA_VERSION};
use crate::connectors::ConnectorJob;
use crate::embedding::check_ollama;
use crate::index_meta::read_index_meta;
use crate::mcp;

#[derive(Debug, Serialize)]
struct DoctorConnectorStatus {
    id: String,
    connector_type: String,
    enabled: bool,
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
    connectors_configured: Option<usize>,
    connector_details: Vec<DoctorConnectorStatus>,
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
            connectors_configured: None,
            connector_details: Vec::new(),
        }
    }
}

/// Diagnose a single connector, returning per-connector status and issues.
fn diagnose_connector(job: &ConnectorJob) -> DoctorConnectorStatus {
    let mut issues = Vec::new();

    if job.connector_type == "filesystem" {
        // Check root_path exists.
        if let Some(raw_path) = super::config_string(&job.config, "root_path") {
            let expanded = super::connectors::expand_tilde(&raw_path);
            if !std::path::Path::new(&expanded).exists() {
                issues.push(format!("root_path {raw_path} does not exist"));
            }
        } else {
            issues.push("root_path is not configured".into());
        }

        // Check required tools based on extensions.
        let exts: Vec<String> =
            super::connectors::parse_string_array(&job.config, "include_extensions")
                .unwrap_or_default()
                .into_iter()
                .map(|s| s.to_lowercase())
                .collect();

        if exts.iter().any(|e| e == ".pdf") && !tool_on_path("docling") {
            issues.push("docling not found (needed for .pdf)".into());
        }
        if exts.iter().any(|e| e == ".docx" || e == ".epub") && !tool_on_path("pandoc") {
            issues.push("pandoc not found (needed for .docx/.epub)".into());
        }
        if exts.iter().any(|e| e == ".pptx")
            && !tool_on_path("markitdown")
            && !tool_on_path("pandoc")
        {
            issues.push("markitdown or pandoc not found (needed for .pptx)".into());
        }
    } else if job.connector_type == "zephyr_scale" {
        // Check project_key is configured.
        if super::config_string(&job.config, "project_key").is_none() {
            issues.push("project_key is not configured".into());
        }

        // Check API token is available (config field or env var).
        let token_env = super::config_string(&job.config, "token_env")
            .unwrap_or_else(|| "ZEPHYR_API_TOKEN".into());
        let has_token = super::config_string(&job.config, "token").is_some()
            || std::env::var(&token_env)
                .ok()
                .filter(|s| !s.is_empty())
                .is_some();
        if !has_token {
            issues.push(format!(
                "no API token — set `token` in config or env var {token_env}"
            ));
        }
    }

    let status = if issues.is_empty() {
        "ok".into()
    } else {
        "warn".into()
    };

    DoctorConnectorStatus {
        id: job.id.clone(),
        connector_type: job.connector_type.clone(),
        enabled: job.enabled,
        status,
        issues,
    }
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
                eprintln!("  CoLibri home: {}", config.colibri_home.display());
                eprintln!("  Data dir: {}", config.colibri_home.display());
                eprintln!("  Canonical dir: {}", config.canonical_dir.display());
                eprintln!("  Metadata DB: {}", config.metadata_db_path.display());
                eprintln!("  Active generation: {}", config.active_generation);
                eprintln!(
                    "  Default embedding profile: {}",
                    config.default_embedding_profile
                );
                eprintln!("  LanceDB dir: {}", config.lancedb_dir.display());
            }

            // 1b. Connectors
            if !json {
                eprint!("\nConnectors ... ");
            }
            let n = config.connector_jobs.len();
            report.connectors_configured = Some(n);

            let mut connector_details = Vec::new();
            for job in &config.connector_jobs {
                connector_details.push(diagnose_connector(job));
            }

            let has_connector_issues = connector_details.iter().any(|d| d.status == "warn");
            if n == 0 {
                if !json {
                    eprintln!("OK (no connectors configured)");
                }
            } else if has_connector_issues {
                if !json {
                    eprintln!("WARN ({n} configured)");
                }
            } else if !json {
                eprintln!("OK ({n} configured)");
            }

            if !json {
                for detail in &connector_details {
                    let label = if detail.enabled {
                        "enabled"
                    } else {
                        "disabled"
                    };
                    let status_label = detail.status.to_uppercase();
                    eprintln!(
                        "  - {} ({}) [{label}] {status_label}",
                        detail.id, detail.connector_type
                    );
                    for issue in &detail.issues {
                        eprintln!("    - {issue}");
                    }
                }
            }

            report.connector_details = connector_details;

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
