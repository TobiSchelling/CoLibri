//! `colibri doctor` — health check command.

use serde::Serialize;

use crate::config::{self, load_config};
use crate::connectors::ConnectorJob;
use crate::embedding::check_ollama;

#[derive(Debug, Serialize)]
struct DoctorConnectorStatus {
    id: String,
    connector_type: String,
    enabled: bool,
    status: String,
    issues: Vec<String>,
}

#[derive(Debug, Serialize, Default)]
struct DoctorReport {
    strict: bool,
    strict_violation: bool,
    config_path: String,
    config_ok: bool,
    config_error: Option<String>,
    colibri_home: Option<String>,
    metadata_documents: Option<usize>,
    metadata_error: Option<String>,
    ollama_reachable: Option<bool>,
    ollama_error: Option<String>,
    index_queryable: Option<bool>,
    index_chunk_count: Option<u64>,
    index_model: Option<String>,
    index_issues: Vec<String>,
    connectors_configured: Option<usize>,
    connector_details: Vec<DoctorConnectorStatus>,
    mirror_details: Vec<DoctorConnectorStatus>,
}

/// Diagnose a single connector, returning per-connector status and issues.
fn diagnose_connector(job: &ConnectorJob) -> DoctorConnectorStatus {
    let mut issues = Vec::new();

    if job.connector_type == "zephyr_scale" {
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

    if job.connector_type == "filesystem" {
        issues.push("filesystem connectors were replaced by `mirrors:`; move this entry".into());
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
    let say = |line: String| {
        if !json {
            eprintln!("{line}");
        }
    };
    say("CoLibri Doctor\n==============\n".into());

    let mut report = DoctorReport {
        strict,
        config_path: config::AppConfig::config_path().display().to_string(),
        ..Default::default()
    };
    let mut strict_violation = false;

    // 1. Config
    let config = match load_config() {
        Ok(config) => config,
        Err(e) => {
            report.config_error = Some(e.to_string());
            say(format!("Config ... FAILED: {e}"));
            report.strict_violation = strict;
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            }
            if strict {
                anyhow::bail!("doctor strict mode failed");
            }
            return Ok(());
        }
    };
    report.config_ok = true;
    report.colibri_home = Some(config.colibri_home.display().to_string());
    say(format!("Config ... OK ({})", report.config_path));
    say(format!("  Data dir: {}", config.colibri_home.display()));
    say(format!("  Index: {}", config.index_dir.display()));

    // 2. Connectors
    let details: Vec<DoctorConnectorStatus> = config
        .connector_jobs
        .iter()
        .map(diagnose_connector)
        .collect();
    report.connectors_configured = Some(details.len());
    let warn = details.iter().any(|d| d.status == "warn");
    say(format!(
        "\nConnectors ... {} ({} configured)",
        if warn { "WARN" } else { "OK" },
        details.len()
    ));
    for d in &details {
        let label = if d.enabled { "enabled" } else { "disabled" };
        say(format!(
            "  - {} ({}) [{label}] {}",
            d.id,
            d.connector_type,
            d.status.to_uppercase()
        ));
        for issue in &d.issues {
            say(format!("    - {issue}"));
        }
    }
    report.connector_details = details;

    // 2b. Mirrors
    let mirrors: Vec<DoctorConnectorStatus> = config
        .mirrors
        .iter()
        .map(|m| {
            let mut issues = super::missing_tools(&m.include);
            if !m.path.is_dir() {
                issues.insert(0, format!("folder {} does not exist", m.path.display()));
            }
            DoctorConnectorStatus {
                id: m.name.clone(),
                connector_type: "mirror".into(),
                enabled: true,
                status: if issues.is_empty() { "ok" } else { "warn" }.into(),
                issues,
            }
        })
        .collect();
    let warn = mirrors.iter().any(|d| d.status == "warn");
    say(format!(
        "\nMirrors ... {} ({} configured)",
        if warn { "WARN" } else { "OK" },
        mirrors.len()
    ));
    for m in &mirrors {
        say(format!("  - {} {}", m.id, m.status.to_uppercase()));
        for issue in &m.issues {
            say(format!("    - {issue}"));
        }
    }
    report.mirror_details = mirrors;

    // 3. Metadata DB (read-only)
    match config.open_read().and_then(|store| store.document_count()) {
        Ok(n) => {
            report.metadata_documents = Some(n);
            say(format!("\nMetadata DB ... OK ({n} documents)"));
        }
        Err(e) => {
            report.metadata_error = Some(e.to_string());
            say(format!("\nMetadata DB ... {e}"));
            if strict {
                strict_violation = true;
            }
        }
    }

    // 4. Ollama
    match check_ollama(&config.embedding_endpoint).await {
        Ok(reachable) => {
            report.ollama_reachable = Some(reachable);
            say(format!(
                "\nOllama ... {} ({}, model {})",
                if reachable { "OK" } else { "UNREACHABLE" },
                config.embedding_endpoint,
                config.embedding_model
            ));
        }
        Err(e) => {
            report.ollama_error = Some(e.to_string());
            say(format!("\nOllama ... ERROR: {e}"));
        }
    }

    // 5. Index readiness
    match crate::serve_ready::check(&config) {
        Ok(ready) => {
            report.index_queryable = Some(ready.queryable);
            report.index_chunk_count = ready.chunk_count;
            report.index_model = ready.embedding_model.clone();
            report.index_issues = ready.issues.clone();
            if ready.queryable {
                say(format!(
                    "\nIndex ... OK ({} chunks, model {})",
                    ready.chunk_count.unwrap_or(0),
                    ready.embedding_model.unwrap_or_default()
                ));
            } else {
                say("\nIndex ... NOT READY".into());
                for issue in &ready.issues {
                    say(format!("  - {issue}"));
                }
                if strict {
                    strict_violation = true;
                }
            }
        }
        Err(e) => {
            report.index_issues = vec![e.to_string()];
            say(format!("\nIndex ... ERROR: {e}"));
            if strict {
                strict_violation = true;
            }
        }
    }

    report.strict_violation = strict_violation;
    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        eprintln!();
    }
    if strict && strict_violation {
        anyhow::bail!("doctor strict mode failed");
    }
    Ok(())
}
