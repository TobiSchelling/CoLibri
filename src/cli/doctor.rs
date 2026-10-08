//! `colibri doctor` — health check command.

use serde::Serialize;

use crate::config::{self, load_config};
use crate::embedding::check_ollama;

#[derive(Debug, Serialize)]
struct DoctorMirrorStatus {
    name: String,
    kind: String,
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
    mirror_details: Vec<DoctorMirrorStatus>,
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

    // 2b. Mirrors
    let mirrors: Vec<DoctorMirrorStatus> = config
        .mirrors
        .iter()
        .map(|m| {
            let mut issues = super::missing_tools(&m.include);
            match &m.fetch {
                Some(crate::config::FetchConfig::ZephyrScale(z)) => {
                    if std::env::var(&z.token_env).map_or(true, |t| t.trim().is_empty()) {
                        issues.push(format!("no Zephyr API token in ${}", z.token_env));
                    }
                }
                Some(crate::config::FetchConfig::Command { run }) => {
                    if run.first().is_none_or(|p| !super::tool_on_path(p)) {
                        issues.push(format!("fetch command not found: {:?}", run.first()));
                    }
                }
                None if !m.path.is_dir() => {
                    issues.insert(0, format!("folder {} does not exist", m.path.display()));
                }
                None => {}
            }
            DoctorMirrorStatus {
                name: m.name.clone(),
                kind: match &m.fetch {
                    Some(crate::config::FetchConfig::ZephyrScale(_)) => "zephyr fetch",
                    Some(crate::config::FetchConfig::Command { .. }) => "command fetch",
                    None => "folder",
                }
                .into(),
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
        say(format!("  - {} {}", m.name, m.status.to_uppercase()));
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
