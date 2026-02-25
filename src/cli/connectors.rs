//! `colibri connectors` — native connector management and sync.

use serde::Serialize;
use serde_json::Value;

use crate::canonical_store::{ingest_envelopes, CanonicalIngestReport};
use crate::cli::config_string;
use crate::config::{load_config, load_config_no_bootstrap, AppConfig};
use crate::connectors::filesystem::FilesystemConnector;
use crate::connectors::{Connector, ConnectorJob};
use crate::indexer::index_library;

/// Build a concrete `Connector` from a resolved `ConnectorJob`.
fn build_connector(job: &ConnectorJob) -> anyhow::Result<Box<dyn Connector>> {
    match job.connector_type.as_str() {
        "filesystem" => {
            let root_path = config_string(&job.config, "root_path")
                .ok_or_else(|| anyhow::anyhow!("connector '{}': missing root_path", job.id))?;
            let root_path = expand_tilde(&root_path);

            let include_extensions = parse_string_array(&job.config, "include_extensions")
                .unwrap_or_else(|| vec![".md".into(), ".markdown".into()]);

            let exclude_globs =
                parse_string_array(&job.config, "exclude_globs").unwrap_or_default();

            let doc_type = config_string(&job.config, "doc_type").unwrap_or_else(|| "note".into());

            let classification =
                config_string(&job.config, "classification").unwrap_or_else(|| "internal".into());

            let plantuml_summaries = job
                .config
                .get("plantuml_summaries")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);

            Ok(Box::new(FilesystemConnector {
                id: job.id.clone(),
                root_path: root_path.into(),
                include_extensions,
                exclude_globs,
                doc_type,
                classification,
                plantuml_summaries,
            }))
        }
        other => anyhow::bail!("connector '{}': unknown connector type '{}'", job.id, other),
    }
}

/// Expand leading `~/` to the user's home directory.
fn expand_tilde(path: &str) -> String {
    if let Some(rest) = path.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            return format!("{}/{rest}", home.display());
        }
    } else if path == "~" {
        if let Some(home) = dirs::home_dir() {
            return home.display().to_string();
        }
    }
    path.to_string()
}

/// Parse an optional JSON array of strings from a config value.
fn parse_string_array(config: &Value, key: &str) -> Option<Vec<String>> {
    config.get(key).and_then(|v| v.as_array()).map(|arr| {
        arr.iter()
            .filter_map(|item| item.as_str().map(|s| s.to_string()))
            .collect()
    })
}

// ---------------------------------------------------------------------------
// Sync report types
// ---------------------------------------------------------------------------

/// Result of syncing a single connector.
#[derive(Debug, Serialize)]
pub struct SyncReport {
    pub connector_id: String,
    pub connector_type: String,
    pub status: String,
    pub envelope_count: Option<usize>,
    pub ingest: Option<CanonicalIngestReport>,
    pub error: Option<String>,
}

/// Aggregate report for `colibri sync`.
#[derive(Debug, Serialize)]
struct SyncAllReport {
    dry_run: bool,
    connectors_requested: usize,
    connectors_selected: usize,
    connectors_run: usize,
    connectors_succeeded: usize,
    connectors_failed: usize,
    connectors_skipped: usize,
    index: Option<SyncIndexReport>,
    results: Vec<SyncReport>,
}

#[derive(Debug, Serialize)]
struct SyncIndexReport {
    status: String,
    force: bool,
    files_indexed: Option<usize>,
    files_skipped: Option<usize>,
    files_deleted: Option<usize>,
    total_chunks: Option<usize>,
    errors: Option<usize>,
    error: Option<String>,
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Options passed to `sync_all`.
pub struct SyncAllOptions {
    pub requested_connectors: Vec<String>,
    pub include_disabled: bool,
    pub fail_fast: bool,
    pub index: bool,
    pub index_force: bool,
    pub dry_run: bool,
    pub json: bool,
}

/// List configured connectors.
pub async fn list(json: bool) -> anyhow::Result<()> {
    let config = load_config_no_bootstrap()?;

    if json {
        let view: Vec<_> = config
            .connector_jobs
            .iter()
            .map(|j| {
                serde_json::json!({
                    "id": j.id,
                    "type": j.connector_type,
                    "enabled": j.enabled,
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&view)?);
        return Ok(());
    }

    if config.connector_jobs.is_empty() {
        eprintln!("No connectors configured. Add a `connectors:` section in config.yaml.");
        return Ok(());
    }

    eprintln!("Configured Connectors");
    eprintln!("=====================");
    for job in &config.connector_jobs {
        let status = if job.enabled { "enabled" } else { "disabled" };
        eprintln!("  {} [{}] type={}", job.id, status, job.connector_type);
    }
    Ok(())
}

/// Run sync for all (or selected) connectors.
pub async fn sync_all(opts: SyncAllOptions) -> anyhow::Result<()> {
    if opts.dry_run && opts.index {
        anyhow::bail!("`--no-index false` cannot be used with `--dry-run`");
    }

    let app_config = if opts.dry_run {
        load_config_no_bootstrap()?
    } else {
        load_config()?
    };

    let mut selected = select_connectors(&app_config.connector_jobs, &opts.requested_connectors)?;
    selected.sort_by(|a, b| a.id.cmp(&b.id));

    if selected.is_empty() {
        anyhow::bail!("No connectors configured. Add a `connectors:` section in config.yaml.");
    }

    let selected_total = selected.len();
    let mut results = Vec::new();
    let mut run_count = 0usize;
    let mut succeeded = 0usize;
    let mut failed = 0usize;
    let mut skipped = 0usize;

    for job in &selected {
        if !job.enabled && !opts.include_disabled {
            skipped += 1;
            results.push(SyncReport {
                connector_id: job.id.clone(),
                connector_type: job.connector_type.clone(),
                status: "skipped_disabled".into(),
                envelope_count: None,
                ingest: None,
                error: None,
            });
            continue;
        }

        run_count += 1;

        match run_connector(&app_config, job, opts.dry_run).await {
            Ok(report) => {
                succeeded += 1;
                results.push(report);
            }
            Err(e) => {
                failed += 1;
                results.push(SyncReport {
                    connector_id: job.id.clone(),
                    connector_type: job.connector_type.clone(),
                    status: "error".into(),
                    envelope_count: None,
                    ingest: None,
                    error: Some(e.to_string()),
                });
                if opts.fail_fast {
                    break;
                }
            }
        }
    }

    let mut report = SyncAllReport {
        dry_run: opts.dry_run,
        connectors_requested: opts.requested_connectors.len(),
        connectors_selected: selected_total,
        connectors_run: run_count,
        connectors_succeeded: succeeded,
        connectors_failed: failed,
        connectors_skipped: skipped,
        index: None,
        results,
    };

    let mut index_failed = false;
    if opts.index {
        if report.connectors_failed > 0 {
            report.index = Some(SyncIndexReport {
                status: "skipped_due_to_failures".into(),
                force: opts.index_force,
                files_indexed: None,
                files_skipped: None,
                files_deleted: None,
                total_chunks: None,
                errors: None,
                error: None,
            });
        } else {
            let index_run = if opts.json {
                index_library(&app_config, opts.index_force, |_e| {}).await
            } else {
                let progress = crate::cli::index::CliProgress::new();
                index_library(&app_config, opts.index_force, |e| {
                    progress.handle(e);
                })
                .await
            };

            match index_run {
                Ok(index_result) => {
                    let has_errors = index_result.errors > 0;
                    if has_errors {
                        index_failed = true;
                    }
                    report.index = Some(SyncIndexReport {
                        status: if has_errors { "error" } else { "ok" }.into(),
                        force: opts.index_force,
                        files_indexed: Some(index_result.files_indexed),
                        files_skipped: Some(index_result.files_skipped),
                        files_deleted: Some(index_result.files_deleted),
                        total_chunks: Some(index_result.total_chunks),
                        errors: Some(index_result.errors),
                        error: None,
                    });
                }
                Err(e) => {
                    index_failed = true;
                    report.index = Some(SyncIndexReport {
                        status: "error".into(),
                        force: opts.index_force,
                        files_indexed: None,
                        files_skipped: None,
                        files_deleted: None,
                        total_chunks: None,
                        errors: None,
                        error: Some(e.to_string()),
                    });
                }
            }
        }
    }

    // Output
    if opts.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        eprintln!("Connector Sync");
        eprintln!("==============");
        eprintln!("Dry run: {}", report.dry_run);
        eprintln!(
            "Connectors: selected={}, run={}, succeeded={}, failed={}, skipped={}",
            report.connectors_selected,
            report.connectors_run,
            report.connectors_succeeded,
            report.connectors_failed,
            report.connectors_skipped
        );
        if let Some(index) = &report.index {
            eprintln!(
                "Index: status={} force={} indexed={} skipped={} removed={} chunks={} errors={}",
                index.status,
                index.force,
                index.files_indexed.unwrap_or(0),
                index.files_skipped.unwrap_or(0),
                index.files_deleted.unwrap_or(0),
                index.total_chunks.unwrap_or(0),
                index.errors.unwrap_or(0)
            );
            if let Some(err) = &index.error {
                eprintln!("Index error: {err}");
            }
        }
        for row in &report.results {
            match row.status.as_str() {
                "ok" => {
                    let ingest = row.ingest.as_ref();
                    eprintln!(
                        "  {} [{}] envelopes={} written={} unchanged={} tombstoned={}",
                        row.connector_id,
                        row.status,
                        row.envelope_count.unwrap_or(0),
                        ingest.map(|i| i.written).unwrap_or(0),
                        ingest.map(|i| i.unchanged).unwrap_or(0),
                        ingest.map(|i| i.tombstoned).unwrap_or(0),
                    );
                }
                "error" => eprintln!(
                    "  {} [{}] {}",
                    row.connector_id,
                    row.status,
                    row.error.as_deref().unwrap_or("unknown error")
                ),
                _ => eprintln!("  {} [{}]", row.connector_id, row.status),
            }
        }
    }

    if report.connectors_failed > 0 || index_failed {
        std::process::exit(1);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Select connector jobs by requested ids, or all if none specified.
fn select_connectors<'a>(
    jobs: &'a [ConnectorJob],
    requested: &[String],
) -> anyhow::Result<Vec<&'a ConnectorJob>> {
    if requested.is_empty() {
        return Ok(jobs.iter().collect());
    }
    let mut selected = Vec::new();
    for req in requested {
        let job = jobs
            .iter()
            .find(|j| j.id == *req)
            .ok_or_else(|| anyhow::anyhow!("Connector '{}' not found in config", req))?;
        selected.push(job);
    }
    Ok(selected)
}

/// Build, sync, and ingest a single connector.
async fn run_connector(
    config: &AppConfig,
    job: &ConnectorJob,
    dry_run: bool,
) -> anyhow::Result<SyncReport> {
    let connector = build_connector(job)?;
    let connector_id = connector.id().to_string();
    let envelopes = connector.sync().await?;
    let envelope_count = envelopes.len();
    let ingest_report = ingest_envelopes(config, &envelopes, dry_run)?;

    Ok(SyncReport {
        connector_id,
        connector_type: job.connector_type.clone(),
        status: "ok".into(),
        envelope_count: Some(envelope_count),
        ingest: Some(ingest_report),
        error: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expand_tilde_home_prefix() {
        let expanded = expand_tilde("~/Documents");
        assert!(!expanded.starts_with('~'));
        assert!(expanded.ends_with("/Documents"));
    }

    #[test]
    fn expand_tilde_bare() {
        let expanded = expand_tilde("~");
        assert!(!expanded.starts_with('~'));
        assert!(!expanded.is_empty());
    }

    #[test]
    fn expand_tilde_no_tilde() {
        assert_eq!(expand_tilde("/absolute/path"), "/absolute/path");
        assert_eq!(expand_tilde("relative/path"), "relative/path");
    }

    #[test]
    fn expand_tilde_does_not_expand_other_users() {
        let result = expand_tilde("~otheruser/foo");
        assert_eq!(result, "~otheruser/foo");
    }

    #[test]
    fn build_connector_unknown_type_errors() {
        let job = ConnectorJob {
            id: "test".into(),
            connector_type: "unknown_type".into(),
            enabled: true,
            config: serde_json::json!({}),
        };
        let err = build_connector(&job).err().expect("should fail");
        assert!(err.to_string().contains("unknown connector type"));
    }

    #[test]
    fn build_connector_missing_root_path_errors() {
        let job = ConnectorJob {
            id: "test".into(),
            connector_type: "filesystem".into(),
            enabled: true,
            config: serde_json::json!({}),
        };
        let err = build_connector(&job).err().expect("should fail");
        assert!(err.to_string().contains("root_path"));
    }

    #[test]
    fn select_connectors_empty_returns_all() {
        let jobs = vec![
            ConnectorJob {
                id: "a".into(),
                connector_type: "filesystem".into(),
                enabled: true,
                config: serde_json::json!({}),
            },
            ConnectorJob {
                id: "b".into(),
                connector_type: "filesystem".into(),
                enabled: false,
                config: serde_json::json!({}),
            },
        ];
        let selected = select_connectors(&jobs, &[]).unwrap();
        assert_eq!(selected.len(), 2);
    }

    #[test]
    fn select_connectors_filters_by_id() {
        let jobs = vec![
            ConnectorJob {
                id: "a".into(),
                connector_type: "filesystem".into(),
                enabled: true,
                config: serde_json::json!({}),
            },
            ConnectorJob {
                id: "b".into(),
                connector_type: "filesystem".into(),
                enabled: true,
                config: serde_json::json!({}),
            },
        ];
        let selected = select_connectors(&jobs, &["b".into()]).unwrap();
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].id, "b");
    }

    #[test]
    fn select_connectors_unknown_id_errors() {
        let jobs = vec![ConnectorJob {
            id: "a".into(),
            connector_type: "filesystem".into(),
            enabled: true,
            config: serde_json::json!({}),
        }];
        let result = select_connectors(&jobs, &["nonexistent".into()]);
        assert!(result.is_err());
    }

    #[test]
    fn parse_string_array_extracts_strings() {
        let config = serde_json::json!({
            "exts": [".md", ".pdf"]
        });
        let result = parse_string_array(&config, "exts");
        assert_eq!(result, Some(vec![".md".to_string(), ".pdf".to_string()]));
    }

    #[test]
    fn parse_string_array_missing_key_returns_none() {
        let config = serde_json::json!({});
        assert!(parse_string_array(&config, "missing").is_none());
    }
}
