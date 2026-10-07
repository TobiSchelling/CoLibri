//! `colibri connectors` — native connector management and sync.

use serde::Serialize;

use crate::canonical_store::{ingest_envelopes, CanonicalIngestReport};
use crate::cli::config_string;
use crate::config::{load_config, AppConfig};
use crate::connectors::zephyr_scale::ZephyrScaleConnector;
use crate::connectors::{Connector, ConnectorJob};
use crate::embedding::OllamaEmbedder;
use crate::indexer::{index_library, IndexOptions};
use crate::metadata_store::MetadataStore;

/// Build a concrete `Connector` from a resolved `ConnectorJob`.
fn build_connector(job: &ConnectorJob) -> anyhow::Result<Box<dyn Connector>> {
    match job.connector_type.as_str() {
        "filesystem" => anyhow::bail!(
            "connector '{}': filesystem connectors were replaced by `mirrors:` in config.yaml \
             (name, path, include, exclude, doc_type); then run `colibri update`",
            job.id
        ),
        "zephyr_scale" => {
            let project_key = config_string(&job.config, "project_key")
                .ok_or_else(|| anyhow::anyhow!("connector '{}': missing project_key", job.id))?;

            let api_base_url = config_string(&job.config, "api_base_url")
                .unwrap_or_else(|| "https://api.zephyrscale.smartbear.com/v2".into());

            // Resolve token: direct config value or env var
            let token_env = config_string(&job.config, "token_env")
                .unwrap_or_else(|| "ZEPHYR_API_TOKEN".into());
            let token = config_string(&job.config, "token")
                .or_else(|| std::env::var(&token_env).ok().filter(|s| !s.is_empty()))
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "connector '{}': no API token — set `token` in config or env var {}",
                        job.id,
                        token_env
                    )
                })?;

            let folder_path = config_string(&job.config, "folder_path");

            let doc_type =
                config_string(&job.config, "doc_type").unwrap_or_else(|| "test_case".into());

            let include_steps = job
                .config
                .get("include_steps")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);

            let include_links = job
                .config
                .get("include_links")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);

            Ok(Box::new(ZephyrScaleConnector {
                id: job.id.clone(),
                project_key,
                api_base_url,
                token,
                folder_path,
                doc_type,
                include_steps,
                include_links,
            }))
        }
        other => anyhow::bail!("connector '{}': unknown connector type '{}'", job.id, other),
    }
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
    let config = load_config()?;

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
pub async fn sync_all(mut opts: SyncAllOptions) -> anyhow::Result<()> {
    // Dry-run implies no indexing — silently disable rather than erroring.
    if opts.dry_run {
        opts.index = false;
    }

    let app_config = load_config()?;
    // Dry runs read existing state (if any) without the lock; real runs hold
    // the write lock for ingest and indexing.
    let (_lock, store) = if opts.dry_run {
        // A missing DB means a first run; an unusable one must be reported.
        let store = if app_config.metadata_db_path.exists() {
            Some(app_config.open_read()?)
        } else {
            None
        };
        (None, store)
    } else {
        let (lock, store) = app_config.open_for_write()?;
        (Some(lock), Some(store))
    };
    let _awake = (!opts.dry_run).then(crate::power::keep_awake);

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

        match run_connector(&app_config, store.as_ref(), job, opts.dry_run).await {
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
        } else if let Some(store) = store.as_ref() {
            let embedder = OllamaEmbedder::from_config(&app_config);
            let index_opts = IndexOptions {
                force: opts.index_force,
                ..Default::default()
            };
            let progress = crate::cli::index::CliProgress::new();
            let index_run = index_library(&app_config, store, &embedder, &index_opts, |e| {
                if !opts.json {
                    progress.handle(e);
                }
            })
            .await;

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
    store: Option<&MetadataStore>,
    job: &ConnectorJob,
    dry_run: bool,
) -> anyhow::Result<SyncReport> {
    let connector = build_connector(job)?;
    let connector_id = connector.id().to_string();
    let envelopes = connector.sync().await?;
    let envelope_count = envelopes.len();
    // The connector id is the collection name.
    let ingest_report = ingest_envelopes(config, store, &job.id, &envelopes, dry_run)?;

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
    fn filesystem_connector_points_to_mirrors() {
        let job = ConnectorJob {
            id: "test".into(),
            connector_type: "filesystem".into(),
            enabled: true,
            config: serde_json::json!({}),
        };
        let err = build_connector(&job).err().expect("should fail");
        assert!(err.to_string().contains("mirrors:"));
    }

    #[test]
    fn select_connectors_empty_returns_all() {
        let jobs = vec![
            ConnectorJob {
                id: "a".into(),
                connector_type: "zephyr_scale".into(),
                enabled: true,
                config: serde_json::json!({}),
            },
            ConnectorJob {
                id: "b".into(),
                connector_type: "zephyr_scale".into(),
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
                connector_type: "zephyr_scale".into(),
                enabled: true,
                config: serde_json::json!({}),
            },
            ConnectorJob {
                id: "b".into(),
                connector_type: "zephyr_scale".into(),
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
            connector_type: "zephyr_scale".into(),
            enabled: true,
            config: serde_json::json!({}),
        }];
        let result = select_connectors(&jobs, &["nonexistent".into()]);
        assert!(result.is_err());
    }

    #[test]
    fn build_zephyr_connector_missing_project_key_errors() {
        let job = ConnectorJob {
            id: "zephyr-test".into(),
            connector_type: "zephyr_scale".into(),
            enabled: true,
            config: serde_json::json!({"token": "test-token"}),
        };
        let err = build_connector(&job).err().expect("should fail");
        assert!(err.to_string().contains("project_key"));
    }

    #[test]
    fn build_zephyr_connector_missing_token_errors() {
        // Ensure env var is not set for this test
        std::env::remove_var("ZEPHYR_API_TOKEN");
        let job = ConnectorJob {
            id: "zephyr-test".into(),
            connector_type: "zephyr_scale".into(),
            enabled: true,
            config: serde_json::json!({"project_key": "PROJ"}),
        };
        let err = build_connector(&job).err().expect("should fail");
        assert!(err.to_string().contains("token"));
    }

    #[test]
    fn build_zephyr_connector_with_config_token_succeeds() {
        let job = ConnectorJob {
            id: "zephyr-test".into(),
            connector_type: "zephyr_scale".into(),
            enabled: true,
            config: serde_json::json!({
                "project_key": "PROJ",
                "token": "my-secret-token"
            }),
        };
        let connector = build_connector(&job).expect("should succeed");
        assert_eq!(connector.id(), "zephyr-test");
    }

    #[test]
    fn build_zephyr_connector_with_env_token_succeeds() {
        std::env::set_var("TEST_ZEPHYR_TOKEN", "env-token");
        let job = ConnectorJob {
            id: "zephyr-test".into(),
            connector_type: "zephyr_scale".into(),
            enabled: true,
            config: serde_json::json!({
                "project_key": "PROJ",
                "token_env": "TEST_ZEPHYR_TOKEN"
            }),
        };
        let connector = build_connector(&job).expect("should succeed");
        assert_eq!(connector.id(), "zephyr-test");
        std::env::remove_var("TEST_ZEPHYR_TOKEN");
    }
}
