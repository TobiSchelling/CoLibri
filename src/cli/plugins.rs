//! `colibri plugins` — run and validate ingestion plugins.

use std::path::PathBuf;

use anyhow::Context;
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::canonical_store::{ingest_envelopes, CanonicalIngestReport};
use crate::config::{load_config, load_config_no_bootstrap, AppConfig, PluginJob};
use crate::indexer::index_library;
use crate::metadata_store::{MetadataStore, SyncStateEntry};
use crate::plugin_host::{run_plugin_manifest, PluginRunReport};

fn parse_plugin_config(
    config_json: Option<String>,
    config_file: Option<PathBuf>,
) -> anyhow::Result<Value> {
    if config_json.is_some() && config_file.is_some() {
        anyhow::bail!("Use either --config-json or --config-file, not both.");
    }

    let config: Value = if let Some(raw) = config_json {
        serde_json::from_str(&raw).context("Failed to parse --config-json as JSON object")?
    } else if let Some(path) = config_file {
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("Failed to read config file {}", path.display()))?;
        serde_json::from_str(&text)
            .with_context(|| format!("Failed to parse config file {}", path.display()))?
    } else {
        Value::Object(serde_json::Map::new())
    };

    if !config.is_object() {
        anyhow::bail!("Plugin config must be a JSON object.");
    }

    Ok(config)
}

pub async fn run(
    manifest: PathBuf,
    config_json: Option<String>,
    config_file: Option<PathBuf>,
    include_envelopes: bool,
    json: bool,
) -> anyhow::Result<()> {
    let config = parse_plugin_config(config_json, config_file)?;

    let mut report = run_plugin_manifest(&manifest, &config, None).await?;
    if !include_envelopes {
        report.envelopes.clear();
    }

    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }

    eprintln!("Plugin Run");
    eprintln!("==========");
    eprintln!("Plugin: {}", report.plugin_id);
    eprintln!("Runtime: {}", report.runtime);
    eprintln!("Manifest: {}", report.manifest_path);
    eprintln!("Envelopes: {}", report.envelope_count);
    eprintln!("Deleted envelopes: {}", report.deleted_count);
    eprintln!(
        "Next cursor: {}",
        report
            .next_cursor
            .as_ref()
            .map(|v| v.to_string())
            .unwrap_or_else(|| "none".into())
    );

    if !report.stderr.trim().is_empty() {
        eprintln!("\nPlugin stderr:");
        eprintln!("{}", report.stderr.trim());
    }

    if include_envelopes {
        eprintln!("\nEnvelopes (JSONL on stdout):");
        for env in &report.envelopes {
            println!("{}", serde_json::to_string(env)?);
        }
    } else {
        eprintln!("\nUse --include-envelopes to print envelope payloads.");
    }

    Ok(())
}

pub async fn ingest(
    manifest: PathBuf,
    config_json: Option<String>,
    config_file: Option<PathBuf>,
    dry_run: bool,
    json: bool,
) -> anyhow::Result<()> {
    let config = parse_plugin_config(config_json, config_file)?;
    let mut plugin_report = run_plugin_manifest(&manifest, &config, None).await?;
    let app_config = if dry_run {
        load_config_no_bootstrap()?
    } else {
        load_config()?
    };
    let ingest_report = ingest_envelopes(&app_config, &plugin_report.envelopes, dry_run)?;

    if json {
        let payload = serde_json::json!({
            "plugin": plugin_report,
            "ingest": ingest_report
        });
        println!("{}", serde_json::to_string_pretty(&payload)?);
        return Ok(());
    }

    eprintln!("Plugin Ingest");
    eprintln!("=============");
    eprintln!("Plugin: {}", plugin_report.plugin_id);
    eprintln!("Runtime: {}", plugin_report.runtime);
    eprintln!("Manifest: {}", plugin_report.manifest_path);
    eprintln!("Envelopes processed: {}", ingest_report.processed);
    eprintln!("Written markdown: {}", ingest_report.written);
    eprintln!("Unchanged: {}", ingest_report.unchanged);
    eprintln!("Tombstones: {}", ingest_report.tombstoned);
    eprintln!("Deleted markdown files: {}", ingest_report.deleted_files);
    eprintln!(
        "Duplicate doc_ids in run: {}",
        ingest_report.duplicate_doc_ids
    );
    eprintln!(
        "Next cursor: {}",
        plugin_report
            .next_cursor
            .as_ref()
            .map(|v| v.to_string())
            .unwrap_or_else(|| "none".into())
    );
    eprintln!("Canonical dir: {}", ingest_report.canonical_dir);
    eprintln!("Metadata DB: {}", ingest_report.metadata_db_path);
    eprintln!("Dry run: {}", ingest_report.dry_run);

    if !plugin_report.stderr.trim().is_empty() {
        eprintln!("\nPlugin stderr:");
        eprintln!("{}", plugin_report.stderr.trim());
    }

    if !dry_run {
        plugin_report.envelopes.clear();
        eprintln!("\nNext step: run `colibri index --force` to index updated canonical corpus.");
    } else {
        eprintln!("\nDry-run mode enabled. No files were written.");
    }

    Ok(())
}

fn stable_json(value: &Value) -> anyhow::Result<String> {
    Ok(serde_json::to_string(value)?)
}

fn sync_key(manifest_path: &std::path::Path, config: &Value) -> anyhow::Result<(String, String)> {
    let manifest_abs = manifest_path
        .canonicalize()
        .unwrap_or_else(|_| manifest_path.to_path_buf());
    let cfg = stable_json(config)?;
    let mut hasher = Sha256::new();
    hasher.update(manifest_abs.to_string_lossy().as_bytes());
    hasher.update(b"\n");
    hasher.update(cfg.as_bytes());
    let key = format!("{:x}", hasher.finalize());
    Ok((key, cfg))
}

fn load_sync_cursor(metadata_path: &std::path::Path, key: &str) -> anyhow::Result<Option<Value>> {
    if !metadata_path.exists() {
        return Ok(None);
    }
    let store = MetadataStore::open(metadata_path)?;
    Ok(store.get_sync_cursor(key)?)
}

struct SyncSuccessUpdate<'a> {
    key: &'a str,
    manifest_path: &'a std::path::Path,
    config_hash: &'a str,
    plugin_id: &'a str,
    previous_cursor: Option<&'a Value>,
    next_cursor: Option<&'a Value>,
    envelope_count: usize,
}

fn upsert_sync_state_success(
    metadata_path: &std::path::Path,
    update: SyncSuccessUpdate<'_>,
) -> anyhow::Result<()> {
    let store = MetadataStore::open(metadata_path)?;
    let cursor = update
        .next_cursor
        .cloned()
        .or_else(|| update.previous_cursor.cloned());
    store.upsert_sync_success(
        update.key,
        &update.manifest_path.display().to_string(),
        update.config_hash,
        update.plugin_id,
        cursor,
        update.envelope_count,
    )?;
    store.touch_updated_at()?;
    Ok(())
}

fn upsert_sync_state_error(
    metadata_path: &std::path::Path,
    key: &str,
    manifest_path: &std::path::Path,
    config_hash: &str,
    error: &str,
) -> anyhow::Result<()> {
    let store = MetadataStore::open(metadata_path)?;
    store.upsert_sync_error(
        key,
        &manifest_path.display().to_string(),
        config_hash,
        error,
    )?;
    store.touch_updated_at()?;
    Ok(())
}

#[derive(Debug, Serialize)]
struct SyncRunResult {
    sync_key: String,
    previous_cursor: Option<Value>,
    next_cursor: Option<Value>,
    plugin: PluginRunReport,
    ingest: CanonicalIngestReport,
}

async fn sync_once(
    app_config: &AppConfig,
    manifest: PathBuf,
    config: Value,
    dry_run: bool,
) -> anyhow::Result<SyncRunResult> {
    let (key, config_hash) = sync_key(&manifest, &config)?;
    let previous_cursor = load_sync_cursor(&app_config.metadata_db_path, &key)?;

    let plugin_report = match run_plugin_manifest(&manifest, &config, previous_cursor.clone()).await
    {
        Ok(report) => report,
        Err(e) => {
            if !dry_run {
                let _ = upsert_sync_state_error(
                    &app_config.metadata_db_path,
                    &key,
                    &manifest,
                    &config_hash,
                    &e.to_string(),
                );
            }
            return Err(anyhow::Error::new(e));
        }
    };

    let ingest_report = match ingest_envelopes(app_config, &plugin_report.envelopes, dry_run) {
        Ok(report) => report,
        Err(e) => {
            if !dry_run {
                let _ = upsert_sync_state_error(
                    &app_config.metadata_db_path,
                    &key,
                    &manifest,
                    &config_hash,
                    &e.to_string(),
                );
            }
            return Err(anyhow::Error::new(e));
        }
    };

    if !dry_run {
        upsert_sync_state_success(
            &app_config.metadata_db_path,
            SyncSuccessUpdate {
                key: &key,
                manifest_path: &manifest,
                config_hash: &config_hash,
                plugin_id: &plugin_report.plugin_id,
                previous_cursor: previous_cursor.as_ref(),
                next_cursor: plugin_report.next_cursor.as_ref(),
                envelope_count: plugin_report.envelope_count,
            },
        )?;
    }

    Ok(SyncRunResult {
        sync_key: key,
        previous_cursor,
        next_cursor: plugin_report.next_cursor.clone(),
        plugin: plugin_report,
        ingest: ingest_report,
    })
}

#[derive(Debug, Serialize)]
struct SyncStateEntrySummary {
    key: String,
    plugin_id: Option<String>,
    manifest_path: Option<String>,
    has_cursor: bool,
    last_success_at: Option<String>,
    last_error_at: Option<String>,
    envelope_count_last_run: Option<u64>,
    updated_at: Option<String>,
}

fn summarize_sync_state_entry(row: &SyncStateEntry) -> SyncStateEntrySummary {
    SyncStateEntrySummary {
        key: row.sync_key.clone(),
        plugin_id: row.plugin_id.clone(),
        manifest_path: Some(row.manifest_path.clone()),
        has_cursor: row.cursor.is_some(),
        last_success_at: row.last_success_at.clone(),
        last_error_at: row.last_error_at.clone(),
        envelope_count_last_run: row.envelope_count_last_run,
        updated_at: Some(row.updated_at.clone()),
    }
}

pub async fn state_list(json: bool) -> anyhow::Result<()> {
    let app_config = load_config_no_bootstrap()?;
    let entries_raw = if app_config.metadata_db_path.exists() {
        let store = MetadataStore::open(&app_config.metadata_db_path)?;
        store.list_sync_entries()?
    } else {
        Vec::new()
    };
    let entries: Vec<_> = entries_raw.iter().map(summarize_sync_state_entry).collect();

    if json {
        let payload = serde_json::json!({
            "metadata_db_path": app_config.metadata_db_path.display().to_string(),
            "count": entries.len(),
            "entries": entries
        });
        println!("{}", serde_json::to_string_pretty(&payload)?);
        return Ok(());
    }

    eprintln!("Plugin Sync State");
    eprintln!("=================");
    eprintln!("Metadata DB: {}", app_config.metadata_db_path.display());
    eprintln!("Entries: {}", entries.len());
    for entry in entries {
        eprintln!(
            "- {} | plugin={} | cursor={} | success={} | error={}",
            entry.key,
            entry.plugin_id.unwrap_or_else(|| "unknown".into()),
            entry.has_cursor,
            entry.last_success_at.unwrap_or_else(|| "never".into()),
            entry.last_error_at.unwrap_or_else(|| "none".into())
        );
    }
    Ok(())
}

pub async fn state_show(
    manifest: PathBuf,
    config_json: Option<String>,
    config_file: Option<PathBuf>,
    json: bool,
) -> anyhow::Result<()> {
    let config = parse_plugin_config(config_json, config_file)?;
    let app_config = load_config_no_bootstrap()?;
    let (key, _) = sync_key(&manifest, &config)?;
    let row = if app_config.metadata_db_path.exists() {
        let store = MetadataStore::open(&app_config.metadata_db_path)?;
        store.get_sync_entry(&key)?
    } else {
        None
    };

    if json {
        let entry_json = row.as_ref().map(sync_entry_to_json);
        let payload = serde_json::json!({
            "metadata_db_path": app_config.metadata_db_path.display().to_string(),
            "sync_key": key,
            "entry": entry_json
        });
        println!("{}", serde_json::to_string_pretty(&payload)?);
        return Ok(());
    }

    eprintln!("Plugin Sync Entry");
    eprintln!("=================");
    eprintln!("Metadata DB: {}", app_config.metadata_db_path.display());
    eprintln!("Sync key: {}", key);
    match row {
        Some(v) => eprintln!("{}", serde_json::to_string_pretty(&sync_entry_to_json(&v))?),
        None => eprintln!("Entry not found."),
    }
    Ok(())
}

pub async fn state_reset(
    manifest: PathBuf,
    config_json: Option<String>,
    config_file: Option<PathBuf>,
    yes: bool,
    json: bool,
) -> anyhow::Result<()> {
    if !yes {
        anyhow::bail!("Refusing reset without --yes");
    }
    let config = parse_plugin_config(config_json, config_file)?;
    let app_config = load_config_no_bootstrap()?;
    let (key, _) = sync_key(&manifest, &config)?;

    if !app_config.metadata_db_path.exists() {
        if json {
            let payload = serde_json::json!({
                "metadata_db_path": app_config.metadata_db_path.display().to_string(),
                "sync_key": key,
                "removed": false
            });
            println!("{}", serde_json::to_string_pretty(&payload)?);
            return Ok(());
        }
        eprintln!("Metadata DB not found. Nothing to reset.");
        return Ok(());
    }

    let store = MetadataStore::open(&app_config.metadata_db_path)?;
    let removed = store.delete_sync_entry(&key)?;
    store.touch_updated_at()?;

    if json {
        let payload = serde_json::json!({
            "metadata_db_path": app_config.metadata_db_path.display().to_string(),
            "sync_key": key,
            "removed": removed
        });
        println!("{}", serde_json::to_string_pretty(&payload)?);
        return Ok(());
    }

    eprintln!("Plugin Sync Reset");
    eprintln!("=================");
    eprintln!("Metadata DB: {}", app_config.metadata_db_path.display());
    eprintln!("Sync key: {}", key);
    eprintln!("Removed: {}", removed);
    Ok(())
}

fn sync_entry_to_json(entry: &SyncStateEntry) -> Value {
    serde_json::json!({
        "manifest_path": entry.manifest_path,
        "config_hash": entry.config_hash,
        "plugin_id": entry.plugin_id,
        "cursor": entry.cursor,
        "last_success_at": entry.last_success_at,
        "last_error_at": entry.last_error_at,
        "last_error": entry.last_error,
        "envelope_count_last_run": entry.envelope_count_last_run,
        "updated_at": entry.updated_at
    })
}

#[cfg(test)]
mod tests {
    use super::select_jobs;
    use crate::config::PluginJob;
    use serde_json::{Map, Value};
    use std::path::PathBuf;

    fn job(id: &str) -> PluginJob {
        PluginJob {
            id: id.to_string(),
            manifest: PathBuf::from(format!("/tmp/{id}.json")),
            enabled: true,
            config: Value::Object(Map::new()),
        }
    }

    #[test]
    fn select_jobs_returns_all_when_request_empty() {
        let jobs = vec![job("a"), job("b")];
        let selected = select_jobs(&jobs, &[]).expect("selection should succeed");
        assert_eq!(selected.len(), 2);
    }

    #[test]
    fn select_jobs_filters_by_requested_ids() {
        let jobs = vec![job("a"), job("b"), job("c")];
        let requested = vec!["b".to_string(), "c".to_string()];
        let selected = select_jobs(&jobs, &requested).expect("selection should succeed");
        let ids: Vec<String> = selected.into_iter().map(|j| j.id).collect();
        assert_eq!(ids, vec!["b".to_string(), "c".to_string()]);
    }

    #[test]
    fn select_jobs_errors_for_unknown_id() {
        let jobs = vec![job("a")];
        let requested = vec!["missing".to_string()];
        let err = select_jobs(&jobs, &requested).expect_err("selection should fail");
        assert!(err
            .to_string()
            .contains("Unknown plugin job id(s): missing"));
    }
}

pub async fn sync(
    manifest: PathBuf,
    config_json: Option<String>,
    config_file: Option<PathBuf>,
    dry_run: bool,
    json: bool,
) -> anyhow::Result<()> {
    let config = parse_plugin_config(config_json, config_file)?;
    let app_config = if dry_run {
        load_config_no_bootstrap()?
    } else {
        load_config()?
    };
    let result = sync_once(&app_config, manifest, config, dry_run).await?;

    if json {
        let payload = serde_json::json!({
            "sync_key": result.sync_key,
            "previous_cursor": result.previous_cursor,
            "next_cursor": result.next_cursor,
            "plugin": result.plugin,
            "ingest": result.ingest
        });
        println!("{}", serde_json::to_string_pretty(&payload)?);
        return Ok(());
    }

    eprintln!("Plugin Sync");
    eprintln!("===========");
    eprintln!("Sync key: {}", result.sync_key);
    eprintln!("Plugin: {}", result.plugin.plugin_id);
    eprintln!("Manifest: {}", result.plugin.manifest_path);
    eprintln!(
        "Previous cursor: {}",
        result
            .previous_cursor
            .as_ref()
            .map(|v| v.to_string())
            .unwrap_or_else(|| "none".into())
    );
    eprintln!(
        "Next cursor: {}",
        result
            .next_cursor
            .as_ref()
            .map(|v| v.to_string())
            .unwrap_or_else(|| "none".into())
    );
    eprintln!("Envelopes processed: {}", result.ingest.processed);
    eprintln!("Written markdown: {}", result.ingest.written);
    eprintln!("Unchanged: {}", result.ingest.unchanged);
    eprintln!("Tombstones: {}", result.ingest.tombstoned);
    eprintln!("Dry run: {}", result.ingest.dry_run);

    if !result.plugin.stderr.trim().is_empty() {
        eprintln!("\nPlugin stderr:");
        eprintln!("{}", result.plugin.stderr.trim());
    }

    if dry_run {
        eprintln!("\nDry-run mode enabled. No files or cursor state were written.");
    } else {
        eprintln!("\nSync state updated in metadata.db.");
    }
    Ok(())
}

#[derive(Debug, Serialize)]
struct SyncAllJobResult {
    job_id: String,
    enabled: bool,
    manifest: String,
    status: String,
    sync_key: Option<String>,
    processed: Option<usize>,
    written: Option<usize>,
    unchanged: Option<usize>,
    tombstones: Option<usize>,
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct SyncAllReport {
    dry_run: bool,
    jobs_requested: usize,
    jobs_selected: usize,
    jobs_run: usize,
    jobs_succeeded: usize,
    jobs_failed: usize,
    jobs_skipped: usize,
    index: Option<SyncAllIndexReport>,
    results: Vec<SyncAllJobResult>,
}

#[derive(Debug, Serialize)]
struct SyncAllIndexReport {
    status: String,
    force: bool,
    files_indexed: Option<usize>,
    files_skipped: Option<usize>,
    files_deleted: Option<usize>,
    total_chunks: Option<usize>,
    errors: Option<usize>,
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct PluginJobView {
    id: String,
    enabled: bool,
    manifest: String,
    manifest_exists: Option<bool>,
    manifest_valid: Option<bool>,
    manifest_error: Option<String>,
    plugin_id: Option<String>,
    runtime: Option<String>,
    version: Option<String>,
    config: Value,
}

fn select_jobs(all_jobs: &[PluginJob], requested: &[String]) -> anyhow::Result<Vec<PluginJob>> {
    if requested.is_empty() {
        return Ok(all_jobs.to_vec());
    }

    let mut known: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for job in all_jobs {
        known.insert(job.id.as_str());
    }

    let mut missing = Vec::new();
    for req in requested {
        if !known.contains(req.as_str()) {
            missing.push(req.clone());
        }
    }
    if !missing.is_empty() {
        anyhow::bail!("Unknown plugin job id(s): {}", missing.join(", "));
    }

    let requested_set: std::collections::HashSet<&str> =
        requested.iter().map(String::as_str).collect();
    Ok(all_jobs
        .iter()
        .filter(|job| requested_set.contains(job.id.as_str()))
        .cloned()
        .collect())
}

pub async fn jobs(json: bool, validate_manifests: bool) -> anyhow::Result<()> {
    let app_config = load_config_no_bootstrap()?;
    let mut rows = Vec::new();
    for job in &app_config.plugin_jobs {
        let mut manifest_error: Option<String> = None;
        let mut plugin_id: Option<String> = None;
        let mut runtime: Option<String> = None;
        let mut version: Option<String> = None;
        let mut manifest_valid: Option<bool> = None;
        let exists = if validate_manifests {
            let exists = job.manifest.exists();
            if exists {
                match crate::plugin_host::load_plugin_manifest(&job.manifest) {
                    Ok(m) => {
                        manifest_valid = Some(true);
                        plugin_id = Some(m.plugin_id);
                        runtime = Some(m.runtime);
                        version = Some(m.version);
                    }
                    Err(e) => {
                        manifest_valid = Some(false);
                        manifest_error = Some(e.to_string());
                    }
                }
            } else {
                manifest_valid = Some(false);
                manifest_error = Some("manifest file not found".into());
            }
            Some(exists)
        } else {
            None
        };

        rows.push(PluginJobView {
            id: job.id.clone(),
            enabled: job.enabled,
            manifest: job.manifest.display().to_string(),
            manifest_exists: exists,
            manifest_valid,
            manifest_error,
            plugin_id,
            runtime,
            version,
            config: job.config.clone(),
        });
    }
    rows.sort_by(|a, b| a.id.cmp(&b.id));

    if json {
        let payload = serde_json::json!({
            "config_path": crate::config::AppConfig::config_path().display().to_string(),
            "job_count": rows.len(),
            "jobs": rows
        });
        println!("{}", serde_json::to_string_pretty(&payload)?);
        return Ok(());
    }

    eprintln!("Plugin Jobs");
    eprintln!("===========");
    eprintln!(
        "Config: {}",
        crate::config::AppConfig::config_path().display()
    );
    eprintln!("Jobs: {}", rows.len());
    for row in &rows {
        if validate_manifests {
            eprintln!(
                "- {} enabled={} manifest={} exists={} valid={} plugin={} runtime={} version={}",
                row.id,
                row.enabled,
                row.manifest,
                row.manifest_exists.unwrap_or(false),
                row.manifest_valid.unwrap_or(false),
                row.plugin_id.as_deref().unwrap_or("-"),
                row.runtime.as_deref().unwrap_or("-"),
                row.version.as_deref().unwrap_or("-")
            );
            if let Some(err) = &row.manifest_error {
                eprintln!("    error: {err}");
            }
        } else {
            eprintln!(
                "- {} enabled={} manifest={}",
                row.id, row.enabled, row.manifest
            );
        }
    }
    if validate_manifests {
        let missing = rows
            .iter()
            .filter(|r| r.manifest_exists == Some(false))
            .count();
        let invalid = rows
            .iter()
            .filter(|r| r.manifest_exists == Some(true) && r.manifest_valid == Some(false))
            .count();
        if missing > 0 {
            eprintln!("\nWarning: {missing} job(s) reference missing manifest files.");
        }
        if invalid > 0 {
            eprintln!("\nWarning: {invalid} job(s) have invalid manifest files.");
        }
    }
    Ok(())
}

pub struct SyncAllOptions {
    pub requested_jobs: Vec<String>,
    pub include_disabled: bool,
    pub fail_fast: bool,
    pub index: bool,
    pub index_force: bool,
    pub dry_run: bool,
    pub json: bool,
}

pub async fn sync_all(opts: SyncAllOptions) -> anyhow::Result<()> {
    if opts.dry_run && opts.index {
        anyhow::bail!("`--index` cannot be used with `--dry-run`");
    }

    let app_config = if opts.dry_run {
        load_config_no_bootstrap()?
    } else {
        load_config()?
    };

    let mut selected = select_jobs(&app_config.plugin_jobs, &opts.requested_jobs)?;
    selected.sort_by(|a, b| a.id.cmp(&b.id));

    if selected.is_empty() {
        anyhow::bail!("No plugin jobs configured. Add plugins.jobs in config.yaml.");
    }
    let selected_total = selected.len();

    let mut results = Vec::new();
    let mut jobs_run = 0usize;
    let mut jobs_succeeded = 0usize;
    let mut jobs_failed = 0usize;
    let mut jobs_skipped = 0usize;

    for job in selected {
        if !job.enabled && !opts.include_disabled {
            jobs_skipped += 1;
            results.push(SyncAllJobResult {
                job_id: job.id,
                enabled: job.enabled,
                manifest: job.manifest.display().to_string(),
                status: "skipped_disabled".into(),
                sync_key: None,
                processed: None,
                written: None,
                unchanged: None,
                tombstones: None,
                error: None,
            });
            continue;
        }

        jobs_run += 1;
        match sync_once(
            &app_config,
            job.manifest.clone(),
            job.config.clone(),
            opts.dry_run,
        )
        .await
        {
            Ok(run) => {
                jobs_succeeded += 1;
                results.push(SyncAllJobResult {
                    job_id: job.id,
                    enabled: job.enabled,
                    manifest: job.manifest.display().to_string(),
                    status: "ok".into(),
                    sync_key: Some(run.sync_key),
                    processed: Some(run.ingest.processed),
                    written: Some(run.ingest.written),
                    unchanged: Some(run.ingest.unchanged),
                    tombstones: Some(run.ingest.tombstoned),
                    error: None,
                });
            }
            Err(e) => {
                jobs_failed += 1;
                results.push(SyncAllJobResult {
                    job_id: job.id,
                    enabled: job.enabled,
                    manifest: job.manifest.display().to_string(),
                    status: "error".into(),
                    sync_key: None,
                    processed: None,
                    written: None,
                    unchanged: None,
                    tombstones: None,
                    error: Some(e.to_string()),
                });
                if opts.fail_fast {
                    break;
                }
            }
        }
    }

    let report = SyncAllReport {
        dry_run: opts.dry_run,
        jobs_requested: opts.requested_jobs.len(),
        jobs_selected: selected_total,
        jobs_run,
        jobs_succeeded,
        jobs_failed,
        jobs_skipped,
        index: None,
        results,
    };

    let mut report = report;

    let mut index_failed = false;
    if opts.index {
        if report.jobs_failed > 0 {
            report.index = Some(SyncAllIndexReport {
                status: "skipped_due_job_failures".into(),
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

                    report.index = Some(SyncAllIndexReport {
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
                    report.index = Some(SyncAllIndexReport {
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

    if opts.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        eprintln!("Plugin Sync All");
        eprintln!("===============");
        eprintln!("Dry run: {}", report.dry_run);
        eprintln!(
            "Jobs: selected={}, run={}, succeeded={}, failed={}, skipped={}",
            report.jobs_selected,
            report.jobs_run,
            report.jobs_succeeded,
            report.jobs_failed,
            report.jobs_skipped
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
                eprintln!("Index error: {}", err);
            }
        }
        for row in &report.results {
            match row.status.as_str() {
                "ok" => eprintln!(
                    "- {} [{}] processed={} written={} unchanged={} tombstones={}",
                    row.job_id,
                    row.status,
                    row.processed.unwrap_or(0),
                    row.written.unwrap_or(0),
                    row.unchanged.unwrap_or(0),
                    row.tombstones.unwrap_or(0)
                ),
                "error" => eprintln!(
                    "- {} [{}] {}",
                    row.job_id,
                    row.status,
                    row.error.as_deref().unwrap_or("unknown error")
                ),
                _ => eprintln!("- {} [{}]", row.job_id, row.status),
            }
        }
    }

    if report.jobs_failed > 0 || index_failed {
        std::process::exit(1);
    }

    Ok(())
}

pub async fn configure(job_id: String, json: bool) -> anyhow::Result<()> {
    let app_config = load_config_no_bootstrap()?;

    // Find the job
    let job = app_config
        .plugin_jobs
        .iter()
        .find(|j| j.id == job_id)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Plugin job '{}' not found. Run `colibri plugins jobs` to list configured jobs.",
                job_id
            )
        })?;

    // Load manifest to check for configure hook
    let manifest = crate::plugin_host::load_plugin_manifest(&job.manifest)
        .with_context(|| format!("Failed to load manifest for job '{}'", job_id))?;

    if manifest.configure.is_none() {
        anyhow::bail!(
            "Plugin '{}' does not support interactive configuration (no configure hook in manifest).",
            manifest.plugin_id
        );
    }

    // Write current config to temp file
    let tmp_dir = std::env::temp_dir();
    let config_file_path = tmp_dir.join(format!("colibri-cfg-{}.json", job_id));
    let config_json =
        serde_json::to_string_pretty(&job.config).context("Failed to serialize current config")?;
    std::fs::write(&config_file_path, &config_json).with_context(|| {
        format!(
            "Failed to write temp config file: {}",
            config_file_path.display()
        )
    })?;

    // Run the configure hook with inherited TTY
    let result = crate::plugin_host::run_plugin_configure(&job.manifest, &config_file_path).await;

    // Read back and clean up regardless of outcome
    let readback = match &result {
        Ok(r) if !r.cancelled => {
            let text = std::fs::read_to_string(&config_file_path).ok();
            let _ = std::fs::remove_file(&config_file_path);
            text
        }
        _ => {
            let _ = std::fs::remove_file(&config_file_path);
            None
        }
    };

    let result = result?;

    if result.cancelled {
        if json {
            let payload = serde_json::json!({
                "job_id": job_id,
                "plugin_id": result.plugin_id,
                "status": "cancelled"
            });
            println!("{}", serde_json::to_string_pretty(&payload)?);
        } else {
            eprintln!("Configuration cancelled. No changes were made.");
        }
        return Ok(());
    }

    // Parse and validate new config
    let new_config_text = readback.ok_or_else(|| {
        anyhow::anyhow!("Failed to read back config file after plugin configure hook")
    })?;
    let new_config: Value = serde_json::from_str(&new_config_text)
        .context("Plugin wrote invalid JSON to config file")?;
    if !new_config.is_object() {
        anyhow::bail!("Plugin config must be a JSON object, got: {}", new_config);
    }

    // Write back to config.yaml
    let config_path = crate::config::AppConfig::config_path();
    crate::config::update_plugin_job_config(&config_path, &job_id, &new_config)?;

    if json {
        let payload = serde_json::json!({
            "job_id": job_id,
            "plugin_id": result.plugin_id,
            "status": "ok",
            "config": new_config
        });
        println!("{}", serde_json::to_string_pretty(&payload)?);
    } else {
        eprintln!("Plugin Configured");
        eprintln!("=================");
        eprintln!("Job: {}", job_id);
        eprintln!("Plugin: {}", result.plugin_id);
        eprintln!("Config updated in: {}", config_path.display());
    }

    Ok(())
}

// canonical indexing is handled by the indexer directly (no per-folder profiles).
