//! `colibri generations` — list and activate index generations.

use std::collections::HashMap;

use serde::Serialize;

use crate::config::{load_config, DEFAULT_ACTIVE_GENERATION, SCHEMA_VERSION};
use crate::index_meta::read_index_meta;
use crate::metadata_store::MetadataStore;

#[derive(Debug, Clone, Serialize)]
struct GenerationProfileStatus {
    profile_id: String,
    status: String,
    lifecycle_status: Option<String>,
    lifecycle_activated_at: Option<String>,
    indexed_docs: Option<u64>,
    error_docs: Option<u64>,
    deleted_docs: Option<u64>,
    schema_version: Option<u32>,
    file_count: Option<u64>,
    chunk_count: Option<u64>,
    index_path: String,
}

#[derive(Debug, Clone, Serialize)]
struct GenerationStatus {
    generation_id: String,
    active: bool,
    ready_profiles: usize,
    total_profiles: usize,
    profiles: Vec<GenerationProfileStatus>,
}

#[derive(Debug, Serialize)]
struct GenerationReport {
    active_generation: String,
    generations: Vec<GenerationStatus>,
}

pub async fn list(json: bool) -> anyhow::Result<()> {
    let config = load_config()?;
    let mut generations = config.list_generations()?;
    let metadata_store = if config.metadata_db_path.exists() {
        Some(MetadataStore::open(&config.metadata_db_path)?)
    } else {
        None
    };
    let generation_rows = if let Some(store) = &metadata_store {
        store.list_generation_entries()?
    } else {
        Vec::new()
    };
    let mut generation_state: HashMap<(String, String), (String, Option<String>)> = HashMap::new();
    for row in generation_rows {
        generation_state.insert(
            (row.generation_id, row.embedding_profile_id),
            (row.status, row.activated_at),
        );
    }

    if !generations.iter().any(|g| g == &config.active_generation) {
        generations.push(config.active_generation.clone());
        generations.sort();
    }

    let mut report_rows = Vec::new();
    for generation_id in generations {
        let mut profile_ids: Vec<String> = config.embedding_profiles.keys().cloned().collect();
        profile_ids.sort();

        let mut profiles = Vec::new();
        let mut ready_profiles = 0usize;
        for profile_id in profile_ids {
            let index_path = config.lancedb_dir_for_generation_profile(&generation_id, &profile_id);
            let meta = read_index_meta(&index_path)?;
            let lifecycle_state = generation_state
                .get(&(generation_id.clone(), profile_id.clone()))
                .cloned();
            let state_counts = if let Some(store) = &metadata_store {
                Some(store.document_index_state_counts(&generation_id, &profile_id)?)
            } else {
                None
            };

            let schema_version = meta
                .get("schema_version")
                .and_then(|v| v.as_u64())
                .map(|v| v as u32);
            let file_count = meta.get("file_count").and_then(|v| v.as_u64());
            let chunk_count = meta.get("chunk_count").and_then(|v| v.as_u64());

            let status = if meta.is_empty() {
                "not_indexed".to_string()
            } else if schema_version != Some(SCHEMA_VERSION) {
                format!(
                    "outdated(v{}, need v{})",
                    schema_version.unwrap_or(0),
                    SCHEMA_VERSION
                )
            } else {
                ready_profiles += 1;
                "ready".to_string()
            };

            profiles.push(GenerationProfileStatus {
                profile_id,
                status,
                lifecycle_status: lifecycle_state.as_ref().map(|s| s.0.clone()),
                lifecycle_activated_at: lifecycle_state.and_then(|s| s.1),
                indexed_docs: state_counts.as_ref().map(|c| c.indexed),
                error_docs: state_counts.as_ref().map(|c| c.error),
                deleted_docs: state_counts.as_ref().map(|c| c.deleted),
                schema_version,
                file_count,
                chunk_count,
                index_path: index_path.display().to_string(),
            });
        }

        report_rows.push(GenerationStatus {
            generation_id: generation_id.clone(),
            active: generation_id == config.active_generation,
            ready_profiles,
            total_profiles: config.embedding_profiles.len(),
            profiles,
        });
    }

    let report = GenerationReport {
        active_generation: config.active_generation,
        generations: report_rows,
    };

    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }

    eprintln!("CoLibri Generations");
    eprintln!("===================\n");
    eprintln!("Active generation: {}", report.active_generation);
    eprintln!("Known generations: {}", report.generations.len());

    for generation in &report.generations {
        let marker = if generation.active { "*" } else { " " };
        eprintln!(
            "\n{} {} ({}/{}) ready profiles",
            marker, generation.generation_id, generation.ready_profiles, generation.total_profiles
        );

        for profile in &generation.profiles {
            eprintln!(
                "    - {} status={} lifecycle={} model_files={} chunks={} docs(indexed/error/deleted)={}/{}/{}",
                profile.profile_id,
                profile.status,
                profile
                    .lifecycle_status
                    .clone()
                    .unwrap_or_else(|| "-".into()),
                profile
                    .file_count
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "-".into()),
                profile
                    .chunk_count
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "-".into()),
                profile
                    .indexed_docs
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "-".into()),
                profile
                    .error_docs
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "-".into()),
                profile
                    .deleted_docs
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "-".into())
            );
            if let Some(activated_at) = &profile.lifecycle_activated_at {
                eprintln!("      activated_at={activated_at}");
            }
        }
    }

    eprintln!("\nUse `colibri generations activate <id>` to switch.");
    Ok(())
}

fn generation_serve_ready_profile_count(
    config: &crate::config::AppConfig,
    generation_id: &str,
) -> Result<usize, crate::error::ColibriError> {
    let mut ready = 0usize;
    let mut profile_ids: Vec<String> = config.embedding_profiles.keys().cloned().collect();
    profile_ids.sort();

    for profile_id in profile_ids {
        let profile = config.embedding_profile(&profile_id)?;
        let index_path = config.lancedb_dir_for_generation_profile(generation_id, &profile_id);
        let meta = read_index_meta(&index_path)?;
        if meta.is_empty() {
            continue;
        }
        let schema_version = meta
            .get("schema_version")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32);
        let index_model = meta.get("embedding_model").and_then(|v| v.as_str());
        if schema_version == Some(SCHEMA_VERSION) && index_model == Some(profile.model.as_str()) {
            ready += 1;
        }
    }

    Ok(ready)
}

pub async fn activate(generation: String, allow_unready: bool) -> anyhow::Result<()> {
    let config = load_config()?;
    let generation = config.validate_generation_id(&generation)?;
    let ready_profiles = generation_serve_ready_profile_count(&config, &generation)?;
    if ready_profiles == 0 && !allow_unready {
        anyhow::bail!(
            "Refusing to activate generation '{}' because no serve-ready profiles were found. \
             Use --allow-unready to override, or index and validate first.",
            generation
        );
    }

    let previous = config.active_generation.clone();
    let activated = config.set_active_generation(&generation)?;

    if previous == activated {
        eprintln!("Generation already active: {}", activated);
        return Ok(());
    }

    eprintln!("Active generation updated:");
    eprintln!("  from: {}", previous);
    eprintln!("  to:   {}", activated);
    eprintln!(
        "  serve-ready profiles in activated generation: {}",
        ready_profiles
    );
    eprintln!("\nNext step: run `colibri index --force` to populate the new generation.");
    Ok(())
}

pub async fn create(generation: String, activate: bool) -> anyhow::Result<()> {
    let config = load_config()?;
    let generation = config.validate_generation_id(&generation)?;

    let gen_cfg = config.with_generation(&generation)?;
    std::fs::create_dir_all(gen_cfg.indexes_dir.join(&generation))?;
    for profile_id in config.embedding_profiles.keys() {
        std::fs::create_dir_all(gen_cfg.lancedb_dir_for_profile(profile_id))?;
    }
    config.ensure_generation_metadata(&generation, "prepared")?;

    eprintln!("Generation prepared: {}", generation);

    if activate {
        let previous = config.active_generation.clone();
        let activated = config.set_active_generation(&generation)?;
        if previous == activated {
            eprintln!("Generation already active: {}", activated);
        } else {
            eprintln!("Active generation updated:");
            eprintln!("  from: {}", previous);
            eprintln!("  to:   {}", activated);
        }
    } else {
        eprintln!(
            "Use `colibri generations activate {}` to switch traffic.",
            generation
        );
    }

    Ok(())
}

pub async fn delete(generation: String, confirm: String, force: bool) -> anyhow::Result<()> {
    let config = load_config()?;
    let generation = config.validate_generation_id(&generation)?;

    if confirm != generation {
        anyhow::bail!(
            "Confirmation mismatch. Use --confirm {} to delete this generation.",
            generation
        );
    }

    let gen_dir = config.generation_dir(&generation);
    if !gen_dir.exists() {
        anyhow::bail!("Generation not found: {}", generation);
    }

    let is_active = generation == config.active_generation;
    if is_active && !force {
        anyhow::bail!(
            "Refusing to delete active generation '{}'. Re-run with --force and --confirm {}.",
            generation,
            generation
        );
    }

    if is_active {
        let mut alternatives = config
            .list_generations()?
            .into_iter()
            .filter(|g| g != &generation)
            .collect::<Vec<_>>();
        alternatives.sort();

        let fallback = if let Some(next) = alternatives.first() {
            next.clone()
        } else {
            DEFAULT_ACTIVE_GENERATION.to_string()
        };

        // Ensure fallback exists and switch before delete.
        config.set_active_generation(&fallback)?;
        eprintln!(
            "Active generation switched before delete:\n  from: {}\n  to:   {}",
            generation, fallback
        );
    }

    std::fs::remove_dir_all(&gen_dir)?;
    if config.metadata_db_path.exists() {
        let store = MetadataStore::open(&config.metadata_db_path)?;
        store.delete_generation_rows(&generation)?;
        store.touch_updated_at()?;
    }
    eprintln!("Deleted generation: {}", generation);
    Ok(())
}
