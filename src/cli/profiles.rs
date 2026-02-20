//! `colibri profiles` — show embedding profile and routing status.

use std::collections::HashMap;

use serde::Serialize;

use crate::config::{load_config, EmbeddingLocality, SCHEMA_VERSION};

#[derive(Debug, Clone, Serialize)]
struct ProfileStatusRow {
    id: String,
    provider: String,
    locality: String,
    model: String,
    endpoint: String,
    index_path: String,
    status: String,
    serve_ready: bool,
    schema_version: Option<u32>,
    file_count: Option<u64>,
    chunk_count: Option<u64>,
}

#[derive(Debug, Serialize)]
struct ProfilesReport {
    colibri_home: String,
    active_generation: String,
    default_embedding_profile: String,
    profiles: Vec<ProfileStatusRow>,
    routing_policy: HashMap<String, String>,
}

pub async fn run(json: bool) -> anyhow::Result<()> {
    let config = load_config()?;
    let checks = crate::serve_ready::profile_checks(&config)?;
    let check_map: HashMap<String, crate::serve_ready::ServeReadyProfileCheck> = checks
        .into_iter()
        .map(|c| (c.profile_id.clone(), c))
        .collect();

    let mut profile_ids: Vec<String> = config.embedding_profiles.keys().cloned().collect();
    profile_ids.sort();

    let mut profiles = Vec::new();
    for profile_id in profile_ids {
        let profile = config.embedding_profile(&profile_id)?;
        let check = check_map.get(&profile_id);
        let index_path = check.map(|c| c.index_path.clone()).unwrap_or_else(|| {
            config
                .lancedb_dir_for_profile(&profile_id)
                .display()
                .to_string()
        });

        let schema_version = check.and_then(|c| c.schema_version);
        let file_count = check.and_then(|c| c.file_count);
        let chunk_count = check.and_then(|c| c.chunk_count);

        let status = if schema_version.is_none() {
            "not_indexed".to_string()
        } else if schema_version != Some(SCHEMA_VERSION) {
            format!(
                "outdated(v{}, need v{})",
                schema_version.unwrap_or(0),
                SCHEMA_VERSION
            )
        } else {
            "ready".to_string()
        };
        let serve_ready = check.is_some_and(|c| c.queryable);

        profiles.push(ProfileStatusRow {
            id: profile.id.clone(),
            provider: profile.provider.clone(),
            locality: match profile.locality {
                EmbeddingLocality::Local => "local".to_string(),
                EmbeddingLocality::Cloud => "cloud".to_string(),
            },
            model: profile.model.clone(),
            endpoint: profile.endpoint.clone(),
            index_path,
            status,
            serve_ready,
            schema_version,
            file_count,
            chunk_count,
        });
    }

    let report = ProfilesReport {
        colibri_home: config.colibri_home.display().to_string(),
        active_generation: config.active_generation.clone(),
        default_embedding_profile: config.default_embedding_profile.clone(),
        profiles,
        routing_policy: config.routing_policy.clone(),
    };

    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }

    eprintln!("CoLibri Profiles");
    eprintln!("================\n");
    eprintln!("Home: {}", report.colibri_home);
    eprintln!("Active generation: {}", report.active_generation);
    eprintln!(
        "Default embedding profile: {}",
        report.default_embedding_profile
    );

    eprintln!("\nEmbedding Profiles:");
    for profile in &report.profiles {
        eprintln!(
            "  - {} [{}] provider={} model={} status={} serve_ready={}",
            profile.id,
            profile.locality,
            profile.provider,
            profile.model,
            profile.status,
            profile.serve_ready
        );
        eprintln!("    endpoint: {}", profile.endpoint);
        eprintln!("    index: {}", profile.index_path);
        if let Some(v) = profile.schema_version {
            eprintln!("    schema: v{}", v);
        }
        if let Some(n) = profile.file_count {
            eprintln!("    files: {}", n);
        }
        if let Some(n) = profile.chunk_count {
            eprintln!("    chunks: {}", n);
        }
    }

    eprintln!("\nRouting Policy:");
    let mut rules: Vec<_> = report.routing_policy.iter().collect();
    rules.sort_by(|a, b| a.0.cmp(b.0));
    for (classification, profile_id) in rules {
        eprintln!("  {} -> {}", classification, profile_id);
    }

    Ok(())
}
