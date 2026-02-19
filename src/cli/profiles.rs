//! `colibri profiles` — show embedding profile and routing status.

use std::collections::HashMap;

use serde::Serialize;

use crate::config::{load_config, EmbeddingLocality, SCHEMA_VERSION};
use crate::index_meta::read_index_meta;
use crate::metadata_store::MetadataStore;

#[derive(Debug, Clone, Serialize)]
struct ProfileStatusRow {
    id: String,
    provider: String,
    locality: String,
    model: String,
    endpoint: String,
    index_path: String,
    status: String,
    lifecycle_status: Option<String>,
    serve_ready: bool,
    schema_version: Option<u32>,
    file_count: Option<u64>,
    chunk_count: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
struct SourceRouteRow {
    source: String,
    classification: String,
    profile_id: String,
}

#[derive(Debug, Serialize)]
struct ProfilesReport {
    colibri_home: String,
    active_generation: String,
    default_embedding_profile: String,
    profiles: Vec<ProfileStatusRow>,
    routing_policy: HashMap<String, String>,
    source_routes: Vec<SourceRouteRow>,
}

pub async fn run(json: bool) -> anyhow::Result<()> {
    let config = load_config()?;
    let generation_status = if config.metadata_db_path.exists() {
        let store = MetadataStore::open(&config.metadata_db_path)?;
        let rows = store.list_generation_entries()?;
        let mut map = HashMap::new();
        for row in rows {
            if row.generation_id == config.active_generation {
                map.insert(row.embedding_profile_id, row.status);
            }
        }
        map
    } else {
        HashMap::new()
    };

    let mut profile_ids: Vec<String> = config.embedding_profiles.keys().cloned().collect();
    profile_ids.sort();

    let mut profiles = Vec::new();
    for profile_id in profile_ids {
        let profile = config.embedding_profile(&profile_id)?;
        let index_path = config.lancedb_dir_for_profile(&profile_id);
        let meta = read_index_meta(&index_path)?;

        let schema_version = meta
            .get("schema_version")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32);
        let file_count = meta.get("file_count").and_then(|v| v.as_u64());
        let chunk_count = meta.get("chunk_count").and_then(|v| v.as_u64());
        let index_model = meta.get("embedding_model").and_then(|v| v.as_str());

        let status = if meta.is_empty() {
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
        let lifecycle_status = generation_status.get(&profile_id).cloned();
        let model_aligned = index_model.is_some_and(|m| m == profile.model);
        let lifecycle_ready = if generation_status.is_empty() {
            true
        } else {
            lifecycle_status.as_deref() == Some("ready")
        };
        let serve_ready = status == "ready" && model_aligned && lifecycle_ready;

        profiles.push(ProfileStatusRow {
            id: profile.id.clone(),
            provider: profile.provider.clone(),
            locality: match profile.locality {
                EmbeddingLocality::Local => "local".to_string(),
                EmbeddingLocality::Cloud => "cloud".to_string(),
            },
            model: profile.model.clone(),
            endpoint: profile.endpoint.clone(),
            index_path: index_path.display().to_string(),
            status,
            lifecycle_status,
            serve_ready,
            schema_version,
            file_count,
            chunk_count,
        });
    }

    let mut source_routes = Vec::new();
    for source in &config.sources {
        source_routes.push(SourceRouteRow {
            source: source.display_name().to_string(),
            classification: source.classification.clone(),
            profile_id: config.resolve_embedding_profile_id(&source.classification),
        });
    }

    source_routes.sort_by(|a, b| a.source.cmp(&b.source));

    let report = ProfilesReport {
        colibri_home: config.colibri_home.display().to_string(),
        active_generation: config.active_generation.clone(),
        default_embedding_profile: config.default_embedding_profile.clone(),
        profiles,
        routing_policy: config.routing_policy.clone(),
        source_routes,
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
            "  - {} [{}] provider={} model={} status={} lifecycle={} serve_ready={}",
            profile.id,
            profile.locality,
            profile.provider,
            profile.model,
            profile.status,
            profile
                .lifecycle_status
                .clone()
                .unwrap_or_else(|| "-".into()),
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

    eprintln!("\nSource Routes:");
    for row in &report.source_routes {
        eprintln!(
            "  {} ({}) -> {}",
            row.source, row.classification, row.profile_id
        );
    }

    Ok(())
}
