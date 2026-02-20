//! MCP stdio server (JSON-RPC over stdio).
//!
//! Implements the Model Context Protocol for Claude integration.
//! Supports `search_library` and `list_books` tools.

use std::io::{BufRead, Write};

use serde::Serialize;
use serde_json::{json, Value};
use tracing::{debug, info};

use crate::config::AppConfig;
use crate::error::ColibriError;
use crate::query::SearchEngine;

#[derive(Debug, Clone, Serialize)]
pub struct StartupProfileCheck {
    pub profile_id: String,
    pub queryable: bool,
    pub issues: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StartupReport {
    pub active_generation: String,
    pub total_profiles: usize,
    pub queryable_profiles: usize,
    pub issues: Vec<String>,
    pub profiles: Vec<StartupProfileCheck>,
}

/// Run the MCP server, reading JSON-RPC from stdin and writing to stdout.
pub async fn run_server(config: &AppConfig) -> Result<(), ColibriError> {
    info!("Starting CoLibri MCP server...");
    let report = startup_report(config)?;
    eprintln!(
        "MCP startup profile check: queryable_profiles={}/{} (active generation: {})",
        report.queryable_profiles, report.total_profiles, report.active_generation
    );
    for issue in &report.issues {
        eprintln!("  - {}", issue);
    }
    if report.queryable_profiles == 0 {
        return Err(ColibriError::Mcp(
            "No queryable embedding profile is ready for serving. Run `colibri doctor`, then rebuild/activate a ready generation.".into(),
        ));
    }

    let engine = SearchEngine::new(config).await.map_err(|e| {
        ColibriError::Mcp(format!(
            "Search engine initialization failed at MCP startup: {e}"
        ))
    })?;

    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut stdout = stdout.lock();

    for line in stdin.lock().lines() {
        let line = line.map_err(|e| ColibriError::Mcp(format!("stdin read error: {e}")))?;
        let line = line.trim();

        if line.is_empty() {
            continue;
        }

        debug!("MCP recv: {line}");

        let request: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(e) => {
                let err_resp = json!({
                    "jsonrpc": "2.0",
                    "id": null,
                    "error": {
                        "code": -32700,
                        "message": format!("Parse error: {e}")
                    }
                });
                write_response(&mut stdout, &err_resp)?;
                continue;
            }
        };

        let id = request.get("id").cloned();
        let method = request.get("method").and_then(|m| m.as_str()).unwrap_or("");

        let response = match method {
            "initialize" => handle_initialize(id),
            "initialized" => {
                // Notification — no response needed
                continue;
            }
            "tools/list" => handle_tools_list(id),
            "tools/call" => handle_tools_call(id, &request, &engine).await,
            "notifications/cancelled" | "ping" => {
                // Ignore notifications, respond to ping
                if method == "ping" {
                    json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "result": {}
                    })
                } else {
                    continue;
                }
            }
            _ => error_response(id, -32601, &format!("Method not found: {method}")),
        };

        write_response(&mut stdout, &response)?;
    }

    Ok(())
}

pub fn startup_report(config: &AppConfig) -> Result<StartupReport, ColibriError> {
    let checks = crate::serve_ready::profile_checks(config)?;
    let profiles = checks
        .iter()
        .map(|c| StartupProfileCheck {
            profile_id: c.profile_id.clone(),
            queryable: c.queryable,
            issues: c.issues.clone(),
        })
        .collect::<Vec<_>>();

    let queryable_profiles = profiles.iter().filter(|p| p.queryable).count();
    let mut issues = Vec::new();
    for p in &profiles {
        if !p.queryable {
            issues.push(format!("{}: {}", p.profile_id, p.issues.join("; ")));
        }
    }

    Ok(StartupReport {
        active_generation: config.active_generation.clone(),
        total_profiles: config.embedding_profiles.len(),
        queryable_profiles,
        issues,
        profiles,
    })
}

fn handle_initialize(id: Option<Value>) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {}
            },
            "serverInfo": {
                "name": "colibri",
                "version": env!("CARGO_PKG_VERSION")
            }
        }
    })
}

fn handle_tools_list(id: Option<Value>) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": {
            "tools": [
                {
                    "name": "search_library",
                    "description": "Search across all indexed content sources including technical books, architecture documents, notes, and other reference materials. Performs semantic search to find relevant passages.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Natural language description of what to search for."
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum results to return (default: 5, max: 10)",
                                "default": 5,
                                "minimum": 1,
                                "maximum": 10
                            }
                        },
                        "required": ["query"]
                    }
                },
                {
                    "name": "search_books",
                    "description": "Search only imported technical books and reference materials. Filters to documents with type 'book'.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "What to search for in the books"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum results to return (default: 5, max: 10)",
                                "default": 5,
                                "minimum": 1,
                                "maximum": 10
                            }
                        },
                        "required": ["query"]
                    }
                },
                {
                    "name": "list_books",
                    "description": "List all indexed books with metadata. Returns title, chunk count, and file path.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {}
                    }
                },
                {
                    "name": "browse_topics",
                    "description": "List all topics (tags) with document counts. Optionally filter by classification.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "classification": {
                                "type": "string",
                                "description": "Optional classification to filter by (restricted/confidential/internal/public)."
                            }
                        }
                    }
                }
            ]
        }
    })
}

async fn handle_tools_call(id: Option<Value>, request: &Value, engine: &SearchEngine) -> Value {
    let params = request.get("params").cloned().unwrap_or(json!({}));
    let tool_name = params.get("name").and_then(|n| n.as_str()).unwrap_or("");
    let arguments = params.get("arguments").cloned().unwrap_or(json!({}));

    let result = match tool_name {
        "search_library" => {
            let query = arguments
                .get("query")
                .and_then(|q| q.as_str())
                .unwrap_or("");
            let limit = arguments
                .get("limit")
                .and_then(|l| l.as_u64())
                .map(|l| l.min(10) as usize)
                .unwrap_or(5);

            match engine.search_library(query, limit).await {
                Ok(results) => {
                    let output = json!({
                        "query": query,
                        "total_results": results.len(),
                        "results": results,
                    });
                    Ok(serde_json::to_string_pretty(&output).unwrap_or_default())
                }
                Err(e) => Err(format!("{e}")),
            }
        }
        "search_books" => {
            let query = arguments
                .get("query")
                .and_then(|q| q.as_str())
                .unwrap_or("");
            let limit = arguments
                .get("limit")
                .and_then(|l| l.as_u64())
                .map(|l| l.min(10) as usize)
                .unwrap_or(5);

            match engine.search_books(query, limit).await {
                Ok(results) => {
                    let output = json!({
                        "query": query,
                        "total_results": results.len(),
                        "results": results,
                    });
                    Ok(serde_json::to_string_pretty(&output).unwrap_or_default())
                }
                Err(e) => Err(format!("{e}")),
            }
        }
        "list_books" => match engine.list_books().await {
            Ok(books) => {
                let output = json!({
                    "total_books": books.len(),
                    "books": books,
                });
                Ok(serde_json::to_string_pretty(&output).unwrap_or_default())
            }
            Err(e) => Err(format!("{e}")),
        },
        "browse_topics" => {
            let classification = arguments.get("classification").and_then(|f| f.as_str());
            match engine.browse_topics(classification).await {
                Ok(topics) => {
                    let output = json!({
                        "total_topics": topics.len(),
                        "topics": topics,
                    });
                    Ok(serde_json::to_string_pretty(&output).unwrap_or_default())
                }
                Err(e) => Err(format!("{e}")),
            }
        }
        _ => Err(format!("Unknown tool: {tool_name}")),
    };

    match result {
        Ok(text) => json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": {
                "content": [{
                    "type": "text",
                    "text": text
                }]
            }
        }),
        Err(err_msg) => json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": {
                "content": [{
                    "type": "text",
                    "text": json!({"error": err_msg}).to_string()
                }],
                "isError": true
            }
        }),
    }
}

fn error_response(id: Option<Value>, code: i32, message: &str) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": {
            "code": code,
            "message": message
        }
    })
}

fn write_response(stdout: &mut impl Write, response: &Value) -> Result<(), ColibriError> {
    let line = serde_json::to_string(response)?;
    debug!("MCP send: {line}");
    writeln!(stdout, "{line}")?;
    stdout.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::startup_report;
    use crate::config::{
        AppConfig, EmbeddingLocality, EmbeddingProfile, DEFAULT_ACTIVE_GENERATION,
    };
    use crate::index_meta::write_index_meta;
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};

    fn temp_root() -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "colibri-mcp-test-{}-{}",
            std::process::id(),
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0)
        ));
        path
    }

    fn test_config(root: &Path) -> AppConfig {
        let active_generation = DEFAULT_ACTIVE_GENERATION.to_string();
        let mut embedding_profiles = HashMap::new();
        embedding_profiles.insert(
            "local_default".to_string(),
            EmbeddingProfile {
                id: "local_default".into(),
                provider: "ollama".into(),
                endpoint: "http://localhost:11434".into(),
                model: "bge-m3".into(),
                locality: EmbeddingLocality::Local,
            },
        );
        let mut routing_policy = HashMap::new();
        for class in ["restricted", "confidential", "internal", "public"] {
            routing_policy.insert(class.to_string(), "local_default".to_string());
        }

        let colibri_home = root.to_path_buf();
        let indexes_dir = colibri_home.join("indexes");
        let lancedb_dir = indexes_dir
            .join(&active_generation)
            .join("local_default")
            .join("lancedb");
        AppConfig {
            plugin_jobs: Vec::new(),
            colibri_home: colibri_home.clone(),
            data_dir: colibri_home.clone(),
            canonical_dir: colibri_home.join("canonical"),
            indexes_dir,
            state_dir: colibri_home.join("state"),
            backups_dir: colibri_home.join("backups"),
            logs_dir: colibri_home.join("logs"),
            metadata_db_path: colibri_home.join("metadata.db"),
            active_generation,
            index_dir_name: "lancedb".into(),
            embedding_profiles,
            routing_policy,
            default_embedding_profile: "local_default".into(),
            lancedb_dir,
            ollama_base_url: "http://localhost:11434".into(),
            embedding_model: "bge-m3".into(),
            top_k: 10,
            similarity_threshold: 0.3,
            chunk_size: 3000,
            chunk_overlap: 200,
        }
    }

    #[test]
    fn startup_report_flags_missing_index_metadata() {
        let root = temp_root();
        let cfg = test_config(&root);
        cfg.ensure_storage_layout().expect("bootstrap layout");

        let report = startup_report(&cfg).expect("startup report");
        assert_eq!(report.queryable_profiles, 0);
        assert_eq!(report.total_profiles, 1);
        assert_eq!(report.profiles.len(), 1);
        assert!(!report.profiles[0].queryable);
        assert!(report.profiles[0]
            .issues
            .iter()
            .any(|s| s.contains("index metadata missing")));

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn startup_report_marks_profile_queryable_when_ready() {
        let root = temp_root();
        let cfg = test_config(&root);
        cfg.ensure_storage_layout().expect("bootstrap layout");

        // Write index metadata for active generation/profile.
        std::fs::create_dir_all(&cfg.lancedb_dir).expect("create lancedb dir");
        write_index_meta(&cfg.lancedb_dir, "bge-m3", &serde_json::Map::new())
            .expect("write index meta");

        let report = startup_report(&cfg).expect("startup report");
        assert_eq!(report.queryable_profiles, 1);
        assert!(report.issues.is_empty());
        assert!(report.profiles[0].queryable);

        let _ = std::fs::remove_dir_all(root);
    }
}
