//! MCP stdio server (JSON-RPC over stdio).
//!
//! Implements the Model Context Protocol for Claude integration.
//! Supports `search_library` and `list_books` tools.

use std::io::{BufRead, Write};

use serde_json::{json, Value};
use tracing::{debug, info};

use crate::config::AppConfig;
use crate::error::ColibriError;
use crate::query::SearchEngine;

/// Run the MCP server, reading JSON-RPC from stdin and writing to stdout.
pub async fn run_server(config: &AppConfig) -> Result<(), ColibriError> {
    info!("Starting CoLibri MCP server...");

    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut stdout = stdout.lock();

    // Lazily initialize the search engine on first tool call
    let mut engine: Option<SearchEngine> = None;

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
            "tools/call" => {
                // Lazy-init the search engine
                if engine.is_none() {
                    match SearchEngine::new(config).await {
                        Ok(e) => engine = Some(e),
                        Err(e) => {
                            let resp =
                                error_response(id, -32603, &format!("Engine init failed: {e}"));
                            write_response(&mut stdout, &resp)?;
                            continue;
                        }
                    }
                }
                handle_tools_call(id, &request, engine.as_ref().unwrap()).await
            }
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
                    "description": "List all topics (tags) with document counts. Optionally filter by folder.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "folder": {
                                "type": "string",
                                "description": "Optional folder to filter by (e.g. 'Books', 'Notes')."
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
            let folder = arguments.get("folder").and_then(|f| f.as_str());
            match engine.browse_topics(folder).await {
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
