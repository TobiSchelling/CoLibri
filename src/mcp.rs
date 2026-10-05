//! MCP stdio server (JSON-RPC over stdio).
//!
//! Implements the Model Context Protocol for Claude integration.
//! Supports `search_library` and `list_books` tools.

use std::io::{BufRead, Write};

use serde_json::{json, Value};
use tracing::{debug, info};

use crate::config::AppConfig;
use crate::error::ColibriError;
use crate::query::{SearchEngine, SearchMode};

/// Run the MCP server, reading JSON-RPC from stdin and writing to stdout.
pub async fn run_server(config: &AppConfig) -> Result<(), ColibriError> {
    info!("Starting CoLibri MCP server...");
    let ready = crate::serve_ready::check(config)?;
    if !ready.queryable {
        return Err(ColibriError::Mcp(format!(
            "Index not ready for serving: {}. Run `colibri doctor`.",
            ready.issues.join("; ")
        )));
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
            "tools/list" => handle_tools_list(id, config.top_k),
            "tools/call" => handle_tools_call(id, &request, &engine, config.top_k).await,
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

fn handle_tools_list(id: Option<Value>, top_k: usize) -> Value {
    let max_limit = top_k.saturating_mul(2).max(10);
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": {
            "tools": [
                {
                    "name": "search_library",
                    "description": "Search across all indexed content sources including technical books, architecture documents, notes, and other reference materials. Performs hybrid search (BM25 + semantic vectors) by default. Optional filters scope by collection (e.g. books, vault), document path, parsed YAML frontmatter fields (e.g. area, status, DocumentType), and update time.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Natural language description of what to search for."
                            },
                            "limit": {
                                "type": "integer",
                                "description": format!("Maximum results to return (default: {top_k}, max: {max_limit})"),
                                "default": top_k,
                                "minimum": 1,
                                "maximum": max_limit
                            },
                            "mode": {
                                "type": "string",
                                "description": "Search mode. Use 'keyword' for exact terms, names, acronyms, or IDs (e.g. 'ATAM', 'C4 model', test case IDs). Use 'semantic' for conceptual/natural language queries. Use 'hybrid' (default) for queries mixing concepts with specific terms.",
                                "enum": ["hybrid", "semantic", "keyword"]
                            },
                            "collection": {
                                "type": "string",
                                "description": "Optional. Restrict to one collection (e.g. \"books\", \"vault\", \"zephyr-ctslab\")."
                            },
                            "path_includes": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Optional. Restrict to docs whose source path (relative to its source root) contains ANY of these substrings. e.g. [\"03_MY_PROJECTS/02_HEIMDALL\"]. Combine multiple substrings to broaden inclusion."
                            },
                            "path_excludes": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Optional. Drop docs whose source path contains ANY of these substrings. e.g. [\"06_ARCHIVE\", \".trash\"]."
                            },
                            "frontmatter": {
                                "type": "object",
                                "additionalProperties": {"type": "string"},
                                "description": "Optional equality match on parsed YAML frontmatter fields. Multiple keys combine with AND. Example: {\"area\":\"SIT\", \"status\":\"active\"}. Only string/number/bool values are matched (string-compared). Filters that target fields not present in a doc's frontmatter exclude that doc."
                            },
                            "since": {
                                "type": "string",
                                "format": "date-time",
                                "description": "Optional. Only docs updated on or after this RFC 3339 timestamp. e.g. \"2026-04-01T00:00:00Z\"."
                            },
                            "group_by_doc": {
                                "type": "boolean",
                                "default": true,
                                "description": "Default true. Returns one result per document with the best matching chunk plus chunk_count and frontmatter. Set false to get chunk-level results (legacy behavior)."
                            }
                        },
                        "required": ["query"]
                    }
                },
                {
                    "name": "search_books",
                    "description": "Search only imported technical books and reference materials. Filters to documents with type 'book'. Same filter and grouping options as search_library.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "What to search for in the books"
                            },
                            "limit": {
                                "type": "integer",
                                "description": format!("Maximum results to return (default: {top_k}, max: {max_limit})"),
                                "default": top_k,
                                "minimum": 1,
                                "maximum": max_limit
                            },
                            "mode": {
                                "type": "string",
                                "description": "Search mode. Use 'keyword' for exact terms, names, acronyms, or IDs (e.g. 'ATAM', 'C4 model', test case IDs). Use 'semantic' for conceptual/natural language queries. Use 'hybrid' (default) for queries mixing concepts with specific terms.",
                                "enum": ["hybrid", "semantic", "keyword"]
                            },
                            "path_includes": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Optional. Restrict to books whose source path contains ANY of these substrings."
                            },
                            "path_excludes": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Optional. Drop books whose source path contains ANY of these substrings."
                            },
                            "frontmatter": {
                                "type": "object",
                                "additionalProperties": {"type": "string"},
                                "description": "Optional equality match on parsed YAML frontmatter fields (typically empty for books)."
                            },
                            "since": {
                                "type": "string",
                                "format": "date-time",
                                "description": "Optional. Only books updated on or after this RFC 3339 timestamp."
                            },
                            "group_by_doc": {
                                "type": "boolean",
                                "default": true,
                                "description": "Default true. One result per book with the best matching chunk plus chunk_count. Set false for chunk-level results."
                            }
                        },
                        "required": ["query"]
                    }
                },
                {
                    "name": "list_books",
                    "description": "List all indexed books with metadata. Returns title, authors, source path, and chunk count.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {}
                    }
                },
                {
                    "name": "browse_topics",
                    "description": "List all topics (tags) with document counts. Optionally limited to one collection.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "collection": {
                                "type": "string",
                                "description": "Optional collection to count tags in (e.g. \"vault\")."
                            }
                        }
                    }
                }
            ]
        }
    })
}

/// Parse the new Wave 2 Cluster E filter inputs from MCP tool arguments.
/// Returns `Err(msg)` if any field has an invalid type/format.
fn parse_filter_extras(arguments: &Value) -> Result<crate::query::SearchFilter, String> {
    use std::collections::BTreeMap;

    let mut filter = crate::query::SearchFilter::default();

    if let Some(v) = arguments.get("collection") {
        let s = v
            .as_str()
            .ok_or_else(|| "collection must be a string".to_string())?;
        filter.collection = Some(s.to_string());
    }

    if let Some(arr) = arguments.get("path_includes") {
        let arr = arr
            .as_array()
            .ok_or_else(|| "path_includes must be an array of strings".to_string())?;
        for v in arr {
            let s = v
                .as_str()
                .ok_or_else(|| "path_includes entries must be strings".to_string())?;
            filter.path_includes.push(s.to_string());
        }
    }
    if let Some(arr) = arguments.get("path_excludes") {
        let arr = arr
            .as_array()
            .ok_or_else(|| "path_excludes must be an array of strings".to_string())?;
        for v in arr {
            let s = v
                .as_str()
                .ok_or_else(|| "path_excludes entries must be strings".to_string())?;
            filter.path_excludes.push(s.to_string());
        }
    }
    if let Some(obj) = arguments.get("frontmatter") {
        let obj = obj
            .as_object()
            .ok_or_else(|| "frontmatter must be an object of {field: value} strings".to_string())?;
        let mut map = BTreeMap::new();
        for (k, v) in obj {
            let s = match v {
                Value::String(s) => s.clone(),
                Value::Number(n) => n.to_string(),
                Value::Bool(b) => b.to_string(),
                _ => {
                    return Err(format!(
                        "frontmatter[{k}] must be a string, number, or bool"
                    ))
                }
            };
            map.insert(k.clone(), s);
        }
        filter.frontmatter = map;
    }
    if let Some(v) = arguments.get("since") {
        let s = v
            .as_str()
            .ok_or_else(|| "since must be an RFC 3339 string".to_string())?;
        let parsed = chrono::DateTime::parse_from_rfc3339(s)
            .map_err(|e| format!("since not RFC 3339: {e}"))?
            .with_timezone(&chrono::Utc);
        filter.since = Some(parsed);
    }

    Ok(filter)
}

async fn handle_tools_call(
    id: Option<Value>,
    request: &Value,
    engine: &SearchEngine,
    top_k: usize,
) -> Value {
    let max_limit = top_k.saturating_mul(2).max(10);
    let params = request.get("params").cloned().unwrap_or(json!({}));
    let tool_name = params.get("name").and_then(|n| n.as_str()).unwrap_or("");
    let arguments = params.get("arguments").cloned().unwrap_or(json!({}));

    let mode = arguments
        .get("mode")
        .and_then(|m| m.as_str())
        .map(|m| m.parse::<SearchMode>())
        .transpose()
        .map_err(|e| e.to_string());

    let mode = match mode {
        Ok(m) => m.unwrap_or_default(),
        Err(e) => {
            return error_response(id, -32602, &e);
        }
    };

    // Wave 2 Cluster E filter parsing — shared between search_library and search_books.
    let filter_extras = match parse_filter_extras(&arguments) {
        Ok(f) => f,
        Err(msg) => return error_response(id, -32602, &msg),
    };
    // MCP default: group_by_doc = true (LLM clients almost always want
    // document-level results, not chunk lists). Override with `group_by_doc: false`.
    let group_by_doc = arguments
        .get("group_by_doc")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);

    let result = match tool_name {
        "search_library" => {
            let query = arguments
                .get("query")
                .and_then(|q| q.as_str())
                .unwrap_or("");
            let limit = arguments
                .get("limit")
                .and_then(|l| l.as_u64())
                .map(|l| (l as usize).min(max_limit))
                .unwrap_or(top_k);

            let filter = crate::query::SearchFilter {
                doc_type: None,
                ..filter_extras.clone()
            };
            match engine
                .search(query, &filter, group_by_doc, limit, mode)
                .await
            {
                Ok(results) => {
                    let output = json!({
                        "query": query,
                        "search_mode": mode.to_string(),
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
                .map(|l| (l as usize).min(max_limit))
                .unwrap_or(top_k);

            let filter = crate::query::SearchFilter {
                doc_type: Some("book".to_string()),
                ..filter_extras.clone()
            };
            match engine
                .search(query, &filter, group_by_doc, limit, mode)
                .await
            {
                Ok(results) => {
                    let output = json!({
                        "query": query,
                        "search_mode": mode.to_string(),
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
            let collection = arguments.get("collection").and_then(|f| f.as_str());
            match engine.browse_topics(collection).await {
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
    use super::{handle_tools_list, parse_filter_extras};
    use serde_json::{json, Value};

    fn tool<'a>(list: &'a Value, name: &str) -> &'a Value {
        list["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .find(|t| t["name"] == name)
            .unwrap_or_else(|| panic!("tool {name} missing"))
    }

    fn param_names(tool: &Value) -> Vec<String> {
        let mut names: Vec<String> = tool["inputSchema"]["properties"]
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect();
        names.sort();
        names
    }

    // AC-008.1: tool names and parameters stay compatible.
    #[test]
    fn tool_names_and_parameters() {
        let list = handle_tools_list(Some(json!(1)), 10);
        let names: Vec<&str> = list["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["name"].as_str().unwrap())
            .collect();
        assert_eq!(
            names,
            [
                "search_library",
                "search_books",
                "list_books",
                "browse_topics"
            ]
        );

        let search = [
            "collection",
            "frontmatter",
            "group_by_doc",
            "limit",
            "mode",
            "path_excludes",
            "path_includes",
            "query",
            "since",
        ];
        assert_eq!(param_names(tool(&list, "search_library")), search);
        let books: Vec<&str> = search
            .iter()
            .copied()
            .filter(|p| *p != "collection")
            .collect();
        assert_eq!(param_names(tool(&list, "search_books")), books);
        assert!(param_names(tool(&list, "list_books")).is_empty());
        assert_eq!(param_names(tool(&list, "browse_topics")), ["collection"]);
    }

    #[test]
    fn filter_extras_parse_collection_and_reject_bad_types() {
        let f =
            parse_filter_extras(&json!({"collection": "books", "path_includes": ["a"]})).unwrap();
        assert_eq!(f.collection.as_deref(), Some("books"));
        assert_eq!(f.path_includes, ["a"]);
        assert!(parse_filter_extras(&json!({"collection": 3})).is_err());
        assert!(parse_filter_extras(&json!({"since": "yesterday"})).is_err());
    }
}
