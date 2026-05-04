//! `colibri search` — search command with hybrid/semantic/keyword modes.

use std::collections::BTreeMap;

use chrono::{DateTime, Utc};

use crate::config::load_config;
use crate::query::{SearchEngine, SearchFilter, SearchMode};

#[allow(clippy::too_many_arguments)]
pub async fn run(
    query: String,
    limit: usize,
    json: bool,
    doc_type: Option<String>,
    classification: Option<String>,
    mode: SearchMode,
    path_includes: Vec<String>,
    path_excludes: Vec<String>,
    frontmatter: Vec<String>,
    since: Option<String>,
    group_by_doc: bool,
) -> anyhow::Result<()> {
    let config = load_config()?;
    let engine = SearchEngine::new(&config).await?;

    let limit = limit.min(config.top_k);

    // Parse `--frontmatter key=value` pairs into the filter map.
    let mut fm_map = BTreeMap::new();
    for entry in frontmatter {
        match entry.split_once('=') {
            Some((k, v)) => {
                fm_map.insert(k.trim().to_string(), v.trim().to_string());
            }
            None => {
                anyhow::bail!("--frontmatter expects KEY=VALUE; got '{entry}'");
            }
        }
    }

    let since_parsed = match since {
        Some(s) => Some(
            DateTime::parse_from_rfc3339(&s)
                .map_err(|e| anyhow::anyhow!("--since must be RFC 3339 (got '{s}'): {e}"))?
                .with_timezone(&Utc),
        ),
        None => None,
    };

    let filter = SearchFilter {
        classification,
        doc_type,
        path_includes,
        path_excludes,
        frontmatter: fm_map,
        since: since_parsed,
    };

    let results = engine
        .search(&query, &filter, group_by_doc, limit, mode)
        .await?;

    if json {
        let output = serde_json::json!({
            "query": query,
            "search_mode": mode.to_string(),
            "total_results": results.len(),
            "results": results,
        });
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        if results.is_empty() {
            eprintln!("No results found for: {query}");
            return Ok(());
        }

        for (i, result) in results.iter().enumerate() {
            println!("{}. {} (score: {:.4})", i + 1, result.title, result.score);
            println!("   File: {}", result.file);
            if !result.classification.is_empty() {
                println!("   Classification: {}", result.classification);
            }
            let preview: String = result.text.chars().take(200).collect();
            let ellipsis = if result.text.len() > 200 { "..." } else { "" };
            println!("   {preview}{ellipsis}");
            println!();
        }

        eprintln!("{} result(s) found", results.len());
    }

    Ok(())
}
