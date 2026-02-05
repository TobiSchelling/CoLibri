//! `colibri search` — semantic search command.

use crate::config::load_config;
use crate::query::SearchEngine;

pub async fn run(
    query: String,
    limit: usize,
    json: bool,
    doc_type: Option<String>,
    folder: Option<String>,
) -> anyhow::Result<()> {
    let config = load_config()?;
    let engine = SearchEngine::new(&config).await?;

    let limit = limit.min(10);
    let results = engine
        .search(&query, folder.as_deref(), doc_type.as_deref(), limit)
        .await?;

    if json {
        let output = serde_json::json!({
            "query": query,
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
            if !result.folder.is_empty() {
                println!("   Folder: {}", result.folder);
            }
            // Show truncated text preview
            let preview: String = result.text.chars().take(200).collect();
            let ellipsis = if result.text.len() > 200 { "..." } else { "" };
            println!("   {preview}{ellipsis}");
            println!();
        }

        eprintln!("{} result(s) found", results.len());
    }

    Ok(())
}
