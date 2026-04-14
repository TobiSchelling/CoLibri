//! `colibri search` — search command with hybrid/semantic/keyword modes.

use crate::config::load_config;
use crate::query::{SearchEngine, SearchMode};

pub async fn run(
    query: String,
    limit: usize,
    json: bool,
    doc_type: Option<String>,
    classification: Option<String>,
    mode: SearchMode,
) -> anyhow::Result<()> {
    let config = load_config()?;
    let engine = SearchEngine::new(&config).await?;

    let limit = limit.min(config.top_k);
    let results = engine
        .search(
            &query,
            classification.as_deref(),
            doc_type.as_deref(),
            limit,
            mode,
        )
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
