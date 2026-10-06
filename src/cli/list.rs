//! `colibri list` — books in the library.

use serde::Serialize;

use crate::config::{load_config, LIBRARY_COLLECTION};

#[derive(Serialize)]
struct BookRow {
    id: String,
    title: String,
    authors: Vec<String>,
    format: Option<String>,
    chunks: i64,
    source_path: Option<String>,
    source_missing: bool,
    status: String,
}

pub async fn run(json: bool, include_removed: bool) -> anyhow::Result<()> {
    let config = load_config()?;
    let store = config.open_read()?;
    let mut rows: Vec<BookRow> = store
        .list_documents_in(LIBRARY_COLLECTION)?
        .into_iter()
        .filter(|d| include_removed || d.is_searchable())
        .map(|d| BookRow {
            authors: serde_json::from_str(&d.authors_json).unwrap_or_default(),
            id: d.doc_id,
            title: d.title,
            format: d.format,
            chunks: d.chunk_count.unwrap_or(0),
            source_path: d.source_path,
            source_missing: d.source_missing,
            status: d.status.as_str().to_string(),
        })
        .collect();
    rows.sort_by_key(|r| r.title.to_lowercase());

    if json {
        println!("{}", serde_json::to_string_pretty(&rows)?);
        return Ok(());
    }
    for r in &rows {
        let mut flags = Vec::new();
        if r.source_missing {
            flags.push("source missing");
        }
        if r.status != "active" {
            flags.push("removed");
        }
        let flags = if flags.is_empty() {
            String::new()
        } else {
            format!(" ({})", flags.join(", "))
        };
        println!(
            "{} — {} [{}, {} chunks]{flags}",
            r.title,
            if r.authors.is_empty() {
                "unknown author".to_string()
            } else {
                r.authors.join(", ")
            },
            r.format.as_deref().unwrap_or("?"),
            r.chunks
        );
    }
    eprintln!("{} book(s)", rows.len());
    Ok(())
}
