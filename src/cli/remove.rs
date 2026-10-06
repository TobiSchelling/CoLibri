//! `colibri remove` — take a book out of the library.

use crate::config::load_config;
use crate::ingest::library::{find_books, remove_book};

pub async fn run(query: String) -> anyhow::Result<()> {
    let config = load_config()?;
    let (_lock, store) = config.open_for_write()?;
    let matches = find_books(&store, &query)?;
    match matches.as_slice() {
        [] => anyhow::bail!("No book matches '{query}'. See `colibri list`."),
        [book] => {
            remove_book(&store, &book.doc_id)?;
            eprintln!("Removed: {}", book.title);
            eprintln!("It no longer appears in search; its index chunks are dropped by the next `colibri update`.");
            if let Some(path) = &book.source_path {
                eprintln!("Sweeps will not add it again. To restore it: colibri add \"{path}\"");
            }
            Ok(())
        }
        many => {
            eprintln!("'{query}' matches {} books:", many.len());
            for b in many {
                eprintln!("  {}  {}", b.doc_id, b.title);
            }
            anyhow::bail!("Be more specific, or pass the id.")
        }
    }
}
