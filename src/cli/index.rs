//! `colibri index` — index markdown corpus command.

use crate::config::load_config;
use crate::indexer::index_library;

pub async fn run(folder: Option<String>, force: bool) -> anyhow::Result<()> {
    let config = load_config()?;

    let result = index_library(&config, folder.as_deref(), force).await?;

    // Exit with non-zero if there were errors
    if result.errors > 0 {
        std::process::exit(1);
    }

    Ok(())
}
