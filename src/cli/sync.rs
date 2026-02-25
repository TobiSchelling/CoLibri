//! `colibri sync` — ingest configured sources into the canonical store.

use crate::cli::connectors;

pub async fn run(
    connectors_filter: Vec<String>,
    include_disabled: bool,
    fail_fast: bool,
    no_index: bool,
    force: bool,
    dry_run: bool,
    json: bool,
) -> anyhow::Result<()> {
    connectors::sync_all(connectors::SyncAllOptions {
        requested_connectors: connectors_filter,
        include_disabled,
        fail_fast,
        index: !no_index,
        index_force: force,
        dry_run,
        json,
    })
    .await
}
