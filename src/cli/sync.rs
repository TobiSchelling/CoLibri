//! `colibri sync` — ingest configured sources into the canonical store.

use crate::cli::plugins;

pub async fn run(
    jobs: Vec<String>,
    include_disabled: bool,
    fail_fast: bool,
    no_index: bool,
    force: bool,
    dry_run: bool,
    json: bool,
) -> anyhow::Result<()> {
    plugins::sync_all(plugins::SyncAllOptions {
        requested_jobs: jobs,
        include_disabled,
        fail_fast,
        index: !no_index,
        index_force: force,
        dry_run,
        json,
    })
    .await
}
