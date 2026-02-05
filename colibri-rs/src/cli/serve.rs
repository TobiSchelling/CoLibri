//! `colibri serve` — MCP stdio server command.

use crate::config::load_config;
use crate::mcp;

pub async fn run() -> anyhow::Result<()> {
    let config = load_config()?;
    mcp::run_server(&config).await?;
    Ok(())
}
