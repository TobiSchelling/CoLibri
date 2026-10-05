//! `colibri serve` — MCP stdio server command.

use crate::config::load_config;
use crate::mcp;

pub async fn run(check: bool, json: bool) -> anyhow::Result<()> {
    if json && !check {
        anyhow::bail!("`--json` requires `--check`");
    }

    let config = load_config()?;
    if check {
        let ready = crate::serve_ready::check(&config)?;
        if json {
            println!("{}", serde_json::to_string_pretty(&ready)?);
        } else if ready.queryable {
            eprintln!("Index is ready to serve ({}).", ready.index_path);
        } else {
            eprintln!("Index is not ready to serve:");
            for issue in &ready.issues {
                eprintln!("  - {issue}");
            }
        }
        if !ready.queryable {
            anyhow::bail!("Index not ready for serving. Run `colibri doctor`.");
        }
        return Ok(());
    }

    mcp::run_server(&config).await?;
    Ok(())
}
