//! `colibri serve` — MCP stdio server command.

use crate::config::load_config;
use crate::mcp;

pub async fn run(check: bool, json: bool) -> anyhow::Result<()> {
    if json && !check {
        anyhow::bail!("`--json` requires `--check`");
    }

    let config = load_config()?;
    if check {
        let report = mcp::startup_report(&config)?;
        if json {
            println!("{}", serde_json::to_string_pretty(&report)?);
        } else {
            eprintln!(
                "MCP startup profile check: queryable_profiles={}/{} (active generation: {})",
                report.queryable_profiles, report.total_profiles, report.active_generation
            );
            for issue in &report.issues {
                eprintln!("  - {}", issue);
            }
            if report.issues.is_empty() {
                eprintln!("All profiles are serve-ready.");
            }
        }

        if report.queryable_profiles == 0 {
            anyhow::bail!(
                "No queryable embedding profile is ready for serving. Run `colibri doctor`."
            );
        }
        return Ok(());
    }

    mcp::run_server(&config).await?;
    Ok(())
}
