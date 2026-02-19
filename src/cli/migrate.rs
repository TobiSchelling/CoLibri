//! `colibri migrate` — inspect/apply storage schema migrations.

use crate::config::load_config_no_bootstrap;

pub async fn run(dry_run: bool, json: bool) -> anyhow::Result<()> {
    let config = load_config_no_bootstrap()?;

    let report = if dry_run {
        config.inspect_migrations()?
    } else {
        config.apply_migrations()?
    };

    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }

    if dry_run {
        eprintln!("CoLibri Migration Check");
        eprintln!("======================");
    } else {
        eprintln!("CoLibri Migration Apply");
        eprintln!("=======================");
    }
    eprintln!("Home: {}", report.colibri_home);
    eprintln!("Up to date: {}", report.up_to_date);

    eprintln!("\nChecks:");
    for check in &report.checks {
        let current = check
            .current
            .map(|v| format!("v{v}"))
            .unwrap_or_else(|| "missing".to_string());
        eprintln!(
            "  - {}: {} (current={}, target=v{})",
            check.component, check.status, current, check.target
        );
    }

    if dry_run && !report.up_to_date {
        eprintln!("\nRun `colibri migrate` to apply pending migrations.");
    }

    Ok(())
}
