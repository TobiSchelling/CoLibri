//! `colibri tour` — human-oriented help topics.

use std::collections::BTreeMap;

fn topics() -> BTreeMap<&'static str, &'static str> {
    let mut map = BTreeMap::new();
    map.insert(
        "getting-started",
        include_str!("../../docs/user/getting-started.md"),
    );
    map.insert("concepts", include_str!("../../docs/user/concepts.md"));
    map.insert("config", include_str!("../../docs/user/configuration.md"));
    map.insert("use-cases", include_str!("../../docs/user/use-cases.md"));
    map.insert(
        "troubleshooting",
        include_str!("../../docs/user/troubleshooting.md"),
    );
    map
}

pub async fn run(topic: Option<String>) -> anyhow::Result<()> {
    let map = topics();

    let Some(raw) = topic else {
        eprintln!("CoLibri Tour");
        eprintln!("===========\n");
        eprintln!("Available topics:\n");
        for key in map.keys() {
            eprintln!("  - {key}");
        }
        eprintln!("\nRun: colibri tour <topic>");
        eprintln!("Docs live in: docs/user/");
        return Ok(());
    };

    let key = raw.trim();
    let Some(body) = map.get(key) else {
        anyhow::bail!(
            "Unknown topic '{key}'. Try one of: {}",
            map.keys().cloned().collect::<Vec<_>>().join(", ")
        );
    };

    println!("{body}");
    Ok(())
}
