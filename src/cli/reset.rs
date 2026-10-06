//! `colibri reset` — delete CoLibri's data so it can be rebuilt from sources.
//!
//! Only paths inside the data directory that CoLibri owns are removed. The
//! config file, mirror folders and library sources are never touched.

use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};

use crate::config::{load_config, AppConfig};
use crate::lock::WriteLock;

/// Word the user has to type to confirm.
const CONFIRM_WORD: &str = "reset";

/// Data paths owned by CoLibri, including leftovers of older versions.
/// Generic directory names are only included when they carry CoLibri's
/// layout, so a data dir pointed at a shared folder loses nothing else.
fn owned_paths(config: &AppConfig) -> Vec<PathBuf> {
    let home = &config.colibri_home;
    let mut paths = vec![
        config.metadata_db_path.clone(),
        home.join("metadata.db-journal"),
        home.join("metadata.db-wal"),
        home.join("metadata.db-shm"),
        home.join("metadata.legacy-json.bak"),
        config.canonical_dir.clone(),
        config.conversions_dir.clone(),
    ];
    if home.join("index").join("lancedb").exists() {
        paths.push(home.join("index"));
    }
    // Pre-v0.15 layout, recognised by its manifest or per-generation indexes.
    if home.join("manifest.json").exists() || home.join("indexes").exists() {
        for legacy in [
            "indexes",
            "manifest.json",
            "state",
            "backups",
            "logs",
            "plugins",
        ] {
            paths.push(home.join(legacy));
        }
    }
    paths
}

fn existing(paths: Vec<PathBuf>) -> Vec<PathBuf> {
    paths.into_iter().filter(|p| p.exists()).collect()
}

fn remove(path: &Path) -> std::io::Result<()> {
    if path.is_dir() {
        std::fs::remove_dir_all(path)
    } else {
        std::fs::remove_file(path)
    }
}

/// Delete all owned data under the write lock. Returns the removed paths.
pub fn reset_data(config: &AppConfig) -> anyhow::Result<Vec<PathBuf>> {
    let _lock = WriteLock::acquire(&config.lock_path)?;
    let targets = existing(owned_paths(config));
    for path in &targets {
        remove(path)?;
    }
    Ok(targets)
}

fn is_confirmed(input: &str) -> bool {
    input.trim() == CONFIRM_WORD
}

pub async fn run(yes: bool) -> anyhow::Result<()> {
    let config = load_config()?;
    let targets = existing(owned_paths(&config));
    if targets.is_empty() {
        eprintln!("Nothing to reset in {}.", config.colibri_home.display());
        return Ok(());
    }

    eprintln!("This deletes CoLibri's data (sources and config stay untouched):");
    for path in &targets {
        eprintln!("  {}", path.display());
    }
    if !yes {
        eprint!("Type '{CONFIRM_WORD}' to continue: ");
        std::io::stderr().flush()?;
        let mut answer = String::new();
        std::io::stdin().lock().read_line(&mut answer)?;
        if !is_confirmed(&answer) {
            eprintln!("Aborted. Nothing was deleted.");
            return Ok(());
        }
    }

    let removed = reset_data(&config)?;
    eprintln!(
        "Removed {} path(s). Running `colibri serve` processes need a restart.",
        removed.len()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // AC-004.1: reset removes owned data and nothing else.
    #[test]
    fn reset_removes_owned_data_only() {
        let root = tempfile::TempDir::new().unwrap();
        let home = root.path().join("colibri");
        let config = AppConfig::for_test(&home);
        {
            let (_lock, store) = config.open_for_write().unwrap();
            drop(store);
        }
        std::fs::create_dir_all(config.canonical_dir.join("vault")).unwrap();
        std::fs::write(config.canonical_dir.join("vault/a.md"), "x").unwrap();
        std::fs::create_dir_all(home.join("indexes/gen_default")).unwrap();
        std::fs::write(home.join("manifest.json"), "{}").unwrap();
        let config_file = root.path().join("config.yaml");
        std::fs::write(&config_file, "{}").unwrap();
        let mirror = root.path().join("mirrors/zephyr");
        std::fs::create_dir_all(&mirror).unwrap();
        std::fs::write(mirror.join("T-1.md"), "keep").unwrap();

        let removed = reset_data(&config).unwrap();
        assert!(removed.contains(&config.metadata_db_path));
        assert!(!config.metadata_db_path.exists());
        assert!(!config.canonical_dir.exists());
        assert!(!home.join("index").exists());
        assert!(!home.join("indexes").exists());
        assert!(!home.join("manifest.json").exists());
        assert!(config_file.exists());
        assert_eq!(
            std::fs::read_to_string(mirror.join("T-1.md")).unwrap(),
            "keep"
        );
    }

    #[test]
    fn generic_dirs_survive_without_colibri_markers() {
        let root = tempfile::TempDir::new().unwrap();
        let config = AppConfig::for_test(root.path());
        std::fs::create_dir_all(root.path().join("logs")).unwrap();
        std::fs::create_dir_all(root.path().join("index")).unwrap();
        std::fs::write(root.path().join("logs/app.log"), "x").unwrap();

        reset_data(&config).unwrap();
        assert!(root.path().join("logs/app.log").exists());
        assert!(root.path().join("index").exists());
    }

    // AC-004.2: only the exact word confirms.
    #[test]
    fn confirmation_requires_exact_word() {
        assert!(is_confirmed("reset\n"));
        assert!(!is_confirmed("yes\n"));
        assert!(!is_confirmed("Reset"));
        assert!(!is_confirmed(""));
    }

    #[test]
    fn reset_refuses_while_a_writer_holds_the_lock() {
        let root = tempfile::TempDir::new().unwrap();
        let config = AppConfig::for_test(&root.path().join("colibri"));
        let (_lock, _store) = config.open_for_write().unwrap();
        assert!(reset_data(&config).is_err());
        assert!(config.metadata_db_path.exists());
    }
}
