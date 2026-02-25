//! `colibri bootstrap` — first-time setup wizard.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use serde::Serialize;

use super::tool_on_path;
use crate::config::{load_config, AppConfig};
use crate::embedding::check_ollama;
use crate::error::ColibriError;

const DEFAULT_OLLAMA_BASE_URL: &str = "http://localhost:11434";
const DEFAULT_OLLAMA_MODEL: &str = "bge-m3";

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct BootstrapOptions {
    pub config_path: Option<PathBuf>,
    pub data_dir: Option<PathBuf>,
    pub init_path: Option<PathBuf>,
    pub classification: String,
    pub non_interactive: bool,
    pub json: bool,
}

#[derive(Debug, Serialize)]
struct BootstrapReport {
    config_path: String,
    wrote_config: bool,
    data_dir: String,
    sqlite3_ok: bool,
    ollama_installed: bool,
    ollama_reachable: bool,
    ollama_model: String,
    ollama_model_present: Option<bool>,
    suggested_commands: Vec<String>,
}

#[derive(Debug, Serialize)]
struct YamlDataConfig {
    directory: String,
}

#[derive(Debug, Serialize)]
struct YamlOllamaConfig {
    base_url: String,
    embedding_model: String,
}

#[derive(Debug, Serialize)]
struct YamlConfig {
    data: YamlDataConfig,
    ollama: YamlOllamaConfig,
}

fn default_config_path() -> PathBuf {
    AppConfig::config_path()
}

fn default_data_dir() -> PathBuf {
    // Mirror config.rs resolution (XDG_DATA_HOME else ~/.local/share).
    if let Ok(val) = std::env::var("COLIBRI_HOME") {
        let trimmed = val.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed);
        }
    }
    if let Ok(val) = std::env::var("COLIBRI_DATA_DIR") {
        let trimmed = val.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed);
        }
    }
    if let Ok(val) = std::env::var("XDG_DATA_HOME") {
        let trimmed = val.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed).join("colibri");
        }
    }
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".local")
        .join("share")
        .join("colibri")
}

fn prompt_line(prompt: &str, default: &str) -> anyhow::Result<String> {
    use std::io::{stdin, stdout, Write};
    let mut out = stdout();
    write!(out, "{prompt} [{default}]: ")?;
    out.flush()?;
    let mut buf = String::new();
    stdin().read_line(&mut buf)?;
    let trimmed = buf.trim();
    Ok(if trimmed.is_empty() {
        default.to_string()
    } else {
        trimmed.to_string()
    })
}

#[allow(dead_code)]
fn prompt_yes_no(prompt: &str, default_yes: bool) -> anyhow::Result<bool> {
    let default = if default_yes { "Y/n" } else { "y/N" };
    let input = prompt_line(prompt, default)?;
    let normalized = input.trim().to_ascii_lowercase();
    if normalized == "y" || normalized == "yes" {
        return Ok(true);
    }
    if normalized == "n" || normalized == "no" {
        return Ok(false);
    }
    Ok(default_yes)
}

async fn ollama_model_present(base_url: &str, model: &str) -> Result<Option<bool>, ColibriError> {
    let ok = check_ollama(base_url).await?;
    if !ok {
        return Ok(None);
    }
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .map_err(|e| ColibriError::Embedding(format!("Failed to build HTTP client: {e}")))?;
    let url = format!("{base_url}/api/tags");
    let resp = client
        .get(&url)
        .send()
        .await
        .map_err(|e| ColibriError::Embedding(format!("Ollama request failed: {e}")))?;
    if !resp.status().is_success() {
        return Ok(None);
    }
    let payload: serde_json::Value = resp.json().await.unwrap_or(serde_json::Value::Null);
    let models = payload.get("models").and_then(|v| v.as_array()).cloned();
    let Some(models) = models else {
        return Ok(None);
    };
    let needle = model.trim();
    if needle.is_empty() {
        return Ok(None);
    }
    for m in models {
        let Some(name) = m.get("name").and_then(|v| v.as_str()) else {
            continue;
        };
        if name == needle || name.starts_with(&format!("{needle}:")) {
            return Ok(Some(true));
        }
    }
    Ok(Some(false))
}

fn write_config(config_path: &Path, data_dir: &Path) -> anyhow::Result<bool> {
    if let Some(parent) = config_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let cfg = YamlConfig {
        data: YamlDataConfig {
            directory: data_dir.display().to_string(),
        },
        ollama: YamlOllamaConfig {
            base_url: DEFAULT_OLLAMA_BASE_URL.to_string(),
            embedding_model: DEFAULT_OLLAMA_MODEL.to_string(),
        },
    };

    let yaml = serde_yaml::to_string(&cfg)?;
    let wrote = if config_path.exists() {
        let existing = std::fs::read_to_string(config_path).unwrap_or_default();
        if existing == yaml {
            false
        } else {
            std::fs::write(config_path, yaml)?;
            true
        }
    } else {
        std::fs::write(config_path, yaml)?;
        true
    };
    Ok(wrote)
}

pub async fn run(opts: BootstrapOptions) -> anyhow::Result<()> {
    let mut config_path = opts.config_path.unwrap_or_else(default_config_path);
    let mut data_dir = opts.data_dir.unwrap_or_else(default_data_dir);

    let wrote_config = if opts.non_interactive {
        write_config(&config_path, &data_dir)?
    } else {
        let cfg_default = config_path.display().to_string();
        config_path = PathBuf::from(prompt_line("Config path", &cfg_default)?);
        let dir_default = data_dir.display().to_string();
        data_dir = PathBuf::from(prompt_line("Data directory (COLIBRI_HOME)", &dir_default)?);

        write_config(&config_path, &data_dir)?
    };

    // Dependency checks
    let sqlite3_ok = crate::metadata_store::require_sqlite3().is_ok();
    let ollama_installed = tool_on_path("ollama");
    let ollama_reachable = check_ollama(DEFAULT_OLLAMA_BASE_URL).await.unwrap_or(false);
    let model = DEFAULT_OLLAMA_MODEL.to_string();
    let model_present = ollama_model_present(DEFAULT_OLLAMA_BASE_URL, &model)
        .await
        .ok()
        .flatten();

    let mut suggested: BTreeSet<String> = BTreeSet::new();
    if !sqlite3_ok {
        suggested.insert("Install sqlite3 (on macOS: brew install sqlite)".into());
    }
    if !ollama_installed {
        suggested.insert("brew install ollama".into());
    }
    if !ollama_reachable {
        suggested.insert("ollama serve".into());
    }
    if let Some(false) = model_present {
        suggested.insert(format!("ollama pull {model}"));
    }

    // Load and validate config.
    {
        let _prev = std::env::var("COLIBRI_CONFIG_PATH").ok();
        std::env::set_var("COLIBRI_CONFIG_PATH", &config_path);
        let cfg = load_config().map_err(|e| anyhow::anyhow!(e.to_string()))?;

        // Initialize storage metadata when sqlite3 is available.
        if sqlite3_ok {
            let _ = cfg.apply_migrations();
        }

        if let Some(prev) = _prev {
            std::env::set_var("COLIBRI_CONFIG_PATH", prev);
        } else {
            std::env::remove_var("COLIBRI_CONFIG_PATH");
        }
    }

    let report = BootstrapReport {
        config_path: config_path.display().to_string(),
        wrote_config,
        data_dir: data_dir.display().to_string(),
        sqlite3_ok,
        ollama_installed,
        ollama_reachable,
        ollama_model: model,
        ollama_model_present: model_present,
        suggested_commands: suggested.into_iter().collect(),
    };

    if opts.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }

    eprintln!("CoLibri Bootstrap");
    eprintln!("=================\n");
    eprintln!("Config: {}", report.config_path);
    eprintln!("Data dir: {}", report.data_dir);
    eprintln!("Wrote config: {}", report.wrote_config);

    eprintln!("\nCore dependencies:");
    eprintln!(
        "  sqlite3: {}",
        if report.sqlite3_ok { "OK" } else { "MISSING" }
    );
    eprintln!(
        "  ollama: {}",
        if report.ollama_installed {
            "OK"
        } else {
            "MISSING"
        }
    );
    eprintln!(
        "  ollama running: {}",
        if report.ollama_reachable {
            "OK"
        } else {
            "UNREACHABLE"
        }
    );
    if let Some(present) = report.ollama_model_present {
        eprintln!(
            "  ollama model {}: {}",
            report.ollama_model,
            if present { "OK" } else { "MISSING" }
        );
    } else {
        eprintln!(
            "  ollama model {}: unknown (server not reachable)",
            report.ollama_model
        );
    }

    if !report.suggested_commands.is_empty() {
        eprintln!("\nSuggested commands:");
        for cmd in &report.suggested_commands {
            eprintln!("  {cmd}");
        }
    }

    eprintln!("\nNext:");
    eprintln!("  colibri doctor");
    eprintln!("  colibri sync");

    Ok(())
}
