//! `colibri bootstrap` — first-time setup wizard.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::Serialize;

use crate::config::{load_config, AppConfig};
use crate::embedding::check_ollama;
use crate::error::ColibriError;
use crate::plugin_host::{load_plugin_manifest, RequiredEnvVar, RequiredTool};

const DEFAULT_OLLAMA_BASE_URL: &str = "http://localhost:11434";
const DEFAULT_OLLAMA_MODEL: &str = "bge-m3";

#[derive(Debug, Clone)]
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
    plugin_tools_missing: Vec<String>,
    plugin_env_missing: Vec<String>,
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

#[derive(Debug, Clone, Serialize)]
struct YamlPluginJob {
    id: String,
    manifest: String,
    enabled: bool,
    config: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct YamlPluginsConfig {
    jobs: Vec<YamlPluginJob>,
}

#[derive(Debug, Serialize)]
struct YamlConfig {
    data: YamlDataConfig,
    ollama: YamlOllamaConfig,
    plugins: YamlPluginsConfig,
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

fn tool_on_path(tool: &str) -> bool {
    Command::new("which")
        .arg(tool)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn tool_available(spec: &str) -> bool {
    let trimmed = spec.trim();
    if trimmed.is_empty() {
        return false;
    }
    let path = PathBuf::from(trimmed);
    if path.is_absolute() || trimmed.contains('/') {
        return path.exists();
    }
    tool_on_path(trimmed)
}

fn config_string(config: &serde_json::Value, key: &str) -> Option<String> {
    config
        .get(key)
        .and_then(|v| v.as_str())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
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

fn resolve_required_tool_spec(
    tool: &RequiredTool,
    job_config: &serde_json::Value,
) -> Option<String> {
    if let Some(key) = tool.check_from_config.as_deref() {
        config_string(job_config, key).or_else(|| tool.default.clone())
    } else {
        tool.check.clone()
    }
}

fn resolve_required_env_name(
    var: &RequiredEnvVar,
    job_config: &serde_json::Value,
) -> Option<String> {
    if let Some(key) = var.name_from_config.as_deref() {
        config_string(job_config, key).or_else(|| var.default.clone())
    } else {
        var.name.clone()
    }
}

struct PluginRequirementsCheck {
    missing_tools: Vec<String>,
    missing_env: Vec<String>,
    suggested_commands: Vec<String>,
}

fn gather_plugin_requirements(
    jobs: &[(PathBuf, serde_json::Value)],
) -> Result<PluginRequirementsCheck, ColibriError> {
    let mut missing_tools: BTreeSet<String> = BTreeSet::new();
    let mut missing_env: BTreeSet<String> = BTreeSet::new();
    let mut commands: BTreeSet<String> = BTreeSet::new();

    for (manifest_path, job_config) in jobs {
        let manifest = load_plugin_manifest(manifest_path)?;
        if let Some(req) = &manifest.requirements {
            if let Some(tools) = &req.tools {
                for tool in tools {
                    let spec = resolve_required_tool_spec(tool, job_config);
                    let Some(spec) = spec else {
                        continue;
                    };
                    if !tool_available(&spec) {
                        let optional = tool.optional.unwrap_or(false);
                        let prefix = if optional { "optional " } else { "" };
                        missing_tools.insert(format!("{prefix}{spec} ({})", manifest.plugin_id));
                        if let Some(formula) = tool.brew.as_deref() {
                            commands.insert(format!("brew install {formula}"));
                        }
                        if let Some(cask) = tool.brew_cask.as_deref() {
                            commands.insert(format!("brew install --cask {cask}"));
                        }
                        if let Some(pkg) = tool.pipx.as_deref() {
                            commands.insert(format!("pipx install {pkg}"));
                        }
                        if let Some(hint) = tool.install_hint.as_deref() {
                            commands.insert(hint.to_string());
                        }
                    }
                }
            }
            if let Some(env) = &req.env {
                for var in env {
                    let Some(name) = resolve_required_env_name(var, job_config) else {
                        continue;
                    };
                    let present = std::env::var(&name).ok().is_some_and(|v| !v.is_empty());
                    if var.required && !present {
                        missing_env.insert(format!("{name} ({})", manifest.plugin_id));
                        if let Some(hint) = var.hint.as_deref() {
                            commands.insert(hint.to_string());
                        } else {
                            commands.insert(format!("export {name}=..."));
                        }
                    }
                }
            }
        }
    }

    Ok(PluginRequirementsCheck {
        missing_tools: missing_tools.into_iter().collect(),
        missing_env: missing_env.into_iter().collect(),
        suggested_commands: commands.into_iter().collect(),
    })
}

fn write_config(
    config_path: &Path,
    data_dir: &Path,
    init_job: Option<YamlPluginJob>,
) -> anyhow::Result<bool> {
    if let Some(parent) = config_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let jobs = init_job.into_iter().collect::<Vec<_>>();
    let cfg = YamlConfig {
        data: YamlDataConfig {
            directory: data_dir.display().to_string(),
        },
        ollama: YamlOllamaConfig {
            base_url: DEFAULT_OLLAMA_BASE_URL.to_string(),
            embedding_model: DEFAULT_OLLAMA_MODEL.to_string(),
        },
        plugins: YamlPluginsConfig { jobs },
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

    let init_job = if let Some(root) = opts.init_path.clone() {
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("plugins/bundled/filesystem_documents/plugin_manifest.json");
        let job_config = serde_json::json!({
            "root_path": root.display().to_string(),
            "classification": opts.classification,
            "include_extensions": [".md", ".markdown"]
        });
        Some(YamlPluginJob {
            id: "docs".to_string(),
            manifest: manifest.display().to_string(),
            enabled: true,
            config: job_config,
        })
    } else {
        None
    };

    let wrote_config = if opts.non_interactive {
        write_config(&config_path, &data_dir, init_job.clone())?
    } else {
        let cfg_default = config_path.display().to_string();
        config_path = PathBuf::from(prompt_line("Config path", &cfg_default)?);
        let dir_default = data_dir.display().to_string();
        data_dir = PathBuf::from(prompt_line("Data directory (COLIBRI_HOME)", &dir_default)?);

        let init = if init_job.is_none() {
            prompt_yes_no("Initialize a folder sync job (markdown-only)?", true)?
        } else {
            true
        };

        let job = if init {
            let root_default = opts
                .init_path
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| {
                    dirs::home_dir()
                        .unwrap_or_else(|| PathBuf::from("."))
                        .join("Documents")
                        .display()
                        .to_string()
                });
            let root = prompt_line("Root path", &root_default)?;
            let class = prompt_line(
                "Classification (restricted/confidential/internal/public)",
                "internal",
            )?;
            let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("plugins/bundled/filesystem_documents/plugin_manifest.json");
            Some(YamlPluginJob {
                id: "docs".to_string(),
                manifest: manifest.display().to_string(),
                enabled: true,
                config: serde_json::json!({
                    "root_path": root,
                    "classification": class,
                    "include_extensions": [".md", ".markdown"]
                }),
            })
        } else {
            None
        };

        write_config(&config_path, &data_dir, job)?
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

    // Gather plugin requirements from the config we just wrote.
    // We load config with env overrides by setting COLIBRI_CONFIG_PATH for this process.
    let (plugin_tools_missing, plugin_env_missing) = {
        let _prev = std::env::var("COLIBRI_CONFIG_PATH").ok();
        std::env::set_var("COLIBRI_CONFIG_PATH", &config_path);
        let cfg = load_config().map_err(|e| anyhow::anyhow!(e.to_string()))?;

        // Initialize storage metadata when sqlite3 is available.
        if sqlite3_ok {
            let _ = cfg.apply_migrations();
        }

        let mut manifests: Vec<(PathBuf, serde_json::Value)> = Vec::new();
        for job in &cfg.plugin_jobs {
            if !job.enabled {
                continue;
            }
            manifests.push((job.manifest.clone(), job.config.clone()));
        }
        let check = gather_plugin_requirements(&manifests)?;

        for cmd in &check.suggested_commands {
            suggested.insert(cmd.to_string());
        }

        if let Some(prev) = _prev {
            std::env::set_var("COLIBRI_CONFIG_PATH", prev);
        } else {
            std::env::remove_var("COLIBRI_CONFIG_PATH");
        }
        (check.missing_tools, check.missing_env)
    };

    let report = BootstrapReport {
        config_path: config_path.display().to_string(),
        wrote_config,
        data_dir: data_dir.display().to_string(),
        sqlite3_ok,
        ollama_installed,
        ollama_reachable,
        ollama_model: model,
        ollama_model_present: model_present,
        plugin_tools_missing,
        plugin_env_missing,
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

    if !report.plugin_tools_missing.is_empty() || !report.plugin_env_missing.is_empty() {
        eprintln!("\nPlugin requirements:");
        if !report.plugin_tools_missing.is_empty() {
            eprintln!("  Missing tools:");
            for item in &report.plugin_tools_missing {
                eprintln!("    - {item}");
            }
        }
        if !report.plugin_env_missing.is_empty() {
            eprintln!("  Missing env vars:");
            for item in &report.plugin_env_missing {
                eprintln!("    - {item}");
            }
        }
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
