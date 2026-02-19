//! Plugin host runtime for ingestion connectors.

use std::path::{Path, PathBuf};
use std::process::Stdio;

use chrono::DateTime;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

use crate::error::ColibriError;

#[derive(Debug, Deserialize)]
pub struct PluginManifest {
    pub schema_version: u32,
    pub plugin_id: String,
    pub runtime: String,
    pub entrypoint: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct EnvelopeSource {
    pub plugin_id: String,
    pub connector_instance: String,
    pub external_id: String,
    pub uri: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct EnvelopeDocument {
    pub doc_id: String,
    pub title: String,
    pub markdown: String,
    pub content_hash: String,
    pub source_updated_at: String,
    pub deleted: bool,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct EnvelopeMetadata {
    pub doc_type: String,
    pub classification: String,
    pub tags: Option<Vec<String>>,
    pub language: Option<String>,
    pub acl_tags: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct DocumentEnvelope {
    pub schema_version: u32,
    pub source: EnvelopeSource,
    pub document: EnvelopeDocument,
    pub metadata: EnvelopeMetadata,
}

#[derive(Debug, Serialize)]
pub struct PluginRunReport {
    pub plugin_id: String,
    pub runtime: String,
    pub manifest_path: String,
    pub envelope_count: usize,
    pub deleted_count: usize,
    pub next_cursor: Option<Value>,
    pub envelopes: Vec<DocumentEnvelope>,
    pub stderr: String,
}

#[derive(Debug, Serialize)]
struct PluginRunRequest<'a> {
    config: &'a Value,
    cursor: Option<Value>,
}

pub async fn run_plugin_manifest(
    manifest_path: &Path,
    config: &Value,
    cursor: Option<Value>,
) -> Result<PluginRunReport, ColibriError> {
    let manifest_text = std::fs::read_to_string(manifest_path)?;
    let manifest: PluginManifest = serde_json::from_str(&manifest_text)?;

    if manifest.schema_version != 1 {
        return Err(ColibriError::Config(format!(
            "Unsupported plugin manifest schema_version {} (expected 1)",
            manifest.schema_version
        )));
    }

    let manifest_dir = manifest_path.parent().unwrap_or(Path::new("."));
    let entrypoint_path = resolve_entrypoint(manifest_dir, &manifest.entrypoint);

    let mut cmd = build_command(&manifest.runtime, &entrypoint_path)?;
    cmd.stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = cmd.spawn().map_err(|e| {
        ColibriError::Config(format!(
            "Failed to launch plugin '{}' entrypoint '{}': {e}",
            manifest.plugin_id,
            entrypoint_path.display()
        ))
    })?;

    if let Some(mut stdin) = child.stdin.take() {
        let req = PluginRunRequest { config, cursor };
        let payload = serde_json::to_vec(&req)?;
        stdin.write_all(&payload).await.map_err(|e| {
            ColibriError::Config(format!(
                "Failed writing plugin stdin for {}: {e}",
                manifest.plugin_id
            ))
        })?;
    }

    let output = child.wait_with_output().await.map_err(|e| {
        ColibriError::Config(format!(
            "Failed waiting for plugin '{}': {e}",
            manifest.plugin_id
        ))
    })?;
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    if !output.status.success() {
        return Err(ColibriError::Config(format!(
            "Plugin '{}' failed (exit {}): {}",
            manifest.plugin_id, output.status, stderr
        )));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let (envelopes, next_cursor) = parse_plugin_stdout(&stdout, &manifest.plugin_id)?;

    let deleted_count = envelopes.iter().filter(|e| e.document.deleted).count();

    Ok(PluginRunReport {
        plugin_id: manifest.plugin_id,
        runtime: manifest.runtime,
        manifest_path: manifest_path.display().to_string(),
        envelope_count: envelopes.len(),
        deleted_count,
        next_cursor,
        envelopes,
        stderr,
    })
}

fn parse_plugin_stdout(
    stdout: &str,
    plugin_id: &str,
) -> Result<(Vec<DocumentEnvelope>, Option<Value>), ColibriError> {
    let mut envelopes = Vec::new();
    let mut next_cursor: Option<Value> = None;

    for (idx, line) in stdout.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let parsed: Value = serde_json::from_str(line).map_err(|e| {
            ColibriError::Config(format!(
                "Plugin '{}' emitted invalid JSON at line {}: {e}",
                plugin_id,
                idx + 1
            ))
        })?;

        // Control line for incremental sync cursor.
        if let Some(obj) = parsed.as_object() {
            if obj.get("type").and_then(Value::as_str) == Some("cursor") {
                if let Some(cursor) = obj.get("cursor") {
                    next_cursor = Some(cursor.clone());
                    continue;
                }
                return Err(ColibriError::Config(format!(
                    "Plugin '{}' emitted cursor control line without 'cursor' at line {}",
                    plugin_id,
                    idx + 1
                )));
            }
        }

        let envelope: DocumentEnvelope = serde_json::from_value(parsed).map_err(|e| {
            ColibriError::Config(format!(
                "Plugin '{}' emitted invalid envelope at line {}: {e}",
                plugin_id,
                idx + 1
            ))
        })?;
        validate_envelope(&envelope, plugin_id)?;
        envelopes.push(envelope);
    }

    Ok((envelopes, next_cursor))
}

fn resolve_entrypoint(manifest_dir: &Path, entrypoint: &str) -> PathBuf {
    let candidate = PathBuf::from(entrypoint);
    if candidate.is_absolute() {
        candidate
    } else {
        manifest_dir.join(candidate)
    }
}

fn build_command(runtime: &str, entrypoint_path: &Path) -> Result<Command, ColibriError> {
    match runtime {
        "python" => {
            let mut cmd = Command::new("python3");
            cmd.arg(entrypoint_path);
            Ok(cmd)
        }
        "rust" | "external" => {
            let cmd = Command::new(entrypoint_path);
            Ok(cmd)
        }
        other => Err(ColibriError::Config(format!(
            "Unsupported plugin runtime '{other}'"
        ))),
    }
}

fn validate_envelope(
    envelope: &DocumentEnvelope,
    expected_plugin_id: &str,
) -> Result<(), ColibriError> {
    if envelope.schema_version != 1 {
        return Err(ColibriError::Config(format!(
            "Invalid envelope schema_version {} (expected 1)",
            envelope.schema_version
        )));
    }
    if envelope.source.plugin_id != expected_plugin_id {
        return Err(ColibriError::Config(format!(
            "Envelope plugin_id mismatch: expected '{}', got '{}'",
            expected_plugin_id, envelope.source.plugin_id
        )));
    }
    if envelope.document.doc_id.trim().is_empty() {
        return Err(ColibriError::Config(
            "Envelope document.doc_id cannot be empty".into(),
        ));
    }
    if envelope.document.content_hash.trim().is_empty() {
        return Err(ColibriError::Config(
            "Envelope document.content_hash cannot be empty".into(),
        ));
    }
    let hash_re = Regex::new(r"^sha256:[a-f0-9]{64}$")
        .map_err(|e| ColibriError::Config(format!("Regex compile failed: {e}")))?;
    if !hash_re.is_match(&envelope.document.content_hash) {
        return Err(ColibriError::Config(format!(
            "Envelope content_hash has invalid format: {}",
            envelope.document.content_hash
        )));
    }

    if DateTime::parse_from_rfc3339(&envelope.document.source_updated_at).is_err() {
        return Err(ColibriError::Config(format!(
            "Envelope source_updated_at is not RFC3339: {}",
            envelope.document.source_updated_at
        )));
    }

    if envelope.metadata.doc_type.trim().is_empty() {
        return Err(ColibriError::Config(
            "Envelope metadata.doc_type cannot be empty".into(),
        ));
    }

    match envelope.metadata.classification.as_str() {
        "restricted" | "confidential" | "internal" | "public" => {}
        other => {
            return Err(ColibriError::Config(format!(
                "Envelope classification must be one of restricted/confidential/internal/public, got '{other}'"
            )))
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::parse_plugin_stdout;

    fn envelope_line() -> String {
        r##"{"schema_version":1,"source":{"plugin_id":"filesystem_markdown","connector_instance":"x","external_id":"y","uri":null},"document":{"doc_id":"d1","title":"t","markdown":"# x","content_hash":"sha256:0000000000000000000000000000000000000000000000000000000000000000","source_updated_at":"2026-02-18T00:00:00Z","deleted":false},"metadata":{"doc_type":"note","classification":"internal","tags":null,"language":null,"acl_tags":null}}"##.to_string()
    }

    #[test]
    fn parse_stdout_accepts_cursor_control_line() {
        let stdout = format!(
            "{}\n{}\n",
            envelope_line(),
            r#"{"type":"cursor","cursor":{"last_scan_at":"2026-02-18T00:00:00Z"}}"#
        );
        let (envelopes, cursor) =
            parse_plugin_stdout(&stdout, "filesystem_markdown").expect("parse should succeed");
        assert_eq!(envelopes.len(), 1);
        assert!(cursor.is_some());
    }

    #[test]
    fn parse_stdout_rejects_cursor_without_payload() {
        let stdout = r#"{"type":"cursor"}"#;
        let err = parse_plugin_stdout(stdout, "filesystem_markdown")
            .err()
            .expect("expected parse error");
        assert!(
            err.to_string().contains("without 'cursor'"),
            "unexpected error: {err}"
        );
    }
}
