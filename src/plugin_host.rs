//! Plugin host runtime for ingestion connectors.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::OnceLock;
use std::time::Duration;

use chrono::DateTime;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::io::AsyncWriteExt;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, BufReader};
use tokio::process::Command;
use tokio::time::timeout;

use crate::error::ColibriError;

static CONTENT_HASH_RE: OnceLock<Regex> = OnceLock::new();

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(dead_code)] // Fields validated by serde deny_unknown_fields during deserialization.
pub struct PluginCapabilities {
    snapshot: bool,
    incremental: bool,
    webhook: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RequiredTool {
    pub check: Option<String>,
    pub check_from_config: Option<String>,
    pub default: Option<String>,
    pub brew: Option<String>,
    pub brew_cask: Option<String>,
    pub pipx: Option<String>,
    pub install_hint: Option<String>,
    pub optional: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RequiredEnvVar {
    pub name: Option<String>,
    pub name_from_config: Option<String>,
    pub default: Option<String>,
    pub required: bool,
    pub hint: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PluginRequirements {
    pub tools: Option<Vec<RequiredTool>>,
    pub env: Option<Vec<RequiredEnvVar>>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(dead_code)] // Used by run_plugin_configure(); wired in Task 4.
pub struct PluginConfigureHook {
    pub entrypoint: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PluginManifest {
    pub schema_version: u32,
    pub plugin_id: String,
    pub version: String,
    pub runtime: String,
    pub entrypoint: String,
    #[allow(dead_code)] // Required by serde schema validation (deny_unknown_fields).
    pub capabilities: PluginCapabilities,
    pub requirements: Option<PluginRequirements>,
    #[allow(dead_code)] // Used by run_plugin_configure(); wired in Task 4.
    pub configure: Option<PluginConfigureHook>,
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
    pub stderr_truncated: bool,
}

/// Result of running a plugin's configure hook.
#[derive(Debug)]
#[allow(dead_code)] // Wired in Task 4 (CLI command).
pub struct PluginConfigureResult {
    pub plugin_id: String,
    pub exit_code: i32,
    pub cancelled: bool,
}

#[derive(Debug, Serialize)]
struct PluginRunRequest<'a> {
    config: &'a Value,
    cursor: Option<Value>,
}

#[derive(Debug, Clone, Copy)]
struct PluginHostLimits {
    timeout_secs: u64,
    max_envelopes: usize,
    max_stdout_bytes: usize,
    max_stdout_line_bytes: usize,
    max_stderr_bytes: usize,
}

impl PluginHostLimits {
    fn from_env() -> Self {
        Self {
            timeout_secs: env_u64("COLIBRI_PLUGIN_TIMEOUT_SECS").unwrap_or(300),
            max_envelopes: env_usize("COLIBRI_PLUGIN_MAX_ENVELOPES").unwrap_or(10_000),
            max_stdout_bytes: env_usize("COLIBRI_PLUGIN_MAX_STDOUT_BYTES")
                .unwrap_or(128 * 1024 * 1024),
            max_stdout_line_bytes: env_usize("COLIBRI_PLUGIN_MAX_LINE_BYTES")
                .unwrap_or(16 * 1024 * 1024),
            max_stderr_bytes: env_usize("COLIBRI_PLUGIN_MAX_STDERR_BYTES").unwrap_or(64 * 1024),
        }
    }
}

fn env_u64(key: &str) -> Option<u64> {
    std::env::var(key).ok().and_then(|v| v.parse::<u64>().ok())
}

fn env_usize(key: &str) -> Option<usize> {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
}

pub fn load_plugin_manifest(manifest_path: &Path) -> Result<PluginManifest, ColibriError> {
    let manifest_text = std::fs::read_to_string(manifest_path)?;
    let manifest: PluginManifest = serde_json::from_str(&manifest_text)?;
    validate_manifest(&manifest)?;
    Ok(manifest)
}

fn validate_document_envelope_value(
    value: &Value,
    expected_plugin_id: &str,
) -> Result<(), ColibriError> {
    let Some(obj) = value.as_object() else {
        return Err(ColibriError::Config(
            "document_envelope must be a JSON object".into(),
        ));
    };

    for key in obj.keys() {
        if key != "schema_version" && key != "source" && key != "document" && key != "metadata" {
            return Err(ColibriError::Config(format!(
                "document_envelope has unknown field '{key}'"
            )));
        }
    }

    let schema_version = obj
        .get("schema_version")
        .and_then(Value::as_u64)
        .ok_or_else(|| ColibriError::Config("document_envelope.schema_version missing".into()))?;
    if schema_version != 1 {
        return Err(ColibriError::Config(format!(
            "Invalid envelope schema_version {schema_version} (expected 1)"
        )));
    }

    let Some(source) = obj.get("source").and_then(Value::as_object) else {
        return Err(ColibriError::Config(
            "document_envelope.source must be an object".into(),
        ));
    };
    for key in source.keys() {
        if key != "plugin_id" && key != "connector_instance" && key != "external_id" && key != "uri"
        {
            return Err(ColibriError::Config(format!(
                "document_envelope.source has unknown field '{key}'"
            )));
        }
    }
    let plugin_id = source
        .get("plugin_id")
        .and_then(Value::as_str)
        .ok_or_else(|| ColibriError::Config("document_envelope.source.plugin_id missing".into()))?;
    if plugin_id != expected_plugin_id {
        return Err(ColibriError::Config(format!(
            "Envelope plugin_id mismatch: expected '{}', got '{}'",
            expected_plugin_id, plugin_id
        )));
    }
    for key in ["connector_instance", "external_id"] {
        if !source
            .get(key)
            .is_some_and(|v| v.as_str().is_some() && !v.as_str().unwrap().trim().is_empty())
        {
            return Err(ColibriError::Config(format!(
                "document_envelope.source.{key} missing or empty"
            )));
        }
    }
    if let Some(uri) = source.get("uri") {
        if uri.is_null() || uri.as_str().is_none() {
            return Err(ColibriError::Config(
                "document_envelope.source.uri must be a string when present".into(),
            ));
        }
    }

    let Some(doc) = obj.get("document").and_then(Value::as_object) else {
        return Err(ColibriError::Config(
            "document_envelope.document must be an object".into(),
        ));
    };
    let allowed_doc = [
        "doc_id",
        "title",
        "markdown",
        "content_hash",
        "source_updated_at",
        "deleted",
    ];
    for key in doc.keys() {
        if !allowed_doc.contains(&key.as_str()) {
            return Err(ColibriError::Config(format!(
                "document_envelope.document has unknown field '{key}'"
            )));
        }
    }
    for key in ["doc_id", "title", "markdown"] {
        if !doc.get(key).is_some_and(|v| v.as_str().is_some()) {
            return Err(ColibriError::Config(format!(
                "document_envelope.document.{key} missing or not a string"
            )));
        }
        if key == "doc_id"
            && doc
                .get(key)
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim()
                .is_empty()
        {
            return Err(ColibriError::Config(
                "document_envelope.document.doc_id cannot be empty".into(),
            ));
        }
    }
    let content_hash = doc
        .get("content_hash")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            ColibriError::Config("document_envelope.document.content_hash missing".into())
        })?;
    let hash_re = CONTENT_HASH_RE.get_or_init(|| {
        Regex::new(r"^sha256:[a-f0-9]{64}$").expect("content hash regex must compile")
    });
    if !hash_re.is_match(content_hash) {
        return Err(ColibriError::Config(format!(
            "Envelope content_hash has invalid format: {content_hash}"
        )));
    }
    let updated_at = doc
        .get("source_updated_at")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            ColibriError::Config("document_envelope.document.source_updated_at missing".into())
        })?;
    if DateTime::parse_from_rfc3339(updated_at).is_err() {
        return Err(ColibriError::Config(format!(
            "Envelope source_updated_at is not RFC3339: {updated_at}"
        )));
    }
    if !doc.get("deleted").is_some_and(Value::is_boolean) {
        return Err(ColibriError::Config(
            "document_envelope.document.deleted must be boolean".into(),
        ));
    }

    let Some(meta) = obj.get("metadata").and_then(Value::as_object) else {
        return Err(ColibriError::Config(
            "document_envelope.metadata must be an object".into(),
        ));
    };
    let doc_type = meta.get("doc_type").and_then(Value::as_str).unwrap_or("");
    if doc_type.trim().is_empty() {
        return Err(ColibriError::Config(
            "Envelope metadata.doc_type cannot be empty".into(),
        ));
    }
    let classification = meta
        .get("classification")
        .and_then(Value::as_str)
        .unwrap_or("");
    match classification {
        "restricted" | "confidential" | "internal" | "public" => {}
        other => {
            return Err(ColibriError::Config(format!(
                "Envelope classification must be one of restricted/confidential/internal/public, got '{other}'"
            )))
        }
    }
    if let Some(tags) = meta.get("tags") {
        if tags.is_null() {
            return Err(ColibriError::Config(
                "document_envelope.metadata.tags cannot be null".into(),
            ));
        }
        let Some(arr) = tags.as_array() else {
            return Err(ColibriError::Config(
                "document_envelope.metadata.tags must be an array".into(),
            ));
        };
        for t in arr {
            if t.as_str().is_none() {
                return Err(ColibriError::Config(
                    "document_envelope.metadata.tags entries must be strings".into(),
                ));
            }
        }
    }
    if let Some(lang) = meta.get("language") {
        if lang.is_null() || lang.as_str().is_none() {
            return Err(ColibriError::Config(
                "document_envelope.metadata.language must be a string when present".into(),
            ));
        }
    }
    if let Some(acl) = meta.get("acl_tags") {
        if acl.is_null() {
            return Err(ColibriError::Config(
                "document_envelope.metadata.acl_tags cannot be null".into(),
            ));
        }
        let Some(arr) = acl.as_array() else {
            return Err(ColibriError::Config(
                "document_envelope.metadata.acl_tags must be an array".into(),
            ));
        };
        for t in arr {
            if t.as_str().is_none() {
                return Err(ColibriError::Config(
                    "document_envelope.metadata.acl_tags entries must be strings".into(),
                ));
            }
        }
    }

    Ok(())
}

fn validate_manifest(manifest: &PluginManifest) -> Result<(), ColibriError> {
    if manifest.schema_version != 1 {
        return Err(ColibriError::Config(format!(
            "Unsupported plugin manifest schema_version {} (expected 1)",
            manifest.schema_version
        )));
    }

    if !is_valid_plugin_id(&manifest.plugin_id) {
        return Err(ColibriError::Config(format!(
            "Invalid plugin_id '{}'. Allowed characters: [a-z0-9._-]",
            manifest.plugin_id
        )));
    }

    if manifest.version.trim().is_empty() {
        return Err(ColibriError::Config(
            "Plugin manifest version cannot be empty".into(),
        ));
    }

    if manifest.entrypoint.trim().is_empty() {
        return Err(ColibriError::Config(
            "Plugin manifest entrypoint cannot be empty".into(),
        ));
    }

    match manifest.runtime.as_str() {
        "python" | "rust" | "external" => {}
        "wasm" => {
            return Err(ColibriError::Config(
                "Plugin runtime 'wasm' is not supported yet by this host".into(),
            ));
        }
        other => {
            return Err(ColibriError::Config(format!(
                "Unsupported plugin runtime '{other}'"
            )));
        }
    }

    Ok(())
}

fn is_valid_plugin_id(raw: &str) -> bool {
    !raw.trim().is_empty()
        && raw.chars().all(|c| {
            c.is_ascii_lowercase() || c.is_ascii_digit() || c == '.' || c == '_' || c == '-'
        })
}

pub async fn run_plugin_manifest(
    manifest_path: &Path,
    config: &Value,
    cursor: Option<Value>,
) -> Result<PluginRunReport, ColibriError> {
    let manifest = load_plugin_manifest(manifest_path)?;

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

    let limits = PluginHostLimits::from_env();
    let timeout_duration = Duration::from_secs(limits.timeout_secs);

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| ColibriError::Config("Plugin stdout pipe unavailable".into()))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| ColibriError::Config("Plugin stderr pipe unavailable".into()))?;

    let plugin_id = manifest.plugin_id.clone();
    let stdout_handle = tokio::spawn(read_plugin_stdout(
        BufReader::new(stdout),
        plugin_id.clone(),
        limits,
    ));
    let stderr_handle = tokio::spawn(read_plugin_stderr(stderr, limits));

    let status = match timeout(timeout_duration, child.wait()).await {
        Ok(Ok(status)) => status,
        Ok(Err(e)) => {
            let _ = child.kill().await;
            return Err(ColibriError::Config(format!(
                "Failed waiting for plugin '{}': {e}",
                plugin_id
            )));
        }
        Err(_) => {
            let _ = child.kill().await;
            let _ = stdout_handle.await;
            let _ = stderr_handle.await;
            return Err(ColibriError::Config(format!(
                "Plugin '{}' timed out after {}s",
                plugin_id, limits.timeout_secs
            )));
        }
    };

    let stderr_out = stderr_handle
        .await
        .map_err(|e| ColibriError::Config(format!("Plugin stderr task failed: {e}")))??;
    let stdout_out = stdout_handle
        .await
        .map_err(|e| ColibriError::Config(format!("Plugin stdout task failed: {e}")))??;

    if !status.success() {
        return Err(ColibriError::Config(format!(
            "Plugin '{}' failed (exit {}): {}",
            plugin_id, status, stderr_out.stderr
        )));
    }

    Ok(PluginRunReport {
        plugin_id: manifest.plugin_id,
        runtime: manifest.runtime,
        manifest_path: manifest_path.display().to_string(),
        envelope_count: stdout_out.envelopes.len(),
        deleted_count: stdout_out.deleted_count,
        next_cursor: stdout_out.next_cursor,
        envelopes: stdout_out.envelopes,
        stderr: stderr_out.stderr,
        stderr_truncated: stderr_out.stderr_truncated,
    })
}

/// Run a plugin's configure entrypoint with inherited TTY.
///
/// The plugin receives the config file path as its first CLI argument and has
/// full terminal access (stdin/stdout/stderr inherited). On exit 0, the caller
/// reads back the file. Exit 1 means user cancelled.
#[allow(dead_code)] // Wired in Task 4 (CLI command).
pub async fn run_plugin_configure(
    manifest_path: &Path,
    config_file_path: &Path,
) -> Result<PluginConfigureResult, ColibriError> {
    let manifest = load_plugin_manifest(manifest_path)?;

    let hook = manifest.configure.ok_or_else(|| {
        ColibriError::Config(format!(
            "Plugin '{}' does not declare a configure hook",
            manifest.plugin_id
        ))
    })?;

    let manifest_dir = manifest_path.parent().unwrap_or(Path::new("."));
    let entrypoint_path = resolve_entrypoint(manifest_dir, &hook.entrypoint);

    if !entrypoint_path.exists() {
        return Err(ColibriError::Config(format!(
            "Configure entrypoint not found: {}",
            entrypoint_path.display()
        )));
    }

    let mut cmd = build_command(&manifest.runtime, &entrypoint_path)?;
    cmd.arg(config_file_path);
    cmd.stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());

    let status = cmd
        .spawn()
        .map_err(|e| {
            ColibriError::Config(format!(
                "Failed to launch configure hook for '{}': {e}",
                manifest.plugin_id
            ))
        })?
        .wait()
        .await
        .map_err(|e| {
            ColibriError::Config(format!(
                "Failed waiting for configure hook '{}': {e}",
                manifest.plugin_id
            ))
        })?;

    let exit_code = status.code().unwrap_or(-1);

    if exit_code >= 2 {
        return Err(ColibriError::Config(format!(
            "Plugin '{}' configure hook failed (exit code {exit_code})",
            manifest.plugin_id
        )));
    }

    Ok(PluginConfigureResult {
        plugin_id: manifest.plugin_id,
        exit_code,
        cancelled: exit_code == 1,
    })
}

#[derive(Debug)]
struct StdoutReadResult {
    envelopes: Vec<DocumentEnvelope>,
    next_cursor: Option<Value>,
    deleted_count: usize,
}

async fn read_plugin_stdout(
    mut reader: BufReader<impl tokio::io::AsyncRead + Unpin>,
    plugin_id: String,
    limits: PluginHostLimits,
) -> Result<StdoutReadResult, ColibriError> {
    let mut envelopes = Vec::new();
    let mut next_cursor: Option<Value> = None;
    let mut deleted_count = 0usize;

    let mut total_bytes = 0usize;
    let mut line_num = 0usize;

    loop {
        let mut buf: Vec<u8> = Vec::new();
        let n = reader.read_until(b'\n', &mut buf).await?;
        if n == 0 {
            break;
        }
        line_num += 1;
        total_bytes += n;

        if total_bytes > limits.max_stdout_bytes {
            return Err(ColibriError::Config(format!(
                "Plugin '{}' exceeded max stdout bytes ({} > {}). Increase COLIBRI_PLUGIN_MAX_STDOUT_BYTES (bytes) if this source legitimately emits large documents.",
                plugin_id, total_bytes, limits.max_stdout_bytes
            )));
        }
        if buf.len() > limits.max_stdout_line_bytes {
            return Err(ColibriError::Config(format!(
                "Plugin '{}' emitted an oversized stdout line at {} ({} > {} bytes). This usually means a single JSONL envelope (including markdown) is large. Increase COLIBRI_PLUGIN_MAX_LINE_BYTES (bytes).",
                plugin_id,
                line_num,
                buf.len(),
                limits.max_stdout_line_bytes
            )));
        }

        while buf.last() == Some(&b'\n') || buf.last() == Some(&b'\r') {
            buf.pop();
        }
        if buf.is_empty() {
            continue;
        }

        let parsed: Value = serde_json::from_slice(&buf).map_err(|e| {
            ColibriError::Config(format!(
                "Plugin '{}' emitted invalid JSON at line {}: {e}",
                plugin_id, line_num
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
                    plugin_id, line_num
                )));
            }
        }

        validate_document_envelope_value(&parsed, &plugin_id)?;

        let envelope: DocumentEnvelope = serde_json::from_value(parsed).map_err(|e| {
            ColibriError::Config(format!(
                "Plugin '{}' emitted invalid envelope at line {}: {e}",
                plugin_id, line_num
            ))
        })?;
        validate_envelope(&envelope, &plugin_id)?;
        if envelope.document.deleted {
            deleted_count += 1;
        }
        envelopes.push(envelope);

        if envelopes.len() > limits.max_envelopes {
            return Err(ColibriError::Config(format!(
                "Plugin '{}' exceeded max envelope count ({} > {})",
                plugin_id,
                envelopes.len(),
                limits.max_envelopes
            )));
        }
    }

    Ok(StdoutReadResult {
        envelopes,
        next_cursor,
        deleted_count,
    })
}

struct StderrReadResult {
    stderr: String,
    stderr_truncated: bool,
}

async fn read_plugin_stderr(
    mut stderr: impl tokio::io::AsyncRead + Unpin,
    limits: PluginHostLimits,
) -> Result<StderrReadResult, ColibriError> {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 8192];
    let mut truncated = false;

    loop {
        let n = stderr.read(&mut chunk).await?;
        if n == 0 {
            break;
        }
        if buf.len() < limits.max_stderr_bytes {
            let remaining = limits.max_stderr_bytes - buf.len();
            let take = remaining.min(n);
            buf.extend_from_slice(&chunk[..take]);
            if take < n {
                truncated = true;
            }
        } else {
            truncated = true;
        }
    }

    Ok(StderrReadResult {
        stderr: String::from_utf8_lossy(&buf).to_string(),
        stderr_truncated: truncated,
    })
}

#[cfg(test)]
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
    let hash_re = CONTENT_HASH_RE.get_or_init(|| {
        Regex::new(r"^sha256:[a-f0-9]{64}$").expect("content hash regex must compile")
    });
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
mod manifest_tests {
    use super::{is_valid_plugin_id, validate_manifest, PluginCapabilities, PluginManifest};

    fn base_manifest() -> PluginManifest {
        PluginManifest {
            schema_version: 1,
            plugin_id: "filesystem_documents".into(),
            version: "0.1.0".into(),
            runtime: "python".into(),
            entrypoint: "plugin.py".into(),
            capabilities: PluginCapabilities {
                snapshot: true,
                incremental: true,
                webhook: false,
            },
            requirements: None,
            configure: None,
        }
    }

    #[test]
    fn plugin_id_validation() {
        assert!(is_valid_plugin_id("abc"));
        assert!(is_valid_plugin_id("a_b-c.1"));
        assert!(!is_valid_plugin_id("ABC"));
        assert!(!is_valid_plugin_id("a b"));
        assert!(!is_valid_plugin_id(""));
    }

    #[test]
    fn manifest_rejects_unsupported_runtime() {
        let mut m = base_manifest();
        m.runtime = "wasm".into();
        let err = validate_manifest(&m).unwrap_err().to_string();
        assert!(err.contains("not supported"));
    }
}

#[cfg(test)]
mod tests {
    use super::{parse_plugin_stdout, read_plugin_stdout, PluginHostLimits};
    use tokio::io::{AsyncWriteExt, BufReader};

    fn envelope_line() -> String {
        r##"{"schema_version":1,"source":{"plugin_id":"filesystem_documents","connector_instance":"x","external_id":"y"},"document":{"doc_id":"d1","title":"t","markdown":"# x","content_hash":"sha256:0000000000000000000000000000000000000000000000000000000000000000","source_updated_at":"2026-02-18T00:00:00Z","deleted":false},"metadata":{"doc_type":"note","classification":"internal"}}"##.to_string()
    }

    #[test]
    fn parse_stdout_accepts_cursor_control_line() {
        let stdout = format!(
            "{}\n{}\n",
            envelope_line(),
            r#"{"type":"cursor","cursor":{"last_scan_at":"2026-02-18T00:00:00Z"}}"#
        );
        let (envelopes, cursor) =
            parse_plugin_stdout(&stdout, "filesystem_documents").expect("parse should succeed");
        assert_eq!(envelopes.len(), 1);
        assert!(cursor.is_some());
    }

    #[test]
    fn parse_stdout_rejects_cursor_without_payload() {
        let stdout = r#"{"type":"cursor"}"#;
        let err = parse_plugin_stdout(stdout, "filesystem_documents")
            .err()
            .expect("expected parse error");
        assert!(
            err.to_string().contains("without 'cursor'"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn streaming_parser_accepts_valid_envelope_and_cursor() {
        let (mut tx, rx) = tokio::io::duplex(64 * 1024);
        tx.write_all(envelope_line().as_bytes()).await.unwrap();
        tx.write_all(b"\n").await.unwrap();
        tx.write_all(br#"{"type":"cursor","cursor":{"k":"v"}}"#)
            .await
            .unwrap();
        tx.write_all(b"\n").await.unwrap();
        drop(tx);

        let limits = PluginHostLimits {
            timeout_secs: 1,
            max_envelopes: 10,
            max_stdout_bytes: 1024 * 1024,
            max_stdout_line_bytes: 1024 * 1024,
            max_stderr_bytes: 1024,
        };
        let out = read_plugin_stdout(BufReader::new(rx), "filesystem_documents".into(), limits)
            .await
            .unwrap();
        assert_eq!(out.envelopes.len(), 1);
        assert!(out.next_cursor.is_some());
    }

    #[tokio::test]
    async fn streaming_parser_rejects_schema_invalid_envelope() {
        let (mut tx, rx) = tokio::io::duplex(64 * 1024);
        // `uri` cannot be null per schema (if present).
        tx.write_all(br##"{"schema_version":1,"source":{"plugin_id":"filesystem_documents","connector_instance":"x","external_id":"y","uri":null},"document":{"doc_id":"d1","title":"t","markdown":"# x","content_hash":"sha256:0000000000000000000000000000000000000000000000000000000000000000","source_updated_at":"2026-02-18T00:00:00Z","deleted":false},"metadata":{"doc_type":"note","classification":"internal"}}"##)
            .await
            .unwrap();
        tx.write_all(b"\n").await.unwrap();
        drop(tx);

        let limits = PluginHostLimits {
            timeout_secs: 1,
            max_envelopes: 10,
            max_stdout_bytes: 1024 * 1024,
            max_stdout_line_bytes: 1024 * 1024,
            max_stderr_bytes: 1024,
        };
        let err = read_plugin_stdout(BufReader::new(rx), "filesystem_documents".into(), limits)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("uri"));
    }
}

#[cfg(test)]
mod configure_tests {
    use super::*;

    #[test]
    fn resolve_configure_entrypoint_relative() {
        let manifest_dir = Path::new("/tmp/plugins/myplugin");
        let resolved = resolve_entrypoint(manifest_dir, "configure.py");
        assert_eq!(resolved, PathBuf::from("/tmp/plugins/myplugin/configure.py"));
    }

    #[test]
    fn resolve_configure_entrypoint_absolute() {
        let manifest_dir = Path::new("/tmp/plugins/myplugin");
        let resolved = resolve_entrypoint(manifest_dir, "/usr/local/bin/configure");
        assert_eq!(resolved, PathBuf::from("/usr/local/bin/configure"));
    }
}
