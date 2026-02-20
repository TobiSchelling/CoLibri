//! `colibri import` — import PDF/EPUB files as markdown.

use std::path::{Path, PathBuf};
use std::process::Stdio;

use anyhow::{bail, Context};
use clap::ValueEnum;
use sha2::{Digest, Sha256};
use tokio::process::Command;

use crate::canonical_store::ingest_envelopes;
use crate::config::load_config;
use crate::indexer::index_library;
use crate::plugin_host::{DocumentEnvelope, EnvelopeDocument, EnvelopeMetadata, EnvelopeSource};

const IMPORT_PLUGIN_ID: &str = "cli_import";

/// PDF converter choice.
#[derive(Debug, Clone, Copy, Default, ValueEnum)]
pub enum PdfConverter {
    /// Docling — best quality for tables, equations, technical content
    #[default]
    Docling,
    /// Marker — faster, good for bulk imports
    Marker,
}

/// Image export mode for PDF conversion.
#[derive(Debug, Clone, Copy, Default, ValueEnum)]
pub enum ImageMode {
    /// Only mark image positions, no image data (smallest files)
    #[default]
    Placeholder,
    /// Export images as separate files and reference them
    Referenced,
    /// Embed images as base64 (largest files)
    Embedded,
}

impl ImageMode {
    /// Convert to docling CLI argument value.
    fn as_docling_arg(&self) -> &'static str {
        match self {
            ImageMode::Placeholder => "placeholder",
            ImageMode::Referenced => "referenced",
            ImageMode::Embedded => "embedded",
        }
    }
}

/// Supported input formats.
#[derive(Debug, Clone, Copy)]
enum Format {
    Pdf,
    Epub,
}

/// Detect format from file extension.
fn detect_format(path: &Path) -> anyhow::Result<Format> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase());

    match ext.as_deref() {
        Some("pdf") => Ok(Format::Pdf),
        Some("epub") => Ok(Format::Epub),
        Some(other) => bail!("Unsupported format '.{other}'. Supported: .pdf, .epub"),
        None => bail!("File has no extension. Supported: .pdf, .epub"),
    }
}

fn sha256_hex(input: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn sha256_content_hash(markdown: &str) -> String {
    format!("sha256:{}", sha256_hex(markdown))
}

fn title_from_path(path: &Path) -> String {
    path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("Imported Document")
        .to_string()
}

fn unique_work_dir(prefix: &str) -> PathBuf {
    let n = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    std::env::temp_dir().join(format!("{prefix}-{n}"))
}

/// Check if a tool is available on PATH.
async fn check_tool_available(tool: &str) -> anyhow::Result<()> {
    let status = Command::new("which")
        .arg(tool)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .await
        .context("Failed to run 'which' command")?;

    if !status.success() {
        let install_hint = match tool {
            "docling" => "pip install docling",
            "marker_single" => "pip install marker-pdf",
            "pandoc" => "brew install pandoc",
            _ => "check documentation for install instructions",
        };
        bail!("{tool} not found. Install with: {install_hint}");
    }
    Ok(())
}

/// Convert PDF using docling.
async fn convert_pdf_docling(
    input: &Path,
    out_dir: &Path,
    image_mode: ImageMode,
    _attachments_dir: Option<&Path>,
) -> anyhow::Result<PathBuf> {
    check_tool_available("docling").await?;

    eprintln!("Converting PDF with docling (this may take a while)...");

    let output = Command::new("docling")
        .arg(input)
        .arg("--to")
        .arg("md")
        .arg("--image-export-mode")
        .arg(image_mode.as_docling_arg())
        .arg("--output")
        .arg(out_dir)
        .output()
        .await
        .context("Failed to run docling")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("docling failed (exit {}): {stderr}", output.status);
    }

    // Docling outputs to <out_dir>/<stem>.md
    let stem = input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");
    let output_file = out_dir.join(format!("{stem}.md"));

    if !output_file.exists() {
        bail!("Expected output file not found: {}", output_file.display());
    }

    Ok(output_file)
}

/// Convert PDF using marker.
async fn convert_pdf_marker(input: &Path, out_dir: &Path) -> anyhow::Result<PathBuf> {
    check_tool_available("marker_single").await?;

    eprintln!("Converting PDF with marker...");

    // marker_single outputs to current directory by default
    // We run it and then move the output
    let output = Command::new("marker_single")
        .arg(input)
        .arg("--output_dir")
        .arg(out_dir)
        .output()
        .await
        .context("Failed to run marker_single")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("marker_single failed (exit {}): {stderr}", output.status);
    }

    // Marker outputs to <out_dir>/<stem>/<stem>.md
    let stem = input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");
    let marker_output = out_dir.join(stem).join(format!("{stem}.md"));

    if !marker_output.exists() {
        bail!(
            "Expected output file not found: {}",
            marker_output.display()
        );
    }

    // Move to out_dir root for consistency
    let final_path = out_dir.join(format!("{stem}.md"));
    std::fs::rename(&marker_output, &final_path).context("Failed to move marker output")?;

    // Clean up the subdirectory marker created
    let subdir = out_dir.join(stem);
    if subdir.is_dir() {
        let _ = std::fs::remove_dir_all(&subdir);
    }

    Ok(final_path)
}

/// Convert EPUB using pandoc.
async fn convert_epub(
    input: &Path,
    out_dir: &Path,
    attachments_dir: Option<&Path>,
) -> anyhow::Result<PathBuf> {
    check_tool_available("pandoc").await?;

    eprintln!("Converting EPUB with pandoc...");

    let stem = input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");
    let output_file = out_dir.join(format!("{stem}.md"));

    // Build pandoc command
    let mut cmd = Command::new("pandoc");
    cmd.arg("-f").arg("epub").arg("-t").arg("markdown");

    // If attachments dir specified, extract media there
    if let Some(attach_dir) = attachments_dir {
        std::fs::create_dir_all(attach_dir)?;
        cmd.arg("--extract-media").arg(attach_dir);
    }

    cmd.arg(input).arg("-o").arg(&output_file);

    let output = cmd.output().await.context("Failed to run pandoc")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("pandoc failed (exit {}): {stderr}", output.status);
    }

    if !output_file.exists() {
        bail!("Expected output file not found: {}", output_file.display());
    }

    Ok(output_file)
}

/// Run the import command.
pub async fn run(
    input: PathBuf,
    output_dir: Option<PathBuf>,
    converter: PdfConverter,
    image_mode: ImageMode,
    attachments_dir: Option<PathBuf>,
    reindex: bool,
) -> anyhow::Result<()> {
    // Validate input file
    if !input.exists() {
        bail!("File not found: {}", input.display());
    }

    let input = input
        .canonicalize()
        .context("Failed to resolve input path")?;

    // Detect format
    let format = detect_format(&input)?;

    if matches!(image_mode, ImageMode::Referenced) || attachments_dir.is_some() {
        bail!(
            "Image attachments are not supported by `colibri import` (canonical store is markdown-only). Use --image-mode placeholder or embedded."
        );
    }

    // Determine conversion working directory (used only for intermediate tool output).
    let (out_dir, cleanup_out_dir) = if let Some(dir) = output_dir {
        (dir, false)
    } else {
        (unique_work_dir("colibri-import"), true)
    };

    // Ensure output directory exists
    std::fs::create_dir_all(&out_dir).context("Failed to create output directory")?;

    // Run conversion
    let output_file = match format {
        Format::Pdf => match converter {
            PdfConverter::Docling => {
                convert_pdf_docling(&input, &out_dir, image_mode, None).await?
            }
            PdfConverter::Marker => convert_pdf_marker(&input, &out_dir).await?,
        },
        Format::Epub => convert_epub(&input, &out_dir, None).await?,
    };

    let markdown = std::fs::read_to_string(&output_file).with_context(|| {
        format!(
            "Failed to read converted markdown {}",
            output_file.display()
        )
    })?;

    let config = load_config()?;

    let doc_id = format!(
        "{}:{}",
        IMPORT_PLUGIN_ID,
        sha256_hex(input.to_string_lossy().as_ref())
    );

    let envelope = DocumentEnvelope {
        schema_version: 1,
        source: EnvelopeSource {
            plugin_id: IMPORT_PLUGIN_ID.into(),
            connector_instance: "import".into(),
            external_id: input.to_string_lossy().to_string(),
            uri: Some(format!("file://{}", input.to_string_lossy())),
        },
        document: EnvelopeDocument {
            doc_id,
            title: title_from_path(&input),
            markdown: markdown.clone(),
            content_hash: sha256_content_hash(&markdown),
            source_updated_at: chrono::Utc::now().to_rfc3339(),
            deleted: false,
        },
        metadata: EnvelopeMetadata {
            doc_type: "book".into(),
            classification: "internal".into(),
            tags: None,
            language: None,
            acl_tags: None,
        },
    };

    let report = ingest_envelopes(&config, &[envelope], false)?;
    eprintln!(
        "✓ Ingested into canonical store: written={} unchanged={} tombstoned={}",
        report.written, report.unchanged, report.tombstoned
    );

    // Re-index if requested
    if reindex {
        eprintln!("Indexing canonical store...");
        let result = index_library(&config, false, |_| {}).await?;

        if result.errors > 0 {
            eprintln!("Warning: {} indexing errors occurred", result.errors);
        }
    }

    if cleanup_out_dir {
        let _ = std::fs::remove_dir_all(&out_dir);
    }

    Ok(())
}
