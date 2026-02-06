//! `colibri import` — import PDF/EPUB files as markdown.

use std::path::{Path, PathBuf};
use std::process::Stdio;

use anyhow::{bail, Context};
use clap::ValueEnum;
use tokio::process::Command;

use crate::config::load_config;
use crate::indexer::index_library;

/// PDF converter choice.
#[derive(Debug, Clone, Copy, Default, ValueEnum)]
pub enum PdfConverter {
    /// Docling — best quality for tables, equations, technical content
    #[default]
    Docling,
    /// Marker — faster, good for bulk imports
    Marker,
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
async fn convert_pdf_docling(input: &Path, out_dir: &Path) -> anyhow::Result<PathBuf> {
    check_tool_available("docling").await?;

    eprintln!("Converting PDF with docling (this may take a while)...");

    let output = Command::new("docling")
        .arg(input)
        .arg("--to")
        .arg("md")
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
        bail!(
            "Expected output file not found: {}",
            output_file.display()
        );
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
async fn convert_epub(input: &Path, out_dir: &Path) -> anyhow::Result<PathBuf> {
    check_tool_available("pandoc").await?;

    eprintln!("Converting EPUB with pandoc...");

    let stem = input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");
    let output_file = out_dir.join(format!("{stem}.md"));

    let output = Command::new("pandoc")
        .arg("-f")
        .arg("epub")
        .arg("-t")
        .arg("markdown")
        .arg(input)
        .arg("-o")
        .arg(&output_file)
        .output()
        .await
        .context("Failed to run pandoc")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("pandoc failed (exit {}): {stderr}", output.status);
    }

    if !output_file.exists() {
        bail!(
            "Expected output file not found: {}",
            output_file.display()
        );
    }

    Ok(output_file)
}

/// Run the import command.
pub async fn run(
    input: PathBuf,
    output_dir: Option<PathBuf>,
    converter: PdfConverter,
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

    // Determine output directory
    let out_dir = if let Some(dir) = output_dir {
        dir
    } else {
        let config = load_config()?;
        if let Some(books) = config.books_source() {
            PathBuf::from(&books.path)
        } else {
            bail!(
                "No books source in config. Use --output-dir or add a source with doc_type: book"
            );
        }
    };

    // Ensure output directory exists
    std::fs::create_dir_all(&out_dir).context("Failed to create output directory")?;

    // Run conversion
    let output_file = match format {
        Format::Pdf => match converter {
            PdfConverter::Docling => convert_pdf_docling(&input, &out_dir).await?,
            PdfConverter::Marker => convert_pdf_marker(&input, &out_dir).await?,
        },
        Format::Epub => convert_epub(&input, &out_dir).await?,
    };

    eprintln!("✓ Imported: {}", output_file.display());

    // Re-index if requested
    if reindex {
        eprintln!("Re-indexing books folder...");
        let config = load_config()?;

        // Find the books source name for targeted indexing
        let folder_filter = config.books_source().map(|s| s.display_name().to_string());

        let result = index_library(&config, folder_filter.as_deref(), false).await?;

        if result.errors > 0 {
            eprintln!("Warning: {} indexing errors occurred", result.errors);
        }
    }

    Ok(())
}
