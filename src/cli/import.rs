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

/// Move images from docling output to attachments folder and update markdown references.
fn relocate_images(
    md_path: &Path,
    stem: &str,
    out_dir: &Path,
    attachments_dir: &Path,
) -> anyhow::Result<()> {
    // Docling creates images in <out_dir>/<stem>/ folder
    let images_source_dir = out_dir.join(stem);

    if !images_source_dir.exists() || !images_source_dir.is_dir() {
        return Ok(()); // No images exported
    }

    // Ensure attachments directory exists
    std::fs::create_dir_all(attachments_dir).context("Failed to create attachments directory")?;

    // Read the markdown content
    let md_content =
        std::fs::read_to_string(md_path).context("Failed to read markdown for image relocation")?;

    let mut updated_content = md_content.clone();

    // Find and move all image files
    for entry in std::fs::read_dir(&images_source_dir)? {
        let entry = entry?;
        let file_path = entry.path();

        if file_path.is_file() {
            if let Some(ext) = file_path.extension().and_then(|e| e.to_str()) {
                if ["png", "jpg", "jpeg", "gif", "webp"].contains(&ext.to_lowercase().as_str()) {
                    let file_name = file_path.file_name().unwrap();
                    let dest_path = attachments_dir.join(file_name);

                    // Move the image
                    std::fs::rename(&file_path, &dest_path).with_context(|| {
                        format!("Failed to move image to {}", dest_path.display())
                    })?;

                    // Update references in markdown
                    // Docling uses relative paths like: ./stem/image.png
                    let old_ref = format!("./{}/{}", stem, file_name.to_string_lossy());

                    // Calculate relative path from md_path to attachments_dir
                    let md_dir = md_path.parent().unwrap_or(Path::new("."));
                    let new_ref = if let Ok(rel) = attachments_dir.strip_prefix(md_dir) {
                        format!("./{}/{}", rel.display(), file_name.to_string_lossy())
                    } else {
                        // Use absolute path if can't make relative
                        dest_path.to_string_lossy().to_string()
                    };

                    updated_content = updated_content.replace(&old_ref, &new_ref);
                }
            }
        }
    }

    // Write updated markdown if changed
    if updated_content != md_content {
        std::fs::write(md_path, updated_content).context("Failed to update markdown references")?;
    }

    // Clean up empty source directory
    let _ = std::fs::remove_dir(&images_source_dir);

    Ok(())
}

/// Convert PDF using docling.
async fn convert_pdf_docling(
    input: &Path,
    out_dir: &Path,
    image_mode: ImageMode,
    attachments_dir: Option<&Path>,
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
        bail!(
            "Expected output file not found: {}",
            output_file.display()
        );
    }

    // If referenced mode with attachments dir, relocate images
    if matches!(image_mode, ImageMode::Referenced) {
        if let Some(attach_dir) = attachments_dir {
            relocate_images(&output_file, stem, out_dir, attach_dir)?;
        }
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
            PdfConverter::Docling => {
                convert_pdf_docling(&input, &out_dir, image_mode, attachments_dir.as_deref())
                    .await?
            }
            PdfConverter::Marker => convert_pdf_marker(&input, &out_dir).await?,
        },
        Format::Epub => convert_epub(&input, &out_dir, attachments_dir.as_deref()).await?,
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
