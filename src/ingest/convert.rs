//! Conversion of non-markdown sources (PDF, EPUB, DOCX, PPTX) to markdown
//! via external tools.

use std::path::Path;
use std::process::Command;

/// Describes how to convert a file extension to Markdown.
enum ConversionPipeline {
    /// PDF via `docling`.
    Pdf,
    /// Generic pandoc conversion from the given source format.
    Pandoc { from_format: String },
    /// PPTX: try `markitdown` first, fall back to pandoc.
    Pptx,
}

/// Determine the conversion pipeline for a file extension.
///
/// Returns `None` for extensions with no known converter.
fn conversion_command(ext: &str) -> Option<ConversionPipeline> {
    match ext {
        ".pdf" => Some(ConversionPipeline::Pdf),
        ".epub" => Some(ConversionPipeline::Pandoc {
            from_format: "epub".into(),
        }),
        ".docx" => Some(ConversionPipeline::Pandoc {
            from_format: "docx".into(),
        }),
        ".pptx" => Some(ConversionPipeline::Pptx),
        _ => None,
    }
}

/// Convert a file to Markdown using the appropriate external tool.
pub fn convert_to_markdown(ext: &str, file_path: &Path) -> Result<String, String> {
    match conversion_command(ext) {
        Some(ConversionPipeline::Pdf) => convert_pdf(file_path),
        Some(ConversionPipeline::Pandoc { from_format }) => {
            convert_with_pandoc(file_path, &from_format)
        }
        Some(ConversionPipeline::Pptx) => convert_pptx(file_path),
        None => Err(format!("No converter for extension: {ext}")),
    }
}

/// Convert a PDF to Markdown using `docling`.
fn convert_pdf(file_path: &Path) -> Result<String, String> {
    let tmp = tempfile::tempdir().map_err(|e| format!("tempdir: {e}"))?;
    let output = Command::new("docling")
        .arg(file_path)
        .args([
            "--to",
            "md",
            "--image-export-mode",
            "placeholder",
            "--output",
        ])
        .arg(tmp.path())
        .stdout(std::process::Stdio::null())
        .output()
        .map_err(|e| format!("Failed to run docling: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("docling failed: {stderr}"));
    }

    // docling writes <stem>.md in the output directory.
    let stem = file_path.file_stem().unwrap_or_default().to_string_lossy();
    let out_md = tmp.path().join(format!("{stem}.md"));
    if !out_md.exists() {
        // docling sometimes uses different naming; find any .md file.
        let candidates: Vec<_> = std::fs::read_dir(tmp.path())
            .map_err(|e| e.to_string())?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map(|x| x == "md").unwrap_or(false))
            .collect();
        if let Some(entry) = candidates.first() {
            return std::fs::read_to_string(entry.path()).map_err(|e| e.to_string());
        }
        return Err("docling produced no .md output".into());
    }
    std::fs::read_to_string(out_md).map_err(|e| e.to_string())
}

/// Convert a file to Markdown using `pandoc`.
fn convert_with_pandoc(file_path: &Path, from_format: &str) -> Result<String, String> {
    let output = Command::new("pandoc")
        .args(["-f", from_format, "-t", "gfm", "--wrap=none"])
        .arg(file_path)
        .output()
        .map_err(|e| format!("Failed to run pandoc: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("pandoc failed: {stderr}"));
    }
    String::from_utf8(output.stdout).map_err(|e| format!("pandoc output not UTF-8: {e}"))
}

/// Convert a PPTX to Markdown, trying `markitdown` first, then pandoc.
fn convert_pptx(file_path: &Path) -> Result<String, String> {
    if which_exists("markitdown") {
        let output = Command::new("markitdown")
            .arg(file_path)
            .output()
            .map_err(|e| format!("Failed to run markitdown: {e}"))?;

        if output.status.success() {
            let text = String::from_utf8(output.stdout)
                .map_err(|e| format!("markitdown output not UTF-8: {e}"))?;
            if !text.trim().is_empty() {
                return Ok(text);
            }
        }
    }
    // Fallback to pandoc.
    convert_with_pandoc(file_path, "pptx")
}

/// Check whether an external tool is available on `$PATH`.
fn which_exists(tool: &str) -> bool {
    Command::new("which")
        .arg(tool)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Converts a source file to markdown. [`ExternalConverter`] runs the real
/// tools; tests use fakes that count calls.
pub trait Converter {
    /// Tool name recorded with a cached conversion.
    fn tool_for(&self, ext: &str) -> &'static str;
    fn convert(&self, ext: &str, path: &Path) -> Result<String, String>;
}

/// docling for PDF, pandoc for EPUB/DOCX, markitdown (or pandoc) for PPTX.
pub struct ExternalConverter;

impl Converter for ExternalConverter {
    fn tool_for(&self, ext: &str) -> &'static str {
        match conversion_command(ext) {
            Some(ConversionPipeline::Pdf) => "docling",
            Some(ConversionPipeline::Pptx) => "markitdown",
            _ => "pandoc",
        }
    }

    fn convert(&self, ext: &str, path: &Path) -> Result<String, String> {
        convert_to_markdown(ext, path)
    }
}

/// Extensions (lowercase, with dot) that need an external converter.
pub fn is_convertible(ext: &str) -> bool {
    conversion_command(ext).is_some()
}

/// SHA-256 of a file's bytes, hex encoded.
pub fn file_sha256(path: &Path) -> std::io::Result<String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 16];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Markdown for a convertible file, converting each distinct content (by
/// SHA-256) only once. Returns (markdown, tool).
///
/// The cache is content-addressed (`<conversions_dir>/<sha[..2]>/<sha>.md`)
/// and written atomically, so a cached file is reused even if the row that
/// recorded it was rolled back with an interrupted run.
pub fn convert_cached(
    store: &crate::metadata_store::MetadataStore,
    conversions_dir: &Path,
    converter: &dyn Converter,
    ext: &str,
    path: &Path,
    sha256: &str,
) -> Result<(String, String), String> {
    let rel = format!("{}/{sha256}.md", &sha256[..2.min(sha256.len())]);
    let abs = conversions_dir.join(&rel);
    if let Ok(markdown) = std::fs::read_to_string(&abs) {
        let tool = store
            .get_conversion(sha256)
            .ok()
            .flatten()
            .map(|(tool, _)| tool)
            .unwrap_or_else(|| converter.tool_for(ext).to_string());
        return Ok((markdown, tool));
    }
    let markdown = converter.convert(ext, path)?;
    let tool = converter.tool_for(ext);
    let dir = abs.parent().unwrap_or(conversions_dir);
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let tmp = dir.join(format!("{sha256}.md.tmp"));
    std::fs::write(&tmp, &markdown).map_err(|e| e.to_string())?;
    std::fs::rename(&tmp, &abs).map_err(|e| e.to_string())?;
    store
        .put_conversion(sha256, tool, &rel)
        .map_err(|e| e.to_string())?;
    Ok((markdown, tool.to_string()))
}

#[cfg(test)]
pub(crate) mod fakes {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Returns `# converted <file name>` plus the file's text; counts calls.
    #[derive(Default)]
    pub(crate) struct CountingConverter {
        pub calls: AtomicUsize,
    }

    impl CountingConverter {
        pub(crate) fn calls(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
    }

    impl Converter for CountingConverter {
        fn tool_for(&self, _ext: &str) -> &'static str {
            "fake"
        }

        fn convert(&self, _ext: &str, path: &Path) -> Result<String, String> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let text = std::fs::read_to_string(path).unwrap_or_default();
            Ok(format!(
                "# converted {}\n\n{text}",
                path.file_name().unwrap_or_default().to_string_lossy()
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::fakes::CountingConverter;
    use super::*;

    #[test]
    fn identical_bytes_are_converted_once() {
        let dir = tempfile::TempDir::new().unwrap();
        let store =
            crate::metadata_store::MetadataStore::open_rw(&dir.path().join("m.db")).unwrap();
        let conv_dir = dir.path().join("conversions");
        let a = dir.path().join("a.epub");
        let b = dir.path().join("copy-of-a.epub");
        std::fs::write(&a, "same bytes").unwrap();
        std::fs::write(&b, "same bytes").unwrap();
        let fake = CountingConverter::default();

        let sha_a = file_sha256(&a).unwrap();
        let sha_b = file_sha256(&b).unwrap();
        assert_eq!(sha_a, sha_b);
        let (md, tool) = convert_cached(&store, &conv_dir, &fake, ".epub", &a, &sha_a).unwrap();
        assert!(md.contains("same bytes"));
        assert_eq!(tool, "fake");
        convert_cached(&store, &conv_dir, &fake, ".epub", &b, &sha_b).unwrap();
        assert_eq!(fake.calls(), 1);
    }

    #[test]
    fn cached_file_is_reused_even_without_its_row() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("m.db");
        let conv_dir = dir.path().join("conversions");
        let src = dir.path().join("a.pdf");
        std::fs::write(&src, "pdf bytes").unwrap();
        let sha = file_sha256(&src).unwrap();
        let fake = CountingConverter::default();
        {
            let store = crate::metadata_store::MetadataStore::open_rw(&path).unwrap();
            let tx = store.begin().unwrap();
            convert_cached(&store, &conv_dir, &fake, ".pdf", &src, &sha).unwrap();
            drop(tx); // rolled back: the conversions row is gone
            assert_eq!(store.get_conversion(&sha).unwrap(), None);
            convert_cached(&store, &conv_dir, &fake, ".pdf", &src, &sha).unwrap();
        }
        assert_eq!(fake.calls(), 1);
    }

    #[test]
    fn test_convert_command_for_pdf() {
        assert!(matches!(
            conversion_command(".pdf"),
            Some(ConversionPipeline::Pdf)
        ));
    }

    #[test]
    fn test_convert_command_for_docx() {
        assert!(matches!(
            conversion_command(".docx"),
            Some(ConversionPipeline::Pandoc { .. })
        ));
    }

    #[test]
    fn test_convert_command_for_pptx() {
        assert!(matches!(
            conversion_command(".pptx"),
            Some(ConversionPipeline::Pptx)
        ));
    }

    #[test]
    fn test_convert_command_for_epub() {
        assert!(matches!(
            conversion_command(".epub"),
            Some(ConversionPipeline::Pandoc { .. })
        ));
    }

    #[test]
    fn test_convert_command_for_unsupported() {
        assert!(conversion_command(".csv").is_none());
    }

    // -- PlantUML enrichment tests -------------------------------------------
}
