//! Conversion of non-markdown sources (PDF, EPUB, DOCX, PPTX) to markdown
//! via external tools.

use std::io::Read;
use std::path::Path;
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

/// docling needs about 0.4 s per page; 20 minutes covers ~2,500 pages and
/// stops a hung run (seen in practice: a deadlocked docling at 0% CPU).
const DOCLING_TIMEOUT: Duration = Duration::from_secs(20 * 60);
/// pandoc and markitdown finish in seconds.
const TEXT_TOOL_TIMEOUT: Duration = Duration::from_secs(5 * 60);

/// Run a command, killing it after `timeout`. Output pipes are drained on
/// threads so a chatty tool cannot block on a full pipe.
fn run_with_timeout(mut cmd: Command, timeout: Duration) -> Result<Output, String> {
    let name = cmd.get_program().to_string_lossy().to_string();
    let mut child = cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .stdin(Stdio::null())
        .spawn()
        .map_err(|e| format!("Failed to run {name}: {e}"))?;
    let drain = |pipe: Option<Box<dyn Read + Send>>| {
        std::thread::spawn(move || {
            let mut buf = Vec::new();
            if let Some(mut p) = pipe {
                let _ = p.read_to_end(&mut buf);
            }
            buf
        })
    };
    let stdout = drain(
        child
            .stdout
            .take()
            .map(|p| Box::new(p) as Box<dyn Read + Send>),
    );
    let stderr = drain(
        child
            .stderr
            .take()
            .map(|p| Box::new(p) as Box<dyn Read + Send>),
    );
    let deadline = Instant::now() + timeout;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if Instant::now() >= deadline => {
                let _ = child.kill();
                let _ = child.wait();
                // Reader threads are left to finish on their own: a helper
                // process of the tool may still hold the pipes open.
                return Err(format!(
                    "{name} did not finish within {} minutes and was stopped",
                    timeout.as_secs() / 60
                ));
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(200)),
            Err(e) => return Err(format!("{name}: {e}")),
        }
    };
    Ok(Output {
        status,
        stdout: stdout.join().unwrap_or_default(),
        stderr: stderr.join().unwrap_or_default(),
    })
}

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
    let mut cmd = Command::new("docling");
    cmd.arg(file_path)
        .args([
            "--to",
            "md",
            "--image-export-mode",
            "placeholder",
            "--output",
        ])
        .arg(tmp.path());
    let output = run_with_timeout(cmd, DOCLING_TIMEOUT)?;

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
    let mut cmd = Command::new("pandoc");
    cmd.args(["-f", from_format, "-t", "gfm", "--wrap=none"])
        .arg(file_path);
    let output = run_with_timeout(cmd, TEXT_TOOL_TIMEOUT)?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("pandoc failed: {stderr}"));
    }
    String::from_utf8(output.stdout).map_err(|e| format!("pandoc output not UTF-8: {e}"))
}

/// Convert a PPTX to Markdown, trying `markitdown` first, then pandoc.
fn convert_pptx(file_path: &Path) -> Result<String, String> {
    if which_exists("markitdown") {
        let mut cmd = Command::new("markitdown");
        cmd.arg(file_path);
        let output = run_with_timeout(cmd, TEXT_TOOL_TIMEOUT)?;

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
    /// `Err` when the tool for `ext` is not installed. Such failures are not
    /// remembered, so installing the tool is enough to make the next run work.
    fn check_available(&self, _ext: &str) -> Result<(), String> {
        Ok(())
    }
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

    fn check_available(&self, ext: &str) -> Result<(), String> {
        let ok = match conversion_command(ext) {
            Some(ConversionPipeline::Pdf) => which_exists("docling"),
            Some(ConversionPipeline::Pptx) => which_exists("markitdown") || which_exists("pandoc"),
            Some(ConversionPipeline::Pandoc { .. }) => which_exists("pandoc"),
            None => false,
        };
        if ok {
            Ok(())
        } else {
            Err(format!(
                "{} is not installed (needed for {ext})",
                self.tool_for(ext)
            ))
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

fn cache_paths(conversions_dir: &Path, sha256: &str) -> (String, std::path::PathBuf) {
    let rel = format!("{}/{sha256}.md", &sha256[..2.min(sha256.len())]);
    let abs = conversions_dir.join(&rel);
    (rel, abs)
}

/// Forget the cached result (or failure) for these bytes, e.g. before `--reconvert`.
pub fn clear_cached(
    store: &crate::metadata_store::MetadataStore,
    conversions_dir: &Path,
    sha256: &str,
) -> Result<(), String> {
    let (_, abs) = cache_paths(conversions_dir, sha256);
    let _ = std::fs::remove_file(abs);
    store.delete_conversion(sha256).map_err(|e| e.to_string())
}

/// Markdown for a convertible file, converting each distinct content (by
/// SHA-256) only once. Returns (markdown, tool).
///
/// The cache is content-addressed (`<conversions_dir>/<sha[..2]>/<sha>.md`)
/// and written atomically, so a cached file is reused even if the row that
/// recorded it was rolled back with an interrupted run. A failed conversion
/// is remembered and not retried unless `retry_failed` is set.
pub fn convert_cached(
    store: &crate::metadata_store::MetadataStore,
    conversions_dir: &Path,
    converter: &dyn Converter,
    ext: &str,
    path: &Path,
    sha256: &str,
    retry_failed: bool,
) -> Result<(String, String), String> {
    let (rel, abs) = cache_paths(conversions_dir, sha256);
    let row = store.get_conversion(sha256).map_err(|e| e.to_string())?;
    if let Ok(markdown) = std::fs::read_to_string(&abs) {
        let tool = row
            .map(|r| r.converter)
            .unwrap_or_else(|| converter.tool_for(ext).to_string());
        return Ok((markdown, tool));
    }
    if let Some(error) = row.and_then(|r| r.error).filter(|_| !retry_failed) {
        return Err(format!(
            "conversion failed before: {error} (retry with --retry-failed)"
        ));
    }
    converter.check_available(ext)?;
    let tool = converter.tool_for(ext);
    let markdown = match converter.convert(ext, path) {
        Ok(markdown) => markdown,
        Err(error) => {
            store
                .put_conversion(sha256, tool, "", Some(&error))
                .map_err(|e| e.to_string())?;
            return Err(error);
        }
    };
    let dir = abs.parent().unwrap_or(conversions_dir);
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let tmp = dir.join(format!("{sha256}.md.tmp"));
    std::fs::write(&tmp, &markdown).map_err(|e| e.to_string())?;
    std::fs::rename(&tmp, &abs).map_err(|e| e.to_string())?;
    store
        .put_conversion(sha256, tool, &rel, None)
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
        /// Fail for files whose name contains this text.
        pub fail_on: Option<&'static str>,
        /// Return this markdown regardless of the input.
        pub fixed_output: Option<&'static str>,
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
            let name = path.file_name().unwrap_or_default().to_string_lossy();
            if self.fail_on.is_some_and(|f| name.contains(f)) {
                return Err(format!("cannot convert {name}"));
            }
            if let Some(fixed) = self.fixed_output {
                return Ok(fixed.to_string());
            }
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
    fn hung_tools_are_stopped_at_the_timeout() {
        let started = Instant::now();
        let mut cmd = Command::new("sleep");
        cmd.arg("30");
        let err = run_with_timeout(cmd, Duration::from_millis(500)).unwrap_err();
        assert!(err.contains("did not finish"), "{err}");
        assert!(started.elapsed() < Duration::from_secs(5));

        let mut cmd = Command::new("sh");
        cmd.args(["-c", "yes colibri | head -c 200000; echo done >&2"]);
        let out = run_with_timeout(cmd, Duration::from_secs(10)).unwrap();
        assert!(out.status.success());
        assert_eq!(out.stdout.len(), 200_000, "large output is drained");
        assert_eq!(String::from_utf8_lossy(&out.stderr).trim(), "done");
    }

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
        let (md, tool) =
            convert_cached(&store, &conv_dir, &fake, ".epub", &a, &sha_a, false).unwrap();
        assert!(md.contains("same bytes"));
        assert_eq!(tool, "fake");
        convert_cached(&store, &conv_dir, &fake, ".epub", &b, &sha_b, false).unwrap();
        assert_eq!(fake.calls(), 1);
    }

    #[test]
    fn failures_are_remembered_until_retry() {
        let dir = tempfile::TempDir::new().unwrap();
        let store =
            crate::metadata_store::MetadataStore::open_rw(&dir.path().join("m.db")).unwrap();
        let conv_dir = dir.path().join("conversions");
        let src = dir.path().join("broken.pdf");
        std::fs::write(&src, "garbage").unwrap();
        let sha = file_sha256(&src).unwrap();
        let fake = CountingConverter {
            fail_on: Some("broken"),
            ..Default::default()
        };
        for _ in 0..3 {
            assert!(convert_cached(&store, &conv_dir, &fake, ".pdf", &src, &sha, false).is_err());
        }
        assert_eq!(fake.calls(), 1, "a failure is not retried");
        let err = convert_cached(&store, &conv_dir, &fake, ".pdf", &src, &sha, true).unwrap_err();
        assert!(err.contains("cannot convert"));
        assert_eq!(fake.calls(), 2);

        clear_cached(&store, &conv_dir, &sha).unwrap();
        assert!(convert_cached(&store, &conv_dir, &fake, ".pdf", &src, &sha, false).is_err());
        assert_eq!(fake.calls(), 3);
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
            convert_cached(&store, &conv_dir, &fake, ".pdf", &src, &sha, false).unwrap();
            drop(tx); // rolled back: the conversions row is gone
            assert_eq!(store.get_conversion(&sha).unwrap(), None);
            convert_cached(&store, &conv_dir, &fake, ".pdf", &src, &sha, false).unwrap();
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
