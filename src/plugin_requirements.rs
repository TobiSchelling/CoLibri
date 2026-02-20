use serde_json::Value;
use std::collections::BTreeSet;

fn config_string_array(config: &Value, key: &str) -> Option<Vec<String>> {
    let arr = config.get(key)?.as_array()?;
    let mut out: Vec<String> = Vec::new();
    for v in arr {
        let Some(s) = v.as_str() else { continue };
        let s = s.trim();
        if s.is_empty() {
            continue;
        }
        out.push(s.to_string());
    }
    Some(out)
}

/// Returns a set of tool checks that are relevant for this job config.
///
/// `None` means "unknown / don't filter" (use manifest requirements as-is).
pub fn tool_checks_relevant_for_job(
    plugin_id: &str,
    job_config: &Value,
) -> Option<BTreeSet<String>> {
    match plugin_id {
        "filesystem_documents" => filesystem_documents_tool_checks(job_config),
        _ => None,
    }
}

fn filesystem_documents_tool_checks(job_config: &Value) -> Option<BTreeSet<String>> {
    let include_exts = config_string_array(job_config, "include_extensions")?;
    let exts: BTreeSet<String> = include_exts.into_iter().map(|s| s.to_lowercase()).collect();

    let markdown_only = exts.iter().all(|e| e == ".md" || e == ".markdown");
    if markdown_only {
        return Some(BTreeSet::new());
    }

    let mut tools: BTreeSet<String> = BTreeSet::new();

    if exts.contains(".epub") || exts.contains(".docx") {
        tools.insert("pandoc".into());
    }
    if exts.contains(".pdf") {
        tools.insert("docling".into());
        tools.insert("pdftotext".into());
    }
    if exts.contains(".pptx") {
        let backend = job_config
            .get("pptx_backend")
            .and_then(Value::as_str)
            .unwrap_or("soffice_pdf_docling");
        match backend {
            "soffice_pdf_docling" => {
                tools.insert("soffice".into());
                tools.insert("docling".into());
            }
            "pandoc" => {
                tools.insert("pandoc".into());
            }
            // `python_pptx` and `markitdown` are Python-module based; they don't map to CLI tools.
            _ => {}
        }
    }

    Some(tools)
}
