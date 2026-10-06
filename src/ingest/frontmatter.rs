//! YAML frontmatter parsing for markdown sources.

use std::collections::BTreeSet;

/// Parse a leading YAML frontmatter block from a Markdown source.
///
/// Returns `(tags, frontmatter_map, body)`:
/// - `tags`: extracted from a `tags:` sequence in the frontmatter, normalized
///   (trimmed, deduplicated, leading `#` stripped, empties dropped). `None`
///   if no `tags:` key.
/// - `frontmatter_map`: the entire top-level mapping serialised as
///   `serde_json::Map`. `None` if no frontmatter.
/// - `body`: the markdown content with the frontmatter stripped.
///
/// Behaviour for non-frontmatter or malformed input:
/// - File doesn't start with `---\n`: returns `(None, None, full_text)`.
/// - Closing `\n---` not found: log warning, return `(None, None, full_text)`.
/// - Invalid YAML: log warning, return `(None, None, full_text)`.
/// - `tags` value is a scalar (not a list): log warning at trace level,
///   return `tags = None` but the rest of the frontmatter is still parsed.
pub type FrontmatterParseResult<'a> = (
    Option<Vec<String>>,
    Option<serde_json::Map<String, serde_json::Value>>,
    &'a str,
);

pub fn parse_frontmatter<'a>(text: &'a str, rel_path: &str) -> FrontmatterParseResult<'a> {
    // Skip the leading marker line — try CRLF first, then LF; bail if neither.
    let after_marker = if let Some(stripped) = text.strip_prefix("---\r\n") {
        stripped
    } else if let Some(stripped) = text.strip_prefix("---\n") {
        stripped
    } else {
        return (None, None, text);
    };

    // Find closing marker — accepts "\n---\n", "\n---\r\n", or "\n---" at EOF
    let (yaml_block, body) = if let Some(idx) = after_marker.find("\n---\n") {
        (&after_marker[..idx], &after_marker[idx + 5..])
    } else if let Some(idx) = after_marker.find("\n---\r\n") {
        (&after_marker[..idx], &after_marker[idx + 6..])
    } else if let Some(stripped) = after_marker.strip_suffix("\n---") {
        (stripped, "")
    } else {
        eprintln!("[frontmatter] {rel_path}: no closing --- found; treating as body-only");
        return (None, None, text);
    };

    let parsed: serde_yaml::Value = match serde_yaml::from_str(yaml_block) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[frontmatter] {rel_path}: parse error: {e}; treating as body-only");
            return (None, None, text);
        }
    };

    // Extract & normalise tags
    let tags = parsed
        .get("tags")
        .and_then(|t| t.as_sequence())
        .map(|seq| {
            let mut seen = BTreeSet::new();
            for v in seq {
                if let Some(s) = v.as_str() {
                    let normalised = s.trim().trim_start_matches('#').to_string();
                    if !normalised.is_empty() {
                        seen.insert(normalised);
                    }
                }
            }
            seen.into_iter().collect::<Vec<String>>()
        })
        .filter(|v: &Vec<String>| !v.is_empty());

    // Convert top-level mapping to a serde_json::Map for portability
    let frontmatter_map = match serde_json::to_value(&parsed) {
        Ok(serde_json::Value::Object(map)) => Some(map),
        _ => None,
    };

    (tags, frontmatter_map, body)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ----- Frontmatter parsing tests -----

    #[test]
    fn parse_frontmatter_extracts_tags_and_body() {
        let text = "---\ntags:\n  - foo\n  - bar\nDocumentType: meeting\n---\n# Body\nContent";
        let (tags, fm, body) = parse_frontmatter(text, "test.md");
        assert_eq!(tags.unwrap(), vec!["bar".to_string(), "foo".to_string()]); // BTreeSet sorts
        let fm = fm.unwrap();
        assert_eq!(fm.get("DocumentType").unwrap(), "meeting");
        assert_eq!(body, "# Body\nContent");
    }

    #[test]
    fn parse_frontmatter_no_frontmatter_returns_full_body() {
        let text = "# Just a body\nNo frontmatter here";
        let (tags, fm, body) = parse_frontmatter(text, "test.md");
        assert!(tags.is_none());
        assert!(fm.is_none());
        assert_eq!(body, text);
    }

    #[test]
    fn parse_frontmatter_malformed_yaml_returns_full_body() {
        let text = "---\ninvalid: yaml: : bad\n---\n# Body";
        let (tags, fm, body) = parse_frontmatter(text, "test.md");
        assert!(tags.is_none());
        assert!(fm.is_none());
        assert_eq!(body, text);
    }

    #[test]
    fn parse_frontmatter_no_closing_marker_returns_full_body() {
        let text = "---\ntags: [foo]\n# Body without closing";
        let (tags, fm, body) = parse_frontmatter(text, "test.md");
        assert!(tags.is_none());
        assert!(fm.is_none());
        assert_eq!(body, text);
    }

    #[test]
    fn parse_frontmatter_normalizes_tag_strings() {
        let text =
            "---\ntags:\n  - \"  hashed  \"\n  - \"#leading-hash\"\n  - \"\"\n  - foo\n---\nbody";
        let (tags, _fm, _body) = parse_frontmatter(text, "test.md");
        let tags = tags.unwrap();
        // BTreeSet dedup + sort
        assert_eq!(
            tags,
            vec![
                "foo".to_string(),
                "hashed".to_string(),
                "leading-hash".to_string()
            ]
        );
    }

    #[test]
    fn parse_frontmatter_scalar_tags_yields_no_tags() {
        let text = "---\ntags: not-a-list\nstatus: active\n---\nbody";
        let (tags, fm, body) = parse_frontmatter(text, "test.md");
        // tags is a scalar, so `as_sequence()` returns None → tags stays None
        assert!(tags.is_none());
        // Frontmatter map still parses fine
        let fm = fm.unwrap();
        assert_eq!(fm.get("status").unwrap(), "active");
        assert_eq!(body, "body");
    }

    #[test]
    fn parse_frontmatter_strips_body_correctly() {
        let text = "---\nstatus: active\n---\n\n# Real content\n\nMore text\n";
        let (_, _, body) = parse_frontmatter(text, "test.md");
        assert_eq!(body, "\n# Real content\n\nMore text\n");
    }
}
