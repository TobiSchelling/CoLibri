//! Regex-based HTML to Markdown converter for Zephyr Scale HTML fields.

use regex::Regex;
use std::sync::OnceLock;

/// Compiled regex patterns for HTML-to-Markdown conversion.
struct Patterns {
    br: Regex,
    bold_open: Regex,
    bold_close: Regex,
    italic_open: Regex,
    italic_close: Regex,
    link: Regex,
    li_open: Regex,
    li_close: Regex,
    p_open: Regex,
    p_close: Regex,
    strip_tags: Regex,
    collapse_newlines: Regex,
}

static PATTERNS: OnceLock<Patterns> = OnceLock::new();

fn patterns() -> &'static Patterns {
    PATTERNS.get_or_init(|| Patterns {
        br: Regex::new(r"(?i)<br\s*/?>").unwrap(),
        bold_open: Regex::new(r"(?i)<(strong|b)>").unwrap(),
        bold_close: Regex::new(r"(?i)</(strong|b)>").unwrap(),
        italic_open: Regex::new(r"(?i)<(em|i)>").unwrap(),
        italic_close: Regex::new(r"(?i)</(em|i)>").unwrap(),
        link: Regex::new(r#"(?is)<a\s[^>]*href=["']([^"']*)["'][^>]*>(.*?)</a>"#).unwrap(),
        li_open: Regex::new(r"(?i)<li[^>]*>").unwrap(),
        li_close: Regex::new(r"(?i)</li>").unwrap(),
        p_open: Regex::new(r"(?i)<p[^>]*>").unwrap(),
        p_close: Regex::new(r"(?i)</p>").unwrap(),
        strip_tags: Regex::new(r"<[^>]+>").unwrap(),
        collapse_newlines: Regex::new(r"\n{3,}").unwrap(),
    })
}

/// Convert simple HTML (from Zephyr Scale API) to Markdown.
///
/// Handles: `<br>`, `<strong>`/`<b>`, `<em>`/`<i>`, `<a href>`,
/// `<li>`, `<p>`, entity decoding, and strips remaining tags.
pub fn html_to_md(html: &str) -> String {
    if html.is_empty() {
        return String::new();
    }

    let p = patterns();
    let mut text = html.to_string();

    // Line breaks
    text = p.br.replace_all(&text, "\n").to_string();

    // Bold
    text = p.bold_open.replace_all(&text, "**").to_string();
    text = p.bold_close.replace_all(&text, "**").to_string();

    // Italic
    text = p.italic_open.replace_all(&text, "_").to_string();
    text = p.italic_close.replace_all(&text, "_").to_string();

    // Links: <a href="url">text</a> → [text](url)
    text = p.link.replace_all(&text, "[$2]($1)").to_string();

    // List items
    text = p.li_open.replace_all(&text, "- ").to_string();
    text = p.li_close.replace_all(&text, "\n").to_string();

    // Paragraphs
    text = p.p_open.replace_all(&text, "").to_string();
    text = p.p_close.replace_all(&text, "\n\n").to_string();

    // Strip remaining HTML tags
    text = p.strip_tags.replace_all(&text, "").to_string();

    // Decode HTML entities
    text = text
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&nbsp;", " ");

    // Collapse excessive newlines
    text = p.collapse_newlines.replace_all(&text, "\n\n").to_string();

    text.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input() {
        assert_eq!(html_to_md(""), "");
    }

    #[test]
    fn plain_text_passthrough() {
        assert_eq!(html_to_md("Hello world"), "Hello world");
    }

    #[test]
    fn br_tags() {
        assert_eq!(html_to_md("line1<br>line2"), "line1\nline2");
        assert_eq!(html_to_md("line1<br/>line2"), "line1\nline2");
        assert_eq!(html_to_md("line1<br />line2"), "line1\nline2");
        assert_eq!(html_to_md("line1<BR>line2"), "line1\nline2");
    }

    #[test]
    fn bold_tags() {
        assert_eq!(html_to_md("<strong>bold</strong>"), "**bold**");
        assert_eq!(html_to_md("<b>bold</b>"), "**bold**");
        assert_eq!(html_to_md("<STRONG>bold</STRONG>"), "**bold**");
    }

    #[test]
    fn italic_tags() {
        assert_eq!(html_to_md("<em>italic</em>"), "_italic_");
        assert_eq!(html_to_md("<i>italic</i>"), "_italic_");
        assert_eq!(html_to_md("<EM>italic</EM>"), "_italic_");
    }

    #[test]
    fn links() {
        assert_eq!(
            html_to_md(r#"<a href="https://example.com">Example</a>"#),
            "[Example](https://example.com)"
        );
    }

    #[test]
    fn links_with_attributes() {
        assert_eq!(
            html_to_md(r#"<a href="https://example.com" target="_blank">Link</a>"#),
            "[Link](https://example.com)"
        );
    }

    #[test]
    fn list_items() {
        let html = "<ul><li>first</li><li>second</li></ul>";
        let md = html_to_md(html);
        assert!(md.contains("- first"));
        assert!(md.contains("- second"));
    }

    #[test]
    fn paragraphs() {
        let html = "<p>First paragraph</p><p>Second paragraph</p>";
        let md = html_to_md(html);
        assert!(md.contains("First paragraph"));
        assert!(md.contains("Second paragraph"));
        // Should have paragraph break between them
        assert!(md.contains("\n\n"));
    }

    #[test]
    fn html_entities() {
        assert_eq!(html_to_md("&amp; &lt; &gt; &quot;"), "& < > \"");
        assert_eq!(html_to_md("non&nbsp;breaking"), "non breaking");
    }

    #[test]
    fn strips_remaining_tags() {
        assert_eq!(html_to_md("<div>content</div>"), "content");
        assert_eq!(html_to_md("<span class=\"x\">text</span>"), "text");
    }

    #[test]
    fn collapses_blank_lines() {
        assert_eq!(html_to_md("a\n\n\n\n\nb"), "a\n\nb");
    }

    #[test]
    fn combined_formatting() {
        let html = "<p><strong>Bold</strong> and <em>italic</em> with <a href=\"#\">link</a></p>";
        let md = html_to_md(html);
        assert!(md.contains("**Bold**"));
        assert!(md.contains("_italic_"));
        assert!(md.contains("[link](#)"));
    }

    #[test]
    fn multiline_link() {
        let html = "<a href=\"https://example.com\">\n  Link text\n</a>";
        let md = html_to_md(html);
        assert!(md.contains("(https://example.com)"));
    }
}
