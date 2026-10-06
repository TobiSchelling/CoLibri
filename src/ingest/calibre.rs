//! calibre `metadata.opf` parsing: stable book identity and bibliographic data.

use std::path::Path;

const DC: &str = "http://purl.org/dc/elements/1.1/";
const OPF: &str = "http://www.idpf.org/2007/opf";

/// Bibliographic data of one book.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BookMeta {
    /// calibre's stable book UUID (`dc:identifier opf:scheme="uuid"`).
    pub uuid: Option<String>,
    pub calibre_id: Option<String>,
    pub title: Option<String>,
    pub authors: Vec<String>,
    pub language: Option<String>,
    pub tags: Vec<String>,
    pub publisher: Option<String>,
    pub date: Option<String>,
    pub isbn: Option<String>,
}

fn text(node: roxmltree::Node<'_, '_>) -> Option<String> {
    node.text()
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
}

/// Parse an OPF document (calibre writes OPF 2.0 with `opf:` attributes).
pub fn parse_opf_str(xml: &str) -> Result<BookMeta, String> {
    let doc = roxmltree::Document::parse(xml).map_err(|e| format!("invalid OPF: {e}"))?;
    let mut meta = BookMeta::default();
    for node in doc
        .descendants()
        .filter(|n| n.tag_name().namespace() == Some(DC))
    {
        match node.tag_name().name() {
            "identifier" => {
                let scheme = node
                    .attribute((OPF, "scheme"))
                    .or_else(|| node.attribute("scheme"))
                    .unwrap_or("")
                    .to_ascii_lowercase();
                match scheme.as_str() {
                    "uuid" => meta.uuid = text(node),
                    "calibre" => meta.calibre_id = text(node),
                    "isbn" => meta.isbn = text(node),
                    _ => {}
                }
            }
            "title" if meta.title.is_none() => meta.title = text(node),
            "creator" => {
                let role = node
                    .attribute((OPF, "role"))
                    .or_else(|| node.attribute("role"))
                    .unwrap_or("aut");
                if role == "aut" {
                    meta.authors.extend(text(node));
                }
            }
            "language" if meta.language.is_none() => meta.language = text(node),
            "subject" => meta.tags.extend(text(node)),
            "publisher" if meta.publisher.is_none() => meta.publisher = text(node),
            "date" if meta.date.is_none() => meta.date = text(node),
            _ => {}
        }
    }
    Ok(meta)
}

pub fn parse_opf(path: &Path) -> Result<BookMeta, String> {
    let xml = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    parse_opf_str(&xml)
}

#[cfg(test)]
mod tests {
    use super::*;

    const PRO_GIT: &str = r#"<?xml version='1.0' encoding='utf-8'?>
<package xmlns="http://www.idpf.org/2007/opf" unique-identifier="uuid_id" version="2.0">
    <metadata xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:opf="http://www.idpf.org/2007/opf">
        <dc:identifier opf:scheme="calibre" id="calibre_id">90</dc:identifier>
        <dc:identifier opf:scheme="uuid" id="uuid_id">f939e65f-2b6c-495d-a300-af21ee1a49b9</dc:identifier>
        <dc:identifier opf:scheme="ISBN">9781484200773</dc:identifier>
        <dc:title>Pro Git</dc:title>
        <dc:creator opf:file-as="Chacon, Scott &amp; Straub, Ben" opf:role="aut">Scott Chacon</dc:creator>
        <dc:creator opf:file-as="Chacon, Scott &amp; Straub, Ben" opf:role="aut">Ben Straub</dc:creator>
        <dc:contributor opf:file-as="calibre" opf:role="bkp">calibre (8.16.2)</dc:contributor>
        <dc:date>2025-12-12T00:00:00+00:00</dc:date>
        <dc:publisher>Carl Hanser Verlag GmbH &amp; Co. KG</dc:publisher>
        <dc:language>eng</dc:language>
        <dc:subject>Software Development &amp; Engineering</dc:subject>
        <dc:subject>Git</dc:subject>
    </metadata>
</package>"#;

    // AC-020.1
    #[test]
    fn parses_calibre_opf() {
        let m = parse_opf_str(PRO_GIT).unwrap();
        assert_eq!(
            m.uuid.as_deref(),
            Some("f939e65f-2b6c-495d-a300-af21ee1a49b9")
        );
        assert_eq!(m.calibre_id.as_deref(), Some("90"));
        assert_eq!(m.isbn.as_deref(), Some("9781484200773"));
        assert_eq!(m.title.as_deref(), Some("Pro Git"));
        assert_eq!(m.authors, ["Scott Chacon", "Ben Straub"]);
        assert_eq!(m.language.as_deref(), Some("eng"));
        assert_eq!(
            m.publisher.as_deref(),
            Some("Carl Hanser Verlag GmbH & Co. KG")
        );
        assert_eq!(m.tags, ["Software Development & Engineering", "Git"]);
    }

    #[test]
    fn broken_opf_is_an_error() {
        assert!(parse_opf_str("<package><metadata>").is_err());
    }
}
