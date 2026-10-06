//! PlantUML enrichment: entity/relation summaries appended as HTML comments
//! so diagrams become searchable.

use std::collections::{BTreeMap, BTreeSet};

/// Summary extracted from a PlantUML code block.
struct PlantUmlSummary {
    entities: Vec<String>,
    relations: Vec<String>,
}

/// Trim whitespace and surrounding single/double quotes.
fn norm(s: &str) -> String {
    let s = s.trim();
    let s = s.strip_prefix('"').unwrap_or(s);
    let s = s.strip_suffix('"').unwrap_or(s);
    let s = s.strip_prefix('\'').unwrap_or(s);
    let s = s.strip_suffix('\'').unwrap_or(s);
    s.to_string()
}

/// Arrow operators recognized in PlantUML diagrams, ordered longest-first
/// so that `<-->` is tried before `-->`.
const ARROWS: &[&str] = &[
    "<-->", "--|>", "<->", "-->", "..>", "-|>", "->", "<--", ".>", "<-",
];

/// Declaration keywords that introduce named entities in PlantUML.
const DECL_KEYWORDS: &[&str] = &[
    "participant",
    "actor",
    "boundary",
    "control",
    "entity",
    "database",
    "component",
    "interface",
    "class",
    "object",
    "usecase",
];

/// Parse a PlantUML code block and extract entities and relations.
fn parse_plantuml(text: &str) -> PlantUmlSummary {
    let mut alias_to_name: BTreeMap<String, String> = BTreeMap::new();
    let mut entities: BTreeSet<String> = BTreeSet::new();
    let mut relations: Vec<String> = Vec::new();

    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('\'') {
            continue;
        }

        // Alias lines: `"User" as U` or `participant "User" as U`
        if line.contains(" as ") {
            let parts: Vec<&str> = line.splitn(2, " as ").collect();
            let left_tokens: Vec<&str> = parts[0].split_whitespace().collect();
            let left = norm(left_tokens.last().unwrap_or(&parts[0]));
            let right_tokens: Vec<&str> = parts[1].split_whitespace().collect();
            let right = norm(right_tokens.first().unwrap_or(&""));
            if !left.is_empty() && !right.is_empty() {
                alias_to_name.insert(right.clone(), left.clone());
                entities.insert(left);
            }
        }

        // Declaration keywords: `participant Foo`, `actor "Bar"`
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if let Some(first) = tokens.first() {
            let first_lower = first.to_lowercase();
            if DECL_KEYWORDS.iter().any(|kw| *kw == first_lower) && tokens.len() >= 2 {
                let name = norm(tokens[1]);
                if !name.is_empty() {
                    entities.insert(name);
                }
            }
        }

        // Arrow relations: A -> B : label
        for arrow in ARROWS {
            if line.contains(arrow) {
                let (left_part, rest) = match line.split_once(arrow) {
                    Some(pair) => pair,
                    None => continue,
                };
                let left_tokens: Vec<&str> = left_part.split_whitespace().collect();
                let left = norm(left_tokens.last().unwrap_or(&""));
                let rest = rest.trim();

                // Separate entity from label: split on `:` first, then
                // take the last whitespace-delimited token of the entity
                // portion.  This handles both `B : label` and `B: label`.
                let (entity_part, label) =
                    if let Some((before_colon, after_colon)) = rest.split_once(':') {
                        let entity_tokens: Vec<&str> = before_colon.split_whitespace().collect();
                        let ent = norm(entity_tokens.first().unwrap_or(&""));
                        (ent, after_colon.trim().to_string())
                    } else {
                        let entity_tokens: Vec<&str> = rest.split_whitespace().collect();
                        let ent = norm(entity_tokens.first().unwrap_or(&""));
                        (ent, String::new())
                    };
                let right = entity_part;

                if left.is_empty() || right.is_empty() {
                    continue;
                }

                let left = alias_to_name.get(&left).cloned().unwrap_or(left);
                let right = alias_to_name.get(&right).cloned().unwrap_or(right);

                entities.insert(left.clone());
                entities.insert(right.clone());

                let mut rel = format!("{left} {arrow} {right}");
                if !label.is_empty() {
                    rel = format!("{rel}: {label}");
                }
                relations.push(rel);
                break; // only match the first arrow per line
            }
        }
    }

    PlantUmlSummary {
        entities: entities.into_iter().collect(),
        relations,
    }
}

/// Remove existing PlantUML summary blocks from markdown content.
fn strip_existing_plantuml_summaries(md: &str) -> String {
    const START: &str = "<!-- colibri:plantuml-summary:start -->";
    const END: &str = "<!-- colibri:plantuml-summary:end -->";

    let mut out = Vec::new();
    let mut in_block = false;

    for line in md.lines() {
        let trimmed = line.trim();
        if trimmed == START {
            in_block = true;
            continue;
        }
        if trimmed == END {
            in_block = false;
            continue;
        }
        if !in_block {
            out.push(line);
        }
    }
    out.join("\n")
}

/// Enrich markdown content by adding text summaries after PlantUML code blocks.
///
/// Each PlantUML block gets an HTML comment appended listing the entities and
/// relations found in the diagram.  This makes diagram content searchable in the
/// vector index without altering how the markdown renders visually.
pub fn enrich_plantuml_blocks(md: &str) -> String {
    let md = strip_existing_plantuml_summaries(md);

    let lines: Vec<&str> = md.lines().collect();
    let mut out: Vec<String> = Vec::new();
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i];
        out.push(line.to_string());

        let trimmed = line.trim();
        if trimmed.starts_with("```") && (trimmed == "```plantuml" || trimmed == "```puml") {
            // Collect the code block content
            i += 1;
            let mut block = Vec::new();
            while i < lines.len() && lines[i].trim() != "```" {
                block.push(lines[i]);
                out.push(lines[i].to_string());
                i += 1;
            }
            if i < lines.len() {
                out.push(lines[i].to_string()); // closing fence
            }

            let summary = parse_plantuml(&block.join("\n"));
            if !summary.entities.is_empty() || !summary.relations.is_empty() {
                out.push(String::new());
                out.push("<!-- colibri:plantuml-summary:start -->".to_string());
                out.push("[CoLibri PlantUML summary]".to_string());
                if !summary.entities.is_empty() {
                    out.push(format!("Entities: {}", summary.entities.join(", ")));
                }
                if !summary.relations.is_empty() {
                    out.push("Relations:".to_string());
                    for r in summary.relations.iter().take(50) {
                        out.push(format!("- {r}"));
                    }
                }
                out.push("<!-- colibri:plantuml-summary:end -->".to_string());
                out.push(String::new());
            }
        }

        i += 1;
    }

    let result = out.join("\n");
    // Match Python behavior: strip trailing whitespace and ensure single trailing newline
    result.trim_end().to_string() + "\n"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plantuml_enrichment_adds_summary() {
        let input = r#"# Doc

```plantuml
actor User
User -> Server: request
Server -> DB: query
```

More text.
"#;
        let enriched = enrich_plantuml_blocks(input);
        assert!(enriched.contains("colibri:plantuml-summary:start"));
        assert!(enriched.contains("User"));
        assert!(enriched.contains("Server"));
        assert!(enriched.contains("DB"));
    }

    #[test]
    fn test_plantuml_enrichment_no_plantuml() {
        let input = "# Just markdown\n\nNo diagrams here.\n";
        let enriched = enrich_plantuml_blocks(input);
        assert!(!enriched.contains("plantuml-summary"));
        assert_eq!(enriched.trim(), input.trim());
    }

    #[test]
    fn test_plantuml_strips_existing_summaries() {
        let input = r#"# Doc

```plantuml
A -> B
```

<!-- colibri:plantuml-summary:start -->
old summary
<!-- colibri:plantuml-summary:end -->

More text.
"#;
        let enriched = enrich_plantuml_blocks(input);
        assert!(!enriched.contains("old summary"));
        assert!(enriched.contains("colibri:plantuml-summary:start"));
    }

    #[test]
    fn test_plantuml_parse_entities_sorted() {
        let text = "actor Zulu\nactor Alpha\nAlpha -> Zulu: hello";
        let summary = parse_plantuml(text);
        assert_eq!(summary.entities, vec!["Alpha", "Zulu"]);
    }

    #[test]
    fn test_plantuml_parse_alias() {
        // `"User" as U` — Python's behavior: takes last token of left side,
        // so `"User"` becomes the entity after norm strips quotes.
        let text = "\"User\" as U\nactor Other\nU -> Other: msg";
        let summary = parse_plantuml(text);
        assert!(summary.entities.contains(&"User".to_string()));
        assert!(summary.entities.contains(&"Other".to_string()));
        // Alias U is resolved to User in relations
        assert!(summary.relations[0].contains("User"));
    }

    #[test]
    fn test_plantuml_skips_comments() {
        let text = "' this is a comment\nactor A\nA -> B";
        let summary = parse_plantuml(text);
        assert!(summary.entities.contains(&"A".to_string()));
        assert!(summary.entities.contains(&"B".to_string()));
        assert_eq!(summary.relations.len(), 1);
    }

    #[test]
    fn test_plantuml_relations_capped_at_50() {
        let lines: Vec<String> = (0..60).map(|i| format!("A{i} -> B{i}: rel{i}")).collect();
        let text = lines.join("\n");
        let _summary = parse_plantuml(&text);
        // parse_plantuml collects all; enrich_plantuml_blocks caps output at 50
        let md = format!("```plantuml\n{text}\n```\n");
        let enriched = enrich_plantuml_blocks(&md);
        let rel_lines: Vec<&str> = enriched.lines().filter(|l| l.starts_with("- ")).collect();
        assert_eq!(rel_lines.len(), 50);
    }

    #[test]
    fn test_norm_strips_quotes() {
        assert_eq!(norm("  \"Foo\"  "), "Foo");
        assert_eq!(norm("'Bar'"), "Bar");
        assert_eq!(norm("  plain  "), "plain");
    }

    #[test]
    fn test_plantuml_puml_fence() {
        let input = "```puml\nA -> B\n```\n";
        let enriched = enrich_plantuml_blocks(input);
        assert!(enriched.contains("colibri:plantuml-summary:start"));
        assert!(enriched.contains("A"));
        assert!(enriched.contains("B"));
    }

    #[test]
    fn test_plantuml_various_arrows() {
        let text = "A --> B\nC <-- D\nE ..> F\nG -|> H";
        let summary = parse_plantuml(text);
        assert_eq!(summary.relations.len(), 4);
        assert!(summary.relations[0].contains("-->"));
        assert!(summary.relations[1].contains("<--"));
        assert!(summary.relations[2].contains("..>"));
        assert!(summary.relations[3].contains("-|>"));
    }
}
