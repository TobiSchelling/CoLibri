//! Markdown renderer for Zephyr Scale test cases.
//!
//! Renders test cases into markdown with YAML frontmatter,
//! matching the output format of the Python Jinja2 template.

use std::collections::HashMap;

use super::api::{ApiLink, ApiTestCase, ApiTestStep};
use super::folders::FolderTree;
use super::html_to_md::html_to_md;

/// Rendered test case output.
pub struct RenderedTestCase {
    pub markdown: String,
    pub title: String,
}

/// Render a test case into markdown with YAML frontmatter.
///
/// Converts HTML fields to markdown using `html_to_md()`,
/// builds YAML frontmatter, and assembles the full document.
pub fn render_test_case(
    tc: &ApiTestCase,
    project_key: &str,
    folder_tree: &FolderTree,
    steps: &[ApiTestStep],
    links: &[ApiLink],
    status_lookup: &HashMap<i64, String>,
    priority_lookup: &HashMap<i64, String>,
) -> RenderedTestCase {
    let mut out = String::new();

    // Resolve folder path
    let folder_path = tc
        .folder
        .as_ref()
        .and_then(|f| f.id)
        .and_then(|id| folder_tree.get_path(id))
        .unwrap_or("(unfiled)")
        .to_string();

    // Status and priority names — try inline name, then lookup by ID
    let status = tc
        .status
        .as_ref()
        .and_then(|s| {
            s.name
                .as_deref()
                .or_else(|| s.id.and_then(|id| status_lookup.get(&id).map(|n| n.as_str())))
        })
        .unwrap_or("Unknown");
    let priority = tc
        .priority
        .as_ref()
        .and_then(|p| {
            p.name
                .as_deref()
                .or_else(|| p.id.and_then(|id| priority_lookup.get(&id).map(|n| n.as_str())))
        })
        .unwrap_or("Unknown");

    let owner = tc.owner.as_deref().unwrap_or("");
    let created_on = tc.created_on.as_deref().unwrap_or("");
    let updated_on = tc.updated_on.as_deref().unwrap_or("");

    // YAML frontmatter
    out.push_str("---\n");
    out.push_str(&format!("key: {}\n", tc.key));
    out.push_str(&format!("name: {}\n", yaml_escape(&tc.name)));
    out.push_str(&format!("project: {project_key}\n"));
    out.push_str(&format!("folder: {folder_path}\n"));
    out.push_str(&format!("status: {status}\n"));
    out.push_str(&format!("priority: {priority}\n"));
    out.push_str(&format!("owner: {owner}\n"));

    // Labels as YAML array
    if !tc.labels.is_empty() {
        let labels: Vec<String> = tc.labels.iter().map(|l| l.to_string()).collect();
        out.push_str(&format!("labels: [{}]\n", labels.join(", ")));
    } else {
        out.push_str("labels: []\n");
    }

    // Links as YAML array
    if !links.is_empty() {
        out.push_str("links:\n");
        for link in links {
            if let Some(key) = &link.issue_key {
                out.push_str(&format!("  - issue: {key}\n"));
            }
            if let Some(url) = &link.url {
                out.push_str(&format!("  - url: {url}\n"));
            }
        }
    }

    out.push_str(&format!("created_on: {created_on}\n"));
    out.push_str(&format!("updated_on: {updated_on}\n"));
    out.push_str("---\n\n");

    // Title
    let title = format!("{}: {}", tc.key, tc.name);
    out.push_str(&format!("# {title}\n"));

    // Objective
    if let Some(objective) = &tc.objective {
        let md = html_to_md(objective);
        if !md.is_empty() {
            out.push_str(&format!("\n## Objective\n\n{md}\n"));
        }
    }

    // Precondition
    if let Some(precondition) = &tc.precondition {
        let md = html_to_md(precondition);
        if !md.is_empty() {
            out.push_str(&format!("\n## Precondition\n\n{md}\n"));
        }
    }

    // Test Steps
    if !steps.is_empty() {
        out.push_str("\n## Test Steps\n");

        let mut sorted_steps: Vec<&ApiTestStep> = steps.iter().collect();
        sorted_steps.sort_by_key(|s| s.index.unwrap_or(0));

        for (i, step) in sorted_steps.iter().enumerate() {
            let step_num = i + 1;
            let zephyr_idx = step.index.unwrap_or(0);
            if zephyr_idx > 0 {
                out.push_str(&format!(
                    "\n### Step {step_num} (Zephyr Index {zephyr_idx})\n"
                ));
            } else {
                out.push_str(&format!("\n### Step {step_num}\n"));
            }

            if let Some(desc) = &step.description {
                let md = html_to_md(desc);
                if !md.is_empty() {
                    out.push_str(&format!("\n**Description**\n{md}\n"));
                }
            }

            if let Some(data) = &step.test_data {
                let md = html_to_md(data);
                if !md.is_empty() {
                    out.push_str(&format!("\n**Test Data**\n{md}\n"));
                }
            }

            if let Some(expected) = &step.expected_result {
                let md = html_to_md(expected);
                if !md.is_empty() {
                    out.push_str(&format!("\n**Expected Result**\n{md}\n"));
                }
            }

            // Custom fields
            if let Some(cf) = &step.custom_fields {
                if let Some(obj) = cf.as_object() {
                    if !obj.is_empty() {
                        out.push_str("\n**Custom Fields**\n");
                        for (name, value) in obj {
                            let val_str = match value {
                                serde_json::Value::String(s) => s.clone(),
                                other => other.to_string(),
                            };
                            out.push_str(&format!("- `{name}`: {val_str}\n"));
                        }
                    }
                }
            }
        }
    } else if let Some(script) = &tc.test_script {
        if let Some(text) = &script.text {
            if !text.is_empty() {
                out.push_str("\n## Test Script (No Step Sequence Available)\n\n");
                out.push_str(&format!("```text\n{text}\n```\n"));
            }
        }
    } else {
        out.push_str("\n_No test steps available._\n");
    }

    RenderedTestCase {
        markdown: out,
        title,
    }
}

/// Escape a YAML string value if it contains special characters.
fn yaml_escape(s: &str) -> String {
    if s.contains(':') || s.contains('#') || s.contains('"') || s.contains('\'') {
        format!("\"{}\"", s.replace('"', "\\\""))
    } else {
        s.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connectors::zephyr_scale::api::{ApiRef, ApiTestScript};
    use crate::connectors::zephyr_scale::folders::RawFolder;

    fn empty_lookups() -> (HashMap<i64, String>, HashMap<i64, String>) {
        (HashMap::new(), HashMap::new())
    }

    fn sample_tree() -> FolderTree {
        FolderTree::build(vec![
            RawFolder {
                id: 1,
                name: "Regression".into(),
                parent_id: None,
            },
            RawFolder {
                id: 2,
                name: "Login".into(),
                parent_id: Some(1),
            },
        ])
    }

    fn sample_test_case() -> ApiTestCase {
        ApiTestCase {
            id: 123,
            key: "CTSLAB-T123".into(),
            name: "Verify login flow".into(),
            folder: Some(ApiRef {
                id: Some(2),
                name: Some("Login".into()),
                ref_type: None,
            }),
            status: Some(ApiRef {
                id: Some(1),
                name: Some("Approved".into()),
                ref_type: None,
            }),
            priority: Some(ApiRef {
                id: Some(2),
                name: Some("High".into()),
                ref_type: None,
            }),
            owner: Some("user@example.com".into()),
            objective: Some("<p>User can log in with valid credentials.</p>".into()),
            precondition: Some("<p>User account exists in the system.</p>".into()),
            labels: vec!["smoke".into(), "regression".into()],
            test_script: Some(ApiTestScript {
                script_type: Some("STEP_BY_STEP".into()),
                steps: None,
                text: None,
            }),
            custom_fields: None,
            created_on: Some("2025-01-15T10:30:00Z".into()),
            updated_on: Some("2025-02-20T14:15:00Z".into()),
        }
    }

    fn sample_steps() -> Vec<ApiTestStep> {
        vec![ApiTestStep {
            index: Some(1),
            description: Some("<p>Navigate to login page</p>".into()),
            test_data: Some("<p>URL: https://example.com/login</p>".into()),
            expected_result: Some("<p>Login page is displayed</p>".into()),
            custom_fields: None,
        }]
    }

    #[test]
    fn render_full_test_case() {
        let tree = sample_tree();
        let tc = sample_test_case();
        let steps = sample_steps();
        let links = vec![ApiLink {
            url: None,
            issue_key: Some("JIRA-456".into()),
        }];

        let (sl, pl) = empty_lookups();
        let result = render_test_case(&tc, "CTSLAB", &tree, &steps, &links, &sl, &pl);

        assert_eq!(result.title, "CTSLAB-T123: Verify login flow");
        assert!(result.markdown.contains("key: CTSLAB-T123"));
        assert!(result.markdown.contains("folder: /Regression/Login"));
        assert!(result.markdown.contains("status: Approved"));
        assert!(result.markdown.contains("priority: High"));
        assert!(result.markdown.contains("labels: [smoke, regression]"));
        assert!(result.markdown.contains("# CTSLAB-T123: Verify login flow"));
        assert!(result
            .markdown
            .contains("User can log in with valid credentials."));
        assert!(result
            .markdown
            .contains("User account exists in the system."));
        assert!(result.markdown.contains("### Step 1 (Zephyr Index 1)"));
        assert!(result.markdown.contains("Navigate to login page"));
        assert!(result.markdown.contains("issue: JIRA-456"));
    }

    #[test]
    fn render_empty_objective_omitted() {
        let tree = sample_tree();
        let mut tc = sample_test_case();
        tc.objective = None;
        tc.precondition = None;

        let (sl, pl) = empty_lookups();
        let result = render_test_case(&tc, "CTSLAB", &tree, &sample_steps(), &[], &sl, &pl);

        assert!(!result.markdown.contains("## Objective"));
        assert!(!result.markdown.contains("## Precondition"));
    }

    #[test]
    fn render_no_steps_fallback() {
        let tree = sample_tree();
        let mut tc = sample_test_case();
        tc.test_script = None;

        let (sl, pl) = empty_lookups();
        let result = render_test_case(&tc, "CTSLAB", &tree, &[], &[], &sl, &pl);

        assert!(result.markdown.contains("_No test steps available._"));
    }

    #[test]
    fn render_script_text_fallback() {
        let tree = sample_tree();
        let mut tc = sample_test_case();
        tc.test_script = Some(ApiTestScript {
            script_type: Some("PLAIN_TEXT".into()),
            steps: None,
            text: Some("Manual test steps here".into()),
        });

        let (sl, pl) = empty_lookups();
        let result = render_test_case(&tc, "CTSLAB", &tree, &[], &[], &sl, &pl);

        assert!(result
            .markdown
            .contains("## Test Script (No Step Sequence Available)"));
        assert!(result.markdown.contains("Manual test steps here"));
    }

    #[test]
    fn frontmatter_is_valid_yaml() {
        let tree = sample_tree();
        let tc = sample_test_case();
        let (sl, pl) = empty_lookups();
        let result = render_test_case(&tc, "CTSLAB", &tree, &sample_steps(), &[], &sl, &pl);

        // Extract frontmatter between --- markers
        let parts: Vec<&str> = result.markdown.splitn(3, "---").collect();
        assert_eq!(parts.len(), 3, "Should have frontmatter delimiters");
        let frontmatter = parts[1].trim();

        // Should be parseable as YAML
        let parsed: serde_yaml::Value = serde_yaml::from_str(frontmatter).unwrap();
        assert_eq!(parsed["key"].as_str(), Some("CTSLAB-T123"));
        assert_eq!(parsed["project"].as_str(), Some("CTSLAB"));
    }

    #[test]
    fn render_unfiled_test_case() {
        let tree = FolderTree::build(vec![]);
        let mut tc = sample_test_case();
        tc.folder = None;

        let (sl, pl) = empty_lookups();
        let result = render_test_case(&tc, "CTSLAB", &tree, &[], &[], &sl, &pl);
        assert!(result.markdown.contains("folder: (unfiled)"));
    }

    #[test]
    fn yaml_escape_special_chars() {
        assert_eq!(yaml_escape("simple"), "simple");
        assert_eq!(yaml_escape("has: colon"), "\"has: colon\"");
        assert_eq!(yaml_escape("has # hash"), "\"has # hash\"");
    }

    #[test]
    fn render_step_custom_fields() {
        let tree = sample_tree();
        let tc = sample_test_case();
        let steps = vec![ApiTestStep {
            index: Some(1),
            description: Some("Do something".into()),
            test_data: None,
            expected_result: None,
            custom_fields: Some(serde_json::json!({"Automation": "Yes"})),
        }];

        let (sl, pl) = empty_lookups();
        let result = render_test_case(&tc, "CTSLAB", &tree, &steps, &[], &sl, &pl);
        assert!(result.markdown.contains("**Custom Fields**"));
        assert!(result.markdown.contains("`Automation`: Yes"));
    }
}
