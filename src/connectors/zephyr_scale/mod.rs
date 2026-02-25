//! Zephyr Scale test case connector.
//!
//! Syncs test cases from the Zephyr Scale Cloud API into CoLibri's
//! canonical store as searchable markdown with YAML frontmatter.

pub mod api;
pub mod folders;
pub mod html_to_md;
pub mod render;

use async_trait::async_trait;

use super::Connector;
use crate::envelope::{
    content_hash, DocumentEnvelope, EnvelopeDocument, EnvelopeMetadata, EnvelopeSource,
};
use crate::error::ColibriError;

use api::ZephyrApiClient;
use folders::{FolderTree, RawFolder};
use render::render_test_case;

const PLUGIN_ID: &str = "zephyr_scale";

/// A connector that syncs test cases from the Zephyr Scale Cloud API.
pub struct ZephyrScaleConnector {
    pub id: String,
    pub project_key: String,
    pub api_base_url: String,
    pub token: String,
    pub folder_path: Option<String>,
    pub doc_type: String,
    pub classification: String,
    pub include_steps: bool,
    pub include_links: bool,
}

#[async_trait]
impl Connector for ZephyrScaleConnector {
    fn id(&self) -> &str {
        &self.id
    }

    async fn sync(&self) -> Result<Vec<DocumentEnvelope>, ColibriError> {
        let client = ZephyrApiClient::new(&self.api_base_url, &self.token);

        // 1. Fetch all folders and build tree
        let api_folders = client.get_folders(&self.project_key).await?;
        let folder_tree = FolderTree::build(
            api_folders
                .into_iter()
                .map(|f| RawFolder {
                    id: f.id,
                    name: f.name,
                    parent_id: f.parent_id,
                })
                .collect(),
        );

        // 2. Determine folder scope filter
        let scope_ids = if let Some(ref path) = self.folder_path {
            match folder_tree.find_by_path(path) {
                Some(folder_id) => Some(folder_tree.get_subtree_ids(folder_id)),
                None => {
                    return Err(ColibriError::Config(format!(
                        "Zephyr connector '{}': folder_path '{}' not found in project {}",
                        self.id, path, self.project_key
                    )));
                }
            }
        } else {
            None
        };

        // 3. Fetch all test cases
        let test_cases = client.get_test_cases(&self.project_key, None).await?;

        // 4. Filter by folder scope if set
        let test_cases: Vec<_> = if let Some(ref scope) = scope_ids {
            test_cases
                .into_iter()
                .filter(|tc| {
                    tc.folder
                        .as_ref()
                        .and_then(|f| f.id)
                        .map(|id| scope.contains(&id))
                        .unwrap_or(false)
                })
                .collect()
        } else {
            test_cases
        };

        // 5. Process each test case
        let connector_instance = format!("{PLUGIN_ID}:{}", self.project_key);
        let mut envelopes = Vec::with_capacity(test_cases.len());

        for tc in &test_cases {
            // Fetch steps: use inline if available, otherwise fetch separately
            let steps = if self.include_steps {
                if let Some(ref script) = tc.test_script {
                    if let Some(ref inline_steps) = script.steps {
                        inline_steps.clone()
                    } else {
                        client.get_test_steps(&tc.key).await.unwrap_or_default()
                    }
                } else {
                    client.get_test_steps(&tc.key).await.unwrap_or_default()
                }
            } else {
                Vec::new()
            };

            // Fetch links if enabled
            let links = if self.include_links {
                client.get_links(&tc.key).await.unwrap_or_default()
            } else {
                Vec::new()
            };

            // Render markdown
            let rendered =
                render_test_case(tc, &self.project_key, &folder_tree, &steps, &links);

            let markdown = rendered.markdown;
            let doc_id = format!("{PLUGIN_ID}:{}:{}", self.project_key, tc.key);
            let updated_at = tc
                .updated_on
                .as_deref()
                .or(tc.created_on.as_deref())
                .unwrap_or("1970-01-01T00:00:00Z");

            // Build tags
            let mut tags = vec!["zephyr".to_string(), self.project_key.clone()];
            tags.extend(tc.labels.iter().cloned());

            envelopes.push(DocumentEnvelope {
                schema_version: 1,
                source: EnvelopeSource {
                    plugin_id: PLUGIN_ID.into(),
                    connector_instance: connector_instance.clone(),
                    external_id: tc.key.clone(),
                    uri: None,
                },
                document: EnvelopeDocument {
                    doc_id,
                    title: rendered.title,
                    content_hash: content_hash(&markdown),
                    markdown,
                    source_updated_at: updated_at.to_string(),
                    deleted: false,
                },
                metadata: EnvelopeMetadata {
                    doc_type: self.doc_type.clone(),
                    classification: self.classification.clone(),
                    tags: Some(tags),
                    language: None,
                    acl_tags: None,
                },
            });
        }

        Ok(envelopes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn doc_id_format() {
        let doc_id = format!("{PLUGIN_ID}:{}:{}", "CTSLAB", "CTSLAB-T123");
        assert_eq!(doc_id, "zephyr_scale:CTSLAB:CTSLAB-T123");
    }

    #[test]
    fn connector_instance_format() {
        let instance = format!("{PLUGIN_ID}:{}", "CTSLAB");
        assert_eq!(instance, "zephyr_scale:CTSLAB");
    }

    #[test]
    fn plugin_id_is_correct() {
        assert_eq!(PLUGIN_ID, "zephyr_scale");
    }
}
