//! Zephyr Scale REST API v2 types and HTTP client.
//!
//! Provides serde types for API responses and a thin `reqwest`-based client
//! with pagination support.

// API response types include all fields from the Zephyr API schema for
// correct deserialization, even when not all fields are read in logic.
#![allow(dead_code)]

use reqwest::header::{AUTHORIZATION, CONTENT_TYPE};
use serde::Deserialize;

use crate::error::ColibriError;

// ───────────────────────────────────────────────────────────────
// API response types
// ───────────────────────────────────────────────────────────────

/// Paginated response wrapper from the Zephyr Scale API.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PaginatedResponse<T> {
    pub values: Vec<T>,
    pub next: Option<String>,
    pub is_last: Option<bool>,
    pub start_at: Option<u64>,
    pub max_results: Option<u64>,
    pub total: Option<u64>,
}

/// A reference to another entity (folder, status, priority, etc.).
#[derive(Debug, Deserialize, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct ApiRef {
    pub id: Option<i64>,
    pub name: Option<String>,
    #[serde(rename = "type")]
    pub ref_type: Option<String>,
}

/// A folder from the Zephyr Scale API.
#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ApiFolder {
    pub id: i64,
    pub name: String,
    pub folder_type: Option<String>,
    pub parent_id: Option<i64>,
}

/// A test case from the Zephyr Scale API.
#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ApiTestCase {
    pub id: i64,
    pub key: String,
    pub name: String,
    #[serde(default)]
    pub folder: Option<ApiRef>,
    #[serde(default)]
    pub status: Option<ApiRef>,
    #[serde(default)]
    pub priority: Option<ApiRef>,
    /// Owner can be a string or a Jira user object `{"accountId": "..."}`.
    #[serde(default, deserialize_with = "deserialize_owner")]
    pub owner: Option<String>,
    #[serde(default)]
    pub objective: Option<String>,
    #[serde(default)]
    pub precondition: Option<String>,
    #[serde(default)]
    pub labels: Vec<String>,
    #[serde(default)]
    pub test_script: Option<ApiTestScript>,
    #[serde(default)]
    pub custom_fields: Option<serde_json::Value>,
    #[serde(default)]
    pub created_on: Option<String>,
    #[serde(default)]
    pub updated_on: Option<String>,
}

/// Deserialize owner from either a plain string or a Jira user object.
fn deserialize_owner<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value: Option<serde_json::Value> = Option::deserialize(deserializer)?;
    Ok(value.and_then(|v| match v {
        serde_json::Value::String(s) => Some(s),
        serde_json::Value::Object(ref obj) => obj
            .get("accountId")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),
        _ => None,
    }))
}

/// Inline test script within a test case.
#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ApiTestScript {
    #[serde(rename = "type")]
    pub script_type: Option<String>,
    #[serde(default)]
    pub steps: Option<Vec<ApiTestStep>>,
    #[serde(default)]
    pub text: Option<String>,
}

/// A single test step (flat representation used internally).
#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ApiTestStep {
    #[serde(default)]
    pub index: Option<u32>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub test_data: Option<String>,
    #[serde(default)]
    pub expected_result: Option<String>,
    #[serde(default)]
    pub custom_fields: Option<serde_json::Value>,
}

/// Wrapper for a test step in the paginated response.
/// Steps are wrapped as `{"inline": {...}, "testCase": ...}`.
#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ApiTestStepWrapper {
    #[serde(default)]
    pub inline: Option<ApiTestStep>,
}

/// A named entity from status/priority lookup endpoints.
#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ApiNamedEntity {
    pub id: i64,
    pub name: String,
}

/// A linked issue or URL.
#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ApiLink {
    pub url: Option<String>,
    #[serde(default)]
    pub issue_key: Option<String>,
}

/// Wrapper for the links response.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ApiLinksResponse {
    #[serde(default)]
    pub issue_links: Vec<ApiIssueLink>,
    #[serde(default)]
    pub web_links: Vec<ApiWebLink>,
}

/// An issue link from the links endpoint.
#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ApiIssueLink {
    pub target: Option<ApiIssueLinkTarget>,
}

/// Target of an issue link.
#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ApiIssueLinkTarget {
    pub issue_key: Option<String>,
}

/// A web link from the links endpoint.
#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ApiWebLink {
    pub url: Option<String>,
}

// ───────────────────────────────────────────────────────────────
// HTTP client
// ───────────────────────────────────────────────────────────────

const MAX_RESULTS: u64 = 50;

/// Thin HTTP client for the Zephyr Scale REST API v2.
pub struct ZephyrApiClient {
    base_url: String,
    client: reqwest::Client,
    token: String,
}

impl ZephyrApiClient {
    /// Create a new client with the given base URL and Bearer token.
    pub fn new(base_url: &str, token: &str) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .expect("reqwest client build");

        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            client,
            token: token.to_string(),
        }
    }

    /// Fetch all folders for a project.
    pub async fn get_folders(&self, project_key: &str) -> Result<Vec<ApiFolder>, ColibriError> {
        let url = format!(
            "{}/folders?folderType=TEST_CASE&projectKey={}&maxResults={}",
            self.base_url, project_key, MAX_RESULTS
        );
        self.paginate::<ApiFolder>(&url).await
    }

    /// Fetch all test cases for a project, optionally filtered by folder.
    pub async fn get_test_cases(
        &self,
        project_key: &str,
        folder_id: Option<i64>,
    ) -> Result<Vec<ApiTestCase>, ColibriError> {
        let mut url = format!(
            "{}/testcases?projectKey={}&maxResults={}",
            self.base_url, project_key, MAX_RESULTS
        );
        if let Some(fid) = folder_id {
            url.push_str(&format!("&folderId={fid}"));
        }
        self.paginate::<ApiTestCase>(&url).await
    }

    /// Fetch test steps for a specific test case.
    ///
    /// The API wraps steps in `{"inline": {...}}` objects; this unwraps them.
    pub async fn get_test_steps(
        &self,
        test_case_key: &str,
    ) -> Result<Vec<ApiTestStep>, ColibriError> {
        let url = format!(
            "{}/testcases/{}/teststeps?maxResults={}",
            self.base_url, test_case_key, MAX_RESULTS
        );
        let wrappers = self.paginate::<ApiTestStepWrapper>(&url).await?;
        Ok(wrappers.into_iter().filter_map(|w| w.inline).collect())
    }

    /// Fetch all test case statuses for a project (for name lookup).
    pub async fn get_statuses(
        &self,
        project_key: &str,
    ) -> Result<Vec<ApiNamedEntity>, ColibriError> {
        let url = format!(
            "{}/statuses?projectKey={}&statusType=TEST_CASE&maxResults={}",
            self.base_url, project_key, MAX_RESULTS
        );
        self.paginate::<ApiNamedEntity>(&url).await
    }

    /// Fetch all priorities for a project (for name lookup).
    pub async fn get_priorities(
        &self,
        project_key: &str,
    ) -> Result<Vec<ApiNamedEntity>, ColibriError> {
        let url = format!(
            "{}/priorities?projectKey={}&maxResults={}",
            self.base_url, project_key, MAX_RESULTS
        );
        self.paginate::<ApiNamedEntity>(&url).await
    }

    /// Fetch links for a specific test case.
    pub async fn get_links(&self, test_case_key: &str) -> Result<Vec<ApiLink>, ColibriError> {
        let url = format!("{}/testcases/{}/links", self.base_url, test_case_key);
        let resp = self
            .client
            .get(&url)
            .header(AUTHORIZATION, format!("Bearer {}", self.token))
            .header(CONTENT_TYPE, "application/json")
            .send()
            .await
            .map_err(|e| ColibriError::Api(format!("Zephyr API request failed: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(ColibriError::Api(format!(
                "Zephyr API error {status}: {body}"
            )));
        }

        let links_resp: ApiLinksResponse = resp.json().await.map_err(|e| {
            ColibriError::Api(format!("Failed to parse Zephyr links response: {e}"))
        })?;

        let mut links = Vec::new();
        for il in links_resp.issue_links {
            if let Some(target) = il.target {
                links.push(ApiLink {
                    url: None,
                    issue_key: target.issue_key,
                });
            }
        }
        for wl in links_resp.web_links {
            links.push(ApiLink {
                url: wl.url,
                issue_key: None,
            });
        }

        Ok(links)
    }

    /// Generic paginated fetch.
    async fn paginate<T: serde::de::DeserializeOwned>(
        &self,
        initial_url: &str,
    ) -> Result<Vec<T>, ColibriError> {
        let mut all_values = Vec::new();
        let mut url = initial_url.to_string();

        loop {
            let resp = self
                .client
                .get(&url)
                .header(AUTHORIZATION, format!("Bearer {}", self.token))
                .header(CONTENT_TYPE, "application/json")
                .send()
                .await
                .map_err(|e| ColibriError::Api(format!("Zephyr API request failed: {e}")))?;

            if !resp.status().is_success() {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                return Err(ColibriError::Api(format!(
                    "Zephyr API error {status}: {body}"
                )));
            }

            let page: PaginatedResponse<T> = resp.json().await.map_err(|e| {
                ColibriError::Api(format!("Failed to parse Zephyr API response: {e}"))
            })?;

            all_values.extend(page.values);

            // Check if this is the last page
            if page.is_last.unwrap_or(true) {
                break;
            }

            // Use `next` URL if provided, otherwise calculate next offset
            if let Some(next) = page.next {
                url = if next.starts_with("http") {
                    next
                } else {
                    format!("{}{}", self.base_url, next)
                };
            } else {
                break;
            }
        }

        Ok(all_values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialize_paginated_folders() {
        let json = r#"{
            "values": [
                {"id": 1, "name": "Regression", "folderType": "TEST_CASE", "parentId": null},
                {"id": 2, "name": "API", "folderType": "TEST_CASE", "parentId": 1}
            ],
            "next": "https://api.zephyrscale.smartbear.com/v2/folders?startAt=2",
            "isLast": false,
            "startAt": 0,
            "maxResults": 50,
            "total": 10
        }"#;

        let resp: PaginatedResponse<ApiFolder> = serde_json::from_str(json).unwrap();
        assert_eq!(resp.values.len(), 2);
        assert_eq!(resp.values[0].name, "Regression");
        assert_eq!(resp.values[1].parent_id, Some(1));
        assert_eq!(resp.is_last, Some(false));
        assert!(resp.next.is_some());
    }

    #[test]
    fn deserialize_paginated_last_page() {
        let json = r#"{
            "values": [{"id": 3, "name": "Smoke", "parentId": null}],
            "isLast": true,
            "startAt": 2,
            "maxResults": 50
        }"#;

        let resp: PaginatedResponse<ApiFolder> = serde_json::from_str(json).unwrap();
        assert_eq!(resp.values.len(), 1);
        assert_eq!(resp.is_last, Some(true));
        assert!(resp.next.is_none());
    }

    #[test]
    fn deserialize_test_case() {
        let json = r#"{
            "id": 123,
            "key": "CTSLAB-T123",
            "name": "Verify login flow",
            "folder": {"id": 5, "name": "Login"},
            "status": {"id": 1, "name": "Approved"},
            "priority": {"id": 2, "name": "High"},
            "owner": "user@example.com",
            "objective": "<p>User can log in</p>",
            "precondition": "<p>Account exists</p>",
            "labels": ["smoke", "regression"],
            "testScript": {
                "type": "STEP_BY_STEP",
                "steps": [
                    {
                        "index": 1,
                        "description": "<p>Navigate to login</p>",
                        "testData": "<p>URL: https://example.com</p>",
                        "expectedResult": "<p>Page loads</p>"
                    }
                ]
            },
            "createdOn": "2025-01-15T10:30:00Z",
            "updatedOn": "2025-02-20T14:15:00Z"
        }"#;

        let tc: ApiTestCase = serde_json::from_str(json).unwrap();
        assert_eq!(tc.key, "CTSLAB-T123");
        assert_eq!(tc.name, "Verify login flow");
        assert_eq!(tc.folder.as_ref().unwrap().id, Some(5));
        assert_eq!(
            tc.status.as_ref().unwrap().name.as_deref(),
            Some("Approved")
        );
        assert_eq!(tc.labels, vec!["smoke", "regression"]);
        assert!(tc.test_script.is_some());
        let steps = tc.test_script.unwrap().steps.unwrap();
        assert_eq!(steps.len(), 1);
        assert_eq!(steps[0].index, Some(1));
    }

    #[test]
    fn deserialize_test_case_minimal() {
        let json = r#"{
            "id": 456,
            "key": "PROJ-T1",
            "name": "Minimal test"
        }"#;

        let tc: ApiTestCase = serde_json::from_str(json).unwrap();
        assert_eq!(tc.key, "PROJ-T1");
        assert!(tc.folder.is_none());
        assert!(tc.objective.is_none());
        assert!(tc.labels.is_empty());
    }

    #[test]
    fn deserialize_test_step() {
        let json = r#"{
            "index": 1,
            "description": "<p>Do something</p>",
            "testData": "input data",
            "expectedResult": "<p>Expected output</p>"
        }"#;

        let step: ApiTestStep = serde_json::from_str(json).unwrap();
        assert_eq!(step.index, Some(1));
        assert_eq!(step.description.as_deref(), Some("<p>Do something</p>"));
        assert_eq!(step.test_data.as_deref(), Some("input data"));
    }

    #[test]
    fn deserialize_links_response() {
        let json = r#"{
            "issueLinks": [
                {"target": {"issueKey": "JIRA-123"}}
            ],
            "webLinks": [
                {"url": "https://example.com/docs"}
            ]
        }"#;

        let resp: ApiLinksResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.issue_links.len(), 1);
        assert_eq!(
            resp.issue_links[0]
                .target
                .as_ref()
                .unwrap()
                .issue_key
                .as_deref(),
            Some("JIRA-123")
        );
        assert_eq!(resp.web_links.len(), 1);
        assert_eq!(
            resp.web_links[0].url.as_deref(),
            Some("https://example.com/docs")
        );
    }

    #[test]
    fn deserialize_empty_paginated() {
        let json = r#"{
            "values": [],
            "isLast": true
        }"#;

        let resp: PaginatedResponse<ApiFolder> = serde_json::from_str(json).unwrap();
        assert!(resp.values.is_empty());
        assert_eq!(resp.is_last, Some(true));
    }

    #[test]
    fn deserialize_api_ref() {
        let json = r#"{"id": 42, "name": "Active", "type": "STATUS"}"#;
        let r: ApiRef = serde_json::from_str(json).unwrap();
        assert_eq!(r.id, Some(42));
        assert_eq!(r.name.as_deref(), Some("Active"));
        assert_eq!(r.ref_type.as_deref(), Some("STATUS"));
    }
}
