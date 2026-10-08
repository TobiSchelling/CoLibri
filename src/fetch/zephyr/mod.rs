//! Zephyr Scale fetcher: one `<KEY>.md` file per test case, with YAML
//! frontmatter, in the mirror folder.
//!
//! Steps and links (one or two API calls per test case) are only fetched
//! again when a test case's fingerprint changed, or on the periodic full
//! refresh. Files are deleted only after a complete listing, and only for
//! test cases this fetcher wrote.

pub mod api;
pub mod folders;
pub mod html_to_md;
pub mod render;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{prepare_folder, write_if_changed, FetchReport};
use crate::config::PruneConfig;
use crate::error::ColibriError;
use crate::ingest::mirror::Problem;
use api::{ApiFolder, ApiLink, ApiNamedEntity, ApiRef, ApiTestCase, ApiTestStep, ZephyrApiClient};
use folders::{FolderTree, RawFolder};
use render::render_test_case;

/// Fetch state kept next to the files.
const STATE_FILE: &str = ".fetch-state.json";
/// Save the state after this many written test cases, so an interrupted
/// first fetch keeps its progress.
const SAVE_EVERY: usize = 50;
/// Fingerprint of a test case whose last fetch failed: never matches, so the
/// next run fetches it again, but the file stays owned by the fetcher.
const RETRY: &str = "";

#[derive(Debug, Clone)]
pub struct ZephyrFetchConfig {
    pub project_key: String,
    pub api_base_url: String,
    /// Environment variable holding the API token.
    pub token_env: String,
    /// Only test cases below this folder path (e.g. `/Regression`).
    pub folder_path: Option<String>,
    pub include_steps: bool,
    pub include_links: bool,
    /// Fetch every test case's steps and links again after this many days.
    pub full_refresh_days: i64,
}

/// Where test cases come from: the Zephyr API, or a fake in tests.
#[allow(async_fn_in_trait)]
pub trait ZephyrSource {
    async fn folders(&self) -> Result<Vec<ApiFolder>, ColibriError>;
    async fn statuses(&self) -> Result<Vec<ApiNamedEntity>, ColibriError>;
    async fn priorities(&self) -> Result<Vec<ApiNamedEntity>, ColibriError>;
    /// Test cases and whether the listing is complete.
    async fn test_cases(&self) -> Result<(Vec<ApiTestCase>, bool), ColibriError>;
    async fn steps(&self, key: &str) -> Result<Vec<ApiTestStep>, ColibriError>;
    async fn links(&self, key: &str) -> Result<Vec<ApiLink>, ColibriError>;
}

/// The Zephyr Scale Cloud API.
pub struct ApiSource {
    client: ZephyrApiClient,
    project_key: String,
}

impl ApiSource {
    pub fn from_config(cfg: &ZephyrFetchConfig) -> Result<Self, String> {
        let token = std::env::var(&cfg.token_env)
            .ok()
            .filter(|t| !t.trim().is_empty())
            .ok_or_else(|| format!("no Zephyr API token in ${}", cfg.token_env))?;
        Ok(Self {
            client: ZephyrApiClient::new(&cfg.api_base_url, &token),
            project_key: cfg.project_key.clone(),
        })
    }
}

impl ZephyrSource for ApiSource {
    async fn folders(&self) -> Result<Vec<ApiFolder>, ColibriError> {
        self.client.get_folders(&self.project_key).await
    }
    async fn statuses(&self) -> Result<Vec<ApiNamedEntity>, ColibriError> {
        self.client.get_statuses(&self.project_key).await
    }
    async fn priorities(&self) -> Result<Vec<ApiNamedEntity>, ColibriError> {
        self.client.get_priorities(&self.project_key).await
    }
    async fn test_cases(&self) -> Result<(Vec<ApiTestCase>, bool), ColibriError> {
        self.client.get_test_cases(&self.project_key, None).await
    }
    async fn steps(&self, key: &str) -> Result<Vec<ApiTestStep>, ColibriError> {
        self.client.get_test_steps(key).await
    }
    async fn links(&self, key: &str) -> Result<Vec<ApiLink>, ColibriError> {
        self.client.get_links(key).await
    }
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct FetchState {
    last_full_refresh: Option<String>,
    /// Test case key → fingerprint of what was rendered.
    cases: BTreeMap<String, String>,
}

fn read_state(dir: &Path) -> FetchState {
    std::fs::read_to_string(dir.join(STATE_FILE))
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_default()
}

fn file_for(dir: &Path, key: &str) -> std::path::PathBuf {
    let safe: String = key
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    dir.join(format!("{safe}.md"))
}

fn resolve_name(r: Option<&ApiRef>, lookup: &HashMap<i64, String>) -> String {
    r.and_then(|r| {
        r.name
            .clone()
            .or_else(|| r.id.and_then(|id| lookup.get(&id).cloned()))
    })
    .unwrap_or_else(|| "Unknown".into())
}

fn fingerprint(
    tc: &ApiTestCase,
    folder: &str,
    status: &str,
    priority: &str,
    cfg: &ZephyrFetchConfig,
) -> String {
    let mut h = Sha256::new();
    h.update(serde_json::to_string(tc).unwrap_or_default());
    for part in [folder, status, priority] {
        h.update([0]);
        h.update(part);
    }
    h.update([cfg.include_steps as u8, cfg.include_links as u8]);
    format!("{:x}", h.finalize())
}

fn fail(report: &mut FetchReport, what: &str, e: impl std::fmt::Display) -> FetchReport {
    report
        .problems
        .push(Problem::new(what, "fetch_failed", e.to_string()));
    report.complete = false;
    std::mem::take(report)
}

/// Fetch test cases into `dir`. Never deletes anything unless the listing
/// is complete; a failure for one test case keeps its existing file.
/// `prune` limits how many files one run may delete (`None`: no limit).
pub async fn fetch<S: ZephyrSource>(
    source: &S,
    cfg: &ZephyrFetchConfig,
    dir: &Path,
    now: DateTime<Utc>,
    prune: Option<PruneConfig>,
) -> FetchReport {
    let mut report = FetchReport::default();
    if let Err(e) = prepare_folder(dir) {
        return fail(&mut report, &dir.display().to_string(), e);
    }
    let old = read_state(dir);

    let tree = match source.folders().await {
        Ok(f) => FolderTree::build(
            f.into_iter()
                .map(|f| RawFolder {
                    id: f.id,
                    name: f.name,
                    parent_id: f.parent_id,
                })
                .collect(),
        ),
        Err(e) => return fail(&mut report, "folders", e),
    };
    let scope = match &cfg.folder_path {
        Some(path) => match tree.find_by_path(path) {
            Some(id) => Some(tree.get_subtree_ids(id)),
            None => {
                return fail(
                    &mut report,
                    "folder_path",
                    format!("'{path}' not found in project {}", cfg.project_key),
                )
            }
        },
        None => None,
    };
    let lookup = |r: Result<Vec<ApiNamedEntity>, ColibriError>| {
        r.map(|v| {
            v.into_iter()
                .map(|e| (e.id, e.name))
                .collect::<HashMap<_, _>>()
        })
    };
    let statuses = match lookup(source.statuses().await) {
        Ok(m) => m,
        Err(e) => return fail(&mut report, "statuses", e),
    };
    let priorities = match lookup(source.priorities().await) {
        Ok(m) => m,
        Err(e) => return fail(&mut report, "priorities", e),
    };
    let (cases, complete) = match source.test_cases().await {
        Ok(r) => r,
        Err(e) => return fail(&mut report, "test cases", e),
    };
    let cases: Vec<ApiTestCase> = cases
        .into_iter()
        .filter(|tc| {
            scope.as_ref().is_none_or(|s| {
                tc.folder
                    .as_ref()
                    .and_then(|f| f.id)
                    .is_some_and(|id| s.contains(&id))
            })
        })
        .collect();
    report.listed = cases.len();
    report.complete = complete;
    if !complete {
        report.problems.push(Problem::new(
            "test cases",
            "incomplete_listing",
            "Zephyr returned an incomplete listing; nothing was deleted",
        ));
    }

    let full_refresh = old
        .last_full_refresh
        .as_deref()
        .and_then(|t| DateTime::parse_from_rfc3339(t).ok())
        .is_none_or(|t| now - t.with_timezone(&Utc) >= Duration::days(cfg.full_refresh_days));
    // Start from the old state: every file written earlier stays owned
    // (and deletable) even if this run is interrupted or a case fails.
    let mut state = FetchState {
        last_full_refresh: old.last_full_refresh.clone(),
        cases: old.cases.clone(),
    };
    let mut since_save = 0;

    let mut listed: HashSet<String> = HashSet::new();
    for tc in &cases {
        listed.insert(tc.key.clone());
        let path = file_for(dir, &tc.key);
        let folder = tc
            .folder
            .as_ref()
            .and_then(|f| f.id)
            .and_then(|id| tree.get_path(id))
            .unwrap_or("(unfiled)")
            .to_string();
        let status = resolve_name(tc.status.as_ref(), &statuses);
        let priority = resolve_name(tc.priority.as_ref(), &priorities);
        let fp = fingerprint(tc, &folder, &status, &priority, cfg);
        if !full_refresh && old.cases.get(&tc.key) == Some(&fp) && path.exists() {
            report.unchanged += 1;
            continue;
        }

        let inline_steps = tc.test_script.as_ref().and_then(|s| s.steps.clone());
        let steps = match (cfg.include_steps, inline_steps) {
            (false, _) => Ok(Vec::new()),
            (true, Some(s)) => Ok(s),
            (true, None) => source.steps(&tc.key).await,
        };
        let links = match &steps {
            Ok(_) if cfg.include_links => source.links(&tc.key).await,
            _ => Ok(Vec::new()),
        };
        let (steps, links) = match (steps, links) {
            (Ok(s), Ok(l)) => (s, l),
            (Err(e), _) | (_, Err(e)) => {
                report.problems.push(Problem::new(
                    tc.key.clone(),
                    "fetch_failed",
                    format!("{e}; kept the previous version"),
                ));
                mark_retry(&mut state, &tc.key);
                continue;
            }
        };
        let rendered = render_test_case(
            tc,
            &cfg.project_key,
            &tree,
            &steps,
            &links,
            &statuses,
            &priorities,
        );
        match write_if_changed(&path, &rendered.markdown) {
            Ok(true) => report.written += 1,
            Ok(false) => report.unchanged += 1,
            Err(e) => {
                report.problems.push(Problem::new(
                    path.display().to_string(),
                    "write_failed",
                    e.to_string(),
                ));
                mark_retry(&mut state, &tc.key);
                continue;
            }
        }
        state.cases.insert(tc.key.clone(), fp);
        since_save += 1;
        if since_save >= SAVE_EVERY {
            save_state(dir, &state, &mut report);
            since_save = 0;
        }
    }

    let gone: Vec<String> = state
        .cases
        .keys()
        .filter(|k| !listed.contains(*k))
        .cloned()
        .collect();
    let limit = prune.map(|p| p.limit(old.cases.len()));
    match limit {
        _ if !complete || gone.is_empty() => {}
        Some(limit) if gone.len() > limit => {
            report.complete = false;
            report.problems.push(Problem::new(
                "test cases",
                "mass_delete_blocked",
                format!(
                    "{} test cases are no longer listed (limit {limit}); nothing was deleted. \
                     Use --allow-mass-prune if this is intended",
                    gone.len()
                ),
            ));
        }
        _ => {
            for key in gone {
                match std::fs::remove_file(file_for(dir, &key)) {
                    Ok(()) => report.deleted += 1,
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                    Err(e) => {
                        report.problems.push(Problem::new(
                            key.clone(),
                            "delete_failed",
                            e.to_string(),
                        ));
                        continue;
                    }
                }
                state.cases.remove(&key);
            }
        }
    }

    if full_refresh {
        // Cases that failed carry the RETRY fingerprint and are fetched
        // again next run, so the refresh counts as done.
        state.last_full_refresh = Some(now.to_rfc3339());
    }
    save_state(dir, &state, &mut report);
    report
}

/// Keep ownership of a known file but force a fetch next run.
fn mark_retry(state: &mut FetchState, key: &str) {
    if let Some(fp) = state.cases.get_mut(key) {
        *fp = RETRY.to_string();
    }
}

fn save_state(dir: &Path, state: &FetchState, report: &mut FetchReport) {
    let result = serde_json::to_string_pretty(state)
        .map_err(std::io::Error::other)
        .and_then(|json| write_if_changed(&dir.join(STATE_FILE), &json));
    if let Err(e) = result {
        report
            .problems
            .push(Problem::new(STATE_FILE, "write_failed", e.to_string()));
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::cell::{Cell, RefCell};

    /// In-memory Zephyr project.
    pub(crate) struct FakeZephyr {
        pub cases: RefCell<Vec<serde_json::Value>>,
        pub complete: Cell<bool>,
        pub step_calls: Cell<usize>,
        pub link_calls: Cell<usize>,
        pub fail_steps_for: RefCell<Option<String>>,
        /// The steps request for this key never returns.
        pub hang_steps_for: RefCell<Option<String>>,
    }

    impl FakeZephyr {
        pub(crate) fn new(n: usize) -> Self {
            let cases = (1..=n)
                .map(|i| {
                    serde_json::json!({
                        "id": i, "key": format!("CTSLAB-T{i}"), "name": format!("Case {i}"),
                        "folder": {"id": 2}, "status": {"id": 10}, "priority": {"id": 20},
                        "labels": ["smoke"], "objective": "<p>Check <b>it</b></p>",
                        "updatedOn": "2026-10-01T00:00:00Z"
                    })
                })
                .collect();
            Self {
                cases: RefCell::new(cases),
                complete: Cell::new(true),
                step_calls: Cell::new(0),
                link_calls: Cell::new(0),
                fail_steps_for: RefCell::new(None),
                hang_steps_for: RefCell::new(None),
            }
        }
    }

    impl ZephyrSource for FakeZephyr {
        async fn folders(&self) -> Result<Vec<ApiFolder>, ColibriError> {
            Ok(serde_json::from_value(serde_json::json!([
                {"id": 1, "name": "Regression", "parentId": null},
                {"id": 2, "name": "FOTA", "parentId": 1}
            ]))
            .unwrap())
        }
        async fn statuses(&self) -> Result<Vec<ApiNamedEntity>, ColibriError> {
            Ok(vec![ApiNamedEntity {
                id: 10,
                name: "Approved".into(),
            }])
        }
        async fn priorities(&self) -> Result<Vec<ApiNamedEntity>, ColibriError> {
            Ok(vec![ApiNamedEntity {
                id: 20,
                name: "High".into(),
            }])
        }
        async fn test_cases(&self) -> Result<(Vec<ApiTestCase>, bool), ColibriError> {
            let cases =
                serde_json::from_value(serde_json::Value::Array(self.cases.borrow().clone()))
                    .unwrap();
            Ok((cases, self.complete.get()))
        }
        async fn steps(&self, key: &str) -> Result<Vec<ApiTestStep>, ColibriError> {
            self.step_calls.set(self.step_calls.get() + 1);
            if self.hang_steps_for.borrow().as_deref() == Some(key) {
                std::future::pending::<()>().await;
            }
            if self.fail_steps_for.borrow().as_deref() == Some(key) {
                return Err(ColibriError::Api("503 Service Unavailable".into()));
            }
            Ok(serde_json::from_value(serde_json::json!([
                {"index": 1, "description": "Start update", "expectedResult": "Update succeeds"}
            ]))
            .unwrap())
        }
        async fn links(&self, _key: &str) -> Result<Vec<ApiLink>, ColibriError> {
            self.link_calls.set(self.link_calls.get() + 1);
            Ok(vec![ApiLink {
                url: None,
                issue_key: Some("BUTSAM-1".into()),
            }])
        }
    }

    pub(crate) fn config() -> ZephyrFetchConfig {
        ZephyrFetchConfig {
            project_key: "CTSLAB".into(),
            api_base_url: "http://unused".into(),
            token_env: "UNUSED".into(),
            folder_path: None,
            include_steps: true,
            include_links: true,
            full_refresh_days: 7,
        }
    }

    fn now() -> DateTime<Utc> {
        DateTime::parse_from_rfc3339("2026-10-08T10:00:00Z")
            .unwrap()
            .with_timezone(&Utc)
    }

    fn frontmatter(path: &Path) -> serde_yaml::Value {
        let text = std::fs::read_to_string(path).unwrap();
        serde_yaml::from_str(text.split("---").nth(1).unwrap()).unwrap()
    }

    // AC-025.1
    #[tokio::test]
    async fn writes_one_markdown_file_per_test_case() {
        let dir = tempfile::TempDir::new().unwrap();
        let fake = FakeZephyr::new(3);
        let r = fetch(&fake, &config(), dir.path(), now(), None).await;
        assert_eq!((r.listed, r.written, r.deleted), (3, 3, 0));
        assert!(r.complete && r.problems.is_empty(), "{:?}", r.problems);
        let fm = frontmatter(&dir.path().join("CTSLAB-T2.md"));
        for key in [
            "title",
            "key",
            "name",
            "folder",
            "status",
            "priority",
            "labels",
            "updated_on",
        ] {
            assert!(fm.get(key).is_some(), "missing {key}");
        }
        assert_eq!(fm["status"].as_str(), Some("Approved"));
        assert_eq!(fm["folder"].as_str(), Some("/Regression/FOTA"));
        assert!(dir.path().join(crate::fetch::MARKER).exists());
    }

    // AC-027.2, AC-028.1
    #[tokio::test]
    async fn unchanged_cases_cost_no_requests_and_no_writes() {
        let dir = tempfile::TempDir::new().unwrap();
        let fake = FakeZephyr::new(3);
        fetch(&fake, &config(), dir.path(), now(), None).await;
        let mtime = |k: &str| {
            std::fs::metadata(dir.path().join(format!("{k}.md")))
                .unwrap()
                .modified()
                .unwrap()
        };
        let before = mtime("CTSLAB-T1");
        let (steps, links) = (fake.step_calls.get(), fake.link_calls.get());

        let r = fetch(
            &fake,
            &config(),
            dir.path(),
            now() + Duration::hours(1),
            None,
        )
        .await;
        assert_eq!((r.written, r.unchanged), (0, 3));
        assert_eq!(
            (fake.step_calls.get(), fake.link_calls.get()),
            (steps, links)
        );
        assert_eq!(mtime("CTSLAB-T1"), before);

        fake.cases.borrow_mut()[1]["updatedOn"] = "2026-10-08T09:00:00Z".into();
        let r = fetch(
            &fake,
            &config(),
            dir.path(),
            now() + Duration::hours(2),
            None,
        )
        .await;
        assert_eq!(r.written, 1);
        assert_eq!(
            (fake.step_calls.get(), fake.link_calls.get()),
            (steps + 1, links + 1)
        );
    }

    #[tokio::test]
    async fn full_refresh_refetches_everything() {
        let dir = tempfile::TempDir::new().unwrap();
        let fake = FakeZephyr::new(2);
        fetch(&fake, &config(), dir.path(), now(), None).await;
        let calls = fake.step_calls.get();
        fetch(
            &fake,
            &config(),
            dir.path(),
            now() + Duration::days(8),
            None,
        )
        .await;
        assert_eq!(fake.step_calls.get(), calls + 2);
    }

    // AC-026.2
    #[tokio::test]
    async fn failed_steps_keep_the_previous_file() {
        let dir = tempfile::TempDir::new().unwrap();
        let fake = FakeZephyr::new(2);
        fetch(&fake, &config(), dir.path(), now(), None).await;
        let before = std::fs::read(dir.path().join("CTSLAB-T1.md")).unwrap();

        fake.cases.borrow_mut()[0]["name"] = "Renamed".into();
        *fake.fail_steps_for.borrow_mut() = Some("CTSLAB-T1".into());
        let r = fetch(&fake, &config(), dir.path(), now(), None).await;
        assert_eq!(
            std::fs::read(dir.path().join("CTSLAB-T1.md")).unwrap(),
            before
        );
        assert!(r
            .problems
            .iter()
            .any(|p| p.kind == "fetch_failed" && p.source_path == "CTSLAB-T1"));

        // Next run retries it.
        *fake.fail_steps_for.borrow_mut() = None;
        let r = fetch(&fake, &config(), dir.path(), now(), None).await;
        assert_eq!(r.written, 1);
    }

    // A failure during a full refresh is retried on the next run, not
    // only at the next full refresh (the fingerprint alone is unchanged).
    #[tokio::test]
    async fn failed_full_refresh_cases_are_retried_next_run() {
        let dir = tempfile::TempDir::new().unwrap();
        let fake = FakeZephyr::new(2);
        fetch(&fake, &config(), dir.path(), now(), None).await;

        *fake.fail_steps_for.borrow_mut() = Some("CTSLAB-T1".into());
        let refresh = now() + Duration::days(8);
        let r = fetch(&fake, &config(), dir.path(), refresh, None).await;
        assert!(r.problems.iter().any(|p| p.kind == "fetch_failed"));

        *fake.fail_steps_for.borrow_mut() = None;
        let calls = fake.step_calls.get();
        fetch(
            &fake,
            &config(),
            dir.path(),
            refresh + Duration::hours(1),
            None,
        )
        .await;
        assert_eq!(fake.step_calls.get(), calls + 1, "only the failed case");
    }

    // An empty but "complete" listing (e.g. a token without permission)
    // must not delete every fetched file.
    #[tokio::test]
    async fn mass_deletion_is_blocked_without_override() {
        let dir = tempfile::TempDir::new().unwrap();
        let fake = FakeZephyr::new(30);
        fetch(&fake, &config(), dir.path(), now(), None).await;
        fake.cases.borrow_mut().clear();
        let guard = Some(PruneConfig {
            max_fraction: 0.2,
            min_count: 25,
        });

        let r = fetch(&fake, &config(), dir.path(), now(), guard).await;
        assert_eq!(r.deleted, 0);
        assert!(!r.complete, "blocks the reconcile prune too");
        assert!(r.problems.iter().any(|p| p.kind == "mass_delete_blocked"));
        assert!(dir.path().join("CTSLAB-T30.md").exists());

        let r = fetch(&fake, &config(), dir.path(), now(), None).await;
        assert_eq!(r.deleted, 30);
    }

    // State is saved while fetching, so files from an interrupted run stay
    // owned and are deleted later when the source drops them.
    #[tokio::test]
    async fn state_is_saved_during_long_fetches() {
        let dir = tempfile::TempDir::new().unwrap();
        let fake = FakeZephyr::new(SAVE_EVERY + 1);
        *fake.hang_steps_for.borrow_mut() = Some(format!("CTSLAB-T{}", SAVE_EVERY + 1));
        let cfg = config();
        let run = fetch(&fake, &cfg, dir.path(), now(), None);
        let interrupted = tokio::time::timeout(std::time::Duration::from_millis(500), run).await;
        assert!(interrupted.is_err(), "the fetch hangs on the last case");
        assert_eq!(read_state(dir.path()).cases.len(), SAVE_EVERY);

        // The source drops one of them: the next complete run deletes it.
        *fake.hang_steps_for.borrow_mut() = None;
        fake.cases.borrow_mut().remove(0);
        let r = fetch(&fake, &config(), dir.path(), now(), None).await;
        assert_eq!(r.deleted, 1);
        assert!(!dir.path().join("CTSLAB-T1.md").exists());
    }

    // AC-025.2: fetched files become documents with filterable frontmatter.
    #[tokio::test]
    async fn fetched_test_cases_reconcile_into_documents() {
        use crate::config::{AppConfig, MirrorConfig};
        use crate::ingest::convert::fakes::CountingConverter;
        use crate::ingest::mirror::{reconcile_mirror, ReconcileOptions};

        let dir = tempfile::TempDir::new().unwrap();
        let folder = dir.path().join("zephyr-ctslab");
        let fake = FakeZephyr::new(2);
        fetch(&fake, &config(), &folder, now(), None).await;

        let config = AppConfig::for_test(&dir.path().join("home"));
        let mirror = MirrorConfig {
            name: "zephyr-ctslab".into(),
            path: folder,
            fetch: None,
            doc_type: "test_case".into(),
            include: vec!["**/*.md".into()],
            exclude: vec![],
            plantuml_summaries: false,
        };
        let (_lock, store) = config.open_for_write().unwrap();
        let r = reconcile_mirror(
            &config,
            Some(&store),
            &mirror,
            &CountingConverter::default(),
            ReconcileOptions::default(),
        )
        .unwrap();
        assert_eq!(r.added, 2);
        let doc = store
            .get_document("zephyr-ctslab:CTSLAB-T1.md")
            .unwrap()
            .unwrap();
        assert_eq!(doc.title, "CTSLAB-T1: Case 1");
        assert_eq!(doc.doc_type, "test_case");
        assert!(
            doc.frontmatter_json.contains("\"status\":\"Approved\""),
            "{}",
            doc.frontmatter_json
        );
        assert!(doc.tags_json.contains("zephyr"));
    }

    // AC-026.1 (deletion part)
    #[tokio::test]
    async fn deletes_only_after_complete_listings() {
        let dir = tempfile::TempDir::new().unwrap();
        let fake = FakeZephyr::new(3);
        fetch(&fake, &config(), dir.path(), now(), None).await;
        std::fs::write(dir.path().join("notes.md"), "not ours").unwrap();
        fake.cases.borrow_mut().pop();

        fake.complete.set(false);
        let r = fetch(&fake, &config(), dir.path(), now(), None).await;
        assert_eq!(r.deleted, 0);
        assert!(dir.path().join("CTSLAB-T3.md").exists());
        assert!(r.problems.iter().any(|p| p.kind == "incomplete_listing"));

        fake.complete.set(true);
        let r = fetch(&fake, &config(), dir.path(), now(), None).await;
        assert_eq!(r.deleted, 1);
        assert!(!dir.path().join("CTSLAB-T3.md").exists());
        assert!(
            dir.path().join("notes.md").exists(),
            "files we did not write stay"
        );
    }
}
