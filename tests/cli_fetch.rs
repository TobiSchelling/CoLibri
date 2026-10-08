//! Fetchers through the CLI: Zephyr (fake HTTP API) and script fetchers.

mod common;

use std::sync::atomic::Ordering;

use common::{FakeOllama, FakeZephyrApi, TestHome};
use serde_json::Value;

fn json(out: &common::Output) -> Value {
    out.assert_ok();
    serde_json::from_str(&out.stdout).unwrap_or_else(|e| panic!("{e}: {}", out.stdout))
}

// AC-025.1, AC-025.2, AC-026.1, AC-028.1 over real HTTP.
#[test]
fn zephyr_fetch_writes_files_and_updates_incrementally() {
    let ollama = FakeOllama::start();
    let zephyr = FakeZephyrApi::start(60);
    let home = TestHome::new(&ollama);
    let folder = home.path().join("mirrors/zephyr-ctslab");
    let mut config = std::fs::read_to_string(home.config_path()).unwrap();
    config.push_str(&format!(
        "  - name: zephyr-ctslab\n    path: {}\n    doc_type: test_case\n    fetch: {{type: zephyr_scale, project_key: CTSLAB, api_base_url: '{}', token_env: FAKE_ZEPHYR_TOKEN}}\n",
        folder.display(),
        zephyr.url
    ));
    std::fs::write(home.config_path(), config).unwrap();
    std::env::set_var("FAKE_ZEPHYR_TOKEN", "t");

    let report = json(&home.colibri(&["update", "zephyr-ctslab", "--json"]));
    let m = &report["mirrors"][0];
    assert_eq!(m["fetch"]["listed"], 60, "both pages fetched: {m}");
    assert_eq!(m["fetch"]["complete"], true);
    assert_eq!(m["added"], 60);
    assert!(folder.join("CTSLAB-T59.md").exists());
    assert!(folder.join(".colibri-mirror").exists());

    let hits = json(&home.colibri(&[
        "search",
        "FOTA case",
        "--mode",
        "keyword",
        "--json",
        "--collection",
        "zephyr-ctslab",
        "--frontmatter",
        "status=Approved",
    ]));
    assert!(hits["total_results"].as_u64().unwrap() > 0);

    let steps_before = zephyr.step_requests.load(Ordering::SeqCst);
    let report = json(&home.colibri(&["update", "zephyr-ctslab", "--json"]));
    assert_eq!(report["mirrors"][0]["fetch"]["written"], 0);
    assert_eq!(report["mirrors"][0]["added"], 0);
    assert_eq!(
        zephyr.step_requests.load(Ordering::SeqCst),
        steps_before,
        "no step requests for unchanged cases"
    );
}

// AC-029.1
#[test]
fn script_fetchers_fill_folders_and_failures_block_pruning() {
    let ollama = FakeOllama::start();
    let home = TestHome::new(&ollama);
    let folder = home.path().join("mirrors/wiki");
    let script = home.path().join("export.sh");
    std::fs::write(
        &script,
        "#!/bin/sh\nset -e\nif [ -f \"$(dirname \"$0\")/fail\" ]; then rm -f \"$1/b.md\"; echo 'export failed' >&2; exit 1; fi\nprintf '# Alpha page\\nwiki alpha text\\n' > \"$1/a.md\"\nprintf '# Beta page\\nwiki beta text\\n' > \"$1/b.md\"\n",
    )
    .unwrap();
    let mut config = std::fs::read_to_string(home.config_path()).unwrap();
    config.push_str(&format!(
        "  - name: wiki\n    path: {}\n    fetch: {{type: command, run: [sh, '{}', '{{path}}']}}\n",
        folder.display(),
        script.display()
    ));
    std::fs::write(home.config_path(), config).unwrap();

    let report = json(&home.colibri(&["update", "wiki", "--json"]));
    assert_eq!(report["mirrors"][0]["added"], 2);

    std::fs::write(home.path().join("fail"), "").unwrap();
    let out = home.colibri(&["update", "wiki", "--json"]);
    assert!(!out.success, "a failed fetch is an error exit");
    let report: Value = serde_json::from_str(&out.stdout).unwrap();
    let m = &report["mirrors"][0];
    assert_eq!(m["pruned"], 0, "a failed fetch must not prune: {m}");
    assert!(m["prune_blocked"]
        .as_str()
        .unwrap()
        .contains("fetch was incomplete"));
    assert!(m["problems"]
        .as_array()
        .unwrap()
        .iter()
        .any(|p| p["kind"] == "fetch_failed"));

    std::fs::remove_file(home.path().join("fail")).unwrap();
    let report = json(&home.colibri(&["update", "wiki", "--json"]));
    assert_eq!(report["mirrors"][0]["pruned"], 0);
}
