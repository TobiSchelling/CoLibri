//! Mirror behaviour through the CLI: update, prune, dry run, status.

mod common;

use common::{FakeOllama, TestHome};
use serde_json::Value;

fn json(out: &common::Output) -> Value {
    out.assert_ok();
    serde_json::from_str(&out.stdout).unwrap_or_else(|e| panic!("{e}: {}", out.stdout))
}

fn collection<'a>(status: &'a Value, name: &str) -> &'a Value {
    status["collections"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == name)
        .unwrap_or_else(|| panic!("collection {name} missing: {status}"))
}

#[test]
fn update_prune_dry_run_and_status() {
    let ollama = FakeOllama::start();
    // A second mirror whose folder does not exist produces a problem.
    let home = TestHome::with_mirrors(&ollama, "  - name: gone\n    path: /definitely/not/here\n");
    home.write_source(
        "plans/alpha.md",
        "---\nstatus: active\n---\n# Alpha\nalpha plan\n",
    );
    home.write_source(
        "plans/draft.md",
        "---\nstatus: draft\n---\n# Draft\nalpha draft\n",
    );
    home.write_source(
        "arch/billing-service.yaml",
        "service: billing\nowner: payments-team\n",
    );
    home.write_source("notes/remove-me.md", "# Scratch\nthrowaway zebra note\n");

    // AC-013.1: dry run reports the plan and writes nothing. It already
    // fails (exit 1) because mirror `gone` has no folder.
    let out = home.colibri(&["update", "--dry-run", "--json"]);
    assert!(!out.success);
    let plan: Value = serde_json::from_str(&out.stdout).unwrap();
    assert_eq!(plan["mirrors"][1]["problems"][0]["kind"], "root_missing");
    assert_eq!(plan["mirrors"][0]["added"], 4);
    assert!(
        !home.data_dir().exists(),
        "dry run must not create the data dir"
    );

    // First update: exit code 1 because mirror `gone` is an error; `notes` is ingested.
    let out = home.colibri(&["update", "--json"]);
    assert!(!out.success, "missing mirror folder must fail the run");
    let report: Value = serde_json::from_str(&out.stdout).unwrap();
    let notes = report["mirrors"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["name"] == "notes")
        .unwrap();
    assert_eq!(notes["added"], 4);
    assert_eq!(report["index"]["indexed"], 4);

    // AC-015.1: frontmatter is filterable.
    let hits = json(&home.colibri(&[
        "search",
        "alpha",
        "--mode",
        "keyword",
        "--json",
        "--frontmatter",
        "status=active",
    ]));
    assert_eq!(hits["total_results"], 1);
    let hits = json(&home.colibri(&[
        "search",
        "alpha",
        "--mode",
        "keyword",
        "--json",
        "--frontmatter",
        "status=archived",
    ]));
    assert_eq!(hits["total_results"], 0);

    // AC-015.2: yaml files are indexed.
    let hits = json(&home.colibri(&["search", "payments-team", "--mode", "keyword", "--json"]));
    assert!(hits["results"][0]["file"]
        .as_str()
        .unwrap()
        .ends_with("arch/billing-service.yaml"));

    // AC-010.1: a deleted file disappears from search after the next update.
    std::fs::remove_file(home.source_dir().join("notes/remove-me.md")).unwrap();
    let report = json(&home.colibri(&["update", "notes", "--json"]));
    assert_eq!(report["mirrors"][0]["pruned"], 1);
    let hits = json(&home.colibri(&["search", "zebra", "--mode", "keyword", "--json"]));
    assert_eq!(hits["total_results"], 0);

    // AC-013.1 with existing data: pending add, change and delete are
    // reported, and neither the DB nor any file in the data dir changes.
    home.write_source("plans/new.md", "# New\nfresh\n");
    home.write_source(
        "plans/alpha.md",
        "---\nstatus: active\n---\n# Alpha\nalpha plan v2\n",
    );
    std::fs::remove_file(home.source_dir().join("plans/draft.md")).unwrap();
    let before = common::snapshot(&home.data_dir());
    let plan = json(&home.colibri(&["update", "notes", "--dry-run", "--json"]));
    let m = &plan["mirrors"][0];
    assert_eq!(
        (
            m["added"].as_u64(),
            m["changed"].as_u64(),
            m["pruned"].as_u64()
        ),
        (Some(1), Some(1), Some(1))
    );
    assert_eq!(
        common::snapshot(&home.data_dir()),
        before,
        "dry run must not write"
    );

    // Orphans vs pending: an edited but not re-embedded note is pending, not orphaned.
    let report = json(&home.colibri(&["update", "notes", "--no-index", "--json"]));
    assert_eq!(report["mirrors"][0]["pruned"], 1);
    let status = json(&home.colibri(&["status", "--json"]));
    // Only the pruned draft's single chunk is orphaned; alpha's outdated chunk is not.
    assert_eq!(status["orphan_chunks"], 1);
    let pending = collection(&status, "notes")["pending_index"]
        .as_u64()
        .unwrap();
    assert_eq!(pending, 2, "new + changed note await embedding");
    home.colibri(&["update", "notes", "--json"]).assert_ok();

    // AC-014.1 and AC-006.2: status per collection.
    let status = json(&home.colibri(&["status", "--json"]));
    assert_eq!(status["index_ready"], true);
    assert_eq!(status["orphan_chunks"], 0);
    let notes = collection(&status, "notes");
    assert_eq!(notes["kind"], "mirror");
    assert_eq!(notes["active"], 3);
    assert_eq!(notes["removed"], 2);
    assert_eq!(notes["pending_index"], 0);
    assert_eq!(notes["last_run_status"], "ok");
    assert!(notes["problems"].as_array().unwrap().is_empty());
    let gone = collection(&status, "gone");
    assert_eq!(gone["last_run_status"], "error");
    assert_eq!(gone["problems"][0]["kind"], "root_missing");

    // Human-readable status works too.
    let out = home.colibri(&["status"]);
    out.assert_ok();
    assert!(out.stdout.contains("orphan chunks: 0"), "{}", out.stdout);
}
