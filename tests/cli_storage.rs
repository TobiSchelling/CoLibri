//! End-to-end CLI tests against a temp data dir and a fake Ollama server.
//!
//! Covers update -> index -> search -> doctor -> reset with the real binary.

mod common;

use common::{FakeOllama, TestHome};
use serde_json::Value;

#[test]
fn sync_search_doctor_reset_round_trip() {
    let ollama = FakeOllama::start();
    let home = TestHome::new(&ollama);
    home.write_source(
        "projects/alpha.md",
        "---\nstatus: active\n---\n# Alpha\nThe alpha plan.\n",
    );
    home.write_source("notes/beta.md", "# Beta\nBeta notes mention alpha once.\n");

    // AC-002.2: before any data exists, a read command fails and creates nothing.
    let out = home.colibri(&["search", "alpha"]);
    assert!(!out.success, "search without data must fail");
    assert!(
        !home.data_dir().exists(),
        "read path must not create the data dir"
    );

    home.colibri(&["update"]).assert_ok();

    // Keyword, semantic and hybrid search all return documents of the collection.
    for mode in ["keyword", "semantic", "hybrid"] {
        let out = home.colibri(&[
            "search",
            "alpha",
            "--mode",
            mode,
            "--json",
            "--group-by-doc",
        ]);
        out.assert_ok();
        let json: Value = serde_json::from_str(&out.stdout).unwrap();
        let results = json["results"].as_array().unwrap();
        assert!(!results.is_empty(), "{mode}: {}", out.stdout);
        assert!(results.iter().all(|r| r["collection"] == "notes"));
    }

    // Frontmatter is filterable; path filters use the source-relative path.
    let out = home.colibri(&[
        "search",
        "alpha",
        "--mode",
        "keyword",
        "--json",
        "--frontmatter",
        "status=active",
    ]);
    out.assert_ok();
    let json: Value = serde_json::from_str(&out.stdout).unwrap();
    assert_eq!(json["total_results"], 1);
    assert!(json["results"][0]["file"]
        .as_str()
        .unwrap()
        .ends_with("projects/alpha.md"));

    // An unchanged second update embeds nothing.
    let embeds_before = ollama.embed_requests();
    home.colibri(&["update"]).assert_ok();
    assert_eq!(ollama.embed_requests(), embeds_before);

    let out = home.colibri(&["doctor", "--json"]);
    out.assert_ok();
    let doctor: Value = serde_json::from_str(&out.stdout).unwrap();
    assert_eq!(doctor["metadata_documents"], 2);
    assert_eq!(doctor["index_queryable"], true);

    // AC-004.2: a wrong confirmation word deletes nothing.
    let out = home.colibri_with_stdin(&["reset"], "yes\n");
    out.assert_ok();
    assert!(out.stderr.contains("Aborted"), "{}", out.stderr);
    assert!(home.data_dir().join("metadata.db").exists());

    // AC-004.1: reset removes CoLibri's data but not sources or config.
    home.colibri_with_stdin(&["reset"], "reset\n").assert_ok();
    assert!(!home.data_dir().join("metadata.db").exists());
    assert!(!home.data_dir().join("index").exists());
    assert!(home.source_dir().join("projects/alpha.md").exists());
    assert!(home.config_path().exists());

    // After reset, reads fail again without recreating the metadata DB.
    let out = home.colibri(&["search", "alpha"]);
    assert!(!out.success);
    assert!(!home.data_dir().join("metadata.db").exists());
}

#[test]
fn old_metadata_db_is_reported_and_left_unchanged() {
    let ollama = FakeOllama::start();
    let home = TestHome::new(&ollama);
    home.write_source("a.md", "# A\n");
    std::fs::create_dir_all(home.data_dir()).unwrap();
    let db = home.data_dir().join("metadata.db");
    {
        let conn = rusqlite::Connection::open(&db).unwrap();
        conn.execute_batch(
            "CREATE TABLE documents (doc_id TEXT PRIMARY KEY, classification TEXT);",
        )
        .unwrap();
    }
    let original = std::fs::read(&db).unwrap();

    let out = home.colibri(&["update"]);
    assert!(!out.success);
    assert!(out.stderr.contains("colibri reset"), "{}", out.stderr);
    assert_eq!(std::fs::read(&db).unwrap(), original);
}
