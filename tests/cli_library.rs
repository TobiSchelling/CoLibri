//! Library behaviour through the CLI with real EPUBs (built with pandoc).

mod common;

use std::path::Path;
use std::process::Command;

use common::{FakeOllama, TestHome};
use serde_json::Value;

fn pandoc_available() -> bool {
    let ok = Command::new("pandoc")
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success());
    if !ok && std::env::var("CI").is_ok() {
        panic!("pandoc is required for the library tests on CI");
    }
    ok
}

fn epub(path: &Path, title: &str, body: &str) {
    let md = path.with_extension("md");
    std::fs::write(&md, format!("# {title}\n\n{body}\n")).unwrap();
    let out = Command::new("pandoc")
        .arg(&md)
        .args(["--metadata", &format!("title={title}"), "-o"])
        .arg(path)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    std::fs::remove_file(md).unwrap();
}

fn calibre_book(root: &Path, rel: &str, uuid: &str, title: &str, body: &str) -> std::path::PathBuf {
    let dir = root.join(rel);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("metadata.opf"),
        format!(
            r#"<?xml version='1.0' encoding='utf-8'?>
<package xmlns="http://www.idpf.org/2007/opf" version="2.0">
<metadata xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:opf="http://www.idpf.org/2007/opf">
<dc:identifier opf:scheme="uuid">{uuid}</dc:identifier><dc:title>{title}</dc:title>
<dc:creator opf:role="aut">Scott Chacon</dc:creator><dc:creator opf:role="aut">Ben Straub</dc:creator>
<dc:language>eng</dc:language></metadata></package>"#
        ),
    )
    .unwrap();
    epub(&dir.join(format!("{title}.epub")), title, body);
    dir
}

fn json(out: &common::Output) -> Value {
    out.assert_ok();
    serde_json::from_str(&out.stdout).unwrap_or_else(|e| panic!("{e}: {}", out.stdout))
}

fn hits(home: &TestHome, query: &str) -> u64 {
    let out = home.colibri(&[
        "search",
        query,
        "--mode",
        "keyword",
        "--collection",
        "books",
        "--json",
    ]);
    json(&out)["total_results"].as_u64().unwrap()
}

#[test]
fn add_list_remove_restore_and_status() {
    if !pandoc_available() {
        eprintln!("skipping: pandoc not installed");
        return;
    }
    let ollama = FakeOllama::start();
    let books = tempfile::TempDir::new().unwrap();
    let home = TestHome::with_extra(
        &ollama,
        &format!(
            "library:\n  roots:\n    - path: {}\n",
            books.path().display()
        ),
    );
    let dir = calibre_book(
        books.path(),
        "Scott Chacon/Pro Git (90)",
        "uuid-progit",
        "Pro Git",
        "Branching with octopus merges.",
    );
    std::fs::write(dir.join("Pro Git.original_epub"), "ignored").unwrap();

    // AC-017.1 / AC-020.1: sweep adds the calibre book with its metadata, once.
    let report = json(&home.colibri(&["add", "--json"]));
    assert_eq!(report["added"], 1);
    let list = json(&home.colibri(&["list", "--json"]));
    assert_eq!(list[0]["title"], "Pro Git");
    assert_eq!(
        list[0]["authors"],
        serde_json::json!(["Scott Chacon", "Ben Straub"])
    );
    assert_eq!(list[0]["format"], "epub");
    assert!(list[0]["chunks"].as_i64().unwrap() > 0);
    assert_eq!(hits(&home, "octopus"), 1);
    let report = json(&home.colibri(&["add", "--json"]));
    assert_eq!(
        (report["added"].as_u64(), report["unchanged"].as_u64()),
        (Some(0), Some(1))
    );

    // AC-018.1: an explicit file is searchable right away; repeating it is a no-op.
    let loose = books.path().parent().unwrap().join("Loose Notes.epub");
    epub(&loose, "Loose Notes", "Zettelkasten with quokka examples.");
    let out = home.colibri(&["add", loose.to_str().unwrap()]);
    out.assert_ok();
    assert_eq!(hits(&home, "quokka"), 1);
    let out = home.colibri(&["add", loose.to_str().unwrap()]);
    out.assert_ok();
    assert!(out.stderr.contains("already in library"), "{}", out.stderr);

    // AC-021.1: remove hides it at once; sweeps skip it; explicit add restores it.
    let out = home.colibri(&["remove", "Pro Git"]);
    out.assert_ok();
    assert_eq!(hits(&home, "octopus"), 0);
    let report = json(&home.colibri(&["add", "--json"]));
    assert_eq!(report["skipped_removed"], 1);
    assert_eq!(hits(&home, "octopus"), 0);
    let out = home.colibri(&["add", dir.join("Pro Git.epub").to_str().unwrap()]);
    out.assert_ok();
    assert!(out.stderr.contains("restored"), "{}", out.stderr);
    assert_eq!(hits(&home, "octopus"), 1);

    // AC-022.1 and AC-014.1 (library row): a vanished file stays searchable.
    std::fs::remove_dir_all(&dir).unwrap();
    let report = json(&home.colibri(&["update", "books", "--json"]));
    assert_eq!(report["library"]["source_missing"], 1);
    assert_eq!(hits(&home, "octopus"), 1);
    let status = json(&home.colibri(&["status", "--json"]));
    let lib = status["collections"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == "books")
        .unwrap();
    assert_eq!(lib["kind"], "library");
    assert_eq!(lib["active"], 2);
    assert_eq!(lib["source_missing"], 1);
    assert_eq!(lib["last_run_status"], "ok");
    assert_eq!(status["orphan_chunks"], 0);
}
