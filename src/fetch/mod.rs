//! Fetchers: bring remote content into a mirror folder as markdown files.
//!
//! A fetcher owns its folder. It only writes into folders carrying the
//! [`MARKER`] file (created when the folder is new or empty), never rewrites
//! a file whose bytes are unchanged, and only deletes files it wrote itself
//! after a complete listing. The mirror reconcile then ingests the folder
//! like any other.

pub mod command;
pub mod zephyr;

use std::path::Path;

use serde::Serialize;

use crate::ingest::mirror::Problem;

/// Marks a folder as owned by a CoLibri fetcher.
pub const MARKER: &str = ".colibri-mirror";

/// Outcome of one fetch.
#[derive(Debug, Clone, Default, Serialize)]
pub struct FetchReport {
    /// Items listed by the source.
    pub listed: usize,
    pub written: usize,
    pub unchanged: usize,
    pub deleted: usize,
    /// The listing is known to be complete (deletions were allowed).
    pub complete: bool,
    pub problems: Vec<Problem>,
}

/// Make sure `dir` can be owned by a fetcher: create it (with marker) if
/// missing, adopt it if empty, refuse if it holds files of someone else.
pub fn prepare_folder(dir: &Path) -> Result<(), String> {
    if dir.join(MARKER).is_file() {
        return Ok(());
    }
    if dir.exists() {
        let foreign = std::fs::read_dir(dir)
            .map_err(|e| format!("{}: {e}", dir.display()))?
            .flatten()
            .any(|e| e.file_name() != ".DS_Store");
        if foreign {
            return Err(format!(
                "{} is not empty and not managed by CoLibri (no {MARKER} file); refusing to write into it",
                dir.display()
            ));
        }
    }
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    std::fs::write(
        dir.join(MARKER),
        "This folder is written by CoLibri. Edits are overwritten on the next fetch.\n",
    )
    .map_err(|e| format!("{}: {e}", dir.display()))
}

/// Write `content` atomically unless the file already has exactly these
/// bytes. Returns whether the file was written.
pub fn write_if_changed(path: &Path, content: &str) -> std::io::Result<bool> {
    if std::fs::read(path).is_ok_and(|old| old == content.as_bytes()) {
        return Ok(false);
    }
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, content)?;
    std::fs::rename(&tmp, path)?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    // AC-027.1
    #[test]
    fn foreign_non_empty_folders_are_refused_and_left_alone() {
        let dir = tempfile::TempDir::new().unwrap();
        let foreign = dir.path().join("notes");
        std::fs::create_dir_all(&foreign).unwrap();
        std::fs::write(foreign.join("mine.md"), "keep").unwrap();
        let err = prepare_folder(&foreign).unwrap_err();
        assert!(err.contains("refusing"), "{err}");
        assert!(!foreign.join(MARKER).exists());
        assert_eq!(
            std::fs::read_to_string(foreign.join("mine.md")).unwrap(),
            "keep"
        );

        let fresh = dir.path().join("zephyr");
        prepare_folder(&fresh).unwrap();
        assert!(fresh.join(MARKER).is_file());
        std::fs::write(fresh.join("T-1.md"), "x").unwrap();
        prepare_folder(&fresh).unwrap();
    }

    // AC-027.2 (write part)
    #[test]
    fn unchanged_content_is_not_rewritten() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("a.md");
        assert!(write_if_changed(&path, "one").unwrap());
        let mtime = std::fs::metadata(&path).unwrap().modified().unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));
        assert!(!write_if_changed(&path, "one").unwrap());
        assert_eq!(std::fs::metadata(&path).unwrap().modified().unwrap(), mtime);
        assert!(write_if_changed(&path, "two").unwrap());
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "two");
    }
}
