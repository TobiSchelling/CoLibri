//! Walk a source folder and list the files a mirror or library should see.
//!
//! The walk never stops at the first unreadable directory; instead it marks
//! the result as incomplete so callers can refuse to prune on partial data.

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use glob::{MatchOptions, Pattern};

/// macOS `SF_DATALESS`: the file is an online-only cloud placeholder.
const SF_DATALESS: u32 = 0x4000_0000;

/// Case-insensitive like the macOS file system, so `**/*.pdf` also matches `Scan.PDF`.
const MATCH: MatchOptions = MatchOptions {
    case_sensitive: false,
    require_literal_separator: true,
    require_literal_leading_dot: false,
};

/// A file that matched the include patterns.
#[derive(Debug, Clone)]
pub struct FoundFile {
    /// Path relative to the walk root, with `/` separators.
    pub rel: String,
    pub abs: PathBuf,
    pub size: u64,
    pub mtime: SystemTime,
    /// Online-only cloud placeholder; reading it would trigger a download.
    pub online_only: bool,
}

#[derive(Debug, Default)]
pub struct WalkResult {
    pub files: Vec<FoundFile>,
    /// False when any directory or entry could not be read.
    pub complete: bool,
    /// (path relative to root, message) for everything that could not be read.
    pub unreadable: Vec<(String, String)>,
    /// Skipped entries that do not make the walk incomplete (broken symlinks,
    /// symlink loops).
    pub notices: Vec<(String, String)>,
}

/// Compiled include/exclude patterns, matched case-insensitively against the
/// path relative to the root. `*` does not cross `/`; `**/` matches any depth,
/// including none (`**/*.md` matches `a.md`).
pub struct Filter {
    include: Vec<Pattern>,
    exclude: Vec<Pattern>,
}

impl Filter {
    /// Invalid patterns are skipped (config validation reports them earlier).
    pub fn new(include: &[String], exclude: &[String]) -> Self {
        let compile = |ps: &[String]| -> Vec<Pattern> {
            ps.iter().filter_map(|p| Pattern::new(p).ok()).collect()
        };
        Self {
            include: compile(include),
            exclude: compile(exclude),
        }
    }

    fn excluded_file(&self, rel: &str) -> bool {
        self.exclude.iter().any(|p| p.matches_with(rel, MATCH))
    }

    /// A directory is skipped when a pattern matches its path (`notes/*`
    /// matches `notes/deep`, like gitignore) or covers its whole subtree
    /// (`.obsidian/**`, `**/node_modules/**`).
    fn excluded_dir(&self, rel: &str) -> bool {
        let probe = format!("{rel}/x");
        self.exclude.iter().any(|p| {
            p.matches_with(rel, MATCH)
                || (p.as_str().ends_with("/**") && p.matches_with(&probe, MATCH))
        })
    }

    fn included(&self, rel: &str) -> bool {
        self.include.iter().any(|p| p.matches_with(rel, MATCH))
    }
}

pub fn is_dataless_flags(flags: u32) -> bool {
    flags & SF_DATALESS != 0
}

#[cfg(target_os = "macos")]
fn is_online_only(meta: &std::fs::Metadata) -> bool {
    use std::os::macos::fs::MetadataExt;
    is_dataless_flags(meta.st_flags())
}

#[cfg(not(target_os = "macos"))]
fn is_online_only(_meta: &std::fs::Metadata) -> bool {
    false
}

/// List matching files below `root`. Symlinks are followed; a directory
/// reached twice (symlink loop) is visited once.
pub fn walk(root: &Path, filter: &Filter) -> WalkResult {
    let mut result = WalkResult {
        complete: true,
        ..Default::default()
    };
    let mut visited = std::collections::HashSet::new();
    if let Ok(canonical) = root.canonicalize() {
        visited.insert(canonical);
    }
    let mut stack = vec![(root.to_path_buf(), String::new())];
    while let Some((dir, rel_dir)) = stack.pop() {
        let entries = match std::fs::read_dir(&dir) {
            Ok(e) => e,
            Err(e) => {
                result.complete = false;
                result.unreadable.push((rel_dir, e.to_string()));
                continue;
            }
        };
        // Sorted, so the path through which a directory is first reached
        // (and thus every document id below it) is the same on every run.
        let mut sorted = Vec::new();
        for entry in entries {
            match entry {
                Ok(e) => sorted.push(e),
                Err(e) => {
                    result.complete = false;
                    result.unreadable.push((rel_dir.clone(), e.to_string()));
                }
            }
        }
        sorted.sort_by_key(|e| e.file_name());
        for entry in sorted {
            let name = entry.file_name().to_string_lossy().to_string();
            let rel = if rel_dir.is_empty() {
                name
            } else {
                format!("{rel_dir}/{name}")
            };
            let path = entry.path();
            // Follows symlinks.
            let meta = match std::fs::metadata(&path) {
                Ok(m) => m,
                Err(e) => {
                    let is_link = std::fs::symlink_metadata(&path)
                        .map(|m| m.file_type().is_symlink())
                        .unwrap_or(false);
                    if is_link && e.kind() == std::io::ErrorKind::NotFound {
                        result.notices.push((rel, "broken symlink".into()));
                    } else {
                        result.complete = false;
                        result.unreadable.push((rel, e.to_string()));
                    }
                    continue;
                }
            };
            if meta.is_dir() {
                if filter.excluded_dir(&rel) {
                    continue;
                }
                let first_visit = path
                    .canonicalize()
                    .map(|canonical| visited.insert(canonical))
                    .unwrap_or(true);
                if first_visit {
                    stack.push((path, rel));
                } else {
                    result
                        .notices
                        .push((rel, "directory already visited (symlink loop)".into()));
                }
                continue;
            }
            if !meta.is_file() || filter.excluded_file(&rel) || !filter.included(&rel) {
                continue;
            }
            result.files.push(FoundFile {
                online_only: is_online_only(&meta),
                size: meta.len(),
                mtime: meta.modified().unwrap_or(SystemTime::UNIX_EPOCH),
                abs: path,
                rel,
            });
        }
    }
    result.files.sort_by(|a, b| a.rel.cmp(&b.rel));
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tree() -> tempfile::TempDir {
        let dir = tempfile::TempDir::new().unwrap();
        for rel in [
            "a.md",
            "notes/b.md",
            "notes/deep/c.md",
            "notes/skip.txt",
            ".obsidian/workspace.md",
            "node/node_modules/x/readme.md",
            "arch/model.yaml",
        ] {
            let p = dir.path().join(rel);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, "x").unwrap();
        }
        dir
    }

    fn rels(r: &WalkResult) -> Vec<&str> {
        r.files.iter().map(|f| f.rel.as_str()).collect()
    }

    #[test]
    fn include_and_exclude_patterns() {
        let dir = tree();
        let f = Filter::new(
            &["**/*.md".into()],
            &[".obsidian/**".into(), "**/node_modules/**".into()],
        );
        let r = walk(dir.path(), &f);
        assert!(r.complete);
        assert_eq!(rels(&r), ["a.md", "notes/b.md", "notes/deep/c.md"]);

        let f = Filter::new(
            &["**/*.md".into(), "**/*.yaml".into()],
            &["notes/deep/**".into()],
        );
        let r = walk(dir.path(), &f);
        assert!(rels(&r).contains(&"arch/model.yaml"));
        assert!(!rels(&r).contains(&"notes/deep/c.md"));
    }

    #[test]
    fn root_level_pattern_does_not_cross_directories() {
        let dir = tree();
        let r = walk(dir.path(), &Filter::new(&["*.md".into()], &[]));
        assert_eq!(rels(&r), ["a.md"]);
    }

    #[test]
    fn matching_is_case_insensitive_and_subtree_excludes_need_double_star() {
        let dir = tree();
        std::fs::write(dir.path().join("Scan.PDF"), "x").unwrap();
        let r = walk(dir.path(), &Filter::new(&["**/*.pdf".into()], &[]));
        assert_eq!(rels(&r), ["Scan.PDF"]);

        // A file pattern ending in `x` must not make every directory look
        // excluded (the subtree probe only applies to `/**` patterns).
        let r = walk(
            dir.path(),
            &Filter::new(&["**/*.md".into()], &["**/*x".into()]),
        );
        assert!(rels(&r).contains(&"notes/deep/c.md"));

        // Like gitignore, `notes/*` excludes everything directly in notes/,
        // including the `notes/deep/` directory.
        let r = walk(
            dir.path(),
            &Filter::new(&["**/*.md".into()], &["notes/*".into()]),
        );
        assert!(
            rels(&r).iter().all(|p| !p.starts_with("notes/")),
            "{:?}",
            rels(&r)
        );
        assert!(rels(&r).contains(&"a.md"));
    }

    #[test]
    fn symlinks_are_followed_and_loops_and_broken_links_do_not_block() {
        let dir = tree();
        let link = dir.path().join("linked-notes");
        std::os::unix::fs::symlink(dir.path().join("notes"), &link).unwrap();
        std::os::unix::fs::symlink(dir.path(), dir.path().join("notes/loop")).unwrap();
        std::os::unix::fs::symlink(
            dir.path().join("missing.md"),
            dir.path().join("dangling.md"),
        )
        .unwrap();
        let r = walk(dir.path(), &Filter::new(&["**/*.md".into()], &[]));
        assert!(r.complete, "{:?}", r.unreadable);
        assert!(rels(&r).contains(&"a.md"));
        // notes/ is reached once, either directly or through linked-notes/.
        assert_eq!(rels(&r).iter().filter(|p| p.ends_with("/b.md")).count(), 1);
        assert!(r
            .notices
            .iter()
            .any(|(p, m)| p == "dangling.md" && m == "broken symlink"));
    }

    // AC-011.2 (walk part): an unreadable directory makes the walk incomplete.
    #[test]
    fn unreadable_directory_marks_walk_incomplete() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tree();
        let locked = dir.path().join("notes");
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000)).unwrap();
        let r = walk(dir.path(), &Filter::new(&["**/*.md".into()], &[]));
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o755)).unwrap();
        assert!(!r.complete);
        assert_eq!(r.unreadable[0].0, "notes");
        assert!(rels(&r).contains(&"a.md"));
    }

    // AC-016.1
    #[test]
    fn dataless_flag_detection() {
        assert!(is_dataless_flags(0x4000_0000));
        assert!(is_dataless_flags(0x4000_0020));
        assert!(!is_dataless_flags(0x0000_0020));
        assert!(!is_dataless_flags(0));
    }
}
