//! Script fetcher: any command that writes files into the mirror folder.
//!
//! `{path}` in the arguments is replaced by the folder. A non-zero exit or
//! a run longer than [`TIMEOUT`] marks the fetch as failed, which disables
//! pruning for that mirror.

use std::path::Path;
use std::process::Command;
use std::time::Duration;

use super::{prepare_folder, FetchReport};
use crate::ingest::convert::run_with_timeout;
use crate::ingest::mirror::Problem;

/// A fetch command still running after this is stopped.
pub const TIMEOUT: Duration = Duration::from_secs(60 * 60);

pub fn fetch(run: &[String], dir: &Path) -> FetchReport {
    fetch_with_timeout(run, dir, TIMEOUT)
}

fn fetch_with_timeout(run: &[String], dir: &Path, timeout: Duration) -> FetchReport {
    let mut report = FetchReport::default();
    if let Err(e) = prepare_folder(dir) {
        report
            .problems
            .push(Problem::new(dir.display().to_string(), "fetch_failed", e));
        return report;
    }
    let Some((program, args)) = run.split_first() else {
        report.problems.push(Problem::new(
            dir.display().to_string(),
            "fetch_failed",
            "fetch.run is empty",
        ));
        return report;
    };
    let folder = dir.display().to_string();
    let args: Vec<String> = args.iter().map(|a| a.replace("{path}", &folder)).collect();
    let mut cmd = Command::new(program.replace("{path}", &folder));
    cmd.args(&args);
    match run_with_timeout(cmd, timeout) {
        Ok(out) if out.status.success() => report.complete = true,
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            let tail: String = stderr.lines().rev().take(3).collect::<Vec<_>>().join(" | ");
            report.problems.push(Problem::new(
                program.clone(),
                "fetch_failed",
                format!("exited with {}: {tail}", out.status),
            ));
        }
        Err(e) => report
            .problems
            .push(Problem::new(program.clone(), "fetch_failed", e)),
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sh(script: &str) -> Vec<String> {
        vec![
            "sh".into(),
            "-c".into(),
            script.into(),
            "fetch".into(),
            "{path}".into(),
        ]
    }

    #[test]
    fn successful_command_is_complete_and_failing_one_is_partial() {
        let dir = tempfile::TempDir::new().unwrap();
        let folder = dir.path().join("pages");
        let r = fetch(&sh("printf '# A' > \"$1/a.md\""), &folder);
        assert!(r.complete, "{:?}", r.problems);
        assert!(folder.join("a.md").exists());

        let r = fetch(&sh("rm \"$1/a.md\"; echo boom >&2; exit 3"), &folder);
        assert!(!r.complete);
        assert_eq!(r.problems[0].kind, "fetch_failed");
        assert!(r.problems[0].message.contains("boom"));
    }

    #[test]
    fn hung_commands_are_stopped() {
        let dir = tempfile::TempDir::new().unwrap();
        let r = fetch_with_timeout(
            &sh("sleep 30"),
            &dir.path().join("pages"),
            Duration::from_millis(300),
        );
        assert!(!r.complete);
        assert!(
            r.problems[0].message.contains("stopped"),
            "{:?}",
            r.problems
        );
    }
}
