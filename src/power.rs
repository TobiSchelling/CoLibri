//! Keep the machine awake while a long write command runs.
//!
//! A rebuild can take hours; macOS idle sleep would otherwise pause it for
//! most of the night (observed: 3 minutes of work per ~19 minutes). On macOS
//! this holds `caffeinate -i` for the lifetime of the command; the display
//! may still sleep. Elsewhere it does nothing.

use std::process::{Child, Command, Stdio};

/// Releases the assertion when dropped (or when the process exits).
pub struct KeepAwake(Option<Child>);

impl KeepAwake {
    #[cfg(test)]
    fn is_active(&mut self) -> bool {
        self.0
            .as_mut()
            .is_some_and(|c| matches!(c.try_wait(), Ok(None)))
    }
}

impl Drop for KeepAwake {
    fn drop(&mut self) {
        if let Some(child) = &mut self.0 {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

/// Prevent idle system sleep until the returned guard is dropped.
pub fn keep_awake() -> KeepAwake {
    if !cfg!(target_os = "macos") {
        return KeepAwake(None);
    }
    let child = Command::new("caffeinate")
        .args(["-i", "-w", &std::process::id().to_string()])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .ok();
    KeepAwake(child)
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;

    #[test]
    fn caffeinate_runs_while_the_guard_lives() {
        let mut guard = keep_awake();
        assert!(guard.is_active());
        let pid = guard.0.as_ref().unwrap().id();
        drop(guard);
        let alive = Command::new("kill")
            .args(["-0", &pid.to_string()])
            .status()
            .is_ok_and(|s| s.success());
        assert!(!alive, "caffeinate stops with the guard");
    }
}
