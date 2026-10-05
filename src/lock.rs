//! Exclusive write lock for commands that modify CoLibri's data.
//!
//! Readers (search, serve) never take it. The lock is an OS file lock on
//! `<home>/write.lock`, released when the [`WriteLock`] is dropped or the
//! process exits. The holder's PID is written into the file for diagnostics.

use std::fs::{File, OpenOptions, TryLockError};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::error::ColibriError;

pub struct WriteLock {
    _file: File,
}

impl WriteLock {
    /// Take the lock or fail immediately if another process holds it.
    pub fn acquire(lock_path: &Path) -> Result<Self, ColibriError> {
        if let Some(parent) = lock_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(lock_path)?;

        match file.try_lock() {
            Ok(()) => {}
            Err(TryLockError::WouldBlock) => {
                let mut holder = String::new();
                let _ = file.read_to_string(&mut holder);
                let holder = holder.trim();
                let who = if holder.is_empty() {
                    "another process".to_string()
                } else {
                    format!("pid {holder}")
                };
                return Err(ColibriError::Config(format!(
                    "Another colibri command is changing the data ({who}). \
                     Wait for it to finish and retry."
                )));
            }
            Err(TryLockError::Error(e)) => return Err(ColibriError::Io(e)),
        }

        file.set_len(0)?;
        file.seek(SeekFrom::Start(0))?;
        write!(file, "{}", std::process::id())?;
        file.flush()?;
        Ok(Self { _file: file })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    // AC-003.1: a second writer fails fast and names the holder.
    #[test]
    fn second_writer_fails_fast_with_holder_pid() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("write.lock");
        let _held = WriteLock::acquire(&path).unwrap();

        let started = std::time::Instant::now();
        let err = WriteLock::acquire(&path)
            .err()
            .expect("second lock must fail");
        assert!(started.elapsed() < std::time::Duration::from_secs(1));
        let msg = err.to_string();
        assert!(
            msg.contains(&format!("pid {}", std::process::id())),
            "{msg}"
        );
    }

    #[test]
    fn lock_is_released_on_drop() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("write.lock");
        drop(WriteLock::acquire(&path).unwrap());
        assert!(WriteLock::acquire(&path).is_ok());
    }
}
