//! Bundled plugin installation for binary distributions.
//!
//! CoLibri ships a small set of plugins as embedded assets. On first run (or on upgrade),
//! we materialize those assets into `<COLIBRI_HOME>/plugins/...` so users can reference
//! stable manifest paths without having a checkout of the repository.

use std::path::{Path, PathBuf};

use crate::error::ColibriError;

struct BundledFile {
    rel_path: &'static str,
    bytes: &'static [u8],
    executable: bool,
}

fn bundled_files() -> &'static [BundledFile] {
    &[
        // filesystem_documents plugin
        BundledFile {
            rel_path: "bundled/filesystem_documents/README.md",
            bytes: include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/plugins/bundled/filesystem_documents/README.md"
            )),
            executable: false,
        },
        BundledFile {
            rel_path: "bundled/filesystem_documents/bootstrap.sh",
            bytes: include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/plugins/bundled/filesystem_documents/bootstrap.sh"
            )),
            executable: true,
        },
        BundledFile {
            rel_path: "bundled/filesystem_documents/plugin.py",
            bytes: include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/plugins/bundled/filesystem_documents/plugin.py"
            )),
            executable: false,
        },
        BundledFile {
            rel_path: "bundled/filesystem_documents/plugin_manifest.json",
            bytes: include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/plugins/bundled/filesystem_documents/plugin_manifest.json"
            )),
            executable: false,
        },
        BundledFile {
            rel_path: "bundled/filesystem_documents/requirements.txt",
            bytes: include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/plugins/bundled/filesystem_documents/requirements.txt"
            )),
            executable: false,
        },
        BundledFile {
            rel_path: "bundled/filesystem_documents/run.sh",
            bytes: include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/plugins/bundled/filesystem_documents/run.sh"
            )),
            executable: true,
        },
        // python-sdk used by plugin.py (sys.path hack expects ../python-sdk)
        BundledFile {
            rel_path: "python-sdk/README.md",
            bytes: include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/plugins/python-sdk/README.md"
            )),
            executable: false,
        },
        BundledFile {
            rel_path: "python-sdk/pyproject.toml",
            bytes: include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/plugins/python-sdk/pyproject.toml"
            )),
            executable: false,
        },
        BundledFile {
            rel_path: "python-sdk/colibri_plugin_sdk/__init__.py",
            bytes: include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/plugins/python-sdk/colibri_plugin_sdk/__init__.py"
            )),
            executable: false,
        },
        BundledFile {
            rel_path: "python-sdk/colibri_plugin_sdk/envelope.py",
            bytes: include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/plugins/python-sdk/colibri_plugin_sdk/envelope.py"
            )),
            executable: false,
        },
    ]
}

fn bundled_version_path(colibri_home: &Path) -> PathBuf {
    colibri_home.join("plugins").join(".bundled_version")
}

fn ensure_parent(path: &Path) -> Result<(), ColibriError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    Ok(())
}

#[cfg(unix)]
fn set_executable(path: &Path) -> Result<(), ColibriError> {
    use std::os::unix::fs::PermissionsExt;
    let meta = std::fs::metadata(path)?;
    let mut perms = meta.permissions();
    let mode = perms.mode();
    perms.set_mode(mode | 0o111);
    std::fs::set_permissions(path, perms)?;
    Ok(())
}

#[cfg(not(unix))]
fn set_executable(_path: &Path) -> Result<(), ColibriError> {
    Ok(())
}

/// Ensure bundled plugins are installed into `<COLIBRI_HOME>/plugins`.
///
/// Re-installs when the stored bundled version differs from the current binary version.
pub fn ensure_bundled_plugins(colibri_home: &Path) -> Result<(), ColibriError> {
    let plugins_root = colibri_home.join("plugins");
    std::fs::create_dir_all(&plugins_root)?;

    let version_path = bundled_version_path(colibri_home);
    let want_version = env!("CARGO_PKG_VERSION");
    let have_version = std::fs::read_to_string(&version_path)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());

    let needs_install = have_version.as_deref() != Some(want_version);
    if !needs_install {
        return Ok(());
    }

    for f in bundled_files() {
        let dst = plugins_root.join(f.rel_path);
        ensure_parent(&dst)?;
        std::fs::write(&dst, f.bytes)?;
        if f.executable {
            set_executable(&dst)?;
        }
    }

    std::fs::write(&version_path, format!("{want_version}\n"))?;
    Ok(())
}

pub fn filesystem_documents_manifest_path(colibri_home: &Path) -> PathBuf {
    colibri_home
        .join("plugins")
        .join("bundled/filesystem_documents/plugin_manifest.json")
}
