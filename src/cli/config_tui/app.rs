//! Application state for the TUI configuration editor.

use std::path::{Path, PathBuf};

use crate::config::{save_config, AppConfig, FolderProfile, IndexMode, SCHEMA_VERSION};
use crate::embedding::check_ollama;
use crate::index_meta::read_index_meta;
use crate::indexer::{index_library, IndexResult};
use crate::manifest::{get_manifest_path, Manifest};

/// Which screen is currently active.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Screen {
    FolderList,
    EditFolder,
    AddFolder,
    Indexing,
    Status,
}

/// Field being edited in EditFolder/AddFolder screens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EditField {
    Path,
    Name,
    DocType,
    Mode,
    Extensions,
    ChunkSize,
    ChunkOverlap,
}

impl EditField {
    pub fn all() -> &'static [EditField] {
        &[
            EditField::Path,
            EditField::Name,
            EditField::DocType,
            EditField::Mode,
            EditField::Extensions,
            EditField::ChunkSize,
            EditField::ChunkOverlap,
        ]
    }

    pub fn label(&self) -> &'static str {
        match self {
            EditField::Path => "Path",
            EditField::Name => "Name",
            EditField::DocType => "Doc Type",
            EditField::Mode => "Mode",
            EditField::Extensions => "Extensions",
            EditField::ChunkSize => "Chunk Size",
            EditField::ChunkOverlap => "Chunk Overlap",
        }
    }
}

/// Indexing progress state.
#[derive(Debug, Clone, Default)]
pub struct IndexingProgress {
    pub folder: Option<String>,
    pub force: bool,
    pub is_running: bool,
    pub message: String,
    pub result: Option<IndexResult>,
    pub error: Option<String>,
}

/// Status information about the system.
#[derive(Debug, Clone, Default)]
pub struct StatusInfo {
    pub schema_version: u32,
    pub stored_version: u32,
    pub file_count: u64,
    pub chunk_count: u64,
    pub embedding_model: String,
    pub last_indexed: Option<String>,
    pub ollama_status: OllamaStatus,
    pub config_path: String,
    pub data_dir: String,
    pub lancedb_dir: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum OllamaStatus {
    #[default]
    Unknown,
    Checking,
    Connected(String),
    Disconnected(String),
    Error(String),
}

/// Path completion state.
#[derive(Debug, Clone, Default)]
pub struct PathCompletion {
    /// Available completions for current input
    pub completions: Vec<String>,
    /// Currently selected completion index
    pub selected: usize,
    /// Whether completion popup is visible
    pub is_active: bool,
}

/// Main application state.
pub struct App {
    pub config: AppConfig,
    pub screen: Screen,
    pub quit: bool,

    // Folder list state
    pub selected_folder: usize,

    // Edit/Add folder state
    pub editing_folder: Option<FolderProfile>,
    pub editing_index: Option<usize>, // None for new folder
    pub edit_field: usize,
    pub edit_input: String,
    pub is_editing_field: bool,

    // Path completion state
    pub path_completion: PathCompletion,

    // Indexing state
    pub indexing: IndexingProgress,

    // Status state
    pub status: StatusInfo,
    pub status_loaded: bool,

    // Error display
    pub error_message: Option<String>,

    // Track if config has unsaved changes
    pub has_changes: bool,
}

impl App {
    pub fn new(config: AppConfig) -> Self {
        let config_path = AppConfig::config_path().display().to_string();
        let data_dir = config.data_dir.display().to_string();
        let lancedb_dir = config.lancedb_dir.display().to_string();

        Self {
            config,
            screen: Screen::FolderList,
            quit: false,
            selected_folder: 0,
            editing_folder: None,
            editing_index: None,
            edit_field: 0,
            edit_input: String::new(),
            is_editing_field: false,
            path_completion: PathCompletion::default(),
            indexing: IndexingProgress::default(),
            status: StatusInfo {
                schema_version: SCHEMA_VERSION,
                config_path,
                data_dir,
                lancedb_dir,
                ..Default::default()
            },
            status_loaded: false,
            error_message: None,
            has_changes: false,
        }
    }

    pub fn should_quit(&self) -> bool {
        self.quit
    }

    pub fn set_error(&mut self, msg: String) {
        self.error_message = Some(msg);
    }

    pub fn clear_error(&mut self) {
        self.error_message = None;
    }

    // Navigation
    pub fn next_folder(&mut self) {
        if !self.config.sources.is_empty() {
            self.selected_folder = (self.selected_folder + 1) % self.config.sources.len();
        }
    }

    pub fn previous_folder(&mut self) {
        if !self.config.sources.is_empty() {
            self.selected_folder = self
                .selected_folder
                .checked_sub(1)
                .unwrap_or(self.config.sources.len() - 1);
        }
    }

    // Screen transitions
    pub fn go_to_folder_list(&mut self) {
        self.screen = Screen::FolderList;
        self.editing_folder = None;
        self.editing_index = None;
        self.is_editing_field = false;
    }

    pub fn go_to_edit_folder(&mut self) {
        if self.config.sources.is_empty() {
            return;
        }
        let folder = self.config.sources[self.selected_folder].clone();
        self.editing_folder = Some(folder);
        self.editing_index = Some(self.selected_folder);
        self.edit_field = 0;
        self.is_editing_field = false;
        self.screen = Screen::EditFolder;
    }

    pub fn go_to_add_folder(&mut self) {
        self.editing_folder = Some(FolderProfile::default());
        self.editing_index = None;
        self.edit_field = 0;
        // Start in edit mode on the path field with home directory prepopulated
        self.edit_input = dirs::home_dir()
            .map(|h| format!("{}/", h.display()))
            .unwrap_or_else(|| "/".to_string());
        self.is_editing_field = true;
        self.path_completion = PathCompletion::default();
        self.screen = Screen::AddFolder;
        // Trigger initial completions (shows home directory contents)
        self.trigger_path_completion();
    }

    pub fn go_to_status(&mut self) {
        self.screen = Screen::Status;
        self.status_loaded = false;
    }

    pub fn go_to_indexing(&mut self) {
        self.screen = Screen::Indexing;
    }

    // Edit field navigation
    pub fn next_field(&mut self) {
        let fields = EditField::all();
        self.edit_field = (self.edit_field + 1) % fields.len();
    }

    pub fn previous_field(&mut self) {
        let fields = EditField::all();
        self.edit_field = self.edit_field.checked_sub(1).unwrap_or(fields.len() - 1);
    }

    pub fn current_field(&self) -> EditField {
        EditField::all()[self.edit_field]
    }

    pub fn start_editing_field(&mut self) {
        if let Some(ref folder) = self.editing_folder {
            self.edit_input = match self.current_field() {
                EditField::Path => folder.path.clone(),
                EditField::Name => folder.name.clone().unwrap_or_default(),
                EditField::DocType => folder.doc_type.clone(),
                EditField::Mode => format_mode(folder.mode).to_string(),
                EditField::Extensions => folder.extensions.join(", "),
                EditField::ChunkSize => {
                    folder.chunk_size.map(|s| s.to_string()).unwrap_or_default()
                }
                EditField::ChunkOverlap => folder
                    .chunk_overlap
                    .map(|s| s.to_string())
                    .unwrap_or_default(),
            };
            self.is_editing_field = true;
        }
    }

    pub fn finish_editing_field(&mut self) {
        let field = self.current_field();
        let input = self.edit_input.trim().to_string();

        if let Some(ref mut folder) = self.editing_folder {
            match field {
                EditField::Path => {
                    // Remove trailing slash for storage
                    folder.path = input.trim_end_matches('/').to_string();
                }
                EditField::Name => {
                    folder.name = if input.is_empty() { None } else { Some(input) };
                }
                EditField::DocType => folder.doc_type = input,
                EditField::Mode => {
                    folder.mode = parse_mode(&input);
                }
                EditField::Extensions => {
                    folder.extensions = input
                        .split(',')
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                        .collect();
                }
                EditField::ChunkSize => {
                    folder.chunk_size = input.parse().ok();
                }
                EditField::ChunkOverlap => {
                    folder.chunk_overlap = input.parse().ok();
                }
            }
        }
        self.is_editing_field = false;
        self.edit_input.clear();
    }

    pub fn cancel_editing_field(&mut self) {
        self.is_editing_field = false;
        self.edit_input.clear();
        self.path_completion = PathCompletion::default();
    }

    // Path completion methods

    /// Trigger path completion based on current input.
    pub fn trigger_path_completion(&mut self) {
        if self.current_field() != EditField::Path {
            return;
        }

        let input = &self.edit_input;
        let completions = get_path_completions(input);

        if completions.is_empty() {
            self.path_completion.is_active = false;
            return;
        }

        // Always show the popup so user can see available options
        self.path_completion = PathCompletion {
            completions,
            selected: 0,
            is_active: true,
        };
    }

    /// Move to next completion.
    pub fn next_completion(&mut self) {
        if !self.path_completion.is_active || self.path_completion.completions.is_empty() {
            return;
        }
        self.path_completion.selected =
            (self.path_completion.selected + 1) % self.path_completion.completions.len();
    }

    /// Move to previous completion.
    pub fn previous_completion(&mut self) {
        if !self.path_completion.is_active || self.path_completion.completions.is_empty() {
            return;
        }
        self.path_completion.selected = self
            .path_completion
            .selected
            .checked_sub(1)
            .unwrap_or(self.path_completion.completions.len() - 1);
    }

    /// Apply selected completion to input and drill into folder.
    pub fn apply_completion(&mut self) {
        if !self.path_completion.is_active || self.path_completion.completions.is_empty() {
            return;
        }

        let selected = &self.path_completion.completions[self.path_completion.selected];
        self.edit_input = selected.clone();

        // Add trailing slash for directories and immediately show contents
        if Path::new(&self.edit_input).is_dir() && !self.edit_input.ends_with('/') {
            self.edit_input.push('/');
        }

        // Immediately trigger completions for the new directory
        self.path_completion.is_active = false;
        self.trigger_path_completion();
    }

    /// Apply completion and stay (don't drill down) - for Enter key.
    pub fn apply_completion_final(&mut self) {
        if !self.path_completion.is_active || self.path_completion.completions.is_empty() {
            return;
        }

        let selected = &self.path_completion.completions[self.path_completion.selected];
        self.edit_input = selected.clone();

        // Remove any trailing spaces
        self.edit_input = self.edit_input.trim_end().to_string();

        self.path_completion.is_active = false;
    }

    /// Cancel completion popup.
    pub fn cancel_completion(&mut self) {
        self.path_completion.is_active = false;
    }

    /// Check if currently editing the path field.
    pub fn is_editing_path(&self) -> bool {
        self.is_editing_field && self.current_field() == EditField::Path
    }

    pub fn cycle_mode(&mut self) {
        if let Some(ref mut folder) = self.editing_folder {
            folder.mode = match folder.mode {
                IndexMode::Static => IndexMode::Incremental,
                IndexMode::Incremental => IndexMode::AppendOnly,
                IndexMode::AppendOnly => IndexMode::Disabled,
                IndexMode::Disabled => IndexMode::Static,
            };
            self.has_changes = true;
        }
    }

    // Save edited folder
    pub fn save_folder(&mut self) {
        if let Some(folder) = self.editing_folder.take() {
            if folder.path.is_empty() {
                self.set_error("Path cannot be empty".into());
                self.editing_folder = Some(folder);
                return;
            }

            if let Some(idx) = self.editing_index {
                // Update existing
                self.config.sources[idx] = folder;
            } else {
                // Add new
                self.config.sources.push(folder);
                self.selected_folder = self.config.sources.len() - 1;
            }
            self.has_changes = true;
            self.go_to_folder_list();
        }
    }

    // Delete folder
    pub fn delete_folder(&mut self) {
        if self.config.sources.is_empty() {
            return;
        }
        self.config.sources.remove(self.selected_folder);
        if self.selected_folder >= self.config.sources.len() && !self.config.sources.is_empty() {
            self.selected_folder = self.config.sources.len() - 1;
        }
        self.has_changes = true;
    }

    // Save config to file
    pub fn save_config(&mut self) -> Result<(), crate::error::ColibriError> {
        save_config(&self.config)?;
        self.has_changes = false;
        Ok(())
    }

    // Indexing
    pub fn start_indexing(&mut self, folder: Option<String>, force: bool) {
        self.indexing = IndexingProgress {
            folder,
            force,
            is_running: true,
            message: "Starting indexing...".into(),
            result: None,
            error: None,
        };
        self.go_to_indexing();
    }

    /// Called periodically to update async operations.
    pub async fn tick(&mut self) {
        // Load status info if needed
        if self.screen == Screen::Status && !self.status_loaded {
            self.load_status().await;
            self.status_loaded = true;
        }

        // Run indexing if requested
        if self.indexing.is_running && self.screen == Screen::Indexing {
            self.run_indexing().await;
        }
    }

    async fn load_status(&mut self) {
        // Load index metadata
        if let Ok(meta) = read_index_meta(&self.config.lancedb_dir) {
            self.status.stored_version = meta
                .get("schema_version")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32;
            self.status.file_count = meta.get("file_count").and_then(|v| v.as_u64()).unwrap_or(0);
            self.status.chunk_count = meta
                .get("chunk_count")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            self.status.embedding_model = meta
                .get("embedding_model")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string();
            self.status.last_indexed = meta
                .get("last_indexed_at")
                .and_then(|v| v.as_str())
                .map(String::from);
        }

        // Check Ollama connection
        self.status.ollama_status = OllamaStatus::Checking;
        match check_ollama(&self.config.ollama_base_url).await {
            Ok(true) => {
                self.status.ollama_status =
                    OllamaStatus::Connected(self.config.ollama_base_url.clone());
            }
            Ok(false) => {
                self.status.ollama_status =
                    OllamaStatus::Disconnected(self.config.ollama_base_url.clone());
            }
            Err(e) => {
                self.status.ollama_status = OllamaStatus::Error(e.to_string());
            }
        }
    }

    async fn run_indexing(&mut self) {
        self.indexing.message = "Indexing in progress...".into();

        let result = index_library(
            &self.config,
            self.indexing.folder.as_deref(),
            self.indexing.force,
        )
        .await;

        self.indexing.is_running = false;

        match result {
            Ok(res) => {
                self.indexing.message = format!(
                    "Done: {} files indexed ({} chunks)",
                    res.files_indexed, res.total_chunks
                );
                self.indexing.result = Some(res);
            }
            Err(e) => {
                self.indexing.message = "Indexing failed".into();
                self.indexing.error = Some(e.to_string());
            }
        }
    }

    // Get folder stats from manifest
    pub fn get_folder_stats(&self, folder: &FolderProfile) -> (usize, usize) {
        let manifest_path = get_manifest_path(&self.config.data_dir);
        if let Ok(manifest) = Manifest::load(&manifest_path) {
            let source_id = crate::manifest::source_id_for_root(Path::new(&folder.path));
            let prefix = format!("{source_id}:");

            let mut file_count = 0;
            let mut chunk_count = 0;

            for (key, entry) in &manifest.files {
                if key.starts_with(&prefix) {
                    file_count += 1;
                    chunk_count += entry.chunk_count;
                }
            }

            (file_count, chunk_count)
        } else {
            (0, 0)
        }
    }
}

fn format_mode(mode: IndexMode) -> &'static str {
    match mode {
        IndexMode::Static => "static",
        IndexMode::Incremental => "incremental",
        IndexMode::AppendOnly => "append_only",
        IndexMode::Disabled => "disabled",
    }
}

fn parse_mode(s: &str) -> IndexMode {
    match s.to_lowercase().as_str() {
        "static" => IndexMode::Static,
        "incremental" => IndexMode::Incremental,
        "append_only" | "appendonly" => IndexMode::AppendOnly,
        "disabled" => IndexMode::Disabled,
        _ => IndexMode::Incremental,
    }
}

/// Get path completions for the given input.
pub fn get_path_completions(input: &str) -> Vec<String> {
    if input.is_empty() {
        // Start with home directory
        if let Some(home) = dirs::home_dir() {
            return vec![home.display().to_string()];
        }
        return vec!["/".to_string()];
    }

    // Expand ~ to home directory
    let expanded = expand_tilde(input);

    // Determine directory to list and prefix to filter
    // Handle special case: if input ends with /. (wanting hidden files)
    let (dir_to_list, prefix) = if expanded.ends_with("/.") {
        // User typed "/path/." - they want hidden files in /path/
        let dir = &expanded[..expanded.len() - 2];
        (PathBuf::from(if dir.is_empty() { "/" } else { dir }), ".".to_string())
    } else if expanded.ends_with('/') {
        // Input ends with /, list directory contents
        (PathBuf::from(&expanded), String::new())
    } else {
        // Get parent directory and filename prefix
        // Find the last slash to split directory from prefix
        if let Some(last_slash) = expanded.rfind('/') {
            let dir = &expanded[..=last_slash];
            let prefix = &expanded[last_slash + 1..];
            (PathBuf::from(dir), prefix.to_string())
        } else {
            // No slash - treat as prefix in current directory
            (PathBuf::from("."), expanded.clone())
        }
    };

    // List directory entries
    let entries = match std::fs::read_dir(&dir_to_list) {
        Ok(entries) => entries,
        Err(_) => return vec![],
    };

    let mut completions: Vec<String> = entries
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            // Check if it's a directory or a symlink pointing to a directory
            // Use std::fs::metadata on the full path - this follows symlinks
            let full_path = entry.path();
            std::fs::metadata(&full_path)
                .map(|m| m.is_dir())
                .unwrap_or(false)
        })
        .filter(|entry| {
            // Filter by prefix
            let name = entry.file_name().to_string_lossy().to_string();
            // Skip hidden directories unless explicitly typed
            if name.starts_with('.') && !prefix.starts_with('.') {
                return false;
            }
            prefix.is_empty() || name.to_lowercase().starts_with(&prefix.to_lowercase())
        })
        .map(|entry| entry.path().display().to_string())
        .collect();

    completions.sort_by_key(|a| a.to_lowercase());

    // Limit to reasonable number
    completions.truncate(20);

    completions
}

/// Expand ~ to home directory.
pub fn expand_tilde(input: &str) -> String {
    if input.starts_with('~') {
        if let Some(home) = dirs::home_dir() {
            return input.replacen('~', &home.display().to_string(), 1);
        }
    }
    input.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expand_tilde() {
        let home = dirs::home_dir().unwrap();
        let home_str = home.display().to_string();

        assert_eq!(expand_tilde("~"), home_str);
        assert_eq!(expand_tilde("~/Documents"), format!("{}/Documents", home_str));
        assert_eq!(expand_tilde("/absolute/path"), "/absolute/path");
        assert_eq!(expand_tilde("relative/path"), "relative/path");
    }

    #[test]
    fn test_empty_input_returns_home() {
        let completions = get_path_completions("");
        assert_eq!(completions.len(), 1);
        assert_eq!(completions[0], dirs::home_dir().unwrap().display().to_string());
    }

    #[test]
    fn test_tilde_expansion_in_completions() {
        let completions = get_path_completions("~/");
        // Should return directories in home
        assert!(!completions.is_empty());
        // All should start with home directory path
        let home = dirs::home_dir().unwrap().display().to_string();
        for c in &completions {
            assert!(c.starts_with(&home), "Expected {} to start with {}", c, home);
        }
    }

    #[test]
    fn test_hidden_dirs_excluded_by_default() {
        let completions = get_path_completions("~/");
        // None should start with a dot (hidden)
        for c in &completions {
            let name = Path::new(c).file_name().unwrap().to_string_lossy();
            assert!(!name.starts_with('.'), "Hidden dir {} should be excluded", c);
        }
    }

    #[test]
    fn test_hidden_dirs_included_when_typed() {
        let completions = get_path_completions("~/.");
        // All should be hidden directories
        assert!(!completions.is_empty(), "Should find hidden directories");
        for c in &completions {
            let name = Path::new(c).file_name().unwrap().to_string_lossy();
            assert!(name.starts_with('.'), "Expected hidden dir, got {}", c);
        }
    }

    #[test]
    fn test_case_insensitive_filtering() {
        // Test with a common directory that exists (Documents or downloads)
        let completions_lower = get_path_completions("~/doc");
        let completions_upper = get_path_completions("~/Doc");

        // Both should return the same results
        assert_eq!(completions_lower, completions_upper);
    }

    #[test]
    fn test_symlinks_are_included() {
        // Check if OneDrive symlink is found
        let home = dirs::home_dir().unwrap();
        let onedrive_path = home.join("OneDrive - Hilti");

        if onedrive_path.exists() {
            // Test with partial match
            let completions = get_path_completions("~/One");
            assert!(
                completions.iter().any(|c| c.contains("OneDrive")),
                "OneDrive symlink should be in completions with ~/One: {:?}",
                completions
            );

            // Test listing home directory
            let completions = get_path_completions("~/");
            assert!(
                completions.iter().any(|c| c.contains("OneDrive")),
                "OneDrive symlink should be in completions with ~/: {:?}",
                completions
            );
        }
    }

    #[test]
    fn test_spaces_in_path_names() {
        // Check if directories with spaces are handled correctly
        let home = dirs::home_dir().unwrap();
        let onedrive_path = home.join("OneDrive - Hilti");

        if onedrive_path.exists() {
            // The path contains spaces - make sure it's returned correctly
            let completions = get_path_completions("~/OneDrive");
            assert!(
                completions.iter().any(|c| c.contains("OneDrive - Hilti")),
                "Path with spaces should be in completions: {:?}",
                completions
            );
        }
    }

    #[test]
    fn test_partial_path_completion() {
        let home = dirs::home_dir().unwrap().display().to_string();

        // Test with partial path
        let completions = get_path_completions(&format!("{}/Doc", home));

        // Should find Documents if it exists
        let docs_path = format!("{}/Documents", home);
        if Path::new(&docs_path).exists() {
            assert!(
                completions.iter().any(|c| c.contains("Documents")),
                "Documents should be in completions: {:?}",
                completions
            );
        }
    }

    #[test]
    fn test_trailing_slash_lists_contents() {
        let home = dirs::home_dir().unwrap().display().to_string();

        // Without trailing slash - might complete the directory name itself
        let completions_no_slash = get_path_completions(&home);

        // With trailing slash - should list contents
        let completions_with_slash = get_path_completions(&format!("{}/", home));

        // With slash should have more results (contents of home)
        assert!(
            completions_with_slash.len() >= completions_no_slash.len(),
            "Trailing slash should list contents"
        );
    }

    #[test]
    fn test_nonexistent_path_returns_empty() {
        let completions = get_path_completions("/nonexistent/path/that/doesnt/exist/");
        assert!(completions.is_empty());
    }

    #[test]
    fn test_completions_are_sorted() {
        let completions = get_path_completions("~/");

        // Check that completions are sorted case-insensitively
        let mut sorted = completions.clone();
        sorted.sort_by(|a, b| a.to_lowercase().cmp(&b.to_lowercase()));

        assert_eq!(completions, sorted, "Completions should be sorted");
    }

    #[test]
    fn test_max_completions_limit() {
        // Root directory typically has many entries
        let completions = get_path_completions("/");
        assert!(completions.len() <= 20, "Should limit to 20 completions");
    }
}
