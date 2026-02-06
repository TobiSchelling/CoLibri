//! Keyboard event handling for the TUI.

use std::time::Duration;

use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};

use super::app::{App, Screen};

/// Actions that can be triggered by user input.
#[derive(Debug, Clone)]
pub enum Action {
    Quit,
    Index { folder: Option<String>, force: bool },
    Save,
    Tick,
}

/// Handle keyboard events and return an action if applicable.
pub fn handle_events(app: &mut App) -> std::io::Result<Option<Action>> {
    // Poll with a timeout for async operations
    if event::poll(Duration::from_millis(100))? {
        if let Event::Key(key) = event::read()? {
            // Only handle key press events
            if key.kind != KeyEventKind::Press {
                return Ok(None);
            }

            // Clear error on any key press
            app.clear_error();

            // Handle based on current screen
            return Ok(match app.screen {
                Screen::FolderList => handle_folder_list(app, key.code, key.modifiers),
                Screen::EditFolder | Screen::AddFolder => {
                    handle_edit_folder(app, key.code, key.modifiers)
                }
                Screen::Indexing => handle_indexing(app, key.code),
                Screen::Status => handle_status(app, key.code),
            });
        }
    }

    // Return tick for async operations
    Ok(Some(Action::Tick))
}

fn handle_folder_list(app: &mut App, code: KeyCode, modifiers: KeyModifiers) -> Option<Action> {
    match code {
        KeyCode::Char('q') => {
            app.quit = true;
            Some(Action::Quit)
        }
        KeyCode::Char('c') if modifiers.contains(KeyModifiers::CONTROL) => {
            app.quit = true;
            Some(Action::Quit)
        }
        KeyCode::Up | KeyCode::Char('k') => {
            app.previous_folder();
            None
        }
        KeyCode::Down | KeyCode::Char('j') => {
            app.next_folder();
            None
        }
        KeyCode::Enter | KeyCode::Char('e') => {
            app.go_to_edit_folder();
            None
        }
        KeyCode::Char('a') => {
            app.go_to_add_folder();
            None
        }
        KeyCode::Char('d') | KeyCode::Delete => {
            app.delete_folder();
            None
        }
        KeyCode::Char('i') => {
            // Index selected folder
            let folder = app
                .config
                .sources
                .get(app.selected_folder)
                .map(|f| f.display_name().to_string());
            Some(Action::Index {
                folder,
                force: false,
            })
        }
        KeyCode::Char('I') => {
            // Force index selected folder
            let folder = app
                .config
                .sources
                .get(app.selected_folder)
                .map(|f| f.display_name().to_string());
            Some(Action::Index {
                folder,
                force: true,
            })
        }
        KeyCode::Char('r') => {
            // Index all
            Some(Action::Index {
                folder: None,
                force: false,
            })
        }
        KeyCode::Char('R') => {
            // Force rebuild all
            Some(Action::Index {
                folder: None,
                force: true,
            })
        }
        KeyCode::Char('s') if modifiers.contains(KeyModifiers::CONTROL) => {
            // Save config
            Some(Action::Save)
        }
        KeyCode::Char('S') => {
            // Save config
            Some(Action::Save)
        }
        KeyCode::Char('?') | KeyCode::F(1) => {
            app.go_to_status();
            None
        }
        _ => None,
    }
}

fn handle_edit_folder(app: &mut App, code: KeyCode, modifiers: KeyModifiers) -> Option<Action> {
    if app.is_editing_field {
        // Check if path completion popup is active
        if app.path_completion.is_active {
            match code {
                KeyCode::Down => {
                    app.next_completion();
                    None
                }
                KeyCode::Up | KeyCode::BackTab => {
                    app.previous_completion();
                    None
                }
                KeyCode::Tab => {
                    // Tab applies selection and drills into the folder
                    app.apply_completion();
                    None
                }
                KeyCode::Enter => {
                    // Enter confirms path and finishes field editing
                    if app.path_completion.selected < app.path_completion.completions.len() {
                        app.apply_completion_final();
                    }
                    app.finish_editing_field();
                    None
                }
                KeyCode::Esc => {
                    app.cancel_completion();
                    None
                }
                KeyCode::Char(c) => {
                    // Add character and re-trigger completion to filter
                    app.edit_input.push(c);
                    app.trigger_path_completion();
                    None
                }
                KeyCode::Backspace => {
                    app.edit_input.pop();
                    if app.edit_input.is_empty() {
                        app.cancel_completion();
                    } else {
                        app.trigger_path_completion();
                    }
                    None
                }
                _ => None,
            }
        } else {
            // Text input mode (no completion active)
            match code {
                KeyCode::Enter => {
                    app.finish_editing_field();
                    None
                }
                KeyCode::Esc => {
                    app.cancel_editing_field();
                    None
                }
                KeyCode::Backspace => {
                    app.edit_input.pop();
                    // Auto-trigger completion for path field
                    if app.is_editing_path() && !app.edit_input.is_empty() {
                        app.trigger_path_completion();
                    }
                    None
                }
                KeyCode::Tab => {
                    // Trigger path completion for Path field
                    if app.is_editing_path() {
                        app.trigger_path_completion();
                    }
                    None
                }
                KeyCode::Char(c) => {
                    app.edit_input.push(c);
                    // Auto-trigger completion for path field
                    if app.is_editing_path() {
                        app.trigger_path_completion();
                    }
                    None
                }
                _ => None,
            }
        }
    } else {
        // Navigation mode
        match code {
            KeyCode::Esc | KeyCode::Char('q') => {
                app.go_to_folder_list();
                None
            }
            KeyCode::Char('c') if modifiers.contains(KeyModifiers::CONTROL) => {
                app.go_to_folder_list();
                None
            }
            KeyCode::Up | KeyCode::Char('k') => {
                app.previous_field();
                None
            }
            KeyCode::Down | KeyCode::Char('j') => {
                app.next_field();
                None
            }
            KeyCode::Enter | KeyCode::Char('e') => {
                // For Mode field, cycle through options instead of text edit
                if app.current_field() == super::app::EditField::Mode {
                    app.cycle_mode();
                } else {
                    app.start_editing_field();
                }
                None
            }
            KeyCode::Tab => {
                // Cycle mode for Mode field
                if app.current_field() == super::app::EditField::Mode {
                    app.cycle_mode();
                }
                None
            }
            KeyCode::Char('s') if modifiers.contains(KeyModifiers::CONTROL) => {
                app.save_folder();
                Some(Action::Save)
            }
            KeyCode::Char('S') => {
                app.save_folder();
                Some(Action::Save)
            }
            _ => None,
        }
    }
}

fn handle_indexing(app: &mut App, code: KeyCode) -> Option<Action> {
    match code {
        KeyCode::Esc | KeyCode::Char('q') => {
            if !app.indexing.is_running {
                app.go_to_folder_list();
            }
            None
        }
        KeyCode::Enter => {
            if !app.indexing.is_running {
                app.go_to_folder_list();
            }
            None
        }
        _ => None,
    }
}

fn handle_status(app: &mut App, code: KeyCode) -> Option<Action> {
    match code {
        KeyCode::Esc | KeyCode::Char('q') | KeyCode::Enter => {
            app.go_to_folder_list();
            None
        }
        _ => None,
    }
}
