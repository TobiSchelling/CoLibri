//! TUI configuration editor for CoLibri.
//!
//! Provides an interactive terminal interface for managing folder sources,
//! viewing index status, and triggering re-indexing operations.

mod app;
mod event;
mod ui;

use std::io::stdout;

use crossterm::{
    event::{DisableMouseCapture, EnableMouseCapture},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::prelude::*;

use crate::config::load_config;

pub use app::App;

/// Run the TUI configuration editor.
pub async fn run() -> anyhow::Result<()> {
    // Load configuration
    let config = load_config()?;

    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Create app and run
    let mut app = App::new(config);
    let result = run_app(&mut terminal, &mut app).await;

    // Restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    result
}

/// Main application loop.
async fn run_app<B: Backend>(terminal: &mut Terminal<B>, app: &mut App) -> anyhow::Result<()> {
    loop {
        // Draw UI
        terminal.draw(|frame| ui::draw(frame, app))?;

        // Handle events
        if let Some(action) = event::handle_events(app)? {
            match action {
                event::Action::Quit => break,
                event::Action::Index { folder, force } => {
                    // Trigger indexing in a way that shows progress
                    app.start_indexing(folder, force);
                }
                event::Action::Save => {
                    if let Err(e) = app.save_config() {
                        app.set_error(format!("Failed to save: {e}"));
                    }
                }
                event::Action::Tick => {
                    // Update any async operations
                    app.tick().await;
                }
            }
        }

        // Check if we should quit
        if app.should_quit() {
            break;
        }
    }

    Ok(())
}
