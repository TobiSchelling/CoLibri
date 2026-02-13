//! UI rendering for the TUI configuration editor.

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Modifier, Style, Stylize},
    text::{Line, Span},
    widgets::{Block, Borders, Cell, Clear, Gauge, List, ListItem, Paragraph, Row, Table, Wrap},
    Frame,
};

use super::app::{App, EditField, OllamaStatus, Screen};
use crate::config::IndexMode;

// Catppuccin Mocha color palette
// Full palette defined for consistency; some colors reserved for future use
#[allow(dead_code)]
mod catppuccin {
    use ratatui::style::Color;

    pub const ROSEWATER: Color = Color::Rgb(245, 224, 220);
    pub const FLAMINGO: Color = Color::Rgb(242, 205, 205);
    pub const PINK: Color = Color::Rgb(245, 194, 231);
    pub const MAUVE: Color = Color::Rgb(203, 166, 247);
    pub const RED: Color = Color::Rgb(243, 139, 168);
    pub const MAROON: Color = Color::Rgb(235, 160, 172);
    pub const PEACH: Color = Color::Rgb(250, 179, 135);
    pub const YELLOW: Color = Color::Rgb(249, 226, 175);
    pub const GREEN: Color = Color::Rgb(166, 227, 161);
    pub const TEAL: Color = Color::Rgb(148, 226, 213);
    pub const SKY: Color = Color::Rgb(137, 220, 235);
    pub const SAPPHIRE: Color = Color::Rgb(116, 199, 236);
    pub const BLUE: Color = Color::Rgb(137, 180, 250);
    pub const LAVENDER: Color = Color::Rgb(180, 190, 254);
    pub const TEXT: Color = Color::Rgb(205, 214, 244);
    pub const SUBTEXT1: Color = Color::Rgb(186, 194, 222);
    pub const SUBTEXT0: Color = Color::Rgb(166, 173, 200);
    pub const OVERLAY2: Color = Color::Rgb(147, 153, 178);
    pub const OVERLAY1: Color = Color::Rgb(127, 132, 156);
    pub const OVERLAY0: Color = Color::Rgb(108, 112, 134);
    pub const SURFACE2: Color = Color::Rgb(88, 91, 112);
    pub const SURFACE1: Color = Color::Rgb(69, 71, 90);
    pub const SURFACE0: Color = Color::Rgb(49, 50, 68);
    pub const BASE: Color = Color::Rgb(30, 30, 46);
    pub const MANTLE: Color = Color::Rgb(24, 24, 37);
    pub const CRUST: Color = Color::Rgb(17, 17, 27);
}

use catppuccin::*;

/// Main draw function dispatches to screen-specific renderers.
pub fn draw(frame: &mut Frame, app: &App) {
    let area = frame.area();

    // Main layout: header, content, footer
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Header
            Constraint::Min(0),    // Content
            Constraint::Length(3), // Footer/help
        ])
        .split(area);

    draw_header(frame, app, chunks[0]);

    match app.screen {
        Screen::FolderList => draw_folder_list(frame, app, chunks[1]),
        Screen::EditFolder => draw_edit_folder(frame, app, chunks[1], "Edit Folder"),
        Screen::AddFolder => draw_edit_folder(frame, app, chunks[1], "Add Folder"),
        Screen::Indexing => draw_indexing(frame, app, chunks[1]),
        Screen::Status => draw_status(frame, app, chunks[1]),
    }

    draw_footer(frame, app, chunks[2]);

    // Draw error popup if present
    if let Some(ref error) = app.error_message {
        draw_error_popup(frame, error, area);
    }
}

fn draw_header(frame: &mut Frame, app: &App, area: Rect) {
    let title = match app.screen {
        Screen::FolderList => "CoLibri Configuration",
        Screen::EditFolder => "Edit Folder",
        Screen::AddFolder => "Add Folder",
        Screen::Indexing => "Indexing",
        Screen::Status => "System Status",
    };

    let modified = if app.has_changes { " [modified]" } else { "" };

    let header = Paragraph::new(format!("{title}{modified}"))
        .style(Style::default().fg(MAUVE).bold())
        .block(
            Block::default()
                .borders(Borders::BOTTOM)
                .border_style(Style::default().fg(SURFACE1)),
        );

    frame.render_widget(header, area);
}

fn draw_footer(frame: &mut Frame, app: &App, area: Rect) {
    let help_text = match app.screen {
        Screen::FolderList => {
            "j/k: navigate | Enter/e: edit | a: add | d: delete | i: index | S: save | ?: status | q: quit"
        }
        Screen::EditFolder | Screen::AddFolder => {
            if app.is_editing_field {
                if app.path_completion.is_active {
                    "↑/↓: select | Tab: enter folder | Enter: confirm path | Esc: cancel"
                } else if app.is_editing_path() {
                    "Tab: browse folders | Enter: confirm | Esc: cancel"
                } else {
                    "Enter: confirm | Esc: cancel"
                }
            } else {
                "j/k: navigate | Enter: edit | Tab: cycle mode | S: save folder | Esc: back"
            }
        }
        Screen::Indexing => {
            if app.indexing.is_running {
                "Indexing in progress..."
            } else {
                "Enter/Esc: back to folder list"
            }
        }
        Screen::Status => "Enter/Esc: back | q: quit",
    };

    let footer = Paragraph::new(help_text)
        .style(Style::default().fg(OVERLAY1))
        .block(
            Block::default()
                .borders(Borders::TOP)
                .border_style(Style::default().fg(SURFACE1)),
        );

    frame.render_widget(footer, area);
}

fn draw_folder_list(frame: &mut Frame, app: &App, area: Rect) {
    if app.config.sources.is_empty() {
        let text = Paragraph::new("No folders configured. Press 'a' to add one.")
            .style(Style::default().fg(YELLOW))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(SURFACE2))
                    .title("Folders"),
            );
        frame.render_widget(text, area);
        return;
    }

    let total = app.config.sources.len();
    let selected = app.selected_folder;

    // Calculate visible rows (account for borders and header)
    let max_visible = (area.height.saturating_sub(5)) as usize; // borders + header + margin

    // Calculate scroll offset to keep selection visible
    let scroll_offset = if selected < max_visible / 2 {
        0
    } else if selected >= total.saturating_sub(max_visible / 2) {
        total.saturating_sub(max_visible)
    } else {
        selected.saturating_sub(max_visible / 2)
    };

    let visible_end = (scroll_offset + max_visible).min(total);

    // Create table rows for visible range
    let rows: Vec<Row> = app.config.sources[scroll_offset..visible_end]
        .iter()
        .enumerate()
        .map(|(i, folder)| {
            let actual_index = scroll_offset + i;
            let (files, chunks) = app.get_folder_stats(folder);
            let path_exists = std::path::Path::new(&folder.path).exists();

            let status_style = if !path_exists {
                Style::default().fg(RED)
            } else {
                Style::default().fg(GREEN)
            };

            let status = if !path_exists { "MISSING" } else { "OK" };

            let style = if actual_index == selected {
                Style::default().bg(SURFACE1).fg(TEXT)
            } else {
                Style::default().fg(SUBTEXT1)
            };

            Row::new(vec![
                Cell::from(folder.display_name()),
                Cell::from(format_mode(folder.mode)),
                Cell::from(folder.doc_type.as_str()),
                Cell::from(format!("{files}")),
                Cell::from(format!("{chunks}")),
                Cell::from(status).style(status_style),
            ])
            .style(style)
        })
        .collect();

    // Build title with scroll indicator
    let title = if total > max_visible {
        let up = if scroll_offset > 0 { "↑ " } else { "" };
        let down = if visible_end < total { " ↓" } else { "" };
        format!("{}Folders ({}/{}){}", up, selected + 1, total, down)
    } else {
        "Folders".to_string()
    };

    let table = Table::new(
        rows,
        [
            Constraint::Percentage(25), // Name
            Constraint::Length(12),     // Mode
            Constraint::Length(10),     // Type
            Constraint::Length(8),      // Files
            Constraint::Length(10),     // Chunks
            Constraint::Length(10),     // Status
        ],
    )
    .header(
        Row::new(vec!["Name", "Mode", "Type", "Files", "Chunks", "Status"])
            .style(Style::default().fg(LAVENDER).bold())
            .bottom_margin(1),
    )
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(SURFACE2))
            .title(title),
    )
    .row_highlight_style(Style::default().add_modifier(Modifier::BOLD));

    frame.render_widget(table, area);
}

fn draw_edit_folder(frame: &mut Frame, app: &App, area: Rect, title: &str) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(SURFACE2))
        .title(title);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let Some(ref folder) = app.editing_folder else {
        return;
    };

    // Split inner area for fields and hints
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(10),   // Fields table
            Constraint::Length(2), // Field hint
        ])
        .split(inner);

    let fields = EditField::all();
    let rows: Vec<Row> = fields
        .iter()
        .enumerate()
        .map(|(i, field)| {
            let is_selected = i == app.edit_field;
            let is_editing = is_selected && app.is_editing_field;

            let value = if is_editing {
                format!("{}|", app.edit_input)
            } else {
                match field {
                    EditField::Path => folder.path.clone(),
                    EditField::Name => folder.name.clone().unwrap_or_else(|| "(auto)".into()),
                    EditField::DocType => folder.doc_type.clone(),
                    EditField::Mode => format_mode(folder.mode).to_string(),
                    EditField::Extensions => folder.extensions.join(", "),
                    EditField::ChunkSize => folder
                        .chunk_size
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| "(default)".into()),
                    EditField::ChunkOverlap => folder
                        .chunk_overlap
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| "(default)".into()),
                }
            };

            let style = if is_selected {
                Style::default().bg(SURFACE1).fg(TEXT)
            } else {
                Style::default().fg(SUBTEXT1)
            };

            let value_style = if is_editing {
                style.add_modifier(Modifier::UNDERLINED)
            } else {
                style
            };

            Row::new(vec![
                Cell::from(field.label()).style(Style::default().fg(LAVENDER)),
                Cell::from(value).style(value_style),
            ])
            .style(style)
        })
        .collect();

    let table = Table::new(
        rows,
        [
            Constraint::Length(15), // Label
            Constraint::Min(20),    // Value
        ],
    );

    frame.render_widget(table, chunks[0]);

    // Show field-specific hint
    let hint = get_field_hint(app.current_field());
    let hint_para = Paragraph::new(hint).style(Style::default().fg(OVERLAY1).italic());
    frame.render_widget(hint_para, chunks[1]);

    // Draw completion popup if active
    if app.path_completion.is_active && !app.path_completion.completions.is_empty() {
        draw_completion_popup(frame, app, area);
    }
}

/// Get hint text for a field.
fn get_field_hint(field: EditField) -> &'static str {
    match field {
        EditField::Path => "Absolute path to folder. Use Tab for autocomplete, ~ expands to home.",
        EditField::Name => "Display name (optional). Defaults to folder name if empty.",
        EditField::DocType => "Document type label (e.g., 'book', 'note', 'article').",
        EditField::Mode => "static: index once | incremental: track changes",
        EditField::Extensions => "File extensions to index, comma-separated (e.g., .md, .txt).",
        EditField::ChunkSize => "Characters per chunk. Leave empty for default (3000).",
        EditField::ChunkOverlap => "Overlap between chunks. Leave empty for default (200).",
    }
}

/// Draw the path completion popup.
fn draw_completion_popup(frame: &mut Frame, app: &App, area: Rect) {
    let completions = &app.path_completion.completions;
    let selected = app.path_completion.selected;
    let total = completions.len();

    // Extract just folder names for display
    let folder_names: Vec<String> = completions
        .iter()
        .map(|path| {
            std::path::Path::new(path)
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| path.clone())
        })
        .collect();

    // Calculate popup size based on folder names (not full paths)
    let max_width = folder_names
        .iter()
        .map(|n| n.len())
        .max()
        .unwrap_or(20)
        .clamp(20, 50) as u16
        + 6; // Extra space for scroll indicator

    let max_visible = 10;
    let height = (total.min(max_visible) + 2) as u16;

    // Position popup below the path field (roughly row 1-2 of the form)
    let popup_x = area.x + 17; // After label column
    let popup_y = area.y + 4; // Below header + first field

    let popup_area = Rect::new(
        popup_x.min(area.width.saturating_sub(max_width)),
        popup_y.min(area.height.saturating_sub(height)),
        max_width.min(area.width.saturating_sub(popup_x)),
        height.min(area.height.saturating_sub(popup_y)),
    );

    // Clear background
    frame.render_widget(Clear, popup_area);

    // Calculate visible window (scroll to keep selection visible)
    let visible_start = if selected < max_visible / 2 {
        0
    } else if selected >= total.saturating_sub(max_visible / 2) {
        total.saturating_sub(max_visible)
    } else {
        selected.saturating_sub(max_visible / 2)
    };
    let visible_end = (visible_start + max_visible).min(total);

    // Create list items for visible window only
    let items: Vec<ListItem> = folder_names[visible_start..visible_end]
        .iter()
        .enumerate()
        .map(|(i, name)| {
            let actual_index = visible_start + i;
            let style = if actual_index == selected {
                Style::default().bg(BLUE).fg(CRUST)
            } else {
                Style::default().fg(TEXT)
            };
            ListItem::new(name.as_str()).style(style)
        })
        .collect();

    // Build title with scroll indicators
    let title = if total > max_visible {
        let up_arrow = if visible_start > 0 { "↑ " } else { "" };
        let down_arrow = if visible_end < total { " ↓" } else { "" };
        format!("{}Folders ({}/{}){}", up_arrow, selected + 1, total, down_arrow)
    } else {
        "Folders".to_string()
    };

    let list = List::new(items).block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(SAPPHIRE))
            .style(Style::default().bg(SURFACE0))
            .title(Span::styled(title, Style::default().fg(LAVENDER))),
    );

    frame.render_widget(list, popup_area);
}

fn draw_indexing(frame: &mut Frame, app: &App, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(SURFACE2))
        .title("Indexing");
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2), // Target
            Constraint::Length(3), // Progress bar
            Constraint::Min(0),    // Results/messages
        ])
        .margin(1)
        .split(inner);

    // Target folder
    let target = if let Some(ref folder) = app.indexing.folder {
        format!("Target: {folder}")
    } else {
        "Target: All folders".into()
    };
    let force_str = if app.indexing.force { " (force)" } else { "" };
    let target_para =
        Paragraph::new(format!("{target}{force_str}")).style(Style::default().fg(SAPPHIRE));
    frame.render_widget(target_para, chunks[0]);

    // Progress indicator
    if app.indexing.is_running {
        let gauge = Gauge::default()
            .block(Block::default())
            .gauge_style(Style::default().fg(GREEN).bg(SURFACE0))
            .percent(50) // Indeterminate - could enhance with actual progress
            .label("Processing...");
        frame.render_widget(gauge, chunks[1]);
    } else {
        let status_style = if app.indexing.error.is_some() {
            Style::default().fg(RED)
        } else {
            Style::default().fg(GREEN)
        };
        let status = Paragraph::new(&*app.indexing.message).style(status_style);
        frame.render_widget(status, chunks[1]);
    }

    // Results
    if let Some(ref result) = app.indexing.result {
        let results_text = vec![
            Line::from(vec![
                Span::styled("Files indexed: ", Style::default().fg(LAVENDER)),
                Span::styled(result.files_indexed.to_string(), Style::default().fg(TEXT)),
            ]),
            Line::from(vec![
                Span::styled("Chunks created: ", Style::default().fg(LAVENDER)),
                Span::styled(result.total_chunks.to_string(), Style::default().fg(TEXT)),
            ]),
            Line::from(vec![
                Span::styled("Files skipped: ", Style::default().fg(LAVENDER)),
                Span::styled(result.files_skipped.to_string(), Style::default().fg(TEXT)),
            ]),
            Line::from(vec![
                Span::styled("Files deleted: ", Style::default().fg(LAVENDER)),
                Span::styled(result.files_deleted.to_string(), Style::default().fg(TEXT)),
            ]),
            Line::from(vec![
                Span::styled("Errors: ", Style::default().fg(LAVENDER)),
                Span::styled(
                    result.errors.to_string(),
                    if result.errors > 0 {
                        Style::default().fg(RED)
                    } else {
                        Style::default().fg(TEXT)
                    },
                ),
            ]),
        ];
        let results = Paragraph::new(results_text);
        frame.render_widget(results, chunks[2]);
    } else if let Some(ref error) = app.indexing.error {
        let error_text = Paragraph::new(error.as_str())
            .style(Style::default().fg(RED))
            .wrap(Wrap { trim: true });
        frame.render_widget(error_text, chunks[2]);
    }
}

fn draw_status(frame: &mut Frame, app: &App, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(SURFACE2))
        .title(Span::styled("System Status", Style::default().fg(MAUVE)));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(6), // Config info
            Constraint::Length(6), // Index info
            Constraint::Length(4), // Ollama info
            Constraint::Min(0),    // Sources
        ])
        .margin(1)
        .split(inner);

    // Config info
    let config_text = vec![
        Line::from(vec![
            Span::styled("Config: ", Style::default().fg(PEACH)),
            Span::styled(&app.status.config_path, Style::default().fg(TEXT)),
        ]),
        Line::from(vec![
            Span::styled("Data dir: ", Style::default().fg(PEACH)),
            Span::styled(&app.status.data_dir, Style::default().fg(TEXT)),
        ]),
        Line::from(vec![
            Span::styled("LanceDB dir: ", Style::default().fg(PEACH)),
            Span::styled(&app.status.lancedb_dir, Style::default().fg(TEXT)),
        ]),
    ];
    let config_para = Paragraph::new(config_text);
    frame.render_widget(config_para, chunks[0]);

    // Index info
    let schema_status = if app.status.stored_version == app.status.schema_version {
        Span::styled("OK", Style::default().fg(GREEN))
    } else if app.status.stored_version == 0 {
        Span::styled("NOT FOUND", Style::default().fg(YELLOW))
    } else {
        Span::styled(
            format!(
                "OUTDATED (v{} -> v{})",
                app.status.stored_version, app.status.schema_version
            ),
            Style::default().fg(RED),
        )
    };

    let index_text = vec![
        Line::from(vec![
            Span::styled("Schema version: ", Style::default().fg(PEACH)),
            schema_status,
        ]),
        Line::from(vec![
            Span::styled("Files: ", Style::default().fg(PEACH)),
            Span::styled(app.status.file_count.to_string(), Style::default().fg(TEXT)),
        ]),
        Line::from(vec![
            Span::styled("Chunks: ", Style::default().fg(PEACH)),
            Span::styled(app.status.chunk_count.to_string(), Style::default().fg(TEXT)),
        ]),
        Line::from(vec![
            Span::styled("Model: ", Style::default().fg(PEACH)),
            Span::styled(&app.status.embedding_model, Style::default().fg(TEXT)),
        ]),
        Line::from(vec![
            Span::styled("Last indexed: ", Style::default().fg(PEACH)),
            Span::styled(
                app.status.last_indexed.as_deref().unwrap_or("never"),
                Style::default().fg(TEXT),
            ),
        ]),
    ];
    let index_para = Paragraph::new(index_text);
    frame.render_widget(index_para, chunks[1]);

    // Ollama info
    let ollama_status = match &app.status.ollama_status {
        OllamaStatus::Unknown => Span::styled("Unknown", Style::default().fg(OVERLAY0)),
        OllamaStatus::Checking => Span::styled("Checking...", Style::default().fg(YELLOW)),
        OllamaStatus::Connected(url) => Span::styled(
            format!("Connected ({url})"),
            Style::default().fg(GREEN),
        ),
        OllamaStatus::Disconnected(url) => Span::styled(
            format!("Disconnected ({url})"),
            Style::default().fg(RED),
        ),
        OllamaStatus::Error(e) => {
            Span::styled(format!("Error: {e}"), Style::default().fg(RED))
        }
    };

    let ollama_text = vec![Line::from(vec![
        Span::styled("Ollama: ", Style::default().fg(PEACH)),
        ollama_status,
    ])];
    let ollama_para = Paragraph::new(ollama_text);
    frame.render_widget(ollama_para, chunks[2]);

    // Sources list
    if !app.config.sources.is_empty() {
        let sources_text: Vec<Line> = app
            .config
            .sources
            .iter()
            .map(|s| {
                let path_exists = std::path::Path::new(&s.path).exists();
                let status = if path_exists {
                    Span::styled("OK", Style::default().fg(GREEN))
                } else {
                    Span::styled("MISSING", Style::default().fg(RED))
                };
                Line::from(vec![
                    Span::styled(
                        format!("  {} ", s.display_name()),
                        Style::default().fg(SKY),
                    ),
                    Span::styled("... ", Style::default().fg(OVERLAY0)),
                    status,
                ])
            })
            .collect();

        let mut all_lines = vec![Line::from(Span::styled(
            "Sources:",
            Style::default().fg(PEACH),
        ))];
        all_lines.extend(sources_text);

        let sources_para = Paragraph::new(all_lines);
        frame.render_widget(sources_para, chunks[3]);
    }
}

fn draw_error_popup(frame: &mut Frame, error: &str, area: Rect) {
    // Center the popup
    let popup_width = 60.min(area.width.saturating_sub(4));
    let popup_height = 5;
    let popup_area = Rect::new(
        (area.width.saturating_sub(popup_width)) / 2,
        (area.height.saturating_sub(popup_height)) / 2,
        popup_width,
        popup_height,
    );

    // Clear the background
    frame.render_widget(Clear, popup_area);

    let popup = Paragraph::new(error)
        .style(Style::default().fg(TEXT))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(RED))
                .style(Style::default().bg(SURFACE0))
                .title(Span::styled("Error", Style::default().fg(RED).bold())),
        )
        .wrap(Wrap { trim: true });

    frame.render_widget(popup, popup_area);
}

fn format_mode(mode: IndexMode) -> &'static str {
    match mode {
        IndexMode::Static => "static",
        IndexMode::Incremental => "incremental",
    }
}
