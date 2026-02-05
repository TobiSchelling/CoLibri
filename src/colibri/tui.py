"""Textual TUI for managing CoLibri source profiles.

Launched via ``colibri config --tui``.  Provides a visual interface for
adding, editing, and removing per-folder source profiles.
"""

from __future__ import annotations

from pathlib import Path

from textual import on
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    Select,
    Static,
)

from colibri.config import get_sources_raw, set_sources_raw

_MODE_CHOICES = [
    ("static", "static"),
    ("incremental", "incremental"),
    ("append_only", "append_only"),
    ("disabled", "disabled"),
]


# ---------------------------------------------------------------------------
# Edit / Add modal
# ---------------------------------------------------------------------------


class ProfileForm(ModalScreen[dict | None]):
    """Modal form for creating or editing a source profile."""

    CSS = """
    ProfileForm {
        align: center middle;
    }
    #form-container {
        width: 70;
        height: auto;
        max-height: 80%;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    #form-container Label {
        margin-top: 1;
    }
    #form-buttons {
        margin-top: 1;
        height: 3;
        align: right middle;
    }
    #form-buttons Button {
        margin-left: 1;
    }
    """

    def __init__(
        self,
        profile: dict | None = None,
        existing_paths: list[str] | None = None,
    ) -> None:
        super().__init__()
        self._profile = profile  # None → add mode
        self._existing_paths = existing_paths or []

    def compose(self) -> ComposeResult:
        p = self._profile or {}
        title = "Edit Source Profile" if self._profile else "Add Source Profile"
        with Vertical(id="form-container"):
            yield Label(f"[bold]{title}[/bold]")

            yield Label("Path (absolute directory path)")
            yield Input(
                value=p.get("path", ""),
                placeholder="e.g. /Users/you/Documents/Notes",
                id="path",
                disabled=self._profile is not None,
            )

            yield Label("Mode")
            yield Select(
                _MODE_CHOICES,
                value=p.get("mode", "incremental"),
                id="mode",
            )

            yield Label("Doc Type")
            yield Input(
                value=p.get("doc_type", "note"),
                placeholder="e.g. note, book, journal",
                id="doc_type",
            )

            yield Label("Display Name (blank = directory name)")
            yield Input(
                value=p.get("name", ""),
                placeholder="e.g. My Books",
                id="name",
            )

            yield Label("Extensions (comma-separated, blank = .md only)")
            yield Input(
                value=", ".join(p.get("extensions", [])) if p.get("extensions") else "",
                placeholder="e.g. .md, .yaml, .yml",
                id="extensions",
            )

            yield Label("Chunk Size (blank = global default)")
            yield Input(
                value=str(p["chunk_size"]) if p.get("chunk_size") else "",
                placeholder="e.g. 3000",
                id="chunk_size",
            )

            yield Label("Chunk Overlap (blank = global default)")
            yield Input(
                value=str(p["chunk_overlap"]) if p.get("chunk_overlap") else "",
                placeholder="e.g. 200",
                id="chunk_overlap",
            )

            with Horizontal(id="form-buttons"):
                yield Button("Cancel", variant="default", id="cancel")
                yield Button("Save", variant="primary", id="save")

    @on(Button.Pressed, "#cancel")
    def _cancel(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#save")
    def _save(self) -> None:
        path_val = self.query_one("#path", Input).value.strip()
        if not path_val:
            self.notify("Path is required", severity="error")
            return

        # Validate path exists
        if not Path(path_val).exists():
            self.notify(f"Path does not exist: {path_val}", severity="error")
            return

        # Prevent duplicate paths on add
        if self._profile is None and path_val in self._existing_paths:
            self.notify(f"Source for '{path_val}' already exists", severity="error")
            return

        mode = self.query_one("#mode", Select).value
        doc_type = self.query_one("#doc_type", Input).value.strip() or "note"
        name_val = self.query_one("#name", Input).value.strip()
        extensions_str = self.query_one("#extensions", Input).value.strip()
        chunk_size_str = self.query_one("#chunk_size", Input).value.strip()
        chunk_overlap_str = self.query_one("#chunk_overlap", Input).value.strip()

        result: dict = {
            "path": path_val,
            "mode": str(mode),
            "doc_type": doc_type,
        }

        if name_val:
            result["name"] = name_val

        if extensions_str:
            result["extensions"] = [e.strip() for e in extensions_str.split(",") if e.strip()]

        if chunk_size_str:
            try:
                result["chunk_size"] = int(chunk_size_str)
            except ValueError:
                self.notify("Chunk size must be a number", severity="error")
                return

        if chunk_overlap_str:
            try:
                result["chunk_overlap"] = int(chunk_overlap_str)
            except ValueError:
                self.notify("Chunk overlap must be a number", severity="error")
                return

        self.dismiss(result)


# ---------------------------------------------------------------------------
# Confirm delete modal
# ---------------------------------------------------------------------------


class ConfirmDelete(ModalScreen[bool]):
    """Confirmation dialog for deleting a source profile."""

    CSS = """
    ConfirmDelete {
        align: center middle;
    }
    #confirm-container {
        width: 50;
        height: auto;
        border: thick $error;
        background: $surface;
        padding: 1 2;
    }
    #confirm-buttons {
        margin-top: 1;
        height: 3;
        align: right middle;
    }
    #confirm-buttons Button {
        margin-left: 1;
    }
    """

    def __init__(self, display_name: str) -> None:
        super().__init__()
        self._display_name = display_name

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-container"):
            yield Static(f"Remove source profile for [bold]{self._display_name}[/bold]?")
            yield Static(
                "[dim]This only removes the profile from config. "
                "Existing index data is not deleted.[/dim]"
            )
            with Horizontal(id="confirm-buttons"):
                yield Button("Cancel", variant="default", id="cancel")
                yield Button("Remove", variant="error", id="confirm")

    @on(Button.Pressed, "#cancel")
    def _cancel(self) -> None:
        self.dismiss(False)

    @on(Button.Pressed, "#confirm")
    def _confirm(self) -> None:
        self.dismiss(True)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


class SourceProfileApp(App):
    """TUI for managing CoLibri source profiles."""

    TITLE = "CoLibri Source Profiles"
    CSS = """
    #toolbar {
        dock: bottom;
        height: 3;
        padding: 0 1;
    }
    #toolbar Button {
        margin-right: 1;
    }
    DataTable {
        height: 1fr;
    }
    """
    BINDINGS = [
        ("a", "add_profile", "Add"),
        ("e", "edit_profile", "Edit"),
        ("d", "delete_profile", "Delete"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._sources: list[dict] = []

    def compose(self) -> ComposeResult:
        yield Header()
        yield DataTable(id="profiles")
        with Horizontal(id="toolbar"):
            yield Button("Add (a)", variant="primary", id="btn-add")
            yield Button("Edit (e)", variant="default", id="btn-edit")
            yield Button("Delete (d)", variant="error", id="btn-delete")
            yield Button("Quit (q)", variant="default", id="btn-quit")
        yield Footer()

    def on_mount(self) -> None:
        self._load_profiles()

    def _display_name(self, src: dict) -> str:
        """Get display name for a source dict."""
        return src.get("name") or Path(src.get("path", "")).name

    def _load_profiles(self) -> None:
        """Load source profiles from config and populate the table."""
        self._sources = get_sources_raw()
        table = self.query_one("#profiles", DataTable)
        table.clear(columns=True)
        table.add_columns("Name", "Path", "Mode", "Doc Type", "Chunk Size")
        for src in self._sources:
            table.add_row(
                self._display_name(src),
                src.get("path", ""),
                src.get("mode", "incremental"),
                src.get("doc_type", "note"),
                str(src.get("chunk_size", "default")),
                key=src.get("path", ""),
            )

    def _save_and_reload(self) -> None:
        """Persist current sources to config and refresh the table."""
        set_sources_raw(self._sources)
        self._load_profiles()
        self.notify("Config saved")

    def _get_selected_path(self) -> str | None:
        """Return the path of the currently selected row."""
        table = self.query_one("#profiles", DataTable)
        if table.cursor_row is not None and table.row_count > 0:
            row_key = table.get_row_at(table.cursor_row)
            # Path is the second column (index 1)
            return str(row_key[1]) if row_key and len(row_key) > 1 else None
        return None

    def _get_selected_profile(self) -> dict | None:
        """Return the full profile dict for the selected row."""
        path = self._get_selected_path()
        if path is None:
            return None
        return next((s for s in self._sources if s.get("path") == path), None)

    # --- Actions ---

    def action_add_profile(self) -> None:
        existing = [s.get("path", "") for s in self._sources]
        self.push_screen(ProfileForm(existing_paths=existing), self._on_add_result)

    def _on_add_result(self, result: dict | None) -> None:
        if result is not None:
            self._sources.append(result)
            self._save_and_reload()

    def action_edit_profile(self) -> None:
        profile = self._get_selected_profile()
        if profile is None:
            self.notify("No profile selected", severity="warning")
            return
        existing = [s.get("path", "") for s in self._sources]
        form = ProfileForm(profile=profile, existing_paths=existing)
        self.push_screen(form, self._on_edit_result)

    def _on_edit_result(self, result: dict | None) -> None:
        if result is None:
            return
        path = result["path"]
        for i, src in enumerate(self._sources):
            if src.get("path") == path:
                self._sources[i] = result
                break
        self._save_and_reload()

    def action_delete_profile(self) -> None:
        profile = self._get_selected_profile()
        if profile is None:
            self.notify("No profile selected", severity="warning")
            return
        display = self._display_name(profile)
        self.push_screen(ConfirmDelete(display), self._on_delete_result)

    def _on_delete_result(self, confirmed: bool) -> None:
        if not confirmed:
            return
        path = self._get_selected_path()
        self._sources = [s for s in self._sources if s.get("path") != path]
        self._save_and_reload()

    # --- Button handlers ---

    @on(Button.Pressed, "#btn-add")
    def _btn_add(self) -> None:
        self.action_add_profile()

    @on(Button.Pressed, "#btn-edit")
    def _btn_edit(self) -> None:
        self.action_edit_profile()

    @on(Button.Pressed, "#btn-delete")
    def _btn_delete(self) -> None:
        self.action_delete_profile()

    @on(Button.Pressed, "#btn-quit")
    def _btn_quit(self) -> None:
        self.exit()


def run_tui() -> None:
    """Launch the Textual TUI for source profile management."""
    app = SourceProfileApp()
    app.run()
