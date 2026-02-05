"""Setup wizard for CoLibri installation and configuration.

This module provides an interactive setup experience that:
1. Checks prerequisites (Python, Ollama)
2. Installs missing dependencies
3. Creates configuration files
4. Sets up MCP integration for Claude
"""

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt

from colibri.config import CONFIG_DIR, CONFIG_FILE, EMBEDDING_MODEL, OLLAMA_BASE_URL

console = Console()


class SetupWizard:
    """Interactive setup wizard for CoLibri."""

    def __init__(self) -> None:
        self.system = platform.system().lower()
        self.issues: list[str] = []

    def run(self) -> bool:
        """Run the full setup wizard.

        Returns:
            True if setup completed successfully
        """
        console.print(
            Panel.fit(
                "[bold cyan]CoLibri Setup Wizard[/bold cyan]\n"
                "This will help you configure CoLibri for first use.",
                border_style="cyan",
            )
        )
        console.print()

        # Step 1: Check prerequisites
        if not self._check_prerequisites():
            return False

        # Step 2: Configure library path
        if not self._configure_library():
            return False

        # Step 3: Setup MCP integration (optional)
        self._setup_mcp_integration()

        # Step 4: Final verification
        self._show_summary()

        return True

    def _check_prerequisites(self) -> bool:
        """Check and install prerequisites."""
        console.print("[bold]Checking prerequisites...[/bold]\n")

        # Python version
        py_version = sys.version_info
        if py_version >= (3, 11):
            console.print(f"  [green]✓[/green] Python {py_version.major}.{py_version.minor}")
        else:
            console.print(
                f"  [red]✗[/red] Python {py_version.major}.{py_version.minor} (requires 3.11+)"
            )
            self.issues.append("Python version too old")

        # Ollama
        ollama_ok = self._check_ollama()

        console.print()
        return ollama_ok and not self.issues

    def _check_ollama(self) -> bool:
        """Check Ollama installation and model availability."""
        # Check if Ollama is installed
        ollama_path = shutil.which("ollama")

        if not ollama_path:
            console.print("  [yellow]![/yellow] Ollama not found")
            return self._offer_ollama_install()

        console.print(f"  [green]✓[/green] Ollama installed ({ollama_path})")

        # Check if Ollama is running
        try:
            response = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            models = [m["name"] for m in response.json().get("models", [])]

            # Check for embedding model
            has_model = EMBEDDING_MODEL in models or any(EMBEDDING_MODEL in m for m in models)

            if has_model:
                console.print(f"  [green]✓[/green] {EMBEDDING_MODEL} model available")
                return True
            else:
                console.print(f"  [yellow]![/yellow] {EMBEDDING_MODEL} model not found")
                return self._offer_model_pull()

        except httpx.ConnectError:
            console.print("  [yellow]![/yellow] Ollama not running")
            console.print("    [dim]Start with: ollama serve[/dim]")

            if Confirm.ask("    Would you like to start Ollama now?", default=True):
                return self._start_ollama()
            return False

        except Exception as e:
            console.print(f"  [red]✗[/red] Ollama error: {e}")
            return False

    def _offer_ollama_install(self) -> bool:
        """Offer to install Ollama."""
        console.print()

        if self.system == "darwin":
            install_cmd = "brew install ollama"
            console.print("    [dim]Install with Homebrew: brew install ollama[/dim]")
        elif self.system == "linux":
            install_cmd = "curl -fsSL https://ollama.com/install.sh | sh"
            console.print("    [dim]Install: curl -fsSL https://ollama.com/install.sh | sh[/dim]")
        else:
            console.print("    [dim]Download from: https://ollama.com/download[/dim]")
            return False

        if not Confirm.ask("    Would you like to install Ollama now?", default=True):
            self.issues.append("Ollama not installed")
            return False

        console.print(f"    [dim]Running: {install_cmd}[/dim]")

        try:
            if self.system == "darwin":
                subprocess.run(["brew", "install", "ollama"], check=True)
            elif self.system == "linux":
                subprocess.run(
                    ["sh", "-c", "curl -fsSL https://ollama.com/install.sh | sh"], check=True
                )

            console.print("  [green]✓[/green] Ollama installed")

            # Start Ollama
            return self._start_ollama()

        except subprocess.CalledProcessError as e:
            console.print(f"  [red]✗[/red] Installation failed: {e}")
            self.issues.append("Ollama installation failed")
            return False

    def _start_ollama(self) -> bool:
        """Start Ollama service."""
        console.print("    [dim]Starting Ollama...[/dim]")

        try:
            # Start Ollama in background
            if self.system == "darwin":
                subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
            else:
                subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    preexec_fn=os.setpgrp,
                )

            # Wait for it to be ready
            import time

            for _ in range(10):
                time.sleep(1)
                try:
                    httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
                    console.print("  [green]✓[/green] Ollama started")
                    return self._offer_model_pull()
                except Exception:
                    pass

            console.print("  [yellow]![/yellow] Ollama started but not responding yet")
            console.print("    [dim]Try running 'ollama serve' in another terminal[/dim]")
            return False

        except Exception as e:
            console.print(f"  [red]✗[/red] Failed to start Ollama: {e}")
            return False

    def _offer_model_pull(self) -> bool:
        """Offer to pull the embedding model."""
        if not Confirm.ask(f"    Would you like to download {EMBEDDING_MODEL}?", default=True):
            self.issues.append(f"{EMBEDDING_MODEL} model not available")
            return False

        console.print(f"    [dim]Pulling {EMBEDDING_MODEL} (this may take a moment)...[/dim]")

        try:
            result = subprocess.run(
                ["ollama", "pull", EMBEDDING_MODEL],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                console.print(f"  [green]✓[/green] {EMBEDDING_MODEL} downloaded")
                return True
            else:
                console.print(f"  [red]✗[/red] Download failed: {result.stderr}")
                return False

        except Exception as e:
            console.print(f"  [red]✗[/red] Download failed: {e}")
            return False

    def _configure_library(self) -> bool:
        """Configure the library path and create config file."""
        console.print("[bold]Configure your library:[/bold]\n")

        # Default library path
        default_library = Path.home() / "Documents" / "CoLibri"
        if not default_library.exists():
            # Try common alternatives
            for alt in ["Documents/Notes", "Notes", "Library"]:
                alt_path = Path.home() / alt
                if alt_path.exists():
                    default_library = alt_path
                    break

        # Prompt for library path
        library_path_str = Prompt.ask(
            "  Library/notes path",
            default=str(default_library),
        )
        library_path = Path(library_path_str).expanduser().resolve()

        # Validate or create
        if not library_path.exists():
            if Confirm.ask(f"  Path doesn't exist. Create {library_path}?", default=True):
                library_path.mkdir(parents=True, exist_ok=True)
                console.print(f"  [green]✓[/green] Created {library_path}")
            else:
                console.print("  [yellow]![/yellow] Library path not configured")
                return False

        # Books folder
        books_folder = Prompt.ask("  Books folder name", default="Books")
        books_path = library_path / books_folder
        if not books_path.exists():
            books_path.mkdir(parents=True, exist_ok=True)
            console.print(f"  [green]✓[/green] Created {books_path}")

        # Create config file
        return self._write_config(library_path, books_folder)

    def _write_config(self, library_path: Path, books_folder: str) -> bool:
        """Write the configuration file."""
        import yaml

        config = {
            "sources": [
                {
                    "path": str(library_path / books_folder),
                    "mode": "static",
                    "doc_type": "book",
                },
            ],
            "index": {
                "directory": "lancedb",
            },
            "ollama": {
                "base_url": OLLAMA_BASE_URL,
                "embedding_model": EMBEDDING_MODEL,
            },
            "retrieval": {
                "top_k": 10,
                "similarity_threshold": 0.3,
            },
            "chunking": {
                "chunk_size": 3000,
                "chunk_overlap": 200,
            },
        }

        CONFIG_DIR.mkdir(parents=True, exist_ok=True)

        try:
            with open(CONFIG_FILE, "w") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

            console.print(f"  [green]✓[/green] Config saved to {CONFIG_FILE}")
            return True

        except Exception as e:
            console.print(f"  [red]✗[/red] Failed to write config: {e}")
            return False

    def _setup_mcp_integration(self) -> None:
        """Setup MCP integration for Claude Code."""
        console.print()
        console.print("[bold]Claude Code integration:[/bold]\n")

        mcp_config_path = Path.home() / ".mcp.json"

        if not Confirm.ask("  Would you like to enable Claude Code integration?", default=True):
            console.print("  [dim]Skipped MCP configuration[/dim]")
            return

        # Find the Python executable in the venv or current interpreter
        python_path = sys.executable

        # MCP server configuration
        colibri_config = {
            "command": python_path,
            "args": ["-m", "colibri.mcp_server"],
        }

        # Load existing config or create new
        if mcp_config_path.exists():
            try:
                with open(mcp_config_path) as f:
                    mcp_config = json.load(f)
            except Exception:
                mcp_config = {"mcpServers": {}}
        else:
            mcp_config = {"mcpServers": {}}

        # Ensure mcpServers key exists
        if "mcpServers" not in mcp_config:
            mcp_config["mcpServers"] = {}

        # Check if already configured
        if "colibri" in mcp_config["mcpServers"] and not Confirm.ask(
            "  CoLibri already configured in MCP. Overwrite?", default=False
        ):
            console.print("  [dim]Kept existing MCP configuration[/dim]")
            return

        # Write config
        mcp_config["mcpServers"]["colibri"] = colibri_config

        try:
            with open(mcp_config_path, "w") as f:
                json.dump(mcp_config, f, indent=2)

            console.print(f"  [green]✓[/green] MCP config written to {mcp_config_path}")
            console.print("  [dim]Restart Claude Code to activate[/dim]")

        except Exception as e:
            console.print(f"  [red]✗[/red] Failed to write MCP config: {e}")

    def _show_summary(self) -> None:
        """Show setup summary and next steps."""
        console.print()

        if self.issues:
            console.print(
                Panel(
                    "[yellow]Setup completed with warnings:[/yellow]\n"
                    + "\n".join(f"  • {issue}" for issue in self.issues),
                    title="Summary",
                    border_style="yellow",
                )
            )
        else:
            console.print(
                Panel(
                    "[green]Setup completed successfully![/green]",
                    title="Summary",
                    border_style="green",
                )
            )

        console.print()
        console.print("[bold]Next steps:[/bold]")
        console.print("  1. Import a book:    [cyan]colibri import ~/Downloads/book.pdf[/cyan]")
        console.print("  2. Build the index:  [cyan]colibri index[/cyan]")
        console.print('  3. Search your books: [cyan]colibri search "your query"[/cyan]')
        console.print()
        console.print("[dim]For Claude integration, restart Claude Code after setup.[/dim]")


def run_setup() -> bool:
    """Run the setup wizard."""
    wizard = SetupWizard()
    return wizard.run()


def check_health() -> dict:
    """Check system health and return status.

    Returns:
        Dict with status of each component
    """
    status = {
        "python": {
            "ok": sys.version_info >= (3, 11),
            "version": f"{sys.version_info.major}.{sys.version_info.minor}",
        },
        "ollama": {
            "installed": shutil.which("ollama") is not None,
            "running": False,
            "model_available": False,
        },
        "config": {
            "exists": CONFIG_FILE.exists(),
            "path": str(CONFIG_FILE),
        },
    }

    # Check Ollama status
    if status["ollama"]["installed"]:
        try:
            response = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            status["ollama"]["running"] = True

            models = [m["name"] for m in response.json().get("models", [])]
            status["ollama"]["model_available"] = EMBEDDING_MODEL in models or any(
                EMBEDDING_MODEL in m for m in models
            )
        except Exception:
            pass

    return status
